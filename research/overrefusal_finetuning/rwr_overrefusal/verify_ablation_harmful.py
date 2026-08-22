#!/usr/bin/env python3
"""Re-run the single-direction ablations SAVING ALL RESPONSE TEXT, so the harmful-refusal
column can be judged rather than regex-matched.

Why this exists. search_selective_dirs.py:146 saved response text only for the baseline on
the harmful set (`if cname in ("baseline",) or ename == "or_rewrites"`), so every ablated
harmful number in our tables is a start-anchored regex rate that cannot be audited after the
fact. That matters because the regex reads a moralising refusal as compliance: on this model
the Arditi direction at L57 scores 3.1% by regex and 93.8% by judge on the same responses.
Qwen's d1 sits at cos +0.217 with the unembedding refusal-minus-compliance direction --
higher than the L60 direction we already retracted as a phrasing artifact -- and we reported
its harmful refusal falling 96.0% -> 0.5%. That number is unverified and is the sole basis
for the paper's two-model claim, so it gets re-measured with the text kept.

Everything is asserted rather than assumed: directions finite and unit-norm, layer in range,
eval sets non-empty and disjoint from training, the ablation hook demonstrably changing the
output, and the output path writable BEFORE any generation runs.
"""
import argparse, csv, json, os, random, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from killswitch_ablate_r17 import is_refusal, degenerate, read_col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--dirs_npz", required=True)
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--advbench", default="probe_or/data/advbench_harmful_behaviors.csv")
    ap.add_argument("--n_eval", type=int, default=400)
    ap.add_argument("--n_harmful", type=int, default=200)
    ap.add_argument("--n_random", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=128,
                    help="long enough for a judge to see whether a refusal is being argued")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    # ---- pre-flight: fail in seconds, not after an hour of generation -------------------
    os.makedirs(a.out_dir, exist_ok=True)
    probe = os.path.join(a.out_dir, ".writeprobe")
    open(probe, "w").close(); os.remove(probe)
    for p in (a.dirs_npz, a.atlas, a.advbench):
        assert os.path.exists(p), f"[preflight] missing {p}"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    Z = np.load(a.dirs_npz, allow_pickle=True)
    dirs = Z["dirs"].astype(np.float64)
    L = int(Z["layer"])
    assert np.isfinite(dirs).all(), "[preflight] non-finite direction"
    nrm = np.linalg.norm(dirs, axis=1)
    assert (nrm > 1e-6).all(), f"[preflight] zero-norm direction at {np.where(nrm<=1e-6)[0]}"
    dirs = dirs / nrm[:, None]                      # ablation assumes unit rows
    atlas_all = np.load(a.atlas, allow_pickle=True)["d"].astype(np.float64)
    assert L < atlas_all.shape[0], f"[preflight] layer {L} >= atlas layers {atlas_all.shape[0]}"
    r_atlas = atlas_all[L]
    assert np.isfinite(r_atlas).all() and np.linalg.norm(r_atlas) > 1e-6
    r_atlas = r_atlas / np.linalg.norm(r_atlas)
    print(f"[preflight] {len(dirs)} directions @L{L}, all finite and unit-normalised", flush=True)

    rnd = random.Random(a.seed)
    ho_o = [str(x) for x in Z["heldout_originals"]]
    ho_w = [str(x) for x in Z["heldout_rewrites"]]
    seen, pairs = set(), []
    for o, w in zip(ho_o, ho_w):
        if o not in seen:
            seen.add(o); pairs.append((o, w))
    rnd.shuffle(pairs); pairs = pairs[: a.n_eval]
    evalsets = {"or_rewrites": [w for _, w in pairs],
                "advbench": read_col(a.advbench, "goal", a.n_harmful)}
    for k, v in evalsets.items():
        assert len(v) > 0, f"[preflight] empty eval set {k}"
    train = set(str(x) for x in Z["train_originals"]) if "train_originals" in Z.files else set()
    leak = sum(1 for o, _ in pairs if o in train)
    assert leak == 0, f"[preflight] {leak} eval originals are in the training split"
    print(f"[preflight] {len(evalsets['or_rewrites'])} held-out OR (0 train leakage) | "
          f"{len(evalsets['advbench'])} AdvBench", flush=True)

    tok = AutoTokenizer.from_pretrained(a.base_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base_model, device_map="auto",
                                                 torch_dtype=torch.bfloat16).eval()
    dev, nL = model.device, model.config.num_hidden_layers
    layers_mod, embed = model.model.layers, model.model.embed_tokens
    is_qwen = "qwen" in a.base_model.lower()
    assert L <= nL, f"[preflight] direction layer {L} > model layers {nL}"

    def fmt(p):
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": p}]
        kw = {"enable_thinking": False} if is_qwen else {}
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)

    def ablate_fn(M):
        T = torch.tensor(M, dtype=torch.bfloat16, device=dev)      # [k, H], orthonormal rows
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h - (h.to(T.dtype) @ T.T) @ T
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return fn

    def all_layer(M):
        M = np.atleast_2d(M)
        return [(embed, ablate_fn(M))] + [(layers_mod[i], ablate_fn(M)) for i in range(nL)]

    def generate(prompts, specs):
        handles = [m.register_forward_hook(fn) for m, fn in specs]
        out = []
        try:
            for i in range(0, len(prompts), a.batch_size):
                b = prompts[i:i + a.batch_size]
                enc = tok([fmt(p) for p in b], return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(dev)
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                       pad_token_id=tok.pad_token_id)
                for j in range(len(b)):
                    out.append(tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        finally:
            for h in handles:
                h.remove()
        return out

    # ---- in-run check: the hook must actually change the output -------------------------
    smoke = evalsets["advbench"][:4]
    s_base = generate(smoke, [])
    s_abl = generate(smoke, all_layer(dirs[0]))
    changed = sum(1 for x, y in zip(s_base, s_abl) if x != y)
    assert changed > 0, "[check] ablation changed NOTHING -- the hook is a silent no-op"
    print(f"[check] ablation hook fires: {changed}/{len(smoke)} smoke responses changed", flush=True)

    names = ([str(x) for x in Z["labels"]] if "labels" in Z.files else
            [str(x) for x in Z["names"]] if "names" in Z.files else
            [f"d{i+1}" for i in range(len(dirs))])
    conds = [("baseline", None)] + [(names[i], dirs[i]) for i in range(len(dirs))]
    rng = np.random.default_rng(a.seed)
    for r in range(a.n_random):
        G = rng.standard_normal((dirs.shape[1], 1)); Q, _ = np.linalg.qr(G)
        conds.append((f"random_{r+1}", Q.T[0]))
    conds.append(("atlas_rhat", r_atlas))

    R, rows = {}, []
    for cname, M in conds:
        specs = [] if M is None else all_layer(M)
        for ename, prompts in evalsets.items():
            t0 = time.time()
            reps = generate(prompts, specs)
            R[f"{ename}__{cname}"] = 100.0 * sum(is_refusal(x) for x in reps) / len(reps)
            R[f"{ename}__{cname}__degen"] = 100.0 * sum(degenerate(x) for x in reps) / len(reps)
            for p, t in zip(prompts, reps):                 # SAVE EVERYTHING, both eval sets
                rows.append({"condition": cname, "evalset": ename, "prompt": p, "response": t,
                             "regex_refused": int(bool(is_refusal(t))), "degenerate": int(degenerate(t))})
            print(f"  {cname:18s} {ename:12s} regex-refusal {R[f'{ename}__{cname}']:6.2f}%  "
                  f"degen {R[f'{ename}__{cname}__degen']:5.2f}%  ({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(a.out_dir, "generations.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump({"rates_regex": R, "layer": L, "model": a.base_model,
               "conditions": [c for c, _ in conds], "n_rows": len(rows),
               "max_new_tokens": a.max_new_tokens},
              open(os.path.join(a.out_dir, "meta.json"), "w"), indent=1)
    print(f"[done] {len(rows)} responses -> {a.out_dir}/generations.jsonl", flush=True)


if __name__ == "__main__":
    main()
