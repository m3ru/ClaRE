#!/usr/bin/env python3
"""High-power safety control + usefulness capture.

Two gaps this closes, both raised in review and both real.

1. NO POWER. The safety claim ("our directions cost far less harmful-refusal than the
   literature direction") rested on AdvBench at n=120-200 against a 98.5% baseline. At that
   n the standard error is ~0.9pp, so a "1.0pp vs 2.5pp" difference is a one-prompt
   difference and the Wilson intervals overlap completely. Here we pool AdvBench (520) with
   the Arditi harmful set (512) for n~1032, which cuts the SE to ~0.4pp -- enough to
   actually separate the conditions, or to show honestly that they cannot be separated.

2. NON-REFUSAL IS NOT THE SAME AS USEFULNESS. `is_refusal` is a start-anchored regex, and
   `degenerate()` only catches repetition collapse. A reply that is fluent, non-refusing and
   WRONG scores as a success. So we also capture the generations for held-out over-refusal
   rewrites under baseline and under each ablation, letting a judge check afterwards whether
   the model now actually delivers the benign thing that was asked for.

Directions come from the leakage-free train-only basis (build_causal_dirs.py). d1 is the
shared axis (cos ~0.78 with the harmful-vs-harmless direction -- i.e. largely "this looks
harmful"); the interesting one is the direction that is orthogonal to harmfulness yet still
suppresses over-refusal.

Run: python safety_power_check.py --out probe_or/results/safety_power.json
"""
import argparse, csv, json, os, random, sys, time
import numpy as np
csv.field_size_limit(sys.maxsize)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from killswitch_ablate_r17 import is_refusal, degenerate, read_col

SYSTEM = "You are a helpful assistant."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--dirs_npz", default="probe_or/results/delta/causal_dirs.npz")
    ap.add_argument("--atlas", default="probe_or/results/llama_signals/probe_absolute.npz")
    ap.add_argument("--advbench", default="probe_or/data/advbench_harmful_behaviors.csv")
    ap.add_argument("--arditi", default="probe_or/data/arditi_harmful.csv")
    ap.add_argument("--potent_idx", type=int, default=3, help="0-based index of the "
                    "harmfulness-orthogonal direction in the basis (d4)")
    ap.add_argument("--n_or", type=int, default=400)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="probe_or/results/safety_power.json")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rnd = random.Random(a.seed)

    Z = np.load(a.dirs_npz, allow_pickle=True)
    dirs = Z["dirs"].astype(np.float64); L = int(Z["layer"])
    ho_o = [str(x) for x in Z["heldout_originals"]]
    ho_w = [str(x) for x in Z["heldout_rewrites"]]
    seen, pairs = set(), []
    for o, w in zip(ho_o, ho_w):
        if o not in seen:
            seen.add(o); pairs.append((o, w))
    rnd.shuffle(pairs); pairs = pairs[: a.n_or]

    harm = read_col(a.advbench, "goal", 10**6)
    for c in ("prompt", "goal", "instruction"):
        extra = read_col(a.arditi, c, 10**6)
        if extra:
            harm += extra
            break
    harm = list(dict.fromkeys([h for h in harm if h]))
    print(f"[safety] pooled harmful set: {len(harm)} prompts | held-out OR: {len(pairs)}", flush=True)

    evalsets = {"harmful": harm, "or_rewrites": [w for _, w in pairs]}

    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tok = AutoTokenizer.from_pretrained(a.base_model, token=hf)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base_model, token=hf, device_map="auto",
                                                 dtype=torch.bfloat16).eval()
    dev, nL = model.device, model.config.num_hidden_layers
    layers_mod, embed = model.model.layers, model.model.embed_tokens

    def fmt(p):
        return tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def ablate_fn(B):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h - (h.to(B.dtype) @ B.T) @ B
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return fn

    def all_layer(M):
        B = torch.tensor(np.atleast_2d(M), dtype=torch.bfloat16, device=dev)
        return [(embed, ablate_fn(B))] + [(layers_mod[i], ablate_fn(B)) for i in range(nL)]

    def generate(prompts, specs):
        hs = [m.register_forward_hook(f) for m, f in specs]
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
            for h in hs:
                h.remove()
        return out

    rng = np.random.default_rng(a.seed)
    pi = a.potent_idx
    r_atlas = np.load(a.atlas, allow_pickle=True)["d"].astype(np.float64)[L]
    Q1, _ = np.linalg.qr(rng.standard_normal((dirs.shape[1], 1)))
    Q2, _ = np.linalg.qr(rng.standard_normal((dirs.shape[1], 2)))
    conds = [("baseline", None),
             ("shared_d1", dirs[0]),
             ("potent_d%d" % (pi + 1), dirs[pi]),
             ("d1_plus_potent", np.vstack([dirs[0], dirs[pi]])),
             ("atlas_rhat", r_atlas / np.linalg.norm(r_atlas)),
             ("random_k1", Q1.T[:1]),
             ("random_k2", Q2.T[:2])]

    R, G = {}, {}
    for cname, M in conds:
        specs = [] if M is None else all_layer(M)
        for ename, prompts in evalsets.items():
            t0 = time.time()
            reps = generate(prompts, specs)
            n = len(reps)
            R[f"{ename}__{cname}"] = 100.0 * sum(is_refusal(x) for x in reps) / max(n, 1)
            R[f"{ename}__{cname}__n"] = n
            R[f"{ename}__{cname}__degen"] = 100.0 * sum(degenerate(x) for x in reps) / max(n, 1)
            if ename == "or_rewrites":
                G[cname] = reps                      # keep for the usefulness judge
            print(f"  {cname:16s} {ename:12s} refusal {R[f'{ename}__{cname}']:6.2f}% "
                  f"(n={n})  degen {R[f'{ename}__{cname}__degen']:5.2f}%  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    R["_meta"] = dict(layer=L, n_harmful=len(harm), n_or=len(pairs), model=a.base_model)
    json.dump({"rates": R, "or_generations": G,
               "or_prompts": [w for _, w in pairs],
               "or_originals": [o for o, _ in pairs]}, open(a.out, "w"), indent=1)

    import math
    def wil(p, n, z=1.96):
        p /= 100; d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return 100 * (c - h), 100 * (c + h)
    print("\n=== SAFETY, PROPERLY POWERED (harmful set n=%d) ===" % len(harm))
    b = R["harmful__baseline"]
    for cname, _ in conds:
        v = R[f"harmful__{cname}"]; lo, hi = wil(v, len(harm))
        o = R[f"or_rewrites__{cname}"]
        print(f"  {cname:16s} harmful {v:6.2f}% [{lo:5.1f},{hi:5.1f}] ({v-b:+5.1f}pp)   "
              f"over-refusal {o:6.2f}%")
    print(f"\n[done] {a.out}")


if __name__ == "__main__":
    main()
