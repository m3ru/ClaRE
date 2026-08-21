#!/usr/bin/env python3
"""Selective frame ablation — the causal test of the frame hypothesis.

Correlational finding to be tested: over-refusal Δ decomposes into a dominant SHARED danger
axis plus small, mutually orthogonal FRAME-SPECIFIC residuals that reproduce across edit
sizes (cross-bin diagonal +0.42, off-diagonal -0.04).

If that decomposition is causal, two predictions follow and they are different:

  ablate SHARED      -> over-refusal falls for EVERY frame, roughly equally.
  ablate RESIDUAL_f  -> over-refusal falls MORE for frame f than for the other frames.

The second is the one that matters. A frame-specific direction that selectively suppresses
its own frame is interpretable causal control over one over-refusal reflex; if instead every
residual suppresses every frame equally, the residuals are noise and the shared axis is the
whole story (a clean negative).

Controls: a random direction at matched norm, AdvBench harmful refusal (a selective fix must
not cost safety), and a degeneracy check per cell -- an ablation that breaks the model scores
0% refusal on everything and would otherwise read as a perfect result.

Run: python ablate_frames.py --out probe_or/results/frame_ablation.json
"""
import argparse, csv, json, os, random, re, sys, time
import numpy as np
csv.field_size_limit(sys.maxsize)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_edit_distance import pair_metrics
from analyze_frames import frames_of
from killswitch_ablate_r17 import is_refusal, degenerate, read_col

SYSTEM = "You are a helpful assistant."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--frames_npz", default="probe_or/results/delta/frame_directions.npz")
    ap.add_argument("--sets", default="probe_or/results/delta/prompt_sets.csv")
    ap.add_argument("--advbench", default="probe_or/data/advbench_harmful_behaviors.csv")
    ap.add_argument("--n_per_frame", type=int, default=120)
    ap.add_argument("--n_harmful", type=int, default=120)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="probe_or/results/frame_ablation.json")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rnd = random.Random(a.seed)

    Z = np.load(a.frames_npz, allow_pickle=True)
    frames = [str(x) for x in Z["frames"]]
    shared = Z["shared"].astype(np.float64)
    u_low, u_high = Z["u_low"].astype(np.float64), Z["u_high"].astype(np.float64)

    def resid(u):
        r = u - (u @ shared) * shared
        return r / (np.linalg.norm(r) + 1e-9)

    # Residual direction per frame, averaged over the two INDEPENDENT estimates (LOW and HIGH)
    # so the ablated direction is not tied to either edit regime.
    res_dirs = {}
    for i, f in enumerate(frames):
        m = resid(u_low[i]) + resid(u_high[i])
        res_dirs[f] = m / (np.linalg.norm(m) + 1e-9)

    # ---- eval prompts: held-out OR rewrites grouped by frame, one per original ----
    rows = [r for r in csv.DictReader(open(a.sets)) if r["set"] == "or_high"]
    by_frame = {f: {} for f in frames}
    for r in rows:
        o, w = r["original"].strip(), r["rewrite"].strip()
        for f in frames_of(pair_metrics(o, w)["introduced_words"]):
            if f in by_frame and o not in by_frame[f]:
                by_frame[f][o] = w
    evalsets = {}
    for f in frames:
        ks = sorted(by_frame[f]); rnd.shuffle(ks)
        evalsets[f] = [by_frame[f][k] for k in ks[:a.n_per_frame]]
        print(f"[eval] {f}: {len(evalsets[f])} rewrites", flush=True)
    evalsets["advbench"] = read_col(a.advbench, "goal", a.n_harmful)

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

    def ablate_fn(dh):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h - (h.to(dh.dtype) @ dh).unsqueeze(-1) * dh
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return fn

    def all_layer(v):
        dh = torch.tensor(v, dtype=torch.bfloat16, device=dev)
        return [(embed, ablate_fn(dh))] + [(layers_mod[i], ablate_fn(dh)) for i in range(nL)]

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

    rng = np.random.default_rng(a.seed)
    rv = rng.standard_normal(shared.shape[0]); rv /= np.linalg.norm(rv)
    conditions = [("baseline", None), ("ablate_shared", shared)]
    conditions += [(f"ablate_resid_{f}", res_dirs[f]) for f in frames]
    conditions += [("ablate_random", rv)]

    R, S = {}, {}
    for cname, v in conditions:
        specs = [] if v is None else all_layer(v)
        for ename, prompts in evalsets.items():
            t0 = time.time()
            reps = generate(prompts, specs)
            rr = 100.0 * sum(is_refusal(x) for x in reps) / max(len(reps), 1)
            dg = 100.0 * sum(degenerate(x) for x in reps) / max(len(reps), 1)
            R[f"{ename}__{cname}"] = rr
            R[f"{ename}__{cname}__degen"] = dg
            S[f"{ename}__{cname}"] = reps[:3]
            print(f"  {cname:26s} {ename:14s} refusal {rr:6.2f}%  degen {dg:5.2f}%  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    R["_meta"] = dict(frames=frames, n_per_frame=a.n_per_frame, model=a.base_model,
                      greedy=True, max_new_tokens=a.max_new_tokens)
    json.dump({"rates": R, "samples": S}, open(a.out, "w"), indent=1)

    # ---- selectivity: does residual_f hit frame f harder than the other frames? ----
    print("\n=== SELECTIVITY (refusal drop from baseline, percentage points) ===")
    hdr = "ablated \\ measured   " + "".join(f"{f[:11]:>13s}" for f in frames)
    print(hdr)
    for f in frames:
        c = f"ablate_resid_{f}"
        cells = []
        for g in frames:
            drop = R[f"{g}__baseline"] - R[f"{g}__{c}"]
            cells.append(f"{drop:>12.1f}{'*' if f == g else ' '}")
        print(f"resid_{f[:12]:14s}" + "".join(cells))
    print("\n(* = own frame; the frame hypothesis predicts the starred value leads its row)")
    print(f"\nAdvBench: baseline {R['advbench__baseline']:.1f}%  shared "
          f"{R['advbench__ablate_shared']:.1f}%  " +
          "  ".join(f"{f[:6]} {R[f'advbench__ablate_resid_{f}']:.1f}%" for f in frames))
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()
