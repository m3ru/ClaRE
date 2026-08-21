#!/usr/bin/env python3
"""Phase C: do OUR directions suppress over-refusals produced by a DIFFERENT attack method?

The single biggest limitation of everything else here is that the corpus comes from our own
RWR attacker, so the directions could encode that attacker's habits rather than anything
about the model. The GCG corpus is the test: it was built by a different method, on different
originals, and the lexical comparison already showed the two attacks are nearly disjoint --
`exploit` appears in 453 of our rewrites and 1 of 1,220 GCG ones.

So if directions fitted entirely on RWR pairs also suppress GCG-induced over-refusal, they are
a property of the model, not of our attacker. That is the generalisation claim.

Controls: GCG ORIGINALS (which the model complies with) must not start refusing; AdvBench must
stay refused; a rank-matched random direction must do nothing; degeneracy measured everywhere.

Caveat recorded in the output: the GCG filter verifies the original was complied with but does
NOT enforce intent preservation, so it admits rewrites our rubric would exclude (e.g. "reduce
plastic use" -> "increase plastic use"). This measures refusal, not over-refusal by our rubric.

Run: python gcg_transfer.py --out probe_or/results/gcg_transfer.json
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
    ap.add_argument("--gcg", default="incoming/sonnet_filtered_strict.json")
    ap.add_argument("--advbench", default="probe_or/data/advbench_harmful_behaviors.csv")
    ap.add_argument("--n_gcg", type=int, default=400)
    ap.add_argument("--n_harmful", type=int, default=150)
    ap.add_argument("--kmax", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="probe_or/results/gcg_transfer.json")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rnd = random.Random(a.seed)

    Z = np.load(a.dirs_npz, allow_pickle=True)
    dirs = Z["dirs"].astype(np.float64); L = int(Z["layer"])

    raw = json.load(open(a.gcg))
    rowsg = raw["rows"] if isinstance(raw, dict) and "rows" in raw else raw
    seen, gp = set(), []
    for r in rowsg:
        o, w = str(r.get("original", "")).strip(), str(r.get("rewrite", "")).strip()
        if o and w and o not in seen:
            seen.add(o); gp.append((o, w))
    rnd.shuffle(gp); gp = gp[: a.n_gcg]
    evalsets = {"gcg_rewrites": [w for _, w in gp],
                "gcg_originals": [o for o, _ in gp],
                "advbench": read_col(a.advbench, "goal", a.n_harmful)}
    print(f"[gcg] {len(gp)} GCG pairs (distinct originals) | directions from RWR only", flush=True)

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
    K = min(a.kmax, len(dirs))
    conds = [("baseline", None)] + [(f"ours_k{k}", dirs[:k]) for k in range(1, K + 1)]
    G = rng.standard_normal((dirs.shape[1], K)); Q, _ = np.linalg.qr(G)
    conds.append((f"random_k{K}", Q.T[:K]))
    r_atlas = np.load(a.atlas, allow_pickle=True)["d"].astype(np.float64)[L]
    conds.append(("atlas_rhat", r_atlas / np.linalg.norm(r_atlas)))

    R, S = {}, {}
    for cname, M in conds:
        specs = [] if M is None else all_layer(M)
        for ename, prompts in evalsets.items():
            t0 = time.time()
            reps = generate(prompts, specs)
            R[f"{ename}__{cname}"] = 100.0 * sum(is_refusal(x) for x in reps) / max(len(reps), 1)
            R[f"{ename}__{cname}__degen"] = 100.0 * sum(degenerate(x) for x in reps) / max(len(reps), 1)
            S[f"{ename}__{cname}"] = reps[:3]
            print(f"  {cname:12s} {ename:14s} refusal {R[f'{ename}__{cname}']:6.2f}%  "
                  f"degen {R[f'{ename}__{cname}__degen']:5.2f}%  ({time.time()-t0:.0f}s)", flush=True)

    R["_meta"] = dict(layer=L, n_gcg=len(gp), directions_from="RWR pairs only",
                      caveat="GCG filter does not enforce intent preservation")
    json.dump({"rates": R, "samples": S}, open(a.out, "w"), indent=1)

    b = R["gcg_rewrites__baseline"]
    print("\n=== CROSS-ATTACKER TRANSFER ===")
    print(f"GCG refusal baseline {b:.2f}%  (originals {R['gcg_originals__baseline']:.2f}%)")
    for k in range(1, K + 1):
        v = R[f"gcg_rewrites__ours_k{k}"]
        print(f"  ablate OUR k={k}: {v:6.2f}%  ({v-b:+.1f}pp)  | GCG originals "
              f"{R[f'gcg_originals__ours_k{k}']:5.2f}%  AdvBench {R[f'advbench__ours_k{k}']:5.2f}%")
    print(f"  random k={K}      : {R[f'gcg_rewrites__random_k{K}']:6.2f}%  "
          f"({R[f'gcg_rewrites__random_k{K}']-b:+.1f}pp)")
    print(f"  atlas r-hat       : {R['gcg_rewrites__atlas_rhat']:6.2f}%  "
          f"({R['gcg_rewrites__atlas_rhat']-b:+.1f}pp)  AdvBench "
          f"{R['advbench__atlas_rhat']:.2f}%")
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()
