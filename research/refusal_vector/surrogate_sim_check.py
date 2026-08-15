#!/usr/bin/env python3
"""Is a Llama-space similarity surrogate a valid stand-in for MiniLM cosine?

PEZ cannot optimize MiniLM cosine: different tokenizer, no gradient path back into Llama
token space. The proposed fix is a surrogate computed INSIDE Llama, which is differentiable
end-to-end and therefore usable in PEZ's gradient (unlike GCG, where similarity only reranks
proposals). That only works if the surrogate tracks MiniLM -- otherwise PEZ satisfies the
surrogate while real similarity collapses, and the constraint stops constraining.

v1 tested only raw mean-pooling and it FAILED: cosine ~0.99 with std ~0.001-0.013 at every
mid layer, i.e. a non-constraint. That is textbook anisotropy -- raw LM states sit in a
narrow cone dominated by a shared component (format/position/register), with semantics a
thin perturbation on top.

v2 therefore sweeps the two fixes that address exactly that:
  pooling   : mean over prompt tokens  vs  LAST token (what our refusal direction uses, and
              the only position a causal model lets see the whole prompt)
  post-proc : raw | centered (subtract corpus mean) | centered + drop top-3 PCs
              ("all-but-the-top", Mu & Viswanath 2018 -- removes the dominant directions
              that make everything look identical)
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--layers", default="4,8,12,16,17,20,24,28,32")
    ap.add_argument("--n_pc", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    import torch
    from scipy.stats import pearsonr, spearmanr
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tok = AutoTokenizer.from_pretrained(args.base_model, token=hf)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, token=hf, device_map="auto",
                                                 torch_dtype=torch.bfloat16)
    model.eval()
    dev = model.device
    layers = [int(x) for x in args.layers.split(",")]

    pairs = []
    for f in sorted(glob.glob(args.glob)):
        with open(f) as fh:
            D = json.load(fh)
        for r in D["results"]:
            if r.get("similarity") is not None:
                pairs.append((r["original"], r["rewrite"], float(r["similarity"])))
    print(f"[data] {len(pairs)} (original, rewrite, minilm_sim) triples", flush=True)

    def encode(texts):
        """-> mean-pooled [N, nL+1, H] and last-token [N, nL+1, H] (unnormalised)."""
        mp, lp = [], []
        for i in range(0, len(texts), args.batch_size):
            b = texts[i:i + args.batch_size]
            enc = tok(b, return_tensors="pt", padding=True, truncation=True,
                      max_length=256, add_special_tokens=True).to(dev)
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            am = enc["attention_mask"]
            m = am.unsqueeze(-1).float()
            mp.append(torch.stack([(h.float() * m).sum(1) / m.sum(1).clamp(min=1e-9) for h in hs], 1).cpu())
            last = am.sum(1) - 1
            idx = torch.arange(len(b), device=dev)
            lp.append(torch.stack([h.float()[idx, last] for h in hs], 1).cpu())
        return torch.cat(mp, 0).numpy(), torch.cat(lp, 0).numpy()

    o_mean, o_last = encode([p[0] for p in pairs])
    r_mean, r_last = encode([p[1] for p in pairs])
    mini = np.array([p[2] for p in pairs])

    def cos(a, b):
        a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
        b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
        return (a * b).sum(-1)

    def drop_pcs(stack, k):
        """stack: [M, H] centred. Remove the top-k principal directions."""
        if k <= 0:
            return stack
        _, _, vt = np.linalg.svd(stack, full_matrices=False)
        V = vt[:k]                                   # [k, H]
        return stack - (stack @ V.T) @ V

    rows = []
    for pool, (O, R) in [("mean", (o_mean, r_mean)), ("last", (o_last, r_last))]:
        for L in layers:
            a, b = O[:, L, :], R[:, L, :]
            for proc in ("raw", "center", f"center_drop{args.n_pc}"):
                if proc == "raw":
                    x, y = a, b
                else:
                    mu = np.concatenate([a, b], 0).mean(0, keepdims=True)
                    x, y = a - mu, b - mu
                    if proc.startswith("center_drop"):
                        M = drop_pcs(np.concatenate([x, y], 0), args.n_pc)
                        x, y = M[:len(x)], M[len(x):]
                s = cos(x, y)
                rows.append({"pool": pool, "layer": L, "proc": proc,
                             "pearson": float(pearsonr(s, mini)[0]),
                             "spearman": float(spearmanr(s, mini).statistic),
                             "mean": float(s.mean()), "std": float(s.std())})

    best = max(rows, key=lambda r: abs(r["spearman"]))
    with open(args.out, "w") as f:
        json.dump({"n": len(pairs), "rows": rows, "best": best,
                   "minilm_mean": float(mini.mean()), "minilm_std": float(mini.std())}, f, indent=2)

    print(f"\n{'pool':5s} {'L':>3} {'proc':16s} {'pearson':>8} {'spearman':>9} {'mean':>7} {'std':>7}")
    for r in sorted(rows, key=lambda r: -abs(r["spearman"]))[:24]:
        print(f"{r['pool']:5s} {r['layer']:>3} {r['proc']:16s} {r['pearson']:>8.3f} "
              f"{r['spearman']:>9.3f} {r['mean']:>7.3f} {r['std']:>7.3f}")
    print(f"\n[verdict] best: pool={best['pool']} L{best['layer']} {best['proc']} "
          f"spearman={best['spearman']:.3f} std={best['std']:.3f}")
    print("  need spearman >=0.8 AND std well above ~0.02 to be worth optimizing.")
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
