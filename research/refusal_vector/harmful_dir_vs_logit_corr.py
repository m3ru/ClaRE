#!/usr/bin/env python3
"""Correlate the ARDITI-faithful refusal direction (harmful-vs-harmless diff-of-means)
against the behavioral 'I cannot' logit signal (dP), per layer, over the Claude/Sonnet
rewrites. Compares to the old JAILBREAK-derived vector (which gave Pearson ~0.21 @L32,
~0.45 @L17). Question: does a properly-built refusal direction also predict benign
over-refusal (dP) better?

For each layer L:
  d[L]  = mean_harmful - mean_harmless  (AdvBench vs Alpaca, post-instruction token)
  Dproj = proj_L(rewrite) - proj_L(original)   over the pairs
  corr(Dproj, dP)  (Pearson + Spearman, full + high-sim>=0.85)
"""
import argparse
import csv
import json
import os
import time

import numpy as np


def read_col(path, col, n):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        c = col if col in r.fieldnames else r.fieldnames[0]
        rows = [(_r.get(c) or "").strip() for _r in r]
    return [v for v in rows if v and v.lower() != col][:n]


def read_pairs(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    sim = float(row["similarity"]); dP = float(row["dP"])
                except (KeyError, ValueError):
                    continue
                o = (row.get("original") or "").strip()
                rw = (row.get("rewrite") or "").strip()
                if o and rw:
                    rows.append((o, rw, sim, dP))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harmful_csv", required=True)
    ap.add_argument("--pairs_csv", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--n_dir", type=int, default=256)
    ap.add_argument("--sim_hi", type=float, default=0.85)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from scipy.stats import spearmanr
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tok = AutoTokenizer.from_pretrained(args.base_model, token=hf)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, token=hf, device_map="auto",
                                                 torch_dtype=torch.bfloat16)
    model.eval()
    dev = model.device
    H = model.config.hidden_size
    nL = model.config.num_hidden_layers

    def fmt(p):
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": p}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def last_all_layers(texts):
        for i in range(0, len(texts), args.batch_size):
            b = texts[i:i + args.batch_size]
            enc = tok([fmt(p) for p in b], return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to(dev)
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            yield torch.stack([h[:, -1, :] for h in hs], dim=1).float().cpu().numpy()

    # harmful-vs-harmless directions
    harmful = read_col(args.harmful_csv, "goal", args.n_dir)
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    harmless = []
    for ex in ds:
        inst = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        pp = f"{inst}\n\n{inp}" if inp else inst
        if inst and len(pp) < 1500:
            harmless.append(pp)
        if len(harmless) >= args.n_dir:
            break
    print(f"[dir] harmful {len(harmful)} vs harmless {len(harmless)}", flush=True)

    def cmean(texts):
        sm = np.zeros((nL + 1, H), np.float64)
        c = 0
        for arr in last_all_layers(texts):
            sm += arr.sum(0)
            c += arr.shape[0]
        return sm / c
    d = cmean(harmful) - cmean(harmless)                     # [nL+1, H]
    dn = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    d_unit = d / dn

    # project pairs
    pairs = read_pairs(args.pairs_csv)
    print(f"[proj] {len(pairs)} pairs", flush=True)
    uniq = list(dict.fromkeys([o for o, _, _, _ in pairs] + [r for _, r, _, _ in pairs]))
    proj = {}
    t0 = time.time()
    for k, i in enumerate(range(0, len(uniq), args.batch_size)):
        b = uniq[i:i + args.batch_size]
        enc = tok([fmt(p) for p in b], return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(dev)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        arr = torch.stack([h[:, -1, :] for h in hs], dim=1).float().cpu().numpy()  # [B,nL+1,H]
        pr = np.einsum("blh,lh->bl", arr, d_unit)             # [B, nL+1]
        for j, txt in enumerate(b):
            proj[txt] = pr[j]
        if k % 60 == 0:
            print(f"  [proj] {i+len(b)}/{len(uniq)} ({time.time()-t0:.0f}s)", flush=True)

    dP = np.array([p[3] for p in pairs])
    sim = np.array([p[2] for p in pairs])
    hi = sim >= args.sim_hi
    layers = []
    for L in range(1, nL + 1):
        rd = np.array([proj[r][L] - proj[o][L] for o, r, _, _ in pairs])
        pe = float(np.corrcoef(rd, dP)[0, 1])
        sp = float(spearmanr(rd, dP).statistic)
        pe_hi = float(np.corrcoef(rd[hi], dP[hi])[0, 1]) if hi.sum() > 2 else float("nan")
        layers.append({"layer": L, "pearson": pe, "spearman": sp, "pearson_hi": pe_hi})
    best = max(layers, key=lambda x: abs(x["pearson"]))
    summary = {"n_pairs": len(pairs), "n_high_sim": int(hi.sum()), "direction": "harmful_vs_harmless",
               "layers": layers, "best_layer": best,
               "note": "compare to jailbreak-derived vector: pearson ~0.21@L32, ~0.45@L17"}
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== corr(harmful-vs-harmless dir delta, dP) per layer ===")
    print(f"{'L':>3} {'pearson':>9} {'spearman':>9} {'pear_hi':>9}")
    for x in layers:
        mark = "  <-best" if x is best else ("  <-L12" if x["layer"] == 12 else ("  <-L17" if x["layer"] == 17 else ("  <-L32" if x["layer"] == 32 else "")))
        print(f"{x['layer']:>3} {x['pearson']:>9.3f} {x['spearman']:>9.3f} {x['pearson_hi']:>9.3f}{mark}")
    print(f"[done] best L{best['layer']} pearson {best['pearson']:.3f}; wrote {args.out}")


if __name__ == "__main__":
    main()
