#!/usr/bin/env python3
"""Build the Qwen behavioral-set pairs from the harm-filtered Sonnet benign pool.

Ordering matters: high-signal rewrites FIRST (sorted by Llama P("I cannot") desc),
then a random low-signal tail. So `gen_qwen_refusal.py --limit 200` calibrates on
the most-likely-refused rewrites, while the full set adds clear non-refusal
negatives for the combiner. Row order is fixed (idx-aligned across pair-acts,
generation, and the analysis).

Columns match build_pair_eval.py so extract_pair_acts.py / gen_qwen_refusal.py
read them unchanged: original, rewrite, p_orig, p_rw, dP, similarity.
(p_* here are the *Llama* labels — used only for selection/ordering; the Qwen
behavioral dP is measured separately by gen_qwen_refusal.py.)
"""
import argparse
import csv
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_csv", required=True, help="icannot_or_pairs_benign.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_high", type=int, default=1000, help="top-N by Llama P('I cannot')")
    ap.add_argument("--n_rand", type=int, default=500, help="random low-signal tail (negatives)")
    ap.add_argument("--sim_floor", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = []
    for r in csv.DictReader(open(args.pool_csv)):
        if float(r["similarity"]) < args.sim_floor:
            continue
        rows.append({"original": r["original"], "rewrite": r["rewrite"],
                     "p_orig": float(r["p_icannot_orig"]), "p_rw": float(r["p_icannot_rewrite"]),
                     "similarity": float(r["similarity"])})
    # dedup identical (original, rewrite)
    seen, uniq = set(), []
    for r in rows:
        k = (r["original"], r["rewrite"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)

    uniq.sort(key=lambda r: r["p_rw"], reverse=True)
    high = uniq[:args.n_high]                                   # high-signal (front)
    rest = uniq[args.n_high:]
    rng.shuffle(rest)
    tail = rest[:args.n_rand]                                   # random low-signal negatives
    keep = high + tail                                          # order preserved: high first

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["original", "rewrite", "p_orig", "p_rw", "dP", "similarity"])
        for r in keep:
            w.writerow([r["original"], r["rewrite"], f"{r['p_orig']:.6g}", f"{r['p_rw']:.6g}",
                        f"{r['p_rw'] - r['p_orig']:.6g}", f"{r['similarity']:.4f}"])

    print(f"[pairs] pool(uniq,sim>={args.sim_floor})={len(uniq)} -> kept={len(keep)} "
          f"(high={len(high)}, rand_tail={len(tail)})")
    print(f"[pairs] high band Llama p_rw: [{high[-1]['p_rw']:.4f}, {high[0]['p_rw']:.4f}] "
          f"| calibration prefix (first 200) all from high band")
    print(f"[pairs] wrote {args.out}")


if __name__ == "__main__":
    main()
