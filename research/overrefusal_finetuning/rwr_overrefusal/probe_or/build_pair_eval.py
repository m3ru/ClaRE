#!/usr/bin/env python3
"""Assemble the paired eval set for the DELTA probe experiment.

From the icannot-OR scored pools, build (original, rewrite) pairs with:
  p_orig, p_rw = P("I cannot") of each
  dP           = p_rw - p_orig   (the induced change in refusal propensity)
  similarity

The delta probe is scored as proj(rewrite) - proj(original), and we test whether
that ranks dP (induced refusal) better than the absolute probe ranked P (~0.29).

Selection keeps the whole signal-bearing tail (rare induced cases) + a random
sample of the near-zero-dP mass, capped so the paired extraction stays cheap.
Focus on benign originals (low p_orig): there dP reflects PHRASING, not a
pre-existing sensitive topic.
"""
import argparse
import csv
import os
import random


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append({
            "original": r["original"], "rewrite": r["rewrite"],
            "p_orig": float(r["p_icannot_orig"]), "p_rw": float(r["p_icannot_rewrite"]),
            "similarity": float(r["similarity"]),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sim_floor", type=float, default=0.5)
    ap.add_argument("--cap", type=int, default=6000, help="total pairs (each -> 2 forward passes)")
    ap.add_argument("--signal_p_rw", type=float, default=1e-3, help="keep all pairs with p_rw above this")
    ap.add_argument("--signal_dP", type=float, default=5e-3, help="keep all pairs with |dP| above this")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = []
    for path in args.csvs:
        rows += load(path)
    rows = [r for r in rows if r["similarity"] >= args.sim_floor]
    for r in rows:
        r["dP"] = r["p_rw"] - r["p_orig"]
    # dedup identical (original, rewrite)
    seen, uniq = set(), []
    for r in rows:
        k = (r["original"], r["rewrite"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    rows = uniq

    # index-based partition (O(n)); dict-membership `not in` would be O(n*|signal|)
    sig_idx = {i for i, r in enumerate(rows)
               if r["p_rw"] > args.signal_p_rw or abs(r["dP"]) > args.signal_dP}
    signal = [rows[i] for i in sig_idx]
    rest = [rows[i] for i in range(len(rows)) if i not in sig_idx]
    rng.shuffle(rest)
    keep = signal + rest[:max(0, args.cap - len(signal))]
    rng.shuffle(keep)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["original", "rewrite", "p_orig", "p_rw", "dP", "similarity"])
        for r in keep:
            w.writerow([r["original"], r["rewrite"], f"{r['p_orig']:.6g}", f"{r['p_rw']:.6g}",
                        f"{r['dP']:.6g}", f"{r['similarity']:.4f}"])

    pos = sum(1 for r in keep if r["dP"] > 0.01)
    print(f"[pairs] total={len(rows)} sim>={args.sim_floor}; kept={len(keep)} "
          f"(signal={len(signal)}, sampled={len(keep)-len(signal)})")
    print(f"[pairs] induced (dP>0.01): {pos} | dP p99={sorted(r['dP'] for r in keep)[int(len(keep)*0.99)]:.4f} "
          f"max={max(r['dP'] for r in keep):.4f}")
    print(f"[pairs] wrote {args.out}")


if __name__ == "__main__":
    main()
