#!/usr/bin/env python3
"""Build RWR training shards from icannot-OR-scored pools.

Consumes the per-pair CSVs written by score_icannot_or.py (Claude rewrites + orp3k),
pools them, applies a similarity floor and a per-source cap, and writes shard JSONs
in the format rwr_data.load_shards expects.

Reward wiring: we store `refusal_delta = P("I cannot" | rewrite)` and `similarity = sim`.
The training pipeline's recompute_or_scores then produces
    or_score_raw = exp(k*(sim - c)) * refusal_delta / d
so with k=18.4, c=0.75, d=1.0 the reward IS the icannot-OR score. Bin on it with
absolute thresholds (bin_edges) because the distribution is extremely skewed.

Per-source cap: keep at most --per_source_cap rewrites per unique original, the
highest-P("I cannot") ones, so a handful of refusal-prone sources (moonshine,
Wagner, Covenant, PII...) can't dominate the high-reward bins and get memorized.

Selection is high-OR regardless of the original's own refusal-proneness (P_orig),
per the research decision: we want rewrites with high OR, whether or not the
original was already refusal-adjacent.
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict


def load_csv(path, source_tag):
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append({
            "original": r["original"],
            "rewrite": r["rewrite"],
            "similarity": float(r["similarity"]),
            "p_rw": float(r["p_icannot_rewrite"]),
            "p_orig": float(r["p_icannot_orig"]),
            "source": source_tag,
        })
    return rows


def icannot_or(sim, p_rw, k, c, d):
    return math.exp(k * (sim - c)) * p_rw / d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True,
                    help="pairs CSVs (score_icannot_or.py output). Prefix with tag= to name the source, "
                         "e.g. claude=path/icannot_or_pairs.csv orp3k=path/icannot_or_pairs.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sim_floor", type=float, default=0.5)
    ap.add_argument("--per_source_cap", type=int, default=8,
                    help="Max rewrites kept per unique original (highest P('I cannot') first).")
    ap.add_argument("--shard_size", type=int, default=200, help="Originals per shard file.")
    # OR formula (must match training config: k=18.4, c=0.75, d=1.0)
    ap.add_argument("--k", type=float, default=18.4)
    ap.add_argument("--c", type=float, default=0.75)
    ap.add_argument("--d", type=float, default=1.0)
    # Reporting: bin edges to summarize population under (must match training bin_edges)
    ap.add_argument("--bin_edges", type=str, default="1e-4,1e-3,1e-2,1e-1")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load
    rows = []
    for spec in args.csvs:
        tag, path = spec.split("=", 1) if "=" in spec else (os.path.basename(os.path.dirname(spec)), spec)
        n0 = len(rows)
        rows += load_csv(path, tag)
        print(f"[load] {tag}: {len(rows) - n0} pairs from {path}")
    print(f"[load] total {len(rows)} pairs")

    # ---- similarity floor
    rows = [r for r in rows if r["similarity"] >= args.sim_floor]
    print(f"[filter] {len(rows)} pairs with sim >= {args.sim_floor}")

    # ---- group by original, dedup identical rewrites, per-source cap by P('I cannot')
    by_orig = defaultdict(list)
    for r in rows:
        by_orig[r["original"]].append(r)

    records = []
    kept = 0
    for orig, rs in by_orig.items():
        seen = set()
        uniq = []
        for r in sorted(rs, key=lambda x: -x["p_rw"]):
            key = " ".join(r["rewrite"].split())
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        uniq = uniq[:args.per_source_cap]
        paras = [{
            "paraphrase": r["rewrite"],
            "refusal_delta": r["p_rw"],          # <-- P("I cannot"|rewrite) drives the reward
            "similarity": r["similarity"],
            "or_score_raw": icannot_or(r["similarity"], r["p_rw"], args.k, args.c, args.d),
            "p_icannot_orig": r["p_orig"],
        } for r in uniq]
        records.append({"original": orig, "paraphrases": paras})
        kept += len(paras)
    print(f"[cap] {len(records)} unique originals, {kept} pairs after cap={args.per_source_cap}")

    # ---- write shards
    for i in range(0, len(records), args.shard_size):
        shard = records[i:i + args.shard_size]
        idx = i // args.shard_size
        path = os.path.join(args.out_dir, f"or_susceptibility_rankings_shard{idx}.json")
        json.dump(shard, open(path, "w"))
    n_shards = (len(records) + args.shard_size - 1) // args.shard_size
    print(f"[write] {n_shards} shards -> {args.out_dir}")

    # ---- bin-population report (so we can pick sane bin_weights)
    edges = [float(x) for x in args.bin_edges.split(",")]
    all_or = [p["or_score_raw"] for rec in records for p in rec["paraphrases"]]
    all_or.sort()

    def which_bin(x):
        b = 0
        for e in edges:
            if x >= e:
                b += 1
        return b

    pops = defaultdict(int)
    src_in_bin = defaultdict(set)
    for rec in records:
        for p in rec["paraphrases"]:
            b = which_bin(p["or_score_raw"])
            pops[b] += 1
            src_in_bin[b].add(rec["original"])
    nb = len(edges) + 1
    labels = [f"OR<{edges[0]:g}"] + [f"[{edges[i]:g},{edges[i+1]:g})" for i in range(len(edges) - 1)] + [f">={edges[-1]:g}"]
    print(f"\n[bins] absolute-threshold populations (edges={edges}):")
    print(f"  {'bin':>3} {'range':>14} {'pairs':>7} {'sources':>8}")
    for b in range(nb):
        print(f"  {b:>3} {labels[b]:>14} {pops[b]:>7} {len(src_in_bin[b]):>8}")
    print(f"\n[dist] OR p50={all_or[len(all_or)//2]:.2e} p90={all_or[int(len(all_or)*0.9)]:.2e} "
          f"p99={all_or[int(len(all_or)*0.99)]:.2e} max={all_or[-1]:.3f}")


if __name__ == "__main__":
    main()
