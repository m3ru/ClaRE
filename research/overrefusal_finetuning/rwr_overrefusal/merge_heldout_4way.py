#!/usr/bin/env python3
"""Merge held-out eval result JSONs into one N-way table (per corpus, per metric).

All inputs must be scored at the same k=5.0/c=0.75/d=100 scale on the SAME held-out
prompts (eval_seed=99) — true for the rescored k5 file and any fresh eval_held_out.py
run. Used to put llama_self_rwr next to baseline / claude_rwr / rwr_v3.

Usage:
    python merge_heldout_4way.py \
        --inputs prompt_iteration_results/held_out_eval/held_out_eval_results_k5.json \
                 prompt_iteration_results/held_out_eval_llama_self/held_out_eval_results.json
"""
import argparse
import json
import os

METRICS = ("or_score_raw", "refusal_delta", "similarity")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--corpora", nargs="+", default=["alpaca", "dolly"])
    args = ap.parse_args()

    # label -> corpus -> {n, stats}
    merged = {}
    for path in args.inputs:
        if not os.path.isfile(path):
            print(f"[warn] missing input: {path}")
            continue
        d = json.load(open(path))
        for label, by_corpus in d.get("results", {}).items():
            for corpus, c in by_corpus.items():
                merged.setdefault(label, {})[corpus] = {
                    "n": len(c["generations"]),
                    "stats": c["stats"],
                }

    for corpus in args.corpora:
        present = [(lbl, v[corpus]) for lbl, v in merged.items() if corpus in v]
        if not present:
            continue
        print(f"\n{'='*104}")
        print(f"HELD-OUT {corpus}  (k=5.0, c=0.75, d=100; same 200 prompts, seed=99)")
        print(f"{'='*104}")
        for metric in METRICS:
            print(f"\n  {metric}:")
            print(f"    {'model':24s} {'n':>5s} {'mean':>9s} {'median':>9s} "
                  f"{'p75':>9s} {'p90':>9s} {'p95':>9s} {'%pos':>8s}")
            rows = sorted(present, key=lambda r: -r[1]["stats"][metric]["p90"])
            for lbl, c in rows:
                s = c["stats"][metric]
                print(f"    {lbl:24s} {c['n']:5d} {s['mean']:9.4f} {s['median']:9.4f} "
                      f"{s['p75']:9.4f} {s['p90']:9.4f} {s['p95']:9.4f} {s['pct_positive']:7.1%}")

        # Headline: self-distillation vs base policy on p90 OR.
        d = {lbl: c["stats"]["or_score_raw"]["p90"] for lbl, c in present}
        if "llama_self_rwr" in d and "baseline" in d and d["baseline"]:
            print(f"\n  p90 OR — self-distill vs base policy: "
                  f"llama_self_rwr={d['llama_self_rwr']:.4f}  baseline={d['baseline']:.4f}  "
                  f"ratio={d['llama_self_rwr']/d['baseline']:.2f}x")


if __name__ == "__main__":
    main()
