#!/usr/bin/env python3
"""Side-by-side head-to-head on the held-out alpaca set:
raw-Claude teacher vs the distilled claude_rwr student (+ baseline, rwr_v3).

Pulls claude_rwr / baseline / rwr_v3 alpaca stats from the saved held-out eval
JSON, and the raw-Claude arm from the scored generate_claude_heldout.py output.
Both were scored with the same ORRewardModel (k=5.0, c=0.75, d=100) on the SAME
200 held-out alpaca prompts, so the OR numbers are directly comparable.

Usage:
    python compare_heldout_claude.py \
        --held_out_results prompt_iteration_results/held_out_eval/held_out_eval_results_k5.json \
        --claude_scored   prompt_iteration_results/claude_heldout_alpaca.json
"""
import argparse
import json
import os

METRICS = ("or_score_raw", "refusal_delta", "similarity")
COLS = ("mean", "median", "p75", "p90", "p95", "pct_positive")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--held_out_results",
                    default="prompt_iteration_results/held_out_eval/held_out_eval_results_k5.json")
    ap.add_argument("--claude_scored",
                    default="prompt_iteration_results/claude_heldout_alpaca.json")
    ap.add_argument("--corpus", default="alpaca")
    args = ap.parse_args()

    rows = []  # (label, n, stats_by_metric)

    with open(args.held_out_results) as f:
        ho = json.load(f)
    for label, by_corpus in ho["results"].items():
        c = by_corpus.get(args.corpus)
        if not c:
            continue
        rows.append((label, len(c["generations"]), c["stats"]))

    if os.path.isfile(args.claude_scored):
        with open(args.claude_scored) as f:
            cs = json.load(f)
        # generate_claude_heldout.py writes a single variant; take the first scored one.
        for vname, vdata in cs.get("variants", {}).items():
            if "stats" not in vdata:
                continue
            rows.append((f"raw_claude:{vname}", len(vdata["generations"]), vdata["stats"]))
    else:
        print(f"[warn] raw-Claude scored file not found: {args.claude_scored}")
        print("       Showing held-out models only (run generation+scoring first).\n")

    print(f"\n{'='*104}")
    print(f"HEAD-TO-HEAD on held-out {args.corpus}  (k=5.0, c=0.75, d=100; same 200 prompts)")
    print(f"{'='*104}")
    for metric in METRICS:
        print(f"\n  {metric}:")
        print(f"    {'model':24s} {'n':>5s} {'mean':>9s} {'median':>9s} "
              f"{'p75':>9s} {'p90':>9s} {'p95':>9s} {'%pos':>8s}")
        ranked = sorted(rows, key=lambda r: -r[2][metric]["p90"])
        for label, n, stats in ranked:
            s = stats[metric]
            print(f"    {label:24s} {n:5d} {s['mean']:9.4f} {s['median']:9.4f} "
                  f"{s['p75']:9.4f} {s['p90']:9.4f} {s['p95']:9.4f} {s['pct_positive']:7.1%}")

    # Headline ratio on p90 OR.
    or_p90 = {label: stats["or_score_raw"]["p90"] for label, _, stats in rows}
    if any(k.startswith("raw_claude") for k in or_p90) and "claude_rwr" in or_p90:
        rc = next(v for k, v in or_p90.items() if k.startswith("raw_claude"))
        st = or_p90["claude_rwr"]
        print(f"\n  p90 OR — student vs teacher: claude_rwr={st:.4f}  raw_claude={rc:.4f}  "
              f"student/teacher={st/rc:.2f}x" if rc else "")


if __name__ == "__main__":
    main()
