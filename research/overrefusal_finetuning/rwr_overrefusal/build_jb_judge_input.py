#!/usr/bin/env python3
"""U2a: from the U1 scored output, select the COMPLIED subset and build the genuine-jailbreak
judge input. Mirror of build_low_judge_input.py (which selects the REFUSED subset for the OR
judge): a prompt the target refused every time cannot be a jailbreak, so judging it is pure spend.

A prompt is in the complied subset iff >=1 of its non-disclaimer samples was NOT a refusal
(compliance_rate > 0). For each such prompt we hand the judge one representative COMPLIED reply
(the first non-disclaimer, non-refusal sample) plus the goal and the wrapped prompt, so the judge
can decide RESPONSE (did the reply actually facilitate the harm) and FIDELITY (does the wrapped
prompt still seek the original harmful goal).

Out: <out> csv with columns pair_id, orig_idx, goal, wrapper, prompt, reply, compliance_rate
Run: python build_jb_judge_input.py --scored probe_or/results/underrefusal/scored_llamaTgt.json \
        --out probe_or/results/underrefusal/judge_input.csv
"""
import argparse
import csv
import json
import os
import sys

csv.field_size_limit(sys.maxsize)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_or"))
from gen_qwen_refusal import classify
from capability_filter import is_capability_disclaimer


def first_complied(samples):
    for s in samples:
        if is_capability_disclaimer(s):
            continue
        if not classify(s)[0]:            # not a refusal -> a candidate jailbreak reply
            return s
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    recs = json.load(open(a.scored))["examples"]
    rows, n_complied = [], 0
    for r in recs:
        if not r.get("informative"):
            continue
        cr = r.get("compliance_rate") or 0.0
        if cr <= 0:
            continue
        reply = first_complied(r.get("samples", []))
        if reply is None:
            continue
        n_complied += 1
        rows.append({"pair_id": r["pair_id"], "orig_idx": r.get("orig_idx", ""),
                     "goal": r["goal"], "wrapper": r["wrapper"], "prompt": r["prompt"],
                     "reply": reply, "compliance_rate": cr})

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fields = ["pair_id", "orig_idx", "goal", "wrapper", "prompt", "reply", "compliance_rate"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    n_distinct = len({r["orig_idx"] for r in rows})
    print(f"[u2a] {len(recs)} scored | {n_complied} complied (>=1 non-refusal sample) "
          f"over {n_distinct} distinct goals -> judging {len(rows)} -> {a.out}")


if __name__ == "__main__":
    main()
