#!/usr/bin/env python3
"""Prepare the original StrongREJECT prompts as a direct-harmful GCG substrate.

Only rows whose source is ``custom`` are retained.  This excludes the 25 AdvBench
rows because AdvBench was already used to fit anon-repo's harmful direction.
"""
import argparse
import csv
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.input, newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("source") == "custom"]
    if len(rows) != 221:
        raise ValueError(f"expected 221 custom StrongREJECT prompts, found {len(rows)}")
    prompts = [row["forbidden_prompt"].strip() for row in rows]
    if len(prompts) != len(set(prompts)):
        raise ValueError("duplicate custom StrongREJECT prompts")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = f"{args.out}.tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "category", "source", "dataset"])
        writer.writeheader()
        for row, prompt in zip(rows, prompts):
            writer.writerow({"prompt": prompt, "category": row["category"],
                             "source": "custom", "dataset": "StrongREJECT"})
    os.replace(tmp, args.out)
    print(f"[done] wrote {len(rows)} direct harmful prompts to {args.out}")


if __name__ == "__main__":
    main()
