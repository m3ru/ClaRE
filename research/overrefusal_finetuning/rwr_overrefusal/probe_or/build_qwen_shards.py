#!/usr/bin/env python3
"""Convert the Qwen-scored CSV (score_qwen_or.py) into RWR shards grouped by
original, in the format load_shards() expects:
  {original, paraphrases:[{paraphrase, refusal_delta, similarity, or_score_raw}]}

refusal_delta = Qwen probe_delta; or_score_raw = qwen_or (train recomputes it from
refusal_delta+similarity, so this is just a placeholder there). All pairs are
written; filter_and_bin drops sim<floor and OR<0 at train time.
"""
import argparse
import csv
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored_csv", required=True)
    ap.add_argument("--out_shards", required=True)
    ap.add_argument("--n_shards", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(args.out_shards, exist_ok=True)

    by_orig = {}
    n_pairs = 0
    for r in csv.DictReader(open(args.scored_csv)):
        by_orig.setdefault(r["original"], []).append({
            "paraphrase": r["rewrite"],
            "refusal_delta": float(r["probe_delta"]),
            "similarity": float(r["similarity"]),
            "or_score_raw": float(r["qwen_or"]),
        })
        n_pairs += 1

    recs = [{"original": o, "paraphrases": ps} for o, ps in by_orig.items()]
    per = (len(recs) + args.n_shards - 1) // args.n_shards
    written = 0
    for si in range(args.n_shards):
        chunk = recs[si * per:(si + 1) * per]
        if chunk:
            json.dump(chunk, open(os.path.join(args.out_shards, f"or_susceptibility_rankings_shard{si}.json"), "w"))
            written += len(chunk)
    print(f"[shards] {written} originals / {n_pairs} pairs -> {args.out_shards}")


if __name__ == "__main__":
    main()
