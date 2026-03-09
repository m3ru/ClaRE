#!/usr/bin/env python3
"""Merge shard outputs from parallel paraphrase_and_score.py runs.

Usage:
    python merge_shards.py --output_dir <dir> --num_shards <N>
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Merge paraphrase scoring shards")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory containing shard output files")
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--output", type=str, default=None,
                        help="Merged output path (default: <output_dir>/or_susceptibility_rankings.json)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.output_dir, "or_susceptibility_rankings.json")

    all_results = []
    missing = []
    for shard_id in range(args.num_shards):
        shard_file = os.path.join(
            args.output_dir,
            f"or_susceptibility_rankings_shard{shard_id}.json")
        if not os.path.exists(shard_file):
            missing.append(shard_id)
            continue
        with open(shard_file, "r") as f:
            shard_data = json.load(f)
        all_results.extend(shard_data)
        print(f"[merge] Shard {shard_id}: {len(shard_data)} prompts")

    if missing:
        print(f"[warn] Missing shards: {missing}")

    all_results.sort(key=lambda x: x.get("prompt_idx", 0))

    from paraphrase_and_score import rank_prompts, _print_summary
    ranked = rank_prompts(all_results)

    with open(args.output, "w") as f:
        json.dump(ranked, f, indent=2)
    print(f"\n[done] Merged {len(ranked)} prompts → {args.output}")

    _print_summary(ranked)


if __name__ == "__main__":
    main()
