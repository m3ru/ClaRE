#!/usr/bin/env python3
"""Extract top-scoring paraphrases from each dataset's OR ranking results.

For each dataset, outputs:
  - A JSONL with the best paraphrase per prompt (highest or_score_raw)
  - A combined JSONL merging all datasets
  - Summary statistics

Usage:
    python extract_top_paraphrases.py \
        --input_dirs cybermetric=./or_paraphrase_cybermetric,sciq=./or_paraphrase_sciq,medqa=./or_paraphrase_medqa,pubmedqa=./or_paraphrase_pubmedqa \
        --output_dir ./top_paraphrases \
        --top_k 1
"""

import argparse
import json
import os
from typing import Dict, List


def load_rankings(path: str) -> List[Dict]:
    """Load a single or_susceptibility_rankings JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def load_all_shards(input_dir: str) -> List[Dict]:
    """Load all shard files or a single rankings file from a directory."""
    entries = []

    # Check for single file first
    single = os.path.join(input_dir, "or_susceptibility_rankings.json")
    if os.path.isfile(single):
        return load_rankings(single)

    # Otherwise load shards
    for fname in sorted(os.listdir(input_dir)):
        if fname.startswith("or_susceptibility_rankings") and fname.endswith(".json"):
            entries.extend(load_rankings(os.path.join(input_dir, fname)))

    return entries


def extract_top_k(entries: List[Dict], top_k: int = 1, min_similarity: float = 0.3) -> List[Dict]:
    """For each prompt, extract the top_k paraphrases by or_score_raw."""
    results = []
    for entry in entries:
        paraphrases = entry.get("paraphrases", [])
        # Filter by minimum similarity
        valid = [p for p in paraphrases if p.get("similarity", 0) >= min_similarity]
        if not valid:
            continue

        # Sort by or_score_raw descending
        valid.sort(key=lambda x: x.get("or_score_raw", 0), reverse=True)
        top = valid[:top_k]

        for p in top:
            results.append({
                "original": entry["original"],
                "paraphrase": p["paraphrase"],
                "or_score_raw": p["or_score_raw"],
                "refusal_delta": p["refusal_delta"],
                "similarity": p["similarity"],
                "prompt_idx": entry.get("prompt_idx", -1),
            })

    return results


def save_jsonl(records: List[Dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def print_stats(name: str, records: List[Dict]):
    if not records:
        print(f"  {name}: 0 records")
        return
    scores = [r["or_score_raw"] for r in records]
    deltas = [r["refusal_delta"] for r in records]
    sims = [r["similarity"] for r in records]
    pos = sum(1 for s in deltas if s > 0)
    print(f"  {name}: {len(records)} pairs")
    print(f"    or_score:  mean={sum(scores)/len(scores):.4f}  "
          f"max={max(scores):.4f}  min={min(scores):.4f}")
    print(f"    refusal_delta: mean={sum(deltas)/len(deltas):.4f}  %pos={pos/len(deltas)*100:.1f}%")
    print(f"    similarity: mean={sum(sims)/len(sims):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Extract top paraphrases from OR ranking results")
    parser.add_argument("--input_dirs", type=str, required=True,
                        help="Comma-separated name=path pairs, e.g. cybermetric=./dir1,sciq=./dir2")
    parser.add_argument("--output_dir", type=str, default="./top_paraphrases")
    parser.add_argument("--top_k", type=int, default=1,
                        help="Number of top paraphrases to extract per prompt")
    parser.add_argument("--min_similarity", type=float, default=0.3,
                        help="Minimum similarity threshold")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse input dirs
    dataset_dirs = {}
    for part in args.input_dirs.split(","):
        name, path = part.strip().split("=", 1)
        dataset_dirs[name.strip()] = path.strip()

    all_combined = []
    print(f"\nExtracting top-{args.top_k} paraphrases (min_sim={args.min_similarity}):\n")

    for name, input_dir in dataset_dirs.items():
        if not os.path.isdir(input_dir):
            print(f"  [skip] {name}: directory not found ({input_dir})")
            continue

        entries = load_all_shards(input_dir)
        top = extract_top_k(entries, top_k=args.top_k, min_similarity=args.min_similarity)

        # Add dataset source tag
        for rec in top:
            rec["dataset"] = name

        # Sort by or_score descending
        top.sort(key=lambda x: x["or_score_raw"], reverse=True)

        # Save per-dataset
        out_path = os.path.join(args.output_dir, f"top_{name}.jsonl")
        save_jsonl(top, out_path)
        print_stats(name, top)
        print(f"    -> {out_path}")

        all_combined.extend(top)

    # Save combined
    all_combined.sort(key=lambda x: x["or_score_raw"], reverse=True)
    combined_path = os.path.join(args.output_dir, "top_all_combined.jsonl")
    save_jsonl(all_combined, combined_path)
    print(f"\nCombined:")
    print_stats("all", all_combined)
    print(f"  -> {combined_path}")

    print(f"\n[done] All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
