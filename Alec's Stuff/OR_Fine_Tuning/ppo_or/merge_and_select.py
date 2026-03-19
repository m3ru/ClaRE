"""Merge shard files and select top 20% prompts by mean OR score.

Usage:
    python merge_and_select.py --shard_dir <dir> [--num_shards 60] [--min_paraphrases 20]
"""
import argparse
import json
import os
import statistics
import sys


DEFAULT_SHARD_DIR = os.path.join(os.path.dirname(__file__), "or_paraphrase_3k")
NUM_SHARDS = 60
TOP_FRACTION = 0.20
MIN_PARAPHRASES = 20


def main():
    parser = argparse.ArgumentParser(description="Merge shards and select top OR prompts")
    parser.add_argument("--shard_dir", type=str, default=DEFAULT_SHARD_DIR,
                        help="Directory containing shard JSON files")
    parser.add_argument("--num_shards", type=int, default=NUM_SHARDS)
    parser.add_argument("--min_paraphrases", type=int, default=MIN_PARAPHRASES,
                        help="Minimum valid paraphrases required per prompt")
    parser.add_argument("--top_fraction", type=float, default=TOP_FRACTION)
    parser.add_argument("--output_jsonl", type=str, default=None,
                        help="Output JSONL path for RL training (default: <shard_dir>/top20pct_or_prompts.jsonl)")
    args = parser.parse_args()

    shard_dir = args.shard_dir
    all_prompts = []
    for i in range(args.num_shards):
        path = os.path.join(shard_dir, f"or_susceptibility_rankings_shard{i}.json")
        if not os.path.exists(path):
            print(f"WARNING: missing shard {i}")
            continue
        with open(path) as f:
            shard = json.load(f)
        all_prompts.extend(shard)
        print(f"  Shard {i}: {len(shard)} prompts")

    print(f"\nTotal prompts loaded: {len(all_prompts)}")

    # Compute per-prompt stats
    for p in all_prompts:
        scores = [pp["or_score_clamped"] for pp in p["paraphrases"]]
        p["mean_or"] = statistics.mean(scores) if scores else 0.0
        p["max_or"] = max(scores) if scores else 0.0
        p["std_or"] = statistics.stdev(scores) if len(scores) > 1 else 0.0
        p["n_paraphrases"] = len(scores)

    # Filter for minimum paraphrases
    before = len(all_prompts)
    all_prompts = [p for p in all_prompts if p["n_paraphrases"] >= args.min_paraphrases]
    dropped = before - len(all_prompts)
    if dropped:
        print(f"Dropped {dropped} prompts with < {args.min_paraphrases} paraphrases")
    print(f"Prompts after filtering: {len(all_prompts)}")

    # Sort by mean OR score descending
    all_prompts.sort(key=lambda x: x["mean_or"], reverse=True)

    # Stats
    means = [p["mean_or"] for p in all_prompts]
    n_para = [p["n_paraphrases"] for p in all_prompts]
    print(f"Mean OR across prompts: {statistics.mean(means):.4f}")
    print(f"Median OR across prompts: {statistics.median(means):.4f}")
    print(f"Mean paraphrases per prompt: {statistics.mean(n_para):.1f}")
    print(f"Min paraphrases: {min(n_para)}, Max: {max(n_para)}")

    # Select top fraction
    n_select = int(len(all_prompts) * args.top_fraction)
    top_prompts = all_prompts[:n_select]
    bottom = all_prompts[n_select:]

    print(f"\n--- Selection ---")
    print(f"Top {args.top_fraction*100:.0f}%: {n_select} prompts")
    top_means = [p["mean_or"] for p in top_prompts]
    bot_means = [p["mean_or"] for p in bottom]
    print(f"  Top mean OR: {statistics.mean(top_means):.4f} (range {min(top_means):.4f} - {max(top_means):.4f})")
    if bot_means:
        print(f"  Bottom mean OR: {statistics.mean(bot_means):.4f}")

    # Show some examples
    print(f"\n--- Top 10 prompts ---")
    for p in top_prompts[:10]:
        orig = p["original"][:120].replace("\n", " ")
        print(f"  mean={p['mean_or']:.3f}  std={p['std_or']:.3f}  n={p['n_paraphrases']}  | {orig}...")

    print(f"\n--- Bottom 10 of selected ---")
    for p in top_prompts[-10:]:
        orig = p["original"][:120].replace("\n", " ")
        print(f"  mean={p['mean_or']:.3f}  std={p['std_or']:.3f}  n={p['n_paraphrases']}  | {orig}...")

    # Save full ranked list
    full_out = os.path.join(shard_dir, "all_prompts_ranked.json")
    with open(full_out, "w") as f:
        json.dump(all_prompts, f, indent=2)
    print(f"\nFull ranked list saved: {full_out}")

    # Save top prompts as JSONL for RL training
    jsonl_out = args.output_jsonl or os.path.join(shard_dir, "top20pct_or_prompts.jsonl")
    with open(jsonl_out, "w") as f:
        for p in top_prompts:
            f.write(json.dumps({"prompt": p["original"]}) + "\n")
    print(f"Top {n_select} prompts saved as JSONL: {jsonl_out}")

    # Save top with full data for inspection
    top_full_out = os.path.join(shard_dir, "top20pct_full.json")
    with open(top_full_out, "w") as f:
        json.dump(top_prompts, f, indent=2)
    print(f"Top {n_select} full data saved: {top_full_out}")

if __name__ == "__main__":
    main()
