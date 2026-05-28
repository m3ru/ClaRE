#!/usr/bin/env python3
"""Dedupe scored RWR shards in place (or to a new directory).

Two filters, each opt-in:

  --dedupe_exact      (default ON): within each source prompt's `paraphrases` list,
                      drop exact text duplicates of the rewrite, keeping the entry
                      with the highest or_score_raw (ties broken by first occurrence).
                      This handles the "5 samples per prompt, Claude converged on the
                      same wording 3 times" case observed in the medium pilot.

  --max_sim FLOAT     (default 1.0 = off): drop pairs whose similarity to the source
                      is above this threshold. Set e.g. 0.995 to filter out rewrites
                      that are essentially the source verbatim ("model just retyped it").

  --min_sim FLOAT     (default 0.5): drop pairs below this similarity. Same as the
                      `BinningConfig.similarity_floor` applied later by `rwr_data` —
                      useful here too because it shrinks downstream IO.

  --min_or FLOAT      (default None = off): drop pairs with or_score_raw below this.
                      Default off so we preserve negative-OR pairs for analysis;
                      `rwr_data.filter_and_bin` filters them at training time.

The shards stay in the same `or_susceptibility_rankings_shardN.json` format consumed
by `rwr_data.load_shards`, so the deduped output drops straight into `train_rwr.py`.

Usage:
    python dedupe_shards.py \\
        --input_dir  prompt_iteration_results/dataset_research_framing_full_shards \\
        --output_dir prompt_iteration_results/dataset_research_framing_full_shards_deduped
"""
import argparse
import glob
import json
import os
from collections import Counter
from typing import Dict, List


def dedup_paraphrases(
    paraphrases: List[Dict],
    dedupe_exact: bool,
    max_sim: float,
    min_sim: float,
    min_or,
) -> Dict:
    """Apply per-prompt filters. Returns dict with the kept list and counts."""
    counts = {"in": len(paraphrases), "exact_dup": 0, "above_max_sim": 0,
              "below_min_sim": 0, "below_min_or": 0}
    kept: List[Dict] = []

    # Pre-sort by or_score_raw desc so when we encounter duplicates we keep the
    # highest-scoring one (first seen wins, ties broken by original order).
    sorted_paraphrases = sorted(paraphrases, key=lambda p: -p["or_score_raw"])

    seen_text: set = set()
    for p in sorted_paraphrases:
        text = p["paraphrase"].strip()
        sim = p["similarity"]
        or_score = p["or_score_raw"]

        if dedupe_exact and text in seen_text:
            counts["exact_dup"] += 1
            continue
        if sim > max_sim:
            counts["above_max_sim"] += 1
            continue
        if sim < min_sim:
            counts["below_min_sim"] += 1
            continue
        if min_or is not None and or_score < min_or:
            counts["below_min_or"] += 1
            continue

        seen_text.add(text)
        kept.append(p)

    counts["out"] = len(kept)
    return {"kept": kept, "counts": counts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir",  required=True,
                    help="Directory with or_susceptibility_rankings_shard*.json")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write deduped shards. Created if missing.")
    ap.add_argument("--dedupe_exact", action=argparse.BooleanOptionalAction, default=True,
                    help="Drop exact-text duplicate rewrites within a source prompt (default ON)")
    ap.add_argument("--max_sim", type=float, default=1.0,
                    help="Drop pairs above this similarity (1.0 = no max filter). "
                         "Use 0.995 to filter out 'model just retyped it' near-copies.")
    ap.add_argument("--min_sim", type=float, default=0.5,
                    help="Drop pairs below this similarity (matches BinningConfig.similarity_floor)")
    ap.add_argument("--min_or",  type=float, default=None,
                    help="Drop pairs with or_score_raw below this. Off by default so "
                         "negative pairs are preserved for analysis (training filter applies later).")
    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"input_dir not a directory: {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    shard_files = sorted(glob.glob(os.path.join(args.input_dir, "or_susceptibility_rankings_shard*.json")))
    if not shard_files:
        raise SystemExit(f"no shards found in {args.input_dir}")

    totals = Counter()
    n_empty_prompts = 0
    n_kept_prompts = 0
    print(f"[dedup] processing {len(shard_files)} shard files")
    for path in shard_files:
        with open(path) as f:
            shard = json.load(f)
        new_shard = []
        for entry in shard:
            result = dedup_paraphrases(
                entry["paraphrases"],
                dedupe_exact=args.dedupe_exact,
                max_sim=args.max_sim,
                min_sim=args.min_sim,
                min_or=args.min_or,
            )
            for k, v in result["counts"].items():
                totals[k] += v
            new_paras = result["kept"]
            if not new_paras:
                n_empty_prompts += 1
                continue  # drop prompts whose every rewrite was filtered out
            n_kept_prompts += 1
            new_entry = dict(entry)
            new_entry["paraphrases"] = new_paras
            # Recompute per-prompt aggregates so downstream tooling shows the
            # post-dedup state, not the stale pre-dedup numbers.
            ors = [p["or_score_raw"] for p in new_paras]
            sims = [p["similarity"] for p in new_paras]
            import statistics
            new_entry["mean_or"]      = statistics.fmean(ors) if ors else 0.0
            new_entry["max_or"]       = max(ors) if ors else 0.0
            new_entry["min_or"]       = min(ors) if ors else 0.0
            new_entry["std_or"]       = (statistics.pstdev(ors) if len(ors) > 1 else 0.0)
            new_entry["n_positive"]   = sum(1 for o in ors if o > 0)
            new_entry["n_paraphrases"] = len(new_paras)
            new_entry["mean_sim"]     = statistics.fmean(sims) if sims else 0.0
            new_entry["rank_score"]   = new_entry["mean_or"] + new_entry["std_or"]
            new_shard.append(new_entry)

        out_path = os.path.join(args.output_dir, os.path.basename(path))
        with open(out_path, "w") as f:
            json.dump(new_shard, f, indent=2)
        print(f"[dedup]   {os.path.basename(path)}: {len(new_shard)} prompts retained "
              f"(of {len(shard)})")

    in_total  = totals["in"]
    out_total = totals["out"]
    print(f"\n[dedup] summary across all shards:")
    print(f"  input pairs:         {in_total}")
    print(f"  exact duplicates:    {totals['exact_dup']:>6d}")
    print(f"  above --max_sim:     {totals['above_max_sim']:>6d}")
    print(f"  below --min_sim:     {totals['below_min_sim']:>6d}")
    print(f"  below --min_or:      {totals['below_min_or']:>6d}")
    print(f"  output pairs:        {out_total}  ({out_total/max(in_total,1)*100:.1f}% of input)")
    print(f"  prompts kept:        {n_kept_prompts}")
    print(f"  prompts emptied:     {n_empty_prompts}")
    print(f"  output dir:          {args.output_dir}")


if __name__ == "__main__":
    main()
