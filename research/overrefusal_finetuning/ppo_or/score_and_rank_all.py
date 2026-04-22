#!/usr/bin/env python3
"""Score and rank paraphrases across multiple datasets.

Reads each {dataset}/paraphrases.json (produced by paraphrase_openai_batch.py)
and, using the same ORRewardModel + rank_prompts logic as paraphrase_and_score.py,
writes:

    {dataset}/ranked.json           -- fully scored + ranked results
    {dataset}/top20pct_prompts.json -- top 20% by rank_score

Each dataset is processed fully before moving on so the reward model stays in
memory (big win: avoids reloading Llama-3-8B five times).

Designed to be the only thing you run for Phase 2 + Phase 3 — all 5 datasets
in one shot:

    python score_and_rank_all.py \
        --paraphrases_root research/overrefusal_finetuning/paraphrases_api \
        --refusal_vector_path research/refusal_vector/3_Vector_Extraction/refusal_vector.layer032.npz
"""

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


DEFAULT_DATASETS = ["cybermetric", "sciq", "medqa", "pubmedqa", "dolly"]


def score_paraphrases(reward_model, original: str, paraphrases: List[str],
                      batch_size: int = 8) -> List[Dict]:
    """Same signature / output as paraphrase_and_score.score_paraphrases()."""
    rc = reward_model.reward_config
    results = []
    for i in range(0, len(paraphrases), batch_size):
        batch = paraphrases[i:i + batch_size]
        originals = [original] * len(batch)
        reward_out = reward_model.compute_rewards(originals, batch)
        raw_or = (reward_out["sim_weight"] * reward_out["refusal_delta"]
                  / rc.refusal_divisor)
        for j in range(len(batch)):
            results.append({
                "paraphrase": batch[j],
                "or_score_raw": raw_or[j].item(),
                "or_score_clamped": reward_out["rewards"][j].item(),
                "refusal_delta": reward_out["refusal_delta"][j].item(),
                "similarity": reward_out["similarities"][j].item(),
                "refusal_original": reward_out["refusal_scores_original"][j].item(),
                "refusal_paraphrase": reward_out["refusal_scores_rewritten"][j].item(),
            })
    return results


def rank_prompts(all_results: List[Dict]) -> List[Dict]:
    for entry in all_results:
        scores = [p["or_score_raw"] for p in entry["paraphrases"]]
        if scores:
            entry["mean_or"] = float(np.mean(scores))
            entry["std_or"] = float(np.std(scores))
            entry["max_or"] = float(np.max(scores))
            entry["min_or"] = float(np.min(scores))
            entry["n_positive"] = sum(1 for s in scores if s > 0)
            entry["n_paraphrases"] = len(scores)
            entry["mean_sim"] = float(np.mean(
                [p["similarity"] for p in entry["paraphrases"]]))
        else:
            entry.update({"mean_or": 0, "std_or": 0, "max_or": 0, "min_or": 0,
                          "n_positive": 0, "n_paraphrases": 0, "mean_sim": 0})
        entry["rank_score"] = entry["mean_or"] + entry["std_or"]
    all_results.sort(key=lambda x: x["rank_score"], reverse=True)
    return all_results


def process_dataset(reward_model, dataset_name: str, dataset_dir: str,
                    batch_size: int, log_every: int = 100,
                    max_prompts: int = 0, output_subdir: str = "") -> None:
    paraphrases_path = os.path.join(dataset_dir, "paraphrases.json")
    if not os.path.exists(paraphrases_path):
        print(f"[{dataset_name}] SKIP: no paraphrases.json at {paraphrases_path}")
        return

    with open(paraphrases_path) as f:
        cached = json.load(f)
    print(f"[{dataset_name}] Loaded {len(cached)} prompts from {paraphrases_path}")

    if max_prompts > 0 and len(cached) > max_prompts:
        cached = cached[:max_prompts]
        print(f"[{dataset_name}] Capped to first {max_prompts} prompts")

    out_dir = os.path.join(dataset_dir, output_subdir) if output_subdir else dataset_dir
    os.makedirs(out_dir, exist_ok=True)

    # Resume support: partial ranked-scratch file
    scratch_path = os.path.join(out_dir, "ranked.scratch.json")
    if os.path.exists(scratch_path):
        with open(scratch_path) as f:
            all_results = json.load(f)
        done_indices = {r.get("prompt_idx") for r in all_results}
        print(f"[{dataset_name}] Resuming — {len(all_results)} already scored")
    else:
        all_results = []
        done_indices = set()

    start_ts = time.time()
    for i, entry in enumerate(cached):
        if entry.get("prompt_idx", i) in done_indices:
            continue
        original = entry["original"]
        paraphrases = entry["paraphrases_text"]

        if not paraphrases:
            all_results.append({
                "prompt_idx": entry.get("prompt_idx", i),
                "original": original,
                "paraphrases": [],
            })
            continue

        scores = score_paraphrases(reward_model, original, paraphrases,
                                   batch_size=batch_size)
        all_results.append({
            "prompt_idx": entry.get("prompt_idx", i),
            "original": original,
            "paraphrases": scores,
        })

        if (i + 1) % log_every == 0:
            elapsed = time.time() - start_ts
            rate = (i + 1) / max(elapsed, 1e-3)
            or_vals = [s["or_score_raw"] for s in scores]
            print(f"[{dataset_name}] {i+1}/{len(cached)}  "
                  f"rate={rate:.2f}/s  mean_or={np.mean(or_vals):.4f} "
                  f"std={np.std(or_vals):.4f}  — {original[:60]}")
            # checkpoint
            with open(scratch_path, "w") as f:
                json.dump(all_results, f)

    ranked = rank_prompts(all_results)
    out_path = os.path.join(out_dir, "ranked.json")
    with open(out_path, "w") as f:
        json.dump(ranked, f, indent=2)
    print(f"[{dataset_name}] Wrote {out_path}")

    # Top 20%
    n_top = max(1, int(round(0.2 * len(ranked))))
    top = ranked[:n_top]
    top_path = os.path.join(out_dir, "top20pct_prompts.json")
    with open(top_path, "w") as f:
        json.dump(top, f, indent=2)
    print(f"[{dataset_name}] Wrote top-{n_top} to {top_path}")

    if os.path.exists(scratch_path):
        os.remove(scratch_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paraphrases_root", required=True,
                        help="Directory containing per-dataset subdirectories")
    parser.add_argument("--refusal_vector_path", required=True)
    parser.add_argument("--scoring_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--activation_layer", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help=f"Subset of datasets (default: {DEFAULT_DATASETS})")
    parser.add_argument("--no_quantize", action="store_true",
                        help="Disable 4-bit quantization (needs more VRAM)")
    parser.add_argument("--max_prompts_per_dataset", type=int, default=0,
                        help="Cap per-dataset prompts (0 = all). Use for subset run.")
    parser.add_argument("--output_subdir", type=str, default="",
                        help="Write outputs under {dataset}/{output_subdir}/ instead of {dataset}/")
    args = parser.parse_args()

    datasets = args.datasets or DEFAULT_DATASETS
    print(f"[setup] Will process: {datasets}")

    from config import ModelConfig, RewardConfig
    from reward_model import ORRewardModel

    model_config = ModelConfig(
        base_model_id=args.scoring_model,
        refusal_vector_path=args.refusal_vector_path,
        activation_layer=args.activation_layer,
        quantize_target_model=not args.no_quantize,
    )
    reward_config = RewardConfig()

    print(f"[setup] Loading reward model ({args.scoring_model}, "
          f"layer {args.activation_layer})...")
    reward_model = ORRewardModel(model_config, reward_config)

    for ds in datasets:
        ds_dir = os.path.join(args.paraphrases_root, ds)
        process_dataset(reward_model, ds, ds_dir,
                        batch_size=args.batch_size,
                        max_prompts=args.max_prompts_per_dataset,
                        output_subdir=args.output_subdir)

    reward_model.cleanup()
    gc.collect()
    print("\n[all done] Scored + ranked all datasets.")


if __name__ == "__main__":
    main()
