#!/usr/bin/env python3
"""Score API-generated overrefusal candidates with ORRewardModel.

Takes JSONL from overrefusal_sampling/run_generate.py, computes refusal_delta
and similarity via the target model + refusal vector, and outputs shard JSONs
compatible with rwr_data.py.

Usage:
    python score_candidates.py \
        --input_jsonl ../../../overrefusal_sampling/outputs/taskaware_5k.jsonl \
        --refusal_vector_path ../../refusal_vector/Vector_Extraction/refusal_vector.layer032.npz \
        --output_dir ./scored_taskaware \
        --target_model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

# ORRewardModel lives in ppo_or/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ppo_or"))


def load_candidates(jsonl_path: str) -> Dict[int, Dict]:
    """Load JSONL and group by benign_id. Returns {benign_id: {original, candidates[]}}."""
    prompts = {}
    n_skipped = 0
    with open(jsonl_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            # Skip errored or refused generations
            if rec.get("error") or rec.get("refused"):
                n_skipped += 1
                continue
            text = (rec.get("normalized_overrefusal_prompt") or rec.get("overrefusal_prompt", "")).strip()
            if not text or len(text) < 10:
                n_skipped += 1
                continue

            bid = rec["benign_id"]
            if bid not in prompts:
                prompts[bid] = {
                    "original": rec["benign_prompt"],
                    "candidates": [],
                }
            prompts[bid]["candidates"].append({
                "text": text,
                "provider": rec.get("provider", ""),
                "variant": rec.get("prompt_variant", ""),
            })

    total = sum(len(p["candidates"]) for p in prompts.values())
    print(f"[load] {total} valid candidates across {len(prompts)} prompts ({n_skipped} skipped)")
    return prompts


def score_batch(reward_model, originals: List[str], rewrites: List[str], k: float, c: float, d: float):
    """Score a batch and return raw (unclamped) results."""
    out = reward_model.compute_rewards(originals, rewrites)
    # Recompute or_score_raw with our chosen exponent (unclamped)
    sim_weight = torch.exp(k * (out["similarities"] - c))
    or_raw = sim_weight * out["refusal_delta"] / d
    return {
        "or_score_raw": or_raw.cpu().numpy(),
        "refusal_delta": out["refusal_delta"].cpu().numpy(),
        "similarity": out["similarities"].cpu().numpy(),
        "refusal_original": out["refusal_scores_original"].cpu().numpy(),
        "refusal_paraphrase": out["refusal_scores_rewritten"].cpu().numpy(),
    }


def score_all(prompts: Dict[int, Dict], reward_model, batch_size: int, k: float, c: float, d: float):
    """Score all candidates and attach results back to the prompts dict."""
    # Flatten for batched scoring
    flat_orig, flat_rewrite, flat_keys = [], [], []
    for bid, data in prompts.items():
        for i, cand in enumerate(data["candidates"]):
            flat_orig.append(data["original"])
            flat_rewrite.append(cand["text"])
            flat_keys.append((bid, i))

    print(f"[score] Scoring {len(flat_orig)} pairs in batches of {batch_size}...")

    all_results = {k: [] for k in ["or_score_raw", "refusal_delta", "similarity", "refusal_original", "refusal_paraphrase"]}

    for i in range(0, len(flat_orig), batch_size):
        orig_batch = flat_orig[i:i + batch_size]
        rew_batch = flat_rewrite[i:i + batch_size]
        batch_results = score_batch(reward_model, orig_batch, rew_batch, k, c, d)
        for key in all_results:
            all_results[key].extend(batch_results[key].tolist())

        if (i // batch_size + 1) % 50 == 0:
            print(f"  {i + len(orig_batch)}/{len(flat_orig)} scored")

    # Attach scores back to candidates
    for idx, (bid, cand_idx) in enumerate(flat_keys):
        cand = prompts[bid]["candidates"][cand_idx]
        cand["or_score_raw"] = all_results["or_score_raw"][idx]
        cand["refusal_delta"] = all_results["refusal_delta"][idx]
        cand["similarity"] = all_results["similarity"][idx]
        cand["refusal_original"] = all_results["refusal_original"][idx]
        cand["refusal_paraphrase"] = all_results["refusal_paraphrase"][idx]

    print(f"[score] Done")


def save_shards(prompts: Dict[int, Dict], output_dir: str, shard_size: int = 50):
    """Save in the same shard JSON format as or_paraphrase_3k."""
    os.makedirs(output_dir, exist_ok=True)

    # Convert to list sorted by benign_id
    entries = []
    for bid in sorted(prompts.keys()):
        data = prompts[bid]
        paraphrases = []
        for cand in data["candidates"]:
            paraphrases.append({
                "paraphrase": cand["text"],
                "or_score_raw": cand["or_score_raw"],
                "or_score_clamped": min(max(cand["or_score_raw"], 0), 10.0),
                "refusal_delta": cand["refusal_delta"],
                "similarity": cand["similarity"],
                "refusal_original": cand["refusal_original"],
                "refusal_paraphrase": cand["refusal_paraphrase"],
            })
        entries.append({
            "prompt_idx": bid,
            "original": data["original"],
            "paraphrases": paraphrases,
        })

    # Split into shards
    n_shards = max(1, (len(entries) + shard_size - 1) // shard_size)
    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, len(entries))
        shard = entries[start:end]

        # Compute per-prompt stats
        for entry in shard:
            scores = [p["or_score_raw"] for p in entry["paraphrases"]]
            entry["mean_or"] = float(np.mean(scores)) if scores else 0
            entry["std_or"] = float(np.std(scores)) if scores else 0
            entry["max_or"] = float(np.max(scores)) if scores else 0
            entry["min_or"] = float(np.min(scores)) if scores else 0
            entry["n_positive"] = sum(1 for s in scores if s > 0)
            entry["n_paraphrases"] = len(scores)
            entry["mean_sim"] = float(np.mean([p["similarity"] for p in entry["paraphrases"]])) if scores else 0
            entry["rank_score"] = entry["mean_or"] + entry["std_or"]

        path = os.path.join(output_dir, f"or_susceptibility_rankings_shard{shard_idx}.json")
        with open(path, "w") as f:
            json.dump(shard, f, indent=2)

    # Print summary stats
    all_or = [p["or_score_raw"] for e in entries for p in e["paraphrases"]]
    all_delta = [p["refusal_delta"] for e in entries for p in e["paraphrases"]]
    all_sim = [p["similarity"] for e in entries for p in e["paraphrases"]]
    pos = sum(1 for s in all_delta if s > 0)
    print(f"\n[save] {n_shards} shards, {len(entries)} prompts, {len(all_or)} pairs")
    print(f"  refusal_delta: mean={np.mean(all_delta):.3f} median={np.median(all_delta):.3f} %pos={pos/len(all_delta)*100:.1f}%")
    print(f"  similarity:    mean={np.mean(all_sim):.3f} median={np.median(all_sim):.3f}")
    print(f"  or_score_raw:  mean={np.mean(all_or):.3f} median={np.median(all_or):.3f}")
    print(f"  Saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Score API-generated overrefusal candidates")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./scored_taskaware")
    parser.add_argument("--refusal_vector_path", type=str, required=True)
    parser.add_argument("--target_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--activation_layer", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--shard_size", type=int, default=50, help="Prompts per shard file")
    # Reward formula: or_score_raw = exp(k * (sim - c)) * delta / d
    parser.add_argument("--similarity_exponent", type=float, default=5.0,
                        help="k in exp(k*(sim-c)). Lower = delta matters more. Original=18.4, recommended=5.0")
    parser.add_argument("--similarity_center", type=float, default=0.75)
    parser.add_argument("--refusal_divisor", type=float, default=100.0)
    args = parser.parse_args()

    from config import ModelConfig as PPOModelConfig, RewardConfig
    from reward_model import ORRewardModel

    # Load candidates
    prompts = load_candidates(args.input_jsonl)

    # Set up reward model
    model_config = PPOModelConfig(
        base_model_id=args.target_model,
        refusal_vector_path=args.refusal_vector_path,
        activation_layer=args.activation_layer,
    )
    reward_config = RewardConfig()
    reward_model = ORRewardModel(model_config, reward_config)

    # Score
    score_all(prompts, reward_model, args.batch_size, args.similarity_exponent, args.similarity_center, args.refusal_divisor)
    reward_model.cleanup()

    # Save
    save_shards(prompts, args.output_dir, args.shard_size)


if __name__ == "__main__":
    main()
