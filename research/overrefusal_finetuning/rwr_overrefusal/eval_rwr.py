#!/usr/bin/env python3
"""Evaluate a trained RWR model against held-out prompts.

Generates paraphrases from the RWR-trained model (and optionally the base model
as a baseline), scores them with the ORRewardModel, and compares against the
training data distribution.

Usage:
    python eval_rwr.py \
        --adapter_dir ./rwr_checkpoints/final \
        --shard_dir ../or_paraphrase_3k \
        --refusal_vector_path ../../refusal_vector/3_Vector_Extraction/refusal_vector.layer032.npz \
        --output eval_results.json
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ORRewardModel lives in ppo_or/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ppo_or"))

from rwr_config import BinningConfig, DataConfig, SYSTEM_PROMPT, PROMPT_TEMPLATE
from rwr_data import filter_and_bin, load_shards, train_val_split


def load_generator(base_model_id: str, adapter_dir: str = None, torch_dtype="bfloat16"):
    """Load base model, optionally with a LoRA adapter merged in."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map.get(torch_dtype, torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched generation with decoder-only models

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, token=hf_token, torch_dtype=dtype,
        device_map="auto", attn_implementation="eager",
    )

    if adapter_dir:
        print(f"[eval] Loading LoRA adapter from {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def generate_paraphrases(
    model, tokenizer, prompts: List[str],
    n_per_prompt: int = 5, temperature: float = 0.7,
    max_new_tokens: int = 64, batch_size: int = 4,
) -> Dict[str, List[str]]:
    """Generate n paraphrases per prompt. Returns {original: [generations]}."""
    results = {}

    for i, p in enumerate(prompts):
        user_content = PROMPT_TEMPLATE.format(prompt=p)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        inputs = tokenizer(
            [text], return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                num_return_sequences=n_per_prompt,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        results[p] = []
        for ids in output_ids:
            gen_text = tokenizer.decode(ids[prompt_len:], skip_special_tokens=True).strip()
            results[p].append(gen_text)

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{len(prompts)} prompts")

    return results


def score_generations(generations: Dict[str, List[str]], refusal_vector_path: str, base_model_id: str):
    """Score all (original, generation) pairs using ORRewardModel.
    Uses scoring params from rwr_config.BinningConfig (k=5.0, c=0.75, d=100) so
    eval OR is on the same scale as the training-data scoring, not RewardConfig defaults."""
    from config import ModelConfig as PPOModelConfig, RewardConfig
    from reward_model import ORRewardModel
    from rwr_config import BinningConfig

    binning = BinningConfig()
    model_config = PPOModelConfig(
        base_model_id=base_model_id,
        refusal_vector_path=refusal_vector_path,
        activation_layer=32,
    )
    reward_config = RewardConfig()
    reward_config.similarity_exponent = binning.similarity_exponent
    reward_config.similarity_center = binning.similarity_center
    reward_config.refusal_divisor = binning.refusal_divisor
    reward_model = ORRewardModel(model_config, reward_config)

    scored = []
    all_originals, all_gens = [], []
    for orig, gens in generations.items():
        for g in gens:
            all_originals.append(orig)
            all_gens.append(g)

    # Score in batches
    batch_size = 8
    for i in range(0, len(all_originals), batch_size):
        orig_batch = all_originals[i:i + batch_size]
        gen_batch = all_gens[i:i + batch_size]
        rewards = reward_model.compute_rewards(orig_batch, gen_batch)

        for j in range(len(orig_batch)):
            sim = rewards["similarities"][j].item()
            delta = rewards["refusal_delta"][j].item()
            sim_weight = rewards["sim_weight"][j].item()
            rc = reward_config
            or_raw = sim_weight * delta / rc.refusal_divisor
            scored.append({
                "original": orig_batch[j],
                "generation": gen_batch[j],
                "refusal_delta": delta,
                "similarity": sim,
                "or_score_raw": or_raw,
            })

    reward_model.cleanup()
    return scored


def compute_stats(scores: List[float], label: str) -> Dict:
    """Compute summary statistics for a list of scores."""
    arr = np.array(scores)
    return {
        "label": label,
        "n": len(arr),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "pct_positive": float((arr > 0).mean()),
    }


def get_training_data_stats(val_pairs: List[Dict]) -> Dict:
    """Compute stats from the held-out data for comparison."""
    or_scores = [p["or_score_raw"] for p in val_pairs]
    deltas = [p["refusal_delta"] for p in val_pairs]
    sims = [p["similarity"] for p in val_pairs]
    return {
        "or_score_raw": compute_stats(or_scores, "val_data_or_score"),
        "refusal_delta": compute_stats(deltas, "val_data_refusal_delta"),
        "similarity": compute_stats(sims, "val_data_similarity"),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RWR-trained model")
    parser.add_argument("--adapter_dir", type=str, required=True,
                        help="Path to trained LoRA adapter (e.g., rwr_checkpoints/final)")
    parser.add_argument("--shard_dir", type=str, default="../or_paraphrase_3k")
    parser.add_argument("--refusal_vector_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--n_per_prompt", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gen_batch_size", type=int, default=4)
    parser.add_argument("--score_batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_eval_prompts", type=int, default=200,
                        help="Cap on number of val prompts to evaluate (random subset)")
    parser.add_argument("--eval_base_model", action="store_true",
                        help="Also generate from the base model (no adapter) as a baseline")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # --- Load val prompts from the same split used for training ---
    data_config = DataConfig(shard_dir=args.shard_dir)
    binning_config = BinningConfig()
    pairs = load_shards(data_config)
    filtered, weights = filter_and_bin(pairs, binning_config)
    _, _, val_pairs, _ = train_val_split(filtered, weights, val_fraction=0.1, seed=args.seed)

    val_originals = list({p["original"] for p in val_pairs})
    if args.max_eval_prompts and len(val_originals) > args.max_eval_prompts:
        import random as _rnd
        _rnd.seed(args.seed)
        val_originals = _rnd.sample(val_originals, args.max_eval_prompts)
    print(f"[eval] {len(val_originals)} held-out prompts")

    results = {"val_prompts": len(val_originals), "n_per_prompt": args.n_per_prompt}

    # --- Training data baseline stats ---
    print("[eval] Computing training data baseline stats...")
    results["data_baseline"] = get_training_data_stats(val_pairs)

    # --- Generate from RWR model ---
    print(f"[eval] Generating from RWR model ({args.adapter_dir})...")
    t0 = time.time()
    rwr_model, tokenizer = load_generator(args.base_model, args.adapter_dir)
    rwr_generations = generate_paraphrases(
        rwr_model, tokenizer, val_originals,
        n_per_prompt=args.n_per_prompt, temperature=args.temperature,
        batch_size=args.gen_batch_size,
    )
    del rwr_model
    torch.cuda.empty_cache()
    print(f"  RWR generation done in {time.time() - t0:.0f}s")

    # --- Optionally generate from base model ---
    base_generations = None
    if args.eval_base_model:
        print(f"[eval] Generating from base model (no adapter)...")
        t0 = time.time()
        base_model, tokenizer = load_generator(args.base_model, adapter_dir=None)
        base_generations = generate_paraphrases(
            base_model, tokenizer, val_originals,
            n_per_prompt=args.n_per_prompt, temperature=args.temperature,
            batch_size=args.gen_batch_size,
        )
        del base_model
        torch.cuda.empty_cache()
        print(f"  Base generation done in {time.time() - t0:.0f}s")

    # --- Score everything ---
    print("[eval] Scoring RWR generations...")
    rwr_scored = score_generations(rwr_generations, args.refusal_vector_path, args.base_model)
    results["rwr"] = {
        "generations": rwr_scored,
        "stats": {
            "or_score_raw": compute_stats([s["or_score_raw"] for s in rwr_scored], "rwr_or_score"),
            "refusal_delta": compute_stats([s["refusal_delta"] for s in rwr_scored], "rwr_refusal_delta"),
            "similarity": compute_stats([s["similarity"] for s in rwr_scored], "rwr_similarity"),
        },
    }

    if base_generations:
        print("[eval] Scoring base model generations...")
        base_scored = score_generations(base_generations, args.refusal_vector_path, args.base_model)
        results["base_model"] = {
            "generations": base_scored,
            "stats": {
                "or_score_raw": compute_stats([s["or_score_raw"] for s in base_scored], "base_or_score"),
                "refusal_delta": compute_stats([s["refusal_delta"] for s in base_scored], "base_refusal_delta"),
                "similarity": compute_stats([s["similarity"] for s in base_scored], "base_similarity"),
            },
        }

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    _print_comparison(results)

    # --- Save ---
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] Results saved to {args.output}")


def _print_comparison(results):
    """Print a side-by-side comparison table."""
    sections = [("RWR Model", results.get("rwr", {}).get("stats", {}))]
    if "base_model" in results:
        sections.append(("Base Model", results["base_model"]["stats"]))
    sections.append(("Val Data", results.get("data_baseline", {})))

    for metric in ["or_score_raw", "refusal_delta", "similarity"]:
        print(f"\n  {metric}:")
        print(f"  {'':20s} {'mean':>8s} {'median':>8s} {'p90':>8s} {'p95':>8s} {'%pos':>8s}")
        for label, stats in sections:
            s = stats.get(metric, {})
            if not s:
                continue
            print(f"  {label:20s} {s['mean']:8.3f} {s['median']:8.3f} "
                  f"{s['p90']:8.3f} {s['p95']:8.3f} {s['pct_positive']:7.1%}")


if __name__ == "__main__":
    main()
