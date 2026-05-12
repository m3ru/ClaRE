#!/usr/bin/env python3
"""Evaluate the v1-dolly RWR model on multiple prompt sources.

Generates paraphrases from the trained adapter (and optionally the base
Meta-Llama-3-8B-Instruct), scores with ORRewardModel, writes per-source
JSON.  Supports two prompt sources:

    - dolly_val: held-out val split from the same dolly ranked.json the
      model trained on (uses load_shards/train_val_split with seed=42).
    - alpaca:    fresh out-of-distribution prompts from yahma/alpaca-cleaned.

Reuses utilities from eval_rwr.py (load_generator, generate_paraphrases,
score_generations, compute_stats).
"""
import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, "..", "ppo_or"))

from rwr_config import BinningConfig, DataConfig
from rwr_data import filter_and_bin, load_shards, train_val_split
from eval_rwr import (
    load_generator,
    generate_paraphrases,
    score_generations,
    compute_stats,
    get_training_data_stats,
)


def load_alpaca_prompts(n: int, seed: int = 42, hf_cache: str = None) -> List[str]:
    """Pull N prompts from yahma/alpaca-cleaned.

    Combines `instruction` (and `input` if non-empty) into a single string
    matching the way other datasets are treated in this repo.
    """
    from datasets import load_dataset
    kwargs = {}
    if hf_cache:
        kwargs["cache_dir"] = hf_cache
    ds = load_dataset("yahma/alpaca-cleaned", split="train", **kwargs)

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    prompts = []
    for idx in indices:
        ex = ds[idx]
        inst = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        if not inst:
            continue
        if inp:
            prompt = f"{inst}\n\n{inp}"
        else:
            prompt = inst
        # Skip absurdly long ones — keep eval tractable
        if len(prompt) > 1500:
            continue
        prompts.append(prompt)
        if len(prompts) >= n:
            break
    print(f"[alpaca] selected {len(prompts)} prompts")
    return prompts


def get_dolly_val_prompts(shard_dir: str, n: int, seed: int) -> (List[str], List[Dict]):
    data_config = DataConfig(shard_dir=shard_dir)
    binning_config = BinningConfig()
    pairs = load_shards(data_config)
    filtered, weights = filter_and_bin(pairs, binning_config)
    _, _, val_pairs, _ = train_val_split(filtered, weights, val_fraction=0.1, seed=seed)

    val_originals = list({p["original"] for p in val_pairs})
    if n and len(val_originals) > n:
        rng = random.Random(seed)
        val_originals = rng.sample(val_originals, n)
    print(f"[dolly_val] selected {len(val_originals)} held-out prompts")
    return val_originals, val_pairs


def run_one_source(
    label: str,
    prompts: List[str],
    rwr_adapter_dir: str,
    base_model_id: str,
    refusal_vector_path: str,
    n_per_prompt: int,
    temperature: float,
    output_path: str,
    eval_base_model: bool = True,
    val_pairs_for_baseline: List[Dict] = None,
    scoring_model_id: str = None,
):
    if scoring_model_id is None:
        scoring_model_id = base_model_id
    print(f"\n{'='*70}\n[{label}] starting eval — {len(prompts)} prompts\n{'='*70}")
    print(f"[{label}] generator={base_model_id}  scorer={scoring_model_id}")
    results = {
        "source": label,
        "n_prompts": len(prompts),
        "n_per_prompt": n_per_prompt,
        "temperature": temperature,
        "adapter": rwr_adapter_dir,
        "generator_model": base_model_id,
        "scoring_model": scoring_model_id,
    }

    if val_pairs_for_baseline is not None:
        results["data_baseline"] = get_training_data_stats(val_pairs_for_baseline)

    # --- Generate from RWR adapter ---
    t0 = time.time()
    print(f"[{label}] loading RWR generator (with adapter)...")
    rwr_model, tokenizer = load_generator(base_model_id, rwr_adapter_dir)
    rwr_gens = generate_paraphrases(
        rwr_model, tokenizer, prompts,
        n_per_prompt=n_per_prompt, temperature=temperature, batch_size=1,
    )
    del rwr_model
    torch.cuda.empty_cache()
    print(f"[{label}] RWR generation done in {time.time()-t0:.0f}s")

    # --- Generate from base ---
    base_gens = None
    if eval_base_model:
        t0 = time.time()
        print(f"[{label}] loading base generator (no adapter)...")
        base_model, tokenizer = load_generator(base_model_id, adapter_dir=None)
        base_gens = generate_paraphrases(
            base_model, tokenizer, prompts,
            n_per_prompt=n_per_prompt, temperature=temperature, batch_size=1,
        )
        del base_model
        torch.cuda.empty_cache()
        print(f"[{label}] base generation done in {time.time()-t0:.0f}s")

    # --- Score everything (using scoring_model_id, which may differ from generator) ---
    print(f"[{label}] scoring RWR generations with {scoring_model_id}...")
    rwr_scored = score_generations(rwr_gens, refusal_vector_path, scoring_model_id)
    results["rwr"] = {
        "generations": rwr_scored,
        "stats": {
            "or_score_raw": compute_stats([s["or_score_raw"] for s in rwr_scored], "rwr_or_score"),
            "refusal_delta": compute_stats([s["refusal_delta"] for s in rwr_scored], "rwr_refusal_delta"),
            "similarity": compute_stats([s["similarity"] for s in rwr_scored], "rwr_similarity"),
        },
    }

    if base_gens is not None:
        print(f"[{label}] scoring base generations with {scoring_model_id}...")
        base_scored = score_generations(base_gens, refusal_vector_path, scoring_model_id)
        results["base_model"] = {
            "generations": base_scored,
            "stats": {
                "or_score_raw": compute_stats([s["or_score_raw"] for s in base_scored], "base_or_score"),
                "refusal_delta": compute_stats([s["refusal_delta"] for s in base_scored], "base_refusal_delta"),
                "similarity": compute_stats([s["similarity"] for s in base_scored], "base_similarity"),
            },
        }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{label}] wrote {output_path}")
    print_summary(label, results)
    return results


def print_summary(label, results):
    sections = [("RWR Model", results.get("rwr", {}).get("stats", {}))]
    if "base_model" in results:
        sections.append(("Base Model", results["base_model"]["stats"]))
    if "data_baseline" in results:
        sections.append(("Val Data", results["data_baseline"]))

    print(f"\n--- {label} summary ---")
    for metric in ["or_score_raw", "refusal_delta", "similarity"]:
        print(f"\n  {metric}:")
        print(f"  {'':20s} {'mean':>8s} {'median':>8s} {'p90':>8s} {'%pos':>8s}")
        for name, stats in sections:
            s = stats.get(metric, {})
            if not s:
                continue
            print(f"  {name:20s} {s['mean']:8.3f} {s['median']:8.3f} "
                  f"{s['p90']:8.3f} {s['pct_positive']:7.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--shard_dir", required=True,
                    help="Directory with the staged dolly ranked.json (or_susceptibility_rankings_shard0.json)")
    ap.add_argument("--refusal_vector_path", required=True)
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                    help="Generator model — the LLM the adapter sits on top of.")
    ap.add_argument("--scoring_model", default=None,
                    help="Scoring model — the model whose activations are projected "
                         "onto the refusal vector. Defaults to --base_model. Use a different "
                         "value (e.g. meta-llama/Llama-Guard-3-8B) when the refusal vector was "
                         "extracted from a different model than the one generating.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_per_prompt", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_eval_prompts", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_base_eval", action="store_true")
    ap.add_argument("--sources", nargs="+", default=["dolly_val", "alpaca"],
                    help="Prompt sources to evaluate against")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if "dolly_val" in args.sources:
        prompts, val_pairs = get_dolly_val_prompts(args.shard_dir, args.max_eval_prompts, args.seed)
        run_one_source(
            "dolly_val", prompts,
            args.adapter_dir, args.base_model, args.refusal_vector_path,
            args.n_per_prompt, args.temperature,
            os.path.join(args.output_dir, "eval_dolly_val.json"),
            eval_base_model=not args.no_base_eval,
            val_pairs_for_baseline=val_pairs,
            scoring_model_id=args.scoring_model,
        )

    if "alpaca" in args.sources:
        prompts = load_alpaca_prompts(args.max_eval_prompts, args.seed)
        run_one_source(
            "alpaca", prompts,
            args.adapter_dir, args.base_model, args.refusal_vector_path,
            args.n_per_prompt, args.temperature,
            os.path.join(args.output_dir, "eval_alpaca.json"),
            eval_base_model=not args.no_base_eval,
            scoring_model_id=args.scoring_model,
        )


if __name__ == "__main__":
    main()
