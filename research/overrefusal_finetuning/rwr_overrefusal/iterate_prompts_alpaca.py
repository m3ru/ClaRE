#!/usr/bin/env python3
"""Iterate (system_prompt, prompt_template) variants on baseline Llama-3-8B-Instruct,
generate overrefusal-targeted rewrites of alpaca-cleaned prompts, and rank variants
by OR score (which combines refusal_delta and similarity).

Loads the variants defined in prompt_variants.py. The base model and reward model
are each loaded exactly once and reused across variants.

Scoring uses the v3-consistent formula matched to BinningConfig:
    or_score_raw = exp(k * (similarity - c)) * refusal_delta / d   with k=5.0, c=0.75, d=100

Usage:
    python iterate_prompts_alpaca.py \\
        --refusal_vector_path ../../refusal_vector/Vector_Extraction/refusal_vector.layer032.npz \\
        --output_dir ./prompt_iteration_results \\
        --n_prompts 20 --n_per_prompt 3
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, "..", "ppo_or"))

from eval_rwr import load_generator
from eval_v1_dolly import load_alpaca_prompts
from rwr_config import BinningConfig
from prompt_variants import VARIANTS, PromptVariant, BASELINE_REFERENCE


def generate_for_variant(
    model, tokenizer, prompts: List[str], variant: PromptVariant,
    n_per_prompt: int, temperature: float, max_new_tokens: int,
) -> Dict[str, List[str]]:
    """Generate n_per_prompt rewrites per prompt using the variant's system/template."""
    if "{prompt}" not in variant.prompt_template:
        raise ValueError(f"variant '{variant.name}' prompt_template must contain '{{prompt}}'")

    results: Dict[str, List[str]] = {}
    for i, p in enumerate(prompts):
        user_content = variant.prompt_template.format(prompt=p)
        messages = [
            {"role": "system", "content": variant.system_prompt},
            {"role": "user", "content": user_content},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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
        gens = [tokenizer.decode(ids[prompt_len:], skip_special_tokens=True).strip()
                for ids in output_ids]
        results[p] = gens

        if (i + 1) % 10 == 0:
            print(f"  [{variant.name}] {i+1}/{len(prompts)} prompts done")
    return results


def score_variant(
    generations: Dict[str, List[str]], reward_model, batch_size: int,
    k: float, c: float, d: float,
) -> List[Dict]:
    """Score all (original, rewrite) pairs with v3-consistent OR scoring (unclamped)."""
    flat_orig, flat_gen = [], []
    for orig, gens in generations.items():
        for g in gens:
            flat_orig.append(orig)
            flat_gen.append(g)

    scored: List[Dict] = []
    for i in range(0, len(flat_orig), batch_size):
        ob = flat_orig[i:i + batch_size]
        gb = flat_gen[i:i + batch_size]
        out = reward_model.compute_rewards(ob, gb)
        sim_weight = torch.exp(k * (out["similarities"] - c))
        or_raw = (sim_weight * out["refusal_delta"] / d).cpu().numpy()
        sims = out["similarities"].cpu().numpy()
        deltas = out["refusal_delta"].cpu().numpy()
        for j in range(len(ob)):
            scored.append({
                "original": ob[j],
                "generation": gb[j],
                "refusal_delta": float(deltas[j]),
                "similarity": float(sims[j]),
                "or_score_raw": float(or_raw[j]),
            })
    return scored


def compute_stats(scored: List[Dict]) -> Dict[str, Dict]:
    def _s(values: List[float]) -> Dict:
        arr = np.array(values)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "pct_positive": float((arr > 0).mean()),
        }
    return {
        "or_score_raw": _s([s["or_score_raw"] for s in scored]),
        "refusal_delta": _s([s["refusal_delta"] for s in scored]),
        "similarity": _s([s["similarity"] for s in scored]),
    }


def print_comparison(per_variant: Dict[str, Dict]):
    print("\n" + "=" * 88)
    print("PROMPT ITERATION SUMMARY (sorted by metric mean, descending)")
    print("=" * 88)
    for metric in ["or_score_raw", "refusal_delta", "similarity"]:
        print(f"\n  {metric}:")
        print(f"  {'variant':30s} {'mean':>10s} {'median':>10s} {'p90':>10s} {'%pos':>9s}")
        items = sorted(per_variant.items(),
                       key=lambda kv: kv[1]["stats"][metric]["mean"],
                       reverse=True)
        for name, info in items:
            s = info["stats"][metric]
            print(f"  {name:30s} {s['mean']:10.3f} {s['median']:10.3f} "
                  f"{s['p90']:10.3f} {s['pct_positive']:8.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refusal_vector_path", required=True)
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--output_dir", default="./prompt_iteration_results")
    ap.add_argument("--n_prompts", type=int, default=20,
                    help="Number of alpaca-cleaned prompts to sample")
    ap.add_argument("--n_per_prompt", type=int, default=3,
                    help="Number of generations per prompt")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--score_batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_baseline", action="store_true",
                    help="Skip the BASELINE_REFERENCE control variant.")
    ap.add_argument("--variant_names", nargs="+", default=None,
                    help="If set, only run variants whose PromptVariant.name is in this list "
                         "(applied AFTER the baseline injection, so it can filter the baseline too).")
    ap.add_argument("--output_name", default="iterate_alpaca_results.json",
                    help="Filename for the per-variant results JSON")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Variant list ---
    variants: List[PromptVariant] = list(VARIANTS)
    if not args.no_baseline:
        variants = [BASELINE_REFERENCE] + variants
    if args.variant_names:
        requested = set(args.variant_names)
        kept = [v for v in variants if v.name in requested]
        missing = requested - {v.name for v in kept}
        if missing:
            print(f"[warn] requested variant_names not found in prompt_variants.py: {sorted(missing)}")
        variants = kept
    if not variants:
        raise SystemExit("No variants to run. Fill in prompt_variants.py:VARIANTS or check --variant_names.")
    for v in variants:
        if v.system_prompt.strip().startswith("TODO"):
            print(f"[warn] variant '{v.name}' still has a TODO system prompt")
    print(f"[iter] running {len(variants)} variants: {[v.name for v in variants]}")

    # --- Alpaca prompts ---
    prompts = load_alpaca_prompts(args.n_prompts, args.seed)

    # --- Generate (load model once, reuse across variants) ---
    print(f"[iter] loading base generator {args.base_model}")
    model, tokenizer = load_generator(args.base_model, adapter_dir=None)

    variant_generations: Dict[str, Dict[str, List[str]]] = {}
    for v in variants:
        print(f"\n[iter] generating for variant '{v.name}'")
        t0 = time.time()
        variant_generations[v.name] = generate_for_variant(
            model, tokenizer, prompts, v,
            n_per_prompt=args.n_per_prompt,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"[iter] '{v.name}' done in {time.time()-t0:.0f}s")

    del model
    torch.cuda.empty_cache()

    # --- Score (load reward model once) ---
    from config import ModelConfig as PPOModelConfig, RewardConfig
    from reward_model import ORRewardModel

    binning = BinningConfig()  # k=5.0, c=0.75, d=100
    pmc = PPOModelConfig(
        base_model_id=args.base_model,
        refusal_vector_path=args.refusal_vector_path,
        activation_layer=32,
    )
    rc = RewardConfig()
    rc.similarity_exponent = binning.similarity_exponent
    rc.similarity_center = binning.similarity_center
    rc.refusal_divisor = binning.refusal_divisor
    reward_model = ORRewardModel(pmc, rc)

    per_variant: Dict[str, Dict] = {}
    for v in variants:
        print(f"\n[iter] scoring variant '{v.name}'")
        t0 = time.time()
        scored = score_variant(
            variant_generations[v.name], reward_model,
            batch_size=args.score_batch_size,
            k=binning.similarity_exponent,
            c=binning.similarity_center,
            d=binning.refusal_divisor,
        )
        per_variant[v.name] = {
            "system_prompt": v.system_prompt,
            "prompt_template": v.prompt_template,
            "generations": scored,
            "stats": compute_stats(scored),
        }
        print(f"[iter] '{v.name}' scored in {time.time()-t0:.0f}s")

    reward_model.cleanup()

    # --- Save + summarize ---
    results = {
        "n_prompts": args.n_prompts,
        "n_per_prompt": args.n_per_prompt,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "base_model": args.base_model,
        "refusal_vector_path": args.refusal_vector_path,
        "scoring_formula": {
            "k_similarity_exponent": binning.similarity_exponent,
            "c_similarity_center": binning.similarity_center,
            "d_refusal_divisor": binning.refusal_divisor,
        },
        "variants": per_variant,
    }
    out_path = os.path.join(args.output_dir, args.output_name)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[iter] wrote {out_path}")

    print_comparison(per_variant)


if __name__ == "__main__":
    main()
