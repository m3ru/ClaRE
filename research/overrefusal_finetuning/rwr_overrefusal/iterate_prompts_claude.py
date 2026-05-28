#!/usr/bin/env python3
"""Generate prompt rewrites via Anthropic API and score them with ORRewardModel.

Two-phase pipeline:

  Phase 1 (no GPU, needs internet+API key):
    Generate rewrites via Anthropic API and save to ``<output_name>_unscored.json``.

  Phase 2 (GPU, no internet needed):
    Load the unscored JSON, score every (original, rewrite) pair with the
    activation-based ORRewardModel, save to ``<output_name>``.

Skip flags let you run either phase in isolation — useful because compute nodes
on Compute Canada typically can't reach the Anthropic API, so generation happens
on a login node and scoring on a slurm job:

    # On login node (has internet)
    ANTHROPIC_API_KEY=sk-... python iterate_prompts_claude.py \\
        --skip_scoring --output_name claude_run1.json

    # As a slurm job (has GPU, no internet)
    sbatch run_score_claude.slurm   # internally calls iterate_prompts_claude.py
                                    # with --skip_generation --output_name claude_run1.json
"""
import argparse
import json
import os
import sys
import statistics
import time
from typing import Dict, List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, "..", "ppo_or"))

from claude_prompt_variants import VARIANTS, PromptVariant


def load_alpaca_prompts_local(n: int, seed: int = 42) -> List[str]:
    """Same alpaca-cleaned loader used by iterate_prompts_alpaca.py / eval_v1_dolly.py.

    Inlined here to avoid pulling in eval_v1_dolly which imports the GPU pipeline.
    """
    import random
    from datasets import load_dataset
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    prompts: List[str] = []
    for idx in indices:
        ex = ds[idx]
        inst = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        if not inst:
            continue
        prompt = f"{inst}\n\n{inp}" if inp else inst
        if len(prompt) > 1500:
            continue
        prompts.append(prompt)
        if len(prompts) >= n:
            break
    print(f"[alpaca] selected {len(prompts)} prompts")
    return prompts


def call_claude_with_retry(client, model_id: str, variant: PromptVariant, user_prompt: str,
                           temperature: float, max_tokens: int, retries: int = 4,
                           base_sleep: float = 2.0) -> str:
    """Call Anthropic messages API with exponential backoff on transient errors."""
    user_content = variant.prompt_template.format(prompt=user_prompt)
    attempt = 0
    while True:
        try:
            resp = client.messages.create(
                model=model_id,
                system=variant.system_prompt,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return "".join(block.text for block in resp.content if hasattr(block, "text")).strip()
        except Exception as e:
            attempt += 1
            if attempt > retries:
                raise
            name = e.__class__.__name__.lower()
            msg = str(e).lower()
            is_rate = ("ratelimit" in name) or ("rate_limit" in msg) or ("429" in msg)
            sleep_s = (5.0 if is_rate else base_sleep) * (2 ** (attempt - 1))
            sleep_s = min(sleep_s, 60.0)
            print(f"[gen]   retry {attempt}/{retries} after {e.__class__.__name__}: sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)


def phase1_generate(args, variants: List[PromptVariant], unscored_path: str) -> None:
    """Call Anthropic API to produce rewrites for every (variant, prompt, sample) tuple."""
    from anthropic import Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set in environment")
    client = Anthropic(api_key=api_key)

    prompts = load_alpaca_prompts_local(args.n_prompts, args.seed)

    all_data: Dict = {
        "n_prompts":     args.n_prompts,
        "n_per_prompt":  args.n_per_prompt,
        "temperature":   args.temperature,
        "max_tokens":    args.max_tokens,
        "claude_model":  args.claude_model,
        "seed":          args.seed,
        "variants":      {},
    }

    for v in variants:
        print(f"\n[gen] variant '{v.name}'")
        t0 = time.time()
        per_variant_gens: List[Dict] = []
        for i, p in enumerate(prompts):
            for sample_idx in range(args.n_per_prompt):
                try:
                    text = call_claude_with_retry(
                        client, args.claude_model, v, p,
                        temperature=args.temperature, max_tokens=args.max_tokens,
                    )
                except Exception as e:
                    print(f"[gen]   prompt {i+1}/{len(prompts)} sample {sample_idx}: hard-fail {e}")
                    text = ""
                per_variant_gens.append({"original": p, "generation": text, "sample_idx": sample_idx})
            if (i + 1) % 5 == 0:
                print(f"[gen]   {i+1}/{len(prompts)} prompts done ({time.time()-t0:.0f}s)")
        all_data["variants"][v.name] = {
            "system_prompt":   v.system_prompt,
            "prompt_template": v.prompt_template,
            "generations":     per_variant_gens,
        }
        elapsed = time.time() - t0
        n_empty = sum(1 for g in per_variant_gens if not g["generation"])
        print(f"[gen] '{v.name}' done in {elapsed:.0f}s ({n_empty} empty)")
        # Write incrementally after each variant so a crash or kill doesn't
        # discard hours of API calls.
        with open(unscored_path, "w") as f:
            json.dump(all_data, f, indent=2)
        print(f"[gen] checkpoint: {unscored_path}")

    print(f"\n[gen] wrote unscored generations to {unscored_path}")


def _percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    rank = (pct / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def _stats(values: List[float]) -> Dict:
    s = sorted(values)
    return {
        "mean":         statistics.fmean(values),
        "median":       statistics.median(values),
        "p75":          _percentile(s, 75),
        "p90":          _percentile(s, 90),
        "p95":          _percentile(s, 95),
        "pct_positive": sum(1 for v in values if v > 0) / len(values),
    }


def phase2_score(args, unscored_path: str, scored_path: str) -> None:
    """Load unscored JSON, score every (orig, rewrite) pair with ORRewardModel."""
    if not os.path.isfile(unscored_path):
        raise SystemExit(f"unscored generations file missing: {unscored_path}")

    with open(unscored_path) as f:
        all_data = json.load(f)

    import torch
    from rwr_config import BinningConfig
    from config import ModelConfig as PPOModelConfig, RewardConfig
    from reward_model import ORRewardModel

    binning = BinningConfig()
    pmc = PPOModelConfig(
        base_model_id=args.scoring_model,
        refusal_vector_path=args.refusal_vector_path,
        activation_layer=32,
    )
    rc = RewardConfig()
    rc.similarity_exponent = binning.similarity_exponent
    rc.similarity_center   = binning.similarity_center
    rc.refusal_divisor     = binning.refusal_divisor
    reward_model = ORRewardModel(pmc, rc)

    for vname, vdata in all_data["variants"].items():
        gens = [g for g in vdata["generations"] if g["generation"]]  # drop empties
        if not gens:
            print(f"[score] {vname}: no non-empty generations, skipping")
            continue
        print(f"[score] {vname}: {len(gens)} pairs")
        t0 = time.time()
        flat_orig = [g["original"]   for g in gens]
        flat_gen  = [g["generation"] for g in gens]
        for i in range(0, len(flat_orig), args.score_batch_size):
            ob = flat_orig[i:i + args.score_batch_size]
            gb = flat_gen [i:i + args.score_batch_size]
            out = reward_model.compute_rewards(ob, gb)
            sim_weight = torch.exp(
                binning.similarity_exponent * (out["similarities"] - binning.similarity_center)
            )
            or_raw = (sim_weight * out["refusal_delta"] / binning.refusal_divisor).cpu().numpy()
            sims   = out["similarities"].cpu().numpy()
            deltas = out["refusal_delta"].cpu().numpy()
            for j in range(len(ob)):
                gens[i + j]["refusal_delta"] = float(deltas[j])
                gens[i + j]["similarity"]    = float(sims[j])
                gens[i + j]["or_score_raw"]  = float(or_raw[j])
        vdata["generations"] = gens  # replace, dropping empties
        vdata["stats"] = {
            "or_score_raw":  _stats([g["or_score_raw"]  for g in gens]),
            "refusal_delta": _stats([g["refusal_delta"] for g in gens]),
            "similarity":    _stats([g["similarity"]    for g in gens]),
        }
        print(f"[score] {vname} done in {time.time()-t0:.0f}s")

    reward_model.cleanup()

    all_data["scoring_formula"] = {
        "k_similarity_exponent": binning.similarity_exponent,
        "c_similarity_center":   binning.similarity_center,
        "d_refusal_divisor":     binning.refusal_divisor,
    }
    all_data["refusal_vector_path"] = args.refusal_vector_path
    all_data["scoring_model"]       = args.scoring_model

    with open(scored_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\n[score] wrote scored results to {scored_path}")

    # Summary, sorted by p90 (top-decile) not mean, per the project's RWR-bin philosophy.
    print("\n" + "=" * 92)
    print("CLAUDE PROMPT ITERATION SUMMARY (sorted by p90)")
    print("=" * 92)
    for metric in ["or_score_raw", "refusal_delta", "similarity"]:
        print(f"\n  {metric}:")
        print(f"  {'variant':28s} {'n':>4s} {'mean':>9s} {'median':>9s} {'p75':>9s} {'p90':>9s} {'p95':>9s} {'%pos':>8s}")
        items = sorted(all_data["variants"].items(),
                       key=lambda kv: -kv[1].get("stats", {}).get(metric, {}).get("p90", 0))
        for name, info in items:
            s = info.get("stats", {}).get(metric)
            if not s:
                continue
            n = len(info["generations"])
            print(f"  {name:28s} {n:4d} {s['mean']:9.3f} {s['median']:9.3f} "
                  f"{s['p75']:9.3f} {s['p90']:9.3f} {s['p95']:9.3f} {s['pct_positive']:7.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refusal_vector_path", default="../../refusal_vector/Vector_Extraction/refusal_vector.layer032.npz")
    ap.add_argument("--scoring_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--claude_model", default="claude-haiku-4-5-20251001",
                    help="Anthropic model id. Defaults to Haiku 4.5 (matches existing taskaware_5k pipeline). "
                         "Try claude-sonnet-4-6 or claude-opus-4-7 for higher capability.")
    ap.add_argument("--output_dir", default="./prompt_iteration_results")
    ap.add_argument("--output_name", default="claude_results.json",
                    help="Final scored JSON filename. Unscored intermediate is "
                         "saved at <output_name>_unscored.json in the same dir.")
    ap.add_argument("--n_prompts", type=int, default=20)
    ap.add_argument("--n_per_prompt", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=200)
    ap.add_argument("--score_batch_size", type=int, default=8)
    ap.add_argument("--variant_names", nargs="+", default=None,
                    help="If set, only run variants whose PromptVariant.name is in this list.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_generation", action="store_true",
                    help="Skip Anthropic API calls; load existing <output_name>_unscored.json.")
    ap.add_argument("--skip_scoring", action="store_true",
                    help="Generate only; don't score (e.g. when running on a node without a GPU).")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    unscored_path = os.path.join(args.output_dir, args.output_name.replace(".json", "_unscored.json"))
    scored_path   = os.path.join(args.output_dir, args.output_name)

    variants: List[PromptVariant] = list(VARIANTS)
    if args.variant_names:
        requested = set(args.variant_names)
        kept = [v for v in variants if v.name in requested]
        missing = requested - {v.name for v in kept}
        if missing:
            print(f"[warn] requested variant_names not found: {sorted(missing)}")
        variants = kept
    if not variants:
        raise SystemExit("No variants to run.")
    print(f"[iter] {len(variants)} variants: {[v.name for v in variants]}")
    print(f"[iter] claude_model={args.claude_model}  scoring_model={args.scoring_model}")

    if not args.skip_generation:
        phase1_generate(args, variants, unscored_path)
    if not args.skip_scoring:
        phase2_score(args, unscored_path, scored_path)


if __name__ == "__main__":
    main()
