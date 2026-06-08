#!/usr/bin/env python3
"""Generate raw-Claude rewrites on the EXACT held-out alpaca prompts (Plan B head-to-head).

This builds the "raw Claude teacher" arm for a true head-to-head against the
distilled `claude_rwr` Llama student. Both are then compared on the same 200
held-out alpaca prompts, scored with the same ORRewardModel (k=5.0, c=0.75, d=100).

Why a dedicated script (vs iterate_prompts_claude.py): iterate_prompts_claude.py
generates on `seed=42` alpaca prompts, which are the *prompt-iteration* set — NOT
the held-out eval set. To be apples-to-apples with `claude_rwr`'s held-out numbers,
we must rewrite the identical prompts the held-out eval scored. The safest way to
guarantee identity is to read them straight out of the saved held-out results JSON
(`held_out_prompts.alpaca`), rather than reconstructing the seed=99 shuffle.

Default variant is `imitation_research_framing` — the SAME prompt that generated
`dataset_research_framing_full`, the data `claude_rwr` was trained on. That makes
this the true teacher distribution, so the comparison is "teacher rewrites vs the
student that distilled them". Switch with --variant_name / --variants_module.

Two-phase, mirroring iterate_prompts_claude.py:

  Phase 1 (login node, needs internet + ANTHROPIC_API_KEY):
    python generate_claude_heldout.py --skip_scoring
    # -> writes prompt_iteration_results/claude_heldout_alpaca_unscored.json
    #    in the iterate_prompts_claude schema, so phase-2 scoring is reusable as-is.

  Phase 2 (GPU slurm job, no internet): reuse the existing scorer unchanged:
    sbatch --export=ALL,OUTPUT_NAME=claude_heldout_alpaca.json run_score_claude.slurm
    # which runs: iterate_prompts_claude.py --skip_generation --output_name claude_heldout_alpaca.json

Then: python compare_heldout_claude.py  (prints the side-by-side table).
"""
import argparse
import importlib
import json
import os
import sys
import time
from typing import Dict, List, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, "..", "ppo_or"))

# Reuse the retry-wrapped Anthropic call from the existing pipeline.
from iterate_prompts_claude import call_claude_with_retry


def load_heldout_alpaca_prompts(held_out_results_path: str) -> List[Tuple[int, str]]:
    """Read the exact held-out alpaca prompts from a saved held-out eval results JSON.

    Guarantees identity with whatever `claude_rwr` was scored on (no seed/exclusion
    reconstruction needed). Returns (idx, prompt) tuples in stored order.
    """
    if not os.path.isfile(held_out_results_path):
        raise SystemExit(
            f"held-out results file not found: {held_out_results_path}\n"
            "Point --held_out_results at the JSON written by eval_held_out.py "
            "(or its rescored k=5.0 variant)."
        )
    with open(held_out_results_path) as f:
        d = json.load(f)
    hp = d.get("held_out_prompts", {}).get("alpaca")
    if not hp:
        raise SystemExit(
            f"no held_out_prompts.alpaca in {held_out_results_path}; "
            "this file does not carry the prompt set."
        )
    out = [(int(e["idx"]), e["prompt"]) for e in hp]
    print(f"[heldout] loaded {len(out)} held-out alpaca prompts from {os.path.basename(held_out_results_path)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--held_out_results",
                    default="prompt_iteration_results/held_out_eval/held_out_eval_results_k5.json",
                    help="Saved held-out eval JSON to pull the exact alpaca prompt set from.")
    ap.add_argument("--variants_module", default="claude_pilot_variants",
                    help="Module defining VARIANTS (claude_pilot_variants or claude_prompt_variants).")
    ap.add_argument("--variant_name", default="imitation_research_framing",
                    help="Variant to generate with. Default = the training-data teacher variant.")
    ap.add_argument("--claude_model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--output_dir", default="./prompt_iteration_results")
    ap.add_argument("--output_name", default="claude_heldout_alpaca.json",
                    help="Final scored filename (phase 2). Unscored intermediate is "
                         "<output_name>_unscored.json in output_dir.")
    ap.add_argument("--n_per_prompt", type=int, default=3,
                    help="Match the held-out eval (3 generations per prompt).")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=256,
                    help="Generation cap. NOTE: the Llama held-out arm used "
                         "max_new_tokens=64; Claude paraphrases of long prompts can run "
                         "longer. Kept higher to avoid truncating valid rewrites; the "
                         "asymmetry is a documented caveat in the comparison.")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only generate for the first N held-out prompts (smoke test).")
    ap.add_argument("--skip_scoring", action="store_true",
                    help="Generate only (no GPU). Score later via run_score_claude.slurm.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    unscored_path = os.path.join(args.output_dir, args.output_name.replace(".json", "_unscored.json"))
    scored_path = os.path.join(args.output_dir, args.output_name)

    variants_mod = importlib.import_module(args.variants_module)
    variant = next((v for v in variants_mod.VARIANTS if v.name == args.variant_name), None)
    if variant is None:
        avail = [v.name for v in variants_mod.VARIANTS]
        raise SystemExit(f"unknown variant '{args.variant_name}' in {args.variants_module}; available: {avail}")

    prompts = load_heldout_alpaca_prompts(args.held_out_results)
    if args.limit > 0:
        prompts = prompts[:args.limit]
        print(f"[heldout] --limit set: generating for first {len(prompts)} prompts only")

    # ---- Phase 1: generate ----
    from anthropic import Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set in environment")
    client = Anthropic(api_key=api_key)

    print(f"[gen] variant='{variant.name}' from {args.variants_module}  model={args.claude_model}")
    print(f"[gen] {len(prompts)} prompts x {args.n_per_prompt} = {len(prompts)*args.n_per_prompt} calls")
    t0 = time.time()
    gens: List[Dict] = []
    for i, (idx, p) in enumerate(prompts):
        for sample_idx in range(args.n_per_prompt):
            try:
                text = call_claude_with_retry(
                    client, args.claude_model, variant, p,
                    temperature=args.temperature, max_tokens=args.max_tokens,
                )
            except Exception as e:
                print(f"[gen]   prompt {i+1}/{len(prompts)} sample {sample_idx}: hard-fail {e}")
                text = ""
            gens.append({"original": p, "generation": text, "sample_idx": sample_idx, "alpaca_idx": idx})
        if (i + 1) % 10 == 0:
            n_empty = sum(1 for g in gens if not g["generation"])
            print(f"[gen]   {i+1}/{len(prompts)} prompts done ({time.time()-t0:.0f}s, {n_empty} empty so far)")
            # Incremental checkpoint so a crash/kill doesn't discard API spend.
            _write_unscored(unscored_path, variant, args, gens)

    _write_unscored(unscored_path, variant, args, gens)
    n_empty = sum(1 for g in gens if not g["generation"])
    print(f"[gen] done in {time.time()-t0:.0f}s; {len(gens)} generations ({n_empty} empty)")
    print(f"[gen] wrote {unscored_path}")

    if args.skip_scoring:
        print("[gen] --skip_scoring set. Score on a GPU node with:")
        print(f"      sbatch --export=ALL,OUTPUT_NAME={args.output_name} run_score_claude.slurm")
        return

    # ---- Phase 2: score (only if a GPU is actually present) ----
    print("[score] scoring inline (requires GPU + refusal vector). "
          "Prefer the slurm path if this node has no GPU.")
    from iterate_prompts_claude import phase2_score

    class _A:  # minimal shim for phase2_score's expected args
        refusal_vector_path = "../../refusal_vector/Vector_Extraction/refusal_vector.layer032.npz"
        scoring_model = "meta-llama/Meta-Llama-3-8B-Instruct"
        score_batch_size = 8
    phase2_score(_A(), unscored_path, scored_path)


def _write_unscored(unscored_path: str, variant, args, gens: List[Dict]) -> None:
    """Write in the iterate_prompts_claude schema so its phase-2 scorer works unchanged."""
    payload = {
        "n_prompts": len({g["alpaca_idx"] for g in gens}),
        "n_per_prompt": args.n_per_prompt,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "claude_model": args.claude_model,
        "source": "held_out_alpaca",
        "held_out_results": args.held_out_results,
        "variants": {
            variant.name: {
                "system_prompt": variant.system_prompt,
                "prompt_template": variant.prompt_template,
                "generations": gens,
            }
        },
    }
    with open(unscored_path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
