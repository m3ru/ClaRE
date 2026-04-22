#!/usr/bin/env python3
"""Generate paraphrases of Dolly prompts and rank them by OR susceptibility.

For each prompt:
1. Generate N paraphrases using gpt-oss-120b (plain paraphrasing, not OR)
2. Compute OR score for each (original, paraphrase) pair
3. Rank prompts by mean + std of OR scores across their paraphrases

Two-phase design: generation finishes and caches before scoring begins,
so each phase only loads one model at a time and can resume from checkpoint.

Usage:
    python paraphrase_and_score.py \\
        --data_path dolly15k.jsonl \\
        --refusal_vector_path refusal_vector.layer032.npz \\
        --output or_rankings.json \\
        --max_prompts 500 --n_paraphrases 20
"""

import argparse
import gc
import json
import math
import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PARAPHRASE_SYSTEM_PROMPT = (
    "You are a paraphrasing tool. Given a user prompt, output ONLY a single "
    "paraphrased version that preserves the original meaning but uses different "
    "wording. Do NOT include any analysis, reasoning, thinking, commentary, "
    "labels, or explanation. Do NOT repeat the instructions. Output ONLY the "
    "paraphrased text and nothing else."
)

# Patterns that indicate the model leaked reasoning into the output
_REASONING_PREFIXES = [
    "analysis",
    "reasoning",
    "thinking",
    "the user",
    "the task",
    "we need to",
    "we should",
    "so we",
    "let me",
    "i need to",
    "paraphrase:",
    "paraphrased version:",
    "output:",
    "result:",
    "here is",
    "here's",
]


def _clean_paraphrase(text: str, original: str) -> str:
    """Strip reasoning preamble that gpt-oss sometimes prepends."""
    import re

    stripped = text.strip()
    if not stripped:
        return ""

    # If text starts with a known reasoning prefix, try to extract the
    # actual paraphrase that usually follows after a newline or quote.
    lower = stripped.lower()
    has_prefix = any(lower.startswith(p) for p in _REASONING_PREFIXES)

    if has_prefix:
        # Strategy 1: find text in quotes
        quoted = re.findall(r'"([^"]{15,})"', stripped)
        if quoted:
            best = max(quoted, key=len)
            return best.strip()

        # Strategy 2: take the last paragraph (reasoning is usually first)
        paragraphs = [p.strip() for p in stripped.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            candidate = paragraphs[-1]
            # Make sure it's not also reasoning
            c_lower = candidate.lower()
            if not any(c_lower.startswith(p) for p in _REASONING_PREFIXES):
                return candidate

        # Strategy 3: take last line
        lines = [ln.strip() for ln in stripped.split("\n") if ln.strip()]
        if len(lines) > 1:
            candidate = lines[-1]
            c_lower = candidate.lower()
            if not any(c_lower.startswith(p) for p in _REASONING_PREFIXES):
                return candidate

        # Strategy 4: find a sentence that looks like a paraphrase
        # (contains some overlap with original words)
        orig_words = set(original.lower().split())
        sentences = re.split(r'(?<=[.?!])\s+', stripped)
        for sent in reversed(sentences):
            sent_words = set(sent.lower().split())
            overlap = len(orig_words & sent_words) / max(len(orig_words), 1)
            if overlap > 0.15 and len(sent.split()) > 5:
                return sent.strip()

    return stripped


def load_prompts(path: str, max_prompts: int = 0) -> List[str]:
    from data import load_jsonl, load_txt, load_csv
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        data = load_jsonl(path)
    elif ext == ".txt":
        data = load_txt(path)
    elif ext == ".csv":
        data = load_csv(path)
    else:
        data = load_jsonl(path)
    prompts = [d["prompt"] for d in data]
    if max_prompts > 0:
        prompts = prompts[:max_prompts]
    return prompts


def load_generation_pipeline(model_id: str):
    """Load gpt-oss via the Transformers pipeline (handles harmony format)."""
    from transformers import pipeline as hf_pipeline

    print(f"[model] Loading {model_id} for paraphrase generation...")
    pipe = hf_pipeline(
        "text-generation",
        model=model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    print("[model] Generation model loaded")
    return pipe


def generate_paraphrases(
    pipe,
    prompt: str,
    n: int = 20,
    temperature: float = 0.9,
    max_new_tokens: int = 512,
    top_p: float = 0.95,
    batch_size: int = 5,
) -> List[str]:
    messages = [
        {"role": "developer", "content": PARAPHRASE_SYSTEM_PROMPT},
        {"role": "user", "content": f'Please paraphrase the following: "{prompt}"'},
    ]

    paraphrases = []
    remaining = n
    while remaining > 0:
        k = min(remaining, batch_size)
        try:
            outputs = pipe(
                messages,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=50,
                return_full_text=True,
                num_return_sequences=k,
            )
            for out in outputs:
                assistant_msg = out["generated_text"][-1]
                raw = assistant_msg.get("content", "").strip()
                text = _clean_paraphrase(raw, prompt)
                if text and len(text.split()) >= 3:
                    paraphrases.append(text)
        except Exception as e:
            print(f"  [gen] Error generating paraphrase batch: {e}")
        remaining -= k
    return paraphrases


def score_paraphrases(
    reward_model,
    original: str,
    paraphrases: List[str],
    batch_size: int = 8,
) -> List[Dict]:
    """Compute raw (unclamped) OR scores for (original, paraphrase) pairs."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Paraphrase Dolly prompts and rank by OR susceptibility")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--refusal_vector_path", type=str, required=True)
    parser.add_argument("--output", type=str,
                        default="or_susceptibility_rankings.json")
    parser.add_argument("--gen_model", type=str,
                        default="openai/gpt-oss-120b",
                        help="Model for paraphrase generation")
    parser.add_argument("--scoring_model", type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Model for refusal scoring")
    parser.add_argument("--n_paraphrases", type=int, default=20)
    parser.add_argument("--max_prompts", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--activation_layer", type=int, default=32)
    parser.add_argument("--score_batch_size", type=int, default=8)
    parser.add_argument("--checkpoint_every", type=int, default=50)
    parser.add_argument("--shard_id", type=int, default=0,
                        help="This shard's index (0-based)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of parallel shards")
    parser.add_argument("--gen_batch_size", type=int, default=5,
                        help="num_return_sequences per pipeline call")
    args = parser.parse_args()

    all_prompts = load_prompts(args.data_path, args.max_prompts)
    prompts = all_prompts[args.shard_id::args.num_shards]
    prompt_indices = list(range(args.shard_id, len(all_prompts), args.num_shards))
    print(f"[data] Loaded {len(all_prompts)} prompts total, "
          f"shard {args.shard_id}/{args.num_shards} → {len(prompts)} prompts")

    # ==================================================================
    # Phase 1: Generate paraphrases (caches to disk, resumable)
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"Phase 1 — Generate paraphrases with {args.gen_model} "
          f"(shard {args.shard_id}/{args.num_shards})")
    print("=" * 60)

    shard_suffix = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    paraphrase_cache = args.output.replace(
        ".json", f"_paraphrases{shard_suffix}.json")
    shard_output = args.output.replace(".json", f"{shard_suffix}.json")

    if os.path.exists(paraphrase_cache):
        with open(paraphrase_cache, "r") as f:
            cached = json.load(f)
        start_idx = len(cached)
        print(f"[phase1] Resuming from prompt {start_idx}/{len(prompts)}")
    else:
        cached = []
        start_idx = 0

    if start_idx < len(prompts):
        pipe = load_generation_pipeline(args.gen_model)

        for i in range(start_idx, len(prompts)):
            prompt = prompts[i]
            global_idx = prompt_indices[i]
            paraphrases = generate_paraphrases(
                pipe, prompt,
                n=args.n_paraphrases, temperature=args.temperature,
                batch_size=args.gen_batch_size,
            )
            cached.append({
                "prompt_idx": global_idx,
                "original": prompt,
                "paraphrases_text": paraphrases,
            })

            if (i + 1) % 5 == 0 or i == 0:
                print(f"[phase1] {i+1}/{len(prompts)} — "
                      f"{len(paraphrases)} paraphrases for: {prompt[:80]}")

            if (i + 1) % args.checkpoint_every == 0:
                with open(paraphrase_cache, "w") as f:
                    json.dump(cached, f)
                print(f"[phase1] Checkpoint saved ({i+1} prompts)")

        with open(paraphrase_cache, "w") as f:
            json.dump(cached, f)
        print(f"[phase1] Done — saved to {paraphrase_cache}")

        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[model] Generation model freed")
    else:
        print("[phase1] All paraphrases already cached — skipping generation")

    # ==================================================================
    # Phase 2: Score every (original, paraphrase) pair
    # ==================================================================
    print("\n" + "=" * 60)
    print("Phase 2 — Score paraphrases")
    print("=" * 60)

    from config import ModelConfig, RewardConfig
    from reward_model import ORRewardModel

    model_config = ModelConfig(
        base_model_id=args.scoring_model,
        refusal_vector_path=args.refusal_vector_path,
        activation_layer=args.activation_layer,
    )
    reward_config = RewardConfig()
    reward_model = ORRewardModel(model_config, reward_config)

    all_results = []
    for i, entry in enumerate(cached):
        original = entry["original"]
        paraphrases = entry["paraphrases_text"]

        if not paraphrases:
            all_results.append({
                "prompt_idx": entry["prompt_idx"],
                "original": original,
                "paraphrases": [],
            })
            continue

        scores = score_paraphrases(
            reward_model, original, paraphrases,
            batch_size=args.score_batch_size,
        )
        all_results.append({
            "prompt_idx": entry["prompt_idx"],
            "original": original,
            "paraphrases": scores,
        })

        if (i + 1) % 10 == 0:
            or_vals = [s["or_score_raw"] for s in scores]
            print(f"[phase2] {i+1}/{len(cached)} — "
                  f"mean_or={np.mean(or_vals):.4f} std={np.std(or_vals):.4f} "
                  f"— {original[:80]}")

    reward_model.cleanup()
    print("[phase2] Done — reward model freed")

    # ==================================================================
    # Phase 3: Rank and save
    # ==================================================================
    print("\n" + "=" * 60)
    print("Phase 3 — Rank prompts")
    print("=" * 60)

    ranked = rank_prompts(all_results)

    with open(shard_output, "w") as f:
        json.dump(ranked, f, indent=2)
    print(f"[done] Shard results saved to {shard_output}")

    if args.num_shards > 1:
        print(f"[done] This was shard {args.shard_id}/{args.num_shards}. "
              f"Run merge_shards.py to combine all shard outputs.")
    else:
        _print_summary(ranked)


def _print_summary(ranked: List[Dict]):
    print("\n" + "=" * 80)
    print("TOP 30 OR-SUSCEPTIBLE PROMPTS  (ranked by mean + std)")
    print("=" * 80)
    for rank, entry in enumerate(ranked[:30], 1):
        print(f"\n#{rank}  rank_score={entry['rank_score']:.4f}  "
              f"mean={entry['mean_or']:.4f}  std={entry['std_or']:.4f}  "
              f"max={entry['max_or']:.4f}  mean_sim={entry['mean_sim']:.3f}  "
              f"pos={entry['n_positive']}/{entry['n_paraphrases']}")
        print(f"  Prompt: {entry['original'][:200]}")
        top_p = sorted(entry["paraphrases"],
                       key=lambda x: x["or_score_raw"], reverse=True)
        for j, p in enumerate(top_p[:3], 1):
            print(f"  Top-{j}: or={p['or_score_raw']:.4f}  "
                  f"sim={p['similarity']:.3f}  delta={p['refusal_delta']:.3f}")
            print(f"          {p['paraphrase'][:180]}")

    print("\n" + "=" * 80)
    print("BOTTOM 10 (least OR-susceptible)")
    print("=" * 80)
    for rank, entry in enumerate(ranked[-10:], len(ranked) - 9):
        print(f"\n#{rank}  rank_score={entry['rank_score']:.4f}  "
              f"mean={entry['mean_or']:.4f}  std={entry['std_or']:.4f}")
        print(f"  Prompt: {entry['original'][:200]}")


if __name__ == "__main__":
    main()
