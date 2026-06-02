#!/usr/bin/env python3
"""Held-out eval for the Claude→Llama RWR distillation experiment (Plan B).

Compares one or more trained RWR adapters against the baseline Llama-3-8B-Instruct
on prompts that are *explicitly disjoint* from everything we've used so far:
the 2500 alpaca + 2500 dolly seeds used for training, the 100/300/600/2500 alpaca
seeds used for pilots, and the 20 alpaca prompts used for prompt-iteration. Each
configuration generates `n_per_prompt` rewrites, scores them with the same
ORRewardModel used at training time (k=5.0, c=0.75, d=100), and produces a
per-(model × corpus) distribution comparison.

Models are passed as positional --models entries, each of the form
"label=adapter_path" where `adapter_path` is either a LoRA adapter directory or
the literal string "baseline" (which means "no adapter — raw Llama-3-8B-Instruct").

Usage:
    sbatch run_eval_held_out.slurm
    # which calls:
    python eval_held_out.py \\
        --models baseline=baseline \\
                 claude_rwr=./rwr_claude_checkpoints/final \\
                 rwr_v3=<v3-checkpoint-path> \\
        --refusal_vector_path ../../refusal_vector/Vector_Extraction/refusal_vector.layer032.npz \\
        --dolly_csv ../../overrefusal_sampling/data/benign_dolly.csv \\
        --output_dir prompt_iteration_results/held_out_eval \\
        --n_alpaca 200 --n_dolly 200 --n_per_prompt 3
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, "..", "ppo_or"))

from rwr_config import BinningConfig, DataConfig, SYSTEM_PROMPT, PROMPT_TEMPLATE
from eval_rwr import load_generator, generate_paraphrases, score_generations


# ----------------------------------------------------------------------------
# Held-out set construction
# ----------------------------------------------------------------------------

def _alpaca_training_indices(training_seed: int, training_n: int) -> set:
    """Reconstruct the alpaca-cleaned indices used by generate_claude_dataset.py
    / generate_claude_batch.py at training time. The seed offset (+1) mirrors
    load_alpaca_seeds in those scripts. Returns indices in the cached dataset.
    """
    from datasets import load_dataset
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    rng = random.Random(training_seed + 1)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    out = set()
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
        if len(prompt) > 1500:
            continue
        out.add(idx)
        if len(out) >= training_n:
            break
    return out


def _alpaca_iteration_indices(iter_seed: int, iter_n: int) -> set:
    """Reconstruct the indices used by iterate_prompts_alpaca.py — same shuffle
    function as eval_v1_dolly.load_alpaca_prompts (no +1 offset on the seed).
    """
    from datasets import load_dataset
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    rng = random.Random(iter_seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    out = set()
    for idx in indices:
        ex = ds[idx]
        inst = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        if not inst:
            continue
        prompt = f"{inst}\n\n{inp}" if inp else inst
        if len(prompt) > 1500:
            continue
        out.add(idx)
        if len(out) >= iter_n:
            break
    return out


def load_alpaca_held_out(
    n_held_out: int,
    eval_seed: int = 99,
    training_seed: int = 42,
    training_n: int = 2500,
    iter_seed: int = 42,
    iter_n: int = 20,
) -> List[Tuple[int, str]]:
    """Returns (idx, prompt) tuples that are disjoint from training + iteration sets."""
    from datasets import load_dataset
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    print("[heldout] reconstructing training + iteration indices to exclude...")
    excluded = _alpaca_training_indices(training_seed, training_n) | \
               _alpaca_iteration_indices(iter_seed, iter_n)
    print(f"[heldout] excluding {len(excluded)} alpaca indices used for training/iteration")

    rng = random.Random(eval_seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    out: List[Tuple[int, str]] = []
    for idx in indices:
        if idx in excluded:
            continue
        ex = ds[idx]
        inst = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        if not inst:
            continue
        prompt = f"{inst}\n\n{inp}" if inp else inst
        if len(prompt) > 1500:
            continue
        out.append((idx, prompt))
        if len(out) >= n_held_out:
            break
    print(f"[heldout] alpaca: selected {len(out)} held-out prompts (eval_seed={eval_seed})")
    return out


def _dolly_training_ids(csv_path: str, training_seed: int, training_n: int) -> set:
    """Reconstruct the Dolly row-ids used at training time (mirror load_dolly_seeds)."""
    import csv
    rows: List[Tuple[int, str]] = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rid = int(r["id"])
            except (KeyError, ValueError):
                continue
            text = (r.get("prompt") or "").strip()
            if not text or len(text) > 1500:
                continue
            rows.append((rid, text))
    rng = random.Random(training_seed)
    rng.shuffle(rows)
    return {rid for rid, _ in rows[:training_n]}


def load_dolly_held_out(
    csv_path: str,
    n_held_out: int,
    eval_seed: int = 99,
    training_seed: int = 42,
    training_n: int = 2500,
) -> List[Tuple[int, str]]:
    import csv
    excluded = _dolly_training_ids(csv_path, training_seed, training_n)
    print(f"[heldout] excluding {len(excluded)} dolly row-ids used for training")

    rows: List[Tuple[int, str]] = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rid = int(r["id"])
            except (KeyError, ValueError):
                continue
            text = (r.get("prompt") or "").strip()
            if not text or len(text) > 1500:
                continue
            rows.append((rid, text))

    rng = random.Random(eval_seed)
    rng.shuffle(rows)

    out = [(rid, text) for rid, text in rows if rid not in excluded][:n_held_out]
    print(f"[heldout] dolly: selected {len(out)} held-out prompts (eval_seed={eval_seed})")
    return out


# ----------------------------------------------------------------------------
# Stats helpers
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Main eval loop
# ----------------------------------------------------------------------------

def parse_models(arglist: List[str]) -> List[Tuple[str, str]]:
    """Parse "label=path" tokens into (label, adapter_dir) tuples.
    adapter_dir == "baseline" means no adapter."""
    out = []
    for entry in arglist:
        if "=" not in entry:
            raise SystemExit(f"--models entry must be label=path, got: {entry}")
        label, path = entry.split("=", 1)
        out.append((label.strip(), path.strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help='List of "label=adapter_path" entries. '
                         'Use "label=baseline" for raw Llama-3-8B-Instruct (no adapter).')
    ap.add_argument("--refusal_vector_path", required=True)
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--dolly_csv", default="../../overrefusal_sampling/data/benign_dolly.csv")
    ap.add_argument("--output_dir", default="prompt_iteration_results/held_out_eval")
    ap.add_argument("--n_alpaca", type=int, default=200)
    ap.add_argument("--n_dolly", type=int, default=200)
    ap.add_argument("--n_per_prompt", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--gen_batch_size", type=int, default=4)
    ap.add_argument("--score_batch_size", type=int, default=8)
    ap.add_argument("--eval_seed", type=int, default=99)
    ap.add_argument("--training_seed", type=int, default=42)
    ap.add_argument("--training_n_alpaca", type=int, default=2500)
    ap.add_argument("--training_n_dolly",  type=int, default=2500)
    ap.add_argument("--iter_seed", type=int, default=42)
    ap.add_argument("--iter_n",   type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    models = parse_models(args.models)
    print(f"[eval] models to evaluate ({len(models)}):")
    for label, path in models:
        print(f"  {label:20s}  adapter_dir={path}")

    # --- Build held-out prompt sets ---
    held_out: Dict[str, List[Tuple[int, str]]] = {}
    if args.n_alpaca > 0:
        held_out["alpaca"] = load_alpaca_held_out(
            args.n_alpaca, args.eval_seed, args.training_seed, args.training_n_alpaca,
            args.iter_seed, args.iter_n,
        )
    if args.n_dolly > 0:
        held_out["dolly"] = load_dolly_held_out(
            args.dolly_csv, args.n_dolly, args.eval_seed, args.training_seed, args.training_n_dolly,
        )
    if not held_out:
        raise SystemExit("no held-out corpora to evaluate")

    # --- Generate for each model × corpus, then score ---
    # Outer loop over models (so we load each model only once);
    # inner loop over corpora.
    all_results: Dict = {
        "config": {
            "models":               [{"label": l, "adapter_dir": p} for l, p in models],
            "n_alpaca":             args.n_alpaca,
            "n_dolly":              args.n_dolly,
            "n_per_prompt":         args.n_per_prompt,
            "temperature":          args.temperature,
            "max_new_tokens":       args.max_new_tokens,
            "eval_seed":            args.eval_seed,
            "training_seed":        args.training_seed,
            "training_n_alpaca":    args.training_n_alpaca,
            "training_n_dolly":     args.training_n_dolly,
            "iter_seed":            args.iter_seed,
            "iter_n":               args.iter_n,
            "scoring_formula":      {"k": 5.0, "c": 0.75, "d": 100.0},
        },
        "held_out_prompts": {
            corpus: [{"idx": idx, "prompt": prompt} for idx, prompt in lst]
            for corpus, lst in held_out.items()
        },
        "results": {},
    }

    # Pre-extract just the prompt strings per corpus
    prompts_by_corpus = {corpus: [p for _, p in lst] for corpus, lst in held_out.items()}

    # Stage 1: generate rewrites with every model
    print(f"\n{'='*80}\nSTAGE 1: GENERATION\n{'='*80}")
    generations_by_model: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for label, adapter_path in models:
        adapter_arg = None if adapter_path == "baseline" else adapter_path
        print(f"\n[gen] loading model: {label} (adapter={adapter_arg})")
        t0 = time.time()
        model, tokenizer = load_generator(args.base_model, adapter_dir=adapter_arg)
        print(f"[gen] '{label}' model loaded in {time.time()-t0:.0f}s")
        generations_by_model[label] = {}
        for corpus, prompts in prompts_by_corpus.items():
            print(f"[gen] {label} × {corpus}: {len(prompts)} prompts × {args.n_per_prompt}")
            t1 = time.time()
            gens = generate_paraphrases(
                model, tokenizer, prompts,
                n_per_prompt=args.n_per_prompt,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.gen_batch_size,
            )
            print(f"[gen] {label} × {corpus} done in {time.time()-t1:.0f}s")
            generations_by_model[label][corpus] = gens
        del model
        torch.cuda.empty_cache()

    # Stage 2: score every (model × corpus) batch with ORRewardModel
    print(f"\n{'='*80}\nSTAGE 2: SCORING\n{'='*80}")
    for label, _ in models:
        all_results["results"][label] = {}
        for corpus, gens in generations_by_model[label].items():
            print(f"\n[score] {label} × {corpus}")
            t0 = time.time()
            scored = score_generations(gens, args.refusal_vector_path, args.base_model)
            print(f"[score] {label} × {corpus} done in {time.time()-t0:.0f}s ({len(scored)} pairs)")
            all_results["results"][label][corpus] = {
                "generations": scored,
                "stats": {
                    "or_score_raw":  _stats([s["or_score_raw"]  for s in scored]),
                    "refusal_delta": _stats([s["refusal_delta"] for s in scored]),
                    "similarity":    _stats([s["similarity"]    for s in scored]),
                },
            }

    # --- Save ---
    out_path = os.path.join(args.output_dir, "held_out_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[done] wrote {out_path}")

    # --- Summary comparison table ---
    print(f"\n{'='*100}")
    print(f"HELD-OUT EVAL SUMMARY")
    print(f"{'='*100}")
    for corpus in prompts_by_corpus:
        print(f"\n--- corpus: {corpus} (n_prompts={len(prompts_by_corpus[corpus])}, n_per_prompt={args.n_per_prompt}) ---")
        for metric in ("or_score_raw", "refusal_delta", "similarity"):
            print(f"\n  {metric}:")
            print(f"    {'model':24s} {'n':>5s} {'mean':>8s} {'median':>8s} "
                  f"{'p75':>8s} {'p90':>8s} {'p95':>8s} {'%pos':>8s}")
            rows = []
            for label, _ in models:
                s = all_results["results"][label][corpus]["stats"][metric]
                n = len(all_results["results"][label][corpus]["generations"])
                rows.append((label, n, s["mean"], s["median"], s["p75"], s["p90"], s["p95"], s["pct_positive"]))
            rows.sort(key=lambda r: -r[5])  # sort by p90 desc
            for label, n, m, md, p75, p90, p95, pp in rows:
                print(f"    {label:24s} {n:5d} {m:8.4f} {md:8.4f} "
                      f"{p75:8.4f} {p90:8.4f} {p95:8.4f} {pp:7.1%}")


if __name__ == "__main__":
    main()
