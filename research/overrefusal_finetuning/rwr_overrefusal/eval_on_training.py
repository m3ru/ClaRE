#!/usr/bin/env python3
"""In-distribution counterpart to eval_held_out.py.

Runs the IDENTICAL generate -> OR-score -> stats pipeline as eval_held_out.py
(same scorer k=5.0/c=0.75/d=100, same n_per_prompt / temperature / max_new_tokens),
but evaluates each model on a 200-prompt sample drawn FROM the training seed sets
(the same index/id functions eval_held_out.py uses to EXCLUDE training) instead of
the disjoint held-out set. Lets us read the train-vs-held-out (overfitting) gap.

Usage: same CLI as eval_held_out.py (e.g. --models, --output_dir, n_alpaca, ...).
Note: these are the training *seed* prompts; a few may have been filtered out of
the actual Claude rewrite shards, but they define the in-distribution population.
"""
import csv
import random
from typing import List, Tuple

import eval_held_out as E


def load_alpaca_training(n_held_out, eval_seed=99, training_seed=42, training_n=2500,
                         iter_seed=42, iter_n=20) -> List[Tuple[int, str]]:
    from datasets import load_dataset
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    included = E._alpaca_training_indices(training_seed, training_n)
    print(f"[train-eval] sampling {n_held_out} from {len(included)} alpaca TRAINING indices")
    rng = random.Random(eval_seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    out: List[Tuple[int, str]] = []
    for idx in idxs:
        if idx not in included:
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
    print(f"[train-eval] alpaca: selected {len(out)} TRAINING prompts (eval_seed={eval_seed})")
    return out


def load_dolly_training(csv_path, n_held_out, eval_seed=99, training_seed=42,
                        training_n=2500) -> List[Tuple[int, str]]:
    included = E._dolly_training_ids(csv_path, training_seed, training_n)
    print(f"[train-eval] sampling {n_held_out} from {len(included)} dolly TRAINING ids")
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
    out = [(rid, text) for rid, text in rows if rid in included][:n_held_out]
    print(f"[train-eval] dolly: selected {len(out)} TRAINING prompts (eval_seed={eval_seed})")
    return out


# Swap the held-out loaders for training-set loaders, then reuse main() verbatim.
E.load_alpaca_held_out = load_alpaca_training
E.load_dolly_held_out = load_dolly_training

if __name__ == "__main__":
    E.main()
