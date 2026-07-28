"""Data pipeline for RWR overrefusal training.

Loads or_paraphrase_3k shards, filters by similarity, assigns quantile bins,
and provides a weighted sampler that upweights high-reward examples.
"""

import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from rwr_config import BinningConfig, DataConfig


def load_shards(data_config: DataConfig) -> List[Dict]:
    """Load all shard JSON files and flatten to (original, paraphrase, scores) dicts.

    Supports multiple shard directories (comma-separated in data_config.shard_dir)
    and auto-detects shard count by globbing for matching files.
    """
    import glob

    shard_dirs = [d.strip() for d in data_config.shard_dir.split(",") if d.strip()]
    pairs = []
    total_shards = 0

    for shard_dir in shard_dirs:
        shard_files = sorted(glob.glob(os.path.join(shard_dir, "or_susceptibility_rankings_shard*.json")))
        if not shard_files:
            print(f"[data] Warning: no shards found in {shard_dir}")
            continue
        total_shards += len(shard_files)
        for path in shard_files:
            with open(path, "r") as f:
                shard = json.load(f)
            for item in shard:
                for p in item["paraphrases"]:
                    pairs.append({
                        "original": item["original"],
                        "paraphrase": p["paraphrase"],
                        "or_score_raw": p["or_score_raw"],
                        "refusal_delta": p["refusal_delta"],
                        "similarity": p["similarity"],
                    })

    print(f"[data] Loaded {len(pairs)} pairs from {total_shards} shards across {len(shard_dirs)} dir(s)")
    return pairs


def recompute_or_scores(pairs: List[Dict], binning_config: BinningConfig) -> None:
    """Recompute or_score_raw in-place from refusal_delta and similarity.

    Different data sources may have used different exponents (k=18.4 vs k=5.0).
    This ensures consistent scoring across all data before binning.
    """
    import math
    k = binning_config.similarity_exponent
    c = binning_config.similarity_center
    d = binning_config.refusal_divisor
    for p in pairs:
        p["or_score_raw"] = math.exp(k * (p["similarity"] - c)) * p["refusal_delta"] / d
    print(f"[data] Recomputed or_score_raw with k={k}, c={c}, d={d}")


def filter_and_bin(
    pairs: List[Dict],
    binning_config: BinningConfig,
) -> Tuple[List[Dict], np.ndarray]:
    """Filter by similarity floor, assign quantile bins, return (pairs, sample_weights)."""
    # Filter: similarity floor + non-negative reward
    # Negative-reward examples would teach the model to produce paraphrases that
    # DECREASE refusal — the opposite of what we want. Since this is SFT (not RL),
    # there's no mechanism to push the model away from bad examples, only toward them.
    pre_count = len(pairs)
    filtered = [p for p in pairs
                if p["similarity"] >= binning_config.similarity_floor
                and p[binning_config.reward_key] >= 0]
    n_sim_dropped = sum(1 for p in pairs if p["similarity"] < binning_config.similarity_floor)
    n_neg_dropped = pre_count - n_sim_dropped - len(filtered)
    print(f"[data] Filtered: {n_sim_dropped} below similarity {binning_config.similarity_floor}, "
          f"{n_neg_dropped} with negative reward "
          f"({len(filtered)} remaining)")

    if not filtered:
        raise ValueError("No pairs survived filtering")

    # Extract rewards and assign bins.
    rewards = np.array([p[binning_config.reward_key] for p in filtered])
    if binning_config.bin_edges is not None:
        # Absolute-threshold bins: robust to heavily-skewed rewards where quantile
        # bins collapse the whole signal-bearing tail into one bin with the noise.
        interior = np.asarray(binning_config.bin_edges, dtype=float)
        if len(interior) != binning_config.num_bins - 1:
            raise ValueError(
                f"bin_edges must have num_bins-1={binning_config.num_bins - 1} interior "
                f"boundaries, got {len(interior)}")
        bin_indices = np.digitize(rewards, interior)  # 0..num_bins-1
    else:
        percentiles = np.linspace(0, 100, binning_config.num_bins + 1)
        edges = np.percentile(rewards, percentiles)
        bin_indices = np.digitize(rewards, edges[1:-1])  # 0-indexed, 0..num_bins-1

    # Map bin index -> sampling weight
    weights = np.array(binning_config.bin_weights)
    sample_weights = weights[bin_indices]

    # Log bin stats
    for b in range(binning_config.num_bins):
        mask = bin_indices == b
        count = mask.sum()
        r_mean = rewards[mask].mean() if count > 0 else 0
        print(f"  Bin {b}: n={count}, reward_mean={r_mean:.3f}, weight={weights[b]}")

    return filtered, sample_weights


def train_val_split(
    pairs: List[Dict],
    sample_weights: np.ndarray,
    val_fraction: float,
    seed: int,
) -> Tuple[List[Dict], np.ndarray, List[Dict], np.ndarray]:
    """Split by unique original prompts so no prompt leaks between train and val."""
    rng = random.Random(seed)

    # Group pair indices by original prompt
    prompt_to_indices: Dict[str, List[int]] = {}
    for i, p in enumerate(pairs):
        prompt_to_indices.setdefault(p["original"], []).append(i)

    unique_prompts = list(prompt_to_indices.keys())
    rng.shuffle(unique_prompts)
    n_val = max(1, int(len(unique_prompts) * val_fraction))

    val_prompts = set(unique_prompts[:n_val])
    train_idx, val_idx = [], []
    for prompt, indices in prompt_to_indices.items():
        if prompt in val_prompts:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)

    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    train_weights = sample_weights[train_idx]
    val_weights = sample_weights[val_idx]

    print(f"[data] Split: {len(train_pairs)} train, {len(val_pairs)} val "
          f"({len(unique_prompts) - n_val} / {n_val} prompts)")
    return train_pairs, train_weights, val_pairs, val_weights


def tokenize_chat(original: str, paraphrase: str, tokenizer, data_config: DataConfig, max_seq_length: int):
    """Tokenize prompt and completion separately, then concatenate.

    Returns (input_ids, attention_mask, labels) where labels are -100 for the
    prompt tokens and real token ids for the completion tokens.  This avoids any
    boundary ambiguity from tokenizing a single concatenated string.
    """
    user_content = data_config.prompt_template.format(prompt=original)

    # Tokenize the prompt portion: system + user + assistant header
    prompt_messages = [
        {"role": "system", "content": data_config.system_prompt},
        {"role": "user", "content": user_content},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,   # no-op for Llama; disables Qwen3 <think> so completion follows directly
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    # Tokenize the completion portion: paraphrase + eos
    completion_ids = tokenizer.encode(paraphrase, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        completion_ids.append(tokenizer.eos_token_id)

    # Concatenate and truncate
    input_ids = (prompt_ids + completion_ids)[:max_seq_length]
    attention_mask = [1] * len(input_ids)

    # Labels: -100 on prompt, real ids on completion
    prompt_len = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    return input_ids, attention_mask, labels


class RWRDataset(Dataset):
    """SFT-style dataset: pre-tokenized (prompt -> paraphrase) examples."""

    def __init__(
        self,
        pairs: List[Dict],
        tokenizer,
        data_config: DataConfig,
        max_seq_length: int = 512,
    ):
        # Pre-tokenize everything upfront to avoid per-batch overhead
        print(f"[data] Pre-tokenizing {len(pairs)} examples...")
        self.examples = []
        for i, pair in enumerate(pairs):
            input_ids, attention_mask, labels = tokenize_chat(
                pair["original"], pair["paraphrase"],
                tokenizer, data_config, max_seq_length,
            )
            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(pairs)} tokenized")
        print(f"[data] Pre-tokenization done")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def build_weighted_sampler(weights: np.ndarray, num_samples: int) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler from per-example weights."""
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=num_samples,
        replacement=True,
    )
