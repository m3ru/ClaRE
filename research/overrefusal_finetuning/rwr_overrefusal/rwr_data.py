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
    """Load all shard JSON files and flatten to (original, paraphrase, scores) dicts."""
    pairs = []
    for i in range(data_config.num_shards):
        path = os.path.join(data_config.shard_dir, f"or_susceptibility_rankings_shard{i}.json")
        if not os.path.exists(path):
            print(f"[data] Warning: shard {i} not found at {path}, skipping")
            continue
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
    print(f"[data] Loaded {len(pairs)} pairs from {data_config.num_shards} shards")
    return pairs


def filter_and_bin(
    pairs: List[Dict],
    binning_config: BinningConfig,
) -> Tuple[List[Dict], np.ndarray]:
    """Filter by similarity floor, assign quantile bins, return (pairs, sample_weights)."""
    # Filter
    filtered = [p for p in pairs if p["similarity"] >= binning_config.similarity_floor]
    n_dropped = len(pairs) - len(filtered)
    print(f"[data] Filtered: {n_dropped} pairs below similarity {binning_config.similarity_floor} "
          f"({len(filtered)} remaining)")

    if not filtered:
        raise ValueError("No pairs survived filtering")

    # Extract rewards and compute quantile bin edges
    rewards = np.array([p[binning_config.reward_key] for p in filtered])
    percentiles = np.linspace(0, 100, binning_config.num_bins + 1)
    edges = np.percentile(rewards, percentiles)

    # Assign bins (digitize returns 1-indexed, clip to num_bins)
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
    """SFT-style dataset: each item is a tokenized (prompt -> paraphrase) example."""

    def __init__(
        self,
        pairs: List[Dict],
        tokenizer,
        data_config: DataConfig,
        max_seq_length: int = 512,
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        input_ids, attention_mask, labels = tokenize_chat(
            pair["original"], pair["paraphrase"],
            self.tokenizer, self.data_config, self.max_seq_length,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def build_weighted_sampler(weights: np.ndarray, num_samples: int) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler from per-example weights."""
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=num_samples,
        replacement=True,
    )
