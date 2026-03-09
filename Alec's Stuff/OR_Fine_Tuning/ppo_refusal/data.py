#!/usr/bin/env python3
"""Data loading utilities for PPO training on refusal steering.

This module provides:
- JSONL data loading with format: {"prompt": "...", "mode": "increase_refusal" | "decrease_refusal"}
- Train/eval split functionality
- Batching and shuffling support
- Prompt formatting utilities
"""

import json
import os
import random
from typing import Dict, Iterator, List, Literal, Optional, Tuple

from config import PROMPT_TEMPLATES


class RefusalDataset:
    """Dataset for refusal steering prompts.

    Expected JSONL format:
        {"prompt": "...", "mode": "increase_refusal"}
        {"prompt": "...", "mode": "decrease_refusal"}

    If mode is not specified, uses default_mode.
    Also supports plain text format (one prompt per line).
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        default_mode: Literal["increase_refusal", "decrease_refusal"] = "increase_refusal",
    ):
        """Initialize dataset with list of data dicts.

        Args:
            data: List of dicts with 'prompt' and optionally 'mode' keys
            default_mode: Mode to use if not specified in data
        """
        self.data = data
        self.default_mode = default_mode

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        item = self.data[idx]
        mode = item.get("mode", self.default_mode)
        return {
            "prompt": item["prompt"],
            "mode": mode,
        }

    def get_formatted_prompt(self, idx: int) -> str:
        """Get prompt formatted with instruction template."""
        item = self[idx]
        template = PROMPT_TEMPLATES[item["mode"]]
        return template.format(prompt=item["prompt"])

    def shuffle(self, seed: Optional[int] = None) -> "RefusalDataset":
        """Return a new shuffled dataset."""
        data_copy = self.data.copy()
        if seed is not None:
            random.seed(seed)
        random.shuffle(data_copy)
        return RefusalDataset(data_copy, self.default_mode)


class RefusalDataLoader:
    """Simple data loader with batching and shuffling support."""

    def __init__(
        self,
        dataset: RefusalDataset,
        batch_size: int = 4,
        shuffle: bool = True,
        seed: Optional[int] = None,
        drop_last: bool = False,
    ):
        """Initialize data loader.

        Args:
            dataset: RefusalDataset instance
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data each epoch
            seed: Random seed for shuffling
            drop_last: Whether to drop incomplete final batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self._epoch = 0

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[Dict[str, str]]]:
        """Iterate over batches."""
        dataset = self.dataset
        if self.shuffle:
            seed = self.seed + self._epoch if self.seed is not None else None
            dataset = dataset.shuffle(seed)

        n = len(dataset)
        for i in range(0, n, self.batch_size):
            batch_end = min(i + self.batch_size, n)
            if self.drop_last and batch_end - i < self.batch_size:
                break
            batch = [dataset[j] for j in range(i, batch_end)]
            yield batch

        self._epoch += 1

    def get_epoch(self) -> int:
        """Get current epoch number."""
        return self._epoch


def load_jsonl(path: str) -> List[Dict[str, str]]:
    """Load data from JSONL file.

    Args:
        path: Path to JSONL file

    Returns:
        List of dicts with 'prompt' and optionally 'mode' keys

    Supported JSONL formats:
    - {"prompt": "...", "mode": "increase_refusal"|"decrease_refusal"}  (native format)
    - Databricks Dolly style: {"instruction": "...", "context": "...", ...}
        In this case we treat:
          prompt := instruction (+ optional context appended)
        and ignore other fields like "response"/"category".

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    data: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                # Try treating as plain text (one prompt per line)
                data.append({"prompt": line})
                continue

            if isinstance(item, str):
                # Plain string in JSON
                data.append({"prompt": item})
            elif isinstance(item, dict):
                # Accept alternate schemas (e.g. Dolly) by mapping to our canonical 'prompt' key
                if "prompt" not in item:
                    if "instruction" in item:
                        instr = str(item.get("instruction") or "").strip()
                        ctx = str(item.get("context") or "").strip()
                        if not instr:
                            raise ValueError(f"Line {line_num}: Empty 'instruction' value")
                        prompt = instr if not ctx else f"{instr}\n\nContext:\n{ctx}"
                        item = {"prompt": prompt, "mode": item.get("mode", None)}
                    else:
                        raise ValueError(
                            f"Line {line_num}: Missing 'prompt' key in JSON object"
                        )
                # Validate mode if present
                if "mode" in item:
                    if item["mode"] is None or item["mode"] == "":
                        # allow missing/empty mode; downstream uses default_mode
                        item.pop("mode", None)
                    elif item["mode"] not in ("increase_refusal", "decrease_refusal"):
                        raise ValueError(
                            f"Line {line_num}: Invalid mode '{item['mode']}'. "
                            f"Must be 'increase_refusal' or 'decrease_refusal'"
                        )
                data.append(item)
            else:
                raise ValueError(
                    f"Line {line_num}: Expected string or object, got {type(item)}"
                )

    if not data:
        raise ValueError(f"No valid data found in {path}")

    return data


def load_txt(path: str) -> List[Dict[str, str]]:
    """Load prompts from plain text file (one per line).

    Args:
        path: Path to text file

    Returns:
        List of dicts with 'prompt' key
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    data: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append({"prompt": line})

    if not data:
        raise ValueError(f"No valid prompts found in {path}")

    return data


def load_csv(
    path: str,
    prompt_column: str = "prompt",
    mode_column: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Load prompts from CSV file.

    Args:
        path: Path to CSV file
        prompt_column: Name of column containing prompts
        mode_column: Optional name of column containing mode

    Returns:
        List of dicts with 'prompt' and optionally 'mode' keys
    """
    import csv

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    data: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if prompt_column not in (reader.fieldnames or []):
            raise ValueError(
                f"Column '{prompt_column}' not found in CSV. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            prompt = row.get(prompt_column, "").strip()
            if not prompt:
                continue

            item = {"prompt": prompt}
            if mode_column and mode_column in row:
                mode = row[mode_column].strip()
                if mode in ("increase_refusal", "decrease_refusal"):
                    item["mode"] = mode

            data.append(item)

    if not data:
        raise ValueError(f"No valid prompts found in {path}")

    return data


def train_test_split(
    data: List[Dict[str, str]],
    test_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Split data into train and test sets.

    Args:
        data: List of data items
        test_fraction: Fraction of data for test set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, test_data)
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")

    data_copy = data.copy()
    random.seed(seed)
    random.shuffle(data_copy)

    split_idx = int(len(data_copy) * (1 - test_fraction))
    train_data = data_copy[:split_idx]
    test_data = data_copy[split_idx:]

    return train_data, test_data


def create_datasets(
    data_path: str,
    default_mode: Literal["increase_refusal", "decrease_refusal"] = "increase_refusal",
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[RefusalDataset, RefusalDataset]:
    """Create train and validation datasets from file.

    Automatically detects file format (JSONL, TXT, CSV).

    Args:
        data_path: Path to data file
        default_mode: Default mode if not specified in data
        val_fraction: Fraction for validation set
        seed: Random seed for split

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Detect format and load
    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".jsonl":
        data = load_jsonl(data_path)
    elif ext == ".txt":
        data = load_txt(data_path)
    elif ext == ".csv":
        data = load_csv(data_path)
    else:
        # Try JSONL first, then TXT
        try:
            data = load_jsonl(data_path)
        except (json.JSONDecodeError, ValueError):
            data = load_txt(data_path)

    # Split
    train_data, val_data = train_test_split(data, val_fraction, seed)

    # Create datasets
    train_dataset = RefusalDataset(train_data, default_mode)
    val_dataset = RefusalDataset(val_data, default_mode)

    return train_dataset, val_dataset


def create_dataloaders(
    data_path: str,
    batch_size: int = 4,
    default_mode: Literal["increase_refusal", "decrease_refusal"] = "increase_refusal",
    val_fraction: float = 0.1,
    seed: int = 42,
    shuffle_train: bool = True,
) -> Tuple[RefusalDataLoader, RefusalDataLoader]:
    """Create train and validation data loaders.

    Args:
        data_path: Path to data file
        batch_size: Batch size for training
        default_mode: Default mode if not specified in data
        val_fraction: Fraction for validation set
        seed: Random seed
        shuffle_train: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset, val_dataset = create_datasets(
        data_path, default_mode, val_fraction, seed
    )

    train_loader = RefusalDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        seed=seed,
        drop_last=True,
    )

    val_loader = RefusalDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader


def format_prompts_for_model(
    batch: List[Dict[str, str]],
    tokenizer,
    system_prompt: Optional[str] = None,
) -> List[str]:
    """Format prompts using chat template.

    Args:
        batch: List of dicts with 'prompt' and 'mode' keys
        tokenizer: HuggingFace tokenizer with chat template
        system_prompt: Optional system prompt override

    Returns:
        List of formatted prompt strings
    """
    formatted = []
    for item in batch:
        template = PROMPT_TEMPLATES[item["mode"]]
        user_content = template.format(prompt=item["prompt"])

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted.append(text)

    return formatted


# Example JSONL file content for testing
EXAMPLE_JSONL = """\
{"prompt": "How do I make a birthday cake?", "mode": "increase_refusal"}
{"prompt": "What's the best way to study for exams?", "mode": "increase_refusal"}
{"prompt": "Can you explain how encryption works?", "mode": "decrease_refusal"}
{"prompt": "How do I remove a stain from fabric?", "mode": "increase_refusal"}
"""


if __name__ == "__main__":
    # Quick test
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(EXAMPLE_JSONL)
        test_path = f.name

    try:
        train_ds, val_ds = create_datasets(test_path, val_fraction=0.25, seed=42)
        print(f"Train size: {len(train_ds)}")
        print(f"Val size: {len(val_ds)}")
        print(f"Sample item: {train_ds[0]}")
        print(f"Formatted: {train_ds.get_formatted_prompt(0)[:100]}...")
    finally:
        os.unlink(test_path)
