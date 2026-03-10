#!/usr/bin/env python3
"""Data loading utilities for OR PPO training.

Supports Databricks Dolly 15k JSONL, plain-text, and CSV formats.
"""

import json
import os
import random
from typing import Dict, Iterator, List, Optional, Tuple

from config import FEW_SHOT_EXAMPLES, PROMPT_TEMPLATE, SYSTEM_PROMPT, PromptConfig


class PromptDataset:
    def __init__(self, data: List[Dict[str, str]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {"prompt": self.data[idx]["prompt"]}

    def get_formatted_prompt(self, idx: int) -> str:
        return PROMPT_TEMPLATE.format(prompt=self.data[idx]["prompt"])

    def shuffle(self, seed: Optional[int] = None) -> "PromptDataset":
        data_copy = self.data.copy()
        if seed is not None:
            random.seed(seed)
        random.shuffle(data_copy)
        return PromptDataset(data_copy)


class PromptDataLoader:
    def __init__(
        self,
        dataset: PromptDataset,
        batch_size: int = 4,
        shuffle: bool = True,
        seed: Optional[int] = None,
        drop_last: bool = False,
    ):
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
        dataset = self.dataset
        if self.shuffle:
            seed = self.seed + self._epoch if self.seed is not None else None
            dataset = dataset.shuffle(seed)
        n = len(dataset)
        for i in range(0, n, self.batch_size):
            batch_end = min(i + self.batch_size, n)
            if self.drop_last and batch_end - i < self.batch_size:
                break
            yield [dataset[j] for j in range(i, batch_end)]
        self._epoch += 1


def load_jsonl(path: str) -> List[Dict[str, str]]:
    """Load JSONL — supports native {prompt} and Dolly {instruction, context} formats."""
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
                data.append({"prompt": line})
                continue
            if isinstance(item, str):
                data.append({"prompt": item})
            elif isinstance(item, dict):
                if "prompt" in item:
                    data.append({"prompt": item["prompt"]})
                elif "instruction" in item:
                    instr = str(item.get("instruction") or "").strip()
                    ctx = str(item.get("context") or "").strip()
                    if not instr:
                        raise ValueError(f"Line {line_num}: empty instruction")
                    prompt = instr if not ctx else f"{instr}\n\nContext:\n{ctx}"
                    data.append({"prompt": prompt})
                else:
                    raise ValueError(f"Line {line_num}: missing 'prompt' or 'instruction'")
            else:
                raise ValueError(f"Line {line_num}: unexpected type {type(item)}")
    if not data:
        raise ValueError(f"No valid data in {path}")
    return data


def load_txt(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    data: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append({"prompt": line})
    if not data:
        raise ValueError(f"No prompts in {path}")
    return data


def load_csv(path: str, prompt_column: str = "prompt") -> List[Dict[str, str]]:
    import csv
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    data: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if prompt_column not in (reader.fieldnames or []):
            raise ValueError(f"Column '{prompt_column}' not found. Available: {reader.fieldnames}")
        for row in reader:
            prompt = row.get(prompt_column, "").strip()
            if prompt:
                data.append({"prompt": prompt})
    if not data:
        raise ValueError(f"No prompts in {path}")
    return data


def train_test_split(
    data: List[Dict[str, str]],
    test_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    data_copy = data.copy()
    random.seed(seed)
    random.shuffle(data_copy)
    split_idx = int(len(data_copy) * (1 - test_fraction))
    return data_copy[:split_idx], data_copy[split_idx:]


def create_dataloaders(
    data_path: str,
    batch_size: int = 4,
    val_fraction: float = 0.1,
    seed: int = 42,
    shuffle_train: bool = True,
) -> Tuple[PromptDataLoader, PromptDataLoader]:
    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".jsonl":
        data = load_jsonl(data_path)
    elif ext == ".txt":
        data = load_txt(data_path)
    elif ext == ".csv":
        data = load_csv(data_path)
    else:
        try:
            data = load_jsonl(data_path)
        except (json.JSONDecodeError, ValueError):
            data = load_txt(data_path)

    train_data, val_data = train_test_split(data, val_fraction, seed)
    train_loader = PromptDataLoader(
        PromptDataset(train_data), batch_size=batch_size,
        shuffle=shuffle_train, seed=seed, drop_last=True,
    )
    val_loader = PromptDataLoader(
        PromptDataset(val_data), batch_size=batch_size,
        shuffle=False, drop_last=False,
    )
    return train_loader, val_loader


def _build_few_shot_turns(
    few_shot_examples: Optional[List] = None,
    prompt_template: str = PROMPT_TEMPLATE,
) -> List[Dict[str, str]]:
    """Build user/assistant turn pairs from few-shot examples."""
    examples = few_shot_examples if few_shot_examples is not None else FEW_SHOT_EXAMPLES
    turns = []
    for pair in examples:
        original, rewrite = pair[0], pair[1]
        turns.append({"role": "user", "content": prompt_template.format(prompt=original)})
        turns.append({"role": "assistant", "content": rewrite})
    return turns


_CACHED_FEW_SHOT_TURNS = _build_few_shot_turns()


def format_prompts_for_model(
    batch: List[Dict[str, str]],
    tokenizer,
    prompt_config: Optional[PromptConfig] = None,
) -> List[str]:
    """Format prompts using chat template with OR system prompt + few-shot examples."""
    if prompt_config is not None:
        system_prompt = prompt_config.system_prompt
        prompt_template = prompt_config.prompt_template
        few_shot_turns = _build_few_shot_turns(
            prompt_config.few_shot_examples, prompt_template,
        )
    else:
        system_prompt = SYSTEM_PROMPT
        prompt_template = PROMPT_TEMPLATE
        few_shot_turns = _CACHED_FEW_SHOT_TURNS

    formatted = []
    for item in batch:
        user_content = prompt_template.format(prompt=item["prompt"])
        messages = [
            {"role": "system", "content": system_prompt},
            *few_shot_turns,
            {"role": "user", "content": user_content},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted.append(text)
    return formatted
