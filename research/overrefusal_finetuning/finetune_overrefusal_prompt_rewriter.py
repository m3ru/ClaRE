#!/usr/bin/env python3
"""Fine-tune Llama-8B to rewrite benign prompts into over-refusal-triggering prompts.

Dataset format expected (CSV):
- input (benign): sanitized_prompt
- target (over-refusal prompt): Request

This trains a prompt-rewriter via SFT (QLoRA/LoRA).
"""

import argparse
import os
from dataclasses import asdict, dataclass
from typing import Dict, List

import pandas as pd
from datasets import Dataset


SYSTEM_PROMPT = (
    "You rewrite benign prompts into versions that are likely to trigger safety over-refusals, "
    "while preserving the original benign intent. "
    "Output ONLY the rewritten prompt text with no preamble, no explanation, and no quotes."
)


@dataclass
class DataStats:
    total_rows: int
    kept_rows: int
    removed_empty: int
    removed_analysis: int


def build_chat_text(tokenizer, benign_prompt: str, overrefusal_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Benign prompt:\n{benign_prompt}"},
        {"role": "assistant", "content": overrefusal_prompt},
    ]
    # For SFT we want the full conversation including the assistant content.
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def load_and_prepare_dataset(csv_path: str, seed: int, val_frac: float) -> tuple[Dataset, Dataset, DataStats]:
    df = pd.read_csv(csv_path)
    if "sanitized_prompt" not in df.columns or "Request" not in df.columns:
        raise ValueError(
            f"CSV must contain columns 'sanitized_prompt' and 'Request'. Got: {list(df.columns)}"
        )

    total = len(df)

    # Normalize and filter
    s = df["sanitized_prompt"].fillna("").astype(str)
    t = df["Request"].fillna("").astype(str)

    empty_mask = (s.str.strip() == "") | (t.str.strip() == "")
    df = df.loc[~empty_mask].copy()

    # Remove rows where sanitized_prompt starts with "analysis" (common LLM artifact)
    s2 = df["sanitized_prompt"].fillna("").astype(str)
    analysis_mask = s2.str.lstrip().str.lower().str.startswith("analysis")
    df = df.loc[~analysis_mask].copy()

    stats = DataStats(
        total_rows=total,
        kept_rows=len(df),
        removed_empty=int(empty_mask.sum()),
        removed_analysis=int(analysis_mask.sum()),
    )

    ds = Dataset.from_pandas(df, preserve_index=False)
    split = ds.train_test_split(test_size=val_frac, seed=seed, shuffle=True)
    return split["train"], split["test"], stats


def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument(
        "--csv",
        required=True,
        help="Path to PHTest_train_sanitized_cleaned.csv (must include sanitized_prompt and Request columns).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.05)

    # Model
    ap.add_argument(
        "--model_id",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HF model id for llama-8b instruct.",
    )
    ap.add_argument("--max_seq_len", type=int, default=512)

    # Training
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_train_epochs", type=float, default=2.0)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lr_scheduler_type", default="cosine")
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--save_total_limit", type=int, default=3)

    # QLoRA / LoRA
    ap.add_argument(
        "--use_4bit",
        type=int,
        default=0,
        help="1 to enable 4-bit QLoRA (requires bitsandbytes), 0 for bf16/full precision base",
    )
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Packing
    ap.add_argument("--packing", type=int, default=1, help="1 to pack multiple samples into sequences")

    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Heavy deps after args parsing
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig
    from trl import SFTTrainer

    # Load and split data
    train_ds, eval_ds, stats = load_and_prepare_dataset(args.csv, seed=args.seed, val_frac=args.val_frac)
    print("[data]", asdict(stats), flush=True)

    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARNING: HUGGING_FACE_HUB_TOKEN not set; gated models will fail.", flush=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, token=hf_token)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)

    # Helper to precompute a plain-text field for TRL versions that support `dataset_text_field`.
    def add_text_field(batch: Dict[str, List[str]]) -> Dict[str, List[str]]:
        sps = batch.get("sanitized_prompt") or []
        rqs = batch.get("Request") or []
        texts: List[str] = []
        for benign, target in zip(sps, rqs):
            texts.append(build_chat_text(tokenizer, benign_prompt=benign, overrefusal_prompt=target))
        return {"text": texts}

    # Model (optionally 4-bit)
    quantization_config = None
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if int(args.use_4bit) == 1:
        try:
            import bitsandbytes as _bnb  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "--use_4bit=1 requires bitsandbytes. Either install it, or rerun with --use_4bit 0."
            ) from e
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=hf_token,
        device_map="auto",
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # Format dataset into chat text.
    # TRL may call formatting_func in batched OR non-batched mode depending on version.
    def formatting_func(examples):
        sp = examples.get("sanitized_prompt")
        rq = examples.get("Request")
        if sp is None or rq is None:
            raise ValueError("Dataset missing required columns for formatting_func")

        # Batched mode => return List[str]
        if isinstance(sp, list):
            outs: List[str] = []
            for benign, target in zip(sp, rq):
                outs.append(build_chat_text(tokenizer, benign_prompt=benign, overrefusal_prompt=target))
            return outs

        # Single-example mode => return str
        return build_chat_text(tokenizer, benign_prompt=sp, overrefusal_prompt=rq)

    # transformers recently renamed evaluation_strategy -> eval_strategy.
    # To stay compatible across cluster environments, pick the supported kwarg dynamically.
    import inspect

    ta_sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in ta_sig.parameters:
        eval_key = "evaluation_strategy"
    elif "eval_strategy" in ta_sig.parameters:
        eval_key = "eval_strategy"
    else:
        eval_key = None

    ta_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        bf16=torch.cuda.is_available(),
        fp16=False,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=[],
        remove_unused_columns=False,
        optim="paged_adamw_8bit" if int(args.use_4bit) == 1 else "adamw_torch",
    )
    if eval_key is not None:
        ta_kwargs[eval_key] = "steps"
    else:
        print(
            "[warn] TrainingArguments has neither evaluation_strategy nor eval_strategy; disabling periodic eval.",
            flush=True,
        )

    training_args = TrainingArguments(**ta_kwargs)

    # TRL SFTTrainer signature has changed across versions. Build kwargs dynamically.
    import inspect as _inspect

    sft_sig = _inspect.signature(SFTTrainer.__init__)
    sft_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    # Prefer dataset_text_field if supported
    if "dataset_text_field" in sft_sig.parameters:
        train_ds = train_ds.map(add_text_field, batched=True, desc="Formatting train chat text")
        eval_ds = eval_ds.map(add_text_field, batched=True, desc="Formatting eval chat text")
        sft_kwargs["train_dataset"] = train_ds
        sft_kwargs["eval_dataset"] = eval_ds
        sft_kwargs["dataset_text_field"] = "text"
    elif "formatting_func" in sft_sig.parameters:
        # fallback for newer TRL that supports formatting_func
        sft_kwargs["formatting_func"] = formatting_func

    # Tokenizer / processor
    if "tokenizer" in sft_sig.parameters:
        sft_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in sft_sig.parameters:
        sft_kwargs["processing_class"] = tokenizer  # best-effort

    # PEFT config
    if "peft_config" in sft_sig.parameters:
        sft_kwargs["peft_config"] = peft_config

    # Max length kwarg varies
    if "max_seq_length" in sft_sig.parameters:
        sft_kwargs["max_seq_length"] = args.max_seq_len
    elif "max_length" in sft_sig.parameters:
        sft_kwargs["max_length"] = args.max_seq_len

    # Packing support varies
    if "packing" in sft_sig.parameters:
        sft_kwargs["packing"] = bool(int(args.packing))

    trainer = SFTTrainer(**sft_kwargs)

    trainer.train()

    # Save adapter + tokenizer
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save stats for reproducibility
    with open(os.path.join(args.output_dir, "data_stats.json"), "w", encoding="utf-8") as f:
        import json

        json.dump(asdict(stats), f, indent=2)

    print(f"[done] Saved adapter+tokenizer to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
