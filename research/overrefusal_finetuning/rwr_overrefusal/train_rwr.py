#!/usr/bin/env python3
"""RWR (Reward-Weighted Regression) training for overrefusal prompt rewriting.

Offline RL: trains Llama-3-8B-Instruct (LoRA) on pre-scored (original, paraphrase)
pairs, sampling high-reward examples more frequently via quantile binning.

Usage:
    python train_rwr.py --shard_dir ../or_paraphrase_3k --output_dir ./rwr_checkpoints
"""

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rwr_config import BinningConfig, DataConfig, LoRAConfig, ModelConfig, TrainingConfig
from rwr_data import (
    RWRDataset,
    build_weighted_sampler,
    filter_and_bin,
    load_shards,
    recompute_or_scores,
    train_val_split,
)


def collate_fn(batch, pad_token_id: int):
    """Pad batch to max length in the batch."""
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, attention_mask, labels = [], [], []
    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(b["attention_mask"] + [0] * pad_len)
        labels.append(b["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def load_model_and_tokenizer(model_config: ModelConfig, lora_config: LoRAConfig, training_config: TrainingConfig):
    """Load base model with LoRA adapter."""
    from peft import LoraConfig as PeftLoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = model_config.hf_token or os.environ.get("HF_TOKEN")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map.get(model_config.torch_dtype, torch.float32)

    print(f"[model] Loading {model_config.base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.base_model_id, use_fast=True, token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_config.base_model_id,
        token=hf_token,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation="eager",
    )

    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    peft_config = PeftLoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        task_type=lora_config.task_type,
        target_modules=lora_config.target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


def evaluate(model, val_loader, device):
    """Compute average loss on validation set."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # Count non-masked tokens for proper averaging
            n_tokens = (batch["labels"] != -100).sum().item()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens
    model.train()
    return total_loss / max(total_tokens, 1)


def train(
    model_config: ModelConfig,
    lora_config: LoRAConfig,
    binning_config: BinningConfig,
    training_config: TrainingConfig,
    data_config: DataConfig,
    num_samples_per_epoch: int = 0,
):
    torch.manual_seed(training_config.seed)

    # --- Data ---
    pairs = load_shards(data_config)
    if binning_config.recompute_or_score:
        recompute_or_scores(pairs, binning_config)
    filtered, sample_weights = filter_and_bin(pairs, binning_config)
    train_pairs, train_weights, val_pairs, val_weights = train_val_split(
        filtered, sample_weights, training_config.val_fraction, training_config.seed,
    )

    # --- Model ---
    model, tokenizer = load_model_and_tokenizer(model_config, lora_config, training_config)
    device = next(model.parameters()).device

    # --- Datasets ---
    train_dataset = RWRDataset(train_pairs, tokenizer, data_config, training_config.max_seq_length)
    val_dataset = RWRDataset(val_pairs, tokenizer, data_config, training_config.max_seq_length)

    sampler_size = num_samples_per_epoch if num_samples_per_epoch > 0 else len(train_dataset)
    print(f"[train] WeightedRandomSampler size per epoch: {sampler_size}")
    sampler = build_weighted_sampler(train_weights, num_samples=sampler_size)
    pad_id = tokenizer.pad_token_id

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.per_device_batch_size,
        sampler=sampler,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.per_device_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=0,
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    total_steps = len(train_loader) * training_config.num_epochs // training_config.gradient_accumulation_steps
    warmup_steps = int(total_steps * training_config.warmup_ratio)

    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # --- Training loop ---
    os.makedirs(training_config.output_dir, exist_ok=True)
    save_config(training_config.output_dir, model_config, lora_config, binning_config, training_config, data_config)

    print(f"\n[train] Starting RWR training: {training_config.num_epochs} epochs, "
          f"{len(train_loader)} steps/epoch, grad_accum={training_config.gradient_accumulation_steps}")

    global_step = 0
    model.train()
    print("[train] Entering training loop...")

    for epoch in range(training_config.num_epochs):
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            if step == 0 and epoch == 0:
                print("[train] First batch loaded, starting forward pass...")
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            if step == 0 and epoch == 0:
                print(f"[train] First forward pass done, loss={outputs.loss.item():.4f}")
            loss = outputs.loss / training_config.gradient_accumulation_steps
            loss.backward()

            n_tokens = (batch["labels"] != -100).sum().item()
            epoch_loss += outputs.loss.item() * n_tokens
            epoch_tokens += n_tokens

            if (step + 1) % training_config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % training_config.log_every_steps == 0:
                    avg = epoch_loss / max(epoch_tokens, 1)
                    lr = scheduler.get_last_lr()[0]
                    print(f"  step {global_step} | loss {avg:.4f} | lr {lr:.2e}")

        epoch_avg = epoch_loss / max(epoch_tokens, 1)
        val_loss = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"[epoch {epoch+1}/{training_config.num_epochs}] "
              f"train_loss={epoch_avg:.4f} val_loss={val_loss:.4f} time={elapsed:.0f}s")

        if (epoch + 1) % training_config.save_every_epochs == 0:
            ckpt_dir = os.path.join(training_config.output_dir, f"epoch_{epoch+1}")
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  Saved checkpoint to {ckpt_dir}")

    # Save final
    final_dir = os.path.join(training_config.output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n[done] Final model saved to {final_dir}")


def save_config(output_dir, model_config, lora_config, binning_config, training_config, data_config):
    """Dump all config to JSON for reproducibility."""
    from dataclasses import asdict
    config = {
        "model": asdict(model_config),
        "lora": asdict(lora_config),
        "binning": asdict(binning_config),
        "training": asdict(training_config),
        "data": asdict(data_config),
    }
    path = os.path.join(output_dir, "rwr_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[config] Saved to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="RWR training for overrefusal prompt rewriting")
    parser.add_argument("--shard_dir", type=str, default="../or_paraphrase_3k")
    parser.add_argument("--output_dir", type=str, default="./rwr_checkpoints")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1.5e-5)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_bins", type=int, default=5)
    parser.add_argument("--similarity_floor", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    # BinningConfig overrides (defaults match the v3 BinningConfig dataclass)
    parser.add_argument("--bin_weights", type=str, default="",
                        help="Comma-separated bin weights (default: BinningConfig default).")
    parser.add_argument("--similarity_exponent", type=float, default=None,
                        help="OR formula k. Default: BinningConfig default (5.0).")
    parser.add_argument("--similarity_center", type=float, default=None,
                        help="OR formula c. Default: BinningConfig default (0.75).")
    parser.add_argument("--refusal_divisor", type=float, default=None,
                        help="OR formula d. Default: BinningConfig default (100.0).")
    parser.add_argument("--num_samples_per_epoch", type=int, default=0,
                        help="WeightedRandomSampler size (0 = len(train_dataset)).")
    return parser.parse_args()


def main():
    args = parse_args()

    model_config = ModelConfig(base_model_id=args.base_model)
    lora_config = LoRAConfig()

    # Build BinningConfig with optional overrides
    binning_kwargs = dict(
        num_bins=args.num_bins,
        similarity_floor=args.similarity_floor,
    )
    if args.bin_weights:
        binning_kwargs["bin_weights"] = [float(x) for x in args.bin_weights.split(",")]
    if args.similarity_exponent is not None:
        binning_kwargs["similarity_exponent"] = args.similarity_exponent
    if args.similarity_center is not None:
        binning_kwargs["similarity_center"] = args.similarity_center
    if args.refusal_divisor is not None:
        binning_kwargs["refusal_divisor"] = args.refusal_divisor
    binning_config = BinningConfig(**binning_kwargs)

    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        output_dir=args.output_dir,
        gradient_checkpointing=args.gradient_checkpointing and not args.no_gradient_checkpointing,
    )
    data_config = DataConfig(shard_dir=args.shard_dir)

    train(model_config, lora_config, binning_config, training_config, data_config,
          num_samples_per_epoch=args.num_samples_per_epoch)


if __name__ == "__main__":
    main()
