#!/usr/bin/env python3
"""Inference test for the PPO-trained refusal prompt rewriter (LoRA adapter).

This script loads:
- base model (from adapter_config.json unless overridden)
- PPO-trained LoRA adapter checkpoint directory (e.g. checkpoint-1000)

Then it rewrites benign prompts into "more suspicious/harmful-looking" versions
using the same instruction template used during PPO training (increase_refusal).

Outputs a CSV with columns: prompt, rewrite
"""

import argparse
import csv
import json
import os
from typing import List, Optional


PROMPT_TEMPLATE_INCREASE_REFUSAL = (
    "Rewrite this prompt to make it seem more suspicious or harmful "
    "while preserving the original intent:\n\n"
    "Original: {prompt}\n\n"
    "Rewritten:"
)


def read_prompts_txt(path: str) -> List[str]:
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.rstrip("\n")
            if not s.strip():
                continue
            prompts.append(s)
    return prompts


def infer_base_model_id(checkpoint_dir: str) -> Optional[str]:
    cfg_path = os.path.join(checkpoint_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("base_model_name_or_path")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, help="Path to PPO checkpoint directory (e.g. checkpoint-1000)")
    ap.add_argument("--base_model", default="", help="Optional override base model HF id")
    ap.add_argument("--prompts_file", default="", help="Optional .txt file (one prompt per line)")
    ap.add_argument("--prompt", action="append", default=[], help="May be passed multiple times")
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--do_sample", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--system_prompt", default="", help="Optional system prompt override")
    args = ap.parse_args()

    prompts: List[str] = []
    if args.prompts_file:
        prompts.extend(read_prompts_txt(args.prompts_file))
    prompts.extend(args.prompt)
    if not prompts:
        raise SystemExit("No prompts provided. Use --prompts_file or --prompt.")

    ckpt = args.checkpoint_dir
    if not os.path.isdir(ckpt):
        raise SystemExit(f"checkpoint_dir not found or not a directory: {ckpt}")

    base_model_id = args.base_model.strip() or infer_base_model_id(ckpt)
    if not base_model_id:
        raise SystemExit("Could not infer base model id. Pass --base_model explicitly.")

    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Tokenizer: prefer checkpoint dir (it contains chat template + special tokens)
    try:
        tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True, token=hf_token)
    except Exception:
        tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True, token=hf_token)

    # Generation prefers left-padding for decoder-only models
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        token=hf_token,
    )

    model = PeftModel.from_pretrained(base, ckpt)
    model.eval()

    def build_chat(prompt_text: str) -> str:
        user_content = PROMPT_TEMPLATE_INCREASE_REFUSAL.format(prompt=prompt_text)
        messages = []
        if args.system_prompt.strip():
            messages.append({"role": "system", "content": args.system_prompt.strip()})
        messages.append({"role": "user", "content": user_content})
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def rewrite_one(p: str) -> str:
        chat = build_chat(p)
        enc = tok(chat, return_tensors="pt").to(model.device)

        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=bool(int(args.do_sample)),
            pad_token_id=tok.eos_token_id,
        )
        if bool(int(args.do_sample)):
            gen_kwargs["temperature"] = max(1e-6, float(args.temperature))
            gen_kwargs["top_p"] = float(args.top_p)

        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)

        gen = out[0, enc["input_ids"].shape[1] :]
        return tok.decode(gen, skip_special_tokens=True).strip()

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "rewrite"])
        for p in prompts:
            w.writerow([p, rewrite_one(p)])

    print(f"Wrote: {args.output_csv}")


if __name__ == "__main__":
    main()

