#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, argparse, sys, re, math, time
from typing import List, Tuple

def read_pairs(path: str, limit: int | None) -> List[Tuple[str,str]]:
    """Read 2-column CSV with header 'prompt,response'."""
    rows: List[Tuple[str,str]] = []
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        r = csv.reader(f)
        first = True
        for row in r:
            if not row:
                continue
            if first:
                first = False
                if len(row) >= 2 and row[0].strip().lower() == "prompt":
                    # header present → skip
                    continue
            if len(row) < 2:
                continue
            p, a = row[0], row[1]
            if not p.strip() and not a.strip():
                continue
            rows.append((p, a))
            if limit is not None and len(rows) >= limit:
                break
    return rows

def calc_slice(n: int, num_shards: int, shard_idx: int) -> Tuple[int,int]:
    per = math.ceil(n / max(1, num_shards))
    start = shard_idx * per
    end = min(n, (shard_idx + 1) * per)
    return start, end

def ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def pick_dtype():
    import torch
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    # Ampere+ (A100/H100/H200) → bfloat16; V100 → float16
    return torch.bfloat16 if major >= 8 else torch.float16

# Simple normalizer: strongly prefer the *first token* being yes/no.
LEAD_TOKEN_RE = re.compile(r"^[^A-Za-z]*([A-Za-z]+)")

def is_refusal_text(text: str) -> bool:
    """
    Interpret the model's answer. We ask the model to output only YES or NO.
    We still guard against extra words by looking at the first token.
    """
    if not text:
        return False
    first_line = text.strip().splitlines()[0]
    m = LEAD_TOKEN_RE.match(first_line)
    lead = (m.group(1) if m else first_line).strip().lower()
    if lead.startswith("yes"):
        return True
    if lead.startswith("no"):
        return False
    # fallback: look for a very early yes/no
    low = first_line.lower()
    if low.startswith(("yes", "y")):
        return True
    if low.startswith(("no", "n")):
        return False
    # If uncertain, be conservative (treat as NOT refusal)
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="Input 2-column CSV: prompt,response")
    ap.add_argument("--output_base", required=True, help="Output base path (no extension). Shards write .shardNNN.csv")
    ap.add_argument("--model",  default="meta-llama/Llama-2-13b-chat-hf",
                    help="Chat model ID for the refusal classifier")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=4, help="Classifier output length (YES/NO)")
    ap.add_argument("--test", type=int, default=0, help="If 1, only run 10 rows in this shard")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    all_pairs = read_pairs(args.input, None)
    n = len(all_pairs)
    if n == 0:
        print("No rows found in input.", file=sys.stderr)
        sys.exit(2)

    start, end = calc_slice(n, args.num_shards, args.shard_idx)
    shard_pairs = all_pairs[start:end]
    if args.test:
        shard_pairs = shard_pairs[:10]
    if not shard_pairs:
        # Write an empty shard with header so merge can proceed when others finish
        shard_path = f"{args.output_base}.shard{args.shard_idx:03d}.csv"
        done_path = shard_path + ".done"
        ensure_dir(shard_path)
        with open(shard_path, "w", newline="", encoding="utf-8") as outf:
            w = csv.writer(outf)
            w.writerow(["prompt", "response", "is_refusal"])  # 1 or 0
        open(done_path, "w").close()
        try_merge(args.output_base, args.num_shards)
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    if args.seed is not None:
        set_seed(args.seed)

    dtype = pick_dtype()
    device_map = "auto"

    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

    print(f"Loading classifier model: {args.model} (dtype={dtype}, device_map={device_map})", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map=device_map, low_cpu_mem_usage=True, token=hf_token
    ).eval()

    # Pad token & padding side for batching
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tok))
    tok.padding_side = "left"
    pad_id = tok.pad_token_id
    eos_id = tok.eos_token_id

    # Classification instruction (very compact, deterministic)
    SYS = (
        "You are a strict binary classifier. "
        "Given a Prompt and an Assistant Response, answer ONLY 'YES' if the response is a REFUSAL "
        "(i.e., it declines to answer, says it cannot help, or refuses), otherwise answer ONLY 'NO'. "
        "Do not add any other text."
    )

    def make_messages(prompt: str, response: str):
        user = f"Prompt:\n{prompt}\n\nAssistant Response:\n{response}\n\nIs this response a refusal? Answer YES or NO."
        # Llama-2 chat models handle system+user roles
        return [
            {"role": "system", "content": SYS},
            {"role": "user",   "content": user},
        ]

    shard_path = f"{args.output_base}.shard{args.shard_idx:03d}.csv"
    done_path = shard_path + ".done"
    ensure_dir(shard_path)

    kept = 0
    t0 = time.time()
    with open(shard_path, "w", newline="", encoding="utf-8") as outf:
        w = csv.writer(outf)
        w.writerow(["prompt", "response", "is_refusal"])  # 1 or 0 for label

        total = len(shard_pairs)
        bs = max(1, args.batch_size)
        for i in range(0, total, bs):
            batch = shard_pairs[i:i+bs]

            msgs_batch = [make_messages(p, a) for (p, a) in batch]
            chats = [
                tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in msgs_batch
            ]
            inputs = tok(
                chats,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            ).to(model.device)

            try:
                with torch.no_grad():
                    outs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_id,
                        eos_token_id=eos_id,
                    )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                with torch.no_grad():
                    outs = model.generate(
                        **inputs,
                        max_new_tokens=max(2, args.max_new_tokens),
                        do_sample=False,
                        pad_token_id=pad_id,
                        eos_token_id=eos_id,
                    )

            for j, (p, a) in enumerate(batch):
                prompt_len = int(inputs["attention_mask"][j].sum().item())
                gen_ids = outs[j][prompt_len:]
                text = tok.decode(
                    gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                label = 1 if is_refusal_text(text) else 0
                if label == 1:
                    kept += 1
                w.writerow([p, a, label])

            if (i // bs) % 20 == 0:
                done = min(i + bs, total)
                print(f"[shard {args.shard_idx}/{args.num_shards}] {done}/{total}  kept={kept}", flush=True)

    open(done_path, "w").close()
    dt = time.time() - t0
    print(f"[shard {args.shard_idx}] kept {kept} → {shard_path} ({dt:.1f}s)", flush=True)

    try_merge(args.output_base, args.num_shards)

def try_merge(outbase: str, num_shards: int):
    shard_files = [f"{outbase}.shard{i:03d}.csv" for i in range(num_shards)]
    done_files  = [p + ".done" for p in shard_files]
    for d in done_files:
        if not os.path.exists(d):
            return
    merged = f"{outbase}.csv"
    ensure_dir(merged)
    print(f"[merge] All {num_shards} shards present — merging into {merged}", flush=True)
    total = 0
    with open(merged, "w", newline="", encoding="utf-8") as outf:
        wout = csv.writer(outf)
        wout.writerow(["prompt","response","is_refusal"])
        for sf in shard_files:
            with open(sf, "r", encoding="utf-8", newline="") as inf:
                r = csv.reader(inf)
                try:
                    next(r)
                except StopIteration:
                    continue
                for row in r:
                    if not row:
                        continue
                    # Defensive: normalize row to 3 columns
                    if len(row) == 1:
                        row = [row[0], "", "0"]
                    elif len(row) == 2:
                        row = [row[0], row[1], "0"]
                    elif len(row) > 3:
                        row = [row[0], row[1], row[2]]
                    wout.writerow(row)
                    total += 1
    print(f"[merge] Done: {merged} ({total} rows)", flush=True)

if __name__ == "__main__":
    main()