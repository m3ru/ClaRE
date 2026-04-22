#!/usr/bin/env python3
import os, sys, csv, math, argparse, time
from typing import List, Tuple

# ------------------------------
# Input: one prompt per line (no CSV parsing). Output: CSV prompt,response
# Sharded execution across SLURM array; each shard writes shardNNN.csv + .done
# When all .done files exist, an automatic merge creates OUTBASE.csv (once).
# ------------------------------

def read_prompts_txt(path: str) -> List[str]:
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            # Keep line exactly as-is other than stripping the trailing newline
            if line == "":
                continue
            prompts.append(line)
    return prompts

def calc_slice(n: int, num_shards: int, shard_idx: int) -> Tuple[int,int]:
    # ceil-based even splitting so all items are covered
    per = math.ceil(n / num_shards) if num_shards > 0 else n
    start = shard_idx * per
    end   = min(n, (shard_idx + 1) * per)
    return start, end

def ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to prompts .txt or .csv with 1 prompt per line")
    ap.add_argument("--output_base", required=True, help="Output base path (no extension); shards will append .shardXXX.csv")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--do_sample", action="store_true", help="Enable sampling with temperature/top_p")
    ap.add_argument("--test", type=int, default=0, help="If 1, only run 10 lines total in this shard")
    args = ap.parse_args()

    # --- Read all prompts (one per line) ---
    prompts = read_prompts_txt(args.input)
    n = len(prompts)
    if n == 0:
        print("No prompts found. Exiting.", file=sys.stderr)
        sys.exit(2)

    # --- Compute this shard's slice ---
    start, end = calc_slice(n, args.num_shards, args.shard_idx)
    shard_prompts = prompts[start:end]
    if args.test:
        shard_prompts = shard_prompts[:10]

    total_this_shard = len(shard_prompts)
    print(f"[shard {args.shard_idx}/{args.num_shards}] slice [{start}:{end}) => {total_this_shard} prompts (test={args.test})", flush=True)
    if total_this_shard == 0:
        # Still write an empty shard with header so merge logic remains simple
        shard_path = f"{args.output_base}.shard{args.shard_idx:03d}.csv"
        ensure_dir(shard_path)
        with open(shard_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["prompt","response"])
        open(shard_path + ".done", "w").close()
        try_merge(args.output_base, args.num_shards)
        return

    # --- Lazy import heavy deps ---
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # --- Dtype / device ---
    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    device_map = "auto"

    # --- Load model/tokenizer ---
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARNING: HUGGING_FACE_HUB_TOKEN not set; gated models will fail.", file=sys.stderr)

    print(f"Loading model {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, token=hf_token)
    # decoder-only models + generation => left padding helps preserve right context
    tok.padding_side = "left"
    if tok.pad_token is None:
        # Use eos as pad to satisfy padding for batched generation
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype,
        device_map=device_map,
        token=hf_token,
    )

    # --- Output paths for this shard ---
    shard_path = f"{args.output_base}.shard{args.shard_idx:03d}.csv"
    done_path  = shard_path + ".done"
    ensure_dir(shard_path)

    # --- Write header ---
    with open(shard_path, "w", newline="", encoding="utf-8") as outf:
        w = csv.writer(outf)
        w.writerow(["prompt","response"])

        # --- Batch loop ---
        bs = max(1, args.batch_size)
        t0 = time.time()
        written = 0

        # Prebuild the constant system message once
        def build_chat(u: str) -> str:
            msgs = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": u},
            ]
            # Build chat text; we do NOT want any roles in outputs, only assistant text
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        for i in range(0, total_this_shard, bs):
            batch_prompts = shard_prompts[i:i+bs]
            chats = [build_chat(p) for p in batch_prompts]

            enc = tok(
                chats,
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(model.device)

            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=bool(args.do_sample),
                pad_token_id=tok.eos_token_id,
            )
            if args.do_sample:
                gen_kwargs["temperature"] = max(1e-6, args.temperature)
                gen_kwargs["top_p"] = args.top_p

            with torch.no_grad():
                out = model.generate(**enc, **gen_kwargs)

            # For each item in batch: strip off the prompt tokens and decode only the continuation
            input_lens = [enc["input_ids"].shape[1]] * out.shape[0]
            for j, prompt_text in enumerate(batch_prompts):
                gen_ids = out[j][input_lens[j]:]
                text = tok.decode(gen_ids, skip_special_tokens=True)

                # Robust post-process: keep only the first "assistant" turn if template included roles
                # Some chat templates may echo roles. We defensively split on role markers if present.
                cleaned = strip_role_markers(text)

                w.writerow([prompt_text, cleaned])
                written += 1

            if (i // bs) % 10 == 0:
                sofar = min(i + bs, total_this_shard)
                print(f"[shard {args.shard_idx}/{args.num_shards}] {sofar}/{total_this_shard} done", flush=True)

        dt = time.time() - t0
        print(f"[shard {args.shard_idx}] wrote {written} rows → {shard_path} + {done_path}", flush=True)

    # create a sentinel for this shard
    open(done_path, "w").close()

    # Try merging if all shards are done
    try_merge(args.output_base, args.num_shards)

def strip_role_markers(text: str) -> str:
    """
    Defensive cleaning to remove any stray role labels that might appear from tokenization/template.
    We only want the assistant's first turn content. We do *not* alter user text itself.
    """
    # Fast path: common artifacts
    # Remove any leading whitespace/newlines
    s = text.lstrip()

    # If the model happened to literally output "assistant" or "user" or "system" alone,
    # try to drop that bare role token and keep the remainder (if any).
    for role in ("assistant", "user", "system", "Assistant", "User", "System"):
        if s == role:
            return ""  # no content followed

        # If it starts with "<role>\n\n", drop that prefix
        prefix = role + "\n"
        if s.startswith(prefix):
            s = s[len(prefix):].lstrip()

        prefix2 = role + "\n\n"
        if s.startswith(prefix2):
            s = s[len(prefix2):].lstrip()

        # If template used "role: assistant" style:
        label = f"{role}:"
        if s.startswith(label):
            s = s[len(label):].lstrip()

    return s

def try_merge(outbase: str, num_shards: int):
    # Merge only when *all* .done files exist
    shard_files = [f"{outbase}.shard{i:03d}.csv" for i in range(num_shards)]
    done_files  = [p + ".done" for p in shard_files]
    for d in done_files:
        if not os.path.exists(d):
            return  # not ready yet

    merged = f"{outbase}.csv"
    ensure_dir(merged)
    print(f"[merge] All {num_shards} shards present — merging into {merged}", flush=True)

    # Perform a safe merge with one header
    with open(merged, "w", newline="", encoding="utf-8") as outf:
        w_out = csv.writer(outf)
        w_out.writerow(["prompt","response"])
        total = 0
        for sf in shard_files:
            with open(sf, "r", encoding="utf-8", newline="") as inf:
                r = csv.reader(inf)
                try:
                    header = next(r)  # drop header
                except StopIteration:
                    continue
                for row in r:
                    # Guard against malformed rows
                    if not row:
                        continue
                    if len(row) == 1:
                        row = [row[0], ""]
                    elif len(row) > 2:
                        row = [row[0], row[1]]
                    w_out.writerow(row)
                    total += 1
    print(f"[merge] Done: {merged} ({total} rows)", flush=True)

if __name__ == "__main__":
    main()
