#!/usr/bin/env python3
"""Generate OR (over-refusal) adaptations of the UNUSED alpaca-cleaned prompts
with the claude_rwr attacker (Llama-3-8B-Instruct + LoRA).

Reconstructs the 49,020 alpaca prompts that were NOT used for training (seed 43,
n=2500), prompt-iteration (seed 42, n=20), or held-out eval (seed 99, n=200) --
same exclusion logic as eval_held_out.py -- then shards them (SLURM array) and
writes 4 rewrites per prompt at the given temperature.

Out: <output_dir>/gen_alpaca_or_shard{idx:03d}.json
     = [{"alpaca_idx": int, "original": str, "rewrites": [str, ...]}, ...]
Resumable: skips a shard whose output already exists.
"""
import argparse
import json
import os
import random
from typing import Set


def _valid_prompt(ex):
    inst = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    if not inst:
        return None
    p = f"{inst}\n\n{inp}" if inp else inst
    return None if len(p) > 1500 else p


def _pick(ds, N, seed, n) -> Set[int]:
    rng = random.Random(seed)
    idx = list(range(N))
    rng.shuffle(idx)
    out = set()
    for i in idx:
        if _valid_prompt(ds[i]):
            out.add(i)
        if len(out) >= n:
            break
    return out


def remaining_prompts(ds):
    N = len(ds)
    used = _pick(ds, N, 43, 2500) | _pick(ds, N, 42, 20)         # training + iteration
    rng = random.Random(99)   # eval (disjoint)
    idx = list(range(N))
    rng.shuffle(idx)
    ev = set()
    for i in idx:
        if i in used:
            continue
        if _valid_prompt(ds[i]):
            ev.add(i)
        if len(ev) >= 200:
            break
    used |= ev
    out = []
    for i in range(N):
        if i in used:
            continue
        p = _valid_prompt(ds[i])
        if p is not None:
            out.append((i, p))
    return out  # deterministic, sorted by index


def main():
    ap = argparse.ArgumentParser()
    # "none" -> generate with the untrained base model (base-as-attacker control arm)
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--shard_idx", type=int, required=True)
    ap.add_argument("--shuffle_seed", type=int, default=-1,
                    help="if >=0, shuffle the remaining pool with this seed (43 = Sonnet-pool order) "
                         "before skip/cap; matches generate_or_sonnet's sampling order")
    ap.add_argument("--skip_first", type=int, default=0,
                    help="skip the first N of the shuffled pool (6000 = the originals the Sonnet "
                         "training pool already consumed) -> provably-disjoint scale-up pool")
    ap.add_argument("--num_prompts", type=int, default=0,
                    help="cap pool size after skipping (0 = all remaining)")
    ap.add_argument("--n_per_prompt", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"gen_alpaca_or_shard{args.shard_idx:03d}.json")
    if os.path.exists(out_path):
        print(f"[shard {args.shard_idx}] exists, skipping: {out_path}")
        return

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from rwr_config import SYSTEM_PROMPT, PROMPT_TEMPLATE

    hf = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    allp = remaining_prompts(ds)
    if args.shuffle_seed >= 0:
        random.Random(args.shuffle_seed).shuffle(allp)   # match Sonnet-pool sampling order
    if args.skip_first > 0:
        allp = allp[args.skip_first:]                    # drop originals the Sonnet pool consumed
    if args.num_prompts > 0:
        allp = allp[:args.num_prompts]
    # contiguous shard slice
    per = (len(allp) + args.num_shards - 1) // args.num_shards
    s, e = args.shard_idx * per, min(len(allp), (args.shard_idx + 1) * per)
    shard = allp[s:e]
    print(f"[shard {args.shard_idx}/{args.num_shards}] {len(shard)} prompts [{s}:{e}) of {len(allp)} remaining", flush=True)
    if not shard:
        json.dump([], open(out_path, "w"))
        return

    tok = AutoTokenizer.from_pretrained(args.base_model, token=hf)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, token=hf, device_map="auto", torch_dtype=torch.bfloat16)
    if args.adapter_dir.lower() == "none":
        print("[gen] BASE-as-attacker: no LoRA adapter loaded", flush=True)
    else:
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()
    dev = model.device

    def fmt(p):
        return tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": PROMPT_TEMPLATE.format(prompt=p)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def clean(t):
        t = t.strip()
        if len(t) >= 2 and t[0] in "\"'" and t[-1] == t[0]:
            t = t[1:-1].strip()
        return t

    import time
    results = []
    t0 = time.time()
    for i in range(0, len(shard), args.batch_size):
        batch = shard[i:i + args.batch_size]
        texts = [fmt(p) for _, p in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(dev)
        with torch.no_grad():
            gen = model.generate(
                **enc, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                num_return_sequences=args.n_per_prompt, max_new_tokens=args.max_new_tokens,
                min_new_tokens=1, pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        dec = tok.batch_decode(new, skip_special_tokens=True)
        for j, (aidx, orig) in enumerate(batch):
            rw = [clean(dec[j * args.n_per_prompt + k]) for k in range(args.n_per_prompt)]
            results.append({"alpaca_idx": aidx, "original": orig, "rewrites": rw})
        if (i // args.batch_size) % 20 == 0:
            done = i + len(batch)
            rate = done / max(time.time() - t0, 1e-3)
            print(f"[shard {args.shard_idx}] {done}/{len(shard)} prompts  {rate:.2f}/s", flush=True)
            json.dump(results, open(out_path + ".tmp", "w"))

    json.dump(results, open(out_path, "w"))
    if os.path.exists(out_path + ".tmp"):
        os.remove(out_path + ".tmp")
    print(f"[shard {args.shard_idx}] DONE -> {out_path} ({len(results)} prompts, "
          f"{sum(len(r['rewrites']) for r in results)} rewrites)", flush=True)


if __name__ == "__main__":
    main()
