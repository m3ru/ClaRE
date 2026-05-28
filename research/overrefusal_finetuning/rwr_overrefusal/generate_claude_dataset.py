#!/usr/bin/env python3
"""Generate a Claude-rewritten dataset for RWR training.

Seeds prompts from Dolly + Alpaca-cleaned, calls Anthropic API async (concurrency
limited), and writes JSONL in the format `score_candidates.py` already consumes —
so the output drops straight into the existing training pipeline.

For the small pilot (500 pairs) we run sync-async from the login node; for the
full run (thousands of pairs) we'd switch to the Message Batches API.

Output JSONL one record per (seed_prompt, sample_idx):
    benign_id (int, unique across corpora)
    benign_prompt (str)
    source ("dolly" | "alpaca")
    provider ("anthropic")
    model (claude model id)
    prompt_variant (variant name)
    sample_idx (0..n_per_prompt-1)
    overrefusal_prompt (raw text)
    normalized_overrefusal_prompt (stripped)
    refused (False)
    error (None | dict)
    ts (epoch seconds)

Usage:
    ANTHROPIC_API_KEY=sk-... python generate_claude_dataset.py \\
        --n_dolly 50 --n_alpaca 50 --n_per_prompt 5 \\
        --output_jsonl pilot/dataset_pilot.jsonl \\
        --variant_name imitation \\
        --concurrency 5
"""
import argparse
import asyncio
import csv
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from claude_pilot_variants import VARIANTS, PromptVariant


def load_dolly_seeds(csv_path: str, n: int, seed: int) -> List[Tuple[int, str, str]]:
    """Return (benign_id, prompt, source) tuples sampled from the Dolly CSV."""
    rows: List[Tuple[int, str]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rid = int(r["id"])
            except (KeyError, ValueError):
                continue
            text = (r.get("prompt") or "").strip()
            if not text or len(text) > 1500:
                continue
            rows.append((rid, text))
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n]
    out = [(rid, text, "dolly") for rid, text in rows]
    print(f"[seeds] dolly: sampled {len(out)} from {csv_path}")
    return out


def load_alpaca_seeds(n: int, seed: int) -> List[Tuple[int, str, str]]:
    """Return (benign_id, prompt, source) tuples from yahma/alpaca-cleaned.

    benign_id namespace: alpaca rows get ids 100_000 + alpaca_index to keep them
    disjoint from Dolly ids (which top out around 15k).
    """
    from datasets import load_dataset
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    rng = random.Random(seed + 1)  # distinct stream from Dolly shuffle
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    out: List[Tuple[int, str, str]] = []
    for idx in indices:
        ex = ds[idx]
        inst = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        if not inst:
            continue
        prompt = f"{inst}\n\n{inp}" if inp else inst
        if len(prompt) > 1500:
            continue
        out.append((100_000 + idx, prompt, "alpaca"))
        if len(out) >= n:
            break
    print(f"[seeds] alpaca: sampled {len(out)}")
    return out


async def call_one(
    client, model: str, variant: PromptVariant, prompt_text: str,
    temperature: float, max_tokens: int, retries: int = 3,
) -> Tuple[str, Dict]:
    """Single async API call. Returns (text, error_dict_or_None)."""
    user_content = variant.prompt_template.format(prompt=prompt_text)
    attempt = 0
    while True:
        try:
            resp = await client.messages.create(
                model=model,
                system=variant.system_prompt,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
            return text, None
        except Exception as e:
            attempt += 1
            if attempt > retries:
                return "", {
                    "type": e.__class__.__name__,
                    "message": str(e)[:500],
                    "traceback": traceback.format_exc(limit=4),
                }
            msg = str(e).lower()
            name = e.__class__.__name__.lower()
            is_rate = "rate" in name or "429" in msg or "rate" in msg
            sleep_s = (5.0 if is_rate else 2.0) * (2 ** (attempt - 1))
            sleep_s = min(sleep_s, 30.0)
            await asyncio.sleep(sleep_s)


async def generate_all(
    seeds: List[Tuple[int, str, str]], variant: PromptVariant,
    model: str, n_per_prompt: int, concurrency: int,
    temperature: float, max_tokens: int, output_path: Path,
) -> None:
    from anthropic import AsyncAnthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set in environment")
    client = AsyncAnthropic(api_key=api_key)

    sem = asyncio.Semaphore(concurrency)
    total = len(seeds) * n_per_prompt
    print(f"[gen] total calls: {total}  concurrency: {concurrency}  model: {model}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Open in append mode so partial runs preserve progress; we de-dup on restart
    # by reading existing records.
    existing_keys = set()
    if output_path.exists():
        with output_path.open("r") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    existing_keys.add((r["benign_id"], r["sample_idx"]))
                except Exception:
                    pass
    if existing_keys:
        print(f"[gen] resuming: {len(existing_keys)} records already in {output_path}")

    f = output_path.open("a", encoding="utf-8")
    lock = asyncio.Lock()
    done_count = [0]
    t0 = time.time()

    async def one_job(bid: int, prompt: str, source: str, sample_idx: int):
        if (bid, sample_idx) in existing_keys:
            return
        async with sem:
            text, error = await call_one(
                client, model, variant, prompt, temperature, max_tokens,
            )
        rec = {
            "ts": time.time(),
            "benign_id": bid,
            "benign_prompt": prompt,
            "source": source,
            "provider": "anthropic",
            "model": model,
            "prompt_variant": variant.name,
            "sample_idx": sample_idx,
            "overrefusal_prompt": text,
            "normalized_overrefusal_prompt": text,
            "refused": False,
            "error": error,
        }
        async with lock:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done_count[0] += 1
            if done_count[0] % 25 == 0 or done_count[0] == total:
                elapsed = time.time() - t0
                rate = done_count[0] / max(elapsed, 1e-6)
                print(f"[gen]   {done_count[0]}/{total} done ({elapsed:.0f}s, {rate:.2f}/s)")

    tasks = []
    for bid, prompt, source in seeds:
        for sample_idx in range(n_per_prompt):
            tasks.append(one_job(bid, prompt, source, sample_idx))

    await asyncio.gather(*tasks)
    f.close()
    print(f"[gen] done. wrote {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dolly_csv", default="../../overrefusal_sampling/data/benign_dolly.csv")
    ap.add_argument("--n_dolly", type=int, default=50)
    ap.add_argument("--n_alpaca", type=int, default=50)
    ap.add_argument("--n_per_prompt", type=int, default=5)
    ap.add_argument("--variant_name", default="imitation",
                    help="Name in claude_pilot_variants.VARIANTS")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=200)
    ap.add_argument("--output_jsonl", default="prompt_iteration_results/dataset_pilot.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    variant = next((v for v in VARIANTS if v.name == args.variant_name), None)
    if variant is None:
        raise SystemExit(f"unknown variant: {args.variant_name}; available: {[v.name for v in VARIANTS]}")

    dolly_seeds  = load_dolly_seeds(args.dolly_csv, args.n_dolly, args.seed) if args.n_dolly > 0 else []
    alpaca_seeds = load_alpaca_seeds(args.n_alpaca, args.seed)               if args.n_alpaca > 0 else []
    seeds = dolly_seeds + alpaca_seeds
    if not seeds:
        raise SystemExit("no seed prompts loaded")
    print(f"[seeds] total seed prompts: {len(seeds)}  ({len(dolly_seeds)} dolly, {len(alpaca_seeds)} alpaca)")

    output_path = Path(args.output_jsonl)
    asyncio.run(generate_all(
        seeds, variant, args.model, args.n_per_prompt, args.concurrency,
        args.temperature, args.max_tokens, output_path,
    ))


if __name__ == "__main__":
    main()
