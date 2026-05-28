#!/usr/bin/env python3
"""Submit a Claude rewrite-generation job via the Anthropic Message Batches API.

Batch API trade-offs vs the sync async path in `generate_claude_dataset.py`:
  - 50% cheaper per token
  - Each request retried independently → near-zero error rate
  - Async: submit, wait for `processing_status == "ended"`, then download results.
    Typically minutes for small batches, up to 24h max for large ones.
  - Higher max request count (100k per batch) so a single full-dataset run fits.

Workflow:
    1) Build batch requests from (seed_prompt, sample_idx) tuples.
    2) Submit via client.messages.batches.create(...)
    3) Poll status (or pass --batch_id to resume polling a previously-submitted batch).
    4) Stream results back into the same JSONL format as generate_claude_dataset.py
       so downstream `score_candidates.py` consumes it identically.

Usage (submit a new batch):
    ANTHROPIC_API_KEY=sk-... python generate_claude_batch.py \\
        --n_dolly 30 --n_alpaca 30 --n_per_prompt 3 \\
        --variant_name imitation_research_framing \\
        --output_jsonl prompt_iteration_results/dataset_batch_pilot.jsonl \\
        --poll_interval 30

Usage (resume polling/downloading a batch):
    ANTHROPIC_API_KEY=sk-... python generate_claude_batch.py \\
        --batch_id msgbatch_... \\
        --output_jsonl prompt_iteration_results/dataset_batch_pilot.jsonl
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from claude_pilot_variants import VARIANTS, PromptVariant


def load_dolly_seeds(csv_path: str, n: int, seed: int) -> List[Tuple[int, str, str]]:
    rows: List[Tuple[int, str]] = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
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
    from datasets import load_dataset
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    rng = random.Random(seed + 1)
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


def build_batch_requests(
    seeds: List[Tuple[int, str, str]],
    variant: PromptVariant,
    model: str,
    n_per_prompt: int,
    temperature: float,
    max_tokens: int,
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Return (requests, meta_by_custom_id). meta lets us reconstruct the JSONL record
    after the batch finishes (we only get the response, custom_id, and any error back).
    """
    from anthropic.types.messages.batch_create_params import Request
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    requests: List[Dict] = []
    meta: Dict[str, Dict] = {}
    for bid, prompt, source in seeds:
        for sample_idx in range(n_per_prompt):
            custom_id = f"bid{bid}_s{sample_idx}"
            params = MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=variant.system_prompt,
                messages=[
                    {"role": "user", "content": variant.prompt_template.format(prompt=prompt)},
                ],
            )
            requests.append(Request(custom_id=custom_id, params=params))
            meta[custom_id] = {
                "benign_id": bid,
                "benign_prompt": prompt,
                "source": source,
                "sample_idx": sample_idx,
            }
    return requests, meta


def submit_batch(client, requests: List[Dict]) -> str:
    print(f"[batch] submitting {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    print(f"[batch] batch_id={batch.id}  processing_status={batch.processing_status}")
    return batch.id


def poll_until_done(client, batch_id: str, poll_interval: int) -> str:
    """Poll until batch processing_status == 'ended'. Returns final status."""
    while True:
        b = client.messages.batches.retrieve(batch_id)
        rc = b.request_counts
        print(f"[batch] {batch_id}: status={b.processing_status}  "
              f"processing={rc.processing}  succeeded={rc.succeeded}  "
              f"errored={rc.errored}  canceled={rc.canceled}  expired={rc.expired}")
        if b.processing_status == "ended":
            return b.processing_status
        time.sleep(poll_interval)


def download_and_write(client, batch_id: str, meta: Dict[str, Dict],
                       model: str, variant_name: str, output_path: Path) -> None:
    """Stream batch results into JSONL format compatible with score_candidates.py."""
    print(f"[batch] downloading results...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok, n_err = 0, 0
    with output_path.open("a", encoding="utf-8") as f:
        for entry in client.messages.batches.results(batch_id):
            custom_id = entry.custom_id
            m = meta.get(custom_id)
            if m is None:
                # No meta: write minimal record (shouldn't happen if same meta is used
                # for submission + download).
                print(f"[batch] warning: no meta for custom_id={custom_id}")
                continue
            result = entry.result
            text = ""
            error_dict = None
            if result.type == "succeeded":
                msg = result.message
                text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
                n_ok += 1
            else:
                # result.type in {"errored", "canceled", "expired"}
                n_err += 1
                err_obj = getattr(result, "error", None) or getattr(result, result.type, None)
                error_dict = {
                    "type": result.type,
                    "detail": str(err_obj)[:500] if err_obj is not None else None,
                }
            rec = {
                "ts": time.time(),
                "benign_id": m["benign_id"],
                "benign_prompt": m["benign_prompt"],
                "source": m["source"],
                "provider": "anthropic",
                "model": model,
                "prompt_variant": variant_name,
                "sample_idx": m["sample_idx"],
                "overrefusal_prompt": text,
                "normalized_overrefusal_prompt": text,
                "refused": False,
                "error": error_dict,
                "batch_id": batch_id,
                "custom_id": custom_id,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[batch] wrote {output_path}  succeeded={n_ok}  errored={n_err}")


def save_meta(meta: Dict[str, Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(meta, f)


def load_meta(path: Path) -> Dict[str, Dict]:
    with path.open() as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dolly_csv", default="../../overrefusal_sampling/data/benign_dolly.csv")
    ap.add_argument("--n_dolly", type=int, default=30)
    ap.add_argument("--n_alpaca", type=int, default=30)
    ap.add_argument("--n_per_prompt", type=int, default=3)
    ap.add_argument("--variant_name", default="imitation_research_framing",
                    help="Name in claude_pilot_variants.VARIANTS")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=200)
    ap.add_argument("--output_jsonl", default="prompt_iteration_results/dataset_batch_pilot.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--poll_interval", type=int, default=30,
                    help="Seconds between batch-status polls")
    ap.add_argument("--batch_id", default=None,
                    help="If set, skip submit and resume polling/downloading this batch.")
    ap.add_argument("--meta_path", default=None,
                    help="Where to persist per-custom-id seed metadata (default: <output>.meta.json)")
    ap.add_argument("--submit_only", action="store_true",
                    help="Submit batch and exit (don't poll). Use --batch_id later to resume.")
    args = ap.parse_args()

    from anthropic import Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set in environment")
    client = Anthropic(api_key=api_key)

    output_path = Path(args.output_jsonl)
    meta_path = Path(args.meta_path) if args.meta_path else output_path.with_suffix(".meta.json")

    if args.batch_id:
        # Resume an existing batch
        if not meta_path.exists():
            raise SystemExit(f"meta file missing: {meta_path} — needed to reconstruct seed info")
        meta = load_meta(meta_path)
        print(f"[resume] batch_id={args.batch_id}  meta_path={meta_path}  n_meta={len(meta)}")
        poll_until_done(client, args.batch_id, args.poll_interval)
        download_and_write(client, args.batch_id, meta, args.model, args.variant_name, output_path)
        return

    # Fresh submission
    variant = next((v for v in VARIANTS if v.name == args.variant_name), None)
    if variant is None:
        raise SystemExit(f"unknown variant: {args.variant_name}; available: {[v.name for v in VARIANTS]}")

    dolly_seeds  = load_dolly_seeds(args.dolly_csv, args.n_dolly, args.seed)  if args.n_dolly  > 0 else []
    alpaca_seeds = load_alpaca_seeds(args.n_alpaca, args.seed)                if args.n_alpaca > 0 else []
    seeds = dolly_seeds + alpaca_seeds
    if not seeds:
        raise SystemExit("no seed prompts loaded")
    print(f"[seeds] {len(seeds)} total ({len(dolly_seeds)} dolly + {len(alpaca_seeds)} alpaca)")

    requests, meta = build_batch_requests(
        seeds, variant, args.model, args.n_per_prompt, args.temperature, args.max_tokens,
    )
    save_meta(meta, meta_path)
    print(f"[meta] saved {len(meta)} entries to {meta_path}")

    batch_id = submit_batch(client, requests)
    print(f"[batch] submitted. batch_id={batch_id}")
    print(f"[batch] resume with: --batch_id {batch_id} --meta_path {meta_path}")

    if args.submit_only:
        return

    poll_until_done(client, batch_id, args.poll_interval)
    download_and_write(client, batch_id, meta, args.model, args.variant_name, output_path)


if __name__ == "__main__":
    main()
