#!/usr/bin/env python3
"""Generate paraphrases for prompts via the OpenAI Batch API.

Drop-in Phase-1 replacement for paraphrase_and_score.py. Takes a JSONL of
{"prompt": "..."} records, submits a batch request to OpenAI using gpt-4o-mini
with n=N paraphrases per prompt, polls for completion, and writes the
paraphrases in the same _paraphrases.json format that Phase 2 scoring expects:

    [{"prompt_idx": int, "original": str, "paraphrases_text": [str, ...]}, ...]

Usage:
    # Submit and wait to completion:
    python paraphrase_openai_batch.py \
        --data_path datasets/cybermetric.jsonl \
        --output_dir ../paraphrases_api/cybermetric \
        --dataset_name cybermetric

    # Small test run (first 10 prompts):
    python paraphrase_openai_batch.py ... --test_n 10

    # Submit only (don't wait):
    python paraphrase_openai_batch.py ... --submit_only

    # Poll an existing batch by id and save results:
    python paraphrase_openai_batch.py ... --poll_batch_id batch_abc123
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional


PARAPHRASE_SYSTEM_PROMPT = (
    "You are a paraphrasing tool. Given a user prompt, output ONLY a single "
    "paraphrased version that preserves the original meaning but uses different "
    "wording. Do NOT include any analysis, reasoning, thinking, commentary, "
    "labels, or explanation. Do NOT repeat the instructions. Do NOT answer the "
    "question. Output ONLY the paraphrased text and nothing else."
)

# Patterns that indicate the model leaked reasoning into the output
_REASONING_PREFIXES = [
    "analysis",
    "reasoning",
    "thinking",
    "the user",
    "the task",
    "we need to",
    "we should",
    "so we",
    "let me",
    "i need to",
    "paraphrase:",
    "paraphrased version:",
    "output:",
    "result:",
    "here is",
    "here's",
    "sure,",
    "sure!",
    "certainly",
    "of course",
]


def _clean_paraphrase(text: str, original: str) -> str:
    """Strip reasoning preamble that LLMs sometimes prepend."""
    stripped = (text or "").strip()
    if not stripped:
        return ""

    # strip wrapping quotes if model added them
    if (stripped.startswith('"') and stripped.endswith('"')) or \
       (stripped.startswith("'") and stripped.endswith("'")):
        inner = stripped[1:-1].strip()
        if len(inner) >= 15:
            stripped = inner

    lower = stripped.lower()
    has_prefix = any(lower.startswith(p) for p in _REASONING_PREFIXES)

    if has_prefix:
        quoted = re.findall(r'"([^"]{15,})"', stripped)
        if quoted:
            return max(quoted, key=len).strip()

        paragraphs = [p.strip() for p in stripped.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            candidate = paragraphs[-1]
            if not any(candidate.lower().startswith(p) for p in _REASONING_PREFIXES):
                return candidate

        lines = [ln.strip() for ln in stripped.split("\n") if ln.strip()]
        if len(lines) > 1:
            candidate = lines[-1]
            if not any(candidate.lower().startswith(p) for p in _REASONING_PREFIXES):
                return candidate

        orig_words = set(original.lower().split())
        sentences = re.split(r'(?<=[.?!])\s+', stripped)
        for sent in reversed(sentences):
            sent_words = set(sent.lower().split())
            overlap = len(orig_words & sent_words) / max(len(orig_words), 1)
            if overlap > 0.15 and len(sent.split()) > 5:
                return sent.strip()

    return stripped


def _load_env(env_path: str = ".env") -> None:
    """Minimal .env loader (no external dep)."""
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)


def load_prompts(path: str, max_prompts: int = 0) -> List[str]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            p = rec.get("prompt") or rec.get("instruction") or ""
            if p:
                prompts.append(p)
    if max_prompts > 0:
        prompts = prompts[:max_prompts]
    return prompts


def build_batch_requests(
    prompts: List[str],
    model: str,
    n_paraphrases: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> List[Dict]:
    """Produce a list of one request per prompt (n=N paraphrases per request)."""
    requests = []
    for idx, prompt in enumerate(prompts):
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT},
                {"role": "user",
                 "content": f'Please paraphrase the following: "{prompt}"'},
            ],
            "n": n_paraphrases,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        requests.append({
            "custom_id": f"prompt-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        })
    return requests


def save_jsonl(records: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_batch_state(path: str) -> Optional[Dict]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_batch_state(state: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def submit_batch(client, batch_jsonl_path: str, completion_window: str = "24h"):
    """Upload request file and create a batch."""
    print(f"[submit] Uploading {batch_jsonl_path} ...")
    with open(batch_jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"[submit] File uploaded: {file_obj.id}")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
        metadata={"description": os.path.basename(batch_jsonl_path)},
    )
    print(f"[submit] Batch created: {batch.id} (status={batch.status})")
    return file_obj.id, batch.id


def poll_batch(client, batch_id: str, poll_interval: int = 60,
               max_wait_seconds: int = 24 * 3600):
    """Poll until the batch reaches a terminal state."""
    start = time.time()
    while True:
        batch = client.batches.retrieve(batch_id)
        elapsed = int(time.time() - start)
        counts = getattr(batch, "request_counts", None)
        total = getattr(counts, "total", "?") if counts else "?"
        completed = getattr(counts, "completed", "?") if counts else "?"
        failed = getattr(counts, "failed", "?") if counts else "?"
        print(f"[poll] {elapsed}s  status={batch.status}  "
              f"completed={completed}/{total}  failed={failed}")
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            return batch
        if elapsed > max_wait_seconds:
            print(f"[poll] Max wait {max_wait_seconds}s exceeded; returning current state")
            return batch
        time.sleep(poll_interval)


def download_results(client, output_file_id: str, out_path: str) -> None:
    """Download raw batch results to a local JSONL file."""
    print(f"[download] Fetching output file {output_file_id} -> {out_path}")
    resp = client.files.content(output_file_id)
    content = resp.read()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(content)
    size = os.path.getsize(out_path)
    print(f"[download] Saved {size} bytes to {out_path}")


def parse_batch_results(raw_path: str, prompts: List[str]) -> List[Dict]:
    """Map batch response JSONL back to the Phase-2 compatible format."""
    by_custom_id = {}
    n_parse_errors = 0
    n_response_errors = 0
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_parse_errors += 1
                continue
            cid = rec.get("custom_id")
            if rec.get("error"):
                n_response_errors += 1
                by_custom_id[cid] = {"error": rec["error"], "choices": []}
                continue
            body = (rec.get("response") or {}).get("body") or {}
            choices = body.get("choices") or []
            by_custom_id[cid] = {"choices": choices}

    results = []
    for idx, prompt in enumerate(prompts):
        cid = f"prompt-{idx}"
        entry = by_custom_id.get(cid)
        paraphrases_text: List[str] = []
        if entry and entry.get("choices"):
            for choice in entry["choices"]:
                msg = (choice or {}).get("message") or {}
                raw = (msg.get("content") or "").strip()
                cleaned = _clean_paraphrase(raw, prompt)
                if cleaned and len(cleaned.split()) >= 3:
                    paraphrases_text.append(cleaned)
        results.append({
            "prompt_idx": idx,
            "original": prompt,
            "paraphrases_text": paraphrases_text,
        })

    if n_parse_errors:
        print(f"[parse] WARNING: {n_parse_errors} unparseable JSON lines in results")
    if n_response_errors:
        print(f"[parse] WARNING: {n_response_errors} individual request errors")
    n_empty = sum(1 for r in results if not r["paraphrases_text"])
    print(f"[parse] {len(results)} prompts total, {n_empty} with zero paraphrases")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Paraphrase prompts via OpenAI Batch API")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Input JSONL of {\"prompt\": \"...\"} records")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Per-dataset output directory")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Label used in file names (e.g. cybermetric)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--n_paraphrases", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_prompts", type=int, default=0,
                        help="Cap total prompts (0 = all)")
    parser.add_argument("--test_n", type=int, default=0,
                        help="Shortcut: only submit this many prompts")
    parser.add_argument("--completion_window", type=str, default="24h",
                        choices=["24h"])
    parser.add_argument("--poll_interval", type=int, default=60)
    parser.add_argument("--submit_only", action="store_true",
                        help="Submit the batch, save state, and exit")
    parser.add_argument("--poll_batch_id", type=str, default=None,
                        help="Skip submission; just poll this existing batch id and write results")
    parser.add_argument("--env_file", type=str, default=".env")
    args = parser.parse_args()

    _load_env(args.env_file)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set (missing .env or env var)")

    from openai import OpenAI
    client = OpenAI()

    os.makedirs(args.output_dir, exist_ok=True)
    state_path = os.path.join(args.output_dir, "batch_state.json")
    batch_jsonl_path = os.path.join(args.output_dir, f"batch_requests_{args.dataset_name}.jsonl")
    raw_results_path = os.path.join(args.output_dir, f"batch_results_{args.dataset_name}.jsonl")
    paraphrases_path = os.path.join(args.output_dir, "paraphrases.json")

    effective_max = args.test_n if args.test_n > 0 else args.max_prompts
    prompts = load_prompts(args.data_path, effective_max)
    print(f"[data] Loaded {len(prompts)} prompts from {args.data_path}")

    # === Poll-only path ===
    if args.poll_batch_id:
        batch = poll_batch(client, args.poll_batch_id, args.poll_interval)
        if batch.status != "completed":
            print(f"[done] Batch ended in status={batch.status}; no output file")
            state = load_batch_state(state_path) or {}
            state.update({"last_status": batch.status})
            save_batch_state(state, state_path)
            return
        download_results(client, batch.output_file_id, raw_results_path)
        results = parse_batch_results(raw_results_path, prompts)
        with open(paraphrases_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
        print(f"[done] Wrote {paraphrases_path}")
        return

    # === Normal path: build + submit + (optionally) wait ===
    existing = load_batch_state(state_path)
    if existing and existing.get("batch_id") and not args.submit_only:
        print(f"[resume] Found existing batch state: {existing.get('batch_id')}")
        batch_id = existing["batch_id"]
    else:
        requests = build_batch_requests(
            prompts,
            model=args.model,
            n_paraphrases=args.n_paraphrases,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        save_jsonl(requests, batch_jsonl_path)
        print(f"[build] Wrote {len(requests)} requests to {batch_jsonl_path}")

        file_id, batch_id = submit_batch(client, batch_jsonl_path,
                                         completion_window=args.completion_window)
        state = {
            "dataset": args.dataset_name,
            "model": args.model,
            "n_paraphrases": args.n_paraphrases,
            "input_file_id": file_id,
            "batch_id": batch_id,
            "n_prompts": len(prompts),
            "submitted_ts": int(time.time()),
        }
        save_batch_state(state, state_path)

    if args.submit_only:
        print(f"[submit_only] Batch {batch_id} submitted; state saved to {state_path}")
        return

    batch = poll_batch(client, batch_id, args.poll_interval)
    state = load_batch_state(state_path) or {}
    state["last_status"] = batch.status

    if batch.status != "completed":
        print(f"[done] Batch ended in status={batch.status}; no output written")
        state["error_file_id"] = getattr(batch, "error_file_id", None)
        save_batch_state(state, state_path)
        return

    state["output_file_id"] = batch.output_file_id
    state["completed_ts"] = int(time.time())
    save_batch_state(state, state_path)

    download_results(client, batch.output_file_id, raw_results_path)
    results = parse_batch_results(raw_results_path, prompts)
    with open(paraphrases_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"[done] Wrote {paraphrases_path}")

    # quick stats
    counts = [len(r["paraphrases_text"]) for r in results]
    if counts:
        import statistics as st
        print(f"[stats] paraphrases per prompt: "
              f"mean={st.mean(counts):.2f} min={min(counts)} max={max(counts)}")


if __name__ == "__main__":
    main()
