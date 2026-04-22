#!/usr/bin/env python3
"""Autonomous orchestrator for paraphrase generation via OpenAI Batch API.

Runs end-to-end without babysitting:

  1. Retry test batch submission until billing is unblocked (backoff)
  2. Wait for the test batch to complete; sanity-check output
  3. Submit full batches for all 5 datasets (cybermetric, sciq, medqa, pubmedqa, dolly)
  4. Poll every N seconds until all batches are done
  5. Download each, parse, and write paraphrases.json per dataset

Writes a continuous log to <status_dir>/orchestrator.log. A final flag file
<status_dir>/DONE is written on completion (or FAILED on unrecoverable error).
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional

# Reuse helpers from paraphrase_openai_batch.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paraphrase_openai_batch import (  # noqa: E402
    PARAPHRASE_SYSTEM_PROMPT,
    _clean_paraphrase,
    _load_env,
    build_batch_requests,
    download_results,
    load_batch_state,
    load_prompts,
    parse_batch_results,
    save_batch_state,
    save_jsonl,
    submit_batch,
)


DATASETS = ["cybermetric", "sciq", "medqa", "pubmedqa", "dolly"]


def _log(path: str, msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def _build_and_save_requests(dataset_name: str, prompts: List[str],
                             out_dir: str, model: str, n_par: int,
                             temperature: float, top_p: float,
                             max_tokens: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    requests = build_batch_requests(
        prompts, model=model, n_paraphrases=n_par,
        temperature=temperature, top_p=top_p, max_tokens=max_tokens,
    )
    req_path = os.path.join(out_dir, f"batch_requests_{dataset_name}.jsonl")
    save_jsonl(requests, req_path)
    return req_path


def _try_submit(client, req_path: str, log_path: str,
                completion_window: str = "24h"):
    """Returns (file_id, batch_id) on success, raises on unrecoverable error,
    returns None on retryable billing error."""
    try:
        return submit_batch(client, req_path, completion_window=completion_window)
    except Exception as e:
        msg = str(e)
        if "billing_hard_limit_reached" in msg or "billing" in msg.lower():
            _log(log_path, f"[billing-block] {msg[:200]}")
            return None
        raise


def retry_until_unblocked(client, req_path: str, log_path: str,
                          retry_minutes: int, max_hours: float) -> Optional[tuple]:
    """Poll until the test batch can be submitted."""
    start = time.time()
    attempt = 0
    while True:
        attempt += 1
        _log(log_path, f"[retry] attempt #{attempt} — trying test batch submission")
        result = _try_submit(client, req_path, log_path)
        if result is not None:
            _log(log_path, f"[retry] BILLING UNLOCKED on attempt #{attempt}")
            return result
        elapsed_h = (time.time() - start) / 3600
        if elapsed_h >= max_hours:
            _log(log_path, f"[retry] giving up after {elapsed_h:.1f}h")
            return None
        _log(log_path, f"[retry] sleeping {retry_minutes} min "
                      f"(total waited {elapsed_h:.2f}h / {max_hours}h)")
        time.sleep(retry_minutes * 60)


def wait_for_batch(client, batch_id: str, log_path: str, label: str,
                   poll_s: int = 60, max_wait_s: int = 24 * 3600):
    start = time.time()
    last_status = None
    while True:
        batch = client.batches.retrieve(batch_id)
        elapsed = int(time.time() - start)
        counts = getattr(batch, "request_counts", None)
        total = getattr(counts, "total", "?") if counts else "?"
        completed = getattr(counts, "completed", "?") if counts else "?"
        failed = getattr(counts, "failed", "?") if counts else "?"
        if batch.status != last_status or elapsed % 300 < poll_s:
            _log(log_path, f"[{label}] {elapsed}s status={batch.status} "
                          f"completed={completed}/{total} failed={failed}")
            last_status = batch.status
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            return batch
        if elapsed > max_wait_s:
            _log(log_path, f"[{label}] max_wait exceeded ({max_wait_s}s)")
            return batch
        time.sleep(poll_s)


def run_for_dataset(client, dataset_name: str, dataset_jsonl: str,
                    out_root: str, model: str, n_par: int,
                    temperature: float, top_p: float, max_tokens: int,
                    max_prompts: int, poll_s: int, log_path: str) -> str:
    """Submit a batch for one dataset; return batch_id."""
    out_dir = os.path.join(out_root, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    state_path = os.path.join(out_dir, "batch_state.json")
    existing = load_batch_state(state_path)
    if existing and existing.get("batch_id"):
        _log(log_path, f"[{dataset_name}] resuming existing batch "
                      f"{existing['batch_id']}")
        return existing["batch_id"]

    prompts = load_prompts(dataset_jsonl, max_prompts)
    req_path = _build_and_save_requests(
        dataset_name, prompts, out_dir, model, n_par,
        temperature, top_p, max_tokens,
    )
    _log(log_path, f"[{dataset_name}] built {len(prompts)} requests at {req_path}")

    file_id, batch_id = submit_batch(client, req_path, completion_window="24h")
    state = {
        "dataset": dataset_name,
        "model": model,
        "n_paraphrases": n_par,
        "input_file_id": file_id,
        "batch_id": batch_id,
        "n_prompts": len(prompts),
        "submitted_ts": int(time.time()),
    }
    save_batch_state(state, state_path)
    _log(log_path, f"[{dataset_name}] submitted batch={batch_id} file={file_id}")
    return batch_id


def finalize_dataset(client, dataset_name: str, dataset_jsonl: str,
                     out_root: str, batch, max_prompts: int,
                     log_path: str) -> Dict:
    out_dir = os.path.join(out_root, dataset_name)
    state_path = os.path.join(out_dir, "batch_state.json")
    raw_path = os.path.join(out_dir, f"batch_results_{dataset_name}.jsonl")
    paraphrases_path = os.path.join(out_dir, "paraphrases.json")

    state = load_batch_state(state_path) or {}
    state["last_status"] = batch.status

    if batch.status != "completed":
        _log(log_path, f"[{dataset_name}] batch status={batch.status}; not writing output")
        state["error_file_id"] = getattr(batch, "error_file_id", None)
        save_batch_state(state, state_path)
        return {"dataset": dataset_name, "status": batch.status, "ok": False}

    state["output_file_id"] = batch.output_file_id
    state["completed_ts"] = int(time.time())
    save_batch_state(state, state_path)

    download_results(client, batch.output_file_id, raw_path)
    prompts = load_prompts(dataset_jsonl, max_prompts)
    results = parse_batch_results(raw_path, prompts)
    with open(paraphrases_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    counts = [len(r["paraphrases_text"]) for r in results]
    mean_n = sum(counts) / max(len(counts), 1)
    _log(log_path, f"[{dataset_name}] wrote {paraphrases_path} "
                  f"(n_prompts={len(results)} mean_paraphrases={mean_n:.2f})")
    return {
        "dataset": dataset_name,
        "status": batch.status,
        "ok": True,
        "n_prompts": len(results),
        "mean_paraphrases": mean_n,
        "path": paraphrases_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", required=True,
                        help="Directory containing {name}.jsonl files")
    parser.add_argument("--out_root", required=True,
                        help="Where per-dataset outputs live")
    parser.add_argument("--status_dir", required=True,
                        help="Where to write the orchestrator log + DONE flag")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--n_paraphrases", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_prompts", type=int, default=0,
                        help="Cap per dataset (0 = all)")
    parser.add_argument("--test_n", type=int, default=10,
                        help="Size of the initial test batch")
    parser.add_argument("--test_dataset", default="cybermetric")
    parser.add_argument("--retry_minutes", type=int, default=20,
                        help="Minutes between billing retries")
    parser.add_argument("--max_retry_hours", type=float, default=18,
                        help="Give up after this long with billing blocked")
    parser.add_argument("--poll_seconds", type=int, default=60)
    parser.add_argument("--env_file", default=".env")
    parser.add_argument("--skip_test", action="store_true",
                        help="Skip the 10-prompt test (assume billing works)")
    args = parser.parse_args()

    log_path = os.path.join(args.status_dir, "orchestrator.log")
    done_flag = os.path.join(args.status_dir, "DONE")
    fail_flag = os.path.join(args.status_dir, "FAILED")
    os.makedirs(args.status_dir, exist_ok=True)
    _log(log_path, f"=== Orchestrator started, argv={sys.argv} ===")

    _load_env(args.env_file)
    if not os.environ.get("OPENAI_API_KEY"):
        _log(log_path, "[fatal] OPENAI_API_KEY not set")
        with open(fail_flag, "w") as f:
            f.write("missing OPENAI_API_KEY\n")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI()

    try:
        # ---- TEST PHASE ----
        if not args.skip_test:
            _log(log_path, "=== TEST PHASE ===")
            test_ds = args.test_dataset
            test_jsonl = os.path.join(args.datasets_dir, f"{test_ds}.jsonl")
            test_dir = os.path.join(args.out_root, "_test_run")
            os.makedirs(test_dir, exist_ok=True)
            test_state_path = os.path.join(test_dir, "batch_state.json")
            test_prompts = load_prompts(test_jsonl, args.test_n)
            req_path = _build_and_save_requests(
                f"{test_ds}_test", test_prompts, test_dir,
                args.model, args.n_paraphrases,
                args.temperature, args.top_p, args.max_tokens,
            )

            existing = load_batch_state(test_state_path)
            if existing and existing.get("batch_id"):
                _log(log_path, f"[test] resuming existing test batch "
                              f"{existing['batch_id']}")
                test_batch_id = existing["batch_id"]
            else:
                result = retry_until_unblocked(
                    client, req_path, log_path,
                    args.retry_minutes, args.max_retry_hours,
                )
                if result is None:
                    _log(log_path, "[fatal] billing never unblocked")
                    with open(fail_flag, "w") as f:
                        f.write("billing never unblocked\n")
                    sys.exit(2)
                file_id, test_batch_id = result
                save_batch_state({
                    "dataset": f"{test_ds}_test",
                    "model": args.model,
                    "n_paraphrases": args.n_paraphrases,
                    "input_file_id": file_id,
                    "batch_id": test_batch_id,
                    "n_prompts": len(test_prompts),
                    "submitted_ts": int(time.time()),
                }, test_state_path)

            batch = wait_for_batch(client, test_batch_id, log_path, "test",
                                   poll_s=args.poll_seconds)
            if batch.status != "completed":
                _log(log_path, f"[fatal] test batch ended status={batch.status}")
                with open(fail_flag, "w") as f:
                    f.write(f"test batch {batch.status}\n")
                sys.exit(3)

            test_raw = os.path.join(test_dir, f"batch_results_{test_ds}_test.jsonl")
            download_results(client, batch.output_file_id, test_raw)
            test_results = parse_batch_results(test_raw, test_prompts)
            ok_count = sum(1 for r in test_results if len(r["paraphrases_text"]) >= 5)
            _log(log_path, f"[test] {ok_count}/{len(test_results)} prompts have ≥5 "
                          f"valid paraphrases")
            if ok_count < max(1, int(0.5 * len(test_results))):
                _log(log_path, "[fatal] test quality too low — aborting")
                with open(fail_flag, "w") as f:
                    f.write(f"test quality too low ({ok_count}/{len(test_results)})\n")
                sys.exit(4)
            test_paraphrases_path = os.path.join(test_dir, "paraphrases.json")
            with open(test_paraphrases_path, "w", encoding="utf-8") as f:
                json.dump(test_results, f, ensure_ascii=False)

        # ---- SUBMIT FULL ----
        _log(log_path, "=== SUBMIT FULL BATCHES ===")
        batch_ids: Dict[str, str] = {}
        for ds in DATASETS:
            ds_jsonl = os.path.join(args.datasets_dir, f"{ds}.jsonl")
            if not os.path.exists(ds_jsonl):
                _log(log_path, f"[{ds}] SKIP: missing {ds_jsonl}")
                continue
            try:
                bid = run_for_dataset(
                    client, ds, ds_jsonl, args.out_root,
                    args.model, args.n_paraphrases,
                    args.temperature, args.top_p, args.max_tokens,
                    args.max_prompts, args.poll_seconds, log_path,
                )
                batch_ids[ds] = bid
            except Exception as e:
                _log(log_path, f"[{ds}] submission error: {e}")
                _log(log_path, traceback.format_exc())

        if not batch_ids:
            _log(log_path, "[fatal] no batches submitted")
            with open(fail_flag, "w") as f:
                f.write("no batches submitted\n")
            sys.exit(5)

        # ---- POLL + FINALIZE ALL ----
        _log(log_path, "=== POLL + FINALIZE ===")
        summaries = []
        for ds, bid in batch_ids.items():
            ds_jsonl = os.path.join(args.datasets_dir, f"{ds}.jsonl")
            _log(log_path, f"[{ds}] polling batch {bid}")
            batch = wait_for_batch(client, bid, log_path, ds,
                                   poll_s=args.poll_seconds)
            try:
                summary = finalize_dataset(
                    client, ds, ds_jsonl, args.out_root, batch,
                    args.max_prompts, log_path,
                )
                summaries.append(summary)
            except Exception as e:
                _log(log_path, f"[{ds}] finalize error: {e}")
                _log(log_path, traceback.format_exc())
                summaries.append({"dataset": ds, "ok": False, "error": str(e)})

        _log(log_path, "=== DONE ===")
        for s in summaries:
            _log(log_path, f"  {s}")
        with open(done_flag, "w") as f:
            json.dump({"summaries": summaries,
                      "completed_ts": int(time.time())}, f, indent=2)
        _log(log_path, f"wrote {done_flag}")

    except Exception as e:
        _log(log_path, f"[uncaught] {e}")
        _log(log_path, traceback.format_exc())
        with open(fail_flag, "w") as f:
            f.write(f"uncaught: {e}\n")
        sys.exit(99)


if __name__ == "__main__":
    main()
