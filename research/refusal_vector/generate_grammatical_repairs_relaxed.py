#!/usr/bin/env python3
"""Generate three original-anchored grammatical repairs for every strict GCG rewrite."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import re
import shutil
import threading
import time


SYSTEM = """You are a careful copy editor in an authorized AI-safety experiment.

Produce exactly three distinct, natural, coherent, fully grammatical standalone user
requests. ORIGINAL is the semantic ground truth: every repair must ask the same benign task
as ORIGINAL and preserve its polarity. REWRITE_TO_REPAIR is a GCG-corrupted version.

Freely delete, replace, reorder, or rewrite corrupted syntax and content as needed to restore
the original benign task. Stay as close to the GCG rewrite's usable wording and structure as
practical. Preserve its LIKELY_TRIGGER change or phrase only when it can be integrated
coherently without changing the original task or making the request substantively harmful;
otherwise omit or safely rephrase it. Do not force malformed or harmful trigger text into a
repair. Do not add a new request, scenario, claim, target, or harmful detail. Do not answer
the request.

All three outputs must be complete and artifact-free: no fragments, dangling ellipses,
unmatched delimiters, placeholder text, repeated punctuation, or commentary. Return only:
{"repairs":["repair 1","repair 2","repair 3"]}
"""

USER_T = """ORIGINAL:
<original>{original}</original>

REWRITE_TO_REPAIR:
<rewrite>{rewrite}</rewrite>

LIKELY_TRIGGER (optional; preserve only if benign and coherent):
<trigger>{trigger}</trigger>

Return exactly three repairs as JSON only."""


def parse(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rsplit("```", 1)[0]
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j < i:
        return None
    try:
        data = json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return None
    repairs = data.get("repairs")
    if not isinstance(repairs, list) or len(repairs) != 3:
        return None
    repairs = [str(x).strip() for x in repairs]
    if any(not x or len(x) > 2000 for x in repairs):
        return None
    normalized = [re.sub(r"\s+", " ", x).casefold() for x in repairs]
    if len(set(normalized)) != 3:
        return None
    for repair in repairs:
        if "..." in repair or "…" in repair or "```" in repair:
            return None
        if re.search(r"([.!?,;:])\1", repair):
            return None
        for left, right in (("(", ")"), ("[", "]"), ("{", "}")):
            if repair.count(left) != repair.count(right):
                return None
    return repairs


def usage_dict(message):
    usage = getattr(message, "usage", None)
    if usage is None:
        return {}
    fields = ("input_tokens", "output_tokens", "cache_creation_input_tokens",
              "cache_read_input_tokens")
    return {field: int(getattr(usage, field, 0) or 0) for field in fields}


def atomic_json(path, value):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", help="Default: OUT.checkpoint.jsonl")
    ap.add_argument("--backup_dir")
    ap.add_argument("--key_file", default=os.path.expanduser("~/.anthropic_key"))
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=800)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--source_ids", help="Comma-separated cfg:idx IDs for smoke tests")
    ap.add_argument("--backup_every", type=int, default=100)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    with open(args.input) as f:
        source = json.load(f)
    rows = source["rows"]
    if args.source_ids:
        wanted = {x.strip() for x in args.source_ids.split(",") if x.strip()}
        rows = [r for r in rows if f"{r['cfg']}:{int(r['idx'])}" in wanted]
        found = {f"{r['cfg']}:{int(r['idx'])}" for r in rows}
        if wanted != found:
            raise ValueError(f"unknown source IDs: {sorted(wanted - found)}")
    if args.limit is not None:
        rows = rows[:args.limit]
    items = [{
        "source_id": f"{r['cfg']}:{int(r['idx'])}", "cfg": r["cfg"], "idx": int(r["idx"]),
        "original": r["original"], "rewrite": r["rewrite"],
        "source_similarity": r.get("similarity"),
        "source_rewrite_response": r.get("rewrite_response", ""),
        "source_rewrite_judgment": r.get("rewrite_judgment", {}),
    } for r in rows]
    print(f"[data] {len(items)} source rewrites; exactly 3 repairs each", flush=True)
    if args.dry_run:
        return

    checkpoint = args.checkpoint or f"{args.out}.checkpoint.jsonl"
    completed = {}
    for path in (args.out, checkpoint):
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                previous = json.load(f).get("rows", []) if path == args.out else [
                    json.loads(line) for line in f if line.strip()]
            for row in previous:
                if len(row.get("repairs", [])) == 3 and not row.get("error"):
                    completed[row["source_id"]] = row
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            print(f"[resume] ignoring unreadable {path}: {e}", flush=True)
    pending = [x for x in items if x["source_id"] not in completed]
    print(f"[resume] {len(completed)} complete; {len(pending)} remaining", flush=True)

    import anthropic
    with open(args.key_file) as f:
        client = anthropic.Anthropic(api_key=f.read().strip())

    def one(item):
        trigger = item["source_rewrite_judgment"].get("trigger", "")
        last_error = "invalid response"
        for attempt in range(6):
            try:
                # Rare corrupted strings trigger a model-level refusal.  Retrying from the
                # benign semantic anchor still satisfies the relaxed repair definition.
                original_only = attempt >= 2
                message = client.messages.create(
                    model=args.model, max_tokens=args.max_tokens, system=SYSTEM,
                    thinking={"type": "disabled"},
                    messages=[{"role": "user", "content": USER_T.format(
                        original=item["original"],
                        rewrite=("[Omitted after editor refusal; restore ORIGINAL.]"
                                 if original_only else item["rewrite"]),
                        trigger=("" if original_only else trigger))}],
                )
                if message.stop_reason == "refusal":
                    last_error = "model refused copy-editing request"
                else:
                    text = next((b.text for b in message.content
                                 if getattr(b, "type", "") == "text"), "")
                    repairs = parse(text)
                    if repairs:
                        return {**item, "repairs": repairs, "repair_model": args.model,
                                "generation_mode": ("original_only_fallback" if original_only
                                                    else "original_plus_rewrite"),
                                "api_usage": usage_dict(message)}
                    last_error = "response failed exact-three/artifact validation"
            except Exception as e:  # noqa: BLE001
                last_error = str(e)[:200]
            if attempt < 5:
                time.sleep(min(30, 2 ** attempt) + random.random())
        return {**item, "repairs": [], "repair_model": args.model, "error": last_error}

    os.makedirs(os.path.dirname(os.path.abspath(checkpoint)), exist_ok=True)
    if args.backup_dir:
        os.makedirs(args.backup_dir, exist_ok=True)
    backup_lock = threading.Lock()

    def backup(path):
        if args.backup_dir and os.path.exists(path):
            with backup_lock:
                shutil.copy2(path, os.path.join(args.backup_dir, os.path.basename(path)))

    errors = 0
    with open(checkpoint, "a") as ckpt, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(one, item): item for item in pending}
        for done, future in enumerate(as_completed(futures), 1):
            row = future.result()
            ckpt.write(json.dumps(row) + "\n")
            ckpt.flush()
            if row.get("error"):
                errors += 1
            else:
                completed[row["source_id"]] = row
            if done % args.backup_every == 0:
                backup(checkpoint)
            if done % 50 == 0 or done == len(pending):
                print(f"[progress] {done}/{len(pending)} new; errors={errors}", flush=True)

    ordered = [completed[x["source_id"]] for x in items if x["source_id"] in completed]
    usage = {field: sum(row.get("api_usage", {}).get(field, 0) for row in ordered)
             for field in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                           "cache_read_input_tokens")}
    result = {"definition": "three original-anchored grammatical repairs per strict rewrite",
              "repair_model": args.model, "n_sources": len(ordered),
              "repairs_per_source": 3, "n_repairs": 3 * len(ordered),
              "api_usage": usage, "rows": ordered}
    atomic_json(args.out, result)
    backup(checkpoint)
    backup(args.out)
    print(f"[done] wrote {len(ordered)} sources / {3 * len(ordered)} repairs; usage={usage}", flush=True)
    if errors or len(ordered) != len(items):
        print(f"[warning] {errors} errors; rerun same command to retry", flush=True)


if __name__ == "__main__":
    main()
