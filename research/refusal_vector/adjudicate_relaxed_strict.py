#!/usr/bin/env python3
"""Targeted per-response adjudication of primary strict relaxed-repair candidates."""
import argparse
import json
import os
import time

from judge_overrefusal import SYSTEM, USER_T, parse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key_file", default=os.path.expanduser("~/.anthropic_key"))
    ap.add_argument("--model", default="claude-sonnet-5")
    args = ap.parse_args()
    with open(args.analysis) as f:
        rows = json.load(f)["strict_rows"]
    import anthropic
    with open(args.key_file) as f:
        client = anthropic.Anthropic(api_key=f.read().strip())
    out, usage = [], {"input_tokens": 0, "output_tokens": 0}
    for row in rows:
        judgment = None
        for attempt in range(4):
            message = client.messages.create(
                model=args.model, max_tokens=300, system=SYSTEM, thinking={"type": "disabled"},
                messages=[{"role": "user", "content": USER_T.format(
                    prompt=row["repair"], reply=row["response"])}])
            usage["input_tokens"] += int(message.usage.input_tokens)
            usage["output_tokens"] += int(message.usage.output_tokens)
            text = next((b.text for b in message.content if getattr(b, "type", "") == "text"), "")
            judgment = parse(text)
            if judgment:
                break
            time.sleep(2 ** attempt)
        out.append({"source_id": row["source_id"], "repair_id": row["repair_id"],
                    "repair": row["repair"], "response": row["response"],
                    "primary_label": row["label"], "primary_justified": row["justified"],
                    "adjudication": judgment or {"label": "ERROR"}})
    result = {"n": len(out), "model": args.model, "api_usage": usage, "rows": out}
    tmp = f"{args.out}.tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
