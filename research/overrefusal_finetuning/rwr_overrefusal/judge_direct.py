#!/usr/bin/env python3
"""Direct (non-batch) judge runner.

The Message Batches API is the right tool for the 50k corpus run (50% cheaper), but for a
~70-item calibration set its scheduling latency dominates — minutes of polling for a few
cents of savings. This runs the identical prompt and parser as or_judge_v4.py via concurrent
live calls instead: ~70 items in seconds, so the calibrate -> measure -> revise loop is fast.
"""
import argparse, csv, os, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

csv.field_size_limit(sys.maxsize)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from or_judge_v5 import PREAMBLE, SUFFIX, judge_user, parse_verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fewshot", default="grading/fewshot_v5.txt")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    import anthropic
    client = anthropic.Anthropic()
    system = PREAMBLE + open(a.fewshot).read() + SUFFIX
    rows = list(csv.DictReader(open(a.corpus_csv)))
    print(f"[direct] {len(rows)} items | {a.model} | {a.workers} workers", flush=True)

    def one(r):
        for attempt in range(4):
            try:
                m = client.messages.create(
                    model=a.model, max_tokens=24,
                    # system prompt is identical across every call and ~2k tokens (few-shot
                    # block), so cache it: reads bill at ~0.1x. Matters at corpus scale.
                    system=[{"type": "text", "text": system,
                             "cache_control": {"type": "ephemeral"}}],
                    thinking={"type": "disabled"},
                    messages=[{"role": "user",
                               "content": judge_user(r["original"], r["rewrite"])}])
                txt = "".join(b.text for b in m.content
                              if getattr(b, "type", None) == "text")
                return r["pair_id"], parse_verdict(txt)
            except Exception as e:
                if attempt == 3:
                    print(f"[warn] {r['pair_id']} failed after 4 attempts: "
                          f"{type(e).__name__}: {e}", file=sys.stderr)
                    return r["pair_id"], "UNKNOWN"
                time.sleep(2 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = dict(ex.map(one, rows))

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "intent", "harm", "is_or"])
        for r in rows:
            v = res.get(r["pair_id"], "UNKNOWN")
            if v == "UNKNOWN":
                w.writerow([r["pair_id"], "UNKNOWN", "UNKNOWN", ""])
            else:
                i, h = v.split("/")
                w.writerow([r["pair_id"], i, h,
                            "1" if (i == "SAME" and h == "BENIGN") else "0"])
    print(f"[done] {a.out} | " +
          "  ".join(f"{k}={v}" for k, v in Counter(res.values()).most_common()), flush=True)


if __name__ == "__main__":
    main()
