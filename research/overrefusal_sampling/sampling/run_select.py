from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def main() -> None:
    ap = argparse.ArgumentParser(description="Select best over-refusal candidates per benign prompt.")
    ap.add_argument("--in_jsonl_glob", required=True, help='Glob like "outputs/samples_*.jsonl"')
    ap.add_argument("--out_csv", required=True, help="Output CSV path with selected pairs")
    ap.add_argument("--top_k_per_prompt", type=int, default=4)
    ap.add_argument("--min_score", type=float, default=-1e9)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.in_jsonl_glob))
    if not paths:
        raise SystemExit(f"No files match: {args.in_jsonl_glob}")

    grouped: Dict[int, List[dict]] = defaultdict(list)
    total = 0
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["_source_file"] = Path(p).name
                total += 1
                if rec.get("refused"):
                    continue
                if float(rec.get("score", -1e9)) < float(args.min_score):
                    continue
                benign_id = int(rec["benign_id"])
                grouped[benign_id].append(rec)

    rows_out: List[dict] = []
    for benign_id, recs in grouped.items():
        # Deduplicate by normalized text
        seen = set()
        uniq = []
        for r in recs:
            t = (r.get("normalized_overrefusal_prompt") or "").strip()
            if not t:
                continue
            if t in seen:
                continue
            seen.add(t)
            uniq.append(r)

        uniq.sort(key=lambda r: float(r.get("score", -1e9)), reverse=True)
        for r in uniq[: int(args.top_k_per_prompt)]:
            rows_out.append(
                {
                    "benign_id": r["benign_id"],
                    "benign_prompt": r["benign_prompt"],
                    "overrefusal_prompt": r["normalized_overrefusal_prompt"],
                    "score": r["score"],
                    "provider": r["provider"],
                    "model": r["model"],
                    "prompt_variant": r["prompt_variant"],
                    "sample_idx": r["sample_idx"],
                    "source_jsonl": r.get("_source_file", ""),
                }
            )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "benign_id",
            "benign_prompt",
            "overrefusal_prompt",
            "score",
            "provider",
            "model",
            "prompt_variant",
            "sample_idx",
            "source_jsonl",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print(f"[done] read_records={total} prompts={len(grouped)} selected_rows={len(rows_out)} -> {out_path}")


if __name__ == "__main__":
    main()

