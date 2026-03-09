from __future__ import annotations

import argparse

from .dataset import build_benign_dataset_from_dolly, save_benign_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Build benign prompt CSV from databricks-dolly-15k.jsonl")
    ap.add_argument("--dolly_jsonl", required=True, help="Path to databricks-dolly-15k.jsonl")
    ap.add_argument("--out_csv", required=True, help="Output CSV path (id,prompt,source,category)")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit rows for debugging")
    ap.add_argument("--min_chars", type=int, default=5)
    ap.add_argument("--max_chars", type=int, default=1500)
    ap.add_argument(
        "--include_context",
        type=int,
        default=0,
        help="1 to include Dolly 'context' field in prompts; 0 to drop it (default).",
    )
    args = ap.parse_args()

    rows = build_benign_dataset_from_dolly(
        args.dolly_jsonl,
        limit=args.limit,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        include_context=bool(int(args.include_context)),
    )
    save_benign_csv(rows, args.out_csv)
    print(f"[ok] wrote {len(rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()

