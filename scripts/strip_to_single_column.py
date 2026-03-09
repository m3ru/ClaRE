#!/usr/bin/env python3
"""
Strip a CSV file down to a single column.

Usage:
  python scripts/strip_to_single_column.py --input INPUT.csv --output OUTPUT.csv [--column 0]

Notes:
- The default column index is 0 (first column).
- Handles CSVs with quoted fields and commas inside quotes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strip a CSV to a single column.")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--output", required=True, help="Path to write the single-column CSV"
    )
    parser.add_argument(
        "--column",
        type=int,
        default=0,
        help="Zero-based index of column to keep (default: 0)",
    )
    parser.add_argument(
        "--has-header",
        action="store_true",
        help="Treat the first row as a header and keep only the selected column header",
    )
    return parser.parse_args()


def strip_to_single_column(
    input_path: Path, output_path: Path, column_index: int, has_header: bool
) -> None:
    if column_index < 0:
        raise ValueError("column index must be >= 0")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", newline="", encoding="utf-8") as infile, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        first_row = True
        for row in reader:
            if not row:
                # Skip empty rows
                continue
            if column_index >= len(row):
                # Skip malformed rows that lack the desired column
                continue

            if first_row:
                if has_header:
                    writer.writerow([row[column_index]])
                else:
                    writer.writerow([row[column_index]])
                first_row = False
                continue

            writer.writerow([row[column_index]])


def main() -> None:
    args = parse_args()
    strip_to_single_column(
        input_path=Path(args.input),
        output_path=Path(args.output),
        column_index=args.column,
        has_header=bool(args.has_header),
    )


if __name__ == "__main__":
    main()


