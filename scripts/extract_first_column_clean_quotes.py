#!/usr/bin/env python3
"""
Extract a single column from a CSV (default: first column) and remove a
single pair of surrounding double quotes from each extracted value when present,
without touching any internal quotes.

Usage:
  python scripts/extract_first_column_clean_quotes.py \
      --input INPUT.csv \
      --output OUTPUT.csv \
      [--column 0] \
      [--has-header]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one column from a CSV and strip a single pair of outer \"\" from values."
        )
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument(
        "--column",
        type=int,
        default=0,
        help="Zero-based index of the column to extract (default: 0)",
    )
    parser.add_argument(
        "--has-header",
        action="store_true",
        help="Treat the first row as a header and keep only that header",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "tsv", "txt"],
        default="csv",
        help="Output format: csv (default), tsv (tab-separated), or txt (one per line)",
    )
    parser.add_argument(
        "--no-quote-output",
        action="store_true",
        help=(
            "When outputting CSV, avoid quoting fields; instead escape delimiters with backslashes."
        ),
    )
    parser.add_argument(
        "--quote-all",
        action="store_true",
        help="When outputting CSV, quote all fields (consistent outer quotes).",
    )
    parser.add_argument(
        "--preserve-outer-quotes",
        action="store_true",
        help="Do not strip existing outer quotes from values before writing.",
    )
    return parser.parse_args()


def strip_outer_double_quotes(value: str) -> str:
    """Remove exactly one leading and one trailing double-quote if both wrap the value.

    This ignores surrounding whitespace when deciding if quotes wrap the entire value,
    and preserves internal quotes without modification.
    """
    if not value:
        return value

    start = 0
    end = len(value) - 1

    # Ignore surrounding whitespace when determining outer quotes
    while start <= end and value[start] in (" ", "\t", "\r", "\n"):
        start += 1
    while end >= start and value[end] in (" ", "\t", "\r", "\n"):
        end -= 1

    if end - start >= 1 and value[start] == '"' and value[end] == '"':
        # Remove the pair of quotes at [start] and [end]
        return value[:start] + value[start + 1 : end] + value[end + 1 :]

    return value


def extract_column(
    input_path: Path,
    output_path: Path,
    column_index: int,
    has_header: bool,
    output_format: str,
    no_quote_output: bool,
    quote_all: bool,
    preserve_outer_quotes: bool,
) -> None:
    if column_index < 0:
        raise ValueError("column index must be >= 0")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate mutually exclusive options
    if output_format == "csv" and no_quote_output and quote_all:
        raise ValueError("--no-quote-output and --quote-all cannot be used together")

    with input_path.open("r", newline="", encoding="utf-8") as infile, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as outfile:
        reader = csv.reader(infile)

        if output_format == "csv":
            if quote_all:
                writer = csv.writer(
                    outfile,
                    delimiter=",",
                    quoting=csv.QUOTE_ALL,
                )
            elif no_quote_output:
                writer = csv.writer(
                    outfile,
                    delimiter=",",
                    quoting=csv.QUOTE_NONE,
                    escapechar="\\",
                )
            else:
                writer = csv.writer(outfile, delimiter=",")
            write_row = lambda v: writer.writerow([v])
        elif output_format == "tsv":
            writer = csv.writer(outfile, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            write_row = lambda v: writer.writerow([v])
        else:  # txt
            def write_row(v: str) -> None:
                # Normalize newlines to spaces to keep one record per line in txt
                normalized = v.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
                outfile.write(normalized + "\n")

        first_row = True
        for row in reader:
            if not row:
                # Skip empty rows
                continue
            if column_index >= len(row):
                # Skip malformed rows lacking the desired column
                continue

            raw_value = row[column_index]
            value = raw_value if preserve_outer_quotes else strip_outer_double_quotes(raw_value)

            if first_row:
                write_row(value if has_header else value)
                first_row = False
                continue

            write_row(value)


def main() -> None:
    args = parse_args()
    extract_column(
        input_path=Path(args.input),
        output_path=Path(args.output),
        column_index=args.column,
        has_header=bool(args.has_header),
        output_format=args.format,
        no_quote_output=bool(args.no_quote_output),
        quote_all=bool(args.quote_all),
        preserve_outer_quotes=bool(args.preserve_outer_quotes),
    )


if __name__ == "__main__":
    main()


