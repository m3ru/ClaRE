#!/usr/bin/env python3
"""
Extract final datasets for refusal classification
- final_refusals.csv: All ML-classified refusals
- final_benign.csv: Cases where both ML and regex agree it's NOT a refusal
"""

import argparse
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

def extract_final_datasets(
    df: pd.DataFrame,
    ml_column: str,
    regex_column: str,
    output_refusals: Path,
    output_benign: Path
):
    """
    Extract final refusal and benign datasets

    Args:
        df: DataFrame with both classification columns
        ml_column: Name of ML classification column
        regex_column: Name of regex classification column
        output_refusals: Path for refusals output
        output_benign: Path for benign output

    Returns:
        Dictionary with statistics
    """
    # Convert to boolean if needed
    ml_refusals = df[ml_column].astype(bool)
    regex_refusals = df[regex_column].astype(bool)

    # Extract ML refusals (all cases where ML says it's a refusal)
    refusal_mask = ml_refusals
    refusals_df = df[refusal_mask].copy()

    # Extract benign (both ML and regex agree it's NOT a refusal)
    benign_mask = (~ml_refusals) & (~regex_refusals)
    benign_df = df[benign_mask].copy()

    # Calculate statistics
    stats = {
        'total_rows': len(df),
        'refusals_count': len(refusals_df),
        'benign_count': len(benign_df),
        'refusals_percentage': (len(refusals_df) / len(df)) * 100,
        'benign_percentage': (len(benign_df) / len(df)) * 100,
        'excluded_count': len(df) - len(refusals_df) - len(benign_df),
        'excluded_percentage': ((len(df) - len(refusals_df) - len(benign_df)) / len(df)) * 100
    }

    # Save datasets
    refusals_df.to_csv(output_refusals, index=False)
    benign_df.to_csv(output_benign, index=False)

    return stats

def main():
    parser = argparse.ArgumentParser(
        description="Extract final refusal and benign datasets"
    )
    parser.add_argument("--input", required=True, help="Input CSV file with both ML and regex classifications")
    parser.add_argument("--output_refusals", default="final_refusals.csv", help="Output CSV file for refusals")
    parser.add_argument("--output_benign", default="final_benign.csv", help="Output CSV file for benign cases")
    parser.add_argument("--ml_column", default="is_refusal", help="Column name for ML refusal classification")
    parser.add_argument("--regex_column", default="regex_refusal", help="Column name for regex refusal classification")

    args = parser.parse_args()

    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file {input_path} not found[/red]")
        return 1

    # Load input data
    console.print(f"[blue]Loading data from {input_path}...[/blue]")
    try:
        df = pd.read_csv(input_path)

        # Check required columns exist
        if args.ml_column not in df.columns:
            console.print(f"[red]Error: ML column '{args.ml_column}' not found in CSV[/red]")
            console.print(f"Available columns: {list(df.columns)}")
            return 1

        if args.regex_column not in df.columns:
            console.print(f"[red]Error: Regex column '{args.regex_column}' not found in CSV[/red]")
            console.print(f"Available columns: {list(df.columns)}")
            return 1

    except Exception as e:
        console.print(f"[red]Error loading CSV: {e}[/red]")
        return 1

    console.print(f"[green]Loaded {len(df)} rows[/green]")

    # Extract datasets
    console.print("\n[blue]Extracting final datasets...[/blue]")

    output_refusals = Path(args.output_refusals)
    output_benign = Path(args.output_benign)

    stats = extract_final_datasets(
        df,
        args.ml_column,
        args.regex_column,
        output_refusals,
        output_benign
    )

    # Display statistics
    table = Table(title="Dataset Extraction Results")
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_column("Percentage", style="green")

    table.add_row("Total Input Rows", f"{stats['total_rows']:,}", "100.0%")
    table.add_row("", "", "")
    table.add_row("Final Refusals (ML=Yes)", f"{stats['refusals_count']:,}", f"{stats['refusals_percentage']:.1f}%")
    table.add_row("Final Benign (ML=No, Regex=No)", f"{stats['benign_count']:,}", f"{stats['benign_percentage']:.1f}%")
    table.add_row("", "", "")
    table.add_row("Excluded (ML=No, Regex=Yes)", f"{stats['excluded_count']:,}", f"{stats['excluded_percentage']:.1f}%")

    console.print(table)

    # Success messages
    console.print(f"\n[bold green]✅ Datasets created successfully![/bold green]")
    console.print(f"[green]Refusals saved to: {output_refusals}[/green]")
    console.print(f"[green]Benign cases saved to: {output_benign}[/green]")

    # Summary
    console.print(f"\n[bold blue]Summary:[/bold blue]")
    console.print(f"• {stats['refusals_count']:,} refusals ({stats['refusals_percentage']:.1f}%)")
    console.print(f"• {stats['benign_count']:,} benign cases ({stats['benign_percentage']:.1f}%)")
    console.print(f"• {stats['excluded_count']:,} cases excluded (ML=No but Regex=Yes) ({stats['excluded_percentage']:.1f}%)")

    return 0


if __name__ == "__main__":
    exit(main())
