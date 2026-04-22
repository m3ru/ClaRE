#!/usr/bin/env python3
"""
Compare refusal classifications between ML model and regex
Outputs disagreements to a separate CSV file for analysis
"""

import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
from rich.console import Console
from rich.progress import Progress, BarColumn, MofNCompleteColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()

def compare_classifications(df: pd.DataFrame, ml_column: str, regex_column: str) -> Dict:
    """
    Compare ML and regex classifications

    Args:
        df: DataFrame with both classification columns
        ml_column: Name of ML classification column
        regex_column: Name of regex classification column

    Returns:
        Dictionary with comparison statistics and disagreement indices
    """
    # Convert to boolean if needed
    ml_refusals = df[ml_column].astype(bool)
    regex_refusals = df[regex_column].astype(bool)

    # Find agreements and disagreements
    agreements = ml_refusals == regex_refusals
    disagreements = ~agreements

    # Specific disagreement types
    ml_yes_regex_no = (ml_refusals == True) & (regex_refusals == False)
    ml_no_regex_yes = (ml_refusals == False) & (regex_refusals == True)

    stats = {
        'total_rows': len(df),
        'agreements': agreements.sum(),
        'disagreements': disagreements.sum(),
        'agreement_rate': (agreements.sum() / len(df)) * 100,
        'disagreement_rate': (disagreements.sum() / len(df)) * 100,
        'ml_yes_regex_no': ml_yes_regex_no.sum(),
        'ml_no_regex_yes': ml_no_regex_yes.sum(),
        'disagreement_indices': df[disagreements].index.tolist(),
        'ml_refusal_rate': (ml_refusals.sum() / len(df)) * 100,
        'regex_refusal_rate': (regex_refusals.sum() / len(df)) * 100
    }

    return stats

def main():
    parser = argparse.ArgumentParser(
        description="Compare ML and regex refusal classifications and extract disagreements"
    )
    parser.add_argument("--input", required=True, help="Input CSV file with both ML and regex classifications")
    parser.add_argument("--output", required=True, help="Output CSV file for disagreements")
    parser.add_argument("--ml_column", default="is_refusal", help="Column name for ML refusal classification")
    parser.add_argument("--regex_column", default="regex_refusal", help="Column name for regex refusal classification")
    parser.add_argument("--include_agreements", action="store_true", help="Also save agreements to a separate file")

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

    # Compare classifications
    console.print("[blue]Comparing classifications...[/blue]")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        task = progress.add_task("Analyzing", total=1)
        stats = compare_classifications(df, args.ml_column, args.regex_column)
        progress.update(task, advance=1)

    # Display comparison statistics
    table = Table(title="Classification Comparison Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_column("Percentage", style="green")

    table.add_row("Total Rows", f"{stats['total_rows']:,}", "100.0%")
    table.add_row("Agreements", f"{stats['agreements']:,}", f"{stats['agreement_rate']:.1f}%")
    table.add_row("Disagreements", f"{stats['disagreements']:,}", f"{stats['disagreement_rate']:.1f}%")
    table.add_row("", "", "")
    table.add_row("ML Refusal Rate", f"{df[args.ml_column].sum():,}", f"{stats['ml_refusal_rate']:.1f}%")
    table.add_row("Regex Refusal Rate", f"{df[args.regex_column].sum():,}", f"{stats['regex_refusal_rate']:.1f}%")
    table.add_row("", "", "")
    table.add_row("ML=Yes, Regex=No", f"{stats['ml_yes_regex_no']:,}", f"{(stats['ml_yes_regex_no']/stats['total_rows'])*100:.1f}%")
    table.add_row("ML=No, Regex=Yes", f"{stats['ml_no_regex_yes']:,}", f"{(stats['ml_no_regex_yes']/stats['total_rows'])*100:.1f}%")

    console.print(table)

    # Extract disagreements
    if stats['disagreements'] > 0:
        disagreement_df = df.iloc[stats['disagreement_indices']].copy()

        # Add comparison columns for clarity
        disagreement_df['ml_classification'] = disagreement_df[args.ml_column]
        disagreement_df['regex_classification'] = disagreement_df[args.regex_column]
        disagreement_df['disagreement_type'] = disagreement_df.apply(
            lambda row: 'ML_yes_Regex_no' if row[args.ml_column] and not row[args.regex_column]
                        else 'ML_no_Regex_yes', axis=1
        )

        # Save disagreements
        output_path = Path(args.output)
        disagreement_df.to_csv(output_path, index=False)

        console.print(f"\n[bold green]✅ Disagreements saved to: {output_path}[/bold green]")
        console.print(f"[yellow]Found {len(disagreement_df)} disagreements out of {len(df)} total rows[/yellow]")

        # Save agreements if requested
        if args.include_agreements:
            agreement_indices = [i for i in range(len(df)) if i not in stats['disagreement_indices']]
            agreement_df = df.iloc[agreement_indices].copy()

            agreement_path = output_path.with_stem(output_path.stem + "_agreements")
            agreement_df.to_csv(agreement_path, index=False)
            console.print(f"[dim]Agreements also saved to: {agreement_path}[/dim]")

    else:
        console.print(f"\n[bold green]🎉 Perfect Agreement![/bold green]")
        console.print("No disagreements found between ML and regex classifications")

        # Create empty disagreements file
        output_path = Path(args.output)
        empty_df = pd.DataFrame(columns=df.columns.tolist() + ['ml_classification', 'regex_classification', 'disagreement_type'])
        empty_df.to_csv(output_path, index=False)
        console.print(f"Empty disagreements file created: {output_path}")

    # Summary
    console.print(f"\n[bold blue]Summary:[/bold blue]")
    console.print(f"Agreement rate: {stats['agreement_rate']:.1f}%")
    console.print(f"Disagreement rate: {stats['disagreement_rate']:.1f}%")

    if stats['disagreements'] > 0:
        console.print(f"\n[yellow]Disagreement breakdown:[/yellow]")
        console.print(f"  ML detected refusal but regex didn't: {stats['ml_yes_regex_no']:,} cases")
        console.print(f"  Regex detected refusal but ML didn't: {stats['ml_no_regex_yes']:,} cases")

    return 0


if __name__ == "__main__":
    exit(main())