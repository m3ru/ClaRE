import argparse
import pandas as pd


def extract_final_response(text: str) -> str:
    """Extract text after 'assistantfinal', filtering out analysis tokens."""
    if pd.isna(text):
        return text
    text = str(text)
    if "assistantfinal" in text:
        return text.split("assistantfinal")[-1].strip()
    return text.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Post-process sanitized CSV to extract only final responses."
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV. Defaults to <input>_cleaned.csv",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="sanitized_prompt",
        help="Column name to process.",
    )

    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = args.input_csv.replace(".csv", "_cleaned.csv")

    df = pd.read_csv(args.input_csv)

    if args.column not in df.columns:
        raise ValueError(f"Column '{args.column}' not found. Available: {list(df.columns)}")

    df[args.column] = df[args.column].apply(extract_final_response)

    df.to_csv(args.output_csv, index=False)
    print(f"Cleaned {len(df)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
