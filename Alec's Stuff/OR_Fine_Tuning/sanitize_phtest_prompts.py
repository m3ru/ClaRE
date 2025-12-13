import argparse
import os
from typing import List

import pandas as pd
from datasets import load_dataset, Dataset
from transformers import pipeline
from tqdm import tqdm


MODEL_ID = "openai/gpt-oss-120b"
HF_DATASET_ID = "furonghuang-lab/PHTest"


def build_sanitization_prompt(original_prompt: str) -> List[dict]:
    """Construct a chat-style message for gpt-oss-120b to sanitize a prompt.

    The goal is to preserve semantics while making the prompt clearly benign
    and non-refusal-triggering.
    """
    user_content = (
        "You are given a single input prompt that was designed to look potentially "
        "sensitive or harmful, but is actually meant to be benign. "
        "Your task is to REPHRASE the prompt so that:\n"
        "1. It preserves the underlying informational intent and semantics as much as possible.\n"
        "2. It is phrased in a clearly safe, neutral, and benign way that should not trigger safety refusals.\n"
        "3. It does not explicitly request any harmful, abusive, illegal, or policy-violating content.\n"
        "4. It keeps roughly the same level of specificity as the original.\n\n"
        "Original prompt:\n"  # we avoid backticks in case they confuse downstream tooling
        f"{original_prompt}\n\n"
        "Return ONLY the rephrased, sanitized prompt text, with no quotes, no preamble, and no explanation."
    )

    return [
        {"role": "user", "content": user_content},
    ]


def load_pipeline(device_map: str = "auto"):
    """Load the gpt-oss-120b text-generation pipeline.

    This assumes you have enough GPU memory or suitable hardware to host the model.
    """
    text_gen = pipeline(
        "text-generation",
        model=MODEL_ID,
        torch_dtype="auto",
        device_map=device_map,
    )
    return text_gen


def load_prompts_from_hf(split: str, prompt_column: str) -> Dataset:
    """Load PHTest prompts directly from Hugging Face as a Dataset."""
    ds = load_dataset(HF_DATASET_ID, split=split)
    if prompt_column not in ds.column_names:
        raise ValueError(
            f"Column '{prompt_column}' not found in HF dataset '{HF_DATASET_ID}' "
            f"split '{split}'. Available columns: {ds.column_names}"
        )
    return ds


def load_prompts_from_csv(input_path: str, prompt_column: str) -> Dataset:
    """Load prompts from a local CSV file as a Dataset."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if prompt_column not in df.columns:
        raise ValueError(
            f"Column '{prompt_column}' not found in input CSV. "
            f"Available columns: {list(df.columns)}"
        )
    return Dataset.from_pandas(df)


def extract_response(output) -> str:
    """Extract the assistant's response from pipeline output."""
    try:
        last_message = output[0]["generated_text"][-1]
        content = last_message.get("content", "").strip()
        return content
    except Exception:
        return str(output)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sanitize PHTest over-refusal prompts using the openai/gpt-oss-120b model. "
            "You can either read from a local CSV, or load PHTest directly from "
            "Hugging Face and write a CSV with an added sanitized column."
        )
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=False,
        help=(
            "Path to a CSV file containing the PHTest over-refusal prompts. "
            "If omitted, you must pass --use_hf to load PHTest directly from Hugging Face."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=(
            "Where to save the CSV with sanitized prompts. "
            "Defaults to <input_stem>_sanitized.csv in the same directory."
        ),
    )
    parser.add_argument(
        "--prompt_column",
        type=str,
        default="Request",
        help="Name of the column in the CSV that contains the over-refusal prompts.",
    )
    parser.add_argument(
        "--sanitized_column",
        type=str,
        default="sanitized_prompt",
        help="Name of the output column that will contain the sanitized prompts.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate per sanitized prompt.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map argument for the transformers pipeline (e.g. 'auto' or 'cuda:0').",
    )
    parser.add_argument(
        "--use_hf",
        action="store_true",
        help=(
            "If set, load prompts directly from the Hugging Face dataset "
            f"'{HF_DATASET_ID}' instead of a local CSV."
        ),
    )
    parser.add_argument(
        "--hf_split",
        type=str,
        default="train",
        help="Which split of the Hugging Face PHTest dataset to use (e.g. 'train').",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for pipeline inference.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="Save checkpoint every N prompts to avoid losing progress.",
    )

    args = parser.parse_args()

    if args.use_hf:
        # Load directly from Hugging Face
        ds = load_prompts_from_hf(args.hf_split, args.prompt_column)
        if args.output_path is None:
            args.output_path = f"PHTest_{args.hf_split}_sanitized.csv"
    else:
        if not args.input_path:
            parser.error("Either --input_path must be provided or --use_hf must be set.")

        ds = load_prompts_from_csv(args.input_path, args.prompt_column)

        if args.output_path is None:
            base, ext = os.path.splitext(args.input_path)
            args.output_path = base + "_sanitized" + (ext or ".csv")

    text_gen = load_pipeline(device_map=args.device_map)

    # Prepare chat-formatted prompts for the dataset
    def prepare_messages(example):
        return {"messages": build_sanitization_prompt(str(example[args.prompt_column]))}

    ds = ds.map(prepare_messages)

    # Use the pipeline with the dataset for efficient batched inference
    print(f"Processing {len(ds)} prompts with batch_size={args.batch_size}...")

    # Check for existing checkpoint to resume from
    checkpoint_path = args.output_path.replace(".csv", "_checkpoint.csv")
    start_idx = 0
    sanitized_values = []

    if os.path.exists(checkpoint_path):
        checkpoint_df = pd.read_csv(checkpoint_path)
        if args.sanitized_column in checkpoint_df.columns:
            # Count non-null sanitized values to determine resume point
            existing = checkpoint_df[args.sanitized_column].dropna().tolist()
            start_idx = len(existing)
            sanitized_values = existing
            print(f"Resuming from checkpoint at index {start_idx}...")

    if start_idx < len(ds):
        # Only process remaining prompts
        remaining_messages = ds["messages"][start_idx:]

        for idx, output in enumerate(tqdm(
            text_gen(
                remaining_messages,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
            ),
            total=len(remaining_messages),
            desc="Sanitizing prompts",
            initial=0,
        )):
            sanitized_values.append(extract_response([output]))

            # Periodic checkpoint saving
            if (start_idx + idx + 1) % args.save_every == 0:
                # Save partial results
                partial_ds = ds.select(range(len(sanitized_values)))
                partial_ds = partial_ds.add_column(args.sanitized_column, sanitized_values)
                partial_ds = partial_ds.remove_columns(["messages"])
                partial_ds.to_csv(checkpoint_path)
                print(f"\nCheckpoint saved at {len(sanitized_values)} prompts -> {checkpoint_path}")

    # Add results back to dataset and save final output
    ds = ds.add_column(args.sanitized_column, sanitized_values)
    ds = ds.remove_columns(["messages"])  # Remove intermediate column
    ds.to_csv(args.output_path)
    print(f"Wrote sanitized prompts to: {args.output_path}")

    # Clean up checkpoint file if we completed successfully
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Removed checkpoint file: {checkpoint_path}")


if __name__ == "__main__":
    main()
