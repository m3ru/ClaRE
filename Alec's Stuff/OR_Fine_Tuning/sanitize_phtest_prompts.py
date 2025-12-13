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
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map=device_map,
    )

    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    return text_gen, tokenizer


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
        default=1,
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

    text_gen, tokenizer = load_pipeline(device_map=args.device_map)

    # Convert to DataFrame for simpler processing
    df = ds.to_pandas()

    # Check for existing checkpoint to resume from
    checkpoint_path = args.output_path.replace(".csv", "_checkpoint.csv")
    start_idx = 0

    if os.path.exists(checkpoint_path):
        checkpoint_df = pd.read_csv(checkpoint_path)
        if args.sanitized_column in checkpoint_df.columns:
            # Count non-null sanitized values to determine resume point
            start_idx = checkpoint_df[args.sanitized_column].notna().sum()
            df = checkpoint_df
            print(f"Resuming from checkpoint at index {start_idx}...")

    if args.sanitized_column not in df.columns:
        df[args.sanitized_column] = None

    print(f"Processing {len(df)} prompts (starting from {start_idx})...")

    for idx in tqdm(range(start_idx, len(df)), desc="Sanitizing prompts"):
        original_prompt = str(df.loc[idx, args.prompt_column])
        messages = build_sanitization_prompt(original_prompt)

        # Apply chat template to get properly formatted prompt
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        outputs = text_gen(
            prompt_text,
            max_new_tokens=args.max_new_tokens,
            return_full_text=False,  # Only return the generated text, not the prompt
        )

        # Extract the generated text
        sanitized = outputs[0]["generated_text"].strip()
        df.loc[idx, args.sanitized_column] = sanitized

        # Periodic checkpoint saving
        if (idx + 1) % args.save_every == 0:
            df.to_csv(checkpoint_path, index=False)
            print(f"\nCheckpoint saved at {idx + 1} prompts -> {checkpoint_path}")

    # Save final output
    df.to_csv(args.output_path, index=False)
    print(f"Wrote sanitized prompts to: {args.output_path}")

    # Clean up checkpoint file if we completed successfully
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Removed checkpoint file: {checkpoint_path}")


if __name__ == "__main__":
    main()
