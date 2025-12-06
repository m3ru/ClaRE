import argparse
import os
from typing import List

import pandas as pd
from datasets import load_dataset
from transformers import pipeline


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


def load_prompts_from_hf(split: str, prompt_column: str) -> pd.DataFrame:
    """Load PHTest prompts directly from Hugging Face as a DataFrame."""
    ds = load_dataset(HF_DATASET_ID, split=split)
    df = ds.to_pandas()
    if prompt_column not in df.columns:
        raise ValueError(
            f"Column '{prompt_column}' not found in HF dataset '{HF_DATASET_ID}' "
            f"split '{split}'. Available columns: {list(df.columns)}"
        )
    return df


def sanitize_prompt(text_gen, prompt: str, max_new_tokens: int = 256) -> str:
    messages = build_sanitization_prompt(prompt)
    outputs = text_gen(messages, max_new_tokens=max_new_tokens)

    # According to the gpt-oss-120b HF examples, the pipeline returns a list
    # where each element has a "generated_text" key containing a list of
    # message dicts; we take the last message's content.
    try:
        last_message = outputs[0]["generated_text"][-1]
        content = last_message.get("content", "").strip()
        return content
    except Exception:
        # Fallback: best-effort string conversion
        return str(outputs)


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
        default="requests",
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

    args = parser.parse_args()

    if args.use_hf:
        # Load directly from Hugging Face
        df = load_prompts_from_hf(args.hf_split, args.prompt_column)
        if args.output_path is None:
            args.output_path = f"PHTest_{args.hf_split}_sanitized.csv"
    else:
        if not args.input_path:
            parser.error("Either --input_path must be provided or --use_hf must be set.")

        if not os.path.exists(args.input_path):
            raise FileNotFoundError(f"Input CSV not found: {args.input_path}")

        df = pd.read_csv(args.input_path)
        if args.prompt_column not in df.columns:
            raise ValueError(
                f"Column '{args.prompt_column}' not found in input CSV. "
                f"Available columns: {list(df.columns)}"
            )

        if args.output_path is None:
            base, ext = os.path.splitext(args.input_path)
            args.output_path = base + "_sanitized" + (ext or ".csv")

    text_gen = load_pipeline(device_map=args.device_map)

    sanitized_values: List[str] = []
    for idx, original_prompt in enumerate(df[args.prompt_column].astype(str).tolist()):
        sanitized = sanitize_prompt(text_gen, original_prompt, max_new_tokens=args.max_new_tokens)
        sanitized_values.append(sanitized)
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} prompts...", flush=True)

    df[args.sanitized_column] = sanitized_values
    df.to_csv(args.output_path, index=False)
    print(f"Wrote sanitized prompts to: {args.output_path}")


if __name__ == "__main__":
    main()
