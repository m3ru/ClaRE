#!/usr/bin/env python3
"""
Main orchestration pipeline for scoring candidate prompt transformations
and selecting top candidates for SFT training.

Usage:
    python pipeline.py --config config.yaml
    python pipeline.py --candidates data/candidates.csv --benign data/benign.csv
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data.loader import (
    load_candidate_transformations,
    save_scored_candidates,
    save_sft_dataset,
    load_checkpoint,
    save_checkpoint,
    clear_checkpoint,
)
from scoring.semantic_similarity import SemanticSimilarityScorer
from scoring.refusal_activation import RefusalActivationScorer
from scoring.harmfulness import HarmfulnessScorer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_default_config() -> Dict[str, Any]:
    """Return default configuration."""
    return {
        # Paths
        "candidates_path": "data/candidates.csv",
        "output_dir": "output",
        "refusal_vector_path": "../Alec's Stuff/Getting_Refusal_Vector/Vector_Extraction/refusal_vector.layer032.npz",

        # Model settings
        "llama_model": "meta-llama/Llama-3.1-8B-Instruct",
        "llama_guard_model": "meta-llama/Llama-Guard-3-8B",
        "hf_token": None,  # Will try to get from environment
        "dtype": "bfloat16",

        # Scoring settings
        "embedding_layer": 16,  # Middle layer for semantic similarity
        "refusal_layer": None,  # Use layer from vector file if None
        "normalization": "minmax",  # 'minmax' or 'zscore_sigmoid'

        # Filtering thresholds
        "harmfulness_threshold": 0.3,  # Reject if above
        "semantic_similarity_threshold": 0.5,  # Reject if below
        "top_k_percent": 5,  # Select top 5%

        # Processing settings
        "batch_size": 8,
        "batch_size_guard": 4,  # Smaller batches for Llama Guard
        "use_checkpoints": True,
        "resume_from_checkpoint": True,
    }


def merge_configs(default: Dict, file_config: Optional[Dict], cli_args: argparse.Namespace) -> Dict[str, Any]:
    """Merge configurations with priority: CLI > file > default."""
    config = default.copy()

    # Merge file config
    if file_config:
        for key, value in file_config.items():
            if value is not None:
                config[key] = value

    # Merge CLI args (only non-None values)
    cli_dict = vars(cli_args)
    for key, value in cli_dict.items():
        if value is not None and key in config:
            config[key] = value

    # Special handling for paths from CLI
    if cli_args.candidates:
        config["candidates_path"] = cli_args.candidates
    if cli_args.output:
        config["output_dir"] = cli_args.output
    if cli_args.refusal_vector:
        config["refusal_vector_path"] = cli_args.refusal_vector

    # Get HF token from environment if not specified
    if config["hf_token"] is None:
        config["hf_token"] = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    return config


def run_pipeline(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Run the full scoring and selection pipeline.

    Args:
        config: Configuration dictionary

    Returns:
        DataFrame with final SFT dataset
    """
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Check for checkpoint
    checkpoint = None
    if config["resume_from_checkpoint"]:
        checkpoint = load_checkpoint(output_dir)
        if checkpoint:
            print(f"Found checkpoint at stage: {checkpoint['stage']}")

    # Determine starting stage
    stages = ["load", "semantic_similarity", "refusal_activation", "harmfulness", "filter_and_rank"]
    start_stage = 0
    df = None

    if checkpoint:
        completed_stage = checkpoint["stage"]
        if completed_stage in stages:
            start_stage = stages.index(completed_stage) + 1
            df = checkpoint["data"]
            print(f"Resuming from stage: {stages[start_stage] if start_stage < len(stages) else 'complete'}")

    # Stage 0: Load data
    if start_stage <= 0:
        print("\n" + "="*60)
        print("STAGE: Loading candidate transformations")
        print("="*60)

        df = load_candidate_transformations(config["candidates_path"])
        print(f"Loaded {len(df)} candidates")

        if config["use_checkpoints"]:
            save_checkpoint(output_dir, "load", df)

    # Stage 1: Semantic similarity
    if start_stage <= 1:
        print("\n" + "="*60)
        print("STAGE: Computing semantic similarity scores")
        print("="*60)

        scorer = SemanticSimilarityScorer(
            model_name=config["llama_model"],
            embedding_layer=config["embedding_layer"],
            dtype=config["dtype"],
            hf_token=config["hf_token"],
        )

        try:
            scores = scorer.score_candidates(
                original_prompts=df["original_prompt"].tolist(),
                transformed_prompts=df["transformed_prompt"].tolist(),
                batch_size=config["batch_size"],
            )
            df["semantic_similarity"] = scores
        finally:
            scorer.unload_model()

        if config["use_checkpoints"]:
            save_checkpoint(output_dir, "semantic_similarity", df)

    # Stage 2: Refusal activation
    if start_stage <= 2:
        print("\n" + "="*60)
        print("STAGE: Computing refusal activation scores")
        print("="*60)

        # Resolve refusal vector path relative to script location if needed
        refusal_vector_path = config["refusal_vector_path"]
        if not os.path.isabs(refusal_vector_path):
            script_dir = Path(__file__).parent
            refusal_vector_path = str(script_dir / refusal_vector_path)

        scorer = RefusalActivationScorer(
            model_name=config["llama_model"],
            refusal_vector_path=refusal_vector_path,
            extraction_layer=config["refusal_layer"],
            dtype=config["dtype"],
            hf_token=config["hf_token"],
            normalization=config["normalization"],
        )

        try:
            scores = scorer.score_candidates(
                transformed_prompts=df["transformed_prompt"].tolist(),
                batch_size=config["batch_size"],
            )
            df["refusal_activation"] = scores
        finally:
            scorer.unload_model()

        if config["use_checkpoints"]:
            save_checkpoint(output_dir, "refusal_activation", df)

    # Stage 3: Harmfulness
    if start_stage <= 3:
        print("\n" + "="*60)
        print("STAGE: Computing harmfulness scores")
        print("="*60)

        scorer = HarmfulnessScorer(
            model_name=config["llama_guard_model"],
            dtype=config["dtype"],
            hf_token=config["hf_token"],
        )

        try:
            scores = scorer.score_candidates_batch(
                prompts=df["transformed_prompt"].tolist(),
                batch_size=config["batch_size_guard"],
            )
            df["harmfulness"] = scores
        finally:
            scorer.unload_model()

        if config["use_checkpoints"]:
            save_checkpoint(output_dir, "harmfulness", df)

    # Stage 4: Filter and rank
    if start_stage <= 4:
        print("\n" + "="*60)
        print("STAGE: Filtering and ranking candidates")
        print("="*60)

        # Apply filters
        harmfulness_threshold = config["harmfulness_threshold"]
        similarity_threshold = config["semantic_similarity_threshold"]
        top_k_percent = config["top_k_percent"]

        # Create passed_filters column
        df["passed_filters"] = (
            (df["harmfulness"] < harmfulness_threshold) &
            (df["semantic_similarity"] > similarity_threshold)
        )

        # Stats before filtering
        n_total = len(df)
        n_passed_harm = (df["harmfulness"] < harmfulness_threshold).sum()
        n_passed_sim = (df["semantic_similarity"] > similarity_threshold).sum()
        n_passed_both = df["passed_filters"].sum()

        print(f"\nFiltering results:")
        print(f"  Total candidates: {n_total}")
        print(f"  Passed harmfulness filter (<{harmfulness_threshold}): {n_passed_harm} ({100*n_passed_harm/n_total:.1f}%)")
        print(f"  Passed similarity filter (>{similarity_threshold}): {n_passed_sim} ({100*n_passed_sim/n_total:.1f}%)")
        print(f"  Passed both filters: {n_passed_both} ({100*n_passed_both/n_total:.1f}%)")

        # Save all scored candidates
        scored_path = os.path.join(output_dir, "scored_candidates.csv")
        save_scored_candidates(
            df, scored_path,
            columns=["original_prompt", "transformed_prompt", "semantic_similarity",
                     "refusal_activation", "harmfulness", "passed_filters"]
        )

        # Filter and rank
        df_filtered = df[df["passed_filters"]].copy()

        if len(df_filtered) == 0:
            print("\nWARNING: No candidates passed the filters!")
            print("Consider adjusting thresholds:")
            print(f"  - Increase harmfulness_threshold (current: {harmfulness_threshold})")
            print(f"  - Decrease semantic_similarity_threshold (current: {similarity_threshold})")
            return df

        # Sort by refusal activation (descending)
        df_filtered = df_filtered.sort_values("refusal_activation", ascending=False)

        # Select top k%
        n_select = max(1, int(len(df_filtered) * top_k_percent / 100))
        df_top = df_filtered.head(n_select)

        print(f"\nSelection results:")
        print(f"  Candidates after filtering: {len(df_filtered)}")
        print(f"  Top {top_k_percent}% selected: {n_select}")
        print(f"  Refusal activation range: [{df_top['refusal_activation'].min():.4f}, {df_top['refusal_activation'].max():.4f}]")

        # Save final SFT dataset
        sft_path = os.path.join(output_dir, "sft_dataset.csv")
        save_sft_dataset(df_top, sft_path)

        # Clear checkpoint on successful completion
        if config["use_checkpoints"]:
            clear_checkpoint(output_dir)

        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"Outputs saved to: {output_dir}")
        print(f"  - scored_candidates.csv: All {n_total} candidates with scores")
        print(f"  - sft_dataset.csv: Top {n_select} candidates for SFT training")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Score candidate prompt transformations and select top candidates for SFT training."
    )

    # Config file
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML configuration file"
    )

    # Path arguments
    parser.add_argument(
        "--candidates",
        type=str,
        help="Path to candidates CSV (columns: original_prompt, transformed_prompt)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory"
    )
    parser.add_argument(
        "--refusal-vector",
        type=str,
        help="Path to refusal vector .npz file"
    )

    # Model arguments
    parser.add_argument(
        "--llama-model",
        type=str,
        dest="llama_model",
        help="Llama 8B model name/path"
    )
    parser.add_argument(
        "--guard-model",
        type=str,
        dest="llama_guard_model",
        help="Llama Guard model name/path"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        dest="hf_token",
        help="HuggingFace token"
    )

    # Threshold arguments
    parser.add_argument(
        "--harmfulness-threshold",
        type=float,
        dest="harmfulness_threshold",
        help="Harmfulness threshold (reject if above)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        dest="semantic_similarity_threshold",
        help="Semantic similarity threshold (reject if below)"
    )
    parser.add_argument(
        "--top-k-percent",
        type=float,
        dest="top_k_percent",
        help="Percentage of top candidates to select"
    )

    # Processing arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        help="Batch size for model inference"
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpointing"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing checkpoints and start fresh"
    )

    args = parser.parse_args()

    # Load and merge configurations
    default_config = get_default_config()
    file_config = None
    if args.config:
        file_config = load_config(args.config)

    config = merge_configs(default_config, file_config, args)

    # Handle checkpoint flags
    if args.no_checkpoint:
        config["use_checkpoints"] = False
    if args.fresh:
        config["resume_from_checkpoint"] = False

    # Print configuration
    print("="*60)
    print("SFT Scoring Pipeline")
    print("="*60)
    print("\nConfiguration:")
    for key, value in sorted(config.items()):
        if key == "hf_token" and value:
            print(f"  {key}: ***")
        else:
            print(f"  {key}: {value}")

    # Check for required files
    if not os.path.exists(config["candidates_path"]):
        print(f"\nERROR: Candidates file not found: {config['candidates_path']}")
        sys.exit(1)

    # Run pipeline
    run_pipeline(config)


if __name__ == "__main__":
    main()
