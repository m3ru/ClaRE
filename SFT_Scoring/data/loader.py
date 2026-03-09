"""Data loading and saving utilities for the SFT scoring pipeline."""

import os
import json
import pandas as pd
from typing import Optional, Dict, Any, List
from pathlib import Path


def load_benign_prompts(path: str) -> pd.DataFrame:
    """
    Load benign prompts from a CSV file.

    Args:
        path: Path to CSV file with 'prompt' column

    Returns:
        DataFrame with prompts
    """
    df = pd.read_csv(path)
    if 'prompt' not in df.columns:
        raise ValueError(f"CSV file must contain 'prompt' column. Found: {list(df.columns)}")
    return df


def load_candidate_transformations(path: str) -> pd.DataFrame:
    """
    Load candidate transformations from a CSV file.

    Args:
        path: Path to CSV file with 'original_prompt' and 'transformed_prompt' columns

    Returns:
        DataFrame with candidate transformations
    """
    df = pd.read_csv(path)
    required_cols = ['original_prompt', 'transformed_prompt']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV file missing required columns: {missing}. Found: {list(df.columns)}")
    return df


def save_scored_candidates(
    df: pd.DataFrame,
    path: str,
    columns: Optional[List[str]] = None
) -> None:
    """
    Save scored candidates to a CSV file.

    Args:
        df: DataFrame with scored candidates
        path: Output path
        columns: Optional list of columns to save (saves all if None)
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if columns:
        df[columns].to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"Saved scored candidates to {path}")


def save_sft_dataset(
    df: pd.DataFrame,
    path: str,
    input_col: str = 'original_prompt',
    output_col: str = 'transformed_prompt'
) -> None:
    """
    Save final SFT dataset with 'input' and 'output' columns.

    Args:
        df: DataFrame with selected candidates
        path: Output path
        input_col: Column name for input (original prompt)
        output_col: Column name for output (transformed prompt)
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    sft_df = pd.DataFrame({
        'input': df[input_col],
        'output': df[output_col]
    })
    sft_df.to_csv(path, index=False)
    print(f"Saved SFT dataset ({len(sft_df)} examples) to {path}")


def get_checkpoint_path(output_dir: str) -> str:
    """Get the path for checkpoint file."""
    return os.path.join(output_dir, 'checkpoint.json')


def save_checkpoint(
    output_dir: str,
    stage: str,
    data: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save a checkpoint with intermediate results.

    Args:
        output_dir: Directory to save checkpoint
        stage: Name of the completed stage (e.g., 'semantic_similarity', 'refusal_activation')
        data: DataFrame with current state
        metadata: Optional metadata dict
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save data
    data_path = os.path.join(output_dir, f'checkpoint_{stage}.csv')
    data.to_csv(data_path, index=False)

    # Save checkpoint info
    checkpoint_info = {
        'stage': stage,
        'data_path': data_path,
        'num_rows': len(data),
        'columns': list(data.columns),
        'metadata': metadata or {}
    }

    checkpoint_path = get_checkpoint_path(output_dir)
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_info, f, indent=2)

    print(f"Checkpoint saved: stage='{stage}', rows={len(data)}")


def load_checkpoint(output_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load the most recent checkpoint if it exists.

    Args:
        output_dir: Directory containing checkpoint

    Returns:
        Dict with 'stage', 'data' (DataFrame), and 'metadata', or None if no checkpoint
    """
    checkpoint_path = get_checkpoint_path(output_dir)

    if not os.path.exists(checkpoint_path):
        return None

    try:
        with open(checkpoint_path, 'r') as f:
            info = json.load(f)

        data_path = info['data_path']
        if not os.path.exists(data_path):
            print(f"Warning: Checkpoint data file not found: {data_path}")
            return None

        data = pd.read_csv(data_path)

        return {
            'stage': info['stage'],
            'data': data,
            'metadata': info.get('metadata', {})
        }
    except Exception as e:
        print(f"Warning: Failed to load checkpoint: {e}")
        return None


def clear_checkpoint(output_dir: str) -> None:
    """Remove checkpoint files."""
    checkpoint_path = get_checkpoint_path(output_dir)
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                info = json.load(f)
            data_path = info.get('data_path')
            if data_path and os.path.exists(data_path):
                os.remove(data_path)
            os.remove(checkpoint_path)
            print("Checkpoint cleared.")
        except Exception as e:
            print(f"Warning: Failed to clear checkpoint: {e}")
