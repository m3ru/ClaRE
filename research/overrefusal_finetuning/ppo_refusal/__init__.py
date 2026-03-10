"""PPO training for refusal steering.

This package provides tools for training a prompt-rewriting model using
Proximal Policy Optimization (PPO) to either increase or decrease a target
model's refusal probability while maintaining semantic similarity.

Modules:
    config: Configuration dataclasses for all hyperparameters
    data: Data loading utilities (JSONL, CSV, TXT)
    reward_model: RefusalSteeringReward class for computing rewards
    train_ppo: Main PPO training script
"""

from config import (
    FullConfig,
    GenerationConfig,
    LoRAConfig,
    ModelConfig,
    PPOConfig,
    RewardConfig,
    TrainingConfig,
    get_default_config,
    load_config_from_json,
    save_config_to_json,
)
from data import (
    RefusalDataset,
    RefusalDataLoader,
    create_dataloaders,
    create_datasets,
)
from reward_model import RefusalSteeringReward, create_reward_model

__all__ = [
    # Config
    "FullConfig",
    "GenerationConfig",
    "LoRAConfig",
    "ModelConfig",
    "PPOConfig",
    "RewardConfig",
    "TrainingConfig",
    "get_default_config",
    "load_config_from_json",
    "save_config_to_json",
    # Data
    "RefusalDataset",
    "RefusalDataLoader",
    "create_dataloaders",
    "create_datasets",
    # Reward
    "RefusalSteeringReward",
    "create_reward_model",
]
