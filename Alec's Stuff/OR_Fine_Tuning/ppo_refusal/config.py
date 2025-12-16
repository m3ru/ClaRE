#!/usr/bin/env python3
"""Configuration dataclasses for PPO training on refusal steering.

This module defines all hyperparameters and paths for:
- Model configuration (base model, LoRA checkpoint, refusal vector)
- LoRA configuration (rank, alpha, target modules)
- PPO training configuration (learning rate, batch size, KL target, etc.)
- Reward model configuration (weights, thresholds)
- Training settings (epochs, checkpointing, logging)
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class ModelConfig:
    """Model paths and loading configuration."""

    # Base model (Llama-3.1-8B-Instruct)
    base_model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Path to pre-trained LoRA checkpoint (from SFT phase)
    lora_checkpoint_path: Optional[str] = None

    # Path to refusal vector .npz file
    refusal_vector_path: str = ""

    # Layer to extract activations from (1-based index, middle of network)
    activation_layer: int = 16

    # Use 'max' to auto-select layer with highest L2 norm
    select_layer_strategy: str = "15"  # 0-based index (layer 16 in 1-based)

    # Data type for models
    torch_dtype: str = "bfloat16"

    # Whether to use 4-bit quantization for target model (reward computation)
    quantize_target_model: bool = True

    # HuggingFace token (loaded from env if not set)
    hf_token: Optional[str] = None


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""

    # LoRA rank
    r: int = 32

    # LoRA alpha (scaling factor)
    lora_alpha: int = 64

    # LoRA dropout
    lora_dropout: float = 0.05

    # Bias handling
    bias: str = "none"

    # Task type
    task_type: str = "CAUSAL_LM"

    # Target modules for LoRA (attention + MLP)
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ])


@dataclass
class PPOConfig:
    """PPO algorithm configuration."""

    # Learning rate
    learning_rate: float = 1e-5

    # Batch size (number of prompts per PPO step)
    batch_size: int = 4

    # Mini-batch size for PPO updates
    mini_batch_size: int = 1

    # Gradient accumulation steps
    gradient_accumulation_steps: int = 4

    # Number of PPO epochs per batch
    ppo_epochs: int = 4

    # Target KL divergence (for early stopping within PPO step)
    target_kl: float = 0.1

    # Initial KL coefficient
    init_kl_coef: float = 0.2

    # Adaptive KL control target
    adap_kl_ctrl: bool = True

    # Clip range for PPO
    cliprange: float = 0.2

    # Value function clip range
    cliprange_value: float = 0.2

    # Value function coefficient
    vf_coef: float = 0.1

    # Discount factor
    gamma: float = 1.0

    # GAE lambda
    lam: float = 0.95

    # Whether to whiten rewards
    whiten_rewards: bool = False

    # Score normalization
    use_score_scaling: bool = True
    use_score_norm: bool = True

    # Maximum gradient norm for clipping
    max_grad_norm: float = 1.0


@dataclass
class RewardConfig:
    """Reward model configuration."""

    # Weight for semantic similarity in composite reward
    similarity_weight: float = 0.3

    # Minimum similarity threshold before penalty
    similarity_threshold: float = 0.7

    # Penalty for dropping below similarity threshold
    threshold_penalty: float = -0.5

    # Sentence transformer model for similarity
    sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Scale factor for refusal delta
    refusal_scale: float = 1.0

    # Clamp refusal delta to this range
    refusal_delta_clamp: float = 2.0


@dataclass
class GenerationConfig:
    """Generation configuration for PPO rollouts."""

    # Temperature for sampling
    temperature: float = 0.7

    # Top-p (nucleus) sampling
    top_p: float = 0.9

    # Maximum new tokens to generate
    max_new_tokens: int = 128

    # Minimum new tokens
    min_new_tokens: int = 10

    # Do sampling (vs greedy)
    do_sample: bool = True

    # Top-k sampling (0 = disabled)
    top_k: int = 0

    # Repetition penalty
    repetition_penalty: float = 1.0


@dataclass
class TrainingConfig:
    """Overall training configuration."""

    # Training mode: 'increase_refusal' or 'decrease_refusal'
    mode: Literal["increase_refusal", "decrease_refusal"] = "increase_refusal"

    # Number of training steps (PPO updates)
    num_training_steps: int = 1000

    # Logging frequency (steps)
    log_every: int = 10

    # Evaluation frequency (steps)
    eval_every: int = 50

    # Number of eval samples
    num_eval_samples: int = 32

    # Checkpoint save frequency (steps)
    save_every: int = 100

    # Maximum checkpoints to keep
    save_total_limit: int = 3

    # Output directory
    output_dir: str = "./ppo_checkpoints"

    # Random seed
    seed: int = 42

    # Enable gradient checkpointing
    gradient_checkpointing: bool = True

    # Enable wandb logging
    use_wandb: bool = False

    # Wandb project name
    wandb_project: str = "refusal-ppo"

    # Wandb run name (auto-generated if None)
    wandb_run_name: Optional[str] = None

    # Data path (JSONL file)
    data_path: str = ""

    # Validation split fraction
    val_fraction: float = 0.1

    # Number of dataloader workers
    num_workers: int = 0


@dataclass
class FullConfig:
    """Complete configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "FullConfig":
        """Create config from nested dictionary."""
        return cls(
            model=ModelConfig(**d.get("model", {})),
            lora=LoRAConfig(**d.get("lora", {})),
            ppo=PPOConfig(**d.get("ppo", {})),
            reward=RewardConfig(**d.get("reward", {})),
            generation=GenerationConfig(**d.get("generation", {})),
            training=TrainingConfig(**d.get("training", {})),
        )

    def to_dict(self) -> dict:
        """Convert config to nested dictionary."""
        from dataclasses import asdict
        return asdict(self)


# Prompt templates for different modes
PROMPT_TEMPLATES = {
    "increase_refusal": (
        "Rewrite this prompt to make it seem more suspicious or harmful "
        "while preserving the original intent:\n\n"
        "Original: {prompt}\n\n"
        "Rewritten:"
    ),
    "decrease_refusal": (
        "Rewrite this prompt to make it seem more benign "
        "while preserving the original intent:\n\n"
        "Original: {prompt}\n\n"
        "Rewritten:"
    ),
}


def get_default_config() -> FullConfig:
    """Return default configuration suitable for single H200 GPU."""
    return FullConfig()


def load_config_from_json(path: str) -> FullConfig:
    """Load configuration from JSON file."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return FullConfig.from_dict(d)


def save_config_to_json(config: FullConfig, path: str) -> None:
    """Save configuration to JSON file."""
    import json
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
