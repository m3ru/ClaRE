#!/usr/bin/env python3
"""PPO training script for refusal steering.

This script trains a prompt-rewriting model using PPO to either increase or decrease
a target model's refusal probability while maintaining semantic similarity.

Key features:
- Uses TRL's PPOTrainer with AutoModelForCausalLMWithValueHead
- Loads existing LoRA checkpoint and continues training
- Computes rewards using refusal vector projection + semantic similarity
- Supports both increase_refusal and decrease_refusal modes
- Memory optimized for single H200 GPU (80GB)
- Supports wandb logging (optional)

Usage:
    python train_ppo.py --config config.json
    python train_ppo.py --data_path prompts.jsonl --refusal_vector_path vector.npz
"""

import argparse
import gc
import json
import os
import sys
from dataclasses import asdict
from typing import Dict, List, Optional

import torch

# Local imports (add parent to path if needed)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    FullConfig,
    GenerationConfig,
    LoRAConfig,
    ModelConfig,
    PPOConfig,
    RewardConfig,
    TrainingConfig,
    PROMPT_TEMPLATES,
    get_default_config,
    load_config_from_json,
    save_config_to_json,
)
from data import create_dataloaders, format_prompts_for_model
from reward_model import RefusalSteeringReward


def clear_cuda_cache():
    """Clear CUDA cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def setup_wandb(config: TrainingConfig) -> Optional["wandb.run"]:
    """Initialize wandb logging if enabled.

    Args:
        config: Training configuration

    Returns:
        wandb run object or None if disabled
    """
    if not config.use_wandb:
        return None

    try:
        import wandb

        run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=asdict(config) if hasattr(config, "__dataclass_fields__") else config,
        )
        return run
    except ImportError:
        print("[wandb] wandb not installed, disabling logging")
        return None
    except Exception as e:
        print(f"[wandb] Failed to initialize: {e}")
        return None


def load_policy_model(
    model_config: ModelConfig,
    lora_config: LoRAConfig,
    ppo_config: PPOConfig,
    training_config: TrainingConfig,
):
    """Load the policy model with LoRA adapters and value head.

    Args:
        model_config: Model paths and settings
        lora_config: LoRA configuration
        ppo_config: PPO hyperparameters
        training_config: Training settings

    Returns:
        Tuple of (model, tokenizer, peft_config)
    """
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig as PeftLoraConfig, get_peft_model, PeftModel
    from trl import AutoModelForCausalLMWithValueHead
    from huggingface_hub import login

    print("[model] Loading policy model...")

    hf_token = model_config.hf_token or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    ) or os.environ.get("HF_TOKEN")

    # Login first to authenticate all subsequent HF API calls
    if hf_token:
        login(token=hf_token)
        print(f"[model] Authenticated with HuggingFace (token: {hf_token[:10]}...)")
    else:
        print("[model] WARNING: No HF token found! Gated models will fail to load.")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.base_model_id,
        use_fast=True,
        token=hf_token,
    )
    tokenizer.padding_side = "left"  # For generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine dtype
    if model_config.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif model_config.torch_dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Create PEFT config
    peft_config = PeftLoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        task_type=lora_config.task_type,
        target_modules=lora_config.target_modules,
    )

    # Check if we have an existing LoRA checkpoint to load
    if model_config.lora_checkpoint_path and os.path.exists(model_config.lora_checkpoint_path):
        print(f"[model] Loading from existing LoRA checkpoint: {model_config.lora_checkpoint_path}")

        # Load base model with value head, then load LoRA weights
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_config.base_model_id,
            token=hf_token,
            torch_dtype=torch_dtype,
            device_map="auto",
            peft_config=peft_config,
        )

        # Load the pre-trained LoRA weights
        # The model already has LoRA initialized, we need to load the weights
        try:
            from peft import set_peft_model_state_dict
            import safetensors.torch

            # Try loading adapter weights
            adapter_path = os.path.join(model_config.lora_checkpoint_path, "adapter_model.safetensors")
            if os.path.exists(adapter_path):
                adapter_weights = safetensors.torch.load_file(adapter_path)
                set_peft_model_state_dict(model.pretrained_model, adapter_weights)
                print(f"[model] Loaded LoRA weights from {adapter_path}")
            else:
                # Try .bin format
                adapter_path = os.path.join(model_config.lora_checkpoint_path, "adapter_model.bin")
                if os.path.exists(adapter_path):
                    adapter_weights = torch.load(adapter_path, map_location="cpu")
                    set_peft_model_state_dict(model.pretrained_model, adapter_weights)
                    print(f"[model] Loaded LoRA weights from {adapter_path}")
                else:
                    print(f"[model] Warning: No adapter weights found at {model_config.lora_checkpoint_path}")
        except Exception as e:
            print(f"[model] Warning: Could not load LoRA weights: {e}")
            print("[model] Starting with fresh LoRA initialization")
    else:
        print("[model] Initializing fresh LoRA adapters")
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_config.base_model_id,
            token=hf_token,
            torch_dtype=torch_dtype,
            device_map="auto",
            peft_config=peft_config,
        )

    # Enable gradient checkpointing if requested
    if training_config.gradient_checkpointing:
        model.pretrained_model.gradient_checkpointing_enable()
        model.pretrained_model.config.use_cache = False

    print(f"[model] Policy model loaded with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

    return model, tokenizer, peft_config


def create_reference_model(
    model_config: ModelConfig,
    peft_config,
):
    """Create a frozen reference model for KL penalty computation.

    The reference model shares base weights but has separate LoRA adapters.
    It remains frozen during training.

    Args:
        model_config: Model configuration
        peft_config: PEFT configuration

    Returns:
        Reference model (frozen)
    """
    from trl import AutoModelForCausalLMWithValueHead

    print("[model] Creating reference model...")

    hf_token = model_config.hf_token or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    ) or os.environ.get("HF_TOKEN")

    if model_config.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif model_config.torch_dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_config.base_model_id,
        token=hf_token,
        torch_dtype=torch_dtype,
        device_map="auto",
        peft_config=peft_config,
    )

    # Load LoRA checkpoint if available (same as policy initial state)
    if model_config.lora_checkpoint_path and os.path.exists(model_config.lora_checkpoint_path):
        try:
            from peft import set_peft_model_state_dict
            import safetensors.torch

            adapter_path = os.path.join(model_config.lora_checkpoint_path, "adapter_model.safetensors")
            if os.path.exists(adapter_path):
                adapter_weights = safetensors.torch.load_file(adapter_path)
                set_peft_model_state_dict(ref_model.pretrained_model, adapter_weights)
            else:
                adapter_path = os.path.join(model_config.lora_checkpoint_path, "adapter_model.bin")
                if os.path.exists(adapter_path):
                    adapter_weights = torch.load(adapter_path, map_location="cpu")
                    set_peft_model_state_dict(ref_model.pretrained_model, adapter_weights)
        except Exception as e:
            print(f"[model] Warning: Could not load reference LoRA weights: {e}")

    # Freeze reference model
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    print("[model] Reference model created (frozen)")
    return ref_model


def create_ppo_trainer(
    model,
    ref_model,
    tokenizer,
    ppo_config: PPOConfig,
    training_config: TrainingConfig,
):
    """Create PPO trainer with configured hyperparameters.

    Args:
        model: Policy model
        ref_model: Reference model
        tokenizer: Tokenizer
        ppo_config: PPO hyperparameters
        training_config: Training settings

    Returns:
        PPOTrainer instance
    """
    from trl import PPOConfig as TRLPPOConfig, PPOTrainer

    print("[trainer] Creating PPO trainer...")

    # Build config with only supported parameters for newer TRL versions
    # The API changed significantly - some params were renamed/removed
    config_kwargs = {
        "learning_rate": ppo_config.learning_rate,
        "batch_size": ppo_config.batch_size,
        "mini_batch_size": ppo_config.mini_batch_size,
        "gradient_accumulation_steps": ppo_config.gradient_accumulation_steps,
        "seed": training_config.seed,
    }

    # Try to add optional parameters that may or may not exist in this TRL version
    optional_params = {
        "ppo_epochs": ppo_config.ppo_epochs,  # Older TRL
        "num_ppo_epochs": ppo_config.ppo_epochs,  # Newer TRL
        "target_kl": ppo_config.target_kl,
        "init_kl_coef": ppo_config.init_kl_coef,
        "cliprange": ppo_config.cliprange,
        "cliprange_value": ppo_config.cliprange_value,
        "vf_coef": ppo_config.vf_coef,
        "gamma": ppo_config.gamma,
        "lam": ppo_config.lam,
        "max_grad_norm": ppo_config.max_grad_norm,
    }

    # Inspect PPOConfig to see which parameters it accepts
    import inspect
    try:
        sig = inspect.signature(TRLPPOConfig.__init__)
        valid_params = set(sig.parameters.keys())
        for param, value in optional_params.items():
            if param in valid_params:
                config_kwargs[param] = value
    except Exception:
        # Fallback: just try the new API parameter names
        config_kwargs["num_ppo_epochs"] = ppo_config.ppo_epochs

    # Add wandb logging if enabled
    if training_config.use_wandb:
        config_kwargs["log_with"] = "wandb"

    config = TRLPPOConfig(**config_kwargs)

    trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    print("[trainer] PPO trainer created")
    return trainer


def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    generation_config: GenerationConfig,
) -> List[str]:
    """Generate responses from the policy model.

    Args:
        model: Policy model
        tokenizer: Tokenizer
        prompts: List of formatted prompt strings
        generation_config: Generation settings

    Returns:
        List of generated response strings
    """
    # Tokenize
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    input_ids = inputs["input_ids"].to(model.pretrained_model.device)
    attention_mask = inputs["attention_mask"].to(model.pretrained_model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=generation_config.max_new_tokens,
            min_new_tokens=generation_config.min_new_tokens,
            do_sample=generation_config.do_sample,
            temperature=max(1e-6, generation_config.temperature),
            top_p=generation_config.top_p,
            top_k=generation_config.top_k if generation_config.top_k > 0 else None,
            repetition_penalty=generation_config.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part
    responses = []
    for i, output in enumerate(outputs):
        prompt_len = input_ids[i].shape[0]
        generated = output[prompt_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        responses.append(text)

    return responses


def evaluate(
    model,
    tokenizer,
    reward_model: RefusalSteeringReward,
    eval_loader,
    generation_config: GenerationConfig,
    training_config: TrainingConfig,
) -> Dict[str, float]:
    """Run evaluation on held-out data.

    Args:
        model: Policy model
        tokenizer: Tokenizer
        reward_model: Reward model
        eval_loader: Evaluation data loader
        generation_config: Generation settings
        training_config: Training configuration

    Returns:
        Dict of evaluation metrics
    """
    model.eval()

    all_rewards = []
    all_refusal_deltas = []
    all_similarities = []

    num_samples = 0

    for batch in eval_loader:
        if num_samples >= training_config.num_eval_samples:
            break

        # Get original prompts and modes
        original_prompts = [item["prompt"] for item in batch]
        modes = [item["mode"] for item in batch]

        # Format prompts
        formatted = format_prompts_for_model(batch, tokenizer)

        # Generate rewrites
        rewrites = generate_responses(model, tokenizer, formatted, generation_config)

        # Compute rewards (using first mode in batch for simplicity)
        mode = modes[0] if modes else training_config.mode
        results = reward_model.compute_rewards(original_prompts, rewrites, mode)

        all_rewards.extend(results["rewards"].cpu().tolist())
        all_refusal_deltas.extend(results["refusal_delta"].cpu().tolist())
        all_similarities.extend(results["similarities"].cpu().tolist())

        num_samples += len(batch)

    model.train()

    if not all_rewards:
        return {"eval_reward": 0.0, "eval_refusal_delta": 0.0, "eval_similarity": 0.0}

    return {
        "eval_reward": sum(all_rewards) / len(all_rewards),
        "eval_refusal_delta": sum(all_refusal_deltas) / len(all_refusal_deltas),
        "eval_similarity": sum(all_similarities) / len(all_similarities),
    }


def save_checkpoint(
    model,
    tokenizer,
    step: int,
    output_dir: str,
    config: FullConfig,
):
    """Save model checkpoint.

    Args:
        model: Policy model
        tokenizer: Tokenizer
        step: Current training step
        output_dir: Output directory
        config: Full configuration
    """
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save model (LoRA adapters)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    # Save config
    save_config_to_json(config, os.path.join(checkpoint_dir, "config.json"))

    print(f"[checkpoint] Saved to {checkpoint_dir}")


def train(config: FullConfig):
    """Main training loop.

    Args:
        config: Full configuration
    """
    print("=" * 60)
    print("PPO Training for Refusal Steering")
    print("=" * 60)

    # Set seed
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)

    # Setup output directory
    os.makedirs(config.training.output_dir, exist_ok=True)
    save_config_to_json(config, os.path.join(config.training.output_dir, "config.json"))

    # Setup wandb
    wandb_run = setup_wandb(config.training)

    # Load data
    print("\n[data] Loading data...")
    train_loader, eval_loader = create_dataloaders(
        config.training.data_path,
        batch_size=config.ppo.batch_size,
        default_mode=config.training.mode,
        val_fraction=config.training.val_fraction,
        seed=config.training.seed,
    )
    print(f"[data] Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")

    # Load policy model
    model, tokenizer, peft_config = load_policy_model(
        config.model,
        config.lora,
        config.ppo,
        config.training,
    )
    clear_cuda_cache()

    # Create reference model
    ref_model = create_reference_model(config.model, peft_config)
    clear_cuda_cache()

    # Create reward model
    print("\n[reward] Loading reward model...")
    reward_model = RefusalSteeringReward(config.model, config.reward)

    # Create PPO trainer
    ppo_trainer = create_ppo_trainer(
        model, ref_model, tokenizer, config.ppo, config.training
    )

    # Training loop
    print("\n[train] Starting training...")
    print(f"[train] Mode: {config.training.mode}")
    print(f"[train] Steps: {config.training.num_training_steps}")

    global_step = 0
    best_eval_reward = float("-inf")

    try:
        while global_step < config.training.num_training_steps:
            for batch in train_loader:
                if global_step >= config.training.num_training_steps:
                    break

                # Get original prompts and modes
                original_prompts = [item["prompt"] for item in batch]
                modes = [item["mode"] for item in batch]
                mode = modes[0] if modes else config.training.mode

                # Format prompts for model input
                formatted_prompts = format_prompts_for_model(batch, tokenizer)

                # Tokenize queries
                query_tensors = []
                for prompt in formatted_prompts:
                    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                    query_tensors.append(tokens["input_ids"].squeeze(0))

                # Generate responses
                response_tensors = []
                responses_text = []
                for query in query_tensors:
                    query = query.to(model.pretrained_model.device)
                    with torch.no_grad():
                        output = model.generate(
                            input_ids=query.unsqueeze(0),
                            max_new_tokens=config.generation.max_new_tokens,
                            min_new_tokens=config.generation.min_new_tokens,
                            do_sample=config.generation.do_sample,
                            temperature=max(1e-6, config.generation.temperature),
                            top_p=config.generation.top_p,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    response = output[0, query.shape[0]:]
                    response_tensors.append(response)
                    responses_text.append(tokenizer.decode(response, skip_special_tokens=True).strip())

                # Compute rewards
                reward_results = reward_model.compute_rewards(
                    original_prompts, responses_text, mode
                )
                rewards = [r.unsqueeze(0) for r in reward_results["rewards"]]

                # PPO step
                try:
                    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                except Exception as e:
                    print(f"[train] PPO step error: {e}")
                    continue

                global_step += 1

                # Logging
                if global_step % config.training.log_every == 0:
                    mean_reward = reward_results["rewards"].mean().item()
                    mean_refusal_delta = reward_results["refusal_delta"].mean().item()
                    mean_similarity = reward_results["similarities"].mean().item()

                    log_dict = {
                        "step": global_step,
                        "train/reward": mean_reward,
                        "train/refusal_delta": mean_refusal_delta,
                        "train/similarity": mean_similarity,
                        "train/kl": stats.get("objective/kl", 0),
                        "train/entropy": stats.get("objective/entropy", 0),
                    }

                    print(
                        f"[step {global_step}] "
                        f"reward={mean_reward:.4f} "
                        f"refusal_delta={mean_refusal_delta:.4f} "
                        f"similarity={mean_similarity:.4f} "
                        f"kl={stats.get('objective/kl', 0):.4f}"
                    )

                    if wandb_run:
                        import wandb
                        wandb.log(log_dict)

                # Evaluation
                if global_step % config.training.eval_every == 0:
                    print("\n[eval] Running evaluation...")
                    eval_metrics = evaluate(
                        model,
                        tokenizer,
                        reward_model,
                        eval_loader,
                        config.generation,
                        config.training,
                    )

                    print(
                        f"[eval] "
                        f"reward={eval_metrics['eval_reward']:.4f} "
                        f"refusal_delta={eval_metrics['eval_refusal_delta']:.4f} "
                        f"similarity={eval_metrics['eval_similarity']:.4f}"
                    )

                    if wandb_run:
                        import wandb
                        wandb.log(eval_metrics)

                    # Track best model
                    if eval_metrics["eval_reward"] > best_eval_reward:
                        best_eval_reward = eval_metrics["eval_reward"]
                        save_checkpoint(
                            model, tokenizer, global_step,
                            os.path.join(config.training.output_dir, "best"),
                            config,
                        )

                # Checkpointing
                if global_step % config.training.save_every == 0:
                    save_checkpoint(
                        model, tokenizer, global_step,
                        config.training.output_dir, config,
                    )

                # Clear cache periodically
                if global_step % 50 == 0:
                    clear_cuda_cache()

    except KeyboardInterrupt:
        print("\n[train] Training interrupted by user")
    finally:
        # Save final checkpoint
        save_checkpoint(
            model, tokenizer, global_step,
            os.path.join(config.training.output_dir, "final"),
            config,
        )

        # Cleanup
        reward_model.cleanup()
        if wandb_run:
            import wandb
            wandb.finish()

    print("\n[train] Training complete!")
    print(f"[train] Best eval reward: {best_eval_reward:.4f}")
    print(f"[train] Checkpoints saved to: {config.training.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="PPO training for refusal steering"
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to JSON config file (overrides other args)",
    )

    # Model args
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Base model ID",
    )
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default="",
        help="Path to existing LoRA checkpoint",
    )
    parser.add_argument(
        "--refusal_vector_path",
        type=str,
        required=False,
        help="Path to refusal vector .npz file",
    )
    parser.add_argument(
        "--activation_layer",
        type=int,
        default=16,
        help="Layer to extract activations from (1-based)",
    )

    # Training args
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        help="Path to training data (JSONL/CSV/TXT)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ppo_checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["increase_refusal", "decrease_refusal"],
        default="increase_refusal",
        help="Training mode",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1000,
        help="Number of training steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for PPO",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Reward args
    parser.add_argument(
        "--similarity_weight",
        type=float,
        default=0.3,
        help="Weight for similarity in reward",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.7,
        help="Minimum similarity threshold",
    )

    # Logging
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="refusal-ppo",
        help="Wandb project name",
    )

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = load_config_from_json(args.config)
    else:
        config = get_default_config()

        # Override from command line args
        config.model.base_model_id = args.base_model
        if args.lora_checkpoint:
            config.model.lora_checkpoint_path = args.lora_checkpoint
        if args.refusal_vector_path:
            config.model.refusal_vector_path = args.refusal_vector_path
        config.model.activation_layer = args.activation_layer

        if args.data_path:
            config.training.data_path = args.data_path
        config.training.output_dir = args.output_dir
        config.training.mode = args.mode
        config.training.num_training_steps = args.num_steps
        config.training.seed = args.seed
        config.training.use_wandb = args.use_wandb
        config.training.wandb_project = args.wandb_project

        config.ppo.batch_size = args.batch_size
        config.ppo.learning_rate = args.learning_rate

        config.reward.similarity_weight = args.similarity_weight
        config.reward.similarity_threshold = args.similarity_threshold

    # Validate required paths
    if not config.training.data_path:
        parser.error("--data_path is required")
    if not config.model.refusal_vector_path:
        parser.error("--refusal_vector_path is required")

    # Run training
    train(config)


if __name__ == "__main__":
    main()
