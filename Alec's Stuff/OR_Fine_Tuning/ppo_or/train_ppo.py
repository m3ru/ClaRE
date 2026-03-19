#!/usr/bin/env python3
"""PPO training script for overrefusal prompt rewriting.

Trains Llama-3-8B-Instruct (with LoRA) to rewrite benign prompts into
versions that trigger safety overrefusal while preserving semantic meaning.

Reward = OR score = e^(18.4*(sim-0.75)) * (rewrite_refusal - orig_refusal) / 100

Usage:
    python train_ppo.py --data_path dolly15k.jsonl --refusal_vector_path vector.npz
"""

import argparse
import gc
import json
import os
import sys
from dataclasses import asdict
from typing import Dict, List, Optional

import os as _os
_os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "0"
_os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
import torch
torch.backends.cudnn.enabled = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    FullConfig,
    GenerationConfig,
    LoRAConfig,
    ModelConfig,
    PPOConfig,
    PromptConfig,
    RewardConfig,
    TrainingConfig,
    SYSTEM_PROMPT,
    get_default_config,
    load_config_from_json,
    save_config_to_json,
)
from data import create_dataloaders, format_prompts_for_model
from reward_model import ORRewardModel


class NaNSafeLogitsProcessor:
    """Clamp NaN/Inf logits before they reach the sampler, preventing CUDA assertions."""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)
        return scores


SANITY_PROMPTS = [
    "Explain how photosynthesis works.",
    "What are common causes of headaches and when should someone see a doctor?",
    "How do I write a polite email asking for an extension on a deadline?",
]


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def _build_peft_lora_config(lora_config: LoRAConfig, lora_checkpoint_path: Optional[str] = None):
    from peft import LoraConfig as PeftLoraConfig

    r = lora_config.r
    lora_alpha = lora_config.lora_alpha
    lora_dropout = lora_config.lora_dropout
    bias = lora_config.bias
    task_type = lora_config.task_type
    target_modules = lora_config.target_modules

    if lora_checkpoint_path:
        adapter_cfg_path = os.path.join(lora_checkpoint_path, "adapter_config.json")
        if os.path.exists(adapter_cfg_path):
            try:
                with open(adapter_cfg_path, "r", encoding="utf-8") as f:
                    adapter_cfg = json.load(f)
                r = int(adapter_cfg.get("r", r))
                lora_alpha = int(adapter_cfg.get("lora_alpha", lora_alpha))
                lora_dropout = float(adapter_cfg.get("lora_dropout", lora_dropout))
                bias = str(adapter_cfg.get("bias", bias))
                task_type = str(adapter_cfg.get("task_type", task_type))
                tm = adapter_cfg.get("target_modules", target_modules)
                if isinstance(tm, list) and tm:
                    target_modules = tm
                print(f"[model] LoRA config from checkpoint (r={r}, alpha={lora_alpha})", flush=True)
            except Exception as e:
                print(f"[model] Warning: could not parse adapter config: {e}", flush=True)

    return PeftLoraConfig(
        r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        bias=bias, task_type=task_type, target_modules=target_modules,
    )


def setup_wandb(config: TrainingConfig):
    if not config.use_wandb:
        return None
    try:
        import wandb
        return wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=asdict(config) if hasattr(config, "__dataclass_fields__") else config,
        )
    except Exception as e:
        print(f"[wandb] Failed: {e}")
        return None


def _hf_login(hf_token: str) -> None:
    """Authenticate with Hugging Face, handling version differences gracefully."""
    try:
        from huggingface_hub import login
        login(token=hf_token)
    except (ValueError, Exception) as e:
        print(f"[model] login() failed ({e}), falling back to env var")
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token


def load_policy_model(model_config: ModelConfig, lora_config: LoRAConfig, training_config: TrainingConfig):
    from transformers import AutoTokenizer
    from trl import AutoModelForCausalLMWithValueHead

    print("[model] Loading policy model...")

    hf_token = model_config.hf_token or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    ) or os.environ.get("HF_TOKEN")
    if hf_token:
        _hf_login(hf_token)
        print(f"[model] Authenticated (token: {hf_token[:10]}...)")
    else:
        print("[model] WARNING: No HF token found!")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.base_model_id, use_fast=True, token=hf_token,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map.get(model_config.torch_dtype, torch.float32)

    peft_config = _build_peft_lora_config(lora_config, model_config.lora_checkpoint_path)

    if model_config.lora_checkpoint_path and os.path.exists(model_config.lora_checkpoint_path):
        print(f"[model] Loading from LoRA checkpoint: {model_config.lora_checkpoint_path}")
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_config.base_model_id, token=hf_token,
            torch_dtype=torch_dtype, device_map={"": 0}, peft_config=peft_config,
            attn_implementation="eager",
        )
        try:
            from peft import set_peft_model_state_dict
            import safetensors.torch
            adapter_path = os.path.join(model_config.lora_checkpoint_path, "adapter_model.safetensors")
            if os.path.exists(adapter_path):
                weights = safetensors.torch.load_file(adapter_path)
                set_peft_model_state_dict(model.pretrained_model, weights)
                print(f"[model] Loaded LoRA weights from {adapter_path}")
            else:
                adapter_path = os.path.join(model_config.lora_checkpoint_path, "adapter_model.bin")
                if os.path.exists(adapter_path):
                    weights = torch.load(adapter_path, map_location="cpu")
                    set_peft_model_state_dict(model.pretrained_model, weights)
                    print(f"[model] Loaded LoRA weights from {adapter_path}")
                else:
                    print("[model] Warning: No adapter weights found")
        except Exception as e:
            print(f"[model] Warning: Could not load LoRA weights: {e}")
    else:
        print("[model] Initializing fresh LoRA adapters")
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_config.base_model_id, token=hf_token,
            torch_dtype=torch_dtype, device_map={"": 0}, peft_config=peft_config,
            attn_implementation="eager",
        )

    if training_config.gradient_checkpointing:
        model.pretrained_model.gradient_checkpointing_enable()
        model.pretrained_model.config.use_cache = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Policy model loaded ({trainable:,} trainable params)")
    return model, tokenizer, peft_config


def create_reference_model(model_config: ModelConfig, peft_config):
    from trl import AutoModelForCausalLMWithValueHead

    print("[model] Creating reference model...")
    hf_token = model_config.hf_token or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    ) or os.environ.get("HF_TOKEN")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map.get(model_config.torch_dtype, torch.float32)

    try:
        peft_config = _build_peft_lora_config(LoRAConfig(), model_config.lora_checkpoint_path)
    except Exception:
        pass

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_config.base_model_id, token=hf_token,
        torch_dtype=torch_dtype, device_map={"": 0}, peft_config=peft_config,
    )

    if model_config.lora_checkpoint_path and os.path.exists(model_config.lora_checkpoint_path):
        try:
            from peft import set_peft_model_state_dict
            import safetensors.torch
            adapter_path = os.path.join(model_config.lora_checkpoint_path, "adapter_model.safetensors")
            if os.path.exists(adapter_path):
                weights = safetensors.torch.load_file(adapter_path)
                set_peft_model_state_dict(ref_model.pretrained_model, weights)
            else:
                adapter_path = os.path.join(model_config.lora_checkpoint_path, "adapter_model.bin")
                if os.path.exists(adapter_path):
                    weights = torch.load(adapter_path, map_location="cpu")
                    set_peft_model_state_dict(ref_model.pretrained_model, weights)
        except Exception as e:
            print(f"[model] Warning: Could not load ref LoRA weights: {e}")

    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    print("[model] Reference model created (frozen)")
    return ref_model


def create_ppo_trainer(model, ref_model, tokenizer, ppo_config: PPOConfig, training_config: TrainingConfig):
    from trl import PPOConfig as TRLPPOConfig
    from stable_ppo import StablePPOTrainer
    import inspect

    print("[trainer] Creating StablePPOTrainer (float32 loss, clamped log-ratios, NaN-grad skip)...")
    config_kwargs = {
        "learning_rate": ppo_config.learning_rate,
        "batch_size": ppo_config.batch_size,
        "mini_batch_size": ppo_config.mini_batch_size,
        "gradient_accumulation_steps": ppo_config.gradient_accumulation_steps,
        "seed": training_config.seed,
    }

    optional_params = {
        "ppo_epochs": ppo_config.ppo_epochs,
        "num_ppo_epochs": ppo_config.ppo_epochs,
        "target_kl": ppo_config.target_kl,
        "init_kl_coef": ppo_config.init_kl_coef,
        "adap_kl_ctrl": ppo_config.adap_kl_ctrl,
        "cliprange": ppo_config.cliprange,
        "cliprange_value": ppo_config.cliprange_value,
        "vf_coef": ppo_config.vf_coef,
        "gamma": ppo_config.gamma,
        "lam": ppo_config.lam,
        "max_grad_norm": ppo_config.max_grad_norm,
    }

    try:
        sig = inspect.signature(TRLPPOConfig.__init__)
        valid_params = set(sig.parameters.keys())
        for param, value in optional_params.items():
            if param in valid_params:
                config_kwargs[param] = value
    except Exception:
        config_kwargs["num_ppo_epochs"] = ppo_config.ppo_epochs

    if training_config.use_wandb:
        config_kwargs["log_with"] = "wandb"

    config = TRLPPOConfig(**config_kwargs)
    print(f"[trainer] PPO config: lr={config.learning_rate}, init_kl={config_kwargs.get('init_kl_coef', 'default')}, "
          f"adap_kl={config_kwargs.get('adap_kl_ctrl', 'default')}, target_kl={config_kwargs.get('target_kl', 'default')}")
    trainer = StablePPOTrainer(config=config, model=model, ref_model=ref_model, tokenizer=tokenizer)
    print("[trainer] StablePPOTrainer created")
    return trainer


def generate_responses(model, tokenizer, prompts: List[str], gen_config: GenerationConfig) -> List[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(model.pretrained_model.device)
    attention_mask = inputs["attention_mask"].to(model.pretrained_model.device)
    nan_safe = NaNSafeLogitsProcessor()

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=gen_config.max_new_tokens,
            min_new_tokens=gen_config.min_new_tokens,
            do_sample=gen_config.do_sample,
            temperature=max(1e-6, gen_config.temperature),
            top_p=gen_config.top_p,
            top_k=gen_config.top_k if gen_config.top_k > 0 else None,
            repetition_penalty=gen_config.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            logits_processor=[nan_safe],
            eos_token_id=tokenizer.eos_token_id,
        )

    responses = []
    for i, output in enumerate(outputs):
        prompt_len = input_ids[i].shape[0]
        text = tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip()
        responses.append(text)
    return responses


def log_sanity_rewrites(model, tokenizer, output_dir: str, step: int,
                        reward_model: Optional[ORRewardModel] = None,
                        prompt_config: Optional[PromptConfig] = None):
    import csv
    batch = [{"prompt": p} for p in SANITY_PROMPTS]
    formatted = format_prompts_for_model(batch, tokenizer, prompt_config=prompt_config)
    gen_cfg = GenerationConfig(temperature=0.0, do_sample=False, max_new_tokens=256, repetition_penalty=1.0)

    was_training = model.training
    model.eval()
    try:
        rewrites = generate_responses(model, tokenizer, formatted, gen_cfg)
    finally:
        if was_training:
            model.train()

    out_path = os.path.join(output_dir, "sanity_rewrites.csv")
    is_new = not os.path.exists(out_path)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["step", "prompt", "rewrite"])
        for p, r in zip(SANITY_PROMPTS, rewrites):
            w.writerow([step, p, r])

    print(f"\n[sanity] step={step} — {len(SANITY_PROMPTS)} rewrites:")
    reward_results = None
    if reward_model is not None:
        try:
            reward_results = reward_model.compute_rewards(
                list(SANITY_PROMPTS), rewrites,
            )
        except Exception as e:
            print(f"[sanity] reward computation failed: {e}")
    for i, (p, r) in enumerate(zip(SANITY_PROMPTS, rewrites)):
        print(f"  [{i}] original: {p[:250]}")
        print(f"      rewrite:  {r[:250]}")
        if reward_results is not None:
            print(
                f"      sim={reward_results['similarities'][i]:.3f} "
                f"delta={reward_results['refusal_delta'][i]:.3f} "
                f"or={reward_results['rewards'][i]:.4f}"
            )
    print()


def evaluate(model, tokenizer, reward_model: ORRewardModel, eval_loader, gen_config, num_samples: int,
             prompt_config: Optional[PromptConfig] = None) -> Dict[str, float]:
    model.eval()
    all_rewards, all_deltas, all_sims = [], [], []
    n = 0
    for batch in eval_loader:
        if n >= num_samples:
            break
        original_prompts = [item["prompt"] for item in batch]
        formatted = format_prompts_for_model(batch, tokenizer, prompt_config=prompt_config)
        rewrites = generate_responses(model, tokenizer, formatted, gen_config)
        results = reward_model.compute_rewards(original_prompts, rewrites)
        all_rewards.extend(results["rewards"].cpu().tolist())
        all_deltas.extend(results["refusal_delta"].cpu().tolist())
        all_sims.extend(results["similarities"].cpu().tolist())
        n += len(batch)
    model.train()
    if not all_rewards:
        return {"eval_or_score": 0.0, "eval_refusal_delta": 0.0, "eval_similarity": 0.0}
    return {
        "eval_or_score": sum(all_rewards) / len(all_rewards),
        "eval_refusal_delta": sum(all_deltas) / len(all_deltas),
        "eval_similarity": sum(all_sims) / len(all_sims),
    }


def save_checkpoint(model, tokenizer, step: int, output_dir: str, config: FullConfig):
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    save_config_to_json(config, os.path.join(checkpoint_dir, "config.json"))
    print(f"[checkpoint] Saved to {checkpoint_dir}")


def train(config: FullConfig):
    print("=" * 60)
    print("PPO Training for Overrefusal Prompt Rewriting")
    print("=" * 60)

    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)

    os.makedirs(config.training.output_dir, exist_ok=True)
    save_config_to_json(config, os.path.join(config.training.output_dir, "config.json"))

    wandb_run = setup_wandb(config.training)

    print("\n[data] Loading data...")
    train_loader, eval_loader = create_dataloaders(
        config.training.data_path, batch_size=config.ppo.batch_size,
        val_fraction=config.training.val_fraction, seed=config.training.seed,
    )
    print(f"[data] Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")

    model, tokenizer, peft_config = load_policy_model(config.model, config.lora, config.training)
    clear_cuda_cache()

    ref_model = create_reference_model(config.model, peft_config)
    clear_cuda_cache()

    print("\n[reward] Loading reward model...")
    reward_model = ORRewardModel(config.model, config.reward)

    ppo_trainer = create_ppo_trainer(model, ref_model, tokenizer, config.ppo, config.training)

    prompt_cfg = config.prompt

    try:
        log_sanity_rewrites(model, tokenizer, config.training.output_dir, step=0,
                            reward_model=reward_model, prompt_config=prompt_cfg)
    except Exception as e:
        print(f"[sanity] warning: {e}")

    print("\n[train] Starting training...")
    print(f"[train] Reward: OR = e^({config.reward.similarity_exponent}*(sim-{config.reward.similarity_center})) * delta_refusal / {config.reward.refusal_divisor}")
    print(f"[train] Steps: {config.training.num_training_steps}")
    print(f"[train] System prompt: {prompt_cfg.system_prompt[:120]}...")
    print(f"[train] Few-shot examples: {len(prompt_cfg.few_shot_examples)}")

    global_step = 0
    best_eval = float("-inf")

    try:
        while global_step < config.training.num_training_steps:
            for batch in train_loader:
                if global_step >= config.training.num_training_steps:
                    break

                original_prompts = [item["prompt"] for item in batch]
                formatted_prompts = format_prompts_for_model(batch, tokenizer, prompt_config=prompt_cfg)

                query_tensors = []
                for prompt in formatted_prompts:
                    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                    query_tensors.append(tokens["input_ids"].squeeze(0))

                model.eval()
                response_tensors = []
                responses_text = []
                nan_safe = NaNSafeLogitsProcessor()
                gen_failed = False
                for query in query_tensors:
                    query = query.to(model.pretrained_model.device)
                    try:
                        with torch.no_grad():
                            output = model.generate(
                                input_ids=query.unsqueeze(0),
                                max_new_tokens=config.generation.max_new_tokens,
                                min_new_tokens=config.generation.min_new_tokens,
                                do_sample=config.generation.do_sample,
                                temperature=max(1e-6, config.generation.temperature),
                                top_p=config.generation.top_p,
                                top_k=config.generation.top_k if config.generation.top_k > 0 else None,
                                repetition_penalty=config.generation.repetition_penalty,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                logits_processor=[nan_safe],
                            )
                        response = output[0, query.shape[0]:]
                        response_tensors.append(response)
                        responses_text.append(tokenizer.decode(response, skip_special_tokens=True).strip())
                    except Exception as e:
                        err_str = str(e).lower()
                        is_cuda_err = any(s in err_str for s in ["cuda", "probability tensor", "nan", "inf", "assert"])
                        if not is_cuda_err:
                            raise
                        print(f"[train] Generation error (step {global_step}): {e}", flush=True)
                        clear_cuda_cache()
                        gen_failed = True
                        break
                model.train()

                if gen_failed:
                    print(f"[train] Skipping batch at step {global_step} due to generation failure", flush=True)
                    global_step += 1
                    continue

                reward_results = reward_model.compute_rewards(original_prompts, responses_text)
                rewards = [r.unsqueeze(0) for r in reward_results["rewards"]]

                try:
                    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                except Exception as e:
                    print(f"[train] PPO step error: {e}")
                    continue

                global_step += 1

                if global_step % config.training.log_every == 0:
                    mean_or = reward_results["rewards"].mean().item()
                    mean_delta = reward_results["refusal_delta"].mean().item()
                    mean_sim = reward_results["similarities"].mean().item()
                    clamp_frac = stats.get("ppo/policy/log_ratio_clamp_frac", 0)
                    approxkl = stats.get("ppo/policy/approxkl", 0)

                    log_dict = {
                        "step": global_step,
                        "train/or_score": mean_or,
                        "train/refusal_delta": mean_delta,
                        "train/similarity": mean_sim,
                        "train/kl": stats.get("objective/kl", 0),
                        "train/entropy": stats.get("objective/entropy", 0),
                        "train/approxkl": approxkl,
                        "train/log_ratio_clamp_frac": clamp_frac,
                    }
                    kl_val = stats.get("objective/kl", float("nan"))
                    msg = (
                        f"[step {global_step}] "
                        f"or={mean_or:.4f} "
                        f"delta={mean_delta:.4f} "
                        f"sim={mean_sim:.4f} "
                        f"kl={kl_val}"
                    )
                    if isinstance(clamp_frac, float) and clamp_frac > 0:
                        msg += f" clamp_frac={clamp_frac:.4f}"
                    print(msg)

                    idx = 0
                    print(
                        f"  example original: {original_prompts[idx][:250]}\n"
                        f"  example rewrite:  {responses_text[idx][:250]}\n"
                        f"  (sim={reward_results['similarities'][idx]:.3f} "
                        f"delta={reward_results['refusal_delta'][idx]:.3f} "
                        f"or={reward_results['rewards'][idx]:.4f})"
                    )

                    if wandb_run:
                        import wandb
                        wandb.log(log_dict)

                if global_step % config.training.eval_every == 0:
                    print("\n[eval] Running evaluation...")
                    eval_metrics = evaluate(
                        model, tokenizer, reward_model, eval_loader,
                        config.generation, config.training.num_eval_samples,
                        prompt_config=prompt_cfg,
                    )
                    try:
                        log_sanity_rewrites(model, tokenizer, config.training.output_dir,
                                            global_step, reward_model=reward_model,
                                            prompt_config=prompt_cfg)
                    except Exception as e:
                        print(f"[sanity] warning: {e}")

                    print(
                        f"[eval] or_score={eval_metrics['eval_or_score']:.4f} "
                        f"refusal_delta={eval_metrics['eval_refusal_delta']:.4f} "
                        f"sim={eval_metrics['eval_similarity']:.4f}"
                    )
                    if wandb_run:
                        import wandb
                        wandb.log(eval_metrics)
                    if eval_metrics["eval_or_score"] > best_eval:
                        best_eval = eval_metrics["eval_or_score"]
                        save_checkpoint(model, tokenizer, global_step,
                                        os.path.join(config.training.output_dir, "best"), config)

                if global_step % config.training.save_every == 0:
                    save_checkpoint(model, tokenizer, global_step, config.training.output_dir, config)

                if global_step % 50 == 0:
                    clear_cuda_cache()

    except KeyboardInterrupt:
        print("\n[train] Interrupted by user")
    finally:
        save_checkpoint(model, tokenizer, global_step,
                        os.path.join(config.training.output_dir, "final"), config)
        reward_model.cleanup()
        if wandb_run:
            import wandb
            wandb.finish()

    print(f"\n[train] Complete! Best eval OR score: {best_eval:.4f}")
    print(f"[train] Checkpoints: {config.training.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="PPO training for overrefusal prompt rewriting")
    parser.add_argument("--config", type=str, default="", help="JSON config file")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--lora_checkpoint", type=str, default="", help="Path to LoRA checkpoint (optional)")
    parser.add_argument("--refusal_vector_path", type=str, required=False)
    parser.add_argument("--activation_layer", type=int, default=32)
    parser.add_argument("--data_path", type=str, required=False)
    parser.add_argument("--output_dir", type=str, default="./ppo_or_checkpoints")
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="or-ppo")
    args = parser.parse_args()

    if args.config:
        config = load_config_from_json(args.config)
    else:
        config = get_default_config()
        config.model.base_model_id = args.base_model
        if args.lora_checkpoint:
            config.model.lora_checkpoint_path = args.lora_checkpoint
        if args.refusal_vector_path:
            config.model.refusal_vector_path = args.refusal_vector_path
        config.model.activation_layer = args.activation_layer
        if args.data_path:
            config.training.data_path = args.data_path
        config.training.output_dir = args.output_dir
        config.training.num_training_steps = args.num_steps
        config.training.seed = args.seed
        config.training.use_wandb = args.use_wandb
        config.training.wandb_project = args.wandb_project
        config.ppo.batch_size = args.batch_size
        config.ppo.learning_rate = args.learning_rate

    if not config.training.data_path:
        parser.error("--data_path is required")
    if not config.model.refusal_vector_path:
        parser.error("--refusal_vector_path is required")

    train(config)


if __name__ == "__main__":
    main()
