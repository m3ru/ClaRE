#!/usr/bin/env python3
"""Reward model for PPO training on refusal steering.

This module provides the RefusalSteeringReward class that computes rewards based on:
1. Refusal score: How likely the target model is to refuse the rewritten prompt
2. Semantic similarity: How similar the rewritten prompt is to the original
3. Threshold penalties: Penalties for low similarity

The refusal score is computed by projecting activations onto a pre-computed
"refusal vector" direction in the model's activation space.
"""

import os
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig, RewardConfig


class ActivationHook:
    """Forward hook to capture activations from a specific layer."""

    def __init__(self):
        self.activation: Optional[torch.Tensor] = None

    def __call__(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        """Capture the output activation.

        For LlamaDecoderLayer, output may be tuple (hidden_states, ...) or tensor.
        """
        if isinstance(output, tuple):
            self.activation = output[0].detach()
        else:
            self.activation = output.detach()

    def get_last_token_activation(self) -> Optional[torch.Tensor]:
        """Get activation of the last token position.

        Returns:
            Tensor of shape [batch_size, hidden_size] or None if no activation captured
        """
        if self.activation is None:
            return None
        # Shape: [batch, seq_len, hidden] -> [batch, hidden]
        return self.activation[:, -1, :]

    def clear(self) -> None:
        """Clear stored activation to free memory."""
        self.activation = None


def load_refusal_vector(
    vector_path: str,
    select_layer: str = "16",
) -> Tuple[int, torch.Tensor]:
    """Load refusal vector from .npz file.

    Args:
        vector_path: Path to .npz file containing refusal vector
        select_layer: Layer to select ('max' for highest L2 norm, or integer index)

    Returns:
        Tuple of (layer_index, vector) where vector is shape [hidden_size]

    Raises:
        FileNotFoundError: If vector file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"Refusal vector not found: {vector_path}")

    data = np.load(vector_path, allow_pickle=True)
    vector = data.get("vector")

    if vector is None:
        raise ValueError("npz file must contain 'vector' array")

    if vector.ndim == 1:
        # Single layer vector
        layer = int(data.get("layer", -1))
        if layer < 0:
            raise ValueError("Single-vector npz missing 'layer' metadata")
        return layer, torch.from_numpy(vector.astype(np.float32))

    elif vector.ndim == 2:
        # Multi-layer vector [num_layers, hidden_size]
        layers = data.get("layers")
        if layers is None:
            raise ValueError("Multi-layer npz missing 'layers' metadata")
        layers = layers.astype(np.int32).tolist()

        if select_layer == "max":
            # Select layer with highest L2 norm
            l2 = data.get("l2_per_layer")
            if l2 is None:
                l2 = np.linalg.norm(vector, axis=1)
            idx = int(np.argmax(l2))
        else:
            idx = int(select_layer)
            if idx < 0 or idx >= vector.shape[0]:
                raise ValueError(f"Layer index {idx} out of range [0, {vector.shape[0]})")

        layer = int(layers[idx])
        return layer, torch.from_numpy(vector[idx].astype(np.float32))

    else:
        raise ValueError(f"Unexpected vector shape: {vector.shape}")


class RefusalSteeringReward:
    """Computes rewards for refusal steering based on activation projections.

    This class:
    1. Loads a target model and registers a hook to extract activations
    2. Loads a pre-computed refusal vector
    3. Computes refusal scores by projecting activations onto the refusal direction
    4. Computes semantic similarity using sentence-transformers
    5. Combines these into a composite reward

    The reward formula is:
        reward = (1 - sim_weight) * refusal_delta + sim_weight * similarity + penalty

    Where:
    - refusal_delta: Change in refusal score (positive = more refusal for increase mode)
    - similarity: Cosine similarity between original and rewritten prompts
    - penalty: Applied when similarity drops below threshold
    """

    def __init__(
        self,
        model_config: ModelConfig,
        reward_config: RewardConfig,
        device: Optional[torch.device] = None,
    ):
        """Initialize reward model.

        Args:
            model_config: Model paths and configuration
            reward_config: Reward weights and thresholds
            device: Device to load models on (auto-detected if None)
        """
        self.model_config = model_config
        self.reward_config = reward_config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Will be loaded lazily
        self._target_model = None
        self._target_tokenizer = None
        self._activation_hook = None
        self._hook_handle = None
        self._refusal_vector = None
        self._refusal_layer = None
        self._similarity_model = None

    def _ensure_target_model_loaded(self) -> None:
        """Load target model if not already loaded."""
        if self._target_model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print("[reward_model] Loading target model...")

        hf_token = self.model_config.hf_token or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        ) or os.environ.get("HF_TOKEN")

        # Determine dtype
        if self.model_config.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif self.model_config.torch_dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # Load tokenizer
        self._target_tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model_id,
            use_fast=True,
            token=hf_token,
        )
        if self._target_tokenizer.pad_token is None:
            self._target_tokenizer.pad_token = self._target_tokenizer.eos_token

        # Quantization config for memory efficiency
        quantization_config = None
        if self.model_config.quantize_target_model:
            try:
                import bitsandbytes  # noqa: F401
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                print("[reward_model] bitsandbytes not available, loading without quantization")

        # Load model
        self._target_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.base_model_id,
            token=hf_token,
            device_map="auto",
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
        )
        self._target_model.eval()

        # Register activation hook
        self._activation_hook = ActivationHook()
        layer_idx = self.model_config.activation_layer - 1  # Convert to 0-based
        layer_idx = max(0, min(layer_idx, len(self._target_model.model.layers) - 1))
        target_layer = self._target_model.model.layers[layer_idx]
        self._hook_handle = target_layer.register_forward_hook(self._activation_hook)

        print(f"[reward_model] Target model loaded, hook on layer {layer_idx + 1}")

    def _ensure_refusal_vector_loaded(self) -> None:
        """Load refusal vector if not already loaded."""
        if self._refusal_vector is not None:
            return

        print("[reward_model] Loading refusal vector...")

        self._refusal_layer, vec = load_refusal_vector(
            self.model_config.refusal_vector_path,
            self.model_config.select_layer_strategy,
        )

        # Keep vector unnormalized to preserve magnitude information
        # The raw dot product will give larger, more meaningful reward signals
        self._refusal_vector = vec.to(self.device)
        print(f"[reward_model] Refusal vector L2 norm: {vec.norm().item():.4f}")

        print(f"[reward_model] Refusal vector loaded from layer {self._refusal_layer}")

    def _ensure_similarity_model_loaded(self) -> None:
        """Load sentence transformer for similarity computation."""
        if self._similarity_model is not None:
            return

        print("[reward_model] Loading similarity model...")

        from sentence_transformers import SentenceTransformer

        self._similarity_model = SentenceTransformer(
            self.reward_config.sentence_transformer_model,
            device=str(self.device),
        )

        print("[reward_model] Similarity model loaded")

    def _compute_refusal_score(self, prompts: List[str]) -> torch.Tensor:
        """Compute refusal scores for prompts.

        Args:
            prompts: List of prompt strings

        Returns:
            Tensor of shape [batch_size] with refusal scores
        """
        self._ensure_target_model_loaded()
        self._ensure_refusal_vector_loaded()

        # Format prompts as chat
        formatted = []
        for p in prompts:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": p},
            ]
            text = self._target_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted.append(text)

        # Tokenize
        inputs = self._target_tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self._target_model.device)

        # Forward pass to get activations
        self._activation_hook.clear()
        with torch.no_grad():
            self._target_model(**inputs, use_cache=False)

        # Get last token activations
        activations = self._activation_hook.get_last_token_activation()
        if activations is None:
            raise RuntimeError("No activations captured from hook")

        # Project onto refusal vector
        activations = activations.float()
        refusal_vec = self._refusal_vector.to(activations.device).float()

        # Raw dot product (not cosine similarity) to preserve magnitude
        # This gives larger reward signals, especially at later layers
        raw_scores = torch.matmul(activations, refusal_vec)
        
        # Scale down by refusal vector norm to keep scores in reasonable range
        # but still proportional to magnitude (unlike cosine similarity which
        # completely removes magnitude information)
        refusal_vec_norm = refusal_vec.norm()
        refusal_scores = raw_scores / (refusal_vec_norm + 1e-9)

        self._activation_hook.clear()
        return refusal_scores

    def _compute_similarity(
        self,
        original_prompts: List[str],
        rewritten_prompts: List[str],
    ) -> torch.Tensor:
        """Compute semantic similarity between original and rewritten prompts.

        Args:
            original_prompts: Original input prompts
            rewritten_prompts: Rewritten prompts from the policy

        Returns:
            Tensor of shape [batch_size] with similarity scores in [0, 1]
        """
        self._ensure_similarity_model_loaded()

        # Encode both sets of prompts
        orig_embeddings = self._similarity_model.encode(
            original_prompts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        rewrite_embeddings = self._similarity_model.encode(
            rewritten_prompts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        # Compute cosine similarity
        orig_norm = F.normalize(orig_embeddings, dim=-1)
        rewrite_norm = F.normalize(rewrite_embeddings, dim=-1)
        similarities = (orig_norm * rewrite_norm).sum(dim=-1)

        return similarities.to(self.device)

    def compute_rewards(
        self,
        original_prompts: List[str],
        rewritten_prompts: List[str],
        mode: Literal["increase_refusal", "decrease_refusal"] = "increase_refusal",
    ) -> Dict[str, torch.Tensor]:
        """Compute composite rewards for rewritten prompts.

        Args:
            original_prompts: Original input prompts
            rewritten_prompts: Rewritten prompts from the policy
            mode: 'increase_refusal' rewards higher refusal, 'decrease_refusal' rewards lower

        Returns:
            Dict with keys:
                - 'rewards': Composite reward tensor [batch_size]
                - 'refusal_scores_original': Refusal scores for original prompts
                - 'refusal_scores_rewritten': Refusal scores for rewritten prompts
                - 'refusal_delta': Change in refusal scores
                - 'similarities': Semantic similarity scores
                - 'penalties': Threshold penalties applied
        """
        # Compute refusal scores
        refusal_original = self._compute_refusal_score(original_prompts)
        refusal_rewritten = self._compute_refusal_score(rewritten_prompts)

        # Compute refusal delta
        raw_delta = refusal_rewritten - refusal_original

        # Apply mode-specific sign
        if mode == "increase_refusal":
            # Positive reward for increased refusal
            refusal_delta = raw_delta
        else:
            # Positive reward for decreased refusal
            refusal_delta = -raw_delta

        # Scale and clamp
        refusal_delta = refusal_delta * self.reward_config.refusal_scale
        refusal_delta = torch.clamp(
            refusal_delta,
            -self.reward_config.refusal_delta_clamp,
            self.reward_config.refusal_delta_clamp,
        )

        # ============================================================
        # TEMPORARILY COMMENTED OUT: Similarity and penalty constraints
        # Uncomment to re-enable semantic similarity in reward signal
        # ============================================================
        # # Compute similarity
        # similarities = self._compute_similarity(original_prompts, rewritten_prompts)
        #
        # # Compute threshold penalties
        # penalties = torch.zeros_like(similarities)
        # below_threshold = similarities < self.reward_config.similarity_threshold
        # penalties[below_threshold] = self.reward_config.threshold_penalty
        #
        # # Composite reward
        # sim_weight = self.reward_config.similarity_weight
        # rewards = (
        #     (1 - sim_weight) * refusal_delta
        #     + sim_weight * similarities
        #     + penalties
        # )

        # REFUSAL-ONLY REWARD: Just the refusal delta, no similarity constraint
        rewards = refusal_delta
        similarities = torch.zeros_like(refusal_delta)  # placeholder
        penalties = torch.zeros_like(refusal_delta)      # placeholder

        return {
            "rewards": rewards,
            "refusal_scores_original": refusal_original,
            "refusal_scores_rewritten": refusal_rewritten,
            "refusal_delta": raw_delta,
            "similarities": similarities,
            "penalties": penalties,
        }

    def compute_refusal_score_only(self, prompts: List[str]) -> torch.Tensor:
        """Compute just the refusal score (useful for evaluation).

        Args:
            prompts: List of prompts to score

        Returns:
            Tensor of refusal scores [batch_size]
        """
        return self._compute_refusal_score(prompts)

    def cleanup(self) -> None:
        """Release model resources and clear CUDA cache."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

        self._activation_hook = None
        self._target_model = None
        self._target_tokenizer = None
        self._similarity_model = None
        self._refusal_vector = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[reward_model] Cleanup complete")


def create_reward_model(
    model_config: ModelConfig,
    reward_config: RewardConfig,
    device: Optional[torch.device] = None,
) -> RefusalSteeringReward:
    """Factory function to create reward model.

    Args:
        model_config: Model configuration
        reward_config: Reward configuration
        device: Device to use

    Returns:
        Initialized RefusalSteeringReward instance
    """
    return RefusalSteeringReward(model_config, reward_config, device)


if __name__ == "__main__":
    # Quick test (requires models to be available)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_path", required=True, help="Path to refusal vector .npz")
    parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    model_cfg = ModelConfig(
        base_model_id=args.model_id,
        refusal_vector_path=args.vector_path,
    )
    reward_cfg = RewardConfig()

    reward_model = RefusalSteeringReward(model_cfg, reward_cfg)

    test_original = ["How do I make a cake?", "What's the weather like?"]
    test_rewritten = [
        "How do I create a potentially dangerous explosive cake?",
        "What's the weather forecast for planning outdoor activities?",
    ]

    results = reward_model.compute_rewards(test_original, test_rewritten, "increase_refusal")
    print("Rewards:", results["rewards"])
    print("Refusal delta:", results["refusal_delta"])
    print("Similarities:", results["similarities"])

    reward_model.cleanup()
