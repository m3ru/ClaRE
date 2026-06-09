#!/usr/bin/env python3
"""Reward model for OR (overrefusal) PPO training.

Computes the OR score for a (original, rewrite) prompt pair:

    OR = e^(9.2 * (similarity - 0.5)) * (rewrite_refusal - original_refusal) / 100

Higher OR → the rewrite sounds scarier (triggers more refusal) while remaining
semantically close to the benign original.
"""

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig, RewardConfig


class ActivationHook:
    """Forward hook to capture last-token hidden state from a decoder layer."""

    def __init__(self):
        self.activation: Optional[torch.Tensor] = None

    def __call__(self, module: nn.Module, input, output) -> None:
        if isinstance(output, tuple):
            self.activation = output[0].detach()
        else:
            self.activation = output.detach()

    def get_last_token(self, attention_mask: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Last *real* (non-pad) token hidden state, robust to padding side.

        If an attention_mask is given, gathers the activation at the last
        attended position of each row (works for left OR right padding); this
        avoids the bug where right-padding makes position -1 a PAD token. With
        no mask it falls back to [:, -1, :] (correct only for left padding).
        """
        if self.activation is None:
            return None
        if attention_mask is None:
            return self.activation[:, -1, :]  # [B, H]
        # last index where mask == 1, per row (contiguous standard padding)
        seq = attention_mask.shape[1]
        last_idx = seq - 1 - attention_mask.to(torch.int).flip(1).argmax(dim=1)  # [B]
        rows = torch.arange(self.activation.size(0), device=self.activation.device)
        return self.activation[rows, last_idx.to(self.activation.device)]  # [B, H]

    def clear(self) -> None:
        self.activation = None


def load_refusal_vector(
    vector_path: str,
    select_layer: str = "31",
) -> Tuple[int, torch.Tensor]:
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"Refusal vector not found: {vector_path}")
    data = np.load(vector_path, allow_pickle=True)
    vector = data.get("vector")
    if vector is None:
        raise ValueError("npz must contain 'vector' array")

    if vector.ndim == 1:
        layer = int(data.get("layer", -1))
        if layer < 0:
            raise ValueError("Single-vector npz missing 'layer' metadata")
        return layer, torch.from_numpy(vector.astype(np.float32))
    elif vector.ndim == 2:
        layers = data.get("layers")
        if layers is None:
            raise ValueError("Multi-layer npz missing 'layers' metadata")
        layers = layers.astype(np.int32).tolist()
        if select_layer == "max":
            l2 = data.get("l2_per_layer")
            if l2 is None:
                l2 = np.linalg.norm(vector, axis=1)
            idx = int(np.argmax(l2))
        else:
            idx = int(select_layer)
            if idx < 0 or idx >= vector.shape[0]:
                raise ValueError(f"Layer index {idx} out of range")
        return int(layers[idx]), torch.from_numpy(vector[idx].astype(np.float32))
    else:
        raise ValueError(f"Unexpected vector shape: {vector.shape}")


class ORRewardModel:
    """Computes OR score rewards for PPO training.

    OR = e^(k * (similarity - c)) * (rewrite_refusal - original_refusal) / d

    where k, c, d are configurable (default 9.2, 0.5, 100).
    """

    def __init__(
        self,
        model_config: ModelConfig,
        reward_config: RewardConfig,
        device: Optional[torch.device] = None,
    ):
        self.model_config = model_config
        self.reward_config = reward_config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._target_model = None
        self._target_tokenizer = None
        self._hook = None
        self._hook_handle = None
        self._refusal_vector = None
        self._refusal_layer = None
        self._similarity_model = None

    def _ensure_target_model(self) -> None:
        if self._target_model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print("[reward] Loading target model for refusal scoring...")
        hf_token = self.model_config.hf_token or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        ) or os.environ.get("HF_TOKEN")

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        torch_dtype = dtype_map.get(self.model_config.torch_dtype, torch.float32)

        self._target_tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model_id, use_fast=True, token=hf_token,
        )
        # CRITICAL: last-token activation is read as [:, -1, :]. The tokenizer
        # default is right-padding, which puts a PAD token at position -1 for
        # every sequence shorter than the longest in its batch -> corrupted
        # activations and a systematic, length-dependent scoring bias (the
        # original was scored as a uniform unpadded batch, paraphrases as a
        # padded mixed batch). Must match extract_activations_sharded.py.
        self._target_tokenizer.padding_side = "left"
        if self._target_tokenizer.pad_token is None:
            self._target_tokenizer.pad_token = self._target_tokenizer.eos_token

        # Some chat templates (e.g. Llama-Guard-3) reject a system role and
        # require alternating user/assistant turns. Detect once; fall back to a
        # user-only message for those. Must match extract_activations_sharded.py
        # so the extracted vector and the scoring activations stay consistent.
        try:
            self._target_tokenizer.apply_chat_template(
                [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                tokenize=False, add_generation_prompt=True,
            )
            self._use_system_role = True
        except Exception:
            self._use_system_role = False
        print(f"[reward] chat template accepts system role: {self._use_system_role}")

        quant_config = None
        if self.model_config.quantize_target_model:
            try:
                import bitsandbytes  # noqa: F401
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype, bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                print("[reward] bitsandbytes unavailable, loading without quantization")

        self._target_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.base_model_id, token=hf_token,
            device_map="auto", torch_dtype=torch_dtype,
            quantization_config=quant_config,
        )
        self._target_model.eval()

        self._hook = ActivationHook()
        layer_idx = max(0, self.model_config.activation_layer - 1)
        layer_idx = min(layer_idx, len(self._target_model.model.layers) - 1)
        self._hook_handle = self._target_model.model.layers[layer_idx].register_forward_hook(self._hook)
        print(f"[reward] Target model loaded, hook on layer {layer_idx + 1}")

    def _ensure_refusal_vector(self) -> None:
        if self._refusal_vector is not None:
            return
        print("[reward] Loading refusal vector...")
        self._refusal_layer, vec = load_refusal_vector(
            self.model_config.refusal_vector_path,
            self.model_config.select_layer_strategy,
        )
        norm = vec.norm().item()
        self._refusal_vector = vec.to(self.device)
        print(f"[reward] Refusal vector loaded (layer {self._refusal_layer}, L2={norm:.4f})")

    def _ensure_similarity_model(self) -> None:
        if self._similarity_model is not None:
            return
        print("[reward] Loading similarity model...")
        from sentence_transformers import SentenceTransformer
        self._similarity_model = SentenceTransformer(
            self.reward_config.sentence_transformer_model, device="cpu",
        )
        print("[reward] Similarity model loaded")

    def _compute_refusal_scores(self, prompts: List[str]) -> torch.Tensor:
        """Dot product of last-token hidden state with refusal direction."""
        self._ensure_target_model()
        self._ensure_refusal_vector()

        formatted = []
        for p in prompts:
            if self._use_system_role:
                msgs = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": p},
                ]
            else:
                msgs = [{"role": "user", "content": p}]
            formatted.append(self._target_tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            ))

        inputs = self._target_tokenizer(
            formatted, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(self._target_model.device)

        self._hook.clear()
        with torch.no_grad():
            self._target_model(**inputs, use_cache=False)

        activations = self._hook.get_last_token(inputs["attention_mask"])  # [B, H]
        if activations is None:
            raise RuntimeError("No activations captured")

        activations = activations.float()
        refusal_vec = self._refusal_vector.to(activations.device).float()
        scores = torch.matmul(activations, refusal_vec)
        scores = scores / (refusal_vec.norm() + 1e-9)

        self._hook.clear()
        return scores

    def _compute_similarity(
        self, originals: List[str], rewrites: List[str],
    ) -> torch.Tensor:
        self._ensure_similarity_model()
        orig_emb = self._similarity_model.encode(originals, convert_to_tensor=True, show_progress_bar=False)
        rew_emb = self._similarity_model.encode(rewrites, convert_to_tensor=True, show_progress_bar=False)
        orig_n = F.normalize(orig_emb, dim=-1)
        rew_n = F.normalize(rew_emb, dim=-1)
        return (orig_n * rew_n).sum(dim=-1).to(self.device)

    def compute_rewards(
        self,
        original_prompts: List[str],
        rewritten_prompts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Compute OR score rewards.

        OR = e^(k * (similarity - c)) * (rewrite_refusal - original_refusal) / d
        """
        refusal_orig = self._compute_refusal_scores(original_prompts)
        refusal_rew = self._compute_refusal_scores(rewritten_prompts)
        similarities = self._compute_similarity(original_prompts, rewritten_prompts)

        refusal_delta = refusal_rew - refusal_orig

        k = self.reward_config.similarity_exponent
        c = self.reward_config.similarity_center
        d = self.reward_config.refusal_divisor
        sim_weight = torch.exp(k * (similarities - c))
        or_scores = sim_weight * refusal_delta / d

        or_scores = torch.clamp(or_scores, 0, self.reward_config.reward_clamp)

        return {
            "rewards": or_scores,
            "refusal_scores_original": refusal_orig,
            "refusal_scores_rewritten": refusal_rew,
            "refusal_delta": refusal_delta,
            "similarities": similarities,
            "sim_weight": sim_weight,
        }

    def cleanup(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._hook = None
        self._target_model = None
        self._target_tokenizer = None
        self._similarity_model = None
        self._refusal_vector = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[reward] Cleanup complete")
