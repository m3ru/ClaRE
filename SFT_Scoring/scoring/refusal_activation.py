"""Refusal activation scoring using refusal vector similarity."""

import os
import torch
import numpy as np
from typing import List, Optional, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_refusal_vector(
    vector_path: str,
    select_layer: str = "auto"
) -> Tuple[int, np.ndarray]:
    """
    Load refusal vector from .npz file.

    Args:
        vector_path: Path to the refusal vector .npz file
        select_layer: 'auto' to use metadata, or specific layer index

    Returns:
        Tuple of (layer_index, vector_array)
    """
    data = np.load(vector_path, allow_pickle=True)

    vector = data.get('vector')
    if vector is None:
        raise ValueError("NPZ file must contain 'vector' array")

    if vector.ndim == 1:
        # Single-layer vector
        layer = data.get('layer')
        if layer is None:
            raise ValueError("Single-layer vector NPZ missing 'layer' metadata")
        layer_idx = int(layer)
        return layer_idx, vector.astype('float32')

    elif vector.ndim == 2:
        # Multi-layer vector [L, H]
        layers = data.get('layers')
        if layers is None:
            raise ValueError("Multi-layer vector NPZ missing 'layers' metadata")

        if select_layer == 'auto' or select_layer == 'max':
            # Select layer with highest L2 magnitude
            l2_per_layer = data.get('l2_per_layer')
            if l2_per_layer is None:
                l2_per_layer = np.linalg.norm(vector, axis=1)
            idx = int(np.argmax(l2_per_layer))
        else:
            idx = int(select_layer)
            if idx < 0 or idx >= vector.shape[0]:
                raise ValueError(f"select_layer index out of range: {idx}")

        layer_idx = int(layers[idx])
        return layer_idx, vector[idx].astype('float32')
    else:
        raise ValueError(f"Unexpected vector shape: {vector.shape}")


class RefusalActivationScorer:
    """
    Computes refusal activation scores by measuring similarity between
    transformed prompt activations and the refusal vector at a specific layer.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        refusal_vector_path: str = None,
        extraction_layer: Optional[int] = None,  # If None, use layer from vector file
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        hf_token: Optional[str] = None,
        normalization: str = "minmax",  # 'minmax' or 'zscore_sigmoid'
    ):
        """
        Initialize the refusal activation scorer.

        Args:
            model_name: HuggingFace model name or local path
            refusal_vector_path: Path to refusal vector .npz file
            extraction_layer: Layer to extract activations from (overrides vector metadata)
            device: Device to use (auto-detected if None)
            dtype: Model dtype
            hf_token: HuggingFace token for gated models
            normalization: Method to normalize scores ('minmax' or 'zscore_sigmoid')
        """
        self.model_name = model_name
        self.refusal_vector_path = refusal_vector_path
        self.extraction_layer = extraction_layer
        self.normalization = normalization
        self.hf_token = hf_token

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(dtype, torch.bfloat16)

        self.model = None
        self.tokenizer = None
        self.refusal_vector = None
        self.vector_layer = None

    def load_model(self) -> None:
        """Load the model, tokenizer, and refusal vector."""
        if self.model is not None:
            return

        # Load refusal vector
        if self.refusal_vector_path is None:
            raise ValueError("refusal_vector_path must be specified")

        print(f"Loading refusal vector from: {self.refusal_vector_path}")
        self.vector_layer, refusal_vec = load_refusal_vector(
            self.refusal_vector_path
        )

        if self.extraction_layer is not None:
            self.vector_layer = self.extraction_layer

        print(f"Using layer {self.vector_layer} for refusal activation extraction")

        # Normalize the refusal vector
        self.refusal_vector = refusal_vec / (np.linalg.norm(refusal_vec) + 1e-9)

        # Load model
        print(f"Loading model for refusal activation: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            token=self.hf_token,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            token=self.hf_token,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.model.eval()
        print(f"Model loaded on {self.device}")

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.refusal_vector = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Model unloaded")

    def _build_chat_prompt(self, user_prompt: str) -> str:
        """Build a chat-formatted prompt."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _extract_activation(self, text: str, use_chat_format: bool = True) -> np.ndarray:
        """Extract activation at the target layer for a single text."""
        if use_chat_format:
            text = self._build_chat_prompt(text)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # Get hidden states from the target layer
        # hidden_states[0] is embeddings, [1] is layer 0, etc.
        # So for layer N (0-indexed), we need hidden_states[N+1]
        hidden_states = outputs.hidden_states[self.vector_layer + 1]

        # Get the last token's activation (most relevant for next-token prediction)
        last_token_activation = hidden_states[:, -1, :].float().cpu().numpy()[0]

        return last_token_activation

    def _extract_activations_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
        use_chat_format: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Extract activations for a batch of texts."""
        all_activations = []

        # Format texts if needed
        if use_chat_format:
            formatted_texts = [self._build_chat_prompt(t) for t in texts]
        else:
            formatted_texts = texts

        iterator = range(0, len(formatted_texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting activations")

        for i in iterator:
            batch_texts = formatted_texts[i : i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )

            hidden_states = outputs.hidden_states[self.vector_layer + 1]

            # Get last non-padding token for each sequence
            # Find the position of the last non-padding token
            attention_mask = inputs["attention_mask"]
            seq_lengths = attention_mask.sum(dim=1) - 1  # 0-indexed last position

            batch_activations = []
            for j in range(hidden_states.shape[0]):
                last_pos = int(seq_lengths[j])
                activation = hidden_states[j, last_pos, :].float().cpu().numpy()
                batch_activations.append(activation)

            all_activations.extend(batch_activations)

        return np.array(all_activations)

    def _normalize_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        """Normalize raw scores to 0-1 range."""
        if self.normalization == "minmax":
            # Min-max normalization
            min_score = raw_scores.min()
            max_score = raw_scores.max()
            if max_score - min_score < 1e-9:
                return np.ones_like(raw_scores) * 0.5
            return (raw_scores - min_score) / (max_score - min_score)

        elif self.normalization == "zscore_sigmoid":
            # Z-score normalization + sigmoid
            mean = raw_scores.mean()
            std = raw_scores.std()
            if std < 1e-9:
                return np.ones_like(raw_scores) * 0.5
            z_scores = (raw_scores - mean) / std
            return 1 / (1 + np.exp(-z_scores))

        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")

    def score_candidates(
        self,
        transformed_prompts: List[str],
        batch_size: int = 8,
        use_chat_format: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Compute refusal activation scores for candidate transformations.

        Args:
            transformed_prompts: List of transformed prompts to score
            batch_size: Batch size for activation extraction
            use_chat_format: Whether to format prompts as chat
            show_progress: Whether to show progress bar

        Returns:
            Array of normalized refusal activation scores (0-1, higher = more refusal-like)
        """
        self.load_model()

        print(f"Computing refusal activation for {len(transformed_prompts)} candidates...")

        # Extract activations
        activations = self._extract_activations_batch(
            transformed_prompts, batch_size, use_chat_format, show_progress
        )

        # Compute similarity with refusal vector (cosine similarity or dot product)
        print("Computing refusal vector similarities...")
        raw_scores = []
        for act in activations:
            # Normalize activation
            act_norm = act / (np.linalg.norm(act) + 1e-9)
            # Cosine similarity with refusal vector
            similarity = np.dot(act_norm, self.refusal_vector)
            raw_scores.append(similarity)

        raw_scores = np.array(raw_scores)
        print(f"Raw refusal scores: min={raw_scores.min():.4f}, max={raw_scores.max():.4f}, mean={raw_scores.mean():.4f}")

        # Normalize to 0-1 range
        normalized_scores = self._normalize_scores(raw_scores)
        print(f"Normalized refusal scores: min={normalized_scores.min():.4f}, max={normalized_scores.max():.4f}, mean={normalized_scores.mean():.4f}")

        return normalized_scores
