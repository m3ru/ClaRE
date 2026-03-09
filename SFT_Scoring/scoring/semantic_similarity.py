"""Semantic similarity scoring using Llama 8B embeddings."""

import torch
import numpy as np
from typing import List, Optional, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


class SemanticSimilarityScorer:
    """
    Computes semantic similarity between original and transformed prompts
    using mean-pooled hidden states from a middle layer of Llama 8B.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        embedding_layer: int = 16,  # Middle layer for Llama 8B (32 layers total)
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        hf_token: Optional[str] = None,
    ):
        """
        Initialize the semantic similarity scorer.

        Args:
            model_name: HuggingFace model name or local path
            embedding_layer: Layer to extract embeddings from (0-indexed)
            device: Device to use (auto-detected if None)
            dtype: Model dtype ('bfloat16', 'float16', 'float32')
            hf_token: HuggingFace token for gated models
        """
        self.model_name = model_name
        self.embedding_layer = embedding_layer
        self.hf_token = hf_token

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Set dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(dtype, torch.bfloat16)

        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        """Load the model and tokenizer."""
        if self.model is not None:
            return

        print(f"Loading model for semantic similarity: {self.model_name}")

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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Model unloaded")

    def _extract_embedding(self, text: str) -> np.ndarray:
        """Extract mean-pooled embedding for a single text."""
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

        # Get hidden states from specified layer (add 1 because index 0 is embeddings)
        hidden_states = outputs.hidden_states[self.embedding_layer + 1]

        # Mean pool over sequence length (excluding padding)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        hidden_states = hidden_states * attention_mask
        pooled = hidden_states.sum(dim=1) / attention_mask.sum(dim=1)

        return pooled.float().cpu().numpy()[0]

    def _extract_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Extract embeddings for a batch of texts."""
        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting embeddings")

        for i in iterator:
            batch_texts = texts[i : i + batch_size]

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

            hidden_states = outputs.hidden_states[self.embedding_layer + 1]

            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            hidden_states = hidden_states * attention_mask
            pooled = hidden_states.sum(dim=1) / attention_mask.sum(dim=1)

            all_embeddings.append(pooled.float().cpu().numpy())

        return np.vstack(all_embeddings)

    def compute_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """Compute cosine similarity between two texts."""
        self.load_model()

        emb1 = self._extract_embedding(text1)
        emb2 = self._extract_embedding(text2)

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-9
        )

        return float(similarity)

    def score_candidates(
        self,
        original_prompts: List[str],
        transformed_prompts: List[str],
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Compute semantic similarity scores for candidate transformations.

        Args:
            original_prompts: List of original benign prompts
            transformed_prompts: List of transformed prompts
            batch_size: Batch size for embedding extraction
            show_progress: Whether to show progress bar

        Returns:
            Array of similarity scores (already in 0-1 range for similar texts)
        """
        self.load_model()

        if len(original_prompts) != len(transformed_prompts):
            raise ValueError("Number of original and transformed prompts must match")

        print(f"Computing semantic similarity for {len(original_prompts)} candidates...")

        # Extract embeddings for both sets
        print("Extracting original prompt embeddings...")
        original_embeddings = self._extract_embeddings_batch(
            original_prompts, batch_size, show_progress
        )

        print("Extracting transformed prompt embeddings...")
        transformed_embeddings = self._extract_embeddings_batch(
            transformed_prompts, batch_size, show_progress
        )

        # Compute cosine similarities
        print("Computing cosine similarities...")
        similarities = []
        for i in range(len(original_prompts)):
            emb1 = original_embeddings[i]
            emb2 = transformed_embeddings[i]
            sim = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-9
            )
            similarities.append(sim)

        scores = np.array(similarities)

        # Clip to 0-1 range (cosine similarity can technically be negative)
        scores = np.clip(scores, 0, 1)

        print(f"Semantic similarity: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

        return scores
