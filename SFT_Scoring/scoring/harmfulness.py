"""Harmfulness scoring using Llama Guard."""

import re
import torch
import numpy as np
from typing import List, Optional, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


class HarmfulnessScorer:
    """
    Computes harmfulness scores for prompts using Llama Guard.
    Returns scores in 0-1 range where 0 = safe, 1 = harmful.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-Guard-3-8B",
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        hf_token: Optional[str] = None,
    ):
        """
        Initialize the harmfulness scorer.

        Args:
            model_name: HuggingFace model name or local path for Llama Guard
            device: Device to use (auto-detected if None)
            dtype: Model dtype
            hf_token: HuggingFace token for gated models
        """
        self.model_name = model_name
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

    def load_model(self) -> None:
        """Load the Llama Guard model and tokenizer."""
        if self.model is not None:
            return

        print(f"Loading Llama Guard model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            token=self.hf_token,
        )
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
        print(f"Llama Guard loaded on {self.device}")

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
        print("Llama Guard unloaded")

    def _format_prompt_for_guard(self, user_prompt: str) -> str:
        """Format a prompt for Llama Guard classification."""
        # Llama Guard 3 uses a specific chat format
        conversation = [
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _parse_guard_output(self, output_text: str) -> Tuple[bool, float]:
        """
        Parse Llama Guard output to determine harmfulness.

        Returns:
            Tuple of (is_unsafe, confidence_score)
        """
        output_text = output_text.strip().lower()

        # Llama Guard outputs "safe" or "unsafe" followed by category
        if output_text.startswith("safe"):
            return False, 0.0
        elif output_text.startswith("unsafe"):
            # Could parse the category for more nuanced scoring
            # For now, return 1.0 for any unsafe classification
            return True, 1.0
        else:
            # Fallback: check if unsafe is mentioned anywhere
            if "unsafe" in output_text:
                return True, 0.8
            elif "safe" in output_text:
                return False, 0.2
            else:
                # Unknown output, return middle value
                return False, 0.5

    def _get_classification_logprobs(
        self,
        formatted_prompt: str,
    ) -> Tuple[float, float]:
        """
        Get log probabilities for 'safe' vs 'unsafe' tokens.
        This provides a more nuanced score than binary classification.

        Returns:
            Tuple of (safe_prob, unsafe_prob)
        """
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
            logits = outputs.logits[:, -1, :]  # Last token logits

            # Get token IDs for 'safe' and 'unsafe'
            # Note: These might be subwords depending on tokenizer
            safe_tokens = self.tokenizer.encode("safe", add_special_tokens=False)
            unsafe_tokens = self.tokenizer.encode("unsafe", add_special_tokens=False)

            # Get logits for first subword of each
            safe_logit = logits[0, safe_tokens[0]].item() if safe_tokens else -float('inf')
            unsafe_logit = logits[0, unsafe_tokens[0]].item() if unsafe_tokens else -float('inf')

            # Convert to probabilities
            probs = torch.softmax(
                torch.tensor([safe_logit, unsafe_logit]),
                dim=0
            )

            return probs[0].item(), probs[1].item()

    def _score_single(
        self,
        prompt: str,
        use_logprobs: bool = True,
        max_new_tokens: int = 20,
    ) -> float:
        """Score a single prompt for harmfulness."""
        formatted = self._format_prompt_for_guard(prompt)

        if use_logprobs:
            # Use log probabilities for a continuous score
            safe_prob, unsafe_prob = self._get_classification_logprobs(formatted)
            # Return unsafe probability as harmfulness score
            return unsafe_prob
        else:
            # Generate and parse output
            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the generated tokens
            prompt_len = inputs["input_ids"].shape[1]
            generated = outputs[0, prompt_len:]
            output_text = self.tokenizer.decode(generated, skip_special_tokens=True)

            _, score = self._parse_guard_output(output_text)
            return score

    def score_candidates(
        self,
        prompts: List[str],
        batch_size: int = 4,  # Smaller batches for generation
        use_logprobs: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Compute harmfulness scores for a list of prompts.

        Args:
            prompts: List of prompts to score
            batch_size: Batch size for scoring (generation is slower)
            use_logprobs: Whether to use log probabilities (more nuanced) or generation
            show_progress: Whether to show progress bar

        Returns:
            Array of harmfulness scores (0 = safe, 1 = harmful)
        """
        self.load_model()

        print(f"Computing harmfulness scores for {len(prompts)} prompts...")

        scores = []
        iterator = range(len(prompts))
        if show_progress:
            iterator = tqdm(iterator, desc="Scoring harmfulness")

        for i in iterator:
            try:
                score = self._score_single(prompts[i], use_logprobs=use_logprobs)
                scores.append(score)
            except Exception as e:
                print(f"Warning: Error scoring prompt {i}: {e}")
                scores.append(0.5)  # Default to middle value on error

        scores = np.array(scores)
        print(f"Harmfulness scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

        return scores

    def score_candidates_batch(
        self,
        prompts: List[str],
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Batch scoring using log probabilities (faster than generation).

        Args:
            prompts: List of prompts to score
            batch_size: Batch size
            show_progress: Whether to show progress bar

        Returns:
            Array of harmfulness scores (0 = safe, 1 = harmful)
        """
        self.load_model()

        print(f"Computing harmfulness scores for {len(prompts)} prompts (batched)...")

        all_scores = []
        iterator = range(0, len(prompts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Scoring harmfulness")

        for i in iterator:
            batch_prompts = prompts[i : i + batch_size]
            formatted_prompts = [self._format_prompt_for_guard(p) for p in batch_prompts]

            inputs = self.tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True)
                logits = outputs.logits[:, -1, :]  # Last token logits for each

                # Get token IDs
                safe_tokens = self.tokenizer.encode("safe", add_special_tokens=False)
                unsafe_tokens = self.tokenizer.encode("unsafe", add_special_tokens=False)

                if not safe_tokens or not unsafe_tokens:
                    # Fallback if tokens not found
                    all_scores.extend([0.5] * len(batch_prompts))
                    continue

                safe_token_id = safe_tokens[0]
                unsafe_token_id = unsafe_tokens[0]

                for j in range(logits.shape[0]):
                    safe_logit = logits[j, safe_token_id].item()
                    unsafe_logit = logits[j, unsafe_token_id].item()

                    probs = torch.softmax(
                        torch.tensor([safe_logit, unsafe_logit]),
                        dim=0
                    )
                    unsafe_prob = probs[1].item()
                    all_scores.append(unsafe_prob)

        scores = np.array(all_scores)
        print(f"Harmfulness scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

        return scores
