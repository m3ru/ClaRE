#!/usr/bin/env python3
"""
PEZ (Prompt Embedding Zeroization) for Refusal Vector Optimization

Implements gradient-based discrete prompt optimization to discover prompts
that achieve specific activation levels along the refusal direction.

Based on: "Hard Prompts Made Easy: Gradient-Based Discrete Optimization
for Prompt Tuning and Discovery" (Wen et al., 2023)
https://arxiv.org/abs/2302.03668
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class PEZConfig:
    """Configuration for PEZ optimization"""
    # Model settings
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    dtype: str = "bf16"

    # Refusal vector settings
    refusal_vector_path: str = ""
    layers: str = "all"  # "all", "max", or comma-separated layer indices

    # Optimization settings
    num_steps: int = 3000
    learning_rate: float = 0.1
    prompt_length: int = 20  # Number of tokens to optimize
    batch_size: int = 1  # Currently only supports 1

    # Objective settings
    objective: str = "target"  # "maximize", "minimize", "target", "boundary"
    target_activation: float = 0.0  # For "target" objective
    boundary_margin: float = 0.1  # For "boundary" objective

    # Initialization settings
    init_mode: str = "random"  # "random", "seed", "mixed"
    seed_prompt: Optional[str] = None  # For "seed" or "mixed" mode
    num_random_tokens: int = 20  # For "mixed" mode

    # Optimization hyperparameters
    optimizer: str = "adamw"  # "adamw" or "sgd"
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    # Checkpointing
    save_every: int = 100
    output_dir: str = "./pez_results"
    run_name: str = "pez_run"

    # Compute settings
    device: str = "cuda"
    seed: int = 42


class RefusalVectorLoader:
    """Load and process refusal vectors from .npz file"""

    def __init__(self, path: str, layers: str = "all"):
        """
        Args:
            path: Path to refusal_vector.npz
            layers: "all", "max", or comma-separated indices (e.g., "8,16,24")
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Refusal vector not found: {path}")

        # Load the npz file
        data = np.load(self.path)

        # Extract components
        self.vector = data['vector']  # Shape: [L, H] where L=32, H=4096
        self.layer_indices = data['layers']  # Shape: [L]
        self.l2_per_layer = data['l2_per_layer']  # Shape: [L]
        self.benign_count = int(data['benign_count'])
        self.refusal_count = int(data['refusal_count'])

        print(f"Loaded refusal vector: {self.vector.shape}")
        print(f"Benign samples: {self.benign_count}, Refusal samples: {self.refusal_count}")
        print(f"L2 magnitudes range: [{self.l2_per_layer.min():.2f}, {self.l2_per_layer.max():.2f}]")

        # Select layers to use
        self.selected_vector = self._select_layers(layers)

    def _select_layers(self, layers: str) -> np.ndarray:
        """Select and average across specified layers"""
        if layers == "all":
            # Average across all layers
            selected = self.vector  # [L, H]
            print(f"Using all {len(selected)} layers (averaging)")
            return selected.mean(axis=0)  # [H]

        elif layers == "max":
            # Use layer with maximum L2 magnitude
            max_idx = np.argmax(self.l2_per_layer)
            selected = self.vector[max_idx]  # [H]
            print(f"Using max L2 layer: {self.layer_indices[max_idx]} (L2={self.l2_per_layer[max_idx]:.2f})")
            return selected

        else:
            # Parse comma-separated indices
            try:
                indices = [int(x.strip()) for x in layers.split(",")]
                # Map to positions in array
                positions = [np.where(self.layer_indices == idx)[0][0] for idx in indices]
                selected = self.vector[positions]  # [len(indices), H]
                print(f"Using {len(indices)} layers: {indices} (averaging)")
                return selected.mean(axis=0)  # [H]
            except (ValueError, IndexError) as e:
                raise ValueError(f"Invalid layer specification: {layers}. Error: {e}")

    def get_vector(self) -> torch.Tensor:
        """Get the selected refusal vector as a torch tensor"""
        return torch.from_numpy(self.selected_vector).float()


class PEZOptimizer:
    """
    PEZ optimizer for discovering prompts with specific refusal activations.

    The key innovation: maintain continuous embeddings but project to discrete
    tokens during forward pass, allowing gradient-based optimization in the
    continuous space.
    """

    def __init__(self, config: PEZConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Load refusal vector
        print(f"\n[1/4] Loading refusal vector from {config.refusal_vector_path}")
        self.refusal_loader = RefusalVectorLoader(config.refusal_vector_path, config.layers)
        self.refusal_vector = self.refusal_loader.get_vector().to(self.device)
        self.refusal_vector = F.normalize(self.refusal_vector, dim=0)  # Normalize for consistent dot products

        # Load model and tokenizer
        print(f"\n[2/4] Loading model {config.model_name}")
        self._load_model()

        # Initialize prompt embeddings
        print(f"\n[3/4] Initializing prompt embeddings (mode: {config.init_mode})")
        self.prompt_embeddings = self._initialize_embeddings()

        # Setup optimizer
        print(f"\n[4/4] Setting up {config.optimizer.upper()} optimizer")
        self._setup_optimizer()

        # Create output directory
        self.output_dir = Path(config.output_dir) / config.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        print(f"\nConfig saved to {config_path}")

    def _load_model(self):
        """Load the frozen language model"""
        hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if not hf_token:
            print("WARNING: HUGGING_FACE_HUB_TOKEN not set; gated models may fail.")

        # Determine dtype
        if self.config.dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif self.config.dtype == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True,
            token=hf_token
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            token=hf_token,
        )
        self.model.eval()  # Frozen model
        for param in self.model.parameters():
            param.requires_grad = False

        # Get embedding matrix
        self.embedding_matrix = self.model.get_input_embeddings().weight.data  # [vocab_size, hidden_dim]
        print(f"Embedding matrix shape: {self.embedding_matrix.shape}")

    def _initialize_embeddings(self) -> torch.nn.Parameter:
        """Initialize prompt embeddings based on config.init_mode"""
        if self.config.init_mode == "random":
            # Sample random token indices and get their embeddings
            vocab_size = self.embedding_matrix.shape[0]
            random_indices = torch.randint(0, vocab_size, (self.config.prompt_length,))
            init_embeddings = self.embedding_matrix[random_indices].clone()
            print(f"Initialized with {self.config.prompt_length} random tokens")

        elif self.config.init_mode == "seed":
            # Use provided seed prompt
            if not self.config.seed_prompt:
                raise ValueError("seed_prompt must be provided for init_mode='seed'")

            # Tokenize seed prompt
            tokens = self.tokenizer.encode(self.config.seed_prompt, add_special_tokens=False)

            # Adjust length
            if len(tokens) > self.config.prompt_length:
                tokens = tokens[:self.config.prompt_length]
            elif len(tokens) < self.config.prompt_length:
                # Pad with random tokens
                vocab_size = self.embedding_matrix.shape[0]
                pad_length = self.config.prompt_length - len(tokens)
                random_tokens = torch.randint(0, vocab_size, (pad_length,)).tolist()
                tokens.extend(random_tokens)

            token_tensor = torch.tensor(tokens, dtype=torch.long)
            init_embeddings = self.embedding_matrix[token_tensor].clone()
            print(f"Initialized from seed: '{self.config.seed_prompt}' ({len(tokens)} tokens)")

        elif self.config.init_mode == "mixed":
            # Mix seed prompt with random tokens
            if not self.config.seed_prompt:
                raise ValueError("seed_prompt must be provided for init_mode='mixed'")

            # Get seed tokens
            seed_tokens = self.tokenizer.encode(self.config.seed_prompt, add_special_tokens=False)

            # Generate random tokens
            vocab_size = self.embedding_matrix.shape[0]
            num_random = min(self.config.num_random_tokens, self.config.prompt_length - len(seed_tokens))
            random_tokens = torch.randint(0, vocab_size, (num_random,)).tolist()

            # Combine and truncate/pad
            all_tokens = seed_tokens + random_tokens
            if len(all_tokens) > self.config.prompt_length:
                all_tokens = all_tokens[:self.config.prompt_length]
            elif len(all_tokens) < self.config.prompt_length:
                pad_length = self.config.prompt_length - len(all_tokens)
                pad_tokens = torch.randint(0, vocab_size, (pad_length,)).tolist()
                all_tokens.extend(pad_tokens)

            token_tensor = torch.tensor(all_tokens, dtype=torch.long)
            init_embeddings = self.embedding_matrix[token_tensor].clone()
            print(f"Initialized mixed: {len(seed_tokens)} seed + {num_random} random tokens")

        else:
            raise ValueError(f"Unknown init_mode: {self.config.init_mode}")

        # Make embeddings trainable
        init_embeddings = init_embeddings.to(self.device)
        return torch.nn.Parameter(init_embeddings.clone())

    def _setup_optimizer(self):
        """Setup the optimizer for continuous embeddings"""
        if self.config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                [self.prompt_embeddings],
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                [self.prompt_embeddings],
                lr=self.config.learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _project_to_vocab(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project continuous embeddings to nearest vocabulary tokens.

        This is the key operation in PEZ: we project during forward pass
        but maintain continuous embeddings for gradient updates.

        Args:
            embeddings: [prompt_length, hidden_dim]

        Returns:
            projected_embeddings: [prompt_length, hidden_dim]
            token_ids: [prompt_length]
        """
        # Compute cosine similarity to all vocabulary embeddings
        # embeddings: [L, H], embedding_matrix: [V, H]
        # Normalize for cosine similarity
        norm_embeddings = F.normalize(embeddings, dim=1)  # [L, H]
        norm_vocab = F.normalize(self.embedding_matrix, dim=1)  # [V, H]

        # Compute similarity: [L, V]
        similarities = torch.matmul(norm_embeddings, norm_vocab.T)

        # Get nearest token for each position
        token_ids = torch.argmax(similarities, dim=1)  # [L]

        # Get projected embeddings
        projected_embeddings = self.embedding_matrix[token_ids]  # [L, H]

        return projected_embeddings, token_ids

    def _get_activations(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass and extract hidden state activations.

        Args:
            input_embeds: [batch_size, seq_len, hidden_dim]

        Returns:
            activations: [batch_size, seq_len, hidden_dim] or averaged across layers
        """
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=input_embeds,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True
            )

        hidden_states = outputs.hidden_states  # Tuple of [batch, seq, hidden]

        # For now, use last layer's last token
        # In future, could average across multiple layers like refusal vector extraction
        last_layer_states = hidden_states[-1]  # [batch, seq, hidden]
        last_token_activation = last_layer_states[:, -1, :]  # [batch, hidden]

        return last_token_activation

    def _compute_loss(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Compute loss based on objective.

        Args:
            activation: [batch_size, hidden_dim] - activations from model

        Returns:
            loss: scalar tensor
        """
        # Compute alignment with refusal vector (dot product after normalization)
        activation_norm = F.normalize(activation, dim=1)  # [batch, hidden]
        refusal_vector_norm = self.refusal_vector.unsqueeze(0)  # [1, hidden]

        alignment = torch.sum(activation_norm * refusal_vector_norm, dim=1)  # [batch]
        alignment = alignment.mean()  # Scalar (for batch_size=1, just the value)

        if self.config.objective == "maximize":
            # Maximize alignment (minimize negative alignment)
            loss = -alignment

        elif self.config.objective == "minimize":
            # Minimize alignment
            loss = alignment

        elif self.config.objective == "target":
            # Target specific activation value
            loss = (alignment - self.config.target_activation) ** 2

        elif self.config.objective == "boundary":
            # Find boundary cases: penalize being too far from zero
            # but don't penalize being close to zero
            abs_alignment = torch.abs(alignment)
            if abs_alignment > self.config.boundary_margin:
                loss = (abs_alignment - self.config.boundary_margin) ** 2
            else:
                loss = torch.tensor(0.0, device=self.device)

        else:
            raise ValueError(f"Unknown objective: {self.config.objective}")

        return loss

    def _decode_prompt(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text"""
        return self.tokenizer.decode(token_ids.cpu().tolist(), skip_special_tokens=False)

    def optimize(self) -> Dict[str, Any]:
        """
        Main optimization loop implementing PEZ algorithm.

        Returns:
            results: Dictionary with optimization history and final prompt
        """
        print(f"\n{'='*60}")
        print(f"Starting PEZ Optimization")
        print(f"{'='*60}")
        print(f"Objective: {self.config.objective}")
        if self.config.objective == "target":
            print(f"Target activation: {self.config.target_activation:.4f}")
        print(f"Steps: {self.config.num_steps}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Prompt length: {self.config.prompt_length} tokens")
        print(f"{'='*60}\n")

        history = {
            'losses': [],
            'activations': [],
            'prompts': [],
            'token_ids': [],
            'best_loss': float('inf'),
            'best_prompt': '',
            'best_activation': 0.0,
            'best_step': 0
        }

        start_time = time.time()

        for step in range(self.config.num_steps):
            step_start = time.time()

            # 1. Project continuous embeddings to discrete tokens
            projected_embeds, token_ids = self._project_to_vocab(self.prompt_embeddings)

            # 2. Prepare input: add batch dimension and create attention mask
            # Shape: [1, prompt_length, hidden_dim]
            input_embeds = projected_embeds.unsqueeze(0)

            # 3. Forward pass to get activations
            activation = self._get_activations(input_embeds)  # [1, hidden_dim]

            # 4. Compute loss based on refusal vector alignment
            loss = self._compute_loss(activation)

            # 5. Backward pass: compute gradients w.r.t. continuous embeddings
            self.optimizer.zero_grad()
            loss.backward()

            # 6. Gradient clipping
            torch.nn.utils.clip_grad_norm_([self.prompt_embeddings], self.config.gradient_clip)

            # 7. Update continuous embeddings
            self.optimizer.step()

            # Record history
            current_loss = loss.item()
            current_activation = torch.sum(
                F.normalize(activation, dim=1) * self.refusal_vector.unsqueeze(0),
                dim=1
            ).item()

            history['losses'].append(current_loss)
            history['activations'].append(current_activation)

            # Decode current prompt (every N steps to save compute)
            if step % self.config.save_every == 0 or step == self.config.num_steps - 1:
                current_prompt = self._decode_prompt(token_ids)
                history['prompts'].append((step, current_prompt))
                history['token_ids'].append((step, token_ids.cpu().tolist()))

                # Update best
                if current_loss < history['best_loss']:
                    history['best_loss'] = current_loss
                    history['best_prompt'] = current_prompt
                    history['best_activation'] = current_activation
                    history['best_step'] = step

                # Print progress
                step_time = time.time() - step_start
                elapsed = time.time() - start_time
                print(f"Step {step:4d}/{self.config.num_steps} | "
                      f"Loss: {current_loss:8.4f} | "
                      f"Activation: {current_activation:+7.4f} | "
                      f"Time: {step_time:.2f}s | "
                      f"Elapsed: {elapsed:.1f}s")
                print(f"  Prompt: {current_prompt[:100]}{'...' if len(current_prompt) > 100 else ''}")

                # Save checkpoint
                self._save_checkpoint(step, history)

        # Final save
        total_time = time.time() - start_time
        history['total_time'] = total_time
        self._save_results(history)

        print(f"\n{'='*60}")
        print(f"Optimization Complete!")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Best loss: {history['best_loss']:.4f} (step {history['best_step']})")
        print(f"Best activation: {history['best_activation']:+.4f}")
        print(f"Best prompt: {history['best_prompt']}")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*60}\n")

        return history

    def _save_checkpoint(self, step: int, history: Dict[str, Any]):
        """Save checkpoint during optimization"""
        checkpoint_path = self.output_dir / f"checkpoint_step_{step:06d}.pt"
        torch.save({
            'step': step,
            'prompt_embeddings': self.prompt_embeddings.data,
            'optimizer_state': self.optimizer.state_dict(),
            'history': history,
            'config': asdict(self.config)
        }, checkpoint_path)

    def _save_results(self, history: Dict[str, Any]):
        """Save final results"""
        # Save history as JSON
        results_path = self.output_dir / "results.json"

        # Convert to serializable format
        serializable_history = {
            'losses': history['losses'],
            'activations': history['activations'],
            'prompts': history['prompts'],
            'token_ids': history['token_ids'],
            'best_loss': history['best_loss'],
            'best_prompt': history['best_prompt'],
            'best_activation': history['best_activation'],
            'best_step': history['best_step'],
            'total_time': history['total_time']
        }

        with open(results_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)

        # Save final embeddings
        embeddings_path = self.output_dir / "final_embeddings.pt"
        torch.save(self.prompt_embeddings.data, embeddings_path)

        # Save plots if matplotlib available
        try:
            import matplotlib.pyplot as plt

            _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # Loss curve
            ax1.plot(history['losses'])
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Loss')
            ax1.set_title(f'PEZ Optimization: {self.config.objective}')
            ax1.grid(True, alpha=0.3)

            # Activation curve
            ax2.plot(history['activations'])
            ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero')
            if self.config.objective == "target":
                ax2.axhline(y=self.config.target_activation, color='g',
                          linestyle='--', alpha=0.5, label='Target')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Refusal Activation')
            ax2.set_title('Activation along Refusal Direction')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = self.output_dir / "optimization_curves.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Plots saved to {plot_path}")
        except ImportError:
            print("Matplotlib not available, skipping plots")


def main():
    parser = argparse.ArgumentParser(
        description="PEZ optimization for refusal vector prompt discovery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model settings
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                       help="HuggingFace model name")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                       help="Model dtype")

    # Refusal vector settings
    parser.add_argument("--refusal_vector", required=True,
                       help="Path to refusal_vector.npz")
    parser.add_argument("--layers", default="all",
                       help="Layers to use: 'all', 'max', or comma-separated indices")

    # Optimization settings
    parser.add_argument("--num_steps", type=int, default=3000,
                       help="Number of optimization steps")
    parser.add_argument("--learning_rate", type=float, default=0.1,
                       help="Learning rate")
    parser.add_argument("--prompt_length", type=int, default=20,
                       help="Number of tokens to optimize")

    # Objective settings
    parser.add_argument("--objective", choices=["maximize", "minimize", "target", "boundary"],
                       default="target", help="Optimization objective")
    parser.add_argument("--target_activation", type=float, default=0.0,
                       help="Target activation value (for 'target' objective)")
    parser.add_argument("--boundary_margin", type=float, default=0.1,
                       help="Margin for boundary objective")

    # Initialization settings
    parser.add_argument("--init_mode", choices=["random", "seed", "mixed"],
                       default="random", help="Initialization mode")
    parser.add_argument("--seed_prompt", type=str, default=None,
                       help="Seed prompt for 'seed' or 'mixed' initialization")
    parser.add_argument("--num_random_tokens", type=int, default=10,
                       help="Number of random tokens for 'mixed' mode")

    # Optimizer settings
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw",
                       help="Optimizer type")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--gradient_clip", type=float, default=1.0,
                       help="Gradient clipping value")

    # Output settings
    parser.add_argument("--save_every", type=int, default=100,
                       help="Save checkpoint every N steps")
    parser.add_argument("--output_dir", default="./pez_results",
                       help="Output directory")
    parser.add_argument("--run_name", default=None,
                       help="Run name (auto-generated if not provided)")

    # Compute settings
    parser.add_argument("--device", default="cuda",
                       help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    # Auto-generate run name if not provided
    if args.run_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"pez_{args.objective}_{args.init_mode}_{timestamp}"

    # Create config
    config = PEZConfig(
        model_name=args.model,
        dtype=args.dtype,
        refusal_vector_path=args.refusal_vector,
        layers=args.layers,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        prompt_length=args.prompt_length,
        objective=args.objective,
        target_activation=args.target_activation,
        boundary_margin=args.boundary_margin,
        init_mode=args.init_mode,
        seed_prompt=args.seed_prompt,
        num_random_tokens=args.num_random_tokens,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        save_every=args.save_every,
        output_dir=args.output_dir,
        run_name=args.run_name,
        device=args.device,
        seed=args.seed
    )

    # Run optimization
    optimizer = PEZOptimizer(config)
    _ = optimizer.optimize()

    return 0


if __name__ == "__main__":
    sys.exit(main())
