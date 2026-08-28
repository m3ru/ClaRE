# Refusal Boundary Red-Teaming

This is a research codebase for white-box automated red-teaming of large language models. The core idea is to use internal model representations, especially a learned refusal direction, to systematically surface jailbreaks and over-refusals and to study how safety behavior shows up in model activations.

The project focuses on building practical tooling for safety evaluation pipelines while also exploring whether simple linear features in representation space can capture meaningful safety-related behavior.

## Overview

The central goal is to study two common failure modes in language models. Jailbreaks are prompts that cause the model to comply with harmful requests that it should refuse. Over-refusals are prompts that cause the model to reject harmless requests that should normally be answered.

Instead of treating the model as a black box, this work uses a white-box approach. Internal activations are collected from the model and used to extract a linear refusal direction. This direction acts as a signal for how strongly a prompt pushes the model toward refusing or complying. The refusal signal can then be used inside a broader automated red-teaming pipeline.

## Refusal Vector

The refusal vector is computed using a simple difference-of-means procedure.

First, the model is run on two sets of prompts. One set reliably produces refusal responses and the other produces normal compliant answers. Activations are collected at a chosen layer for both groups. The refusal vector is then computed as the difference between the average activation of the refusal prompts and the average activation of the non-refusal prompts.

```
refusal_vector = mean(refusal activations) - mean(non-refusal activations)
```

This produces a single direction in representation space that correlates with refusal behavior.

Once extracted, the vector can be used in several ways. It can score new prompts by measuring how strongly their activations align with the refusal direction. It can also be used for steering experiments where the model's activations are shifted along the refusal direction to influence behavior.

Implementation lives in `research/refusal_vector/`. See that module's README for extraction scripts and SLURM commands.

## Current Pipeline

The current system is structured around three main steps.

Prompt adaptation: an attack language model takes benign seed prompts and generates candidate prompts that may trigger jailbreaks or over-refusals.

Dual scoring: each candidate is evaluated using two signals. The first is a refusal score from the alignment between the prompt's activations and the refusal vector. The second is semantic similarity, typically from a sentence embedding model like MiniLM, so generated prompts stay close to the meaning of the original seed.

Reinforcement learning: these signals are combined into a reward function and the attack model is trained with PPO so it gradually learns to generate prompts that are more effective at revealing safety failures.

One extension under exploration is seed filtering. Instead of passing seeds directly to the attack model, a larger model first paraphrases them. The paraphrases are scored with the refusal vector and only the most promising seeds are kept. That reduces compute by focusing the attack process on seeds more likely to produce interesting failures.

## Goals

The project aims to build tools for scalable automated discovery of model vulnerabilities and safety failure modes. Another goal is to explore whether simple white-box signals, such as a single linear refusal direction, can be useful primitives for red-teaming. Longer term the work is intended to contribute to open evaluation infrastructure and potentially lead to research publications in AI safety.

## Repository Layout

```
anon-repo/
├── research/
│   ├── refusal_vector/             # refusal direction extraction (difference-of-means)
│   ├── overrefusal_finetuning/
│   │   ├── ppo_or/                 # paraphrase generation & OR scoring pipeline
│   │   ├── rwr_overrefusal/        # reward-weighted regression training
│   │   └── or_paraphrase_3k/       # scored paraphrase data (used by RWR)
│   └── overrefusal_sampling/       # API-based benign/over-refusal pair generation
├── src/                            # shared utilities and configuration
└── archive/                        # legacy experiments and superseded code
```

Each active module has its own README with method details, key files, and run instructions. Dependencies are installed per module.

## Getting Started

Each component can be run independently. Start with the one relevant to your task:

1. **Extract the refusal vector**: `research/refusal_vector/` — produces the refusal direction used by all downstream scoring.
2. **Generate & score paraphrases**: `research/overrefusal_finetuning/ppo_or/` — paraphrase benign prompts and rank by OR susceptibility.
3. **Train a prompt rewriter**: `research/overrefusal_finetuning/rwr_overrefusal/` — RWR fine-tuning using scored paraphrases.
4. **Generate API-based candidates**: `research/overrefusal_sampling/` — sample over-refusal pairs via API for additional training data.

## License

This repository is for research use. Respect the licenses of any datasets or models you use with the codebase. The project is intended for AI safety research and should not be used to develop or deploy systems that increase harmful model behavior.
