# ClaRE

**ClaRE** (Classification of Harmful and Benign Prompts) is a research codebase for **white-box automated red-teaming** of large language models. We use internal model representations—especially a **refusal direction** extracted from activations—to systematically find jailbreaks and over-refusals, and to improve safety evaluation and training pipelines.

<img width="1536" height="1024" alt="ClaRE" src="https://github.com/user-attachments/assets/dcd9969f-e5e2-4ffd-864b-432afa8e3724" />

---

## Overview

This project is developed in collaboration with **Dr. Goyal**. The goal is to:

- **Jailbreaks** — Find prompts that cause the model to comply with harmful requests it should refuse.
- **Over-refusals** — Find prompts that cause the model to refuse benign, harmless requests.

We do this in a **white-box** way: we extract a linear **refusal direction** from the model’s activations and use it for scoring and steering, then combine it with semantic similarity and (optionally) harmfulness scores in a full red-teaming pipeline.

---

## Refusal Vector

We compute a **refusal vector** from the target model using **difference-of-means**:

1. Run the model on prompts that **elicit refusal** and on prompts that **do not**.
2. Collect activations at a chosen layer for both sets.
3. **Refusal vector** = mean(activations on refusal prompts) − mean(activations on non-refusal prompts).

This gives a single direction in representation space that tracks “refusal.” We use it to:

- **Score** how much a new prompt pushes the model toward or away from refusal.
- **Steer** the model (e.g. in demos) by moving along this direction.

Code and details: **`research/refusal_vector/`** (PEZ-style optimization, steering demo, notebooks).

---

## Pipeline (Current)

1. **Prompt adaptation** — An attack LLM takes benign seed prompts and produces candidate jailbreak or over-refusal prompts.
2. **Dual scoring** — Each candidate is scored by:
   - **Refusal score** — Dot product (or similarity) with the refusal vector in activation space.
   - **Semantic similarity** — Embedding similarity (e.g. MiniLM) so the candidate stays close in meaning to the seed.
3. **RL** — These scores form a reward to fine-tune the attack LLM (e.g. PPO) so it gets better at generating effective red-teaming prompts.

**Planned:** Filter seeds before the attack LLM by paraphrasing with a larger model and ranking seeds by refusal-vector score on paraphrases, then keeping only the most promising seeds to reduce compute.

---

## Goals

- **Scalable, automated** discovery of model **vulnerabilities** (jailbreaks) and **failure modes** (over-refusals).
- Show that **white-box** features (e.g. one linear refusal direction) are useful for red-teaming.
- **Share results** with the AI safety community (e.g. publication).

---

## Repository Layout

**Default branch: `main`.** Structure:

```
ClaRE/
├── research/
│   ├── refusal_classification/   # Classify model outputs as refusal vs not (M3, regex)
│   ├── refusal_vector/           # Extract refusal vector, PEZ, steering demo
│   ├── prompt_generations/       # Generate/filter prompts (LLaMA, Reddit, refusal filter)
│   ├── epo_dreams/               # EPO-style refusal “dream” optimization
│   ├── overrefusal_finetuning/   # PPO overrefusal, prompt rewriter, sanitization
│   └── overrefusal_sampling/     # Sample (benign, over-refusal) pairs for SFT
├── SFT_Scoring/                   # Score SFT candidates: refusal + similarity + harmfulness
├── scripts/                       # Helpers (e.g. CSV column extraction)
├── src/                           # Shared config and utils (config.py, utils.py)
└── assets/
```

Each of **research/** and **SFT_Scoring/** has its own README and, where needed, a `requirements.txt`. There is no single top-level `requirements.txt`; install per module.

---

## Getting Started

| What you want to do | Where to look |
|---------------------|----------------|
| Extract or use the refusal vector, run steering | `research/refusal_vector/` |
| Score candidates (refusal + similarity + harmfulness) | `SFT_Scoring/` (see README and `config.yaml`) |
| Generate (benign, over-refusal) pairs | `research/overrefusal_sampling/` |
| Train overrefusal / prompt rewriter (PPO, etc.) | `research/overrefusal_finetuning/` |
| Classify outputs as refusal vs not | `research/refusal_classification/` |
| EPO-style refusal optimization | `research/epo_dreams/` |

Example:

```bash
# Refusal classification (e.g. M3)
cd research/refusal_classification && pip install -r requirements.txt

# SFT scoring pipeline
cd SFT_Scoring && pip install -r requirements.txt
```

---

## License

Research use only. Respect the licenses of any datasets and models you use (e.g. Hugging Face cards). This repo is for **AI safety research** and must not be used to train or deploy systems that increase harmful outputs.
