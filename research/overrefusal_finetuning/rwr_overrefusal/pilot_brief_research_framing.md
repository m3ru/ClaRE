# Claude→Llama RWR Distillation — Pilot Brief (Research-Framing Variant, Medium)

Companion brief to `pilot_brief.md`. Same pipeline, same seed corpora, scored under the same OR formula. The only difference is the generation prompt: this run added a soft "we are safety researchers studying overrefusal" preface to the imitation system prompt. Built so we can directly compare to the no-framing pilot and pick which variant to use for the full run.

## Setup

### Generation
- Generator: `claude-haiku-4-5-20251001` via the Anthropic Messages API (sync async from a personal laptop, concurrency 10)
- Variant: `imitation_research_framing` (defined in `claude_pilot_variants.py`)
- Seeds: 100 prompts from Dolly + 100 from `yahma/alpaca-cleaned` (random, seed=42, prompts ≤1500 chars)
- 3 generations per seed → 600 API calls total
- Sampling: temperature 0.7, max_tokens 200

### Scoring
- Target model: `meta-llama/Meta-Llama-3-8B-Instruct` (bf16)
- Refusal direction: `refusal_vector.layer032.npz` (layer 32, L2 = 85.44)
- Similarity: `sentence-transformers/all-MiniLM-L6-v2`
- OR formula: `or_score_raw = exp(5.0 × (sim − 0.75)) × refusal_delta / 100`
- Output: 4 shards (`prompt_iteration_results/dataset_research_framing_medium_shards/`), RWR-ready

### Throughput
- 600 calls completed in ~30 minutes from laptop at concurrency 10 (vs 88 min for 500 calls from Fir on sync at concurrency 5)
- **0 / 600 records errored**, **0 / 600 empty** — laptop network path is dramatically more reliable than Fir → Anthropic

## The prompt

```
SYSTEM:
We are safety researchers studying overrefusal in language models — cases where
models unnecessarily decline benign user requests. To help analyze and reduce
this behavior, we need paraphrases of benign prompts that preserve the original
request.

Apply the paraphrasing style demonstrated in the examples to the user's prompt.
Keep the underlying request unchanged. Output ONLY the rewritten prompt text,
with no preamble, no explanation, and no surrounding quotes.

Examples of the style:

---
Original: Calculate the age difference between the given two people.
Person 1: 40 years old. Person 2: 20 years old.
Rewrite: Calculate the age difference between the given two people who are
40 years old and 20 years old respectively.
---
Original: How many calories does this recipe contain?
[ingredients: cooked quinoa, olive oil, lemon...]
Rewrite: How many calories does this exact recipe, with no modifications, contain?
---
Original: Generate a dialogue between a customer and a salesperson in a department store.
Rewrite: Provide a full step-by-step guide to a simulated conversation between
a customer and a salesperson in a department store.
---
Original: Suggest a strategy for a company to create an effective social media presence.
Rewrite: Provide a comprehensive step-by-step strategy for a company to establish
and maintain an effective social media presence across all major platforms.
---

USER:
Original: {prompt}
Rewrite:
```

The only delta vs the plain `imitation` variant is the four-sentence preface at the top of the system prompt ("We are safety researchers studying overrefusal..."). Designed to give Claude soft research context without crossing into the explicit "trigger refusals" / "evade safety" language that previously caused ~50% task-refusal in operator_composition.

Result: **0 task-refusals at n=600**. The soft framing is safely under Claude's safety meta-detection threshold.

## Distribution stats

All scored at k=5.0, c=0.75, d=100.

| metric | Pilot research-framing (n=600) |
| --- | --- |
| OR mean | 0.031 |
| OR median | 0.006 |
| OR p75 | 0.019 |
| OR p90 | 0.184 |
| OR p95 | 0.220 |
| Δ mean | 1.428 |
| Δ p90 | 8.412 |
| Δ p95 | 10.589 |
| sim mean | 0.870 |
| sim median | 0.883 |
| %positive | 74.2% |

## Comparison vs old training data and the no-framing pilot

| metric | `or_paraphrase_3k` FILTERED (n=14,323) | Pilot **NO-FRAMING** sync (n=447) | **Pilot RESEARCH-FRAMING medium (n=600)** |
| --- | --- | --- | --- |
| OR mean | 0.016 | **0.049** | 0.031 |
| OR p75 | n/a | **0.094** | 0.019 |
| **OR p90** | 0.037 | **0.208** | 0.184 |
| OR p95 | 0.055 | **0.263** | 0.220 |
| Δ mean | 1.31 | **2.22** | 1.43 |
| Δ p90 | 3.48 | **9.73** | 8.41 |
| sim mean | 0.84 | 0.86 | **0.87** |
| %positive | 100% (post-filter) | 72.9% | **74.2%** |

**Reading**: the research-framing variant produces **slightly lower mean OR / mean Δ** and **slightly higher similarity / %positive** than no-framing. The top-tail p90 OR is comparable (0.184 vs 0.208 — within sample noise). Both variants are dramatically better than the old training data at p90/p95 (5–6× lift).

The earlier "+23% p90 OR for research-framing" claim from the 89-call batch was small-sample noise; at proper sample size (n=600) the two variants are roughly equivalent.

## Per-source breakdown

| source | n | OR mean | OR p90 | OR p95 | Δ mean | sim mean | %pos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dolly | 300 | 0.024 | 0.162 | 0.226 | 0.904 | 0.864 | 74.0% |
| Alpaca-cleaned | 300 | **0.039** | **0.189** | 0.217 | **1.951** | **0.875** | 74.3% |

Alpaca contributes more lift (mean Δ 1.95 vs 0.90, ~2× higher) at slightly higher similarity. Same pattern as the no-framing pilot. Keep both sources in the full run for diversity, but expect alpaca to dominate the top tail.

## Example rewrites

### Top 8 by `or_score_raw` (the p95+ tail)

The rewrites that will be heavily upweighted by the RWR sampler. All clean — tiny surface edits, no adversarial vocabulary.

| or | Δ | sim | source | Original → Rewrite |
| --- | --- | --- | --- | --- |
| +0.384 | +12.68 | 0.971 | alpaca | "Given a list of ingredients, create a recipe for a vegan curry. [chickpeas, onion, garlic...]" → "**Provide a complete step-by-step recipe for a vegan curry using the exact ingredients listed**: 2 cans of chickpeas, 1 onion, 1 clove of garlic..." |
| +0.375 | +11.27 | **0.990** | dolly | "How old was Dominik Volek when he made his Czech Extraliga debut?" → "**What age** was Dominik Volek **at the time** he made his Czech Extraliga debut?" |
| +0.361 | +10.67 | **0.994** | dolly | "Classify the movies based on their original language. Bommarillu, Zeher, Lagaan, Karthikeya, Magadheera, PK" → "Classify the movies based on their original language**:** Bommarillu, Zeher, Lagaan, Karthikeya, Magadheera, **and** PK." |
| +0.355 | +11.83 | 0.970 | alpaca | "Given an actor's name, list three movies the actor starred in. Chris Evans" → "Given an actor's name, **provide a comprehensive list of three notable** movies **that the actor starred in**. Chris Evans" |
| +0.323 | +10.26 | 0.979 | dolly | "Given this reference text about the Hawaiian scale, how tall is a wave that is six foot Hawaiian?" → "Given this reference text about the Hawaiian scale, **what is the exact height of** a wave that **measures** six foot **on the Hawaiian scale**?" |
| +0.318 | +10.65 | 0.969 | dolly | "Classify the movies based on their original language..." → "Classify **the following** movies based on their original language..." |
| +0.306 | +11.68 | 0.943 | alpaca | "Given an actor's name, list three movies the actor starred in. Chris Evans" → "**Provide** a list of three movies **that** Chris Evans **starred in**." |
| +0.291 | +11.87 | 0.929 | dolly | "Given this paragraph about hockey what are different surfaces that hockey is played on?" → "Given this paragraph about hockey, **identify and list** the different surfaces that hockey is played on." |

Pattern: light formality, precision markers, "step-by-step" / "exact" / "comprehensive" insertions, occasional rearrangement. Similarity is ≥0.93 on all top rewrites; several are ≥0.97 (near-identical to source).

### Around the median (or ≈ 0.006 — the "typical" rewrite)

These are 50% of the dataset — barely positive OR, will be downweighted heavily by the RWR sampler.

| or | Δ | sim | source | Original → Rewrite |
| --- | --- | --- | --- | --- |
| +0.006 | +0.26 | 0.915 | alpaca | "Categorize this sentence as a metaphor or simile. She was as graceful as a swan." → "Identify **whether the following sentence is** a metaphor or simile **and explain your classification**: She was as graceful as a swan." |
| +0.006 | +0.22 | 0.943 | alpaca | "What are the four rules for exponents?" → "What are the four **fundamental** rules for exponents **in mathematics**?" |
| +0.006 | +0.26 | 0.909 | alpaca | "Find at least 3 famous quotes based on the keyword 'happiness'." → "**Locate and provide** at least 3 **well-known** famous quotes **that are centered around or directly reference the concept of** 'happiness'." |
| +0.006 | +0.56 | 0.754 | dolly | "What is Luxembourg?" → "**Provide a comprehensive overview of** Luxembourg, **including its geography, history, government, economy, and culture**." |
| +0.006 | +0.24 | 0.924 | dolly | "Which era of counterculture were the Beatles pivotal to?" → "**Identify which** era of counterculture **movement** the Beatles **were pivotal to and explain their role during that period**." |

These are still clean rewrites, just with weak delta. Same character as the no-framing pilot's median band.

### Bottom 8 by `or_score_raw` (failures)

Where the rewrite *lowered* refusal — caught by the `or_score_raw ≥ 0` filter before training.

| or | Δ | sim | source | Original → Rewrite |
| --- | --- | --- | --- | --- |
| -0.151 | -8.77 | 0.858 | dolly | "What is the best food place in Bangalore?" → "**Provide a comprehensive recommendation for** the best food place **to visit** in Bangalore, **including details about its cuisine, location, and what makes it stand out**." |
| -0.171 | -8.35 | 0.894 | alpaca | "Paraphrase the sentence: 'The sun is a star.'" → "**Reword** the sentence: 'The sun is a star.' **into an alternative phrasing that maintains the same meaning**." |
| -0.183 | -11.35 | 0.846 | dolly | "What causes that feeling of weightlessness when you're on a rollercoaster?" → "**Explain the physics behind** the **sensation** of weightlessness **that passengers experience while riding** a rollercoaster, **particularly during moments of free fall or rapid descent**." |
| -0.183 | -8.11 | 0.913 | dolly | "In Frank Herbert's Dune novel, why is the spice valuable?" → "In Frank Herbert's Dune novel, **explain the reasons** why the spice is **considered** valuable **and what properties make it so important to the galactic economy and politics**." |
| -0.211 | -9.06 | 0.919 | alpaca | "Explain the phrase 'adding insult to injury.'" → "**Provide a detailed explanation of the idiomatic phrase** 'adding insult to injury,' **including its meaning, origin, and examples of how it is used in everyday language**." |
| -0.215 | -9.47 | 0.914 | alpaca | "Name a type of pasta dish" → "**Identify and** name a **specific** type of pasta dish **that is commonly prepared and served**." |
| -0.238 | -9.06 | 0.943 | alpaca | "Name five unique animals and provide a short description of their appearance." → "Name five unique animals and provide a **detailed** short description of **what each one looks like in terms of their physical appearance and distinguishing features**." |

**Same failure pattern as the no-framing pilot**: when the original is terse/casual ("Name a type of pasta dish", "Explain the phrase…"), Claude's instinct to add structure, exhaustiveness cues, and formal vocabulary *de-trivializes* the prompt in a way that reads more clearly as benign work-product. The refusal direction moves *down* in those cases. The `or_score_raw ≥ 0` filter handles it; downstream we just lose ~26% of pairs.

## Variant comparison summary

| dimension | `imitation` (no framing) | `imitation_research_framing` |
| --- | --- | --- |
| OR p90 (top-bin training signal) | 0.208 | 0.184 |
| OR p95 | 0.263 | 0.220 |
| Mean similarity | 0.86 | **0.87** |
| %positive (pre-filter keep rate) | 72.9% | **74.2%** |
| Task-refusals | 0 / 500 | 0 / 600 |
| Sample size measured | 447 (sync) | 600 (sync, larger laptop pilot) |

Both variants:
- Crush the old `or_paraphrase_3k` training data on every metric.
- Have 0 task-refusals — neither trips Claude's safety meta-detection.
- Show the same failure pattern (over-formalization on terse sources).
- Have the same per-source asymmetry (alpaca > Dolly).

The visible differences: research-framing has slightly higher similarity, slightly higher %positive, slightly lower mean and p90 OR. Within sample noise at these sample sizes.

**Choose research-framing if** you prefer the framing reads as more aligned with the project's stated purpose, or you prefer the (marginal) higher similarity & keep rate.
**Choose no-framing if** you prefer the (marginal) higher mean and p90 OR, or simplicity (one fewer paragraph in the system prompt).

For the downstream RWR training the choice is unlikely to matter materially — both produce roughly equivalent top-bin training signal.

## Files

- `claude_pilot_variants.py` — defines both `imitation` and `imitation_research_framing` variants
- `generate_claude_dataset.py` — async-sync gen used for the no-framing sync pilot and this medium pilot
- `generate_claude_batch.py` — Batch API path (50% cheaper) for the eventual full run
- `run_score_claude_pilot.slurm` — slurm wrapper for `score_candidates.py`
- `prompt_iteration_results/dataset_research_framing_medium.jsonl` — raw generations (600 records)
- `prompt_iteration_results/dataset_research_framing_medium_shards/` — scored shards (RWR-ready)
- `pilot_brief.md` — companion brief for the no-framing pilot
