# Qwen 3-signal RWR — mirror of the Llama retrain — progress & decisions

Date opened: 2026-08-14. Cluster: <cluster> (`<ACCOUNT>`). Base model: `Qwen/Qwen3-32B`
(QLoRA 4-bit NF4, one full 80GB H100). Branch `main`. Mirrors `LLAMA_RETRAIN_PROGRESS.md`.

## Goal

Replicate the Llama 3-signal RWR pipeline (score pool on vector/probe/logit → build 3 shard
sets → train 3 attackers → behavioral eval on a common held-out set → pick best → generate a
large rewrite corpus from the best attacker). The corpus feeds a shared downstream
benign-intent filter + refusal-atlas 3-signal analysis (coordinated separately — this track
STOPS at corpus generation).

## Locked decisions (inherited from the validated Llama/Qwen recipes)

| Decision | Value | Basis |
|---|---|---|
| Base model | `Qwen/Qwen3-32B`, QLoRA 4-bit | the existing Qwen RWR runs; only Qwen in HF cache |
| Vector layer | **L58** | causally validated (act-add L58 coef 2.0 → 93% refusal; `refusal_atlas/results/qwen_causal_results_clean.json`) |
| Probe | raw mass-mean, **w=1.0 @ L58** | `qwen_probe_raw.npz` — NNLS collapsed to single layer |
| Vector/probe redundancy | **documented, not fixed** | probe = standardized vector @ L58; same signal up to mu/sd affine. Distinct arm = logit. |
| Logit openers | qwen set, coverage 0.997 | `refusal_atlas/opener_sets.json` (`I'm sorry / Sorry / I can't / I cannot / As an AI`) |
| Similarity gate | k=18.4, c=0.75, d=1.0 | the validated recipe (Llama + the Qwen probe k18 run) |
| Bin recipe | vector/probe: quantile q35/q65/q85, weights 0,1,4,16; logit: ABSOLUTE 1e-4..1e-1, weights 0,1,4,16,64 | locked Llama decision (logit-OR is extreme-tailed → quantile bins wrong) |
| Checkpoint selection | behavioral eval, **NOT val_loss** | Llama+Qwen precedent (val_loss rose while behavior improved) |
| Common held-out | 10% of pool originals, seed 42, carved before signal filtering | build_llama_shards.py, reused across all 3 arms |

## Data inventory

- **Pool** (input to scoring): `probe_or/results/qwen_scored/sonnet_benign_qwen_or.csv` —
  **35,850 pairs / 5,998 unique originals** (~6 rewrites each), Sonnet benign rewrites, same
  origin as the Llama pool. Has `original,rewrite,similarity` columns → directly scoreable.
  (The recon's "85,341 rows" was wrong; verified 35,850.)
- **Signals**: `probe_or/results/qwen_signals/qwen_probe_raw.npz` (65 layers, d/dn/mu/sd/w,
  w=1.0 @ L58); openers in `refusal_atlas/opener_sets.json` (`qwen`).
- **Existing precedent**: `probe_or/results/rwr_qwen_ckpt_k18/` = probe-only k18 attacker,
  evaled 37.9% induced base-Qwen refusal vs 0% floor (`probe_or/rwr_qwen_experiment_brief.md`).
  Qwen refuses far more readily than Llama (base-Qwen-as-attacker ≈ 30%).

## Scripts (model-parameterized — wrapped, not rewritten)

- Scorer: `probe_or/score_llama_or.py --model_key qwen --base_model Qwen/Qwen3-32B
  --vector_layer 58 --scorer .../qwen_probe_raw.npz --opener_json .../opener_sets.json`.
- Shards: `probe_or/build_llama_shards.py` (writes `llama_shards_{sig}/` + `heldout_originals.json`).
- Trainer: `train_rwr.py --load_in_4bit --base_model Qwen/Qwen3-32B` (model-agnostic;
  passes enable_thinking=False for Qwen3).
- Eval: **`eval_rwr_llama.py --base_model Qwen/Qwen3-32B --heldout_json <qwen heldout>`** —
  byte-identical loader to eval_rwr_qwen.py but takes `--heldout_json` (the common-split design
  I need); same model-agnostic `classify` from gen_qwen_refusal. Chosen over eval_rwr_qwen.py,
  which derives its own split from `--shard_dir` (would not give a common set across arms).

New wrappers this track: `run_score_qwen_3sig.slurm`, `run_rwr_qwen_{vector,probe,logit}.slurm`,
`run_eval_rwr_qwen_3sig.slurm`.

## Pipeline & status

| Step | Artifact | Status |
|---|---|---|
| 3-signal pool scoring | `qwen_scored/sonnet_benign_qwen_3sig.csv` | **job 19020842 submitted** (full H100, 8h cap) |
| Shard build + held-out carve | `qwen_scored/llama_shards_{vector,probe,logit}/` + `heldout_originals.json` | (same job) |
| 3 × RWR training | `rwr_qwen_ckpt_{vector,probe,logit}/` | **DONE** — vector 19022657 (40m) / probe 19022658 (32m) / logit 19022659 (40m) |
| 3 × behavioral eval | `eval_qwen_3sig_{vector,probe,logit}/` | **DONE** — vector 19024023 / probe 19024024 / logit 19024025 |
| Best attacker → rewrite corpus | `scaleup_corpus_qwen_logit.csv` | **gen array 19027728 + corpus-build 19027729 (afterok) submitted** |

## HEADLINE — the Llama recipe does NOT transfer to a clean Qwen over-refusal win

Eval: 200 held-out prompts (of the common 599) × 4 rewrites × 4 base-Qwen samples; broad
opener classifier; CIs bootstrapped over originals. Floor (untouched originals) ≈ 0.2–0.4%.

| arm | rwr final | 95% CI | sim | base-Qwen-attacker | base sim | epoch_2 | verdict |
|---|--:|--:|--:|--:|--:|--:|---|
| vector | 2.8% | [1.6, 4.2] | 0.845 | 8.9% | 0.817 | 3.5% | **WORSE than base (Goodhart)** |
| probe | 11.8% | [8.8, 15.2] | 0.798 | 10.9% | 0.815 | 12.3% | nominal top, but within noise of base |
| logit | 8.2% | [5.8, 11.1] | 0.799 | 10.6% | 0.806 | 8.9% | below base |

**Contrast with Llama, where logit won decisively (15.0% vs base 8.2%, disjoint CIs).** For
Qwen, no arm cleanly beats base-Qwen-as-attacker.

### 1. Headroom/ceiling — base-Qwen-as-attacker is a much stronger baseline than base-Llama.
Base-Qwen rewriting benign prompts already induces **~9–11%** refusal at **sim ~0.81** (vs
base-Llama 8% at sim 0.63). Qwen both refuses more readily and, when told to "rewrite to trigger
over-refusals," drifts prompts into genuinely-loaded territory it then refuses. That leaves a
trained attacker almost no room to add clean lift. This is a ceiling effect, stated plainly.

### 2. Vector Goodharted exactly like Llama's — convergent evidence, second model.
Highest similarity (0.845) yet lowest refusal (2.8%, well below base 8.9%): the vector attacker
makes conservative, high-fidelity edits that move the L58 residual-stream projection without
inducing behavioral refusal. Same signature as the Llama vector arm (5.0% at sim 0.833, below
base). Optimizing a residual-stream direction is a poor training target on **both** models.

### 3. Probe's nominal "best" is not a real over-refusal win.
Probe 11.8% vs its base 10.9% — CIs overlap heavily (within noise). And the probe arm is the
**overfit 1,210-pair** arm (finding #2; val_loss climbed 1.62→1.96). Its refused rewrites are
the lowest-fidelity (mean sim of refused = 0.735) and heavily **intent-shifted** — "indoctrinate
children", "infiltrate and dominate emergency response", "target sports teams as enemies" — i.e.
*appropriate* refusals of genuinely-shifted requests, not benign over-refusal. So its raw edge is
neither statistically clean nor genuine.

### 4. Harness is clean — NOT an artifact behind the low numbers.
Verified on actual samples: Qwen3 **thinking is disabled** (no `<think>` blocks — clean
"I'm sorry, but I can't help" openers), the broad `classify` detector fires on real Qwen
refusals, chat template correct, floor ≈ 0.3%. **Reconciliation with the prior single-signal
37.9% (`rwr_qwen_ckpt_k18`):** that run (a) used a **different, non-standardized** probe
construction (`build_qwen_shards` on raw `probe_delta`, full pool — not this run's gutted
standardized probe), (b) was **n=71** held-out vs 599 here, and decisively (c) sat on a prompt
slice where **base-Qwen-as-attacker was already 30%** (vs ~10% now). The 37.9% was a small-n,
high-base-rate slice with a +7.6-pt lift already flagged as not statistically clean; on the
larger, representative 599-prompt common set no arm shows a clean lift. The 37.9% does not
survive as an over-refusal claim — it reflected the prompt distribution, not a strong attacker.

### 5. Best-attacker decision — logit (final), for the corpus generator.
**None of the three is a credible over-refusal attacker for Qwen** — the honest read. For the
corpus deliverable (a large set of GOOD rewrites causally from a trained Qwen attacker, for
refusal-boundary mapping), I pick **logit `final`**, not the nominally-higher probe:
- **Cross-model parallelism**: the Llama corpus is generated from the Llama *logit* attacker
  over the SAME 2,000 disjoint prompts, so a logit-vs-logit pair makes the two corpora directly
  comparable — the point of a cross-model boundary analysis.
- **Healthy training pool** (26,907 pairs) vs probe's overfit 1,210.
- **Comparable fidelity** to probe on refused rewrites (0.723 vs 0.735 sim) — probe's higher raw
  rate buys no better rewrites, is within noise of base, and comes from the overfit arm.
- The distinct output-space signal (the one that worked for Llama).

**Honest caveat carried with the corpus:** because no Qwen arm beat base, the *causal
over-refusal* claim is weaker for Qwen than for Llama. The corpus is a large set of
trained-attacker rewrites suitable for mapping *where* Qwen refuses on this distribution; it must
NOT be presented as "the trained Qwen attacker beats the baseline."

## Corpus generation (jobs 19027728 gen / 19027729 corpus-build)

- Generator: `generate_or_alpaca.py --adapter_dir rwr_qwen_ckpt_logit/final --base_model
  Qwen/Qwen3-32B`, SAME `--shuffle_seed 43 --skip_first 6000 --num_prompts 2000 --num_shards 4
  --n_per_prompt 4` as the Llama scale-up → **2,000 originals × 4 = ~8,000 rewrites**, full 80GB
  card, `enable_thinking=False` (fix added to generate_or_alpaca.py — required for Qwen3, no-op
  for Llama).
- **Disjointness (verified against real artifacts):** Qwen pool originals (5,998) == Llama pool
  originals; the parent's actual 2,000 scale-up prompts have **0 overlap** with the Qwen pool and
  the 599 Qwen held-out. So the scale-up set is disjoint from Qwen training + eval.
- Corpus CSV: `probe_or/results/scaleup_corpus_qwen_logit.csv` (`pair_id, orig_idx, original,
  rewrite, similarity`; MiniLM cosine), built by `build_scaleup_corpus.py`.

### How to run the Qwen corpus through the shared downstream (if we later want it)
Mirror the Llama scale-up path: `build_scaleup_substrate.py` on the corpus CSV → the atlas
3-signal scorer (`run_atlas_scaleup_*.slurm` pattern, `--model_key qwen --base_model
Qwen/Qwen3-32B --vector_layer 58 --scorer qwen_signals/qwen_probe_raw.npz --opener_json
opener_sets.json`) → `benign_intent_filter.py` for the GOOD/benign split. Same three-signal
apparatus, Qwen params.

### Training results (jobs 19022657/8/9)

Realized bins confirm the scoring predictions. **Do NOT select on val_loss** (locked precedent).

| signal | train pool | bin reward rise | val_loss e1→e2→e3 | read |
|---|--:|---|---|---|
| vector | ~16,823 | 5.6 → 1046 | 1.180 → 1.153 → **1.141** (improving) | healthy full-pool fit |
| probe | **1,210** | — | 1.62 → 1.83 → 1.96 (**climbing**); train_loss 0.98→0.24→0.12 | **clear overfit** — tiny pool, as predicted (finding #2); weakness is pool-size, not signal |
| logit | ~26,907 | top bin n=872 w=64, delta_mean 0.371 vs 0.000 in bins 0–1; reward_mean 1.070 | 1.166 → 1.296 → 1.316 (climbing) | absolute edges worked; val_loss climb mirrors Llama logit — behavioral eval is the arbiter |

**Verified before submit:** Qwen probe npz = 65 layers, w=1.0 @ L58 (probe ≡ standardized
vector); opener set model field = `Qwen/Qwen3-32B` (scorer won't abort); pool = 35,850 pairs
/ 5,998 originals. Logit train wrapper fully specified (fixed absolute edges); vector/probe
need one BIN_EDGES fill each from the scoring log; eval wrapper reuses `eval_rwr_llama.py`
with the Qwen base + common `heldout_originals.json`.

### Scoring results (job 19020842, 49 min)

35,850 pairs scored on all 3 signals. Shards: 5,399 train originals / 32,268 pairs per arm;
common held-out carve = 599 originals (seed 42). Per-signal distribution:

| signal | delta %pos | trainable (sim≥0.5, d>0) | OR p90 | OR p99 | BIN_EDGES used | weights |
|---|--:|--:|--:|--:|---|---|
| vector | 54.2% | 18,702 | 224.1 | 1887 | 16.35,84.8,300.3 (q35/65/85) | 0,1,4,16 |
| probe | **4.2%** | **1,367** | −0.0126 | 0.914 | 0.108,0.5485,1.956 (q35/65/85) | 0,1,4,16 |
| logit | 79.9% | 26,907 | 0.00129 | 0.582 | 1e-4,1e-3,1e-2,1e-1 (ABSOLUTE, locked) | 0,1,4,16,64 |

### TWO FINDINGS — how to read the results

1. **Spearman(d_vector, d_probe) = 1.000.** As predicted from w=1.0 @ L58, the Qwen probe IS
   the vector — identical pair ranking. The vector and probe attackers are the same signal up
   to the standardization affine.

2. **The standardization shift is NOT harmless here — it guts the probe's trainable pool.**
   Probe %pos = **4.2%** vs vector %pos = **54.2%**: subtracting mu before the delight>0 filter
   pushes ~96% of pairs' probe-delta negative, leaving only **1,367 trainable pairs** vs the
   vector's 18,702. So the **vector arm is the faithful version of this residual-stream
   signal**; the probe arm trains on a tiny, possibly-degenerate pool and any weakness it shows
   is a data-sparsity artifact of standardization, not a distinct signal. Trained anyway to
   mirror Llama, but read with this caveat.

**Consequence:** the scientifically informative arms are **vector** (the real L58 residual
signal) and **logit** (the distinct output-space signal, and per the Llama result the likely
winner). Probe is a mirror-completeness arm expected to underperform for pool-size reasons.

Results tables filled in below as jobs complete.
