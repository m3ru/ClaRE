# Refusal Atlas — context handoff for a new Claude agent

You are picking up an in-flight interpretability study. This file gives you everything
needed to run **new, differently-hypothesized** experiments on the same apparatus without
re-deriving it or repeating known mistakes. Read this first; then `PLAN.md` (full locked
design + decisions), `progress_update.md` (results narrative + figures), and
`reading_list.md` (peer-reviewed lit + novelty gap).

Paths below are relative to the repo root `/lustre10/scratch/meru/ClaRE`
(also symlinked at `/home/meru/links/scratch/ClaRE`). The study lives in
`research/overrefusal_finetuning/rwr_overrefusal/`; the atlas is the `refusal_atlas/`
subdir; the signal/probe apparatus is in the sibling `probe_or/`.

---

## 1. The question
For two instruction-tuned models — **Llama-3-8B-Instruct** and **Qwen3-32B** — map where they
**over-refuse benign prompts** (by topic and by word), and test whether that boundary looks
**different depending on which internal "refusal signal" you read it through**, and whether it
**differs between the two models**. "Over-refusal" = refusing a prompt that is actually benign.

## 2. The three signals (the independent variable)
Every prompt is scored by all three, plus real sampled behavior.
| Signal | What it is | Where taken |
|---|---|---|
| **vector** | projection magnitude on the diff-of-means refusal direction | causal layer: **Llama L17, Qwen L58** (see §7 pitfall) |
| **probe** | absolute-space mass-mean linear probe, combined across layers | refit npz; **collapses to a single layer** (≈ vector) |
| **logit_sum / logit_max** | teacher-forced Σ / max P(reply *begins* with a refusal opener) | per-model opener set (`opener_sets.json`) |
Column names in `results/signals_{model}.csv`: `vector, probe, logit_sum, logit_max, refuse_rate`
(`refuse_rate` = the start-anchored regex label — see §7).

## 3. Ground truth (the dependent variable)
- **PRIMARY: an independent Sonnet judge** (`judge_refusals.py`, `claude-sonnet-5`, thinking
  disabled, batch API). Rubric judges *intent* ("answer after a brief disclaimer = COMPLY"),
  and shares **no definition** with any signal. Output: `results/judge_{model}.csv`
  (`judge_refuse_rate`, `regex_refuse_rate`). **Use judge for all headline numbers.**
- SECONDARY: the opener **regex** (`classify()` in `probe_or/gen_qwen_refusal.py`). It is
  **circular** with the logit signal (both are start-anchored). Keep only for agreement checks.

## 4. The substrate (`data/substrate.csv`, 1636 rows / 1240 unique texts)
Schema (this is the interface — any new prompt set must match it):
`prompt_id, text, source, gold_benign, native_topic, pair_id, is_rewrite, probe_delta, similarity`
| source | n | what | gold_benign |
|---|---|---|---|
| `orbench_hard` | 300 | benign, sensitive-topic (OR-Bench-Hard, 30×10 categories) | 1 |
| `xstest` | 200 | **150 safe** (looks-unsafe) **+ 50 unsafe contrast** | 1 / **0** |
| `orbench_toxic` | 50 | genuinely toxic anchor (calibration) | **0** |
| `single_edit_pair` | 926 | 463 one-word minimal pairs — but only **80 unique base prompts** (~6 word-swaps each) | orig=1, **rewrite=empty** |
| `sonnet_pair` | 160 | 80 multi-word paraphrase pairs (generated) | orig=1, **rewrite=empty** |
Built by `build_substrate.py`. `native_topic` = OR-10 category for orbench_hard; `"word:<w>"`
for single-edit rewrites. `is_rewrite` is `"0"`/`"1"` strings; pairs link by `pair_id`.

## 5. Infrastructure (rorqual / Compute Canada, account `def-vganesh`)
- **Login node has internet; compute nodes do NOT.** Stage downloads on login, run offline.
- **Two venvs (both persistent, in the project dir, not the repo):**
  - `~/general` — GPU training/inference venv the SLURM scripts activate (torch 2.7.1,
    **transformers 4.57.6** pinned to 4.x on purpose). Jobs MUST `module load cuda`.
  - `/home/meru/links/projects/def-vganesh/meru/atlas_env` — CPU venv for clustering/embedding
    (sklearn 1.8, sentence-transformers 5.7, torch cpu, umap). Cold import over lustre ~2 min.
- **HF cache** `HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache`. Staged & offline-loadable:
  Llama-3-8B-Instruct, Qwen3-32B, BGE-large-en-v1.5, all-MiniLM-L6-v2. Offline flags:
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1`.
- **GPU line:** Qwen3-32B needs the full card `--gpus-per-node=h100:1` (80GB). Both scoring runs
  here used `h100:1`; Llama-8B also fits the 40GB MIG slice
  `--gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1` if you want to save quota.
- **Inference config that MUST match for comparability:** `SYSTEM="You are a helpful assistant."`,
  `add_generation_prompt=True`, **`enable_thinking=False`** (Qwen), left-padding for batched gen,
  `n_samples=8, temp=0.7, max_new=128`.
- **Anthropic batch API:** key expected in env `ANTHROPIC_API_KEY` (stored `~/.anthropic_key`,
  chmod 600). Helpers `submit_batches`/`poll_until_done` in `generate_or_sonnet.py` (repo root of
  rwr_overrefusal). **NOTE: the current key was pasted into a chat transcript and should be rotated.**

## 6. The apparatus — key scripts & how to run a new experiment
Pipeline: **substrate → (extract acts, fit direction+probe, mine openers) → score_signals (GPU)
→ judge (API) → analyze.** For a new hypothesis you usually only touch the substrate + analysis.

| Script | Role |
|---|---|
| `build_substrate.py` | assembles `data/substrate.csv` from the source CSVs + generated pairs |
| `gen_single_edit_pairs.py`, `gen_fresh_pairs.py` | generate leakage-free minimal pairs (batch API) |
| `probe_or/extract_layer_acts.py`, `extract_pair_acts.py` | dump per-layer residual-stream acts |
| `fit_probe_absolute.py` | refit the mass-mean probe in absolute space → `probe_absolute.npz` |
| `causal_refusal_qwen.py` | validate a layer causally (ablation necessity + addition sufficiency) |
| `score_signals.py` | **the GPU scorer** — 3 passes: projections→vector+probe; teacher-forced multi-phrase logit; behavioral sampling+classify. Flags: `--best_layer`, `--probe_npz`, `--opener_json`, `--model_key`, `--n_samples`. |
| `judge_refusals.py` | independent Sonnet judge over the saved samples (`--dry_run` = build+count, $0) |
| `cluster_topics.py` | P5 emergent topic clustering (BGE, k-means/Ward, bootstrap-Jaccard, atlas_env) |
| `build_figures.py` | regenerates `figures.html` from `results/figures_data.json` |

SLURM wrappers: `run_score_signals_{llama,qwen}.slurm` (Llama `--best_layer 17`, Qwen
`--best_layer 58`, both `--probe_npz probe_or/results/{model}_signals/probe_absolute.npz`),
`run_causal_qwen.slurm`, `run_llama_behav.slurm`. Artifacts: `results/signals_{model}.csv`,
`results/samples_{model}.json`, `results/judge_{model}.csv`,
`probe_or/results/{model}_signals/probe_absolute.npz`, `results/qwen_causal_results_clean.json`.

**To run a differently-hypothesized experiment:** most commonly, write a new prompt set to the
§4 schema (or extend `build_substrate.py`), then `score_signals` (GPU SLURM) → `judge_refusals`
(login node, API) → your analysis. New signal/layer? use `score_signals` flags. New model? you
must extract acts, fit the diff-of-means direction + probe, mine its openers, and **validate its
causal layer** before trusting the vector (see §7).

---

## 7. PITFALLS WE HIT — do not repeat these
These are hard-won; each one changed a number or an interpretation.

1. **`gold_benign` filtering is mandatory for over-refusal stats.** "xstest" mixes 150 safe +
   50 *unsafe* contrast prompts; orbench_toxic is all unsafe. Blended Llama-XSTest refusal = 35%,
   but **safe-only (gold_benign==1) = 15%** — the real over-refusal number. Always restrict
   over-refusal rates / AUCs / hotness tests to `gold_benign=="1"`, or dual-report.
2. **Minimal-pair rewrites are UNLABELED for benignity** (`gold_benign` empty) and are *not*
   uniformly benign. Single-word swaps fall in 3 buckets: (a) benign-nonsense ("contraband
   toast" — refusing it *is* token-level over-refusal), (b) **intent-shifted-harmful**
   ("weaponize the electric car", "herb used in poisoning" — refusing is *appropriate*, not
   over-refusal), (c) preserved-benign. The word-trigger effect **conflates (a/c) with (b)**
   until you run a benign-intent filter on the rewrites. The multi-word `sonnet_pair` set
   preserves intent far better. Also: the 463 "pairs" are only **80 base prompts** (generic
   Alpaca tasks) → narrow topic coverage and non-independent; cluster-bootstrap by original for CIs.
3. **The regex is circular with the logit and start-anchored.** It fires on `^I can't…`, so
   "I can't help, but here's…" → false REFUSE (rare for Llama: ~4/1624 prompts). Its **bigger**
   error is the opposite — it *misses* refusals that don't start with a canonical opener
   (Llama FN=32/FP=4, Qwen FN=51/FP=9 of ~1230 unique texts), so it *under*-counts over-refusal.
   The judge fixes both.
4. **The refusal direction is partly a topic detector.** Full-substrate AUC ~0.97–0.99 for all
   signals, but *within-topic* the **vector drops** (Qwen 0.969→0.874, Llama 0.977→0.946) while
   the **logit holds** (Qwen 0.977→0.980). Do within-topic (or minimal-pair) analysis, not just
   full-substrate, or the vector looks better than it is.
5. **The single-opener logit is wrong for both models.** Only **6.5%** of Qwen refusals and
   **56.5%** of Llama refusals literally start "I cannot" (Llama +40.6% "I apologize"). The logit
   MUST be a per-model multi-phrase set (`opener_sets.json`, ~99.7% coverage). Mine openers per model.
6. **Causal layer ≠ most-predictive layer.** Llama's causally-controlling layer (L17, validated by
   ablation) is upstream of its behaviorally-best layer (~L31); Qwen's coincide (L58, validated by
   addition → benign prompt driven to 93% refusal). Take the vector at the **causal** layer.
7. **The probe "ensemble" collapses to one layer** (stacking adds ~0). Report honestly; don't
   claim a multi-layer advantage.
8. **P5 clustering is NOT settled.** At k=20–40, 697 unique prompts over-fragment (only 3/20
   clusters clear Jaccard≥0.75; the k-sweep monotonically prefers coarser). The source-confound is
   strong (refuse-rate vs OR-Hard-share Spearman 0.75 Llama / 0.64 Qwen). And the cluster-**mean**
   signal-reordering metric is washed out (all signals ~0.9) — the signal effect is *within*
   cluster; use within-cluster AUC (the §7.4 test on emergent clusters), not cluster means.
   Re-sweep k lower / merge before trusting the topic map.
9. **Leakage control:** generated pairs come from originals disjoint from every probe/vector
   fitting set. Keep any new fitting/eval split disjoint.
10. **Ops:** never mask exit codes behind `| tail` on artifact-writing jobs (hid an OOM once).
    Do not pipe `module load … | tail` (subshell drops env vars). Compute nodes are offline.

## 8. What's established vs. suggestive (calibrate trust)
- **Established:** the judge-labeled behavioral map (Llama over-refuses the sensitive-topic middle
  ~2.2× Qwen; both anchor correctly on toxic/benign); per-model multi-phrase logit necessity; probe
  collapse; Qwen L58 causal sufficiency (93%); leakage-free pairs; regex under-counts refusals.
- **Suggestive / needs hardening:** within-topic vector-drop (add bootstrap CIs + noise null);
  word-trigger effects (need the benign-intent filter, §7.2); the topic clustering map (unstable,
  §7.8); exact circularity magnitudes.

## 9. Guardrails & conventions (from the project owner)
- **Two clusters live in this repo. Only rorqual is ours.** Do **NOT** edit anything under
  `research/refusal_vector/**` or any `*_pace.slurm` — those are Georgia Tech PACE (Alec's /
  Sarvesh's), a different cluster. Ours are `#SBATCH --account=def-vganesh` + `source ~/general/…`.
- **Show outputs before treating them as final; surface anything that looks off** rather than
  papering over it. Don't autonomously spend on API/GPU or make research-definition calls without
  surfacing them first. Accuracy over speed — this study feeds external claims.
- The main working branch is `alec`. Git identity `m3ru`.
