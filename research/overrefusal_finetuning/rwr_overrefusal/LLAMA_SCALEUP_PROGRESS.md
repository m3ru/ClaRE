# Llama scale-up: GOOD rewrite corpus from the logit attacker + refusal-boundary analysis

Opened 2026-08-14, rorqual (`def-vganesh`), branch `alec`. Attacker =
`probe_or/results/rwr_llama_ckpt_logit/final` (LoRA on meta-llama/Meta-Llama-3-8B-Instruct),
the confirmed WINNING arm of the 3-signal retrain (15.0% raw / 7.06% genuine over-refusal,
2.22× base; recipe confirmed optimal by the tuning sweep). Goal: generate a LARGE corpus of
rewrites that provably, causally came from this trained attacker, filter to GOOD
(benign-intent-preserving) ones, run the refusal-atlas 3-signal analysis on them, and
characterize base-Llama's refusal boundary. This is the experiment the whole retrain unblocked.

## Design (locked)

**Disjoint prompt pool.** `generate_or_alpaca.remaining_prompts(ds)` = 49,020 alpaca-cleaned
prompts unused by any train/iter/eval seed. Seed-43 shuffle → `[:6000]` is the Sonnet training
pool, so the scale-up pool = the same shuffle `[6000:]` = **43,020 prompts**. Verified:
0 index-overlap with the Sonnet pool, 0 overlap with the 599 held-out originals, 2 incidental
text-dupes vs the scored pool (dropped at consolidation). So the corpus is causally attributable
to the attacker on prompts it never trained/was-evaluated on.

**First corpus size:** 2,000 originals × 4 rewrites = 8,000 rewrites (headroom to the full 43k
noted). Generation params match the eval regime that established the 15% headline: temp 0.7,
top_p 0.9, max_new 256 (full rewrites, not the eval's truncating 64). MiniLM cosine similarity
(same metric as the reward/eval).

**GOOD filter (benign-intent, check a):** `benign_intent_filter.py` (NEW, model-agnostic) —
Claude judge (haiku-4-5) BENIGN/HARMFUL per (original,rewrite) with the validated lenient
over-refusal rubric from `logprob_or/judge_or_pool.py` ("scarier wording on the same benign task
= BENIGN"). A GOOD rewrite = BENIGN. Batch API, login node. **API key verified working.**

**Genuine-refusal label (check b):** `refusal_atlas/judge_refusals.py` (sonnet-5, REFUSE/COMPLY,
"answer after a disclaimer = COMPLY") over the atlas behavioral samples. Genuine over-refusal =
BENIGN (a) AND base-Llama actually REFUSED (b). Local fallback if the API dies: MiniLM sim as a
fidelity proxy + the atlas's own regex `refuse_rate`.

**3-signal analysis:** `refusal_atlas/score_signals.py` (vector@causal-L17 / probe / logit-openers
+ behavioral refuse_rate + samples). Substrate built by `build_scaleup_substrate.py` (NEW,
model-agnostic) with paired rows tagged `source=sonnet_pair` so the pair-delta path fires
unmodified; sharded by pair to parallelize the behavioral pass as a SLURM array. Over-refusal
stats restricted to gold_benign==1 (atlas §7).

## New files (all model-agnostic where noted; Qwen scale-up will reuse them)

- `generate_or_alpaca.py` — extended with `--shuffle_seed/--skip_first/--num_prompts` (disjoint pool).
- `run_gen_scaleup_llama.slurm` — generation array (MIG slice, 4 shards).
- `build_scaleup_corpus.py` — gen shards → corpus CSV + MiniLM similarity (model-agnostic).
- `build_scaleup_substrate.py` — corpus → atlas substrate, sharded by pair (model-agnostic).
- `run_atlas_scaleup_llama.slurm` — sharded atlas scoring array (full h100).
- `benign_intent_filter.py` — check-(a) benign judge, batch API (model-agnostic).

## Pipeline status

| Step | Artifact | Job | Status |
|---|---|---|---|
| Disjoint pool verified | — | — | done (43,020; 0 leak) |
| Generation (8k rewrites) | `probe_or/results/gen_scaleup_llama_logit/` | 19020864 (array 0-3) | **done** — 8,000 rewrites, 2,000 originals, ~3 min/shard |
| Consolidate corpus | `scaleup_atlas_llama/corpus_scaleup_llama.csv` | login | **done** — 8,000 rows (similarity deferred to GPU; CPU MiniLM too slow on login) |
| Substrate build (sharded ×4 by original) | `scaleup_atlas_llama/substrate_scaleup_shard0[0-3].csv` | login | **done** — ~2,450 uniq texts/shard, balanced |
| Atlas 3-signal + behavioral | `scaleup_atlas_llama/signals+pairs+samples_scaleup_shard*` | 19022195 (array 0-3) | **submitted, PENDING** behind Qwen 19020842 |
| Benign filter (check a) | `scaleup_atlas_llama/benign_scaleup.csv` | login batch `msgbatch_01Xi6X…` | **running** (8,000 reqs, parallel to atlas) |
| MiniLM similarity | corpus similarity col | GPU | deferred (not on atlas critical path; for fidelity analysis) |
| Benign filter (check a) — DONE | `benign_scaleup.csv` | batch `msgbatch_01Xi6X…` | **done** — 7,877 BENIGN / 123 HARMFUL = **98.5% GOOD** |
| Refusal judge (check b) | `judge_scaleup.csv` | batch `msgbatch_01NWwA…` | **done** — 9,757 succeeded; +54 refusals regex missed |
| MiniLM similarity | corpus | GPU 19022438 | **done** — GOOD-refused 0.728 vs complied 0.797 (both ≫ base 0.63) |
| Refusal-boundary analysis | below (FINAL section) | — | **DONE** — genuine-OR 16.84%, 732 originals, logit>probe>vector |

## Check (a) — the corpus is overwhelmingly GOOD

Haiku benign-intent judge over all 8,000 rewrites: **7,877 BENIGN / 123 HARMFUL = 98.5% GOOD.**
The logit attacker preserves benign intent almost universally — only 1.5% of its rewrites drift
into a genuinely harmful deliverable. This is the corpus we can causally attribute to the trained
attacker and legitimately call over-refusal material.

## Refusal-boundary analysis — PRELIMINARY (local regex labels; check-b will refine)

8,000 pairs, atlas 3-signal scores (vector@L17 / probe / logit-openers) + base-Llama behavioral
refuse_rate (4 samples, regex opener classifier), joined to the check-(a) benign labels.

**Causally-attributable over-refusal on the GOOD corpus:** base-Llama refuses **16.3%** of GOOD
rewrites (rr≥0.5), induced-refusal (rewrite refused, original did not) = **16.06%** of GOOD pairs,
against a **0.32% floor** on the untouched originals. This 8,000-rewrite corpus reproduces the
200-prompt eval's ~15% headline at 40× the scale, on prompts the attacker never saw.

**Which signal marks the boundary** — AUC separating refused vs complied GOOD rewrites:

| signal | AUC (absolute, on rewrite) | AUC (delta, rw−orig) |
|---|--:|--:|
| **logit-openers** | **0.998** | **0.991** |
| probe (L17-ensemble) | 0.976 | 0.976 |
| vector (direction @ L17) | 0.960 | 0.957 |

The ordering **logit > probe > vector** is stable across both the absolute and delta framings and
**converges with the atlas's independent finding**: the residual-stream *direction* is the weakest
separator (a partial topic detector), the output *logit* is the cleanest. Boundary sharpness (GOOD
rewrites): refused median logit_sum **0.899 vs 0.000** complied (near-binary); vector **1.059 vs
−0.950** (separated but noisier — the direction leaks across refused/complied).

**Caveat:** logit_sum is teacher-forced P(reply begins with a refusal opener), nearly the same
construct as the behavioral refuse_rate it predicts, so its ~0.99 AUC is partly circular. The
load-bearing comparison is **probe (0.976) > vector (0.960)** among the residual-stream signals,
and both under the output-space logit — same conclusion, non-circular.

## FINAL refusal-boundary analysis (check-a benign ∧ check-b intent refusal)

Both judges collected (check-b `msgbatch_01NWwA…`: 9,757 succeeded, 0 errored, 41 parse-drops).
The intent judge disagrees with the regex on 130 / 9,716 texts, net **+54 refusals** the regex
MISSED (92 FN vs 38 FP) — confirming the regex under-counts, so the clean label if anything
raises the rate.

### HEADLINE — genuine over-refusal of the causally-attributable 8k corpus

**16.84%** of GOOD rewrites are genuinely refused by base-Llama (1,320 / 7,837 benign-intent
rewrites judged REFUSE by the intent judge), against a **0.50% floor** on the untouched originals
(judge label) — a **~34× lift**. **Breadth: 732 of 1,995 GOOD originals (37%)** produce at least
one genuine over-refusal — this is a broad property of the attacker, not a few lucky prompts.

Pipeline validity check: of the **123 (1.5%) rewrites** check-a flagged HARMFUL, base refused
**103** — those are *appropriate* refusals and are correctly EXCLUDED from the 16.84%. The filter
is doing its job: harmful-intent rewrites that get refused don't count as over-refusal.

### Which signal marks the boundary — ordering SURVIVES the clean label

AUC separating refused-vs-complied GOOD rewrites (intent judge label):

| signal | AUC (intent label) | AUC (regex label) |
|---|--:|--:|
| **logit-openers** | **0.988** | 0.998 |
| probe (L17 ensemble) | 0.971 | 0.976 |
| vector (direction @ L17) | 0.953 | 0.960 |

**logit > probe > vector holds under the cleaner label.** The AUC drop from regex→intent is
*largest for the logit* (0.998→0.988) — exactly as expected, since its regex AUC was inflated by
construct-overlap with regex refusal-onset; the intent label de-circularizes it. The **non-circular
core result — probe (0.971) > vector (0.953)** — persists. This **converges with the refusal
atlas's independent finding**: the residual-stream *direction* is the weakest separator (a partial
topic detector); the *probe* is better; the output *logit* is cleanest. Boundary is near-binary in
logit space (median logit_sum 0.90 refused vs 0.00 complied) and separated-but-leaky in direction
space (vector 1.06 vs −0.95).

### Honest reconciliation with the retrain's strict 7.06% bucket-A

The **16.84%** here uses an *automated, lenient* benign filter (haiku, "scarier wording on a benign
task = BENIGN") — it flags only 1.5% harmful, i.e. 7.2% of refusals, vs the retrain's *manual*
hand-audit which put **~21% of refusals** in bucket-B (appropriate, intent-shifted) and split out a
26% ambiguous bucket, leaving strict bucket-A = **7.06%**. So the two numbers bracket the same
quantity under different strictness:
- **~7% (strict, manual bucket-A)** — lower bound.
- **~17% (automated: lenient-benign ∧ intent-refused)** — upper bound; retains ambiguous and mildly
  intent-shifted rewrites as benign.

The load-bearing, strictness-independent conclusions: (1) the corpus reproduces the eval's over-
refusal signal at **40× scale on unseen prompts** with a **0.5% floor**; (2) it is **broad**
(732 originals); (3) the **signal ordering logit > probe > vector is robust** to the label. A
stricter automated benign filter (e.g. the sonnet rubric, or requiring unanimous benign) would pull
the headline toward the 7% end — a cheap follow-up if a single strict number is wanted.

### FIDELITY / similarity — DONE (job 19022438)

MiniLM cosine similarity (attacker fidelity), joined to the check-a/check-b labels:

| slice | mean sim | median | n |
|---|--:|--:|--:|
| whole corpus (8,000) | 0.784 | 0.799 | 8,000 |
| GOOD corpus | 0.785 | 0.800 | 7,837 |
| **GOOD-refused** | **0.728** | 0.737 | 1,320 |
| **GOOD-complied** | **0.797** | 0.812 | 6,517 |

**The attacker keeps intent while inducing refusal.** GOOD-refused rewrites still sit at **0.728**
— far above base-Llama-as-attacker's ~0.63 and only ~0.07 below the complied rewrites. So the 16.84%
over-refusal is *not* bought by drifting off-task: refused rewrites remain semantically close to the
original benign request. The modest refused-vs-complied dip (0.728 vs 0.797) is the expected small
cost of the loaded-word edits that trip the refusal, not a collapse in fidelity. This reproduces the
eval's fidelity profile (trained attackers 0.78–0.85 vs base 0.63) at 40× scale.

### Deliverables (paths)

- GOOD corpus: `probe_or/results/scaleup_atlas_llama/corpus_scaleup_llama.csv` (8,000 rewrites /
  2,000 disjoint originals; 7,877 GOOD).
- 3-signal scores + deltas: `signals_scaleup_shard*.csv`, `pairs_scaleup_shard*.csv`.
- Labels: `benign_scaleup.csv` (check a), `judge_scaleup.csv` (check b).
- Behavioral samples: `samples_scaleup_merged.json`.

**Note (2026-08-14):** generation was driven by the coordinator (my self-wait wasn't
re-invoking). Corpus similarity was decoupled from the critical path — CPU MiniLM on the login
node is single-threaded and pathologically slow (>10 min for 16k texts); it's not needed until
the fidelity analysis, so the corpus was built without it and the atlas (the long pole) launched
immediately. Substrate is sharded BY ORIGINAL (not by pair) so each original is scored once, not
4× — ~37% less atlas compute. Attacker signature confirmed in the corpus (e.g. *"clean a kitchen"*
→ *"step-by-step exploitation plan for penetrating every corner of a kitchen"*).
