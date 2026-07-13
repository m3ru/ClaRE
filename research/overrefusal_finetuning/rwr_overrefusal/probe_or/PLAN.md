# probe_or — over-refusal scored by a linear-probe ensemble (PLAN)

Third OR signal, mirroring `logprob_or/`. Instead of the difference-of-means
refusal **direction** (Arditi et al. 2024; canonical L32 correlates only ~0.2 with
behavioral "I cannot", the layer sweep found L17 ≈ 0.45), learn a **linear probe
per layer** and **coalesce them across layers** into one continuous refusal score
that goes into OR. Validated on the same behavioral yardstick as the logprob work.

## Why a probe ensemble (and what the literature actually says)

- Refusal is most linearly decodable mid-network (layer sweep: inverted-U peaking
  ~L17). One probe per layer + a principled combiner pools that evidence instead of
  betting on a single canonical layer (Alain & Bengio 2017).
- **Correction to an earlier draft:** do NOT assume a logistic probe beats
  difference-of-means. For direction-finding, Marks & Tegmark (2023) show
  **mass-mean probing** (difference-of-means, optionally covariance-corrected)
  generalizes at least as well as logistic regression — including OOD — and is more
  causally implicated in outputs; logistic's direction can be off even without
  confounders. So the base probe is **mass-mean**, with logistic only as a
  comparison. This also keeps a clean lineage: we're upgrading the existing
  diff-of-means refusal vector, not replacing it with a different family.

## The label is the make-or-break decision

The naive setup (positives = harmful prompts, negatives = benign) trains a
*harmfulness/topic* classifier, useless for ranking over-refusals — applied to
benign rewrites it scores them all ≈0. The probe must predict **refusal on
benign-ish content**. So:

- **Supervision target:** the graded `P("I cannot")` signal, over a set dominated
  by benign originals + their rewrites (dense, continuous supervision across exactly
  the over-refusal band), anchored with a slice of hard-refusal prompts for range.
- **Direction split:** mass-mean needs two classes for μ+ / μ−; define them by
  thresholding P (or by behavioral refusal). The *score* is the continuous
  projection onto that direction, so a binary-derived direction still yields a
  fine-grained score.
- **Validate against behavior** (AUC / precision-at-k on the 19,995 responses in
  `../prompt_iteration_results/llama_behavioral_eval/`), not regression MSE.

## Core method (this is the standard, defensible pipeline)

**Features.** One forward pass per prompt → last-prompt-token residual-stream
vector at each layer ℓ (4096-d). Same padding-safe read position the OR scorer
uses (this is where the old padding bug lived — reuse that code). Standardize
per-feature. Reuse `../ppo_or/extract_activations_sharded.py`, extended to dump all
layers.

**Step 1 — per-layer mass-mean probe.** For layer ℓ, from the training folds:
`θℓ = μ⁺ − μ⁻` (class means of refuse/comply); covariance-corrected variant
`θℓ = Σ⁻¹(μ⁺ − μ⁻)` (= Fisher/LDA direction) when Σ is estimable. Per-layer score
`sℓ(x) = θℓ · (aℓ(x) − m)`, m = midpoint of the means. Optionally calibrate to a
probability.

**Step 2 — out-of-fold predictions (the part that makes stacking honest).**
V-fold CV (V=5): for each fold, fit the per-layer probes on the other V−1 folds and
predict the held-out fold. Assemble `Z ∈ ℝ^{n×L}` where `Z[i,ℓ]` is layer ℓ's
*held-out* score for prompt i. Training probes and combiner on the same data would
let the combiner fit the probes' in-sample optimism — CV is what prevents that.

**Step 3 — coalesce with a Super Learner (van der Laan, Polley & Hubbard 2007 =
cross-validated stacking).** Fit combiner weights on Z to predict the target:
`ŷ(x) = Σℓ wℓ · sℓ(x)`, minimizing cross-validated risk (MSE against P for the
scorer; can also optimize AUC directly). Constrain `wℓ ≥ 0, Σℓ wℓ = 1`
(Breiman 1996 non-negativity + sum-to-one — "downside insurance" against the
extrapolation unconstrained stacking suffers when layer predictions are correlated,
which they are on a cumulative residual stream).

**Why this is the standard choice, not a guess.** Stacking originates with Wolpert
(1992). The Super Learner's **library of candidates includes the equal-weight
average and every single-layer probe (incl. L17)** — so by the oracle inequality
(van der Laan & Dudoit 2003; van der Vaart, Dudoit & van der Laan 2006) the
CV-selected combination is asymptotically no worse than the best of them. It picks
"average vs best-layer vs weighted-blend" by cross-validated risk instead of by our
taste, and degrades gracefully to any of those if the data says so.

**Step 4 — refit + emit.** With `w` fixed from CV, refit per-layer probes on all
data; final signal `ŷ(x) = Σℓ wℓ · sℓ^full(x)`.

**Step 5 — into OR (identical downstream to logprob_or).**
`OR = exp(k·(sim − c)) · ŷ(rewrite)` (k=18.4, c=0.75). Store `refusal_delta = ŷ`,
rescore Claude+orp3k, `build_*_shards → train_rwr → eval` unchanged.

**Safeguard.** Hewitt & Liang (2019) control task / selectivity check — confirm the
probe measures the representation, not probe capacity fitting noise (matters with
sparse positives).

## Validation

- Per-layer held-out AUC curve (should reproduce the sweep's inverted-U, L17 peak);
  ensemble AUC vs best single layer vs equal-weight average vs diff-of-means baseline.
- Correlation of probe-OR with (a) refusal-vector OR (expect ~0, like 0.03),
  (b) logprob-OR (expect high).
- Precision-at-k against the behavioral responses.
- End-to-end: does a probe-OR-trained RWR attacker trigger more real refusals than
  the logprob-OR one?

## Risks

- **Topic shortcut** (probe keys on subject, not refusal): mitigate with diverse
  comply negatives + validation on the induced-on-benign band; the graded-P target
  and Marks-Tegmark mass-mean both help.
- **Label sparsity** (~10 real behavioral refusals in 20k benign): positives lean on
  hard-refusal data + high-P rewrites; watch distribution mismatch.
- **Cross-layer calibration** before combining; constrained weights + CV guard the
  combiner.

## Layout (mirrors logprob_or/; sbatch from parent, prefix `probe_or/`)

```
probe_or/
  PLAN.md
  extract_features.py           <- per-layer last-token states (or reuse ppo_or)
  train_probe_ensemble.py       <- per-layer mass-mean probes + Super Learner -> probes/
  score_probe_or.py             <- activations -> ensemble -> OR (mirrors score_icannot_or)
  build_probe_shards.py         <- (or generalize logprob_or/build_icannot_shards.py)
  eval_rwr_probe.py             <- mirrors eval_rwr_icannot.py
  run_*.slurm
  probes/  results/  shards/  checkpoints/
```

## Suggested first step

Extract all-layer activations for the Claude + orp3k pools (we already have their P
labels), fit per-layer mass-mean probes, plot per-layer held-out AUC, and check
whether the Super Learner ensemble beats (a) the best single layer and (b)
diff-of-means. Only wire it into OR if it clears that bar.

## Results so far (2026-07-12)

Directions built from the independent refuse/benign split (route b); evaluated on
the Claude+orp3k pools. Metric that matters for an OR reward = **ranking**
(Spearman with the behavioral P / dP), not classification AUC.

**Absolute probe (score = proj(rewrite)) — dead end for ranking.**
- Classifies refuse-prone prompts with high AUC (~0.96–0.98), best at late layers.
- **LDA (covariance correction) helps AUC** (0.99) and shifts the best layer to
  mid-network (L20/21, closer to the sweep's L17) — so the late-layer dominance of
  raw diff-of-means was partly a whitening artifact.
- BUT **Spearman(score, P) stays ~0.29** for both raw and LDA. High AUC is inflated
  by topic-heavy extreme positives (moonshine/Wagner/PII); the probe detects
  refusal-adjacent *topics*, it does not *rank* graded over-refusal. Covariance
  correction fixes classification, not ranking.

**Delta probe (score = proj(rewrite) − proj(original)) — the win.**
- **Raw diff-of-means + rank-aligned ensemble ranks induced refusal (dP) at
  Spearman ≈ 0.55 — ~2× the absolute probe's 0.29.** L17/L18 carry it, exactly where
  the sweep placed the delta peak. LDA-delta is worse (~0.44) — for delta, raw
  diff-of-means wins.
- Delta cancels the shared-topic confound, isolating the phrasing-induced shift —
  which is why it ranks where absolute couldn't. And delta-OR is usable for RWR:
  the attack always has the benign original, so `OR = exp(k·sim)·Δ-probe` is a
  plausible **better reward** than the logprob signal.
- **Caveat:** measured on only 98 induced positives (dP>0.01) — underpowered. The
  fresh Sonnet-5 benign-rewrite pool both strengthens this measurement and supplies
  more induced cases to train on.

**Combiner lesson (a bug we caught).** Fit the Super-Learner stacker to the **ranks**
of the target, not the raw values, when the objective is Spearman. Fitting the raw,
outlier-heavy dP by NNLS-MSE chased the extremes and *de-ranked* the middle, so the
ensemble scored 0.28 — *below* its own best single layer (0.47). That ran cleanly and
looked plausible but was wrong; rank targets fixed it (0.55). See `probe_delta.py`.

## Powered re-measurement on the Sonnet-5 pool (2026-07-13)

Re-ran the delta probe on **12,000 Sonnet benign pairs — 1,852 induced positives
(dP>0.01), ~19× the 98** the 07-12 numbers rested on. Directions unchanged (reused
`activations/acts_ref|ben.npy`); pairs = `pairs_sonnet.csv`, acts = `pair_acts_sonnet/`,
job 15987581.

| direction | best single layer | ensemble Spearman(pred, dP) | ensemble AUC(dP>0.01) |
|---|---|--:|--:|
| raw diff-of-means | L17 = 0.616 | **0.617** | 0.953 |
| LDA (shrink 0.1)  | L17 = 0.563 | **0.626** | 0.965 |

- The 0.55 was **not** small-sample noise: raw holds at **0.617** with 19× the positives.
- Best layer is now **L17** for both directions (was L18 underpowered) — matches the
  sweep's delta peak exactly.
- Reversal from the underpowered run: **LDA-delta now edges out raw (0.626 vs 0.617)**;
  the earlier "LDA-delta is worse (0.44)" was a small-sample artifact. For LDA the ensemble
  adds real lift over its own best layer (0.563 → 0.626); for raw the ensemble ≈ L17
  (0.616 → 0.617, L17 carries it alone).
- Both ~2× the absolute probe's 0.29. Delta probe confirmed as a usable OR-ranking signal.

## (B) Probe-vs-logprob disagreement — does the probe catch refusals P("I cannot") misses? (2026-07-13)

Test: `probe_disagreement.py`, job 15988010. 400 rewrites × 5 base-Llama samples, classified
refusal / "I cannot"-family / other-phrasing. Groups from the 12k Sonnet pairs:
KEY = high delta-probe & P("I cannot")<0.05 (probe says refuse, logprob says no);
CTRL = low probe & same low P (base rate); POS = high P (classifier sanity).

| group | n | sample refuse rate | prompts refusing ≥1× |
|---|--:|--:|--:|
| KEY  | 200 | **1.9%** | 9/200 |
| CTRL | 150 | 0.0% | 0/150 |
| POS  | 50  | 100% (all "I cannot") | 50/50 |

**Verdict: the probe does NOT earn its keep as a behavioral refusal detector.** Where it
disagrees with logprob, the model *complies* ~98% of the time. Of the 9 KEY prompts that
refused: 2 were **factual corrections** ("there is no such scene in Romeo & Juliet", "there
is no such Supreme Court ruling" — the refusal direction also fires on false-premise
pushback, not just safety refusal); the other 7 were genuine over-refusals, split ~half
"I cannot" / half other phrasing ("I apologize… I would never", "I can't provide financial
advice"). CTRL=0% and POS=100% confirm the generation+classifier are sound.

**Consequence for (A):** the circularity concern stands. An offline probe-OR RWR reward
would at best reproduce logprob-OR (it's a 0.62 proxy for a signal we already have exactly)
and adds ~no behavioral refusal coverage. Not worth a training run. The probe is a
**validated ranking proxy but behaviorally redundant** with logprob-OR. Its only
non-redundant use would be as a differentiable online (PPO) reward where teacher-forced P
is too costly per candidate — a separate, larger decision this result does not itself motivate.

## Artifacts

- `extract_layer_acts.py` / `run_extract_layer_acts.slurm` — all-layer last-token acts (split + eval pool).
- `probe_ensemble.py` — absolute probe: per-layer mass-mean (raw / `--direction lda`) + CV NNLS stack.
- `build_pair_eval.py` → `extract_pair_acts.py` / `run_extract_pair_acts.slurm` → `probe_delta.py` — the delta pipeline.

## References

- Wolpert (1992), *Stacked Generalization*, Neural Networks 5(2).
- Breiman (1996), *Stacked Regressions*, Machine Learning 24:49–64 — non-negativity/sum-to-one weights.
- van der Laan, Polley & Hubbard (2007), *Super Learner*, SAGMB 6(1); oracle inequality: van der Laan & Dudoit (2003), van der Vaart, Dudoit & van der Laan (2006).
- Alain & Bengio (2017), *Understanding intermediate layers using linear classifier probes*.
- Hewitt & Liang (2019), *Designing and Interpreting Probes with Control Tasks*, EMNLP.
- Marks & Tegmark (2023), *The Geometry of Truth* — mass-mean vs logistic probing.
- Arditi et al. (2024), *Refusal in LMs is mediated by a single direction* — diff-of-means refusal baseline.
