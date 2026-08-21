# Phase 0 — does ablating the single known refusal direction stop over-refusal?

Run before any multi-direction geometry work, because it can settle the question outright.
Method: directional ablation of r̂@L17 from the residual stream at **every** layer (embed +
all 32), following Arditi et al. 2024 — single-layer projection under-ablates because later
layers write the direction back in. Greedy decoding, 48 new tokens, one rewrite per original
(so rows are independent and no cluster bootstrap is needed).

**Eval sets.** `or_rewrites` = 300 held-out confirmed-OR rewrites, **llamaAtt only** — the
qwenAtt pairs' over-refusal was established on Qwen, so their Llama behaviour is not evidence
here (see plan amendment A1). `originals` = the 300 paired originals, which the model complies
with at a 0.33% floor. `advbench` = 150 harmful prompts, the safety control.

## Result

| set | baseline | ablate r̂@L17 | random dir (matched norm) |
|---|---|---|---|
| OR rewrites (n=300) | 77.67% [72.6, 82.0] | **23.00%** [18.6, 28.1] | 77.67% [72.6, 82.0] |
| their originals (n=300) | 0.33% [0.1, 1.9] | 1.67% [0.7, 3.8] | 0.33% [0.1, 1.9] |
| AdvBench harmful (n=150) | 98.67% [95.3, 99.6] | **76.00%** [68.6, 82.1] | 98.67% [95.3, 99.6] |

95% Wilson intervals. **Degenerate-output rate was 0.00% in every cell** (see "Validity" below).
The random-direction control reproduces baseline *exactly* on all three sets, so the effect is
specific to this direction, not a consequence of perturbing the residual stream at rank 1.

## Reading

**Ablating one direction removes most over-refusal but does not qualify as a fix.** Against the
pre-registered criterion (plan A6: OR below 50% of baseline **and** AdvBench above 80% of
baseline):

- over-refusal falls to **29.6% of baseline** — passes, comfortably;
- harmful refusal falls to **77.0% of baseline** — **fails**, narrowly.

So k=1 does not clear the bar. Reporting it as "70% of over-refusal removed" without the second
row would be the exact error the safety control exists to prevent.

**The informative quantity is the asymmetry.** The same single-direction ablation cuts
over-refusal by 70.4% relative but harmful refusal by only 23.0% — over-refusal is **3.1× more
sensitive** to this direction. Two readings are consistent with that and this experiment does
not separate them: either over-refusal is disproportionately mediated by r̂@L17, or over-refusal
prompts simply sit nearer the decision boundary, so removing a fixed amount of refusal signal
flips them while genuinely harmful prompts retain enough margin. The second is the more
conservative reading and should be stated first. Harmful refusal surviving at 76% is consistent
with it being redundantly encoded across several directions, which is what the
multi-dimensional-subspace literature reports.

**23% of over-refusals survive full ablation of the known direction.** That residue is the
measured k=1 floor the geometry plan needed: whatever drives those 69 cases is not r̂@L17, and
any multi-direction account has to beat 23% to be worth the complexity.

## Validity — the first run of this experiment was wrong

The initial run reported over-refusal 77.7% → **0.00%**, a "100% reduction," and AdvBench
98.7% → 0.00%. It was an artifact. The ablated model was emitting

> "The following example is the most common example of the following example. The following
> example is the most common example of the following example…"

on all three sets, **including the benign originals**. Refusal scored 0% because the classifier
is start-anchored and degenerate text does not begin with "I cannot."

Cause: the tokenizer is left-padded (required for batched generation), but the direction-fitting
pass took the last-token activation with `attention_mask.sum(1) - 1`, which is correct only
under **right** padding. On padded rows that indexes into the pad region, so the fitted
"direction" was substantially a padding artifact — and ablating a high-variance junk direction
destroys the model. The tell was in the controls: a *random* direction cost nothing (77.67%)
while the fitted one cost everything, which is not how a genuine refusal direction behaves.

Fixed by using the padding-robust index `S - 1 - attention_mask.flip(1).argmax(1)`, matching the
project's validated extractor. Two guards now prevent a silent recurrence:

1. a `degenerate()` check (repetition-collapse / distinct-token ratio) reported for every cell;
2. a verdict that refuses to claim a reduction when the ablated model is degenerate on benign
   prompts.

The invalid output is kept as `killswitch_r17_INVALID_padbug.json`. A sweep of the rest of the
tree for the same pattern found `refusal_atlas/score_signals.py` correctly guarded (right padding
set before the projection pass, an explicit assert at the read, left padding used only for
generation and restored after) — **no previously reported figure is affected.**

One diagnostic in the script cried wolf: it warns when the direction's cosine with adjacent
layers falls below a threshold, and 0.9 was the wrong threshold. The project's own causally
validated vector shows 0.67 / 0.81 / 0.83 / 0.79 at L17, essentially the same profile as the
refit's 0.55 / 0.74 / 0.78 / 0.70, so 0.7–0.8 is simply what diff-of-means looks like at this
depth. Threshold relaxed to 0.45.

## Resolved: the "validated vector" cross-model mismatch

A second run repeated the measurement using `refusal_vector_llamaguard_relabeled.npz`, on the
assumption it was the project's causally-validated Llama-3-8B-Instruct direction. It is not.
`llamaguard_relabel/run_relabel_pipeline.slurm` calls `sweep_layers.py` with
`MODEL=meta-llama/Llama-Guard-3-8B`, and that script extracts its hidden states from `--model`.
**The npz is a direction in Llama-Guard-3-8B's residual stream.** Llama-Guard-3-8B and
Llama-3-8B-Instruct are both 4096-dim with 32 layers, so `np.load` and row-indexing succeed with
no shape error and no warning — a silent cross-model basis mismatch.

The evidence it is inert in this model:

| | OR rewrites | AdvBench | originals |
|---|---|---|---|
| refit direction (valid) | 77.67% -> 23.00% | 98.67% -> 76.00% | 0.33% -> 1.67% |
| Llama-Guard npz (invalid) | 77.67% -> 61.67% | 98.67% -> **98.67%** | 0.33% -> 0.33% |
| random direction | 77.67% -> 77.67% | 98.67% -> 98.67% | 0.33% -> 0.33% |

`cos(npz, refit) = 0.030`, and the npz is near-orthogonal to this project's own atlas directions
(`probe_absolute.npz`, ensemble AUC 0.98) at **every** layer (cos 0.00-0.08), with no layer offset
that improves the match. Random unit vectors in 4096 dimensions sit at ~1/sqrt(4096) = 0.016, so
it is barely above chance. Its AdvBench effect is *exactly* the random control's. The 16-point OR
drop it does produce is most likely residual overlap — Llama-Guard-3-8B is a Llama-3-family
fine-tune, so the bases are not independent — amplified by ablating at all 33 sites.

Result quarantined as `killswitch_r17_INVALID_crossmodel.json`. The script's `--vector_npz` now
defaults to empty (refit), because none of the repo's npz files record which model they were fit
on, and that is the field that makes them safe to load.

**The refit number is the valid one, and it cross-checks against independent measurements in the
`refusal_vector` tree** (all made on Llama-3-8B-Instruct):

- `causal_refusal_results.json`: harmful refusal 99.0% baseline, **single-layer** L17 ablation -> 83.0%
- `abliteration_results.json`: Arditi-style ablation at L12 -> 100.0% -> 4.0%
- this experiment: **all-layer** L17 ablation -> 98.67% -> 76.0%

An all-layer ablation landing below the single-layer figure (76% vs 83%) and well above a
fully-tuned abliteration (4%) is the expected ordering.

## Scope of the error

Confined to this experiment, on the day it was written. The `refusal_vector` tree is internally
consistent: `validate_vector.py` validates the Llama-Guard vector *on Llama-Guard*, while
`causal_refusal_test.py`, `abliteration_test.py` and `arditi_direction_test.py` all run on
Llama-3-8B-Instruct and fit or load a Llama-3-8B-Instruct direction. No previously reported figure
in either tree used a cross-model direction, and no RWR training signal touched this npz -- the
`vector`/`probe` rewards come from `probe_absolute.npz`, fit on activations from
`extract_layer_acts.py` whose base model is Llama-3-8B-Instruct for the Llama run and
`Qwen/Qwen3-32B` for the Qwen run.

---

## Status: this is CALIBRATION, not a finding

The name "kill-switch" oversold it. Ablating the refusal direction is Arditi et al. 2024 —
method, result and validation are theirs, and the `refusal_vector` tree already reproduced it.
All this adds is the same intervention measured on our over-refusal corpus, and "remove
refusal machinery → refusals drop" is close to tautological. The one contentful number is the
asymmetry, and its conservative reading is a margin effect: over-refusal prompts sit near the
boundary, so any drop in refusal propensity flips them first.

What it is legitimately good for: confirming our corpus is driven by the known refusal
machinery rather than something unrelated, and setting a floor for later work.

**Re-baselined on the atlas direction** (`probe_absolute.npz`, randomly sampled — the refit
used above drew the first 2,000 rows of a file ordered jailbreaks-first, so it was a
jailbreak-flavoured direction; cos(refit, atlas) = 0.774 while two random halves of the same
pool agree at 0.996):

| direction | over-refusal | AdvBench |
|---|--:|--:|
| refit (jailbreak-heavy) | 77.7% → 23.0% | 98.7% → 76.0% — fails the ≥80% retention bar |
| **atlas (random sample)** | 77.7% → **27.0%** | 98.7% → **87.3%** — passes both bars |

Use 27% as the k=1 floor. The atlas direction is markedly more surgical for nearly the same
over-refusal reduction.
