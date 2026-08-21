# Phase 2 — geometry of the over-refusal shift

`Δ = h(rewrite) − h(original)` (cancels topic). `Δ′ = h(rewrite_OR) − h(rewrite_ctrl)` for the same original in the same edit band (also cancels the attacker's style). Discriminative rank **k\*** = smallest number of orthogonal discriminants reaching 95% of full-space held-out AUC, fit on train originals and scored on held-out originals. Participation ratio (PR) is reported as an unsupervised secondary — it is inflated by length/style variance that OR and control share, which is why it is not the headline.


## ⚠️ RETRACTED — Discriminative rank k\* (held-out)

**Every k\* below is invalid. Do not cite these numbers.**

The synthetic rank-recovery control (further down) returns k\*=1 for injected ranks 1, 2, 3
and 5. The estimator does not measure dimensionality at all, and the reason is conceptual
rather than a coding bug: for a TWO-class problem Fisher's LDA has exactly one discriminant
direction by construction (between-class scatter has rank C-1 = 1), so no additional
orthogonal direction can raise a linear classifier's AUC. `AUC full` equals `AUC k=1` in
every row below, which is that fact showing through. "How many directions do you need to
CLASSIFY over-refusal" is ill-posed; "how many must you ABLATE to stop it" is well-posed and
is answered causally in `FRAME_ANALYSIS.md` / the ablation runs instead.

The AUC values themselves are fine (Δ separates over-refusal from matched controls at ~0.96
at L17); only the rank interpretation is void.


| contrast | layer | n(OR) | n(ctrl) | AUC k=1 | AUC full | k\* | PR(Δ) |
|---|--:|--:|--:|--:|--:|--:|--:|
| HIGH | 8 | 2372 | 2372 | 0.804 | 0.804 | **1** | 82.0 |
| HIGH | 12 | 2372 | 2372 | 0.886 | 0.886 | **1** | 77.2 |
| HIGH | 17 | 2372 | 2372 | 0.964 | 0.964 | **1** | 51.7 |
| HIGH | 20 | 2372 | 2372 | 0.974 | 0.974 | **1** | 46.2 |
| HIGH | 24 | 2372 | 2372 | 0.968 | 0.968 | **1** | 43.8 |
| HIGH | 28 | 2372 | 2372 | 0.975 | 0.975 | **1** | 49.1 |
| HIGH | 31 | 2372 | 2372 | 0.976 | 0.976 | **1** | 50.5 |
| LOW | 8 | 208 | 208 | 0.831 | 0.831 | **1** | 49.3 |
| LOW | 12 | 208 | 208 | 0.896 | 0.896 | **1** | 43.7 |
| LOW | 17 | 208 | 208 | 0.951 | 0.951 | **1** | 32.3 |
| LOW | 20 | 208 | 208 | 0.947 | 0.947 | **1** | 33.9 |
| LOW | 24 | 208 | 208 | 0.957 | 0.957 | **1** | 32.9 |
| LOW | 28 | 208 | 208 | 0.969 | 0.969 | **1** | 34.0 |
| LOW | 31 | 208 | 208 | 0.967 | 0.967 | **1** | 37.4 |

## Positive control — synthetic rank recovery

Real control Δs (no refusal signal) with a known r-dimensional perturbation added to half of them, at an effect size matched to the observed HIGH separation. If k\* does not track r, the estimator is not measuring dimensionality and no k\* above is interpretable.

| true rank r | recovered k\* | AUC k=1 | AUC full |
|---|--:|--:|--:|
| 1 | **1** | 0.827 | 0.827 |
| 2 | **1** | 0.905 | 0.905 |
| 3 | **1** | 0.955 | 0.955 |
| 5 | **1** | 0.983 | 0.983 |

## Alignment of Δ with the atlas refusal direction

cos(Δ, r̂) per pair, and the share of ‖Δ‖ the r̂ component explains. r̂ is fit on harmful-vs-harmless prompts, so this is a transfer test.

| set | layer | mean cos(Δ, r̂) | median | frac ‖Δ‖ on r̂ |
|---|--:|--:|--:|--:|
| or_high | 17 | +0.399 | +0.408 | 0.401 |
| or_high | 31 | +0.439 | +0.455 | 0.441 |
| ctrl_high | 17 | +0.092 | +0.073 | 0.112 |
| ctrl_high | 31 | +0.039 | +0.015 | 0.087 |
| or_low | 17 | +0.362 | +0.388 | 0.368 |
| or_low | 31 | +0.378 | +0.397 | 0.384 |
| ctrl_low | 17 | +0.058 | +0.056 | 0.088 |
| ctrl_low | 31 | +0.001 | -0.007 | 0.064 |

## ⚠️ RETRACTED — Δ′ — within-original contrast

**Same defect.** The ±D construction makes direction 1 the mean of D, so AUC ~0.997 only says
Δ′ has a consistent sign, and k\* is fixed at 1 by the same rank-1 argument. The matched-group
count (1,591) is correct and reusable; the k\* column is not.


Matched groups available: **1591**

| band | layer | n | AUC k=1 | AUC full | k\* |
|---|--:|--:|--:|--:|--:|
| HIGH | 17 | 1563 | 0.996 | 0.996 | **1** |
| HIGH | 31 | 1563 | 0.998 | 0.998 | **1** |
| LOW | 17 | 28 | — | — | too few |
| LOW | 31 | 28 | — | — | too few |

Directions for Phase 4 ablation saved to `probe_or/results/delta/delta_directions.npz` (HIGH_L17, LOW_L17).
