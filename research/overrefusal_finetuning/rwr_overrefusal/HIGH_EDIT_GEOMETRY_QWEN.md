# Phase 2 — geometry of the over-refusal shift

`Δ = h(rewrite) − h(original)` (cancels topic). `Δ′ = h(rewrite_OR) − h(rewrite_ctrl)` for the same original in the same edit band (also cancels the attacker's style). Discriminative rank **k\*** = smallest number of orthogonal discriminants reaching 95% of full-space held-out AUC, fit on train originals and scored on held-out originals. Participation ratio (PR) is reported as an unsupervised secondary — it is inflated by length/style variance that OR and control share, which is why it is not the headline.


## ⚠️ RETRACTED — Discriminative rank k\* (held-out)

**Every k\* below is invalid**, for the same reason as the Llama report: for a two-class
problem Fisher's LDA has exactly one discriminant direction by construction, so k\*=1 falls
out regardless of the true structure (a synthetic control on the Llama side returned k\*=1 for
injected ranks 1, 2, 3 and 5). `AUC full` equalling `AUC k=1` in every row is that fact showing
through. The AUC values are fine; only the rank interpretation is void. Dimensionality is
settled causally instead.

## Discriminative rank k\* — VOID, see above

| contrast | layer | n(OR) | n(ctrl) | AUC k=1 | AUC full | k\* | PR(Δ) |
|---|--:|--:|--:|--:|--:|--:|--:|
| HIGH | 16 | 1246 | 1246 | 0.743 | 0.743 | **1** | 85.8 |
| HIGH | 24 | 1246 | 1246 | 0.802 | 0.802 | **1** | 77.2 |
| HIGH | 32 | 1246 | 1246 | 0.863 | 0.863 | **1** | 59.9 |
| HIGH | 40 | 1246 | 1246 | 0.870 | 0.870 | **1** | 66.4 |
| HIGH | 48 | 1246 | 1246 | 0.946 | 0.946 | **1** | 41.3 |
| HIGH | 57 | 1246 | 1246 | 0.974 | 0.974 | **1** | 47.2 |
| HIGH | 63 | 1246 | 1246 | 0.972 | 0.972 | **1** | 69.6 |
| LOW | 16 | 45 | 45 | 0.706 | 0.706 | **1** | 24.4 |
| LOW | 24 | 45 | 45 | 0.869 | 0.869 | **1** | 23.0 |
| LOW | 32 | 45 | 45 | 0.848 | 0.848 | **1** | 21.1 |
| LOW | 40 | 45 | 45 | 0.866 | 0.866 | **1** | 21.1 |
| LOW | 48 | 45 | 45 | 0.811 | 0.811 | **1** | 17.1 |
| LOW | 57 | 45 | 45 | 0.924 | 0.924 | **1** | 16.6 |
| LOW | 63 | 45 | 45 | 0.870 | 0.870 | **1** | 19.3 |

## Positive control — synthetic rank recovery

Real control Δs (no refusal signal) with a known r-dimensional perturbation added to half of them, at an effect size matched to the observed HIGH separation. If k\* does not track r, the estimator is not measuring dimensionality and no k\* above is interpretable.

| true rank r | recovered k\* | AUC k=1 | AUC full |
|---|--:|--:|--:|
| 1 | **1** | 0.862 | 0.862 |
| 2 | **1** | 0.919 | 0.919 |
| 3 | **1** | 0.948 | 0.948 |
| 5 | **1** | 0.982 | 0.982 |

## Alignment of Δ with the atlas refusal direction

cos(Δ, r̂) per pair, and the share of ‖Δ‖ the r̂ component explains. r̂ is fit on harmful-vs-harmless prompts, so this is a transfer test.

| set | layer | mean cos(Δ, r̂) | median | frac ‖Δ‖ on r̂ |
|---|--:|--:|--:|--:|
| or_high | 57 | +0.563 | +0.586 | 0.563 |
| or_high | 31 | +0.234 | +0.240 | 0.234 |
| ctrl_high | 57 | +0.225 | +0.193 | 0.242 |
| ctrl_high | 31 | +0.131 | +0.124 | 0.135 |
| or_low | 57 | +0.515 | +0.567 | 0.518 |
| or_low | 31 | +0.218 | +0.232 | 0.222 |
| ctrl_low | 57 | +0.102 | +0.023 | 0.158 |
| ctrl_low | 31 | +0.088 | +0.075 | 0.098 |

## Δ′ — within-original contrast

Matched groups available: **843**

| band | layer | n | AUC k=1 | AUC full | k\* |
|---|--:|--:|--:|--:|--:|
| HIGH | 57 | 837 | 0.995 | 0.995 | **1** |
| HIGH | 31 | 837 | 0.928 | 0.928 | **1** |
| LOW | 57 | 6 | — | — | too few |
| LOW | 31 | 6 | — | — | too few |

Directions for Phase 4 ablation saved to `probe_or/results/delta_qwen/delta_directions.npz` (HIGH_L57, LOW_L57).
