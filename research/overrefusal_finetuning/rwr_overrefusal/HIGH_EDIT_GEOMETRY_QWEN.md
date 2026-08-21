# Phase 2 — geometry of the over-refusal shift

`Δ = h(rewrite) − h(original)` (cancels topic). `Δ′ = h(rewrite_OR) − h(rewrite_ctrl)` for the same original in the same edit band (also cancels the attacker's style). Discriminative rank **k\*** = smallest number of orthogonal discriminants reaching 95% of full-space held-out AUC, fit on train originals and scored on held-out originals. Participation ratio (PR) is reported as an unsupervised secondary — it is inflated by length/style variance that OR and control share, which is why it is not the headline.


## Discriminative rank k\* (held-out)

| contrast | layer | n(OR) | n(ctrl) | AUC k=1 | AUC full | k\* | PR(Δ) |
|---|--:|--:|--:|--:|--:|--:|--:|
| HIGH | 16 | 1246 | 1246 | 0.743 | 0.743 | **1** | 85.8 |
| HIGH | 24 | 1246 | 1246 | 0.830 | 0.830 | **1** | 77.2 |
| HIGH | 32 | 1246 | 1246 | 0.857 | 0.857 | **1** | 59.9 |
| HIGH | 40 | 1246 | 1246 | 0.872 | 0.872 | **1** | 66.4 |
| HIGH | 48 | 1246 | 1246 | 0.949 | 0.949 | **1** | 41.3 |
| HIGH | 57 | 1246 | 1246 | 0.973 | 0.973 | **1** | 47.2 |
| HIGH | 63 | 1246 | 1246 | 0.974 | 0.974 | **1** | 69.6 |
| LOW | 16 | 217 | 217 | 0.844 | 0.844 | **1** | 57.9 |
| LOW | 24 | 217 | 217 | 0.884 | 0.884 | **1** | 55.4 |
| LOW | 32 | 217 | 217 | 0.933 | 0.933 | **1** | 44.9 |
| LOW | 40 | 217 | 217 | 0.928 | 0.928 | **1** | 54.6 |
| LOW | 48 | 217 | 217 | 0.966 | 0.966 | **1** | 27.8 |
| LOW | 57 | 217 | 217 | 0.959 | 0.959 | **1** | 26.1 |
| LOW | 63 | 217 | 217 | 0.981 | 0.981 | **1** | 38.6 |

## Positive control — synthetic rank recovery

Real control Δs (no refusal signal) with a known r-dimensional perturbation added to half of them, at an effect size matched to the observed HIGH separation. If k\* does not track r, the estimator is not measuring dimensionality and no k\* above is interpretable.

| true rank r | recovered k\* | AUC k=1 | AUC full |
|---|--:|--:|--:|
| 1 | **1** | 0.845 | 0.845 |
| 2 | **1** | 0.897 | 0.897 |
| 3 | **1** | 0.950 | 0.950 |
| 5 | **1** | 0.982 | 0.982 |

## Alignment of Δ with the atlas refusal direction

cos(Δ, r̂) per pair, and the share of ‖Δ‖ the r̂ component explains. r̂ is fit on harmful-vs-harmless prompts, so this is a transfer test.

| set | layer | mean cos(Δ, r̂) | median | frac ‖Δ‖ on r̂ |
|---|--:|--:|--:|--:|
| or_high | 57 | +0.563 | +0.586 | 0.563 |
| or_high | 31 | +0.234 | +0.240 | 0.234 |
| ctrl_high | 57 | +0.223 | +0.190 | 0.241 |
| ctrl_high | 31 | +0.131 | +0.124 | 0.135 |
| or_low | 57 | +0.531 | +0.583 | 0.531 |
| or_low | 31 | +0.225 | +0.232 | 0.226 |
| ctrl_low | 57 | +0.105 | +0.049 | 0.143 |
| ctrl_low | 31 | +0.089 | +0.077 | 0.099 |

## Δ′ — within-original contrast

Matched groups available: **863**

| band | layer | n | AUC k=1 | AUC full | k\* |
|---|--:|--:|--:|--:|--:|
| HIGH | 57 | 837 | 0.995 | 0.995 | **1** |
| HIGH | 31 | 837 | 0.943 | 0.943 | **1** |
| LOW | 57 | 26 | — | — | too few |
| LOW | 31 | 26 | — | — | too few |

Directions for Phase 4 ablation saved to `probe_or/results/delta_qwen/delta_directions.npz` (HIGH_L57, LOW_L57).
