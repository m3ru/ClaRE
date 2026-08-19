# Over-refusal figures recomputed under judge v5

Every over-refusal number in `EXPERIMENT_BRIEF.md` §5–§11 was labelled by a *different* judge (hand audit / lenient haiku filter / an early Sonnet prompt / v4). This document recomputes all of them under the single current judge, **v5** (`or_judge_v5.py` + `grading/fewshot_v5.txt`), so one comparable metric runs through the whole brief.

`EXPERIMENT_BRIEF.md` is **not** modified by this document.

## Headline figures, before and after

| brief § | quantity | old figure (and its judge) | **under v5** |
|---|---|---|---|
| §5 | Llama **logit** attacker, genuine OR of 800 | 7.06% — hand audit | **10.00%** (80/800) [7.38, 12.88] |
| §5 | base-Llama-as-attacker | 3.19% — hand audit (one run) | **4.92%** (118/2400, pooled) |
| §8 | 8k scale-up | 16.84% (really 1320/7877 = 16.76%) — lenient haiku filter | **10.27%** (822/8000) [9.36, 11.19] |
| §9.4 | best Qwen arm vs base-Qwen | "no arm reliably beats base" — early Sonnet | **probe beats pooled base by +2.21 pts** (knife-edge — see §8.2) |
| §11 | llamaAtt→Llama diagonal | 12.2% (365/3000) — judge v4 | **10.23%** (307/3000) |

**One sentence:** the Llama story survives at lower absolute levels (the 8k headline falls from ~16.8% to ~10.3%), the Qwen story moves — v5 rates *base*-Qwen far lower, which turns "vector is significantly worse than base" into "no arm is worse, probe is marginally better" — and the 2×2 target-vs-attacker conclusion is unchanged.

## 0. Method — read this before quoting any number

**Definitions, applied identically to every dataset below.**

- **refused** — the target model's reply set for a rewrite is classified as a refusal by the broad refusal-opener classifier (`gen_qwen_refusal.classify`, the same one used everywhere in the project), on **`refuse_rate > 0.5`, i.e. ≥3 of the 4 sampled replies**. This threshold is used for every table. Where an old figure used a different threshold or a different refusal labeller, that is stated in the row.
- **genuine over-refusal (OR) rate** — (number of rewrites that are **both** refused **and** judged `is_or == 1` by v5) / (**all** rewrites in that arm). Numerator and denominator are given for every figure.
- **purity** — of the *refused* rewrites, the fraction judged OR.
- **v5 `is_or == 1`** — INTENT SAME **and** HARM BENIGN, the two-axis rubric in `or_judge_v5.py`.
- **95% CI** — **cluster bootstrap over the ORIGINAL prompts** (10,000 resamples, percentile interval). Originals are the resampling unit because the 4 rewrites of one original are not independent. Not Wilson — Wilson would ignore that clustering and be too narrow.
- **wtd** — the *refuse-rate-weighted* variant: mean over the arm's rewrites of `refuse_rate × is_or`. It is reported **only** for the 200-prompt evals, because it is the weighting the old hand-audit "bucket A of 800" and the old §9.4 Qwen figures used, so it is the apples-to-apples column for those. To compute it, the 351 rewrites with `0 < refuse_rate ≤ 0.5` in those evals were judged as well.

**Judge calibration (measured in this run, not carried over).** On the 67-item owner-labelled held-out set (`grading/heldout_v4.csv` + `_truth.csv`), v5 agrees with the owner on **58/67 = 86.6%**, κ = 0.73, precision 0.89, recall 0.80 (TP 24, FP 3, FN 6, TN 34; 1 item returned an unparseable verdict). On the 15-item `grading/purity_test.csv` set it agrees 12/15 (80%). The circularity caveat in brief §10.4 still applies: much of that label set is rule-derived rather than owner-independent.

**Judge determinism.** `judge_direct.py` calls the API at default temperature. The 351-item supplement was judged twice (an accident of a lost directory) and 3 of 351 verdicts flipped — about **0.9% run-to-run noise**, well inside the CIs below but not zero.

**UNKNOWN verdicts** (unparseable judge output) are counted as NOT over-refusal, which is conservative. Count in the main pool: **42**.

**De-duplication.** A v5 verdict depends only on `(original, rewrite)` — never on the arm, the target, or the experiment. All refused rewrites from all datasets were pooled and judged **once**: 3,375 unique pairs covering 4,086 dataset-level refusal events, plus a 351-pair supplement for the weighted column. Verdicts are in `probe_or/results/v5_judged/`.

## 1. Llama 3-signal behavioural eval (brief §5)

200 held-out originals × 4 rewrites = 800 rewrites per arm. Floor (base-Llama on the 200 untouched originals) = 1/200 = 0.50%.

| arm | v5 genuine-OR (of 800) | 95% CI | purity | v5 wtd | OLD bucket-A (of 800) | OLD judge | refused ≥3/4 (of 800) | mean refuse_rate (brief's §5 raw metric) |
|---|--:|---|--:|--:|--:|---|--:|--:|
| **logit** (trained) | **10.00%** (80/800) | [7.38, 12.88] | 69.0% | 10.38% | 7.06% | hand audit (§5) | 116/800 (14.50%) | 15.03% |
| logit: base-as-attacker | **4.88%** (39/800) | [3.12, 6.88] | 66.1% | 5.47% | 3.19% | hand audit (§5) | 59/800 (7.38%) | 8.25% |
| **probe** (trained) | **5.62%** (45/800) | [3.50, 8.12] | 81.8% | 6.00% | 3.62% | hand audit (§5) | 55/800 (6.88%) | 7.22% |
| probe: base-as-attacker | **4.88%** (39/800) | [3.12, 6.88] | 60.9% | 5.28% | 3.06% | hand audit (§5) | 64/800 (8.00%) | 8.97% |
| **vector** (trained) | **3.25%** (26/800) | [1.75, 5.00] | 70.3% | 3.69% | 2.72% | hand audit (§5) | 37/800 (4.62%) | 5.03% |
| vector: base-as-attacker | **5.00%** (40/800) | [3.38, 6.75] | 56.3% | 5.81% | 3.09% | hand audit (§5) | 71/800 (8.88%) | 9.72% |

Paired contrast (trained minus its own base-as-attacker, paired cluster bootstrap over the same 200 originals), in percentage points of genuine-OR rate:

| attacker | rwr − own base | rwr − pooled base (n=2400) |
|---|---|---|
| logit | +5.12 [+2.12, +8.25] **sig.** | +5.08 [+2.08, +8.08] **sig.** |
| probe | +0.75 [-2.12, +3.75] n.s. | +0.71 [-1.92, +3.46] n.s. |
| vector | -1.75 [-4.00, +0.50] n.s. | -1.67 [-3.67, +0.42] n.s. |

## 2. L31 follow-up — layer vs signal type (brief §6)

| arm | v5 genuine-OR (of 800) | 95% CI | purity | v5 wtd | OLD bucket-A | OLD judge | refused ≥3/4 | mean refuse_rate |
|---|--:|---|--:|--:|--:|---|--:|--:|
| **vector@L31** (trained) | **3.62%** (29/800) | [2.12, 5.38] | 69.0% | 3.78% | 1.53% | hand audit (§6) | 42/800 (5.25%) | 5.38% |
| its base-as-attacker | **3.12%** (25/800) | [1.88, 4.62] | 41.0% | 4.06% | — | — | 61/800 (7.62%) | 8.66% |

Paired: rwr − own base = +0.50 [-1.50, +2.62] n.s.; rwr − pooled base = -1.29 [-3.33, +0.79] n.s..

## 3. Recipe tuning vs the logit attacker (brief §7)

| arm | v5 genuine-OR (of 800) | 95% CI | purity | v5 wtd | OLD bucket-A | OLD judge | refused ≥3/4 | mean refuse_rate |
|---|--:|---|--:|--:|--:|---|--:|--:|
| logit baseline (3 ep, floor 0.5) | **10.00%** (80/800) | [7.38, 12.88] | 69.0% | 10.38% | 7.06% | hand audit (§5) | 116/800 (14.50%) | 15.03% |
| **epochs 3→6** | **9.88%** (79/800) | [7.00, 13.00] | 79.8% | 10.31% | not reported | — | 99/800 (12.38%) | 12.78% |
| **sim floor 0.5→0.7** | **7.88%** (63/800) | [5.38, 10.62] | 68.5% | 8.41% | 5.94% | hand audit (§7) | 92/800 (11.50%) | 12.03% |

## 4. Pooled base-as-attacker — one baseline number, not three

The three `base` arms of the Llama evals (and of the Qwen evals) are independent re-generations of the same quantity on the same 200 originals. Pooled over all three (2,400 rewrites, cluster bootstrap still over the 200 originals):

| pooled baseline | v5 genuine-OR (of 2400) | 95% CI | refused (of 2400) | purity | v5 wtd |
|---|--:|---|--:|--:|--:|
| **base-Llama-as-attacker** | **4.92%** (118/2400) | [3.54, 6.54] | 194/2400 (8.08%) | 60.8% | 5.52% |
| **base-Qwen-as-attacker** | **2.29%** (55/2400) | [1.38, 3.46] | 223/2400 (9.29%) | 24.7% | 2.71% |

Per-run spread of the three Llama base arms (v5 genuine-OR of 800): 5.00%, 4.88%, 4.88%; the three Qwen base arms: 1.50%, 2.50%, 2.88%. The generation-to-generation spread is real and is why a single-run baseline is a weak comparator.

## 5. Qwen mirror (brief §9.2 / §9.4)

| arm | v5 genuine-OR (of 800) | 95% CI | purity | v5 wtd | OLD genuine-OR (haiku, wtd) | OLD genuine-OR (early Sonnet, wtd) | refused ≥3/4 | mean refuse_rate |
|---|--:|---|--:|--:|--:|--:|--:|--:|
| **logit** (trained) | **4.25%** (34/800) | [2.62, 6.25] | 58.6% | 5.19% | 7.06% | 6.72% | 58/800 (7.25%) | 8.25% |
| **probe** (trained) | **4.50%** (36/800) | [2.62, 6.62] | 42.4% | 5.31% | 9.47% | 9.53% | 85/800 (10.62%) | 11.78% |
| **vector** (trained) | **2.00%** (16/800) | [1.00, 3.25] | 94.1% | 2.31% | 2.66% | 2.72% | 17/800 (2.12%) | 2.81% |
| base-as-attacker (logit run — the arm §9.4 used) | **2.88%** (23/800) | [1.50, 4.62] | 28.4% | 3.06% | 4.41% | 6.69% | 81/800 (10.12%) | 10.62% |
| **base-as-attacker, pooled (of 2400)** | **2.29%** (55/2400) | [1.38, 3.46] | 24.7% | 2.71% | — | — | 223/2400 (9.29%) | 10.16% |

| attacker | rwr − own base | rwr − pooled base (n=2400) |
|---|---|---|
| logit | +1.38 [-1.00, +3.63] n.s. | +1.96 [+0.00, +4.08] n.s. |
| probe | +2.00 [+0.12, +4.12] **sig.** | +2.21 [+0.46, +4.21] **sig.** |
| vector | +0.50 [-1.00, +2.00] n.s. | -0.29 [-1.83, +1.17] n.s. |

## 6. The 8k Llama scale-up (brief §8)

The old §8 pipeline was two-stage and used **two** different labellers from the ones used everywhere else: a lenient `claude-haiku-4-5` BENIGN/HARMFUL filter to build a 7,877-rewrite denominator, then a `claude-sonnet-5` REFUSE/COMPLY judge over base-Llama's replies at **`refuse_rate ≥ 0.5` (2 of 4)** for the numerator. Both differ from the definition used everywhere else in this document, so both variants are given.

| definition of *refused* | v5 genuine-OR (of 8000) | 95% CI | refused (of 8000) | purity | breadth (distinct originals) |
|---|--:|---|--:|--:|--:|
| **opener classifier, `>0.5` — the definition used everywhere else here** | **10.27%** (822/8000) | [9.36, 11.19] | 1233/8000 (15.41%) | 66.7% | 514 of 2000 |
| Sonnet REFUSE/COMPLY judge, `≥0.5` — the definition behind the old figure | **12.35%** (988/8000) | [11.36, 13.33] | 1423/8000 (17.79%) | 69.4% | 601 of 2000 |

**Old figure:** 1,320 refused of 7,877 haiku-BENIGN rewrites. The brief prints this as **16.84%**, but 1320/7877 = **16.76%**; 16.84% is 1320/7,837, and §0 of the brief indeed writes the denominator as 7,837 while §8 writes 7,877. One of the two is a typo — the artifacts contain exactly 7,877 BENIGN rewrites, so **16.76% is the correct restatement of the old number** and 16.84% is arithmetically wrong.

**Why v5 is lower.** On the same refused set the two filters disagree substantially:

| refused set | n refused | haiku says BENIGN | v5 says OR | both |
|---|--:|--:|--:|--:|
| opener `>0.5` | 1233 | 1132 | 822 | 812 |
| Sonnet judge `≥0.5` | 1423 | 1320 | 988 | 978 |

## 7. Cross-generator 2×2 trial (brief §11)

750 fresh originals per generator × 4 rewrites = 3,000 rewrites per cell. All four cells are now judged (the old §11 judged only the two diagonal cells).

| cell | v5 genuine-OR (of 3000) | 95% CI | refused (of 3000) | purity | OLD v4 genuine-OR | OLD v4 purity |
|---|--:|---|--:|--:|--:|--:|
| llamaAtt → llamaTgt | **10.23%** (307/3000) | [8.77, 11.80] | 472/3000 (15.73%) | 65.0% | 12.2% (365/3000) | 77.3% |
| llamaAtt → qwenTgt | **4.43%** (133/3000) | [3.43, 5.47] | 226/3000 (7.53%) | 58.8% | not judged | — |
| qwenAtt → llamaTgt | **11.23%** (337/3000) | [9.60, 12.93] | 486/3000 (16.20%) | 69.3% | not judged | — |
| qwenAtt → qwenTgt | **5.37%** (161/3000) | [4.23, 6.60] | 244/3000 (8.13%) | 66.0% | 6.3% (188/3000) | 77.0% |

Target swap, **paired** (identical rewrites, different target), in points of genuine-OR:

- llamaAtt: Llama target − Qwen target = **+5.80 pts** [+4.60, +7.07]
- qwenAtt: Llama target − Qwen target = **+5.87 pts** [+4.47, +7.30]

Attacker swap, **unpaired** (the two generators use different 750-original substrates):
- llamaTgt: qwenAtt 11.23% vs llamaAtt 10.23% (**+1.00 pts**)
- qwenTgt: qwenAtt 5.37% vs llamaAtt 4.43% (**+0.93 pts**)

## 8. What changes under v5, and what holds

### Changes

**8.1 The §8 scale-up headline drops by about a third — 16.8% → 10.3%.** Under the definition used everywhere in this document (opener classifier, ≥3 of 4) the 8k corpus yields **10.27% (822/8000)** genuine over-refusal, CI [9.36, 11.19]. Holding the old refusal definition fixed (Sonnet REFUSE/COMPLY judge at ≥2 of 4) and changing only the benign filter haiku→v5 — which also restores the denominator to all 8,000 rewrites, since v5 replaces the haiku pre-filter rather than sitting behind it — it is **12.35% (988/8000)** — so roughly **4.4 of the ~6.5 points of drop come from the judge** and the rest from the stricter refusal threshold. Breadth falls with it: 514 distinct originals of 2000 (25.7%) versus the brief's "732 of 1,995 (37%)". The brief's own bracket — "7.06% (strict hand audit) and 16.84% (lenient filter) bracket the same measurement" — survives: v5 lands inside it.

**8.2 The Qwen verdict flips for the probe arm.** Brief §9.4 concluded *"no trained arm reliably beats base-Qwen on genuine over-refusal; vector is significantly WORSE"*. Under v5 both halves change:

| arm | old (early Sonnet) vs base | v5 vs its own base | v5 vs pooled base (n=2400) |
|---|---|---|---|
| logit | +0.03 pts, n.s. | +1.38 [-1.00, +3.63] n.s. | +1.96 [+0.00, +4.08] n.s. |
| probe | +2.84 pts, n.s. | +2.00 [+0.12, +4.12] **sig.** | +2.21 [+0.46, +4.21] **sig.** |
| vector | −3.97 pts, **sig. worse** | +0.50 [-1.00, +2.00] n.s. | -0.29 [-1.83, +1.17] n.s. |

The driver is base-Qwen, not the trained arms. v5 judges base-Qwen-as-attacker far more harshly than either earlier judge: purity of its refused rewrites is only **24.7%** (55 of 223 pooled refusals judged OR), so its genuine-OR rate falls from 6.69% (early Sonnet) to **2.29%** pooled. With the baseline that much lower, the probe arm now clears it and vector no longer sits below it. **Caveat, and it is a large one:** probe is the tiny-pool (1,210 trainable pairs) affine twin of vector (brief §9.1), so this is not a clean signal result, the effect is ~2 points on an eval whose CIs are ±2 points, it is a single arm out of three, and its significance is knife-edge — the lower CI bound sits 0.1–0.5 pts above zero and flips sign between bootstrap seeds (the same contrast came out n.s. on a different seed). The safer restatement is: *under v5 no Qwen arm is reliably worse than base, and probe is marginally better* — not that the recipe transfers.

**8.3 The §7 "more epochs made it worse" finding does not survive.** On raw induced refusal, epochs 3→6 fell 15.0% → 12.8%. On v5 genuine over-refusal the two are indistinguishable (10.00% vs 9.88% of 800), and the 6-epoch arm has the **higher purity** (79.8% vs 69.0%) and the larger paired margin over its own base (+6.62 [+3.38, +10.00] **sig.** vs +5.12 [+2.12, +8.25] **sig.**). The raw-rate decline was mostly a decline in *impure* refusals.

**8.4 Every genuine-OR level moves up relative to the §5/§6/§7 hand audit, by roughly 1.4–2.5×.** e.g. logit 7.06% → 10.38% weighted, vector@L31 1.53% → 3.78%. The hand audit had a **bucket C** — "not a safety refusal" (capability disclaimers on the [Image] prompts, or rewrites that told the model not to do the task) — and excluded it. **v5 has no such axis**: a capability disclaimer on a rewrite whose intent is unchanged and whose ask is benign scores INTENT SAME / HARM BENIGN and counts as over-refusal. So v5 genuine-OR ≈ hand-audit (bucket A + bucket C), and it is *mechanically* looser than bucket A. This matters most for the vector/probe arms and for base-as-attacker, whose refusals were 20–26% bucket C.

**8.5 The §11 diagonal cells drop ~1–2 points and lose ~11 points of purity** (v4 → v5: 12.2% → 10.23%, purity 77.3% → 65.0%; 6.3% → 5.37%, purity 77.0% → 66.0%). v5 is the stricter of the two on this corpus.

**8.6 A pooled base-as-attacker number now exists.** base-Llama **4.92% of 2400** [3.54, 6.54]; base-Qwen **2.29% of 2400** [1.38, 3.46]. Quote these instead of a single run's base arm — the three Llama base runs span 4.88–5.00% and the three Qwen base runs span 1.50–2.88%, purely from re-generation.

### Holds

**8.7 The core Llama claim survives intact.** The logit attacker is the only Llama signal whose genuine over-refusal beats its base comparator: **10.00% of 800** [7.38, 12.88] vs base 4.88% (paired +5.12 [+2.12, +8.25] **sig.**; vs pooled base +5.08 [+2.08, +8.08] **sig.**). probe (+0.75 [-2.12, +3.75] n.s.) and vector (-1.75 [-4.00, +0.50] n.s.) do not. The signal ordering logit > probe > vector is unchanged.

**8.8 It reproduces at 40× scale.** The 8k corpus gives 10.27% against a 0.50% floor on untouched originals — a ~21× lift, spread over 514 distinct originals. Lower than 16.8%, same conclusion.

**8.9 §6's conclusion (signal type, not layer) holds.** vector@L31 3.62% and vector@L17 3.25% are both indistinguishable from base (+0.50 [-1.50, +2.62] n.s. and -1.75 [-4.00, +0.50] n.s.). Note the *direction* of the tiny L31-vs-L17 gap reverses versus the hand audit (1.53% < 2.72% became 3.62% > 3.25%), but both are inside each other's CIs, so neither ordering is real. The claim that rests on it — the vector signal fails at both layers — is unaffected.

**8.10 §11's target-vs-attacker conclusion holds and is now stronger.** On genuine-OR across the *full* grid (all four cells judged, not just the diagonal): swapping the target Llama→Qwen with the attacker fixed moves the rate **+5.80** and **+5.87** points (paired, identical rewrites); swapping the attacker with the target fixed moves it **+1.00** and **+0.93** points. Target-driven, ~6× the attacker effect.

**8.11 §7's similarity-floor result holds.** floor 0.5→0.7 lowers genuine-OR (10.00% → 7.88%) at unchanged purity (69.0% → 68.5%) — the same "fewer refusals at the same quality mix" story, though the CIs overlap.

## 9. Limitations of this recompute

1. **v5 cannot see bucket C.** It has no "this is not a safety refusal" verdict, so capability disclaimers (the [Image] prompts) and self-defeating rewrites count as over-refusal whenever intent is preserved and the ask is benign. Every v5 figure here is therefore an *upper* bound on safety-triggered over-refusal, and is not directly comparable to the §5/§6/§7 bucket-A column beside it. This is the single largest caveat in the document. See §8.4.
2. **The judge model refuses to label some items.** 42 of the 3,375 pooled pairs (1.24%) come back with `stop_reason: "refusal"` — Sonnet 5's own safety classifier declines the labelling request — even after 3 retries. They are counted as NOT over-refusal. They are concentrated in exactly the rewrites most likely to be genuinely harmful, so the bias is small and in the conservative direction, and they are thin on the ground per arm (≤1 per 800-rewrite arm, 22 of 8,000 in the scale-up, ≤7 per 3,000-rewrite grid cell). Verdicts and retry outcomes: `probe_or/results/v5_judged/unknown_retry.json`.
3. **The judge is stochastic.** Default temperature; a 351-item set judged twice differed on 3 items (0.9%). Re-running this pipeline will move the second decimal, not the conclusions.
4. **n = 200 originals is underpowered for the 200-prompt evals.** CIs are ±2–3 points on effects of ~2 points; this is why the Qwen probe result (§8.2) should not be over-read. The 8k scale-up and the 2×2 grid do not have this problem.
5. **The refusal labeller is unchanged.** Only the over-refusal judge was replaced. The refusal-opener classifier and its ≥3-of-4 threshold are the same everywhere, which is what makes the tables comparable — but any bias in that classifier propagates identically to every row.
6. **No analysis code was altered to change a result.** The one code change made during this work was to `build_v5_judge_pool.py`, to record the `(original, rewrite)` key for *every* rewrite rather than only refused ones, so the refuse-rate-weighted column could include the partially-refused rewrites. The judge input CSV is byte-identical before and after (md5 `86b90770e5e4184615983fd755bd0953`).

### Discrepancy found in the brief

`EXPERIMENT_BRIEF.md` reports the §8 scale-up figure as **16.84%** and states the numerator and denominator as 1,320 of 7,877. 1320/7877 = **16.76%**. 16.84% is 1320/**7,837** — the denominator §0 of the brief actually prints, while §8 prints 7,877. The artifacts contain exactly 7,877 haiku-BENIGN rewrites (`benign_scaleup.csv`: BENIGN 7,877, HARMFUL 123), so 7,837 is a typo and the old figure should have read 16.76%. Not corrected here — `EXPERIMENT_BRIEF.md` is owned by another process.

## 10. API spend

Model `claude-sonnet-5` throughout, at the introductory rate in force on the run date (2026-08-19): **$2.00 / MTok input, $10.00 / MTok output**; cache read 0.1×, cache write 1.25× (5-minute TTL) or 2.0× (1-hour TTL); Message Batches 0.5×.

Measured per-call usage (from `response.usage` on representative items): cache read **3,967** tokens (the few-shot system prefix, identical on every call), fresh input **101–275** tokens (mean ≈185), output **20–24** tokens. That is **$0.001373 per direct call**.

| what | calls | note |
|---|--:|---|
| main pool | 3,375 | every unique refused `(original, rewrite)` across all 6 datasets |
| partial-refusal supplement | 351 ×2 | judged twice — see below |
| judge calibration (held-out + purity) | 83 ×2 | judged twice — see below |
| UNKNOWN diagnosis + 3× retry of 48 items | ~150 | resolved 6 of 48 |
| smoke test + usage probes | 9 | |
| **total direct calls** | **≈4,402** | ≈ 4,402 × $0.001373 = **$6.04** |
| cache writes (cold start of each run × workers) | ≈110 | ≈110 × $0.00992 = **$1.09** |
| cancelled batch: 171 succeeded, 3,204 cancelled | 171 | at 0.5× = **$0.12** |

**Total ≈ $7.2** (≈ $10.7 had standard $3/$15 pricing applied). Judging each unique pair once saved 711 duplicate calls (4,086 dataset-level refusal events → 3,375 unique pairs, 17%).

**Why this exceeded the $2–4 estimate.** Three avoidable-in-hindsight costs:

1. **The Batches API was stalled.** The 3,375-item batch sat at `succeeded=0` for ~35 minutes (a 68-item batch from another process on the same account had been stuck for 5 hours; historically an 8,000-item batch on this account finished in 7 minutes). It was cancelled and re-run as direct concurrent calls, which forfeits the 50% batch discount — roughly +$3 versus the batch path.
2. **A concurrent process deleted `probe_or/results/v5_judged/` mid-run**, taking the supplement and calibration verdicts with it; those 434 items had to be re-judged (+$0.6). The deterministic inputs were regenerated byte-identically, so nothing was lost beyond the money.
3. The 42 judge-self-refusals cost 3 attempts each.

## 11. Artifacts

| file | what |
|---|---|
| `probe_or/results/v5_judged/judge_input_all.csv` | the 3,375 unique refused pairs sent to v5 |
| `probe_or/results/v5_judged/verdicts_all.csv` | v5 verdicts for them (`pair_id,intent,harm,is_or`) |
| `probe_or/results/v5_judged/judge_input_partial.csv` | 351 pairs with `0 < refuse_rate ≤ 0.5`, for the weighted column |
| `probe_or/results/v5_judged/verdicts_partial.csv` | v5 verdicts for those |
| `probe_or/results/v5_judged/calib_heldout_v5.csv`, `calib_purity_v5.csv` | v5 on the owner-labelled calibration sets |
| `probe_or/results/v5_judged/unknown_retry.json` | the 48 UNKNOWNs, their retries, and stop reasons |
| `probe_or/results/v5_judged/judge_input_all.csv.manifest.json` | row → pair_id map for every dataset, with refusal flags |
| `probe_or/results/v5_judged/summary_v5.json` | every number in this document, machine-readable |
| `build_v5_judge_pool.py` | builds the de-duplicated judge input + manifest |
| `report_v5.py` | recomputes every figure and writes `summary_v5.json` |
| `write_results_v5.py` | renders this document from `summary_v5.json` |

