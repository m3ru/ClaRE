# Plan — geometry of the high-edit over-refusal stratum

## 1. Question

Our confirmed over-refusals split into two strata by word-level edit distance:

| stratum | pairs | distinct originals | character |
|---|--:|--:|---|
| LOW (≤2 content edits) | 82 | 70 | one word carries the flip; causal attribution is direct |
| HIGH (rest) | 3,618 | 1,922 | wholesale rewording; no single word is to blame |

For LOW we already have the lexical answer (weighted log-odds → `weaponized`, `exploit`,
`counterfeit`, `exfiltrate`). For HIGH the lexical route is unavailable by construction. The
question is therefore representational:

> **When a rewrite induces over-refusal without a single identifiable trigger word, does it move
> the model along one shared direction, or along many? And does that dimensionality depend on how
> much the prompt was perturbed?**

## 2. Prior work, and what it leaves open

- **Arditi et al., NeurIPS 2024** — refusal mediated by a single direction; the basis of our
  causally-validated L17 vector (ablate → harmful refusal 99→83%, add → benign refusal 0→99%).
- **Multi-dimensional refusal subspaces (2026)** — a single ablated direction is insufficient for
  models ≥8B; Qwen3-8B needs ≥3 directions to pass 50% ASR. Refusal is a subspace.
- **Maskey, Dras & Naseem 2026, "Over-Refusal and Representation Subspaces"** — the closest work.
  Harmful-refusal directions are *task-agnostic* (one global vector); **over-refusal directions are
  *task-dependent*: they sit inside benign task-representation clusters, vary across tasks, and span
  a higher-dimensional subspace.** This is their explanation for why global ablation cannot fix
  over-refusal.
- **LessWrong, "Exploring the multi-dimensional refusal subspace"** — practical multi-direction
  extraction: sentence-transformer embeddings → HDBSCAN into 74 topic clusters → per-cluster
  difference-in-means → **MINCOS** selection (greedily minimize pairwise cosine) to pick a spanning
  set; reports DIM-vs-probe cosines as low as 0.3, ablation of k directions vs ASR, and MMLU to
  confirm capabilities survive. States its own key limitation: clustering "doesn't guarantee full
  coverage of the refusal cone", and with ~120 original examples probes diverged across seeds.

**The shared weakness.** Both the LessWrong post and Maskey et al. derive directions from
*unpaired* sets — harmful vs harmless prompts, or per-topic clusters. Topic and refusal are
therefore entangled: a per-cluster difference-in-means partly encodes what the cluster is *about*.
Maskey et al. essentially prove this is happening for over-refusal ("directions reside within task
clusters"), and it is why naive clustering here mostly rediscovers topic.

**What we have that they do not: pairing.** Every confirmed over-refusal is a minimal pair against
an original we *verified the model complied with* (5,999 originals scored; floor Llama 0.27%, Qwen
0.15%). So we can analyse

$$\Delta = h(\text{rewrite}) - h(\text{original})$$

which cancels the task/topic component to first order and isolates *what the rewrite did*. That
turns their central confound into a controlled variable, and it is the methodological core of this
plan.

**Our novel axis: edit distance.** Nobody has asked whether the dimensionality of the over-refusal
shift depends on perturbation size. Our two strata make that a controlled contrast.

## 3. Data

| set | n | role |
|---|--:|---|
| HIGH-OR | 3,618 pairs / 1,922 originals | the stratum under study |
| LOW-OR | 82 pairs / 70 originals | one-word contrast |
| HIGH-control | ~3,600, stratum-matched | same attacker, same edit-distance band, **not** refused |
| LOW-control | ~1,800 available, matched | same, for the LOW band |
| harmful-refusal reference | 512 AdvBench (already staged) | the task-agnostic direction, per Maskey et al. |

Controls are matched on edit-distance band and attacker so the contrast is refusal, not verbosity
or perturbation size. All strata already saved in `probe_or/results/edit_strata/` with `edit_ops`,
`introduced_words`, `removed_words`, all four distance metrics and `orig_idx` for leakage-safe
grouping by original.

## 4. Procedure

### Phase 1 — activation extraction (only GPU step)
Last-token residual-stream activations under the project's standard measurement context
(`system="You are a helpful assistant."`, `add_generation_prompt=True`, right-pad, last **real**
token, `add_special_tokens=False`) for every original and every rewrite in all five sets. Layer
sweep {8, 12, 17, 20, 24, 28, 31} to avoid privileging L17 by fiat, since Maskey et al. report task
clusters emerging mid-layer. Forward passes only, no generation. Store as fp16 `.npy` keyed by
`pair_id`. **Est. ~16k prompts × 2 models is unnecessary — Llama only for now (~16k forwards, ≈30
min on the MIG slice).**

### Phase 2 — geometry (descriptive)
Per stratum, on Δ:
- **Effective dimensionality** via participation ratio PR = (Σλ)² / Σλ² over the PCA spectrum —
  the standard estimator, and a continuous alternative to "how many directions must I ablate".
- **Alignment with the known refusal direction**: distribution of cos(Δ, r̂_L17), and the fraction
  of ‖Δ‖ explained by the r̂ component.
- **cos(PC1, r̂)** and the variance explained by PC1..PC5.
- Compare HIGH vs LOW vs controls, and against the AdvBench harmful-refusal Δ as the
  "task-agnostic, low-dimensional" reference point.

### Phase 3 — multi-direction extraction, three ways (comparability)
On HIGH-OR Δ:
1. **PCA** top-k.
2. **Cluster-DIM**: HDBSCAN on Δ → per-cluster difference-in-means (the LessWrong recipe, but on
   Δ rather than raw activations, so topic is already removed).
3. **MINCOS** greedy selection over cluster directions, as in the post.
Report the pairwise cosine matrix across methods. Their reported DIM-vs-probe cosine ≈0.3 is the
benchmark; if our Δ-based directions agree more closely across methods than that, it is evidence
the disagreement in prior work was topic contamination rather than genuine multi-dimensionality.

### Phase 4 — causal validation (the part that makes it a finding, not a plot)
Ablate the top-k Δ directions from the residual stream and re-measure, for k = 1..5:
- **Over-refusal rate** on a held-out slice of HIGH-OR rewrites — does removing the subspace
  actually stop the over-refusal? This is the ASR-analogue.
- **Harmful-prompt refusal** on AdvBench — the safety control. Fixing over-refusal by destroying
  refusal is not a fix. (LessWrong used MMLU for capability retention; refusal retention is the
  sharper control for us, and is the concern SafeConstellations targets.)
- Report the k at which over-refusal drops below 50% of baseline, directly comparable to the
  "Qwen3-8B needs ≥3 directions" result.

### Phase 5 — interpretation
Label Δ-clusters by their over-represented `introduced_words` (weighted log-odds, ≥3 distinct
originals, bootstrap over originals — the same rigour as the LOW-stratum trigger table). Ask
whether clusters correspond to *strategies* ("covert/undetected framing", "weaponize an artifact",
"exploit a system") rather than topics. Because Δ removes topic, a topic-shaped clustering here
would be an informative negative.

## 5. Predictions (pre-registered)

| outcome | interpretation |
|---|---|
| PR(LOW) ≈ 1–2, cos(PC1, r̂) high | single-word over-refusals are the clean one-directional case |
| PR(HIGH) ≫ PR(LOW) | dimensionality tracks perturbation size; "no single word to blame" has a representational meaning |
| PR(HIGH) ≈ PR(LOW), both > harmful-refusal PR | Maskey et al.'s higher-dimensional over-refusal subspace is intrinsic, not an artifact of prompt construction |
| ablation k*≥3 needed | replicates the multi-dimensional-subspace result on an independent, paired corpus |
| harmful refusal survives ablation | the OR subspace is partly separable from the safety-critical one — the SafeConstellations claim, tested causally |

Every row is publishable; there is no outcome that wastes the run.

## 6. Confounds and controls

- **Length.** Δ norm correlates with rewrite length; report PR on length-normalised Δ and include
  ‖Δ‖ as a covariate.
- **Originals repeat.** Up to 4 rewrites share an original; all statistics cluster-bootstrap over
  `orig_idx`, never over pairs.
- **Judge noise.** Corpus purity is 67% [49, 81], so ~⅓ of "confirmed OR" is mislabelled. Re-run
  the headline geometry on the subset both v5 and the owner-calibrated rubric agree on, and report
  sensitivity.
- **Our direction is a transfer.** r̂_L17 was fit on refused-vs-complied jailbreak prompts, so
  projecting over-refusal Δ onto it is a transfer test — state it, don't hide it.
- **Single model.** Llama-3-8B first; Qwen3-32B is a replication if the result holds.

## 7. Deliverables

`extract_delta_acts.py` (Phase 1), `analyze_delta_geometry.py` (Phases 2–3, 5), `ablate_or_subspace.py`
(Phase 4), and `HIGH_EDIT_GEOMETRY.md` with the PR table, cosine matrices, ablation curves, and
labelled clusters.

## 8. Cost

One ~30 min GPU job for extraction; one ~1 h GPU job for ablation re-measurement. Everything else
is CPU. No API spend.

---

# Amendments (post-review)

A Fable research agent reviewed the plan above against the cited papers and the LessWrong
post. Five of its objections survived checking and change the design; they are folded in
below. Where its objection did not survive, that is recorded too — the plan above is left
intact so the revision is auditable.

## A1. Target split — the HIGH pool was not a single population (**verified, plan was wrong**)

The plan's data table pooled all 3,618 HIGH pairs. But over-refusal was *confirmed on a
specific target*, and the two attackers were confirmed on different ones:

| stratum | llamaAtt (confirmed on Llama) | qwenAtt (confirmed on Qwen) |
|---|--:|--:|
| HIGH | 2,372 pairs / 1,481 originals | 1,246 pairs / 849 originals |
| LOW  | 37 pairs / 31 originals | 45 pairs / 40 originals |

Feeding qwenAtt pairs through Llama and calling the result "over-refusal geometry" is
invalid: Llama refuses those prompts only ~17.7% of the time, so ~82% of them are, for
Llama, ordinary compliant prompts. **All Llama-side analysis uses llamaAtt pairs only.**
The qwenAtt pairs become the *replication* set, run on Qwen, not extra n. This is already
enforced in `killswitch_ablate_r17.py` (`attacker == "llamaAtt"`).

Consequence: LOW drops from 82 to **37 pairs / 31 originals** — too few for a stable
covariance estimate in 4096 dimensions. Hence A2.

## A2. Powering LOW by generation, not by pooling (**running now**)

The honest fix for n=37 is more data, not a looser cut. Edit distance is computable with no
model, so the pipeline generates broadly and spends GPU only on survivors:

```
32,000 fresh Alpaca originals (disjoint: shuffle_seed 43, offset 14,000; 35,020 unused remain)
  -> llama-logit attacker, 4 rewrites each          [job 19275835, 16 shards]
  -> filter_low_edit.py, CPU, <=2 content edits     ~5,400 candidates (4.2% hit rate, measured)
  -> score_target_refusal.py on Llama               ~5,400 generations, not 128,000
  -> or_judge_v5 on the refused subset              ~300 items, ~$1
  -> expected ~200 confirmed LOW pairs
```

Rates are measured on the existing 24,000-rewrite corpus, not assumed: 4.17% of rewrites
are ≤2 content edits; 5.4% of those are refused (vs 15.2% corpus-wide — minimal edits trip
refusal less often, as expected); ~68% of refusals survive the judge. The new originals are
disjoint from training, from the three-signal evals, and from corpus2, so the LOW stratum
stays a held-out set.

## A3. Participation ratio is the wrong headline statistic (**objection upheld**)

PR over the Δ spectrum measures how spread Δ's *total* variance is. But Δ's variance is
dominated by nuisance — rewrite length, the attacker's house style, residual topic — none of
which is refusal. A large PR would then read as "over-refusal is high-dimensional" when it
actually says "the attacker writes varied prompts." The prediction table above is exposed to
exactly this failure.

Replace the headline with a **discriminative rank**, which asks the question we actually
care about:

1. Fit a linear discriminant on Δ to separate **OR pairs from edit-band-matched control
   pairs** (same attacker, same edit-distance band, *not* refused). Nuisance variance shared
   by both classes contributes nothing to this axis.
2. Deflate: project Δ off direction 1, refit, repeat → an orthogonal sequence d₁…d_k.
3. **k\*** = smallest k reaching 95% of the AUC achievable with the full space.
4. Fit on train originals, measure AUC on **held-out originals** — otherwise k\* just counts
   directions the fit could overfit, and grows with n.

PR stays in the report as a descriptive secondary, explicitly labelled as unsupervised and
nuisance-sensitive. k\* is the number that gets compared to "Qwen3-8B needs ≥3 directions",
because k\* and the ablation-k are the same quantity measured two ways — one correlational,
one causal — and their agreement is itself a check.

## A4. Δ′ — a second contrast that also cancels "being a rewrite" (**adopted**)

Δ = h(rewrite) − h(original) cancels topic. It does **not** cancel the generic direction of
*having been rewritten by this attacker* (longer, more imperative, more formal), which is
common to OR and control alike and is pure nuisance for us.

Where an original has both an OR rewrite and a control rewrite in the same edit band, use

$$\Delta' = h(\text{rewrite}_{\text{OR}}) - h(\text{rewrite}_{\text{control}})$$

a within-original, within-band contrast that differences out topic *and* the attacker's
style, leaving only what distinguishes a refused rewrite from an unrefused one. Δ′ has the
smaller usable n (it needs both rewrite types for one original), so both are reported: Δ for
power, Δ′ for cleanliness. **If the two disagree, Δ′ wins and the gap is the nuisance
estimate.**

## A5. Positive control and a stated null (**adopted — the plan had neither**)

The prediction table claimed "every row is publishable." That was the weak point: a table
with no losing outcome is a table that will rationalize whatever it sees.

**Positive control.** Run the identical pipeline on AdvBench-harmful vs matched-benign Δ,
where the literature answer is known (Arditi et al.: essentially one direction, and our own
L17 vector reproduces it causally). If the estimator reports high dimensionality *there*,
the estimator is inflated by noise and every downstream k is uninterpretable. This runs
before the OR analysis, not after.

**Stated null.** Added as a row the original table lacked:

| outcome | interpretation |
|---|---|
| k\*(OR) ≈ k\*(control), or held-out AUC ≈ 0.5 | Δ geometry does not separate over-refusal from ordinary rewriting. The Δ framing is uninformative for this corpus and the result is negative — report it as such rather than reaching for an unsupervised statistic that still shows structure. |

## A6. Phase 4 was under-specified (**objection upheld**)

"Ablate top-k and re-measure" left three things unstated that decide whether the number
means anything:

- **Split by original.** Directions are fit on train-half originals; ablation is evaluated on
  held-out originals. Fitting and evaluating on the same pairs makes k\* a measure of fit
  capacity.
- **Rank-matched random control.** For each k, ablate k random orthonormal directions at
  matched norm. "OR fell at k=3" is only a finding if the k=3 random control did not.
- **Both failure modes reported.** k\* is the smallest k where held-out OR refusal drops
  below 50% of baseline **and** AdvBench refusal stays above 80% of baseline. A k that
  clears the first and fails the second is a broken model, not a fix, and is reported as
  such rather than as k\*.

## A7. Phase 0 — the kill-switch, which can end the program early (**added, running**)

Before any multi-direction work, ablate the single known r̂@L17 (all layers, following
Arditi) and measure held-out OR rewrites, their originals, and AdvBench, against a
norm-matched random direction. If k=1 already collapses over-refusal, the multi-direction
program is unnecessary and that is the result. If it does not, we have a measured k=1 floor
that every later k must beat. `killswitch_ablate_r17.py`, job 19275840.

## Not adopted

- **Reviewer suggested HDBSCAN on raw activations as in the post.** Declined: the post's own
  stated limitation is that clustering does not guarantee coverage of the refusal cone, and
  on raw activations the clusters are topic. Clustering stays where the plan put it — on Δ,
  in Phase 3, as one of three methods to be compared, not as the primary extractor.
- **Reviewer suggested adding MMLU for capability retention.** Declined for now: AdvBench
  refusal retention is the sharper control for this claim, and MMLU costs a GPU job to
  answer a question no reviewer of this result is likely to press. Revisit if Phase 4 shows
  a large ablation effect.
