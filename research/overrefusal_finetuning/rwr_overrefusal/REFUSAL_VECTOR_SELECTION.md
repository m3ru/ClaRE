# How we choose the refusal vector (Llama & Qwen) — rationale for Alec

Short version: **the four criteria you list measure different constructs and genuinely
disagree, so "which is best" has no answer without naming the use case.** Below: what we
currently use, what each criterion actually measures (and its failure mode), and a
recommended weighting — plus concrete methodology issues to double-check.

## What we currently use

| model | direction construction | layer | selected by |
|---|---|---|---|
| **Llama-3-8B** | diff-of-means, **refused-vs-complied** on jailbreak attempts | **L17** | causal abliteration (ablation + addition) |
| **Qwen3-32B** | diff-of-means, refused-vs-complied | **L58** | causal **addition** (ablation was data-limited) |

Both are raw mass-mean (`μ_refuse − μ_benign`), last-token read, not LDA (LDA measured worse
— see §probe note). The retired Llama vector was L32, chosen by *largest L2 norm* with no
causal or predictive test — that criterion is not on your list and should never be used.

## The four criteria, what each measures, and its failure mode

Naming them by what they're actually a proxy for:

**1. Best via abliteration test** — *causal necessity + sufficiency.* Ablate the direction →
does refusal on harmful prompts collapse; add it to benign prompts → does refusal appear.
This is the Arditi gold standard for "is this **the** refusal direction mechanistically."
*Failure mode:* the layer where ablation most efficiently kills refusal is **not** the layer
whose projection best **ranks graded over-refusal** — those are different questions and they
diverged for us (causal-best ≠ predictive-best). Also noisy at small n and only probes the
extremes (fully harmful / fully benign), telling you nothing about the benign-but-borderline
middle where over-refusal lives.

**2. Best at separating OUR data (refused vs complied on jailbreaks)** — *supervised AUC on
the behavioral outcome.* Tempting because it's "real" refusal behavior. *Failure mode:* it is
**partly circular and topic-confounded.** On jailbreaks, whether the model refused correlates
heavily with prompt content, so a direction that scores high here may be a *jailbreak-topic
detector*, not a refusal-decision direction. And it's the objective our "ours" direction is
literally fit on, so it's a training target, not an independent test.

**3. Best at separating harmful vs helpful prompts (Arditi/AdvBench-vs-Alpaca)** — *how well
the direction detects harmful **intent**.* Clean, well-defined classes. *Failure mode:* it
measures a construct **upstream** of refusal — "is this prompt harmful," not "did the model
decide to refuse." A direction that maxes this can be a harmfulness detector. This is exactly
the mechanism behind the atlas finding *"the refusal direction is partly a topic detector."*

**4. Correlation with the logit score** — *agreement with an output-space, behavior-proximal
refusal signal.* The logit = P(reply begins with a refusal opener) is the closest cheap proxy
to the actual refusal decision, and it's the signal that **actually predicted behavior** in
our experiments. A direction correlating with it is reading the refusal decision, not just
topic. *Failure mode:* the logit is itself start-anchored (misses non-opener refusals), so
it's a proxy, not truth — but it's the least topic-confounded of the four.

## They disagree — here is the evidence (all measured, not asserted)

**Llama causal (our direction), `causal_refusal_results.json`:** baseline harmful refusal 99%.
- **L17:** ablate → **83%** (necessity, partial), add@coef2 → **99%** from 0% (sufficiency, clean).
- **L32:** ablate → **98%** (does nothing — causally inert; this is why the old vector failed).

**Llama Arditi replication (harmful-vs-harmless), `arditi_direction_out/`:** selected **L12**,
ablate 100→**4%** (a *stronger* necessity result than L17). BUT the candidate grid was
**even layers only [8,10,…,28] — L17 was never tested**, n_dir=256, n_val=40, and the
selection surface is non-monotonic (L12=12.5, L14=80, L16=80, L18=45), so L12's win carries
real selection noise.

**Predictive ranking of graded over-refusal** (`direction_comparison.json`; target = behavioral
dP on 6,000 pairs, Spearman):

| direction | @L12 | @L17 | @L31 (best) |
|---|--:|--:|--:|
| ours (refused-vs-complied) | 0.139 | 0.261 | **0.291** |
| arditi (harmful-vs-harmless) | **0.058** | 0.256 | 0.282 |

- **L12 is the worst layer for either direction** (criterion 1's winner is criterion 2/4's
  loser). **L31 is the best predictor** but is causally untested and adjacent to the inert L32.
- The two *constructions* nearly coincide at late layers (cosine ours-vs-arditi **0.84 @L31**,
  0.78 @L17) but diverge in the middle (**0.50 @L12**) — i.e. the harmful-vs-harmless vs
  refused-vs-complied choice matters **most exactly where the causal signal is**.
- `ours@L17` beats `arditi@L12` by **+0.203** [CI +0.169, +0.236]; at their best layers the two
  constructions are a near-tie (+0.009 [+0.001, +0.017]).

**Qwen causal, `qwen_causal_results_clean.json`:** selection leans almost entirely on **addition**
— L58 add@coef2 drives disjoint benign prompts **5%→93%**, far above L32/L45/L63 (≤40%).
Ablation was **data-limited** (harmful baseline only 2%, so necessity isn't testable), so Qwen
currently rests on **one** of the four criteria. This is the biggest gap to double-check.

**The decisive external check — behavioral training (this retrain):** we trained an attacker on
the L17 vector-projection reward and it **failed to beat an untrained baseline** on induced
refusal, while the **logit-reward attacker won (2.2× on genuine over-refusal)**. So for a
*scoring/reward* use, the direction is the wrong tool at **any** layer — which means the layer
argument matters far less for our application than it does for interpretability claims.

## Recommended weighting

It depends on what the vector is **for**:

- **For causal intervention / steering / abliteration / the "this is the refusal direction"
  interpretability claim** → weight **(1) abliteration heavily**, done properly, with **(4)
  correlation-with-logit as the tiebreaker** among causal candidates. This gives Llama L17,
  Qwen L58. (1) tells you it's mechanistically refusal; (4) keeps you from picking a
  topic/harmfulness detector.

- **For a reward/score that ranks over-refusal (our RWR + atlas use)** → the direction lost to
  the logit behaviorally, so **don't select a direction for this at all if you can use the
  logit.** If you must, weight **(4) and direct behavioral validation**, which points at L31 —
  but L31 Goodharts (best predictor, no causal grounding, next to inert L32), so this is a
  cautionary, not a recommendation.

- **(2) and (3) are sanity floors, not selectors.** A genuine refusal direction *should*
  separate both — but maxing either rewards the wrong construct: (2) rewards jailbreak-topic
  detection, (3) rewards harmfulness detection. Use them to *reject* a candidate that fails
  them, never to *pick* the winner among candidates that pass.

Net: **(1) causal is primary, (4) logit-correlation is the tiebreaker, (2)/(3) are floors.**
The single most useful thing to internalize: **causal-best ≠ predictive-best ≠
topic-separation-best, and they diverge by ~15 layers.** Any single-number ranking hides that.

## Methodology issues to double-check in the current selection

1. **Llama Arditi sweep skipped odd layers** — L17 (our causal layer) was never a candidate, so
   "L12 is best" was never actually tested against it. Re-run over **all** layers.
2. **Selection-on-the-eval-set** — the abliteration "best layer" is picked on the same val set
   it's scored on (n_val=40). Use a **held-out** layer-selection set, larger n.
3. **Necessity vs sufficiency can pick different layers** — report **both** (ablation *and*
   addition) and, like Arditi, require both, plus a **KL/coherence check** on harmless prompts
   so the "direction" isn't just breaking the model.
4. **Qwen has only sufficiency** — stage a proper harmful set (AdvBench) so **ablation necessity**
   is testable; right now Qwen's L58 rests on one criterion.
5. **Construction choice (harmful-vs-harmless vs refused-vs-complied)** matters most at mid
   layers (cosine 0.50 @L12) and barely at late layers (0.84 @L31) — so if you settle on a late
   causal layer the construction is nearly moot; if a mid layer, it's decisive.
6. **raw vs LDA:** we tested covariance-corrected (LDA) directions; **raw mass-mean won** on both
   models (Llama raw 0.291 vs best-LDA 0.261; Qwen raw 0.809 vs LDA 0.646) because Σ is estimated
   from ~2.5k samples in 4096–5120 dims. Don't switch to LDA without re-measuring at your n.
