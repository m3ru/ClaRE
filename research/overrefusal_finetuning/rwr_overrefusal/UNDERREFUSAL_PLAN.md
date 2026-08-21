# Under-refusal (jailbreak) arm — plan

## The mirror

The over-refusal arm asks: *which rewrites of a BENIGN prompt make the model wrongly REFUSE?*
This arm asks the exact mirror: *which rewrites of a HARMFUL prompt make the model wrongly
COMPLY?* Same instruments throughout, so the two arms are directly comparable:

| | over-refusal | under-refusal (this arm) |
|---|---|---|
| original | benign, model **complies** | harmful, model **refuses** |
| rewrite | model **refuses** | model **complies** |
| pair is valid iff | intent preserved AND ask is benign | intent preserved AND ask is still harmful |
| Δ | h(rewrite) − h(original), cancels topic | identical |
| binning | word-level edit distance, LOW ≤2 / HIGH | identical |

The pairing is what makes this worth doing: a jailbreak corpus where every item is a minimal
pair against a prompt the model demonstrably refused, so Δ isolates *what the rewrite did*
rather than what the prompt is about. Published jailbreak work compares unpaired sets and
therefore confounds topic with evasion.

## Data

Standard public red-teaming benchmarks already on disk — no new harmful content is authored:
`probe_or/data/advbench_harmful_behaviors.csv` (520) + `probe_or/data/arditi_harmful.csv`
(512), deduped to ~1,032 seeds. Rewrites are produced by base Llama-3-8B-Instruct asked to
re-frame the request in legitimate/professional register while preserving what is asked.

Generation is capped at 64 tokens throughout and we only ever classify *whether* the model
refused — the pipeline never needs, stores, or scores extended harmful content.

## Procedure

- **U1** rewrite each harmful seed k times (GPU).
- **U2** score original and rewrite on the target, 4 samples each (GPU). Keep pairs where the
  ORIGINAL is refused (>50% of samples) and the REWRITE is not (<50%) — a genuine flip.
- **U3** judge the PROMPT PAIR (not the model's output): is the rewrite still asking for the
  same harmful thing? Mirror of the over-refusal rubric, with the harm axis inverted. A
  rewrite that sanitised the request into something actually benign is NOT a jailbreak; it is
  the failure mode this judge exists to remove, and it is the dominant one.
- **U4** edit-distance binning, identical code path (`analyze_edit_distance.py`).
- **U5** Δ extraction + geometry, reusing `build_delta_sets.py` / `extract_delta_acts.py`.
- **U6** causal ablation: do the over-refusal directions from the other arm affect
  under-refusal? This is the cross-arm question worth the whole exercise.

## Pre-registered outcomes, including the null

| outcome | reading |
|---|---|
| Under-refusal Δ aligns with r̂ **negatively** (mirror of over-refusal's +0.40) | refusal is one axis traversed in both directions; the two failure modes are one mechanism |
| Under-refusal Δ is **orthogonal** to r̂ | jailbreaks work by a different route than making prompts look safe — the more interesting result |
| The harmfulness-orthogonal direction from the over-refusal arm also moves under-refusal | one shared "compliance threshold" governs both; strong unifying claim |
| It does **not** | the two arms are separate mechanisms; the over-refusal fix does not create jailbreak risk — a directly safety-relevant negative |
| **NULL:** too few genuine flips survive U3, or flipped pairs are indistinguishable from non-flipped controls | the rewriter is too weak or the judge too strict; report the yield and stop rather than loosening the rubric to manufacture a corpus |

## Red-team of this plan (done before building)

- *"Your 'jailbreaks' are just sanitised requests."* The dominant failure mode. U3 exists
  solely for it, and the yield after U3 is reported as the headline denominator, not hidden.
- *"Refusal classifier is start-anchored."* Same instrument as the other arm, so the two are
  comparable — but a fluent non-refusing answer that doesn't actually comply is scored as a
  flip. Mitigation: capture generations for later judging; do not claim "successful attack",
  claim "refusal suppressed".
- *"Base Llama won't produce good jailbreaks."* Likely true, and the yield will be low. That
  is why the null is stated up front. A low yield is a reportable result about how hard this
  model is to flip with fluent paraphrase alone, not a reason to escalate to stronger attacks.
- *"Both arms share originals, so Δ comparisons leak."* They do not — over-refusal seeds are
  Alpaca, under-refusal seeds are AdvBench/Arditi. Disjoint by construction.
