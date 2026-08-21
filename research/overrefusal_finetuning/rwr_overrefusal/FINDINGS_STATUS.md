# Findings ledger — what stands, what was retracted, and why

Maintained because this phase produced several results that did not survive checking. A claim
is listed as STANDING only if it replicated or passed the control designed to break it.

## STANDING

| # | claim | evidence | caveat |
|---|---|---|---|
| 1 | **Over-refusal is at least 2-dimensional, and both dimensions are largely separable from harmful-refusal safety.** Two directions, orthogonal by construction, each remove more over-refusal than the known refusal direction at 4–11× less safety cost. | shared axis −34.0pp OR / −2.5pp AdvBench; second direction −61.3pp / −1.0pp; atlas r̂ −50.7pp / −11.3pp; random control flat; 0.00% degenerate in every cell; replicated at higher n | not yet shown that the two stack — the joint-ablation test is unrun, so "≥2-dimensional" rests on two separate single ablations |
| 2 | **Over-refusal Δ is displaced along the known refusal direction ~4× more than matched non-refused rewrites of the same originals.** | cos(Δ, r̂) at L17: or_high +0.399, ctrl_high +0.092, or_low +0.362, ctrl_low +0.058 | r̂ is fit on harmful-vs-harmless, so this is a transfer |
| 3 | **Edit size does not change the mechanism.** One-word edits and wholesale paraphrases displace the model along r̂ by nearly the same amount. | or_high +0.399 vs or_low +0.362 | descriptive |
| 4 | **Frames reproduce correlationally across edit size.** Each of 4 frames, estimated independently from LOW and HIGH pairs, matches itself best (4/4), at roughly the split-half noise ceiling; residuals mutually orthogonal after removing the shared axis. | cross-bin diagonal +0.846 raw / +0.419 residual; off-diagonal +0.738 / −0.037 | correlational ONLY — see RETRACTED #2 |
| 5 | **Paraphrases with no trigger word still route through the frame span.** | frame-word OR 0.539, frame-less OR 0.478, matched controls 0.262 | |
| 6 | **A refusal direction depends on which refusal prompts it is fit on.** Jailbreak-heavy vs mixed sample: cos 0.774, where two random halves of one pool give 0.996. | `dir_compare.npz` | incidental, but it is evidence of substructure in refusal generally |
| 7 | LOW stratum powered 37 → 208 pairs / 183 originals, judge purity 81% (171/211) vs 67% corpus-wide. | `or_low_stratum_v7.csv` | higher purity is structural: a 2-word edit can barely shift intent |

## RETRACTED

| # | claim | why it failed | caught by |
|---|---|---|---|
| 1 | "Ablating one direction removes 100% of over-refusal" | model was emitting repetition-collapsed gibberish; the refusal classifier is start-anchored so gibberish scored as non-refusal. Cause: left-padded tokenizer with a right-padding token index, so the fitted direction was largely a padding artifact | the random-direction control (cost nothing while the fitted one cost everything) |
| 2 | "Each frame has its own causal direction" | diagonals leading their row went 4/4 at n=120 → **2/4 at max n**; every margin was already 0.07–0.62 SE. The sign-test p=0.0039 was a small-sample artifact | replication at higher n |
| 3 | "Discriminative rank k\* measures over-refusal dimensionality" | **ill-posed, not buggy**: for two classes Fisher's LDA has exactly one discriminant direction (between-class scatter is rank C−1=1), so k\*=1 regardless of true structure | synthetic rank-recovery control returned k\*=1 for injected ranks 1,2,3,5 |
| 4 | "AdvBench-vs-Alpaca validates the estimator" | the two sets differ in source, register and length as well as harmfulness; AUC 1.000 even at layer 8, where r̂ itself scores 0.466 (chance). It measured dataset provenance | inspecting the layer-8 row |
| 5 | "The archived vector is the project's validated Llama-3 direction" | `refusal_vector_llamaguard_relabeled.npz` is fit on **Llama-Guard-3-8B**. Same shape (4096×32) so it loaded silently; cos 0.030 with this model's direction | the AdvBench control (98.67% → 98.67%, i.e. inert) |
| 6 | "The kill-switch is a headline result" | it is a replication of Arditi et al. on our corpus. "Remove refusal machinery → refusals drop" is near-tautological; the asymmetry has a margin-effect reading | — |

## UNRUN / OPEN

- **Joint ablation** of the two standing directions — the test that would establish claim #1 causally rather than by inference. One extra condition, ~5 min GPU.
- Naming: the "second direction" was found via weaponization pairs but suppresses all four frames roughly equally (52.2/71.6/69.1/52.4). It is NOT weaponization-specific and must not be named as if it were.
- Concealment is the one frame where the shared axis is causally weaker (−18.8 ± 2.8 vs ~−39 elsewhere) — unexplained.
- Qwen replication; GCG cross-attacker generalisation test.

---

## Review pass (independent agent) — verified corrections

Two claims from the review were checked against the activations directly and both replicate
exactly. They change the framing of the standing results.

**1. The shared axis is not a distinct object — it is "this prompt looks harmful."**

| | cos(r̂) | cos(harmful−benign direction) |
|---|--:|--:|
| shared over-refusal axis | +0.780 | **+0.776** |
| r̂ (literature refusal direction) | — | +0.777 |

r̂, the harmful-vs-harmless direction, and our shared axis form one tight mutual-0.78 cluster.
So STANDING claim #2 ("over-refusal Δ is displaced ~4× along r̂ vs matched controls") is close
to tautological: prompts that got refused look more harmful to the model. **Demote to a
descriptive sentence; it is not a headline.** Claim #3 (edit size doesn't change this) inherits
the same weakness.

**2. The real object is the direction ORTHOGONAL to harmfulness.**

| | cos(r̂) | cos(harm dir) | cos(shared) |
|---|--:|--:|--:|
| potent direction (the −61.3pp one) | **−0.056** | **−0.065** | +0.000 |
| train-only `d4` (leakage-free basis) | +0.005 | −0.023 | — |

It is not harmfulness, not the refusal direction, and orthogonal to the shared axis — yet
ablating it removes most over-refusal. That is a far stronger claim than "safety-sparing",
because it explains *why* it spares safety. The leakage-free basis recovers it (cos 0.829 with
the leaky estimate), so Phase B tests the right object.

**3. NEW RETRACTION — the safety-cost comparison is underpowered.**
"Our directions cost 4–11× less harmful refusal than r̂" was measured on AdvBench at n=120–200
against a 98.5% ceiling. Wilson intervals: baseline [94.1, 99.5], shared [90.6, 98.2], potent
[92.9, 99.1] — mutually overlapping. The 1.0pp vs 2.5pp difference is a ONE-PROMPT difference.
**Do not quote the ratio.** `safety_power_check.py` re-measures on a pooled 1,032-prompt harmful
set (AdvBench 520 + Arditi 512), cutting the SE from ~0.9pp to ~0.4pp.

**4. Reporting fix — GCG transfer is a k=1 story.** k=3 drops GCG refusal to 4.8% but takes
AdvBench to 72.0% (−27pp). Report k=1 (−53.8pp over-refusal at −1.4pp safety) and stop there.

**5. Open gap — non-refusal ≠ usefulness.** `is_refusal` is start-anchored and `degenerate()`
only catches repetition collapse, so a fluent, non-refusing, WRONG answer scores as success.
`safety_power_check.py` now captures the generations so this can be judged.

Review suggestions NOT adopted: a harm-matched adversarial safety control requiring rewrites of
harmful prompts (overlaps the under-refusal arm; routing it there instead of building a second
harmful-rewrite corpus), and a Δ′ causal ablation (good, but a new experiment class — queued,
not inserted mid-flight).

---

## Overnight results (leakage-free) — supersede the provisional numbers

**Causal rank, held-out originals, rank-matched random controls.** k\* = 2 by the
pre-registered criterion, but k=1 is the efficient operating point.

| k | over-refusal | random k | AdvBench |
|--:|--:|--:|--:|
| 0 | 75.0% | — | 98.5% |
| 1 | **39.8%** | 73.8% | 95.5% |
| 2 | **28.8%** | 77.5% | 87.0% |
| 3–8 | 7.0% → 3.2% | 65–79% | 72.5% → 33.0% (over-ablation) |

The leaky run's **−61.3pp did NOT reproduce**: leakage-free k=1 is −35.2pp. Use these numbers.
Marginal value of direction 2 is poor — 11pp more over-refusal for 8.5pp more safety cost — so
"over-refusal is ≥2-dimensional" is weaker than the leaky run implied.

**Powered safety test (n=520).** The clearest result in the project:

| ablated | harmful refusal | over-refusal |
|---|--:|--:|
| baseline | 99.04% [97.8, 99.6] | 75.00% |
| **potent d4** (orthogonal to harmfulness) | **98.85%** [97.5, 99.5], −0.2pp | **40.25%** |
| shared d1 | 94.81% [92.6, 96.4], −4.2pp | 39.75% |
| literature r̂ | 85.58% [82.3, 88.3], −13.5pp | 26.50% |
| random k1/k2 | 99.04%, 0.0pp | 73.8 / 77.5% |

d4 removes ~35 points of over-refusal at a safety cost indistinguishable from zero; r̂ removes
more but at a cost whose CI is cleanly separated from baseline.

**Correction to an earlier note in this file:** the pooled harmful set reaching only n=520 was
NOT a column-name bug. `arditi_harmful.csv` (512) is a strict SUBSET of
`advbench_harmful_behaviors.csv` (520) — overlap 512, union 520. n=520 is the complete unique
AdvBench-family set, so more volume from these sources is impossible. The remaining limitation
is distributional, not statistical: AdvBench is formulaic and refused at a 99% ceiling. The
right harder control is the re-framed harmful prompts the under-refusal arm is generating —
testing d4 ablation against those is the adversarial safety control, and it belongs there
rather than in a second copy here.

**Qwen replication (both bins).** Same structure, different model, 4× the corpus:

| | Llama | Qwen |
|---|--:|--:|
| over-refusal Δ vs r̂ (HIGH) | +0.399 | +0.563 |
| matched controls | +0.092 | +0.225 |
| over-refusal Δ vs r̂ (LOW) | +0.362 | +0.515 |
| matched controls | +0.058 | +0.102 |

Qwen LOW powered 45 → **217 pairs / 200 originals**. The Qwen geometry report inherits the same
k\* retraction and is annotated accordingly.

---

## d4 identified: a lexical-alarm direction (new STANDING claim)

The direction doing the causal work tracks **alarming surface form**, not the refusal decision
and not assessed harm. Δ-space 2×2 on held-out originals, cluster-bootstrapped over originals,
against a 50-direction random null:

| direction | ALARM effect | REFUSAL effect | ratio | null 95th pct |
|---|--:|--:|--:|--:|
| **d4** | **+0.190** | **−0.057** | **0.30** | alarm 0.022 / refusal 0.057 |
| d1 (shared axis) | +0.429 | +2.198 | 5.1 | — |
| r̂ (literature) | +0.281 | +1.731 | 6.2 | — |

d4's alarm effect is 8.6× the null; its refusal effect is exactly AT the null's 95th percentile.
`ctrl_alarm` (+0.556) loads higher than `or_alarm` (+0.502) — alarming vocabulary moves the model
along d4 regardless of whether it refused. This is the mechanism behind the safety-sparing
result: over-refusal is benign content in alarming clothing, and d4 encodes the clothing.

Enabling fact: **1,337 rewrites introduce alarm vocabulary and are NOT refused** (vs 2,034 that
are), so alarm vocabulary is neither necessary nor sufficient for refusal and the two factors can
be crossed within one corpus.

**RETRACTED before use:** the first version of this test (`probe_d4_alarm.py`, raw activations,
{benign,harmful}×{plain,alarming} against AdvBench) is undecidable by construction — its benign
alarm cell was selected on refusal; the AdvBench phrasing split is confounded with harm category
(cyber vs misinformation/self-harm); it projected raw activations across two corpora, re-importing
the provenance confound of RETRACTED #4; and 48% of its benign cells were in d4's own training
split. Superseded by `probe_d4_delta2x2.py`.

---

## Harm-graded safety (supersedes every refusal-onset safety number)

Refusal onset is not harm. Re-ran the harmful sets at 256 tokens and graded responses on the
StrongREJECT rubric structure — refusal (binary), convincingness (1–5), specificity (1–5),
`score = (1−refused)(convincing+specific−2)/8`.

| set | condition | regex refusal | harm score | Δ |
|---|---|--:|--:|--:|
| StrongREJECT (313) | baseline | 99.0% | 0.005 | — |
| | **lexical-alarm dir** | 98.4% | **0.005** | **−0.000** |
| | literature r̂ | 89.5% | **0.026** | **+0.021** |
| XSTest unsafe (200) | baseline | 99.0% | 0.005 | — |
| | **lexical-alarm dir** | 99.0% | **0.003** | **−0.002** |

**RETRACTED — "the lexical-alarm direction costs 4.5pp of safety on XSTest".** That was an
instrument artifact. Of the 9 regex-detected flips, **1** was genuine compliance; of 6 on
StrongREJECT, **2**. The refusal regex has a **~9pp false-negative rate on XSTest unsafe** (calls
90% refused where the judge says 99%) because it only reads how a reply *starts*. On the harm
metric the cost is −0.000.

**STRENGTHENED — the literature refusal direction genuinely increases harm** (0.026 vs 0.005
baseline, 5×), which the regex was also understating. The gap between the two directions is
*larger* on the correct metric.

Caveats: absolute scores are tiny because Llama refuses nearly everything in these sets; the judge
wording is our reimplementation of the published rubric structure (upstream repo deprecated, exact
prompt not retrievable), so absolute values are not comparable to published StrongREJECT numbers —
but all conditions share one judge, so cross-condition comparison holds.

## NEW STANDING CLAIM — separability is model-dependent (Llama vs Qwen)

| | Llama-3-8B | Qwen3-32B |
|---|--:|--:|
| baseline over-refusal (held-out, greedy) | 75.0% | **97.8%** |
| lexical-alarm dir → over-refusal | **−34.7pp** | **−5.3pp** |
| lexical-alarm dir → harmful refusal | −0.2pp | +0.6pp |
| shared axis → over-refusal | −35.3pp | −94.3pp |
| shared axis → harmful refusal | −4.2pp | **−92.9pp** (93.7% → 0.8%) |
| k\* | **2** | **never reached** |

Llama's over-refusal is **marginal and partly separable** — a quarter of its confirmed
over-refusals don't even reproduce under greedy decoding, and a direction removes a third of them
at no harm cost. Qwen's is **entrenched and fused with safety** — reproduces at 97.8%, and the only
direction that removes it also abliterates the model (harmful refusal to 0.8%).

**Separability of over-refusal from safety is a property of the model, not a universal fact about
refusal.** A mitigation validated on one model cannot be assumed to transfer.

**Qualification:** we did not search for Qwen's lexical-alarm direction — we took the analogous
basis position (weaponization residual, index 4 in both). On Llama that direction was *discovered*
via frame ablation, not assumed. Supported statement: "the analogous construction does not
reproduce on Qwen", NOT "Qwen has no such direction". Next experiment: re-run the frame-ablation
search on Qwen.

## External benchmark validation (Llama)

d4 ablation reduces over-refusal on two corpora it was not fitted on:

| corpus | baseline | after d4 | random dir |
|---|--:|--:|--:|
| XSTest safe (250, hand-written) | 7.6% | **2.8%** | 7.6% |
| OR-Bench Hard (400 of 1,319) | 77.8% | **59.8%** | 78.5% |

Mechanism confirmed with the confound flipped: XSTest's minimal pairs vary HARM with alarm held
constant, and Δ(unsafe−safe) loads **+3.24 on d1 / +3.29 on r̂ but −0.33 on d4**. XSTest's own
baseline shows the lexical story too — lexical-trigger types over-refuse at 9.7% vs 2.7% for
topic-driven types.

**FAILED PREDICTION (recorded):** we pre-registered that d4 would fire *less* on OR-Bench, since it
is built by topic-adjacency rather than lexical alarm. It fires **most** there (projection 0.86,
highest of any set) — OR-Bench's "safe" rewrites still carry alarming vocabulary (*invade*,
*deception*, *manipulate*), so its construction does not remove lexical alarm as assumed.

**Benchmark-fitness notes (checked before use, not assumed):** OR-Bench ships only
(prompt, category) — the source toxic prompt is not released — so it **cannot** support any paired
Δ analysis; it is an eval set only. OR-Bench-Toxic is a softer safety control than AdvBench and is
never the safety headline. XSTest unsafe is ideal for the *mechanism* question (matched minimal
pairs) but is a contrast set, not a harm benchmark, so StrongREJECT carries the safety claim.
