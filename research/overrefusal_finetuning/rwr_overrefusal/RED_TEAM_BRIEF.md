# Red-team brief: does the abliterated refusal vector work as an RWR training signal?

For Alec. Everything you need to attack the design; nothing else. Background detail is in
`REVIEW_VERDICT.md`, don't read it unless a section here points you there.

## The claim we want to test

> The direction selected by **abliteration** — Arditi's criterion, "ablate it and refusal on
> harmful prompts collapses" — is the direction the field treats as *the* refusal direction.
> We test whether it is also a good **training signal**: used as the OR reward that ranks
> candidate rewrites, does it produce an attacker that induces more over-refusal?
>
> Run on **both** Llama-3-8B and Qwen3-32B.

Two outcomes, both publishable. If it wins, the causal direction is the right reward and our
current layer choice is wrong. If it loses, then **the direction that mediates refusal
causally is not the direction that predicts which edits will trigger it** — a real dissociation,
and it explains why our vector-reward attackers have underperformed.

## The pipeline (identical for every arm, only the vector changes)

1. Pick a direction `d` at a layer `L`.
2. Score every candidate pair: `signal(p) = <h_L(p), d̂>`, `Δ = signal(rewrite) − signal(original)`.
3. `OR = exp(18.4·(sim − 0.75)) · Δ`, sim = MiniLM cosine.
4. Bin by OR, weights [0, 1, 4, 16], LoRA SFT r=32, 3 epochs, lr 1.5e-5. **Identical across arms.**
5. Generate rewrites on held-out originals → two-axis judge → over-refusal rate.

## Arms

Existing checkpoints, same hyperparameters, so new arms drop straight into the comparison:

| model | arm | layer | selected by | status |
|---|---|--:|---|---|
| Llama | vector | 17 | behavioural correlation | **trained** |
| Llama | vector | 31 | AUC | **trained** |
| Llama | **vector (abliterated)** | **12** | **ablation argmin** | **← to run** |
| Llama | logit | — | n/a | trained (current best) |
| Qwen | vector | 58 | AUC | **trained** |
| Qwen | **vector (abliterated)** | **60** | **ablation argmin** | **← to run** |
| Qwen | logit | — | n/a | trained |

**One new training arm per model.** Everything else is a comparator that already exists.

## Settled — please don't spend time here

**Construction (refusal-vs-compliance vs harmful-vs-harmless) does not matter at matched
layers.** `direction_comparison.json`, Spearman vs behavioural ΔP, 6,000 Llama pairs:

| direction | @L12 | @L17 | @L31 |
|---|--:|--:|--:|
| ours (refusal-vs-compliance) | 0.139 | 0.2615 | 0.2912 |
| arditi (harmful-vs-harmless) | 0.058 | 0.2563 | 0.2824 |

Δ is 0.005 at L17 and 0.009 at L31. The often-quoted "0.261 vs 0.058" gap is a **layer** effect,
not a construction effect. **Layer is the live variable; construction is not.**

**Sample size is not a threat.** Fitting on 512/class vs 2500/class moves the Spearman by
0.0005 (`ours-raw-n512` @L17 0.2620 vs 0.2615).

## What we predict, stated before running

L12 has the **worst** behavioural Spearman of any layer for both constructions (0.139 / 0.058).
So we expect the abliterated arm to **lose**. We are running it anyway because that prediction
has never been tested by training, and "causally necessary ≠ good reward" is the finding.
**If you think this makes the experiment not worth running, say so now** — that is the single
most useful thing you could push back on.

## Decisions to attack

1. **Is "the abliterated vector" the Arditi-construction direction at the ablation-selected
   layer, or our own construction at that layer?** We assume the former. The choice matters
   because at L12 the two differ (cos 0.50) far more than at late layers (cos 0.84 @L31).
2. ~~Qwen's ablation-selected layer is not yet known.~~ **RESOLVED 2026-08-21.** Job 19396818
   finished in 17 min. Qwen selects **L60** (val harmful refusal 91.4% → 0.0%; necessity
   91.4 → 1.6% with harmless unmoved). Llama's mirror independently reproduces **L12** on our
   instrument (9.4% argmin; necessity 100 → 5.5%; sufficiency 0 → 99.2% at coef 1).
   *Still worth attacking:* Qwen's top is a flat plateau — L56 4.7%, L57 3.1%, L60 0.0% — so the
   argmin is within noise of our existing AUC layer L57. Llama's surface is sharply
   non-monotonic instead (L12 9.4, L14 82.8, L18 43.8). Is an argmin over a flat plateau a
   meaningful selection, or should Qwen's arm use L57 so the layer is held constant against the
   existing comparator?
3. **Bin edges are re-derived per arm** from that signal's own OR distribution. This is
   necessary (distributions differ by orders of magnitude: Llama L17 edges 0.099/0.58/2.04 vs
   L31 0.37/2.21/8.14) but it means arms differ in *which* pairs land in the top bin, not just
   how they rank. Is that the right control, or should the top bin be size-matched?
4. **The comparator baseline.** Is the honest bar the untrained base model, the logit arm, or
   both? Prior result: the L17 vector arm failed to beat the untrained baseline while the logit
   arm won.
5. **One seed per arm.** No variance estimate on the training itself. Acceptable?

## Confounds we are carrying, not fixing

- **Wrapper/topic confound.** Both models' refusal-vs-compliance vectors are fit on one Reddit
  prompt pool wrapped in jailbreak templates and split by whether the model refused, so wrapper
  identity correlates with class. This affects the Llama vector too — it is correctly *labelled*,
  not confound-free.
- **Qwen's vector was fit on Llama's labels.** Both models' diff-of-means directions read the
  same Llama-labelled CSVs; Qwen refuses 21% of the "benign" side. We have rebuilt a Qwen-native
  version (389/class from Qwen's own measured refusal rates, cos 0.805 with the old one at L57).
  Whether the Qwen arms use the old or new vector is **decision 6** — we propose the new one.
- **Judge purity ~67%** corpus-wide on the final over-refusal rate, same for every arm.
- **Qwen's sufficiency test is not clean.** Adding the direction to harmless prompts gives
  32.8% refusal at coef 1 (14.8% degenerate) and 98.4% at coef 2 — but on a model that is
  **84.4% degenerate**. Llama's is clean (0 → 99.2% at coef 1, 0% degenerate). So Qwen's
  direction passes necessity but not sufficiency, and the brief's framing of it as "the"
  refusal direction is weaker for Qwen than for Llama.

## Cost and status

New work is one scoring pass + one training + one eval per model. Evals are 6–29 min.
Scoring is the long pole on Qwen (~1h H100). **Queue is the real constraint** — the partition has
~1,800 pending jobs and our next job is estimated to start ~20h out.

Job 19397514 is **held**: it would have compared two directions at mismatched layers against a
target that is circular with one of the candidates (`or_logit = gate · d_logit`). It needs
cancel-and-resubmit, not editing — sbatch froze its script at submit.
