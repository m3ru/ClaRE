# Refusal Atlas — in-progress update

**Shareable figures:** <internal-link-removed>

## Bottom line
Both models are scored end-to-end against an **independent judge**. The behavioral map
is clean, and the headline holds: **the over-refusal boundary looks different depending
on which internal "refusal signal" you read it through** — the refusal *direction* is
partly a topic detector, while the output *logit* is not.

## What the experiment is designed to do
Map the **over-refusal boundary** of two models (Llama-3-8B-Instruct and Qwen3-32B):
which **topics** and which **words** push a *benign* prompt into a refusal — and test
whether that boundary looks **different depending on which internal "refusal signal"**
you measure it with, and whether it **differs between the two models**.

Three refusal "signals," measured against real (behavioral) refusals as ground truth:
1. **Refusal direction** — projection onto the diff-of-means refusal vector (Arditi-style).
2. **Probe ensemble** — a mass-mean linear probe combined across layers.
3. **Output logit** — the model's own probability of *starting* its answer with a refusal.

## The setup
- **One common 1,636-prompt substrate**, both models: OR-Bench-Hard (10 topic categories),
  XSTest, a toxic "should-refuse" anchor, plus **463 freshly-generated single-word minimal
  pairs** (one word changed, e.g. "reduce"→"attack") that isolate the causal effect of a
  single word. Every prompt scored by all three signals + real refusal behavior (n=8 samples).
- **Independent ground truth.** Behavior is labeled by a **Claude-Sonnet judge**, not the
  opener regex — the regex shares its definition with the logit signal, so grading the logit
  with the regex is circular. Every number below is judge-labeled.
- **Novelty:** no prior (peer-reviewed) work puts all three signals × both models × both
  topic and word level on the same benign boundary.

## What we found (the important ones)

1. **The behavioral map is clean, but the two models differ in degree *and* profile.**
   Behavior orders exactly as it should — near-ceiling on the toxic anchor (Llama 0.98,
   Qwen 0.99), near-floor on benign minimal pairs (~0.02–0.04 both). The gap is in the
   sensitive-but-benign middle: on OR-Bench-Hard, **Llama over-refuses ~2.2× as often as
   Qwen (78% vs 35%)**. And the *shape* differs — Llama sits near-ceiling across
   hate / self-harm / unethical / violence (~0.90), while **Qwen elevates mainly on
   violence (~0.50)** and is much lower elsewhere. (Figure 1.)

2. **[Headline] The refusal *direction* is partly a topic detector; the output *logit*
   is not.** Across the whole substrate all three signals separate refuse from comply well
   (AUC ≈ 0.97–0.99). But the boundary that matters for over-refusal lives *inside* a topic
   (same subject, benign vs not). Restricted to within-topic, the **refusal-direction** AUC
   drops (Qwen 0.969→**0.874**; Llama 0.977→0.946) while the **output logit** holds flat
   (Qwen 0.977→0.980; Llama 0.990→0.993). Much of the direction's apparent skill was reading
   the *topic*, not the *refusal*. Practically: steering/localizing refusal through the
   direction vector partly moves "this is a violence-adjacent topic," not "this request should
   be refused." (Figure 2.)

3. **Circularity was real — and it inflated the logit most.** Swapping the shared-definition
   regex for the independent judge shrinks every signal's AUC, and shrinks the **logit's**
   most (Qwen −2.0pp, Llama −1.0pp vs vector/probe −0.3 to −0.6pp) — exactly the signal the
   circularity predicts. The judge also shows the regex *under-counts* real refusals (misses
   ~30–50 of ~1,230 unique replies per model), so raw behavioral rates are slight
   *under*estimates. (Figure 3.)

4. **Single words move the boundary, with a shared core across both models.** On the
   leakage-free single-word pairs, one word flips an otherwise-benign task into a refusal.
   Five words — **attack, contraband, exploit, lethal, weaponize** — trigger *both* models;
   others are model-specific (Llama: "interrogation"; Qwen: "manipulation", "dissect").
   Top single-word effects: Llama "contraband" **+42pp**, "exploit" **+36pp**, "attack"
   **+33pp**; Qwen "attack"/"manipulation"/"dissect" **+33pp**. (Figure 4.)

## Method safeguards (what keeps these clean)
- **Independent judge** de-circularizes the whole signal comparison.
- **Leakage-free pairs:** the 463 single-word pairs come from prompts disjoint from every
  probe/vector fitting set — both models see them for the first time.
- **Causal layer validation:** the direction is taken at the causally-controlling layer
  (Llama L17, validated by ablation; Qwen L58, validated by addition — driving a benign
  prompt to 93% refusal), not merely the most predictive one.
- **The "probe ensemble" collapses to a single refusal direction** for both models (stacking
  layers adds ~0), consistent with the single-direction view — reported honestly, not hidden.
- **Adversarial code/method reviews caught real bugs before GPU spend** (a padding bug that
  would have corrupted the behavioral ground truth; a wrong-layer fallback) — all fixed pre-run.

## Where it stands
Both models fully scored; judge-labeled analysis done for the four findings above. **Next:**
(P5) unsupervised topic-clustering map beyond the fixed 10 categories with bootstrap-Jaccard
stability; (P6) full 463-pair per-word effect atlas with CIs; and bootstrap CIs + a noise null
on every AUC and rate reported here.
