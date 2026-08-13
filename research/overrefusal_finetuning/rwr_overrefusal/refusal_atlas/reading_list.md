# Refusal Atlas — annotated reading list (over-refusal & refusal representations)

Compiled 2026-08-02. Prioritized 2024–2026 arXiv/ACL/NeurIPS/ICLR/EMNLP. Very recent
2026 preprints (arXiv 26xx) are unreviewed and sometimes single-author — flagged inline.

**Peer-review status (per meru's steer — anchor on reviewed work):** the operative,
review-sorted anchor list lives in `PLAN.md` → "Related work". Down-weight the 2026
preprints (2602.02132, 2512.01037, 2511.19009, 2604.18901) — verify their methodology before
trusting. Curated field resource: **Awesome-Over-Refusal**
(https://github.com/abbottyanginchina/Awesome-Over-Refusal), organized by benchmarks +
mitigation (no internal-signal section). Two verified peer-reviewed additions from it:
- **Pan et al., EMNLP 2025** "…from an Unveiling Perspective of Safety Decision Boundary"
  (2025.emnlp-main.1065 / arXiv 2505.18325) — over-refusal decision boundary via steering
  vectors (RASS). Our peer-reviewed boundary anchor (replaces preprint 2602.02132).
- **"Surgical, Cheap, and Flexible: Mitigating False Refusal via Single Vector Ablation",
  ICLR 2025** — peer-reviewed causal single-vector ablation on false refusals.
- Concept Cones venue verified: **ICML 2025**, PMLR v267 pp 66945–66970.

## Bucket 1 — Over-refusal benchmarks & their TOPIC taxonomies

- **XSTest** — Röttger et al., NAACL 2024. https://aclanthology.org/2024.naacl-long.301/ ·
  code https://github.com/paul-rottger/xstest
  250 hand-written benign prompts (+200 unsafe contrasts), 10 prompt types; 4 harm
  dimensions (violence, illegal, discrimination/hate, privacy). Canonical "scary word in
  a benign prompt" design. **SATURATED for SOTA models** — use as a validated anchor, not
  a discriminating benchmark. Small/hand-crafted.
- **OR-Bench** — Cui et al., ICML 2025. https://arxiv.org/html/2405.20947v2 ·
  https://openreview.net/forum?id=obYVdcMMIT
  80K "seemingly toxic but benign" prompts, **10 categories** (Deception, Harassment,
  Harmful, Hate, Illegal, Privacy, Self-harm, Sexual, Unethical, Violence) + OR-Bench-Hard-1K
  + 600 toxic validation. **Almost certainly the taxonomy behind our benign pool.** Prompts
  LLM-generated (Mixtral) + LLM-moderated (GPT-4-turbo/Llama-3-70B/Gemini) — labels noisy,
  "benign" = ensemble-judged, not human ground truth.
- **SORRY-Bench** — Xie et al., ICLR 2025. https://arxiv.org/html/2406.14598v1 ·
  code https://github.com/SORRY-Bench/sorry-bench
  Fine-grained **45-category** taxonomy (4 domains), 450 unsafe instructions + 9,000
  linguistic-mutation variants; fine-tuned 7B judge. Most granular refusal topic taxonomy
  available. Measures refusal of genuinely UNSAFE prompts — repurpose the taxonomy, not the
  safe/unsafe framing.
- **PHTest** — An et al., 2024. https://arxiv.org/html/2409.00598v2 · https://phtest-frf.github.io/
  3,260 auto-generated pseudo-harmful prompts, human-labeled harmless (2,069) vs
  controversial (1,191). Introduces the "controversial" middle — the boundary is fuzzy,
  not binary. Model-dependent generation.
- **FalseReject** — Zhang et al., COLM 2025. https://arxiv.org/html/2505.08054
  Graph-informed pseudo-harmful corpus + structured-reasoning mitigation data. Extra
  topic-organized data source / cross-check vs OR-Bench. LLM-generated.
- Secondary: **ORFuzz** (fuzzing to discover OR triggers, arXiv 2508.11222); **EvoRefuse**
  (evolutionary OR prompt optimization, 2505.23473); **Beyond Over-Refusal** (scenario
  diagnostics, 2510.08158).

## Bucket 2 — Refusal as a linear direction / mechanistic interpretability

- **Refusal is Mediated by a Single Direction** — Arditi et al., NeurIPS 2024.
  https://proceedings.neurips.cc/paper_files/paper/2024/file/f545448535dfde4f9786555403ab7c49-Paper-Conference.pdf
  · code https://github.com/andyrdt/refusal_direction
  Single residual-stream diff-of-means direction: ablation disables refusal, addition
  induces it, across many open models. **Foundation of our Signal (1).**
- **The Geometry of Refusal: Concept Cones** — Wollschläger et al., arXiv 2502.17420 (ICML 2025).
  https://arxiv.org/pdf/2502.17420
  Refusal is NOT one direction but multi-dimensional "concept cones" of independent
  directions (Gemma, Qwen2, Llama). Primary critique of single-direction; motivates our
  probe ensemble and the "which signal you use changes the picture" thesis.
- **There Is More to Refusal than a Single Direction** — Joad et al. (QCRI), arXiv 2602.02132.
  https://arxiv.org/html/2602.02132v1
  Across 11 refusal categories: multiple geometrically distinct directions (pairwise cos
  0.4–0.6) with nearly IDENTICAL behavioral trade-offs — shared control knob, stylistic
  variation. Llama-3.1-8B + Gemma-2-9B; GemmaScope SAEs; XSTest/SORRY-Bench/CoCoNot.
  **KEY: internal geometry differs by topic while behavior collapses** — closest published
  work to our topic × signal cross-cut.
- **Understanding Refusal with SAEs** — Yeo et al., EMNLP Findings 2025 (2505.23556).
  https://aclanthology.org/2025.findings-emnlp.338/
  SAE latents that causally mediate refusal; upstream→downstream structure. Feature-level
  lens complementing our direction/probe signals.
- **Beyond "I'm Sorry, I Can't": Dissecting LLM Refusal** — Prakash et al., arXiv 2509.09708.
  https://arxiv.org/html/2509.09708v1
  SAE + ablation locate sparse causal refusal features with redundant "hydra" structure &
  non-linear interactions. On Llama-3.1-8B-IT compares Factorization Machines vs linear
  probes (FM 330 jailbreaks vs 101 linear) — **direct evidence our linear probe misses
  non-linear structure.**
- Secondary: **CRaFT** (2604.01604); **GSAE** (2512.06655); **Refusal Lives Downstream of
  Persona** (2606.26161); "Finding Features Causally Upstream of Refusal" (LessWrong).

## Bucket 3 — Probing safety behavior; internal-signal vs behavior agreement

- **Harmful Intent as a Geometrically Recoverable Feature** — Llorente-Saguer, arXiv 2604.18901.
  https://arxiv.org/html/2604.18901v1
  Linear harmful-intent probes across 12 models (Qwen2.5/Qwen3, Llama-3.2, Gemma-3;
  base/instruct/abliterated), AUROC ~0.98. **Probe stays intact in ABLITERATED models that
  no longer refuse → probe-signal and refusal-behavior are functionally dissociated.**
  Covers Qwen + Llama.
- **Refusal Before Decoding** — arXiv 2605.28553. https://arxiv.org/html/2605.28553v1
  Layer-wise SVM probes; per-layer-logit ENSEMBLE reaches near-perfect AUC — refusal is
  linearly evident before any token generated. Independent construction of our Signal (2).
- **Understanding & Mitigating Over-refusal via Safety Representation (MOSR)** — Zhang et al.,
  arXiv 2511.19009. https://arxiv.org/html/2511.19009v1
  Over-refusal prompts occupy an INTERMEDIATE representational region; a benign/malicious
  probe MISclassifies them as malicious; LogitLens shows contradictory next-token
  predictions. Llama3-8B-Instruct + Qwen2.5-7B; OR-Bench-80K/PHTest/FalseReject.
  **Demonstrates probe-signal vs logit-signal disagreement on exactly the over-refusal
  region — very close to our thesis.**
- **Do LMs Know When They'll Refuse?** — Gondil, arXiv 2604.00228. https://arxiv.org/html/2604.00228
  Models predict own refusal (d′ 2.4–3.5) with a drop AT the boundary; broken out by 10
  topics × 5 harm levels. Mostly closed models + Llama-3.1-405B; prompted self-report, not
  internal probing.
- Secondary: **SafeSwitch** (2502.01042); **Geometry of Refusal: Linear Instability**
  (TrustNLP 2026, 2606.22686).

## Bucket 4 — What triggers (over-)refusal: token/word/topic attribution

- **When Safety Blocks Sense** — Anonto & Nahiyan, arXiv 2512.01037. https://arxiv.org/html/2512.01037
  "Semantic confusion" = accept one phrasing, refuse a paraphrase. ParaGuard (10K
  paraphrase clusters) + token-level metrics (embedding drift, next-token-prob shift,
  perplexity). **Uses a logit signal (our Signal 3), tests Qwen3-8B vs Llama vs Mistral,
  finds boundary shape differs by model** (Llama "globally unstable," Qwen "most
  consistent"). Closest work on the word × cross-model axis. No direction/probe signal, no
  topic taxonomy.
- **Safety Mirage** — Zhang et al., arXiv 2503.11832. https://arxiv.org/html/2503.11832v4
  Safety rides on spurious word↔refusal correlations; one-word change flips refuse↔comply.
  Cleanest "safety shortcut / trigger word" statement. (VLM, mechanism is language-side.)
- **Think Before Refusal** — arXiv 2503.17882. https://arxiv.org/pdf/2503.17882
  Refusal-token attribution to "sensitive" tokens drops when swapped for neutral ones —
  word-level trigger claim, quantified.
- **Auditing Attribution Methods for Causal Faithfulness** — Eswar et al., arXiv 2607.05355.
  https://arxiv.org/html/2607.05355
  Causally audits LRP/IG/SHAP via neuron zeroing; contrastive refusal masks from
  refusal-onset vs compliance-onset logit margin; refusal lives in a redundant subspace.
  Methodology for faithful word/feature attribution + how to build a logit-margin target.
- Secondary: **Jacobian Scopes** (2601.16407); **Minimal, Local, Causal Explanations for
  Jailbreak Success** (2605.00123).

## Bucket 5 — Cross-model comparison of refusal behavior / representations

- **The Refusal–Compliance Tradeoff** — Hasan & Biswas, arXiv 2605.05427.
  https://arxiv.org/html/2605.05427
  21 open models, 7.1M+ pairs; over-refusal (ORR) and harmful-compliance (HCR) nearly
  uncorrelated (r=-0.032). **Llama = conservative (ORR 11.1%), Qwen-2.5 = balanced (low on
  both).** Post-training, not architecture, drives the boundary. LLM-as-judge.
- **Universal Refusal Circuits Across LLMs** — arXiv 2601.16034. https://arxiv.org/html/2601.16034v2
  Transfers refusal "recipes" across 8 model pairs via concept-fingerprint layer alignment,
  incl. Dense↔MoE. Answers "do refusal directions transfer?" — relevant to Qwen3-32B (large)
  vs Llama-3-8B (dense). Very recent; transfer = attenuation-of-refusal, not boundary-topic.
- **Refusal Direction is Universal Across Languages** — Wang et al., arXiv 2505.17306.
  https://arxiv.org/pdf/2505.17306 · code https://github.com/mainlp/Multilingual-Refusal
  Refusal directions ~parallel across languages and transfer cross-lingually. Evidence the
  direction is a stable, transferable object.
- Secondary: **Activation Space Interventions Transferred Between LLMs** (2503.04429);
  **No for Some, Yes for Others** (persona-driven false refusal, 2509.08075).

## Top 12 by relevance to this project
1. There Is More to Refusal than a Single Direction (2602.02132) — topic × direction; geometry-vs-behavior split; Llama-3.1-8B.
2. When Safety Blocks Sense (2512.01037) — word × model, logit signal; Qwen3-8B vs Llama.
3. MOSR (2511.19009) — over-refusal = intermediate rep; probe vs logit DISAGREE; Llama3-8B + Qwen2.5-7B.
4. Harmful Intent Geometrically Recoverable (2604.18901) — probe-vs-behavior dissociation; Qwen + Llama.
5. Refusal is a Single Direction — Arditi NeurIPS 2024 — Signal (1) foundation.
6. Geometry of Refusal: Concept Cones (2502.17420) — multi-direction critique; motivates probe ensemble.
7. Refusal Before Decoding (2605.28553) — layer-probe ensemble = Signal (2).
8. Beyond "I'm Sorry, I Can't" (2509.09708) — token/feature attribution; linear-probe misses non-linear.
9. The Refusal–Compliance Tradeoff (2605.05427) — cross-model behavioral audit; Llama vs Qwen.
10. OverKill / OKTest (2401.17633) — shortcut-keyword mechanism, lower layers; word-trigger foundation.
11. OR-Bench (2405.20947) — 10-category taxonomy; likely our data source.
12. SORRY-Bench (2406.14598) — 45-topic taxonomy for fine-grained topic analysis.

## Gap / novelty (from the survey)
No paper does the full thing: jointly cross-cutting the over-refusal boundary by ALL THREE
signals (diff-of-means direction + layer probe ensemble + teacher-forced refusal-onset
logprob) at BOTH topic and token/word granularity, contrasting **Llama-3-8B vs Qwen3-32B**.
The pieces are siloed:
- topic × direction exists (2602.02132) — direction + SAE only, one model family, no logprob, no word-level;
- word × model exists (2512.01037) — logit/embedding only, no direction/probe, no topic taxonomy;
- signal-disagreement shown in fragments — probe vs behavior (2604.18901), probe vs logit on
  the OR region (2511.19009), linear vs non-linear (2509.09708) — but nobody puts direction,
  probe, and logprob side-by-side on the SAME benign boundary and asks if they rank
  topics/words differently;
- cross-model work is either purely behavioral (2605.05427) or about direction TRANSFER for
  jailbreaking (2601.16034, 2505.17306) — not whether the OR boundary's topic/word
  composition differs between Llama and Qwen;
- almost nothing targets Qwen3-32B at large scale; internal-signal work centers on 2B–9B.

**Clearest novelty:** the "same benign boundary, three signals, two models, topic + token"
matrix — (a) do direction, probe-ensemble, and logprob disagree about which topics/words
push a benign prompt over, and (b) does that disagreement structure itself differ between
Llama-3-8B and Qwen3-32B. Bonus: every large OR benchmark is synthetic/LLM-moderated, so a
hand-validated slice would strengthen claims.
