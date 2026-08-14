# RWR over-refusal attacker — consolidated experiment brief

One-file narrative of every important decision and step, Llama first, then Qwen. Detailed
per-stage docs it summarizes: `LLAMA_RETRAIN_PROGRESS.md` (retrain + audit + L31),
`LOGIT_TUNING_EXPERIMENTS.md` (tuning), `LLAMA_SCALEUP_PROGRESS.md` (scale-up + atlas),
`QWEN_3SIGNAL_PROGRESS.md` (Qwen), `SCALEUP_QWEN_COORDINATION.md` (top-level tracker).
Cluster: rorqual (`def-vganesh`). Base models: `meta-llama/Meta-Llama-3-8B-Instruct`,
`Qwen/Qwen3-32B`. Branch `alec`. All dates 2026.

---

## 0. Bottom line

- **What "RWR" is:** a reward-weighted-regression attacker — a fine-tuned model that rewrites a
  benign prompt into a version *more likely to trip a safety over-refusal, while preserving the
  benign intent*. The reward is an "over-refusal" (OR) score = a similarity gate × a refusal-delta
  signal. We tested three refusal-delta signals: a residual-stream **vector** projection, a
  multi-layer **probe** ensemble, and an output-space **logit** (P(reply starts with a refusal
  opener)).
- **Llama result (solid):** only the **logit** signal trains a real attacker. Confirmed at 40×
  scale on a fresh 8,000-rewrite corpus: **16.84% genuine over-refusal vs a 0.50% floor (~34×)**,
  intent preserved. The residual-stream **vector** signal *Goodharts* — it moves the projection
  without changing behavior — at every layer tested.
- **Qwen result (honest, weaker):** the recipe does **not** transfer to a clean win — no arm beats
  base-Qwen-as-attacker on raw rate. Best current explanation is a headroom/ceiling effect, but
  **the comparison is incomplete** (raw rate only; see §9.3) — do not over-conclude yet.

---

## 1. Why we started over — the old attackers were invalid

Every Llama RWR attacker trained before Aug 2026 was scored against `refusal_vector.layer032.npz`,
which was wrong three ways:
1. **Causally inert layer.** Ablating the L32 direction leaves harmful-prompt refusal at 98%
   (baseline 99%). L32 was picked by *largest L2 norm*, not by any causal or predictive test.
2. **Wrong extraction** — all-token mean-pooled activations, not a last-token read.
3. **A right-padding bug** (fixed 2026-06-08) that read the "last token" off right-padded batches.

It failed behaviorally, not just in theory: the vector-OR attacker induced real refusals on
**0.38%** of rewrites vs **11.25%** for the *untrained* baseline — it Goodharted the signal. The
logit/"I cannot" line and the Qwen probe line were not affected by the L32 bug.

## 2. Locked decisions for the corrected setup (owner-approved 2026-08-13)

| Decision | Value | Basis |
|---|---|---|
| Vector layer | **L17** | causally validated: ablate refusal 99→83, add-to-benign 0→99; delta-signal Spearman peaks here |
| Direction | ours (refused-vs-complied), **raw mass-mean** | measured best; Arditi harmful-vs-harmless near-equal at its best layer |
| Probe variant | **raw**, NOT covariance-corrected (LDA) | raw Spearman 0.291 > best LDA 0.261; LDA's Σ is rank-deficient at n=2,500 |
| Delta forms | vector/probe/logit all **delta** = signal(rewrite) − signal(original) | faithful to the retired reward |
| Similarity gate | **k=18.4, c=0.75** in `OR = exp(k·(sim−c))·delta` | the Qwen recipe that had worked |
| Signals trained | one attacker each: vector / probe / logit | the core comparison |
| AdvBench | direction-fitting only, closed out | owner call |

Two supporting studies behind row 1–3 (both in `LLAMA_RETRAIN_PROGRESS.md`):
- **Layer test L12 vs L17 vs L31:** L12 (Arditi's abliteration pick) is the *worst* layer for
  *ranking* which benign rewrite induces over-refusal (AUC ≈ chance); L17 is the only layer with
  both causal validation and near-best prediction; L31 is the single best *predictor* (0.291) but
  causally untested and adjacent to the inert L32.
- **Probe raw vs LDA:** raw wins; LDA improves monotonically only as it degrades toward raw — the
  signature of covariance-estimation error.

## 3. Data inventory (what is used where)

- **Direction fitting** (never used for eval): our refused-vs-complied prompt sets, 2,500/class.
- **Signal evaluation** (fitting the probe weights, choosing layers): 6,000 (original, rewrite)
  pairs, alpaca-cleaned originals, target = behavioral `dP` over 4 sampled generations.
- **RWR training pool:** 6,000 unused alpaca-cleaned prompts (seed 43) × 6 `claude-sonnet-5`
  rewrites, harm-filtered by `claude-haiku-4-5` → **35,873 pairs / 5,998 unique originals**.
- **Held-out set:** 10% of pool originals (**599**, seed 42), carved *before* any signal-specific
  filtering and excluded from all three shard sets → the three attackers eval on identical prompts
  with zero train/eval overlap.
- **Why alpaca originals for eval (not OR-Bench-Hard):** measured base-Llama refusal on *untouched*
  benign prompts — alpaca-style ≈ **0%** (full headroom), XSTest-safe 15%, OR-Bench-Hard **77.6%**
  (a ceiling that would make any attacker look useless). Alpaca gives the full 0→100 range for
  measuring *induced* over-refusal.

## 4. The 3-signal retrain (score → shards → train → eval)

- **Signals, all measured under** `system="You are a helpful assistant."`, last real prompt token,
  right-pad, thinking off:
  - vector = `proj_17(rw) − proj_17(orig)`, raw.
  - probe = weighted, standardized multi-layer delta (stored fit-time μ/σ/w).
  - logit = `P(reply begins with ANY of 5 mined Llama openers | rw) − same for orig`,
    teacher-forced. Multi-phrase openers cover **99.7%** of real Llama refusals vs **56.5%** for a
    single "I cannot" — this breadth is why the logit signal is trustworthy.
- **Bin edges (per signal, from training-only pairs):** vector/probe use **quantile** edges
  (q35/q65/q85, weights 0,1,4,16). logit uses **absolute** edges `1e-4,1e-3,1e-2,1e-1` (weights
  0,1,4,16,64) — its OR is extreme-tailed, so quantile bins would waste sampling on near-zero
  pairs; absolute edges put ~79% of training on the ~1,027 pairs with a real refusal-onset shift.
- **Training:** ~13–16 min each. **Do not select checkpoints on val_loss** — the logit run's
  val_loss *rose* across epochs while behavioral performance *improved* (matches the Qwen
  precedent). The behavioral eval is the arbiter.

## 5. Headline result + the genuine-over-refusal audit

**Behavioral eval** (200 held-out × 4 rewrites × 4 base-Llama samples, broad opener classifier,
CIs bootstrapped over originals, floor 0.5% raw / **0.0% true**):

| signal | trained attacker | base-as-attacker | verdict |
|---|--:|--:|---|
| **logit** | **15.0%** [11.7, 18.4] | 8.2% [6.1, 10.6] | **beats base — CIs disjoint** |
| probe | 7.2% [4.8, 9.9] | 9.0% [6.7, 11.3] | tie |
| vector | 5.0% [3.1, 7.2] | 9.7% [7.5, 12.3] | **worse than base — CIs disjoint** |

Trained attackers keep similarity **0.78–0.85** vs base **0.63** — base earns its raw rate by
drifting off-task, the trained attackers stay faithful *and* induce more refusals.

**The 15.0% is NOT an over-refusal rate.** Every refused rewrite was hand-classified: A = genuine
over-refusal (benign request refused on safety grounds), B = appropriate (intent genuinely shifted
to harmful), C = artifact (capability disclaimer / "instruction not to do it"), D = ambiguous.
Refuse-rate-weighted **genuine over-refusal (bucket A)**:

| signal | rwr genuine-A | base genuine-A | multiple |
|---|--:|--:|--:|
| **logit** | **7.06%** | 3.19% | **2.22×** |
| probe | 3.62% | 3.06% | 1.18× (noise) |
| vector | 2.72% | 3.09% | 0.88× (tie) |

Two things the audit corrected: (a) base's raw rate is **inflated** by a degenerate mode — it
rewrites the task into an *instruction not to do it* ("Please don't calculate the hypotenuse…" →
"I won't"), which is 20–26% of its refusals, none of the trained attackers do this; (b) the logit
advantage is *wider* on genuine over-refusal (2.22×) than on raw rate (1.82×). **This is the load-
bearing conclusion: optimizing the output logit yields a genuinely better attacker; optimizing the
residual-stream direction yields a worse-than-nothing one.**

## 6. L31 follow-up — is it the layer or the signal type?

The vector arm failed at the *causal* layer L17. Was that because L17 is causal-but-not-the-best-
*predictor* (L31 predicts better, Spearman 0.291)? Re-scored the pool with vector@L31 (probe/logit
columns byte-identical — verified), trained + evaled on the identical held-out set:
**vector-L31 = 5.4% [3.3, 7.6] vs base 8.7%** — replicates the L17 failure. **Verdict: it's the
signal *type*, not the layer.** A residual-stream projection is a poor training target at every
layer tested; the output logit is a good one. (L31 hand-classification: genuine-A 1.53%, ≤ L17's
2.72% — no better.)

## 7. Recipe tuning — both knobs tested, both lost

Two single-variable changes vs the logit baseline (`LOGIT_TUNING_EXPERIMENTS.md`):
- **More epochs (3→6): reject.** Monotonic degradation — epoch_6 12.8% < epoch_4 13.8% <
  baseline epoch_3 15.0%, at flat similarity. Epoch 3 is the behavioral peak.
- **Higher similarity floor (0.5→0.7): reject.** Raw drops to 12.0%, similarity rises 0.780→0.801,
  but the bucket-A *share* is unchanged (49.6% vs 47.9%) → **genuine over-refusal falls 7.06%→
  5.94%.** It's a proportional shrink, not a higher-precision attacker.

**Recipe confirmed optimal** (3 epochs, floor 0.5) — the scale-up uses `rwr_llama_ckpt_logit/final`
as-is. Only untested lever left: a bin-weight-profile sweep off the scarce top bin.

## 8. Scale-up — the payoff (the whole point of the retrain)

Generated a large, causally-attributable corpus from the winning logit attacker, then ran the
refusal-atlas 3-signal analysis on it (`LLAMA_SCALEUP_PROGRESS.md`):

- **Corpus:** 8,000 rewrites over 2,000 alpaca originals **provably disjoint** from all
  train/held-out/eval sets (seed-43 pool `[6000:]`; 0 leakage verified against real artifacts).
- **GOOD filter (check a, haiku benign judge):** **98.5% benign-intent-preserving** (7,877/8,000).
  Pipeline self-validated — of the 123 HARMFUL rewrites it caught, base refused 103 (appropriate).
- **Genuine over-refusal (check b, intent-based sonnet judge):** **16.84% of GOOD rewrites** are
  genuinely refused by base-Llama vs a **0.50% floor** — a **~34× lift**, spanning **732/1,995
  originals (37%)**. The 200-prompt ~15% headline reproduced at **40× scale on unseen prompts**.
- **Refusal boundary:** which signal best separates refused-vs-complied GOOD rewrites →
  **logit AUC 0.988 > probe 0.971 > vector 0.953** (clean label; regex→intent drop largest for
  logit, de-circularizing it; non-circular probe>vector holds). Converges with the atlas: the
  residual-stream direction is the weakest separator (partial topic detector), the output logit the
  cleanest.
- **Fidelity:** GOOD-refused rewrites keep MiniLM similarity **0.728** vs complied 0.797, both far
  above base 0.63 — the over-refusal is not bought by drifting off-task.
- **Honest bound:** 16.84% uses the *lenient* automated benign filter, so it is the **upper** bound
  of the same quantity whose *strict* manual **lower** bound was §5's 7.06%. Strictness-independent
  load-bearing facts: 40× scale reproduction, 0.5% floor, 37% breadth, robust signal ordering.

Corpus + scores + labels: `probe_or/results/scaleup_atlas_llama/` (`corpus_scaleup_llama.csv`,
`signals_scaleup_shard*.csv`, `pairs_scaleup_shard*.csv`, `benign_scaleup.csv`, `judge_scaleup.csv`,
`samples_scaleup_merged.json`).

---

## 9. Qwen mirror

### 9.1 Setup (faithful mirror of the Llama pipeline)

Qwen3-32B, QLoRA 4-bit on one 80GB H100. All three signal fits exist at Qwen's **causal layer
L58** (activation-addition → 93% refusal, validated). Scored the same-scale pool (35,850 pairs /
5,998 originals), common held-out carve (599, seed 42), trained vector/probe/logit, evaled each vs
base-Qwen-as-attacker. One structural fact: **for Qwen the probe ≡ the vector** (Spearman(d_vector,
d_probe) = **1.000**, both L58), and the standardization shift guts the probe's trainable pool to
**1,210 pairs vs vector's 16,823** — so the probe arm is a mirror-completeness arm expected to
underperform on pool size, and the informative arms are **vector** (faithful residual-stream) and
**logit** (distinct output-space, the Llama winner).

### 9.2 Results (599 held-out, floor ~0.3%)

| arm | rwr final | base-as-attacker | read |
|---|--:|--:|---|
| vector | 2.8% [1.6,4.2], sim 0.845 | 8.9%, sim 0.817 | **Goodhart, below base** — high sim, lowest rate |
| probe | 11.8% [8.8,15.2], sim 0.798 | 10.9%, sim 0.815 | nominally best but ≈ base (CIs overlap), overfit pool |
| logit | 8.2% [5.8,11.1], sim 0.799 | 10.6%, sim 0.806 | below base |

**No arm cleanly beats base-Qwen-as-attacker.** Corpus generated from the **logit** arm (pool
health + cross-model comparability — same 2,000 prompts as Llama): `scaleup_corpus_qwen_logit.csv`
(8,000 rewrites, 0 `<think>` leakage). One convergent finding worth keeping: **the vector arm
Goodharted on Qwen too** (highest similarity, lowest/below-base rate) — second-model evidence the
residual-stream direction is a poor training target.

### 9.3 "Why didn't anything beat Qwen?" — is the training bad? (read this)

Short answer: **probably not a training bug, but the comparison is not yet complete enough to
close the question.** Reasons to think the training is fine, then the real gap:

**The training worked mechanically.** Bins realized as designed, reward rises monotonically across
bins, the logit top bin carried real signal (delta_mean 0.371 vs 0.000 in low bins), and the
attacker demonstrably learned the loaded-word strategy (the corpus shows "clean a kitchen" →
"exploitation plan to strip a kitchen of all grime"). The **logit** arm — the one that won on
Llama — trained on a healthy 24k-pair pool and still didn't beat base, so this is not a pool-size
or optimization artifact for the arm that matters.

**The most likely benign cause is headroom.** Base-Qwen-as-attacker already induces ~10% refusal
**at high fidelity** (sim ~0.81). Base-*Llama*-as-attacker only reaches ~8% and does it by drifting
off-task (sim 0.63). So base-Qwen is genuinely a much stronger zero-shot attacker, and "beat ~10%
while staying faithful" is a far higher bar than "beat ~8% that's half off-task." We also verified
this is **not a harness artifact** (thinking disabled, real refusals detected, floor ~0.3%) and
debunked the earlier 37.9% Qwen number as a small-n (71 vs 599) / high-base (~30%) /
different-probe artifact that doesn't survive on the representative set.

**But here is the real gap, and it's the thing to check before concluding "no transfer":** the
Qwen comparison is **raw-rate only.** On Llama, the raw comparison was *misleading in both
directions* — base's raw rate was inflated ~20–26% by the "instruction not to do it" artifact, and
the trained attacker's genuine-over-refusal advantage (2.22×) was *larger* than its raw advantage
(1.82×). **We have not run the genuine-over-refusal bucket decomposition on the Qwen arms or on
base-Qwen.** So "probe 11.8% ≈ base 10.9%" could be hiding either (a) a base-Qwen raw rate equally
inflated by artifacts → a real trained-attacker win on *genuine* over-refusal, or (b) a probe whose
extra refusals are intent-shifted appropriate refusals → a true no-win. The agent's light
qualitative look leaned toward (b) for probe (its refusals looked intent-shifted: "indoctrinate
children", "infiltrate and dominate"), but that was not the systematic exhaustive audit we did for
Llama.

**Recommended next step (§10) to settle it:** run the same A/B/C/D genuine-over-refusal audit on
the Qwen arms and base-Qwen — cheaply, by pushing the Qwen corpus through the *already-built*
shared downstream (substrate → atlas 3-signal → benign filter). That either confirms the headroom
story (genuine ≈ base) or flips it (genuine > base despite raw ties). Until then, the honest
statement is **"no Qwen arm beats base on raw induced-refusal rate,"** not "Qwen training failed."

## 10. Open items / next steps

1. **Settle the Qwen question:** run the genuine-over-refusal audit / shared downstream on the Qwen
   corpus + base-Qwen (substrate → atlas → benign filter; Qwen params in `QWEN_3SIGNAL_PROGRESS.md`).
   This is the rigorous follow-up to §9.3 and the first thing to do before any Qwen conclusion.
2. **Strict Llama number:** a stricter automated benign filter to pin the Llama over-refusal figure
   nearer its ~7% strict floor rather than the 16.84% lenient ceiling.
3. Optional: bin-weight-profile sweep on the Llama logit recipe (only untested lever).
4. Optional: causally validate Qwen headroom directly — does base-Qwen's ~10% survive the
   genuine-OR filter, or is it artifact like base-Llama's was?

---

*Everything above is reproducible from the per-stage docs and the artifacts under
`probe_or/results/`. Job IDs and exact commands are in those docs.*
