# Plan review — what survived, what didn't, and the revised plan

29 findings raised across 4 lenses; the refutation pass was cut short (5 verdicts), so the
triage below is mine, verified directly against the repo. Numbers here are re-checked, not
quoted from the reviewers.

## The three that change the plan

**1. For Llama, the construction question is already answered, and it's null.**
`probe_or/results/direction_comparison.json`, Spearman vs behavioural dP (6,000 pairs):

| direction | @L12 | @L17 | @L31 |
|---|--:|--:|--:|
| ours (refusal-vs-compliance) | 0.139 | 0.2615 | 0.2912 |
| arditi (harmful-vs-harmless) | 0.058 | 0.2563 | 0.2824 |

At **matched** layers the two constructions are indistinguishable (Δ 0.005 at L17, 0.009 at
L31). The "ours@L17 0.261 vs arditi@L12 0.058" contrast I cited earlier is **almost entirely a
layer effect, not a construction effect** — comparing each vector at its own selected layer
measures the layer. Llama does not need Phase B; Qwen does, because Qwen's labels were
genuinely wrong.

Second-order but useful: **L12 is a bad reward layer for both constructions** (0.139 / 0.058).
The ablation-selected layer is not the layer to score with, whichever vector you build.

**2. We DO have the Llama Arditi ingredients — my earlier correction was wrong.**
`probe_or/activations_arditi/` holds `acts_ref` (512, 33, 4096) harmful and `acts_ben`
(512, 33, 4096) harmless Llama activations at every layer, in the same right-padded
last-real-token context as everything else. The Llama Arditi direction is a diff-of-means away
— seconds of CPU, no GPU, no queue. What was never saved is the *selected-layer npz*, which no
longer matters since we can build it at any layer.

**3. Swapping `d` in `probe_absolute.npz` would corrupt the file.**
`mu`, `sd`, `w` and `best_layer` are all derived from projections onto that same `d`
(`fit_probe_absolute.py`). Replacing the direction alone leaves a standardisation and a layer
choice fitted to a vector that is no longer there. New file + re-fit, never in place. I already
wrote the new Qwen direction to `probe_absolute_qwennative.npz` rather than overwriting.

Related and load-bearing: **the deployed Qwen RWR reward does not read `probe_absolute.npz`**.
`run_score_qwen_3sig.slurm` uses `--scorer qwen_signals/qwen_probe_raw.npz`, built by
`probe_qwen_signals.py` from the same Llama-labelled activations. Fixing only `probe_absolute`
would leave the reward untouched.

## Two safety items, one already acted on

- **19397514 was live and flawed.** It was PENDING on a dependency with no hold, so it would
  have fired tonight comparing two directions at mismatched layers against `d_logit` as the
  target — and `or_logit = gate * d_logit`, so a logit candidate scores ~1.0 against itself by
  construction. **Held.** Its batch script was frozen at submit, so it must be cancelled and
  resubmitted, not edited.
- **Do not write to `probe_or/qwen_activations/`.** Those `.npy` files are gitignored and exist
  nowhere else; they generated every currently-reported Qwen number. New paths only.

## What the reviewers got wrong

- **"The two pools share zero prompts"** — false. All **1,045** behav originals are present in
  the 5,998-original scored pool. The join works. The real issue is smaller: my script's
  `--n_max 20000` default truncates a 35,850-row file read in file order.
- **"Qwen's best_layer was produced by probe_ensemble.py"** — wrong producer, and the refuter
  found the consequence empirically nil.

## Sample size: 389/class is fine

The repo already ran this control: `ours-raw-n512` @L17 = **0.2620** vs full 2,500/class
**0.2615**. Cutting the fitting set to 512/class moves the behavioural Spearman by 0.0005. Our
Qwen-native 389/class needs no GPU generation pass to be adequate.

## Two confounds worth stating rather than fixing

- **Wrapper/topic confound in both models' refusal-vs-compliance vectors.** Both classes come
  from one Reddit pool wrapped in jailbreak templates and split by whether the model refused, so
  wrapper identity correlates with class. This affects the Llama vector too — it is not
  "correct", just correctly *labelled*.
- **Source correlates with label in the Qwen-native split.** Qwen-refused items still come
  mostly from the Llama-refusal pool. Stratifying across both source pools would fix it; worth
  reporting either way.

## Revised plan

**Tonight, CPU only, no queue (~1 h):**
- A1/A2 — Qwen-native refusal-vs-compliance direction from stored acts + Qwen's own refusal
  rates. **Already built**: 389/class, cos 0.805 with the Llama-labelled one at L57.
- A3 — Llama **and** Qwen Arditi directions. Llama from `activations_arditi` (CPU).
  Qwen needs one ~10 min forward pass for harmful/harmless acts — the only GPU item.
- B — per-layer Spearman vs **dP_behav** (not d_logit), all candidates, **matched layers**,
  paired bootstrap. Llama arm is confirmatory; Qwen arm is the real question.

**Deferred:** Phase C training. Llama's construction answer is already null at matched layers,
so the only honest training arm is Qwen, and only if B shows separation.

**Still queued:** 19396818 (ablation layer selection, ~20 h out). Not on the critical path any
more — it answers "which layer is causally necessary", not "which vector rewards better".
