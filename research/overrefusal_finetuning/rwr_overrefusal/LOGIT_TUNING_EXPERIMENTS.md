# Logit-reward RWR attacker — training-tuning experiments

Opened 2026-08-14, <cluster> (`<ACCOUNT>`), branch `main`. Base model
`meta-llama/Meta-Llama-3-8B-Instruct`. All runs reuse the validated logit shard set
`probe_or/results/llama_scored/llama_shards_logit` and the common held-out eval split
`probe_or/results/llama_scored/heldout_originals.json` (200 prompts, identical to every
prior arm — directly comparable, zero train/eval overlap).

## Baseline to beat (job 18997726 / eval 18999052)

The winning arm from the retrain. Raw refusal induced on base-Llama, 200 held-out prompts
× 4 rewrites × 4 samples, broad opener classifier, 95% CI bootstrapped over originals:

| ckpt | raw refuse | 95% CI | similarity | genuine OR (audit) |
|---|--:|--:|--:|--:|
| logit `final` | **15.0%** | [11.7, 18.4] | 0.780 | 7.06% (bucket A) |
| logit `epoch_2` | 11.5% | — | — | — |
| its matched base | 8.2% | [6.1, 10.6] | 0.63 | 3.19% |
| floor (orig) | 0.5% | [0.0, 1.5] | 1.00 | 0.0% |

Genuine-over-refusal is the honest metric (raw × bucket-A share ≈ 48%). Fidelity
(similarity) matters because ~21% of the winner's refusals were bucket-B *appropriate*
refusals of genuinely intent-shifted rewrites — buying raw rate by dropping similarity
just trades genuine OR for bucket B. **A tuning win must hold or raise similarity, not
only raw rate.**

## What the levers actually do (pre-registered analysis, no GPU)

Recomputed OR = `exp(18.4·(sim−0.75))·delta` over the 32,268 logit-shard pairs at load
(`rwr_data.recompute_or_scores`), so floor / k / edge changes need **no rescore job**.

**Baseline bin profile (k=18.4, floor 0.5), reproduced exactly:**
bins 18685 / 1272 / 882 / 775 / 1027, sampling 0 / 1.5 / 4.3 / 15.0 / **79.3%**.
Top bin (weight 64) carries 79% of training on 1,027 pairs: `d_mean=+0.581`, `sim_mean=0.783`.

- **Similarity floor 0.5 → 0.7.** Drops the pool 22,641 → 16,576 but the *top bin stays
  intact* (1,027 → 839 pairs) and its `sim_mean` rises 0.783 → 0.809; the weight-16 bin's
  `d_mean` collapses (+0.264 → +0.022, i.e. floor 0.5 was admitting low-delta pairs there).
  Prediction: modestly higher-fidelity rewrites at similar or slightly lower raw rate —
  aimed straight at the genuine-OR metric. **Cheapest, highest-information lever.**
- **More epochs (3 → 6).** Top bin is only 1,027 pairs; at 4,000 samples/epoch that's
  ~3× repetition/epoch. The winner's `final` (15.0%) beat `epoch_2` (11.5%) despite rising
  val_loss, so behavioral performance was *still climbing* at epoch 3 — more passes may
  extend that. Risk: memorizing the scarce top bin (the Qwen precedent plateaued). Eval
  `final`(=epoch 6) and `epoch_4` to see the trajectory.
- **Sharper sim gate k=18.4 → 35 (analyzed, NOT run).** Only reshuffles 19% of the top bin,
  and the pairs it *adds* are high-sim/near-zero-delta (`d=0.008, sim=0.884`) while the ones
  it *drops* are high-delta/low-sim (`d=0.744, sim=0.674`) — i.e. k=35 would *demote the
  strongest refusal-moving pairs* to chase similarity. Rejected: it optimizes the gate, not
  the behavior. Floor is the cleaner fidelity lever.
- **Bin edges / weights.** Already tuned once (absolute edges validated on this signal
  family; top-bin `delta_mean=0.581` vs 0.000 in bin 0). Not touched here — would confound
  the two clean single-variable runs.

## Runs

| tag | change vs baseline | train job | eval job | ckpts evaled |
|---|---|--:|--:|---|
| `_tune1_e6` | epochs 3 → **6** | 19013240 | 19013242 | final(=e6), epoch_4 |
| `_tune2_f70` | sim_floor 0.5 → **0.7** | 19013241 | 19013243 | final, epoch_2 |

Both hold everything else at the baseline logit recipe (LR 1.5e-5, bs4×ga4, LoRA r32,
edges 1e-4..1e-1, weights 0/1/4/16/64, k18.4/c0.75/d1.0, 4000 samples/epoch, seed 42).
Single-variable by design so each result is attributable.

## Results

All four jobs COMPLETED clean. Raw behavioral numbers (200 held-out prompts, same as
every prior arm; 95% CI over originals):

| arm | ckpt | raw refuse | 95% CI | similarity | ge1x | its base |
|---|---|--:|--:|--:|--:|--:|
| **baseline** (e3, floor .5) | final | **15.0%** | [11.7, 18.4] | 0.780 | — | 8.2% |
| tune1_e6 (epochs→6) | final(=e6) | 12.78% | [9.5, 16.4] | 0.782 | 14.6% | 8.38% |
| tune1_e6 | epoch_4 | 13.81% | [10.7, 17.2] | 0.779 | 16.2% | — |
| tune2_f70 (floor→.7) | final | 12.03% | [8.9, 15.4] | **0.801** | 14.1% | 8.72% |
| tune2_f70 | epoch_2 | 10.19% | [7.0, 13.6] | **0.812** | 11.6% | — |

**Neither knob beat baseline on raw rate.** All point estimates fall below 15.0%, though
every CI overlaps the baseline CI (no variant is *significantly* worse either).

### More epochs → monotonically worse (VERDICT: reject)

e6 12.78% < e4 13.81% < baseline-e3 15.0%, at flat similarity (0.78). Epoch 3 was the
behavioral peak; past it the attacker degrades. This is the *opposite* of the naive
val-loss reading and confirms epoch 3 was already the right stopping point — the scarce
1,027-pair top bin (~3× repetition/epoch) starts to overfit beyond epoch 3. Not
classified further: lower raw at unchanged fidelity cannot yield higher genuine-OR, and
there is no fidelity story to redeem it.

### Higher similarity floor (0.5 → 0.7) → fidelity up, genuine-OR NOT redeemed (VERDICT: reject)

The one variant with a fidelity story: similarity 0.780 → 0.801 (+0.021), the predicted
effect. It traded ~3 raw points for that. The question was whether the higher fidelity cut
bucket-B intent-shifts, letting genuine over-refusal hold near baseline. **It did not.**

Exhaustive hand-classification of all 113 f70-`final` refused rewrites (A/B/C/D rubric
from `LLAMA_RETRAIN_PROGRESS.md`; genuine-A = Σ refuse_rate over bucket-A / 800, matching
the doc's 7.06%):

| arm | n refused | A | B | C | D | A-share | **genuine-A** |
|---|--:|--:|--:|--:|--:|--:|--:|
| baseline logit final | 142 | 68 | 30 | 7 | 37 | 47.9% | **7.06%** |
| **tune2_f70 final** | 113 | 56 | 21 | 10 | 26 | **49.6%** | **5.94%** |

The bucket-A *share* is essentially unchanged (49.6% vs 47.9%) — **raising the floor did
not improve the quality mix, it just made the attacker fire less often.** The raw-rate
drop flows straight through to genuine over-refusal: 7.06% → 5.94%. f70 is a *proportional
shrink* of the baseline, not a higher-precision version of it. The predicted
fidelity/rate tradeoff happened and landed on the wrong side.

f70 still beats its own base arm on genuine over-refusal (base arm hand-classified: bucket-A
share 17.9%, genuine-A 1.69%), but that comparison is **labeler-confounded** — this pass
graded the base arm's "may be harmful"-qualifier reframes more strictly (→ bucket D) than
the original audit did (which put base A-share at 36.4% / genuine 3.19%), so the apparent
3.5× multiple is not comparable to the doc's 2.21× and should not be quoted. The
labeler-independent facts are: (a) f70 raw < baseline raw, (b) f70 A-share ≈ baseline
A-share, therefore (c) f70 genuine-A < baseline genuine-A. The trained attacker's refusals
remain far more often genuine over-refusal than base's (49.6% vs 17.9% A-share, same
labeler) — the core retrain finding survives, f70 just doesn't improve on it.

## Bottom line — recipe confirmed, no change for scale-up

**Keep the original logit recipe: 3 epochs, similarity floor 0.5, edges 1e-4..1e-1,
weights 0/1/4/16/64, k18.4/c0.75, LR 1.5e-5, 4000 samples/epoch.** Both tuning directions
were tested single-variable and both lost:
- **epochs**: 3 is the behavioral peak; 4 and 6 monotonically degrade raw rate at flat
  fidelity.
- **similarity floor**: 0.7 buys +0.02 similarity but loses raw rate *and* genuine-OR with
  no change in refusal quality — a net loss on the metric that matters.

The scale-up run (generate fresh OR rewrites with the logit attacker for the atlas 3-signal
analysis) should use the **baseline `probe_or/results/rwr_llama_ckpt_logit/final`**
checkpoint unchanged. This tuning pass is negative evidence that the recipe was already at
a local optimum on both knobs — useful to have ruled out before the scale-up spend, not a
lever to pull.

### Not tested (candidates for a future pass, if warranted)
- LoRA rank / LR / bin weights — held fixed to keep these two runs single-variable; a
  weight-profile sweep (e.g. flatten 0/1/4/16/64 → 0/1/2/4/8 to spread sampling off the
  scarce top bin) is the most plausible remaining lever, but is lower-priority than
  running the scale-up on the confirmed recipe.
- The k=35 sharper-gate idea was analyzed and rejected pre-training (it demotes the
  highest-delta pairs to chase similarity — see the lever analysis above).

