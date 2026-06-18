# Bucketing experiment — Fir runbook

Goal: test whether finer top-bin granularity on the Claude data lifts held-out p90 OR vs the original `claude_rwr` (which used `num_bins=5`, `bin_weights=[0,0,0,1,16]`).

Wall-clock budget ~3 h on Fir. Three steps in order.

## Why this variant (V1 — finer top bins)

A laptop-side dry run of `analyze_bucketing.py` on the deduped Claude shards (12,964 pairs, 9,369 after filter) gave the following — re-run on Fir to confirm but the numbers should match exactly:

| dataset | bin (of 5) | weight | n | OR mean | OR p90 | OR max | sim mean |
|---|---|---:|---:|---:|---:|---:|---:|
| claude | 2 (dropped) | 0 | 1873 | 0.0096 | 0.0122 | 0.0132 | 0.872 |
| claude | 3 (kept)    | 1 | 1874 | 0.0492 | 0.1083 | 0.1267 | 0.875 |
| claude | 4 (kept)    | 16 | 1874 | **0.2126** | 0.2952 | 0.4638 | 0.916 |
| orp3k  | 4 (kept)    | 16 | 2840 | 0.0488 | 0.0900 | 0.2484 | 0.818 |

Two takeaways:

1. **Admitting Claude bin 2 would HURT** (mean OR 0.0096 vs orp3k bin 4 at 0.0488). The original drop scheme is correct for the bottom three quintiles of Claude. So *don't* run the "admit bin 2" variant.
2. **Within Claude bin 4 there's a lot of spread** (max 0.46 vs min ~0.13). At `num_bins=10` it gets stratified:
   - bin 9 (top 10%): OR mean **0.2585**
   - bin 8 (top 20–10%): 0.1666
   - bin 7 (top 30–20%): 0.0794

   Bin 9 alone is 1.55× bin 8 and 3.26× bin 7 — the current flat weight-16 lumps them together. V1 puts the training weight on bin 9 specifically.

## Step 1 — Diagnostic (no GPU, ~1–2 min)

From `research/overrefusal_finetuning/rwr_overrefusal/`:

```
python -u analyze_bucketing.py
```

Defaults already point at the two deduped shard dirs. Outputs:

- `prompt_iteration_results/bucketing_analysis.json`
- `prompt_iteration_results/bucketing_analysis.md`
- stdout HEADLINE block

Confirm the per-bin numbers match the table above. If they don't (e.g. fewer pairs survive the filter, very different OR distribution), check that the deduped shards on Fir match the ones used here — should be the same git-tracked dirs.

## Step 2 — Train V1 (one GPU, ~1.5–2 h)

```
sbatch run_rwr_v1_finer_top.slurm
```

CLI diff from `run_rwr_claude_only.slurm`:

- `--num_bins 10` (was 5)
- `--bin_weights 0,0,0,0,0,0,0,1,8,32` (was implicit `[0,0,0,1,16]`)
- `--output_dir ./rwr_claude_v1_checkpoints`

Same data, same LoRA/LR/epochs. Watch train_loss — claude_rwr ended at 0.077 at epoch 3; expect V1 to be similar or slightly lower (it's training on a tighter top slice).

## Step 3 — Head-to-head eval (one GPU, ~1 h)

After step 2 finishes:

```
sbatch run_eval_bucketing_compare.slurm
```

Evaluates `baseline` + `claude_rwr` (original) + `claude_rwr_v1` on the published held-out set (eval_seed=99, 200 alpaca + 200 dolly × 3 gens).

Then re-score at k=5.0 to compare apples-to-apples with published numbers:

```
python -u rescore_held_out.py \
    --input  prompt_iteration_results/held_out_eval_bucketing/held_out_eval_results.json \
    --output prompt_iteration_results/held_out_eval_bucketing/held_out_eval_results_k5.json
```

## What "wins" looks like

Compare to published claude_rwr numbers from `claude_training_results.md`:

| | alpaca p90 OR | dolly p90 OR | alpaca mean sim |
|---|---:|---:|---:|
| baseline | 0.0502 | 0.0772 | 0.59 |
| claude_rwr | 0.1501 | 0.1852 | 0.87 |
| claude_rwr_v1 | **?** | **?** | **?** |

- **Clear win**: p90 OR ≥ 0.17 / 0.21 (alpaca/dolly) at sim ≥ 0.85.
- **Wash**: within ±5% of claude_rwr at similar sim — means weight redistribution within the top 30% didn't matter; the top 20% as-is was fine.
- **Regression**: p90 OR drops, or sim drops noticeably — means the further concentration on top-decile examples either overfit to a narrow style or sacrificed similarity for delta.

## If V1 wins — next-shelf variants (one-line CLI diffs, not in this batch)

- **V1-aggressive**: `--bin_weights 0,0,0,0,0,0,0,0,1,32` — drop bin 7 entirely, train only on top 20%.
- **V1-include-tail**: `--bin_weights 0,0,0,0,0,0,1,4,8,32` — admit bin 6 with low weight; tests whether the noise floor is actually noise.
- **V1-more-bins**: `--num_bins 20 --bin_weights 0,0,...,1,4,8,32` (top 4 of 20) — finer still.

## If V1 is a wash or regression

The bottleneck isn't bucketing — the student is already near the teacher's ceiling (~92% of teacher p90). Pivot to (a) self-distillation iteration, (b) Claude+orp3k+self union runs, or (c) the "test Claude adaptations on Llama" thread you queued separately. Not in this batch.

## Files in this batch

- `analyze_bucketing.py` — diagnostic (CPU-only)
- `run_rwr_v1_finer_top.slurm` — V1 training
- `run_eval_bucketing_compare.slurm` — head-to-head eval
- `bucketing_experiment_runbook.md` — this file
