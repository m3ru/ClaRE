# Bucketing diagnostic

BinningConfig: `num_bins=5`, `bin_weights=[0.0, 0.0, 0.0, 1.0, 16.0]`, `k=5.0`, `c=0.75`, `d=100.0`, `similarity_floor=0.5`

## Per-dataset, per-bin

| dataset | bin | weight | n | unique_origs | OR mean | OR median | OR p90 | OR max | OR std | sim mean | >0.05 | >0.10 | >0.20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| claude_FIXED | 0 | 0.0 | 1663 | 1087 | 0.0025 | 0.0025 | 0.0045 | 0.0050 | 0.0014 | 0.854 | 0.00 | 0.00 | 0.00 |
| claude_FIXED | 1 | 0.0 | 1665 | 1064 | 0.0079 | 0.0079 | 0.0103 | 0.0109 | 0.0017 | 0.853 | 0.00 | 0.00 | 0.00 |
| claude_FIXED | 2 | 0.0 | 1662 | 1026 | 0.0147 | 0.0146 | 0.0178 | 0.0185 | 0.0022 | 0.859 | 0.00 | 0.00 | 0.00 |
| claude_FIXED | 3 | 1.0 | 1665 | 966 | 0.0242 | 0.0238 | 0.0296 | 0.0320 | 0.0038 | 0.870 | 0.00 | 0.00 | 0.00 |
| claude_FIXED | 4 | 16.0 | 1664 | 768 | 0.0582 | 0.0461 | 0.0914 | 0.3255 | 0.0356 | 0.890 | 0.42 | 0.08 | 0.01 |
| orp3k | 0 | 0.0 | 2840 | 1428 | 0.0015 | 0.0015 | 0.0027 | 0.0030 | 0.0009 | 0.849 | 0.00 | 0.00 | 0.00 |
| orp3k | 1 | 0.0 | 2840 | 1334 | 0.0050 | 0.0049 | 0.0066 | 0.0070 | 0.0011 | 0.839 | 0.00 | 0.00 | 0.00 |
| orp3k | 2 | 0.0 | 2839 | 1303 | 0.0095 | 0.0094 | 0.0118 | 0.0124 | 0.0016 | 0.841 | 0.00 | 0.00 | 0.00 |
| orp3k | 3 | 1.0 | 2840 | 1110 | 0.0170 | 0.0166 | 0.0217 | 0.0234 | 0.0031 | 0.837 | 0.00 | 0.00 | 0.00 |
| orp3k | 4 | 16.0 | 2840 | 813 | 0.0488 | 0.0368 | 0.0900 | 0.2484 | 0.0313 | 0.818 | 0.30 | 0.08 | 0.00 |

## Headline: are Claude's dropped bins above orp3k's kept-bin OR?

orp3k bin 4 (the only bin orp3k weights heavily): mean OR **0.0488**, median 0.0368

| dataset | bin | weight | n | OR mean | OR mean ÷ orp3k bin4 mean | beats orp3k bin4? |
|---|---|---:|---:|---:|---:|---|
| claude_FIXED | 0 | 0.0 | 1663 | 0.0025 | 0.05× | no |
| claude_FIXED | 1 | 0.0 | 1665 | 0.0079 | 0.16× | no |
| claude_FIXED | 2 | 0.0 | 1662 | 0.0147 | 0.30× | no |
| claude_FIXED | 3 | 1.0 | 1665 | 0.0242 | 0.50× | no |
| claude_FIXED | 4 | 16.0 | 1664 | 0.0582 | 1.19× | **YES** |

## Within-prompt vs across-prompt OR variance

If within ≫ across, top-N per prompt loses a lot of signal — argues for keeping bins.
If within ≪ across, bins on raw pool are fine; you could equivalently take top-K per prompt.

| dataset | prompts w/ multi samples | within-prompt var mean | across-prompt var | within/across |
|---|---:|---:|---:|---:|
| claude_FIXED | 2773 | 0.00011 | 0.00055 | 0.19 |
| orp3k | 2054 | 0.00015 | 0.00026 | 0.56 |

## Decision criteria

- Compare Claude's dropped bins to orp3k bin 4. If any are above, un-dropping is
  motivated. (On the laptop dry run: Claude bin 2 was only 0.20× orp3k bin 4 — so
  the original drop scheme is correct and V1 (finer top bins) is the better bet.)
- Inspect the alternate `num_bins=10` table on Claude. If the top decile (bin 9)
  has noticeably higher OR than bin 8 (e.g. > 1.4×), the current flat weight-16
  wastes signal by lumping the top 20% together. → `run_rwr_v1_finer_top.slurm`.
- If within/across OR variance is high (> ~1.0): per-prompt top-K sampling may
  beat the pooled binning entirely (not in this batch).