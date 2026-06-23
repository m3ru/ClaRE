# Bucketing diagnostic

BinningConfig: `num_bins=5`, `bin_weights=[0.0, 0.0, 0.0, 1.0, 16.0]`, `k=5.0`, `c=0.75`, `d=100.0`, `similarity_floor=0.5`

## Per-dataset, per-bin

| dataset | bin | weight | n | unique_origs | OR mean | OR median | OR p90 | OR max | OR std | sim mean | >0.05 | >0.10 | >0.20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| claude | 0 | 0.0 | 1663 | 1087 | 0.0025 | 0.0025 | 0.0045 | 0.0050 | 0.0014 | 0.854 | 0.00 | 0.00 | 0.00 |
| claude | 1 | 0.0 | 1665 | 1064 | 0.0079 | 0.0079 | 0.0103 | 0.0109 | 0.0017 | 0.853 | 0.00 | 0.00 | 0.00 |
| claude | 2 | 0.0 | 1662 | 1026 | 0.0147 | 0.0146 | 0.0178 | 0.0185 | 0.0022 | 0.859 | 0.00 | 0.00 | 0.00 |
| claude | 3 | 1.0 | 1665 | 966 | 0.0242 | 0.0238 | 0.0296 | 0.0320 | 0.0038 | 0.870 | 0.00 | 0.00 | 0.00 |
| claude | 4 | 16.0 | 1664 | 768 | 0.0582 | 0.0461 | 0.0914 | 0.3255 | 0.0356 | 0.890 | 0.42 | 0.08 | 0.01 |
| orp3k | 0 | 0.0 | 2719 | 1372 | 0.0014 | 0.0014 | 0.0027 | 0.0030 | 0.0009 | 0.847 | 0.00 | 0.00 | 0.00 |
| orp3k | 1 | 0.0 | 2719 | 1322 | 0.0049 | 0.0049 | 0.0065 | 0.0070 | 0.0011 | 0.836 | 0.00 | 0.00 | 0.00 |
| orp3k | 2 | 0.0 | 2718 | 1281 | 0.0097 | 0.0095 | 0.0122 | 0.0130 | 0.0017 | 0.839 | 0.00 | 0.00 | 0.00 |
| orp3k | 3 | 1.0 | 2719 | 1071 | 0.0178 | 0.0173 | 0.0228 | 0.0248 | 0.0033 | 0.837 | 0.00 | 0.00 | 0.00 |
| orp3k | 4 | 16.0 | 2719 | 766 | 0.0523 | 0.0402 | 0.0954 | 0.2645 | 0.0332 | 0.814 | 0.34 | 0.09 | 0.00 |

## Headline: are Claude's dropped bins above orp3k's kept-bin OR?

orp3k bin 4 (the only bin orp3k weights heavily): mean OR **0.0523**, median 0.0402

| dataset | bin | weight | n | OR mean | OR mean ÷ orp3k bin4 mean | beats orp3k bin4? |
|---|---|---:|---:|---:|---:|---|
| claude | 0 | 0.0 | 1663 | 0.0025 | 0.05× | no |
| claude | 1 | 0.0 | 1665 | 0.0079 | 0.15× | no |
| claude | 2 | 0.0 | 1662 | 0.0147 | 0.28× | no |
| claude | 3 | 1.0 | 1665 | 0.0242 | 0.46× | no |
| claude | 4 | 16.0 | 1664 | 0.0582 | 1.11× | **YES** |

## Within-prompt vs across-prompt OR variance

If within ≫ across, top-N per prompt loses a lot of signal — argues for keeping bins.
If within ≪ across, bins on raw pool are fine; you could equivalently take top-K per prompt.

| dataset | prompts w/ multi samples | within-prompt var mean | across-prompt var | within/across |
|---|---:|---:|---:|---:|
| claude | 2773 | 0.00011 | 0.00055 | 0.19 |
| orp3k | 2033 | 0.00016 | 0.00029 | 0.54 |

## Alternate num_bins on Claude (visibility only — does not change training)

### num_bins=10

| bin | n | OR mean | OR median | OR max |
|---|---:|---:|---:|---:|
| 0 | 832 | 0.0013 | 0.0013 | 0.0025 |
| 1 | 831 | 0.0037 | 0.0037 | 0.0050 |
| 2 | 833 | 0.0064 | 0.0064 | 0.0079 |
| 3 | 832 | 0.0094 | 0.0094 | 0.0109 |
| 4 | 831 | 0.0128 | 0.0128 | 0.0146 |
| 5 | 831 | 0.0166 | 0.0166 | 0.0185 |
| 6 | 833 | 0.0210 | 0.0209 | 0.0238 |
| 7 | 832 | 0.0274 | 0.0272 | 0.0320 |
| 8 | 832 | 0.0382 | 0.0375 | 0.0461 |
| 9 | 832 | 0.0782 | 0.0642 | 0.3255 |

## Decision criteria

- Compare Claude's dropped bins to orp3k bin 4. If any are above, un-dropping is
  motivated. (On the laptop dry run: Claude bin 2 was only 0.20× orp3k bin 4 — so
  the original drop scheme is correct and V1 (finer top bins) is the better bet.)
- Inspect the alternate `num_bins=10` table on Claude. If the top decile (bin 9)
  has noticeably higher OR than bin 8 (e.g. > 1.4×), the current flat weight-16
  wastes signal by lumping the top 20% together. → `run_rwr_v1_finer_top.slurm`.
- If within/across OR variance is high (> ~1.0): per-prompt top-K sampling may
  beat the pooled binning entirely (not in this batch).