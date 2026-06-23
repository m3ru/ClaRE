# Bucketing diagnostic

BinningConfig: `num_bins=5`, `bin_weights=[0.0, 0.0, 0.0, 1.0, 16.0]`, `k=5.0`, `c=0.75`, `d=100.0`, `similarity_floor=0.5`

## Per-dataset, per-bin

| dataset | bin | weight | n | unique_origs | OR mean | OR median | OR p90 | OR max | OR std | sim mean | >0.05 | >0.10 | >0.20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| claude | 0 | 0.0 | 1874 | 1334 | 0.0017 | 0.0018 | 0.0031 | 0.0034 | 0.0010 | 0.838 | 0.00 | 0.00 | 0.00 |
| claude | 1 | 0.0 | 1874 | 1320 | 0.0051 | 0.0051 | 0.0066 | 0.0070 | 0.0011 | 0.848 | 0.00 | 0.00 | 0.00 |
| claude | 2 | 0.0 | 1873 | 1264 | 0.0096 | 0.0094 | 0.0122 | 0.0132 | 0.0017 | 0.872 | 0.00 | 0.00 | 0.00 |
| claude | 3 | 1.0 | 1874 | 1208 | 0.0492 | 0.0327 | 0.1083 | 0.1267 | 0.0363 | 0.875 | 0.39 | 0.14 | 0.00 |
| claude | 4 | 16.0 | 1874 | 1121 | 0.2126 | 0.2023 | 0.2952 | 0.4638 | 0.0572 | 0.916 | 1.00 | 1.00 | 0.52 |
| orp3k | 0 | 0.0 | 2840 | 1428 | 0.0015 | 0.0015 | 0.0027 | 0.0030 | 0.0009 | 0.849 | 0.00 | 0.00 | 0.00 |
| orp3k | 1 | 0.0 | 2840 | 1334 | 0.0050 | 0.0049 | 0.0066 | 0.0070 | 0.0011 | 0.839 | 0.00 | 0.00 | 0.00 |
| orp3k | 2 | 0.0 | 2839 | 1303 | 0.0095 | 0.0094 | 0.0118 | 0.0124 | 0.0016 | 0.841 | 0.00 | 0.00 | 0.00 |
| orp3k | 3 | 1.0 | 2840 | 1110 | 0.0170 | 0.0166 | 0.0217 | 0.0234 | 0.0031 | 0.837 | 0.00 | 0.00 | 0.00 |
| orp3k | 4 | 16.0 | 2840 | 813 | 0.0488 | 0.0368 | 0.0900 | 0.2484 | 0.0313 | 0.818 | 0.30 | 0.08 | 0.00 |

## Headline: are Claude's dropped bins above orp3k's kept-bin OR?

orp3k bin 4 (the only bin orp3k weights heavily): mean OR **0.0488**, median 0.0368

| dataset | bin | weight | n | OR mean | OR mean ÷ orp3k bin4 mean | beats orp3k bin4? |
|---|---|---:|---:|---:|---:|---|
| claude | 0 | 0.0 | 1874 | 0.0017 | 0.04× | no |
| claude | 1 | 0.0 | 1874 | 0.0051 | 0.10× | no |
| claude | 2 | 0.0 | 1873 | 0.0096 | 0.20× | no |
| claude | 3 | 1.0 | 1874 | 0.0492 | 1.01× | **YES** |
| claude | 4 | 16.0 | 1874 | 0.2126 | 4.36× | **YES** |

## Within-prompt vs across-prompt OR variance

If within ≫ across, top-N per prompt loses a lot of signal — argues for keeping bins.
If within ≪ across, bins on raw pool are fine; you could equivalently take top-K per prompt.

| dataset | prompts w/ multi samples | within-prompt var mean | across-prompt var | within/across |
|---|---:|---:|---:|---:|
| claude | 3326 | 0.00149 | 0.00605 | 0.25 |
| orp3k | 2054 | 0.00015 | 0.00026 | 0.56 |

## Alternate num_bins on Claude (visibility only — does not change training)

### num_bins=10

| bin | n | OR mean | OR median | OR max |
|---|---:|---:|---:|---:|
| 0 | 937 | 0.0009 | 0.0009 | 0.0018 |
| 1 | 937 | 0.0026 | 0.0026 | 0.0034 |
| 2 | 937 | 0.0042 | 0.0042 | 0.0051 |
| 3 | 937 | 0.0060 | 0.0060 | 0.0070 |
| 4 | 936 | 0.0082 | 0.0081 | 0.0094 |
| 5 | 937 | 0.0110 | 0.0109 | 0.0132 |
| 6 | 937 | 0.0190 | 0.0175 | 0.0327 |
| 7 | 937 | 0.0794 | 0.0814 | 0.1267 |
| 8 | 937 | 0.1666 | 0.1675 | 0.2022 |
| 9 | 937 | 0.2585 | 0.2503 | 0.4638 |

## Decision criteria

- Compare Claude's dropped bins to orp3k bin 4. If any are above, un-dropping is
  motivated. (On the laptop dry run: Claude bin 2 was only 0.20× orp3k bin 4 — so
  the original drop scheme is correct and V1 (finer top bins) is the better bet.)
- Inspect the alternate `num_bins=10` table on Claude. If the top decile (bin 9)
  has noticeably higher OR than bin 8 (e.g. > 1.4×), the current flat weight-16
  wastes signal by lumping the top 20% together. → `run_rwr_v1_finer_top.slurm`.
- If within/across OR variance is high (> ~1.0): per-prompt top-K sampling may
  beat the pooled binning entirely (not in this batch).