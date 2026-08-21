# Causal results — how many directions carry over-refusal, and do they generalise?

All numbers below are measured on **held-out originals**: the directions are fitted on a train half and never see the prompts they are evaluated on. This closes the leakage in the earlier frame ablation, which fitted on the same `or_high` pool it evaluated against.

The rank question is posed **causally** (how many directions must be ablated before the behaviour stops), because the correlational version is ill-posed — for a two-class problem Fisher's LDA has exactly one discriminant by construction, which is why the retracted k\* returned 1 for injected ranks 1, 2, 3 and 5.


## Causal rank sweep (held-out: 400 originals, 200 AdvBench)

`ours k` ablates the first k directions jointly; `random k` ablates k random orthonormal directions — the rank-matched control. A drop at k is only a result if the random control at the same k did not produce it.

| k | over-refusal (ours) | over-refusal (random k) | AdvBench | originals | degenerate |
|--:|--:|--:|--:|--:|--:|
| 0 (baseline) | 75.0% | — | 98.5% | 1.0% | 0.0% |
| 1 | **39.8%** [35, 45] | 73.8% | 95.5% | 1.0% | 0.0% |
| 2 | **28.8%** [25, 33] | 77.5% | 87.0% | 1.0% | 0.0% |
| 3 | **7.0%** [5, 10] | 75.0% | 72.5% | 0.5% | 0.0% |
| 4 | **2.5%** [1, 5] | 70.5% | 64.5% | 0.8% | 0.0% |
| 5 | **1.5%** [1, 3] | 65.2% | 59.5% | 0.5% | 0.0% |
| 6 | **2.2%** [1, 4] | 75.0% | 57.5% | 1.0% | 0.0% |
| 7 | **2.5%** [1, 5] | 70.5% | 49.0% | 1.2% | 0.0% |
| 8 | **3.2%** [2, 5] | 79.2% | 33.0% | 1.8% | 0.0% |

| reference: atlas r̂ (k=1, literature direction) | 26.5% | | 86.5% | | |

**k\*** (over-refusal below 50% of baseline **and** AdvBench above 80% of baseline **and** coherent output) = **2**

Stacking check: k=1 leaves 39.8% over-refusal, k=2 leaves 28.8%. A further drop from k=1 to k=2 is the evidence that the second direction carries over-refusal the first does not — i.e. that over-refusal is genuinely more than one-dimensional.

## Cross-attacker transfer — do RWR-derived directions suppress GCG over-refusals?

The directions are fitted **entirely on our own attacker's pairs**. The GCG corpus comes from a different method on different originals, and the two attacks are almost lexically disjoint (`exploit`: 453 of our rewrites, 1 of 1,220 GCG). If our directions still work here, they describe the model rather than our attacker.

| condition | GCG rewrites refused | GCG originals | AdvBench |
|---|--:|--:|--:|
| baseline | 82.2% | 0.0% | 98.7% |
| ablate our k=1 | **28.5%** (-53.8pp) | 0.5% | 97.3% |
| ablate our k=2 | **18.8%** (-63.5pp) | 0.2% | 86.7% |
| ablate our k=3 | **4.8%** (-77.5pp) | 0.2% | 72.0% |
| random (rank-matched) | 80.5% (-1.8pp) | | |
| atlas r̂ (literature) | 44.8% (-37.5pp) | | 87.3% |

**Caveat**: the GCG filter verifies the original was complied with but does not enforce intent preservation, so it admits rewrites our rubric excludes (e.g. "reduce plastic use" → "increase plastic use"). This measures refusal, not over-refusal under our rubric.

---
See `FINDINGS_STATUS.md` for the full ledger of standing and retracted claims.
