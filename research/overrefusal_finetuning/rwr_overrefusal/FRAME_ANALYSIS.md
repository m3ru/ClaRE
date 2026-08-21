# Frame analysis — is over-refusal organised into reusable danger frames?

Layer 17. `Δ = h(rewrite) − h(original)` (cancels topic). A frame direction is the mean Δ of pairs introducing that frame's vocabulary, minus the mean Δ of all non-refused control rewrites (which removes the generic 'was rewritten' component, shared by every pair regardless of refusal).

The key test is **cross-bin**: each frame is estimated twice on disjoint pairs — from LOW (one/two-word edits) and from HIGH (wholesale paraphrases introducing the same word). A high diagonal means the frame is a real object of the model, recoverable however much of the prompt changed.


## Frame inventory

| frame | LOW pairs | HIGH pairs |
|---|--:|--:|
| exploitation | 78 | 778 |
| weaponization | 38 | 217 |
| exfiltration | 15 | 73 |
| concealment | 20 | 393 |
| intrusion | 8 | 130 |
| forgery | 10 | 7 |

Frames with ≥12 pairs in **both** bins (usable for the cross-bin test): **exploitation, weaponization, exfiltration, concealment**

## Split-half reliability (the noise floor)

Two disjoint halves of the SAME frame in the SAME bin. Any cross-bin cosine must be read against this: a frame cannot agree with itself across bins more than it agrees with itself within a bin.

| frame | bin | n | split-half cos (mean of 200 splits) |
|---|---|--:|--:|
| exploitation | LOW | 78 | +0.949 |
| exploitation | HIGH | 778 | +0.993 |
| weaponization | LOW | 38 | +0.836 |
| weaponization | HIGH | 217 | +0.974 |
| exfiltration | LOW | 15 | +0.722 |
| exfiltration | HIGH | 73 | +0.923 |
| concealment | LOW | 20 | +0.756 |
| concealment | HIGH | 393 | +0.985 |

## Cross-bin frame matrix — cos(u_f from LOW, u_g from HIGH)

Rows = frame estimated from one-word edits. Columns = frame estimated from wholesale paraphrases. **Diagonal should dominate its row and column if frames are real.**

| LOW ↓ / HIGH → | exploitation | weaponization | exfiltration | concealment |
|---|---|---|---|---|
| **exploitation** | **+0.899** | +0.868 | +0.751 | +0.809 |
| **weaponization** | +0.833 | **+0.873** | +0.767 | +0.784 |
| **exfiltration** | +0.693 | +0.619 | **+0.855** | +0.626 |
| **concealment** | +0.735 | +0.622 | +0.747 | **+0.757** |

mean diagonal **+0.846** vs mean off-diagonal **+0.738** (gap +0.108)

frames whose diagonal is the largest entry in its row: **4/4**

## Beyond a single shared 'danger' axis

Every frame may just be the same global over-refusal direction. Removing the shared component from each frame direction and re-running the matrix tests whether frame-specific structure survives.

| frame | cos(u_f^LOW, shared) | cos(u_f^HIGH, shared) | residual diagonal cos |
|---|--:|--:|--:|
| exploitation | +0.882 | +0.991 | **+0.397** |
| weaponization | +0.842 | +0.923 | **+0.459** |
| exfiltration | +0.698 | +0.837 | **+0.691** |
| concealment | +0.770 | +0.951 | **+0.128** |

residual: mean diagonal **+0.419** vs off-diagonal **-0.037**

## Each frame vs the global refusal direction r̂

| frame | cos(u_f^LOW, r̂) | cos(u_f^HIGH, r̂) |
|---|--:|--:|
| exploitation | +0.663 | +0.763 |
| weaponization | +0.624 | +0.706 |
| exfiltration | +0.499 | +0.630 |
| concealment | +0.631 | +0.748 |
| _shared OR axis_ | +0.780 | |

## Do frame-less over-refusals still live in the frame span?

Fraction of ‖Δ − μ_ctrl‖ inside the 4-dimensional frame span.

| group | n | fraction in frame span |
|---|--:|--:|
| HIGH over-refusals **with** a frame word | 1455 | 0.539 |
| HIGH over-refusals **without** a frame word | 917 | 0.478 |
| HIGH matched controls (not refused) | 2372 | 0.262 |

Frame directions saved to `probe_or/results/delta/frame_directions.npz` for selective ablation (Phase 4).

---

## Causal test outcome — the frame hypothesis FAILED

The correlational structure above (cross-bin diagonal +0.42, off-diagonal −0.04, replicating
at the split-half noise ceiling) predicted that ablating a frame's residual direction would
suppress that frame's over-refusals more than the other frames'. It does not.

| eval size | diagonals leading their row |
|---|---|
| n=120 per frame | 4/4 (sign test p=0.0039) |
| **max available n** (400/176/55/313) | **2/4** |

The 4/4 result did not replicate; at n=120 every margin was already inside noise (0.07–0.62
SE), and the apparent significance was a small-sample artifact. **Frames are correlationally
real but are not a causal handle**, and the p=0.0039 should not be quoted.

What replicated instead (max n, all cells 0.00% degenerate, random control flat):

| ablated direction | over-refusal | AdvBench | ratio |
|---|--:|--:|--:|
| second general direction (found via weaponization pairs) | −61.3pp | 98.5→97.5% (−1.0) | 61× |
| shared over-refusal axis | −34.0pp | 98.5→96.0% (−2.5) | 14× |
| atlas refusal direction r̂ | −50.7pp | 98.7→87.3% (−11.3) | 4.5× |

Two directions, orthogonal by construction, each removing more over-refusal than the known
refusal direction at a fraction of the safety cost. NOTE: the second direction is NOT
weaponization-specific — it suppresses all four frames roughly equally (52.2/71.6/69.1/52.4).
It was found by looking at weaponization pairs; naming it after them would mislead.

One place frame structure does show causally: the shared axis is uniform on three frames
(−41.0, −38.1, −38.2) but markedly weaker on concealment (−18.8 ± 2.8).
