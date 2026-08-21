# Frame analysis — is over-refusal organised into reusable danger frames?

Layer 57. `Δ = h(rewrite) − h(original)` (cancels topic). A frame direction is the mean Δ of pairs introducing that frame's vocabulary, minus the mean Δ of all non-refused control rewrites (which removes the generic 'was rewritten' component, shared by every pair regardless of refusal).

The key test is **cross-bin**: each frame is estimated twice on disjoint pairs — from LOW (one/two-word edits) and from HIGH (wholesale paraphrases introducing the same word). A high diagonal means the frame is a real object of the model, recoverable however much of the prompt changed.


## Frame inventory

| frame | LOW pairs | HIGH pairs |
|---|--:|--:|
| exploitation | 12 | 318 |
| weaponization | 15 | 137 |
| exfiltration | 5 | 29 |
| concealment | 3 | 273 |
| intrusion | 2 | 168 |
| forgery | 0 | 1 |

Frames with ≥12 pairs in **both** bins (usable for the cross-bin test): **exploitation, weaponization**

## Split-half reliability (the noise floor)

Two disjoint halves of the SAME frame in the SAME bin. Any cross-bin cosine must be read against this: a frame cannot agree with itself across bins more than it agrees with itself within a bin.

| frame | bin | n | split-half cos (mean of 200 splits) |
|---|---|--:|--:|
| exploitation | LOW | 12 | +0.755 |
| exploitation | HIGH | 318 | +0.989 |
| weaponization | LOW | 15 | +0.850 |
| weaponization | HIGH | 137 | +0.979 |

## Cross-bin frame matrix — cos(u_f from LOW, u_g from HIGH)

Rows = frame estimated from one-word edits. Columns = frame estimated from wholesale paraphrases. **Diagonal should dominate its row and column if frames are real.**

| LOW ↓ / HIGH → | exploitation | weaponization |
|---|---|---|
| **exploitation** | **+0.830** | +0.788 |
| **weaponization** | +0.911 | **+0.946** |

mean diagonal **+0.888** vs mean off-diagonal **+0.849** (gap +0.039)

frames whose diagonal is the largest entry in its row: **2/2**

## Beyond a single shared 'danger' axis

Every frame may just be the same global over-refusal direction. Removing the shared component from each frame direction and re-running the matrix tests whether frame-specific structure survives.

| frame | cos(u_f^LOW, shared) | cos(u_f^HIGH, shared) | residual diagonal cos |
|---|--:|--:|--:|
| exploitation | +0.829 | +0.989 | **+0.120** |
| weaponization | +0.914 | +0.962 | **+0.601** |

residual: mean diagonal **+0.361** vs off-diagonal **+0.021**

## Each frame vs the global refusal direction r̂

| frame | cos(u_f^LOW, r̂) | cos(u_f^HIGH, r̂) |
|---|--:|--:|
| exploitation | +0.675 | +0.784 |
| weaponization | +0.734 | +0.778 |
| _shared OR axis_ | +0.805 | |

## Do frame-less over-refusals still live in the frame span?

Fraction of ‖Δ − μ_ctrl‖ inside the 2-dimensional frame span.

| group | n | fraction in frame span |
|---|--:|--:|
| HIGH over-refusals **with** a frame word | 838 | 0.606 |
| HIGH over-refusals **without** a frame word | 408 | 0.535 |
| HIGH matched controls (not refused) | 1246 | 0.330 |

Frame directions saved to `probe_or/results/delta_qwen/frame_directions.npz` for selective ablation (Phase 4).
