# Figure specifications

Each entry is written so that a person or model who has **not** read the paper can build the
figure faithfully. Format: the claim the figure must carry → the data → the encoding → what
must not be implied. Numbers are current as of this draft; the Qwen panels of Figures 3 and 4
are pending a rerun (flagged inline).

Shared conventions across all five figures:
- **Model identity is colour, fixed everywhere:** Llama-3-8B-Instruct = blue `#2a78d6`,
  Qwen3-32B = orange `#eb6834`. Never swap, never recycle these hues for anything else.
- Surface `#fcfcfb`; primary ink `#0b0b0b`; secondary ink `#52514e`; grid and axes recessive
  (1px, `#e6e6e3`), never competing with the data.
- Sans-serif throughout, ~9pt tick labels, ~10pt axis titles. No chart junk: no 3-D, no drop
  shadows, no gradient fills, no background tint.
- Single measurement axis per panel. Where two models appear, they are **separate panels or
  separate bars — never two y-scales.**
- Every panel states its own n in the subtitle or axis label. A reader must never have to hunt
  for the denominator.

---

## Figure 1 — The edit-distance distribution

**The claim this figure must carry.** Our attacker converged on *wholesale rewording* rather
than minimal perturbation, so near-minimal rewrites are rare in the corpus by construction. This
motivates both the two-bin split and the extra generation needed to populate the small bin. The
shape of the distribution is itself the result — this is not a descriptive warm-up figure.

**Data.** Word-level content-token Levenshtein distance $D$ computed for every confirmed
over-refusal pair, per model. Llama n = 2,580 pairs; Qwen n = 1,463 pairs. Distances are
integers ≥ 0; plot the range 0–20 and fold everything above 20 into a final ">20" bin so the
tail does not stretch the axis.

**Encoding.** Two panels stacked vertically sharing an x-axis (Llama on top), or a single panel
with two overlaid step-outlines — panels preferred, since the two models have different n and
overlaid histograms invite a false density comparison. Bars in the model colour, thin, 2px gap
between adjacent bars. x = content-word edits; y = number of pairs.

**Required annotations.**
- A vertical rule at $D = 2$ marking the low/high bin boundary, labelled "bin cut τ = 2".
- The low-bin mass called out directly: "≤2 edits: 4.2% of rewrites" with a leader line to the
  shaded region left of the cut. Shade that region lightly (5% ink) — it is the interpretable
  stratum and it is nearly empty.
- Mark the mode (6–7 edits) with a small label.
- Median normalised distance (0.92) stated in the subtitle, not as a second axis.

**What must NOT be implied.** Do not imply the distribution is a property of over-refusal in
general — it is a property of *this reward design*, which gates on semantic similarity and
imposes no lexical cost. The caption must say so.

---

## Figure 2 — Which words trip a refusal

**The claim.** The words over-represented among refused rewrites split into two kinds, and only
one kind is causal: words the **edit introduced** (weaponised, exploit, counterfeit, exfiltrate,
produce) versus **topic markers** already present in both the original and the rewrite
(database, sql, email, query…). The second kind indicates *which prompts this attacker rewrites
into refusals*, not *what the edit did*. Conflating them yields the false claim "refusal is
triggered by database queries".

**Data.** Weighted log-odds with informative Dirichlet prior, recomputed at document level
(each original counted once) — the $z_{\text{doc}}$ statistic — for the Llama low-edit bin,
filtered to words in ≥3 distinct originals. Values: `weaponized` 4.19 (16 originals), `exploit`
4.10 (16), `counterfeit` 3.24 (3), `exfiltrate` 3.20 (4), `produce` 2.48 (5); topic markers
`database`, `query`, `user`, `email`, `sql`, `computer`, `stock`, `price` in the range 2.2–3.4
(3–5 originals each).

**Encoding.** Horizontal bars, sorted descending by $z_{\text{doc}}$, all words in one column so
the interleaving of the two kinds is visible. **Colour carries the distinction and is the point
of the figure:** introduced-by-edit = blue `#2a78d6` (saturated, foreground); topic marker =
neutral grey `#8a8a85` (recessive). Bootstrap intervals over originals as thin whiskers.

**Required annotations.**
- Distinct-original count printed at the end of each bar (e.g. "16 orig."). This is essential —
  a $z$ of 3.24 backed by 3 originals is weaker evidence than 4.19 backed by 16, and the reader
  must be able to see that without consulting a table.
- A legend with exactly two entries: "introduced by the edit" / "topic marker (present in both)".
- Do **not** print the token-count $z$; it is unstable across bin cuts and is not what we report.

**What must NOT be implied.** Bars must not be read as effect sizes on refusal probability —
this is an association statistic. And no word here is *sufficient* for refusal: 1,337 rewrites
introduce these words and were not refused. The caption states this explicitly.

---

## Figure 3 — The causal result *(the paper's centrepiece)*

**The claim.** Removing a single direction from the residual stream can eliminate a large share
of over-refusal **without** reducing the model's refusal of genuinely harmful prompts — and the
literature's refusal direction cannot: it removes more over-refusal but pays for it in safety.
Both models have such a direction. Where the models differ is the overall refusal direction.

**Data.** One point per ablated direction, per model, from the symmetric-basis single-direction
scan. Axes: x = over-refusal removed (percentage points, held-out set of 400 confirmed
over-refusals); y = harmful refusal lost (percentage points, 200 AdvBench prompts).

Llama (baseline over-refusal 74.2%, harmful refusal 98.5%): weaponisation (38.2, 0.5),
concealment (22.0, −0.5), intrusion (22.0, −0.5), exfiltration (9.8, 0.0), exploitation
(5.2, 0.0), shared axis d1 (34.2, 3.0), published refusal vector (47.8, 12.5), 12 random directions
spanning x ∈ [−1.0, 8.5], y ≈ 0.

Qwen — **pending rerun on the corrected low-bin basis; do not finalise this panel until those
numbers land.** Prior values, for layout only: exfiltration (60.2, 0.0), concealment
(45.0, 21.0), weaponisation (12.8, −1.0), shared axis d1 (94.2, 95.5), published refusal vector
(51.5, 67.0), randoms x ∈ [−0.8, 2.0].

**Encoding.** Two panels side by side, one per model, **sharing both axis scales** so the
Llama/Qwen contrast is readable by position. Points ≥8px. Direct-label every non-random point —
no legend lookup, there are only ~7 per panel. Colour by direction class: frame residuals blue
`#2a78d6`, shared axis orange `#eb6834`, literature direction aqua `#1baf7a`. The 12 random
directions are small grey `#8a8a85` dots plus a shaded band spanning their range.

**Required annotations.**
- The random band must show the **full spread, not a single line** — Llama's randoms reach
  x = +8.5, which is not negligible, and a single mean line would overstate how clean the null is.
- A faint horizontal reference at y = 0 labelled "no safety cost".
- Shade or otherwise mark the desirable quadrant (large x, y ≈ 0) and label it once, e.g.
  "selective: removes over-refusal, keeps safety".
- Axis directions must be unambiguous: label x "over-refusal removed (pp) →better" and y
  "harmful refusal lost (pp) →worse". The two axes mean opposite things and this is the single
  most likely misreading.

**What must NOT be implied.** Do not connect points with lines — these are independent
single-direction ablations, not a trajectory. Do not imply the y ≈ 0 points have *proven* zero
harm cost; the claim is "no measurable cost on this set at this n", and the harm-graded
follow-up (Figure 5's companion analysis) is what supports the stronger reading.

---

## Figure 4 — What the effective direction encodes

**The claim.** The direction that does the causal work responds to **alarming wording**, not to
the refusal decision. Alarming words move a prompt along it whether or not the model went on to
refuse. The shared axis and the literature direction show the opposite signature.

**Data.** Mean projection of Δ onto each direction, in four cells crossing *introduced alarming
vocabulary* (yes/no) with *refused* (yes/no), held-out originals only, cluster-bootstrapped over
originals. Llama:

| direction | alarm+refused | alarm, not refused | plain+refused | plain, not refused |
|---|--:|--:|--:|--:|
| frame residual (weaponisation) | +0.502 | +0.556 | +0.309 | +0.369 |
| shared axis d1 | +2.948 | +0.735 | +2.504 | +0.321 |
| published refusal vector | +2.259 | +0.524 | +1.974 | +0.247 |

Cell n: 997 / 695 / 261 / 613. Main effects — frame residual: ALARM +0.190, REFUSAL −0.057;
d1: ALARM +0.429, REFUSAL +2.198; r̂: ALARM +0.281, REFUSAL +1.731. Random-direction null (50
directions), 95th percentile: ALARM 0.022, REFUSAL 0.057.

**Encoding.** Three small panels side by side, one per direction, sharing a y-axis. Within each
panel: x-axis has two positions ("no alarm words", "alarm words"); two lines connect them, one
for refused and one for not-refused pairs. Points ≥8px with bootstrap CI whiskers.

**Why this encoding and not grouped bars.** The finding is a *pattern of slopes*: for the frame
residual the two lines nearly coincide and both rise left-to-right (alarm matters, refusal does
not); for d1 and r̂ the lines are far apart and nearly flat (refusal matters, alarm does not).
That contrast is legible instantly as slopes and requires arithmetic as bars.

**Required annotations.**
- Print the two main effects inside each panel: "ALARM +0.190 / REFUSAL −0.057".
- Mark the random-null 95th percentile as a reference so the reader can see that the frame
  residual's REFUSAL effect (−0.057) sits *exactly at* the noise floor while its ALARM effect is
  8.6× it.
- Note the y-axes differ in range between the frame residual (~0.3–0.6) and d1/r̂ (~0.2–3.0). If
  a shared y-axis flattens the frame-residual panel into a line, use per-panel y-axes and say so
  explicitly in the caption — do not silently rescale.

**What must NOT be implied.** This is a population-level effect and **not** a per-prompt
predictor: projections vary widely within cells and there are inversions. The direction is also
not identical to a pure alarm direction — it correlates with a directly constructed one at
cos ≈ 0.37–0.42. The caption must not say "this direction is the alarm feature".

---

## Figure 5 — Does it generalise beyond our own attacker?

**The claim.** The direction was fitted entirely on our attacker's rewrites, yet removing it
reduces over-refusal on corpora built by three other methods, including a different team's
attack and two external benchmarks. Random directions do nothing on any of them.

**Data.** Refusal rate before and after ablating the frame-residual direction, plus a
rank-matched random control, Llama:

| corpus | how built | baseline | after ablation | random |
|---|---|--:|--:|--:|
| our rewrites | RWR attacker | 74.2% | 41.0% | 74.0% |
| GCG corpus | different attack method, near-disjoint vocabulary | 82.2% | 28.5% | 80.5% |
| XSTest safe (250) | hand-written | 7.6% | 2.8% | 7.6% |
| OR-Bench Hard (400) | auto-generated from toxic seeds | 77.8% | 59.8% | 78.5% |

**Encoding.** Four rows, one per corpus, ordered by how independent they are from us (ours
first, then GCG, then the two external benchmarks). Per row: a dumbbell — baseline dot and
after-ablation dot joined by a line, with an arrowhead pointing at the after value. Random
control as a hollow marker at its own value. x = refusal rate (%), 0–100.

**Why a dumbbell and not grouped bars.** The quantity of interest is the *drop*, and a dumbbell
encodes the drop as length while keeping both absolute rates readable. Grouped bars force the
reader to subtract, and the four corpora have wildly different baselines (7.6% to 82.2%) which
bars handle badly.

**Required annotations.**
- The delta printed at the end of each row ("−33.2 pp").
- The construction method printed under each corpus name — the whole force of this figure is
  that the corpora are independently built, and a reader who does not know that sees four
  arbitrary datasets.
- The XSTest row will look small in absolute terms (7.6 → 2.8) but is a **63% relative
  reduction**; print the relative change for that row so it is not dismissed as noise.

**What must NOT be implied.** Do not aggregate the four corpora into an average — they measure
different populations with different baselines. Do not imply the GCG corpus is judged under our
over-refusal rubric; it is filtered differently, so its rates measure refusal, not confirmed
over-refusal. Say so in the caption.
