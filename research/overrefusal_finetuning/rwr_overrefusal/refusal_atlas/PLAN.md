# Refusal Atlas — what Llama-3-8B & Qwen3-32B (over-)refuse, and whether the picture changes by signal

Study goal: map the **over-refusal boundary** of two models and test whether that
boundary *looks different depending on which internal refusal signal you view it
through*. Concretely: which **topics** and which **words/tokens** push a benign
prompt over into refusal, per model, and does that ranking shift across the three
signals (refusal vector / probe ensemble / logit).

Locked design decisions (2026-08-02, extended 2026-08-10):
- **Prompts:** OR-Bench-Hard-1K (discriminating set) + XSTest anchor + OR-Bench-Toxic
  (should-refuse ceiling) + **freshly generated** leakage-free minimal pairs. All staged.
- **Measure:** BOTH — internal signal scores AND behavioral sampling (generate +
  classify) as ground truth for over-refusal.
- **Topics:** HYBRID — OR-Bench 10-category taxonomy + unsupervised clustering. Clustering
  method = FULL peer-reviewed redesign (see P5), not the original HDBSCAN plan.
- **Signals:** all three, BOTH models. Built the per-model multi-phrase logit (openers
  mined from real generations → `opener_sets.json`). Per-prompt signal = **absolute**
  projection (Arditi-style), since most substrate prompts are singletons; minimal pairs
  additionally get the **delta** (rw−orig) form for the word analysis.
- **Leakage control (2026-08-10):** minimal pairs must be disjoint from all direction/
  probe fitting sets. The existing pool can't supply 50 clean pairs (fitting consumed
  5,315/5,998 originals → only ~14 clean), so we GENERATE ~60 fresh pairs from untouched
  benign originals (also removes model-contamination). Directions are fit on an
  independent refuse/benign split, so absolute projections carry no leakage regardless.

STATUS (2026-08-10): benchmarks staged; opener sets mined; substrate built (pairs slice
pending fresh generation); `score_signals.py` written + under fable code review; fresh-pair
generation blocked on ANTHROPIC_API_KEY. Analysis packages → separate venv (not ~general).

## The three signals (both models) — LOCKED (2026-08-10)
| signal | Llama | Qwen | how |
|---|---|---|---|
| **vector** = raw absolute proj on the CAUSAL diff-of-means direction | **L17** | **L58** | both causally validated (ablation+addition) |
| **probe** = absolute mass-mean ensemble, broad-behavioral label, L0 excluded | ~L31 (L29/31/32) | ≡ L58 (collapses) | `fit_probe_absolute.py` → `probe_absolute.npz` |
| **logit** = teacher-forced Σ P(response begins with opener) | 5-opener set | 5-opener set | `opener_sets.json` (mined) |

### Signal-side findings (locked)
- **Vector layers are causally validated.** Llama L17 (<collaborator>'s ablation: ablate→refusal
  99→83%, add→0→99%; L32 does neither). **Qwen L58**: clean sufficiency is decisive —
  adding L58 to benign prompts drives refusal **5%→93%** @coef2 on DISJOINT harmless
  prompts (`--test_offset 2000`), far above L32/L45/L63; first-run necessity (ablate L58
  63→36%) was directionally consistent though circular. Clean-run necessity was
  data-limited (disjoint harmful slice = only 2% Qwen baseline; the file is harmful-first,
  so Qwen-refused prompts sit in the fit rows). L64 discounted (post-final-RMSNorm).
  Optional textbook-clean necessity: stage AdvBench (not yet done).
- **The multi-layer probe beats a single layer for NEITHER model** (Llama ensemble −0.0005,
  Qwen −0.0000 over best single layer, absolute space, broad labels). → a single refusal
  direction suffices; stacking adds nothing (bears on single-direction vs Concept-Cones).
- **Cross-model layer geometry differs:** Llama causal L17 ≠ behavioral-best L31 (diverge);
  Qwen causal L58 ≈ behavioral-best L57/58 (coincide). → for Llama the 3 signals are
  genuinely distinct; for Qwen **probe ≡ vector** (both L58) — reported transparently.
- **Logit = multi-phrase, per model.** Only 6.5% of Qwen refusals and 56.5% of Llama's
  literally start "I cannot" (Llama's "I apologize" = +40.6%); each model uses its own
  5-opener set (`opener_sets.json`), ~99.7% coverage.
- **Bonus behavioral obs:** Qwen refuses **21%** of the "benign" split vs Llama's 0% —
  markedly more trigger-happy on borderline-benign prompts.

### Caveats logged (fable reviews)
- Probe `ensemble_auc` is optimistic (duplicate originals leak across random CV folds); the
  APPLIED probe (full-data NNLS) is valid and the "adds nothing" verdict holds a fortiori
  (inflated yet still ≤ single layer). Fix if reported: group CV folds by unique original.
- Qwen probe fit label is ~"is-a-refused-rewrite" (538/543 positives are rewrites) — note.

## Prompt substrate (`data/substrate.csv`, built, seed 42)
Schema: `prompt_id, text, source, gold_benign, native_topic, pair_id, is_rewrite,
probe_delta, similarity`. ~650 prompts (pairs slice regenerating clean):

| source | n | role |
|---|--:|---|
| orbench_hard | 300 (30×10 cat) | over-refusal core + native 10-cat taxonomy |
| xstest | 200 (150 safe / 50 unsafe) | clean topic types + safe/unsafe contrast (anchor) |
| sonnet_pair | ~60 fresh pairs | word-level carriers (original→edited), leakage-free |
| orbench_toxic | 50 | should-refuse anchor / calibration ceiling |

Staged offline in `data/`: xstest.csv (450), or-bench-hard-1k.csv (1319),
or-bench-toxic.csv (655). `build_substrate.py` assembles the table; the pair-selection
block is being replaced by the fresh-generation output.

## Opener sets (`opener_sets.json`, mined from real generations)
Per-model refusal openers, each covering ~99.7% of that model's refusals:
- **Qwen** {I'm sorry, Sorry, I can't, I cannot, As an AI} — only 6.5% start "I cannot".
- **Llama** {I cannot, I apologize, I can't, Sorry, I'm sorry} — "I cannot" 56.5%,
  "I apologize" +40.6% (the single-phrase logit_or signal missed the latter).
logit signal = Σ teacher-forced P(response begins with opener_i) over the set (openers are
token-prefix-disjoint → sum ≈ P(begins with any opener)); logit_max = max_i.

## Experiments (cheap-first)

**P1 — Signals over a common set (both models × 3 signals).**
Re-score a common sampled subset with all three signals per model. Reuse
`extract_layer_acts.py` (acts, float32), the fitted directions/probes, and the logit
readers. Build + calibrate the Qwen multi-phrase logit here.

**P2 — Behavioral ground truth.**
k samples/prompt/model → broad opener classifier → `refuse_rate` per (model, prompt).
Reuse the eval harness + classifier. This defines actual over-refusal.

**P3 — Signal↔behavior fidelity (per model, per signal).**
AUC + calibration of each signal predicting behavioral refusal → *which signal is the
truest view of the boundary*, and where each systematically over/under-predicts
(e.g. the L32 vector Goodhart pattern).

**P4 — Signal↔signal agreement (per model).**
Correlation matrix among the 3 signals + disagreement cases (prompts one flags, another
doesn't). Starting point: `probe_or/probe_disagreement.py`. → *do the boundaries change
between signals.*

**P5 — Topic map (hybrid; FULL peer-reviewed clustering redesign, see reading_list).**
Fixed axis: OR-Bench 10-category classifier. Emergent axis (clustering):
- **Embed** with BGE-large-en-v1.5 (SIGIR 2024; MiniLM as replication check), L2-norm.
- **Cluster** deterministically — k-means / Ward, OVER-clustered k≈25–40, k by bootstrap
  stability. HDBSCAN demoted to a sensitivity check (KONVENS 2023: worst at this N).
- **Exclude the minimal-pair rewrites** from clustering (near-duplicates inflate density
  & bias cluster OR-rate); assign each to its original's cluster post-hoc; route the pair
  contrast to P6. Dedup-scan remaining prompts at cosine>0.9.
- **Label** clusters with an LLM (TopicGPT-style) + c-TF-IDF keywords; intruder-test.
- **Cluster ONCE on prompt embeddings (model-independent)**, then overlay each model's &
  signal's rates on the fixed partition (never recluster per model/signal).
- **Validate:** AMI + ARI vs OR-Bench-10 (not raw NMI); bootstrap Jaccard ≥0.75 to report
  a cluster; per-cluster OR rate with Wilson CIs + binomial test + BH-FDR; **source-
  composition control** (OR-Bench-Hard is selected-high-refusal → test hotness within-source).
- **Results:** behavioral OR rate by topic × model (heatmap), cross-model diffs; per signal,
  does it *reorder* the topics vs behavior (rank-corr) → topic-level "boundary by signal".

**P6 — Word / trigger map.**
- Minimal-pair deltas: from original→rewrite edits, regress refusal-Δ (behavioral and
  per-signal) on added/changed words → top trigger words per model.
- Token attribution: per-token projection on the vector; per-token contribution to the
  logit — do the 3 signals localize to the *same* tokens?
- Cross-model & cross-signal trigger-word comparison.

**P7 — Synthesis brief + figures.**
Facts-only brief (examples included) + figures: topic×model heatmap, signal-agreement
matrix, trigger-word tables, "signal reorders topics" plot. Feeds a LessWrong framing.

## Reuse vs new
- **Reuse:** `extract_layer_acts.py`, fitted directions/probes (`probe_or/results/`),
  Llama logit reader (`logprob_or/`), broad refusal classifier, eval harness, scored
  Sonnet pool, MiniLM embeddings.
- **New:** Qwen multi-phrase logit; benchmark ingestion (offline); topic
  classifier+clustering; word/token attribution; cross-model/-signal analysis + figures.

## Open items — RESOLVED
1. **Benchmark:** OR-Bench-Hard-1K (discriminating) + XSTest anchor + OR-Bench-Toxic. ✓ staged
2. **Behavioral budget:** ~650 prompts × 4 samples/model (a few GPU-hrs on the 32B). ✓
3. **Taxonomy:** OR-Bench 10 categories. ✓ (native to the data + comparable to the literature)
4. **Leakage:** exclude-overlap → fresh-generate ~60 clean pairs. ✓ decided (blocked on API key)
5. **Analysis venv:** separate from pinned ~general. ✓ decided
6. **Clustering:** full peer-reviewed redesign. ✓ decided (see P5)

## Remaining to build
- Fresh-pair generation (needs ANTHROPIC_API_KEY; ~60 originals, Sonnet batch API, <$1).
- `score_signals.py` SLURM wrappers (2, one per model); needs PYTHONPATH=probe_or for the
  `probe_ensemble` / `gen_qwen_refusal` imports.
- Separate analysis venv + P5 clustering code + P3/P4/P6/P7 analysis + figures.
- Apply fable code-review fixes to `score_signals.py` before the GPU run.

## Related work / reading — see `reading_list.md` (full annotated list)
The sweep **confirms our novelty**: nobody has put all three signals (diff-of-means
direction + layer probe ensemble + refusal-onset logprob) side-by-side on the *same* benign
boundary, at both topic and word granularity, contrasting **Llama-3-8B vs Qwen3-32B**. The
pieces are siloed — and, tellingly, the closest attempts at the cross-cut are all *unreviewed
preprints*, so our novelty against peer-reviewed work is even safer.

Curated field resource: **Awesome-Over-Refusal**
(https://github.com/abbottyanginchina/Awesome-Over-Refusal) — organized by *benchmarks* +
*mitigation*, NOT by internal-signal characterization (no probing/refusal-direction section;
omits Arditi/Concept Cones as it is over-refusal-specific). Its very structure shows the
field catalogues "measure it / reduce it," not "view the boundary through different signals."

**Peer-reviewed backbone — anchor claims on these:**
- **Arditi et al., NeurIPS 2024** — single refusal direction; Signal (1) foundation.
- **Concept Cones — Wollschläger et al., ICML 2025** (PMLR v267, pp 66945–66970; verified) —
  multiple independent refusal directions; "orthogonality ≠ independence, incl. NON-LINEAR
  effects." Peer-reviewed basis for the "linear signals may miss structure" caveat; motivates
  the probe ensemble and a possible non-linear 4th diagnostic.
- **Pan et al., EMNLP 2025** "…Safety Decision Boundary" (2025.emnlp-main.1065 / arXiv
  2505.18325; verified) — over-refusal decision boundary via steering vectors in
  representation space (RASS). Our peer-reviewed *boundary* anchor.
- **"Surgical, Cheap, and Flexible: Mitigating False Refusal via Single Vector Ablation",
  ICLR 2025** — peer-reviewed causal use of a refusal vector on false refusals.
- Benchmarks/taxonomies: **OR-Bench (ICML 2025)**, **SORRY-Bench (ICLR 2025)**, **XSTest
  (NAACL 2024)**, **OKTest/OverKill (ACL 2024 Findings)** — argues over-refusal = lower-layer
  shortcut-keyword detection (word-trigger foundation), **PHTest (COLM 2024)**, **FalseReject
  (COLM 2025)**. **Refusal–Compliance Tradeoff** (audit: Llama conservative vs Qwen balanced)
  — check its venue before citing.

**Preprints — SUGGESTIVE ONLY, verify methodology before trusting (do NOT cite as
established):** these happen to be the closest to our exact study.
- 2602.02132 "There Is More to Refusal than a Single Direction" — topic × direction; geometry
  differs by topic while behavior collapses. (unreviewed)
- 2512.01037 "When Safety Blocks Sense" — word × model via logit signal; boundary differs by
  model (Llama unstable, Qwen consistent). (unreviewed)
- 2511.19009 MOSR — over-refusal = intermediate rep where probe vs logit disagree. (unreviewed)
- 2604.18901 — linear probe intact in abliterated models → signal ≠ behavior. (unreviewed)
- 2509.09708 "Beyond I'm Sorry" — FM vs linear (330 vs 101) → linear misses non-linear
  structure. (unreviewed; but Concept Cones ICML 2025 makes the peer-reviewed version of this
  point.)
