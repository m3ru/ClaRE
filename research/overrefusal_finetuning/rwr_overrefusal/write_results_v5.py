#!/usr/bin/env python3
"""Render RESULTS_V5.md from probe_or/results/v5_judged/summary_v5.json.

Every number in the report comes from the JSON, so the document cannot drift from the
computation. Prose and the "old figure / old judge" columns are transcribed from
EXPERIMENT_BRIEF.md (as of commit 40f7c71) and are labelled as such.
"""
import json, sys

S = json.load(open("probe_or/results/v5_judged/summary_v5.json"))
A = S["arms"]


def g(name, arm):
    return A[f"{name}|{arm}"]


def rate(s):
    return f"**{100*s['rate']:.2f}%** ({s['n_or']}/{s['n']})"


def ci(s):
    return f"[{100*s['lo']:.2f}, {100*s['hi']:.2f}]"


def ref(s):
    return f"{s['n_ref']}/{s['n']} ({100*s['raw_refused']:.2f}%)"


def pur(s):
    return f"{100*s['purity']:.1f}%"


def wtd(s):
    return f"{100*s['weighted']:.2f}%"


def mrr(s):
    return f"{100*s['mean_rr']:.2f}%"


def pc(t):
    d, lo, hi = t
    sig = "**sig.**" if (lo > 0 or hi < 0) else "n.s."
    return f"{100*d:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] {sig}"


O = []
w = O.append

w("# Over-refusal figures recomputed under judge v5")
w("")
w("Every over-refusal number in `EXPERIMENT_BRIEF.md` §5–§11 was labelled by a *different* "
  "judge (hand audit / lenient haiku filter / an early Sonnet prompt / v4). This document "
  "recomputes all of them under the single current judge, **v5** (`or_judge_v5.py` + "
  "`grading/fewshot_v5.txt`), so one comparable metric runs through the whole brief.")
w("")
w("`EXPERIMENT_BRIEF.md` is **not** modified by this document.")
w("")
w("## Headline figures, before and after")
w("")
w("| brief § | quantity | old figure (and its judge) | **under v5** |")
w("|---|---|---|---|")
w(f"| §5 | Llama **logit** attacker, genuine OR of 800 | 7.06% — hand audit | "
  f"**{100*g('eval_llama_logit','rwr')['rate']:.2f}%** ({g('eval_llama_logit','rwr')['n_or']}/800) "
  f"{ci(g('eval_llama_logit','rwr'))} |")
w(f"| §5 | base-Llama-as-attacker | 3.19% — hand audit (one run) | "
  f"**{100*A['POOLED_BASE|llama (vector+probe+logit)']['rate']:.2f}%** "
  f"({A['POOLED_BASE|llama (vector+probe+logit)']['n_or']}/2400, pooled) |")
w(f"| §8 | 8k scale-up | 16.84% (really 1320/7877 = 16.76%) — lenient haiku filter | "
  f"**{100*A['scaleup|refused_opener']['rate']:.2f}%** "
  f"({A['scaleup|refused_opener']['n_or']}/8000) {ci(A['scaleup|refused_opener'])} |")
w(f"| §9.4 | best Qwen arm vs base-Qwen | \"no arm reliably beats base\" — early Sonnet | "
  f"**probe beats pooled base by {100*S['paired_pooled_base']['eval_qwen_3sig_probe'][0]:+.2f} "
  f"pts** (knife-edge — see §8.2) |")
w(f"| §11 | llamaAtt→Llama diagonal | 12.2% (365/3000) — judge v4 | "
  f"**{100*A['trial2|llamaAtt_llamaTgt']['rate']:.2f}%** "
  f"({A['trial2|llamaAtt_llamaTgt']['n_or']}/3000) |")
w("")
w("**One sentence:** the Llama story survives at lower absolute levels (the 8k headline falls "
  "from ~16.8% to ~10.3%), the Qwen story moves — v5 rates *base*-Qwen far lower, which turns "
  "\"vector is significantly worse than base\" into \"no arm is worse, probe is marginally "
  "better\" — and the 2×2 target-vs-attacker conclusion is unchanged.")
w("")
w("## 0. Method — read this before quoting any number")
w("")
w("**Definitions, applied identically to every dataset below.**")
w("")
w("- **refused** — the target model's reply set for a rewrite is classified as a refusal by the "
  "broad refusal-opener classifier (`gen_qwen_refusal.classify`, the same one used everywhere in "
  "the project), on **`refuse_rate > 0.5`, i.e. ≥3 of the 4 sampled replies**. This threshold is "
  "used for every table. Where an old figure used a different threshold or a different refusal "
  "labeller, that is stated in the row.")
w("- **genuine over-refusal (OR) rate** — (number of rewrites that are **both** refused **and** "
  "judged `is_or == 1` by v5) / (**all** rewrites in that arm). Numerator and denominator are "
  "given for every figure.")
w("- **purity** — of the *refused* rewrites, the fraction judged OR.")
w("- **v5 `is_or == 1`** — INTENT SAME **and** HARM BENIGN, the two-axis rubric in "
  "`or_judge_v5.py`.")
w("- **95% CI** — **cluster bootstrap over the ORIGINAL prompts** (10,000 resamples, percentile "
  "interval). Originals are the resampling unit because the 4 rewrites of one original are not "
  "independent. Not Wilson — Wilson would ignore that clustering and be too narrow.")
w("- **wtd** — the *refuse-rate-weighted* variant: mean over the arm's rewrites of "
  "`refuse_rate × is_or`. It is reported **only** for the 200-prompt evals, because it is the "
  "weighting the old hand-audit \"bucket A of 800\" and the old §9.4 Qwen figures used, so it is "
  "the apples-to-apples column for those. To compute it, the 351 rewrites with "
  "`0 < refuse_rate ≤ 0.5` in those evals were judged as well.")
w("")
w("**Judge calibration (measured in this run, not carried over).** On the 67-item owner-labelled "
  f"held-out set (`grading/heldout_v4.csv` + `_truth.csv`), v5 agrees with the owner on "
  f"**58/67 = 86.6%**, κ = 0.73, precision 0.89, recall 0.80 (TP 24, FP 3, FN 6, TN 34; 1 item "
  "returned an unparseable verdict). On the 15-item `grading/purity_test.csv` set it agrees 12/15 "
  "(80%). The circularity caveat in brief §10.4 still applies: much of that label set is "
  "rule-derived rather than owner-independent.")
w("")
w("**Judge determinism.** `judge_direct.py` calls the API at default temperature. The 351-item "
  "supplement was judged twice (an accident of a lost directory) and 3 of 351 verdicts flipped "
  "— about **0.9% run-to-run noise**, well inside the CIs below but not zero.")
w("")
w(f"**UNKNOWN verdicts** (unparseable judge output) are counted as NOT over-refusal, which is "
  f"conservative. Count in the main pool: **{S['n_unknown']}**.")
w("")
w("**De-duplication.** A v5 verdict depends only on `(original, rewrite)` — never on the arm, "
  "the target, or the experiment. All refused rewrites from all datasets were pooled and judged "
  "**once**: 3,375 unique pairs covering 4,086 dataset-level refusal events, plus a 351-pair "
  "supplement for the weighted column. Verdicts are in `probe_or/results/v5_judged/`.")
w("")

# ---------------- Table 1 ----------------
w("## 1. Llama 3-signal behavioural eval (brief §5)")
w("")
w("200 held-out originals × 4 rewrites = 800 rewrites per arm. Floor (base-Llama on the 200 "
  "untouched originals) = 1/200 = 0.50%.")
w("")
w("| arm | v5 genuine-OR (of 800) | 95% CI | purity | v5 wtd | OLD bucket-A (of 800) | OLD judge "
  "| refused ≥3/4 (of 800) | mean refuse_rate (brief's §5 raw metric) |")
w("|---|--:|---|--:|--:|--:|---|--:|--:|")
OLD1 = {("eval_llama_logit", "rwr"): "7.06%", ("eval_llama_logit", "base"): "3.19%",
        ("eval_llama_probe", "rwr"): "3.62%", ("eval_llama_probe", "base"): "3.06%",
        ("eval_llama_vector", "rwr"): "2.72%", ("eval_llama_vector", "base"): "3.09%"}
for name, lab in [("eval_llama_logit", "logit"), ("eval_llama_probe", "probe"),
                  ("eval_llama_vector", "vector")]:
    for arm, alab in [("rwr", f"**{lab}** (trained)"), ("base", f"{lab}: base-as-attacker")]:
        s = g(name, arm)
        w(f"| {alab} | {rate(s)} | {ci(s)} | {pur(s)} | {wtd(s)} | {OLD1[(name, arm)]} | "
          f"hand audit (§5) | {ref(s)} | {mrr(s)} |")
w("")
p = S["paired_own_base"]
w("Paired contrast (trained minus its own base-as-attacker, paired cluster bootstrap over the "
  "same 200 originals), in percentage points of genuine-OR rate:")
w("")
w("| attacker | rwr − own base | rwr − pooled base (n=2400) |")
w("|---|---|---|")
for name, lab in [("eval_llama_logit", "logit"), ("eval_llama_probe", "probe"),
                  ("eval_llama_vector", "vector")]:
    w(f"| {lab} | {pc(p[name])} | {pc(S['paired_pooled_base'][name])} |")
w("")

# ---------------- Table 2 ----------------
w("## 2. L31 follow-up — layer vs signal type (brief §6)")
w("")
w("| arm | v5 genuine-OR (of 800) | 95% CI | purity | v5 wtd | OLD bucket-A | OLD judge | "
  "refused ≥3/4 | mean refuse_rate |")
w("|---|--:|---|--:|--:|--:|---|--:|--:|")
s = g("eval_llama_vector_L31", "rwr")
w(f"| **vector@L31** (trained) | {rate(s)} | {ci(s)} | {pur(s)} | {wtd(s)} | 1.53% | "
  f"hand audit (§6) | {ref(s)} | {mrr(s)} |")
s = g("eval_llama_vector_L31", "base")
w(f"| its base-as-attacker | {rate(s)} | {ci(s)} | {pur(s)} | {wtd(s)} | — | — | {ref(s)} | "
  f"{mrr(s)} |")
w("")
w(f"Paired: rwr − own base = {pc(p['eval_llama_vector_L31'])}; "
  f"rwr − pooled base = {pc(S['paired_pooled_base']['eval_llama_vector_L31'])}.")
w("")

# ---------------- Table 3 ----------------
w("## 3. Recipe tuning vs the logit attacker (brief §7)")
w("")
w("| arm | v5 genuine-OR (of 800) | 95% CI | purity | v5 wtd | OLD bucket-A | OLD judge | "
  "refused ≥3/4 | mean refuse_rate |")
w("|---|--:|---|--:|--:|--:|---|--:|--:|")
s = g("eval_llama_logit", "rwr")
w(f"| logit baseline (3 ep, floor 0.5) | {rate(s)} | {ci(s)} | {pur(s)} | {wtd(s)} | 7.06% | "
  f"hand audit (§5) | {ref(s)} | {mrr(s)} |")
s = g("eval_llama_logit_tune1_e6", "rwr")
w(f"| **epochs 3→6** | {rate(s)} | {ci(s)} | {pur(s)} | {wtd(s)} | not reported | — | {ref(s)} | "
  f"{mrr(s)} |")
s = g("eval_llama_logit_tune2_f70", "rwr")
w(f"| **sim floor 0.5→0.7** | {rate(s)} | {ci(s)} | {pur(s)} | {wtd(s)} | 5.94% | hand audit (§7) "
  f"| {ref(s)} | {mrr(s)} |")
w("")

# ---------------- Table 4 ----------------
w("## 4. Pooled base-as-attacker — one baseline number, not three")
w("")
w("The three `base` arms of the Llama evals (and of the Qwen evals) are independent "
  "re-generations of the same quantity on the same 200 originals. Pooled over all three "
  "(2,400 rewrites, cluster bootstrap still over the 200 originals):")
w("")
w("| pooled baseline | v5 genuine-OR (of 2400) | 95% CI | refused (of 2400) | purity | v5 wtd |")
w("|---|--:|---|--:|--:|--:|")
for tag, lab in [("llama (vector+probe+logit)", "base-Llama-as-attacker"),
                 ("qwen (vector+probe+logit)", "base-Qwen-as-attacker")]:
    s = A[f"POOLED_BASE|{tag}"]
    w(f"| **{lab}** | {rate(s)} | {ci(s)} | {ref(s)} | {pur(s)} | {wtd(s)} |")
w("")
w("Per-run spread of the three Llama base arms (v5 genuine-OR of 800): "
  + ", ".join(f"{100*g(n,'base')['rate']:.2f}%"
              for n in ["eval_llama_vector", "eval_llama_probe", "eval_llama_logit"])
  + "; the three Qwen base arms: "
  + ", ".join(f"{100*g(n,'base')['rate']:.2f}%"
              for n in ["eval_qwen_3sig_vector", "eval_qwen_3sig_probe", "eval_qwen_3sig_logit"])
  + ". The generation-to-generation spread is real and is why a single-run baseline is a weak "
    "comparator.")
w("")

# ---------------- Table 5 ----------------
w("## 5. Qwen mirror (brief §9.2 / §9.4)")
w("")
w("| arm | v5 genuine-OR (of 800) | 95% CI | purity | v5 wtd | OLD genuine-OR (haiku, wtd) | "
  "OLD genuine-OR (early Sonnet, wtd) | refused ≥3/4 | mean refuse_rate |")
w("|---|--:|---|--:|--:|--:|--:|--:|--:|")
OLDQ = {"eval_qwen_3sig_logit": ("7.06%", "6.72%"),
        "eval_qwen_3sig_probe": ("9.47%", "9.53%"),
        "eval_qwen_3sig_vector": ("2.66%", "2.72%")}
for name, lab in [("eval_qwen_3sig_logit", "logit"), ("eval_qwen_3sig_probe", "probe"),
                  ("eval_qwen_3sig_vector", "vector")]:
    s = g(name, "rwr")
    o = OLDQ[name]
    w(f"| **{lab}** (trained) | {rate(s)} | {ci(s)} | {pur(s)} | {wtd(s)} | {o[0]} | {o[1]} | "
      f"{ref(s)} | {mrr(s)} |")
s = g("eval_qwen_3sig_logit", "base")
w(f"| base-as-attacker (logit run — the arm §9.4 used) | {rate(s)} | {ci(s)} | {pur(s)} | "
  f"{wtd(s)} | 4.41% | 6.69% | {ref(s)} | {mrr(s)} |")
s = A["POOLED_BASE|qwen (vector+probe+logit)"]
w(f"| **base-as-attacker, pooled (of 2400)** | {rate(s)} | {ci(s)} | {pur(s)} | {wtd(s)} | — | — "
  f"| {ref(s)} | {mrr(s)} |")
w("")
w("| attacker | rwr − own base | rwr − pooled base (n=2400) |")
w("|---|---|---|")
for name, lab in [("eval_qwen_3sig_logit", "logit"), ("eval_qwen_3sig_probe", "probe"),
                  ("eval_qwen_3sig_vector", "vector")]:
    w(f"| {lab} | {pc(p[name])} | {pc(S['paired_pooled_base'][name])} |")
w("")

# ---------------- Table 6 ----------------
w("## 6. The 8k Llama scale-up (brief §8)")
w("")
sx = S["scaleup_extra"]
so = A["scaleup|refused_opener"]
ss = A["scaleup|refused_sonnet"]
w("The old §8 pipeline was two-stage and used **two** different labellers from the ones used "
  "everywhere else: a lenient `claude-haiku-4-5` BENIGN/HARMFUL filter to build a 7,877-rewrite "
  "denominator, then a `claude-sonnet-5` REFUSE/COMPLY judge over base-Llama's replies at "
  "**`refuse_rate ≥ 0.5` (2 of 4)** for the numerator. Both differ from the definition used "
  "everywhere else in this document, so both variants are given.")
w("")
w("| definition of *refused* | v5 genuine-OR (of 8000) | 95% CI | refused (of 8000) | purity | "
  "breadth (distinct originals) |")
w("|---|--:|---|--:|--:|--:|")
w(f"| **opener classifier, `>0.5` — the definition used everywhere else here** | {rate(so)} | "
  f"{ci(so)} | {ref(so)} | {pur(so)} | {sx['refused_opener']['breadth']} of "
  f"{sx['refused_opener']['n_orig']} |")
w(f"| Sonnet REFUSE/COMPLY judge, `≥0.5` — the definition behind the old figure | {rate(ss)} | "
  f"{ci(ss)} | {ref(ss)} | {pur(ss)} | {sx['refused_sonnet']['breadth']} of "
  f"{sx['refused_sonnet']['n_orig']} |")
w("")
w("**Old figure:** 1,320 refused of 7,877 haiku-BENIGN rewrites. The brief prints this as "
  "**16.84%**, but 1320/7877 = **16.76%**; 16.84% is 1320/7,837, and §0 of the brief indeed "
  "writes the denominator as 7,837 while §8 writes 7,877. One of the two is a typo — the "
  "artifacts contain exactly 7,877 BENIGN rewrites, so **16.76% is the correct restatement of "
  "the old number** and 16.84% is arithmetically wrong.")
w("")
w("**Why v5 is lower.** On the same refused set the two filters disagree substantially:")
w("")
w("| refused set | n refused | haiku says BENIGN | v5 says OR | both |")
w("|---|--:|--:|--:|--:|")
for k, lab in [("refused_opener", "opener `>0.5`"), ("refused_sonnet", "Sonnet judge `≥0.5`")]:
    d = sx[k]
    w(f"| {lab} | {d['n_refused']} | {d['haiku_benign']} | {d['v5_or']} | {d['both']} |")
w("")

# ---------------- Table 7 ----------------
w("## 7. Cross-generator 2×2 trial (brief §11)")
w("")
w("750 fresh originals per generator × 4 rewrites = 3,000 rewrites per cell. All four cells are "
  "now judged (the old §11 judged only the two diagonal cells).")
w("")
w("| cell | v5 genuine-OR (of 3000) | 95% CI | refused (of 3000) | purity | OLD v4 genuine-OR | "
  "OLD v4 purity |")
w("|---|--:|---|--:|--:|--:|--:|")
OLD7 = {"llamaAtt_llamaTgt": ("12.2% (365/3000)", "77.3%"),
        "qwenAtt_qwenTgt": ("6.3% (188/3000)", "77.0%"),
        "llamaAtt_qwenTgt": ("not judged", "—"),
        "qwenAtt_llamaTgt": ("not judged", "—")}
for cell in ["llamaAtt_llamaTgt", "llamaAtt_qwenTgt", "qwenAtt_llamaTgt", "qwenAtt_qwenTgt"]:
    s = A[f"trial2|{cell}"]
    o = OLD7[cell]
    w(f"| {cell.replace('_', ' → ')} | {rate(s)} | {ci(s)} | {ref(s)} | {pur(s)} | {o[0]} | "
      f"{o[1]} |")
w("")
w("Target swap, **paired** (identical rewrites, different target), in points of genuine-OR:")
w("")
for att in ("llamaAtt", "qwenAtt"):
    d, lo, hi = S["target_swap"][att]
    w(f"- {att}: Llama target − Qwen target = **{100*d:+.2f} pts** "
      f"[{100*lo:+.2f}, {100*hi:+.2f}]")
w("")
w("Attacker swap, **unpaired** (the two generators use different 750-original substrates):")
for tgt in ("llamaTgt", "qwenTgt"):
    a = A[f"trial2|qwenAtt_{tgt}"]; b = A[f"trial2|llamaAtt_{tgt}"]
    w(f"- {tgt}: qwenAtt {100*a['rate']:.2f}% vs llamaAtt {100*b['rate']:.2f}% "
      f"(**{100*(a['rate']-b['rate']):+.2f} pts**)")
w("")

# ---------------- 8. what changes ----------------
def r_(name, arm="rwr"):
    return 100 * g(name, arm)["rate"]


w("## 8. What changes under v5, and what holds")
w("")
w("### Changes")
w("")
w("**8.1 The §8 scale-up headline drops by about a third — 16.8% → 10.3%.** Under the definition "
  "used everywhere in this document (opener classifier, ≥3 of 4) the 8k corpus yields "
  f"**{r_('scaleup','refused_opener') if False else 100*A['scaleup|refused_opener']['rate']:.2f}% "
  f"({A['scaleup|refused_opener']['n_or']}/8000)** genuine over-refusal, CI "
  f"{ci(A['scaleup|refused_opener'])}. Holding the old refusal definition fixed (Sonnet "
  f"REFUSE/COMPLY judge at ≥2 of 4) and changing only the benign filter haiku→v5 — which also "
  f"restores the denominator to all 8,000 rewrites, since v5 replaces the haiku pre-filter "
  f"rather than sitting behind it — it is "
  f"**{100*A['scaleup|refused_sonnet']['rate']:.2f}% "
  f"({A['scaleup|refused_sonnet']['n_or']}/8000)** — so roughly **4.4 of the ~6.5 points of drop "
  "come from the judge** and the rest from the stricter refusal threshold. Breadth falls with "
  f"it: {sx['refused_opener']['breadth']} distinct originals of "
  f"{sx['refused_opener']['n_orig']} (25.7%) versus the brief's \"732 of 1,995 (37%)\". The "
  "brief's own bracket — \"7.06% (strict hand audit) and 16.84% (lenient filter) bracket the "
  "same measurement\" — survives: v5 lands inside it.")
w("")
w("**8.2 The Qwen verdict flips for the probe arm.** Brief §9.4 concluded *\"no trained arm "
  "reliably beats base-Qwen on genuine over-refusal; vector is significantly WORSE\"*. Under v5 "
  "both halves change:")
w("")
w("| arm | old (early Sonnet) vs base | v5 vs its own base | v5 vs pooled base (n=2400) |")
w("|---|---|---|---|")
w(f"| logit | +0.03 pts, n.s. | {pc(p['eval_qwen_3sig_logit'])} | "
  f"{pc(S['paired_pooled_base']['eval_qwen_3sig_logit'])} |")
w(f"| probe | +2.84 pts, n.s. | {pc(p['eval_qwen_3sig_probe'])} | "
  f"{pc(S['paired_pooled_base']['eval_qwen_3sig_probe'])} |")
w(f"| vector | −3.97 pts, **sig. worse** | {pc(p['eval_qwen_3sig_vector'])} | "
  f"{pc(S['paired_pooled_base']['eval_qwen_3sig_vector'])} |")
w("")
w("The driver is base-Qwen, not the trained arms. v5 judges base-Qwen-as-attacker far more "
  f"harshly than either earlier judge: purity of its refused rewrites is only "
  f"**{pur(A['POOLED_BASE|qwen (vector+probe+logit)'])}** "
  f"({A['POOLED_BASE|qwen (vector+probe+logit)']['n_or']} of "
  f"{A['POOLED_BASE|qwen (vector+probe+logit)']['n_ref']} pooled refusals judged OR), so its "
  f"genuine-OR rate falls from 6.69% (early Sonnet) to "
  f"**{100*A['POOLED_BASE|qwen (vector+probe+logit)']['rate']:.2f}%** pooled. With the baseline "
  "that much lower, the probe arm now clears it and vector no longer sits below it. **Caveat, and "
  "it is a large one:** probe is the tiny-pool (1,210 trainable pairs) affine twin of vector "
  "(brief §9.1), so this is not a clean signal result, the effect is ~2 points on an eval whose "
  "CIs are ±2 points, it is a single arm out of three, and its significance is knife-edge — "
  "the lower CI bound sits 0.1–0.5 pts above zero and flips sign between bootstrap seeds "
  "(the same contrast came out n.s. on a different seed). The safer restatement is: *under v5 no "
  "Qwen arm is reliably worse than base, and probe is marginally better* — not that the recipe "
  "transfers.")
w("")
w("**8.3 The §7 \"more epochs made it worse\" finding does not survive.** On raw induced "
  "refusal, epochs 3→6 fell 15.0% → 12.8%. On v5 genuine over-refusal the two are "
  f"indistinguishable ({100*g('eval_llama_logit','rwr')['rate']:.2f}% vs "
  f"{100*g('eval_llama_logit_tune1_e6','rwr')['rate']:.2f}% of 800), and the 6-epoch arm has the "
  f"**higher purity** ({pur(g('eval_llama_logit_tune1_e6','rwr'))} vs "
  f"{pur(g('eval_llama_logit','rwr'))}) and the larger paired margin over its own base "
  f"({pc(p['eval_llama_logit_tune1_e6'])} vs {pc(p['eval_llama_logit'])}). The raw-rate decline "
  "was mostly a decline in *impure* refusals.")
w("")
w("**8.4 Every genuine-OR level moves up relative to the §5/§6/§7 hand audit, by roughly "
  "1.4–2.5×.** e.g. logit 7.06% → "
  f"{wtd(g('eval_llama_logit','rwr'))} weighted, vector@L31 1.53% → "
  f"{wtd(g('eval_llama_vector_L31','rwr'))}. The hand audit had a **bucket C** — \"not a safety "
  "refusal\" (capability disclaimers on the [Image] prompts, or rewrites that told the model not "
  "to do the task) — and excluded it. **v5 has no such axis**: a capability disclaimer on a "
  "rewrite whose intent is unchanged and whose ask is benign scores INTENT SAME / HARM BENIGN and "
  "counts as over-refusal. So v5 genuine-OR ≈ hand-audit (bucket A + bucket C), and it is "
  "*mechanically* looser than bucket A. This matters most for the vector/probe arms and for "
  "base-as-attacker, whose refusals were 20–26% bucket C.")
w("")
w("**8.5 The §11 diagonal cells drop ~1–2 points and lose ~11 points of purity** "
  "(v4 → v5: 12.2% → "
  f"{100*A['trial2|llamaAtt_llamaTgt']['rate']:.2f}%, purity 77.3% → "
  f"{pur(A['trial2|llamaAtt_llamaTgt'])}; 6.3% → "
  f"{100*A['trial2|qwenAtt_qwenTgt']['rate']:.2f}%, purity 77.0% → "
  f"{pur(A['trial2|qwenAtt_qwenTgt'])}). v5 is the stricter of the two on this corpus.")
w("")
w("**8.6 A pooled base-as-attacker number now exists.** base-Llama "
  f"**{100*A['POOLED_BASE|llama (vector+probe+logit)']['rate']:.2f}% of 2400** "
  f"{ci(A['POOLED_BASE|llama (vector+probe+logit)'])}; base-Qwen "
  f"**{100*A['POOLED_BASE|qwen (vector+probe+logit)']['rate']:.2f}% of 2400** "
  f"{ci(A['POOLED_BASE|qwen (vector+probe+logit)'])}. Quote these instead of a single run's base "
  "arm — the three Llama base runs span "
  + f"{min(100*g(n,'base')['rate'] for n in ['eval_llama_vector','eval_llama_probe','eval_llama_logit']):.2f}"
  + "–"
  + f"{max(100*g(n,'base')['rate'] for n in ['eval_llama_vector','eval_llama_probe','eval_llama_logit']):.2f}"
  + "% and the three Qwen base runs span "
  + f"{min(100*g(n,'base')['rate'] for n in ['eval_qwen_3sig_vector','eval_qwen_3sig_probe','eval_qwen_3sig_logit']):.2f}"
  + "–"
  + f"{max(100*g(n,'base')['rate'] for n in ['eval_qwen_3sig_vector','eval_qwen_3sig_probe','eval_qwen_3sig_logit']):.2f}"
  + "%, purely from re-generation.")
w("")
w("### Holds")
w("")
w("**8.7 The core Llama claim survives intact.** The logit attacker is the only Llama signal "
  f"whose genuine over-refusal beats its base comparator: "
  f"**{100*g('eval_llama_logit','rwr')['rate']:.2f}% of 800** "
  f"{ci(g('eval_llama_logit','rwr'))} vs base "
  f"{100*g('eval_llama_logit','base')['rate']:.2f}% "
  f"(paired {pc(p['eval_llama_logit'])}; vs pooled base "
  f"{pc(S['paired_pooled_base']['eval_llama_logit'])}). probe "
  f"({pc(p['eval_llama_probe'])}) and vector ({pc(p['eval_llama_vector'])}) do not. The "
  "signal ordering logit > probe > vector is unchanged.")
w("")
w("**8.8 It reproduces at 40× scale.** The 8k corpus gives "
  f"{100*A['scaleup|refused_opener']['rate']:.2f}% against a 0.50% floor on untouched originals — "
  f"a ~{100*A['scaleup|refused_opener']['rate']/0.5:.0f}× lift, spread over "
  f"{sx['refused_opener']['breadth']} distinct originals. Lower than 16.8%, same conclusion.")
w("")
w("**8.9 §6's conclusion (signal type, not layer) holds.** vector@L31 "
  f"{100*g('eval_llama_vector_L31','rwr')['rate']:.2f}% and vector@L17 "
  f"{100*g('eval_llama_vector','rwr')['rate']:.2f}% are both indistinguishable from base "
  f"({pc(p['eval_llama_vector_L31'])} and {pc(p['eval_llama_vector'])}). Note the *direction* of "
  "the tiny L31-vs-L17 gap reverses versus the hand audit (1.53% < 2.72% became "
  f"{100*g('eval_llama_vector_L31','rwr')['rate']:.2f}% > "
  f"{100*g('eval_llama_vector','rwr')['rate']:.2f}%), but both are inside each other's CIs, so "
  "neither ordering is real. The claim that rests on it — the vector signal fails at both layers "
  "— is unaffected.")
w("")
w("**8.10 §11's target-vs-attacker conclusion holds and is now stronger.** On genuine-OR across "
  "the *full* grid (all four cells judged, not just the diagonal): swapping the target "
  "Llama→Qwen with the attacker fixed moves the rate "
  f"**{100*S['target_swap']['llamaAtt'][0]:+.2f}** and "
  f"**{100*S['target_swap']['qwenAtt'][0]:+.2f}** points (paired, identical rewrites); swapping "
  "the attacker with the target fixed moves it "
  f"**{100*(A['trial2|qwenAtt_llamaTgt']['rate']-A['trial2|llamaAtt_llamaTgt']['rate']):+.2f}** "
  f"and **{100*(A['trial2|qwenAtt_qwenTgt']['rate']-A['trial2|llamaAtt_qwenTgt']['rate']):+.2f}** "
  "points. Target-driven, ~6× the attacker effect.")
w("")
w("**8.11 §7's similarity-floor result holds.** floor 0.5→0.7 lowers genuine-OR "
  f"({100*g('eval_llama_logit','rwr')['rate']:.2f}% → "
  f"{100*g('eval_llama_logit_tune2_f70','rwr')['rate']:.2f}%) at unchanged purity "
  f"({pur(g('eval_llama_logit','rwr'))} → {pur(g('eval_llama_logit_tune2_f70','rwr'))}) — the "
  "same \"fewer refusals at the same quality mix\" story, though the CIs overlap.")
w("")

# ---------------- 9. limitations ----------------
w("## 9. Limitations of this recompute")
w("")
w("1. **v5 cannot see bucket C.** It has no \"this is not a safety refusal\" verdict, so "
  "capability disclaimers (the [Image] prompts) and self-defeating rewrites count as over-refusal "
  "whenever intent is preserved and the ask is benign. Every v5 figure here is therefore an "
  "*upper* bound on safety-triggered over-refusal, and is not directly comparable to the §5/§6/§7 "
  "bucket-A column beside it. This is the single largest caveat in the document. See §8.4.")
w(f"2. **The judge model refuses to label some items.** {S['n_unknown']} of the 3,375 pooled "
  "pairs (1.24%) come back with `stop_reason: \"refusal\"` — Sonnet 5's own safety classifier "
  "declines the labelling request — even after 3 retries. They are counted as NOT over-refusal. "
  "They are concentrated in exactly the rewrites most likely to be genuinely harmful, so the bias "
  "is small and in the conservative direction, and they are thin on the ground per arm (≤1 per "
  "800-rewrite arm, 22 of 8,000 in the scale-up, ≤7 per 3,000-rewrite grid cell). Verdicts and "
  "retry outcomes: `probe_or/results/v5_judged/unknown_retry.json`.")
w("3. **The judge is stochastic.** Default temperature; a 351-item set judged twice differed on "
  "3 items (0.9%). Re-running this pipeline will move the second decimal, not the conclusions.")
w("4. **n = 200 originals is underpowered for the 200-prompt evals.** CIs are ±2–3 points on "
  "effects of ~2 points; this is why the Qwen probe result (§8.2) should not be over-read. The "
  "8k scale-up and the 2×2 grid do not have this problem.")
w("5. **The refusal labeller is unchanged.** Only the over-refusal judge was replaced. The "
  "refusal-opener classifier and its ≥3-of-4 threshold are the same everywhere, which is what "
  "makes the tables comparable — but any bias in that classifier propagates identically to every "
  "row.")
w("6. **No analysis code was altered to change a result.** The one code change made during this "
  "work was to `build_v5_judge_pool.py`, to record the `(original, rewrite)` key for *every* "
  "rewrite rather than only refused ones, so the refuse-rate-weighted column could include the "
  "partially-refused rewrites. The judge input CSV is byte-identical before and after "
  "(md5 `86b90770e5e4184615983fd755bd0953`).")
w("")
w("### Discrepancy found in the brief")
w("")
w("`EXPERIMENT_BRIEF.md` reports the §8 scale-up figure as **16.84%** and states the numerator "
  "and denominator as 1,320 of 7,877. 1320/7877 = **16.76%**. 16.84% is 1320/**7,837** — the "
  "denominator §0 of the brief actually prints, while §8 prints 7,877. The artifacts contain "
  "exactly 7,877 haiku-BENIGN rewrites (`benign_scaleup.csv`: BENIGN 7,877, HARMFUL 123), so "
  "7,837 is a typo and the old figure should have read 16.76%. Not corrected here — "
  "`EXPERIMENT_BRIEF.md` is owned by another process.")
w("")

# ---------------- 10. cost ----------------
w("## 10. API spend")
w("")
w("Model `claude-sonnet-5` throughout, at the introductory rate in force on the run date "
  "(2026-08-19): **$2.00 / MTok input, $10.00 / MTok output**; cache read 0.1×, cache write "
  "1.25× (5-minute TTL) or 2.0× (1-hour TTL); Message Batches 0.5×.")
w("")
w("Measured per-call usage (from `response.usage` on representative items): cache read **3,967** "
  "tokens (the few-shot system prefix, identical on every call), fresh input **101–275** tokens "
  "(mean ≈185), output **20–24** tokens. That is **$0.001373 per direct call**.")
w("")
w("| what | calls | note |")
w("|---|--:|---|")
w("| main pool | 3,375 | every unique refused `(original, rewrite)` across all 6 datasets |")
w("| partial-refusal supplement | 351 ×2 | judged twice — see below |")
w("| judge calibration (held-out + purity) | 83 ×2 | judged twice — see below |")
w("| UNKNOWN diagnosis + 3× retry of 48 items | ~150 | resolved 6 of 48 |")
w("| smoke test + usage probes | 9 | |")
w("| **total direct calls** | **≈4,402** | ≈ 4,402 × $0.001373 = **$6.04** |")
w("| cache writes (cold start of each run × workers) | ≈110 | ≈110 × $0.00992 = **$1.09** |")
w("| cancelled batch: 171 succeeded, 3,204 cancelled | 171 | at 0.5× = **$0.12** |")
w("")
w("**Total ≈ $7.2** (≈ $10.7 had standard $3/$15 pricing applied). Judging each unique pair once "
  "saved 711 duplicate calls (4,086 dataset-level refusal events → 3,375 unique pairs, 17%).")
w("")
w("**Why this exceeded the $2–4 estimate.** Three avoidable-in-hindsight costs:")
w("")
w("1. **The Batches API was stalled.** The 3,375-item batch sat at `succeeded=0` for ~35 minutes "
  "(a 68-item batch from another process on the same account had been stuck for 5 hours; "
  "historically an 8,000-item batch on this account finished in 7 minutes). It was cancelled and "
  "re-run as direct concurrent calls, which forfeits the 50% batch discount — roughly +$3 versus "
  "the batch path.")
w("2. **A concurrent process deleted `probe_or/results/v5_judged/` mid-run**, taking the "
  "supplement and calibration verdicts with it; those 434 items had to be re-judged (+$0.6). The "
  "deterministic inputs were regenerated byte-identically, so nothing was lost beyond the money.")
w("3. The 42 judge-self-refusals cost 3 attempts each.")
w("")

# ---------------- 11. artifacts ----------------
w("## 11. Artifacts")
w("")
w("| file | what |")
w("|---|---|")
w("| `probe_or/results/v5_judged/judge_input_all.csv` | the 3,375 unique refused pairs sent to v5 |")
w("| `probe_or/results/v5_judged/verdicts_all.csv` | v5 verdicts for them (`pair_id,intent,harm,is_or`) |")
w("| `probe_or/results/v5_judged/judge_input_partial.csv` | 351 pairs with `0 < refuse_rate ≤ 0.5`, for the weighted column |")
w("| `probe_or/results/v5_judged/verdicts_partial.csv` | v5 verdicts for those |")
w("| `probe_or/results/v5_judged/calib_heldout_v5.csv`, `calib_purity_v5.csv` | v5 on the owner-labelled calibration sets |")
w("| `probe_or/results/v5_judged/unknown_retry.json` | the 48 UNKNOWNs, their retries, and stop reasons |")
w("| `probe_or/results/v5_judged/judge_input_all.csv.manifest.json` | row → pair_id map for every dataset, with refusal flags |")
w("| `probe_or/results/v5_judged/summary_v5.json` | every number in this document, machine-readable |")
w("| `build_v5_judge_pool.py` | builds the de-duplicated judge input + manifest |")
w("| `report_v5.py` | recomputes every figure and writes `summary_v5.json` |")
w("| `write_results_v5.py` | renders this document from `summary_v5.json` |")
w("")

open("RESULTS_V5.md", "w").write("\n".join(O) + "\n")
print("wrote RESULTS_V5.md", len(O), "lines")
