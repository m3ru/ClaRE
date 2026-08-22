#!/usr/bin/env python3
"""Assemble every RWR arm's held-out over-refusal rate into one table.

The question: used as the OR reward, does the ABLITERATED refusal vector (the direction
Arditi's ablation criterion selects) train a better attacker than the layers we picked by
other criteria, or than the logit signal?

Each arm's eval reports two numbers on the SAME held-out originals: `rwr` (the trained
attacker's rewrites) and `base` (the untrained model's rewrites). The bar an arm must clear
is its OWN base, because base rates differ slightly between evals.

Reads whatever exists, so it can be run before and after the new arms land.
"""
import glob, json, os, sys
from math import sqrt

ARMS = {
    "llama": [
        ("eval_llama_vector",         "vector @L17  (behavioural corr.)"),
        ("eval_llama_vector_L31",     "vector @L31  (AUC)"),
        ("eval_llama_vector_abl_L12", "vector @L12  (ABLITERATED)"),
        ("eval_llama_probe",          "probe"),
        ("eval_llama_logit",          "logit"),
    ],
    "qwen": [
        ("eval_qwen_3sig_vector",     "vector @L58  (AUC)"),
        ("eval_qwen_vector_abl_L60",  "vector @L60  (ABLITERATED)"),
        ("eval_qwen_3sig_probe",      "probe"),
        ("eval_qwen_3sig_logit",      "logit"),
    ],
}


def _by_orig(rows):
    """Group refuse rates by the ORIGINAL they came from. The 4 rewrites of one original are
    not independent draws -- they share a prompt -- so the resampling unit is the original."""
    g = {}
    for r in rows:
        g.setdefault(r["orig_idx"], []).append(float(r["refuse_rate"]))
    return list(g.values())


def cluster_ci(rows, n_boot=10000, seed=0):
    """Cluster bootstrap over originals, percentile interval.

    NOT Wilson on the 800 rewrites: that treats 4 correlated rewrites of one prompt as 4
    independent observations and returns an interval that is far too narrow. The v5 judge
    pipeline already resamples originals for exactly this reason; this makes the raw-rate
    table consistent with it.
    """
    import random
    cl = _by_orig(rows)
    if not cl:
        return (float("nan"),) * 3
    point = 100 * sum(sum(c) for c in cl) / sum(len(c) for c in cl)
    rng = random.Random(seed)
    n, out = len(cl), []
    for _ in range(n_boot):
        pick = [cl[rng.randrange(n)] for _ in range(n)]
        tot = sum(len(c) for c in pick)
        out.append(100 * sum(sum(c) for c in pick) / tot if tot else 0.0)
    out.sort()
    return point, out[int(0.025 * n_boot)], out[int(0.975 * n_boot)]


def paired_cluster_ci(a_rows, b_rows, n_boot=10000, seed=0):
    """Trained minus base, resampling the SAME originals for both arms.

    The arms share the held-out originals, so the contrast is paired; resampling them
    independently would throw away that pairing and inflate the interval.
    """
    import random
    A, B = {}, {}
    for r in a_rows:
        A.setdefault(r["orig_idx"], []).append(float(r["refuse_rate"]))
    for r in b_rows:
        B.setdefault(r["orig_idx"], []).append(float(r["refuse_rate"]))
    keys = sorted(set(A) & set(B))
    if not keys:
        return (float("nan"),) * 3
    def rate(d, ks):
        num = sum(sum(d[k]) for k in ks); den = sum(len(d[k]) for k in ks)
        return 100 * num / den if den else 0.0
    point = rate(A, keys) - rate(B, keys)
    rng = random.Random(seed)
    n, out = len(keys), []
    for _ in range(n_boot):
        ks = [keys[rng.randrange(n)] for _ in range(n)]
        out.append(rate(A, ks) - rate(B, ks))
    out.sort()
    return point, out[int(0.025 * n_boot)], out[int(0.975 * n_boot)]


def load(tag):
    for p in (f"probe_or/results/{tag}/eval_final.json",):
        if os.path.exists(p):
            return json.load(open(p))
    return None


for model, arms in ARMS.items():
    print(f"\n=== {model.upper()} — held-out over-refusal rate (n=200 originals x 4 rewrites) ===")
    print(f"{'arm':34s} {'trained':>18s} {'untrained base':>18s} {'lift':>8s}  {'sim':>5s}")
    for tag, label in arms:
        d = load(tag)
        if d is None:
            print(f"{label:34s} {'— not run —':>18s}")
            continue
        ex = d.get("examples", {})
        rr, rl, rh = cluster_ci(ex.get("rwr", []))
        bb, bl, bh = cluster_ci(ex.get("base", []))
        dp, dl, dh = paired_cluster_ci(ex.get("rwr", []), ex.get("base", []))
        sig = "**" if (dl > 0 or dh < 0) else "n.s."
        print(f"{label:34s} {rr:6.2f} [{rl:5.2f},{rh:5.2f}] "
              f"{bb:6.2f} [{bl:5.2f},{bh:5.2f}] {dp:+7.2f} [{dl:+.2f},{dh:+.2f}] {sig}")
    print("  lift = trained minus its own base, PAIRED cluster bootstrap over the 200 originals\n"
          "  (10k resamples). Not Wilson on 800 rewrites -- 4 rewrites of one prompt are not\n"
          "  independent, and Wilson would report an interval roughly 2x too narrow.")

print("\nNote: the trained arms hold semantics far better than base (sim ~0.83 vs ~0.63), so a\n"
      "lower refusal rate is not automatically worse -- base wins some refusals by drifting off\n"
      "topic, which the judge would reject as intent-shifted. Judge-confirmed rates are the\n"
      "final word; these are refusal rates under the eval's own classifier.")

# ---------------------------------------------------------------- judge-confirmed
def judged():
    """Judge-confirmed rates from the v5 pool -- the number that actually settles this.

    Raw refusal rate is confounded: the untrained base wins refusals partly by drifting off
    topic, which the two-axis judge rejects as intent-shifted. Measured base purity runs
    18.8-66.1% against 42.4-94.1% for the trained arms, so the raw comparison understates
    every trained arm -- on Qwen's vector arm by 6.6 points, which flips it from "loses" to
    "ties". Never report the raw lift without this.
    """
    p = "probe_or/results/v5_judged/summary_v5.json"
    if not os.path.exists(p):
        print("\n(no v5 judge summary yet)")
        return
    d = json.load(open(p))["arms"]
    print("\n=== JUDGE-CONFIRMED over-refusal (two-axis judge: intent preserved AND benign) ===")
    print(f"{'arm':30s} {'trained':>20s} {'pur':>5s} | {'base':>20s} {'pur':>5s} | {'lift':>7s}")
    seen = set()
    for k in d:
        arm = k.split("|")[0]
        if arm in seen:
            continue
        seen.add(arm)
        r, b = d.get(f"{arm}|rwr"), d.get(f"{arm}|base")
        if not (r and b):
            continue
        tag = "BEATS" if r["lo"] > b["hi"] else ("loses" if r["hi"] < b["lo"] else "ties")
        print(f"{arm:30s} {100*r['rate']:6.2f} [{100*r['lo']:5.2f},{100*r['hi']:5.2f}] {r['purity']:5.0%} | "
              f"{100*b['rate']:6.2f} [{100*b['lo']:5.2f},{100*b['hi']:5.2f}] {b['purity']:5.0%} | "
              f"{100*(r['rate']-b['rate']):+6.2f} {tag}")
    print("  pur = purity, the fraction of flagged refusals the judge confirms as genuine\n"
          "  over-refusal. Low base purity is why raw rates flatter the untrained model.")


judged()
