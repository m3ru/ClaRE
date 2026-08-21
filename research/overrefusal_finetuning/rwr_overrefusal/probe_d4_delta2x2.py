#!/usr/bin/env python3
"""Does d4 track alarming SURFACE FORM, or the refusal decision itself?

Replaces the raw-activation 2x2 in probe_d4_alarm.py, which could not discriminate its own
hypotheses: under narrow-lexical, model-assessed-harm, AND refusal-tracking accounts the
main-effect contrast returns the same tie. Three defects made it undecidable --
  (a) the "alarming" benign cell was selected on REFUSAL, so alarm and refusal were 100%
      confounded there, even though 1,337 rewrites introduce alarm vocabulary and are NOT
      refused;
  (b) the harmful row split AdvBench by phrasing, but that split is also a harm-CATEGORY
      split (cyber on one side, misinformation/self-harm on the other);
  (c) it projected RAW activations from two different corpora, re-importing the dataset-
      provenance confound this project already retracted once.

This version uses the factor that is actually manipulable within one corpus:

                    introduced ALARM vocab      no ALARM vocab
    REFUSED         or_alarm                    or_plain
    NOT REFUSED     ctrl_alarm                  ctrl_plain

Same attacker, same originals, same edit-size bins, and everything in Delta space so topic
cancels. That yields the comparison the previous design structurally could not make:
**non-refused rewrites that use the same alarming vocabulary.**

  If d4 is SURFACE FORM  -> ctrl_alarm loads on d4 nearly as much as or_alarm; the ALARM
                            factor dominates and the REFUSAL factor is small.
  If d4 tracks REFUSAL   -> or_* load and ctrl_* do not, at matched vocabulary; the REFUSAL
                            factor dominates.

Controls: held-out originals only (d4's own training originals are excluded -- 48% of the
low-edit bin was in-sample); cluster bootstrap over ORIGINALS, never over pairs; and a
random-direction null, because every other causal claim in this project earned one.

Run: python probe_d4_delta2x2.py --out probe_or/results/d4_delta2x2.json
"""
import argparse, csv, json, os, re, sys
import numpy as np
csv.field_size_limit(sys.maxsize)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_edit_distance import pair_metrics

ALARM = re.compile(r"\b(weaponi[sz]\w*|exploit\w*|exfiltrat\w*|covert\w*|untraceable|"
                   r"infiltrat\w*|counterfeit\w*|malicious\w*|illegal\w*|hack\w*|steal\w*|"
                   r"bomb\w*|weapon\w*|attack\w*|destroy\w*|undetect\w*|evade|evading|"
                   r"bypass\w*|smuggl\w*|forge\w*|sabotag\w*|poison\w*|manipulat\w*)\b", re.I)


def boot_ci(vals, groups, B=2000, seed=0):
    """Cluster bootstrap over originals."""
    rng = np.random.default_rng(seed)
    by = {}
    for v, g in zip(vals, groups):
        by.setdefault(g, []).append(v)
    keys = list(by)
    if not keys:
        return (float("nan"),) * 2
    out = []
    for _ in range(B):
        s = rng.integers(0, len(keys), len(keys))
        pool = [x for i in s for x in by[keys[i]]]
        out.append(np.mean(pool))
    out.sort()
    return float(out[int(.025 * B)]), float(out[int(.975 * B)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta_dir", default="probe_or/results/delta")
    ap.add_argument("--dirs_npz", default="probe_or/results/delta/causal_dirs.npz")
    ap.add_argument("--atlas", default="probe_or/results/llama_signals/probe_absolute.npz")
    ap.add_argument("--d4_idx", type=int, default=3)
    ap.add_argument("--n_null", type=int, default=50)
    ap.add_argument("--heldout_only", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="probe_or/results/d4_delta2x2.json")
    a = ap.parse_args()

    acts = np.load(os.path.join(a.delta_dir, "acts.npy"), mmap_mode="r")
    idx = json.load(open(os.path.join(a.delta_dir, "prompt_index.json")))
    rows = list(csv.DictReader(open(os.path.join(a.delta_dir, "prompt_sets.csv"))))
    Z = np.load(a.dirs_npz, allow_pickle=True)
    dirs = Z["dirs"].astype(np.float64); L = int(Z["layer"])
    train = {str(x) for x in Z["train_originals"]}
    A = np.asarray(acts[:, L, :], dtype=np.float64)
    u = lambda x: x / (np.linalg.norm(x) + 1e-9)
    d4 = u(dirs[a.d4_idx]); d1 = u(dirs[0])
    r_atlas = u(np.load(a.atlas, allow_pickle=True)["d"].astype(np.float64)[L])

    cells = {k: {"D": [], "g": []} for k in
             ("or_alarm", "or_plain", "ctrl_alarm", "ctrl_plain")}
    n_skip_train = 0
    for r in rows:
        s = r["set"]
        if not s.startswith(("or_", "ctrl_")):
            continue
        o, w = r["original"].strip(), r["rewrite"].strip()
        if not o or o not in idx or w not in idx:
            continue
        if a.heldout_only and o in train:
            n_skip_train += 1
            continue
        intro = pair_metrics(o, w)["introduced_words"]
        key = ("or" if s.startswith("or_") else "ctrl") + ("_alarm" if ALARM.search(intro) else "_plain")
        cells[key]["D"].append(A[idx[w]] - A[idx[o]])
        cells[key]["g"].append(o)
    print(f"[2x2] excluded {n_skip_train} pairs whose original was in d4's train split", flush=True)
    for k in cells:
        cells[k]["D"] = np.array(cells[k]["D"])
        print(f"  {k:12s} n={len(cells[k]['D']):5d}  originals={len(set(cells[k]['g']))}", flush=True)

    rng = np.random.default_rng(a.seed)
    R = np.linalg.qr(rng.standard_normal((dirs.shape[1], a.n_null)))[0].T

    def report(name, direction):
        out = {}
        for k, c in cells.items():
            if not len(c["D"]):
                continue
            p = c["D"] @ direction
            lo, hi = boot_ci(p, c["g"], seed=a.seed)
            out[k] = dict(mean=float(p.mean()), lo=lo, hi=hi, n=len(p))
        return out

    res = {"d4": report("d4", d4), "d1": report("d1", d1), "r_atlas": report("r", r_atlas)}
    null = np.array([[ (cells[k]["D"] @ rv).mean() if len(cells[k]["D"]) else 0.0
                       for k in ("or_alarm","or_plain","ctrl_alarm","ctrl_plain")] for rv in R])

    def effects(t):
        oa, op = t["or_alarm"]["mean"], t["or_plain"]["mean"]
        ca, cp = t["ctrl_alarm"]["mean"], t["ctrl_plain"]["mean"]
        return dict(alarm=((oa-op)+(ca-cp))/2, refusal=((oa-ca)+(op-cp))/2,
                    interaction=(oa-op)-(ca-cp))

    print("\n=== Δ projections, held-out originals, cluster-bootstrap 95% CI ===")
    for nm, t in res.items():
        print(f"\n  --- {nm} ---")
        for k in ("or_alarm", "or_plain", "ctrl_alarm", "ctrl_plain"):
            if k in t:
                v = t[k]
                print(f"    {k:12s} n={v['n']:5d}  mean {v['mean']:+8.3f}  [{v['lo']:+.3f}, {v['hi']:+.3f}]")
        e = effects(t)
        print(f"    ALARM effect {e['alarm']:+.3f} | REFUSAL effect {e['refusal']:+.3f} "
              f"| interaction {e['interaction']:+.3f}")

    e4 = effects(res["d4"])
    nul_al = np.abs((null[:, 0] - null[:, 1] + null[:, 2] - null[:, 3]) / 2)
    nul_rf = np.abs((null[:, 0] - null[:, 2] + null[:, 1] - null[:, 3]) / 2)
    print(f"\n=== random-direction null ({a.n_null} directions) ===")
    print(f"  |ALARM effect|   d4 {abs(e4['alarm']):.3f}   null 95th pct {np.percentile(nul_al,95):.3f}")
    print(f"  |REFUSAL effect| d4 {abs(e4['refusal']):.3f}   null 95th pct {np.percentile(nul_rf,95):.3f}")
    ratio = abs(e4["refusal"]) / (abs(e4["alarm"]) + 1e-9)
    print(f"\n  REFUSAL/ALARM ratio for d4 = {ratio:.2f}")
    print("  >1 means d4 tracks the refusal decision at matched vocabulary (surface-form account weakens)")
    print("  <1 means d4 tracks alarming vocabulary regardless of refusal (surface-form account holds)")

    json.dump({"cells": {k: {"n": len(v["D"]), "originals": len(set(v["g"]))} for k, v in cells.items()},
               "projections": res,
               "effects": {k: effects(v) for k, v in res.items()},
               "null_p95": {"alarm": float(np.percentile(nul_al, 95)),
                            "refusal": float(np.percentile(nul_rf, 95))}},
              open(a.out, "w"), indent=1)
    print(f"\n[done] {a.out}")


if __name__ == "__main__":
    main()
