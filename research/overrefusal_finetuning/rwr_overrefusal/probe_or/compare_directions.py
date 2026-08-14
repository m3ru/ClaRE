#!/usr/bin/env python3
"""Which refusal DIRECTION (and which layer) best serves the over-refusal reward?

Compares, on one common labeled set, four candidate mass-mean directions:

  ours-raw    mass-mean(refused - complied)   Reddit/jailbreak split, 2500/class -> our L17
  ours-lda    covariance-corrected version of the above (Sigma^-1 (mu+ - mu-))
  arditi-raw  mass-mean(harmful - harmless)   AdvBench vs alpaca-cleaned, 512/class -> Alec's L12
  arditi-lda  covariance-corrected version of the above

Two questions, deliberately kept separate because they have different power:
  Q1 (direction+layer): ours-raw vs arditi-raw, like-for-like. An `ours-raw-n512`
     variant subsampled to the Arditi class size isolates a pure sample-size effect.
  Q2 (probe variant):   raw vs LDA *within our split* (n=2500/class) -- the setting
     the deployed probe actually runs in. LDA on the 512/class Arditi split is
     reported but flagged: a 4096x4096 covariance from 512 samples is far weaker.

TARGET = behaviorally-induced over-refusal on benign rewrites (the thing the reward
must rank): dP_behav = refuse_rate(rewrite) - refuse_rate(original), broad opener
classifier, n=4 samples (results/llama_behav/behav.csv), aligned to pair_acts rows
by `idx`. Metrics: Spearman (graded ranking -- the operative one for a reward) and
AUC (induced, dP>thresh). Paired bootstrap over pairs for the headline contrasts.

Also reports per-layer cosine(ours, arditi): if the two directions are near-parallel
the choice is immaterial, which is worth knowing before spending a GPU on it.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_ensemble import directions, directions_lda   # same fns the deployed scorers use

csv.field_size_limit(sys.maxsize)


def project_chunked(acts_path, d, dn, idxs, chunk=500):
    """[len(idxs), nL] delta-ready projections, streamed so we never hold a float32
    copy of the whole (6000,33,4096) array."""
    A = np.load(acts_path, mmap_mode="r")
    out = np.empty((len(idxs), d.shape[0]), dtype=np.float64)
    for s in range(0, len(idxs), chunk):
        sel = idxs[s:s + chunk]
        blk = np.asarray(A[sel], dtype=np.float32)              # [c, nL, H]
        out[s:s + len(sel)] = np.einsum("ilh,lh->il", blk, d) / dn.squeeze(1)[None, :]
    return out


def evaluate(Dproj, dP, y):
    """Per-layer Spearman vs graded dP and AUC vs the induced label."""
    nL = Dproj.shape[1]
    sp = np.array([spearmanr(Dproj[:, L], dP).statistic for L in range(nL)])
    valid = 0 < int(y.sum()) < len(y)
    auc = np.array([roc_auc_score(y, Dproj[:, L]) if valid else np.nan for L in range(nL)])
    return sp, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours_acts", default="probe_or/activations")
    ap.add_argument("--arditi_acts", default="probe_or/activations_arditi")
    ap.add_argument("--pair_dir", default="probe_or/pair_acts")
    ap.add_argument("--behav_csv", default="probe_or/results/llama_behav/behav.csv")
    ap.add_argument("--out", default="probe_or/results/direction_comparison.json")
    ap.add_argument("--dP_thresh", type=float, default=0.01)
    ap.add_argument("--labels_from_meta", action="store_true",
                    help="take dP from pair_meta.json instead of --behav_csv (for the 12k "
                         "pair_acts_sonnet replication, which has no behavioral CSV)")
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--shrink_sweep", default="0.01,0.1,0.5,0.9",
                    help="LDA ridge values to try on OUR split; the covariance is estimated from "
                         "~2500 samples in 4096 dims, so the theoretical LDA optimum can be lost "
                         "to estimation error -- sweep before concluding raw>LDA")
    ap.add_argument("--layers_of_interest", default="12,17")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    LOI = [int(x) for x in args.layers_of_interest.split(",")]
    rng = np.random.RandomState(args.seed)

    # ---- labels, aligned to pair-acts rows by idx (same guard as probe_qwen_signals) ----
    meta = json.load(open(os.path.join(args.pair_dir, "pair_meta.json")))
    if args.labels_from_meta:
        # higher-power replication (12k Sonnet pairs, ~1852 induced) using the STORED
        # logprob dP. NB: that label is the single-phrase P("I cannot") delta, so it is
        # circular with the LOGIT signal -- fine here (we only rank VECTOR variants),
        # but it is a robustness check, not the primary behavioral target.
        idxs = np.arange(len(meta))
        dP = np.array([float(m["dP"]) for m in meta], dtype=np.float64)
        label_src = f"pair_meta dP (logprob) @ {args.pair_dir}"
    else:
        brows = list(csv.DictReader(open(args.behav_csv)))
        idxs = np.array([int(r["idx"]) for r in brows])
        dP = np.array([float(r["dP_behav"]) for r in brows], dtype=np.float64)
        p_meta = np.array([meta[i]["p_rw"] for i in idxs])
        p_behav = np.array([float(r["p_rw_llama"]) for r in brows])
        if not np.allclose(p_meta, p_behav, atol=1e-4):
            raise SystemExit("[abort] behav_csv does not align with pair_meta -- wrong --pair_dir?")
        label_src = f"behavioral dP_behav @ {args.behav_csv}"
    y = (dP > args.dP_thresh).astype(int)
    print(f"[data] {len(dP)} labeled pairs | induced(dP>{args.dP_thresh}) = {int(y.sum())} "
          f"| target = {label_src}", flush=True)

    # ---- build the candidate directions ----
    cands = {}
    ours_ref = np.load(os.path.join(args.ours_acts, "acts_ref.npy"))
    ours_ben = np.load(os.path.join(args.ours_acts, "acts_ben.npy"))
    print(f"[dirs] ours split: ref={ours_ref.shape} ben={ours_ben.shape}", flush=True)
    cands["ours-raw"] = directions(ours_ref, ours_ben)
    for sh in [float(x) for x in args.shrink_sweep.split(",")]:
        tag = "ours-lda" if sh == args.shrink else f"ours-lda-s{sh:g}"
        cands[tag] = directions_lda(ours_ref, ours_ben, sh)
        print(f"[dirs] built {tag} (shrink={sh:g})", flush=True)

    ard_ref = np.load(os.path.join(args.arditi_acts, "acts_ref.npy"))
    ard_ben = np.load(os.path.join(args.arditi_acts, "acts_ben.npy"))
    n_ard = min(len(ard_ref), len(ard_ben))
    print(f"[dirs] arditi split: harmful={ard_ref.shape} harmless={ard_ben.shape}", flush=True)
    cands["arditi-raw"] = directions(ard_ref, ard_ben)
    cands["arditi-lda"] = directions_lda(ard_ref, ard_ben, args.shrink)

    # sample-size control: our split cut down to the Arditi class size
    sub_r = rng.choice(len(ours_ref), n_ard, replace=False)
    sub_b = rng.choice(len(ours_ben), n_ard, replace=False)
    cands[f"ours-raw-n{n_ard}"] = directions(ours_ref[sub_r], ours_ben[sub_b])

    # ---- geometry: are the two directions even different? ----
    (d_ours, _), (d_ard, _) = cands["ours-raw"], cands["arditi-raw"]
    cos = (np.einsum("lh,lh->l", d_ours, d_ard)
           / (np.linalg.norm(d_ours, axis=1) * np.linalg.norm(d_ard, axis=1) + 1e-9))
    print("\n[geometry] cosine(ours-raw, arditi-raw) per layer: "
          f"L12={cos[12]:+.3f} L17={cos[17]:+.3f} L31={cos[31]:+.3f} "
          f"| mean={cos.mean():+.3f} max={cos.max():+.3f}", flush=True)

    # ---- evaluate every candidate on the common labeled set ----
    results, Dprojs = {}, {}
    for name, (d, dn) in cands.items():
        Po = project_chunked(os.path.join(args.pair_dir, "acts_orig.npy"), d, dn, idxs)
        Pr = project_chunked(os.path.join(args.pair_dir, "acts_rw.npy"), d, dn, idxs)
        D = Pr - Po                       # delta form: the reward's refusal term
        Dprojs[name] = D
        sp, auc = evaluate(D, dP, y)
        best = int(np.nanargmax(sp))
        results[name] = {
            "per_layer_spearman": sp.tolist(), "per_layer_auc": auc.tolist(),
            "best_layer": best, "best_spearman": float(sp[best]), "best_auc": float(auc[best]),
            "at_layers": {str(L): {"spearman": float(sp[L]), "auc": float(auc[L])}
                          for L in LOI if L < len(sp)},
        }
        loi_s = "  ".join(f"L{L}: sp={sp[L]:+.4f} auc={auc[L]:.4f}" for L in LOI if L < len(sp))
        print(f"[{name:16s}] best L{best:2d} sp={sp[best]:+.4f} auc={auc[best]:.4f}   {loi_s}", flush=True)

    # ---- headline contrasts: paired bootstrap over pairs on the Spearman difference ----
    def boot_diff(a_name, a_L, b_name, b_L):
        A, B = Dprojs[a_name][:, a_L], Dprojs[b_name][:, b_L]
        obs = spearmanr(A, dP).statistic - spearmanr(B, dP).statistic
        r = np.random.RandomState(args.seed)
        diffs = np.empty(args.n_boot)
        n = len(dP)
        for i in range(args.n_boot):
            s = r.randint(0, n, n)
            diffs[i] = spearmanr(A[s], dP[s]).statistic - spearmanr(B[s], dP[s]).statistic
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        return {"delta_spearman": float(obs), "ci95": [float(lo), float(hi)],
                "excludes_zero": bool(lo > 0 or hi < 0)}

    print(f"\n[bootstrap] paired, {args.n_boot} resamples over pairs", flush=True)
    contrasts = {}
    best_ours, best_ard = results["ours-raw"]["best_layer"], results["arditi-raw"]["best_layer"]
    # LDA gets its best shot: the winning shrinkage from the sweep
    lda_names = [n for n in results if n.startswith("ours-lda")]
    best_lda = max(lda_names, key=lambda n: results[n]["best_spearman"])
    print(f"  (best LDA variant across the shrink sweep: {best_lda})", flush=True)
    for label, (an, aL, bn, bL) in {
        "ours-raw@L17_vs_arditi-raw@L12": ("ours-raw", 17, "arditi-raw", 12),
        "ours-raw@best_vs_arditi-raw@best": ("ours-raw", best_ours, "arditi-raw", best_ard),
        f"{best_lda}@best_vs_ours-raw@best": (best_lda, int(results[best_lda]["best_layer"]),
                                              "ours-raw", best_ours),
        f"ours-raw@best_vs_ours-raw-n{n_ard}@best": (
            "ours-raw", best_ours, f"ours-raw-n{n_ard}",
            int(results[f"ours-raw-n{n_ard}"]["best_layer"])),
    }.items():
        c = boot_diff(an, aL, bn, bL)
        contrasts[label] = c
        verdict = "SIGNIFICANT" if c["excludes_zero"] else "not distinguishable"
        print(f"  {label:44s} d={c['delta_spearman']:+.4f} "
              f"CI[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]  {verdict}", flush=True)

    out = {"n_pairs": int(len(dP)), "n_induced": int(y.sum()), "dP_thresh": args.dP_thresh,
           "shrink": args.shrink, "n_arditi_per_class": int(n_ard),
           "cosine_ours_vs_arditi": cos.tolist(),
           "candidates": results, "contrasts": contrasts}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
