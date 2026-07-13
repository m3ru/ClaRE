#!/usr/bin/env python3
"""DELTA probe ensemble (CPU). Tests whether scoring a rewrite by the INDUCED
shift along the refusal direction ranks induced refusal better than the absolute
probe ranked P (~0.29 Spearman).

  direction d_L : from the independent refuse/benign split (acts_ref/acts_ben) -- raw or LDA
  Dproj_L(pair) = proj_L(rewrite) - proj_L(original)
  target        : dP = P("I cannot"|rewrite) - P("I cannot"|original)

Reports, per layer and for the CV Super-Learner ensemble:
  Spearman(Dproj, dP)   <- the ranking metric that matters (beat absolute ~0.29?)
  AUC(dP > thresh)      <- separating induced-refusal pairs
Compared against the best single layer and L17 (the sweep's delta-best layer).
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score

from probe_ensemble import directions, directions_lda, project, nnls_stack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", required=True, help="dir with acts_ref.npy / acts_ben.npy (the split)")
    ap.add_argument("--pair_dir", required=True, help="dir with acts_orig.npy / acts_rw.npy / pair_meta.json")
    ap.add_argument("--direction", choices=["raw", "lda"], default="raw")
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--dP_thresh", type=float, default=0.01, help="induced-refusal label: dP > thresh")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or args.pair_dir

    acts_ref = np.load(os.path.join(args.acts_dir, "acts_ref.npy"))
    acts_ben = np.load(os.path.join(args.acts_dir, "acts_ben.npy"))
    acts_o = np.load(os.path.join(args.pair_dir, "acts_orig.npy"))
    acts_r = np.load(os.path.join(args.pair_dir, "acts_rw.npy"))
    meta = json.load(open(os.path.join(args.pair_dir, "pair_meta.json")))
    dP = np.array([m["dP"] for m in meta], dtype=np.float64)
    y = (dP > args.dP_thresh).astype(int)
    nL = acts_ref.shape[1]
    print(f"[data] pairs={len(dP)} layers={nL} | induced(dP>{args.dP_thresh})={y.sum()} "
          f"| dP mean={dP.mean():.4f} max={dP.max():.4f}")

    print(f"[dir] {args.direction}" + (f" (shrink={args.shrink})" if args.direction == "lda" else ""))
    d, dn = (directions_lda(acts_ref, acts_ben, args.shrink) if args.direction == "lda"
             else directions(acts_ref, acts_ben))

    # delta projection: proj(rewrite) - proj(original), per layer
    Dproj = project(acts_r, d, dn) - project(acts_o, d, dn)   # [n_pairs, nL]
    mu, sd = Dproj.mean(0), Dproj.std(0) + 1e-9
    Ds = (Dproj - mu) / sd

    # per-layer: Spearman vs dP (ranking) and AUC vs induced label
    per_layer_sp = np.array([spearmanr(Dproj[:, L], dP).statistic for L in range(nL)])
    per_layer_auc = np.array([roc_auc_score(y, Dproj[:, L]) if y.sum() else float("nan") for L in range(nL)])
    best_sp_layer = int(np.nanargmax(np.abs(per_layer_sp)))
    print(f"\n[per-layer Spearman(Dproj, dP)]  best L{best_sp_layer}={per_layer_sp[best_sp_layer]:.4f} | "
          f"L17={per_layer_sp[17]:.4f} | L32={per_layer_sp[32]:.4f}")

    # ensemble: CV NNLS stack fit to dP (regression -> ranking); only weights are CV'd
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(dP))
    folds = np.array_split(idx, args.folds)
    oof = np.zeros(len(dP))
    for f in range(args.folds):
        te = folds[f]
        tr = np.concatenate([folds[j] for j in range(args.folds) if j != f])
        # Fit the stacker to the RANKS of dP (per-fold, no leak). dP is outlier-heavy
        # (mean~0, a few large induced shifts), so fitting raw dP by NNLS-MSE chases the
        # extremes and de-ranks the middle -> ensemble underperformed its best layer.
        # Rank targets make the least-squares objective align with Spearman.
        oof[te] = Ds[te] @ nnls_stack(Ds[tr], rankdata(dP[tr]) / len(tr))

    ens_sp = spearmanr(oof, dP).statistic
    ens_auc = roc_auc_score(y, oof) if y.sum() else float("nan")
    best_layer_sp = float(per_layer_sp[best_sp_layer])
    l17_sp = float(per_layer_sp[17]) if nL > 17 else float("nan")

    print("\n========== held-out ranking of induced refusal (dP) ==========")
    print(f"  ensemble Spearman(pred, dP)   : {ens_sp:.4f}   <-- beat absolute ~0.29 ?")
    print(f"  best single layer (L{best_sp_layer}) Spearman: {best_layer_sp:.4f}")
    print(f"  L17 Spearman                  : {l17_sp:.4f}")
    print(f"  ensemble AUC(dP>{args.dP_thresh})       : {ens_auc:.4f}")
    print(f"  best-layer AUC                : {np.nanmax(per_layer_auc):.4f}")

    summary = {
        "n_pairs": int(len(dP)), "induced": int(y.sum()), "dP_thresh": args.dP_thresh,
        "direction": args.direction,
        "ensemble_spearman_dP": float(ens_sp), "ensemble_auc": float(ens_auc),
        "best_layer": best_sp_layer, "best_layer_spearman": best_layer_sp,
        "l17_spearman": l17_sp,
        "per_layer_spearman": per_layer_sp.tolist(), "per_layer_auc": per_layer_auc.tolist(),
    }
    json.dump(summary, open(os.path.join(out_dir, f"probe_delta_{args.direction}_summary.json"), "w"), indent=2)
    print(f"\n[done] wrote probe_delta_{args.direction}_summary.json -> {out_dir}")


if __name__ == "__main__":
    main()
