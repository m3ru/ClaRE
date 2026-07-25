#!/usr/bin/env python3
"""Probe-ensemble go/no-go (CPU). Given per-layer activations from
extract_layer_acts.py:

  1. build per-layer diff-of-means (mass-mean) directions from the INDEPENDENT
     refusal/benign split;
  2. project the eval pool onto each layer's direction -> per-layer scores Z;
  3. per-layer AUC vs behavioral label (P('I cannot')>thresh) -> should reproduce
     the inverted-U with an L17 peak;
  4. combine layers with a cross-validated non-negative-least-squares stacker
     (Breiman 1996 NNLS; a Super Learner whose library contains the equal-weight
     average and every single layer) and compare held-out ensemble AUC against
     the best single layer (honest CV), L17, L32, and the equal-weight average.

Decision rule: adopt the ensemble only if it beats the best single layer AND the
L32/L17 diff-of-means baselines on held-out AUC.

Also reports Spearman(score, P) as a graded check, and saves the directions for
reuse as score_probe_or.py.
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def directions(acts_ref, acts_ben):
    """Per-layer mass-mean direction d_L = mean_ref - mean_ben (raw diff-of-means)."""
    d = acts_ref.mean(0) - acts_ben.mean(0)            # [nL+1, H]
    dn = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    return d, dn


def directions_lda(acts_ref, acts_ben, shrink=0.1):
    """Covariance-corrected (LDA / mass-mean-with-correction) direction per layer:
       d_L = Sigma_L^{-1} (mu_ref - mu_ben), with ridge shrinkage on Sigma
       (Sigma is 4096x4096 and near-singular at ~2.5k samples, so shrink is required)."""
    nLp, H = acts_ref.shape[1], acts_ref.shape[2]
    ar, ab = acts_ref.astype(np.float32), acts_ben.astype(np.float32)
    nr, nb = len(ar), len(ab)
    d = np.zeros((nLp, H), dtype=np.float32)
    for L in range(nLp):
        Xr, Xb = ar[:, L, :], ab[:, L, :]
        mr, mb = Xr.mean(0), Xb.mean(0)
        Xrc, Xbc = Xr - mr, Xb - mb
        S = (Xrc.T @ Xrc + Xbc.T @ Xbc) / (nr + nb - 2)   # pooled within-class covariance
        # trace-proportional ridge + absolute floor: keeps degenerate layers (e.g. the
        # constant last-token embedding at L0 -> S=0, trace=0) non-singular. Negligible
        # for normal layers whose trace(S)/H is O(1e3+).
        S.flat[::H + 1] += shrink * (np.trace(S) / H) + 1e-6
        d[L] = np.linalg.solve(S, mr - mb)
    dn = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    return d.astype(np.float64), dn


def project(acts, d, dn):
    """Z[i, L] = <acts[i,L], d_L> / ||d_L||."""
    return np.einsum("ilh,lh->il", acts.astype(np.float32), d) / dn.squeeze(1)[None, :]


def nnls_stack(Ztr, ytr):
    """Breiman non-negative least squares stacking; weights normalized to sum 1."""
    w, _ = nnls(Ztr, ytr)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--p_thresh", type=float, default=0.1, help="refuse if P('I cannot') > thresh")
    ap.add_argument("--direction", choices=["raw", "lda"], default="raw",
                    help="raw diff-of-means, or covariance-corrected LDA direction")
    ap.add_argument("--shrink", type=float, default=0.1, help="LDA covariance ridge shrinkage")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out_dir = args.out_dir or args.acts_dir

    acts_ref = np.load(os.path.join(args.acts_dir, "acts_ref.npy"))
    acts_ben = np.load(os.path.join(args.acts_dir, "acts_ben.npy"))
    acts_eval = np.load(os.path.join(args.acts_dir, "acts_eval.npy"))
    meta = json.load(open(os.path.join(args.acts_dir, "eval_meta.json")))
    p = np.array([m["p_icannot"] for m in meta], dtype=np.float32)
    y = (p > args.p_thresh).astype(int)
    nL = acts_ref.shape[1]
    print(f"[data] ref={len(acts_ref)} ben={len(acts_ben)} eval={len(acts_eval)} "
          f"layers={nL} | positives(P>{args.p_thresh})={y.sum()}/{len(y)}")

    # --- directions from the independent split; project eval ---
    print(f"[dir] {args.direction} direction" + (f" (shrink={args.shrink})" if args.direction == "lda" else ""))
    d, dn = (directions_lda(acts_ref, acts_ben, args.shrink) if args.direction == "lda"
             else directions(acts_ref, acts_ben))
    Z = project(acts_eval, d, dn)                       # [n_eval, nL]
    # standardize each layer's score (needed before combining)
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    Zs = (Z - mu) / sd

    # --- per-layer AUC (parameter-free: the direction IS the probe) ---
    per_layer_auc = np.array([roc_auc_score(y, Z[:, L]) for L in range(nL)])
    best_layer = int(np.argmax(per_layer_auc))
    print(f"\n[per-layer AUC vs P('I cannot')>{args.p_thresh}]  (L0=embed ... L{nL-1}=last)")
    for L in range(nL):
        tag = ("  <- best" if L == best_layer else "") + ("  <- L17" if L == 17 else "") \
            + ("  <- L32(canonical)" if L == 32 else "")
        if tag:
            print(f"  L{L:2d}  AUC={per_layer_auc[L]:.4f}{tag}")

    # --- Only the STACK WEIGHTS are fit on the eval set (the directions come from
    #     the independent split), so per-layer AUCs and the equal-weight average are
    #     already honest; only the NNLS ensemble needs cross-validation. NNLS keeps
    #     w>=0, so the fair single-layer baseline is also non-flipped -> max(per_layer_auc).
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, args.folds)
    oof_ens = np.zeros(len(y))
    for f in range(args.folds):
        te = folds[f]
        tr = np.concatenate([folds[j] for j in range(args.folds) if j != f])
        w = nnls_stack(Zs[tr], y[tr].astype(float))
        oof_ens[te] = Zs[te] @ w

    ens_auc = roc_auc_score(y, oof_ens)
    best_single_auc = float(per_layer_auc[best_layer])   # honest: directions external, no fitting
    avg_auc = roc_auc_score(y, Zs.mean(1))
    l17_auc = float(per_layer_auc[17]) if nL > 17 else float("nan")
    l32_auc = float(per_layer_auc[32]) if nL > 32 else float("nan")

    print("\n========== held-out AUC (predict behavioral refusal) ==========")
    print(f"  ensemble (CV NNLS stack)     : {ens_auc:.4f}")
    print(f"  best single layer (L{best_layer:<2d})     : {best_single_auc:.4f}")
    print(f"  equal-weight average         : {avg_auc:.4f}")
    print(f"  L17 diff-of-means            : {l17_auc:.4f}")
    print(f"  L32 diff-of-means (canonical): {l32_auc:.4f}")
    print(f"  Spearman(ensemble, P)        : {spearmanr(oof_ens, p).statistic:.4f}")
    verdict = ens_auc > best_single_auc   # strongest non-flipped baseline; already >= L17/L32
    print(f"\n  VERDICT: ensemble {'BEATS' if verdict else 'does NOT beat'} the best single layer (L{best_layer})")

    # refit weights on all data for the final scorer
    w_full = nnls_stack(Zs, y.astype(float))
    summary = {
        "n_eval": int(len(y)), "positives": int(y.sum()), "p_thresh": args.p_thresh,
        "per_layer_auc": per_layer_auc.tolist(),
        "ensemble_auc": ens_auc, "best_layer": best_layer, "best_single_layer_auc": best_single_auc,
        "equal_avg_auc": avg_auc, "l17_auc": l17_auc, "l32_auc": l32_auc,
        "spearman_ensemble_p": float(spearmanr(oof_ens, p).statistic),
        "beats_best_single_layer": bool(verdict),
        "ensemble_weights_full": w_full.tolist(),
    }
    json.dump(summary, open(os.path.join(out_dir, "probe_ensemble_summary.json"), "w"), indent=2)
    # save directions + standardization + weights so score_probe_or.py can reuse
    np.savez(os.path.join(out_dir, "probe_ensemble.npz"),
             d=d, dn=dn, mu=mu, sd=sd, w=w_full)
    print(f"\n[done] wrote probe_ensemble_summary.json + probe_ensemble.npz -> {out_dir}")


if __name__ == "__main__":
    main()
