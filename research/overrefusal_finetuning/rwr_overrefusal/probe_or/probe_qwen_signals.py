#!/usr/bin/env python3
"""Build the Qwen refusal VECTOR + PROBE ensemble from mass-mean directions,
using a BEHAVIORAL dP target (from gen_qwen_refusal.py) to pick the layer and
fit the combiner.

  direction d_L : mass-mean (mean_ref - mean_ben) from the refuse/benign split
  Dproj_L(pair): proj_L(rewrite) - proj_L(original)          [delta form]
  target dP     : refuse_rate(rewrite) - refuse_rate(original)  [behavioral]

Outputs:
  qwen_signals_summary.json  -- per-layer Spearman/AUC, best layer, ensemble
  qwen_vector.npz            -- single best-layer direction + delta standardization
  qwen_probe.npz             -- all-layer dirs + per-layer delta standardization + stack weights

Both scorers compute a delta refusal score for any (original, rewrite):
  vector: s = (proj_L(rw) - proj_L(orig) - mu_L) / sd_L
  probe : s = sum_L w_L * (proj_L(rw) - proj_L(orig) - mu_L) / sd_L
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score

from probe_ensemble import directions, directions_lda, project, nnls_stack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", required=True, help="dir with acts_ref.npy / acts_ben.npy")
    ap.add_argument("--pair_dir", required=True, help="dir with acts_orig.npy / acts_rw.npy")
    ap.add_argument("--behav_csv", required=True, help="gen_qwen_refusal.py output (idx-aligned dP_behav)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dP_thresh", type=float, default=0.01, help="induced-refusal label: dP_behav > thresh")
    ap.add_argument("--direction", choices=["raw", "lda"], default="raw",
                    help="raw mass-mean, or covariance-corrected (LDA) directions")
    ap.add_argument("--shrink", type=float, default=0.1, help="LDA covariance ridge shrinkage")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out_dir = args.out_dir or args.pair_dir
    sfx = f"_{args.direction}"

    # float32 on load: fp16 means/norms overflow to inf for large-activation models (Qwen)
    acts_ref = np.load(os.path.join(args.acts_dir, "acts_ref.npy")).astype(np.float32)
    acts_ben = np.load(os.path.join(args.acts_dir, "acts_ben.npy")).astype(np.float32)
    acts_o = np.load(os.path.join(args.pair_dir, "acts_orig.npy")).astype(np.float32)
    acts_r = np.load(os.path.join(args.pair_dir, "acts_rw.npy")).astype(np.float32)
    for nm, a in (("acts_ref", acts_ref), ("acts_ben", acts_ben), ("acts_orig", acts_o), ("acts_rw", acts_r)):
        assert np.isfinite(a).all(), f"non-finite in {nm} -- fp16 overflow at extraction? re-run with --acts_dtype float32"
    pair_meta = json.load(open(os.path.join(args.pair_dir, "pair_meta.json")))

    # behavioral dP, aligned to pair-acts rows by `idx` (same pairs CSV order)
    brows = list(csv.DictReader(open(args.behav_csv)))
    idxs = np.array([int(r["idx"]) for r in brows])
    dP = np.array([float(r["dP_behav"]) for r in brows], dtype=np.float64)
    assert idxs.max() < len(acts_o), f"behav idx {idxs.max()} exceeds pair-acts rows {len(acts_o)}"
    # cross-check the behav CSV points at the RIGHT pair-acts (guards a wrong/stale --pair_dir)
    p_meta = np.array([pair_meta[i]["p_rw"] for i in idxs])
    p_behav = np.array([float(r["p_rw_llama"]) for r in brows])
    if not np.allclose(p_meta, p_behav, atol=1e-4):
        bad = int((~np.isclose(p_meta, p_behav, atol=1e-4)).sum())
        raise SystemExit(f"[abort] behav_csv p_rw_llama != pair_meta p_rw for {bad}/{len(idxs)} rows "
                         f"-- wrong/stale --pair_dir for this behav_csv?")
    acts_o, acts_r = acts_o[idxs], acts_r[idxs]      # subset/align pair-acts to the labeled rows
    y = (dP > args.dP_thresh).astype(int)
    nL = acts_ref.shape[1]
    print(f"[data] ref={len(acts_ref)} ben={len(acts_ben)} labeled_pairs={len(dP)} layers={nL} "
          f"| induced(dP>{args.dP_thresh})={int(y.sum())} | dP mean={dP.mean():.4f} max={dP.max():.4f}", flush=True)
    if y.sum() < 10:
        print(f"[warn] only {int(y.sum())} induced positives -- layer/stack fit will be noisy", flush=True)

    d, dn = (directions_lda(acts_ref, acts_ben, args.shrink) if args.direction == "lda"
             else directions(acts_ref, acts_ben))            # mass-mean directions (raw or LDA)
    assert np.isfinite(dn).all(), "non-finite direction norm (fp16 overflow) -- re-extract dirs with --acts_dtype float32"
    Dproj = project(acts_r, d, dn) - project(acts_o, d, dn)   # [n_pairs, nL]
    mu, sd = Dproj.mean(0), Dproj.std(0) + 1e-9
    Ds = (Dproj - mu) / sd

    # per-layer ranking of behavioral dP + separation of induced pairs.
    # AUC needs both classes; Spearman needs dP variance.
    valid_auc = 0 < int(y.sum()) < len(y)
    per_layer_sp = np.array([spearmanr(Dproj[:, L], dP).statistic for L in range(nL)])
    per_layer_auc = np.array([roc_auc_score(y, Dproj[:, L]) if valid_auc else np.nan for L in range(nL)])
    if np.all(np.isnan(per_layer_sp)):
        raise SystemExit("[abort] all per-layer Spearman are NaN -- dP_behav has no variance "
                         "(Qwen didn't differentiate rewrites). Re-check generation / need labels with spread.")
    best_L = int(np.nanargmax(per_layer_sp))                  # sign-correct best layer = the vector
    print(f"[per-layer Spearman(Dproj, dP_behav)] best L{best_L}={per_layer_sp[best_L]:.4f} "
          f"| worst={np.nanmin(per_layer_sp):.4f}", flush=True)

    # ensemble: CV NNLS stack fit to RANKS of dP (aligns least-squares with Spearman)
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(dP))
    folds = np.array_split(order, args.folds)
    oof = np.zeros(len(dP))
    for f in range(args.folds):
        te = folds[f]
        tr = np.concatenate([folds[j] for j in range(args.folds) if j != f])
        oof[te] = Ds[te] @ nnls_stack(Ds[tr], rankdata(dP[tr]) / len(tr))
    ens_sp = spearmanr(oof, dP).statistic
    ens_auc = roc_auc_score(y, oof) if valid_auc else float("nan")
    w_full = nnls_stack(Ds, rankdata(dP) / len(dP))           # refit on all data for the saved scorer

    print("\n========== Qwen refusal signals (behavioral dP target) ==========")
    print(f"  VECTOR  best layer L{best_L}  Spearman={per_layer_sp[best_L]:.4f}  AUC={per_layer_auc[best_L]:.4f}")
    print(f"  PROBE   ensemble           Spearman={ens_sp:.4f}  AUC={ens_auc:.4f}")
    print(f"  ensemble adds over best layer: {ens_sp - per_layer_sp[best_L]:+.4f}")
    top_w = sorted(((float(w_full[L]), L) for L in range(nL)), reverse=True)[:5]
    print(f"  top stack weights: " + ", ".join(f"L{L}={w:.3f}" for w, L in top_w))

    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "direction": args.direction,
        "n_labeled_pairs": int(len(dP)), "induced": int(y.sum()), "dP_thresh": args.dP_thresh,
        "layers": int(nL), "best_layer": best_L,
        "vector_spearman": float(per_layer_sp[best_L]), "vector_auc": float(per_layer_auc[best_L]),
        "ensemble_spearman": float(ens_sp), "ensemble_auc": float(ens_auc),
        "per_layer_spearman": per_layer_sp.tolist(), "per_layer_auc": per_layer_auc.tolist(),
        "stack_weights": w_full.tolist(),
    }
    json.dump(summary, open(os.path.join(out_dir, f"qwen_signals_summary{sfx}.json"), "w"), indent=2)
    # scorers: directions + per-layer delta standardization (mu, sd) + weights
    np.savez(os.path.join(out_dir, f"qwen_vector{sfx}.npz"),
             d=d[best_L], dn=dn[best_L], mu=mu[best_L], sd=sd[best_L], layer=best_L)
    np.savez(os.path.join(out_dir, f"qwen_probe{sfx}.npz"),
             d=d, dn=dn, mu=mu, sd=sd, w=w_full, best_layer=best_L)
    print(f"\n[done] wrote qwen_signals_summary{sfx}.json + qwen_vector{sfx}.npz + qwen_probe{sfx}.npz -> {out_dir}")


if __name__ == "__main__":
    main()
