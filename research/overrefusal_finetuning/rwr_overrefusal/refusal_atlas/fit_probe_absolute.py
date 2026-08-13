#!/usr/bin/env python3
"""Refit the mass-mean PROBE ensemble in ABSOLUTE space (the space the atlas applies it in).

The Qwen/Llama probes were originally fit on DELTA projections (proj(rw)-proj(orig)) but the
atlas scores singletons (absolute projections). This refits the combiner in absolute space,
against the BROAD behavioral refusal label, so the probe is coherent with its application AND
symmetric across models.

  directions d_L = mean_ref - mean_ben           (mass-mean, from the refuse/benign split)
  X          = the pair singletons (originals + rewrites), absolute activations
  proj_L(x)  = <x_L, d_L> / ||d_L||               (absolute, per layer)
  y          = broad refuse label per singleton   (refuse_rate >= thresh)
  probe(x)   = sum_L w_L * (proj_L(x) - mu_L)/sd_L , w from CV-NNLS stack (L0 excluded)

Memory-safe: activations are mmap'd and streamed through the projection in chunks (never
materialised as float32 in full -- Qwen's arrays are multi-GB). Saves d, dn, mu, sd, w,
best_layer (absolute-space) for score_signals.py to load and apply with the STORED mu/sd.
"""
import argparse, csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "probe_or"))
from probe_ensemble import nnls_stack
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def chunked_mean(path, bs=256):
    a = np.load(path, mmap_mode="r")
    n, nLp, H = a.shape
    acc = np.zeros((nLp, H), np.float64)
    for s in range(0, n, bs):
        acc += np.asarray(a[s:s + bs], dtype=np.float64).sum(0)
    return acc / n, nLp, H


def proj_subset(path, idx, d, dn, bs=256):
    """Absolute per-layer projection <x_L,d_L>/||d_L|| for rows `idx`, streamed in chunks."""
    a = np.load(path, mmap_mode="r")
    dd = d.astype(np.float32)
    ddn = dn.squeeze(1).astype(np.float32)
    out = np.zeros((len(idx), d.shape[0]), np.float32)
    for s in range(0, len(idx), bs):
        sl = idx[s:s + bs]
        chunk = np.asarray(a[sl], dtype=np.float32)          # [b, nLp, H]
        out[s:s + len(sl)] = np.einsum("ilh,lh->il", chunk, dd) / ddn[None, :]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", required=True, help="dir with acts_ref.npy / acts_ben.npy (directions)")
    ap.add_argument("--pair_dir", required=True, help="dir with acts_orig.npy / acts_rw.npy (singletons)")
    ap.add_argument("--behav_csv", required=True, help="broad labels: refuse_rate_orig / refuse_rate_rw, idx-aligned")
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--refuse_thresh", type=float, default=0.25, help="singleton refused if rate >= thresh")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out_npz) or ".", exist_ok=True)

    # --- directions from the refuse/benign split (chunked float64 means) ---
    m_ref, nLp, H = chunked_mean(os.path.join(a.acts_dir, "acts_ref.npy"))
    m_ben, _, _ = chunked_mean(os.path.join(a.acts_dir, "acts_ben.npy"))
    d = (m_ref - m_ben)
    dn = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    assert np.isfinite(d).all() and np.isfinite(dn).all(), "non-finite direction (fp16 overflow at extraction?)"
    nL = nLp

    # --- labels + singleton projections (streamed) ---
    rows = list(csv.DictReader(open(a.behav_csv)))
    idx = np.array([int(r["idx"]) for r in rows])
    ro = np.array([float(r["refuse_rate_orig"]) for r in rows])
    rr = np.array([float(r["refuse_rate_rw"]) for r in rows])
    # guards: idx must index the INTENDED pair arrays, and behav must belong to this pair_dir
    for nm in ("acts_orig.npy", "acts_rw.npy"):
        n_rows = np.load(os.path.join(a.pair_dir, nm), mmap_mode="r").shape[0]
        assert idx.max() < n_rows, f"behav idx {idx.max()} exceeds {nm} rows {n_rows} -- wrong --pair_dir?"
    pm_path = os.path.join(a.pair_dir, "pair_meta.json")
    if os.path.exists(pm_path):
        pm = json.load(open(pm_path))
        p_meta = np.array([float(pm[i]["p_rw"]) for i in idx])
        p_behav = np.array([float(r.get("p_rw_llama") or "nan") for r in rows])
        if np.isfinite(p_behav).all() and not np.allclose(p_meta, p_behav, atol=1e-4):
            bad = int((~np.isclose(p_meta, p_behav, atol=1e-4)).sum())
            raise SystemExit(f"[abort] behav p_rw_llama != pair_meta p_rw for {bad}/{len(idx)} rows -- wrong/stale --pair_dir")
    P_o = proj_subset(os.path.join(a.pair_dir, "acts_orig.npy"), idx, d, dn)
    P_r = proj_subset(os.path.join(a.pair_dir, "acts_rw.npy"), idx, d, dn)
    P = np.concatenate([P_o, P_r], axis=0)                    # [2N, nL+1] absolute projections
    rate = np.concatenate([ro, rr])
    y = (rate >= a.refuse_thresh).astype(int)
    assert np.isfinite(P).all(), "non-finite projections"
    print(f"[data] dirs [{nL},{H}] | singletons={len(P)} | positives(rate>={a.refuse_thresh})={int(y.sum())} "
          f"| rate mean={rate.mean():.4f} max={rate.max():.3f}", flush=True)
    if y.sum() < 10 or y.sum() == len(y):
        raise SystemExit(f"[abort] degenerate label: {int(y.sum())} positives of {len(y)}")

    mu, sd = P.mean(0), P.std(0) + 1e-9
    Ps = (P - mu) / sd
    per_layer_auc = np.array([roc_auc_score(y, P[:, L]) for L in range(nL)])
    best_layer = int(np.nanargmax(per_layer_auc))
    print(f"[per-layer absolute AUC vs broad refusal] best L{best_layer}={per_layer_auc[best_layer]:.4f}", flush=True)

    # CV-NNLS stack; EXCLUDE L0 (degenerate constant embedding column)
    cols = [L for L in range(nL) if L != 0]
    rng = np.random.RandomState(a.seed)
    order = rng.permutation(len(y))
    folds = np.array_split(order, a.folds)
    oof = np.zeros(len(y))
    for f in range(a.folds):
        te = folds[f]; tr = np.concatenate([folds[j] for j in range(a.folds) if j != f])
        wj = nnls_stack(Ps[tr][:, cols], y[tr].astype(float))
        oof[te] = Ps[te][:, cols] @ wj
    ens_auc = roc_auc_score(y, oof)
    w_cols = nnls_stack(Ps[:, cols], y.astype(float))
    w = np.zeros(nL); w[cols] = w_cols

    top = sorted(((float(w[L]), L) for L in range(nL)), reverse=True)[:6]
    collapse = max(w) / (w.sum() + 1e-9)
    print("\n========== absolute-space probe (broad behavioral label) ==========")
    print(f"  best single layer L{best_layer}  AUC={per_layer_auc[best_layer]:.4f}")
    print(f"  ensemble (CV-NNLS, L0 excluded)  AUC={ens_auc:.4f}  (adds {ens_auc-per_layer_auc[best_layer]:+.4f})")
    print(f"  Spearman(ensemble_oof, rate) = {spearmanr(oof, rate).statistic:.4f}")
    print("  top stack weights: " + ", ".join(f"L{L}={ww:.3f}" for ww, L in top))
    print(f"  weight concentration on top layer: {collapse:.3f}  ({'COLLAPSES to ~1 layer' if collapse>0.8 else 'multi-layer'})")

    np.savez(a.out_npz, d=d, dn=dn, mu=mu, sd=sd, w=w, best_layer=best_layer,
             per_layer_auc=per_layer_auc, ensemble_auc=ens_auc)
    json.dump({"best_layer": best_layer, "best_single_auc": float(per_layer_auc[best_layer]),
               "ensemble_auc": float(ens_auc), "positives": int(y.sum()), "n": int(len(y)),
               "top_weights": {f"L{L}": float(w[L]) for _, L in top},
               "weight_concentration": float(collapse)},
              open(a.out_npz.replace(".npz", "_summary.json"), "w"), indent=2)
    print(f"\n[done] wrote {a.out_npz}")


if __name__ == "__main__":
    main()
