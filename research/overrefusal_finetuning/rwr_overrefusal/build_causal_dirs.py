#!/usr/bin/env python3
"""Phase A: derive candidate over-refusal directions on a TRAIN split, leakage-free.

Two problems this fixes.

1. LEAKAGE. The earlier frame ablation fitted directions on all or_high pairs and then
   ablated against prompts drawn from that same set. Directions here are fitted on pairs
   whose ORIGINAL falls in the train half; the ablation is evaluated only on originals in
   the held-out half.

2. THE RANK QUESTION IS CAUSAL, NOT CORRELATIONAL. Asking "how many directions separate
   over-refusal from controls" is ill-posed -- for two classes Fisher's LDA has exactly one
   discriminant by construction, which is why the earlier k* returned 1 for injected ranks
   1,2,3 and 5. The well-posed version is "how many must you ABLATE before the behaviour
   stops", so this script only proposes an ORDERED, ORTHONORMAL candidate basis and the
   ranking is settled on the GPU by ablation.

Candidate basis:
  d1      = mean(Δ_OR) − mean(Δ_ctrl)      -- the shared axis; the whole rank-1 mean shift
  d2..dk  = residual principal components, ranked by how well each SEPARATES OR from ctrl
            (|AUC−0.5|) rather than by variance, since the largest residual variance is
            length/style that both classes share.
All Gram-Schmidt orthonormalised, so a rank-k ablation removes exactly k dimensions.

Run: python build_causal_dirs.py --out probe_or/results/delta/causal_dirs.npz
"""
import argparse, csv, json, os, sys
import numpy as np
csv.field_size_limit(sys.maxsize)


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def auc(s, y):
    o = np.argsort(s); r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    if n1 == 0 or n0 == 0:
        return 0.5
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def randomized_pca(X, k, seed=0, over=24):
    rng = np.random.default_rng(seed)
    mu = X.mean(0); Xc = X - mu
    k = int(min(k, min(Xc.shape) - 1))
    Q, _ = np.linalg.qr(Xc @ rng.standard_normal((Xc.shape[1], k + over)))
    for _ in range(2):
        Q, _ = np.linalg.qr(Xc.T @ Q); Q, _ = np.linalg.qr(Xc @ Q)
    _, _, Vt = np.linalg.svd(Q.T @ Xc, full_matrices=False)
    return Vt[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta_dir", default="probe_or/results/delta")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n_pca", type=int, default=60)
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="probe_or/results/delta/causal_dirs.npz")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    acts = np.load(os.path.join(a.delta_dir, "acts.npy"), mmap_mode="r")
    idx = json.load(open(os.path.join(a.delta_dir, "prompt_index.json")))
    rows = list(csv.DictReader(open(os.path.join(a.delta_dir, "prompt_sets.csv"))))
    A = np.asarray(acts[:, a.layer, :], dtype=np.float64)      # one layer, all prompts

    def vec(t):
        return A[idx[t.strip()]]

    # ---- split by ORIGINAL, never by pair ----
    origs = sorted({r["original"].strip() for r in rows
                    if r["set"].startswith(("or_", "ctrl_")) and r["original"].strip()})
    perm = rng.permutation(len(origs))
    train = {origs[i] for i in perm[: int(len(origs) * a.train_frac)]}
    print(f"[dirs] {len(origs)} originals -> {len(train)} train / {len(origs)-len(train)} held out",
          flush=True)

    D_or, D_ct, heldout = [], [], []
    for r in rows:
        s = r["set"]
        o, w = r["original"].strip(), r["rewrite"].strip()
        if not o or o not in idx or w not in idx:
            continue
        d = vec(w) - vec(o)
        if o in train:
            if s.startswith("or_"):
                D_or.append(d)
            elif s.startswith("ctrl_"):
                D_ct.append(d)
        elif s.startswith("or_"):
            heldout.append((o, w))
    D_or, D_ct = np.array(D_or), np.array(D_ct)
    print(f"[dirs] train: {len(D_or)} OR / {len(D_ct)} ctrl | held-out OR pairs: {len(heldout)}",
          flush=True)

    # ---- d1: the shared axis (the entire rank-1 mean shift) ----
    d1 = unit(D_or.mean(0) - D_ct.mean(0))
    dirs = [d1]

    # ---- frame-residual candidates, refit on TRAIN only ----
    # The most causally potent direction found so far was a frame residual, not a principal
    # component -- and it was fitted with leakage, on the same pool it was evaluated against.
    # Refitting it here train-only puts it in the basis so the causal sweep can test it
    # honestly. Frame residuals are only weakly discriminative (that is the point: the earlier
    # correlational screen would have discarded the direction that turned out to matter most),
    # so they are placed in the basis by construction rather than selected by AUC.
    from analyze_frames import frames_of, FRAMES
    from analyze_edit_distance import pair_metrics
    fr_pairs = {f: [] for f in FRAMES}
    for r in rows:
        if not r["set"].startswith("or_"):
            continue
        o, w = r["original"].strip(), r["rewrite"].strip()
        if o not in train or o not in idx or w not in idx:
            continue
        for f in frames_of(pair_metrics(o, w)["introduced_words"]):
            fr_pairs[f].append(vec(w) - vec(o))
    mu_ct = D_ct.mean(0)
    frame_cands = []
    for f in sorted(fr_pairs, key=lambda k: -len(fr_pairs[k])):
        if len(fr_pairs[f]) < 12:
            continue
        u = unit(np.mean(fr_pairs[f], axis=0) - mu_ct)
        u = u - (u @ d1) * d1                       # residual: orthogonal to the shared axis
        n = np.linalg.norm(u)
        if n > 1e-6:
            frame_cands.append((f, len(fr_pairs[f]), u / n))
    print("[dirs] frame residuals (train-only): " +
          ", ".join(f"{f}(n={n})" for f, n, _ in frame_cands), flush=True)
    for f, n_, v in frame_cands:
        if len(dirs) >= a.k:
            break
        for d in dirs:
            v = v - (v @ d) * d
        nn = np.linalg.norm(v)
        if nn > 1e-6:
            dirs.append(v / nn)

    # ---- residual PCs, ranked by separation rather than variance ----
    Ro = D_or - np.outer(D_or @ d1, d1)
    Rc = D_ct - np.outer(D_ct @ d1, d1)
    V = randomized_pca(np.vstack([Ro, Rc]), a.n_pca, a.seed)
    y = np.r_[np.ones(len(Ro)), np.zeros(len(Rc))]
    scored = []
    for i in range(V.shape[0]):
        s = np.r_[Ro @ V[i], Rc @ V[i]]
        scored.append((abs(auc(s, y) - 0.5), i))
    scored.sort(reverse=True)
    print("[dirs] top residual PCs by |AUC-0.5|: " +
          ", ".join(f"PC{i}={g:.3f}" for g, i in scored[:6]), flush=True)

    for _, i in scored:
        if len(dirs) >= a.k:
            break
        v = V[i].copy()
        for d in dirs:                       # Gram-Schmidt
            v = v - (v @ d) * d
        n = np.linalg.norm(v)
        if n > 1e-6:
            dirs.append(v / n)
    Dm = np.array(dirs)
    off = np.abs(Dm @ Dm.T - np.eye(len(Dm))).max()
    print(f"[dirs] {len(Dm)} directions, max off-orthonormality {off:.2e}", flush=True)

    ho_o = [o for o, _ in heldout]
    ho_w = [w for _, w in heldout]
    np.savez(a.out, dirs=Dm, layer=a.layer,
             heldout_originals=np.array(ho_o, dtype=object),
             heldout_rewrites=np.array(ho_w, dtype=object),
             train_originals=np.array(sorted(train), dtype=object))
    print(f"[dirs] wrote {a.out}")


if __name__ == "__main__":
    main()
