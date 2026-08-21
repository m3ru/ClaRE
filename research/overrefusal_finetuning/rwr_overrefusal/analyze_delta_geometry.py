#!/usr/bin/env python3
"""Phase 2: how many directions separate over-refusal from matched non-refusal?

Delta = h(rewrite) - h(original) cancels topic. Delta' = h(rewrite_OR) - h(rewrite_ctrl) for
rewrites of the SAME original in the SAME edit band, which also cancels the attacker's house
style. Both are reported; Delta' is the cleaner one and wins any disagreement.

The headline statistic is a DISCRIMINATIVE RANK, not a variance statistic. Participation ratio
over the Delta spectrum answers "how spread is Delta's total variance", and Delta's variance is
dominated by nuisance (length, style, residual topic) that OR and control share. So instead:

  1. fit a linear discriminant separating OR-Delta from control-Delta,
  2. project the data off that direction, refit, repeat -> orthogonal d1..dk,
  3. k* = smallest k reaching 95% of the AUC achievable with the full space,
  4. fit on TRAIN originals, score on HELD-OUT originals -- otherwise k* measures fit capacity
     and grows with n.

Grouping is by ORIGINAL everywhere (rewrites of one original are not independent).

Run: python analyze_delta_geometry.py --report HIGH_EDIT_GEOMETRY.md
"""
import argparse, csv, json, os, sys
import numpy as np
csv.field_size_limit(sys.maxsize)


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)


def auc(scores, labels):
    """Rank-based AUC; no sklearn dependency."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels == 1, labels == 0
    n1, n0 = pos.sum(), neg.sum()
    if n1 == 0 or n0 == 0:
        return 0.5
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def randomized_pca(X, k, seed=0, oversample=24):
    """Top-k right singular vectors of centred X. Randomized, because a full 4096-wide SVD per
    layer per contrast costs minutes and we only need the leading subspace."""
    rng = np.random.default_rng(seed)
    mu = X.mean(0)
    Xc = X - mu
    k = int(min(k, min(Xc.shape) - 1))
    Om = rng.standard_normal((Xc.shape[1], k + oversample))
    Y = Xc @ Om
    Q, _ = np.linalg.qr(Y)
    for _ in range(2):                       # power iterations sharpen the spectrum
        Q, _ = np.linalg.qr(Xc.T @ Q)
        Q, _ = np.linalg.qr(Xc @ Q)
    B = Q.T @ Xc
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    return mu, Vt[:k]                        # [k, d], orthonormal rows


def shrunk_lda(X, y, shrink=0.15):
    """Regularised linear discriminant, run in the PCA subspace by the caller.

    Shrinkage is not optional: with 4096 raw dimensions and ~2k samples the empirical
    covariance is singular, and an unregularised LDA would fit noise exactly and report a
    k* that measures sample size rather than structure. Working in a ~256-dim PCA basis
    makes the estimate well-conditioned as well as fast."""
    m1, m0 = X[y == 1].mean(0), X[y == 0].mean(0)
    Xc = np.vstack([X[y == 1] - m1, X[y == 0] - m0])
    C = (Xc.T @ Xc) / max(len(Xc) - 2, 1)
    C.flat[:: C.shape[0] + 1] += shrink * np.trace(C) / C.shape[0]
    try:
        w = np.linalg.solve(C, m1 - m0)
    except np.linalg.LinAlgError:
        w = m1 - m0
    return w / (np.linalg.norm(w) + 1e-9)


def deflating_rank(Xtr, ytr, Xte, yte, kmax=8, shrink=0.15, n_pca=256, seed=0):
    """Sequence of orthogonal discriminants; held-out AUC after each. Returns (aucs, dirs).

    PCA basis is fit on TRAIN ONLY (no test leakage), then everything happens in that basis;
    returned directions are mapped back to the full residual-stream space so Phase 4 can
    ablate them directly."""
    mu, V = randomized_pca(Xtr, n_pca, seed)
    Xtr = (Xtr - mu) @ V.T
    Xte = (Xte - mu) @ V.T
    Xtr0, Xte0 = Xtr.copy(), Xte.copy()      # undeflated, for the per-k refits below
    dirs, aucs = [], []
    for _ in range(kmax):
        w = shrunk_lda(Xtr, ytr, shrink)
        for d in dirs:                                  # orthogonalise against earlier ones
            w = w - (w @ d) * d
        nw = np.linalg.norm(w)
        if nw < 1e-8:
            break
        w = w / nw
        dirs.append(w)
        Xtr = Xtr - np.outer(Xtr @ w, w)                # deflate
        Xte = Xte - np.outer(Xte @ w, w)

    # Held-out AUC of a classifier REFIT on the first k directions, for each k. Accumulating
    # raw projections with equal weight (the earlier version) is not a k-dimensional
    # classifier -- extra directions could only ever hurt, which forced k*=1 by construction.
    for k in range(1, len(dirs) + 1):
        W = np.array(dirs[:k]).T                        # [n_pca, k]
        wk = shrunk_lda(Xtr0 @ W, ytr, shrink)
        aucs.append(auc((Xte0 @ W) @ wk, yte))

    # True full-subspace AUC: one discriminant using ALL retained PCA dimensions.
    w_full = shrunk_lda(Xtr0, ytr, shrink)
    auc_full = auc(Xte0 @ w_full, yte)

    D = np.array(dirs) @ V if dirs else np.zeros((0, V.shape[1]))   # back to full space
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    return np.array(aucs), D, auc_full


def participation_ratio(X):
    Xc = X - X.mean(0)
    n = len(Xc)
    if n < 3:
        return float("nan")
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2 / max(n - 1, 1)
    lam = lam[lam > 1e-12]
    return float(lam.sum() ** 2 / (lam ** 2).sum()) if len(lam) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta_dir", default="probe_or/results/delta")
    ap.add_argument("--atlas", default="probe_or/results/llama_signals/probe_absolute.npz")
    ap.add_argument("--layers", default="8,12,17,20,24,28,31")
    ap.add_argument("--head_layer", type=int, default=17)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--shrink", type=float, default=0.15)
    ap.add_argument("--n_pca", type=int, default=256,
                    help="PCA dims the discriminant is fit in; keeps the covariance estimate "
                         "well-conditioned relative to n and makes the sweep tractable")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default="HIGH_EDIT_GEOMETRY.md")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    acts = np.load(os.path.join(a.delta_dir, "acts.npy"), mmap_mode="r")
    idx = json.load(open(os.path.join(a.delta_dir, "prompt_index.json")))
    rows = list(csv.DictReader(open(os.path.join(a.delta_dir, "prompt_sets.csv"))))
    P = np.load(a.atlas, allow_pickle=True)
    r_atlas = unit(P["d"].astype(np.float64))
    layers = [int(x) for x in a.layers.split(",")]

    def vec(t, L):
        return np.asarray(acts[idx[t.strip()], L, :], dtype=np.float64)

    def deltas(setname, L):
        """-> X, originals[] for one set."""
        X, og = [], []
        for r in rows:
            if r["set"] != setname:
                continue
            o, w = r["original"].strip(), r["rewrite"].strip()
            if not o or o not in idx or w not in idx:
                continue
            X.append(vec(w, L) - vec(o, L)); og.append(o)
        return np.array(X), np.array(og)

    def split_by_orig(og, frac=0.5):
        u = np.unique(og); rng.shuffle(u)
        tr = set(u[: int(len(u) * frac)])
        m = np.array([g in tr for g in og])
        return m, ~m

    out = ["# Phase 2 — geometry of the over-refusal shift\n",
           "`Δ = h(rewrite) − h(original)` (cancels topic). "
           "`Δ′ = h(rewrite_OR) − h(rewrite_ctrl)` for the same original in the same edit band "
           "(also cancels the attacker's style). Discriminative rank **k\\*** = smallest number of "
           "orthogonal discriminants reaching 95% of full-space held-out AUC, fit on train "
           "originals and scored on held-out originals. Participation ratio (PR) is reported as "
           "an unsupervised secondary — it is inflated by length/style variance that OR and "
           "control share, which is why it is not the headline.\n"]

    # ---------- headline: discriminative rank, per contrast, per layer ----------
    # NOTE: an "AdvBench vs Alpaca-benign" positive control was removed. Those two sets differ
    # in source, register and length as well as harmfulness, so a discriminant separates them
    # at AUC 1.000 even at layer 8 -- where the actual refusal direction scores 0.466 (chance).
    # It measured dataset provenance, i.e. exactly the unpaired confound this design exists to
    # avoid. It is replaced by synthetic rank recovery below, which validates the ESTIMATOR
    # directly and cannot be confounded by dataset artifacts.
    contrasts = [("HIGH", "or_high", "ctrl_high"), ("LOW", "or_low", "ctrl_low")]
    res = {}
    out += ["\n## Discriminative rank k\\* (held-out)\n",
            "| contrast | layer | n(OR) | n(ctrl) | AUC k=1 | AUC full | k\\* | PR(Δ) |",
            "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for name, s_or, s_ct in contrasts:
        for L in layers:
            if s_or.startswith("adv"):
                Xo = np.array([vec(r["rewrite"], L) for r in rows if r["set"] == s_or])
                Xc = np.array([vec(r["rewrite"], L) for r in rows if r["set"] == s_ct])
                og = np.arange(len(Xo)).astype(str); ogc = np.arange(len(Xc)).astype(str)
            else:
                Xo, og = deltas(s_or, L); Xc, ogc = deltas(s_ct, L)
            if len(Xo) < 20 or len(Xc) < 20:
                continue
            X = np.vstack([Xo, Xc]); y = np.r_[np.ones(len(Xo)), np.zeros(len(Xc))]
            g = np.r_[og, ogc]
            tr, te = split_by_orig(g)
            if y[tr].sum() < 10 or y[te].sum() < 10:
                continue
            aucs, dirs, full = deflating_rank(X[tr], y[tr], X[te], y[te], a.kmax, a.shrink,
                                              a.n_pca, a.seed)
            target = 0.5 + 0.95 * (full - 0.5)
            kstar = int(np.argmax(aucs >= target) + 1) if (aucs >= target).any() else a.kmax
            res[(name, L)] = dict(aucs=aucs.tolist(), kstar=kstar, n_or=len(Xo), n_ct=len(Xc),
                                  dirs=dirs, X=X, y=y, g=g)
            out.append(f"| {name} | {L} | {len(Xo)} | {len(Xc)} | {aucs[0]:.3f} | {full:.3f} | "
                       f"**{kstar}** | {participation_ratio(Xo):.1f} |")

    # ---------- positive control: can the estimator recover a KNOWN rank? ----------
    out += ["\n## Positive control — synthetic rank recovery\n",
            "Real control Δs (no refusal signal) with a known r-dimensional perturbation added "
            "to half of them, at an effect size matched to the observed HIGH separation. "
            "If k\\* does not track r, the estimator is not measuring dimensionality and no k\\* "
            "above is interpretable.\n",
            "| true rank r | recovered k\\* | AUC k=1 | AUC full |", "|---|--:|--:|--:|"]
    base = np.array([d for d, in [(x,) for x in
                                  [v for v, _, _ in []]]]) if False else None
    Xc, ogc = deltas("ctrl_high", a.head_layer)
    for r in (1, 2, 3, 5):
        n = len(Xc) // 2
        A, B = Xc[:n].copy(), Xc[n:2 * n].copy()
        G = rng.standard_normal((r, Xc.shape[1]))
        G, _ = np.linalg.qr(G.T); G = G.T[:r]
        scale = 0.6 * np.linalg.norm(Xc - Xc.mean(0), axis=1).mean() / np.sqrt(r)
        A = A + (rng.standard_normal((n, r)) * scale + scale) @ G
        X = np.vstack([A, B]); y = np.r_[np.ones(n), np.zeros(len(B))]
        g = np.r_[ogc[:n], ogc[n:2 * n]]
        tr, te = split_by_orig(g)
        if y[tr].sum() < 10 or y[te].sum() < 10:
            continue
        aucs, _, full = deflating_rank(X[tr], y[tr], X[te], y[te], a.kmax, a.shrink,
                                       a.n_pca, a.seed)
        target = 0.5 + 0.95 * (full - 0.5)
        k = int(np.argmax(aucs >= target) + 1) if (aucs >= target).any() else a.kmax
        out.append(f"| {r} | **{k}** | {aucs[0]:.3f} | {full:.3f} |")

    # ---------- alignment with the known refusal direction ----------
    out += ["\n## Alignment of Δ with the atlas refusal direction\n",
            "cos(Δ, r̂) per pair, and the share of ‖Δ‖ the r̂ component explains. "
            "r̂ is fit on harmful-vs-harmless prompts, so this is a transfer test.\n",
            "| set | layer | mean cos(Δ, r̂) | median | frac ‖Δ‖ on r̂ |", "|---|--:|--:|--:|--:|"]
    for s in ("or_high", "ctrl_high", "or_low", "ctrl_low"):
        for L in (a.head_layer, 31):
            X, _ = deltas(s, L)
            if not len(X):
                continue
            c = unit(X) @ r_atlas[L]
            frac = np.abs(X @ r_atlas[L]) / (np.linalg.norm(X, axis=1) + 1e-9)
            out.append(f"| {s} | {L} | {c.mean():+.3f} | {np.median(c):+.3f} | {frac.mean():.3f} |")

    # ---------- Delta' : within-original, within-band ----------
    out += ["\n## Δ′ — within-original contrast\n"]
    groups = {}
    for r in rows:
        if r["pair_group"]:
            groups.setdefault(r["pair_group"], {})[r["set"]] = r
    ok = [g for g, d in groups.items() if len(d) == 2]
    out.append(f"Matched groups available: **{len(ok)}**\n")
    out += ["| band | layer | n | AUC k=1 | AUC full | k\\* |", "|---|--:|--:|--:|--:|--:|"]
    for band, ors, cts in (("HIGH", "or_high", "ctrl_high"), ("LOW", "or_low", "ctrl_low")):
        for L in (a.head_layer, 31):
            D, og = [], []
            for g in ok:
                d = groups[g]
                if ors not in d or cts not in d:
                    continue
                a_, b_ = d[ors]["rewrite"].strip(), d[cts]["rewrite"].strip()
                if a_ in idx and b_ in idx:
                    D.append(vec(a_, L) - vec(b_, L)); og.append(d[ors]["original"].strip())
            D = np.array(D)
            if len(D) < 40:
                out.append(f"| {band} | {L} | {len(D)} | — | — | too few |"); continue
            # Delta' has no natural negative class -> use sign-flip: -D is a valid "control"
            X = np.vstack([D, -D]); y = np.r_[np.ones(len(D)), np.zeros(len(D))]
            g2 = np.r_[np.array(og), np.array(og)]
            tr, te = split_by_orig(g2)
            aucs, _, full = deflating_rank(X[tr], y[tr], X[te], y[te], a.kmax, a.shrink,
                                           a.n_pca, a.seed)
            target = 0.5 + 0.95 * (full - 0.5)
            k = int(np.argmax(aucs >= target) + 1) if (aucs >= target).any() else a.kmax
            out.append(f"| {band} | {L} | {len(D)} | {aucs[0]:.3f} | {full:.3f} | **{k}** |")

    # ---------- save directions for Phase 4 ----------
    save = {}
    for (name, L), d in res.items():
        if L == a.head_layer and not name.startswith("POSITIVE"):
            save[f"{name}_L{L}"] = d["dirs"]
    if save:
        np.savez(os.path.join(a.delta_dir, "delta_directions.npz"), **save)
        out.append(f"\nDirections for Phase 4 ablation saved to `{a.delta_dir}/delta_directions.npz` "
                   f"({', '.join(save)}).")

    open(a.report, "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nwrote {a.report}")


if __name__ == "__main__":
    main()
