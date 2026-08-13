#!/usr/bin/env python3
"""P5 — Emergent topic map for the Refusal Atlas (peer-reviewed clustering redesign).

Fixed axis (OR-Bench-10) lives elsewhere; this builds the EMERGENT axis:

  embed (BGE-large, MiniLM check) -> dedup>0.9 -> over-cluster k=20..40 with
  bootstrap-Jaccard stability (k-means primary, Ward agreement) -> c-TF-IDF keywords
  -> validate (AMI/ARI vs OR-10; per-cluster Wilson CI + binomial + BH-FDR;
  source-composition control) -> overlay each model's judge-refusal + each signal,
  and test whether a signal REORDERS the topics vs behavior (Spearman).

Rewrites (is_rewrite=1) are EXCLUDED from clustering (one-word near-dups inflate
density and bias cluster refusal rates); each is assigned to its original's cluster
post-hoc and its contrast is routed to P6. Clustered ONCE on prompt embeddings
(model-independent); model/signal rates are overlaid on the fixed partition.

Deterministic (fixed seeds). CPU-only. LLM cluster labels are a separate opt-in
(--llm_label) so this never spends autonomously.
"""
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np

SEED = 20260811
PAIR_SOURCES = ("single_edit_pair", "sonnet_pair")
OR10 = ("deception", "harassment", "harmful", "hate", "illegal",
        "privacy", "self-harm", "sexual", "unethical", "violence")


# --------------------------------------------------------------------- IO
def load_substrate(path):
    return list(csv.DictReader(open(path)))


def load_scores(sig_csv, judge_csv):
    """prompt_id -> {vector, probe, logit_sum, judge}  (judge_refuse_rate)."""
    d = {}
    for r in csv.DictReader(open(sig_csv)):
        d[r["prompt_id"]] = {k: float(r[k]) for k in ("vector", "probe", "logit_sum")}
    for r in csv.DictReader(open(judge_csv)):
        if r["prompt_id"] in d:
            d[r["prompt_id"]]["judge"] = float(r["judge_refuse_rate"])
    return d


# --------------------------------------------------------------------- embedding
def embed(texts, model_name, cache_path):
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        if list(z["texts"]) == list(texts):
            print(f"[embed] cache hit {cache_path}")
            return z["emb"].astype(np.float64)
        print(f"[embed] cache stale ({cache_path}) -> recompute")
    from sentence_transformers import SentenceTransformer
    print(f"[embed] loading {model_name} (CPU) ...")
    m = SentenceTransformer(model_name, device="cpu")
    emb = m.encode(texts, batch_size=32, normalize_embeddings=True,
                   convert_to_numpy=True, show_progress_bar=True)
    np.savez(cache_path, emb=emb.astype(np.float32), texts=np.array(texts, dtype=object))
    return emb.astype(np.float64)


# --------------------------------------------------------------------- dedup
def dedup(emb, texts, thr=0.9):
    """Greedy: drop any prompt with cosine>thr to an already-kept prompt (L2-normed emb)."""
    keep, kept_idx = [], []
    sims = emb @ emb.T
    dropped = []
    for i in range(len(texts)):
        dup_of = None
        for j in kept_idx:
            if sims[i, j] > thr:
                dup_of = j
                break
        if dup_of is None:
            kept_idx.append(i); keep.append(True)
        else:
            keep.append(False); dropped.append((i, dup_of, float(sims[i, dup_of])))
    return np.array(keep), dropped


# --------------------------------------------------------------------- clustering + stability
def kmeans_labels(X, k, seed, n_init=10):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit_predict(X)


def jaccard_stability(X, k, B, frac, seed):
    """Clusterwise Jaccard stability (Hennig 2007) by subsampling.
    Returns (ref_labels, {ref_cluster: mean max-Jaccard})."""
    ref = kmeans_labels(X, k, seed)
    n = len(X); rng = np.random.default_rng(seed)
    per = defaultdict(list)
    for b in range(B):
        idx = rng.choice(n, int(frac * n), replace=False)
        lab = kmeans_labels(X[idx], k, seed + b + 1, n_init=5)
        refsub = ref[idx]
        # precompute bootstrap cluster membership sets
        bmemb = {lc: set(np.where(lab == lc)[0]) for lc in np.unique(lab)}
        for rc in np.unique(refsub):
            A = set(np.where(refsub == rc)[0])
            best = max((len(A & Bc) / len(A | Bc)) for Bc in bmemb.values())
            per[rc].append(best)
    return ref, {int(rc): float(np.mean(v)) for rc, v in per.items()}


# --------------------------------------------------------------------- c-TF-IDF keywords
def ctfidf_keywords(labels, texts, topn=8):
    from sklearn.feature_extraction.text import CountVectorizer
    K = int(labels.max()) + 1
    docs = [" ".join(t for t, l in zip(texts, labels) if l == c) for c in range(K)]
    cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.5)
    X = cv.fit_transform(docs).toarray().astype(np.float64)   # (K, V)
    vocab = np.array(cv.get_feature_names_out())
    tf = X / np.maximum(X.sum(axis=1, keepdims=True), 1)       # within-class freq
    f_x = X.sum(axis=0)                                        # total across classes
    A = X.sum(axis=1).mean()                                   # avg words/class
    idf = np.log(1 + A / np.maximum(f_x, 1))                   # BERTopic c-TF-IDF
    ct = tf * idf
    out = {}
    for c in range(K):
        top = np.argsort(ct[c])[::-1][:topn]
        out[c] = [vocab[i] for i in top if ct[c, i] > 0]
    return out


# --------------------------------------------------------------------- stats helpers
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def bh_fdr(pvals):
    p = np.asarray(pvals, float); n = len(p); order = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for rank, i in enumerate(order[::-1]):
        r = n - rank
        prev = min(prev, p[i] * n / r)
        q[i] = prev
    return q


def spearman(a, b):
    from scipy.stats import spearmanr
    r, pv = spearmanr(a, b)
    return float(r), float(pv)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", default="data/substrate.csv")
    ap.add_argument("--outdir", default="results/clusters")
    ap.add_argument("--bge", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--minilm", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k_lo", type=int, default=20)
    ap.add_argument("--k_hi", type=int, default=40)
    ap.add_argument("--sweep_B", type=int, default=40)
    ap.add_argument("--final_B", type=int, default=150)
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--stability_report", type=float, default=0.75)
    ap.add_argument("--dedup_thr", type=float, default=0.9)
    ap.add_argument("--skip_minilm", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    rows = load_substrate(a.substrate)
    scores = {"llama": load_scores("results/signals_llama.csv", "results/judge_llama.csv"),
              "qwen":  load_scores("results/signals_qwen.csv",  "results/judge_qwen.csv")}

    # clustering candidates = all EXCEPT rewrites; keep pair originals + non-pair
    cand = [r for r in rows if not (r["source"] in PAIR_SOURCES and r.get("is_rewrite") == "1")]
    # unique by text (candidates are already unique, but be safe)
    seen, cand_u = set(), []
    for r in cand:
        if r["text"] not in seen:
            seen.add(r["text"]); cand_u.append(r)
    cand = cand_u
    texts = [r["text"] for r in cand]
    print(f"[data] {len(rows)} rows -> {len(cand)} clustering candidates (rewrites held out)")

    emb = embed(texts, a.bge, os.path.join(a.outdir, "emb_bge.npz"))

    keep, dropped = dedup(emb, texts, a.dedup_thr)
    print(f"[dedup] cosine>{a.dedup_thr}: dropped {len(dropped)} near-duplicates -> {int(keep.sum())} kept")
    cand = [r for r, kp in zip(cand, keep) if kp]
    texts = [r["text"] for r in cand]
    X = emb[keep]

    # ---- choose k by mean bootstrap-Jaccard stability
    print(f"[k-sweep] k={a.k_lo}..{a.k_hi} (B={a.sweep_B}, frac={a.frac}) ...")
    sweep = {}
    for k in range(a.k_lo, a.k_hi + 1):
        _, stab = jaccard_stability(X, k, a.sweep_B, a.frac, SEED)
        sweep[k] = float(np.mean(list(stab.values())))
        print(f"   k={k:2d}  mean-Jaccard={sweep[k]:.3f}")
    best_k = max(sweep, key=sweep.get)
    print(f"[k-sweep] chosen k={best_k} (mean-Jaccard={sweep[best_k]:.3f})")

    # ---- final partition + per-cluster stability
    ref, stab = jaccard_stability(X, best_k, a.final_B, a.frac, SEED)
    labels = ref
    keywords = ctfidf_keywords(labels, texts)

    # ---- Ward agreement + MiniLM embedder-robustness
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI
    ward = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X)
    robustness = {"kmeans_vs_ward_AMI": float(AMI(labels, ward)),
                  "kmeans_vs_ward_ARI": float(ARI(labels, ward))}
    if not a.skip_minilm:
        emb_m = embed(texts, a.minilm, os.path.join(a.outdir, "emb_minilm.npz"))
        lab_m = kmeans_labels(emb_m, best_k, SEED)
        robustness["kmeans_bge_vs_minilm_AMI"] = float(AMI(labels, lab_m))
        robustness["kmeans_bge_vs_minilm_ARI"] = float(ARI(labels, lab_m))

    # ---- AMI/ARI vs OR-Bench-10 (only prompts that carry a native OR-10 topic)
    or_mask = [i for i, r in enumerate(cand) if r["native_topic"] in OR10]
    or_true = [cand[i]["native_topic"] for i in or_mask]
    or_pred = [labels[i] for i in or_mask]
    vs_or10 = {"n": len(or_mask),
               "AMI": float(AMI(or_true, or_pred)) if or_mask else None,
               "ARI": float(ARI(or_true, or_pred)) if or_mask else None}

    # ---- per-cluster behavioral refusal (judge) + Wilson + binomial + BH, per model
    #      source-composition control alongside.
    from scipy.stats import binomtest
    K = best_k
    pid = [r["prompt_id"] for r in cand]
    src = [r["source"] for r in cand]
    cluster_rows = []
    # global base rate per model over candidates
    base = {}
    for mdl in ("llama", "qwen"):
        vals = [scores[mdl][p]["judge"] for p in pid if "judge" in scores[mdl].get(p, {})]
        base[mdl] = float(np.mean([v >= 0.5 for v in vals]))
    per_model_p = {mdl: [] for mdl in ("llama", "qwen")}
    tmp = {mdl: [] for mdl in ("llama", "qwen")}
    for c in range(K):
        idx = [i for i in range(len(cand)) if labels[i] == c]
        comp = {s: sum(1 for i in idx if src[i] == s) for s in set(src)}
        row = {"cluster": c, "size": len(idx), "jaccard": round(stab.get(c, float("nan")), 3),
               "keywords": ", ".join(keywords.get(c, [])[:8]),
               "source_comp": comp}
        for mdl in ("llama", "qwen"):
            jb = [scores[mdl][pid[i]]["judge"] >= 0.5 for i in idx if "judge" in scores[mdl].get(pid[i], {})]
            n = len(jb); kk = int(sum(jb))
            p, lo, hi = wilson_ci(kk, n)
            bt = binomtest(kk, n, base[mdl]).pvalue if n else 1.0
            row[f"{mdl}_refuse"] = round(p, 3); row[f"{mdl}_ci"] = [round(lo, 3), round(hi, 3)]
            row[f"{mdl}_n"] = n
            per_model_p[mdl].append(bt)
            # signal means (overlay)
            for s in ("vector", "probe", "logit_sum"):
                sv = [scores[mdl][pid[i]][s] for i in idx if pid[i] in scores[mdl]]
                row[f"{mdl}_{s}"] = round(float(np.mean(sv)), 4) if sv else None
            tmp[mdl].append(p)
        cluster_rows.append(row)
    # BH-FDR across clusters, per model
    for mdl in ("llama", "qwen"):
        q = bh_fdr(per_model_p[mdl])
        for c in range(K):
            cluster_rows[c][f"{mdl}_binom_q"] = round(float(q[c]), 4)

    # ---- source-composition confound: does cluster refusal track OR-Bench-Hard share?
    hard_frac = [ (r["source_comp"].get("orbench_hard", 0) / r["size"]) for r in cluster_rows ]
    src_confound = {}
    for mdl in ("llama", "qwen"):
        rr = [r[f"{mdl}_refuse"] for r in cluster_rows]
        rho, pv = spearman(hard_frac, rr)
        src_confound[mdl] = {"spearman_refuse_vs_hardfrac": round(rho, 3), "p": round(pv, 4)}

    # ---- SIGNAL REORDERING (the payoff): across clusters, does each signal rank
    #      topics like behavior? low rho / reordering vs behavior = "boundary by signal".
    reorder = {}
    for mdl in ("llama", "qwen"):
        beh = [r[f"{mdl}_refuse"] for r in cluster_rows]
        reorder[mdl] = {}
        for s in ("vector", "probe", "logit_sum"):
            sig = [r[f"{mdl}_{s}"] for r in cluster_rows]
            rho, pv = spearman(sig, beh)
            reorder[mdl][s] = {"spearman_vs_behavior": round(rho, 3), "p": round(pv, 4)}

    # ---- post-hoc: assign held-out rewrites to their original's cluster
    text2cluster = {t: int(l) for t, l in zip(texts, labels)}
    orig_by_pair = {r["pair_id"]: r["text"] for r in rows
                    if r["source"] in PAIR_SOURCES and r.get("is_rewrite") == "0"}
    assign = []
    for r in rows:
        c = None
        if r["text"] in text2cluster:
            c = text2cluster[r["text"]]
        elif r["source"] in PAIR_SOURCES and r.get("is_rewrite") == "1":
            ot = orig_by_pair.get(r["pair_id"])
            c = text2cluster.get(ot)
        assign.append({"prompt_id": r["prompt_id"], "source": r["source"],
                       "native_topic": r["native_topic"], "pair_id": r["pair_id"],
                       "is_rewrite": r.get("is_rewrite", ""), "cluster": ("" if c is None else c)})

    # ---- write outputs
    with open(os.path.join(a.outdir, "assignments.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(assign[0].keys())); w.writeheader(); w.writerows(assign)
    with open(os.path.join(a.outdir, "cluster_summary.json"), "w") as f:
        json.dump(cluster_rows, f, indent=1)
    # flat CSV for eyeballing
    with open(os.path.join(a.outdir, "cluster_summary.csv"), "w", newline="") as f:
        cols = ["cluster", "size", "jaccard", "llama_refuse", "llama_binom_q", "qwen_refuse",
                "qwen_binom_q", "keywords"]
        w = csv.writer(f); w.writerow(cols)
        for r in sorted(cluster_rows, key=lambda x: -x["llama_refuse"]):
            w.writerow([r[c] for c in cols])
    report = {"k_chosen": best_k, "k_sweep": sweep, "n_clustered": len(cand),
              "n_dropped_dup": len(dropped), "stability_report_thr": a.stability_report,
              "n_clusters_stable": int(sum(v >= a.stability_report for v in stab.values())),
              "robustness": robustness, "vs_or10": vs_or10,
              "source_confound": src_confound, "signal_reordering": reorder,
              "base_refuse": base}
    with open(os.path.join(a.outdir, "validation.json"), "w") as f:
        json.dump(report, f, indent=1)

    # ---- console summary
    print("\n================= P5 SUMMARY =================")
    print(f"k={best_k} | clustered={len(cand)} | dropped-dup={len(dropped)} | "
          f"stable clusters (J>= {a.stability_report}): {report['n_clusters_stable']}/{best_k}")
    print(f"vs OR-10: AMI={vs_or10['AMI']:.3f} ARI={vs_or10['ARI']:.3f} (n={vs_or10['n']})")
    print(f"robustness: {robustness}")
    print(f"source confound (refuse vs OR-Hard share): {src_confound}")
    print("signal reordering (Spearman signal-mean vs behavior across clusters):")
    for mdl in ("llama", "qwen"):
        print(f"  {mdl}: " + "  ".join(f"{s}={reorder[mdl][s]['spearman_vs_behavior']:+.3f}"
                                       for s in ('vector', 'probe', 'logit_sum')))
    print(f"\nwrote -> {a.outdir}/ (assignments.csv, cluster_summary.{{json,csv}}, validation.json)")


if __name__ == "__main__":
    main()
