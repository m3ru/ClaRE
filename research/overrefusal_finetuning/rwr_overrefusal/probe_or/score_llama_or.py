#!/usr/bin/env python3
"""Score the harm-filtered Sonnet benign rewrite pool with the THREE corrected
Llama refusal signals, in one GPU pass. Replaces the retired layer-32 refusal-vector
OR score (causally inert layer + all-token mean-pooled extraction + padding bug).

Per (original, rewrite) pair, all in the SAME measurement context as fit time
(system="You are a helpful assistant.", add_generation_prompt=True,
enable_thinking=False, right-pad, last-real-token read, add_special_tokens=False):

  d_vector  RAW delta projection at the CAUSAL layer (L17, validated by ablation
            99->83 / addition 0->99):  proj_17(rw) - proj_17(orig).  Deliberately NOT
            standardized: the retired reward's refusal term was the raw dot-product
            delta, so only the LAYER and the extraction recipe are being corrected.
            (Standardizing would subtract mu_17=0.075 BEFORE the per-pair similarity
            gate multiplies it, which reorders pairs -- Spearman 0.987, not 1.0 --
            and shifts the delta>0 keep-set by ~2%, concentrated exactly in the
            small-delta/high-similarity minimal-edit region k=18.4 exists to promote.)
  d_probe   fitted delta-probe ensemble:  sum_L w_L * (proj_L(rw)-proj_L(orig)-mu_L)/sd_L
  d_logit   multi-phrase refusal-onset delta:  P(reply begins with ANY of the 5 mined
            Llama openers | rw) - P(... | orig)   [teacher-forced, opener_sets.json;
            the old single-phrase "I cannot" label covers only ~57% of Llama refusals]

mu/sd/w/d come from the Aug-10 refit (probe_qwen_signals.py on the 6k broad-behavioral
pairs) -> results/llama_signals/qwen_probe_raw.npz. The stored fit-time standardization
is APPLIED, never recomputed here. The vector layer is FIXED at the causal layer via
--vector_layer (do NOT use the npz best_layer: that is the behavioral-best L31).

For each signal an OR reward is emitted at the k=18.4 similarity gate (the recipe the
Qwen run validated):  or_<sig> = exp(k*(sim - 0.75)) * d_<sig>.  Similarity is reused
from the pool CSV (MiniLM).

Writes <out> CSV: original, rewrite, similarity, d_vector, d_probe, d_logit,
or_vector, or_probe, or_logit, p_icannot_rw, p_icannot_orig -- plus a summary JSON
(distributions + suggested absolute bin edges per signal) next to it.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                            # probe_ensemble
sys.path.insert(0, os.path.join(_HERE, "..", "refusal_atlas"))       # score_signals

# proven passes from the atlas scorer: right-pad last-token projections + teacher-forced
# multi-phrase opener probability (identical measurement context, truncation warnings)
from score_signals import project_texts, logit_openers

SIGNALS = ("vector", "probe", "logit")


def read_pairs(paths):
    """Pool rows (dedup by (original, rewrite)); keeps the old single-phrase icannot
    probabilities for the sanity cross-check."""
    rows, seen = [], set()
    for path in paths:
        for r in csv.DictReader(open(path)):
            o = (r.get("original") or "").strip()
            rw = (r.get("rewrite") or "").strip()
            if not (o and rw):
                continue
            k = (o, rw)
            if k in seen:
                continue
            try:
                sim = float(r["similarity"])
            except (KeyError, ValueError):
                continue
            seen.add(k)
            rows.append({"original": o, "rewrite": rw, "similarity": sim,
                         "p_icannot_rw": r.get("p_icannot_rewrite", ""),
                         "p_icannot_orig": r.get("p_icannot_orig", "")})
    return rows


def suggest_edges(or_scores, sims, deltas, sim_floor, quantiles=(0.35, 0.65, 0.85)):
    """Absolute bin edges for the k=18.4 recipe: quantiles of OR among the pairs the
    trainer will actually keep (sim>=floor & delta>0), mirroring where the Qwen edges
    (0.1,0.5,2 / bins 462/399/303/203) sat in its filtered pool."""
    keep = (sims >= sim_floor) & (deltas > 0)
    kept = or_scores[keep]
    if len(kept) < 20:
        return None, int(keep.sum())
    return [float(np.quantile(kept, q)) for q in quantiles], int(keep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", nargs="+", required=True)
    ap.add_argument("--scorer", required=True,
                    help="results/llama_signals/qwen_probe_raw.npz (d, dn, mu, sd, w; 33 Llama layers)")
    ap.add_argument("--vector_layer", type=int, default=17,
                    help="CAUSAL layer for the vector signal (hidden_states index; Llama=17)")
    ap.add_argument("--opener_json", required=True)
    ap.add_argument("--model_key", default="llama", choices=["llama", "qwen"])
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=float, default=18.4)
    ap.add_argument("--c", type=float, default=0.75)
    ap.add_argument("--sim_floor", type=float, default=0.5, help="only for the edge suggestion")
    ap.add_argument("--proj_bs", type=int, default=16)
    ap.add_argument("--logit_bs", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    z = np.load(args.scorer)
    d, dn, mu, sd, w = z["d"], z["dn"], z["mu"], z["sd"], z["w"]
    nL = d.shape[0]
    assert 0 <= args.vector_layer < nL, f"--vector_layer {args.vector_layer} out of range [0,{nL})"
    if "best_layer" in z and int(z["best_layer"]) != args.vector_layer:
        print(f"[note] npz best_layer=L{int(z['best_layer'])} (behavioral-best) IGNORED; "
              f"vector fixed at causal L{args.vector_layer}", flush=True)
    print(f"[scorer] layers={nL} vector=L{args.vector_layer} probe_nonzero_w={(w > 1e-4).sum()}", flush=True)

    opener_meta = json.load(open(args.opener_json))[args.model_key]
    openers = opener_meta["openers"]
    exp_model = opener_meta.get("model", "")
    if exp_model and exp_model != args.base_model:
        raise SystemExit(f"[abort] opener set '{args.model_key}' is for {exp_model!r} "
                         f"but --base_model={args.base_model!r}")
    print(f"[openers:{args.model_key}] {openers}", flush=True)

    pairs = read_pairs(args.pairs_csv)
    uniq = list(dict.fromkeys([p["original"] for p in pairs] + [p["rewrite"] for p in pairs]))
    print(f"[data] {len(pairs)} unique pairs | {len(uniq)} unique texts", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"      # project_texts asserts right padding (fit-time read)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, token=os.environ.get("HF_TOKEN"), torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="eager").eval()

    print("[passA] per-layer projections...", flush=True)
    proj = project_texts(model, tok, uniq, d, dn, args.proj_bs, args.max_length)

    print("[passB] multi-phrase opener logit...", flush=True)
    lg = logit_openers(model, tok, uniq, openers, args.logit_bs, args.max_length)

    Dproj = np.array([proj[p["rewrite"]] - proj[p["original"]] for p in pairs])   # [n, nL]
    assert np.isfinite(Dproj).all(), "non-finite projections -- check directions / dtype"
    zD = (Dproj - mu) / sd                       # stored fit-time delta standardization (PROBE only)
    d_vector = Dproj[:, args.vector_layer]       # RAW delta at the causal layer -- see header
    d_probe = zD @ w                             # w was fit on standardized deltas; must match
    d_logit = np.array([lg[p["rewrite"]]["sum"] - lg[p["original"]]["sum"] for p in pairs])
    sims = np.array([p["similarity"] for p in pairs])
    gate = np.exp(args.k * (sims - args.c))
    ors = {"vector": gate * d_vector, "probe": gate * d_probe, "logit": gate * d_logit}
    deltas = {"vector": d_vector, "probe": d_probe, "logit": d_logit}

    with open(args.out, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["original", "rewrite", "similarity", "d_vector", "d_probe", "d_logit",
                      "or_vector", "or_probe", "or_logit", "p_icannot_rw", "p_icannot_orig"])
        for i, p in enumerate(pairs):
            wtr.writerow([p["original"], p["rewrite"], f"{p['similarity']:.4f}",
                          f"{d_vector[i]:.6g}", f"{d_probe[i]:.6g}", f"{d_logit[i]:.6g}",
                          f"{ors['vector'][i]:.6g}", f"{ors['probe'][i]:.6g}", f"{ors['logit'][i]:.6g}",
                          p["p_icannot_rw"], p["p_icannot_orig"]])
    print(f"[done] wrote {args.out}", flush=True)

    # ---- sanity: the multi-phrase delta should track (and broaden) the old single-phrase one
    summary = {"n_pairs": len(pairs), "k": args.k, "c": args.c,
               "vector_layer": args.vector_layer, "signals": {}}
    try:
        from scipy.stats import spearmanr
        have_p = np.array([bool(p["p_icannot_rw"]) and bool(p["p_icannot_orig"]) for p in pairs])
        if have_p.sum() > 100:
            d_ic = np.array([float(p["p_icannot_rw"]) - float(p["p_icannot_orig"])
                             for p, h in zip(pairs, have_p) if h])
            rho = spearmanr(d_logit[have_p], d_ic).statistic
            print(f"[sanity] Spearman(d_logit, old single-phrase d_icannot) = {rho:.3f} "
                  f"(n={int(have_p.sum())})", flush=True)
            summary["spearman_dlogit_vs_dicannot"] = float(rho)
        print(f"[sanity] Spearman(d_vector, d_probe) = {spearmanr(d_vector, d_probe).statistic:.3f}",
              flush=True)
    except ImportError:
        print("[warn] scipy unavailable -- skipping sanity correlations", flush=True)

    # ---- per-signal distribution + suggested absolute bin edges for the k=18.4 recipe
    for sig in SIGNALS:
        o, dl = ors[sig], deltas[sig]
        q = {f"p{int(100*x)}": float(np.quantile(o, x)) for x in (0.5, 0.9, 0.99)}
        edges, n_kept = suggest_edges(o, sims, dl, args.sim_floor)
        summary["signals"][sig] = {
            "delta_mean": float(dl.mean()), "delta_pct_pos": float((dl > 0).mean() * 100),
            "or_quantiles": q, "n_trainable": n_kept,
            "suggested_bin_edges": edges,
        }
        e = ",".join(f"{x:.4g}" for x in edges) if edges else "n/a"
        print(f"[{sig}] delta %pos={(dl > 0).mean()*100:.1f}  OR p50={q['p50']:.4g} "
              f"p90={q['p90']:.4g} p99={q['p99']:.4g}  trainable(sim>={args.sim_floor},d>0)={n_kept}"
              f"  suggested --bin_edges {e}", flush=True)
        top = np.argsort(-o)[:5]
        for t in top:
            p = pairs[t]
            print(f"    OR={o[t]:.3g} d={dl[t]:.3g} sim={p['similarity']:.3f} :: "
                  f"{p['original'][:40]!r} -> {p['rewrite'][:55]!r}", flush=True)

    spath = os.path.splitext(args.out)[0] + "_summary.json"
    json.dump(summary, open(spath, "w"), indent=2)
    print(f"[done] wrote {spath}", flush=True)


if __name__ == "__main__":
    main()
