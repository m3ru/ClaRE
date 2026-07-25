#!/usr/bin/env python3
"""Score rewrite pools with the FITTED Qwen delta-probe scorer -> Qwen-OR.

Unlike score_probe_or.py (which re-derives directions + re-fits the ensemble on the
scored pairs' own dP), this LOADS the scorer fitted on the behavioral set
(probe_qwen_signals.py -> qwen_probe_raw.npz: d, dn, mu, sd, w) and APPLIES it to
new pairs -- we have no Qwen dP for the pool.

  probe_delta(pair) = sum_L w_L * (proj_L(rw) - proj_L(orig) - mu_L) / sd_L
  Qwen-OR           = exp(k*(sim - c)) * probe_delta        (k=18.4, c=0.75; sim = MiniLM, reused from pool)

CRITICAL: the forward-pass read MUST match extract_layer_acts.py at fit time
(right-pad, attention_mask.sum-1 last-real-token, add_special_tokens=False,
enable_thinking=False, system="You are a helpful assistant.") or mu/sd/d don't apply.
Streams projections (scalars) -- activations never stored.

Writes <out> (scored CSV: original, rewrite, similarity, probe_delta, qwen_or,
p_rw_llama, or_llama) and prints the distribution + top examples for inspection.
"""
import argparse
import csv
import os
import time

import numpy as np

from probe_ensemble import project   # same projection the scorer was built with

SYSTEM = "You are a helpful assistant."


def read_pairs(paths):
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
            seen.add(k)          # only after a valid parse, so a bad-sim row can't block a later good dup
            p_llama = r.get("p_icannot_rewrite", "") or r.get("p_rw", "")
            or_llama = r.get("or_score", "")
            rows.append({"original": o, "rewrite": rw, "similarity": sim,
                         "p_rw_llama": p_llama, "or_llama": or_llama})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", nargs="+", required=True)
    ap.add_argument("--scorer", required=True, help="qwen_probe_raw.npz (d, dn, mu, sd, w)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen3-32B")
    ap.add_argument("--k", type=float, default=18.4)
    ap.add_argument("--c", type=float, default=0.75)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    z = np.load(args.scorer)
    d, dn, mu, sd, w = z["d"], z["dn"], z["mu"], z["sd"], z["w"]
    best_layer = int(z["best_layer"]) if "best_layer" in z else int(np.argmax(w))
    print(f"[scorer] layers={d.shape[0]} best_layer=L{best_layer} nonzero_weights={(w>1e-4).sum()}", flush=True)

    pairs = read_pairs(args.pairs_csv)
    uniq = list(dict.fromkeys([p["original"] for p in pairs] + [p["rewrite"] for p in pairs]))
    print(f"[data] {len(pairs)} unique pairs | {len(uniq)} unique texts to project", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"          # match extract_layer_acts: last real token = attn_mask.sum-1
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, token=os.environ.get("HF_TOKEN"), torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="eager").eval()

    def fmt(p):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)

    proj = {}
    n_trunc = 0
    t0 = time.time()
    for i in range(0, len(uniq), args.batch_size):
        batch = uniq[i:i + args.batch_size]
        enc = tok([fmt(p) for p in batch], return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_length, add_special_tokens=False).to(model.device)
        last = enc["attention_mask"].sum(1) - 1
        n_trunc += int((enc["attention_mask"].sum(1) >= args.max_length).sum())  # right-pad: full row => truncated
        rows = torch.arange(len(batch), device=model.device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        acts = torch.stack([h[rows, last, :] for h in hs], dim=1).float().cpu().numpy()  # [B, nL+1, H]
        pr = project(acts, d, dn)       # [B, nL+1]
        for j, txt in enumerate(batch):
            proj[txt] = pr[j]
        if (i // args.batch_size) % 40 == 0:
            print(f"  [proj {i + len(batch)}/{len(uniq)}] {(i+len(batch))/max(time.time()-t0,1e-3):.1f}/s", flush=True)

    if n_trunc:
        print(f"[warn] {n_trunc} text(s) hit max_length={args.max_length} and were truncated "
              f"-> last-token read is off-position for those; consider raising --max_length", flush=True)

    # apply the fitted scorer: standardize the per-layer delta, weighted-sum -> probe_delta
    Dproj = np.array([proj[p["rewrite"]] - proj[p["original"]] for p in pairs])   # [n, nL+1]
    Ds = (Dproj - mu) / sd
    probe_delta = Ds @ w
    sims = np.array([p["similarity"] for p in pairs])
    qwen_or = np.exp(args.k * (sims - args.c)) * probe_delta

    with open(args.out, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["original", "rewrite", "similarity", "probe_delta", "qwen_or", "p_rw_llama", "or_llama"])
        for p, pd, orv in zip(pairs, probe_delta, qwen_or):
            wtr.writerow([p["original"], p["rewrite"], f"{p['similarity']:.4f}",
                          f"{float(pd):.6g}", f"{float(orv):.6g}", p["p_rw_llama"], p["or_llama"]])

    # distribution + top examples
    q = np.quantile(probe_delta, [0.5, 0.9, 0.99])
    oq = np.quantile(qwen_or, [0.5, 0.9, 0.99])
    print(f"\n[summary] n={len(pairs)}")
    print(f"  probe_delta: mean={probe_delta.mean():.4f} p50={q[0]:.4f} p90={q[1]:.4f} p99={q[2]:.4f} %pos={(probe_delta>0).mean()*100:.1f}%")
    print(f"  qwen_or    : mean={qwen_or.mean():.4f} p50={oq[0]:.4f} p90={oq[1]:.4f} p99={oq[2]:.4f}")
    top = np.argsort(-qwen_or)[:8]
    print("\n[top-8 Qwen-OR rewrites]")
    for t in top:
        p = pairs[t]
        print(f"  OR={qwen_or[t]:.2f} pd={probe_delta[t]:.2f} sim={p['similarity']:.3f} :: "
              f"{p['original'][:45]!r} -> {p['rewrite'][:60]!r}")
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
