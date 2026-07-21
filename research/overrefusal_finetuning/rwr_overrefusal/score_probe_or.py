#!/usr/bin/env python3
"""Score rewrite pairs with the raw mass-mean DELTA-probe ensemble and emit RWR
shards where refusal_delta = probe_delta (the ensemble score). Wires the probe
into OR (the step probe_or/ left unbuilt), so train_rwr can bin on it.

Pipeline (all on one GPU, memory-light — projections streamed, acts never stored):
  1. per-layer raw mass-mean directions d_L from the refuse/benign split
     (probe_ensemble.directions), same chat format as the OR scorer.
  2. project every unique pair text onto d_L -> proj[text] (all layers).
  3. Dproj(pair) = proj(rewrite) - proj(original); standardize -> Ds.
  4. ensemble: w = NNLS-rank stack fit to rank(dP) (probe_ensemble.nnls_stack),
     probe_delta = Ds @ w  (refit-and-emit; raw ensemble ~= L17, Spearman~0.62).
  5. write shards grouped by original: {original, paraphrases:[{paraphrase,
     refusal_delta=probe_delta, similarity, dP}]}.
"""
import argparse
import csv
import json
import os
import sys
import time
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS, "probe_or"))
from probe_ensemble import project, nnls_stack   # noqa: E402
from scipy.stats import spearmanr, rankdata                   # noqa: E402


def read_split(path):
    out = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        col = "prompt" if "prompt" in r.fieldnames else r.fieldnames[0]
        for row in r:
            p = (row.get(col) or "").strip()
            if p and p.lower() != "prompt":
                out.append(p)
    return out


def read_pairs(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    sim = float(row["similarity"])
                    dP = float(row["dP"])
                except (KeyError, ValueError):
                    continue
                o = (row.get("original") or "").strip()
                rw = (row.get("rewrite") or "").strip()
                if o and rw:
                    rows.append((o, rw, sim, dP))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benign_csv", required=True)
    ap.add_argument("--refusal_csv", required=True)
    ap.add_argument("--pairs_csv", nargs="+", required=True)
    ap.add_argument("--out_shards", required=True)
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--n_shards", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(args.out_shards, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tok = AutoTokenizer.from_pretrained(args.base_model, token=hf)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    use_system = True
    try:
        tok.apply_chat_template([{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                                tokenize=False, add_generation_prompt=True)
    except Exception:
        use_system = False
    model = AutoModelForCausalLM.from_pretrained(args.base_model, token=hf, device_map="auto",
                                                 torch_dtype=torch.bfloat16)
    model.eval()

    def fmt(p):
        msgs = ([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": p}]
                if use_system else [{"role": "user", "content": p}])
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def batches(texts):
        for i in range(0, len(texts), args.batch_size):
            b = texts[i:i + args.batch_size]
            enc = tok([fmt(p) for p in b], return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_length).to(model.device)
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            yield b, torch.stack([h[:, -1, :] for h in hs], dim=1).float().cpu().numpy()  # [B, nL+1, H]

    # --- 1. directions from split (accumulate sums -> means) ---
    print("[dir] building raw mass-mean directions from split", flush=True)
    sum_ref = sum_ben = None
    n_ref = n_ben = 0
    t0 = time.time()
    for label, path in [("ref", args.refusal_csv), ("ben", args.benign_csv)]:
        texts = read_split(path)
        print(f"[dir] {label}: {len(texts)}")
        for k, (_, arr) in enumerate(batches(texts)):
            s = arr.astype(np.float64).sum(0)  # [nL+1, H]
            if label == "ref":
                sum_ref = s if sum_ref is None else sum_ref + s
                n_ref += arr.shape[0]
            else:
                sum_ben = s if sum_ben is None else sum_ben + s
                n_ben += arr.shape[0]
            if k % 40 == 0:
                print(f"  [{label}] {k*args.batch_size} ({time.time()-t0:.0f}s)", flush=True)
    # reuse teammate's exact math via fake 1-sample class-mean arrays
    d = (sum_ref / n_ref) - (sum_ben / n_ben)          # [nL+1, H]
    dn = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9

    # --- 2. project unique pair texts ---
    pairs = read_pairs(args.pairs_csv)
    print(f"[proj] {len(pairs)} pairs", flush=True)
    uniq = list(dict.fromkeys([o for o, _, _, _ in pairs] + [r for _, r, _, _ in pairs]))
    proj = {}
    t0 = time.time()
    for k, (b, arr) in enumerate(batches(uniq)):
        pr = project(arr, d, dn)  # [B, nL+1]
        for i, txt in enumerate(b):
            proj[txt] = pr[i]
        if k % 40 == 0:
            print(f"  [proj] {k*args.batch_size}/{len(uniq)} ({time.time()-t0:.0f}s)", flush=True)

    # --- 3-4. delta -> standardize -> NNLS-rank ensemble -> probe_delta ---
    Dproj = np.array([proj[r] - proj[o] for o, r, _, _ in pairs])   # [n, nL+1]
    dP = np.array([p[3] for p in pairs])
    mu, sd = Dproj.mean(0), Dproj.std(0) + 1e-9
    Ds = (Dproj - mu) / sd
    w = nnls_stack(Ds, rankdata(dP) / len(dP))
    probe_delta = Ds @ w
    sp = spearmanr(probe_delta, dP).statistic
    top_layers = np.argsort(-w)[:5]
    print(f"[ensemble] Spearman(probe_delta, dP) = {sp:.4f}  (expect ~0.62)")
    print(f"[ensemble] top weighted layers: {[(int(L), round(float(w[L]),3)) for L in top_layers]}")

    # --- 5. emit shards grouped by original ---
    by_orig = {}
    for (o, r, sim, dp), pd in zip(pairs, probe_delta):
        # or_score_raw is required by load_shards; train recomputes it from
        # refusal_delta+similarity, so this value is just a placeholder.
        orv = float(np.exp(18.4 * (sim - 0.75)) * pd)
        by_orig.setdefault(o, []).append({"paraphrase": r, "refusal_delta": float(pd),
                                          "similarity": float(sim), "or_score_raw": orv,
                                          "dP": float(dp)})
    recs = [{"original": o, "paraphrases": ps} for o, ps in by_orig.items()]
    per = (len(recs) + args.n_shards - 1) // args.n_shards
    for si in range(args.n_shards):
        chunk = recs[si * per:(si + 1) * per]
        if chunk:
            json.dump(chunk, open(os.path.join(args.out_shards, f"or_susceptibility_rankings_shard{si}.json"), "w"))
    print(f"[done] wrote {len(recs)} originals / {len(pairs)} pairs to {args.out_shards}")


if __name__ == "__main__":
    main()
