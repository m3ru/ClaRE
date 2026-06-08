#!/usr/bin/env python3
"""Per-layer separation sweep for the Llama-Guard 'unsafe' direction.

Given prompts labelled by Llama-Guard's own verdict (from
label_prompts_llamaguard.py), this finds which layer's difference-of-means
direction best separates safe vs unsafe — instead of blindly taking the
max-L2 layer (which is just the final-layer outlier-norm spike).

For a balanced sample it captures the last-token hidden state at every layer,
splits train/test by prompt, builds diff-of-means on TRAIN per layer, and
measures test AUC (raw and per-dim-standardized). Reports the table, picks the
best layer, and saves that layer's vector in the npz format reward_model.py
expects (1-D 'vector' + 'layer'), plus an all-layer npz and a JSON summary.

Formatting matches reward_model/extract_activations (user-only chat template,
last token, truncation) so the vector transfers to scoring.
"""
import argparse
import csv
import json
import os
import random
from typing import List

import numpy as np
import torch


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC = P(score[pos] > score[neg]); rank-based, handles ties."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load_prompts(path: str) -> List[str]:
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append(row["prompt"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unsafe_csv", required=True)
    ap.add_argument("--safe_csv", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-Guard-3-8B")
    ap.add_argument("--n_per_class", type=int, default=2000)
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

    unsafe = load_prompts(args.unsafe_csv)
    safe = load_prompts(args.safe_csv)
    n = min(args.n_per_class, len(unsafe), len(safe))
    print(f"[sweep] available unsafe={len(unsafe)} safe={len(safe)} -> {n}/class", flush=True)
    rng.shuffle(unsafe)
    rng.shuffle(safe)
    unsafe, safe = unsafe[:n], safe[:n]
    prompts = unsafe + safe
    labels = np.array([1] * n + [0] * n)

    tok = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=hf_token, device_map="auto", torch_dtype=torch.bfloat16)
    model.eval()

    n_layers = model.config.num_hidden_layers  # 32
    H = model.config.hidden_size
    acts = np.zeros((len(prompts), n_layers, H), dtype=np.float32)  # layer 1..n_layers
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i + args.batch_size]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                 tokenize=False, add_generation_prompt=True) for p in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_length).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # len n_layers+1; [0]=embeddings
        for lyr in range(1, n_layers + 1):
            acts[i:i + len(batch), lyr - 1, :] = hs[lyr][:, -1, :].float().cpu().numpy()
        if (i // args.batch_size) % 10 == 0:
            print(f"[sweep] activations {i + len(batch)}/{len(prompts)}", flush=True)

    # train/test split by index (balanced labels already interleaved by class blocks)
    idx = np.arange(len(prompts))
    rng2 = np.random.RandomState(args.seed)
    rng2.shuffle(idx)
    n_val = int(len(idx) * args.val_frac)
    test_idx, train_idx = idx[:n_val], idx[n_val:]
    ytr, yte = labels[train_idx], labels[test_idx]

    layers = list(range(1, n_layers + 1))
    results = []
    all_vecs = np.zeros((n_layers, H), dtype=np.float32)
    for li, lyr in enumerate(layers):
        A = acts[:, li, :]
        Atr, Ate = A[train_idx], A[test_idx]
        # raw diff-of-means
        dom = Atr[ytr == 1].mean(0) - Atr[ytr == 0].mean(0)
        all_vecs[li] = dom
        auc_raw = auc(Ate @ dom, yte)
        # standardized (z-score per dim using train stats)
        mu, sd = Atr.mean(0), Atr.std(0) + 1e-6
        Ztr, Zte = (Atr - mu) / sd, (Ate - mu) / sd
        dom_z = Ztr[ytr == 1].mean(0) - Ztr[ytr == 0].mean(0)
        auc_std = auc(Zte @ dom_z, yte)
        results.append({"layer": lyr, "auc_raw": round(float(auc_raw), 4),
                        "auc_std": round(float(auc_std), 4),
                        "l2": round(float(np.linalg.norm(dom)), 4)})

    results_sorted = sorted(results, key=lambda r: r["auc_raw"], reverse=True)
    best = results_sorted[0]
    best_std = sorted(results, key=lambda r: r["auc_std"], reverse=True)[0]
    print("\n[sweep] layer |  auc_raw | auc_std |   L2", flush=True)
    for r in results:
        mark = "  <== best raw" if r["layer"] == best["layer"] else ""
        print(f"   {r['layer']:>3} |  {r['auc_raw']:.4f} | {r['auc_std']:.4f} | {r['l2']:7.2f}{mark}", flush=True)
    print(f"\n[sweep] BEST raw-AUC layer={best['layer']} auc={best['auc_raw']}", flush=True)
    print(f"[sweep] BEST std-AUC layer={best_std['layer']} auc={best_std['auc_std']}", flush=True)
    print(f"[sweep] (old pipeline picked max-L2 layer = "
          f"{max(results, key=lambda r: r['l2'])['layer']})", flush=True)

    # Save the best-layer vector (built on ALL sampled data) in reward_model format
    bl = int(best["layer"])
    A = acts[:, bl - 1, :]
    final_dom = (A[labels == 1].mean(0) - A[labels == 0].mean(0)).astype(np.float32)
    out_vec = os.path.join(args.output_dir, f"refusal_vector_llamaguard_relabeled.layer{bl:03d}.npz")
    np.savez(out_vec, vector=final_dom, layer=int(bl),
             description="Llama-Guard relabel diff-of-means; layer chosen by separation AUC")
    np.savez(os.path.join(args.output_dir, "refusal_vector_llamaguard_relabeled.npz"),
             vector=all_vecs, layers=np.array(layers, dtype="int32"),
             l2_per_layer=np.linalg.norm(all_vecs, axis=1).astype("float32"),
             description="per-layer diff-of-means (Llama-Guard relabel)")
    with open(os.path.join(args.output_dir, "layer_sweep_auc.json"), "w") as f:
        json.dump({"n_per_class": n, "val_frac": args.val_frac,
                   "max_length": args.max_length, "results": results,
                   "best_raw": best, "best_std": best_std}, f, indent=2)
    print(f"\n[sweep] saved best vector -> {out_vec}", flush=True)


if __name__ == "__main__":
    main()
