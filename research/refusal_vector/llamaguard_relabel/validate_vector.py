#!/usr/bin/env python3
"""Validate the relabeled Llama-Guard refusal vector and diagnose the
negative-delta bias seen when scoring dolly paraphrases.

Three checks:
  (A) VECTOR VALIDITY: on a mixed safe/unsafe sample (from the LG-labelled
      split), correlate the vector projection with Llama-Guard's own verdict
      (AUC vs hard label) and stored p_unsafe. High => vector faithfully
      measures Llama-Guard's safe/unsafe judgement.
  (B) BIAS — REAL vs ARTIFACT: for dolly (original, paraphrase) pairs, compute
      the projection delta two ways:
        - clean: every text scored alone (batch=1, no padding)
        - pipeline: scored the way score_and_rank does it (original as a batch
          of identical copies, paraphrases as one left-padded mixed batch)
      If clean is ~50/50 but pipeline is ~16% positive, the bias is a
      left-padding/batching artifact, not a real Llama-Guard effect.
  (C) GROUND TRUTH: for the same dolly pairs, Llama-Guard's actual p_unsafe
      (generation-based) for original vs paraphrase — does the model itself
      see paraphrases as safer?
"""
import argparse
import csv
import json
import os
import random
from typing import List

import numpy as np
import torch


def auc(scores, labels):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--dolly_paraphrases", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-Guard-3-8B")
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--n_validity", type=int, default=400)
    ap.add_argument("--n_dolly_prompts", type=int, default=150)
    ap.add_argument("--k_paraphrases", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    rng = random.Random(args.seed)

    vec = np.load(args.vector, allow_pickle=True)["vector"].astype(np.float32)
    vec_t = torch.from_numpy(vec)
    vec_unit = vec_t / (vec_t.norm() + 1e-9)

    tok = AutoTokenizer.from_pretrained(args.model, token=hf)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=hf, device_map="auto", torch_dtype=torch.bfloat16)
    model.eval()
    dev = model.device
    vu = vec_unit.to(dev).to(torch.float32)
    safe_id = tok.encode("safe", add_special_tokens=False)[0]
    unsafe_id = tok.encode("unsafe", add_special_tokens=False)[0]

    def fmt(p):
        return tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)

    def proj_batch(texts):
        """Last-token layer-L projection for a batch (left-padded)."""
        enc = tok([fmt(t) for t in texts], return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_length).to(dev)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[args.layer][:, -1, :].to(torch.float32)
        return (hs @ vu).cpu().numpy()

    def proj_single(texts):
        return np.array([proj_batch([t])[0] for t in texts])

    def p_unsafe(texts):
        """Llama-Guard actual p(unsafe) via short greedy gen + verdict-step softmax."""
        out_ps = []
        for t in texts:
            enc = tok(fmt(t), return_tensors="pt", truncation=True,
                      max_length=args.max_length).to(dev)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=6, do_sample=False,
                                   output_scores=True, return_dict_in_generate=True,
                                   pad_token_id=tok.pad_token_id)
            gen = g.sequences[0, enc["input_ids"].shape[1]:]
            pv = float("nan")
            for step in range(len(g.scores)):
                tid = int(gen[step])
                if tid in (safe_id, unsafe_id):
                    lg = g.scores[step][0]
                    pair = torch.softmax(torch.stack([lg[safe_id], lg[unsafe_id]]), 0)
                    pv = float(pair[1]); break
            out_ps.append(pv)
        return np.array(out_ps)

    # ---------- (A) VECTOR VALIDITY ----------
    rows = [r for r in csv.DictReader(open(args.labels_csv)) if r.get("p_unsafe")]
    rng.shuffle(rows)
    rows = rows[:args.n_validity]
    texts = [r["prompt"] for r in rows]
    y = np.array([1 if r["label"] == "unsafe" else 0 for r in rows])
    pq = np.array([float(r["p_unsafe"]) for r in rows], dtype=np.float64)
    proj_list = []
    for i in range(0, len(texts), 8):
        proj_list.extend(np.atleast_1d(proj_batch(texts[i:i+8])).astype(np.float64).tolist())
    proj = np.array(proj_list, dtype=np.float64)
    print("==== (A) VECTOR VALIDITY (mixed safe/unsafe, n=%d) ====" % len(texts))
    print("  AUC(projection vs Llama-Guard hard verdict): %.4f" % auc(proj, y))
    m = ~np.isnan(pq)
    print("  corr(projection, stored p_unsafe): %.3f" % float(np.corrcoef(proj[m], pq[m])[0, 1]))
    print("  projection mean: unsafe=%.2f  safe=%.2f" % (proj[y == 1].mean(), proj[y == 0].mean()))

    # ---------- (B/C) DOLLY BIAS: real vs artifact ----------
    d = json.load(open(args.dolly_paraphrases))
    rng.shuffle(d)
    d = d[:args.n_dolly_prompts]
    clean_pos = pipe_pos = pun_pos = 0; clean_n = pun_n = 0
    clean_deltas = []; pipe_deltas = []; pun_deltas = []
    for e in d:
        o = e["original"]; paras = e["paraphrases_text"][:args.k_paraphrases]
        if not paras:
            continue
        # clean: each text alone
        po = proj_single([o])[0]; pp = proj_single(paras)
        clean_deltas.extend((pp - po).tolist())
        clean_pos += int((pp - po > 0).sum()); clean_n += len(paras)
        # pipeline: original as batch of copies, paraphrases as one batch
        po_pipe = proj_batch([o] * len(paras))   # uniform batch (no pad variance)
        pp_pipe = proj_batch(paras)              # mixed, left-padded
        pipe_deltas.extend((pp_pipe - po_pipe).tolist())
        pipe_pos += int((pp_pipe - po_pipe > 0).sum())
        # ground truth p_unsafe on a subset (first 3 paraphrases to bound cost)
        sub = paras[:3]
        pq_o = p_unsafe([o])[0]; pq_p = p_unsafe(sub)
        dd = pq_p - pq_o
        dd = dd[~np.isnan(dd)]
        pun_deltas.extend(dd.tolist()); pun_pos += int((dd > 0).sum()); pun_n += len(dd)

    print("\n==== (B) DOLLY DELTA: real vs batching artifact ====")
    print("  CLEAN  (batch=1, no padding): %.1f%% positive  (median delta %.3f)"
          % (100 * clean_pos / max(clean_n, 1), np.median(clean_deltas)))
    print("  PIPELINE(orig=copies, paras=left-padded batch): %.1f%% positive  (median delta %.3f)"
          % (100 * pipe_pos / max(clean_n, 1), np.median(pipe_deltas)))
    print("\n==== (C) GROUND TRUTH: Llama-Guard actual p_unsafe delta ====")
    print("  %.1f%% of paraphrases RAISE actual p_unsafe  (median delta %.4f, n=%d)"
          % (100 * pun_pos / max(pun_n, 1), np.median(pun_deltas) if pun_deltas else float('nan'), pun_n))


if __name__ == "__main__":
    main()
