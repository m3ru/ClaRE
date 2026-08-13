#!/usr/bin/env python3
"""Refusal Atlas P1+P2: score the substrate with all three refusal signals AND
behavioral ground truth, for ONE model, in one load.

For every substrate prompt we record, in a single shared measurement context
(system="You are a helpful assistant.", add_generation_prompt=True,
enable_thinking=False):

  vector    absolute projection on the fitted best-layer refusal direction
  probe     fitted layer-weighted combination of standardized per-layer projections
  logit_sum teacher-forced P(response begins with ANY of the model's refusal openers)
  logit_max teacher-forced max single-opener prefix probability
  refuse    behavioral refusal rate over n sampled responses (broad opener classifier)

The per-prompt signal is the ABSOLUTE projection (the Arditi-style refusal signal),
since most substrate prompts are singletons. For minimal pairs we ALSO emit the
delta (rw-orig) form for the word analysis.

Signals (from --probe_npz = fit_probe_absolute.py's probe_absolute.npz):
  - VECTOR = raw absolute projection at the CAUSAL layer (--best_layer overrides the
    npz's best_layer): Llama L17, Qwen L58 (both causally validated by ablation/addition).
  - PROBE  = (P - mu)/sd @ w, using the STORED fit-time mu/sd and combiner weights w
    from the absolute-space refit (mass-mean directions + broad-behavioral labels,
    L0 excluded). Do NOT recompute mu/sd on the substrate.
  - Directions are fit on an INDEPENDENT refuse/benign split, so absolute projections
    carry no train/test leakage; the fresh minimal pairs keep the delta form clean too.
Distinctness: for Llama vector(L17) != probe(~L31) -> 3 genuine signals; for Qwen the
probe collapses to L58 == vector, so vector ~ probe (reported transparently).
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

# probe_ensemble.project and gen_qwen_refusal.classify live in ../probe_or
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "probe_or"))
from probe_ensemble import project   # <acts,d>/||d||, identical to fit-time projection

SYSTEM = "You are a helpful assistant."   # MUST match extract_layer_acts.py at fit time


# --------------------------------------------------------------- formatting
def make_fmt(tok):
    def fmt(p):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    return fmt


# --------------------------------------------------------------- pass A: projections
def project_texts(model, tok, texts, d, dn, batch_size, max_length):
    """proj[text] = per-layer absolute projection [nL+1] on the fitted directions."""
    assert tok.padding_side == "right", "pass-A last-token read assumes right padding"
    fmt = make_fmt(tok)
    import torch
    proj, n_trunc, t0 = {}, 0, time.time()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok([fmt(p) for p in batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length, add_special_tokens=False).to(model.device)
        last = enc["attention_mask"].sum(1) - 1
        n_trunc += int((enc["attention_mask"].sum(1) >= max_length).sum())
        rows = torch.arange(len(batch), device=model.device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        acts = torch.stack([h[rows, last, :] for h in hs], dim=1).float().cpu().numpy()  # [B,nL+1,H]
        pr = project(acts, d, dn)      # [B, nL+1]
        for j, txt in enumerate(batch):
            proj[txt] = pr[j]
        if (i // batch_size) % 20 == 0:
            print(f"  [projA {i+len(batch)}/{len(texts)}] {(i+len(batch))/max(time.time()-t0,1e-3):.1f}/s", flush=True)
    if n_trunc:
        print(f"[warn] passA: {n_trunc} text(s) truncated at max_length={max_length}; last-token read off-position", flush=True)
    return proj


# --------------------------------------------------------------- pass B: multi-phrase logit
def logit_openers(model, tok, texts, openers, batch_size, max_prefix):
    """For each text: teacher-forced prefix probability for each opener, then
    logit_sum = sum_i P(begins with opener_i)  (openers are token-prefix-disjoint,
    so the sum approximates P(begins with ANY opener)); logit_max = max_i.
    Returns {text: {"sum":.., "max":.., "per":{opener:prob}}}."""
    import torch
    fmt = make_fmt(tok)
    opener_ids = [tok.encode(o, add_special_tokens=False) for o in openers]
    for o, ids in zip(openers, opener_ids):
        print(f"[logit] {o!r} -> {ids} -> {[tok.decode([t]) for t in ids]}", flush=True)
    # guard: no opener's token sequence is a prefix of another's (else the sum double-counts)
    for a in range(len(opener_ids)):
        for b in range(len(opener_ids)):
            if a != b and opener_ids[b][:len(opener_ids[a])] == opener_ids[a]:
                raise SystemExit(f"[abort] opener {openers[a]!r} is a token-prefix of {openers[b]!r}; "
                                 "sum would double-count. Edit the opener set.")
    out, n_trunc, t0 = {}, 0, time.time()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        prefix_ids = [tok(fmt(p), add_special_tokens=False, truncation=True,
                          max_length=max_prefix)["input_ids"] for p in batch]
        n_trunc += sum(1 for pi in prefix_ids if len(pi) >= max_prefix)
        # one padded forward pass per opener (keeps memory flat; openers are few)
        per_text = [{} for _ in batch]
        for oi, cont in enumerate(opener_ids):
            C = len(cont)
            seqs = [pi + cont for pi in prefix_ids]
            plen = [len(pi) for pi in prefix_ids]
            L = max(len(s) for s in seqs)
            input_ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
            attn = torch.zeros((len(seqs), L), dtype=torch.long)
            for b, s in enumerate(seqs):
                input_ids[b, :len(s)] = torch.tensor(s, dtype=torch.long)
                attn[b, :len(s)] = 1
            input_ids, attn = input_ids.to(model.device), attn.to(model.device)
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
            logp = torch.log_softmax(logits.float(), dim=-1)
            for b in range(len(batch)):
                P = plen[b]
                lp = sum(float(logp[b, P - 1 + j, cont[j]]) for j in range(C))
                per_text[b][openers[oi]] = float(np.exp(lp))
        for b, txt in enumerate(batch):
            probs = per_text[b]
            out[txt] = {"sum": float(sum(probs.values())), "max": float(max(probs.values())), "per": probs}
        if (i // batch_size) % 20 == 0:
            done = min(i + batch_size, len(texts))
            print(f"  [logitB {done}/{len(texts)}] {done/max(time.time()-t0,1e-3):.1f}/s", flush=True)
    if n_trunc:
        print(f"[warn] passB: {n_trunc} prefix(es) hit max_length={max_prefix}; template tail cut -> logit meaningless for those", flush=True)
    return out


# --------------------------------------------------------------- pass C: behavioral
def behavioral(model, tok, prompts, n, temp, max_new, bs, max_length):
    """n sampled responses per prompt -> refusal rate via the broad opener classifier.
    Generation needs LEFT padding (mirrors gen_qwen_refusal.py / eval_rwr_qwen.py)."""
    import torch
    from gen_qwen_refusal import classify
    fmt = make_fmt(tok)
    prev_side = tok.padding_side
    tok.padding_side = "left"                     # CRITICAL: right pad corrupts generated openers
    term = [tok.eos_token_id]
    for t in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0 and tid not in term:
            term.append(tid)
    rates, samples, n_trunc, t0 = {}, {}, 0, time.time()
    try:
        for i in range(0, len(prompts), bs):
            chunk = prompts[i:i + bs]
            enc = tok([fmt(p) for p in chunk], return_tensors="pt", padding=True, truncation=True,
                      max_length=max_length, add_special_tokens=False).to(model.device)
            n_trunc += int((enc["attention_mask"].sum(1) >= max_length).sum())
            L = enc["input_ids"].shape[1]
            with torch.no_grad():
                g = model.generate(**enc, do_sample=True, temperature=temp, top_p=0.9,
                                   num_return_sequences=n, max_new_tokens=max_new,
                                   eos_token_id=term, pad_token_id=tok.pad_token_id)
            dec = tok.batch_decode(g[:, L:], skip_special_tokens=True)   # left pad => uniform L strip
            for b, p in enumerate(chunk):
                outs = [d.strip() for d in dec[b * n:(b + 1) * n]]
                rates[p] = float(np.mean([classify(s)[0] for s in outs]))
                samples[p] = outs
            if (i // bs) % 20 == 0:
                print(f"  [behavC {i+len(chunk)}/{len(prompts)}] {(i+len(chunk))/max(time.time()-t0,1e-3):.1f}/s", flush=True)
    finally:
        tok.padding_side = prev_side
    if n_trunc:
        print(f"[warn] passC: {n_trunc} prompt(s) truncated at max_length={max_length}", flush=True)
    return rates, samples


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--probe_npz", required=True, help="fitted directions: d, dn, w (+best_layer for Qwen)")
    ap.add_argument("--best_layer", type=int, default=None,
                    help="best-layer index for the vector signal; overrides npz. REQUIRED if npz lacks it (Llama: 27)")
    ap.add_argument("--opener_json", required=True)
    ap.add_argument("--model_key", required=True, choices=["qwen", "llama"], help="which opener set to use")
    ap.add_argument("--out", required=True, help="per-prompt CSV")
    ap.add_argument("--pairs_out", default=None, help="minimal-pair delta CSV (rw-orig per signal)")
    ap.add_argument("--samples_out", default=None, help="optional JSON of raw behavioral samples")
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--proj_bs", type=int, default=8)
    ap.add_argument("--logit_bs", type=int, default=8)
    ap.add_argument("--gen_bs", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = list(csv.DictReader(open(args.substrate)))
    texts = list(dict.fromkeys(r["text"].strip() for r in rows))   # unique, order-stable
    print(f"[data] {len(rows)} substrate rows | {len(texts)} unique texts", flush=True)

    z = np.load(args.probe_npz)
    d, dn, w = z["d"], z["dn"], z["w"]
    mu_fit, sd_fit = z["mu"], z["sd"]        # STORED fit-time standardization (do not recompute on substrate)
    if args.best_layer is not None:
        best_layer = args.best_layer
    elif "best_layer" in z:
        best_layer = int(z["best_layer"])
    else:
        raise SystemExit("[abort] --probe_npz has no 'best_layer'; pass --best_layer explicitly "
                         "(Llama fitted best single layer = 27; do NOT fall back to argmax(w)=L30).")
    assert 0 <= best_layer < d.shape[0], f"best_layer {best_layer} out of range [0,{d.shape[0]})"
    print(f"[scorer] layers={d.shape[0]} vector=L{best_layer} probe_nonzero_w={(w>1e-4).sum()}", flush=True)

    opener_meta = json.load(open(args.opener_json))[args.model_key]
    openers = opener_meta["openers"]
    exp_model = opener_meta.get("model", "")
    if exp_model and exp_model != args.base_model:
        raise SystemExit(f"[abort] opener set '{args.model_key}' is for {exp_model!r} but --base_model={args.base_model!r}")
    print(f"[openers:{args.model_key}] {openers}", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base_model, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"        # pass A needs right pad; pass C flips to left internally
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, token=os.environ.get("HF_TOKEN"), torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="eager").eval()

    print("[passA] projections...", flush=True)
    proj = project_texts(model, tok, texts, d, dn, args.proj_bs, args.max_length)
    P = np.array([proj[t] for t in texts])                         # [n_texts, nL+1]
    assert np.isfinite(P).all(), "non-finite projections (fp16 overflow?) -- check directions"
    Ps = (P - mu_fit) / sd_fit                                     # apply the FITTED probe's standardization
    vec = {t: float(P[i, best_layer]) for i, t in enumerate(texts)}   # vector = raw absolute proj at causal layer
    prb = {t: float(Ps[i] @ w) for i, t in enumerate(texts)}

    print("[passB] multi-phrase logit...", flush=True)
    lg = logit_openers(model, tok, texts, openers, args.logit_bs, args.max_length)

    print("[passC] behavioral sampling...", flush=True)
    rates, samples = behavioral(model, tok, texts, args.n_samples, args.temp,
                                args.max_new, args.gen_bs, args.max_length)

    # --- write per-prompt CSV (aligned to substrate rows) ---
    fields = ["prompt_id", "source", "native_topic", "gold_benign", "pair_id", "is_rewrite",
              "vector", "probe", "logit_sum", "logit_max", "refuse_rate"]
    with open(args.out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=fields)
        wtr.writeheader()
        for r in rows:
            t = r["text"].strip()
            wtr.writerow(dict(
                prompt_id=r["prompt_id"], source=r["source"], native_topic=r["native_topic"],
                gold_benign=r["gold_benign"], pair_id=r["pair_id"], is_rewrite=r["is_rewrite"],
                vector=f"{vec[t]:.6g}", probe=f"{prb[t]:.6g}",
                logit_sum=f"{lg[t]['sum']:.6g}", logit_max=f"{lg[t]['max']:.6g}",
                refuse_rate=f"{rates[t]:.4f}"))
    print(f"[done] wrote {args.out}", flush=True)

    # --- minimal-pair deltas (rw - orig) for the word-level analysis ---
    if args.pairs_out:
        by_pair = {}
        for r in rows:
            if r["source"] in ("sonnet_pair", "single_edit_pair") and r["pair_id"]:
                e = by_pair.setdefault(r["pair_id"], {"src": r["source"], "word": ""})
                e[r["is_rewrite"]] = r["text"].strip()
                if r["is_rewrite"] == "1":
                    e["word"] = r["native_topic"].replace("word:", "")   # trigger word for single-edit pairs
        dropped = 0
        with open(args.pairs_out, "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["pair_id", "source", "trigger_word", "original", "rewrite", "d_vector",
                          "d_probe", "d_logit_sum", "d_logit_max", "refuse_orig", "refuse_rw", "d_refuse"])
            for pid, sides in sorted(by_pair.items()):
                o, rw = sides.get("0"), sides.get("1")
                if not (o and rw):
                    dropped += 1
                    continue
                wtr.writerow([pid, sides["src"], sides["word"], o, rw,
                              f"{vec[rw]-vec[o]:.6g}", f"{prb[rw]-prb[o]:.6g}",
                              f"{lg[rw]['sum']-lg[o]['sum']:.6g}", f"{lg[rw]['max']-lg[o]['max']:.6g}",
                              f"{rates[o]:.4f}", f"{rates[rw]:.4f}", f"{rates[rw]-rates[o]:.4f}"])
        if dropped:
            print(f"[warn] {dropped} pair(s) missing a side -> dropped from {args.pairs_out}", flush=True)
        print(f"[done] wrote {args.pairs_out}", flush=True)

    if args.samples_out:
        json.dump({r["prompt_id"]: {"text": r["text"], "samples": samples[r["text"].strip()]}
                   for r in rows}, open(args.samples_out, "w"))
        print(f"[done] wrote {args.samples_out}", flush=True)

    # --- quick sanity summary ---
    rr = np.array([rates[t] for t in texts])
    ls = np.array([lg[t]['sum'] for t in texts])
    print(f"\n[summary:{args.model_key}] behavioral refuse: mean={rr.mean():.3f} %prompts>0={100*(rr>0).mean():.1f}%")
    print(f"  logit_sum: mean={ls.mean():.4f} p90={np.quantile(ls,0.9):.4f} max={ls.max():.4f}")
    print(f"  vector(L{best_layer}): mean={np.mean(list(vec.values())):.3f}  probe: mean={np.mean(list(prb.values())):.3f}")


if __name__ == "__main__":
    main()
