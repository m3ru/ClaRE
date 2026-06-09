#!/usr/bin/env python3
"""Diagnose whether the OR negative-delta bias is a padding/implementation bug.

reward_model.py never sets tokenizer.padding_side and reads the last-token
activation [:, -1, :]. With RIGHT padding, every sequence shorter than the
longest in its batch has its "last token" land on a PAD token -> corrupted
activation. The original is scored as a uniform batch (no padding); paraphrases
are a mixed-length batch (padded) -> systematic original-vs-paraphrase bias.

Tests, using the validated relabeled layer-30 vector:
  (1) default tokenizer.padding_side for Llama-Guard
  (2) dolly delta %positive under: left/bs1, left/bs32, right/bs32
  (3) adversarial Claude adaptations %positive under left vs right padding
If left ~50% but right << 50%, the bias is the padding bug.
"""
import argparse, json, random
import os
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector", required=True)
    ap.add_argument("--dolly", required=True)
    ap.add_argument("--claude", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-Guard-3-8B")
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--n_prompts", type=int, default=120)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    rng = random.Random(42)
    vec = np.load(args.vector, allow_pickle=True)["vector"].astype(np.float32)
    vt = torch.from_numpy(vec); vu = (vt / (vt.norm() + 1e-9))

    tok = AutoTokenizer.from_pretrained(args.model, token=hf)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print("DEFAULT tokenizer.padding_side =", tok.padding_side)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=hf, device_map="auto", torch_dtype=torch.bfloat16).eval()
    dev = model.device
    vu = vu.to(dev).to(torch.float32)

    def fmt(p):
        return tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)

    def proj(texts, side, bs):
        tok.padding_side = side
        out = []
        for i in range(0, len(texts), bs):
            b = texts[i:i+bs]
            enc = tok([fmt(t) for t in b], return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_length).to(dev)
            with torch.no_grad():
                o = model(**enc, output_hidden_states=True, use_cache=False)
            hs = o.hidden_states[args.layer][:, -1, :].to(torch.float32)
            out.extend((hs @ vu).cpu().numpy().tolist())
        return np.array(out)

    # ---- (2) dolly: replicate pipeline (original as uniform batch of copies) ----
    d = json.load(open(args.dolly)); rng.shuffle(d); d = d[:args.n_prompts]
    def dolly_pct(side, bs):
        pos = n = 0
        for e in d:
            paras = e["paraphrases_text"][:args.k]
            if not paras:
                continue
            po = proj([e["original"]] * len(paras), side, bs)   # uniform, no pad
            pp = proj(paras, side, bs)                           # mixed, padded
            dlt = pp - po
            pos += int((dlt > 0).sum()); n += len(paras)
        return 100 * pos / max(n, 1)
    print("\n==== (2) DOLLY %positive delta (relabeled vector) ====")
    print("  left  / batch=1 :", round(dolly_pct("left", 1), 1))
    print("  left  / batch=32:", round(dolly_pct("left", 32), 1))
    print("  right / batch=32:", round(dolly_pct("right", 32), 1), "  <- reward_model default if right")

    # ---- (3) Claude adversarial adaptations (1:1 pairs) ----
    cj = json.load(open(args.claude))
    gens = cj["variants"][list(cj["variants"].keys())[0]]["generations"]
    rng.shuffle(gens); gens = gens[:600]
    origs = [g["original"] for g in gens]; adapts = [g["generation"] for g in gens]
    def claude_pct(side, bs):
        po = proj(origs, side, bs); pa = proj(adapts, side, bs)
        dlt = pa - po
        return 100 * (dlt > 0).mean(), float(np.median(dlt))
    print("\n==== (3) CLAUDE adversarial adaptations %positive delta ====")
    for side in ("left", "right"):
        pct, med = claude_pct(side, 32)
        print("  %-5s/ batch=32: %.1f%% positive  (median delta %.3f)" % (side, pct, med))
    print("[done] diagnosis complete")


if __name__ == "__main__":
    main()
