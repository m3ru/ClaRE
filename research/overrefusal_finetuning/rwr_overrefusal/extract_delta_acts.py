#!/usr/bin/env python3
"""Phase 1: last-token residual-stream activations for every prompt in the Delta sets.

Formatting is deliberately IDENTICAL to probe_or/extract_layer_acts.py (system="You are a
helpful assistant.", add_generation_prompt=True, add_special_tokens=False, RIGHT padding with
the matching last-real-token index) so these activations live in the same space as the atlas
directions in probe_absolute.npz and projections onto them are meaningful.

Forward passes only, no generation. Saves acts[i] for the i-th unique prompt as fp16
[n_layers+1, hidden], plus prompts.json mapping text -> row, so Delta for any pair is one
subtraction on CPU.

Run: python extract_delta_acts.py --sets probe_or/results/delta/prompt_sets.csv \
        --out_dir probe_or/results/delta
"""
import argparse, csv, json, os, sys, time
import numpy as np
csv.field_size_limit(sys.maxsize)

SYSTEM = "You are a helpful assistant."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="probe_or/results/delta/prompt_sets.csv")
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--out_dir", default="probe_or/results/delta")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = list(csv.DictReader(open(a.sets)))
    texts = []
    seen = set()
    for r in rows:
        for k in ("original", "rewrite"):
            t = (r[k] or "").strip()
            if t and t not in seen:
                seen.add(t); texts.append(t)
    print(f"[extract] {len(rows)} set rows -> {len(texts)} unique prompts", flush=True)

    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tok = AutoTokenizer.from_pretrained(a.base_model, token=hf)
    tok.padding_side = "right"          # matches extract_layer_acts.py; index below assumes it
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base_model, token=hf, device_map="auto",
                                                 dtype=torch.bfloat16).eval()
    H, nL = model.config.hidden_size, model.config.num_hidden_layers

    def fmt(p):
        return tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    os.makedirs(a.out_dir, exist_ok=True)
    acts = np.zeros((len(texts), nL + 1, H), dtype=np.float16)
    n_trunc, t0 = 0, time.time()
    for i in range(0, len(texts), a.batch_size):
        batch = texts[i:i + a.batch_size]
        enc = tok([fmt(p) for p in batch], return_tensors="pt", padding=True, truncation=True,
                  max_length=a.max_length, add_special_tokens=False).to(model.device)
        am = enc["attention_mask"].to(torch.int)
        n_trunc += int((am.sum(1) >= a.max_length).sum())
        # right padding -> last real token is sum-1; assert the side so this stays true
        assert tok.padding_side == "right"
        idx = am.sum(1) - 1
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        last = torch.stack([h[torch.arange(h.shape[0]), idx, :] for h in hs], dim=1)
        acts[i:i + len(batch)] = last.float().cpu().numpy().astype(np.float16)
        if (i // a.batch_size) % 25 == 0:
            done = i + len(batch)
            print(f"  [{done}/{len(texts)}] {done/max(time.time()-t0,1e-3):.1f}/s", flush=True)

    np.save(os.path.join(a.out_dir, "acts.npy"), acts)
    json.dump({t: i for i, t in enumerate(texts)},
              open(os.path.join(a.out_dir, "prompt_index.json"), "w"))
    meta = dict(model=a.base_model, n_prompts=len(texts), n_layers=nL + 1, hidden=H,
                truncated=n_trunc, system=SYSTEM, add_special_tokens=False, padding="right")
    json.dump(meta, open(os.path.join(a.out_dir, "extract_meta.json"), "w"), indent=1)
    if n_trunc:
        print(f"[warn] {n_trunc} prompts hit max_length={a.max_length}; last-token read off-position")
    print(f"[done] acts {acts.shape} -> {a.out_dir}/acts.npy  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
