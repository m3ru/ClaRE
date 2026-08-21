#!/usr/bin/env python3
"""Fit the diff-of-means direction in the KILL-SWITCH's exact measurement context and
compare it, per layer, to the atlas direction in probe_absolute.npz.

Split-half resampling already showed the estimator is stable (cos 0.996 at L17), so the
only untested difference between the two candidates is tokenization/padding context:
extract_layer_acts.py uses add_special_tokens=False + right padding, the kill-switch used
the chat-template default + left padding. This measures whether that matters.
"""
import argparse, csv, os, sys
import numpy as np
csv.field_size_limit(sys.maxsize)
SYSTEM = "You are a helpful assistant."


def read_col(path, col, n):
    out = []
    for r in csv.DictReader(open(path)):
        v = (r.get(col) or "").strip()
        if v:
            out.append(v)
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--refusal_csv", default="../../refusal_vector/3_Vector_Extraction/final_refusals_prompts.csv")
    ap.add_argument("--benign_csv", default="../../refusal_vector/3_Vector_Extraction/final_benign_prompts.csv")
    ap.add_argument("--atlas_npz", default="probe_or/results/llama_signals/probe_absolute.npz")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", default="probe_or/results/dir_compare.npz")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tok = AutoTokenizer.from_pretrained(a.base_model, token=hf)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base_model, token=hf, device_map="auto",
                                                 dtype=torch.bfloat16).eval()
    H, nL = model.config.hidden_size, model.config.num_hidden_layers

    def fmt(p):
        return tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def class_mean(texts, add_special):
        sm, cnt = np.zeros((nL + 1, H), np.float64), 0
        for i in range(0, len(texts), a.bs):
            enc = tok([fmt(p) for p in texts[i:i + a.bs]], return_tensors="pt", padding=True,
                      truncation=True, max_length=512,
                      add_special_tokens=add_special).to(model.device)
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            am = enc["attention_mask"].to(torch.int)
            idx = am.shape[1] - 1 - am.flip(1).argmax(dim=1)
            last = torch.stack([h[torch.arange(h.shape[0]), idx, :] for h in hs], dim=1)
            sm += last.float().cpu().numpy().sum(0)
            cnt += last.shape[0]
        return sm / cnt

    ref, ben = read_col(a.refusal_csv, "prompt", a.n), read_col(a.benign_csv, "prompt", a.n)
    print(f"[fit] {len(ref)} refusal / {len(ben)} benign", flush=True)
    res = {}
    for tag, add_special in (("addspecial_true", True), ("addspecial_false", False)):
        d = class_mean(ref, add_special) - class_mean(ben, add_special)
        res[tag] = d
        print(f"[fit] {tag} done", flush=True)
    P = np.load(a.atlas_npz, allow_pickle=True)
    da = P["d"].astype(np.float64)
    n = lambda X: X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)
    print("\nlayer | cos(refit_addspecial_true, atlas) | cos(refit_addspecial_false, atlas)")
    for L in (8, 12, 16, 17, 18, 20, 24, 28, 31, 32):
        print(f"  {L:2d}  |            {float(n(res['addspecial_true'])[L] @ n(da)[L]):+.4f}             |"
              f"            {float(n(res['addspecial_false'])[L] @ n(da)[L]):+.4f}")
    np.savez(a.out, atlas=da, **res)
    print(f"\n[done] {a.out}")


if __name__ == "__main__":
    main()
