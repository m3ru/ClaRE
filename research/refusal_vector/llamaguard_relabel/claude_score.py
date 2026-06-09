#!/usr/bin/env python3
"""Score the Claude adversarial adaptations with the FIXED (left-padding)
scorer, to see the true %positive now that the padding bug is removed.

Two configs:
  - Llama-3 vector (layer 32) + Meta-Llama-3-8B-Instruct  <- the adaptations'
    ACTUAL optimization target; this is the clean version of the original
    pipeline's headline number.
  - Llama-Guard relabeled vector (layer 30) + Llama-Guard-3-8B  <- cross-model
    transfer.
Also compares to the stored (padding-corrupted) refusal_delta in the file.
"""
import argparse, json, os
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude", required=True)
    ap.add_argument("--llama3_vector", required=True)
    ap.add_argument("--lg_vector", required=True)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    cj = json.load(open(args.claude))
    gens = cj["variants"][list(cj["variants"].keys())[0]]["generations"]
    origs = [g["original"] for g in gens]
    adapts = [g["generation"] for g in gens]
    stored = np.array([g["refusal_delta"] for g in gens], dtype=np.float64)
    print("claude adaptations: n=%d  (stored buggy refusal_delta: %.1f%% positive)"
          % (len(gens), 100 * (stored > 0).mean()))

    def run(model_id, vec_path, layer, label):
        v = np.load(vec_path, allow_pickle=True)["vector"].astype(np.float32)
        vt = torch.from_numpy(v); vu = vt / (vt.norm() + 1e-9)
        tok = AutoTokenizer.from_pretrained(model_id, token=hf)
        tok.padding_side = "left"                      # the FIX
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, token=hf, device_map="auto", torch_dtype=torch.bfloat16).eval()
        dev = model.device; vu = vu.to(dev).to(torch.float32)

        def fmt(p):
            return tok.apply_chat_template(
                [{"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            ) if _accepts_system(tok) else tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)

        def proj(texts):
            out = []
            for i in range(0, len(texts), args.bs):
                enc = tok([fmt(t) for t in texts[i:i+args.bs]], return_tensors="pt",
                          padding=True, truncation=True, max_length=args.max_length).to(dev)
                with torch.no_grad():
                    o = model(**enc, output_hidden_states=True, use_cache=False)
                hs = o.hidden_states[layer][:, -1, :].to(torch.float32)
                out.extend((hs @ vu).cpu().numpy().tolist())
            return np.array(out)

        d = proj(adapts) - proj(origs)
        print("  [%s] %.1f%% positive  (median delta %.4f, mean %.4f)"
              % (label, 100 * (d > 0).mean(), np.median(d), d.mean()))
        del model; torch.cuda.empty_cache()
        return d

    print("\n==== Claude adaptations, FIXED scorer (left padding) ====")
    d3 = run("meta-llama/Meta-Llama-3-8B-Instruct", args.llama3_vector, 32,
             "Llama-3 vector (their target)")
    dg = run("meta-llama/Llama-Guard-3-8B", args.lg_vector, 30,
             "Llama-Guard relabeled vector (cross-model)")
    print("\n  corr(fixed Llama-3 delta, stored buggy delta): %.3f"
          % float(np.corrcoef(d3, stored)[0, 1]))
    print("[done]")


def _accepts_system(tok):
    try:
        tok.apply_chat_template([{"role": "system", "content": "x"},
                                 {"role": "user", "content": "y"}],
                                tokenize=False, add_generation_prompt=True)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
