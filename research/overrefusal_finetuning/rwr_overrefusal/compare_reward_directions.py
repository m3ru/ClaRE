#!/usr/bin/env python3
"""Head-to-head: does the direction you pick change the RWR attacker you train?

Question. Two criteria select different layers/directions for the same model:
  A. ABLATION-selected  -- Arditi's rule: argmin refusal after ablating d[L] at every
     layer. Measures causal necessity for refusing harmful prompts.
  B. AUC-selected       -- the layer whose projection best separates refused from
     non-refused. Measures how decodable refusal is.
Both have been used as the "refusal vector". This asks which makes a better OR reward
for post-training an attacker -- i.e. which one, used as signal(p), yields an attacker
that induces more genuine over-refusal.

Design note -- why this script is a GATE, not the experiment.
Training two RWR attackers and judging their output is ~2 full training + generation +
judging cycles. That is only worth spending if the two rewards would actually produce
DIFFERENT training sets. The RWR objective sees the pool only through (a) the ranking of
pairs by OR reward and (b) which pairs land in the top weighted bin. So we measure that
directly and cheaply first:

  - cos(dA, dB)                     do the directions even differ?
  - Spearman(or_A, or_B)            do they rank the pool differently?
  - Jaccard of the top bin          would RWR train on different pairs?
  - Spearman(or_X, behavioural dP)  does either rank actual refusal behaviour better?

If the top bins overlap almost entirely the post-training comparison is a foregone null
and the GPU is better spent elsewhere. That verdict is printed explicitly.

The reward is the project's: OR(o,r) = exp(k*(sim-c)) * (proj(r) - proj(o)), k=18.4,
c=0.75, raw (unstandardized) projection delta, matching probe_or/score_llama_or.py.
"""
import argparse, csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def spearman(x, y):
    from scipy.stats import spearmanr
    m = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[m], y[m]).statistic) if m.sum() > 2 else float("nan")


def load_dir(spec):
    """spec = path.npz:LAYER:KEY  -> (unit vector, layer, label)"""
    parts = spec.split(":")
    path, layer = parts[0], int(parts[1])
    key = parts[2] if len(parts) > 2 else None
    z = np.load(path, allow_pickle=True)
    if key is None:
        key = "all_d_hat" if "all_d_hat" in z.files else ("d" if "d" in z.files else z.files[0])
    arr = np.asarray(z[key], dtype=np.float64)
    v = arr[layer] if arr.ndim == 2 else arr        # 1-D npz already IS the layer's vector
    v = v / (np.linalg.norm(v) + 1e-12)
    return v, layer, f"{os.path.basename(path)}[{key}]@L{layer}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_csv", required=True, help="scored pool with original,rewrite,similarity")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--dir_a", required=True, help="ablation-selected, as path.npz:LAYER[:KEY]")
    ap.add_argument("--dir_b", required=True, help="AUC-selected, as path.npz:LAYER[:KEY]")
    ap.add_argument("--behaviour_col", default="d_logit",
                    help="column holding the behavioural refusal delta, the ranking target")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=float, default=18.4)
    ap.add_argument("--c", type=float, default=0.75)
    ap.add_argument("--top_frac", type=float, default=0.15,
                    help="top reward fraction standing in for the RWR top weighted bin")
    ap.add_argument("--n_max", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dA, LA, labA = load_dir(a.dir_a)
    dB, LB, labB = load_dir(a.dir_b)
    cos = float(dA @ dB)
    print(f"[dirs] A(ablation) {labA}\n[dirs] B(auc)      {labB}\n[dirs] cos(A,B) = {cos:+.4f}",
          flush=True)

    rows = []
    with open(a.pool_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("original") and r.get("rewrite"):
                rows.append(r)
            if len(rows) >= a.n_max:
                break
    print(f"[pool] {len(rows)} pairs from {a.pool_csv}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.base_model)
    tok.padding_side = "right"                    # matches score_llama_or.py's fit-time recipe
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base_model, device_map="auto",
                                                 torch_dtype=torch.bfloat16).eval()
    dev = model.device
    is_qwen = "qwen" in a.base_model.lower()

    def fmt(p):
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": p}]
        kw = {"enable_thinking": False} if is_qwen else {}
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)

    texts = sorted({r["original"] for r in rows} | {r["rewrite"] for r in rows})
    print(f"[proj] {len(texts)} unique texts, layers {LA} and {LB}", flush=True)
    tA = torch.tensor(dA, dtype=torch.bfloat16, device=dev)
    tB = torch.tensor(dB, dtype=torch.bfloat16, device=dev)
    projA, projB = {}, {}
    for i in range(0, len(texts), a.batch_size):
        b = texts[i:i + a.batch_size]
        enc = tok([fmt(p) for p in b], return_tensors="pt", padding=True,
                  truncation=True, max_length=a.max_length, add_special_tokens=False).to(dev)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        am = enc["attention_mask"]
        idx = am.sum(1) - 1                       # right padding -> last real token
        for j, p in enumerate(b):
            projA[p] = float(hs[LA][j, idx[j], :].to(tA.dtype) @ tA)
            projB[p] = float(hs[LB][j, idx[j], :].to(tB.dtype) @ tB)
        if i % (a.batch_size * 200) == 0:
            print(f"[proj] {i}/{len(texts)}", flush=True)

    sim = np.array([float(r.get("similarity") or 0.0) for r in rows])
    gate = np.exp(a.k * (sim - a.c))
    dvA = np.array([projA[r["rewrite"]] - projA[r["original"]] for r in rows])
    dvB = np.array([projB[r["rewrite"]] - projB[r["original"]] for r in rows])
    orA, orB = gate * dvA, gate * dvB

    n_top = max(1, int(a.top_frac * len(rows)))
    topA = set(np.argsort(-orA)[:n_top].tolist())
    topB = set(np.argsort(-orB)[:n_top].tolist())
    jac = len(topA & topB) / len(topA | topB)

    R = {"dir_a": labA, "dir_b": labB, "cos_a_b": cos, "n_pairs": len(rows),
         "spearman_orA_orB": spearman(orA, orB),
         "top_frac": a.top_frac, "n_top": n_top,
         "top_bin_jaccard": jac,
         "top_bin_overlap_pct": 100.0 * len(topA & topB) / n_top}

    if a.behaviour_col in rows[0]:
        beh = np.array([float(r.get(a.behaviour_col) or "nan") for r in rows])
        R["behaviour_col"] = a.behaviour_col
        R["spearman_orA_behaviour"] = spearman(orA, beh)
        R["spearman_orB_behaviour"] = spearman(orB, beh)
        R["spearman_dvA_behaviour"] = spearman(dvA, beh)
        R["spearman_dvB_behaviour"] = spearman(dvB, beh)

    # explicit go / no-go on whether the expensive arm is worth running
    differ = (jac < 0.80) and (abs(R["spearman_orA_orB"]) < 0.95)
    R["verdict"] = ("RUN the training arms: the two rewards select materially different "
                    "training sets" if differ else
                    "SKIP the training arms: the two rewards select nearly the same training "
                    "set, so a post-training difference is not measurable")
    R["train_arms_worthwhile"] = bool(differ)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(R, f, indent=2)
    print("\n=== reward-direction head-to-head ===")
    for k2, v in R.items():
        print(f"  {k2}: {v}")
    print(f"\n[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
