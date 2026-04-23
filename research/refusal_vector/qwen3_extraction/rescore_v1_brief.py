#!/usr/bin/env python3
"""Rescore the 10 v1_brief adaptations using a second model's refusal feature.

Loads a Qwen3 (or any causal LM) refusal vector produced by
compute_refusal_vector.py, recomputes refusal_delta for each (original, adaptation)
pair, and writes side-by-side JSON + markdown comparing old vs new scores.
"""
import argparse
import json
import os
import re
import sys
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


V1_BRIEF_PAIRS: List[Dict] = []  # populated from --brief


def parse_brief_md(path: str) -> List[Dict]:
    """Parse pairs from v1_brief markdown: tables under '## v1' and '## v3' headings."""
    with open(path, "r") as f:
        text = f.read()
    pairs: List[Dict] = []
    current_section = None
    for line in text.splitlines():
        m = re.match(r"^##\s+(v\d+)", line)
        if m:
            current_section = m.group(1)
            continue
        if current_section and line.startswith("|") and not line.startswith("|---") and "Adaptation" not in line and "or_score" not in line:
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cols) >= 5:
                try:
                    or_score = float(cols[0])
                    refusal_delta = float(cols[1])
                    similarity = float(cols[2])
                except ValueError:
                    continue
                original = cols[3]
                adaptation = cols[4]
                pairs.append({
                    "section": current_section,
                    "or_score_llama": or_score,
                    "refusal_delta_llama": refusal_delta,
                    "similarity_llama": similarity,
                    "original": original,
                    "adaptation": adaptation,
                })
    return pairs


def load_refusal_vector(vector_path: str, select_layer: str = "max") -> Tuple[int, torch.Tensor]:
    data = np.load(vector_path, allow_pickle=True)
    vec = data["vector"]
    if vec.ndim == 1:
        layer = int(data["layer"])
        return layer, torch.from_numpy(vec.astype(np.float32))
    layers = data["layers"].astype(np.int32).tolist()
    if select_layer == "max":
        l2 = data["l2_per_layer"] if "l2_per_layer" in data.files else np.linalg.norm(vec, axis=1)
        idx = int(np.argmax(l2))
    else:
        layer_num = int(select_layer)
        idx = layers.index(layer_num)
    return int(layers[idx]), torch.from_numpy(vec[idx].astype(np.float32))


class RefusalScorer:
    def __init__(self, model_id: str, vector_path: str, select_layer: str = "max"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        print(f"[scorer] Loading {model_id} ...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=hf_token)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto", token=hf_token,
        )
        self.model.eval()
        self.layer, vec = load_refusal_vector(vector_path, select_layer)
        self.vec = vec.to(self.model.device, dtype=torch.float32)
        self.vec_norm = self.vec.norm().item() + 1e-9
        print(f"[scorer] Model loaded; refusal vector layer={self.layer} L2={self.vec_norm:.4f}", flush=True)

    def _chat(self, prompt: str) -> str:
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def score(self, prompts: List[str], batch_size: int = 4) -> np.ndarray:
        all_scores: List[float] = []
        # hidden_states[layer] where layer is 1-indexed over transformer blocks (0 = embeddings)
        hs_idx = self.layer  # layer count from extractor matches (1..num_layers)
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            chats = [self._chat(p) for p in batch]
            enc = self.tok(chats, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
            out = self.model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
            hs = out.hidden_states[hs_idx]  # [B, S, H]
            last_tok = hs[:, -1, :].to(torch.float32)
            s = torch.matmul(last_tok, self.vec) / self.vec_norm
            all_scores.extend(s.cpu().numpy().tolist())
        return np.array(all_scores, dtype=np.float32)


def compute_similarity(originals: List[str], rewrites: List[str],
                        st_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print(f"[sim] Loading {st_model}", flush=True)
    st = SentenceTransformer(st_model, device="cpu")
    a = st.encode(originals, convert_to_tensor=True, show_progress_bar=False)
    b = st.encode(rewrites, convert_to_tensor=True, show_progress_bar=False)
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brief", required=True, help="Path to v1_brief markdown")
    ap.add_argument("--vector", required=True, help="Path to Qwen refusal vector .npz")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--select_layer", default="max")
    ap.add_argument("--k", type=float, default=5.0)
    ap.add_argument("--c", type=float, default=0.75)
    ap.add_argument("--d", type=float, default=100.0)
    ap.add_argument("--output", required=True, help="JSON output path")
    args = ap.parse_args()

    pairs = parse_brief_md(args.brief)
    if len(pairs) == 0:
        print("ERROR: no pairs parsed from brief", file=sys.stderr)
        sys.exit(2)
    print(f"[rescore] Parsed {len(pairs)} pairs", flush=True)

    originals = [p["original"] for p in pairs]
    adaptations = [p["adaptation"] for p in pairs]

    scorer = RefusalScorer(args.model, args.vector, args.select_layer)
    print("[rescore] Scoring originals...", flush=True)
    r_orig = scorer.score(originals)
    print("[rescore] Scoring adaptations...", flush=True)
    r_rew = scorer.score(adaptations)
    refusal_delta_new = r_rew - r_orig

    sim_new = compute_similarity(originals, adaptations)
    sim_weight = np.exp(args.k * (sim_new - args.c))
    or_new = sim_weight * refusal_delta_new / args.d

    results = []
    for p, ro, rr, rd, sim, orn in zip(pairs, r_orig, r_rew, refusal_delta_new, sim_new, or_new):
        results.append({
            **p,
            "refusal_original_qwen": float(ro),
            "refusal_paraphrase_qwen": float(rr),
            "refusal_delta_qwen": float(rd),
            "similarity_qwen": float(sim),
            "or_score_qwen": float(orn),
        })

    with open(args.output, "w") as f:
        json.dump({
            "meta": {
                "model": args.model,
                "vector": os.path.basename(args.vector),
                "select_layer": args.select_layer,
                "layer_used": scorer.layer,
                "k": args.k, "c": args.c, "d": args.d,
            },
            "pairs": results,
        }, f, indent=2)
    print(f"[rescore] Wrote {args.output}", flush=True)

    # Quick markdown summary
    md_path = os.path.splitext(args.output)[0] + ".md"
    with open(md_path, "w") as f:
        f.write(f"# Rescored v1_brief adaptations with {args.model} refusal feature\n\n")
        f.write(f"- Layer used: {scorer.layer}\n- OR formula: exp({args.k}*(sim-{args.c})) * refusal_delta / {args.d}\n\n")
        for section in ["v1", "v3"]:
            rows = [r for r in results if r["section"] == section]
            if not rows:
                continue
            f.write(f"## {section}\n\n")
            f.write("| or_llama | or_qwen | Δrefusal_llama | Δrefusal_qwen | sim_llama | sim_qwen | original | adaptation |\n")
            f.write("|---------:|--------:|---------------:|--------------:|----------:|---------:|----------|------------|\n")
            for r in rows:
                orig = r["original"].replace("|", "\\|")
                adp = r["adaptation"].replace("|", "\\|")
                f.write(f"| {r['or_score_llama']:.3f} | {r['or_score_qwen']:.3f} | {r['refusal_delta_llama']:.2f} | {r['refusal_delta_qwen']:.2f} | {r['similarity_llama']:.3f} | {r['similarity_qwen']:.3f} | {orig[:60]}… | {adp[:60]}… |\n")
            f.write("\n")
    print(f"[rescore] Wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
