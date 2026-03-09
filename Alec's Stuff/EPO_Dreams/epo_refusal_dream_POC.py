#!/usr/bin/env python3
import os, sys, argparse, math, csv, random, time
from typing import List, Tuple, Dict, Any, Optional


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector_npz", required=True, help="Path to refusal_vector.npz or single-layer npz")
    ap.add_argument("--select_layer", default="max", help="'max' or integer index when vector has multiple layers")
    ap.add_argument("--last_k", type=int, default=1, help="Average over the last K user tokens for feature")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--population", type=int, default=12)
    ap.add_argument("--steps_per_member", type=int, default=200)
    ap.add_argument("--topk_per_pos", type=int, default=64)
    ap.add_argument("--positions_per_step", type=int, default=8, help="Number of positions (ranked by grad norm) to evaluate each step")
    ap.add_argument("--eval_per_pos", type=int, default=8, help="Number of candidate tokens evaluated per position")
    ap.add_argument("--max_user_tokens", type=int, default=48)
    ap.add_argument("--seeds_csv", default="", help="Optional CSV/txt of seed prompts (first column); if empty, use generic seeds")
    ap.add_argument("--num_seeds", type=int, default=12)
    ap.add_argument("--lambda_min", type=float, default=0.0)
    ap.add_argument("--lambda_max", type=float, default=2.0)
    ap.add_argument("--feature_weight", type=float, default=1.0, help="Multiplier on feature term in objective")
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test", type=int, default=0, help="If 1, shorter run for smoke test")
    return ap.parse_args()


def load_vector(npz_path: str, select_layer: str) -> Tuple[int, "np.ndarray"]:
    import numpy as np
    d = np.load(npz_path, allow_pickle=True)
    v = d["vector"]
    if v.ndim == 1:
        layer_1based = int(d["layer"].item() if "layer" in d else -1)
        if layer_1based < 0:
            raise ValueError("Single-layer npz missing 'layer'")
        vec = v.astype("float32")
        return layer_1based, vec
    elif v.ndim == 2:
        layers = d["layers"].astype("int32").tolist()
        if select_layer == "max":
            l2 = d["l2_per_layer"] if "l2_per_layer" in d else np.linalg.norm(v, axis=1)
            idx = int(np.argmax(l2))
        else:
            idx = int(select_layer)
            if idx < 0 or idx >= v.shape[0]:
                raise ValueError(f"select_layer index out of range: {idx}")
        layer_1based = int(layers[idx])
        vec = v[idx].astype("float32")
        return layer_1based, vec
    else:
        raise ValueError("Unexpected 'vector' shape")


def build_chat(tok, user_text: str) -> str:
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def find_user_span(tok, user_text: str) -> Tuple[List[int], List[int]]:
    """
    Returns (input_ids, user_indices) where user_indices are token positions that correspond
    to the user content region in the chat template (best-effort by subtracting an empty-user template).
    """
    chat_full = build_chat(tok, user_text)
    chat_empty = build_chat(tok, "")
    enc_full = tok(chat_full, return_tensors="pt", add_special_tokens=False)
    enc_empty = tok(chat_empty, return_tensors="pt", add_special_tokens=False)
    ids_full = enc_full["input_ids"][0].tolist()
    ids_empty = enc_empty["input_ids"][0].tolist()
    # Heuristic: user content appears as a contiguous suffix before the generation prompt marker(s)
    # Find longest common prefix
    pref = 0
    while pref < min(len(ids_full), len(ids_empty)) and ids_full[pref] == ids_empty[pref]:
        pref += 1
    # User content spans from pref to somewhere before the final template tail; we approximate by
    # taking everything after the empty template as user content.
    user_indices = list(range(pref, len(ids_full)))
    return ids_full, user_indices


def get_dtype(dtype_str):
    import torch
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    return torch.float32


def ce_on_user_tokens(logits, input_ids, user_indices) -> "torch.Tensor":
    """
    Cross-entropy over user content tokens. Next-token prediction loss for positions inside user_indices.
    logits: [1, S, V], input_ids: [1, S]
    """
    import torch
    # shift for teacher-forcing: predict token t using logits at t-1
    S = input_ids.shape[1]
    if S < 2:
        return torch.zeros((), device=logits.device)
    logp = torch.log_softmax(logits[:, :-1, :], dim=-1)  # [1, S-1, V]
    target = input_ids[:, 1:]  # [1, S-1]
    # mask only targets whose index is in user_indices
    mask = torch.zeros((1, S - 1), dtype=torch.bool, device=logits.device)
    for t in user_indices:
        if 1 <= t < S:
            mask[0, t - 1] = True
    # gather logp at targets
    tgt_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [1, S-1]
    sel = tgt_logp[mask]
    if sel.numel() == 0:
        return torch.zeros((), device=logits.device)
    return -sel.mean()


def feature_score_from_hidden(hidden_states, layer_1based: int, user_indices: List[int], last_k: int, vec_unit) -> "torch.Tensor":
    """
    hidden_states: tuple length L+1 with [emb, layer1, ..., layerL], each [1, S, H]
    Take last_k tokens within user_indices at specified layer and project on vec_unit.
    """
    import torch
    hs = hidden_states[layer_1based]  # [1, S, H]
    S = hs.shape[1]
    user_tok_indices = [i for i in user_indices if 0 <= i < S]
    if not user_tok_indices:
        return torch.zeros((), device=hs.device, dtype=hs.dtype)
    indices = user_tok_indices[-max(1, last_k):]
    sel = hs[:, indices, :]  # [1, k, H]
    mean_act = sel.mean(dim=1).squeeze(0)  # [H]
    return (mean_act.to(vec_unit.dtype) * vec_unit).sum()


def prepare_inputs_from_ids(model, tok, ids: List[int]):
    import torch
    ids_t = torch.tensor([ids], device=model.device)
    attn = torch.ones_like(ids_t, device=model.device)
    return {"input_ids": ids_t, "attention_mask": attn}


def forward_objective(model, tok, ids: List[int], user_indices: List[int], layer_1based: int, last_k: int, vec_unit, feature_weight: float, lambda_ce: float, need_grads: bool):
    import torch
    inputs = prepare_inputs_from_ids(model, tok, ids)
    inputs["output_hidden_states"] = True
    inputs["use_cache"] = False
    inputs["return_dict"] = True

    out = model(**inputs)
    feat = feature_score_from_hidden(out.hidden_states, layer_1based, user_indices, last_k, vec_unit)
    ce = ce_on_user_tokens(out.logits, inputs["input_ids"], user_indices)
    J = feature_weight * feat - (lambda_ce * ce)
    if not need_grads:
        return J, feat, ce, None

    # Create embedding grads by re-running with inputs_embeds that require grad
    emb = model.get_input_embeddings()
    with torch.no_grad():
        ids_t = inputs["input_ids"]
    embeds = emb(ids_t).detach().clone().requires_grad_(True)  # [1, S, H]
    out2 = model(inputs_embeds=embeds, attention_mask=inputs["attention_mask"], output_hidden_states=True, use_cache=False, return_dict=True)
    feat2 = feature_score_from_hidden(out2.hidden_states, layer_1based, user_indices, last_k, vec_unit)
    ce2 = ce_on_user_tokens(out2.logits, ids_t, user_indices)
    J2 = feature_weight * feat2 - (lambda_ce * ce2)
    model.zero_grad(set_to_none=True)
    if embeds.grad is not None:
        embeds.grad.zero_()
    J2.backward()
    grads = embeds.grad.detach()  # [1, S, H]
    return J.detach(), feat.detach(), ce.detach(), grads


def token_filter(tok) -> List[int]:
    # Allowed token ids: exclude special tokens
    specials = set(tok.all_special_ids or [])
    # Keep a cached allowed list by vocab size
    return [i for i in range(tok.vocab_size) if i not in specials]


def topk_replacements_for_pos(grads_pos, emb_weight, current_id: int, allow_ids: List[int], k: int) -> List[int]:
    import torch
    # Linearized objective: maximize dot(emb_new, grad). Ignore constant -dot(emb_old, grad)
    # emb_weight expected shape: [V, H]; grads_pos: [H]
    g = grads_pos.to(emb_weight.dtype).to(emb_weight.device)  # [H]
    scores = emb_weight @ g  # [V]
    scores[ current_id ] = -1e30
    if allow_ids is not None:
        mask = torch.full_like(scores, -1e30)
        mask[allow_ids] = 0.0
        scores = scores + mask
    topk = torch.topk(scores, k=min(k, scores.numel()), dim=0)
    return topk.indices.tolist()


def non_dominated_filter(points: List[Tuple[float,float,int]]) -> List[int]:
    # points: list of ( -feat, ce, idx ) so we minimize both for Pareto;
    # return list of idx on Pareto front
    pts = sorted(points)
    front: List[int] = []
    best_ce = float("inf")
    for neg_feat, ce, idx in pts:
        if ce < best_ce:
            front.append(idx)
            best_ce = ce
    return front


def main():
    args = parse_args()
    random.seed(args.seed)

    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    layer_1based, vec = load_vector(args.vector_npz, args.select_layer)
    vec = vec.astype("float32")
    vec = vec / (np.linalg.norm(vec) + 1e-9)

    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARNING: HUGGING_FACE_HUB_TOKEN not set; gated models may fail.", file=sys.stderr)

    dtype = get_dtype(args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, token=hf_token)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="auto", token=hf_token)

    # Seed prompts
    seeds: List[str] = []
    if args.seeds_csv and os.path.exists(args.seeds_csv):
        try:
            with open(args.seeds_csv, "r", encoding="utf-8", newline="") as f:
                r = csv.reader(f)
                for row in r:
                    if row:
                        seeds.append(row[0])
                        if len(seeds) >= args.num_seeds:
                            break
        except Exception:
            pass
    if not seeds:
        seeds = [
            "Please explain the benefits of regular physical exercise.",
            "Describe a typical day in a small coastal town.",
            "How does photosynthesis work in plants?",
            "What are effective time management strategies?",
            "Explain the process of baking a loaf of bread.",
            "Summarize the causes of the seasons.",
            "What factors influence climate in a region?",
            "Describe the lifecycle of a butterfly.",
            "How do vaccines help the immune system?",
            "Outline steps to troubleshoot a slow computer.",
            "What makes a compelling short story opening?",
            "Explain how musical scales are constructed.",
        ][: args.num_seeds]

    # Build population: tokens and user spans
    pop_ids: List[List[int]] = []
    pop_user_spans: List[List[int]] = []
    for s in seeds:
        ids, user_idx = find_user_span(tok, s)
        # Truncate user region to max_user_tokens by trimming from the left if needed
        if len(user_idx) > args.max_user_tokens:
            trim = len(user_idx) - args.max_user_tokens
            keep_user = user_idx[trim:]
            ids = ids[: user_idx[0]] + [ids[i] for i in keep_user]
            user_idx = list(range(user_idx[0], user_idx[0] + len(keep_user)))
        pop_ids.append(ids)
        pop_user_spans.append(user_idx)
    # If population larger than seed prompts, replicate
    while len(pop_ids) < args.population:
        i = len(pop_ids) % len(seeds)
        pop_ids.append(pop_ids[i][:])
        pop_user_spans.append(pop_user_spans[i][:])
    if len(pop_ids) > args.population:
        pop_ids = pop_ids[: args.population]
        pop_user_spans = pop_user_spans[: args.population]

    allow_ids = token_filter(tok)
    emb_weight = model.get_input_embeddings().weight.detach()
    vec_unit = torch.from_numpy(vec).to(model.device).to(dtype=model.dtype)

    # Lambda schedule
    lam_vals = [args.lambda_min + (args.lambda_max - args.lambda_min) * i / max(1, args.population - 1) for i in range(args.population)]
    if args.test:
        args.steps_per_member = min(args.steps_per_member, 20)
        args.topk_per_pos = min(args.topk_per_pos, 16)
        args.positions_per_step = min(args.positions_per_step, 4)
        args.eval_per_pos = min(args.eval_per_pos, 4)

    # Track best per member
    best_records: List[Dict[str, Any]] = []
    t0 = time.time()
    for m in range(args.population):
        lam = lam_vals[m]
        ids = pop_ids[m]
        user_idx = pop_user_spans[m]
        # Evaluate initial
        J, feat, ce, _ = forward_objective(model, tok, ids, user_idx, layer_1based, args.last_k, vec_unit, args.feature_weight, lam, need_grads=False)
        best = dict(ids=ids[:], feat=float(feat), ce=float(ce), J=float(J), lam=float(lam))

        for step in range(args.steps_per_member):
            J, feat, ce, grads = forward_objective(model, tok, ids, user_idx, layer_1based, args.last_k, vec_unit, args.feature_weight, lam, need_grads=True)
            if grads is None:
                break
            grads = grads[0]  # [S, H]

            # Rank user positions by gradient norm
            pos_scores: List[Tuple[float, int]] = []
            for pos in user_idx:
                if 0 <= pos < grads.shape[0]:
                    gn = float(grads[pos].norm().item())
                    pos_scores.append((gn, pos))
            pos_scores.sort(reverse=True)
            pos_list = [pos for _, pos in pos_scores[: max(1, args.positions_per_step)]]

            best_candidate: Optional[Dict[str, Any]] = None
            for pos in pos_list:
                top_ids = topk_replacements_for_pos(grads[pos], emb_weight, ids[pos], allow_ids, args.topk_per_pos)
                if not top_ids:
                    continue
                for new_tok in top_ids[: max(1, args.eval_per_pos)]:
                    if ids[pos] == new_tok:
                        continue
                    cand = ids[:]
                    cand[pos] = new_tok
                    J_new, feat_new, ce_new, _ = forward_objective(
                        model, tok, cand, user_idx, layer_1based, args.last_k, vec_unit, args.feature_weight, lam, need_grads=False
                    )
                    score = float(J_new)
                    if best_candidate is None or score > best_candidate["J"]:
                        best_candidate = dict(
                            ids=cand,
                            feat=float(feat_new),
                            ce=float(ce_new),
                            J=score,
                            lam=float(lam),
                        )

            if best_candidate is not None and best_candidate["J"] > float(J) + 1e-6:
                ids = best_candidate["ids"]
                J = torch.tensor(best_candidate["J"], device=model.device, dtype=feat.dtype)
                feat = torch.tensor(best_candidate["feat"], device=model.device, dtype=feat.dtype)
                ce = torch.tensor(best_candidate["ce"], device=model.device, dtype=feat.dtype)
                if best_candidate["J"] > best["J"]:
                    best = dict(
                        ids=ids[:],
                        feat=best_candidate["feat"],
                        ce=best_candidate["ce"],
                        J=best_candidate["J"],
                        lam=float(lam),
                    )
            else:
                # No further improvement this step
                break
            if (step + 1) % 20 == 0:
                print(f"[{m}/{args.population}] step {step+1} J={float(J):.4f} feat={float(feat):.4f} CE={float(ce):.4f} lam={lam:.3f}", flush=True)
        best_records.append(best)

    # Compute Pareto front across all members
    points = [(-rec["feat"], rec["ce"], i) for i, rec in enumerate(best_records)]
    front_idx = set(non_dominated_filter(points))

    # Save CSV
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "feat", "ce", "J", "lambda", "pareto"])
        for i, rec in enumerate(best_records):
            text = build_chat(tok, "").join([])  # placeholder to ensure tok defined; unused
            # Re-decode user content by subtracting the empty template prefix
            ids = rec["ids"]
            # Recover user content text best-effort: strip the empty template prefix tokens
            chat_empty = build_chat(tok, "")
            empty_ids = tok(chat_empty, add_special_tokens=False)["input_ids"]
            pref = 0
            while pref < min(len(ids), len(empty_ids)) and ids[pref] == empty_ids[pref]:
                pref += 1
            user_ids = ids[pref:]
            prompt_text = tok.decode(user_ids, skip_special_tokens=True)
            w.writerow([prompt_text, f"{rec['feat']:.6f}", f"{rec['ce']:.6f}", f"{rec['J']:.6f}", f"{rec['lam']:.6f}", 1 if i in front_idx else 0])
    dt = time.time() - t0
    print(f"[done] Saved {args.output_csv} in {dt/60:.1f} min")


if __name__ == "__main__":
    main()