#!/usr/bin/env python3
"""Consolidate attacker rewrite shards -> a flat corpus CSV with MiniLM similarity.

Model-agnostic: reads gen_*_shard*.json = [{"alpaca_idx"|"idx", "original", "rewrites":[...]}]
and writes rows (pair_id, orig_idx, original, rewrite, similarity). Both the Llama and Qwen
scale-up tracks use this. MiniLM (all-MiniLM-L6-v2) cosine on normalized embeddings, same
metric the RWR reward/eval used. Optionally drops originals that collide with a train/eval set.
"""
import argparse, csv, glob, json, os, sys
csv.field_size_limit(sys.maxsize)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--glob", default="gen_*_shard*.json")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--exclude_json", default=None,
                    help="optional JSON list of original texts to drop (disjointness guard)")
    ap.add_argument("--st_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    excl = set()
    if args.exclude_json and os.path.exists(args.exclude_json):
        excl = set(x.strip() for x in json.load(open(args.exclude_json)))

    files = sorted(glob.glob(os.path.join(args.gen_dir, args.glob)))
    assert files, f"no files match {args.glob} in {args.gen_dir}"
    origs, rws, idxs = [], [], []
    for fp in files:
        for rec in json.load(open(fp)):
            o = (rec.get("original") or "").strip()
            oidx = rec.get("alpaca_idx", rec.get("idx", -1))
            if not o or o in excl:
                continue
            for rw in rec.get("rewrites", []):
                rw = (rw or "").strip()
                if not rw:
                    continue
                origs.append(o); rws.append(rw); idxs.append(oidx)
    print(f"[corpus] {len(files)} shards -> {len(origs)} rewrites "
          f"({len(set(idxs))} unique originals); {len(excl)} excluded texts", flush=True)

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(args.st_model)
    eo = st.encode(origs, batch_size=args.batch, normalize_embeddings=True, show_progress_bar=False)
    er = st.encode(rws, batch_size=args.batch, normalize_embeddings=True, show_progress_bar=False)
    sim = (eo * er).sum(1)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["pair_id", "orig_idx", "original", "rewrite", "similarity"])
        for i, (o, rw, oi) in enumerate(zip(origs, rws, idxs)):
            wtr.writerow([f"su_{i:06d}", oi, o, rw, f"{float(sim[i]):.4f}"])
    print(f"[done] wrote {args.out_csv} | sim mean={float(sim.mean()):.3f} "
          f"p10={float(sorted(sim)[len(sim)//10]):.3f}", flush=True)


if __name__ == "__main__":
    main()
