#!/usr/bin/env python3
"""Lexical comparison: GCG-generated rewrites vs the RWR (llama-logit) rewrites.

Question: which words does each method reach for? Uses the same weighted log-odds-ratio with
an informative Dirichlet prior (Monroe et al. 2008) as the edit-distance analysis, so results
are ranked by evidence rather than raw frequency, and reports DISTINCT-ORIGINAL counts because
a word appearing 40 times across 3 prompts is not a finding.

Two confounds this script measures rather than hides:
  1. GCG output is often ungrammatical / token-fragmenty. A raw comparison will surface that
     artifact rather than a strategy difference, so we report an OOV + non-alphabetic rate per
     corpus and (with --alpha_only) can restrict to dictionary-shaped tokens.
  2. If the two sets are built on different original prompts, vocabulary differences may just
     be topic differences. We report the overlap in originals and, when it is large enough,
     restrict to shared originals with --shared_only.

Input: --gcg accepts JSON (list of dicts, or dict-of-lists) or CSV. Field names are
auto-detected from common spellings; override with --gcg_rewrite_field / --gcg_original_field.

Run: python compare_gcg_vs_rwr.py --gcg <file> [--gcg_refused_only] [--shared_only] [--alpha_only]
"""
import argparse, csv, glob, json, math, os, re, sys
from collections import Counter, defaultdict
csv.field_size_limit(sys.maxsize)

WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
STOP = set("""a an the of to in for on with and or but is are was were be been being it its this
that these those as at by from into if then than so such not no nor do does did have has had i
you he she they we them his her their our your my me us can could will would shall should may
might must about over under again further once here there when where why how all any both each
few more most other some only own same too very s t just don now""".split())


def toks(s, alpha_only=False):
    t = [w.lower() for w in WORD.findall(s or "")]
    return [w for w in t if w not in STOP and len(w) > 2]


def load_any(path, rewrite_field=None, original_field=None):
    """-> list of {original, rewrite, (refused)} from JSON or CSV, tolerant of field names."""
    RW = [rewrite_field] if rewrite_field else ["rewrite", "rewritten", "adv_prompt", "attack",
                                                "adversarial", "output", "prompt_rewrite", "text",
                                                "generation", "candidate", "suffix_prompt"]
    OR = [original_field] if original_field else ["original", "orig", "prompt", "base_prompt",
                                                  "goal", "instruction", "source", "input"]
    RF = ["refused", "is_refused", "refusal", "refuse", "success", "jailbroken", "or"]
    if path.lower().endswith((".csv", ".tsv")):
        raw = list(csv.DictReader(open(path), delimiter="\t" if path.endswith(".tsv") else ","))
    else:
        d = json.load(open(path))
        if isinstance(d, dict):
            # dict-of-lists, or a wrapper with a single list value
            listvals = [v for v in d.values() if isinstance(v, list)]
            if len(listvals) == 1 and all(isinstance(x, dict) for x in listvals[0]):
                raw = listvals[0]
            elif listvals and all(isinstance(v, list) for v in d.values()):
                n = min(len(v) for v in d.values() if isinstance(v, list))
                raw = [{k: d[k][i] for k in d if isinstance(d[k], list)} for i in range(n)]
            else:
                raw = [d]
        else:
            raw = d
    if raw and not isinstance(raw[0], dict):        # plain list of strings
        return [{"original": "", "rewrite": str(x), "refused": None} for x in raw]
    keys = list(raw[0].keys())
    def pick(cands):
        for c in cands:
            for k in keys:
                if c and k.lower() == c.lower():
                    return k
        for c in cands:
            for k in keys:
                if c and c.lower() in k.lower():
                    return k
        return None
    kr, ko, kf = pick(RW), pick(OR), pick(RF)
    if kr is None:
        raise SystemExit(f"could not find a rewrite field in {keys}; pass --gcg_rewrite_field")
    print(f"[gcg] fields -> rewrite={kr!r} original={ko!r} refused={kf!r}", file=sys.stderr)
    out = []
    for r in raw:
        out.append({"original": str(r.get(ko, "") or ""), "rewrite": str(r.get(kr, "") or ""),
                    "refused": r.get(kf) if kf else None})
    return out


def load_rwr(confirmed_or_only=True):
    """Our llama-logit rewrites. Confirmed-OR set if available, else all refused."""
    rows = []
    ref = "probe_or/results/corpus2/llamaAtt_refused.csv"
    jud = "probe_or/results/corpus2/llamaAtt_judged.csv"
    if os.path.exists(ref) and os.path.exists(jud):
        v = {r["pair_id"]: r.get("is_or") for r in csv.DictReader(open(jud))}
        for r in csv.DictReader(open(ref)):
            if (not confirmed_or_only) or v.get(r["pair_id"]) == "1":
                rows.append({"original": r["original"], "rewrite": r["rewrite"]})
    if rows:
        return rows
    for f in sorted(glob.glob("probe_or/results/gen_batch2_llama_logit/*.json")):
        for r in json.load(open(f)):
            for x in r["rewrites"]:
                rows.append({"original": r["original"], "rewrite": x})
    return rows


def logodds(a_counts, b_counts, a0=10.0):
    """Monroe et al. weighted log-odds with informative Dirichlet prior. -> word: z."""
    vocab = set(a_counts) | set(b_counts)
    bg = {w: a_counts.get(w, 0) + b_counts.get(w, 0) for w in vocab}
    n_bg = sum(bg.values())
    na, nb = sum(a_counts.values()), sum(b_counts.values())
    z = {}
    for w in vocab:
        aw = a0 * bg[w] / n_bg if n_bg else 0.0
        ya, yb = a_counts.get(w, 0), b_counts.get(w, 0)
        if ya + yb < 1:
            continue
        try:
            la = math.log((ya + aw) / (na + a0 - ya - aw))
            lb = math.log((yb + aw) / (nb + a0 - yb - aw))
            var = 1.0 / (ya + aw) + 1.0 / (yb + aw)
            z[w] = (la - lb) / math.sqrt(var)
        except (ValueError, ZeroDivisionError):
            continue
    return z


def profile(rows, alpha_only):
    tf, df = Counter(), defaultdict(set)
    oov = nonalpha = ntok = 0
    for i, r in enumerate(rows):
        key = r["original"].strip() or f"__row{i}"
        raw = (r["rewrite"] or "").split()
        ntok += len(raw)
        nonalpha += sum(1 for t in raw if not WORD.fullmatch(t.strip(".,!?;:\"'()[]")))
        for w in toks(r["rewrite"], alpha_only):
            tf[w] += 1
            df[w].add(key)
    return tf, df, dict(tokens=ntok, nonalpha=nonalpha,
                        nonalpha_rate=100 * nonalpha / ntok if ntok else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcg", required=True)
    ap.add_argument("--gcg_rewrite_field")
    ap.add_argument("--gcg_original_field")
    ap.add_argument("--gcg_refused_only", action="store_true",
                    help="keep only GCG rows whose refusal/success field is truthy")
    ap.add_argument("--rwr_all", action="store_true",
                    help="compare against ALL llama-logit rewrites, not just confirmed-OR")
    ap.add_argument("--shared_only", action="store_true")
    ap.add_argument("--alpha_only", action="store_true")
    ap.add_argument("--min_origs", type=int, default=3)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default="GCG_VS_RWR.md")
    a = ap.parse_args()

    gcg = load_any(a.gcg, a.gcg_rewrite_field, a.gcg_original_field)
    if a.gcg_refused_only:
        keep = [r for r in gcg if str(r.get("refused")).strip().lower() in ("1", "true", "yes", "refused")]
        print(f"[gcg] refused-only: {len(keep)}/{len(gcg)}", file=sys.stderr)
        gcg = keep or gcg
    rwr = load_rwr(confirmed_or_only=not a.rwr_all)

    og = {r["original"].strip() for r in gcg if r["original"].strip()}
    orw = {r["original"].strip() for r in rwr if r["original"].strip()}
    shared = og & orw
    if a.shared_only and len(shared) >= 20:
        gcg = [r for r in gcg if r["original"].strip() in shared]
        rwr = [r for r in rwr if r["original"].strip() in shared]

    gtf, gdf, gstat = profile(gcg, a.alpha_only)
    rtf, rdf, rstat = profile(rwr, a.alpha_only)
    z = logodds(gtf, rtf)

    def table(sign, dfmap, tfmap, other_tf):
        rows = [(w, zz, tfmap.get(w, 0), len(dfmap.get(w, ())), other_tf.get(w, 0))
                for w, zz in z.items() if (zz > 0) == sign and len(dfmap.get(w, ())) >= a.min_origs]
        rows.sort(key=lambda t: -abs(t[1]))
        return rows[:a.top]

    lines = ["# GCG rewrites vs RWR (llama-logit) rewrites — lexical comparison\n",
             f"- GCG rows: **{len(gcg)}** ({len(og)} distinct originals)  ",
             f"- RWR rows: **{len(rwr)}** ({len(orw)} distinct originals)"
             f"{' — confirmed-OR only' if not a.rwr_all else ' — all rewrites'}  ",
             f"- originals shared by both sets: **{len(shared)}**"
             f"{' (restricted to these)' if a.shared_only and len(shared)>=20 else ''}  ",
             f"- non-alphabetic token rate: GCG **{gstat['nonalpha_rate']:.1f}%** vs "
             f"RWR **{rstat['nonalpha_rate']:.1f}%** "
             f"{'<- GCG fluency artifact; consider --alpha_only' if gstat['nonalpha_rate'] > 2*max(rstat['nonalpha_rate'],0.5) else ''}\n",
             "Weighted log-odds with informative Dirichlet prior (Monroe et al. 2008). "
             "`origs` = distinct original prompts the word appears in — a high count on few "
             "originals is not evidence.\n",
             "## Over-represented in GCG\n",
             "| word | z | n(GCG) | origs | n(RWR) |", "|---|--:|--:|--:|--:|"]
    for w, zz, n, o, other in table(True, gdf, gtf, rtf):
        lines.append(f"| `{w}` | {zz:.2f} | {n} | {o} | {other} |")
    lines += ["", "## Over-represented in RWR (llama-logit)\n",
              "| word | z | n(RWR) | origs | n(GCG) |", "|---|--:|--:|--:|--:|"]
    for w, zz, n, o, other in table(False, rdf, rtf, gtf):
        lines.append(f"| `{w}` | {abs(zz):.2f} | {n} | {o} | {other} |")
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:10]))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
