#!/usr/bin/env python3
"""What vocabulary do the GCG attacks reach for, and is any of it about danger?

`compare_gcg_vs_rwr.py` answered a version of this for the Llama GCG corpus, but it
profiles the WHOLE rewrite, so the counts mix the original prompt's words in with the
ones the search inserted. The question here is what the attack ADDS, so the primary unit
is the added-word multiset (rewrite tokens minus original tokens); the whole-rewrite view
is kept alongside it for continuity with the existing GCG_VS_RWR.md numbers.

Five contrasts, in decreasing order of how much they assume:

  A  Qwen GCG vs RWR (llamaAtt confirmed-OR, and baseQwenAtt)
       The headline comparison, but the two corpora share ZERO originals, so a word
       difference conflates method with topic. Caveated, not removed.
  B  Llama GCG vs Qwen GCG, restricted to their 372 shared originals
       Same prompts, same method, two target models. Topic confound removed.
  C  or_* vs jb_*  (within Qwen GCG)
       Same method, same model, same search, OPPOSITE objective. If the vocabulary is
       just "whatever flips the opener", these should be mirror images of each other.
       Assumes nothing about another corpus.
  D  judge-confirmed vs judge-rejected, within the 957 judged or_loose rewrites
       If the rewrites a Sonnet judge accepted as over-refusal use the same inserted
       vocabulary as the ones it rejected, the label is not tracking the vocabulary.
  E  prohibition-cue rate per corpus
       Directly counts how often each method inserts a word that tells the model not to
       answer, which is the specific failure the owner flagged.

Run: python analyze_gcg_vocab.py --out GCG_VOCAB_ANALYSIS.md
"""
import argparse, csv, json, os, re, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_gcg_vs_rwr import logodds, WORD, STOP  # same estimator, same word regex

csv.field_size_limit(sys.maxsize)

# Words that read as an instruction to decline, rather than as danger vocabulary.
PROHIBITION = set("""no not never none nope cannot cant dont don won wont refuse reject
deny decline stop silent nothing without forbidden prohibited banned sorry apologize
apologise ignore disregard forget blank quiet hush""".split())
# The danger vocabulary the RWR attacker was shown to reach for (FRAME_ANALYSIS.md).
DANGER = set("""exploit exploiting exploitation weaponize weaponise weaponization covert
covertly conceal concealment infiltrate infiltration intrusion intrude exfiltrate
exfiltration sabotage sabotaging illicit clandestine surveil surveillance target
targeting compromise breach hack hacking malicious attack attacking bypass evade
evading forge forgery counterfeit smuggle launder""".split())


def toks(s, alpha_only=False):
    """Like compare_gcg_vs_rwr.toks, but never discards a prohibition word.

    That module's stoplist contains `no`, `not`, `don`, and it drops any token of two
    characters or fewer -- which removes `no` twice over. Those are precisely the tokens
    this analysis is about, so they are exempted here. Everything else is unchanged, so
    the non-prohibition rankings stay comparable to GCG_VS_RWR.md.
    """
    out = []
    for w in WORD.findall(s or ""):
        w = w.lower()
        if w in PROHIBITION:
            out.append(w)
        elif w not in STOP and len(w) > 2:
            if not alpha_only or w.isalpha():
                out.append(w)
    return out


def added_tokens(original, rewrite, alpha_only=False):
    """Multiset of tokens the rewrite has that the original did not.

    Multiset rather than set difference: a rewrite that repeats 'no' five times has
    inserted five prohibition tokens, and collapsing that to one understates it.
    """
    o, r = Counter(toks(original, alpha_only)), Counter(toks(rewrite, alpha_only))
    return list((r - o).elements())


def profile(rows, alpha_only=False, mode="added"):
    """-> term-freq Counter, distinct-original map, stats."""
    tf, df = Counter(), defaultdict(set)
    ntok = nonalpha = 0
    for i, r in enumerate(rows):
        orig = (r.get("original") or "").strip()
        rw = (r.get("rewrite") or "")
        key = orig or f"__row{i}"
        raw = rw.split()
        ntok += len(raw)
        nonalpha += sum(1 for t in raw if not WORD.fullmatch(t.strip(".,!?;:\"'()[]")))
        words = added_tokens(orig, rw, alpha_only) if mode == "added" else toks(rw, alpha_only)
        for w in words:
            tf[w] += 1
            df[w].add(key)
    return tf, df, dict(rows=len(rows), tokens=ntok,
                        nonalpha_rate=100 * nonalpha / ntok if ntok else 0.0)


def cue_rates(rows, alpha_only=False):
    """Share of rewrites that INSERT at least one prohibition / danger word."""
    n = len(rows)
    if not n:
        return dict(n=0, prohibition=0.0, danger=0.0, added_per_row=0.0)
    proh = dang = tot = 0
    for r in rows:
        a = set(added_tokens((r.get("original") or ""), (r.get("rewrite") or ""), alpha_only))
        tot += len(a)
        proh += bool(a & PROHIBITION)
        dang += bool(a & DANGER)
    return dict(n=n, prohibition=100.0 * proh / n, danger=100.0 * dang / n,
                added_per_row=tot / n)


def table(tf_a, df_a, tf_b, df_b, name_a, name_b, top, min_origs):
    z = logodds(tf_a, tf_b)
    def side(sign):
        c = [(w, s) for w, s in z.items() if (s > 0) == sign and len(df_a[w] if sign else df_b[w]) >= min_origs]
        c.sort(key=lambda x: -abs(x[1]))
        return c[:top]
    out = []
    for sign, na, nb, ta, tb, da in ((True, name_a, name_b, tf_a, tf_b, df_a),
                                     (False, name_b, name_a, tf_b, tf_a, df_b)):
        out.append(f"\n**Over-represented in {na}**\n")
        out.append(f"| word | z | n({na}) | origs | n({nb}) |")
        out.append("|---|--:|--:|--:|--:|")
        for w, s in side(sign):
            tag = " ⛔" if w in PROHIBITION else (" ⚠️" if w in DANGER else "")
            out.append(f"| `{w}`{tag} | {abs(s):.2f} | {ta.get(w,0)} | {len(da[w])} | {tb.get(w,0)} |")
    return "\n".join(out)


def load_qwen(path):
    d = json.load(open(path))
    return d["rows"], d


def load_llama_gcg(path):
    return json.load(open(path))["rows"]


def load_rwr_llama():
    v = {r["pair_id"]: r.get("is_or")
         for r in csv.DictReader(open("probe_or/results/corpus2/llamaAtt_judged.csv"))}
    return [{"original": r["original"], "rewrite": r["rewrite"]}
            for r in csv.DictReader(open("probe_or/results/corpus2/llamaAtt_refused.csv"))
            if v.get(r["pair_id"]) == "1"]


def load_rwr_qwen():
    return [{"original": r["original"], "rewrite": r["rewrite"]}
            for r in csv.DictReader(open("probe_or/results/corpus2/baseQwenAtt_rewrites.csv"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen_gcg", default="incoming/qwen_gcg_all.json")
    ap.add_argument("--llama_gcg", default="incoming/sonnet_filtered_strict.json")
    ap.add_argument("--mode", default="added", choices=["added", "whole"],
                    help="added = words the rewrite inserted; whole = full rewrite text")
    ap.add_argument("--alpha_only", action="store_true")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--min_origs", type=int, default=5)
    ap.add_argument("--out", default="GCG_VOCAB_ANALYSIS.md")
    a = ap.parse_args()

    qrows, qmeta = load_qwen(a.qwen_gcg)
    lg = load_llama_gcg(a.llama_gcg)
    rwr_l, rwr_q = load_rwr_llama(), load_rwr_qwen()
    arm = lambda n: [r for r in qrows if r["arm"] == n]
    q_or = arm("or_loose") + arm("or_strict")
    q_jb = arm("jb_loose") + arm("jb_strict")

    P = lambda rows: profile(rows, a.alpha_only, a.mode)
    L = []
    w = L.append
    w(f"# GCG rewrite vocabulary — what do these attacks actually insert?\n")
    w(f"Unit of analysis: **{'words the rewrite ADDED to the original' if a.mode=='added' else 'the whole rewrite text'}**. "
      f"Estimator: weighted log-odds with an informative Dirichlet prior (Monroe et al. 2008), "
      f"ranked by evidence. `origs` = distinct original prompts a word appears in, so a word "
      f"repeated inside a handful of prompts cannot look like a finding.\n")
    w("⛔ marks a prohibition word (reads as an instruction to decline). "
      "⚠️ marks danger vocabulary of the kind the RWR attacker reached for.\n")

    w("\n## Corpora\n")
    w("| corpus | rewrites | distinct originals | non-alpha token rate |")
    w("|---|--:|--:|--:|")
    for nm, rows in (("Qwen GCG (all arms)", qrows), ("Qwen GCG or_*", q_or),
                     ("Qwen GCG jb_*", q_jb), ("Llama GCG", lg),
                     ("RWR llamaAtt (confirmed-OR)", rwr_l), ("RWR baseQwenAtt (unjudged)", rwr_q)):
        _, _, st = P(rows)
        no = len({(r.get("original") or "").strip() for r in rows})
        w(f"| {nm} | {len(rows)} | {no} | {st['nonalpha_rate']:.1f}% |")

    w("\n## E. How often does each method insert a prohibition word?\n")
    w("The specific failure mode: a rewrite that tells the model not to answer is not "
      "eliciting over-refusal, it is issuing an instruction.\n")
    w("| corpus | n | inserts a ⛔ prohibition word | inserts a ⚠️ danger word | mean words added |")
    w("|---|--:|--:|--:|--:|")
    for nm, rows in (("Qwen GCG or_*", q_or), ("Qwen GCG jb_*", q_jb), ("Llama GCG", lg),
                     ("RWR llamaAtt (confirmed-OR)", rwr_l), ("RWR baseQwenAtt", rwr_q)):
        c = cue_rates(rows, a.alpha_only)
        w(f"| {nm} | {c['n']} | **{c['prohibition']:.1f}%** | {c['danger']:.1f}% | {c['added_per_row']:.1f} |")

    def contrast(title, note, ra, rb, na, nb):
        ta, da, _ = P(ra)
        tb, db, _ = P(rb)
        w(f"\n## {title}\n\n{note}\n")
        w(table(ta, da, tb, db, na, nb, a.top, a.min_origs))

    contrast("A1. Qwen GCG (or_*) vs RWR llamaAtt (confirmed-OR)",
             "The headline comparison. **Zero shared originals**, so differences conflate "
             "method with topic — read B and C before trusting this one.",
             q_or, rwr_l, "Qwen GCG or_*", "RWR llamaAtt")
    contrast("A2. Qwen GCG (or_*) vs RWR baseQwenAtt (model-matched, unjudged)",
             "Same target model as the GCG run, removing the model confound. baseQwenAtt is "
             "raw rewrites, not filtered to confirmed over-refusal. Still zero shared originals.",
             q_or, rwr_q, "Qwen GCG or_*", "RWR baseQwenAtt")

    shared = ({(r.get("original") or "").strip() for r in lg}
              & {(r.get("original") or "").strip() for r in q_or})
    lg_s = [r for r in lg if (r.get("original") or "").strip() in shared]
    q_s = [r for r in q_or if (r.get("original") or "").strip() in shared]
    contrast(f"B. Llama GCG vs Qwen GCG — {len(shared)} SHARED originals only",
             "Same prompts, same method, two target models. The topic confound is removed by "
             "construction, so anything here is a real method/model difference.",
             lg_s, q_s, "Llama GCG (shared)", "Qwen GCG (shared)")

    contrast("C. or_* vs jb_* within Qwen GCG — opposite objectives, same everything else",
             "`or_*` minimises the NLL of refusal openers, `jb_*` maximises it. Same search, "
             "same model, same code. If the inserted vocabulary is about danger, these should "
             "differ topically; if it is about steering the opener, they should be mirror "
             "images (prohibition vs permission). Assumes no other corpus.",
             q_or, q_jb, "Qwen GCG or_*", "Qwen GCG jb_*")

    conf = [r for r in qrows if r.get("judge_label") == "REFUSE" and r.get("judge_justified") == "NO"]
    rej = [r for r in qrows if r.get("judge_label") and r not in conf]
    contrast("D. Judge-confirmed vs judge-rejected, within judged or_loose",
             "Both sides are the same arm, same search, same originals pool. If the rewrites a "
             "Sonnet judge accepted insert the same words as the ones it rejected, the judged "
             "label is not tracking the inserted vocabulary.",
             conf, rej, "judge-confirmed", "judge-rejected")

    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"[done] wrote {a.out}  (mode={a.mode})")
    print(f"  qwen or_*={len(q_or)} jb_*={len(q_jb)} llamaGCG={len(lg)} "
          f"rwr_llama={len(rwr_l)} rwr_qwen={len(rwr_q)} shared={len(shared)} "
          f"confirmed={len(conf)} rejected={len(rej)}")


if __name__ == "__main__":
    main()
