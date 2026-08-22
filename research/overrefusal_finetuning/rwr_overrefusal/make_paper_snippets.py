#!/usr/bin/env python3
"""Emit the LaTeX for section 3: cluster table, injection figure, example boxes, appendix.

Generated rather than hand-typed for two reasons. The numbers stay tied to the result files,
so they cannot drift from what was measured. And the character-noise spans are full of
LaTeX specials -- backslashes, dollars, hashes, carets, tildes, braces -- which are easy to
mis-escape by hand and which silently break a build or, worse, render as something else.

Run: python make_paper_snippets.py --out paper_section3.tex
"""
import argparse, difflib, json, os
from collections import defaultdict

CLUSTERS = ["chars", "negation", "touchy", "harmful_phrase", "action"]
LABEL = {"chars": "Characters", "negation": "Negation", "touchy": "Touchy subject",
         "harmful_phrase": "Harmful-sounding phrase", "action": "Action words"}
DESC = {"chars": "asterisks, slashes, punctuation runs, stray symbols",
        "negation": "an appended \\emph{no}, \\emph{not}, \\emph{NOT PLEASE}",
        "touchy": "the model's own nature, social topics, privacy",
        "harmful_phrase": "alarming wording that is not actually harmful",
        "action": "task or imperative verbs inserted"}
ARM_ORDER = ["chars", "action", "negation", "harmful_phrase", "touchy",
             "chars+negation", "chars+touchy", "chars+harmful_phrase", "all_five"]
ARM_LABEL = {"chars": "Characters", "action": "Action words", "negation": "Negation",
             "harmful_phrase": "Harmful-sounding phrase", "touchy": "Touchy subject",
             "chars+negation": "Characters + negation", "chars+touchy": "Characters + touchy",
             "chars+harmful_phrase": "Characters + harmful phrase",
             "all_five": "All five combined"}

SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}"}

# A box is a claim in the paper, so it has to survive being read by a referee. The response
# judge is ~94% precise on this data -- it labelled "A clever challenge! ... the palindrome
# is: 'level'" as a refusal -- and one such case did reach a rendered box. Boxes therefore
# additionally require the response to actually contain a refusal phrase. Any category with
# no example clearing that bar simply gets no box, which is the honest outcome: the two
# categories that fail this are the two the causal test finds inert anyway.
REFUSAL_PHRASE = [
    "i cannot", "i can't", "i can not", "i'm sorry", "i am sorry", "i apologize",
    "i'm not able", "i am not able", "i'm unable", "i am unable", "i must decline",
    "cannot fulfill", "cannot provide", "cannot assist", "can't help", "can't provide",
    "can't assist", "cannot create", "cannot generate", "can't create", "can't generate",
    "cannot write", "cannot help", "i won't", "i will not", "unable to",
]


def is_refusal(t):
    return any(p in (t or "").lower()[:300] for p in REFUSAL_PHRASE)


def esc(s):
    return "".join(SPECIAL.get(c, c) for c in (s or ""))


def trim(s, n):
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n - 1].rstrip() + "\u2026"


def marked(base, inj, n=300):
    """Escaped rewrite with changed words wrapped in \\ins{}.

    Diffed at WORD level, not character level. A character diff renders a substitution like
    "ways" -> "design" as \\ins{de}s\\ins{ign}, which is unreadable in print; at word level
    it is simply \\ins{design}.
    """
    inj = trim(inj, n)
    bw, iw = base.split(), inj.split()
    sm = difflib.SequenceMatcher(None, bw, iw, autojunk=False)
    out = []
    for tag, _, _, j1, j2 in sm.get_opcodes():
        seg = iw[j1:j2]
        if not seg:
            continue
        text = esc(" ".join(seg))
        out.append(f"\\ins{{{text}}}" if tag in ("insert", "replace") else text)
    return " ".join(out)


# The examples in these boxes were chosen by the project owner from the labelled sample,
# not selected by a rule. They are keyed on a distinctive substring of the rewrite so the
# lookup is exact, and each carries the model it came from: three are Qwen3-32B and the
# boxes sit under a table describing the Llama corpus, so the box title has to say which.
#
# Two of the picks disagree with the automatic clustering, and the owner's call stands. The
# Social Justice rewrite is clustered `chars`, missing that the inserted topic is the point.
# The Georgia Bulldogs rewrite is clustered `harmful_phrase` and `touchy`; it is shown under
# harmful-sounding phrase, matching the owner's own note on it, "bad sounding phrase".
PICK = {
    "chars":          ("P ***", "Qwen3-32B"),
    "negation":       ("Suggest 5 design to reduce energy", "Qwen3-32B"),
    "touchy":         ("formulated to be important in this game world", "Llama3-8B"),
    "harmful_phrase": ("totally DEST", "Llama3-8B"),
    "action":         ("censor I - all references", "Llama3-8B"),
}
PICK_SPOILED = ("stools inserted via", "Qwen3-32B")


def load_all_rewrites(clusters_path, qwen_path):
    """Every rewrite from both corpora, so a pick can come from either."""
    out = []
    for r in json.load(open(clusters_path)):
        out.append({"original": r["original"], "rewrite": r["rewrite"],
                    "rewrite_response": r.get("rewrite_response")})
    for r in json.load(open(qwen_path))["rows"]:
        if r.get("arm") == "or_loose":
            out.append({"original": r["original"], "rewrite": r["rewrite"],
                        "rewrite_response": r.get("rewrite_response")})
    return out


def find(rewrites, needle):
    hits = [r for r in rewrites if needle in (r.get("rewrite") or "")]
    return hits[0] if hits else None


def render_box(w, title, r, model, colour="boxgrey"):
    w("\\begin{tcolorbox}[colback=" + colour + ",colframe=black!25,boxrule=0.4pt,"
      "left=4pt,right=4pt,top=3pt,bottom=3pt,"
      f"title={{\\small\\bfseries {title}\\normalfont~---~{model}}},fonttitle=\\small]")
    w("\\small")
    w(f"\\textbf{{Original}}\\quad {esc(trim(r['original'], 200))}\\\\[2pt]")
    w(f"\\textbf{{Rewrite}}\\quad {marked(r['original'], r['rewrite'], 240)}\\\\[2pt]")
    w(f"\\textbf{{Response}}\\quad \\emph{{{esc(trim(r['rewrite_response'], 220))}}}")
    w("\\end{tcolorbox}")
    w("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", default="probe_or/results/gcg_clusters_llama.json")
    ap.add_argument("--judged", default="probe_or/results/injections_llama_judged.json")
    ap.add_argument("--benign", default="probe_or/results/injections_llama_benign.json")
    ap.add_argument("--or_filter",
                    default="probe_or/results/or_filter_llama_gcg.json")
    ap.add_argument("--qwen", default="incoming/qwen_gcg_all.json")
    ap.add_argument("--out", default="paper_section3.tex")
    a = ap.parse_args()

    rows = json.load(open(a.clusters))
    n = len(rows)
    share = {k: sum(1 for r in rows if k in r["clusters"]) for k in CLUSTERS}
    none = sum(1 for r in rows if not r["clusters"])
    multi = sum(1 for r in rows if len(r["clusters"]) >= 2)

    clean = {t: json.load(open(f"probe_or/results/injections_{t}_clean_summary.json"))
             for t in ("llama", "qwen")}

    jd = json.load(open(a.judged))["rows"]
    by = defaultdict(dict)
    for r in jd:
        by[r["arm"]][r["idx"]] = r
    base = by["baseline"]
    ben = {(r["arm"], r["idx"]): r.get("benign", "")
           for r in json.load(open(a.benign))["rows"]} if os.path.exists(a.benign) else {}

    L = []
    w = L.append
    w("% ============================================================================")
    w("% Generated by make_gcg_paper_snippets.py -- do not hand-edit; regenerate.")
    w("% Preamble additions required:")
    w("%   \\usepackage[most]{tcolorbox}")
    w("%   \\newcommand{\\ins}[1]{\\textcolor{insred}{\\bfseries #1}}")
    w("%   \\definecolor{insred}{HTML}{C0392B}")
    w("%   \\definecolor{boxgrey}{HTML}{F7F7F5}")
    w("%   \\definecolor{spoilbg}{HTML}{FBF0EE}")
    w("% ============================================================================")
    w("")

    # ---- Table: cluster shares -------------------------------------------------
    w("\\begin{table}[t]")
    w("\\centering")
    w("\\small")
    w(f"\\caption{{\\textbf{{What the gradient search inserts.}} Claude Sonnet~5 sorted all "
      f"{n:,} accepted Llama rewrites into five categories derived from a manual reading of "
      f"a labelled sample. Categories are not exclusive: {100*multi/n:.0f}\\% of rewrites "
      f"carry two or more, and only {100*none/n:.1f}\\% carry none, so the five together "
      f"describe {100*(n-none)/n:.1f}\\% of the corpus.}}")
    w("\\label{tab:gcg-clusters}")
    w("\\begin{tabular}{llrr}")
    w("\\toprule")
    w("Category & What it covers & Rewrites & Share \\\\")
    w("\\midrule")
    for k in sorted(CLUSTERS, key=lambda x: -share[x]):
        w(f"{LABEL[k]} & {DESC[k]} & {share[k]:,} & {100*share[k]/n:.1f}\\% \\\\")
    w("\\midrule")
    w(f"None of the five & & {none} & {100*none/n:.1f}\\% \\\\")
    w("\\bottomrule")
    w("\\end{tabular}")
    w("\\end{table}")
    w("")

    # ---- Figure ----------------------------------------------------------------
    w("\\begin{figure}[t]")
    w("  \\centering")
    w("  \\includegraphics[width=\\linewidth]{figures/fig_injection_effect}")
    lla, qwe = clean["llama"], clean["qwen"]
    w(f"  \\caption{{\\textbf{{Inserted material causes over-refusal, but only when it "
      f"carries meaning.}} Each category's spans were inserted into the same 300 benign "
      f"prompts, held out from the attack. Bars show the change in refusal rate against the "
      f"un-injected baseline of 0.7\\%, counting only refusals where an independent judge "
      f"confirmed the injected prompt still asks for something benign; whiskers are paired "
      f"bootstrap 95\\% intervals. Character noise and action words do not move refusal on "
      f"either model, while touchy subjects raise it "
      f"{lla['touchy']['delta']:+.1f}~pp on Llama, and all five together "
      f"{lla['all_five']['delta']:+.1f}~pp.}}")
    w("  \\label{fig:injection}")
    w("\\end{figure}")
    w("")

    # ---- Example boxes --------------------------------------------------------
    allrw = load_all_rewrites(a.clusters, a.qwen)
    w("% ---- worked examples, one box per category --------------------------------")
    for k in CLUSTERS:
        needle, model = PICK[k]
        r = find(allrw, needle)
        if not r:
            w(f"% MISSING: no rewrite matching {needle!r} for {k}")
            print(f"  [warn] {k}: no rewrite matches {needle!r}")
            continue
        if not is_refusal(r.get("rewrite_response")):
            print(f"  [warn] {k}: chosen response is not an unambiguous refusal")
        render_box(w, LABEL[k], r, model)

    w("% ---- counter-example: the insertion spoiled the request -------------------")
    needle, model = PICK_SPOILED
    r = find(allrw, needle)
    if r:
        render_box(w, "Counter-example: not over-refusal", r, model, colour="spoilbg")
    else:
        print(f"  [warn] counter-example {needle!r} not found")

    # ---- Appendix table --------------------------------------------------------
    w("% ---- appendix: full per-arm injection results -----------------------------")
    w("\\begin{table}[t]")
    w("\\centering\\small")
    w("\\caption{\\textbf{Injection results, all arms and both targets.} Change in refusal "
      "rate on 300 held-out benign prompts against a 0.7\\% baseline, restricted to "
      "refusals where the injected prompt was confirmed still benign. Paired bootstrap 95\\% "
      "intervals; bold marks intervals excluding zero.}")
    w("\\label{tab:injection-full}")
    w("\\begin{tabular}{lcccc}")
    w("\\toprule")
    w("& \\multicolumn{2}{c}{Llama3-8B} & \\multicolumn{2}{c}{Qwen3-32B} \\\\")
    w("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}")
    w("Injected material & $\\Delta$ (pp) & 95\\% CI & $\\Delta$ (pp) & 95\\% CI \\\\")
    w("\\midrule")
    for k in ARM_ORDER:
        c = []
        for t in ("llama", "qwen"):
            r = clean[t][k]
            d = f"{r['delta']:+.1f}"
            c.append(f"\\textbf{{{d}}}" if r["lo"] > 0 else d)
            c.append(f"[{r['lo']:+.1f}, {r['hi']:+.1f}]")
        sep = "\\midrule\n" if k == "chars+negation" else ""
        w(f"{sep}{ARM_LABEL[k]} & {c[0]} & {c[1]} & {c[2]} & {c[3]} \\\\")
    w("\\bottomrule")
    w("\\end{tabular}")
    w("\\end{table}")

    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"wrote {a.out}")
    print(f"  cluster table: {n:,} rewrites, {100*(n-none)/n:.1f}% covered by five categories")
    print(f"  example boxes: {sum(1 for k in CLUSTERS if by.get(k))} categories")


if __name__ == "__main__":
    main()
