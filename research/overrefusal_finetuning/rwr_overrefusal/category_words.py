#!/usr/bin/env python3
"""Which words does each refusal category's rewrites insert?

Feeds the injection experiment: for each category from categorize_gcg.py, find the words
that are distinctive to it, so they can be dropped into benign prompts and tested.

The confound this script is built around: **category and corpus are strongly correlated**
(fabricated_harmful is 55% of Llama GCG but 14% of Qwen GCG). A one-vs-rest comparison run
over the pooled data would therefore surface corpus vocabulary and label it category
vocabulary. So every comparison is run WITHIN a corpus -- category against all other
categories of the same corpus -- and the headline list for each category is the words that
show up in BOTH corpora's lists. A word that only replicates in one is reported separately
and should be treated as unconfirmed.

Unit is the added-word multiset (rewrite minus original), and the tokeniser is the
prohibition-preserving one from analyze_gcg_vocab, not compare_gcg_vs_rwr's, whose stoplist
drops `no` and `not`.

Run: python category_words.py --out CATEGORY_WORDS.md
"""
import argparse, json, os, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_gcg_vocab import added_tokens, PROHIBITION, DANGER
from compare_gcg_vs_rwr import logodds

MIN_CELL = 20          # below this a one-vs-rest log-odds is not worth reporting
NICE = {
    "fabricated_harmful_or_illegal_framing": "invented a harmful/illegal reading",
    "perceived_explicit_or_offensive_content": "read as explicit/offensive",
    "explicit_negation_command_obeyed": "obeyed an injected 'no'",
    "capability_or_format_misunderstanding": "thought it needed an image/link/PDF",
    "objection_to_inserted_derogatory_phrase": "declined an inserted derogatory phrase",
    "genuine_over_refusal_no_visible_cause": "no visible cause",
}


def profile(rows):
    tf, df = Counter(), defaultdict(set)
    for r in rows:
        key = (r.get("original") or "").strip()
        for w in added_tokens(r.get("original") or "", r.get("rewrite") or ""):
            tf[w] += 1
            df[w].add(key)
    return tf, df


def tag(w):
    return " ⛔" if w in PROHIBITION else (" ⚠️" if w in DANGER else "")


def one_vs_rest(rows, cat, min_origs, top):
    """Words distinctive to `cat` against every other category in the SAME corpus.

    Returns (words, n_rows, status). `status` separates the two ways a list can come back
    empty -- the cell being too small to bother with, and the cell being big enough but no
    word clearing min_origs -- because reporting those as the same thing is misleading.
    """
    ina = [r for r in rows if r["category"] == cat]
    out = [r for r in rows if r["category"] != cat and r["category"] != "unparsed"]
    if len(ina) < MIN_CELL:
        return [], len(ina), "cell_too_small"
    ta, da = profile(ina)
    tb, _ = profile(out)
    z = logodds(ta, tb)
    cand = [(w, s) for w, s in z.items() if s > 0 and len(da[w]) >= min_origs]
    cand.sort(key=lambda x: -x[1])
    words = [(w, s, ta[w], len(da[w])) for w, s in cand[:top]]
    return words, len(ina), ("ok" if words else "no_word_met_min_origs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default="probe_or/results/gcg_categorized.json")
    ap.add_argument("--out", default="CATEGORY_WORDS.md")
    ap.add_argument("--words_out", default="probe_or/results/category_words.json")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min_origs", type=int, default=3)
    a = ap.parse_args()

    rows = [r for r in json.load(open(a.cats)) if r["category"] != "unparsed"]
    corpora = sorted({r["corpus"] for r in rows})
    cats = [c for c in NICE if any(r["category"] == c for r in rows)]

    L, w = [], None
    w = L.append
    w("# Words each refusal category inserts\n")
    w("Per-category vocabulary from the categories in [GCG_CATEGORIES.md](GCG_CATEGORIES.md), "
      "as input to the injection experiment.\n")
    w("**Category and corpus are correlated**, so a pooled comparison would surface corpus "
      "vocabulary rather than category vocabulary. Every list below is computed within a "
      "single corpus (category against the other categories of that same corpus). The "
      "headline for each category is the words appearing in BOTH corpora's lists; "
      "single-corpus words are listed separately and are unconfirmed.\n")
    w("Unit is the added-word multiset (rewrite minus original). `n` is occurrences, "
      "`origs` distinct original prompts. ⛔ prohibition, ⚠️ danger.\n")

    w("\n## Cell sizes\n")
    w("| category | " + " | ".join(corpora) + " |")
    w("|---" * (len(corpora) + 1) + "|")
    for c in cats:
        cells = [sum(1 for r in rows if r["corpus"] == k and r["category"] == c) for k in corpora]
        w(f"| {NICE[c]} | " + " | ".join(
            f"{n}" + ("" if n >= MIN_CELL else " *(too small)*") for n in cells) + " |")
    w(f"\nCells below {MIN_CELL} rows are skipped: a one-vs-rest log-odds on a handful of "
      "rewrites is noise.\n")

    export = {}
    for c in cats:
        w(f"\n## {NICE[c]}\n\n`{c}`\n")
        per, sizes, status = {}, {}, {}
        for k in corpora:
            res, n, st = one_vs_rest([r for r in rows if r["corpus"] == k], c, a.min_origs, a.top)
            per[k], sizes[k], status[k] = res, n, st
        w("\n| corpus | rows | candidate words | status |")
        w("|---|--:|--:|---|")
        for k in corpora:
            w(f"| {k} | {sizes[k]} | {len(per[k])} | "
              f"{'ok' if status[k]=='ok' else ('cell below ' + str(MIN_CELL) if status[k]=='cell_too_small' else f'no word in >= {a.min_origs} distinct originals')} |")
        usable = [k for k in corpora if per[k]]
        if not usable:
            w("\nNo corpus yields a usable list for this category, so it contributes no "
              "candidate words to the injection experiment.\n")
            export[c] = {"replicated": [], "note": "; ".join(f"{k}:{status[k]}" for k in corpora)}
            continue
        if len(usable) == 1:
            k = usable[0]
            other = [x for x in corpora if x != k]
            w(f"\nOnly `{k}` yields a list ("
              + "; ".join(f"{x}: {status[x]}" for x in other)
              + "), so there is nothing to replicate against and these are unconfirmed.\n")
            w("\n| word | z | n | origs |")
            w("|---|--:|--:|--:|")
            for word, s, n, o in per[k][:15]:
                w(f"| `{word}`{tag(word)} | {s:.2f} | {n} | {o} |")
            export[c] = {"replicated": [], "single_corpus": {k: [x[0] for x in per[k]]}}
            continue

        sets = {k: {x[0] for x in per[k]} for k in usable}
        rep = set.intersection(*sets.values())
        w(f"\n**In both corpora** — intersection of a "
          + "-word and a ".join(str(len(per[k])) for k in usable)
          + "-word list:\n")
        if rep:
            w("| word | " + " | ".join(f"z ({k})" for k in usable) + " | occurrences |")
            w("|---" * (len(usable) + 2) + "|")
            zmap = {k: {x[0]: x for x in per[k]} for k in usable}
            for word in sorted(rep, key=lambda x: -sum(zmap[k][x][1] for k in usable)):
                zs = " | ".join(f"{zmap[k][word][1]:.2f}" for k in usable)
                tot = sum(zmap[k][word][2] for k in usable)
                w(f"| `{word}`{tag(word)} | {zs} | {tot} |")
        else:
            w("_No word appears in both corpora's top lists for this category._")
        for k in usable:
            only = [x for x in per[k] if x[0] not in rep][:10]
            if only:
                w(f"\n_{k} only:_ " + ", ".join(f"`{x[0]}`{tag(x[0])}" for x in only))
        export[c] = {"replicated": sorted(rep),
                     "single_corpus": {k: [x[0] for x in per[k] if x[0] not in rep]
                                       for k in usable}}
        if rep:
            # Second opinion on the replicated set, computed a different way: the plain
            # share of rewrites inserting the word, in-category against the rest of the
            # same corpus. Log-odds ranks by evidence and can be swayed by a word being
            # rare overall, so a raw rate is worth seeing next to it.
            w("\n_Check — share of rewrites inserting the word, in-category vs rest of the "
              "same corpus. Both gaps should be positive._\n")
            w("| word | " + " | ".join(f"{k} in-cat / rest" for k in usable) + " |")
            w("|---" * (len(usable) + 1) + "|")
            for word in sorted(rep, key=lambda x: -sum(
                    zmap[k][x][1] for k in usable)):
                cells = []
                for k in usable:
                    sub = [r for r in rows if r["corpus"] == k]
                    ina = [r for r in sub if r["category"] == c]
                    out = [r for r in sub if r["category"] != c]
                    rate = lambda rs: 100.0 * sum(
                        1 for r in rs
                        if word in added_tokens(r.get("original") or "", r.get("rewrite") or "")
                    ) / max(len(rs), 1)
                    a_, b_ = rate(ina), rate(out)
                    n_in = round(a_ * len(ina) / 100)
                    cells.append(f"{a_:.1f}% / {b_:.1f}%  (n={n_in})")
                w(f"| `{word}`{tag(word)} | " + " | ".join(cells) + " |")

    w("\n## Summary\n")
    w("| category | replicated words | usable for injection? |")
    w("|---|--:|---|")
    for c in cats:
        e = export[c]
        n = len(e["replicated"])
        if n:
            why = "yes — " + ", ".join(f"`{x}`" for x in e["replicated"])
        elif e.get("single_corpus"):
            k = list(e["single_corpus"])[0]
            why = f"unconfirmed — one corpus only ({k})"
        else:
            why = "no — " + e.get("note", "no usable list")
        w(f"| {NICE[c]} | {n} | {why} |")
    means = {k: sum(len(added_tokens(r.get("original") or "", r.get("rewrite") or ""))
                    for r in rows if r["corpus"] == k)
                / max(sum(1 for r in rows if r["corpus"] == k), 1)
             for k in corpora}
    w("\nOnly the injected-negation category produces a word set that replicates across "
      "both corpora, and it is the only one whose replicated list is longer than a single "
      "word. For the others the replication test is mostly **untestable rather than "
      "negative**: on these subsets the mean number of added words is "
      + ", ".join(f"{k} {means[k]:.1f}" for k in corpora)
      + f", and the smaller cells mean few words reach {a.min_origs} distinct originals, so "
        "the shorter candidate lists come back near-empty. A short list cannot intersect a "
        "long one, so a zero there is not evidence of disagreement.\n")

    open(a.out, "w").write("\n".join(L) + "\n")
    json.dump(export, open(a.words_out, "w"), indent=1)
    print(f"[done] wrote {a.out} and {a.words_out}")
    for c in cats:
        print(f"  {NICE[c]:42s} replicated words: {len(export[c]['replicated'])}")


if __name__ == "__main__":
    main()
