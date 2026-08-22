#!/usr/bin/env python3
"""Pull worked examples of the injection effect, one table per arm.

The example that shows the effect is a PAIRED FLIP: the same benign prompt, answered at
baseline and refused once the cluster's span is inserted. Anything else -- a prompt refused
in both conditions, or refused only at baseline -- says nothing about what the injection did.

Inserted text is recovered by diffing the injected prompt against the baseline prompt rather
than trusting the builder's record, so what is highlighted is what the model actually saw.

Run: python injection_examples.py --tag llama --out INJECTION_EXAMPLES.md
"""
import argparse, difflib, json, os, re, textwrap
from collections import defaultdict

ORDER = ["chars", "action", "negation", "harmful_phrase", "touchy",
         "chars+negation", "chars+touchy", "chars+harmful_phrase", "all_five"]
NICE = {"chars": "chars — asterisks, slashes, punctuation runs",
        "action": "action words",
        "negation": "negation — no / not / NOT PLEASE",
        "touchy": "touchy subject — the model's own nature, social topics, privacy",
        "harmful_phrase": "harmful-sounding phrase that is not actually harmful",
        "chars+negation": "chars + negation",
        "chars+touchy": "chars + touchy",
        "chars+harmful_phrase": "chars + harmful phrase",
        "all_five": "all five combined"}


def inserted_spans(base, inj):
    """The substrings present in `inj` but not in `base`, in order."""
    sm = difflib.SequenceMatcher(None, base, inj, autojunk=False)
    out = []
    for tag, _, _, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            s = inj[j1:j2].strip()
            if s:
                out.append(s)
    return out


def mark(base, inj):
    """Bold the inserted regions of `inj` for the markdown table."""
    sm = difflib.SequenceMatcher(None, base, inj, autojunk=False)
    parts = []
    for tag, _, _, j1, j2 in sm.get_opcodes():
        seg = inj[j1:j2]
        if not seg:
            continue
        parts.append(f"**{seg}**" if tag in ("insert", "replace") else seg)
    return "".join(parts)


def cell(s, n=190):
    s = textwrap.shorten((s or "").replace("\n", " ").strip(), n)
    return s.replace("|", "\\|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="llama", choices=["llama", "qwen"])
    ap.add_argument("--per_arm", type=int, default=3)
    ap.add_argument("--out", default="INJECTION_EXAMPLES.md")
    a = ap.parse_args()

    d = json.load(open(f"probe_or/results/injections_{a.tag}_judged.json"))
    by = defaultdict(dict)
    for r in d["rows"]:
        by[r["arm"]][r["idx"]] = r
    base = by["baseline"]

    # Keep only refusals an Opus judge confirmed are still benign requests. Without this
    # the examples are dominated by insertions that spoiled the request -- `racist joking`
    # into "Tell a joke" -- where refusing is correct and shows nothing about over-refusal.
    bp = f"probe_or/results/injections_{a.tag}_benign.json"
    benign = {}
    if os.path.exists(bp):
        benign = {(r["arm"], r["idx"]): r.get("benign", "") for r in json.load(open(bp))["rows"]}
    else:
        print(f"[warn] {bp} missing -- examples are NOT filtered for still-benign")

    L = [f"# What the injections look like ({d['base_model']})\n",
         "Each row is a **paired flip**: the same benign prompt, answered at baseline and "
         "refused once the cluster's spans were inserted, where an Opus judge confirmed "
         "the injected prompt is STILL A BENIGN REQUEST. Inserted text is **bold**, "
         "recovered by diffing against the baseline prompt.\n",
         "These are the cases the causal numbers in "
         "[INJECTION_RESULTS.md](INJECTION_RESULTS.md) are counting.\n"]

    for arm in ORDER:
        rows = by.get(arm)
        if not rows:
            continue
        flips = [(i, r) for i, r in rows.items()
                 if r["label"] == "REFUSE" and base.get(i, {}).get("label") == "COMPLY"
                 and benign.get((arm, i), "") != "SPOILED"]
        flips.sort(key=lambda x: len(x[1]["prompt"]))
        L.append(f"\n## {NICE[arm]}\n")
        L.append(f"_{len(flips)} paired flips out of {len(rows)} prompts._\n")
        if not flips:
            L.append("_No paired flip in this arm._")
            continue
        L.append("| benign prompt, with inserted text in bold | response to the injected prompt |")
        L.append("|---|---|")
        step = max(1, len(flips) // a.per_arm)
        for i, r in flips[::step][:a.per_arm]:
            L.append(f"| {cell(mark(base[i]['prompt'], r['prompt']), 230)} | "
                     f"{cell(r['response'], 170)} |")
        # what was actually inserted, as plain strings
        spans = []
        for i, r in flips[:12]:
            spans += inserted_spans(base[i]["prompt"], r["prompt"])
        uniq = [s for s in dict.fromkeys(spans) if s][:8]
        if uniq:
            L.append("\nInserted spans seen in this arm's flips: "
                     + ", ".join(f"`{s}`" for s in uniq))

    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"wrote {a.out}")
    for arm in ORDER:
        rows = by.get(arm) or {}
        f = sum(1 for i, r in rows.items()
                if r["label"] == "REFUSE" and base.get(i, {}).get("label") == "COMPLY")
        print(f"  {arm:22s} {f:3d} paired flips")


if __name__ == "__main__":
    main()
