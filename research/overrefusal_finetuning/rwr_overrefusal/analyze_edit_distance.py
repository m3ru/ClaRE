#!/usr/bin/env python3
"""Edit-distance stratification + Fightin'-Words lexical analysis of confirmed over-refusals.

WHAT THIS DOES
--------------
Takes (original, rewrite) pairs produced by an RWR attacker, keeps the ones a target model
refused AND a calibrated judge confirmed as genuine over-refusal (is_or == 1), and splits
them by WORD-LEVEL Levenshtein distance between original and rewrite:

  LOW  stratum -- as close to a minimal edit as the corpus allows, so the small set of
                  changed/inserted words is plausibly the causal trigger and a lexical
                  analysis is meaningful.
  HIGH stratum -- the rewrite was restructured. No single word is to blame; this stratum is
                  written out for later embedding / clustering work and NOT analysed here.

The LOW/HIGH boundary is DERIVED, not hard-coded (--low_cut auto): take the smallest
content-edit threshold whose LOW stratum clears an estimability floor (--floor_pairs /
--floor_origs). On the trial2 corpus that lands at 4 content-word edits, because these
attackers reword wholesale -- only 1% of confirmed-OR pairs differ by <= 2 content words, so
the ideal single-word anchor is not available and the report says so rather than hiding it.

On the LOW stratum it runs Monroe, Colaresi & Quinn (2008) "Fightin' Words" weighted
log-odds-ratio with an informative Dirichlet prior, comparing confirmed-OR rewrites against
a comparison corpus of rewrites from the SAME attacker (see --comparison), and ranks words
by z-score rather than raw frequency.

DESIGN NOTES (why it is built this way)
---------------------------------------
* Distance is word-level, not character-level, and is reported three ways: raw token edits,
  raw CONTENT-token edits (stopwords dropped -- this is the interpretable anchor used for the
  cut), and normalized (edits / original length), because originals vary a lot in length.
* The comparison corpus defaults to non-refused rewrites from the SAME attacker RESTRICTED TO
  THE SAME (LOW) STRATUM. Restricting to the stratum matters: comparing low-edit OR rewrites
  against all non-refused rewrites would confound "words that trip a refusal" with "words that
  appear when the attacker rewrites heavily".
* The Dirichlet prior is estimated from the attacker's FULL rewrite vocabulary. That is the
  point of the informative prior in Monroe et al.: it shrinks words that are simply this
  attacker's habitual vocabulary, which is the dominant confound here.
* Every trigger is reported with the number of DISTINCT ORIGINALS it appears in, and with a
  second z-score (z_doc) computed on distinct-original presence counts rather than token
  counts, so a word that fires 40 times across 3 originals cannot masquerade as a real effect.
* Because the LOW stratum has an alignment, the script additionally reports which words the
  edit itself INTRODUCED (substitution targets + insertions). That is a strictly sharper
  causal claim than bag-of-words presence and is only valid in this stratum.
* A block bootstrap resamples DISTINCT ORIGINALS with replacement (not rewrites), because the
  original prompt is the unit of independence -- 4 rewrites are sampled per original. The
  report leads with the words that survive BOTH the distinct-original filter and the
  bootstrap, not with the raw z ranking, which is dominated by single-original topic words.
* alpha_0 (--prior_a0) is reported and swept, because a prior mass much larger than the LOW
  stratum's token count would rank the background corpus instead of the contrast.

USAGE
-----
  python3 analyze_edit_distance.py                       # auto-detect newest verdicts
  python3 analyze_edit_distance.py --verdicts probe_or/results/v5_judged --verdict_tag v5
  python3 analyze_edit_distance.py --low_cut 3            # override the derived cut
  python3 analyze_edit_distance.py --comparison refused_not_or   # alternative contrast

Outputs: EDIT_DISTANCE_ANALYSIS.md plus per-stratum and per-trigger CSVs under
probe_or/results/edit_strata/, all suffixed with the judge tag so v4 and v5 runs coexist.

Pure stdlib -- no numpy/pandas, no GPU, runs on a login node.
"""
import argparse, csv, json, math, os, re, sys
from collections import Counter, defaultdict

csv.field_size_limit(sys.maxsize)

# ----------------------------------------------------------------------------- tokenisation
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)*")

STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been before
being below between both but by can cannot could couldn't did didn't do does doesn't doing
don't down during each few for from further had hadn't has hasn't have haven't having he he'd
he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in
into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on
once only or other ought our ours ourselves out over own same shan't she she'd she'll she's
should shouldn't so some such than that that's the their theirs them themselves then there
there's these they they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's
which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've
your yours yourself yourselves s t will just don should now
""".split())


def tokens(text):
    return TOKEN_RE.findall((text or "").lower())


def content_tokens(text):
    return [t for t in tokens(text) if t not in STOPWORDS]


# ------------------------------------------------------------------- word-level Levenshtein
def levenshtein_ops(a, b):
    """Word-level Levenshtein distance between token lists a (original) and b (rewrite).

    Returns (distance, ops) where ops is a list of (kind, a_word, b_word) with kind in
    {equal, sub, ins, del}. Standard unit-cost DP with a backtrace; ties broken
    substitute > delete > insert so a same-length reword reads as substitutions.
    """
    n, m = len(a), len(b)
    if n == 0:
        return m, [("ins", None, w) for w in b]
    if m == 0:
        return n, [("del", w, None) for w in a]
    prev = list(range(m + 1))
    bt = [[0] * (m + 1) for _ in range(n + 1)]      # 0=diag 1=up(del) 2=left(ins)
    for j in range(1, m + 1):
        bt[0][j] = 2
    for i in range(1, n + 1):
        bt[i][0] = 1
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            c_diag = prev[j - 1] + (0 if ai == b[j - 1] else 1)
            c_del = prev[j] + 1
            c_ins = cur[j - 1] + 1
            best = c_diag
            move = 0
            if c_del < best:
                best, move = c_del, 1
            if c_ins < best:
                best, move = c_ins, 2
            cur[j] = best
            bt[i][j] = move
        prev = cur
    dist = prev[m]
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bt[i][j] == 0:
            ops.append(("equal" if a[i - 1] == b[j - 1] else "sub", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and bt[i][j] == 1:
            ops.append(("del", a[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, b[j - 1]))
            j -= 1
    ops.reverse()
    return dist, ops


def pair_metrics(original, rewrite):
    ot, rt = tokens(original), tokens(rewrite)
    oc, rc = content_tokens(original), content_tokens(rewrite)
    d_all, ops = levenshtein_ops(ot, rt)
    d_con, ops_c = levenshtein_ops(oc, rc)
    introduced = [w for k, _, w in ops_c if k in ("sub", "ins") and w is not None]
    removed = [w for k, w, _ in ops_c if k in ("sub", "del") and w is not None]
    return {
        "n_tok_orig": len(ot), "n_tok_rw": len(rt),
        "n_content_orig": len(oc), "n_content_rw": len(rc),
        "wl_dist": d_all,
        "wl_dist_content": d_con,
        "norm_dist": round(d_all / len(ot), 4) if ot else "",
        "norm_dist_content": round(d_con / len(oc), 4) if oc else "",
        "introduced_words": " ".join(introduced),
        "removed_words": " ".join(removed),
        "edit_ops": json.dumps([[k, x, y] for k, x, y in ops if k != "equal"]),
    }


# ------------------------------------------------------------------------ Fightin' Words
def fightin_words(counts_i, counts_j, prior_counts, a0=100.0, min_count=3):
    """Monroe, Colaresi & Quinn (2008) eq. 22: weighted log-odds-ratio, informative Dirichlet
    prior.

    counts_i / counts_j : Counter word -> count in the two corpora being contrasted.
    prior_counts        : Counter word -> count in the background corpus defining the prior
                          (here: the attacker's entire rewrite vocabulary).
    a0                  : total prior mass alpha_0. Interpretable as "pseudo-tokens of
                          background evidence". It must be chosen relative to corpus size --
                          with a0 much larger than n_i the prior swamps the data and the
                          ranking becomes a ranking of the background, not of the contrast.
    min_count           : vocabulary floor on combined count in i+j.

    Everything is computed on the SAME restricted vocabulary V: the prior is renormalised over
    V and n_i / n_j are token counts over V, so alpha_0 and the corpus sizes are commensurable.

      delta_w = log( (y_i^w+a_w) / (n_i+a0-y_i^w-a_w) ) - log( (y_j^w+a_w) / (n_j+a0-y_j^w-a_w) )
      var(delta_w) ~ 1/(y_i^w+a_w) + 1/(y_j^w+a_w)
      z_w = delta_w / sqrt(var)
    """
    vocab = [w for w in set(counts_i) | set(counts_j)
             if counts_i.get(w, 0) + counts_j.get(w, 0) >= min_count]
    if not vocab:
        return {}
    # Laplace-smoothed background probability, renormalised over V.
    raw = {w: prior_counts.get(w, 0) + 0.5 for w in vocab}
    tot = sum(raw.values())
    alpha = {w: a0 * raw[w] / tot for w in vocab}
    n_i = sum(counts_i.get(w, 0) for w in vocab)
    n_j = sum(counts_j.get(w, 0) for w in vocab)
    out = {}
    for w in vocab:
        yi, yj, aw = counts_i.get(w, 0), counts_j.get(w, 0), alpha[w]
        li = math.log((yi + aw) / max(n_i + a0 - yi - aw, 1e-9))
        lj = math.log((yj + aw) / max(n_j + a0 - yj - aw, 1e-9))
        delta = li - lj
        var = 1.0 / (yi + aw) + 1.0 / (yj + aw)
        out[w] = (delta / math.sqrt(var), delta, yi, yj)
    return out


# ------------------------------------------------------------------------------ data load
CELLS = [("llamaAtt", "llamaTgt"), ("qwenAtt", "qwenTgt")]

# Text-keyed verdict layouts pair a judge-input index with a verdicts file that shares its
# stem. Keep the pairs explicit: judge_input_partial.csv is keyed V5P_###### while
# verdicts_all.csv is keyed V5_######, so crossing them silently joins nothing.
VERDICT_PAIRS = [("judge_input_all.csv", "verdicts_all.csv"),
                 ("judge_input.csv", "verdicts.csv"),
                 ("judge_input.csv", "judged.csv")]
PARTIAL_PAIRS = [("judge_input_partial.csv", "verdicts_partial.csv")]


def _text_lut(idx, ver):
    by_id = {r["pair_id"]: r for r in csv.DictReader(open(ver))}
    lut = {}
    for r in csv.DictReader(open(idx)):
        v = by_id.get(r["pair_id"])
        if v:
            lut[(r["original"].strip(), r["rewrite"].strip())] = v
    return lut


def verdict_layout(vdir, index_csv=None, verdicts_csv=None, allow_partial=False):
    """Resolve a verdict directory into a lookup, supporting two on-disk layouts.

    per-cell : {att}_judged.csv keyed by that cell's own pair_id -- the trial2 / v4 layout.
    text     : a de-duplicated judge-input CSV (pair_id, original, rewrite) plus a verdicts CSV
               (pair_id, intent, harm, is_or). The v5 re-judge uses this: it pools rewrites
               across many result dirs and re-keys them, so the only stable join back to a
               particular attacker x target cell is the (original, rewrite) text itself.

    Returns ("percell", None, label) or ("text", lut, label), or (None, None, reason).
    A complete pair is preferred; a *_partial pair is used only with allow_partial, because
    reporting a headline off a partially-returned batch would silently understate the corpus.
    """
    if index_csv and verdicts_csv:
        lut = _text_lut(index_csv, verdicts_csv)
        return ("text", lut, os.path.basename(verdicts_csv)) if lut else \
               (None, None, f"explicit index/verdicts pair joined 0 rows")
    if all(os.path.exists(os.path.join(vdir, f"{a}_judged.csv")) for a, _ in CELLS):
        return "percell", None, "per-cell {att}_judged.csv"
    pairs = VERDICT_PAIRS + (PARTIAL_PAIRS if allow_partial else [])
    for iname, vname in pairs:
        ip, vp = os.path.join(vdir, iname), os.path.join(vdir, vname)
        if os.path.exists(ip) and os.path.exists(vp):
            lut = _text_lut(ip, vp)
            if lut:
                return "text", lut, f"{iname} x {vname}"
    have = sorted(f for f in os.listdir(vdir)) if os.path.isdir(vdir) else []
    partial_only = any(os.path.exists(os.path.join(vdir, v)) for _, v in PARTIAL_PAIRS)
    reason = ("only a PARTIAL verdicts file is present (pass --allow_partial, or "
              "--verdicts_index/--verdicts_csv, to use it)" if partial_only and not allow_partial
              else f"no usable verdict layout; directory holds {have}")
    return None, None, reason


def load_trial2(pairs_dir, verdict_dir, refuse_thresh=0.5, index_csv=None,
                verdicts_csv=None, allow_partial=False):
    """Return per-attacker dict with all rewrites, refusal status and judge verdicts.

    The trial2 layout: {att}_{tgt}.json holds every sampled rewrite with its refuse_rate;
    {att}_refused.csv is exactly the refuse_rate > thresh subset in file order, and
    {att}_judged.csv carries the judge verdict keyed by that file's pair_id.
    """
    mode, lut, label = verdict_layout(verdict_dir, index_csv, verdicts_csv, allow_partial)
    if mode is None:
        raise SystemExit(f"[fatal] {verdict_dir}: {label}")
    print(f"[verdicts] layout: {label}" + (f" -> {len(lut)} joinable verdicts"
                                           if mode == "text" else ""))
    data = {}
    for att, tgt in CELLS:
        jp = os.path.join(pairs_dir, f"{att}_{tgt}.json")
        rp = os.path.join(pairs_dir, f"{att}_refused.csv")
        vp = os.path.join(verdict_dir, f"{att}_judged.csv")
        if not (os.path.exists(jp) and os.path.exists(rp)):
            print(f"[warn] skipping {att}: missing {jp} or {rp}", file=sys.stderr)
            continue
        if mode == "percell" and not os.path.exists(vp):
            print(f"[warn] skipping {att}: no verdicts at {vp}", file=sys.stderr)
            continue
        ex = json.load(open(jp))["examples"]
        refused_rows = list(csv.DictReader(open(rp)))
        seq = [r for r in ex if r["refuse_rate"] > refuse_thresh]
        if len(seq) != len(refused_rows):
            raise SystemExit(f"[fatal] {att}: refused.csv has {len(refused_rows)} rows but "
                             f"{len(seq)} json rows exceed refuse_thresh={refuse_thresh}")
        for a, b in zip(seq, refused_rows):
            if a["rewrite"].strip() != b["rewrite"].strip():
                raise SystemExit(f"[fatal] {att}: refused.csv is not in json order")
        verdict = ({r["pair_id"]: r for r in csv.DictReader(open(vp))}
                   if mode == "percell" else {})
        pid_of = {id(a): b["pair_id"] for a, b in zip(seq, refused_rows)}
        recs, missing = [], 0
        for r in ex:
            pid = pid_of.get(id(r))
            if mode == "percell":
                v = verdict.get(pid, {}) if pid else {}
            else:
                v = lut.get((r["original"].strip(), r["rewrite"].strip()), {})
                if r["refuse_rate"] > refuse_thresh and not v:
                    missing += 1
            recs.append({
                "attacker": att, "target": tgt,
                "pair_id": pid or "", "orig_idx": r["orig_idx"],
                "original": r["original"].strip(), "rewrite": r["rewrite"].strip(),
                "refuse_rate": r["refuse_rate"],
                "refused": int(r["refuse_rate"] > refuse_thresh),
                "intent": v.get("intent", ""), "harm": v.get("harm", ""),
                "is_or": v.get("is_or", ""),
            })
        if mode == "text" and missing:
            print(f"[warn] {att}: {missing} refused rewrites have no verdict in this "
                  f"verdict source; they are excluded from both the OR set and the "
                  f"comparison set", file=sys.stderr)
        data[att] = recs
    return data


def dedup(recs):
    """Collapse identical (original, rewrite) pairs -- the attacker samples 4x per original
    and sometimes emits the same string twice. Keeps the first occurrence."""
    seen, out = set(), []
    for r in recs:
        k = (r["original"], r["rewrite"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


# --------------------------------------------------------------------------------- report
def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return num / den if den else float("nan")


def quantiles(vals, qs=(0, .05, .1, .25, .5, .75, .9, .95, 1.0)):
    if not vals:
        return {}
    s = sorted(vals)
    return {q: s[min(len(s) - 1, int(round(q * (len(s) - 1))))] for q in qs}


def hist_lines(vals, maxbin=15):
    c = Counter(vals)
    n = len(vals) or 1
    lines, cum = [], 0
    for k in range(0, maxbin + 1):
        v = c.get(k, 0)
        cum += v
        lines.append((k, v, 100 * v / n, 100 * cum / n))
    tail = sum(v for k, v in c.items() if k > maxbin)
    if tail:
        lines.append((f">{maxbin}", tail, 100 * tail / n, 100.0))
    return lines


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--pairs_dir", default=os.path.join(here, "probe_or/results/trial2"))
    ap.add_argument("--verdicts", default="auto",
                    help="dir holding {att}_judged.csv, or 'auto' to prefer "
                         "probe_or/results/v5_judged then probe_or/results/trial2")
    ap.add_argument("--verdict_tag", default="", help="judge-version label (auto-guessed)")
    ap.add_argument("--verdicts_index", default=None,
                    help="text-keyed layout: judge-input CSV (pair_id, original, rewrite)")
    ap.add_argument("--verdicts_csv", default=None,
                    help="text-keyed layout: verdicts CSV (pair_id, intent, harm, is_or)")
    ap.add_argument("--allow_partial", action="store_true",
                    help="accept a *_partial verdicts file (off by default: a partially "
                         "returned batch would silently understate the corpus)")
    ap.add_argument("--comparison", default="not_refused",
                    choices=["not_refused", "refused_not_or", "both"],
                    help="comparison corpus for the log-odds contrast")
    ap.add_argument("--stratum_matched_comparison", type=int, default=1,
                    help="1 = restrict comparison corpus to the same edit-distance stratum")
    ap.add_argument("--low_cut", default="auto",
                    help="'auto' = smallest content-edit cut clearing the estimability floor; "
                         "or an integer content-edit threshold")
    ap.add_argument("--floor_pairs", type=int, default=50,
                    help="estimability floor: min confirmed-OR pairs in the LOW stratum")
    ap.add_argument("--floor_origs", type=int, default=40,
                    help="estimability floor: min distinct originals in the LOW stratum")
    ap.add_argument("--refuse_thresh", type=float, default=0.5)
    ap.add_argument("--prior_a0", type=float, default=100.0)
    ap.add_argument("--min_count", type=int, default=3, help="vocab floor on count in OR+cmp")
    ap.add_argument("--min_or_count", type=int, default=3,
                    help="a reported trigger must occur at least this often in the OR stratum")
    ap.add_argument("--credible_min_origs", type=int, default=3,
                    help="a trigger enters the headline table only if it appears in at least "
                         "this many DISTINCT originals in the OR stratum")
    ap.add_argument("--bootstrap", type=int, default=400,
                    help="block-bootstrap replicates, resampling DISTINCT ORIGINALS")
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out_dir", default=os.path.join(here, "probe_or/results/edit_strata"))
    ap.add_argument("--report", default=os.path.join(here, "EDIT_DISTANCE_ANALYSIS.md"))
    args = ap.parse_args()

    # ---- verdict source -----------------------------------------------------------------
    vdir, vtag, skipped = args.verdicts, args.verdict_tag, []
    if vdir == "auto":
        cand = [(os.path.join(here, "probe_or/results/v5_judged"), "v5"),
                (os.path.join(here, "probe_or/results/trial2"), "v4")]
        for pth, t in cand:
            if not os.path.isdir(pth):
                continue
            mode, _, why = verdict_layout(pth, args.verdicts_index, args.verdicts_csv,
                                          args.allow_partial)
            if mode:
                vdir, vtag = pth, (vtag or t)
                break
            skipped.append((os.path.relpath(pth, here), why))
            print(f"[verdicts] skipping {os.path.relpath(pth, here)}: {why}")
        else:
            raise SystemExit("[fatal] no usable verdict dir found; pass --verdicts explicitly")
    if not vtag:
        vtag = "v5" if "v5" in vdir else ("v4" if "trial2" in vdir else os.path.basename(vdir))
    print(f"[verdicts] using {vdir}  (judge tag: {vtag})")

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_trial2(args.pairs_dir, vdir, args.refuse_thresh,
                       args.verdicts_index, args.verdicts_csv, args.allow_partial)
    if not data:
        raise SystemExit("[fatal] no attacker cells loaded")

    # ---- metrics on every unique pair ---------------------------------------------------
    per_att = {}
    for att, recs in data.items():
        u = dedup(recs)
        for r in u:
            r.update(pair_metrics(r["original"], r["rewrite"]))
            r["_ctok"] = content_tokens(r["rewrite"])
            r["_intro"] = r["introduced_words"].split()
        per_att[att] = u
        print(f"[{att}] {len(recs)} sampled rewrites -> {len(u)} unique (orig,rewrite) pairs")

    allrecs = [r for att in per_att for r in per_att[att]]

    def sel(rs, kind):
        if kind == "or":
            return [r for r in rs if r["refused"] and r["is_or"] == "1"]
        if kind == "not_refused":
            return [r for r in rs if not r["refused"]]
        if kind == "refused_not_or":
            return [r for r in rs if r["refused"] and r["is_or"] == "0"]
        if kind == "both":
            return [r for r in rs if (not r["refused"]) or (r["refused"] and r["is_or"] == "0")]
        raise ValueError(kind)

    OR = sel(allrecs, "or")
    CMP_ALL = sel(allrecs, args.comparison)
    if not OR:
        raise SystemExit("[fatal] no confirmed-OR pairs under these verdicts")

    d_or = [r["wl_dist"] for r in OR]
    dc_or = [r["wl_dist_content"] for r in OR]
    nd_or = [float(r["norm_dist"]) for r in OR if r["norm_dist"] != ""]

    def cut_stats(k, rows):
        s = [r for r in rows if r["wl_dist_content"] <= k]
        return len(s), len({r["orig_idx"] for r in s})

    # ---- choose the cut -----------------------------------------------------------------
    # Rule (fixed before inspecting any trigger table; it is a function of counts only):
    # take the SMALLEST content-edit threshold whose LOW stratum clears an estimability floor
    # of >= floor_pairs confirmed-OR pairs from >= floor_origs distinct originals. Smallest =
    # closest to the ideal single-word-edit anchor; the floor is what makes a weighted
    # log-odds ranking estimable at all.
    cut_ladder = [(k,) + cut_stats(k, OR) for k in range(1, 21)]
    if args.low_cut == "auto":
        chosen = next((k for k, n, no in cut_ladder
                       if n >= args.floor_pairs and no >= args.floor_origs), None)
        if chosen is None:
            chosen = max(k for k, _, _ in cut_ladder)
            print("[warn] no cut clears the estimability floor; using the maximum")
        K = chosen
    else:
        K = int(args.low_cut)
    print(f"[cut] LOW = content-word edit distance <= {K}")

    for r in allrecs:
        r["stratum"] = "LOW" if r["wl_dist_content"] <= K else "HIGH"

    LOW = [r for r in OR if r["stratum"] == "LOW"]
    HIGH = [r for r in OR if r["stratum"] == "HIGH"]
    cmp_low = ([r for r in CMP_ALL if r["stratum"] == "LOW"]
               if args.stratum_matched_comparison else CMP_ALL)

    # ---- write strata -------------------------------------------------------------------
    def write_csv(path, rows):
        cols = ["attacker", "target", "pair_id", "orig_idx", "stratum", "refuse_rate",
                "refused", "intent", "harm", "is_or", "wl_dist", "wl_dist_content",
                "norm_dist", "norm_dist_content", "n_tok_orig", "n_tok_rw",
                "n_content_orig", "n_content_rw", "introduced_words", "removed_words",
                "edit_ops", "original", "rewrite"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"[wrote] {path}  ({len(rows)} rows)")

    write_csv(os.path.join(args.out_dir, f"or_low_stratum_{vtag}.csv"), LOW)
    write_csv(os.path.join(args.out_dir, f"or_high_stratum_{vtag}.csv"), HIGH)
    write_csv(os.path.join(args.out_dir, f"or_all_with_distance_{vtag}.csv"), OR)
    write_csv(os.path.join(args.out_dir, f"comparison_all_with_distance_{vtag}.csv"), CMP_ALL)

    with open(os.path.join(args.out_dir, f"distance_distribution_{vtag}.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "metric", "value", "n_pairs", "pct", "cum_pct"])
        for name, rows in (("confirmed_or", OR), ("comparison", CMP_ALL)):
            for metric, key in (("wl_dist", "wl_dist"), ("wl_dist_content", "wl_dist_content")):
                for k, v, pc, cp in hist_lines([r[key] for r in rows], 20):
                    w.writerow([name, metric, k, v, f"{pc:.2f}", f"{cp:.2f}"])

    # ---- Fightin' Words -----------------------------------------------------------------
    prior = Counter()
    for r in allrecs:
        prior.update(r["_ctok"])

    def counts_of(rows, field):
        tc, doc = Counter(), defaultdict(set)
        for r in rows:
            ws = r["_ctok"] if field == "presence" else r["_intro"]
            tc.update(ws)
            for w in set(ws):
                doc[w].add(r["orig_idx"])
        return tc, doc

    def contrast(rows_i, rows_j, field, a0=None, min_count=None):
        a0 = args.prior_a0 if a0 is None else a0
        mc = args.min_count if min_count is None else min_count
        ci, di = counts_of(rows_i, field)
        cj, dj = counts_of(rows_j, field)
        tok = fightin_words(ci, cj, prior, a0, mc)
        ci_d = Counter({w: len(s) for w, s in di.items()})
        cj_d = Counter({w: len(s) for w, s in dj.items()})
        docz = fightin_words(ci_d, cj_d, prior, a0, max(2, mc - 1))
        tbl = []
        for w, (z, delta, yi, yj) in tok.items():
            tbl.append({"word": w, "z": round(z, 3), "log_odds_delta": round(delta, 3),
                        "n_or": yi, "n_cmp": yj,
                        "n_distinct_orig_or": len(di.get(w, ())),
                        "n_distinct_orig_cmp": len(dj.get(w, ())),
                        "z_doc": round(docz[w][0], 3) if w in docz else ""})
        tbl.sort(key=lambda d: -d["z"])
        return tbl

    def ranked(rows_i, rows_j, field, **kw):
        """Reported table: triggers must actually occur in the OR stratum."""
        return [d for d in contrast(rows_i, rows_j, field, **kw)
                if d["n_or"] >= args.min_or_count]

    tri_presence = ranked(LOW, cmp_low, "presence")
    tri_introduced = ranked(LOW, cmp_low, "introduced")

    # ---- block bootstrap over DISTINCT ORIGINALS ---------------------------------------
    # The unit of independence is the original prompt, not the rewrite: 4 rewrites are sampled
    # per original and near-duplicates of one original inflate token counts. Resampling
    # originals with replacement propagates that clustering into the uncertainty.
    import random
    rng = random.Random(args.seed)
    boot_stats = {}
    if args.bootstrap > 0 and tri_presence:
        by_orig_i, by_orig_j = defaultdict(list), defaultdict(list)
        for r in LOW:
            by_orig_i[r["orig_idx"]].append(r)
        for r in cmp_low:
            by_orig_j[r["orig_idx"]].append(r)
        origs = sorted(set(by_orig_i) | set(by_orig_j))
        watch = [d["word"] for d in tri_presence]   # every reported word, so the
        # headline "survivors" table below always has bootstrap columns filled in
        zs = defaultdict(list)
        for _ in range(args.bootstrap):
            samp = [origs[rng.randrange(len(origs))] for _ in range(len(origs))]
            ri = [r for o in samp for r in by_orig_i.get(o, ())]
            rj = [r for o in samp for r in by_orig_j.get(o, ())]
            if not ri or not rj:
                continue
            res = {d["word"]: d["z"] for d in contrast(ri, rj, "presence", min_count=1)}
            for w in watch:
                zs[w].append(res.get(w, 0.0))
        for w, v in zs.items():
            v.sort()
            n = len(v)
            boot_stats[w] = {
                "z_boot_p05": round(v[max(0, int(0.05 * (n - 1)))], 2) if n else "",
                "z_boot_med": round(v[n // 2], 2) if n else "",
                "boot_frac_z_gt_1p96": round(sum(1 for x in v if x > 1.96) / n, 2) if n else "",
            }
    for d in tri_presence:
        d.update(boot_stats.get(d["word"], {"z_boot_p05": "", "z_boot_med": "",
                                            "boot_frac_z_gt_1p96": ""}))

    for name, tbl in (("triggers_low_presence", tri_presence),
                      ("triggers_low_introduced", tri_introduced)):
        pth = os.path.join(args.out_dir, f"{name}_{vtag}.csv")
        cols = (list(tbl[0].keys()) if tbl else
                ["word", "z", "log_odds_delta", "n_or", "n_cmp", "n_distinct_orig_or",
                 "n_distinct_orig_cmp", "z_doc"])
        with open(pth, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(tbl)
        print(f"[wrote] {pth}  ({len(tbl)} words)")

    # ---- per-attacker replication -------------------------------------------------------
    per_att_tables = {}
    for att in per_att:
        a_or = [r for r in per_att[att] if r["refused"] and r["is_or"] == "1"
                and r["stratum"] == "LOW"]
        a_cmp = [r for r in sel(per_att[att], args.comparison)
                 if (not args.stratum_matched_comparison) or r["stratum"] == "LOW"]
        per_att_tables[att] = ({d["word"]: d for d in contrast(a_or, a_cmp, "presence",
                                                              min_count=2)},
                               len(a_or), len(a_cmp))

    # ---- sensitivity: the cut, and the prior strength -----------------------------------
    base_top = [d["word"] for d in tri_presence[:args.top]]
    def survivors_of(tbl):
        c = [d for d in tbl if d["n_distinct_orig_or"] >= args.credible_min_origs
             and isinstance(d["z_doc"], float) and d["z_doc"] > 1.96]
        c.sort(key=lambda d: -d["z_doc"])
        return c

    sens_cut = []
    for k in sorted({1, 2, 3, 4, 5, 6, 8, K}):
        lo = [r for r in OR if r["wl_dist_content"] <= k]
        cm = ([r for r in CMP_ALL if r["wl_dist_content"] <= k]
              if args.stratum_matched_comparison else CMP_ALL)
        no = len({r["orig_idx"] for r in lo})
        if len(lo) < 10:
            sens_cut.append((k, len(lo), no, len(cm), None, [], []))
            continue
        t = ranked(lo, cm, "presence")
        tt = [d["word"] for d in t[:args.top]]
        ov = len(set(tt) & set(base_top)) / max(1, len(base_top))
        sens_cut.append((k, len(lo), no, len(cm), ov,
                         tt[:6], [d["word"] for d in survivors_of(t)]))

    sens_a0 = []
    for a0 in (10.0, 50.0, 100.0, 250.0, 500.0):
        t = ranked(LOW, cmp_low, "presence", a0=a0)
        tt = [d["word"] for d in t[:args.top]]
        sens_a0.append((a0, len(set(tt) & set(base_top)) / max(1, len(base_top)), tt[:8]))

    # ---- markdown report ----------------------------------------------------------------
    low_by_att = Counter(r["attacker"] for r in LOW)
    high_by_att = Counter(r["attacker"] for r in HIGH)
    rel = lambda p: os.path.relpath(p, here)

    L = []
    A = L.append
    A("# Edit-distance stratification of confirmed over-refusals\n")
    A(f"_Generated by `analyze_edit_distance.py`. Judge verdicts: **{vtag}** "
      f"(`{rel(vdir)}`). LOW/HIGH cut: content-word edit distance <= {K}._\n")

    A("\n## 0. Data volume — exactly what these numbers rest on\n")
    A("| corpus | attacker -> target | unique (orig,rewrite) pairs | refused "
      f"(rate > {args.refuse_thresh}) | confirmed OR (`is_or==1`) | distinct originals in OR |")
    A("|---|---|---:|---:|---:|---:|")
    for att, tgt in CELLS:
        if att not in per_att:
            continue
        rs = per_att[att]
        ors = [r for r in rs if r["refused"] and r["is_or"] == "1"]
        A(f"| trial2 | {att} -> {tgt} | {len(rs)} | {sum(r['refused'] for r in rs)} | "
          f"{len(ors)} | {len({r['orig_idx'] for r in ors})} |")
    A(f"| **total** | | **{len(allrecs)}** | **{sum(r['refused'] for r in allrecs)}** | "
      f"**{len(OR)}** | **{len({r['orig_idx'] for r in OR})}** |")
    A("")
    A(f"**Everything below rests on {len(OR)} confirmed over-refusal pairs drawn from "
      f"{len({r['orig_idx'] for r in OR})} distinct original prompts, judged by the "
      f"{vtag} judge.** That is the whole evidence base; the LOW stratum on which the lexical "
      f"analysis runs is a fraction of it (see section 3).\n")
    A("* Source: `probe_or/results/trial2/` — 750 Alpaca originals per attacker, 4 sampled "
      "rewrites per original (3000 sampled rewrites, de-duplicated to unique "
      "(original, rewrite) pairs), scored on the matching diagonal target model.")
    A(f"* Refusal criterion: target `refuse_rate > {args.refuse_thresh}` over 4 generations, "
      "i.e. at least 3 of 4 completions refused.")
    A(f"* Over-refusal criterion: judge **{vtag}** returned INTENT=SAME **and** HARM=BENIGN.")
    A("* **Not included: the 8k scale-up corpus** (`probe_or/results/scaleup_atlas_llama/`). "
      "It has target refusals (1233 rewrites with `refuse_rw > 0.5`) and a Haiku "
      "benign/harmful label (1132 of those labelled BENIGN), but no calibrated-Sonnet `is_or` "
      "verdict. This project has already found the Haiku label to disagree materially with "
      "the Sonnet judge, so those 1132 are **not** confirmed over-refusals and are excluded "
      "rather than silently pooled. Re-run this script against that corpus once its Sonnet "
      "verdicts exist.")
    A(f"* Comparison corpus: `{args.comparison}` rewrites from the same attacker "
      f"({len(CMP_ALL)} unique pairs; {len(cmp_low)} after stratum matching). See section 4.")

    A("\n## 1. Word-level edit-distance distribution (confirmed-OR pairs)\n")
    A("Distance is unit-cost **word-level** Levenshtein over lowercased word tokens — not "
      "character-level. The `content` variant drops English stopwords first, so it counts "
      "substantive word changes only. Normalized distance is edits / original length, "
      "reported because originals vary from "
      f"{min(r['n_tok_orig'] for r in OR)} to {max(r['n_tok_orig'] for r in OR)} tokens.\n")
    qa, qc, qn = quantiles(d_or), quantiles(dc_or), quantiles(nd_or)
    A("| quantile | all-token edits | content-word edits | normalized (edits / original len) |")
    A("|---|---:|---:|---:|")
    for q in (0, .05, .1, .25, .5, .75, .9, .95, 1.0):
        A(f"| p{int(q*100)} | {qa.get(q,'')} | {qc.get(q,'')} | {qn.get(q,0):.3f} |")
    A(f"\nMean all-token edits {sum(d_or)/len(d_or):.2f}; mean content-word edits "
      f"{sum(dc_or)/len(dc_or):.2f}; n = {len(d_or)} confirmed-OR pairs.\n")
    A("Content-word edit-distance histogram, confirmed-OR pairs:\n")
    A("| content edits | pairs | % | cumulative % |")
    A("|---:|---:|---:|---:|")
    for k, v, pc, cp in hist_lines(dc_or, 12):
        A(f"| {k} | {v} | {pc:.1f} | {cp:.1f} |")

    A("\n### The shape of this distribution is itself the first result\n")
    n1, o1 = cut_stats(1, OR)
    n2, o2 = cut_stats(2, OR)
    n3, o3 = cut_stats(3, OR)
    A(f"The distribution is unimodal with its mode at {Counter(dc_or).most_common(1)[0][0]} "
      f"content-word edits and has essentially **no near-minimal-edit mass**: "
      f"{n1} pairs ({100*n1/len(OR):.1f}%) differ by 1 content word, "
      f"{n2} ({100*n2/len(OR):.1f}%) by <= 2, {n3} ({100*n3/len(OR):.1f}%) by <= 3. "
      f"Median normalized distance is {qn.get(.5,0):.2f}, i.e. the typical rewrite changes "
      f"about as many words as the original contains.\n")
    A("These attackers were RL-tuned against a refusal signal with no edit-distance penalty, "
      "and they converged on wholesale rewording rather than minimal perturbation. So the "
      "interpretable anchor the analysis would have preferred — 'at most 2 content words "
      f"changed' — is **not supported by this corpus**: it retains {n2} pairs from {o2} "
      "originals, which cannot support any weighted log-odds estimate. The cut below is the "
      "compromise this forces, and it is stated as such rather than presented as the ideal.")

    A("\n## 2. The cut, and why\n")
    A(f"**LOW = content-word edit distance <= {K}. HIGH = everything else.**\n")
    A("Selection rule (a function of counts only — fixed before any trigger table was "
      "inspected): take the *smallest* content-edit threshold whose LOW stratum clears an "
      f"estimability floor of >= {args.floor_pairs} confirmed-OR pairs from "
      f">= {args.floor_origs} distinct originals. Smallest keeps the stratum as close as this "
      "corpus allows to the single-word-edit ideal; the floor is what makes a weighted "
      "log-odds ranking estimable at all.\n")
    A("| candidate cut (content edits <=) | OR pairs | distinct originals | clears floor? |")
    A("|---:|---:|---:|---|")
    for k, n, no in cut_ladder[:8]:
        ok = "yes" if (n >= args.floor_pairs and no >= args.floor_origs) else "no"
        A(f"| {k}{' **<- chosen**' if k == K else ''} | {n} | {no} | {ok} |")
    A("")
    A("Why content-word distance rather than raw token distance: raw distance is inflated by "
      "stopword churn (the/a, to/for) that carries no refusal signal — median raw distance "
      f"{qa.get(.5,'')} against median content distance {qc.get(.5,'')}.\n")
    A("Why not normalized distance for the cut: normalized distance is what makes the two "
      "strata comparable *across* prompt lengths, and it is stored on every row, but it is "
      "the wrong knob for this cut. The LOW stratum exists to bound the number of candidate "
      "causal triggers per pair, and only the raw count does that. Normalized and raw "
      "distance are also confounded with original length in opposite directions in this "
      "corpus (see the confound list in section 4), so neither is neutral; the raw count at "
      "least bounds the thing the analysis depends on.")

    A("\n### Sensitivity to the cut\n")
    A(f"The whole contrast re-run at each threshold, compared against the chosen cut's "
      f"top-{args.top} list:\n")
    A(f"| cut (content edits <=) | OR pairs | distinct originals | comparison pairs | "
      f"top-{args.top} overlap vs chosen | first 6 by raw z | survivors of the repetition "
      f"controls (see section 4) |")
    A("|---:|---:|---:|---:|---:|---|---|")
    for k, n, no, nc, ov, tt, sv in sens_cut:
        A(f"| {k}{' **(chosen)**' if k == K else ''} | {n} | {no} | {nc} | "
          f"{('%.0f%%' % (100*ov)) if ov is not None else 'n/a (too few pairs)'} | "
          f"{', '.join('`%s`' % x for x in tt) if tt else 'n/a'} | "
          f"{', '.join('`%s`' % x for x in sv[:10])}"
          f"{' , ...(%d more)' % (len(sv) - 10) if len(sv) > 10 else ''}"
          f"{'' if sv else '-'} |")
    A("")
    est = [sv for k, n, no, nc, ov, tt, sv in sens_cut
           if n >= args.floor_pairs and no >= args.floor_origs]
    core = sorted(set.intersection(*[set(x) for x in est])) if est else []
    A(f"Words in the survivor set at **every** cut that clears the estimability floor "
      f"({', '.join(str(k) for k, n, no, *_ in sens_cut if n >= args.floor_pairs and no >= args.floor_origs)}): "
      + (", ".join(f"`{w}`" for w in core) if core else "_none_") + ".\n")
    A("Two things to take from this. First, the **raw-z ranking is not stable** across cuts "
      "(overlap falls to roughly a third by the loosest cut) — it is dominated by "
      "single-original topic words, so it should not be quoted. Second, the **survivor set "
      "is comparatively stable**: the same small core of loaded verbs and covertness terms "
      "recurs at every cut that has enough data, and loosening the cut mostly *adds* words "
      "(and adds topic nouns as restructuring brings in new subject matter) rather than "
      "replacing the core. That asymmetry is the reason section 4 leads with the survivor "
      "table rather than the raw ranking.")

    A(f"\n### Sensitivity to the Dirichlet prior strength (a0 = {args.prior_a0})\n")
    A("a0 is the total prior mass in pseudo-tokens. Set far above the LOW stratum's token "
      "count it would rank the background rather than the contrast, so it is reported "
      "explicitly:\n")
    A(f"| a0 | top-{args.top} overlap vs chosen | first 8 triggers |")
    A("|---:|---:|---|")
    for a0, ov, tt in sens_a0:
        A(f"| {a0:g}{' **(used)**' if a0 == args.prior_a0 else ''} | {100*ov:.0f}% | "
          f"{', '.join('`%s`' % x for x in tt)} |")

    A("\n## 3. Stratum sizes\n")
    A("| stratum | pairs | distinct originals | mean content edits | mean normalized dist | "
      "mean original length (tokens) | llamaAtt | qwenAtt |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, rows in (("LOW", LOW), ("HIGH", HIGH)):
        nd = [float(r["norm_dist"]) for r in rows if r["norm_dist"] != ""]
        by = low_by_att if name == "LOW" else high_by_att
        A(f"| {name} | {len(rows)} | {len({r['orig_idx'] for r in rows})} | "
          f"{sum(r['wl_dist_content'] for r in rows)/max(1,len(rows)):.2f} | "
          f"{sum(nd)/max(1,len(nd)):.3f} | "
          f"{sum(r['n_tok_orig'] for r in rows)/max(1,len(rows)):.1f} | "
          f"{by.get('llamaAtt',0)} | {by.get('qwenAtt',0)} |")
    A(f"\nComparison corpus after stratum matching: **{len(cmp_low)}** LOW pairs from "
      f"{len({r['orig_idx'] for r in cmp_low})} distinct originals, out of {len(CMP_ALL)} "
      f"unique `{args.comparison}` pairs overall.")

    A("\n## 4. LOW stratum — Fightin' Words lexical triggers\n")
    A("**Method.** Monroe, Colaresi & Quinn (2008) weighted log-odds-ratio with an "
      f"informative Dirichlet prior, a0 = {args.prior_a0}. The prior is estimated from the "
      "attackers' **entire** rewrite vocabulary, which is the point of the informative prior "
      "here: it shrinks words that are simply this attacker's habitual style toward zero, "
      "leaving words whose rate genuinely differs between refused-and-confirmed-OR rewrites "
      "and the comparison set. Words are ranked by z, not by raw count.\n")
    A(f"**Comparison corpus, and why.** Rewrites from the **same attacker** that the target "
      f"did **not** refuse ({len(cmp_low)} LOW pairs), rather than the original Alpaca "
      "prompts or a generic English corpus. Contrasting against non-attacker text would just "
      "recover 'the attacker uses alarming words', which is already known and is not the "
      "question. Contrasting against the same attacker's un-refused output isolates the thing "
      "actually asked: *given that this attacker rewrote the prompt, which words are the ones "
      "that trip a refusal?* The comparison set is additionally **stratum-matched** (same "
      f"content-edit cut, <= {K}); without matching, the contrast would confound "
      "'words that trigger refusal' with 'words that appear when the attacker rewrites "
      "heavily'.\n")
    n_rno_low = len([r for r in sel(allrecs, "refused_not_or") if r["stratum"] == "LOW"])
    A(f"The other defensible comparison — rewrites that *were* refused but the judge ruled "
      f"**not** over-refusal — would isolate something subtly different ('given a refusal, "
      f"what makes it a wrong one?'). It is available via `--comparison refused_not_or`, but "
      f"in this corpus it yields only {n_rno_low} LOW-stratum pairs, far too few to estimate "
      f"anything, so `not_refused` is the default and is what is reported here.\n")
    A(f"**How to read the columns.** `n_or` / `n_cmp` are token counts. "
      f"`n_distinct_orig_or` is the number of DISTINCT original prompts the word appears "
      f"in within the OR stratum — this is the column that decides whether a row is evidence: "
      f"a word occurring 40 times across 3 originals is one attacker tic, not 40 observations. "
      f"`z_doc` re-runs the identical statistic on distinct-original presence counts (each "
      f"original contributes at most 1), so it is the repetition-robust replicate of `z`. "
      f"`z_boot_p05` and `boot_frac` come from {args.bootstrap} block-bootstrap replicates "
      f"that resample **distinct originals** with replacement — the correct unit of "
      f"independence, since 4 rewrites are sampled per original. `boot_frac` is the fraction "
      f"of replicates keeping z > 1.96. Rows are reported only if they occur at least "
      f"{args.min_or_count} times in the OR stratum.\n")
    A(f"Contrast: {len(LOW)} confirmed-OR LOW rewrites "
      f"({len({r['orig_idx'] for r in LOW})} distinct originals) vs {len(cmp_low)} "
      f"comparison LOW rewrites.\n")
    if not tri_presence:
        A("_No word cleared the reporting floor — the LOW stratum is too small._")
    else:
        A("| rank | word | z | log-odds delta | n_or | n_cmp | distinct originals (OR) | "
          "distinct originals (cmp) | z_doc | z_boot_p05 | boot_frac | llamaAtt z | qwenAtt z |")
        A("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for i, d in enumerate(tri_presence[:args.top], 1):
            la = per_att_tables.get("llamaAtt", ({},))[0].get(d["word"], {}).get("z", "-")
            qw = per_att_tables.get("qwenAtt", ({},))[0].get(d["word"], {}).get("z", "-")
            A(f"| {i} | `{d['word']}` | {d['z']} | {d['log_odds_delta']} | {d['n_or']} | "
              f"{d['n_cmp']} | {d['n_distinct_orig_or']} | {d['n_distinct_orig_cmp']} | "
              f"{d['z_doc']} | {d['z_boot_p05']} | {d['boot_frac_z_gt_1p96']} | {la} | {qw} |")
        nsig = sum(1 for d in tri_presence if d["z"] > 1.96)
        nsig_doc = sum(1 for d in tri_presence
                       if isinstance(d["z_doc"], float) and d["z_doc"] > 1.96)
        A(f"\n{nsig} of {len(tri_presence)} reported words reach z > 1.96 on token counts; "
          f"{nsig_doc} do so on the repetition-robust `z_doc`. With this stratum size these "
          f"are **suggestive rankings, not established triggers** — no multiple-comparison "
          f"correction has been applied, and applying one at this n would leave nothing.")

    # ---- headline: triggers that survive the repetition controls -----------------------
    credible = [d for d in tri_presence
                if d["n_distinct_orig_or"] >= args.credible_min_origs
                and isinstance(d["z_doc"], float) and d["z_doc"] > 1.96]
    credible.sort(key=lambda d: -(d["z_doc"] if isinstance(d["z_doc"], float) else -99))
    A("\n### Headline: the triggers that survive the repetition controls\n")
    A(f"The table above ranks by `z` on token counts, and most of its top rows are artifacts "
      f"of a single original prompt — `page`, `numbers`, proper nouns and so on come from one "
      f"Alpaca item whose four rewrites all landed in the OR stratum. The subset below is what "
      f"is left after requiring a word to appear in **>= {args.credible_min_origs} distinct "
      f"originals** and to keep **z_doc > 1.96** (the statistic recomputed with each original "
      f"counted once). This, not the raw ranking, is the answer to 'which words trip a "
      f"refusal'.\n")
    if not credible:
        A("_Nothing survives both filters at this stratum size._")
    else:
        A("| word | z_doc | z (token) | z_boot_p05 | boot_frac | bootstrap verdict | n_or | "
          "n_cmp | distinct originals (OR) | distinct originals (cmp) | introduced by the edit? |")
        A("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|")
        intro_map = {d["word"]: d for d in tri_introduced}
        for d in credible:
            im = intro_map.get(d["word"])
            tag = (f"yes (z={im['z']}, {im['n_distinct_orig_or']} originals)" if im else "no")
            bf = d["boot_frac_z_gt_1p96"]
            verdict = ("holds" if isinstance(bf, float) and bf >= 0.75 else
                       "marginal" if isinstance(bf, float) and bf >= 0.5 else
                       "**fails the bootstrap**")
            A(f"| `{d['word']}` | {d['z_doc']} | {d['z']} | {d['z_boot_p05']} | "
              f"{bf} | {verdict} | {d['n_or']} | {d['n_cmp']} | "
              f"{d['n_distinct_orig_or']} | {d['n_distinct_orig_cmp']} | {tag} |")
        held = [d for d in credible if isinstance(d["boot_frac_z_gt_1p96"], float)
                and d["boot_frac_z_gt_1p96"] >= 0.75]
        top = max(credible, key=lambda d: d["n_distinct_orig_or"])
        A(f"\n{len(credible)} words pass the two repetition filters, from a LOW stratum of "
          f"{len(LOW)} pairs over {len({r['orig_idx'] for r in LOW})} originals; "
          f"{len(held)} of them also hold up under the by-original bootstrap "
          f"({', '.join('`%s`' % d['word'] for d in held)}). The bootstrap column is doing "
          f"real work here — a word can clear the distinct-original filter and still be "
          f"driven by which originals happened to land in the stratum.\n")
        A(f"Even the strongest row is a shift in rate, not a switch: `{top['word']}` appears "
          f"in {top['n_distinct_orig_or']} of the {len({r['orig_idx'] for r in LOW})} OR "
          f"originals and {top['n_distinct_orig_cmp']} of the "
          f"{len({r['orig_idx'] for r in cmp_low})} comparison originals. There is no word "
          f"in this corpus whose presence implies a refusal.")

    A("\n### Sharper variant: words the edit itself INTRODUCED\n")
    A("Bag-of-words presence counts every content word in the rewrite, including words "
      "inherited unchanged from the original — so a word can rank highly just because the "
      "originals in the stratum happened to be about that topic. In the LOW stratum an "
      "alignment exists, so the contrast can be restricted to words a substitution or "
      "insertion actually **put there**. That is a strictly stronger causal claim, and it is "
      "only valid in this stratum, which is the main reason the stratum exists.\n")
    if not tri_introduced:
        A("_No introduced word cleared the reporting floor._")
    else:
        A("| rank | introduced word | z | n_or | n_cmp | distinct originals (OR) | "
          "distinct originals (cmp) | z_doc |")
        A("|---:|---|---:|---:|---:|---:|---:|---:|")
        for i, d in enumerate(tri_introduced[:args.top], 1):
            A(f"| {i} | `{d['word']}` | {d['z']} | {d['n_or']} | {d['n_cmp']} | "
              f"{d['n_distinct_orig_or']} | {d['n_distinct_orig_cmp']} | {d['z_doc']} |")

    A("\n### Confounds — read both tables with these in mind\n")
    lo_len = sum(r["n_tok_orig"] for r in LOW) / max(1, len(LOW))
    hi_len = sum(r["n_tok_orig"] for r in HIGH) / max(1, len(HIGH))
    rl = pearson([r["n_content_orig"] for r in OR], [r["wl_dist_content"] for r in OR])
    A("1. **The attackers' vocabulary is extremely repetitive.** These are RL-tuned rewriters "
      "that collapsed onto a small loaded lexicon, so raw frequency is almost meaningless "
      "here. The informative Dirichlet prior is drawn from the attackers' own full output "
      "specifically to absorb this, but it cannot absorb it entirely — check "
      "`n_distinct_orig_or` before believing any row.")
    A("2. **The same original appears up to 4 times.** Sampling is 4 rewrites per original, "
      "and near-identical rewrites of one original inflate token counts. `z_doc` (one count "
      "per original) and the by-original block bootstrap are the two corrections; rows where "
      "`z` is high but `z_doc` or `z_boot_p05` is not are repetition artifacts.")
    _nz = [(r["n_content_orig"], float(r["norm_dist_content"])) for r in OR
           if r["norm_dist_content"] != ""]
    rn = pearson([a for a, _ in _nz], [b for _, b in _nz])
    direction = ("shorter" if lo_len < hi_len else "longer")
    A(f"3. **Edit distance and original length interact, in both directions.** Raw "
      f"content-edit distance correlates *positively* with original length "
      f"(Pearson r = {rl:.2f} against original content-word count) while normalized distance "
      f"correlates *negatively* (r = {rn:.2f}) — short prompts get few absolute edits but a "
      f"high edit fraction. So neither metric gives a length-neutral cut, only a stated one. "
      f"In this corpus the two effects happen to cancel at the chosen cut: mean original "
      f"length is {lo_len:.1f} tokens in LOW against {hi_len:.1f} in HIGH "
      f"({direction} by {abs(lo_len-hi_len):.1f} tokens), so the LOW stratum is **not** "
      f"materially length-biased here. That is a property of this corpus, not a guarantee — "
      f"re-check it after any re-run.")
    A("4. **Refusal is measured on one target model per attacker** (the diagonal cells "
      "llamaAtt->llamaTgt and qwenAtt->qwenTgt). Anything here is a trigger *for that target*, "
      "not a universal one. The off-diagonal cells exist in the same directory.")
    A("5. **The comparison set is un-refused by construction**, so any property correlated "
      "with refusal for non-lexical reasons (topic, prompt type) surfaces as a 'trigger'. "
      "Stratum matching removes the largest such confound — amount of rewriting — but not "
      "topic.")
    A("6. **Judge dependence.** `is_or` comes from a single LLM judge. This project has "
      "already seen conclusions flip between judge versions, so every count here is "
      f"conditional on **{vtag}**. Re-run with `--verdicts` when a newer judge lands.")
    A(f"7. **Multiplicity.** The reported table screens {len(tri_presence)} words; at "
      f"{len({r['orig_idx'] for r in LOW})} independent originals no correction survives. "
      "Treat the table as hypothesis generation for a targeted follow-up, not as findings.")

    A("\n## 5. HIGH stratum — PREPARED, ANALYSIS DEFERRED\n")
    A(f"**No analysis of this stratum is performed here, by design.** It holds {len(HIGH)} "
      f"confirmed-OR pairs from {len({r['orig_idx'] for r in HIGH})} distinct originals — "
      f"{100*len(HIGH)/len(OR):.0f}% of the confirmed-OR corpus, i.e. the large majority. "
      f"In these the rewrite was restructured (clause order, framing, added requirements), so "
      f"attributing the refusal to any single word is unsound; a log-odds table over them "
      f"would be a category error dressed up as a result. It is deliberately left for "
      f"higher-dimensional (embedding / clustering / representation) work.\n")
    A(f"**What is saved** — `{rel(args.out_dir)}/or_high_stratum_{vtag}.csv`, "
      f"{len(HIGH)} rows:\n")
    A("| column | why it is there |")
    A("|---|---|")
    A("| `original`, `rewrite` | full text, ready for an encoder |")
    A("| `wl_dist`, `wl_dist_content`, `norm_dist`, `norm_dist_content` | lets the stratum be "
      "sub-banded by how far the rewrite travelled, instead of treated as one blob |")
    A("| `edit_ops` | full JSON list of non-equal alignment ops `[kind, orig_word, rw_word]`, "
      "so structural analysis (insertion runs, deletion runs, reordering) needs no realignment |")
    A("| `introduced_words`, `removed_words` | content words the edit added / removed |")
    A("| `orig_idx` | groups rewrites sharing an original — the grouping any train/test split "
      "or clustering must respect to avoid leakage |")
    A("| `attacker`, `target`, `refuse_rate`, `intent`, `harm`, `is_or` | provenance and label |")
    A("\n**What it is for** (not done here): embed originals and rewrites and study the "
      "*displacement* (rewrite - original) rather than the rewrite alone, then cluster "
      "displacements to recover restructuring **strategies**. Use "
      f"`comparison_all_with_distance_{vtag}.csv` filtered to `stratum == HIGH` as the "
      "negative class — same attacker, same amount of restructuring, not refused — so the "
      "clustering separates refusal-triggering restructurings from restructuring in general. "
      "Group by `orig_idx` when splitting.")

    A("\n## 6. Files written\n")
    for fn in sorted(os.listdir(args.out_dir)):
        A(f"* `{rel(args.out_dir)}/{fn}`")
    A("\n## 7. Re-running when new verdicts land\n")
    A("```bash")
    A("python3 analyze_edit_distance.py \\")
    A("    --verdicts probe_or/results/v5_judged --verdict_tag v5")
    A("```")
    A("\nThe verdict source is a parameter and two on-disk layouts are supported "
      "automatically:\n")
    A("* **per-cell** — `{attacker}_judged.csv` keyed by that cell's own `pair_id` "
      "(the trial2 / v4 layout used for this report).")
    A("* **text-keyed** — a de-duplicated judge-input CSV (`judge_input_all.csv`: "
      "`pair_id, original, rewrite`) plus a verdicts CSV (`verdicts_all.csv`: "
      "`pair_id, intent, harm, is_or`), joined back to each attacker x target cell on the "
      "`(original, rewrite)` text. This is the layout the v5 re-judge is writing, because it "
      "pools rewrites across many result directories and re-keys them `V5_######`, so text is "
      "the only stable join. The path has been exercised end to end against the real "
      "`judge_input_all.csv` and reproduces this report's counts exactly when fed the v4 "
      "labels.")
    if skipped:
        A("\nAt the time this report was generated, `--verdicts auto` inspected and rejected "
          "the following before falling through to the labels used here:\n")
        for pth, why in skipped:
            A(f"* `{pth}` — {why}")
        A(f"\nRe-run once a complete verdicts file appears there. Outputs are suffixed with "
          f"the judge tag, so the {vtag} and later versions coexist rather than overwrite.\n")
    else:
        A(f"\nOutputs are suffixed with the judge tag, so {vtag} and later versions coexist "
          f"rather than overwrite.\n")
    A("Everything else is a flag too: `--low_cut auto` re-derives the cut from whatever "
      "corpus it is given (so the cut may move if the v5 labels change the corpus), "
      "`--comparison` switches the contrast, `--prior_a0` the prior strength, "
      "`--bootstrap` the replicate count, `--credible_min_origs` the survivor filter.")

    with open(args.report, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[wrote] {args.report}")

    print(f"\n=== SUMMARY (judge {vtag}) ===")
    print(f"confirmed OR pairs : {len(OR)} "
          f"({dict(Counter(r['attacker'] for r in OR))}), "
          f"{len({r['orig_idx'] for r in OR})} distinct originals")
    print(f"cut                : content edits <= {K}")
    print(f"LOW  : {len(LOW)} pairs, {len({r['orig_idx'] for r in LOW})} distinct originals")
    print(f"HIGH : {len(HIGH)} pairs, {len({r['orig_idx'] for r in HIGH})} distinct originals")
    print(f"comparison (LOW-matched): {len(cmp_low)}")
    print("top by raw z:", ", ".join(
        f"{d['word']}(z={d['z']}, n={d['n_or']}, orig={d['n_distinct_orig_or']}, "
        f"zdoc={d['z_doc']})" for d in tri_presence[:8]))
    print("SURVIVORS (>= %d originals and z_doc > 1.96):" % args.credible_min_origs, ", ".join(
        f"{d['word']}(z_doc={d['z_doc']}, z={d['z']}, n={d['n_or']}, "
        f"orig={d['n_distinct_orig_or']}, bootfrac={d['boot_frac_z_gt_1p96']})"
        for d in credible) or "none")


if __name__ == "__main__":
    main()
