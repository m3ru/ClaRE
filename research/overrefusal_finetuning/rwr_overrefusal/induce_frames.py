#!/usr/bin/env python3
"""Vocabulary and prompt pool for the INDUCTION experiment.

Everything upstream in this project is subtractive: ablate a direction, over-refusal falls.
This is the forward direction -- can over-refusal be MANUFACTURED on demand by putting the
discovered trigger vocabulary into prompts the model otherwise answers?

This module holds the two things both halves of the pipeline need: the per-model frame
vocabulary, and a pool of fresh benign originals. The insertion itself is done by an LLM
rewriter (sonnet_insert_frames.py) rather than a regex, because a programmatic rule that drops
a word into a fixed syntactic slot produces stilted text, and the model might then be refusing
the oddity rather than the semantics.

VOCABULARY IS PER-MODEL AND NEVER CROSSED. Llama's frames come from analyze_frames.FRAMES
(mined from Llama's own low-edit triggers); Qwen's from probe_or/results/delta_qwen/
qwen_frames.json (mined from Qwen's, and it adds `fabrication`). Only adjectival/participial
surface forms are used, and each must still MATCH its own model's frame regex so the project's
own frame detector would label the rewritten prompt with that frame. That is asserted at
import time by _check_vocab(), not left as a comment.

exploitation uses `exploited`/`exploitable` rather than `exploitative`: the latter matches
Qwen's regex but NOT Llama's, and using the intersection keeps the arm lexically identical
across the two models.

CONTROLS. ctrl_rare is the load-bearing one: rare, neutral, participial modifiers matched to
the frame words on corpus frequency (measured, see --freq) that produce EQUALLY incongruous
noun phrases -- "a refrigerated method to increase customer engagement" is exactly as odd as
"a weaponized method". Without it the finding is only "an LLM rewrite raises refusal".
ctrl_common is congruous and frequent, and bounds "any inserted adjective at all".

PROMPT POOL. alpaca-cleaned, seed-43 shuffle of generate_or_alpaca.remaining_prompts, offset
46000 -- the only region of the 49,020-prompt pool no prior job consumed (Sonnet pool
[0,6000), scale-up [6000,8000), batch2 [8000,14000), low-edit [14000,46000)).

Run standalone: python induce_frames.py --model llama --n 200 --freq
"""
import argparse, json, os, random, re, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FRAME_WORDS_LLAMA = {
    "weaponization": ["weaponized"],
    "concealment":   ["covert", "untraceable", "undetectable", "stealthy", "clandestine"],
    "exploitation":  ["exploited", "exploitable"],
    "intrusion":     ["infiltrated", "breached", "hacked"],
    "exfiltration":  ["exfiltrated"],
}
FRAME_WORDS_QWEN = {
    "weaponization": ["weaponized"],
    "concealment":   ["covert", "untraceable", "undetectable", "stealthy", "clandestine"],
    "exploitation":  ["exploited", "exploitable"],
    "fabrication":   ["fabricated"],
    "exfiltration":  ["exfiltrated"],
}
CTRL_RARE = ["laminated", "refrigerated", "alphabetized", "upholstered", "nocturnal"]
CTRL_COMMON = ["detailed", "updated", "printed", "standard", "seasonal"]

# Union of both models' frames -- the rewriter generates once for this set and each model
# scores only its OWN frames, which halves the API spend without crossing vocabularies.
ALL_FRAMES = dict(FRAME_WORDS_LLAMA)
ALL_FRAMES.update(FRAME_WORDS_QWEN)

# An original containing any of these already is confounded whichever arm it lands in.
_BANNED = set(sum(ALL_FRAMES.values(), []) + CTRL_RARE + CTRL_COMMON
              + ["weapon", "weapons", "exploit", "exploits", "exfiltrate", "infiltrate",
                 "hack", "breach", "covertly", "stealth", "malware", "payload"])


def frame_words(model):
    return FRAME_WORDS_LLAMA if model == "llama" else FRAME_WORDS_QWEN


def frame_regexes(model):
    """The model's OWN mined frame regexes -- never the other model's."""
    if model == "llama":
        from analyze_frames import FRAMES
        return dict(FRAMES)
    here = os.path.dirname(os.path.abspath(__file__))
    return json.load(open(os.path.join(here, "probe_or/results/delta_qwen/qwen_frames.json")))


def _check_vocab(model):
    """Every surface form must be labelled with its own frame by that model's own detector."""
    rx = frame_regexes(model)
    for f, ws in frame_words(model).items():
        assert f in rx, f"{model}: frame {f} has no regex"
        c = re.compile(rx[f])
        for w in ws:
            assert c.match(w), f"{model}: '{w}' does not match frame regex for {f}: {rx[f]}"
    for f, p in rx.items():
        c = re.compile(p)
        for w in CTRL_RARE + CTRL_COMMON:
            assert not c.match(w), f"{model}: control word '{w}' matches frame {f}"


def matches_frame(model, frame, words):
    """Does any of `words` trip `frame`'s regex under this model's detector?"""
    c = re.compile(frame_regexes(model)[frame])
    return any(c.match(w.lower()) for w in (words or "").split())


def select_originals(n, skip_first=46000, shuffle_seed=43, max_words=40, seed=0, exclude=()):
    """Fresh benign Alpaca originals from the unconsumed tail of the pool.

    Filters: instruction-only (no `input` block, so the whole prompt is the ask and the
    rewriter has one clean sentence to work with); 4..max_words words; no frame or control
    vocabulary already present. There is no syntactic-slot filter -- the rewriter places the
    word itself, so the pool does not need to be shaped around an insertion rule.
    """
    from datasets import load_dataset
    from generate_or_alpaca import remaining_prompts
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    pool = remaining_prompts(ds)
    random.Random(shuffle_seed).shuffle(pool)
    tail = pool[skip_first:]
    excl = set(exclude)
    kept, stats = [], Counter()
    for idx, p in tail:
        stats["tail"] += 1
        if "\n\n" in p:
            stats["has_input"] += 1; continue
        w = p.split()
        if len(w) > max_words or len(w) < 4:
            stats["length"] += 1; continue
        if any(t.lower().strip(".,;:!?\"'()") in _BANNED for t in w):
            stats["already_loaded"] += 1; continue
        if p.strip() in excl:
            stats["fit_overlap"] += 1; continue
        kept.append((idx, p.strip()))
        stats["kept"] += 1
    random.Random(seed).shuffle(kept)
    return kept[:n], dict(stats), len(kept)


def assign_words(originals, words):
    """Round-robin so an arm's rate is a frame effect, not one word's. Reproducible."""
    return [words[k % len(words)] for k in range(len(originals))]


def alpaca_counts(words):
    """Corpus frequency in alpaca-cleaned -- the evidence for the frame/control match."""
    from datasets import load_dataset
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    tok = re.compile(r"[a-z]+")
    c, want = Counter(), set(words)
    for ex in ds:
        for f in ("instruction", "input", "output"):
            for t in tok.findall((ex.get(f) or "").lower()):
                if t in want:
                    c[t] += 1
    return {w: c.get(w, 0) for w in words}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["llama", "qwen"], default="llama")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--skip_first", type=int, default=46000)
    ap.add_argument("--freq", action="store_true")
    a = ap.parse_args()
    _check_vocab("llama"); _check_vocab("qwen")
    print("[vocab] both models' frame regexes OK; no control word trips any frame\n")
    origs, stats, navail = select_originals(a.n, skip_first=a.skip_first)
    print(f"[pool] offset {a.skip_first}+ : {stats}")
    print(f"[pool] {navail} usable originals available, taking {len(origs)}")
    for i, o in origs[:5]:
        print(f"   [{i}] {o}")
    if a.freq:
        allw = sorted(set(sum(ALL_FRAMES.values(), []) + CTRL_RARE + CTRL_COMMON))
        cnt = alpaca_counts(allw)
        inv = {w: f for f, ws in ALL_FRAMES.items() for w in ws}
        print("\nword              arm            alpaca-cleaned count")
        for w in sorted(allw, key=lambda z: -cnt[z]):
            arm = inv.get(w, "ctrl_rare" if w in CTRL_RARE else "ctrl_common")
            print(f"  {w:16s} {arm:14s} {cnt[w]:6d}")


if __name__ == "__main__":
    main()
