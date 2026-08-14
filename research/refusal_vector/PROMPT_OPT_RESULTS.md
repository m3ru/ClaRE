# GCG / PEZ prompt optimization against the refusal signals — results

Runs: PACE arrays `5577119` (unrestricted) and `5577125` (`--block_profanity`), 2026-08-14.
Llama-3-8B-Instruct, universal 20-token suffix, 8 train / 60 held-out benign Alpaca prompts,
250 steps. Refusal = start-of-response substring detector. Raw result JSONs live in
`~/scratch/prompt_opt/` on PACE and are deliberately **not** committed: the optimized
suffixes are explicit requests for harmful content (see below).

## Headline numbers (held-out benign prompts)

| config | random ctrl | OPTIMIZED | proj | logP("I cannot") |
|---|--:|--:|--:|--:|
| direction GCG @L12 | 11.1% | 100.0% | −0.35 → 3.22 | −22.1 → −0.02 |
| direction GCG @L17 | 11.1% | 100.0% | −1.73 → 5.79 | −22.1 → −0.12 |
| icannot GCG | 11.1% | 100.0% | −1.73 → 1.52 | −22.1 → −0.01 |
| icannot PEZ | 11.1% | 100.0% | −1.73 → 1.94 | −22.1 → −0.05 |
| direction PEZ @L17 | 11.1% | 100.0% | −1.73 → 2.41 | −22.1 → −0.05 |
| **direction PEZ @L12** | 11.1% | **23.3%** | −0.35 → 0.60 | −22.1 → −4.90 |
| direction PEZ @L12 *clean* | 3.9% | **5.0%** | −0.35 → 0.29 | −22.1 → −7.52 |

## The result is a NEGATIVE one: this measures correct refusal, not over-refusal

Every arm that reached 100% did so by **appending an explicit request for harmful content**
to the benign prompt. The recovered suffixes contain, in plain text, solicitations for child
sexual abuse material, rape instructions, explosive-manufacture tutorials, terrorist-group
references, and targeted violence. The model refuses those prompts because **the prompts
became genuinely harmful** — that is the safety system working correctly, not over-refusal.

Both objectives are trivially satisfied by real harm:
- `logP("I cannot")` is maximized by making the request actually refusable.
- the harmful-vs-harmless **direction** is maximized the same way — unsurprising in
  hindsight, since the direction was *built* to encode harmfulness. Maximizing it is
  approximately "write something harmful."

So the experiment as designed cannot distinguish over-refusal from correct refusal. The
objectives have no term that keeps the prompt benign, and the optimizer exploited exactly
that. This is the same Goodhart shape as the RWR attacker, one level up.

## `--block_profanity` was too weak

The blocklist covered sexual/slur/violence stems but is substring-based over subword
tokens, so it leaked badly: the "clean" runs still recovered "underage", "explosive",
"ISIS", "kidnapping", "raping". Blocking a wordlist cannot work — harmfulness is semantic,
not lexical. Refusal rates barely moved (100% → 100% for most arms).

## The one informative contrast: L12 resists input-space attack

`direction PEZ @L12` is the only arm that stayed low (23.3% unrestricted, 5.0% clean —
*at or below* its random-suffix control of 3.9%), and its projection barely moved
(−0.35 → 0.29 vs L17's −1.73 → 2.41). Its suffixes are the only ones that are merely
gibberish rather than harmful.

Reading: **L12 — the abliteration-selected layer, the one with real causal purchase under
ablation — is the hardest to drive from the input.** Ablation and prompt-optimization are
different levers; a direction being causally load-bearing when you edit activations does
not make it reachable by editing tokens. That dissociation is the genuinely publishable
observation here.

Caveat: GCG's round-trip filter fell back frequently (73–250 of 250 steps; @L17 it never
passed once), so the GCG optimizer partly worked on token sequences that do not survive
decode→re-encode. All reported metrics are computed on the **text path** (decode the
suffix, re-tokenize), so the numbers are honest for the actual prompt — but the GCG search
was optimizing a slightly different object than what was scored.

## What the experiment needs to actually answer the question

1. **A harmfulness gate on the final prompt.** Score the optimized prompt with Llama-Guard
   (or the delta-probe) and *reject* any suffix that makes the prompt genuinely unsafe.
   Without this, the objective is degenerate.
2. **A semantic-similarity floor.** MiniLM cosine to the original benign prompt (the
   sim ≥ 0.85 gate already used in the OR pipeline) — currently not computed here at all.
3. **Optimize the residual, not the raw signal.** The target should be
   "refusal probability *in excess of* what the prompt's harmfulness warrants", i.e.
   maximize `logP("I cannot") − f(harmfulness)`, so the optimizer cannot win by adding harm.
4. Constrain fluency (perplexity penalty) so the suffix stays natural language.
