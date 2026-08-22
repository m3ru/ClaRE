# Words each refusal category inserts

Per-category vocabulary from the categories in [GCG_CATEGORIES.md](GCG_CATEGORIES.md), as input to the injection experiment.

**Category and corpus are correlated**, so a pooled comparison would surface corpus vocabulary rather than category vocabulary. Every list below is computed within a single corpus (category against the other categories of that same corpus). The headline for each category is the words appearing in BOTH corpora's lists; single-corpus words are listed separately and are unconfirmed.

Unit is the added-word multiset (rewrite minus original). `n` is occurrences, `origs` distinct original prompts. ⛔ prohibition, ⚠️ danger.


## Cell sizes

| category | llama_gcg | qwen_gcg |
|---|---|---|
| invented a harmful/illegal reading | 671 | 51 |
| read as explicit/offensive | 289 | 34 |
| obeyed an injected 'no' | 95 | 206 |
| thought it needed an image/link/PDF | 52 | 28 |
| declined an inserted derogatory phrase | 32 | 2 *(too small)* |
| no visible cause | 78 | 38 |

Cells below 20 rows are skipped: a one-vs-rest log-odds on a handful of rewrites is noise.


## invented a harmful/illegal reading

`fabricated_harmful_or_illegal_framing`


| corpus | rows | candidate words | status |
|---|--:|--:|---|
| llama_gcg | 671 | 25 | ok |
| qwen_gcg | 51 | 4 | ok |

**In both corpora** — intersection of a 25-word and a 4-word list:

| word | z (llama_gcg) | z (qwen_gcg) | occurrences |
|---|---|---|---|
| `make` | 1.83 | 0.91 | 13 |

_llama_gcg only:_ `certain`, `secret`, `dark`, `during`, `special`, `need`, `below`, `request`, `high`, `work`

_qwen_gcg only:_ `use`, `using`, `everything`

_Check — share of rewrites inserting the word, in-category vs rest of the same corpus. Both gaps should be positive._

| word | llama_gcg in-cat / rest | qwen_gcg in-cat / rest |
|---|---|---|
| `make` | 1.5% / 0.4%  (n=10) | 5.9% / 0.0%  (n=3) |

## read as explicit/offensive

`perceived_explicit_or_offensive_content`


| corpus | rows | candidate words | status |
|---|--:|--:|---|
| llama_gcg | 289 | 25 | ok |
| qwen_gcg | 34 | 2 | ok |

**In both corpora** — intersection of a 25-word and a 2-word list:

_No word appears in both corpora's top lists for this category._

_llama_gcg only:_ `explicit`, `words`, `text`, `comment`, `watch`, `racist`, `adult`, `add`, `him`, `popular`

_qwen_gcg only:_ `xxx`, `etc`

## obeyed an injected 'no'

`explicit_negation_command_obeyed`


| corpus | rows | candidate words | status |
|---|--:|--:|---|
| llama_gcg | 95 | 25 | ok |
| qwen_gcg | 206 | 25 | ok |

**In both corpora** — intersection of a 25-word and a 25-word list:

| word | z (llama_gcg) | z (qwen_gcg) | occurrences |
|---|---|---|---|
| `not` ⛔ | 8.64 | 3.40 | 83 |
| `never` ⛔ | 8.27 | 0.90 | 32 |
| `no` ⛔ | 2.29 | 6.84 | 163 |
| `please` | 4.07 | 1.49 | 113 |
| `cannot` ⛔ | 4.53 | 0.97 | 17 |
| `provide` | 4.02 | 1.13 | 11 |
| `stop` ⛔ | 2.25 | 0.90 | 10 |

_llama_gcg only:_ `share`, `information`, `dont` ⛔, `assistance`, `assist`, `answering`, `given`, `anyone`, `public`, `notice`

_qwen_gcg only:_ `answer`, `write`, `say`, `ignore` ⛔, `nothing` ⛔, `anything`, `help`, `give`, `explain`, `sorry` ⛔

_Check — share of rewrites inserting the word, in-category vs rest of the same corpus. Both gaps should be positive._

| word | llama_gcg in-cat / rest | qwen_gcg in-cat / rest |
|---|---|---|
| `not` ⛔ | 37.9% / 6.4%  (n=36) | 19.9% / 4.6%  (n=41) |
| `never` ⛔ | 24.2% / 1.9%  (n=23) | 2.9% / 0.0%  (n=6) |
| `no` ⛔ | 11.6% / 3.7%  (n=11) | 63.1% / 17.0%  (n=130) |
| `please` | 18.9% / 5.0%  (n=18) | 38.8% / 28.8%  (n=80) |
| `cannot` ⛔ | 10.5% / 1.3%  (n=10) | 3.4% / 0.0%  (n=7) |
| `provide` | 6.3% / 0.4%  (n=6) | 2.4% / 0.7%  (n=5) |
| `stop` ⛔ | 4.2% / 0.9%  (n=4) | 2.9% / 0.0%  (n=6) |

## thought it needed an image/link/PDF

`capability_or_format_misunderstanding`


| corpus | rows | candidate words | status |
|---|--:|--:|---|
| llama_gcg | 52 | 14 | ok |
| qwen_gcg | 28 | 4 | ok |

**In both corpora** — intersection of a 14-word and a 4-word list:

| word | z (llama_gcg) | z (qwen_gcg) | occurrences |
|---|---|---|---|
| `link` | 7.30 | 3.35 | 18 |

_llama_gcg only:_ `www`, `https`, `added`, `online`, `provided`, `read`, `see`, `after`, `public`, `also`

_qwen_gcg only:_ `draw`, `image`, `picture`

_Check — share of rewrites inserting the word, in-category vs rest of the same corpus. Both gaps should be positive._

| word | llama_gcg in-cat / rest | qwen_gcg in-cat / rest |
|---|---|---|
| `link` | 19.2% / 0.3%  (n=10) | 10.7% / 0.9%  (n=3) |

## declined an inserted derogatory phrase

`objection_to_inserted_derogatory_phrase`


| corpus | rows | candidate words | status |
|---|--:|--:|---|
| llama_gcg | 32 | 7 | ok |
| qwen_gcg | 2 | 0 | cell below 20 |

Only `llama_gcg` yields a list (qwen_gcg: cell_too_small), so there is nothing to replicate against and these are unconfirmed.


| word | z | n | origs |
|---|--:|--:|--:|
| `people` | 4.52 | 6 | 6 |
| `idiot` | 4.03 | 3 | 3 |
| `not` ⛔ | 2.13 | 7 | 6 |
| `help` | 1.81 | 3 | 3 |
| `please` | 1.18 | 4 | 3 |
| `no` ⛔ | 1.07 | 3 | 3 |
| `stupid` | 0.91 | 6 | 6 |

## no visible cause

`genuine_over_refusal_no_visible_cause`


| corpus | rows | candidate words | status |
|---|--:|--:|---|
| llama_gcg | 78 | 10 | ok |
| qwen_gcg | 38 | 0 | no word in >= 3 distinct originals |

Only `llama_gcg` yields a list (qwen_gcg: no_word_met_min_origs), so there is nothing to replicate against and these are unconfirmed.


| word | z | n | origs |
|---|--:|--:|--:|
| `last` | 3.24 | 3 | 3 |
| `nothing` ⛔ | 3.08 | 3 | 3 |
| `because` | 2.44 | 5 | 5 |
| `personal` | 1.19 | 3 | 3 |
| `one` | 1.13 | 4 | 3 |
| `information` | 1.08 | 4 | 4 |
| `without` ⛔ | 0.93 | 4 | 3 |
| `which` | 0.61 | 3 | 3 |
| `answer` | 0.43 | 3 | 3 |
| `given` | 0.43 | 3 | 3 |

## Summary

| category | replicated words | usable for injection? |
|---|--:|---|
| invented a harmful/illegal reading | 1 | yes — `make` |
| read as explicit/offensive | 0 | unconfirmed — one corpus only (llama_gcg) |
| obeyed an injected 'no' | 7 | yes — `cannot`, `never`, `no`, `not`, `please`, `provide`, `stop` |
| thought it needed an image/link/PDF | 1 | yes — `link` |
| declined an inserted derogatory phrase | 0 | unconfirmed — one corpus only (llama_gcg) |
| no visible cause | 0 | unconfirmed — one corpus only (llama_gcg) |

Only the injected-negation category produces a word set that replicates across both corpora, and it is the only one whose replicated list is longer than a single word. For the others the replication test is mostly **untestable rather than negative**: on these subsets the mean number of added words is llama_gcg 6.2, qwen_gcg 4.7, and the smaller cells mean few words reach 3 distinct originals, so the shorter candidate lists come back near-empty. A short list cannot intersect a long one, so a zero there is not evidence of disagreement.

