# GCG rewrite vocabulary — what do these attacks actually insert?

Unit of analysis: **the whole rewrite text**. Estimator: weighted log-odds with an informative Dirichlet prior (Monroe et al. 2008), ranked by evidence. `origs` = distinct original prompts a word appears in, so a word repeated inside a handful of prompts cannot look like a finding.

⛔ marks a prohibition word (reads as an instruction to decline). ⚠️ marks danger vocabulary of the kind the RWR attacker reached for.


## Corpora

| corpus | rewrites | distinct originals | non-alpha token rate |
|---|--:|--:|--:|
| Qwen GCG (all arms) | 2308 | 1181 | 22.8% |
| Qwen GCG or_* | 1920 | 960 | 25.0% |
| Qwen GCG jb_* | 388 | 221 | 17.6% |
| Llama GCG | 1220 | 1220 | 6.6% |
| RWR llamaAtt (confirmed-OR) | 2409 | 1496 | 1.7% |
| RWR baseQwenAtt (unjudged) | 21880 | 5999 | 2.3% |

## E. How often does each method insert a prohibition word?

The specific failure mode: a rewrite that tells the model not to answer is not eliciting over-refusal, it is issuing an instruction.

| corpus | n | inserts a ⛔ prohibition word | inserts a ⚠️ danger word | mean words added |
|---|--:|--:|--:|--:|
| Qwen GCG or_* | 1920 | **21.9%** | 0.4% | 2.8 |
| Qwen GCG jb_* | 388 | **21.4%** | 1.0% | 3.9 |
| Llama GCG | 1220 | **24.0%** | 0.8% | 6.1 |
| RWR llamaAtt (confirmed-OR) | 2409 | **6.1%** | 45.7% | 6.9 |
| RWR baseQwenAtt | 21880 | **7.9%** | 3.5% | 5.7 |

## A1. Qwen GCG (or_*) vs RWR llamaAtt (confirmed-OR)

The headline comparison. **Zero shared originals**, so differences conflate method with topic — read B and C before trusting this one.


**Over-represented in Qwen GCG or_***

| word | z | n(Qwen GCG or_*) | origs | n(RWR llamaAtt) |
|---|--:|--:|--:|--:|
| `what` | 13.40 | 262 | 137 | 70 |
| `no` ⛔ | 11.38 | 288 | 250 | 8 |
| `not` ⛔ | 9.23 | 117 | 102 | 15 |
| `sentence` | 8.84 | 148 | 93 | 63 |
| `write` | 8.50 | 149 | 94 | 70 |
| `describe` | 8.41 | 109 | 66 | 33 |
| `following` | 8.13 | 189 | 125 | 117 |
| `answer` | 8.06 | 109 | 100 | 6 |
| `between` | 7.21 | 80 | 44 | 24 |
| `make` | 7.02 | 69 | 43 | 15 |
| `image` | 6.69 | 61 | 54 | 10 |
| `link` | 6.26 | 55 | 50 | 6 |
| `best` | 6.24 | 59 | 36 | 17 |
| `name` | 6.17 | 88 | 47 | 46 |
| `using` | 5.46 | 68 | 48 | 35 |
| `create` | 5.07 | 98 | 51 | 1 |
| `find` | 5.00 | 46 | 28 | 2 |
| `who` | 4.87 | 47 | 25 | 21 |
| `movie` | 4.86 | 33 | 17 | 7 |
| `help` | 4.71 | 34 | 28 | 10 |

**Over-represented in RWR llamaAtt**

| word | z | n(RWR llamaAtt) | origs | n(Qwen GCG or_*) |
|---|--:|--:|--:|--:|
| `construct` | 8.14 | 213 | 171 | 20 |
| `extract` | 7.04 | 157 | 122 | 12 |
| `produce` | 6.94 | 184 | 148 | 5 |
| `manipulate` | 6.93 | 402 | 291 | 2 |
| `without` ⛔ | 6.41 | 135 | 118 | 14 |
| `down` | 6.40 | 130 | 114 | 10 |
| `customer` | 5.16 | 126 | 61 | 26 |
| `detail` | 5.13 | 122 | 103 | 2 |
| `control` | 5.07 | 93 | 76 | 3 |
| `system` | 4.93 | 111 | 74 | 22 |
| `through` | 4.75 | 73 | 65 | 4 |
| `scheme` | 4.65 | 74 | 59 | 3 |
| `draft` | 4.47 | 68 | 62 | 8 |
| `compile` | 4.42 | 65 | 53 | 3 |
| `break` | 4.30 | 116 | 105 | 1 |
| `systems` | 4.13 | 55 | 39 | 5 |
| `exactly` | 4.05 | 53 | 49 | 3 |
| `tactics` | 4.02 | 89 | 68 | 1 |
| `sequence` | 3.88 | 48 | 34 | 4 |
| `personal` | 3.86 | 49 | 37 | 5 |

## A2. Qwen GCG (or_*) vs RWR baseQwenAtt (model-matched, unjudged)

Same target model as the GCG run, removing the model confound. baseQwenAtt is raw rewrites, not filtered to confirmed over-refusal. Still zero shared originals.


**Over-represented in Qwen GCG or_***

| word | z | n(Qwen GCG or_*) | origs | n(RWR baseQwenAtt) |
|---|--:|--:|--:|--:|
| `please` | 36.30 | 325 | 285 | 214 |
| `no` ⛔ | 33.70 | 288 | 250 | 254 |
| `answer` | 18.86 | 109 | 100 | 172 |
| `image` | 15.13 | 61 | 54 | 21 |
| `link` | 14.37 | 55 | 50 | 19 |
| `sorry` ⛔ | 11.59 | 34 | 33 | 16 |
| `not` ⛔ | 11.40 | 117 | 102 | 585 |
| `give` | 11.28 | 68 | 43 | 226 |
| `https` | 11.15 | 31 | 30 | 19 |
| `say` | 10.18 | 31 | 31 | 46 |
| `wait` | 10.07 | 26 | 24 | 11 |
| `tell` | 9.95 | 42 | 24 | 109 |
| `best` | 9.35 | 59 | 36 | 240 |
| `nothing` ⛔ | 8.40 | 19 | 17 | 21 |
| `which` | 8.22 | 90 | 53 | 560 |
| `what` | 7.71 | 262 | 137 | 2523 |
| `reply` | 7.61 | 15 | 15 | 14 |
| `pdf` | 7.46 | 14 | 14 | 7 |
| `photo` | 7.24 | 14 | 12 | 15 |
| `etc` | 7.16 | 13 | 12 | 6 |

**Over-represented in RWR baseQwenAtt**

| word | z | n(RWR baseQwenAtt) | origs | n(Qwen GCG or_*) |
|---|--:|--:|--:|--:|
| `way` | 7.84 | 1802 | 1069 | 17 |
| `used` | 7.11 | 1647 | 968 | 25 |
| `including` | 6.70 | 1389 | 794 | 7 |
| `without` ⛔ | 5.50 | 968 | 615 | 14 |
| `avoid` | 5.46 | 875 | 624 | 7 |
| `using` | 4.97 | 1974 | 1045 | 68 |
| `create` | 4.93 | 2559 | 1080 | 98 |
| `considered` | 4.91 | 804 | 607 | 3 |
| `dangerous` | 4.77 | 670 | 486 | 5 |
| `includes` | 4.75 | 733 | 444 | 3 |
| `inappropriate` | 4.70 | 712 | 525 | 3 |
| `like` | 4.69 | 930 | 622 | 21 |
| `someone` | 4.58 | 660 | 405 | 9 |
| `detailed` | 4.48 | 595 | 382 | 4 |
| `potential` | 4.30 | 553 | 358 | 6 |
| `manipulate` | 4.30 | 656 | 393 | 2 |
| `unethical` | 4.14 | 588 | 422 | 2 |
| `even` | 4.05 | 553 | 355 | 2 |
| `something` | 3.97 | 522 | 373 | 2 |
| `sound` | 3.93 | 508 | 343 | 2 |

## B. Llama GCG vs Qwen GCG — 372 SHARED originals only

Same prompts, same method, two target models. The topic confound is removed by construction, so anything here is a real method/model difference.


**Over-represented in Llama GCG (shared)**

| word | z | n(Llama GCG (shared)) | origs | n(Qwen GCG (shared)) |
|---|--:|--:|--:|--:|
| `certain` | 2.96 | 14 | 13 | 1 |
| `while` | 2.80 | 12 | 12 | 1 |
| `because` | 2.70 | 11 | 11 | 1 |
| `never` ⛔ | 2.55 | 21 | 16 | 13 |
| `anyone` | 2.48 | 9 | 8 | 1 |
| `get` | 2.41 | 13 | 12 | 6 |
| `not` ⛔ | 2.36 | 44 | 40 | 40 |
| `note` | 2.28 | 9 | 9 | 3 |
| `information` | 2.22 | 12 | 12 | 6 |
| `read` | 2.08 | 7 | 7 | 2 |
| `also` | 2.07 | 8 | 8 | 3 |
| `including` | 2.05 | 6 | 6 | 1 |
| `like` | 2.05 | 9 | 9 | 4 |
| `first` | 2.05 | 9 | 9 | 4 |
| `even` | 1.85 | 5 | 5 | 1 |
| `secret` | 1.85 | 5 | 5 | 1 |
| `know` | 1.85 | 5 | 5 | 1 |
| `which` | 1.81 | 22 | 22 | 19 |
| `without` ⛔ | 1.80 | 12 | 12 | 8 |
| `cannot` ⛔ | 1.80 | 10 | 10 | 6 |

**Over-represented in Qwen GCG (shared)**

| word | z | n(Qwen GCG (shared)) | origs | n(Llama GCG (shared)) |
|---|--:|--:|--:|--:|
| `please` | 6.69 | 145 | 126 | 19 |
| `no` ⛔ | 4.78 | 104 | 91 | 23 |
| `wait` | 2.15 | 14 | 12 | 1 |
| `link` | 2.11 | 17 | 15 | 3 |
| `following` | 2.01 | 74 | 51 | 32 |
| `between` | 1.96 | 41 | 23 | 15 |
| `come` | 1.81 | 21 | 11 | 6 |
| `https` | 1.73 | 16 | 16 | 4 |
| `response` | 1.69 | 9 | 9 | 1 |
| `anything` | 1.68 | 11 | 10 | 2 |
| `customer` | 1.64 | 13 | 5 | 3 |
| `generate` | 1.47 | 77 | 42 | 38 |
| `difference` | 1.46 | 16 | 11 | 5 |
| `product` | 1.45 | 18 | 7 | 6 |
| `best` | 1.44 | 20 | 13 | 7 |
| `photo` | 1.43 | 7 | 6 | 1 |
| `construct` | 1.39 | 9 | 6 | 2 |
| `whether` | 1.35 | 11 | 7 | 3 |
| `question` | 1.33 | 13 | 9 | 4 |
| `line` | 1.33 | 13 | 6 | 4 |

## C. or_* vs jb_* within Qwen GCG — opposite objectives, same everything else

`or_*` minimises the NLL of refusal openers, `jb_*` maximises it. Same search, same model, same code. If the inserted vocabulary is about danger, these should differ topically; if it is about steering the opener, they should be mirror images (prohibition vs permission). Assumes no other corpus.


**Over-represented in Qwen GCG or_***

| word | z | n(Qwen GCG or_*) | origs | n(Qwen GCG jb_*) |
|---|--:|--:|--:|--:|
| `please` | 7.40 | 325 | 285 | 38 |
| `following` | 5.21 | 189 | 125 | 2 |
| `sentence` | 4.84 | 148 | 93 | 2 |
| `answer` | 4.81 | 109 | 100 | 5 |
| `list` | 4.58 | 120 | 68 | 13 |
| `no` ⛔ | 4.49 | 288 | 250 | 64 |
| `name` | 4.31 | 88 | 47 | 5 |
| `describe` | 3.83 | 109 | 66 | 1 |
| `explain` | 3.67 | 83 | 53 | 10 |
| `three` | 3.60 | 61 | 43 | 3 |
| `create` | 3.57 | 98 | 51 | 15 |
| `new` | 3.53 | 76 | 52 | 9 |
| `image` | 3.47 | 61 | 54 | 5 |
| `which` | 3.39 | 90 | 53 | 14 |
| `identify` | 3.12 | 56 | 30 | 1 |
| `write` | 3.11 | 149 | 94 | 34 |
| `link` | 3.11 | 55 | 50 | 1 |
| `product` | 3.05 | 44 | 20 | 2 |
| `words` | 3.03 | 51 | 31 | 1 |
| `text` | 2.80 | 41 | 31 | 1 |

**Over-represented in Qwen GCG jb_***

| word | z | n(Qwen GCG jb_*) | origs | n(Qwen GCG or_*) |
|---|--:|--:|--:|--:|
| `use` | 8.36 | 78 | 46 | 35 |
| `want` | 7.87 | 65 | 39 | 7 |
| `without` ⛔ | 7.49 | 54 | 33 | 14 |
| `get` | 6.74 | 45 | 26 | 14 |
| `people` | 6.47 | 58 | 31 | 36 |
| `detailed` | 6.31 | 44 | 30 | 4 |
| `instructions` | 6.31 | 38 | 26 | 8 |
| `make` | 5.88 | 75 | 47 | 69 |
| `i'm` | 5.73 | 38 | 19 | 3 |
| `give` | 5.33 | 69 | 44 | 68 |
| `specific` | 5.26 | 29 | 19 | 11 |
| `ensure` | 5.22 | 29 | 16 | 3 |
| `fake` | 5.11 | 25 | 17 | 5 |
| `avoid` | 5.07 | 25 | 16 | 7 |
| `don't` | 5.05 | 26 | 14 | 9 |
| `out` | 4.78 | 32 | 19 | 20 |
| `including` | 4.56 | 21 | 14 | 7 |
| `post` | 4.41 | 22 | 13 | 10 |
| `know` | 4.34 | 18 | 10 | 4 |
| `methods` | 4.31 | 20 | 13 | 2 |

## D. Judge-confirmed vs judge-rejected, within judged or_loose

Both sides are the same arm, same search, same originals pool. If the rewrites a Sonnet judge accepted insert the same words as the ones it rejected, the judged label is not tracking the inserted vocabulary.


**Over-represented in judge-confirmed**

| word | z | n(judge-confirmed) | origs | n(judge-rejected) |
|---|--:|--:|--:|--:|
| `no` ⛔ | 8.88 | 183 | 156 | 101 |
| `please` | 2.70 | 146 | 124 | 175 |
| `code` | 2.55 | 10 | 8 | 3 |
| `allowed` | 2.53 | 9 | 9 | 2 |
| `write` | 2.53 | 44 | 41 | 41 |
| `not` ⛔ | 2.42 | 51 | 49 | 51 |
| `examples` | 2.35 | 8 | 8 | 2 |
| `ignore` ⛔ | 2.34 | 14 | 14 | 8 |
| `after` | 2.15 | 7 | 6 | 2 |
| `suggest` | 2.15 | 7 | 7 | 2 |
| `learning` | 2.15 | 7 | 7 | 2 |
| `everything` | 2.13 | 10 | 10 | 5 |
| `answer` | 2.01 | 49 | 48 | 53 |
| `computer` | 1.91 | 7 | 7 | 3 |
| `cannot` ⛔ | 1.91 | 7 | 7 | 3 |
| `forget` ⛔ | 1.91 | 7 | 7 | 3 |
| `come` | 1.91 | 10 | 10 | 6 |
| `etc` | 1.91 | 8 | 7 | 4 |
| `plan` | 1.90 | 5 | 5 | 1 |
| `including` | 1.90 | 5 | 5 | 1 |

**Over-represented in judge-rejected**

| word | z | n(judge-rejected) | origs | n(judge-confirmed) |
|---|--:|--:|--:|--:|
| `image` | 3.66 | 49 | 45 | 7 |
| `link` | 3.49 | 46 | 42 | 7 |
| `following` | 3.08 | 63 | 62 | 17 |
| `https` | 2.97 | 28 | 27 | 2 |
| `classify` | 2.62 | 23 | 23 | 1 |
| `sentence` | 2.52 | 59 | 54 | 19 |
| `text` | 2.30 | 20 | 19 | 3 |
| `categorize` | 2.10 | 14 | 14 | 1 |
| `draw` | 1.90 | 23 | 22 | 6 |
| `below` | 1.87 | 18 | 18 | 4 |
| `see` | 1.75 | 10 | 10 | 1 |
| `pdf` | 1.73 | 12 | 12 | 2 |
| `photo` | 1.64 | 9 | 9 | 1 |
| `either` | 1.64 | 9 | 9 | 1 |
| `like` | 1.60 | 11 | 11 | 2 |
| `http` | 1.60 | 11 | 11 | 2 |
| `number` | 1.60 | 11 | 10 | 2 |
| `wait` | 1.57 | 20 | 18 | 6 |
| `images` | 1.51 | 8 | 8 | 1 |
| `thanks` | 1.51 | 8 | 8 | 1 |
