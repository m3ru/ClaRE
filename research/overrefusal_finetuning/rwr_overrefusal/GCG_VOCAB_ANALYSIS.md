# GCG rewrite vocabulary — what do these attacks actually insert?

Unit of analysis: **words the rewrite ADDED to the original**. Estimator: weighted log-odds with an informative Dirichlet prior (Monroe et al. 2008), ranked by evidence. `origs` = distinct original prompts a word appears in, so a word repeated inside a handful of prompts cannot look like a finding.

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
| `no` ⛔ | 9.96 | 284 | 248 | 3 |
| `not` ⛔ | 8.90 | 96 | 91 | 5 |
| `image` | 8.56 | 61 | 54 | 10 |
| `write` | 6.44 | 36 | 35 | 15 |
| `answer` | 5.74 | 97 | 93 | 1 |
| `help` | 5.06 | 21 | 21 | 4 |
| `new` | 4.90 | 26 | 24 | 17 |
| `below` | 4.84 | 25 | 24 | 16 |
| `given` | 4.75 | 19 | 19 | 7 |
| `text` | 4.72 | 23 | 21 | 14 |
| `anything` | 4.60 | 20 | 19 | 2 |
| `draw` | 4.41 | 29 | 28 | 1 |
| `reply` | 4.18 | 15 | 15 | 6 |
| `make` | 4.09 | 14 | 14 | 5 |
| `stop` ⛔ | 3.92 | 13 | 13 | 2 |
| `following` | 3.72 | 24 | 24 | 25 |
| `different` | 3.71 | 12 | 12 | 5 |
| `sentence` | 3.62 | 14 | 14 | 1 |
| `video` | 3.62 | 14 | 13 | 1 |
| `first` | 3.62 | 11 | 10 | 4 |

**Over-represented in RWR llamaAtt**

| word | z | n(RWR llamaAtt) | origs | n(Qwen GCG or_*) |
|---|--:|--:|--:|--:|
| `manipulate` | 6.10 | 402 | 291 | 2 |
| `generate` | 5.11 | 155 | 127 | 9 |
| `without` ⛔ | 4.77 | 133 | 116 | 7 |
| `down` | 4.54 | 126 | 113 | 3 |
| `construct` | 4.34 | 208 | 167 | 1 |
| `detail` | 4.24 | 121 | 102 | 2 |
| `produce` | 4.20 | 184 | 148 | 1 |
| `break` | 3.69 | 116 | 105 | 1 |
| `identify` | 3.54 | 75 | 63 | 2 |
| `used` | 3.54 | 75 | 65 | 2 |
| `through` | 3.52 | 73 | 65 | 4 |
| `tactics` | 3.40 | 89 | 68 | 1 |
| `system` | 3.26 | 68 | 52 | 5 |
| `which` | 3.01 | 65 | 47 | 6 |
| `exploit` ⚠️ | 3.00 | 453 | 351 | 0 |
| `exactly` | 2.99 | 53 | 49 | 3 |
| `engineer` | 2.88 | 54 | 45 | 1 |
| `outcomes` | 2.80 | 50 | 35 | 1 |
| `emotional` | 2.75 | 48 | 39 | 1 |
| `aggressively` | 2.75 | 48 | 38 | 1 |

## A2. Qwen GCG (or_*) vs RWR baseQwenAtt (model-matched, unjudged)

Same target model as the GCG run, removing the model confound. baseQwenAtt is raw rewrites, not filtered to confirmed over-refusal. Still zero shared originals.


**Over-represented in Qwen GCG or_***

| word | z | n(Qwen GCG or_*) | origs | n(RWR baseQwenAtt) |
|---|--:|--:|--:|--:|
| `please` | 40.13 | 319 | 283 | 178 |
| `no` ⛔ | 37.92 | 284 | 248 | 169 |
| `answer` | 21.89 | 97 | 93 | 54 |
| `not` ⛔ | 15.11 | 96 | 91 | 392 |
| `image` | 14.77 | 61 | 54 | 11 |
| `say` | 12.33 | 31 | 31 | 29 |
| `sorry` ⛔ | 12.11 | 32 | 32 | 12 |
| `wait` | 11.06 | 26 | 24 | 11 |
| `below` | 10.99 | 25 | 24 | 26 |
| `nothing` ⛔ | 9.62 | 19 | 17 | 10 |
| `show` | 9.18 | 20 | 20 | 34 |
| `new` | 9.17 | 26 | 24 | 72 |
| `anything` | 8.85 | 20 | 19 | 40 |
| `article` | 8.35 | 16 | 16 | 5 |
| `everything` | 8.34 | 15 | 15 | 19 |
| `video` | 8.33 | 14 | 13 | 11 |
| `output` | 8.28 | 14 | 14 | 13 |
| `text` | 8.21 | 23 | 21 | 73 |
| `given` | 8.13 | 19 | 19 | 47 |
| `write` | 7.98 | 36 | 35 | 191 |

**Over-represented in RWR baseQwenAtt**

| word | z | n(RWR baseQwenAtt) | origs | n(Qwen GCG or_*) |
|---|--:|--:|--:|--:|
| `way` | 6.01 | 1627 | 1000 | 5 |
| `including` | 5.63 | 1377 | 789 | 5 |
| `what` | 5.47 | 1251 | 689 | 6 |
| `using` | 5.20 | 1411 | 866 | 19 |
| `create` | 5.16 | 1300 | 683 | 3 |
| `used` | 4.88 | 1356 | 869 | 2 |
| `make` | 4.55 | 1068 | 717 | 14 |
| `without` ⛔ | 4.53 | 873 | 584 | 7 |
| `avoid` | 4.46 | 853 | 615 | 7 |
| `like` | 4.19 | 821 | 593 | 9 |
| `inappropriate` | 4.09 | 712 | 525 | 3 |
| `dangerous` | 3.82 | 662 | 484 | 2 |
| `manipulate` | 3.80 | 656 | 393 | 2 |
| `detailed` | 3.65 | 561 | 372 | 4 |
| `considered` | 3.59 | 778 | 597 | 1 |
| `includes` | 3.47 | 696 | 430 | 1 |
| `sensitive` | 3.47 | 696 | 484 | 1 |
| `someone` | 3.44 | 526 | 362 | 5 |
| `offensive` | 3.33 | 608 | 450 | 1 |
| `potential` | 3.10 | 488 | 334 | 1 |

## B. Llama GCG vs Qwen GCG — 372 SHARED originals only

Same prompts, same method, two target models. The topic confound is removed by construction, so anything here is a real method/model difference.


**Over-represented in Llama GCG (shared)**

| word | z | n(Llama GCG (shared)) | origs | n(Qwen GCG (shared)) |
|---|--:|--:|--:|--:|
| `never` ⛔ | 2.83 | 18 | 14 | 3 |
| `which` | 2.43 | 13 | 13 | 2 |
| `while` | 2.37 | 12 | 12 | 1 |
| `because` | 2.27 | 11 | 11 | 1 |
| `anyone` | 2.06 | 9 | 8 | 1 |
| `information` | 2.04 | 10 | 10 | 2 |
| `online` | 1.79 | 7 | 7 | 1 |
| `read` | 1.79 | 7 | 7 | 1 |
| `without` ⛔ | 1.78 | 10 | 10 | 3 |
| `including` | 1.63 | 6 | 6 | 1 |
| `what` | 1.60 | 9 | 9 | 3 |
| `type` | 1.60 | 9 | 8 | 3 |
| `note` | 1.60 | 9 | 9 | 3 |
| `given` | 1.58 | 12 | 11 | 5 |
| `after` | 1.52 | 7 | 7 | 2 |
| `secret` | 1.44 | 5 | 5 | 1 |
| `description` | 1.44 | 5 | 5 | 1 |
| `know` | 1.44 | 5 | 5 | 1 |
| `dark` | 1.44 | 5 | 5 | 1 |
| `also` | 1.40 | 8 | 8 | 3 |

**Over-represented in Qwen GCG (shared)**

| word | z | n(Qwen GCG (shared)) | origs | n(Llama GCG (shared)) |
|---|--:|--:|--:|--:|
| `please` | 8.71 | 145 | 126 | 19 |
| `no` ⛔ | 6.80 | 103 | 90 | 23 |
| `link` | 2.83 | 17 | 15 | 3 |
| `wait` | 2.60 | 14 | 12 | 1 |
| `https` | 2.55 | 16 | 16 | 4 |
| `write` | 2.20 | 12 | 12 | 3 |
| `anything` | 2.12 | 9 | 9 | 1 |
| `below` | 2.05 | 14 | 13 | 5 |
| `answer` | 2.03 | 29 | 29 | 16 |
| `response` | 2.00 | 8 | 8 | 1 |
| `say` | 1.88 | 13 | 13 | 5 |
| `text` | 1.88 | 10 | 10 | 3 |
| `show` | 1.85 | 7 | 7 | 1 |
| `article` | 1.80 | 8 | 8 | 2 |
| `url` | 1.69 | 6 | 5 | 1 |
| `explain` | 1.61 | 7 | 7 | 2 |
| `following` | 1.61 | 7 | 7 | 2 |
| `difference` | 1.50 | 5 | 5 | 1 |
| `image` | 1.44 | 25 | 21 | 0 |
| `three` | 1.41 | 9 | 9 | 4 |

## C. or_* vs jb_* within Qwen GCG — opposite objectives, same everything else

`or_*` minimises the NLL of refusal openers, `jb_*` maximises it. Same search, same model, same code. If the inserted vocabulary is about danger, these should differ topically; if it is about steering the opener, they should be mirror images (prohibition vs permission). Assumes no other corpus.


**Over-represented in Qwen GCG or_***

| word | z | n(Qwen GCG or_*) | origs | n(Qwen GCG jb_*) |
|---|--:|--:|--:|--:|
| `please` | 6.92 | 319 | 283 | 8 |
| `answer` | 3.38 | 97 | 93 | 1 |
| `link` | 2.74 | 55 | 50 | 1 |
| `no` ⛔ | 2.35 | 284 | 248 | 57 |
| `not` ⛔ | 1.92 | 96 | 91 | 16 |
| `below` | 1.91 | 25 | 24 | 1 |
| `write` | 1.75 | 36 | 35 | 4 |
| `help` | 1.73 | 21 | 21 | 1 |
| `following` | 1.64 | 24 | 24 | 2 |
| `explain` | 1.57 | 18 | 18 | 1 |
| `image` | 1.56 | 61 | 54 | 0 |
| `article` | 1.45 | 16 | 16 | 1 |
| `say` | 1.45 | 31 | 31 | 4 |
| `three` | 1.31 | 19 | 19 | 2 |
| `stop` ⛔ | 1.24 | 13 | 13 | 1 |
| `picture` | 1.24 | 13 | 13 | 1 |
| `see` | 1.16 | 12 | 12 | 1 |
| `ten` | 1.16 | 12 | 12 | 1 |
| `https` | 1.11 | 31 | 30 | 0 |
| `draw` | 1.07 | 29 | 28 | 0 |

**Over-represented in Qwen GCG jb_***

| word | z | n(Qwen GCG jb_*) | origs | n(Qwen GCG or_*) |
|---|--:|--:|--:|--:|
| `wait` | 5.01 | 28 | 27 | 26 |
| `need` | 3.78 | 14 | 13 | 11 |
| `get` | 3.57 | 9 | 8 | 2 |
| `materials` | 3.31 | 9 | 8 | 1 |
| `detailed` | 3.22 | 8 | 6 | 4 |
| `use` | 3.15 | 15 | 14 | 18 |
| `ways` | 3.07 | 8 | 7 | 5 |
| `okay` | 2.88 | 9 | 8 | 8 |
| `hmm` | 2.76 | 7 | 7 | 5 |
| `feel` | 2.64 | 5 | 5 | 1 |
| `common` | 2.62 | 5 | 5 | 2 |
| `modern` | 2.62 | 5 | 5 | 2 |
| `well` | 2.24 | 5 | 5 | 4 |
| `steps` | 2.24 | 5 | 5 | 4 |
| `ideas` | 2.24 | 5 | 5 | 4 |
| `people` | 2.02 | 6 | 5 | 7 |
| `including` | 2.02 | 5 | 5 | 5 |
| `someone` | 2.02 | 5 | 5 | 5 |
| `make` | 1.96 | 9 | 7 | 14 |
| `never` ⛔ | 1.84 | 6 | 6 | 8 |

## D. Judge-confirmed vs judge-rejected, within judged or_loose

Both sides are the same arm, same search, same originals pool. If the rewrites a Sonnet judge accepted insert the same words as the ones it rejected, the judged label is not tracking the inserted vocabulary.


**Over-represented in judge-confirmed**

| word | z | n(judge-confirmed) | origs | n(judge-rejected) |
|---|--:|--:|--:|--:|
| `no` ⛔ | 8.52 | 182 | 156 | 101 |
| `not` ⛔ | 2.70 | 50 | 48 | 43 |
| `allowed` | 2.26 | 8 | 8 | 2 |
| `content` | 2.20 | 7 | 7 | 1 |
| `ignore` ⛔ | 2.17 | 14 | 14 | 8 |
| `please` | 2.13 | 145 | 124 | 173 |
| `everything` | 2.00 | 10 | 10 | 5 |
| `examples` | 1.83 | 5 | 5 | 1 |
| `never` ⛔ | 1.83 | 6 | 6 | 2 |
| `good` | 1.83 | 6 | 6 | 2 |
| `write` | 1.83 | 18 | 18 | 14 |
| `cannot` ⛔ | 1.81 | 7 | 7 | 3 |
| `forget` ⛔ | 1.81 | 7 | 7 | 3 |
| `etc` | 1.78 | 8 | 7 | 4 |
| `think` | 1.57 | 5 | 5 | 2 |
| `nothing` ⛔ | 1.55 | 11 | 9 | 8 |
| `explain` | 1.53 | 9 | 9 | 6 |
| `response` | 1.26 | 7 | 7 | 5 |
| `request` | 1.24 | 5 | 5 | 3 |
| `include` | 1.24 | 5 | 5 | 3 |

**Over-represented in judge-rejected**

| word | z | n(judge-rejected) | origs | n(judge-confirmed) |
|---|--:|--:|--:|--:|
| `image` | 3.87 | 49 | 45 | 7 |
| `link` | 3.69 | 46 | 42 | 7 |
| `https` | 3.08 | 28 | 27 | 2 |
| `below` | 2.14 | 17 | 17 | 3 |
| `draw` | 2.07 | 23 | 22 | 6 |
| `pdf` | 1.83 | 12 | 12 | 2 |
| `text` | 1.83 | 12 | 11 | 2 |
| `see` | 1.82 | 10 | 10 | 1 |
| `wait` | 1.74 | 20 | 18 | 6 |
| `http` | 1.70 | 11 | 11 | 2 |
| `images` | 1.59 | 8 | 8 | 1 |
| `thanks` | 1.59 | 8 | 8 | 1 |
| `okay` | 1.45 | 7 | 7 | 1 |
| `photo` | 1.45 | 7 | 7 | 1 |
| `show` | 1.37 | 13 | 13 | 4 |
| `different` | 1.29 | 6 | 6 | 1 |
| `attached` | 1.29 | 6 | 6 | 1 |
| `need` | 1.25 | 8 | 8 | 2 |
| `click` | 1.22 | 10 | 10 | 3 |
| `help` | 1.07 | 15 | 15 | 6 |
