# GCG rewrites vs RWR (llama-logit) rewrites — lexical comparison

- GCG rows: **1220** (1220 distinct originals)  
- RWR rows: **2409** (1496 distinct originals) — confirmed-OR only  
- originals shared by both sets: **0**  
- non-alphabetic token rate: GCG **6.6%** vs RWR **1.7%** <- GCG fluency artifact; consider --alpha_only

Weighted log-odds with informative Dirichlet prior (Monroe et al. 2008). `origs` = distinct original prompts the word appears in — a high count on few originals is not evidence.

## Over-represented in GCG

| word | z | n(GCG) | origs | n(RWR) |
|---|--:|--:|--:|--:|
| `what` | 11.89 | 196 | 186 | 70 |
| `answer` | 6.23 | 48 | 47 | 6 |
| `sentence` | 6.10 | 92 | 82 | 63 |
| `help` | 5.99 | 44 | 43 | 10 |
| `certain` | 5.62 | 39 | 38 | 5 |
| `following` | 5.54 | 128 | 126 | 117 |
| `people` | 5.26 | 50 | 50 | 26 |
| `describe` | 5.02 | 54 | 54 | 33 |
| `like` | 4.99 | 30 | 30 | 5 |
| `share` | 4.65 | 32 | 32 | 2 |
| `make` | 4.63 | 34 | 33 | 15 |
| `find` | 4.61 | 31 | 29 | 2 |
| `create` | 4.60 | 55 | 55 | 1 |
| `never` | 4.60 | 55 | 47 | 1 |
| `because` | 4.56 | 30 | 30 | 2 |
| `important` | 4.46 | 24 | 22 | 4 |
| `first` | 4.44 | 26 | 24 | 8 |
| `after` | 4.42 | 25 | 25 | 7 |
| `also` | 4.39 | 24 | 23 | 6 |
| `using` | 4.31 | 49 | 47 | 35 |

## Over-represented in RWR (llama-logit)

| word | z | n(RWR) | origs | n(GCG) |
|---|--:|--:|--:|--:|
| `construct` | 6.97 | 213 | 171 | 5 |
| `produce` | 6.05 | 184 | 148 | 3 |
| `down` | 5.96 | 130 | 114 | 10 |
| `devise` | 5.92 | 237 | 181 | 2 |
| `exploit` | 5.70 | 453 | 351 | 1 |
| `break` | 5.43 | 116 | 105 | 4 |
| `system` | 5.31 | 111 | 74 | 12 |
| `extract` | 5.30 | 157 | 122 | 2 |
| `customer` | 5.30 | 126 | 61 | 18 |
| `exploiting` | 5.03 | 96 | 78 | 4 |
| `control` | 4.97 | 93 | 76 | 4 |
| `covert` | 4.75 | 201 | 161 | 1 |
| `tactics` | 4.74 | 89 | 68 | 3 |
| `designed` | 4.67 | 186 | 150 | 1 |
| `scheme` | 4.48 | 74 | 59 | 6 |
| `draft` | 4.32 | 68 | 62 | 4 |
| `detail` | 4.21 | 122 | 103 | 1 |
| `data` | 3.80 | 110 | 73 | 26 |
| `target` | 3.76 | 80 | 68 | 1 |
| `systems` | 3.76 | 55 | 39 | 2 |
