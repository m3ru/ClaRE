# Claude rewrites → Llama behavioral eval — runbook

Goal: of the 14,996 Claude rewrites, how many actually elicit a refusal *response* from bare Llama-3-8B-Instruct? Compares to a control set of the ~5,000 original benign prompts so we can compute a refusal-rate delta.

Split: Llama generation on Fir (GPU), Claude judge on the laptop (API). Same architecture as the data-gen run.

## Step 1 — Llama generation + regex pass (Fir, one GPU, ~1.5–2 h)

```
sbatch run_test_claude_on_llama.slurm
```

What it does:

- For every record in `dataset_research_framing_full.jsonl` (14,996 rewrites), generates a Llama-3-8B-Instruct response to the rewrite as if it's a normal user request (system = "You are a helpful assistant.", greedy decoding, max_new_tokens=256).
- Also runs the ~5,000 unique benign originals as a control (one per `benign_id`).
- Regex-classifies each response as `refused` or `compliant`. Only catches obvious refusals at the response start — exactly what the Claude judge stage will compensate for.

Output: `prompt_iteration_results/llama_behavioral_eval/llama_responses.jsonl`, ~20,000 records, one per `(role, benign_id, sample_idx)`.

Resumable: re-running skips records already in the output file.

To skip the originals control: `--no-include_originals` (cuts ~25% of GPU time, but you lose the comparison baseline).

To smoke-test: `--limit 100`.

## Step 2 — scp the JSONL to the laptop

```
scp <user>@<cluster>:$REPO/research/overrefusal_finetuning/rwr_overrefusal/prompt_iteration_results/llama_behavioral_eval/llama_responses.jsonl \
    prompt_iteration_results/llama_behavioral_eval/
```

## Step 3 — Claude judge pass (laptop, ~20–40 min via Batch API)

```
export ANTHROPIC_API_KEY=...
python judge_with_claude.py \
    --input  prompt_iteration_results/llama_behavioral_eval/llama_responses.jsonl \
    --output prompt_iteration_results/llama_behavioral_eval/llama_responses_judged.jsonl
```

What it judges (defaults):

- **All records the regex labeled `compliant`** — catches soft refusals, mid-response reversals, deflection-without-refusal phrases.
- **All records the regex labeled `refused`** — checks regex false positives. (Default changed from a 10% sample to all-of-them, because the expected refusal rate is low so subsampling provides little savings.)

Estimated cost: ~$2-3 via Batch API.

Output: same JSONL with three added fields per record:
- `claude_label` — REFUSED / COMPLIANT / UNCLEAR / None (None = not judged)
- `judge_reason` — why this record was judged (`compliant_all`, `refused_sampled`, or `regex_refused_not_sampled`)
- `final_label` — Claude's verdict where available, else regex

Stdout prints final per-role refusal rates.

## What the numbers mean

The headline is the **delta** between rewrite and original refusal rates:

```
delta = refusal_rate(rewrites) - refusal_rate(originals)
```

Possible outcomes:

- **Positive delta (e.g. 12 % vs 3 %)**: Claude's rewrites do trigger meaningful additional refusal — the activation-based OR proxy translates to actual behavior.
- **Near-zero delta**: rewrites don't behaviorally trigger refusal more than originals. The OR signal may be picking up something that doesn't manifest at the response level. Worth checking whether the top-OR rewrites specifically refuse more than low-OR ones — that'd at least validate the proxy for the high-OR tail.
- **Negative delta**: rewrites are LESS likely to be refused than originals. Would be surprising; likely a bug.

## What to do next based on the result

- **Big positive delta** → the data is doing what we hoped; the Llama-Guard 0.10-correlation finding from `<commit-or-job-id>` was specific to the Llama-Guard taxonomy, not behavioral refusal on the model the rewrites were optimized for. Strong validation for the project.
- **Near-zero delta** → the activation signal isn't translating to behavior. Consider (a) cross-tabulating refusal-rate by OR quantile to see if the proxy at least correlates in the tail, and (b) reconsidering the reward function for future RWR iterations.

## Files in this batch

- `test_claude_rewrites_on_llama.py` — Fir-side: Llama gen + regex
- `run_test_claude_on_llama.slurm` — sbatch wrapper for the above
- `judge_with_claude.py` — laptop-side: Anthropic Batch API judge pass
- `claude_behavioral_eval_runbook.md` — this file
