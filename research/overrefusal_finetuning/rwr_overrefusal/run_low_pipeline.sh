#!/bin/bash
# LOW-stratum power-up, end to end. Run on the LOGIN node (judge step needs internet).
#   1. CPU filter to <=2 content edits   (cheap; this is what makes the run affordable)
#   2. GPU score on the Llama target     (submitted + waited on)
#   3. build judge input for refused only
#   4. v5 judge
#   5. merge into or_low_stratum_v7.csv
set -euo pipefail
cd "$(dirname "$0")"
source ~/general/bin/activate
D=probe_or/results/low_power
mkdir -p "$D"

echo "=== [1/5] edit-distance filter (CPU, no model) ==="
python -u filter_low_edit.py \
  --gen_dir probe_or/results/gen_low_llama_logit --max_edits 2 \
  --exclude_gen_dirs probe_or/results/gen_batch2_llama_logit \
                     probe_or/results/gen_scaleup_llama_logit \
  --out "$D/low_candidates.csv"

N=$(( $(wc -l < "$D/low_candidates.csv") - 1 ))
echo "    -> $N candidates to score"

echo "=== [2/5] target scoring on Llama (GPU) ==="
JID=$(sbatch --parsable run_low_score.slurm)
echo "    submitted $JID; waiting..."
while squeue -j "$JID" -h -o %T 2>/dev/null | grep -qE "PENDING|RUNNING|COMPLETING"; do sleep 60; done
ST=$(sacct -j "$JID" --format=State -n -X | head -1 | tr -d ' ')
echo "    $JID -> $ST"
[ "$ST" = "COMPLETED" ] || { echo "scoring did not complete; stopping"; exit 1; }

echo "=== [3/5] judge input (refused subset only) ==="
python -u build_low_judge_input.py \
  --scored "$D/low_scored_llamaTgt.json" --candidates "$D/low_candidates.csv" \
  --out "$D/judge_input.csv"

echo "=== [4/5] judge v5 ==="
python -u judge_direct.py --corpus_csv "$D/judge_input.csv" --out "$D/verdicts.csv" \
  --fewshot grading/fewshot_v5.txt --model claude-sonnet-5 --workers 12

echo "=== [5/5] merge into LOW stratum v7 ==="
python -u merge_low_stratum.py --judge_input "$D/judge_input.csv" \
  --verdicts "$D/verdicts.csv" --out probe_or/results/edit_strata/or_low_stratum_v7.csv
echo "=== done ==="
