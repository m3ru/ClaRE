#!/bin/bash
# Qwen mirror: same binning + analyses as Llama, on Qwen-confirmed over-refusals.
# Detached; survives terminal close. Progress -> logs/PIPELINE_QWEN.log
set -uo pipefail
cd "$(dirname "$0")"
source ~/general/bin/activate
export ANTHROPIC_API_KEY="$(tr -d '[:space:]' < ~/.anthropic_key 2>/dev/null || echo '')"
LOG=logs/PIPELINE_QWEN.log
L=57                      # Qwen causal/best layer (probe_absolute best_layer=57, AUC 0.988)
QD=probe_or/results/delta_qwen
QP=probe_or/results/low_power_qwen
ATLAS=probe_or/results/qwen_signals/probe_absolute.npz
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
wait_job(){ local st
  while true; do
    st=$(sacct -j "$1" --format=State -n -X 2>/dev/null | head -1 | tr -d ' ')
    case "$st" in
      COMPLETED) say "  $2 ($1) COMPLETED"; return 0;;
      FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL) say "  $2 ($1) $st"; return 1;;
    esac; sleep 60
  done; }
wait_array(){ local st
  while true; do
    st=$(sacct -j "$1" --format=State -n -X 2>/dev/null | grep -cE "PENDING|RUNNING")
    [ "$st" -eq 0 ] && { say "  $2 ($1) array finished: $(sacct -j "$1" --format=State -n -X | sort | uniq -c | tr '\n' ' ')"; return 0; }
    sleep 60
  done; }

say "=== QWEN PIPELINE START ==="
EXTRACT_JOB="${1:-}"; GEN_JOB="${2:-}"

# ---- Q1: HIGH-bin analyses as soon as activations exist ----
if [ -n "$EXTRACT_JOB" ]; then
  say "Q1: waiting on Qwen activation extraction $EXTRACT_JOB"
  wait_job "$EXTRACT_JOB" "qwen-extract" || say "  extraction failed -- HIGH analyses skipped"
fi
if [ -f "$QD/acts.npy" ]; then
  say "Q1: Qwen HIGH-bin geometry + frames (layer $L)"
  python -u analyze_frames.py --delta_dir "$QD" --atlas "$ATLAS" --layer "$L" \
      --report FRAME_ANALYSIS_QWEN.md >> "$LOG" 2>&1 && say "  frames OK" || say "  frames FAILED"
  python -u analyze_delta_geometry.py --delta_dir "$QD" --atlas "$ATLAS" \
      --layers 16,24,32,40,48,57,63 --head_layer "$L" \
      --report HIGH_EDIT_GEOMETRY_QWEN.md >> "$LOG" 2>&1 && say "  geometry OK" || say "  geometry FAILED"
fi

# ---- Q2: power the Qwen LOW bin (45 pairs is too few to analyse) ----
if [ -n "$GEN_JOB" ]; then
  say "Q2: waiting on Qwen LOW generation $GEN_JOB"
  wait_array "$GEN_JOB" "qwen-low-gen"
fi
if ls probe_or/results/gen_low_qwen_probe/*.json >/dev/null 2>&1; then
  say "Q2: filtering to <=2 content edits (CPU, before any GPU spend)"
  python -u filter_low_edit.py --gen_dir probe_or/results/gen_low_qwen_probe --max_edits 2 \
      --exclude_gen_dirs probe_or/results/gen_batch2_qwen_probe probe_or/results/gen_scaleup_qwen_logit \
      --out "$QP/low_candidates.csv" >> "$LOG" 2>&1 && say "  filter OK" || say "  filter FAILED"
  if [ -s "$QP/low_candidates.csv" ]; then
    say "Q2: scoring LOW candidates on Qwen target"
    JS=$(sbatch --parsable run_qwen_score_low.slurm) && say "  submitted $JS" && wait_job "$JS" "qwen-low-score"
    if [ -f "$QP/low_scored_qwenTgt.json" ]; then
      python -u build_low_judge_input.py --scored "$QP/low_scored_qwenTgt.json" \
          --candidates "$QP/low_candidates.csv" --out "$QP/judge_input.csv" >> "$LOG" 2>&1
      say "Q2: judging refused subset (v5)"
      python -u judge_direct.py --corpus_csv "$QP/judge_input.csv" --out "$QP/verdicts.csv" \
          --fewshot grading/fewshot_v5.txt --model claude-sonnet-5 --workers 12 >> "$LOG" 2>&1 \
          && say "  judge OK" || say "  judge FAILED"
      python -u merge_low_stratum.py --old probe_or/results/edit_strata/or_low_stratum_v6.csv \
          --attacker_only qwenAtt --judge_input "$QP/judge_input.csv" --verdicts "$QP/verdicts.csv" \
          --out probe_or/results/edit_strata/or_low_stratum_qwen_v7.csv >> "$LOG" 2>&1 \
          && say "  merge OK" || say "  merge FAILED"
    fi
  fi
fi

# ---- Q3: rebuild sets with the powered LOW bin, re-extract, re-analyse ----
if [ -f probe_or/results/edit_strata/or_low_stratum_qwen_v7.csv ]; then
  say "Q3: rebuilding Qwen sets with powered LOW bin"
  python -u build_delta_sets.py --attacker qwenAtt \
      --scored probe_or/results/corpus2/qwenAtt_qwenTgt.json \
      --low_scored "$QP/low_scored_qwenTgt.json" \
      --or_low probe_or/results/edit_strata/or_low_stratum_qwen_v7.csv \
      --out "$QD/prompt_sets.csv" >> "$LOG" 2>&1 && say "  sets rebuilt" || say "  sets FAILED"
  JE2=$(sbatch --parsable run_extract_delta_qwen.slurm) && say "  re-extract submitted $JE2" \
      && wait_job "$JE2" "qwen-extract-2"
  say "Q3: re-running Qwen analyses on both bins"
  python -u analyze_frames.py --delta_dir "$QD" --atlas "$ATLAS" --layer "$L" \
      --report FRAME_ANALYSIS_QWEN.md >> "$LOG" 2>&1 && say "  frames OK" || say "  frames FAILED"
  python -u analyze_delta_geometry.py --delta_dir "$QD" --atlas "$ATLAS" \
      --layers 16,24,32,40,48,57,63 --head_layer "$L" \
      --report HIGH_EDIT_GEOMETRY_QWEN.md >> "$LOG" 2>&1 && say "  geometry OK" || say "  geometry FAILED"
fi

say "=== QWEN PIPELINE COMPLETE ==="
say "Reports: FRAME_ANALYSIS_QWEN.md, HIGH_EDIT_GEOMETRY_QWEN.md"
say "LOW stratum: probe_or/results/edit_strata/or_low_stratum_qwen_v7.csv"
