#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
source ~/general/bin/activate
export ANTHROPIC_API_KEY="$(tr -d '[:space:]' < ~/.anthropic_key 2>/dev/null || echo '')"
LOG=logs/PIPELINE_QWEN_SAFETY.log
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }
say "waiting on qwen safety response generation 19341944"
while true; do
  st=$(sacct -j 19341944 --format=State -n -X 2>/dev/null | head -1 | tr -d ' ')
  case "$st" in
    COMPLETED) say "  19341944 COMPLETED"; break;;
    FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL) say "  19341944 $st -- aborting"; exit 1;;
  esac
  sleep 60
done
say "harm-grading Qwen responses"
python -u grade_strongreject.py --responses probe_or/results/safety_responses_qwen.json   --out probe_or/results/strongreject_graded_qwen.json >> $LOG 2>&1   && say "grading OK" || say "grading FAILED"
say "=== qwen safety confirmation complete ==="
