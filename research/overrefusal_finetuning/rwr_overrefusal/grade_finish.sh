#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
source ~/general/bin/activate
export ANTHROPIC_API_KEY="$(tr -d '[:space:]' < ~/.anthropic_key 2>/dev/null || echo '')"
LOG=logs/PIPELINE_SAFETY.log
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }
say "waiting on safety response generation 19325483"
while true; do
  st=$(sacct -j 19325483 --format=State -n -X 2>/dev/null | head -1 | tr -d ' ')
  case "$st" in
    COMPLETED) say "  19325483 COMPLETED"; break;;
    FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL) say "  19325483 $st -- aborting grading"; exit 1;;
  esac
  sleep 45
done
say "grading responses on harmfulness (StrongREJECT-style rubric)"
python -u grade_strongreject.py --responses probe_or/results/safety_responses.json \
  --out probe_or/results/strongreject_graded.json >> $LOG 2>&1 \
  && say "grading OK -> probe_or/results/strongreject_graded.json" || say "grading FAILED"
say "=== safety grading complete ==="
