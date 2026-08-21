#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
source ~/general/bin/activate
LOG=logs/PIPELINE.log
while true; do
  st=$(sacct -j 19290917 --format=State -n -X 2>/dev/null | head -1 | tr -d ' ')
  case "$st" in
    COMPLETED) echo "[$(date '+%H:%M:%S')]   causal_rank (19290917) COMPLETED" | tee -a $LOG; break;;
    FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL) echo "[$(date '+%H:%M:%S')]   causal_rank (19290917) $st" | tee -a $LOG; break;;
  esac
  sleep 45
done
python -u write_final_report.py >> $LOG 2>&1 && echo "[$(date '+%H:%M:%S')] CAUSAL_RESULTS.md regenerated with rank sweep" | tee -a $LOG
