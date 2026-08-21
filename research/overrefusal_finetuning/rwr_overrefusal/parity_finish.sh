#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
source ~/general/bin/activate
LOG=logs/PIPELINE_PARITY.log
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }
say "watching qwen causal 19320059; writes LLAMA_VS_QWEN.md on completion"
while true; do
  st=$(sacct -j 19320059 --format=State -n -X 2>/dev/null | head -1 | tr -d ' ')
  case "$st" in
    COMPLETED) say "  19320059 COMPLETED"; break;;
    FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL) say "  19320059 $st -- writing report with whatever landed"; break;;
  esac
  sleep 60
done
python -u write_parity_report.py >> $LOG 2>&1 && say "LLAMA_VS_QWEN.md written" || say "report FAILED"
say "=== parity complete ==="
