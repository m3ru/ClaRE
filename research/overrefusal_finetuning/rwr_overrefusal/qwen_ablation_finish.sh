#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
source ~/general/bin/activate
LOG=logs/PIPELINE_QWEN_LOWFIX.log
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }
for j in 19361870 19361871; do
  while true; do
    st=$(sacct -j $j --format=State -n -X 2>/dev/null | head -1 | tr -d ' ')
    case "$st" in
      COMPLETED) say "  ablation $j COMPLETED"; break;;
      FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL) say "  ablation $j $st"; break;;
    esac; sleep 45
  done
done
python -u write_parity_report.py >> $LOG 2>&1 && say "LLAMA_VS_QWEN.md regenerated"
module load scipy-stack/2024b >/dev/null 2>&1
python -u make_figures.py 3 4 >> $LOG 2>&1 && say "figures 3 and 4 refreshed with corrected Qwen"
say "=== QWEN ABLATIONS + FIGURES COMPLETE ==="
