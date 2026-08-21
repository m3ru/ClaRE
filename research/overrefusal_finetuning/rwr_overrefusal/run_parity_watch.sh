#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
source ~/general/bin/activate
LOG=logs/PIPELINE_PARITY.log
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }
say "watching qwen causal parity (19320059) + d4 probe (19320012)"
for jid in 19320059 19320012; do
  while true; do
    st=$(sacct -j $jid --format=State -n -X 2>/dev/null | head -1 | tr -d ' ')
    case "$st" in
      COMPLETED) say "  $jid COMPLETED"; break;;
      FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL) say "  $jid $st"; break;;
    esac
    sleep 60
  done
done
say "=== parity + d4 probe finished ==="
