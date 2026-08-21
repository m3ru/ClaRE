#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
source ~/general/bin/activate
LOG=logs/PIPELINE_QWEN_LOWFIX.log
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }
say "waiting on Qwen re-extraction 19359530 (low bin 45 -> 217)"
while true; do
  st=$(sacct -j 19359530 --format=State -n -X 2>/dev/null | head -1 | tr -d ' ')
  case "$st" in
    COMPLETED) say "  19359530 COMPLETED"; break;;
    FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL) say "  19359530 $st -- aborting"; exit 1;;
  esac; sleep 60
done
say "rebuilding Qwen bases + re-running low-bin-dependent geometry"
python -u build_causal_dirs.py --delta_dir probe_or/results/delta_qwen --layer 57 --k 8   --out probe_or/results/delta_qwen/causal_dirs.npz >> $LOG 2>&1 && say "  causal basis OK"
python -u build_symmetric_frame_dirs.py --delta_dir probe_or/results/delta_qwen --layer 57   --src_dirs probe_or/results/delta_qwen/causal_dirs.npz   --frames_json probe_or/results/delta_qwen/qwen_frames.json   --out probe_or/results/delta_qwen/qwen_own_frame_dirs.npz >> $LOG 2>&1 && say "  symmetric basis OK"
python -u analyze_delta_geometry.py --delta_dir probe_or/results/delta_qwen   --atlas probe_or/results/qwen_signals/probe_absolute.npz   --layers 16,24,32,40,48,57,63 --head_layer 57   --report HIGH_EDIT_GEOMETRY_QWEN.md >> $LOG 2>&1 && say "  geometry OK (low-bin alignment now n=217)"
say "=== qwen low-bin fix complete; ablations need a rerun on the new basis ==="
