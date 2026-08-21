#!/bin/bash
# Unattended driver: runs every remaining phase to completion, chaining SLURM jobs.
# Survives the terminal closing (launched with setsid). Progress -> logs/PIPELINE.log
set -uo pipefail
cd "$(dirname "$0")"
source ~/general/bin/activate
LOG=logs/PIPELINE.log
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

wait_job(){   # $1=jobid $2=label
  local st
  while true; do
    st=$(sacct -j "$1" --format=State -n -X 2>/dev/null | head -1 | tr -d ' ')
    case "$st" in
      COMPLETED) say "  $2 ($1) COMPLETED"; return 0;;
      FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL) say "  $2 ($1) $st"; return 1;;
    esac
    sleep 45
  done
}

say "=== PIPELINE START ==="

# ---------- Phase A: leakage-free direction basis (CPU) ----------
say "Phase A: building causal direction basis on TRAIN originals"
if python -u build_causal_dirs.py --k 8 --out probe_or/results/delta/causal_dirs.npz >> "$LOG" 2>&1; then
  say "  Phase A OK"
else
  say "  Phase A FAILED -- stopping"; exit 1
fi

# ---------- Phase B: causal rank sweep (GPU) ----------
say "Phase B: causal rank sweep k=1..5 on HELD-OUT originals"
JB=$(sbatch --parsable run_causal_rank.slurm) && say "  submitted $JB" || { say "  submit failed"; exit 1; }
wait_job "$JB" "causal_rank" || say "  Phase B did not complete -- continuing to C anyway"

# ---------- Phase C: cross-attacker transfer (GPU) ----------
say "Phase C: GCG cross-attacker transfer"
JC=$(sbatch --parsable run_gcg_transfer.slurm) && say "  submitted $JC" || say "  submit failed"
[ -n "${JC:-}" ] && wait_job "$JC" "gcg_transfer"

# ---------- Phase D: consolidate (CPU) ----------
say "Phase D: writing consolidated report"
python -u write_final_report.py >> "$LOG" 2>&1 && say "  Phase D OK" || say "  Phase D FAILED"

say "=== PIPELINE COMPLETE ==="
say "Results: probe_or/results/causal_rank.json, probe_or/results/gcg_transfer.json"
say "Report : CAUSAL_RESULTS.md ; ledger: FINDINGS_STATUS.md"
