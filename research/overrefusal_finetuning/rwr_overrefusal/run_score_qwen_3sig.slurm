#!/bin/bash
#SBATCH --job-name=score-qwen-3sig
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/score_qwen_3sig_%j.out
#SBATCH --error=logs/score_qwen_3sig_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mgopalan6@gatech.edu
#SBATCH --export=ALL

# Score the SAME Sonnet benign Qwen pool the probe-only run used, but with all THREE
# corrected refusal signals (vector@L58 causal / delta-probe / multi-phrase opener logit)
# via the model-parameterized score_llama_or.py. Mirrors run_score_llama_or.slurm exactly,
# only the Qwen flags differ:
#   --model_key qwen --base_model Qwen/Qwen3-32B --vector_layer 58
#   --scorer qwen_signals/qwen_probe_raw.npz (65 layers, w=1.0 @ L58)
# NOTE (documented, not a bug): the Qwen probe collapses to L58 (NNLS w=1.0 @ L58), so
# d_probe = (Dproj[:,58]-mu)/sd is the STANDARDIZED vector -- probe and vector are the same
# signal up to the mu/sd affine shift. The genuinely distinct signal is the logit arm.
# Full 80GB H100 (bf16 32B ~64GB); MIG 40GB slice would OOM.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs probe_or/results/qwen_scored

module load cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "Qwen 3-signal pool scoring at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python -u probe_or/score_llama_or.py \
    --pairs_csv probe_or/results/qwen_scored/sonnet_benign_qwen_or.csv \
    --scorer probe_or/results/qwen_signals/qwen_probe_raw.npz \
    --vector_layer 58 \
    --opener_json refusal_atlas/opener_sets.json --model_key qwen \
    --base_model Qwen/Qwen3-32B \
    --out probe_or/results/qwen_scored/sonnet_benign_qwen_3sig.csv \
    --k 18.4 --c 0.75 --proj_bs 8 --logit_bs 8

# Build the three RWR shard sets + carve ONE common held-out split (seed 42), exactly as
# the Llama run did. Writes qwen_scored/llama_shards_{vector,probe,logit}/ + heldout_originals.json.
python -u probe_or/build_llama_shards.py \
    --scored_csv probe_or/results/qwen_scored/sonnet_benign_qwen_3sig.csv \
    --out_root probe_or/results/qwen_scored \
    --heldout_fraction 0.1 --seed 42 --n_shards 40

echo "Done at $(date)"
