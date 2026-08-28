#!/bin/bash
#SBATCH --job-name=abl-qwen-e2e
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-02:55:00
#SBATCH --output=logs/abl_qwen_%j.out
#SBATCH --error=logs/abl_qwen_%j.err
#SBATCH --export=ALL

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"

# Qwen mirror of run_abliterated_llama.slurm. Does the ABLITERATED refusal vector make a
# better RWR training signal?
#
# The vector is the Arditi harmful-vs-harmless diff-of-means at L60, selected by job 19396818's
# ablation sweep (val harmful refusal 91.4% -> 0.0%; necessity 91.4 -> 1.6% with harmless
# unmoved). Comparator: the existing rwr_qwen_ckpt_vector arm at L58 (AUC-selected). Recipe is
# otherwise identical, so the reward layer is the only variable.
#
# Full 80GB H100: Qwen3-32B bf16 is ~64GB for the scoring pass. Training uses 4-bit, matching
# the existing Qwen vector arm exactly.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs probe_or/results/qwen_scored_abl probe_or/results/eval_qwen_vector_abl_L60

module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

SCORER=probe_or/results/qwen_signals/arditi_abliterated.npz
SCORED=probe_or/results/qwen_scored_abl/sonnet_benign_qwen_abl.csv
ROOT=probe_or/results/qwen_scored_abl
HELDOUT=probe_or/results/qwen_scored/heldout_originals.json     # same split as the L58 arm
CKPT=probe_or/results/rwr_qwen_ckpt_vector_abl_L60

[ -f "$SCORER" ]  || { echo "ERROR: missing $SCORER"; exit 1; }
[ -f "$HELDOUT" ] || { echo "ERROR: missing $HELDOUT"; exit 1; }

echo "### 1/4 score the pool with the abliterated vector @L60  $(date)"
python -u probe_or/score_llama_or.py \
    --pairs_csv probe_or/results/qwen_scored/sonnet_benign_qwen_or.csv \
    --scorer "$SCORER" --vector_layer 60 \
    --opener_json refusal_atlas/opener_sets.json --model_key qwen \
    --base_model Qwen/Qwen3-32B \
    --out "$SCORED" --k 18.4 --c 0.75 --proj_bs 8 --logit_bs 8

echo "### 2/4 build shards (vector only)  $(date)"
python -u probe_or/build_llama_shards.py \
    --scored_csv "$SCORED" --out_root "$ROOT" \
    --signals vector --heldout_json "$HELDOUT" --seed 42 --n_shards 40 \
    2>&1 | tee logs/abl_qwen_shards_${SLURM_JOB_ID}.txt

BIN_EDGES=$(grep -oP 'candidate --bin_edges \K[0-9.eE+,-]+' logs/abl_qwen_shards_${SLURM_JOB_ID}.txt | tail -1)
[ -n "$BIN_EDGES" ] || { echo "ERROR: could not parse bin edges from the shard builder"; exit 1; }
echo "### bin edges for this arm: $BIN_EDGES"

echo "### 3/4 train  $(date)"
python -u train_rwr.py \
    --base_model Qwen/Qwen3-32B --load_in_4bit \
    --shard_dir "$ROOT/llama_shards_vector" \
    --output_dir "$CKPT" \
    --num_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 --learning_rate 1.5e-5 \
    --max_seq_length 512 \
    --num_bins 4 --bin_edges "$BIN_EDGES" --bin_weights 0,1,4,16 \
    --similarity_exponent 18.4 --similarity_center 0.75 --refusal_divisor 1.0 \
    --similarity_floor 0.5 --num_samples_per_epoch 4000 --seed 42

echo "### 4/4 eval vs base  $(date)"
python -u eval_rwr_llama.py \
    --adapter_dir "$CKPT/final" --heldout_json "$HELDOUT" \
    --output probe_or/results/eval_qwen_vector_abl_L60/eval_final.json \
    --base_model Qwen/Qwen3-32B \
    --n_rewrites 4 --n_refusal_samples 4 --max_eval_prompts 200 --eval_base_model

echo "### done $(date)"
