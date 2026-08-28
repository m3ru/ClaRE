#!/bin/bash
#SBATCH --job-name=abl-llama-e2e
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-01:30:00
#SBATCH --output=logs/abl_llama_%j.out
#SBATCH --error=logs/abl_llama_%j.err
#SBATCH --export=ALL

# End-to-end: does the ABLITERATED refusal vector make a better RWR training signal?
#
# The vector is the Arditi harmful-vs-harmless diff-of-means at L12, the layer Arditi's own
# ablation criterion selected (ablate -> harmful refusal 100% -> 4%). Everything else is held
# identical to the existing L17 and L31 vector arms, so the reward layer is the only variable.
#
# All four stages are chained into ONE job on purpose: the partition has ~1800 pending jobs
# and queue time dominates wall time, so four separate submissions would cost four waits.
#
# PRE-REGISTERED PREDICTION: L12 has the WORST behavioural Spearman of any layer for either
# construction (arditi 0.058, ours 0.139, vs ~0.26 at L17 and ~0.29 at L31). We expect this arm
# to LOSE. Running it converts a correlational prediction into a behavioural test; a loss is the
# finding "causally necessary != good reward", a win overturns our layer choice.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs probe_or/results/llama_scored_abl probe_or/results/eval_llama_vector_abl_L12

module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

SCORER=probe_or/results/llama_signals/arditi_abliterated.npz
SCORED=probe_or/results/llama_scored_abl/sonnet_benign_llama_abl.csv
ROOT=probe_or/results/llama_scored_abl
# Reuse the EXISTING held-out split so this arm is evaluated on the same originals as the
# L17/L31/logit arms. Without this the comparison is against a different eval set.
HELDOUT=probe_or/results/llama_scored/heldout_originals.json
CKPT=probe_or/results/rwr_llama_ckpt_vector_abl_L12

[ -f "$SCORER" ]  || { echo "ERROR: missing $SCORER"; exit 1; }
[ -f "$HELDOUT" ] || { echo "ERROR: missing $HELDOUT"; exit 1; }

echo "### 1/4 score the pool with the abliterated vector @L12  $(date)"
python -u probe_or/score_llama_or.py \
    --pairs_csv logprob_or/results/icannot_or_sonnet/icannot_or_pairs_benign.csv \
    --scorer "$SCORER" --vector_layer 12 \
    --opener_json refusal_atlas/opener_sets.json --model_key llama \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --out "$SCORED" --k 18.4 --c 0.75 --proj_bs 16 --logit_bs 16

echo "### 2/4 build shards (vector only; probe column is inert here, w=0)  $(date)"
python -u probe_or/build_llama_shards.py \
    --scored_csv "$SCORED" --out_root "$ROOT" \
    --signals vector --heldout_json "$HELDOUT" --seed 42 --n_shards 40 \
    2>&1 | tee logs/abl_llama_shards_${SLURM_JOB_ID}.txt

# Bin edges are data-dependent (q35/q65/q85 of THIS signal's OR over training-only pairs).
# Parsed from the builder rather than guessed, matching how the L17/L31 arms were set.
BIN_EDGES=$(grep -oP 'candidate --bin_edges \K[0-9.eE+,-]+' logs/abl_llama_shards_${SLURM_JOB_ID}.txt | tail -1)
[ -n "$BIN_EDGES" ] || { echo "ERROR: could not parse bin edges from the shard builder"; exit 1; }
echo "### bin edges for this arm: $BIN_EDGES"

echo "### 3/4 train  $(date)"
python -u train_rwr.py \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
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
    --output probe_or/results/eval_llama_vector_abl_L12/eval_final.json \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --n_rewrites 4 --n_refusal_samples 4 --max_eval_prompts 200 --eval_base_model

echo "### done $(date)"
