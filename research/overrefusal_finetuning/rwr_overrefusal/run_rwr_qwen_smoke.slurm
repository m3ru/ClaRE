#!/bin/bash
#SBATCH --job-name=rwr-qwen-smoke
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-00:40:00
#SBATCH --output=logs/rwr_qwen_smoke_%j.out
#SBATCH --error=logs/rwr_qwen_smoke_%j.err
#SBATCH --export=ALL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module load cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "SMOKE: QLoRA path validation (4-bit load -> backward -> save -> bf16 merge -> generate) $(date)"

# tiny train: 1 epoch, 32 samples -> ~8 optimizer steps. Validates the untestable-on-login pieces.
python -u train_rwr.py \
    --base_model Qwen/Qwen3-32B --load_in_4bit \
    --shard_dir probe_or/results/qwen_shards \
    --output_dir probe_or/results/rwr_qwen_smoke \
    --num_epochs 1 --batch_size 4 --gradient_accumulation_steps 4 --learning_rate 1.5e-5 \
    --max_seq_length 512 \
    --num_bins 4 --bin_edges 0.15,0.5,1.5 --bin_weights 0,1,4,16 \
    --similarity_exponent 5.0 --similarity_center 0.75 --refusal_divisor 1.0 \
    --similarity_floor 0.5 --num_samples_per_epoch 32 --seed 42

echo "=== smoke: load+merge adapter in bf16 + generate one rewrite ==="
python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rwr_config import SYSTEM_PROMPT, PROMPT_TEMPLATE
b, a = "Qwen/Qwen3-32B", "probe_or/results/rwr_qwen_smoke/final"
tok = AutoTokenizer.from_pretrained(b); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(b, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
m = PeftModel.from_pretrained(m, a).merge_and_unload().eval()
msgs = [{"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PROMPT_TEMPLATE.format(prompt="Write a poem about spring.")}]
t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
enc = tok([t], return_tensors="pt", add_special_tokens=False).to(m.device)
out = m.generate(**enc, max_new_tokens=48, do_sample=True, temperature=0.7, pad_token_id=tok.pad_token_id)
print("SMOKE REWRITE:", repr(tok.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()))
print("SMOKE OK")
PY
echo "Smoke done at $(date)"
