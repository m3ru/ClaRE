# Qwen3-32B RWR scale-up — training-stage brief

Date: 2026-07-28. Cluster: rorqual. Signals-stage writeup: `probe_or/qwen_experiment_brief.md`.

End-to-end move of the RWR over-refusal attacker from Llama-3-8B to Qwen3-32B:
rebuild the activation OR signal on Qwen, rescore the Sonnet benign-rewrite pool
with it, and train the Qwen attacker on the result with QLoRA.

## What was done

1. **Refusal signals on Qwen3-32B** (commit `fdaa13b`; details in
   `probe_or/qwen_experiment_brief.md`). Mass-mean refusal vector + per-layer probe
   ensemble rebuilt on Qwen against a behavioral target (dP = induced refusal rate
   measured by generation — the teacher-forced `P("I cannot")` logit signal was
   deferred because 93% of Qwen's refusal mass uses other phrasing). Raw mass-mean
   delta at L58: **Spearman 0.809 / AUC 0.976** vs dP; the NNLS ensemble puts weight
   1.0 on L58, so on Qwen the single vector ≡ the ensemble.
2. **OR rescoring of the Sonnet pool** (commit `14dee44`).
   `probe_or/score_qwen_or.py` scores the Sonnet benign-rewrite pool with the Qwen
   delta-probe: `qwen_or = exp(18.4·(sim−0.75)) · Δprobe` →
   `probe_or/results/qwen_scored/sonnet_benign_qwen_or.csv`.
3. **Shards** (this commit). `probe_or/build_qwen_shards.py` converts the scored CSV
   into 40 `load_shards()`-format shards → `probe_or/results/qwen_shards/`
   (5,998 originals / 35,850 pairs; `refusal_delta` = Qwen probe delta).
4. **Qwen loading / QLoRA** (this commit). `train_rwr.py` + `rwr_config.py` gain
   `--load_in_4bit` (NF4, double quant, `prepare_model_for_kbit_training`) so the
   32B base fits a single 80GB H100 for LoRA training; `rwr_data.py` passes
   `enable_thinking=False` to the chat template (no-op on Llama, disables Qwen3
   thinking so the completion follows the generation prompt directly).
5. **Training run** (this commit; in flight at time of writing).
   `run_rwr_qwen.slurm`: RWR on the Qwen-OR-scored pool. After the training filter
   (sim ≥ 0.5, OR ≥ 0): 1,367 pairs / 714 originals; Qwen-OR bins
   <0.1 / 0.1–0.5 / 0.5–2 / >2 = 462/399/303/203 with weights [0,1,4,16];
   3 epochs, batch 4 × grad-accum 4, lr 1.5e-5, 4,000 weighted samples/epoch →
   `probe_or/results/rwr_qwen_ckpt/`.
6. **Behavioral eval harness** (this commit; runs after training).
   `eval_rwr_qwen.py` / `run_eval_rwr_qwen.slurm`: the trained attacker (base +
   LoRA) rewrites ≤200 held-out originals (same filter + split as training), base
   Qwen under a helpful-assistant system responds, and the broad refusal-opener
   detector classifies each response. Reports the induced refusal rate vs a
   base-Qwen-as-attacker baseline plus MiniLM similarity, for the `final` and
   `epoch_2` checkpoints → `probe_or/results/eval_qwen/`.

## Open

- Training + eval results not yet in; once `eval_final.json` lands, compare against
  the Llama probe-OR run (9.25% of rewrites refused ≥1×).
