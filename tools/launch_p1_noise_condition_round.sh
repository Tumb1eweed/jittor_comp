#!/usr/bin/env bash
# Train zero-initialized, sigma-conditioned stage gates on train-only data.
set -euo pipefail

ROOT=/home/PGD
PYTHON_BIN=/root/miniconda3/envs/jittor/bin/python3.7
OUT_ROOT=${OUT_ROOT:-"$ROOT/experiments/score82_p1_noise_condition/round300"}
STEPS=${STEPS:-300}
BASE_WEIGHTS="$ROOT/experiments/score82_plain_infocd_lr/lr5e6/pgd-shapenet-epoch00-loss4.52375959.npz"

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/root/miniconda3/envs/jittor/lib:/root/miniconda3/envs/jittor/lib/python3.7/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/jittor/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export nvcc_path=/usr/local/cuda/bin/nvcc
export cache_name=pgd_cuda
export DISABLE_MULTIPROCESSING=1
export use_parallel_op_compiler=0

common=(
  "$PYTHON_BIN" "$ROOT/tools/train_shapenet_one_epoch.py"
  --use_cuda
  --dataset_root /home/dataset_train
  --datalist_dir "$ROOT/datalist"
  --train_list "$ROOT/experiments/score82_round1/train_90.txt"
  --holdout_list "$ROOT/experiments/score82_round1/holdout_10.txt"
  --precomputed_points_dir /home/dataset_train_pgd_points_50k
  --sample_points 50000
  --noise_std_min 0.005 --noise_std_max 0.020 --noise_types gaussian
  --patch_size 1500 --patches_per_shape 4
  --batch_size 4 --epochs 1 --train_steps_per_epoch "$STEPS" --val_steps_per_epoch 0
  --lr 0.001 --lr_schedule warmup_cosine --lr_warmup_steps 30
  --lr_warmup_start 0.0001 --lr_min 0.0001 --grad_clip_norm 1.0
  --freeze_batchnorm_stats
  --model pgd --loss infocd --init_weights "$BASE_WEIGHTS"
  --pgd_two_stage --pgd_second_stage_scale 0.5
  --pgd_use_refine_gate --pgd_refine_gate_scale 0.25
  --pgd_use_noise_conditioning
  --pgd_noise_condition_hidden_dim 16
  --pgd_noise_condition_scale 0.50
  --pgd_noise_condition_min 0.005 --pgd_noise_condition_max 0.020
  --pgd_train_noise_condition_only
  --pgd_composite_loss
  --loss_relative_weight 0.5
  --loss_pred_cd_weight 0.25
  --loss_clean_cd_weight 0.5
  --loss_score_relative_weight 0.5
  --loss_infocd_weight 0.05
  --loss_uniform_weight 0.02
  --loss_stage_weight 0.20
  --seed 8290 --step_metrics_every 10
)

names=(score_gate corr_gate)
devices=(4 6)
pids=()

for i in "${!names[@]}"; do
  name=${names[$i]}
  if [[ -n "${BRANCH_FILTER:-}" && ",${BRANCH_FILTER}," != *",${name},"* ]]; then
    continue
  fi
  branch_args=()
  if [[ "$name" == "corr_gate" ]]; then
    branch_args=(
      --pgd_use_normal_corr_loss --pgd_normal_corr_relative
      --normal_corr_normal_weight 2.0 --normal_corr_tangent_weight 1.0
      --loss_corr_weight 0.25
    )
  else
    branch_args=(--loss_corr_weight 0.0)
  fi
  out="$OUT_ROOT/$name"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=${devices[$i]} "${common[@]}" "${branch_args[@]}" \
    --log_dir "$out" >"$out/train.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
