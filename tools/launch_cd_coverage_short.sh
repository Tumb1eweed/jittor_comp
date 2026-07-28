#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/PGD
PYTHON_BIN=/root/miniconda3/envs/jittor/bin/python3.7
OUT=${OUT:-$ROOT/experiments/score82_cd_coverage_short}
mkdir -p "$OUT"
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export nvcc_path=/usr/local/cuda/bin/nvcc
export DISABLE_MULTIPROCESSING=1
common=("$PYTHON_BIN" "$ROOT/tools/train_shapenet_one_epoch.py"
  --use_cuda --dataset_root /home/dataset_train --datalist_dir "$ROOT/datalist"
  --train_list "$ROOT/experiments/score82_round1/train_90.txt"
  --holdout_list "$ROOT/experiments/score82_round1/holdout_10.txt"
  --precomputed_points_dir /home/dataset_train_pgd_points_50k
  --sample_points 50000 --noise_std_min 0.005 --noise_std_max 0.020 --noise_types gaussian
  --patch_size 1500 --patches_per_shape 4 --batch_size 8
  --epochs 1 --train_steps_per_epoch 160 --val_steps_per_epoch 0
  --lr 0.00001 --lr_schedule constant --grad_clip_norm 0.2 --freeze_batchnorm_stats
  --model pgd --loss infocd
  --init_weights "$ROOT/experiments/pgd_normalcorr_continue400_from7908/pgd-shapenet-epoch00-loss4.53743574.npz"
  --pgd_two_stage --pgd_second_stage_scale 0.5 --pgd_use_refine_gate --pgd_refine_gate_scale 0.25
  --pgd_composite_loss --loss_corr_weight 1.0 --loss_relative_weight 0.5
  --loss_pred_cd_weight 0.10 --loss_clean_cd_weight 0.10 --loss_infocd_weight 0.15
  --loss_uniform_weight 0.02 --loss_score_relative_weight 0.10 --loss_stage_weight 0.20
  --coverage_tail_points 256 --coverage_tail_fraction 0.20
  --soft_coverage_points 256 --soft_coverage_power 2.0
  --corr_huber_delta 0.01 --step_metrics_every 20)
names=(control tail005 tail010 soft005)
devices=(0,1 2,3 4,5 6,7)
tail=(0.0 0.005 0.01 0.0)
soft=(0.0 0.0 0.0 0.005)
pids=()
for i in "${!names[@]}"; do
  d="$OUT/${names[$i]}"; mkdir -p "$d"
  "$PYTHON_BIN" "$ROOT/tools/launch_manual_multigpu.py" --devices "${devices[$i]}" --log_dir "$d" -- \
    "${common[@]}" --loss_coverage_tail_weight "${tail[$i]}" --loss_soft_coverage_weight "${soft[$i]}" \
    --seed "$((9701+i))" --log_dir "$d" >"$d/launcher.log" 2>&1 & pids+=("$!")
done
status=0; for p in "${pids[@]}"; do wait "$p" || status=1; done; exit "$status"
