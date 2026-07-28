#!/usr/bin/env bash
# Train-only DCD multiplicity continuation. The official validation split is
# never opened here; candidates are selected later on the fixed train holdout.
set -euo pipefail

ROOT=/home/PGD
PYTHON_BIN=/root/miniconda3/envs/jittor/bin/python3.7
ROUND_ROOT=${ROUND_ROOT:-"$ROOT/experiments/score82_density_aware_cd_round"}
mkdir -p "$ROUND_ROOT"
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export nvcc_path=/usr/local/cuda/bin/nvcc
export cache_name=pgd_cuda
export DISABLE_MULTIPROCESSING=1

common=(
  "$PYTHON_BIN" "$ROOT/tools/train_shapenet_one_epoch.py"
  --use_cuda --dataset_root /home/dataset_train --datalist_dir "$ROOT/datalist"
  --train_list "$ROOT/experiments/score82_round1/train_90.txt"
  --holdout_list "$ROOT/experiments/score82_round1/holdout_10.txt"
  --precomputed_points_dir /home/dataset_train_pgd_points_50k
  --sample_points 50000 --noise_std_min 0.005 --noise_std_max 0.020 --noise_types gaussian
  --patch_size 1500 --patches_per_shape 4 --batch_size 8
  # Unlike the earlier 600-step 1e-5 continuations that were frequently
  # indistinguishable from their controls, this round deliberately uses a
  # calibrated stronger schedule and retains a zero-DCD control to attribute
  # any change to optimization versus the new loss.
  --epochs 1 --train_steps_per_epoch 1000 --val_steps_per_epoch 0
  --lr 0.00003 --lr_schedule warmup_cosine --lr_warmup_steps 30 --lr_warmup_start 0.00001 --lr_min 0.000005 \
  --grad_clip_norm 0.2 --freeze_batchnorm_stats
  --model pgd --loss infocd
  --init_weights "$ROOT/experiments/pgd_normalcorr_continue400_from7908/pgd-shapenet-epoch00-loss4.53743574.npz"
  --pgd_two_stage --pgd_second_stage_scale 0.5 --pgd_use_refine_gate --pgd_refine_gate_scale 0.25
  --pgd_composite_loss --loss_corr_weight 1.0 --loss_relative_weight 0.5
  --loss_pred_cd_weight 0.10 --loss_clean_cd_weight 0.10 --loss_infocd_weight 0.15
  --loss_uniform_weight 0.02 --loss_score_relative_weight 0.10 --loss_stage_weight 0.20
  --density_aware_cd_points 256 --density_aware_cd_alpha 1.0 --density_aware_cd_lambda 1.0
  --corr_huber_delta 0.01 --step_metrics_every 10
)

names=(control dcd002 dcd005 dcd010)
devices=(0,1 2,3 4,5 6,7)
weights=(0.00 0.02 0.05 0.10)
pids=()
for i in "${!names[@]}"; do
  out="$ROUND_ROOT/${names[$i]}"
  mkdir -p "$out"
  "$PYTHON_BIN" "$ROOT/tools/launch_manual_multigpu.py" --devices "${devices[$i]}" --log_dir "$out" -- \
    "${common[@]}" --loss_density_aware_cd_weight "${weights[$i]}" \
    --seed "$((9401 + i))" --log_dir "$out" >"$out/launcher.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
