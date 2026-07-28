#!/usr/bin/env bash
# Train-only continuation for the balanced local-OT coverage loss.  This
# launcher never opens validate.txt: the fixed train holdout is reserved for
# post-training selection by the existing mesh-evaluation tooling.
set -euo pipefail

ROOT=/home/PGD
PYTHON_BIN=/root/miniconda3/envs/jittor/bin/python3.7
ROUND_ROOT=${ROUND_ROOT:-"$ROOT/experiments/score82_sinkhorn_coverage_round"}
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
  --epochs 1 --train_steps_per_epoch 600 --val_steps_per_epoch 0
  --lr 0.00001 --lr_schedule constant --grad_clip_norm 0.2 --freeze_batchnorm_stats
  --model pgd --loss infocd
  --init_weights "$ROOT/experiments/pgd_normalcorr_continue400_from7908/pgd-shapenet-epoch00-loss4.53743574.npz"
  --pgd_two_stage --pgd_second_stage_scale 0.5 --pgd_use_refine_gate --pgd_refine_gate_scale 0.25
  --pgd_composite_loss --loss_corr_weight 1.0 --loss_relative_weight 0.5
  --loss_pred_cd_weight 0.10 --loss_clean_cd_weight 0.10 --loss_infocd_weight 0.15
  --loss_uniform_weight 0.02 --loss_score_relative_weight 0.10 --loss_stage_weight 0.20
  --sinkhorn_coverage_points 128 --sinkhorn_coverage_iterations 5 --sinkhorn_coverage_blur 2.0
  --corr_huber_delta 0.01 --step_metrics_every 10
)

# Same continuation setup, four weights, and a true zero-weight control.  Each
# branch owns two GPUs; no process reads the official validation split.
names=(control ot002 ot005 ot010)
devices=(0,1 2,3 4,5 6,7)
weights=(0.00 0.02 0.05 0.10)
pids=()
for i in "${!names[@]}"; do
  out="$ROUND_ROOT/${names[$i]}"
  mkdir -p "$out"
  "$PYTHON_BIN" "$ROOT/tools/launch_manual_multigpu.py" --devices "${devices[$i]}" --log_dir "$out" -- \
    "${common[@]}" --loss_sinkhorn_coverage_weight "${weights[$i]}" \
    --seed "$((9301 + i))" --log_dir "$out" >"$out/launcher.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
