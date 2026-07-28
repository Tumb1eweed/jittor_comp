#!/usr/bin/env bash
# Four-way, eight-GPU screen for a learned normal/tangent stage-2 gate.
# Only train_90 is used here; validation is deliberately disabled.
set -euo pipefail

ROOT=/home/PGD
PYTHON_BIN=/root/miniconda3/envs/jittor/bin/python3.7
ROUND_ROOT=${ROUND_ROOT:-"$ROOT/experiments/stage2_surface_gate_round"}
STEPS=${STEPS:-400}

mkdir -p "$ROUND_ROOT"
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/root/miniconda3/envs/jittor/lib/python3.7/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/jittor/lib:${LD_LIBRARY_PATH:-}
export nvcc_path=/usr/local/cuda/bin/nvcc
export DISABLE_MULTIPROCESSING=1
export disable_lock=1
export use_parallel_op_compiler=0

common=(
  "$PYTHON_BIN" "$ROOT/tools/train_shapenet_one_epoch.py"
  --use_cuda --dataset_root /home/dataset_train --datalist_dir "$ROOT/datalist"
  --train_list "$ROOT/experiments/score82_round1/train_90.txt"
  --holdout_list "$ROOT/experiments/score82_round1/holdout_10.txt"
  --precomputed_points_dir /home/dataset_train_pgd_points_50k
  --sample_points 50000 --noise_std_min 0.005 --noise_std_max 0.020 --noise_types gaussian
  --patch_size 1500 --patches_per_shape 4 --batch_size 4
  --epochs 1 --train_steps_per_epoch "$STEPS" --val_steps_per_epoch 0
  --lr 0.0001 --lr_schedule warmup_cosine --lr_warmup_steps 20
  --lr_warmup_start 0.00002 --lr_min 0.00001
  --grad_clip_norm 0.2 --freeze_batchnorm_stats
  --model pgd --loss infocd
  --init_weights "$ROOT/experiments/pgd_normalcorr_continue400_from7908/pgd-shapenet-epoch00-loss4.53743574.npz"
  --pgd_two_stage --pgd_second_stage_scale 0.5
  --pgd_use_refine_gate --pgd_refine_gate_scale 0.25
  --pgd_use_stage2_dual_gate --pgd_stage2_dual_gate_scale 0.90
  --pgd_second_stage_surface_k 8 --pgd_train_stage2_dual_gate_only
  --pgd_composite_loss --loss_corr_weight 1.0 --loss_relative_weight 0.5
  --loss_pred_cd_weight 0.12 --loss_clean_cd_weight 0.08
  --loss_infocd_weight 0.15 --loss_uniform_weight 0.02
  --loss_score_relative_weight 0.10 --loss_stage_weight 0.20
  --corr_huber_delta 0.01 --step_metrics_every 10
)

names=(control plane plane_tangent plane_tangent_seam)
devices=(0,1 2,3 4,5 6,7)
pids=()

for i in "${!names[@]}"; do
  out="$ROUND_ROOT/${names[$i]}"
  mkdir -p "$out"
  extra=()
  case "${names[$i]}" in
    control)
      ;;
    plane)
      extra+=(--loss_stage2_plane_weight 0.20
              --loss_stage2_normal_residual_weight 0.10)
      ;;
    plane_tangent)
      extra+=(--loss_stage2_plane_weight 0.20
              --loss_stage2_normal_residual_weight 0.10
              --loss_stage2_tangent_target_weight 0.02
              --loss_tangent_spacing_weight 0.02
              --tangent_spacing_points 128 --tangent_spacing_k 8)
      ;;
    plane_tangent_seam)
      extra+=(--loss_stage2_plane_weight 0.20
              --loss_stage2_normal_residual_weight 0.10
              --loss_stage2_tangent_target_weight 0.02
              --loss_tangent_spacing_weight 0.02
              --tangent_spacing_points 128 --tangent_spacing_k 8
              --pgd_overlap_consistency_weight 0.05
              --pgd_overlap_stage2_only
              --pgd_overlap_consistency_normalize)
      ;;
  esac
  "$PYTHON_BIN" "$ROOT/tools/launch_manual_multigpu.py" \
    --devices "${devices[$i]}" --log_dir "$out" -- \
    "${common[@]}" "${extra[@]}" \
    --seed "$((12001 + i))" --log_dir "$out" \
    >"$out/launcher.log" 2>&1 &
  pids+=("$!")
  echo "${names[$i]} launcher_pid=${pids[-1]} devices=${devices[$i]}"
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
