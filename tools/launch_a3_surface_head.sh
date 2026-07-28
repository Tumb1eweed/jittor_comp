#!/usr/bin/env bash
# A3: train an explicit learned normal + signed surface-distance residual head.
# The distance layer is zero-initialised, so the starting prediction exactly
# matches the current 78.52 checkpoint.
set -euo pipefail

ROOT=/home/PGD
PYTHON_BIN=/root/miniconda3/envs/jittor/bin/python3.7
BASE_WEIGHTS=${BASE_WEIGHTS:-"$ROOT/experiments/score82_plain_infocd_lr/lr5e6/pgd-shapenet-epoch00-loss4.52375959.npz"}
OUT=${OUT:-"$ROOT/experiments/score82_a3_surface_head/round100"}
TRAIN_STEPS=${TRAIN_STEPS:-100}
DEVICE=${DEVICE:-1}

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/root/miniconda3/envs/jittor/lib:/root/miniconda3/envs/jittor/lib/python3.7/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/jittor/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export nvcc_path=/usr/local/cuda/bin/nvcc
export cache_name=${CACHE_NAME:-pgd_cuda}
export DISABLE_MULTIPROCESSING=1
export use_parallel_op_compiler=0

mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES="$DEVICE" "$PYTHON_BIN" "$ROOT/tools/train_shapenet_one_epoch.py" \
  --use_cuda \
  --dataset_root /home/dataset_train \
  --datalist_dir "$ROOT/datalist" \
  --train_list "$ROOT/experiments/score82_round1/train_90.txt" \
  --holdout_list "$ROOT/experiments/score82_round1/holdout_10.txt" \
  --precomputed_points_dir /home/dataset_train_pgd_points_50k \
  --sample_points 50000 \
  --noise_std_min 0.005 --noise_std_max 0.020 --noise_types gaussian \
  --patch_size 1500 --patches_per_shape 4 --batch_size 4 \
  --epochs 1 --train_steps_per_epoch "$TRAIN_STEPS" --val_steps_per_epoch 0 \
  --lr 0.0005 --lr_schedule warmup_cosine \
  --lr_warmup_steps 20 --lr_warmup_start 0.00005 --lr_min 0.00005 \
  --grad_clip_norm 0.5 --freeze_batchnorm_stats \
  --model pgd --loss infocd \
  --init_weights "$BASE_WEIGHTS" \
  --pgd_two_stage --pgd_second_stage_scale 0.5 \
  --pgd_use_refine_gate --pgd_refine_gate_scale 0.25 \
  --pgd_train_detach_second_stage_backbone \
  --pgd_use_surface_head --pgd_train_surface_head_only \
  --pgd_surface_head_hidden_dim 64 --pgd_surface_head_max_distance 0.02 \
  --pgd_composite_loss \
  --loss_corr_weight 0.0 --loss_relative_weight 0.25 \
  --loss_pred_cd_weight 0.25 --loss_clean_cd_weight 0.50 \
  --loss_score_relative_weight 0.50 \
  --loss_infocd_weight 0.10 --loss_uniform_weight 0.02 \
  --loss_stage_weight 0.0 \
  --pgd_surface_head_normal_weight 0.10 \
  --pgd_surface_head_plane_weight 0.50 \
  --corr_huber_delta 0.01 --seed 8302 --step_metrics_every 10 \
  --log_dir "$OUT"
