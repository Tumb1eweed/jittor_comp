#!/usr/bin/env bash
# Reliable single-GPU DCD continuation. Avoids the manual distributed launcher
# for experiments that only need one physical GPU.
set -euo pipefail

ROOT=/home/PGD
GPU=${1:?physical GPU id required}
WEIGHT=${2:?DCD weight required}
SEED=${3:?seed required}
NAME=${4:?output branch name required}
UNIFORM_WEIGHT=${5:-0.02}
DENSITY_WEIGHT=${6:-0.0}
PYTHON_BIN=/root/miniconda3/envs/jittor/bin/python3.7
OUT="$ROOT/experiments/score82_density_aware_cd_round/$NAME"
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES="$GPU" PATH=/usr/local/cuda/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
nvcc_path=/usr/local/cuda/bin/nvcc cache_name=pgd_cuda DISABLE_MULTIPROCESSING=1 \
"$PYTHON_BIN" "$ROOT/tools/train_shapenet_one_epoch.py" \
  --use_cuda --dataset_root /home/dataset_train --datalist_dir "$ROOT/datalist" \
  --train_list "$ROOT/experiments/score82_round1/train_90.txt" \
  --holdout_list "$ROOT/experiments/score82_round1/holdout_10.txt" \
  --precomputed_points_dir /home/dataset_train_pgd_points_50k \
  --sample_points 50000 --noise_std_min 0.005 --noise_std_max 0.020 --noise_types gaussian \
  --patch_size 1500 --patches_per_shape 4 --batch_size 8 \
  --epochs 1 --train_steps_per_epoch 1000 --val_steps_per_epoch 0 \
  --lr 0.00003 --lr_schedule warmup_cosine --lr_warmup_steps 30 --lr_warmup_start 0.00001 --lr_min 0.000005 \
  --grad_clip_norm 0.2 --freeze_batchnorm_stats --model pgd --loss infocd \
  --init_weights "$ROOT/experiments/pgd_normalcorr_continue400_from7908/pgd-shapenet-epoch00-loss4.53743574.npz" \
  --pgd_two_stage --pgd_second_stage_scale 0.5 --pgd_use_refine_gate --pgd_refine_gate_scale 0.25 \
  --pgd_composite_loss --loss_corr_weight 1.0 --loss_relative_weight 0.5 \
  --loss_pred_cd_weight 0.10 --loss_clean_cd_weight 0.10 --loss_infocd_weight 0.15 \
  --loss_uniform_weight "$UNIFORM_WEIGHT" --loss_score_relative_weight 0.10 --loss_stage_weight 0.20 \
  --loss_density_weight "$DENSITY_WEIGHT" \
  --density_aware_cd_points 256 --density_aware_cd_alpha 1.0 --density_aware_cd_lambda 1.0 \
  --corr_huber_delta 0.01 --step_metrics_every 10 --loss_density_aware_cd_weight "$WEIGHT" \
  --seed "$SEED" --log_dir "$OUT"
