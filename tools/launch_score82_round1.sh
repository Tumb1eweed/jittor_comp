#!/usr/bin/env bash
# Launch the three strict train-only score-82 fine-tuning branches.
set -euo pipefail

ROOT=/home/PGD
PYTHON_BIN=/root/miniconda3/envs/jittor/bin/python3.7
BASE_WEIGHTS="$ROOT/experiments/pgd_normalcorr_continue400_from7908/pgd-shapenet-epoch00-loss4.53743574.npz"
ROUND_ROOT="$ROOT/experiments/score82_round1"
TRAIN_LIST="$ROUND_ROOT/train_90.txt"
HOLDOUT_LIST="$ROUND_ROOT/holdout_10.txt"

export nvcc_path=/usr/local/cuda/bin/nvcc
export use_parallel_op_compiler=0
export disable_lock=1
export DISABLE_MULTIPROCESSING=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/root/miniconda3/envs/jittor/lib:/root/miniconda3/envs/jittor/lib/python3.7/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/jittor/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}

common=(
  "$PYTHON_BIN" tools/train_shapenet_one_epoch.py
  --use_cuda
  --dataset_root /home/dataset_train
  --datalist_dir "$ROOT/datalist"
  --train_list "$TRAIN_LIST"
  --holdout_list "$HOLDOUT_LIST"
  --precomputed_points_dir /home/dataset_train_pgd_points_50k
  --sample_points 50000
  --noise_std_min 0.005 --noise_std_max 0.020 --noise_types gaussian
  --patch_size 1500 --patches_per_shape 4 --batch_size 8
  --epochs 1 --train_steps_per_epoch 600 --val_steps_per_epoch 0
  --lr 0.000005 --lr_schedule constant --grad_clip_norm 0.2 --freeze_batchnorm_stats
  --model pgd --loss infocd
  --init_weights "$BASE_WEIGHTS"
  --pgd_two_stage --pgd_second_stage_scale 0.5
  --pgd_use_refine_gate --pgd_refine_gate_scale 0.25
  --pgd_composite_loss
  --loss_corr_weight 1.0 --loss_relative_weight 0.5
  --loss_pred_cd_weight 0.10 --loss_clean_cd_weight 0.10
  --loss_infocd_weight 0.15 --loss_uniform_weight 0.02
  --loss_score_relative_weight 0.10 --loss_stage_weight 0.20
  --corr_huber_delta 0.01 --seed 8200 --step_metrics_every 10
)

mkdir -p "$ROUND_ROOT/branch_a_fidelity" "$ROUND_ROOT/branch_b_coverage" "$ROUND_ROOT/branch_c_robustness"

"$PYTHON_BIN" tools/launch_manual_multigpu.py --devices 0,1,2 \
  --log_dir "$ROUND_ROOT/branch_a_fidelity" -- \
  "${common[@]}" --log_dir "$ROUND_ROOT/branch_a_fidelity" \
  >"$ROUND_ROOT/branch_a_fidelity/launcher.log" 2>&1 &
pid_a=$!

"$PYTHON_BIN" tools/launch_manual_multigpu.py --devices 3,4,5 \
  --log_dir "$ROUND_ROOT/branch_b_coverage" -- \
  "${common[@]}" --loss_density_weight 0.05 --log_dir "$ROUND_ROOT/branch_b_coverage" \
  >"$ROUND_ROOT/branch_b_coverage/launcher.log" 2>&1 &
pid_b=$!

"$PYTHON_BIN" tools/launch_manual_multigpu.py --devices 6,7 \
  --log_dir "$ROUND_ROOT/branch_c_robustness" -- \
  "${common[@]}" --loss_density_weight 0.05 --pgd_rotation_consistency_weight 0.02 \
  --random_z_rotation --log_dir "$ROUND_ROOT/branch_c_robustness" \
  >"$ROUND_ROOT/branch_c_robustness/launcher.log" 2>&1 &
pid_c=$!

wait "$pid_a"
wait "$pid_b"
wait "$pid_c"
