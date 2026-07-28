#!/usr/bin/env bash
set -euo pipefail
cd /home/PGD
PY=/root/miniconda3/envs/jittor/bin/python3.7
BASE=experiments/pgd_normalcorr_continue400_from7908/pgd-shapenet-epoch00-loss4.53743574.npz
ROOT=experiments/score82_coverage_tail_screen
mkdir -p "$ROOT"
for spec in "ct005 5 .05 9415" "ct010 6 .10 9416" "ct020 7 .20 9417"; do
  read -r name gpu weight seed <<< "$spec"
  out="$ROOT/$name"
  mkdir -p "$out"
  if compgen -G "$out/pgd-shapenet-epoch00-loss*.npz" > /dev/null; then continue; fi
  CUDA_VISIBLE_DEVICES="$gpu" cache_name="pgd_cuda_ct_shared" \
    "$PY" tools/train_shapenet_one_epoch.py --use_cuda \
      --dataset_root /home/dataset_train --datalist_dir datalist \
      --train_list experiments/score82_round1/train_90.txt \
      --holdout_list experiments/score82_round1/holdout_10.txt \
      --precomputed_points_dir /home/dataset_train_pgd_points_50k \
      --sample_points 5000 --noise_std_min .005 --noise_std_max .020 \
      --noise_types gaussian --patch_size 1500 --patches_per_shape 4 \
      --batch_size 4 --epochs 1 --train_steps_per_epoch 250 --val_steps_per_epoch 0 \
      --lr .00001 --lr_schedule constant --grad_clip_norm .2 --freeze_batchnorm_stats \
      --model pgd --loss infocd --init_weights "$BASE" \
      --pgd_two_stage --pgd_second_stage_scale .5 --pgd_use_refine_gate \
      --pgd_refine_gate_scale .25 --pgd_composite_loss \
      --loss_corr_weight 1.0 --loss_relative_weight .5 --loss_pred_cd_weight .10 \
      --loss_clean_cd_weight .10 --loss_infocd_weight .15 --loss_uniform_weight .02 \
      --loss_score_relative_weight .10 --loss_stage_weight .20 \
      --loss_coverage_tail_weight "$weight" --coverage_tail_points 256 \
      --coverage_tail_fraction .20 --corr_huber_delta .01 --step_metrics_every 25 \
      --seed "$seed" --log_dir "$out" >"$out/train.log" 2>&1 &
done
wait
