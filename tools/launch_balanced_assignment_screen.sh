#!/usr/bin/env bash
# Paired train_90-only screen for soft balanced assignment collision loss.
set -euo pipefail
cd /home/PGD
PY=/root/miniconda3/envs/jittor/bin/python3.7
BASE=experiments/pgd_normalcorr_continue400_from7908/pgd-shapenet-epoch00-loss4.53743574.npz
ROOT=${ROOT:-experiments/score82_balanced_assignment_screen}
TRAIN=experiments/score82_round1/train_90.txt
# Fixed train-only screening split; never used for optimization.
HOLDOUT=experiments/score82_round1/holdout_screen_26.txt
mkdir -p "$ROOT"
names=(control ba002)
weights=(0.0 0.02)
seeds=(9480 9481)
pids=()
for i in 0 1; do
  name=${names[$i]}; out="$ROOT/$name"; mkdir -p "$out"
  if compgen -G "$out/pgd-shapenet-epoch*.npz" >/dev/null; then continue; fi
  CUDA_VISIBLE_DEVICES="$i" cache_name="pgd_cuda_ba_screen_$i" \
    "$PY" tools/train_shapenet_one_epoch.py --use_cuda \
      --dataset_root /home/dataset_train --datalist_dir datalist \
      --train_list "$TRAIN" --holdout_list "$HOLDOUT" \
      --precomputed_points_dir /home/dataset_train_pgd_points_50k \
      --sample_points 5000 --noise_std_min .005 --noise_std_max .020 \
      --noise_types gaussian --patch_size 1500 --patches_per_shape 4 \
      --batch_size 4 --epochs 1 --train_steps_per_epoch 300 --val_steps_per_epoch 0 \
      --lr .00001 --lr_schedule constant --grad_clip_norm .2 --freeze_batchnorm_stats \
      --model pgd --loss infocd --init_weights "$BASE" \
      --pgd_two_stage --pgd_second_stage_scale .5 --pgd_use_refine_gate \
      --pgd_refine_gate_scale .25 --pgd_composite_loss \
      --loss_corr_weight 1.0 --loss_relative_weight .5 --loss_pred_cd_weight .10 \
      --loss_clean_cd_weight .10 --loss_infocd_weight .15 --loss_uniform_weight .02 \
      --loss_score_relative_weight .10 --loss_stage_weight .20 \
      --loss_balanced_assignment_weight "${weights[$i]}" \
      --balanced_assignment_points 128 --balanced_assignment_temperature .35 \
      --corr_huber_delta .01 --step_metrics_every 25 --seed "${seeds[$i]}" \
      --log_dir "$out" >"$out/train.log" 2>&1 &
  pids+=("$!")
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
(( status == 0 )) || exit "$status"

# Mesh evaluation is only a fixed holdout screen, never used for training.
for i in 0 1; do
  name=${names[$i]}; out="$ROOT/$name"; ckpt=$(find "$out" -maxdepth 1 -name 'pgd-shapenet-epoch*.npz' -print -quit)
  test -n "$ckpt" || { echo "missing checkpoint $name" >&2; exit 2; }
  evalout="$out/holdout_eval"; mkdir -p "$evalout"
  CUDA_VISIBLE_DEVICES="$i" "$PY" tools/run_sharded_mesh_eval.py --output_root "$evalout" \
    --num_shards 1 --devices "$i" --dataset_root /home/dataset_train --starter_root /home/starter_code \
    --workers 8 --python "$PY" --model pgd --datalist_dir datalist --val_list "$HOLDOUT" \
    --sample_points 5000 --noise_std_min .005 --noise_std_max .020 --seed 8200 --patch_size 1500 \
    --seed_k 7 --seed_k_alpha 10 --patch_batch_size 8 --patch_fusion select \
    --pgd_two_stage --pgd_second_stage_scale .5 --pgd_use_refine_gate --pgd_refine_gate_scale .25 \
    --niters 1 --use_cuda --weights "$ckpt" >"$out/eval.log" 2>&1
  "$PY" tools/report_directional_cd.py --pred_dir "$evalout/pred" --gt_dir "$evalout/gt" \
    --output "$evalout/directional_cd.json" >>"$out/eval.log" 2>&1
done
