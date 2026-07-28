#!/usr/bin/env bash
set -euo pipefail
cd /home/PGD
PY=/root/miniconda3/envs/jittor/bin/python3.7
ROOT=experiments/score82_cd_coverage_short
# Fixed train-only screening split: holdout_10.txt is the full 10% split
# (1573 shapes), while this 26-shape list is the established 77.73/78.03
# short-screen protocol.
HOLDOUT=experiments/score82_round1/holdout_screen_26.txt
names=(control tail005 tail010 soft005)
devices=(0,1 2,3 4,5 6,7)
pids=()
for i in "${!names[@]}"; do
  name="${names[$i]}"
  ckpt=$(find "$ROOT/$name" -maxdepth 1 -name 'pgd-shapenet-epoch*.npz' -print -quit)
  test -n "$ckpt" || { echo "missing checkpoint: $name" >&2; exit 1; }
  out="$ROOT/$name/holdout_eval"; mkdir -p "$out"
  "$PY" tools/run_sharded_mesh_eval.py --output_root "$out" --num_shards 2 \
    --devices "${devices[$i]}" --dataset_root /home/dataset_train --starter_root /home/starter_code \
    --workers 8 --python "$PY" --model pgd --datalist_dir datalist --val_list "$HOLDOUT" \
    --sample_points 50000 --noise_std_min .005 --noise_std_max .020 --seed 8200 --patch_size 1500 \
    --seed_k 7 --seed_k_alpha 10 --patch_batch_size 8 --patch_fusion select \
    --pgd_two_stage --pgd_second_stage_scale .5 --pgd_use_refine_gate --pgd_refine_gate_scale .25 \
    --niters 1 --use_cuda --weights "$ckpt" >"$ROOT/$name/holdout_launcher.log" 2>&1 &
  pids+=("$!")
done
status=0; for p in "${pids[@]}"; do wait "$p" || status=1; done
for name in "${names[@]}"; do
  out="$ROOT/$name/holdout_eval"
  "$PY" tools/report_directional_cd.py --pred_dir "$out/pred" --gt_dir "$out/gt" \
    --output "$out/directional_cd.json" >"$out/directional_cd.log" 2>&1 || status=1
done
exit "$status"
