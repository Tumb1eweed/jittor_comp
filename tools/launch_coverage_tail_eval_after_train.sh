#!/usr/bin/env bash
set -euo pipefail
cd /home/PGD
TRAIN_PID="${1:-1402824}"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
PY=/root/miniconda3/envs/jittor/bin/python3.7
ROOT=experiments/score82_coverage_tail_screen
for spec in "ct005 5" "ct010 6" "ct020 7"; do
  read -r name gpu <<< "$spec"
  ckpt=$(find "$ROOT/$name" -maxdepth 1 -name 'pgd-shapenet-epoch00-loss*.npz' | head -1)
  [[ -n "$ckpt" ]] || { echo "missing checkpoint for $name" >&2; exit 2; }
  out="$ROOT/$name/holdout_eval"
  mkdir -p "$out"
  "$PY" tools/run_sharded_mesh_eval.py --output_root "$out" --num_shards 1 \
    --devices "$gpu" --dataset_root /home/dataset_train --starter_root /home/starter_code \
    --workers 8 --python "$PY" --model pgd --datalist_dir datalist \
    --val_list experiments/score82_round1/holdout_screen_26.txt \
    --precomputed_points_dir /home/dataset_train_pgd_points_50k --sample_points 5000 \
    --noise_std_min .005 --noise_std_max .020 --seed 8200 --patch_size 1500 \
    --seed_k 7 --seed_k_alpha 10 --patch_batch_size 8 --patch_fusion select \
    --pgd_two_stage --pgd_second_stage_scale .5 --pgd_use_refine_gate \
    --pgd_refine_gate_scale .25 --niters 2 --use_cuda --weights "$ckpt" \
    >"$ROOT/$name/eval_launcher.log" 2>&1 &
done
wait
