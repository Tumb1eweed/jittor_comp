#!/usr/bin/env bash
# Evaluate the four plain-InfoCD+DCD candidates on the frozen train holdout.
# This is deliberately selection-only: it never reads validate.txt.
set -euo pipefail

ROOT=/home/PGD
PYTHON_BIN=/root/miniconda3/envs/jittor/bin/python3.7
ROUND_ROOT="$ROOT/experiments/score82_plain_infocd_dcd"
HOLDOUT="$ROOT/experiments/score82_round1/holdout_screen_26.txt"
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export nvcc_path=/usr/local/cuda/bin/nvcc
export cache_name=pgd_cuda
export DISABLE_MULTIPROCESSING=1

names=(dcd001 dcd002 dcd005 dcd010)
devices=(0,1 2,3 4,5 6,7)
pids=()
for i in "${!names[@]}"; do
  name="${names[$i]}"
  ckpt=$(find "$ROUND_ROOT/$name" -maxdepth 1 -name 'pgd-shapenet-epoch*.npz' -print -quit)
  test -n "$ckpt" || { echo "missing checkpoint: $name" >&2; exit 1; }
  out="$ROUND_ROOT/$name/holdout_screen"
  mkdir -p "$out"
  "$PYTHON_BIN" "$ROOT/tools/run_sharded_mesh_eval.py" --output_root "$out" \
    --num_shards 2 --devices "${devices[$i]}" --dataset_root /home/dataset_train \
    --starter_root /home/starter_code --workers 8 --python "$PYTHON_BIN" --model pgd \
    --datalist_dir "$ROOT/datalist" --val_list "$HOLDOUT" --sample_points 50000 \
    --noise_std_min 0.005 --noise_std_max 0.020 --seed 8200 --patch_size 1500 \
    --seed_k 7 --seed_k_alpha 10 --patch_batch_size 8 --patch_fusion select \
    --pgd_two_stage --pgd_second_stage_scale 0.5 --pgd_use_refine_gate \
    --pgd_refine_gate_scale 0.25 --niters 1 --use_cuda --weights "$ckpt" \
    >"$out/launch.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0
for name in "${names[@]}"; do
  out="$ROUND_ROOT/$name/holdout_screen"
  "$PYTHON_BIN" "$ROOT/tools/report_directional_cd.py" --pred_dir "$out/pred" \
    --gt_dir "$out/gt" --output "$out/directional_cd.json" >"$out/directional_cd.log" 2>&1
done
