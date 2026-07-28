#!/usr/bin/env bash
# Select Sinkhorn candidates on the fixed *training* holdout only.
set -euo pipefail

ROOT=/home/PGD
PYTHON_BIN=/root/miniconda3/envs/jittor/bin/python3.7
ROUND_ROOT=${ROUND_ROOT:-"$ROOT/experiments/score82_sinkhorn_coverage_round"}
HOLDOUT_LIST=${HOLDOUT_LIST:-"$ROOT/experiments/score82_round1/holdout_screen_26.txt"}
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export nvcc_path=/usr/local/cuda/bin/nvcc
export cache_name=pgd_cuda

common=(
  --dataset_root /home/dataset_train --starter_root /home/starter_code
  --workers 8 --python "$PYTHON_BIN"
  --model pgd --datalist_dir "$ROOT/datalist" --val_list "$HOLDOUT_LIST"
  --precomputed_points_dir /home/dataset_train_pgd_points_50k
  --sample_points 50000 --noise_std_min 0.005 --noise_std_max 0.020 --seed 8200
  --patch_size 1500 --seed_k 7 --seed_k_alpha 10 --patch_batch_size 8 --patch_fusion select
  --pgd_two_stage --pgd_second_stage_scale 0.5
  --pgd_use_refine_gate --pgd_refine_gate_scale 0.25 --niters 1 --use_cuda
)
names=(control ot002 ot005 ot010)
devices=(0,1 2,3 4,5 6,7)
pids=()
for i in "${!names[@]}"; do
  name="${names[$i]}"
  ckpt=$(find "$ROUND_ROOT/$name" -maxdepth 1 -name 'pgd-shapenet-epoch*.npz' -print -quit)
  test -n "$ckpt" || { echo "missing checkpoint: $name" >&2; exit 1; }
  out="$ROUND_ROOT/holdout_screen/$name"
  mkdir -p "$out"
  done_count=$(find "$out/pred" -name denoised.npy -type f 2>/dev/null | wc -l)
  if [ "$done_count" -eq 26 ] && [ -s "$out/evaluate.log" ]; then
    echo "reuse completed screen: $name"
    continue
  fi
  "$PYTHON_BIN" "$ROOT/tools/run_sharded_mesh_eval.py" --output_root "$out" --num_shards 2 \
    --devices "${devices[$i]}" "${common[@]}" --weights "$ckpt" >"$out/launch.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [ "$status" -ne 0 ]; then exit "$status"; fi
for name in "${names[@]}"; do
  out="$ROUND_ROOT/holdout_screen/$name"
  "$PYTHON_BIN" "$ROOT/tools/report_directional_cd.py" \
    --pred_dir "$out/pred" --gt_dir "$out/gt" --output "$out/directional_cd.json" \
    >"$out/directional_cd.log" 2>&1
done
