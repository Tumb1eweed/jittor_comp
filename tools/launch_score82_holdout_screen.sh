#!/usr/bin/env bash
# Run the fast, train-only holdout screen after score82_round1 training.
# This script intentionally never reads datalist/validate.txt.
set -euo pipefail

ROOT=/home/PGD
PYTHON_BIN=/root/miniconda3/envs/jittor/bin/python3.7
ROUND_ROOT=${ROUND_ROOT:-"$ROOT/experiments/score82_round1"}
HOLDOUT_LIST=${HOLDOUT_LIST:-"$ROUND_ROOT/holdout_screen_26.txt"}

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export nvcc_path=/usr/local/cuda/bin/nvcc
export cache_name=pgd_cuda

common=(
  "$ROOT/tools/eval_shapenet_mesh_val.py"
  --model pgd
  --dataset_root /home/dataset_train
  --datalist_dir "$ROOT/datalist"
  --val_list "$HOLDOUT_LIST"
  --sample_points 50000
  --noise_std_min 0.005 --noise_std_max 0.020 --seed 8200
  --patch_size 1500 --seed_k 7 --seed_k_alpha 10
  --patch_batch_size 8 --patch_fusion select
  --pgd_two_stage --pgd_second_stage_scale 0.5
  --pgd_use_refine_gate --pgd_refine_gate_scale 0.25
  --niters 1 --use_cuda --run_evaluate
)

branches=(branch_a_fidelity branch_b_coverage branch_c_robustness)
gpus=(0 1 2)
pids=()
for i in "${!branches[@]}"; do
  branch=${branches[$i]}
  gpu=${gpus[$i]}
  ckpt=$(find "$ROUND_ROOT/$branch" -maxdepth 1 -name 'pgd-shapenet-epoch*.npz' -print -quit)
  if [[ -z "$ckpt" ]]; then
    echo "Missing checkpoint for $branch under $ROUND_ROOT" >&2
    exit 1
  fi
  out="$ROUND_ROOT/holdout_screen/$branch"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" "${common[@]}" --weights "$ckpt" --output_root "$out" \
    >"$out/run.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
