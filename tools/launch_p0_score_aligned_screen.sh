#!/usr/bin/env bash
# Evaluate the fixed train-only 26-shape holdout. All branches reuse identical
# cached clean/noisy arrays from the established seed-8200 reference run.
set -euo pipefail

ROOT=/home/PGD
PYTHON_BIN=/root/miniconda3/envs/jittor/bin/python3.7
ROUND_ROOT=${ROUND_ROOT:-"$ROOT/experiments/score82_p0_objective/round300"}
REFERENCE_ROOT=${REFERENCE_ROOT:-"$ROOT/experiments/score82_round1/holdout_screen/branch_a_fidelity/aggregate"}
HOLDOUT_LIST="$ROOT/experiments/score82_round1/holdout_screen_26.txt"
BASE_WEIGHTS="$ROOT/experiments/score82_plain_infocd_lr/lr5e6/pgd-shapenet-epoch00-loss4.52375959.npz"

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/root/miniconda3/envs/jittor/lib:/root/miniconda3/envs/jittor/lib/python3.7/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/jittor/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export nvcc_path=/usr/local/cuda/bin/nvcc
export cache_name=pgd_cuda
export DISABLE_MULTIPROCESSING=1
export use_parallel_op_compiler=0

common=(
  "$PYTHON_BIN" "$ROOT/tools/eval_shapenet_mesh_val.py"
  --model pgd
  --dataset_root /home/dataset_train
  --datalist_dir "$ROOT/datalist"
  --val_list "$HOLDOUT_LIST"
  --reference_eval_root "$REFERENCE_ROOT"
  --sample_points 50000
  --noise_std_min 0.005 --noise_std_max 0.020 --seed 8200
  --patch_size 1500 --seed_k 7 --seed_k_alpha 10
  --patch_batch_size 8 --patch_fusion select
  --pgd_two_stage --pgd_second_stage_scale 0.5
  --pgd_use_refine_gate --pgd_refine_gate_scale 0.25
  --niters 1 --workers 8 --use_cuda --run_evaluate
)

names=(baseline_current control_plain score_only corr025 corr_tangent2)
devices=(1 3 5 7 0)
pids=()

for i in "${!names[@]}"; do
  name=${names[$i]}
  if [[ -n "${EVAL_FILTER:-}" && ",${EVAL_FILTER}," != *",${name},"* ]]; then
    continue
  fi
  if [[ "$name" == "baseline_current" ]]; then
    ckpt="$BASE_WEIGHTS"
  else
    ckpt=$(find "$ROUND_ROOT/$name" -maxdepth 1 -name 'pgd-shapenet-epoch*.npz' -print -quit)
  fi
  if [[ -z "${ckpt:-}" || ! -f "$ckpt" ]]; then
    echo "missing checkpoint for $name" >&2
    exit 1
  fi
  out="$ROUND_ROOT/holdout26/$name"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=${devices[$i]} "${common[@]}" \
    --weights "$ckpt" --output_root "$out" >"$out/run.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

for name in "${names[@]}"; do
  if [[ -n "${EVAL_FILTER:-}" && ",${EVAL_FILTER}," != *",${name},"* ]]; then
    continue
  fi
  out="$ROUND_ROOT/holdout26/$name"
  "$PYTHON_BIN" "$ROOT/tools/report_directional_cd.py" \
    --pred_dir "$out/pred" --gt_dir "$out/gt" \
    --output "$out/directional_cd.json" >"$out/directional_cd.log"
done
