#!/usr/bin/env bash
# Paired train-holdout evaluation for learned sigma-conditioned stage gates.
set -euo pipefail

ROOT=/home/PGD
PYTHON_BIN=/root/miniconda3/envs/jittor/bin/python3.7
ROUND_ROOT=${ROUND_ROOT:-"$ROOT/experiments/score82_p1_noise_condition/round300"}
REFERENCE_ROOT=${REFERENCE_ROOT:-"$ROOT/experiments/score82_round1/holdout_screen/branch_a_fidelity/aggregate"}
HOLDOUT_LIST="$ROOT/experiments/score82_round1/holdout_screen_26.txt"

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
  --pgd_use_noise_conditioning
  --pgd_noise_condition_hidden_dim 16
  --pgd_noise_condition_scale 0.50
  --pgd_noise_condition_min 0.005 --pgd_noise_condition_max 0.020
  --pgd_gate_noise_source estimate
  # Train-only holdout calibration: true_sigma ~= 1.992141 * raw - 0.004548.
  --pgd_noise_estimate_scale 1.992141
  --pgd_noise_estimate_bias -0.004548
  --niters 1 --workers 8 --use_cuda --run_evaluate
)

names=(score_gate corr_gate)
devices=(4 6)
pids=()

for i in "${!names[@]}"; do
  name=${names[$i]}
  ckpt=$(find "$ROUND_ROOT/$name" -maxdepth 1 -name 'pgd-shapenet-epoch*.npz' -print -quit)
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
  out="$ROUND_ROOT/holdout26/$name"
  "$PYTHON_BIN" "$ROOT/tools/report_directional_cd.py" \
    --pred_dir "$out/pred" --gt_dir "$out/gt" \
    --output "$out/directional_cd.json" >"$out/directional_cd.log"
done
