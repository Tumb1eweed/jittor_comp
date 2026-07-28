#!/usr/bin/env bash
# Paired 26-shape evaluation for the stage-2 normal/tangent gate.
set -euo pipefail

ROOT=/home/PGD
PYTHON_BIN=/root/miniconda3/envs/jittor/bin/python3.7
ROUND_ROOT=${ROUND_ROOT:-"$ROOT/experiments/score82_p2_dual_gate/round300"}
REFERENCE_ROOT=${REFERENCE_ROOT:-"$ROOT/experiments/score82_round1/holdout_screen/branch_a_fidelity/aggregate"}
HOLDOUT_LIST="$ROOT/experiments/score82_round1/holdout_screen_26.txt"
CKPT=$(find "$ROUND_ROOT/dual_gate_score" -maxdepth 1 -name 'pgd-shapenet-epoch*.npz' -print -quit)

if [[ -z "${CKPT:-}" || ! -f "$CKPT" ]]; then
  echo "missing P2 checkpoint" >&2
  exit 1
fi

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/root/miniconda3/envs/jittor/lib:/root/miniconda3/envs/jittor/lib/python3.7/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/jittor/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export nvcc_path=/usr/local/cuda/bin/nvcc
export cache_name=pgd_cuda
export DISABLE_MULTIPROCESSING=1
export use_parallel_op_compiler=0

OUT="$ROUND_ROOT/holdout26/dual_gate_score"
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE:-6} "$PYTHON_BIN" "$ROOT/tools/eval_shapenet_mesh_val.py" \
  --model pgd \
  --dataset_root /home/dataset_train \
  --datalist_dir "$ROOT/datalist" \
  --val_list "$HOLDOUT_LIST" \
  --reference_eval_root "$REFERENCE_ROOT" \
  --sample_points 50000 \
  --noise_std_min 0.005 --noise_std_max 0.020 --seed 8200 \
  --patch_size 1500 --seed_k 7 --seed_k_alpha 10 \
  --patch_batch_size 8 --patch_fusion select \
  --pgd_two_stage --pgd_second_stage_scale 0.5 \
  --pgd_use_refine_gate --pgd_refine_gate_scale 0.25 \
  --pgd_use_stage2_dual_gate --pgd_stage2_dual_gate_scale 0.90 \
  --pgd_second_stage_surface_k 16 \
  --niters 1 --workers 8 --use_cuda --run_evaluate \
  --weights "$CKPT" --output_root "$OUT" >"$OUT/run.log" 2>&1

"$PYTHON_BIN" "$ROOT/tools/report_directional_cd.py" \
  --pred_dir "$OUT/pred" --gt_dir "$OUT/gt" \
  --output "$OUT/directional_cd.json" >"$OUT/directional_cd.log"
