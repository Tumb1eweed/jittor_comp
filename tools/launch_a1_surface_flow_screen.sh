#!/usr/bin/env bash
# Fixed paired 26-shape A1 screen. It reuses baseline clean/noisy samples so
# score, CD, and P2S deltas are directly comparable.
set -euo pipefail

ROOT=/home/PGD
PYTHON_BIN=/root/miniconda3/envs/jittor/bin/python3.7
ROUND_ROOT=${ROUND_ROOT:-"$ROOT/experiments/score82_a1_surface_flow"}
BRANCH=${BRANCH:-distance300}
DEVICE=${DEVICE:-0}
REFERENCE_ROOT="$ROOT/experiments/score82_plain_infocd_lr/lr5e6/holdout_screen"
HOLDOUT_LIST="$ROOT/experiments/score82_round1/holdout_screen_26.txt"

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/root/miniconda3/envs/jittor/lib:/root/miniconda3/envs/jittor/lib/python3.7/site-packages/nvidia/cudnn/lib:/root/miniconda3/envs/jittor/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export nvcc_path=/usr/local/cuda/bin/nvcc
export cache_name=pgd_cuda
export DISABLE_MULTIPROCESSING=1
export use_parallel_op_compiler=0

ckpt=$(find "$ROUND_ROOT/$BRANCH" -maxdepth 1 -name 'pgd-shapenet-epoch*.npz' -print -quit)
if [[ -z "${ckpt:-}" || ! -f "$ckpt" ]]; then
  echo "missing A1 checkpoint under $ROUND_ROOT/$BRANCH" >&2
  exit 1
fi

out="$ROUND_ROOT/holdout26/$BRANCH"
mkdir -p "$out"
CUDA_VISIBLE_DEVICES="$DEVICE" "$PYTHON_BIN" "$ROOT/tools/eval_shapenet_mesh_val.py" \
  --weights "$ckpt" --model pgd \
  --dataset_root /home/dataset_train --datalist_dir "$ROOT/datalist" \
  --val_list "$HOLDOUT_LIST" --reference_eval_root "$REFERENCE_ROOT" \
  --output_root "$out" \
  --sample_points 50000 --noise_std_min 0.005 --noise_std_max 0.020 --seed 8200 \
  --patch_size 1500 --seed_k 7 --seed_k_alpha 10 \
  --patch_batch_size 8 --patch_fusion select \
  --pgd_two_stage --pgd_second_stage_scale 0.5 \
  --pgd_use_refine_gate --pgd_refine_gate_scale 0.25 \
  --pgd_use_surface_flow \
  --niters 1 --workers 8 --use_cuda --run_evaluate \
  >"$out/run.log" 2>&1

"$PYTHON_BIN" "$ROOT/tools/report_directional_cd.py" \
  --pred_dir "$out/pred" --gt_dir "$out/gt" \
  --output "$out/directional_cd.json" >"$out/directional_cd.log"
