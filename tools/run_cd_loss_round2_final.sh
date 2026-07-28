#!/usr/bin/env bash
# Select one candidate from train-only holdout evidence, then run complete
# validation once for that winner; validation never selects a candidate.
set -euo pipefail
ROOT=/home/PGD
PYTHON_BIN=/root/miniconda3/envs/jittor/bin/python3.7
ROUND_ROOT=${ROUND_ROOT:-"$ROOT/experiments/score82_cd_loss_round2"}
FINAL_ROOT="$ROUND_ROOT/final_once"
mkdir -p "$FINAL_ROOT"
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export nvcc_path=/usr/local/cuda/bin/nvcc
export cache_name=pgd_cuda

"$PYTHON_BIN" "$ROOT/tools/select_cd_loss_round2_winner.py" --round_root "$ROUND_ROOT" \
  --output "$FINAL_ROOT/winner.json" >"$FINAL_ROOT/selection.log"
readarray -t selection < <("$PYTHON_BIN" - <<'PY' "$FINAL_ROOT/winner.json"
import json, sys
x=json.load(open(sys.argv[1])); print(x['checkpoint']); print('1' if x['postprocess'] else '0'); print(x['winner']['name'])
PY
)
WEIGHT=${selection[0]}; USE_POST=${selection[1]}; WINNER=${selection[2]}
RAW="$FINAL_ROOT/raw_${WINNER}"
RAW_EVAL=()
if [ "$USE_POST" = 1 ]; then
  # Raw predictions are only an intermediate for the selected postprocessor;
  # do not score them as an extra complete-validation candidate.
  RAW_EVAL=(--no_run_evaluate)
fi
"$PYTHON_BIN" "$ROOT/tools/run_sharded_mesh_eval.py" --output_root "$RAW" --dataset_root /home/dataset_train \
  --starter_root /home/starter_code --num_shards 8 --devices 0,1,2,3,4,5,6,7 --workers 8 --python "$PYTHON_BIN" \
  --model pgd --datalist_dir "$ROOT/datalist" --val_list "$ROOT/datalist/validate.txt" --sample_points 50000 \
  --noise_std_min 0.005 --noise_std_max 0.020 --seed 8200 --patch_size 1500 --seed_k 7 --seed_k_alpha 10 \
  --patch_batch_size 8 --patch_fusion select --pgd_two_stage --pgd_second_stage_scale 0.5 \
  --pgd_use_refine_gate --pgd_refine_gate_scale 0.25 --niters 1 --use_cuda --weights "$WEIGHT" "${RAW_EVAL[@]}"
FINAL="$RAW"
if [ "$USE_POST" = 1 ]; then
  PCA="$FINAL_ROOT/pca015_${WINNER}"; FINAL="$FINAL_ROOT/pca015_tangent125_${WINNER}"
  # Keep the expensive learned inference above on all eight GPUs.  The small
  # Jittor postprocessors run serially: concurrent shard startup can recurse
  # into compiler workers on this host.
  for shard in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$shard "$PYTHON_BIN" "$ROOT/tools/postprocess_pca_projection.py" --input_root "$RAW" \
      --output_root "$PCA" --k 16 --strength 0.15 --num_shards 8 --shard_index "$shard" >"$PCA.shard${shard}.log" 2>&1
  done
  for shard in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$shard "$PYTHON_BIN" "$ROOT/tools/postprocess_tangent_repulsion.py" --input_root "$PCA" \
      --output_root "$FINAL" --k 8 --margin 0.85 --step 0.125 --iterations 1 --normal_source pred \
      --num_shards 8 --shard_index "$shard" >"$FINAL.shard${shard}.log" 2>&1
  done
  "$PYTHON_BIN" /home/starter_code/evaluate.py --pred_dir "$FINAL/pred" --gt_dir "$FINAL/gt" --noisy_dir "$FINAL/noisy" \
    --mesh_dir /home/dataset_train --pred_filename denoised.npy --gt_filename clean.npy --noisy_filename noisy.npy \
    --workers 8 --verbose >"$FINAL/evaluate.log" 2>&1
fi
"$PYTHON_BIN" "$ROOT/tools/report_directional_cd.py" --pred_dir "$FINAL/pred" --gt_dir "$FINAL/gt" \
  --output "$FINAL/directional_cd.json" >"$FINAL/directional_cd.log" 2>&1
printf '%s\n' "$FINAL" >"$FINAL_ROOT/final_output_root.txt"
