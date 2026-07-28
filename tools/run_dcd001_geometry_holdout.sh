#!/usr/bin/env bash
# Fixed, no-GT geometry correction for the retained DCD-0.01 checkpoint.
# This is one pre-selected combination comparison on the frozen train holdout,
# not a parameter sweep and never consumes validation data for selection.
set -euo pipefail

ROOT=/home/PGD
PYTHON_BIN=/root/miniconda3/envs/jittor/bin/python3.7
SOURCE=${SOURCE:-"$ROOT/experiments/score82_plain_infocd_dcd/dcd001/holdout_screen"}
OUT=${OUT:-"$ROOT/experiments/score82_plain_infocd_dcd/dcd001/holdout_pca020_tangent010"}
PCA="$OUT/pca020"
FINAL="$OUT/final"

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export nvcc_path=/usr/local/cuda/bin/nvcc
export cache_name=pgd_cuda_dcd001_geometry

test -d "$SOURCE/pred"
mkdir -p "$OUT"

# These Jittor utilities are deliberately sharded serially.  Concurrent
# utility startup has previously caused recursive compiler workers; learned
# inference remains the parallel portion of the experiment.
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="$shard" "$PYTHON_BIN" "$ROOT/tools/postprocess_pca_projection.py" \
    --input_root "$SOURCE" --output_root "$PCA" --k 16 --strength 0.20 \
    --num_shards 2 --shard_index "$shard" >"$OUT/pca_shard${shard}.log" 2>&1
done
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="$shard" "$PYTHON_BIN" "$ROOT/tools/postprocess_tangent_repulsion.py" \
    --input_root "$PCA" --output_root "$FINAL" --k 8 --margin 0.85 --step 0.10 \
    --iterations 1 --normal_source pred --num_shards 2 --shard_index "$shard" \
    >"$OUT/tangent_shard${shard}.log" 2>&1
done

"$PYTHON_BIN" /home/starter_code/evaluate.py \
  --pred_dir "$FINAL/pred" --gt_dir "$FINAL/gt" --noisy_dir "$FINAL/noisy" \
  --mesh_dir /home/dataset_train --pred_filename denoised.npy --gt_filename clean.npy \
  --noisy_filename noisy.npy --workers 8 --verbose >"$OUT/evaluate.log" 2>&1
"$PYTHON_BIN" "$ROOT/tools/report_directional_cd.py" --pred_dir "$FINAL/pred" --gt_dir "$FINAL/gt" \
  --output "$OUT/directional_cd.json" >"$OUT/directional_cd.log" 2>&1
