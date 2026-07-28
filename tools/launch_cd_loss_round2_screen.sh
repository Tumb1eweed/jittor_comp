#!/usr/bin/env bash
# Screen the four round-2 checkpoints on the fixed train-only holdout.
# This script deliberately never accesses datalist/validate.txt.
set -euo pipefail

ROOT=/home/PGD
PYTHON_BIN=/root/miniconda3/envs/jittor/bin/python3.7
ROUND_ROOT=${ROUND_ROOT:-"$ROOT/experiments/score82_cd_loss_round2"}
HOLDOUT_LIST=${HOLDOUT_LIST:-"$ROOT/experiments/score82_round1/holdout_screen_26.txt"}
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export nvcc_path=/usr/local/cuda/bin/nvcc
export cache_name=pgd_cuda

common=(
  --dataset_root /home/dataset_train --starter_root /home/starter_code
  --workers 8 --python "$PYTHON_BIN"
  --model pgd --datalist_dir "$ROOT/datalist" --val_list "$HOLDOUT_LIST"
  --sample_points 50000 --noise_std_min 0.005 --noise_std_max 0.020 --seed 8200
  --patch_size 1500 --seed_k 7 --seed_k_alpha 10 --patch_batch_size 8 --patch_fusion select
  --pgd_two_stage --pgd_second_stage_scale 0.5
  --pgd_use_refine_gate --pgd_refine_gate_scale 0.25 --niters 1 --use_cuda
)
names=(anti_strong density_strong local_surface combined)
devices=(0,1 2,3 4,5 6,7)
pids=()
for i in "${!names[@]}"; do
  name=${names[$i]}
  ckpt=$(find "$ROUND_ROOT/$name" -maxdepth 1 -name 'pgd-shapenet-epoch*.npz' -print -quit)
  test -n "$ckpt" || { echo "missing checkpoint: $name" >&2; exit 1; }
  out="$ROUND_ROOT/holdout_screen/$name"
  mkdir -p "$out"
  # The sharded wrapper can report a non-zero status after its evaluator has
  # already written a complete result (Jittor child teardown).  A complete
  # prediction set plus an official score is authoritative and must be
  # reusable when resuming this long train-only screen.
  done_count=$(find "$out/pred" -name denoised.npy -type f 2>/dev/null | wc -l)
  if [ "$done_count" -eq 26 ] && [ -s "$out/evaluate.log" ]; then
    echo "reuse completed raw screen: $name"
    continue
  fi
  "$PYTHON_BIN" "$ROOT/tools/run_sharded_mesh_eval.py" --output_root "$out" --num_shards 2 \
    --devices "${devices[$i]}" "${common[@]}" --weights "$ckpt" >"$out/launch.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [ "$status" -eq 0 ]; then
  for name in "${names[@]}"; do
    out="$ROUND_ROOT/holdout_screen/$name"
    "$PYTHON_BIN" "$ROOT/tools/report_directional_cd.py" \
      --pred_dir "$out/pred" --gt_dir "$out/gt" --output "$out/directional_cd.json" \
      >"$out/directional_cd.log" 2>&1 || status=1
  done
fi
if [ "$status" -eq 0 ]; then
  # The combination was the best geometry-only train-holdout correction in
  # the baseline screen.  Apply it identically to every learned candidate;
  # selection remains entirely on the fixed train holdout.
  # These small Jittor postprocessors must be launched in a controlled
  # sequence.  Starting all shards together can make Jittor recursively
  # fan out compiler workers on this host.  The learned-model screen above
  # remains fully eight-GPU parallel; this step is CPU/lightweight enough
  # that correctness and determinism matter more than launch fan-out.
  for i in "${!names[@]}"; do
    name=${names[$i]}
    source="$ROUND_ROOT/holdout_screen/$name"
    pca="$ROUND_ROOT/holdout_screen/${name}_pca015"
    target="$ROUND_ROOT/holdout_screen/${name}_pca015_tangent125"
    pca_count=$(find "$pca/pred" -name denoised.npy -type f 2>/dev/null | wc -l)
    for shard in 0 1; do
      if [ "$shard" -eq 0 ]; then gpu="${devices[$i]%%,*}"; else gpu="${devices[$i]##*,}"; fi
      if [ "$pca_count" -ne 26 ]; then
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$ROOT/tools/postprocess_pca_projection.py" \
          --input_root "$source" --output_root "$pca" --k 16 --strength 0.15 --num_shards 2 --shard_index "$shard" \
          >"$pca.shard${shard}.log" 2>&1 || status=1
      fi
      if [ "$status" -eq 0 ]; then
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$ROOT/tools/postprocess_tangent_repulsion.py" \
          --input_root "$pca" --output_root "$target" --k 8 --margin 0.85 --step 0.125 --iterations 1 --normal_source pred \
          --num_shards 2 --shard_index "$shard" >"$target.shard${shard}.log" 2>&1 || status=1
      fi
    done
  done
  if [ "$status" -eq 0 ]; then
    eval_pids=()
    for name in "${names[@]}"; do
      target="$ROUND_ROOT/holdout_screen/${name}_pca015_tangent125"
      "$PYTHON_BIN" /home/starter_code/evaluate.py \
        --pred_dir "$target/pred" --gt_dir "$target/gt" --noisy_dir "$target/noisy" --mesh_dir /home/dataset_train \
        --pred_filename denoised.npy --gt_filename clean.npy --noisy_filename noisy.npy --workers 8 --verbose \
        >"$target/evaluate.log" 2>&1 &
      eval_pids+=("$!")
    done
    for pid in "${eval_pids[@]}"; do wait "$pid" || status=1; done
    if [ "$status" -eq 0 ]; then
      for name in "${names[@]}"; do
        target="$ROUND_ROOT/holdout_screen/${name}_pca015_tangent125"
        "$PYTHON_BIN" "$ROOT/tools/report_directional_cd.py" \
          --pred_dir "$target/pred" --gt_dir "$target/gt" --output "$target/directional_cd.json" \
          >"$target/directional_cd.log" 2>&1 || status=1
      done
    fi
  fi
fi
if [ "$status" -eq 0 ]; then
  bash "$ROOT/tools/run_cd_loss_round2_final.sh" >"$ROUND_ROOT/final_once.log" 2>&1 || status=1
fi
exit "$status"
