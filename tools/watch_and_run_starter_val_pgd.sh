#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT=/home/chenrui/workspace/logs/pgd_starter_val_pretrained_niters2
LOG_ROOT=/home/chenrui/workspace/logs
ENV_PY=/home/chenrui/miniconda3/envs/pgdpcf/bin/python
THRESHOLD_MB=4000

cd /home/chenrui/workspace/PGD
mkdir -p "$LOG_ROOT"

while true; do
  for GPU in 2 7; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    if [ "$USED" -lt "$THRESHOLD_MB" ]; then
      echo "$(date -Is) starting PGD starter val on GPU $GPU, used=${USED}MiB"
      CUDA_VISIBLE_DEVICES="$GPU" \
      LD_LIBRARY_PATH=/home/chenrui/miniconda3/envs/pgdpcf/lib:${LD_LIBRARY_PATH:-} \
      PYTHONUNBUFFERED=1 \
      "$ENV_PY" tools/eval_starter_val_pgd.py \
        --output-root "$OUT_ROOT" \
        --patch-size 1000 --seed-k 6 --seed-k-alpha 10 --niters 2 \
        --run-evaluate --workers 8
      exit $?
    fi
  done
  echo "$(date -Is) waiting for GPU 2/7, current used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,7 | paste -sd ',') MiB"
  sleep 60
done
