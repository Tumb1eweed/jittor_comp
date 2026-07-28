#!/usr/bin/env bash
set -euo pipefail
cd /home/PGD
TRAIN_PID="${1:-1369282}"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
mkdir -p experiments/score82_coverage_tail_screen
exec tools/launch_coverage_tail_screen.sh
