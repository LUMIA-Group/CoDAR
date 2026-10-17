#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
NUM_GPUS="${NUM_GPUS:-8}"
torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
  train/train_dit.py \
  --train_config config/train_config/config_dit.yaml \
  --num_gpus "$NUM_GPUS" \
  "$@"
