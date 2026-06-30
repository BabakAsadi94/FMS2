#!/usr/bin/env bash
# Train the mask generator (class-conditional CFM on grayscale masks).
# Run from the FMS repo root:
#   bash scripts/synflow_train_mask_mse_cfg.sh

set -euo pipefail

# CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$(pwd)" \
python FMS/synflow/train_mask.py \
    --model        "icfm" \
    --data_dir     "data" \
    --dataset_csv  "data/CRACK500O/CRACK500O.csv" \
    --output_dir   "outputs/mask_generator" \
    --conditional \
    --batch_size   16 \
    --total_steps  80001 \
    --save_step    10000 \
    --omega        0.4 \
    --cfg \
    --parallel
