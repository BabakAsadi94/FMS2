#!/usr/bin/env bash
# Run mask generator inference from a CSV of class-conditioned samples.
# Run from the FMS repo root:
#   bash scripts/synflow_sample_mask.sh

set -euo pipefail

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$(pwd)" \
python FMS/synflow/sample_mask.py \
    --model_dir          "outputs/mask_generator" \
    --model_name         "icfm" \
    --ckpt_step          50000 \
    --model_type         "ema_model" \
    --data_dir           "data/" \
    --dataset_csv        "data/CRACK500O/CRACK500O.csv" \
    --save_dir           "outputs/mask_samples" \
    --num_images         400 \
    --batch_size         8 \
    --omega              0.4 \
    --num_steps          200 \
    --integration_method "euler" \
    --surfix             "_fake" \
    --parallel
