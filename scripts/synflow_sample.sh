#!/usr/bin/env bash
# Run SynFlow inference on a folder of mask annotations.
# Run from the FMS repo root:
#   bash scripts/synflow_sample.sh

set -euo pipefail

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$(pwd)" \
python FMS/synflow/sample.py \
    --model_dir          "outputs/synflow_crack500_256" \
    --model_name         "icfm" \
    --ckpt_step          150000 \
    --model_type         "ema_model" \
    --data_dir           "data/CRACK500O/training/annotations/" \
    --save_dir           "outputs/samples" \
    --num_images         400 \
    --batch_size         8 \
    --omega              0.4 \
    --num_steps          200 \
    --integration_method "euler" \
    --sample_resolution  256 \
    --parallel
