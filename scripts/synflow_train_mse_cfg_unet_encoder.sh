#!/usr/bin/env bash
# Train SynFlow (mask-conditional CFM) on CRACK500.
# Run from the FMS repo root:
#   bash scripts/crack_sfm_train_mse_cfg_new_unet_encoder.sh

set -euo pipefail

# CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$(pwd)" \
python FMS/synflow/train.py \
    --model        "icfm" \
    --data_type    "fms" \
    --data_dir     "data/CRACK500O" \
    --output_dir   "outputs/synflow_crack500_256" \
    --conditional \
    --batch_size   4 \
    --total_steps  150000 \
    --save_step    10000 \
    --omega        0.4 \
    --cfg          True \
    --parallel \
    --drop_rate    0.1 \
    --img_size     256
