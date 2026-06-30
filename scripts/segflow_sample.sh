#!/usr/bin/env bash
# Run SegFlow inference on a directory of images.
# Run from the FMS repo root:
#   bash scripts/segflow_sample.sh
#
# --self_cond MUST match how the checkpoint was trained. If the model was
# trained with self-conditioning (segflow_train.sh with --self_cond), keep
# --self_cond here; otherwise remove it (or pass --noself_cond).

set -euo pipefail

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$(pwd)" \
python FMS/segflow/sample.py \
    --ckpt_path           "outputs/segflow_crack500/icfm_40000.pt" \
    --model_type          "ema_model" \
    --image_dir           "data/CRACK500O/testing/images/" \
    --save_dir            "outputs/segflow_crack500/predictions" \
    --img_size            256 \
    --batch_size          4 \
    --num_inference_steps 10 \
    --integration_method  "euler" \
    --self_cond
