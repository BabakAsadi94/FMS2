#!/usr/bin/env bash
# Train SegFlow (image-to-mask flow matching) on CRACK500O.
# Run from the FMS repo root:
#   bash scripts/segflow_train.sh
#
# Self-conditioning: pass --self_cond to feed the model its own previous x1
# estimate (UNet in_channels become 6). Remove --self_cond (or pass
# --noself_cond) for the standard 3-channel SegFlow. The sampling script
# (segflow_sample.sh) MUST use the same setting as training.

set -euo pipefail

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$(pwd)" \
python FMS/segflow/train.py \
    --model                  "icfm" \
    --image_dirs_training    "data/CRACK500O/training/images/" \
    --class_dirs_training    "data/CRACK500O/training/annotations/" \
    --image_dirs_validation  "data/CRACK500O/validation/images/" \
    --class_dirs_validation  "data/CRACK500O/validation/annotations/" \
    --image_dirs_testing     "data/CRACK500O/testing/images/" \
    --class_dirs_testing     "data/CRACK500O/testing/annotations/" \
    --output_dir             "outputs/segflow_crack500" \
    --sample_saved_dir       "outputs/segflow_crack500/predictions" \
    --batch_size             4 \
    --total_steps            48002 \
    --save_step              2000 \
    --visit_step             2000 \
    --img_size               256 \
    --lr                     1e-5 \
    --warmup                 10000 \
    --num_inference_steps    10 \
    --integration_method     "euler" \
    --self_cond \
    --self_cond_prob         0.5 \
    --parallel
