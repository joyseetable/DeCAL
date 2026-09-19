#!/bin/bash
# Train Pure MAIL (AL layers only) on MS-COCO
# Usage: bash scripts/mail_pure/train_coco.sh [seed] [gpu_devices]
SEED=${1:-42}
GPU=${2:-${GPU_DEVICES:-"4,5,6,7"}}
NPROC=$(echo $GPU | tr ',' '\n' | wc -l)
echo "=== Pure-MAIL COCO | GPU=$GPU | Seed=$SEED ==="
CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=$NPROC train.py \
    --config-name=train_mail_pure_coco \
    seed=$SEED
