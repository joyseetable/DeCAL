#!/bin/bash
# Train Pure MAIL (AL layers only) on RSITMD
# Usage: bash scripts/mail_pure/train_rsitmd.sh [seed] [gpu_devices]
SEED=${1:-42}
GPU=${2:-${GPU_DEVICES:-"0,1,2,3"}}
NPROC=$(echo $GPU | tr ',' '\n' | wc -l)
echo "=== Pure-MAIL RSITMD | GPU=$GPU | Seed=$SEED ==="
CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=$NPROC train.py \
    --config-name=train_mail_pure_rsitmd \
    seed=$SEED
