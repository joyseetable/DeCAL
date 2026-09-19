#!/bin/bash
# Evaluate a Pure MAIL checkpoint on MS-COCO
# Usage: bash scripts/mail_pure/eval_coco.sh <checkpoint.ckpt> [gpu]
CKPT=${1:?"用法: bash scripts/mail_pure/eval_coco.sh <checkpoint.ckpt> [gpu]"}
GPU=${2:-${GPU_DEVICES:-"0"}}
echo "=== Pure-MAIL COCO eval | CKPT=$CKPT | GPU=$GPU ==="
cd "$(dirname "$0")/../.."
CUDA_VISIBLE_DEVICES=$GPU python eval/eval_coco_new.py \
    checkpoint_path="$CKPT"
