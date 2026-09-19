#!/bin/bash
# Evaluate a Pure MAIL checkpoint on Flickr30K
# Usage: bash scripts/mail_pure/eval_flickr30k.sh <checkpoint.ckpt> [gpu]
CKPT=${1:?"用法: bash scripts/mail_pure/eval_flickr30k.sh <checkpoint.ckpt> [gpu]"}
GPU=${2:-${GPU_DEVICES:-"0"}}
echo "=== Pure-MAIL Flickr30K eval | CKPT=$CKPT | GPU=$GPU ==="
cd "$(dirname "$0")/../.."
CUDA_VISIBLE_DEVICES=$GPU python eval/eval_flickr2.py \
    checkpoint_path="$CKPT"
