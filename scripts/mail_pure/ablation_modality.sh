#!/bin/bash
# ============================================================================
# Modality-axis ablation: independent vs coupled modulation (Flickr30K / MSCOCO)
#
# Both branches keep their own scale and shift in every variant; a coupled
# variant ADDS a term generated from the other modality, so the independent
# variant is the special case with that term removed.  The bridge is the only
# difference, which keeps the comparison controlled.
#
#   V1  MaPLe-style: one dense linear map per injection point
#         model.mail_variant=bridged model.bridge_rank=0
#         params 38.66M  -> 155.7x the budget of BSI
#         Gives the coupling maximum expressive power; use it as the ceiling.
#
#   V2  LoRA-style: low-rank map U V, rank r, scaled by alpha/r
#         model.mail_variant=bridged model.bridge_rank=1
#         params 0.25M   -> 1.01x the budget of BSI (parameter-matched)
#         This is the controlled test: it isolates coupling from capacity.
#         rank 4 / 8 / 16 give 2.5x / 4.6x / 8.6x if you want a capacity sweep.
#
# Other variants available through VARIANTS=...:
#   bridged_rev:R    textual parameters generated from the visual ones
#   shared_scalar:1  one scale and one shift shared by both modalities (99 params)
#
# Usage:
#   bash scripts/mail_pure/ablation_modality.sh [dataset] [seed] [gpu_devices]
#     dataset : flickr30k (default) | coco
#     seed    : default 42
#     gpu     : default $GPU_DEVICES or 0,1,2,3
#
# Examples:
#   bash scripts/mail_pure/ablation_modality.sh                     # V1 + V2, Flickr30K
#   bash scripts/mail_pure/ablation_modality.sh coco 42 4,5,6,7     # V1 + V2, MSCOCO
#   VARIANTS="bridged:1"  bash scripts/mail_pure/ablation_modality.sh   # V2 only
#   VARIANTS="bridged:0"  bash scripts/mail_pure/ablation_modality.sh   # V1 only
# ============================================================================
set -u

# This machine's root filesystem holds /tmp and runs full, while the repository
# lives on the large data disk.  Checkpoints are written through a temporary
# location, so keep that location next to the repository instead of on /tmp,
# otherwise saving fails with "OSError: [Errno 28] No space left on device".
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# guard against being sourced or copied elsewhere
[ -f "$REPO_ROOT/train.py" ] || REPO_ROOT=$(pwd)
export TMPDIR=${TMPDIR:-$REPO_ROOT/.tmp}
mkdir -p "$TMPDIR"
AVAIL_GB=$(( $(df -Pk "$REPO_ROOT" | awk 'NR==2{print $4}') / 1024 / 1024 ))
echo "TMPDIR=$TMPDIR  (repo disk free: ${AVAIL_GB} GB)"
[ "$AVAIL_GB" -lt 5 ] && echo "!!! WARNING: less than 5 GB free on the repository disk"

DATASET=${1:-flickr30k}
SEED=${2:-42}
GPU=${3:-${GPU_DEVICES:-"0,1,2,3"}}
NPROC=$(echo "$GPU" | tr ',' '\n' | wc -l)

# variant:rank   (rank is ignored by independent / shared_scalar)
VARIANTS=${VARIANTS:-"bridged:4 bridged:0"}
EXTRA_VARIANTS=${EXTRA_VARIANTS:-""}

echo "============================================================"
echo " Modality-axis ablation | dataset=$DATASET seed=$SEED GPU=$GPU ($NPROC procs)"
echo " variants: $VARIANTS $EXTRA_VARIANTS"
echo "============================================================"

run_one () {
    local spec=$1
    local variant=${spec%%:*}
    local rank=${spec##*:}
    local tag="ABL-${variant}-r${rank}-${DATASET}"

    echo ""
    echo ">>> [$tag] starting at $(date '+%Y-%m-%d %H:%M:%S')"
    # Every rank writes to the shared stderr; the per-run training log only
    # covers rank 0, so tee the console output as well to keep tracebacks.
    CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=$NPROC train.py \
        --config-name=train_mail_pure_${DATASET} \
        seed=$SEED \
        model.mail_variant=$variant \
        model.bridge_rank=$rank \
        project_name=$tag 2>&1 | tee "logs/${tag}_console.log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "!!! [$tag] exited with code $rc -- see logs/${tag}_console.log"
    fi
    echo "<<< [$tag] finished at $(date '+%Y-%m-%d %H:%M:%S') (exit $rc)"
}

for spec in $VARIANTS $EXTRA_VARIANTS; do
    run_one "$spec"
done

echo ""
echo "============================================================"
echo " All ablation runs finished."
echo " Take the best RSUM of each run from logs/ (grep 'RSUM:', skip the"
echo " 12-image sanity check at the very beginning) and fill the ablation"
echo " table in the paper.  The independent baseline is Table I:"
echo "   Flickr30K 546.80 | MSCOCO 437.33"
echo "============================================================"
