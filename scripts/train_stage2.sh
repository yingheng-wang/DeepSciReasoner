#!/bin/bash
# Stage II: train mass-conserving iterative refinement (MIR) on a frozen Stage I model.
#
#   bash   scripts/train_stage2.sh <benchmark> <seed> [extra args]
#   sbatch scripts/train_stage2.sh <benchmark> <seed> [extra args]   # from the repo root
#
# Requires TRAINED/stage1/<benchmark>_stage1_seed<seed>/model.ckpt
# (pass --stage1_ckpt <dir name> to use another Stage I run).
# Extra arguments are forwarded to train_stage2.py and override the settings below.
# Output: TRAINED/stage2/<benchmark>_stage2_seed<seed>/{model.ckpt,log.txt}
#
# Adapt the Slurm resources to your cluster (add -p/--qos/GPU type as needed).
#SBATCH -J dsr_stage2
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -N 1 -n 1 -c 16
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -G 1

set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <benchmark> <seed> [extra args]" >&2
    exit 1
fi

source "$(dirname "$0")/common.sh" 2>/dev/null || source "${SLURM_SUBMIT_DIR:?}/scripts/common.sh"
cd_repo_root "$0"

BENCH=$1
SEED=$2
shift 2
set_benchmark "$BENCH"

python train_stage2.py \
    --data_src "$DATA_SRC" \
    --label_dim "$LABEL_DIM" \
    --scaling_law 1.0 \
    --train \
    --lr 1. \
    --num-epochs 200 \
    --batch-size 128 \
    --d_model 512 \
    --graph_dropout 0.2 \
    --dec_dropout 0.0 \
    --dec_layers 6 \
    --h 8 \
    --accum_step 1 \
    --weight_decay 1e-3 \
    --MIR_lr 1e-3 \
    --step 100 \
    --grad_clip 1.0 \
    --seed "$SEED" \
    --stage1_ckpt "${BENCH}_stage1_seed${SEED}" \
    --save_dir "${BENCH}_stage2_seed${SEED}" \
    "$@"
