#!/bin/bash
# Stage I: train the structure-aware spectrum decoder.
#
#   bash   scripts/train_stage1.sh <benchmark> <seed> [extra args]
#   sbatch scripts/train_stage1.sh <benchmark> <seed> [extra args]   # from the repo root
#
# <benchmark>: edos_caltech | edos_tri | phdos | xanes_full | xanes_o | raman
# Extra arguments are forwarded to train_stage1.py and override the settings below.
# Output: TRAINED/stage1/<benchmark>_stage1_seed<seed>/{model.ckpt,log.txt}
#
# Adapt the Slurm resources to your cluster (add -p/--qos/GPU type as needed).
#SBATCH -J dsr_stage1
#SBATCH -o slurm_output/%x-%j.out
#SBATCH -e slurm_output/%x-%j.err
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

python train_stage1.py \
    --data_src "$DATA_SRC" \
    --label_dim "$LABEL_DIM" \
    --scaling_law 1.0 \
    --train \
    --lr 1. \
    --warmup 300 \
    --T0 120 \
    --num-epochs 200 \
    --c_epochs 121 \
    --batch-size 128 \
    --d_model 512 \
    --graph_dropout 0.2 \
    --dec_dropout 0.0 \
    --dec_layers 6 \
    --temp 1.0 \
    --h 8 \
    --lambda_scale 1.0 \
    --lambda_factor 1.0 \
    --accum_step 1 \
    --clip 1e-2 \
    --weight_decay 0. \
    --rate_decay 4. \
    --seed "$SEED" \
    --save_dir "${BENCH}_stage1_seed${SEED}" \
    "$@"
