#!/bin/bash
# Evaluate the equal-weight ensemble of two Stage I / Stage II pairs on the test split.
#
#   bash   scripts/test.sh <benchmark> [seed_a seed_b [extra args]]
#   sbatch scripts/test.sh <benchmark> [seed_a seed_b [extra args]]   # from the repo root
#
# Seeds default to 1 and 4. Requires, for each seed N,
#   TRAINED/stage1/<benchmark>_stage1_seedN/model.ckpt
#   TRAINED/stage2/<benchmark>_stage2_seedN/model.ckpt
# Prints R2, MAE, MSE and WD.
#
# Adapt the Slurm resources to your cluster (add -p/--qos/GPU type as needed).
#SBATCH -J dsr_test
#SBATCH -o slurm_output/%x-%j.out
#SBATCH -e slurm_output/%x-%j.err
#SBATCH -N 1 -n 1 -c 16
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -G 1

set -euo pipefail
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <benchmark> [seed_a seed_b [extra args]]" >&2
    exit 1
fi

source "$(dirname "$0")/common.sh" 2>/dev/null || source "${SLURM_SUBMIT_DIR:?}/scripts/common.sh"
cd_repo_root "$0"

BENCH=$1
SEED_A=${2:-1}
SEED_B=${3:-4}
shift $(( $# < 3 ? $# : 3 ))
set_benchmark "$BENCH"

python test.py \
    --data_src "$DATA_SRC" \
    --label_dim "$LABEL_DIM" \
    --scaling_law 1.0 \
    --batch-size 128 \
    --d_model 512 \
    --dec_dropout 0.0 \
    --dec_layers 6 \
    --h 8 \
    --step 100 \
    --stage1_ckpt_1 "${BENCH}_stage1_seed${SEED_A}" \
    --stage2_ckpt_1 "${BENCH}_stage2_seed${SEED_A}" \
    --stage1_ckpt_2 "${BENCH}_stage1_seed${SEED_B}" \
    --stage2_ckpt_2 "${BENCH}_stage2_seed${SEED_B}" \
    "$@"
