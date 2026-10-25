# Shared helpers for the launchers in scripts/. Sourced, not executed.

BENCHMARKS="edos_caltech edos_tri phdos xanes_full xanes_o raman"

# Run from the repository root. Under sbatch the launcher executes from a spooled
# copy, so fall back to the directory the job was submitted from.
cd_repo_root() {
    local here
    here="$(cd "$(dirname "$1")" && pwd)"
    if [[ -f "$here/common.sh" ]]; then
        cd "$here/.."
    else
        cd "${SLURM_SUBMIT_DIR:?}"
    fi
    if [[ ! -f train_stage1.py ]]; then
        echo "Cannot find the repository root; submit jobs from the repository root." >&2
        exit 1
    fi
}

# Map a benchmark name to the --data_src / --label_dim used by the Python entry points.
set_benchmark() {
    case "$1" in
        edos_caltech) DATA_SRC=binned_dos_128;       LABEL_DIM=128 ;;
        edos_tri)     DATA_SRC=binned_dos_256;       LABEL_DIM=256 ;;
        phdos)        DATA_SRC=ph_dos_51;            LABEL_DIM=51  ;;
        xanes_full)   DATA_SRC=xanes_128_sigma0.5;   LABEL_DIM=128 ;;
        xanes_o)      DATA_SRC=xanes_O_128_sigma0.5; LABEL_DIM=128 ;;
        raman)        DATA_SRC=raman_128_sigma10;    LABEL_DIM=128 ;;
        *)
            echo "Unknown benchmark '$1'. Expected one of: $BENCHMARKS" >&2
            exit 1 ;;
    esac
}

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
