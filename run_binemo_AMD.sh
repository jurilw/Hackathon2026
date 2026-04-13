#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --partition=amd
#SBATCH --export=ALL

module load apptainer
module load cuda

cd "${SLURM_SUBMIT_DIR:-.}"
export PROJ_ROOT="/workspace"

# Use local container if present, otherwise fall back to shared space
CONTAINER="${HOME}/aging-challenge-2026/container/bionemo-framework_nightly.sif"
if [ ! -f "$CONTAINER" ]; then
    CONTAINER="/scratch/aazd1f17/shared_space/aging-challenge-2026/container/bionemo-framework_nightly.sif"
fi

apptainer run \
    --nv \
    --bind "$PWD:$PWD" \
    --pwd "$PWD" \
    "$CONTAINER" \
    python -u "$@"
