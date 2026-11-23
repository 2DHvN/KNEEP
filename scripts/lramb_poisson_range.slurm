#!/bin/bash
#SBATCH --job-name=lramb_poisson
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --output=lramb_poisson_%j.log

set -euo pipefail

cd /pds/pds1/slurm_workspace/idh1203/KNEEP/scripts

PYTHON="${KNEEP_PYTHON:-/pds/pds0/home/idh1203/anaconda3/envs/kneep/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  printf 'Python executable is missing or not executable: %s\n' "$PYTHON" >&2
  printf 'Set KNEEP_PYTHON to the kneep environment Python before submitting.\n' >&2
  exit 127
fi
printf 'Python: %s\n' "$PYTHON"
export MPLCONFIGDIR=/pds/pds1/slurm_workspace/idh1203/.cache/matplotlib
mkdir -p "$MPLCONFIGDIR"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# One SLURM task starts one experiment worker per visible GPU. Each model is
# fitted on one GPU; condition/fit-seed jobs run independently in parallel.
# production: g=(-.2,-.1,-.04,-.02,0,.02,.04), ell=2.
# n=64, D=.1, tanh droplet radius=16, no mean shift.
# dt=.01, burn=2,000,000 steps; 32 train/4 validation/5 test trajectories.
# 8192 production steps, radius 6, logical batch 8192, microbatch 1024.
# Five fits per g, hidden channels 64; constant LR=1e-3.
# Default: generation + training only. Pass plot or resume as first argument.
# Override host RAM or retained mmap cache explicitly when submitting;
# a smaller cache does not guarantee a lower RAM peak and increases file I/O.
STAGE=training
case "${1:-}" in
  all|generate|training|train|resume|evaluate|plot|movies)
    STAGE="$1"
    shift
    ;;
esac
srun --ntasks=1 "$PYTHON" -u lramb_poisson_range.py "$STAGE" \
  --preset production \
  --num-gpus 4 \
  --cpu-threads 1 \
  "$@"
