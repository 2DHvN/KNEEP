#!/bin/bash
#SBATCH --job-name=lramb_poisson
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=gpu1
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
# production_v3: g=(0.5,1,2) x ell=(1,2,4,8), plus one g=0 control.
# D=0.1, mean=0, periodic +/-1 domain-wall initialization BEFORE burn-in.
# 1024 train/64 validation/128 test trajectories x 1024 observed pairs;
# radius 15, logical batch 1024, microbatch 128, 10,000 AdamW updates.
# LR starts at 1e-3 and halves only on a validation plateau (floor 1e-5).
# all saves evaluation arrays, grouped PNGs and profile/medium/KNEEP movies.
# Override host RAM or retained mmap cache explicitly when submitting;
# a smaller cache does not guarantee a lower RAM peak and increases file I/O.
srun --ntasks=1 "$PYTHON" -u lramb_poisson_range.py all \
  --preset production_v3 \
  --num-gpus 4 \
  --cpu-threads 1 \
  "$@"
