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

PYTHON=/pds/pds0/idh1203/anaconda3/envs/kneep/bin/python
export MPLCONFIGDIR=/pds/pds1/slurm_workspace/idh1203/.cache/matplotlib
mkdir -p "$MPLCONFIGDIR"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# One SLURM task starts one experiment worker per visible GPU. Each model is
# fitted on one GPU; condition/fit-seed jobs run independently in parallel.
# The production_v2 preset includes 4 Poisson ranges plus one g=0 control,
# 2,097,152 train pairs/condition, radius 15 and logical batch 1024. It uses
# 500-step warmup plus cosine decay from peak 1e-3 to 1e-5 over 10,000 steps.
# The all stage saves evaluation arrays and PNGs after the fits finish.
srun --ntasks=1 "$PYTHON" -u lramb_poisson_range.py all \
  --preset production_v2 \
  --num-gpus 4 \
  --cpu-threads 1 \
  "$@"
