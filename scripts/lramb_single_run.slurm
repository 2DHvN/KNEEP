#!/bin/bash
#SBATCH --job-name=lramb_single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=lramb_single_%j.log

set -euo pipefail
cd /pds/pds1/slurm_workspace/idh1203/KNEEP/scripts

PYTHON="${KNEEP_PYTHON:-/pds/pds0/home/idh1203/anaconda3/envs/kneep/bin/python}"
export MPLCONFIGDIR=/pds/pds1/slurm_workspace/idh1203/.cache/matplotlib
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

srun --ntasks=1 "$PYTHON" -u lramb_single_run.py --device cuda:0 --cpu-threads 1 "$@"
