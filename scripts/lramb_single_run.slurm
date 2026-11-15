#!/bin/bash
#SBATCH --job-name=lramb_single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclude=gpu1
#SBATCH --gres=gpu:1
#SBATCH --output=lramb_single_%j.log

set -euo pipefail

# Submit from KNEEP/, KNEEP/scripts/, or the parent workspace.
# For another submission directory, export KNEEP_ROOT=/absolute/path/to/KNEEP.
# Slurm copies this file into its spool, so do not locate the repo from $0.
submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
repo_root="${KNEEP_ROOT:-}"
if [[ -z "$repo_root" ]]; then
  for candidate in "$submit_dir" "$submit_dir/.." "$submit_dir/KNEEP"; do
    if [[ -f "$candidate/scripts/lramb_single_run.py" ]]; then
      repo_root="$(cd "$candidate" && pwd -P)"
      break
    fi
  done
fi
if [[ -z "$repo_root" || ! -f "$repo_root/scripts/lramb_single_run.py" ]]; then
  printf 'Cannot find scripts/lramb_single_run.py. Export KNEEP_ROOT before sbatch.\n' >&2
  exit 2
fi

# Use the activated environment, or explicitly export KNEEP_PYTHON.
# No hard-coded conda installation path is assumed.
python_bin="$(command -v "${KNEEP_PYTHON:-python}" || true)"
if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
  printf 'Python not found. Activate the environment or export KNEEP_PYTHON=/path/to/env/bin/python.\n' >&2
  exit 127
fi

# Keep relative --config/--output-dir arguments relative to the submission folder.
cd "$submit_dir"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME:-$HOME/.cache}/matplotlib}"
mkdir -p "$MPLCONFIGDIR"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

printf 'Job: %s | host: %s\n' "${SLURM_JOB_ID:-manual}" "$(hostname)"
printf 'Repository: %s\nPython: %s\n' "$repo_root" "$python_bin"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-set by Slurm at task launch}"

# One experiment on one allocated GPU. cuda:0 is the first GPU visible inside
# the allocation; do not replace CUDA_VISIBLE_DEVICES with a physical GPU ID.
# Additional CLI arguments are forwarded, e.g. --config, --resume, --stage evaluate.
srun --ntasks=1 "$python_bin" -u "$repo_root/scripts/lramb_single_run.py" \
  --device cuda:0 \
  --cpu-threads 1 \
  "$@"
