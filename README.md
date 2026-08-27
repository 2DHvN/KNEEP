# KNEEP

Minimal research code for the figures used in the KNEEP study.

```text
KNEEP/
├── shell_force.py  # periodic shell-force KNEEP estimator
├── results/        # generated data and figures
├── models/         # physical models (SAOU, later LABP)
├── utils/          # training utilities
├── demos/          # focused demonstrations
└── scripts/        # reproducible figure runs
```

The first implemented experiment measures SAOU performance over

```text
T = 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1
```

using 40 independent trainings per temperature and `alpha=-0.5`. All other
scientific settings match `Corr_SAOU.ipynb`. Run it from this directory:

```bash
python -m pip install -r requirements.txt
python -m scripts.saou_temperature --device cuda
```

The script can also be launched from a parent workspace without changing the
working directory. For this layout,

```text
workspace/
├── run_job.sh
└── kneep/
```

`run_job.sh` can use:

```bash
#!/usr/bin/env bash
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

WORKSPACE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
python "$WORKSPACE_DIR/kneep/scripts/saou_temperature.py" --num-gpus 4
```

Imports and outputs are resolved from the script location, not the scheduler's
working directory. Results therefore still go to
`kneep/results/saou_temperature/`.

The Slurm option for four GPUs is cluster-dependent; some sites use
`#SBATCH --gpus=4` instead of `#SBATCH --gres=gpu:4`. Slurm sets
`CUDA_VISIBLE_DEVICES`, so the script uses its four allocated logical devices
as `cuda:0` through `cuda:3`; do not overwrite that variable in a batch job.
Outside a scheduler, the equivalent launch is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python "$WORKSPACE_DIR/kneep/scripts/saou_temperature.py" --num-gpus 4
```

The Python parent process assigns one independent run at a time to each GPU
worker and is the only process that writes `runs.csv` and the final figures.
Do not launch four separate copies of the script against the same result
directory.

The fixed output directory is `results/saou_temperature/`. Re-running the
same command resumes from `runs.csv`; completed temperature/seed pairs are
not trained again. The script writes only:

- `config.json`
- `runs.csv`
- `summary.csv`
- `temperature_performance.png`
- `temperature_performance.pdf`

No trajectories or model checkpoints are retained.

Python 3.10 or newer is required. A full run contains 280 independent
trainings. Each concurrent worker uses about 7.63 GiB of CPU RAM for its
float32 training trajectories, so four GPU workers require about 30.5 GiB
before Python, statistics, and operating-system overhead. A 64 GiB node is
recommended. The reference notebook took about 3.5 minutes for the training
portion of one run.

The homogeneous SAOU model has stationary covariance proportional to `T`,
while its exact entropy-production rate is independent of `T`. Because each
training set is standardized, the temperature curve is therefore expected to
be approximately flat; its spread measures finite-data and optimization
variation.
