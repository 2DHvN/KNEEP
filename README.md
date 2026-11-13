# KNEEP

Run from the parent workspace.

Train:

```bash
python KNEEP/scripts/saou_perform.py
python KNEEP/scripts/saou_temperature.py
```

For multiple GPUs, add `--num-gpus 2`; for CPU, add `--device cpu`.
Rerun the same command to resume.

Plot saved results:

```bash
python KNEEP/scripts/plot_saou_perform.py
python KNEEP/scripts/plot_saou_temperature.py
```

Results are saved under `KNEEP/results/saou_perform/` and
`KNEEP/results/saou_temperature/`, with plots in each `figures/` directory.
Both plot commands accept `--results-dir PATH` and `--dpi 300`.
Move previous results aside before training with changed experiment settings.

## Poisson circular-cluster sanity notebook

Open [demos/poisson_sanity.ipynb](demos/poisson_sanity.ipynb) for editable
64x64 circular-droplet experiments, centred on g=1 and ell=8. Initialization
solves a radial local-passive boundary-value problem with centre/exterior
Neumann conditions and fixed disk mass, then by default continues to a
stationary solution of the actual periodic Poisson+active lattice equation.
The chemical potential is solved with the field; curved-droplet bulk values
are not artificially pinned to +/-1. Failed initialization is reported separately
from dynamical instability, with no tanh fallback.

The notebook compares a reference droplet with an equal-mass shape perturbation
using float64 Euler-Maruyama and shared conserved noise on CUDA or CPU.
Defaults include deterministic controls, g/ell/radius changes, and D=0.1.
Editable cells show profile movies, radius, circularity, components, bulk values,
mass drift and paired shape-mode growth. Sparse trajectories are cached under
`results/poisson_sanity/`. No EP is calculated; D=0 is supported exactly.
Set `INITIALIZATION="local_bvp"` for a common-profile quench instead of a
stationary-branch experiment. Requires the sibling LRAMB repository and existing
NumPy/SciPy/PyTorch/Matplotlib/tqdm dependencies. The notebook has not been executed;
survival labels describe only the selected observation window.

## LRAMB Poisson interaction range

Open [demos/lramb_poisson_range.ipynb](demos/lramb_poisson_range.ipynb) for separate
generation, training, evaluation and inline plotting cells. Generation uses the
sibling `../LRAMB` solver with Euler-Maruyama integration and **every-step
midpoint medium EP**. Training observes field pairs, never EP reference labels.

For one-condition, one-fit debugging, open
[demos/lramb_single_run.ipynb](demos/lramb_single_run.ipynb). It contains editable
simulation, model, training and matplotlib cells: a burn-in/production profile
movie and steady diagnostics, live raw/EMA losses, cumulative/window EP,
time-averaged medium versus predicted maps, and separate mean/fluctuation
predicted shell spectra. It follows the selected architecture in
`CNEEP_v2/notebooks/Corr_AMB2D_ShellForce.ipynb`: width 64, depth 3, radius 3,
ELU, absolute shell inputs without nonlocal biases, AdamW lr=1e-3 / wd=0,
logical batch 2048, 5000 updates, one fit seed (3). Microbatch 128 limits GPU
activation memory. There are no plotting/training utility wrappers in its cells.
The physical defaults are the LRAMB domain-wall case, 64x64, D=0.1,
Poisson g=1 and ell=8. Its train/validation/test ensembles are 102/26/1,
with 1000/1000/5000 production transitions and disjoint noise streams derived
from one master seed. Data and checkpoints are cached under
`results/lramb_single_run/`. The sibling LRAMB package is required; CNEEP_v2
is a design reference, not a runtime import. This notebook has not been executed
as part of its creation.
Set `FIT_OUTPUT_ROOT` in the single-run settings cell to place checkpoints and
figures on another filesystem while keeping the data cache under `OUTPUT_ROOT`.
Checkpoint writes use a temporary file and atomic replacement. On a write
failure, the training cell stops with the destination/free-space details and
keeps an independent CPU payload in `checkpoint_recovery`; preserve the live
kernel to recover it. Filesystem free space does not account for user quotas.

The default `production_v3` compares **13 conditions**:
`g=(0.5,1,2) x ell=(1,2,4,8)` plus one `g=0` control. The Poisson multiplier is
`Khat(q)=exp(-ell*q)` for nonzero q and `Khat(0)=0`. The control retains local
bulk, gradient and active terms. Matching trajectory numbers share initial
conditions and noise seeds across conditions within one backend; train,
validation and test use independent seeds.

### Production v3 settings

| Setting | Value |
| --- | --- |
| Grid / solver step / saved-pair stride | 32 x 32 / 0.01 / 1 |
| Couplings / interaction lengths | g=0.5,1,2 / ell=1,2,4,8, plus one g=0 |
| Noise / conserved mean | D=0.1 / mean=0 |
| Physical parameters | a=-1, b=kappa=M=1, lambda=0.8 |
| Initial field | Periodic slab, interfaces at L/4 and 3L/4 |
| Burn / production steps per trajectory | 16,000 / 1,024 |
| Burn / production physical time | 160 / 10.24 |
| Independent train / validation / test trajectories per condition | 1,024 / 64 / 128 |
| Adjacent train pairs per condition | **1,048,576** |
| Maximum shell radius / hidden width / depth | 15 (31 x 31 stencil) / 32 / 2 |
| Logical batch / GPU microbatch | 1,024 / 128 |
| AdamW initial LR / weight decay | 1e-3 / 1e-4 |
| Iterations / fit seeds / full-validation interval | 10,000 / 3 / 250 iterations |
| LR scheduler | ReduceLROnPlateau on full validation loss |
| Plateau factor / patience / cooldown | 0.5 / 3 events / 1 event |
| Absolute improvement threshold / minimum LR / warmup | 1e-3 / 1e-5 / 0 |
| Movie trajectories / averaging window | First test trajectory / 16 observed intervals |

The initial bulk values are `+/-sqrt(-a/b)` (here +/-1), with diffuse interfaces
and exactly conserved mean zero. The wall is initialized **before burn-in**;
it can relax or disappear before production. Inspect production fields rather
than assuming the wall survives. Adjacent pairs remain correlated; more
independent trajectories improve ensemble coverage, but production time 10.24
limits slow-mode conclusions.

Plateau scheduling retains its current LR while validation improves. It uses an
**absolute** improvement threshold because the alpha loss is negative. PyTorch
`patience=3` reduces LR after **four** consecutive insufficient improvements;
cooldown suppresses counting for one event. The scheduler sees each complete
validation once and is independent of gradient microbatches. Checkpoint history
stores LR used by the update and LR after validation. Batch 1024 uses eight
microbatches of 128; `--train-micro-batch-size 1024` needs sufficient VRAM.

New output: `KNEEP/results/lramb_poisson_production_v3/`.
Large v3 generation and training have not been run in this workspace.

### Run stages independently or together

From the parent workspace:

```bash
python -m pip install -r KNEEP/requirements.txt
python KNEEP/scripts/lramb_poisson_range.py all --preset production_v3 --num-gpus 4 --cpu-threads 1

python KNEEP/scripts/lramb_poisson_range.py generate --preset production_v3 --num-gpus 4
python KNEEP/scripts/lramb_poisson_range.py train --preset production_v3 --num-gpus 4
python KNEEP/scripts/lramb_poisson_range.py evaluate --preset production_v3 --num-gpus 4

# CPU: cached results only, no training or inference.
python KNEEP/scripts/lramb_poisson_range.py plot --preset production_v3
# Export only cached profile, medium-EP and KNEEP-EP movies:
python KNEEP/scripts/lramb_poisson_range.py movies --preset production_v3
```

Only generation needs LRAMB (`--lramb-root PATH` overrides its location).
Compatible completed trajectory shards and fits are reused. Incompatible settings
need a new `--output-dir` or explicit `--force`. An interrupted unfinished fit
repeat restarts; completed repeats remain available.

Explicit physical and optimizer options:

```bash
python KNEEP/scripts/lramb_poisson_range.py all --preset production_v3 --num-gpus 4 \
  --strengths 0.5 1 2 --ells 1 2 4 8 --noise 0.1 --initial-mode domain_wall \
  --train-trajectories 1024 --validation-trajectories 64 --test-trajectories 128 \
  --production-steps 1024 --sample-every 1 --max-distance 15 \
  --train-batch-size 1024 --train-micro-batch-size 128 --iterations 10000 \
  --learning-rate 1e-3 --weight-decay 1e-4 --lr-scheduler plateau \
  --validate-every 250 --lr-plateau-factor 0.5 --lr-plateau-patience 3 \
  --lr-plateau-threshold 1e-3 --lr-plateau-cooldown 1 --lr-min 1e-5 \
  --lr-warmup-steps 0 --movie-trajectories 1 --movie-window-steps 16
```

Legacy profiles remain available. `production_v2` keeps its previous physical
parameters and cosine schedule in `KNEEP/results/lramb_poisson_production_v2_range15/`.
Other legacy profiles use `KNEEP/results/lramb_poisson_range/<preset>/`.

| Preset | Grid | Burn / production steps | Train / val / test | Fits x iterations |
| --- | --- | --- | --- | --- |
| quick | 8 x 8 | 16 / 64 | 2 / 1 / 2 | 2 x 4 |
| demo | 32 x 32 | 4,000 / 8,192 | 4 / 2 / 3 | 3 x 2,000 |
| refined | 32 x 32 | 16,000 / 8,192 | 8 / 4 / 4 | 3 x 20,000 |
| production | 32 x 32 | 16,000 / 8,192 | 128 / 4 / 4 | 3 x 10,000 |
| production_v2 | 32 x 32 | 16,000 / 8,192 | 256 / 8 / 4 | 3 x 10,000 |
| production_v3 (default) | 32 x 32 | 16,000 / 1,024 | 1,024 / 64 / 128 | 3 x 10,000 |
| reference | 64 x 64 | 40,000 / 64,000 | 4 / 2 / 3 | 5 x 5,000 |

Legacy profiles use D=0.05, mean=-0.2, g=0.35 and stripe initialization;
production/v2 also include g=0. `quick` checks execution only.
`--batch-size` changes evaluation batches; `--train-batch-size` changes training.
`--min-train-pairs` raises train trajectory count for a custom cadence.

### Figures and movies

`all` and `plot` export global kernel/performance figures and comparisons grouped
by g, with the same g=0 control in each group. The four interaction lengths are
overlaid directly. Main figures include:

- Two-dimensional signed kernel maps.
- Full-validation learning curves and recorded LR changes.
- EP rates in consecutive windows (`--window-steps 64`).
- Mean EP per observed step and KNEEP/direct-medium mean-rate ratio.
- Signed shell means and temporal variances (two columns): `ep_spectrum.png`
  uses linear mean EP; `ep_spectrum_log.png` uses signed logarithmic mean EP.
  Symlog retains negative/zero EP with a linear region within +/-1e-8;
  the radius axis stays linear including rho=0. Variance uses symlog in both.
- Overlaid Var(j_rho) and Var(j_rho)/rho^2 in separate method rows.
- Time-averaged EP maps with linear colors.

The fluctuation figure excludes rho=0 and cumulative shell sums. Temporal
variance is calculated within each trajectory. KNEEP bands show fit-seed SD
following trajectory averaging; medium bands show test-trajectory SD. These are
descriptive spreads, not confidence intervals. Signed colors preserve negative EP.

Movies show profile, full-step midpoint medium EP and predicted KNEEP EP for one
saved held-out trajectory per condition. Medium maps average every solver-step
contribution in each saved movie window; they are not endpoint reconstructions.
Predicted maps use matching observed pairs and window durations. MP4 is selected
when FFmpeg is available, otherwise GIF. Defaults:
`--movie-trajectory 0 --movie-fps 10 --movie-max-frames 64`.
`--no-movies` skips export. Retaining other medium-EP trajectories requires
setting `--movie-trajectories` during generation.

Figures go to `<output-dir>/figures/`; `--figure-dir PATH` overrides this.
The notebook constructs inline figures and a selected cached movie using
`HTML(animation.to_jshtml())`. After fresh notebook training set
`EVALUATE_NEW=True` explicitly; missing caches never silently launch training
or inference. Legacy arrays have no medium-EP movie windows but can still be
plotted numerically:

```bash
python KNEEP/scripts/lramb_poisson_range.py plot \
  --output-dir KNEEP/results/lramb_poisson_range/production --no-movies
```

Keep `dataset.json` and `training.json` beside `evaluation/*.npz`. Cached-only
loading validates historical provenance, without requiring today's source hash,
CUDA, training movies or checkpoints.

### Parallel execution, RAM and SLURM

`--num-gpus 4` assigns independent jobs to one process per visible GPU.
Generation, condition/fit-seed training and condition evaluation run in parallel.
Each fit uses one GPU; VRAM is not combined. Use `--workers` to cap concurrency
or `--device cpu --workers 4` for CPU workers.

For small-grid generation, the CPU can spend most of its time launching tiny
CUDA kernels. An optional burn-in graph runs up to 100 full Euler-Maruyama steps
per replay (64 suggested), reusing the graph across trajectories in each worker:

```bash
# From the KNEEP directory, inside an activated Python environment / tmux:
python -u scripts/lramb_poisson_range.py all --preset production_v3 \
  --num-gpus 2 --cpu-threads 1 --cuda-graph-steps 64 \
  --output-dir results/lramb_poisson_production_v3_graph64 \
  2>&1 | tee production_v3_graph64.log
```

The option defaults to 0 (eager), requires CUDA graph support including custom
generator registration, and is included in dataset provenance. Every burn-in
step still computes its own noise, drift and canonical FFT. The custom generator
advances on every replay; warmup does not consume the trajectory RNG stream.
Production retains every-step midpoint EP. Saved fields are staged in a bounded
64-frame device buffer before copying to CPU. Each generated trajectory reports
elapsed seconds and steps/s, including reference checks and writing the shard;
the first trajectory on each worker also includes graph warmup/capture overhead.
Compare subsequent timings rather than GPU utilization alone.

The graph path has not been executed or benchmarked on CUDA in the development
environment. Capture errors propagate; restart with `--cuda-graph-steps 0` in a
separate output directory if the installed CUDA/PyTorch stack cannot capture it.
Generator source changes invalidate older generation signatures. Use a new
output directory for this revision; keep existing shards rather than using
`--force` on a running/previous dataset. Existing completed datasets remain
loadable for training/evaluation. Do not launch old and new generators into the
same output directory.

Trajectory shards use `torch.load(..., mmap=True)` and pairs are fetched without
copying the full dataset to RAM/GPU. V3 float32 field movies occupy about
**4.30 GB per train condition**, **55.9 GB for all 13 train conditions**, or
**66.4 GB including val/test**. References, diagnostics, movie windows and
checkpoints add storage. OS page cache counts toward RAM, so mapping cannot
promise a fixed peak. The default mapping cache remains 128. An optional
`--shard-cache-size 2` limits retained mappings but can cause substantial file
reopening and metadata I/O for random batches across 1,024 trajectories.
SLURM `--mem` requests host RAM, not GPU VRAM.

Submit [scripts/lramb_poisson_range.slurm](scripts/lramb_poisson_range.slurm):

```bash
cd /pds/pds1/slurm_workspace/idh1203/KNEEP
sbatch scripts/lramb_poisson_range.slurm
# Check measured peak RAM before adopting a lower request:
sbatch --mem=16G scripts/lramb_poisson_range.slurm --shard-cache-size 2
```

The script requests four GPUs/four CPUs, excludes gpu1, sets the supplied cache,
and runs `all --preset production_v3`. Python defaults to
`/pds/pds0/home/idh1203/anaconda3/envs/kneep/bin/python`; set `KNEEP_PYTHON` before
submission to override it. The script checks the executable before `srun`.
No job has been submitted from this workspace.

### Interpretation and numerical checks

The reference is **medium entropy**, excluding stochastic system entropy.
KNEEP estimates observed-pair irreversibility. Their finite-time maps and shell
fluctuations have different boundary terms and force gauges: direct midpoint
medium EP is not exact stationary total EP. At stationarity mean system-entropy
boundary increments vanish, so a persistent mean-rate gap cannot be assigned
solely to that boundary.

Additive Chebyshev-shell branches are architecture-dependent, not unique
physical interactions. General center-neighbor products may be unrepresentable.
Radius 15 on n=32 excludes the opposite radius-16 row/column; physical reference
shells include the entire lattice. Each conservative quadratic shell is a
free-energy boundary difference. Increasing ell at fixed g also attenuates every
nonzero Fourier mode. Monotonic EP/fluctuation trends are not assumed. Check
structural drift, timestep sensitivity and independent data replicates before
claiming stationary scaling or exact recovery. See
[Otsubo et al. (2022)](https://www.nature.com/articles/s42005-021-00787-x) and
[Kwon and Baek (2024)](https://doi.org/10.1103/PhysRevE.109.014143)
for estimator assumptions.

Generator revision: `canonical-fourier-v4-strengths-wall-movies`.
FFT(real field) is recomputed after every Euler step, supplying a consistent
spectrum for the next drift, midpoint entropy and shell endpoint energies.
Kernel-shell sums are checked per trajectory before finalizing the split.
Reported CUDA assertion failures were not reproduced locally; tolerances were
not relaxed. New physical settings require a fresh v3 directory.
