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

## LRAMB numerical implementation

All LRAMB datasets, single-run demos/scripts, and Poisson sanity paths use
the sibling LRAMB package's Tjhung real-space stencils: Lap9 conservative
stiffness, eighth-order centered passive/noise currents, and isotropic 3x3
active currents. Local time stepping uses Euler?Maruyama and Torch
convolutions (or SciPy on the NumPy backend). FFTs serve long-range kernels
and diagnostics, including the effective active EP potential. See
[LRAMB numerical details](../LRAMB/README.md#real-space-amb-discretization).

`zeta` defaults to zero; it can enable the extra AMB+ current. On even grids,
the centered derivatives conserve the mean and three checkerboard modes.
Stationary initialization and active EP pseudoinverses handle all four.
The active EP map uses the zero-null-mode pseudoinverse convention.
The generator revision and source fingerprints have changed: older
trajectories/checkpoints belong to the former discretization and are not
reused under the new signature. Regenerate data before retraining.

## Poisson circular-cluster sanity notebook

Open [demos/poisson_sanity.ipynb](demos/poisson_sanity.ipynb) for editable
64x64 circular-droplet experiments matching the Corr_AMB2D_ShellForce setup:
`a=-0.25, b=0.25, kappa=4, M=1, lambda=10, zeta=0, D=0.1, g=0, ell=1`,
with radius 16 and `dt=0.001`. The default initialization is the analytic
tanh profile. `local_bvp` solves a radial local-passive boundary-value problem;
`periodic_stationary` continues that field to zero drift of the periodic
Poisson+active stencil equation, preserving all D8 conserved modes.
Failed BVP/root initialization is reported with no silent fallback.

Each condition evolves one field using float64 Euler-Maruyama and conserved
noise on CUDA or CPU. `RUN["burn_in"]` sets the equilibration time, followed by
`RUN["duration"]` of production. Defaults are 10 and 4.999 physical time units, respectively.
Editable cells show x/y averaged-profile heatmaps, axial Fourier mode powers
and amplitudes, field snapshots and a field/profile movie. Steady-state
diagnostics cover initialization, the entire burn-in, and production, with
the production boundary marked. They include mass, variance, conservative
energy, structure mean q, drift RMS and interval EP rates, plus production
EP block means. Extending burn-in may be necessary when structural drift remains.
For D>0, midpoint medium EP is accumulated at every solver step; time-averaged
local conservative, Poisson, active and total EP maps include production only.
Their spatial integrals are checked against production entropy increments.
Deterministic D=0 controls retain field/mode diagnostics; entropy divided by
D is undefined for them. The default D=0.1 case supplies the EP maps. Set BASE D
positive to compare EP across all controls. Numerical arrays are cached under
`results/poisson_sanity/`; figures are built inline.

The standalone equivalent is `scripts/poisson_sanity.py` (no notebook runtime).
Run `python scripts/poisson_sanity.py --device cuda:0` to generate/reuse trajectories,
save diagnostic PNGs, and export an HTML movie under the output directory.
Use `--movie mp4` with ffmpeg, or `--movie none` to skip video export.
`--output-dir PATH` selects the destination; `--config settings.json` overrides
`GRID`, `BASE`, `RUN`, `BVP`, `ROOT`, `RULES`, `CASES`, `INITIALIZATION`,
`REMOVE_ZERO_MODE`, or `MODE_MAX`. CASES entries override BASE per condition.
Both versions end with a field movie and the central cut at fixed x=L/2:
`phi(y) = fields[frame, :, n//2]`, with the cut marked on the field panel.
Set `INITIALIZATION="local_bvp"` for a common-profile quench instead of a
stationary-branch experiment. Requires the sibling LRAMB repository and existing
NumPy/SciPy/PyTorch/Matplotlib/tqdm dependencies. Short-run validation covers the
new burn-in/production bookkeeping and plots; default-duration stationarity
must be assessed from the diagnostics.

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
`CNEEP_v2/notebooks/Corr_AMB2D_ShellForce.ipynb`, with width 64, hidden depth 3,
radius 3, ELU, absolute shell inputs without nonlocal biases, AdamW lr=1e-3 /
wd=0, logical batch 2048, 5000 updates, and fit seed 3. Microbatch 128 limits
training activation memory; validation batch is 2048 and prediction batch is 256.
Both the notebook and script default to 64x64, dx=1, a=-0.25, b=0.25,
kappa=4, mobility=1, lambda=10, D=0.1, g=0, dt=0.001, and tanh radius 16.
CNEEP uses -a*phi while KNEEP uses a*phi, hence the sign change.
Train/validation/test ensembles are 102/26/1, with 999/999/4999 transitions
(1000/1000/5000 frames), and 10,000 burn-in steps (physical time 10).
Burn-in combines all 129 members into **one batch**. `DATA["ensemble_batch"]`
controls production batches only. Each member receives independent noise;
the first test member's complete burn/production diagnostic trace is retained.
The completed terminal fields are saved in `burn_state.pt` and reused after
an interrupted generation. Burn and production use separate random streams.
KNEEP retains its spatial discretization, noise streams, and EP estimators;
matching parameters does not imply identical numerical trajectories.

`INITIALIZATION = "tanh"` uses
`sqrt(-a/b) * tanh((radius-r) / sqrt(2*kappa/(-a)))`, with radius and width in
physical length units, preserving the sampled mean before burn-in.
`local_bvp` selects the radial Neumann/mass-constrained passive solution;
`periodic_stationary` additionally performs periodic stationary continuation.
The script also accepts `{"INITIALIZATION": "tanh"}` through `--config`.
Solver residuals and mass drift are diagnostic; nonfinite states stop execution.
The initial-field hash and initializer settings enter the new data cache key.
Data and checkpoints are cached under
`results/lramb_single_run/`. The sibling LRAMB package is required; CNEEP_v2
is a design reference, not a runtime import. This notebook has not been executed
as part of its creation.
Set `FIT_OUTPUT_ROOT` in the single-run settings cell to place checkpoints and
figures on another filesystem while keeping the data cache under `OUTPUT_ROOT`.
Checkpoint writes use a temporary file and atomic replacement. On a write
failure, the training cell stops with the destination/free-space details and
keeps an independent CPU payload in `checkpoint_recovery`; preserve the live
kernel to recover it. Filesystem free space does not account for user quotas.

### Single-run script (Linux / tmux)

[scripts/lramb_single_run.py](scripts/lramb_single_run.py) is a standalone
version of the current single-run notebook. It contains the solver, initializer,
model, training and plotting code; it does not execute notebook cells or require
Jupyter/IPython. Defaults match the notebook settings above. Run from `KNEEP/`
with your activated Python environment:

```bash
tmux new -s lramb-single
python -u scripts/lramb_single_run.py --device cuda:0 2>&1 | tee single_run.log
```

Detach with `Ctrl-b`, then `d`; reconnect with `tmux attach -t lramb-single`.
Use `--device cuda:1` to select another GPU, or `--device cpu` for CPU execution.
Each invocation runs one experiment on one selected device.

```bash
# Continue an interrupted fit from its matching last.pt:
python -u scripts/lramb_single_run.py --device cuda:0 --resume

# Generate/cache data and save profile/steady diagnostics only:
python -u scripts/lramb_single_run.py --device cuda:0 --stage generate

# Re-evaluate a saved model and regenerate figures, without training:
python -u scripts/lramb_single_run.py --device cuda:0 --stage evaluate
```

Default output is `results/lramb_single_run/data_<hash>/fit_<hash>/`.
The data cache and checkpoint signatures follow the notebook's conventions;
matching completed data are reused. `--resume` restores model, optimizer,
scheduler, sampler and history at the last saved iteration (normally every 100
updates), up to the configured iteration limit. Keep the same settings when
resuming: changing training settings, including the iteration limit, selects a
different fit directory. A completed fit proceeds to evaluation with `--resume`.
Incomplete production generation restarts from the saved joint burn endpoint;
partial burn-in or production loops are not resumed mid-step.

Training logs include sampled train loss, full-validation loss, LR and best
iteration. `training_loss.png` is refreshed at each recorded evaluation.
Both the single-run notebook and script overlay the full medium reference,
the active-chemical-potential-only reference, and KNEEP prediction in cumulative
and window-averaged EP plots. The active contribution uses
`mu_A_eff=L8^+(active_drift/M)` in `-mu_A_eff*dphi/D`, with the same midpoint and
spatial integration as full medium. Neither physical reference adds system
entropy. The time-averaged map figure has three panels with one colour scale.
`evaluation.npz` also saves `active_increment` and `active_mean_density`.

Existing data and trained checkpoints can be reused with `--stage evaluate`;
in the notebook, set `RUN_TRAINING=False` and rerun from the top. Scalar active
increments already exist in medium component index 2. New generation stores
`test_active_density.npy` directly from float64 solver states. For older caches,
only the missing active map is reconstructed from float32 cached fields, without
repeating burn-in. Its provenance is recorded in `test_active_density.json`,
evaluation metadata and the map title; its spatial integral may differ slightly
from the stored scalar reference due to field quantization.

Other outputs include `training_loss_raw.png`, `steady_diagnostics.png`,
`performance.png`, `time_averaged_medium_vs_pred.png`,
`pred_spectrum_mean.png`, `pred_spectrum_fluctuation.png`, `evaluation.npz`,
checkpoints and the resolved configuration. The initial profile is saved as
`data_<hash>/initial_profile.png`. The profile movie includes burn-in and
production: `profile.mp4` when ffmpeg is available, otherwise `profile.html`.
Use `--movie html`, `--movie mp4`, or `--movie none` to select explicitly.

Edit the settings dictionaries in the script or provide a JSON override:

```json
{"TRAIN": {"microbatch": 64}, "DATA": {"ensemble_batch": 128}}
```

```bash
python -u scripts/lramb_single_run.py --device cuda:0 --config single_run.json
```

`--output-dir` changes the data root; `--fit-output-dir` places checkpoints and
figures on a separate filesystem; `--lramb-root` selects the LRAMB repository.
Use the same path/config options for subsequent resume/evaluate commands.
Plot blocks are directly editable in the script. The notebook and script are
separate source files: later notebook edits are not imported automatically.
The script has not been executed as part of this change.

### Single-run Slurm job

[scripts/lramb_single_run.slurm](scripts/lramb_single_run.slurm) uses the same
fixed working directory and Matplotlib cache path as the existing Poisson Slurm
job. Python comes from `KNEEP_PYTHON`, or `python` on the inherited PATH;
no conda installation path is hard-coded. It requests one GPU, four CPU cores and 16 GB RAM, with `gpu1`
excluded. Edit these paths/directives directly if needed. There is no explicit
`mkdir` or automatic repository/environment discovery.

From `KNEEP/`, activate the intended environment and export its interpreter
(the environment must also be accessible on the allocated compute node):

```bash
conda activate kneep
export KNEEP_PYTHON="$(command -v python)"
sbatch scripts/lramb_single_run.slurm
sbatch scripts/lramb_single_run.slurm --config "$PWD/my_run.json"
sbatch scripts/lramb_single_run.slurm --config "$PWD/my_run.json" --resume
sbatch scripts/lramb_single_run.slurm --config "$PWD/my_run.json" --stage evaluate
```

Use absolute config/output paths: the job changes directory to
`/pds/pds1/slurm_workspace/idh1203/KNEEP/scripts`.
`KNEEP_PYTHON` can override the default interpreter. The log is
`lramb_single_<jobid>.log` in the submission directory. Resource overrides go
before the filename, e.g. `sbatch --mem=32G scripts/lramb_single_run.slurm`.
The job has not been submitted or executed as part of this change.

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
- Signed shell means, raw temporal variances, and pixel-normalized variances
  (three columns): `ep_spectrum.png`
  uses linear mean EP; `ep_spectrum_log.png` uses signed logarithmic mean EP.
  Symlog retains negative/zero EP with a linear region within +/-1e-8;
  the radius axis stays linear including rho=0. Variance uses symlog in both.
- Overlaid Var(j_rho) and Var(j_rho)/N_rho^2 in separate method rows.
  N counts force-input pixels from each branch mask: 1, 8, 16, 24, ... .
  The physical reference uses its own periodic shell sizes.
- Time-averaged EP maps with linear colors.

The fluctuation figure includes rho=0 (N=1) and shows individual branches. Temporal
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

Generation first runs **one joint burn batch per physical condition**, combining
train, validation and test (1024+64+128 = 1216 members for production_v3).
Only current fields are resident during burn-in. Each condition saves
`data/<condition>/burn_state.pt`; production trajectory tasks then read their
own terminal field, without repeating burn-in. Conditions can burn on separate
GPUs; a barrier separates the burn phase from distributed production.
Matching completed burn checkpoints and production shards are reused on restart.
The ordered batch and separate production streams are reproducible across
conditions within a backend. Changing split counts changes the burn noise sequence.

For small-grid generation, the CPU can spend most of its time launching tiny
CUDA kernels. An optional burn-in graph runs up to 100 full Euler-Maruyama steps
per replay, operating on the entire condition's batch:

```bash
# From the KNEEP directory, inside an activated Python environment / tmux:
python -u scripts/lramb_poisson_range.py all --preset production_v3 \
  --num-gpus 2 --cpu-threads 1 --cuda-graph-steps 64 \
  --output-dir results/lramb_poisson_production_v3_graph64 \
  2>&1 | tee production_v3_graph64.log
```

The option defaults to 0 (eager), requires CUDA graph support including custom
generator registration, and is included in dataset provenance. Every burn-in
step still computes its own noise and real-space drift. A diagnostic spectrum
is constructed once at the end. The custom generator advances on every replay;
warmup does not consume the burn RNG stream.
Production retains every-step midpoint EP. Saved fields are staged in a bounded
64-frame device buffer before copying to CPU. Each generated trajectory reports
production seconds and steps/s, including reference checks and writing the shard.
Joint burn-in timing is reported separately and includes graph warmup/capture.

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
