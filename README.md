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

### Predicted and active-only EPR versus g (default)

`lramb_poisson_range.py` now defaults to `--preset production`:

- g = -0.2, -0.1, -0.04, -0.02, 0, 0.02, 0.04; ell = 2.
- dt = 0.01, burn-in = 2,000,000 steps (physical time 20,000).
- n=64, dx=1, a=-0.25, b=0.25, kappa=1, lambda=1, zeta=0, D=0.1.
- Initial field: radius-16 tanh droplet, with its geometric mean conserved
  (about -0.5972084); no shift to the config's random/stripe `mean` parameter.
- Per g: 32 train, 4 validation, 5 test trajectories; 8192 production steps
  (time 81.92), every adjacent pair retained. These defaults retain about
  36 GiB of raw field data over seven conditions, before other artifacts.
- Five independent fits per g; 10000 updates, logical batch 8192,
  microbatch 1024, hidden channels 64, two hidden layers, no LR scheduler,
  branch radius 6 (7 branches, radii 0 through 6). EP is inferred from the existing
  unsupervised objective; reference EP is used only for evaluation.

Run from KNEEP (or use the updated four-GPU SLURM script):

```bash
python scripts/lramb_poisson_range.py training --num-gpus 4
python scripts/lramb_poisson_range.py plot --num-gpus 4
sbatch scripts/lramb_poisson_range.slurm training
```

Workers train independent models, starting different g conditions first.
With one GPU use `--device cuda --workers 2` to run two concurrent processes
if VRAM permits; each has its own model and optimizer. `--device cuda`
alone uses one worker, scheduling conditions sequentially. CPU parallelism
uses `--device cpu --workers 2`. CUDA burn-in automatically uses graph chunks
of 20 steps for this preset; `--cuda-graph-steps 0` disables it.

The new default output is `results/lramb_poisson_g_steady`. Figures are:

- Per condition: `loss.png`, `ep_maps.png`, `branch_ep_average.png`,
  `branch_ep_fluctuation.png`, `branch_ep_fluctuation_per_pixel.png`.
- Across g: `branch_fluctuation_overlay.png`,
  `branch_fluctuation_overlay_normalized.png`, `epr_vs_g.png`,
  `pred_true_ratio_vs_g.png`.
- Numeric values: `branch_statistics.csv`, `epr_vs_g.csv`, and definitions
  in `definitions.json`.
- Test trajectory 0: separate profile, predicted-EP and active-only-EP
  MP4 videos (GIF fallback without ffmpeg). EP frames average 100 solver
  steps; the export selects at most 64 evenly spaced cached frames.

Active-only reference maps are accumulated at every float64 solver step,
including steps between saved observations. All maps use EP/time/site.
Predicted and active maps share a color scale; `--include-medium-map` adds
a medium map with a separate scale. Maps average time, test trajectories,
and (for prediction) independent fits.

Branch fluctuation is the temporal variance of saved-interval EPR *within*
each trajectory, then averaged over trajectories/fits. Pixel correction
divides by the square of the actual branch force-input pixel count. The
normalized overlay divides variance by the square of the ensemble mean
total predicted EPR. This equals the variance of the normalized rate;
near-zero denominators produce NaN. Both overlays include raw and per-pixel
panels. Fluctuations depend on the observation interval.

EPR error bars are SEM across test-trajectory means after averaging fits;
fit-seed SEM is reported separately in CSV. The ratio is a ratio of ensemble
means, not an average of noisy individual ratios. The active-only rate is
the primary reference; no system-entropy correction is added.

Stages are separate: `training` (the default) generates/reuses data and fits,
without evaluation or rendering; `train` only fits an existing dataset.
`plot` evaluates the latest saved best models (reusing matching evaluation
arrays), then writes figures and movies, without simulation or training.
`all` remains available for the entire pipeline. Use the same `--output-dir`
for every stage. To render without simulation/training:

```bash
python scripts/lramb_poisson_range.py plot --no-movies --include-medium-map
python scripts/lramb_poisson_range.py movies
```

Extend an existing fit to a **total** of 20,000 optimizer updates:

```bash
python scripts/lramb_poisson_range.py resume --iterations 20000 --num-gpus 4
python scripts/lramb_poisson_range.py plot --num-gpus 4
# SLURM equivalents (submit plot after resume finishes):
sbatch scripts/lramb_poisson_range.slurm resume --iterations 20000
sbatch scripts/lramb_poisson_range.slurm plot
```

`resume` loads the original fit configuration from training.json, so the
original preset/model settings need not be repeated. Only the iteration
target may change; data, branch radius, batch size, repeat count, and optimizer
settings remain fixed. No trajectories are regenerated. Without --iterations,
resume continues an interrupted run toward its saved target. Completed repeats
are reused, unfinished repeats resume from their latest validation checkpoint,
and repeats that never started begin normally.

New `repeat_*.last.pt` checkpoints save the last model, AdamW state, plateau
scheduler, sampling/global RNG states, normalization, full loss history, and
best model at each validation event. Evaluation continues to use the best
validation model across the entire training history. Cosine extension retains
its original schedule horizon and stays at lr_min beyond that horizon.

Historical best-only checkpoints cannot recover optimizer/RNG state that was
never saved: resume automatically continues from their best weights with a
fresh optimizer and RNG, and logs `legacy best-weight continuation`. This is
further fitting, not an exact restoration of the historical trajectory.

Old datasets without `active_rate_maps` can still produce static active
maps, but need fresh generation to produce exact full-step active movies.
Completed stage artifacts are reused when signatures match. Burn-in checkpoints are saved at completion of
each condition, not periodically inside its 2-million-step burn.

Small end-to-end validation:

```bash
python scripts/lramb_poisson_range.py all --preset quick --g -0.02 0 0.02 --ells 2 --movie-trajectories 1 --movie-window-steps 8 --movie-max-frames 3 --device cpu --workers 2 --output-dir results/g_fit_smoke
python -m unittest discover -s tests -p test_poisson_g_sweep.py -v
```

The CLI presets are `production`, `quick`, `demo`, and `reference`.
Current production uses max_distance=6. Existing result files are retained;
resume reads the original fit configuration from the saved training manifest
and is independent of subsequent preset edits. To change model architecture
rather than extend training, use a fresh output directory.

### Batched stationary g sweep (no fitting)

From the KNEEP directory:

```bash
python scripts/lramb_g_steady.py --device auto
```

This uses the sibling LRAMB Poisson solver at fixed `--ell 2`, with
`--burn-time 20000`, `--dt 0.01` (2,000,000 burn steps), and
`--production-time 1000`. Other defaults are g=[-0.02,-0.01,0,0.01,0.02], 5 replicas per g,
n=64, a=-0.25, b=0.25, kappa=1, D=0.1, lambda=1, and zeta=0.
Initialization is a centered tanh droplet with radius 16, amplitude
sqrt(-a/b), and width sqrt(2*kappa/(-a)), matching poisson_sanity.
There is no mean shift: the geometric initial mean is conserved and recorded
as resolved_mean. All replicas start from the same droplet and receive
independent noise. Use --radius to change droplet mass, or --initial
random/stripe with optional --mean (default -0.2). initial_fields.npy stores
the actual starting fields. Diagnostics are recorded every 10 steps
(time 0.1), movies every 100 steps (time 1). All g/replica combinations
are evolved in one float64 tensor batch with independent noise entries.
CUDA uses graph replay (`--graph-steps 20`, 0 disables); CPU uses eager
PyTorch. Actual graph length divides both sampling cadences and phase
lengths and is recorded in config.json. Larger batches need more memory.

Outputs under `results/g_steady`:

- `g_*/traj_*/field_fluctuation.png`: spatial field variance from t=0,
  with the burn-in boundary marked. Inspect its plateau; no automatic
  stationary-state claim is made from elapsed time alone.
- `g_*/traj_*/steady_state.mp4`: production-window field movie for every
  replica. MP4 requires installed ffmpeg; use `--movie-format gif` if unavailable.
- `ensemble_epr_vs_g.png` and CSV: spatially integrated active-only EPR,
  computed at **every solver step**, regardless of movie/diagnostic cadence.
  Each replica's EP/time is averaged with equal weight; error bars are SEM
  across independent replica means, not across correlated time samples.
  With one replica SEM is undefined (NaN in CSV, no error bar).
- `ensemble_ep_vs_g.png` and CSV: cumulative production-window active EP.
- `cluster_size_vs_g.png` and CSV: replica/time-averaged cluster area.
  A cluster is a four-connected `phi > --cluster-threshold` component
  (default 0), merging periodic boundaries. The per-frame number-average
  area is averaged in time by trapezoidal integration. No positive-phase
  sites means area 0. This measures area, not radius or spectral wavelength.
- `fields.npy`, `observables.npz`, `config.json`: sampled movie fields,
  full burn/production variance trace, cumulative production EP, mass errors,
  and resolved settings. Burn-in fields are not stored as movies.

Active-only increments are `-sum(mu_active_eff(phi_mid)*dphi)*dx^2/D`;
there is no system-entropy term or conservative boundary term. "Total"
here means integrated over the entire spatial grid, not EP per unit area.
The variance diagnostic is necessary evidence to inspect, not proof that
slow morphology has equilibrated; compare independent replicas as well.

Use a fresh `--output` for each experiment. Plotting can be rerun without
simulation using `--plot-only --output results/g_steady`. The production
run is long; a CPU smoke check is:

```bash
python scripts/lramb_g_steady.py --g 0 0.5 --trajectories 2 --n 8 --radius 2 --movie-format gif --burn-time 0.04 --production-time 0.06 --diagnostic-every 2 --movie-every 2 --device cpu --output results/g_steady_smoke
```
