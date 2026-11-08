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

## LRAMB Poisson interaction range

Open [demos/lramb_poisson_range.ipynb](demos/lramb_poisson_range.ipynb) for a
Korean-language experiment with separate dataset generation, training and held-out
evaluation cells. Figures are constructed with Matplotlib in notebook cells;
evaluation saves numerical arrays and checkpoints, without rendering saved images.

The experiment uses the sibling `../LRAMB/lramb/model.py` solver and follows
`LRAMB/poisson_ep.ipynb` / `lramb/range_sweep.py`: fresh dynamics for
`ell=(1,2,4,8)`, `Khat(q)=exp(-ell*q)`, `Khat(0)=0`, fixed `g=0.35`, noisy stripe
initialization, Euler–Maruyama dynamics and full-step midpoint entropy. Each
trajectory seed is shared across ell values; train, validation and test seeds
are independent. The default `production` profile also runs one **g=0
(no long-range)** control, retaining the local bulk, gradient and active terms.
No supervised EP labels are used for training.

From the parent workspace, run the three stages independently:

```bash
python -m pip install -r KNEEP/requirements.txt
python KNEEP/scripts/lramb_poisson_range.py generate --preset production --num-gpus 4
python KNEEP/scripts/lramb_poisson_range.py train --preset production --num-gpus 4
python KNEEP/scripts/lramb_poisson_range.py evaluate --preset production --num-gpus 4
# Optional standalone Matplotlib display:
python KNEEP/scripts/lramb_poisson_range.py plot --preset production
# Or run generation, training and evaluation in sequence:
python KNEEP/scripts/lramb_poisson_range.py all --preset production --num-gpus 4 --cpu-threads 1
```

Only generation requires LRAMB (use `--lramb-root PATH` for another location).
Training and evaluation operate entirely on saved datasets. The notebook also
requires a Jupyter kernel (`python -m pip install jupyterlab ipykernel`). Outputs
are under `KNEEP/results/lramb_poisson_range/<preset>/`. Matching completed runs
are reused; incompatible settings require a new `--output-dir` or explicit
`--force`. A training interruption restarts the unfinished repeat and keeps
completed repeats.

| Preset | Grid | Burn / production steps | Saved-pair stride | Fits per ell |
| --- | --- | --- | --- | --- |
| `quick` | 8 × 8 | 16 / 64 | 1 | 2 × 4 iterations |
| `demo` | 32 × 32 | 4,000 / 8,192 | 8 | 3 × 2,000 iterations |
| `refined` | 32 × 32 | 16,000 / 8,192 | 1 | 3 × 20,000 iterations |
| `production` (default) | 32 × 32 | 16,000 / 8,192 | 1 | 3 × 10,000 iterations |
| `reference` | 64 × 64 | 40,000 / 64,000 | 40 | 5 × 5,000 iterations |

`quick` checks execution only. `demo` makes the complete workflow accessible;
`reference` matches LRAMB's simulation durations and lattice, and can require
substantial runtime. All use `dt=0.01`, `lambda=0.8`, `D=0.05`, `a=-1`, `b=kappa=1`.
The notebook exposes the physical parameters and run lengths through dataclass
configuration. CLI options include `--ells`, `--sample-every`, `--burn-steps`,
`--production-steps`, `--iterations`, `--repeats`, `--max-distance`,
`--train-trajectories`, `--validation-trajectories`, `--test-trajectories`,
`--include-no-interaction` and `--no-include-no-interaction`.
The default stage is `all`, preset is `production`, and device is `auto`.
`--batch-size` sets evaluation batches only; use `--train-batch-size` for
training. `--min-train-pairs 1000000` can increase the trajectory count for
custom observation protocols.

Plots include cumulative EP in time and by receiving shell radius, time averaged
EP maps, signed shell EP spectra with training-seed variation, temporal shell-rate
variance, temporal PSD and the global EP response to ell. All temporal spectra
are computed within individual held-out trajectories. Global variance/PSD sum
signed branches before calculating statistics, preserving their covariance.
Map contributions sum to the reported global rate; LRAMB density maps are
integrated with `dx**2`. Kernel diagnostics show attenuation and the field state.

Interpret the reference as **medium entropy**, while unsupervised KNEEP learns
sampled path irreversibility, including a system-entropy boundary term. Their
finite-time maps, cumulative curves and shell fluctuations need not coincide.
The full-step reference retains sub-sampling dynamics; saved endpoint pairs can
lose entropy. Regenerate at a smaller `sample_every` to check this effect.
KNEEP's learned Chebyshev shell contributions are architecture-dependent, and are
compared with a physical kernel decomposition on the same shell geometry. They
are not uniquely identifiable physical interactions. Conservative kernel EP is
a free-energy boundary difference, so its stationary long-time mean vanishes.
The production fitted radius is 12 lattice sites; inspect larger radii for omitted
tails. Wider Poisson kernels also attenuate nonzero Fourier modes at fixed g,
so a monotonic EP increase is not assumed. Training-seed spread measures fitting
variation conditional on fixed data; repeat the data sweep with independent
seeds and inspect structural drift before making a stationary or significance
claim.

### Production run: million-pair datasets and a g=0 control

The notebook and CLI now select `production`. The new large simulation and fits
have not been run locally. Results go to
`KNEEP/results/lramb_poisson_range/production/`; legacy profiles remain available.

| Setting | `production` |
| --- | --- |
| Observation stride / interval | 1 step / 0.01 |
| Burn time / production time per trajectory | 160 / 81.92 |
| Independent train / validation / test trajectories per condition | 128 / 4 / 4 |
| Train pairs per condition | 128 × 8,192 = **1,048,576** |
| Maximum learned shell radius | **12** (outer spatial footprint 25 × 25) |
| Logical training batch | **1,024** |
| Default GPU micro-batch | 128; gradients accumulated across 8 chunks |
| AdamW learning rate / weight decay | **1e-3 / 1e-4** |
| Optimizer iterations / independent fits | **10,000 / 3** |
| Full validation every | 500 iterations |

Width 32, depth 2, ELU and alpha=-0.5 remain unchanged. Radius 12 means the
maximum Chebyshev shell distance, matching the model's `max_distance` setting.
The logical batch is sampled once and divided into GPU micro-batches, with
the same objective scaling and one optimizer step per 1,024 pairs. Set
`--train-micro-batch-size 1024` to process the entire batch at once if GPU memory
allows. The optimizer and architecture can also be set explicitly:

```bash
python KNEEP/scripts/lramb_poisson_range.py all --preset production --num-gpus 4 \
  --max-distance 12 --train-batch-size 1024 --train-micro-batch-size 128 \
  --learning-rate 1e-3 --weight-decay 1e-4 --iterations 10000 --cpu-threads 1
```

Each trajectory is saved as a `.pt` float32 movie shard and loaded through
`torch.load(..., mmap=True)`; split JSON indexes identify the shards. Training
reads selected pairs rather than copying the full million-pair
dataset into RAM or GPU memory. Train movies use approximately **4.30 GB per
condition**, **21.5 GB for five conditions**, or **22.8 GB including validation
and test**, plus smaller metadata, reference arrays and checkpoints. Consecutive
pairs are temporally correlated: one million pairs is not one million
independent observations.

The additional condition is saved as `no_interaction`, with metadata `ell=0`
and effective coupling strength 0. This is a control identifier, **not** a
physical zero-range Poisson kernel. It is generated and fitted once, with the
same split seeds and protocol. Plot labels say `g=0 (no long-range)` and do not
treat the control as a point on a logarithmic range axis. Its kernel is exactly
zero, so a normalized kernel tail is undefined and is omitted.

### Parallel execution and SLURM

`--num-gpus 4` starts one experiment worker on each of the first four visible
GPUs. Trajectory generation jobs, independent condition/fit-seed runs, and condition evaluations are distributed
across workers; each training model uses one GPU. This does not combine GPU
memory or use distributed gradient synchronization. `--workers` can limit
concurrency, and CPU-only jobs use e.g. `--device cpu --workers 4`.
`--cpu-threads 1` keeps each worker's Torch thread count within the requested
CPU allocation. All stages reuse compatible completed outputs on rerun.
`--shard-cache-size 128` controls the maximum memory-mapped trajectory files
retained by each training worker; it does not preload their movie contents.

The notebook cells run each stage sequentially in the notebook process. For a
large sweep, run the CLI or submit the provided
[scripts/lramb_poisson_range.slurm](scripts/lramb_poisson_range.slurm), then open
the notebook with the same output directory to inspect the results:

```bash
cd /pds/pds1/slurm_workspace/idh1203/KNEEP
sbatch scripts/lramb_poisson_range.slurm
```

The SLURM file requests one task, four GPUs, four CPUs and 32 GB RAM, excludes
`gpu1`, and uses the Python and Matplotlib cache paths supplied for the server.
It runs `all --preset production --num-gpus 4`; it does not launch plotting or
require an interactive display. The job script has been written but not
submitted from this workspace.

At stationarity the mean system-entropy boundary increment vanishes. A sustained
mean-rate gap therefore cannot be assigned solely to system entropy. Finite
observation intervals, nonstationarity, discrete-time errors and the restricted
force family also matter. Each nonlocal branch sees only its own shell, so the
additive branches cannot in general express center-neighbor products in the
AMB force. The production settings do not resolve that architectural restriction
or guarantee agreement with the medium reference. For the inference and
sampling assumptions, see [Otsubo et al. (2022)](https://www.nature.com/articles/s42005-021-00787-x)
and [Kwon and Baek (2024)](https://doi.org/10.1103/PhysRevE.109.014143).

### Kernel-shell assertion after a long burn-in

The original Hermitian projection change did not resolve the reported remote
CUDA failure; the exact CUDA cause has not been reproduced locally.
`canonical-fourier-v2` introduced explicit recomputation of FFT(real field)
after **every** Euler step in both its NumPy and Torch paths. That recomputed
spectrum is used for the next drift, midpoint entropy and endpoint shell
energies. This enforces field/spectrum consistency within KNEEP independently
of the imported LRAMB step implementation. It costs an additional forward FFT
per solver step. The integrator is still Euler–Maruyama and entropy still uses
the midpoint of each solver step; the assertion tolerances are unchanged.

The current sharded generator retains that calculation and uses revision
`canonical-fourier-v3-shards`. Update `KNEEP/models/lramb_poisson.py`, restart the
notebook kernel and confirm that generation prints this revision with the actual module
path, LRAMB path, backend and Torch version. The new generator checks the
kernel-only shell sum against the kernel midpoint EP after **each trajectory**,
before computing the rest of the split. A failure reports ell, split, seed,
interval, both kernel values and the largest difference between the external
solver's propagated spectrum and FFT(real field). The latter is also saved as
`diagnostics["fourier_state_error"]` in successful datasets.

Regenerate into a fresh directory, for example by setting the following in
the configuration cell:

```python
OUTPUT_DIR = KNEEP_ROOT / "results" / "lramb_poisson_range" / "production_v3_shards"
```

Existing caches have different solver/generator signatures and are not reused.
The fix has not been exercised on the reporting remote CUDA environment.
