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
are independent. No supervised EP labels are used for training.

From the parent workspace, run the three stages independently:

```bash
python -m pip install -r KNEEP/requirements.txt
python KNEEP/scripts/lramb_poisson_range.py generate --preset demo
python KNEEP/scripts/lramb_poisson_range.py train --preset demo --device auto
python KNEEP/scripts/lramb_poisson_range.py evaluate --preset demo --device auto
# Optional standalone Matplotlib display:
python KNEEP/scripts/lramb_poisson_range.py plot --preset demo
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
| `refined` (notebook default) | 32 × 32 | 16,000 / 8,192 | 1 | 3 × 20,000 iterations |
| `reference` | 64 × 64 | 40,000 / 64,000 | 40 | 5 × 5,000 iterations |

`quick` checks execution only. `demo` makes the complete workflow accessible;
`reference` matches LRAMB's simulation durations and lattice, and can require
substantial runtime. All use `dt=0.01`, `lambda=0.8`, `D=0.05`, `a=-1`, `b=kappa=1`.
The notebook exposes the physical parameters and run lengths through dataclass
configuration. CLI options include `--ells`, `--sample-every`, `--burn-steps`,
`--production-steps`, `--iterations`, `--repeats` and `--max-distance`.

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
The default fitted radius is 8 lattice sites; inspect larger radii for omitted
tails. Wider Poisson kernels also attenuate nonzero Fourier modes at fixed g,
so a monotonic EP increase is not assumed. Training-seed spread measures fitting
variation conditional on fixed data; repeat the data sweep with independent
seeds and inspect structural drift before making a stationary or significance
claim.

### Next run after the 10,000-iteration comparison

The notebook now selects `refined`, a new result directory. This is a proposed
next configuration; its simulation and training have not been run. The supplied
curves show similar validation losses across fit seeds and little late training
improvement, alongside learned cumulative EP below the full-step medium
reference. The exact configuration used for those supplied figures is not saved
locally, so the changes below compare against the repository's `demo` preset.

| Setting | `demo` | `refined` |
| --- | --- | --- |
| Observation stride / interval | 8 steps / 0.08 | 1 step / 0.01 |
| Burn time / production time | 40 / 81.92 | 160 / 81.92 |
| Independent train / validation / test trajectories | 4 / 2 / 3 | 8 / 4 / 4 |
| Training batch | 32 | 128 |
| AdamW learning rate / weight decay | 1e-3 / 1e-4 | 3e-4 / 1e-5 |
| Optimizer iterations | 2,000 | 20,000 |
| Full validation every | 100 iterations | 500 iterations |

Both use the same physical parameters, grid, radius 8, width 32, depth 2,
ELU, alpha=-0.5 and three fit seeds. More frequent observations address missing
sub-sampling dynamics; larger batches and a lower learning rate address noisy
optimization. Longer burn-in and more independent trajectories improve the next
stationarity/generalization check without establishing convergence by themselves.
The larger saved dataset also makes each full validation pass more expensive.

```bash
python KNEEP/scripts/lramb_poisson_range.py generate --preset refined
python KNEEP/scripts/lramb_poisson_range.py train --preset refined --device auto
python KNEEP/scripts/lramb_poisson_range.py evaluate --preset refined --device auto
```

Results go to `KNEEP/results/lramb_poisson_range/refined/`. Reopen the notebook
kernel and run its cells in order to use the new preset. Its previous quick-run
outputs were cleared when changing the default configuration.

At stationarity the mean system-entropy boundary increment vanishes. A sustained
mean-rate gap therefore cannot be assigned solely to system entropy. Finite
observation intervals, nonstationarity, discrete-time errors and the restricted
force family also matter. Each nonlocal branch sees only its own shell, so the
additive branches cannot in general express center-neighbor products in the
AMB force. The refined settings do not resolve that architectural restriction
or guarantee agreement with the medium reference. For the inference and
sampling assumptions, see [Otsubo et al. (2022)](https://www.nature.com/articles/s42005-021-00787-x)
and [Kwon and Baek (2024)](https://doi.org/10.1103/PhysRevE.109.014143).
