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

The focused SAOU performance experiment uses

```text
A = 0.1, 0.2, 0.3, 0.4, 0.5
d_w = 0.0, 0.1, 0.2
a_r = A / sqrt(r),  w_0 = sum_r(a_r) + d_w,  T = 1
```

with one fixed train/test dataset per condition and five training seeds
(`alpha=-0.5`). The remaining simulation and learning settings come from the
current `Corr_SAOU.ipynb`. From the parent workspace, run either:

```bash
python KNEEP/scripts/saou_perform.py
python KNEEP/scripts/saou_perform.py --num-gpus 2
```

To redraw completed results without loading trajectories or models:

```bash
python KNEEP/scripts/plot_saou_perform.py
```

This reads only `runs.csv` and `kernel_runs.csv`. The performance plot uses
the actual A-squared coordinates, mean and sample-standard-deviation error bars
over training seeds, and the one-step continuous theory as solid straight
lines. Kernel spectra follow the grouped-histogram style used in
`Corr_SAOU.ipynb`.

Outputs are written under `results/saou_perform/`: the 30 fixed trajectory
files, 75 trained-model checkpoints, run/summary/kernel CSV files, unsmoothed
loss histories and plots, and the derived figures. Running
`plot_saou_perform.py` replaces `figures/a2_delta_s.png` with the
publication-style error-bar version and writes 15 grouped histograms under
`figures/kernel_spectra/`. Completed data and checkpoints are reused when the
training command is restarted. Use one controller process per results
directory; multi-GPU work is handled by `--num-gpus`.

The notebook-sized float32 trajectories occupy about 115.6 GiB in total
(7.7 GiB per condition); an atomic train-data save temporarily needs another
7.6 GiB. Each active GPU worker also holds roughly one condition's data in CPU
memory (about 31 GiB for four workers). Both references are stored without
sampling a trajectory: the exact stationary Euler-transition value and the
continuous-time `sigma * dt` value. The visualization script uses the latter
as the one-step theory because it is exactly linear in (A^2), matching the
`Corr_SAOU.ipynb` convention.
