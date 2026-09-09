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

Outputs are written under `results/saou_perform/`: the 30 fixed trajectory
files, 75 trained-model checkpoints, run/summary/kernel CSV files, unsmoothed
loss histories and plots, `figures/a2_delta_s.png`, and 15 condition-wise
kernel-decomposition figures. The performance figure uses boxes and individual
points for the five training seeds; exact one-step theory is shown by solid
lines. Completed data and checkpoints are reused when the command is restarted.
Use one controller process per results directory; multi-GPU work is handled by
`--num-gpus`.

The notebook-sized float32 trajectories occupy about 115.6 GiB in total
(7.7 GiB per condition); an atomic train-data save temporarily needs another
7.6 GiB. Each active GPU worker also holds roughly one condition's data in CPU
memory (about 31 GiB for four workers). The plotted target is the exact
stationary ensemble EPR of one `dt=1e-2` Euler transition, evaluated
analytically in Fourier space rather than from a sampled trajectory. The
continuous-time `sigma * dt` reference remains available in the CSV files as
`continuous_delta_s`; it is about 0.52--0.99% smaller on this grid and is not
used as the plotted ground truth.
