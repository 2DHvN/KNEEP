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

The SAOU temperature experiment uses

```text
T = 0.1, 0.3, 1.0, 3.0, 10.0
(w0,a1,a2,a3,a4) = (3,0,1,1,0), (4,1,2,0,1), (2,0,1,0,1)
```

with one fixed train/test dataset per condition and 40 training seeds
(`alpha=-0.5`). The remaining settings come from the current source of
`Corr_SAOU.ipynb`. From the parent workspace, run either:

```bash
python KNEEP/scripts/saou_temperature.py
python KNEEP/scripts/saou_temperature.py --num-gpus 2
```

Outputs are written under `results/saou_temperature/`: checkpoints for all 600
models, run/summary CSV files, the temperature-performance figure, and 15
kernel-spectrum figures. Completed checkpoints are reused when the command is
restarted. Use one controller process per results directory; multi-GPU work is
handled by `--num-gpus`. The center-source comparison is in
`demos/saou_center_spin.ipynb`.

The notebook-source integrator is retained (`Euler-Maruyama`, `dt=1e-2`), while
the analytic target is the continuous-time ensemble EPR. Its expected
finite-step offset is about 1--4% for these three parameter sets.
