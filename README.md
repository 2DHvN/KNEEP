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
