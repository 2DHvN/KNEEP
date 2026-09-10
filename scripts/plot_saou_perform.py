"""Plot completed SAOU performance results without loading trajectories or models."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results" / "saou_perform"
KERNEL_LABELS = ("local", "r=1", "r=2", "r=3", "r=4")
COLORS = ("#1f77b4", "#d95f02", "#2ca02c")
MARKERS = ("o", "v", "^")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required result table is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"result table is empty: {path}")
    return rows


def _mean_std(values) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    standard_deviation = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return float(array.mean()), standard_deviation


def _pyplot(results_dir: Path):
    mpl_dir = results_dir / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 14,
            "axes.labelsize": 20,
            "axes.linewidth": 1.25,
            "legend.fontsize": 13,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    return plt


def _style_axis(axis) -> None:
    axis.tick_params(which="both", width=1.15)
    axis.tick_params(which="major", length=6)
    axis.tick_params(which="minor", length=3)


def _performance_figure(runs, figure_dir: Path, dpi: int, plt) -> None:
    grouped: dict[tuple[float, float], list[dict[str, str]]] = defaultdict(list)
    for row in runs:
        grouped[(float(row["d_w"]), float(row["amplitude_squared"]))].append(row)

    dws = sorted({key[0] for key in grouped})
    amplitude_squared = sorted({key[1] for key in grouped})
    figure, axis = plt.subplots(figsize=(7.2, 5.2))

    for index, d_w in enumerate(dws):
        color = COLORS[index % len(COLORS)]
        marker = MARKERS[index % len(MARKERS)]
        available_x = [
            value for value in amplitude_squared if (d_w, value) in grouped
        ]
        theory = np.asarray(
            [
                float(grouped[(d_w, value)][0]["continuous_delta_s"])
                for value in available_x
            ],
            dtype=np.float64,
        )
        predicted = [
            _mean_std(
                float(row["predicted_delta_s"])
                for row in grouped[(d_w, value)]
            )
            for value in available_x
        ]

        x = np.asarray(available_x, dtype=np.float64)
        if len(x) >= 2:
            slope, intercept = np.polyfit(x, theory, deg=1)
            dense_x = np.linspace(x.min(), x.max(), 200)
            axis.plot(
                dense_x,
                slope * dense_x + intercept,
                color=color,
                linewidth=2.0,
                zorder=1,
            )
        else:
            axis.plot(x, theory, color=color, marker="_", markersize=12)

        axis.errorbar(
            x,
            [value[0] for value in predicted],
            yerr=[value[1] for value in predicted],
            color=color,
            marker=marker,
            markersize=7,
            markerfacecolor=color,
            markeredgecolor=color,
            linestyle="none",
            elinewidth=1.5,
            capsize=4,
            capthick=1.4,
            label=rf"$d_w={d_w:g}$",
            zorder=3,
        )

    axis.set_xticks(amplitude_squared)
    axis.set_xlabel(r"$A^2$")
    axis.set_ylabel(r"EP per saved step $\langle\Delta S\rangle$")
    axis.legend(
        loc="upper left",
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        title="solid: theory; symbols: KNEEP",
    )
    axis.margins(x=0.05, y=0.08)
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(figure_dir / "a2_delta_s.png", dpi=dpi)
    figure.savefig(figure_dir / "a2_delta_s.pdf")
    plt.close(figure)


def _kernel_figures(kernel_runs, figure_dir: Path, dpi: int, plt) -> int:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in kernel_runs:
        grouped[row["condition_id"]].append(row)

    output_dir = figure_dir / "kernel_spectra"
    output_dir.mkdir(parents=True, exist_ok=True)
    for condition_id, rows in sorted(grouped.items()):
        by_kernel: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_kernel[int(row["kernel_index"])].append(row)
        indices = sorted(by_kernel)
        predicted = [
            _mean_std(
                float(row["predicted_delta_s"]) for row in by_kernel[index]
            )
            for index in indices
        ]
        theory = [
            float(by_kernel[index][0]["continuous_delta_s"])
            for index in indices
        ]

        x = np.arange(len(indices), dtype=float)
        width = 0.36
        figure, axis = plt.subplots(figsize=(6.4, 4.4))
        axis.bar(
            x - width / 2,
            [value[0] for value in predicted],
            width=width,
            yerr=[value[1] for value in predicted],
            capsize=3,
            color="steelblue",
            alpha=0.82,
            label="Predicted",
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )
        axis.bar(
            x + width / 2,
            theory,
            width=width,
            color="mediumseagreen",
            alpha=0.82,
            label="Theory",
        )
        first = rows[0]
        axis.set_xticks(
            x,
            [
                KERNEL_LABELS[index]
                if index < len(KERNEL_LABELS)
                else f"k={index}"
                for index in indices
            ],
        )
        axis.set_xlabel("Shell / kernel distance")
        axis.set_ylabel(r"EP contribution per saved step")
        axis.set_title(
            rf"$A={float(first['amplitude']):g},\ "
            rf"d_w={float(first['d_w']):g}$"
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.legend(frameon=False)
        _style_axis(axis)
        figure.tight_layout()
        figure.savefig(output_dir / f"{condition_id}.png", dpi=dpi)
        plt.close(figure)
    return len(grouped)


def make_figures(results_dir: Path, dpi: int = 300) -> None:
    results_dir = results_dir.resolve()
    figure_dir = results_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    runs = _read_csv(results_dir / "runs.csv")
    kernel_runs = _read_csv(results_dir / "kernel_runs.csv")
    plt = _pyplot(results_dir)
    _performance_figure(runs, figure_dir, dpi, plt)
    kernel_count = _kernel_figures(kernel_runs, figure_dir, dpi, plt)
    print(f"Saved {figure_dir / 'a2_delta_s.png'}")
    print(f"Saved {kernel_count} kernel spectra under {figure_dir / 'kernel_spectra'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="directory containing runs.csv and kernel_runs.csv",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    make_figures(args.results_dir, args.dpi)


if __name__ == "__main__":
    main()
