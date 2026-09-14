"""Plot saved SAOU temperature tables without loading trajectories or models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if __package__:
    from .plot_saou_perform import COLORS, MARKERS, KERNEL_LABELS, _pyplot, _style_axis
else:
    from plot_saou_perform import COLORS, MARKERS, KERNEL_LABELS, _pyplot, _style_axis

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results" / "saou_temperature"
KERNEL_NAMES = ("local", "r=1", "r=2", "r=3", "r=4")


def _temperature_token(temperature: float) -> str:
    return f"{temperature:g}".replace("-", "m").replace(".", "p")


def _condition_indices(config):
    return [(pi, ti) for pi in range(len(config.parameters))
            for ti in range(len(config.temperatures))]


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"result table is empty: {path}")
    return rows


def _make_figures(
    config,
    summary: list[dict[str, object]],
    kernel_summary: list[dict[str, object]],
    results_dir: Path,
    figure_dir: Path,
    dpi: int,
) -> None:
    plt = _pyplot(results_dir)
    spectrum_dir = figure_dir / "kernel_spectra"
    spectrum_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    for parameter_index, parameters in enumerate(config.parameters):
        selected = [
            row
            for row in summary
            if int(row["parameter_index"]) == parameter_index + 1
        ]
        selected.sort(key=lambda row: float(row["temperature"]))
        if not selected:
            continue
        x = np.asarray([float(row["temperature"]) for row in selected])
        predicted = np.asarray([float(row["predicted_epr_mean"]) for row in selected])
        predicted_std = np.asarray(
            [float(row["predicted_epr_std"]) for row in selected]
        )
        theory = np.asarray([float(row["theoretical_epr_rate"]) for row in selected])
        amplitude_text = ",".join(f"{value:g}" for value in parameters.amplitudes)
        label = rf"$P_{parameter_index + 1}=({parameters.omega0:g},{amplitude_text})$"
        color = COLORS[parameter_index % len(COLORS)]
        marker = MARKERS[parameter_index % len(MARKERS)]
        # The stored continuous-time theory is independent of temperature.
        axis.plot(x, theory, color=color, linewidth=2.0, zorder=1)
        axis.errorbar(
            x,
            predicted,
            yerr=predicted_std,
            color=color,
            marker=marker,
            markersize=7,
            markerfacecolor=color,
            markeredgecolor=color,
            linestyle="none",
            elinewidth=1.5,
            capsize=4,
            capthick=1.4,
            label=label,
            zorder=3,
        )
    axis.set_xscale("log")
    axis.set_xticks(config.temperatures, [f"{value:g}" for value in config.temperatures])
    axis.set_xlabel(r"Temperature $T$")
    axis.set_ylabel(r"Entropy production rate $\sigma$")
    axis.legend(
        loc="best", frameon=True, fancybox=True, framealpha=0.9,
        title="Solid: theory; markers: KNEEP",
    )
    axis.margins(x=0.05, y=0.08)
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(figure_dir / "temperature_performance.png", dpi=dpi)
    figure.savefig(figure_dir / "temperature_performance.pdf")
    plt.close(figure)

    for parameter_index, temperature_index in _condition_indices(config):
        selected = [
            row
            for row in kernel_summary
            if int(row["parameter_index"]) == parameter_index + 1
            and int(row["temperature_index"]) == temperature_index + 1
        ]
        selected.sort(key=lambda row: int(row["kernel_index"]))
        if len(selected) != len(KERNEL_NAMES):
            continue
        theory = np.asarray([float(row["theoretical_epr_rate"]) for row in selected])
        predicted = np.asarray([float(row["predicted_epr_mean"]) for row in selected])
        predicted_std = np.asarray([float(row["predicted_epr_std"]) for row in selected])
        x = np.arange(len(KERNEL_NAMES))
        figure, axis = plt.subplots(figsize=(6.4, 4.4))
        width = 0.36
        axis.bar(
            x - width / 2,
            predicted,
            width=width,
            yerr=predicted_std,
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
        parameters = config.parameters[parameter_index]
        temperature = config.temperatures[temperature_index]
        parameter_text = ", ".join(
            f"{value:g}" for value in (parameters.omega0, *parameters.amplitudes)
        )
        axis.set_title(
            rf"$P_{parameter_index + 1}=({parameter_text}),\quad T={temperature:g}$"
        )
        axis.set_xticks(x, KERNEL_LABELS)
        axis.set_xlabel("Kernel $k$")
        axis.set_ylabel(r"Kernel EPR rate $\sigma_k$")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.legend(frameon=False)
        _style_axis(axis)
        figure.tight_layout()
        figure.savefig(
            spectrum_dir
            / f"p{parameter_index + 1:02d}_T_{_temperature_token(temperature)}.png",
            dpi=dpi,
        )
        plt.close(figure)


def run(results_dir: Path = DEFAULT_RESULTS_DIR, figure_dir: Path | None = None,
        dpi: int = 300) -> None:
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    payload = json.loads((results_dir / "config.json").read_text(encoding="utf-8"))["experiment"]
    config = SimpleNamespace(
        parameters=[SimpleNamespace(**item) for item in payload["parameters"]],
        temperatures=payload["temperatures"],
    )
    summary = _read_csv(results_dir / "summary.csv")
    kernel_summary = _read_csv(results_dir / "kernel_summary.csv")
    _make_figures(config, summary, kernel_summary, results_dir,
                  figure_dir or results_dir / "figures", dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    run(args.results_dir, args.figure_dir, args.dpi)


if __name__ == "__main__":
    main()
