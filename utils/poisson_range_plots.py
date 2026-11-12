"""In-memory figures for the LRAMB Poisson interaction-range experiment.

Plotting never writes files.  Evaluation records contain *total* KNEEP scores;
the architecture's spatial mean is already accounted for by the evaluator.
Reference maps are densities and are converted to per-site contributions with
``dx**2``.  All temporal statistics keep independent trajectories separate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter


Records = Sequence[Mapping[str, Any]]
_LEARNED = "#2466a0"
_REFERENCE = "#c46c28"
_TRAJECTORY = "#8997a6"
_COMPONENTS = ("local conservative", "long-range conservative", "active")


def condition_strength(record: Mapping[str, Any]) -> float:
    """Effective coupling, including the single control in legacy archives."""
    if record.get("is_control") or float(record["ell"]) == 0:
        return 0.0
    metadata = record.get("metadata", {})
    for source in (record, metadata.get("condition", {}), metadata.get("dataset", {})):
        value = source.get("strength", source.get("g"))
        if value is not None:
            value = float(value)
            if not np.isfinite(value):
                raise ValueError("condition strength must be finite")
            return value
    # Historical single-coupling records used this fixed value.
    return 0.35


def condition_label(record: Mapping[str, Any]) -> str:
    """Distinguish the g=0 control from a physical zero-range kernel."""
    if condition_strength(record) == 0:
        return "g=0 (no long-range)"
    return rf"$g={condition_strength(record):g},\ \ell={float(record['ell']):g}$"


def group_records_by_strength(records: Records, *, include_control: bool = True) -> dict[float, list[Mapping[str, Any]]]:
    """One report per interacting coupling; reuse the same g=0 control."""
    ordered = _records(records)
    controls = [record for record in ordered if condition_strength(record) == 0]
    strengths = sorted({condition_strength(record) for record in ordered if condition_strength(record) != 0})
    if not strengths:
        return {0.0: controls}
    return {strength: ([*controls] if include_control else []) + [
        record for record in ordered if condition_strength(record) == strength
    ] for strength in strengths}


def _condition_axis(ax: Any, records: Records) -> None:
    """Use condition ticks so the control is never fitted as a range value."""
    ax.set_xticks(np.arange(len(records)))
    labels = ["g=0\ncontrol" if condition_strength(record) == 0 else
              f"g={condition_strength(record):g}\nell={float(record['ell']):g}" for record in records]
    ax.set_xticklabels(labels, fontsize=8, rotation=35 if len(records) > 6 else 0,
                       ha="right" if len(records) > 6 else "center")
    ax.set_xlabel("long-range interaction condition")


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _records(records: Records) -> list[Mapping[str, Any]]:
    result = sorted(records, key=lambda record: (condition_strength(record), float(record["ell"])))
    if not result:
        raise ValueError("At least one evaluated interaction range is required.")
    return result


def _condition_colors(records: Records) -> list[Any]:
    """Keep physical ranges on one palette and the g=0 control gray."""
    ells = sorted({float(record["ell"]) for record in records if condition_strength(record) != 0})
    palette = dict(zip(ells, plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(ells)))))
    return ["0.45" if condition_strength(record) == 0 else palette[float(record["ell"])] for record in records]


def _condition_linestyles(records: Records) -> list[str]:
    strengths = sorted({condition_strength(record) for record in records if condition_strength(record) != 0})
    patterns = ("-", "--", ":", "-.")
    styles = {strength: patterns[index % len(patterns)] for index, strength in enumerate(strengths)}
    return [styles.get(condition_strength(record), "-") for record in records]


def _signed_norm(limit: float, log_scale: bool) -> Normalize:
    limit = max(float(limit), 1e-12)
    if log_scale:
        return SymLogNorm(linthresh=limit * 1e-3, vmin=-limit, vmax=limit, base=10)
    return Normalize(vmin=-limit, vmax=limit)


def _nonnegative_scale(axes: Sequence[Any], values: Sequence[np.ndarray]) -> None:
    """Use matched symlog limits, retaining exact zeros as plotted zeros."""
    maximum = 0.0
    for value in values:
        array = _array(value)
        if not np.isfinite(array).all():
            raise ValueError("Fluctuation summaries contain nonfinite values")
        if array.size:
            maximum = max(maximum, float(array.max()))
    threshold = maximum * 1e-6 if maximum > 0 else 1.0
    upper = maximum * 1.2 if maximum > 0 else 1.0
    for ax in axes:
        ax.set_yscale("symlog", linthresh=threshold)
        ax.set_ylim(0, upper)


def _increments(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    predicted = _array(record["predicted_increment"])
    reference = _array(record["reference_increment"])
    interval = float(record["interval"])
    if predicted.ndim != 4 or reference.ndim != 3:
        raise ValueError("Expected predictions [seed, trajectory, time, branch] and references [trajectory, time, component].")
    if predicted.shape[1:3] != reference.shape[:2] or not np.isfinite(interval) or interval <= 0 or predicted.shape[2] < 1:
        raise ValueError("Predictions and references must share trajectories, times, and a positive interval.")
    return predicted, reference, interval


def _std(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Descriptive population spread; a single seed has zero measured spread."""
    return np.std(values, axis=axis, ddof=0)


def _style(ax: Any, *, zero: bool = True) -> None:
    ax.grid(alpha=0.2)
    if zero:
        ax.axhline(0, color="0.45", linewidth=0.65, zorder=0)
    ax.tick_params(labelsize=9)


def plot_cumulative_ep(records: Records) -> Figure:
    """Cumulative EP, resetting at the start of each held-out trajectory.

    Bands show standard deviations, not confidence intervals: independent
    trajectories after averaging fit seeds, and fit seeds after averaging
    trajectories.  The medium reference differs by finite-time boundary terms.
    """
    records = _records(records)
    fig, axes = plt.subplots(len(records), 2, figsize=(13, 3.1 * len(records)), squeeze=False)
    for row, record in enumerate(records):
        prediction, reference, interval = _increments(record)
        cumulative = np.concatenate((np.zeros((*prediction.shape[:2], 1)), prediction.sum(-1).cumsum(-1)), axis=-1)
        ref_cumulative = np.concatenate((np.zeros((reference.shape[0], 1, reference.shape[-1])), reference.cumsum(1)), axis=1)
        time = np.arange(cumulative.shape[-1]) * interval
        ax, component_ax = axes[row]
        mean = cumulative.mean((0, 1))
        trajectory_spread = _std(cumulative.mean(0), axis=0)
        seed_spread = _std(cumulative.mean(1), axis=0)
        ax.fill_between(time, mean - trajectory_spread, mean + trajectory_spread, color=_TRAJECTORY, alpha=0.2, label="independent trajectories: +/-1 SD")
        ax.fill_between(time, mean - seed_spread, mean + seed_spread, color=_LEARNED, alpha=0.25, label="fit seeds: +/-1 SD")
        ax.plot(time, mean, color=_LEARNED, label="KNEEP stationary path EP")
        ref_total = ref_cumulative.sum(-1).mean(0)
        ax.plot(time, ref_total, color=_REFERENCE, linestyle="--", label="medium reference (boundary gauge)")
        for component, label in enumerate(_COMPONENTS):
            component_ax.plot(time, ref_cumulative[..., component].mean(0), label=label)
        component_ax.plot(time, ref_total, color="black", linewidth=1.2, linestyle="--", label="medium total")
        ax.set_title(f"{condition_label(record)}: cumulative total")
        component_ax.set_title("Reference components; conservative terms are boundaries")
        for panel in (ax, component_ax):
            panel.set(xlabel="elapsed time within each trajectory", ylabel="cumulative EP")
            _style(panel)
        if row == 0:
            ax.legend(fontsize=7, loc="best")
            component_ax.legend(fontsize=8, loc="best")
    fig.suptitle("Cumulative EP on independent held-out trajectories", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_window_averaged_ep(records: Records, window_steps: int = 64) -> Figure:
    """Ordered, nonoverlapping temporal windows, never pooled across trajectories.

    A window contains ``window_steps`` observed intervals (not necessarily
    solver steps). Its signed increments are summed, then divided by its actual
    physical duration. A final shorter window is retained with its own duration.
    Every learned branch/reference component is included before averaging.
    """
    if isinstance(window_steps, bool) or not isinstance(window_steps, int) or window_steps < 1:
        raise ValueError("window_steps must be a positive integer")
    ordered = _records(records)
    groups = group_records_by_strength(ordered, include_control=False)
    controls = [record for record in ordered if condition_strength(record) == 0]
    if controls and any(strength != 0 for strength in groups):
        groups = {0.0: controls, **groups}
    fig, axes = plt.subplots(len(groups), 2, figsize=(13, 3.3 * len(groups)), squeeze=False)
    for row, (strength, group) in enumerate(groups.items()):
        member_ids = {id(record) for record in group}
        for color, linestyle, record in zip(_condition_colors(ordered), _condition_linestyles(ordered), ordered):
            if id(record) not in member_ids:
                continue
            predicted, reference, interval = _increments(record)
            count = predicted.shape[2]
            starts = np.arange(0, count, window_steps)
            ends = np.minimum(starts + window_steps, count)
            duration = (ends - starts) * interval
            learned = np.add.reduceat(predicted.sum(-1), starts, axis=2) / duration
            direct = np.add.reduceat(reference.sum(-1), starts, axis=1) / duration
            time = (starts + ends) * interval / 2
            mean = learned.mean((0, 1))
            seed_sd = _std(learned.mean(1))
            trajectory_sd = _std(learned.mean(0))
            axes[row, 0].plot(time, mean, color=color, linestyle=linestyle, label=condition_label(record))
            axes[row, 0].fill_between(time, mean - trajectory_sd, mean + trajectory_sd, color=color, alpha=0.08)
            axes[row, 0].fill_between(time, mean - seed_sd, mean + seed_sd, color=color, alpha=0.2)
            mean, spread = direct.mean(0), _std(direct)
            axes[row, 1].plot(time, mean, color=color, linestyle=linestyle, label=condition_label(record))
            axes[row, 1].fill_between(time, mean - spread, mean + spread, color=color, alpha=0.13)
        group_label = "g=0 (no long-range)" if strength == 0 else f"g={strength:g}"
        for col, method in enumerate(("KNEEP stationary path estimate", "Direct midpoint medium reference")):
            axes[row, col].set(title=f"{group_label}: {method}", xlabel="elapsed time (window center)", ylabel="window mean EP / time")
            axes[row, col].legend(fontsize=8)
            _style(axes[row, col])
    fig.suptitle(f"EP rate in consecutive windows of up to {window_steps} observed intervals", fontsize=13)
    fig.text(0.5, 0.015, "Signed sums include all local, active and nonlocal contributions; no cumulative curves.\n"
             "Left bands: trajectory SD (faint), fit-seed SD (dark). Right band: trajectory SD. Different entropy gauges; bands are not confidence intervals.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.065, 1, 0.97))
    return fig


def plot_time_averaged_ep(records: Records, trajectory: int = 0, *, log_scale: bool = False) -> Figure:
    """Rate maps; optional signed logarithmic colors preserve negative EP."""
    records = _records(records)
    fig, axes = plt.subplots(len(records), 4, figsize=(15, 3.15 * len(records)), squeeze=False)
    learned_limit, reference_limit, spread_limit, field_limit = 1e-12, 1e-12, 1e-12, 1e-12
    for record in records:
        maps = _array(record["predicted_map"])
        if trajectory < 0 or trajectory >= maps.shape[1]:
            raise IndexError(f"trajectory={trajectory} is outside the held-out ensemble")
        seed_maps = maps[:, trajectory].sum(1)
        medium_map = _array(record["reference_map"])[trajectory].sum(0) * float(record["dx"]) ** 2
        learned_limit = max(learned_limit, float(np.abs(seed_maps.mean(0)).max()))
        reference_limit = max(reference_limit, float(np.abs(medium_map).max()))
        spread_limit = max(spread_limit, float(_std(seed_maps).max()))
        field_limit = max(field_limit, float(np.abs(_array(record["fields"])[trajectory, -1]).max()))
    for row, record in enumerate(records):
        predicted = _array(record["predicted_map"])
        reference = _array(record["reference_map"])
        if trajectory < 0 or trajectory >= predicted.shape[1]:
            raise IndexError(f"trajectory={trajectory} is outside the held-out ensemble")
        learned_seeds = predicted[:, trajectory].sum(1)
        learned = learned_seeds.mean(0)
        medium = reference[trajectory].sum(0) * float(record["dx"]) ** 2
        seed_spread = _std(learned_seeds)
        fields = _array(record["fields"])
        final = fields[trajectory, -1]
        extent = (0, final.shape[-1] * float(record["dx"]), 0, final.shape[-2] * float(record["dx"]))
        images = (
            axes[row, 0].imshow(final, origin="lower", cmap="RdBu_r", vmin=-field_limit, vmax=field_limit, extent=extent),
            axes[row, 1].imshow(learned, origin="lower", cmap="RdBu_r", norm=_signed_norm(learned_limit, log_scale), extent=extent),
            axes[row, 2].imshow(medium, origin="lower", cmap="RdBu_r", norm=_signed_norm(reference_limit, log_scale), extent=extent),
            axes[row, 3].imshow(seed_spread, origin="lower", cmap="magma", vmin=0, vmax=spread_limit, extent=extent),
        )
        titles = (
            f"{condition_label(record)}: final $\\phi$",
            f"KNEEP seed mean; sum={learned.sum():.3g}",
            f"Medium reference; sum={medium.sum():.3g}",
            "KNEEP fit-seed SD",
        )
        for col, (image, title) in enumerate(zip(images, titles)):
            axes[row, col].set(title=title, xlabel="x", ylabel="y")
            colorbar = fig.colorbar(image, ax=axes[row, col], shrink=0.78, label="field" if col == 0 else "EP / time / site")
            if log_scale and col in (1, 2):
                limit = float(image.norm.vmax)
                colorbar.set_ticks(np.asarray([-1, -.01, 0, .01, 1]) * limit)
                colorbar.formatter = FuncFormatter(lambda value, _: "0" if value == 0 else f"{value:.0e}")
                colorbar.update_ticks()
    scale = "signed logarithmic EP colors" if log_scale else "linear EP colors"
    fig.suptitle(f"Time-averaged maps, trajectory {trajectory}; {scale}\nColor scales differ between gauges, fixed across conditions", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_ep_spectrum(records: Records, *, log_scale: bool = False,
                     linthresh: float = 1e-8) -> Figure:
    """Signed mean and temporal variance by shell, without cumulative panels.

    The learned branches form a coordinate-dependent decomposition.  Physical
    shells in the reference partition the actual Poisson kernel and are not
    branch-wise training targets.  Variances here are per-shell, not additive.
    log_scale selects a signed logarithmic y axis for the mean shell rate;
    radius stays linear, retaining the local rho=0 bin. Variance always uses
    symlog to retain exact zeros. linthresh is in the respective y-axis units.
    """
    if not np.isfinite(linthresh) or linthresh <= 0:
        raise ValueError("linthresh must be finite and positive")
    records = _records(records)
    fig, axes = plt.subplots(len(records), 2, figsize=(11, 3.15 * len(records)), squeeze=False)
    for row, record in enumerate(records):
        prediction, _, interval = _increments(record)
        rate = prediction / interval
        reference = _array(record["reference_branch_increment"]) / interval
        radius = np.arange(rate.shape[-1]) * float(record["dx"])
        ref_radius = np.arange(reference.shape[-1]) * float(record["dx"])
        seed_means = rate.mean((1, 2))
        trajectory_means = rate.mean((0, 2))
        mean = seed_means.mean(0)
        reference_mean = reference.mean((0, 1))
        ax = axes[row, 0]
        ax.errorbar(radius, mean, yerr=_std(trajectory_means), fmt="o", color=_TRAJECTORY, alpha=0.8, capsize=5, label="independent trajectory SD")
        ax.errorbar(radius, mean, yerr=_std(seed_means), fmt="o-", color=_LEARNED, capsize=2, label="KNEEP mean +/- fit-seed SD")
        ax.plot(ref_radius, reference_mean, ".--", color=_REFERENCE, label="physical reference shells")
        ax.set_title(f"{condition_label(record)}: signed mean shell rate")
        ax.set_ylabel("EP / time")
        if log_scale:
            ax.set_yscale("symlog", linthresh=linthresh)
            ax.axhline(0, color="0.6", linewidth=0.6)
            ax.set_ylabel("EP / time (symlog)")
        variance = np.var(rate, axis=2).mean(1)
        reference_variance = np.var(reference, axis=1).mean(0)
        ax = axes[row, 1]
        variance_mean, variance_spread = variance.mean(0), _std(variance)
        ax.plot(radius, variance_mean, "o-", color=_LEARNED, label="KNEEP within-trajectory variance")
        ax.fill_between(radius, np.maximum(variance_mean - variance_spread, 0), variance_mean + variance_spread, color=_LEARNED, alpha=0.2, label="fit-seed SD")
        ax.plot(ref_radius, reference_variance, ".--", color=_REFERENCE, label="physical reference shells")
        ax.set(title="Temporal variance by shell", ylabel="Var(EP / time)")
        ax.set_yscale("symlog", linthresh=linthresh)
        for ax in axes[row]:
            ax.set_xlabel("Chebyshev shell radius (physical length)")
            _style(ax)
            if row == 0:
                ax.legend(fontsize=7, loc="best")
    scale = f"symlog mean (linear within +/-{linthresh:g})" if log_scale else "linear mean"
    fig.suptitle(f"EP shell spectra: {scale}; symlog variance\n"
                 "Learned branches and physical shells have different gauges", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_fluctuation_comparison(records: Records) -> Figure:
    """Overlay interaction ranges, with learned and direct gauges in two rows.

    Rates are saved increments divided by the physical observation interval.
    Both columns omit radius zero and show individual-shell fluctuations.
    Variance is computed within each trajectory (ddof=0), then averaged over
    trajectories and, for KNEEP, fit seeds. Bands are descriptive standard
    deviations across fit seeds (top) or trajectories (bottom), not CIs.

    Direct values are midpoint medium entropy from the simulated fields, not
    exact stationary path entropy. Learned branches and physical kernel shells
    have different gauges and may have different maximum radii.
    """
    records = _records(records)
    colors = _condition_colors(records)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex="col", sharey="col")
    maxima: list[list[np.ndarray]] = [[], []]
    titles = (r"Shell variance: $\mathrm{Var}_t(j_\rho)$",
              r"Distance scaling: $\mathrm{Var}_t(j_\rho)/\rho^2$")
    for color, linestyle, record in zip(colors, _condition_linestyles(records), records):
        predicted, _, interval = _increments(record)
        direct = _array(record["reference_branch_increment"])
        if direct.ndim != 3 or direct.shape[:2] != predicted.shape[1:3]:
            raise ValueError("Physical shell increments must match prediction trajectories and times")
        dx = float(record["dx"])
        if not np.isfinite(dx) or dx <= 0:
            raise ValueError("dx must be positive and finite")
        learned_rate, direct_rate = predicted[..., 1:] / interval, direct[..., 1:] / interval
        radii = (np.arange(1, predicted.shape[-1]) * dx,
                 np.arange(1, direct.shape[-1]) * dx)
        # Keep independent trajectories separate, and retain fit-seed variation.
        learned_shell = np.var(learned_rate, axis=2, ddof=0).mean(axis=1)
        direct_shell = np.var(direct_rate, axis=1, ddof=0)
        statistics = (
            (learned_shell, learned_shell / radii[0] ** 2),
            (direct_shell, direct_shell / radii[1] ** 2),
        )
        for row in range(2):
            for col, samples in enumerate(statistics[row]):
                mean, spread = samples.mean(axis=0), _std(samples, axis=0)
                axes[row, col].plot(radii[row], mean, marker="o", markersize=3,
                                    color=color, linestyle=linestyle, label=condition_label(record))
                lower, upper = np.maximum(mean - spread, 0), mean + spread
                axes[row, col].fill_between(radii[row], lower, upper,
                                           color=color, alpha=0.13)
                maxima[col].append(upper)
    for col in range(2):
        _nonnegative_scale(axes[:, col], maxima[col])
        for row in range(2):
            method = "KNEEP" if row == 0 else "Direct medium reference"
            axes[row, col].set_title(f"{method}\n{titles[col]}", fontsize=11)
            axes[row, col].set_xlabel("Chebyshev shell radius rho (physical length)")
            axes[row, col].set_ylabel("Var(EP / time) / length squared" if col == 1
                                      else "Var(EP / time)")
            axes[row, col].set_xlim(left=0)
            _style(axes[row, col])
    axes[0, 0].text(0.98, 0.97, "band: fit-seed SD", transform=axes[0, 0].transAxes,
                    ha="right", va="top", fontsize=8)
    axes[1, 0].text(0.98, 0.97, "band: independent-trajectory SD",
                    transform=axes[1, 0].transAxes, ha="right", va="top", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=min(len(records), 4), fontsize=8)
    fig.suptitle("Interaction-range dependence of nonlocal shell fluctuations", y=0.995, fontsize=14)
    fig.text(0.5, 0.025,
             "All panels omit rho=0. Color identifies ell; line style identifies g. Rates use increment / saved interval.\n"
             "The two decompositions have different gauges. Direct midpoint medium EP is not exact stationary total EP; shared symlog scales retain zeros.",
             ha="center", va="bottom", fontsize=8.5)
    legend_top = 0.91 if len(records) <= 5 else 0.82
    fig.tight_layout(rect=(0, 0.075, 1, legend_top))
    return fig


def _periodogram(rate: np.ndarray, interval: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided density, independently detrended along each final time axis."""
    count = rate.shape[-1]
    centered = rate - rate.mean(axis=-1, keepdims=True)
    power = (interval / count) * np.abs(np.fft.rfft(centered, axis=-1)) ** 2
    if count % 2 == 0:
        power[..., 1:-1] *= 2
    else:
        power[..., 1:] *= 2
    return np.fft.rfftfreq(count, d=interval), power


def plot_temporal_spectrum(records: Records) -> Figure:
    """Across-range PSD and TOTAL-rate variance, in separate method rows.

    Unlike ``plot_fluctuation_comparison``, these global statistics include
    radius zero, all nonlocal shells, and active/local reference components.
    Signed contributions are summed before periodograms or variances, and
    independent trajectories are never concatenated across boundaries.
    """
    records = _records(records)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col", sharey="col")
    colors = _condition_colors(records)
    psd_limits, variance_limits = [], []
    for index, (color, linestyle, record) in enumerate(zip(colors, _condition_linestyles(records), records)):
        prediction, reference, interval = _increments(record)
        total_rate = prediction.sum(-1) / interval
        direct_rate = reference.sum(-1) / interval
        frequency, psd = _periodogram(total_rate, interval)
        direct_frequency, direct_psd = _periodogram(direct_rate, interval)
        # Top: average trajectories within each fit seed before seed SD.
        # Bottom: direct trajectory SD, without a learned-seed dimension.
        summaries = ((psd.mean(axis=1), frequency), (direct_psd, direct_frequency))
        for row, (samples, frequencies) in enumerate(summaries):
            mean, spread = samples.mean(axis=0), _std(samples, axis=0)
            selected = slice(1, None)  # DC is zero after per-trajectory demeaning.
            axes[row, 0].plot(frequencies[selected], mean[selected], color=color, linestyle=linestyle,
                              linewidth=1.1, label=condition_label(record))
            lower, upper = np.maximum(mean - spread, 0), mean + spread
            axes[row, 0].fill_between(frequencies[selected], lower[selected], upper[selected],
                                     color=color, alpha=0.12)
            psd_limits.append(upper[selected])
        variance = np.var(total_rate, axis=-1, ddof=0)
        mean = float(variance.mean())
        seed_sd = float(_std(variance.mean(axis=1)))
        trajectory_sd = float(_std(variance.mean(axis=0)))
        axes[0, 1].errorbar(index, mean, yerr=trajectory_sd, fmt="none",
                            color=_TRAJECTORY, capsize=6,
                            label="independent-trajectory SD" if index == 0 else None)
        axes[0, 1].errorbar(index, mean, yerr=seed_sd, fmt="o", color=color,
                            capsize=3, label="fit-seed SD" if index == 0 else None)
        direct_variance = np.var(direct_rate, axis=-1, ddof=0)
        direct_mean, direct_sd = float(direct_variance.mean()), float(_std(direct_variance))
        axes[1, 1].errorbar(index, direct_mean, yerr=direct_sd, fmt="o", color=color,
                            capsize=5, label="independent-trajectory SD" if index == 0 else None)
        variance_limits.extend((np.asarray([mean + max(seed_sd, trajectory_sd)]),
                                np.asarray([direct_mean + direct_sd])))
    _nonnegative_scale(axes[:, 0], psd_limits)
    _nonnegative_scale(axes[:, 1], variance_limits)
    for row, method in enumerate(("KNEEP", "Direct midpoint medium reference")):
        axes[row, 0].set(title=f"{method}: global-rate PSD", xlabel="frequency (1 / time)",
                         ylabel="PSD of total EP rate", xscale="log")
        axes[row, 1].set(title=f"{method}: total temporal variance",
                         ylabel="mean within-trajectory Var(total EP / time)")
        _condition_axis(axes[row, 1], records)
        axes[row, 1].legend(fontsize=8)
        for ax in axes[row]:
            _style(ax, zero=False)
    axes[0, 0].text(0.02, 0.98, "band: fit-seed SD", transform=axes[0, 0].transAxes,
                    va="top", fontsize=8)
    axes[1, 0].text(0.02, 0.98, "band: independent-trajectory SD",
                    transform=axes[1, 0].transAxes, va="top", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=min(len(records), 5), fontsize=9)
    fig.suptitle("Global fluctuations across interaction ranges, including all local and active terms",
                 fontsize=13, y=0.995)
    fig.text(0.5, 0.02,
             "Signed contributions are summed before statistics. Periodograms are demeaned separately per trajectory; DC omitted.\n"
             "Shared symlog scales; bands/error bars are descriptive SDs. Direct medium EP is not exact stationary total EP.",
             ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.075, 1, 0.91))
    return fig


def _kernel_shells(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mass of periodic offsets grouped by minimum-image Chebyshev radius."""
    n = weights.shape[-1]
    offset = np.minimum(np.arange(n), n - np.arange(n))
    shell = np.maximum(offset[:, None], offset[None, :])
    return np.arange(n // 2 + 1), np.bincount(shell.ravel(), weights=weights.ravel(), minlength=n // 2 + 1)


def plot_ep_performance(records: Records, *, reference_atol: float = 1e-10) -> Figure:
    """Mean EP per observed transition and the ratio to direct medium EP.

    Ratios are ratios of ensemble mean rates, not averages of instantaneous
    ratios. Near-zero denominators are omitted and listed, never regularized
    by an epsilon. Error bars are descriptive fit-seed and paired-trajectory
    SDs; the medium gauge is not an exact stationary path-entropy target.
    """
    if not np.isfinite(reference_atol) or reference_atol < 0:
        raise ValueError("reference_atol must be finite and nonnegative")
    records = _records(records)
    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(records) * 0.95), 5.5))
    omitted, intervals, omitted_pairs = [], [], 0
    first_ratio = True
    for index, (record, color) in enumerate(zip(records, _condition_colors(records))):
        predicted, reference, interval = _increments(record)
        intervals.append(interval)
        learned = predicted.sum(-1).mean(-1)  # [fit, trajectory], EP / observed step
        direct = reference.sum(-1).mean(-1)  # [trajectory]
        learned_mean, direct_mean = float(learned.mean()), float(direct.mean())
        fit_sd = float(_std(learned.mean(1)))
        trajectory_sd = float(_std(learned.mean(0)))
        axes[0].errorbar(index - 0.11, learned_mean, yerr=trajectory_sd, fmt="none", color=_TRAJECTORY, capsize=6,
                        label="KNEEP trajectory SD" if index == 0 else None)
        axes[0].errorbar(index - 0.11, learned_mean, yerr=fit_sd, fmt="o", color=color, capsize=3,
                        label="KNEEP mean; fit-seed SD" if index == 0 else None)
        axes[0].errorbar(index + 0.11, direct_mean, yerr=float(_std(direct)), fmt="s", color=_REFERENCE, capsize=3,
                        label="direct medium; trajectory SD" if index == 0 else None)
        direct_rates = direct / interval
        denominator = float(direct_rates.mean())
        if abs(denominator) <= reference_atol:
            omitted.append(condition_label(record))
            continue
        ratio = (learned_mean / interval) / denominator
        fit_ratios = learned.mean(1) / interval / denominator
        valid = np.abs(direct_rates) > reference_atol
        omitted_pairs += int((~valid).sum())
        paired_ratios = learned.mean(0)[valid] / interval / direct_rates[valid]
        paired_sd = float(_std(paired_ratios)) if paired_ratios.size else 0.0
        axes[1].errorbar(index, ratio, yerr=paired_sd, fmt="none", color=_TRAJECTORY, capsize=6,
                        label="paired-trajectory ratio SD" if first_ratio else None)
        axes[1].errorbar(index, ratio, yerr=float(_std(fit_ratios)), fmt="o", color=color, capsize=3,
                        label="ratio of means; fit-seed SD" if first_ratio else None)
        first_ratio = False
    axes[0].set(title="Mean total-field EP per observed step", ylabel="total-field EP / observed transition")
    axes[1].set(title="KNEEP / direct-medium mean EP rate", ylabel="ratio of ensemble mean rates")
    axes[1].axhline(1, color="black", linestyle="--", linewidth=1, label="unity")
    for ax in axes:
        _condition_axis(ax, records)
        ax.legend(fontsize=8)
        _style(ax)
    cadence = f"Observed-step duration = {intervals[0]:g}" if len(set(intervals)) == 1 else "Observed-step duration varies between conditions; use each saved interval."
    note = f"{cadence}. Signed sums include all branches/components. Error bars are descriptive SDs, not CIs.\nThe direct midpoint medium reference and stationary learned estimate have different finite-time gauges."
    if omitted:
        note += "\nUndefined near-zero-reference ratios omitted: " + "; ".join(omitted)
    if omitted_pairs:
        note += f"\nPaired-trajectory ratio SD excludes {omitted_pairs} near-zero-reference trajectories."
    fig.text(0.5, 0.015, note, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.15 if omitted or omitted_pairs else 0.1, 1, 1))
    return fig


def plot_range_response(records: Records) -> Figure:
    """Compatibility alias for the two-panel EP performance comparison."""
    return plot_ep_performance(records)


def plot_training_history(records: Records) -> Figure:
    """Training/validation losses, plus recorded learning rates when available.

    Learning-rate values must align with each fit's recorded iterations. Old
    checkpoints lacking them retain the original single-axis loss plot.
    """
    records = _records(records)
    columns = min(2, len(records))
    rows = (len(records) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3.7 * rows), squeeze=False)
    for ax, record in zip(axes.flat, records):
        lr_ax = None
        for repeat, history in enumerate(record.get("history", [])):
            iterations = _array(history["iterations"])
            color = f"C{repeat % 10}"
            ax.plot(iterations, _array(history["train_losses"]), color=color, alpha=0.35, linewidth=0.8)
            ax.plot(iterations, _array(history["validation_losses"]), color=color, label=f"fit seed {repeat}: validation")
            learning_rates = history.get("learning_rates")
            if learning_rates is None and record.get("learning_rates") is not None:
                saved = _array(record["learning_rates"])
                if saved.ndim == 1:
                    learning_rates = saved
                elif saved.ndim == 2 and repeat < saved.shape[0]:
                    learning_rates = saved[repeat]
            if learning_rates is not None:
                values = _array(learning_rates)
                if values.shape == iterations.shape and values.size:
                    if not np.isfinite(values).all() or np.any(values < 0):
                        raise ValueError("Recorded learning rates must be finite and nonnegative")
                    if lr_ax is None:
                        lr_ax = ax.twinx()
                        lr_ax.set_ylabel("learning rate (dashed)", fontsize=8)
                        lr_ax.tick_params(labelsize=8)
                        lr_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                    lr_ax.plot(iterations, values, color=color, linestyle="--",
                               linewidth=0.9, alpha=0.55)
                    lr_ax.set_ylim(bottom=0)
        ax.set(title=f"{condition_label(record)}; faint = sampled training loss", xlabel="optimizer iteration", ylabel="alpha-divergence objective")
        _style(ax)
        if record.get("history"):
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "Training history unavailable", ha="center", transform=ax.transAxes)
    for ax in list(axes.flat)[len(records):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_kernel_diagnostics(records: Records) -> Figure:
    """Horizontal real-space kernels, with one row per effective coupling.

    Images show ``g * K_ell`` on one shared signed color scale. Archived
    ``kernel_weights`` are unit-coupling convolution weights, so the explicit
    strength factor is required. The g=0 control appears exactly once.
    """
    records = _records(records)
    strengths = sorted({condition_strength(record) for record in records if condition_strength(record) != 0})
    ells = sorted({float(record["ell"]) for record in records if condition_strength(record) != 0})
    controls = [record for record in records if condition_strength(record) == 0]
    rows, columns = max(len(strengths), 1), max(len(ells) + bool(controls), 1)
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.2 * rows), squeeze=False)
    effective = {id(record): _array(record.get("effective_kernel_weights",
                 condition_strength(record) * _array(record["kernel_weights"]))) for record in records}
    limit = max((float(np.max(np.abs(weights))) for weights in effective.values()), default=0.0) or 1.0
    cells = {}
    for record in records:
        strength = condition_strength(record)
        if strength != 0:
            cells[(strengths.index(strength), ells.index(float(record["ell"])))] = record
    if controls:
        cells[(0, columns - 1)] = controls[0]
    shown = []
    image = None
    for row in range(rows):
        for col in range(columns):
            ax = axes[row, col]
            record = cells.get((row, col))
            if record is None:
                ax.set_visible(False)
                continue
            weights = effective[id(record)]
            dx = float(record["dx"])
            n = weights.shape[-1]
            half = n * dx / 2
            image = ax.imshow(np.fft.fftshift(weights), origin="lower", cmap="RdBu_r",
                              vmin=-limit, vmax=limit,
                              extent=(-half - dx / 2, half - dx / 2,
                                      -half - dx / 2, half - dx / 2))
            ax.set(title=condition_label(record), xlabel="periodic x offset", ylabel="periodic y offset")
            shown.append(ax)
    fig.suptitle("Effective conservative kernels in real space: g K_ell", fontsize=13)
    fig.subplots_adjust(left=0.055, right=0.9, bottom=0.08, top=0.9, hspace=0.4, wspace=0.3)
    if image is not None:
        fig.colorbar(image, ax=shown, shrink=0.8, label="effective signed convolution weight g K_ell")
    return fig
