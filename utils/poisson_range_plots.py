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
from matplotlib.figure import Figure


Records = Sequence[Mapping[str, Any]]
_LEARNED = "#2466a0"
_REFERENCE = "#c46c28"
_TRAJECTORY = "#8997a6"
_COMPONENTS = ("local conservative", "Poisson conservative", "active")


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _records(records: Records) -> list[Mapping[str, Any]]:
    result = sorted(records, key=lambda record: float(record["ell"]))
    if not result:
        raise ValueError("At least one evaluated interaction range is required.")
    return result


def _increments(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    predicted = _array(record["predicted_increment"])
    reference = _array(record["reference_increment"])
    interval = float(record["interval"])
    if predicted.ndim != 4 or reference.ndim != 3:
        raise ValueError("Expected predictions [seed, trajectory, time, branch] and references [trajectory, time, component].")
    if predicted.shape[1:3] != reference.shape[:2] or interval <= 0:
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
        ax.set_title(rf"$\ell={float(record['ell']):g}$: cumulative total")
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


def plot_time_averaged_ep(records: Records, trajectory: int = 0) -> Figure:
    """Rate maps for one held-out trajectory; sums equal global mean rates."""
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
            axes[row, 1].imshow(learned, origin="lower", cmap="RdBu_r", vmin=-learned_limit, vmax=learned_limit, extent=extent),
            axes[row, 2].imshow(medium, origin="lower", cmap="RdBu_r", vmin=-reference_limit, vmax=reference_limit, extent=extent),
            axes[row, 3].imshow(seed_spread, origin="lower", cmap="magma", vmin=0, vmax=spread_limit, extent=extent),
        )
        titles = (
            rf"$\ell={float(record['ell']):g}$: final $\phi$",
            f"KNEEP seed mean; sum={learned.sum():.3g}",
            f"Medium reference; sum={medium.sum():.3g}",
            "KNEEP fit-seed SD",
        )
        for col, (image, title) in enumerate(zip(images, titles)):
            axes[row, col].set(title=title, xlabel="x", ylabel="y")
            fig.colorbar(image, ax=axes[row, col], shrink=0.78, label="field" if col == 0 else "EP / time / site")
    fig.suptitle(f"Time-averaged maps, trajectory {trajectory}; color scales differ between gauges, fixed across ell", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_ep_spectrum(records: Records) -> Figure:
    """Signed shell contributions, temporal variation, and shell cumulants.

    The learned branches form a coordinate-dependent decomposition.  Physical
    shells in the reference partition the actual Poisson kernel and are not
    branch-wise training targets.  Variances here are per-shell, not additive.
    """
    records = _records(records)
    fig, axes = plt.subplots(len(records), 3, figsize=(15, 3.15 * len(records)), squeeze=False)
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
        ax.set_title(rf"$\ell={float(record['ell']):g}$: signed mean shell rate")
        ax.set_ylabel("EP / time")
        variance = np.var(rate, axis=2).mean(1)
        reference_variance = np.var(reference, axis=1).mean(0)
        ax = axes[row, 1]
        variance_mean, variance_spread = variance.mean(0), _std(variance)
        ax.plot(radius, variance_mean, "o-", color=_LEARNED, label="KNEEP within-trajectory variance")
        ax.fill_between(radius, np.maximum(variance_mean - variance_spread, 0), variance_mean + variance_spread, color=_LEARNED, alpha=0.2, label="fit-seed SD")
        ax.plot(ref_radius, reference_variance, ".--", color=_REFERENCE, label="physical reference shells")
        ax.set(title="Temporal variance by shell", ylabel="Var(EP / time)")
        ax.set_yscale("symlog", linthresh=1e-8)
        ax = axes[row, 2]
        cumulative = seed_means.cumsum(-1)
        cumulative_mean, cumulative_spread = cumulative.mean(0), _std(cumulative)
        ax.plot(radius, cumulative_mean, "o-", color=_LEARNED, label="KNEEP shells <= radius")
        ax.fill_between(radius, cumulative_mean - cumulative_spread, cumulative_mean + cumulative_spread, color=_LEARNED, alpha=0.2, label="fit-seed SD")
        ax.plot(ref_radius, reference_mean.cumsum(), ".--", color=_REFERENCE, label="physical shells <= radius")
        ax.set(title="Signed cumulative shell rate", ylabel="EP / time")
        for ax in axes[row]:
            ax.set_xlabel("Chebyshev shell radius (physical length)")
            _style(ax)
            if row == 0:
                ax.legend(fontsize=7, loc="best")
    fig.suptitle("EP shell spectra: learned branches and physical shells have different gauges", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
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
    """Global-rate PSD and variance with all branch covariances retained."""
    records = _records(records)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(records)))
    ells, learned_variances, learned_seed_sd, learned_trajectory_sd, reference_variances = [], [], [], [], []
    for color, record in zip(colors, records):
        prediction, reference, interval = _increments(record)
        # Sum first: var(sum(branches)) retains cross-branch covariances.
        total_rate = prediction.sum(-1) / interval
        ref_rate = reference.sum(-1) / interval
        frequency, psd = _periodogram(total_rate, interval)
        _, reference_psd = _periodogram(ref_rate, interval)
        mean_psd = psd.mean((0, 1))
        seed_psd_spread = _std(psd.mean(1))
        ell = float(record["ell"])
        axes[0].plot(frequency[1:], mean_psd[1:], color=color, label=rf"$\ell={ell:g}$ KNEEP")
        axes[0].fill_between(frequency[1:], np.maximum(mean_psd[1:] - seed_psd_spread[1:], 1e-30), mean_psd[1:] + seed_psd_spread[1:], color=color, alpha=0.13)
        axes[0].plot(frequency[1:], reference_psd.mean(0)[1:], color=color, linestyle="--", alpha=0.8)
        variance = np.var(total_rate, axis=-1)
        ells.append(ell)
        learned_variances.append(variance.mean())
        learned_seed_sd.append(_std(variance.mean(1)))
        learned_trajectory_sd.append(_std(variance.mean(0)))
        reference_variances.append(np.var(ref_rate, axis=-1).mean())
    axes[0].set(xlabel="frequency (1 / time)", ylabel="PSD of global EP rate", xscale="log", yscale="log", title="Per-trajectory periodograms; no boundary concatenation")
    axes[0].legend(fontsize=8)
    axes[0].text(0.02, 0.02, "solid: KNEEP; dashed: medium reference\nband: fit-seed SD; DC removed", transform=axes[0].transAxes, fontsize=8)
    axes[1].errorbar(ells, learned_variances, yerr=learned_trajectory_sd, fmt="o", color=_TRAJECTORY, capsize=5, label="independent trajectory SD")
    axes[1].errorbar(ells, learned_variances, yerr=learned_seed_sd, fmt="o-", color=_LEARNED, capsize=2, label="KNEEP mean +/- fit-seed SD")
    axes[1].plot(ells, reference_variances, "s--", color=_REFERENCE, label="medium reference")
    axes[1].set(xlabel=r"Poisson interaction length $\ell$", ylabel="mean within-trajectory Var(global EP / time)", title="Global temporal variation; branch covariances included")
    axes[1].legend(fontsize=8)
    for ax in axes:
        _style(ax, zero=False)
    fig.tight_layout()
    return fig


def _kernel_shells(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mass of periodic offsets grouped by minimum-image Chebyshev radius."""
    n = weights.shape[-1]
    offset = np.minimum(np.arange(n), n - np.arange(n))
    shell = np.maximum(offset[:, None], offset[None, :])
    return np.arange(n // 2 + 1), np.bincount(shell.ravel(), weights=weights.ravel(), minlength=n // 2 + 1)


def plot_range_response(records: Records) -> Figure:
    """Summary of range manipulation and learned response, without trend assumptions."""
    records = _records(records)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(records)))
    ells, mean_rates, seed_sd, trajectory_sd, medium_rates = [], [], [], [], []
    for color, record in zip(colors, records):
        prediction, reference, interval = _increments(record)
        rate = prediction.sum(-1).mean(-1) / interval
        ell = float(record["ell"])
        ells.append(ell)
        mean_rates.append(rate.mean())
        seed_sd.append(_std(rate.mean(1)))
        trajectory_sd.append(_std(rate.mean(0)))
        medium_rates.append(reference.sum(-1).mean() / interval)
        # K(0)=0 in LRAMB: actual real-space weights are signed and sum to
        # zero. Sum absolute *site* weights before grouping into shells.
        shells, absolute_mass = _kernel_shells(np.abs(_array(record["kernel_weights"])))
        total_mass = absolute_mass.sum()
        tail = (total_mass - absolute_mass.cumsum()) / max(total_mass, 1e-30)
        axes[1].plot(shells * float(record["dx"]), np.maximum(tail, 0), "o-", markersize=3, color=color, label=rf"$\ell={ell:g}$")
        signed_means = prediction.mean((1, 2)) / interval
        absolute_means = np.abs(signed_means)
        totals = absolute_means.sum(-1, keepdims=True)
        learned_tail = 1 - np.cumsum(absolute_means, axis=-1) / np.maximum(totals, 1e-30)
        learned_radius = np.arange(learned_tail.shape[-1]) * float(record["dx"])
        axes[2].plot(learned_radius, learned_tail.mean(0), "o-", markersize=3, color=color, label=rf"$\ell={ell:g}$")
        axes[2].fill_between(learned_radius, learned_tail.mean(0) - _std(learned_tail), learned_tail.mean(0) + _std(learned_tail), color=color, alpha=0.13)
    axes[0].errorbar(ells, mean_rates, yerr=trajectory_sd, fmt="o", color=_TRAJECTORY, capsize=5, label="independent trajectory SD")
    axes[0].errorbar(ells, mean_rates, yerr=seed_sd, fmt="o-", color=_LEARNED, capsize=2, label="KNEEP mean +/- fit-seed SD")
    axes[0].plot(ells, medium_rates, "s--", color=_REFERENCE, label="medium reference")
    axes[0].set(xlabel=r"Poisson interaction length $\ell$", ylabel="global mean EP / time", title="Does KNEEP respond to the range intervention?")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Chebyshev radius (physical length)", ylabel="fraction of sum |actual signed kernel weights| beyond radius", title="Actual Poisson kernel absolute-weight tail", ylim=(-0.04, 1.04))
    axes[2].set(xlabel="learned branch radius (physical length)", ylabel="absolute signed-mean contribution fraction beyond radius", title="Learned shell response; band = fit-seed SD", ylim=(-0.04, 1.04))
    for ax in axes:
        _style(ax)
    for ax in axes[1:]:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_training_history(records: Records) -> Figure:
    """Training and held-out validation losses for every independent fit seed."""
    records = _records(records)
    columns = min(2, len(records))
    rows = (len(records) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3.7 * rows), squeeze=False)
    for ax, record in zip(axes.flat, records):
        for repeat, history in enumerate(record.get("history", [])):
            iterations = _array(history["iterations"])
            color = f"C{repeat % 10}"
            ax.plot(iterations, _array(history["train_losses"]), color=color, alpha=0.35, linewidth=0.8)
            ax.plot(iterations, _array(history["validation_losses"]), color=color, label=f"fit seed {repeat}: validation")
        ax.set(title=rf"$\ell={float(record['ell']):g}$; faint = sampled training loss", xlabel="optimizer iteration", ylabel="alpha-divergence objective")
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
    """The kernel actually used by the simulator, with FFT and shell views."""
    records = _records(records)
    fig, axes = plt.subplots(len(records), 3, figsize=(13, 3.2 * len(records)), squeeze=False)
    for row, record in enumerate(records):
        weights = _array(record["kernel_weights"])
        multiplier = _array(record["kernel_multiplier"])
        dx, ell = float(record["dx"]), float(record["ell"])
        n = weights.shape[-1]
        half = n * dx / 2
        limit = max(float(np.max(np.abs(weights))), 1e-30)
        image = axes[row, 0].imshow(np.fft.fftshift(weights), origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit, extent=(-half, half, -half, half))
        axes[row, 0].set(title=rf"$\ell={ell:g}$: actual Poisson weights", xlabel="periodic x offset", ylabel="periodic y offset")
        fig.colorbar(image, ax=axes[row, 0], shrink=0.8)
        radius, mass = _kernel_shells(weights)
        axes[row, 1].plot(radius * dx, mass, "o-", label="signed shell mass")
        axes[row, 1].plot(radius * dx, mass.cumsum(), ".--", label="cumulative signed mass")
        axes[row, 1].set(title=f"mass sum={weights.sum():.6g}", xlabel="Chebyshev radius (physical length)", ylabel="kernel weight")
        axes[row, 1].legend(fontsize=8)
        wavevector = 2 * np.pi * np.fft.fftfreq(n, d=dx)
        q = 2 * np.abs(np.sin(wavevector * dx / 2)) / dx
        # Axial modes make no radial binning approximation.
        positive = np.arange(n // 2 + 1)
        axes[row, 2].plot(q[positive], multiplier[0, positive], "o-", markersize=3, label="simulator multiplier, qy=0")
        axes[row, 2].set(title="Poisson Fourier attenuation; K(0)=0", xlabel="lattice |qx| (1 / length)", ylabel="kernel Fourier multiplier")
        axes[row, 2].legend(fontsize=8)
        for ax in axes[row, 1:]:
            _style(ax)
    fig.tight_layout()
    return fig
