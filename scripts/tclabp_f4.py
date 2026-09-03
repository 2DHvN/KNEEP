"""Map TC-LABP clustering and entropy production in the ``(phi, v0)`` plane.

The clustering observable follows Whitelam, Klymko, and Mandal,
"Phase separation and large deviations of lattice active matter",
J. Chem. Phys. 148, 154902 (2018), Fig. 2
(https://doi.org/10.1063/1.5023403):

* ``phi = N / L**2`` is the occupied-site fraction on an ``L x L`` lattice.
* ``f4 = N4 / N`` is the fraction of particles whose four cardinal
  nearest-neighbour sites are all occupied.
* Their Fig. 2 plots ``<f4>`` and ``log(<f4**2> - <f4>**2)`` against ``phi``
  and the *forward hopping rate* ``v+``.  The labelled ticks are
  ``phi = 0.1, ..., 0.5`` and ``v+ = 5, 10, ..., 25``; ``L=100``,
  ``v- = v0 = 1``, and ``D+ = D- = D_rot = 0.1``.  The paper does not list
  the underlying numerical grid points, so tick locations must not be called
  the exact raw sampling grid.
* Because that paper uses event-driven continuous-time Monte Carlo, its state
  averages are residence-time weighted by ``1/R(C)``.  The TC-LABP frames in
  this script are uniformly separated in physical time, so their ordinary
  arithmetic mean is the corresponding time average.

This script applies those ``phi`` and ``f4`` definitions to the repository's
TC-LABP model; it is not a reproduction of the paper's dynamics.  In
particular, ``v0`` below means ``TCLABPConfig.speed``, the continuous-angle
self-propulsion speed.  It is not the paper's lateral hopping rate also named
``v0``.  With the default ``lattice_spacing = translational_diffusion = 1``,
TC-LABP has its own grid Péclet number ``Pe = v0``.  The default ``v0`` values
use spacing 2.5, half the paper's labelled ``v+`` tick interval; the default
``phi`` spacing is similarly 0.05, half its labelled interval.  These are
explicit resolution choices because the raw paper grid was not reported, and
do not identify the two rates or their differently defined Péclet numbers.

Each parameter point defaults to four independent trajectories evaluated as
one GPU batch.  Every trajectory is burned in for 100,000 microscopic MC
steps, followed by 1,000 measurements separated by 100 microscopic MC steps.
Thus one point contains 4,004 ``f4`` frame values and 4,000 EPR intervals.

Two figures are produced: the paper-style ``f4`` mean/log-variance maps and a
map of the simulator's exact medium entropy-production rate per particle.  No
KNEEP model is trained by this script.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import multiprocessing
import os
import sys
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.tc_labp import TCLABPConfig, simulate_trajectories


OUTPUT_DIR = ROOT / "results" / "tclabp_f4"
PAPER_URL = "https://doi.org/10.1063/1.5023403"
PAPER_ARXIV_URL = "https://arxiv.org/abs/1709.03951"
PAPER_PHI_TICKS = (0.1, 0.2, 0.3, 0.4, 0.5)
PAPER_V_PLUS_TICKS = (5.0, 10.0, 15.0, 20.0, 25.0)
PAPER_F4_DISPLAY_RANGE = (0.0, 0.8)

# The paper does not publish its raw grid.  These TC-LABP defaults use half
# the distance between the labelled Fig. 2 ticks: Delta phi=0.05 and
# Delta v0=2.5.  Reusing the v+ numerical scale for TC-LABP v0 is only a grid
# choice; the two rates and their Péclet numbers are not identified.
DEFAULT_PHIS = tuple(round(0.05 * index, 2) for index in range(1, 11))
DEFAULT_V0_VALUES = tuple(2.5 * index for index in range(11))


@dataclass(frozen=True)
class SweepConfig:
    """Numerical settings for the TC-LABP ``(phi, v0)`` sweep."""

    phis: tuple[float, ...] = DEFAULT_PHIS
    v0_values: tuple[float, ...] = DEFAULT_V0_VALUES
    lattice_size: int = 100
    trajectories: int = 4
    trajectory_batch_size: int = 4
    steps: int = 1_000
    burn_steps: int = 100_000
    sampling_steps: int = 100
    rotational_diffusion: float = 1.0
    translational_diffusion: float = 1.0
    dt: float = 1.0e-3
    lattice_spacing: float = 1.0
    base_seed: int = 17_090_395


CSV_FIELDS = (
    "requested_phi",
    "phi",
    "v0",
    "Pe",
    "n_particles",
    "mean_f4",
    "f4_variance",
    "log_f4_variance",
    "f4_trajectory_sem",
    "mean_medium_ep_per_interval",
    "medium_epr_total",
    "medium_epr_per_particle",
    "medium_epr_per_site",
    "medium_epr_per_particle_sem",
)


def _strictly_increasing(name: str, values: Sequence[float]) -> None:
    if not values:
        raise ValueError(f"{name} must be nonempty")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if array.size > 1 and not np.all(np.diff(array) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")


def _validate_config(config: SweepConfig) -> None:
    _strictly_increasing("phis", config.phis)
    _strictly_increasing("v0_values", config.v0_values)
    if any(phi <= 0.0 or phi > 1.0 for phi in config.phis):
        raise ValueError("every phi must lie in (0, 1]")
    if any(v0 < 0.0 for v0 in config.v0_values):
        raise ValueError("every v0 must be nonnegative")
    for name in (
        "lattice_size",
        "trajectories",
        "trajectory_batch_size",
        "steps",
        "sampling_steps",
    ):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config.lattice_size % 5 != 0:
        raise ValueError("lattice_size must be a multiple of 5")
    if (
        isinstance(config.burn_steps, bool)
        or not isinstance(config.burn_steps, int)
        or config.burn_steps < 0
    ):
        raise ValueError("burn_steps must be a nonnegative integer")
    if isinstance(config.base_seed, bool) or not isinstance(config.base_seed, int):
        raise ValueError("base_seed must be an integer")

    # Construct every edge-case physical configuration now, before creating
    # any output.  TCLABPConfig also checks the fixed-step probability bound.
    for phi in config.phis:
        for v0 in (config.v0_values[0], config.v0_values[-1]):
            _physical_config(config, phi, v0)

    actual_phis = np.asarray(
        [
            _physical_config(config, phi, config.v0_values[0]).n_particles
            for phi in config.phis
        ],
        dtype=np.float64,
    ) / float(config.lattice_size**2)
    if actual_phis.size > 1 and not np.all(np.diff(actual_phis) > 0.0):
        raise ValueError(
            "requested phis collapse to duplicate particle counts; use a larger lattice "
            "or more widely spaced phis"
        )


def _physical_config(config: SweepConfig, phi: float, v0: float) -> TCLABPConfig:
    return TCLABPConfig(
        lattice_size=config.lattice_size,
        density=float(phi),
        speed=float(v0),
        rotational_diffusion=config.rotational_diffusion,
        translational_diffusion=config.translational_diffusion,
        dt=config.dt,
        lattice_spacing=config.lattice_spacing,
    )


def f4_per_frame(occupancy: torch.Tensor) -> torch.Tensor:
    """Return ``f4`` for each frame of an occupancy tensor.

    ``occupancy`` may have any number of leading dimensions but its final two
    axes must be the periodic square lattice.  Only the four cardinal nearest
    neighbours are counted; diagonal sites never enter ``f4``.
    """

    if not isinstance(occupancy, torch.Tensor):
        raise TypeError("occupancy must be a torch.Tensor")
    if occupancy.ndim < 2 or occupancy.shape[-2] != occupancy.shape[-1]:
        raise ValueError("occupancy must end in two equally sized lattice axes")
    occupied = occupancy.to(dtype=torch.bool)
    particle_count = occupied.sum(dim=(-2, -1))
    if bool((particle_count == 0).any()):
        raise ValueError("f4 is undefined for an empty configuration")
    neighbour_count = (
        torch.roll(occupied, 1, dims=-2).to(torch.int8)
        + torch.roll(occupied, -1, dims=-2).to(torch.int8)
        + torch.roll(occupied, 1, dims=-1).to(torch.int8)
        + torch.roll(occupied, -1, dims=-1).to(torch.int8)
    )
    surrounded = occupied & neighbour_count.eq(4)
    return surrounded.sum(dim=(-2, -1), dtype=torch.float64) / particle_count


def _sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(values.size))


def _condition_seed(config: SweepConfig, condition_index: int, batch_start: int) -> int:
    return int(config.base_seed + condition_index * 1_000_003 + batch_start)


def _simulate_condition(
    config: SweepConfig,
    requested_phi: float,
    v0: float,
    condition_index: int,
    device: torch.device,
) -> dict[str, float | int]:
    physical = _physical_config(config, requested_phi, v0)
    interval_duration = float(config.sampling_steps * physical.dt)
    f4_samples: list[np.ndarray] = []
    f4_trajectory_means: list[float] = []
    epr_trajectory_means: list[float] = []
    ep_interval_means: list[float] = []

    # Replicas are vectorized in batches on one GPU.  Limiting the batch size
    # bounds the CPU trajectory storage retained by the simulator.
    for batch_start in range(0, config.trajectories, config.trajectory_batch_size):
        batch_size = min(
            config.trajectory_batch_size,
            config.trajectories - batch_start,
        )
        result = simulate_trajectories(
            physical,
            n_trajectories=batch_size,
            n_steps=config.steps,
            burn_steps=config.burn_steps,
            sampling_steps=config.sampling_steps,
            seed=_condition_seed(config, condition_index, batch_start),
            simulation_device=device,
            storage_dtype=torch.float32,
            progress=False,
        )
        batch_f4 = f4_per_frame(result.occupancy).cpu().numpy()
        batch_ep = result.exact_ep.to(torch.float64).cpu().numpy()
        batch_epr_per_particle = batch_ep.mean(axis=1) / (
            interval_duration * physical.n_particles
        )
        f4_samples.append(batch_f4.reshape(-1))
        f4_trajectory_means.extend(batch_f4.mean(axis=1).tolist())
        epr_trajectory_means.extend(batch_epr_per_particle.tolist())
        ep_interval_means.extend(batch_ep.mean(axis=1).tolist())
        del result
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    pooled_f4 = np.concatenate(f4_samples)
    mean_f4 = float(pooled_f4.mean())
    f4_variance = float(np.mean((pooled_f4 - mean_f4) ** 2))
    # This is exactly log(<f4^2>-<f4>^2), with zero represented by -inf.
    log_f4_variance = float(np.log(f4_variance)) if f4_variance > 0.0 else -math.inf
    mean_ep_per_interval = float(np.mean(ep_interval_means))
    total_epr = mean_ep_per_interval / interval_duration
    per_particle_epr = float(np.mean(epr_trajectory_means))

    return {
        "requested_phi": float(requested_phi),
        "phi": float(physical.n_particles / physical.lattice_size**2),
        "v0": float(v0),
        "Pe": float(physical.Pe),
        "n_particles": int(physical.n_particles),
        "mean_f4": mean_f4,
        "f4_variance": f4_variance,
        "log_f4_variance": log_f4_variance,
        "f4_trajectory_sem": _sem(np.asarray(f4_trajectory_means)),
        "mean_medium_ep_per_interval": mean_ep_per_interval,
        "medium_epr_total": float(total_epr),
        "medium_epr_per_particle": per_particle_epr,
        "medium_epr_per_site": float(total_epr / physical.lattice_size**2),
        "medium_epr_per_particle_sem": _sem(np.asarray(epr_trajectory_means)),
    }


def _grid(rows: Sequence[dict[str, float | int]], key: str) -> np.ndarray:
    phis = sorted({float(row["phi"]) for row in rows})
    v0_values = sorted({float(row["v0"]) for row in rows})
    matrix = np.full((len(v0_values), len(phis)), np.nan, dtype=np.float64)
    phi_index = {value: index for index, value in enumerate(phis)}
    v0_index = {value: index for index, value in enumerate(v0_values)}
    for row in rows:
        matrix[v0_index[float(row["v0"])], phi_index[float(row["phi"])]] = float(row[key])
    if np.isnan(matrix).any():
        raise ValueError(f"missing values in {key} grid")
    if not np.isfinite(matrix).all() and key != "log_f4_variance":
        raise ValueError(f"non-finite or missing values in {key} grid")
    return matrix


def _cell_edges(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 1:
        width = max(abs(float(values[0])) * 0.1, 0.05)
        return np.asarray((values[0] - width, values[0] + width))
    midpoints = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(
        ([values[0] - (midpoints[0] - values[0])], midpoints,
         [values[-1] + (values[-1] - midpoints[-1])])
    )


def _plot_f4(rows: Sequence[dict[str, float | int]], output_dir: Path) -> Path:
    phis = sorted({float(row["phi"]) for row in rows})
    v0_values = sorted({float(row["v0"]) for row in rows})
    mean_f4 = _grid(rows, "mean_f4")
    log_variance = _grid(rows, "log_f4_variance")
    finite_log = log_variance[np.isfinite(log_variance)]
    display_floor = float(finite_log.min() - 1.0) if finite_log.size else -30.0
    display_log_variance = np.where(np.isfinite(log_variance), log_variance, display_floor)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), constrained_layout=True)
    mesh0 = axes[0].pcolormesh(
        _cell_edges(phis), _cell_edges(v0_values), mean_f4,
        shading="flat", cmap="viridis", vmin=PAPER_F4_DISPLAY_RANGE[0],
        vmax=PAPER_F4_DISPLAY_RANGE[1],
    )
    mesh1 = axes[1].pcolormesh(
        _cell_edges(phis), _cell_edges(v0_values), display_log_variance,
        shading="flat", cmap="magma",
    )
    fig.colorbar(mesh0, ax=axes[0], label=r"$\langle f_4\rangle$", extend="max")
    fig.colorbar(mesh1, ax=axes[1], label=r"$\log(\langle f_4^2\rangle-\langle f_4\rangle^2)$")
    axes[0].set_title(r"TC-LABP clustering: $\langle f_4\rangle$")
    axes[1].set_title(r"TC-LABP clustering fluctuations")
    for axis in axes:
        axis.set_xlabel(r"site filling $\phi=N/L^2$")
        axis.set_ylabel(r"TC-LABP propulsion speed $v_0$")
        axis.set_xticks(phis)
        axis.set_yticks(v0_values)
    path = output_dir / "tclabp_f4_phase_diagram.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _plot_epr(rows: Sequence[dict[str, float | int]], output_dir: Path) -> Path:
    phis = sorted({float(row["phi"]) for row in rows})
    v0_values = sorted({float(row["v0"]) for row in rows})
    epr = _grid(rows, "medium_epr_per_particle")
    fig, ax = plt.subplots(figsize=(6.2, 4.9), constrained_layout=True)
    mesh = ax.pcolormesh(
        _cell_edges(phis), _cell_edges(v0_values), epr, shading="flat", cmap="inferno"
    )
    fig.colorbar(mesh, ax=ax, label=r"medium EPR / particle / time")
    ax.set(
        title="TC-LABP exact medium entropy-production rate",
        xlabel=r"site filling $\phi=N/L^2$",
        ylabel=r"TC-LABP propulsion speed $v_0$",
    )
    ax.set_xticks(phis)
    ax.set_yticks(v0_values)
    path = output_dir / "tclabp_epr_phase_diagram.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _write_csv(rows: Iterable[dict[str, float | int]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _paper_metadata() -> dict[str, object]:
    return {
        "citation": (
            "S. Whitelam, K. Klymko, and D. Mandal, Phase separation and large "
            "deviations of lattice active matter, J. Chem. Phys. 148, 154902 (2018)"
        ),
        "url": PAPER_URL,
        "arxiv_url": PAPER_ARXIV_URL,
        "phi_definition": "N / L^2 (occupied-site fraction)",
        "f4_definition": (
            "N4 / N, where N4 counts particles with all four cardinal "
            "neighbours occupied"
        ),
        "figure_2_quantities": ["mean(f4)", "log(mean(f4^2)-mean(f4)^2)"],
        "figure_2_lattice_size": 100,
        "figure_2_initial_condition": "disordered",
        "figure_2_simulations_per_parameter_point": 1,
        "figure_2_average": (
            "continuous-time residence weighting: sum_k Q(C_k)/R(C_k) "
            "divided by sum_k 1/R(C_k)"
        ),
        "figure_2_approximate_visible_phi_range": [0.05, 0.5],
        "figure_2_approximate_visible_v_plus_range": [0.5, 25.0],
        "figure_2_labelled_phi_ticks": list(PAPER_PHI_TICKS),
        "figure_2_labelled_v_plus_ticks": list(PAPER_V_PLUS_TICKS),
        "figure_2_mean_f4_display_range": list(PAPER_F4_DISPLAY_RANGE),
        "figure_2_exact_sampling_grid_reported": False,
        "fixed_rates": {"v_minus": 1.0, "v0_lateral": 1.0, "D_plus": 0.1, "D_minus": 0.1},
        "paper_activity_relation": "Pe = 5 * (v_plus - 1)",
        "epr_calculated_in_paper": False,
        "default_grid_caveat": (
            "Default spacings Delta phi=0.05 and Delta v0=2.5 are half the "
            "paper's labelled tick intervals. This is a resolution choice, not "
            "a mapping between rates or Pe definitions."
        ),
        "model_difference": (
            "The paper scans its discrete-orientation CTMC forward rate v_plus. "
            "This script scans TCLABPConfig.speed, denoted v0, in the "
            "continuous-angle TC-LABP model."
        ),
    }


def describe_paper_conventions() -> str:
    """Return the paper definitions printed at the start of every run."""

    return (
        "Paper convention (Whitelam, Klymko & Mandal, Fig. 2):\n"
        "  phi = N/L^2; visible range is about 0.05--0.50; labelled ticks "
        "= 0.1, 0.2, 0.3, 0.4, 0.5\n"
        "  f4 = fraction of particles with all four cardinal neighbours occupied\n"
        "  plotted mean-f4 colour range = 0--0.8 (mathematical range is 0--1)\n"
        "  plotted activity = v_plus, not v0; labelled v_plus ticks = 5, 10, 15, 20, 25\n"
        "  L=100, v_minus=v0(lateral)=1, D_plus=D_minus=0.1\n"
        "  exact simulation grid/run length and EPR were not reported in that paper\n"
        "TC-LABP convention in this script:\n"
        "  v0 = TCLABPConfig.speed (self-propulsion); it is not the paper's lateral v0\n"
        "  grid spacing is half the labelled paper tick spacing; dynamics still differ"
    )


def _condition_specs(config: SweepConfig) -> list[tuple[int, float, float]]:
    return [
        (v0_index * len(config.phis) + phi_index, phi, v0)
        for v0_index, v0 in enumerate(config.v0_values)
        for phi_index, phi in enumerate(config.phis)
    ]


def _allocated_cpu_threads(n_workers: int) -> int:
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    return max(1, allocated // n_workers)


def _run_condition_on_device(
    config: SweepConfig,
    condition_index: int,
    phi: float,
    v0: float,
    device_name: str,
    cpu_threads: int,
) -> dict[str, float | int]:
    torch.set_num_threads(cpu_threads)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return _simulate_condition(config, phi, v0, condition_index, device)


def _sort_rows(
    rows: Iterable[dict[str, float | int]],
) -> list[dict[str, float | int]]:
    return sorted(rows, key=lambda row: (float(row["v0"]), float(row["phi"])))


def _run_serial(
    config: SweepConfig,
    device: torch.device,
    conditions: Sequence[tuple[int, float, float]],
    output_dir: Path,
) -> list[dict[str, float | int]]:
    torch.set_num_threads(_allocated_cpu_threads(1))
    rows: list[dict[str, float | int]] = []
    total = len(conditions)
    for completed, (condition_index, phi, v0) in enumerate(conditions, start=1):
        print(
            f"[{completed:>3}/{total}] phi={phi:g}, v0={v0:g} -> {device}",
            flush=True,
        )
        rows.append(_simulate_condition(config, phi, v0, condition_index, device))
        _write_csv(_sort_rows(rows), output_dir / "tclabp_f4_epr.csv")
    return rows


def _run_parallel(
    config: SweepConfig,
    devices: tuple[torch.device, ...],
    conditions: Sequence[tuple[int, float, float]],
    output_dir: Path,
) -> list[dict[str, float | int]]:
    pending = deque(conditions)
    context = multiprocessing.get_context("spawn")
    executors = [
        ProcessPoolExecutor(max_workers=1, mp_context=context) for _ in devices
    ]
    cpu_threads = _allocated_cpu_threads(len(devices))
    active: dict[object, tuple[int, int, float, float]] = {}
    rows: list[dict[str, float | int]] = []
    total = len(conditions)

    def submit(worker_index: int) -> None:
        if not pending:
            return
        condition_index, phi, v0 = pending.popleft()
        device = devices[worker_index]
        print(
            f"[dispatch {condition_index + 1:>3}/{total}] "
            f"phi={phi:g}, v0={v0:g} -> {device}",
            flush=True,
        )
        future = executors[worker_index].submit(
            _run_condition_on_device,
            config,
            condition_index,
            phi,
            v0,
            str(device),
            cpu_threads,
        )
        active[future] = (worker_index, condition_index, phi, v0)

    try:
        for worker_index in range(min(len(devices), len(pending))):
            submit(worker_index)
        while active:
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                worker_index, condition_index, phi, v0 = active.pop(future)
                try:
                    row = future.result()
                except Exception as error:
                    raise RuntimeError(
                        f"condition phi={phi:g}, v0={v0:g} failed on "
                        f"{devices[worker_index]}"
                    ) from error
                rows.append(row)
                print(
                    f"[complete {len(rows):>3}/{total}] "
                    f"phi={phi:g}, v0={v0:g} <- {devices[worker_index]}",
                    flush=True,
                )
                _write_csv(_sort_rows(rows), output_dir / "tclabp_f4_epr.csv")
                submit(worker_index)
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)
    return rows


def _estimated_worker_storage_gib(config: SweepConfig) -> float:
    batch = min(config.trajectories, config.trajectory_batch_size)
    frames = config.steps + 1
    sites = batch * frames * 2 * 8
    angles = batch * frames * 4
    occupancy = batch * frames * config.lattice_size**2
    ep_maps = batch * config.steps * config.lattice_size**2 * 4
    hops = batch * config.steps * 8
    max_particles = max(
        _physical_config(config, phi, config.v0_values[0]).n_particles
        for phi in config.phis
    )
    return float((max_particles * (sites + angles) + occupancy + ep_maps + hops) / 1024**3)


def _run_metadata(
    config: SweepConfig,
    devices: Sequence[torch.device],
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "paper": _paper_metadata(),
        "sweep": asdict(config),
        "execution": {
            "devices": [str(device) for device in devices],
            "condition_parallelism": len(devices),
            "trajectories_per_point": config.trajectories,
            "trajectory_batch_size": min(
                config.trajectories, config.trajectory_batch_size
            ),
            "f4_values_per_point": config.trajectories * (config.steps + 1),
            "epr_intervals_per_point": config.trajectories * config.steps,
            "burn_steps_per_trajectory": config.burn_steps,
            "sampled_microscopic_steps_per_trajectory": (
                config.steps * config.sampling_steps
            ),
            "batch_seed_semantics": (
                "one deterministic generator seed per condition and trajectory batch"
            ),
        },
        "semantics": {
            "f4_average": "equal-time average over post-burn fixed-step TC-LABP frames",
            "f4_variance": "population variance over all saved frames and replicas",
            "epr": (
                "exact accepted-hop medium entropy production; system boundary "
                "entropy omitted"
            ),
            "epr_plot_normalization": "per particle per physical time",
            "recorded_interval_duration": config.sampling_steps * config.dt,
            "grid_rationale": (
                "The paper does not report raw grid points. Delta phi=0.05 and "
                "Delta v0=2.5 are half its labelled Fig. 2 tick intervals; the "
                "v0 numerical scale is not a mapping to the paper's v_plus."
            ),
            "phase_classification_threshold_used": False,
            "paper_simulations_per_point": 1,
            "replicas_per_point_rationale": (
                "four independent TC-LABP trajectories provide a minimal between-run SEM; "
                "the paper used one simulation per parameter point"
            ),
            "burn_in_rationale": (
                "100000 microscopic steps is the requested production default; the paper "
                "does not report a Fig. 2 burn-in length or guarantee convergence at this value"
            ),
        },
    }


def run(
    config: SweepConfig,
    *,
    output_dir: Path = OUTPUT_DIR,
    devices: Sequence[torch.device | str] | None = None,
) -> list[dict[str, float | int]]:
    """Run the sweep across devices and save its CSV, metadata, and figures."""

    _validate_config(config)
    selected_devices = (
        _automatic_devices()
        if devices is None
        else tuple(torch.device(device) for device in devices)
    )
    _validate_devices(selected_devices)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(_run_metadata(config, selected_devices), indent=2),
        encoding="utf-8",
    )
    print(describe_paper_conventions())
    print(f"devices={', '.join(map(str, selected_devices))}; output={output_dir}")
    print(
        f"grid={len(config.phis)} phi x {len(config.v0_values)} v0 "
        f"= {len(config.phis) * len(config.v0_values)} points; "
        f"replicas/point={config.trajectories}, "
        f"trajectory batch={min(config.trajectories, config.trajectory_batch_size)}"
    )
    print(
        f"per point: f4={config.trajectories * (config.steps + 1)} values, "
        f"EPR={config.trajectories * config.steps} intervals; "
        f"burn-in={config.burn_steps} microscopic steps/trajectory"
    )
    estimated_storage = _estimated_worker_storage_gib(config)
    print(f"estimated peak stored trajectory per worker: {estimated_storage:.2f} GiB")

    conditions = _condition_specs(config)
    rows = (
        _run_serial(config, selected_devices[0], conditions, output_dir)
        if len(selected_devices) == 1
        else _run_parallel(config, selected_devices, conditions, output_dir)
    )
    rows = _sort_rows(rows)
    if len(rows) != len(conditions):
        raise RuntimeError(f"sweep ended with {len(rows)}/{len(conditions)} points")

    csv_path = output_dir / "tclabp_f4_epr.csv"
    _write_csv(rows, csv_path)
    f4_path = _plot_f4(rows, output_dir)
    epr_path = _plot_epr(rows, output_dir)
    print("saved:")
    for path in (f4_path, epr_path, csv_path, metadata_path):
        print(f"  {path}")
    return rows


def _device_from_argument(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda:0")
    return device


def _cuda_devices(count: int) -> tuple[torch.device, ...]:
    if count <= 0:
        raise ValueError("--num-gpus must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    visible = torch.cuda.device_count()
    if count > visible:
        raise RuntimeError(
            f"--num-gpus={count}, but CUDA_VISIBLE_DEVICES exposes only {visible}"
        )
    return tuple(torch.device(f"cuda:{index}") for index in range(count))


def _automatic_devices(max_gpus: int = 4) -> tuple[torch.device, ...]:
    """Use up to the first four visible GPUs, falling back to one CPU."""

    if max_gpus <= 0:
        raise ValueError("max_gpus must be positive")
    if not torch.cuda.is_available():
        return (torch.device("cpu"),)
    count = min(max_gpus, torch.cuda.device_count())
    if count <= 0:
        return (torch.device("cpu"),)
    return tuple(torch.device(f"cuda:{index}") for index in range(count))


def _validate_devices(devices: Sequence[torch.device]) -> None:
    if not devices:
        raise ValueError("at least one device is required")
    names = [str(device) for device in devices]
    if len(set(names)) != len(names):
        raise ValueError("devices must be unique")
    for device in devices:
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("devices must be cpu or cuda")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise RuntimeError(f"CUDA device index is unavailable: {device}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phis", type=float, nargs="+", default=list(DEFAULT_PHIS))
    parser.add_argument(
        "--v0-values", type=float, nargs="+", default=list(DEFAULT_V0_VALUES)
    )
    parser.add_argument("--lattice-size", type=int, default=100)
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--trajectory-batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--burn-steps", type=int, default=100_000)
    parser.add_argument("--sampling-steps", type=int, default=100)
    parser.add_argument("--rotational-diffusion", type=float, default=1.0)
    parser.add_argument("--translational-diffusion", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--lattice-spacing", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17_090_395)
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--device",
        default=None,
        help="force one device (examples: cpu, cuda:1, auto)",
    )
    device_group.add_argument(
        "--num-gpus",
        type=int,
        help="use the first N visible GPUs (default: automatically use up to 4)",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config = SweepConfig(
        phis=tuple(args.phis),
        v0_values=tuple(args.v0_values),
        lattice_size=args.lattice_size,
        trajectories=args.trajectories,
        trajectory_batch_size=args.trajectory_batch_size,
        steps=args.steps,
        burn_steps=args.burn_steps,
        sampling_steps=args.sampling_steps,
        rotational_diffusion=args.rotational_diffusion,
        translational_diffusion=args.translational_diffusion,
        dt=args.dt,
        lattice_spacing=args.lattice_spacing,
        base_seed=args.seed,
    )
    devices = (
        _cuda_devices(args.num_gpus)
        if args.num_gpus is not None
        else (
            (_device_from_argument(args.device),)
            if args.device is not None
            else _automatic_devices()
        )
    )
    run(config, output_dir=args.output_dir, devices=devices)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
