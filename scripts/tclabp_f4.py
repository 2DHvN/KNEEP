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

from models.tc_labp import TCLABPConfig, TCLABPTrajectory, simulate_trajectories


OUTPUT_DIR = ROOT / "results" / "tclabp_f4"
DEFAULT_PHIS = tuple(round(0.05 * index, 2) for index in range(1, 11))
DEFAULT_V0_VALUES = tuple(2.5 * index for index in range(11))
TRAJECTORY_FIELDS = (
    "sites",
    "angles",
    "occupancy",
    "exact_ep",
    "exact_ep_maps",
    "accepted_hops",
    "times",
)


@dataclass(frozen=True)
class SweepConfig:
    """Numerical settings for the TC-LABP ``(phi, v0)`` sweep."""

    phis: tuple[float, ...] = DEFAULT_PHIS
    v0_values: tuple[float, ...] = DEFAULT_V0_VALUES
    lattice_size: int = 30
    trajectories: int = 16
    trajectory_batch_size: int = 16
    steps: int = 1_000
    burn_steps: int = 100_000
    sampling_steps: int = 100
    rotational_diffusion: float = 1.5
    translational_diffusion: float = 1.0
    dt: float = 1.0e-4
    lattice_spacing: float = 0.5
    base_seed: int = 17_090_395
    save_trajectories: bool = True


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


def _save_trajectory_batch(
    result: TCLABPTrajectory,
    *,
    trajectory_dir: Path,
    config: SweepConfig,
    physical: TCLABPConfig,
    requested_phi: float,
    v0: float,
    condition_index: int,
    batch_start: int,
) -> Path:
    """Persist one condition batch without holding the full sweep in memory."""

    trajectory_dir.mkdir(parents=True, exist_ok=True)
    batch_stop = batch_start + int(result.sites.shape[0])
    path = trajectory_dir / (
        f"condition_{condition_index:03d}_batch_{batch_start:03d}-{batch_stop - 1:03d}.pt"
    )
    temporary_path = path.with_suffix(".tmp")
    seed = _condition_seed(config, condition_index, batch_start)
    torch.save(
        {
            "schema_version": 1,
            "condition": {
                "condition_index": condition_index,
                "requested_phi": float(requested_phi),
                "phi": float(physical.n_particles / physical.lattice_size**2),
                "v0": float(v0),
                "Pe": float(physical.Pe),
                "batch_start": batch_start,
                "batch_stop": batch_stop,
                "seed": seed,
            },
            "physics": asdict(physical),
            "sampling": {
                "steps": config.steps,
                "burn_steps": config.burn_steps,
                "sampling_steps": config.sampling_steps,
                "storage_dtype": "torch.float32",
            },
            "trajectory": {
                field: getattr(result, field) for field in TRAJECTORY_FIELDS
            },
        },
        temporary_path,
    )
    temporary_path.replace(path)
    return path


def _simulate_condition(
    config: SweepConfig,
    requested_phi: float,
    v0: float,
    condition_index: int,
    device: torch.device,
    trajectory_dir: Path | None = None,
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
        if trajectory_dir is not None:
            path = _save_trajectory_batch(
                result,
                trajectory_dir=trajectory_dir,
                config=config,
                physical=physical,
                requested_phi=requested_phi,
                v0=v0,
                condition_index=condition_index,
                batch_start=batch_start,
            )
            print(
                f"[trajectory saved] {path.name} "
                f"({path.stat().st_size / 1024**3:.3f} GiB)",
                flush=True,
            )
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
        shading="flat", cmap="viridis", vmin=0, vmax=0.8
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
    trajectory_dir: str | None = None,
) -> dict[str, float | int]:
    torch.set_num_threads(cpu_threads)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return _simulate_condition(
        config,
        phi,
        v0,
        condition_index,
        device,
        None if trajectory_dir is None else Path(trajectory_dir),
    )


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
    trajectory_dir = output_dir / "trajectories" if config.save_trajectories else None
    for completed, (condition_index, phi, v0) in enumerate(conditions, start=1):
        print(
            f"[{completed:>3}/{total}] phi={phi:g}, v0={v0:g} -> {device}",
            flush=True,
        )
        rows.append(
            _simulate_condition(
                config, phi, v0, condition_index, device, trajectory_dir
            )
        )
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
    trajectory_dir = output_dir / "trajectories" if config.save_trajectories else None

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
            None if trajectory_dir is None else str(trajectory_dir),
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


def _trajectory_storage_bytes(
    *, batch: int, frames: int, steps: int, particles: int, lattice_size: int
) -> int:
    sites_and_angles = batch * frames * particles * (2 * 8 + 4)
    occupancy = batch * frames * lattice_size**2
    ep_maps = batch * steps * lattice_size**2 * 4
    exact_ep_and_hops = batch * steps * (4 + 8)
    times = frames * 4
    return sites_and_angles + occupancy + ep_maps + exact_ep_and_hops + times


def _estimated_worker_storage_gib(config: SweepConfig) -> float:
    batch = min(config.trajectories, config.trajectory_batch_size)
    frames = config.steps + 1
    max_particles = max(
        _physical_config(config, phi, config.v0_values[0]).n_particles
        for phi in config.phis
    )
    size = _trajectory_storage_bytes(
        batch=batch,
        frames=frames,
        steps=config.steps,
        particles=max_particles,
        lattice_size=config.lattice_size,
    )
    return float(size / 1024**3)


def _estimated_total_trajectory_storage_gib(config: SweepConfig) -> float:
    frames = config.steps + 1
    size = 0
    for phi in config.phis:
        particles = _physical_config(config, phi, config.v0_values[0]).n_particles
        condition_size = _trajectory_storage_bytes(
            batch=config.trajectories,
            frames=frames,
            steps=config.steps,
            particles=particles,
            lattice_size=config.lattice_size,
        )
        size += len(config.v0_values) * condition_size
    return float(size / 1024**3)


def _run_metadata(
    config: SweepConfig,
    devices: Sequence[torch.device],
) -> dict[str, object]:
    return {
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
            "trajectories_saved": config.save_trajectories,
            "trajectory_directory": "trajectories" if config.save_trajectories else None,
            "estimated_total_trajectory_storage_gib": (
                _estimated_total_trajectory_storage_gib(config)
                if config.save_trajectories
                else 0.0
            ),
            "batch_seed_semantics": (
                "one deterministic generator seed per condition and trajectory batch"
            ),
        }
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
    if config.save_trajectories:
        estimated_total = _estimated_total_trajectory_storage_gib(config)
        print(
            f"saving complete trajectories to {output_dir / 'trajectories'}; "
            f"estimated total: {estimated_total:.2f} GiB"
        )
    else:
        print("complete trajectory saving is disabled")

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
    parser.add_argument("--lattice-size", type=int, default=30)
    parser.add_argument("--trajectories", type=int, default=16)
    parser.add_argument("--trajectory-batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--burn-steps", type=int, default=100_000)
    parser.add_argument("--sampling-steps", type=int, default=100)
    parser.add_argument("--rotational-diffusion", type=float, default=1.5)
    parser.add_argument("--translational-diffusion", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--lattice-spacing", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=17_090_395)
    parser.add_argument(
        "--save-trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save every returned trajectory tensor (default: enabled)",
    )
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
        save_trajectories=args.save_trajectories,
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
