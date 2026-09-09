"""Focused SAOU performance sweep: A squared versus one-step entropy production."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from models.saou import (
    SAOUConfig,
    simulate_trajectories,
    theoretical_epr_components_absolute,
    theoretical_epr_rate,
)
from utils.training import (
    TrainingConfig,
    channel_normalization,
    predict_epr_branch_increments,
    train_model,
)


OUTPUT_DIR = ROOT / "results" / "saou_perform"
DATA_DIR = OUTPUT_DIR / "data"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOSS_FIGURE_DIR = FIGURE_DIR / "losses"
KERNEL_NAMES = ("local", "r=1", "r=2", "r=3", "r=4")


def _notebook_saou() -> SAOUConfig:
    shells = tuple(0.5 / math.sqrt(radius) for radius in range(1, 5))
    return SAOUConfig(
        lattice_size=32,
        radii=(1, 2, 3, 4),
        amplitudes=shells,
        gamma=1.0,
        omega0=sum(shells) + 0.2,
        temperature=1.0,
        dt=1e-2,
        weight_normalization="mean",
    )


@dataclass(frozen=True)
class ExperimentConfig:
    amplitudes: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
    dws: tuple[float, ...] = (0.0, 0.1, 0.2)
    repeats: int = 5
    base_data_seed: int = 5
    base_training_seed: int = 5
    train_trajectories: int = 1_000
    train_samples: int = 1_000
    test_trajectories: int = 1
    test_samples: int = 10_000
    burn_steps: int = 10_000
    hidden_channels: int = 64
    hidden_layers: int = 3
    activation: str = "elu"
    saou: SAOUConfig = field(default_factory=_notebook_saou)
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            alpha=-0.5,
            iterations=3_000,
            train_batch_size=2_048,
            validation_batch_size=2_048,
            prediction_batch_size=256,
            learning_rate=3e-3,
            weight_decay=1e-4,
            gradient_clip=1.0,
            validate_every=100,
            train_fraction=0.8,
        )
    )


def _validate(config: ExperimentConfig) -> None:
    if config.repeats <= 0 or not config.amplitudes or not config.dws:
        raise ValueError("the sweep and repeat counts must be nonempty")
    invalid_sweep = (
        len(set(config.amplitudes)) != len(config.amplitudes)
        or len(set(config.dws)) != len(config.dws)
        or any(not np.isfinite(x) or x < 0 for x in config.amplitudes)
        or any(not np.isfinite(x) for x in config.dws)
    )
    if invalid_sweep:
        raise ValueError("sweep values must be unique, finite, and A >= 0")
    if config.saou.radii != (1, 2, 3, 4):
        raise ValueError("this study requires shells (1,2,3,4)")
    if config.saou.temperature != 1.0 or config.saou.weight_normalization != "mean":
        raise ValueError("this study fixes T=1 and mean-normalized shells")
    invalid_sizes = config.train_trajectories < 2 or config.test_trajectories < 1
    invalid_sizes |= min(config.train_samples, config.test_samples) < 2 or config.burn_steps < 0
    invalid_sizes |= min(config.hidden_channels, config.hidden_layers) <= 0
    if invalid_sizes or config.activation not in {"elu", "relu"}:
        raise ValueError("invalid data/model size or activation")


def _token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _condition_id(amplitude: float, d_w: float) -> str:
    return f"A_{_token(amplitude)}_dw_{_token(d_w)}"


def _indices(config: ExperimentConfig) -> list[tuple[int, int]]:
    return [(ai, di) for di in range(len(config.dws)) for ai in range(len(config.amplitudes))]


def _condition_saou(config: ExperimentConfig, ai: int, di: int) -> SAOUConfig:
    amplitude = config.amplitudes[ai]
    shells = tuple(amplitude / math.sqrt(r) for r in config.saou.radii)
    return replace(
        config.saou,
        amplitudes=shells,
        omega0=sum(shells) + config.dws[di],
        temperature=1.0,
    )


def _data_seeds(config: ExperimentConfig, ai: int, di: int) -> tuple[int, int]:
    number = di * len(config.amplitudes) + ai
    train = config.base_data_seed + number * (config.train_trajectories + config.test_trajectories)
    return train, train + config.train_trajectories


def _training_seed(config: ExperimentConfig, repeat: int) -> int:
    return config.base_training_seed + repeat


def _euler_reference(saou: SAOUConfig) -> tuple[float, np.ndarray]:
    """Exact stationary ensemble EP of one finite Euler transition."""
    size = saou.lattice_size
    kernels = [np.ones((size, size), dtype=np.float64)]
    lower = 0
    for radius in saou.radii:
        offsets = [
            (dy, dx)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if lower < max(abs(dy), abs(dx)) <= radius
        ]
        kernel = np.zeros((size, size), dtype=np.float64)
        for dy, dx in offsets:
            kernel[(-dy) % size, (-dx) % size] = 1.0 / len(offsets)
        kernels.append(np.fft.fftn(kernel).real)
        lower = radius

    coefficients = np.asarray(
        (saou.omega0 - sum(saou.amplitudes), *saou.amplitudes),
        dtype=np.float64,
    )
    terms = coefficients[:, None, None] * np.stack(kernels)
    frequency = terms.sum(axis=0)
    denominator = 2.0 * saou.gamma - saou.dt * (
        saou.gamma**2 + frequency**2
    )
    if np.any(denominator <= 0):
        raise ValueError("Euler transition is not mean-square stable")
    components = np.sum(
        terms * frequency[None] * (4.0 * saou.dt / denominator)[None],
        axis=(1, 2),
    )
    return float(components.sum()), components


def _record(config: ExperimentConfig, ai: int, di: int) -> dict[str, object]:
    amplitude, d_w = config.amplitudes[ai], config.dws[di]
    saou = _condition_saou(config, ai, di)
    train_seed, test_seed = _data_seeds(config, ai, di)
    exact, exact_k = _euler_reference(saou)
    continuous_k = theoretical_epr_components_absolute(saou) * saou.effective_dt
    continuous = theoretical_epr_rate(saou) * saou.effective_dt
    if not np.isclose(exact_k.sum(), exact) or not np.isclose(
        continuous_k.sum(), continuous
    ):
        raise RuntimeError("kernel references do not sum to their totals")
    return {
        "condition_id": _condition_id(amplitude, d_w),
        "amplitude_index": ai + 1,
        "d_w_index": di + 1,
        "amplitude": amplitude,
        "amplitude_squared": amplitude**2,
        "d_w": d_w,
        "omega0": saou.omega0,
        **{f"a{i + 1}": value for i, value in enumerate(saou.amplitudes)},
        "temperature": saou.temperature,
        "train_data_seed": train_seed,
        "test_data_seed": test_seed,
        "theoretical_delta_s": exact,
        "continuous_delta_s": continuous,
        **{f"theoretical_k{i}_delta_s": float(x) for i, x in enumerate(exact_k)},
        **{f"continuous_k{i}_delta_s": float(x) for i, x in enumerate(continuous_k)},
    }


def _directory(config: ExperimentConfig, ai: int, di: int, root: Path) -> Path:
    return root / _condition_id(config.amplitudes[ai], config.dws[di])


def _data_path(config: ExperimentConfig, ai: int, di: int, role: str) -> Path:
    return _directory(config, ai, di, DATA_DIR) / f"{role}.pt"


def _checkpoint_path(config: ExperimentConfig, ai: int, di: int, repeat: int) -> Path:
    return _directory(config, ai, di, CHECKPOINT_DIR) / f"repeat_{repeat + 1:03d}.pt"


def _relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _atomic_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load(path: Path, mmap: bool = False):
    kwargs = {"map_location": "cpu", "weights_only": True}
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        try:
            return torch.load(path, **kwargs)
        except TypeError:
            kwargs.pop("mmap", None)
            return torch.load(path, **kwargs)


def _hash(config: ExperimentConfig) -> str:
    encoded = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot infer CSV columns for empty table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _prepare(config: ExperimentConfig) -> str:
    for path in (OUTPUT_DIR, DATA_DIR, CHECKPOINT_DIR, FIGURE_DIR, LOSS_FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)
    experiment_hash = _hash(config)
    path = OUTPUT_DIR / "config.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("experiment_sha256") != experiment_hash:
            raise RuntimeError(f"{OUTPUT_DIR} contains another experiment")
    else:
        temporary = path.with_suffix(".json.tmp")
        payload = {
            "format_version": 1,
            "experiment_sha256": experiment_hash,
            "experiment": asdict(config),
        }
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)
    _write_csv(
        OUTPUT_DIR / "conditions.csv",
        [_record(config, ai, di) for ai, di in _indices(config)],
    )
    return experiment_hash


def _trajectory(
    config: ExperimentConfig,
    experiment_hash: str,
    ai: int,
    di: int,
    role: str,
    device: torch.device,
) -> torch.Tensor:
    path = _data_path(config, ai, di, role)
    n_trajectories, n_samples = (
        (config.train_trajectories, config.train_samples)
        if role == "train"
        else (config.test_trajectories, config.test_samples)
    )
    seed = _data_seeds(config, ai, di)[role == "test"]
    condition = _condition_id(config.amplitudes[ai], config.dws[di])
    if path.exists():
        payload = _load(path, mmap=True)
        if not isinstance(payload, dict) or payload.get("experiment_sha256") != experiment_hash:
            raise RuntimeError(f"trajectory metadata mismatch: {path}")
        trajectories = payload.get("trajectories")
    else:
        print(f"[{condition}] generating fixed {role} trajectories on {device}", flush=True)
        trajectories = simulate_trajectories(
            _condition_saou(config, ai, di),
            n_trajectories=n_trajectories,
            n_samples=n_samples,
            burn_steps=config.burn_steps,
            seed=seed,
            simulation_device=device,
            storage_dtype=torch.float32,
        )
        payload = {
            "format_version": 1, "experiment_sha256": experiment_hash,
            "condition_id": condition, "role": role, "seed": seed,
            "trajectories": trajectories,
        }
        _atomic_save(path, payload)
    expected = (n_trajectories, n_samples, 2, config.saou.lattice_size, config.saou.lattice_size)
    if not isinstance(trajectories, torch.Tensor) or tuple(trajectories.shape) != expected:
        raise RuntimeError(f"invalid trajectory tensor: {path}")
    if trajectories.dtype != torch.float32:
        raise RuntimeError(f"invalid trajectory dtype: {path}")
    return trajectories


def _split(video: torch.Tensor, fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    stop = int(len(video) * fraction)
    if not 0 < stop < len(video):
        raise ValueError("train_fraction leaves an empty split")
    return video[:stop], video[stop:]


def _run_condition(
    config: ExperimentConfig,
    experiment_hash: str,
    ai: int,
    di: int,
    device: torch.device,
) -> int:
    pending = [r for r in range(config.repeats) if not _checkpoint_path(config, ai, di, r).exists()]
    missing_data = [
        _data_path(config, ai, di, role)
        for role in ("train", "test")
        if not _data_path(config, ai, di, role).exists()
    ]
    if len(pending) < config.repeats and missing_data:
        raise RuntimeError(
            "trajectory data are missing for existing checkpoints: "
            + ", ".join(map(str, missing_data))
        )
    if not pending:
        return 0
    condition = _record(config, ai, di)
    saou = _condition_saou(config, ai, di)
    train_data = _trajectory(config, experiment_hash, ai, di, "train", device)
    train_video, validation_video = _split(train_data, config.training.train_fraction)
    normalization = channel_normalization(train_video)
    test_video = _trajectory(config, experiment_hash, ai, di, "test", device)
    exact_k = np.asarray([condition[f"theoretical_k{i}_delta_s"] for i in range(5)])
    continuous_k = np.asarray([condition[f"continuous_k{i}_delta_s"] for i in range(5)])
    for repeat in pending:
        started = time.perf_counter()
        seed = _training_seed(config, repeat)
        print(
            f"[{condition['condition_id']}] training "
            f"{repeat + 1}/{config.repeats} (seed={seed})",
            flush=True,
        )
        trained = train_model(
            train_video, validation_video, config.training, model_seed=seed, device=device,
            n_components=2,
            hidden_channels=config.hidden_channels,
            hidden_layers=config.hidden_layers,
            max_distance=4,
            activation=config.activation,
            normalization=normalization,
            progress=False,
        )
        increments = predict_epr_branch_increments(
            trained, test_video, batch_size=config.training.prediction_batch_size, device=device,
        )
        predicted_k = increments.mean(axis=0, dtype=np.float64)
        if predicted_k.shape != exact_k.shape or not np.all(np.isfinite(predicted_k)):
            raise RuntimeError("invalid kernel prediction")
        predicted = float(predicted_k.sum())
        train_path = _data_path(config, ai, di, "train")
        test_path = _data_path(config, ai, di, "test")
        checkpoint = {
            "format_version": 1,
            "experiment_sha256": experiment_hash,
            "condition": condition,
            "repeat": repeat + 1,
            "training_seed": seed,
            "saou_config": asdict(saou),
            "model_config": {
                "n_components": 2,
                "hidden_channels": config.hidden_channels,
                "hidden_layers": config.hidden_layers,
                "max_distance": 4,
                "ep_component_indices": None,
                "activation": config.activation,
            },
            "training_config": asdict(config.training),
            "model_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in trained.model.state_dict().items()
            },
            "mean": trained.mean.detach().cpu(),
            "std": trained.std.detach().cpu(),
            "best_iteration": trained.best_iteration,
            "best_validation_loss": trained.best_validation_loss,
            "history": {
                "iterations": list(trained.history_iterations),
                "train_loss": list(trained.train_losses),
                "validation_loss": list(trained.validation_losses),
            },
            "metrics": {
                "predicted_delta_s": predicted,
                "theoretical_delta_s": condition["theoretical_delta_s"],
                "continuous_delta_s": condition["continuous_delta_s"],
                "predicted_over_theoretical": predicted / condition["theoretical_delta_s"],
                "predicted_kernel_delta_s": predicted_k.tolist(),
                "theoretical_kernel_delta_s": exact_k.tolist(),
                "continuous_kernel_delta_s": continuous_k.tolist(),
            },
            "data_files": {"train": _relative(train_path), "test": _relative(test_path)},
            "n_test_transitions": int(increments.shape[0]),
            "elapsed_seconds": time.perf_counter() - started,
        }
        path = _checkpoint_path(config, ai, di, repeat)
        _atomic_save(path, checkpoint)
        print(f"[{condition['condition_id']}] saved {path.name}", flush=True)
        del checkpoint, trained, increments, predicted_k
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del train_data, train_video, validation_video, normalization, test_video
    return len(pending)


def _worker(
    config: ExperimentConfig,
    experiment_hash: str,
    conditions: list[tuple[int, int]],
    device_name: str,
    threads: int,
) -> int:
    torch.set_num_threads(threads)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return sum(
        _run_condition(config, experiment_hash, ai, di, device)
        for ai, di in conditions
    )


def _checkpoint_row(
    path: Path,
    config: ExperimentConfig,
    experiment_hash: str,
    ai: int,
    di: int,
    repeat: int,
):
    checkpoint = _load(path)
    expected = (ai + 1, di + 1, repeat + 1, _training_seed(config, repeat))
    condition = checkpoint.get("condition", {})
    actual = (
        condition.get("amplitude_index"),
        condition.get("d_w_index"),
        checkpoint.get("repeat"),
        checkpoint.get("training_seed"),
    )
    if checkpoint.get("experiment_sha256") != experiment_hash or actual != expected:
        raise RuntimeError(f"checkpoint metadata mismatch: {path}")
    metrics = checkpoint["metrics"]
    record = _record(config, ai, di)
    row = {
        **record,
        "repeat": repeat + 1,
        "training_seed": expected[-1],
        "best_iteration": checkpoint["best_iteration"],
        "best_validation_loss": checkpoint["best_validation_loss"],
        "predicted_delta_s": metrics["predicted_delta_s"],
        "predicted_over_theoretical": metrics["predicted_over_theoretical"],
        **{f"predicted_k{i}_delta_s": x for i, x in enumerate(metrics["predicted_kernel_delta_s"])},
        "n_test_transitions": checkpoint["n_test_transitions"],
        "train_data": checkpoint["data_files"]["train"],
        "test_data": checkpoint["data_files"]["test"],
        "checkpoint": _relative(path),
        "elapsed_seconds": checkpoint["elapsed_seconds"],
    }
    return row, checkpoint


def _collect(config: ExperimentConfig, experiment_hash: str):
    runs, losses = [], []
    for ai, di in _indices(config):
        for repeat in range(config.repeats):
            path = _checkpoint_path(config, ai, di, repeat)
            if not path.exists():
                continue
            row, checkpoint = _checkpoint_row(
                path, config, experiment_hash, ai, di, repeat
            )
            runs.append(row)
            history = checkpoint["history"]
            base = {key: row[key] for key in (
                "condition_id", "amplitude", "amplitude_squared", "d_w",
                "repeat", "training_seed",
            )}
            for iteration, train_loss, validation_loss in zip(
                history["iterations"], history["train_loss"], history["validation_loss"]
            ):
                losses.append({
                    **base, "iteration": iteration, "train_loss": train_loss,
                    "validation_loss": validation_loss,
                })
    return runs, losses


def _stats(values) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def _tables(config: ExperimentConfig, runs: list[dict[str, object]]):
    kernel_runs, summary, kernel_summary = [], [], []
    for row in runs:
        for i, kernel in enumerate(KERNEL_NAMES):
            kernel_runs.append(
                {
                    "condition_id": row["condition_id"],
                    "amplitude": row["amplitude"],
                    "amplitude_squared": row["amplitude_squared"],
                    "d_w": row["d_w"],
                    "repeat": row["repeat"],
                    "training_seed": row["training_seed"],
                    "kernel_index": i,
                    "kernel": kernel,
                    "predicted_delta_s": row[f"predicted_k{i}_delta_s"],
                    "theoretical_delta_s": row[f"theoretical_k{i}_delta_s"],
                    "continuous_delta_s": row[f"continuous_k{i}_delta_s"],
                }
            )
    for ai, di in _indices(config):
        condition = _record(config, ai, di)
        selected = [r for r in runs if r["condition_id"] == condition["condition_id"]]
        if not selected:
            continue
        pred_mean, pred_std = _stats(r["predicted_delta_s"] for r in selected)
        ratio_mean, ratio_std = _stats(
            r["predicted_over_theoretical"] for r in selected
        )
        summary.append(
            {
                "condition_id": condition["condition_id"],
                "amplitude": condition["amplitude"],
                "amplitude_squared": condition["amplitude_squared"],
                "d_w": condition["d_w"],
                "n": len(selected),
                "theoretical_delta_s": condition["theoretical_delta_s"],
                "continuous_delta_s": condition["continuous_delta_s"],
                "predicted_delta_s_mean": pred_mean,
                "predicted_delta_s_std": pred_std,
                "predicted_over_theoretical_mean": ratio_mean,
                "predicted_over_theoretical_std": ratio_std,
            }
        )
        for i, kernel in enumerate(KERNEL_NAMES):
            mean, std = _stats(r[f"predicted_k{i}_delta_s"] for r in selected)
            kernel_summary.append(
                {
                    "condition_id": condition["condition_id"],
                    "amplitude": condition["amplitude"],
                    "amplitude_squared": condition["amplitude_squared"],
                    "d_w": condition["d_w"],
                    "kernel_index": i,
                    "kernel": kernel,
                    "n": len(selected),
                    "theoretical_delta_s": condition[f"theoretical_k{i}_delta_s"],
                    "continuous_delta_s": condition[f"continuous_k{i}_delta_s"],
                    "predicted_delta_s_mean": mean,
                    "predicted_delta_s_std": std,
                }
            )
    return kernel_runs, summary, kernel_summary


def _pyplot():
    mpl_dir = OUTPUT_DIR / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _draw_box(axis, values, position: float, width: float, color: str) -> None:
    box = axis.boxplot(
        [values],
        positions=[position],
        widths=width,
        patch_artist=True,
        manage_ticks=False,
        showfliers=False,
    )
    box["boxes"][0].set(facecolor=color, edgecolor=color, alpha=0.25)
    box["medians"][0].set(color=color, linewidth=2.0)
    for item in (*box["whiskers"], *box["caps"]):
        item.set(color=color, linewidth=1.2)
    jitter = np.linspace(-0.18 * width, 0.18 * width, len(values))
    axis.scatter(
        position + jitter,
        values,
        s=14,
        color=color,
        alpha=0.65,
        linewidths=0,
        zorder=3,
    )


def _figures(config: ExperimentConfig, runs, summary, losses) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    base_positions = np.arange(len(config.amplitudes), dtype=float)
    group_width = 0.66
    box_width = group_width / max(len(config.dws), 1) * 0.62
    for di, d_w in enumerate(config.dws):
        rows = sorted(
            (row for row in summary if float(row["d_w"]) == float(d_w)),
            key=lambda row: row["amplitude_squared"],
        )
        if not rows:
            continue
        offset = (di - 0.5 * (len(config.dws) - 1)) * (
            group_width / len(config.dws)
        )
        positions = base_positions + offset
        color = f"C{di}"
        axis.plot(
            positions,
            [row["theoretical_delta_s"] for row in rows],
            color=color,
            linewidth=1.8,
            label=rf"$d_w={d_w:g}$",
        )
        for position, row in zip(positions, rows):
            predictions = [
                float(run["predicted_delta_s"])
                for run in runs
                if run["condition_id"] == row["condition_id"]
            ]
            _draw_box(axis, predictions, position, box_width, color)
    axis.set_xticks(
        base_positions,
        [f"{amplitude**2:g}" for amplitude in config.amplitudes],
    )
    axis.set_xlim(-0.55, len(config.amplitudes) - 0.45)
    axis.set(
        xlabel=r"$A^2$",
        ylabel=r"EP per saved step $\langle\Delta S\rangle$",
    )
    axis.legend(
        frameon=False,
        title="solid: exact theory; boxes/points: 5 training seeds",
    )
    axis.grid(alpha=0.18)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "a2_delta_s.png", dpi=300)
    plt.close(figure)

    kernel_dir = FIGURE_DIR / "kernel_decomposition"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_positions = np.arange(len(KERNEL_NAMES), dtype=float)
    for ai, di in _indices(config):
        condition = _record(config, ai, di)
        condition_runs = [
            run for run in runs if run["condition_id"] == condition["condition_id"]
        ]
        if not condition_runs:
            continue
        figure, axis = plt.subplots(figsize=(6.6, 4.5))
        for kernel_index, position in enumerate(kernel_positions):
            predictions = [
                float(run[f"predicted_k{kernel_index}_delta_s"])
                for run in condition_runs
            ]
            _draw_box(axis, predictions, position, 0.42, "C0")
        theory = [
            condition[f"theoretical_k{index}_delta_s"]
            for index in range(len(KERNEL_NAMES))
        ]
        axis.plot(
            kernel_positions,
            theory,
            color="black",
            marker="D",
            linewidth=1.8,
            markersize=4,
            label="Exact Euler ensemble",
            zorder=4,
        )
        axis.axhline(0.0, color="0.55", linewidth=0.8)
        axis.set_xticks(kernel_positions, KERNEL_NAMES)
        axis.set_ylabel(r"EP contribution per saved step")
        axis.set_title(
            rf"$A={condition['amplitude']:g},\ d_w={condition['d_w']:g}$"
        )
        axis.legend(frameon=False, title="boxes/points: training seeds")
        axis.grid(axis="y", alpha=0.18)
        figure.tight_layout()
        figure.savefig(
            kernel_dir / f"{condition['condition_id']}.png",
            dpi=250,
        )
        plt.close(figure)

    for ai, di in _indices(config):
        condition = _record(config, ai, di)
        rows = [r for r in losses if r["condition_id"] == condition["condition_id"]]
        if not rows:
            continue
        figure, axes = plt.subplots(2, 1, figsize=(6.8, 6.2), sharex=True)
        for repeat in range(config.repeats):
            run = sorted(
                (row for row in rows if int(row["repeat"]) == repeat + 1),
                key=lambda row: row["iteration"],
            )
            if not run:
                continue
            x = [r["iteration"] for r in run]
            label = f"seed {_training_seed(config, repeat)}"
            axes[0].plot(x, [r["train_loss"] for r in run], label=label)
            axes[1].plot(x, [r["validation_loss"] for r in run])
        axes[0].set_ylabel("Raw train loss")
        axes[1].set(xlabel="Iteration", ylabel="Validation loss")
        axes[0].set_title(
            rf"$A={condition['amplitude']:g},\ d_w={condition['d_w']:g}$ "
            "(no smoothing)"
        )
        axes[0].legend(frameon=False, fontsize=8)
        for axis in axes:
            axis.grid(alpha=0.18)
        figure.tight_layout()
        figure.savefig(LOSS_FIGURE_DIR / f"{condition['condition_id']}.png", dpi=220)
        plt.close(figure)


def _write_outputs(config: ExperimentConfig, experiment_hash: str) -> int:
    runs, losses = _collect(config, experiment_hash)
    kernel_runs, summary, kernel_summary = _tables(config, runs)
    for name, rows in (
        ("runs", runs), ("kernel_runs", kernel_runs),
        ("loss_history", losses), ("summary", summary),
        ("kernel_summary", kernel_summary),
    ):
        _write_csv(OUTPUT_DIR / f"{name}.csv", rows)
    if runs:
        _figures(config, runs, summary, losses)
    return len(runs)


def _complete(config: ExperimentConfig, ai: int, di: int) -> bool:
    return all(_data_path(config, ai, di, role).exists() for role in ("train", "test")) and all(
        _checkpoint_path(config, ai, di, repeat).exists()
        for repeat in range(config.repeats)
    )


def _execute(
    config: ExperimentConfig,
    experiment_hash: str,
    devices: tuple[torch.device, ...],
    pending: list[tuple[int, int]],
) -> None:
    if not pending:
        return
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    if len(devices) == 1:
        _worker(config, experiment_hash, pending, str(devices[0]), max(1, allocated))
        return
    chunks = [pending[index:: len(devices)] for index in range(len(devices))]
    context = multiprocessing.get_context("spawn")
    executors, futures = [], []
    try:
        for device, chunk in zip(devices, chunks):
            if not chunk:
                continue
            executor = ProcessPoolExecutor(max_workers=1, mp_context=context)
            executors.append(executor)
            futures.append(
                executor.submit(
                    _worker,
                    config,
                    experiment_hash,
                    chunk,
                    str(device),
                    max(1, allocated // len(devices)),
                )
            )
        for future in futures:
            future.result()
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)


def run(config: ExperimentConfig, devices: tuple[torch.device, ...]) -> None:
    _validate(config)
    if not devices:
        raise ValueError("at least one device is required")
    experiment_hash = _prepare(config)
    pending = [(ai, di) for ai, di in _indices(config) if not _complete(config, ai, di)]
    total = len(config.amplitudes) * len(config.dws) * config.repeats
    bytes_per_condition = (
        (
            config.train_trajectories * config.train_samples
            + config.test_trajectories * config.test_samples
        )
        * 2 * config.saou.lattice_size**2 * 4
    )
    print(f"Devices: {', '.join(map(str, devices))}")
    print(f"Conditions: {len(config.amplitudes)} x {len(config.dws)}; trainings: {total}")
    print(
        f"Trajectory storage: {bytes_per_condition / 1024**3:.2f} GiB per condition; "
        f"{bytes_per_condition * len(config.amplitudes) * len(config.dws) / 1024**3:.1f} GiB total"
    )
    print(f"Output: {OUTPUT_DIR}")
    if any(device.type == "cpu" for device in devices):
        print("WARNING: the full 75-training sweep is intended for CUDA.")
    _execute(config, experiment_hash, devices, pending)
    completed = _write_outputs(config, experiment_hash)
    if completed != total:
        raise RuntimeError(f"study ended with {completed}/{total} checkpoints")
    print(f"Saved {total} checkpoints and figures under {OUTPUT_DIR}")


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _cuda_devices(count: int) -> tuple[torch.device, ...]:
    if count <= 0 or count > torch.cuda.device_count():
        raise ValueError("--num-gpus exceeds the visible positive GPU count")
    return tuple(torch.device(f"cuda:{index}") for index in range(count))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--device",
        default=None,
        help="single PyTorch device (default: auto; examples: cpu, cuda:0)",
    )
    group.add_argument(
        "--num-gpus",
        type=int,
        help="use the first N CUDA devices visible to this job",
    )
    args = parser.parse_args()
    devices = (
        _cuda_devices(args.num_gpus)
        if args.num_gpus is not None
        else (_device(args.device or "auto"),)
    )
    run(ExperimentConfig(), devices)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
