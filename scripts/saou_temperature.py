"""Train the SAOU temperature sweep and create its publication figures."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import multiprocessing
import os
import platform
import sys
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
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


OUTPUT_DIR = ROOT / "results" / "saou_temperature"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
SPECTRUM_DIR = FIGURE_DIR / "kernel_spectra"
KERNEL_NAMES = ("local", "r=1", "r=2", "r=3", "r=4")

CONDITION_FIELDS = (
    "condition_id",
    "parameter_index",
    "temperature_index",
    "temperature",
    "omega0",
    "a1",
    "a2",
    "a3",
    "a4",
    "train_data_seed",
    "test_data_seed",
    "theoretical_epr_rate",
    "theoretical_k0_rate",
    "theoretical_k1_rate",
    "theoretical_k2_rate",
    "theoretical_k3_rate",
    "theoretical_k4_rate",
)
RUN_FIELDS = (
    *CONDITION_FIELDS,
    "repeat",
    "training_seed",
    "best_iteration",
    "best_validation_loss",
    "predicted_epr_rate",
    "predicted_over_theoretical",
    "predicted_k0_rate",
    "predicted_k1_rate",
    "predicted_k2_rate",
    "predicted_k3_rate",
    "predicted_k4_rate",
    "n_test_transitions",
    "checkpoint",
    "elapsed_seconds",
)


@dataclass(frozen=True)
class ParameterSet:
    omega0: float
    amplitudes: tuple[float, float, float, float]


@dataclass(frozen=True)
class ExperimentConfig:
    parameters: tuple[ParameterSet, ...] = (
        ParameterSet(3.0, (0.0, 1.0, 1.0, 0.0)),
        ParameterSet(4.0, (1.0, 2.0, 0.0, 1.0)),
        ParameterSet(2.0, (0.0, 1.0, 0.0, 1.0)),
    )
    temperatures: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)
    repeats: int = 40
    base_data_seed: int = 3
    base_training_seed: int = 100_003
    train_trajectories: int = 100
    train_samples: int = 1_000
    test_trajectories: int = 1
    test_samples: int = 10_000
    burn_steps: int = 10_000
    hidden_channels: int = 16
    hidden_layers: int = 3
    activation: str = "relu"
    saou: SAOUConfig = field(default_factory=SAOUConfig)
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            alpha=-0.5,
            iterations=5_000,
            train_batch_size=4_096,
            validation_batch_size=2_048,
            prediction_batch_size=256,
            learning_rate=1e-2,
            weight_decay=1e-3,
            gradient_clip=1.0,
            validate_every=100,
            train_fraction=0.8,
        )
    )


def _validate_config(config: ExperimentConfig) -> None:
    if config.repeats <= 0:
        raise ValueError("repeats must be positive")
    if config.saou.radii != (1, 2, 3, 4):
        raise ValueError("this figure requires the notebook shells (1,2,3,4)")
    if not config.parameters or not config.temperatures:
        raise ValueError("parameters and temperatures must be nonempty")
    if any(temperature <= 0 for temperature in config.temperatures):
        raise ValueError("temperatures must be positive")


def _condition_id(parameter_index: int, temperature_index: int) -> str:
    return f"p{parameter_index + 1:02d}_t{temperature_index + 1:02d}"


def _temperature_token(temperature: float) -> str:
    return f"{temperature:g}".replace("-", "m").replace(".", "p")


def _checkpoint_path(
    config: ExperimentConfig,
    parameter_index: int,
    temperature_index: int,
    repeat_index: int,
) -> Path:
    temperature = config.temperatures[temperature_index]
    return (
        CHECKPOINT_DIR
        / f"p{parameter_index + 1:02d}"
        / f"T_{_temperature_token(temperature)}"
        / f"repeat_{repeat_index + 1:03d}.pt"
    )


def _condition_indices(config: ExperimentConfig) -> list[tuple[int, int]]:
    return [
        (parameter_index, temperature_index)
        for parameter_index in range(len(config.parameters))
        for temperature_index in range(len(config.temperatures))
    ]


def _condition_number(
    config: ExperimentConfig, parameter_index: int, temperature_index: int
) -> int:
    return parameter_index * len(config.temperatures) + temperature_index


def _data_seeds(
    config: ExperimentConfig, parameter_index: int, temperature_index: int
) -> tuple[int, int]:
    condition_number = _condition_number(config, parameter_index, temperature_index)
    train_seed = config.base_data_seed + 2 * condition_number
    return train_seed, train_seed + 1


def _training_seed(
    config: ExperimentConfig,
    parameter_index: int,
    temperature_index: int,
    repeat_index: int,
) -> int:
    condition_number = _condition_number(config, parameter_index, temperature_index)
    return config.base_training_seed + condition_number * config.repeats + repeat_index


def _condition_saou(
    config: ExperimentConfig, parameter_index: int, temperature_index: int
) -> SAOUConfig:
    parameters = config.parameters[parameter_index]
    return replace(
        config.saou,
        omega0=parameters.omega0,
        amplitudes=parameters.amplitudes,
        temperature=config.temperatures[temperature_index],
    )


def _scientific_payload(config: ExperimentConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["trajectory_reuse"] = (
        "one fixed train ensemble and one fixed test trajectory per "
        "(parameter, temperature), reused by every training seed"
    )
    payload["seed_rules"] = {
        "condition_number": "parameter_index * n_temperatures + temperature_index",
        "train_data_seed": "base_data_seed + 2 * condition_number",
        "test_data_seed": "train_data_seed + 1",
        "training_seed": (
            "base_training_seed + condition_number * repeats + repeat_index"
        ),
    }
    payload["theory"] = (
        "continuous-time stationary ensemble average in the learned absolute-shell "
        "basis; independent of sampled trajectories and temperature"
    )
    payload["discretization_note"] = (
        f"Euler-Maruyama with dt={config.saou.dt:g}; its finite-step "
        "stationary process has a small systematic offset from continuous-time theory"
    )
    payload["uncertainty"] = "sample standard deviation across training seeds"
    return json.loads(json.dumps(payload))


def _experiment_hash(scientific: dict[str, object]) -> str:
    encoded = json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_hashes() -> dict[str, str]:
    paths = (
        "shell_force.py",
        "models/saou.py",
        "utils/training.py",
        "scripts/saou_temperature.py",
    )
    return {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths
    }


def _execution_payload(devices: tuple[torch.device, ...]) -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "devices": [str(device) for device in devices],
    }


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_write_csv(
    path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _condition_record(
    config: ExperimentConfig, parameter_index: int, temperature_index: int
) -> dict[str, object]:
    saou = _condition_saou(config, parameter_index, temperature_index)
    train_seed, test_seed = _data_seeds(config, parameter_index, temperature_index)
    components = theoretical_epr_components_absolute(saou)
    total = theoretical_epr_rate(saou)
    if not np.isclose(components.sum(), total, rtol=1e-12, atol=1e-12):
        raise RuntimeError("absolute kernel theory does not sum to total theory")
    return {
        "condition_id": _condition_id(parameter_index, temperature_index),
        "parameter_index": parameter_index + 1,
        "temperature_index": temperature_index + 1,
        "temperature": saou.temperature,
        "omega0": saou.omega0,
        "a1": saou.amplitudes[0],
        "a2": saou.amplitudes[1],
        "a3": saou.amplitudes[2],
        "a4": saou.amplitudes[3],
        "train_data_seed": train_seed,
        "test_data_seed": test_seed,
        "theoretical_epr_rate": total,
        **{
            f"theoretical_k{index}_rate": float(value)
            for index, value in enumerate(components)
        },
    }


def _prepare_output(
    config: ExperimentConfig, devices: tuple[torch.device, ...]
) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    SPECTRUM_DIR.mkdir(parents=True, exist_ok=True)
    scientific = _scientific_payload(config)
    experiment_hash = _experiment_hash(scientific)
    source_hashes = _source_hashes()
    execution = _execution_payload(devices)
    config_path = OUTPUT_DIR / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("experiment") != scientific:
            raise RuntimeError(
                f"{config_path} belongs to a different experiment; move that "
                "directory before starting this sweep"
            )
        if existing.get("experiment_sha256") != experiment_hash:
            raise RuntimeError("config.json has an invalid experiment hash")
        if existing.get("source_sha256") != source_hashes:
            raise RuntimeError(
                "experiment source changed after checkpoints were created; "
                "move the existing results directory before running new code"
            )
        previous_execution = existing.get("environment", {})
        for key in ("python", "torch", "cuda_runtime"):
            if previous_execution.get(key) != execution[key]:
                raise RuntimeError(
                    f"execution environment changed ({key}); move the existing "
                    "results directory before running the new environment"
                )
    else:
        _atomic_write_json(
            config_path,
            {
                "experiment": scientific,
                "experiment_sha256": experiment_hash,
                "source_sha256": source_hashes,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "environment": execution,
            },
        )
    conditions = [
        _condition_record(config, parameter_index, temperature_index)
        for parameter_index, temperature_index in _condition_indices(config)
    ]
    _atomic_write_csv(OUTPUT_DIR / "conditions.csv", CONDITION_FIELDS, conditions)
    return experiment_hash


def _load_checkpoint(path: Path) -> dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _checkpoint_row(
    path: Path,
    config: ExperimentConfig,
    experiment_hash: str,
    parameter_index: int,
    temperature_index: int,
    repeat_index: int,
) -> dict[str, object]:
    checkpoint = _load_checkpoint(path)
    expected_seed = _training_seed(
        config, parameter_index, temperature_index, repeat_index
    )
    if checkpoint.get("format_version") != 1:
        raise RuntimeError(f"unsupported checkpoint format: {path}")
    if checkpoint.get("experiment_sha256") != experiment_hash:
        raise RuntimeError(f"checkpoint belongs to another experiment: {path}")
    condition = checkpoint.get("condition", {})
    expected = (
        parameter_index + 1,
        temperature_index + 1,
        repeat_index + 1,
        expected_seed,
    )
    actual = (
        condition.get("parameter_index"),
        condition.get("temperature_index"),
        checkpoint.get("repeat"),
        checkpoint.get("training_seed"),
    )
    if actual != expected:
        raise RuntimeError(f"checkpoint metadata do not match its path: {path}")
    record = _condition_record(config, parameter_index, temperature_index)
    metrics = checkpoint["metrics"]
    try:
        checkpoint_name = path.relative_to(ROOT).as_posix()
    except ValueError:
        checkpoint_name = str(path)
    return {
        **record,
        "repeat": repeat_index + 1,
        "training_seed": expected_seed,
        "best_iteration": checkpoint["best_iteration"],
        "best_validation_loss": checkpoint["best_validation_loss"],
        "predicted_epr_rate": metrics["predicted_epr_rate"],
        "predicted_over_theoretical": metrics["predicted_over_theoretical"],
        **{
            f"predicted_k{index}_rate": value
            for index, value in enumerate(metrics["predicted_kernel_epr_rates"])
        },
        "n_test_transitions": checkpoint["n_test_transitions"],
        "checkpoint": checkpoint_name,
        "elapsed_seconds": checkpoint["elapsed_seconds"],
    }


def _collect_rows(
    config: ExperimentConfig, experiment_hash: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for parameter_index, temperature_index in _condition_indices(config):
        for repeat_index in range(config.repeats):
            path = _checkpoint_path(
                config, parameter_index, temperature_index, repeat_index
            )
            if path.exists():
                rows.append(
                    _checkpoint_row(
                        path,
                        config,
                        experiment_hash,
                        parameter_index,
                        temperature_index,
                        repeat_index,
                    )
                )
    rows.sort(
        key=lambda row: (
            int(row["parameter_index"]),
            int(row["temperature_index"]),
            int(row["repeat"]),
        )
    )
    return rows


def _split_train_validation(
    trajectories: torch.Tensor, fraction: float
) -> tuple[torch.Tensor, torch.Tensor]:
    split = int(trajectories.shape[0] * fraction)
    if split <= 0 or split >= trajectories.shape[0]:
        raise ValueError("train_fraction leaves an empty split")
    return trajectories[:split], trajectories[split:]


def _atomic_save_checkpoint(path: Path, checkpoint: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _run_condition(
    config: ExperimentConfig,
    experiment_hash: str,
    parameter_index: int,
    temperature_index: int,
    device: torch.device,
) -> int:
    pending = [
        repeat_index
        for repeat_index in range(config.repeats)
        if not _checkpoint_path(
            config, parameter_index, temperature_index, repeat_index
        ).exists()
    ]
    if not pending:
        return 0
    condition = _condition_record(config, parameter_index, temperature_index)
    saou = _condition_saou(config, parameter_index, temperature_index)
    print(
        f"[{condition['condition_id']}] generating fixed train/test data on {device}",
        flush=True,
    )
    train_data = simulate_trajectories(
        saou,
        n_trajectories=config.train_trajectories,
        n_samples=config.train_samples,
        burn_steps=config.burn_steps,
        seed=int(condition["train_data_seed"]),
        simulation_device=device,
        storage_dtype=torch.float32,
    )
    train_video, validation_video = _split_train_validation(
        train_data, config.training.train_fraction
    )
    normalization = channel_normalization(train_video)
    test_video = simulate_trajectories(
        saou,
        n_trajectories=config.test_trajectories,
        n_samples=config.test_samples,
        burn_steps=config.burn_steps,
        seed=int(condition["test_data_seed"]),
        simulation_device=device,
        storage_dtype=torch.float32,
    )
    theoretical_components = np.asarray(
        [condition[f"theoretical_k{index}_rate"] for index in range(5)],
        dtype=np.float64,
    )
    theoretical_total = float(condition["theoretical_epr_rate"])
    completed = 0
    for repeat_index in pending:
        started = time.perf_counter()
        training_seed = _training_seed(
            config, parameter_index, temperature_index, repeat_index
        )
        print(
            f"[{condition['condition_id']}] training "
            f"{repeat_index + 1}/{config.repeats} (seed={training_seed})",
            flush=True,
        )
        trained = train_model(
            train_video,
            validation_video,
            config.training,
            model_seed=training_seed,
            device=device,
            n_components=2,
            hidden_channels=config.hidden_channels,
            hidden_layers=config.hidden_layers,
            max_distance=4,
            activation=config.activation,
            normalization=normalization,
            progress=False,
        )
        branch_increments = predict_epr_branch_increments(
            trained,
            test_video,
            batch_size=config.training.prediction_batch_size,
            device=device,
        )
        branch_rates = branch_increments.mean(axis=0, dtype=np.float64) / saou.effective_dt
        predicted_total = float(branch_rates.sum())
        if branch_rates.shape != theoretical_components.shape:
            raise RuntimeError("model branches and analytic kernels are misaligned")
        if not np.all(np.isfinite(branch_rates)):
            raise RuntimeError("non-finite EPR prediction")
        checkpoint = {
            "format_version": 1,
            "experiment_sha256": experiment_hash,
            "condition": condition,
            "repeat": repeat_index + 1,
            "training_seed": training_seed,
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
                "predicted_epr_rate": predicted_total,
                "theoretical_epr_rate": theoretical_total,
                "predicted_over_theoretical": predicted_total / theoretical_total,
                "predicted_kernel_epr_rates": branch_rates.tolist(),
                "theoretical_kernel_epr_rates": theoretical_components.tolist(),
            },
            "n_test_transitions": int(branch_increments.shape[0]),
            "elapsed_seconds": time.perf_counter() - started,
        }
        path = _checkpoint_path(
            config, parameter_index, temperature_index, repeat_index
        )
        _atomic_save_checkpoint(path, checkpoint)
        completed += 1
        print(
            f"[{condition['condition_id']}] saved {path.name}; "
            f"sigma_pred={predicted_total:.6g}",
            flush=True,
        )
        del checkpoint, trained, branch_increments, branch_rates
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del test_video, normalization, train_video, validation_video, train_data
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return completed


def _run_condition_on_device(
    config: ExperimentConfig,
    experiment_hash: str,
    parameter_index: int,
    temperature_index: int,
    device_name: str,
    cpu_threads: int,
) -> int:
    torch.set_num_threads(cpu_threads)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return _run_condition(
        config,
        experiment_hash,
        parameter_index,
        temperature_index,
        device,
    )


def _condition_is_complete(
    config: ExperimentConfig, parameter_index: int, temperature_index: int
) -> bool:
    return all(
        _checkpoint_path(config, parameter_index, temperature_index, repeat_index).exists()
        for repeat_index in range(config.repeats)
    )


def _write_run_tables(rows: list[dict[str, object]]) -> None:
    _atomic_write_csv(OUTPUT_DIR / "runs.csv", RUN_FIELDS, rows)
    kernel_rows: list[dict[str, object]] = []
    fields = (
        "condition_id",
        "parameter_index",
        "temperature_index",
        "temperature",
        "repeat",
        "training_seed",
        "kernel_index",
        "kernel",
        "predicted_epr_rate",
        "theoretical_epr_rate",
    )
    for row in rows:
        for kernel_index, kernel in enumerate(KERNEL_NAMES):
            kernel_rows.append(
                {
                    "condition_id": row["condition_id"],
                    "parameter_index": row["parameter_index"],
                    "temperature_index": row["temperature_index"],
                    "temperature": row["temperature"],
                    "repeat": row["repeat"],
                    "training_seed": row["training_seed"],
                    "kernel_index": kernel_index,
                    "kernel": kernel,
                    "predicted_epr_rate": row[f"predicted_k{kernel_index}_rate"],
                    "theoretical_epr_rate": row[
                        f"theoretical_k{kernel_index}_rate"
                    ],
                }
            )
    _atomic_write_csv(OUTPUT_DIR / "kernel_runs.csv", fields, kernel_rows)


def _summary_rows(
    config: ExperimentConfig, rows: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary: list[dict[str, object]] = []
    kernel_summary: list[dict[str, object]] = []
    for parameter_index, temperature_index in _condition_indices(config):
        selected = [
            row
            for row in rows
            if int(row["parameter_index"]) == parameter_index + 1
            and int(row["temperature_index"]) == temperature_index + 1
        ]
        if not selected:
            continue
        condition = _condition_record(config, parameter_index, temperature_index)
        predicted = np.asarray(
            [float(row["predicted_epr_rate"]) for row in selected]
        )
        ratios = np.asarray(
            [float(row["predicted_over_theoretical"]) for row in selected]
        )
        summary.append(
            {
                "condition_id": condition["condition_id"],
                "parameter_index": parameter_index + 1,
                "temperature_index": temperature_index + 1,
                "temperature": condition["temperature"],
                "n": len(selected),
                "theoretical_epr_rate": condition["theoretical_epr_rate"],
                "predicted_epr_mean": float(predicted.mean()),
                "predicted_epr_std": (
                    float(predicted.std(ddof=1)) if len(predicted) > 1 else 0.0
                ),
                "predicted_over_theoretical_mean": float(ratios.mean()),
                "predicted_over_theoretical_std": (
                    float(ratios.std(ddof=1)) if len(ratios) > 1 else 0.0
                ),
            }
        )
        for kernel_index, kernel in enumerate(KERNEL_NAMES):
            values = np.asarray(
                [float(row[f"predicted_k{kernel_index}_rate"]) for row in selected]
            )
            kernel_summary.append(
                {
                    "condition_id": condition["condition_id"],
                    "parameter_index": parameter_index + 1,
                    "temperature_index": temperature_index + 1,
                    "temperature": condition["temperature"],
                    "kernel_index": kernel_index,
                    "kernel": kernel,
                    "n": len(selected),
                    "theoretical_epr_rate": condition[
                        f"theoretical_k{kernel_index}_rate"
                    ],
                    "predicted_epr_mean": float(values.mean()),
                    "predicted_epr_std": (
                        float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    ),
                }
            )
    return summary, kernel_summary


def _write_summaries(
    config: ExperimentConfig, rows: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary, kernel_summary = _summary_rows(config, rows)
    summary_fields = (
        "condition_id",
        "parameter_index",
        "temperature_index",
        "temperature",
        "n",
        "theoretical_epr_rate",
        "predicted_epr_mean",
        "predicted_epr_std",
        "predicted_over_theoretical_mean",
        "predicted_over_theoretical_std",
    )
    kernel_fields = (
        "condition_id",
        "parameter_index",
        "temperature_index",
        "temperature",
        "kernel_index",
        "kernel",
        "n",
        "theoretical_epr_rate",
        "predicted_epr_mean",
        "predicted_epr_std",
    )
    _atomic_write_csv(OUTPUT_DIR / "summary.csv", summary_fields, summary)
    _atomic_write_csv(OUTPUT_DIR / "kernel_summary.csv", kernel_fields, kernel_summary)
    return summary, kernel_summary


def _matplotlib_pyplot():
    config_dir = OUTPUT_DIR / ".matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _make_figures(
    config: ExperimentConfig,
    summary: list[dict[str, object]],
    kernel_summary: list[dict[str, object]],
) -> None:
    plt = _matplotlib_pyplot()
    colors = ("tab:blue", "tab:orange", "tab:green")
    markers = ("o", "v", "s")
    figure, axes = plt.subplots(2, 1, figsize=(7.4, 8.2), sharex=True)
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
        ratio = np.asarray(
            [float(row["predicted_over_theoretical_mean"]) for row in selected]
        )
        ratio_std = np.asarray(
            [float(row["predicted_over_theoretical_std"]) for row in selected]
        )
        amplitude_text = ",".join(f"{value:g}" for value in parameters.amplitudes)
        label = rf"$P_{parameter_index + 1}=({parameters.omega0:g},{amplitude_text})$"
        axes[0].plot(x, theory, color=colors[parameter_index], linewidth=2.0)
        axes[0].errorbar(
            x,
            predicted,
            yerr=predicted_std,
            color=colors[parameter_index],
            marker=markers[parameter_index],
            linestyle="--",
            linewidth=1.5,
            capsize=3,
            label=label,
        )
        axes[1].errorbar(
            x,
            ratio,
            yerr=ratio_std,
            color=colors[parameter_index],
            marker=markers[parameter_index],
            linestyle="--",
            linewidth=1.5,
            capsize=3,
        )
    axes[0].set_ylabel(r"Entropy production rate $\sigma$")
    axes[0].legend(
        frameon=False,
        title=r"solid: theory; markers: KNEEP ($\alpha=-0.5$)",
    )
    axes[1].axhline(1.0, color="0.55", linestyle="--", linewidth=1.5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"Temperature $T$")
    axes[1].set_ylabel(r"$\sigma_{\rm pred}/\sigma_{\rm true}$")
    for axis in axes:
        axis.grid(alpha=0.18, which="both")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "temperature_performance.png", dpi=300)
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
        figure, axis = plt.subplots(figsize=(6.6, 4.4))
        width = 0.36
        axis.bar(
            x - width / 2,
            theory,
            width,
            facecolor="white",
            edgecolor="black",
            linewidth=1.2,
            label="Theory",
        )
        axis.bar(
            x + width / 2,
            predicted,
            width,
            yerr=predicted_std,
            color="tab:blue",
            alpha=0.8,
            capsize=3,
            label="KNEEP",
        )
        parameters = config.parameters[parameter_index]
        temperature = config.temperatures[temperature_index]
        parameter_text = ", ".join(
            f"{value:g}" for value in (parameters.omega0, *parameters.amplitudes)
        )
        axis.set_title(
            rf"$P_{parameter_index + 1}=({parameter_text}),\quad T={temperature:g}$"
        )
        axis.set_xticks(x, KERNEL_NAMES)
        axis.set_ylabel("Kernel EPR rate")
        axis.axhline(0.0, color="0.35", linewidth=0.8)
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=0.18)
        figure.tight_layout()
        figure.savefig(
            SPECTRUM_DIR
            / f"p{parameter_index + 1:02d}_T_{_temperature_token(temperature)}.png",
            dpi=300,
        )
        plt.close(figure)


def _write_derived_outputs(
    config: ExperimentConfig, rows: list[dict[str, object]], make_figures: bool
) -> None:
    _write_run_tables(rows)
    summary, kernel_summary = _write_summaries(config, rows)
    if make_figures:
        _make_figures(config, summary, kernel_summary)


def _allocated_cpu_threads(n_workers: int) -> int:
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    return max(1, allocated // n_workers)


def _run_serial(
    config: ExperimentConfig,
    experiment_hash: str,
    device: torch.device,
    pending: list[tuple[int, int]],
) -> None:
    torch.set_num_threads(_allocated_cpu_threads(1))
    for parameter_index, temperature_index in pending:
        _run_condition(
            config,
            experiment_hash,
            parameter_index,
            temperature_index,
            device,
        )
        rows = _collect_rows(config, experiment_hash)
        _write_derived_outputs(config, rows, make_figures=False)


def _run_parallel(
    config: ExperimentConfig,
    experiment_hash: str,
    devices: tuple[torch.device, ...],
    pending_conditions: list[tuple[int, int]],
) -> None:
    pending = deque(pending_conditions)
    context = multiprocessing.get_context("spawn")
    executors = [
        ProcessPoolExecutor(max_workers=1, mp_context=context) for _ in devices
    ]
    cpu_threads = _allocated_cpu_threads(len(devices))
    active: dict[object, tuple[int, int, int]] = {}

    def submit(worker_index: int) -> None:
        if not pending:
            return
        parameter_index, temperature_index = pending.popleft()
        device = devices[worker_index]
        print(
            f"[dispatch] {_condition_id(parameter_index, temperature_index)} -> {device}",
            flush=True,
        )
        future = executors[worker_index].submit(
            _run_condition_on_device,
            config,
            experiment_hash,
            parameter_index,
            temperature_index,
            str(device),
            cpu_threads,
        )
        active[future] = (worker_index, parameter_index, temperature_index)

    try:
        for worker_index in range(min(len(devices), len(pending))):
            submit(worker_index)
        while active:
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                worker_index, parameter_index, temperature_index = active.pop(future)
                try:
                    trained_count = future.result()
                except Exception as error:
                    raise RuntimeError(
                        f"{_condition_id(parameter_index, temperature_index)} "
                        f"failed on {devices[worker_index]}"
                    ) from error
                print(
                    f"[complete] {_condition_id(parameter_index, temperature_index)} "
                    f"({trained_count} new checkpoints)",
                    flush=True,
                )
                rows = _collect_rows(config, experiment_hash)
                _write_derived_outputs(config, rows, make_figures=False)
                submit(worker_index)
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)


def run(config: ExperimentConfig, devices: tuple[torch.device, ...]) -> None:
    _validate_config(config)
    if not devices:
        raise ValueError("at least one device is required")
    experiment_hash = _prepare_output(config, devices)
    rows = _collect_rows(config, experiment_hash)
    _write_derived_outputs(config, rows, make_figures=False)
    pending = [
        condition
        for condition in _condition_indices(config)
        if not _condition_is_complete(config, *condition)
    ]
    total_trainings = len(config.parameters) * len(config.temperatures) * config.repeats
    stored_gib = (
        (
            config.train_trajectories * config.train_samples
            + config.test_trajectories * config.test_samples
        )
        * 2
        * config.saou.lattice_size**2
        * 4
        / 1024**3
    )
    print(f"Devices: {', '.join(map(str, devices))}")
    print(
        f"Conditions: {len(config.parameters)} x {len(config.temperatures)}; "
        f"trainings: {total_trainings}; completed: {len(rows)}"
    )
    print(f"Fixed trajectory storage per worker: {stored_gib:.2f} GiB")
    print(f"Output: {OUTPUT_DIR}")
    if any(device.type == "cpu" for device in devices):
        print("WARNING: the full 600-training sweep is intended for CUDA.")
    if len(devices) == 1:
        _run_serial(config, experiment_hash, devices[0], pending)
    else:
        _run_parallel(config, experiment_hash, devices, pending)
    rows = _collect_rows(config, experiment_hash)
    if len(rows) != total_trainings:
        raise RuntimeError(
            f"sweep ended with {len(rows)}/{total_trainings} checkpoints"
        )
    _write_derived_outputs(config, rows, make_figures=True)
    figure_count = 1 + len(config.parameters) * len(config.temperatures)
    print(
        f"Saved tables, {total_trainings} checkpoints, and "
        f"{figure_count} figures under {OUTPUT_DIR}"
    )


def _device_from_argument(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train KNEEP for the fixed SAOU temperature experiment."
    )
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--device",
        default=None,
        help="single PyTorch device (default: cuda; examples: cpu, cuda:1, auto)",
    )
    device_group.add_argument(
        "--num-gpus",
        type=int,
        help="one condition worker on each of the first N visible CUDA devices",
    )
    args = parser.parse_args()
    devices = (
        _cuda_devices(args.num_gpus)
        if args.num_gpus is not None
        else (_device_from_argument(args.device or "cuda"),)
    )
    run(ExperimentConfig(), devices)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
