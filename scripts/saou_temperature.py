"""Train the SAOU temperature sweep and save trajectories and result tables."""

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
DATA_DIR = OUTPUT_DIR / "data"
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
    "train_data",
    "test_data",
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
        # Original couplings scaled by 0.3.
        ParameterSet(0.9, (0.0, 0.3, 0.3, 0.0)),
        ParameterSet(1.2, (0.3, 0.6, 0.0, 0.3)),
        ParameterSet(0.6, (0.0, 0.3, 0.0, 0.3)),
    )
    temperatures: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)
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
    saou: SAOUConfig = field(default_factory=SAOUConfig)
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            alpha=-0.5,
            iterations=3_000,
            train_batch_size=512,
            validation_batch_size=512,
            prediction_batch_size=256,
            learning_rate=1e-3,
            weight_decay=1e-4,
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
    train_seed = config.base_data_seed + condition_number * (
        config.train_trajectories + config.test_trajectories
    )
    return train_seed, train_seed + config.train_trajectories


def _training_seed(
    config: ExperimentConfig,
    parameter_index: int,
    temperature_index: int,
    repeat_index: int,
) -> int:
    return config.base_training_seed + repeat_index


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
        "train_data_seed": "base_data_seed + condition_number * (train_trajectories + test_trajectories)",
        "test_data_seed": "train_data_seed + train_trajectories",
        "training_seed": (
            "base_training_seed + repeat_index"
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
    DATA_DIR.mkdir(parents=True, exist_ok=True)
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
        "train_data": checkpoint["data_files"]["train"],
        "test_data": checkpoint["data_files"]["test"],
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


def _relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


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


def _data_path(config: ExperimentConfig, ai: int, di: int, role: str) -> Path:
    return (DATA_DIR / f"p{ai + 1:02d}"
            / f"T_{_temperature_token(config.temperatures[di])}" / f"{role}.pt")


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
    condition = _condition_id(ai, di)
    if path.exists():
        payload = _load(path, mmap=True)
        expected_metadata = {
            "format_version": 1, "experiment_sha256": experiment_hash,
            "condition_id": condition, "role": role, "seed": seed,
        }
        if not isinstance(payload, dict) or any(
            payload.get(key) != value for key, value in expected_metadata.items()
        ):
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
        _atomic_save_checkpoint(path, payload)
    expected = (n_trajectories, n_samples, 2, config.saou.lattice_size, config.saou.lattice_size)
    if not isinstance(trajectories, torch.Tensor) or tuple(trajectories.shape) != expected:
        raise RuntimeError(f"invalid trajectory tensor: {path}")
    if trajectories.dtype != torch.float32:
        raise RuntimeError(f"invalid trajectory dtype: {path}")
    return trajectories


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
    missing_data = [
        _data_path(config, parameter_index, temperature_index, role)
        for role in ("train", "test")
        if not _data_path(config, parameter_index, temperature_index, role).exists()
    ]
    if len(pending) < config.repeats and missing_data:
        raise RuntimeError(
            "trajectory data are missing for existing checkpoints: "
            + ", ".join(map(str, missing_data))
        )
    if not pending:
        return 0
    condition = _condition_record(config, parameter_index, temperature_index)
    saou = _condition_saou(config, parameter_index, temperature_index)
    train_data = _trajectory(
        config, experiment_hash, parameter_index, temperature_index, "train", device
    )
    train_video, validation_video = _split_train_validation(
        train_data, config.training.train_fraction
    )
    normalization = channel_normalization(train_video)
    test_video = _trajectory(
        config, experiment_hash, parameter_index, temperature_index, "test", device
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
            "data_files": {
                role: _relative(_data_path(config, parameter_index, temperature_index, role))
                for role in ("train", "test")
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
        _data_path(config, parameter_index, temperature_index, role).exists()
        for role in ("train", "test")
    ) and all(
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


def _write_derived_outputs(
    config: ExperimentConfig, rows: list[dict[str, object]]
) -> None:
    _write_run_tables(rows)
    _write_summaries(config, rows)
    fields = (
        "condition_id", "parameter_index", "temperature_index", "temperature",
        "repeat", "training_seed", "iteration", "train_loss", "validation_loss",
    )
    losses = []
    for row in rows:
        history = _load_checkpoint(ROOT / str(row["checkpoint"]))["history"]
        base = {key: row[key] for key in fields[:6]}
        for iteration, train_loss, validation_loss in zip(
            history["iterations"], history["train_loss"], history["validation_loss"]
        ):
            losses.append({
                **base, "iteration": iteration, "train_loss": train_loss,
                "validation_loss": validation_loss,
            })
    _atomic_write_csv(OUTPUT_DIR / "loss_history.csv", fields, losses)


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
        _write_derived_outputs(config, rows)


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
                _write_derived_outputs(config, rows)
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
    _write_derived_outputs(config, rows)
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
    print(
        f"Trajectory storage: {stored_gib:.2f} GiB per condition; "
        f"{stored_gib * len(config.parameters) * len(config.temperatures):.1f} GiB total"
    )
    print(f"Output: {OUTPUT_DIR}")
    if any(device.type == "cpu" for device in devices):
        print("WARNING: the full temperature sweep is intended for CUDA.")
    if len(devices) == 1:
        _run_serial(config, experiment_hash, devices[0], pending)
    else:
        _run_parallel(config, experiment_hash, devices, pending)
    rows = _collect_rows(config, experiment_hash)
    if len(rows) != total_trainings:
        raise RuntimeError(
            f"sweep ended with {len(rows)}/{total_trainings} checkpoints"
        )
    _write_derived_outputs(config, rows)
    print(f"Saved tables and {total_trainings} checkpoints under {OUTPUT_DIR}")


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
