"""Train KNEEP across homogeneous SAOU temperatures and make the figure."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import multiprocessing
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.saou import (
    SAOUConfig,
    exact_epr_increments,
    simulate_trajectories,
    theoretical_epr_rate,
)
from utils.training import TrainingConfig, predict_epr_increments, train_model


OUTPUT_DIR = ROOT / "results" / "saou_temperature"
RUN_FIELDS = (
    "temperature_index",
    "temperature",
    "repeat",
    "train_seed",
    "test_seed",
    "model_seed",
    "best_iteration",
    "best_validation_loss",
    "mse",
    "epr_ratio",
    "predicted_epr_rate",
    "exact_epr_rate",
    "empirical_epr_rate",
    "empirical_epr_ratio",
    "n_test_transitions",
    "elapsed_seconds",
)


@dataclass(frozen=True)
class ExperimentConfig:
    temperatures: tuple[float, ...] = (
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
        1e-1,
    )
    repeats: int = 40
    base_seed: int = 3
    train_trajectories: int = 1_000
    train_samples: int = 1_000
    test_trajectories: int = 1
    test_samples: int = 10_000
    burn_steps: int = 10_000
    n_components: int = 2
    hidden_channels: int = 8
    hidden_layers: int = 2
    max_distance: int = 4
    saou: SAOUConfig = field(default_factory=SAOUConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _scientific_payload(config: ExperimentConfig) -> dict:
    payload = asdict(config)
    payload["seed_rule"] = {
        "run_seed": "base_seed + temperature_index * repeats + repeat_index",
        "train_seed": "run_seed",
        "test_seed": "run_seed + train_trajectories",
        "model_seed": "run_seed + 2 * train_trajectories",
    }
    payload["metrics"] = {
        "mse": "mean((predicted_transition_ep - exact_transition_ep)**2)",
        "epr_ratio": "mean(predicted_transition_ep) / (exact_epr_rate * effective_dt)",
        "error_bars": "sample standard deviation across independent trainings",
    }
    return json.loads(json.dumps(payload))


def _source_hashes() -> dict[str, str]:
    relative_paths = (
        "shell_force.py",
        "models/saou.py",
        "utils/training.py",
        "scripts/saou_temperature.py",
    )
    return {
        relative_path: hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        for relative_path in relative_paths
    }


def _execution_payload(devices: tuple[torch.device, ...]) -> dict[str, object]:
    device_records = []
    for device in devices:
        device_records.append(
            {
                "device": str(device),
                "name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else "CPU"
                ),
            }
        )
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "devices": device_records,
    }


def _prepare_output(
    config: ExperimentConfig, devices: tuple[torch.device, ...]
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "config.json"
    runs_path = OUTPUT_DIR / "runs.csv"
    scientific = _scientific_payload(config)
    scientific["source_sha256"] = _source_hashes()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("experiment") != scientific:
            raise RuntimeError(
                f"{path} contains a different experiment. Move the existing "
                "results directory before starting a new configuration."
            )
        return

    if runs_path.exists():
        raise RuntimeError("runs.csv exists without config.json; refusing to mix results")

    document = {
        "experiment": scientific,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _execution_payload(devices),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_runs(config: ExperimentConfig) -> list[dict[str, str]]:
    path = OUTPUT_DIR / "runs.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RUN_FIELDS:
            raise RuntimeError("runs.csv has an unexpected schema")
        rows = list(reader)

    seen: set[tuple[int, int]] = set()
    for row in rows:
        temperature_index = int(row["temperature_index"])
        repeat = int(row["repeat"])
        if not 0 <= temperature_index < len(config.temperatures):
            raise RuntimeError("runs.csv contains an invalid temperature index")
        if not 1 <= repeat <= config.repeats:
            raise RuntimeError("runs.csv contains an invalid repeat index")
        key = (temperature_index, repeat)
        if key in seen:
            raise RuntimeError(f"runs.csv contains duplicate run {key}")
        seen.add(key)

        expected_seed = (
            config.base_seed
            + temperature_index * config.repeats
            + repeat
            - 1
        )
        expected = (
            config.temperatures[temperature_index],
            expected_seed,
            expected_seed + config.train_trajectories,
            expected_seed + 2 * config.train_trajectories,
        )
        actual = (
            float(row["temperature"]),
            int(row["train_seed"]),
            int(row["test_seed"]),
            int(row["model_seed"]),
        )
        if actual != expected:
            raise RuntimeError(f"runs.csv metadata do not match config for run {key}")
    return rows


def _write_runs(rows: list[dict[str, object]]) -> None:
    path = OUTPUT_DIR / "runs.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _split_train_validation(
    trajectories: torch.Tensor, fraction: float
) -> tuple[torch.Tensor, torch.Tensor]:
    split = int(trajectories.shape[0] * fraction)
    if split <= 0 or split >= trajectories.shape[0]:
        raise ValueError("train_fraction leaves an empty train or validation set")
    return trajectories[:split], trajectories[split:]


def _run_once(
    config: ExperimentConfig,
    temperature_index: int,
    repeat_index: int,
    device: torch.device,
) -> dict[str, object]:
    started = time.perf_counter()
    temperature = config.temperatures[temperature_index]
    run_seed = config.base_seed + temperature_index * config.repeats + repeat_index
    train_seed = run_seed
    test_seed = run_seed + config.train_trajectories
    model_seed = run_seed + 2 * config.train_trajectories
    saou = replace(config.saou, temperature=temperature)

    print("  generating training trajectories")
    train_trajectories = simulate_trajectories(
        saou,
        n_trajectories=config.train_trajectories,
        n_samples=config.train_samples,
        burn_steps=config.burn_steps,
        seed=train_seed,
        simulation_device=device,
        storage_dtype=torch.float32,
    )
    train_video, validation_video = _split_train_validation(
        train_trajectories, config.training.train_fraction
    )
    print("  training and selecting the best validation state")
    trained = train_model(
        train_video,
        validation_video,
        config.training,
        model_seed=model_seed,
        device=device,
        n_components=config.n_components,
        hidden_channels=config.hidden_channels,
        hidden_layers=config.hidden_layers,
        max_distance=config.max_distance,
    )
    del train_video, validation_video, train_trajectories
    gc.collect()

    print("  generating test trajectory and evaluating EP")
    test_video = simulate_trajectories(
        saou,
        n_trajectories=config.test_trajectories,
        n_samples=config.test_samples,
        burn_steps=config.burn_steps,
        seed=test_seed,
        simulation_device=device,
        storage_dtype=torch.float64,
    )
    predicted = predict_epr_increments(
        trained,
        test_video,
        batch_size=config.training.prediction_batch_size,
        device=device,
    )
    exact = exact_epr_increments(
        test_video,
        saou,
        batch_size=config.training.prediction_batch_size,
        device=device,
    )
    if predicted.shape != exact.shape:
        raise RuntimeError("prediction and exact EP arrays are misaligned")

    exact_rate = theoretical_epr_rate(saou)
    predicted_rate = float(predicted.mean() / saou.effective_dt)
    empirical_rate = float(exact.mean() / saou.effective_dt)
    row: dict[str, object] = {
        "temperature_index": temperature_index,
        "temperature": f"{temperature:.12g}",
        "repeat": repeat_index + 1,
        "train_seed": train_seed,
        "test_seed": test_seed,
        "model_seed": model_seed,
        "best_iteration": trained.best_iteration,
        "best_validation_loss": f"{trained.best_validation_loss:.12g}",
        "mse": f"{np.mean((predicted - exact) ** 2):.12g}",
        "epr_ratio": f"{predicted_rate / exact_rate:.12g}",
        "predicted_epr_rate": f"{predicted_rate:.12g}",
        "exact_epr_rate": f"{exact_rate:.12g}",
        "empirical_epr_rate": f"{empirical_rate:.12g}",
        "empirical_epr_ratio": f"{empirical_rate / exact_rate:.12g}",
        "n_test_transitions": predicted.size,
        "elapsed_seconds": f"{time.perf_counter() - started:.3f}",
    }

    del trained, test_video, predicted, exact
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def _run_once_on_device(
    config: ExperimentConfig,
    temperature_index: int,
    repeat_index: int,
    device_name: str,
) -> dict[str, object]:
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return _run_once(config, temperature_index, repeat_index, device)


def _write_summary_and_figure(rows: list[dict[str, object]]) -> None:
    parsed = [
        {
            "temperature": float(row["temperature"]),
            "mse": float(row["mse"]),
            "epr_ratio": float(row["epr_ratio"]),
        }
        for row in rows
    ]
    temperatures = sorted({row["temperature"] for row in parsed})
    summary: list[dict[str, object]] = []
    for temperature in temperatures:
        selected = [row for row in parsed if row["temperature"] == temperature]
        mse = np.asarray([row["mse"] for row in selected])
        ratio = np.asarray([row["epr_ratio"] for row in selected])
        summary.append(
            {
                "temperature": f"{temperature:.12g}",
                "n": len(selected),
                "mse_mean": f"{mse.mean():.12g}",
                "mse_std": f"{mse.std(ddof=1):.12g}" if len(mse) > 1 else "0",
                "epr_ratio_mean": f"{ratio.mean():.12g}",
                "epr_ratio_std": (
                    f"{ratio.std(ddof=1):.12g}" if len(ratio) > 1 else "0"
                ),
            }
        )

    summary_path = OUTPUT_DIR / "summary.csv"
    temporary = summary_path.with_suffix(".csv.tmp")
    fields = (
        "temperature",
        "n",
        "mse_mean",
        "mse_std",
        "epr_ratio_mean",
        "epr_ratio_std",
    )
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    temporary.replace(summary_path)

    x = np.asarray([float(row["temperature"]) for row in summary])
    mse_mean = np.asarray([float(row["mse_mean"]) for row in summary])
    mse_std = np.asarray([float(row["mse_std"]) for row in summary])
    ratio_mean = np.asarray([float(row["epr_ratio_mean"]) for row in summary])
    ratio_std = np.asarray([float(row["epr_ratio_std"]) for row in summary])

    figure, axes = plt.subplots(2, 1, figsize=(7.2, 8.5), sharex=True)
    axes[0].errorbar(
        x,
        mse_mean,
        yerr=mse_std,
        color="tab:blue",
        marker="o",
        linewidth=1.8,
        capsize=4,
        label=r"$\alpha=-0.5$",
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("MSE")
    axes[0].legend(frameon=False)

    axes[1].errorbar(
        x,
        ratio_mean,
        yerr=ratio_std,
        color="tab:blue",
        marker="o",
        linewidth=1.8,
        capsize=4,
    )
    axes[1].axhline(1.0, color="0.55", linestyle="--", linewidth=1.5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$T$")
    axes[1].set_ylabel(r"$\sigma_{\mathrm{pred}}/\sigma$")

    for axis in axes:
        axis.grid(alpha=0.18, which="both")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "temperature_performance.png", dpi=300)
    figure.savefig(OUTPUT_DIR / "temperature_performance.pdf")
    plt.close(figure)


def _sort_rows(rows: list[dict[str, object]]) -> None:
    rows.sort(
        key=lambda row: (int(row["temperature_index"]), int(row["repeat"]))
    )


def _pending_runs(
    config: ExperimentConfig, completed: set[tuple[int, int]]
) -> deque[tuple[int, int]]:
    return deque(
        (temperature_index, repeat_index)
        for temperature_index in range(len(config.temperatures))
        for repeat_index in range(config.repeats)
        if (temperature_index, repeat_index + 1) not in completed
    )


def _run_serial(
    config: ExperimentConfig,
    device: torch.device,
    rows: list[dict[str, object]],
    completed: set[tuple[int, int]],
) -> None:
    total = len(config.temperatures) * config.repeats
    for temperature_index, repeat_index in _pending_runs(config, completed):
        temperature = config.temperatures[temperature_index]
        print(
            f"[{len(completed) + 1}/{total}] T={temperature:g}, "
            f"repeat={repeat_index + 1}/{config.repeats}, device={device}"
        )
        row = _run_once(config, temperature_index, repeat_index, device=device)
        rows.append(row)
        _sort_rows(rows)
        _write_runs(rows)
        completed.add((temperature_index, repeat_index + 1))


def _run_parallel(
    config: ExperimentConfig,
    devices: tuple[torch.device, ...],
    rows: list[dict[str, object]],
    completed: set[tuple[int, int]],
) -> None:
    pending = _pending_runs(config, completed)
    if not pending:
        return

    total = len(config.temperatures) * config.repeats
    context = multiprocessing.get_context("spawn")
    executors = [
        ProcessPoolExecutor(max_workers=1, mp_context=context) for _ in devices
    ]
    active = {}

    def submit(worker_index: int) -> None:
        if not pending:
            return
        temperature_index, repeat_index = pending.popleft()
        device = devices[worker_index]
        temperature = config.temperatures[temperature_index]
        print(
            f"[dispatch {len(completed) + len(active) + 1}/{total}] "
            f"T={temperature:g}, repeat={repeat_index + 1}/{config.repeats} "
            f"-> {device}"
        )
        future = executors[worker_index].submit(
            _run_once_on_device,
            config,
            temperature_index,
            repeat_index,
            str(device),
        )
        active[future] = (worker_index, temperature_index, repeat_index)

    try:
        for worker_index in range(min(len(devices), len(pending))):
            submit(worker_index)

        while active:
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                worker_index, temperature_index, repeat_index = active.pop(future)
                try:
                    row = future.result()
                except Exception as error:
                    device = devices[worker_index]
                    raise RuntimeError(
                        f"T={config.temperatures[temperature_index]:g}, "
                        f"repeat={repeat_index + 1} failed on {device}"
                    ) from error

                rows.append(row)
                _sort_rows(rows)
                _write_runs(rows)
                completed.add((temperature_index, repeat_index + 1))
                print(f"[complete {len(completed)}/{total}] {devices[worker_index]}")
                submit(worker_index)
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)


def run(config: ExperimentConfig, devices: tuple[torch.device, ...]) -> None:
    if not devices:
        raise ValueError("at least one device is required")
    _prepare_output(config, devices)
    rows: list[dict[str, object]] = list(_read_runs(config))
    completed = {
        (int(row["temperature_index"]), int(row["repeat"])) for row in rows
    }
    total = len(config.temperatures) * config.repeats
    train_gib = (
        config.train_trajectories
        * config.train_samples
        * config.n_components
        * config.saou.lattice_size**2
        * 4
        / 1024**3
    )
    print(f"Devices: {', '.join(map(str, devices))}")
    print(
        f"Runs: {total}; train trajectory storage per worker: {train_gib:.2f} GiB"
    )
    print(f"Output: {OUTPUT_DIR}")
    if any(device.type == "cpu" for device in devices):
        print("WARNING: the full 280-run experiment is intended for CUDA execution.")

    if len(devices) == 1:
        _run_serial(config, devices[0], rows, completed)
    else:
        _run_parallel(config, devices, rows, completed)

    if len(rows) != total or len(completed) != total:
        raise RuntimeError("temperature sweep finished with missing or duplicate runs")
    _write_summary_and_figure(rows)
    print(f"Saved temperature results to {OUTPUT_DIR}")


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
            f"--num-gpus={count} was requested, but only {visible} CUDA devices "
            "are visible"
        )
    return tuple(torch.device(f"cuda:{index}") for index in range(count))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--device",
        default=None,
        help="PyTorch device (default: cuda; examples: cpu, cuda, cuda:1, auto)",
    )
    device_group.add_argument(
        "--num-gpus",
        type=int,
        help="run one spawned worker on each of the first N visible CUDA devices",
    )
    args = parser.parse_args()
    devices = (
        _cuda_devices(args.num_gpus)
        if args.num_gpus is not None
        else (_device_from_argument(args.device or "cuda"),)
    )
    run(ExperimentConfig(), devices)


if __name__ == "__main__":
    main()
