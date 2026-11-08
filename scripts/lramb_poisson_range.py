"""Separate generation, unsupervised fitting and evaluation of a Poisson range sweep.

Run ``python KNEEP/scripts/lramb_poisson_range.py --help`` from the workspace.
Only the generate command needs the sibling LRAMB repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import multiprocessing
import sys
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from models.lramb_poisson import (
    condition_id,
    dataset_preset,
    finalize_generation,
    generate_datasets,
    generate_trajectory_task,
    load_dataset,
    open_video_dataset,
    prepare_generation,
    read_manifest,
)
from shell_force import ShellForceKNEEP2D
from utils.training import TrainingConfig, TrainingResult, train_model


@dataclass(frozen=True)
class FitConfig:
    max_distance: int = 8
    hidden_channels: int = 32
    hidden_layers: int = 2
    repeats: int = 3
    base_seed: int = 1701
    activation: str = "elu"
    training: TrainingConfig = field(default_factory=lambda: TrainingConfig(
        iterations=2_000, train_batch_size=32, validation_batch_size=64,
        prediction_batch_size=32, validate_every=100, weight_decay=1e-4,
    ))


def fit_preset(name: str) -> FitConfig:
    if name == "quick":
        return FitConfig(
            max_distance=2, hidden_channels=4, hidden_layers=1, repeats=2,
            training=TrainingConfig(iterations=4, train_batch_size=4,
                validation_batch_size=16, prediction_batch_size=16, validate_every=2),
        )
    if name == "demo":
        return FitConfig()
    if name == "refined":
        # Tune sampling/optimization without changing the force function class.
        # Full validation is larger at sample_every=1, so evaluate less often.
        return FitConfig(training=TrainingConfig(
            iterations=20_000, train_batch_size=128, validation_batch_size=128,
            prediction_batch_size=64, learning_rate=3e-4, weight_decay=1e-5,
            validate_every=500,
        ))
    if name == "production":
        return FitConfig(max_distance=12, training=TrainingConfig(
            iterations=10_000, train_batch_size=1024,
            train_micro_batch_size=128, validation_batch_size=128,
            prediction_batch_size=128, learning_rate=1e-3, weight_decay=1e-4,
            validate_every=500,
        ))
    if name == "reference":
        return FitConfig(hidden_layers=3, repeats=5,
            training=replace(FitConfig().training, iterations=5_000))
    raise ValueError("preset must be quick, demo, refined, production or reference")


def _device(value: str | torch.device) -> torch.device:
    value = str(value)
    target = torch.device("cuda" if value == "auto" and torch.cuda.is_available()
                          else "cpu" if value == "auto" else value)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda:0")
    return target


def _hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _source_hash() -> str:
    return hashlib.sha256(b"".join(path.read_bytes() for path in (
        Path(__file__), ROOT / "shell_force.py", ROOT / "utils" / "training.py",
        ROOT / "utils" / "video_data.py",
    ))).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def _complete_dataset(output_dir: Path) -> dict:
    manifest = read_manifest(output_dir)
    if not manifest.get("complete"):
        raise RuntimeError("dataset generation is incomplete; run the generate stage first")
    return manifest


def _validate_fit(config: FitConfig, n: int) -> None:
    if not 0 <= config.max_distance < n / 2:
        raise ValueError("max_distance must be nonnegative and smaller than n/2")
    if min(config.hidden_channels, config.hidden_layers, config.repeats) < 1:
        raise ValueError("hidden sizes and repeats must be positive")
    t = config.training
    if min(t.iterations, t.train_batch_size, t.validation_batch_size,
           t.prediction_batch_size, t.validate_every) < 1:
        raise ValueError("training iteration and batch counts must be positive")
    if t.train_micro_batch_size is not None and t.train_micro_batch_size < 1:
        raise ValueError("train_micro_batch_size must be positive")
    if not all(math.isfinite(value) for value in (
        t.alpha, t.learning_rate, t.weight_decay, t.gradient_clip,
    )) or t.alpha == -1 or t.learning_rate <= 0 or t.weight_decay < 0 or t.gradient_clip <= 0:
        raise ValueError("invalid training loss or optimizer settings")


def _condition_label(condition: dict) -> str:
    if condition.get("is_control") or float(condition["ell"]) == 0:
        return "g=0 (no long-range)"
    return f"ell={float(condition['ell']):g}"


def _worker_task(task: dict, device: str, cpu_threads: int):
    """Top-level spawn entry; each process uses only its assigned GPU."""
    torch.set_num_threads(cpu_threads)
    target = _device(device)
    if target.type == "cuda":
        torch.cuda.set_device(target)
    if task["stage"] == "generate":
        return generate_trajectory_task(task["payload"], device=str(target),
                                        progress=task.get("progress", True))
    if task["stage"] == "train":
        return _train_repeat(task, target)
    if task["stage"] == "evaluate":
        return _evaluate_condition(task, target)
    raise ValueError(f"unknown worker stage: {task['stage']}")


def _run_tasks(tasks: list[dict], devices, *, cpu_threads=1, progress=True):
    """One persistent spawned process per device; parent owns manifests."""
    devices = tuple(str(_device(value)) for value in devices)
    if not devices or cpu_threads < 1:
        raise ValueError("supply at least one device and positive CPU threads")
    cuda_devices = [value for value in devices if torch.device(value).type == "cuda"]
    if len(cuda_devices) != len(set(cuda_devices)):
        raise ValueError("only one experiment worker may be assigned to each CUDA device")
    if not tasks:
        return []
    if len(devices) == 1:
        return [_worker_task(task, devices[0], cpu_threads) for task in tasks]
    pending = deque(enumerate(tasks))
    context = multiprocessing.get_context("spawn")
    executors = [ProcessPoolExecutor(max_workers=1, mp_context=context)
                 for _ in devices]
    active, results = {}, [None] * len(tasks)
    completed = 0
    aborted = False

    def submit(worker):
        if pending:
            index, task = pending.popleft()
            future = executors[worker].submit(_worker_task, task, devices[worker], cpu_threads)
            active[future] = worker, index, task
            if progress:
                print(f"[dispatch {index + 1}/{len(tasks)}] "
                      f"{task.get('label', task['stage'])} -> {devices[worker]}", flush=True)

    try:
        for worker in range(min(len(devices), len(tasks))):
            submit(worker)
        while active:
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                worker, index, task = active.pop(future)
                try:
                    results[index] = future.result()
                except Exception as error:
                    for other in active:
                        other.cancel()
                    raise RuntimeError(
                        f"{task.get('label', task['stage'])} failed on {devices[worker]}"
                    ) from error
                completed += 1
                if progress:
                    print(f"[complete {completed}/{len(tasks)}] "
                          f"{task.get('label', task['stage'])}", flush=True)
                submit(worker)
    except BaseException:
        aborted = True
        for future in active:
            future.cancel()
        for executor in executors:
            # Python 3.14 provides public termination. Older interpreters
            # require terminating the processes owned by this one pool.
            terminate = getattr(executor, "terminate_workers", None)
            if terminate is not None:
                terminate()
            else:
                processes = tuple((getattr(executor, "_processes", None) or {}).values())
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                executor.shutdown(wait=True, cancel_futures=True)
        raise
    finally:
        if not aborted:
            for executor in executors:
                executor.shutdown(wait=True, cancel_futures=True)
    return results


def generate_sweep(config, output_dir, *, lramb_root=None, device="auto",
                   devices=None, cpu_threads=1, progress=True, force=False):
    targets = tuple(devices) if devices is not None else (str(_device(device)),)
    if not targets:
        raise ValueError("supply at least one execution device")
    manifest = prepare_generation(config, output_dir, lramb_root=lramb_root,
                                  device=targets[0], force=force)
    if progress:
        print(f"generator={manifest['generator_revision']}; "
              f"LRAMB={manifest['source']['lramb_root']}; backend={manifest['backend']}; "
              f"train pairs/condition={config.train_pairs:,}; torch={torch.__version__}", flush=True)
    tasks = [dict(stage="generate", payload=item, progress=progress,
                  label=f"generate {item['condition_id']}/{item['split']}/{item['trajectory_index']}")
             for item in manifest["trajectory_tasks"]]
    _run_tasks(tasks, targets, cpu_threads=cpu_threads, progress=progress)
    return finalize_generation(output_dir)


def _train_repeat(task: dict, target: torch.device) -> dict:
    config = FitConfig(**{**task["config"],
        "training": TrainingConfig(**task["config"]["training"])})
    folder, condition = Path(task["output_dir"]), task["condition"]
    ell, repeat = float(condition["ell"]), task["repeat"]
    path = Path(task["checkpoint"])
    if path.exists() and not task["force"]:
        if _load_checkpoint(path).get("signature") != task["signature"]:
            raise ValueError(f"stale checkpoint: {path}; use --force")
        return dict(checkpoint=str(path), reused=True)
    data = _complete_dataset(folder)
    if data.get("storage") == "trajectory_shards":
        train_video = open_video_dataset(folder, ell, "train", cache_size=task["shard_cache_size"])
        validation_video = open_video_dataset(folder, ell, "validation", cache_size=task["shard_cache_size"])
    else:
        train_video = load_dataset(folder, ell, "train")["video"]
        validation_video = load_dataset(folder, ell, "validation")["video"]
    if task["progress"]:
        print(f"{_condition_label(condition)}, fit seed {repeat}: "
              f"{train_video.shape[0] * (train_video.shape[1] - 1):,} train pairs", flush=True)
    result = train_model(
        train_video, validation_video, config.training,
        model_seed=config.base_seed + repeat, device=target, n_components=1,
        hidden_channels=config.hidden_channels, hidden_layers=config.hidden_layers,
        max_distance=config.max_distance, activation=config.activation, progress=task["progress"],
    )
    if not np.isfinite(result.best_validation_loss):
        raise FloatingPointError("nonfinite validation loss; reduce learning rate or batch interval")
    _save_checkpoint(path, dict(
        signature=task["signature"], config=asdict(config), repeat=repeat,
        condition=condition, device=str(target), training_seed=config.base_seed + repeat,
        model_state={key: val.detach().cpu() for key, val in result.model.state_dict().items()},
        mean=result.mean.detach().cpu(), std=result.std.detach().cpu(),
        best_iteration=result.best_iteration, best_validation_loss=result.best_validation_loss,
        history=dict(iterations=list(result.history_iterations),
            train_losses=list(result.train_losses), validation_losses=list(result.validation_losses)),
    ))
    return dict(checkpoint=str(path), reused=False)


def train_sweep(output_dir: str | Path, config: FitConfig, *, device="auto",
                devices=None, cpu_threads=1, shard_cache_size=128,
                progress=True, force=False) -> dict:
    """Fit only saved train/validation videos; never generate or inspect test data.

    Completed repeats are reused. A changed dataset, model or source requires
    another output directory or explicit ``force=True``.
    """
    folder = Path(output_dir)
    data = _complete_dataset(folder)
    _validate_fit(config, data["config"]["n"])
    targets = tuple(devices) if devices is not None else (str(_device(device)),)
    if not targets:
        raise ValueError("supply at least one execution device")
    if shard_cache_size < 1:
        raise ValueError("shard_cache_size must be positive")
    target = _device(targets[0])
    protocol = dict(dataset_signature=data["signature"], config=asdict(config),
                    source_hash=_source_hash(), device=target.type)
    signature = _hash(protocol)
    path = folder / "training.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["signature"] != signature and not force:
            raise ValueError("training settings changed; use a new output directory or --force")
    manifest = dict(signature=signature, **protocol, complete=False)
    _write_json(path, manifest)
    tasks = []
    for condition in data["conditions"]:
        ell = float(condition["ell"])
        for repeat in range(config.repeats):
            checkpoint_path = folder / "checkpoints" / condition_id(ell) / f"repeat_{repeat:03d}.pt"
            if checkpoint_path.exists() and not force:
                saved = _load_checkpoint(checkpoint_path)
                if saved.get("signature") != signature:
                    raise ValueError(f"stale checkpoint: {checkpoint_path}; use --force")
                if progress:
                    print(f"ell={ell:g}, repeat={repeat + 1}: reusing completed fit", flush=True)
                continue
            tasks.append(dict(stage="train", output_dir=str(folder.resolve()),
                config=asdict(config), condition=condition, repeat=repeat,
                checkpoint=str(checkpoint_path.resolve()), signature=signature,
                force=force, progress=progress, shard_cache_size=shard_cache_size,
                label=f"train {_condition_label(condition)}, repeat={repeat}"))
    _run_tasks(tasks, targets, cpu_threads=cpu_threads, progress=progress)
    manifest["complete"] = True
    _write_json(path, manifest)
    return manifest


def _restore(checkpoint: dict, target: torch.device) -> TrainingResult:
    config = checkpoint["config"]
    model = ShellForceKNEEP2D(n_components=1, hidden_channels=config["hidden_channels"],
        hidden_layers=config["hidden_layers"], max_distance=config["max_distance"],
        activation=config["activation"]).to(target)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return TrainingResult(model=model, mean=checkpoint["mean"].to(target),
        std=checkpoint["std"].to(target), best_iteration=checkpoint["best_iteration"],
        best_validation_loss=checkpoint["best_validation_loss"])


@torch.no_grad()
def _predict(result: TrainingResult, video: torch.Tensor, interval: float,
             batch_size: int, target: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Stream maps, retaining ordered increments and only time averaged maps.

    Existing ShellForce outputs are spatial means. Divide raw maps by n*n
    to make their spatial SUM equal the trained global entropy score. There
    is no subsequent area multiplication of the score.
    """
    members, frames, _, height, width = video.shape
    transitions = frames - 1
    branches = result.model.max_distance + 1
    increments = np.empty((members * transitions, branches), dtype=np.float64)
    map_sum = torch.zeros(members, branches, height, width, dtype=torch.float64)
    for start in range(0, members * transitions, batch_size):
        stop = min(start + batch_size, members * transitions)
        flat = torch.arange(start, stop)
        ensemble, time = flat // transitions, flat % transitions
        pair = torch.stack((video[ensemble, time], video[ensemble, time + 1]), dim=1)
        pair = pair.to(target, dtype=torch.float32)
        pair = (pair - result.mean) / result.std
        raw_maps = result.model(pair, return_maps=True).to(device="cpu", dtype=torch.float64)
        increments[start:stop] = raw_maps.mean(dim=(-2, -1)).numpy()
        map_sum.index_add_(0, ensemble, raw_maps)
    maps = map_sum.numpy() / (height * width * transitions * interval)
    if not np.isfinite(increments).all() or not np.isfinite(maps).all():
        raise FloatingPointError("nonfinite held-out predictions")
    return increments.reshape(members, transitions, branches), maps


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _save_record(path: Path, record: dict, signature: str) -> None:
    arrays = {key: value for key, value in record.items() if isinstance(value, np.ndarray)}
    extras = {key: value for key, value in record.items()
              if key not in arrays and key != "diagnostics"}
    for key, value in record.get("diagnostics", {}).items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            arrays[f"diagnostic_{key}"] = _numpy(value)
    extras["signature"] = signature
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, metadata=json.dumps(extras), **arrays)
    temporary.replace(path)


def _load_record(path: Path, signature: str) -> dict | None:
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"]))
        if metadata.pop("signature") != signature:
            return None
        record = {key: stored[key].copy() for key in stored.files
                  if key != "metadata" and not key.startswith("diagnostic_")}
        record.update(metadata)
        record["diagnostics"] = {key.removeprefix("diagnostic_"): stored[key].copy()
                                  for key in stored.files if key.startswith("diagnostic_")}
        return record


def _evaluate_condition(task: dict, target: torch.device) -> dict:
    folder, condition = Path(task["output_dir"]), task["condition"]
    data = _complete_dataset(folder)
    training = json.loads((folder / "training.json").read_text(encoding="utf-8"))
    path, signature = Path(task["path"]), task["signature"]
    if path.exists() and not task["force"] and _load_record(path, signature) is not None:
        return dict(path=str(path), signature=signature, reused=True)
    test = load_dataset(folder, float(condition["ell"]), "test")
    interval = float(data["config"]["dt"] * data["config"]["sample_every"])
    increments, maps, histories = [], [], []
    for checkpoint_path in task["checkpoints"]:
        checkpoint = _load_checkpoint(Path(checkpoint_path))
        if checkpoint["signature"] != training["signature"]:
            raise ValueError(f"checkpoint does not match training manifest: {checkpoint_path}")
        result = _restore(checkpoint, target)
        increment, mean_map = _predict(result, test["video"], interval, task["batch_size"], target)
        increments.append(increment)
        maps.append(mean_map)
        histories.append(checkpoint["history"])
        del result
    record = dict(ell=float(condition["ell"]),
        condition_id=condition["condition_id"], label=_condition_label(condition),
        strength=float(condition.get("strength", data["config"]["strength"])),
        is_control=bool(condition.get("is_control", False)),
        interval=interval, dx=float(data["config"]["dx"]),
        predicted_increment=np.stack(increments), predicted_map=np.stack(maps),
        reference_increment=_numpy(test["reference_increment"]),
        reference_map=_numpy(test["reference_map"]),
        reference_branch_increment=_numpy(test["reference_branch_increment"]),
        kernel_weights=_numpy(test["kernel_weights"]),
        kernel_multiplier=_numpy(test["kernel_multiplier"]),
        fields=_numpy(test["video"][:, [0, -1], 0]), history=histories,
        metadata=dict(dataset=data["config"], fit=training["config"],
            condition=condition, dataset_signature=data["signature"],
            training_signature=training["signature"]), diagnostics=test.get("diagnostics", {}))
    _save_record(path, record, signature)
    return dict(path=str(path), signature=signature, reused=False)


def evaluate_sweep(output_dir: str | Path, *, device="auto", devices=None,
                   batch_size=32, cpu_threads=1, progress=True, force=False) -> list[dict]:
    """Evaluate held-out data in independent workers; parent saves the summary."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    folder = Path(output_dir)
    data = _complete_dataset(folder)
    training = json.loads((folder / "training.json").read_text(encoding="utf-8"))
    if not training.get("complete") or training["dataset_signature"] != data["signature"]:
        raise ValueError("complete matching training is required before evaluation")
    if training.get("source_hash") != _source_hash():
        raise ValueError("model or training source changed; rerun the train stage with --force before evaluation")
    targets = tuple(devices) if devices is not None else (str(_device(device)),)
    if not targets:
        raise ValueError("supply at least one execution device")
    tasks = []
    for condition in data["conditions"]:
        cid = condition["condition_id"]
        paths = [folder / "checkpoints" / cid / f"repeat_{repeat:03d}.pt"
                 for repeat in range(training["config"]["repeats"])]
        signature = _hash(dict(data=data["signature"], training=training["signature"],
                              evaluator=_source_hash(), checkpoints=[_file_hash(p) for p in paths]))
        tasks.append(dict(stage="evaluate", output_dir=str(folder.resolve()), condition=condition,
            path=str((folder / "evaluation" / f"{cid}.npz").resolve()), signature=signature,
            checkpoints=[str(path.resolve()) for path in paths], batch_size=batch_size,
            force=force, label=f"evaluate {_condition_label(condition)}"))
    completed = _run_tasks(tasks, targets, cpu_threads=cpu_threads, progress=progress)
    records = [_load_record(Path(item["path"]), item["signature"]) for item in completed]
    if any(record is None for record in records):
        raise RuntimeError("evaluation signature changed while workers were running")
    _save_summary(folder, records)
    return records


def _save_summary(folder: Path, records: list[dict]) -> None:
    rows = []
    for r in records:
        # Sum signed shells/channels first: all temporal covariances survive.
        predicted_rate = r["predicted_increment"].sum(axis=-1) / r["interval"]
        reference_rate = r["reference_increment"].sum(axis=-1) / r["interval"]
        repeat_means = predicted_rate.mean(axis=(1, 2))
        rows.append(dict(condition=r.get("condition_id", condition_id(r["ell"])),
            ell=None if r.get("is_control") else r["ell"],
            strength=r.get("strength", r["metadata"]["dataset"]["strength"]),
            label=r.get("label", f"ell={r['ell']:g}"),
            kneep_mean_rate=float(repeat_means.mean()),
            kneep_training_seed_std=float(repeat_means.std(ddof=0)),
            medium_mean_rate=float(reference_rate.mean()),
            kneep_temporal_variance=float(predicted_rate.var(axis=-1, ddof=0).mean()),
            medium_temporal_variance=float(reference_rate.var(axis=-1, ddof=0).mean()),
            active_mean_rate=float((r["reference_increment"][..., 2] / r["interval"]).mean())))
    control = next((row for row in rows if row["ell"] is None), None)
    if control is not None:
        for row in rows:
            row["kneep_minus_no_long_range"] = row["kneep_mean_rate"] - control["kneep_mean_rate"]
            row["medium_minus_no_long_range"] = row["medium_mean_rate"] - control["medium_mean_rate"]
    _write_json(folder / "evaluation" / "summary.json", dict(
        rows=rows, uncertainty="descriptive population SD (ddof=0); training-seed spread is conditional on fixed data, not a confidence interval",
        reference="full-step midpoint medium EP; KNEEP infers sampled path irreversibility",
    ))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (folder / "evaluation" / "summary.csv").write_text(stream.getvalue(), encoding="utf-8")


def _execution_devices(args) -> tuple[str, ...]:
    if args.num_gpus is not None:
        if args.num_gpus < 1 or not torch.cuda.is_available():
            raise ValueError("--num-gpus requires a positive count and available CUDA")
        if args.num_gpus > torch.cuda.device_count():
            raise ValueError(f"requested {args.num_gpus} GPUs, but only "
                             f"{torch.cuda.device_count()} are visible to this job")
        count = args.workers if args.workers is not None else args.num_gpus
        if not 1 <= count <= args.num_gpus:
            raise ValueError("--workers must be between 1 and --num-gpus")
        return tuple(f"cuda:{index}" for index in range(count))
    target = _device(args.device or "auto")
    count = args.workers if args.workers is not None else 1
    if count < 1:
        raise ValueError("--workers must be positive")
    if target.type == "cuda" and count != 1:
        raise ValueError("use --num-gpus N to select one worker per GPU")
    return (str(target),) * count


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="all",
                        choices=("all", "generate", "train", "evaluate", "plot"))
    parser.add_argument("--preset", choices=("quick", "demo", "refined", "production", "reference"),
                        default="production")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--lramb-root", type=Path)
    placement = parser.add_mutually_exclusive_group()
    placement.add_argument("--device", help="one device: cpu, cuda[:index], auto (default)")
    placement.add_argument("--num-gpus", type=int,
                           help="one process per GPU, using the first N visible CUDA devices")
    parser.add_argument("--workers", type=int,
                        help="CPU worker count, or cap the number of allocated GPU workers")
    parser.add_argument("--ells", nargs="+", type=float)
    parser.add_argument("--include-no-interaction", action=argparse.BooleanOptionalAction,
                        default=None, help="include one g=0 no-long-range control")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--max-distance", type=int, help="maximum shell radius; 12 means a 25x25 outer stencil")
    parser.add_argument("--train-batch-size", type=int, help="logical number of transition pairs per optimizer step")
    parser.add_argument("--train-micro-batch-size", type=int,
                        help="GPU chunk size with exact gradient accumulation within the logical batch")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--validate-every", type=int)
    parser.add_argument("--sample-every", type=int)
    parser.add_argument("--burn-steps", type=int)
    parser.add_argument("--production-steps", type=int)
    parser.add_argument("--train-trajectories", type=int)
    parser.add_argument("--validation-trajectories", type=int)
    parser.add_argument("--test-trajectories", type=int)
    parser.add_argument("--min-train-pairs", type=int,
                        help="increase train trajectory count to retain at least this many adjacent pairs")
    parser.add_argument("--batch-size", type=int, help="evaluation batch size; uses the fit preset by default")
    parser.add_argument("--shard-cache-size", type=int, default=128,
                        help="maximum retained memory-mapped trajectory files per training worker")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="explicitly replace incompatible stage artifacts")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.cpu_threads < 1 or args.shard_cache_size < 1:
        parser.error("--cpu-threads and --shard-cache-size must be positive")
    try:
        devices = _execution_devices(args)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    torch.set_num_threads(args.cpu_threads)
    output = args.output_dir or ROOT / "results" / "lramb_poisson_range" / args.preset
    data_config, fit_config = dataset_preset(args.preset), fit_preset(args.preset)
    data_overrides = {name: getattr(args, name) for name in (
        "sample_every", "burn_steps", "production_steps", "train_trajectories",
        "validation_trajectories", "test_trajectories", "include_no_interaction",
    ) if getattr(args, name) is not None}
    if args.ells is not None:
        data_overrides["ells"] = tuple(args.ells)
    data_config = replace(data_config, **data_overrides)
    if args.min_train_pairs is not None:
        if args.min_train_pairs < 1:
            parser.error("--min-train-pairs must be positive")
        data_config = replace(data_config, train_trajectories=max(
            data_config.train_trajectories,
            math.ceil(args.min_train_pairs / (data_config.production_steps // data_config.sample_every))))
    optimizer_overrides = {name: getattr(args, name) for name in (
        "iterations", "train_batch_size", "train_micro_batch_size", "learning_rate",
        "weight_decay", "validate_every",
    ) if getattr(args, name) is not None}
    fit_config = replace(fit_config, training=replace(fit_config.training, **optimizer_overrides),
        **{name: getattr(args, name) for name in ("repeats", "max_distance")
           if getattr(args, name) is not None})
    prediction_batch = args.batch_size if args.batch_size is not None else fit_config.training.prediction_batch_size
    if prediction_batch < 1:
        parser.error("--batch-size must be positive")
    print(f"Devices: {', '.join(devices)}; CPU threads/worker: {args.cpu_threads}")
    print(f"Output: {output}")
    print(f"Configured train pairs/condition: {data_config.train_pairs:,}; "
          f"logical batch={fit_config.training.train_batch_size}, "
          f"microbatch={fit_config.training.train_micro_batch_size}, "
          f"radius={fit_config.max_distance}, iterations={fit_config.training.iterations}")
    if args.stage in {"all", "generate"}:
        generate_sweep(data_config, output, lramb_root=args.lramb_root,
            devices=devices, cpu_threads=args.cpu_threads, force=args.force, progress=not args.quiet)
        print(f"Dataset ready: {output}")
    if args.stage in {"all", "train"}:
        train_sweep(output, fit_config, devices=devices, cpu_threads=args.cpu_threads,
            shard_cache_size=args.shard_cache_size, force=args.force, progress=not args.quiet)
        print(f"Training complete: {output}")
    if args.stage in {"all", "evaluate", "plot"}:
        records = evaluate_sweep(output, devices=devices, batch_size=prediction_batch,
            cpu_threads=args.cpu_threads, progress=not args.quiet, force=args.force)
        if args.stage == "plot":
            import matplotlib.pyplot as plt
            from utils.poisson_range_plots import (
                plot_cumulative_ep, plot_time_averaged_ep, plot_ep_spectrum,
                plot_temporal_spectrum, plot_range_response, plot_training_history,
                plot_kernel_diagnostics,
            )
            for plot in (plot_kernel_diagnostics, plot_training_history, plot_cumulative_ep,
                         plot_time_averaged_ep, plot_ep_spectrum, plot_temporal_spectrum,
                         plot_range_response):
                plot(records)
            plt.show()
            return
        for record in records:
            print(f"{record.get('label', record['ell'])}: predictions {record['predicted_increment'].shape}")
        print(f"Evaluation arrays: {output / 'evaluation'}; open demos/lramb_poisson_range.ipynb for inline plots")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
