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
import sys
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
    generate_datasets,
    load_dataset,
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
    if name == "reference":
        return FitConfig(hidden_layers=3, repeats=5,
            training=replace(FitConfig().training, iterations=5_000))
    raise ValueError("preset must be quick, demo, refined or reference")


def _device(value: str | torch.device) -> torch.device:
    value = str(value)
    target = torch.device("cuda" if value == "auto" and torch.cuda.is_available()
                          else "cpu" if value == "auto" else value)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return target


def _hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _source_hash() -> str:
    return hashlib.sha256(b"".join(path.read_bytes() for path in (
        Path(__file__), ROOT / "shell_force.py", ROOT / "utils" / "training.py",
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
    if not all(math.isfinite(value) for value in (
        t.alpha, t.learning_rate, t.weight_decay, t.gradient_clip,
    )) or t.alpha == -1 or t.learning_rate <= 0 or t.weight_decay < 0 or t.gradient_clip <= 0:
        raise ValueError("invalid training loss or optimizer settings")


def train_sweep(output_dir: str | Path, config: FitConfig, *, device="auto",
                progress=True, force=False) -> dict:
    """Fit only saved train/validation videos; never generate or inspect test data.

    Completed repeats are reused. A changed dataset, model or source requires
    another output directory or explicit ``force=True``.
    """
    folder = Path(output_dir)
    data = _complete_dataset(folder)
    _validate_fit(config, data["config"]["n"])
    target = _device(device)
    protocol = dict(dataset_signature=data["signature"], config=asdict(config),
                    source_hash=_source_hash(), device=str(target))
    signature = _hash(protocol)
    path = folder / "training.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["signature"] != signature and not force:
            raise ValueError("training settings changed; use a new output directory or --force")
    manifest = dict(signature=signature, **protocol, complete=False)
    _write_json(path, manifest)
    for ell in data["config"]["ells"]:
        # Ground-truth EP labels are deliberately not used as learning targets.
        train_video = load_dataset(folder, ell, "train")["video"]
        validation_video = load_dataset(folder, ell, "validation")["video"]
        for repeat in range(config.repeats):
            checkpoint_path = folder / "checkpoints" / condition_id(ell) / f"repeat_{repeat:03d}.pt"
            if checkpoint_path.exists() and not force:
                saved = _load_checkpoint(checkpoint_path)
                if saved.get("signature") != signature:
                    raise ValueError(f"stale checkpoint: {checkpoint_path}; use --force")
                if progress:
                    print(f"ell={ell:g}, repeat={repeat + 1}: reusing completed fit", flush=True)
                continue
            if progress:
                print(f"ell={ell:g}, repeat={repeat + 1}/{config.repeats}: training", flush=True)
            result = train_model(
                train_video, validation_video, config.training,
                model_seed=config.base_seed + repeat, device=target, n_components=1,
                hidden_channels=config.hidden_channels, hidden_layers=config.hidden_layers,
                max_distance=config.max_distance, activation=config.activation, progress=progress,
            )
            if not np.isfinite(result.best_validation_loss):
                raise FloatingPointError("nonfinite validation loss; reduce learning rate or batch interval")
            _save_checkpoint(checkpoint_path, dict(
                signature=signature, config=asdict(config), repeat=repeat,
                training_seed=config.base_seed + repeat,
                model_state={key: val.detach().cpu() for key, val in result.model.state_dict().items()},
                mean=result.mean.detach().cpu(), std=result.std.detach().cpu(),
                best_iteration=result.best_iteration,
                best_validation_loss=result.best_validation_loss,
                history=dict(iterations=list(result.history_iterations),
                    train_losses=list(result.train_losses), validation_losses=list(result.validation_losses)),
            ))
            del result
        del train_video, validation_video
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


def evaluate_sweep(output_dir: str | Path, *, device="auto", batch_size=32,
                   force=False) -> list[dict]:
    """Evaluate independent held-out trajectories; save arrays, never figures."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    folder = Path(output_dir)
    data = _complete_dataset(folder)
    training = json.loads((folder / "training.json").read_text(encoding="utf-8"))
    if not training.get("complete") or training["dataset_signature"] != data["signature"]:
        raise ValueError("complete matching training is required before evaluation")
    if training.get("source_hash") != _source_hash():
        raise ValueError("model or training source changed; rerun the train stage with --force before evaluation")
    target = _device(device)
    records = []
    for ell in data["config"]["ells"]:
        paths = [folder / "checkpoints" / condition_id(ell) / f"repeat_{repeat:03d}.pt"
                 for repeat in range(training["config"]["repeats"])]
        signature = _hash(dict(data=data["signature"], training=training["signature"],
                              evaluator=_source_hash(), checkpoints=[_file_hash(p) for p in paths]))
        path = folder / "evaluation" / f"{condition_id(ell)}.npz"
        record = _load_record(path, signature) if path.exists() and not force else None
        if record is None:
            test = load_dataset(folder, ell, "test")
            interval = float(data["config"]["dt"] * data["config"]["sample_every"])
            increments, maps, histories = [], [], []
            for checkpoint_path in paths:
                checkpoint = _load_checkpoint(checkpoint_path)
                if checkpoint["signature"] != training["signature"]:
                    raise ValueError(f"checkpoint does not match training manifest: {checkpoint_path}")
                result = _restore(checkpoint, target)
                increment, mean_map = _predict(result, test["video"], interval, batch_size, target)
                increments.append(increment)
                maps.append(mean_map)
                histories.append(checkpoint["history"])
                del result
            record = dict(ell=float(ell), interval=interval, dx=float(data["config"]["dx"]),
                predicted_increment=np.stack(increments), predicted_map=np.stack(maps),
                reference_increment=_numpy(test["reference_increment"]),
                reference_map=_numpy(test["reference_map"]),
                reference_branch_increment=_numpy(test["reference_branch_increment"]),
                kernel_weights=_numpy(test["kernel_weights"]),
                kernel_multiplier=_numpy(test["kernel_multiplier"]),
                fields=_numpy(test["video"][:, [0, -1], 0]), history=histories,
                metadata=dict(dataset=data["config"], fit=training["config"],
                    dataset_signature=data["signature"], training_signature=training["signature"]),
                diagnostics=test.get("diagnostics", {}))
            _save_record(path, record, signature)
        records.append(record)
    _save_summary(folder, records)
    return records


def _save_summary(folder: Path, records: list[dict]) -> None:
    rows = []
    for r in records:
        # Sum signed shells/channels first: all temporal covariances survive.
        predicted_rate = r["predicted_increment"].sum(axis=-1) / r["interval"]
        reference_rate = r["reference_increment"].sum(axis=-1) / r["interval"]
        repeat_means = predicted_rate.mean(axis=(1, 2))
        rows.append(dict(ell=r["ell"], kneep_mean_rate=float(repeat_means.mean()),
            kneep_training_seed_std=float(repeat_means.std(ddof=0)),
            medium_mean_rate=float(reference_rate.mean()),
            kneep_temporal_variance=float(predicted_rate.var(axis=-1, ddof=0).mean()),
            medium_temporal_variance=float(reference_rate.var(axis=-1, ddof=0).mean()),
            active_mean_rate=float((r["reference_increment"][..., 2] / r["interval"]).mean())))
    _write_json(folder / "evaluation" / "summary.json", dict(
        rows=rows, uncertainty="descriptive population SD (ddof=0); training-seed spread is conditional on fixed data, not a confidence interval",
        reference="full-step midpoint medium EP; KNEEP infers sampled path irreversibility",
    ))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (folder / "evaluation" / "summary.csv").write_text(stream.getvalue(), encoding="utf-8")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "train", "evaluate", "plot"))
    parser.add_argument("--preset", choices=("quick", "demo", "refined", "reference"), default="demo")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--lramb-root", type=Path)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or auto")
    parser.add_argument("--ells", nargs="+", type=float)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--max-distance", type=int)
    parser.add_argument("--sample-every", type=int)
    parser.add_argument("--burn-steps", type=int)
    parser.add_argument("--production-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="explicitly replace incompatible stage artifacts")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    output = args.output_dir or ROOT / "results" / "lramb_poisson_range" / args.preset
    if args.stage == "generate":
        config = dataset_preset(args.preset)
        overrides = {name: getattr(args, name) for name in
                     ("sample_every", "burn_steps", "production_steps") if getattr(args, name) is not None}
        if args.ells is not None:
            overrides["ells"] = tuple(args.ells)
        generate_datasets(replace(config, **overrides), output, lramb_root=args.lramb_root,
                          device=args.device, force=args.force, progress=not args.quiet)
        print(f"Dataset ready: {output}")
    elif args.stage == "train":
        config = fit_preset(args.preset)
        overrides = {name: getattr(args, name) for name in
                     ("repeats", "max_distance") if getattr(args, name) is not None}
        if args.iterations is not None:
            config = replace(config, training=replace(config.training, iterations=args.iterations))
        train_sweep(output, replace(config, **overrides), device=args.device,
                    force=args.force, progress=not args.quiet)
        print(f"Training complete: {output}")
    else:
        records = evaluate_sweep(output, device=args.device, batch_size=args.batch_size, force=args.force)
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
            print(f"ell={record['ell']:g}: predictions {record['predicted_increment'].shape}")
        print(f"Evaluation arrays: {output / 'evaluation'}; open demos/lramb_poisson_range.ipynb for inline plots")


if __name__ == "__main__":
    main()
