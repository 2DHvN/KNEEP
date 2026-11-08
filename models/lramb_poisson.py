"""Generate self-contained Poisson-range datasets with the sibling LRAMB solver.

Generation imports LRAMB; loading and fitting saved datasets only need PyTorch.
Ground-truth entropy uses every solver step, never differences of saved movies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import torch


SCHEMA_VERSION = 1
CHANNEL_NAMES = ("local conservative", "kernel conservative", "local active")
SPLITS = ("train", "validation", "test")
_SEED_OFFSETS = {"train": 0, "validation": 100_000, "test": 200_000}


@dataclass(frozen=True)
class DatasetConfig:
    """A fresh dynamical sweep at fixed coupling and physical protocol.

    ``production_steps / sample_every`` transitions are retained per trajectory.
    Each role has separate seeds; corresponding trajectories share their initial
    field and noise realization across ell, within one backend.
    """

    ells: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    n: int = 32
    dx: float = 1.0
    dt: float = 0.01
    a: float = -1.0
    b: float = 1.0
    kappa: float = 1.0
    lam: float = 0.8
    mobility: float = 1.0
    D: float = 0.05
    mean: float = -0.2
    strength: float = 0.35
    burn_steps: int = 4_000
    production_steps: int = 8_192
    sample_every: int = 8
    blocks: int = 8
    train_trajectories: int = 4
    validation_trajectories: int = 2
    test_trajectories: int = 3
    base_seed: int = 17
    initial_mode: str = "stripe"

    def __post_init__(self) -> None:
        object.__setattr__(self, "ells", tuple(float(x) for x in self.ells))
        if (len(self.ells) < 2 or len(set(self.ells)) != len(self.ells)
                or any(not math.isfinite(x) or x <= 0 for x in self.ells)):
            raise ValueError("ells must contain at least two distinct positive finite ranges")
        integer_fields = ("n", "burn_steps", "production_steps", "sample_every", "blocks",
                          "train_trajectories", "validation_trajectories", "test_trajectories",
                          "base_seed")
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.n < 4 or self.n % 2:
            raise ValueError("n must be an even integer >= 4")
        if self.burn_steps < 0 or self.base_seed < 0 or self.blocks < 2:
            raise ValueError("burn_steps/base_seed must be nonnegative and blocks >= 2")
        if min(self.production_steps, self.sample_every, self.train_trajectories,
               self.validation_trajectories, self.test_trajectories) <= 0:
            raise ValueError("production, sampling, and trajectory counts must be positive")
        if self.production_steps % (self.sample_every * self.blocks):
            raise ValueError("production_steps must be divisible by sample_every * blocks")
        if self.production_steps // (self.sample_every * self.blocks) < 2:
            raise ValueError("at least two saved EP intervals per time block are needed")
        for name in ("dx", "dt", "a", "b", "kappa", "lam", "mobility", "D", "mean", "strength"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if min(self.dx, self.dt, self.b, self.mobility, self.D) <= 0 or self.kappa < 0:
            raise ValueError("dx, dt, b, mobility, D must be positive; kappa >= 0")
        if self.initial_mode not in {"stripe", "random"}:
            raise ValueError("initial_mode must be stripe or random")

    @property
    def interval(self) -> float:
        return self.dt * self.sample_every

    @property
    def duration(self) -> float:
        return self.dt * self.production_steps

    @property
    def max_distance(self) -> int:
        return self.n // 2

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ells"] = list(self.ells)
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DatasetConfig":
        return cls(**value)

    @classmethod
    def preset(cls, name: str) -> "DatasetConfig":
        return dataset_preset(name)

    def seeds(self, split: str) -> tuple[int, ...]:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        count = getattr(self, f"{split}_trajectories")
        if count >= 100_000:
            raise ValueError("each split must contain fewer than 100000 trajectories")
        return tuple(self.base_seed + _SEED_OFFSETS[split] + j for j in range(count))


def dataset_preset(name: str = "demo") -> DatasetConfig:
    """Quick checks, demo, finer observation, or the LRAMB time protocol."""
    if name == "quick":
        return DatasetConfig(n=8, burn_steps=16, production_steps=64,
                             sample_every=1, blocks=4, train_trajectories=2,
                             validation_trajectories=1, test_trajectories=2)
    if name == "demo":
        return DatasetConfig()
    if name == "refined":
        # Retain every Euler step to remove the demo's eight-step observation
        # gap. Keep its physical parameters, grid and production duration.
        return DatasetConfig(burn_steps=16_000, sample_every=1,
                             train_trajectories=8, validation_trajectories=4,
                             test_trajectories=4)
    if name == "reference":
        return DatasetConfig(n=64, burn_steps=40_000, production_steps=64_000,
                             sample_every=40, blocks=8)
    raise ValueError("preset must be quick, demo, refined or reference")


def condition_id(ell: float) -> str:
    value = float(ell)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("ell must be positive and finite")
    return f"ell_{value:.12g}"


def read_manifest(output_dir: str | Path) -> dict[str, Any]:
    with (Path(output_dir) / "dataset.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def load_config(output_dir: str | Path) -> DatasetConfig:
    return DatasetConfig.from_dict(read_manifest(output_dir)["config"])


def load_dataset(output_dir: str | Path, ell: float, split: str = "test") -> dict[str, Any]:
    """Load only CPU tensors and JSON-like metadata; no LRAMB import is needed."""
    if split == "val":
        split = "validation"
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    manifest = read_manifest(output_dir)
    matching = [c for c in manifest["conditions"] if float(c["ell"]) == float(ell)]
    if not matching:
        raise ValueError(f"ell={ell} is absent from dataset.json")
    payload = torch.load(Path(output_dir) / matching[0]["files"][split],
                         map_location="cpu", weights_only=True)
    if payload["metadata"]["signature"] != manifest["signature"]:
        raise ValueError("saved split does not match dataset.json; regenerate the dataset")
    return payload


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_torch(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _lramb_module(lramb_root: str | Path | None):
    candidates = ([Path(lramb_root)] if lramb_root is not None else
                  [Path(__file__).resolve().parents[2] / "LRAMB",
                   Path.cwd() / "LRAMB", Path.cwd().parent / "LRAMB"])
    root = next((p.resolve() for p in candidates if (p / "lramb" / "model.py").is_file()), None)
    if root is None:
        raise FileNotFoundError("LRAMB/lramb/model.py was not found. Supply lramb_root with the LRAMB repository path.")
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("lramb.model")
    except ModuleNotFoundError as error:
        raise ImportError("Dataset generation needs the LRAMB dependencies, including scipy>=1.10.") from error
    if Path(module.__file__).resolve() != root / "lramb" / "model.py":
        raise ImportError(f"A different LRAMB package is already imported: {module.__file__}")
    return root, module


def _backend(device: str) -> dict[str, str]:
    target = str(device)
    if target == "auto":
        target = "cuda" if torch.cuda.is_available() else "cpu"
    parsed = torch.device(target)
    if parsed.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu, cuda[:index] or auto")
    if parsed.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return {"name": "numpy" if parsed.type == "cpu" else "torch",
            "device": str(parsed), "dtype": "float64"}


def _hash_array(array: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _shell_multipliers(model) -> np.ndarray:
    """Exact periodic Chebyshev shells matching ShellForceKNEEP2D branches."""
    offsets = np.fft.fftfreq(model.p.n) * model.p.n
    yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
    distance = np.maximum(np.abs(xx), np.abs(yy)).round().astype(int)
    return np.asarray([np.fft.fft2(model.weights * (distance == r)).real
                       for r in range(model.p.n // 2 + 1)])


def _run_numpy(model, cfg: DatasetConfig, multipliers: np.ndarray) -> dict[str, Any]:
    phi = model.initial_field(cfg.initial_mode)
    initial_hash = _hash_array(phi)
    ph = np.fft.fft2(phi, norm="ortho")
    for step in range(cfg.burn_steps):
        phi, ph = model.step(phi, ph)
        if step % 100 == 0 and (not np.isfinite(phi).all() or np.max(np.abs(phi)) > 30):
            raise FloatingPointError("Unstable burn-in trajectory; reduce dt")
    count = cfg.production_steps // cfg.sample_every
    video = np.empty((count + 1, 1, cfg.n, cfg.n), dtype=np.float32)
    reference = np.empty((count, 3), dtype=np.float64)
    branches = np.empty((count, cfg.max_distance + 1), dtype=np.float64)
    mass, variance = np.empty(count + 1), np.empty(count + 1)
    energies = np.empty((count + 1, 3), dtype=np.float64)
    ep_maps = np.zeros((3, cfg.n, cfg.n), dtype=np.float64)
    interval = np.zeros(3, dtype=np.float64)
    max_balance_error = 0.0

    def record(index):
        video[index, 0] = phi
        mass[index], variance[index] = phi.mean(), phi.var()
        energies[index] = model.energy_channels(phi, ph)[0]
        return (model.dV * cfg.strength / 2
                * np.einsum("rij,ij->r", multipliers, np.abs(ph) ** 2, optimize=False))

    previous_shell_energy = record(0)
    for step in range(1, cfg.production_steps + 1):
        new, new_hat = model.step(phi, ph)
        ds = model.entropy_maps(phi, new, ph, new_hat)
        ep_maps += ds
        interval += model.dV * ds.sum(axis=(1, 2))
        phi, ph = new, new_hat
        if step % 100 == 0 and (not np.isfinite(phi).all() or np.max(np.abs(phi)) > 30):
            raise FloatingPointError("Unstable production trajectory; reduce dt")
        if step % cfg.sample_every == 0:
            index = step // cfg.sample_every
            shell_energy = record(index)
            reference[index - 1] = interval
            branches[index - 1] = -(shell_energy - previous_shell_energy) / cfg.D
            branches[index - 1, 0] += interval[0] + interval[2]
            conservative_boundary = -(energies[index].sum() - energies[index - 1].sum()) / cfg.D
            max_balance_error = max(max_balance_error, abs(interval[:2].sum() - conservative_boundary))
            previous_shell_energy = shell_energy
            interval.fill(0)
    if not np.isfinite(video).all() or np.max(np.abs(mass - mass[0])) > 1e-10:
        raise FloatingPointError("Nonfinite trajectory or mass conservation failure")
    return dict(video=video, reference_increment=reference,
                reference_branch_increment=branches, reference_map=ep_maps / cfg.duration,
                mass=mass, variance=variance, energy=energies,
                initial_hash=initial_hash, quartic_balance_error=max_balance_error)


def _run_torch(model, cfg: DatasetConfig, multipliers: np.ndarray, device: str) -> dict[str, Any]:
    # Reuse the actual LRAMB solver kernels, with float64 dynamics and entropy.
    from lramb.torch_backend import _step, _midpoint

    initial = model.initial_field(cfg.initial_mode)
    initial_hash = _hash_array(initial)
    kw = dict(dtype=torch.float64, device=device)
    phi = torch.as_tensor(initial, **kw)
    ph = torch.fft.rfft2(phi, norm="ortho")
    nh = cfg.n // 2 + 1
    q2 = torch.as_tensor(model.q2[:, :nh], **kw)
    kernel = torch.as_tensor(model.K[:, :nh], **kw)
    linear = torch.as_tensor(model.linear_mu[:, :nh], **kw)
    noise_scale = torch.as_tensor(model.noise_scale[:, :nh], **kw)
    shell = torch.as_tensor(multipliers[:, :, :nh], **kw)
    multiplicity = torch.full((nh,), 2.0, **kw)
    multiplicity[0] = multiplicity[-1] = 1.0
    rng = torch.Generator(device=device).manual_seed(model.p.seed)
    count = cfg.production_steps // cfg.sample_every
    video = torch.empty((count + 1, 1, cfg.n, cfg.n), dtype=torch.float32)
    reference = torch.empty((count, 3), **kw)
    branches = torch.empty((count, cfg.max_distance + 1), **kw)
    mass = torch.empty(count + 1, **kw)
    variance = torch.empty(count + 1, **kw)
    energies = torch.empty((count + 1, 3), **kw)
    ep_maps = torch.zeros((3, cfg.n, cfg.n), **kw)
    interval = torch.zeros(3, **kw)
    max_balance_error = torch.zeros((), **kw)

    def advance():
        normal = torch.randn(phi.shape, generator=rng, **kw)
        return _step(phi, ph, normal, linear, q2, noise_scale,
                     cfg.dt * cfg.mobility, cfg.b, cfg.lam, cfg.dx)

    def record(index):
        video[index, 0] = phi.float().cpu()
        mass[index], variance[index] = phi.mean(), phi.var(correction=0)
        power = ph.abs().square() * multiplicity
        bulk = model.dV * (cfg.a / 2 * phi.square() + cfg.b / 4 * phi.pow(4)).sum()
        gradient = model.dV * cfg.kappa / 2 * (q2 * power).sum()
        lr_energy = model.dV * cfg.strength / 2 * (kernel * power).sum()
        energies[index] = torch.stack((bulk, gradient, lr_energy))
        return model.dV * cfg.strength / 2 * (shell * power).sum(dim=(1, 2))

    with torch.inference_mode():
        for step in range(cfg.burn_steps):
            phi, ph = advance()
            if step % 100 == 0 and (not torch.isfinite(phi).all() or phi.abs().max() > 30):
                raise FloatingPointError("Unstable burn-in trajectory; reduce dt")
        previous_shell_energy = record(0)
        for step in range(1, cfg.production_steps + 1):
            new, new_hat = advance()
            ds, _ = _midpoint(phi, new, ph, new_hat, kernel, cfg.a, cfg.b,
                              cfg.kappa, cfg.lam, cfg.strength, cfg.dx, cfg.D)
            ep_maps += ds
            interval += model.dV * ds.sum(dim=(1, 2))
            phi, ph = new, new_hat
            if step % 100 == 0 and (not torch.isfinite(phi).all() or phi.abs().max() > 30):
                raise FloatingPointError("Unstable production trajectory; reduce dt")
            if step % cfg.sample_every == 0:
                index = step // cfg.sample_every
                shell_energy = record(index)
                reference[index - 1] = interval
                branches[index - 1] = -(shell_energy - previous_shell_energy) / cfg.D
                branches[index - 1, 0] += interval[0] + interval[2]
                boundary = -(energies[index].sum() - energies[index - 1].sum()) / cfg.D
                max_balance_error = torch.maximum(max_balance_error, (interval[:2].sum() - boundary).abs())
                previous_shell_energy = shell_energy
                interval.zero_()
        if not torch.isfinite(video).all() or (mass - mass[0]).abs().max() > 1e-10:
            raise FloatingPointError("Nonfinite trajectory or mass conservation failure")
    return dict(video=video, reference_increment=reference.cpu(),
                reference_branch_increment=branches.cpu(), reference_map=(ep_maps / cfg.duration).cpu(),
                mass=mass.cpu(), variance=variance.cpu(), energy=energies.cpu(),
                initial_hash=initial_hash, quartic_balance_error=float(max_balance_error.cpu()))


def generate_datasets(config: DatasetConfig, output_dir: str | Path,
                      lramb_root: str | Path | None = None, device: str = "cpu",
                      progress: bool = True, force: bool = False) -> dict[str, Any]:
    """Integrate independent split trajectories and atomically cache their data.

    Existing data are reused only when config, solver/generator source and
    backend signatures match. Mismatches require ``force=True`` or a new folder.
    CPU uses LRAMB's NumPy solver; CUDA uses its actual Torch solver kernels.
    """
    if not isinstance(config, DatasetConfig):
        raise TypeError("config must be a DatasetConfig")
    root, lramb = _lramb_module(lramb_root)
    backend = _backend(device)
    source = {"lramb_root": str(root),
              "lramb_model_sha256": sha256((root / "lramb/model.py").read_bytes()).hexdigest(),
              "lramb_torch_backend_sha256": sha256((root / "lramb/torch_backend.py").read_bytes()).hexdigest(),
              "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    # The root path is provenance, not a physical or numerical input.
    signature_input = dict(schema_version=SCHEMA_VERSION, config=config.to_dict(),
                           source={k: v for k, v in source.items() if k != "lramb_root"}, backend=backend)
    signature = sha256(json.dumps(signature_input, sort_keys=True).encode()).hexdigest()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    old = read_manifest(output_dir) if (output_dir / "dataset.json").exists() else None
    if old and old.get("signature") != signature and not force:
        raise ValueError("Dataset configuration, solver source or backend changed. Use a new output directory or force=True.")
    ids = [condition_id(ell) for ell in config.ells]
    if len(ids) != len(set(ids)):
        raise ValueError("ell values are too close to produce distinct condition names")
    manifest = dict(schema_version=SCHEMA_VERSION, signature=signature,
                    config=config.to_dict(), source=source, backend=backend,
                    splits={split: {"trajectories": len(config.seeds(split)),
                                    "seeds": list(config.seeds(split))} for split in SPLITS},
                    conditions=[], complete=False)
    for ell in config.ells:
        manifest["conditions"].append(dict(ell=ell, condition_id=condition_id(ell),
            files={split: f"data/{condition_id(ell)}/{split}.pt" for split in SPLITS}))
    _atomic_json(manifest, output_dir / "dataset.json")

    for condition in manifest["conditions"]:
        ell = condition["ell"]
        for split in SPLITS:
            path = output_dir / condition["files"][split]
            if path.exists() and not force:
                cached = torch.load(path, map_location="cpu", weights_only=True)
                if cached.get("metadata", {}).get("signature") != signature:
                    raise ValueError(f"Cached {path} has a mismatching signature; use force=True or a new directory")
                if progress:
                    print(f"ell={ell:g}, {split}: using matching dataset", flush=True)
                condition["kernel_multiplier_sha256"] = cached["metadata"]["kernel_multiplier_sha256"]
                condition.setdefault("initial_hashes", {})[split] = cached["diagnostics"]["initial_hashes"]
                continue
            trajectories = []
            model = None
            for index, seed in enumerate(config.seeds(split)):
                parameters = lramb.Parameters(**{k: getattr(config, k) for k in
                    ("n", "dx", "dt", "a", "b", "kappa", "lam", "mobility", "D", "mean")}, seed=seed)
                model = lramb.ActiveModelB(parameters, lramb.KernelSpec("poisson", strength=config.strength, ell=ell))
                multipliers = _shell_multipliers(model)
                if progress:
                    print(f"ell={ell:g}, {split} {index + 1}/{len(config.seeds(split))}: "
                          f"{config.burn_steps + config.production_steps:,} steps ({backend['name']}/{backend['device']})", flush=True)
                runner = (_run_numpy(model, config, multipliers) if backend["name"] == "numpy"
                          else _run_torch(model, config, multipliers, backend["device"]))
                trajectories.append(runner)
            def stacked(name):
                return torch.stack([torch.as_tensor(item[name]) for item in trajectories]).cpu()
            multiplier_hash = _hash_array(model.K)
            initial_hashes = [item["initial_hash"] for item in trajectories]
            payload = {name: stacked(name) for name in
                       ("video", "reference_increment", "reference_branch_increment", "reference_map")}
            payload.update(kernel_weights=torch.as_tensor(model.weights.copy(), dtype=torch.float64),
                           kernel_multiplier=torch.as_tensor(model.K.copy(), dtype=torch.float64),
                           time=torch.arange(config.production_steps // config.sample_every + 1,
                                             dtype=torch.float64) * config.interval,
                           diagnostics={name: stacked(name) for name in ("mass", "variance", "energy")})
            payload["diagnostics"].update(initial_hashes=initial_hashes,
                quartic_balance_error=torch.tensor([item["quartic_balance_error"] for item in trajectories], dtype=torch.float64))
            payload["metadata"] = dict(schema_version=SCHEMA_VERSION, signature=signature,
                source=source, config=config.to_dict(), ell=ell, split=split,
                seeds=list(config.seeds(split)), backend=backend, dt=config.dt,
                interval=config.interval, duration=config.duration, dV=config.dx ** 2,
                area=(config.n * config.dx) ** 2, channel_names=list(CHANNEL_NAMES),
                branch_distance="periodic Chebyshev lattice distance; local conservative + active in radius 0",
                reference_increment_units="spatially integrated medium entropy per saved interval",
                reference_map_units="medium EP density per physical time; spatial integral multiplies by dx**2",
                entropy_convention="Euler-Maruyama drift; midpoint chemical potentials at every solver step",
                interpretation="medium entropy excludes stochastic system entropy; learned branch maps need not match this gauge",
                kernel_multiplier_sha256=multiplier_hash)
            # The quadratic shell identity must reconstruct the full reference.
            torch.testing.assert_close(payload["reference_branch_increment"].sum(-1),
                                       payload["reference_increment"].sum(-1), atol=1e-8, rtol=1e-9,
                                       msg=lambda detail: (
                                           "Kernel-shell EP does not reconstruct midpoint medium EP. "
                                           "Update LRAMB/lramb/model.py and torch_backend.py to the "
                                           "Hermitian-projected Euler step and regenerate in a new "
                                           "output directory. Increasing the tolerance would conceal "
                                           "a Fourier/real-field energy inconsistency.\n" + detail))
            torch.testing.assert_close(payload["reference_map"].sum((-2, -1)) * config.dx ** 2,
                                       payload["reference_increment"].sum(1) / config.duration,
                                       atol=1e-9, rtol=1e-9)
            _atomic_torch(payload, path)
            condition["kernel_multiplier_sha256"] = multiplier_hash
            condition.setdefault("initial_hashes", {})[split] = initial_hashes
            _atomic_json(manifest, output_dir / "dataset.json")
    manifest["complete"] = True
    _atomic_json(manifest, output_dir / "dataset.json")
    return manifest


__all__ = ["DatasetConfig", "dataset_preset", "condition_id", "generate_datasets",
           "load_dataset", "load_config", "read_manifest", "CHANNEL_NAMES", "SPLITS"]
