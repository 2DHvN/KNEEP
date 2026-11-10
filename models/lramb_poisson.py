"""Generate self-contained Poisson-range datasets with the sibling LRAMB solver.

Generation imports LRAMB; loading and fitting saved datasets only need PyTorch.
Ground-truth entropy uses every solver step, never differences of saved movies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
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


SCHEMA_VERSION = 2
GENERATOR_REVISION = "canonical-fourier-v4-strengths-wall-movies"
CHANNEL_NAMES = ("local conservative", "kernel conservative", "local active")
SPLITS = ("train", "validation", "test")
_SEED_OFFSETS = {"train": 0, "validation": 100_000, "test": 200_000}


@dataclass(frozen=True)
class DatasetConfig:
    """A fresh dynamical sweep over ranges and optional coupling strengths.

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
    strengths: tuple[float, ...] | None = None
    include_no_interaction: bool = False
    burn_steps: int = 4_000
    production_steps: int = 8_192
    sample_every: int = 8
    blocks: int = 8
    train_trajectories: int = 4
    validation_trajectories: int = 2
    test_trajectories: int = 3
    base_seed: int = 17
    initial_mode: str = "stripe"
    movie_trajectories: int = 0
    movie_window_steps: int = 16

    def __post_init__(self) -> None:
        object.__setattr__(self, "ells", tuple(float(x) for x in self.ells))
        if self.strengths is not None:
            object.__setattr__(self, "strengths", tuple(float(x) for x in self.strengths))
            if (not self.strengths or len(set(self.strengths)) != len(self.strengths)
                    or any(not math.isfinite(x) or x <= 0 for x in self.strengths)):
                raise ValueError("strengths must contain distinct positive finite couplings")
        if (len(self.ells) < 2 or len(set(self.ells)) != len(self.ells)
                or any(not math.isfinite(x) or x <= 0 for x in self.ells)):
            raise ValueError("ells must contain at least two distinct positive finite ranges")
        integer_fields = ("n", "burn_steps", "production_steps", "sample_every", "blocks",
                          "train_trajectories", "validation_trajectories", "test_trajectories",
                          "base_seed", "movie_trajectories", "movie_window_steps")
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
        if not 0 <= self.movie_trajectories <= self.test_trajectories or self.movie_window_steps < 1:
            raise ValueError("movie_trajectories must lie in [0,test_trajectories]; movie_window_steps must be positive")
        if self.production_steps % (self.sample_every * self.blocks):
            raise ValueError("production_steps must be divisible by sample_every * blocks")
        if self.production_steps // (self.sample_every * self.blocks) < 2:
            raise ValueError("at least two saved EP intervals per time block are needed")
        for name in ("dx", "dt", "a", "b", "kappa", "lam", "mobility", "D", "mean", "strength"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if min(self.dx, self.dt, self.b, self.mobility, self.D) <= 0 or self.kappa < 0:
            raise ValueError("dx, dt, b, mobility, D must be positive; kappa >= 0")
        if self.initial_mode not in {"stripe", "random", "domain_wall"}:
            raise ValueError("initial_mode must be stripe, random or domain_wall")
        if self.initial_mode == "domain_wall" and (self.a >= 0 or self.kappa <= 0 or abs(self.mean) > 1e-12):
            raise ValueError("domain_wall initialization requires a<0, kappa>0 and mean=0")
        if not isinstance(self.include_no_interaction, bool):
            raise ValueError("include_no_interaction must be bool")

    @property
    def interval(self) -> float:
        return self.dt * self.sample_every

    @property
    def duration(self) -> float:
        return self.dt * self.production_steps

    @property
    def max_distance(self) -> int:
        return self.n // 2

    @property
    def train_pairs(self) -> int:
        return self.train_trajectories * (self.production_steps // self.sample_every)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ells"] = list(self.ells)
        result["strengths"] = list(self.strengths) if self.strengths is not None else None
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
    if name == "production":
        # 128 independent trajectories x 8192 adjacent frame pairs = 2**20
        # training pairs per condition, including the single g=0 control.
        return DatasetConfig(burn_steps=16_000, sample_every=1,
                             train_trajectories=128, validation_trajectories=4,
                             test_trajectories=4, include_no_interaction=True)
    if name == "production_v2":
        # Increase independent trajectories while keeping observation cadence,
        # dynamics and held-out test seeds identical to production.
        return replace(dataset_preset("production"), train_trajectories=256,
                       validation_trajectories=8)
    if name == "production_v3":
        return DatasetConfig(strengths=(0.5, 1.0, 2.0), D=0.1, mean=0.0,
            burn_steps=16_000, production_steps=1_024, sample_every=1,
            train_trajectories=1_024, validation_trajectories=64, test_trajectories=128,
            include_no_interaction=True, initial_mode="domain_wall",
            movie_trajectories=1, movie_window_steps=16)
    if name == "reference":
        return DatasetConfig(n=64, burn_steps=40_000, production_steps=64_000,
                             sample_every=40, blocks=8)
    raise ValueError("preset must be quick, demo, refined, production, production_v2, production_v3 or reference")


def condition_id(ell: float, strength: float | None = None) -> str:
    value = float(ell)
    if not math.isfinite(value) or value < 0:
        raise ValueError("ell must be nonnegative and finite; zero denotes the g=0 control")
    if value == 0:
        return "no_interaction"
    if strength is not None:
        coupling = float(strength)
        if not math.isfinite(coupling) or coupling <= 0:
            raise ValueError("strength must be positive and finite for a range condition")
        return f"g_{coupling:.12g}_ell_{value:.12g}"
    return f"ell_{value:.12g}"


def dataset_conditions(config: DatasetConfig) -> list[dict[str, Any]]:
    """Physical range conditions and one optional independent g=0 control.

    The control's ``ell=0`` is a manifest identifier only. The solver receives
    the valid range ell=1 and strength=0, so it has no conservative LR force.
    Its active term and local conservative free energy are unchanged.
    """
    couplings = config.strengths if config.strengths is not None else (config.strength,)
    result = [dict(ell=ell, solver_ell=ell, strength=strength,
                   condition_id=condition_id(ell, strength if config.strengths is not None else None), is_control=False,
                   label=f"Poisson ell={ell:g}, g={strength:g}")
              for strength in couplings for ell in config.ells]
    if config.include_no_interaction:
        result.append(dict(ell=0.0, solver_ell=1.0, strength=0.0,
                           condition_id="no_interaction", is_control=True,
                           label="No long-range interaction (g=0)"))
    ids = [item["condition_id"] for item in result]
    if len(set(ids)) != len(ids):
        raise ValueError("Range/coupling values are too close to produce distinct condition names")
    return result


def read_manifest(output_dir: str | Path) -> dict[str, Any]:
    with (Path(output_dir) / "dataset.json").open(encoding="utf-8") as stream:
        return json.load(stream)


@lru_cache(maxsize=2)
def _prepared_manifest(path: str, stamp: tuple[int, int]) -> dict[str, Any]:
    """Workers reuse large immutable manifests until the parent rewrites them."""
    del stamp  # The stat stamp is a cache key, never a scientific input.
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def load_config(output_dir: str | Path) -> DatasetConfig:
    return DatasetConfig.from_dict(read_manifest(output_dir)["config"])


def _split_record(output_dir: str | Path, ell: float | str, split: str,
                  strength: float | None = None):
    if split == "val":
        split = "validation"
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    directory = Path(output_dir)
    manifest_path = directory / "dataset.json"
    stat = manifest_path.stat()
    manifest = _prepared_manifest(str(manifest_path.resolve()), (stat.st_mtime_ns, stat.st_size))
    if isinstance(ell, str):
        matching = [c for c in manifest["conditions"] if c["condition_id"] == ell]
    else:
        matching = [c for c in manifest["conditions"] if float(c["ell"]) == float(ell)]
    if strength is not None:
        matching = [c for c in matching
                    if float(c.get("strength", manifest["config"]["strength"])) == float(strength)]
    if not matching:
        raise ValueError(f"Condition {ell!r}, strength={strength!r} is absent from dataset.json")
    if len(matching) != 1:
        raise ValueError(f"ell={ell!r} is ambiguous across coupling strengths; "
                         "pass an exact condition_id string or strength=...")
    path = directory / matching[0]["files"][split]
    if path.suffix != ".json":
        return manifest, matching[0], path, None
    if not path.is_file():
        raise FileNotFoundError(f"Split is not finalized yet: {path}")
    with path.open(encoding="utf-8") as stream:
        index = json.load(stream)
    if index.get("signature") != manifest["signature"]:
        raise ValueError("saved split does not match dataset.json; regenerate the dataset")
    return manifest, matching[0], path, index


def open_video_dataset(output_dir: str | Path, ell: float | str, split: str = "train",
                       cache_size: int = 128, *, strength: float | None = None):
    """Open a trajectory-sharded movie without concatenating training fields.

    The adapter maps at most ``cache_size`` shard files and retrieves adjacent
    pairs on demand. Loading saved data does not import the LRAMB solver.
    """
    from utils.video_data import ShardedVideo

    manifest, _, path, index = _split_record(output_dir, ell, split, strength)
    if index is None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["metadata"]["signature"] != manifest["signature"]:
            raise ValueError("saved split does not match dataset.json; regenerate the dataset")
        return payload["video"]
    paths = [Path(output_dir) / shard["file"] for shard in index["shards"]]
    return ShardedVideo(paths, max_cached_shards=cache_size,
                        shape=tuple(index["video_shape"]))


def load_dataset(output_dir: str | Path, ell: float | str, split: str = "test", *,
                 lazy: bool | None = None, cache_size: int = 128,
                 strength: float | None = None) -> dict[str, Any]:
    """Load references and CPU fields, mapping large train movies on demand.

    Schema-1 dense files remain supported. For schema-2 shards, train defaults
    to ``ShardedVideo`` and validation/test default to dense tensors. Explicit
    ``lazy=False`` concatenates any split; avoid it for million-pair training.
    """
    manifest, _, path, index = _split_record(output_dir, ell, split, strength)
    if index is None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["metadata"]["signature"] != manifest["signature"]:
            raise ValueError("saved split does not match dataset.json; regenerate the dataset")
        return payload
    if lazy is None:
        lazy = split == "train"
    tensor_names = ["reference_increment", "reference_branch_increment", "reference_map"]
    if not lazy:
        tensor_names.append("video")
    tensors = {name: [] for name in tensor_names}
    diagnostics: dict[str, list] = {name: [] for name in
        ("mass", "variance", "energy", "quartic_balance_error", "fourier_state_error")}
    initial_hashes = []
    common = {}
    for shard in index["shards"]:
        payload = torch.load(Path(output_dir) / shard["file"], map_location="cpu",
                             weights_only=True, mmap=True)
        if payload.get("metadata", {}).get("signature") != manifest["signature"]:
            raise ValueError(f"Shard signature mismatch: {shard['file']}")
        for name in tensor_names:
            tensors[name].append(payload[name])
        for name in diagnostics:
            diagnostics[name].append(payload["diagnostics"][name])
        initial_hashes.extend(payload["diagnostics"]["initial_hashes"])
        if not common:
            common = {name: payload[name] for name in
                      ("kernel_weights", "kernel_multiplier", "time")}
    result = {name: torch.cat(values, dim=0) for name, values in tensors.items()}
    result.update(common)
    result["diagnostics"] = {name: torch.cat(values, dim=0)
                             for name, values in diagnostics.items()}
    result["diagnostics"]["initial_hashes"] = initial_hashes
    result["metadata"] = index["metadata"]
    if lazy:
        result["video"] = open_video_dataset(output_dir, ell, split, cache_size, strength=strength)
    if result["metadata"]["signature"] != manifest["signature"]:
        raise ValueError("saved split does not match dataset.json; regenerate the dataset")
    return result


def load_movie_dataset(output_dir: str | Path, condition_id: str | float,
                       trajectory: int = 0, *, strength: float | None = None) -> dict[str, Any]:
    """Load only one test trajectory's full-step window-averaged medium maps.

    ``fields`` are window-end fields, while ``time`` locates the midpoint of
    each EP averaging window. Starts/ends index saved production intervals.
    Movie maps already include dx**2: their spatial sum is total EP / time.
    No movie tensor is concatenated by ``load_dataset``.
    """
    manifest, _, path, index = _split_record(output_dir, condition_id, "test", strength)
    if isinstance(trajectory, bool) or not isinstance(trajectory, int) or trajectory < 0:
        raise ValueError("trajectory must be a nonnegative integer")
    if index is None:
        raise ValueError("This legacy dense dataset has no saved window-resolved medium EP movie")
    if trajectory >= len(index["shards"]):
        raise IndexError("Requested movie trajectory is outside the test split")
    shard = index["shards"][trajectory]
    payload = torch.load(Path(output_dir) / shard["file"], map_location="cpu",
                         weights_only=True, mmap=True)
    if payload.get("metadata", {}).get("signature") != manifest["signature"]:
        raise ValueError("Movie shard signature does not match dataset.json")
    if "movie" not in payload:
        raise ValueError(f"No medium EP movie was captured for test trajectory {trajectory}; "
                         "increase movie_trajectories when generating the dataset")
    result = {name: value.clone() if isinstance(value, torch.Tensor) else value
              for name, value in payload["movie"].items()}
    result["metadata"] = dict(payload["metadata"], trajectory_index=trajectory,
        field_time="end of averaging window", ep_time="midpoint of averaging window",
        movie_units="per-site medium EP per physical time; sum sites without another dx**2",
        movie_window_steps=manifest["config"].get("movie_window_steps", 16))
    return result


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


def initial_field(model, cfg: DatasetConfig) -> np.ndarray:
    """Initialize the configured field without changing the sibling solver.

    ``domain_wall`` is deterministic and periodic: interfaces sit at L/4 and
    3L/4, the two bulks approach +/-sqrt(-a/b), and the tanh width is
    sqrt(2*kappa/abs(a)). Its zero mean follows from half-period antisymmetry.
    Other modes retain LRAMB's seeded initialization unchanged.
    """
    if cfg.initial_mode != "domain_wall":
        return model.initial_field(cfg.initial_mode)
    length = cfg.n * cfg.dx
    x = np.arange(cfg.n, dtype=np.float64) * cfg.dx
    signed_distance = length / 4 - np.abs(x - length / 2)
    bulk = math.sqrt(-cfg.a / cfg.b)
    width = math.sqrt(2 * cfg.kappa / abs(cfg.a))
    profile = bulk * np.tanh(signed_distance / width)
    field = np.tile(profile, (cfg.n, 1))
    # Remove floating-point mean roundoff only; never shift the bulk phases.
    return field - field.mean()


def _movie_layout(cfg: DatasetConfig) -> dict[str, np.ndarray]:
    """Windows are counted in saved production intervals, including a tail."""
    intervals = cfg.production_steps // cfg.sample_every
    starts = np.arange(0, intervals, cfg.movie_window_steps, dtype=np.int64)
    ends = np.minimum(starts + cfg.movie_window_steps, intervals)
    return dict(fields=np.empty((len(starts), cfg.n, cfg.n), dtype=np.float32),
        medium_rate_maps=np.empty((len(starts), cfg.n, cfg.n), dtype=np.float64),
        time=(starts + ends) * (cfg.interval / 2),
        window_start=starts, window_end=ends,
        window_duration=(ends - starts) * cfg.interval,
        field_time=ends * cfg.interval)


def _run_numpy(model, cfg: DatasetConfig, multipliers: np.ndarray, *,
               capture_movie: bool = False) -> dict[str, Any]:
    phi = initial_field(model, cfg)
    initial_hash = _hash_array(phi)
    ph = np.fft.fft2(phi, norm="ortho")
    max_fourier_state_error = 0.0

    def advance():
        nonlocal max_fourier_state_error
        new, propagated = model.step(phi, ph)
        # The real field is authoritative. Do not carry a spectrum that an
        # external solver's inverse FFT may have projected differently.
        canonical = np.fft.fft2(new, norm="ortho")
        max_fourier_state_error = max(
            max_fourier_state_error, float(np.max(np.abs(propagated - canonical))))
        return new, canonical

    for step in range(cfg.burn_steps):
        phi, ph = advance()
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
    movie = _movie_layout(cfg) if capture_movie else None
    movie_index = 0
    movie_entropy = np.zeros((cfg.n, cfg.n), dtype=np.float64) if capture_movie else None

    def record(index):
        video[index, 0] = phi
        mass[index], variance[index] = phi.mean(), phi.var()
        energies[index] = model.energy_channels(phi, ph)[0]
        return (model.dV * cfg.strength / 2
                * np.einsum("rij,ij->r", multipliers, np.abs(ph) ** 2, optimize=False))

    previous_shell_energy = record(0)
    for step in range(1, cfg.production_steps + 1):
        new, new_hat = advance()
        ds = model.entropy_maps(phi, new, ph, new_hat)
        ep_maps += ds
        if capture_movie:
            movie_entropy += model.dV * ds.sum(axis=0)
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
            if capture_movie and index == movie["window_end"][movie_index]:
                movie["fields"][movie_index] = phi
                movie["medium_rate_maps"][movie_index] = movie_entropy / movie["window_duration"][movie_index]
                movie_entropy.fill(0)
                movie_index += 1
    if not np.isfinite(video).all() or np.max(np.abs(mass - mass[0])) > 1e-10:
        raise FloatingPointError("Nonfinite trajectory or mass conservation failure")
    result = dict(video=video, reference_increment=reference,
                reference_branch_increment=branches, reference_map=ep_maps / cfg.duration,
                mass=mass, variance=variance, energy=energies,
                initial_hash=initial_hash, quartic_balance_error=max_balance_error,
                fourier_state_error=max_fourier_state_error)
    if capture_movie:
        result["movie"] = movie
    return result


def _run_torch(model, cfg: DatasetConfig, multipliers: np.ndarray, device: str, *,
               capture_movie: bool = False) -> dict[str, Any]:
    # Reuse the actual LRAMB solver kernels, with float64 dynamics and entropy.
    from lramb.torch_backend import _step, _midpoint

    initial = initial_field(model, cfg)
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
    max_fourier_state_error = torch.zeros((), **kw)
    movie = _movie_layout(cfg) if capture_movie else None
    movie_index = 0
    movie_entropy = torch.zeros((cfg.n, cfg.n), **kw) if capture_movie else None

    def advance():
        nonlocal max_fourier_state_error
        normal = torch.randn(phi.shape, generator=rng, **kw)
        new, propagated = _step(phi, ph, normal, linear, q2, noise_scale,
                                cfg.dt * cfg.mobility, cfg.b, cfg.lam, cfg.dx)
        # This generator enforces field/spectrum consistency independently of
        # the imported LRAMB version and the FFT backend's inverse projection.
        canonical = torch.fft.rfft2(new, norm="ortho")
        max_fourier_state_error = torch.maximum(
            max_fourier_state_error, (propagated - canonical).abs().max())
        return new, canonical

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
            if capture_movie:
                movie_entropy += model.dV * ds.sum(dim=0)
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
                if capture_movie and index == movie["window_end"][movie_index]:
                    movie["fields"][movie_index] = phi.float().cpu().numpy()
                    movie["medium_rate_maps"][movie_index] = (
                        movie_entropy / movie["window_duration"][movie_index]).cpu().numpy()
                    movie_entropy.zero_()
                    movie_index += 1
        if not torch.isfinite(video).all() or (mass - mass[0]).abs().max() > 1e-10:
            raise FloatingPointError("Nonfinite trajectory or mass conservation failure")
    result = dict(video=video, reference_increment=reference.cpu(),
                reference_branch_increment=branches.cpu(), reference_map=(ep_maps / cfg.duration).cpu(),
                mass=mass.cpu(), variance=variance.cpu(), energy=energies.cpu(),
                initial_hash=initial_hash, quartic_balance_error=float(max_balance_error.cpu()),
                fourier_state_error=float(max_fourier_state_error.cpu()))
    if capture_movie:
        result["movie"] = movie
    return result


def _check_trajectory_reference(trajectory, *, ell, split, seed, backend):
    """Check one trajectory before generating the rest of an expensive split."""
    channels = torch.as_tensor(trajectory["reference_increment"])
    branches = torch.as_tensor(trajectory["reference_branch_increment"])
    # Remove the local terms that are assigned to radius zero. This exposes
    # the kernel channel alone instead of masking it with large local EP.
    kernel_shells = branches.clone()
    kernel_shells[:, 0] -= channels[:, 0] + channels[:, 2]
    try:
        torch.testing.assert_close(kernel_shells.sum(-1), channels[:, 1],
                                   atol=1e-8, rtol=1e-9)
    except AssertionError as error:
        mismatch = (kernel_shells.sum(-1) - channels[:, 1]).abs()
        index = int(mismatch.argmax())
        raise RuntimeError(
            f"{GENERATOR_REVISION}: kernel EP inconsistency; ell={ell:g}, "
            f"split={split}, seed={seed}, backend={backend}, interval={index}. "
            f"shell={float(kernel_shells[index].sum()):.17g}, "
            f"midpoint={float(channels[index, 1]):.17g}, "
            f"max_abs_error={float(mismatch.max()):.6g}, "
            f"max_propagated_spectrum_error={trajectory['fourier_state_error']:.6g}. "
            f"torch={torch.__version__}, generator={Path(__file__).resolve()}. "
            "This check compares the kernel alone, before stacking trajectories."
        ) from error


def _generation_source(root: Path) -> dict[str, str]:
    return dict(lramb_root=str(root),
                lramb_model_sha256=sha256((root / "lramb/model.py").read_bytes()).hexdigest(),
                lramb_torch_backend_sha256=sha256((root / "lramb/torch_backend.py").read_bytes()).hexdigest(),
                generator_sha256=sha256(Path(__file__).read_bytes()).hexdigest())


def prepare_generation(config: DatasetConfig, output_dir: str | Path,
                       lramb_root: str | Path | None = None, device: str = "cpu",
                       force: bool = False) -> dict[str, Any]:
    """Prepare a manifest once, before independent worker processes are started.

    Returned ``trajectory_tasks`` are JSON-serializable. Each task writes only
    its own atomic trajectory shard; workers never update shared manifest JSON.
    CUDA device indices do not enter the dataset signature, allowing the same
    prepared tasks to be distributed over cuda:0, cuda:1, and so on.
    """
    if not isinstance(config, DatasetConfig):
        raise TypeError("config must be a DatasetConfig")
    root, _ = _lramb_module(lramb_root)
    backend = _backend(device)
    backend["device"] = torch.device(backend["device"]).type
    source = _generation_source(root)
    signature_input = dict(schema_version=SCHEMA_VERSION, config=config.to_dict(),
                           source={k: v for k, v in source.items() if k != "lramb_root"},
                           backend=backend)
    signature = sha256(json.dumps(signature_input, sort_keys=True).encode()).hexdigest()
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    old = read_manifest(directory) if (directory / "dataset.json").exists() else None
    if old and old.get("signature") != signature and not force:
        raise ValueError("Dataset configuration, solver source or backend changed. "
                         "Use a new output directory or force=True.")
    frames = config.production_steps // config.sample_every + 1
    manifest = dict(schema_version=SCHEMA_VERSION, generator_revision=GENERATOR_REVISION,
        storage="trajectory_shards", signature=signature, config=config.to_dict(),
        source=source, backend=backend, complete=False,
        splits={split: dict(trajectories=len(config.seeds(split)), seeds=list(config.seeds(split)),
                            adjacent_pairs=len(config.seeds(split)) * (frames - 1))
                for split in SPLITS},
        train_pairs_per_condition=config.train_pairs,
        train_video_bytes_per_condition=config.train_trajectories * frames * config.n ** 2 * 4,
        conditions=[], trajectory_tasks=[])
    for condition in dataset_conditions(config):
        cid = condition["condition_id"]
        entry = dict(condition, files={split: f"data/{cid}/{split}.json" for split in SPLITS},
                     shards={split: [] for split in SPLITS})
        for split in SPLITS:
            for index, seed in enumerate(config.seeds(split)):
                file = f"data/{cid}/{split}/trajectory_{index:05d}_seed_{seed}.pt"
                shard = dict(file=file, trajectory_index=index, seed=seed,
                             frames=frames, pairs=frames - 1,
                             has_movie=split == "test" and index < config.movie_trajectories)
                entry["shards"][split].append(shard)
                manifest["trajectory_tasks"].append(dict(shard, output_dir=str(directory),
                    signature=signature, condition_id=cid, ell=condition["ell"],
                    strength=condition["strength"],
                    split=split, force=bool(force)))
        manifest["conditions"].append(entry)
    _atomic_json(manifest, directory / "dataset.json")
    return manifest


def _trajectory_metadata(manifest, condition, split, seeds, multiplier_hash, backend):
    config = DatasetConfig.from_dict(manifest["config"])
    return dict(schema_version=SCHEMA_VERSION, signature=manifest["signature"],
        generator_revision=GENERATOR_REVISION, source=manifest["source"],
        config=config.to_dict(), ell=condition["ell"], solver_ell=condition["solver_ell"],
        strength=condition["strength"], is_control=condition["is_control"],
        condition_id=condition["condition_id"], label=condition["label"],
        split=split, seeds=list(seeds), backend=backend, dt=config.dt,
        interval=config.interval, duration=config.duration, dV=config.dx ** 2,
        area=(config.n * config.dx) ** 2, channel_names=list(CHANNEL_NAMES),
        branch_distance="periodic Chebyshev lattice distance; local conservative + active in radius 0",
        reference_increment_units="spatially integrated medium entropy per saved interval",
        reference_map_units="medium EP density per physical time; spatial integral multiplies by dx**2",
        entropy_convention="Euler-Maruyama drift; midpoint chemical potentials at every solver step",
        interpretation="medium entropy excludes stochastic system entropy; learned branch maps need not match this gauge",
        movie_window_steps=config.movie_window_steps,
        movie_units="per-site medium EP per physical time, including dx**2",
        kernel_multiplier_sha256=multiplier_hash)


def generate_trajectory_task(task: dict[str, Any], *, device: str = "cpu",
                             progress: bool = True) -> dict[str, Any]:
    """Generate or reuse one prepared trajectory; safe for separate processes.

    Give every concurrent worker a distinct prepared task. The scheduler owns
    preparation/finalization, and must not schedule the same task twice.
    """
    directory = Path(task["output_dir"])
    manifest_path = directory / "dataset.json"
    stat = manifest_path.stat()
    manifest = _prepared_manifest(str(manifest_path.resolve()), (stat.st_mtime_ns, stat.st_size))
    if manifest["signature"] != task["signature"]:
        raise ValueError("Prepared task signature no longer matches dataset.json")
    condition = next((item for item in manifest["conditions"]
                      if item["condition_id"] == task["condition_id"]), None)
    if condition is None or task["split"] not in SPLITS:
        raise ValueError("Task condition/split is absent from the prepared manifest")
    split = task["split"]
    expected = condition["shards"][split][task["trajectory_index"]]
    if any(task[key] != expected[key] for key in ("file", "seed", "trajectory_index")):
        raise ValueError("Task trajectory does not match the prepared manifest")
    backend = _backend(device)
    identity = dict(backend, device=torch.device(backend["device"]).type)
    if identity != manifest["backend"]:
        raise ValueError("Worker backend differs from the prepared dataset backend")
    path = directory / task["file"]
    if path.is_file() and not task.get("force", False):
        cached = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        metadata = cached.get("metadata", {})
        if (metadata.get("signature") != manifest["signature"]
                or metadata.get("condition_id") != condition["condition_id"]
                or metadata.get("seeds") != [task["seed"]]):
            raise ValueError(f"Cached {path} has mismatching metadata; use force=True or a new directory")
        if progress:
            print(f"{condition['condition_id']}, {split} seed={task['seed']}: using matching shard", flush=True)
        return dict(file=task["file"], condition_id=condition["condition_id"], split=split,
                    seed=task["seed"], cached=True)
    config = DatasetConfig.from_dict(manifest["config"])
    root, lramb = _lramb_module(manifest["source"]["lramb_root"])
    if _generation_source(root) != manifest["source"]:
        raise ValueError("Solver/generator source changed after preparation; prepare again")
    effective = replace(config, strength=condition["strength"])
    parameters = lramb.Parameters(**{key: getattr(effective, key) for key in
        ("n", "dx", "dt", "a", "b", "kappa", "lam", "mobility", "D", "mean")},
        seed=task["seed"])
    kernel = lramb.KernelSpec("poisson", strength=condition["strength"], ell=condition["solver_ell"])
    model = lramb.ActiveModelB(parameters, kernel,
        multiplier=np.zeros((config.n, config.n)) if condition["is_control"] else None)
    multipliers = _shell_multipliers(model)
    if progress:
        print(f"{GENERATOR_REVISION}: {condition['condition_id']}, {split} "
              f"seed={task['seed']}, {config.burn_steps + config.production_steps:,} steps "
              f"({backend['name']}/{backend['device']})", flush=True)
    capture_movie = split == "test" and task["trajectory_index"] < config.movie_trajectories
    trajectory = (_run_numpy(model, effective, multipliers, capture_movie=capture_movie)
                  if backend["name"] == "numpy" else
                  _run_torch(model, effective, multipliers, backend["device"], capture_movie=capture_movie))
    _check_trajectory_reference(trajectory, ell=condition["ell"], split=split,
                                seed=task["seed"], backend=backend)
    payload = {name: torch.as_tensor(trajectory[name]).unsqueeze(0).cpu() for name in
               ("video", "reference_increment", "reference_branch_increment", "reference_map")}
    payload.update(kernel_weights=torch.as_tensor(model.weights.copy(), dtype=torch.float64),
        kernel_multiplier=torch.as_tensor(model.K.copy(), dtype=torch.float64),
        time=torch.arange(config.production_steps // config.sample_every + 1,
                          dtype=torch.float64) * config.interval,
        diagnostics={name: torch.as_tensor(trajectory[name]).unsqueeze(0).cpu()
                     for name in ("mass", "variance", "energy")})
    payload["diagnostics"].update(initial_hashes=[trajectory["initial_hash"]],
        quartic_balance_error=torch.tensor([trajectory["quartic_balance_error"]], dtype=torch.float64),
        fourier_state_error=torch.tensor([trajectory["fourier_state_error"]], dtype=torch.float64))
    payload["metadata"] = _trajectory_metadata(manifest, condition, split, [task["seed"]],
                                               _hash_array(model.K), backend)
    if capture_movie:
        payload["movie"] = {name: torch.as_tensor(value).cpu()
                            for name, value in trajectory["movie"].items()}
    # Runtime scientific identities are retained, with their original tolerances.
    torch.testing.assert_close(payload["reference_branch_increment"].sum(-1),
                               payload["reference_increment"].sum(-1), atol=1e-8, rtol=1e-9)
    torch.testing.assert_close(payload["reference_map"].sum((-2, -1)) * config.dx ** 2,
                               payload["reference_increment"].sum(1) / config.duration,
                               atol=1e-9, rtol=1e-9)
    _atomic_torch(payload, path)
    return dict(file=task["file"], condition_id=condition["condition_id"], split=split,
                seed=task["seed"], cached=False)


def finalize_generation(output_dir: str | Path) -> dict[str, Any]:
    """Parent-only completion: validate every shard and write split indexes."""
    directory = Path(output_dir)
    manifest = read_manifest(directory)
    config = DatasetConfig.from_dict(manifest["config"])
    frames = config.production_steps // config.sample_every + 1
    for condition in manifest["conditions"]:
        for split in SPLITS:
            initial_hashes = []
            multiplier_hash = None
            shards = condition["shards"][split]
            for shard in shards:
                path = directory / shard["file"]
                if not path.is_file():
                    raise FileNotFoundError(f"Missing trajectory shard: {path}")
                payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                metadata = payload.get("metadata", {})
                if (metadata.get("signature") != manifest["signature"]
                        or metadata.get("condition_id") != condition["condition_id"]
                        or metadata.get("split") != split or metadata.get("seeds") != [shard["seed"]]):
                    raise ValueError(f"Trajectory shard has mismatching metadata: {path}")
                if tuple(payload["video"].shape) != (1, frames, 1, config.n, config.n):
                    raise ValueError(f"Trajectory shard has the wrong movie shape: {path}")
                if shard.get("has_movie", False):
                    movie = payload.get("movie")
                    windows = math.ceil((frames - 1) / config.movie_window_steps)
                    if (movie is None or tuple(movie["fields"].shape) != (windows, config.n, config.n)
                            or movie["medium_rate_maps"].shape != movie["fields"].shape):
                        raise ValueError(f"Selected trajectory is missing complete medium EP movie windows: {path}")
                current_hash = metadata["kernel_multiplier_sha256"]
                if multiplier_hash is not None and current_hash != multiplier_hash:
                    raise ValueError(f"Kernel hash differs within a split: {path}")
                multiplier_hash = current_hash
                initial_hashes.extend(payload["diagnostics"]["initial_hashes"])
            condition["kernel_multiplier_sha256"] = multiplier_hash
            condition.setdefault("initial_hashes", {})[split] = initial_hashes
            index = dict(schema_version=SCHEMA_VERSION, signature=manifest["signature"],
                storage="trajectory_shards", complete=True, shards=shards,
                video_shape=[len(shards), frames, 1, config.n, config.n],
                adjacent_pairs=len(shards) * (frames - 1),
                metadata=_trajectory_metadata(manifest, condition, split, config.seeds(split),
                                               multiplier_hash, manifest["backend"]))
            _atomic_json(index, directory / condition["files"][split])
    # Same seed gives the same initial field across ranges, including g=0.
    for split in SPLITS:
        expected_hashes = manifest["conditions"][0]["initial_hashes"][split]
        if any(item["initial_hashes"][split] != expected_hashes for item in manifest["conditions"]):
            raise ValueError(f"Initial fields differ across conditions for split={split}")
    manifest["complete"] = True
    _atomic_json(manifest, directory / "dataset.json")
    return manifest


def generate_datasets(config: DatasetConfig, output_dir: str | Path,
                      lramb_root: str | Path | None = None, device: str = "cpu",
                      progress: bool = True, force: bool = False) -> dict[str, Any]:
    """Serial convenience wrapper around the parallel-safe trajectory task API.

    A single shard is resident while simulating. Matching shards resume after
    interruption; no full multi-trajectory train movie is concatenated in memory.
    The script scheduler can distribute ``trajectory_tasks`` among GPUs and
    call ``finalize_generation`` once all workers have finished.
    """
    manifest = prepare_generation(config, output_dir, lramb_root, device, force)
    if progress:
        print(f"generator={GENERATOR_REVISION}; module={Path(__file__).resolve()}; "
              f"LRAMB={manifest['source']['lramb_root']}; backend={manifest['backend']}; "
              f"train pairs/condition={config.train_pairs:,}; torch={torch.__version__}", flush=True)
    for task in manifest["trajectory_tasks"]:
        generate_trajectory_task(task, device=device, progress=progress)
    return finalize_generation(output_dir)


__all__ = ["DatasetConfig", "dataset_preset", "condition_id", "dataset_conditions",
           "generate_datasets", "prepare_generation", "generate_trajectory_task",
           "finalize_generation", "open_video_dataset", "load_dataset", "load_config",
           "load_movie_dataset", "initial_field", "read_manifest", "CHANNEL_NAMES", "SPLITS"]
