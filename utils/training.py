"""Training and ordered evaluation for periodic-field KNEEP experiments."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch
from tqdm import trange

from shell_force import ShellForceKNEEP2D
from utils.video_data import ShardedVideo, video_channel_normalization


Video = torch.Tensor | ShardedVideo


@dataclass(frozen=True)
class TrainingConfig:
    alpha: float = -0.5
    iterations: int = 10_000
    train_batch_size: int = 2_048
    validation_batch_size: int = 2_048
    prediction_batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    validate_every: int = 100
    train_fraction: float = 0.8
    # None retains the historical single forward/backward per optimizer step.
    # Smaller microbatches accumulate the same logical-batch mean gradient.
    train_micro_batch_size: int | None = None
    # Cosine counts optimizer updates; plateau counts full validation events.
    lr_scheduler: str = "none"
    lr_min: float = 1e-5
    lr_warmup_steps: int = 0
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 3
    lr_plateau_threshold: float = 1e-3
    lr_plateau_cooldown: int = 1

    def __post_init__(self) -> None:
        if self.lr_scheduler not in {"none", "cosine", "plateau"}:
            raise ValueError("lr_scheduler must be 'none', 'cosine' or 'plateau'")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.lr_min) or self.lr_min < 0:
            raise ValueError("lr_min must be finite and nonnegative")
        if self.lr_scheduler != "none":
            if self.lr_min > self.learning_rate:
                raise ValueError("lr_min must not exceed learning_rate")
            if (isinstance(self.lr_warmup_steps, bool)
                    or not isinstance(self.lr_warmup_steps, Integral)
                    or not 0 <= self.lr_warmup_steps < self.iterations):
                raise ValueError(
                    "lr_warmup_steps must be an integer in [0, iterations)"
                )
        if self.lr_scheduler == "plateau":
            if (not math.isfinite(self.lr_plateau_factor)
                    or not 0 < self.lr_plateau_factor < 1):
                raise ValueError("lr_plateau_factor must be finite and in (0, 1)")
            if (not math.isfinite(self.lr_plateau_threshold)
                    or self.lr_plateau_threshold < 0):
                raise ValueError("lr_plateau_threshold must be finite and nonnegative")
            for name in ("lr_plateau_patience", "lr_plateau_cooldown"):
                value = getattr(self, name)
                if (isinstance(value, bool) or not isinstance(value, Integral)
                        or value < 0):
                    raise ValueError(f"{name} must be a nonnegative integer")


@dataclass
class TrainingResult:
    model: ShellForceKNEEP2D
    mean: torch.Tensor
    std: torch.Tensor
    best_iteration: int
    best_validation_loss: float
    history_iterations: tuple[int, ...] = ()
    train_losses: tuple[float, ...] = ()
    validation_losses: tuple[float, ...] = ()
    learning_rates: tuple[float, ...] = ()
    best_learning_rate: float | None = None
    # Optimizer LR immediately after each recorded validation event. Plateau
    # reductions are visible here before their use by the following update.
    next_learning_rates: tuple[float, ...] = ()


def _learning_rate_at_iteration(config: TrainingConfig, iteration: int) -> float:
    """Learning rate used by the one-based optimizer update ``iteration``.

    Linear warmup uses ``learning_rate / lr_warmup_steps`` on the first
    update and reaches ``learning_rate`` on the last warmup update. Cosine
    decay then reaches ``lr_min`` on the final update. Without warmup, the
    first update uses ``learning_rate``; a one-update run uses ``lr_min``.
    ``lr_min`` is the decay floor, so warmup may start below it. For plateau,
    this helper supplies only the warmup ramp; after warmup the optimizer
    retains the LR selected by ReduceLROnPlateau.
    """
    if config.lr_scheduler == "none":
        return config.learning_rate
    warmup = config.lr_warmup_steps
    if iteration <= warmup:
        return config.learning_rate * iteration / warmup
    if config.lr_scheduler == "plateau":
        return config.learning_rate
    if warmup:
        phase = (iteration - warmup) / (config.iterations - warmup)
    else:
        phase = (iteration - 1) / max(config.iterations - 1, 1)
        if config.iterations == 1:
            phase = 1.0
    return config.lr_min + 0.5 * (config.learning_rate - config.lr_min) * (
        1.0 + math.cos(math.pi * phase)
    )


def _alpha_terms(ep: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha == 0.0:
        return -ep + torch.exp(-ep) - 1.0
    if alpha == -1.0:
        raise ValueError("alpha=-1 is singular")
    return -torch.expm1(alpha * ep) / alpha + torch.expm1(
        -(1.0 + alpha) * ep
    ) / (1.0 + alpha)


def _video_pairs(
    video: Video,
    ensemble: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    if isinstance(video, ShardedVideo):
        return video.fetch_pairs(ensemble, time)
    return torch.stack((video[ensemble, time], video[ensemble, time + 1]), dim=1)


def _normalized_pairs(
    video: Video,
    ensemble: torch.Tensor,
    time: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    pair = _video_pairs(video, ensemble, time).to(device=device, dtype=torch.float32)
    return (pair - mean) / std


def channel_normalization(video: Video) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute global per-channel sample moments with float64 CPU reductions."""
    return video_channel_normalization(video)


@torch.no_grad()
def _validation_loss(
    model: ShellForceKNEEP2D,
    video: Video,
    mean: torch.Tensor,
    std: torch.Tensor,
    alpha: float,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    transitions = video.shape[1] - 1
    total_pairs = video.shape[0] * transitions
    loss_sum = 0.0

    for start in range(0, total_pairs, batch_size):
        stop = min(start + batch_size, total_pairs)
        flat = torch.arange(start, stop)
        ensemble = flat // transitions
        time = flat % transitions
        pair = _normalized_pairs(video, ensemble, time, mean, std, device)
        ep = model(pair).sum(dim=1)
        loss_sum += _alpha_terms(ep, alpha).sum().item()

    return loss_sum / total_pairs


def train_model(
    train_video: Video,
    validation_video: Video,
    config: TrainingConfig,
    model_seed: int,
    device: torch.device | str,
    n_components: int = 2,
    hidden_channels: int = 8,
    hidden_layers: int = 2,
    max_distance: int = 4,
    progress: bool = True,
    ep_component_indices: tuple[int, ...] | None = None,
    activation: str = "elu",
    normalization: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> TrainingResult:
    """Fit on a dense movie or memory-mapped independent trajectory shards.

    ``train_batch_size`` is the number of uniformly sampled transitions in an
    optimizer step. ``train_micro_batch_size`` only limits activation memory:
    losses are summed and divided by the logical batch size before backward,
    gradients are clipped once, and AdamW takes exactly one step.  The model
    has no batch normalization or dropout, so this preserves the objective.
    Optional linear warmup and cosine decay advance once per optimizer step.
    Plateau scheduling instead observes each full validation after warmup;
    its absolute improvement threshold handles the negative alpha loss.
    Checkpoint learning rates record the rate actually used by that update;
    next_learning_rates record the optimizer rate after validation.
    """
    if config.train_batch_size <= 0 or config.validation_batch_size <= 0:
        raise ValueError("training and validation batch sizes must be positive")
    if config.iterations <= 0 or config.validate_every <= 0:
        raise ValueError("iterations and validate_every must be positive")
    micro_batch_size = config.train_micro_batch_size or config.train_batch_size
    if (config.train_micro_batch_size is not None
            and config.train_micro_batch_size <= 0):
        raise ValueError("train_micro_batch_size must be positive or None")
    for video in (train_video, validation_video):
        if video.ndim != 5 or video.shape[0] < 1 or video.shape[1] < 2:
            raise ValueError("expected video with shape [M>=1,time>=2,C,H,W]")
        if video.shape[2] != n_components:
            raise ValueError("video channel count does not match n_components")
    target = torch.device(device)
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if target.type == "cuda":
        torch.cuda.manual_seed_all(model_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    model = ShellForceKNEEP2D(
        n_components=n_components,
        hidden_channels=hidden_channels,
        hidden_layers=hidden_layers,
        max_distance=max_distance,
        ep_component_indices=ep_component_indices,
        activation=activation,
    ).to(target)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    plateau_scheduler = None
    if config.lr_scheduler == "plateau":
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", threshold_mode="abs",
            factor=config.lr_plateau_factor, patience=config.lr_plateau_patience,
            threshold=config.lr_plateau_threshold,
            cooldown=config.lr_plateau_cooldown, min_lr=config.lr_min,
        )

    if normalization is None:
        mean_cpu, std_cpu = channel_normalization(train_video)
    else:
        mean_cpu, std_cpu = normalization
        if mean_cpu.shape != std_cpu.shape or torch.any(std_cpu <= 0):
            raise ValueError("invalid precomputed normalization")
        
    mean = mean_cpu.to(device=target, dtype=torch.float32)
    std = std_cpu.to(device=target, dtype=torch.float32)

    sampler = torch.Generator(device="cpu")
    sampler.manual_seed(model_seed)
    best_loss = float("inf")
    best_iteration = 0
    best_learning_rate: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    history_iterations: list[int] = []
    train_losses: list[float] = []
    validation_losses: list[float] = []
    learning_rates: list[float] = []
    next_learning_rates: list[float] = []

    iterator = trange(
        1,
        config.iterations + 1,
        desc="KNEEP training",
        leave=False,
        disable=not progress,
    )
    for iteration in iterator:
        # Once warmup finishes, plateau owns the optimizer LR. Reassigning
        # the base rate here would silently undo every validation reduction.
        if plateau_scheduler is None or iteration <= config.lr_warmup_steps:
            scheduled_rate = _learning_rate_at_iteration(config, iteration)
            for param_group in optimizer.param_groups:
                param_group["lr"] = scheduled_rate
        learning_rate = float(optimizer.param_groups[0]["lr"])
        model.train()
        ensemble = torch.randint(
            train_video.shape[0],
            (config.train_batch_size,),
            generator=sampler,
        )
        time = torch.randint(
            train_video.shape[1] - 1,
            (config.train_batch_size,),
            generator=sampler,
        )
        # Fetch once before slicing: otherwise random microbatches repeatedly
        # reopen the same trajectory shards when the mapping cache is small.
        pair_cpu = _video_pairs(train_video, ensemble, time)
        optimizer.zero_grad(set_to_none=True)
        loss_value = torch.zeros((), device=target)
        for start in range(0, config.train_batch_size, micro_batch_size):
            pair = pair_cpu[start:start + micro_batch_size].to(
                device=target, dtype=torch.float32
            )
            pair = (pair - mean) / std
            ep = model(pair).sum(dim=1)
            loss = _alpha_terms(ep, config.alpha).sum() / config.train_batch_size
            loss.backward()
            loss_value += loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()

        if (iteration == 1 or iteration % config.validate_every == 0
                or iteration == config.iterations):
            validation_loss = _validation_loss(
                model,
                validation_video,
                mean,
                std,
                config.alpha,
                config.validation_batch_size,
                target,
            )
            history_iterations.append(iteration)
            train_losses.append(float(loss_value.cpu()))
            validation_losses.append(validation_loss)
            learning_rates.append(learning_rate)
            if (plateau_scheduler is not None
                    and iteration >= config.lr_warmup_steps):
                # One observation of the complete validation set, regardless
                # of the number of optimizer steps or gradient microbatches.
                plateau_scheduler.step(validation_loss)
            next_learning_rates.append(float(optimizer.param_groups[0]["lr"]))
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_iteration = iteration
                best_learning_rate = learning_rate
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }

    if best_state is None:
        raise RuntimeError("training finished without a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return TrainingResult(
        model=model,
        mean=mean,
        std=std,
        best_iteration=best_iteration,
        best_validation_loss=best_loss,
        history_iterations=tuple(history_iterations),
        train_losses=tuple(train_losses),
        validation_losses=tuple(validation_losses),
        learning_rates=tuple(learning_rates),
        best_learning_rate=best_learning_rate,
        next_learning_rates=tuple(next_learning_rates),
    )


@torch.no_grad()
def predict_epr_increments(
    result: TrainingResult,
    video: Video,
    batch_size: int,
    device: torch.device | str,
) -> np.ndarray:
    target = torch.device(device)
    transitions = video.shape[1] - 1
    total_pairs = video.shape[0] * transitions
    predictions: list[torch.Tensor] = []

    for start in range(0, total_pairs, batch_size):
        stop = min(start + batch_size, total_pairs)
        flat = torch.arange(start, stop)
        ensemble = flat // transitions
        time = flat % transitions
        pair = _normalized_pairs(
            video,
            ensemble,
            time,
            result.mean,
            result.std,
            target,
        )
        predictions.append(result.model(pair).sum(dim=1).cpu())

    return torch.cat(predictions).numpy()


@torch.no_grad()
def predict_epr_branch_increments(
    result: TrainingResult,
    video: Video,
    batch_size: int,
    device: torch.device | str,
) -> np.ndarray:
    """Return ordered branch scores with shape ``[transitions, branches]``."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if video.ndim != 5 or video.shape[1] < 2:
        raise ValueError("expected video with shape [M, time>=2, C, H, W]")

    target = torch.device(device)
    transitions = video.shape[1] - 1
    total_pairs = video.shape[0] * transitions
    predictions: list[torch.Tensor] = []
    for start in range(0, total_pairs, batch_size):
        stop = min(start + batch_size, total_pairs)
        flat = torch.arange(start, stop)
        ensemble = flat // transitions
        time = flat % transitions
        pair = _normalized_pairs(
            video, ensemble, time, result.mean, result.std, target
        )
        predictions.append(result.model(pair).cpu())
    return torch.cat(predictions).numpy()


@torch.no_grad()
def predict_epr_branch_maps(
    result: TrainingResult,
    video: Video,
    batch_size: int,
    device: torch.device | str,
) -> np.ndarray:
    """Return ordered raw branch maps for all consecutive video pairs.

    The output has shape ``[M * (time - 1), branches, height, width]`` in
    ensemble-major, time-minor order. These are the model's raw maps: their
    spatial mean equals the branch score. To obtain local increment maps whose
    spatial sum equals the predicted total increment, sum over branches and
    divide by ``height * width``.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if video.ndim != 5 or video.shape[1] < 2:
        raise ValueError("expected video with shape [M, time>=2, C, H, W]")

    target = torch.device(device)
    transitions = video.shape[1] - 1
    total_pairs = video.shape[0] * transitions
    predictions: list[torch.Tensor] = []

    for start in range(0, total_pairs, batch_size):
        stop = min(start + batch_size, total_pairs)
        flat = torch.arange(start, stop)
        ensemble = flat // transitions
        time = flat % transitions
        pair = _normalized_pairs(
            video,
            ensemble,
            time,
            result.mean,
            result.std,
            target,
        )
        predictions.append(result.model(pair, return_maps=True).cpu())

    return torch.cat(predictions).numpy()


@torch.no_grad()
def predict_epr_component_maps(
    result: TrainingResult,
    video: Video,
    batch_size: int,
    device: torch.device | str,
) -> np.ndarray:
    """Return ordered raw EP maps resolved by shell and state component.

    The output shape is ``[M * (time - 1), branches, components, height,
    width]`` in ensemble-major, time-minor order.  The component axis follows
    ``result.model.ep_component_indices``.  Summing it recovers
    :func:`predict_epr_branch_maps`; spatially averaging and summing both the
    shell and component axes recovers :func:`predict_epr_increments`.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if video.ndim != 5 or video.shape[1] < 2:
        raise ValueError("expected video with shape [M, time>=2, C, H, W]")

    target = torch.device(device)
    transitions = video.shape[1] - 1
    total_pairs = video.shape[0] * transitions
    predictions: list[torch.Tensor] = []

    for start in range(0, total_pairs, batch_size):
        stop = min(start + batch_size, total_pairs)
        flat = torch.arange(start, stop)
        ensemble = flat // transitions
        time = flat % transitions
        pair = _normalized_pairs(
            video,
            ensemble,
            time,
            result.mean,
            result.std,
            target,
        )
        predictions.append(result.model.component_ep_maps(pair).cpu())

    return torch.cat(predictions).numpy()
