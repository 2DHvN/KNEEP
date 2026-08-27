"""Training and ordered evaluation for the SAOU KNEEP experiment."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import trange

from shell_force import ShellForceKNEEP2D


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


@dataclass
class TrainingResult:
    model: ShellForceKNEEP2D
    mean: torch.Tensor
    std: torch.Tensor
    best_iteration: int
    best_validation_loss: float


def _alpha_terms(ep: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha == 0.0:
        return -ep + torch.exp(-ep) - 1.0
    if alpha == -1.0:
        raise ValueError("alpha=-1 is singular")
    return -torch.expm1(alpha * ep) / alpha + torch.expm1(
        -(1.0 + alpha) * ep
    ) / (1.0 + alpha)


def _normalized_pairs(
    video: torch.Tensor,
    ensemble: torch.Tensor,
    time: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    pair = torch.stack(
        (video[ensemble, time], video[ensemble, time + 1]), dim=1
    ).to(device=device, dtype=torch.float32)
    return (pair - mean) / std


@torch.no_grad()
def _validation_loss(
    model: ShellForceKNEEP2D,
    video: torch.Tensor,
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
    train_video: torch.Tensor,
    validation_video: torch.Tensor,
    config: TrainingConfig,
    model_seed: int,
    device: torch.device | str,
    n_components: int = 2,
    hidden_channels: int = 8,
    hidden_layers: int = 2,
    max_distance: int = 4,
    progress: bool = True,
) -> TrainingResult:
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
    ).to(target)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    mean_cpu = train_video.mean(dim=(0, 1, 3, 4), keepdim=True)
    std_cpu = train_video.std(dim=(0, 1, 3, 4), keepdim=True, unbiased=True)
    if torch.any(std_cpu <= 0):
        raise RuntimeError("training data contain a zero-variance channel")
    mean = mean_cpu.to(target)
    std = std_cpu.to(target)

    sampler = torch.Generator(device="cpu")
    sampler.manual_seed(model_seed)
    best_loss = float("inf")
    best_iteration = 0
    best_state: dict[str, torch.Tensor] | None = None

    iterator = trange(
        1,
        config.iterations + 1,
        desc="KNEEP training",
        leave=False,
        disable=not progress,
    )
    for iteration in iterator:
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
        pair = _normalized_pairs(train_video, ensemble, time, mean, std, target)
        ep = model(pair).sum(dim=1)
        loss = _alpha_terms(ep, config.alpha).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()

        if iteration == 1 or iteration % config.validate_every == 0:
            validation_loss = _validation_loss(
                model,
                validation_video,
                mean,
                std,
                config.alpha,
                config.validation_batch_size,
                target,
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_iteration = iteration
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
    )


@torch.no_grad()
def predict_epr_increments(
    result: TrainingResult,
    video: torch.Tensor,
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
