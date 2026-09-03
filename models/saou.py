"""Periodic shell-resolved antisymmetric Ornstein--Uhlenbeck field model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from tqdm import trange


Offset = tuple[int, int]


@dataclass(frozen=True)
class SAOUConfig:
    lattice_size: int = 32
    radii: tuple[int, ...] = (1, 2, 3, 4)
    amplitudes: tuple[float, ...] = (0.0, 1.0, 1.0, 0.0)
    gamma: float = 1.0
    omega0: float = 3.0
    temperature: float = 0.1
    dt: float = 1e-2
    weight_normalization: str = "mean"

    def __post_init__(self) -> None:
        if self.lattice_size <= 0:
            raise ValueError("lattice_size must be positive")
        if len(self.radii) != len(self.amplitudes) or not self.radii:
            raise ValueError("radii and amplitudes must have equal nonzero length")
        scalars = (*self.amplitudes, self.gamma, self.omega0, self.temperature, self.dt)
        if not all(np.isfinite(value) for value in scalars):
            raise ValueError("SAOU parameters must be finite")
        if self.gamma <= 0 or self.temperature <= 0 or self.dt <= 0:
            raise ValueError("gamma, temperature, and dt must be positive")
        if max(self.radii) >= self.lattice_size / 2:
            raise ValueError("shell radii must be smaller than half the lattice size")

    @property
    def effective_dt(self) -> float:
        return self.dt


@dataclass(frozen=True)
class Shell:
    radius: int
    offsets: tuple[Offset, ...]
    weights: np.ndarray
    amplitude: float


@dataclass
class ShellOperator:
    shell: Shell
    kernel_hat: np.ndarray
    kernel_tensor: torch.Tensor | None = None


def _make_shells(config: SAOUConfig) -> list[Shell]:
    if config.weight_normalization != "mean":
        raise ValueError("the research SAOU configuration requires mean shell weights")

    shells: list[Shell] = []
    lower = 0
    for radius, amplitude in zip(config.radii, config.amplitudes):
        if radius <= lower:
            raise ValueError("radii must be strictly increasing")
        offsets = tuple(
            (dy, dx)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if lower < max(abs(dy), abs(dx)) <= radius
        )
        weights = np.ones(len(offsets), dtype=np.float64)
        weights /= weights.sum()
        shells.append(
            Shell(
                radius=radius,
                offsets=offsets,
                weights=weights,
                amplitude=float(amplitude),
            )
        )
        lower = radius
    return shells


def _build_operators(config: SAOUConfig) -> list[ShellOperator]:
    size = config.lattice_size
    operators: list[ShellOperator] = []
    for shell in _make_shells(config):
        kernel = np.zeros((size, size), dtype=np.float64)
        for (dy, dx), weight in zip(shell.offsets, shell.weights):
            kernel[(-dy) % size, (-dx) % size] += weight
        operators.append(
            ShellOperator(shell=shell, kernel_hat=np.fft.rfftn(kernel, axes=(0, 1)))
        )
    return operators


def _rotate90(field: torch.Tensor) -> torch.Tensor:
    return torch.stack((-field[..., 1], field[..., 0]), dim=-1)


def _shell_convolution(field: torch.Tensor, operator: ShellOperator) -> torch.Tensor:
    complex_dtype = (
        torch.complex128 if field.dtype == torch.float64 else torch.complex64
    )
    kernel = operator.kernel_tensor
    if kernel is None or kernel.device != field.device or kernel.dtype != complex_dtype:
        kernel = torch.from_numpy(operator.kernel_hat).to(
            device=field.device, dtype=complex_dtype
        )
        operator.kernel_tensor = kernel
    return torch.fft.irfftn(
        torch.fft.rfftn(field, dim=(-2, -1)) * kernel,
        s=field.shape[-2:],
        dim=(-2, -1),
    ).real


def irreversible_velocity(
    field: torch.Tensor,
    operators: Sequence[ShellOperator],
    omega0: float,
) -> torch.Tensor:
    """Return the irreversible velocity for ``[..., y, x, component]``."""
    velocity = float(omega0) * _rotate90(field)
    for operator in operators:
        shell = operator.shell
        if shell.amplitude == 0.0:
            continue
        convolved = torch.empty_like(field)
        convolved[..., 0] = _shell_convolution(field[..., 0], operator)
        convolved[..., 1] = _shell_convolution(field[..., 1], operator)
        relative = convolved - float(shell.weights.sum()) * field
        velocity = velocity + shell.amplitude * _rotate90(relative)
    return velocity


def _drift(
    field: torch.Tensor,
    operators: Sequence[ShellOperator],
    config: SAOUConfig,
) -> torch.Tensor:
    return -config.gamma * field + irreversible_velocity(
        field, operators, config.omega0
    )


@torch.no_grad()
def simulate_trajectories(
    config: SAOUConfig,
    n_trajectories: int,
    n_samples: int,
    burn_steps: int,
    seed: int,
    simulation_device: torch.device | str,
    storage_dtype: torch.dtype,
    progress: bool = False,
) -> torch.Tensor:
    """Simulate trajectories and always return ``[M, time, 2, L, L]`` on CPU."""
    if n_trajectories <= 0 or n_samples < 2 or burn_steps < 0:
        raise ValueError("invalid trajectory, sample, or burn-in count")
    if not storage_dtype.is_floating_point:
        raise ValueError("storage_dtype must be floating point")

    device = torch.device(simulation_device)
    size = config.lattice_size
    operators = _build_operators(config)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    field = torch.normal(
        mean=0.0,
        std=np.sqrt(config.temperature / config.gamma),
        size=(n_trajectories, size, size, 2),
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    noise_scale = np.sqrt(2.0 * config.temperature * config.dt)

    for _ in trange(
        burn_steps, desc="SAOU burn-in", leave=False, disable=not progress
    ):
        noise = torch.randn(
            field.shape, generator=generator, device=device, dtype=field.dtype
        )
        field = field + _drift(field, operators, config) * config.dt
        field = field + noise_scale * noise

    trajectories = torch.empty(
        (n_trajectories, n_samples, 2, size, size),
        dtype=storage_dtype,
        device="cpu",
    )
    saved = 0
    for _ in trange(
        n_samples, desc="SAOU sampling", leave=False, disable=not progress
    ):
        noise = torch.randn(
            field.shape, generator=generator, device=device, dtype=field.dtype
        )
        field = field + _drift(field, operators, config) * config.dt
        field = field + noise_scale * noise
        trajectories[:, saved].copy_(
            field.permute(0, 3, 1, 2).to(device="cpu", dtype=storage_dtype)
        )
        saved += 1

    return trajectories


@torch.no_grad()
def exact_epr_increments(
    trajectories: torch.Tensor,
    config: SAOUConfig,
    batch_size: int,
    device: torch.device | str,
) -> np.ndarray:
    """Compute exact midpoint EP for every consecutive stored transition."""
    if trajectories.ndim != 5 or trajectories.shape[2] != 2:
        raise ValueError("expected trajectories with shape [M, time, 2, L, L]")
    if tuple(trajectories.shape[-2:]) != (
        config.lattice_size,
        config.lattice_size,
    ):
        raise ValueError("trajectory lattice size does not match SAOUConfig")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    target = torch.device(device)
    n_trajectories, n_samples = trajectories.shape[:2]
    transitions_per_trajectory = n_samples - 1
    total = n_trajectories * transitions_per_trajectory
    output = torch.empty(total, dtype=torch.float64)
    operators = _build_operators(config)

    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        flat = torch.arange(start, stop)
        ensemble = flat // transitions_per_trajectory
        time = flat % transitions_per_trajectory
        first = trajectories[ensemble, time].to(target, dtype=torch.float64)
        second = trajectories[ensemble, time + 1].to(target, dtype=torch.float64)
        first = first.permute(0, 2, 3, 1)
        second = second.permute(0, 2, 3, 1)
        midpoint = 0.5 * (first + second)
        increment = second - first
        velocity = irreversible_velocity(midpoint, operators, config.omega0)
        ep = (velocity * increment).sum(dim=(-3, -2, -1)) / config.temperature
        output[start:stop] = ep.cpu()

    return output.numpy()


def _relative_kernel(shell: Shell) -> dict[Offset, float]:
    kernel: dict[Offset, float] = {}
    for offset, weight in zip(shell.offsets, shell.weights):
        kernel[offset] = kernel.get(offset, 0.0) + float(weight)
    kernel[(0, 0)] = kernel.get((0, 0), 0.0) - float(shell.weights.sum())
    return {offset: value for offset, value in kernel.items() if abs(value) > 1e-15}


def _absolute_kernel(shell: Shell) -> dict[Offset, float]:
    return {
        offset: float(weight)
        for offset, weight in zip(shell.offsets, shell.weights)
        if abs(float(weight)) > 1e-15
    }


def _kernel_inner(
    first: dict[Offset, float], second: dict[Offset, float]
) -> float:
    if len(first) > len(second):
        first, second = second, first
    return float(sum(value * second.get(offset, 0.0) for offset, value in first.items()))


def theoretical_epr_gram(config: SAOUConfig) -> np.ndarray:
    """Return the exact component Gram matrix in the relative-shell basis."""
    shells = _make_shells(config)
    coefficients = [config.omega0, *(shell.amplitude for shell in shells)]
    kernels = [{(0, 0): 1.0}, *(_relative_kernel(shell) for shell in shells)]
    gram = np.empty((len(coefficients), len(coefficients)), dtype=np.float64)
    prefactor = 2.0 * config.lattice_size**2 / config.gamma
    for row in range(len(coefficients)):
        for column in range(len(coefficients)):
            gram[row, column] = (
                prefactor
                * coefficients[row]
                * coefficients[column]
                * _kernel_inner(kernels[row], kernels[column])
            )
    return gram


def theoretical_epr_gram_absolute(config: SAOUConfig) -> np.ndarray:
    """Return the exact Gram matrix in the estimator's absolute-shell basis.

    The physical velocity is rewritten as a local term plus disjoint absolute
    shell convolutions.  This is the basis produced by
    :class:`shell_force.ShellForceKNEEP2D`, so its row sums are the analytic
    targets for the learned branch spectrum.  The stationary ensemble average
    is independent of both trajectories and temperature.
    """
    shells = _make_shells(config)
    center_coefficient = config.omega0 - sum(
        shell.amplitude * float(shell.weights.sum()) for shell in shells
    )
    coefficients = [center_coefficient, *(shell.amplitude for shell in shells)]
    kernels = [{(0, 0): 1.0}, *(_absolute_kernel(shell) for shell in shells)]
    gram = np.empty((len(coefficients), len(coefficients)), dtype=np.float64)
    prefactor = 2.0 * config.lattice_size**2 / config.gamma
    for row in range(len(coefficients)):
        for column in range(len(coefficients)):
            gram[row, column] = (
                prefactor
                * coefficients[row]
                * coefficients[column]
                * _kernel_inner(kernels[row], kernels[column])
            )
    return gram


def theoretical_epr_components_absolute(config: SAOUConfig) -> np.ndarray:
    """Return analytic rates for ``[local, shell 1, ..., shell R]``."""
    return theoretical_epr_gram_absolute(config).sum(axis=1)


def theoretical_epr_rate(config: SAOUConfig) -> float:
    return float(theoretical_epr_gram(config).sum())
