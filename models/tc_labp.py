"""Thermodynamically consistent hard-core lattice active Brownian particles.

This module independently implements the state-dependent ``C_v`` transition
law of Kim, Kwon, and Baek (arXiv:2503.16958, Eq. (5)).  It deliberately has
one physical model and one Torch implementation: particles occupy at most one
site, occupied destinations are forbidden, and there is no pair potential or
alternative hopping prefactor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import torch
from tqdm import trange


# Site coordinates and fields both use (x, y) axis order.  Keeping this tensor
# on the CPU avoids a persistent device-specific global; it is tiny and is
# copied once for each simulation.
_CARDINAL_DIRECTIONS = torch.tensor(
    ((1, 0), (-1, 0), (0, 1), (0, -1)), dtype=torch.long
)


def _finite_scalar(name: str, value: float) -> float:
    """Return ``value`` as a finite float or raise a configuration error."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite real number")
    return converted


def _x_coth_x(value: float) -> float:
    """Evaluate ``x*coth(x)`` with its removable singularity at zero."""
    magnitude = abs(value)
    if magnitude < 1.0e-5:
        squared = magnitude * magnitude
        return 1.0 + squared / 3.0 - squared * squared / 45.0
    return magnitude / math.tanh(magnitude)


@dataclass(frozen=True)
class TCLABPConfig:
    """Physical and integration parameters for the hard-core TC-LABP model.

    ``density`` is the requested occupied-site fraction and is converted to a
    particle count by nearest-integer rounding.  One Monte Carlo step consists
    of five independently and uniformly sampled colour substeps followed by a
    single angular Brownian update, so its physical duration is ``dt``.
    """

    lattice_size: int = 40
    density: float = 0.30
    speed: float = 10.0
    rotational_diffusion: float = 1.0
    translational_diffusion: float = 1.0
    dt: float = 1.0e-3
    lattice_spacing: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.lattice_size, bool) or not isinstance(
            self.lattice_size, Integral
        ):
            raise ValueError("lattice_size must be an integer")
        if self.lattice_size <= 0:
            raise ValueError("lattice_size must be positive")
        if self.lattice_size % 5 != 0:
            raise ValueError(
                "lattice_size must be a multiple of 5 for periodic five-colour updates"
            )

        density = _finite_scalar("density", self.density)
        speed = _finite_scalar("speed", self.speed)
        rotational = _finite_scalar(
            "rotational_diffusion", self.rotational_diffusion
        )
        translational = _finite_scalar(
            "translational_diffusion", self.translational_diffusion
        )
        dt = _finite_scalar("dt", self.dt)
        spacing = _finite_scalar("lattice_spacing", self.lattice_spacing)

        # Normalize accepted scalar-like inputs once.  The frozen dataclass
        # then exposes a stable numeric contract to every downstream formula.
        object.__setattr__(self, "lattice_size", int(self.lattice_size))
        object.__setattr__(self, "density", density)
        object.__setattr__(self, "speed", speed)
        object.__setattr__(self, "rotational_diffusion", rotational)
        object.__setattr__(self, "translational_diffusion", translational)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "lattice_spacing", spacing)

        if not 0.0 < density <= 1.0:
            raise ValueError("density must lie in (0, 1]")
        if speed < 0.0:
            raise ValueError("speed must be nonnegative")
        if rotational < 0.0:
            raise ValueError("rotational_diffusion must be nonnegative")
        if translational <= 0.0:
            raise ValueError("translational_diffusion must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if spacing <= 0.0:
            raise ValueError("lattice_spacing must be positive")
        if self.n_particles <= 0:
            raise ValueError(
                "density is too small to place one particle on this lattice"
            )

        maximum = self.max_hop_probability
        if not math.isfinite(maximum):
            raise ValueError("derived hopping probabilities must be finite")
        if maximum > 1.0:
            raise ValueError(
                "the maximum total hop probability exceeds 1 "
                f"({maximum:.8g}); reduce dt or speed, increase lattice_spacing, "
                "or increase translational_diffusion"
            )

    @property
    def n_particles(self) -> int:
        """Number of particles obtained from the requested site density."""
        requested = float(self.density) * int(self.lattice_size) ** 2
        return int(math.floor(requested + 0.5))

    @property
    def effective_dt(self) -> float:
        """Physical duration of one complete five-substep Monte Carlo step."""
        return float(self.dt)

    @property
    def Pe(self) -> float:
        """Grid Péclet number ``speed * lattice_spacing / D_t``."""
        return (
            float(self.speed)
            * float(self.lattice_spacing)
            / float(self.translational_diffusion)
        )

    @property
    def max_hop_probability(self) -> float:
        """Maximum, over orientation, of the four-direction probability sum.

        For the ``C_v`` law the maximum occurs when the propulsion direction
        lies halfway between lattice axes.  This exact bound is used to reject
        invalid fixed-time-step configurations before simulation starts.
        """
        base = (
            float(self.dt)
            * float(self.translational_diffusion)
            / float(self.lattice_spacing) ** 2
        )
        diagonal_half_affinity = abs(self.Pe) / (2.0 * math.sqrt(2.0))
        return 4.0 * base * _x_coth_x(diagonal_half_affinity)

    @property
    def max_hop_prob(self) -> float:
        """Short alias for :attr:`max_hop_probability`."""
        return self.max_hop_probability


@dataclass(frozen=True)
class TCLABPTrajectory:
    """Saved TC-LABP trajectories and exact hop entropy production.

    Every tensor is on the CPU.  With ``M`` trajectories, ``S`` recorded
    transitions, ``T=S+1`` saved frames, ``N`` particles, and lattice side
    ``L``, shapes are:

    - ``sites``: ``[M, T, N, 2]`` integer ``(x, y)`` sites.
    - ``angles``: ``[M, T, N]`` orientations in ``[0, 2*pi)``.
    - ``occupancy``: ``[M, T, L, L]`` Boolean hard-core fields.
    - ``exact_ep`` and ``accepted_hops``: ``[M, S]`` transition totals.
    - ``exact_ep_maps``: ``[M, S, L, L]``, in destination-site gauge.
    - ``times``: ``[T]`` physical times relative to the post-burn frame.
    """

    sites: torch.Tensor
    angles: torch.Tensor
    occupancy: torch.Tensor
    exact_ep: torch.Tensor
    exact_ep_maps: torch.Tensor
    accepted_hops: torch.Tensor
    times: torch.Tensor


# A descriptive alias makes downstream code resilient while keeping one data
# container implementation.
TCLABPResult = TCLABPTrajectory


def _cv_bernoulli_factor(affinity: torch.Tensor) -> torch.Tensor:
    """Return ``a / (1-exp(-a))`` using stable small/large-``a`` branches."""
    absolute = affinity.abs()
    small = absolute < (1.0e-4 if affinity.dtype == torch.float32 else 1.0e-7)
    squared = affinity * affinity
    series = 1.0 + affinity / 2.0 + squared / 12.0 - squared * squared / 720.0

    # torch.where evaluates both operands.  Replacing small arguments and
    # clamping the middle branch prevent a dormant 0/0 or exponential overflow.
    middle_argument = torch.where(
        small, torch.ones_like(affinity), affinity.clamp(min=-50.0, max=50.0)
    )
    middle = middle_argument / (-torch.expm1(-middle_argument))
    positive_large = affinity.clamp_min(0.0)
    negative_large = (-affinity).clamp_min(0.0) * torch.exp(
        affinity.clamp(max=0.0)
    )
    nonsmall = torch.where(
        affinity > 50.0,
        positive_large,
        torch.where(affinity < -50.0, negative_large, middle),
    )
    return torch.where(small, series, nonsmall)


def _free_hop_probabilities(
    config: TCLABPConfig, angles: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return free-site probabilities and EP affinities with shape ``[..., 4]``.

    Direction order is ``(+x, -x, +y, -y)``.  The second result is
    ``A/D_t`` and therefore equals the true medium EP if that hop is accepted.
    """
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    affinities = config.Pe * torch.stack((cosine, -cosine, sine, -sine), dim=-1)
    base = (
        config.dt
        * config.translational_diffusion
        / config.lattice_spacing**2
    )
    probabilities = float(base) * _cv_bernoulli_factor(affinities)
    return probabilities, affinities


def _occupancy_from_sites(sites: torch.Tensor, lattice_size: int) -> torch.Tensor:
    """Build a Boolean ``[M, L, L]`` occupancy tensor from ``[M, N, 2]`` sites."""
    n_trajectories, n_particles = sites.shape[:2]
    linear = sites[..., 0] * lattice_size + sites[..., 1]
    counts = torch.zeros(
        (n_trajectories, lattice_size * lattice_size),
        dtype=torch.long,
        device=sites.device,
    )
    counts.scatter_add_(1, linear, torch.ones_like(linear))
    if counts.numel() and bool((counts > 1).any().detach().cpu()):
        raise RuntimeError("hard-core state contains multiply occupied sites")
    return counts.view(n_trajectories, lattice_size, lattice_size).bool()


def _synchronous_colour_substep(
    config: TCLABPConfig,
    sites: torch.Tensor,
    occupancy: torch.Tensor,
    free_probabilities: torch.Tensor,
    entropy_increments: torch.Tensor,
    chosen_colours: torch.Tensor,
    uniform_draws: torch.Tensor,
    *,
    collect_ep: bool,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Mutate one synchronous colour substep and return ``([M,L,L], [M])``.

    ``sites`` is ``[M,N,2]``; probabilities, entropy increments, and candidate
    direction axes are ``[M,N,4]``; ``chosen_colours`` is ``[M]``; and draws
    are ``[M,N]``.  EP maps use the accepted hop's destination-site gauge.
    """
    device = sites.device
    n_trajectories, n_particles = sites.shape[:2]
    lattice_size = config.lattice_size
    directions = _CARDINAL_DIRECTIONS.to(device=device)
    destinations = (
        sites.unsqueeze(2) + directions.view(1, 1, 4, 2)
    ).remainder(lattice_size)

    batch = torch.arange(device=device, end=n_trajectories).view(-1, 1, 1)
    blocked = occupancy[
        batch,
        destinations[..., 0],
        destinations[..., 1],
    ]
    probabilities = torch.where(
        blocked, torch.zeros_like(free_probabilities), free_probabilities
    )

    particle_colours = (sites[..., 0] + 3 * sites[..., 1]).remainder(5)
    selected = particle_colours == chosen_colours.view(-1, 1)
    cumulative = probabilities.cumsum(dim=-1)
    # Half-open inverse-CDF bins are essential when a leading direction is
    # blocked and torch.rand happens to return exactly zero.
    direction_index = (uniform_draws.unsqueeze(-1) >= cumulative).sum(dim=-1)
    moved = selected & (direction_index < 4)
    safe_direction = direction_index.clamp_max(3)

    gather_sites = safe_direction.view(n_trajectories, n_particles, 1, 1).expand(
        -1, -1, 1, 2
    )
    proposed_sites = destinations.gather(2, gather_sites).squeeze(2)
    gather_scalar = safe_direction.unsqueeze(-1)
    chosen_entropy = entropy_increments.gather(2, gather_scalar).squeeze(-1)
    ep_values = torch.where(moved, chosen_entropy, torch.zeros_like(chosen_entropy))

    ep_map: torch.Tensor | None = None
    if collect_ep:
        ep_map = torch.zeros(
            (n_trajectories, lattice_size * lattice_size),
            dtype=entropy_increments.dtype,
            device=device,
        )
        destination_linear = (
            proposed_sites[..., 0] * lattice_size + proposed_sites[..., 1]
        )
        ep_map.scatter_add_(1, destination_linear, ep_values)
        ep_map = ep_map.view(n_trajectories, lattice_size, lattice_size)

    moved_ensemble, moved_particle = torch.where(moved)
    old_sites = sites[moved_ensemble, moved_particle]
    new_sites = proposed_sites[moved_ensemble, moved_particle]
    occupancy[moved_ensemble, old_sites[:, 0], old_sites[:, 1]] = False
    occupancy[moved_ensemble, new_sites[:, 0], new_sites[:, 1]] = True
    sites[moved_ensemble, moved_particle] = new_sites

    return ep_map, moved.sum(dim=1, dtype=torch.long)


def _advance_step(
    config: TCLABPConfig,
    sites: torch.Tensor,
    angles: torch.Tensor,
    occupancy: torch.Tensor,
    generator: torch.Generator,
    *,
    collect_ep: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Advance one full MC step and return angles, EP map, and hop count."""
    n_trajectories, n_particles = sites.shape[:2]
    probabilities, entropy = _free_hop_probabilities(config, angles)
    sampled_colours = torch.randint(
        0,
        5,
        (5, n_trajectories),
        generator=generator,
        device=sites.device,
    )
    total_map = (
        torch.zeros(
            (
                n_trajectories,
                config.lattice_size,
                config.lattice_size,
            ),
            dtype=angles.dtype,
            device=sites.device,
        )
        if collect_ep
        else None
    )
    total_hops = torch.zeros(
        n_trajectories, dtype=torch.long, device=sites.device
    )

    for substep in range(5):
        draws = torch.rand(
            (n_trajectories, n_particles),
            generator=generator,
            dtype=angles.dtype,
            device=sites.device,
        )
        ep_map, hops = _synchronous_colour_substep(
            config,
            sites,
            occupancy,
            probabilities,
            entropy,
            sampled_colours[substep],
            draws,
            collect_ep=collect_ep,
        )
        total_hops += hops
        if total_map is not None and ep_map is not None:
            total_map += ep_map

    if config.rotational_diffusion > 0.0:
        angular_noise = torch.randn(
            angles.shape,
            generator=generator,
            dtype=angles.dtype,
            device=angles.device,
        )
        angles = torch.remainder(
            angles
            + math.sqrt(2.0 * config.rotational_diffusion * config.dt)
            * angular_noise,
            2.0 * math.pi,
        )
    return angles, total_map, total_hops


def _validate_run_arguments(
    n_trajectories: int,
    n_steps: int,
    burn_steps: int,
    sampling_steps: int,
    seed: int,
    storage_dtype: torch.dtype,
) -> None:
    """Validate simulator controls independently of the physical config."""
    integer_arguments = {
        "n_trajectories": n_trajectories,
        "n_steps": n_steps,
        "burn_steps": burn_steps,
        "sampling_steps": sampling_steps,
        "seed": seed,
    }
    for name, value in integer_arguments.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
    if n_trajectories <= 0:
        raise ValueError("n_trajectories must be positive")
    if n_steps < 0:
        raise ValueError("n_steps must be nonnegative")
    if burn_steps < 0:
        raise ValueError("burn_steps must be nonnegative")
    if sampling_steps <= 0:
        raise ValueError("sampling_steps must be positive")
    if storage_dtype not in (torch.float32, torch.float64):
        raise ValueError("storage_dtype must be torch.float32 or torch.float64")


@torch.no_grad()
def simulate_trajectories(
    config: TCLABPConfig,
    n_trajectories: int,
    n_steps: int,
    burn_steps: int,
    sampling_steps: int,
    seed: int,
    simulation_device: torch.device | str = "cpu",
    storage_dtype: torch.dtype = torch.float32,
    progress: bool = False,
) -> TCLABPTrajectory:
    """Simulate and return a post-burn trajectory entirely on the CPU.

    Frame zero is the state immediately after ``burn_steps``.  The run records
    exactly ``n_steps`` transitions and therefore returns ``n_steps + 1``
    frames.  Each recorded transition contains ``sampling_steps`` full MC
    steps.  With ``S=n_steps`` and ``T=S+1``, output shapes are
    ``sites [M,T,N,2]``, ``angles [M,T,N]``, ``occupancy [M,T,L,L]``, interval
    arrays ``[M,S]``, EP maps ``[M,S,L,L]``, and times ``[T]``.  All accepted
    hops in a recorded transition contribute; sampling never drops EP.
    """
    if not isinstance(config, TCLABPConfig):
        raise TypeError("config must be a TCLABPConfig")
    _validate_run_arguments(
        n_trajectories,
        n_steps,
        burn_steps,
        sampling_steps,
        seed,
        storage_dtype,
    )
    device = torch.device(simulation_device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("simulation_device must be a CPU or CUDA device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA simulation requested, but CUDA is unavailable")

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    lattice_size = config.lattice_size
    n_particles = config.n_particles
    n_frames = n_steps + 1

    sites = torch.empty(
        (n_trajectories, n_particles, 2), dtype=torch.long, device=device
    )
    for ensemble in range(n_trajectories):
        chosen = torch.randperm(
            lattice_size * lattice_size,
            generator=generator,
            device=device,
        )[:n_particles]
        sites[ensemble, :, 0] = torch.div(
            chosen, lattice_size, rounding_mode="floor"
        )
        sites[ensemble, :, 1] = chosen.remainder(lattice_size)
    angles = 2.0 * math.pi * torch.rand(
        (n_trajectories, n_particles),
        generator=generator,
        dtype=storage_dtype,
        device=device,
    )
    occupancy = _occupancy_from_sites(sites, lattice_size)

    for _ in trange(
        burn_steps,
        desc="TC-LABP burn-in",
        leave=False,
        disable=not progress,
    ):
        angles, _, _ = _advance_step(
            config,
            sites,
            angles,
            occupancy,
            generator,
            collect_ep=False,
        )

    saved_sites = torch.empty(
        (n_trajectories, n_frames, n_particles, 2), dtype=torch.long
    )
    saved_angles = torch.empty(
        (n_trajectories, n_frames, n_particles), dtype=storage_dtype
    )
    saved_occupancy = torch.empty(
        (n_trajectories, n_frames, lattice_size, lattice_size), dtype=torch.bool
    )
    saved_ep_maps = torch.empty(
        (
            n_trajectories,
            n_steps,
            lattice_size,
            lattice_size,
        ),
        dtype=storage_dtype,
    )
    saved_hops = torch.empty(
        (n_trajectories, n_steps), dtype=torch.long
    )

    def save_frame(index: int) -> None:
        saved_sites[:, index].copy_(sites.detach().cpu())
        saved_angles[:, index].copy_(angles.detach().cpu())
        saved_occupancy[:, index].copy_(occupancy.detach().cpu())

    save_frame(0)
    for interval in trange(
        n_steps,
        desc="TC-LABP sampling",
        leave=False,
        disable=not progress,
    ):
        interval_map = torch.zeros(
            (n_trajectories, lattice_size, lattice_size),
            dtype=storage_dtype,
            device=device,
        )
        interval_hops = torch.zeros(
            n_trajectories, dtype=torch.long, device=device
        )
        for _ in range(sampling_steps):
            angles, step_map, step_hops = _advance_step(
                config,
                sites,
                angles,
                occupancy,
                generator,
                collect_ep=True,
            )
            if step_map is None:  # pragma: no cover - guarded by collect_ep
                raise RuntimeError("internal EP collection error")
            interval_map += step_map
            interval_hops += step_hops
        saved_ep_maps[:, interval].copy_(interval_map.detach().cpu())
        saved_hops[:, interval].copy_(interval_hops.detach().cpu())
        save_frame(interval + 1)

    # Deriving totals from the returned CPU maps fixes one unambiguous local
    # gauge and makes the documented map-sum invariant exact.
    exact_ep = saved_ep_maps.sum(dim=(-2, -1))
    times = (
        torch.arange(n_frames, dtype=storage_dtype)
        * float(sampling_steps)
        * config.effective_dt
    )
    return TCLABPTrajectory(
        sites=saved_sites,
        angles=saved_angles,
        occupancy=saved_occupancy,
        exact_ep=exact_ep,
        exact_ep_maps=saved_ep_maps,
        accepted_hops=saved_hops,
        times=times,
    )


def encode_observations(
    result: TCLABPTrajectory,
    include_angle: bool,
) -> torch.Tensor:
    """Encode a trajectory as a CPU video for the ShellForce network.

    ``include_angle=False`` returns ``[M,T,1,L,L]`` with the occupancy channel.
    ``include_angle=True`` returns ``[M,T,3,L,L]`` with channels
    ``(rho, rho*cos(theta), rho*sin(theta))``.  Empty sites are zero and the
    output floating dtype matches ``result.angles``.
    """
    if not isinstance(result, TCLABPTrajectory):
        raise TypeError("result must be a TCLABPTrajectory")
    if not isinstance(include_angle, bool):
        raise TypeError("include_angle must be a bool")
    tensors = (
        result.sites,
        result.angles,
        result.occupancy,
        result.exact_ep,
        result.exact_ep_maps,
        result.accepted_hops,
        result.times,
    )
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("TCLABPTrajectory tensors must be on the CPU")
    if result.sites.ndim != 4 or result.sites.shape[-1] != 2:
        raise ValueError("sites must have shape [M,T,N,2]")
    if result.angles.shape != result.sites.shape[:-1]:
        raise ValueError("angles must have shape [M,T,N] matching sites")
    if result.occupancy.ndim != 4:
        raise ValueError("occupancy must have shape [M,T,L,L]")
    if result.occupancy.shape[:2] != result.sites.shape[:2]:
        raise ValueError("occupancy ensemble/time axes must match sites")
    if result.occupancy.shape[-2] != result.occupancy.shape[-1]:
        raise ValueError("occupancy lattice must be square")
    if not result.angles.dtype.is_floating_point:
        raise ValueError("angles must have a floating dtype")

    density = result.occupancy.to(dtype=result.angles.dtype).unsqueeze(2)
    if not include_angle:
        return density

    n_trajectories, n_frames, n_particles = result.angles.shape
    lattice_size = result.occupancy.shape[-1]
    if result.sites.numel() and (
        int(result.sites.min()) < 0 or int(result.sites.max()) >= lattice_size
    ):
        raise ValueError("site index lies outside the occupancy lattice")
    linear_sites = (
        result.sites[..., 0] * lattice_size + result.sites[..., 1]
    ).view(n_trajectories * n_frames, n_particles)
    cosine_field = torch.zeros(
        (n_trajectories * n_frames, lattice_size * lattice_size),
        dtype=result.angles.dtype,
    )
    sine_field = torch.zeros_like(cosine_field)
    flat_angles = result.angles.reshape(n_trajectories * n_frames, n_particles)
    cosine_field.scatter_add_(1, linear_sites, torch.cos(flat_angles))
    sine_field.scatter_add_(1, linear_sites, torch.sin(flat_angles))
    cosine_field = cosine_field.view(
        n_trajectories, n_frames, lattice_size, lattice_size
    )
    sine_field = sine_field.view_as(cosine_field)
    return torch.cat(
        (density, cosine_field.unsqueeze(2), sine_field.unsqueeze(2)), dim=2
    )


__all__ = [
    "TCLABPConfig",
    "TCLABPTrajectory",
    "TCLABPResult",
    "simulate_trajectories",
    "encode_observations",
]
