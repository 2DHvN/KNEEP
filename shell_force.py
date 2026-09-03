"""Periodic shell-force KNEEP architecture used by the field experiments."""

from __future__ import annotations

from numbers import Integral

import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    raise ValueError("activation must be 'relu' or 'elu'")


class ShellAbsoluteConv2d(nn.Module):
    """Learn a full channel-mixing convolution on one Chebyshev shell."""

    def __init__(self, in_channels: int, out_channels: int, radius: int) -> None:
        super().__init__()
        if radius <= 0:
            raise ValueError("radius must be positive")

        self.radius = radius
        kernel_size = 2 * radius + 1
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        mask = torch.zeros(1, 1, kernel_size, kernel_size)
        mask[0, 0, 0, :] = 1.0
        mask[0, 0, -1, :] = 1.0
        mask[0, 0, :, 0] = 1.0
        mask[0, 0, :, -1] = 1.0
        mask[0, 0, radius, radius] = 0.0
        self.register_buffer("mask", mask)

        # Preserve the initialization sequence and state-dict compatibility of
        # the source model. Standard 1x1 convolutions are initialized later.
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.bias)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        radius = self.radius
        padded = F.pad(field, (radius, radius, radius, radius), mode="circular")
        return F.conv2d(padded, self.weight * self.mask, bias=self.bias)


class _ShellForceBranch2D(nn.Module):
    def __init__(
        self,
        radius: int,
        input_components: int,
        output_components: int,
        hidden_channels: int,
        hidden_layers: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.radius = radius

        if radius == 0:
            self.center_proj = nn.Conv2d(
                input_components, hidden_channels, kernel_size=1, bias=True
            )
            self.rel_proj = None
        else:
            self.center_proj = None
            self.rel_proj = ShellAbsoluteConv2d(
                input_components, hidden_channels, radius=radius
            )

        layers: list[nn.Module] = []
        for _ in range(max(hidden_layers - 1, 0)):
            layers.extend(
                (
                    nn.Conv2d(
                        hidden_channels, hidden_channels, kernel_size=1, bias=True
                    ),
                    _activation(activation),
                )
            )
        layers.append(
            nn.Conv2d(hidden_channels, output_components, kernel_size=1, bias=True)
        )
        self.activation = _activation(activation)
        self.force_head = nn.Sequential(*layers)

    def forward(self, midpoint: torch.Tensor) -> torch.Tensor:
        if self.radius == 0:
            hidden = self.center_proj(midpoint)
        else:
            hidden = self.rel_proj(midpoint)
        return self.force_head(self.activation(hidden))


class ShellForceKNEEP2D(nn.Module):
    """Periodic KNEEP with a local branch and learned absolute shells.

    Input shape is ``[batch, 2, components, height, width]``. The ordinary
    output contains one spatially averaged EP increment per branch and has
    shape ``[batch, max_distance + 1]``. All components condition the learned
    force. ``ep_component_indices`` can restrict the increment components in
    the final force--increment contraction. This lets reversible variables,
    such as LABP orientation, provide context without assigning them a direct
    entropy-production contribution.
    """

    def __init__(
        self,
        n_components: int = 2,
        hidden_channels: int = 8,
        hidden_layers: int = 2,
        max_distance: int = 4,
        ep_component_indices: tuple[int, ...] | None = None,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        if n_components <= 0 or hidden_channels <= 0 or hidden_layers <= 0:
            raise ValueError("channel and layer counts must be positive")
        if max_distance < 0:
            raise ValueError("max_distance must be nonnegative")
        _activation(activation)

        if ep_component_indices is None:
            ep_component_indices = tuple(range(n_components))
        else:
            if any(
                isinstance(index, bool) or not isinstance(index, Integral)
                for index in ep_component_indices
            ):
                raise ValueError("ep_component_indices must contain integers")
            ep_component_indices = tuple(int(index) for index in ep_component_indices)
        if not ep_component_indices:
            raise ValueError("ep_component_indices must not be empty")
        if len(set(ep_component_indices)) != len(ep_component_indices):
            raise ValueError("ep_component_indices must be unique")
        if any(index < 0 or index >= n_components for index in ep_component_indices):
            raise ValueError("ep_component_indices contain an out-of-range channel")

        self.n_components = n_components
        self.hidden_channels = hidden_channels
        self.hidden_layers = hidden_layers
        self.ep_component_indices = ep_component_indices
        self.max_distance = max_distance
        self.activation_name = activation
        self.branches = nn.ModuleList(
            [
                _ShellForceBranch2D(
                    radius=radius,
                    input_components=n_components,
                    output_components=len(ep_component_indices),
                    hidden_channels=hidden_channels,
                    hidden_layers=hidden_layers,
                    activation=activation,
                )
                for radius in range(max_distance + 1)
            ]
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _split_pair(self, pair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pair.ndim != 5:
            raise ValueError("expected [batch, 2, components, height, width]")
        if pair.shape[1] != 2 or pair.shape[2] != self.n_components:
            raise ValueError(
                f"expected pair shape [batch, 2, {self.n_components}, height, width]"
            )
        first, second = pair[:, 0], pair[:, 1]
        return 0.5 * (first + second), second - first

    def component_ep_maps(self, pair: torch.Tensor) -> torch.Tensor:
        """Return raw EP maps resolved by shell and contracted component.

        The result has shape ``[batch, branch, component, height, width]``.
        Its component axis follows :attr:`ep_component_indices`, and summing
        that axis exactly recovers ``forward(pair, return_maps=True)``.  The
        decomposition is coordinate dependent, but makes each selected state
        increment's learned contribution directly inspectable.
        """
        midpoint, increment = self._split_pair(pair)
        contracted_increment = increment[:, self.ep_component_indices]
        return torch.stack(
            [branch(midpoint) * contracted_increment for branch in self.branches],
            dim=1,
        )

    def component_ep_scores(self, pair: torch.Tensor) -> torch.Tensor:
        """Return spatially averaged component EP scores ``[batch, branch, component]``."""
        return self.component_ep_maps(pair).mean(dim=(-2, -1))

    def forward(
        self,
        pair: torch.Tensor,
        return_maps: bool = False,
        return_forces: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Evaluate branch scores, raw score maps, or normalized-coordinate forces.

        ``return_maps=True`` returns per-site raw scores whose spatial mean is
        the ordinary branch output. ``return_forces=True`` alone returns only
        forces for the contracted components; requesting both returns
        ``(maps, forces)``.
        """
        midpoint, increment = self._split_pair(pair)
        contracted_increment = increment[:, self.ep_component_indices]
        branch_values: list[torch.Tensor] = []
        maps: list[torch.Tensor] = []
        forces: list[torch.Tensor] = []

        for branch in self.branches:
            force = branch(midpoint)
            local_ep = (force * contracted_increment).sum(dim=1, keepdim=True)
            if return_forces:
                forces.append(force.unsqueeze(1))
            if return_maps:
                maps.append(local_ep)
            else:
                branch_values.append(local_ep.mean(dim=(2, 3)))

        if return_maps:
            map_tensor = torch.cat(maps, dim=1)
            if return_forces:
                return map_tensor, torch.cat(forces, dim=1)
            return map_tensor

        if return_forces:
            return torch.cat(forces, dim=1)
        return torch.cat(branch_values, dim=1)
