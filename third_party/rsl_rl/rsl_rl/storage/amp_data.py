"""Task-blind expert-data protocol for AMP.

The AMP algorithm consumes expert motion data only through the ``AmpData``
protocol: a ``num_disc_groups`` count and a ``sample`` method. This keeps the
algorithm free of any task vocabulary (motion files, datasets, pairs) -- the
downstream task builds a conforming object from its own already-loaded motions.

``sample`` serves both the merged draw over every group and the per-group
matched draw.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class AmpData(Protocol):
    """Expert AMP-feature window source.

    ``num_disc_groups`` is the number of discriminator partitions (1 for the
    ``single`` discriminator mode). ``sample`` returns expert feature windows of
    shape ``(batch_size, num_frames, feature_dim)``.
    """

    num_disc_groups: int

    def sample(self, batch_size: int, disc_groups: torch.Tensor | None = None) -> torch.Tensor:
        """Draw ``batch_size`` expert windows.

        - ``disc_groups is None``: draw from the merged pool over all groups.
        - ``disc_groups`` a ``(batch_size,)`` long tensor: row ``i`` is drawn
          from group ``disc_groups[i]`` (per-row group-matched sampling).

        Returns ``(batch_size, num_frames, feature_dim)``.
        """
        ...


class TensorAmpData:
    """In-memory :class:`AmpData` over expert windows materialized as tensors.

    Reference / default implementation: holds one window pool per discriminator
    group and draws uniformly with replacement. A task may instead provide its
    own ``AmpData`` (e.g. windowing motion frames on the fly) as long as it
    satisfies the protocol.

    Args:
        group_windows: list of length ``num_disc_groups``; entry ``g`` is a
            ``(N_g, num_frames, feature_dim)`` tensor of expert windows.
        device: device the samples are returned on.
    """

    def __init__(self, group_windows: list[torch.Tensor], device: str | torch.device):
        if len(group_windows) < 1:
            raise ValueError("group_windows must have at least one group")
        self.device = device
        self._groups = [w.to(device) for w in group_windows]
        self.num_disc_groups = len(self._groups)
        self.num_frames = self._groups[0].shape[1]
        self.feature_dim = self._groups[0].shape[2]
        for g, w in enumerate(self._groups):
            if w.ndim != 3 or w.shape[1:] != (self.num_frames, self.feature_dim):
                raise ValueError(
                    f"group {g} windows have shape {tuple(w.shape)}, expected "
                    f"(N, {self.num_frames}, {self.feature_dim})"
                )
        self._merged = self._groups[0] if self.num_disc_groups == 1 else torch.cat(self._groups, dim=0)

    def sample(self, batch_size: int, disc_groups: torch.Tensor | None = None) -> torch.Tensor:
        if disc_groups is None:
            idx = torch.randint(self._merged.shape[0], (batch_size,), device=self.device)
            return self._merged[idx]
        out = torch.empty(batch_size, self.num_frames, self.feature_dim, device=self.device)
        for g in range(self.num_disc_groups):
            mask = disc_groups == g
            n = int(mask.sum())
            if n == 0:
                continue
            pool = self._groups[g]
            idx = torch.randint(pool.shape[0], (n,), device=self.device)
            out[mask] = pool[idx]
        return out
