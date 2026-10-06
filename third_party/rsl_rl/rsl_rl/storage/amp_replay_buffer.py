"""AMP replay buffer for storing multi-frame policy observations.

Group-aware: each stored window carries an optional per-sample ``disc_group``
index so multi_head / separate discriminators can route policy samples; in
single-discriminator mode the group is all-zeros and ignored.
"""

from __future__ import annotations

import torch


class AMPReplayBuffer:
    """Fixed-size circular buffer for multi-frame AMP observation states."""

    def __init__(
        self,
        amp_obs_dim: int,
        buffer_size: int,
        amp_num_frames: int,
        device: str | torch.device,
        num_disc_groups: int = 1,
    ):
        self.states = torch.zeros(buffer_size, amp_num_frames, amp_obs_dim).to(device)
        self.disc_groups = torch.zeros(buffer_size, dtype=torch.long, device=device)
        self.amp_num_frames = amp_num_frames
        self.buffer_size = buffer_size
        self.num_disc_groups = num_disc_groups
        self.device = device

        self.step = 0
        self.num_samples = 0

    def insert(self, states: torch.Tensor, disc_group: torch.Tensor | None = None) -> None:
        """Add new multi-frame states (and their discriminator groups) to the buffer.

        Args:
            states: (batch, amp_num_frames, amp_obs_dim)
            disc_group: (batch,) long, per-sample discriminator-group index.
                ``None`` (single-discriminator mode) stores zeros.
        """
        num_states = states.shape[0]
        if disc_group is None:
            disc_group = torch.zeros(num_states, dtype=torch.long, device=self.device)
        start_idx = self.step
        end_idx = self.step + num_states
        if end_idx > self.buffer_size:
            wrap = self.buffer_size - self.step
            self.states[self.step : self.buffer_size] = states[:wrap]
            self.states[: end_idx - self.buffer_size] = states[wrap:]
            self.disc_groups[self.step : self.buffer_size] = disc_group[:wrap]
            self.disc_groups[: end_idx - self.buffer_size] = disc_group[wrap:]
        else:
            self.states[start_idx:end_idx] = states
            self.disc_groups[start_idx:end_idx] = disc_group

        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size

    def sample(self, batch_size: int, balanced: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a random mini-batch of ``(states, disc_group)``.

        - ``balanced=False`` (default): a single uniform draw over all stored
          windows, so the minibatch's group composition is *proportional* to the
          stored (env-assignment) distribution.
        - ``balanced=True``: draw ~``batch_size // num_disc_groups`` windows from
          EACH of the ``num_disc_groups`` configured groups (a group with no
          stored samples yet is skipped). The denominator is the *configured*
          ``num_disc_groups`` -- NOT the set currently present -- so a
          transiently-empty group does not inflate the others' share, and each
          sub-discriminator trains on an equal share. The per-group *count* is
          ``batch_size // K`` rather than the full ``batch_size``, keeping the
          total minibatch budget the same as the single / multi_head modes.

        Returns ``states`` ``(N, amp_num_frames, amp_obs_dim)`` and ``disc_group``
        ``(N,)`` long (``N == batch_size`` when unbalanced).
        """
        if not balanced:
            idx = torch.randint(self.num_samples, (batch_size,), device=self.device)
            return self.states[idx], self.disc_groups[idx]

        stored_groups = self.disc_groups[: self.num_samples]
        per_group = max(1, batch_size // self.num_disc_groups)
        picks = []
        for g in range(self.num_disc_groups):
            g_idx = (stored_groups == g).nonzero(as_tuple=False).squeeze(1)
            if g_idx.numel() == 0:
                continue  # group not populated yet -- skip (don't reweight others)
            sel = g_idx[torch.randint(g_idx.numel(), (per_group,), device=self.device)]
            picks.append(sel)
        if not picks:
            # Degenerate: no configured group has stored samples -> uniform draw.
            idx = torch.randint(self.num_samples, (batch_size,), device=self.device)
            return self.states[idx], self.disc_groups[idx]
        idx = torch.cat(picks)
        return self.states[idx], self.disc_groups[idx]
