"""AMP observation normalizer using running mean/std.

``RunningMeanStd`` tracks per-feature moments with a parallel (Chan) update;
``AMPNormalizer`` adds clipping and a torch path for the discriminator inputs.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


class RunningMeanStd:
    """Online running mean and variance using Welford's algorithm."""

    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state_dict: dict) -> None:
        self.mean = state_dict["mean"]
        self.var = state_dict["var"]
        self.count = state_dict["count"]


class AMPNormalizer(RunningMeanStd):
    """Normalizer for AMP observations with torch support."""

    def __init__(self, input_dim: int, epsilon: float = 1e-4, clip_obs: float = 10.0):
        super().__init__(shape=(input_dim,))
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize(self, input: np.ndarray) -> np.ndarray:
        return np.clip(
            (input - self.mean) / np.sqrt(self.var + self.epsilon),
            -self.clip_obs,
            self.clip_obs,
        )

    def normalize_torch(self, input: torch.Tensor, device: str | torch.device) -> torch.Tensor:
        """Normalize a torch tensor of AMP observations.

        Handles both 2D (batch, obs_dim) and 3D (batch, num_frames, obs_dim) inputs.
        """
        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=device)
        std = torch.as_tensor(self.var + self.epsilon, dtype=torch.float32, device=device).sqrt()
        return ((input - mean) / std).clamp(-self.clip_obs, self.clip_obs)
