"""Generic mixin for algorithms to report per-episode metrics.

Supports three reduction modes matching common RL framework patterns:

- ``"mean"``: per-step average (sum / step_count). Default.
  Matches mjlab's MetricsManager ``reduce="mean"``.
- ``"last"``: value from the final step of the episode.
  Matches mjlab's MetricsManager ``reduce="last"``.
- ``"constant"``: episode sum divided by a constant.
  Matches mjlab's RewardManager (sum / max_episode_length_s).

Usage::

    class MyAlgorithm(AlgorithmEpisodeMetricMixin, PPO):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.init_algorithm_metric_buffers(env.num_envs, self.device)
            self.register_episode_metric("disc_reward")
            self.register_episode_metric("episode_return", reduce="constant", divisor=20.0)
            self.register_episode_metric("success", reduce="last")

        def process_env_step(self, obs, rewards, dones, extras):
            self.log_episode_metric(extras, dones, "disc_reward", values)
            super().process_env_step(obs, rewards, dones, extras)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

EPISODE_METRIC_KEYS = ("episode", "log")


@dataclass
class _MetricConfig:
    reduce: Literal["mean", "last", "constant"]
    divisor: float
    prefix: str
    rolling: bool


class AlgorithmEpisodeMetricMixin:
    """Mixin that gives algorithms per-episode metric tracking.

    Call ``init_algorithm_metric_buffers()`` after ``super().__init__()``
    to set up per-env accumulators. Register metrics with
    ``register_episode_metric()`` to configure reduction mode and
    optional divisor. Then call ``log_episode_metric()`` during
    ``process_env_step()`` to record per-step values.
    """

    def init_algorithm_metric_buffers(self, num_envs: int, device: str, metric_prefix: str = "Episode_Metrics") -> None:
        self._metric_configs: dict[str, _MetricConfig] = {}
        self._metric_episode_sums: dict[str, torch.Tensor] = {}
        self._metric_step_counts: dict[str, torch.Tensor] = {}
        self._metric_step_values: dict[str, torch.Tensor] = {}
        self._metric_num_envs = num_envs
        self._metric_device = device
        self._metric_prefix = metric_prefix

    def register_episode_metric(
        self,
        name: str,
        reduce: Literal["mean", "last", "constant"] = "mean",
        divisor: float = 1.0,
        prefix: str | None = None,
        rolling: bool = False,
    ) -> None:
        """Register a metric with its reduction mode.

        Args:
            name: Metric name.
            reduce: ``"mean"`` (sum/count), ``"last"`` (final step value),
                or ``"constant"`` (sum/divisor).
            divisor: Constant divisor for ``"constant"`` mode
                (e.g. ``max_episode_length_s``).
            prefix: Logging prefix. Defaults to the global
                ``metric_prefix`` set in ``init_algorithm_metric_buffers``.
            rolling: The per-iteration mean is always emitted under
                ``{prefix}/{name}``. If True, *additionally* emit raw
                per-episode values to ``extras["metric_buffers"][name]`` for
                the Logger's rolling maxlen=100 smoothing (logged as
                ``Train/mean_{name}``), matching the main reward buffer.
        """
        self._metric_configs[name] = _MetricConfig(
            reduce=reduce, divisor=divisor, prefix=prefix or self._metric_prefix, rolling=rolling
        )
        self._metric_episode_sums[name] = torch.zeros(self._metric_num_envs, device=self._metric_device)
        self._metric_step_counts[name] = torch.zeros(self._metric_num_envs, device=self._metric_device)
        self._metric_step_values[name] = torch.zeros(self._metric_num_envs, device=self._metric_device)

    def log_episode_metric(
        self,
        extras: dict,
        dones: torch.Tensor,
        name: str,
        values: torch.Tensor,
    ) -> None:
        """Accumulate a per-step value and emit episode result on done.

        Args:
            extras: The extras dict from env.step().
            dones: Done flags, shape ``(num_envs,)`` or ``(num_envs, 1)``.
            name: Metric name (must be registered or will auto-register
                with ``reduce="mean"``).
            values: Per-env values for this step, broadcastable to
                ``(num_envs,)``.
        """
        assert name in self._metric_configs, (
            f"Metric '{name}' not registered. Call register_episode_metric('{name}') first."
        )

        values = values.detach().reshape(self._metric_num_envs).to(self._metric_device)

        self._metric_episode_sums[name] += values
        self._metric_step_counts[name] += 1
        self._metric_step_values[name] = values

        new_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
        if new_ids.numel() == 0:
            return

        cfg = self._metric_configs[name]
        if cfg.reduce == "mean":
            episode_values = self._metric_episode_sums[name][new_ids] / self._metric_step_counts[name][new_ids].clamp(
                min=1
            )
        elif cfg.reduce == "last":
            episode_values = self._metric_step_values[name][new_ids]
        elif cfg.reduce == "constant":
            episode_values = self._metric_episode_sums[name][new_ids] / cfg.divisor

        # Always emit the per-iteration mean via the episode-extras channel.
        ep_dict = self._resolve_episode_dict(extras)
        assert ep_dict is not None, (
            f"extras must contain 'episode' or 'log' when envs are done, but got keys: {list(extras.keys())}"
        )
        ep_dict[f"{cfg.prefix}/{name}"] = torch.mean(episode_values)

        # Additionally feed the Logger's rolling maxlen=100 metric buffer.
        if cfg.rolling:
            buffers = extras.setdefault("metric_buffers", {})
            buffers[name] = episode_values.cpu().tolist()

        self._metric_episode_sums[name][new_ids] = 0
        self._metric_step_counts[name][new_ids] = 0
        self._metric_step_values[name][new_ids] = 0

    @staticmethod
    def _resolve_episode_dict(extras: dict) -> dict | None:
        """Find the episode metrics dict in extras, if present."""
        for key in EPISODE_METRIC_KEYS:
            if key in extras:
                return extras[key]
        return None
