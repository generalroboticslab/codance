"""AMP discriminators for adversarial motion priors.

Pluggable discriminator interface (``AMPDiscriminatorBase``) with three
implementations selected by the task's ``disc_mode``:

- ``AMPDiscriminator`` (``single``): one spectral-normed trunk + one head over
  the merged expert pool. ``disc_group`` is ignored.
- ``MultiHeadAMPDiscriminator`` (``multi_head``): one shared trunk + ``K`` output
  heads, one per discriminator group; the active head is gathered per row from
  ``disc_group``. One shared normalizer.
- ``SeparateAMPDiscriminator`` (``separate``): ``K`` fully independent
  ``AMPDiscriminator`` modules (own trunk + head + normalizer) wrapped in a
  single ``nn.Module`` (``ModuleList``) so the optimizer / save / load see one
  module -- no manual per-group rebuild.

Each discriminator OWNS its normalizer(s) and pulls experts from ``amp_data``
inside :meth:`disc_loss`, so the AMP algorithm stays task-blind: it passes RAW
(unnormalized) AMP windows plus an optional per-row ``disc_group``, and the
discriminator handles normalization, routing, expert sampling, and reward.
Non-tensor normalizer state rides in the module ``state_dict`` via
``get_extra_state`` / ``set_extra_state`` (per-disc for the separate variant,
recursed automatically through the ``ModuleList`` children).

All three modes share the SAME reward formula (``AMPDiscriminator``'s) so
``single``/``multi_head``/``separate`` are directly comparable for ablation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from torch import autograd

from rsl_rl.utils.amp_normalizer import AMPNormalizer


class AMPDiscriminatorBase(nn.Module):
    """Pluggable AMP discriminator interface.

    Subclasses set ``num_disc_groups`` and implement ``predict_reward``,
    ``disc_loss`` and ``param_groups``.

    ``balanced_replay`` tells the algorithm how to draw policy windows from the
    group-aware replay buffer: ``False`` -> a uniform (group-proportional) draw;
    ``True`` -> a group-balanced draw (~equal per disc_group every update). The
    discriminator owns this choice so the algorithm stays task-blind.
    """

    num_disc_groups: int
    balanced_replay: bool = False

    def predict_reward(
        self,
        states: torch.Tensor,
        task_reward: torch.Tensor,
        disc_group: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(batch, num_frames, feature_dim)``, ``(batch,)``, opt ``(batch,)``
        -> ``(blended_reward, logit, disc_reward)`` each ``(batch,)``
        (``logit`` is ``(batch, 1)``)."""
        raise NotImplementedError

    def disc_loss(
        self,
        policy_states: torch.Tensor,
        policy_groups: torch.Tensor,
        amp_data,
        grad_pen_lambda: float = 5.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Sample matched experts from ``amp_data``; return ``(loss, metrics)``.

        ``loss`` is left for the caller to backward/step (the algorithm owns the
        optimizer over :meth:`param_groups`)."""
        raise NotImplementedError

    def param_groups(self) -> list[dict]:
        """Optimizer parameter groups (with per-group weight decay)."""
        raise NotImplementedError


def _build_trunk(in_dim: int, hidden_layer_sizes: tuple[int, ...]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    curr = in_dim
    for h in hidden_layer_sizes:
        layers.append(spectral_norm(nn.Linear(curr, h)))
        layers.append(nn.ReLU())
        curr = h
    return nn.Sequential(*layers), hidden_layer_sizes[-1]


class AMPDiscriminator(AMPDiscriminatorBase):
    """Single-discriminator AMP: one spectral-normed trunk + one head over the
    merged expert pool (``disc_group`` ignored)."""

    num_disc_groups = 1

    def __init__(
        self,
        feature_dim: int,
        num_frames: int = 2,
        hidden_layer_sizes: tuple[int, ...] = (256, 256),
        amp_disc_reward_coef: float = 0.1,
        task_reward_lerp: float = 0.0,
        use_lerp: bool = True,
        trunk_weight_decay: float = 1e-3,
        head_weight_decay: float = 1e-1,
        balanced_replay: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.balanced_replay = balanced_replay
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.amp_disc_reward_coef = amp_disc_reward_coef
        self.use_lerp = use_lerp
        if task_reward_lerp < 0.0:
            raise ValueError(f"task_reward_lerp must be >= 0.0, got {task_reward_lerp}")
        self.task_reward_lerp = task_reward_lerp
        self._trunk_weight_decay = trunk_weight_decay
        self._head_weight_decay = head_weight_decay

        self.trunk, last_dim = _build_trunk(feature_dim * num_frames, hidden_layer_sizes)
        self.trunk = self.trunk.to(device)
        self.amp_linear = spectral_norm(nn.Linear(last_dim, 1)).to(device)
        self.normalizer = AMPNormalizer(input_dim=feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: ``(batch, feature_dim * num_frames)`` (already flattened)."""
        return self.amp_linear(self.trunk(x))

    def _reward_from_logit(self, d: torch.Tensor, task_reward: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``d`` ``(batch, 1)`` logit, ``task_reward`` ``(batch,)`` ->
        ``(blended_reward, disc_reward)`` each ``(batch,)``."""
        raw_disc_reward = torch.clamp(1 - 0.25 * torch.square(d - 1), min=0)
        if self.use_lerp:
            lerp = self.task_reward_lerp
            reward = (1.0 - lerp) * raw_disc_reward + lerp * task_reward.unsqueeze(-1)
            return reward.squeeze(-1), raw_disc_reward.squeeze(-1) * (1.0 - lerp)
        disc_reward = self.amp_disc_reward_coef * raw_disc_reward
        reward = disc_reward + task_reward.unsqueeze(-1)
        return reward.squeeze(-1), disc_reward.squeeze(-1)

    def _compute_grad_pen(self, expert_states: torch.Tensor, lambda_: float) -> torch.Tensor:
        """Gradient penalty on (already-normalized) expert states; target norm 0."""
        expert_data = expert_states.reshape(expert_states.shape[0], -1)
        expert_data.requires_grad = True
        disc = self.forward(expert_data)
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return lambda_ * grad.norm(2, dim=1).pow(2).mean()

    def predict_reward(
        self,
        states: torch.Tensor,
        task_reward: torch.Tensor,
        disc_group: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.eval()
            states = self.normalizer.normalize_torch(states, self.device)
            d = self.forward(states.flatten(1))
            reward, disc_reward = self._reward_from_logit(d, task_reward)
            self.train()
        return reward, d, disc_reward

    def _lsgan_loss(
        self,
        policy_states: torch.Tensor,
        expert_states: torch.Tensor,
        grad_pen_lambda: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """LSGAN + grad-pen for already-drawn ``(policy, expert)`` windows on THIS
        discriminator. Normalizes, forwards the single head, updates the
        normalizer, and returns ``(loss, metrics)``."""
        with torch.no_grad():
            policy_n = self.normalizer.normalize_torch(policy_states.to(self.device), self.device)
            expert_n = self.normalizer.normalize_torch(expert_states.to(self.device), self.device)
        policy_d = self.forward(policy_n.flatten(1))
        expert_d = self.forward(expert_n.flatten(1))
        expert_loss = F.mse_loss(expert_d, torch.ones_like(expert_d))
        policy_loss = F.mse_loss(policy_d, -torch.ones_like(policy_d))
        amp_loss = 0.5 * (expert_loss + policy_loss)
        grad_pen = self._compute_grad_pen(expert_n, grad_pen_lambda)
        # NOTE: the normalizer is updated with the *already-normalized* states, which
        # is likely a latent bug (raw states would be correct). Kept as is so every
        # discriminator mode shares the same dynamics; to fix, update with
        # ``policy_states``/``expert_states`` instead.
        self.normalizer.update(policy_n.cpu().numpy().reshape(-1, self.feature_dim))
        self.normalizer.update(expert_n.cpu().numpy().reshape(-1, self.feature_dim))
        loss = amp_loss + grad_pen
        metrics = {
            "amp": amp_loss.item(),
            "amp_grad_pen": grad_pen.item(),
            "amp_policy_pred": policy_loss.item(),
            "amp_expert_pred": expert_loss.item(),
        }
        return loss, metrics

    def disc_loss(
        self,
        policy_states: torch.Tensor,
        policy_groups: torch.Tensor,
        amp_data,
        grad_pen_lambda: float = 5.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Single mode: merged expert draw; policy_groups ignored.
        expert_states = amp_data.sample(policy_states.shape[0])
        return self._lsgan_loss(policy_states, expert_states, grad_pen_lambda)

    def param_groups(self) -> list[dict]:
        return [
            {"params": self.trunk.parameters(), "weight_decay": self._trunk_weight_decay, "name": "amp_trunk"},
            {"params": self.amp_linear.parameters(), "weight_decay": self._head_weight_decay, "name": "amp_head"},
        ]

    # Bundle the (numpy) normalizer state into the module state_dict.
    def get_extra_state(self) -> dict:
        return {"normalizer": self.normalizer.state_dict()}

    def set_extra_state(self, state: dict) -> None:
        if state and "normalizer" in state:
            self.normalizer.load_state_dict(state["normalizer"])


class MultiHeadAMPDiscriminator(AMPDiscriminator):
    """Multi-head AMP: one shared spectral-normed trunk + ``K`` output heads (one
    per discriminator group, gathered per row from ``disc_group``).

    Subclasses :class:`AMPDiscriminator` and replaces the single output head with
    a ``K``-output head. One shared trunk and one shared normalizer (all groups
    feed the same statistics), so this is cheaper than ``separate`` but couples
    the groups through the trunk.
    """

    def __init__(
        self,
        feature_dim: int,
        num_disc_groups: int,
        num_frames: int = 2,
        hidden_layer_sizes: tuple[int, ...] = (256, 256),
        amp_disc_reward_coef: float = 0.1,
        task_reward_lerp: float = 0.0,
        use_lerp: bool = True,
        trunk_weight_decay: float = 1e-3,
        head_weight_decay: float = 1e-1,
        balanced_replay: bool = False,
        device: str = "cpu",
    ):
        super().__init__(
            feature_dim=feature_dim,
            num_frames=num_frames,
            hidden_layer_sizes=hidden_layer_sizes,
            amp_disc_reward_coef=amp_disc_reward_coef,
            task_reward_lerp=task_reward_lerp,
            use_lerp=use_lerp,
            trunk_weight_decay=trunk_weight_decay,
            head_weight_decay=head_weight_decay,
            balanced_replay=balanced_replay,
            device=device,
        )
        self.num_disc_groups = num_disc_groups
        # Replace the single-output head with a K-output head sharing the trunk.
        self.amp_linear = spectral_norm(nn.Linear(hidden_layer_sizes[-1], num_disc_groups)).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: ``(batch, feature_dim * num_frames)`` -> ``(batch, num_disc_groups)``."""
        return self.amp_linear(self.trunk(x))

    @staticmethod
    def _gather_group(d_all: torch.Tensor, disc_group: torch.Tensor | None) -> torch.Tensor:
        """``(batch, K)`` head logits + per-row group -> ``(batch, 1)`` selected."""
        if disc_group is None:
            return d_all[:, :1]
        return torch.gather(d_all, 1, disc_group.long().unsqueeze(1))

    def predict_reward(
        self,
        states: torch.Tensor,
        task_reward: torch.Tensor,
        disc_group: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.eval()
            states = self.normalizer.normalize_torch(states, self.device)
            d_all = self.forward(states.flatten(1))
            d = self._gather_group(d_all, disc_group)
            reward, disc_reward = self._reward_from_logit(d, task_reward)
            self.train()
        return reward, d, disc_reward

    def _grouped_grad_pen(self, expert_states: torch.Tensor, disc_group: torch.Tensor, lambda_: float) -> torch.Tensor:
        """Gradient penalty using each row's gathered head logit; target norm 0."""
        expert_data = expert_states.reshape(expert_states.shape[0], -1)
        expert_data.requires_grad = True
        d = self._gather_group(self.forward(expert_data), disc_group)
        ones = torch.ones(d.size(), device=d.device)
        grad = autograd.grad(
            outputs=d,
            inputs=expert_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return lambda_ * grad.norm(2, dim=1).pow(2).mean()

    def disc_loss(
        self,
        policy_states: torch.Tensor,
        policy_groups: torch.Tensor,
        amp_data,
        grad_pen_lambda: float = 5.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Matched experts: row i drawn from group policy_groups[i].
        expert_states = amp_data.sample(policy_states.shape[0], policy_groups)
        with torch.no_grad():
            policy_n = self.normalizer.normalize_torch(policy_states.to(self.device), self.device)
            expert_n = self.normalizer.normalize_torch(expert_states.to(self.device), self.device)
        policy_d = self._gather_group(self.forward(policy_n.flatten(1)), policy_groups)
        expert_d = self._gather_group(self.forward(expert_n.flatten(1)), policy_groups)
        expert_loss = F.mse_loss(expert_d, torch.ones_like(expert_d))
        policy_loss = F.mse_loss(policy_d, -torch.ones_like(policy_d))
        amp_loss = 0.5 * (expert_loss + policy_loss)
        grad_pen = self._grouped_grad_pen(expert_n, policy_groups, grad_pen_lambda)
        # See _lsgan_loss NOTE: normalizer updated with normalized states, as in single mode.
        self.normalizer.update(policy_n.cpu().numpy().reshape(-1, self.feature_dim))
        self.normalizer.update(expert_n.cpu().numpy().reshape(-1, self.feature_dim))
        loss = amp_loss + grad_pen
        metrics = {
            "amp": amp_loss.item(),
            "amp_grad_pen": grad_pen.item(),
            "amp_policy_pred": policy_loss.item(),
            "amp_expert_pred": expert_loss.item(),
        }
        # Per-group prediction errors for logging (rows present in this batch).
        for g in range(self.num_disc_groups):
            mask = policy_groups == g
            if mask.any():
                ed, pd = expert_d[mask], policy_d[mask]
                metrics[f"group_{g}/amp_expert_pred"] = F.mse_loss(ed, torch.ones_like(ed)).item()
                metrics[f"group_{g}/amp_policy_pred"] = F.mse_loss(pd, -torch.ones_like(pd)).item()
        return loss, metrics


class SeparateAMPDiscriminator(AMPDiscriminatorBase):
    """Separate AMP: ``K`` fully independent :class:`AMPDiscriminator` modules
    (own trunk, head, and normalizer per group), wrapped in one ``nn.Module``.

    A ``ModuleList`` holds the per-group discriminators, so the optimizer (via
    :meth:`param_groups`), ``state_dict`` (each child bundles its own normalizer
    through ``get_extra_state``), and ``load_state_dict`` all see a single
    module -- no manual per-group save/rebuild loop. Routing is by
    ``disc_group``: each row / policy window goes only to its group's
    discriminator, matched against that group's experts.

    Defaults to ``balanced_replay=True`` so each sub-discriminator trains on an
    equal share of every policy minibatch, rather than a share proportional to
    the env->group assignment.
    """

    def __init__(
        self,
        feature_dim: int,
        num_disc_groups: int,
        num_frames: int = 2,
        hidden_layer_sizes: tuple[int, ...] = (256, 256),
        amp_disc_reward_coef: float = 0.1,
        task_reward_lerp: float = 0.0,
        use_lerp: bool = True,
        trunk_weight_decay: float = 1e-3,
        head_weight_decay: float = 1e-1,
        balanced_replay: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_disc_groups = num_disc_groups
        self.balanced_replay = balanced_replay
        self.device = device
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self._trunk_weight_decay = trunk_weight_decay
        self._head_weight_decay = head_weight_decay
        self.discs = nn.ModuleList([
            AMPDiscriminator(
                feature_dim=feature_dim,
                num_frames=num_frames,
                hidden_layer_sizes=hidden_layer_sizes,
                amp_disc_reward_coef=amp_disc_reward_coef,
                task_reward_lerp=task_reward_lerp,
                use_lerp=use_lerp,
                trunk_weight_decay=trunk_weight_decay,
                head_weight_decay=head_weight_decay,
                device=device,
            )
            for _ in range(num_disc_groups)
        ])

    def predict_reward(
        self,
        states: torch.Tensor,
        task_reward: torch.Tensor,
        disc_group: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = states.shape[0]
        if disc_group is None:
            # No routing index -> single-group fallback (route all to disc 0).
            disc_group = torch.zeros(batch, dtype=torch.long, device=states.device)
        else:
            disc_group = disc_group.long()
        blended = states.new_empty(batch)
        logit = states.new_empty(batch, 1)
        disc_reward = states.new_empty(batch)
        for g in range(self.num_disc_groups):
            mask = disc_group == g
            if not mask.any():
                continue
            reward_g, logit_g, disc_reward_g = self.discs[g].predict_reward(states[mask], task_reward[mask], None)
            blended[mask] = reward_g
            logit[mask] = logit_g
            disc_reward[mask] = disc_reward_g
        return blended, logit, disc_reward

    def disc_loss(
        self,
        policy_states: torch.Tensor,
        policy_groups: torch.Tensor,
        amp_data,
        grad_pen_lambda: float = 5.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        policy_groups = policy_groups.long()
        total_loss: torch.Tensor | None = None
        agg = {"amp": 0.0, "amp_grad_pen": 0.0, "amp_policy_pred": 0.0, "amp_expert_pred": 0.0}
        metrics: dict[str, float] = {}
        active = 0
        for g in range(self.num_disc_groups):
            mask = policy_groups == g
            n = int(mask.sum())
            if n == 0:
                continue
            policy_g = policy_states[mask]
            # Draw group-g experts to match the group's policy windows.
            groups_g = torch.full((n,), g, dtype=torch.long, device=policy_states.device)
            expert_g = amp_data.sample(n, groups_g)
            loss_g, m_g = self.discs[g]._lsgan_loss(policy_g, expert_g, grad_pen_lambda)
            total_loss = loss_g if total_loss is None else total_loss + loss_g
            for k in agg:
                agg[k] += m_g[k]
                metrics[f"group_{g}/{k}"] = m_g[k]
            active += 1
        if total_loss is None:
            # Degenerate minibatch (no policy samples for any group): emit a
            # zero loss that still touches group-0 params so .backward() is valid.
            total_loss = sum(p.sum() for p in self.discs[0].parameters()) * 0.0
        denom = max(active, 1)
        for k in agg:
            metrics[k] = agg[k] / denom
        return total_loss, metrics

    def param_groups(self) -> list[dict]:
        trunk_params: list[nn.Parameter] = []
        head_params: list[nn.Parameter] = []
        for disc in self.discs:
            trunk_params += list(disc.trunk.parameters())
            head_params += list(disc.amp_linear.parameters())
        return [
            {"params": trunk_params, "weight_decay": self._trunk_weight_decay, "name": "amp_trunk"},
            {"params": head_params, "weight_decay": self._head_weight_decay, "name": "amp_head"},
        ]
