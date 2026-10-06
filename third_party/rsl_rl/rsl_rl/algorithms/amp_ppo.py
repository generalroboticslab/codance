"""AMP algorithm mixin and AMP-PPO composition.

``AMPMixin`` adds adversarial-motion-prior discriminator training to any
PPO-like base algorithm. AMP observations are read from ``obs["amp"]`` (a
TensorDict obs group with ``flatten_history_dim=False``, shape
``(num_envs, num_frames, feature_dim)``) and an optional per-env discriminator
routing index from ``obs["amp_group"]``.

The algorithm is task-blind: it forwards RAW (unnormalized) AMP windows plus the
group index to a pluggable discriminator that owns its normalizer, expert
sampling (via the ``amp_data`` protocol), routing, and reward. Construction
grounds ``feature_dim``/``num_frames`` from ``obs["amp"]`` and
``num_disc_groups`` from ``amp_data`` -- no env-derived fields travel through the
config. See ``rsl_rl.algorithms.amp_contract`` for the env<->algorithm
handshake (obs keys + terminal-observation contract).

Usage::

    class AMP_PPO(AMPMixin, PPO):
        pass
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.algorithm_metrics_mixin import AlgorithmEpisodeMetricMixin
from rsl_rl.algorithms.amp_contract import (
    AMP_GROUP_KEY,
    AMP_OBS_KEY,
    TERMINAL_ENV_IDS,
    TERMINAL_KEY,
    TERMINAL_OBS,
    validate_disc_mode,
)
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.modules.amp_discriminator import (
    AMPDiscriminator,
    AMPDiscriminatorBase,
    MultiHeadAMPDiscriminator,
    SeparateAMPDiscriminator,
)
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.amp_data import AmpData
from rsl_rl.storage.amp_replay_buffer import AMPReplayBuffer
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


def _patch_terminal_obs(obs: TensorDict, terminal: dict | None) -> TensorDict:
    """Return an obs view with done-env rows replaced by the captured terminal
    snapshot (mjlab's ``capture_terminal_observations`` option,
    https://github.com/mujocolab/mjlab/pull/1030).

    Under auto_reset, done envs return post-reset obs; this substitutes the true
    terminal observation (delay+history applied, noise-free) for every captured
    group at the done-env indices, so a consumer (e.g. ``amp_gate_fn``) sees the
    terminal transition rather than the next episode's initial state. Shallow:
    unpatched groups are shared with ``obs`` and only patched groups are cloned;
    returns ``obs`` unchanged when there is no terminal snapshot.
    """
    if terminal is None:
        return obs
    env_ids = terminal[TERMINAL_ENV_IDS]
    terminal_obs = terminal[TERMINAL_OBS]
    patched = obs.copy()
    for group, value in terminal_obs.items():
        if group in patched.keys():
            g = patched[group].clone()
            g[env_ids] = value
            patched[group] = g
    return patched


def _build_discriminator(
    disc_mode: str,
    feature_dim: int,
    num_frames: int,
    num_disc_groups: int,
    device: str,
    disc_cfg: dict,
) -> AMPDiscriminatorBase:
    """Construct the discriminator for ``disc_mode`` (validated against groups).

    Each mode subclasses ``AMPDiscriminatorBase`` and owns its normalizer(s),
    routing, expert sampling and reward; ``feature_dim``/``num_frames`` are
    grounded from the obs and ``num_disc_groups`` from ``amp_data``.
    """
    validate_disc_mode(disc_mode, num_disc_groups)
    if disc_mode == "single":
        return AMPDiscriminator(feature_dim=feature_dim, num_frames=num_frames, device=device, **disc_cfg)
    if disc_mode == "multi_head":
        return MultiHeadAMPDiscriminator(
            feature_dim=feature_dim,
            num_disc_groups=num_disc_groups,
            num_frames=num_frames,
            device=device,
            **disc_cfg,
        )
    if disc_mode == "separate":
        return SeparateAMPDiscriminator(
            feature_dim=feature_dim,
            num_disc_groups=num_disc_groups,
            num_frames=num_frames,
            device=device,
            **disc_cfg,
        )
    raise ValueError(f"unknown disc_mode {disc_mode!r}")  # unreachable; validate_disc_mode guards


class AMPMixin(AlgorithmEpisodeMetricMixin):
    """Mixin that adds AMP discriminator training to an RL algorithm.

    Reads multi-frame AMP observations from ``obs["amp"]`` and an optional
    per-env routing index from ``obs["amp_group"]``, blends the discriminator
    style reward into the task reward, and trains the discriminator either
    sequentially (a separate pass with its own optimizer) or interleaved with
    the PPO update.

    Expects the base class to provide ``self.actor``, ``self.critic``,
    ``self.storage``, ``self.optimizer``, and standard PPO-like methods.
    """

    discriminator: AMPDiscriminatorBase
    amp_storage: AMPReplayBuffer
    amp_data: AmpData

    def __init__(
        self,
        *args,
        discriminator: AMPDiscriminatorBase,
        amp_storage: AMPReplayBuffer,
        amp_data: AmpData,
        amp_gate_fn=None,
        amp_interleaved_update: bool = False,
        amp_metrics: dict[str, dict] | None = None,
        amp_use_separate_optimizer: bool = False,
        amp_disc_optimizer: str = "adam",
        amp_disc_learning_rate: float = 1e-3,
        amp_grad_pen_lambda: float = 5.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # super().__init__() (PPO) created self.optimizer over actor+critic.

        self.discriminator = discriminator.to(self.device)
        self.amp_storage = amp_storage
        self.amp_data = amp_data
        self.amp_gate_fn = amp_gate_fn
        self.amp_interleaved_update = amp_interleaved_update
        self.amp_grad_pen_lambda = amp_grad_pen_lambda

        # Sequential mode trains the discriminator in a separate pass, so it
        # needs its own optimizer. A combined optimizer would double-step
        # actor/critic (zero-grad disc pass + real PPO pass).
        assert amp_interleaved_update or amp_use_separate_optimizer, (
            "Sequential AMP update (amp_interleaved_update=False) requires amp_use_separate_optimizer=True."
        )

        self.init_algorithm_metric_buffers(self.storage.num_envs, self.device)
        assert amp_metrics is not None, "amp_metrics must be provided (built in construct_algorithm)"
        for name, metric_cfg in amp_metrics.items():
            self.register_episode_metric(name, **metric_cfg)

        # The discriminator owns its parameter groups (and per-group weight
        # decay). Sequential -> a dedicated optimizer; interleaved -> fold the
        # disc groups into the PPO optimizer so one backward updates all.
        if amp_use_separate_optimizer:
            self.disc_optimizer = resolve_optimizer(amp_disc_optimizer)(
                self.discriminator.param_groups(), lr=amp_disc_learning_rate
            )
        else:
            ppo_optimizer_cls = type(self.optimizer)
            self.optimizer = ppo_optimizer_cls(
                [
                    {"params": self.actor.parameters(), "name": "actor"},
                    {"params": self.critic.parameters(), "name": "critic"},
                    *self.discriminator.param_groups(),
                ],
                lr=self.learning_rate,
            )
            self.disc_optimizer = None

    # -- rollout hooks --------------------------------------------------------

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        amp_obs = obs.get(AMP_OBS_KEY)
        if amp_obs is not None:
            amp_group = obs.get(AMP_GROUP_KEY)
            # Under auto_reset, done envs return post-reset obs instead of
            # terminal obs. Use the captured terminal snapshot (delay+history
            # applied, noise-free) to avoid cross-episode AMP frames.
            # mjlab's capture_terminal_observations option fills extras["terminal"].
            terminal = extras.get(TERMINAL_KEY)
            if terminal is not None:
                terminal_env_ids = terminal[TERMINAL_ENV_IDS]
                terminal_obs = terminal[TERMINAL_OBS]
                terminal_amp = terminal_obs.get(AMP_OBS_KEY)
                if terminal_amp is not None:
                    amp_obs = amp_obs.clone()
                    amp_obs[terminal_env_ids] = terminal_amp
                if amp_group is not None:
                    terminal_group = terminal_obs.get(AMP_GROUP_KEY)
                    if terminal_group is not None:
                        amp_group = amp_group.clone()
                        amp_group[terminal_env_ids] = terminal_group

            group_idx = None if amp_group is None else amp_group.reshape(-1).long()
            self.amp_storage.insert(amp_obs, group_idx)
            blended_reward, _, disc_reward = self.discriminator.predict_reward(amp_obs, rewards, group_idx)
            if self.amp_gate_fn is not None:
                # Gate the AMP reward on the SAME state the reward was computed
                # on: for done envs use the terminal snapshot, not the post-reset
                # obs, so the gate decision matches the terminal transition it
                # gates (consistent with the disc's terminal-patched amp_obs).
                mask = self.amp_gate_fn(_patch_terminal_obs(obs, terminal))
                rewards = torch.where(mask, blended_reward, rewards)
                disc_reward = torch.where(mask, disc_reward, torch.zeros_like(disc_reward))
            else:
                rewards = blended_reward
            self.log_episode_metric(extras, dones, "disc_reward", disc_reward)
        super().process_env_step(obs, rewards, dones, extras)

    # -- discriminator / PPO updates -----------------------------------------

    def _amp_minibatch_size(self) -> int:
        return self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches

    def _train_discriminator(self) -> dict[str, float]:
        """Sequential disc update: a separate pass over replay+expert samples.

        The discriminator owns normalization, expert sampling (from
        ``amp_data``), the LSGAN loss, the gradient penalty, and the normalizer
        update -- this method just samples policy windows, backprops, and steps.
        """
        mini_batch_size = self._amp_minibatch_size()
        num_total_batches = self.num_learning_epochs * self.num_mini_batches

        # Aggregate every key the discriminator returns (the aggregate amp/* keys
        # plus any per-group group_*/ keys) so per-disc_group metrics flow to
        # logging without the algorithm knowing the group vocabulary.
        sums: dict[str, float] = {}
        for _ in range(num_total_batches):
            policy_states, policy_groups = self.amp_storage.sample(
                mini_batch_size, balanced=self.discriminator.balanced_replay
            )
            loss, metrics = self.discriminator.disc_loss(
                policy_states, policy_groups, self.amp_data, self.amp_grad_pen_lambda
            )
            self.disc_optimizer.zero_grad()
            loss.backward()
            self.disc_optimizer.step()
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + v

        return {k: v / num_total_batches for k, v in sums.items()}

    def _update_interleaved(self) -> dict[str, float]:
        """Interleaved update: disc + PPO share the same mini-batch loop."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None
        # Aggregate every disc metric key (aggregate amp/* + per-group group_*/).
        disc_sums: dict[str, float] = {}

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        mini_batch_size = self._amp_minibatch_size()

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[1],
            )
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            policy_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None

            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    policy_loss = policy_loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            # AMP discriminator loss for this mini-batch. The discriminator
            # samples matched experts from amp_data and returns the combined
            # (LSGAN + grad-pen) loss; we backprop/step it here.
            policy_states, policy_groups = self.amp_storage.sample(
                mini_batch_size, balanced=self.discriminator.balanced_replay
            )
            amp_loss, disc_metrics = self.discriminator.disc_loss(
                policy_states, policy_groups, self.amp_data, self.amp_grad_pen_lambda
            )

            if self.disc_optimizer is not None:
                self.optimizer.zero_grad()
                policy_loss.backward()
                if self.rnd:
                    self.rnd.optimizer.zero_grad()
                    rnd_loss.backward()
                if self.is_multi_gpu:
                    self.reduce_parameters()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if self.rnd:
                    self.rnd.optimizer.step()

                self.disc_optimizer.zero_grad()
                amp_loss.backward()
                self.disc_optimizer.step()
            else:
                loss = policy_loss + amp_loss
                self.optimizer.zero_grad()
                loss.backward()
                if self.rnd:
                    self.rnd.optimizer.zero_grad()
                    rnd_loss.backward()
                if self.is_multi_gpu:
                    self.reduce_parameters()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if self.rnd:
                    self.rnd.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            for k, v in disc_metrics.items():
                disc_sums[k] = disc_sums.get(k, 0.0) + v

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            **{k: v / num_updates for k, v in disc_sums.items()},
        }
        if mean_rnd_loss is not None:
            loss_dict["rnd"] = mean_rnd_loss / num_updates
        if mean_symmetry_loss is not None:
            loss_dict["symmetry"] = mean_symmetry_loss / num_updates
        return loss_dict

    def update(self) -> dict[str, float]:
        if self.amp_interleaved_update:
            return self._update_interleaved()
        disc_metrics = self._train_discriminator()
        loss_dict = super().update()
        loss_dict.update(disc_metrics)
        return loss_dict

    def train_mode(self) -> None:
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.discriminator.eval()

    # -- checkpoint / multi-gpu ----------------------------------------------

    def save(self) -> dict:
        saved = super().save()
        # The discriminator state_dict bundles its normalizer via extra_state.
        saved["discriminator_state_dict"] = self.discriminator.state_dict()
        if self.disc_optimizer is not None:
            saved["disc_optimizer_state_dict"] = self.disc_optimizer.state_dict()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        result = super().load(loaded_dict, load_cfg, strict)
        # Respect load_cfg like PPO does for its keys: "discriminator" gates the
        # disc weights (default True), "disc_optimizer" gates its optimizer state
        # and follows "optimizer" unless keyed explicitly -- so a weights-only /
        # fork-style load_cfg ({"optimizer": False, ...}) skips it, and
        # {"discriminator": False} forks with a fresh discriminator.
        cfg = load_cfg if load_cfg is not None else {}
        if cfg.get("discriminator", True) and "discriminator_state_dict" in loaded_dict:
            self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"], strict=strict)
        load_disc_optimizer = cfg.get("disc_optimizer", cfg.get("optimizer", True))
        if load_disc_optimizer and self.disc_optimizer is not None and "disc_optimizer_state_dict" in loaded_dict:
            self.disc_optimizer.load_state_dict(loaded_dict["disc_optimizer_state_dict"])
        return result

    def broadcast_parameters(self) -> None:
        super().broadcast_parameters()
        model_params = [self.discriminator.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.discriminator.load_state_dict(model_params[0])

    # -- construction ---------------------------------------------------------

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> AMP_PPO:
        # Work on shallow copies of the dicts we destructively .pop() from, so the
        # caller's cfg stays intact -- the runner logs it to wandb, and popping
        # `amp` here would otherwise strip the discriminator + blend knobs
        # (use_lerp / task_reward_lerp / disc_mode / ...) from the
        # logged config (only disc_mode would survive, via env_cfg interpolation).
        # We only pop top-level keys and reassign `algorithm`, never mutate nested
        # values, so a shallow copy is enough. The runtime-only amp["data"] tensors
        # that ride along are dropped at the log boundary (WandbLogWriter).
        cfg = {**cfg, "algorithm": {**cfg["algorithm"]}, "actor": {**cfg["actor"]}, "critic": {**cfg["critic"]}}
        alg_class: type[AMP_PPO] = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))

        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        amp = cfg["algorithm"].pop("amp")

        # Ground the AMP feature shape from the obs group (like actor/critic),
        # and the number of discriminator groups from amp_data -- no env-derived
        # fields travel through amp_cfg.
        amp_obs = obs[AMP_OBS_KEY]
        num_frames = amp_obs.shape[1]
        feature_dim = amp_obs.shape[2]
        amp_data: AmpData = amp["data"]
        num_disc_groups = amp_data.num_disc_groups
        disc_mode = amp.get("disc_mode", "single")

        # Default metric: episode-summed disc_reward divided by max episode
        # length. Always logged as Episode_Reward/AMP/disc_reward (per-iter
        # mean); rolling=True additionally logs Train/mean_disc_reward with
        # the logger's maxlen=100 rolling window.
        amp_metrics = amp.get(
            "metrics",
            {
                "disc_reward": {
                    "reduce": "constant",
                    "divisor": float(env.max_episode_length),
                    "prefix": "Episode_Reward/AMP",
                    "rolling": True,
                }
            },
        )

        discriminator = _build_discriminator(
            disc_mode,
            feature_dim,
            num_frames,
            num_disc_groups,
            device,
            amp["discriminator"],
        )
        amp_replay_buffer = AMPReplayBuffer(
            feature_dim, amp["replay_buffer_size"], num_frames, device, num_disc_groups=num_disc_groups
        )

        alg: AMP_PPO = alg_class(
            actor,
            critic,
            storage,
            discriminator=discriminator,
            amp_storage=amp_replay_buffer,
            amp_data=amp_data,
            amp_gate_fn=amp.get("gate_fn"),
            amp_interleaved_update=amp["interleaved_update"],
            amp_metrics=amp_metrics,
            amp_use_separate_optimizer=amp["amp_use_separate_optimizer"],
            amp_grad_pen_lambda=amp.get("grad_pen_lambda", 5.0),
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
            **amp["disc_optimizer"],
        )

        alg.compile(cfg.get("torch_compile_mode"))
        return alg


class AMP_PPO(AMPMixin, PPO):
    """PPO with AMP discriminator reward."""

    pass
