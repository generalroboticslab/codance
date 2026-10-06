"""Extended RL runner configurations for AMP."""

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl.config import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass(kw_only=True)
class RslRlAmpDiscriminatorCfg:
  """The AMP discriminator's own knobs (``algorithm.amp.discriminator``).

  Mirrors ``AMPDiscriminator.__init__`` (the ``single`` disc_mode) exactly --
  ``AMP_PPO.construct_algorithm`` spreads this dict into the discriminator
  (``**disc_cfg``). The feature width / frame count are grounded from
  ``obs["amp"]`` at construct, so only the design knobs live here.

  The rsl_rl class actually built is chosen by ``disc_mode`` in
  ``_build_discriminator`` (single, multi_head or separate), so this cfg only
  supplies the typed knobs and defaults for every mode. The run files state the
  ``balanced_replay`` they use, so this default is never silently the source of
  behavior (hydra spreads whatever the yaml carries).
  """

  hidden_layer_sizes: tuple[int, ...] = (256, 256)
  amp_disc_reward_coef: float = 2.0
  task_reward_lerp: float = 0.5
  use_lerp: bool = True
  trunk_weight_decay: float = 1e-3
  head_weight_decay: float = 1e-1
  # rsl_rl's per-mode default: single/multi_head=False, separate=True.
  balanced_replay: bool = False


@dataclass(kw_only=True)
class RslRlAmpDiscOptimizerCfg:
  """The AMP discriminator optimizer (``algorithm.amp.disc_optimizer``).

  Spread into ``AMPMixin.__init__`` (``**amp["disc_optimizer"]``); only consulted
  when ``amp_use_separate_optimizer`` is set (otherwise the discriminator's
  param groups fold into the PPO optimizer).
  """

  amp_disc_optimizer: str = "adam"
  amp_disc_learning_rate: float = 1e-3


@dataclass(kw_only=True)
class RslRlAmpCfg:
  """The ``algorithm.amp`` entrypoint sub-config, mirroring what
  ``AMP_PPO.construct_algorithm`` consumes (rsl_rl ``third_party/rsl_rl``).

  ``data`` (the env-derived expert pool) and ``metrics`` are NOT modeled here:
  they are attached at construct time (``CodancingAMPRunner._build_amp_train_cfg``
  builds ``amp.data`` from the env's loaded reference motions; ``metrics``
  defaults inside ``construct_algorithm`` using ``env.max_episode_length``).
  ``num_frames`` is the co-design window length read by the ENV (its
  ``amp_feature`` + the ``amp`` obs history); rsl_rl re-grounds it from
  ``obs["amp"]`` so the two agree by construction.

  Two members of ``construct_algorithm``'s ``amp`` dict are deliberately NOT
  modeled here:
    - ``gate_fn`` -- the AMP transition-gate hook (``amp.get("gate_fn")``,
      amp_ppo.py); a *callable* (zeroes the disc reward / skips replay near a
      group switch), with no YAML form.
    - ``metrics`` -- a derived default built inside ``construct_algorithm`` from
      ``env.max_episode_length`` (not a tunable knob).
  """

  num_frames: int = 5
  # Per-motion discriminator mode -- the rsl_rl `DiscMode` vocabulary
  # (amp_contract.py): "single" (one pool over all motions), "multi_head" (one
  # discriminator with one head per motion), "separate" (independent
  # discriminators). The env's expert grouping is derived from this same value
  # (single -> 1 group, multi_head/separate -> one group per motion).
  disc_mode: Literal["single", "multi_head", "separate"] = "single"
  discriminator: RslRlAmpDiscriminatorCfg = field(
    default_factory=RslRlAmpDiscriminatorCfg
  )
  disc_optimizer: RslRlAmpDiscOptimizerCfg = field(
    default_factory=RslRlAmpDiscOptimizerCfg
  )
  replay_buffer_size: int = 100000
  grad_pen_lambda: float = 5.0
  amp_use_separate_optimizer: bool = False
  interleaved_update: bool = True

  def __post_init__(self) -> None:
    # disc_mode is the rsl_rl DiscMode vocabulary; a YAML bool/null (e.g.
    # `disc_mode: false`) is a config error. Validate here so
    # it dies at compose (this runs when _build_runner instantiates the cfg),
    # not at construct.
    if self.disc_mode not in ("single", "multi_head", "separate"):
      raise ValueError(
        f"AMP disc_mode must be one of ('single', 'multi_head', 'separate'), "
        f"got {self.disc_mode!r}. YAML bools / null are not accepted -- use "
        "the string 'single' for one shared discriminator."
      )
    # rsl_rl's AMPMixin.__init__ asserts the same invariant at construct: a
    # sequential update (not interleaved) needs its own optimizer, else the
    # discriminator would never step. Catch it at compose so a bad config dies
    # before the env is built.
    if not (self.interleaved_update or self.amp_use_separate_optimizer):
      raise ValueError(
        "AMP: a sequential update (interleaved_update=false) requires "
        "amp_use_separate_optimizer=true (otherwise the discriminator never "
        "steps). Set one of them true."
      )


@dataclass(kw_only=True)
class RslRlAmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO algorithm cfg with the typed AMP entrypoint sub-config.

  ``class_name`` is the AMP class, and ``amp`` is the typed ``algorithm.amp``
  tree ``AMP_PPO.construct_algorithm`` reads -- the run file is the source of
  truth for the design knobs (env-derived ``amp.data`` is attached at
  construct).
  """

  class_name: str = "rsl_rl.algorithms.amp_ppo:AMP_PPO"
  amp: RslRlAmpCfg = field(default_factory=RslRlAmpCfg)


@dataclass(kw_only=True)
class RslRlAMPOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner configuration for AMPOnPolicyRunner.

  The AMP design knobs live in the typed ``algorithm.amp`` sub-config (aligned
  with rsl_rl); the runner adds nothing beyond the base PPO runner cfg plus the
  AMP algorithm class.
  """

  class_name: str = "AMPOnPolicyRunner"
  algorithm: RslRlAmpPpoAlgorithmCfg = field(default_factory=RslRlAmpPpoAlgorithmCfg)
  load: dict[str, dict[str, bool]] = field(
    default_factory=lambda: {
      "session": {
        "actor": True,
        "critic": True,
        "rnd": True,
        "discriminator": False,
        "disc_optimizer": False,
        "optimizer": False,
        "iteration": False,
      },
      "fork": {
        "actor": True,
        "critic": True,
        "rnd": True,
        "discriminator": True,
        "disc_optimizer": False,
        "optimizer": False,
        "iteration": False,
      },
    }
  )
  """AMP load table: adds the discriminator pair to the PPO vocabulary.
  Session/eval is actor-only inference (``discriminator: False``); a fork keeps
  the trained discriminator (``runner.load.fork.discriminator=false`` forks a
  fresh one)."""
