import copy
import os
from typing import cast

import wandb
from rsl_rl.algorithms.amp_contract import AMP_OBS_KEY
from rsl_rl.env import VecEnv

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.codancing.rl.amp_data_builder import build_amp_data
from mjlab.tasks.codancing.rl.exporter import (
  attach_onnx_metadata,
  save_iter_snapshot,
)


def _save_onnx_and_wandb(runner, path: str) -> None:
  """Shared ONNX export and wandb upload logic for codancing runners."""
  policy_path = path.split("model")[0]
  filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
  # The export goes through the policy's own `as_onnx()` graph (obs
  # normalization baked in), driven by MjlabOnPolicyRunner.export_policy_to_onnx.
  runner.export_policy_to_onnx(path=policy_path, filename=filename)
  run_name = "local"
  if runner.logger.logger_type in ("wandb", "WandbLogWriter"):
    wb_run = wandb.run
    if wb_run is not None and wb_run.name is not None:
      run_name = wb_run.name
  attach_onnx_metadata(
    cast(RslRlVecEnvWrapper, runner.env).unwrapped,
    run_name,
    path=policy_path,
    filename=filename,
  )
  save_iter_snapshot(policy_path + filename, runner.current_learning_iteration)
  # The export itself always happens (the onnx is a run-dir artifact); only the
  # wandb copy is gated, same knob as the .pt uploads in MjlabOnPolicyRunner.
  if runner.logger.logger_type in ("wandb", "WandbLogWriter") and runner.cfg.get(
    "upload_model"
  ):
    wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class CodancingOnPolicyRunner(MjlabOnPolicyRunner):
  """PPO codancing runner.

  Inherits env-state persistence and ONNX export from ``MjlabOnPolicyRunner``;
  the ``save()`` override adds codancing ONNX metadata, the iter-stamped
  snapshot, and the W&B upload.
  """

  # The frozen resolved-config tree (set by the executor before `learn()`), so
  # every checkpoint carries the config it was trained with (freeze contract).
  resolved_cfg_yaml: str | None = None
  # The dedicated eval env's frozen tree (set for every train run), embedded
  # alongside so resume / offline `run eval` reproduce it exactly.
  resolved_eval_cfg_yaml: str | None = None
  # The run's STABLE identity (set by the executor before `learn()`), embedded
  # into every checkpoint's `infos['run_meta']` so a `.pt` self-describes which
  # run / commit / host produced it. Merged with the live wandb fields at save.
  run_meta: dict | None = None
  # Train-invocation provenance (set by the executor): the session block the
  # freeze strips, plus the launch CLI -- baked into `infos` as a record of the
  # launch only; frozen loads still graft a fresh session.
  train_session_yaml: str | None = None
  train_command: list[str] | None = None

  def save(self, path: str, infos=None):
    from mjlab.tasks.codancing.config.freeze import (
      embed_resolved_cfg,
      embed_run_meta,
      embed_train_provenance,
    )
    from mjlab.tasks.shared.utils.wandb_runtime import current_run_meta

    infos = embed_resolved_cfg(
      infos, self.resolved_cfg_yaml, self.resolved_eval_cfg_yaml
    )
    infos = embed_train_provenance(infos, self.train_session_yaml, self.train_command)
    # Also stamp the run's stable identity (the same set meta.json records) plus
    # the live wandb fields, so the checkpoint names the run it came from without
    # its sibling meta.json. wandb.run exists by save time (rank 0 only).
    if self.run_meta is not None:
      infos = embed_run_meta(infos, current_run_meta(self.run_meta))
    super().save(path, infos)
    _save_onnx_and_wandb(self, path)


def _build_amp_train_cfg(env: VecEnv, train_cfg: dict, device: str) -> dict:
  """Attach the env-derived expert pool to the typed ``algorithm.amp`` sub-cfg.

  The AMP design knobs (``discriminator`` / ``disc_optimizer`` / ``disc_mode`` /
  ``replay_buffer_size`` / ``grad_pen_lambda`` / ``interleaved_update`` /
  ``amp_use_separate_optimizer``) are set in the run file (e.g.
  conf/decoupled.yaml) and arrive in ``train_cfg["algorithm"]["amp"]`` already
  typed (``RslRlAmpCfg``). The only construct-time piece is ``amp["data"]`` -- the expert windows built from the
  command's already-loaded reference motions. ``feature_dim`` / ``num_frames`` /
  ``num_disc_groups`` are grounded by rsl_rl from ``obs["amp"]`` + ``amp_data``,
  and ``metrics`` defaults inside ``construct_algorithm`` (env-derived divisor).
  """
  obs_keys = {str(key) for key in env.get_observations().keys()}
  if AMP_OBS_KEY not in obs_keys:
    raise ValueError(
      f"AMP runner requires the {AMP_OBS_KEY!r} observation group, but the "
      f"composed env only has {sorted(obs_keys)}: a non-AMP env cfg was paired "
      "with the AMP runner. Declare env.observations.amp (and amp_group) in the "
      "run file, as conf/decoupled.yaml does."
    )
  mjlab_env = cast(RslRlVecEnvWrapper, env).unwrapped
  amp_data = build_amp_data(mjlab_env, device=device)

  cfg = copy.deepcopy(train_cfg)
  cfg["algorithm"]["amp"]["data"] = amp_data
  return cfg


class CodancingAMPRunner(CodancingOnPolicyRunner):
  """AMP runner: builds the task-blind ``AMP_PPO`` via ``construct_algorithm``.

  ``amp_data`` is built from the codancing command's already-loaded reference
  motions and attached to the algorithm's typed ``amp`` sub-config; the
  algorithm class is ``AMP_PPO`` (rsl_rl resolves it and grounds
  ``feature_dim`` / ``num_frames`` / ``num_disc_groups`` from ``obs["amp"]`` +
  ``amp_data``). All AMP logic lives in the algorithm
  (``AMPMixin.process_env_step`` reads ``obs["amp"]`` / ``obs["amp_group"]``), so
  the stock ``OnPolicyRunner.learn`` loop runs unmodified. ONNX export + state
  persistence are inherited from ``CodancingOnPolicyRunner``.
  """

  def __init__(
    self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"
  ):
    train_cfg = _build_amp_train_cfg(env, train_cfg, device)
    super().__init__(env, train_cfg, log_dir, device)
