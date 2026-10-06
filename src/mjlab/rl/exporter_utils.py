"""Shared utilities for ONNX policy export across RL tasks."""

import json

import numpy as np
import onnx
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction


def list_to_csv_str(arr, *, decimals: int | None = None, delimiter: str = ",") -> str:
  """Convert list to CSV string.

  ``decimals=None`` (the default) writes each float at full round-trip
  precision, so reading the value back gives the identical float. Pass an int
  to round instead.

  Rounding would make the metadata unusable as a source: at 3 decimals a deploy
  runtime reading ``action_scale`` gets 0.075 where the policy was trained at
  0.0745, a 0.67 percent error on the wrist channel, so every consumer would
  keep its own copy of the real numbers and treat the metadata as a checksum.
  Rounding a value that a robot will act on is not a formatting choice.
  """

  def render(x) -> str:
    if not isinstance(x, (int, float)) or isinstance(x, bool):
      return str(x)  # strings pass through as-is
    if decimals is not None:
      # Explicit rounding formats every number alike, ints included, which is
      # the behaviour callers passing `decimals` already rely on.
      return f"{{:.{decimals}f}}".format(x)
    # Full precision: ints stay ints. A count or a width written as "10.0"
    # makes every reader parse a float and round it back, for no gain.
    return str(x) if isinstance(x, int) else repr(x)

  return delimiter.join(render(x) for x in arr)


# What JSON can carry, and what a deploy runtime can act on.
_PLAIN_TYPES = (str, bool, int, float, type(None))


def _plain_params(params: dict) -> dict:
  """A term's params reduced to the ones that describe its contract.

  Kept: strings, numbers, bools, None, and flat lists/tuples of those, which
  is how a term states its frame and scope (``body_names``,
  ``base_body_name``, ``ref_face``, ``body_set``). Dropped: anything else,
  because a callable, a tensor or a config object describes the training env
  rather than the contract and could not be written out anyway.

  ``command_name`` is dropped too: it points at a command term that exists
  only inside the training env, so it tells a consumer nothing.
  """
  out = {}
  for key, value in params.items():
    if key == "command_name":
      continue
    if isinstance(value, _PLAIN_TYPES):
      out[key] = value
    elif isinstance(value, (list, tuple)) and all(
      isinstance(v, _PLAIN_TYPES) for v in value
    ):
      out[key] = list(value)
  return out


def get_base_metadata(
  env: ManagerBasedRlEnv, run_path: str
) -> dict[str, list | str | float | int]:
  """Get base metadata common to all RL policy exports.

  Args:
    env: The RL environment.
    run_path: W&B run path or other identifier.

  Returns:
    Dictionary of metadata fields that are common across all tasks.
  """
  robot: Entity = env.scene["robot"]
  joint_action = env.action_manager.get_term("joint_pos")
  assert isinstance(joint_action, JointPositionAction)
  # Build mapping from joint name to actuator ID for natural joint order.
  # Each spec actuator controls exactly one joint (via its target field).
  joint_name_to_ctrl_id = {}
  for actuator in robot.spec.actuators:
    joint_name = actuator.target.split("/")[-1]
    joint_name_to_ctrl_id[joint_name] = actuator.id
  # Get actuator IDs in natural joint order (same order as robot.joint_names).
  ctrl_ids_natural = [
    joint_name_to_ctrl_id[jname]
    for jname in robot.joint_names  # global joint order
    if jname in joint_name_to_ctrl_id  # skip non-actuated joints
  ]
  mj_model = env.sim.mj_model
  joint_stiffness = mj_model.actuator_gainprm[ctrl_ids_natural, 0]
  joint_damping = -mj_model.actuator_biasprm[ctrl_ids_natural, 2]
  # Torque ceiling and joint range, both of which a deploy runtime needs from
  # the artifact. The ranges are indexed by JOINT id, not actuator id, so they
  # are looked up separately.
  joint_torque_limits = mj_model.actuator_forcerange[ctrl_ids_natural, 1]
  joint_ids_natural = [mj_model.actuator_trnid[a][0] for a in ctrl_ids_natural]
  joint_ranges = mj_model.jnt_range[joint_ids_natural]

  observation_names = env.observation_manager.active_terms["actor"]
  # Flattened width per term, history included, in the same order as the
  # names. With these a consumer can slice the vector directly instead of
  # rebuilding the widths from its own table of what each term ought to be.
  observation_dims = [
    int(np.prod(dim)) for dim in env.observation_manager.group_obs_term_dim["actor"]
  ]
  # Per TERM, not one number for the group. The group config's history_length
  # is optional (``None`` means every term keeps its own), so a single scalar
  # cannot describe every valid config, and the one case it silently loses is
  # exactly the one a consumer cannot recover: with mixed depths the flattened
  # widths alone do not say how each term was stacked. Read off the RESOLVED
  # term cfgs, so it is right whichever way the config expressed it.
  # 0 means "no history" in the config and 1 frame in the vector, so it is
  # reported as the multiplier a consumer actually needs.
  history_lengths = [
    max(int(env.observation_manager.get_term_cfg("actor", n).history_length or 0), 1)
    for n in observation_names
  ]
  # Each term's frame and scope, same order as the names, as one JSON string.
  # The names alone do not carry it: two configs whose partner observation
  # differs only in the robot link it is expressed in export the identical obs
  # key, so a deploy runtime reading only the names has to guess between the
  # torso and the pelvis, and a wrong guess is a silently
  # rotated observation rather than an error.
  observation_params = json.dumps(
    [
      _plain_params(env.observation_manager.get_term_cfg("actor", n).params)
      for n in observation_names
    ]
  )

  return {
    "run_path": run_path,
    "joint_names": list(robot.joint_names),
    # Body order as the motion NPZs index it: MJCF body order minus the world
    # body. Those files carry no name array, so without this a consumer has to
    # hardcode positions, and a wrong one is undetectable: the runtime's
    # 14-name tracked subset puts torso at 7, which lands on a hip. This is
    # also the one structural fact NOT recoverable from the resolved config,
    # since the axis only exists after the MJCF is compiled.
    "body_names": list(robot.body_names),
    "joint_stiffness": joint_stiffness.tolist(),
    "joint_damping": joint_damping.tolist(),
    "joint_torque_limits": joint_torque_limits.tolist(),
    "joint_lower_limits": joint_ranges[:, 0].tolist(),
    "joint_upper_limits": joint_ranges[:, 1].tolist(),
    "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
    "command_names": list(env.command_manager.active_terms),
    "observation_names": observation_names,
    "observation_dims": observation_dims,
    "observation_history_lengths": history_lengths,
    "observation_params": observation_params,
    # The rate the policy was trained to run at, so a deploy runtime need not
    # assume one (an assumed control rate is wrong silently).
    "control_dt": env.step_dt,
    "control_hz": 1.0 / env.step_dt,
    "physics_dt": env.physics_dt,
    "action_scale": joint_action._scale[0].cpu().tolist()
    if isinstance(joint_action._scale, torch.Tensor)
    else joint_action._scale,
  }


def attach_metadata_to_onnx(
  onnx_path: str, metadata: dict[str, list | str | float | int]
) -> None:
  """Attach metadata to an ONNX model file.

  Args:
    onnx_path: Path to the ONNX model file.
    metadata: Dictionary of metadata key-value pairs to attach.
  """
  model = onnx.load(onnx_path)

  for k, v in metadata.items():
    entry = onnx.StringStringEntryProto()
    entry.key = k
    entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
    model.metadata_props.append(entry)

  onnx.save(model, onnx_path)
