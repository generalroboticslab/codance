import os
import shutil
from typing import Any

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)


def save_iter_snapshot(onnx_path: str, iter_num: int) -> str:
  """Copy <run>.onnx to <dir>/model_{iter}.onnx, preserving history per save.

  The latest ONNX keeps its canonical path (overwritten each save for
  downstream tooling); the iter-stamped copy mirrors the `model_{iter}.pt`
  checkpoint naming so they line up one-to-one on disk.
  """
  iter_path = os.path.join(os.path.dirname(onnx_path), f"model_{iter_num}.onnx")
  shutil.copy2(onnx_path, iter_path)
  return iter_path


def _contact_slot_names(env: Any) -> list[str] | None:
  """The multi-link contact link roster, or None when the run has no force field.

  ``env`` is anything exposing ``command_manager``, the same latitude
  ``get_codancing_command`` takes, because the lookup below reads nothing
  else and a wrapper is as good a source as the raw env.

  This is the order every slot-major compliance channel packs, including the
  ``desired_stiffness`` observation the actor reads. It comes from the contact
  files the run trained against, so it is a property of the DATA release, not
  of the code: a later release can reorder or resize the slots, and the
  training side already refuses to infer a side from a slot position for that
  reason. Without it in the artifact a deploy runtime has to assume the order,
  and assuming wrong sends a per-hand stiffness to the other wrist, which
  looks entirely normal on the robot.

  Not part of ``get_base_metadata``: the force field is a codancing concept
  and the other tasks sharing that helper have no contact provider.
  """
  for name in env.command_manager.active_terms:
    provider = getattr(env.command_manager.get_term(name), "_contact_provider", None)
    names = getattr(provider, "_slot_link_names", None)
    if names:
      return [str(n) for n in names]
  return None


def attach_onnx_metadata(
  env: ManagerBasedRlEnv, run_path: str, path: str, filename="policy.onnx"
) -> None:
  """Attach codancing metadata to ONNX model.

  Args:
    env: The RL environment.
    run_path: W&B run path or other identifier.
    path: Directory containing the ONNX file.
    filename: Name of the ONNX file.
  """
  onnx_path = os.path.join(path, filename)
  metadata = get_base_metadata(env, run_path)
  slot_names = _contact_slot_names(env)
  if slot_names is not None:
    metadata["slot_link_names"] = slot_names
  attach_metadata_to_onnx(onnx_path, metadata)
