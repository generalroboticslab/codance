"""The G1's joint order, the compliance contact slots, the deploy deviation
bands, the robot NIC check, and the policy artifact's metadata.

The per-joint control numbers (gains, action scale, default pose, joint
ranges) are not here: every exported policy carries them in its ONNX metadata
at full precision, and :func:`control_table` reads them from there.

Nothing here imports mjlab, torch, or mujoco.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NUM_JOINTS = 29

# The rate every deploy loop here runs at. Training states its own in the
# artifact (``control_dt``), and the layout resolvers check it against this
# rather than assuming, because an assumed control rate is wrong silently:
# the policy still infers, the targets are still in range, and only the
# closed-loop behaviour is off.
CONTROL_DT = 0.02  # 50 Hz policy rate; the firmware plays the 200 Hz servo.

REPO_ROOT = Path(__file__).resolve().parents[5]

# The free face of the solo-stand reference: the pose the actor observes. The
# stand pool's manifest entries all point their ``free_motion_file`` here.
STAND_REFERENCE = REPO_ROOT / "data/reference_motion_stand/20260814_001_robot.npz"

# Natural model order. Identical for actions and every joint-indexed
# observation, and the SDK motor index map for the 29-joint G1 is the identity.
JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

# Index of the three waist joints, which carry the pelvis-to-torso rotation.
WAIST_YAW = JOINT_NAMES.index("waist_yaw_joint")
WAIST_ROLL = JOINT_NAMES.index("waist_roll_joint")
WAIST_PITCH = JOINT_NAMES.index("waist_pitch_joint")

# The two compliance contact slots, in the order every slot-major channel packs
# them: left_wrist_yaw_link then right_wrist_yaw_link. Per slot, the wrist
# joints are where a push at the hand is measured, and the arm chain is what
# yields when the policy goes along with it. Wrist pitch and yaw are limited to
# 5 N.m, so they will not fight a human grip; the compliance worth watching is
# the arm and the body giving way, not the wrist going slack.
CONTACT_SLOTS: tuple[str, ...] = ("L", "R")
_SIDE = {"L": "left", "R": "right"}


def _joints_matching(pattern: str) -> tuple[int, ...]:
  """Indices of the joints whose names match, in natural joint order.

  Derived from JOINT_NAMES rather than typed out, so the positions cannot
  drift from the names beside them. Hand-written indices are the failure this
  avoids: a wrong one makes the compliance readout name the wrong joint and
  the waist chain compose the wrong angles, and neither says anything.
  """
  return tuple(i for i, n in enumerate(JOINT_NAMES) if re.fullmatch(pattern, n))


WRIST_JOINTS: dict[str, tuple[int, ...]] = {
  slot: _joints_matching(rf"{side}_wrist_.*") for slot, side in _SIDE.items()
}
ARM_JOINTS: dict[str, tuple[int, ...]] = {
  slot: _joints_matching(rf"{side}_(shoulder|elbow|wrist)_.*")
  for slot, side in _SIDE.items()
}

# How far a joint may sit from the FREE reference before the deploy
# joint-deviation check counts it toward an abort. `inf` exempts it outright.
#
# Sized from |q_adapted - q_free| over the compliance pools, because a push
# hard enough to get the yield the policy was trained to give moves joints
# that far, and damping out right then defeats the demonstration. On the
# released pools these bands trip on 0.04 percent of the standing frames and
# on none of the waltz frames; one 1.2 rad band for every joint would trip on
# 1.66 and 0.10 percent.
#
# These are sized against DATA, so a pool that widens the pose distribution
# can invalidate them.
#
# The wrists are EXEMPT rather than banded, and the reason differs by runner:
#   * standing: the free reference IS the default keyframe, so a wrist's
#     deviation is bounded by its own travel from default (1.97 rad roll,
#     1.61 pitch and yaw), and the standing pool reaches 1.97. Every band that
#     clears the pool is one no wrist can reach, so a number here would be an
#     exemption dressed up as a threshold.
#   * dancing: the reference moves, so the bound is the full joint range
#     (3.94 rad). Exempting them is a deliberate choice, and the waltz pool
#     backs it: with the wrists out, no frame trips at all.
# Either way this is a FALL detector, and a hand held at its stop by the
# person the robot was handed to is the demonstration, not a fall. Tilt,
# joint velocity, clamp streak and staleness are what catch a real fall.
_JOINT_DEV_BANDS: dict[str, float] = {
  r".*_wrist_.*": np.inf,
  r".*_(shoulder|elbow)_.*": 2.1,
  r".*_(hip|knee|ankle)_.*": 1.2,
  r"waist_.*": 1.2,
}


def _band_vector(bands: dict[str, float]) -> np.ndarray:
  """One band per joint, in natural joint order.

  Built from the patterns rather than typed out for the same reason
  :func:`_joints_matching` exists: a band sitting against the wrong joint
  reads as a tuning choice, not as a mistake. A joint no pattern covers is a
  hard error, so adding a joint cannot silently leave it unbanded.
  """
  out = np.full(NUM_JOINTS, np.nan)
  for pattern, band in bands.items():
    out[list(_joints_matching(pattern))] = band
  missing = [n for n, b in zip(JOINT_NAMES, out, strict=True) if np.isnan(b)]
  if missing:
    raise ValueError(f"no deviation band covers {missing}")
  return out


JOINT_DEV_BAND: np.ndarray = _band_vector(_JOINT_DEV_BANDS)

# How each metadata key is parsed. An unlisted key comes back as its raw
# string, which is the failure this table exists to prevent: a numeric array
# read as text does not raise until something tries to do arithmetic on it,
# often several layers away from the typo.
_NAME_LISTS = frozenset(
  {
    "joint_names",
    "observation_names",
    "command_names",
    "body_names",
    "slot_link_names",
  }
)
_FLOAT_ARRAYS = frozenset(
  {
    "joint_stiffness",
    "joint_damping",
    "default_joint_pos",
    "action_scale",
    "joint_torque_limits",
    "joint_lower_limits",
    "joint_upper_limits",
  }
)
_INT_ARRAYS = frozenset({"observation_dims", "observation_history_lengths"})
_FLOAT_SCALARS = frozenset({"control_dt", "control_hz", "physics_dt"})


def read_onnx_metadata(path: str | Path) -> dict:
  """The artifact's ``metadata_props``, with the known arrays parsed."""
  import onnx

  model = onnx.load(str(path), load_external_data=False)
  out: dict = {}
  for prop in model.metadata_props:
    if prop.key in _NAME_LISTS:
      out[prop.key] = [v.strip() for v in prop.value.split(",")]
    elif prop.key in _FLOAT_ARRAYS:
      out[prop.key] = np.array([float(v) for v in prop.value.split(",")])
    elif prop.key in _INT_ARRAYS:
      out[prop.key] = [int(v) for v in prop.value.split(",")]
    elif prop.key in _FLOAT_SCALARS:
      out[prop.key] = float(prop.value)
    else:
      out[prop.key] = prop.value
  return out


# Every key the runners read. An export from this repository's training writes
# all of them.
REQUIRED_METADATA: tuple[str, ...] = (
  "joint_names",
  "joint_stiffness",
  "joint_damping",
  "action_scale",
  "default_joint_pos",
  "joint_lower_limits",
  "joint_upper_limits",
  "body_names",
  "slot_link_names",
  "control_dt",
  "observation_names",
  "observation_dims",
  "observation_history_lengths",
  "observation_params",
)


def policy_metadata(path: str | Path) -> dict:
  """The artifact's metadata, checked for every key the runners read and for
  per-joint arrays that agree with its own joint list."""
  meta = read_onnx_metadata(path)
  missing = [k for k in REQUIRED_METADATA if k not in meta]
  if missing:
    raise ValueError(
      f"{path} carries no {', '.join(missing)} metadata; export the policy with "
      "this repository's training, which writes every key the runners read"
    )
  names = meta["joint_names"]
  for key in ("joint_stiffness", "joint_damping", "default_joint_pos", "action_scale"):
    if len(meta[key]) != len(names):
      raise ValueError(
        f"{key} in {path} has {len(meta[key])} entries against "
        f"{len(names)} joint_names; the ONNX metadata contradicts itself"
      )
  return meta


def check_contact_slots(meta: dict, path: str | Path) -> None:
  """Confirm the artifact packs its contact slots left wrist then right.

  Every slot-major compliance channel here is ordered by :data:`CONTACT_SLOTS`:
  the four numbers of ``desired_stiffness``, and the per-hand deploy flags that
  fill them. That order comes from the contact files the run trained against,
  and the training side deliberately resolves a side by NAME rather than by
  position, because a new data release can reorder or resize the slots.

  A reordered artifact is the one failure worth spending a check on. Per-hand
  stiffness applied to the wrong wrist produces a run that looks entirely
  normal, holds still where it should yield, and measures the other hand.
  """
  names = meta["slot_link_names"]
  if len(names) != len(CONTACT_SLOTS):
    raise ValueError(
      f"{path} trained against {len(names)} contact slots {list(names)}; the "
      f"deploy compliance channel carries {len(CONTACT_SLOTS)} "
      f"({', '.join(CONTACT_SLOTS)}) and cannot address the rest"
    )
  for slot, name in zip(CONTACT_SLOTS, names, strict=True):
    if _SIDE[slot] not in name:
      raise ValueError(
        f"{path} packs its contact slots as {list(names)}; this deploy path "
        f"packs them {list(CONTACT_SLOTS)} (slot {CONTACT_SLOTS.index(slot)} "
        f"must be a {_SIDE[slot]!r} link). Per-hand stiffness would go to the "
        "wrong wrist."
      )


def net_if_problem(nic: str) -> str | None:
  """Why ``nic`` cannot carry DDS right now, or None when it can.

  CycloneDDS refuses a link with no carrier the same way it refuses a name
  that does not exist, so a dead robot link reads as a typo in the flag.
  One sysfs read names the real condition before any DDS participant is
  built.
  """
  path = Path("/sys/class/net") / nic
  if not path.is_dir():
    return f"{nic!r} is not an interface on this host"
  try:
    if path.joinpath("carrier").read_text().strip() == "0":
      return f"{nic} has NO CARRIER: the cable is unplugged or the robot is off"
  except OSError:
    # carrier is EINVAL while the device is administratively down.
    return f"{nic} is DOWN: `sudo ip link set {nic} up`, or plug the cable in"
  return None


@dataclass(frozen=True)
class ControlTable:
  """The per-joint numbers a control step acts on, as the artifact states them."""

  joint_names: tuple[str, ...]
  kp: np.ndarray
  kd: np.ndarray
  action_scale: np.ndarray
  default_pos: np.ndarray
  limit_lo: np.ndarray
  limit_hi: np.ndarray


def control_table(meta: dict) -> ControlTable:
  """The control numbers off the artifact's metadata, which writes every array
  at full round-trip precision."""
  return ControlTable(
    joint_names=tuple(meta["joint_names"]),
    kp=np.asarray(meta["joint_stiffness"], dtype=np.float64),
    kd=np.asarray(meta["joint_damping"], dtype=np.float64),
    action_scale=np.asarray(meta["action_scale"], dtype=np.float64),
    default_pos=np.asarray(meta["default_joint_pos"], dtype=np.float64),
    limit_lo=np.asarray(meta["joint_lower_limits"], dtype=np.float64),
    limit_hi=np.asarray(meta["joint_upper_limits"], dtype=np.float64),
  )
