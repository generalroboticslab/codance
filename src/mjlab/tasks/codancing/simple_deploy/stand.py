"""Standing control: the policy artifact plus a direct observation assembler.

The artifact is the whole contract. The ONNX takes one already-stacked
``[1, N]`` float32 vector, carries its observation normalizer baked in, and
returns the deterministic mean action; nothing here simulates, and no physics
engine, task manager, or training config is involved at any point.

**The layout comes from the artifact, not from a constant in this file.** Its
metadata names the terms in order (``observation_names``) and states each
term's history depth (``observation_history_lengths``), and the per-frame
widths are structural. So an artifact is either assembled correctly or refused
by name, saying which input is missing, rather than silently mis-packed or
rejected as an unexplained width.

The standing policy resolves to eight terms, 161 per frame, five frames, 805
total. The flattened layout is term-major, and within a term oldest first::

    base_ang_vel            3     pelvis gyro, body frame, rad/s
    joint_pos              29     encoders minus the default pose
    joint_vel              29     encoder velocities
    actions                29     previous RAW network output, unscaled
    projected_gravity       3     body-frame gravity direction from the IMU
    robot_ref_anchor_ori_b  6     live torso orientation against the reference
    robot_ref_joint_state  58     reference joint positions (29) and velocities
    desired_stiffness       4     log compliance channel, slot-major L then R

Every term is proprioception, a reference lookup, or a commanded constant, so
the vector holds no phase, clock, or cursor. The standing policy is therefore
time-invariant at run time: how long an episode lasts is an operator and safety
choice, not something the observation demands.

The reference is read from the free-face motion NPZ with plain numpy. It is
constant, so the cursor never changes what is served.

**The runner never reads the checkpoint's resolved config.** Reaching it means
opening the 20 MB ``.pt`` with torch, and the runner is deliberately a small
ONNX file plus onnxruntime. The numbers it needs (gains, action scale, default
pose, joint ranges) are baked into the ONNX at export, frozen alongside the
weights. The resolved config only names them through ``_target_`` import
paths, which resolve against the current ``asset_zoo`` rather than the values
the policy trained with.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mjlab.tasks.codancing.simple_deploy import constants as ctl
from mjlab.tasks.codancing.simple_deploy.io import SensorFrame

# Per-frame width of every actor term the runners know. The layout is NOT
# pinned here: an artifact names which of these it wants, and in what order,
# through its ``observation_names`` metadata, so the controller reads its shape
# off the policy it is given. Widths are structural (joint counts, a rotation
# matrix's first two columns, two compliance slots), so this table is the
# stable part.
TERM_WIDTHS: dict[str, int] = {
  "base_lin_vel": 3,
  "base_ang_vel": 3,
  "joint_pos": ctl.NUM_JOINTS,
  "joint_vel": ctl.NUM_JOINTS,
  "actions": ctl.NUM_JOINTS,
  "projected_gravity": 3,
  "robot_ref_anchor_pos_b": 3,
  "robot_ref_anchor_ori_b": 6,
  "robot_ref_joint_state": 2 * ctl.NUM_JOINTS,
  "desired_stiffness": 4,
}

# Terms that need an input the runners do not have. Saying WHICH input is
# missing beats a bare width mismatch: a reader who sees the reason knows
# immediately whether the artifact is deployable at all.
UNSUPPORTED_TERMS: dict[str, str] = {
  "base_lin_vel": "needs odometry, which the runners do not have",
  "robot_ref_anchor_pos_b": "needs odometry for the robot's absolute world pose",
}


def _unsupported_reason(name: str) -> str | None:
  """Why the standing runner cannot supply a term, or None. Partner terms
  match by prefix, so every partner body set is covered without enumerating
  keys."""
  if name.startswith("human_"):
    return "needs a partner: live VICON"
  return UNSUPPORTED_TERMS.get(name)


# Anchor body, by NAME. Its index on the motion NPZ's body axis (MJCF body
# order minus the world body) comes from the artifact's ``body_names``: the NPZ
# has no name array of its own, so a wrong index would go unnoticed.
ANCHOR_BODY_NAME = "torso_link"

GRAVITY_DIRECTION = np.array([0.0, 0.0, -1.0])


def newest_frame(
  obs: np.ndarray, term: str, terms: tuple[str, ...], history: dict[str, int] | int
) -> np.ndarray:
  """The live frame of one term, out of a flat observation vector.

  Terms are concatenated in the artifact's own order and each carries its
  history oldest first, so the current value of a term sits at the END of its
  block, not the start.
  """
  offset = 0
  for name in terms:
    width = TERM_WIDTHS[name]
    depth = history if isinstance(history, int) else history[name]
    if name == term:
      return obs[offset + (depth - 1) * width : offset + depth * width]
    offset += depth * width
  raise KeyError(f"{term!r} is not one of {list(terms)}")


# --------------------------------------------------------------- quaternions
# All quaternions are wxyz, matching both the DDS messages and the clip files.


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b
  return np.array(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]
  )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
  """The inverse of a unit quaternion."""
  return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_matrix(q: np.ndarray) -> np.ndarray:
  w, x, y, z = q
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
  )


def yaw_quat(q: np.ndarray) -> np.ndarray:
  """The heading-only part of a rotation."""
  yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
  return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def axis_quat(axis: int, angle: float) -> np.ndarray:
  """Rotation of ``angle`` about principal axis 0=x, 1=y, 2=z."""
  q = np.zeros(4)
  q[0] = math.cos(angle / 2)
  q[axis + 1] = math.sin(angle / 2)
  return q


def normalize_quat(q: np.ndarray) -> np.ndarray:
  return np.asarray(q, dtype=np.float64) / np.linalg.norm(q)


def torso_from_pelvis(joint_pos: np.ndarray) -> np.ndarray:
  """Pelvis-to-torso rotation carried by the three waist joints.

  Verified on the compiled G1: every body in the pelvis to torso_link chain
  sits at identity relative orientation and the joint axes are z, x, y in
  chain order, so composing the three joint rotations is exact rather than an
  approximation of a forward-kinematics solve.
  """
  return quat_mul(
    quat_mul(
      axis_quat(2, float(joint_pos[ctl.WAIST_YAW])),
      axis_quat(0, float(joint_pos[ctl.WAIST_ROLL])),
    ),
    axis_quat(1, float(joint_pos[ctl.WAIST_PITCH])),
  )


# ----------------------------------------------------------------- reference


class ReferenceMotion:
  """A free-face motion NPZ, read with plain numpy.

  Cursor rules copied from the training registry: one frame per control step
  starting at 0, clamped at the last frame, with ``done`` reported one step
  after that. The clamp means the last frame is served on two consecutive
  steps, which is what the registry does.
  """

  def __init__(self, path: str | Path, torso_body_index: int):
    self.path = Path(path)
    data = np.load(self.path, allow_pickle=False)
    self.joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
    self.joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
    self.torso_quat = np.asarray(
      data["body_quat_w"][:, torso_body_index], dtype=np.float32
    )
    self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    self.num_frames = int(self.joint_pos.shape[0])
    if self.joint_pos.shape[1] != ctl.NUM_JOINTS:
      raise ValueError(
        f"{self.path} has {self.joint_pos.shape[1]} joints, expected {ctl.NUM_JOINTS}"
      )
    self.is_static = bool(
      np.array_equal(
        self.joint_pos, np.broadcast_to(self.joint_pos[0], self.joint_pos.shape)
      )
      and np.array_equal(
        self.torso_quat, np.broadcast_to(self.torso_quat[0], self.torso_quat.shape)
      )
    )

  @property
  def name(self) -> str:
    return self.path.stem

  def at(self, cursor: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = min(max(cursor, 0), self.num_frames - 1)
    return self.joint_pos[frame], self.joint_vel[frame], self.torso_quat[frame]

  def done(self, cursor: int) -> bool:
    """Whether the cursor has run past the reference.

    A static reference never runs out: every frame carries the same numbers,
    so clamping at the last one serves exactly what frame 0 did and there is
    nothing for the cursor to reach the end OF. Reporting done there would
    stop an episode after however many frames the clip file happens to hold,
    which for the standing clip is 10.02 s of no significance whatever. Let
    the operator's wall clock own episode length instead, which is what the
    contract says it is.
    """
    if self.is_static:
      return False
    return cursor >= self.num_frames


# ------------------------------------------------------------------ commands


@dataclass
class StiffnessCommand:
  """The compliance channel the policy is told to expect, slot-major L then R.

  It is a command, not a sensor, and it has no physical consumer: nothing on
  the robot becomes mechanically softer when it reads 140 N/m. It only shapes
  what the policy expects, so it is fixed for a run and tuned between runs.

  The released pools write 1.0 into the ROTATIONAL pair on every frame of
  every clip, so the policies never saw another value there and 1.0 is the
  only in-distribution setting. The linear pair spans 10 to 1000 N/m with a
  median of 140, which is also what the augmentation writes into a free frame,
  so the defaults are neutral.
  """

  linear_left: float = 140.0
  rotational_left: float = 1.0
  linear_right: float = 140.0
  rotational_right: float = 1.0

  def as_log(self) -> np.ndarray:
    # float32 throughout, because that is what training computed the log in:
    # log(1.0 + 1e-6) differs between the two precisions in the sixth digit.
    values = np.array(
      [
        self.linear_left,
        self.rotational_left,
        self.linear_right,
        self.rotational_right,
      ],
      dtype=np.float32,
    )
    return np.log(values + np.float32(1e-6))


@dataclass
class ControlStep:
  """One control step's outputs."""

  target: np.ndarray  # (29,) joint position targets, raw * scale + default
  raw_action: np.ndarray  # (29,) network output before scale and offset
  obs: np.ndarray  # the vector handed to the policy (805 standing, 1255 waltz)
  reference_done: bool


# ---------------------------------------------------------------- controller


class StandController:
  """The 50 Hz control step: sensors in, joint position targets out."""

  # The observation carries no clock, so nothing here bounds an episode. The
  # runner applies its own wall-clock limit, and the reference reports done.
  max_episode_ticks: int | None = None

  def __init__(
    self,
    onnx_path: str | Path,
    *,
    reference_path: str | Path = ctl.STAND_REFERENCE,
    stiffness: StiffnessCommand | None = None,
  ) -> None:
    import onnxruntime as ort

    self.onnx_path = Path(onnx_path)
    self.stiffness = stiffness or StiffnessCommand()

    self._session = ort.InferenceSession(
      str(self.onnx_path), providers=["CPUExecutionProvider"]
    )
    self._input_name = self._session.get_inputs()[0].name
    # The layout, the control table and the anchor body all come off the
    # artifact, so it is read before anything that depends on them.
    self._resolve_layout()
    self.reference = ReferenceMotion(
      reference_path, torso_body_index=self.anchor_body_index
    )
    # A wrong reference file silently shifts the 24 non-zero entries of the
    # reference joint block, which no shape check would catch.
    if self.reference.is_static and not np.allclose(
      self.reference.joint_pos[0], self.control.default_pos, atol=1e-6
    ):
      raise ValueError(
        f"{self.reference.path} is a constant clip whose pose is not the "
        "knees-bent keyframe; this is not the standing reference"
      )

    # Control constants, exposed for the IO layer and the safety clamp.
    self.default_joint_pos = self.control.default_pos
    self.kp = self.control.kp
    self.kd = self.control.kd

    self._stiffness_log = self.stiffness.as_log()
    self._history = {
      name: np.zeros((self.history_lens[name], TERM_WIDTHS[name]), dtype=np.float32)
      for name in self.terms
    }
    self._cursor = 0
    self._prev_action = np.zeros(ctl.NUM_JOINTS, dtype=np.float32)
    self._heading_offset: np.ndarray | None = None

  @property
  def clip_name(self) -> str:
    return self.reference.name

  # The linear channel's trained range (log-uniform over it, median 140); a
  # runtime nudge never leaves it, so the policy is never told a value it has
  # not seen.
  STIFFNESS_LINEAR_RANGE = (10.0, 1000.0)

  def nudge_stiffness(self, delta: float) -> StiffnessCommand:
    """Move BOTH wrists' linear stiffness by ``delta`` N/m from the next step
    on, clamped to the trained range; the rotational pair stays. The term's
    history then holds old and new values for four ticks, which is also what a
    training clip's stiffness switch looks like. Returns the new command."""
    lo, hi = self.STIFFNESS_LINEAR_RANGE
    s = self.stiffness
    self.stiffness = StiffnessCommand(
      float(np.clip(s.linear_left + delta, lo, hi)),
      s.rotational_left,
      float(np.clip(s.linear_right + delta, lo, hi)),
      s.rotational_right,
    )
    self._stiffness_log = self.stiffness.as_log()
    return self.stiffness

  # ------------------------------------------------------------ control step

  def start(self, frame: SensorFrame) -> ControlStep:
    """Begin an episode from the robot's current state.

    Declares the robot's present heading to be the identity yaw training
    spawned at, rewinds the reference, and backfills every history slot with
    this first frame. Never resume a stopped episode: a second ``start`` is a
    full restart, which is what the policy was trained to see.
    """
    self._heading_offset = quat_conjugate(
      yaw_quat(normalize_quat(np.asarray(frame.quat_wxyz)))
    )
    self._cursor = 0
    # Training zeroes the action buffer at reset. Backfilling uniformly below
    # reproduces that only because the previous action is zero here; do not
    # carry an action across a restart.
    self._prev_action = np.zeros(ctl.NUM_JOINTS, dtype=np.float32)
    values = self._assemble_terms(frame)
    for name in self.terms:
      self._history[name][:] = values[name]
    return self._infer()

  def step(self, frame: SensorFrame) -> ControlStep:
    """One control step of a running episode."""
    if self._heading_offset is None:
      raise RuntimeError("call start() before step(); never begin mid-episode")
    self._cursor += 1
    values = self._assemble_terms(frame)
    for name in self.terms:
      buffer = self._history[name]
      buffer[:-1] = buffer[1:]  # oldest first, newest last
      buffer[-1] = values[name]
    return self._infer()

  def newest(self, obs: np.ndarray, term: str) -> np.ndarray:
    """The live frame of one term, using this artifact's own layout."""
    return newest_frame(obs, term, self.terms, self.history_lens)

  def reference_joint_pos(self) -> np.ndarray:
    """Reference joint positions at the current cursor, for safety checks."""
    return self.reference.at(self._cursor)[0].astype(np.float64)

  def clamp_to_limits(self, target: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Clamp to the joint ranges.

    A deploy safety addition, not a fidelity requirement: training never
    clipped the position target anywhere.
    """
    return np.clip(
      target, self.control.limit_lo + margin, self.control.limit_hi - margin
    )

  def action_to_target(self, raw_action: np.ndarray) -> np.ndarray:
    return (
      np.asarray(raw_action, dtype=np.float64) * self.control.action_scale
      + self.control.default_pos
    )

  # ----------------------------------------------------------------- internals

  def _assemble_terms(self, frame: SensorFrame) -> dict[str, np.ndarray]:
    joint_pos = np.asarray(frame.joint_pos, dtype=np.float64)
    joint_vel = np.asarray(frame.joint_vel, dtype=np.float64)
    assert self._heading_offset is not None
    # Subtract the heading declared at start, so the live world and the
    # reference world share the identity yaw training used. The G1's yaw is
    # gyro-integrated and drifts, which is why it is re-zeroed every start.
    pelvis = quat_mul(self._heading_offset, normalize_quat(np.asarray(frame.quat_wxyz)))
    torso = quat_mul(pelvis, torso_from_pelvis(joint_pos))

    ref_pos, ref_vel, ref_torso = self.reference.at(self._cursor)
    # The rotation of the reference torso frame seen from the live torso
    # frame, packed as its first two COLUMNS read row-major.
    relative = quat_matrix(
      quat_mul(quat_conjugate(torso), ref_torso.astype(np.float64))
    )

    # Every term the standing runner can produce. Only the ones the artifact asked for
    # are read out of here, so an artifact that wants fewer simply ignores the
    # rest rather than needing a second code path.
    return {
      "base_ang_vel": np.asarray(frame.gyro, dtype=np.float64),
      "joint_pos": joint_pos - self.control.default_pos,
      "joint_vel": joint_vel,
      "actions": self._prev_action,
      "projected_gravity": quat_matrix(pelvis).T @ GRAVITY_DIRECTION,
      "robot_ref_anchor_ori_b": relative[:, :2].reshape(-1),
      "robot_ref_joint_state": np.concatenate([ref_pos, ref_vel]),
      "desired_stiffness": self._stiffness_log,
    }

  def _infer(self) -> ControlStep:
    obs = np.concatenate(
      [self._history[name].reshape(-1) for name in self.terms]
    ).astype(np.float32)
    outputs = self._session.run(None, {self._input_name: obs.reshape(1, -1)})
    raw = np.asarray(outputs[0], dtype=np.float32)[0]
    self._prev_action = raw.astype(np.float32)
    return ControlStep(
      target=self.action_to_target(raw),
      raw_action=raw.astype(np.float64),
      obs=obs,
      reference_done=self.reference.done(self._cursor),
    )

  def _resolve_layout(self) -> None:
    """Read the observation layout off the artifact, not off a pinned constant.

    An artifact names its terms, in order, in ``observation_names``, states
    each term's history depth in ``observation_history_lengths`` and its total
    width in ``observation_dims``, and the per-frame widths are structural.
    Actors differ only by which terms are present (805 wide for the standing
    policy, 1255 for the waltz ones), so an artifact from another policy either
    works here or is refused by NAME, with the missing input said out loud,
    rather than by an unexplained width mismatch.
    """
    out_shape = list(self._session.get_outputs()[0].shape)
    if out_shape != [1, ctl.NUM_JOINTS]:
      raise ValueError(f"{self.onnx_path} outputs {out_shape}, expected [1, 29]")

    meta = ctl.policy_metadata(self.onnx_path)
    names = meta["observation_names"]
    # Slot order for the compliance channel, and the rate the loop runs at.
    # Both are wrong SILENTLY when assumed: a swapped slot sends per-hand
    # stiffness to the other wrist, and a mismatched rate leaves every target
    # plausible and only the closed loop off.
    ctl.check_contact_slots(meta, self.onnx_path)
    if abs(meta["control_dt"] - ctl.CONTROL_DT) > 1e-9:
      raise ValueError(
        f"{self.onnx_path} trained at control_dt={meta['control_dt']}; this loop "
        f"runs at {ctl.CONTROL_DT} ({1.0 / ctl.CONTROL_DT:g} Hz)"
      )

    # Blocked-first: a partner-observing policy must be refused with WHY
    # (needs a partner), not reported as a never-seen width.
    blocked = [
      (n, reason) for n in names if (reason := _unsupported_reason(n)) is not None
    ]
    if blocked:
      detail = "; ".join(f"{n} ({why})" for n, why in blocked)
      raise ValueError(
        f"{self.onnx_path} observes terms the standing runner cannot supply: {detail}"
      )
    unknown = [n for n in names if n not in TERM_WIDTHS]
    if unknown:
      raise ValueError(
        f"{self.onnx_path} wants observation terms this controller has never "
        f"seen: {unknown}. Add their widths to TERM_WIDTHS and implement them "
        "before deploying."
      )

    # Which FACE of the reference each term trained against. The standing
    # runner serves the free face and nothing else: it reads the free-face NPZ
    # directly and has no adapted face to offer. The names alone do not say
    # which one a term wants, so an artifact trained on the served face would
    # be fed the wrong reference at every step with no shape to catch it.
    params: list[dict] = json.loads(meta["observation_params"])
    if len(params) != len(names):
      raise ValueError(
        f"{self.onnx_path} states {len(params)} observation_params against "
        f"{len(names)} observation terms"
      )
    for name, term_params in zip(names, params, strict=True):
      face = term_params.get("ref_face")
      if name.startswith("robot_ref_") and face not in (None, "free"):
        raise ValueError(
          f"{self.onnx_path}'s {name} trained against the {face!r} reference; the "
          "standing runner serves the free (unforced) reference only"
        )

    self.control = ctl.control_table(meta)
    if len(self.control.joint_names) != ctl.NUM_JOINTS:
      raise ValueError(
        f"{self.onnx_path} describes {len(self.control.joint_names)} joints, "
        f"this controller drives {ctl.NUM_JOINTS}"
      )

    # Anchor body index on the motion NPZ's body axis, by name: those files
    # carry no name array of their own.
    body_names = list(meta["body_names"])
    if ANCHOR_BODY_NAME not in body_names:
      raise ValueError(
        f"{self.onnx_path} lists body_names without {ANCHOR_BODY_NAME!r}; "
        "this controller reads that body's orientation"
      )
    self.anchor_body_index = body_names.index(ANCHOR_BODY_NAME)

    self.terms: tuple[str, ...] = tuple(names)
    shape = list(self._session.get_inputs()[0].shape)
    if len(shape) != 2 or shape[0] != 1:
      raise ValueError(f"{self.onnx_path} takes input {shape}, expected [1, N]")
    self.obs_dim = int(shape[1])

    # History depth is PER TERM: the observation group may set one depth for
    # everything or leave each term to its own, and with mixed depths no single
    # number describes the vector.
    stated = meta["observation_history_lengths"]
    if len(stated) != len(self.terms):
      raise ValueError(
        f"{self.onnx_path} states {len(stated)} history lengths against "
        f"{len(self.terms)} observation terms"
      )
    self.history_lens = {n: int(h) for n, h in zip(self.terms, stated, strict=True)}

    # Per-term widths. The sum check below catches the same disagreements, but
    # only ever blames the whole vector, and "805 wide against 825" says nothing
    # about WHICH term moved. The one that realistically moves is
    # `desired_stiffness`: it is 2 per contact slot, so a data release with a
    # different slot count changes its width and nothing else.
    stated_dims = meta["observation_dims"]
    if len(stated_dims) != len(self.terms):
      raise ValueError(
        f"{self.onnx_path} states {len(stated_dims)} observation_dims "
        f"against {len(self.terms)} observation terms"
      )
    for name, dim in zip(self.terms, stated_dims, strict=True):
      want = TERM_WIDTHS[name] * self.history_lens[name]
      if dim != want:
        raise ValueError(
          f"{self.onnx_path}'s {name} is {dim} wide over "
          f"{self.history_lens[name]} frames; this controller assembles "
          f"{TERM_WIDTHS[name]} per frame ({want} total). It was trained "
          "against a different shape of that term."
        )

    total = sum(TERM_WIDTHS[n] * self.history_lens[n] for n in self.terms)
    if total != self.obs_dim:
      raise ValueError(
        f"{self.onnx_path} is {self.obs_dim} wide but its stated terms and "
        f"history lengths add up to {total}"
      )
