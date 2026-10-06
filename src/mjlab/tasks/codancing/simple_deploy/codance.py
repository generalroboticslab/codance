"""Codancing deploy: the policy artifact plus a direct observation assembler.

The stand.py recipe, extended to the partner-observing waltz policies: the
ONNX takes one already-stacked ``[1, N]`` float32 vector with its normalizer
baked in and returns the mean action; nothing here simulates, and no mjlab
env, task manager, or training config is involved at any point.

**The layout comes from the artifact.** The export names its terms in
``observation_names``, states per-term history depths in
``observation_history_lengths`` (they are NOT uniform: the partner term keeps
50 frames while everything else keeps 5), and carries each term's parameters
(``body_names``, ``base_body_name``, ``ref_face``) as one JSON string in
``observation_params``. So the partner term's width, the objects a mocap
source must serve, and the reference face all come off the policy.

The waltz policies resolve to nine terms and 1255 values (five frames of each
term, fifty of the partner term), term-major, each history oldest first; the
widths below are per frame::

    base_ang_vel                  3   pelvis gyro, body frame, rad/s
    joint_pos                    29   encoders minus the default pose
    joint_vel                    29   encoder velocities
    actions                      29   previous RAW network output, unscaled
    projected_gravity             3   body-frame gravity from the pelvis IMU
    robot_ref_anchor_ori_b        6   live TORSO orientation vs the free
                                      reference's torso, first two columns
    robot_ref_joint_state        58   free-face reference joints at the cursor
    human_ref_pelvis_knees_pos_b  9   partner pelvis + knees, in the robot
                                      pelvis frame (live VICON at deploy)
    desired_stiffness             4   log compliance channel, slot-major L, R

The reference is the fwd/rev waltz pair read with plain numpy, on the
schedule the training pipeline itself serves: the engage tick observes
frame 1, not frame 0 (frame 0 is the spawn pose training reset to: the
waltz HOLD, which deploy
reproduces by ramping to that very frame, NOT to the default keyframe: they
differ by 1.5 rad on the arms; and the spawn HEADING is that frame's root
yaw, so engage declares the live heading to be it, not identity), a
transition swaps to the next clip's frame 0 the tick after the previous
clip's last frame, and the episode is done after the planned clips. The rev
clip is the time-reversed fwd clip, so the transition boundary is continuous
by construction and needs no re-anchor math here. ``--rounds N`` tiles the
plan: ``single`` becomes the same clip N times (a joint and heading jump at
each boundary, the same-direction clip chaining that training's random pool
selection also produces) and ``transition`` becomes fwd -> rev -> fwd -> rev...
(continuous throughout).

Launch with the venv python, NOT ``uv run`` (measured: the uv wrapper takes
Ctrl-C's SIGINT too and kills the process 0.9 s in, truncating the terminal
damp hold; see the package README)::

    .venv/bin/python3 -m mjlab.tasks.codancing.simple_deploy.codance \\
        --onnx data/checkpoints/decoupled.onnx --motion fwd --net-if <nic> \\
        --dry-run --partner zeros
    .venv/bin/python3 -m mjlab.tasks.codancing.simple_deploy.codance \\
        --onnx data/checkpoints/decoupled.onnx --motion fwd --net-if <nic> \\
        --vicon-ip <tracker> --partner-profile partner-a --demo transition

Hardware order is the standing runner's: remote probe, dry run, e-stop
drills, run. From the dry run on, the robot must already be hung.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import signal
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from mjlab.tasks.codancing.simple_deploy import constants as ctl
from mjlab.tasks.codancing.simple_deploy.io import SensorFrame
from mjlab.tasks.codancing.simple_deploy.stand import (
  ANCHOR_BODY_NAME,
  GRAVITY_DIRECTION,
  TERM_WIDTHS,
  UNSUPPORTED_TERMS,
  ControlStep,
  StiffnessCommand,
  normalize_quat,
  quat_conjugate,
  quat_matrix,
  quat_mul,
  torso_from_pelvis,
  yaw_quat,
)

CONTROL_DT = ctl.CONTROL_DT  # artifacts state theirs and are checked against it.

# The fwd/rev waltz pair every waltz policy trained against. ``rev`` is the
# time-reversed ``fwd`` (verified exact on the arrays, velocities negated), so
# a fwd -> rev transition continues the motion rather than jumping.
WALTZ_DIR = ctl.REPO_ROOT / "data/reference_motion_edits_g1_g1_waltz"
MOTION_FILES: dict[str, Path] = {
  "fwd": WALTZ_DIR / "20260224_001_robot.npz",
  "rev": WALTZ_DIR / "20260408_001_robot.npz",
}

# VICON wiring: the tracker streams its rigid objects at 100 Hz. Only the
# human objects' POSITIONS are read (the partner terms consume positions, and
# the one quaternion read is the robot object's), so their axes need not be
# aligned. One trunk cluster serves whichever trunk body a policy
# observes: pelvis and torso_link both read the same object. This table is the
# DEFAULT wearer's (the human_* sets partner-a wears); a wearer with different
# Tracker objects states `vicon_objects` in partner_profiles.yaml and that
# wins.
VICON_ROBOT_OBJECT = "g1_pelvis"
VICON_OBJECT_BY_BODY = {
  "pelvis": "human_root",
  "torso_link": "human_root",
  "left_knee_link": "human_left_knee",
  "right_knee_link": "human_right_knee",
}
# The operator launches the run and then needs about a minute to walk into
# the capture volume wearing the marker sets; the run waits this long for
# every object to be tracked before any DDS comes up.
VICON_WAIT_S = 180.0

# Real-partner height correction: each human object's world z is shifted by
# the wearer's per-body offset before projection (the trained partner is a
# G1 proxy whose bodies differ in height from the worn clusters). The offsets live
# in partner_profiles.yaml, per wearer, chosen with --partner-profile and
# recorded with the run (profiles.py).


def vicon_objects_for(
  bodies: Sequence[str], object_by_body: Mapping[str, str] | None = None
) -> tuple[str, ...]:
  """Tracker objects for a partner term's body list, in term order. A wearer
  whose sets are not the default human_* objects passes its own profile map
  (``PartnerProfile.objects``), which is also what refuses a body they do not
  wear: an artifact that observes such a body stops here instead of reading
  someone else's markers."""
  table = dict(object_by_body or VICON_OBJECT_BY_BODY)
  unknown = [b for b in bodies if b not in table]
  if unknown:
    raise ValueError(
      f"no Tracker object is mapped for partner bodies {unknown}; known "
      f"bodies: {sorted(table)}. Build the object, then extend the wearer's "
      "vicon_objects (partner_profiles.yaml) or VICON_OBJECT_BY_BODY."
    )
  return tuple(table[b] for b in bodies)


def plan_motions(motion: str, demo: str, rounds: int = 1) -> tuple[str, ...]:
  """The clip directions a run plays, in order: the demo shape tiled ``rounds``
  times (``single`` chains the same clip, ``transition`` alternates)."""
  if motion not in MOTION_FILES:
    raise ValueError(f"--motion must be one of {sorted(MOTION_FILES)}, got {motion!r}")
  if int(rounds) != rounds or rounds < 1:
    raise ValueError(f"--rounds must be a positive integer, got {rounds!r}")
  if demo == "single":
    return (motion,) * rounds
  if demo == "transition":
    (other,) = [m for m in MOTION_FILES if m != motion]
    return (motion, other) * rounds
  raise ValueError(f"--demo must be 'single' or 'transition', got {demo!r}")


def engage_heading_offset(
  live_quat_wxyz: np.ndarray, spawn_root_quat_wxyz: np.ndarray
) -> np.ndarray:
  """The yaw rotation applied to every live quaternion: it carries the
  robot's heading at engage onto the first clip's frame-0 root heading, where
  the training env spawns the robot (a keyframe or rsi reset writes the
  reference root quaternion, world yaw included). The observed
  reference-relative orientation then reads as in training whichever way the
  robot faces in the room."""
  live = yaw_quat(normalize_quat(np.asarray(live_quat_wxyz, dtype=np.float64)))
  spawn = yaw_quat(normalize_quat(np.asarray(spawn_root_quat_wxyz, dtype=np.float64)))
  return quat_mul(spawn, quat_conjugate(live))


class DanceReference:
  """The planned clips as one table, on the training pipeline's own schedule.

  The reset
  observation serves the first clip's frame 1 (frame 0 is the spawn pose),
  each later clip starts at its frame 0 on the tick after the previous clip's
  last frame, and the clip-end pulse the plan counts fires on that same tick.
  So the table is ``first[1:]`` followed by every later clip whole, the
  cursor is the tick count, ``done`` is the cursor running off the table, and
  past the end the last frame is served clamped (the runner ramps back on
  done, so a run never serves a clamped tick).
  """

  def __init__(self, paths: Sequence[str | Path], torso_body_index: int):
    self.paths = tuple(Path(p) for p in paths)
    if not self.paths:
      raise ValueError("the plan needs at least one reference clip")
    joint_pos, joint_vel, torso_quat = [], [], []
    self.clip_ticks: list[int] = []
    for i, path in enumerate(self.paths):
      data = np.load(path, allow_pickle=False)
      jp = np.asarray(data["joint_pos"], dtype=np.float32)
      if jp.shape[1] != ctl.NUM_JOINTS:
        raise ValueError(f"{path} has {jp.shape[1]} joints, expected {ctl.NUM_JOINTS}")
      fps = float(np.asarray(data["fps"]).reshape(-1)[0])
      if abs(fps - 1.0 / CONTROL_DT) > 1e-6:
        raise ValueError(f"{path} is {fps:g} fps; the control loop serves 50")
      quats = np.asarray(data["body_quat_w"], dtype=np.float32)
      if torso_body_index >= quats.shape[1]:
        raise ValueError(
          f"{path} carries {quats.shape[1]} bodies; anchor index "
          f"{torso_body_index} is off its axis"
        )
      start = 1 if i == 0 else 0
      if i == 0:
        # Frame 0 is the pose training SPAWNED at (the waltz hold, arms
        # raised), served to nobody: the engage tick observes frame 1.
        self.spawn_joint_pos = jp[0].astype(np.float64).copy()
        # ... and the HEADING training spawned at: a keyframe or rsi reset
        # writes the reference root quaternion, world yaw included, so the
        # robot faces the way frame 0 does (fwd clip -0.047 rad, rev clip
        # -0.249 rad, never identity). A clip without a root stream carries
        # the root as body 0.
        root = data["root_quat"] if "root_quat" in data.files else quats[:, 0]
        self.spawn_root_quat = np.asarray(root[0], dtype=np.float64).copy()
      joint_pos.append(jp[start:])
      joint_vel.append(np.asarray(data["joint_vel"], dtype=np.float32)[start:])
      torso_quat.append(quats[start:, torso_body_index])
      self.clip_ticks.append(jp.shape[0] - start)
    self.joint_pos = np.concatenate(joint_pos)
    self.joint_vel = np.concatenate(joint_vel)
    self.torso_quat = np.concatenate(torso_quat)
    self.num_ticks = int(self.joint_pos.shape[0])
    # A wrong file pairing shows up as a joint jump at a clip boundary; the
    # trained fwd/rev pair is exactly continuous there.
    self.boundary_gap = 0.0
    edge = 0
    for ticks in self.clip_ticks[:-1]:
      edge += ticks
      gap = float(np.abs(self.joint_pos[edge] - self.joint_pos[edge - 1]).max())
      self.boundary_gap = max(self.boundary_gap, gap)

  @property
  def name(self) -> str:
    return " -> ".join(p.stem for p in self.paths)

  def at(self, cursor: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tick = min(max(cursor, 0), self.num_ticks - 1)
    return self.joint_pos[tick], self.joint_vel[tick], self.torso_quat[tick]

  def done(self, cursor: int) -> bool:
    return cursor >= self.num_ticks


class CodanceController:
  """The 50 Hz control step: sensors + a partner vector in, joint targets out.

  The runner drives the same surface as on StandController: ``start``/``step``
  returning target/raw_action/obs/reference_done, plus kp, kd,
  default_joint_pos, clamp_to_limits, reference_joint_pos, clip_name.
  """

  # The plan bounds the episode through reference_done; the runner's wall
  # clock is the independent ceiling.
  max_episode_ticks: int | None = None

  def __init__(
    self,
    onnx_path: str | Path,
    *,
    motions: Sequence[str] = ("fwd",),
    stiffness: StiffnessCommand | None = None,
    partner_fn: Callable[[], np.ndarray] | None = None,
  ) -> None:
    import onnxruntime as ort

    self.onnx_path = Path(onnx_path)
    self.stiffness = stiffness or StiffnessCommand()
    self.partner_fn = partner_fn

    self._session = ort.InferenceSession(
      str(self.onnx_path), providers=["CPUExecutionProvider"]
    )
    self._input_name = self._session.get_inputs()[0].name
    self._resolve_layout()
    self.motions = tuple(motions)
    self.reference = DanceReference(
      [MOTION_FILES[m] for m in self.motions],
      torso_body_index=self.anchor_body_index,
    )

    # The runner ramps to and holds default_joint_pos before START. Training
    # spawns AT the first clip's frame 0 (the waltz HOLD pose), NOT at the
    # robot's default keyframe: on the fwd clip four arm joints differ by up
    # to 1.5 rad, so ramping to the keyframe would engage from a pose training
    # never resets to.
    # The ACTION offset stays control.default_pos, the trained contract.
    self.default_joint_pos = self.reference.spawn_joint_pos
    self.kp = self.control.kp
    self.kd = self.control.kd

    self._stiffness_log = self.stiffness.as_log()
    self._history = {
      name: np.zeros((self.history_lens[name], self.widths[name]), dtype=np.float32)
      for name in self.terms
    }
    self._cursor = 0
    self._prev_action = np.zeros(ctl.NUM_JOINTS, dtype=np.float32)
    self._heading_offset: np.ndarray | None = None

  @property
  def clip_name(self) -> str:
    return " -> ".join(self.motions)

  # ------------------------------------------------------------ control step

  def start(self, frame: SensorFrame) -> ControlStep:
    """Begin an episode from the robot's current state.

    Declares the robot's present heading to be the first clip's frame-0 root
    yaw, the heading the training env spawns it with, rewinds the plan, and
    backfills every history slot with this first frame. A second ``start`` is
    a full restart.
    """
    if self.partner_term is not None and self.partner_fn is None:
      raise RuntimeError(
        f"{self.onnx_path.name} observes {self.partner_term}; set partner_fn "
        "(live VICON, or zeros for a hung dry run) before starting"
      )
    self._heading_offset = engage_heading_offset(
      frame.quat_wxyz, self.reference.spawn_root_quat
    )
    self._cursor = 0
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
    offset = 0
    for name in self.terms:
      width, depth = self.widths[name], self.history_lens[name]
      if name == term:
        return obs[offset + (depth - 1) * width : offset + depth * width]
      offset += depth * width
    raise KeyError(f"{term!r} is not one of {list(self.terms)}")

  def reference_joint_pos(self) -> np.ndarray:
    """Reference joint positions at the current cursor, for safety checks."""
    return self.reference.at(self._cursor)[0].astype(np.float64)

  def clamp_to_limits(self, target: np.ndarray, margin: float = 0.0) -> np.ndarray:
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
    pelvis = quat_mul(self._heading_offset, normalize_quat(np.asarray(frame.quat_wxyz)))
    torso = quat_mul(pelvis, torso_from_pelvis(joint_pos))

    ref_pos, ref_vel, ref_torso = self.reference.at(self._cursor)
    relative = quat_matrix(
      quat_mul(quat_conjugate(torso), ref_torso.astype(np.float64))
    )

    values = {
      "base_ang_vel": np.asarray(frame.gyro, dtype=np.float64),
      "joint_pos": joint_pos - self.control.default_pos,
      "joint_vel": joint_vel,
      "actions": self._prev_action,
      "projected_gravity": quat_matrix(pelvis).T @ GRAVITY_DIRECTION,
      "robot_ref_anchor_ori_b": relative[:, :2].reshape(-1),
      "robot_ref_joint_state": np.concatenate([ref_pos, ref_vel]),
      "desired_stiffness": self._stiffness_log,
    }
    if self.partner_term is not None:
      assert self.partner_fn is not None
      partner = np.asarray(self.partner_fn(), dtype=np.float64).reshape(-1)
      if partner.shape[0] != self.widths[self.partner_term]:
        raise RuntimeError(
          f"partner source returned {partner.shape[0]} values; "
          f"{self.partner_term} is {self.widths[self.partner_term]} wide"
        )
      values[self.partner_term] = partner
    return values

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
    """Read the observation layout off the artifact.

    stand.py's pattern plus the partner term: ``observation_params`` sizes the
    partner term and says which face and frame each reference term was trained
    against, and per-term history depths are stated (and mixed, so no single
    number describes the vector). An artifact this runner cannot serve is
    refused by name with the missing input said out loud.
    """
    out_shape = list(self._session.get_outputs()[0].shape)
    if out_shape != [1, ctl.NUM_JOINTS]:
      raise ValueError(f"{self.onnx_path} outputs {out_shape}, expected [1, 29]")

    meta = ctl.policy_metadata(self.onnx_path)
    names = meta["observation_names"]
    if abs(meta["control_dt"] - CONTROL_DT) > 1e-9:
      raise ValueError(
        f"{self.onnx_path} trained at control_dt={meta['control_dt']}; this loop "
        "is 50 Hz"
      )
    # Same packing as the standing policy (this module reuses stand.py's
    # StiffnessCommand), so the same slot-order guard applies.
    ctl.check_contact_slots(meta, self.onnx_path)

    params: list[dict] = json.loads(meta["observation_params"])
    if len(params) != len(names):
      raise ValueError(
        f"{self.onnx_path} states {len(params)} observation_params against "
        f"{len(names)} observation terms"
      )

    blocked = [(n, UNSUPPORTED_TERMS[n]) for n in names if n in UNSUPPORTED_TERMS]
    if blocked:
      detail = "; ".join(f"{n} ({why})" for n, why in blocked)
      raise ValueError(
        f"{self.onnx_path} observes terms this runner cannot supply: {detail}"
      )

    # The partner term: sized by its own body list, projected into the frame
    # its params state. This runner serves exactly one, in the robot PELVIS
    # frame, because that is what the VICON source projects into (the
    # g1_pelvis object); a torso-based term would need mocap-side FK that does
    # not exist.
    partner_terms = [n for n in names if n.startswith("human_")]
    if len(partner_terms) > 1:
      raise ValueError(
        f"{self.onnx_path} observes several partner terms {partner_terms}; "
        "this runner wires exactly one live source"
      )
    self.partner_term = partner_terms[0] if partner_terms else None
    self.partner_bodies: tuple[str, ...] = ()
    widths: dict[str, int] = {}
    for name, term_params in zip(names, params, strict=True):
      if name == self.partner_term:
        if not name.endswith("_pos_b"):
          raise ValueError(
            f"{self.onnx_path} observes {name}, which is not a partner "
            "position term; nothing here can produce it"
          )
        bodies = term_params.get("body_names")
        if not bodies:
          raise ValueError(
            f"{self.onnx_path} carries no body_names for {name}; the partner "
            "source cannot be sized"
          )
        base = term_params.get("base_body_name")
        if base != "pelvis":
          raise ValueError(
            f"{self.onnx_path}'s {name} projects into the {base!r} frame; the "
            "VICON source serves the robot pelvis frame only"
          )
        self.partner_bodies = tuple(bodies)
        widths[name] = 3 * len(bodies)
      elif name in TERM_WIDTHS:
        face = term_params.get("ref_face")
        if name.startswith("robot_ref_") and face not in (None, "free"):
          raise ValueError(
            f"{self.onnx_path}'s {name} trained against the {face!r} reference; "
            "the deploy tables serve the free (unforced) reference"
          )
        widths[name] = TERM_WIDTHS[name]
      else:
        raise ValueError(
          f"{self.onnx_path} wants observation terms this controller has "
          f"never seen: {name!r}. Implement them before deploying."
        )
    self.terms: tuple[str, ...] = tuple(names)
    self.widths = widths

    self.control = ctl.control_table(meta)
    if len(self.control.joint_names) != ctl.NUM_JOINTS:
      raise ValueError(
        f"{self.onnx_path} describes {len(self.control.joint_names)} joints, "
        f"this controller drives {ctl.NUM_JOINTS}"
      )

    body_names = list(meta["body_names"])
    if ANCHOR_BODY_NAME not in body_names:
      raise ValueError(
        f"{self.onnx_path} lists body_names without {ANCHOR_BODY_NAME!r}; "
        "this controller reads that body's orientation"
      )
    self.anchor_body_index = body_names.index(ANCHOR_BODY_NAME)

    shape = list(self._session.get_inputs()[0].shape)
    if len(shape) != 2 or shape[0] != 1:
      raise ValueError(f"{self.onnx_path} takes input {shape}, expected [1, N]")
    self.obs_dim = int(shape[1])

    stated = meta["observation_history_lengths"]
    if len(stated) != len(self.terms):
      raise ValueError(
        f"{self.onnx_path} states {len(stated)} history lengths against "
        f"{len(self.terms)} observation terms"
      )
    self.history_lens = {n: int(h) for n, h in zip(self.terms, stated, strict=True)}

    for name, dim in zip(self.terms, meta["observation_dims"], strict=True):
      expect = widths[name] * self.history_lens[name]
      if int(dim) != expect:
        raise ValueError(
          f"{self.onnx_path}'s {name} states {dim} values; its params and "
          f"history say {expect}"
        )
    total = sum(widths[n] * self.history_lens[n] for n in self.terms)
    if total != self.obs_dim:
      raise ValueError(
        f"{self.onnx_path} is {self.obs_dim} wide but its stated terms and "
        f"history lengths add up to {total}"
      )


# ----------------------------------------------------------------------- run


def _tilt_degrees(quat_wxyz: np.ndarray) -> float:
  up = quat_matrix(np.asarray(quat_wxyz, dtype=np.float64))[:, 2]
  return float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))


def _third_diagonal(ori: np.ndarray) -> float:
  """m22 of a rotation matrix, from its first two columns."""
  return float(ori[0] * ori[3] - ori[1] * ori[2])


def dry_run(controller: CodanceController, io, seconds: float) -> int:
  """Read, assemble (live partner included), infer, and damp. Never commands.

  With the robot hung and slack, this confirms motor order and the waist
  chain exactly like the stand dry run, and additionally that the partner
  vector is live and sane: walk around the robot and watch the partner
  distance follow you.
  """

  def _bail(signum, _frame):
    raise KeyboardInterrupt(f"signal {signum}")

  for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    try:
      signal.signal(sig, _bail)
    except ValueError:
      pass  # not the main thread

  io.start()
  print(f"dry run: {seconds:g} s, damp only, no position target is ever sent")
  last_q = np.asarray(io.read().joint_pos).copy()
  t0 = time.monotonic()
  deadline = t0 + seconds
  next_print = t0 + 0.25
  ticks = 0
  started = False
  try:
    while time.monotonic() < deadline:
      tick_t0 = time.monotonic()
      frame = io.read()
      last_q = np.asarray(frame.joint_pos).copy()
      io.damp(last_q)
      step = controller.start(frame) if not started else controller.step(frame)
      started = True
      ticks += 1

      if tick_t0 >= next_print:
        next_print = tick_t0 + 0.25
        ori = controller.newest(step.obs, "robot_ref_anchor_ori_b")
        cos_angle = (ori[0] + ori[3] + _third_diagonal(ori) - 1.0) / 2.0
        ref_err = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        command = np.abs(step.target - last_q).max()
        partner = ""
        if controller.partner_term is not None:
          vec = controller.newest(step.obs, controller.partner_term)
          dists = [
            float(np.linalg.norm(vec[3 * i : 3 * i + 3])) for i in range(len(vec) // 3)
          ]
          partner = "  partner[m] " + " ".join(f"{d:.2f}" for d in dists)
        print(
          f"{ticks / max(time.monotonic() - t0, 1e-9):5.1f} Hz  "
          f"tilt {_tilt_degrees(frame.quat_wxyz):5.1f} deg  "
          f"ref-ori {ref_err:5.1f} deg  "
          f"would-command {command:5.3f} rad{partner}",
          flush=True,
        )

      sleep = tick_t0 + CONTROL_DT - time.monotonic()
      if sleep > 0:
        time.sleep(sleep)
  except BaseException as exc:  # noqa: BLE001 -- every path must damp
    print(f"\n{type(exc).__name__}: {exc}", flush=True)
    return 1
  finally:
    print("damping for 1.2 s before exit", flush=True)
    for _ in range(60):
      try:
        io.damp(last_q)
      except Exception:
        break
      time.sleep(CONTROL_DT)
    io.stop()
    print("exiting; the robot holds the last damp command", flush=True)
  return 0


def _per_slot(values: list[float], flag: str) -> tuple[float, float]:
  """One value covers both wrists; two are left then right."""
  if len(values) == 1:
    return values[0], values[0]
  if len(values) == 2:
    return values[0], values[1]
  raise SystemExit(
    f"{flag}: give one value for both wrists, or two as left then right; "
    f"got {len(values)}"
  )


def _partner_source(controller: CodanceController, args):
  """The live VICON source: Tracker objects derived from the artifact's own
  body list (``--vicon-objects`` overrides), then a blocking wait for every
  object to be tracked, so the operator can launch and walk into the volume."""
  from mjlab.tasks.codancing.simple_deploy.profiles import offsets_for, partner_profile
  from mjlab.tasks.codancing.simple_deploy.vicon import (
    ViconObjects,
    ViconPartnerSource,
  )

  if not args.vicon_ip:
    raise SystemExit("--vicon-ip is required with --partner vicon")
  bodies = controller.partner_bodies
  try:
    profile = partner_profile(args.partner_profile)
    offsets = offsets_for(profile, bodies)
  except ValueError as exc:
    raise SystemExit(str(exc)) from exc
  if args.vicon_objects:
    names = tuple(n.strip() for n in args.vicon_objects.split(",") if n.strip())
    if len(names) != len(bodies):
      raise SystemExit(
        f"{controller.partner_term} observes {len(bodies)} bodies "
        f"({', '.join(bodies)}); --vicon-objects lists {len(names)} ({names}). "
        "Give one Tracker object per body, in that order."
      )
  else:
    try:
      names = vicon_objects_for(bodies, profile.objects or None)
    except ValueError as exc:
      raise SystemExit(str(exc)) from exc
  print(
    f"vicon: partner profile {profile.name!r} (measured {profile.measured}): "
    f"objects {dict(zip(bodies, names, strict=True))}, "
    f"z offsets {dict(zip(bodies, offsets, strict=True))}",
    flush=True,
  )
  source = ViconPartnerSource(
    ViconObjects(robot=args.vicon_robot_object, bodies=names),
    ip=args.vicon_ip,
    body_z_offsets_m=offsets,
    profile=profile.name,
  )
  wait_names = (
    (args.vicon_robot_object,)
    if args.no_wait_human
    else (args.vicon_robot_object, *names)
  )
  hint = "" if args.no_wait_human else " (walk into the volume)"
  print(
    f"vicon: waiting up to {VICON_WAIT_S:g} s for {', '.join(wait_names)}{hint}",
    flush=True,
  )
  source.wait_until_tracked(timeout_s=VICON_WAIT_S, names=wait_names)
  if args.no_wait_human:
    print(
      f"human objects NOT waited for (--no-wait-human): {', '.join(names)} "
      "must be tracked, in the hold, by R1+A; the first sample requires them",
      flush=True,
    )
  return source


def run(args) -> int:
  from mjlab.tasks.codancing.simple_deploy.loop import LoopCfg, Runner
  from mjlab.tasks.codancing.simple_deploy.record import RunRecorder

  # FIRST, before the artifact and long before the VICON walk-in gate: this
  # touches nothing (no DDS participant, no motion-service release), and a
  # dead robot link reaches CycloneDDS as "does not match an available
  # interface", the same words a misspelled name gets. Finding it after three
  # minutes in the volume is the wrong time.
  if args.comms != "dummy":
    if not args.net_if:
      raise SystemExit(
        "--net-if is required with --comms bind: the NIC the robot is on "
        "(`ip -br link` lists them)"
      )
    problem = ctl.net_if_problem(args.net_if)
    if problem:
      raise SystemExit(f"--net-if {args.net_if}: {problem}")

  linear_left, linear_right = _per_slot(args.stiffness_linear, "--stiffness-linear")
  rot_left, rot_right = _per_slot(args.stiffness_rotational, "--stiffness-rotational")
  stiffness = StiffnessCommand(
    linear_left=linear_left,
    rotational_left=rot_left,
    linear_right=linear_right,
    rotational_right=rot_right,
  )
  controller = CodanceController(
    args.onnx,
    motions=plan_motions(args.motion, args.demo, args.rounds),
    stiffness=stiffness,
  )
  vicon_source = None
  if controller.partner_term is not None:
    if args.partner == "vicon":
      vicon_source = _partner_source(controller, args)
      controller.partner_fn = lambda: vicon_source.sample().numpy()
    elif not args.dry_run:
      raise SystemExit(
        "--partner zeros feeds a partner standing inside the robot's pelvis, "
        "which no policy trained on; it is a hung dry-run convenience only"
      )
    else:
      width = controller.widths[controller.partner_term]
      controller.partner_fn = lambda: np.zeros(width)

  plan_s = controller.reference.num_ticks * CONTROL_DT
  print(
    f"plan: {controller.clip_name} ({controller.reference.num_ticks} ticks, "
    f"{plan_s:.1f} s), boundary gap {controller.reference.boundary_gap:.4f} rad"
  )
  print(
    f"stiffness: L {stiffness.linear_left:g} N/m / {stiffness.rotational_left:g} rot"
    f"   R {stiffness.linear_right:g} N/m / {stiffness.rotational_right:g} rot"
  )
  if controller.partner_term is not None:
    pairing = (
      ", ".join(
        f"{b} <- {o}"
        for b, o in zip(
          controller.partner_bodies, vicon_source.objects.bodies, strict=True
        )
      )
      if vicon_source is not None
      else "zeros"
    )
    print(f"partner: {controller.partner_term} ({pairing})")

  if args.comms == "dummy":
    from mjlab.tasks.codancing.simple_deploy.io import DummyIO

    io = DummyIO(controller.default_joint_pos)
  else:
    from mjlab.tasks.codancing.simple_deploy.io import UnitreeBindCfg, UnitreeBindIO

    io = UnitreeBindIO(UnitreeBindCfg(net_if=args.net_if))

  if args.dry_run:
    return dry_run(controller, io, args.seconds)

  cfg = LoopCfg(
    max_episode_s=plan_s + 3.0,
    auto_exit_s=args.seconds if args.comms == "dummy" else 0.0,
  )
  if args.comms == "dummy":
    # The dummy body never tracks the reference, so the joint-deviation trip
    # and the clamp streak would damp within seconds of START on any moving
    # clip. Those checks protect a real plant; disarm them here only.
    cfg.safety.disarm_joint_trips()
  if args.no_joint_trips:
    # Operator opt-out for contact-heavy demos: the two JOINT-SPACE trips only.
    cfg.safety.disarm_joint_trips()
    print(
      "joint-space trips DISARMED (--no-joint-trips): deviation-vs-reference "
      "and clamp-streak; tilt/velocity/staleness/overrun stay armed"
    )
  recorder = RunRecorder(
    "logs/simple_deploy/runs",
    identity={
      "artifact": str(Path(args.onnx).resolve()),
      "plan": controller.clip_name,
      "references": [str(p) for p in controller.reference.paths],
      "stiffness": dataclasses.asdict(stiffness),
      "partner": args.partner,
      "vicon": {
        "ip": args.vicon_ip,
        "no_wait_human": bool(args.no_wait_human),
        **vicon_source.calibration,
      }
      if vicon_source is not None
      else None,
      "comms": args.comms,
      "net_if": args.net_if,
      "no_joint_trips": bool(args.no_joint_trips),
      "kp": controller.kp.tolist(),
      "kd": controller.kd.tolist(),
    },
  )
  if vicon_source is not None:
    # World-frame object poses land in ticks.npz next to the policy I/O.
    recorder.add_provider(vicon_source.record_channels)
  Runner(controller, io, cfg, recorder).run()
  return 0


# ----------------------------------------------------------------------- cli


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Deploy a CoDance policy on a Unitree G1, from its ONNX."
  )
  parser.add_argument(
    "--onnx",
    required=True,
    help="path to the exported policy (.onnx)",
  )
  parser.add_argument(
    "--partner-profile",
    default=None,
    help="who wears the marker sets: a measured name in partner_profiles.yaml "
    "(required with --partner vicon; recorded in run.json)",
  )
  parser.add_argument("--motion", default="fwd", choices=sorted(MOTION_FILES))
  parser.add_argument("--demo", default="single", choices=("single", "transition"))
  parser.add_argument(
    "--rounds",
    type=int,
    default=1,
    help="repeat the demo shape in place: single = the same clip again, "
    "transition = fwd -> rev -> fwd -> rev...",
  )

  parser.add_argument(
    "--net-if",
    default=None,
    help="NIC the robot is on (`ip -br link` lists them); required with --comms "
    "bind, ignored with --comms dummy",
  )
  parser.add_argument("--comms", default="bind", choices=("bind", "dummy"))
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="read, assemble and infer (live partner included), but publish damp only",
  )
  parser.add_argument("--seconds", type=float, default=30.0, help="dry-run duration")
  parser.add_argument(
    "--partner",
    default="vicon",
    choices=("vicon", "zeros"),
    help="live mocap, or zeros for a hung dry run before Tracker is up",
  )
  parser.add_argument(
    "--no-wait-human",
    action="store_true",
    help="wait only for the robot object before bringing DDS up; the human "
    "objects can enter the volume any time before R1+A (the episode's first "
    "sample still requires all of them). The service release then happens "
    "right after launch, so hang the robot snug BEFORE launching",
  )
  parser.add_argument(
    "--no-joint-trips",
    action="store_true",
    help="disarm the two JOINT-SPACE trips (deviation vs reference, "
    "joint-limit clamp streak) for contact-heavy demos; targets are still "
    "clamped to the limits, and tilt/joint-velocity/staleness/overrun stay "
    "armed",
  )
  parser.add_argument(
    "--vicon-ip",
    default=None,
    help="the VICON tracker's address on the mocap LAN (required with --partner vicon)",
  )
  parser.add_argument("--vicon-robot-object", default=VICON_ROBOT_OBJECT)
  parser.add_argument(
    "--vicon-objects",
    default=None,
    help="override the Tracker objects (comma list, in the term's body_names "
    "order); default derives from the body list in the ONNX metadata through the "
    "wearer's --partner-profile (its vicon_objects in partner_profiles.yaml)",
  )
  parser.add_argument(
    "--stiffness-linear",
    type=float,
    nargs="+",
    default=[140.0],
    metavar="N/m",
    help="what the policy is told to expect at the wrists: one value for "
    "both, or two as LEFT RIGHT (trained log-uniform 10 to 1000, median 140)",
  )
  parser.add_argument(
    "--stiffness-rotational",
    type=float,
    nargs="+",
    default=[1.0],
    metavar="K",
    help="one value for both wrists, or two as LEFT RIGHT; pinned at 1.0 in "
    "every training frame",
  )

  args = parser.parse_args()
  return run(args)


if __name__ == "__main__":
  raise SystemExit(main())
