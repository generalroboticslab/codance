"""Live VICON partner source for the codancing deploy.

Built on wire facts checked against a live tracker (DataStream port 801): raw
``pyvicon_datastream.PyViconDatastream`` client (NOT the euler ``tools``
wrapper), subject name == segment name for rigid Tracker objects,
``get_segment_global_translation`` in MILLIMETRES, and
``get_segment_global_quaternion`` in WXYZ (verified against the rotation
matrix; same convention as MuJoCo, no reorder). No quality flag is
available. The tracker streams 100 Hz (``get_frame_rate()`` reads 100.0
with subjects up, 0.0 with none; never gated on), latency_total ~8 ms, and
ServerPush buffers only the NEWEST frame: a client that went idle reads the
live frame, not a backlog, so one ``get_frame`` per 50 Hz tick always serves
the latest sample (the fresh-frame wait blocks ~10 ms only when reading
faster than 100 Hz).

The Tracker objects: ``g1_pelvis`` on the robot, and each wearer's own set on
the partner (``human_root``, ``human_left_knee`` and ``human_right_knee`` for
partner-a; ``partner_profiles.yaml`` lists every wearer's). Only the human
objects' positions are read, which is the term contract anyway, so their axes
need not be aligned (the one quaternion consumed is the robot object's).

The per-tick contract is the partner term's (``human_ref_pelvis_knees_pos_b``
for the waltz policies): for each human object,
``R_base^T (p_body - p_base)`` in the robot pelvis LINK frame, meters. Both
sides come out of the same VICON world frame, so no extra registration is
needed. Two fixed corrections sit between the raw stream and the contract:
the base pose is mapped from the Tracker OBJECT frame to the pelvis LINK
frame through ``MOUNT_POS_M`` / ``MOUNT_QUAT_WXYZ`` (the object pose in the
link frame; identity by default, which assumes the object was built aligned to
the link; ``MOUNT_POS_M`` says how to measure it), and
each human object's world z is shifted by ``body_z_offsets_m`` (the trained
partner is a G1 PROXY whose bodies differ in height from a real human's worn
clusters; the per-wearer values live in ``partner_profiles.yaml``).

Occlusion: a failed ``get_frame`` or a missing object HOLDS the last good
vector for up to ``max_stale_reads`` samples, then raises (the runner damps
on any controller exception). The very first sample must see every object; the
operator needs about a minute after launch to walk into the capture volume
wearing the marker sets, so ``wait_until_tracked`` blocks until
every object is tracked BEFORE any DDS comes up, and the episode itself
starts with the partner already in place.

NETWORK: ``connect()`` has NO timeout and blocks for minutes on a dead port,
so the constructor pre-checks TCP 801 with a 1 s socket connect first. Only
a machine on the tracker's network reaches it, so run this source there.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

MM_TO_M = 1e-3
VICON_PORT = 801

# Mounting calibration (identity by default): the pose of the g1_pelvis Tracker
# OBJECT expressed in the robot pelvis LINK frame the terms trained against.
# Identity assumes the object was built aligned to the link. To measure:
# stand the robot at the default keyframe, level, pelvis facing the Vicon
# world +x; the object reading at that instant IS this pose.
MOUNT_POS_M = (0.0, 0.0, 0.0)
MOUNT_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
  """Rotation matrix from a WXYZ quaternion (the verified stream order)."""
  w, x, y, z = (float(v) for v in q)
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
  )


def partner_vector(
  base_pos: np.ndarray, base_rot: np.ndarray, body_positions: Sequence[np.ndarray]
) -> np.ndarray:
  """Concatenated ``R_base^T (p - p_base)`` per body, the term's contract."""
  out = [
    base_rot.T @ (np.asarray(p, dtype=np.float64) - base_pos) for p in body_positions
  ]
  return np.concatenate(out)


def link_pose_from_object(
  obj_pos: np.ndarray,
  obj_rot: np.ndarray,
  mount_pos: np.ndarray,
  mount_rot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Robot pelvis LINK pose from the Tracker OBJECT pose.

  ``mount_pos`` / ``mount_rot`` are the object pose expressed in the link
  frame (identity when the object is aligned to the link)."""
  link_rot = obj_rot @ mount_rot.T
  return obj_pos - link_rot @ mount_pos, link_rot


@dataclass(frozen=True)
class ViconObjects:
  """Tracker object names: the robot pelvis + the human bodies IN TERM ORDER
  (the artifact's ``body_names`` order; for ``human_ref_pelvis_knees_pos_b``
  that is pelvis, left knee, right knee)."""

  robot: str
  bodies: tuple[str, ...]


def _connect_client(ip: str):
  """Pre-checked connect: fail in ~1 s instead of blocking minutes."""
  try:
    socket.create_connection((ip, VICON_PORT), timeout=1.0).close()
  except OSError as exc:
    raise RuntimeError(
      f"VICON tracker {ip}:{VICON_PORT} unreachable ({exc}); is Vicon Tracker "
      "running and streaming, and is this machine on the mocap network?"
    ) from exc
  import pyvicon_datastream as pv

  client = pv.PyViconDatastream()
  if client.connect(ip) != pv.Result.Success:
    raise RuntimeError(f"VICON connect({ip}) failed after the port pre-check")
  client.enable_segment_data()
  client.set_stream_mode(pv.StreamMode.ServerPush)
  return client


class ViconPartnerSource:
  """Per-tick partner vector from live VICON (``fn`` feeds the partner term)."""

  def __init__(
    self,
    objects: ViconObjects,
    *,
    ip: str | None = None,
    client=None,
    max_stale_reads: int = 25,  # 0.5 s at 50 Hz
    mount_pos_m: Sequence[float] = MOUNT_POS_M,
    mount_quat_wxyz: Sequence[float] = MOUNT_QUAT_WXYZ,
    body_z_offsets_m: Sequence[float] | None = None,
    profile: str | None = None,  # the wearer the offsets belong to (record)
  ) -> None:
    if client is None:
      if ip is None:
        raise ValueError("pass a client or the VICON tracker ip")
      client = _connect_client(ip)
    self.objects = objects
    self.client = client
    self.profile = profile
    self.max_stale_reads = max_stale_reads
    self._mount_pos = np.asarray(mount_pos_m, dtype=np.float64)
    self._mount_rot = quat_wxyz_to_matrix(np.asarray(mount_quat_wxyz, dtype=np.float64))
    offsets = (
      body_z_offsets_m if body_z_offsets_m is not None else [0.0] * len(objects.bodies)
    )
    self._z_offsets = np.asarray(offsets, dtype=np.float64)
    if self._z_offsets.shape != (len(objects.bodies),):
      raise ValueError(
        f"body_z_offsets_m needs one value per body "
        f"({len(objects.bodies)}), got shape {self._z_offsets.shape}"
      )
    self._last_vector: np.ndarray | None = None
    self._stale = 0
    # World-frame poses of the last good frame, for the run record.
    self._last_raw: dict[str, np.ndarray] | None = None

  def _hold(self, what: str) -> torch.Tensor:
    """Serve the last good vector through a hiccup; bounded, then damp."""
    if self._last_vector is None:
      raise RuntimeError(
        f"VICON: {what} on the very first sample; check Tracker and the "
        f"object names {(self.objects.robot, *self.objects.bodies)!r}."
      )
    self._stale += 1
    if self._stale > self.max_stale_reads:
      raise RuntimeError(
        f"VICON: {what} for {self._stale} consecutive samples "
        f"(> {self.max_stale_reads}); damp rather than dance blind."
      )
    return torch.as_tensor(self._last_vector, dtype=torch.float32)

  def missing_objects(
    self, names: Sequence[str] | None = None
  ) -> tuple[str, ...] | None:
    """Object names absent from one fresh frame; None when get_frame fails.

    ``names`` scopes the check (default: the robot and every body)."""
    import pyvicon_datastream as pv

    c = self.client
    if c.get_frame() != pv.Result.Success:
      return None
    if names is None:
      names = (self.objects.robot, *self.objects.bodies)
    return tuple(
      name for name in names if c.get_segment_global_translation(name, name) is None
    )

  def wait_until_tracked(
    self,
    timeout_s: float,
    settle_reads: int = 10,
    names: Sequence[str] | None = None,
  ) -> None:
    """Block until every object (or the ``names`` subset, e.g. the robot
    alone when the operator skips the human wait) is tracked for
    ``settle_reads`` consecutive frames (0.1 s at the tracker's 100 Hz),
    reporting what is still missing every few seconds. This is the
    walk-into-the-volume gate; ``sample`` still requires full visibility on
    its own first read."""
    t0 = time.monotonic()
    next_report = 5.0
    good = 0
    while True:
      missing = self.missing_objects(names)
      if missing == ():
        good += 1
        if good >= settle_reads:
          print("[vicon] all waited objects tracked", flush=True)
          return
        continue
      good = 0
      if missing is None:
        time.sleep(0.05)  # do not spin hot while the tracker serves nothing
      waited = time.monotonic() - t0
      if waited > timeout_s:
        what = (
          "get_frame still fails"
          if missing is None
          else f"{', '.join(missing)} still not tracked"
        )
        raise RuntimeError(
          f"VICON: waited {waited:.0f} s and {what}; check Tracker and the marker sets."
        )
      if waited >= next_report:
        next_report = waited + 5.0
        what = "get_frame failing" if missing is None else ", ".join(missing)
        print(f"[vicon] waiting for {what} ({waited:.0f} s)", flush=True)

  def sample(self) -> torch.Tensor:
    """ONE get_frame, then every object off that frame (no per-object frames)."""
    import pyvicon_datastream as pv

    c = self.client
    if c.get_frame() != pv.Result.Success:
      return self._hold("get_frame failed")
    name = self.objects.robot
    p_base = c.get_segment_global_translation(name, name)
    q_base = c.get_segment_global_quaternion(name, name)
    if p_base is None or q_base is None:
      return self._hold(f"robot object {name!r} missing")
    base_pos, base_rot = link_pose_from_object(
      np.asarray(p_base, dtype=np.float64) * MM_TO_M,
      quat_wxyz_to_matrix(np.asarray(q_base, dtype=np.float64)),
      self._mount_pos,
      self._mount_rot,
    )
    bodies = []
    for i, name in enumerate(self.objects.bodies):
      p = c.get_segment_global_translation(name, name)
      if p is None:
        return self._hold(f"object {name!r} missing")
      pos = np.asarray(p, dtype=np.float64) * MM_TO_M
      pos[2] -= self._z_offsets[i]
      bodies.append(pos)
    vec = partner_vector(base_pos, base_rot, bodies)
    self._last_vector = vec
    self._last_raw = {
      "vicon_robot_obj_pos": np.asarray(p_base, dtype=np.float64) * MM_TO_M,
      "vicon_robot_obj_quat": np.asarray(q_base, dtype=np.float64),
      "vicon_pelvis_pos": np.asarray(base_pos, dtype=np.float64),
      # As tracked (m), BEFORE the per-body z offsets the term subtracts.
      "vicon_human_pos": np.stack(bodies)
      + np.c_[np.zeros((len(bodies), 2)), self._z_offsets],
    }
    self._stale = 0
    return torch.as_tensor(vec, dtype=torch.float32)

  @property
  def fn(self) -> Callable[[], torch.Tensor]:
    return self.sample

  @property
  def calibration(self) -> dict:
    """The constants that turn the raw objects into the term: for the run
    record, so the world-frame channels can be re-projected offline."""
    return {
      "robot_object": self.objects.robot,
      "objects": list(self.objects.bodies),
      "partner_profile": self.profile,
      "mount_pos_m": self._mount_pos.tolist(),
      "mount_rot": self._mount_rot.tolist(),
      "body_z_offsets_m": self._z_offsets.tolist(),
    }

  def record_channels(self) -> dict[str, np.ndarray | float]:
    """Per-tick world-frame channels for ``RunRecorder.add_provider``: the raw
    robot object pose (m, wxyz), the pelvis link position it resolves to, every
    human object position as tracked, and how many consecutive samples have
    been held (0 = this tick's frame was fresh). NaN before the first sample.
    """
    n = len(self.objects.bodies)
    if self._last_raw is None:
      return {
        "vicon_robot_obj_pos": np.full(3, np.nan),
        "vicon_robot_obj_quat": np.full(4, np.nan),
        "vicon_pelvis_pos": np.full(3, np.nan),
        "vicon_human_pos": np.full((n, 3), np.nan),
        "vicon_stale": float(self._stale),
      }
    return {**self._last_raw, "vicon_stale": float(self._stale)}
