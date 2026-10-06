"""Contact-script reference provider (SoftMimic compliance augmentation).

Data model: a SoftMimic augmented clip is **one timeline** carrying several typed
channels. Two of them are full poses, the FREE (original) face and the ADAPTED
(pushed out by the force) face, plus the contact "script" (wrench / stiffness /
link / setpoint) and the **event table schedule** that cuts the script into
segments (each event: ramp-up / hold / ramp-down phases + forced link +
stiffness). Because the augmenter displaces links only **during** force events,
``adapted == free`` and ``wrench == 0`` outside events, so:

  * tracking always reads the ADAPTED face (``get_reference``): correct on both
    free and forced frames, with no ``if event`` branch;
  * visualization reads **both faces** (``get_free`` -> the original ghost; the
    adapted face = the served reference / robot); the difference between the two
    faces is the compliant yield;
  * the contact ``contact_state(t)`` (hold-left sampled wrench / stiffness / link
    / setpoint / phase) drives the force arrows and the online ForceField
    event term.

The single-source-of-truth **pure-function physics** also lives here:
``reactive_wrench`` (reactive spring force + torque) and ``reanchor_*`` (yaw
re-anchoring frozen at the rising edge). The force event applies them and the
visualization draws them: **defined once, called everywhere**, so visualization
and physics never drift apart.

Per-clip data (all from the offline augmenter):
  free_motion_file    -- original (un-augmented) motion NPZ (FK body poses)
  adapted_motion_file -- force-displaced motion NPZ (FK body poses)
  contact_file        -- native contact NPZ: named fields indexed by key
                         [link_id, link_name, force(3), torque(3), robot_stiffness,
                         robot_rot_stiffness, forcefield_stiffness,
                         forcefield_rot_stiffness, setpoint_pos(3),
                         setpoint_quat(4, WXYZ), plane_normal(3)] per-frame
                         channels, plus the event table
                         [event_t_start/t_ramp_end/t_hold_end/t_end,
                         event_link_name, event_k_robot/k_ff/...], all in the
                         WORLD frame, written by ``write_contact_npz`` (single
                         contact) or ``write_contact_npz_multi`` (K slots).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from mjlab.tasks.codancing.motion.loaders import MotionLoader, check_body_order
from mjlab.tasks.codancing.motion.registry import MotionFrames, MotionRegistry
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_apply,
  quat_inv,
  quat_mul,
  yaw_quat,
)

# Contact phase codes (tensor-friendly integer codes).
PHASE_FREE = 0
PHASE_RAMP_UP = 1
PHASE_HOLD = 2
PHASE_RAMP_DOWN = 3


@dataclass
class ContactStateMulti:
  """multi-link contact script (each row = a query, each slot = one contact keyed by
  link).

  multi-link sibling of the single-contact ``ContactState``. The slot axis has a
  fixed size K = max_contacts; inactive slots have force=0 and link_id=-1, and
  ``active`` is derived per slot from |F|/|tau|>eps.
  """

  active: torch.Tensor  # (N,K) bool
  link_id: torch.Tensor  # (N,K) long (augmenter body id, -1=empty)
  force: torch.Tensor  # (N,K,3) WORLD
  torque: torch.Tensor  # (N,K,3) WORLD
  robot_stiffness: torch.Tensor  # (N,K)
  robot_rot_stiffness: torch.Tensor  # (N,K)
  forcefield_stiffness: torch.Tensor  # (N,K)
  forcefield_rot_stiffness: torch.Tensor  # (N,K)
  setpoint_pos: torch.Tensor  # (N,K,3)
  setpoint_quat: torch.Tensor  # (N,K,4) WXYZ
  plane_normal: torch.Tensor  # (N,K,3)
  phase: torch.Tensor  # (N,K) long


# ── Pure-function physics (single source of truth) ─────────────────────────────
# Shared by the force event and the visualization.


def delta_quat(
  robot_quat: torch.Tensor, ref_quat: torch.Tensor, rotation: str = "yaw"
) -> torch.Tensor:
  """Re-anchor delta quaternion (wxyz) of the robot anchor vs the reference anchor.

  ``rotation="yaw"`` (default) = yaw only, the same expression as the tracking
  reward (``_anchor_aligned`` in commands.py,
  ``yaw_quat(quat_mul(q_robot, quat_inv(q_ref)))``), so the force re-anchoring
  and tracking share one yaw frame. ``"full_quat"`` = the
  full quaternion delta (the AMP features convention in amp_features.py)."""
  full = quat_mul(robot_quat, quat_inv(ref_quat))
  return yaw_quat(full) if rotation == "yaw" else full


def reanchor_points(
  points: torch.Tensor,
  robot_anchor_pos: torch.Tensor,
  robot_anchor_quat: torch.Tensor,
  ref_anchor_pos: torch.Tensor,
  ref_anchor_quat: torch.Tensor,
  rotation: str = "yaw",
) -> torch.Tensor:
  """Re-anchor points from the reference/clip frame into the robot frame: robot XY
  and yaw, reference Z (height preserved).

  Same idiom as the tracking reward (``_anchor_aligned`` in commands.py): the
  base point takes
  the robot anchor's XY but the reference anchor's Z; the offset is rotated by the
  re-anchor delta. With zero drift (robot anchor == reference anchor) this reduces
  to the identity."""
  yd = delta_quat(robot_anchor_quat, ref_anchor_quat, rotation)
  base = robot_anchor_pos.clone()
  base[..., 2] = ref_anchor_pos[..., 2]
  return base + quat_apply(yd, points - ref_anchor_pos)


def reanchor_quats(
  quats: torch.Tensor,
  robot_anchor_quat: torch.Tensor,
  ref_anchor_quat: torch.Tensor,
  rotation: str = "yaw",
) -> torch.Tensor:
  """Left-multiply reference-frame orientations by the re-anchor delta to carry
  them into the robot frame (for the rotational setpoint)."""
  yd = delta_quat(robot_anchor_quat, ref_anchor_quat, rotation)
  return quat_mul(yd, quats)


def reactive_wrench(
  link_pos: torch.Tensor,
  link_quat: torch.Tensor,
  setpoint_pos: torch.Tensor,
  setpoint_quat: torch.Tensor,
  k_ff: torch.Tensor,
  k_rot_ff: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """SoftMimic reactive wrench: a single environment spring toward the
  (re-anchored) setpoint.

  ``F = K_env·(setpoint − live_link_pos)``;
  ``tau = K_rot_env·axis_angle(setpoint_quat ⊗ live_quat⁻¹)``. Geometry (consistent
  with the augmenter, setpoint = p_ref + F/K_rob + F/K_env):
    * link at the setpoint -> F = 0;
    * link at the adapted pose (= p_ref + F/K_rob, green ghost / tracking target)
      -> F = the desired force (nonzero);
    * link at p_ref -> F = desired force·(1 + K_env/K_rob).
  Pose tracking (to adapted) and force matching (applied=desired) **both peak at
  the adapted pose**: the policy holds the desired force there. **No clamping**
  (faithful to SoftMimic; the feasibility-limited setpoint + the fall termination
  are the safety net)."""
  f = k_ff.unsqueeze(-1) * (setpoint_pos - link_pos)
  dq = quat_mul(setpoint_quat, quat_inv(link_quat))
  tau = k_rot_ff.unsqueeze(-1) * axis_angle_from_quat(dq)
  return f, tau


class ContactReferenceProvider:
  """Two-face (free / adapted) reference + contact script + event table. Implements
  ReferenceProvider."""

  def __init__(
    self,
    free_loaders: list[MotionLoader],
    adapted_loaders: list[MotionLoader],
    contact_channels: dict[str, torch.Tensor],  # flat pool, frame-aligned to adapted
    control_dt: float,
    device: str,
    *,
    events: dict[str, torch.Tensor],  # flat event table (across clips)
    event_link_names: list[str],  # forced link name per global event
    active_force_eps: float = 1e-2,
    adapted_registry: MotionRegistry | None = None,
    slot_link_names: list[str] | None = None,  # K slots: fixed link per slot
  ) -> None:
    self.device = device
    self._free = MotionRegistry(free_loaders, control_dt, device)
    # The adapted face is the served clip pool: reuse the registry the command
    # already built if given (no double loading), otherwise build one.
    self._adapted = adapted_registry or MotionRegistry(
      adapted_loaders, control_dt, device
    )
    self._ch = contact_channels
    self._eps = active_force_eps
    # Reuse the adapted registry's flat-pool geometry for hold-left contact lookups.
    self._num_frames = self._adapted.motion_num_frames
    self._dt = self._adapted.motion_dt
    self._lengths = self._adapted.motion_lengths
    self._starts = self._adapted.length_starts
    # Event table (the ContactEvent schedule). Event times are stored as **frame
    # indices**; here they are converted to seconds with each clip's real
    # motion_dt (the runtime time base = the adapted NPZ's fps, independent of the
    # augmenter's sim/recording rate).
    self._ev = _resolve_event_seconds(events, self._dt)
    self.event_link_names: list[str] = event_link_names
    # K slots: non-empty slot_link_names -> multi. Each slot is bound to one fixed
    # link; _event_slot maps each global event to a slot index by link name (in no
    # slot = -1). K=1 reduces to a single slot.
    self._slot_link_names: list[str] | None = (
      list(slot_link_names) if slot_link_names else None
    )
    self._K: int = len(self._slot_link_names) if self._slot_link_names else 1
    if self._slot_link_names is not None:
      slot_of = {n: i for i, n in enumerate(self._slot_link_names)}
      self._event_slot = torch.tensor(
        [slot_of.get(n, -1) for n in self.event_link_names],
        dtype=torch.long,
        device=self.device,
      )
    else:
      self._event_slot = None

  # ── ReferenceProvider protocol ───────────────────────────────────────────────
  def get_reference(
    self, motion_ids: torch.Tensor, motion_times: torch.Tensor
  ) -> MotionFrames:
    """The ADAPTED (force-displaced) face: the tracking target."""
    return self._adapted.get_frames(motion_ids, motion_times)

  def num_frames(self, motion_ids: torch.Tensor) -> torch.Tensor:
    return self._adapted.motion_num_frames[motion_ids].float()

  # ── Visualization / contact extras ───────────────────────────────────────────
  def get_free(
    self, motion_ids: torch.Tensor, motion_times: torch.Tensor
  ) -> MotionFrames:
    """The FREE (original, un-yielded) face: for the original reference ghost."""
    return self._free.get_frames(motion_ids, motion_times)

  def _hold_left_frames(
    self, motion_ids: torch.Tensor, motion_times: torch.Tensor
  ) -> torch.Tensor:
    """Global nearest-left frame index into the contact channels (no
    interpolation)."""
    dt = self._dt[motion_ids]
    t = motion_times.clamp(min=0.0).minimum(self._lengths[motion_ids])
    # 1e-3 frame tolerance: the cursor accumulates float32 control steps, so
    # at frame nodes it sits a hair BELOW n*dt and a bare floor() resolves to
    # frame n-1 while the (lerped) motion serve is already at frame n. That
    # one-frame shear between the contact and motion channels scales with
    # link speed times K_env (measured up to about 11 N on stiff events).
    # Shifting the hold-left boundary by 0.1 percent of a frame (20 us of
    # clip time) removes the shear without changing event semantics.
    f0 = (t / dt + 1e-3).floor().long().clamp(max=self._num_frames[motion_ids] - 1)
    return self._starts[motion_ids] + f0

  # ── Event table (stateless rising edge + links by name + phase) ─────────────

  # ── multi-link accessors (multi-contact clips) ───────────────────────────────────
  def slot_link_robot_ids(self, robot_body_names: list[str]) -> torch.Tensor:
    """Fixed link of each slot -> robot body id (resolved by name; -1 if not on the
    robot). (K,)"""
    names = self._slot_link_names or []
    ids = [robot_body_names.index(n) if n in robot_body_names else -1 for n in names]
    return torch.tensor(ids, dtype=torch.long, device=self.device)

  def event_indices_at(
    self, motion_ids: torch.Tensor, motion_times: torch.Tensor
  ) -> torch.Tensor:
    """Global index of each slot's active event (N,K); -1 where slot k has no
    active event.

    Each column is the link-filtered version of the single-contact
    ``event_index_at``: slot k only matches events with ``event_slot == k``. The
    precondition (≤1 active event per link per frame) makes each column
    single-valued."""
    if self._event_slot is None:
      raise ValueError("event_indices_at needs slot_link_names (a multi provider).")
    return _event_indices_at(
      self._ev, self._event_slot, motion_ids, motion_times, self._K, self.device
    )

  def phase_at_multi(
    self, motion_ids: torch.Tensor, motion_times: torch.Tensor
  ) -> torch.Tensor:
    """Per-slot contact phase code (N,K) (free/ramp_up/hold/ramp_down)."""
    idx = self.event_indices_at(motion_ids, motion_times)
    return _phase_from_events(self._ev, idx, motion_times)

  def contact_state_multi(
    self, motion_ids: torch.Tensor, motion_times: torch.Tensor
  ) -> ContactStateMulti:
    """multi-link per-env contact script (hold-left sampled). The channel pool is
    (G,K,·), so the gather yields (N,K,·) for free."""
    g = self._hold_left_frames(motion_ids, motion_times)
    force = self._ch["force"][g]  # (N,K,3)
    torque = self._ch["torque"][g]
    active = (force.norm(dim=-1) > self._eps) | (torque.norm(dim=-1) > self._eps)
    return ContactStateMulti(
      active=active,
      link_id=self._ch["link_id"][g].long(),
      force=force,
      torque=torque,
      robot_stiffness=self._ch["robot_stiffness"][g],
      robot_rot_stiffness=self._ch["robot_rot_stiffness"][g],
      forcefield_stiffness=self._ch["forcefield_stiffness"][g],
      forcefield_rot_stiffness=self._ch["forcefield_rot_stiffness"][g],
      setpoint_pos=self._ch["setpoint_pos"][g],
      setpoint_quat=self._ch["setpoint_quat"][g],
      plane_normal=self._ch["plane_normal"][g],
      phase=self.phase_at_multi(motion_ids, motion_times),
    )

  # ── Unified multi-link view (single contact = K=1) ───────────────────────────────
  # The only read path for the visualization.
  @property
  def num_slots(self) -> int:
    """Slot count K. A single-contact provider has 1 (it is just the K=1 case)."""
    return self._K

  def hold_windows(self) -> torch.Tensor:
    """(E, 3) rows ``[clip, t_ramp_end, t_hold_end]`` from the event table: the
    HOLD phase of every force event, in that clip's own seconds. The `rsi`
    phase sampler's `force_event` term puts its mass on these windows."""
    ev = self._ev
    return torch.stack([ev["clip"].float(), ev["t_ramp_end"], ev["t_hold_end"]], dim=1)

  def contact_state_slots(
    self, motion_ids: torch.Tensor, motion_times: torch.Tensor
  ) -> tuple[ContactStateMulti, torch.Tensor]:
    """multi-link-unified contact script + per-slot event indices: a single-contact
    provider also returns (N,1,·) / (N,1).

    The visualization goes through this one path only; single contact is just
    K=1, so there is no second way of drawing. The forced link per (env, slot) is
    resolved from the **active event** (``event_link_names[event_index]``), not
    from the slot's fixed name: in single-contact clips the link changes with the
    event (waltz left wrist -> right wrist), in multi-link clips the two agree anyway,
    so one rule covers both."""
    return (
      self.contact_state_multi(motion_ids, motion_times),
      self.event_indices_at(motion_ids, motion_times),
    )


# ── Native contact NPZ format (per-frame channels + event table) ───────────────
# mjlab conventions: WORLD-frame force/torque, ``setpoint_quat`` in WXYZ, named
# fields indexed by key. ``link_id`` and ``link_name`` are stored per frame for
# inspection; the provider resolves the forced link by name from the event table.
_NPZ_CHANNELS: tuple[str, ...] = (
  "link_id",
  "force",
  "torque",
  "robot_stiffness",
  "robot_rot_stiffness",
  "forcefield_stiffness",
  "forcefield_rot_stiffness",
  "setpoint_pos",
  "setpoint_quat",
  "plane_normal",
)
# Event table fields (inside each clip's NPZ, length = that clip's event count).
# Times are stored as **frame indices** (integers) and converted to seconds by the
# provider with the clip's real motion_dt, which makes them independent of the
# augmenter's sim/recording rate (forced frames map 1:1 onto the adapted NPZ, and
# the runtime time base = the adapted NPZ's fps), removing any fps assumption.
_EVENT_FRAME_KEYS: tuple[str, ...] = ("f_start", "f_ramp_end", "f_hold_end", "f_end")
_EVENT_K_KEYS: tuple[str, ...] = ("k_robot", "k_ff", "k_rot_robot", "k_rot_ff")
_EVENT_KEYS: tuple[str, ...] = _EVENT_FRAME_KEYS + _EVENT_K_KEYS


def _contiguous_spans(mask: np.ndarray) -> list[tuple[int, int]]:
  """Contiguous True spans [f0, f1) of a boolean mask (f1 exclusive)."""
  spans: list[tuple[int, int]] = []
  i, n = 0, len(mask)
  while i < n:
    if mask[i]:
      j = i
      while j < n and mask[j]:
        j += 1
      spans.append((i, j))
      i = j
    else:
      i += 1
  return spans


def _event_indices_at(
  ev: dict[str, torch.Tensor],
  event_slot: torch.Tensor,
  motion_ids: torch.Tensor,
  motion_times: torch.Tensor,
  k: int,
  device: str,
) -> torch.Tensor:
  """Global index of each slot's active event (N,K), -1 if none (K slots).

  ``event_slot`` (E,) maps each global event to a slot index. Each slot = the
  single-contact time match + argmax first match, with the extra constraint
  ``event_slot == slot``. K is small (2), so the Python loop over slots is cheap."""
  n = motion_ids.numel()
  none = torch.full((n, k), -1, dtype=torch.long, device=device)
  if ev["clip"].numel() == 0:
    return none
  mids = motion_ids.view(-1, 1)
  t = motion_times.view(-1, 1)
  base = (
    (ev["clip"].view(1, -1) == mids)
    & (ev["t_start"].view(1, -1) <= t)
    & (t < ev["t_end"].view(1, -1))
  )  # (n, E)
  out = none.clone()
  for s in range(k):
    m = base & (event_slot == s).view(1, -1)
    has = m.any(dim=1)
    idx = m.float().argmax(dim=1).long()
    out[:, s] = torch.where(has, idx, none[:, s])
  return out


def _phase_from_events(
  ev: dict[str, torch.Tensor], ev_idx: torch.Tensor, motion_times: torch.Tensor
) -> torch.Tensor:
  """Per-slot phase (N,K): decoded from the ramp/hold boundaries of each slot's
  active event (ev_idx>=0); free slots get PHASE_FREE."""
  phase = torch.full_like(ev_idx, PHASE_FREE)
  valid = ev_idx >= 0
  if not bool(valid.any()):
    return phase
  e = ev_idx.clamp(min=0)
  ramp_end = ev["t_ramp_end"][e]
  hold_end = ev["t_hold_end"][e]
  t = motion_times.view(-1, 1)
  p = torch.where(
    t < ramp_end,
    torch.full_like(ev_idx, PHASE_RAMP_UP),
    torch.where(
      t < hold_end,
      torch.full_like(ev_idx, PHASE_HOLD),
      torch.full_like(ev_idx, PHASE_RAMP_DOWN),
    ),
  )
  return torch.where(valid, p, phase)


def _events_to_tensors(rows: list[dict], device: str) -> dict[str, torch.Tensor]:
  """List of frame-indexed event dicts -> flat tensor table (an empty table still
  gets correct zero-length keys)."""
  clip = torch.tensor([r["clip"] for r in rows], dtype=torch.long, device=device)
  out = {"clip": clip}
  for k in _EVENT_KEYS:
    out[k] = torch.tensor([r[k] for r in rows], dtype=torch.float32, device=device)
  return out


def _resolve_event_seconds(
  raw: dict[str, torch.Tensor], dt_per_clip: torch.Tensor
) -> dict[str, torch.Tensor]:
  """Frame-indexed event table -> seconds (converted with each clip's motion_dt).
  clip + k_* are carried through unchanged."""
  clip = raw["clip"]
  dt = dt_per_clip[clip] if clip.numel() else dt_per_clip.new_zeros(0)
  out: dict[str, torch.Tensor] = {"clip": clip}
  for fk, tk in (
    ("f_start", "t_start"),
    ("f_ramp_end", "t_ramp_end"),
    ("f_hold_end", "t_hold_end"),
    ("f_end", "t_end"),
  ):
    out[tk] = raw[fk] * dt
  for k in _EVENT_K_KEYS:
    out[k] = raw[k]
  return out


def _load_contact_channels(
  npz_paths: list[str], num_frames: torch.Tensor, device: str
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[str]]:
  """Load the named per-frame channels of native contact NPZs (flat pool) + the
  cross-clip event table.

  Each NPZ's per-frame channels are cut to that clip's ``num_frames``, so the flat
  pool stays frame-aligned with the adapted registry.
  """
  per_clip: dict[str, list[torch.Tensor]] = {k: [] for k in _NPZ_CHANNELS}
  ev_rows: list[dict] = []
  ev_names: list[str] = []
  for clip, (path, nf) in enumerate(zip(npz_paths, num_frames.tolist(), strict=True)):
    d = np.load(path, allow_pickle=False)
    for k in _NPZ_CHANNELS:
      v = torch.from_numpy(np.asarray(d[k])[:nf]).to(torch.float32).to(device)
      per_clip[k].append(v)
    ev_names.extend(str(s) for s in np.asarray(d["event_link_name"]))
    for j in range(len(np.asarray(d["event_f_start"]))):
      ev_rows.append(
        {
          "clip": clip,
          **{k: float(np.asarray(d[f"event_{k}"])[j]) for k in _EVENT_KEYS},
        }
      )
  channels = {k: torch.cat(v, dim=0) for k, v in per_clip.items()}
  return channels, _events_to_tensors(ev_rows, device), ev_names


def _read_slot_link_names(npz_paths: list[str]) -> list[str] | None:
  """Read ``slot_link_names`` (the multi-link marker) from the contact NPZs.

  If any clip carries this key, the clip set is treated as multi-link (multi) and the
  list of slot link names is returned; single-contact NPZs lack the key -> None,
  and the provider takes the single-contact path.

  Clips that declare the key must **all agree**: the slot table is resolved once
  on the provider and shared by the whole pool (``slot_link_robot_ids`` is also
  cached on the command), so if only the first clip's table were taken here, a
  mixed pool would silently map forces onto the wrong bodies, and the physics
  would be wrong before any reward is evaluated. So every clip is compared at
  load time, and a mismatch raises on the spot, naming the conflicting files."""
  first: list[str] | None = None
  first_path: str | None = None
  for path in npz_paths:
    d = np.load(path, allow_pickle=False)
    if "slot_link_names" not in d:
      continue
    names = [str(s) for s in np.asarray(d["slot_link_names"])]
    if first is None:
      first, first_path = names, path
    elif names != first:
      raise ValueError(
        "slot_link_names disagree across the clip pool: "
        f"{first_path} declares {first} but {path} declares {names}. "
        "The slot table is resolved once for the whole pool, so a mixed "
        "pool would silently map forces onto the wrong bodies."
      )
  return first


def write_contact_npz(
  npz_path: str,
  *,
  link_id: np.ndarray,
  link_name: np.ndarray,
  force: np.ndarray,
  torque: np.ndarray,
  robot_stiffness: np.ndarray,
  robot_rot_stiffness: np.ndarray,
  forcefield_stiffness: np.ndarray,
  forcefield_rot_stiffness: np.ndarray,
  setpoint_pos: np.ndarray,
  setpoint_quat_wxyz: np.ndarray,
  plane_normal: np.ndarray,
  events: list[dict],
) -> None:
  """The single-contact native NPZ writer (the one place this format is defined).

  ``setpoint_quat_wxyz`` must already be WXYZ (callers convert scipy-xyzw -> wxyz
  at the source). ``events`` is a list of per-event dicts
  (f_start/f_ramp_end/f_hold_end/f_end **frame indices** + link_name + the Ks);
  the provider converts the time base with the clip's motion_dt, so only frames
  are stored here.
  """
  # One dict, then unpack: Any values keep pyright from misreading the dict
  # unpacking as np.savez's allow_pickle keyword.
  arrays: dict[str, Any] = {
    "link_id": link_id,
    "link_name": link_name,
    "force": force,
    "torque": torque,
    "robot_stiffness": robot_stiffness,
    "robot_rot_stiffness": robot_rot_stiffness,
    "forcefield_stiffness": forcefield_stiffness,
    "forcefield_rot_stiffness": forcefield_rot_stiffness,
    "setpoint_pos": setpoint_pos,
    "setpoint_quat": setpoint_quat_wxyz,
    "plane_normal": plane_normal,
    "event_link_name": np.array([e["link_name"] for e in events]),
  }
  for k in _EVENT_FRAME_KEYS:
    arrays[f"event_{k}"] = np.array([e[k] for e in events], dtype=np.int64)
  for k in _EVENT_K_KEYS:
    arrays[f"event_{k}"] = np.array([e[k] for e in events], dtype=np.float64)
  np.savez(npz_path, **arrays)


def write_contact_npz_multi(
  npz_path: str,
  *,
  link_id: np.ndarray,  # (T,K)
  link_name: np.ndarray,  # (T,K)
  force: np.ndarray,  # (T,K,3)
  torque: np.ndarray,  # (T,K,3)
  robot_stiffness: np.ndarray,  # (T,K)
  robot_rot_stiffness: np.ndarray,  # (T,K)
  forcefield_stiffness: np.ndarray,  # (T,K)
  forcefield_rot_stiffness: np.ndarray,  # (T,K)
  setpoint_pos: np.ndarray,  # (T,K,3)
  setpoint_quat_wxyz: np.ndarray,  # (T,K,4) already WXYZ
  plane_normal: np.ndarray,  # (T,K,3)
  slot_link_names: np.ndarray,  # (K,) fixed link name per slot
  events: list[dict],  # all slots' events in one flat table (each has link_name)
  run_meta: str
  | None = None,  # optional provenance JSON (datagen_meta.build_datagen_meta)
  # Optional bookkeeping channels ((T,K) or the same shape).
  extra_channels: dict[str, np.ndarray] | None = None,
) -> None:
  """multi-link native contact NPZ writer (format = single contact plus a slot axis +
  ``slot_link_names``).

  Same keys as ``write_contact_npz``, with an extra K axis on the per-frame
  channels; ``slot_link_names`` binds slot k to a fixed link (the read side uses
  it to map events to slots).

  ``extra_channels`` holds the producer's bookkeeping channels (sampling intent,
  guidance branch, fallback accounting, etc., for auditing and visualization):
  the training read side takes keys from a fixed allowlist, so extra keys are
  inert and affect no consumer. Key names must not clash with the built-in
  keys."""
  arrays: dict[str, Any] = {
    "link_id": link_id,
    "link_name": link_name,
    "force": force,
    "torque": torque,
    "robot_stiffness": robot_stiffness,
    "robot_rot_stiffness": robot_rot_stiffness,
    "forcefield_stiffness": forcefield_stiffness,
    "forcefield_rot_stiffness": forcefield_rot_stiffness,
    "setpoint_pos": setpoint_pos,
    "setpoint_quat": setpoint_quat_wxyz,
    "plane_normal": plane_normal,
    "slot_link_names": np.asarray(slot_link_names),
    "event_link_name": np.array([e["link_name"] for e in events]),
  }
  for k in _EVENT_FRAME_KEYS:
    arrays[f"event_{k}"] = np.array([e[k] for e in events], dtype=np.int64)
  for k in _EVENT_K_KEYS:
    arrays[f"event_{k}"] = np.array([e[k] for e in events], dtype=np.float64)
  if run_meta is not None:
    arrays["run_meta"] = run_meta
  if extra_channels:
    clash = set(extra_channels) & set(arrays)
    assert not clash, f"extra_channels clash with built-in keys: {sorted(clash)}"
    arrays.update(extra_channels)
  np.savez(npz_path, **arrays)


def derive_contact_events_multi(
  link_id: np.ndarray,  # (T,K)
  force: np.ndarray,  # (T,K,3)
  torque: np.ndarray,  # (T,K,3)
  robot_stiffness: np.ndarray,  # (T,K)
  robot_rot_stiffness: np.ndarray,  # (T,K)
  forcefield_stiffness: np.ndarray,  # (T,K)
  forcefield_rot_stiffness: np.ndarray,  # (T,K)
  id2name: dict[int, str],
  eps: float = 1e-2,
) -> list[dict]:
  """Run ``derive_contact_events`` per slot, then merge into one flat table
  (multi-link event derivation).

  Under the precondition (≤1 active event per link per frame) each slot is
  naturally single-valued, and events of different slots never overlap on the
  same link."""
  k = link_id.shape[1]
  events: list[dict] = []
  for s in range(k):
    events.extend(
      derive_contact_events(
        link_id[:, s],
        force[:, s],
        torque[:, s],
        robot_stiffness[:, s],
        robot_rot_stiffness[:, s],
        forcefield_stiffness[:, s],
        forcefield_rot_stiffness[:, s],
        id2name,
        eps,
      )
    )
  return events


def derive_contact_events(
  link_id: np.ndarray,
  force: np.ndarray,
  torque: np.ndarray,
  robot_stiffness: np.ndarray,
  robot_rot_stiffness: np.ndarray,
  forcefield_stiffness: np.ndarray,
  forcefield_rot_stiffness: np.ndarray,
  id2name: dict[int, str],
  eps: float = 1e-2,
) -> list[dict]:
  """Derive the event table from contiguous active spans of per-frame arrays (the
  augmenters call it on their own channels, so the table agrees with them by
  construction).

  Times are stored as **frame indices** (f_start/f_ramp_end/f_hold_end/f_end); the
  ramp/hold boundaries are estimated from whether |F| reaches 95% of its peak; in
  degenerate cases the whole span is taken as hold.
  """
  fmag = np.linalg.norm(force, axis=-1)
  tmag = np.linalg.norm(torque, axis=-1)
  active = (fmag > eps) | (tmag > eps)
  events: list[dict] = []
  for f0, f1 in _contiguous_spans(active):
    seg = slice(f0, f1)
    fseg = fmag[seg]
    peak = float(fseg.max()) if fseg.size else 0.0
    hot = fseg >= 0.95 * peak if peak > 0 else np.ones_like(fseg, dtype=bool)
    ramp_end = f0 + (int(np.argmax(hot)) if hot.any() else 0)
    hold_end = f0 + (int(len(hot) - np.argmax(hot[::-1])) if hot.any() else (f1 - f0))
    lid = int(np.bincount(np.maximum(link_id[seg], 0)).argmax())
    events.append(
      {
        "f_start": f0,
        "f_ramp_end": ramp_end,
        "f_hold_end": hold_end,
        "f_end": f1,
        "link_name": id2name.get(lid, ""),
        "k_robot": float(np.median(robot_stiffness[seg])),
        "k_ff": float(np.median(forcefield_stiffness[seg])),
        "k_rot_robot": float(np.median(robot_rot_stiffness[seg])),
        "k_rot_ff": float(np.median(forcefield_rot_stiffness[seg])),
      }
    )
  return events


def build_contact_provider_from_clips(
  clips,  # Sequence[MotionClip | PairedMotionClip] with free_motion_file + contact_file
  adapted_registry: MotionRegistry,
  control_dt: float,
  device: str = "cpu",
  expected_body_names: tuple[str, ...] | None = None,
) -> ContactReferenceProvider:
  """Build the provider directly from self-describing manifest clips.

  The clips carry their own visualization/force data: ``robot_motion_file`` is the
  ADAPTED (served) face (reused via ``adapted_registry``), ``free_motion_file`` is
  the FREE face, and ``contact_file`` is the native contact NPZ (named fields,
  WORLD frame, wxyz setpoint_quat, event table). So visualization + the force
  event read the **clips**, not a separate config. Every clip must carry both
  files (the cursor's motion_id indexes the whole pool).
  """
  free = [MotionLoader(c.free_motion_file, device=device) for c in clips]
  if expected_body_names is not None:
    # FREE-face body rows must be MODEL order like every other npz; an npz that
    # records body_names turns the assumption into a check.
    for clip, loader in zip(clips, free, strict=True):
      check_body_order(loader.body_names, expected_body_names, clip.free_motion_file)
  contact_paths = [c.contact_file for c in clips]
  channels, events, event_link_names = _load_contact_channels(
    contact_paths, adapted_registry.motion_num_frames, device
  )
  return ContactReferenceProvider(
    free,
    [],  # adapted comes from adapted_registry (reused, not rebuilt)
    channels,
    control_dt,
    device,
    adapted_registry=adapted_registry,
    events=events,
    event_link_names=event_link_names,
    slot_link_names=_read_slot_link_names(contact_paths),  # non-None if multi-link
  )


def _self_check() -> None:
  """Assert the geometry of the reactive force law and the re-anchor identity
  (no data needed)."""
  k_rob, k_env = 200.0, 100.0
  f = torch.tensor([[80.0, 0.0, 0.0]])
  p_ref = torch.zeros(1, 3)
  adapted = p_ref + f / k_rob  # tracking target (green ghost), +0.40 m
  setpoint = p_ref + f / k_rob + f / k_env  # magenta, +1.20 m (F/K_env past adapted)
  q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
  k_ff = torch.tensor([k_env])
  k_rot = torch.tensor([0.0])
  f_at_set, _ = reactive_wrench(setpoint, q, setpoint, q, k_ff, k_rot)
  f_at_adapt, _ = reactive_wrench(adapted, q, setpoint, q, k_ff, k_rot)
  f_at_ref, _ = reactive_wrench(p_ref, q, setpoint, q, k_ff, k_rot)
  assert f_at_set.norm() < 1e-4, "force at the setpoint should be 0"
  assert abs(f_at_adapt.norm().item() - 80.0) < 1e-3, (
    "force at the adapted pose should equal the desired force, 80 N"
  )
  assert abs(f_at_ref.norm().item() - 80.0 * (1 + k_env / k_rob)) < 1e-3, (
    "force at p_ref should equal F·(1+K_env/K_rob)"
  )
  # Re-anchoring: zero drift -> identity; Z takes the reference anchor's Z (height
  # preserved).
  ap = torch.tensor([[1.0, 2.0, 0.7]])
  aq = torch.tensor([[0.7071, 0.0, 0.0, 0.7071]])  # yaw 90°
  pts = torch.tensor([[1.5, 2.5, 1.3]])
  assert torch.allclose(reanchor_points(pts, ap, aq, ap, aq), pts, atol=1e-5), (
    "zero-drift re-anchoring should be the identity"
  )
  # Robot anchor XYZ each +1: XY follows the robot (the setpoint shifts by +1), Z
  # drops the robot drift (the point keeps its own Z).
  drifted = reanchor_points(pts, ap + 1.0, aq, ap, aq)
  assert torch.allclose(drifted[0, :2], pts[0, :2] + 1.0, atol=1e-5), (
    "re-anchored XY should follow the robot anchor"
  )
  assert abs(drifted[0, 2].item() - pts[0, 2].item()) < 1e-5, (
    "re-anchoring should ignore robot Z drift (keep the point's Z)"
  )
  print("OK: reactive force geometry + re-anchor identity pass.")


if __name__ == "__main__":
  _self_check()
