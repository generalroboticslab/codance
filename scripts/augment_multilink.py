"""Multi-link SoftMimic offline augmentation: K-contact motion augmentation along the
reference trajectory + the zero-wrench contact face.

A standalone script that reuses G1_Mink_IK_Solver / constants / single-link reference
functions from softmimic_mink_augment.py (import only, no changes). The multi-contact
CoM target / feasibility gate / sampler / IK step match the single-link version bit
for bit at K=1.

Usage:
  uv run python scripts/augment_multilink.py motion --motion <origin>.npz --out adapted.csv
  uv run python scripts/augment_multilink.py zerowrench --motion <origin>.npz --out zw.npz
  # after motion: qpos_csv_to_motion_npz.py makes adapted.csv a motion-library NPZ.
"""

import copy
import json
import os
import sys
from dataclasses import dataclass

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from mjlab.tasks.codancing.datagen_meta import build_datagen_meta
from mjlab.tasks.codancing.motion.contact_reference import (
  _contiguous_spans,
  derive_contact_events_multi,
  write_contact_npz_multi,
)

# Put the script's directory on sys.path to import the sibling softmimic_mink_augment.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import softmimic_mink_augment as solver_module  # noqa: E402, F401  (needs sys.path set)

# Canonical order of the G1's 29 actuated joints (identical to
# qpos_csv_to_motion_npz.py: left leg 6 → right leg 6 → waist 3 → left arm 7 → right
# arm 7). The pose library and the motion library both use this order.
G1_JOINT_NAMES = [
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
  "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
]  # fmt: skip

# Both wrists: the bimanual sampler only acts on these two (as in FORCEABLE_LINKS).
WRISTS = ["left_wrist_yaw_link", "right_wrist_yaw_link"]


@dataclass
class Contact:
  """One multi-link contact: forced link + world-frame force/torque wrench + robot
  stiffness + force-field (environment) stiffness."""

  link_name: str
  force: np.ndarray  # (3,) external force (N), world frame
  torque: np.ndarray  # (3,) external torque (N·m), world frame
  stiffness: float  # K_rob (N/m): robot position spring; target = ref + F/K_rob
  rot_stiffness: float  # K_rot_rob (N·m/rad): robot rotational spring
  # Force-field (environment) stiffness: used for the setpoint in the contact log
  # (series springs: setpoint = ref + F/K_env + F/K_rob). The defaults keep the
  # positional construction Contact(name, f, tau, k, k_rot) valid.
  ff_stiffness: float = 100.0  # K_env (N/m): force-field position spring
  ff_rot_stiffness: float = 1.0  # K_rot_env (N·m/rad): force-field rotational spring
  intent: str = ""  # bimanual intent (along/open/twist) for audit; empty if independent
  # Per-frame audit (filled from events by active_contacts_at; goes only into the
  # contact NPZ's visualization/audit channels, which the training reader, taking keys
  # from an allowlist, never consumes):
  pair_key: float = -1.0  # schedule time shared by a sampled pair; -1 = no source event
  guide_branch: str = ""  # push / minor / pause / none
  rot_deg: float = 0.0  # accumulated backoff outward rotation (degrees)
  ev_scale: float = 1.0  # whole-event backoff scale (0 = abandoned, never active)


def forward_ref(model: mujoco.MjModel, qpos_ref: np.ndarray) -> mujoco.MjData:
  """Temporary MjData forwarded at qpos_ref, for link world poses and the CoM."""
  data = mujoco.MjData(model)
  data.qpos[:] = qpos_ref
  mujoco.mj_forward(model, data)
  return data


# Penetration depth tolerance (meters) of the self-collision criterion: for geom pairs
# **new** relative to the reference, a deepest penetration ≤ this value counts as a
# graze and passes.
# A constant, not a CLI knob: 5mm (invisible to the eye, far below the scale of yield
# displacements) until the online consumer's real tolerance of light self-contact is
# measured. Judging per geom pair + depth, not by contact count: depth
# shrinks monotonically with force, so the whole-event backoff's force-shrinking
# ladder works on self-collision failures.
SELFCOL_NEW_DEPTH_TOL = 0.005


def _selfcollision_pair_depths(data) -> dict[tuple[int, int], float]:
  """Ordered geom pair -> that pair's deepest penetration (meters). dist ≥ 0 (no
  penetration) counts as 0."""
  out: dict[tuple[int, int], float] = {}
  for i in range(data.ncon):
    c = data.contact[i]
    key = (min(int(c.geom1), int(c.geom2)), max(int(c.geom1), int(c.geom2)))
    out[key] = max(out.get(key, 0.0), max(0.0, -float(c.dist)))
  return out


def new_selfcollision_depth(ik_data, ref_data) -> float:
  """Deepest penetration (meters) among the self-collision geom pairs the IK solution
  adds **relative to the reference**; 0 when no pair is new.

  Identity is judged per geom pair, not by ncon count: a count both misses cases (the
  yield resolves a pair the reference had while adding another, so the count stays
  level and passes) and kills wrongly (one geom pair gains an extra contact point from
  a small pose change, with no new pair). Pairs the reference already has are never
  charged to the augmentation, however deep (the relative criterion is kept at the
  pair level).
  The granularity is the geom pair; if adjacent geoms between the same two bodies
  "swap pairs" and cause false alarms, group by body pair instead.
  """
  ref_pairs = _selfcollision_pair_depths(ref_data)
  ik_pairs = _selfcollision_pair_depths(ik_data)
  return max((d for k, d in ik_pairs.items() if k not in ref_pairs), default=0.0)


def multilink_cop_aware_com_target(
  ref_data: mujoco.MjData, total_mass: float, contacts: list["Contact"]
) -> np.ndarray:
  """Multi-contact COP-aware CoM target: sum the overturning moments of all contacts,
  then shift the CoM in the horizontal plane to cancel them.

  Net moment M = Σᵢ (lever_i × F_i) + Σᵢ τ_i, where lever_i = force point − foot
  midpoint (desired COP). CoM shift Δ = (−M_y, M_x)/(m·g): gravity mg cancels the net
  moment instead of bracing against it. Empty contacts → no external force, return the
  reference CoM. At K=1 (single contact) this matches the single-link
  calculate_cop_aware_com_target bit for bit.
  """
  left = ref_data.body("left_ankle_roll_link").xpos
  right = ref_data.body("right_ankle_roll_link").xpos
  target_cop = (left + right) / 2.0

  total_moment = np.zeros(3)
  for c in contacts:
    lever = ref_data.body(c.link_name).xpos - target_cop  # lever arm about the COP
    total_moment += np.cross(lever, c.force) + c.torque  # lever × F + torque, summed

  mg = total_mass * 9.81
  ref_com = ref_data.subtree_com[0]
  if mg < 1e-3:
    return ref_com
  return ref_com + np.array([-total_moment[1] / mg, total_moment[0] / mg, 0.0])


def is_multilink_feasible(
  ik_data: mujoco.MjData,
  ref_data: mujoco.MjData,
  contacts: list["Contact"],
  total_mass: float,
  *,
  check_self_collision: bool = True,
) -> tuple[bool, dict[str, float]]:
  """Multi-contact feasibility gate: per-contact checks (force / displacement / torque
  / tracking error), then foot displacement + aggregate CoM + self-collision.

  Any limit exceeded means infeasible. At K=1 this reproduces the single-link
  is_ik_solution_feasible decision (self-collision aside).
  """
  sm = solver_module
  violations: dict[str, float] = {}
  ok = True

  # Per contact: force magnitude / spring displacement F/K / torque / rotational
  # displacement / tracking error of that link.
  for i, c in enumerate(contacts):
    fmag = float(np.linalg.norm(c.force))
    if fmag > sm.IK_CHECK_MAX_FORCE_MAGNITUDE:
      ok = False
    disp = fmag / c.stiffness if c.stiffness > 1e-6 else 0.0
    if disp > sm.IK_CHECK_MAX_DISPLACEMENT_MAGNITUDE:
      ok = False
    tmag = float(np.linalg.norm(c.torque))
    if tmag > sm.IK_CHECK_MAX_TORQUE_MAGNITUDE:
      ok = False
    rdisp = tmag / c.rot_stiffness if c.rot_stiffness > 1e-6 else 0.0
    if rdisp > sm.IK_CHECK_MAX_ROTATIONAL_DISPLACEMENT_MAGNITUDE:
      ok = False
    p_ref = ref_data.body(c.link_name).xpos
    p_target = p_ref + c.force / max(c.stiffness, 1e-6)
    terr = float(np.linalg.norm(ik_data.body(c.link_name).xpos - p_target))
    violations[f"track_{i}"] = terr
    if terr > sm.IK_CHECK_MAX_IK_TRACKING_ERROR:
      ok = False

  # Both feet must stay essentially still.
  max_foot = 0.0
  for foot in sm.FOOT_NAMES:
    d = float(np.linalg.norm(ik_data.body(foot).xpos - ref_data.body(foot).xpos))
    max_foot = max(max_foot, d)
  violations["max_foot_disp"] = max_foot
  if max_foot > sm.IK_CHECK_MAX_FOOT_DISP:
    ok = False

  # Aggregate CoM XY: the solved CoM vs the multi-contact COP target.
  com_target = multilink_cop_aware_com_target(ref_data, total_mass, contacts)
  com_err = float(np.linalg.norm(ik_data.subtree_com[0][:2] - com_target[:2]))
  violations["com_err"] = com_err
  if com_err > sm.IK_CHECK_MAX_COM_TRACKING_ERROR:
    ok = False

  # Self-collision: in the robot-only model every contact after mj_forward is a
  # self-contact (no ground). Self-collision contacts fire in this model, so no
  # fallback is needed.
  #
  # Criterion = geom pairs **new** relative to the reference + a penetration depth
  # tolerance (see new_selfcollision_depth): the reference itself has self-contacts
  # (measured on waltz with zero force and no IK: 64 of 487 frames have contacts), and
  # its pairs are not charged to the augmentation; a new pair only fails when its
  # penetration exceeds SELFCOL_NEW_DEPTH_TOL. A contact-count criterion ("more than
  # the reference fails") is **not monotonic** in force shrinking: a contact pushed in
  # along a direction has ncon 1 however small the force gets, so the whole-event
  # backoff ladder would spin until the event wore away; depth shrinks with force, so
  # the ladder converges and grazes pass.
  if check_self_collision:
    violations["ncon"] = float(ik_data.ncon)
    violations["ncon_ref"] = float(ref_data.ncon)
    depth = new_selfcollision_depth(ik_data, ref_data)
    violations["selfcol_new_depth"] = depth
    if depth > SELFCOL_NEW_DEPTH_TOL:
      ok = False

  return ok, violations


def _draw_stiffness(rng) -> float:
  """Log-uniform sample of the robot position stiffness K_rob."""
  sm = solver_module
  return float(
    np.exp(rng.uniform(np.log(sm.MIN_ROBOT_STIFFNESS), np.log(sm.MAX_ROBOT_STIFFNESS)))
  )


def _draw_ff_stiffness(rng) -> tuple[float, float]:
  """Log-uniform sample of the force-field (environment) position/rotational stiffness
  (K_env, K_rot_env), over the ranges of the original single-link force-field mode."""
  sm = solver_module
  k_env = float(
    np.exp(
      rng.uniform(
        np.log(sm.MIN_FORCEFIELD_STIFFNESS), np.log(sm.MAX_FORCEFIELD_STIFFNESS)
      )
    )
  )
  k_rot_env = float(
    np.exp(
      rng.uniform(
        np.log(sm.MIN_FORCEFIELD_ROTATIONAL_STIFFNESS),
        np.log(sm.MAX_FORCEFIELD_ROTATIONAL_STIFFNESS),
      )
    )
  )
  return k_env, k_rot_env


def _force_from_disp(rng, disp):
  """Displacement → (stiffness k, force = k·displacement). Clamps the displacement so
  that |force| ≤ IK_CHECK_MAX_FORCE_MAGNITUDE (140N), the 140/k half of the original
  single-link displacement cap min(0.7, 140/k) (this function does not enforce the
  hard 0.7 half); otherwise k·0.15 reaches up to 150N, over the cap."""
  k = _draw_stiffness(rng)
  cap = solver_module.IK_CHECK_MAX_FORCE_MAGNITUDE / k
  dmag = float(np.linalg.norm(disp))
  if dmag > cap and dmag > 1e-9:
    disp = disp * (cap / dmag)
  return k, k * disp


def _maybe_torque(rng, with_torque, max_rot_disp=2.0):
  """Torque augmentation knob. Default with_torque=False → pure position compliance
  (zero torque + placeholder rotational stiffness 1.0). True → draw a bounded torque
  (τ = K_rot·rotational displacement, clamped to 10 N·m) so the torque branches in
  Contact / CoM / feasibility / IK actually take effect (the original single-link code
  samples torque in every mode; here it is an explicit decision)."""
  sm = solver_module
  if not with_torque:
    return np.zeros(3), 1.0
  k_rot = float(
    np.exp(
      rng.uniform(
        np.log(sm.MIN_ROBOT_ROTATIONAL_STIFFNESS),
        np.log(sm.MAX_ROBOT_ROTATIONAL_STIFFNESS),
      )
    )
  )
  axis = rng.normal(size=3)
  axis /= max(np.linalg.norm(axis), 1e-9)
  mag = min(
    float(rng.uniform(0.0, max_rot_disp)) * k_rot, sm.IK_CHECK_MAX_TORQUE_MAGNITUDE
  )
  return axis * mag, k_rot


# Partner-aware direction prior: the sampler reads the dance-step phase and couples the
# force direction to "who is leading, and where to".
# Probability of the common branch (push along the partner's velocity, shared by both
# modes); the rest take the minor branch, directed along the robot's heading axis on
# the side opposite to travel: mode A (human forward, robot backward) = +u_r pulls
# toward the partner, mode B (human backward, robot forward) = -u_r frame resistance;
# the sign of v_h·u_r picks the mode. Default 0.85.
GUIDE_PROB = 0.85
# Gaussian jitter scale on the base direction (unit-vector scale), giving the guide a
# cone of directions instead of a single line.
GUIDE_JITTER = 0.4
# Below this partner horizontal speed (m/s) there is no phase information, so fall
# back to isotropic sampling (pauses in the dance).
MIN_GUIDE_SPEED = 0.1
# On infeasible backoff, the angle per step by which the force direction rotates
# outward about the vertical axis, and the step cap: rotate first, then shrink the
# magnitude; the shrink ladder only starts once the steps run out, so convergence is
# still guaranteed. The cap is 12 steps (12×15°, nominally 180°), but
# _rotate_force_outward turns toward the outward side every step and, once there,
# only swings ±7.5° around it, so the real end point is "at most straight outward",
# never past it into the reverse. A cap that stops short of outward (5 steps, 75°,
# keeping a backward component) cannot break directional dead ends: the discarded
# events use up their rotation steps and then wear through the shrink ladder. Losing
# a whole event costs far more than turning past 90°, so the cap goes all the way to
# straight outward.
GUIDE_ROT_STEP_DEG = 15.0
GUIDE_MAX_ROT_STEPS = 12
# Displacement cap of the minor branch (pull/resist): the connecting force in a hold
# is light frame pressure, not a driving push. Beyond physics there is survival: the
# resisting force points straight at the robot's torso (mode B's -u_r), and a
# full-scale 0.15 m displacement drags the wrist into the chest (measured: 4/4 events
# dropped whole), hence the 0.08 m cap.
GUIDE_MINOR_MAX_DISP = 0.08
# Share of single-hand events in a guided bimanual schedule: SoftMimic is natively
# single-link, so here a per-event draw sends some events to a single hand (left/right
# drawn uniformly), with the full single-hand displacement range and the
# single-contact CoM budget (see _sample_single). The default 0.25 is about the
# single-hand share that pair deaths leave behind without it (one wrist surviving a
# pair); drawing it on purpose makes that regime left/right-symmetric and
# full-budget. The unguided path does not read this value.
SINGLE_HAND_PROB = 0.25


class PartnerGuide:
  """Partner phase-aligned direction prior, defined only for paired data with a
  lead/follow structure, with two predefined modes.

  Gives one horizontal base direction per event. Common branch (probability
  guide_prob, shared by both modes): "push" along the partner's current horizontal
  root velocity, i.e. along the pair's shared direction of travel. Minor branch
  (1 - guide_prob): along the robot's heading axis on the side opposite to travel,
  b = -sign(v_h·u_r)·u_r: mode A (20260224_001, v_h·u_r<0) gives +u_r, the human
  pulling the hold toward themselves; mode B (20260408_001, v_h·u_r>0) gives -u_r,
  frame resistance pressing on the forward-moving robot while the partner backs up.
  Returns None when the partner barely moves, and the sampler falls back to
  isotropic. The velocity is read straight from the NPZ's root_lin_vel (stored in
  the tracked NPZs); a root_pos difference is the fallback
  when that key is missing.
  """

  def __init__(self, npz_path: str, guide_prob: float = GUIDE_PROB):
    self.path = str(npz_path)
    d = np.load(npz_path)
    self.fps = float(np.asarray(d["fps"]).reshape(-1)[0]) if "fps" in d else 50.0
    if "root_lin_vel" in d:
      v = d["root_lin_vel"][:, :2].astype(np.float64)
    elif "root_pos" in d:
      v = np.gradient(d["root_pos"][:, :2].astype(np.float64), axis=0) * self.fps
    else:
      raise KeyError(
        f"partner NPZ lacks root_lin_vel/root_pos: {npz_path} (expected an origin "
        "motion NPZ)"
      )
    # Events last 0.9 to 2s and the guide follows the phase trend, not per-frame
    # jitter: ~0.4s moving average.
    w = max(1, int(round(0.4 * self.fps)))
    kern = np.ones(w) / w
    self.vel = np.stack(
      [np.convolve(v[:, i], kern, mode="same") for i in range(2)], axis=1
    )
    self.guide_prob = float(guide_prob)

  def event_dir(
    self, rng: np.random.Generator, t: float, ref_data: mujoco.MjData
  ) -> tuple[np.ndarray | None, str]:
    """(base direction, branch name) of this event at time t. The direction is a
    horizontal unit vector; None = the partner barely moves, fall back to isotropic
    (branch name "pause"). The branch name (push/minor/pause, none when unguided) goes
    into run_meta so survival/discard can be attributed per branch."""
    i = min(max(int(round(t * self.fps)), 0), len(self.vel) - 1)
    vh = self.vel[i]
    if float(np.linalg.norm(vh)) < MIN_GUIDE_SPEED:
      return None, "pause"
    if float(rng.uniform()) < self.guide_prob:
      base = np.array([vh[0], vh[1], 0.0])  # push: partner's travel dir (both modes)
      branch = "push"
    else:
      # Minor branch: the heading-axis side opposite to travel (mode A pulls toward
      # the partner, mode B is frame resistance).
      x = ref_data.body("pelvis").xmat.reshape(3, 3)[:, 0]
      s = 1.0 if float(vh[0] * x[0] + vh[1] * x[1]) <= 0.0 else -1.0
      base = s * np.array([x[0], x[1], 0.0])
      branch = "minor"
    n = float(np.linalg.norm(base))
    return (base / n, branch) if n > 1e-9 else (None, "pause")


def _base_dir(rng, guide_dir=None, z_scale=0.3):
  """Displacement base direction (unit vector). Unguided = an isotropic draw;
  guided = the base direction plus a GUIDE_JITTER Gaussian jitter cone. The vertical
  component is scaled by ``z_scale`` (default 0.3, mostly horizontal; larger allows
  more downward drag / upward lift, and SoftMimic is equivalent to 1.0). Both paths
  consume the rng the same number of times."""
  d = rng.normal(size=3)
  if guide_dir is not None:
    d = guide_dir + GUIDE_JITTER * d
  d[2] *= z_scale
  return d / max(np.linalg.norm(d), 1e-9)


def _bimanual_disps(rng, max_disp, guide_dir=None, z_scale=0.3, intent=None):
  """Derive both hands' displacements from the intent, returning (left disp, right
  disp, intent). along = same direction (halved per hand, since same-direction pushes
  add up), open = mirrored, twist = vertically opposite; with guide_dir set, the base
  direction jitters around it.
  Note that twist's horizontal component is also scaled by mag (otherwise the O(1)
  unit vector → ~1m displacement → ~1000N force).
  An explicit ``intent`` skips the draw (sticky intent, see _sample_bimanual)."""
  if intent is None:
    intent = str(rng.choice(["along", "open", "twist"]))
  d = _base_dir(rng, guide_dir, z_scale)
  mag = float(rng.uniform(0.03, max_disp))
  if intent == "along":
    mag *= 0.5
    return d * mag, d * mag, intent
  if intent == "open":
    return (
      np.array([d[0], +abs(d[1]), d[2]]) * mag,
      np.array([d[0], -abs(d[1]), d[2]]) * mag,
      intent,
    )
  # twist: horizontal parts scaled by mag too; the vertical sign flips between hands
  return (
    np.array([d[0] * mag, d[1] * mag, +abs(mag)]),
    np.array([d[0] * mag, d[1] * mag, -abs(mag)]),
    intent,
  )


def _sample_bimanual(
  rng,
  ref_data,
  max_disp,
  hand_gap,
  with_torque,
  tries=20,
  guide_dir=None,
  z_scale=0.3,
  sticky_intent=False,
  max_rot_disp=2.0,
):
  """Correlated hands: resample until the two hands' target gap falls inside
  hand_gap; if a bounded number of tries misses, shrink the displacements step by
  step (×0.8) until the gap is back in range (the reference hand gap base ∈ range,
  and shrinking → gap tends to base, so it must converge). It shrinks rather than
  solving for the scale, because the gap is a vector norm, nonlinear in the scale.
  Every resample re-jitters around guide_dir.

  ``sticky_intent``: the intent is drawn once and resampling only changes direction
  and magnitude. Without it (default False) the intent is redrawn on every try, so
  the gap check weights the intents: along moves both hands the same way, the gap
  never changes and it always passes, while open/twist get rejected over and over at
  large ranges (measured at range 0.45 over 30 seeds: along 35% vs open 10%,
  nominally 1/3 each). The draw/guided paths are always sticky, so their intent
  shares equal the nominal values; the plain unguided path (no guide, every draw
  probability zero) is not."""
  lo, hi = hand_gap
  w0, w1 = ref_data.body(WRISTS[0]).xpos, ref_data.body(WRISTS[1]).xpos
  fixed = str(rng.choice(["along", "open", "twist"])) if sticky_intent else None
  dl, dr, intent = _bimanual_disps(rng, max_disp, guide_dir, z_scale, fixed)
  for _ in range(tries):
    if lo <= float(np.linalg.norm((w0 + dl) - (w1 + dr))) <= hi:
      break
    dl, dr, intent = _bimanual_disps(rng, max_disp, guide_dir, z_scale, fixed)
  for _ in range(20):  # fallback: if resampling missed, shrink into the range
    if lo <= float(np.linalg.norm((w0 + dl) - (w1 + dr))) <= hi:
      break
    dl, dr = dl * 0.8, dr * 0.8
  contacts = []
  for name, disp in zip(WRISTS, [dl, dr], strict=True):
    k, f = _force_from_disp(rng, disp)
    tau, k_rot = _maybe_torque(rng, with_torque, max_rot_disp)
    k_env, k_rot_env = _draw_ff_stiffness(rng)
    contacts.append(Contact(name, f, tau, k, k_rot, k_env, k_rot_env, intent))
  return contacts


def _sample_single(
  rng, link, max_disp, with_torque, guide_dir=None, z_scale=0.3, max_rot_disp=2.0
):
  """Single-hand event: SoftMimic single-link semantics. The displacement spans the
  full single-hand range U(0.03, max_disp), with no along halving and no hand-gap
  clamp; the scheduler applies the single-contact CoM budget (a one-hand 30N forward
  push shifts the CoM about 7.8cm, still inside the 0.10 cap; only same-direction
  two-hand pushes need halving). intent is recorded as "single", the same name the
  per-frame intent_final uses after pair resolution."""
  d = _base_dir(rng, guide_dir, z_scale)
  k, f = _force_from_disp(rng, d * float(rng.uniform(0.03, max_disp)))
  tau, k_rot = _maybe_torque(rng, with_torque, max_rot_disp)
  k_env, k_rot_env = _draw_ff_stiffness(rng)
  return [Contact(link, f, tau, k, k_rot, k_env, k_rot_env, intent="single")]


def _sample_antisym(rng, ref_data, intent, max_disp, with_torque, max_rot_disp=2.0):
  """Antisymmetric two-hand pair: oppose runs along the wrist-to-wrist line (squeeze
  / pull, half each), counter along its horizontal perpendicular (a pure couple about
  the vertical axis, like turning a steering wheel). Both hands share one stiffness
  and exactly opposite displacements, so the forces are equal and opposite and the
  net external force is about zero: nearly free for balance (the moments cancel),
  with the magnitude capped by the arm actuators rather than the CoM budget.
  Dance meaning: oppose = the frame's elasticity as a two-hand hold closes/opens;
  counter = twisting the frame (the shape of a turning lead, a disturbance regime in
  the current straight-line clips, handedness 50/50). The direction has 20% jitter
  (vertical ×0.3), and the gap clamp uses the same range as bimanual."""
  w0 = ref_data.body(WRISTS[0]).xpos.copy()
  w1 = ref_data.body(WRISTS[1]).xpos.copy()
  u = w1 - w0
  u[2] = 0.0
  u /= max(np.linalg.norm(u), 1e-9)
  if intent == "counter":
    u = np.array([-u[1], u[0], 0.0])  # z × u: horizontal perpendicular
  s = 1.0 if float(rng.uniform()) < 0.5 else -1.0  # squeeze/pull, or couple direction
  mag = float(rng.uniform(0.03, max_disp))
  j = rng.normal(size=3)
  j[2] *= 0.3
  d = s * u + 0.2 * j
  d /= max(np.linalg.norm(d), 1e-9)
  dl, dr = d * mag, -d * mag
  for _ in range(20):  # gap clamp: a squeeze stays above 0.15, a pull below 0.60
    if 0.15 <= float(np.linalg.norm((w0 + dl) - (w1 + dr))) <= 0.60:
      break
    dl, dr = dl * 0.8, dr * 0.8
  k = _draw_stiffness(rng)  # shared stiffness keeps the forces exactly antisymmetric
  cap = solver_module.IK_CHECK_MAX_FORCE_MAGNITUDE / k
  dmag = float(np.linalg.norm(dl))
  if dmag > cap and dmag > 1e-9:
    dl, dr = dl * (cap / dmag), dr * (cap / dmag)
  contacts = []
  for name, disp in zip(WRISTS, [dl, dr], strict=True):
    tau, k_rot = _maybe_torque(rng, with_torque, max_rot_disp)
    k_env, k_rot_env = _draw_ff_stiffness(rng)
    contacts.append(Contact(name, k * disp, tau, k, k_rot, k_env, k_rot_env, intent))
  return contacts


def _sample_independent(
  rng, max_disp, with_torque, guide_dir=None, z_scale=0.3, max_rot_disp=2.0
):
  """Independent subset: draw 1..N links from FORCEABLE_LINKS (not the hardcoded
  WRISTS), each with its own displacement (an ablation axis; more rejections). With
  guide_dir set, the base direction uses _base_dir's guided jitter cone (vertical
  scaled by z_scale)."""
  links = solver_module.FORCEABLE_LINKS
  k = int(rng.integers(1, len(links) + 1))
  chosen = list(rng.choice(links, size=k, replace=False))
  contacts = []
  for name in chosen:
    # Unguided: a 3D-isotropic direction (vertical not squashed).
    if guide_dir is None:
      d = rng.normal(size=3)
      d /= max(np.linalg.norm(d), 1e-9)
    else:
      d = _base_dir(rng, guide_dir, z_scale)
    kk, f = _force_from_disp(rng, d * float(rng.uniform(0.03, max_disp)))
    tau, k_rot = _maybe_torque(rng, with_torque, max_rot_disp)
    k_env, k_rot_env = _draw_ff_stiffness(rng)
    contacts.append(Contact(name, f, tau, kk, k_rot, k_env, k_rot_env))
  return contacts


def sample_contacts(
  mode,
  rng,
  ref_data,
  *,
  max_disp=0.15,
  hand_gap=(0.15, 0.60),
  with_torque=False,
  guide_dir=None,
  z_scale=0.3,
  sticky_intent=False,
  max_rot_disp=2.0,
):
  """Sample one multi-contact interaction. mode='bimanual' (default) / 'independent'.

  Force = stiffness × displacement, with the displacement clamped so |force| ≤ 140N
  (as in the original single-link code). bimanual derives both hands' displacements
  from the intent, halves same-direction ones, and resamples the hand gap into
  hand_gap (the lower bound keeps the hands out of the torso, the upper bound
  prevents overstretching). with_torque defaults to False = pure position compliance
  (position first); True turns on bounded torque augmentation. ref_data is only
  needed for the bimanual gap clamp; independent does not use it and accepts None.
  With guide_dir set, the base direction jitters around it (partner guidance, see
  PartnerGuide); None keeps the original isotropic draw.
  """
  if mode == "bimanual":
    return _sample_bimanual(
      rng,
      ref_data,
      max_disp,
      hand_gap,
      with_torque,
      guide_dir=guide_dir,
      z_scale=z_scale,
      sticky_intent=sticky_intent,
      max_rot_disp=max_rot_disp,
    )
  if mode == "independent":
    return _sample_independent(
      rng,
      max_disp,
      with_torque,
      guide_dir=guide_dir,
      z_scale=z_scale,
      max_rot_disp=max_rot_disp,
    )
  raise ValueError(
    f"unknown sampling mode: {mode!r} (expected 'bimanual' or 'independent')"
  )


def multilink_ik_step(
  solver,
  qpos_ref: np.ndarray,
  contacts: list[Contact],
  dt: float,
  *,
  pin_free_wrists: bool = False,
) -> np.ndarray:
  """Assemble every active task for this step and solve one constrained QP. Each of
  the K forced links gets a hard force_task (reference + F/K), the other keypoints
  stick to the reference, and the CoM uses the multi-contact target. Integrates the
  solved joint velocity into the configuration and returns vel.

  ``pin_free_wrists`` (on under partner guidance): a wrist **not forced** in the
  event instead borrows its force_task (cost 5.0) with a zero-displacement target,
  rather than the cost-1e-2 soft keypoint. In the QP a soft keypoint is whole-body
  slack: measured during right-wrist events, the left wrist strayed 2 to 13cm from
  the reference (the forced right wrist itself only moved 0.3 to 1.6cm), while in the
  reference the left wrist is only 0 to 4cm from the torso (the right wrist has about
  25cm of clearance), so the slack turns straight into a "new self-collision" of the
  left wrist against the torso, and attributed backoff then eats the left event (the
  main cause of a left/right event imbalance). Pinned to the per-frame reference, the
  free wrist's self-collision state matches the reference and is never judged new.
  Unguided runs pin only when asked (--pin-free-wrists).
  """
  ref_config = mink.Configuration(solver.model)
  ref_config.update(q=qpos_ref)
  mujoco.mj_forward(ref_config.model, ref_config.data)

  # Posture regularizer + waist/knee + pelvis orientation + feet (all toward reference).
  solver.posture_task.set_target(qpos_ref)
  active = [solver.posture_task]
  if solver.waist_task is not None:
    solver.waist_task.set_target(qpos_ref)
    active.append(solver.waist_task)
  if solver.knee_task is not None:
    solver.knee_task.set_target(qpos_ref)
    active.append(solver.knee_task)
  solver.pelvis_pitch_task.set_target(
    ref_config.get_transform_frame_to_world("pelvis", "body")
  )
  active.append(solver.pelvis_pitch_task)
  for name, task in solver.foot_tasks.items():
    task.set_target(ref_config.get_transform_frame_to_world(name, "body"))
    active.append(task)

  # Multi-contact CoM target.
  solver.com_task.set_target(
    multilink_cop_aware_com_target(ref_config.data, solver.total_mass, contacts)
  )
  active.append(solver.com_task)

  forced = {c.link_name for c in contacts}
  pinned = set(solver.force_tasks) - forced if pin_free_wrists and contacts else set()
  # Unforced keypoint links stick to the reference (forced ones use the hard
  # force_task below; pinned wrists are handled separately).
  for name, task in solver.keypoint_tasks.items():
    if name not in forced and name not in pinned:
      task.set_target(ref_config.get_transform_frame_to_world(name, "body"))
      active.append(task)

  # Free wrists: force_task with a zero-displacement target, i.e. hard-pinned to the
  # per-frame reference (see docstring).
  for name in pinned:
    task = solver.force_tasks[name]
    task.set_target(ref_config.get_transform_frame_to_world(name, "body"))
    active.append(task)

  # Forced links: hard target = reference pose + spring displacement (position F/K,
  # orientation τ/K_rot).
  for c in contacts:
    p_ref = ref_config.data.body(c.link_name).xpos.copy()
    r_ref = Rotation.from_matrix(ref_config.data.body(c.link_name).xmat.reshape(3, 3))
    p_t = p_ref + c.force / max(c.stiffness, 1e-6)
    r_t = r_ref
    if np.linalg.norm(c.torque) > 1e-4 and c.rot_stiffness > 1e-4:
      r_t = Rotation.from_rotvec(c.torque / c.rot_stiffness) * r_ref
    q = r_t.as_quat()  # xyzw
    q_wxyz = np.array([q[3], q[0], q[1], q[2]])
    task = solver.force_tasks[c.link_name]
    task.set_target(mink.SE3(np.concatenate([q_wxyz, p_t])))
    active.append(task)

  vel = mink.solve_ik(
    solver.configuration, active, dt, solver="daqp", limits=solver.limits, damping=1e-5
  )
  solver.configuration.integrate_inplace(vel, dt)
  mujoco.mj_forward(solver.model, solver.data)
  return vel


def _qpos_index(model, joint_name: str) -> int:
  """Joint name -> its qpos index (model.jnt_qposadr[mj_name2id(...JOINT...)]).
  Defined once, reused in several places."""
  return int(
    model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)]
  )


def extract_pose_29(solver) -> list[float]:
  """Read the 29 actuated joint angles from configuration.q by name, in
  G1_JOINT_NAMES order (same as the motion library)."""
  q = solver.configuration.q
  return [float(q[_qpos_index(solver.model, name)]) for name in G1_JOINT_NAMES]


def _scaled(contacts: list[Contact], scale: float) -> list[Contact]:
  """Scale each contact's force/torque by scale (for backoff), returning a new list
  (the originals are not modified)."""
  out = []
  for c in contacts:
    c2 = copy.copy(c)
    c2.force = c.force * scale
    c2.torque = c.torque * scale
    out.append(c2)
  return out


# CoM cap: the clip reference gets baked into the beyondmimic tracking target, so an
# unbalanced reference costs correctness, not just yield; hence 0.10 for the dance
# clips (the stand pool recipe passes --com-cap 0.15).
COM_CAP = 0.10
# A per-frame jump of the reference root position above this threshold (meters) is
# read as a clip loop/skip -> teleport resync (see generate_motion). At a normal
# 50fps the root moves < 5cm per frame, while looping back to the start jumps a long
# way, so 0.3m sits safely in between.
TELEPORT_JUMP = 0.3
# Whole-event backoff (mirrors the single-link _handle_infeasible_event): when a frame
# is infeasible, the event covering it is discounted **as a whole** and rewound to
# the event start to rerun, instead of discounting only that frame's force. Per-frame
# discounting tears the triangle wave into steps (seen in a seed clip: "hold plateau
# 4.33N -> steps down to 1.77N -> flat for 18 frames -> back to 4.07N within one
# frame"), while the event table is derived back from "|F| reaches 95% of peak", so
# the triangle wave the event table describes stops matching the per-frame channels
# and the online force-matching reward gets a target it cannot follow.
# Rewinding drops no frames: the rewound frames are rewritten as usual, only with
# consistently scaled force.
EVENT_BACKOFF = 0.8
# Below this scale the whole event is abandoned (force set to 0), so the backoff loop
# always converges. 0.8^6≈0.262 < 0.30 -> at most 6 rounds per event.
EVENT_SCALE_FLOOR = 0.30
# Without --motion (zero standing pose) there is no reference duration, so this
# fallback in seconds is used.
STAND_DURATION = 8.0


def _violating_links(
  model: mujoco.MjModel,
  ik_data: mujoco.MjData,
  ref_data: mujoco.MjData,
  contacts: list[Contact],
  violations: dict[str, float],
) -> set[str]:
  """Set of violating links on an infeasible frame: contact links over the tracking
  limit + body names of new self-collision pairs over the tolerance.

  Backoff only acts on violating events, so the other hand's feasible event is not
  dragged along: measured on the 20260408_001 clip, the left wrist's self-contact
  with the torso otherwise pulls the right wrist's full-magnitude, feasible event
  into the shrink ladder too (the pair drops together). CoM/foot violations cannot be
  attributed to a single link, so the empty set is returned and the caller falls back
  to backing off everything. Self-collision only matches body names exactly against
  event link names, so a violation mid-arm (e.g. elbow against torso) does not match
  and also takes the back-off-everything fallback; a subtree map would attribute it
  more precisely.
  """
  sm = solver_module
  out: set[str] = set()
  for i, c in enumerate(contacts):
    if violations.get(f"track_{i}", 0.0) > sm.IK_CHECK_MAX_IK_TRACKING_ERROR:
      out.add(c.link_name)
  ref_pairs = set(_selfcollision_pair_depths(ref_data))
  for (g1, g2), d in _selfcollision_pair_depths(ik_data).items():
    if (g1, g2) in ref_pairs or d <= SELFCOL_NEW_DEPTH_TOL:
      continue
    for g in (g1, g2):
      body = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])
      )
      if body:
        out.add(body)
  return out


def _rotate_force_outward(
  force: np.ndarray, ref_data: mujoco.MjData, link_name: str
) -> np.ndarray:
  """Rotate the force's horizontal component about the vertical axis by
  GUIDE_ROT_STEP_DEG degrees toward the side "away from the torso", keeping the
  magnitude and the vertical component.

  The rotation sense turns the direction toward outward (the forced link's horizontal
  bearing relative to the pelvis): the left hand yields out to the left, the right
  hand out to the right, directly targeting the measured "left arm/hand against the
  torso" self-collision pattern. Returned unchanged when the horizontal component is
  near zero.
  """
  f = np.array(force, dtype=float)
  fh = f[:2]
  if float(np.linalg.norm(fh)) < 1e-9:
    return f
  out = (ref_data.body(link_name).xpos - ref_data.body("pelvis").xpos)[:2]
  sign = 1.0 if float(fh[0] * out[1] - fh[1] * out[0]) >= 0.0 else -1.0
  th = np.radians(GUIDE_ROT_STEP_DEG) * sign
  c, s = float(np.cos(th)), float(np.sin(th))
  f[0], f[1] = c * fh[0] - s * fh[1], s * fh[0] + c * fh[1]
  return f


def clamp_forces_to_com_cap(ref_data, total_mass, contacts, cap):
  """Scale all contacts' forces/torques by one common factor so the predicted CoM
  shift is ≤ cap.

  The CoM shift is **linear** in force/torque (Δ=(−M_y,M_x)/mg, M=Σ lever×F+τ), so
  scale=cap/shift lands exactly on cap (verified on the real G1: one hand
  30N→0.078m, two hands→0.156m, exactly 2×).
  """
  if not contacts:
    return contacts
  target = multilink_cop_aware_com_target(ref_data, total_mass, contacts)
  shift = float(np.linalg.norm((target - ref_data.subtree_com[0])[:2]))
  if shift <= cap or shift < 1e-9:
    return contacts
  return _scaled(contacts, cap / shift)


def _com_shift(ref_data, total_mass, contacts) -> float:
  """CoM shift (meters) these contacts demand at this reference pose. Proportional to
  force (ref_com cancels out in the difference)."""
  target = multilink_cop_aware_com_target(ref_data, total_mass, contacts)
  return float(np.linalg.norm((target - ref_data.subtree_com[0])[:2]))


def fit_to_com_budget(ref_at, total_mass, contacts, cap, t0, t1, dt):
  """Scale a group of contacts as a whole so "the event stays inside the CoM budget
  throughout", returning the scaled contacts.

  The magnitude is set from the pose's balance margin at sampling time, rather than
  drawing a large force first and trimming it with the per-frame gate. The difference
  is the shape: when 103.5N is drawn but balance allows only 29N, the force hits the
  cap partway up the ramp and then rides the cap all along, so what gets recorded is
  the **gate envelope**, not the scheduled triangle wave (measured: 62 of 75 frames of
  an event clamped, the ramp/hold durations lose all effect, and the event table's
  waveform stops matching the per-frame channels). Sizing from
  the tightest moment over the event span keeps the gate from ever firing, so what
  gets recorded is the triangle wave itself.

  shift is proportional to force, so cap/shift lands in one go, with no iteration.
  The per-frame gate stays as a fallback (the reference pose can still be tighter
  between probes).
  """
  if not contacts:
    return contacts
  scale = 1.0
  probes = list(np.arange(t0, t1, max(5 * dt, 1e-3))) + [t1]  # every ~0.1s + the end
  for tp in probes:
    shift = _com_shift(ref_at(float(tp)), total_mass, contacts)
    if shift > 1e-9:
      scale = min(scale, cap / shift)
  return _scaled(contacts, scale) if scale < 1.0 else contacts


def build_schedule(
  rng,
  ref_at,
  duration,
  mode,
  dt=0.02,
  with_torque=False,
  *,
  total_mass,
  com_cap,
  guide=None,
  single_prob=0.0,
  oppose_prob=0.0,
  counter_prob=0.0,
  vertical_scale=0.3,
  max_disp=0.15,
  max_load_vel=None,
  async_singles=False,
  max_rot_disp=2.0,
):
  """Build force events that may overlap across different links (never on one link).

  ``ref_at(t)`` gives the real reference MjData at time t. **Each event samples
  against the reference at its own time**, instead of faking the whole clip with the
  t=0 frame: the hand-gap clamp and the CoM budget both depend on the pose, and the
  wrong pose means sizing on someone else's geometry.

  Each event is a triangle wave (ramp up, hold, ramp down); the ramp-down duration is
  **velocity-limited** by the peak displacement (ramp_dn ≥ displacement /
  MAX_RELEASE_LINEAR_VEL), so a large release displacement with a short ramp-down
  does not make a velocity spike that the bridge's finite differences bake into the
  tracking velocity target. ``fit_to_com_budget`` squeezes the magnitude into the
  balance budget (see its docstring).

  ``guide`` (PartnerGuide, optional) gives the base direction at each event's own
  time, biasing the direction prior by the dance phase; None consumes no extra random
  numbers. Each event also records requested_N
  (the magnitude the sampler drew, before the CoM budget) and rot_steps (the backoff
  outward-rotation count), so run_meta can show "requested vs landed" side by side.

  Event-type draw (bimanual, active under guidance or when any probability knob is
  > 0): a new event starts only when both wrists are free (prevents single-hand
  chains, see the comment in the loop); one uniform draw falls through
  ``single_prob`` / ``oppose_prob`` / ``counter_prob`` in order to a single hand, an
  opposing internal-force pair, or a yaw couple, and the rest take the usual
  correlated pair. A single hand uses the full single-hand range (with the
  single-contact CoM budget: the budget follows the number of hands that land);
  oppose/counter use the full ``max_disp`` (without the guided minor branch's
  narrowing). All three probabilities zero and no guide = the plain correlated-pair
  path, which consumes no extra random numbers.

  ``vertical_scale`` raises the vertical share of the sampled direction (0.3 = mostly
  horizontal, the default; SoftMimic is equivalent to 1.0). ``max_disp`` is the
  displacement range cap (default 0.15; widened for the stand pool). A non-None
  ``max_load_vel`` applies the same velocity limit to the **loading ramp** as to the
  release (ramp_up ≥ displacement / that value): with a 0.15 range and 1.0 m/s it can
  never trigger mathematically, and it only bites once the range is widened,
  preventing "snatch" loading.
  """
  sm = solver_module
  events = []
  last_end = {name: -1.0 for name in sm.FORCEABLE_LINKS}
  t = 0.0
  while t < duration:
    t += float(rng.uniform(0.5, 1.5))
    if t >= duration:
      break
    ref_t = ref_at(t)
    gdir, branch = (
      guide.event_dir(rng, t, ref_t) if guide is not None else (None, "none")
    )
    md = GUIDE_MINOR_MAX_DISP if branch == "minor" else max_disp
    lottery = single_prob + oppose_prob + counter_prob
    if mode == "bimanual" and (guide is not None or lottery > 0):
      # A new event starts only when both wrists are free: paired events naturally
      # start and end together (ramp_dn's floor is ramp_up, so both hands' ev_end
      # are always equal), and only single-hand events leave one wrist busy; if the
      # free wrist got another single-hand event while the other is busy, one single
      # would perpetuate itself into a whole chain of singles and the share would
      # drift from the draw probability. With both free, one draw picks the event
      # type, so the shares are exactly the probabilities.
      # With async_singles on, this becomes "start whenever a wrist is free": a
      # single lands on the free wrist, the two hands' event streams decouple, and
      # left/right asynchronous overlap (one hand joins/releases while the other is
      # loading) becomes a first-class regime; two-hand events (pair / oppose /
      # couple) still need both wrists free and are skipped this round otherwise.
      free = [w for w in WRISTS if t >= last_end[w]]
      if (not free) if async_singles else (len(free) < len(WRISTS)):
        continue
      r = float(rng.uniform())
      if r < single_prob:
        link = str(rng.choice(free))
        templates = _sample_single(
          rng,
          link,
          md,
          with_torque,
          guide_dir=gdir,
          z_scale=vertical_scale,
          max_rot_disp=max_rot_disp,
        )
      elif len(free) < len(WRISTS):
        continue  # async: a two-hand event drawn with only one wrist free; skip
      elif r < single_prob + oppose_prob:
        templates = _sample_antisym(
          rng, ref_t, "oppose", max_disp, with_torque, max_rot_disp
        )
      elif r < single_prob + oppose_prob + counter_prob:
        templates = _sample_antisym(
          rng, ref_t, "counter", max_disp, with_torque, max_rot_disp
        )
      else:
        templates = sample_contacts(
          mode,
          rng,
          ref_t,
          max_disp=md,
          with_torque=with_torque,
          guide_dir=gdir,
          z_scale=vertical_scale,
          sticky_intent=True,
          max_rot_disp=max_rot_disp,
        )
    else:
      templates = sample_contacts(
        mode,
        rng,
        ref_t,
        max_disp=md,
        with_torque=with_torque,
        guide_dir=gdir,
        z_scale=vertical_scale,
        max_rot_disp=max_rot_disp,
      )
    requested = [float(np.linalg.norm(c.force)) for c in templates]
    ramp_up = float(rng.uniform(0.2, 0.6))
    hold = float(rng.uniform(0.4, 0.8))
    # The unscaled span is a **superset** of the scaled one (shrinking the force only
    # shortens ramp_dn), so probing the budget over it is conservative.
    span_end = min(
      t
      + ramp_up
      + hold
      + max(
        ramp_up,
        max(
          (float(np.linalg.norm(c.force)) / max(c.stiffness, 1e-6) for c in templates),
          default=0.0,
        )
        / sm.MAX_RELEASE_LINEAR_VEL,
      ),
      duration,
    )
    templates = fit_to_com_budget(
      ref_at, total_mass, templates, com_cap, t, span_end, dt
    )
    if max_load_vel:
      # Loading-ramp velocity limit, the same rule as ramp_dn: large-displacement
      # events load more slowly, so no "push 0.45 m in 0.2 s" snatch transient gets
      # baked into the reference velocity target. Uses the real post-budget
      # displacement.
      peak_disp = max(
        (float(np.linalg.norm(c.force)) / max(c.stiffness, 1e-6) for c in templates),
        default=0.0,
      )
      ramp_up = max(ramp_up, peak_disp / max_load_vel)
    for c, req in zip(templates, requested, strict=True):
      if t < last_end[c.link_name]:
        continue  # a link never overlaps itself (different links may run concurrently)
      disp = float(np.linalg.norm(c.force)) / max(c.stiffness, 1e-6)
      ramp_dn = max(ramp_up, disp / sm.MAX_RELEASE_LINEAR_VEL)  # velocity-limited
      ev_end = t + ramp_up + hold + ramp_dn
      if ev_end > duration:
        continue
      events.append({
        "start": t, "hold_start": t + ramp_up,
        "hold_end": t + ramp_up + hold, "end": ev_end,
        "ramp_up": ramp_up, "ramp_dn": ramp_dn, "contact": c,
        "scale": 1.0,  # multiplied by EVENT_BACKOFF on whole-event backoff; 0 = dropped
        "requested_N": req,  # sampled magnitude (140N-clamped, before the CoM budget)
        "rot_steps": 0,  # backoff outward-rotation count (guided: rotate, then shrink)
        "torque_shed": False,  # torque-shedding rescue fired (see the backoff loop)
        "guide_branch": branch,  # push / minor / pause / none (unguided)
        "intent": c.intent,  # bimanual two-hand geometric intent (along/open/twist)
      })  # fmt: skip
      last_end[c.link_name] = ev_end
  return sorted(events, key=lambda e: e["start"])


def _ramp_factor(ev, t):
  """Triangle-wave amplitude factor: up (0→1), hold (1), down (1→0), 0 outside. The
  up/down ramps may differ in length (the ramp-down is velocity-limited).

  The end uses >= (not >) as an explicit short-circuit: end is computed separately
  (start+ramp_up+hold+ramp_dn), and (t-hold_end)/ramp_dn at t==end is not
  necessarily exactly 1.0 in floating point, so relying on arithmetic would leak a
  ~1e-16 residue at the end of the ramp-down instead of an exact 0 (the start side
  is exact by construction, since t-start subtracts to 0, so it is unaffected).
  """
  if t < ev["start"] or t >= ev["end"]:
    return 0.0
  if t < ev["hold_start"]:
    return (t - ev["start"]) / max(ev["ramp_up"], 1e-6)
  if t <= ev["hold_end"]:
    return 1.0
  return 1.0 - (t - ev["hold_end"]) / max(ev["ramp_dn"], 1e-6)


def active_events_at(schedule, t):
  """Events that **actually carry force** at time t → [(event, amplitude factor)].
  Amplitude = triangle-wave factor × whole-event backoff scale.

  Events backed off to scale=0 (abandoned) are filtered out right here, so the
  contact log never records an "active but zero-force" slot, and the backoff logic
  and active_contacts_at always see the same set of events.
  """
  out = []
  for ev in schedule:
    f = _ramp_factor(ev, t) * ev.get("scale", 1.0)
    if f > 1e-6:
      out.append((ev, f))
  return out


def active_contacts_at(schedule, t):
  """All active events at time t → a Contact list, with force/torque weighted by the
  triangle-wave factor × whole-event backoff scale.

  Also carries each event's audit info (intent, guide branch, outward-rotation
  angle, backoff scale, pair key) onto the frame-level Contact, for the contact NPZ's
  visualization channels to write out; pair_key is the schedule time, shared by both
  hands of one sampled pair.
  """
  out = []
  for ev, f in active_events_at(schedule, t):
    c = ev["contact"]
    out.append(
      Contact(
        c.link_name,
        c.force * f,
        c.torque * f,
        c.stiffness,
        c.rot_stiffness,
        c.ff_stiffness,
        c.ff_rot_stiffness,
        intent=c.intent,
        pair_key=float(ev["start"]),
        guide_branch=ev["guide_branch"],
        rot_deg=ev["rot_steps"] * GUIDE_ROT_STEP_DEG,
        ev_scale=float(ev["scale"]),
      )
    )
  return out


# Audit channel keys (visualization/audit): written through write_contact_npz_multi's
# extra_channels; the training reader takes keys from a fixed allowlist, so these
# extra keys are inert by construction.
BOOKKEEPING_CHANNELS = (
  "intent",
  "intent_final",
  "guide_branch",
  "backoff_rot_deg",
  "backoff_scale",
)


def _empty_bookkeeping(t: int, k: int) -> dict[str, np.ndarray]:
  """All-empty template of the (T,K) audit channels: the zero-force (zero-wrench)
  product keeps the same set of keys as the force products."""
  return {
    "intent": np.full((t, k), "", dtype="<U8"),
    "intent_final": np.full((t, k), "", dtype="<U8"),
    "guide_branch": np.full((t, k), "", dtype="<U8"),
    "backoff_rot_deg": np.zeros((t, k)),
    "backoff_scale": np.zeros((t, k)),
  }


def _frame_slot_channels(_ref, solved, contacts, slot_of, slot_ids):
  """Spread one frame's **final** contact set (after clamping/backoff) into the
  multi-link per-frame channels.

  Each active contact lands in its slot by link name (slot_of gives the key); the
  other slots are zero-filled (force=0, link_id=-1, quat=identity). The setpoint is
  derived from the **actually solved** link pose: position setpoint = p_solved +
  F/K_env, rotational setpoint = R(τ/K_rot_env) ⊗ R_solved. The K_rob-side yield is
  already realized in the solved pose; deriving it as p_ref + F/K_env + F/K_rob
  instead would let the millimeter-level residual of mink's constrained weighted
  solve, multiplied by K_env, leak into force matching (measured per event: median
  0.1 to 4.2 N, peak 12 N), while the self-consistent form makes
  K_env·(setpoint − solved) ≡ F hold bit for bit. Recording an unclipped force would
  push the online force-matching reward below 1, so force must be the final value
  passed in. Non-bimanual non-wrist/>2 links have no slot → caught by the assertion
  (the K slots only cover both wrists).
  """
  k = len(slot_of)
  ch: dict[str, np.ndarray] = {
    "link_id": np.full(k, -1, dtype=np.int64),
    "force": np.zeros((k, 3)),
    "torque": np.zeros((k, 3)),
    "robot_stiffness": np.zeros(k),
    "robot_rot_stiffness": np.zeros(k),
    "forcefield_stiffness": np.zeros(k),
    "forcefield_rot_stiffness": np.zeros(k),
    "setpoint_pos": np.zeros((k, 3)),
    "setpoint_quat": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (k, 1)),  # wxyz identity
    "plane_normal": np.zeros((k, 3)),  # not a collision mode, always zero
    # Audit channels (visualization/audit): sampled intent, intent after pair
    # resolution (the original intent when the pair's other hand still has force this
    # frame, else "single"), guide branch, backoff outward-rotation angle and backoff
    # scale. Free slots are all empty.
    "intent": np.full(k, "", dtype="<U8"),
    "intent_final": np.full(k, "", dtype="<U8"),
    "guide_branch": np.full(k, "", dtype="<U8"),
    "backoff_rot_deg": np.zeros(k),
    "backoff_scale": np.zeros(k),
  }
  for c in contacts:
    assert c.link_name in slot_of, (
      f"contact link {c.link_name!r} is not one of the two wrist slots {list(slot_of)}: the multi-link "
      "contact log supports only the two wrists (bimanual mode); --mode independent "
      "with other links cannot write --contact-out."
    )
    s = slot_of[c.link_name]
    k_rob = max(c.stiffness, 1e-6)
    k_env = max(c.ff_stiffness, 1e-6)
    k_rot_rob = max(c.rot_stiffness, 1e-6)
    k_rot_env = max(c.ff_rot_stiffness, 1e-6)
    p_solved = solved.body(c.link_name).xpos
    r_solved = solved.body(c.link_name).xquat  # wxyz
    ch["link_id"][s] = slot_ids[s]
    ch["force"][s] = c.force
    ch["torque"][s] = c.torque
    ch["robot_stiffness"][s] = c.stiffness
    ch["robot_rot_stiffness"][s] = c.rot_stiffness
    ch["forcefield_stiffness"][s] = c.ff_stiffness
    ch["forcefield_rot_stiffness"][s] = c.ff_rot_stiffness
    ch["setpoint_pos"][s] = p_solved + c.force / k_env
    # Rotational setpoint: the environment spring left-multiplies the **solved**
    # orientation (scipy works in xyzw internally, converted back to wxyz at the end).
    rot = Rotation.from_rotvec(c.torque / k_rot_env) * Rotation.from_quat(
      r_solved[[1, 2, 3, 0]]
    )
    q = rot.as_quat()  # xyzw
    ch["setpoint_quat"][s] = q[[3, 0, 1, 2]]  # -> wxyz
    partner_alive = c.pair_key >= 0 and any(
      o is not c and o.pair_key == c.pair_key and o.link_name != c.link_name
      for o in contacts
    )
    ch["intent"][s] = c.intent
    ch["intent_final"][s] = (
      c.intent if partner_alive else ("single" if c.intent else "")
    )
    ch["guide_branch"][s] = c.guide_branch
    ch["backoff_rot_deg"][s] = c.rot_deg
    ch["backoff_scale"][s] = c.ev_scale
  return ch


# Placeholder robot stiffness on free frames (SoftMimic runner.py _record_step: falls
# back to 140.0 / 1.0 after the last event). Free frames between/before events carry
# the stiffness of the "next event" instead, see _fill_slot_stiffness.
FREE_FRAME_STIFFNESS = 140.0
FREE_FRAME_ROT_STIFFNESS = 1.0
# The activity criterion uses the same eps as event derivation
# (contact_reference.derive_contact_events default 1e-2).
_ACTIVE_EPS = 1e-2


def _fill_slot_stiffness(ch: dict[str, np.ndarray]) -> None:
  """Fill the free frames of the (T,K) robot stiffness channels with SoftMimic
  semantics: never zero, no jump at event onset.

  SoftMimic's recorder writes the stiffness of the "next queued event" on free
  frames (runner.py _record_step: target_event = current_event or event_queue[0]),
  falling back to the 140.0 / 1.0 placeholder only after the last event; so the
  stiffness observation does not change at all at event onset and the policy gets no
  free contact-indicator bit (channel jumps only happen where an event "ends"). This
  does the same to the already-stacked channels: per slot, frames before/between
  force-active spans get the next span's stiffness, and frames after the last span
  (and slots with no events at all) get the placeholder. Only robot_stiffness /
  robot_rot_stiffness are touched; the K_env channels stay zero outside events (the
  online spring and phase gating only read them inside events). In place,
  idempotent.
  """
  fmag = np.linalg.norm(ch["force"], axis=-1)  # (T,K)
  tmag = np.linalg.norm(ch["torque"], axis=-1)
  t, k = fmag.shape
  for s in range(k):
    active = (fmag[:, s] > _ACTIVE_EPS) | (tmag[:, s] > _ACTIVE_EPS)
    spans = _contiguous_spans(active)
    prev_end = 0
    for f0, f1 in spans:
      ch["robot_stiffness"][prev_end:f0, s] = ch["robot_stiffness"][f0, s]
      ch["robot_rot_stiffness"][prev_end:f0, s] = ch["robot_rot_stiffness"][f0, s]
      prev_end = f1
    ch["robot_stiffness"][prev_end:t, s] = FREE_FRAME_STIFFNESS
    ch["robot_rot_stiffness"][prev_end:t, s] = FREE_FRAME_ROT_STIFFNESS


def _write_contact_log(path, frames_ch, slot_link_names, slot_ids, run_meta=None):
  """Stack per-frame multi-link channels into (T,K,·) → fill free-frame stiffness →
  derive the event table → write the native multi-link contact NPZ."""
  ch = {key: np.stack([f[key] for f in frames_ch], axis=0) for key in frames_ch[0]}
  _fill_slot_stiffness(ch)
  names = np.asarray(slot_link_names)
  link_name = np.where(ch["link_id"] >= 0, names[None, :], "")
  id2name = {int(i): n for i, n in zip(slot_ids, slot_link_names, strict=True)}
  events = derive_contact_events_multi(
    ch["link_id"], ch["force"], ch["torque"],
    ch["robot_stiffness"], ch["robot_rot_stiffness"],
    ch["forcefield_stiffness"], ch["forcefield_rot_stiffness"],
    id2name,
  )  # fmt: skip
  write_contact_npz_multi(
    path,
    link_id=ch["link_id"],
    link_name=link_name,
    force=ch["force"],
    torque=ch["torque"],
    robot_stiffness=ch["robot_stiffness"],
    robot_rot_stiffness=ch["robot_rot_stiffness"],
    forcefield_stiffness=ch["forcefield_stiffness"],
    forcefield_rot_stiffness=ch["forcefield_rot_stiffness"],
    setpoint_pos=ch["setpoint_pos"],
    setpoint_quat_wxyz=ch["setpoint_quat"],
    plane_normal=ch["plane_normal"],
    slot_link_names=names,
    events=events,
    run_meta=run_meta,
    extra_channels={key: ch[key] for key in BOOKKEEPING_CHANNELS},
  )


def _held_segments(rng, t: int, dt: float) -> list[tuple[int, int]]:
  """Cut [0,T) into hold segments [(a,b),...], each uniform(2,5) seconds (SoftMimic's
  hold duration)."""
  segs: list[tuple[int, int]] = []
  pos = 0
  while pos < t:
    n = max(1, int(round(rng.uniform(2.0, 5.0) / dt)))
    segs.append((pos, min(pos + n, t)))
    pos += n
  return segs


def _held_random_stiffness(rng, t: int, dt: float, lo: float, hi: float) -> np.ndarray:
  """(T,) hold-and-redraw stiffness sequence: log-uniform values, each held for
  uniform(2,5) seconds before a redraw.

  SoftMimic's zero-wrench stiffness semantics (a per-frame version of runner.py
  _update_stiffness_state): stiffness varies over time but independently of the
  (nonexistent) force events, teaching the policy that "the stiffness observation by
  itself does not mean contact".
  """
  vals = np.empty(t)
  for a, b in _held_segments(rng, t, dt):
    vals[a:b] = np.exp(rng.uniform(np.log(lo), np.log(hi)))
  return vals


def write_zerowrench_contact(
  motion_npz, out, run_meta=None, seed=0, mode="bimanual", with_torque=False
):
  """Zero-contact multi-link contact log: frame count from the origin NPZ, forces/events
  all empty, stiffness a hold-and-redraw random sequence.

  The contact face of a zero-wrench pair (SoftMimic's zero-wrench mode,
  ``softmimic_mink_augment.py --force_mode zero-wrench``). The pool requires every
  clip to carry a multi-link contact_file, while both the robot and free faces of a
  zero-wrench pair point straight at the origin clip: this NPZ is the only file the
  pair needs generated. The position stiffness channel takes a hold-and-redraw
  sequence (log-uniform [10,1000] N/m, each held 2 to 5 seconds), never zero just
  like the free frames of force-event clips, so the stiffness observation cannot
  serve as a contact indicator.

  ``mode`` has the same name and meaning as in the motion subcommand: ``bimanual``
  (default) = all channels share one set of hold-segment boundaries, with values
  drawn independently per slot and channel, matching the switching structure of
  bimanual force events ("both wrists start and end together, K sampled
  separately"); otherwise "do both wrists switch on the same frame" by itself would
  single out zero-wrench clips among force clips (clip-type leakage).
  ``independent`` = separate boundaries per channel (for ablation).

  ``with_torque`` also matches the motion subcommand: False (default) pins the
  rotational stiffness to FREE_FRAME_ROT_STIFFNESS (1.0) throughout, since in force
  clips without torque augmentation K_rot is a constant 1.0 placeholder, and if
  zero-wrench took its own ladder, "does K_rot change" would again fingerprint the
  clip type; True puts the rotational stiffness on the same hold-and-redraw ladder
  as the position stiffness ([0.1,10] Nm/rad), to pair with --with-torque force
  clips. The output is deterministic for a given (seed, mode, with_torque). Returns
  the frame count.
  """
  d = np.load(motion_npz)
  t = int(d["joint_pos"].shape[0])
  dt = 1.0 / float(np.asarray(d["fps"]).item()) if "fps" in d else 0.02
  k = len(WRISTS)
  rng = np.random.default_rng(seed)
  sm = solver_module
  pos_lo, pos_hi = sm.MIN_ROBOT_STIFFNESS, sm.MAX_ROBOT_STIFFNESS
  rot_lo, rot_hi = (
    sm.MIN_ROBOT_ROTATIONAL_STIFFNESS,
    sm.MAX_ROBOT_ROTATIONAL_STIFFNESS,
  )
  robot_rot_stiffness = np.full((t, k), FREE_FRAME_ROT_STIFFNESS)
  if mode == "bimanual":
    robot_stiffness = np.empty((t, k))
    for a, b in _held_segments(rng, t, dt):
      for s in range(k):
        robot_stiffness[a:b, s] = np.exp(rng.uniform(np.log(pos_lo), np.log(pos_hi)))
        if with_torque:
          robot_rot_stiffness[a:b, s] = np.exp(
            rng.uniform(np.log(rot_lo), np.log(rot_hi))
          )
  elif mode == "independent":
    robot_stiffness = np.stack(
      [_held_random_stiffness(rng, t, dt, pos_lo, pos_hi) for _ in range(k)], axis=1
    )
    if with_torque:
      robot_rot_stiffness = np.stack(
        [_held_random_stiffness(rng, t, dt, rot_lo, rot_hi) for _ in range(k)], axis=1
      )
  else:
    raise ValueError(
      f"unknown zerowrench mode: {mode!r} (expected 'bimanual' or 'independent')"
    )
  write_contact_npz_multi(
    out,
    link_id=np.full((t, k), -1, dtype=np.int64),
    link_name=np.full((t, k), "", dtype="<U20"),
    force=np.zeros((t, k, 3)),
    torque=np.zeros((t, k, 3)),
    robot_stiffness=robot_stiffness,
    robot_rot_stiffness=robot_rot_stiffness,
    forcefield_stiffness=np.zeros((t, k)),
    forcefield_rot_stiffness=np.zeros((t, k)),
    setpoint_pos=np.zeros((t, k, 3)),
    setpoint_quat_wxyz=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (t, k, 1)),
    plane_normal=np.zeros((t, k, 3)),
    slot_link_names=np.asarray(WRISTS),
    events=[],
    run_meta=run_meta,
    extra_channels=_empty_bookkeeping(t, k),
  )
  return t


def generate_motion(
  solver, duration, mode, seed, out_csv, dt=0.02,
  com_cap=COM_CAP, with_torque=False, contact_out=None, run_meta=None, guide=None,
  single_prob=0.0, oppose_prob=0.0, counter_prob=0.0, vertical_scale=0.3,
  max_disp=0.15, max_load_vel=None, pin_free_wrists=None, async_singles=False,
  max_rot_disp=2.0,
):  # fmt: skip
  """Walk along the reference trajectory, **clamp to the feasible set every frame**,
  then run multi-contact IK, recording adapted qpos rows.

  Magnitudes are already set at scheduling time from each event's own balance margin
  (``fit_to_com_budget``), and the per-frame com_cap gate is only a fallback, so the
  recorded force profile is the scheduled triangle wave. After IK comes the
  feasibility gate (self-collision included); on failure the **whole event** is
  discounted, rewound and rerun (see EVENT_BACKOFF), dropping no frames and never
  tearing the triangle wave into steps.
  Output CSV = [root_pos(3), root_quat_xyzw(4), 29 joints], the bridge's input
  format. Returns the number of frames written.

  During events the configuration evolves frame to frame (warm start), so the
  solution follows the moving reference + force; **free frames return to the
  reference** (see the reset in the loop), so the adapted face equals the original
  face bit for bit outside events, the same invariant as SoftMimic.

  With contact_out set, also writes a multi-link contact log NPZ (slots = both wrists,
  slot0 left, slot1 right), recording per frame the **final** force set (after
  clamping/backoff) + the force-field series-spring setpoint, for the online task to
  consume.

  ``guide`` (PartnerGuide, optional) turns on partner guidance: the direction prior
  goes to build_schedule, and backoff first attributes by violation, then rotates
  outward, then shrinks; None samples unguided.

  ``pin_free_wrists``: None (default) = follow the guide (pinned when guided); an
  explicit True makes **unguided** builds pin the free wrist too, and also turns on
  attribution + the outward-rotation rescue (a prerequisite for single-hand events
  on the stand pool, see the slack argument in the pinned-wrist docstring). The
  other knobs (draw probabilities / vertical_scale / max_disp / max_load_vel) are
  described in build_schedule.
  """
  rng = np.random.default_rng(seed)
  assist = bool(pin_free_wrists) if pin_free_wrists is not None else guide is not None

  def ref_at(t: float) -> mujoco.MjData:
    """Real reference pose at time t (the schedule takes geometry at each event's own
    time rather than faking the whole clip with the t=0 frame)."""
    return forward_ref(solver.model, solver.get_reference_motion(t)[0])

  schedule = build_schedule(
    rng, ref_at, duration, mode, dt, with_torque,
    total_mass=solver.total_mass, com_cap=com_cap, guide=guide,
    single_prob=single_prob, oppose_prob=oppose_prob, counter_prob=counter_prob,
    vertical_scale=vertical_scale, max_disp=max_disp, max_load_vel=max_load_vel,
    async_singles=async_singles, max_rot_disp=max_rot_disp,
  )  # fmt: skip
  solver.configuration.update(q=solver.get_reference_motion(0.0)[0])
  mujoco.mj_forward(solver.model, solver.data)

  # multi-link contact log: slots keyed by link = both wrists; slot_of/slot_ids built once.
  log_contacts = contact_out is not None
  slot_of = {name: i for i, name in enumerate(WRISTS)}
  slot_ids = [
    mujoco.mj_name2id(solver.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in WRISTS
  ]
  frames_ch: list[dict[str, np.ndarray]] = []

  rows = []
  # Solver state **before** each frame's IK, for whole-event backoff rewinds (487
  # frames × 36 floats ≈ 140KB, negligible).
  qhist: list[np.ndarray] = []
  prevhist: list[np.ndarray | None] = []
  prev_ref: np.ndarray | None = None
  # round, not truncate: duration = num_frames*dt round-trips through float
  # (501*0.02)/0.02 = 500.99999...; int() would drop the last frame exactly
  # when the clip length is a "nice" number (the stand clip's 10.02s), while
  # the waltz lengths land at N.0000001 and never tripped it. round keeps
  # every existing count (round(487.0000001) == 487) and closes the edge.
  n_frames = int(round(duration / dt))
  # Each event gives up after backing off to EVENT_SCALE_FLOOR (6 rounds), plus up to
  # GUIDE_MAX_ROT_STEPS outward rotations when guided, so this cap is only a guard
  # against pathological cases.
  per_event = 8 + (GUIDE_MAX_ROT_STEPS + 1 if assist else 0)  # +1: torque shedding
  max_rewinds = per_event * max(1, len(schedule)) + 10
  rewinds = 0
  frame = 0
  while frame < n_frames:
    t = frame * dt
    qhist.append(np.array(solver.configuration.q, dtype=float))
    prevhist.append(None if prev_ref is None else prev_ref.copy())
    qpos_ref, _, _ = solver.get_reference_motion(t)
    # On a reference jump (clip loop/skip), the warm-started config mid-event cannot
    # follow the new reference -> teleport resync, or IK starting from the old
    # solution diverges (free frames are reset every frame anyway; only mid-event
    # frames need this guard).
    if (
      prev_ref is not None
      and float(np.linalg.norm(qpos_ref[:3] - prev_ref[:3])) > TELEPORT_JUMP
    ):
      solver.configuration.update(q=qpos_ref)
    prev_ref = np.array(qpos_ref, dtype=float)
    ref = forward_ref(solver.model, qpos_ref)  # real per-frame ref (link poses/CoM)
    contacts = active_contacts_at(schedule, t)
    contacts = clamp_forces_to_com_cap(ref, solver.total_mass, contacts, com_cap)
    # Sub-threshold residual force counts as "no contact", with the same _ACTIVE_EPS
    # threshold as event-table derivation. Each end of a triangle wave has a frame or
    # two with mN-level amplitude: the event table (derive_contact_events, same eps)
    # never includes them, but if the channels recorded them anyway, frames the table
    # calls "free" would keep a nonzero K_env / setpoint and the free-frame reset
    # would not fire either (measured on human_rev frame 262: |F| 8.2 mN, K_env 566
    # N/m, outside the event table). With one threshold throughout, the channels, the
    # event table and the reset always agree.
    contacts = [
      c
      for c in contacts
      if float(np.linalg.norm(c.force)) > _ACTIVE_EPS
      or float(np.linalg.norm(c.torque)) > _ACTIVE_EPS
    ]
    if not contacts:
      # Free frames snap back to the reference (the same rule as SoftMimic's
      # ik_update.py: with no active event and the release finished,
      # configuration.update(q=qpos_ref)). A warm-started QP drifts further and
      # further in the null space: the redundant arm can hold the reference in task
      # space while switching to another joint solution (measured on force-free
      # frames: median joint difference 0.12 rad, peak 1.17 rad), and it never comes
      # back on its own. Zeroing it outside events is what makes the adapted face
      # truly equal the original face (the invariant: observations, rewards and data
      # scaling are all designed around "both faces equal outside events").
      # This only clears the null space, not the geometry: measured, one more
      # contact-free IK solve after the reset gives max|dq| of exactly 0 (every task
      # is already satisfied at the reference), so this line changes no pose that
      # should be there.
      solver.configuration.update(q=qpos_ref)
    multilink_ik_step(solver, qpos_ref, contacts, dt, pin_free_wrists=assist)
    ok = True
    viol: dict[str, float] = {}
    if contacts:  # CoM already clamped; this catches self-collision/tracking error
      ok, viol = is_multilink_feasible(solver.data, ref, contacts, solver.total_mass)
    if not ok and rewinds < max_rewinds:
      # Whole-event backoff: discount the **violating** events covering this frame
      # as a whole and rewind to the earliest of their start frames to rerun, so the
      # whole force profile stays a triangle wave of one scale (per-frame discounting
      # tears the hold plateau into steps, see the note at EVENT_BACKOFF). For
      # attribution see _violating_links: a left-wrist self-contact with the torso does
      # not drag along the right wrist's full-magnitude, feasible event; when
      # nothing can be attributed to an active event (CoM / feet / mid-arm), fall
      # back to backing off everything. Every failure discounts some event, so
      # convergence is still guaranteed.
      rewinds += 1
      active = [ev for ev, _f in active_events_at(schedule, t)]
      # Attribution runs under assist (guided, or --pin-free-wrists); without it
      # every active event backs off together.
      violating = (
        _violating_links(solver.model, solver.data, ref, contacts, viol)
        if assist
        else set()
      )
      backoff_events = [
        ev for ev in active if ev["contact"].link_name in violating
      ] or active
      for ev in backoff_events:
        if assist and float(np.linalg.norm(ev["contact"].torque)) > 1e-9:
          # Torque-shedding rescue stage, first in line: a torque event's IK target
          # turns the wrist orientation through τ/K_rot (up to 2 rad), and for a
          # left wrist close to the chest that orientation sweep self-collides
          # however the force is turned, so outward rotation cannot save it.
          # Measured with torque on: with shedding placed after outward rotation,
          # 12 steps burn for nothing and many events still die after shedding;
          # zeroing τ from the start with the same seed and random stream loses
          # none, so shed on the first failure. The cost is that
          # failures caused by the force direction also lose their torque first (a
          # few percent of events), in exchange for the force event itself
          # surviving. At most once per event (after shedding |τ|=0 and this branch
          # is never entered again), so convergence is still guaranteed.
          ev["contact"].torque = np.zeros(3)
          ev["torque_shed"] = True
          continue
        if assist and ev["rot_steps"] < GUIDE_MAX_ROT_STEPS:
          # Under partner guidance, rotate first and shrink later: the main failure
          # is a backward force pushing the wrist into the torso (measured: the left
          # arm/hand self-collides with the torso). One outward rotation step clears
          # the torso with the magnitude unchanged, keeping the guided trend; only
          # when the rotation steps run out and it is still infeasible does the
          # shrink ladder start, so convergence is still guaranteed.
          ev["rot_steps"] += 1
          ev["contact"].force = _rotate_force_outward(
            ev["contact"].force, ref, ev["contact"].link_name
          )
          continue
        ev["scale"] *= EVENT_BACKOFF
        if ev["scale"] < EVENT_SCALE_FLOOR:
          ev["scale"] = 0.0  # too small -> abandon the event (guarantees convergence)
      back = min(int(ev["start"] / dt) for ev in backoff_events)
      back = max(0, min(back, frame))
      solver.configuration.update(q=qhist[back])
      mujoco.mj_forward(solver.model, solver.data)
      prev_ref = prevhist[back]
      del qhist[back:], prevhist[back:], rows[back:]
      if log_contacts:
        del frames_ch[back:]
      frame = back
      continue
    q = solver.configuration.q
    root_xyzw = np.array([q[4], q[5], q[6], q[3]])  # wxyz → xyzw
    joints = np.array(extract_pose_29(solver))
    rows.append(np.concatenate([q[0:3], root_xyzw, joints]))
    if log_contacts:  # the final force set (after clamping/backoff), not pre-clamp
      mujoco.mj_forward(solver.model, solver.data)  # keep FK in sync with solved q
      frames_ch.append(
        _frame_slot_channels(ref, solver.data, contacts, slot_of, slot_ids)
      )
    frame += 1
  if rewinds >= max_rewinds:
    print(
      f"[WARN] whole-event backoff hit its cap of {max_rewinds} rewinds; later "
      "infeasible frames are written with their original force."
    )
  dropped = sum(1 for ev in schedule if ev["scale"] == 0.0)
  if dropped:
    print(
      f"[WARN] {dropped}/{len(schedule)} events backed off to the floor and were "
      "dropped (force set to 0)."
    )

  write_adapted_csv(out_csv, np.stack(rows, axis=0))
  if log_contacts:
    if run_meta is not None:
      # Write the schedule's "requested vs landed" into run_meta: requested_N is the
      # magnitude the sampler drew (before the CoM budget), scheduled_N the in-budget
      # scheduled magnitude, final_scale the scale after whole-event backoff, and
      # rot_steps the number of outward rotations. Side by side, how much the gate
      # and the backoff each eat turns from inference into measurement.
      meta = json.loads(run_meta)
      if guide is not None:
        # Guide parameters travel with the product: the outward-rotation step size
        # and count are module constants that can differ between pools, so charts
        # converting an event's rot_steps to an angle must use the values recorded
        # here at build time, not the current constants.
        meta["guide"] = {
          "partner_clip": guide.path,
          "guide_prob": guide.guide_prob,
          "rot_step_deg": GUIDE_ROT_STEP_DEG,
          "max_rot_steps": GUIDE_MAX_ROT_STEPS,
          "single_prob": single_prob,
        }
      # Read the CoM weights actually in effect from the solver object (not copied
      # from the CLI), so the self-description is the ground truth.
      com_cost_vec = np.asarray(
        getattr(solver.com_task, "cost", [float("nan")] * 3)
      ).reshape(-1)
      meta["sampler"] = {
        "single_prob": single_prob,
        "oppose_prob": oppose_prob,
        "counter_prob": counter_prob,
        "vertical_scale": vertical_scale,
        "max_disp": max_disp,
        "max_load_vel": max_load_vel,
        "pin_free_wrists": assist,
        "async_singles": bool(async_singles),
        "com_cap": com_cap,
        "com_cost_xy": float(com_cost_vec[0]),
        "com_cost_z": float(com_cost_vec[-1]),
        "with_torque": bool(with_torque),
        "max_rot_disp": max_rot_disp,
        "guided": guide is not None,
      }
      meta["schedule_events"] = [
        {
          "link": ev["contact"].link_name,
          "start_s": round(ev["start"], 3),
          "requested_N": round(ev["requested_N"], 2),
          "scheduled_N": round(float(np.linalg.norm(ev["contact"].force)), 2),
          "final_scale": round(ev["scale"], 4),
          "rot_steps": ev["rot_steps"],
          "torque_shed": bool(ev.get("torque_shed", False)),
          "guide_branch": ev["guide_branch"],
          "intent": ev["intent"],
        }
        for ev in schedule
      ]
      run_meta = json.dumps(meta, ensure_ascii=False)
    _write_contact_log(contact_out, frames_ch, list(WRISTS), slot_ids, run_meta)
  return len(rows)


# Names of the adapted CSV's 36 columns (order = data order): root position 3 + root
# quaternion **xyzw** 4 + 29 joints.
ADAPTED_COLUMNS = [
  "root_pos_x", "root_pos_y", "root_pos_z",
  "root_quat_x", "root_quat_y", "root_quat_z", "root_quat_w",
  *G1_JOINT_NAMES,
]  # fmt: skip


def labeled_path(out_csv: str) -> str:
  """Path of the headered copy: <stem>_labeled.csv, next to the headerless one."""
  stem, ext = os.path.splitext(out_csv)
  return f"{stem}_labeled{ext or '.csv'}"


def write_adapted_csv(out_csv: str, rows: np.ndarray) -> str:
  """Write two adapted CSVs: headerless (the bridge's input contract) + headered
  (for humans). Returns the path of the headered one.

  The bridge / MotionLoader reads by column position and knows no header; an extra
  string row would be parsed as data and fail, so the contract copy must stay
  headerless. But 36 columns of bare numbers give a human no way to tell "is this
  the left elbow or the right wrist", so a copy with the same data plus column names
  is written alongside. The numeric parts of the two are bit-identical.
  """
  assert rows.shape[1] == len(ADAPTED_COLUMNS), (
    f"column count {rows.shape[1]} does not match the {len(ADAPTED_COLUMNS)} names"
  )
  np.savetxt(out_csv, rows, delimiter=",")
  lab = labeled_path(out_csv)
  np.savetxt(lab, rows, delimiter=",", header=",".join(ADAPTED_COLUMNS), comments="")
  return lab


_FPS_HELP = (
  "The **true** frame rate of the reference motion CSV (pass 50 for a 50Hz source). "
  "The CSV has no timestamps, so this number is the only basis for mapping a query "
  "time t to a row index and for the total duration (CsvMotionLib.DATA_FPS). A "
  "wrong value changes the speed of the whole motion: a 50Hz source read as 30 = "
  "1.67x slow motion. The output line of g1_npz_to_qpos_csv.py shows the source "
  "fps. When --motion is an origin NPZ, the frame rate comes from the NPZ's fps key "
  "and this knob is ignored."
)

_MOTION_HELP = (
  "Reference motion: an origin robot NPZ (recommended; root_pos/root_quat/joint_pos, "
  "fps included) or a 38-column qpos CSV (needs --fps). Default None = zero standing "
  "pose. Do not feed the augmenter's output (95-column CSV / motion-library NPZ): it "
  "is a derived product and will be rejected."
)


def main():
  import argparse

  ap = argparse.ArgumentParser(description="Multi-link SoftMimic offline augmentation")
  sub = ap.add_subparsers(dest="cmd", required=True)

  m = sub.add_parser("motion", help="multi-link motion augmentation → adapted qpos CSV")
  m.add_argument("--fps", type=float, default=30.0, help=_FPS_HELP)
  m.add_argument("--mode", choices=["bimanual", "independent"], default="bimanual")
  m.add_argument("--seed", type=int, default=0)
  m.add_argument("--motion", default=None, help=_MOTION_HELP)
  m.add_argument("--com-cap", type=float, default=COM_CAP)
  m.add_argument("--with-torque", action="store_true")
  m.add_argument("--out", required=True)
  m.add_argument(
    "--contact-out",
    default=None,
    help="also write a multi-link contact log NPZ (both wrists); None (default) = CSV only",
  )
  m.add_argument(
    "--partner-aware-augmentation",
    action="store_true",
    help="force direction reads the dance-step phase: with --guide-prob, push along "
    "the partner's root velocity (shared by both modes); the rest take the connection "
    "branch: mode A pulls toward the partner along the heading, mode B resists with "
    "the frame against the heading. Infeasible backoff rotates the direction outward "
    "before shrinking.",
  )
  m.add_argument(
    "--partner-clip",
    default=None,
    help="partner (human) origin NPZ path; its root_lin_vel key (present in the tracked "
    "human NPZs) drives the phase-aligned guidance; goes together with "
    "--partner-aware-augmentation",
  )
  m.add_argument(
    "--guide-prob",
    type=float,
    default=GUIDE_PROB,
    help="event share of the common branch, push along the partner velocity "
    f"(default {GUIDE_PROB:g}, shared by both modes); the rest take the minor "
    "branch: mode A pulls toward the partner along the robot heading, mode B resists "
    "with the frame against the heading",
  )
  m.add_argument(
    "--single-prob",
    type=float,
    default=None,
    help="single-hand event share of the bimanual schedule (default "
    f"{SINGLE_HAND_PROB:g} when guided, 0 unguided); a single hand uses the full "
    "single-hand displacement range + the single-contact CoM budget. Explicit use "
    "without guidance requires --pin-free-wrists (with the free wrist unpinned, "
    "single-hand events hit slack-variable artifacts)",
  )
  m.add_argument(
    "--oppose-prob",
    type=float,
    default=0.0,
    help="event share of opposing internal-force pairs (squeeze/pull along the "
    "wrist-to-wrist line, shared stiffness, net external force about zero); 0 "
    "(default) consumes no random numbers",
  )
  m.add_argument(
    "--counter-prob",
    type=float,
    default=0.0,
    help="event share of yaw couples (the wrists pushing opposite ways along the "
    "horizontal perpendicular, like turning a steering wheel); 0 (default) as above",
  )
  m.add_argument(
    "--vertical-scale",
    type=float,
    default=0.3,
    help="scale of the sampled direction's vertical component (default 0.3 = mostly "
    "horizontal; 1.0 = SoftMimic-style isotropy, allowing full-magnitude downward "
    "drag / upward lift)",
  )
  m.add_argument(
    "--max-disp",
    type=float,
    default=0.15,
    help="displacement range cap, U(0.03, this value) (default 0.15; the stand pool "
    "widens it to 0.45 with --com-cap 0.15)",
  )
  m.add_argument(
    "--max-load-vel",
    type=float,
    default=None,
    help="loading-ramp velocity limit (m/s): ramp_up ≥ displacement/this value, the "
    "same rule as the release. Off by default; with a 0.15 range 1.0 never "
    "triggers, and 1.0 is recommended once the range is widened",
  )
  m.add_argument(
    "--pin-free-wrists",
    action="store_true",
    help="pin the free wrist in unguided builds too, and turn on attribution + the "
    "outward-rotation rescue (always on in guided builds, regardless of this flag). "
    "A prerequisite for single-hand/antisymmetric events on the stand pool",
  )
  m.add_argument(
    "--async-singles",
    action="store_true",
    help="decouple the single-hand event streams per wrist: a single may start "
    "whenever a wrist is free (the other still under force), producing left/right "
    "asynchronous overlap; two-hand events still need both wrists free. Off by "
    "default = the both-wrists-free gate (shares exactly equal the draw "
    "probabilities)",
  )
  m.add_argument(
    "--max-rot-disp",
    type=float,
    default=2.0,
    help="rotational displacement range cap τ/K_rot (rad) for torque sampling. "
    "Default 2.0 = the value ported from SoftMimic's open scenes; unreachable in a "
    "close-to-chest hold, where it only burns the backoff ladder; the paper recipe "
    "(just multilink-rc-paper) narrows it to 0.5 (about 29 degrees)",
  )
  m.add_argument(
    "--com-cost",
    type=float,
    default=0.1,
    help="XY weight of the IK CoM task. Default 0.1 = the value SoftMimic runs with "
    "(its CLI default; its class default is 0.5)",
  )
  m.add_argument(
    "--com-cost-z-factor",
    type=float,
    default=0.00001,
    help="Z weight multiplier of the CoM task. Default 1e-5 = SoftMimic's value "
    "(height free, so vertical yielding can squat); 1.0 pins the height (bending "
    "over only)",
  )

  z = sub.add_parser(
    "zerowrench",
    help="zero-contact multi-link contact NPZ (the contact file of a zero-wrench pair; frame "
    "count from the origin)",
  )
  z.add_argument("--motion", required=True, help="origin robot NPZ (frame count)")
  z.add_argument("--out", required=True)
  z.add_argument("--seed", type=int, default=0, help="hold-and-redraw stiffness seed")
  z.add_argument(
    "--mode",
    choices=["bimanual", "independent"],
    default="bimanual",
    help="bimanual (default) = all channels share the hold-segment boundaries "
    "(matching the switching structure of bimanual force events); independent = "
    "separate boundaries per channel (ablation)",
  )
  z.add_argument(
    "--with-torque",
    action="store_true",
    help="put the rotational stiffness on the hold-and-redraw ladder (pairs with "
    "force clips from motion --with-torque); by default pinned to the 1.0 "
    "placeholder (pairs with force clips without torque augmentation, so a changing "
    "K_rot does not give away the clip type)",
  )

  args = ap.parse_args()
  if args.cmd == "zerowrench":
    # No solver, only stiffness sampling: dispatch early (the DATA_FPS/solver setup
    # below is not needed).
    if not os.path.exists(args.motion):
      ap.error(f"--motion file does not exist: {args.motion}")
    t = write_zerowrench_contact(
      args.motion,
      args.out,
      run_meta=build_datagen_meta(source=str(args.motion), seed=args.seed),
      seed=args.seed,
      mode=args.mode,
      with_torque=args.with_torque,
    )
    rot = "rot ladder" if args.with_torque else "rot pinned at 1.0"
    print(
      f"[zerowrench] {t}-frame zero-force multi-link contact NPZ (hold-and-redraw "
      f"stiffness, {args.mode}, {rot}) → {args.out}"
    )
    return
  # Partner guidance: the two flags come as a pair (--partner-clip alone would
  # silently do nothing, so it must be rejected).
  guide = None
  if args.partner_aware_augmentation:
    if not args.partner_clip:
      ap.error("--partner-aware-augmentation requires --partner-clip")
    if not os.path.exists(args.partner_clip):
      ap.error(f"--partner-clip file does not exist: {args.partner_clip}")
    guide = PartnerGuide(args.partner_clip, guide_prob=args.guide_prob)
    print(
      f"[partner] partner guidance on: {args.partner_clip} "
      f"(guide_prob={args.guide_prob:g})"
    )
  elif args.partner_clip:
    ap.error("--partner-clip must be used with --partner-aware-augmentation")
  if args.single_prob is not None and guide is None and not args.pin_free_wrists:
    ap.error(
      "--single-prob without guidance requires --pin-free-wrists (with the free "
      "wrist unpinned, single-hand events hit QP slack-variable artifacts)"
    )
  single_prob = (
    args.single_prob
    if args.single_prob is not None
    else (SINGLE_HAND_PROB if guide is not None else 0.0)
  )
  lottery = single_prob + args.oppose_prob + args.counter_prob
  if lottery > 1.0 + 1e-9:
    ap.error(f"single/oppose/counter probabilities sum to {lottery:g} > 1")
  if guide is not None or lottery > 0:
    print(
      f"[sampler] single={single_prob:g} oppose={args.oppose_prob:g} "
      f"counter={args.counter_prob:g} vertical={args.vertical_scale:g} "
      f"max_disp={args.max_disp:g} load_vel={args.max_load_vel} "
      f"pin={'auto(guided)' if guide is not None else args.pin_free_wrists}"
    )
  # Must precede building the solver: DATA_FPS is a class attribute, and
  # CsvMotionLib computes dt/total duration from it at construction.
  solver_module.CsvMotionLib.DATA_FPS = args.fps
  # A given --motion must exist: the single-link solver **silently degrades to a
  # standing pose** on a missing path, which here would put a clip's name on a
  # standing-pose augmentation, so it must be rejected.
  if args.motion is not None and not os.path.exists(args.motion):
    ap.error(f"--motion file does not exist: {args.motion}")
  if args.motion is not None and args.motion.endswith(".npz"):
    # Take the origin robot NPZ directly (single source of truth): geometry and
    # frame rate both come from it, with no 38-column CSV bridge and no manual repeat
    # of --fps (one wrong fps = the whole clip changes speed).
    from g1_npz_to_qpos_csv import npz_fps, npz_to_qpos_rows

    d = np.load(args.motion)
    fps = npz_fps(d)
    if fps is not None:
      solver_module.CsvMotionLib.DATA_FPS = fps
      print(f"reference NPZ carries fps={fps:g}, using it (--fps is ignored).")
    else:
      print(f"[WARN] reference NPZ has no fps key, falling back to --fps {args.fps:g}.")
    solver = solver_module.G1_Mink_IK_Solver(
      motion_path=None,
      com_cost=args.com_cost,
      com_cost_z_factor=args.com_cost_z_factor,
    )
    solver.motion_lib = solver_module.CsvMotionLib(npz_to_qpos_rows(d))
    solver._initialize_reference_pose()  # like _load_motion: IK init = t=0 reference
  else:
    solver = solver_module.G1_Mink_IK_Solver(
      motion_path=args.motion,
      com_cost=args.com_cost,
      com_cost_z_factor=args.com_cost_z_factor,
    )
  if args.cmd == "motion":
    # The duration is always the full reference (mirrors single-link: motion_duration
    # = motion_lib.get_max_time()), so the produced clip aligns frame by frame with
    # the original reference and the manifest's free_motion_file can point straight
    # at the original NPZ.
    duration = solver.motion_lib.get_max_time() if solver.motion_lib else STAND_DURATION
    n = generate_motion(
      solver, duration, args.mode, args.seed, args.out,
      com_cap=args.com_cap, with_torque=args.with_torque,
      contact_out=args.contact_out,
      run_meta=build_datagen_meta(
        source=str(args.motion) if args.motion else None, seed=args.seed
      ),
      guide=guide,
      single_prob=single_prob,
      oppose_prob=args.oppose_prob,
      counter_prob=args.counter_prob,
      vertical_scale=args.vertical_scale,
      max_disp=args.max_disp,
      max_load_vel=args.max_load_vel,
      pin_free_wrists=(None if guide is not None else args.pin_free_wrists or None),
      async_singles=args.async_singles,
      max_rot_disp=args.max_rot_disp,
    )  # fmt: skip
    msg = (
      f"[motion] wrote {n} frames of adapted qpos ({duration:.2f}s @ "
      f"{solver_module.CsvMotionLib.DATA_FPS:g}fps source)"
      f" → {args.out} (no header, the bridge's input)\n"
      f"          + {labeled_path(args.out)} (36 named columns, for humans, **not for "
      "the bridge**)\n"
      f"next: uv run python scripts/qpos_csv_to_motion_npz.py "
      f"--input-file {args.out} --output-file <clip>.npz"
    )
    if args.contact_out is not None:
      msg += (
        f"\n[contact] multi-link contact log NPZ → {args.contact_out} (consumed by the "
        "online task).\n"
        f"to preview: MUJOCO_GL=egl uv run python scripts/clip_viz/"
        f"render_clip_video.py --clip <clip>.npz --contact {args.contact_out} "
        "--out <clip>.mp4 (labels the forced link on every frame)"
      )
    print(msg)


if __name__ == "__main__":
  main()
