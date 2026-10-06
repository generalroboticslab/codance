#!/usr/bin/env python3
# ============================================================================
# SoftMimic offline compliant data augmentation pipeline: faithful self-contained port
# ============================================================================
#
# [What this is]
#   The offline pipeline "apply force -> mink IK re-solve -> feasibility rejection
#   sampling -> augmented pose", spread over 9 modules under
#   compliant_motion_augmentation/ in the upstream SoftMimic repository, moved as-is
#   into one standalone script in the mjlab repo, so you can run it, watch it, and
#   judge the quality of this offline pipeline directly.
#
#   Equivalent original command:
#       python mink_generator_ff.py interactive \
#              --motion_path ../datasets/motions_csv/stand.csv --force_mode forcefield
#
#   Recommended use in this repo: decouple solving from viewing. Solve offline once
#   and save a CSV, then replay it freely (solving is slow, replay is fast).
#     1) Solve offline once (no window; IK solve + write each frame's adapted
#        pose/force/setpoint to the CSV):
#        MUJOCO_GL=egl uv run python scripts/softmimic_mink_augment.py generate-data \
#            --force_mode forcefield --num_files 1 --output_dir ./aug_out
#     2a) Replay the previous step's CSV smoothly in the viewer (no IK solve, so no
#         stutter; needs a local display):
#        uv run python scripts/softmimic_mink_augment.py replay \
#            --replay_csv ./aug_out/stand_augmented_mink_001.csv
#     2b) Or re-render the same CSV offscreen to mp4 (no display needed):
#        MUJOCO_GL=egl uv run python scripts/softmimic_mink_augment.py replay \
#            --replay_csv ./aug_out/stand_augmented_mink_001.csv --record_video \
#            --max_seconds 20 --output_filename softmimic_forcefield_demo.mp4
#   (generate-data also supports --record_video for a video in one step; interactive
#   solves while you watch, stutters, and is not recommended.)
#
# [Its role here]
#   The adapted poses the policies train on are precomputed offline, "after being
#   pushed by the force", as SoftMimic publishes them. This script is SoftMimic's
#   single-link precomputation step; scripts/augment_multilink.py builds the
#   multi-contact augmenter on its IK solver and constants.
#
# [Physical intuition for this pipeline (compare while watching the animation)]
#   - The robot stands/jumps following the reference motion (the default
#     data/stand.csv, SoftMimic's standing clip, is not shipped; without it the
#     reference is a static standing pose).
#   - Every so often, a random external wrench (force + torque) is applied to one
#     "forceable link" (by default only the two wrists); the force magnitude,
#     direction and duration are all randomly sampled.
#   - The spring law pushes the Cartesian target of the loaded point away from the
#     reference: p_target = p_ref + F / K_robot. mink IK re-solves the whole-body
#     pose under a set of constraints ("feet fixed, CoM still inside the support
#     polygon, upper body as close to the reference as possible") so the loaded link
#     reaches the displaced target. That is "compliance": yield a little when pushed.
#   - After solving, a feasibility check runs (do force/torque/displacement/foot
#     displacement/CoM tracking error exceed their limits?). If infeasible, the event
#     is rolled back and retried with the force scaled by 0.8, until it is feasible or
#     given up. Only feasible frames enter the dataset: that is "rejection sampling".
#
# [The five force modes, force_mode]
#   triangle             : plain triangle-wave external wrench (ramp up, hold, ramp
#                          down), the simplest.
#   forcefield           : (the mode evaluated by default) on top of the
#                          triangle-wave force, also models "the environment has a
#                          stiffness K_forcefield"; the loaded point's target is
#                          pushed by two springs in series: F/K_forcefield + F/K_robot.
#   collision-emulator   : models the environment as a spring with a fixed setpoint,
#                          force = equivalent stiffness * (setpoint - p_ref).
#   collision-emulator-1d: generates a "collision plane" along the direction of
#                          motion; a 1D normal contact force from the penetration
#                          depth, plus a random torque.
#   zero-wrench          : applies no force, only randomizes the robot's own
#                          stiffness. This is the "zero-wrench / free" branch, used
#                          to show that with F=0 the compliance reward reduces to
#                          plain tracking.
#
# [Differences from the original (engineering only, physics unchanged)]
#   1) The original reads the CSV with a 600-line torch ProceduralMotionLibFromDemo,
#      but this pipeline's IK solver uses only 4 of its outputs. Here ~40 lines of
#      pure numpy reproduce its sampling semantics (30fps, linear pos/dof
#      interpolation, quaternion nlerp, foot contacts from the left frame) with
#      bit-identical results, dropping the torch dependency along the way.
#   2) The original generate-data saves the CSV with pandas; numpy.savetxt is used
#      here instead (this repo's venv has no pandas), with the same column layout.
#   3) The model is mjlab's native G1 rather than a model file in the upstream
#      repository, composed via mjlab's MjSpec scene API
#      (build_g1_scene_model) into "robot + checker ground + two directional lights
#      + skybox" before compiling, so offscreen renders are not pitch black. attach
#      uses an empty prefix, so link/joint names stay as-is and the IK code is
#      unchanged. Verified: the mjlab and upstream G1 models share link/joint names
#      and the order of the 29 non-free joints; the mjlab model has no actuators, so
#      actuated joint indices are derived from the "non-free joints" instead (same
#      result on both models). --model_path switches to any XML that carries its own
#      lighting.
#   Everything else (IK task weights, stiffness sampling ranges, feasibility
#   thresholds, rollback logic, visualization overlays) is copied line by line.
#
# Dependencies: mujoco / mink / scipy / numpy (+ imageio only for video recording).
# All are already in this repo's venv.
# ============================================================================

from __future__ import annotations

import argparse
import multiprocessing
import os
import random
import time
from dataclasses import dataclass
from dataclasses import replace as dataclasses_replace
from typing import Any, Dict, List, Optional, Tuple

import mink
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# mjlab's native G1 + scene composition API: compose "robot + ground + lights +
# skybox" into a lit render scene, instead of rendering the "bare robot" from
# asset_zoo (the bare robot has no ground/skybox and renders pitch black).
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML, get_g1_robot_cfg
from mjlab.entity.entity import Entity
from mjlab.utils.spec_config import LightCfg, MaterialCfg, TextureCfg

try:
  import imageio  # only used with --record_video
except ImportError:  # pragma: no cover
  imageio = None


# ============================================================================
# Section 0: constants (copied line by line from the original constants.py)
# ============================================================================

# Forceable links: only the two wrists, i.e. external forces only land on the hands,
# matching the "a person pushes the robot's hands" setting.
FORCEABLE_LINKS = [
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
]

# Links that may only be pushed downward (the force's z component is forced
# negative). The wrists are not in this list in the default config, so it never
# fires, but it is kept for fidelity (it still applies if FORCEABLE_LINKS changes).
DOWNWARD_ONLY_FORCEABLE_LINKS = [
  "torso_link",
  "left_shoulder_pitch_link",
  "right_shoulder_pitch_link",
]

# Keypoint links: IK attaches "soft" targets (at the reference pose) to these links,
# so the re-solved whole-body pose stays close to the reference overall.
KEYPOINT_BODY_NAMES = [
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
  "left_elbow_link",
  "right_elbow_link",
  "left_shoulder_yaw_link",
  "right_shoulder_yaw_link",
  "left_hip_pitch_link",
  "right_hip_pitch_link",
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "torso_link",
  "pelvis",
]

FOOT_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]

# Release phase: after an event ends, the loaded link slides back to the reference
# pose at a limited speed, avoiding pose jumps.
MAX_RELEASE_LINEAR_VEL = 1.0  # m/s
MAX_RELEASE_ANGULAR_VEL = 2.0  # rad/s
# A reference XY jump larger than this counts as a "teleport": the IK state is
# translated along with it.
TELEPORT_THRESHOLD = 1.0  # m

# Feasibility thresholds: exceeding any one marks the frame infeasible, which
# triggers rollback / retry at a reduced magnitude (the core rejection-sampling gate).
IK_CHECK_MAX_IK_TRACKING_ERROR = 0.05  # loaded-link position error vs desired (m)
IK_CHECK_MAX_FOOT_DISP = 0.05  # foot displacement vs reference (m); feet stay put
IK_CHECK_MAX_COM_TRACKING_ERROR = 0.15  # CoM XY tracking error (m); no tipping over
IK_CHECK_MAX_FORCE_MAGNITUDE = 140.0  # external force magnitude cap (N)
IK_CHECK_MAX_DISPLACEMENT_MAGNITUDE = 0.7  # spring displacement F/K cap (m)
IK_CHECK_MAX_TORQUE_MAGNITUDE = 10.0  # external torque magnitude cap (N·m)
IK_CHECK_MAX_ROTATIONAL_DISPLACEMENT_MAGNITUDE = 2.0  # rotational τ/K_rot cap (rad)

# Stiffness sampling ranges (log-uniform). Unified compliance vocabulary:
#   robot stiffness  = K_rob  = the stiffness the robot itself should exhibit
#                               (*_ROBOT_STIFFNESS here)
#   force-field stiffness = K_env = environment stiffness (*_FORCEFIELD_STIFFNESS here)
# Under an external force F, the target pose moves from the reference pose to the
# "equilibrium point" p_eq = p_ref + F/K. forcefield mode puts two springs in series:
# p_eq = p_ref + F/K_rob + F/K_env (see the target formula in ik_update).
# These are the default ranges; at runtime SimulationConfig's *_stiffness_min/max can
# override them (_apply_config_overrides).
MIN_ROBOT_STIFFNESS = 10.0
MAX_ROBOT_STIFFNESS = 1000.0
MIN_ROBOT_ROTATIONAL_STIFFNESS = 0.1
MAX_ROBOT_ROTATIONAL_STIFFNESS = 10.0
MIN_FORCEFIELD_STIFFNESS = 10.0
MAX_FORCEFIELD_STIFFNESS = 1000.0
MIN_FORCEFIELD_ROTATIONAL_STIFFNESS = 0.1
MAX_FORCEFIELD_ROTATIONAL_STIFFNESS = 10.0

# Default reference: data/stand.csv, SoftMimic's standing clip, which this
# repository does not ship; when it is missing the solver uses a static standing
# pose. To use a tracked clip, convert it with scripts/g1_npz_to_qpos_csv.py and
# pass --motion_path (with --fps 50).
DEFAULT_MOTION_PATH = os.path.abspath(
  os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "stand.csv")
)


def build_g1_scene_model(offwidth: int = 640, offheight: int = 480) -> mujoco.MjModel:
  """Compose a G1 scene model "with ground/lights/skybox" via mjlab's scene
  composition API, compile it and return it.
  ⚠️ This model is for "rendering" only, not for IK solving (see below).

  Why it is needed: g1.xml in asset_zoo is a "bare robot" with no ground and no
  skybox; offscreen renders show an all-black background with a grey figure floating
  in the void. mjlab's env/viewer attaches the robot via MjSpec into a world with
  ground + lights and then compiles it; this reproduces the same approach. attach
  uses prefix="" (empty
  prefix), so robot body/joint names stay as-is and the qpos layout is bit-identical
  to the bare model (nq=36); rendering just copies the IK-solved qpos over and calls
  mj_forward.

  Why not use it for IK: with the ground composed in, mink's ComTask tracks the
  "whole-model CoM" and its daqp QP returns NaN on this model; and the original
  pipeline's COP/CoM/feasibility physics assumes "robot only" anyway. So IK still
  runs on the bare robot model (G1_XML, robot-only, verified solvable) and this
  scene model is for rendering only. Both models share the qpos layout, so copying
  every frame keeps them in sync. Verified: link/joint names match the upstream
  model, and the 29 non-free joints are in the same order.
  """
  spec = mujoco.MjSpec()
  spec.modelname = "g1_softmimic_aug_scene"

  # Skybox (blue to black gradient); otherwise the background is all black.
  TextureCfg(
    name="skybox",
    type="skybox",
    builtin="gradient",
    width=512,
    height=512,
    rgb1=(0.3, 0.5, 0.7),
    rgb2=(0.0, 0.0, 0.0),
  ).edit_spec(spec)
  # Checker ground texture + material + plane geom.
  TextureCfg(
    name="groundplane",
    type="2d",
    builtin="checker",
    mark="edge",
    rgb1=(0.2, 0.3, 0.4),
    rgb2=(0.1, 0.2, 0.3),
    markrgb=(0.8, 0.8, 0.8),
    width=300,
    height=300,
  ).edit_spec(spec)
  MaterialCfg(
    name="groundplane",
    texuniform=True,
    texrepeat=(4, 4),
    reflectance=0.2,
    texture="groundplane",
  ).edit_spec(spec)
  ground = spec.worldbody.add_body(name="ground")
  ground.add_geom(
    name="ground",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(0, 0, 0.01),
    material="groundplane",
  )
  # Two directional lights (one overhead casting shadows, one front-side fill light
  # without shadows).
  LightCfg(pos=(0, 0, 3.0), dir=(0, 0, -1), type="directional").edit_spec(spec)
  LightCfg(
    pos=(1.5, 1.5, 2.0), dir=(-1, -1, -1), type="directional", castshadow=False
  ).edit_spec(spec)

  # Attach mjlab's native G1 robot into the world with an empty prefix.
  robot = Entity(get_g1_robot_cfg())
  frame = spec.worldbody.add_frame()
  spec.attach(robot.spec, prefix="", frame=frame)

  # Offscreen resolution cap: MuJoCo's default offscreen framebuffer is only 640×480,
  # 2K/4K recording must raise it.
  # A render-only scene model, so the FBO is sized to the requested resolution.
  spec.visual.global_.offwidth = offwidth
  spec.visual.global_.offheight = offheight

  model = spec.compile()
  assert model is not None
  return model


# ============================================================================
# Section 1: run configuration (the original config.py)
# ============================================================================


@dataclass
class SimulationConfig:
  """Full config of one simulation/data-generation run; fields map 1:1 to CLI args."""

  # "interactive" (solve + watch live) / "generate-data" (solve + offscreen render) /
  # "replay" (replay a generated CSV)
  mode: str
  model_path: Optional[str]  # None → compose a lit G1 scene via the mjlab API
  motion_path: str
  seed: int
  com_cost: float
  com_cost_z_factor: float
  force_mode: str
  torso_orientation_cost: float = 0.0
  repeat_frame_time: Optional[float] = (
    None  # if set, freeze the reference at this time (handy for static force tests)
  )
  record_video: bool = False
  output_filename: str = "ik_simulation.mp4"
  render_width: int = (
    640  # offscreen video size; 2K = 1920×1080 (FBO follows, see build_g1_scene_model)
  )
  render_height: int = 480
  cam_azimuth: float = (
    90.0  # video camera (side view); smaller distance / other azimuth for a 3/4 view
  )
  cam_elevation: float = -15.0
  cam_distance: float = 4.0
  num_files: int = 10
  output_dir: str = "./augmented_data_mink"
  max_seconds: Optional[float] = (
    None  # if set, run only this many seconds (keeps offline videos watchable)
  )
  replay_csv: Optional[str] = None  # replay mode: path of the generated CSV to replay
  # ── Tunable knobs (unified compliance vocabulary, for large-scale force sweeps) ──
  # fps: true frame rate of the reference motion; also sets the sim/record step rate,
  # so a 50Hz source is augmented at its true speed.
  fps: float = 30.0
  # robot stiffness = K_rob (the stiffness the robot itself should exhibit);
  # force-field stiffness = K_env (environment stiffness). Both turn the external
  # force into an "equilibrium point" displacement F/K: forcefield mode puts the two
  # springs in series (p_eq = p_ref + F/K_rob + F/K_env). Log-uniform sampling ranges.
  robot_stiffness_min: float = 10.0
  robot_stiffness_max: float = 1000.0
  forcefield_stiffness_min: float = 10.0
  forcefield_stiffness_max: float = 1000.0
  max_force: float = 140.0  # force cap (N): feasibility gate and sampling upper bound
  # Forceable links (comma-separated); empty = the default two wrists. Swap in other
  # links to see compliance when "pushing different body parts".
  forceable_links: Optional[str] = None
  # Also save the augmented "adapted pose" trajectory as a plain qpos CSV
  # ([pos3, quat_xyzw4, 29 joints]) that qpos_csv_to_motion_npz.py turns into an AMP
  # expert motion NPZ. Only file 0 is recorded.
  export_qpos_csv: Optional[str] = None


# ============================================================================
# Section 2: reference motion library, a minimal pure-numpy port for 38-column CSVs
# ============================================================================


class CsvMotionLib:
  """Read a CSV reference motion and interpolate (root_pos, root_rot, dof_pos,
  foot_contacts) at any time t.

  Faithfully reproduces the sampling semantics of the original
  ProceduralMotionLibFromDemo:
    * fixed 30fps frame rate (note: the original passes motion_dt=0.02 at
      construction, but data_fps is hard-coded to 30 internally and the motion_dt
      passed in is never used; this reproduces its "actual behavior");
    * linear interpolation of root_pos / dof_pos;
    * quaternion nlerp (pick the shortest path by the dot-product sign, interpolate
      linearly, then normalize);
    * foot contacts are discrete, taken from the left frame (not interpolated).
  CSV column layout (38 columns): [root_pos(3) | root_quat_xyzw(4) | 29 joints |
  2 foot contacts].
  The original's dozen or so other outputs (keypoints, velocities, projected
  gravity, yaw, ...) are never touched by this pipeline's IK solver, so they are
  all omitted (no compute for fields nobody reads).
  """

  DATA_FPS = 30.0  # = original 30.0 * speed (default 1.0)
  NUM_JOINTS = 29

  def __init__(self, csv_path: "str | np.ndarray"):
    if isinstance(csv_path, np.ndarray):
      # (T,38) rows already in memory (e.g. read straight from an NPZ)
      raw = np.asarray(csv_path, dtype=np.float64)
    else:
      raw = np.genfromtxt(csv_path, delimiter=",")
    if raw.ndim == 1:
      raw = raw.reshape(1, -1)
    expected = 7 + self.NUM_JOINTS + 2  # 38
    # Pad missing columns on the right with 1 (as the original does; 36/37 columns =
    # a valid reference without foot contacts).
    if raw.shape[1] < expected:
      pad = np.ones((raw.shape[0], expected - raw.shape[1]))
      raw = np.hstack([raw, pad])
    elif raw.shape[1] > expected:
      # Never silently truncate a wide CSV: a valid reference CSV has exactly 38
      # columns (stand / GMR / the g1_npz_to_qpos_csv bridge); a wider one is almost
      # certainly the **augmenter's output** (95 columns = ref|adapted|contact) fed in
      # by mistake. Truncating it would replay only its embedded ref block, which runs
      # fine but uses a derived artifact as the data source. The reference motion has
      # one source of truth: the original reference.
      raise ValueError(
        f"Reference motion CSV should have ≤{expected} columns [root_pos(3), "
        f"root_quat_xyzw(4), {self.NUM_JOINTS} joints, 2 foot contacts], got "
        f"{raw.shape[1]}. A 95-column file is the augmenter's output "
        "(ref|adapted|contact), not an input; feed the original reference instead "
        "(a 38-column qpos CSV, or the origin robot NPZ directly)."
      )
    self.data = raw.astype(np.float64)
    self.num_frames = self.data.shape[0]
    self.dt = 1.0 / self.DATA_FPS
    # Column slices of each block
    self._pos = slice(0, 3)
    self._quat = slice(3, 7)  # xyzw
    self._joints = slice(7, 7 + self.NUM_JOINTS)
    self._feet = slice(7 + self.NUM_JOINTS, 9 + self.NUM_JOINTS)

  def get_max_time(self) -> float:
    """Total motion duration = frame count * dt (the original get_max_times logic)."""
    return self.num_frames * self.dt

  def get_motion_state(self, t: float) -> Dict[str, np.ndarray]:
    """Interpolate at time t. Returns a dict of numpy arrays; quaternions are xyzw."""
    # Frame index: floor(t*fps + 1e-6); the original uses that 1e-6 to absorb
    # floating-point error and snap onto whole frames.
    ft = t * self.DATA_FPS
    i0_raw = int(np.floor(ft + 1e-6))
    alpha = ft - i0_raw  # interp weight in [0,1); the clamp below handles t<0 / overrun
    # As in the original, clamp the left frame to num_frames-2, leaving room for +1.
    upper = max(0, self.num_frames - 2)
    i0 = min(max(i0_raw, 0), upper)
    i1 = min(max(i0_raw + 1, 0), upper + 1)
    d0, d1 = self.data[i0], self.data[i1]

    root_pos = d0[self._pos] + alpha * (d1[self._pos] - d0[self._pos])
    dof_pos = d0[self._joints] + alpha * (d1[self._joints] - d0[self._joints])

    q0 = d0[self._quat].copy()
    q1 = d1[self._quat].copy()
    if np.dot(q0, q1) < 0.0:  # double cover: flip one on a sign mismatch (shortest arc)
      q1 = -q1
    q = q0 + alpha * (q1 - q0)  # nlerp
    n = np.linalg.norm(q)
    root_rot = q / n if n > 1e-8 else q0

    foot_contacts = d0[self._feet].copy()  # discrete: left frame, not interpolated
    return {
      "root_pos": root_pos,
      "root_rot": root_rot,  # xyzw
      "dof_pos": dof_pos,
      "foot_contacts": foot_contacts,
    }


# ============================================================================
# Section 3: custom mink task, knee hyperextension guard (the original tasks.py)
# ============================================================================


class KneeBendingTask(mink.tasks.task.Task):
  """One-sided task penalizing knee "hyperextension": error/Jacobian are nonzero only
  when current angle < target angle, i.e. it only pulls the knee toward "more bent"
  and never lets it get straighter than the reference (stops IK from producing
  awkward reverse-bent knees)."""

  def __init__(self, model: mujoco.MjModel, cost: float, joint_names: List[str]):
    super().__init__(cost=cost)
    self.model = model
    self.cost = np.array([cost] * len(joint_names), dtype=np.float32)
    self.joint_names = joint_names
    self.target_q: Optional[np.ndarray] = None
    self.dof_indices: List[int] = []
    for name in self.joint_names:
      try:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        self.dof_indices.append(self.model.jnt_dofadr[joint_id])
      except KeyError:
        print(f"Warning: knee joint '{name}' not found; KneeBendingTask ignores it.")
    self.num_knees = len(self.dof_indices)

  def set_target(self, target_q: np.ndarray):
    self.target_q = target_q

  def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
    if self.target_q is None:
      return np.zeros(self.num_knees)
    error = np.zeros(self.num_knees)
    current_q = configuration.q
    for i, dof_idx in enumerate(self.dof_indices):
      joint_id = self.model.dof_jntid[dof_idx]
      qpos_adr = self.model.jnt_qposadr[joint_id]
      e = self.target_q[qpos_adr] - current_q[qpos_adr]
      error[i] = max(0.0, e)  # one-sided: only the "not bent enough" side
    return error

  def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
    jacobian = np.zeros((self.num_knees, self.model.nv))
    current_q = configuration.q
    for i, dof_idx in enumerate(self.dof_indices):
      joint_id = self.model.dof_jntid[dof_idx]
      qpos_adr = self.model.jnt_qposadr[joint_id]
      if current_q[qpos_adr] < self.target_q[qpos_adr]:
        jacobian[i, dof_idx] = -1.0
    return jacobian


# ============================================================================
# Section 4: G1 mink IK solver (original ik_solver.py without torch/keypoint precompute)
# ============================================================================


class G1_Mink_IK_Solver:
  """mink IK wrapped for the G1. All tasks/constraints/velocity limits are built once
  at construction; each sim step then only updates task targets and solves one QP."""

  def __init__(
    self,
    model_path: Optional[str] = None,
    motion_path: Optional[str] = None,
    repeat_frame_time: Optional[float] = None,
    com_cost: float = 0.5,
    com_cost_z_factor: float = 1.0,
    torso_orientation_cost: float = 0.0,
    waist_cost: float = 0.01,
    knee_cost: float = 0.01,
  ):
    # IK model = mjlab's native "bare robot" (robot-only). The COP/CoM/feasibility
    # physics all assume the robot alone, so the scene model with ground must never be
    # used here (with ground, mink ComTask's daqp returns NaN; see
    # build_g1_scene_model). The lit scene model for rendering is built separately in
    # the runner. model_path empty → mjlab's native G1_XML; given → load that XML.
    self.model = mujoco.MjModel.from_xml_path(model_path if model_path else str(G1_XML))
    self.configuration = mink.Configuration(self.model)
    self.data = self.configuration.data
    self.total_mass = sum(self.model.body_mass)  # total mass, for the COP/CoM target

    # qpos indices of the 29 "non-free joints" (hinge joints), in ascending joint id.
    # The original derives them from actuators, but mjlab's native g1.xml has none
    # (nu=0), so joints are counted directly instead: more robust, and it gives the
    # exact same 29 indices and order on both the upstream and mjlab models
    # (verified), matching the CSV dof order (left leg 6 → right leg 6 → waist 3 →
    # left arm 7 → right arm 7).
    actuated_joint_ids = [
      i
      for i in range(self.model.njnt)
      if self.model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    ]
    self.actuated_qpos_indices = [
      self.model.jnt_qposadr[jid] for jid in actuated_joint_ids
    ]
    self.num_dofs = len(self.actuated_qpos_indices)

    # IK tasks: the larger the cost, the "harder" the task (the higher its priority).
    # Posture task: very weak (1e-4); gently pulls the whole solution toward the
    # reference joint angles as regularization, keeping the null space from drifting.
    self.posture_task = mink.PostureTask(self.model, cost=1e-4)
    # CoM task: cost=com_cost in XY, times a small factor in Z (CoM height matters
    # little when standing).
    com_cost_vector = np.array([com_cost, com_cost, com_cost * com_cost_z_factor])
    self.com_task = mink.ComTask(cost=com_cost_vector)
    self.com_task.set_cost(com_cost_vector)

    # Waist task (optional): pulls waist roll/pitch/yaw toward the reference; the
    # cost defaults to 0.01.
    self.waist_task = None
    if waist_cost > 1e-5:
      waist_joint_names = ["waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint"]
      waist_cost_vector = np.zeros(self.model.nv)
      found_joints = []
      for name in waist_joint_names:
        try:
          joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
          dof_adr = self.model.jnt_dofadr[joint_id]
          waist_cost_vector[dof_adr] = waist_cost
          found_joints.append(name)
        except KeyError:
          print(f"Warning: waist joint '{name}' not in the model; skipped.")
      if found_joints:
        self.waist_task = mink.PostureTask(self.model, cost=waist_cost_vector)

    # Torso orientation task (optional, off by default).
    self.torso_orientation_task = None
    if torso_orientation_cost > 1e-5:
      self.torso_orientation_task = mink.FrameTask(
        "torso_link",
        "body",
        position_cost=0.0,
        orientation_cost=np.array([torso_orientation_cost] * 3),
      )
    # Pelvis pitch task: very weak (0.03); gently keeps the pelvis orientation near
    # the reference so the upper body does not tilt as a whole.
    self.pelvis_pitch_task = mink.FrameTask(
      "pelvis", "body", position_cost=0.0, orientation_cost=np.array([0.03, 0.03, 0.03])
    )
    # Knee hyperextension guard (on by default with cost=0.01).
    self.knee_task = None
    if knee_cost > 1e-5:
      self.knee_task = KneeBendingTask(
        self.model, cost=knee_cost, joint_names=["left_knee_joint", "right_knee_joint"]
      )

    # Keypoint tasks: every keypoint link gets a "soft" 6D target (position cost=1e-2,
    # orientation 1e-3) that follows the reference pose.
    self.keypoint_tasks = {
      name: mink.FrameTask(name, "body", position_cost=1e-2, orientation_cost=1e-3)
      for name in KEYPOINT_BODY_NAMES
    }
    # Foot tasks: large cost (position 2.5, orientation 0.5) nearly pins the feet to
    # the reference: a standing pose must not move its feet.
    self.foot_tasks = {
      name: mink.FrameTask(name, "body", position_cost=2.5, orientation_cost=0.5)
      for name in FOOT_NAMES
    }
    # Force tasks: a hard target just for the loaded link (position 5.0, orientation
    # 1.0) that pulls it precisely to the "pushed by the force" position.
    self.force_tasks = {
      name: mink.FrameTask(name, "body", position_cost=5.0, orientation_cost=1.0)
      for name in FORCEABLE_LINKS
    }

    # Velocity limits + joint limits: keep IK steps physically in range and free of
    # sudden jumps.
    joint_names = [
      self.model.joint(i).name
      for i in range(self.model.njnt)
      if self.model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    ]
    velocity_limits = {name: np.pi * 2 for name in joint_names}
    self.velocity_limit = mink.VelocityLimit(self.model, velocity_limits)
    self.limits = [mink.ConfigurationLimit(self.model), self.velocity_limit]

    self.repeat_frame_time = repeat_frame_time
    self.static_qpos_ref: Optional[np.ndarray] = None
    self.static_foot_contacts_ref: Optional[np.ndarray] = None
    self.motion_lib: Optional[CsvMotionLib] = None
    if motion_path:
      self._load_motion(motion_path)
    self._initialize_reference_pose()

  def _load_motion(self, motion_path: str):
    if not os.path.exists(motion_path):
      print(f"Motion file not found: {motion_path} (falling back to a static stand)")
      return
    print(f"Loading reference motion from CSV: {motion_path}")
    self.motion_lib = CsvMotionLib(motion_path)
    print(
      f"Motion loaded: {self.motion_lib.num_frames} frames @ "
      f"{self.motion_lib.DATA_FPS:.0f}fps = {self.motion_lib.get_max_time():.2f}s"
    )

  def _initialize_reference_pose(self):
    """Put the robot in the t=0 reference pose as the IK initial guess."""
    qpos_ref, _, _ = self.get_reference_motion(0.0)
    self.configuration.update(q=qpos_ref)
    mujoco.mj_forward(self.model, self.data)

  def get_reference_motion(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Public reference-motion interface; handles three cases: frozen frame / no
    motion library fallback / normal interpolation."""
    # Case 1: --repeat_frame_time freezes the motion at one time (cached on first
    # read, constant afterwards).
    if self.repeat_frame_time is not None:
      if self.static_qpos_ref is None:
        q_ref, _, c_ref = self._get_raw_motion_from_lib(self.repeat_frame_time)
        self.static_qpos_ref = q_ref.copy()
        self.static_foot_contacts_ref = c_ref.copy()
      return (
        self.static_qpos_ref,
        np.zeros(self.model.nv),
        self.static_foot_contacts_ref,
      )
    # Case 2: no motion library → fall back to a default standing pose (z=0.77, unit
    # quaternion, both feet in contact).
    if not self.motion_lib:
      q_ref = np.zeros(self.model.nq)
      q_ref[2] = 0.77
      q_ref[3] = 1.0
      return q_ref, np.zeros(self.model.nv), np.array([1.0, 1.0])
    # Case 3: normal interpolation in time.
    return self._get_raw_motion_from_lib(t)

  def _get_raw_motion_from_lib(
    self, t: float
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Take the interpolated CsvMotionLib state and lay it out as a MuJoCo qpos."""
    md = self.motion_lib.get_motion_state(t)
    root_pos = md["root_pos"]
    root_rot = md["root_rot"]  # xyzw
    dof_pos = md["dof_pos"]
    foot_contacts = md["foot_contacts"]
    qpos_ref_new = np.zeros(self.model.nq)
    qpos_ref_new[0:3] = root_pos
    qpos_ref_new[3:7] = root_rot[[3, 0, 1, 2]]  # xyzw → MuJoCo wxyz
    min_dofs = min(len(dof_pos), self.num_dofs)
    for i in range(min_dofs):
      self_idx = self.actuated_qpos_indices[i]
      qpos_ref_new[self_idx] = dof_pos[i]
    return qpos_ref_new, np.zeros(self.model.nv), foot_contacts


# ============================================================================
# Section 5: physics, COP-aware CoM target + feasibility check (the original physics.py)
# ============================================================================


def calculate_cop_aware_com_target(
  ref_data: mujoco.MjData,
  total_mass: float,
  com_force_for_ik: np.ndarray,
  com_torque_for_ik: np.ndarray,
  task_body_name: str,
) -> np.ndarray:
  """Compute where the CoM should move once the external moment is accounted for.

  Intuition: the midpoint of the feet is the desired center of pressure (COP). The
  external force acts on task_body with a lever arm relative to the COP, producing a
  tipping moment, plus any external torque. For the net moment not to push the robot
  over, the CoM must shift in the horizontal plane in the direction that cancels it:
  Δ = (−M_y, M_x) / (m·g). This is the key to how SoftMimic lets the robot "take the
  push and stay standing": not by bracing against it, but by moving the center of
  mass to where gravity balances the external moment.
  """
  left_foot_pos = ref_data.body("left_ankle_roll_link").xpos
  right_foot_pos = ref_data.body("right_ankle_roll_link").xpos
  target_cop = (left_foot_pos + right_foot_pos) / 2.0

  force_application_pos = ref_data.body(task_body_name).xpos
  lever_arm = force_application_pos - target_cop  # lever arm about the COP

  moment_from_force = np.cross(lever_arm, com_force_for_ik)  # r × F = tipping moment
  total_moment_ext = moment_from_force + com_torque_for_ik  # plus the external torque

  mg = total_mass * 9.81
  ref_com = ref_data.subtree_com[0]
  if mg < 1e-3:
    return ref_com

  # Horizontal CoM shift: gravity mg cancels the net moment. The x/y signs follow the
  # cross-product convention.
  return ref_com + np.array([-total_moment_ext[1] / mg, total_moment_ext[0] / mg, 0.0])


def is_ik_solution_feasible(
  ik_data: mujoco.MjData,
  ref_data: mujoco.MjData,
  link_name: str,
  force_ext: np.ndarray,
  torque_ext: np.ndarray,
  stiffness: float,
  rotational_stiffness: float,
  total_mass: float,
) -> Tuple[bool, Dict[str, float]]:
  """Check whether this frame's IK solution is within the physical and tracking safety
  limits. Exceeding any one of them makes it infeasible.

  This is the "rejection sampling" gate: only frames that pass every check make it
  into the augmented dataset.
  Returns (is_feasible, dict of violation amounts).
  """
  violations: Dict[str, float] = {}
  is_feasible = True

  # 1) The external force magnitude must not be too large
  force_mag = np.linalg.norm(force_ext)
  violations["force_magnitude"] = force_mag
  if force_mag > IK_CHECK_MAX_FORCE_MAGNITUDE:
    is_feasible = False

  # 2) Spring displacement F/K must not be too large (otherwise the target is too far
  #    from the reference, which amounts to yanking the link)
  displacement_mag = force_mag / stiffness if stiffness > 1e-6 else 0.0
  violations["displacement_magnitude"] = displacement_mag
  if displacement_mag > IK_CHECK_MAX_DISPLACEMENT_MAGNITUDE:
    is_feasible = False

  # 3) External torque magnitude
  torque_mag = np.linalg.norm(torque_ext)
  violations["torque_magnitude"] = torque_mag
  if torque_mag > IK_CHECK_MAX_TORQUE_MAGNITUDE:
    is_feasible = False

  # 4) Rotational spring displacement τ/K_rot
  rot_displacement_mag = (
    torque_mag / rotational_stiffness if rotational_stiffness > 1e-6 else 0.0
  )
  violations["rotational_displacement_magnitude"] = rot_displacement_mag
  if rot_displacement_mag > IK_CHECK_MAX_ROTATIONAL_DISPLACEMENT_MAGNITUDE:
    is_feasible = False

  # 5) Loaded-link tracking error: distance between where IK actually put the link and
  #    the desired position (reference + F/K)
  p_ref_link = ref_data.body(link_name).xpos
  p_target_link = p_ref_link + force_ext / max(stiffness, 1e-6)
  p_ik_link = ik_data.body(link_name).xpos
  link_tracking_error = np.linalg.norm(p_ik_link - p_target_link)
  violations["link_tracking_error"] = link_tracking_error
  if link_tracking_error > IK_CHECK_MAX_IK_TRACKING_ERROR:
    is_feasible = False

  # 6) Foot displacement: both feet must stay essentially still
  max_foot_disp = 0.0
  for foot_name in FOOT_NAMES:
    p_ref_foot = ref_data.body(foot_name).xpos
    p_ik_foot = ik_data.body(foot_name).xpos
    foot_disp = np.linalg.norm(p_ik_foot - p_ref_foot)
    max_foot_disp = max(max_foot_disp, foot_disp)
  violations["max_foot_disp"] = max_foot_disp
  if max_foot_disp > IK_CHECK_MAX_FOOT_DISP:
    is_feasible = False

  # 7) CoM XY tracking error: solved CoM vs the COP-aware CoM target
  com_target = calculate_cop_aware_com_target(
    ref_data, total_mass, force_ext, torque_ext, link_name
  )
  com_ik = ik_data.subtree_com[0]
  com_tracking_error_xy = np.linalg.norm(com_ik[:2] - com_target[:2])
  violations["com_tracking_error"] = com_tracking_error_xy
  if com_tracking_error_xy > IK_CHECK_MAX_COM_TRACKING_ERROR:
    is_feasible = False

  return is_feasible, violations


# ============================================================================
# Section 6: random force profile generation (the original force_profile.py), which
# decides "when, on which link, and how much force"
# ============================================================================


def generate_random_force_profile(
  total_duration: float,
  possible_links: List[str],
  force_mode: str,
  ik_solver: G1_Mink_IK_Solver,
  config: SimulationConfig,
) -> List[Dict[str, Any]]:
  """Generate a sequence of "force events" for force_mode; each event holds start/end
  times, the loaded link, stiffness, amplitude, force/torque axes, etc. Events do not
  overlap (they are sorted by time and consumed in order)."""

  if config.force_mode == "zero-wrench":
    print(
      "zero-wrench mode: no force events (only the robot stiffness is randomized in "
      "the main loop)."
    )
    return []

  profile: List[Dict[str, Any]] = []
  temp_data = mujoco.MjData(ik_solver.model)

  # Precompute the reference motion's link speed profile (collision/forcefield modes
  # rejection-sample by speed: fast-moving links are more likely to get "hit").
  v_norm = 3.0
  if force_mode in ["collision-emulator", "collision-emulator-1d", "forcefield"]:
    print("Precomputing the reference motion's link speed profile...")
    all_velocities = []
    for t in np.arange(0.0, total_duration, 0.02 * 5):
      q_t, _, _ = ik_solver.get_reference_motion(t)
      q_t_minus_dt, _, _ = ik_solver.get_reference_motion(t - 0.02)
      vel_ref = np.zeros(ik_solver.model.nv)
      mujoco.mj_differentiatePos(ik_solver.model, vel_ref, 0.02, q_t_minus_dt, q_t)
      temp_data.qpos[:], temp_data.qvel[:] = q_t, vel_ref
      mujoco.mj_forward(ik_solver.model, temp_data)
      for link_name in possible_links:
        link_id = mujoco.mj_name2id(
          ik_solver.model, mujoco.mjtObj.mjOBJ_BODY, link_name
        )
        link_vel_vec = np.zeros(6)
        mujoco.mj_objectVelocity(
          ik_solver.model, temp_data, mujoco.mjtObj.mjOBJ_BODY, link_id, link_vel_vec, 0
        )
        all_velocities.append(np.linalg.norm(link_vel_vec[:3]))
    if all_velocities:
      # Normalize by the 90th percentile
      v_norm = max(np.percentile(all_velocities, 90), 0.5)
    print(f"Speed normalization factor v_norm = {v_norm:.2f} m/s")

  if force_mode == "collision-emulator-1d":
    # 1D collision-plane mode: spawn a plane along the link's direction of motion and
    # compute the normal contact force from the penetration depth.
    print("Precomputing all candidate collision events (no feasibility pre-check)...")
    candidate_events = []
    MIN_INTERACTION_TIME, SPAWN_BUFFER = 0.2, 0.01
    for t_spawn in np.arange(0.0, total_duration, 0.1):
      link_name = random.choice(possible_links)
      q_t, _, _ = ik_solver.get_reference_motion(t_spawn)
      q_t_minus_dt, _, _ = ik_solver.get_reference_motion(t_spawn - 0.02)
      vel_ref = np.zeros(ik_solver.model.nv)
      mujoco.mj_differentiatePos(ik_solver.model, vel_ref, 0.02, q_t_minus_dt, q_t)
      temp_data.qpos[:], temp_data.qvel[:] = q_t, vel_ref
      mujoco.mj_forward(ik_solver.model, temp_data)
      link_id = mujoco.mj_name2id(ik_solver.model, mujoco.mjtObj.mjOBJ_BODY, link_name)
      link_vel_vec = np.zeros(6)
      mujoco.mj_objectVelocity(
        ik_solver.model, temp_data, mujoco.mjtObj.mjOBJ_BODY, link_id, link_vel_vec, 0
      )
      link_vel, link_vel_mag = link_vel_vec[:3], np.linalg.norm(link_vel_vec[:3])
      # The slower the link, the likelier the rejection (slow links rarely hit things)
      if random.random() > np.clip(link_vel_mag / v_norm, 0.0, 1.0):
        continue
      p_ref_start = temp_data.body(link_name).xpos.copy()
      # Plane normal = against the link's motion (the plane meets the link head-on)
      plane_normal = link_vel / link_vel_mag if link_vel_mag > 1e-6 else np.zeros(3)
      if link_vel_mag > 1e-4 and np.dot(plane_normal, link_vel) > 0:
        plane_normal = -plane_normal
      plane_origin = p_ref_start - plane_normal * SPAWN_BUFFER
      plane_velocity_vec = plane_normal * 0.01  # the plane advances slowly
      penetration_times = []

      # Look ahead 1.5s and collect the window in which the link penetrates the plane
      for t_future in np.arange(t_spawn, min(t_spawn + 1.5, total_duration), 0.02):
        q_future, _, _ = ik_solver.get_reference_motion(t_future)
        temp_data.qpos[:] = q_future
        mujoco.mj_forward(ik_solver.model, temp_data)
        penetration = -np.dot(
          temp_data.body(link_name).xpos.copy() - plane_origin, plane_normal
        )
        if penetration > 0:
          penetration_times.append(t_future)

      if (
        penetration_times
        and (penetration_times[-1] - penetration_times[0]) >= MIN_INTERACTION_TIME
      ):
        event_params = {
          "stiffness": np.exp(
            random.uniform(np.log(MIN_ROBOT_STIFFNESS), np.log(MAX_ROBOT_STIFFNESS))
          ),
          "forcefield_stiffness": np.exp(
            random.uniform(
              np.log(MIN_FORCEFIELD_STIFFNESS), np.log(MAX_FORCEFIELD_STIFFNESS)
            )
          ),
        }

        # Also add a random torque within the contact window (ramp up, hold, ramp down)
        torque_duration = random.uniform(
          0.5, penetration_times[-1] - penetration_times[0]
        )
        torque_start_offset = random.uniform(
          0, (penetration_times[-1] - penetration_times[0]) - torque_duration
        )
        event_params["torque_start_time"] = penetration_times[0] + torque_start_offset
        event_params["torque_end_time"] = (
          event_params["torque_start_time"] + torque_duration
        )
        torque_hold_duration = random.uniform(0.15, 0.5) * torque_duration
        event_params["torque_ramp_duration"] = max(
          0.1, (torque_duration - torque_hold_duration) / 2.0
        )
        event_params["torque_hold_start_time"] = (
          event_params["torque_start_time"] + event_params["torque_ramp_duration"]
        )
        event_params["torque_hold_end_time"] = (
          event_params["torque_hold_start_time"] + torque_hold_duration
        )

        # Torque amplitude = rotational stiffness × rotational displacement, kept in a
        # sane range (resample until the interval is non-empty)
        torque_range, rot_stiff_range, rot_disp_range = (
          (0.0, 10.0),
          (0.3, 30.0),
          (0.0, 2.0),
        )
        while True:
          rot_stiff = np.exp(
            random.uniform(np.log(rot_stiff_range[0]), np.log(rot_stiff_range[1]))
          )
          lower_rd = max(rot_disp_range[0], torque_range[0] / rot_stiff)
          upper_rd = min(rot_disp_range[1], torque_range[1] / rot_stiff)
          if lower_rd < upper_rd:
            break
        event_params["rotational_stiffness"] = rot_stiff
        event_params["rotational_forcefield_stiffness"] = np.exp(
          random.uniform(
            np.log(MIN_FORCEFIELD_ROTATIONAL_STIFFNESS),
            np.log(MAX_FORCEFIELD_ROTATIONAL_STIFFNESS),
          )
        )
        torque_amplitude = rot_stiff * random.uniform(lower_rd, upper_rd)
        torque_axis = np.random.randn(3)
        torque_axis /= np.linalg.norm(torque_axis)
        event_params["torque_amplitude"] = torque_amplitude
        event_params["torque_axis"] = torque_axis
        if np.linalg.norm(event_params["torque_axis"]) < 1e-6:
          continue

        event_params["collision_plane_origin"] = plane_origin
        event_params["collision_plane_normal"] = plane_normal
        event_params["plane_velocity_vec"] = plane_velocity_vec
        event_params["initial_spawn_time"] = t_spawn

        hold_start_time = penetration_times[0]
        hold_end_time = penetration_times[-1]
        ramp_duration = max(0.1, (hold_end_time - hold_start_time) * 0.3)
        final_event = {
          "start_time": hold_start_time - ramp_duration,
          "hold_start_time": hold_start_time,
          "hold_end_time": hold_end_time,
          "end_time": hold_end_time + ramp_duration,
          "ramp_duration": ramp_duration,
          "link_name": link_name,
          **event_params,
        }
        candidate_events.append(final_event)

    # Sort candidates by start time and greedily pick non-overlapping ones
    candidate_events.sort(key=lambda x: x["start_time"])
    last_event_end_time = -np.inf
    for cand in candidate_events:
      if cand["start_time"] >= last_event_end_time:
        profile.append(cand)
        last_event_end_time = cand["end_time"]
  else:
    # triangle / forcefield / collision-emulator: schedule events one after another
    current_time = 0.0
    while current_time < total_duration:
      wait_duration = random.uniform(0.5, 1.5)  # gap between events
      start_time = current_time + wait_duration
      if start_time > total_duration:
        break
      link_name = random.choice(possible_links)
      event_duration = random.uniform(
        2.0, 4.0
      )  # note: unused in the forcefield/triangle branches, kept as in the original
      event_params: Dict[str, Any] = {}

      q_t_start, _, _ = ik_solver.get_reference_motion(start_time)
      temp_data.qpos[:] = q_t_start
      mujoco.mj_forward(ik_solver.model, temp_data)
      p_ref = temp_data.body(link_name).xpos.copy()
      r_ref = Rotation.from_matrix(temp_data.body(link_name).xmat.reshape(3, 3))

      if force_mode == "collision-emulator":
        # Fixed-setpoint spring: the setpoint is the link pose at the event start
        q_t_minus_dt, _, _ = ik_solver.get_reference_motion(start_time - 0.02)
        if start_time - 0.02 < 0:
          current_time = start_time + 0.5
          continue
        vel_ref = np.zeros(ik_solver.model.nv)
        mujoco.mj_differentiatePos(
          ik_solver.model, vel_ref, 0.02, q_t_minus_dt, q_t_start
        )
        temp_data.qvel[:] = vel_ref
        mujoco.mj_forward(ik_solver.model, temp_data)
        link_id = mujoco.mj_name2id(
          ik_solver.model, mujoco.mjtObj.mjOBJ_BODY, link_name
        )
        link_vel_vec = np.zeros(6)
        mujoco.mj_objectVelocity(
          ik_solver.model, temp_data, mujoco.mjtObj.mjOBJ_BODY, link_id, link_vel_vec, 0
        )
        if random.random() > np.clip(
          np.linalg.norm(link_vel_vec[:3]) / v_norm, 0.0, 1.0
        ):
          current_time = start_time + 0.5
          continue

        event_params["forcefield_setpoint_pos"] = p_ref
        event_params["forcefield_setpoint_rot"] = r_ref

        stiffness = np.exp(
          random.uniform(np.log(MIN_ROBOT_STIFFNESS), np.log(MAX_ROBOT_STIFFNESS))
        )
        rot_stiffness = np.exp(
          random.uniform(
            np.log(MIN_ROBOT_ROTATIONAL_STIFFNESS),
            np.log(MAX_ROBOT_ROTATIONAL_STIFFNESS),
          )
        )
        event_params.update(
          {
            "stiffness": stiffness,
            "forcefield_stiffness": np.exp(
              random.uniform(
                np.log(MIN_FORCEFIELD_STIFFNESS), np.log(MAX_FORCEFIELD_STIFFNESS)
              )
            ),
            "rotational_stiffness": rot_stiffness,
            "rotational_forcefield_stiffness": np.exp(
              random.uniform(
                np.log(MIN_FORCEFIELD_ROTATIONAL_STIFFNESS),
                np.log(MAX_FORCEFIELD_ROTATIONAL_STIFFNESS),
              )
            ),
          }
        )
        ramp_duration = random.uniform(0.2, 1.0)

      elif force_mode in ["forcefield", "triangle"]:
        # Both modes first sample the "target peak force/torque".
        # Force amplitude = stiffness × displacement, with displacement in [0, 0.7] and
        # force at most max_force (rejection sampling keeps the interval non-empty).
        force_range = (0.0, IK_CHECK_MAX_FORCE_MAGNITUDE)
        stiffness_range = (MIN_ROBOT_STIFFNESS, MAX_ROBOT_STIFFNESS)
        displacement_range = (0.0, 0.7)
        stiffness = np.exp(
          random.uniform(np.log(stiffness_range[0]), np.log(stiffness_range[1]))
        )
        lower_d = max(displacement_range[0], force_range[0] / stiffness)
        upper_d = min(displacement_range[1], force_range[1] / stiffness)
        if lower_d >= upper_d:
          current_time = start_time + 0.5
          continue

        amplitude = stiffness * random.uniform(lower_d, upper_d)
        force_axis = np.random.randn(3)
        force_axis /= np.linalg.norm(force_axis)
        if link_name in DOWNWARD_ONLY_FORCEABLE_LINKS:
          force_axis[2] = -abs(force_axis[2])  # downward-only link: z points down

        # Same for torque: torque = rotational stiffness × rotational displacement.
        torque_range = (0.0, 10.0)
        rot_stiff_range = (
          MIN_ROBOT_ROTATIONAL_STIFFNESS,
          MAX_ROBOT_ROTATIONAL_STIFFNESS,
        )
        rot_disp_range = (0.0, 2.0)
        rot_stiff = np.exp(
          random.uniform(np.log(rot_stiff_range[0]), np.log(rot_stiff_range[1]))
        )
        lower_rd = max(rot_disp_range[0], torque_range[0] / rot_stiff)
        upper_rd = min(rot_disp_range[1], torque_range[1] / rot_stiff)
        if lower_rd >= upper_rd:
          current_time = start_time + 0.5
          continue
        torque_amplitude = rot_stiff * random.uniform(lower_rd, upper_rd)
        torque_axis = np.random.randn(3)
        torque_axis /= np.linalg.norm(torque_axis)

        event_params["stiffness"] = stiffness
        event_params["rotational_stiffness"] = rot_stiff
        event_params["amplitude"] = amplitude
        event_params["force_axis"] = force_axis
        event_params["torque_amplitude"] = torque_amplitude
        event_params["torque_axis"] = torque_axis

        if force_mode == "forcefield":
          # forcefield also needs an "environment stiffness"; the series springs push
          # the target further out.
          event_params["forcefield_stiffness"] = np.exp(
            random.uniform(
              np.log(MIN_FORCEFIELD_STIFFNESS), np.log(MAX_FORCEFIELD_STIFFNESS)
            )
          )
          event_params["rotational_forcefield_stiffness"] = np.exp(
            random.uniform(
              np.log(MIN_FORCEFIELD_ROTATIONAL_STIFFNESS),
              np.log(MAX_FORCEFIELD_ROTATIONAL_STIFFNESS),
            )
          )

        # The ramp duration is derived from a "target linear speed":
        # ramp = peak force / (stiffness × target speed), clipped to [0.1, 2.0]s. The
        # loaded point is then pushed out at a near-constant speed, a smoother
        # transition.
        target_linear_velocity = random.uniform(0.1, 1.0)
        required_ramp_lin = 0.0
        if target_linear_velocity > 1e-6 and stiffness > 1e-6 and amplitude > 1e-6:
          required_ramp_lin = amplitude / (stiffness * target_linear_velocity)
        ramp_duration = np.clip(required_ramp_lin, 0.1, 2.0)

      # Rebuild the event timing from ramp_duration: ramp up, hold, ramp down
      hold_duration = random.uniform(0.5, 1.0)
      hold_start_time = start_time + ramp_duration
      hold_end_time = hold_start_time + hold_duration
      end_time = hold_end_time + ramp_duration
      if end_time > total_duration:
        break

      final_event = {
        "start_time": start_time,
        "end_time": end_time,
        "hold_start_time": hold_start_time,
        "hold_end_time": hold_end_time,
        "ramp_duration": ramp_duration,
        "link_name": link_name,
        **event_params,
      }
      profile.append(final_event)
      current_time = end_time
  return sorted(profile, key=lambda x: x["start_time"])


# ============================================================================
# Section 7: single-step IK update (the original ik_update.py), chaining "external
# force → target pose → QP solve"
# ============================================================================


def perform_mink_ik_step(
  ik_solver: G1_Mink_IK_Solver,
  qpos_ref: np.ndarray,
  current_task_body_name: str,
  force_ext: np.ndarray,
  torque_ext: np.ndarray,
  com_force_for_ik: np.ndarray,
  stiffness: float,
  rotational_stiffness: float,
  dt: float,
  ik_target_overrides: Optional[Dict[str, Tuple[np.ndarray, Rotation]]] = None,
) -> Tuple[np.ndarray, Rotation, np.ndarray]:
  """Set the targets of all active tasks for this step and solve the constrained IK
  once (daqp QP). Returns the loaded link's target pose and the solved velocity."""
  # A temporary configuration at the reference pose, to read "each link's world pose
  # in the reference state" as targets.
  ref_config = mink.Configuration(ik_solver.model)
  ref_config.update(q=qpos_ref)
  mujoco.mj_forward(ref_config.model, ref_config.data)

  active_tasks = []
  # Posture regularizer + waist/knee + pelvis orientation + feet (all toward reference)
  ik_solver.posture_task.set_target(qpos_ref)
  active_tasks.append(ik_solver.posture_task)
  if ik_solver.waist_task:
    ik_solver.waist_task.set_target(qpos_ref)
    active_tasks.append(ik_solver.waist_task)
  if ik_solver.knee_task:
    ik_solver.knee_task.set_target(qpos_ref)
    active_tasks.append(ik_solver.knee_task)
  ik_solver.pelvis_pitch_task.set_target(
    ref_config.get_transform_frame_to_world("pelvis", "body")
  )
  active_tasks.append(ik_solver.pelvis_pitch_task)
  for name, task in ik_solver.foot_tasks.items():
    task.set_target(ref_config.get_transform_frame_to_world(name, "body"))
    active_tasks.append(task)

  if ik_solver.torso_orientation_task is not None:
    ik_solver.torso_orientation_task.set_target(
      ref_config.get_transform_frame_to_world("torso_link", "body")
    )
    active_tasks.append(ik_solver.torso_orientation_task)

  # CoM target: COP-aware (see Section 5)
  com_target = calculate_cop_aware_com_target(
    ref_config.data,
    ik_solver.total_mass,
    com_force_for_ik,
    torque_ext,
    current_task_body_name,
  )
  ik_solver.com_task.set_target(com_target)
  active_tasks.append(ik_solver.com_task)

  is_force_active = (
    np.linalg.norm(force_ext) > 1e-2 or np.linalg.norm(torque_ext) > 1e-2
  )
  if ik_target_overrides is None:
    ik_target_overrides = {}
  # Keypoint tasks: every link except "the one currently under force" follows the
  # reference. Overridden links (release interpolation) use the hard force_task.
  for name, task in ik_solver.keypoint_tasks.items():
    if name in ik_target_overrides:
      pos, rot = ik_target_overrides[name]
      quat_xyzw = rot.as_quat()
      quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
      target_pose = mink.SE3(np.concatenate([quat_wxyz, pos]))
      release_task = ik_solver.force_tasks[name]
      release_task.set_target(target_pose)
      active_tasks.append(release_task)
    elif not (is_force_active and name == current_task_body_name):
      task.set_target(ref_config.get_transform_frame_to_world(name, "body"))
      active_tasks.append(task)

  # Hard target of the loaded link: reference pose + spring displacement
  task_ref_pos = ref_config.data.body(current_task_body_name).xpos.copy()
  task_ref_rot = Rotation.from_matrix(
    ref_config.data.body(current_task_body_name).xmat.reshape(3, 3)
  )
  task_target_pos, task_target_rot = task_ref_pos, task_ref_rot
  if is_force_active:
    # Spring law: position target = reference + F/K_robot; orientation target = the
    # rotation by (τ/K_rot) composed with the reference orientation.
    task_target_pos = task_ref_pos + force_ext / max(stiffness, 1e-6)
    if np.linalg.norm(torque_ext) > 1e-4 and rotational_stiffness > 1e-4:
      task_target_rot = (
        Rotation.from_rotvec(torque_ext / rotational_stiffness) * task_ref_rot
      )
    quat_xyzw = task_target_rot.as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    target_pose = mink.SE3(np.concatenate([quat_wxyz, task_target_pos]))
    active_force_task = ik_solver.force_tasks[current_task_body_name]
    active_force_task.set_target(target_pose)
    active_tasks.append(active_force_task)
  elif current_task_body_name in ik_target_overrides:
    task_target_pos, task_target_rot = ik_target_overrides[current_task_body_name]

  # Solve the constrained QP once and integrate the solved joint velocity into the
  # configuration (forward Euler).
  vel = mink.solve_ik(
    ik_solver.configuration,
    active_tasks,
    dt,
    solver="daqp",
    limits=ik_solver.limits,
    damping=1e-5,
  )
  ik_solver.configuration.integrate_inplace(vel, dt)
  mujoco.mj_forward(ik_solver.model, ik_solver.data)
  return task_target_pos, task_target_rot, vel


def perform_single_ik_update(
  ik_solver: G1_Mink_IK_Solver,
  ref_data: mujoco.MjData,
  event: Optional[Dict],
  release_info: Optional[Dict],
  sim_time: float,
  dt: float,
  config: SimulationConfig,
  qpos_ref: np.ndarray,
) -> Tuple:
  """Full IK update for one frame: current external wrench → release interpolation →
  IK solve → collect visualization data."""
  force_ext, torque_ext = np.zeros(3), np.zeros(3)
  com_force_for_ik = np.zeros(3)
  # Link to act on: the current event's link first, then the releasing link, else
  # torso by default.
  task_body_name = (
    event["link_name"]
    if event
    else (release_info["link_name"] if release_info else "torso_link")
  )
  stiffness, rot_stiffness = 140.0, 1.0

  # Release phase: after an event ends, the loaded link slides from the "pushed pose"
  # back to the reference pose at a limited speed.
  ik_target_overrides: Dict[str, Tuple[np.ndarray, Rotation]] = {}
  if release_info and not release_info.get("finished", False):
    ref_data.qpos[:] = qpos_ref
    mujoco.mj_forward(ref_data.model, ref_data)
    target_pos = ref_data.body(release_info["link_name"]).xpos
    target_rot = Rotation.from_matrix(
      ref_data.body(release_info["link_name"]).xmat.reshape(3, 3)
    )
    total_dist = np.linalg.norm(target_pos - release_info["start_pos"])
    total_angle = (release_info["start_rot"].inv() * target_rot).magnitude()
    # Total duration from the max release speeds; the slower of position/rotation wins.
    time_needed_pos = (
      total_dist / MAX_RELEASE_LINEAR_VEL if MAX_RELEASE_LINEAR_VEL > 1e-6 else 0.0
    )
    time_needed_rot = (
      total_angle / MAX_RELEASE_ANGULAR_VEL if MAX_RELEASE_ANGULAR_VEL > 1e-6 else 0.0
    )
    total_time_needed = max(time_needed_pos, time_needed_rot, 1e-5)
    time_since_release = sim_time - release_info["start_time"]

    if time_since_release >= total_time_needed:
      release_info["finished"] = True
      com_force_for_ik = np.zeros(3)
    else:
      # Linear/spherical interpolation from the "pushed pose" back to the reference;
      # the external-force CoM term decays proportionally.
      interp_factor = 1.0 - (time_since_release / total_time_needed)
      slerp = Slerp(
        [0, 1], Rotation.concatenate([target_rot, release_info["start_rot"]])
      )
      interp_rot = slerp(interp_factor)
      interp_pos = target_pos + interp_factor * (release_info["start_pos"] - target_pos)
      ik_target_overrides[release_info["link_name"]] = (interp_pos, interp_rot)
      com_force_for_ik = release_info["start_force"] * interp_factor

  # Active event phase: compute this frame's external wrench for force_mode
  if event and sim_time >= event["start_time"]:
    stiffness = event["stiffness"]
    rot_stiffness = event.get("rotational_stiffness", 1.0)

    # Triangle-wave factor: ramp up (linear 0→1), hold (1), ramp down (linear 1→0).
    main_magnitude_factor = 0.0
    if sim_time < event["hold_start_time"]:
      main_magnitude_factor = (
        (sim_time - event["start_time"]) / event["ramp_duration"]
        if event["ramp_duration"] > 1e-5
        else 1.0
      )
    elif sim_time < event["hold_end_time"]:
      main_magnitude_factor = 1.0
    elif sim_time <= event["end_time"]:
      main_magnitude_factor = (
        1.0 - (sim_time - event["hold_end_time"]) / event["ramp_duration"]
        if event["ramp_duration"] > 1e-5
        else 0.0
      )
    main_magnitude_factor = np.clip(main_magnitude_factor, 0.0, 1.0)

    ref_data.qpos[:] = qpos_ref
    mujoco.mj_forward(ref_data.model, ref_data)
    p_ref = ref_data.body(task_body_name).xpos.copy()
    r_ref = Rotation.from_matrix(ref_data.body(task_body_name).xmat.reshape(3, 3))

    if config.force_mode == "triangle":
      # Pure triangle wave: wrench = peak × magnitude factor × unit axis.
      force_ext = event["amplitude"] * main_magnitude_factor * event["force_axis"]
      torque_ext = (
        event["torque_amplitude"] * main_magnitude_factor * event["torque_axis"]
      )

    elif config.force_mode == "forcefield":
      # Force field: same wrench as the triangle wave, but also records the
      # environment setpoint pushed out by the "series springs" (for visualization /
      # export only).
      force_ext = event["amplitude"] * main_magnitude_factor * event["force_axis"]
      torque_ext = (
        event["torque_amplitude"] * main_magnitude_factor * event["torque_axis"]
      )

      k_robot_lin, k_forcefield_lin = stiffness, event["forcefield_stiffness"]
      k_robot_rot = rot_stiffness
      k_forcefield_rot = event["rotational_forcefield_stiffness"]

      # Series springs: the same force F compresses both the "environment spring
      # K_forcefield" and the "robot spring K_robot", so the environment setpoint sits
      # further from the reference: F/K_forcefield + F/K_robot.
      p_forcefield = (
        p_ref
        + force_ext / max(k_forcefield_lin, 1e-6)
        + force_ext / max(k_robot_lin, 1e-6)
      )
      rot_forcefield = (
        Rotation.from_rotvec(torque_ext / max(k_forcefield_rot, 1e-6))
        * Rotation.from_rotvec(torque_ext / max(k_robot_rot, 1e-6))
        * r_ref
      )
      event["forcefield_setpoint_pos"] = p_forcefield
      event["forcefield_setpoint_rot"] = rot_forcefield

    elif config.force_mode == "collision-emulator":
      # Fixed-setpoint spring: force = equivalent series stiffness × (setpoint − ref).
      k_robot_lin, k_forcefield_lin = stiffness, event["forcefield_stiffness"]
      k_eff_lin = (
        (k_robot_lin * k_forcefield_lin) / (k_robot_lin + k_forcefield_lin)
        if (k_robot_lin + k_forcefield_lin) > 1e-6
        else 0.0
      )
      k_robot_rot, k_forcefield_rot = (
        rot_stiffness,
        event["rotational_forcefield_stiffness"],
      )
      k_eff_rot = (
        (k_robot_rot * k_forcefield_rot) / (k_robot_rot + k_forcefield_rot)
        if (k_robot_rot + k_forcefield_rot) > 1e-6
        else 0.0
      )
      delta_p = event["forcefield_setpoint_pos"] - p_ref
      delta_rot_vec = (event["forcefield_setpoint_rot"] * r_ref.inv()).as_rotvec()
      force_ext = k_eff_lin * delta_p
      torque_ext = k_eff_rot * delta_rot_vec

    elif config.force_mode == "collision-emulator-1d":
      # 1D collision plane: normal contact force from the penetration depth (series
      # equivalent stiffness × penetration); the plane advances slowly.
      if not event.get("terminated", False):
        k_forcefield_eff = event["forcefield_stiffness"]
        time_since_spawn = sim_time - event["initial_spawn_time"]
        current_plane_origin = (
          event["collision_plane_origin"]
          + event["plane_velocity_vec"] * time_since_spawn
        )
        penetration = -np.dot(
          p_ref - current_plane_origin, event["collision_plane_normal"]
        )
        if penetration > 0:
          denominator = stiffness + k_forcefield_eff
          force_ext = (
            (stiffness * k_forcefield_eff / denominator)
            * penetration
            * event["collision_plane_normal"]
            if denominator > 1e-6
            else np.zeros(3)
          )
        else:
          event["terminated"] = True  # penetration over → this collision event ends

      # Superimposed random torque (its own ramp up, hold, ramp down timing)
      if "torque_start_time" in event and sim_time >= event["torque_start_time"]:
        torque_magnitude_factor = 0.0
        if sim_time < event["torque_hold_start_time"]:
          torque_magnitude_factor = (
            (sim_time - event["torque_start_time"]) / event["torque_ramp_duration"]
            if event["torque_ramp_duration"] > 1e-5
            else 1.0
          )
        elif sim_time < event["torque_hold_end_time"]:
          torque_magnitude_factor = 1.0
        elif sim_time <= event["torque_end_time"]:
          torque_magnitude_factor = (
            1.0
            - (sim_time - event["torque_hold_end_time"]) / event["torque_ramp_duration"]
            if event["torque_ramp_duration"] > 1e-5
            else 0.0
          )
        torque_ext = (
          event["torque_amplitude"]
          * np.clip(torque_magnitude_factor, 0.0, 1.0)
          * event["torque_axis"]
        )

    com_force_for_ik = force_ext

  elif not release_info or release_info.get("finished", False):
    # No active event and no release: set the configuration straight back to the
    # reference (pure tracking).
    ik_solver.configuration.update(q=qpos_ref)
    mujoco.mj_forward(ik_solver.model, ik_solver.data)

  # Solve this frame's IK
  task_target_pos, task_target_rot, _ = perform_mink_ik_step(
    ik_solver,
    qpos_ref,
    task_body_name,
    force_ext,
    torque_ext,
    com_force_for_ik,
    stiffness,
    rot_stiffness,
    dt,
    ik_target_overrides,
  )

  # Collect the reference / keypoint poses needed for visualization
  ref_data.qpos[:] = qpos_ref
  mujoco.mj_forward(ref_data.model, ref_data)
  task_ref_pos_for_viz = ref_data.body(task_body_name).xpos.copy()
  keypoint_poses = {
    name: (
      ref_data.body(name).xpos.copy(),
      Rotation.from_matrix(ref_data.body(name).xmat.reshape(3, 3)),
    )
    for name in KEYPOINT_BODY_NAMES
  }
  return (
    task_body_name,
    force_ext,
    torque_ext,
    task_ref_pos_for_viz,
    task_target_pos,
    task_target_rot,
    keypoint_poses,
  )


# ============================================================================
# Section 8: visual overlays (the original visualization.py): force arrows, target
# spheres, collision plane, CoM
# ============================================================================


def add_marker(
  scene: mujoco.MjvScene,
  pos: np.ndarray,
  size: List[float],
  rgba: List[float],
  type: int = mujoco.mjtGeom.mjGEOM_SPHERE,
  mat: Optional[np.ndarray] = None,
):
  """Add a geom marker (sphere by default) to the scene, capped by scene.maxgeom."""
  if mat is None:
    # Identity orientation by default (not a default argument, which would be a
    # shared mutable object)
    mat = np.eye(3).flatten()
  if scene.ngeom < scene.maxgeom:
    mujoco.mjv_initGeom(
      scene.geoms[scene.ngeom],
      type=type,
      size=np.asarray(size, dtype=np.float32),
      pos=pos,
      mat=mat,
      rgba=np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def add_arrow(
  scene: mujoco.MjvScene,
  from_: np.ndarray,
  to: np.ndarray,
  radius: float = 0.015,
  rgba: Tuple[float, ...] = (1.0, 0.7, 0.1, 1.0),
):
  """Draw an arrow from from_ to to (used for external forces / coordinate axes)."""
  if scene.ngeom < scene.maxgeom:
    mujoco.mjv_initGeom(
      scene.geoms[scene.ngeom],
      type=mujoco.mjtGeom.mjGEOM_ARROW,
      size=np.zeros(3),
      pos=np.zeros(3),
      mat=np.zeros(9),
      rgba=np.asarray(rgba).astype(np.float32),
    )
    mujoco.mjv_connector(
      scene.geoms[scene.ngeom],
      type=mujoco.mjtGeom.mjGEOM_ARROW,
      width=radius,
      from_=from_,
      to=to,
    )
    scene.ngeom += 1


def add_visual_overlays(
  scene: mujoco.MjvScene,
  current_data: mujoco.MjData,
  task_body_name: str,
  force_ext: np.ndarray,
  torque_ext: np.ndarray,
  task_ref_pos: np.ndarray,
  task_target_pos: np.ndarray,
  task_target_rot: Rotation,
  keypoint_target_6d_poses: Dict[str, Tuple[np.ndarray, Rotation]],
  event: Optional[Dict],
  sim_time: float,
  rewind_indicator_until: float,
):
  """Draw all visual guides for this frame. Color legend for the animation:
  orange arrow = external force; green sphere = the loaded link's current target;
  pink sphere = force-field environment setpoint; cyan box = 1D collision plane;
  RGB axes = target orientation; red sphere = the loaded link's reference position;
  big magenta sphere = whole-body CoM; blue spheres = keypoint references;
  big pulsing red sphere = a "rollback" just happened (the event was infeasible and
  undone)."""
  if np.linalg.norm(force_ext) > 1e-2:
    arrow_start = current_data.body(task_body_name).xpos
    arrow_end = arrow_start + force_ext * 0.05  # 0.05: force → length display scale
    add_arrow(scene, arrow_start, arrow_end, rgba=[1, 0.7, 0.1, 1])
    add_marker(scene, task_target_pos, [0.03, 0, 0], [0, 1, 0, 0.5])

  if event:
    if "forcefield_setpoint_pos" in event:
      add_marker(
        scene, event["forcefield_setpoint_pos"], [0.04, 0, 0], [1, 0.4, 0.8, 0.7]
      )
    if "collision_plane_normal" in event and not event.get("terminated", False):
      time_since_spawn = sim_time - event["initial_spawn_time"]
      plane_pos = event["collision_plane_origin"] + event.get(
        "plane_velocity_vec", np.zeros(3)
      ) * max(0, time_since_spawn)
      plane_normal = event["collision_plane_normal"]
      z_axis = np.array([0, 0, 1])
      rot_axis = np.cross(z_axis, plane_normal)
      angle = np.arccos(np.clip(np.dot(z_axis, plane_normal), -1.0, 1.0))
      if np.linalg.norm(rot_axis) > 1e-6:
        orientation = Rotation.from_rotvec(angle * rot_axis / np.linalg.norm(rot_axis))
      else:
        orientation = Rotation.identity()
      add_marker(
        scene,
        plane_pos,
        [0.2, 0.2, 0.005],
        [0.1, 0.8, 0.9, 0.3],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        mat=orientation.as_matrix().flatten(),
      )

  if np.linalg.norm(torque_ext) > 1e-2:
    # RGB axes of the target orientation
    axis_len, axis_radius = 0.15, 0.007
    target_axes = task_target_rot.as_matrix()
    add_arrow(
      scene,
      task_target_pos,
      task_target_pos + axis_len * target_axes[:, 0],
      radius=axis_radius,
      rgba=[1, 0, 0, 0.7],
    )
    add_arrow(
      scene,
      task_target_pos,
      task_target_pos + axis_len * target_axes[:, 1],
      radius=axis_radius,
      rgba=[0, 1, 0, 0.7],
    )
    add_arrow(
      scene,
      task_target_pos,
      task_target_pos + axis_len * target_axes[:, 2],
      radius=axis_radius,
      rgba=[0, 0, 1, 0.7],
    )

  add_marker(scene, task_ref_pos, [0.03, 0, 0], [1, 0, 0, 0.5])  # red: loaded-link ref
  add_marker(
    scene, current_data.subtree_com[0], [0.035, 0, 0], [0.9, 0.2, 0.8, 0.8]
  )  # magenta: CoM
  for target_pos, _target_rot in keypoint_target_6d_poses.values():
    add_marker(scene, target_pos, [0.02, 0, 0], [0.2, 0.5, 1, 0.5])  # blue: keypoints

  if sim_time < rewind_indicator_until:
    alpha = 0.6 * (rewind_indicator_until - sim_time)
    add_marker(
      scene, current_data.subtree_com[0], [0.4, 0, 0], [1, 0.1, 0.1, alpha]
    )  # red pulse: rollback indicator


# ============================================================================
# Section 9: simulation/generation main loop (the original runner.py)
# ============================================================================


class SimulationRunner:
  """State and control flow of one simulation/generation run. interactive mode opens a
  window to watch the animation; generate-data mode produces CSVs in batch."""

  SIMULATION_FREQUENCY = 30.0  # simulation step rate (Hz)
  LOGGING_FREQUENCY = 30.0  # logging step rate (Hz)

  def __init__(self, config: SimulationConfig, file_index: int):
    self.config = config
    self.file_index = file_index
    self.is_interactive = config.mode == "interactive"
    self.timestep = 1.0 / self.SIMULATION_FREQUENCY
    self.logging_interval_steps = max(
      1, round(self.SIMULATION_FREQUENCY / self.LOGGING_FREQUENCY)
    )
    self.ik_solver: Optional[G1_Mink_IK_Solver] = None
    self.motion_duration = 0.0
    self.force_profile: List[Dict[str, Any]] = []
    self.event_queue: List[Dict[str, Any]] = []
    self.current_event: Optional[Dict[str, Any]] = None
    self.release_info: Optional[Dict[str, Any]] = None
    self.reusable_ref_data: Optional[mujoco.MjData] = None
    self.prev_qpos_ref: Optional[np.ndarray] = None
    self.event_start_data_idx = -1
    self.qpos_before_event: Optional[np.ndarray] = None
    self.rewind_indicator_until = 0.0
    self.viewer = None
    self.camera_tracking = True
    self.renderer = None
    self.cam = None
    # Render-only model/data: the scene with ground + lights + skybox (rendering only,
    # never used for IK); same qpos layout as the IK model.
    self.render_model: Optional[mujoco.MjModel] = None
    self.render_data: Optional[mujoco.MjData] = None
    self.frames: List[np.ndarray] = []
    # Four output buffers: adapted pose / reference pose / force / collision metadata
    self.all_adapted_qpos: List[np.ndarray] = []
    self.all_reference_qpos: List[np.ndarray] = []
    self.all_force_data: List[np.ndarray] = []
    self.all_collision_metadata: List[np.ndarray] = []
    # Stiffness state randomized in zero-wrench mode
    self.stiffness_state = {
      "stiffness": 140.0,
      "rot_stiffness": 1.0,
      "next_update_time": -1.0,
    }
    self.num_steps = 0

  def run(self):
    self._announce_start()
    self._seed_random_generators()
    self._initialize_solver()
    self._prepare_force_profile()
    self._prepare_timing()
    self._initialize_zero_wrench_state()
    self._setup_rendering()
    try:
      self._simulate_loop()
    finally:
      self._teardown_rendering()
    if not self.is_interactive:
      self._save_outputs()

  def _announce_start(self):
    if self.is_interactive:
      print(
        "\n--- Starting interactive simulation "
        f"(force mode: {self.config.force_mode}) ---"
      )
    else:
      os.makedirs(self.config.output_dir, exist_ok=True)
      print(
        f"\n[file {self.file_index + 1}/{self.config.num_files}] generating profile "
        f"(seed={self.config.seed}, mode={self.config.force_mode})..."
      )

  def _seed_random_generators(self):
    random.seed(self.config.seed)
    np.random.seed(self.config.seed)

  def _initialize_solver(self):
    self.ik_solver = G1_Mink_IK_Solver(
      self.config.model_path,
      self.config.motion_path,
      self.config.repeat_frame_time,
      self.config.com_cost,
      self.config.com_cost_z_factor,
      self.config.torso_orientation_cost,
    )
    if self.ik_solver.motion_lib:
      self.motion_duration = self.ik_solver.motion_lib.get_max_time()
    else:
      self.motion_duration = 20.0
    self.reusable_ref_data = mujoco.MjData(self.ik_solver.model)

  def _prepare_force_profile(self):
    self.force_profile = generate_random_force_profile(
      self.motion_duration,
      FORCEABLE_LINKS,
      self.config.force_mode,
      self.ik_solver,
      self.config,
    )
    print(
      f"[file {self.file_index + 1}/{self.config.num_files}] "
      f"generated {len(self.force_profile)} candidate force events."
    )
    self.event_queue = list(self.force_profile)

  def _prepare_timing(self):
    self.num_steps = int(np.ceil(self.motion_duration / self.timestep))
    # Offline rendering truncates the simulation to max_seconds (a long reference
    # need not be rendered in full).
    if self.config.max_seconds is not None:
      self.num_steps = min(
        self.num_steps, int(np.ceil(self.config.max_seconds / self.timestep))
      )

  def _initialize_zero_wrench_state(self):
    if self.config.force_mode == "zero-wrench":
      self._update_stiffness_state(0.0)

  def _update_stiffness_state(self, current_time: float):
    """zero-wrench: resample the robot's own stiffness every 2 to 5 s (log-uniform)."""
    self.stiffness_state["stiffness"] = np.exp(
      random.uniform(np.log(MIN_ROBOT_STIFFNESS), np.log(MAX_ROBOT_STIFFNESS))
    )
    self.stiffness_state["rot_stiffness"] = np.exp(
      random.uniform(
        np.log(MIN_ROBOT_ROTATIONAL_STIFFNESS), np.log(MAX_ROBOT_ROTATIONAL_STIFFNESS)
      )
    )
    hold_duration = random.uniform(2.0, 5.0)
    self.stiffness_state["next_update_time"] = current_time + hold_duration

  def _setup_rendering(self):
    # Interactive mode and video recording both need the lit scene model; compose it
    # once via the mjlab API and sync it to the current IK pose.
    if self.is_interactive or self.config.record_video:
      self.render_model = build_g1_scene_model(
        self.config.render_width, self.config.render_height
      )
      self.render_data = mujoco.MjData(self.render_model)
      self._sync_render_data()

    if self.is_interactive:
      self.viewer = mujoco.viewer.launch_passive(self.render_model, self.render_data)
      self.viewer.cam.azimuth, self.viewer.cam.elevation, self.viewer.cam.distance = (
        90,
        -15,
        4.0,
      )
      self.viewer.cam.lookat[:] = self.render_data.body("torso_link").xpos

      def key_callback(_keycode):
        # Any key: toggle whether the camera follows the torso.
        self.camera_tracking = not self.camera_tracking

      self.viewer.key_callback = key_callback

    if self.config.record_video:
      if imageio is None:
        print("Warning: imageio is not installed, cannot record video.")
        self.config.record_video = False
      else:
        self.renderer = mujoco.Renderer(
          self.render_model,
          height=self.config.render_height,
          width=self.config.render_width,
        )
        self.cam = mujoco.MjvCamera()
        self.cam.azimuth = self.config.cam_azimuth
        self.cam.elevation = self.config.cam_elevation
        self.cam.distance = self.config.cam_distance
        print(f"Recording video to {self.config.output_filename}")

  def _sync_render_data(self):
    """Copy the IK-solved qpos into the render model and run forward kinematics. Both
    models share the exact qpos layout, so a plain copy works."""
    if self.render_data is None:
      return
    self.render_data.qpos[:] = self.ik_solver.data.qpos
    mujoco.mj_forward(self.render_model, self.render_data)

  def _simulate_loop(self):
    step = 0
    while step < self.num_steps:
      if self.is_interactive and not self.viewer.is_running():
        break

      loop_start = time.time()
      current_time = step * self.timestep

      # zero-wrench: resample the stiffness when it is due
      if (
        self.config.force_mode == "zero-wrench"
        and current_time >= self.stiffness_state["next_update_time"]
      ):
        self._update_stiffness_state(current_time)

      qpos_ref, _, contacts_ref = self.ik_solver.get_reference_motion(current_time)
      self._handle_possible_teleport(qpos_ref, current_time)

      self._complete_event_if_finished(current_time, qpos_ref)
      self._maybe_start_next_event(current_time)

      vis_data = perform_single_ik_update(
        self.ik_solver,
        self.reusable_ref_data,
        self.current_event,
        self.release_info,
        current_time,
        self.timestep,
        self.config,
        qpos_ref,
      )

      # With an active event, run the feasibility check; roll back if infeasible
      # (where rejection sampling happens).
      if self.current_event:
        self.reusable_ref_data.qpos[:] = qpos_ref
        mujoco.mj_forward(self.reusable_ref_data.model, self.reusable_ref_data)
        is_feasible, _ = is_ik_solution_feasible(
          self.ik_solver.data,
          self.reusable_ref_data,
          self.current_event["link_name"],
          vis_data[1],  # force_ext
          vis_data[2],  # torque_ext
          self.current_event["stiffness"],
          self.current_event.get("rotational_stiffness", 1.0),
          self.ik_solver.total_mass,
        )
        if not is_feasible:
          step = self._handle_infeasible_event(current_time)
          continue

      self._finalize_release_if_needed()

      if not self.is_interactive:
        self._record_step(step, qpos_ref, contacts_ref, vis_data, current_time)

      self._update_viewer(vis_data, current_time, loop_start)
      self._capture_video_frame(vis_data, current_time)

      step += 1

  def _handle_possible_teleport(self, qpos_ref: np.ndarray, current_time: float):
    """On a reference XY jump (e.g. a loop seam), translate the IK state / event
    setpoints along with it so the pose does not blow up."""
    if self.prev_qpos_ref is None:
      self.prev_qpos_ref = qpos_ref.copy()
      return

    dist_sq = np.sum((qpos_ref[:2] - self.prev_qpos_ref[:2]) ** 2)
    if dist_sq > TELEPORT_THRESHOLD**2:
      teleport_vector = qpos_ref[:3] - self.prev_qpos_ref[:3]
      print(
        f"Teleport detected at t={current_time:.2f}s (dist={np.sqrt(dist_sq):.2f}m); "
        "translating the IK state along."
      )
      current_q = self.ik_solver.configuration.q
      current_q[0:3] += teleport_vector
      self.ik_solver.configuration.update(q=current_q)

      if self.qpos_before_event is not None:
        self.qpos_before_event[0:3] += teleport_vector

      if self.current_event:
        if "forcefield_setpoint_pos" in self.current_event:
          self.current_event["forcefield_setpoint_pos"] += teleport_vector
        if "collision_plane_origin" in self.current_event:
          self.current_event["collision_plane_origin"] += teleport_vector

      if self.release_info:
        self.release_info["start_pos"] += teleport_vector

    self.prev_qpos_ref = qpos_ref.copy()

  def _complete_event_if_finished(self, current_time: float, qpos_ref: np.ndarray):
    """When an event is due to end, end it and register "release" info (the pushed
    pose/force at this moment, for sliding back smoothly)."""
    if not self.current_event or current_time < self.current_event["end_time"]:
      return

    (_task_body_name, last_force_ext, _, _, _, _, _) = perform_single_ik_update(
      self.ik_solver,
      self.reusable_ref_data,
      self.current_event,
      None,
      current_time,
      self.timestep,
      self.config,
      qpos_ref,
    )
    link_name = self.current_event["link_name"]
    self.release_info = {
      "link_name": link_name,
      "start_time": current_time,
      "start_pos": self.ik_solver.data.body(link_name).xpos.copy(),
      "start_rot": Rotation.from_matrix(
        self.ik_solver.data.body(link_name).xmat.reshape(3, 3)
      ),
      "start_force": last_force_ext.copy(),
    }
    self.current_event = None
    self.event_start_data_idx = -1
    self.qpos_before_event = None

  def _maybe_start_next_event(self, current_time: float):
    """When due, pop the next event from the queue and record the "pre-event state"
    (needed for rollback)."""
    if self.current_event or not self.event_queue:
      return
    if current_time < self.event_queue[0]["start_time"]:
      return

    self.current_event = self.event_queue.pop(0)
    if self.is_interactive:
      self.qpos_before_event = self.ik_solver.configuration.q.copy()
    else:
      self.event_start_data_idx = len(self.all_adapted_qpos)

  def _handle_infeasible_event(self, current_time: float) -> int:
    """Rollback of an infeasible event: scale its force/torque/ramp duration by 0.8
    and retry it; give up if it gets too small.
    Returns the "step to rewind to"; after the main loop's continue it reruns from
    that step."""
    event_to_rewind = self.current_event
    if self.config.force_mode in ["triangle", "forcefield", "collision-emulator"]:
      scale_factor = 0.8
      if self.config.force_mode in ["forcefield", "triangle"]:
        new_amplitude = event_to_rewind.get("amplitude", 0.0) * scale_factor
        new_torque_amplitude = (
          event_to_rewind.get("torque_amplitude", 0.0) * scale_factor
        )
        old_ramp_duration = event_to_rewind["ramp_duration"]
        new_ramp_duration = old_ramp_duration * scale_factor
        min_ramp_duration = 0.1
        if (
          new_amplitude < 1.0 and new_torque_amplitude < 1.0
        ) or new_ramp_duration < min_ramp_duration:
          self.current_event = None  # still infeasible at a small force → give up
        else:
          start_time = event_to_rewind["start_time"]
          hold_duration = (
            event_to_rewind["hold_end_time"] - event_to_rewind["hold_start_time"]
          )
          event_to_rewind["amplitude"] = new_amplitude
          event_to_rewind["torque_amplitude"] = new_torque_amplitude
          event_to_rewind["ramp_duration"] = new_ramp_duration
          event_to_rewind["hold_start_time"] = start_time + new_ramp_duration
          event_to_rewind["hold_end_time"] = (
            event_to_rewind["hold_start_time"] + hold_duration
          )
          event_to_rewind["end_time"] = (
            event_to_rewind["hold_end_time"] + new_ramp_duration
          )
      elif self.config.force_mode == "collision-emulator":
        new_end_time = current_time
        start_time = event_to_rewind["start_time"]
        min_valid_duration = 0.2
        if (new_end_time - start_time) < min_valid_duration:
          self.current_event = None
        else:
          event_to_rewind["end_time"] = new_end_time
    else:
      self.current_event = None

    rewind_to_time = event_to_rewind["start_time"]
    self.release_info = None

    if self.is_interactive:
      self.rewind_indicator_until = current_time + 1.0  # show a red pulse for 1s
      if self.qpos_before_event is not None:
        self.ik_solver.configuration.update(q=self.qpos_before_event)
      self.qpos_before_event = None
    else:
      # Generate mode: truncate and drop all data recorded during this event
      self.all_adapted_qpos = self.all_adapted_qpos[: self.event_start_data_idx]
      self.all_reference_qpos = self.all_reference_qpos[: self.event_start_data_idx]
      self.all_force_data = self.all_force_data[: self.event_start_data_idx]
      self.all_collision_metadata = self.all_collision_metadata[
        : self.event_start_data_idx
      ]
      if self.all_adapted_qpos:
        last_valid_qpos = self.all_adapted_qpos[-1]
      else:
        last_valid_qpos = self.ik_solver.get_reference_motion(rewind_to_time)[0]
      self.ik_solver.configuration.update(q=last_valid_qpos)

    return int(rewind_to_time / self.timestep)

  def _finalize_release_if_needed(self):
    if self.release_info and self.release_info.get("finished", False):
      self.release_info = None

  def _record_step(
    self,
    step: int,
    qpos_ref: np.ndarray,
    contacts_ref: np.ndarray,
    vis_data: Tuple,
    current_time: float,
  ):
    """generate-data: append this frame's four data streams to the buffers. The column
    layout matches the original exactly."""
    if step % self.logging_interval_steps != 0:
      return

    # Reference pose (note: the qpos quaternion is reordered from wxyz to xyzw before
    # saving: [4,5,6,3]) + foot contacts
    self.all_reference_qpos.append(
      np.concatenate(
        [
          qpos_ref[0:3],
          qpos_ref[[4, 5, 6, 3]],
          qpos_ref[self.ik_solver.actuated_qpos_indices],
          contacts_ref,
        ]
      )
    )
    # Adapted (IK-solved) pose, also reordered to xyzw
    q = self.ik_solver.configuration.q.copy()
    self.all_adapted_qpos.append(
      np.concatenate([q[0:3], q[[4, 5, 6, 3]], q[self.ik_solver.actuated_qpos_indices]])
    )

    # Force info: [link_id, force(3), torque(3), stiffness, rot_stiffness]
    target_event = (
      self.current_event
      if self.current_event
      else (self.event_queue[0] if self.event_queue else None)
    )
    if target_event:
      link_id = mujoco.mj_name2id(
        self.ik_solver.model, mujoco.mjtObj.mjOBJ_BODY, target_event["link_name"]
      )
      stiffness = target_event["stiffness"]
      rot_stiffness = target_event.get("rotational_stiffness", 1.0)
      force_info = np.array(
        [link_id, *vis_data[1], *vis_data[2], stiffness, rot_stiffness]
      )
    else:
      if self.config.force_mode == "zero-wrench":
        stiffness = self.stiffness_state["stiffness"]
        rot_stiffness = self.stiffness_state["rot_stiffness"]
      else:
        stiffness = 140.0
        rot_stiffness = 1.0
      force_info = np.array([-1, 0, 0, 0, 0, 0, 0, stiffness, rot_stiffness])
    self.all_force_data.append(force_info)

    # Collision/force-field metadata: [forcefield_stiffness, rot_forcefield_stiffness,
    # setpoint_pos(3), setpoint_rot_quat(4), plane_normal(3)]
    forcefield_stiffness, rot_forcefield_stiffness = 0.0, 0.0
    forcefield_setpoint_pos = np.zeros(3)
    forcefield_setpoint_rot_quat = np.array([0.0, 0.0, 0.0, 1.0])
    plane_normal = np.zeros(3)

    if self.current_event and self.config.force_mode in [
      "forcefield",
      "collision-emulator",
      "collision-emulator-1d",
    ]:
      forcefield_stiffness = self.current_event.get("forcefield_stiffness", 0.0)
      rot_forcefield_stiffness = self.current_event.get(
        "rotational_forcefield_stiffness", 0.0
      )
      if (
        self.config.force_mode == "collision-emulator-1d"
        and "collision_plane_normal" in self.current_event
      ):
        forcefield_setpoint_pos = self.current_event[
          "collision_plane_origin"
        ] + self.current_event.get("plane_velocity_vec", np.zeros(3)) * max(
          0, current_time - self.current_event["initial_spawn_time"]
        )
        plane_normal = self.current_event["collision_plane_normal"]
      elif "forcefield_setpoint_pos" in self.current_event:
        forcefield_setpoint_pos = self.current_event["forcefield_setpoint_pos"]
      if "forcefield_setpoint_rot" in self.current_event:
        forcefield_setpoint_rot_quat = self.current_event[
          "forcefield_setpoint_rot"
        ].as_quat()

    self.all_collision_metadata.append(
      np.concatenate(
        [
          [forcefield_stiffness, rot_forcefield_stiffness],
          forcefield_setpoint_pos,
          forcefield_setpoint_rot_quat,
          plane_normal,
        ]
      )
    )

  def _update_viewer(self, vis_data: Tuple, current_time: float, loop_start: float):
    if not (self.is_interactive and self.viewer.is_running()):
      return

    self._sync_render_data()  # sync the IK pose into the lit render scene
    if self.camera_tracking:
      self.viewer.cam.lookat[:] = self.render_data.body("torso_link").xpos
    self.viewer.user_scn.ngeom = 0
    # Overlay geometry uses ik_solver.data for world coordinates (same qpos as
    # render_data, so coordinates agree; the robot-only CoM is more accurate)
    add_visual_overlays(
      self.viewer.user_scn,
      self.ik_solver.data,
      *vis_data,
      self.current_event,
      current_time,
      self.rewind_indicator_until,
    )
    self.viewer.sync()
    time.sleep(max(0, self.timestep - (time.time() - loop_start)))  # real-time pacing

  def _capture_video_frame(self, vis_data: Tuple, current_time: float):
    if not (self.config.record_video and self.renderer):
      return

    self._sync_render_data()  # sync the IK pose into the lit render scene
    active_cam = (
      self.viewer.cam
      if (self.is_interactive and self.viewer.is_running())
      else self.cam
    )
    if not (self.is_interactive and self.viewer.is_running()):
      active_cam.lookat[:] = self.render_data.body("torso_link").xpos
    self.renderer.update_scene(self.render_data, camera=active_cam)
    add_visual_overlays(
      self.renderer.scene,
      self.ik_solver.data,
      *vis_data,
      self.current_event,
      current_time,
      self.rewind_indicator_until,
    )
    self.frames.append(self.renderer.render())

  def _teardown_rendering(self):
    if self.viewer:
      self.viewer.close()
    if self.config.record_video and self.frames:
      print(f"\nSaving {len(self.frames)} frames to '{self.config.output_filename}'...")
      try:
        imageio.mimsave(
          self.config.output_filename,
          self.frames,
          fps=int(1.0 / self.timestep),
          quality=7,
        )
        print("Video saved.")
      except Exception as exc:
        print(f"Error saving video: {exc}")

  def _save_outputs(self):
    """Concatenate the four buffers side by side into a one-row-per-frame CSV (same
    column layout as the original pandas version, written with numpy.savetxt here)."""
    num_frames = len(self.all_adapted_qpos)
    if not (
      len(self.all_reference_qpos) == num_frames
      and len(self.all_force_data) == num_frames
      and len(self.all_collision_metadata) == num_frames
    ):
      print(
        "Error: the four data streams differ in length, skipping save. Lengths: "
        f"{len(self.all_adapted_qpos)}, {len(self.all_reference_qpos)}, "
        f"{len(self.all_force_data)}, {len(self.all_collision_metadata)}"
      )
      return

    print(
      f"\n[file {self.file_index + 1}/{self.config.num_files}] "
      f"saving {num_frames} CSV rows..."
    )
    # Reference pose truncated to nq+2 columns (as in the original); the other three
    # streams keep all columns.
    ref = np.array(self.all_reference_qpos)[:, : self.ik_solver.model.nq + 2]
    adapted = np.array(self.all_adapted_qpos)
    force = np.array(self.all_force_data)
    collision = np.array(self.all_collision_metadata)
    out = np.concatenate([ref, adapted, force, collision], axis=1)

    basename = os.path.splitext(os.path.basename(self.config.motion_path))[0]
    out_path = os.path.join(
      self.config.output_dir, f"{basename}_augmented_mink_{self.file_index + 1:03d}.csv"
    )
    # The original uses pandas.to_csv(header=False); numpy.savetxt output is
    # value-identical and drops the pandas dependency.
    np.savetxt(out_path, out, delimiter=",")
    print(f"Saved to '{out_path}'")

    # Native contact NPZ (per-frame named channels + frame-indexed event table),
    # written beside the CSV. force, 9 columns = [link_id, force(3), torque(3),
    # k_rob, k_rot_rob]; collision, 12 columns = [k_ff, k_rot_ff, setpoint(3),
    # setpoint_quat_xyzw(4), plane_normal(3)]. setpoint_quat is converted here from
    # scipy xyzw to mjlab wxyz. The event table is derived from the per-frame active
    # spans, so it agrees with the per-frame channels by construction.
    from mjlab.tasks.codancing.motion.contact_reference import (
      derive_contact_events,
      write_contact_npz,
    )

    assert self.ik_solver is not None  # save runs after run, so the solver exists
    m = self.ik_solver.model
    aug_id2name = {
      i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)
    }
    link_id_arr = force[:, 0].astype(np.int64)
    link_name_arr = np.array(
      [aug_id2name.get(int(i), "") if i >= 0 else "" for i in link_id_arr]
    )
    force_vec, torque_vec = force[:, 1:4], force[:, 4:7]
    robot_stiffness, robot_rot_stiffness = force[:, 7], force[:, 8]
    forcefield_stiffness, forcefield_rot_stiffness = collision[:, 0], collision[:, 1]
    events = derive_contact_events(
      link_id_arr,
      force_vec,
      torque_vec,
      robot_stiffness,
      robot_rot_stiffness,
      forcefield_stiffness,
      forcefield_rot_stiffness,
      aug_id2name,
    )
    contact_npz_path = os.path.join(
      self.config.output_dir,
      f"{basename}_augmented_mink_{self.file_index + 1:03d}_contact.npz",
    )
    write_contact_npz(
      contact_npz_path,
      link_id=link_id_arr,
      link_name=link_name_arr,
      force=force_vec,
      torque=torque_vec,
      robot_stiffness=robot_stiffness,
      robot_rot_stiffness=robot_rot_stiffness,
      forcefield_stiffness=forcefield_stiffness,
      forcefield_rot_stiffness=forcefield_rot_stiffness,
      setpoint_pos=collision[:, 2:5],
      setpoint_quat_wxyz=collision[:, 5:9][:, [3, 0, 1, 2]],
      plane_normal=collision[:, 9:12],
      events=events,
    )
    print(f"Exported native contact NPZ to '{contact_npz_path}'")

    # AMP bridge: also save the adapted pose trajectory as a plain qpos CSV
    # ([pos3, quat_xyzw4, 29 joints]), exactly the input format of
    # qpos_csv_to_motion_npz.py / csv_to_npz.py. Only file 0 is exported.
    if self.config.export_qpos_csv and self.file_index == 0:
      np.savetxt(self.config.export_qpos_csv, adapted, delimiter=",")
      print(f"Exported adapted-pose qpos CSV to '{self.config.export_qpos_csv}'")


def _apply_config_overrides(config: SimulationConfig) -> None:
  """Write the CLI knobs back into module-level constants / class attributes.
  The stiffness constants are read in 15+ places, so they are overridden here once
  instead of in every reader.
  A fork-based Pool (Linux) inherits the parent's globals, but each worker also calls
  this again at the start of run, so spawn is safe too (config arrives pickled)."""
  global MIN_ROBOT_STIFFNESS, MAX_ROBOT_STIFFNESS
  global MIN_FORCEFIELD_STIFFNESS, MAX_FORCEFIELD_STIFFNESS
  global IK_CHECK_MAX_FORCE_MAGNITUDE, FORCEABLE_LINKS
  MIN_ROBOT_STIFFNESS = config.robot_stiffness_min
  MAX_ROBOT_STIFFNESS = config.robot_stiffness_max
  MIN_FORCEFIELD_STIFFNESS = config.forcefield_stiffness_min
  MAX_FORCEFIELD_STIFFNESS = config.forcefield_stiffness_max
  IK_CHECK_MAX_FORCE_MAGNITUDE = config.max_force
  if config.forceable_links:
    FORCEABLE_LINKS = [
      s.strip() for s in config.forceable_links.split(",") if s.strip()
    ]
  # fps drives both the motion sampling rate and the sim/record step rate, so a 50Hz
  # source is augmented at its true speed.
  CsvMotionLib.DATA_FPS = config.fps
  SimulationRunner.SIMULATION_FREQUENCY = config.fps
  SimulationRunner.LOGGING_FREQUENCY = config.fps


def run_simulation_or_generation(config: SimulationConfig, file_index: int = 0):
  _apply_config_overrides(config)
  SimulationRunner(config, file_index).run()


# ============================================================================
# Section 9.5: replay, which plays back a generated augmentation CSV without solving IK
# ============================================================================
#
# Why it exists: generate-data already saves each frame's "IK-solved adapted pose
# (adapted qpos)" together with that frame's external force and force-field setpoint
# into the CSV. Viewing the result needs no second IK solve: just load the adapted
# qpos into the model, mj_forward, draw the force arrow / setpoint sphere, and call
# viewer.sync(). So (1) the viewer plays back smoothly (no daqp solve, no stutter);
# (2) the same data can be watched as often as you like and turned into a video
# separately, fully decoupling solving from viewing. The original SoftMimic has no
# such mode; it was added here.


def _row36_to_qpos(row36: np.ndarray, nq: int, act_idx: List[int]) -> np.ndarray:
  """Assemble the 36 CSV values [pos(3), quat_xyzw(4), 29 joints] into a model qpos
  (quaternion xyzw→wxyz)."""
  qpos = np.zeros(nq)
  qpos[0:3] = row36[0:3]
  qx, qy, qz, qw = row36[3:7]
  qpos[3:7] = [qw, qx, qy, qz]
  for i in range(29):
    qpos[act_idx[i]] = row36[7 + i]
  return qpos


def _reconstruct_overlay_inputs(arr_row, ref_data, robot_only, f0, c0):
  """Rebuild every input add_visual_overlays needs from one CSV row + the reference
  pose ref_data. The replayed overlays (blue keypoint references, red loaded-link
  reference, green target, torque axes, force arrow, setpoint, CoM) then match the
  generate-data video exactly, because the CSV stores both the adapted and the
  reference pose plus force/torque/stiffness/setpoint, enough to reproduce them
  as-is."""
  link_id = int(round(arr_row[f0]))
  force_ext = arr_row[f0 + 1 : f0 + 4].copy()
  torque_ext = arr_row[f0 + 4 : f0 + 7].copy()
  stiffness = arr_row[f0 + 7]
  rot_stiffness = arr_row[f0 + 8]
  setpoint = arr_row[c0 + 2 : c0 + 5].copy()
  plane_normal = arr_row[c0 + 9 : c0 + 12].copy()  # set only by collision-emulator-1d
  # link_id<0 (no force event) falls back to torso_link, the original's default.
  link_name = (
    mujoco.mj_id2name(robot_only, mujoco.mjtObj.mjOBJ_BODY, link_id)
    if link_id >= 0
    else "torso_link"
  )
  task_ref_pos = ref_data.body(link_name).xpos.copy()
  r_ref = Rotation.from_matrix(ref_data.body(link_name).xmat.reshape(3, 3))
  # Same target pose formula as perform_mink_ik_step: reference + spring displacement.
  task_target_pos = task_ref_pos + force_ext / max(stiffness, 1e-6)
  if np.linalg.norm(torque_ext) > 1e-4 and rot_stiffness > 1e-4:
    task_target_rot = Rotation.from_rotvec(torque_ext / rot_stiffness) * r_ref
  else:
    task_target_rot = r_ref
  # Blue keypoints = the poses of the 12 keypoint links in the reference pose.
  keypoint_poses = {
    name: (
      ref_data.body(name).xpos.copy(),
      Rotation.from_matrix(ref_data.body(name).xmat.reshape(3, 3)),
    )
    for name in KEYPOINT_BODY_NAMES
  }
  # Rebuild the event overlay:
  #   collision-emulator-1d (plane_normal stored) → cyan collision plane; the CSV
  #     setpoint is "this frame's plane origin", so zero the plane velocity + spawn
  #     time to make add_visual_overlays draw the plane at that origin.
  #   Otherwise, if a setpoint is stored (forcefield/collision-emulator) → pink
  #     setpoint sphere.
  if np.linalg.norm(plane_normal) > 1e-6:
    event = {
      "collision_plane_normal": plane_normal,
      "collision_plane_origin": setpoint,
      "plane_velocity_vec": np.zeros(3),
      "initial_spawn_time": 0.0,
    }
  elif np.linalg.norm(setpoint) > 1e-9:
    event = {"forcefield_setpoint_pos": setpoint}
  else:
    event = None
  return (
    link_name,
    force_ext,
    torque_ext,
    task_ref_pos,
    task_target_pos,
    task_target_rot,
    keypoint_poses,
    event,
  )


def run_replay(config: SimulationConfig):
  """Replay a generated augmentation CSV: by default in the viewer, smoothly; with
  --record_video it is re-rendered offscreen into a video."""
  _apply_config_overrides(config)
  if not config.replay_csv or not os.path.exists(config.replay_csv):
    print(
      "replay mode needs --replay_csv pointing to a generated augmentation CSV "
      f"(not found: {config.replay_csv})"
    )
    return
  arr = np.genfromtxt(config.replay_csv, delimiter=",")
  if arr.ndim == 1:
    arr = arr.reshape(1, -1)

  # Replay uses the scene model with ground/lights (no IK solve, hence no ComTask NaN
  # problem, so the scene model is fine).
  model = build_g1_scene_model(config.render_width, config.render_height)
  mdata = mujoco.MjData(model)  # holds the adapted pose
  # Holds the reference pose, used to rebuild the blue keypoints and other reference
  # overlays.
  ref_data = mujoco.MjData(model)
  nq = model.nq
  act_idx = [
    model.jnt_qposadr[i]
    for i in range(model.njnt)
    if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
  ]
  # The robot-only model only decodes the CSV link_id into a link name (body ids
  # differ between the two models, so the scene model cannot be used directly).
  robot_only = mujoco.MjModel.from_xml_path(str(G1_XML))

  # Column slices; must match the write order of _save_outputs:
  # ref(nq+2) | adapted(36) | force(9) | collision(12)
  a0 = nq + 2  # adapted block start
  f0 = a0 + 36  # force block start: [link_id, F(3), T(3), k, krot]
  # collision block start: [forcefield_k, forcefield_krot, setpoint(3), setrot(4),
  # plane_normal(3)]
  c0 = f0 + 9
  expected = c0 + 12
  if arr.shape[1] != expected:
    print(
      f"CSV has {arr.shape[1]} columns != expected {expected}; probably not an "
      "augmentation file produced by this script."
    )
    return

  timestep = 1.0 / SimulationRunner.SIMULATION_FREQUENCY  # 30fps
  n = arr.shape[0]
  max_frames = (
    n
    if config.max_seconds is None
    else min(n, int(np.ceil(config.max_seconds / timestep)))
  )
  print(f"Replaying {max_frames}/{n} frames (from {config.replay_csv})")

  viewer = renderer = cam = None
  frames: List[np.ndarray] = []
  if config.record_video:
    if imageio is None:
      print("imageio is not installed, cannot record video.")
      return
    renderer = mujoco.Renderer(
      model, height=config.render_height, width=config.render_width
    )
    cam = mujoco.MjvCamera()
    cam.azimuth = config.cam_azimuth
    cam.elevation = config.cam_elevation
    cam.distance = config.cam_distance
    print(f"Re-rendering offscreen to {config.output_filename}")
  else:
    viewer = mujoco.viewer.launch_passive(model, mdata)
    viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 90, -15, 4.0

  def _frame_indices():
    # Viewer mode loops forever (until the window closes); recording runs once.
    if viewer is not None:
      while True:
        yield from range(max_frames)
    else:
      yield from range(max_frames)

  try:
    for row in _frame_indices():
      if viewer is not None and not viewer.is_running():
        break
      loop_start = time.time()

      # adapted goes into mdata, reference into ref_data; forward kinematics on each.
      # ref block columns 0:36 = [pos3, quat_xyzw4, 29 joints] (36:38 after it are foot
      # contacts, which qpos does not use).
      mdata.qpos[:] = _row36_to_qpos(arr[row, a0 : a0 + 36], nq, act_idx)
      mujoco.mj_forward(model, mdata)
      ref_data.qpos[:] = _row36_to_qpos(arr[row, 0:36], nq, act_idx)
      mujoco.mj_forward(model, ref_data)

      # Rebuild overlays identical to the generate-data video (blue keypoint references
      # come from ref_data) and draw them on the adapted pose.
      overlay_inputs = _reconstruct_overlay_inputs(
        arr[row], ref_data, robot_only, f0, c0
      )
      lookat = mdata.body("torso_link").xpos

      if viewer is not None:
        viewer.cam.lookat[:] = lookat
        viewer.user_scn.ngeom = 0
        add_visual_overlays(viewer.user_scn, mdata, *overlay_inputs, 0.0, 0.0)
        viewer.sync()
        time.sleep(max(0, timestep - (time.time() - loop_start)))  # keep 30fps
      else:
        cam.lookat[:] = lookat
        renderer.update_scene(mdata, camera=cam)
        add_visual_overlays(renderer.scene, mdata, *overlay_inputs, 0.0, 0.0)
        frames.append(renderer.render())
  finally:
    if viewer is not None:
      viewer.close()

  if config.record_video and frames:
    print(f"\nSaving {len(frames)} frames to '{config.output_filename}'...")
    imageio.mimsave(config.output_filename, frames, fps=int(1.0 / timestep), quality=7)
    print("Replay video saved.")


# ============================================================================
# Section 10: command-line entry point (main of the original mink_generator_ff.py)
# ============================================================================


def main():
  parser = argparse.ArgumentParser(
    description=(
      "SoftMimic offline compliant augmentation pipeline: G1 IK with mink + several "
      "external force models + dynamic rejection sampling."
    )
  )
  parser.add_argument(
    "mode",
    choices=["interactive", "generate-data", "replay"],
    help=(
      "Run mode: interactive = solve + watch live; generate-data = solve + offscreen "
      "render, saved to CSV; replay = replay a generated CSV."
    ),
  )
  parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help=(
      "MuJoCo XML model path; empty (default) composes a G1 scene with "
      "ground/lights/skybox via the mjlab API."
    ),
  )
  parser.add_argument(
    "--motion_path", type=str, default=DEFAULT_MOTION_PATH, help="Reference motion CSV."
  )
  parser.add_argument(
    "--force_mode",
    type=str,
    choices=[
      "triangle",
      "forcefield",
      "collision-emulator",
      "collision-emulator-1d",
      "zero-wrench",
    ],
    default="triangle",
    help="Model used to generate the external force.",
  )
  parser.add_argument("--seed", type=int, default=42, help="Force-profile random seed.")
  parser.add_argument(
    "--com_cost", type=float, default=0.1, help="CoM task cost in the XY plane."
  )
  parser.add_argument(
    "--com_cost_z_factor", type=float, default=0.00001, help="CoM Z cost multiplier."
  )
  parser.add_argument(
    "--torso_orientation_cost", type=float, default=0.0, help="Torso orientation cost."
  )
  parser.add_argument(
    "--repeat_frame_time",
    type=float,
    default=None,
    help="If set, freeze the reference motion at this time (seconds).",
  )
  parser.add_argument(
    "--num_files", type=int, default=10, help="Number of files for generate-data."
  )
  parser.add_argument(
    "--output_dir",
    type=str,
    default="./augmented_data_mink",
    help="Directory for the generated data.",
  )
  parser.add_argument(
    "--record_video", action="store_true", help="Record the simulation to a video."
  )
  parser.add_argument(
    "--max_seconds",
    type=float,
    default=None,
    help=(
      "Run only the first N seconds (trims offline renders/replays to a watchable "
      "length; by default the whole clip runs)."
    ),
  )
  parser.add_argument(
    "--replay_csv",
    type=str,
    default=None,
    help="Augmentation CSV to replay in replay mode (a generate-data output).",
  )
  parser.add_argument(
    "--output_filename",
    type=str,
    default="ik_simulation.mp4",
    help="Output video file name.",
  )
  parser.add_argument(
    "--render_width", type=int, default=640, help="Video width (px); 1920 for 2K."
  )
  parser.add_argument(
    "--render_height", type=int, default=480, help="Video height (px); 1080 for 2K."
  )
  parser.add_argument(
    "--cam_azimuth", type=float, default=90.0, help="Camera azimuth (90 = side view)."
  )
  parser.add_argument(
    "--cam_elevation", type=float, default=-15.0, help="Camera elevation."
  )
  parser.add_argument(
    "--cam_distance",
    type=float,
    default=4.0,
    help="Camera distance (smaller zooms in, the robot looks bigger).",
  )
  # ── Unified compliance vocabulary knobs (for large-scale force sweeps) ──
  parser.add_argument(
    "--fps",
    type=float,
    default=30.0,
    help=(
      "True frame rate of the reference motion (also sets the sim/record step rate); "
      "pass 50 for a 50Hz source."
    ),
  )
  parser.add_argument(
    "--robot_stiffness_min",
    type=float,
    default=10.0,
    help="K_rob robot stiffness lower bound (N/m).",
  )
  parser.add_argument(
    "--robot_stiffness_max",
    type=float,
    default=1000.0,
    help="K_rob robot stiffness upper bound (N/m).",
  )
  parser.add_argument(
    "--forcefield_stiffness_min",
    type=float,
    default=10.0,
    help="K_env environment stiffness lower bound (N/m).",
  )
  parser.add_argument(
    "--forcefield_stiffness_max",
    type=float,
    default=1000.0,
    help="K_env environment stiffness upper bound (N/m).",
  )
  parser.add_argument(
    "--max_force", type=float, default=140.0, help="External force magnitude cap (N)."
  )
  parser.add_argument(
    "--forceable_links",
    type=str,
    default=None,
    help="Forceable links (comma-separated); empty = the default two wrists.",
  )
  parser.add_argument(
    "--export_qpos_csv",
    type=str,
    default=None,
    help=(
      "Also save the adapted pose trajectory as a plain qpos CSV (input to "
      "qpos_csv_to_motion_npz.py). Only file 0 is recorded."
    ),
  )

  args = parser.parse_args()
  if args.model_path is not None and not os.path.exists(args.model_path):
    print(
      f"Error: model file '{args.model_path}' not found. Leave --model_path empty "
      "to use mjlab's built-in G1 scene."
    )
    return

  config = SimulationConfig(**vars(args))

  if config.mode == "replay":
    run_replay(config)
  elif config.mode == "interactive":
    run_simulation_or_generation(config)
  elif config.mode == "generate-data":
    # Generate multiple files in parallel (a new seed per file). With video recording
    # on, only the first file is recorded.
    if config.record_video:
      print("Warning: generate-data with video recording only records the first file.")
      run_simulation_or_generation(config, 0)
      tasks = [
        (dataclasses_replace(config, record_video=False, seed=config.seed + i), i)
        for i in range(1, config.num_files)
      ]
    else:
      tasks = [
        (dataclasses_replace(config, seed=config.seed + i), i)
        for i in range(config.num_files)
      ]

    if tasks:
      num_workers = min(10, len(tasks))
      print(
        f"\n--- Generating {len(tasks)} files in parallel, "
        f"up to {num_workers} processes ---"
      )
      with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(run_simulation_or_generation, tasks)
    print("\n--- Data generation done ---")


if __name__ == "__main__":
  multiprocessing.freeze_support()
  main()
