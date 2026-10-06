"""Default standing-pose origin clip: KNEES_BENT_KEYFRAME held for N frames.

The clip's pose = the **default standing pose** that the stand policy's episode
resets start from (the motion cursor's `default` start, phase 0): at episode frame 0
the robot already sits on the reference, with no convergence from some other pose.
Use: the source clip of the stand pool (`just multilink-rc-paper small false stand`).

The output schema matches the reference_motion_edits origin robot NPZ
(root_pos/root_quat(wxyz)/root_lin_vel/root_ang_vel + joint_pos/joint_vel +
body_*_w + fps + run_meta), so the same file is both the augmentation input
(augment_multilink reads root_pos/root_quat/joint_pos) and the motion library's
free_motion_file (the loader reads body_*).
"""

import numpy as np
import torch
import tyro

import mjlab
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import KNEES_BENT_KEYFRAME
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.codancing.datagen_meta import build_datagen_meta
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg


def main(
  output_file: str = "data/reference_motion_stand/20260814_001_robot.npz",
  frames: int = 501,
  fps: float = 50.0,
  device: str = "cuda:0",
):
  """Generate the standing-pose clip.

  Default 501 frames @50Hz = 10.0s, the same order as a waltz clip's duration.

  Args:
    output_file: output NPZ path.
    frames: frame count (duration = (frames-1)/fps).
    fps: frame rate (50 Hz, the rate of the tracked waltz clips).
    device: torch device (single-frame FK, cpu works too).
  """
  if device.startswith("cuda") and not torch.cuda.is_available():
    device = "cpu"
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / fps
  scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  robot: Entity = scene["robot"]
  scene.reset()

  # Guard that the entity default == KNEES_BENT_KEYFRAME: scene-side drift (a swapped
  # keyframe) fails right here instead of producing a "default stance" clip that is
  # not what its name says.
  kf = KNEES_BENT_KEYFRAME
  jp_default = robot.data.default_joint_pos[0]
  for joint_name, expect in [
    ("left_knee_joint", 0.669),
    ("left_hip_pitch_joint", -0.312),
    ("right_ankle_pitch_joint", -0.363),
    ("right_elbow_joint", 0.6),
    ("right_shoulder_roll_joint", -0.2),
    ("waist_yaw_joint", 0.0),
  ]:
    idx = robot.find_joints([joint_name])[0][0]
    got = float(jp_default[idx])
    assert abs(got - expect) < 1e-6, f"{joint_name}: default {got} != keyframe {expect}"
  assert tuple(kf.pos) == (0, 0, 0.76), kf.pos

  root = robot.data.default_root_state.clone()  # default pose + zero velocity
  root[:, :2] += scene.env_origins[:, :2]
  robot.write_root_state_to_sim(root)
  robot.write_joint_state_to_sim(
    robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
  )
  sim.forward()
  scene.update(sim.mj_model.opt.timestep)

  def tile(x: torch.Tensor) -> np.ndarray:
    frame = x[0].cpu().numpy().astype(np.float32)
    return np.repeat(frame[None], frames, axis=0)

  joint_vel = tile(robot.data.joint_vel)
  assert np.abs(joint_vel).max() == 0.0, "standing clip must be zero-velocity"
  np.savez(
    output_file,
    fps=np.array([fps], dtype=np.float64),
    joint_pos=tile(robot.data.joint_pos),
    joint_vel=joint_vel,
    body_pos_w=tile(robot.data.body_link_pos_w),
    body_quat_w=tile(robot.data.body_link_quat_w),
    body_lin_vel_w=tile(robot.data.body_link_lin_vel_w),
    body_ang_vel_w=tile(robot.data.body_link_ang_vel_w),
    root_pos=tile(robot.data.root_link_pos_w),
    root_quat=tile(robot.data.root_link_quat_w),
    root_lin_vel=np.zeros((frames, 3), dtype=np.float32),
    root_ang_vel=np.zeros((frames, 3), dtype=np.float32),
    run_meta=build_datagen_meta(source="g1_constants.KNEES_BENT_KEYFRAME"),
  )
  print(
    f"wrote {output_file}: {frames} frames @ {fps:g}fps "
    f"({(frames - 1) / fps:.2f}s), root z={float(root[0, 2]):.3f}"
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
