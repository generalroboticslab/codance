"""Convert a G1 qpos CSV trajectory into an NPZ the codancing motion library loads
directly (with body world poses + velocities).

This is the "bridge" from SoftMimic offline compliance augmentation to AMP expert
motions:
  softmimic_mink_augment.py --export_qpos_csv adapted.csv   # adapted pose trajectory
  qpos_csv_to_motion_npz.py --input_file adapted.csv --output_file compliant.npz

Input CSV column layout = [root_pos(3), root_quat_xyzw(4), 29 joints] (= the
augmentation script's adapted block, also the input format of
src/mjlab/scripts/csv_to_npz.py).

It reuses csv_to_npz.py's proven MotionLoader (interpolation + slerp + velocities)
and FK extraction path, dropping only its hardcoded /tmp write + forced
wandb upload in favor of a local np.savez (large-scale experiments need local,
configurable, network-free output).
Output keys match the motion library loader
(src/mjlab/tasks/codancing/motion/loaders.py):
  fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w
  (body axis in whole-model link order, wxyz).
"""

import numpy as np
import torch
import tyro

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader  # reuse interp/velocity/slerp logic
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.codancing.datagen_meta import build_datagen_meta
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

# Canonical order of the G1's 29 actuated joints (same as csv_to_npz.py; CSV columns
# from index 7 on follow this order).
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


def main(
  input_file: str,
  output_file: str,
  input_fps: float = 50.0,
  output_fps: float | None = None,
  device: str = "cuda:0",
):
  """qpos CSV → motion NPZ (saved locally, no wandb).

  The default output_fps=None stores the native frame rate (=input_fps) with no
  offline resampling: the motion library registry (registry.py _frame_blend) does
  continuous-time lerp/slerp to the control rate by fps at consume time, so 30↔50Hz
  is free. Set output_fps only to change the frame rate or fill in frames.

  Args:
    input_file: input qpos CSV ([pos3, quat_xyzw4, 29 joints]).
    output_file: output NPZ path.
    input_fps: input CSV frame rate (e.g. 50Hz for reference_motion_edits).
    output_fps: output frame rate; empty = native (input_fps), no resampling.
    device: torch device.
  """
  if output_fps is None:
    output_fps = input_fps
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARN] CUDA unavailable, falling back to CPU (slow).")
    device = "cpu"

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  motion = MotionLoader(
    motion_file=input_file,
    input_fps=int(input_fps),
    output_fps=int(output_fps),
    device=device,
  )
  robot: Entity = scene["robot"]
  joint_idx = robot.find_joints(G1_JOINT_NAMES, preserve_order=True)[0]

  log: dict[str, list] = {
    "joint_pos": [], "joint_vel": [],
    "body_pos_w": [], "body_quat_w": [],
    "body_lin_vel_w": [], "body_ang_vel_w": [],
  }  # fmt: skip
  scene.reset()
  print(f"FK replay of {motion.output_frames} frames @ {output_fps}fps ...")
  for _ in range(motion.output_frames):
    (pos, rot, lin_vel, ang_vel, dof_pos, dof_vel), _ = motion.get_next_state()
    root = robot.data.default_root_state.clone()
    root[:, 0:3] = pos
    root[:, :2] += scene.env_origins[:, :2]
    root[:, 3:7] = rot
    root[:, 7:10] = lin_vel
    root[:, 10:] = ang_vel
    robot.write_root_state_to_sim(root)
    jp = robot.data.default_joint_pos.clone()
    jv = robot.data.default_joint_vel.clone()
    jp[:, joint_idx] = dof_pos
    jv[:, joint_idx] = dof_vel
    robot.write_joint_state_to_sim(jp, jv)
    sim.forward()
    scene.update(sim.mj_model.opt.timestep)
    log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
    log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0].cpu().numpy().copy())
    log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0].cpu().numpy().copy())

  arr = {k: np.stack(v, axis=0) for k, v in log.items()}
  # Explicit keyword arguments (not **dict), so ty does not match **kwargs against
  # savez's allow_pickle: bool.
  np.savez(
    output_file,
    fps=np.array([output_fps], dtype=np.float32),
    joint_pos=arr["joint_pos"],
    joint_vel=arr["joint_vel"],
    body_pos_w=arr["body_pos_w"],
    body_quat_w=arr["body_quat_w"],
    body_lin_vel_w=arr["body_lin_vel_w"],
    body_ang_vel_w=arr["body_ang_vel_w"],
    run_meta=build_datagen_meta(source=str(input_file)),
  )
  print(
    f"saved {arr['joint_pos'].shape[0]} frames to '{output_file}' "
    f"(body axis: {arr['body_pos_w'].shape[1]} links)."
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
