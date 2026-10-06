"""G1 robot motion NPZ -> 38-col qpos CSV for softmimic_mink_augment.py.

The "convert NPZ to CSV" path: the tracked robot NPZs
(data/reference_motion_edits_g1_g1_waltz/, data/reference_motion_stand/) are G1 @ 50 Hz with joint_pos (T,29) in G1_JOINT_NAMES order -- verified byte-order
identical to the augmenter's id-ascending 29-joint order, so columns map 1:1
(no reorder). Quaternions flip wxyz -> xyzw. Feet are padded (the augmenter pins
them to the reference regardless, so the pad value is inert).

CSV layout (38 cols): [root_pos(3), root_quat_xyzw(4), 29 joints, 2 foot contacts].
Run the augmenter at the NPZ's native fps (e.g. --fps 50); no resampling here --
the motion registry interpolates to the control rate at consume time.
"""

import numpy as np
import tyro

import mjlab


def npz_to_qpos_rows(d) -> np.ndarray:
  """Origin robot NPZ -> (T,38) qpos rows [root_pos, root_quat xyzw, 29 joints, feet].

  The one conversion between the origin NPZ schema (root_pos, root_quat wxyz,
  joint_pos) and the augmenter's CsvMotionLib row layout; the CSV writer below and
  augment_multilink's direct-NPZ input both call this, so the layout has a single
  definition. Raises KeyError with guidance when fed a motion-LIBRARY npz (the
  qpos_csv_to_motion_npz output, which has body_* keys instead) -- outputs are not
  inputs.
  """
  missing = [k for k in ("root_pos", "root_quat", "joint_pos") if k not in d]
  if missing:
    raise KeyError(
      f"NPZ lacks {missing}: expected a source robot motion NPZ "
      "(root_pos, root_quat wxyz, joint_pos). A motion-library NPZ "
      "(body_pos_w/body_quat_w..., the qpos_csv_to_motion_npz OUTPUT) is a derived "
      "artifact, not an augmentation input."
    )
  root_pos = d["root_pos"]  # (T,3)
  quat_xyzw = d["root_quat"][:, [1, 2, 3, 0]]  # wxyz -> xyzw
  joints = d["joint_pos"]  # (T,29), G1_JOINT_NAMES order == augmenter order
  assert joints.shape[1] == 29, f"expected 29 joints, got {joints.shape[1]}"
  feet = np.ones((root_pos.shape[0], 2))  # inert pad (feet pinned in IK)
  return np.concatenate([root_pos, quat_xyzw, joints, feet], axis=1)


def npz_fps(d) -> float | None:
  """The origin NPZ's frame rate, or None when absent."""
  return float(np.asarray(d["fps"]).reshape(-1)[0]) if "fps" in d else None


def main(input_file: str, output_file: str):
  """G1 robot NPZ -> qpos CSV.

  Args:
    input_file: G1 robot motion NPZ (root_pos, root_quat wxyz, joint_pos (T,29)).
    output_file: output 38-col qpos CSV path.
  """
  d = np.load(input_file)
  out = npz_to_qpos_rows(d)
  np.savetxt(output_file, out, delimiter=",")
  fps = npz_fps(d)
  print(
    f"wrote {out.shape} to {output_file} (source fps={fps}) -- run augmenter --fps {fps or '?'}"
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
