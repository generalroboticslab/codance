"""Join the adapted qpos CSV + multi-link contact NPZ into **one wide CSV table with a
header**, one row per frame.

The two products normally live apart: qpos in the CSV (the bridge's input), forces /
stiffnesses / setpoints in the NPZ (consumed by the online task). Looking at them
together (how much force hits the right wrist at frame 215, where the elbow is then,
how far the setpoint is from the wrist) otherwise means writing your own join. This
script joins them once, for pandas / Excel / eyeballing.

**It is a read-only side product, not part of any pipeline.** The bridge only takes
the headerless 36-column CSV and the online task only reads the NPZ; nothing consumes
this wide table, so a broken one does not affect training.

Usage (no GL needed):
    uv run python scripts/multilink_table_csv.py \\
        --adapted <adapted>.csv --contact <contact>.npz

Without --out it lands next to the adapted CSV, named <adapted stem>_table.csv.

Column layout: ``frame`` + 36 qpos columns (same names and order as
augment_multilink.ADAPTED_COLUMNS) + 22 columns per slot, 27 when the NPZ carries the 5
audit channels (prefix ``slot{k}_{link name}``). Slot order = the NPZ's
``slot_link_names``, matching the NPZ's slot axis. Single-contact (K=1) NPZs work
too; their slot name is ``(varies)``, so the prefix degrades to ``slot0``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augment_multilink import ADAPTED_COLUMNS  # noqa: E402

# Per-slot channels to flatten: (NPZ key, column-name suffixes). Order = column order.
_SLOT_CHANNELS: list[tuple[str, tuple[str, ...]]] = [
  ("force", ("force_x", "force_y", "force_z")),
  ("torque", ("torque_x", "torque_y", "torque_z")),
  ("robot_stiffness", ("k_rob",)),
  ("robot_rot_stiffness", ("k_rot_rob",)),
  ("forcefield_stiffness", ("k_ff",)),
  ("forcefield_rot_stiffness", ("k_rot_ff",)),
  ("setpoint_pos", ("setpoint_x", "setpoint_y", "setpoint_z")),
  ("setpoint_quat", ("setpoint_qw", "setpoint_qx", "setpoint_qy", "setpoint_qz")),
  ("plane_normal", ("plane_nx", "plane_ny", "plane_nz")),
  ("link_id", ("link_id",)),
]

# Optional audit channels (the multilink augmenter writes all five, single-contact
# NPZs carry none): sampled intent / intent after pair resolution ("single" when the
# other hand has no force this frame) / guide branch / backoff outward-rotation angle
# / backoff scale. Presence is checked per key and a missing key skips its column.
# String columns go to the CSV as-is, numeric columns as floats.
_OPT_SLOT_CHANNELS: list[tuple[str, bool]] = [
  ("intent", False),
  ("intent_final", False),
  ("guide_branch", False),
  ("backoff_rot_deg", True),
  ("backoff_scale", True),
]


def slot_prefix(link_name: str, k: int) -> str:
  """Column prefix of slot k: ``slot{k}_{link name without _link}``, or just
  ``slot{k}`` when no name is available.

  The slot number **must** come first: the link name alone collides with qpos
  prefixes. Slot name ``left_wrist_yaw_link`` minus its suffix is ``left_wrist_yaw``,
  and qpos happens to have a column named ``left_wrist_yaw_joint``, so filtering on
  ``left_wrist_yaw_`` would pull that joint column in too (column names stay unique,
  so pandas does not complain, but prefix-grouped analysis silently mixes it in).
  The ``slot{k}_`` prefix avoids the collision and also carries the NPZ's slot axis
  index into the table.
  A single-contact NPZ has the placeholder slot name "(varies)", so the prefix
  degrades to a bare ``slot{k}``.
  """
  name = str(link_name)
  if not name or not name.replace("_", "").isalnum():
    return f"slot{k}"
  return f"slot{k}_{name.removesuffix('_link')}"


def build_table(adapted_csv: str, contact_npz: str) -> tuple[list[str], list]:
  """Return (header, list of columns, each (T,)); string audit columns are mixed with
  numeric ones. T is the shorter of the two inputs."""
  qpos = np.loadtxt(adapted_csv, delimiter=",")
  if qpos.ndim == 1:
    qpos = qpos[None, :]
  assert qpos.shape[1] == len(ADAPTED_COLUMNS), (
    f"{adapted_csv} has {qpos.shape[1]} columns, expected {len(ADAPTED_COLUMNS)} "
    "(was the copy with a header passed in?)"
  )
  d = np.load(contact_npz, allow_pickle=False)
  slot_names = [str(s) for s in np.asarray(d["slot_link_names"]).reshape(-1)]
  k_slots = len(slot_names)
  # The producer writes one row per frame on both sides, so the lengths should match;
  # min() covers a manual mismatch (which shows up in the printout).
  n = min(qpos.shape[0], int(np.asarray(d["force"]).shape[0]))

  header = ["frame", *ADAPTED_COLUMNS]
  cols: list[np.ndarray] = [np.arange(n, dtype=float), *qpos[:n].T]
  for k in range(k_slots):
    pre = slot_prefix(slot_names[k], k)
    fmag = np.linalg.norm(np.asarray(d["force"])[:n, k], axis=-1)
    header.append(f"{pre}_fmag")  # redundant but the most-viewed column, so first
    cols.append(fmag)
    for key, suffixes in _SLOT_CHANNELS:
      arr = np.asarray(d[key])[:n, k]
      arr = arr[:, None] if arr.ndim == 1 else arr
      assert arr.shape[1] == len(suffixes), (
        f"{key} slot width {arr.shape[1]} != {len(suffixes)}"
      )
      for j, suf in enumerate(suffixes):
        header.append(f"{pre}_{suf}")
        cols.append(arr[:, j].astype(float))
    for key, numeric in _OPT_SLOT_CHANNELS:
      if key not in d:
        continue  # the NPZ does not carry this channel
      col = np.asarray(d[key])[:n, k]
      header.append(f"{pre}_{key}")
      cols.append(col.astype(float) if numeric else col.astype(str))
  return header, cols


def write_table(adapted_csv: str, contact_npz: str, out: str | None = None) -> str:
  out = out or f"{os.path.splitext(adapted_csv)[0]}_table.csv"
  header, cols = build_table(adapted_csv, contact_npz)
  os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
  with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    # Row-wise zip, not np.stack: the audit columns are strings, and stacking them
    # together would turn every number in the table into a string.
    for row in zip(*cols, strict=True):
      w.writerow([x if isinstance(x, str) else float(x) for x in row])
  print(f"wrote {out}  ({len(cols[0])} frames x {len(header)} columns)")
  return out


def main() -> None:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument(
    "--adapted", required=True, help="adapted qpos CSV (the 36-column headerless one)"
  )
  ap.add_argument("--contact", required=True, help="contact NPZ (multi-link or single)")
  ap.add_argument(
    "--out", default=None, help="output CSV; empty = <adapted stem>_table.csv"
  )
  args = ap.parse_args()
  write_table(args.adapted, args.contact, args.out)


def _demo() -> None:
  """Self-check: synthesize a small input pair and assert column count, alignment and
  values (no real clip needed)."""
  import tempfile

  from mjlab.tasks.codancing.motion.contact_reference import write_contact_npz_multi

  d = tempfile.mkdtemp()
  t, k = 5, 2
  qpos = np.arange(t * len(ADAPTED_COLUMNS), dtype=float).reshape(t, -1)
  acsv = os.path.join(d, "a.csv")
  np.savetxt(acsv, qpos, delimiter=",")
  force = np.zeros((t, k, 3))
  force[2, 1] = [3.0, 4.0, 0.0]  # |F| = 5, checks the fmag column
  npz = os.path.join(d, "c.npz")
  write_contact_npz_multi(
    npz,
    link_id=np.full((t, k), -1, dtype=np.int64),
    link_name=np.full((t, k), "", dtype=object).astype(str),
    force=force,
    torque=np.zeros((t, k, 3)),
    robot_stiffness=np.full((t, k), 140.0),
    robot_rot_stiffness=np.ones((t, k)),
    forcefield_stiffness=np.full((t, k), 100.0),
    forcefield_rot_stiffness=np.ones((t, k)),
    setpoint_pos=np.zeros((t, k, 3)),
    setpoint_quat_wxyz=np.tile([1.0, 0, 0, 0], (t, k, 1)),
    plane_normal=np.zeros((t, k, 3)),
    slot_link_names=np.array(["left_wrist_yaw_link", "right_wrist_yaw_link"]),
    events=[],
  )
  out = write_table(acsv, npz)
  header = open(out).readline().strip().split(",")
  data = np.loadtxt(out, delimiter=",", skiprows=1)

  per_slot = 1 + sum(len(s) for _, s in _SLOT_CHANNELS)  # fmag + every channel
  assert len(header) == 1 + len(ADAPTED_COLUMNS) + k * per_slot, len(header)
  assert data.shape == (t, len(header))
  # the qpos section is copied over exactly, with no shift
  np.testing.assert_allclose(data[:, 1 : 1 + len(ADAPTED_COLUMNS)], qpos)
  np.testing.assert_allclose(data[:, 0], np.arange(t))
  # slot prefixes come from link names, and fmag is right
  assert "slot1_right_wrist_yaw_fmag" in header, header
  assert abs(data[2, header.index("slot1_right_wrist_yaw_fmag")] - 5.0) < 1e-9
  assert data[0, header.index("slot1_right_wrist_yaw_fmag")] == 0.0
  # slot prefixes must not collide with qpos prefixes (else prefix filtering would mix
  # in joint columns)
  qcols = set(ADAPTED_COLUMNS)
  for c in header:
    if c.startswith("slot"):
      assert c not in qcols
  for pre in ("slot0_left_wrist_yaw_", "slot1_right_wrist_yaw_"):
    assert not [c for c in ADAPTED_COLUMNS if c.startswith(pre)]
  print(f"OK:{out}  ({len(header)} columns)")


if __name__ == "__main__":
  _demo() if len(sys.argv) == 1 else main()
