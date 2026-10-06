#!/usr/bin/env python3
"""Deploy the solo-stand policy on a Unitree G1, straight from its ONNX.

    .venv/bin/python3 scripts/deploy_solo_stand.py --net-if <nic> \\
        --onnx data/checkpoints/stand.onnx --stiffness-linear 140 --dry-run
    .venv/bin/python3 scripts/deploy_solo_stand.py --net-if <nic> \\
        --onnx data/checkpoints/stand.onnx --stiffness-linear 140

Everything runs from the policy artifact plus the reference motion NPZ: no
simulator, no training config, no checkpoint.

Launch with the venv python, NOT with ``uv run``. Measured: on Ctrl-C the uv
wrapper takes the SIGINT too and kills this process 0.9 s in, truncating the
terminal damp hold that exists precisely so the robot is left latched on a
damp command. Under the venv python directly, SIGINT, SIGTERM and SIGHUP (an
SSH link dropping) all complete the full hold and exit clean.

Hardware order, in this order, every time:

1. ``scripts/check_g1_remote.py``  confirm the remote decodes; read-only,
   robot not released, no policy artifact involved
2. ``--dry-run``  read and infer on the robot, damp only, robot hung
3. e-stop drills
4. the same command without ``--dry-run``: the policy, robot hung

From step 1 on, constructing the comms talks to the robot, and from step 2 on
it RELEASES the onboard service, so the robot must already be hung or
supported before the process starts.
"""

from __future__ import annotations

import argparse
import dataclasses
import signal
import time
from pathlib import Path

import numpy as np

from mjlab.tasks.codancing.simple_deploy import constants as ctl
from mjlab.tasks.codancing.simple_deploy.stand import (
  ControlStep,
  StandController,
  StiffnessCommand,
  quat_matrix,
)

CONTROL_DT = ctl.CONTROL_DT  # artifacts state theirs and are checked against it.

# The rotational half of the compliance channel. The released pools write 1.0
# there on every frame of every clip, so the policy never saw another value
# and 1.0 is the only in-distribution setting.
#
# The LINEAR half is deliberately NOT defaulted. It is the knob a run should
# say out loud: the pools span 10 to 1000 N/m, with a median of 140.
#
# Neither number has a physical consumer: they change what the policy expects
# to be pushed with, not how stiff the robot actually is.
DEFAULT_STIFFNESS_ROTATIONAL = 1.0
# Wall-clock episode length. The solo-stand observation carries no phase,
# clock or cursor, so the policy is time-invariant at run time and this is
# purely an operator and safety choice. 10 s is the standing reference clip's
# length.
DEFAULT_EPISODE_S = 10.0


# ------------------------------------------------------------------- dry run


def _tilt_degrees(quat_wxyz: np.ndarray) -> float:
  """Angle between the body's up axis and world up."""
  up = quat_matrix(np.asarray(quat_wxyz, dtype=np.float64))[:, 2]
  return float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))


def dry_run(controller: StandController, io, seconds: float) -> int:
  """Read, assemble, infer, and damp. Never commands a position target.

  With the robot hung and slack this is where two things get confirmed before
  any gains go on. Motor order: move one joint by hand and check the joint the
  readout names is the one you moved. IMU and waist chain: bend the waist and
  watch the reference-orientation error grow while the tilt stays put, which
  is exactly the signal term 6 carries and projected gravity does not.
  """

  # Same shape as the runner's terminal damp: every exit path, including a
  # signal, damps for at least a second before the process goes away, because
  # the robot latches the last LowCmd it received and a single publish can be
  # lost. SIGHUP matters as much as SIGINT here: closing the terminal must not
  # leave the robot on an unheld command.
  def _bail(signum, _frame):
    raise KeyboardInterrupt(f"signal {signum}")

  for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    try:
      signal.signal(sig, _bail)
    except ValueError:
      pass  # not the main thread

  io.start()
  print(f"dry run: {seconds:g} s, damp only, no position target is ever sent")
  print("move one joint by hand, then bend the waist, and watch the rows\n")

  last_q = np.asarray(io.read().joint_pos).copy()
  t0 = time.monotonic()
  deadline = t0 + seconds
  next_print = t0 + 0.25  # first row after a real interval, so the rate means something
  step: ControlStep | None = None
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
        offset = last_q - controller.default_joint_pos
        moved = np.argsort(-np.abs(offset))[:3]
        names = "  ".join(
          f"{ctl.JOINT_NAMES[j].removesuffix('_joint')}={offset[j]:+.3f}" for j in moved
        )
        # Term 6 is the rotation from the live torso frame to the reference
        # one; its trace gives the angle, which is 0 when they agree.
        ori = controller.newest(step.obs, "robot_ref_anchor_ori_b")
        cos_angle = (ori[0] + ori[3] + _third_diagonal(ori) - 1.0) / 2.0
        ref_err = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        command = np.abs(step.target - last_q).max()
        print(
          f"{ticks / max(time.monotonic() - t0, 1e-9):5.1f} Hz  "
          f"tilt {_tilt_degrees(frame.quat_wxyz):5.1f} deg  "
          f"ref-ori {ref_err:5.1f} deg  "
          f"would-command {command:5.3f} rad | {names}",
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


def _third_diagonal(ori: np.ndarray) -> float:
  """m22 of a rotation matrix, from its first two columns.

  The third column is the cross product of the first two, and only its last
  entry is needed: m22 = m00*m11 - m01*m10.
  """
  return float(ori[0] * ori[3] - ori[1] * ori[2])


# ----------------------------------------------------------------------- run


def _per_slot(values: list[float], flag: str) -> tuple[float, float]:
  """One value covers both wrists; two are left then right.

  Slot-major L-then-R is the order every compliance channel uses, including
  the four numbers the observation itself carries, so the flag reads in the
  same order as the vector it ends up in.
  """
  if len(values) == 1:
    return values[0], values[0]
  if len(values) == 2:
    return values[0], values[1]
  raise SystemExit(
    f"{flag}: give one value for both wrists, or two as left then right; "
    f"got {len(values)}"
  )


def run(args) -> int:
  from mjlab.tasks.codancing.simple_deploy.loop import LoopCfg, Runner
  from mjlab.tasks.codancing.simple_deploy.record import RunRecorder

  # FIRST, before the artifact is even opened: this touches nothing (no DDS
  # participant, no motion-service release), and a dead robot link reaches
  # CycloneDDS as "does not match an available interface", the same words a
  # misspelled name gets, and without this check a dead link would only show
  # 5 s later as "no LowState", with the robot already hung and released.
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
  controller = StandController(
    args.onnx, reference_path=args.reference, stiffness=stiffness
  )
  print(
    f"stiffness: L {stiffness.linear_left:g} N/m / {stiffness.rotational_left:g} rot"
    f"   R {stiffness.linear_right:g} N/m / {stiffness.rotational_right:g} rot"
    f"   -> log {np.round(stiffness.as_log(), 4)}"
  )

  cfg = LoopCfg(
    max_episode_s=args.episode_s,
    auto_exit_s=args.seconds if args.comms == "dummy" else 0.0,
    stiffness_step=args.stiffness_step,
  )
  if args.stiffness_step > 0:
    lo, hi = StandController.STIFFNESS_LINEAR_RANGE
    print(
      f"stiffness step: remote up +{args.stiffness_step:g} / down "
      f"-{args.stiffness_step:g} N/m on both wrists, clamped to {lo:g} to {hi:g}"
    )
  if args.no_joint_trips:
    # Operator opt-out for contact-heavy runs. The standing reference IS the
    # default keyframe, so pulling a hand far enough to get the yield the
    # policy was trained to give reads to the deviation trip as a fall, and a
    # hand held at its stop keeps a joint clamped for as long as it is held.
    cfg.safety.disarm_joint_trips()
    print(
      "joint-space trips DISARMED (--no-joint-trips): deviation-vs-reference "
      "and clamp-streak; tilt/velocity/staleness/overrun stay armed"
    )

  if args.comms == "dummy":
    from mjlab.tasks.codancing.simple_deploy.io import DummyIO

    io = DummyIO(controller.default_joint_pos)
  else:
    from mjlab.tasks.codancing.simple_deploy.io import UnitreeBindCfg, UnitreeBindIO

    io = UnitreeBindIO(UnitreeBindCfg(net_if=args.net_if))

  if args.dry_run:
    return dry_run(controller, io, args.seconds)

  recorder = RunRecorder(
    "logs/simple_deploy/runs",
    identity={
      "artifact": str(Path(args.onnx).resolve()),
      "reference": str(Path(args.reference).resolve()),
      "stiffness": dataclasses.asdict(stiffness),
      "comms": args.comms,
      "net_if": args.net_if,
      "no_joint_trips": bool(args.no_joint_trips),
      "kp": controller.kp.tolist(),
      "kd": controller.kd.tolist(),
    },
  )
  Runner(controller, io, cfg, recorder).run()
  return 0


# ----------------------------------------------------------------------- cli


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Deploy the solo-stand policy on a Unitree G1, from its ONNX."
  )
  parser.add_argument(
    "--onnx",
    required=True,
    help="path to the exported policy (.onnx)",
  )
  parser.add_argument(
    "--reference",
    default=str(ctl.STAND_REFERENCE),
    help="reference motion NPZ (the unforced standing reference)",
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
    help="read, assemble and infer, but publish damp only",
  )
  parser.add_argument("--seconds", type=float, default=30.0, help="dry-run duration")
  parser.add_argument(
    "--no-joint-trips",
    action="store_true",
    help="disarm the two JOINT-SPACE trips (deviation vs the standing "
    "reference, joint-limit clamp streak) for contact-heavy runs where a "
    "hand is pulled or held at its stop; targets are still clamped to the "
    "limits, and tilt/joint-velocity/staleness/overrun stay armed",
  )
  parser.add_argument(
    "--episode-s",
    type=float,
    default=DEFAULT_EPISODE_S,
    help="episode length in seconds; Y stops sooner, this is the ceiling. "
    "`inf` runs until the operator stops it, which the standing policy "
    "supports outright: the reference is constant, so it serves the same pose forever "
    "and nothing but this clock ends an episode. Budget about 1 GB of RAM per "
    "hour for the tick record, which is held until exit",
  )
  parser.add_argument(
    "--stiffness-linear",
    type=float,
    nargs="+",
    required=True,
    metavar="N/m",
    help="what the policy is told to expect at the wrists: one value for "
    "both, or two as LEFT RIGHT. Trained range is 10 to 1000 N/m sampled "
    "log-uniformly, median 140. No default on purpose: state it every run.",
  )
  parser.add_argument(
    "--stiffness-rotational",
    type=float,
    nargs="+",
    default=[DEFAULT_STIFFNESS_ROTATIONAL],
    metavar="N.m/rad",
    help="one value for both wrists, or two as LEFT RIGHT; every training "
    "frame carries 1.0 here, so the default is the only in-distribution value",
  )
  parser.add_argument(
    "--stiffness-step",
    type=float,
    default=0.0,
    metavar="N/m",
    help="map the remote's up / down to +/- this many N/m on BOTH wrists' "
    "linear stiffness while READY or RUNNING, clamped to the trained 10 to "
    "1000 N/m, so one run sweeps the channel (start it at --stiffness-linear "
    "40 and step by 100, say). Each change is printed and logged as a run "
    "event. 0 (default) leaves the buttons unmapped",
  )

  args = parser.parse_args()
  return run(args)


if __name__ == "__main__":
  raise SystemExit(main())
