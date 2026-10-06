"""CLI: .venv/bin/python3 -m mjlab.tasks.codancing.simple_deploy <subcommand>.

The robot and room tools around the two deploy runners (codance.py and
scripts/deploy_solo_stand.py). Subcommands (tyro):
  bringup / status / restore -- stand the robot up, read it, hand it back.
  measure  -- a wearer's partner profile from VICON.
  vicon    -- the standalone VICON room probe.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import tyro

import mjlab
from mjlab.tasks.codancing.simple_deploy import constants as ctl


def _net_if(value: str) -> str:
  """The NIC the DDS traffic should use, refusing a link that cannot carry it."""
  problem = ctl.net_if_problem(value)
  if problem:
    raise SystemExit(f"--net-if {value}: {problem}")
  return value


@dataclasses.dataclass
class Measure:
  """Measure a wearer's marker-set heights and write their partner profile:
  they stand still in the volume wearing the sets; the mean world z of each
  object over a few seconds, minus the trained proxy's body heights, is the
  profile (partner_profiles.yaml). Re-run to refresh a name. --objects names a
  new wearer's trunk, left-knee and right-knee objects, in that order, and the
  entry keeps them as its vicon_objects; without it the entry's own objects
  are measured, else the human_* set."""

  profile: str
  ip: str  # the VICON tracker's address on the mocap LAN
  objects: str = ""  # comma list: trunk, left knee, right knee
  seconds: float = 5.0
  note: str = ""

  def run(self) -> None:
    from mjlab.tasks.codancing.simple_deploy.codance import VICON_OBJECT_BY_BODY
    from mjlab.tasks.codancing.simple_deploy.profiles import (
      load_profiles,
      measure_cluster_heights,
      object_map,
      write_profile,
    )
    from mjlab.tasks.codancing.simple_deploy.vicon import _connect_client

    if self.objects:
      worn = object_map([n.strip() for n in self.objects.split(",") if n.strip()])
    else:
      entry = load_profiles()[1].get(self.profile)
      worn = (entry.objects if entry else {}) or dict(VICON_OBJECT_BY_BODY)
    names = tuple(dict.fromkeys(worn.values()))
    client = _connect_client(self.ip)
    print(f"measuring {names} for {self.seconds:g} s: stand still")
    heights, frames = measure_cluster_heights(client, names, self.seconds)
    profile = write_profile(
      self.profile,
      heights,
      measured=time.strftime("%Y-%m-%d"),
      note=self.note or f"standing probe, {frames} frames",
      object_by_body=worn,
    )
    print(f"cluster heights [m]: {heights}")
    print(f"profile {profile.name!r} written: offsets {profile.offsets_m}")


@dataclasses.dataclass
class Vicon:
  """Live mocap probe: print the tracker rate and the partner vector the
  policy would see, without a robot. Run it in the mocap room first;
  `codance.py` uses the same code path."""

  ip: str  # the VICON tracker's address on the mocap LAN
  robot_object: str = "g1_pelvis"
  # Human objects, comma list (a wearer's vicon_objects in partner_profiles.yaml,
  # e.g. human_root,human_left_knee,human_right_knee; positions only, rotations
  # are not read). Empty: robot pose only.
  objects: str = ""
  seconds: float = 5.0  # how long to print samples for

  def run(self) -> None:
    from mjlab.tasks.codancing.simple_deploy.vicon import (
      ViconObjects,
      ViconPartnerSource,
    )

    names = tuple(n.strip() for n in self.objects.split(",") if n.strip())
    src = ViconPartnerSource(
      ViconObjects(robot=self.robot_object, bodies=names), ip=self.ip
    )
    rate = None
    try:
      # Informational only: it reads 0.0 while no subject is up; never gate.
      rate = src.client.get_frame_rate()
    except Exception:
      pass
    print(f"tracker {self.ip}: rate={rate} Hz, robot={self.robot_object!r}")
    t0 = time.time()
    while time.time() - t0 < self.seconds:
      if names:
        v = src.sample().numpy()
        print("partner vector [m]:", np.array2string(v, precision=3))
      else:
        import pyvicon_datastream as pv

        if src.client.get_frame() == pv.Result.Success:
          name = self.robot_object
          p = src.client.get_segment_global_translation(name, name)
          q = src.client.get_segment_global_quaternion(name, name)
          print(f"robot {name}: pos_mm={p} quat_wxyz={q}")
        else:
          print("get_frame failed (Tracker down?)")
      time.sleep(0.5)


@dataclasses.dataclass
class Status:
  """Read-only robot status. Subscribes, never commands, never releases.

  Reports all three layers, because they move independently: the switcher
  says WHICH service holds the robot, the loco FSM says WHAT it is doing,
  and the wire says what the motors are actually reporting. Measured on a
  G1, the whole remote sequence (L2+B, L2+UP, R1+X) changed only the FSM.
  """

  net_if: str  # NIC the robot is on (`ip -br link` lists them)
  seconds: float = 2.0

  def run(self) -> None:
    import time as _time

    from mjlab.tasks.codancing.simple_deploy.io import UnitreeBindCfg, UnitreeBindIO

    io = UnitreeBindIO(
      UnitreeBindCfg(net_if=_net_if(self.net_if), release_motion=False)
    )
    io.start()
    mode = io.check_mode()
    fsm_id, fsm_name = io.fsm_state()
    print(f"switcher : check_mode={mode!r}" + ("" if mode else "  (RELEASED)"))
    print(f"loco fsm : id={fsm_id} ({fsm_name})")

    ticks: set[int] = set()
    t0 = _time.perf_counter()
    frame = io.read()
    while _time.perf_counter() - t0 < self.seconds:
      frame = io.read()
      ticks.add(frame.tick)
      _time.sleep(0.0005)
    elapsed = _time.perf_counter() - t0
    io.close()  # read-only probe: release the endpoints rather than reholding

    tau = np.abs(frame.tau_est)
    print(f"wire     : {len(ticks) / elapsed:.0f} Hz  mode_machine={io.mode_machine}")
    attached = int((frame.motor_mode != 0).sum())
    print(f"motors   : {attached}/29 attached (reported mode != 0)")
    print(f"torque   : max {tau.max():.2f}  mean {tau.mean():.2f} N.m")
    print(f"knees    : L {frame.joint_pos[3]:+.3f}  R {frame.joint_pos[9]:+.3f} rad")
    # A 5 N.m threshold from measurement (slack 0.26-0.62, driven
    # 10.9-13.6); replace with a per-joint gravity model if it ever misreads.
    print(f"holding  : {'YES, driven' if tau.max() > 5.0 else 'no, slack or damped'}")


@dataclasses.dataclass
class Bringup:
  """Stand the robot up in software: L2+B -> L2+UP -> R1+X without the remote.

  MOVES THE ROBOT. Drives the loco FSM 1 (damp) -> 4 (stand_up) -> 500
  (start), confirming each step landed before sending the next.

  Works from either starting state: freshly powered on (service already
  loaded at fsm 0) or released by a previous deploy run, in which case it
  restores the service first. No need to run `restore` by hand.

  This automates bring-up. It does NOT replace the remote as the emergency
  damp path: software damp cannot help when the software has stopped running.
  """

  net_if: str  # NIC the robot is on (`ip -br link` lists them)
  stand: bool = True  # --stand False stops after the firmware damp (fsm 1)
  # How long to wait for the loco API before the first FSM command. Raise it
  # if a slower boot still lands in the drop window; the robot the
  # default was measured on answers well inside 10 s.
  loco_ready_s: float = 10.0

  def run(self) -> None:
    from mjlab.tasks.codancing.simple_deploy.io import UnitreeBindCfg, UnitreeBindIO

    io = UnitreeBindIO(
      UnitreeBindCfg(net_if=_net_if(self.net_if), release_motion=False)
    )
    io.start()
    # close() in a finally: bringup hands the robot to its own service, so
    # nothing of ours needs to keep re-holding. Leaving the endpoints open to
    # interpreter teardown makes nanobind report leaked types/functions.
    try:
      print(
        "bringup: the robot WILL move (damp -> stand up -> operating stance)"
        if self.stand
        else "damp only: driving the loco FSM to 1, what the remote's L2+B does"
      )
      observed = io.bringup(
        stand=self.stand,
        loco_ready_s=self.loco_ready_s,
        log=lambda m: print(m, flush=True),
      )
      print("\nsummary:")
      for label, fsm_id, fsm_name in observed:
        print(f"  {label:<9} -> fsm_id={fsm_id} ({fsm_name})")
      frame = io.read()
      tau = float(np.abs(frame.tau_est).max())
      state = "standing" if self.stand else "damped  "
      print(f"{state}: max|tau|={tau:.2f} N.m  knees {frame.joint_pos[3]:+.3f}")
      if self.stand:
        print("release before running a policy")
    finally:
      io.close()


@dataclasses.dataclass
class Restore:
  """Hand the robot back to its onboard service after a deploy run.

  On a real G1 this lands at fsm_id 0 (zero_torque), the same state a
  power cycle gives. The service does NOT grab the joints, so the robot stays
  slack; stand it with L2+B then L2+UP on the remote. Keep it supported.

  Damps first so the joints are slack rather than fighting the arriving
  service. A power cycle restores it equally well if you prefer that.
  """

  net_if: str  # NIC the robot is on (`ip -br link` lists them)
  mode: str = "ai"
  damp_s: float = 1.0

  def run(self) -> None:
    from mjlab.tasks.codancing.simple_deploy.io import UnitreeBindCfg, UnitreeBindIO

    # release_motion=False: restoring is the opposite of releasing.
    io = UnitreeBindIO(
      UnitreeBindCfg(net_if=_net_if(self.net_if), release_motion=False)
    )
    io.start()
    # Same as bringup: the onboard service owns the robot afterwards, so
    # release the endpoints rather than holding them to interpreter teardown.
    try:
      before = io.check_mode()
      print(f"before: check_mode={before!r} fsm={io.fsm_state()}")
      if before:
        print(f"already holding {before!r}; nothing to restore.")
        return
      got = io.restore_ai(self.mode, damp_s=self.damp_s)
      print(
        f"after : check_mode={got['mode']!r} fsm_id={got['fsm_id']} ({got['fsm_name']})"
      )
      if not got["mode"]:
        raise SystemExit(
          f"select_mode({self.mode!r}) did not take. Power-cycle the robot, "
          "which is the measured-good recovery path."
        )
      print(
        "The robot is NOT necessarily holding itself: a freshly loaded service "
        "was measured sitting at fsm_id 0 (zero_torque). Check the fsm above, "
        "and use L2+B then L2+UP on the remote to stand it."
      )
    finally:
      io.close()


def main() -> None:
  cmd = tyro.cli(  # type: ignore[call-overload]
    Vicon | Status | Restore | Bringup | Measure,
    config=mjlab.TYRO_FLAGS,
  )
  cmd.run()


if __name__ == "__main__":
  main()
