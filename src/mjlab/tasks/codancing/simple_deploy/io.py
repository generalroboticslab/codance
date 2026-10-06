"""RobotIO: one small interface, two implementations.

``UnitreeBindIO`` talks DDS to the G1 through unitree-sdk2-bind (the
``deploy`` extra); ``DummyIO`` needs no robot and no DDS. Starting the comms
RELEASES the onboard motion service, so the robot must be hung or
supported before the process starts, and damp is a STATE the loop keeps
publishing (kp 0, kd 8, q = measured), never a one-shot.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

NUM_JOINTS = 29
DAMP_KD = 8.0  # the damping gain of the G1's damp command on hardware

# wireless_remote 16-bit button map (SDK bitfield, verified across sources).
BUTTON_BITS = {
  "R1": 0,
  "L1": 1,
  "start": 2,
  "select": 3,
  "R2": 4,
  "L2": 5,
  "F1": 6,
  "F2": 7,
  "A": 8,
  "B": 9,
  "X": 10,
  "Y": 11,
  "up": 12,
  "right": 13,
  "down": 14,
  "left": 15,
}


class SensorFrame(Protocol):
  """The four channels a controller reads off a state frame.

  A structural type, so both :class:`IOFrame` (which carries more) and a
  plain test frame satisfy it without anyone converting between them.
  """

  joint_pos: np.ndarray  # (29,) rad, absolute encoder angles
  joint_vel: np.ndarray  # (29,) rad/s
  quat_wxyz: np.ndarray  # (4,) pelvis IMU orientation, world frame
  gyro: np.ndarray  # (3,) pelvis angular velocity, body frame, rad/s


@dataclass
class IOFrame:
  """One LowState read. ``tick`` is the firmware counter (staleness signal);
  ``buttons`` is the raw 16-bit field (levels; edge detection is the
  runner's job)."""

  joint_pos: np.ndarray  # (29,)
  joint_vel: np.ndarray  # (29,)
  quat_wxyz: np.ndarray  # (4,) pelvis IMU
  gyro: np.ndarray  # (3,) body frame
  tau_est: np.ndarray  # (29,)
  tick: int
  buttons: int
  # Per-motor REPORTED mode (29,), not the enable flag you send in LowCmd.
  # Tracks whether a service is attached to the low-level loop (0 once
  # released), NOT what that service is doing: it reads 1 in both zero-torque
  # and damping.
  motor_mode: np.ndarray


def pressed(buttons: int, name: str) -> bool:
  return bool((buttons >> BUTTON_BITS[name]) & 1)


class EdgeDetector:
  """Rising-edge detection over the level-valued button field."""

  def __init__(self) -> None:
    self._prev = 0

  def rising(self, buttons: int, name: str) -> bool:
    bit = 1 << BUTTON_BITS[name]
    edge = bool(buttons & bit) and not bool(self._prev & bit)
    return edge

  def update(self, buttons: int) -> None:
    self._prev = buttons


class DummyIO:
  """No robot, no DDS: a plausible standing state whose buttons follow a
  scripted timeline, so the runner and a controller smoke-test unattended
  (the no-DDS tier)."""

  def __init__(
    self,
    default_pose: np.ndarray,
    script: tuple[tuple[float, tuple[str, ...]], ...] = (
      (1.0, ("L1", "A")),  # ramp (4 s lerp + hold)
      (6.5, ("R1", "A")),  # engage, safely after the ramp completes
    ),
  ) -> None:
    self._default = default_pose.astype(np.float32)
    self._script = script
    self._t0 = time.monotonic()
    self._tick = 0

  def start(self) -> None:
    self._t0 = time.monotonic()

  def read(self) -> IOFrame:
    self._tick += 1
    t = time.monotonic() - self._t0
    buttons = 0
    for at, names in self._script:
      if at <= t <= at + 0.3:  # hold each press ~15 ticks
        for n in names:
          buttons |= 1 << BUTTON_BITS[n]
    return IOFrame(
      joint_pos=self._default.copy(),
      joint_vel=np.zeros(NUM_JOINTS, dtype=np.float32),
      quat_wxyz=np.array([1.0, 0, 0, 0], dtype=np.float32),
      gyro=np.zeros(3, dtype=np.float32),
      tau_est=np.zeros(NUM_JOINTS, dtype=np.float32),
      tick=self._tick,
      buttons=buttons,
      motor_mode=np.ones(NUM_JOINTS, dtype=np.uint8),
    )

  def write_targets(self, q: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
    pass

  def damp(self, q_measured: np.ndarray) -> None:
    self.write_targets(q_measured, np.zeros(NUM_JOINTS), np.full(NUM_JOINTS, DAMP_KD))

  def stop(self) -> None:
    pass


@dataclass
class UnitreeBindCfg:
  net_if: str = ""  # robot: the wired NIC. Empty lets CycloneDDS pick: DON'T.
  domain_id: int = 0
  release_motion: bool = True  # hand low-level control over on start()
  wire_hz: float = 200.0  # writer-thread rehold rate


class UnitreeBindIO:
  """unitree-sdk2-bind (nanobind, abi3) comms.

  The binding links the C++ SDK's own CycloneDDS 0.10.2 and has no Python
  `cyclonedds` dependency, so it installs and runs on 3.13+ where
  unitree_sdk2py's 11.0.1 wheel mismatches the G1's 0.10.x wire format.

  Measured on a real G1: LowState arrives 1 ms after subscribe at
  999.7 Hz, and `q` is 35 wide with the G1's 29 in the first slots.

  Command path: `write_targets` stores the LowCmd in a buffer AND publishes it
  inline (so a policy tick is not delayed by up to one wire period), while a
  daemon thread re-publishes the buffered command at `wire_hz`. That rehold is
  what makes damp a STATE rather than a one-shot, and what keeps the robot
  commanded when the control loop hiccups. Publishing only from the loop
  would leave the robot uncommanded for the length of any stall.
  """

  def __init__(self, cfg: UnitreeBindCfg) -> None:
    import unitree_sdk2_bind as u  # from the deploy extra; ships py.typed

    if u.is_stub():
      raise RuntimeError(
        "unitree_sdk2_bind is a STUB build: it fakes a healthy robot "
        "(publishers drop writes, subscribers synthesize upright zeros). "
        "Rebuild against the real unitree_sdk2 before deploying."
      )
    if not cfg.net_if:
      raise ValueError(
        "net_if is required: with an empty interface CycloneDDS picks one by "
        "its own heuristic, which may not be the robot's NIC."
      )
    self._u = u
    self._cfg = cfg
    u.init_channel(cfg.domain_id, cfg.net_if)
    self._msc = u.MotionSwitcherClient()
    self._msc.init(5.0)
    self._sub = u.LowStateSubscriber("rt/lowstate")
    self._pub = u.LowCmdPublisher("rt/lowcmd")
    self._loco = None  # LocoStateClient, built lazily by fsm_state()
    self._setup = None  # AiModeSetupClient, built lazily by bringup()
    self._mode_machine: int | None = None
    self._zeros = np.zeros(NUM_JOINTS, dtype=np.float32)
    # One lock guards both the buffer and the publisher: the wire thread and
    # the control loop must not write the same DDS writer concurrently.
    self._lock = threading.Lock()
    self._latest_cmd = None
    self._running = False
    self._wire_thread: threading.Thread | None = None
    self._wire_dt = 1.0 / cfg.wire_hz

  def start(self) -> None:
    if self._cfg.release_motion:
      # Loop until check_mode reports no service: one call is not guaranteed.
      _form, name = self._msc.check_mode()
      for _ in range(10):
        if not name:
          break
        self._msc.release_mode()
        time.sleep(1.0)
        _form, name = self._msc.check_mode()
      if name:
        raise RuntimeError(f"could not release motion service, still {name!r}")

    self._sub.start(10)
    deadline = time.monotonic() + 5.0
    state = self._sub.latest()
    while state is None:
      if time.monotonic() > deadline:
        raise RuntimeError(
          "no LowState within 5s. Check net_if has an IPv4 address on the "
          "robot subnet (CycloneDDS calls an interface 'not available' when "
          "it is UP but unaddressed), and that the robot is powered."
        )
      time.sleep(0.01)
      state = self._sub.latest()
    # Learn mode_machine from the robot; a wrong value makes it reject commands.
    self._mode_machine = int(state.mode_machine)

    self._running = True
    self._wire_thread = threading.Thread(
      target=self._wire_loop, name="lowcmd_wire", daemon=True
    )
    self._wire_thread.start()

  def _wire_loop(self) -> None:
    """Re-publish the buffered command at wire_hz until process exit."""
    while self._running:
      with self._lock:
        cmd = self._latest_cmd
        if cmd is not None:
          self._pub.write(cmd)
      time.sleep(self._wire_dt)

  def read(self) -> IOFrame:
    state = self._sub.latest()
    if state is None:
      raise RuntimeError("LowState stream dropped")
    remote = bytes(state.wireless_remote)
    return IOFrame(
      joint_pos=np.asarray(state.q, dtype=np.float32)[:NUM_JOINTS],
      joint_vel=np.asarray(state.dq, dtype=np.float32)[:NUM_JOINTS],
      # Binding exposes the WIRE order (wxyz), which is what IOFrame wants.
      quat_wxyz=np.asarray(state.quaternion, dtype=np.float32),
      gyro=np.asarray(state.gyroscope, dtype=np.float32),
      tau_est=np.asarray(state.tau_est, dtype=np.float32)[:NUM_JOINTS],
      tick=int(state.tick),
      buttons=int(remote[2] | (remote[3] << 8)),
      motor_mode=np.asarray(state.motor_mode)[:NUM_JOINTS],
    )

  def write_targets(self, q: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
    if self._mode_machine is None:
      raise RuntimeError("write before start(): mode_machine not learned yet")
    cmd = self._u.LowCmd()
    cmd.mode_pr = 0
    cmd.mode_machine = self._mode_machine
    cmd.set_motors(
      np.asarray(q, dtype=np.float32),
      self._zeros,
      np.asarray(kp, dtype=np.float32),
      np.asarray(kd, dtype=np.float32),
      self._zeros,
      mode=1,
      num_motors=NUM_JOINTS,
    )
    # Buffer it for the wire thread, and send now so this tick is not delayed
    # by up to one wire period.
    with self._lock:
      self._latest_cmd = cmd
      self._pub.write(cmd)  # CRC filled by the publisher, computed last

  def damp(self, q_measured: np.ndarray) -> None:
    self.write_targets(q_measured, np.zeros(NUM_JOINTS), np.full(NUM_JOINTS, DAMP_KD))

  def stop(self) -> None:
    # Deliberately does NOT tear down: the wire thread keeps re-holding the
    # last command (the runner leaves a damp there) until the process exits.
    # Closing the endpoints would end the stream instead, leaving the robot
    # uncommanded. Use close() to actually release the DDS endpoints.
    pass

  def close(self) -> None:
    """Full teardown: stop the wire thread and release the DDS endpoints."""
    self._running = False
    if self._wire_thread is not None:
      self._wire_thread.join(timeout=1.0)
      self._wire_thread = None
    self._sub.close()
    self._pub.close()

  # --------------------------------------------------------------- recovery --

  @property
  def mode_machine(self) -> int | None:
    """The firmware's hardware handshake id, learned from the first LowState."""
    return self._mode_machine

  def check_mode(self) -> str:
    """Name of the high-level service holding the robot; '' if released."""
    _form, name = self._msc.check_mode()
    return name

  def fsm_state(self) -> tuple[int | None, str]:
    """(fsm_id, name) of the loco service's state machine.

    Returns (None, reason) when no service is loaded: the RPC fails with
    rc 3104 after release_mode(), which is exactly while a policy runs.
    """
    if self._loco is None:
      self._loco = self._u.LocoStateClient()
      self._loco.init(5.0)
    try:
      fsm_id = int(self._loco.fsm_id())
    except RuntimeError as exc:
      return None, str(exc)
    return fsm_id, self._u.FSM_IDS.get(fsm_id, "unknown")

  def bringup(
    self,
    stand: bool = True,
    min_dwell_s: float = 5.0,
    settle_s: float = 30.0,
    quiet_dq: float = 0.05,
    auto_restore: bool = True,
    restore_mode: str = "ai",
    loco_ready_s: float = 10.0,
    log=None,
  ):
    """Stand the robot up under the loaded service. MOVES THE ROBOT.

    The software equivalent of the remote's L2+B -> L2+UP -> R1+X, driving
    the loco FSM 1 (damp) -> 4 (stand_up) -> 500 (start). `stand=False` sends
    only the first, the L2+B equivalent.

    Handles both entry states. Freshly powered on the service is already
    loaded (measured: `ai`, fsm 0 zero_torque) and this proceeds directly;
    after a deploy run the robot is released and `auto_restore` reloads the
    service first, since every FSM call fails rc 3104 while nothing is
    loaded. Pass auto_restore=False to require the caller to restore.

    Before the FIRST step, waits up to `loco_ready_s` for the loco API to
    answer at all: `check_mode` goes green when the service is LOADED, which
    is earlier than its loco API accepts commands, and that gap is entered
    from a reboot just as readily as from a restore.

    Each step waits for the FSM id, a SUSTAINED quiet period (1 s of samples
    under quiet_dq), and `min_dwell_s` elapsed, before the next is sent.
    All three are needed: the id flips as soon as the command is accepted,
    and dq is momentarily ~0 right after that because the motion has not
    begun, so either check alone reports "settled" too early. Sending the
    next step early is silently ignored by the service, rc 0 and all.

    The remote stays the operator's independent damp path. This automates
    bring-up, it does not replace the emergency stop.

    `log` is called with progress lines as they happen (pass `print`). The
    sequence takes ~25 s of visible robot motion, so silence for that long is
    unhelpful to whoever is standing next to the hardware.

    Returns the list of (label, fsm_id, fsm_name) actually observed.
    """

    def emit(msg: str) -> None:
      if log is not None:
        log(msg)

    # Two entry states, both normal: freshly powered on (service already
    # loaded, fsm 0) or released by a previous deploy run (no service, every
    # FSM call would fail rc 3104). Restore first in the second case so one
    # command covers both.
    mode = self.check_mode()
    if not mode:
      if not auto_restore:
        raise RuntimeError(
          "no high-level service loaded (check_mode is empty): run restore "
          "first, or the FSM calls will fail with rc 3104."
        )
      emit(f"released: no service loaded, restoring {restore_mode!r} first")
      res = self.restore_ai(restore_mode)
      mode = res["mode"]
      if not mode:
        raise RuntimeError(
          f"restore of {restore_mode!r} failed; power-cycle the robot, which "
          "is the other measured-good recovery path."
        )
      emit(f"restored: {mode!r} at fsm {res['fsm_id']} ({res['fsm_name']})")
    if self._setup is None:
      self._setup = self._u.AiModeSetupClient()
      self._setup.init(5.0)

    # `stand=False` stops after the damp step: the firmware damp the remote's
    # L2+B gives, which is what a hang-and-inspect wants. It is the SERVICE's
    # damp, not our low-level kp0/kd8 LowCmd: while a high-level service holds
    # the robot, a LowCmd is fought at the wire by the service's own stream,
    # so the only damp that lands here is fsm 1.
    steps = (("damp", self._setup.damp, 1),)
    if stand:
      steps += (
        ("stand_up", self._setup.stand_up, 4),
        ("start", self._setup.start, 500),
      )
    # The loco API answers LATER than check_mode goes green, and an FSM
    # command sent into that window is accepted, returns rc 0, and is then
    # dropped: the joints twitch audibly and the robot does not stand, so the
    # operator runs bringup twice. restore_ai waits this out, but ONLY the
    # released entry path reaches it, and a robot that booted straight into
    # 'ai' would skip the wait entirely. Gate here, where every entry path
    # passes.
    here = self._wait_loco_ready(loco_ready_s, emit)
    emit(f"service={mode!r}, starting from fsm {here[0]} ({here[1]})")
    emit(f"gates per step: fsm id, {min_dwell_s:g}s dwell, 1s sustained quiet")

    observed = []
    for n, (label, call, want) in enumerate(steps, 1):
      t_step = time.monotonic()
      rc = call()
      emit(f"[{n}/{len(steps)}] {label}: SetFsmId({want}) rc={rc}  ROBOT MOVING")
      # Two traps, both measured on a real G1:
      #  1. SetFsmId flips the reported id the instant it is ACCEPTED, while
      #     the motion takes seconds. Waiting on the id alone fires the next
      #     step mid-motion; the service then IGNORES it and still returns 0.
      #  2. Right after the id flips the robot has not started moving yet, so
      #     dq is momentarily ~0. A single "is it quiet" sample therefore
      #     reads settled BEFORE the motion, not after.
      # So: hold for min_dwell_s (past motion onset), then require the robot
      # to stay quiet for a sustained run of samples, not just one.
      floor = time.monotonic() + min_dwell_s
      deadline = time.monotonic() + settle_s
      need_quiet = 4  # consecutive samples, 0.25 s apart -> 1 s of stillness
      quiet_run = 0
      fsm_id: int | None = None
      fsm_name = "never polled"
      last_note = 0.0
      while time.monotonic() < deadline:
        time.sleep(0.25)
        fsm_id, fsm_name = self.fsm_state()
        if fsm_id is None:
          # rc 3104 mid-transition is normal; only fatal if the service is
          # genuinely gone, which check_mode can tell us and the FSM cannot.
          if not self.check_mode():
            raise RuntimeError(f"{label}: high-level service disappeared")
          emit(f"      fsm busy ({fsm_name}), normal mid-transition")
          continue
        moving = float(np.abs(self.read().joint_vel).max())
        quiet_run = quiet_run + 1 if moving < quiet_dq else 0
        elapsed = time.monotonic() - t_step
        if elapsed - last_note >= 1.0:  # throttle: one line/sec, not 4
          last_note = elapsed
          emit(
            f"      {elapsed:4.1f}s fsm={fsm_id} ({fsm_name})"
            f" max|dq|={moving:.3f} quiet={quiet_run}/{need_quiet}"
          )
        if fsm_id == want and quiet_run >= need_quiet and time.monotonic() >= floor:
          break
      observed.append((label, fsm_id, fsm_name))
      if rc != 0:
        raise RuntimeError(f"{label}: SetFsmId returned rc={rc}")
      emit(
        f"      settled at fsm {fsm_id} ({fsm_name}) after "
        f"{time.monotonic() - t_step:.1f}s"
      )
      if fsm_id != want:
        raise RuntimeError(
          f"{label}: FSM did not reach {want}, stuck at {fsm_id} ({fsm_name}). "
          "Stopping rather than sending the next step."
        )
    return observed

  def _wait_loco_ready(self, timeout_s: float, emit) -> tuple[int, str]:
    """Block until the loco API answers with a real fsm id.

    `check_mode` reports the service the moment it is LOADED; its loco API
    starts answering a beat later, and in between `fsm_id()` fails with
    rc 7301, which is NOT the rc 3104 that means nothing is loaded at all.
    Both surface here as a None id, so poll rather than read the code: what
    matters is only whether a command sent now would land or be dropped.
    """
    deadline = time.monotonic() + timeout_s
    noted = False
    while True:
      fsm_id, fsm_name = self.fsm_state()
      if fsm_id is not None:
        return fsm_id, fsm_name
      if time.monotonic() >= deadline:
        raise RuntimeError(
          f"loco API still silent after {timeout_s:g}s ({fsm_name}); refusing "
          "to send FSM commands it would accept with rc 0 and then drop."
        )
      if not noted:
        noted = True
        emit(f"      loco API not up yet ({fsm_name}), waiting")
      time.sleep(0.25)

  def restore_ai(self, name: str = "ai", damp_s: float = 1.0, timeout_s: float = 15.0):
    """Hand the robot back to its onboard service after a deploy run.

    Returns {"mode", "fsm_id", "fsm_name"} describing where the robot landed.
    A "mode" of '' means the select did not take.

    Measured on a real G1: select_mode("ai") lands at fsm_id 0 (zero_torque),
    the same state a power cycle produces. The service does NOT take hold of
    the joints, so this leaves the robot slack; use L2+B then L2+UP on the
    remote to stand it. Keep the robot supported: restoring the service does
    not make it hold itself up.

    Damps first for `damp_s` so the joints are slack rather than held at
    policy gains when the service arrives, otherwise the two controllers
    would fight over the same joints during the switch.

    A power cycle also restores the service and is the
    hardware-proven path. Prefer it when the robot is free-standing; this
    exists for when a reboot is not wanted.
    """
    state = self._sub.latest()
    if state is not None:
      deadline = time.monotonic() + damp_s
      while time.monotonic() < deadline:
        current = self._sub.latest()
        if current is not None:
          self.damp(np.asarray(current.q, dtype=np.float32)[:NUM_JOINTS])
        time.sleep(0.02)

    self._msc.select_mode(name)
    deadline = time.monotonic() + timeout_s
    got = ""
    while time.monotonic() < deadline:
      _form, got = self._msc.check_mode()
      if got:
        break
      time.sleep(0.5)
    # check_mode reports the service loaded BEFORE its loco API answers: an
    # immediate fsm read returns rc 7301 (distinct from the rc 3104 you get
    # when nothing is loaded at all). Retry rather than report a false None.
    fsm_id, fsm_name = None, "not polled"
    fsm_deadline = time.monotonic() + 10.0
    while time.monotonic() < fsm_deadline:
      fsm_id, fsm_name = self.fsm_state()
      if fsm_id is not None:
        break
      time.sleep(0.5)
    return {"mode": got, "fsm_id": fsm_id, "fsm_name": fsm_name}
