"""The 50 Hz runner: BOOT -> DAMP_IDLE -> RAMP -> READY -> RUNNING with one
terminal DAMP state reachable from everywhere.

Safety shape: the terminal damp is a STATE that republishes damp
commands at loop rate and never returns to control; every exception,
signal, watchdog trip, or e-stop routes into it; a >= 1 s post-damp hold
precedes process exit (the robot latches the last LowCmd). A separate
daemon watchdog thread damps on control-loop heartbeat stall (a hung loop
otherwise leaves the last full-gain command latched). Software damp covers
live-link failures only; the firmware remote (L2+B) stays the operator's
independent layer, and hardware runs keep the gantry/hang.

Buttons (rising edges, polled in EVERY state): B = damp, from any state;
L1+A = ramp to the controller's pre-engage pose at STIFF holding gains; R1+A = start
the episode (one tick brings gains, policy, cursor and histories up
together); Y in RUNNING = operator stop back to the ramp/hold. Never blend
gains; never resume a stopped episode (restarting is a full start).
"""

from __future__ import annotations

import math
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mjlab.tasks.codancing.simple_deploy import constants as ctl
from mjlab.tasks.codancing.simple_deploy.io import (
  NUM_JOINTS,
  EdgeDetector,
  IOFrame,
)
from mjlab.tasks.codancing.simple_deploy.record import RunRecorder
from mjlab.tasks.codancing.simple_deploy.safety import SafetyCfg, SafetyMonitor

# Holding gains for the ramp and the READY hold (they hold the G1 on the
# gantry): legs kp 100/150 (knee), waist 200, arms 40. LoopCfg overrides them.
HOLD_KP = np.array(
  [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40, 200, 200, 200] + [40] * 14,
  dtype=np.float64,
)
HOLD_KD = np.array(
  [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 6, 6, 6] + [2] * 14, dtype=np.float64
)


@dataclass
class LoopCfg:
  dt: float = 0.02
  ramp_s: float = 4.0
  max_episode_s: float = 12.0
  clamp_margin_rad: float = 0.0
  hold_kp: np.ndarray = field(default_factory=lambda: HOLD_KP.copy())
  hold_kd: np.ndarray = field(default_factory=lambda: HOLD_KD.copy())
  safety: SafetyCfg = field(default_factory=SafetyCfg)
  # DummyIO tier: stop the process cleanly after this many seconds (<=0: run
  # until an OS signal). Real runs use 0.
  auto_exit_s: float = 0.0
  # up / down on the remote move both wrists' linear stiffness by this many
  # N/m (READY and RUNNING), for sweeping the compliance channel without a
  # restart; 0 leaves the buttons unmapped. Needs a controller with
  # ``nudge_stiffness`` (the stand controller).
  stiffness_step: float = 0.0


class Runner:
  def __init__(
    self, controller: Any, io: Any, cfg: LoopCfg, recorder: RunRecorder
  ) -> None:
    # Any object offering start/step/clamp_to_limits/kp/kd/default_joint_pos/
    # reference_joint_pos/max_episode_ticks: scripts/deploy_solo_stand.py hands
    # in a StandController, codance.py a CodanceController.
    self.controller = controller
    self.io = io
    self.cfg = cfg
    self.rec = recorder
    self.safety = SafetyMonitor(cfg.safety)
    self.edges = EdgeDetector()
    self.state = "DAMP_IDLE"
    self._damp_reason: str | None = None
    self._heartbeat = time.monotonic()
    self._last_q = np.zeros(NUM_JOINTS)
    self._ramp_start_q: np.ndarray | None = None
    self._ramp_t0 = 0.0
    self._episode_t0 = 0.0
    self._episode_ticks = 0
    self._run_t0 = time.monotonic()
    self._tick_t0 = self._run_t0
    self._tick_wall = time.time()
    self._next_readout = 0.0

  # ------------------------------------------------------------- lifecycle --

  def run(self) -> None:
    watchdog = threading.Thread(target=self._watchdog, daemon=True)
    watchdog.start()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
      try:
        signal.signal(sig, self._on_signal)
      except ValueError:
        pass  # non-main thread (tests)
    self.io.start()
    self.rec.event(f"runner up: state={self.state}, clip={self.controller.clip_name}")
    run_t0 = time.monotonic()  # fixed run start: auto-exit measures from here
    self._run_t0 = run_t0
    t0 = run_t0  # schedule base, re-based whenever the loop falls behind
    k = 0
    try:
      while True:
        tick_t0 = time.monotonic()
        self._heartbeat = tick_t0
        self._tick_t0 = tick_t0
        self._tick_wall = time.time()
        frame = self.io.read()
        self._last_q = frame.joint_pos.copy()
        self._step_state(frame, tick_t0)
        tick_cost = time.monotonic() - tick_t0
        if self.state in ("RUNNING",):
          reason = self.safety.check_overrun(tick_cost)
          if reason:
            self._to_damp(reason)
        self.edges.update(frame.buttons)
        if self.cfg.auto_exit_s > 0 and tick_t0 - run_t0 > self.cfg.auto_exit_s:
          self.rec.event("auto-exit (dummy comms)")
          break
        k += 1
        deadline = t0 + (k + 1) * self.cfg.dt
        sleep = deadline - time.monotonic()
        if sleep > 0:
          time.sleep(sleep)
        elif -sleep > self.cfg.dt:
          # Behind by more than a period. Do NOT sprint through the backlog to
          # catch up: those ticks would re-read a LowState that has not been
          # republished yet (so every rate-derived check sees frozen data) and
          # emit a command burst at whatever rate the CPU allows. The lost
          # ticks are worthless anyway, the robot has barely moved. Drop them
          # and restart the schedule from now.
          dropped = int(-sleep / self.cfg.dt)
          self.rec.event(
            f"schedule reset: {-sleep * 1e3:.0f} ms behind, dropped {dropped} ticks"
          )
          t0 = time.monotonic()
          k = 0
    except BaseException as exc:  # noqa: BLE001 -- every path must damp
      self._to_damp(f"exception: {type(exc).__name__}: {exc}")
      raise
    finally:
      self._final_damp_hold()
      self.rec.save()

  def _on_signal(self, signum, _frame) -> None:
    raise KeyboardInterrupt(f"signal {signum}")

  def _watchdog(self) -> None:
    """Daemon thread: damp if the control loop stalls while commanding."""
    while True:
      time.sleep(0.05)
      if self.state in ("RAMP", "READY", "RUNNING"):
        if time.monotonic() - self._heartbeat > 0.25:
          try:
            self.io.damp(self._last_q)
          except Exception:
            pass

  def _final_damp_hold(self) -> None:
    """>= 1 s of damp publishes before exit (the robot latches the last cmd)."""
    self.rec.event(f"final damp hold (reason: {self._damp_reason or 'exit'})")
    for _ in range(60):
      try:
        self.io.damp(self._last_q)
      except Exception:
        break
      time.sleep(self.cfg.dt)
    self.io.stop()

  # ----------------------------------------------------------------- states --

  def _to_damp(self, reason: str) -> None:
    if self.state != "DAMP":
      self.rec.event(f"DAMP: {reason}")
    self.state = "DAMP"
    self._damp_reason = reason

  def _step_state(self, frame: IOFrame, now: float) -> None:
    # E-stop first, from any state, rising edge.
    if self.edges.rising(frame.buttons, "B"):
      self._to_damp("operator B")
    reason = self.safety.check_frame(frame)
    if reason and self.state != "DAMP":
      self._to_damp(reason)

    if self.state == "DAMP":
      self.io.damp(frame.joint_pos)
      return

    if self.state == "DAMP_IDLE":
      self.io.damp(frame.joint_pos)
      if self.edges.rising(frame.buttons, "A") and _held(frame, "L1"):
        self._ramp_start_q = frame.joint_pos.astype(np.float64).copy()
        self._ramp_t0 = now
        self.state = "RAMP"
        self.rec.event("RAMP: lerp to the pre-engage pose at holding gains")
      return

    if self.state == "RAMP":
      alpha = min((now - self._ramp_t0) / self.cfg.ramp_s, 1.0)
      assert self._ramp_start_q is not None
      target = (
        1.0 - alpha
      ) * self._ramp_start_q + alpha * self.controller.default_joint_pos
      self.io.write_targets(target, self.cfg.hold_kp, self.cfg.hold_kd)
      if alpha >= 1.0:
        self.state = "READY"
        self.rec.event("READY: holding the pre-engage pose; R1+A starts the episode")
      return

    if self.state == "READY":
      self.io.write_targets(
        self.controller.default_joint_pos, self.cfg.hold_kp, self.cfg.hold_kd
      )
      self._stiffness_buttons(frame)
      if self.edges.rising(frame.buttons, "A") and _held(frame, "R1"):
        self.rec.event("START (single tick; expect the trained first-motion squat)")
        # Episode clock before the first publish: the readout timestamps
        # against it.
        self._episode_t0 = now
        self._next_readout = 0.0
        res = self.controller.start(frame)
        self._publish_policy(frame, res)
        self._episode_ticks = 0
        self.state = "RUNNING"
      return

    if self.state == "RUNNING":
      res = self.controller.step(frame)
      self._publish_policy(frame, res)
      self._episode_ticks += 1
      self._stiffness_buttons(frame)
      dev = self.safety.check_joint_dev(frame.joint_pos, self._ref_joints())
      if dev:
        self._to_damp(dev)
        return
      if self.edges.rising(frame.buttons, "Y"):
        self._back_to_ramp(frame, "operator Y (stop to hold)")
      elif res.reference_done:
        self._back_to_ramp(frame, "reference end")
      elif (
        self.controller.max_episode_ticks is not None
        and self._episode_ticks >= self.controller.max_episode_ticks
      ):
        self._back_to_ramp(frame, "max episode ticks")
      elif now - self._episode_t0 > self.cfg.max_episode_s:
        self._back_to_ramp(frame, "max episode wall clock")
      return

  def _stiffness_buttons(self, frame: IOFrame) -> None:
    """up / down: both wrists' linear stiffness by +-cfg.stiffness_step N/m.
    Rising edges, so a held button counts once; the change lands in the
    record as an event and in every later tick's observation."""
    step = self.cfg.stiffness_step
    nudge = getattr(self.controller, "nudge_stiffness", None)
    if step <= 0 or nudge is None:
      return
    if self.edges.rising(frame.buttons, "up"):
      delta = step
    elif self.edges.rising(frame.buttons, "down"):
      delta = -step
    else:
      return
    s = nudge(delta)
    msg = (
      f"stiffness: L {s.linear_left:g} N/m  R {s.linear_right:g} N/m "
      f"({'+' if delta > 0 else ''}{delta:g} on both wrists)"
    )
    self.rec.event(msg)
    print(msg, flush=True)

  def _back_to_ramp(self, frame: IOFrame, why: str) -> None:
    self.rec.event(f"episode over ({why}); ramp back to hold")
    self._ramp_start_q = frame.joint_pos.astype(np.float64).copy()
    self._ramp_t0 = time.monotonic()
    self.state = "RAMP"

  def _publish_policy(self, frame: IOFrame, res) -> None:
    clamped = self.controller.clamp_to_limits(res.target, self.cfg.clamp_margin_rad)
    n_clamped = int((~np.isclose(clamped, res.target)).sum())
    reason = self.safety.check_clamp(n_clamped)
    if reason:
      self._to_damp(reason)
      self.io.damp(frame.joint_pos)
      return
    self.io.write_targets(clamped, self.controller.kp, self.controller.kd)
    self.rec.tick(
      # Time base: wall clock (epoch s, float64 in the file), seconds since
      # the run started, and the IO layer's own tick counter.
      t_wall=self._tick_wall,
      t_run=self._tick_t0 - self._run_t0,
      io_tick=float(frame.tick),
      q=frame.joint_pos,
      dq=frame.joint_vel,
      quat=frame.quat_wxyz,
      gyro=frame.gyro,
      tau_est=frame.tau_est,
      raw_action=res.raw_action,
      target=res.target,
      target_sent=clamped,
      obs=res.obs,
      buttons=float(frame.buttons),
    )
    self._compliance_readout(frame, clamped)

  def _compliance_readout(self, frame: IOFrame, target: np.ndarray) -> None:
    """Live wrist load per contact slot, at 4 Hz while the policy runs.

    Torque on its own does not say whether the robot is complying: a stiff
    joint and a yielding one can read the same N.m at very different
    deflections. The yield is how far the arm actually fell behind the
    position it was commanded to hold, so the pair is what shows a push
    arriving and the policy going with it. Tilt is there to catch the robot
    being pushed over rather than yielding.
    """
    now = time.monotonic()
    if now < self._next_readout:
      return
    self._next_readout = now + 0.25
    parts = []
    for slot in ctl.CONTACT_SLOTS:
      wrist = list(ctl.WRIST_JOINTS[slot])
      arm = list(ctl.ARM_JOINTS[slot])
      tau = float(np.abs(np.asarray(frame.tau_est)[wrist]).max())
      yielded = float(np.abs(np.asarray(frame.joint_pos)[arm] - target[arm]).max())
      parts.append(f"{slot} |tau| {tau:5.2f} N.m  yield {yielded:.3f} rad")
    x, y = float(frame.quat_wxyz[1]), float(frame.quat_wxyz[2])
    tilt = math.degrees(math.acos(min(max(1.0 - 2.0 * (x * x + y * y), -1.0), 1.0)))
    print(
      f"[deploy] t={now - self._episode_t0:5.1f}s  {'  |  '.join(parts)}"
      f"  |  tilt {tilt:4.1f} deg",
      flush=True,
    )

  def _ref_joints(self) -> np.ndarray | None:
    try:
      return self.controller.reference_joint_pos()
    except Exception:
      return None


def _held(frame: IOFrame, name: str) -> bool:
  from mjlab.tasks.codancing.simple_deploy.io import pressed

  return pressed(frame.buttons, name)
