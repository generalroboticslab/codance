"""Deploy abort rules on OBSERVABLE signals only (never the training
termination thresholds, which were deliberately loose and partly
unobservable). Any trip routes the runner into its terminal DAMP state."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from mjlab.tasks.codancing.simple_deploy import constants as ctl
from mjlab.tasks.codancing.simple_deploy.io import IOFrame


@dataclass
class SafetyCfg:
  tilt_limit_rad: float = 1.0  # absolute IMU tilt (gravity-z angle)
  quat_norm_tol: float = 0.05
  joint_vel_limit: float = 20.0  # rad/s, any joint
  # Staleness is a TIME condition, not a read-count one. Counting reads
  # conflates "the robot stopped talking" with "we read faster than it talks":
  # LowState arrives at ~1 kHz, so a burst of reads inside one millisecond
  # legitimately returns the same frame every time and says nothing about
  # health. 0.2 s of an unchanged tick is ~200 missed frames, unambiguous.
  stale_timeout_s: float = 0.2
  clamp_streak_limit: int = 15  # consecutive ticks with any joint clamped
  overrun_streak_limit: int = 10  # consecutive ticks over budget
  tick_budget_s: float = 0.020
  # Joint-space deviation vs the FREE reference (armed at engage only). One
  # band PER JOINT: a single number cannot describe a robot whose joints
  # differ tenfold in travel, and the wrists are exempt outright. See
  # constants.JOINT_DEV_BAND for how the numbers were sized and why.
  joint_dev_band: np.ndarray = field(default_factory=lambda: ctl.JOINT_DEV_BAND.copy())
  joint_dev_joints: int = 3  # trip only if this many joints deviate at once

  def disarm_joint_dev(self) -> None:
    """Turn the deviation trip off for this run.

    The dummy tier needs it (a scripted plant never tracks the reference, so
    the trip fires seconds after START on any moving clip) and the operator
    opt-out for contact-heavy demos needs it. Nothing else should.
    """
    self.joint_dev_band = np.full(ctl.NUM_JOINTS, np.inf)

  def disarm_joint_trips(self) -> None:
    """Turn BOTH joint-space trips off: deviation, and the clamp streak.

    They always go together, because the two situations that need one need
    the other. A dummy plant never follows the reference AND leaves the
    policy railed against the joint limits within a second. A contact-heavy
    demo pulls a hand far enough to look like a deviation AND holds joints at
    their stops while it is held there.

    What stays armed: targets are still clamped to the joint limits every
    tick, and tilt, joint velocity, staleness and overrun are untouched.
    Those are what catch a real fall.
    """
    self.disarm_joint_dev()
    self.clamp_streak_limit = 10**9


class SafetyMonitor:
  def __init__(self, cfg: SafetyCfg) -> None:
    self.cfg = cfg
    self._last_tick: int | None = None
    self._last_tick_change: float | None = None
    self._clamp_streak = 0
    self._overrun_streak = 0

  def check_frame(self, frame: IOFrame) -> str | None:
    """Armed from BOOT: staleness, quat sanity, tilt, joint velocity."""
    cfg = self.cfg
    now = time.monotonic()
    last_change = self._last_tick_change
    if self._last_tick is None or frame.tick != self._last_tick:
      self._last_tick = frame.tick
      self._last_tick_change = now
    elif last_change is not None and now - last_change > cfg.stale_timeout_s:
      return f"LowState stale for {now - last_change:.2f}s (tick={frame.tick})"

    norm = float(np.linalg.norm(frame.quat_wxyz))
    if abs(norm - 1.0) > cfg.quat_norm_tol:
      return f"IMU quaternion norm {norm:.3f} out of tolerance"

    w, x, y, _z = (frame.quat_wxyz / norm).tolist()
    # Body-frame gravity z for a unit quaternion: -(1 - 2(x^2 + y^2)).
    gz = -(1.0 - 2.0 * (x * x + y * y))
    tilt = float(np.arccos(np.clip(-gz, -1.0, 1.0)))
    if tilt > cfg.tilt_limit_rad:
      return f"tilt {tilt:.2f} rad over limit {cfg.tilt_limit_rad}"

    vmax = float(np.abs(frame.joint_vel).max())
    if vmax > cfg.joint_vel_limit:
      return f"joint velocity {vmax:.1f} rad/s over limit"
    return None

  def check_clamp(self, clamped_joints: int) -> str | None:
    self._clamp_streak = self._clamp_streak + 1 if clamped_joints else 0
    if self._clamp_streak >= self.cfg.clamp_streak_limit:
      return f"joint-limit clamp active {self._clamp_streak} consecutive ticks"
    return None

  def check_overrun(self, tick_s: float) -> str | None:
    self._overrun_streak = (
      self._overrun_streak + 1 if tick_s > self.cfg.tick_budget_s else 0
    )
    if self._overrun_streak >= self.cfg.overrun_streak_limit:
      return (
        f"tick budget blown {self._overrun_streak} consecutive ticks "
        f"(last {tick_s * 1e3:.1f} ms)"
      )
    return None

  def check_joint_dev(
    self, joint_pos: np.ndarray, ref_joint_pos: np.ndarray | None
  ) -> str | None:
    """Armed at ENGAGE: live joints vs the free reference at the cursor."""
    if ref_joint_pos is None:
      return None
    band = self.cfg.joint_dev_band
    dev = np.abs(joint_pos - ref_joint_pos)
    n_over = int((dev > band).sum())
    if n_over >= self.cfg.joint_dev_joints:
      # Name the joint that is furthest PAST its own band, not the one with
      # the largest raw deviation: with per-joint bands those are routinely
      # different joints, and the raw maximum is usually an exempt wrist.
      worst = int(np.argmax(dev - band))
      return (
        f"{n_over} joints past their deviation band vs the reference "
        f"(worst {ctl.JOINT_NAMES[worst]} at {dev[worst]:.2f} rad, "
        f"band {band[worst]:.2f})"
      )
    return None
