#!/usr/bin/env python3
"""Confirm the G1's wireless remote reads correctly, before anything can move.

    .venv/bin/python3 scripts/check_g1_remote.py --net-if <nic>

Read-only, and provably so: it constructs the comms with ``release_motion=
False``, so the onboard service keeps holding the robot, and it never calls
``write_targets`` or ``damp``, so the command buffer stays empty and the wire
thread publishes nothing at all. Safe to run with the robot standing, sitting,
or hung.

The whole operator contract rides on one 16-bit field arriving in LowState and
decoding the way the runner expects: B damps from any state, L1+A ramps to the
pre-engage pose, R1+A starts the episode, Y stops back to the ramp. If that field is
misread, the operator has no software controls at all, so it gets confirmed
first, on its own, with no policy artifact anywhere near it.

Press in this order, safest first, so every abort path is confirmed before
anything that could start motion:

1. B          the damp e-stop. If only one button ever works, this is the one.
2. Y          the other stop.
3. L1+A       first combination that commands motion. HOLD L1 first, then
              tap A: the modifier has to be down before A's press, which is
              the convention the G1 remote itself follows for combinations.
4. R1+A       starts the policy. Hold R1 first, then tap A. Last, because in
              a real run this is the one that triggers the trained squat.
5. up, down   the stiffness step (deploy_solo_stand.py --stiffness-step).
6. A alone, then L1 alone, then R1 alone: each must appear under `held` and
   fire NOTHING. A firing by itself is the decode error that actually hurts.
7. Hold any combination down: it must fire once, not repeat. The runner acts
   on rising edges only.

Exits non-zero and names what is missing if any of the six never fired, so a
clean exit is the pass condition.

Launch with the venv python, not ``uv run``: this one only reads, but keeping
one launch habit across every hardware script is what keeps the damp hold
intact in the ones that command.
"""

from __future__ import annotations

import argparse
import sys
import time

from mjlab.tasks.codancing.simple_deploy import constants as ctl
from mjlab.tasks.codancing.simple_deploy.io import (
  BUTTON_BITS,
  EdgeDetector,
  UnitreeBindCfg,
  UnitreeBindIO,
  pressed,
)

# The combinations the runner acts on, in the order an operator uses them.
# Checked below with the same predicates loop.py uses, so confirming them here
# confirms the real thing rather than a copy of it.
REMOTE_ACTIONS: tuple[tuple[str, tuple[str, ...], str], ...] = (
  ("B", ("B",), "DAMP, from any state, terminal"),
  ("Y", ("Y",), "operator stop, back to the ramp"),
  ("L1+A", ("L1", "A"), "ramp to the pre-engage pose at holding gains"),
  ("R1+A", ("R1", "A"), "start the episode"),
  ("up", ("up",), "stiffness +step on both wrists (--stiffness-step)"),
  ("down", ("down",), "stiffness -step on both wrists (--stiffness-step)"),
)
MEANING = {label: meaning for label, _bits, meaning in REMOTE_ACTIONS}

# Held first, then the action button is tapped: the G1 remote's own convention,
# which declares the modifier set outright ("first button in combination, need
# to be held down to trigger other commands"). The runner inherits the
# requirement from the
# shape of its predicate, `rising(A) and pressed(L1)`: the edge is on A, so a
# modifier arriving after A finds the edge already spent.
MODIFIERS = ("L1", "R1")


def check_remote(net_if: str, seconds: float) -> int:
  problem = ctl.net_if_problem(net_if)
  if problem:
    raise SystemExit(f"--net-if {net_if}: {problem}")
  io = UnitreeBindIO(UnitreeBindCfg(net_if=net_if, release_motion=False))
  io.start()
  print(f"remote readout on {net_if}: read-only, the robot is not released")
  print(f"switcher : check_mode={io.check_mode()!r}  fsm={io.fsm_state()}")
  print("\npress these IN ORDER, safest first, and check what comes back:")
  for n, (label, bits, meaning) in enumerate(REMOTE_ACTIONS, 1):
    how = f"  (hold {bits[0]} FIRST, then tap {bits[-1]})" if len(bits) > 1 else ""
    print(f"    {n}. {label:6s} -> {meaning}{how}")
  print(
    "\n    7. A alone, then L1 alone, then R1 alone: each must show up under\n"
    "       'held' and fire NOTHING. A firing on its own would start the\n"
    "       policy unasked, which is the one decode error that matters most.\n"
    "    8. hold any combination down: it must fire once, not repeat."
  )
  print("\nwaiting for the remote; press Ctrl-C when you are done\n")

  # A live line that keeps refreshing, so you can watch a press arrive and see
  # it stay while you hold it, plus a scrolling log above it for the things you
  # must not miss: a change in what is held, and an action firing. Redirected
  # to a file or a pipe the refresh would just concatenate into one unreadable
  # line, so there it falls back to the event log alone.
  width = 118
  interactive = sys.stdout.isatty()

  def event(line: str) -> None:
    print(f"\r{' ' * width}\r{line}" if interactive else line, flush=True)

  edges = EdgeDetector()
  seen: set[str] = set()
  previous = -1
  ticks: set[int] = set()
  t0 = time.monotonic()
  next_live = 0.0
  hints = 0
  try:
    while seconds <= 0 or time.monotonic() - t0 < seconds:
      now = time.monotonic()
      frame = io.read()
      ticks.add(frame.tick)
      # Rising edges must be sampled before the level state is updated, the
      # same order the runner uses.
      fired = [
        label
        for label, bits, _ in REMOTE_ACTIONS
        if edges.rising(frame.buttons, bits[-1])
        and all(pressed(frame.buttons, b) for b in bits[:-1])
      ]
      a_rising = edges.rising(frame.buttons, "A")
      edges.update(frame.buttons)

      held = [name for name in BUTTON_BITS if pressed(frame.buttons, name)]
      if frame.buttons != previous:
        previous = frame.buttons
        event(f"  0x{frame.buttons:04x}  held: {', '.join(held) if held else '(none)'}")
      for label in fired:
        seen.add(label)
        event(f"      -> {label} FIRED: {MEANING[label]}")

      # A tapped with no modifier down spends its rising edge on a frame where
      # no combination can match, and A stays high afterwards, so bringing L1
      # or R1 in late produces no second edge and nothing fires. Silence is
      # what makes that read as a broken remote, so say it out loud.
      if a_rising and not any(pressed(frame.buttons, m) for m in MODIFIERS):
        hints += 1
        if hints <= 3:
          event(
            "      .. A pressed with no modifier held, so nothing can fire."
            " Hold L1 or R1 FIRST, then tap A."
          )

      if interactive and now >= next_live and now - t0 > 0.5:
        next_live = now + 0.05  # 20 Hz refresh, well under the wire rate
        done = " ".join(
          f"{label}{'+' if label in seen else '-'}"
          for label, _bits, _m in REMOTE_ACTIONS
        )
        live = (
          f"  0x{frame.buttons:04x}  {len(ticks) / (now - t0):4.0f} Hz  "
          f"held: {(', '.join(held) if held else '(none)'):<26}  "
          f"confirmed {len(seen)}/{len(REMOTE_ACTIONS)}  {done}"
        )
        print(f"\r{live[:width]:<{width}}", end="", flush=True)
      time.sleep(0.005)
  except KeyboardInterrupt:
    pass
  finally:
    elapsed = time.monotonic() - t0
    if interactive:
      print(f"\r{' ' * width}\r", end="")
    io.close()  # read-only probe: release the endpoints, do not re-hold

  print(f"\nLowState rate: {len(ticks) / max(elapsed, 1e-9):.0f} Hz")
  if not ticks:
    print("NO LowState arrived: the remote cannot be read at all")
    return 1
  missing = [label for label, _b, _m in REMOTE_ACTIONS if label not in seen]
  if missing:
    print(f"NOT confirmed: {', '.join(missing)}")
    print("Do not run a policy until every combination above has fired here.")
    return 1
  print("every runner combination fired correctly")
  return 0


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Read the G1 wireless remote and nothing else."
  )
  parser.add_argument(
    "--net-if",
    required=True,
    help="NIC the robot is on (`ip -br link` lists them)",
  )
  parser.add_argument(
    "--seconds", type=float, default=0.0, help="0 (default) runs until Ctrl-C"
  )
  args = parser.parse_args()
  return check_remote(args.net_if, args.seconds)


if __name__ == "__main__":
  raise SystemExit(main())
