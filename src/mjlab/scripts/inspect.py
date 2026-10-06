"""`inspect` -- visualize a composed run with no side effects.

The inspect/prepare/run split: `inspect` looks (zero policy, viewer on, video
and metric recorders off), `prepare` creates data, `run` trains/evaluates on
prepared data. This verb is a thin alias over `run play` that pins the
no-output session knobs; every selection flag (`--config` / `--ckpt` /
`--overlay` / `--override`) passes through unchanged.
"""

from __future__ import annotations

import sys


def main() -> None:
  from mjlab.scripts.run import main as run_main

  sys.argv = [
    "run",
    "play",
    *sys.argv[1:],
    "--override",
    "session.agent_type=zero",
    "--override",
    "session.video.enabled=false",
    "--override",
    "session.metric.enabled=false",
  ]
  run_main()


if __name__ == "__main__":
  main()
