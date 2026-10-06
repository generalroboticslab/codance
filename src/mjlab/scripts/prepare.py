"""`prepare` -- the offline motion-data preparation CLI (the inspect/prepare/run split).

`run` consumes prepared data only; everything that CREATES or EDITS that data
lives here, as one verb with subcommands:

  prepare csv        -- replay a CSV motion and bake an npz
  prepare reverse    -- emit the TIME-reverse of a clip npz (frame flip +
                        recomputed velocities)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Union, cast

import tyro

import mjlab


@dataclass(frozen=True)
class Csv:
  """Replay a CSV motion and bake an npz (delegates to ``csv_to_npz``)."""

  input_file: str
  output_name: str
  input_fps: float = 30.0
  output_fps: float = 50.0
  device: str = "cuda:0"
  render: bool = False
  line_range: tuple[int, int] | None = None

  def run(self) -> None:
    from mjlab.scripts.csv_to_npz import main

    main(
      input_file=self.input_file,
      output_name=self.output_name,
      input_fps=self.input_fps,
      output_fps=self.output_fps,
      device=self.device,
      render=self.render,
      line_range=self.line_range,
    )


@dataclass(frozen=True)
class Reverse:
  """Emit the TIME-reverse of a clip npz.

  Frame order flips and the velocity channels are RECOMPUTED from the reversed
  positions (never sign-flipped) -- backward playback with consistent dynamics.
  """

  input_file: str
  output_file: str

  def run(self) -> None:
    import numpy as np

    from mjlab.tasks.shared.mdp.reference_motion_editing import (
      _load_motion_npz,
      _recompute_derived_kinematics_in_place,
      _reverse_motion_in_place,
    )

    motion = _load_motion_npz(self.input_file)
    _reverse_motion_in_place(motion)
    _recompute_derived_kinematics_in_place(motion)
    out = Path(self.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **cast(dict, motion.arrays))
    print(f"[INFO] time-reversed {self.input_file} -> {out}")


Command = Union[
  Annotated[Csv, tyro.conf.subcommand("csv")],
  Annotated[Reverse, tyro.conf.subcommand("reverse")],
]


def main() -> None:
  cmd = tyro.cli(Command, config=mjlab.TYRO_FLAGS)  # type: ignore[var-annotated]
  cmd.run()


if __name__ == "__main__":
  main()
