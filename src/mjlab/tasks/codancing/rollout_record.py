"""The per-step rollout recorder: the per-env rollout table.

:class:`EnvRowsRecorder` writes the long per-(step, env) table the offline
analyzer reads. Every driver (``run_bounded_rollout`` and each viewer's
``_execute_step``) runs it right after ``env.step()``. Rows are written and
flushed as they are built, so a killed rollout keeps every step it reached.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from mjlab.tasks.codancing.mdp.commands import get_codancing_command


def _num(v: Any) -> str:
  if v is None:
    return ""
  f = float(v)
  return "" if f != f else f"{f:.9g}"  # NaN -> empty cell


class EnvRowsRecorder:
  """Long-format per-(step, env) fact table.

  One row per env per step: reset flag, active clip + cursor time + clip
  length (the offline phase-binning pair), this step's per-term termination
  dones and every ``command.metrics`` value.
  The offline analyzer is the consumer: settle masking, per-clip tables, cause
  splits and coverage checks all derive from these columns, so the runtime
  stays a recorder. Rows are written and flushed per step, so a killed rollout
  keeps every step it reached.
  """

  def __init__(self, *, out_path: Path, command_name: str = "codancing") -> None:
    self._path = Path(out_path)
    self._path.parent.mkdir(parents=True, exist_ok=True)
    self._command_name = command_name
    self._step = 0
    self._file: Any = None
    self._writer: Any = None
    self._header: list[str] | None = None

  @property
  def path(self) -> Path:
    return self._path

  def on_step(self, env: Any) -> None:
    u = env.unwrapped
    cmd = get_codancing_command(u, self._command_name)
    num_envs = int(u.num_envs)
    reset = u.reset_buf.bool().tolist()
    clips = cmd.active_clip_names
    time_s = cmd._motion_time.tolist()
    # Phase denominator for the offline per-phase binning: the active clip's
    # length in seconds, from the same clock `_update_command` checks clip
    # ends against (the human clip when present, else the robot reference).
    if cmd._human_registry is not None:
      frames = cmd._human_registry.motion_num_frames[cmd._motion_id].float()
    else:
      frames = cmd._reference_provider.num_frames(cmd._motion_id)
    clip_len_s = (frames * cmd._control_dt).tolist()
    term_names = tuple(u.termination_manager.active_terms)
    terms = {
      name: u.termination_manager.get_term(name).bool().tolist() for name in term_names
    }
    metrics = {name: value.tolist() for name, value in cmd.metrics.items()}

    if self._header is None:
      self._header = (
        ["step", "env", "reset", "clip", "clip_time", "clip_len_s"]
        + [f"term_{name}" for name in term_names]
        + [f"metric/{name}" for name in metrics]
      )
      self._file = self._path.open("w", newline="")
      self._writer = csv.writer(self._file)
      self._writer.writerow(self._header)

    for env_idx in range(num_envs):
      row: list[str] = [
        str(self._step),
        str(env_idx),
        str(int(reset[env_idx])),
        clips[env_idx] if env_idx < len(clips) else "",
        _num(time_s[env_idx]),
        _num(clip_len_s[env_idx]),
      ]
      row += [str(int(terms[name][env_idx])) for name in term_names]
      row += [_num(values[env_idx]) for values in metrics.values()]
      self._writer.writerow(row)
    self._file.flush()
    self._step += 1

  def finalize(self) -> None:
    if self._file is not None:
      self._file.close()
      self._file = None
