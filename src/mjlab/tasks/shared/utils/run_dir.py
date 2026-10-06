"""Experiment run-dir naming (task layer): ``YYYYMMDD_HHMMSS_NNN``."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

# Matches a run-dir name's leading ``YYYYMMDD_HHMMSS_NNN`` and captures the
# counter (``\d+`` so it survives past 999); a ``_<run_name>`` suffix after the
# counter is ignored.
_RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}_(\d+)")


def timestamped_run_dir_name(parent: Path) -> str:
  """A second-resolution experiment run-dir name with a sequence counter:
  ``YYYYMMDD_HHMMSS_NNN`` (e.g. ``20260614_192634_001``).

  Keeping the SECONDS (not minute-only) leaves the timestamp near-unique on its
  own -- the safe choice on remote / docker, where many runs launch close
  together and a coarser stamp would lean entirely on the counter. ``NNN`` is the
  next sequence index among existing ``<...>_NNN*`` run dirs under ``parent``
  (``<log_root>/<experiment>``): a stable ordinal and a same-second tie-breaker.
  A ``run_name`` suffix after the counter doesn't affect it. Best-effort (a plain
  scan, not atomic) -- fine because the seconds already carry the uniqueness.
  """
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  return f"{stamp}_{_next_run_counter(parent):03d}"


def _next_run_counter(parent: Path) -> int:
  if not parent.is_dir():
    return 1
  used = [
    int(match.group(1))
    for entry in parent.iterdir()
    if (match := _RUN_DIR_RE.match(entry.name))
  ]
  return max(used, default=0) + 1
