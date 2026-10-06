"""Logging utilities for training scripts."""

import io
import sys
from pathlib import Path
from typing import IO, Any, Callable


class TeeStream(io.TextIOBase):
  """Duplicate writes to both the original stream and a log file."""

  def __init__(self, original: IO[str], log_file: IO[str]) -> None:
    self._original = original
    self._log_file = log_file

  def write(self, s: str) -> int:
    self._original.write(s)
    self._log_file.write(s)
    self._log_file.flush()
    return len(s)

  def flush(self) -> None:
    self._original.flush()
    self._log_file.flush()

  def __getattr__(self, name: str) -> Any:
    return getattr(self._original, name)


def tee_stdout_stderr(path: Path) -> Callable[[], None]:
  """Tee stdout/stderr to a log file. Returns a teardown function."""
  path.parent.mkdir(parents=True, exist_ok=True)
  fh = open(path, "a")
  old_stdout, old_stderr = sys.stdout, sys.stderr
  sys.stdout = TeeStream(sys.stdout, fh)  # type: ignore[assignment]
  sys.stderr = TeeStream(sys.stderr, fh)  # type: ignore[assignment]

  def teardown() -> None:
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    fh.close()

  return teardown
