"""Ground catalog overlays onto FROZEN (resolved) config trees.

Live composes apply overlays through native Hydra config groups; a frozen
checkpoint tree has no groups left, so this module re-applies the same overlay
files as *tree transformations* -- making ``--overlay`` work uniformly on
checkpoint anchors (play/eval/resume/fork of a frozen base).

Per overlay, the defaults-list entries ground as follows:

* bundler include (``- /overlay/<group>: <option>``) -> recurse into that
  overlay (cycle-guarded; repeats are skipped, so diamond includes converge).
* group RE-SELECTION (``- override /<group>: ...``) -> impossible on a
  resolved tree (config groups exist only at compose time); raise a teaching
  error pointing at a live recompose.
* ``_self_`` -> no-op (the value body always merges last, see below).

After its defaults ground, the overlay's ``# @package _global_`` value body
merges onto the tree. Interpolations in bodies are left unresolved here --
callers resolve once at the end (mirroring live compose, where resolution
happens in ``_build_bundle``), so later overlays can still override the values
an earlier body interpolates.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from mjlab.tasks.codancing.config.compose_run import _CONF


def ground_overlays(
  cfg: DictConfig, names: Sequence[str], *, conf_dir: Path = _CONF
) -> None:
  """Apply the named catalog overlays onto a resolved tree, in place, in order.

  "In order" means LATER WINS, exactly like the CLI contract: an explicit
  repeat (``["a", "b", "a"]``) keeps only its last occurrence, so the final
  ``a`` re-applies over ``b`` instead of being skipped. The ``seen`` set only
  de-duplicates *bundler includes* (diamond includes converge -- two bundlers
  both pulling ``video/res_480p`` apply it once); an explicit top-level name
  always applies.

  Callers resolve interpolations afterwards (``OmegaConf.resolve`` /
  ``_build_bundle``), never here.
  """
  # Keep the LAST occurrence of each explicit name, preserving the order of
  # those last occurrences (CLI later-wins).
  deduped: list[str] = []
  for name in names:
    if name in deduped:
      deduped.remove(name)
    deduped.append(name)
  seen: set[str] = set()
  for name in deduped:
    _ground_one(cfg, name, conf_dir, seen, explicit=True)


def _ground_one(
  cfg: DictConfig, name: str, conf_dir: Path, seen: set[str], *, explicit: bool = False
) -> None:
  if name in seen and not explicit:
    return  # diamond bundler includes converge; one application suffices.
  seen.add(name)
  path = conf_dir / "overlay" / f"{name}.yaml"
  if not path.exists():
    raise FileNotFoundError(
      f"overlay '{name}' not found at {path} (known groups live under "
      f"{conf_dir / 'overlay'})."
    )
  raw = OmegaConf.load(path)
  assert isinstance(raw, DictConfig)
  defaults = raw.pop("defaults", None)
  if defaults is not None:
    for entry in defaults:
      _ground_default_entry(cfg, entry, name, conf_dir, seen)
  # The remaining body is the `# @package _global_` value patch; it merges
  # LAST (the `_self_` position in every catalog overlay).
  cfg.merge_with(raw)


def _ground_default_entry(
  cfg: DictConfig, entry: Any, overlay_name: str, conf_dir: Path, seen: set[str]
) -> None:
  if isinstance(entry, str):
    if entry == "_self_":
      return
    raise ValueError(
      f"overlay '{overlay_name}' carries a defaults entry {entry!r} that cannot "
      "be applied to a checkpoint; an overlay's defaults list may only name "
      "other overlays."
    )
  assert isinstance(entry, DictConfig), f"unexpected defaults entry: {entry!r}"
  ((key, value),) = dict(entry).items()
  key = str(key)
  if key.startswith("/overlay/"):
    group = key.removeprefix("/overlay/")
    _ground_one(cfg, f"{group}/{value}", conf_dir, seen)
    return
  if key.startswith("override /"):
    group = key.removeprefix("override /")
    # Config groups exist only at compose time, so a re-selection cannot ground
    # onto a resolved tree -- teach the live path.
    raise ValueError(
      f"overlay '{overlay_name}' selects config group '{group}' = {value!r} "
      "in its defaults list, which cannot be applied to a checkpoint's frozen "
      "tree (config groups exist only when a config file is composed). Use "
      f"--config instead: run <subcommand> --config FILE --overlay {overlay_name} ..."
    )
  raise ValueError(
    f"overlay '{overlay_name}' carries a defaults entry {{{key}: {value!r}}} "
    "that cannot be applied to a checkpoint."
  )
