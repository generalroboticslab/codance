"""Motion-file path resolution (local path or wandb artifact registry).

``resolve_motion_path`` turns a motion *spec* -- either a local file/dir or a
wandb artifact registry name -- into a concrete local ``.npz`` path, caching
downloaded artifacts under ``$MJLAB_ARTIFACT_CACHE`` (default ``artifacts_cache``).
Lives in ``mjlab.tasks.shared.utils`` (a task-level shared util, not core
``mjlab`` src) so the executor and the codancing scripts can share it without
importing a sibling entry-point module.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import rich


def _artifact_cache_subdir(registry_name: str) -> Path:
  """Compute cache subdir path that reflects artifact identity.

  The path mirrors the artifact identity (entity/project/name_version) so
  different artifacts never collide in the cache.
  """
  parts = registry_name.split(":", 1)
  base = parts[0]
  version = parts[1] if len(parts) > 1 else "latest"
  path_parts = base.split("/")

  # Sanitize for filesystem: no / or : in segment names.
  def sanitize(s: str) -> str:
    return s.replace("/", "_").replace(":", "_").strip() or "unknown"

  if len(path_parts) >= 3:
    entity, project, name = (
      sanitize(path_parts[0]),
      sanitize(path_parts[1]),
      sanitize("_".join(path_parts[2:])),
    )
  else:
    entity, project, name = "wandb", "artifacts", sanitize(base)
  return Path(entity) / project / f"{name}_{version}"


def resolve_motion_path(motion_spec: str) -> str:
  """Resolve a motion spec to a local path (downloading from wandb if needed)."""
  path = Path(motion_spec)
  if path.exists():
    if path.is_dir() and (path / "motion.npz").exists():
      out = str(path / "motion.npz")
      rich.print(
        "[bold green][resolve_motion_path][/] [cyan]local dir[/] (motion.npz) [dim]spec=[/][yellow]{!r}[/] [dim]->[/] [green]{}[/]".format(
          motion_spec, out
        )
      )
      return out
    rich.print(
      "[bold green][resolve_motion_path][/] [cyan]local path[/] [dim]spec=[/][yellow]{!r}[/] [dim]->[/] [green]{}[/]".format(
        motion_spec, str(path)
      )
    )
    return str(path)
  cache_dir = Path(os.environ.get("MJLAB_ARTIFACT_CACHE", "artifacts_cache")).resolve()
  registry_name = motion_spec if ":" in motion_spec else f"{motion_spec}:latest"
  cache_subdir_rel = _artifact_cache_subdir(registry_name)
  cached_npz = cache_dir / cache_subdir_rel / "motion.npz"
  if cached_npz.exists():
    rich.print(
      "[bold green][resolve_motion_path][/] [cyan]cache hit[/] [dim]spec=[/][yellow]{!r}[/] [dim]cache=[/][blue]{}[/] [dim]->[/] [green]{}[/]".format(
        motion_spec, cache_subdir_rel, str(cached_npz)
      )
    )
    return str(cached_npz)
  rich.print(
    "[bold yellow][resolve_motion_path][/] [cyan]downloading from wandb[/] [dim]registry=[/][yellow]{!r}[/] [dim]cache_dir=[/][blue]{}[/]".format(
      registry_name, cache_dir
    )
  )
  import wandb

  api = wandb.Api()
  artifact = api.artifact(registry_name)
  download_root = Path(artifact.download())
  result_npz = download_root / "motion.npz"
  if not result_npz.exists():
    rich.print(
      "[bold yellow][resolve_motion_path][/] [dim]no motion.npz in artifact, using root[/] [green]{}[/]".format(
        str(download_root)
      )
    )
    return str(download_root)
  cache_subdir = cache_dir / cache_subdir_rel
  cache_subdir.mkdir(parents=True, exist_ok=True)
  dest = cache_subdir / "motion.npz"
  if not dest.exists() or dest.stat().st_mtime < result_npz.stat().st_mtime:
    shutil.copy2(result_npz, dest)
    rich.print(
      "[bold green][resolve_motion_path][/] [cyan]cached[/] [dim]dest=[/][green]{}[/]".format(
        str(dest)
      )
    )
  else:
    rich.print(
      "[bold green][resolve_motion_path][/] [cyan]using existing cache[/] [dim]dest=[/][green]{}[/]".format(
        str(dest)
      )
    )
  # Record which artifact and wandb local path this cache corresponds to.
  meta_path = cache_subdir / "_artifact_meta.json"
  meta = {
    "artifact_registry": registry_name,
    "wandb_download_path": str(download_root),
  }
  with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
  rich.print(
    "[dim]  artifact_meta written: registry=[/][yellow]{!r}[/] [dim]wandb_path=[/][blue]{}[/]".format(
      registry_name, str(download_root)
    )
  )
  return str(dest)
