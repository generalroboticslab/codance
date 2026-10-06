import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path


def sync_wandb_run_symlinks(
  actual_wandb_dir: Path | None = None,
  local_wandb_path: Path | None = None,
) -> int:
  """Mirror W&B run subdirectories into local `wandb/` as symlinks.

  This is useful when local `wandb/` is a regular directory and `WANDB_DIR`
  points to a centralized archive path.

  Expected layout is fixed to the current W&B SDK convention:
  `<WANDB_DIR>/wandb/run-*`.

  Returns:
    Number of symlinks created in local `wandb/`.
  """
  created_symlink_links = _sync_wandb_run_symlinks_with_links(
    actual_wandb_dir=actual_wandb_dir, local_wandb_path=local_wandb_path
  )
  return len(created_symlink_links)


def _sync_wandb_run_symlinks_with_links(
  actual_wandb_dir: Path | None = None,
  local_wandb_path: Path | None = None,
) -> list[tuple[Path, Path]]:
  local_path = local_wandb_path or Path("wandb")
  if local_path.is_symlink():
    return []

  if actual_wandb_dir is None:
    wandb_dir_env = os.environ.get("WANDB_DIR")
    if wandb_dir_env is None or wandb_dir_env.strip() == "":
      return []
    actual_path = Path(wandb_dir_env).expanduser().resolve() / "wandb"
  else:
    actual_path = actual_wandb_dir.expanduser().resolve()

  if not actual_path.exists() or not actual_path.is_dir():
    return []

  local_path.mkdir(parents=True, exist_ok=True)
  if local_path.resolve() == actual_path:
    return []

  created_symlink_links: list[tuple[Path, Path]] = []

  def try_link(source_path: Path, target_path: Path) -> bool:
    if target_path.is_symlink():
      try:
        if target_path.resolve() == source_path.resolve():
          return False
      except FileNotFoundError:
        # Broken symlink; keep untouched to avoid destructive changes.
        return False
    elif target_path.exists():
      return False

    try:
      os.symlink(source_path, target_path)
      return True
    except FileExistsError:
      return False

  for entry in actual_path.iterdir():
    if not entry.is_dir():
      continue
    if not (entry.name.startswith("run-") or entry.name.startswith("offline-run-")):
      continue
    if try_link(entry, local_path / entry.name):
      created_symlink_links.append((local_path / entry.name, entry))

  latest_run_path = actual_path / "latest-run"
  if latest_run_path.exists() or latest_run_path.is_symlink():
    if try_link(latest_run_path, local_path / "latest-run"):
      created_symlink_links.append((local_path / "latest-run", latest_run_path))

  return created_symlink_links


def _sync_wandb_run_symlinks_after_checkpoint_with_links(
  log_dir: Path,
  checkpoint_name: str = "model_0.pt",
  max_wait_seconds: float | None = 5.0,
  retry_interval_seconds: float = 0.5,
) -> list[tuple[Path, Path]]:
  checkpoint_path = log_dir / checkpoint_name
  retry_start_time = time.monotonic()
  while True:
    if checkpoint_path.exists():
      created_symlink_links = _sync_wandb_run_symlinks_with_links()
      if created_symlink_links:
        return created_symlink_links

    if max_wait_seconds is not None:
      elapsed_seconds = time.monotonic() - retry_start_time
      if elapsed_seconds >= max_wait_seconds:
        return []

    time.sleep(retry_interval_seconds)


def start_wandb_symlink_watcher(
  log_dir: Path,
  checkpoint_name: str = "model_0.pt",
  retry_interval_seconds: float = 0.5,
) -> threading.Thread:
  """Start a background watcher that syncs symlinks after checkpoint appears."""

  def watcher_func() -> None:
    created_symlink_links = _sync_wandb_run_symlinks_after_checkpoint_with_links(
      log_dir=log_dir,
      checkpoint_name=checkpoint_name,
      max_wait_seconds=None,
      retry_interval_seconds=retry_interval_seconds,
    )
    created_symlink_count = len(created_symlink_links)
    if created_symlink_count > 0:
      print(
        "[INFO] W&B local run symlinks updated by watcher under "
        f"wandb/: +{created_symlink_count}"
      )
      run_links = [
        (local_path, target_path)
        for local_path, target_path in created_symlink_links
        if target_path.name.startswith("run-")
        or target_path.name.startswith("offline-run-")
      ]
      if run_links:
        latest_local_path, latest_target_path = max(
          run_links, key=lambda item: item[1].name
        )
      else:
        latest_local_path, latest_target_path = created_symlink_links[-1]
      print(f"[INFO]   link: {latest_local_path} -> {latest_target_path}")

  watcher_thread = threading.Thread(
    target=watcher_func,
    name=f"wandb-symlink-watcher-{log_dir.name}",
    daemon=True,
  )
  watcher_thread.start()
  return watcher_thread


def setup_wandb_runtime_dir(log_dir_prefix: Path) -> tuple[Path, bool]:
  """Configure W&B runtime directory under the log-dir prefix.

  Returns:
    Tuple of (actual_wandb_run_dir, local_dir_created).
  """
  log_dir_prefix_path = log_dir_prefix.expanduser().resolve()
  # Keep WANDB_DIR as parent root so W&B writes to `<WANDB_DIR>/wandb/run-*`.
  actual_wandb_run_dir = log_dir_prefix_path / "wandb"
  actual_wandb_run_dir.mkdir(parents=True, exist_ok=True)

  local_wandb_path = Path("wandb")
  local_dir_created = False
  if local_wandb_path.is_symlink():
    if local_wandb_path.resolve() != actual_wandb_run_dir:
      raise ValueError(
        "wandb symlink already exists with a different target: "
        f"{local_wandb_path} -> {local_wandb_path.resolve()}, "
        f"expected {actual_wandb_run_dir}"
      )
  elif local_wandb_path.exists():
    # Keep existing local path untouched; WANDB_DIR will still redirect runtime logs.
    pass
  else:
    # Always create a real local directory instead of a top-level symlink.
    local_wandb_path.mkdir(parents=True, exist_ok=True)
    local_dir_created = True

  os.environ["WANDB_DIR"] = str(log_dir_prefix_path)
  print("[INFO] W&B link trajectory:")
  print(f"[INFO]   WANDB_DIR: {log_dir_prefix_path}")
  print(f"[INFO]   run root: {actual_wandb_run_dir}")
  print(f"[INFO]   local dir: {local_wandb_path.resolve()}")
  print(f"[INFO]   link rule: {local_wandb_path}/run-* -> {actual_wandb_run_dir}/run-*")
  # For existing local `wandb/` dir, mirror run-* entries as subdir symlinks.
  sync_wandb_run_symlinks(
    actual_wandb_dir=actual_wandb_run_dir, local_wandb_path=local_wandb_path
  )
  return actual_wandb_run_dir, local_dir_created


def write_git_recovery_snapshot(
  log_dir: Path,
  repository_file_path: str | os.PathLike[str],
  *,
  include_submodules: bool = True,
) -> list[Path]:
  """Write recoverable git snapshot artifacts for the repository.

  The superproject's files are saved under ``<log_dir>/git/``:

  - ``<repo>.commit``: training-time HEAD commit hash
  - ``<repo>.full.patch``: full binary patch from HEAD to working-tree snapshot
  - ``<repo>.source.zip``: full source archive for fallback recovery
  - ``<repo>.sha256``: checksums for snapshot artifacts
  - ``<repo>.manifest.json``: metadata for debugging/recovery

  When ``include_submodules`` is set (the default), every initialized git
  submodule is captured the same way (recursively) under
  ``<log_dir>/git/submodules/<path>/``. This makes a run reproducible even when
  the code that actually trains lives in a submodule on a local/unpushed branch
  with uncommitted edits: the superproject ``git archive`` / ``git diff`` only see
  a submodule as a gitlink, never its contents, so without this the submodule
  would be lost. Capture is independent of how the superproject got its ``.git``
  -- a normal checkout, a ``--recurse-submodules`` clone, or a shallow ``.git``
  all work; each submodule need only be an initialized repo with a working tree.
  Submodule artifacts sit under ``git/submodules/``, so readers that glob
  ``git/*.commit`` see only the superproject.
  """
  repository_file = Path(repository_file_path).resolve()
  repo_root = _run_git_command(
    ["git", "rev-parse", "--show-toplevel"], cwd=repository_file.parent
  )
  if repo_root is None:
    return []
  repo_root_path = Path(repo_root)

  git_log_dir = log_dir / "git"
  artifacts = _snapshot_repo_into(repo_root_path, git_log_dir)
  if not artifacts:
    return []
  if include_submodules:
    for submodule_root, rel_path in _iter_submodule_roots(repo_root_path):
      artifacts += _snapshot_repo_into(
        submodule_root, git_log_dir / "submodules" / rel_path
      )
  return artifacts


def _snapshot_repo_into(repo_root_path: Path, out_dir: Path) -> list[Path]:
  """Write one repo's recovery snapshot (HEAD + working-tree patch + archive)
  into ``out_dir``, named by the repo's directory basename.

  Shared by the superproject and every submodule. Returns the written artifact
  paths, or ``[]`` when the repo has no resolvable HEAD or the snapshot fails --
  a broken/empty submodule is skipped, never fatal to the rest of the run.
  """
  commit_hash = _run_git_command(["git", "rev-parse", "HEAD"], cwd=repo_root_path)
  if commit_hash is None:
    return []

  snapshot_commit = _create_snapshot_commit(repo_root_path, commit_hash)
  if snapshot_commit is None:
    return []

  patch_bytes = _run_git_command_bytes(
    ["git", "diff", "--binary", "--full-index", commit_hash, snapshot_commit],
    cwd=repo_root_path,
  )
  source_zip_bytes = _run_git_command_bytes(
    ["git", "archive", "--format=zip", snapshot_commit],
    cwd=repo_root_path,
  )
  if patch_bytes is None or source_zip_bytes is None:
    return []

  out_dir.mkdir(parents=True, exist_ok=True)
  repo_name = repo_root_path.name
  commit_path = out_dir / f"{repo_name}.commit"
  patch_path = out_dir / f"{repo_name}.full.patch"
  source_zip_path = out_dir / f"{repo_name}.source.zip"
  sha256_path = out_dir / f"{repo_name}.sha256"
  manifest_path = out_dir / f"{repo_name}.manifest.json"

  _atomic_write_text(commit_path, f"{commit_hash}\n")
  _atomic_write_bytes(patch_path, patch_bytes)
  _atomic_write_bytes(source_zip_path, source_zip_bytes)

  sha_records = {
    commit_path.name: _sha256_hex(commit_path.read_bytes()),
    patch_path.name: _sha256_hex(patch_bytes),
    source_zip_path.name: _sha256_hex(source_zip_bytes),
  }
  sha_lines = [
    f"{sha_records[file_name]}  {file_name}" for file_name in sorted(sha_records)
  ]
  _atomic_write_text(sha256_path, "\n".join(sha_lines) + "\n")

  manifest = {
    "schema_version": 1,
    "repo_name": repo_name,
    "repo_root": str(repo_root_path),
    "head_commit": commit_hash,
    "snapshot_commit": snapshot_commit,
    "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "archive_format": "zip",
    "files": {
      commit_path.name: {
        "sha256": sha_records[commit_path.name],
        "size": commit_path.stat().st_size,
      },
      patch_path.name: {
        "sha256": sha_records[patch_path.name],
        "size": len(patch_bytes),
      },
      source_zip_path.name: {
        "sha256": sha_records[source_zip_path.name],
        "size": len(source_zip_bytes),
      },
      sha256_path.name: {"size": sha256_path.stat().st_size},
    },
  }
  _atomic_write_text(
    manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
  )

  return [commit_path, patch_path, source_zip_path, sha256_path, manifest_path]


def _iter_submodule_roots(repo_root_path: Path):
  """Yield ``(submodule_root, rel_path)`` for every initialized submodule,
  recursively.

  Reads ``.gitmodules`` directly rather than ``git submodule status`` so it does
  not depend on submodule init bookkeeping -- the same code works against a plain
  checkout, a ``--recurse-submodules`` clone, or a shallow ``.git`` shipped into
  a staging copy. A declared submodule with no working tree (no ``.git`` entry)
  is skipped. ``rel_path`` is POSIX-style and relative to the top
  ``repo_root_path`` (nested submodules carry the full path), mapping directly
  onto ``git/submodules/<rel_path>/``.
  """
  gitmodules = repo_root_path / ".gitmodules"
  if not gitmodules.is_file():
    return
  listing = _run_git_command(
    ["git", "config", "--file", str(gitmodules), "--get-regexp", r"\.path$"],
    cwd=repo_root_path,
  )
  if not listing:
    return
  for line in listing.splitlines():
    # Each line is "submodule.<name>.path <relpath>"; the path is everything
    # after the first space (submodule names may contain dots and slashes).
    _, _, rel = line.partition(" ")
    rel = rel.strip()
    if not rel:
      continue
    submodule_root = repo_root_path / rel
    if not (submodule_root / ".git").exists():
      continue  # declared in .gitmodules but not checked out in this tree
    yield submodule_root, rel
    for nested_root, nested_rel in _iter_submodule_roots(submodule_root):
      yield nested_root, f"{rel}/{nested_rel}"


def build_run_meta(
  *,
  task_id: str,
  train_script: str,
  agent_cfg: dict[str, object],
  play_script: str | None = None,
  repository_file_path: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
  """The run's STABLE identity -- the single source shared by ``meta.json``
  (:func:`write_training_meta`) and each checkpoint's ``infos['run_meta']``.

  Holds only non-drifting facts: task / experiment / script / time, the git
  ``head_commit``, and ``log_dir_prefix``. NO machine identity (host name,
  checkout path): checkpoints get shared, and these would travel with them. NO
  ``log_dir`` (machine path -- meta.json adds it) and NO wandb fields (runtime --
  merged in by :func:`current_wandb_meta` once ``wandb.run`` exists).
  ``created_at`` is stamped ONCE here, so compute this once per run and reuse it
  for both sinks rather than rebuilding it per checkpoint.
  """
  meta: dict[str, object] = {
    "schema_version": 1,
    "task_id": task_id,
    "train_script": train_script,
    "experiment_name": agent_cfg.get("experiment_name", ""),
    "run_name": agent_cfg.get("run_name", ""),
    "log_dir_prefix": str(agent_cfg.get("log_dir_prefix", "")),
    "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
  }
  if play_script is not None:
    meta["play_script"] = play_script
  if repository_file_path is not None:
    cwd = Path(repository_file_path).resolve().parent
    head_commit = _run_git_command(["git", "rev-parse", "HEAD"], cwd=cwd)
    if head_commit is not None:
      meta["head_commit"] = head_commit
  return meta


def current_wandb_meta() -> dict[str, str]:
  """This run's wandb identity from the live ``wandb.run`` (empty when wandb is
  not running). Populated only on the logging rank (rsl_rl gates wandb to rank
  0) and only after ``wandb.init``, so read it at save time / from the meta
  updater -- not at compose time.
  """
  try:
    import wandb
  except Exception:
    return {}
  run = wandb.run
  if run is None:
    return {}
  return {
    "wandb_run_id": run.id,
    "wandb_run_name": run.name,  # the timestamped display name (== run-dir stamp)
    "wandb_entity": run.entity,
    "wandb_project": run.project,
    "wandb_run_path": f"{run.entity}/{run.project}/{run.id}",
    "wandb_run_url": run.url,
  }


def current_run_meta(stable_meta: Mapping[str, object] | None) -> dict[str, object]:
  """The run's full identity: its stable meta (:func:`build_run_meta`) merged with
  the live wandb ids (:func:`current_wandb_meta`; empty when wandb is not running),
  as the checkpoint embeds it."""
  return {**(stable_meta or {}), **current_wandb_meta()}


def write_training_meta(
  log_dir: Path,
  task_id: str,
  train_script: str,
  agent_cfg: dict[str, object],
  repository_file_path: str | os.PathLike[str] | None = None,
  play_script: str | None = None,
) -> dict[str, object]:
  """Write ``<log_dir>/meta.json`` with the run's STABLE identity and RETURN it.

  The body is :func:`build_run_meta` (the single source of the stable field set,
  shared with each checkpoint's ``infos['run_meta']``) plus the machine-local
  ``log_dir``. wandb fields are patched into meta.json later by
  :func:`start_training_meta_wandb_updater` once ``wandb.run`` exists. The
  returned dict (the stable identity, no ``log_dir``) is what the caller stamps
  onto the runner so every checkpoint self-describes the same run.

  Deliberately omits the raw train invocation (``run_command`` / ``sys_argv``):
  its override knob names drift across config revisions, so it would mislead
  when read against a later revision.
  """
  run_meta = build_run_meta(
    task_id=task_id,
    train_script=train_script,
    agent_cfg=agent_cfg,
    play_script=play_script,
    repository_file_path=repository_file_path,
  )
  meta_path = log_dir / "meta.json"
  _atomic_write_text(
    meta_path,
    json.dumps({**run_meta, "log_dir": str(log_dir)}, indent=2, sort_keys=True) + "\n",
  )
  return run_meta


def start_training_meta_wandb_updater(
  log_dir: Path,
  poll_interval_seconds: float = 1.0,
  max_wait_seconds: float = 120.0,
) -> threading.Thread:
  """Start a daemon thread that patches ``meta.json`` with wandb fields once available.

  ``wandb.init()`` is called lazily by the rsl_rl runner at the start of
  ``learn()``, which happens *after* :func:`write_training_meta`.  This
  thread polls until ``wandb.run`` becomes non-None, then writes the
  run ID, entity, project, and URL into the existing meta file and uploads
  the patched file to the run (Files tab), so the run identity is browsable
  on wandb too.
  """

  def _updater() -> None:
    import wandb

    elapsed = 0.0
    while wandb.run is None and elapsed < max_wait_seconds:
      time.sleep(poll_interval_seconds)
      elapsed += poll_interval_seconds

    if wandb.run is None:
      return

    meta_path = log_dir / "meta.json"
    if not meta_path.exists():
      return

    meta: dict[str, object] = json.loads(meta_path.read_text())
    if "wandb_run_id" in meta:
      return

    meta.update(current_wandb_meta())
    _atomic_write_text(meta_path, json.dumps(meta, indent=2, sort_keys=True) + "\n")
    # Mirror the patched file into the wandb run's Files tab: it is the one
    # browsable copy of the identity fields wandb has no native slot for;
    # everything runs off the local file.
    wandb.save(str(meta_path), base_path=str(log_dir))

  thread = threading.Thread(target=_updater, daemon=True)
  thread.start()
  return thread


def _create_snapshot_commit(repo_root: Path, head_commit: str) -> str | None:
  with tempfile.NamedTemporaryFile(prefix="mjlab_snapshot_index_") as temp_index:
    env = os.environ.copy()
    env["GIT_INDEX_FILE"] = temp_index.name

    if (
      _run_git_command(["git", "read-tree", head_commit], cwd=repo_root, env=env)
      is None
    ):
      return None
    if _run_git_command(["git", "add", "-A"], cwd=repo_root, env=env) is None:
      return None
    snapshot_tree = _run_git_command(["git", "write-tree"], cwd=repo_root, env=env)
    if snapshot_tree is None:
      return None
    return _run_git_command(
      ["git", "commit-tree", snapshot_tree, "-p", head_commit],
      cwd=repo_root,
      env=env,
      stdin_text="mjlab training snapshot\n",
    )


def _run_git_command(
  command: list[str],
  cwd: Path,
  env: dict[str, str] | None = None,
  stdin_text: str | None = None,
) -> str | None:
  try:
    completed = subprocess.run(
      command,
      cwd=str(cwd),
      check=True,
      capture_output=True,
      text=True,
      env=env,
      input=stdin_text,
    )
  except (FileNotFoundError, subprocess.CalledProcessError):
    return None
  return completed.stdout.strip()


def _run_git_command_bytes(
  command: list[str], cwd: Path, env: dict[str, str] | None = None
) -> bytes | None:
  try:
    completed = subprocess.run(
      command,
      cwd=str(cwd),
      check=True,
      capture_output=True,
      env=env,
    )
  except (FileNotFoundError, subprocess.CalledProcessError):
    return None
  return completed.stdout


def _atomic_write_text(target_path: Path, content: str) -> None:
  _atomic_write_bytes(target_path, content.encode("utf-8"))


def _atomic_write_bytes(target_path: Path, content: bytes) -> None:
  target_path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(
    mode="wb",
    dir=target_path.parent,
    prefix=f".{target_path.name}.",
    suffix=".tmp",
    delete=False,
  ) as temp_file:
    temp_file.write(content)
    temp_file.flush()
    os.fsync(temp_file.fileno())
    temp_path = Path(temp_file.name)
  temp_path.replace(target_path)


def _sha256_hex(content: bytes) -> str:
  return hashlib.sha256(content).hexdigest()
