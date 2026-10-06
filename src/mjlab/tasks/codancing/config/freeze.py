"""Config freeze contract: store the resolved trees in the checkpoint.

A composed run's *resolved*, session-stripped config trees
(``RunBundle.resolved_cfg_yaml`` + ``resolved_eval_cfg_yaml``) are the
reproducibility artifacts: embedded -- as text -- into the RSL-RL checkpoint's
``infos`` slot at save time, and read back when a ``--ckpt`` anchor rebuilds
the run (resume / fork / eval / play) from the exact frozen trees rather than
from live config groups that may have drifted. The live-only ``session`` block
never freezes (``strip_session``); every frozen load grafts a fresh
``SessionCfg``-default session before this invocation's deltas apply.

Delta channels on a frozen base (see ``frozen_cfg_to_bundle``): ``overlays``
ground via the grounding engine (value merge / bundler recursion; group
re-selection fails -- recompose for that, i.e. ``fork --config``), then
node-level ``overrides`` (``key=value`` / ``~key``) apply last; both hit the
train AND eval trees symmetrically. ``resume``
additionally diffs the delta-applied tree against the baseline
(``resume_delta_paths``) to enforce its allowed-diff spec.

This module is engine-side and import-light; the runner save hook and
the run script wire it to the checkpoint lifecycle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from mjlab.tasks.codancing.config.compose_run import (
  RunBundle,
  _build_bundle,
  eval_recipe_trees,
  strip_session,
)
from mjlab.tasks.codancing.config.overlay_ground import ground_overlays
from mjlab.tasks.codancing.config.run_cfg import SessionCfg

# Key under the checkpoint's ``infos`` dict that carries the frozen tree text.
FROZEN_CFG_KEY = "resolved_cfg"
# Key for the dedicated eval envs' frozen trees (a recipe -> tree mapping, each
# composed with that recipe's ``overlays``). Embedded alongside the train tree
# for EVERY train run so resume reproduces the *exact* eval envs, and so a
# standalone ``run eval`` of the checkpoint can rebuild them without
# re-composing.
FROZEN_EVAL_CFG_KEY = "resolved_eval_cfg"
# Key under the checkpoint's ``infos`` dict carrying the run's STABLE identity --
# the same non-drifting set ``meta.json`` records (task / experiment / script /
# time / head_commit / wandb ids), single-sourced via
# ``wandb_runtime.build_run_meta``. Embedding it lets a ``.pt`` name the run it
# came from on its own. The drift-prone train CLI command is deliberately NOT
# carried here (its knob names drift across revisions, misleading on old ckpts).
RUN_META_KEY = "run_meta"

# Provenance-ONLY keys: the training invocation's session block and CLI, baked
# so a bare .pt also answers "how was this trained" (params/session.yaml and
# wandb-metadata record the same offline/online). Nothing on the load path
# reads them -- a frozen load still grafts a fresh session (`strip_session`
# stays the freeze boundary) -- and override knob names may drift across
# config revisions, so treat the command as a record of the launch, not a
# guaranteed-replayable string.
TRAIN_SESSION_KEY = "train_session_cfg"
TRAIN_COMMAND_KEY = "train_command"


def embed_train_provenance(
  infos: Mapping[str, Any] | None,
  session_yaml: str | None,
  command: Sequence[str] | None,
) -> dict[str, Any]:
  """Return a new ``infos`` dict carrying the train invocation's provenance.

  A no-op merge when both are ``None``, safe to call unconditionally from the
  runner save hook (mirrors :func:`embed_resolved_cfg`).
  """
  merged: dict[str, Any] = dict(infos or {})
  if session_yaml is not None:
    merged[TRAIN_SESSION_KEY] = session_yaml
  if command is not None:
    merged[TRAIN_COMMAND_KEY] = list(command)
  return merged


def embed_resolved_cfg(
  infos: Mapping[str, Any] | None,
  resolved_cfg_yaml: str | None,
  resolved_eval_cfg_yaml: str | None = None,
) -> dict[str, Any]:
  """Return a new ``infos`` dict carrying the frozen tree text(s) (if any).

  A no-op merge when both args are ``None``, so it is safe to call
  unconditionally from the runner save hook. ``resolved_eval_cfg_yaml`` is the
  dedicated eval env's frozen tree (present for every train run; ``None`` only
  when the eval compose failed).
  """
  merged: dict[str, Any] = dict(infos or {})
  if resolved_cfg_yaml is not None:
    merged[FROZEN_CFG_KEY] = resolved_cfg_yaml
  if resolved_eval_cfg_yaml is not None:
    merged[FROZEN_EVAL_CFG_KEY] = resolved_eval_cfg_yaml
  return merged


def embed_run_meta(
  infos: Mapping[str, Any] | None,
  run_meta: Mapping[str, Any] | None,
) -> dict[str, Any]:
  """Return a new ``infos`` dict carrying the run's STABLE identity under
  ``run_meta`` (a no-op merge when ``run_meta`` is empty / ``None``).

  ``run_meta`` is the same ``build_run_meta`` output ``meta.json`` uses, merged
  with the live wandb fields -- so a checkpoint self-describes which run / commit
  / host produced it without its sibling ``meta.json``. The drift-prone train CLI
  command is intentionally NOT part of it (its knob names drift across config
  revisions and would mislead on an old checkpoint).
  """
  merged: dict[str, Any] = dict(infos or {})
  if run_meta:
    merged[RUN_META_KEY] = dict(run_meta)
  return merged


@dataclass(frozen=True)
class CheckpointMeta:
  """The cheap-to-keep slice of a checkpoint: top-level keys + frozen texts.

  A checkpoint is a multi-hundred-MB pickle (weights + optimizer moments), but
  every *metadata* consumer -- the frozen-tree readers, the per-algorithm
  ``load_cfg`` dispatch (which only inspects the key set) -- needs just this
  slice. One full deserialize per (path, mtime, size) populates it; the tensors
  are dropped immediately, so a fork/eval/play startup pays ONE metadata load
  instead of one per consumer (the runner's own ``load`` of the weights is the
  only other read).
  """

  frozen_cfg: str | None
  frozen_eval_cfg: str | None
  # The run's stable identity (``infos['run_meta']``: run_name / wandb ids /
  # commit / host / log dir), so a play session can name the run that produced
  # the policy in its manifest. ``None`` when the checkpoint carries no
  # run_meta.
  run_meta: Mapping[str, Any] | None


_META_CACHE: dict[tuple[str, int, int, int], CheckpointMeta] = {}


def read_checkpoint_meta(checkpoint_path: str | Path) -> CheckpointMeta:
  """Read (or return the cached) metadata slice of a checkpoint."""
  import zlib

  import torch

  path = Path(checkpoint_path)
  stat = path.stat()
  # (mtime, size) alone is too coarse: a same-size rewrite within one kernel
  # clock tick (coarse mtime granularity) would serve stale metadata. The CRC
  # of the file head (where the pickle's key names live) disambiguates, at
  # negligible cost next to the full deserialize it saves.
  with open(path, "rb") as f:
    head_crc = zlib.crc32(f.read(65536))
  cache_key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size, head_crc)
  cached = _META_CACHE.get(cache_key)
  if cached is not None:
    return cached

  loaded = torch.load(path, map_location="cpu", weights_only=False)
  infos = loaded.get("infos") if isinstance(loaded, Mapping) else None

  def _text(key: str) -> str | None:
    if not isinstance(infos, Mapping):
      return None
    value = infos.get(key)
    return str(value) if value is not None else None

  raw_run_meta = infos.get(RUN_META_KEY) if isinstance(infos, Mapping) else None
  meta = CheckpointMeta(
    frozen_cfg=_text(FROZEN_CFG_KEY),
    frozen_eval_cfg=_text(FROZEN_EVAL_CFG_KEY),
    run_meta=dict(raw_run_meta) if isinstance(raw_run_meta, Mapping) else None,
  )
  _META_CACHE[cache_key] = meta
  return meta


def read_frozen_cfg(checkpoint_path: str | Path) -> str | None:
  """Read the frozen train tree text from a checkpoint, or ``None`` if absent."""
  return read_checkpoint_meta(checkpoint_path).frozen_cfg


def read_frozen_eval_cfg(checkpoint_path: str | Path) -> str | None:
  """Read the frozen eval env tree from a checkpoint (or ``None``)."""
  return read_checkpoint_meta(checkpoint_path).frozen_eval_cfg


def _apply_node_overrides(
  cfg: DictConfig, overrides: Sequence[str], *, strict: bool = True
) -> None:
  """Apply node-level overrides to a (frozen or composed) tree, in place.

  Supports the universal subset of the Hydra override grammar that operates on
  resolved nodes: ``~dotted.key`` deletes a node; ``key=value`` (and ``+key=``)
  set / add one. Group-level defaults-list overrides are *not* available on a
  frozen tree (there are no config groups to re-select); recompose for that.

  With ``strict`` (the default), a delete whose target does not exist raises --
  a delete that matches nothing is a typo or a stale path, and silently
  ignoring it would let the user believe the node is gone. ``strict=False`` is
  for the secondary (eval) tree, where the same override legitimately may not
  apply (e.g. the eval tree composed with play semantics already dropped the
  node the train-tree delete targets). In that mode a SET override whose
  parent node is absent is skipped entirely -- merging it anyway would
  resurrect the dropped node as a broken partial (e.g. a reward term with
  ``params`` but no ``func``).

  With ``strict``, every set override's ROOT segment must be an existing
  top-level block of the tree (``env`` / ``runner`` / ``motion`` / ``eval`` /
  ``session``): a frozen tree has no config groups, so a group-style key like
  ``viz=g1`` would otherwise merge in as a new root that nothing ever reads --
  accepted, then silently ignored. The eval trees carry no ``eval`` block (a
  recipe's spec lives in the train tree), so there an ``eval.*`` override is
  skipped like any other set whose root or parent the tree lacks.
  """
  set_items: list[str] = []
  for ov in overrides:
    if ov.startswith("~"):
      _delete_node(cfg, ov[1:].strip(), strict=strict)
    else:
      set_items.append(ov[1:] if ov.startswith("+") else ov)
  if set_items:
    roots = {str(k) for k in cfg.keys()}

    def root_of(item: str) -> str:
      return item.split("=", 1)[0].split(".", 1)[0].lstrip("+")

    if strict:
      for item in set_items:
        root = root_of(item)
        if root not in roots:
          raise ValueError(
            f"override '{item}': '{root}' is not a top-level block of the "
            f"frozen tree (has: {sorted(roots)}). Config groups (e.g. "
            "'viz=g1') exist only when a config file is composed, not on a "
            "checkpoint's frozen tree; use --config to compose with another "
            "group, or override a concrete node (env.*, runner.*)."
          )
    else:
      set_items = [
        item
        for item in set_items
        if root_of(item) in roots and _set_parent_exists(cfg, item.split("=", 1)[0])
      ]
  if set_items:
    merged = OmegaConf.merge(cfg, OmegaConf.from_dotlist(set_items))
    assert isinstance(merged, DictConfig)
    # Copy back so the caller's reference reflects the merge.
    cfg.merge_with(merged)


def _set_parent_exists(cfg: DictConfig, dotted_key: str) -> bool:
  """True when every segment of ``dotted_key`` except the leaf exists in ``cfg``.

  Non-strict guard for set overrides on the secondary (eval) tree: a missing
  intermediate node means the eval tree dropped the subtree the override
  targets (play semantics), so the override is skipped instead of recreating
  the node as a broken partial.
  """
  *parents, _leaf = dotted_key.split(".")
  node: Any = cfg
  for part in parents:
    if not isinstance(node, DictConfig) or part not in node or node[part] is None:
      return False
    node = node[part]
  return True


def _delete_node(cfg: DictConfig, dotted: str, *, strict: bool = True) -> None:
  if "=" in dotted:
    # Hydra's own parser accepts `~key=value` (delete-if-equal); this node
    # grammar does not -- reject it loudly instead of gluing `=value` onto the
    # leaf name and silently matching nothing.
    raise ValueError(
      f"override '~{dotted}': the delete form takes a bare dotted key "
      "('~env.events.push_robot'); the Hydra '~key=value' conditional-delete "
      "form is not supported on a resolved tree. Drop the '=...' part."
    )
  *parents, leaf = dotted.split(".")
  node: Any = cfg
  for part in parents:
    if part not in node:
      if strict:
        raise ValueError(
          f"override '~{dotted}' matched nothing: '{part}' is not in the "
          "tree. Nothing was deleted -- fix the path (deletes take an "
          "existing dotted key, e.g. '~env.events.push_robot')."
        )
      return
    node = node[part]
  if not (isinstance(node, DictConfig) and leaf in node):
    if strict:
      raise ValueError(
        f"override '~{dotted}' matched nothing: no node '{leaf}' under "
        f"'{'.'.join(parents) or '<root>'}'. Nothing was deleted -- fix the "
        "path (deletes take an existing dotted key)."
      )
    return
  del node[leaf]


def _graft_session(cfg: DictConfig) -> None:
  """Graft a FRESH ``session`` block (dataclass defaults) onto a frozen tree.

  Frozen trees carry no session by construction (``strip_session``); every
  invocation that anchors on one starts its session from the ``SessionCfg``
  defaults, with deltas supplied only by this invocation's overlays/overrides.
  ``asdict`` keeps :class:`SessionCfg` the single source of truth (run.yaml's
  predeclared ``session:`` block mirrors it for live composes).
  """
  cfg["session"] = OmegaConf.create(asdict(SessionCfg()))


def _prepare_frozen_tree(
  text: str,
  *,
  overlays: Sequence[str] = (),
  overrides: Sequence[str] = (),
  strict_overrides: bool = True,
) -> DictConfig:
  """Parse frozen text, graft a fresh session, apply this invocation's deltas.

  The single owner of the delta contract (session graft -> overlays ground ->
  node overrides), so the train tree, the eval tree, and resume's diff baseline
  cannot drift in ordering or guarding.
  """
  cfg = OmegaConf.create(text)
  assert isinstance(cfg, DictConfig)
  # Frozen checkpoint trees are session-stripped, so this always grafts on a
  # --ckpt anchor. A raw --config file MAY carry its own session block; honor it
  # instead of silently discarding it.
  if "session" not in cfg:
    _graft_session(cfg)
  if overlays:
    ground_overlays(cfg, overlays)
  if overrides:
    _apply_node_overrides(cfg, overrides, strict=strict_overrides)
  return cfg


def frozen_cfg_to_bundle(
  resolved_cfg_yaml: str,
  *,
  overlays: Sequence[str] = (),
  overrides: Sequence[str] = (),
  resolved_eval_cfg_yaml: str | None = None,
) -> RunBundle:
  """Rebuild a ``RunBundle`` from frozen tree text + this invocation's deltas.

  Delta order matches live compose: ``overlays`` ground first (in CLI order,
  via :func:`overlay_ground.ground_overlays` -- value merges and bundler
  recursion; group re-selection fails with a teaching error), then node
  ``overrides`` apply last. Both trees get a fresh ``session`` graft
  BEFORE the deltas, so ``session.*`` channels work uniformly on a frozen
  base. The deltas apply to BOTH trees symmetrically: whichever env actually
  runs receives them.

  When ``resolved_eval_cfg_yaml`` is given (the embedded eval env tree), the
  *delta-applied*, session-stripped text is stored as
  ``bundle.resolved_eval_cfg_yaml`` -- the single eval source of truth. The
  eval env itself is not built here; consumers build it lazily via
  ``compose_run.build_eval_env_cfg``.
  """
  cfg = _prepare_frozen_tree(
    resolved_cfg_yaml,
    overlays=overlays,
    overrides=overrides,
  )
  bundle = _build_bundle(cfg)
  if resolved_eval_cfg_yaml is not None:
    # The frozen eval artifact is a mapping recipe -> tree; apply this
    # invocation's deltas to EACH recipe tree, then re-freeze the mapping.
    # Non-strict overrides on the secondary trees: the train tree (above) already
    # validated every override; a delete may legitimately not apply here (a
    # recipe's play semantics can have dropped the node).
    frozen_trees: dict[str, Any] = {}
    for recipe_name, tree_yaml in eval_recipe_trees(resolved_eval_cfg_yaml).items():
      eval_cfg = _prepare_frozen_tree(
        tree_yaml,
        overlays=overlays,
        overrides=overrides,
        strict_overrides=False,
      )
      # Grounded bodies may carry interpolations; resolve before re-freezing
      # (mirrors `_build_bundle`'s resolve for the main tree).
      OmegaConf.resolve(eval_cfg)
      frozen_trees[recipe_name] = strip_session(eval_cfg)
    bundle.resolved_eval_cfg_yaml = OmegaConf.to_yaml(OmegaConf.create(frozen_trees))
  return bundle


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
  """Flatten a plain container to ``{dotted.path: leaf}`` (lists are leaves)."""
  if not isinstance(node, dict):
    return {prefix: node}
  out: dict[str, Any] = {}
  for key, value in node.items():
    dotted = f"{prefix}.{key}" if prefix else str(key)
    out.update(_flatten(value, dotted))
  return out


def changed_paths(a: DictConfig, b: DictConfig) -> list[str]:
  """Dotted paths whose values differ between two trees (added/removed/changed).

  MISSING (``???``) leaves compare as their literal marker, so an unset leaf on
  both sides is "unchanged" and filling one in is a change.
  """
  flat_a = _flatten(OmegaConf.to_container(a, resolve=False, throw_on_missing=False))
  flat_b = _flatten(OmegaConf.to_container(b, resolve=False, throw_on_missing=False))
  missing = object()
  return sorted(
    key
    for key in set(flat_a) | set(flat_b)
    if flat_a.get(key, missing) != flat_b.get(key, missing)
  )


def resume_delta_paths(
  checkpoint_path: str | Path,
  *,
  overlays: Sequence[str] = (),
  overrides: Sequence[str] = (),
) -> list[str]:
  """The dotted paths this invocation's deltas would change on a frozen base.

  Baseline = the checkpoint's frozen tree WITH the fresh session already
  grafted (session keys are fresh on both sides, so only the deltas show up);
  the delta side additionally grounds the overlays and applies the overrides.
  Pure text/tree work -- no env build -- so `resume` (and its --dry-run) can
  enforce the allowed-diff spec before anything expensive happens.
  """
  text = read_frozen_cfg(checkpoint_path)
  if text is None:
    raise ValueError(
      f"No frozen config in checkpoint {checkpoint_path!r} "
      f"(infos['{FROZEN_CFG_KEY}'] missing); resume needs a checkpoint saved by "
      "`run train`."
    )

  def _tree(apply_deltas: bool) -> DictConfig:
    cfg = _prepare_frozen_tree(
      text,
      overlays=overlays if apply_deltas else (),
      overrides=overrides if apply_deltas else (),
    )
    OmegaConf.resolve(cfg)
    return cfg

  return changed_paths(_tree(False), _tree(True))


def load_frozen_run(
  checkpoint_path: str | Path,
  *,
  overlays: Sequence[str] = (),
  overrides: Sequence[str] = (),
) -> RunBundle:
  """Rebuild a run from the frozen tree stored in ``checkpoint_path``.

  Raises if the checkpoint carries no frozen tree (one not saved by
  ``run train``). The dedicated eval env tree is carried through
  (as text) when present, and built lazily on demand, so offline eval
  reproduces it exactly. ``overlays``/``overrides`` are this invocation's
  delta channels (see :func:`frozen_cfg_to_bundle`).
  """
  text = read_frozen_cfg(checkpoint_path)
  if text is None:
    raise ValueError(
      f"No frozen config in checkpoint {checkpoint_path!r} "
      f"(infos['{FROZEN_CFG_KEY}'] missing); it was not saved by `run train`. "
      "Rebuild from a --config file instead of anchoring on this checkpoint."
    )
  return frozen_cfg_to_bundle(
    text,
    overlays=overlays,
    overrides=overrides,
    resolved_eval_cfg_yaml=read_frozen_eval_cfg(checkpoint_path),
  )


def load_raw_run(
  config_path: str | Path,
  *,
  play: bool = False,
  overlays: Sequence[str] = (),
  overrides: Sequence[str] = (),
) -> RunBundle:
  """Build a ``RunBundle`` straight from a raw resolved-tree YAML (no compose).

  The single-file escape hatch for fast iteration: the whole run -- env +
  runner + motion + eval (+ an optional ``session`` block) -- lives in ONE
  editable YAML, the exact shape ``--dry-run`` dumps as
  ``resolved_config.yaml``. This bypasses Hydra composition, so ANY key is
  directly tweakable. Build reuses
  the frozen-anchor path (:func:`frozen_cfg_to_bundle`); ``overlays`` ground and
  ``overrides`` apply on top, same as a ``--ckpt`` anchor.

  The dedicated eval trees (the recipe -> tree mapping ``--dry-run`` dumps as
  ``resolved_eval_config.yaml``) are read from that sibling next to
  ``config_path`` -- the side-by-side layout ``--dry-run`` writes. Without the
  sibling the run trains from the single file alone, and its checkpoints carry
  no eval tree (offline ``run eval`` on them fails loudly and points at play).

  ``play=True`` is the ``run play --config`` path: the tree plays EXACTLY as
  written (add ``--overlay play`` for play semantics), and the eval sibling is
  never read.
  """
  path = Path(config_path)
  text = path.read_text()
  sibling = path.with_name("resolved_eval_config.yaml")
  eval_text = (
    sibling.read_text() if not play and sibling.exists() and sibling != path else None
  )
  if eval_text is not None:
    print(f"[INFO] --config: using the eval tree from {sibling}")

  # Session is live-only: a raw file with no `session:` block gets fresh
  # SessionCfg() defaults (like a --ckpt anchor). The dry-run writes session to a
  # SEPARATE resolved_session.yaml that is NOT auto-loaded (unlike the eval
  # sibling), so a round-trip silently resets host choices (e.g. gpu_ids: all ->
  # [0]). Warn when that sibling exists but the tree carries no session, so the
  # reset is visible -- bake a `session:` block in or pass --override session.*.
  session_sibling = path.with_name("resolved_session.yaml")
  raw = OmegaConf.create(text)
  if (
    session_sibling.exists()
    and session_sibling != path
    and isinstance(raw, DictConfig)
    and "session" not in raw
  ):
    print(
      f"[WARN] --config: {path} has no `session:` block, so the session "
      f"settings fall back to their defaults. {session_sibling.name} next to it "
      "is not loaded automatically: add a session: block to the file or pass "
      "--override session.* (e.g. session.launch.gpu_ids=all / "
      "session.viewer=native)."
    )

  return frozen_cfg_to_bundle(
    text,
    overlays=overlays,
    overrides=overrides,
    resolved_eval_cfg_yaml=eval_text,
  )
