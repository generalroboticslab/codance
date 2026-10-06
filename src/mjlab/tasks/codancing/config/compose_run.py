"""Hydra-native run composition (compose-then-build-once).

A run is one config file in ``conf/``, composed by ``hydra.compose`` and built
into the ``(env_cfg, runner)`` bundle exactly once. Overlays and overrides use
Hydra's own grammar, and the AMP co-design is declarative: the env's
``amp_feature`` interpolates the runner's
``${runner.algorithm.amp.{num_frames, disc_mode}}`` values.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra_zen import instantiate, just
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.tasks.codancing.config.run_cfg import ExecCfg
from mjlab.tasks.codancing.config.term_builders import (
  _CURRICULUM_TERM_TARGET,
  _EVENT_TERM_TARGET,
  _METRICS_TERM_TARGET,
  _OBS_TERM_TARGET,
  _RECORDER_TERM_TARGET,
  _REWARD_TERM_TARGET,
  _TERMINATION_TERM_TARGET,
  _build_terms,
  _prepare,
)
from mjlab.tasks.shared.wrapper.scene import chain_spec_fns

_CONF = Path(__file__).parent / "g1" / "conf"


@dataclass
class RunBundle:
  """A fully composed run: the env cfg, the runner cfg, and the runner class.

  ``exec`` is the typed view of the tree's three non-env root blocks (see
  ``run_cfg``): ``motion`` + ``eval`` (frozen tiers) and ``session`` (live-only
  -- present here for THIS invocation, but stripped from the frozen snapshots
  below so it never travels with a checkpoint).
  """

  env_cfg: Any
  runner_cfg: Any
  runner_cls: type | None
  exec: ExecCfg = field(default_factory=ExecCfg)
  # The session-stripped resolved master tree as text (the "frozen" form stored
  # in checkpoints for reproducible resume / eval): env + runner + motion + eval,
  # NO session. The built `env_cfg` above is its eager build (training needs the
  # env up front; text alone can't start a run).
  resolved_cfg_yaml: str | None = None
  # The *dedicated* eval envs' resolved trees as text: a YAML mapping recipe ->
  # tree, session-stripped like the master tree and without its `eval` block
  # (the recipe specs live in the master tree) -- the single source of truth
  # for eval. Composed (through the same overlay machinery, each recipe's
  # `overlays`) + frozen for EVERY train run, so offline `run eval` can
  # reproduce any checkpoint. `None` only for play runs or if every recipe's
  # compose failed. Logged per recipe to `params/resolved_eval.{recipe}.yaml`.
  # Unlike the train env there is NO pre-built form: offline `run eval` builds
  # a tree lazily via `build_eval_env_cfg`, so a train run pays nothing for it.
  resolved_eval_cfg_yaml: str | None = None
  # Hydra's composition record for THIS run: which option each config group
  # resolved to (``runtime.choices``) plus the launch inputs (config_name /
  # overrides). Injected into the wandb ``train_cfg`` at launch so runs are
  # groupable by the axis that actually distinguishes them (``config_name``),
  # instead of a hand-typed tag. ``None`` on frozen-rebuild paths (resume) that
  # do not re-compose.
  composition: dict[str, Any] | None = None


# Sections that live under `env` but are built by the term-sugar helpers (not by
# `instantiate` directly), so they are stripped from the scaffold before it is
# instantiated and rebuilt afterwards.


def compose_run_from_file(
  config_file: str | Path,
  *,
  play: bool = False,
  overlays: Sequence[str] = (),
  overrides: Sequence[str] = (),
) -> RunBundle:
  """Compose a run from a single COMPOSABLE file in ``conf/`` (see mjlab.scripts.run).

  The file is an ordinary Hydra config IN the codancing config root
  (``conf/<name>.yaml``) whose ``defaults:`` list pulls the base ``run`` config
  and whose body sets any key inline: one file is the WHOLE run. On train its
  dedicated eval tree is derived live (no ``resolved_eval_config.yaml``
  sibling).

  ``conf/`` is the SOLE config dir (the file is part of it), so there is no second
  search path and nothing to shadow: a stray local ``overlay/`` or ``run.yaml``
  next to the file cannot silently override a shipped group. The flip side is the
  file must live directly in ``conf/`` root -- a subdir would make ``- run`` and
  the ``+overlay=[play]`` (play verb / derived eval) resolve relative to the
  subgroup and miss.

  ``play=True`` is the ``run play --config`` path: the default ``play`` overlay
  layers ON TOP of the file's defaults/body, so play semantics win over the
  file's train config, with explicit ``overlays`` after it and CLI ``overrides``
  last. NO eval tree is derived --
  play runs don't eval. With ``play=False`` (train) the eval tree is always
  derived.
  """
  path = Path(config_file).resolve()
  conf = _CONF.resolve()
  if path.parent != conf:
    raise ValueError(
      f"a composable --config file must live directly in the codancing config "
      f"root ({conf}) so it composes from a single path with no group shadowing; "
      f"got {path}. Move it to conf/<name>.yaml (and keep `- run` in its "
      "defaults). A fully resolved --config tree (no `defaults:`) may live "
      "anywhere; only the composable form needs conf/."
    )
  # `conf/run.yaml` is the base config every composable file pulls via `- run`;
  # a file named run.yaml would BE it / self-include. Reject clearly.
  if path.stem == "run":
    raise ValueError(
      f"a composable --config file cannot be conf/run.yaml ({path}): that is the "
      "base config its own `- run` default pulls. Name it conf/<experiment>.yaml."
    )
  config_name = path.stem
  cli_hydra = [str(o) for o in overrides]

  def compose_fn(specs: Sequence[str]) -> tuple[DictConfig, dict[str, Any]]:
    ov: list[str] = []
    if specs:
      ov.append("+overlay=[" + ",".join(specs) + "]")
    ov += cli_hydra
    cfg, choices = _compose(ov, config_name=config_name)
    return cfg, {
      "choices": choices,
      "config_name": config_name,
      "launch": {"overrides": ov},
    }

  # `play=True` layers the default play overlay OVER the file's defaults/body
  # (play semantics win over the file's train config); explicit overlays stack
  # after it, CLI overrides last.
  overlay_specs = (["play"] if play else []) + list(overlays)
  return _assemble_bundle(compose_fn, play=play, overlay_specs=overlay_specs)


def _assemble_bundle(
  compose_fn: Callable[[Sequence[str]], tuple[DictConfig, dict[str, Any]]],
  *,
  play: bool,
  overlay_specs: Sequence[str],
) -> RunBundle:
  """Build the ``(env, runner)`` bundle + derive the frozen eval trees.

  ``compose_fn(extra_overlays)`` composes the base tree (and, for each eval
  recipe, the same tree with that recipe's overlays layered on).
  """
  cfg, composition = compose_fn(overlay_specs)
  # The eval-tree consistency gate compares the COMPOSED runner deltas (the
  # bundle build below grafts the full runner dump into `cfg.runner`).
  composed_node = cfg.get("runner")
  # resolve=True: the env's amp_feature interpolates the runner's co-design
  # knobs (`${runner.algorithm.amp.disc_mode}`); the eval tree is compared
  # post-resolve, so both sides must be resolved values.
  composed_agent = (
    OmegaConf.to_container(composed_node, resolve=True)
    if OmegaConf.is_config(composed_node)
    else {}
  )
  bundle = _build_bundle(cfg)
  bundle.composition = composition

  # Compose one *dedicated* eval env per recipe via the same overlay mechanism
  # (the recipe's `overlays`; run.yaml's `default` recipe uses `["play"]` ->
  # infinite episode, no obs corruption, randomization beyond base_com and
  # foot_friction dropped). This is the declarative, logged, CLI-overridable
  # equivalent of play/runtime eval -- no bespoke "disable DR" runner flag. An
  # explicit empty list evals under the train distribution. NOTE: a recipe's
  # `overlays` are consumed HERE, at live-compose time only -- the eval env
  # trees are frozen below, so the lists are immutable post-freeze.
  #
  # Done for EVERY train run (play runs don't eval): freezing the eval tree is
  # ~free (compose an env cfg + store a yaml) and makes offline `run eval`
  # reproducible for ANY checkpoint. BEST-EFFORT by design: never block
  # training on it -- a bad / missing eval node just disables offline eval for
  # this run, with a warning (`run eval` on such a checkpoint then fails
  # loudly and points at play). The legality + consistency gates are the
  # exception: an eval overlay outside `eval.allowed_overlay_groups`, or one
  # that changes the motion/runner tiers, is a user config error -- not an
  # eval-compose hiccup -- and raises (BEFORE / OUTSIDE the best-effort try).
  if not play:
    allowed = list(bundle.exec.eval.allowed_overlay_groups)
    frozen_trees: dict[str, Any] = {}
    for recipe_name, recipe in bundle.exec.eval.recipes.items():
      eval_overlays = [str(o) for o in recipe.overlays]
      # Legality + consistency gates are user-config errors -> raise (BEFORE the
      # best-effort compose try). Each recipe is checked against the SHARED
      # allowlist; a bad recipe only disables that recipe's offline eval.
      _validate_eval_overlay_groups(eval_overlays, allowed)
      _warn_eval_overlay_session_keys(eval_overlays)
      try:
        eval_cfg, _ = compose_fn([*overlay_specs, *eval_overlays])
        OmegaConf.resolve(eval_cfg)
      except Exception as exc:
        print(
          f"[WARN] failed to compose the dedicated eval env for recipe "
          f"{recipe_name!r}; offline `run eval --recipe {recipe_name}` will be "
          f"unavailable for this run: {exc!r}"
        )
        continue
      # The gate compares COMPOSED runner deltas (cfg.runner now carries the
      # grafted full runner dump, captured pre-build as `composed_agent`).
      _validate_eval_tree_consistency(cfg, eval_cfg, composed_agent=composed_agent)
      # Graft the same full runner dump into the eval tree so every frozen tree
      # carries the identical complete runner catalog.
      with open_dict(eval_cfg):
        eval_cfg.runner = OmegaConf.create(cfg.runner)
      frozen = strip_session(eval_cfg)
      # A recipe tree is only ever built into its eval env; the recipe specs
      # themselves live in the train tree's `eval` block, so this copy of it
      # would be dead weight in every frozen tree.
      with open_dict(frozen):
        del frozen["eval"]
      frozen_trees[recipe_name] = frozen
    if frozen_trees:
      # ONE frozen artifact carrying every recipe: a YAML mapping recipe -> tree.
      # Built lazily per recipe via `eval_recipe_trees` + `build_eval_env_cfg`
      # when an offline eval runs.
      bundle.resolved_eval_cfg_yaml = OmegaConf.to_yaml(OmegaConf.create(frozen_trees))
  return bundle


def _warn_eval_overlay_session_keys(eval_overlays: Sequence[str]) -> None:
  """Warn when an eval overlay sets ``session.*`` keys (inert in the eval tree).

  The dedicated eval tree is session-stripped (``strip_session``), so an
  overlay's ``session`` block -- e.g. ``session.video.enabled`` or
  ``session.camera_source`` -- is frozen-but-dropped and can never take effect
  at eval time. A legal eval overlay group (``video`` is allowed for its
  env-side resolution) can still carry a session half; surface it rather than
  silently dropping it. The eval recorder is each recipe's own knobs
  (``video.enabled`` / ``length_*`` under ``eval.recipes.<name>``); env-side
  resolution rides ``video/res_480p`` (``env.viewer.*``).
  """
  import yaml as _yaml

  for name in eval_overlays:
    path = _CONF / "overlay" / f"{name}.yaml"
    if not path.exists():
      continue
    data = _yaml.safe_load(path.read_text()) or {}
    session = data.get("session") if isinstance(data, dict) else None
    if session:
      keys = sorted(session) if isinstance(session, dict) else [str(session)]
      print(
        f"[WARN] eval recipe overlay '{name}' sets session.* keys "
        f"({keys}); the dedicated eval tree is session-stripped, so these are "
        "inert at eval time. Use the recipe's recorder knobs "
        "(eval.recipes.<name>.video.enabled / .length_*) or an env-side overlay "
        "(e.g. video/res_480p) instead."
      )


def _validate_eval_overlay_groups(
  eval_overlays: Sequence[str], allowed_groups: Sequence[str]
) -> None:
  """Fail fast on eval recipe overlays outside the legal group set.

  The eval tree is a MEASUREMENT spec; only measurement-shaping overlay
  groups may compose into it (``eval.allowed_overlay_groups``, ordinary config
  that the config file or a per-run override may extend). A motion-, runner- or
  session-shaping overlay would be frozen but inert or ignored at runtime: the
  motion tier is shared with train, the policy is already trained, and the eval
  text is session-stripped.
  """
  for name in eval_overlays:
    group = name.split("/", 1)[0]
    if group not in allowed_groups:
      raise ValueError(
        f"eval recipe overlay '{name}': overlay group '{group}' is not legal "
        f"in the dedicated eval tree (allowed: {sorted(allowed_groups)}). "
        "The eval tree is a measurement spec -- motion/runner/session-shaping "
        "overlays would be frozen but never used. If this group genuinely "
        "shapes the measurement for your task, extend the list in the config "
        "file or per run: "
        "--override 'eval.allowed_overlay_groups=[...]' -- the motion/runner "
        "consistency gate still applies."
      )


def _validate_eval_tree_consistency(
  train_cfg: DictConfig,
  eval_cfg: DictConfig,
  *,
  composed_agent: Any = None,
) -> None:
  """Fail fast when a recipe's overlays changed the ``motion`` or ``runner`` tiers.

  Backstop behind the group allowlist: whatever groups are allowed, only the
  ``env`` block of the eval tree may differ from the train tree. The motion
  block is ONE frozen tier shared by both (at runtime the executor resolves
  motion from the train-side ``exec.motion`` -- ``run_exec._resolve_motion``),
  and the runner block describes the already-trained policy; a delta in either
  would be frozen text the executor never uses -- accepted, then silently
  ineffective. ``composed_agent`` is the train tree's runner node as COMPOSED
  (pre-graft deltas), since ``cfg.runner`` carries the full runner dump by the
  time this gate runs. To eval under a different motion, override ``motion.*``
  at eval time (recorded as a deviation) or fork.
  """
  for block in ("motion", "runner"):
    if block == "runner":
      train_node = composed_agent or {}
    else:
      train_node = _drop_missing(train_cfg.get(block)) if block in train_cfg else {}
    eval_node = _drop_missing(eval_cfg.get(block)) if block in eval_cfg else {}
    if train_node != eval_node:
      raise ValueError(
        f"an eval recipe overlay changed the `{block}` block of the dedicated "
        f"eval tree (train: {train_node!r} vs eval: {eval_node!r}). Only the "
        "eval ENV may differ from the train tree -- this delta would be frozen "
        "but never used. Keep the overlay out of the recipe's overlays; to eval "
        "under a different motion, override motion.* at eval time (recorded "
        "as a deviation) or fork."
      )


def eval_recipe_trees(resolved_eval_cfg_yaml: str) -> dict[str, str]:
  """Split the frozen eval mapping (recipe name -> resolved tree) into per-recipe
  tree yamls.

  Compose and ``freeze`` store every recipe as ONE YAML mapping (recipe
  name -> resolved eval tree); this returns ``{name: tree_yaml}`` so each tree
  builds via :func:`build_eval_env_cfg`.
  """
  mapping = OmegaConf.create(resolved_eval_cfg_yaml)
  assert isinstance(mapping, DictConfig)
  return {str(name): OmegaConf.to_yaml(mapping[name]) for name in mapping}


def build_eval_env_cfg(eval_tree_yaml: str) -> Any:
  """Build a dedicated eval env cfg from ONE recipe's resolved tree text.

  Single owner of eval-env construction: fresh runs (compose stores the recipe
  mapping) and resume / offline ``run eval`` (``freeze`` reads the frozen
  mapping) both go through here (after :func:`eval_recipe_trees` splits the
  mapping), so the eval env is built ONE way from the resolved yaml that is its
  single source of truth. Lazy by design -- callers build only when an eval
  actually runs. The text is already fully
  resolved + override-applied, so this is a pure build (no compose, no overrides).
  """
  eval_cfg = OmegaConf.create(eval_tree_yaml)
  assert isinstance(eval_cfg, DictConfig)
  return _build_env(eval_cfg.env)


def strip_session(cfg: DictConfig) -> DictConfig:
  """Return a copy of ``cfg`` without the live-only ``session`` block.

  The freeze boundary in code form: frozen artifacts (checkpoint-embedded
  trees, ``params/resolved*.yaml``) carry env + runner + motion + eval only, so
  a train invocation's session (GPUs, log dirs, recorders) structurally cannot
  leak into a later play/eval. ``copy.deepcopy`` keeps the input intact;
  ``open_dict`` lifts struct mode for the delete (compose outputs are struct).
  """
  import copy

  stripped = copy.deepcopy(cfg)
  if "session" in stripped:
    with open_dict(stripped):
      del stripped["session"]
  return stripped


def _drop_missing(node: Any) -> Any:
  """Recursively convert an OmegaConf node to a plain container, dropping MISSING.

  ``OmegaConf.to_container`` renders a MISSING (``???``) leaf as the literal
  string ``"???"``; here we skip those keys entirely so an unset ``???`` entrance
  in ``run.yaml`` resolves to "not provided" (the dataclass default downstream)
  rather than a bogus value. List elements are kept positionally (a ``???`` inside
  a list is uncommon and has no key to drop).
  """
  if isinstance(node, DictConfig):
    out: dict[str, Any] = {}
    for key in node:
      if OmegaConf.is_missing(node, key):
        continue
      out[str(key)] = _drop_missing(node[key])
    return out
  if isinstance(node, ListConfig):
    return [_drop_missing(v) for v in node]
  return node


def _build_bundle(cfg: DictConfig) -> RunBundle:
  """Resolve interpolations then build ``(env, runner, exec)`` once.

  The session-stripped resolved tree is captured as text AFTER the build (the
  env build pops manager sections off copies, so the canonical tree is
  intact) and AFTER the full runner dump is grafted into ``cfg.runner`` -- so
  the frozen snapshot (and every ``--dry-run`` ``resolved_config.yaml``)
  carries the COMPLETE runner catalog: every runner / algorithm field with its
  effective value, not just the deltas. Explicit over implicit -- the PPO
  entrypoint parameters are readable in the config artifact, and a frozen
  anchor rebuilds the runner from the dump rather than re-deriving it.

  ``_drop_missing`` drops MISSING (``???``) leaves instead of materializing
  them as the literal string ``"???"`` (what ``to_container`` would do): a
  ``???`` in ``run.yaml`` marks a tweak entrance with NO baked default, so an
  unset one must fall back to the dataclass default (e.g.
  ``checkpoint_file=None`` -> zero policy; an unset ``motion.*`` -> the clean
  "Must provide motion" error), never the broken ``"???"`` string.
  """
  OmegaConf.resolve(cfg)
  _validate_partner_mode(cfg)
  env = _build_env(cfg.env)
  runner, runner_cls = _build_runner(cfg)
  # Graft the full runner dump: `runner` becomes the complete typed catalog.
  with open_dict(cfg):
    cfg.runner = OmegaConf.create(just(runner))
  resolved_cfg_yaml = OmegaConf.to_yaml(strip_session(cfg))

  def _block(name: str) -> dict[str, Any]:
    return _drop_missing(cfg[name]) if name in cfg else {}

  # The load table travels on the built runner object's `.load` dict (the
  # runner-owned per-purpose module vocabulary).
  exec_cfg = ExecCfg.from_blocks(
    motion=_block("motion"),
    eval_block=_block("eval"),
    session=_block("session"),
    load_specs=runner.load,
  )
  return RunBundle(
    env_cfg=env,
    runner_cfg=runner,
    runner_cls=runner_cls,
    exec=exec_cfg,
    resolved_cfg_yaml=resolved_cfg_yaml,
  )


def _compose(
  overrides: list[str], *, config_name: str
) -> tuple[DictConfig, dict[str, str]]:
  """Run ``hydra.compose`` against the codancing config tree (``conf/``).

  ``config_name`` is a composable file's ``conf/<name>.yaml`` stem, composed from
  the single config dir (one path, no shadowing).

  Also returns Hydra's ``runtime.choices`` (which option each config group
  resolved to) so the run's composition can be logged to wandb as a groupable
  axis. The
  ``hydra`` node (added only by ``return_hydra_config``) is read for the choices
  then deleted, so the returned tree is identical to the pre-``return_hydra_config``
  form; the ``hydra/*`` internal groups are dropped, keeping only the app's own
  selections.
  """
  GlobalHydra.instance().clear()
  with initialize_config_dir(version_base=None, config_dir=str(_CONF)):
    cfg = compose(
      config_name=config_name, overrides=overrides, return_hydra_config=True
    )
  assert isinstance(cfg, DictConfig)
  raw = OmegaConf.to_container(cfg.hydra.runtime.choices, resolve=True)
  assert isinstance(raw, dict)
  choices = {
    str(k): str(v)
    for k, v in raw.items()
    if v is not None and not str(k).startswith("hydra/")
  }
  with open_dict(cfg):
    del cfg["hydra"]
  return cfg, choices


def _build_env(env_node: DictConfig) -> Any:
  """Build the env cfg from a composed ``env`` tree (compose-then-build-once).

  The scaffold (wrapper + scene/viewer/sim) is instantiated first, then the
  manager dicts -- already merged into their final, ordered form by Hydra -- are
  built from the term sugar and plugged in. The term *content* is whatever the
  composed tree carries (no in-Python whitelist or merge).
  """
  scaffold = OmegaConf.create(OmegaConf.to_container(env_node, resolve=True))
  assert isinstance(scaffold, DictConfig)

  observations = scaffold.pop("observations")
  rewards = scaffold.pop("rewards")
  events = scaffold.pop("events")
  terminations = scaffold.pop("terminations")
  curriculum = scaffold.pop("curriculum")
  metrics = scaffold.pop("metrics")
  recorders = scaffold.pop("recorders")
  actions = scaffold.pop("actions")
  commands = scaffold.pop("commands")
  sensors = scaffold.pop("sensors")
  entities = scaffold.scene.pop("entities")
  # `scene.spec_fns` is a named dict of spec_fn nodes; pop it before scaffold
  # instantiation (SceneCfg takes a single `spec_fn`) and chain the instantiated
  # entries into one ordered spec_fn below.
  spec_fns_node = scaffold.scene.pop("spec_fns", None)

  env = instantiate(_prepare(scaffold, None))
  env.observations = _build_observations(observations)
  env.actions = _build_terms(actions)
  env.commands = _build_terms(commands)
  # A nulled event term is ABSENT from the built cfg (overlay/play nulls the
  # randomization terms by name; the event manager would skip a None anyway).
  # Terminations keep their None entries: `motion_clip_end: null` is read back
  # as "nulled" by the cursor code and the tests.
  env.events = {
    name: term
    for name, term in _build_terms(events, _EVENT_TERM_TARGET).items()
    if term is not None
  }
  env.rewards = _build_terms(rewards, _REWARD_TERM_TARGET)
  env.terminations = _build_terms(terminations, _TERMINATION_TERM_TARGET)
  env.curriculum = _build_terms(curriculum, _CURRICULUM_TERM_TARGET)
  env.metrics = _build_terms(metrics, _METRICS_TERM_TARGET)
  env.recorders = _build_terms(recorders, _RECORDER_TERM_TARGET)
  env.scene.entities = _build_terms(entities)
  env.scene.sensors = tuple(
    instantiate(_prepare(node, None)) for node in sensors.values()
  )
  if spec_fns_node is not None:
    env.scene.spec_fn = chain_spec_fns(
      [instantiate(_prepare(node, None)) for node in spec_fns_node.values()]
    )
  _validate_human_entity_consistency(env)
  _validate_amp_feature(env)
  return env


def _validate_amp_feature(env: Any) -> None:
  """Fail fast on a degenerate AMP feature spec: no bodies, or the anchor
  among the obs bodies.

  An empty ``body_names`` gives a zero-width feature. With ``use_anchor``,
  every body's pose is expressed relative to the anchor; the anchor body
  itself then contributes constant identity features, silently wasting
  discriminator capacity. The check runs on the built command cfg.
  Duck-typed: non-AMP commands pass.
  """
  commands = getattr(env, "commands", None) or {}
  feature = getattr(commands.get("codancing"), "amp_feature", None)
  if feature is None:
    return
  if not tuple(feature.body_names or ()):
    raise ValueError(
      "AMP feature spec is empty: env.commands.codancing.amp_feature.body_names "
      "lists no body, so the feature would have zero width. List the AMP "
      "bodies (the AMP policy configs use ${body_sets.amp_lower_body_with_torso})."
    )
  if not getattr(feature, "use_anchor", False):
    return
  if getattr(feature, "allow_anchor_in_bodies", False):
    return  # deliberate padding (e.g. obs-dim parity across anchor swaps).
  if feature.anchor_body_name in tuple(feature.body_names or ()):
    raise ValueError(
      f"AMP feature spec is degenerate: anchor body "
      f"{feature.anchor_body_name!r} is also an observation body. With "
      "use_anchor=true its relative pose is the constant identity. Move the "
      "anchor (e.g. env.commands.codancing.amp_feature.anchor_body_name="
      "pelvis), drop the body from "
      "amp_feature.body_names, or set amp_feature.allow_anchor_in_bodies=true "
      "to keep the constant block deliberately (obs-dim parity)."
    )


def _validate_partner_mode(cfg: DictConfig) -> None:
  """Fail fast on an unknown ``partner_mode``.

  ``env.commands.codancing.partner_mode`` (``none`` | ``imagined`` | ``g1``) is
  the single source of truth for the human side of a run -- the
  ``load_human_entity`` / ``require_human_motion`` flags are derived read-only
  properties, so they cannot desync and need no cross-check. ``none`` is pure
  robot imitation (no entity, and the motion tier must stay robot-only),
  ``imagined`` reads the human from the motion file with no scene entity, and
  ``g1`` adds the live partner entity.

  The entity-level check in ``_validate_human_entity_consistency`` then verifies
  the composed scene matches the derived ``load_human_entity``. Duck-typed:
  trees without a ``partner_mode`` field (non-codancing commands) pass.
  """
  mode = OmegaConf.select(cfg, "env.commands.codancing.partner_mode")
  if mode is None:
    return
  if mode not in ("none", "imagined", "g1"):
    raise ValueError(
      f"Unknown partner_mode {mode!r}; expected one of none | imagined | g1."
    )


def _validate_human_entity_consistency(env: Any) -> None:
  """Fail fast when ``load_human_entity`` disagrees with the composed scene.

  Declarative composition has no imperative reconciliation: the flag and the
  partner group must agree by construction, and an ``--override`` that flips
  only the flag would otherwise build a silently inconsistent env -- entity
  present but never driven (``_write_human_state`` early-returns, the in-scene
  human collapses while contact sensors feed the critic garbage), or entity
  absent with the command about to crash on the scene lookup. Duck-typed so
  non-codancing command families pass through untouched.
  """
  commands = getattr(env, "commands", None) or {}
  command = commands.get("codancing")
  load_flag = getattr(command, "load_human_entity", None)
  human_asset = getattr(command, "human_asset_name", None)
  if command is None or load_flag is None or human_asset is None:
    return
  has_entity = human_asset in (getattr(env.scene, "entities", None) or {})
  if load_flag and not has_entity:
    raise ValueError(
      f"env.commands.codancing.load_human_entity=true, but entity "
      f"{human_asset!r} is not in the composed scene. A live partner "
      "(partner_mode=g1) needs its entity in env.scene.entities."
    )
  if not load_flag and has_entity:
    raise ValueError(
      f"env.commands.codancing.load_human_entity=false, but entity "
      f"{human_asset!r} (and its cross-entity sensors/terms) is still in the "
      "composed scene -- the human would sit un-driven while contact sensors "
      "observe it. The compose path has no imperative strip: remove the entity "
      "and its sensors, or set partner_mode=g1."
    )


def _build_observations(obs_node: DictConfig) -> dict[str, ObservationGroupCfg]:
  """Wrap each observer's (ordered) terms + flags into an ``ObservationGroupCfg``."""
  groups: dict[str, ObservationGroupCfg] = {}
  for name, ocfg in obs_node.items():
    groups[str(name)] = ObservationGroupCfg(
      terms=_build_terms(ocfg["terms"], _OBS_TERM_TARGET),
      concatenate_terms=bool(ocfg.get("concatenate_terms", True)),
      enable_corruption=bool(ocfg.get("enable_corruption", False)),
      history_length=ocfg.get("history_length", None),
      flatten_history_dim=bool(ocfg.get("flatten_history_dim", True)),
    )
  return groups


def _build_runner(cfg: DictConfig) -> tuple[Any, type | None]:
  """Build the runner cfg from the composed ``runner:`` block + dispatch its class.

  Mirror of ``_build_env``: the FULL typed catalog (``runner._target_`` + every
  runner / algorithm field, with the overlay / CLI ``runner.*`` deltas already
  merged in by Hydra) is instantiated directly into the runner dataclass. The
  ``experiment_name`` comes from the root ``experiment_stem``. The runner CLASS
  is then a typed dispatch on the cfg type (``RUNNER_CLS_BY_CFG``). AMP
  co-design needs no validation step: the runner's
  ``algorithm.amp.{num_frames, disc_mode}`` are the source of truth and the
  env's ``amp_feature`` interpolates them, so they agree by construction (and
  ``RslRlAmpCfg`` self-validates ``disc_mode``).
  """
  from mjlab.tasks.codancing.rl import RUNNER_CLS_BY_CFG

  runner = _build_runner_from_target(cfg.runner)
  runner.experiment_name = str(cfg.experiment_stem)
  runner_cls = RUNNER_CLS_BY_CFG.get(type(runner))
  return runner, runner_cls


def _build_runner_from_target(runner_node: DictConfig) -> Any:
  """Instantiate the ``runner:`` ``_target_`` catalog into the runner dataclass.

  The yaml round-trip turns tuple fields into lists (yaml has no tuple);
  ``_retuple_like`` restores the tuple types using a freshly-constructed default
  instance as the type template (it only fixes tuple-vs-list, preserving the
  instantiated values).
  """
  inst = instantiate(runner_node, _convert_="all")
  ref = type(inst)()
  return _retuple_like(ref, inst)


def _retuple_like(reference: Any, value: Any) -> Any:
  """Recursively restore ``tuple`` types in ``value`` where ``reference`` has them."""
  import dataclasses as _dc

  if isinstance(reference, tuple) and isinstance(value, list):
    ref_items = list(reference) + [reference[-1] if reference else None] * (
      len(value) - len(reference)
    )
    return tuple(_retuple_like(r, v) for r, v in zip(ref_items, value, strict=False))
  if _dc.is_dataclass(reference) and _dc.is_dataclass(value):
    for f in _dc.fields(reference):
      setattr(
        value, f.name, _retuple_like(getattr(reference, f.name), getattr(value, f.name))
      )
    return value
  if isinstance(reference, dict) and isinstance(value, dict):
    return {
      k: _retuple_like(reference.get(k), v) if k in reference else v
      for k, v in value.items()
    }
  return value
