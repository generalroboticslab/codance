"""Train, continue, evaluate and watch CoDance policies.

Usage:

  run train  --config FILE [--overlay NAME]... [--override K=V]...
  run resume --ckpt PATH [--overlay NAME]... [--override K=V]...
  run fork   --ckpt PATH [--config FILE] [--overlay NAME]... [--override K=V]...
  run eval   --ckpt PATH [--recipe NAME|all] [--overlay NAME]... [--override K=V]...
  run play   (--ckpt PATH | --config FILE) [--overlay NAME]... [--override K=V]...

Every subcommand also takes --dry-run [--dry-run-dir DIR]; -h or --help prints
this text.

Subcommands:

  train   Train a policy from a config file. The seven files in
          src/mjlab/tasks/codancing/config/g1/conf/ are complete runs.
  resume  Continue a run from one of its checkpoints, in the same run
          directory: weights, optimizer state and iteration are restored.
          Only session settings, eval settings, runner.max_iterations,
          runner.save_interval and the logging fields may differ from the
          original run; any other change is refused, naming the keys.
  fork    Start a new run from a checkpoint's weights, with a fresh optimizer
          at iteration 0. The configuration comes from the checkpoint, or from
          --config FILE when given. Logs go to <parent run>/forks/.
  eval    Run one of the evaluation recipes frozen in the checkpoint
          (--recipe NAME; `default` when omitted, `all` for every recipe) and
          write the session files next to the checkpoint.
  play    Watch a policy in its training world: the native viewer when a
          display is available, the browser viewer otherwise, or headless.

Options:

  --ckpt PATH       A checkpoint. It supplies the weights and the frozen
                    configurations (environment, runner, motion, eval) the run
                    was trained with; only the session block (viewer, video,
                    recording, GPUs, log directories) starts fresh.
  --config FILE     A whole run in one file. A file with a top-level
                    `defaults:` list (the seven policy configs) is composed by
                    Hydra and must live in conf/; a file without one is a fully
                    resolved tree (what --dry-run writes) and may live anywhere.
                    train, play and fork take it; fork combines it with --ckpt
                    (configuration from the file, weights from the checkpoint).
  --overlay NAME    Apply an overlay from conf/overlay/ (for example `play` or
                    `video/res_480p`); repeatable. On a checkpoint an overlay
                    may change values of the frozen configuration but not swap
                    its Hydra groups.
  --override K=V    Set one leaf of the configuration, applied last;
                    repeatable. Example: --override env.scene.num_envs=4096.
  --recipe NAME     eval only: the recipe to run, or `all`.
  --dry-run         Resolve and validate everything, write resolved_config.yaml,
                    resolved_eval_config.yaml and resolved_session.yaml (into
                    /tmp, or --dry-run-dir DIR), and exit before building the
                    environment.

Notes:

  * eval reproduces the recipe exactly unless --override or --overlay change
    something; the session manifest then lists the changes under
    `deviations`. A recipe's own `overlays` list cannot be overridden on a
    checkpoint; use --overlay instead.
  * play --config: a composable file is played with the `play` overlay (set
    session.checkpoint_file in the file or with --override to load trained
    weights); a fully resolved tree is played exactly as written, so add
    --overlay play for the evaluation conditions.
  * The README shows the commands the released policies are trained,
    evaluated and watched with; config/run_cfg.py is the typed view of the
    configuration.
"""

from __future__ import annotations

import os

# MuJoCo freezes its GL backend at FIRST `import mujoco` (transitively pulled
# in by the mjlab imports below), so the headless-safe default must be set here
# -- not inside the verbs. Train workers re-set it in their fresh processes
# (`_setup_gpu_env`); eval and headless/viser play render offscreen in THIS
# process, where a stale remote `DISPLAY` would crash GLFW before the rollout
# starts. EGL coexists with the native viewer's own GLFW window, and an
# explicit user `MUJOCO_GL` always wins.
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from mjlab.tasks.codancing.config.compose_run import RunBundle, compose_run_from_file
from mjlab.tasks.codancing.config.freeze import (
  load_frozen_run,
  load_raw_run,
  resume_delta_paths,
)
from mjlab.tasks.codancing.run_exec import (
  enforce_resume_allowlist,
  launch_training,
  reject_recipe_overlay_deltas,
  resolve_checkpoint_spec,
  resolve_fork_load_cfg,
  run_eval,
  run_fork,
  run_play,
  run_resume,
  select_eval_recipes,
  session_record_yaml,
)

_VERBS = ("train", "resume", "fork", "eval", "play")


@dataclass
class Invocation:
  """The parsed CLI: verb + selection + delta channels (pure, testable)."""

  verb: str
  ckpt: str | None = None
  config: str | None = None
  """Single-file mode (train, play, fork): ONE yaml is the WHOLE run. Two shapes,
  picked by a top-level ``defaults:`` key: COMPOSABLE (in ``conf/`` root, composed
  live; play composes WITH the play overlay) or RAW (a fully-resolved ``--dry-run``
  ``resolved_config.yaml`` tree, anywhere, built with no compose; play runs it
  exactly as written). ``--override``/``--overlay`` still apply on top. ``fork``
  pairs it with ``--ckpt`` (config from the file, weights from the checkpoint); the
  other verbs reject ``--ckpt`` beside it."""
  overlays: list[str] = field(default_factory=list)
  overrides: list[str] = field(default_factory=list)
  dry_run: bool = False
  recipe: str | None = None
  """Offline ``eval`` recipe selector: a recipe name or ``all`` (eval verb only;
  ``default`` when omitted)."""
  dry_run_dir: str | None = None
  """Directory for the ``--dry-run`` dumps (resolved_config / resolved_eval_config
  / resolved_session). Defaults to ``/tmp`` -- out of the repo, predictable to
  read. Set via ``--dry-run-dir PATH``."""


def _pop_option(args: list[str], flag: str) -> tuple[str | None, list[str]]:
  """Pop a single ``--flag value`` pair from ``args`` (returns value + rest)."""
  if flag not in args:
    return None, args
  idx = args.index(flag)
  if idx + 1 >= len(args):
    raise SystemExit(f"{flag} requires a value.")
  return args[idx + 1], args[:idx] + args[idx + 2 :]


def _pop_options(args: list[str], flag: str) -> tuple[list[str], list[str]]:
  """Pop every ``--flag value`` occurrence (repeatable single-value option)."""
  values: list[str] = []
  while flag in args:
    value, args = _pop_option(args, flag)
    assert value is not None
    values.append(value)
  return values, args


def _pop_flag(args: list[str], flag: str) -> tuple[bool, list[str]]:
  """Pop a valueless boolean ``--flag`` (returns whether it was present)."""
  if flag not in args:
    return False, args
  return True, [a for a in args if a != flag]


def _parse_cli(argv: list[str]) -> Invocation:
  """Parse + validate one invocation (the verb x flag matrix, teaching errors)."""
  if "-h" in argv or "--help" in argv:
    print(__doc__)
    raise SystemExit(0)
  if not argv or argv[0] not in _VERBS:
    raise SystemExit(
      f"run takes a subcommand, one of {', '.join(_VERBS)} "
      f"(got: {argv[0] if argv else 'nothing'}); `run --help` explains them."
    )
  verb, remaining = argv[0], list(argv[1:])
  ckpt, remaining = _pop_option(remaining, "--ckpt")
  config, remaining = _pop_option(remaining, "--config")
  overlays, remaining = _pop_options(remaining, "--overlay")
  overrides, remaining = _pop_options(remaining, "--override")
  dry_run, remaining = _pop_flag(remaining, "--dry-run")
  dry_run_dir, remaining = _pop_option(remaining, "--dry-run-dir")
  recipe, remaining = _pop_option(remaining, "--recipe")

  # Any leftover token is unrecognized; checked up front to fail fast, before
  # any compose.
  if remaining:
    raise SystemExit(
      f"Unrecognized arguments: {remaining}. Settings are changed with "
      "--override key=value."
    )

  inv = Invocation(
    verb=verb,
    ckpt=ckpt,
    config=config,
    overlays=overlays,
    overrides=overrides,
    dry_run=dry_run,
    dry_run_dir=dry_run_dir,
    recipe=recipe,
  )
  _validate(inv)
  return inv


def _validate(inv: Invocation) -> None:
  if inv.dry_run_dir is not None and not inv.dry_run:
    raise SystemExit("--dry-run-dir only applies with --dry-run.")
  if inv.recipe is not None and inv.verb != "eval":
    raise SystemExit(
      f"--recipe selects the evaluation recipe for `run eval`; {inv.verb} takes none."
    )

  if inv.config is not None:
    if inv.verb not in ("train", "play", "fork"):
      raise SystemExit(
        "--config is for train, play and fork only (fork combines it with --ckpt: "
        "configuration from the file, weights from the checkpoint). resume and "
        "eval read their configuration from the checkpoint; use --ckpt."
      )
    if inv.ckpt is not None and inv.verb != "fork":
      raise SystemExit(
        "--config and --ckpt exclude each other outside fork: the file is the "
        "whole run. Drop one, or change the file's settings with --override "
        "key=value."
      )

  if inv.verb == "train":
    if inv.ckpt is not None:
      raise SystemExit(
        "train does not take --ckpt: use `resume` to continue the run (same "
        "directory; weights, optimizer and iteration restored) or `fork` for a "
        "new run from the checkpoint's weights."
      )
    if inv.config is None:
      raise SystemExit(
        "train needs --config FILE: a config in conf/ (for example "
        "src/mjlab/tasks/codancing/config/g1/conf/decoupled.yaml) or a fully "
        "resolved tree (the resolved_config.yaml that --dry-run writes)."
      )
  elif inv.verb in ("resume", "eval"):
    if inv.ckpt is None:
      raise SystemExit(f"{inv.verb} requires --ckpt <run>/model_<it>.pt.")
  elif inv.verb == "fork":
    if inv.ckpt is None:
      raise SystemExit(
        "fork requires --ckpt, the source of the weights. With --config the "
        "configuration comes from that file; without it, from the checkpoint."
      )
  elif inv.verb == "play":
    # --config beside --ckpt is already rejected above.
    if inv.config is None and inv.ckpt is None:
      raise SystemExit(
        "play takes one of --ckpt PATH (the training world and policy of that "
        "checkpoint) or --config FILE (a play session in one file; point it at "
        "weights with --override session.checkpoint_file=...)."
      )


def _load_config_file(config_file: str) -> DictConfig:
  """Load + sanity-check a ``--config`` file, with clean CLI errors (no traceback).

  Covers the common typos -- a missing path, an empty file, or a top-level YAML
  list/scalar -- with a one-line ``SystemExit`` consistent with the parse layer,
  instead of an ``OmegaConf``/``AssertionError`` stack trace from deep in the
  build.
  """
  if not Path(config_file).exists():
    raise SystemExit(f"--config file not found: {config_file}")
  try:
    cfg = OmegaConf.load(config_file)
  except Exception as exc:
    raise SystemExit(f"--config: cannot parse {config_file}: {exc}") from None
  if not isinstance(cfg, DictConfig):
    raise SystemExit(
      f"--config: {config_file} must be a YAML mapping, either a composable "
      "config (with a `defaults:` list) or a fully resolved tree; got a "
      f"top-level {type(cfg).__name__}."
    )
  if len(cfg) == 0:
    raise SystemExit(
      f"--config: {config_file} is empty; it must be a composable config (a "
      "`defaults:` list, in conf/) or a fully resolved tree."
    )
  return cfg


def _assemble(inv: Invocation) -> RunBundle:
  """Run the verb's config assembly (a --config file, or a frozen anchor + deltas)."""
  if inv.config is not None:
    # Single-file mode (train + play + fork). Two shapes, picked by the file itself:
    #   * a top-level `defaults:` list  -> COMPOSABLE: a real hydra.compose of
    #     the file, which must live in conf/ root. A play verb composes WITH the
    #     default play overlay.
    #   * no `defaults:`                -> RAW resolved tree: built straight, no
    #     compose. Train reads the sibling resolved_eval_config.yaml for its
    #     eval trees; play runs the tree EXACTLY as written (add --overlay play
    #     for play semantics) and reads no sibling.
    # `defaults:` is Hydra's own "compose me" marker, so the split is unambiguous.
    # Deltas (--overlay/--override) apply on top in both shapes. The explicit
    # config-locating guards (file missing/empty/non-mapping, not-under-conf/)
    # and build-time ValueErrors surface as clean SystemExits below; a Hydra
    # compose error (e.g. a typo'd group in a COMPOSABLE file's `defaults:`)
    # keeps its own readable traceback.
    cfg = _load_config_file(inv.config)
    play = inv.verb == "play"
    try:
      if "defaults" in cfg:
        if play:
          print(
            "[INFO] play --config: composable file, composed with the `play` "
            "overlay (no episode time limit, no observation noise, no "
            "perturbation)."
          )
        return compose_run_from_file(
          inv.config, play=play, overlays=inv.overlays, overrides=inv.overrides
        )
      if play:
        print(
          "[INFO] play --config: fully resolved tree, played exactly as written "
          "(training conditions: episode time limit, observation noise, the "
          "randomization events). For the evaluation conditions (no time limit, "
          "no observation noise, no randomization beyond base_com and "
          "foot_friction) add --overlay play."
        )
      return load_raw_run(
        inv.config, play=play, overlays=inv.overlays, overrides=inv.overrides
      )
    except (FileNotFoundError, ValueError) as exc:
      raise SystemExit(f"--config: {exc}") from None
  assert inv.ckpt is not None
  return load_frozen_run(inv.ckpt, overlays=inv.overlays, overrides=inv.overrides)


def _write_dry_run(inv: Invocation, bundle: RunBundle, report: list[str]) -> None:
  # Default to /tmp (out of the repo, predictable to read); --dry-run-dir overrides.
  out_dir = Path(inv.dry_run_dir) if inv.dry_run_dir else Path("/tmp")
  out_dir.mkdir(parents=True, exist_ok=True)
  out = out_dir / "resolved_config.yaml"
  out.write_text(bundle.resolved_cfg_yaml or "")
  print(f"[INFO] Wrote resolved config to {out.resolve()}")
  if bundle.resolved_eval_cfg_yaml is not None:
    eval_out = out_dir / "resolved_eval_config.yaml"
    eval_out.write_text(bundle.resolved_eval_cfg_yaml)
    print(f"[INFO] Wrote resolved eval config to {eval_out.resolve()}")
  session_out = out_dir / "resolved_session.yaml"
  session_out.write_text(session_record_yaml(bundle.exec.session))
  print(f"[INFO] Wrote resolved session to {session_out.resolve()}")
  for line in report:
    print(line)
  print(f"[INFO] --dry-run: {inv.verb} assembly complete; exiting before build.")


def main() -> None:
  inv = _parse_cli(sys.argv[1:])
  report: list[str] = []

  if inv.ckpt is not None:
    # Accept wandb run-path specs everywhere a checkpoint anchors a verb;
    # resolution (download/cache) happens once, before the frozen metadata is
    # read.
    inv.ckpt = str(resolve_checkpoint_spec(inv.ckpt))

  # Verb-specific pre-assembly validation: resume's allowed-diff spec runs on
  # the frozen TEXT (cheap, fails before any env build) and doubles as the
  # dry-run report.
  if inv.verb == "resume":
    assert inv.ckpt is not None
    diff = resume_delta_paths(inv.ckpt, overlays=inv.overlays, overrides=inv.overrides)
    enforce_resume_allowlist(diff)
    report.append(
      "[INFO] resume allowed-diff: "
      + (", ".join(diff) if diff else "(none -- exact pickup)")
    )

  bundle = _assemble(inv)
  exec_cfg = bundle.exec

  # On the other frozen anchors (--ckpt without --config) the same text diff
  # guards the recipe overlays: they apply at compose time only, so a delta
  # that changes them could never take effect. Resume's allowlist covers it.
  if (
    inv.verb != "resume"
    and inv.ckpt is not None
    and inv.config is None
    and (inv.overlays or inv.overrides)
  ):
    reject_recipe_overlay_deltas(
      resume_delta_paths(inv.ckpt, overlays=inv.overlays, overrides=inv.overrides)
    )
  if inv.verb == "eval":
    # Also checked before the --dry-run exit, so a dry run rejects an unknown
    # recipe as the real run would.
    select_eval_recipes(bundle.resolved_eval_cfg_yaml, inv.recipe or "default")

  # A checkpoint anchor is both the config source and the policy source: for
  # play, default `session.checkpoint_file` to it (overridable) so the trained
  # policy loads into the rebuilt env.
  if (
    inv.verb == "play"
    and inv.ckpt is not None
    and exec_cfg.session.checkpoint_file is None
  ):
    exec_cfg.session.checkpoint_file = inv.ckpt

  if inv.verb == "fork":
    assert inv.ckpt is not None
    # Honor the --dry-run contract ("exit before any build / weights load"):
    # only resolve the load spec when the checkpoint already exists -- a
    # recompose fork (`fork --config`) may legitimately pre-flight
    # against a checkpoint that is still training. When it does exist, the
    # read shares the cached metadata slice with the anchor load (no extra
    # deserialize).
    if Path(inv.ckpt).exists():
      load_cfg = resolve_fork_load_cfg(exec_cfg)
      report.append(f"[INFO] fork load_cfg: {load_cfg}")
    else:
      report.append(
        "[INFO] fork load_cfg: (resolved at run time -- checkpoint not present yet)"
      )

  if inv.dry_run:
    _write_dry_run(inv, bundle, report)
    return

  deltas = {"overlays": list(inv.overlays), "overrides": list(inv.overrides)}
  if inv.verb == "train":
    launch_training(
      bundle.env_cfg,
      bundle.runner_cfg,
      bundle.runner_cls,
      exec_cfg,
      bundle.resolved_cfg_yaml,
      resolved_eval_cfg_yaml=bundle.resolved_eval_cfg_yaml,
      composition=bundle.composition,
    )
  elif inv.verb == "resume":
    assert inv.ckpt is not None
    run_resume(
      bundle.env_cfg,
      bundle.runner_cfg,
      bundle.runner_cls,
      exec_cfg,
      checkpoint_file=inv.ckpt,
      resolved_cfg_yaml=bundle.resolved_cfg_yaml,
      resolved_eval_cfg_yaml=bundle.resolved_eval_cfg_yaml,
      deltas=deltas,
    )
  elif inv.verb == "fork":
    assert inv.ckpt is not None
    run_fork(
      bundle.env_cfg,
      bundle.runner_cfg,
      bundle.runner_cls,
      exec_cfg,
      checkpoint_file=inv.ckpt,
      resolved_cfg_yaml=bundle.resolved_cfg_yaml,
      resolved_eval_cfg_yaml=bundle.resolved_eval_cfg_yaml,
      deltas=deltas,
      composition=bundle.composition,
    )
  elif inv.verb == "eval":
    assert inv.ckpt is not None
    run_eval(
      bundle.env_cfg,
      bundle.runner_cfg,
      bundle.runner_cls,
      exec_cfg,
      resolved_eval_cfg_yaml=bundle.resolved_eval_cfg_yaml,
      checkpoint_file=inv.ckpt,
      recipe=inv.recipe or "default",
      deviations=deltas,
    )
  else:
    # Name the policy source: a --config file can bury `session.checkpoint_file`
    # in its body, where nothing else would surface it (the zero-policy fallback
    # already prints its own INFO inside run_play).
    if exec_cfg.session.checkpoint_file is not None:
      print(f"[INFO] play policy source: {exec_cfg.session.checkpoint_file}")
    run_play(
      bundle.env_cfg,
      bundle.runner_cfg,
      bundle.runner_cls,
      exec_cfg,
      resolved_cfg_yaml=bundle.resolved_cfg_yaml,
    )


if __name__ == "__main__":
  main()
