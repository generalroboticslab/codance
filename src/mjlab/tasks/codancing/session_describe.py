"""Generated natural-language description of an ENTIRE play session.

The whole-session sibling of `viz_describe`: derived from the built env cfg,
the session cfg and the motion manifest at session start, written into the
manifest (`session_description`) and printed. The line formats carry stable
markers (`face=adapted`, `force_scale=1.0`, `ref_face=free`, `clip_end=chain`,
"truncates the episode" vs "resamples in-episode"), so a script can grep them
to check the executed config against the description instead of trusting
either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mjlab.tasks.codancing.viz_describe import _get, is_twin_pool


def _pool_lines(motions: list[Any]) -> list[str]:
  if not motions:
    return ["pool: none (empty manifest)"]
  n = len(motions)
  names = [str(_get(m, "name", "")) for m in motions]
  ml = sum(1 for name in names if name.startswith("ml_"))
  zw = sum(1 for name in names if name.startswith("zw_"))
  twin = is_twin_pool(motions)
  face = (
    "face=ORIGINAL (served == free: the pool tracks the unforced clip)"
    if twin
    else "face=adapted"
  )
  # A pool's contact files sit in its folder, data/compliant/rc/<pool>/.
  rcs = sorted(
    {
      Path(str(_get(m, "contact_file"))).parent.name
      for m in motions
      if _get(m, "contact_file")
    }
  )
  weights = {float(_get(m, "weight", 1.0)) for m in motions}
  weight_note = "uniform weights" if len(weights) <= 1 else "NON-uniform weights"
  return [
    f"pool: {n} entries ({ml} force + {zw} zero-wrench), {face}, "
    f"rc={rcs or ['?']}, {weight_note}"
  ]


def _cursor_lines(cmd_cfg: Any, terminations: Any) -> list[str]:
  cursor = _get(cmd_cfg, "motion_cursor")
  source = _get(cursor, "source")
  lines = []
  # `<moment>=<mode>` is a stable marker: keep the format so a grep for
  # `clip_end=chain` still matches.
  for moment in ("episode_reset", "clip_end"):
    sampling = _get(source, moment)
    lines.append(
      f"cursor {moment}={_get(cursor, moment)} "
      f"clip_selection={_get(sampling, 'clip_selection')}"
    )
  if _get(cursor, "episode_reset") == "rsi":
    rsi = _get(cursor, "rsi")
    sampler = dict(_get(rsi, "sampler", None) or {})
    lines.append(f"rsi: start_prob={_get(rsi, 'start_prob')} sampler={sampler}")
  clip_end_active = _get(terminations or {}, "motion_clip_end") is not None
  lines.append(
    "clip end: truncates the episode (motion_clip_end active, time_out flag)"
    if clip_end_active
    else "clip end: resamples in-episode (motion_clip_end absent/nulled)"
  )
  return lines


def _event_lines(events: Any) -> list[str]:
  lines = []
  for name, term in (events or {}).items():
    if term is None:
      lines.append(f"event {name}: NULLED")
      continue
    params = _get(term, "params") or {}
    mode = _get(term, "mode", "?")
    if "force_response" in params:
      lines.append(
        f"event {name}({mode}): force_response={params.get('force_response')} "
        f"force_scale={params.get('force_scale')} "
        f"reanchor_mode={params.get('reanchor_mode')} "
        f"reanchor_rotation={params.get('reanchor_rotation')} "
        f"anchor_body={params.get('anchor_body')!r}"
      )
    else:
      keys = ", ".join(sorted(k for k in params if k not in ("asset_cfg",)))
      lines.append(f"event {name}({mode}): params [{keys}]")
  return lines or ["events: none"]


def _termination_lines(terminations: Any) -> list[str]:
  active, nulled = [], []
  for name, term in (terminations or {}).items():
    if term is None:
      nulled.append(name)
      continue
    flag = " [time_out]" if _get(term, "time_out", False) else ""
    threshold = (_get(term, "params") or {}).get("threshold")
    threshold_note = f" threshold={threshold}" if threshold is not None else ""
    active.append(f"{name}{flag}{threshold_note}")
  lines = [f"terminations: {', '.join(active) if active else 'NONE'}"]
  if nulled:
    lines.append(f"terminations nulled: {', '.join(sorted(nulled))}")
  return lines


def _actor_view_line(observations: Any) -> str:
  actor = _get(observations or {}, "actor")
  terms = _get(actor, "terms") or {}
  faces = []
  for name, term in terms.items():
    params = _get(term, "params") or {}
    if "ref_face" in params:
      faces.append(f"{name}=ref_face={params['ref_face']}")
  if not faces:
    return "actor reference view: ref_face=served (term defaults; no explicit params)"
  return "actor reference view: " + ", ".join(sorted(faces))


def _metric_lines(cmd_cfg: Any) -> list[str]:
  """The recorded metric families, derived from the SAME function that
  enumerates the keys (`iter_metric_suite_keys`), so the count is exact."""
  from mjlab.tasks.codancing.mdp.metrics import iter_metric_suite_keys

  suites = tuple(_get(cmd_cfg, "metric_suites", ()) or ())
  groups = tuple(_get(cmd_cfg, "metric_body_groups", ()) or ())
  frame = _get(cmd_cfg, "metric_reference_frame", "both")
  pairs: tuple[tuple[str, str], ...] = tuple(
    (str(p[0]), str(p[1]))
    for p in (_get(cmd_cfg, "metric_compare", ()) or (("served", "live"),))
  )
  try:
    total = len(iter_metric_suite_keys(suites, frame, groups, pairs))
  except Exception:
    total = 0
  lines = [f"metrics: {total} keys per env per step"]
  if "robot_ref_tracking" in suites:
    for a, b in pairs:
      frames = (
        "global(pos, rot) + anchor_aligned(pos, rot)"
        if (a, b) in (("served", "live"), ("free", "live"))
        else "global(pos, rot)"
      )
      lines.append(f"  robot_ref {a}_vs_{b} x {groups}: {frames}")
  if "compliance" in suites:
    lines.append(
      "  compliance: applied_force_mag, desired_force_mag, force_error "
      "(per env, masked over the active contact slots)"
    )
  return lines


def _downstream_lines(session: Any, cmd_cfg: Any) -> list[str]:
  """What can be computed/plotted from THIS session's artifacts (config-gated)."""
  metric = _get(session, "metric")
  video = _get(session, "video")
  pairs = {tuple(p) for p in (_get(cmd_cfg, "metric_compare", ()) or ())}
  lines = ["downstream, from this session's files:"]
  if _get(metric, "enabled"):
    lines.append(
      "  analyzer (rollout-envs.csv): settle-masked flat + per-clip "
      "summary.csv, termination-cause counts per clip, coverage check vs a "
      "pool list, seen/unseen split vs a seen list"
    )
    lines.append("  offline reads: falls per 1000 steps (term_* columns)")
    if ("served", "free") in pairs:
      lines.append("  per-step yield curve (served_vs_free keys)")
    if ("free", "live") in pairs:
      lines.append("  vs-original tracking without a pool swap (free_vs_live keys)")
  if _get(video, "enabled"):
    lines.append("  video: env-0 mp4 (the ghost IS the metric's reference)")
  return lines


def describe_session(
  *,
  env_cfg: Any,
  session: Any,
  motions: list[Any],
  agent_type: str,
  checkpoint_spec: str | None,
  visualization: str | None = None,
) -> str:
  """Deterministic whole-session description; sections in fixed order.

  ``visualization`` (the `viz_describe` text) is appended as the final
  section, so ONE generated text describes the entire session and one
  manifest field carries it.
  """
  cmd_cfg = _get(_get(env_cfg, "commands") or {}, "codancing")
  metric = _get(session, "metric")
  lines: list[str] = []
  lines.append(
    f"policy: agent={agent_type}, checkpoint={checkpoint_spec or 'none'}, "
    f"deterministic mean action; env seed={_get(env_cfg, 'seed')}"
  )
  lines += _pool_lines(motions)
  lines += _cursor_lines(cmd_cfg, _get(env_cfg, "terminations"))
  lines += _event_lines(_get(env_cfg, "events"))
  lines += _termination_lines(_get(env_cfg, "terminations"))
  lines.append(_actor_view_line(_get(env_cfg, "observations")))
  lines.append(
    f"measurement: suites={tuple(_get(cmd_cfg, 'metric_suites', ()) or ())} "
    f"pairs={tuple(tuple(p) for p in _get(cmd_cfg, 'metric_compare', ()) or ())} "
    f"groups={tuple(_get(cmd_cfg, 'metric_body_groups', ()) or ())} "
    f"frame={_get(cmd_cfg, 'metric_reference_frame')}"
  )
  lines.append(
    f"recording: enabled={_get(metric, 'enabled')} "
    f"steps={_get(metric, 'num_steps')} "
    f"num_envs={_get(_get(env_cfg, 'scene'), 'num_envs')}"
  )
  lines += _metric_lines(cmd_cfg)
  lines += _downstream_lines(session, cmd_cfg)
  if visualization:
    lines.append("visualization:")
    lines += [f"  {line}" for line in visualization.splitlines()]
  return "\n".join(lines)
