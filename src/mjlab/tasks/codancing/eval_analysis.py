"""Offline analyzer for the long-format per-(step, env) eval CSV.

CSV-first: the runtime records, this module computes. Settle masking, flat
and per-clip summaries, termination cause splits, seen/unseen splits and the
pool-coverage check all live here, so a session needs no in-process summary
machinery.

CLI:
  uv run python -m mjlab.tasks.codancing.eval_analysis <...-rollout-envs.csv> \\
    [--settle 2] [--pool <superset manifest>] [--seen <own pool manifest>] \\
    [--recipe NAME] [--out <dir>]

Outputs into --out (default: the CSV's directory): `<stem>.summary.csv` in the
established `recipe, clip, <metric>/{mean,max,min,count}` schema (an `ALL` row
plus one row per clip), `<stem>.phase.csv` (the same schema binned by motion
phase instead of clip, the RSI probe; only for records carrying `clip_len_s`)
and `<stem>.analysis.json` with the global, per-clip and per-phase summaries,
per-cause termination counts, the seen/unseen aggregate, and the coverage
report. A pool clip with zero rows fails loudly (exit 2).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_env_rows(path: Path) -> dict[str, Any]:
  """Read the long CSV into columns: ints, floats, bools and the clip strings."""
  with Path(path).open(newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
  if not rows:
    raise ValueError(f"{path}: no data rows.")
  cols: dict[str, Any] = {}
  names = rows[0].keys()
  for name in names:
    raw = [r[name] for r in rows]
    if name in ("step", "env"):
      cols[name] = np.array([int(v) for v in raw])
    elif name == "reset" or name.startswith("term_"):
      cols[name] = np.array([v == "1" for v in raw])
    elif name == "clip":
      cols[name] = raw
    else:
      cols[name] = np.array([float(v) if v else np.nan for v in raw])
  return cols


def settle_valid(
  step: np.ndarray, env: np.ndarray, reset: np.ndarray, settle: int
) -> np.ndarray:
  """Per-row validity: drop the first `settle` rows after each env reset.

  Session start counts as a reset (the recorder's convention). settle=0 keeps
  every row.
  """
  valid = np.ones(len(step), dtype=bool)
  if settle <= 0:
    return valid
  for env_id in np.unique(env):
    idx = np.where(env == env_id)[0]
    idx = idx[np.argsort(step[idx])]
    countdown = settle
    for i in idx:
      if reset[i]:
        countdown = settle
      valid[i] = countdown <= 0
      countdown = max(countdown - 1, 0)
  return valid


def _stats(values: np.ndarray) -> dict[str, float]:
  values = values[~np.isnan(values)]
  if len(values) == 0:
    return {}
  return {
    "mean": float(values.mean()),
    "max": float(values.max()),
    "min": float(values.min()),
    "count": float(len(values)),
  }


def summarize(
  cols: dict[str, Any], valid: np.ndarray
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]]:
  """Global and per-clip {mean,max,min,count} for every metric column."""
  metric_names = [n for n in cols if n.startswith("metric/")]
  clips = np.array(cols["clip"])
  flat: dict[str, dict[str, float]] = {}
  per_clip: dict[str, dict[str, dict[str, float]]] = {}
  for name in metric_names:
    values = cols[name]
    stats = _stats(values[valid])
    if stats:
      flat[name.removeprefix("metric/")] = stats
    buckets: dict[str, dict[str, float]] = {}
    for clip in sorted(set(cols["clip"])):
      mask = valid & (clips == clip)
      stats = _stats(values[mask])
      if stats:
        buckets[clip] = stats
    if buckets:
      per_clip[name.removeprefix("metric/")] = buckets
  return flat, per_clip


def termination_causes(
  cols: dict[str, Any],
) -> dict[str, dict[str, int]]:
  """Per-cause done counts, overall and per clip (every row counts once)."""
  clips = np.array(cols["clip"])
  out: dict[str, dict[str, int]] = {}
  for name in [n for n in cols if n.startswith("term_")]:
    dones = cols[name]
    entry = {"ALL": int(dones.sum())}
    for clip in sorted(set(cols["clip"])):
      count = int(dones[clips == clip].sum())
      if count:
        entry[clip] = count
    out[name.removeprefix("term_")] = entry
  return out


def pool_clip_names(manifest: Path) -> list[str]:
  """Clip names of a clip manifest (``configs/clip_manifests/*.yaml``, a list)."""
  return [entry["name"] for entry in yaml.safe_load(Path(manifest).read_text())]


def coverage_report(cols: dict[str, Any], pool: list[str]) -> dict[str, Any]:
  seen = set(cols["clip"])
  missing = sorted(set(pool) - seen)
  extra = sorted(seen - set(pool))
  return {
    "pool_size": len(pool),
    "visited": len(seen & set(pool)),
    "missing": missing,
    "extra": extra,
  }


def seen_unseen_split(
  per_clip: dict[str, dict[str, dict[str, float]]], seen_names: list[str]
) -> dict[str, dict[str, dict[str, float]]]:
  """Aggregate the per-clip means into seen vs unseen groups (count-weighted)."""
  seen_set = set(seen_names)
  out: dict[str, dict[str, dict[str, float]]] = {}
  for metric, buckets in per_clip.items():
    groups: dict[str, list[tuple[float, float]]] = {"seen": [], "unseen": []}
    for clip, stats in buckets.items():
      groups["seen" if clip in seen_set else "unseen"].append(
        (stats["mean"], stats["count"])
      )
    entry: dict[str, dict[str, float]] = {}
    for group, pairs in groups.items():
      if not pairs:
        continue
      weights = sum(c for _, c in pairs)
      entry[group] = {
        "mean": sum(m * c for m, c in pairs) / weights,
        "clips": float(len(pairs)),
        "count": float(weights),
      }
    if entry:
      out[metric] = entry
  return out


def phase_binned_summary(
  cols: dict[str, Any], valid: np.ndarray, num_bins: int = 10
) -> dict[str, dict[str, dict[str, float]]]:
  """Per-metric {phase bin -> stats} over motion phase (clip_time / clip_len_s).

  The RSI probe: a policy that only ever starts episodes at t=0 reaches late
  phases only by surviving into them, so its tracking quality tends to decay
  with phase; binning the same metric columns by phase shows that shape
  directly, and a policy trained with RSI resets should flatten it. Bin
  labels are the phase interval starts ("0.0" covers [0.0, 0.1) at 10 bins).
  Records without a `clip_len_s` column return {} (the consumer omits the
  block).
  """
  if "clip_len_s" not in cols:
    return {}
  lens = np.asarray(cols["clip_len_s"], dtype=float)
  times = np.asarray(cols["clip_time"], dtype=float)
  with np.errstate(invalid="ignore", divide="ignore"):
    phase = times / np.where(lens > 0, lens, np.nan)
  bins = np.floor(np.clip(phase, 0.0, 1.0 - 1e-9) * num_bins)
  binnable = ~np.isnan(phase)
  out: dict[str, dict[str, dict[str, float]]] = {}
  for name in [n for n in cols if n.startswith("metric/")]:
    values = cols[name]
    buckets: dict[str, dict[str, float]] = {}
    for k in range(num_bins):
      stats = _stats(values[valid & binnable & (bins == k)])
      if stats:
        buckets[f"{k / num_bins:.1f}"] = stats
    if buckets:
      out[name.removeprefix("metric/")] = buckets
  return out


def write_phase_csv(
  out_path: Path,
  recipe: str,
  phase: dict[str, dict[str, dict[str, float]]],
) -> None:
  """Phase twin of the summary schema: recipe, phase, <metric>/{stat} columns."""
  metrics = sorted(phase)
  stats = ("mean", "max", "min", "count")
  header = ["recipe", "phase"] + [f"{m}/{s}" for m in metrics for s in stats]
  bins = sorted({b for buckets in phase.values() for b in buckets})
  with out_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for b in bins:
      row = [recipe, b]
      for m in metrics:
        row += [f"{phase.get(m, {}).get(b, {}).get(s, '')}" for s in stats]
      writer.writerow(row)


def write_summary_csv(
  out_path: Path,
  recipe: str,
  flat: dict[str, dict[str, float]],
  per_clip: dict[str, dict[str, dict[str, float]]],
) -> None:
  """The established summary schema: recipe, clip, <metric>/{stat} columns."""
  metrics = sorted(set(flat) | set(per_clip))
  stats = ("mean", "max", "min", "count")
  header = ["recipe", "clip"] + [f"{m}/{s}" for m in metrics for s in stats]
  clips = sorted({c for buckets in per_clip.values() for c in buckets})
  with out_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    row = ["", "ALL"] if not recipe else [recipe, "ALL"]
    for m in metrics:
      row += [f"{flat.get(m, {}).get(s, '')}" for s in stats]
    writer.writerow(row)
    for clip in clips:
      row = [recipe, clip]
      for m in metrics:
        row += [f"{per_clip.get(m, {}).get(clip, {}).get(s, '')}" for s in stats]
      writer.writerow(row)


def analyze(
  csv_path: Path,
  *,
  settle: int = 0,
  pool: Path | None = None,
  seen: Path | None = None,
  recipe: str = "",
  out_dir: Path | None = None,
) -> dict[str, Any]:
  cols = load_env_rows(csv_path)
  valid = settle_valid(cols["step"], cols["env"], cols["reset"], settle)
  flat, per_clip = summarize(cols, valid)
  result: dict[str, Any] = {
    "source": str(csv_path),
    "settle": settle,
    "rows": int(len(cols["step"])),
    "rows_valid": int(valid.sum()),
    "summary": flat,
    "per_clip_summary": per_clip,
    "termination_causes": termination_causes(cols),
  }
  phase = phase_binned_summary(cols, valid)
  if phase:
    result["phase_summary"] = phase
  if seen is not None:
    result["seen_unseen"] = seen_unseen_split(per_clip, pool_clip_names(seen))
  if pool is not None:
    result["coverage"] = coverage_report(cols, pool_clip_names(pool))
  out = out_dir or csv_path.parent
  out.mkdir(parents=True, exist_ok=True)
  stem = csv_path.stem
  write_summary_csv(out / f"{stem}.summary.csv", recipe, flat, per_clip)
  if phase:
    write_phase_csv(out / f"{stem}.phase.csv", recipe, phase)
  (out / f"{stem}.analysis.json").write_text(json.dumps(result, indent=2))
  return result


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("csv", type=Path)
  parser.add_argument("--settle", type=int, default=0)
  parser.add_argument(
    "--pool",
    type=Path,
    default=None,
    help="superset clip manifest; missing clips fail (exit 2)",
  )
  parser.add_argument(
    "--seen",
    type=Path,
    default=None,
    help="own-pool clip manifest for the seen/unseen split",
  )
  parser.add_argument("--recipe", default="")
  parser.add_argument("--out", type=Path, default=None)
  args = parser.parse_args(argv)
  result = analyze(
    args.csv,
    settle=args.settle,
    pool=args.pool,
    seen=args.seen,
    recipe=args.recipe,
    out_dir=args.out,
  )
  print(
    f"rows {result['rows']} (valid {result['rows_valid']}), "
    f"metrics {len(result['summary'])}, "
    f"clips {len(next(iter(result['per_clip_summary'].values()), {}))}"
  )
  coverage = result.get("coverage")
  if coverage and coverage["missing"]:
    print(
      f"COVERAGE FAILURE: {len(coverage['missing'])} pool clips have no rows: "
      + ", ".join(coverage["missing"][:8])
      + ("..." if len(coverage["missing"]) > 8 else "")
    )
    return 2
  return 0


if __name__ == "__main__":
  sys.exit(main())
