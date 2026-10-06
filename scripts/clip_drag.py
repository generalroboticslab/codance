"""The clips' own force events read as drags: realized stiffness per event.

Every augmented clip carries force events (a ramp, a hold, a release on one
wrist, with a told stiffness and an environment spring drawn per event). The
per-clip eval records, per env and step, each contact slot's applied force
and the wrist's displacement along it away from the FREE reference (the
un-adapted trajectory: the whole yield the pull produced) and away from the
SERVED reference (the adapted one: what tracking left over). This pass slices
those columns into events and reads, over the last `--read-s` seconds of
each hold, the compliance the data asked for: the policy's arm yield against
the reference's own arm yield (served minus free, same frame), the force error
against the data force, and, as a secondary number, K_eff = force / yield
against the told stiffness. Events are grouped by seen / unseen clip, single or
both wrists (the other slot in an event at the same time), told stiffness and
force.

  summarize  --family waltz --policy decoupled --pool own|full [--seed N] [--obs-noise x]
             reads the envs table the per-clip summary came from
             (logs/clip_metrics/<family>/<policy>...source.txt), writes
             logs/clip_drag/<family>/<policy>[.full].s<N>[.noise<x>].events.csv
             (one row per event) and .summary.csv (the groups).
  figures    same selectors: five figures from the events file, beside it:
             realized vs told stiffness (log-log, fit and identity), the ratio
             per told bin for seen and unseen clips, the yield anatomy (asked
             vs realized, and the lag behind the adapted reference) and the
             ratio against the other knobs at told 150 to 400.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_metrics import (  # noqa: E402
  REPO,
  _families,
  own_manifest,
  summary_path,
  trackable_names,
)

OUT = REPO / "logs/clip_drag"
DT = 0.02  # 50 Hz control step
SLOTS = 2
FIELDS = ("in_event", "phase", "told", "kenv", "force", "yield_free", "yield_served")
TOLD_BINS = ((10, 50), (50, 150), (150, 400), (400, 1001))
FORCE_BINS = ((0.5, 5), (5, 15), (15, 40), (40, 1e9))


def _load_table(path: Path) -> dict[int, dict[str, np.ndarray]]:
  """{env: {column: array over steps}} for the columns this pass reads."""
  with path.open() as fh:
    reader = csv.reader(fh)
    header = next(reader)
    want = {"step", "env", "reset", "clip", "metric/compliance_force_error"} | {
      f"metric/clip_drag_k{i}_{f}" for i in range(SLOTS) for f in FIELDS
    }
    idx = {name: j for j, name in enumerate(header) if name in want}
    missing = want - set(idx)
    if missing:
      sys.exit(
        f"{path.name}: no clip_drag columns ({sorted(missing)[:3]}...): re-run the eval"
      )
    per_env: dict[int, dict[str, list]] = {}
    for row in reader:
      env = int(row[idx["env"]])
      d = per_env.setdefault(env, {k: [] for k in idx})
      for k, j in idx.items():
        d[k].append(row[j])
  out: dict[int, dict[str, np.ndarray]] = {}
  for env, d in per_env.items():
    arrs = {
      k: (np.array(v) if k == "clip" else np.array(v, dtype=float))
      for k, v in d.items()
    }
    order = np.argsort(arrs["step"], kind="stable")
    out[env] = {k: v[order] for k, v in arrs.items()}
  return out


def _events(
  env_rows: dict[str, np.ndarray], read_rows: int, min_rows: int
) -> list[dict]:
  """One record per force event of either slot: K_eff over the hold's last
  `read_rows` rows, skipping events cut by a reset or with too short a hold."""
  reset = env_rows["reset"] > 0
  clip = str(env_rows["clip"][0])
  records = []
  for i in range(SLOTS):

    def col(f: str, slot: int = i) -> np.ndarray:
      return env_rows[f"metric/clip_drag_k{slot}_{f}"]

    other = env_rows[f"metric/clip_drag_k{1 - i}_in_event"]
    active = col("in_event") > 0.5
    if not active.any():
      continue
    # Segments of consecutive in-event rows, split where the env reset.
    edges = np.flatnonzero(np.diff(np.concatenate(([0], active.astype(int), [0]))))
    for ordinal, (start, end) in enumerate(zip(edges[::2], edges[1::2], strict=True)):
      if reset[start + 1 : end].any():
        continue
      hold = np.flatnonzero(col("phase")[start:end] == 2) + start
      if hold.size < min_rows:
        continue
      win = hold[-read_rows:]
      force = float(col("force")[win].mean())
      yf = float(col("yield_free")[win].mean())
      ys = float(col("yield_served")[win].mean())
      told = float(col("told")[win].mean())
      if force < 0.5 or told <= 0:
        continue
      k_free = force / yf if abs(yf) >= 5e-3 else float("nan")
      k_served = force / ys if abs(ys) >= 5e-3 else float("nan")
      # The reference's own arm yield along the force (served minus free) and
      # how much of it the policy reproduced: the compliance the data asked for.
      ref = yf - ys
      track = yf / ref if ref >= 5e-3 else float("nan")
      records.append(
        {
          "clip": clip,
          "slot": i,
          "event": ordinal,
          "hold_rows": int(hold.size),
          "told": told,
          "kenv": float(col("kenv")[win].mean()),
          "force_n": force,
          "yield_free_cm": 100 * yf,
          "yield_served_cm": 100 * ys,
          "ref_yield_cm": 100 * ref,
          "track_ratio": track,
          "abs_ln_track": abs(math.log(track))
          if np.isfinite(track) and track > 0
          else float("nan"),
          "force_error_n": float(env_rows["metric/compliance_force_error"][win].mean()),
          "k_eff_free": k_free,
          "k_eff_served": k_served,
          "ratio_free": k_free / told,
          "ratio_served": k_served / told,
          "abs_ln_ratio_free": abs(math.log(k_free / told))
          if k_free > 0
          else float("nan"),
          "both": bool(other[win].mean() > 0.5),
        }
      )
  return records


def _bin(v: float, bins) -> str:
  for lo, hi in bins:
    if lo <= v < hi:
      return f"{lo:g}-{hi:g}" if hi < 1e8 else f"{lo:g}+"
  return "out"


STAT_KEYS = (
  "n",
  "median_track",
  "median_abs_ln_track",
  "frac_track_within_25pct",
  "median_force_error_n",
  "median_ratio_free",
  "slope",
)


def _stats(rows: list[dict]) -> dict[str, float]:
  """The group's compliance numbers: tracking of the reference's yield (the
  primary), the force error, and the secondary realized-over-told read."""
  r = [x for x in rows if np.isfinite(x["track_ratio"]) and x["track_ratio"] > 0]
  n = len(r)
  if n == 0:
    return {"n": 0}
  track = np.array([x["track_ratio"] for x in r])
  errs = np.abs(np.log(track))
  out = {
    "n": n,
    "median_track": float(np.median(track)),
    "median_abs_ln_track": float(np.median(errs)),
    "frac_track_within_25pct": float((errs < 0.22).mean()),
    "median_force_error_n": float(np.median([x["force_error_n"] for x in r])),
  }
  free = [x for x in r if np.isfinite(x["ratio_free"]) and x["ratio_free"] > 0]
  if free:
    out["median_ratio_free"] = float(np.median([x["ratio_free"] for x in free]))
    tolds = np.array([x["told"] for x in free])
    if len(set(np.round(tolds, 3))) >= 2:
      xs, ys = np.log(tolds), np.log([x["k_eff_free"] for x in free])
      out["slope"] = float(
        ((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum()
      )
  return out


def cmd_summarize(args: argparse.Namespace) -> None:
  fam = _families(args)[0]
  summary = summary_path(fam, args.pool, args.seed, args.obs_noise)
  source = summary.with_suffix("").with_suffix(".source.txt")
  if not source.exists():
    sys.exit(f"{source.relative_to(REPO)} missing: run scripts/clip_metrics.py first")
  table = REPO / source.read_text().strip()
  print(f"events from {table.relative_to(REPO)}")
  seen = set(trackable_names(own_manifest(fam)))
  read_rows = max(1, round(args.read_s / DT))
  records: list[dict] = []
  for env, rows in sorted(_load_table(table).items()):
    for rec in _events(rows, read_rows, args.min_hold_rows):
      rec["env"] = env
      rec["seen"] = rec["clip"] in seen
      records.append(rec)
  if not records:
    sys.exit("no events read")
  out_dir = OUT / fam.name
  out_dir.mkdir(parents=True, exist_ok=True)
  stem = summary.name.removesuffix(".summary.csv")
  cols = [
    "env", "clip", "seen", "slot", "event", "both", "hold_rows", "told", "kenv", "force_n",
    "yield_free_cm", "yield_served_cm", "ref_yield_cm", "track_ratio", "abs_ln_track",
    "force_error_n", "k_eff_free", "k_eff_served", "ratio_free", "ratio_served",
    "abs_ln_ratio_free",
  ]  # fmt: skip
  events_csv = out_dir / f"{stem}.events.csv"
  with events_csv.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for rec in records:
      w.writerow(
        {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in rec.items()}
      )
  print(f"{len(records)} events -> {events_csv.relative_to(REPO)}")
  # Groups: seen / unseen x single / both x (all, told bin, force bin).
  summary_csv = out_dir / f"{stem}.summary.csv"
  groups: list = []
  for seen_flag in (True, False):
    groups.append(
      ("all", "any", seen_flag, None, [r for r in records if r["seen"] == seen_flag])
    )
    for both in (False, True):
      base = [r for r in records if r["seen"] == seen_flag and r["both"] == both]
      if not base:
        continue
      groups.append(("all", "", seen_flag, both, base))
      for lo, hi in TOLD_BINS:
        groups.append(
          (
            "told",
            _bin(lo, TOLD_BINS),
            seen_flag,
            both,
            [r for r in base if lo <= r["told"] < hi],
          )
        )
      for lo, hi in FORCE_BINS:
        groups.append(
          (
            "force",
            _bin(lo, FORCE_BINS),
            seen_flag,
            both,
            [r for r in base if lo <= r["force_n"] < hi],
          )
        )
  keys = list(STAT_KEYS)

  def wrists_of(both):
    return "any" if both is None else ("both" if both else "single")

  with summary_csv.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["clips", "wrists", "by", "bin", *keys])
    print(
      f"{'clips':<7} {'wrists':<6} {'by':<5} {'bin':<8} {'n':>5} {'track':>6} {'|ln|':>5} "
      f"{'in25%':>6} {'Ferr N':>7} {'K/told':>7} {'slope':>6}"
    )
    for by, label, seen_flag, both, rows in groups:
      st = _stats(rows)
      w.writerow(
        ["seen" if seen_flag else "unseen", wrists_of(both), by, label]
        + [
          f"{st[k]:.3f}" if isinstance(st.get(k), float) else st.get(k, "")
          for k in keys
        ]
      )
      if st["n"]:
        print(
          f"{'seen' if seen_flag else 'unseen':<7} {wrists_of(both):<6} {by:<5} {label:<8} "
          f"{st['n']:>5} {st['median_track']:6.2f} {st['median_abs_ln_track']:5.2f} "
          f"{100 * st['frac_track_within_25pct']:5.0f}% {st['median_force_error_n']:7.2f} "
          f"{st.get('median_ratio_free', float('nan')):7.2f} {st.get('slope', float('nan')):6.2f}"
        )
  print(summary_csv.relative_to(REPO))


def _read_events(path: Path) -> list[dict]:
  rows = list(csv.DictReader(path.open()))
  for r in rows:
    for k, v in r.items():
      if k in ("clip",):
        continue
      if k in ("seen", "both"):
        r[k] = v == "True"
      else:
        r[k] = float(v)
  return rows


def cmd_figures(args: argparse.Namespace) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  from mjlab.tasks.codancing import viz_palette as pal

  m = "matplotlib"
  ink, muted = pal.hex("ink", m), pal.hex("ink_muted", m)
  c_seen, c_unseen, c_band = (
    pal.hex("blue", m),
    pal.hex("orange", m),
    pal.hex("band", m),
  )
  plt.rcParams.update({"font.size": 8, "axes.edgecolor": muted, "text.color": ink})
  fam = _families(args)[0]
  stem = summary_path(fam, args.pool, args.seed, args.obs_noise).name.removesuffix(
    ".summary.csv"
  )
  events = OUT / fam.name / f"{stem}.events.csv"
  if not events.exists():
    sys.exit(f"{events.relative_to(REPO)} missing: run summarize first")
  rows = [
    r
    for r in _read_events(events)
    if np.isfinite(r["ratio_free"]) and r["ratio_free"] > 0
  ]
  title = f"{fam.base} ({fam.base}.pt), {args.pool} pool, seed {args.seed}: the clips' own force events"
  told = np.array([r["told"] for r in rows])
  realized = np.array([r["k_eff_free"] for r in rows])
  seen = np.array([r["seen"] for r in rows])
  ratio = realized / told

  def clean(ax):
    for sp in ("top", "right"):
      ax.spines[sp].set_visible(False)

  # 1. realized vs told, log-log, with the power-law fit and the identity line.
  fig, ax = plt.subplots(figsize=(5.2, 4.2))
  xs, ys = np.log(told), np.log(realized)
  slope = float(
    ((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum()
  )
  icpt = float(ys.mean() - slope * xs.mean())
  ax.axhspan(
    47,
    116,
    color=c_band,
    alpha=0.5,
    lw=0,
    label="wrist stiffness of the bare PD arm, 47 to 116 N/m",
  )
  for flag, color, label in (
    (True, c_seen, "seen clip"),
    (False, c_unseen, "unseen clip"),
  ):
    sel = seen == flag
    if sel.any():
      ax.scatter(
        told[sel],
        realized[sel],
        s=9,
        color=color,
        alpha=0.55,
        lw=0,
        label=f"{label} (n={sel.sum()})",
      )
  grid = np.geomspace(10, 1000, 50)
  ax.plot(grid, grid, color=muted, lw=1, ls=":", label="realized = told")
  ax.plot(
    grid,
    np.exp(icpt) * grid**slope,
    color=ink,
    lw=1.4,
    label=f"fit: {np.exp(icpt):.0f} x told^{slope:.2f}",
  )
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlabel("told stiffness, N/m")
  ax.set_ylabel("realized stiffness, N/m (force / yield vs the free reference)")
  ax.set_title(title, fontsize=8)
  ax.legend(fontsize=7, frameon=False, loc="upper left")
  clean(ax)
  fig.tight_layout()
  out1 = events.with_name(f"{stem}_realized_vs_told.png")
  fig.savefig(out1, dpi=150)
  plt.close(fig)

  # 2. ratio per told bin, seen and unseen side by side.
  fig, ax = plt.subplots(figsize=(6.0, 3.8))
  positions, labels = [], []
  for j, (lo, hi) in enumerate(TOLD_BINS):
    for k, (flag, color) in enumerate(((True, c_seen), (False, c_unseen))):
      sel = (seen == flag) & (told >= lo) & (told < hi)
      if not sel.any():
        continue
      pos = j * 3 + k
      bp = ax.boxplot(
        [ratio[sel]],
        positions=[pos],
        widths=0.8,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": ink, "lw": 1.2},
        whiskerprops={"color": muted},
        capprops={"color": muted},
        boxprops={"facecolor": color, "alpha": 0.6, "lw": 0.8, "edgecolor": muted},
      )
      ax.text(
        pos,
        np.percentile(ratio[sel], 75) * 1.15,
        f"n={sel.sum()}",
        ha="center",
        fontsize=6,
        color=ink,
      )
    positions.append(j * 3 + 0.5)
    labels.append(_bin(lo, TOLD_BINS))
  ax.axhline(1.0, color=muted, lw=0.9, ls=":")
  ax.set_yscale("log")
  ax.set_xticks(positions, labels)
  ax.set_xlabel("told stiffness, N/m")
  ax.set_ylabel("realized / told")
  ax.set_title(f"{title}; seen (blue) and unseen (orange) clips", fontsize=8)
  clean(ax)
  fig.tight_layout()
  out2 = events.with_name(f"{stem}_ratio_by_told.png")
  fig.savefig(out2, dpi=150)
  plt.close(fig)

  # 3. yield anatomy: asked vs realized yield, and the lag behind the served reference.
  asked = np.array([100 * r["force_n"] / r["told"] for r in rows])
  yf = np.array([r["yield_free_cm"] for r in rows])
  ys_ = np.array([r["yield_served_cm"] for r in rows])
  fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
  ax = axes[0]
  colors = [pal.hex(n, m) for n in ("blue", "orange", "green", "pink")]
  for (lo, hi), color in zip(TOLD_BINS, colors, strict=True):
    sel = (told >= lo) & (told < hi)
    ax.scatter(
      asked[sel],
      yf[sel],
      s=9,
      color=color,
      alpha=0.6,
      lw=0,
      label=f"told {_bin(lo, TOLD_BINS)}",
    )
  top = float(np.percentile(asked, 98))
  ax.plot([0, top], [0, top], color=muted, lw=1, ls=":")
  ax.set_xlim(0, top)
  ax.set_ylim(-5, top)
  ax.set_xlabel("asked yield force / told, cm")
  ax.set_ylabel("realized yield vs the free reference, cm")
  ax.legend(fontsize=7, frameon=False)
  clean(ax)
  ax = axes[1]
  ax.hist(ys_, bins=40, range=(-8, 8), color=c_seen, alpha=0.8)
  ax.axvline(0, color=muted, lw=0.9, ls=":")
  ax.axvline(
    float(np.median(ys_)), color=ink, lw=1.2, label=f"median {np.median(ys_):+.1f} cm"
  )
  ax.set_xlabel("yield vs the served (adapted) reference, cm; negative = lags it")
  ax.set_ylabel("events")
  ax.legend(fontsize=7, frameon=False)
  clean(ax)
  fig.suptitle(title, fontsize=8)
  fig.tight_layout()
  out3 = events.with_name(f"{stem}_yield_anatomy.png")
  fig.savefig(out3, dpi=150)
  plt.close(fig)

  # 4. the other knobs at told 150 to 400: spring, force, wrist, event order.
  mid = (told >= 150) & (told < 400)
  kenv = np.array([r["kenv"] for r in rows])
  force = np.array([r["force_n"] for r in rows])
  slot = np.array([r["slot"] for r in rows])
  ordinal = np.array([r["event"] for r in rows])
  both = np.array([r["both"] for r in rows])
  panels = [
    (
      "environment spring, N/m",
      [
        (f"{lo:g}-{hi:g}" if hi < 1000 else f"{lo:g}+", (kenv >= lo) & (kenv < hi))
        for lo, hi in ((10, 60), (60, 200), (200, 600), (600, 1001))
      ],
    ),
    (
      "pull, N",
      [
        (f"{lo:g}-{hi:g}" if hi < 100 else f"{lo:g}+", (force >= lo) & (force < hi))
        for lo, hi in ((0.5, 5), (5, 15), (15, 40), (40, 1e9))
      ],
    ),
    (
      "wrist",
      [
        ("left", slot == 0),
        ("right", slot == 1),
        ("one wrist", ~both),
        ("both wrists", both),
      ],
    ),
    (
      "event order in the clip",
      [
        ("1st, 2nd", ordinal < 2),
        ("3rd, 4th", (ordinal >= 2) & (ordinal < 4)),
        ("5th and later", ordinal >= 4),
      ],
    ),
  ]
  fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
  for ax, (xlabel, groups) in zip(axes, panels, strict=True):
    meds, ns, labels = [], [], []
    for label, sel in groups:
      sel = sel & mid
      labels.append(label)
      ns.append(int(sel.sum()))
      meds.append(float(np.median(ratio[sel])) if sel.any() else np.nan)
    x = np.arange(len(labels))
    ax.bar(x, np.nan_to_num(meds), color=c_seen, alpha=0.8, width=0.7)
    for xi, mval, n in zip(x, meds, ns, strict=True):
      ax.text(
        xi,
        (mval if np.isfinite(mval) else 0) + 0.03,
        f"{mval:.2f}\n(n={n})",
        ha="center",
        fontsize=6.5,
        color=ink,
      )
    ax.axhline(1.0, color=muted, lw=0.9, ls=":")
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_xlabel(xlabel)
    clean(ax)
  axes[0].set_ylabel("median realized / told, told 150 to 400")
  fig.suptitle(title, fontsize=8)
  fig.tight_layout()
  out4 = events.with_name(f"{stem}_knobs.png")
  fig.savefig(out4, dpi=150)
  plt.close(fig)
  # 5. compliance as tracking of the reference's OWN yield: the adapted
  # reference moved the wrist (served minus free) along the force by what the
  # augmenter's saturated spring gave it, not by force / told; the policy is
  # compliant to the extent it reproduces that yield under the reactive spring.
  ref = yf - ys_
  keep = ref > 0.5
  fig, axes = plt.subplots(
    1, 2, figsize=(8.6, 3.8), gridspec_kw={"width_ratios": [1.15, 1]}
  )
  ax = axes[0]
  for (lo, hi), color in zip(TOLD_BINS, colors, strict=True):
    sel = keep & (told >= lo) & (told < hi)
    ax.scatter(
      ref[sel],
      yf[sel],
      s=9,
      color=color,
      alpha=0.6,
      lw=0,
      label=f"told {_bin(lo, TOLD_BINS)}",
    )
  top = float(np.percentile(ref[keep], 99))
  ax.plot([0, top], [0, top], color=muted, lw=1, ls=":", label="policy = reference")
  r = float(np.corrcoef(ref[keep], yf[keep])[0, 1])
  ax.set_xlim(0, top)
  ax.set_ylim(-3, top)
  ax.set_xlabel("the reference's own yield (adapted minus free), cm")
  ax.set_ylabel("the policy's yield vs the free reference, cm")
  ax.set_title(
    f"r = {r:.2f}; median policy / reference = {np.median(yf[keep] / ref[keep]):.2f}",
    fontsize=8,
  )
  ax.legend(fontsize=7, frameon=False, loc="upper left")
  clean(ax)
  ax = axes[1]
  x = np.arange(len(TOLD_BINS))
  for k, (flag, color, label) in enumerate(
    ((True, c_seen, "seen"), (False, c_unseen, "unseen"))
  ):
    meds, ns = [], []
    for lo, hi in TOLD_BINS:
      sel = keep & (seen == flag) & (told >= lo) & (told < hi)
      meds.append(float(np.median(yf[sel] / ref[sel])) if sel.any() else np.nan)
      ns.append(int(sel.sum()))
    ax.bar(
      x + (k - 0.5) * 0.38,
      np.nan_to_num(meds),
      width=0.36,
      color=color,
      alpha=0.8,
      label=label,
    )
    for xi, mv, n in zip(x + (k - 0.5) * 0.38, meds, ns, strict=True):
      ax.text(
        xi,
        (mv if np.isfinite(mv) else 0) + 0.02,
        f"{mv:.2f}\n({n})",
        ha="center",
        fontsize=6,
        color=ink,
      )
  ax.axhline(1.0, color=muted, lw=0.9, ls=":")
  ax.set_xticks(x, [_bin(lo, TOLD_BINS) for lo, hi in TOLD_BINS])
  ax.set_xlabel("told stiffness, N/m")
  ax.set_ylabel("median policy yield / reference yield")
  ax.legend(fontsize=7, frameon=False)
  clean(ax)
  fig.suptitle(f"{title}; compliance as tracking of the yielded reference", fontsize=8)
  fig.tight_layout()
  out5 = events.with_name(f"{stem}_compliance_tracking.png")
  fig.savefig(out5, dpi=150)
  plt.close(fig)
  for out in (out1, out2, out3, out4, out5):
    print(out.relative_to(REPO))


def main(argv: list[str] | None = None) -> None:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  sp = ap.add_subparsers(dest="cmd", required=True)
  for name in ("summarize", "figures"):
    p = sp.add_parser(name)
    _selectors(p)
  args = ap.parse_args(argv)
  {"summarize": cmd_summarize, "figures": cmd_figures}[args.cmd](args)


def _selectors(p: argparse.ArgumentParser) -> None:
  p.add_argument("--family", choices=("waltz", "stand"), default="waltz")
  p.add_argument("--policy", default="decoupled")
  p.add_argument("--pool", choices=("own", "full"), default="own")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--obs-noise", type=float, default=1.0)
  p.add_argument(
    "--read-s", type=float, default=0.2, help="seconds of the hold read, from its end"
  )
  p.add_argument("--min-hold-rows", type=int, default=5)


if __name__ == "__main__":
  main()
