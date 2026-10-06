"""Per-policy, per-clip metrics on a policy's own pool: every clip once.

One eval session per policy on its frozen standing-start recipe, with as many
envs as the pool has clips (`assign`: env i plays clip i), the tracking,
compliance, partner and clip-drag suites on, no video; then the offline
analyzer, so each policy gets a per-clip summary CSV. `--pool own` is the
policy's training pool; `--pool full` is 200 clips per kind (twice it: the next
seeds of every tag), written as a manifest from the policy's own entries, and
the summary is split into the clips the policy trained on (seen) and the rest
(unseen: new schedules from the same generator and ranges, so in-distribution
but unseen, not out-of-distribution).

A policy is conf/<policy>.yaml (its config) and data/checkpoints/<policy>.pt (its
published checkpoint). Outputs under logs/clip_metrics/<family>/.
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONF = REPO / "src/mjlab/tasks/codancing/config/g1/conf"
OUT = REPO / "logs/clip_metrics"
# The policies the paper evaluates, by the reference clip they dance on.
FAMILIES = {
  "stand": ("stand",),
  "waltz": (
    "decoupled",
    "decoupled_wo_adversarial",
    "decoupled_wo_chaining",
    "wholebody",
    "wholebody_wo_foot_constraint",
    "stiff",
  ),
}
SUITES = "[robot_ref_tracking,compliance,partner,clip_drag]"


# Clips per kind (augmented `ml_`, zero-wrench `zw_`) in the full pool: twice
# every policy's training pool (waltz trains on 50 seeds of each of its two tags,
# stand on 100 of its one), and exactly what the hosted pools hold (waltz seeds
# 0 to 99 of each tag, stand seeds 0 to 199), so the full pool is all of it.
FULL_PER_KIND = 200


@dataclass(frozen=True)
class Family:
  """One policy of a family."""

  name: str  # the family (a FAMILIES key)
  base: str  # the policy: its config and checkpoint stem


def _families(args: argparse.Namespace) -> list[Family]:
  """The selected policies of the family (`--policy all` = every one)."""
  policies = FAMILIES[args.family]
  wanted = list(policies) if args.policy == "all" else [args.policy]
  for policy in wanted:
    if policy not in policies:
      sys.exit(f"{args.family} has no policy {policy!r}; known: {list(policies)}")
  return [Family(args.family, policy) for policy in wanted]


def checkpoint_path(fam: Family) -> Path:
  """The policy's published checkpoint, where the README's dataset download puts
  it. `run eval` writes its sessions beside it, under
  `data/checkpoints/eval_sessions/<recipe>/`, named after the file stem."""
  path = REPO / "data" / "checkpoints" / f"{fam.base}.pt"
  if not path.is_file():
    sys.exit(f"{fam.base}: no checkpoint at {path} (download the dataset, see README)")
  return path


def own_manifest(fam: Family) -> Path:
  """The policy's training pool (its `motion.motions` path)."""
  text = (CONF / f"{fam.base}.yaml").read_text()
  m = re.search(r"^motion:\n  motions: (configs/clip_manifests/\S+\.yaml)$", text, re.M)
  if m is None:
    raise SystemExit(f"{fam.base}: no manifest path under motion.motions")
  return REPO / m.group(1)


def manifest_entries(path: Path) -> list[dict]:
  return yaml.safe_load(path.read_text())


def trackable_names(path: Path) -> list[str]:
  return [e["name"] for e in manifest_entries(path) if e.get("trackable", True)]


_SEED = re.compile(r"^(?P<head>.*_s)(?P<seed>\d+)(?P<tail>.*)$")


def _seed_parts(name: str) -> tuple[str, int, str]:
  """(head, seed, tail) of a clip name like `ml_human_fwd_s12` or `..._s12_expert`."""
  m = _SEED.match(name)
  assert m is not None, (
    f"clip name {name!r} has no `_s<seed>` in it; cannot expand seeds"
  )
  return m["head"], int(m["seed"]), m["tail"]


def full_manifest(fam: Family) -> Path:
  """The full pool in the policy's own entry shape: every kind of entry the policy's
  manifest has (tag, augmented or zero-wrench, expert copies) at the first
  FULL_PER_KIND / tags seeds of its tag, so a policy that serves the free face
  keeps doing so. Written once beside the own manifest under the
  `<N>ml_<N>zw` name."""
  own = own_manifest(fam)
  m = re.search(r"(\d+)ml_(\d+)zw", own.name)
  if m is None:
    raise SystemExit(f"{own.name}: not a `<N>ml_<N>zw` pool; no full pool to build")
  kinds: dict[tuple[str, str], dict] = {}
  order: list[tuple[str, str]] = []
  for e in manifest_entries(own):
    head, seed, tail = _seed_parts(e["name"])
    key = (head, tail)
    if key not in kinds or seed < _seed_parts(kinds[key]["name"])[1]:
      kinds[key] = e
    if key not in order:
      order.append(key)
  # Seeds per tag: FULL_PER_KIND shared by the tags of that kind (`ml_` or
  # `zw_`), counted over every entry shape (an expert copy shares its tag).
  tags_of_kind = collections.Counter(
    head.split("_", 1)[0] for head in {head for head, _ in order}
  )
  seeds_of = {kind: FULL_PER_KIND // n for kind, n in tags_of_kind.items()}
  n_ml = sum(
    seeds_of["ml"] for head, tail in order if head.startswith("ml_") and not tail
  )
  n_zw = sum(
    seeds_of["zw"] for head, tail in order if head.startswith("zw_") and not tail
  )
  out = own.with_name(own.name.replace(m.group(0), f"{n_ml}ml_{n_zw}zw"))
  if out.exists():
    return out
  entries = []
  for key in order:
    template = kinds[key]
    seed0 = _seed_parts(template["name"])[1]
    for seed in range(seeds_of[key[0].split("_", 1)[0]]):
      e = dict(template)
      for k, v in e.items():
        if isinstance(v, str):
          e[k] = v.replace(f"_s{seed0}", f"_s{seed}").replace(
            f"seed{seed0}.npz", f"seed{seed}.npz"
          )
      entries.append(e)
  head = (
    f"# Clip manifest {out.stem}: {len(entries)} entries, a bare list consumed by\n"
    f"# `motion.motions: <this path>` (relative to the repo root). {FULL_PER_KIND}\n"
    f"# clips per kind in the entry shape of {own.name}, the first seeds of each\n"
    "# tag; the policy trained on that file's seeds, the rest are unseen schedules\n"
    "# of the same generator (in-distribution but unseen).\n"
  )
  out.write_text(head + yaml.safe_dump(entries, sort_keys=False))
  return out


def pool_manifest(fam: Family, pool: str) -> Path:
  return own_manifest(fam) if pool == "own" else full_manifest(fam)


def summary_path(
  fam: Family, pool: str, seed: int = 0, noise: float = 0.0, tag: str = ""
) -> Path:
  suffix = ("" if pool == "own" else f".{pool}") + f".s{seed}"
  if noise:
    suffix += f".noise{noise:g}"
  if tag:
    suffix += f".{tag}"
  return OUT / fam.name / f"{fam.base}{suffix}.summary.csv"


def _noise_terms(fam: Family) -> list[tuple[str, float, float]]:
  """(term, n_min, n_max) of every actor term the policy trains with noise on."""
  tree = yaml.safe_load((CONF / f"{fam.base}.yaml").read_text())
  terms = tree["env"]["observations"]["actor"]["terms"]
  return [
    (name, float(t["noise"]["n_min"]), float(t["noise"]["n_max"]))
    for name, t in terms.items()
    if isinstance(t, dict)
    and isinstance(t.get("noise"), dict)
    and "n_min" in t["noise"]
  ]


def _done(out: Path) -> bool:
  """A summary counts as done when its envs table is known and carries every
  suite's columns (the clip_drag suite's come last)."""
  if not out.exists():
    return False
  source = out.with_suffix("").with_suffix(".source.txt")
  if not source.exists():
    return False
  table = REPO / source.read_text().strip()
  if not table.exists():
    return False
  with table.open() as fh:
    header = fh.readline()
  return "metric/clip_drag_k0_force" in header


def run(args: argparse.Namespace) -> None:
  prefix = ["taskset", "-c", args.cpu_mask] if args.cpu_mask else []
  # The standing-pose start: the waltz files freeze it as `stand`; on the stand
  # pool `default` starts from the same pose and is the only frozen recipe.
  recipe = args.recipe or {"waltz": "stand", "stand": "default"}[args.family]
  for fam in _families(args):
    out = summary_path(fam, args.pool, args.seed, args.obs_noise, args.tag)
    if _done(out) and not args.force:
      print(f"skip (done): {fam.base}")
      continue
    manifest = pool_manifest(fam, args.pool)
    n = len(trackable_names(manifest))
    spec = str(checkpoint_path(fam))
    cmd = [
      *prefix,
      "uv",
      "run",
      "python",
      "-m",
      "mjlab.scripts.run",
      "eval",
      "--ckpt",
      spec,
      "--recipe",
      recipe,
      "--override",
      f"eval.recipes.{recipe}.num_envs={n}",
      "--override",
      f"eval.recipes.{recipe}.length_steps={args.length_steps}",
      "--override",
      f"eval.recipes.{recipe}.video.enabled=false",
      "--override",
      "env.commands.codancing.motion_cursor.source.episode_reset.clip_selection=assign",
      "--override",
      f"env.commands.codancing.metric_suites={SUITES}",
      "--override",
      f"motion.motions={manifest.relative_to(REPO)}",
    ]
    # A seed varies the per-env plant draws (CoM offset, foot friction) and
    # the run's RNG; the clip fixes the rest and the policy is deterministic
    # at eval. Seed 0 is the frozen trees' own and is passed like any other.
    cmd += [
      "--override",
      f"runner.seed={args.seed}",
      "--override",
      f"env.seed={args.seed}",
    ]
    if args.obs_noise:
      # The play overlay turns the train-time observation noise off; put it
      # back at `--obs-noise` times the trained ranges (1 = as trained, the
      # default: the policy meets the noise it trained under, and the seed
      # draws it), term by term from the policy's own file.
      cmd += ["--override", "env.observations.actor.enable_corruption=true"]
      for term, lo, hi in _noise_terms(fam):
        cmd += [
          "--override",
          f"env.observations.actor.terms.{term}.noise.n_min={lo * args.obs_noise:g}",
          "--override",
          f"env.observations.actor.terms.{term}.noise.n_max={hi * args.obs_noise:g}",
        ]
    for override in args.override:
      cmd += ["--override", override]
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".log")
    print(
      f"eval: {fam.base} ({args.pool} pool, {n} clips, {n} envs, seed {args.seed}"
      + (f", obs noise x{args.obs_noise:g}" if args.obs_noise else "")
      + ")",
      flush=True,
    )
    # The eval names its files by its start time; take the first table that
    # appears after this launch, so a concurrent eval of the same policy (a later
    # start, still being written) can never be read as ours.
    ckpt = checkpoint_path(fam)
    tables = (
      f"{ckpt.parent.relative_to(REPO)}/eval_sessions/{recipe}/"
      f"{ckpt.stem}-*rollout-envs.csv"
    )
    before = set(REPO.glob(tables))
    with log.open("w") as fh:
      rc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
      sys.exit(f"{fam.base}: run eval exited {rc}; see {log}")
    runs: list[Path] = sorted(set(REPO.glob(tables)) - before)
    if not runs:
      sys.exit(f"{fam.base}: no new envs table under eval_sessions/{recipe}")
    # Which envs table this summary came from, for the passes that read the
    # per-step columns again (scripts/clip_drag.py).
    out.with_suffix("").with_suffix(".source.txt").write_text(
      f"{runs[0].relative_to(REPO)}\n"
    )
    # Imported after the done-check: the analyzer loads torch and mjlab (about
    # 5 s), which a skipped evaluation does not need.
    from mjlab.tasks.codancing.eval_analysis import analyze, seen_unseen_split

    result = analyze(runs[0], settle=args.settle, recipe=recipe, out_dir=out.parent)
    written = out.parent / f"{runs[0].stem}.summary.csv"
    written.replace(out)
    for side in (".phase.csv", ".analysis.json"):
      p = out.parent / f"{runs[0].stem}{side}"
      if p.exists():
        p.replace(out.with_suffix("").with_suffix(side))
    if args.pool != "own":
      # Seen = the clips of the policy's training pool; count-weighted means.
      split = seen_unseen_split(
        result["per_clip_summary"], trackable_names(own_manifest(fam))
      )
      with out.with_suffix("").with_suffix(".split.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "seen_mean", "seen_clips", "unseen_mean", "unseen_clips"])
        for metric, groups in sorted(split.items()):
          w.writerow(
            [metric]
            + [
              f"{groups.get(g, {}).get(k, float('nan')):.5f}"
              for g in ("seen", "unseen")
              for k in ("mean", "clips")
            ]
          )
    print(f"  -> {out.relative_to(REPO)}")


def main(argv: list[str] | None = None) -> None:
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  p.add_argument("--family", choices=FAMILIES, default="waltz")
  p.add_argument("--policy", default="all", help="policy name, or `all`")
  p.add_argument(
    "--obs-noise",
    type=float,
    default=1.0,
    help="observation noise as a multiple of the trained ranges (1 = as trained, the default; 0 = off, the play default)",
  )
  p.add_argument(
    "--pool",
    choices=("own", "full"),
    default="own",
    help="the policy's training pool, or 200 clips per kind (twice it)",
  )
  p.add_argument(
    "--tag",
    default="",
    help="variant tag in the output stem (<policy>.<pool>.s<seed>.noise<x>.<tag>.*), e.g. blind",
  )
  p.add_argument(
    "--override",
    action="append",
    default=[],
    help="extra `run eval` override, repeatable (e.g. an observation scale of 0)",
  )
  p.add_argument(
    "--seed",
    type=int,
    default=0,
    help="runner and env seed of the eval session (0 = the frozen tree's own)",
  )
  p.add_argument(
    "--recipe",
    default=None,
    help="frozen recipe to start from (default: the family's standing start)",
  )
  p.add_argument("--length-steps", type=int, default=500)
  p.add_argument("--settle", type=int, default=25, help="rows dropped after each reset")
  p.add_argument("--force", action="store_true")
  p.add_argument("--cpu-mask", default="", help="pin the eval processes with taskset")
  run(p.parse_args(argv))


if __name__ == "__main__":
  main()
