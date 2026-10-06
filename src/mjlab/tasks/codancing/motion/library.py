"""Typed reference-motion set: ``MotionLibrary`` of robot-only / paired clips.

``build_motion_library`` parses and validates the motion set (the ``motions``
manifest YAML or inline entries, paired vs robot-only) into a typed
``MotionLibrary`` the executor then applies to the command -- the one place that
knows the schema and the partner-axis contract.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
  from mjlab.tasks.codancing.config.run_cfg import MotionCfg


@dataclass(frozen=True)
class MotionClip:
  """A single-entity (robot-only) reference clip -- ``partner_mode: none``."""

  robot_motion_file: str
  weight: float = 1.0
  name: str = ""
  # Two ORTHOGONAL roles (decoupled): `trackable` = is this clip in the cursor's
  # SELECTION DOMAIN (a tracking target random/alternate/pin can pick);
  # `amp_expert` = do its windows feed the AMP discriminator's expert pool.
  # Independent: `trackable:false amp_expert:true` = style-corpus-only;
  # `trackable:true amp_expert:false` = tracked but NOT a style exemplar;
  # both-false is an error.
  trackable: bool = True
  amp_expert: bool = True
  # Optional shared AMP discriminator-group label (which head/pool). Omitted on
  # every clip -> each clip is its own group (group == clip index); shared
  # labels collapse K as the clip pool scales. See `amp_group_assignment`.
  # (Which BODIES AMP judges is a separate discriminator-level config:
  # `AmpFeatureCfg.body_names` (the run file's `amp_feature` block), not a clip
  # field.)
  amp_group: str | None = None
  # SoftMimic AUGMENTED clip viz data (the clip is self-describing): for a
  # force-adapted clip, `robot_motion_file` is the ADAPTED (force-yielded) face;
  # `free_motion_file` is the ORIGINAL (un-forced) reference -> the original
  # ghost; `contact_file` is the native contact NPZ (force / setpoint / K, event
  # table) -> the force arrow + the online ForceField event. Both None for a
  # plain clip.
  free_motion_file: str | None = None
  contact_file: str | None = None


@dataclass(frozen=True)
class PairedMotionClip:
  """A two-entity (human + robot) reference clip -- ``partner_mode: imagined|g1``.

  ``placement_offset`` is the robot's spawn position relative to the human
  (atomic per clip).
  """

  robot_motion_file: str
  human_motion_file: str
  placement_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
  weight: float = 1.0
  name: str = ""
  trackable: bool = True  # cursor selection domain (see MotionClip)
  amp_expert: bool = True  # feeds the AMP expert pool (see MotionClip)
  amp_group: str | None = None
  # Optional foot-follow target offset (the foot reward / termination
  # `offset_xy`). Distinct from `placement_offset` on purpose: root placement
  # and footfall geometry legitimately differ. None -> the task default.
  foot_follow_offset: tuple[float, float] | None = None
  # SoftMimic augmented-clip viz data (self-describing clip; see MotionClip).
  free_motion_file: str | None = None
  contact_file: str | None = None


Clip = MotionClip | PairedMotionClip


def amp_group_assignment(clips: Sequence[Clip]) -> tuple[int, ...]:
  """Clip -> AMP discriminator-group index.

  Default (no ``amp_group`` labels): each clip is its own group (group ==
  clip index). With labels: clips sharing a label share a group; group ids follow first
  appearance. Partial labeling fails fast (ambiguous).
  """
  labels = [c.amp_group for c in clips]
  if all(label is None for label in labels):
    return tuple(range(len(clips)))
  if any(label is None for label in labels):
    raise ValueError(
      "amp_group must be set on every clip or on none -- partial labeling "
      f"is ambiguous (labels={labels})."
    )
  order: dict[str, int] = {}
  for label in labels:
    assert label is not None
    if label not in order:
      order[label] = len(order)
  return tuple(order[label] for label in labels)  # type: ignore[index]


@dataclass(frozen=True)
class MotionLibrary:
  """A pooled set of reference clips, sampled per reset.

  Homogeneous: every clip is the same type (robot-only or paired), which must
  match the run's ``partner_mode`` (paired <-> a human side exists) -- the
  builder enforces it against ``require_human_motion``.
  """

  clips: tuple[Clip, ...]

  def __post_init__(self) -> None:
    if not self.clips:
      raise ValueError("MotionLibrary needs at least one clip.")
    paired = [isinstance(c, PairedMotionClip) for c in self.clips]
    if any(paired) and not all(paired):
      raise ValueError(
        "MotionLibrary clips must be homogeneous: either all robot-only "
        "(MotionClip) or all paired (PairedMotionClip), not a mix."
      )

  @property
  def weights(self) -> tuple[float, ...]:
    return tuple(c.weight for c in self.clips)


def build_motion_library(
  motion: MotionCfg,
  *,
  require_human_motion: bool,
  resolve_path: Callable[[str], str],
) -> MotionLibrary:
  """Parse + validate a ``MotionCfg`` into a typed ``MotionLibrary``.

  ``motion.motions`` is either a manifest YAML path or the clip entries
  inline. ``resolve_path`` maps a motion-file spec to a local path (registry /
  wandb resolution) -- injected so this stays pure and unit-testable. Raises a
  teaching error on any partner-axis / schema mismatch (the single place the
  motion contract is enforced).
  """
  if motion.motions is None:
    raise ValueError(
      "Must provide motion.motions (a clip-manifest YAML path or an inline "
      "clip list; single-clip runs use a one-entry manifest or list)."
    )
  if isinstance(motion.motions, str):
    source = motion.motions
    entries = _load_manifest(Path(motion.motions))
  else:
    source = "inline motion.motions"
    entries = motion.motions
  clips = _parse_entries(
    entries,
    source=source,
    require_human_motion=require_human_motion,
    resolve_path=resolve_path,
  )
  if not any(c.trackable for c in clips):
    raise ValueError(
      f"Every entry in {source} is `trackable: false` -- the cursor "
      "would have nothing to track. Keep at least one trackable clip."
    )
  return MotionLibrary(clips=tuple(clips))


def _load_manifest(manifest_path: Path) -> Any:
  if not manifest_path.exists():
    raise FileNotFoundError(f"Motion manifest not found: {manifest_path}")
  return yaml.safe_load(manifest_path.read_text())


def _parse_entries(
  entries: Any,
  *,
  source: str,
  require_human_motion: bool,
  resolve_path: Callable[[str], str],
) -> list[Clip]:
  # A resolvable source is not enough: an empty file loads as None, `[]` passes
  # the loop -- both then die cryptically downstream.
  if not entries:
    raise ValueError(
      f"No clips found in {source} (the file is empty or an empty "
      "list); provide at least one {robot_motion_file[, human_motion_file]} "
      "entry."
    )
  if not isinstance(entries, list):
    raise ValueError(
      f"{source} must contain a YAML *list* of clip "
      f"entries, got {type(entries).__name__}."
    )

  clips: list[Clip] = []
  for i, p in enumerate(entries):
    name = p.get("name", "")
    has_human = bool(p.get("human_motion_file"))
    if not p.get("robot_motion_file"):
      raise ValueError(
        f"Manifest entry {i} (name={name!r}) in {source} is missing "
        "`robot_motion_file` (every entry imitates a robot reference motion)."
      )
    weight = float(p.get("weight", 1.0))
    trackable = bool(p.get("trackable", True))
    amp_expert = bool(p.get("amp_expert", True))
    if not (trackable or amp_expert):
      raise ValueError(
        f"Manifest entry {i} (name={name!r}) in {source} has both "
        "`trackable: false` and `amp_expert: false` -- it would do nothing "
        "(not a tracking target, not a style exemplar). Set at least one true."
      )
    if trackable and weight <= 0.0:
      raise ValueError(
        f"Manifest entry {i} (name={name!r}) in {source} has "
        f"weight={weight} but is trackable -- a trackable clip needs weight > 0. "
        "For a style-corpus-only entry (AMP experts, never selected for "
        "tracking) set `trackable: false` instead of zeroing the weight."
      )
    # Optional augmented-clip viz data (self-describing clip).
    free_motion_file = (
      resolve_path(p["free_motion_file"]) if p.get("free_motion_file") else None
    )
    contact_file = resolve_path(p["contact_file"]) if p.get("contact_file") else None
    if require_human_motion:
      if not has_human:
        raise ValueError(
          f"Manifest entry {i} (name={name!r}) in {source} is missing "
          "`human_motion_file`, but the composed task needs a human reference "
          "(require_human_motion=true: partner_mode imagined or g1). Add the "
          "human file, or set partner_mode: none for robot-only clips."
        )
      clips.append(
        PairedMotionClip(
          robot_motion_file=resolve_path(p["robot_motion_file"]),
          human_motion_file=resolve_path(p["human_motion_file"]),
          placement_offset=tuple(p.get("placement_offset", (0.0, 0.0, 0.0))),
          weight=weight,
          name=name,
          trackable=trackable,
          amp_expert=amp_expert,
          amp_group=p.get("amp_group"),
          foot_follow_offset=(
            tuple(p["foot_follow_offset"]) if "foot_follow_offset" in p else None
          ),
          free_motion_file=free_motion_file,
          contact_file=contact_file,
        )
      )
    else:
      if has_human:
        raise ValueError(
          f"Manifest entry {i} (name={name!r}) in {source} carries a "
          "`human_motion_file`, but the composed task is partner-free "
          "(require_human_motion=false: partner_mode none). Drop the human file "
          "(each robot-only entry carries `robot_motion_file` only) or set "
          "partner_mode to imagined or g1."
        )
      clips.append(
        MotionClip(
          robot_motion_file=resolve_path(p["robot_motion_file"]),
          weight=weight,
          name=name,
          trackable=trackable,
          amp_expert=amp_expert,
          amp_group=p.get("amp_group"),
          free_motion_file=free_motion_file,
          contact_file=contact_file,
        )
      )
  return clips
