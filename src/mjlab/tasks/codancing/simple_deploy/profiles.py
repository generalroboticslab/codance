"""Real-partner height profiles: who wears the marker sets, which Tracker
objects those sets are, and the per-body z offsets the VICON source subtracts.

The trained partner is a G1 proxy whose bodies differ in height from a real
human's worn clusters, so ``offsets_m[body] = cluster_z_m[object] - proxy_body_z_m[body]``,
keyed by the TRAINED body name (the trunk cluster serves pelvis and
torso_link). The data lives in ``partner_profiles.yaml`` next to this module;
a run names its profile (``--partner-profile``) and the record keeps both the
name and the numbers, so runs of different wearers stay distinguishable and
re-projectable. A wearer whose objects are not the default ``human_*`` set
(partner-b wears ``partner_b_root`` and ``partner_b_{left,right}_knee``) states its own
``vicon_objects`` map; a body absent from that map cannot be deployed for
them, which is how an artifact that observes other bodies is refused for
that wearer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import yaml

PROFILE_FILE = Path(__file__).with_name("partner_profiles.yaml")
MM_TO_M = 1e-3

# A worn cluster sits within about 0.2 m of the proxy body it stands in for
# (partner-a: 0.10 to 0.18 m above). Sets left on the floor (or
# on a table, or a wearer lying down) read about -0.7 m instead, which would
# be written as a perfectly deployable profile and then subtracted from every
# partner position for the whole run. Refuse it at the write instead.
MAX_OFFSET_M = 0.5

HEADER = """# Real-partner height profiles for the VICON partner source: who wears the
# marker sets and the per-body z offsets that follow. The trained partner is a
# G1 proxy whose bodies differ in height from a real human's worn clusters, so
# each human object's world z is shifted by the wearer's offset before the
# pelvis-frame projection: offsets_m[body] = cluster_z_m[object] -
# proxy_body_z_m[body], keyed by the TRAINED body name (the trunk cluster
# serves pelvis and torso_link). A wearer whose Tracker objects are not the
# default human_* set states its own `vicon_objects` (trained body -> object);
# bodies it omits cannot be deployed for them. `measured: null` = not
# deployable yet. Add or refresh a wearer with
#   python -m mjlab.tasks.codancing.simple_deploy measure --profile <name>
# (stand still in the volume wearing the sets); the file is rewritten from its
# data, so keep notes in the entries, not in comments.
"""


@dataclass(frozen=True)
class PartnerProfile:
  name: str
  measured: str | None  # date of the standing probe; None = not deployable
  note: str
  cluster_z_m: dict[str, float]  # worn cluster heights per Tracker object
  offsets_m: dict[str, float] | None  # per trained body
  objects: dict[str, str]  # trained body -> Tracker object; empty = the default

  @property
  def deployable(self) -> bool:
    return self.measured is not None and bool(self.offsets_m)


def _read(path: Path) -> dict:
  return yaml.safe_load(path.read_text()) or {}


def load_profiles(
  path: Path = PROFILE_FILE,
) -> tuple[dict[str, float], dict[str, PartnerProfile]]:
  """(proxy body heights, profiles by name)."""
  raw = _read(path)
  proxy = {str(k): float(v) for k, v in (raw.get("proxy_body_z_m") or {}).items()}
  profiles: dict[str, PartnerProfile] = {}
  for name, entry in (raw.get("profiles") or {}).items():
    entry = entry or {}
    measured = entry.get("measured")
    offsets = entry.get("offsets_m") or None
    profiles[str(name)] = PartnerProfile(
      name=str(name),
      measured=None if measured is None else str(measured),
      note=str(entry.get("note") or ""),
      cluster_z_m={
        str(k): float(v) for k, v in (entry.get("cluster_z_m") or {}).items()
      },
      offsets_m=None
      if offsets is None
      else {str(k): float(v) for k, v in offsets.items()},
      objects={str(k): str(v) for k, v in (entry.get("vicon_objects") or {}).items()},
    )
  return proxy, profiles


def partner_profile(name: str | None, path: Path = PROFILE_FILE) -> PartnerProfile:
  """The deployable profile called ``name``; every failure names the measured
  and the pending profiles so the operator knows what to do."""
  _, profiles = load_profiles(path)
  ready = sorted(n for n, p in profiles.items() if p.deployable)
  pending = sorted(n for n, p in profiles.items() if not p.deployable)
  where = f"measured: {ready}; not measured yet: {pending} ({path.name})"
  if name is None:
    raise ValueError(
      f"--partner-profile is required with a live partner: who wears the "
      f"marker sets? {where}"
    )
  if name not in profiles:
    raise ValueError(
      f"partner profile {name!r} is not in {path.name}; {where}. Add it with "
      f"`simple_deploy measure --profile {name}`."
    )
  if not profiles[name].deployable:
    raise ValueError(
      f"partner profile {name!r} is not measured yet; run `simple_deploy "
      f"measure --profile {name}` with them standing in the volume. {where}"
    )
  return profiles[name]


def offsets_for(profile: PartnerProfile, bodies: Sequence[str]) -> tuple[float, ...]:
  """The profile's offsets in the term's body order; a body the profile does
  not cover is an error, never a silent zero."""
  offsets = profile.offsets_m or {}
  missing = [b for b in bodies if b not in offsets]
  if missing:
    raise ValueError(
      f"partner profile {profile.name!r} has no offset for {missing}; it covers "
      f"{sorted(offsets)}"
    )
  return tuple(float(offsets[b]) for b in bodies)


def derive_offsets(
  cluster_z_m: Mapping[str, float],
  proxy_body_z_m: Mapping[str, float],
  object_by_body: Mapping[str, str] | None = None,
) -> dict[str, float]:
  """offsets_m per trained body from measured cluster heights, rounded to 2
  decimals (a centimetre)."""
  if object_by_body is None:
    # Lazy: codance.py imports this module.
    from mjlab.tasks.codancing.simple_deploy.codance import VICON_OBJECT_BY_BODY

    object_by_body = VICON_OBJECT_BY_BODY
  out: dict[str, float] = {}
  for body, obj in object_by_body.items():
    if obj in cluster_z_m and body in proxy_body_z_m:
      out[body] = round(float(cluster_z_m[obj]) - float(proxy_body_z_m[body]), 2)
  return out


def object_map(names: Sequence[str]) -> dict[str, str]:
  """Trained body -> Tracker object for a wearer's own objects, given in the
  order of the default set's distinct objects (trunk, left knee, right knee);
  fewer names cover the leading bodies only."""
  # Lazy: codance.py imports this module.
  from mjlab.tasks.codancing.simple_deploy.codance import VICON_OBJECT_BY_BODY

  slots = tuple(dict.fromkeys(VICON_OBJECT_BY_BODY.values()))
  if not names or len(names) > len(slots):
    raise ValueError(
      f"give 1 to {len(slots)} objects in the order of {slots}; got {list(names)}"
    )
  by_slot = dict(zip(slots, names, strict=False))  # fewer names: leading bodies
  return {b: by_slot[o] for b, o in VICON_OBJECT_BY_BODY.items() if o in by_slot}


def measure_cluster_heights(
  client,
  objects: Sequence[str],
  seconds: float,
  *,
  dt: float = 0.02,
  now: Callable[[], float] = time.monotonic,
  sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, float], int]:
  """Mean world z (m) per object over ``seconds`` of Tracker frames, the
  wearer standing still. A frame missing an object is skipped for that object;
  an object never seen is an error. Returns (heights, frames read)."""
  import pyvicon_datastream as pv

  sums = {o: 0.0 for o in objects}
  counts = {o: 0 for o in objects}
  frames = 0
  t0 = now()
  while now() - t0 < seconds:
    if client.get_frame() == pv.Result.Success:
      frames += 1
      for o in objects:
        p = client.get_segment_global_translation(o, o)
        if p is not None:
          sums[o] += float(np.asarray(p, dtype=np.float64)[2]) * MM_TO_M
          counts[o] += 1
    sleep(dt)
  unseen = [o for o in objects if counts[o] == 0]
  if unseen:
    raise RuntimeError(f"objects never tracked in {frames} frames: {unseen}")
  return {o: sums[o] / counts[o] for o in objects}, frames


def write_profile(
  name: str,
  cluster_z_m: Mapping[str, float],
  *,
  measured: str,
  note: str = "",
  path: Path = PROFILE_FILE,
  object_by_body: Mapping[str, str] | None = None,
) -> PartnerProfile:
  """Add or refresh a wearer: derive the offsets from the measured heights and
  rewrite the file (header + data). The entry keeps ``object_by_body``, else
  its own object map; the other entries survive, comments do not."""
  raw = _read(path)
  entry = (raw.get("profiles") or {}).get(name) or {}
  worn = dict(object_by_body or entry.get("vicon_objects") or {})
  offsets = derive_offsets(cluster_z_m, raw.get("proxy_body_z_m") or {}, worn or None)
  if not offsets:
    raise ValueError(
      f"no offset could be derived for {name!r}: measured objects "
      f"{sorted(cluster_z_m)} match no trained body of {path.name}"
    )
  implausible = {b: o for b, o in offsets.items() if abs(o) > MAX_OFFSET_M}
  if implausible:
    raise ValueError(
      f"{name!r} was not measured on a standing wearer: offsets {implausible} "
      f"exceed {MAX_OFFSET_M} m (a worn cluster sits within about 0.2 m of the "
      f"proxy body). Measured heights [m]: "
      f"{ {k: round(float(v), 3) for k, v in cluster_z_m.items()} }. Put the "
      "marker sets on, stand in the volume, and run measure again; nothing "
      "was written."
    )
  raw.setdefault("profiles", {})[name] = {
    "measured": measured,
    "note": note,
    **({"vicon_objects": worn} if worn else {}),
    "cluster_z_m": {k: round(float(v), 3) for k, v in cluster_z_m.items()},
    "offsets_m": offsets,
  }
  path.write_text(HEADER + yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
  return load_profiles(path)[1][name]
