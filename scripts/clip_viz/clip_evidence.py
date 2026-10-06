"""The real numbers behind the multilink figures, read straight from the clip NPZs.

Every caption cell, tensor slice, event band, signal curve and spring-law check in a
figure must come from a real NPZ, never be made up. This class wraps a multi-link contact
NPZ (and optionally the paired adapted motion NPZ) so figure scripts can read exact
values: per-frame force / stiffness / setpoint, the event table, the ramp-in row by
row, per-slot |F|(t) curves for signal plots, and the reactive spring-law check
(K_env * |setpoint - live| vs |F|).

Body axis of the motion file (for the ``live`` position in law_check): body_pos_w in
the adapted NPZ drops the world body, so left_wrist_yaw_link is index 22 and the
right wrist is 29 (model ids 23 / 30 minus the dropped world body). Pass the right
index per slot.
"""

from __future__ import annotations

import numpy as np

# Body indices of the two wrists in the adapted motion (the world body is dropped
# relative to the model's body axis).
WRIST_BODY_INDEX = {"left_wrist_yaw_link": 22, "right_wrist_yaw_link": 29}


class ClipEvidence:
  """Typed access to one contact clip (+ optional adapted motion, for the live pose).

  Accepts both NPZ kinds: multi-link (force (T, K, 3) + slot_link_names) and
  single-contact (force (T, 3), the link changes per event). A single-contact clip is
  presented internally as a K=1 view (``single`` is True, slot_link_names is
  ``["(varies)"]``), so the event table / curves / lookup share one interface.
  """

  def __init__(
    self, contact_npz: str, adapted_npz: str | None = None, *, active_eps: float = 1e-2
  ) -> None:
    self._c = np.load(contact_npz, allow_pickle=False)
    self._eps = active_eps
    self.single = self._c["force"].ndim == 2  # single-contact NPZ: no K axis
    self.force = self._chan("force")  # (T, K, 3)
    self.slot_link_names = (
      ["(varies)"]
      if self.single
      else [str(s) for s in np.asarray(self._c["slot_link_names"])]
    )
    self.T, self.K = self.force.shape[0], self.force.shape[1]
    self._adapted = np.load(adapted_npz, allow_pickle=False) if adapted_npz else None

  def _chan(self, key: str) -> np.ndarray:
    """Per-frame channel as (T, K, .): single-contact clips get a K=1 slot axis."""
    arr = self._c[key]
    return arr[:, None] if self.single else arr

  # -- per frame ---------------------------------------------------------------
  def active(self, frame: int) -> list[bool]:
    """Per-slot activity at a frame (|F| or |tau| above eps)."""
    f = np.linalg.norm(self.force[frame], axis=-1)
    t = np.linalg.norm(self._chan("torque")[frame], axis=-1)
    return [
      (bool(fk > self._eps) or bool(tk > self._eps))
      for fk, tk in zip(f, t, strict=True)
    ]

  def _frame_link(self, frame: int, k: int) -> str:
    """Link name of slot k at a frame: the slot's bound link for multi-link clips, the
    per-frame link_name channel for single-contact clips."""
    if not self.single:
      return self.slot_link_names[k]
    name = str(np.asarray(self._c["link_name"][frame]).item())
    return name or "(free)"

  def frame(self, frame: int) -> list[dict]:
    """Per-slot dicts of real values at a frame (force, |F|, stiffness, setpoint)."""
    act = self.active(frame)
    out = []
    for k in range(self.K):
      F = self.force[frame, k]
      out.append(
        {
          "slot": k,
          "link": self._frame_link(frame, k),
          "active": act[k],
          "force": [round(float(x), 2) for x in F],
          "fmag": round(float(np.linalg.norm(F)), 2),
          "torque": [round(float(x), 2) for x in self._chan("torque")[frame, k]],
          "link_id": int(self._chan("link_id")[frame, k]),
          "K_rob": round(float(self._chan("robot_stiffness")[frame, k]), 1),
          "K_env": round(float(self._chan("forcefield_stiffness")[frame, k]), 1),
          "K_rot_rob": round(float(self._chan("robot_rot_stiffness")[frame, k]), 4),
          "setpoint_pos": [
            round(float(x), 3) for x in self._chan("setpoint_pos")[frame, k]
          ],
        }
      )
    return out

  # -- episode structure -------------------------------------------------------
  def events(self) -> list[dict]:
    """Flat event table: one row per event with its slot link, frame span and the
    linear stiffness pair sampled for that event."""
    rows = []
    names = [str(s) for s in np.asarray(self._c["event_link_name"])]
    slot_of = {n: i for i, n in enumerate(self.slot_link_names)}
    for i in range(len(self._c["event_f_start"])):
      link = names[i]
      rows.append(
        {
          "slot": slot_of.get(link, -1),
          "link": link,
          "f_start": int(self._c["event_f_start"][i]),
          "f_ramp_end": int(self._c["event_f_ramp_end"][i]),
          "f_hold_end": int(self._c["event_f_hold_end"][i]),
          "f_end": int(self._c["event_f_end"][i]),
          "K_rob": round(float(self._c["event_k_robot"][i]), 1),
          "K_env": round(float(self._c["event_k_ff"][i]), 1),
        }
      )
    return rows

  def ramp_rows(self, f0: int, f1: int) -> list[dict]:
    """Raw per-slot force rows in [f0, f1): the ramp-in as it shows cell by cell."""
    return [
      {
        "frame": fr,
        **{
          f"slot{k}": [round(float(x), 2) for x in self.force[fr, k]]
          for k in range(self.K)
        },
      }
      for fr in range(f0, f1)
    ]

  def force_curve(self, slot: int) -> np.ndarray:
    """|F|(t) of one slot over the whole clip, for a curve aligned with the events."""
    return np.linalg.norm(self.force[:, slot], axis=-1)

  def stiffness_curve(
    self, slot: int, *, rotational: bool = False, env: bool = False
  ) -> np.ndarray:
    """Stiffness K(t) of one slot over the whole clip, aligned with the event bands.

    ``rotational`` selects the rotational stiffness. ``env=True`` reads the
    force-field (environment) stiffness channel (always 0 outside events); otherwise
    the robot stiffness channel (free frames hold "the next event's value / the 140
    placeholder", never zero)."""
    side = "forcefield" if env else "robot"
    key = f"{side}_rot_stiffness" if rotational else f"{side}_stiffness"
    return self._chan(key)[:, slot]

  # -- spring-law check (needs the adapted motion for the live wrist position) --
  def law_check(self, frame: int, slot: int, body_index: int | None = None) -> dict:
    """K_env * |setpoint - live| vs |F| at a frame/slot (reactive spring identity).

    ``live`` is the wrist world position under the adapted pose (FK from the
    qpos_csv_to_motion_npz.py bridge), i.e. the pose that is tracked, not the raw IK
    solution, so the check is close but not exact to the last digit; any figure
    showing it should say so in a footnote.
    """
    if self._adapted is None:
      raise ValueError("law_check needs adapted_npz (the paired motion file)")
    link = self._frame_link(frame, slot)
    if body_index is None:
      body_index = WRIST_BODY_INDEX[link]
    live = self._adapted["body_pos_w"][frame, body_index]
    setp = self._chan("setpoint_pos")[frame, slot]
    F = self.force[frame, slot]
    k_env = float(self._chan("forcefield_stiffness")[frame, slot])
    gap = float(np.linalg.norm(setp - live))
    return {
      "slot": slot,
      "link": link,
      "live": [round(float(x), 3) for x in live],
      "setpoint": [round(float(x), 3) for x in setp],
      "gap_m": round(gap, 3),
      "K_env": round(k_env, 1),
      "K_env_times_gap": round(k_env * gap, 2),
      "fmag": round(float(np.linalg.norm(F)), 2),
    }

  # -- mirror of the runtime lookup --------------------------------------------
  def lookup(self, frame: int) -> list[dict]:
    """Per-slot (global event row, phase), mirroring contact_reference's lookup.

    The row is the **global** row of the flat event table (multi-link clips filter by
    link per slot, matching ``event_indices_at``; single-contact clips match on the
    time window only, matching ``event_index_at``), or -1 with no active event. Phase
    boundaries follow ``phase_at``: frame < f_ramp_end is ramp_up, < f_hold_end is
    hold, anything later is ramp_down.
    """
    rows = self.events()

    def find(link: str | None) -> int:
      for i, r in enumerate(rows):
        if link is not None and r["link"] != link:
          continue
        if r["f_start"] <= frame < r["f_end"]:
          return i
      return -1

    def phase(i: int) -> str:
      if i < 0:
        return "free"
      r = rows[i]
      if frame < r["f_ramp_end"]:
        return "ramp_up"
      if frame < r["f_hold_end"]:
        return "hold"
      return "ramp_down"

    out = []
    for k in range(self.K):
      link = None if self.single else self.slot_link_names[k]
      i = find(link)
      out.append(
        {
          "slot": k,
          "row": i,
          "link": rows[i]["link"] if i >= 0 else None,
          "phase": phase(i),
        }
      )
    return out

  # -- series-spring points (collinear) ----------------------------------------
  def spring_points(self, frame: int, slot: int) -> dict:
    """Collinear points p_ref -> live -> setpoint and the two spring offsets (mm).

    p_ref = setpoint - F/K_env - F/K_rob (the series-spring law the augmenter uses to
    write the setpoint, solved for p_ref); live - p_ref = F/K_rob and
    setpoint - live = F/K_env. ``live`` needs the adapted motion; without it only
    p_ref / setpoint and the two distances are returned.
    """
    F = self.force[frame, slot]
    fmag = float(np.linalg.norm(F))
    setp = self._chan("setpoint_pos")[frame, slot]
    k_env = float(self._chan("forcefield_stiffness")[frame, slot])
    k_rob = float(self._chan("robot_stiffness")[frame, slot])
    p_ref = setp - F / max(k_env, 1e-9) - F / max(k_rob, 1e-9)
    out = {
      "slot": slot,
      "fmag": round(fmag, 2),
      "p_ref": [round(float(x), 3) for x in p_ref],
      "setpoint": [round(float(x), 3) for x in setp],
      "d_rob_mm": round(fmag / max(k_rob, 1e-9) * 1000, 1),
      "d_env_mm": round(fmag / max(k_env, 1e-9) * 1000, 1),
    }
    if self._adapted is not None:
      link = self._frame_link(frame, slot)
      idx = WRIST_BODY_INDEX.get(link)
      if idx is not None:
        out["live"] = [
          round(float(x), 3) for x in self._adapted["body_pos_w"][frame, idx]
        ]
    return out

  # -- export in figure coordinates (curves and event bands) -------------------
  def curve_path(
    self,
    slot: int,
    *,
    px_per_frame: float = 4.0,
    height: float = 110.0,
    ymax: float | None = None,
    f0: int = 0,
    f1: int | None = None,
  ) -> dict:
    """Path data ("M x y L x y ...") of a slot's |F|(t), scaled to figure coordinates.

    x = (frame - f0) * px_per_frame, y = height - |F|/ymax * height (0 at the bottom
    edge). Flat runs keep only their end points (400 frames compress to about 130
    points). ymax defaults to ceil(max|F| + 1); pass the same ymax to several curves
    to put them on a shared axis. Returns {data, ymax, peak, points}.
    """
    f1 = self.T if f1 is None else f1
    mag = np.linalg.norm(self.force[f0:f1, slot], axis=-1)
    ymax = float(np.ceil(mag.max() + 1)) if ymax is None else float(ymax)
    ys = np.round(height - mag / ymax * height, 1)
    pts = []
    for i in range(len(ys)):
      if 0 < i < len(ys) - 1 and ys[i - 1] == ys[i] == ys[i + 1]:
        continue
      pts.append((round(i * px_per_frame, 1), ys[i]))
    data = " ".join(
      ("M" if i == 0 else "L") + f" {x:g} {y:g}" for i, (x, y) in enumerate(pts)
    )
    return {
      "data": data,
      "ymax": ymax,
      "peak": round(float(mag.max()), 2),
      "points": len(pts),
    }


def _demo() -> None:
  """Print the evidence a caption/cell would read from a hosted seed clip (skipped if
  the file is missing)."""
  import os

  root = os.path.abspath(
    os.path.join(
      os.path.dirname(__file__),
      os.pardir,
      os.pardir,
      "data/compliant/rc/20260830_comz_small_002",
    )
  )
  contact = os.path.join(root, "waltz_20260224_001_multilink_contact_seed0.npz")
  adapted = os.path.join(root, "waltz_20260224_001_multilink_forcefield_seed0.npz")
  if not os.path.exists(contact):
    print(f"skipping demo: {contact} not found")
    return
  ev = ClipEvidence(contact, adapted if os.path.exists(adapted) else None)
  print(f"clip: T={ev.T} K={ev.K} slots={ev.slot_link_names}")
  print("events:", ev.events())
  print("frame 160 active:", ev.active(160))
  for slot in ev.frame(160):
    print("  ", slot)
  print("ramp rows 118..121:", ev.ramp_rows(118, 122))
  print("|F| curve slot 0: max", round(float(ev.force_curve(0).max()), 2), "N")
  if ev._adapted is not None:
    for s in range(ev.K):
      print("law_check f160:", ev.law_check(160, s))
  # Self-check: at frame 160 both slots are active (two-handed push); at an idle
  # frame neither is.
  assert ev.active(160) == [True, True], "frame 160 should be two-handed"
  assert ev.active(200) == [False, False], "frame 200 should be idle"
  # lookup mirrors the runtime semantics (known values for this seed clip):
  # f160 -> global rows [1, 4], both in hold; f200 idle; f176 -> rows [1, 4], both
  # in ramp_down.
  lk160 = ev.lookup(160)
  assert [r["row"] for r in lk160] == [1, 4] and all(
    r["phase"] == "hold" for r in lk160
  ), f"unexpected lookup(160): {lk160}"
  assert all(r["row"] == -1 and r["phase"] == "free" for r in ev.lookup(200))
  lk176 = ev.lookup(176)
  assert [r["row"] for r in lk176] == [1, 4] and all(
    r["phase"] == "ramp_down" for r in lk176
  ), f"unexpected lookup(176): {lk176}"
  # The event table carries the stiffness pair sampled per event (rows 1 and 4).
  evs = ev.events()
  assert (evs[1]["K_rob"], evs[1]["K_env"]) == (219.4, 197.0), evs[1]
  assert (evs[4]["K_rob"], evs[4]["K_env"]) == (58.5, 987.2), evs[4]
  # Series-spring points: F/K_rob is 18.1 mm for both slots, F/K_env is 20.2 / 1.1
  # mm.
  sp = [ev.spring_points(160, k) for k in range(ev.K)]
  print("spring_points f160:", sp)
  assert [s["d_rob_mm"] for s in sp] == [18.1, 18.1]
  assert [s["d_env_mm"] for s in sp] == [20.2, 1.1]
  # Curve path: after flat-run compression there are far fewer points than 2T, and
  # ymax covers the peak.
  cp = ev.curve_path(1)
  assert cp["points"] < ev.T and cp["ymax"] >= cp["peak"], cp
  print(f"curve_path slot1: {cp['points']} pts, ymax {cp['ymax']}, peak {cp['peak']}")
  print("OK: frame 160 two-handed, 200 idle; lookup/spring/curve self-checks pass.")


if __name__ == "__main__":
  _demo()
