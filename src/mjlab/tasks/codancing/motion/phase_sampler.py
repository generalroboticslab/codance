"""Per-(clip, bin) start-phase sampler behind `episode_reset: rsi`.

Phase choice has ONE owner (this class, behind `rsi`): the registry is a pure
data store (frames, lengths, slicing), and the command owns cursor placement
(0 for `default`/`keyframe`, this sampler for `rsi`).

Every sampler contributes a weight ROW per clip; the consuming pipeline is
identical and ordered floor -> non-causal kernel -> normalize (the beyondmimic
order, tracking/mdp/commands.py:261-273). Three silent-failure rules, enforced
here:
  1. the `uniform` floor can never be zero (a bin at zero weight is never
     sampled, so it never fails and never regains weight);
  2. the floor is added BEFORE the kernel, normalization AFTER it;
  3. the kernel pads the RIGHT (replicate) so bins inherit from LATER bins and
     mass moves BACKWARD (spawn before the trouble). Padding the left would
     silently reverse it.

Bins are EQUAL WIDTH per clip (about one per second: `ceil(length)` bins of
`length / nbins` seconds, beyondmimic's own construction) and the padding is
replicated PER ROW at each clip's own last bin. Both are load-bearing: a
partial last bin, or a shared zero region past a shorter clip's end, would let
the kernel bend `{uniform: 1.0}` away from uniform near clip ends, which is
exactly the quiet failure the three rules guard against.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

SAMPLER_KEYS = ("uniform", "force_event", "failure_binned")


class PhaseSampler:
  def __init__(
    self,
    clip_lengths_s: torch.Tensor,  # (C,)
    weights: Mapping[str, float],  # a plain dict or Hydra's DictConfig
    *,
    kernel_lambda: float = 0.8,
    kernel_size: int = 5,
    ema_alpha: float = 0.001,
    hold_windows: torch.Tensor | None = None,  # (E, 3) rows [clip, t0, t1]
    device: str = "cpu",
  ) -> None:
    weights = {str(k): float(v) for k, v in dict(weights).items()}
    unknown = set(weights) - set(SAMPLER_KEYS)
    if unknown:
      raise ValueError(f"unknown sampler keys {sorted(unknown)}; valid: {SAMPLER_KEYS}")
    if weights.get("uniform", 0.0) <= 0.0:
      raise ValueError(
        "rsi.sampler must carry uniform > 0: a bin at zero weight is never "
        "sampled, so it never fails and never regains weight."
      )
    if weights.get("force_event", 0.0) > 0.0 and (
      hold_windows is None or len(hold_windows) == 0
    ):
      raise ValueError(
        "rsi.sampler.force_event needs a contact-event schedule (hold "
        "windows); this pool carries none."
      )
    if kernel_size < 1:
      raise ValueError("rsi.kernel_size must be >= 1 (1 = identity, no spread).")
    self.device = device
    self.w = {k: weights.get(k, 0.0) for k in SAMPLER_KEYS}
    self.ema_alpha = float(ema_alpha)
    self.lengths = clip_lengths_s.to(device).float()
    # Equal-width bins per clip: `ceil(length)` bins (at least one) of
    # `length / nbins` seconds each. A constant row stays constant under
    # replicate padding, so `{uniform: 1.0}` is EXACTLY uniform over
    # [0, length] through the kernel.
    self.nbins = self.lengths.ceil().clamp(min=1).long()
    self.width = self.lengths / self.nbins.float()  # (C,)
    self.max_bins = int(self.nbins.max().item())
    cols = torch.arange(self.max_bins, device=device)
    self.mask = (cols[None, :] < self.nbins[:, None]).float()  # (C, max_bins)
    c = len(self.lengths)
    self._fail_ema = torch.zeros(c, self.max_bins, device=device)
    self._fail_now = torch.zeros(c, self.max_bins, device=device)
    self._force = torch.zeros(c, self.max_bins, device=device)
    if hold_windows is not None and len(hold_windows):
      for clip, t0, t1 in hold_windows.to(device).float():
        ci = int(clip.item())
        starts = cols.float() * self.width[ci]
        lo = torch.maximum(t0, starts)
        hi = torch.minimum(t1, starts + self.width[ci])
        self._force[ci] += (hi - lo).clamp(min=0.0)  # per-bin overlap seconds
    kernel = torch.tensor([kernel_lambda**i for i in range(kernel_size)], device=device)
    self.kernel = kernel / kernel.sum()

  def note_failures(self, clip_ids: torch.Tensor, times_s: torch.Tensor) -> None:
    clip_ids = clip_ids.to(self.device)
    width = self.width[clip_ids].clamp(min=1e-6)
    bins = (times_s.to(self.device) / width).floor().long()
    bins = torch.minimum(bins, self.nbins[clip_ids] - 1).clamp(min=0)
    idx = clip_ids * self.max_bins + bins
    self._fail_now.view(-1).scatter_add_(
      0, idx, torch.ones_like(idx, dtype=torch.float32)
    )

  def decay(self) -> None:
    """EMA fold, the half beyondmimic keeps outside its sampler
    (tracking/mdp/commands.py:416-420). Call once per env step."""
    self._fail_ema = (
      self.ema_alpha * self._fail_now + (1.0 - self.ema_alpha) * self._fail_ema
    )
    self._fail_now.zero_()

  def _row_normalized(self, rows: torch.Tensor) -> torch.Tensor:
    rows = rows * self.mask
    total = rows.sum(dim=1, keepdim=True)
    # Fallback for an all-zero row: flat over the clip's own bins. Deliberate:
    # a clip with no hold windows (a zero-wrench clip in a mixed pool) or no
    # failures yet keeps its uniform floor for that term instead of dropping
    # out of the pool.
    flat = self.mask / self.nbins[:, None].float()
    return torch.where(total > 0, rows / total.clamp(min=1e-12), flat)

  def distribution(self) -> torch.Tensor:
    """(C, max_bins) sampling distribution after floor -> kernel -> normalize."""
    w = (
      self.w["uniform"] * self._row_normalized(self.mask)
      + self.w["force_event"] * self._row_normalized(self._force)
      + self.w["failure_binned"] * self._row_normalized(self._fail_ema)
    )
    # Per-row replicate padding: columns past a clip's last bin take that
    # clip's LAST value, so every clip sees the right-replicate pad at ITS
    # OWN end. A shared zero region would drain the last kernel_size - 1
    # bins of every shorter clip in a mixed-length pool.
    last = w.gather(1, (self.nbins - 1)[:, None])
    w = torch.where(self.mask.bool(), w, last)
    # Non-causal right-pad: output[i] sums input[i:], mass flows BACKWARD.
    padded = torch.nn.functional.pad(
      w.unsqueeze(1), (0, len(self.kernel) - 1), mode="replicate"
    )
    w = torch.nn.functional.conv1d(padded, self.kernel.view(1, 1, -1)).squeeze(1)
    w = w * self.mask
    return w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)

  def sample(self, clip_ids: torch.Tensor) -> torch.Tensor:
    clip_ids = clip_ids.to(self.device)
    d = self.distribution()[clip_ids]
    bins = torch.multinomial(d, 1).squeeze(-1)
    u = torch.rand(len(clip_ids), device=self.device)
    t = (bins.float() + u) * self.width[clip_ids]
    return t.minimum(self.lengths[clip_ids])
