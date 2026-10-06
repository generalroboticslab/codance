"""run_meta provenance string for datagen NPZ products.

The offline datagen writers (augment_multilink.py, qpos_csv_to_motion_npz.py,
make_stand_clip.py) embed one extra NPZ key,
``run_meta``: a 0-d unicode array holding one JSON object. Training never
reads it; it exists for inspection:

    uv run python -c "import numpy as np; print(np.load(p)['run_meta'].item())"

Writers record what they know (source, seed).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime


def build_datagen_meta(source: str | None = None, seed: int | None = None) -> str:
  """One JSON object: created, argv, optional source and seed."""
  meta: dict[str, object] = {
    "created": datetime.now().astimezone().isoformat(timespec="seconds"),
    "argv": list(sys.argv),
  }
  if source is not None:
    meta["source"] = source
  if seed is not None:
    meta["seed"] = seed
  return json.dumps(meta, ensure_ascii=False)
