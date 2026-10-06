"""Motion-data model for the codancing task.

The typed representation of a run's reference-motion SET, decoupled from the
command's loaders and from the raw YAML. A motion set is a ``MotionLibrary`` of
clips; a clip is either robot-only (``MotionClip``) or two-entity
(``PairedMotionClip``).
"""

from mjlab.tasks.codancing.motion.library import (
  MotionClip,
  MotionLibrary,
  PairedMotionClip,
  amp_group_assignment,
  build_motion_library,
)
from mjlab.tasks.codancing.motion.loaders import MotionLoader
from mjlab.tasks.codancing.motion.registry import MotionFrames, MotionRegistry

__all__ = [
  "amp_group_assignment",
  "MotionClip",
  "PairedMotionClip",
  "MotionLibrary",
  "build_motion_library",
  "MotionLoader",
  "MotionRegistry",
  "MotionFrames",
]
