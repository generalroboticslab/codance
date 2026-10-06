# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import dataclasses
import enum
import functools
import math
import os
import pathlib
import warnings
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.utils.log_writer import LogWriter

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

try:
    import numpy as _np
except ModuleNotFoundError:  # pragma: no cover - numpy ships with torch
    _np = None

try:
    from omegaconf import DictConfig, ListConfig, OmegaConf
except ModuleNotFoundError:  # pragma: no cover - omegaconf is optional for rsl_rl
    DictConfig = ListConfig = OmegaConf = None  # type: ignore[assignment,misc]

# Arrays/tensors with more than this many elements are tagged by shape rather
# than inlined, so a config stays small + filterable (the AMP expert pool
# ``algorithm.amp.data`` is the motivating multi-megabyte case).
_MAX_INLINE_ELEMS = 64


def _callable_repr(obj) -> str:
    """A stable, filterable name for a function / class / method / partial.

    A function's default ``repr`` embeds its memory address (``<function f at
    0x7f..>``) which differs every process, so logging it verbatim gives each
    run a unique, unfilterable value. Use the import path instead: two runs with
    the same reward/event ``func`` then log the SAME value and group/filter
    together in the GUI."""
    if isinstance(obj, functools.partial):
        return f"partial({_callable_repr(obj.func)})"
    mod = getattr(obj, "__module__", None)
    qual = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    if qual:
        return f"{mod}:{qual}" if mod else qual
    return f"<{type(obj).__name__}>"


def _safe_key(k):
    """wandb/JSON object keys must be strings; stringify the rest faithfully so a
    non-str-keyed dict (e.g. ranges keyed by axis index) still flattens into
    filterable columns instead of tripping the serializer."""
    if isinstance(k, str):
        return k
    if isinstance(k, enum.Enum):
        return str(k.value)
    return str(k)


def _as_omegaconf_container(obj):
    """Resolve an OmegaConf ``DictConfig``/``ListConfig`` to a plain container so
    interpolations become concrete values (returns ``None`` for anything else)."""
    if OmegaConf is None or not isinstance(obj, (DictConfig, ListConfig)):
        return None
    try:
        return OmegaConf.to_container(obj, resolve=True)
    except Exception:
        return OmegaConf.to_container(obj, resolve=False)


def _json_safe(obj, _seen=None):
    """Recursively coerce a config tree into JSON-serializable, wandb-loggable,
    GUI-filterable leaves.

    "Shown in the wandb GUI" means every knob reaches ``wandb.config`` as a
    primitive leaf: wandb flattens nested config dicts into dot-keyed, filterable
    runs-table columns, so a leaf that survives here as a real value is
    filterable there. A ``<Type>`` stub is *visible* but dead for filtering, so
    we stub ONLY genuinely irreducible runtime objects (tensors, ``nn.Module``,
    live handles) and give everything else a faithful value:

    * ``NaN`` / ``Inf`` floats  -> ``"nan"`` / ``"inf"`` / ``"-inf"`` (JSON has no
      literal for them; wandb would drop or mangle the raw float)
    * enums                     -> their ``.value``
    * ``pathlib`` paths         -> ``str``
    * ``slice``                 -> ``"slice(start, stop, step)"``
    * numpy scalars             -> Python scalar; small arrays -> nested list
    * ``OmegaConf`` nodes       -> resolved plain container
    * dataclass instances       -> their fields (no deepcopy, unlike ``asdict``)
    * callables / classes       -> import path ``module:qualname`` (address-free,
      so reward/event ``func``s become filterable instead of ``<function .. at
      0x..>`` noise)
    * ``range``                 -> ``"range(start, stop, step)"``
    * non-str dict keys         -> stringified
    * tuples / sets             -> lists
    * cycles                    -> ``"<cycle>"``

    An unknown or irreducible object degrades to ``<TypeName>`` (with its
    ``shape`` when it has one) rather than dropping the key. This is best-effort,
    not infallible: a config nested thousands deep can still hit Python's
    recursion limit, which is why ``store_config`` wraps the log call so config
    logging never aborts a training run."""
    if _seen is None:
        _seen = set()
    # Fast path: JSON primitives (bool is an int subclass; both pass through).
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return "nan"
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    if isinstance(obj, enum.Enum):
        return _json_safe(obj.value, _seen)
    if isinstance(obj, pathlib.PurePath):
        return str(obj)
    if isinstance(obj, slice):
        return f"slice({obj.start}, {obj.stop}, {obj.step})"
    if isinstance(obj, range):  # sibling of slice; expand rather than stub
        return f"range({obj.start}, {obj.stop}, {obj.step})"
    if _np is not None:
        if isinstance(obj, _np.generic):  # numpy scalar: float64 / int64 / bool_ / ...
            return _json_safe(obj.item(), _seen)
        if isinstance(obj, _np.ndarray):
            if obj.size <= _MAX_INLINE_ELEMS:
                return _json_safe(obj.tolist(), _seen)
            return f"<ndarray shape={tuple(obj.shape)} dtype={obj.dtype}>"
    container = _as_omegaconf_container(obj)
    if container is not None:
        return _json_safe(container, _seen)
    oid = id(obj)
    if isinstance(obj, dict):
        if oid in _seen:
            return "<cycle>"
        _seen.add(oid)
        out = {_safe_key(k): _json_safe(v, _seen) for k, v in obj.items()}
        _seen.discard(oid)
        return out
    if isinstance(obj, (list, tuple, set, frozenset)):
        if oid in _seen:
            return "<cycle>"
        _seen.add(oid)
        if isinstance(obj, (set, frozenset)):
            # Sort for a stable order, but fall back to unordered if an element's
            # repr raises -- serialization must not blow up on a hostile member.
            try:
                seq = sorted(obj, key=repr)
            except Exception:
                seq = list(obj)
        else:
            seq = obj
        out = [_json_safe(v, _seen) for v in seq]
        _seen.discard(oid)
        return out
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        if oid in _seen:
            return "<cycle>"
        _seen.add(oid)
        out = {f.name: _json_safe(getattr(obj, f.name), _seen) for f in dataclasses.fields(obj)}
        _seen.discard(oid)
        return out
    if callable(obj):  # functions, classes, bound methods, functools.partial
        return _callable_repr(obj)
    # A genuinely irreducible runtime object (tensor, nn.Module, live handle):
    # tag it so the config stays loggable and the type (and shape) is visible.
    shape = getattr(obj, "shape", None)
    if shape is not None:
        try:
            return f"<{type(obj).__name__} shape={tuple(shape)}>"
        except Exception:
            pass
    return f"<{type(obj).__name__}>"


def _config_to_dict(cfg):
    """Reduce an env/train cfg to something ``_json_safe`` can walk.

    ``_json_safe`` already recurses dataclasses and dicts, so we only special-
    case an explicit ``to_dict()`` (some env cfgs expose a curated one); anything
    else is handed through untouched."""
    to_dict = getattr(cfg, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            pass
    return cfg


class WandbLogWriter(SummaryWriter, LogWriter):
    """Summary writer for W&B."""

    def __init__(self, log_dir: str, project_name: str) -> None:
        """Initialize a W&B run for logging."""
        if wandb is None:
            raise ModuleNotFoundError("wandb package is required to log to Weights and Biases.")
        super().__init__(log_dir, flush_secs=10)

        # Get the run name
        run_name = os.path.split(log_dir)[-1]

        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            entity = None

        # Initialize wandb
        wandb.init(
            project=project_name,
            entity=entity,
            name=run_name,
            config={"log_dir": log_dir},
            settings=wandb.Settings(start_method="thread"),
        )

        # Initialize set to keep track of logged videos
        self.logged_videos: set[str] = set()

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        """Log a scalar to both TensorBoard and W&B."""
        super().add_scalar(tag, scalar_value, global_step=global_step, walltime=walltime, new_style=new_style)
        wandb.log({tag: scalar_value}, step=global_step)

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        """Upload environment and training configuration to W&B.

        Both trees go through :func:`_json_safe`, so the FULL resolved config
        reaches ``wandb.config`` as filterable leaves -- the AMP ``amp`` block
        (discriminator + use_lerp / task_reward_lerp / disc_mode)
        on the train side, and the reward/event/observation ``func`` import paths
        (not ``<function .. at 0x..>`` noise) on the env side. Both trees are
        best-effort: a serialization or upload hiccup on either must never abort a
        training run, so each update is wrapped (train_cfg is primary, env_cfg
        secondary; both warn on failure)."""
        try:
            wandb.config.update({"train_cfg": _json_safe(train_cfg)})
        except Exception as exc:  # pragma: no cover - defensive, never fatal
            warnings.warn(f"Could not log train_cfg to wandb: {exc}", stacklevel=2)
        try:
            wandb.config.update({"env_cfg": _json_safe(_config_to_dict(env_cfg))})
        except Exception as exc:  # pragma: no cover - defensive, never fatal
            warnings.warn(f"Could not log env_cfg to wandb: {exc}", stacklevel=2)

    def save_model(self, model_path: str, it: int) -> None:
        """Upload a model checkpoint artifact to W&B."""
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path: str) -> None:
        """Upload an arbitrary file artifact to W&B."""
        wandb.save(path, base_path=os.path.dirname(path))

    def save_video(self, video: pathlib.Path, it: int) -> None:
        """Upload a video artifact once per filename to W&B."""
        if video.name not in self.logged_videos:
            wandb.log({"video": wandb.Video(str(video), format="mp4")}, step=it)
            self.logged_videos.add(video.name)

    def stop(self) -> None:
        """Finish the active W&B run."""
        wandb.finish()
