"""AMP env<->algorithm contract.

The single source of truth for the handshake between the environment (which
writes the AMP observations) and the AMP algorithm (which reads them): the
observation-group keys, the terminal-observation contract, and a check that the
discriminator mode is consistent with the number of discriminator groups.

No task vocabulary lives here -- only the handshake constants and the check. The
rsl_rl AMP algorithm and the downstream task that writes the AMP observations
both import these names so they cannot drift.
"""

from __future__ import annotations

from typing import Literal

# --- Observation-group keys (env writes, AMP algorithm reads) ----------------
# obs[AMP_OBS_KEY]: (num_envs, num_frames, feature_dim) float -- stacked AMP
#   feature windows (an obs group with history_length=num_frames and
#   flatten_history_dim=False).
AMP_OBS_KEY = "amp"
# obs[AMP_GROUP_KEY]: (num_envs,) long -- per-env discriminator-group index in
#   [0, num_disc_groups). Absent (None) in single-discriminator mode.
AMP_GROUP_KEY = "amp_group"

# --- Terminal-observation contract ---------------------------------------------
# Filled by mjlab's capture_terminal_observations option
# (https://github.com/mujocolab/mjlab/pull/1030):
# extras[TERMINAL_KEY] = {
#   TERMINAL_ENV_IDS: LongTensor (k,)            -- done env indices,
#   TERMINAL_OBS: {group: Tensor (k, ...), ...}  -- true terminal obs per group,
# }
# Present only on steps where >=1 env resets, and only when the env cfg enables
# capture_terminal_observations.
TERMINAL_KEY = "terminal"
TERMINAL_ENV_IDS = "env_ids"
TERMINAL_OBS = "observations"

DiscMode = Literal["single", "multi_head", "separate"]


def validate_disc_mode(disc_mode: str, num_disc_groups: int) -> None:
    """Assert the discriminator mode is consistent with the number of groups.

    ``single`` uses one merged discriminator over all experts (exactly 1 group).
    ``multi_head`` / ``separate`` partition experts into ``num_disc_groups`` >= 1
    groups (one shared-trunk head per group, or one independent discriminator per
    group, respectively).
    """
    if disc_mode == "single":
        if num_disc_groups != 1:
            raise ValueError(
                f"disc_mode='single' expects num_disc_groups == 1, got "
                f"{num_disc_groups}. Use 'multi_head' or 'separate' for K>1 groups."
            )
    elif disc_mode in ("multi_head", "separate"):
        if num_disc_groups < 1:
            raise ValueError(f"disc_mode={disc_mode!r} expects num_disc_groups >= 1, got {num_disc_groups}.")
    else:
        raise ValueError(f"unknown disc_mode {disc_mode!r}; expected 'single', 'multi_head', or 'separate'.")
