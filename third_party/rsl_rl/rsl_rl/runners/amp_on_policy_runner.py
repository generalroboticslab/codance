"""AMP on-policy runner.

Thin shell over OnPolicyRunner for config class resolution and
inheritance chain compatibility. All AMP logic lives in AMPMixin
(applied at the algorithm level, not the runner level).
"""

from __future__ import annotations

from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class AMPOnPolicyRunner(OnPolicyRunner):
    """Runner for AMP algorithms.

    With the obs_groups interface, the stock ``OnPolicyRunner.learn()``
    loop works unmodified: AMP observations flow through the standard
    ``obs`` TensorDict and discriminator reward is computed inside
    ``AMPMixin.process_env_step()``.

    This class exists for:
    1. Config class resolution (``class_name: "AMPOnPolicyRunner"``).
    2. Downstream inheritance (mjlab's ``CodancingAMPRunner`` extends this).
    """

    pass
