from mjlab.rl.config import RslRlOnPolicyRunnerCfg
from mjlab.rl.shared.config import RslRlAMPOnPolicyRunnerCfg
from mjlab.tasks.codancing.rl.runner import (
  CodancingAMPRunner as CodancingAMPRunner,
)
from mjlab.tasks.codancing.rl.runner import (
  CodancingOnPolicyRunner as CodancingOnPolicyRunner,
)

# Typed runner-class dispatch: the built runner cfg's type selects the codancing
# runner class. The runner is instantiated from the `runner:` block's
# `_target_`, and compose maps its type here.
RUNNER_CLS_BY_CFG: dict[type, type] = {
  RslRlOnPolicyRunnerCfg: CodancingOnPolicyRunner,
  RslRlAMPOnPolicyRunnerCfg: CodancingAMPRunner,
}
