from typing import Any

from mjlab.envs.mdp import *  # noqa: F401, F403

from . import temporal_stair_rewards as _temporal_stair_rewards
from .curriculums import *  # noqa: F403
from .observations import *  # noqa: F403
from .rewards import *  # noqa: F403
from .target_heading_command import *  # noqa: F403
from .terminations import *  # noqa: F403
from .velocity_command import *  # noqa: F403

toe_step_riser_slab_penalty: Any = _temporal_stair_rewards.toe_step_riser_slab_penalty
toe_step_riser_approach_penalty: Any = (
  _temporal_stair_rewards.toe_step_riser_approach_penalty
)
probe_aware_feet_gait = _temporal_stair_rewards.probe_aware_feet_gait
stair_tread_landing_reward = _temporal_stair_rewards.stair_tread_landing_reward
