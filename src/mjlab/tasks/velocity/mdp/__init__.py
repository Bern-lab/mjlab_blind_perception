from typing import Any

from mjlab.envs.mdp import *  # noqa: F401, F403

from . import temporal_stair_rewards as _temporal_stair_rewards
from .curriculums import *  # noqa: F403
from .observations import *  # noqa: F403
from .rewards import *  # noqa: F403
from .stair_sequence_logging import (
  stair_sequence_event_logger as stair_sequence_event_logger,
)
from .target_heading_command import *  # noqa: F403
from .terminations import *  # noqa: F403
from .velocity_command import *  # noqa: F403

toe_step_riser_slab_penalty: Any = _temporal_stair_rewards.toe_step_riser_slab_penalty
toe_step_riser_approach_penalty: Any = (
  _temporal_stair_rewards.toe_step_riser_approach_penalty
)
stair_aware_feet_gait = _temporal_stair_rewards.stair_aware_feet_gait
stair_stride_phase_reward = _temporal_stair_rewards.stair_stride_phase_reward
stair_skip_layer_penalty = _temporal_stair_rewards.stair_skip_layer_penalty
target_tread_midline_shaping = _temporal_stair_rewards.target_tread_midline_shaping
stair_tread_landing_reward = _temporal_stair_rewards.stair_tread_landing_reward
