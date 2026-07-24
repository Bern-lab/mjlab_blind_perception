"""Plain LSTM ablation on the SlowLatent target-navigation task."""

from mjlab.envs import ManagerBasedRlEnvCfg

from .blind_rough_slow_latent_env_cfg import (
  unitree_g1_blind_rough_target_navigation_slow_latent_env_cfg,
)
from .blind_stairs_flag_teacher_kl_env_cfg import _remove_slow_latent_training_state


def unitree_g1_blind_rough_target_navigation_ablation_env_cfg(
  play: bool = False,
  actor_history_length: int = 5,
) -> ManagerBasedRlEnvCfg:
  """Create a non-latent ablation env with SlowLatent-matched conditions."""
  cfg = unitree_g1_blind_rough_target_navigation_slow_latent_env_cfg(
    play=play,
    actor_history_length=actor_history_length,
  )
  _remove_slow_latent_training_state(cfg)
  return cfg


def unitree_g1_blind_rough_lstm_teacherkl_env_cfg(
  play: bool = False,
  actor_history_length: int = 5,
) -> ManagerBasedRlEnvCfg:
  """Create the plain LSTM ablation with SlowLatent-matched conditions."""
  return unitree_g1_blind_rough_target_navigation_ablation_env_cfg(
    play=play,
    actor_history_length=actor_history_length,
  )
