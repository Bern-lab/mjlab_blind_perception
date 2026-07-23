"""DWAQ ablation of the G1 blind-rough slow-latent target-navigation task."""

from __future__ import annotations

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from .blind_rough_slow_latent_env_cfg import (
  G1SlowLatentEnvParams,
  unitree_g1_blind_rough_target_navigation_slow_latent_env_cfg,
)


def _reset_temporal_state(term: ObservationTermCfg) -> None:
  term.delay_min_lag = 0
  term.delay_max_lag = 0
  term.delay_per_env = True
  term.delay_hold_prob = 0.0
  term.delay_update_period = 0
  term.delay_per_env_phase = True
  term.history_length = 0
  term.flatten_history_dim = True


def _remove_slow_latent_training_only_state(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.observations.pop("latent", None)
  cfg.observations.pop("latent_labels", None)
  cfg.events.pop("reset_stair_latent_cache", None)


def _add_dwaq_observations(
  cfg: ManagerBasedRlEnvCfg,
  history_length: int,
) -> None:
  actor_obs = cfg.observations["actor"]
  actor_obs.history_length = 0
  actor_obs.flatten_history_dim = True

  cfg.observations["dwaq_history"] = ObservationGroupCfg(
    terms=deepcopy(actor_obs.terms),
    concatenate_terms=True,
    enable_corruption=actor_obs.enable_corruption,
    history_length=history_length,
    flatten_history_dim=False,
  )

  velocity_target = deepcopy(cfg.observations["critic"].terms["base_lin_vel"])
  _reset_temporal_state(velocity_target)
  velocity_target.noise = None
  cfg.observations["dwaq_velocity_target"] = ObservationGroupCfg(
    terms={"base_lin_vel": velocity_target},
    concatenate_terms=True,
    enable_corruption=False,
    history_length=0,
  )


def unitree_g1_blind_rough_target_navigation_dwaq_env_cfg(
  play: bool = False,
  params: G1SlowLatentEnvParams | None = None,
  history_length: int = 5,
) -> ManagerBasedRlEnvCfg:
  """Create the clean DWAQ ablation of the slow-latent TeacherKL task.

  Rewards, terrain replay, target-navigation commands, teacher observations, and
  privileged critic observations come from the current SlowLatent task. Only the
  student policy observations are reshaped into DWAQ current/history/velocity
  target groups.
  """
  cfg = unitree_g1_blind_rough_target_navigation_slow_latent_env_cfg(
    play=play,
    params=params,
    actor_history_length=0,
  )
  _remove_slow_latent_training_only_state(cfg)
  _add_dwaq_observations(cfg, history_length=history_length)
  return cfg
