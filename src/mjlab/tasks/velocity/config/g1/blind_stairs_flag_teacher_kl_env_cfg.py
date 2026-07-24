"""Boolean stair-flag ablation on the SlowLatent target-navigation task."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .blind_rough_slow_latent_env_cfg import (
  unitree_g1_blind_rough_target_navigation_slow_latent_env_cfg,
)

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

STAIRS_FLAG_ACTOR_HISTORY_LENGTH = 5
STAIRS_FLAG_CRITIC_HISTORY_LENGTH = 3


def _remove_slow_latent_training_state(cfg: ManagerBasedRlEnvCfg) -> None:
  """Keep the SlowLatent task conditions but remove latent-only training state."""
  cfg.observations.pop("latent", None)
  cfg.observations.pop("latent_labels", None)
  cfg.events.pop("reset_stair_latent_cache", None)


def _configure_actor_history_except_stair_flag(
  cfg: ManagerBasedRlEnvCfg,
  actor_history_length: int,
) -> None:
  """Keep normal actor observations historical and the stair flag current-only."""
  actor_group = cfg.observations["actor"]
  actor_group.history_length = None
  actor_group.flatten_history_dim = True

  for term_cfg in actor_group.terms.values():
    if term_cfg is None:
      continue
    term_cfg.history_length = actor_history_length
    term_cfg.flatten_history_dim = True


def terrain_is_stairs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return a dynamic 1/0 stair flag from the robot's current terrain tile."""
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    return torch.zeros(env.num_envs, 1, device=env.device)

  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None:
    return torch.zeros(env.num_envs, 1, device=env.device)

  asset = env.scene[asset_cfg.name]
  root_pos_w = asset.data.root_link_pos_w

  num_rows, num_cols = terrain.terrain_origins.shape[:2]
  tile_size_x, tile_size_y = terrain_generator.size

  grid_min_x = -0.5 * num_rows * float(tile_size_x)
  grid_min_y = -0.5 * num_cols * float(tile_size_y)
  grid_max_x = grid_min_x + num_rows * float(tile_size_x)
  grid_max_y = grid_min_y + num_cols * float(tile_size_y)

  inside_grid = (
    (root_pos_w[:, 0] >= grid_min_x)
    & (root_pos_w[:, 0] < grid_max_x)
    & (root_pos_w[:, 1] >= grid_min_y)
    & (root_pos_w[:, 1] < grid_max_y)
  )

  terrain_rows = torch.floor(
    (root_pos_w[:, 0] - grid_min_x) / float(tile_size_x)
  ).long()
  terrain_cols = torch.floor(
    (root_pos_w[:, 1] - grid_min_y) / float(tile_size_y)
  ).long()
  terrain_rows = terrain_rows.clamp(0, num_rows - 1)
  terrain_cols = terrain_cols.clamp(0, num_cols - 1)

  step_boundary_counts = getattr(terrain, "step_boundary_counts", None)
  if (
    step_boundary_counts is not None
    and step_boundary_counts.numel() > 0
    and step_boundary_counts.shape[0] == num_rows
    and step_boundary_counts.shape[1] == num_cols
  ):
    is_stairs = step_boundary_counts[terrain_rows, terrain_cols] > 0
    is_stairs = is_stairs & inside_grid
    return is_stairs.float().unsqueeze(-1)

  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  if terrain_generator.curriculum and num_cols == len(sub_terrain_names):
    is_stair_col = torch.zeros(num_cols, dtype=torch.bool, device=env.device)
    for idx, name in enumerate(sub_terrain_names):
      is_stair_col[idx] = "stairs" in name
    is_stairs = is_stair_col[terrain_cols] & inside_grid
    return is_stairs.float().unsqueeze(-1)

  return torch.zeros(env.num_envs, 1, device=env.device)


def terrain_is_stairs_metric(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Scalar metric companion for play-time terrain display."""
  return terrain_is_stairs(env, asset_cfg=asset_cfg).squeeze(-1)


def unitree_g1_blind_stairs_flag_teacherkl_env_cfg(
  play: bool = False,
  actor_history_length: int = STAIRS_FLAG_ACTOR_HISTORY_LENGTH,
  critic_history_length: int = STAIRS_FLAG_CRITIC_HISTORY_LENGTH,
) -> ManagerBasedRlEnvCfg:
  """Create the Boolean-MLP/LSTM ablation with SlowLatent-matched conditions."""
  cfg = unitree_g1_blind_rough_target_navigation_slow_latent_env_cfg(
    play=play,
    actor_history_length=actor_history_length,
  )
  _remove_slow_latent_training_state(cfg)
  cfg.observations["critic"].history_length = critic_history_length

  _configure_actor_history_except_stair_flag(
    cfg,
    actor_history_length=actor_history_length,
  )
  cfg.observations["actor"].terms["terrain_is_stairs"] = ObservationTermCfg(
    func=terrain_is_stairs,
    history_length=0,
    flatten_history_dim=True,
  )
  cfg.metrics["terrain_is_stairs"] = MetricsTermCfg(func=terrain_is_stairs_metric)

  return cfg
