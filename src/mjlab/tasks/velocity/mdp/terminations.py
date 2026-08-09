from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


def out_of_terrain_bounds(
  env: ManagerBasedRlEnv,
  margin: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Truncate if robot leaves the generated terrain footprint.

  Returns all-false for non-generator terrains (e.g. plane).
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_type != "generator":
    return torch.zeros(
      (env.num_envs,),
      device=env.device,
      dtype=torch.bool,
    )

  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None:
    return torch.zeros(
      (env.num_envs,),
      device=env.device,
      dtype=torch.bool,
    )

  asset: Entity = env.scene[asset_cfg.name]
  root_xy_w = asset.data.root_link_pos_w[:, :2]

  standalone = (
    terrain.is_standalone_env()
    if hasattr(terrain, "is_standalone_env")
    else torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  )
  half_x = 0.5 * (terrain_generator.num_rows * terrain_generator.size[0])
  half_y = 0.5 * (terrain_generator.num_cols * terrain_generator.size[1])
  limit_x = max(0.0, half_x - margin)
  limit_y = max(0.0, half_y - margin)

  out = (root_xy_w[:, 0].abs() > limit_x) | (root_xy_w[:, 1].abs() > limit_y)
  return out & ~standalone


def runway_out_of_bounds(
  env: ManagerBasedRlEnv,
  margin: float = 0.2,
  drop_height: float = 0.4,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate standalone runway envs when the robot leaves the track."""
  terrain = env.scene.terrain
  if (
    terrain is None
    or terrain.cfg.terrain_type != "generator"
    or not hasattr(terrain, "is_standalone_env")
  ):
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  standalone = terrain.is_standalone_env()
  if not standalone.any():
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  asset: Entity = env.scene[asset_cfg.name]
  root_pos_w = asset.data.root_link_pos_w
  bounds = terrain.terrain_bounds_by_tile[terrain.terrain_levels, terrain.terrain_types]
  outside_xy = (
    (root_pos_w[:, 0] < bounds[:, 0] - margin)
    | (root_pos_w[:, 0] > bounds[:, 1] + margin)
    | (root_pos_w[:, 1] < bounds[:, 2] - margin)
    | (root_pos_w[:, 1] > bounds[:, 3] + margin)
  )
  below_start = root_pos_w[:, 2] < env.scene.env_origins[:, 2] - drop_height
  return standalone & (outside_xy | below_start) & (env.episode_length_buf > 2)


def runway_target_reached(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """End standalone runway episodes once the fixed end target is reached."""
  terrain = env.scene.terrain
  if terrain is None or not hasattr(terrain, "is_standalone_env"):
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  command_term = env.command_manager.get_term(command_name)
  target_pos_w = getattr(command_term, "target_pos_w", None)
  if not isinstance(target_pos_w, torch.Tensor):
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  if threshold is None:
    command_cfg = getattr(command_term, "cfg", None)
    threshold = float(getattr(command_cfg, "target_reached_threshold", 0.5))

  asset: Entity = env.scene[asset_cfg.name]
  dist = torch.linalg.norm(
    target_pos_w[:, :2] - asset.data.root_link_pos_w[:, :2],
    dim=1,
  )
  target_env = getattr(command_term, "is_target_env", None)
  if not isinstance(target_env, torch.Tensor):
    target_env = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
  reached = terrain.is_standalone_env() & target_env & (dist <= threshold)
  reached_in_episode = getattr(command_term, "target_reached_in_episode", None)
  if isinstance(reached_in_episode, torch.Tensor) and reached.any():
    reached_in_episode[reached] = True
  return reached


def terrain_edge_reached(
  env: ManagerBasedRlEnv,
  threshold_fraction: float = 0.95,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when robot displacement from spawn exceeds sub-terrain size.

  Intended as ``time_out=True`` (successful traversal, not penalized). Skips the first
  2 steps after reset to avoid stale-position triggers.
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_type != "generator":
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  asset: Entity = env.scene[asset_cfg.name]
  displacement = (
    asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  ).abs()

  half_x = terrain_generator.size[0] / 2.0 * threshold_fraction
  half_y = terrain_generator.size[1] / 2.0 * threshold_fraction

  at_edge = (displacement[:, 0] > half_x) | (displacement[:, 1] > half_y)

  # Don't fire on the first 2 steps after reset (position may be stale).
  at_edge &= env.episode_length_buf > 2

  return at_edge
