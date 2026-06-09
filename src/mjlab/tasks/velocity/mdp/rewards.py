from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.velocity.mdp.target_heading_command import (
    TargetHeadingVelocityCommand,
  )
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_FOOT_BODY_CFG = SceneEntityCfg(
  "robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link")
)
_DEFAULT_FOOT_SITE_CFG = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))


def _make_foot_volume_points(
  device: str,
  x_range: tuple[float, float] = (-0.055, 0.132),
  y_range: tuple[float, float] = (-0.030, 0.030),
  z_range: tuple[float, float] = (-0.035, -0.015),
  grid_shape: tuple[int, int, int] = (8, 4, 2),
  heel_x_max: float = -0.020,
  front_sole_x_min: float = 0.070,
  toe_tip_x_min: float = 0.115,
  heel_weight: float = 0.5,
  front_sole_weight: float = 0.7,
  midfoot_weight: float = 1.0,
  toe_tip_weight: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
  xs = torch.linspace(x_range[0], x_range[1], grid_shape[0], device=device)
  ys = torch.linspace(y_range[0], y_range[1], grid_shape[1], device=device)
  zs = torch.linspace(z_range[0], z_range[1], grid_shape[2], device=device)
  xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
  points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

  x = points[:, 0]
  weights = torch.full_like(x, midfoot_weight)
  weights = torch.where(x < heel_x_max, heel_weight, weights)
  front_mask = (x >= front_sole_x_min) & (x < toe_tip_x_min)
  weights = torch.where(front_mask, front_sole_weight, weights)
  weights = torch.where(x >= toe_tip_x_min, toe_tip_weight, weights)
  return points, weights


def _current_step_boundaries(
  env: ManagerBasedRlEnv,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
  terrain = getattr(env.scene, "terrain", None)
  if terrain is None or not hasattr(terrain, "step_boundaries_by_tile"):
    return None, None
  if getattr(terrain, "terrain_levels", None) is None:
    return None, None

  boundaries_by_tile = terrain.step_boundaries_by_tile
  if boundaries_by_tile.shape[2] == 0:
    return None, None

  levels = terrain.terrain_levels
  terrain_types = terrain.terrain_types
  boundaries = boundaries_by_tile[levels, terrain_types]
  counts = terrain.step_boundary_counts[levels, terrain_types]
  boundary_ids = torch.arange(boundaries.shape[1], device=env.device)
  valid = boundary_ids.unsqueeze(0) < counts.unsqueeze(1)
  return boundaries, valid


def _terrain_level_active(
  env: ManagerBasedRlEnv, min_terrain_level: int | None
) -> torch.Tensor:
  if min_terrain_level is None:
    return torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

  terrain = getattr(env.scene, "terrain", None)
  levels = getattr(terrain, "terrain_levels", None)
  if levels is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  return levels >= min_terrain_level


def _step_boundary_layers(
  boundaries: torch.Tensor,
  valid_boundaries: torch.Tensor,
  max_layers: int,
) -> torch.Tensor:
  layers = torch.zeros(
    valid_boundaries.shape,
    device=boundaries.device,
    dtype=torch.long,
  )
  if max_layers <= 0:
    return layers

  z_low = boundaries[..., 9]
  z_high = boundaries[..., 10]
  step_heights = torch.abs(z_high - z_low)
  valid = valid_boundaries & (step_heights > 1.0e-6)
  if not bool(torch.any(valid).item()):
    return layers

  masked_heights = torch.where(
    valid,
    step_heights,
    torch.full_like(step_heights, torch.inf),
  )
  env_step_height = torch.min(masked_heights, dim=-1).values
  env_step_height = torch.where(
    torch.isfinite(env_step_height),
    env_step_height.clamp_min(1.0e-6),
    torch.ones_like(env_step_height),
  )
  distance_from_base = torch.minimum(torch.abs(z_low), torch.abs(z_high))
  candidate_layers = torch.round(distance_from_base / env_step_height[:, None])
  candidate_layers = candidate_layers.long() + 1
  valid_layers = valid & (candidate_layers <= max_layers)
  return torch.where(valid_layers, candidate_layers, layers)


class _StepBoundaryFootVolume:
  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    self._foot_ref_local = torch.tensor(
      params.get("foot_ref_local", (0.04, 0.0, -0.025)),
      device=env.device,
      dtype=torch.float32,
    )
    self._local_points, self._point_weights = _make_foot_volume_points(
      env.device,
      x_range=params.get("x_range", (-0.055, 0.132)),
      y_range=params.get("y_range", (-0.030, 0.030)),
      z_range=params.get("z_range", (-0.035, -0.015)),
      grid_shape=params.get("grid_shape", (8, 4, 2)),
      heel_x_max=params.get("heel_x_max", -0.020),
      front_sole_x_min=params.get("front_sole_x_min", 0.070),
      toe_tip_x_min=params.get("toe_tip_x_min", 0.115),
      heel_weight=params.get("heel_weight", 0.5),
      front_sole_weight=params.get("front_sole_weight", 0.7),
      midfoot_weight=params.get("midfoot_weight", 1.0),
      toe_tip_weight=params.get("toe_tip_weight", 0.3),
    )
    self._local_x = self._local_points[:, 0]
    self._max_point_ref_distance = torch.norm(
      self._local_points - self._foot_ref_local, dim=-1
    ).max()

  def _foot_points_w(
    self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
  ) -> tuple[torch.Tensor, torch.Tensor]:
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    foot_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    foot_lin_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]
    foot_ang_vel_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]

    num_envs, num_feet = foot_pos_w.shape[:2]
    num_points = self._local_points.shape[0]
    local_points = self._local_points.view(1, 1, num_points, 3).expand(
      num_envs, num_feet, num_points, 3
    )
    foot_quat = foot_quat_w[:, :, None, :].expand(num_envs, num_feet, num_points, 4)
    point_offsets_w = quat_apply(foot_quat, local_points)
    points_w = foot_pos_w[:, :, None, :] + point_offsets_w
    point_vel_w = foot_lin_vel_w[:, :, None, :] + torch.cross(
      foot_ang_vel_w[:, :, None, :].expand_as(point_offsets_w),
      point_offsets_w,
      dim=-1,
    )
    return points_w, point_vel_w

  def _foot_ref_w(
    self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    foot_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    num_envs, num_feet = foot_pos_w.shape[:2]
    foot_ref_local = self._foot_ref_local.view(1, 1, 3).expand(num_envs, num_feet, 3)
    return foot_pos_w + quat_apply(foot_quat_w, foot_ref_local)

  def _flat_point_weights(self, num_envs: int, num_feet: int) -> torch.Tensor:
    return (
      self._point_weights.view(1, 1, -1)
      .expand(num_envs, num_feet, -1)
      .reshape(num_envs, num_feet * self._point_weights.shape[0])
    )

  @staticmethod
  def _point_to_segment_distance(
    points: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
  ) -> torch.Tensor:
    segment = p1 - p0
    segment_len_sq = torch.sum(torch.square(segment), dim=-1).clamp_min(1e-12)
    point_delta = points[:, :, :, None, :] - p0[:, :, None, :, :]
    t = torch.sum(point_delta * segment[:, :, None, :, :], dim=-1)
    t = t / segment_len_sq[:, :, None, :]
    t = torch.clamp(t, 0.0, 1.0)
    closest = p0[:, :, None, :, :] + t[..., None] * segment[:, :, None, :, :]
    return torch.norm(points[:, :, :, None, :] - closest, dim=-1)

  @staticmethod
  def _ref_to_segment_distance(
    refs: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
  ) -> torch.Tensor:
    segment = p1 - p0
    segment_len_sq = torch.sum(torch.square(segment), dim=-1).clamp_min(1e-12)
    ref_delta = refs[:, :, None, :] - p0[:, None, :, :]
    t = torch.sum(ref_delta * segment[:, None, :, :], dim=-1)
    t = t / segment_len_sq[:, None, :]
    t = torch.clamp(t, 0.0, 1.0)
    closest = p0[:, None, :, :] + t[..., None] * segment[:, None, :, :]
    return torch.norm(refs[:, :, None, :] - closest, dim=-1)

  @staticmethod
  def _gather_by_foot(
    values: torch.Tensor,
    indices: torch.Tensor,
  ) -> torch.Tensor:
    num_envs, num_feet, num_selected = indices.shape
    value_dim = values.shape[-1]
    expanded = values[:, None, :, :].expand(num_envs, num_feet, -1, value_dim)
    gather_idx = indices[..., None].expand(num_envs, num_feet, num_selected, value_dim)
    return torch.gather(expanded, dim=2, index=gather_idx)

  @staticmethod
  def _gather_mask_by_foot(
    values: torch.Tensor,
    indices: torch.Tensor,
  ) -> torch.Tensor:
    num_envs, num_feet, num_selected = indices.shape
    expanded = values[:, None, :].expand(num_envs, num_feet, -1)
    return torch.gather(expanded, dim=2, index=indices)

  def _nearest_boundary_indices(
    self,
    ref_distance: torch.Tensor,
    valid_boundaries: torch.Tensor,
    influence_radius: torch.Tensor | float,
    nearest_boundaries: int | None,
  ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    num_boundaries = ref_distance.shape[-1]
    if (
      nearest_boundaries is None
      or nearest_boundaries <= 0
      or nearest_boundaries >= num_boundaries
    ):
      return None, None

    candidate_count = torch.sum(
      (ref_distance <= influence_radius) & valid_boundaries[:, None, :], dim=-1
    )
    fallback = candidate_count > nearest_boundaries
    masked_distance = torch.where(
      valid_boundaries[:, None, :],
      ref_distance,
      torch.full_like(ref_distance, torch.inf),
    )
    indices = torch.topk(
      masked_distance, k=nearest_boundaries, dim=-1, largest=False
    ).indices
    return indices, fallback

  def _lip_min_dist(
    self,
    points: torch.Tensor,
    boundaries: torch.Tensor,
    valid_boundaries: torch.Tensor,
    edge_height_band: float | None,
  ) -> torch.Tensor:
    p0 = boundaries[..., 0:3]
    p1 = boundaries[..., 3:6]
    z_high = boundaries[..., 10]

    distances = self._point_to_segment_distance(points, p0, p1)
    valid = valid_boundaries[:, :, None, :]
    if edge_height_band is not None and edge_height_band > 0.0:
      height_ok = points[:, :, :, None, 2] >= z_high[:, :, None, :] - edge_height_band
      valid = valid & height_ok
    distances = torch.where(valid, distances, torch.full_like(distances, torch.inf))
    return torch.min(distances, dim=-1).values

  def _riser_slab_ref_distance(
    self,
    refs: torch.Tensor,
    boundaries: torch.Tensor,
    slab_depth: float,
    u_margin: float,
    v_margin: float,
    surface_tol: float,
  ) -> torch.Tensor:
    p0 = boundaries[:, :, 0:3]
    p1 = boundaries[:, :, 3:6]
    normal_to_low = boundaries[:, :, 6:9]
    z_low = boundaries[:, :, 9]
    z_high = boundaries[:, :, 10]

    tangent_u = p1 - p0
    edge_len = torch.norm(tangent_u, dim=-1).clamp_min(1e-12)
    tangent_u = tangent_u / edge_len[..., None]
    center = 0.5 * (p0 + p1)
    center = center.clone()
    center[:, :, 2] = 0.5 * (z_low + z_high)
    half_u = 0.5 * edge_len
    half_v = 0.5 * (z_high - z_low)

    rel = refs[:, :, None, :] - center[:, None, :, :]
    s = torch.sum(rel * normal_to_low[:, None, :, :], dim=-1)
    u = torch.sum(rel * tangent_u[:, None, :, :], dim=-1)
    v = rel[..., 2]

    du = torch.relu(torch.abs(u) - (half_u[:, None, :] + u_margin))
    ds_low = torch.relu(-surface_tol - s)
    ds_high = torch.relu(s - slab_depth)
    ds = torch.maximum(ds_low, ds_high)
    dv = torch.relu(torch.abs(v) - (half_v[:, None, :] + v_margin))
    return torch.sqrt(torch.square(du) + torch.square(ds) + torch.square(dv))

  def _riser_slab_point_penalty(
    self,
    toe_points: torch.Tensor,
    toe_vel: torch.Tensor,
    boundaries: torch.Tensor,
    valid_boundaries: torch.Tensor,
    slab_depth: float,
    u_margin: float,
    v_margin: float,
    toe_v_threshold: float,
    surface_tol: float,
    boundary_layers: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    p0 = boundaries[..., 0:3]
    p1 = boundaries[..., 3:6]
    normal_to_low = boundaries[..., 6:9]
    z_low = boundaries[..., 9]
    z_high = boundaries[..., 10]

    tangent_u = p1 - p0
    edge_len = torch.norm(tangent_u, dim=-1).clamp_min(1e-12)
    tangent_u = tangent_u / edge_len[..., None]
    center = 0.5 * (p0 + p1)
    center = center.clone()
    center[..., 2] = 0.5 * (z_low + z_high)
    half_u = 0.5 * edge_len
    half_v = 0.5 * (z_high - z_low)

    rel = toe_points[:, :, :, None, :] - center[:, :, None, :, :]
    s = torch.sum(rel * normal_to_low[:, :, None, :, :], dim=-1)
    u = torch.sum(rel * tangent_u[:, :, None, :, :], dim=-1)
    v = rel[..., 2]
    inside_face = (torch.abs(u) <= half_u[:, :, None, :] + u_margin) & (
      torch.abs(v) <= half_v[:, :, None, :] + v_margin
    )
    inside_slab = (s >= -surface_tol) & (s <= slab_depth)
    toe_approach_speed = torch.relu(
      -torch.sum(toe_vel[:, :, :, None, :] * normal_to_low[:, :, None, :, :], dim=-1)
      - toe_v_threshold
    )
    penetration = torch.relu(slab_depth - s)
    valid = valid_boundaries[:, :, None, :] & inside_face & inside_slab
    per_face_penalty = torch.where(
      valid, penetration * toe_approach_speed, torch.zeros_like(penetration)
    )
    point_penalty, best_face_idx = torch.max(per_face_penalty, dim=-1)
    impact_speed_per_point = torch.max(
      torch.where(valid, toe_approach_speed, torch.zeros_like(toe_approach_speed)),
      dim=-1,
    ).values
    active = point_penalty > 0.0
    if boundary_layers is None:
      point_layers = torch.zeros_like(point_penalty, dtype=torch.long)
    else:
      expanded_layers = boundary_layers[:, :, None, :].expand(
        *best_face_idx.shape,
        boundary_layers.shape[-1],
      )
      point_layers = torch.gather(
        expanded_layers,
        dim=-1,
        index=best_face_idx[..., None],
      ).squeeze(-1)
      point_layers = torch.where(active, point_layers, torch.zeros_like(point_layers))
    return point_penalty, active, impact_speed_per_point, point_layers


class foot_step_lip_volume_penalty(_StepBoundaryFootVolume):
  """Hiking-style foot-volume penalty around high-side step lips."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    edge_radius: float = 0.05,
    edge_height_band: float | None = 0.06,
    nearest_boundaries: int | None = None,
    ignore_boundary_layers: int = 0,
    log_only: bool = False,
    min_terrain_level: int | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_BODY_CFG,
    **_: object,
  ) -> torch.Tensor:
    boundaries, valid_boundaries = _current_step_boundaries(env)
    if boundaries is None or valid_boundaries is None:
      return torch.zeros(env.num_envs, device=env.device)

    points_w, point_vel_w = self._foot_points_w(env, asset_cfg)
    num_envs, num_feet, num_points = points_w.shape[:3]

    level_active = _terrain_level_active(env, min_terrain_level)
    base_valid = valid_boundaries & level_active[:, None]
    valid_before_layer_ignore = base_valid
    ignored_layers = _step_boundary_layers(
      boundaries,
      base_valid,
      max(0, int(ignore_boundary_layers)),
    )
    ignored_boundaries = ignored_layers > 0
    if ignore_boundary_layers > 0:
      base_valid = base_valid & ~ignored_boundaries

    p0 = boundaries[:, :, 0:3]
    p1 = boundaries[:, :, 3:6]
    foot_ref_w = self._foot_ref_w(env, asset_cfg)
    ref_dist = self._ref_to_segment_distance(foot_ref_w, p0, p1)
    influence_radius = edge_radius + self._max_point_ref_distance
    selected_idx, fallback = self._nearest_boundary_indices(
      ref_dist, base_valid, influence_radius, nearest_boundaries
    )

    if selected_idx is None:
      expanded_boundaries = boundaries[:, None, :, :].expand(num_envs, num_feet, -1, -1)
      expanded_valid = base_valid[:, None, :].expand(num_envs, num_feet, -1)
      min_dist = self._lip_min_dist(
        points_w, expanded_boundaries, expanded_valid, edge_height_band
      )
      fallback_ratio = torch.zeros((), device=env.device)
    else:
      assert fallback is not None
      selected_boundaries = self._gather_by_foot(boundaries, selected_idx)
      selected_valid = self._gather_mask_by_foot(base_valid, selected_idx)
      min_dist = self._lip_min_dist(
        points_w, selected_boundaries, selected_valid, edge_height_band
      )
      fallback_ratio = fallback.float().mean()
      if bool(torch.any(fallback).item()):
        expanded_boundaries = boundaries[:, None, :, :].expand(
          num_envs, num_feet, -1, -1
        )
        expanded_valid = base_valid[:, None, :].expand(num_envs, num_feet, -1)
        full_min_dist = self._lip_min_dist(
          points_w, expanded_boundaries, expanded_valid, edge_height_band
        )
        min_dist = torch.where(fallback[:, :, None], full_min_dist, min_dist)

    penetration = torch.relu(edge_radius - min_dist)
    point_speed = torch.norm(point_vel_w, dim=-1)
    weights = self._point_weights.view(1, 1, num_points)
    penalty = torch.sum(weights * penetration * (point_speed + 1e-6), dim=(1, 2))

    finite = torch.isfinite(min_dist)
    finite_count = finite.float().sum().clamp_min(1.0)
    min_dist_mean = (
      torch.where(finite, min_dist, torch.zeros_like(min_dist)).sum() / finite_count
    )
    env.extras["log"]["Metrics/step_lip_penalty_mean"] = penalty.mean()
    env.extras["log"]["Metrics/step_lip_penetration_ratio"] = (
      (penetration > 0.0).float().mean()
    )
    env.extras["log"]["Metrics/step_lip_min_dist_mean"] = min_dist_mean
    env.extras["log"]["Metrics/step_lip_nearest_fallback_ratio"] = fallback_ratio
    ignored_count = (valid_before_layer_ignore & ignored_boundaries).float().sum()
    valid_count = valid_before_layer_ignore.float().sum().clamp_min(1.0)
    env.extras["log"]["Metrics/step_lip_ignored_layer_ratio"] = (
      ignored_count / valid_count
    )

    if log_only:
      return torch.zeros_like(penalty)
    return penalty


class toe_step_riser_slab_penalty(_StepBoundaryFootVolume):
  """Penalize toe points entering the low-side danger slab of a riser."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_FOOT_BODY_CFG)
    num_feet = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else 2
    probe_contact_count = max(0, int(cfg.params.get("probe_contact_count", 0)))
    self._probe_layer_hit = torch.zeros(
      (env.num_envs, max(1, probe_contact_count)),
      device=env.device,
      dtype=torch.bool,
    )
    self._probe_hit_cooldown = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float32
    )
    self._second_layer_attraction_progress = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
    self._root_z_baseline = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
    self._max_root_z = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    self._ascent_active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self._needs_root_z_init = torch.ones(
      env.num_envs, device=env.device, dtype=torch.bool
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._probe_layer_hit[env_ids] = False
    self._probe_hit_cooldown[env_ids] = 0.0
    self._second_layer_attraction_progress[env_ids] = 0.0
    self._root_z_baseline[env_ids] = 0.0
    self._max_root_z[env_ids] = 0.0
    self._ascent_active[env_ids] = False
    self._needs_root_z_init[env_ids] = True

  def _ensure_probe_layer_capacity(self, probe_contact_count: int) -> int:
    max_probe_layers = max(0, int(probe_contact_count))
    if max_probe_layers <= self._probe_layer_hit.shape[1]:
      return max_probe_layers

    extra = torch.zeros(
      (
        self._probe_layer_hit.shape[0],
        max_probe_layers - self._probe_layer_hit.shape[1],
      ),
      device=self._probe_layer_hit.device,
      dtype=torch.bool,
    )
    self._probe_layer_hit = torch.cat([self._probe_layer_hit, extra], dim=1)
    return max_probe_layers

  @staticmethod
  def _probe_boundary_layers(
    boundaries: torch.Tensor,
    valid_boundaries: torch.Tensor,
    max_probe_layers: int,
  ) -> torch.Tensor:
    return _step_boundary_layers(boundaries, valid_boundaries, max_probe_layers)

  def _ascent_gate(
    self,
    env: ManagerBasedRlEnv,
    asset: Entity,
    level_active: torch.Tensor,
    interaction_observed: torch.Tensor,
    min_ascent_height: float,
    ascent_velocity_threshold: float,
  ) -> torch.Tensor:
    root_z = asset.data.root_link_pos_w[:, 2]
    needs_init = self._needs_root_z_init
    self._root_z_baseline = torch.where(needs_init, root_z, self._root_z_baseline)
    self._max_root_z = torch.where(needs_init, root_z, self._max_root_z)
    self._needs_root_z_init[:] = False

    self._max_root_z = torch.maximum(self._max_root_z, root_z)
    root_z_gain = self._max_root_z - self._root_z_baseline
    root_vz = asset.data.root_link_lin_vel_w[:, 2]
    ascent_observed = (root_z_gain >= min_ascent_height) | (
      root_vz > ascent_velocity_threshold
    )
    ascent_observed |= interaction_observed

    self._ascent_active |= level_active & ascent_observed
    active_gate = level_active & self._ascent_active
    inactive = ~active_gate
    if bool(torch.any(inactive).item()):
      self._probe_layer_hit[inactive] = False
      self._probe_hit_cooldown[inactive] = 0.0
      self._second_layer_attraction_progress[inactive] = 0.0
    self._probe_hit_cooldown = torch.clamp(
      self._probe_hit_cooldown - env.step_dt, min=0.0
    )
    return active_gate

  @staticmethod
  def _contact_probe_layers(
    contact_pos_w: torch.Tensor,
    boundaries: torch.Tensor,
    probe_boundary_layers: torch.Tensor,
  ) -> torch.Tensor:
    p0 = boundaries[:, :, 0:3]
    p1 = boundaries[:, :, 3:6]
    segment = p1 - p0
    segment_len_sq = torch.sum(torch.square(segment), dim=-1).clamp_min(1.0e-12)
    point_delta = contact_pos_w[:, :, :, None, :] - p0[:, None, None, :, :]
    t = torch.sum(point_delta * segment[:, None, None, :, :], dim=-1)
    t = torch.clamp(t / segment_len_sq[:, None, None, :], 0.0, 1.0)
    closest = p0[:, None, None, :, :] + t[..., None] * segment[:, None, None, :, :]
    distances = torch.norm(contact_pos_w[:, :, :, None, :] - closest, dim=-1)

    valid_probe_layers = probe_boundary_layers > 0
    distances = torch.where(
      valid_probe_layers[:, None, None, :],
      distances,
      torch.full_like(distances, torch.inf),
    )
    nearest_dist, nearest_idx = torch.min(distances, dim=-1)
    expanded_layers = probe_boundary_layers[:, None, None, :].expand(
      *nearest_idx.shape,
      probe_boundary_layers.shape[-1],
    )
    nearest_layers = torch.gather(
      expanded_layers,
      dim=-1,
      index=nearest_idx[..., None],
    ).squeeze(-1)
    return torch.where(
      torch.isfinite(nearest_dist),
      nearest_layers,
      torch.zeros_like(nearest_layers),
    )

  def _contact_probe_terms(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset: Entity,
    asset_cfg: SceneEntityCfg,
    active_gate: torch.Tensor,
    boundaries: torch.Tensor,
    probe_boundary_layers: torch.Tensor,
    max_probe_layers: int,
    toe_x_min: float,
    vertical_normal_z_max: float,
    forward_velocity_threshold: float,
    force_threshold: float,
    force_scale: float,
    contact_penalty_scale: float,
    contact_time_scale: float,
    probe_contact_reward: float,
    probe_cooldown_time: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    sensor = env.scene[sensor_name]
    assert isinstance(sensor, ContactSensor), (
      f"toe_step_riser_slab_penalty requires a ContactSensor for "
      f"contact_sensor_name='{sensor_name}', got {type(sensor).__name__}"
    )
    data = sensor.data
    assert data.found is not None
    assert data.force is not None
    assert data.normal is not None
    assert data.pos is not None

    num_envs = env.num_envs
    num_feet = self._probe_hit_cooldown.shape[1]
    num_contacts = data.found.shape[1]
    assert num_contacts % num_feet == 0, (
      f"Contact sensor '{sensor_name}' has {num_contacts} contact slots, which is "
      f"not divisible by {num_feet} feet"
    )
    num_slots = num_contacts // num_feet

    found = data.found.view(num_envs, num_feet, num_slots) > 0
    force_w = data.force.view(num_envs, num_feet, num_slots, 3)
    normal_w = data.normal.view(num_envs, num_feet, num_slots, 3)
    contact_pos_w = data.pos.view(num_envs, num_feet, num_slots, 3)

    foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    foot_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    foot_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]

    expanded_quat = foot_quat_w[:, :, None, :].expand(num_envs, num_feet, num_slots, 4)
    contact_pos_b = quat_apply_inverse(
      expanded_quat,
      contact_pos_w - foot_pos_w[:, :, None, :],
    )
    is_toe_contact = contact_pos_b[..., 0] >= toe_x_min

    local_forward = torch.tensor(
      (1.0, 0.0, 0.0), device=env.device, dtype=torch.float32
    )
    local_forward = local_forward.view(1, 1, 3).expand(num_envs, num_feet, 3)
    foot_forward_w = quat_apply(foot_quat_w, local_forward)
    foot_forward_xy = foot_forward_w[..., :2]
    foot_forward_xy = foot_forward_xy / torch.clamp(
      torch.norm(foot_forward_xy, dim=-1, keepdim=True), min=1.0e-6
    )

    foot_forward_vel = torch.sum(foot_vel_w[..., :2] * foot_forward_xy, dim=-1)
    is_forward_sweep = foot_forward_vel[:, :, None] > forward_velocity_threshold
    is_vertical_face = torch.abs(normal_w[..., 2]) < vertical_normal_z_max
    force_dot_forward = torch.sum(
      force_w[..., :2] * foot_forward_xy[:, :, None, :], dim=-1
    )
    blocking_force = torch.relu(-force_dot_forward)
    hit_strength = torch.clamp(
      (blocking_force - force_threshold) / force_scale,
      min=0.0,
      max=1.0,
    )

    toe_riser_hit = (
      found
      & is_toe_contact
      & is_vertical_face
      & is_forward_sweep
      & (hit_strength > 0.0)
    )
    hit_by_foot = torch.any(toe_riser_hit, dim=-1)
    current_hit_by_foot = hit_by_foot & active_gate[:, None]
    new_hit_by_foot = current_hit_by_foot & (self._probe_hit_cooldown <= 0.0)

    if bool(torch.any(new_hit_by_foot).item()):
      self._probe_hit_cooldown = torch.where(
        new_hit_by_foot,
        torch.full_like(self._probe_hit_cooldown, probe_cooldown_time),
        self._probe_hit_cooldown,
      )

    current_hit_env = torch.any(current_hit_by_foot, dim=-1)
    new_hit_env = torch.any(new_hit_by_foot, dim=-1)
    contact_layers = self._contact_probe_layers(
      contact_pos_w,
      boundaries,
      probe_boundary_layers,
    )
    active_toe_hit = toe_riser_hit & active_gate[:, None, None]
    probe_layer_contact = active_toe_hit & (contact_layers > 0)
    non_probe_contact = active_toe_hit & ~probe_layer_contact
    non_probe_hit_by_foot = torch.any(non_probe_contact, dim=-1)
    non_probe_hit_env = torch.any(non_probe_hit_by_foot, dim=-1)

    per_foot_non_probe_strength = torch.max(
      torch.where(non_probe_contact, hit_strength, torch.zeros_like(hit_strength)),
      dim=-1,
    ).values
    current_strength = torch.max(
      torch.where(
        non_probe_hit_by_foot,
        per_foot_non_probe_strength,
        torch.zeros_like(per_foot_non_probe_strength),
      ),
      dim=-1,
    ).values

    if max_probe_layers > 0:
      layer_ids = torch.arange(
        1,
        max_probe_layers + 1,
        device=env.device,
        dtype=torch.long,
      )
      current_probe_layers = torch.any(
        probe_layer_contact[..., None] & (contact_layers[..., None] == layer_ids),
        dim=(1, 2),
      )
      tracked_probe_layers = self._probe_layer_hit[:, :max_probe_layers]
      new_probe_layers = current_probe_layers & ~tracked_probe_layers
      if bool(torch.any(current_probe_layers).item()):
        self._probe_layer_hit[:, :max_probe_layers] = (
          tracked_probe_layers | current_probe_layers
        )
      probe_count_by_env = (
        self._probe_layer_hit[:, :max_probe_layers].float().sum(dim=-1)
      )
    else:
      new_probe_layers = torch.zeros(
        (num_envs, 0),
        device=env.device,
        dtype=torch.bool,
      )
      probe_count_by_env = torch.zeros(
        num_envs,
        device=env.device,
        dtype=torch.float32,
      )

    contact_time = data.current_contact_time
    if contact_time is None:
      contact_time = current_hit_by_foot.float() * env.step_dt
    elif contact_time.shape[1] != num_feet:
      if contact_time.shape[1] % num_feet != 0:
        raise RuntimeError(
          f"Contact sensor '{sensor_name}' contact times cannot be grouped by foot: "
          f"{contact_time.shape[1]} entries for {num_feet} feet"
        )
      contact_time = contact_time.view(num_envs, num_feet, -1).max(dim=-1).values

    env_contact_time = torch.max(
      torch.where(current_hit_by_foot, contact_time, torch.zeros_like(contact_time)),
      dim=-1,
    ).values
    penalty_contact_time = torch.max(
      torch.where(non_probe_hit_by_foot, contact_time, torch.zeros_like(contact_time)),
      dim=-1,
    ).values
    time_scale = max(contact_time_scale, 1.0e-6)
    contact_time_weight = torch.clamp(
      penalty_contact_time / time_scale,
      min=0.0,
      max=1.0,
    )

    contact_penalty = (
      non_probe_hit_env.float()
      * current_strength
      * (1.0 + contact_time_weight)
      * contact_penalty_scale
    )
    probe_reward = new_probe_layers.float().sum(dim=-1) * probe_contact_reward

    env.extras["log"]["Metrics/toe_riser_slab_true_contact_ratio"] = (
      current_hit_env.float().mean()
    )
    env.extras["log"]["Metrics/toe_riser_slab_new_contact_ratio"] = (
      new_hit_env.float().mean()
    )
    env.extras["log"]["Metrics/toe_riser_slab_contact_penalty_mean"] = (
      contact_penalty.mean()
    )
    env.extras["log"]["Metrics/toe_riser_slab_probe_reward_mean"] = probe_reward.mean()
    env.extras["log"]["Metrics/toe_riser_slab_probe_count_mean"] = (
      probe_count_by_env.mean()
    )
    if max_probe_layers >= 1:
      env.extras["log"]["Metrics/toe_riser_slab_probe_layer1_hit_ratio"] = (
        self._probe_layer_hit[:, 0].float().mean()
      )
    else:
      env.extras["log"]["Metrics/toe_riser_slab_probe_layer1_hit_ratio"] = torch.zeros(
        (), device=env.device
      )
    if max_probe_layers >= 2:
      env.extras["log"]["Metrics/toe_riser_slab_probe_layer2_hit_ratio"] = (
        self._probe_layer_hit[:, 1].float().mean()
      )
    else:
      env.extras["log"]["Metrics/toe_riser_slab_probe_layer2_hit_ratio"] = torch.zeros(
        (), device=env.device
      )
    env.extras["log"]["Metrics/toe_riser_slab_contact_time_mean"] = (
      env_contact_time.mean()
    )
    return contact_penalty, probe_reward

  def _second_layer_attraction_reward(
    self,
    env: ManagerBasedRlEnv,
    toe_points: torch.Tensor,
    boundaries: torch.Tensor,
    probe_boundary_layers: torch.Tensor,
    active_gate: torch.Tensor,
    reward_scale: float,
    attraction_distance: float,
    u_margin: float,
    v_margin: float,
    surface_tol: float,
  ) -> torch.Tensor:
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    zero = torch.zeros((), device=env.device)
    log = env.extras["log"]
    if reward_scale <= 0.0 or self._probe_layer_hit.shape[1] < 2:
      log["Metrics/toe_riser_slab_second_layer_attraction_mean"] = zero
      log["Metrics/toe_riser_slab_second_layer_attraction_phase"] = zero
      log["Metrics/toe_riser_slab_second_layer_progress_mean"] = zero
      return reward

    first_layer_hit = self._probe_layer_hit[:, 0]
    second_layer_hit = self._probe_layer_hit[:, 1]
    attraction_phase = active_gate & first_layer_hit & ~second_layer_hit
    if not bool(torch.any(attraction_phase).item()):
      self._second_layer_attraction_progress[:] = 0.0
      log["Metrics/toe_riser_slab_second_layer_attraction_mean"] = zero
      log["Metrics/toe_riser_slab_second_layer_attraction_phase"] = zero
      log["Metrics/toe_riser_slab_second_layer_progress_mean"] = zero
      return reward

    second_layer_valid = probe_boundary_layers == 2
    if not bool(torch.any(second_layer_valid).item()):
      self._second_layer_attraction_progress = torch.where(
        attraction_phase,
        self._second_layer_attraction_progress,
        torch.zeros_like(self._second_layer_attraction_progress),
      )
      log["Metrics/toe_riser_slab_second_layer_attraction_mean"] = zero
      log["Metrics/toe_riser_slab_second_layer_attraction_phase"] = (
        attraction_phase.float().mean()
      )
      log["Metrics/toe_riser_slab_second_layer_progress_mean"] = zero
      return reward

    num_envs, num_feet = toe_points.shape[:2]
    expanded_boundaries = boundaries[:, None, :, :].expand(num_envs, num_feet, -1, -1)
    expanded_valid = second_layer_valid[:, None, :].expand(num_envs, num_feet, -1)

    p0 = expanded_boundaries[..., 0:3]
    p1 = expanded_boundaries[..., 3:6]
    normal_to_low = expanded_boundaries[..., 6:9]
    z_low = expanded_boundaries[..., 9]
    z_high = expanded_boundaries[..., 10]

    tangent_u = p1 - p0
    edge_len = torch.norm(tangent_u, dim=-1).clamp_min(1.0e-12)
    tangent_u = tangent_u / edge_len[..., None]
    center = 0.5 * (p0 + p1)
    center = center.clone()
    center[..., 2] = 0.5 * (z_low + z_high)
    half_u = 0.5 * edge_len
    half_v = 0.5 * torch.abs(z_high - z_low)

    rel = toe_points[:, :, :, None, :] - center[:, :, None, :, :]
    s = torch.sum(rel * normal_to_low[:, :, None, :, :], dim=-1)
    u = torch.sum(rel * tangent_u[:, :, None, :, :], dim=-1)
    v = rel[..., 2]

    attraction_distance = max(attraction_distance, 1.0e-6)
    face_band = (torch.abs(u) <= half_u[:, :, None, :] + u_margin) & (
      torch.abs(v) <= half_v[:, :, None, :] + v_margin
    )
    low_side_band = (s >= -surface_tol) & (s <= attraction_distance)
    distance_score = torch.clamp(
      (attraction_distance - torch.relu(s)) / attraction_distance,
      min=0.0,
      max=1.0,
    )
    score = torch.where(
      expanded_valid[:, :, None, :] & face_band & low_side_band,
      distance_score,
      torch.zeros_like(distance_score),
    )
    env_progress = torch.max(score.reshape(num_envs, -1), dim=-1).values
    env_progress = torch.where(
      attraction_phase,
      env_progress,
      torch.zeros_like(env_progress),
    )
    progress_delta = torch.relu(env_progress - self._second_layer_attraction_progress)
    self._second_layer_attraction_progress = torch.where(
      attraction_phase,
      torch.maximum(self._second_layer_attraction_progress, env_progress),
      torch.zeros_like(self._second_layer_attraction_progress),
    )
    reward = progress_delta * reward_scale
    log["Metrics/toe_riser_slab_second_layer_attraction_mean"] = reward.mean()
    log["Metrics/toe_riser_slab_second_layer_attraction_phase"] = (
      attraction_phase.float().mean()
    )
    log["Metrics/toe_riser_slab_second_layer_progress_mean"] = env_progress.mean()
    return reward

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    slab_depth: float = 0.04,
    u_margin: float = 0.02,
    v_margin: float = 0.02,
    toe_x_min: float = 0.09,
    toe_v_threshold: float = 0.05,
    surface_tol: float = 0.005,
    nearest_boundaries: int | None = None,
    log_only: bool = False,
    min_terrain_level: int | None = None,
    contact_sensor_name: str | None = None,
    contact_penalty_scale: float = 0.0,
    contact_time_scale: float = 0.20,
    contact_force_threshold: float = 15.0,
    contact_force_scale: float = 60.0,
    contact_vertical_normal_z_max: float = 0.4,
    contact_forward_velocity_threshold: float = 0.05,
    probe_contact_count: int = 0,
    probe_slab_reward_scale: float = 0.0,
    probe_contact_reward: float = 0.0,
    probe_min_progress: float = 0.08,
    probe_max_safe_force: float | None = None,
    probe_cooldown_time: float = 0.20,
    second_layer_attraction_reward: float = 0.0,
    second_layer_attraction_distance: float = 0.35,
    second_layer_attraction_u_margin: float = 0.04,
    second_layer_attraction_v_margin: float = 0.08,
    min_ascent_height: float = 0.03,
    ascent_velocity_threshold: float = 0.03,
    asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_BODY_CFG,
    **_: object,
  ) -> torch.Tensor:
    boundaries, valid_boundaries = _current_step_boundaries(env)
    if boundaries is None or valid_boundaries is None:
      return torch.zeros(env.num_envs, device=env.device)

    toe_mask = self._local_x >= toe_x_min
    if toe_mask.sum().item() == 0:
      return torch.zeros(env.num_envs, device=env.device)

    points_w, point_vel_w = self._foot_points_w(env, asset_cfg)
    toe_points = points_w[:, :, toe_mask, :]
    toe_vel = point_vel_w[:, :, toe_mask, :]
    num_envs, num_feet = toe_points.shape[:2]

    level_active = _terrain_level_active(env, min_terrain_level)
    base_valid = valid_boundaries & level_active[:, None]
    max_probe_layers = (
      self._ensure_probe_layer_capacity(probe_contact_count)
      if contact_sensor_name is not None
      else 0
    )
    probe_boundary_layers = self._probe_boundary_layers(
      boundaries,
      base_valid,
      max_probe_layers,
    )

    foot_ref_w = self._foot_ref_w(env, asset_cfg)
    ref_dist = self._riser_slab_ref_distance(
      foot_ref_w,
      boundaries,
      slab_depth,
      u_margin,
      v_margin,
      surface_tol,
    )
    toe_ref_radius = torch.norm(
      self._local_points[toe_mask] - self._foot_ref_local, dim=-1
    ).max()
    selected_idx, fallback = self._nearest_boundary_indices(
      ref_dist, base_valid, toe_ref_radius, nearest_boundaries
    )

    if selected_idx is None:
      expanded_boundaries = boundaries[:, None, :, :].expand(num_envs, num_feet, -1, -1)
      expanded_valid = base_valid[:, None, :].expand(num_envs, num_feet, -1)
      expanded_layers = probe_boundary_layers[:, None, :].expand(num_envs, num_feet, -1)
      (
        point_penalty,
        active,
        impact_speed_per_point,
        point_layers,
      ) = self._riser_slab_point_penalty(
        toe_points,
        toe_vel,
        expanded_boundaries,
        expanded_valid,
        slab_depth,
        u_margin,
        v_margin,
        toe_v_threshold,
        surface_tol,
        expanded_layers,
      )
      fallback_ratio = torch.zeros((), device=env.device)
    else:
      assert fallback is not None
      selected_boundaries = self._gather_by_foot(boundaries, selected_idx)
      selected_valid = self._gather_mask_by_foot(base_valid, selected_idx)
      selected_layers = self._gather_mask_by_foot(probe_boundary_layers, selected_idx)
      (
        point_penalty,
        active,
        impact_speed_per_point,
        point_layers,
      ) = self._riser_slab_point_penalty(
        toe_points,
        toe_vel,
        selected_boundaries,
        selected_valid,
        slab_depth,
        u_margin,
        v_margin,
        toe_v_threshold,
        surface_tol,
        selected_layers,
      )
      fallback_ratio = fallback.float().mean()
      if bool(torch.any(fallback).item()):
        expanded_boundaries = boundaries[:, None, :, :].expand(
          num_envs, num_feet, -1, -1
        )
        expanded_valid = base_valid[:, None, :].expand(num_envs, num_feet, -1)
        expanded_layers = probe_boundary_layers[:, None, :].expand(
          num_envs, num_feet, -1
        )
        full_penalty, full_active, full_impact, full_layers = (
          self._riser_slab_point_penalty(
            toe_points,
            toe_vel,
            expanded_boundaries,
            expanded_valid,
            slab_depth,
            u_margin,
            v_margin,
            toe_v_threshold,
            surface_tol,
            expanded_layers,
          )
        )
        fallback_mask = fallback[:, :, None]
        point_penalty = torch.where(fallback_mask, full_penalty, point_penalty)
        active = torch.where(fallback_mask, full_active, active)
        impact_speed_per_point = torch.where(
          fallback_mask, full_impact, impact_speed_per_point
        )
        point_layers = torch.where(fallback_mask, full_layers, point_layers)

    penalty = torch.sum(point_penalty, dim=(1, 2))
    protected_penalty = torch.zeros_like(penalty)
    effective_point_penalty = penalty
    contact_penalty = torch.zeros_like(penalty)
    probe_reward = torch.zeros_like(penalty)
    attraction_reward = torch.zeros_like(penalty)
    raw_penalty = effective_point_penalty

    active_count = active.float().sum().clamp_min(1.0)
    impact_speed_mean = torch.sum(impact_speed_per_point * active.float())
    impact_speed_mean = impact_speed_mean / active_count
    env.extras["log"]["Metrics/toe_riser_slab_penalty_mean"] = penalty.mean()
    env.extras["log"]["Metrics/toe_riser_slab_point_penalty_mean"] = penalty.mean()
    env.extras["log"]["Metrics/toe_riser_slab_active_ratio"] = active.float().mean()
    env.extras["log"]["Metrics/toe_riser_slab_impact_speed_mean"] = impact_speed_mean
    env.extras["log"]["Metrics/toe_riser_slab_nearest_fallback_ratio"] = fallback_ratio

    if log_only:
      return torch.zeros_like(penalty)

    if contact_sensor_name is not None:
      asset: Entity = env.scene[asset_cfg.name]
      active_gate = self._ascent_gate(
        env,
        asset,
        level_active,
        torch.any(active, dim=(1, 2)),
        min_ascent_height,
        ascent_velocity_threshold,
      )
      probe_phase = (
        active_gate if max_probe_layers > 0 else torch.zeros_like(active_gate)
      )
      protected_probe_points = active_gate[:, None, None] & (point_layers > 0)
      if bool(torch.any(protected_probe_points).item()):
        protected_penalty = torch.sum(
          torch.where(
            protected_probe_points,
            point_penalty,
            torch.zeros_like(point_penalty),
          ),
          dim=(1, 2),
        )
      effective_point_penalty = penalty - protected_penalty
      contact_penalty, probe_reward = self._contact_probe_terms(
        env,
        contact_sensor_name,
        asset,
        asset_cfg,
        active_gate,
        boundaries,
        probe_boundary_layers,
        max_probe_layers,
        toe_x_min,
        contact_vertical_normal_z_max,
        contact_forward_velocity_threshold,
        contact_force_threshold,
        contact_force_scale,
        contact_penalty_scale,
        contact_time_scale,
        probe_contact_reward,
        probe_cooldown_time,
      )
      attraction_reward = self._second_layer_attraction_reward(
        env,
        toe_points,
        boundaries,
        probe_boundary_layers,
        active_gate,
        second_layer_attraction_reward,
        second_layer_attraction_distance,
        second_layer_attraction_u_margin,
        second_layer_attraction_v_margin,
        surface_tol,
      )
      raw_penalty = (
        effective_point_penalty + contact_penalty - probe_reward - attraction_reward
      )
      env.extras["log"]["Metrics/toe_riser_slab_probe_active_ratio"] = (
        probe_phase.float().mean()
      )
      env.extras["log"]["Metrics/toe_riser_slab_probe_slab_neutral_ratio"] = (
        protected_probe_points.float().mean()
      )
    else:
      zero = torch.zeros((), device=env.device)
      env.extras["log"]["Metrics/toe_riser_slab_contact_penalty_mean"] = zero
      env.extras["log"]["Metrics/toe_riser_slab_probe_reward_mean"] = zero
      env.extras["log"]["Metrics/toe_riser_slab_probe_count_mean"] = zero
      env.extras["log"]["Metrics/toe_riser_slab_probe_layer1_hit_ratio"] = zero
      env.extras["log"]["Metrics/toe_riser_slab_probe_layer2_hit_ratio"] = zero
      env.extras["log"]["Metrics/toe_riser_slab_contact_time_mean"] = zero
      env.extras["log"]["Metrics/toe_riser_slab_second_layer_attraction_mean"] = zero
      env.extras["log"]["Metrics/toe_riser_slab_second_layer_attraction_phase"] = zero
      env.extras["log"]["Metrics/toe_riser_slab_second_layer_progress_mean"] = zero
      env.extras["log"]["Metrics/toe_riser_slab_probe_active_ratio"] = zero
      env.extras["log"]["Metrics/toe_riser_slab_probe_slab_neutral_ratio"] = zero

    env.extras["log"]["Metrics/toe_riser_slab_protected_point_penalty_mean"] = (
      protected_penalty.mean()
    )
    env.extras["log"]["Metrics/toe_riser_slab_effective_point_penalty_mean"] = (
      effective_point_penalty.mean()
    )
    env.extras["log"]["Metrics/toe_riser_slab_contact_penalty_mean"] = (
      contact_penalty.mean()
    )
    env.extras["log"]["Metrics/toe_riser_slab_probe_reward_mean"] = probe_reward.mean()
    env.extras["log"]["Metrics/toe_riser_slab_second_layer_attraction_mean"] = (
      attraction_reward.mean()
    )
    env.extras["log"]["Metrics/toe_riser_slab_raw_mean"] = raw_penalty.mean()

    return raw_penalty


class toe_step_riser_approach_penalty(toe_step_riser_slab_penalty):
  """Backward-compatible alias for the toe riser slab penalty."""


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  return torch.exp(-ang_vel_error / std**2)


class upright:
  """Reward for keeping the base upright.

  Without ``terrain_sensor_names``, penalizes tilt relative to world up (correct for
  flat ground).

  With ``terrain_sensor_names``, penalizes tilt relative to the terrain surface normal.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._terrain_sensor_names: tuple[str, ...] | None = cfg.params.get(
      "terrain_sensor_names"
    )
    self._debug_vis_enabled = True
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    terrain_sensor_names: tuple[str, ...] | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    if asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
      body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    else:
      body_quat_w = asset.data.root_link_quat_w  # [B, 4]

    if terrain_sensor_names is not None:
      terrain_normal = terrain_normal_from_sensors(env, terrain_sensor_names)  # [B, 3]
      # Project terrain normal into body frame. When aligned with the terrain surface
      # this should be (0, 0, 1); XY measures tilt.
      target_b = quat_apply_inverse(body_quat_w, terrain_normal)  # [B, 3]
      xy_squared = torch.sum(torch.square(target_b[:, :2]), dim=1)
    else:
      gravity_w = asset.data.gravity_vec_w  # [3]
      projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
      xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / std**2)

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids  # Unused.

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled or self._terrain_sensor_names is None:
      return

    env = self._env
    asset: Entity = env.scene[self._asset_cfg.name]

    env_indices = list(visualizer.get_env_indices(env.num_envs))
    if not env_indices:
      return

    terrain_normal = terrain_normal_from_sensors(env, self._terrain_sensor_names)
    if self._asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, self._asset_cfg.body_ids, :].squeeze(
        1
      )
    else:
      body_quat_w = asset.data.root_link_quat_w
    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
      body_quat_w[:, :3]
    )
    body_up_w = quat_apply(body_quat_w, up_local)

    positions = asset.data.root_link_pos_w.cpu().numpy()
    offset = np.array([0.0, 0.3, 0.0])
    terrain_normal_np = terrain_normal.cpu().numpy()
    body_up_np = body_up_w.cpu().numpy()
    scale = 0.25

    for i in env_indices:
      origin = positions[i] + offset
      # Terrain normal (magenta).
      visualizer.add_arrow(
        start=origin,
        end=origin + terrain_normal_np[i] * scale,
        color=(0.8, 0.2, 0.8, 0.8),
        width=0.01,
      )
      # Body up (orange).
      visualizer.add_arrow(
        start=origin,
        end=origin + body_up_np[i] * scale,
        color=(1.0, 0.5, 0.0, 0.8),
        width=0.01,
      )


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.sum(dim=-1).float()


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  height_sensor_name: str,
  target_height: float | None = None,
  min_height: float | None = None,
  max_height: float | None = None,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot clearance outside a target height or height range."""
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    f"feet_clearance requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
  )
  foot_height = height_sensor.data.heights  # [B, F]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, F]
  if min_height is not None or max_height is not None:
    if min_height is None or max_height is None:
      raise ValueError("feet_clearance requires both min_height and max_height.")
    if min_height > max_height:
      raise ValueError("feet_clearance min_height must be <= max_height.")
    min_height_tensor = foot_height.new_tensor(min_height)
    max_height_tensor = foot_height.new_tensor(max_height)
    delta = torch.relu(min_height_tensor - foot_height) + torch.relu(
      foot_height - max_height_tensor
    )
  else:
    if target_height is None:
      raise ValueError("feet_clearance requires target_height or min/max height.")
    delta = torch.abs(foot_height - target_height)  # [B, F]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class feet_swing_height:
  """Penalize swing peaks below the target height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    height_sensor = env.scene[cfg.params["height_sensor_name"]]
    assert isinstance(height_sensor, TerrainHeightSensor), (
      f"feet_swing_height requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
    )
    num_feet = height_sensor.num_frames
    self.peak_heights = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    foot_heights = height_sensor.data.heights
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    shortfall = torch.relu(1.0 - self.peak_heights / target_height)
    cost = torch.sum(torch.square(shortfall) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class toe_riser_contact_memory_penalty:
  """Penalize repeated toe impacts against vertical riser-like terrain contacts.

  The detector uses only contact data and proprioceptive kinematics. It does not
  read step-boundary geometry, height scans, or terrain type names.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_FOOT_BODY_CFG)
    num_feet = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else 2
    self._toe_hit_count = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
    self._toe_hit_cooldown = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float32
    )
    self._root_z_baseline = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
    self._max_root_z = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    self._ascent_active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self._needs_root_z_init = torch.ones(
      env.num_envs, device=env.device, dtype=torch.bool
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._toe_hit_count[env_ids] = 0.0
    self._toe_hit_cooldown[env_ids] = 0.0
    self._root_z_baseline[env_ids] = 0.0
    self._max_root_z[env_ids] = 0.0
    self._ascent_active[env_ids] = False
    self._needs_root_z_init[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    free_hits: int = 1,
    cooldown_time: float = 0.20,
    min_terrain_level: int = 3,
    min_ascent_height: float = 0.03,
    ascent_velocity_threshold: float = 0.03,
    toe_x_min: float = 0.08,
    vertical_normal_z_max: float = 0.4,
    forward_velocity_threshold: float = 0.05,
    force_threshold: float = 15.0,
    force_scale: float = 60.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_BODY_CFG,
  ) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    assert isinstance(sensor, ContactSensor), (
      f"toe_riser_contact_memory_penalty requires a ContactSensor, got "
      f"{type(sensor).__name__}"
    )
    data = sensor.data
    assert data.found is not None
    assert data.force is not None
    assert data.normal is not None
    assert data.pos is not None

    asset: Entity = env.scene[asset_cfg.name]
    root_z = asset.data.root_link_pos_w[:, 2]
    needs_init = self._needs_root_z_init
    self._root_z_baseline = torch.where(needs_init, root_z, self._root_z_baseline)
    self._max_root_z = torch.where(needs_init, root_z, self._max_root_z)
    self._needs_root_z_init[:] = False

    self._max_root_z = torch.maximum(self._max_root_z, root_z)
    root_z_gain = self._max_root_z - self._root_z_baseline
    root_vz = asset.data.root_link_lin_vel_w[:, 2]
    ascent_observed = (root_z_gain >= min_ascent_height) | (
      root_vz > ascent_velocity_threshold
    )

    terrain_level_active = _terrain_level_active(env, min_terrain_level)
    self._ascent_active |= terrain_level_active & ascent_observed
    active_gate = terrain_level_active & self._ascent_active

    inactive = ~active_gate
    if bool(torch.any(inactive).item()):
      self._toe_hit_count[inactive] = 0.0
      self._toe_hit_cooldown[inactive] = 0.0

    self._toe_hit_cooldown = torch.clamp(self._toe_hit_cooldown - env.step_dt, min=0.0)

    num_envs = env.num_envs
    num_feet = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else 2
    num_contacts = data.found.shape[1]
    assert num_contacts % num_feet == 0, (
      f"Contact sensor '{sensor_name}' has {num_contacts} contact slots, which is "
      f"not divisible by {num_feet} feet"
    )
    num_slots = num_contacts // num_feet

    found = data.found.view(num_envs, num_feet, num_slots) > 0
    force_w = data.force.view(num_envs, num_feet, num_slots, 3)
    normal_w = data.normal.view(num_envs, num_feet, num_slots, 3)
    contact_pos_w = data.pos.view(num_envs, num_feet, num_slots, 3)

    foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    foot_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    foot_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]

    expanded_quat = foot_quat_w[:, :, None, :].expand(num_envs, num_feet, num_slots, 4)
    contact_pos_b = quat_apply_inverse(
      expanded_quat,
      contact_pos_w - foot_pos_w[:, :, None, :],
    )
    is_toe_contact = contact_pos_b[..., 0] >= toe_x_min

    local_forward = torch.tensor(
      (1.0, 0.0, 0.0), device=env.device, dtype=torch.float32
    )
    local_forward = local_forward.view(1, 1, 3).expand(num_envs, num_feet, 3)
    foot_forward_w = quat_apply(foot_quat_w, local_forward)
    foot_forward_xy = foot_forward_w[..., :2]
    foot_forward_xy = foot_forward_xy / torch.clamp(
      torch.norm(foot_forward_xy, dim=-1, keepdim=True), min=1e-6
    )

    foot_forward_vel = torch.sum(foot_vel_w[..., :2] * foot_forward_xy, dim=-1)
    is_forward_sweep = foot_forward_vel[:, :, None] > forward_velocity_threshold

    is_vertical_face = torch.abs(normal_w[..., 2]) < vertical_normal_z_max
    force_dot_forward = torch.sum(
      force_w[..., :2] * foot_forward_xy[:, :, None, :], dim=-1
    )
    blocking_force = torch.relu(-force_dot_forward)
    hit_strength = torch.clamp(
      (blocking_force - force_threshold) / force_scale,
      min=0.0,
      max=1.0,
    )

    toe_riser_hit = (
      found
      & is_toe_contact
      & is_vertical_face
      & is_forward_sweep
      & (hit_strength > 0.0)
    )
    hit_by_foot = torch.any(toe_riser_hit, dim=-1)
    new_hit_by_foot = (
      hit_by_foot & active_gate[:, None] & (self._toe_hit_cooldown <= 0.0)
    )

    if bool(torch.any(new_hit_by_foot).item()):
      self._toe_hit_cooldown = torch.where(
        new_hit_by_foot,
        torch.full_like(self._toe_hit_cooldown, cooldown_time),
        self._toe_hit_cooldown,
      )

    new_hit_env = torch.any(new_hit_by_foot, dim=-1)
    self._toe_hit_count += new_hit_env.float()

    per_foot_strength = torch.max(
      torch.where(toe_riser_hit, hit_strength, torch.zeros_like(hit_strength)),
      dim=-1,
    ).values
    env_hit_strength = torch.max(
      torch.where(
        new_hit_by_foot,
        per_foot_strength,
        torch.zeros_like(per_foot_strength),
      ),
      dim=-1,
    ).values
    penalize = new_hit_env & (self._toe_hit_count > float(free_hits))
    penalty = penalize.float() * env_hit_strength

    env.extras["log"]["Metrics/toe_riser_contact_active_ratio"] = (
      active_gate.float().mean()
    )
    env.extras["log"]["Metrics/toe_riser_contact_new_hit_ratio"] = (
      new_hit_env.float().mean()
    )
    env.extras["log"]["Metrics/toe_riser_contact_penalty_mean"] = penalty.mean()
    env.extras["log"]["Metrics/toe_riser_contact_hit_count_mean"] = (
      self._toe_hit_count.mean()
    )
    return penalty


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def idle_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.2,
  velocity_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize standing nearly still when a clear velocity command is given."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  commanded_speed = torch.norm(command[:, :2], dim=1)
  actual_speed = torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)

  penalty = (
    (commanded_speed > command_threshold) & (actual_speed < velocity_threshold)
  ).float()

  env.extras["log"]["Metrics/idle_penalty_ratio"] = torch.mean(penalty)
  return penalty


def feet_gait(
  env: ManagerBasedRlEnv,
  period: float,
  offset: list[float],
  threshold: float,
  command_threshold: float,
  command_name: str,
  sensor_name: str,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  current_contact_time = sensor.data.current_contact_time
  assert current_contact_time is not None, (
    "Enable track_air_time=True for this contact sensor."
  )
  is_contact = current_contact_time > 0
  global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
  offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(
    1, -1
  )
  leg_phase = (global_phase + offsets) % 1.0
  is_stance = leg_phase < threshold
  reward = (is_stance == is_contact).float().mean(dim=1)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def target_progress(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_distance: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward world-frame velocity projected toward the active target point."""
  command_term = env.command_manager.get_term(command_name)
  required_attrs = ("target_pos_w", "has_target", "is_target_env")
  if not all(hasattr(command_term, name) for name in required_attrs):
    return torch.zeros(env.num_envs, device=env.device)
  command_term = cast("TargetHeadingVelocityCommand", command_term)

  asset: Entity = env.scene[asset_cfg.name]
  target_vec_xy = command_term.target_pos_w[:, :2] - asset.data.root_link_pos_w[:, :2]
  target_dist = torch.norm(target_vec_xy, dim=-1)
  target_dir_xy = target_vec_xy / torch.clamp(
    target_dist.unsqueeze(-1), min=min_distance
  )
  lin_vel_w_xy = asset.data.root_link_lin_vel_w[:, :2]
  progress_speed = torch.sum(lin_vel_w_xy * target_dir_xy, dim=-1)
  active = command_term.has_target & command_term.is_target_env

  return torch.clamp(progress_speed, min=0.0) * active.float()


def target_reached_bonus(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Sparse bonus emitted after the command detects target arrival."""
  command_term = env.command_manager.get_term(command_name)
  if not hasattr(command_term, "target_reached_this_step"):
    return torch.zeros(env.num_envs, device=env.device)
  command_term = cast("TargetHeadingVelocityCommand", command_term)
  return command_term.target_reached_this_step.float()


def base_height_above_support_value(
  env: ManagerBasedRlEnv,
  height_sensor_name: str,
  contact_sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_SITE_CFG,
) -> torch.Tensor:
  """Base height relative to the terrain under the support foot/feet."""
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  contact_sensor = env.scene[contact_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    "base_height_above_support_value requires a TerrainHeightSensor, got "
    f"{type(height_sensor).__name__}"
  )
  assert isinstance(contact_sensor, ContactSensor), (
    "base_height_above_support_value requires a ContactSensor, got "
    f"{type(contact_sensor).__name__}"
  )
  assert contact_sensor.data.found is not None

  base_z = asset.data.root_link_pos_w[:, 2]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  terrain_z_under_feet = foot_z - height_sensor.data.heights

  contact = (contact_sensor.data.found > 0).float()
  contact_sum = contact.sum(dim=1).clamp_min(1.0)
  support_terrain_z = (terrain_z_under_feet * contact).sum(dim=1) / contact_sum
  fallback_terrain_z = terrain_z_under_feet.max(dim=1).values
  has_contact = contact.sum(dim=1) > 0
  terrain_z = torch.where(has_contact, support_terrain_z, fallback_terrain_z)

  return base_z - terrain_z


def base_height_above_support(
  env,
  height_sensor_name: str,
  contact_sensor_name: str,
  min_height: float = 0.74,
  error_scale: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_SITE_CFG,
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  contact_sensor = env.scene[contact_sensor_name]

  base_z = asset.data.root_link_pos_w[:, 2]

  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  foot_height_above_terrain = height_sensor.data.heights
  terrain_z_under_feet = foot_z - foot_height_above_terrain

  contact = (contact_sensor.data.found > 0).float()
  contact_sum = contact.sum(dim=1).clamp_min(1.0)

  support_terrain_z = (terrain_z_under_feet * contact).sum(dim=1) / contact_sum
  fallback_terrain_z = terrain_z_under_feet.max(dim=1).values
  has_contact = contact.sum(dim=1) > 0

  terrain_z = torch.where(has_contact, support_terrain_z, fallback_terrain_z)
  base_height_rel = base_z - terrain_z

  height_error = torch.relu(min_height - base_height_rel) * error_scale
  return torch.square(height_error)
