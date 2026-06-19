from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from .rewards import (
  _DEFAULT_FOOT_BODY_CFG,
  _DEFAULT_FOOT_SITE_CFG,
  _current_step_boundaries,
  _step_boundary_layers,
  _StepBoundaryFootVolume,
  _terrain_level_active,
)
from .rewards import (
  toe_step_riser_slab_penalty as _LegacyToeStepRiserSlabPenalty,
)
from .stair_geometry import (
  PROBE_STAGE_KEY,
  PROBE_TARGET_FOOT_KEY,
  cached_stair_shape,
)


class toe_step_riser_slab_penalty(_LegacyToeStepRiserSlabPenalty):
  """Temporal, foot-specific toe-riser probing reward.

  This overrides the legacy layer-based probing implementation while keeping the
  public reward-term name stable for existing configs.
  """

  def __init__(self, cfg: RewardTermCfg, env):
    super(_LegacyToeStepRiserSlabPenalty, self).__init__(cfg, env)
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

    self._probe_phase = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._probe_first_foot = torch.full(
      (env.num_envs,), -1, device=env.device, dtype=torch.long
    )
    self._probe_target_foot = torch.full(
      (env.num_envs,), -1, device=env.device, dtype=torch.long
    )
    self._probe_timer = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._probe_target_start_pos_w = torch.zeros(
      env.num_envs, 3, device=env.device, dtype=torch.float32
    )
    self._probe_ascent_dir = torch.zeros(
      env.num_envs, 2, device=env.device, dtype=torch.float32
    )
    self._probe_target_boundary_idx = torch.full(
      (env.num_envs,), -1, device=env.device, dtype=torch.long
    )
    self._probe_first_toe_z_world = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
    self._probe_success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self._probe_contact_count = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )
    self._probe_target_lift_progress = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
    self._probe_target_forward_progress = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
    self._probe_second_confirmed = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    env.extras[PROBE_STAGE_KEY] = self._probe_phase
    env.extras[PROBE_TARGET_FOOT_KEY] = self._probe_target_foot

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
    self._probe_phase[env_ids] = 0
    self._probe_first_foot[env_ids] = -1
    self._probe_target_foot[env_ids] = -1
    self._probe_timer[env_ids] = 0
    self._probe_target_start_pos_w[env_ids] = 0.0
    self._probe_ascent_dir[env_ids] = 0.0
    self._probe_target_boundary_idx[env_ids] = -1
    self._probe_first_toe_z_world[env_ids] = 0.0
    self._probe_success[env_ids] = False
    self._probe_contact_count[env_ids] = 0
    self._probe_target_lift_progress[env_ids] = 0.0
    self._probe_target_forward_progress[env_ids] = 0.0
    self._probe_second_confirmed[env_ids] = False

  @staticmethod
  def _normalize_xy(vec: torch.Tensor) -> torch.Tensor:
    return vec / torch.norm(vec, dim=-1, keepdim=True).clamp_min(1.0e-6)

  @staticmethod
  def _ascent_progress(
    foot_pos_w: torch.Tensor,
    start_pos_w: torch.Tensor,
    ascent_dir: torch.Tensor,
  ) -> torch.Tensor:
    return torch.sum((foot_pos_w[:, :2] - start_pos_w[:, :2]) * ascent_dir, dim=-1)

  @staticmethod
  def _velocity_guard(
    foot_vel_w: torch.Tensor,
    ascent_dir: torch.Tensor,
    max_forward_vel: float,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forward_vel = torch.sum(foot_vel_w[:, :2] * ascent_dir, dim=-1)
    overspeed = torch.relu(forward_vel - max_forward_vel)
    return forward_vel, forward_vel <= max_forward_vel, overspeed

  @staticmethod
  def _shallow_layer_points(
    toe_points: torch.Tensor,
    boundaries: torch.Tensor,
    valid_boundaries: torch.Tensor,
    boundary_layers: torch.Tensor,
    shallow_depth: float,
    u_margin: float,
    v_margin: float,
    surface_tol: float,
  ) -> torch.Tensor:
    """Return toe points lying close to a layer-2 riser face."""
    p0 = boundaries[:, None, None, :, 0:3]
    p1 = boundaries[:, None, None, :, 3:6]
    normal_to_low = boundaries[:, None, None, :, 6:9]
    z_low = boundaries[:, None, None, :, 9]
    z_high = boundaries[:, None, None, :, 10]

    tangent_u = p1 - p0
    edge_len = torch.norm(tangent_u, dim=-1).clamp_min(1.0e-12)
    tangent_u = tangent_u / edge_len[..., None]
    center = 0.5 * (p0 + p1)
    center[..., 2] = 0.5 * (z_low + z_high)
    rel = toe_points[:, :, :, None, :] - center
    s = torch.sum(rel * normal_to_low, dim=-1)
    u = torch.sum(rel * tangent_u, dim=-1)
    v = rel[..., 2]
    inside_face = (torch.abs(u) <= 0.5 * edge_len + u_margin) & (
      torch.abs(v) <= 0.5 * (z_high - z_low) + v_margin
    )
    shallow = (s >= -surface_tol) & (s <= shallow_depth)
    layer2 = boundary_layers[:, None, None, :] == 2
    valid = valid_boundaries[:, None, None, :] & layer2
    return torch.any(valid & inside_face & shallow, dim=-1)

  def __call__(
    self,
    env,
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
    temporal_probe_first_reward: float = 0.05,
    temporal_probe_confirm_reward: float = 0.20,
    temporal_probe_lift_reward: float = 0.15,
    temporal_probe_forward_reward: float = 0.15,
    temporal_probe_min_lift: float = 0.03,
    temporal_probe_lift_scale: float = 0.08,
    temporal_probe_min_forward: float = 0.08,
    temporal_probe_forward_scale: float = 0.20,
    temporal_probe_timeout: float = 0.80,
    temporal_probe_min_forward_vel: float = 0.03,
    temporal_probe_max_forward_vel: float = 0.45,
    temporal_probe_boundary_normal_cos: float = 0.90,
    temporal_probe_boundary_distance_tolerance: float = 0.12,
    temporal_probe_shallow_depth: float = 0.025,
    temporal_probe_overspeed_penalty: float = 0.10,
    asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_BODY_CFG,
    **_: object,
  ) -> torch.Tensor:
    del (
      probe_slab_reward_scale,
      probe_min_progress,
      probe_max_safe_force,
      second_layer_attraction_reward,
      second_layer_attraction_distance,
      second_layer_attraction_u_margin,
      second_layer_attraction_v_margin,
    )

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
      point_penalty, active, impact_speed_per_point, point_layers = (
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
      fallback_ratio = torch.zeros((), device=env.device)
    else:
      assert fallback is not None
      selected_boundaries = self._gather_by_foot(boundaries, selected_idx)
      selected_valid = self._gather_mask_by_foot(base_valid, selected_idx)
      selected_layers = self._gather_mask_by_foot(probe_boundary_layers, selected_idx)
      point_penalty, active, impact_speed_per_point, point_layers = (
        self._riser_slab_point_penalty(
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
      probe_timeout_steps = max(1, int(temporal_probe_timeout / env.step_dt))

      ctd = self._contact_probe_terms(
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
        event_gate=level_active,
      )
      contact_penalty = ctd["contact_penalty"]
      new_hit_by_foot = ctd["new_hit_by_foot"]
      foot_pos_w = ctd["foot_pos_w"]
      foot_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]
      probe_layer_contact = ctd["probe_layer_contact"]
      contact_layers = ctd["contact_layers"]
      contact_boundary_idx = ctd["contact_boundary_idx"]
      hit_strength = ctd["hit_strength"]

      env_ids = torch.arange(env.num_envs, device=env.device)
      num_slots = contact_layers.shape[-1]
      first_contact = (
        probe_layer_contact & (contact_layers == 1) & new_hit_by_foot[:, :, None]
      )
      first_contact_score = torch.where(
        first_contact,
        hit_strength,
        torch.full_like(hit_strength, -torch.inf),
      )
      first_flat = torch.argmax(first_contact_score.reshape(num_envs, -1), dim=-1)
      first_foot = first_flat // num_slots
      first_slot = first_flat % num_slots
      first_contact_mask = (self._probe_phase == 0) & torch.any(
        first_contact, dim=(1, 2)
      )

      first_boundary_idx = contact_boundary_idx[env_ids, first_foot, first_slot]
      first_normal = self._normalize_xy(boundaries[env_ids, first_boundary_idx, 6:8])
      first_p0 = boundaries[env_ids, first_boundary_idx, 0:2]
      root_xy = asset.data.root_link_pos_w[:, :2]
      root_side = torch.sum((root_xy - first_p0) * first_normal, dim=-1)
      probe_side = torch.where(root_side >= 0.0, 1.0, -1.0)
      first_ascent_dir = -probe_side[:, None] * first_normal
      first_forward_vel = torch.sum(
        foot_vel_w[env_ids, first_foot, :2] * first_ascent_dir,
        dim=-1,
      )
      first_speed_safe = (first_forward_vel > 0.0) & (
        first_forward_vel <= temporal_probe_max_forward_vel
      )
      first_hit_mask = first_contact_mask & first_speed_safe

      first_probe_reward = torch.zeros(env.num_envs, device=env.device)
      if bool(torch.any(first_hit_mask).item()):
        self._probe_first_foot[first_hit_mask] = first_foot[first_hit_mask]
        self._probe_target_foot[first_hit_mask] = 1 - first_foot[first_hit_mask]
        self._probe_phase[first_hit_mask] = 1
        self._probe_timer[first_hit_mask] = 0
        self._probe_contact_count[first_hit_mask] = 1
        self._probe_target_lift_progress[first_hit_mask] = 0.0
        self._probe_target_forward_progress[first_hit_mask] = 0.0
        self._probe_second_confirmed[first_hit_mask] = False
        self._probe_ascent_dir[first_hit_mask] = first_ascent_dir[first_hit_mask]
        self._probe_target_boundary_idx[first_hit_mask] = first_boundary_idx[
          first_hit_mask
        ]
        target_at_entry = self._probe_target_foot[first_hit_mask]
        self._probe_target_start_pos_w[first_hit_mask] = foot_pos_w[
          first_hit_mask, target_at_entry
        ]
        self._probe_first_toe_z_world[first_hit_mask] = self._probe_target_start_pos_w[
          first_hit_mask, 2
        ]
        first_probe_reward[first_hit_mask] = temporal_probe_first_reward
        self._ascent_active[first_hit_mask] = True

      active_gate = level_active & (self._ascent_active | (self._probe_phase > 0))
      self._probe_timer = torch.where(
        self._probe_phase == 1, self._probe_timer + 1, self._probe_timer
      )
      timeout_mask = (self._probe_phase == 1) & (
        self._probe_timer > probe_timeout_steps
      )
      if bool(torch.any(timeout_mask).item()):
        self.reset(timeout_mask.nonzero(as_tuple=False).squeeze(-1))

      phase1_for_step = self._probe_phase == 1
      target_foot = self._probe_target_foot.clamp(0, max(1, num_feet - 1))
      target_pos_w = foot_pos_w[env_ids, target_foot]
      target_vel_w = foot_vel_w[env_ids, target_foot]
      target_forward_vel, target_speed_safe, target_overspeed = self._velocity_guard(
        target_vel_w,
        self._probe_ascent_dir,
        temporal_probe_max_forward_vel,
      )

      target_layer2_contact = probe_layer_contact[env_ids, target_foot] & (
        contact_layers[env_ids, target_foot] == 2
      )
      target_layer2_score = torch.where(
        target_layer2_contact,
        hit_strength[env_ids, target_foot],
        torch.full_like(hit_strength[env_ids, target_foot], -torch.inf),
      )
      target_layer2_slot = torch.argmax(target_layer2_score, dim=-1)
      target_boundary_idx = contact_boundary_idx[
        env_ids, target_foot, target_layer2_slot
      ].clamp(0, boundaries.shape[1] - 1)
      has_target_layer2_contact = torch.any(target_layer2_contact, dim=-1)

      first_boundary_idx = self._probe_target_boundary_idx.clamp(
        0, boundaries.shape[1] - 1
      )
      first_boundary_normal = self._normalize_xy(
        boundaries[env_ids, first_boundary_idx, 6:8]
      )
      target_boundary_normal = self._normalize_xy(
        boundaries[env_ids, target_boundary_idx, 6:8]
      )
      boundary_normal_cos = torch.sum(
        first_boundary_normal * target_boundary_normal, dim=-1
      )
      first_boundary_p0 = boundaries[env_ids, first_boundary_idx, 0:2]
      target_boundary_p0 = boundaries[env_ids, target_boundary_idx, 0:2]
      target_boundary_forward_distance = torch.sum(
        (target_boundary_p0 - first_boundary_p0) * self._probe_ascent_dir,
        dim=-1,
      )
      tread_depth, _riser_height, shape_valid = cached_stair_shape(
        env, boundaries, valid_boundaries
      )
      boundary_distance_gate = (~shape_valid) | (
        torch.abs(target_boundary_forward_distance - tread_depth)
        <= temporal_probe_boundary_distance_tolerance
      )
      target_direction_gate = target_forward_vel >= temporal_probe_min_forward_vel
      target_boundary_gate = (
        has_target_layer2_contact
        & (boundary_normal_cos >= temporal_probe_boundary_normal_cos)
        & (target_boundary_forward_distance > 0.0)
        & boundary_distance_gate
      )
      target_hit_mask = (
        phase1_for_step
        & active_gate
        & target_speed_safe
        & target_direction_gate
        & target_boundary_gate
        & new_hit_by_foot[env_ids, target_foot]
        & ~self._probe_second_confirmed
      )
      second_confirm_reward = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.float32
      )
      second_confirm_reward[target_hit_mask] = temporal_probe_confirm_reward

      target_lift_reward = torch.zeros_like(second_confirm_reward)
      target_forward_reward = torch.zeros_like(second_confirm_reward)
      target_overspeed_penalty = torch.where(
        phase1_for_step,
        target_overspeed * temporal_probe_overspeed_penalty,
        torch.zeros_like(target_overspeed),
      )
      if bool(torch.any(phase1_for_step).item()):
        target_z = target_pos_w[phase1_for_step, 2]
        lift_gain = target_z - self._probe_first_toe_z_world[phase1_for_step]
        lift_score = torch.clamp(
          (lift_gain - temporal_probe_min_lift)
          / max(temporal_probe_lift_scale, 1.0e-6),
          0.0,
          1.0,
        )
        lift_delta = torch.relu(
          lift_score - self._probe_target_lift_progress[phase1_for_step]
        )
        self._probe_target_lift_progress[phase1_for_step] = torch.maximum(
          self._probe_target_lift_progress[phase1_for_step], lift_score
        )
        target_lift_reward[phase1_for_step] = lift_delta * temporal_probe_lift_reward

        forward_gain = self._ascent_progress(
          target_pos_w[phase1_for_step],
          self._probe_target_start_pos_w[phase1_for_step],
          self._probe_ascent_dir[phase1_for_step],
        )
        forward_score = torch.clamp(
          (forward_gain - temporal_probe_min_forward)
          / max(temporal_probe_forward_scale, 1.0e-6),
          0.0,
          1.0,
        )
        forward_delta = torch.relu(
          forward_score - self._probe_target_forward_progress[phase1_for_step]
        )
        self._probe_target_forward_progress[phase1_for_step] = torch.maximum(
          self._probe_target_forward_progress[phase1_for_step], forward_score
        )
        safe_progress = target_speed_safe[phase1_for_step].float()
        target_forward_reward[phase1_for_step] = (
          forward_delta * safe_progress * temporal_probe_forward_reward
        )

      shallow_layer2_points = self._shallow_layer_points(
        toe_points,
        boundaries,
        base_valid,
        probe_boundary_layers,
        temporal_probe_shallow_depth,
        u_margin,
        v_margin,
        surface_tol,
      )
      foot_indices = torch.arange(num_feet, device=env.device).view(1, num_feet, 1)
      first_foot_points = foot_indices == self._probe_first_foot.clamp(
        0, max(1, num_feet - 1)
      ).view(-1, 1, 1)
      target_foot_points = foot_indices == target_foot.view(-1, 1, 1)
      protected_first_points = (
        first_hit_mask[:, None, None] & first_foot_points & (point_layers == 1)
      )
      protected_second_points = (
        phase1_for_step[:, None, None]
        & target_speed_safe[:, None, None]
        & target_foot_points
        & (point_layers == 2)
        & shallow_layer2_points
      )
      temporal_protected_points = protected_first_points | protected_second_points
      if bool(torch.any(temporal_protected_points).item()):
        protected_penalty = torch.sum(
          torch.where(
            temporal_protected_points,
            point_penalty,
            torch.zeros_like(point_penalty),
          ),
          dim=(1, 2),
        )
      effective_point_penalty = penalty - protected_penalty

      protected_first_contact = (
        first_hit_mask[:, None, None] & first_foot_points & (contact_layers == 1)
      )
      target_layer2_point_mask = point_layers[env_ids, target_foot] == 2
      target_shallow_contact = torch.any(target_layer2_point_mask, dim=-1) & ~torch.any(
        target_layer2_point_mask & ~shallow_layer2_points[env_ids, target_foot],
        dim=-1,
      )
      protected_second_contact = (
        phase1_for_step[:, None, None]
        & target_speed_safe[:, None, None]
        & target_foot_points
        & (contact_layers == 2)
        & target_shallow_contact[:, None, None]
      )
      temporal_protected_contact = (
        protected_first_contact | protected_second_contact
      ) & probe_layer_contact
      unprotected_probe_contact = probe_layer_contact & ~temporal_protected_contact
      unprotected_probe_hit_by_foot = torch.any(unprotected_probe_contact, dim=-1)
      unprotected_probe_hit_env = torch.any(unprotected_probe_hit_by_foot, dim=-1)
      unprotected_probe_strength_by_foot = torch.max(
        torch.where(
          unprotected_probe_contact,
          hit_strength,
          torch.zeros_like(hit_strength),
        ),
        dim=-1,
      ).values
      unprotected_probe_strength = torch.max(
        torch.where(
          unprotected_probe_hit_by_foot,
          unprotected_probe_strength_by_foot,
          torch.zeros_like(unprotected_probe_strength_by_foot),
        ),
        dim=-1,
      ).values
      contact_penalty = contact_penalty + (
        unprotected_probe_hit_env.float()
        * unprotected_probe_strength
        * contact_penalty_scale
      )

      if bool(torch.any(target_hit_mask).item()):
        self._probe_phase[target_hit_mask] = 2
        self._probe_contact_count[target_hit_mask] += 1
        self._probe_success[target_hit_mask] = True
        self._probe_second_confirmed[target_hit_mask] = True

      inactive = ~level_active
      if bool(torch.any(inactive).item()):
        self.reset(inactive.nonzero(as_tuple=False).squeeze(-1))

      probe_reward = (
        first_probe_reward
        + second_confirm_reward
        + target_lift_reward
        + target_forward_reward
        - target_overspeed_penalty
      )
      raw_penalty = effective_point_penalty + contact_penalty - probe_reward

      env.extras["log"]["Metrics/toe_riser_slab_probe_active_ratio"] = (
        active_gate.float().mean()
      )
      env.extras["log"]["Metrics/toe_riser_slab_probe_slab_neutral_ratio"] = (
        temporal_protected_points.float().mean()
      )
      env.extras["log"]["Metrics/toe_riser_temporal_probe_phase_mean"] = (
        self._probe_phase.float().mean()
      )
      env.extras["log"]["Metrics/toe_riser_temporal_probe_first_left_ratio"] = (
        (self._probe_first_foot == 0).float().mean()
      )
      env.extras["log"]["Metrics/toe_riser_temporal_probe_first_right_ratio"] = (
        (self._probe_first_foot == 1).float().mean()
      )
      env.extras["log"]["Metrics/toe_riser_temporal_probe_success_ratio"] = (
        self._probe_success.float().mean()
      )
      env.extras["log"]["Metrics/toe_riser_temporal_probe_target_lift_mean"] = (
        self._probe_target_lift_progress.mean()
      )
      env.extras["log"]["Metrics/toe_riser_temporal_probe_target_forward_gain_mean"] = (
        self._probe_target_forward_progress.mean()
      )
      env.extras["log"]["Metrics/toe_riser_temporal_probe_second_confirm_ratio"] = (
        self._probe_second_confirmed.float().mean()
      )
      env.extras["log"]["Metrics/toe_riser_temporal_probe_timeout_ratio"] = (
        timeout_mask.float().mean()
      )
      env.extras["log"]["Metrics/toe_riser_temporal_probe_forward_velocity_mean"] = (
        torch.where(
          phase1_for_step,
          target_forward_vel,
          torch.zeros_like(target_forward_vel),
        ).sum()
        / phase1_for_step.float().sum().clamp_min(1.0)
      )
      phase1_count = phase1_for_step.float().sum().clamp_min(1.0)
      boundary_candidate = phase1_for_step & has_target_layer2_contact
      boundary_candidate_count = boundary_candidate.float().sum().clamp_min(1.0)
      env.extras["log"]["Metrics/toe_riser_probe_target_forward_vel_mean"] = (
        torch.where(
          phase1_for_step,
          target_forward_vel,
          torch.zeros_like(target_forward_vel),
        ).sum()
        / phase1_count
      )
      env.extras["log"]["Metrics/toe_riser_probe_second_hit_dir_gate_ratio"] = (
        phase1_for_step & target_direction_gate
      ).float().sum() / phase1_count
      env.extras["log"]["Metrics/toe_riser_probe_second_hit_boundary_gate_ratio"] = (
        boundary_candidate & target_boundary_gate
      ).float().sum() / boundary_candidate_count
      target_layer2_points = (
        phase1_for_step[:, None, None] & target_foot_points & (point_layers == 2)
      )
      target_layer2_point_count = target_layer2_points.float().sum().clamp_min(1.0)
      env.extras["log"]["Metrics/toe_riser_probe_shallow_protection_ratio"] = (
        protected_second_points.float().sum() / target_layer2_point_count
      )
      env.extras["log"]["Metrics/toe_riser_probe_deep_rejected_ratio"] = (
        target_layer2_points & ~shallow_layer2_points
      ).float().sum() / target_layer2_point_count
      env.extras["log"]["Metrics/toe_riser_temporal_probe_overspeed_ratio"] = (
        (phase1_for_step & ~target_speed_safe).float().mean()
      )
      same_foot_hit = (
        phase1_for_step
        & active_gate
        & new_hit_by_foot[
          env_ids, self._probe_first_foot.clamp(0, max(1, num_feet - 1))
        ]
      )
      env.extras["log"]["Metrics/toe_riser_temporal_probe_same_foot_repeat_ratio"] = (
        same_foot_hit.float().mean()
      )
    else:
      zero = torch.zeros((), device=env.device)
      env.extras["log"]["Metrics/toe_riser_slab_contact_penalty_mean"] = zero
      env.extras["log"]["Metrics/toe_riser_slab_probe_reward_mean"] = zero
      env.extras["log"]["Metrics/toe_riser_slab_probe_count_mean"] = zero
      env.extras["log"]["Metrics/toe_riser_slab_probe_layer1_hit_ratio"] = zero
      env.extras["log"]["Metrics/toe_riser_slab_probe_layer2_hit_ratio"] = zero
      env.extras["log"]["Metrics/toe_riser_slab_contact_time_mean"] = zero
      env.extras["log"]["Metrics/toe_riser_slab_probe_active_ratio"] = zero
      env.extras["log"]["Metrics/toe_riser_slab_probe_slab_neutral_ratio"] = zero
      env.extras["log"]["Metrics/toe_riser_probe_target_forward_vel_mean"] = zero
      env.extras["log"]["Metrics/toe_riser_probe_second_hit_dir_gate_ratio"] = zero
      env.extras["log"]["Metrics/toe_riser_probe_second_hit_boundary_gate_ratio"] = zero
      env.extras["log"]["Metrics/toe_riser_probe_shallow_protection_ratio"] = zero
      env.extras["log"]["Metrics/toe_riser_probe_deep_rejected_ratio"] = zero

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
      torch.zeros((), device=env.device)
    )
    env.extras["log"]["Metrics/toe_riser_slab_raw_mean"] = raw_penalty.mean()

    return raw_penalty


class toe_step_riser_approach_penalty(toe_step_riser_slab_penalty):
  """Backward-compatible alias for the temporal toe riser slab penalty."""


class probe_aware_feet_gait:
  """Use an event-local support/swing schedule during stair probing."""

  def __init__(self, cfg: RewardTermCfg, env) -> None:
    del cfg
    self._phase_offset = torch.zeros(env.num_envs, device=env.device)
    self._was_probe_active = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._phase_offset[env_ids] = 0.0
    self._was_probe_active[env_ids] = False

  @staticmethod
  def _recovery_phase(is_contact: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    left_only = is_contact[:, 0] & ~is_contact[:, 1]
    right_only = is_contact[:, 1] & ~is_contact[:, 0]
    both = is_contact[:, 0] & is_contact[:, 1]
    phase = torch.where(left_only, torch.full_like(fallback, 0.25), fallback)
    phase = torch.where(right_only, torch.full_like(fallback, 0.75), phase)
    return torch.where(both, torch.zeros_like(phase), phase)

  def __call__(
    self,
    env,
    period: float,
    offset: list[float],
    threshold: float,
    command_threshold: float,
    command_name: str,
    sensor_name: str,
    support_speed_scale: float = 0.20,
    asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_BODY_CFG,
  ) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    assert isinstance(sensor, ContactSensor)
    current_contact_time = sensor.data.current_contact_time
    assert current_contact_time is not None, (
      "Enable track_air_time=True for the probe gait contact sensor."
    )
    is_contact = current_contact_time > 0
    if is_contact.shape[-1] != 2:
      raise RuntimeError(
        "probe_aware_feet_gait requires exactly two foot contact channels, got "
        f"{is_contact.shape[-1]}."
      )

    stage = env.extras.get(PROBE_STAGE_KEY)
    target_foot = env.extras.get(PROBE_TARGET_FOOT_KEY)
    if stage is None or target_foot is None:
      stage = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
      target_foot = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    probe_active = stage == 1

    raw_phase = (env.episode_length_buf * env.step_dt) / period
    exit_probe = self._was_probe_active & ~probe_active
    if bool(torch.any(exit_probe).item()):
      desired_phase = self._recovery_phase(is_contact, raw_phase % 1.0)
      self._phase_offset[exit_probe] = (
        desired_phase[exit_probe] - raw_phase[exit_probe]
      ) % 1.0

    global_phase = (raw_phase + self._phase_offset).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(
      1, -1
    )
    normal_stance = ((global_phase + offsets) % 1.0) < threshold
    normal_reward = (normal_stance == is_contact).float().mean(dim=1)

    target_foot = target_foot.clamp(0, 1)
    support_foot = 1 - target_foot
    env_ids = torch.arange(env.num_envs, device=env.device)
    support_contact = is_contact[env_ids, support_foot].float()
    target_swing = (~is_contact[env_ids, target_foot]).float()

    asset: Entity = env.scene[asset_cfg.name]
    support_vel = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]
    support_vel = support_vel[env_ids, support_foot]
    support_speed = torch.norm(support_vel, dim=-1)
    speed_scale = max(support_speed_scale, 1.0e-6)
    stable_support = torch.exp(-torch.square(support_speed / speed_scale))
    probe_reward = 0.5 * (target_swing + support_contact * stable_support)

    command = env.command_manager.get_command(command_name)
    if command is not None:
      total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      normal_reward *= (total_command > command_threshold).float()

    reward = torch.where(probe_active, probe_reward, normal_reward)
    self._was_probe_active.copy_(probe_active)

    log = env.extras["log"]
    log["Metrics/probe_gait_active_ratio"] = probe_active.float().mean()
    log["Metrics/probe_gait_target_swing_ratio"] = torch.where(
      probe_active, target_swing, torch.zeros_like(target_swing)
    ).mean()
    log["Metrics/probe_gait_support_contact_ratio"] = torch.where(
      probe_active, support_contact, torch.zeros_like(support_contact)
    ).mean()
    log["Metrics/probe_gait_support_speed_mean"] = torch.where(
      probe_active, support_speed, torch.zeros_like(support_speed)
    ).sum() / probe_active.float().sum().clamp_min(1.0)
    return reward


class stair_tread_landing_reward(_StepBoundaryFootVolume):
  """Reward stage-2 footfalls near the privileged tread safe target."""

  def __init__(self, cfg: RewardTermCfg, env) -> None:
    super().__init__(cfg, env)

  @staticmethod
  def _toe_slab_mask(
    toe_points: torch.Tensor,
    boundaries: torch.Tensor,
    valid_boundaries: torch.Tensor,
    slab_depth: float,
    u_margin: float,
    v_margin: float,
    surface_tol: float,
  ) -> torch.Tensor:
    """Return a per-foot hard gate for geometric toe-slab penetration."""
    p0 = boundaries[:, None, None, :, 0:3]
    p1 = boundaries[:, None, None, :, 3:6]
    normal_to_low = boundaries[:, None, None, :, 6:9]
    z_low = boundaries[:, None, None, :, 9]
    z_high = boundaries[:, None, None, :, 10]

    tangent_u = p1 - p0
    edge_len = torch.norm(tangent_u, dim=-1).clamp_min(1.0e-12)
    tangent_u = tangent_u / edge_len[..., None]
    center = 0.5 * (p0 + p1)
    center[..., 2] = 0.5 * (z_low + z_high)
    rel = toe_points[:, :, :, None, :] - center
    s = torch.sum(rel * normal_to_low, dim=-1)
    u = torch.sum(rel * tangent_u, dim=-1)
    v = rel[..., 2]
    inside_face = (torch.abs(u) <= 0.5 * edge_len + u_margin) & (
      torch.abs(v) <= 0.5 * (z_high - z_low) + v_margin
    )
    inside_slab = (s >= -surface_tol) & (s <= slab_depth)
    valid = valid_boundaries[:, None, None, :]
    return torch.any(valid & inside_face & inside_slab, dim=(2, 3))

  def __call__(
    self,
    env,
    ground_contact_sensor_name: str,
    toe_contact_sensor_name: str,
    landing_bias: float = 0.60,
    sigma_fraction: float = 0.15,
    min_sigma: float = 0.03,
    back_margin: float = 0.06,
    front_margin: float = 0.08,
    lateral_margin: float = 0.03,
    height_tolerance: float = 0.08,
    short_penalty_scale: float = 0.40,
    over_front_penalty_scale: float = 0.60,
    vertical_normal_z_max: float = 0.40,
    min_terrain_level: int = 3,
    min_landing_layer: int = 3,
    lip_edge_radius: float = 0.07,
    lip_margin: float = 0.01,
    lip_edge_height_band: float = 0.06,
    slab_depth: float = 0.10,
    slab_u_margin: float = 0.02,
    slab_v_margin: float = 0.05,
    toe_x_min: float = 0.08,
    surface_tol: float = 0.005,
    asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_SITE_CFG,
    foot_body_cfg: SceneEntityCfg = _DEFAULT_FOOT_BODY_CFG,
  ) -> torch.Tensor:
    boundaries, valid_boundaries = _current_step_boundaries(env)
    if boundaries is None or valid_boundaries is None:
      return torch.zeros(env.num_envs, device=env.device)

    ground_sensor = env.scene[ground_contact_sensor_name]
    toe_sensor = env.scene[toe_contact_sensor_name]
    assert isinstance(ground_sensor, ContactSensor)
    assert isinstance(toe_sensor, ContactSensor)
    first_contact = ground_sensor.compute_first_contact(dt=env.step_dt)
    num_feet = first_contact.shape[-1]
    if num_feet != 2:
      raise RuntimeError(
        "stair_tread_landing_reward requires exactly two ground-contact channels."
      )

    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
    foot_points_w, _point_vel_w = self._foot_points_w(env, foot_body_cfg)
    toe_points_w = foot_points_w[:, :, self._local_x >= toe_x_min, :]
    tread_depth, _riser_height, shape_valid = cached_stair_shape(
      env, boundaries, valid_boundaries
    )

    p0_xy = boundaries[:, None, :, 0:2]
    p1_xy = boundaries[:, None, :, 3:5]
    edge_xy = p1_xy - p0_xy
    edge_len = torch.norm(edge_xy, dim=-1).clamp_min(1.0e-6)
    rel_xy = foot_pos_w[:, :, None, 0:2] - p0_xy
    edge_fraction = torch.sum(rel_xy * edge_xy, dim=-1) / torch.square(edge_len)
    lateral_slack = lateral_margin / edge_len
    within_edge = (edge_fraction >= -lateral_slack) & (
      edge_fraction <= 1.0 + lateral_slack
    )

    normal_to_low = boundaries[:, None, :, 6:8]
    normal_to_low = normal_to_low / torch.norm(
      normal_to_low, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    landing_s = torch.sum(rel_xy * -normal_to_low, dim=-1)
    depth = tread_depth[:, None, None]
    desired_s = landing_bias * depth
    sigma = torch.clamp(sigma_fraction * depth, min=min_sigma)
    height_error = torch.abs(foot_pos_w[:, :, None, 2] - boundaries[:, None, :, 10])
    candidate = (
      valid_boundaries[:, None, :]
      & within_edge
      & (height_error <= height_tolerance)
      & (landing_s >= 0.0)
      & (landing_s <= depth + front_margin)
    )
    gaussian_score = torch.exp(-torch.square((landing_s - desired_s) / sigma))
    candidate_score = torch.where(
      candidate, gaussian_score, torch.zeros_like(gaussian_score)
    )
    best_score, best_idx = torch.max(candidate_score, dim=-1)
    best_s = torch.gather(landing_s, dim=-1, index=best_idx[..., None]).squeeze(-1)
    has_candidate = torch.any(candidate, dim=-1)

    boundary_layers = _step_boundary_layers(
      boundaries, valid_boundaries, boundaries.shape[1]
    )
    expanded_layers = boundary_layers[:, None, :].expand(env.num_envs, num_feet, -1)
    best_boundary_layer = torch.gather(
      expanded_layers, dim=-1, index=best_idx[..., None]
    ).squeeze(-1)
    landing_layer_valid = best_boundary_layer >= min_landing_layer

    edge_clearance = lip_edge_radius + lip_margin
    effective_back_margin = max(back_margin, edge_clearance)
    effective_front_margin = max(front_margin, edge_clearance)
    safe_band = (best_s >= effective_back_margin) & (
      best_s <= depth.squeeze(-1) - effective_front_margin
    )
    middle_forward_reward = best_score * safe_band.float()
    short_penalty = torch.relu(
      (desired_s.squeeze(-1) - sigma.squeeze(-1) - best_s)
      / depth.squeeze(-1).clamp_min(1.0e-6)
    )
    over_front_penalty = torch.relu(
      (best_s - (depth.squeeze(-1) - effective_front_margin))
      / depth.squeeze(-1).clamp_min(1.0e-6)
    )

    slab_unsafe = self._toe_slab_mask(
      toe_points_w,
      boundaries,
      valid_boundaries,
      slab_depth,
      slab_u_margin,
      slab_v_margin,
      surface_tol,
    )
    expanded_boundaries = boundaries[:, None, :, :].expand(
      env.num_envs, num_feet, -1, -1
    )
    expanded_valid = valid_boundaries[:, None, :].expand(env.num_envs, num_feet, -1)
    lip_min_dist = self._lip_min_dist(
      foot_points_w,
      expanded_boundaries,
      expanded_valid,
      lip_edge_height_band,
    )
    lip_unsafe = torch.any(lip_min_dist < edge_clearance, dim=-1)

    toe_normal = toe_sensor.data.normal
    toe_found = toe_sensor.data.found
    assert toe_normal is not None and toe_found is not None
    toe_normal = toe_normal.reshape(env.num_envs, num_feet, -1, 3)
    toe_found = toe_found.reshape(env.num_envs, num_feet, -1)
    riser_contact = torch.any(
      (toe_found > 0) & (toe_normal[..., 2].abs() < vertical_normal_z_max), dim=-1
    )

    probe_stage = env.extras.get(PROBE_STAGE_KEY)
    if probe_stage is None:
      stage2 = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    else:
      stage2 = probe_stage >= 2
    env_active = stage2 & shape_valid & _terrain_level_active(env, min_terrain_level)
    safety_candidate = (
      first_contact & env_active[:, None] & has_candidate & landing_layer_valid
    )
    touchdown_candidate = safety_candidate & ~riser_contact & ~slab_unsafe & ~lip_unsafe
    landing_active = touchdown_candidate & safe_band
    per_foot = middle_forward_reward * landing_active.float()
    per_foot -= short_penalty_scale * short_penalty * touchdown_candidate.float()
    per_foot -= (
      over_front_penalty_scale * over_front_penalty * touchdown_candidate.float()
    )

    log = env.extras["log"]
    landing_count = landing_active.float().sum().clamp_min(1.0)
    safety_count = safety_candidate.float().sum().clamp_min(1.0)
    log["Metrics/stair_tread_landing_active_ratio"] = landing_active.float().mean()
    log["Metrics/stair_landing_active_ratio"] = landing_active.float().mean()
    log["Metrics/stair_landing_layer_mean"] = (
      best_boundary_layer.float() * landing_active.float()
    ).sum() / landing_count
    log["Metrics/stair_landing_layer_ge3_ratio"] = (
      landing_active & landing_layer_valid
    ).float().sum() / landing_count
    log["Metrics/stair_landing_safe_slab_ratio"] = (
      safety_candidate & ~slab_unsafe
    ).float().sum() / safety_count
    log["Metrics/stair_landing_safe_lip_ratio"] = (
      safety_candidate & ~lip_unsafe
    ).float().sum() / safety_count
    log["Metrics/stair_landing_rejected_by_slab_ratio"] = (
      safety_candidate & slab_unsafe
    ).float().sum() / safety_count
    log["Metrics/stair_landing_rejected_by_lip_ratio"] = (
      safety_candidate & lip_unsafe
    ).float().sum() / safety_count
    log["Metrics/stair_tread_landing_score_mean"] = (
      middle_forward_reward * landing_active.float()
    ).sum() / landing_count
    log["Metrics/stair_tread_landing_short_ratio"] = (
      (landing_active & (short_penalty > 0.0)).float().mean()
    )
    log["Metrics/stair_tread_landing_over_front_ratio"] = (
      (landing_active & (over_front_penalty > 0.0)).float().mean()
    )
    return per_foot.sum(dim=-1)
