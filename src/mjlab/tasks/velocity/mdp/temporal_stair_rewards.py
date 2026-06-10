from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from .rewards import (
  _DEFAULT_FOOT_BODY_CFG,
  _current_step_boundaries,
  _terrain_level_active,
  toe_step_riser_slab_penalty as _LegacyToeStepRiserSlabPenalty,
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
    self._probe_first_toe_x_body = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
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
    temporal_probe_max_forward_vel: float = 0.45,
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
      inactive = ~active_gate
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
      )
      contact_penalty = ctd["contact_penalty"]
      new_hit_by_foot = ctd["new_hit_by_foot"]
      foot_forward_vel = ctd["foot_forward_vel"]
      foot_pos_w = ctd["foot_pos_w"]
      probe_layer_contact = ctd["probe_layer_contact"]
      contact_layers = ctd["contact_layers"]
      hit_strength = ctd["hit_strength"]

      root_pos_w = asset.data.root_link_pos_w[:, None, :]
      root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(
        num_envs, num_feet, 4
      )
      foot_pos_b = quat_apply_inverse(root_quat_w, foot_pos_w - root_pos_w)

      first_layer_contact_by_foot = torch.any(
        probe_layer_contact & (contact_layers == 1), dim=-1
      )
      first_layer_new_hit_by_foot = new_hit_by_foot & first_layer_contact_by_foot
      first_hit_mask = (
        (self._probe_phase == 0)
        & active_gate
        & torch.any(first_layer_new_hit_by_foot, dim=-1)
      )
      first_probe_reward = torch.zeros(env.num_envs, device=env.device)
      if bool(torch.any(first_hit_mask).item()):
        right_hit = first_layer_new_hit_by_foot[:, 1]
        self._probe_first_foot[first_hit_mask] = torch.where(
          right_hit[first_hit_mask],
          torch.tensor(1, device=env.device, dtype=torch.long),
          torch.tensor(0, device=env.device, dtype=torch.long),
        )
        self._probe_target_foot[first_hit_mask] = (
          1 - self._probe_first_foot[first_hit_mask]
        )
        self._probe_phase[first_hit_mask] = 1
        self._probe_timer[first_hit_mask] = 0
        self._probe_contact_count[first_hit_mask] = 1
        self._probe_target_lift_progress[first_hit_mask] = 0.0
        self._probe_target_forward_progress[first_hit_mask] = 0.0
        self._probe_second_confirmed[first_hit_mask] = False
        first_probe_reward[first_hit_mask] = temporal_probe_first_reward
        for foot_idx in range(num_feet):
          foot_mask = first_hit_mask & (self._probe_first_foot == foot_idx)
          if bool(torch.any(foot_mask).item()):
            self._probe_first_toe_x_body[foot_mask] = foot_pos_b[foot_mask, foot_idx, 0]
            self._probe_first_toe_z_world[foot_mask] = foot_pos_w[foot_mask, foot_idx, 2]

      self._probe_timer = torch.where(
        self._probe_phase == 1, self._probe_timer + 1, self._probe_timer
      )
      timeout_mask = (
        self._probe_phase == 1
      ) & (self._probe_timer > probe_timeout_steps)
      if bool(torch.any(timeout_mask).item()):
        self._probe_phase[timeout_mask] = 0
        self._probe_first_foot[timeout_mask] = -1
        self._probe_target_foot[timeout_mask] = -1
        self._probe_timer[timeout_mask] = 0
        self._probe_contact_count[timeout_mask] = 0
        self._probe_target_lift_progress[timeout_mask] = 0.0
        self._probe_target_forward_progress[timeout_mask] = 0.0
        self._probe_second_confirmed[timeout_mask] = False

      in_phase1 = self._probe_phase == 1
      phase1_for_step = in_phase1.clone()
      target_foot = self._probe_target_foot.clamp(0, max(1, num_feet - 1))
      env_ids = torch.arange(env.num_envs, device=env.device)
      second_layer_contact_by_foot = torch.any(
        probe_layer_contact & (contact_layers == 2), dim=-1
      )
      target_hit_mask = (
        in_phase1
        & active_gate
        & new_hit_by_foot[env_ids, target_foot]
        & second_layer_contact_by_foot[env_ids, target_foot]
        & ~self._probe_second_confirmed
      )
      second_confirm_reward = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.float32
      )
      if bool(torch.any(target_hit_mask).item()):
        self._probe_phase[target_hit_mask] = 2
        self._probe_contact_count[target_hit_mask] += 1
        self._probe_success[target_hit_mask] = True
        self._probe_second_confirmed[target_hit_mask] = True
        second_confirm_reward[target_hit_mask] = temporal_probe_confirm_reward

      if bool(torch.any(inactive).item()):
        self._probe_phase[inactive] = 0
        self._probe_first_foot[inactive] = -1
        self._probe_target_foot[inactive] = -1
        self._probe_timer[inactive] = 0
        self._probe_contact_count[inactive] = 0
        self._probe_success[inactive] = False
        self._probe_target_lift_progress[inactive] = 0.0
        self._probe_target_forward_progress[inactive] = 0.0
        self._probe_second_confirmed[inactive] = False

      target_lift_reward = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.float32
      )
      target_forward_reward = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.float32
      )
      target_overspeed_penalty = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.float32
      )
      in_phase1 = self._probe_phase == 1
      if bool(torch.any(in_phase1).item()):
        target_idx = target_foot[in_phase1]
        target_z = foot_pos_w[in_phase1, target_idx, 2]
        lift_gain = target_z - self._probe_first_toe_z_world[in_phase1]
        lift_score = torch.clamp(
          (lift_gain - temporal_probe_min_lift)
          / max(temporal_probe_lift_scale, 1.0e-6),
          0.0,
          1.0,
        )
        lift_delta = torch.relu(
          lift_score - self._probe_target_lift_progress[in_phase1]
        )
        self._probe_target_lift_progress[in_phase1] = torch.maximum(
          self._probe_target_lift_progress[in_phase1], lift_score
        )
        target_lift_reward[in_phase1] = lift_delta * temporal_probe_lift_reward

        target_x_body = foot_pos_b[in_phase1, target_idx, 0]
        forward_gain = target_x_body - self._probe_first_toe_x_body[in_phase1]
        forward_score = torch.clamp(
          (forward_gain - temporal_probe_min_forward)
          / max(temporal_probe_forward_scale, 1.0e-6),
          0.0,
          1.0,
        )
        forward_delta = torch.relu(
          forward_score - self._probe_target_forward_progress[in_phase1]
        )
        self._probe_target_forward_progress[in_phase1] = torch.maximum(
          self._probe_target_forward_progress[in_phase1], forward_score
        )
        target_forward_reward[in_phase1] = (
          forward_delta * temporal_probe_forward_reward
        )

        target_forward_vel = foot_forward_vel[in_phase1, target_idx]
        overspeed = torch.relu(target_forward_vel - temporal_probe_max_forward_vel)
        target_overspeed_penalty[in_phase1] = (
          overspeed * temporal_probe_overspeed_penalty
        )

      foot_indices = torch.arange(num_feet, device=env.device).view(1, num_feet, 1)
      target_foot_for_points = self._probe_target_foot.clamp(0, max(1, num_feet - 1))
      target_foot_point_mask = foot_indices == target_foot_for_points.view(-1, 1, 1)
      temporal_protected_points = (
        active_gate[:, None, None]
        & phase1_for_step[:, None, None]
        & target_foot_point_mask
        & (point_layers > 0)
      )
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

      temporal_protected_contact = (
        active_gate[:, None, None]
        & phase1_for_step[:, None, None]
        & (foot_indices == target_foot_for_points.view(-1, 1, 1))
        & probe_layer_contact
      )
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
        self._probe_first_foot == 0
      ).float().mean()
      env.extras["log"]["Metrics/toe_riser_temporal_probe_first_right_ratio"] = (
        self._probe_first_foot == 1
      ).float().mean()
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
      same_foot_hit = (
        in_phase1
        & active_gate
        & new_hit_by_foot[env_ids, self._probe_first_foot.clamp(0, max(1, num_feet - 1))]
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
