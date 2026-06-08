"""Metrics for fixed velocity terrain evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity import mdp
from mjlab.utils.lab_api.math import quat_apply_inverse

try:
  from mjlab.tasks.velocity.mdp.target_heading_rewards import _current_step_boundaries
except ImportError:
  from mjlab.tasks.velocity.mdp.rewards import _current_step_boundaries

MEAN_METRIC_NAMES = (
  "tracking_error",
  "tracking_lin_error",
  "tracking_yaw_error",
  "base_pitch_roll_rms",
  "action_smoothness",
  "torque_cost",
  "foot_clearance",
)

EVENT_COUNT_NAMES = (
  "toe_riser_collision",
  "heel_riser_collision",
  "foot_lip_collision",
)

LEVEL_EVENT_NAMES = (
  "toe_riser_collision_by_level",
  "heel_riser_collision_by_level",
  "foot_lip_collision_by_level",
)


@dataclass(frozen=True)
class StairMetricParams:
  """Geometry parameters for stair interaction events."""

  contact_sensor_name: str = "toe_terrain_contact"
  vertical_normal_z_max: float = 0.4
  contact_force_threshold: float = 5.0
  contact_cooldown_s: float = 0.10
  edge_radius: float = 0.07
  edge_height_band: float = 0.06
  slab_depth: float = 0.10
  toe_x_min: float = 0.08
  toe_v_threshold: float = 0.02
  approach_speed_floor: float = 0.08
  heel_clearance: float = 0.10
  heel_x_max: float = 0.0
  u_margin: float = 0.04
  v_margin: float = 0.06
  surface_tol: float = 0.005
  nearest_boundaries: int = 4


class StairEventDetector:
  """Reusable detector for toe/heel/lip events around stair risers."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    params: StairMetricParams | None = None,
  ) -> None:
    self.params = params or StairMetricParams()
    self.foot_asset_cfg = SceneEntityCfg(
      "robot",
      body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
      preserve_order=True,
    )
    self.foot_asset_cfg.resolve(env.scene)
    num_feet = len(self._foot_body_ids())

    self._contact_sensor = self._get_contact_sensor(env)
    self.event_source = (
      "true_contact" if self._contact_sensor is not None else "geometry_risk_zone"
    )
    self._prev_toe_contact = torch.zeros(
      env.num_envs, num_feet, device=env.device, dtype=torch.bool
    )
    self._prev_heel_contact = torch.zeros(
      env.num_envs, num_feet, device=env.device, dtype=torch.bool
    )
    self._toe_contact_cooldown = torch.zeros(env.num_envs, num_feet, device=env.device)
    self._heel_contact_cooldown = torch.zeros(env.num_envs, num_feet, device=env.device)
    self._last_true_contact_slots: dict[str, torch.Tensor] | None = None
    self._last_true_contact_pos_w: torch.Tensor | None = None

    self.toe_params = {
      "slab_depth": self.params.slab_depth,
      "u_margin": self.params.u_margin,
      "v_margin": self.params.v_margin,
      "toe_x_min": self.params.toe_x_min,
      "toe_v_threshold": self.params.toe_v_threshold,
      "approach_speed_floor": self.params.approach_speed_floor,
      "surface_tol": self.params.surface_tol,
      "nearest_boundaries": self.params.nearest_boundaries,
      "min_terrain_level": None,
      "asset_cfg": self.foot_asset_cfg,
    }
    self.heel_params = {
      "heel_clearance": self.params.heel_clearance,
      "u_margin": self.params.u_margin,
      "v_margin": self.params.v_margin,
      "heel_x_max": self.params.heel_x_max,
      "surface_tol": self.params.surface_tol,
      "nearest_boundaries": self.params.nearest_boundaries,
      "contact_sensor_name": "feet_ground_contact",
      "min_terrain_level": None,
      "asset_cfg": self.foot_asset_cfg,
    }
    self.lip_params = {
      "edge_radius": self.params.edge_radius,
      "edge_height_band": self.params.edge_height_band,
      "support_speed_floor": 0.08,
      "nearest_boundaries": self.params.nearest_boundaries,
      "contact_sensor_name": "feet_ground_contact",
      "min_terrain_level": None,
      "asset_cfg": self.foot_asset_cfg,
    }

    self._toe = None
    self._heel = None
    self._lip = None
    if self._contact_sensor is None:
      self._toe = self._make_geometry_detector(
        "toe_step_riser_slab_penalty", self.toe_params, env
      )
      self._heel = self._make_geometry_detector(
        "heel_step_riser_clearance_penalty", self.heel_params, env
      )
      self._lip = self._make_geometry_detector(
        "foot_step_lip_volume_penalty", self.lip_params, env
      )

  def _make_geometry_detector(
    self,
    name: str,
    params: dict,
    env: ManagerBasedRlEnv,
  ):
    detector_cls = getattr(mdp, name, None)
    if detector_cls is None:
      return None
    return detector_cls(
      RewardTermCfg(func=detector_cls, weight=0.0, params=params),
      env,
    )

  def _get_contact_sensor(self, env: ManagerBasedRlEnv) -> ContactSensor | None:
    try:
      sensor = env.scene[self.params.contact_sensor_name]
    except (KeyError, AttributeError):
      return None
    if isinstance(sensor, ContactSensor):
      return sensor
    return None

  def _foot_body_ids(self) -> list[int]:
    body_ids = self.foot_asset_cfg.body_ids
    if not isinstance(body_ids, list):
      raise RuntimeError("StairEventDetector foot body IDs were not resolved.")
    return body_ids

  def compute_events(self, env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    """Return per-env binary event indicators for this step."""
    if self._contact_sensor is not None:
      return self._compute_true_riser_contact_events(env)

    zeros = torch.zeros(env.num_envs, device=env.device)
    toe = self._toe(env, **self.toe_params) if self._toe is not None else zeros
    heel = self._heel(env, **self.heel_params) if self._heel is not None else zeros
    lip = self._lip(env, **self.lip_params) if self._lip is not None else zeros
    return {
      "toe_riser_collision": (toe > 0.0).float(),
      "heel_riser_collision": (heel > 0.0).float(),
      "foot_lip_collision": (lip > 0.0).float(),
    }

  def get_last_true_contact_positions(
    self, kind: str = "toe"
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(env_ids, positions_w)`` for latest true riser contact events."""
    device = self._prev_toe_contact.device
    empty_env_ids = torch.empty(0, device=device, dtype=torch.long)
    empty_positions = torch.empty(0, 3, device=device)
    if self._last_true_contact_slots is None or self._last_true_contact_pos_w is None:
      return empty_env_ids, empty_positions

    active_slots = self._last_true_contact_slots.get(kind)
    if active_slots is None:
      return empty_env_ids, empty_positions
    indices = active_slots.nonzero(as_tuple=False)
    if indices.numel() == 0:
      return empty_env_ids, empty_positions
    return indices[:, 0].to(dtype=torch.long), self._last_true_contact_pos_w[
      active_slots
    ]

  def _compute_true_riser_contact_events(
    self,
    env: ManagerBasedRlEnv,
  ) -> dict[str, torch.Tensor]:
    sensor = self._contact_sensor
    if sensor is None:
      raise RuntimeError("True riser contact metrics require a contact sensor.")

    data = sensor.data
    if (
      data.found is None
      or data.force is None
      or data.normal is None
      or data.pos is None
    ):
      missing = [
        name
        for name, value in (
          ("found", data.found),
          ("force", data.force),
          ("normal", data.normal),
          ("pos", data.pos),
        )
        if value is None
      ]
      raise RuntimeError(
        f"Contact sensor '{self.params.contact_sensor_name}' is missing fields: "
        f"{', '.join(missing)}"
      )

    asset = env.scene[self.foot_asset_cfg.name]
    num_envs = env.num_envs
    foot_body_ids = self._foot_body_ids()
    num_feet = len(foot_body_ids)
    num_contacts = data.found.shape[1]
    if num_contacts % num_feet != 0:
      raise RuntimeError(
        f"Contact sensor '{self.params.contact_sensor_name}' has {num_contacts} "
        f"slots, which is not divisible by {num_feet} feet."
      )
    num_slots = num_contacts // num_feet

    found = data.found.view(num_envs, num_feet, num_slots) > 0
    force_w = data.force.view(num_envs, num_feet, num_slots, 3)
    normal_w = data.normal.view(num_envs, num_feet, num_slots, 3)
    contact_pos_w = data.pos.view(num_envs, num_feet, num_slots, 3)

    foot_pos_w = asset.data.body_link_pos_w[:, foot_body_ids, :]
    foot_quat_w = asset.data.body_link_quat_w[:, foot_body_ids, :]
    expanded_quat = foot_quat_w[:, :, None, :].expand(num_envs, num_feet, num_slots, 4)
    contact_pos_b = quat_apply_inverse(
      expanded_quat,
      contact_pos_w - foot_pos_w[:, :, None, :],
    )

    is_vertical_face = torch.abs(normal_w[..., 2]) < self.params.vertical_normal_z_max
    force_mag = torch.norm(force_w, dim=-1)
    forceful = force_mag >= self.params.contact_force_threshold
    valid_riser_contact = found & is_vertical_face & forceful

    toe_hit_by_slot = valid_riser_contact & (
      contact_pos_b[..., 0] >= self.params.toe_x_min
    )
    heel_hit_by_slot = valid_riser_contact & (
      contact_pos_b[..., 0] <= self.params.heel_x_max
    )
    toe_hit_by_foot = toe_hit_by_slot.any(dim=-1)
    heel_hit_by_foot = heel_hit_by_slot.any(dim=-1)

    self._toe_contact_cooldown = torch.clamp(
      self._toe_contact_cooldown - env.step_dt, min=0.0
    )
    self._heel_contact_cooldown = torch.clamp(
      self._heel_contact_cooldown - env.step_dt, min=0.0
    )
    new_toe_by_foot = (
      toe_hit_by_foot & ~self._prev_toe_contact & (self._toe_contact_cooldown <= 0.0)
    )
    new_heel_by_foot = (
      heel_hit_by_foot & ~self._prev_heel_contact & (self._heel_contact_cooldown <= 0.0)
    )
    if bool(new_toe_by_foot.any().item()):
      self._toe_contact_cooldown = torch.where(
        new_toe_by_foot,
        torch.full_like(self._toe_contact_cooldown, self.params.contact_cooldown_s),
        self._toe_contact_cooldown,
      )
    if bool(new_heel_by_foot.any().item()):
      self._heel_contact_cooldown = torch.where(
        new_heel_by_foot,
        torch.full_like(self._heel_contact_cooldown, self.params.contact_cooldown_s),
        self._heel_contact_cooldown,
      )
    self._prev_toe_contact = toe_hit_by_foot
    self._prev_heel_contact = heel_hit_by_foot

    self._last_true_contact_slots = {
      "toe": toe_hit_by_slot & new_toe_by_foot[:, :, None],
      "heel": heel_hit_by_slot & new_heel_by_foot[:, :, None],
    }
    self._last_true_contact_pos_w = contact_pos_w

    zeros = torch.zeros(num_envs, device=env.device)
    return {
      "toe_riser_collision": new_toe_by_foot.float().sum(dim=1),
      "heel_riser_collision": new_heel_by_foot.float().sum(dim=1),
      "foot_lip_collision": zeros,
    }

  def compute_events_by_level(
    self,
    env: ManagerBasedRlEnv,
    *,
    terrain_height_m: float | None,
    max_levels: int,
  ) -> dict[str, torch.Tensor]:
    """Return per-env event indicators grouped by stair level."""
    empty = {
      name: torch.zeros(env.num_envs, max_levels, device=env.device)
      for name in LEVEL_EVENT_NAMES
    }
    if terrain_height_m is None or terrain_height_m <= 0.0:
      return empty

    boundaries, valid = _current_step_boundaries(env)
    if boundaries is None or valid is None or boundaries.shape[1] == 0:
      return empty

    if self._contact_sensor is not None:
      return self._true_contact_events_by_level(
        env, boundaries, valid, terrain_height_m, max_levels
      )

    return {
      "toe_riser_collision_by_level": (
        self._toe_events_by_level(env, boundaries, valid, terrain_height_m, max_levels)
        if self._toe is not None
        else empty["toe_riser_collision_by_level"]
      ),
      "heel_riser_collision_by_level": (
        self._heel_events_by_level(env, boundaries, valid, terrain_height_m, max_levels)
        if self._heel is not None
        else empty["heel_riser_collision_by_level"]
      ),
      "foot_lip_collision_by_level": (
        self._lip_events_by_level(env, boundaries, valid, terrain_height_m, max_levels)
        if self._lip is not None
        else empty["foot_lip_collision_by_level"]
      ),
    }

  def _true_contact_events_by_level(
    self,
    env: ManagerBasedRlEnv,
    boundaries: torch.Tensor,
    valid: torch.Tensor,
    terrain_height_m: float,
    max_levels: int,
  ) -> dict[str, torch.Tensor]:
    if self._last_true_contact_slots is None or self._last_true_contact_pos_w is None:
      return {
        name: torch.zeros(env.num_envs, max_levels, device=env.device)
        for name in LEVEL_EVENT_NAMES
      }

    levels = self._boundary_levels(boundaries, terrain_height_m, max_levels, valid)
    out = {
      "toe_riser_collision_by_level": self._true_contact_slots_to_levels(
        boundaries,
        valid,
        levels,
        self._last_true_contact_slots["toe"],
        self._last_true_contact_pos_w,
        max_levels,
      ),
      "heel_riser_collision_by_level": self._true_contact_slots_to_levels(
        boundaries,
        valid,
        levels,
        self._last_true_contact_slots["heel"],
        self._last_true_contact_pos_w,
        max_levels,
      ),
      "foot_lip_collision_by_level": torch.zeros(
        env.num_envs, max_levels, device=env.device
      ),
    }
    return out

  def _true_contact_slots_to_levels(
    self,
    boundaries: torch.Tensor,
    valid: torch.Tensor,
    levels: torch.Tensor,
    active_slots: torch.Tensor,
    contact_pos_w: torch.Tensor,
    max_levels: int,
  ) -> torch.Tensor:
    s, inside_face, _, _ = self._riser_geometry(boundaries, contact_pos_w)
    near_face = torch.abs(s) <= self.params.slab_depth
    active_boundary = (
      active_slots[:, :, :, None] & inside_face & near_face & valid[:, None, None, :]
    ).any(dim=(1, 2))
    return self._level_any(active_boundary, levels, valid, max_levels)

  def _boundary_levels(
    self,
    boundaries: torch.Tensor,
    terrain_height_m: float,
    max_levels: int,
    valid: torch.Tensor | None = None,
  ) -> torch.Tensor:
    z_low = boundaries[..., 9]
    z_high = boundaries[..., 10]
    z_min = torch.minimum(z_low, z_high)
    if valid is None:
      terrain_min = torch.min(z_min, dim=1, keepdim=True).values
    else:
      terrain_min = torch.min(
        torch.where(valid, z_min, torch.full_like(z_min, torch.inf)),
        dim=1,
        keepdim=True,
      ).values
      terrain_min = torch.where(
        torch.isfinite(terrain_min),
        terrain_min,
        torch.zeros_like(terrain_min),
      )
    levels = (z_min - terrain_min) / terrain_height_m + 1.0
    return torch.round(levels).long().clamp(1, max_levels)

  @staticmethod
  def _level_any(
    active_by_boundary: torch.Tensor,
    levels: torch.Tensor,
    valid: torch.Tensor,
    max_levels: int,
  ) -> torch.Tensor:
    out = torch.zeros(
      active_by_boundary.shape[0], max_levels, device=active_by_boundary.device
    )
    active_by_boundary = active_by_boundary & valid
    for level in range(1, max_levels + 1):
      level_mask = levels == level
      out[:, level - 1] = (active_by_boundary & level_mask).any(dim=1).float()
    return out

  def _riser_geometry(
    self,
    boundaries: torch.Tensor,
    points: torch.Tensor,
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

    rel = points[:, :, :, None, :] - center[:, None, None, :, :]
    s = torch.sum(rel * normal_to_low[:, None, None, :, :], dim=-1)
    u = torch.sum(rel * tangent_u[:, None, None, :, :], dim=-1)
    v = rel[..., 2]
    inside_face = (torch.abs(u) <= half_u[:, None, None, :] + self.params.u_margin) & (
      torch.abs(v) <= half_v[:, None, None, :] + self.params.v_margin
    )
    return s, inside_face, normal_to_low, rel

  def _toe_events_by_level(
    self,
    env: ManagerBasedRlEnv,
    boundaries: torch.Tensor,
    valid: torch.Tensor,
    terrain_height_m: float,
    max_levels: int,
  ) -> torch.Tensor:
    assert self._toe is not None
    points_w, point_vel_w = self._toe._foot_points_w(env, self.foot_asset_cfg)
    toe_mask = self._toe._local_x >= self.params.toe_x_min
    toe_points = points_w[:, :, toe_mask, :]
    toe_vel = point_vel_w[:, :, toe_mask, :]
    s, inside_face, normal_to_low, _ = self._riser_geometry(boundaries, toe_points)
    toe_speed_to_riser = -torch.sum(
      toe_vel[:, :, :, None, :] * normal_to_low[:, None, None, :, :], dim=-1
    )
    inside_slab = (s >= -self.params.surface_tol) & (s <= self.params.slab_depth)
    active = (
      valid[:, None, None, :] & inside_face & inside_slab & (toe_speed_to_riser > 0.0)
    )
    active_boundary = active.any(dim=(1, 2))
    levels = self._boundary_levels(boundaries, terrain_height_m, max_levels, valid)
    return self._level_any(active_boundary, levels, valid, max_levels)

  def _heel_events_by_level(
    self,
    env: ManagerBasedRlEnv,
    boundaries: torch.Tensor,
    valid: torch.Tensor,
    terrain_height_m: float,
    max_levels: int,
  ) -> torch.Tensor:
    assert self._heel is not None
    points_w, _ = self._heel._foot_points_w(env, self.foot_asset_cfg)
    heel_mask = self._heel._local_x <= self.params.heel_x_max
    heel_points = points_w[:, :, heel_mask, :]
    s, inside_face, _, _ = self._riser_geometry(boundaries, heel_points)
    inside_slab = (s >= -self.params.surface_tol) & (s <= self.params.heel_clearance)
    contact_gate = self._heel._foot_contact_gate(env, "feet_ground_contact", 2)
    active = (
      valid[:, None, None, :]
      & inside_face
      & inside_slab
      & (contact_gate[:, :, None, None] > 0.0)
    )
    active_boundary = active.any(dim=(1, 2))
    levels = self._boundary_levels(boundaries, terrain_height_m, max_levels, valid)
    return self._level_any(active_boundary, levels, valid, max_levels)

  def _lip_events_by_level(
    self,
    env: ManagerBasedRlEnv,
    boundaries: torch.Tensor,
    valid: torch.Tensor,
    terrain_height_m: float,
    max_levels: int,
  ) -> torch.Tensor:
    assert self._lip is not None
    points_w, _ = self._lip._foot_points_w(env, self.foot_asset_cfg)
    num_envs, num_feet = points_w.shape[:2]
    expanded_boundaries = boundaries[:, None, :, :].expand(num_envs, num_feet, -1, -1)
    p0 = expanded_boundaries[..., 0:3]
    p1 = expanded_boundaries[..., 3:6]
    z_high = expanded_boundaries[..., 10]
    distance = self._lip._point_to_segment_distance(points_w, p0, p1)
    height_ok = points_w[:, :, :, None, 2] >= z_high[:, :, None, :] - (
      self.params.edge_height_band
    )
    active = valid[:, None, None, :] & (distance <= self.params.edge_radius) & height_ok
    active_boundary = active.any(dim=(1, 2))
    levels = self._boundary_levels(boundaries, terrain_height_m, max_levels, valid)
    return self._level_any(active_boundary, levels, valid, max_levels)


def compute_velocity_metrics(
  env: ManagerBasedRlEnv,
  detector: StairEventDetector,
  *,
  terrain_height_m: float | None,
  max_levels: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
  """Compute scalar step metrics and per-level stair event metrics."""
  asset = env.scene["robot"]
  command = env.command_manager.get_command("twist")
  if command is None:
    raise RuntimeError("Velocity eval requires command 'twist'.")

  actual_lin = asset.data.root_link_lin_vel_b
  actual_ang = asset.data.root_link_ang_vel_b
  lin_error = torch.norm(command[:, :2] - actual_lin[:, :2], dim=-1)
  yaw_error = torch.abs(command[:, 2] - actual_ang[:, 2])
  tracking_error = torch.sqrt(torch.square(lin_error) + torch.square(yaw_error))
  projected_gravity = asset.data.projected_gravity_b
  base_pitch_roll = torch.norm(projected_gravity[:, :2], dim=-1)
  action_delta = env.action_manager.action - env.action_manager.prev_action
  action_smoothness = torch.mean(torch.square(action_delta), dim=-1)
  torque_cost = torch.mean(torch.square(asset.data.qfrc_actuator), dim=-1)

  foot_clearance = torch.zeros(env.num_envs, device=env.device)
  sensor = env.scene.sensors.get("foot_height_scan")
  if isinstance(sensor, TerrainHeightSensor):
    heights = sensor.data.heights
    if heights.ndim == 3:
      heights = heights.min(dim=-1).values
    foot_clearance = heights.mean(dim=-1)

  metrics = {
    "tracking_error": tracking_error,
    "tracking_lin_error": lin_error,
    "tracking_yaw_error": yaw_error,
    "base_pitch_roll_rms": base_pitch_roll,
    "action_smoothness": action_smoothness,
    "torque_cost": torque_cost,
    "foot_clearance": foot_clearance,
  }
  metrics.update(detector.compute_events(env))
  by_level = detector.compute_events_by_level(
    env, terrain_height_m=terrain_height_m, max_levels=max_levels
  )
  return metrics, by_level
