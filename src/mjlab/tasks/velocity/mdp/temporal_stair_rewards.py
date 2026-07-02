from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from .rewards import (
  _DEFAULT_FOOT_BODY_CFG,
  _DEFAULT_FOOT_SITE_CFG,
  _current_step_boundaries,
  _current_step_boundary_metadata,
  _step_boundary_layers,
  _StepBoundaryFootVolume,
  _terrain_level_active,
)
from .stair_geometry import (
  COLLISION_RISK_KEY,
  LANDING_QUALITY_KEY,
  LANDING_TOUCHDOWN_KEY,
  MINIMUM_SAFE_STRIDE_EXACT_KEY,
  MINIMUM_SAFE_STRIDE_INTERVAL_VALID_KEY,
  MINIMUM_SAFE_STRIDE_KEY,
  MINIMUM_SAFE_STRIDE_RAW_KEY,
  MINIMUM_SAFE_STRIDE_UPPER_KEY,
  MINIMUM_SAFE_STRIDE_VALID_KEY,
  MINIMUM_SAFE_STRIDE_WEIGHT_KEY,
  OBSERVED_STEP_STRIDE_KEY,
  SAFE_LANDING_CENTER_KEY,
  SAFE_STRIDE_VALID_KEY,
  SAFE_TREAD_LOWER_BOUND_KEY,
  STAIR_ASCENT_DIR_KEY,
  STAIR_CLEARANCE_ASCENT_DIR_KEY,
  STAIR_CLEARANCE_FOOT_LAYERS_KEY,
  STAIR_CLEARANCE_FOOT_LAYERS_VALID_KEY,
  STAIR_CLEARANCE_SEQUENCE_ID_KEY,
  STAIR_DEPTH_CONFIRMATION_EVENT_KEY,
  STAIR_DEPTH_LABEL_VALID_KEY,
  STAIR_ENTRY_EVENT_KEY,
  STAIR_ENTRY_EVIDENCE_ASCENT_DIR_KEY,
  STAIR_ENTRY_RECENT_EVIDENCE_KEY,
  STAIR_EXIT_EVENT_KEY,
  STAIR_EXPECTED_LAYER_KEY,
  STAIR_LANDING_EXPECTED_LAYER_KEY,
  STAIR_LANDING_SEQUENCE_ID_KEY,
  STAIR_LANDING_TARGET_FOOT_KEY,
  STAIR_PHASE_KEY,
  STAIR_RISER_HEIGHT_LABEL_KEY,
  STAIR_SHAPE_LABEL_VALID_KEY,
  STAIR_SKIP_LAYER_PENALTY_KEY,
  STAIR_TARGET_FOOT_KEY,
  STAIR_TREAD_DEPTH_LABEL_KEY,
  TOE_RISER_CONTACT_KEY,
  TOE_RISER_NEW_HIT_KEY,
  cached_stair_shape,
  stair_shape_for_sequence,
)


def _proximity_risk(clearance: torch.Tensor, safe_margin: float) -> torch.Tensor:
  return torch.clamp(
    (safe_margin - clearance) / max(safe_margin, 1.0e-6), min=0.0, max=1.0
  )


def _landing_quality(
  coverage: torch.Tensor,
  center_score: torch.Tensor,
  edge_clearance_score: torch.Tensor,
  valid: torch.Tensor,
) -> torch.Tensor:
  secondary = 0.5 * center_score + 0.5 * edge_clearance_score
  quality = coverage.clamp(0.0, 1.0) * (0.8 + 0.2 * secondary)
  return quality * valid.float()


def _strict_stair_entry_event(
  stair_phase: torch.Tensor,
  layer1_new_hit: torch.Tensor,
) -> torch.Tensor:
  """Start stair context on the first sequence-local riser hit."""
  return (stair_phase == 0) & layer1_new_hit


def _repeat_sequence_layer1_hit(
  stair_phase: torch.Tensor,
  boundary_contact: torch.Tensor,
  contact_layers: torch.Tensor,
  new_hit_by_foot: torch.Tensor,
  contact_sequence_ids: torch.Tensor,
  sequence_id: torch.Tensor,
) -> torch.Tensor:
  """Detect post-entry layer-1 hits on the active stair sequence."""
  repeat_contact = (
    (stair_phase[:, None, None] >= 1)
    & boundary_contact
    & (contact_layers == 1)
    & new_hit_by_foot[:, :, None]
    & (contact_sequence_ids == sequence_id[:, None, None])
  )
  return torch.any(repeat_contact, dim=(1, 2))


def _minimum_safe_stride_to_layer(
  boundaries: torch.Tensor,
  valid_boundaries: torch.Tensor,
  boundary_sequence_ids: torch.Tensor,
  boundary_layers: torch.Tensor,
  sequence_id: torch.Tensor,
  target_layer: torch.Tensor,
  target_ref_w: torch.Tensor,
  target_sole_points_w: torch.Tensor,
  ascent_dir: torch.Tensor,
  tread_depth: torch.Tensor,
  rear_clearance: float,
  front_clearance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return the forward-translation interval and per-bound observability."""
  boundary_ascent = -toe_step_riser_slab_penalty._normalize_xy(boundaries[..., 6:8])
  target_boundary = (
    valid_boundaries
    & (boundary_sequence_ids == sequence_id[:, None])
    & (boundary_layers == target_layer[:, None])
    & (torch.sum(boundary_ascent * ascent_dir[:, None, :], dim=-1) >= 0.90)
  )
  has_target = torch.any(target_boundary, dim=-1)
  target_idx = torch.argmax(target_boundary.long(), dim=-1)
  env_ids = torch.arange(boundaries.shape[0], device=boundaries.device)
  target_p0_xy = boundaries[env_ids, target_idx, 0:2]
  target_p1_xy = boundaries[env_ids, target_idx, 3:5]
  target_edge = target_p1_xy - target_p0_xy
  target_edge_len = torch.norm(target_edge, dim=-1).clamp_min(1.0e-6)
  target_tangent = target_edge / target_edge_len[:, None]

  target_ref_xy = target_ref_w[:, 0:2]
  sole_rel_p0 = target_sole_points_w[..., 0:2] - target_p0_xy[:, None, :]
  sole_lateral = torch.sum(sole_rel_p0 * target_tangent[:, None, :], dim=-1)
  lateral_valid = torch.all(
    (sole_lateral >= 0.0) & (sole_lateral <= target_edge_len[:, None]),
    dim=-1,
  )
  point_offsets_s = torch.sum(
    (target_sole_points_w[..., 0:2] - target_ref_xy[:, None, :])
    * ascent_dir[:, None, :],
    dim=-1,
  )

  rear = tread_depth.new_tensor(rear_clearance)
  front = tread_depth.new_tensor(front_clearance)
  min_point_offset_s = torch.amin(point_offsets_s, dim=-1)
  max_point_offset_s = torch.amax(point_offsets_s, dim=-1)
  minimum_landing_s = rear - min_point_offset_s
  maximum_landing_s = tread_depth - front - max_point_offset_s
  target_start_s = torch.sum(
    (target_ref_xy - target_p0_xy) * ascent_dir,
    dim=-1,
  )
  minimum_safe_stride = minimum_landing_s - target_start_s
  maximum_safe_stride = maximum_landing_s - target_start_s
  lower_valid = has_target & lateral_valid & torch.isfinite(minimum_safe_stride)
  interval_valid = (
    lower_valid
    & torch.isfinite(maximum_safe_stride)
    & (maximum_safe_stride >= minimum_safe_stride)
    & (maximum_safe_stride > 0.0)
  )
  return (
    torch.where(lower_valid, minimum_safe_stride, 0.0),
    torch.where(interval_valid, maximum_safe_stride, 0.0),
    lower_valid,
    interval_valid,
  )


def _stair_touchdown_progress(
  touchdown_layer: torch.Tensor,
  support_layer: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return layer delta, normal-step mask, and number of skipped layers."""
  delta = touchdown_layer - support_layer
  normal_step = delta == 1
  skipped_layers = torch.clamp(delta - 1, min=0)
  return delta, normal_step, skipped_layers


def _advance_stair_attempt(
  target_foot: torch.Tensor,
  reached_layer: torch.Tensor,
  expected_layer: torch.Tensor,
  expected_tread_contact: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Alternate the target foot while advancing only a contacted expected tread."""
  next_reached_layer = torch.where(
    expected_tread_contact,
    expected_layer,
    reached_layer,
  )
  return 1 - target_foot, next_reached_layer, next_reached_layer + 1


def _stair_tread_overlap_contact(
  active: torch.Tensor,
  ground_contact: torch.Tensor,
  has_support_candidate: torch.Tensor,
  support_fraction: torch.Tensor,
  support_layer: torch.Tensor,
) -> torch.Tensor:
  """Return tread contact suitable for per-leg clearance layer tracking."""
  return (
    active[:, None]
    & ground_contact
    & has_support_candidate
    & (support_fraction > 0.0)
    & (support_layer > 0)
  )


def _safe_stride_tracking_masks(
  geometry_valid: torch.Tensor,
  raw_stride: torch.Tensor,
  max_tracking_stride: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Separate trainable stride targets from stale, unreachable sequence state."""
  stale = geometry_valid & (raw_stride > max_tracking_stride)
  return geometry_valid & ~stale, stale


def _bounded_safe_stride_interval(
  raw_lower: torch.Tensor,
  raw_upper: torch.Tensor,
  lower_geometry_valid: torch.Tensor,
  interval_geometry_valid: torch.Tensor,
  minimum_stride: float,
  maximum_stride: float,
  maximum_tracking_stride: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Clamp bounds while keeping lower and full-interval validity separate."""
  lower_tracking_valid, stale = _safe_stride_tracking_masks(
    lower_geometry_valid,
    raw_lower,
    maximum_tracking_stride,
  )
  lower = torch.clamp(raw_lower, min=minimum_stride, max=maximum_stride)
  upper = torch.clamp(raw_upper, min=minimum_stride, max=maximum_stride)
  interval_tracking_valid = (
    lower_tracking_valid
    & interval_geometry_valid
    & (raw_upper >= minimum_stride)
    & (upper >= lower)
  )
  return lower, upper, lower_tracking_valid, interval_tracking_valid, stale


def _confirmed_stair_depth_evidence(
  active: torch.Tensor,
  shape_valid: torch.Tensor,
  expected_layer: torch.Tensor,
  expected_riser_evidence: torch.Tensor,
) -> torch.Tensor:
  """Confirm tread depth only after observing the next expected riser."""
  return active & shape_valid & (expected_layer >= 2) & expected_riser_evidence


def _blocked_swing_event_evidence(
  vertical_boundary_contact: torch.Tensor,
  previous_forward_speed: torch.Tensor,
  current_forward_speed: torch.Tensor,
  contact_steps: torch.Tensor,
  command_forward: torch.Tensor,
  min_forward_intent_speed: float,
  min_forward_speed_drop: float,
  blocked_forward_speed: float,
  persistent_steps: int,
  command_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Detect deployable abrupt and persistent swing blocking evidence."""
  speed_drop = previous_forward_speed - current_forward_speed
  abrupt = (
    vertical_boundary_contact
    & (previous_forward_speed >= min_forward_intent_speed)
    & (speed_drop >= min_forward_speed_drop)
  )
  persistent = (
    vertical_boundary_contact
    & (contact_steps >= persistent_steps)
    & (current_forward_speed <= blocked_forward_speed)
    & (command_forward[:, None] > command_threshold)
  )
  return abrupt, persistent, abrupt | persistent


class toe_step_riser_slab_penalty(_StepBoundaryFootVolume):
  """Penalty-only riser slab term plus safe stair-entry state tracking.

  The first sequence-local layer-1 contact starts stair tracking. The opposite
  foot then probes layer 1. Every completed swing attempt alternates the target
  foot, independently of landing quality. The reached layer advances only when
  the attempted foot contacts the expected tread. Privileged geometry remains
  training-only and never commands the actor.
  """

  def __init__(self, cfg: RewardTermCfg, env) -> None:
    super().__init__(cfg, env)
    asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_FOOT_BODY_CFG)
    num_feet = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else 2
    self._contact_cooldown = torch.zeros(
      env.num_envs, num_feet, device=env.device, dtype=torch.float32
    )
    self._blocked_contact_steps = torch.zeros(
      env.num_envs, num_feet, device=env.device, dtype=torch.long
    )
    self._previous_foot_forward_speed = torch.zeros(
      env.num_envs, num_feet, device=env.device, dtype=torch.float32
    )
    self._stair_phase = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._entry_event = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self._exit_event = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self._toe_riser_new_hit = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    self._toe_riser_contact = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    self._entry_evidence_timer = torch.zeros(env.num_envs, device=env.device)
    self._recent_entry_evidence = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    self._entry_evidence_ascent_dir = torch.zeros(
      env.num_envs, 2, device=env.device, dtype=torch.float32
    )
    self._first_foot = torch.full(
      (env.num_envs,), -1, device=env.device, dtype=torch.long
    )
    self._target_foot = torch.full(
      (env.num_envs,), -1, device=env.device, dtype=torch.long
    )
    self._support_foot = torch.full(
      (env.num_envs,), -1, device=env.device, dtype=torch.long
    )
    self._support_layer = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._expected_layer = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )
    self._landing_expected_layer = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )
    self._landing_sequence_id = torch.full(
      (env.num_envs,), -1, device=env.device, dtype=torch.long
    )
    self._landing_target_foot = torch.full(
      (env.num_envs,), -1, device=env.device, dtype=torch.long
    )
    self._foot_support_layers = torch.zeros(
      env.num_envs, num_feet, device=env.device, dtype=torch.long
    )
    self._foot_support_layers_valid = torch.zeros(
      env.num_envs, num_feet, device=env.device, dtype=torch.bool
    )
    self._clearance_foot_layers = torch.zeros_like(self._foot_support_layers)
    self._clearance_foot_layers_valid = torch.zeros_like(
      self._foot_support_layers_valid
    )
    self._clearance_sequence_id = torch.full(
      (env.num_envs,), -1, device=env.device, dtype=torch.long
    )
    self._clearance_ascent_dir = torch.zeros(
      env.num_envs, 2, device=env.device, dtype=torch.float32
    )
    self._target_start_pos_w = torch.zeros(
      env.num_envs, 3, device=env.device, dtype=torch.float32
    )
    self._ascent_dir = torch.zeros(
      env.num_envs, 2, device=env.device, dtype=torch.float32
    )
    self._first_boundary_idx = torch.full(
      (env.num_envs,), -1, device=env.device, dtype=torch.long
    )
    self._sequence_id = torch.full(
      (env.num_envs,), -1, device=env.device, dtype=torch.long
    )
    self._context_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._attempt_elapsed_time = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
    self._attempt_air_seen = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    self._attempt_expected_tread_contact = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    self._attempt_expected_tread_full_support = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    self._attempt_expected_riser_contact = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    self._attempt_rear_partial_contact = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    self._attempt_touchdown_layer = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )
    self._repeat_layer1_hit_count = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )
    self._stair_tread_depth_label = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
    self._stair_riser_height_label = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
    self._stair_shape_label_valid = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    self._stair_depth_label_valid = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    self._stair_depth_confirmation_event = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    self._layer1_entry_candidate_total = torch.zeros((), device=env.device)
    self._entry_event_total = torch.zeros((), device=env.device)
    self._safe_stride_valid = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    self._safe_tread_lower_bound = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
    self._minimum_safe_stride = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
    self._minimum_safe_stride_upper = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
    self._minimum_safe_stride_raw = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
    self._minimum_safe_stride_geometry_valid = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    self._minimum_safe_stride_interval_geometry_valid = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    self._minimum_safe_stride_valid = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    self._minimum_safe_stride_interval_valid = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    self._minimum_safe_stride_exact = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    self._minimum_safe_stride_weight = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
    self._safe_stride_evidence_steps = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )
    self._probe_evidence_seen = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    self._stair_skip_layer_penalty = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
    self._safe_landing_center_s = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
    self._observed_step_stride = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
    self._exit_flat_steps = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )
    self._collision_risk_now = torch.zeros(env.num_envs, device=env.device)
    self._landing_touchdown_now = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    self._landing_quality_now = torch.zeros(env.num_envs, device=env.device)
    env.extras[STAIR_PHASE_KEY] = self._stair_phase
    env.extras[STAIR_ENTRY_EVENT_KEY] = self._entry_event
    env.extras[STAIR_EXIT_EVENT_KEY] = self._exit_event
    env.extras[TOE_RISER_NEW_HIT_KEY] = self._toe_riser_new_hit
    env.extras[TOE_RISER_CONTACT_KEY] = self._toe_riser_contact
    env.extras[STAIR_ENTRY_RECENT_EVIDENCE_KEY] = self._recent_entry_evidence
    env.extras[STAIR_ENTRY_EVIDENCE_ASCENT_DIR_KEY] = self._entry_evidence_ascent_dir
    env.extras[STAIR_TARGET_FOOT_KEY] = self._target_foot
    env.extras[STAIR_EXPECTED_LAYER_KEY] = self._expected_layer
    env.extras[STAIR_LANDING_EXPECTED_LAYER_KEY] = self._landing_expected_layer
    env.extras[STAIR_LANDING_SEQUENCE_ID_KEY] = self._landing_sequence_id
    env.extras[STAIR_LANDING_TARGET_FOOT_KEY] = self._landing_target_foot
    env.extras[STAIR_CLEARANCE_FOOT_LAYERS_KEY] = self._clearance_foot_layers
    env.extras[STAIR_CLEARANCE_FOOT_LAYERS_VALID_KEY] = (
      self._clearance_foot_layers_valid
    )
    env.extras[STAIR_CLEARANCE_SEQUENCE_ID_KEY] = self._clearance_sequence_id
    env.extras[STAIR_CLEARANCE_ASCENT_DIR_KEY] = self._clearance_ascent_dir
    env.extras[STAIR_ASCENT_DIR_KEY] = self._ascent_dir
    env.extras[STAIR_TREAD_DEPTH_LABEL_KEY] = self._stair_tread_depth_label
    env.extras[STAIR_RISER_HEIGHT_LABEL_KEY] = self._stair_riser_height_label
    env.extras[STAIR_SHAPE_LABEL_VALID_KEY] = self._stair_shape_label_valid
    env.extras[STAIR_DEPTH_LABEL_VALID_KEY] = self._stair_depth_label_valid
    env.extras[STAIR_DEPTH_CONFIRMATION_EVENT_KEY] = (
      self._stair_depth_confirmation_event
    )
    env.extras[SAFE_STRIDE_VALID_KEY] = self._safe_stride_valid
    env.extras[SAFE_TREAD_LOWER_BOUND_KEY] = self._safe_tread_lower_bound
    env.extras[MINIMUM_SAFE_STRIDE_KEY] = self._minimum_safe_stride
    env.extras[MINIMUM_SAFE_STRIDE_UPPER_KEY] = self._minimum_safe_stride_upper
    env.extras[MINIMUM_SAFE_STRIDE_RAW_KEY] = self._minimum_safe_stride_raw
    env.extras[MINIMUM_SAFE_STRIDE_VALID_KEY] = self._minimum_safe_stride_valid
    env.extras[MINIMUM_SAFE_STRIDE_INTERVAL_VALID_KEY] = (
      self._minimum_safe_stride_interval_valid
    )
    env.extras[MINIMUM_SAFE_STRIDE_EXACT_KEY] = self._minimum_safe_stride_exact
    env.extras[MINIMUM_SAFE_STRIDE_WEIGHT_KEY] = self._minimum_safe_stride_weight
    env.extras[STAIR_SKIP_LAYER_PENALTY_KEY] = self._stair_skip_layer_penalty
    env.extras[SAFE_LANDING_CENTER_KEY] = self._safe_landing_center_s
    env.extras[OBSERVED_STEP_STRIDE_KEY] = self._observed_step_stride
    env.extras[COLLISION_RISK_KEY] = self._collision_risk_now
    env.extras[LANDING_TOUCHDOWN_KEY] = self._landing_touchdown_now
    env.extras[LANDING_QUALITY_KEY] = self._landing_quality_now

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._contact_cooldown[env_ids] = 0.0
    self._blocked_contact_steps[env_ids] = 0
    self._previous_foot_forward_speed[env_ids] = 0.0
    self._stair_phase[env_ids] = 0
    self._entry_event[env_ids] = False
    self._exit_event[env_ids] = False
    self._toe_riser_new_hit[env_ids] = False
    self._toe_riser_contact[env_ids] = False
    self._entry_evidence_timer[env_ids] = 0.0
    self._recent_entry_evidence[env_ids] = False
    self._entry_evidence_ascent_dir[env_ids] = 0.0
    self._first_foot[env_ids] = -1
    self._target_foot[env_ids] = -1
    self._support_foot[env_ids] = -1
    self._support_layer[env_ids] = 0
    self._expected_layer[env_ids] = 0
    self._landing_expected_layer[env_ids] = 0
    self._landing_sequence_id[env_ids] = -1
    self._landing_target_foot[env_ids] = -1
    self._foot_support_layers[env_ids] = 0
    self._foot_support_layers_valid[env_ids] = False
    self._clearance_foot_layers[env_ids] = 0
    self._clearance_foot_layers_valid[env_ids] = False
    self._clearance_sequence_id[env_ids] = -1
    self._clearance_ascent_dir[env_ids] = 0.0
    self._target_start_pos_w[env_ids] = 0.0
    self._ascent_dir[env_ids] = 0.0
    self._first_boundary_idx[env_ids] = -1
    self._sequence_id[env_ids] = -1
    self._context_steps[env_ids] = 0
    self._attempt_elapsed_time[env_ids] = 0.0
    self._attempt_air_seen[env_ids] = False
    self._attempt_expected_tread_contact[env_ids] = False
    self._attempt_expected_tread_full_support[env_ids] = False
    self._attempt_expected_riser_contact[env_ids] = False
    self._attempt_rear_partial_contact[env_ids] = False
    self._attempt_touchdown_layer[env_ids] = 0
    self._repeat_layer1_hit_count[env_ids] = 0
    self._stair_tread_depth_label[env_ids] = 0.0
    self._stair_riser_height_label[env_ids] = 0.0
    self._stair_shape_label_valid[env_ids] = False
    self._stair_depth_label_valid[env_ids] = False
    self._stair_depth_confirmation_event[env_ids] = False
    self._safe_stride_valid[env_ids] = False
    self._safe_tread_lower_bound[env_ids] = 0.0
    self._minimum_safe_stride[env_ids] = 0.0
    self._minimum_safe_stride_upper[env_ids] = 0.0
    self._minimum_safe_stride_raw[env_ids] = 0.0
    self._minimum_safe_stride_geometry_valid[env_ids] = False
    self._minimum_safe_stride_interval_geometry_valid[env_ids] = False
    self._minimum_safe_stride_valid[env_ids] = False
    self._minimum_safe_stride_interval_valid[env_ids] = False
    self._minimum_safe_stride_exact[env_ids] = False
    self._minimum_safe_stride_weight[env_ids] = 0.0
    self._safe_stride_evidence_steps[env_ids] = 0
    self._probe_evidence_seen[env_ids] = False
    self._stair_skip_layer_penalty[env_ids] = 0.0
    self._safe_landing_center_s[env_ids] = 0.0
    self._observed_step_stride[env_ids] = 0.0
    self._exit_flat_steps[env_ids] = 0
    self._collision_risk_now[env_ids] = 0.0
    self._landing_touchdown_now[env_ids] = False
    self._landing_quality_now[env_ids] = 0.0

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
  def _base_heading_cos(
    root_quat_w: torch.Tensor,
    ascent_dir: torch.Tensor,
  ) -> torch.Tensor:
    forward_b = torch.zeros(
      root_quat_w.shape[0], 3, device=root_quat_w.device, dtype=root_quat_w.dtype
    )
    forward_b[:, 0] = 1.0
    forward_w = quat_apply(root_quat_w, forward_b)[:, :2]
    forward_w = toe_step_riser_slab_penalty._normalize_xy(forward_w)
    ascent_norm = torch.norm(ascent_dir, dim=-1, keepdim=True)
    ascent_unit = ascent_dir / ascent_norm.clamp_min(1.0e-6)
    heading_cos = torch.sum(forward_w * ascent_unit, dim=-1)
    return torch.where(
      ascent_norm.squeeze(-1) > 1.0e-6,
      heading_cos,
      torch.zeros_like(heading_cos),
    )

  @staticmethod
  def _advance_flat_exit_confirmation(
    active_phase: torch.Tensor,
    exit_candidate: torch.Tensor,
    touchdown_now: torch.Tensor,
    unsafe_now: torch.Tensor,
    flat_steps: torch.Tensor,
    required_steps: int = 2,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Latch stair context until enough safe flat touchdown events confirm exit."""
    reset_evidence = (~active_phase) | (~exit_candidate) | unsafe_now
    next_flat_steps = torch.where(
      reset_evidence,
      torch.zeros_like(flat_steps),
      flat_steps,
    )
    flat_exit_step = exit_candidate & touchdown_now & ~unsafe_now
    next_flat_steps = torch.where(
      flat_exit_step,
      next_flat_steps + 1,
      next_flat_steps,
    )
    confirmed_exit = active_phase & (next_flat_steps >= required_steps)
    return next_flat_steps, confirmed_exit

  @staticmethod
  def _tread_support_fraction(
    sole_points_w: torch.Tensor,
    boundaries: torch.Tensor,
    tread_depth: torch.Tensor,
  ) -> torch.Tensor:
    """Estimate the fraction of sole samples supported by each upper tread."""
    p0_xy = boundaries[:, None, None, :, 0:2]
    p1_xy = boundaries[:, None, None, :, 3:5]
    edge_xy = p1_xy - p0_xy
    edge_len = torch.norm(edge_xy, dim=-1).clamp_min(1.0e-6)
    tangent = edge_xy / edge_len[..., None]
    normal_to_low = toe_step_riser_slab_penalty._normalize_xy(
      boundaries[:, None, None, :, 6:8]
    )
    rel_xy = sole_points_w[:, :, :, None, :2] - p0_xy
    lateral = torch.sum(rel_xy * tangent, dim=-1)
    tread_s = torch.sum(rel_xy * -normal_to_low, dim=-1)
    depth = tread_depth[:, None, None, None]
    supported = (
      (lateral >= 0.0) & (lateral <= edge_len) & (tread_s >= 0.0) & (tread_s <= depth)
    )
    return supported.float().mean(dim=2)

  @staticmethod
  def _contact_boundary_layers(
    contact_pos_w: torch.Tensor,
    boundaries: torch.Tensor,
    boundary_layers: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    p0 = boundaries[:, :, 0:3]
    p1 = boundaries[:, :, 3:6]
    segment = p1 - p0
    segment_len_sq = torch.sum(torch.square(segment), dim=-1).clamp_min(1.0e-12)
    point_delta = contact_pos_w[:, :, :, None, :] - p0[:, None, None, :, :]
    t = torch.sum(point_delta * segment[:, None, None, :, :], dim=-1)
    t = torch.clamp(t / segment_len_sq[:, None, None, :], 0.0, 1.0)
    closest = p0[:, None, None, :, :] + t[..., None] * segment[:, None, None, :, :]
    distances = torch.norm(contact_pos_w[:, :, :, None, :] - closest, dim=-1)
    distances = torch.where(
      boundary_layers[:, None, None, :] > 0,
      distances,
      torch.full_like(distances, torch.inf),
    )
    nearest_dist, nearest_idx = torch.min(distances, dim=-1)
    expanded_layers = boundary_layers[:, None, None, :].expand(
      *nearest_idx.shape, boundary_layers.shape[-1]
    )
    nearest_layers = torch.gather(
      expanded_layers, dim=-1, index=nearest_idx[..., None]
    ).squeeze(-1)
    nearest_layers = torch.where(
      torch.isfinite(nearest_dist), nearest_layers, torch.zeros_like(nearest_layers)
    )
    return nearest_layers, nearest_idx

  def _riser_contact_terms(
    self,
    env,
    sensor_name: str,
    asset: Entity,
    asset_cfg: SceneEntityCfg,
    level_active: torch.Tensor,
    boundaries: torch.Tensor,
    boundary_layers: torch.Tensor,
    toe_velocity_w: torch.Tensor,
    command_forward: torch.Tensor,
    toe_x_min: float,
    vertical_normal_z_max: float,
    force_threshold: float,
    force_scale: float,
    contact_penalty_scale: float,
    contact_time_scale: float,
    entry_cooldown_time: float,
    event_min_forward_intent_speed: float,
    event_min_forward_speed_drop: float,
    event_blocked_forward_speed: float,
    event_persistent_steps: int,
    event_command_threshold: float,
  ) -> dict[str, torch.Tensor]:
    sensor = env.scene[sensor_name]
    assert isinstance(sensor, ContactSensor), (
      f"toe_step_riser_slab_penalty requires ContactSensor {sensor_name!r}."
    )
    data = sensor.data
    assert data.found is not None
    assert data.force is not None
    assert data.normal is not None
    assert data.pos is not None

    num_envs, num_feet = self._contact_cooldown.shape
    num_contacts = data.found.shape[1]
    if num_contacts % num_feet != 0:
      raise RuntimeError(
        f"Contact sensor {sensor_name!r} has {num_contacts} slots for {num_feet} feet."
      )
    num_slots = num_contacts // num_feet
    found = data.found.view(num_envs, num_feet, num_slots) > 0
    force_w = data.force.view(num_envs, num_feet, num_slots, 3)
    normal_w = data.normal.view(num_envs, num_feet, num_slots, 3)
    contact_pos_w = data.pos.view(num_envs, num_feet, num_slots, 3)

    foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    foot_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    expanded_quat = foot_quat_w[:, :, None, :].expand(num_envs, num_feet, num_slots, 4)
    contact_pos_b = quat_apply_inverse(
      expanded_quat, contact_pos_w - foot_pos_w[:, :, None, :]
    )
    is_toe_contact = contact_pos_b[..., 0] >= toe_x_min
    is_vertical_face = torch.abs(normal_w[..., 2]) < vertical_normal_z_max

    local_forward = torch.zeros(
      num_envs, num_feet, 3, device=env.device, dtype=foot_quat_w.dtype
    )
    local_forward[..., 0] = 1.0
    foot_forward_xy = self._normalize_xy(
      quat_apply(foot_quat_w, local_forward)[..., :2]
    )
    blocking_force = torch.relu(
      -torch.sum(force_w[..., :2] * foot_forward_xy[:, :, None, :], dim=-1)
    )
    hit_strength = torch.clamp(
      (blocking_force - force_threshold) / max(force_scale, 1.0e-6),
      min=0.0,
      max=1.0,
    )
    vertical_riser_contact = found & is_toe_contact & is_vertical_face
    riser_hit = vertical_riser_contact & (hit_strength > 0.0)
    active_hit = riser_hit & level_active[:, None, None]
    active_riser_contact = vertical_riser_contact & level_active[:, None, None]
    hit_by_foot = torch.any(active_hit, dim=-1)

    contact_layers, contact_boundary_idx = self._contact_boundary_layers(
      contact_pos_w, boundaries, boundary_layers
    )
    boundary_contact = active_hit & (contact_layers > 0)
    event_boundary_contact = active_riser_contact & (contact_layers > 0)
    layer1_event_boundary_contact = event_boundary_contact & (contact_layers == 1)
    vertical_boundary_contact_by_foot = torch.any(event_boundary_contact, dim=-1)
    layer1_vertical_boundary_contact_by_foot = torch.any(
      layer1_event_boundary_contact, dim=-1
    )
    self._blocked_contact_steps = torch.where(
      vertical_boundary_contact_by_foot,
      self._blocked_contact_steps + 1,
      torch.zeros_like(self._blocked_contact_steps),
    )

    base_forward_b = torch.zeros(
      num_envs, 3, device=env.device, dtype=foot_quat_w.dtype
    )
    base_forward_b[:, 0] = 1.0
    base_forward_w = self._normalize_xy(
      quat_apply(asset.data.root_link_quat_w, base_forward_b)[:, :2]
    )
    relative_toe_velocity_w = (
      toe_velocity_w - asset.data.root_link_lin_vel_w[:, None, :]
    )
    foot_forward_speed = torch.sum(
      relative_toe_velocity_w[..., :2] * base_forward_w[:, None, :],
      dim=-1,
    )
    abrupt_block, persistent_block, blocked_event_by_foot = (
      _blocked_swing_event_evidence(
        vertical_boundary_contact_by_foot,
        self._previous_foot_forward_speed,
        foot_forward_speed,
        self._blocked_contact_steps,
        command_forward,
        event_min_forward_intent_speed,
        event_min_forward_speed_drop,
        event_blocked_forward_speed,
        event_persistent_steps,
        event_command_threshold,
      )
    )
    self._previous_foot_forward_speed.copy_(foot_forward_speed)

    self._contact_cooldown = torch.clamp(self._contact_cooldown - env.step_dt, min=0.0)
    new_hit_by_foot = blocked_event_by_foot & (self._contact_cooldown <= 0.0)
    self._contact_cooldown = torch.where(
      new_hit_by_foot,
      torch.full_like(self._contact_cooldown, entry_cooldown_time),
      self._contact_cooldown,
    )

    per_foot_strength = torch.max(
      torch.where(active_hit, hit_strength, torch.zeros_like(hit_strength)), dim=-1
    ).values
    env_strength = torch.max(per_foot_strength, dim=-1).values
    hit_env = torch.any(hit_by_foot, dim=-1)

    contact_time = data.current_contact_time
    if contact_time is None:
      contact_time = hit_by_foot.float() * env.step_dt
    elif contact_time.shape[1] != num_feet:
      if contact_time.shape[1] % num_feet != 0:
        raise RuntimeError(
          f"Contact times from {sensor_name!r} cannot be grouped by foot."
        )
      contact_time = contact_time.view(num_envs, num_feet, -1).max(dim=-1).values
    env_contact_time = torch.max(
      torch.where(hit_by_foot, contact_time, torch.zeros_like(contact_time)), dim=-1
    ).values
    contact_time_weight = torch.clamp(
      env_contact_time / max(contact_time_scale, 1.0e-6), min=0.0, max=1.0
    )
    contact_penalty = (
      hit_env.float()
      * env_strength
      * (1.0 + contact_time_weight)
      * contact_penalty_scale
    )

    log = env.extras["log"]
    log["Metrics/toe_riser_contact_active_ratio"] = hit_env.float().mean()
    log["Metrics/toe_riser_contact_new_hit_ratio"] = (
      torch.any(new_hit_by_foot, dim=-1).float().mean()
    )
    log["Metrics/toe_riser_layer1_vertical_contact_ratio"] = (
      layer1_vertical_boundary_contact_by_foot.float().mean()
    )
    log["Metrics/toe_riser_abrupt_block_ratio"] = abrupt_block.float().mean()
    log["Metrics/toe_riser_persistent_block_ratio"] = persistent_block.float().mean()
    log["Metrics/toe_riser_force_gt_threshold_ratio"] = hit_by_foot.float().mean()
    log["Metrics/toe_riser_contact_penalty_mean"] = contact_penalty.mean()
    log["Metrics/toe_riser_contact_time_mean"] = env_contact_time.mean()
    return {
      "contact_penalty": contact_penalty,
      "new_hit_by_foot": new_hit_by_foot,
      "riser_contact": torch.any(active_riser_contact, dim=-1),
      "boundary_contact": boundary_contact,
      "soft_boundary_contact": event_boundary_contact,
      "blocked_riser_by_foot": blocked_event_by_foot,
      "event_boundary_contact": layer1_event_boundary_contact,
      "contact_layers": contact_layers,
      "contact_boundary_idx": contact_boundary_idx,
      "hit_strength": hit_strength,
      "foot_pos_w": foot_pos_w,
    }

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
    event_min_forward_intent_speed: float = 0.04,
    event_min_forward_speed_drop: float = 0.04,
    event_blocked_forward_speed: float = 0.03,
    event_persistent_steps: int = 3,
    stair_entry_cooldown_time: float = 0.20,
    stair_heading_cos: float = 0.70,
    stair_touchdown_height_tolerance: float = 0.08,
    stair_touchdown_lateral_margin: float = 0.03,
    stair_min_safe_stride: float = 0.10,
    stair_max_safe_stride: float = 0.55,
    stair_max_tracking_stride: float = 0.80,
    stair_touchdown_lip_clearance: float = 0.02,
    stair_touchdown_lip_height_band: float = 0.06,
    safe_stride_containment_margin: float = 0.003,
    stair_entry_evidence_time: float = 0.80,
    stair_attempt_period: float = 0.60,
    safe_stride_evidence_window_steps: int = 5,
    safe_stride_rear_partial_weight: float = 2.0,
    safe_stride_riser_weight: float = 3.0,
    collision_risk_margin: float = 0.06,
    command_name: str = "twist",
    command_threshold: float = 0.05,
    ground_contact_sensor_name: str = "feet_ground_contact",
    asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_BODY_CFG,
    **_unused: object,
  ) -> torch.Tensor:
    if stair_attempt_period <= 0.0:
      raise ValueError("stair_attempt_period must be positive.")
    if event_persistent_steps < 1:
      raise ValueError("event_persistent_steps must be at least one.")
    if safe_stride_evidence_window_steps < 1:
      raise ValueError("safe_stride_evidence_window_steps must be at least one.")
    if safe_stride_containment_margin < 0.0:
      raise ValueError("safe_stride_containment_margin must be non-negative.")
    if safe_stride_rear_partial_weight < 1.0 or safe_stride_riser_weight < 1.0:
      raise ValueError("Safe-stride evidence weights must be at least one.")
    if stair_max_tracking_stride < stair_max_safe_stride:
      raise ValueError(
        "stair_max_tracking_stride must be at least stair_max_safe_stride."
      )
    legacy_evidence_time = _unused.get("toe_riser_evidence_time")
    if isinstance(legacy_evidence_time, int | float):
      stair_entry_evidence_time = float(legacy_evidence_time)
    self._entry_event.zero_()
    self._exit_event.zero_()
    self._toe_riser_new_hit.zero_()
    self._toe_riser_contact.zero_()
    self._stair_depth_confirmation_event.zero_()
    self._entry_evidence_timer = torch.clamp(
      self._entry_evidence_timer - env.step_dt,
      min=0.0,
    )
    self._recent_entry_evidence.copy_(self._entry_evidence_timer > 0.0)
    self._entry_evidence_ascent_dir.copy_(
      torch.where(
        self._recent_entry_evidence[:, None],
        self._entry_evidence_ascent_dir,
        torch.zeros_like(self._entry_evidence_ascent_dir),
      )
    )
    self._collision_risk_now.zero_()
    self._landing_touchdown_now.zero_()
    self._landing_quality_now.zero_()
    self._stair_skip_layer_penalty.zero_()
    self._safe_stride_evidence_steps.clamp_min_(0)
    self._safe_stride_evidence_steps.sub_(1).clamp_min_(0)
    self._minimum_safe_stride_weight.copy_(
      torch.where(
        self._minimum_safe_stride_valid,
        torch.where(
          self._safe_stride_evidence_steps > 0,
          self._minimum_safe_stride_weight.clamp_min(1.0),
          torch.ones_like(self._minimum_safe_stride_weight),
        ),
        torch.zeros_like(self._minimum_safe_stride_weight),
      )
    )
    following_at_step_start = self._stair_phase >= 2
    stair_at_step_start = self._stair_phase >= 1
    self._landing_expected_layer.copy_(
      torch.where(
        following_at_step_start,
        self._expected_layer,
        torch.zeros_like(self._expected_layer),
      )
    )
    self._landing_sequence_id.copy_(
      torch.where(
        following_at_step_start,
        self._sequence_id,
        torch.full_like(self._sequence_id, -1),
      )
    )
    self._landing_target_foot.copy_(
      torch.where(
        following_at_step_start,
        self._target_foot,
        torch.full_like(self._target_foot, -1),
      )
    )
    self._clearance_foot_layers.copy_(self._foot_support_layers)
    self._clearance_foot_layers_valid.copy_(
      self._foot_support_layers_valid & stair_at_step_start[:, None]
    )
    self._clearance_sequence_id.copy_(
      torch.where(
        stair_at_step_start,
        self._sequence_id,
        torch.full_like(self._sequence_id, -1),
      )
    )
    self._clearance_ascent_dir.copy_(
      torch.where(
        stair_at_step_start[:, None],
        self._ascent_dir,
        torch.zeros_like(self._ascent_dir),
      )
    )
    boundaries, valid_boundaries = _current_step_boundaries(env)
    if boundaries is None or valid_boundaries is None:
      return torch.zeros(env.num_envs, device=env.device)
    boundary_sequence_ids, boundary_layers = _current_step_boundary_metadata(env)
    if boundary_sequence_ids is None or boundary_layers is None:
      boundary_sequence_ids = torch.ones_like(valid_boundaries, dtype=torch.long)
      boundary_layers = _step_boundary_layers(
        boundaries, valid_boundaries, boundaries.shape[1]
      )

    toe_mask = self._local_x >= toe_x_min
    if not bool(torch.any(toe_mask).item()):
      return torch.zeros(env.num_envs, device=env.device)
    points_w, point_vel_w = self._foot_points_w(env, asset_cfg)
    sole_z = torch.min(self._local_points[:, 2])
    sole_points_w = points_w[:, :, self._local_points[:, 2] <= sole_z + 1.0e-6, :]
    toe_points = points_w[:, :, toe_mask, :]
    toe_vel = point_vel_w[:, :, toe_mask, :]
    num_envs, num_feet = toe_points.shape[:2]

    level_active = _terrain_level_active(env, min_terrain_level)
    base_valid = valid_boundaries & level_active[:, None]
    foot_ref_w = self._foot_ref_w(env, asset_cfg)
    ref_dist = self._riser_slab_ref_distance(
      foot_ref_w, boundaries, slab_depth, u_margin, v_margin, surface_tol
    )
    toe_ref_radius = torch.norm(
      self._local_points[toe_mask] - self._foot_ref_local, dim=-1
    ).max()
    masked_ref_dist = torch.where(
      base_valid[:, None, :], ref_dist, torch.full_like(ref_dist, torch.inf)
    )
    slab_clearance = (
      torch.amin(masked_ref_dist, dim=(1, 2)) - toe_ref_radius
    ).clamp_min(0.0)
    all_boundaries = boundaries[:, None, :, :].expand(num_envs, num_feet, -1, -1)
    all_valid = base_valid[:, None, :].expand(num_envs, num_feet, -1)
    lip_min_dist = self._lip_min_dist(
      points_w,
      all_boundaries,
      all_valid,
      stair_touchdown_lip_height_band,
    )
    lip_clearance = torch.amin(lip_min_dist, dim=(1, 2))
    slab_unsafe_all = stair_tread_landing_reward._toe_slab_mask(
      toe_points,
      boundaries,
      base_valid,
      slab_depth,
      u_margin,
      v_margin,
      surface_tol,
    )
    proximity_risk = torch.maximum(
      _proximity_risk(slab_clearance, collision_risk_margin),
      _proximity_risk(lip_clearance, collision_risk_margin),
    )
    self._collision_risk_now.copy_(
      torch.maximum(proximity_risk, torch.any(slab_unsafe_all, dim=-1).float())
    )
    selected_idx, fallback = self._nearest_boundary_indices(
      ref_dist, base_valid, toe_ref_radius, nearest_boundaries
    )
    if selected_idx is None:
      selected_boundaries = boundaries[:, None, :, :].expand(num_envs, num_feet, -1, -1)
      selected_valid = base_valid[:, None, :].expand(num_envs, num_feet, -1)
      point_penalty, active, impact_speed, _ = self._riser_slab_point_penalty(
        toe_points,
        toe_vel,
        selected_boundaries,
        selected_valid,
        slab_depth,
        u_margin,
        v_margin,
        toe_v_threshold,
        surface_tol,
      )
      fallback_ratio = torch.zeros((), device=env.device)
    else:
      assert fallback is not None
      selected_boundaries = self._gather_by_foot(boundaries, selected_idx)
      selected_valid = self._gather_mask_by_foot(base_valid, selected_idx)
      point_penalty, active, impact_speed, _ = self._riser_slab_point_penalty(
        toe_points,
        toe_vel,
        selected_boundaries,
        selected_valid,
        slab_depth,
        u_margin,
        v_margin,
        toe_v_threshold,
        surface_tol,
      )
      fallback_ratio = fallback.float().mean()
      if bool(torch.any(fallback).item()):
        all_boundaries = boundaries[:, None, :, :].expand(num_envs, num_feet, -1, -1)
        all_valid = base_valid[:, None, :].expand(num_envs, num_feet, -1)
        full_penalty, full_active, full_impact, _ = self._riser_slab_point_penalty(
          toe_points,
          toe_vel,
          all_boundaries,
          all_valid,
          slab_depth,
          u_margin,
          v_margin,
          toe_v_threshold,
          surface_tol,
        )
        fallback_mask = fallback[:, :, None]
        point_penalty = torch.where(fallback_mask, full_penalty, point_penalty)
        active = torch.where(fallback_mask, full_active, active)
        impact_speed = torch.where(fallback_mask, full_impact, impact_speed)

    slab_penalty = torch.sum(point_penalty, dim=(1, 2))
    active_count = active.float().sum().clamp_min(1.0)
    impact_speed_mean = torch.sum(impact_speed * active.float()) / active_count
    log = env.extras["log"]
    log["Metrics/toe_riser_slab_penalty_mean"] = slab_penalty.mean()
    log["Metrics/toe_riser_slab_active_ratio"] = active.float().mean()
    log["Metrics/toe_riser_slab_impact_speed_mean"] = impact_speed_mean
    log["Metrics/toe_riser_slab_nearest_fallback_ratio"] = fallback_ratio
    if log_only:
      return torch.zeros_like(slab_penalty)

    contact_penalty = torch.zeros_like(slab_penalty)
    if contact_sensor_name is not None:
      asset: Entity = env.scene[asset_cfg.name]
      command = env.command_manager.get_command(command_name)
      if command is None:
        command_forward = torch.zeros(num_envs, device=env.device)
      else:
        command_forward = command[:, 0]
      boundary_layers = torch.where(
        base_valid, boundary_layers, torch.zeros_like(boundary_layers)
      )
      contact = self._riser_contact_terms(
        env,
        contact_sensor_name,
        asset,
        asset_cfg,
        level_active,
        boundaries,
        boundary_layers,
        toe_vel.mean(dim=2),
        command_forward,
        toe_x_min,
        contact_vertical_normal_z_max,
        contact_force_threshold,
        contact_force_scale,
        contact_penalty_scale,
        contact_time_scale,
        stair_entry_cooldown_time,
        event_min_forward_intent_speed,
        event_min_forward_speed_drop,
        event_blocked_forward_speed,
        event_persistent_steps,
        command_threshold,
      )
      contact_penalty = contact["contact_penalty"]
      new_hit_by_foot = contact["new_hit_by_foot"].bool()
      riser_contact = contact["riser_contact"].bool()
      toe_riser_new_hit = new_hit_by_foot.reshape(num_envs, -1).any(dim=-1)
      toe_riser_contact = riser_contact.reshape(num_envs, -1).any(dim=-1)
      self._toe_riser_new_hit.copy_(toe_riser_new_hit)
      self._toe_riser_contact.copy_(toe_riser_contact)
      soft_boundary_contact = contact["soft_boundary_contact"].bool()
      blocked_riser_by_foot = contact["blocked_riser_by_foot"].bool()
      event_boundary_contact = contact["event_boundary_contact"].bool()
      contact_layers = contact["contact_layers"].long()
      contact_boundary_idx = contact["contact_boundary_idx"].long()
      expanded_sequence_ids = boundary_sequence_ids[:, None, None, :].expand(
        *contact_boundary_idx.shape, boundary_sequence_ids.shape[-1]
      )
      contact_sequence_ids = torch.gather(
        expanded_sequence_ids,
        dim=-1,
        index=contact_boundary_idx[..., None],
      ).squeeze(-1)
      hit_strength = contact["hit_strength"]
      foot_pos_w = contact["foot_pos_w"]
      env_ids = torch.arange(num_envs, device=env.device)
      num_slots = contact_layers.shape[-1]
      root_xy = asset.data.root_link_pos_w[:, :2]

      layer1_contact = (
        event_boundary_contact & (contact_layers == 1) & new_hit_by_foot[:, :, None]
      )
      contact_score = torch.where(
        layer1_contact, hit_strength, torch.full_like(hit_strength, -torch.inf)
      )
      first_flat = torch.argmax(contact_score.reshape(num_envs, -1), dim=-1)
      first_foot = first_flat // num_slots
      first_slot = first_flat % num_slots
      first_boundary_idx = contact_boundary_idx[env_ids, first_foot, first_slot]
      first_sequence_id = boundary_sequence_ids[env_ids, first_boundary_idx]
      first_normal = self._normalize_xy(boundaries[env_ids, first_boundary_idx, 6:8])
      first_p0 = boundaries[env_ids, first_boundary_idx, 0:2]
      root_side = torch.sum((root_xy - first_p0) * first_normal, dim=-1)
      ascent_dir = -torch.where(root_side >= 0.0, 1.0, -1.0)[:, None] * first_normal
      entry_heading = self._base_heading_cos(asset.data.root_link_quat_w, ascent_dir)
      command_active = torch.ones(num_envs, device=env.device, dtype=torch.bool)
      if command is not None:
        command_active = torch.norm(command[:, :2], dim=-1) > command_threshold
      layer1_new_hit = torch.any(layer1_contact, dim=(1, 2))
      layer1_entry_candidate = _strict_stair_entry_event(
        self._stair_phase, layer1_new_hit
      )
      entry_mask = layer1_entry_candidate
      repeat_layer1_hit = _repeat_sequence_layer1_hit(
        self._stair_phase,
        event_boundary_contact,
        contact_layers,
        new_hit_by_foot,
        contact_sequence_ids,
        self._sequence_id,
      ) & (self._expected_layer == 1)
      event_evidence = entry_mask | repeat_layer1_hit
      self._repeat_layer1_hit_count += repeat_layer1_hit.long()
      self._entry_event.copy_(event_evidence)
      evidence_time = max(float(stair_entry_evidence_time), 0.0)
      self._entry_evidence_timer = torch.where(
        event_evidence,
        torch.full_like(self._entry_evidence_timer, evidence_time),
        self._entry_evidence_timer,
      )
      self._recent_entry_evidence.copy_(self._entry_evidence_timer > 0.0)
      next_entry_dir = torch.where(
        entry_mask[:, None],
        ascent_dir,
        self._entry_evidence_ascent_dir,
      )
      self._entry_evidence_ascent_dir.copy_(
        torch.where(
          self._recent_entry_evidence[:, None],
          next_entry_dir,
          torch.zeros_like(next_entry_dir),
        )
      )
      log["Metrics/toe_riser_new_hit_ratio"] = toe_riser_new_hit.float().mean()
      log["Metrics/toe_riser_contact_ratio"] = toe_riser_contact.float().mean()
      log["Metrics/stair_entry_layer1_new_hit_ratio"] = layer1_new_hit.float().mean()
      layer1_candidate_count = layer1_entry_candidate.float().sum()
      self._layer1_entry_candidate_total += layer1_candidate_count
      self._entry_event_total += entry_mask.float().sum()
      log["Metrics/stair_layer1_hit_to_event_rate"] = (
        self._entry_event_total / self._layer1_entry_candidate_total.clamp_min(1.0)
      )
      log["Metrics/stair_layer1_entry_candidate_ratio"] = (
        layer1_entry_candidate.float().mean()
      )
      log["Metrics/stair_repeat_layer1_new_hit_ratio"] = (
        repeat_layer1_hit.float().mean()
      )
      log["Metrics/stair_repeat_layer1_new_hit_count"] = repeat_layer1_hit.float().sum()
      log["Metrics/stair_event_evidence_ratio"] = event_evidence.float().mean()
      log["Metrics/stair_event_repeat_evidence_ratio"] = (
        repeat_layer1_hit.float().sum() / event_evidence.float().sum().clamp_min(1.0)
      )
      candidate_count_for_metrics = layer1_candidate_count.clamp_min(1.0)
      log["Metrics/stair_entry_heading_valid_ratio"] = (
        layer1_entry_candidate & (entry_heading >= stair_heading_cos)
      ).float().sum() / candidate_count_for_metrics
      log["Metrics/stair_entry_command_active_ratio"] = (
        layer1_entry_candidate & command_active
      ).float().sum() / candidate_count_for_metrics
      log["Metrics/stair_entry_recent_evidence_ratio"] = (
        self._recent_entry_evidence.float().mean()
      )
      entry_stride_label_valid = torch.zeros_like(entry_mask)
      if bool(torch.any(entry_mask).item()):
        entry_ids = entry_mask.nonzero(as_tuple=False).squeeze(-1)
        entry_tread_depth, entry_riser_height, entry_shape_valid = (
          stair_shape_for_sequence(
            boundaries[entry_ids],
            valid_boundaries[entry_ids],
            boundary_sequence_ids[entry_ids],
            first_sequence_id[entry_ids],
          )
        )
        self._first_foot[entry_mask] = first_foot[entry_mask]
        self._target_foot[entry_mask] = 1 - first_foot[entry_mask]
        self._support_foot[entry_mask] = first_foot[entry_mask]
        self._support_layer[entry_mask] = 0
        self._expected_layer[entry_mask] = 1
        self._foot_support_layers[entry_mask] = 0
        self._foot_support_layers_valid[entry_mask] = False
        self._stair_phase[entry_mask] = 1
        self._stair_tread_depth_label[entry_ids] = entry_tread_depth
        self._stair_riser_height_label[entry_ids] = entry_riser_height
        self._stair_shape_label_valid[entry_ids] = entry_shape_valid
        self._stair_depth_label_valid[entry_ids] = False
        self._safe_stride_valid[entry_mask] = False
        self._safe_tread_lower_bound[entry_mask] = 0.0
        self._minimum_safe_stride_valid[entry_mask] = False
        self._minimum_safe_stride_interval_valid[entry_mask] = False
        self._minimum_safe_stride_exact[entry_mask] = False
        self._minimum_safe_stride_weight[entry_mask] = 0.0
        self._safe_stride_evidence_steps[entry_mask] = 0
        self._probe_evidence_seen[entry_mask] = False
        self._minimum_safe_stride_geometry_valid[entry_mask] = False
        self._minimum_safe_stride_interval_geometry_valid[entry_mask] = False
        self._minimum_safe_stride_raw[entry_mask] = 0.0
        self._minimum_safe_stride[entry_mask] = 0.0
        self._minimum_safe_stride_upper[entry_mask] = 0.0
        self._safe_landing_center_s[entry_mask] = 0.0
        self._observed_step_stride[entry_mask] = 0.0
        self._ascent_dir[entry_mask] = ascent_dir[entry_mask]
        self._first_boundary_idx[entry_mask] = first_boundary_idx[entry_mask]
        self._sequence_id[entry_mask] = first_sequence_id[entry_mask]
        self._context_steps[entry_mask] = 0
        self._attempt_elapsed_time[entry_mask] = 0.0
        self._attempt_air_seen[entry_mask] = False
        self._attempt_expected_tread_contact[entry_mask] = False
        self._attempt_expected_tread_full_support[entry_mask] = False
        self._attempt_expected_riser_contact[entry_mask] = False
        self._attempt_rear_partial_contact[entry_mask] = False
        self._attempt_touchdown_layer[entry_mask] = 0
        target_at_entry = self._target_foot[entry_mask]
        self._target_start_pos_w[entry_mask] = foot_pos_w[entry_mask, target_at_entry]
        entry_target_ref_w = foot_ref_w[entry_ids, target_at_entry]
        entry_target_sole_w = sole_points_w[entry_ids, target_at_entry]
        (
          entry_stride_raw,
          entry_stride_upper_raw,
          entry_stride_lower_geometry_valid,
          entry_stride_interval_geometry_valid,
        ) = _minimum_safe_stride_to_layer(
          boundaries[entry_ids],
          valid_boundaries[entry_ids],
          boundary_sequence_ids[entry_ids],
          boundary_layers[entry_ids],
          first_sequence_id[entry_ids],
          torch.ones_like(first_sequence_id[entry_ids]),
          entry_target_ref_w,
          entry_target_sole_w,
          ascent_dir[entry_ids],
          entry_tread_depth,
          safe_stride_containment_margin,
          safe_stride_containment_margin,
        )
        entry_interval_geometry_valid = (
          entry_shape_valid & entry_stride_interval_geometry_valid
        )
        (
          entry_stride_lower,
          entry_stride_upper,
          entry_lower_tracking_valid,
          entry_interval_tracking_valid,
          _entry_stale,
        ) = _bounded_safe_stride_interval(
          entry_stride_raw,
          entry_stride_upper_raw,
          entry_stride_lower_geometry_valid,
          entry_interval_geometry_valid,
          stair_min_safe_stride,
          stair_max_safe_stride,
          stair_max_tracking_stride,
        )
        self._minimum_safe_stride_raw[entry_ids] = entry_stride_raw
        self._minimum_safe_stride[entry_ids] = entry_stride_lower
        self._minimum_safe_stride_upper[entry_ids] = entry_stride_upper
        self._minimum_safe_stride_geometry_valid[entry_ids] = (
          entry_stride_lower_geometry_valid
        )
        self._minimum_safe_stride_interval_geometry_valid[entry_ids] = (
          entry_interval_tracking_valid
        )
        self._minimum_safe_stride_valid[entry_ids] = entry_lower_tracking_valid
        self._minimum_safe_stride_interval_valid[entry_ids] = False
        entry_stride_label_valid[entry_ids] = entry_lower_tracking_valid
        self._minimum_safe_stride_exact[entry_ids] = False
        self._minimum_safe_stride_weight[entry_ids] = torch.where(
          entry_lower_tracking_valid,
          torch.full_like(entry_stride_raw, safe_stride_riser_weight),
          torch.zeros_like(entry_stride_raw),
        )
        self._safe_stride_evidence_steps[entry_ids] = torch.where(
          entry_lower_tracking_valid,
          torch.full_like(
            first_sequence_id[entry_ids],
            safe_stride_evidence_window_steps,
          ),
          torch.zeros_like(first_sequence_id[entry_ids]),
        )

      phase1 = self._stair_phase == 1
      self._context_steps = torch.where(
        self._stair_phase >= 1,
        self._context_steps + 1,
        self._context_steps,
      )
      target_foot = self._target_foot.clamp(0, max(1, num_feet - 1))
      target_pos_w = foot_pos_w[env_ids, target_foot]
      heading_cos = self._base_heading_cos(
        asset.data.root_link_quat_w, self._ascent_dir
      )
      heading_gate = heading_cos >= stair_heading_cos
      tread_depth, _riser_height, shape_valid = cached_stair_shape(
        env, boundaries, valid_boundaries
      )

      ground_sensor = env.scene[ground_contact_sensor_name]
      assert isinstance(ground_sensor, ContactSensor), (
        f"Safe stair touchdown requires ContactSensor {ground_contact_sensor_name!r}."
      )
      ground_first_contact = ground_sensor.compute_first_contact(dt=env.step_dt)
      if ground_first_contact.shape[-1] != num_feet:
        raise RuntimeError("Safe stair touchdown requires one channel per foot.")
      ground_contact_time = ground_sensor.data.current_contact_time
      if ground_contact_time is None:
        ground_contact = ground_first_contact
      else:
        if ground_contact_time.shape[-1] != num_feet:
          if ground_contact_time.shape[-1] % num_feet != 0:
            raise RuntimeError("Safe stair touchdown requires one channel per foot.")
          ground_contact_time = (
            ground_contact_time.view(num_envs, num_feet, -1).max(dim=-1).values
          )
        ground_contact = ground_contact_time > 0.0
      following_before = self._stair_phase == 2
      active_before = phase1 | following_before
      target_touchdown = active_before & ground_first_contact[env_ids, target_foot]
      target_contact = active_before & ground_contact[env_ids, target_foot]

      ref_xy = foot_ref_w[:, :, None, :2]
      p0_xy = boundaries[:, None, :, 0:2]
      edge_xy = boundaries[:, None, :, 3:5] - p0_xy
      edge_len = torch.norm(edge_xy, dim=-1).clamp_min(1.0e-6)
      edge_tangent = edge_xy / edge_len[..., None]
      ref_lateral = torch.sum((ref_xy - p0_xy) * edge_tangent, dim=-1)
      ref_segment_gate = (ref_lateral >= -stair_touchdown_lateral_margin) & (
        ref_lateral <= edge_len + stair_touchdown_lateral_margin
      )
      ref_height_gate = (
        torch.abs(foot_ref_w[:, :, None, 2] - boundaries[:, None, :, 10])
        <= stair_touchdown_height_tolerance
      )
      sequence_boundary = (
        base_valid
        & (boundary_sequence_ids == self._sequence_id[:, None])
        & (boundary_layers > 0)
      )
      target_layer_boundary = sequence_boundary & (
        boundary_layers == self._expected_layer[:, None]
      )
      target_height_gate_by_foot = torch.any(
        target_layer_boundary[:, None, :] & ref_height_gate, dim=-1
      )
      target_geometry_gate_by_foot = torch.any(
        target_layer_boundary[:, None, :] & ref_height_gate & ref_segment_gate,
        dim=-1,
      )

      tread_support_fraction = self._tread_support_fraction(
        sole_points_w, boundaries, tread_depth
      )
      support_candidate = (
        sequence_boundary[:, None, :] & ref_height_gate & ref_segment_gate
      )
      candidate_support_fraction = torch.where(
        support_candidate,
        tread_support_fraction,
        torch.full_like(tread_support_fraction, -1.0),
      )
      best_support_fraction, best_support_idx = torch.max(
        candidate_support_fraction, dim=-1
      )
      has_support_candidate = best_support_fraction >= 0.0
      best_support_fraction = best_support_fraction.clamp_min(0.0)
      best_support_layer = torch.gather(
        boundary_layers[:, None, :].expand(num_envs, num_feet, -1),
        dim=-1,
        index=best_support_idx[..., None],
      ).squeeze(-1)
      best_support_layer = torch.where(
        has_support_candidate,
        best_support_layer,
        torch.zeros_like(best_support_layer),
      )
      full_support_gate = best_support_fraction >= 1.0 - 1.0e-6
      stair_layer_observed = _stair_tread_overlap_contact(
        active_before,
        ground_contact,
        has_support_candidate,
        best_support_fraction,
        best_support_layer,
      )
      partial_layer_update = stair_layer_observed & ~full_support_gate
      ground_layer_confirmed = phase1[:, None] & ground_contact & ~has_support_candidate
      foot_layer_confirmed = stair_layer_observed | ground_layer_confirmed
      confirmed_foot_layer = torch.where(
        stair_layer_observed,
        best_support_layer,
        torch.zeros_like(best_support_layer),
      )
      self._foot_support_layers.copy_(
        torch.where(
          foot_layer_confirmed,
          confirmed_foot_layer,
          self._foot_support_layers,
        )
      )
      self._foot_support_layers_valid.logical_or_(foot_layer_confirmed)

      slab_unsafe = slab_unsafe_all
      lip_unsafe = torch.any(lip_min_dist < stair_touchdown_lip_clearance, dim=-1)
      target_forward_gain = self._ascent_progress(
        target_pos_w, self._target_start_pos_w, self._ascent_dir
      )
      target_geometry_gate = target_geometry_gate_by_foot[env_ids, target_foot]
      target_height_gate = target_height_gate_by_foot[env_ids, target_foot]
      target_full_support_gate = full_support_gate[env_ids, target_foot]
      target_riser_unsafe = riser_contact[env_ids, target_foot]
      target_slab_unsafe = slab_unsafe[env_ids, target_foot]
      target_lip_unsafe = lip_unsafe[env_ids, target_foot]
      target_support_layer = best_support_layer[env_ids, target_foot]
      target_has_stair_support = (
        target_contact
        & has_support_candidate[env_ids, target_foot]
        & (target_support_layer > 0)
      )

      target_riser_evidence = (
        soft_boundary_contact
        & blocked_riser_by_foot[:, :, None]
        & (contact_sequence_ids == self._sequence_id[:, None, None])
      )[env_ids, target_foot]
      target_riser_layers = torch.where(
        target_riser_evidence,
        contact_layers[env_ids, target_foot],
        torch.zeros_like(contact_layers[env_ids, target_foot]),
      )
      target_riser_layer = torch.max(target_riser_layers, dim=-1).values
      target_has_riser_evidence = target_riser_layer > 0
      target_has_expected_riser_evidence = target_riser_layer == self._expected_layer
      expected_riser_event = active_before & target_has_expected_riser_evidence
      self._entry_event.logical_or_(expected_riser_event)
      target_has_surface_evidence = (
        target_slab_unsafe | target_lip_unsafe
      ) & target_height_gate
      probe_stage = phase1 & ~entry_mask
      probe_evidence_now = (
        probe_stage
        & shape_valid
        & (
          target_has_stair_support
          | target_has_riser_evidence
          | target_has_surface_evidence
        )
      )
      depth_evidence_now = _confirmed_stair_depth_evidence(
        active_before,
        shape_valid,
        self._expected_layer,
        target_has_expected_riser_evidence,
      )
      probe_exact_now = probe_stage & depth_evidence_now
      self._probe_evidence_seen.logical_or_(probe_evidence_now)

      target_airborne = active_before & ~ground_contact[env_ids, target_foot]
      self._attempt_air_seen.logical_or_(target_airborne)
      attempt_armed = active_before & self._attempt_air_seen
      self._attempt_elapsed_time += attempt_armed.float() * env.step_dt
      expected_tread_contact_now = (
        attempt_armed
        & target_has_stair_support
        & target_geometry_gate
        & (best_support_fraction[env_ids, target_foot] > 0.0)
        & (target_support_layer == self._expected_layer)
      )
      self._stair_depth_confirmation_event.copy_(
        depth_evidence_now & ~self._stair_depth_label_valid
      )
      self._stair_depth_label_valid.logical_or_(depth_evidence_now)
      confirmed_current_interval = (
        self._stair_depth_label_valid
        & self._minimum_safe_stride_valid
        & self._minimum_safe_stride_interval_geometry_valid
      )
      self._minimum_safe_stride_interval_valid.logical_or_(confirmed_current_interval)
      self._minimum_safe_stride_exact.logical_or_(confirmed_current_interval)
      self._attempt_expected_tread_contact.logical_or_(expected_tread_contact_now)
      self._attempt_expected_tread_full_support.logical_or_(
        expected_tread_contact_now & target_full_support_gate
      )
      self._attempt_expected_riser_contact.logical_or_(
        attempt_armed & target_has_expected_riser_evidence
      )
      target_boundary_idx = best_support_idx[env_ids, target_foot]
      target_boundary_p0 = boundaries[env_ids, target_boundary_idx, 0:2]
      target_sole_s = torch.sum(
        (sole_points_w[env_ids, target_foot, :, :2] - target_boundary_p0[:, None, :])
        * self._ascent_dir[:, None, :],
        dim=-1,
      )
      rear_partial_now = (
        expected_tread_contact_now
        & ~target_full_support_gate
        & (torch.amin(target_sole_s, dim=-1) < 0.0)
      )
      self._attempt_rear_partial_contact.logical_or_(rear_partial_now)
      target_tread_touchdown = (
        attempt_armed & target_touchdown & target_has_stair_support
      )
      self._attempt_touchdown_layer.copy_(
        torch.where(
          target_tread_touchdown,
          torch.maximum(self._attempt_touchdown_layer, target_support_layer),
          self._attempt_touchdown_layer,
        )
      )
      half_cycle_elapsed = self._attempt_elapsed_time >= 0.5 * stair_attempt_period
      attempt_complete = (
        active_before
        & ~entry_mask
        & attempt_armed
        & (target_touchdown | half_cycle_elapsed)
      )
      completed_expected_tread = attempt_complete & self._attempt_expected_tread_contact
      completed_full_support = (
        attempt_complete & self._attempt_expected_tread_full_support
      )
      completed_expected_riser = attempt_complete & self._attempt_expected_riser_contact
      completed_rear_partial = attempt_complete & self._attempt_rear_partial_contact
      completed_by_touchdown = attempt_complete & target_touchdown
      completed_by_timeout = attempt_complete & ~target_touchdown & half_cycle_elapsed
      completed_touchdown_layer = self._attempt_touchdown_layer.clone()
      completed_delta, completed_normal_step, completed_skipped_layers = (
        _stair_touchdown_progress(
          completed_touchdown_layer,
          self._support_layer,
        )
      )
      completed_skip = (
        attempt_complete & (completed_touchdown_layer > 0) & (completed_delta >= 2)
      )
      self._stair_skip_layer_penalty += (
        completed_skipped_layers.float() * completed_skip.float()
      )
      probe_attempt_complete = phase1 & attempt_complete
      probe_skip = phase1 & completed_skip
      following_skip = following_before & completed_skip
      following_normal_step = (
        following_before & completed_expected_tread & completed_normal_step
      )
      stale_stride_reset = torch.zeros_like(attempt_complete)

      if bool(torch.any(attempt_complete).item()):
        completed_ids = attempt_complete.nonzero(as_tuple=False).squeeze(-1)
        previous_target = target_foot[completed_ids]
        next_target, next_reached, next_layer = _advance_stair_attempt(
          previous_target,
          self._support_layer[completed_ids],
          self._expected_layer[completed_ids],
          self._attempt_expected_tread_contact[completed_ids],
        )
        reached_expected = self._attempt_expected_tread_contact[completed_ids]
        self._support_foot[completed_ids] = torch.where(
          reached_expected,
          previous_target,
          self._support_foot[completed_ids],
        )
        self._support_layer[completed_ids] = next_reached
        next_ref_w = foot_ref_w[completed_ids, next_target]
        next_sole_w = sole_points_w[completed_ids, next_target]
        (
          minimum_stride_raw,
          maximum_stride_raw,
          minimum_stride_lower_geometry_valid,
          minimum_stride_interval_geometry_valid,
        ) = _minimum_safe_stride_to_layer(
          boundaries[completed_ids],
          valid_boundaries[completed_ids],
          boundary_sequence_ids[completed_ids],
          boundary_layers[completed_ids],
          self._sequence_id[completed_ids],
          next_layer,
          next_ref_w,
          next_sole_w,
          self._ascent_dir[completed_ids],
          tread_depth[completed_ids],
          safe_stride_containment_margin,
          safe_stride_containment_margin,
        )
        interval_geometry_label_valid = (
          shape_valid[completed_ids] & minimum_stride_interval_geometry_valid
        )
        (
          minimum_stride,
          maximum_stride,
          lower_tracking_label_valid,
          interval_tracking_label_valid,
          stale_stride,
        ) = _bounded_safe_stride_interval(
          minimum_stride_raw,
          maximum_stride_raw,
          minimum_stride_lower_geometry_valid,
          interval_geometry_label_valid,
          stair_min_safe_stride,
          stair_max_safe_stride,
          stair_max_tracking_stride,
        )
        stale_stride_reset[completed_ids] = stale_stride
        self._target_foot[completed_ids] = next_target
        self._expected_layer[completed_ids] = next_layer
        self._target_start_pos_w[completed_ids] = next_ref_w
        self._minimum_safe_stride_raw[completed_ids] = minimum_stride_raw
        self._minimum_safe_stride[completed_ids] = minimum_stride
        self._minimum_safe_stride_upper[completed_ids] = maximum_stride
        self._minimum_safe_stride_geometry_valid[completed_ids] = (
          minimum_stride_lower_geometry_valid
        )
        self._minimum_safe_stride_interval_geometry_valid[completed_ids] = (
          interval_tracking_label_valid
        )
        self._minimum_safe_stride_valid[completed_ids] = lower_tracking_label_valid
        interval_label_valid = (
          self._stair_depth_label_valid[completed_ids] & interval_tracking_label_valid
        )
        self._minimum_safe_stride_interval_valid[completed_ids] = interval_label_valid
        self._minimum_safe_stride_exact[completed_ids] = interval_label_valid
        completed_importance = torch.ones_like(minimum_stride_raw)
        completed_importance = torch.where(
          completed_rear_partial[completed_ids],
          torch.full_like(completed_importance, safe_stride_rear_partial_weight),
          completed_importance,
        )
        completed_importance = torch.where(
          completed_expected_riser[completed_ids],
          torch.full_like(completed_importance, safe_stride_riser_weight),
          completed_importance,
        )
        evidence_valid = lower_tracking_label_valid & (completed_importance > 1.0)
        self._minimum_safe_stride_weight[completed_ids] = torch.where(
          lower_tracking_label_valid,
          completed_importance,
          torch.zeros_like(completed_importance),
        )
        self._safe_stride_evidence_steps[completed_ids] = torch.where(
          evidence_valid,
          torch.full_like(next_layer, safe_stride_evidence_window_steps),
          torch.zeros_like(next_layer),
        )
        self._stair_phase[completed_ids] = 2
        self._attempt_elapsed_time[completed_ids] = 0.0
        self._attempt_air_seen[completed_ids] = False
        self._attempt_expected_tread_contact[completed_ids] = False
        self._attempt_expected_tread_full_support[completed_ids] = False
        self._attempt_expected_riser_contact[completed_ids] = False
        self._attempt_rear_partial_contact[completed_ids] = False
        self._attempt_touchdown_layer[completed_ids] = 0

      safe_touchdown = (
        target_contact
        & heading_gate
        & shape_valid
        & target_geometry_gate
        & target_full_support_gate
        & ~target_riser_unsafe
        & ~target_slab_unsafe
        & ~target_lip_unsafe
      )

      selected_idx = best_support_idx[env_ids, target_foot]
      selected_p0_xy = boundaries[env_ids, selected_idx, 0:2]
      selected_normal = self._normalize_xy(boundaries[env_ids, selected_idx, 6:8])
      target_ref_xy = foot_ref_w[env_ids, target_foot, :2]
      landing_center_s = torch.sum(
        (target_ref_xy - selected_p0_xy) * -selected_normal, dim=-1
      )
      target_coverage = best_support_fraction[env_ids, target_foot]
      quality_sigma = torch.clamp(0.25 * tread_depth, min=0.03)
      center_score = torch.exp(
        -torch.square((landing_center_s - 0.5 * tread_depth) / quality_sigma)
      )
      edge_clearance_score = torch.clamp(
        torch.minimum(landing_center_s, tread_depth - landing_center_s) / 0.06,
        min=0.0,
        max=1.0,
      )
      stage1_quality_valid = (
        target_touchdown
        & heading_gate
        & shape_valid
        & target_geometry_gate
        & ~target_riser_unsafe
        & ~target_slab_unsafe
        & ~target_lip_unsafe
      )
      self._landing_touchdown_now.copy_(target_touchdown & heading_gate)
      self._landing_quality_now.copy_(
        _landing_quality(
          target_coverage,
          center_score,
          edge_clearance_score,
          stage1_quality_valid,
        )
      )
      self._collision_risk_now.copy_(
        torch.maximum(
          self._collision_risk_now,
          torch.any(riser_contact, dim=-1).float(),
        )
      )
      target_toe_points = toe_points[env_ids, target_foot, :, :2]
      toe_front_s = (
        torch.sum(
          (target_toe_points - selected_p0_xy[:, None, :])
          * -selected_normal[:, None, :],
          dim=-1,
        )
        .max(dim=-1)
        .values
      )
      min_memory_stride = torch.minimum(
        torch.full_like(tread_depth, stair_min_safe_stride),
        tread_depth,
      )
      safe_tread_lower_bound = torch.minimum(
        torch.maximum(toe_front_s, min_memory_stride),
        tread_depth,
      )
      safe_landing_center = torch.minimum(
        torch.clamp_min(landing_center_s, 0.0),
        tread_depth,
      )
      safe_stride_label_clamped = (
        torch.abs(safe_tread_lower_bound - toe_front_s) > 1.0e-4
      ) | (torch.abs(safe_landing_center - landing_center_s) > 1.0e-4)
      if bool(torch.any(safe_touchdown).item()):
        self._safe_stride_valid[safe_touchdown] = True
        self._safe_tread_lower_bound[safe_touchdown] = safe_tread_lower_bound[
          safe_touchdown
        ]
        self._safe_landing_center_s[safe_touchdown] = safe_landing_center[
          safe_touchdown
        ]
        self._observed_step_stride[safe_touchdown] = target_forward_gain[safe_touchdown]

      following = self._stair_phase == 2
      active_phase = phase1 | following

      boundary_ascent = -self._normalize_xy(boundaries[..., 6:8])
      aligned_boundary = (
        base_valid
        & (boundary_sequence_ids == self._sequence_id[:, None])
        & (torch.sum(boundary_ascent * self._ascent_dir[:, None, :], dim=-1) >= 0.90)
      )
      boundary_s = torch.sum(
        boundaries[..., 0:2] * self._ascent_dir[:, None, :], dim=-1
      )
      last_boundary_s = torch.max(
        torch.where(
          aligned_boundary, boundary_s, torch.full_like(boundary_s, -torch.inf)
        ),
        dim=-1,
      ).values
      root_s = torch.sum(asset.data.root_link_pos_w[:, :2] * self._ascent_dir, dim=-1)
      past_last_boundary = (
        active_phase
        & torch.isfinite(last_boundary_s)
        & (root_s > last_boundary_s + tread_depth)
      )
      ahead_boundary = aligned_boundary & (boundary_s > root_s[:, None] + 0.05)
      no_higher_tread_ahead = active_phase & ~torch.any(ahead_boundary, dim=-1)
      exit_candidate = (
        active_phase & heading_gate & (past_last_boundary | no_higher_tread_ahead)
      )
      touchdown_now = torch.any(ground_first_contact, dim=-1)
      unsafe_now = torch.any(riser_contact | slab_unsafe | lip_unsafe, dim=-1)
      flat_exit_step = exit_candidate & touchdown_now & ~unsafe_now
      self._exit_flat_steps, confirmed_exit = self._advance_flat_exit_confirmation(
        active_phase,
        exit_candidate,
        touchdown_now,
        unsafe_now,
        self._exit_flat_steps,
      )

      phase1_count = phase1.float().sum().clamp_min(1.0)
      active_phase_count = active_phase.float().sum().clamp_min(1.0)
      touchdown_count = target_touchdown.float().sum().clamp_min(1.0)
      contact_count = target_contact.float().sum().clamp_min(1.0)
      geometry_touchdown = target_touchdown & target_geometry_gate
      geometry_contact = target_contact & target_geometry_gate
      geometry_contact_count = geometry_contact.float().sum().clamp_min(1.0)
      log["Metrics/stair_entry_event_ratio"] = entry_mask.float().mean()
      log["Metrics/stair_entry_event_count"] = entry_mask.float().sum()
      log["Metrics/stair_heading_cos_mean"] = (
        torch.where(phase1, heading_cos, torch.zeros_like(heading_cos)).sum()
        / phase1_count
      )
      log["Metrics/stair_heading_gate_ratio"] = (
        phase1 & heading_gate
      ).float().sum() / phase1_count
      log["Metrics/stair_layer2_touchdown_candidate_ratio"] = (
        target_touchdown.float().sum() / phase1_count
      )
      log["Metrics/stair_layer2_contact_candidate_ratio"] = (
        target_contact.float().sum() / phase1_count
      )
      log["Metrics/stair_layer2_height_gate_ratio"] = (
        target_touchdown & target_height_gate
      ).float().sum() / touchdown_count
      log["Metrics/stair_layer2_contact_height_gate_ratio"] = (
        target_contact & target_height_gate
      ).float().sum() / contact_count
      log["Metrics/stair_layer2_segment_gate_ratio"] = (
        geometry_touchdown.float().sum() / touchdown_count
      )
      log["Metrics/stair_layer2_contact_segment_gate_ratio"] = (
        geometry_contact.float().sum() / contact_count
      )
      log["Metrics/stair_layer2_support_fraction_mean"] = (
        torch.where(
          geometry_contact,
          best_support_fraction[env_ids, target_foot],
          torch.zeros_like(best_support_fraction[:, 0]),
        ).sum()
        / geometry_contact_count
      )
      log["Metrics/stair_layer2_full_support_ratio"] = (
        geometry_contact & target_full_support_gate
      ).float().sum() / geometry_contact_count
      log["Metrics/stair_safe_touchdown_ratio"] = (
        safe_touchdown.float().sum() / phase1_count
      )
      log["Metrics/minimum_safe_stride_probe_evidence_ratio"] = (
        probe_evidence_now.float().sum() / phase1_count
      )
      log["Metrics/minimum_safe_stride_entry_target_valid_ratio"] = (
        entry_stride_label_valid.float().sum() / entry_mask.float().sum().clamp_min(1.0)
      )
      log["Metrics/minimum_safe_stride_probe_evidence_seen_ratio"] = (
        self._probe_evidence_seen & phase1
      ).float().sum() / phase1_count
      log["Metrics/minimum_safe_stride_probe_exact_evidence_ratio"] = (
        probe_exact_now.float().sum() / phase1_count
      )
      log["Metrics/minimum_safe_stride_depth_confirmation_event_ratio"] = (
        self._stair_depth_confirmation_event.float().sum() / active_phase_count
      )
      log["Metrics/minimum_safe_stride_depth_confirmation_event_count"] = (
        self._stair_depth_confirmation_event.float().sum()
      )
      log["Metrics/minimum_safe_stride_probe_attempt_completed_ratio"] = (
        probe_attempt_complete.float().sum() / phase1_count
      )
      log["Metrics/stair_probe_skip_layer_ratio"] = (
        probe_skip.float().sum() / phase1_count
      )
      log["Metrics/stair_following_skip_layer_ratio"] = (
        following_skip.float().sum() / following_before.float().sum().clamp_min(1.0)
      )
      log["Metrics/stair_following_normal_step_ratio"] = (
        following_normal_step.float().sum()
        / following_before.float().sum().clamp_min(1.0)
      )
      log["Metrics/stair_attempt_completed_ratio"] = (
        attempt_complete.float().sum() / active_phase_count
      )
      attempt_complete_count = attempt_complete.float().sum().clamp_min(1.0)
      completed_tread_count = completed_expected_tread.float().sum().clamp_min(1.0)
      log["Metrics/stair_attempt_expected_tread_reached_ratio"] = (
        completed_expected_tread.float().sum() / attempt_complete_count
      )
      log["Metrics/stair_attempt_expected_tread_partial_ratio"] = (
        completed_expected_tread & ~completed_full_support
      ).float().sum() / completed_tread_count
      log["Metrics/stair_attempt_completed_by_touchdown_ratio"] = (
        completed_by_touchdown.float().sum() / attempt_complete_count
      )
      log["Metrics/stair_attempt_completed_by_timeout_ratio"] = (
        completed_by_timeout.float().sum() / attempt_complete_count
      )
      log["Metrics/stair_attempt_expected_riser_ratio"] = (
        completed_expected_riser.float().sum() / attempt_complete_count
      )
      log["Metrics/stair_expected_riser_event_ratio"] = (
        expected_riser_event.float().sum() / active_phase_count
      )
      log["Metrics/stair_memory_expected_riser_event_ratio"] = (
        expected_riser_event & following_before
      ).float().sum() / following_before.float().sum().clamp_min(1.0)
      log["Metrics/stair_attempt_rear_partial_ratio"] = (
        completed_rear_partial.float().sum() / attempt_complete_count
      )
      log["Metrics/stair_attempt_wrong_tread_ratio"] = (
        attempt_complete & (completed_touchdown_layer > 0) & ~completed_expected_tread
      ).float().sum() / attempt_complete_count
      log["Metrics/stair_attempt_no_stair_contact_ratio"] = (
        attempt_complete & (completed_touchdown_layer == 0) & ~completed_expected_riser
      ).float().sum() / attempt_complete_count
      log["Metrics/stair_attempt_riser_or_miss_ratio"] = (
        attempt_complete & ~completed_expected_tread
      ).float().sum() / attempt_complete_count
      log["Metrics/shank_clearance_partial_layer_update_ratio"] = (
        partial_layer_update.float().mean()
      )
      log["Metrics/stair_expected_layer_mismatch_ratio"] = (
        following_before & (self._expected_layer != self._support_layer + 1)
      ).float().sum() / following_before.float().sum().clamp_min(1.0)
      label_geometry_valid = self._minimum_safe_stride_geometry_valid
      label_geometry_count = label_geometry_valid.float().sum().clamp_min(1.0)
      raw_minimum_stride = self._minimum_safe_stride_raw
      log["Metrics/minimum_safe_stride_raw_mean"] = (
        raw_minimum_stride * label_geometry_valid.float()
      ).sum() / label_geometry_count
      if bool(torch.any(label_geometry_valid).item()):
        valid_raw_stride = raw_minimum_stride[label_geometry_valid]
        log["Metrics/minimum_safe_stride_raw_min"] = torch.min(valid_raw_stride)
        log["Metrics/minimum_safe_stride_raw_p01"] = torch.quantile(
          valid_raw_stride, 0.01
        )
        log["Metrics/minimum_safe_stride_raw_p99"] = torch.quantile(
          valid_raw_stride, 0.99
        )
        log["Metrics/minimum_safe_stride_raw_max"] = torch.max(valid_raw_stride)
      else:
        zero = torch.zeros((), device=env.device)
        log["Metrics/minimum_safe_stride_raw_min"] = zero
        log["Metrics/minimum_safe_stride_raw_p01"] = zero
        log["Metrics/minimum_safe_stride_raw_p99"] = zero
        log["Metrics/minimum_safe_stride_raw_max"] = zero
      log["Metrics/minimum_safe_stride_lower_clamp_ratio"] = (
        label_geometry_valid & (raw_minimum_stride < stair_min_safe_stride)
      ).float().sum() / label_geometry_count
      log["Metrics/minimum_safe_stride_upper_clamp_ratio"] = (
        label_geometry_valid & (raw_minimum_stride > stair_max_safe_stride)
      ).float().sum() / label_geometry_count
      log["Metrics/minimum_safe_stride_stale_target_ratio"] = (
        label_geometry_valid & (raw_minimum_stride > stair_max_tracking_stride)
      ).float().sum() / label_geometry_count
      log["Metrics/minimum_safe_stride_stale_reset_ratio"] = (
        stale_stride_reset.float().sum() / attempt_complete_count
      )
      minimum_stride_valid = self._minimum_safe_stride_valid
      minimum_stride_valid_count = minimum_stride_valid.float().sum().clamp_min(1.0)
      log["Metrics/minimum_safe_stride_valid_ratio"] = (
        minimum_stride_valid.float().mean()
      )
      log["Metrics/minimum_safe_stride_exact_ratio"] = (
        self._minimum_safe_stride_exact & minimum_stride_valid
      ).float().sum() / minimum_stride_valid_count
      log["Metrics/minimum_safe_stride_interval_valid_ratio"] = (
        self._minimum_safe_stride_interval_valid & minimum_stride_valid
      ).float().sum() / minimum_stride_valid_count
      log["Metrics/minimum_safe_stride_evidence_weight_mean"] = (
        self._minimum_safe_stride_weight * minimum_stride_valid.float()
      ).sum() / minimum_stride_valid_count
      log["Metrics/minimum_safe_stride_evidence_weighted_ratio"] = (
        minimum_stride_valid & (self._minimum_safe_stride_weight > 1.0)
      ).float().sum() / minimum_stride_valid_count
      log["Metrics/stair_skip_layer_penalty_mean"] = (
        self._stair_skip_layer_penalty.mean()
      )
      log["Metrics/minimum_safe_stride_valid_while_stair_ratio"] = (
        minimum_stride_valid & active_phase
      ).float().sum() / active_phase_count
      flat_count = (~active_phase).float().sum().clamp_min(1.0)
      log["Metrics/minimum_safe_stride_valid_while_flat_ratio"] = (
        minimum_stride_valid & ~active_phase
      ).float().sum() / flat_count
      log["Metrics/minimum_safe_stride_label_mean"] = (
        self._minimum_safe_stride * minimum_stride_valid.float()
      ).sum() / minimum_stride_valid_count
      interval_valid = self._minimum_safe_stride_interval_valid & minimum_stride_valid
      interval_valid_count = interval_valid.float().sum().clamp_min(1.0)
      log["Metrics/minimum_safe_stride_upper_mean"] = (
        self._minimum_safe_stride_upper * interval_valid.float()
      ).sum() / interval_valid_count
      log["Metrics/minimum_safe_stride_interval_width_mean"] = (
        (self._minimum_safe_stride_upper - self._minimum_safe_stride)
        * interval_valid.float()
      ).sum() / interval_valid_count
      safe_touchdown_count = safe_touchdown.float().sum().clamp_min(1.0)
      log["Metrics/stair_safe_stride_label_clamped_ratio"] = (
        safe_touchdown & safe_stride_label_clamped
      ).float().sum() / safe_touchdown_count
      log["Metrics/stair_safe_stride_raw_toe_front_s_mean"] = (
        toe_front_s * safe_touchdown.float()
      ).sum() / safe_touchdown_count
      log["Metrics/stair_safe_landing_raw_center_s_mean"] = (
        landing_center_s * safe_touchdown.float()
      ).sum() / safe_touchdown_count
      log["Metrics/landing_touchdown_now_ratio"] = (
        self._landing_touchdown_now.float().mean()
      )
      log["Metrics/landing_quality_now_mean"] = self._landing_quality_now.mean()
      log["Metrics/stair_touchdown_rejected_riser_ratio"] = (
        geometry_contact & target_riser_unsafe
      ).float().sum() / geometry_contact_count
      log["Metrics/stair_touchdown_rejected_slab_ratio"] = (
        geometry_contact & target_slab_unsafe
      ).float().sum() / geometry_contact_count
      log["Metrics/stair_touchdown_rejected_lip_ratio"] = (
        geometry_contact & target_lip_unsafe
      ).float().sum() / geometry_contact_count
      log["Metrics/stair_following_active_ratio"] = following.float().mean()
      log["Metrics/stair_context_active_ratio"] = active_phase.float().mean()
      log["Metrics/stair_depth_label_valid_ratio"] = (
        (self._stair_depth_label_valid & active_phase).float().mean()
      )
      log["Metrics/stair_depth_label_stair_coverage"] = (
        self._stair_depth_label_valid & active_phase
      ).float().sum() / active_phase_count
      p0_xy = boundaries[..., 0:2]
      p1_xy = boundaries[..., 3:5]
      segment_xy = p1_xy - p0_xy
      segment_len_sq = torch.sum(torch.square(segment_xy), dim=-1).clamp_min(1.0e-12)
      root_delta = root_xy[:, None, :] - p0_xy
      projection = torch.sum(root_delta * segment_xy, dim=-1) / segment_len_sq
      projection = torch.clamp(projection, 0.0, 1.0)
      closest_xy = p0_xy + projection[..., None] * segment_xy
      root_boundary_distance = torch.norm(root_xy[:, None, :] - closest_xy, dim=-1)
      occupancy_distance = 1.25 * tread_depth
      geometry_stair_occupancy = (
        level_active
        & shape_valid
        & torch.any(
          base_valid & (root_boundary_distance <= occupancy_distance[:, None]),
          dim=-1,
        )
      )
      log["Metrics/geometry_stair_occupancy_ratio"] = (
        geometry_stair_occupancy.float().mean()
      )
      log["Metrics/geometry_stair_while_phase_zero_ratio"] = (
        (geometry_stair_occupancy & ~active_phase).float().mean()
      )
      log["Metrics/stair_phase_while_geometry_flat_ratio"] = (
        (active_phase & ~geometry_stair_occupancy).float().mean()
      )
      log["Metrics/stair_following_past_last_exit_ratio"] = (
        past_last_boundary.float().sum() / active_phase_count
      )
      log["Metrics/stair_exit_candidate_ratio"] = (
        exit_candidate.float().sum() / active_phase_count
      )
      log["Metrics/stair_exit_flat_step_ratio"] = (
        flat_exit_step.float().sum() / active_phase_count
      )
      log["Metrics/stair_exit_confirmed_ratio"] = (
        confirmed_exit.float().sum() / active_phase_count
      )
      confirmed_exit_count = confirmed_exit.float().sum().clamp_min(1.0)
      log["Metrics/stair_completed_sequence_ratio"] = confirmed_exit.float().mean()
      log["Metrics/stair_completed_sequence_count"] = confirmed_exit.float().sum()
      log["Metrics/stair_completed_duration_mean"] = (
        self._context_steps.float() * confirmed_exit.float()
      ).sum() / confirmed_exit_count
      log["Metrics/stair_completed_repeat_layer1_hits_mean"] = (
        self._repeat_layer1_hit_count.float() * confirmed_exit.float()
      ).sum() / confirmed_exit_count
      log["Metrics/stair_exit_flat_steps_mean"] = (
        torch.where(
          active_phase,
          self._exit_flat_steps.float(),
          torch.zeros_like(self._exit_flat_steps, dtype=torch.float32),
        ).sum()
        / active_phase_count
      )
      safe_count = self._safe_stride_valid.float().sum().clamp_min(1.0)
      log["Metrics/safe_tread_lower_bound_mean"] = (
        self._safe_tread_lower_bound.sum() / safe_count
      )
      log["Metrics/safe_landing_center_s_mean"] = (
        self._safe_landing_center_s.sum() / safe_count
      )
      log["Metrics/observed_step_stride_mean"] = (
        self._observed_step_stride.sum() / safe_count
      )

      reset_mask = confirmed_exit | (active_phase & ~level_active) | stale_stride_reset
      reset_count = reset_mask.float().sum().clamp_min(1.0)
      log["Metrics/stair_phase_reset_event_ratio"] = (
        reset_mask.float().sum() / active_phase_count
      )
      log["Metrics/stair_phase_reset_population_ratio"] = reset_mask.float().mean()
      log["Metrics/stair_exit_reset_by_past_last_ratio"] = (
        confirmed_exit & past_last_boundary
      ).float().sum() / confirmed_exit_count
      log["Metrics/stair_exit_reset_by_level_ratio"] = (
        active_phase & ~level_active
      ).float().sum() / reset_count
      log["Metrics/stair_phase_reset_by_stale_stride_ratio"] = (
        stale_stride_reset.float().sum() / reset_count
      )
      if bool(torch.any(reset_mask).item()):
        reset_ids = reset_mask.nonzero(as_tuple=False).squeeze(-1)
        exit_ids = confirmed_exit.nonzero(as_tuple=False).squeeze(-1)
        self.reset(reset_ids)
        if bool(torch.any(confirmed_exit).item()):
          self._exit_event[exit_ids] = True
    else:
      zero = torch.zeros((), device=env.device)
      log["Metrics/toe_riser_contact_penalty_mean"] = zero
      log["Metrics/toe_riser_contact_time_mean"] = zero
      log["Metrics/stair_exit_candidate_ratio"] = zero
      log["Metrics/stair_exit_flat_step_ratio"] = zero
      log["Metrics/stair_exit_confirmed_ratio"] = zero
      log["Metrics/stair_exit_flat_steps_mean"] = zero
      log["Metrics/stair_phase_reset_event_ratio"] = zero
      log["Metrics/stair_phase_reset_population_ratio"] = zero
      log["Metrics/stair_exit_reset_by_past_last_ratio"] = zero
      log["Metrics/stair_exit_reset_by_level_ratio"] = zero
      log["Metrics/stair_phase_reset_by_stale_stride_ratio"] = zero

    total_penalty = slab_penalty + contact_penalty
    log["Metrics/toe_riser_total_penalty_mean"] = total_penalty.mean()
    log["Metrics/collision_risk_now_mean"] = self._collision_risk_now.mean()
    log["Metrics/stair_entry_active_ratio"] = (self._stair_phase == 1).float().mean()
    return total_penalty


class toe_step_riser_approach_penalty(toe_step_riser_slab_penalty):
  """Backward-compatible alias for the penalty-only stair entry term."""


def stair_skip_layer_penalty(env) -> torch.Tensor:
  """Return one-frame penalty magnitude for landing above the expected layer."""
  penalty = env.extras.get(STAIR_SKIP_LAYER_PENALTY_KEY)
  if not isinstance(penalty, torch.Tensor):
    return torch.zeros(env.num_envs, device=env.device)
  return penalty


class stair_aware_feet_gait:
  """Relax gait phase during heading-aligned stair entry and following."""

  def __init__(self, cfg: RewardTermCfg, env) -> None:
    del cfg
    self._phase_offset = torch.zeros(env.num_envs, device=env.device)
    self._was_stair_active = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._phase_offset[env_ids] = 0.0
    self._was_stair_active[env_ids] = False

  @staticmethod
  def _recovery_phase(is_contact: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    left_only = is_contact[:, 0] & ~is_contact[:, 1]
    right_only = is_contact[:, 1] & ~is_contact[:, 0]
    both = is_contact[:, 0] & is_contact[:, 1]
    phase = torch.where(left_only, torch.full_like(fallback, 0.25), fallback)
    phase = torch.where(right_only, torch.full_like(fallback, 0.75), phase)
    return torch.where(both, torch.zeros_like(phase), phase)

  @staticmethod
  def _phase_free_following_reward(is_contact: torch.Tensor) -> torch.Tensor:
    """Prefer single support without imposing a clock-driven foot phase."""
    contact_count = is_contact.float().sum(dim=1)
    return 0.5 * ((contact_count > 0).float() + (contact_count == 1).float())

  def __call__(
    self,
    env,
    period: float,
    offset: list[float],
    threshold: float,
    command_threshold: float,
    command_name: str,
    sensor_name: str,
    heading_cos: float = 0.70,
    asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_BODY_CFG,
  ) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    assert isinstance(sensor, ContactSensor)
    current_contact_time = sensor.data.current_contact_time
    assert current_contact_time is not None, (
      "Enable track_air_time=True for the stair gait contact sensor."
    )
    is_contact = current_contact_time > 0
    if is_contact.shape[-1] != 2:
      raise RuntimeError(
        "stair_aware_feet_gait requires exactly two foot contact channels, got "
        f"{is_contact.shape[-1]}."
      )

    asset: Entity = env.scene[asset_cfg.name]
    stage = env.extras.get(STAIR_PHASE_KEY)
    ascent_dir = env.extras.get(STAIR_ASCENT_DIR_KEY)
    recent_evidence = env.extras.get(STAIR_ENTRY_RECENT_EVIDENCE_KEY)
    evidence_ascent_dir = env.extras.get(STAIR_ENTRY_EVIDENCE_ASCENT_DIR_KEY)
    if stage is None or ascent_dir is None:
      stage = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
      ascent_dir = torch.zeros(env.num_envs, 2, device=env.device)
    if recent_evidence is None:
      recent_evidence = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    else:
      recent_evidence = recent_evidence.bool()
    if evidence_ascent_dir is None:
      evidence_ascent_dir = torch.zeros(env.num_envs, 2, device=env.device)
    heading_gate = (
      toe_step_riser_slab_penalty._base_heading_cos(
        asset.data.root_link_quat_w,
        ascent_dir,
      )
      >= heading_cos
    )
    evidence_heading_gate = (
      toe_step_riser_slab_penalty._base_heading_cos(
        asset.data.root_link_quat_w,
        evidence_ascent_dir,
      )
      >= heading_cos
    )
    entry_active = (stage == 1) & heading_gate
    following_active = (stage == 2) & heading_gate
    recent_active = recent_evidence & evidence_heading_gate
    stair_gait_active = entry_active | following_active | recent_active

    raw_phase = (env.episode_length_buf * env.step_dt) / period
    exit_stair = self._was_stair_active & ~stair_gait_active
    if bool(torch.any(exit_stair).item()):
      desired_phase = self._recovery_phase(is_contact, raw_phase % 1.0)
      self._phase_offset[exit_stair] = (
        desired_phase[exit_stair] - raw_phase[exit_stair]
      ) % 1.0

    global_phase = (raw_phase + self._phase_offset).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(
      1, -1
    )
    normal_stance = ((global_phase + offsets) % 1.0) < threshold
    normal_reward = (normal_stance == is_contact).float().mean(dim=1)

    contact_count = is_contact.float().sum(dim=1)
    stair_reward = self._phase_free_following_reward(is_contact)

    command = env.command_manager.get_command(command_name)
    if command is not None:
      total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      normal_reward *= (total_command > command_threshold).float()

    reward = torch.where(stair_gait_active, stair_reward, normal_reward)
    self._was_stair_active.copy_(stair_gait_active)

    log = env.extras["log"]
    log["Metrics/stair_entry_gait_active_ratio"] = entry_active.float().mean()
    log["Metrics/stair_recent_gait_active_ratio"] = recent_active.float().mean()
    log["Metrics/stair_following_gait_active_ratio"] = following_active.float().mean()
    log["Metrics/stair_gait_active_ratio"] = stair_gait_active.float().mean()
    stair_stage = stage > 0
    stair_context = stair_stage | recent_evidence
    context_heading_gate = (stair_stage & heading_gate) | (
      recent_evidence & evidence_heading_gate
    )
    log["Metrics/stair_gait_heading_gate_ratio"] = (
      context_heading_gate.float().sum() / stair_context.float().sum().clamp_min(1.0)
    )
    log["Metrics/stair_gait_single_support_ratio"] = torch.where(
      stair_gait_active,
      (contact_count == 1).float(),
      torch.zeros_like(contact_count),
    ).sum() / stair_gait_active.float().sum().clamp_min(1.0)
    return reward


class stair_tread_landing_reward(_StepBoundaryFootVolume):
  """Reward stage-2 footfalls near the privileged tread safe target."""

  def __init__(self, cfg: RewardTermCfg, env) -> None:
    super().__init__(cfg, env)

  @staticmethod
  def _safe_landing_target(
    safe_center_s: torch.Tensor,
    tread_depth: torch.Tensor,
    lead: float,
    back_margin: float,
    front_margin: float,
  ) -> torch.Tensor:
    """Advance a validated center while keeping the target in the safe band."""
    max_target_s = (tread_depth - front_margin).clamp_min(back_margin)
    return torch.minimum(
      torch.clamp_min(safe_center_s + lead, back_margin),
      max_target_s,
    )

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
    landing_lead: float = 0.04,
    sigma_fraction: float = 0.15,
    min_sigma: float = 0.03,
    back_margin: float = 0.06,
    front_margin: float = 0.08,
    lateral_margin: float = 0.03,
    height_tolerance: float = 0.08,
    support_deficit_scale: float = 0.40,
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
    heading_cos: float = 0.70,
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
    safe_landing_center = env.extras.get(SAFE_LANDING_CENTER_KEY)
    minimum_stride_valid = env.extras.get(MINIMUM_SAFE_STRIDE_VALID_KEY)
    if safe_landing_center is None or minimum_stride_valid is None:
      safe_landing_center = torch.zeros(env.num_envs, device=env.device)
      minimum_stride_valid = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.bool
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
    edge_clearance = lip_edge_radius + lip_margin
    effective_back_margin = max(back_margin, edge_clearance)
    effective_front_margin = max(front_margin, edge_clearance)
    desired_s_env = self._safe_landing_target(
      safe_landing_center,
      tread_depth,
      landing_lead,
      effective_back_margin,
      effective_front_margin,
    )
    depth = tread_depth[:, None, None]
    desired_s = desired_s_env[:, None, None]
    sigma = torch.clamp(sigma_fraction * depth, min=min_sigma)
    height_error = torch.abs(foot_pos_w[:, :, None, 2] - boundaries[:, None, :, 10])
    boundary_sequence_ids, boundary_layers = _current_step_boundary_metadata(env)
    active_sequence_id = env.extras.get(STAIR_LANDING_SEQUENCE_ID_KEY)
    if boundary_sequence_ids is None or active_sequence_id is None:
      active_sequence_boundary = torch.ones_like(valid_boundaries, dtype=torch.bool)
    else:
      active_sequence_boundary = boundary_sequence_ids == active_sequence_id[:, None]
    candidate = (
      valid_boundaries[:, None, :]
      & active_sequence_boundary[:, None, :]
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

    if boundary_layers is None:
      boundary_layers = _step_boundary_layers(
        boundaries, valid_boundaries, boundaries.shape[1]
      )
    boundary_layers = torch.where(
      valid_boundaries, boundary_layers, torch.zeros_like(boundary_layers)
    )
    expanded_layers = boundary_layers[:, None, :].expand(env.num_envs, num_feet, -1)
    best_boundary_layer = torch.gather(
      expanded_layers, dim=-1, index=best_idx[..., None]
    ).squeeze(-1)
    expected_layer = env.extras.get(STAIR_LANDING_EXPECTED_LAYER_KEY)
    target_foot = env.extras.get(STAIR_LANDING_TARGET_FOOT_KEY)
    if expected_layer is None or target_foot is None:
      landing_layer_valid = best_boundary_layer >= min_landing_layer
      target_foot_gate = torch.ones_like(first_contact, dtype=torch.bool)
    else:
      landing_layer_valid = best_boundary_layer == expected_layer[:, None]
      foot_ids = torch.arange(num_feet, device=env.device)
      target_foot_gate = foot_ids[None, :] == target_foot[:, None]

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

    ascent_dir = env.extras.get(STAIR_ASCENT_DIR_KEY)
    if expected_layer is None or ascent_dir is None:
      stage2 = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
      heading_gate = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    else:
      stage2 = expected_layer > 0
      heading_gate = (
        toe_step_riser_slab_penalty._base_heading_cos(
          asset.data.root_link_quat_w,
          ascent_dir,
        )
        >= heading_cos
      )
    env_active = (
      stage2
      & minimum_stride_valid.bool()
      & heading_gate
      & shape_valid
      & _terrain_level_active(env, min_terrain_level)
    )
    safety_candidate = (
      first_contact
      & env_active[:, None]
      & has_candidate
      & landing_layer_valid
      & target_foot_gate
    )
    touchdown_candidate = safety_candidate & ~riser_contact & ~slab_unsafe & ~lip_unsafe
    sole_z = torch.min(self._local_points[:, 2])
    sole_points_w = foot_points_w[:, :, self._local_points[:, 2] <= sole_z + 1.0e-6, :]
    support_fraction = toe_step_riser_slab_penalty._tread_support_fraction(
      sole_points_w,
      boundaries,
      tread_depth,
    )
    best_support_fraction = torch.gather(
      support_fraction, dim=-1, index=best_idx[..., None]
    ).squeeze(-1)
    back_clearance_score = torch.clamp(best_s / effective_back_margin, min=0.0, max=1.0)
    front_clearance_score = torch.clamp(
      (depth.squeeze(-1) - best_s) / effective_front_margin,
      min=0.0,
      max=1.0,
    )
    edge_clearance_score = torch.minimum(back_clearance_score, front_clearance_score)
    landing_quality = _landing_quality(
      best_support_fraction,
      best_score,
      edge_clearance_score,
      touchdown_candidate,
    )
    support_deficit = torch.relu(0.50 - best_support_fraction) / 0.50
    per_foot = landing_quality - (
      support_deficit_scale * support_deficit * touchdown_candidate.float()
    )
    landing_active = touchdown_candidate

    touchdown_now = torch.any(first_contact & env_active[:, None], dim=-1)
    quality_now = torch.max(landing_quality, dim=-1).values
    cached_touchdown = env.extras.get(LANDING_TOUCHDOWN_KEY)
    cached_quality = env.extras.get(LANDING_QUALITY_KEY)
    if isinstance(cached_touchdown, torch.Tensor):
      cached_touchdown.logical_or_(touchdown_now)
    else:
      env.extras[LANDING_TOUCHDOWN_KEY] = touchdown_now
    if isinstance(cached_quality, torch.Tensor):
      cached_quality.copy_(torch.maximum(cached_quality, quality_now))
    else:
      env.extras[LANDING_QUALITY_KEY] = quality_now

    log = env.extras["log"]
    landing_count = landing_active.float().sum().clamp_min(1.0)
    safety_count = safety_candidate.float().sum().clamp_min(1.0)
    log["Metrics/stair_tread_landing_active_ratio"] = landing_active.float().mean()
    log["Metrics/stair_landing_active_ratio"] = landing_active.float().mean()
    log["Metrics/stair_landing_heading_gate_ratio"] = (
      stage2 & heading_gate
    ).float().sum() / stage2.float().sum().clamp_min(1.0)
    log["Metrics/stair_landing_safe_memory_valid_ratio"] = (
      stage2 & minimum_stride_valid.bool()
    ).float().sum() / stage2.float().sum().clamp_min(1.0)
    log["Metrics/stair_landing_target_s_mean"] = torch.where(
      env_active,
      desired_s_env,
      torch.zeros_like(desired_s_env),
    ).sum() / env_active.float().sum().clamp_min(1.0)
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
    log["Metrics/stair_landing_riser_collision_ratio"] = (
      safety_candidate & riser_contact
    ).float().sum() / safety_count
    log["Metrics/stair_tread_landing_score_mean"] = (
      landing_quality * landing_active.float()
    ).sum() / landing_count
    log["Metrics/stair_landing_support_fraction_mean"] = (
      best_support_fraction * safety_candidate.float()
    ).sum() / safety_count
    log["Metrics/stair_landing_support_below_50_ratio"] = (
      safety_candidate & (best_support_fraction < 0.50)
    ).float().sum() / safety_count
    log["Metrics/landing_touchdown_now_ratio"] = touchdown_now.float().mean()
    log["Metrics/landing_quality_now_mean"] = quality_now.mean()
    return per_foot.sum(dim=-1)
