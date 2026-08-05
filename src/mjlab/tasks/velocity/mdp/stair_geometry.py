"""Shared privileged stair geometry helpers for training-only signals."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

STAIR_PHASE_KEY = "slow_latent_stair_phase"
STAIR_SEQUENCE_ID_KEY = "slow_latent_stair_sequence_id"
STAIR_ENTRY_EVENT_KEY = "slow_latent_stair_entry_event"
STAIR_EXIT_EVENT_KEY = "slow_latent_stair_exit_event"
TOE_RISER_NEW_HIT_KEY = "slow_latent_toe_riser_new_hit"
TOE_RISER_CONTACT_KEY = "slow_latent_toe_riser_contact"
TOE_RISER_NEW_HIT_BY_FOOT_KEY = "slow_latent_toe_riser_new_hit_by_foot"
TOE_RISER_CONTACT_BY_FOOT_KEY = "slow_latent_toe_riser_contact_by_foot"
STAIR_TOE_RISER_HIT_LAYER_KEY = "slow_latent_stair_toe_riser_hit_layer"
STAIR_ENTRY_RECENT_EVIDENCE_KEY = "slow_latent_stair_entry_recent_evidence"
STAIR_ENTRY_EVIDENCE_ASCENT_DIR_KEY = "slow_latent_stair_entry_evidence_ascent_dir"
STAIR_TARGET_FOOT_KEY = "slow_latent_stair_target_foot"
STAIR_EXPECTED_LAYER_KEY = "slow_latent_stair_expected_layer"
STAIR_LANDING_EXPECTED_LAYER_KEY = "slow_latent_stair_landing_expected_layer"
STAIR_LANDING_SEQUENCE_ID_KEY = "slow_latent_stair_landing_sequence_id"
STAIR_LANDING_TARGET_FOOT_KEY = "slow_latent_stair_landing_target_foot"
STAIR_CLEARANCE_FOOT_LAYERS_KEY = "slow_latent_stair_clearance_foot_layers"
STAIR_CLEARANCE_FOOT_LAYERS_VALID_KEY = "slow_latent_stair_clearance_foot_layers_valid"
STAIR_CLEARANCE_SEQUENCE_ID_KEY = "slow_latent_stair_clearance_sequence_id"
STAIR_CLEARANCE_ASCENT_DIR_KEY = "slow_latent_stair_clearance_ascent_dir"
STAIR_ASCENT_DIR_KEY = "slow_latent_stair_ascent_dir"
STAIR_TREAD_DEPTH_LABEL_KEY = "slow_latent_stair_tread_depth_label"
STAIR_SAME_FOOT_STRIDE_LABEL_KEY = "slow_latent_stair_same_foot_stride_label"
STAIR_RISER_HEIGHT_LABEL_KEY = "slow_latent_stair_riser_height_label"
STAIR_SHAPE_LABEL_VALID_KEY = "slow_latent_stair_shape_label_valid"
STAIR_DEPTH_LABEL_VALID_KEY = "slow_latent_stair_depth_label_valid"
STAIR_DEPTH_CONFIRMATION_EVENT_KEY = "slow_latent_stair_depth_confirmation_event"
STAIR_DEPTH_CONFIRMATION_AGE_KEY = "slow_latent_stair_depth_confirmation_age"
STAIR_CONFIRMATION_CONDITION_KEY = "slow_latent_stair_confirmation_condition"
STAIR_CONFIRMATION_LAYER_KEY = "slow_latent_stair_confirmation_layer"
STAIR_CONFIRMATION_CONTACT_SEQUENCE_KEY = (
  "slow_latent_stair_confirmation_contact_sequence"
)
STAIR_ORACLE_EXPECTED_RISER_CONTACT_KEY = (
  "slow_latent_stair_oracle_expected_riser_contact"
)
STAIR_ORACLE_EXPECTED_RISER_LAYER_KEY = "slow_latent_stair_oracle_expected_riser_layer"
STAIR_CURRENT_GROUND_CONTACT_KEY = "slow_latent_stair_current_ground_contact"
STAIR_CURRENT_STAIR_SUPPORT_KEY = "slow_latent_stair_current_stair_support"
STAIR_CURRENT_SUPPORT_FRACTION_KEY = "slow_latent_stair_current_support_fraction"
STAIR_CURRENT_SUPPORT_LAYER_KEY = "slow_latent_stair_current_support_layer"
STAIR_CURRENT_CONTACT_DURATION_KEY = "slow_latent_stair_current_contact_duration"
STAIR_GEOMETRY_OCCUPANCY_KEY = "slow_latent_stair_geometry_occupancy"
STAIR_ADJACENT_PAIR_DEPTH_KEY = "slow_latent_stair_adjacent_pair_depth"
STAIR_ADJACENT_PAIR_HEIGHT_KEY = "slow_latent_stair_adjacent_pair_height"
STAIR_ADJACENT_PAIR_VALID_KEY = "slow_latent_stair_adjacent_pair_valid"
STAIR_ADJACENT_PAIR_EVENT_KEY = "slow_latent_stair_adjacent_pair_event"
STAIR_ORACLE_CONTACT_VALID_BY_FOOT_KEY = (
  "slow_latent_stair_oracle_contact_valid_by_foot"
)
STAIR_ORACLE_CONTACT_LAYER_BY_FOOT_KEY = (
  "slow_latent_stair_oracle_contact_layer_by_foot"
)
STAIR_ORACLE_CONTACT_SEQUENCE_BY_FOOT_KEY = (
  "slow_latent_stair_oracle_contact_sequence_by_foot"
)
STAIR_ORACLE_CONTACT_POINT_BY_FOOT_KEY = (
  "slow_latent_stair_oracle_contact_point_by_foot"
)
STAIR_ORACLE_CONTACT_NORMAL_BY_FOOT_KEY = (
  "slow_latent_stair_oracle_contact_normal_by_foot"
)
STAIR_ORACLE_CONTACT_FORCE_BY_FOOT_KEY = (
  "slow_latent_stair_oracle_contact_force_by_foot"
)
STAIR_ORACLE_CONTACT_S_BY_FOOT_KEY = "slow_latent_stair_oracle_contact_s_by_foot"
STAIR_ORACLE_TOE_S_BY_FOOT_KEY = "slow_latent_stair_oracle_toe_s_by_foot"
STAIR_ORACLE_HEEL_S_BY_FOOT_KEY = "slow_latent_stair_oracle_heel_s_by_foot"
STAIR_ORACLE_FOOT_CENTER_S_BY_FOOT_KEY = (
  "slow_latent_stair_oracle_foot_center_s_by_foot"
)
STAIR_ORACLE_RISER_S_BY_FOOT_KEY = "slow_latent_stair_oracle_riser_s_by_foot"
STAIR_ORACLE_ROOT_S_BY_FOOT_KEY = "slow_latent_stair_oracle_root_s_by_foot"
SAFE_STRIDE_VALID_KEY = "slow_latent_safe_stride_valid"
SAFE_TREAD_LOWER_BOUND_KEY = "slow_latent_safe_tread_lower_bound"
MINIMUM_SAFE_STRIDE_KEY = "slow_latent_minimum_safe_stride"
MINIMUM_SAFE_STRIDE_UPPER_KEY = "slow_latent_minimum_safe_stride_upper"
MINIMUM_SAFE_STRIDE_RAW_KEY = "slow_latent_minimum_safe_stride_raw"
MINIMUM_SAFE_STRIDE_VALID_KEY = "slow_latent_minimum_safe_stride_valid"
MINIMUM_SAFE_STRIDE_INTERVAL_VALID_KEY = (
  "slow_latent_minimum_safe_stride_interval_valid"
)
MINIMUM_SAFE_STRIDE_EXACT_KEY = "slow_latent_minimum_safe_stride_exact"
MINIMUM_SAFE_STRIDE_WEIGHT_KEY = "slow_latent_minimum_safe_stride_weight"
STAIR_SKIP_LAYER_PENALTY_KEY = "slow_latent_stair_skip_layer_penalty"
SAFE_LANDING_CENTER_KEY = "slow_latent_safe_landing_center_s"
OBSERVED_STEP_STRIDE_KEY = "slow_latent_observed_step_stride"
COLLISION_RISK_KEY = "slow_latent_collision_risk_now"
LANDING_TOUCHDOWN_KEY = "slow_latent_landing_touchdown_now"
LANDING_QUALITY_KEY = "slow_latent_landing_quality_now"
STAIR_PROBE_STRIDE_REWARD_KEY = "slow_latent_stair_probe_stride_reward"
STAIR_PROBE_STRIDE_REWARD_EVENT_ID_KEY = (
  "slow_latent_stair_probe_stride_reward_event_id"
)
_STAIR_SHAPE_CACHE_KEY = "_privileged_stair_shape_cache"


def stair_shape_from_boundaries(
  boundaries: torch.Tensor,
  valid_boundaries: torch.Tensor,
  normal_alignment_threshold: float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Extract tread depth and riser height from step-boundary geometry.

  Each boundary stores ``[p0, p1, normal_to_low, z_low, z_high]``. Parallel
  boundaries with the same low-side normal belong to successive stair layers;
  their minimum positive normal separation is the tread depth.
  """
  p0_xy = boundaries[..., 0:2]
  normals_xy = boundaries[..., 6:8]
  normals_xy = normals_xy / torch.norm(normals_xy, dim=-1, keepdim=True).clamp_min(
    1.0e-6
  )

  normal_alignment = torch.sum(
    normals_xy[:, :, None, :] * normals_xy[:, None, :, :], dim=-1
  )
  p0_delta = p0_xy[:, None, :, :] - p0_xy[:, :, None, :]
  normal_separation = torch.abs(torch.sum(p0_delta * normals_xy[:, :, None, :], dim=-1))
  pair_valid = (
    valid_boundaries[:, :, None]
    & valid_boundaries[:, None, :]
    & (normal_alignment >= normal_alignment_threshold)
    & (normal_separation > 1.0e-4)
  )
  tread_depth = torch.amin(
    torch.where(
      pair_valid,
      normal_separation,
      torch.full_like(normal_separation, torch.inf),
    ),
    dim=(1, 2),
  )

  boundary_heights = torch.abs(boundaries[..., 10] - boundaries[..., 9])
  height_valid = valid_boundaries & (boundary_heights > 1.0e-4)
  riser_height = torch.min(
    torch.where(
      height_valid,
      boundary_heights,
      torch.full_like(boundary_heights, torch.inf),
    ),
    dim=-1,
  ).values

  shape_valid = torch.isfinite(tread_depth) & torch.isfinite(riser_height)
  tread_depth = torch.where(shape_valid, tread_depth, torch.zeros_like(tread_depth))
  riser_height = torch.where(shape_valid, riser_height, torch.zeros_like(riser_height))
  return tread_depth, riser_height, shape_valid


def stair_shape_for_sequence(
  boundaries: torch.Tensor,
  valid_boundaries: torch.Tensor,
  boundary_sequence_ids: torch.Tensor,
  sequence_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Extract static stair geometry from each environment's active sequence."""
  sequence_boundaries = valid_boundaries & (
    boundary_sequence_ids == sequence_ids[:, None]
  )
  return stair_shape_from_boundaries(boundaries, sequence_boundaries)


def cached_stair_shape(
  env: ManagerBasedRlEnv,
  boundaries: torch.Tensor,
  valid_boundaries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Cache static per-tile shape so pairwise extraction runs only on tile changes."""
  terrain = env.scene.terrain
  levels = getattr(terrain, "terrain_levels", None)
  terrain_types = getattr(terrain, "terrain_types", None)
  if not isinstance(levels, torch.Tensor) or not isinstance(
    terrain_types, torch.Tensor
  ):
    return stair_shape_from_boundaries(boundaries, valid_boundaries)
  cache = env.extras.get(_STAIR_SHAPE_CACHE_KEY)
  if not isinstance(cache, dict):
    cache = {
      "levels": torch.full_like(levels, -1),
      "types": torch.full_like(terrain_types, -1),
      "tread_depth": torch.zeros(env.num_envs, device=env.device),
      "riser_height": torch.zeros(env.num_envs, device=env.device),
      "valid": torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
    }
    env.extras[_STAIR_SHAPE_CACHE_KEY] = cache

  typed_cache = cast(dict[str, torch.Tensor], cache)
  changed = (typed_cache["levels"] != levels) | (typed_cache["types"] != terrain_types)
  if bool(torch.any(changed).item()):
    env_ids = changed.nonzero(as_tuple=False).squeeze(-1)
    tread_depth, riser_height, shape_valid = stair_shape_from_boundaries(
      boundaries[env_ids], valid_boundaries[env_ids]
    )
    typed_cache["tread_depth"][env_ids] = tread_depth
    typed_cache["riser_height"][env_ids] = riser_height
    typed_cache["valid"][env_ids] = shape_valid
    typed_cache["levels"][env_ids] = levels[env_ids]
    typed_cache["types"][env_ids] = terrain_types[env_ids]

  return (
    typed_cache["tread_depth"],
    typed_cache["riser_height"],
    typed_cache["valid"],
  )
