"""Shared privileged stair geometry helpers for training-only signals."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

PROBE_STAGE_KEY = "slow_latent_probe_stage"
PROBE_TARGET_FOOT_KEY = "slow_latent_probe_target_foot"
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
