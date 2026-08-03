from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")
_MIXED_REPLAY_ACTIVE_KEY = "terrain_mixed_replay_active"
_STANDALONE_REPLAY_ACTIVE_KEY = "terrain_standalone_replay_active"
_MIXED_REPLAY_BUCKET_NAMES = ("low", "mid", "high")


def _terrain_family_name(name: str) -> str:
  """Collapse fixed-width terrain variants into their aggregate family name."""
  family, separator, variant = name.rpartition("_w")
  if separator and variant.isdigit():
    return family
  return name


def _validate_mixed_replay_cfg(
  level_ranges: tuple[tuple[int, int], ...],
  weights: tuple[float, ...],
) -> None:
  if len(level_ranges) != len(weights):
    raise ValueError("mixed replay level ranges and weights must have the same length")
  if len(level_ranges) == 0:
    raise ValueError("mixed replay needs at least one level range")
  if any(weight < 0.0 for weight in weights):
    raise ValueError("mixed replay weights must be non-negative")
  if sum(weights) <= 0.0:
    raise ValueError("mixed replay needs at least one positive weight")


def _clamp_mixed_replay_ranges(
  level_ranges: tuple[tuple[int, int], ...],
  num_levels: int,
) -> tuple[tuple[int, int], ...]:
  clamped_ranges: list[tuple[int, int]] = []
  for raw_min, raw_max in level_ranges:
    raw_low = min(int(raw_min), int(raw_max))
    raw_high = max(int(raw_min), int(raw_max))
    min_level = max(0, min(num_levels - 1, raw_low))
    max_level = max(0, min(num_levels - 1, raw_high))
    clamped_ranges.append((min_level, max_level))
  return tuple(clamped_ranges)


def _sample_mixed_replay_levels(
  num_samples: int,
  level_ranges: tuple[tuple[int, int], ...],
  weights: tuple[float, ...],
  device: torch.device,
) -> torch.Tensor:
  weights_tensor = torch.tensor(weights, dtype=torch.float, device=device)
  bucket_ids = torch.multinomial(
    weights_tensor / torch.sum(weights_tensor),
    num_samples,
    replacement=True,
  )
  ranges_tensor = torch.tensor(level_ranges, dtype=torch.long, device=device)
  lows = ranges_tensor[:, 0][bucket_ids]
  highs = ranges_tensor[:, 1][bucket_ids]
  spans = highs - lows + 1
  offsets = torch.floor(torch.rand(num_samples, device=device) * spans.float()).long()
  return lows + offsets


def _mixed_replay_bucket_ratios(
  levels: torch.Tensor,
  level_ranges: tuple[tuple[int, int], ...],
) -> dict[str, torch.Tensor]:
  result: dict[str, torch.Tensor] = {}
  if levels.numel() == 0:
    zero = torch.tensor(0.0, device=levels.device)
    for i in range(len(level_ranges)):
      name = _MIXED_REPLAY_BUCKET_NAMES[i] if i < 3 else f"bucket_{i}"
      result[f"mixed_replay_{name}_ratio"] = zero
    return result

  for i, (min_level, max_level) in enumerate(level_ranges):
    name = _MIXED_REPLAY_BUCKET_NAMES[i] if i < 3 else f"bucket_{i}"
    in_bucket = (levels >= min_level) & (levels <= max_level)
    result[f"mixed_replay_{name}_ratio"] = torch.mean(in_bucket.float())
  return result


def _apply_mixed_terrain_replay(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  terrain,
  mixed_replay_start_level: int | None,
  mixed_replay_level_ranges: tuple[tuple[int, int], ...],
  mixed_replay_weights: tuple[float, ...],
  activation_levels: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
  if mixed_replay_start_level is None:
    return {}

  terrain_origins = terrain.terrain_origins
  if terrain_origins is None:
    return {}

  _validate_mixed_replay_cfg(mixed_replay_level_ranges, mixed_replay_weights)

  num_levels = terrain_origins.shape[0]
  clamped_ranges = _clamp_mixed_replay_ranges(
    mixed_replay_level_ranges,
    num_levels,
  )
  start_level = max(0, min(mixed_replay_start_level, num_levels - 1))

  extras = getattr(env, "extras", None)
  if extras is None:
    extras = {}
    env.extras = extras

  active = extras.get(_MIXED_REPLAY_ACTIVE_KEY)
  if (
    not isinstance(active, torch.Tensor) or active.shape != terrain.terrain_levels.shape
  ):
    active = torch.zeros_like(terrain.terrain_levels, dtype=torch.bool)
    extras[_MIXED_REPLAY_ACTIVE_KEY] = active

  if activation_levels is None:
    effective_activation_levels = terrain.terrain_levels[env_ids]
  else:
    effective_activation_levels = activation_levels
  active[env_ids] |= effective_activation_levels >= start_level
  replay_env_ids = env_ids[active[env_ids]]

  if replay_env_ids.numel() > 0:
    sampled_levels = _sample_mixed_replay_levels(
      int(replay_env_ids.numel()),
      clamped_ranges,
      mixed_replay_weights,
      replay_env_ids.device,
    )
    terrain.terrain_levels[replay_env_ids] = sampled_levels
    sampled_origins = terrain_origins[
      sampled_levels,
      terrain.terrain_types[replay_env_ids],
    ]
    terrain_env_origins = getattr(terrain, "env_origins", None)
    if terrain_env_origins is not None:
      terrain_env_origins[replay_env_ids] = sampled_origins
    scene_env_origins = getattr(env.scene, "env_origins", None)
    if scene_env_origins is not None:
      scene_env_origins[replay_env_ids] = sampled_origins

  active_levels = terrain.terrain_levels[active]
  result = {
    "mixed_replay_active": torch.mean(active.float()),
  }
  result.update(_mixed_replay_bucket_ratios(active_levels, clamped_ranges))
  return result


def _sample_terrain_types_from_mask(
  type_mask: torch.Tensor,
  type_proportions: torch.Tensor,
  num_samples: int,
) -> torch.Tensor:
  type_ids = torch.nonzero(type_mask, as_tuple=False).flatten()
  weights = type_proportions[type_ids].to(dtype=torch.float)
  if torch.sum(weights) <= 0.0:
    weights = torch.ones_like(weights)
  choices = torch.multinomial(
    weights / torch.sum(weights),
    num_samples,
    replacement=True,
  )
  return type_ids[choices]


def _standalone_spawn_probability(
  standalone_mask: torch.Tensor,
  type_proportions: torch.Tensor,
  explicit_probability: float | None,
) -> float:
  if explicit_probability is not None:
    return max(0.0, min(1.0, float(explicit_probability)))

  standalone_weight = torch.sum(type_proportions[standalone_mask])
  grid_weight = torch.sum(type_proportions[~standalone_mask])
  total = standalone_weight + grid_weight
  if total <= 0.0:
    return 0.5
  return float((standalone_weight / total).item())


def _apply_standalone_terrain_replay(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  terrain,
  standalone_replay_start_level: int | None,
  standalone_replay_probability: float | None = None,
  activation_levels: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
  if standalone_replay_start_level is None:
    return {}

  terrain_origins = terrain.terrain_origins
  if terrain_origins is None:
    return {}

  standalone_mask = getattr(terrain, "standalone_terrain_type_mask", None)
  if not isinstance(standalone_mask, torch.Tensor):
    return {}
  num_levels, num_cols = terrain_origins.shape[:2]
  if standalone_mask.numel() != num_cols:
    return {}

  device = terrain.terrain_types.device
  standalone_mask = standalone_mask.to(device=device, dtype=torch.bool)
  if not standalone_mask.any() or not (~standalone_mask).any():
    return {}

  start_level = max(0, min(int(standalone_replay_start_level), num_levels - 1))
  type_proportions = getattr(terrain, "terrain_type_proportions", None)
  if (
    not isinstance(type_proportions, torch.Tensor)
    or type_proportions.numel() != num_cols
  ):
    type_proportions = torch.ones(num_cols, device=device, dtype=torch.float)
  else:
    type_proportions = type_proportions.to(device=device, dtype=torch.float)
  spawn_probability = _standalone_spawn_probability(
    standalone_mask,
    type_proportions,
    standalone_replay_probability,
  )

  extras = getattr(env, "extras", None)
  if extras is None:
    extras = {}
    env.extras = extras

  active = extras.get(_STANDALONE_REPLAY_ACTIVE_KEY)
  if (
    not isinstance(active, torch.Tensor) or active.shape != terrain.terrain_levels.shape
  ):
    active = torch.zeros_like(terrain.terrain_levels, dtype=torch.bool)
    extras[_STANDALONE_REPLAY_ACTIVE_KEY] = active

  if activation_levels is None:
    effective_activation_levels = terrain.terrain_levels[env_ids]
  else:
    effective_activation_levels = activation_levels
  active[env_ids] = effective_activation_levels >= start_level
  active_env_ids = env_ids[active[env_ids]]

  if active_env_ids.numel() > 0:
    replay_count = int(active_env_ids.numel())
    use_standalone = torch.rand(replay_count, device=device) < spawn_probability
    sampled_types = torch.empty(replay_count, device=device, dtype=torch.long)
    if use_standalone.any():
      sampled_types[use_standalone] = _sample_terrain_types_from_mask(
        standalone_mask,
        type_proportions,
        int(use_standalone.sum().item()),
      )
    if (~use_standalone).any():
      sampled_types[~use_standalone] = _sample_terrain_types_from_mask(
        ~standalone_mask,
        type_proportions,
        int((~use_standalone).sum().item()),
      )
    terrain.terrain_types[active_env_ids] = sampled_types

  # Defensive cleanup: envs that have not crossed the gate should remain on
  # regular grid terrains even if an earlier setup assigned them a standalone type.
  inactive_env_ids = env_ids[~active[env_ids]]
  if inactive_env_ids.numel() > 0:
    inactive_types = terrain.terrain_types[inactive_env_ids]
    inactive_standalone = standalone_mask[inactive_types]
    if inactive_standalone.any():
      cleanup_ids = inactive_env_ids[inactive_standalone]
      terrain.terrain_types[cleanup_ids] = _sample_terrain_types_from_mask(
        ~standalone_mask,
        type_proportions,
        int(cleanup_ids.numel()),
      )

  changed_ids = env_ids
  origins = terrain_origins[
    terrain.terrain_levels[changed_ids],
    terrain.terrain_types[changed_ids],
  ]
  terrain_env_origins = getattr(terrain, "env_origins", None)
  if terrain_env_origins is not None:
    terrain_env_origins[changed_ids] = origins
  scene_env_origins = getattr(env.scene, "env_origins", None)
  if scene_env_origins is not None:
    scene_env_origins[changed_ids] = origins

  zero = torch.tensor(0.0, device=device)
  active_types = terrain.terrain_types[active]
  spawn_ratio = (
    torch.mean(standalone_mask[active_types].float()) if active_types.numel() else zero
  )
  current_standalone = standalone_mask[terrain.terrain_types]
  levels = terrain.terrain_levels.float()
  standalone_level_mean = (
    torch.mean(levels[current_standalone]) if current_standalone.any() else zero
  )
  grid_level_mean = (
    torch.mean(levels[~current_standalone]) if (~current_standalone).any() else zero
  )
  return {
    "standalone_replay_active": torch.mean(active.float()),
    "standalone_replay_spawn_ratio": spawn_ratio,
    "standalone_replay_probability": torch.tensor(spawn_probability, device=device),
    "standalone_replay_level_mean": standalone_level_mean,
    "standalone_replay_grid_level_mean": grid_level_mean,
  }


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
  mixed_replay_start_level: int | None = None,
  mixed_replay_level_ranges: tuple[tuple[int, int], ...] = ((0, 2), (3, 5), (6, 9)),
  mixed_replay_weights: tuple[float, ...] = (0.2, 0.3, 0.5),
  standalone_replay_start_level: int | None = None,
  standalone_replay_probability: float | None = None,
) -> dict[str, torch.Tensor]:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  command = command_term.command
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  # Robots that walked far enough progress to harder terrains.
  move_up_by_distance = distance > terrain_generator.size[0] / 2
  move_up = move_up_by_distance

  target_attempted = getattr(command_term, "target_command_in_episode", None)
  target_reached = getattr(command_term, "target_reached_in_episode", None)
  if isinstance(target_attempted, torch.Tensor) and isinstance(
    target_reached, torch.Tensor
  ):
    target_attempted = target_attempted[env_ids].bool()
    target_reached = target_reached[env_ids].bool()
    move_up = torch.where(target_attempted, target_reached, move_up_by_distance)
  else:
    target_attempted = None
    target_reached = None

  # Robots that walked less than half of their required distance go to
  # simpler terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  levels_before_update = terrain.terrain_levels[env_ids].clone()
  terrain.update_env_origins(env_ids, move_up, move_down)
  activation_levels = torch.maximum(
    levels_before_update,
    terrain.terrain_levels[env_ids],
  )
  mixed_replay_result = _apply_mixed_terrain_replay(
    env,
    env_ids,
    terrain,
    mixed_replay_start_level,
    mixed_replay_level_ranges,
    mixed_replay_weights,
    activation_levels,
  )
  standalone_replay_result = _apply_standalone_terrain_replay(
    env,
    env_ids,
    terrain,
    standalone_replay_start_level,
    standalone_replay_probability,
    terrain.terrain_levels[env_ids],
  )

  # Compute per-terrain-type mean levels.
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
    **mixed_replay_result,
    **standalone_replay_result,
  }
  if target_attempted is not None and target_reached is not None:
    result["target_attempted"] = torch.mean(target_attempted.float())
    result["target_reached"] = torch.mean(target_reached.float())

  # In curriculum mode the column index directly maps to the terrain type name.
  # Prefer the compiled entity names because standalone terrain columns are
  # appended after regular sub-terrains and are not present in sub_terrains.
  compiled_terrain_names = getattr(terrain, "terrain_type_names", ())
  terrain_type_names = list(compiled_terrain_names)
  if not terrain_type_names:
    terrain_type_names = list(terrain_generator.sub_terrains.keys())
    standalone_terrains = getattr(terrain_generator, "standalone_terrains", {})
    terrain_type_names.extend(standalone_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(terrain_type_names):
    types = terrain.terrain_types
    family_masks: dict[str, torch.Tensor] = {}
    for i, name in enumerate(terrain_type_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])
        family = _terrain_family_name(name)
        if family != name:
          family_masks[family] = family_masks.get(family, torch.zeros_like(mask)) | mask
    for family, mask in family_masks.items():
      result[family] = torch.mean(levels[mask])

  return result


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }
