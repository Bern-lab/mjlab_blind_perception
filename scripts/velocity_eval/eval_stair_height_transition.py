"""Evaluate stair-height adaptation across two consecutive stair runs."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import traceback
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import mujoco
import numpy as np
import torch
import tyro
from scripts.velocity_eval.eval_policy_goal_pyramid import (
  _fresh_obs_with_history,
  _get_twist_term,
  _normalize_standalone_bool_flags,
)
from scripts.velocity_eval.eval_stair_lift_height_sweep import (
  FOOT_NAMES,
  _diag_vector,
  _finite_float,
  _LiftTrackerParams,
  _linear_fit,
  _record_values,
  _resolve_eval_runtime,
  _stats,
  _task_selector_from_argv,
)
from scripts.velocity_eval.eval_terrains import (
  _disable_actuator_delays,
  _disable_observation_delays,
  _enable_eval_riser_contact_sensor,
  _fix_reset_events,
  _fix_velocity_command,
)
from scripts.velocity_eval.policy_io import (
  get_clip_actions,
  get_policy_output_name,
  load_inference_policy,
  make_timestamped_policy_output_dir,
)

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp.rewards import (
  _current_step_boundaries,
  _current_step_boundary_metadata,
  _StepBoundaryFootVolume,
)
from mjlab.terrains.terrain_generator import (
  FlatPatchSamplingCfg,
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, wrap_to_pi
from mjlab.utils.lstm import (
  extract_dones,
  reset_policy_state,
  reset_policy_state_from_step,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, VerbosityLevel, ViserPlayViewer

TRAIN_STAIR_HEIGHT_RANGE_M = (0.088, 0.25)
TRAIN_STEP_WIDTH_RANGE_M = (0.23, 0.37)
STAGE_NAMES = ("first", "second")


@dataclass(frozen=True)
class StairHeightTransitionEvalConfig:
  """Configuration for the two-stage stair-height transition experiment."""

  checkpoint_file: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  episodes: int = 40
  num_envs: int = 40
  max_episode_length_s: float = 16.0
  seed: int = 12345
  device: str | None = None
  output_root: str = "eval_outputs/stair_height_transition"
  output_dir: str | None = None
  output_file: str | None = None
  records_file: str | None = None
  write_records_csv: bool = True
  clean_observations: bool = True
  disable_observation_delay: bool = True
  disable_actuator_delay: bool = True
  play: bool = False
  viewer: Literal["auto", "native", "viser"] = "auto"

  first_stair_height: float = 0.10
  second_stair_height: float = 0.18
  stair_levels_per_stage: int = 10
  step_width: float = 0.30
  stair_width: float = 3.0
  side_margin_width: float = 0.75
  flat_apron_width: float = 1.20
  middle_platform_width: float = 0.60
  final_platform_width: float = 1.20
  terrain_border_width: float = 12.0
  floor_depth: float = 1.0
  allow_out_of_train_range: bool = False

  spawn_y_half_width: float = 0.50
  spawn_x_jitter: float = 0.05
  start_z_offset: float = 0.03
  goal_radius: float = 0.65
  goal_height_tolerance: float = 0.20

  goal_speed: float = 0.7
  yaw_kp: float = 1.5
  yaw_rate_limit: float = 1.0
  heading_failure_angle_deg: float = 45.0
  heading_failure_grace_s: float = 0.25

  ground_contact_sensor_name: str = "feet_ground_contact"
  height_sensor_name: str = "foot_height_scan"
  landing_height_tolerance: float = 0.10
  min_candidate_support: float = 0.05
  min_swing_duration_s: float = 0.06
  min_peak_lift_m: float = 0.015
  summarize_success_only: bool = True


@dataclass(frozen=True)
class TransitionRunwayGeometry:
  """Derived local-frame geometry for the long two-stage stair strip."""

  tile_length: float
  tile_width: float
  spawn_x: float
  center_y: float
  first_stage_start_x: float
  first_stage_end_x: float
  second_stage_start_x: float
  second_stage_end_x: float
  goal_x: float
  first_top_height: float
  final_top_height: float

  @property
  def route_length(self) -> float:
    return self.goal_x - self.spawn_x


@dataclass
class TransitionSpawnInfo:
  """World-frame spawn/goal state for the transition runway eval."""

  start_xy_w: torch.Tensor
  goal_xy_w: torch.Tensor
  first_top_x_w: torch.Tensor
  second_start_x_w: torch.Tensor
  second_top_x_w: torch.Tensor
  bottom_z_w: torch.Tensor
  first_top_z_w: torch.Tensor
  final_top_z_w: torch.Tensor
  approach_yaw: torch.Tensor
  nominal_root_height: torch.Tensor
  start_distance: torch.Tensor


@dataclass
class _TransitionBatchResult:
  batch_index: int
  episode_offset: int
  success: list[bool]
  first_stage_reached: list[bool]
  fell: list[bool]
  heading_failed: list[bool]
  timeout_failed: list[bool]
  episode_length_steps: list[float]
  max_height_progress_fraction: list[float]
  max_goal_progress_fraction: list[float]
  max_stage_progress_fraction: list[float]
  step_dt: float
  records: list[dict[str, Any]] = field(default_factory=list)


def _transition_runway_geometry(
  cfg: StairHeightTransitionEvalConfig,
) -> TransitionRunwayGeometry:
  first_run = cfg.stair_levels_per_stage * cfg.step_width
  second_run = cfg.stair_levels_per_stage * cfg.step_width
  tile_width = cfg.stair_width + 2.0 * cfg.side_margin_width
  first_start = cfg.flat_apron_width
  first_end = first_start + first_run
  second_start = first_end + cfg.middle_platform_width
  second_end = second_start + second_run
  tile_length = second_end + cfg.final_platform_width
  return TransitionRunwayGeometry(
    tile_length=tile_length,
    tile_width=tile_width,
    spawn_x=0.5 * cfg.flat_apron_width,
    center_y=0.5 * tile_width,
    first_stage_start_x=first_start,
    first_stage_end_x=first_end,
    second_stage_start_x=second_start,
    second_stage_end_x=second_end,
    goal_x=second_end + 0.5 * cfg.final_platform_width,
    first_top_height=cfg.stair_levels_per_stage * cfg.first_stair_height,
    final_top_height=(
      cfg.stair_levels_per_stage * (cfg.first_stair_height + cfg.second_stair_height)
    ),
  )


def _validate_transition_config(cfg: StairHeightTransitionEvalConfig) -> None:
  if cfg.stair_levels_per_stage <= 0:
    raise ValueError("--stair-levels-per-stage must be positive.")
  if cfg.stair_width <= 0.0:
    raise ValueError("--stair-width must be positive.")
  if cfg.side_margin_width < 0.0:
    raise ValueError("--side-margin-width must be non-negative.")
  if cfg.spawn_y_half_width > 0.5 * cfg.stair_width:
    raise ValueError("--spawn-y-half-width must stay within the stair width.")
  if cfg.flat_apron_width <= 0.0 or cfg.final_platform_width <= 0.0:
    raise ValueError("Apron and final platform widths must be positive.")
  if cfg.middle_platform_width < 0.0:
    raise ValueError("--middle-platform-width must be non-negative.")
  if not cfg.allow_out_of_train_range:
    min_h, max_h = TRAIN_STAIR_HEIGHT_RANGE_M
    for name, value in (
      ("--first-stair-height", cfg.first_stair_height),
      ("--second-stair-height", cfg.second_stair_height),
    ):
      if not min_h <= value <= max_h:
        raise ValueError(
          f"{name}={value} is outside the training riser-height range "
          f"[{min_h}, {max_h}] m. Pass --allow-out-of-train-range True to "
          "override intentionally."
        )
    min_w, max_w = TRAIN_STEP_WIDTH_RANGE_M
    if not min_w <= cfg.step_width <= max_w:
      raise ValueError(
        f"--step-width={cfg.step_width} is outside the training tread-depth "
        f"range [{min_w}, {max_w}] m. Pass --allow-out-of-train-range True "
        "to override intentionally."
      )


def _box(
  body: mujoco.MjsBody,
  *,
  x0: float,
  x1: float,
  y0: float,
  y1: float,
  top_z: float,
  depth: float,
) -> mujoco.MjsGeom | None:
  if x1 <= x0 or y1 <= y0:
    return None
  return body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.5 * (x1 - x0), 0.5 * (y1 - y0), 0.5 * depth),
    pos=(0.5 * (x0 + x1), 0.5 * (y0 + y1), top_z - 0.5 * depth),
  )


def _append_straight_riser_boundary(
  boundaries: list[np.ndarray],
  sequence_ids: list[int],
  layers: list[int],
  *,
  x: float,
  y0: float,
  y1: float,
  z_low: float,
  z_high: float,
  sequence_id: int,
  layer: int,
) -> None:
  boundaries.append(
    np.asarray(
      [
        x,
        y0,
        z_high,
        x,
        y1,
        z_high,
        -1.0,
        0.0,
        0.0,
        z_low,
        z_high,
      ],
      dtype=np.float32,
    )
  )
  sequence_ids.append(sequence_id)
  layers.append(layer)


@dataclass(kw_only=True)
class BoxTwoStageStairsTerrainCfg(SubTerrainCfg):
  """Long eval-only stair strip with two consecutive upward stair runs."""

  first_stair_height: float
  second_stair_height: float
  stair_levels_per_stage: int = 10
  step_width: float = 0.30
  stair_width: float = 3.0
  side_margin_width: float = 0.75
  flat_apron_width: float = 1.20
  middle_platform_width: float = 0.60
  final_platform_width: float = 1.20
  floor_depth: float = 1.0

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del difficulty, rng
    body = spec.body("terrain")
    cfg = StairHeightTransitionEvalConfig(
      first_stair_height=self.first_stair_height,
      second_stair_height=self.second_stair_height,
      stair_levels_per_stage=self.stair_levels_per_stage,
      step_width=self.step_width,
      stair_width=self.stair_width,
      side_margin_width=self.side_margin_width,
      flat_apron_width=self.flat_apron_width,
      middle_platform_width=self.middle_platform_width,
      final_platform_width=self.final_platform_width,
      floor_depth=self.floor_depth,
      allow_out_of_train_range=True,
    )
    geom = _transition_runway_geometry(cfg)
    y0 = geom.center_y - 0.5 * self.stair_width
    y1 = geom.center_y + 0.5 * self.stair_width
    total_height = geom.final_top_height
    depth = max(self.floor_depth, total_height + 0.50)
    boxes: list[mujoco.MjsGeom] = []
    colors: list[tuple[float, float, float, float]] = []
    boundaries: list[np.ndarray] = []
    sequence_ids: list[int] = []
    layers: list[int] = []

    def add_plateau(
      x0: float, x1: float, top_z: float, color: tuple[float, float, float, float]
    ) -> None:
      box = _box(body, x0=x0, x1=x1, y0=y0, y1=y1, top_z=top_z, depth=depth)
      if box is not None:
        boxes.append(box)
        colors.append(color)

    low_color = (0.42, 0.46, 0.50, 1.0)
    first_color = (0.18, 0.46, 0.82, 1.0)
    middle_color = (0.34, 0.56, 0.42, 1.0)
    second_color = (0.78, 0.36, 0.30, 1.0)

    add_plateau(0.0, geom.first_stage_start_x, 0.0, low_color)
    for layer in range(1, self.stair_levels_per_stage + 1):
      x0 = geom.first_stage_start_x + (layer - 1) * self.step_width
      x1 = geom.first_stage_start_x + layer * self.step_width
      z_low = (layer - 1) * self.first_stair_height
      z_high = layer * self.first_stair_height
      _append_straight_riser_boundary(
        boundaries,
        sequence_ids,
        layers,
        x=x0,
        y0=y0,
        y1=y1,
        z_low=z_low,
        z_high=z_high,
        sequence_id=1,
        layer=layer,
      )
      add_plateau(x0, x1, z_high, first_color)

    add_plateau(
      geom.first_stage_end_x,
      geom.second_stage_start_x,
      geom.first_top_height,
      middle_color,
    )
    for layer in range(1, self.stair_levels_per_stage + 1):
      x0 = geom.second_stage_start_x + (layer - 1) * self.step_width
      x1 = geom.second_stage_start_x + layer * self.step_width
      z_low = geom.first_top_height + (layer - 1) * self.second_stair_height
      z_high = geom.first_top_height + layer * self.second_stair_height
      _append_straight_riser_boundary(
        boundaries,
        sequence_ids,
        layers,
        x=x0,
        y0=y0,
        y1=y1,
        z_low=z_low,
        z_high=z_high,
        sequence_id=2,
        layer=layer,
      )
      add_plateau(x0, x1, z_high, second_color)

    add_plateau(
      geom.second_stage_end_x,
      geom.tile_length,
      geom.final_top_height,
      second_color,
    )
    geometries = [
      TerrainGeometry(geom=box, color=color)
      for box, color in zip(boxes, colors, strict=True)
    ]
    origin = np.asarray([geom.spawn_x, geom.center_y, 0.0], dtype=np.float32)
    target = np.asarray(
      [[geom.goal_x, geom.center_y, geom.final_top_height]],
      dtype=np.float32,
    )
    return TerrainOutput(
      origin=origin,
      geometries=geometries,
      flat_patches={"target": target},
      step_boundaries=np.asarray(boundaries, dtype=np.float32),
      step_boundary_sequence_ids=np.asarray(sequence_ids, dtype=np.int32),
      step_boundary_layers=np.asarray(layers, dtype=np.int32),
    )


def _make_transition_terrain_generator(
  cfg: StairHeightTransitionEvalConfig,
  *,
  num_envs: int,
  seed: int,
) -> TerrainGeneratorCfg:
  geom = _transition_runway_geometry(cfg)
  subterrain = BoxTwoStageStairsTerrainCfg(
    first_stair_height=cfg.first_stair_height,
    second_stair_height=cfg.second_stair_height,
    stair_levels_per_stage=cfg.stair_levels_per_stage,
    step_width=cfg.step_width,
    stair_width=cfg.stair_width,
    side_margin_width=cfg.side_margin_width,
    flat_apron_width=cfg.flat_apron_width,
    middle_platform_width=cfg.middle_platform_width,
    final_platform_width=cfg.final_platform_width,
    floor_depth=cfg.floor_depth,
    flat_patch_sampling={
      "target": FlatPatchSamplingCfg(
        num_patches=1,
        patch_radius=0.2,
        max_height_diff=0.01,
      )
    },
  )
  return TerrainGeneratorCfg(
    seed=seed,
    curriculum=False,
    size=(geom.tile_length, geom.tile_width),
    border_width=cfg.terrain_border_width,
    num_rows=1,
    num_cols=max(1, num_envs),
    difficulty_range=(0.0, 0.0),
    sub_terrains={"two_stage_stairs": subterrain},
    add_lights=True,
  )


def _configure_goal_command_cfg(
  env_cfg: ManagerBasedRlEnvCfg,
  cfg: StairHeightTransitionEvalConfig,
) -> None:
  twist_cfg = cast(Any, env_cfg.commands.get("twist"))
  if twist_cfg is None:
    return
  if not hasattr(twist_cfg, "target_reached_threshold"):
    _fix_velocity_command(
      env_cfg,
      command=(0.0, 0.0, 0.0),
      max_episode_length_s=cfg.max_episode_length_s,
    )
    return

  twist_cfg.resampling_time_range = (
    cfg.max_episode_length_s + 1.0,
    cfg.max_episode_length_s + 1.0,
  )
  twist_cfg.heading_command = True
  twist_cfg.rel_target_envs = 1.0
  twist_cfg.rel_random_heading_envs = 0.0
  twist_cfg.rel_standing_envs = 0.0
  twist_cfg.target_reached_threshold = cfg.goal_radius
  twist_cfg.target_min_distance = 0.0
  twist_cfg.target_max_distance = (
    _transition_runway_geometry(cfg).route_length + cfg.final_platform_width
  )
  twist_cfg.target_tile_radius = 0
  twist_cfg.include_current_tile = True
  twist_cfg.zero_lateral_velocity = True
  twist_cfg.heading_control_stiffness = cfg.yaw_kp
  twist_cfg.ranges.lin_vel_x = (cfg.goal_speed, cfg.goal_speed)
  twist_cfg.ranges.lin_vel_y = (0.0, 0.0)
  twist_cfg.ranges.ang_vel_z = (-cfg.yaw_rate_limit, cfg.yaw_rate_limit)
  twist_cfg.ranges.heading = (-math.pi, math.pi)


def _apply_transition_eval_overrides(
  env_cfg: ManagerBasedRlEnvCfg,
  cfg: StairHeightTransitionEvalConfig,
  *,
  num_envs: int,
  seed: int,
  enable_riser_contact_sensor: bool,
) -> ManagerBasedRlEnvCfg:
  env_cfg.seed = seed
  env_cfg.scene.num_envs = num_envs
  env_cfg.episode_length_s = cfg.max_episode_length_s
  if env_cfg.scene.terrain is None:
    raise ValueError("Stair-height transition eval requires terrain.")
  env_cfg.scene.terrain.terrain_type = "generator"
  env_cfg.scene.terrain.terrain_generator = _make_transition_terrain_generator(
    cfg,
    num_envs=num_envs,
    seed=seed,
  )
  env_cfg.scene.terrain.max_init_terrain_level = 0
  env_cfg.curriculum = {}
  _fix_reset_events(env_cfg)
  _configure_goal_command_cfg(env_cfg, cfg)
  if enable_riser_contact_sensor:
    _enable_eval_riser_contact_sensor(env_cfg)
  if cfg.clean_observations:
    for group_cfg in env_cfg.observations.values():
      group_cfg.enable_corruption = False
  if cfg.disable_observation_delay:
    _disable_observation_delays(env_cfg)
  if cfg.disable_actuator_delay:
    _disable_actuator_delays(env_cfg)
  return env_cfg


def _spawn_on_transition_runway(
  env: ManagerBasedRlEnv,
  cfg: StairHeightTransitionEvalConfig,
  *,
  seed: int,
  env_ids: torch.Tensor | None = None,
  spawn: TransitionSpawnInfo | None = None,
) -> TransitionSpawnInfo:
  asset = env.scene["robot"]
  device = env.device
  if env_ids is None:
    target_env_ids = torch.arange(env.num_envs, device=device, dtype=torch.long)
  else:
    target_env_ids = env_ids.to(device=device, dtype=torch.long)
  num_targets = int(target_env_ids.numel())
  if num_targets == 0:
    if spawn is None:
      raise ValueError("Cannot create TransitionSpawnInfo with zero env_ids.")
    return spawn

  generator = torch.Generator(device="cpu")
  generator.manual_seed(seed)
  y_jitter = (
    torch.rand(num_targets, generator=generator).to(device=device) * 2.0 - 1.0
  ) * cfg.spawn_y_half_width
  x_jitter = (
    torch.rand(num_targets, generator=generator).to(device=device) * 2.0 - 1.0
  ) * cfg.spawn_x_jitter

  geom = _transition_runway_geometry(cfg)
  env_origins = env.scene.env_origins[target_env_ids]
  start_xy_w = env_origins[:, :2].clone()
  start_xy_w[:, 0] += x_jitter
  start_xy_w[:, 1] += y_jitter
  goal_xy_w = env_origins[:, :2].clone()
  goal_xy_w[:, 0] += geom.goal_x - geom.spawn_x

  first_top_x_w = env_origins[:, 0] + geom.first_stage_end_x - geom.spawn_x
  second_start_x_w = env_origins[:, 0] + geom.second_stage_start_x - geom.spawn_x
  second_top_x_w = env_origins[:, 0] + geom.second_stage_end_x - geom.spawn_x
  bottom_z_w = env_origins[:, 2]
  first_top_z_w = bottom_z_w + geom.first_top_height
  final_top_z_w = bottom_z_w + geom.final_top_height

  default_root_state = asset.data.default_root_state[target_env_ids].clone()
  nominal_root_height = default_root_state[:, 2].clone()
  root_pos_w = torch.zeros(num_targets, 3, device=device)
  root_pos_w[:, :2] = start_xy_w
  root_pos_w[:, 2] = bottom_z_w + nominal_root_height + cfg.start_z_offset

  approach_yaw = torch.zeros(num_targets, device=device)
  zeros = torch.zeros(num_targets, device=device)
  yaw_delta = quat_from_euler_xyz(zeros, zeros, approach_yaw)
  root_quat_w = quat_mul(default_root_state[:, 3:7], yaw_delta)
  root_vel_w = torch.zeros(num_targets, 6, device=device)

  asset.write_root_link_pose_to_sim(
    torch.cat([root_pos_w, root_quat_w], dim=-1),
    env_ids=target_env_ids,
  )
  asset.write_root_link_velocity_to_sim(root_vel_w, env_ids=target_env_ids)
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.sim.sense()

  start_distance = torch.norm(goal_xy_w - start_xy_w, dim=-1)
  if spawn is not None:
    spawn.start_xy_w[target_env_ids] = start_xy_w
    spawn.goal_xy_w[target_env_ids] = goal_xy_w
    spawn.first_top_x_w[target_env_ids] = first_top_x_w
    spawn.second_start_x_w[target_env_ids] = second_start_x_w
    spawn.second_top_x_w[target_env_ids] = second_top_x_w
    spawn.bottom_z_w[target_env_ids] = bottom_z_w
    spawn.first_top_z_w[target_env_ids] = first_top_z_w
    spawn.final_top_z_w[target_env_ids] = final_top_z_w
    spawn.approach_yaw[target_env_ids] = approach_yaw
    spawn.nominal_root_height[target_env_ids] = nominal_root_height
    spawn.start_distance[target_env_ids] = start_distance
    return spawn

  if num_targets != env.num_envs:
    raise ValueError("Initial TransitionSpawnInfo must include all environments.")
  return TransitionSpawnInfo(
    start_xy_w=start_xy_w,
    goal_xy_w=goal_xy_w,
    first_top_x_w=first_top_x_w,
    second_start_x_w=second_start_x_w,
    second_top_x_w=second_top_x_w,
    bottom_z_w=bottom_z_w,
    first_top_z_w=first_top_z_w,
    final_top_z_w=final_top_z_w,
    approach_yaw=approach_yaw,
    nominal_root_height=nominal_root_height,
    start_distance=start_distance,
  )


def _set_target_command_fields(
  term: Any,
  *,
  goal_xy_w: torch.Tensor,
  goal_z_w: torch.Tensor,
  distance: torch.Tensor,
  reached: torch.Tensor,
) -> None:
  if hasattr(term, "target_pos_w"):
    term.target_pos_w[:, :2] = goal_xy_w
    term.target_pos_w[:, 2] = goal_z_w
  if hasattr(term, "target_distance"):
    term.target_distance.copy_(distance)
  if hasattr(term, "has_target"):
    term.has_target.fill_(True)
  if hasattr(term, "is_target_env"):
    term.is_target_env.fill_(True)
  if hasattr(term, "target_command_in_episode"):
    term.target_command_in_episode.fill_(True)
  if hasattr(term, "target_reached"):
    term.target_reached.copy_(reached)
  if hasattr(term, "target_reached_this_step"):
    term.target_reached_this_step.copy_(reached)


def _update_transition_goal_command(
  env: ManagerBasedRlEnv,
  cfg: StairHeightTransitionEvalConfig,
  spawn: TransitionSpawnInfo,
  active: torch.Tensor,
) -> torch.Tensor:
  asset = env.scene["robot"]
  root_xy = asset.data.root_link_pos_w[:, :2]
  delta_xy = spawn.goal_xy_w - root_xy
  distance = torch.norm(delta_xy, dim=-1)
  target_yaw = torch.atan2(delta_xy[:, 1], delta_xy[:, 0])
  yaw_error = wrap_to_pi(target_yaw - asset.data.heading_w)
  speed = cfg.goal_speed * torch.clamp(torch.cos(yaw_error), min=0.0, max=1.0)
  speed = torch.where(distance <= cfg.goal_radius, torch.zeros_like(speed), speed)
  yaw_rate = torch.clamp(
    cfg.yaw_kp * yaw_error,
    min=-cfg.yaw_rate_limit,
    max=cfg.yaw_rate_limit,
  )
  reached = _transition_goal_reached(env, cfg, spawn)

  term = _get_twist_term(env)
  active_ids = active.nonzero(as_tuple=False).flatten()
  term.vel_command_b.zero_()
  if active_ids.numel() > 0:
    term.vel_command_b[active_ids, 0] = speed[active_ids]
    term.vel_command_b[active_ids, 2] = yaw_rate[active_ids]
  term.vel_command_w.zero_()
  term.heading_target.copy_(target_yaw)
  term.is_standing_env.fill_(False)
  term.is_world_env.fill_(False)
  term.is_forward_env.fill_(False)
  term.is_heading_env.fill_(True)
  _set_target_command_fields(
    term,
    goal_xy_w=spawn.goal_xy_w,
    goal_z_w=spawn.final_top_z_w,
    distance=distance,
    reached=reached,
  )
  return yaw_error


def _transition_goal_reached(
  env: ManagerBasedRlEnv,
  cfg: StairHeightTransitionEvalConfig,
  spawn: TransitionSpawnInfo,
) -> torch.Tensor:
  asset = env.scene["robot"]
  root_pos = asset.data.root_link_pos_w
  distance = torch.norm(root_pos[:, :2] - spawn.goal_xy_w, dim=-1)
  estimated_support_z = root_pos[:, 2] - spawn.nominal_root_height
  height_ok = estimated_support_z >= spawn.final_top_z_w - cfg.goal_height_tolerance
  return (distance <= cfg.goal_radius) & height_ok


def _first_stage_top_reached(
  env: ManagerBasedRlEnv,
  cfg: StairHeightTransitionEvalConfig,
  spawn: TransitionSpawnInfo,
) -> torch.Tensor:
  asset = env.scene["robot"]
  root_pos = asset.data.root_link_pos_w
  estimated_support_z = root_pos[:, 2] - spawn.nominal_root_height
  x_ok = root_pos[:, 0] >= spawn.first_top_x_w - cfg.goal_radius
  height_ok = estimated_support_z >= spawn.first_top_z_w - cfg.goal_height_tolerance
  return x_ok & height_ok


def _transition_heading_failure(
  env: ManagerBasedRlEnv,
  cfg: StairHeightTransitionEvalConfig,
  spawn: TransitionSpawnInfo,
  *,
  step_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  asset = env.scene["robot"]
  heading_error = torch.abs(wrap_to_pi(asset.data.heading_w - spawn.approach_yaw))
  grace_done = step_counts * env.step_dt >= cfg.heading_failure_grace_s
  failed = grace_done & (heading_error > math.radians(cfg.heading_failure_angle_deg))
  return failed, heading_error


def _stage_progress_fraction(
  env: ManagerBasedRlEnv,
  spawn: TransitionSpawnInfo,
) -> torch.Tensor:
  asset = env.scene["robot"]
  root_x = asset.data.root_link_pos_w[:, 0]
  first_stage = torch.clamp(
    (root_x - spawn.start_xy_w[:, 0])
    / (spawn.first_top_x_w - spawn.start_xy_w[:, 0]).clamp_min(1.0e-6),
    min=0.0,
    max=1.0,
  )
  second_stage = torch.clamp(
    (root_x - spawn.second_start_x_w)
    / (spawn.second_top_x_w - spawn.second_start_x_w).clamp_min(1.0e-6),
    min=0.0,
    max=1.0,
  )
  return torch.where(
    root_x < spawn.second_start_x_w,
    0.5 * first_stage,
    0.5 + 0.5 * second_stage,
  )


class StairHeightTransitionTracker:
  """Track foot swing peaks and tag each landing by stair segment."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    *,
    cfg: StairHeightTransitionEvalConfig,
    params: _LiftTrackerParams,
    batch_index: int,
    episode_offset: int,
    global_episode_offset: int,
  ) -> None:
    self.cfg = cfg
    self.params = params
    self.batch_index = int(batch_index)
    self.episode_offset = int(episode_offset)
    self.global_episode_offset = int(global_episode_offset)

    self.foot_body_cfg = SceneEntityCfg(
      "robot",
      body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
      preserve_order=True,
    )
    self.foot_body_cfg.resolve(env.scene)
    self._volume = _StepBoundaryFootVolume(
      RewardTermCfg(func=StairHeightTransitionTracker, weight=0.0, params={}),
      env,
    )
    self._ground_sensor = self._get_ground_sensor(env)
    self._height_sensor = self._get_height_sensor(env)

    num_envs = env.num_envs
    num_feet = len(self._foot_body_ids())
    device = env.device
    self._in_swing = torch.zeros(num_envs, num_feet, device=device, dtype=torch.bool)
    self._takeoff_step = torch.full(
      (num_envs, num_feet), -1, device=device, dtype=torch.long
    )
    self._takeoff_foot_z = torch.zeros(num_envs, num_feet, device=device)
    self._takeoff_clearance = torch.zeros(num_envs, num_feet, device=device)
    self._peak_foot_z = torch.zeros(num_envs, num_feet, device=device)
    self._peak_clearance = torch.zeros(num_envs, num_feet, device=device)
    self._pred_count = torch.zeros(num_envs, num_feet, device=device)
    self._pred_riser_sum = torch.zeros(num_envs, num_feet, device=device)
    self._pred_stride_sum = torch.zeros(num_envs, num_feet, device=device)
    self._pred_stair_prob_sum = torch.zeros(num_envs, num_feet, device=device)
    self._pred_event_prob_sum = torch.zeros(num_envs, num_feet, device=device)
    self._gate_memory_sum = torch.zeros(num_envs, num_feet, device=device)
    self._takeoff_pred_riser = torch.full(
      (num_envs, num_feet), torch.nan, device=device
    )
    self._touchdown_counter = torch.zeros(num_envs, device=device, dtype=torch.long)
    self._stage_touchdown_counter = torch.zeros(
      num_envs,
      2,
      device=device,
      dtype=torch.long,
    )
    self.records: list[dict[str, Any]] = []

  def _get_ground_sensor(self, env: ManagerBasedRlEnv) -> ContactSensor:
    sensor = env.scene[self.params.ground_contact_sensor_name]
    if not isinstance(sensor, ContactSensor):
      raise TypeError(
        f"Expected ContactSensor {self.params.ground_contact_sensor_name!r}, "
        f"got {type(sensor).__name__}."
      )
    return sensor

  def _get_height_sensor(self, env: ManagerBasedRlEnv) -> TerrainHeightSensor:
    sensor = env.scene[self.params.height_sensor_name]
    if not isinstance(sensor, TerrainHeightSensor):
      raise TypeError(
        f"Expected TerrainHeightSensor {self.params.height_sensor_name!r}, "
        f"got {type(sensor).__name__}."
      )
    return sensor

  def _foot_body_ids(self) -> list[int]:
    body_ids = self.foot_body_cfg.body_ids
    if not isinstance(body_ids, list):
      raise RuntimeError("Transition tracker foot body IDs were not resolved.")
    return body_ids

  def _foot_z(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    asset = env.scene[self.foot_body_cfg.name]
    return asset.data.body_link_pos_w[:, self._foot_body_ids(), 2]

  def _foot_clearance(self) -> torch.Tensor:
    heights = self._height_sensor.data.heights
    if heights.ndim == 3:
      heights = heights.min(dim=-1).values
    return heights

  def _landing_candidates(
    self,
    env: ManagerBasedRlEnv,
    touchdown: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    num_envs, num_feet = touchdown.shape
    device = touchdown.device
    invalid_level = torch.full(
      (num_envs, num_feet), -1, device=device, dtype=torch.long
    )
    invalid_stage = torch.full(
      (num_envs, num_feet), -1, device=device, dtype=torch.long
    )
    empty_float = torch.zeros(num_envs, num_feet, device=device)
    empty = {
      "valid": torch.zeros_like(touchdown),
      "stage_index": invalid_stage,
      "level": invalid_level,
      "global_level": invalid_level,
      "support": empty_float,
      "target_z": empty_float,
      "riser_height": empty_float,
      "stage_height": empty_float,
    }
    if not bool(touchdown.any().item()):
      return empty

    boundaries, valid_boundaries = _current_step_boundaries(env)
    metadata = _current_step_boundary_metadata(env)
    if (
      boundaries is None
      or valid_boundaries is None
      or metadata[0] is None
      or metadata[1] is None
      or boundaries.shape[1] == 0
    ):
      return empty

    sequence_ids = metadata[0]
    boundary_layers = metadata[1]
    points_w, _point_vel_w = self._volume._foot_points_w(env, self.foot_body_cfg)
    foot_ref_w = self._volume._foot_ref_w(env, self.foot_body_cfg)
    sole_z = torch.min(self._volume._local_points[:, 2])
    sole_mask = self._volume._local_points[:, 2] <= sole_z + 1.0e-6
    sole_points_w = points_w[:, :, sole_mask, :]
    tread_depth = torch.full((num_envs,), self.cfg.step_width, device=device)
    support_fraction = mdp.toe_step_riser_slab_penalty._tread_support_fraction(
      sole_points_w,
      boundaries,
      tread_depth,
    )

    height_error = torch.abs(foot_ref_w[:, :, None, 2] - boundaries[:, None, :, 10])
    valid_stage_boundary = (
      valid_boundaries
      & (sequence_ids >= 1)
      & (sequence_ids <= 2)
      & (boundary_layers >= 1)
      & (boundary_layers <= self.cfg.stair_levels_per_stage)
    )
    candidate = (
      valid_stage_boundary[:, None, :]
      & (height_error <= self.params.landing_height_tolerance)
      & (support_fraction >= self.params.min_candidate_support)
    )
    masked_support = torch.where(
      candidate,
      support_fraction,
      torch.full_like(support_fraction, -torch.inf),
    )
    best_support, best_idx = torch.max(masked_support, dim=-1)
    has_candidate = touchdown & torch.isfinite(best_support)
    safe_idx = best_idx.clamp_min(0)
    stage_index = sequence_ids.gather(1, safe_idx)
    level = boundary_layers.gather(1, safe_idx)
    global_level = torch.where(
      stage_index == 2,
      level + self.cfg.stair_levels_per_stage,
      level,
    )
    target_z = boundaries[..., 10].gather(1, safe_idx)
    riser_height = torch.abs(boundaries[..., 10] - boundaries[..., 9]).gather(
      1,
      safe_idx,
    )
    stage_height = torch.where(
      stage_index == 2,
      torch.full_like(riser_height, self.cfg.second_stair_height),
      torch.full_like(riser_height, self.cfg.first_stair_height),
    )
    return {
      "valid": has_candidate,
      "stage_index": torch.where(has_candidate, stage_index, invalid_stage),
      "level": torch.where(has_candidate, level, invalid_level),
      "global_level": torch.where(has_candidate, global_level, invalid_level),
      "support": torch.where(has_candidate, best_support.clamp_min(0.0), empty_float),
      "target_z": torch.where(has_candidate, target_z, empty_float),
      "riser_height": torch.where(has_candidate, riser_height, empty_float),
      "stage_height": torch.where(has_candidate, stage_height, empty_float),
    }

  def _accumulate_predictions(
    self,
    active_swing: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
  ) -> None:
    if not bool(active_swing.any().item()):
      return
    num_envs = active_swing.shape[0]
    values = {
      "riser": _diag_vector(diagnostics, "stair_shape", num_envs=num_envs, component=1),
      "stride": _diag_vector(
        diagnostics, "stair_shape", num_envs=num_envs, component=0
      ),
      "stair_prob": _diag_vector(diagnostics, "stair_prob", num_envs=num_envs),
      "event_prob": _diag_vector(diagnostics, "event_prob", num_envs=num_envs),
      "gate_mode": _diag_vector(diagnostics, "gate_mode", num_envs=num_envs),
    }
    if all(value is None for value in values.values()):
      return

    finite_sample = torch.zeros_like(active_swing, dtype=torch.bool)
    if values["riser"] is not None:
      riser = cast(torch.Tensor, values["riser"])[:, None].expand_as(
        self._pred_riser_sum
      )
      valid = active_swing & torch.isfinite(riser)
      self._pred_riser_sum += torch.where(valid, riser, torch.zeros_like(riser))
      finite_sample |= valid
    if values["stride"] is not None:
      stride = cast(torch.Tensor, values["stride"])[:, None].expand_as(
        self._pred_stride_sum
      )
      valid = active_swing & torch.isfinite(stride)
      self._pred_stride_sum += torch.where(valid, stride, torch.zeros_like(stride))
      finite_sample |= valid
    if values["stair_prob"] is not None:
      stair_prob = cast(torch.Tensor, values["stair_prob"])[:, None].expand_as(
        self._pred_stair_prob_sum
      )
      valid = active_swing & torch.isfinite(stair_prob)
      self._pred_stair_prob_sum += torch.where(
        valid,
        stair_prob,
        torch.zeros_like(stair_prob),
      )
      finite_sample |= valid
    if values["event_prob"] is not None:
      event_prob = cast(torch.Tensor, values["event_prob"])[:, None].expand_as(
        self._pred_event_prob_sum
      )
      valid = active_swing & torch.isfinite(event_prob)
      self._pred_event_prob_sum += torch.where(
        valid,
        event_prob,
        torch.zeros_like(event_prob),
      )
      finite_sample |= valid
    if values["gate_mode"] is not None:
      gate_mode = cast(torch.Tensor, values["gate_mode"])[:, None].expand_as(
        self._gate_memory_sum
      )
      valid = active_swing & torch.isfinite(gate_mode)
      memory = (gate_mode >= 1.5).to(gate_mode.dtype)
      self._gate_memory_sum += torch.where(valid, memory, torch.zeros_like(memory))
      finite_sample |= valid
    self._pred_count += finite_sample.float()

  def _reset_swing_slots(self, slots: torch.Tensor) -> None:
    if slots.numel() == 0:
      return
    env_ids = slots[:, 0]
    foot_ids = slots[:, 1]
    self._in_swing[env_ids, foot_ids] = False
    self._takeoff_step[env_ids, foot_ids] = -1
    self._takeoff_foot_z[env_ids, foot_ids] = 0.0
    self._takeoff_clearance[env_ids, foot_ids] = 0.0
    self._peak_foot_z[env_ids, foot_ids] = 0.0
    self._peak_clearance[env_ids, foot_ids] = 0.0
    self._pred_count[env_ids, foot_ids] = 0.0
    self._pred_riser_sum[env_ids, foot_ids] = 0.0
    self._pred_stride_sum[env_ids, foot_ids] = 0.0
    self._pred_stair_prob_sum[env_ids, foot_ids] = 0.0
    self._pred_event_prob_sum[env_ids, foot_ids] = 0.0
    self._gate_memory_sum[env_ids, foot_ids] = 0.0
    self._takeoff_pred_riser[env_ids, foot_ids] = torch.nan

  def update(
    self,
    env: ManagerBasedRlEnv,
    *,
    active: torch.Tensor,
    step_index: int,
    diagnostics: dict[str, torch.Tensor],
  ) -> None:
    active = active.to(device=env.device, dtype=torch.bool)
    foot_z = self._foot_z(env)
    clearance = self._foot_clearance()
    if clearance.shape != foot_z.shape:
      raise RuntimeError(
        f"Foot height shape {tuple(clearance.shape)} does not match foot body "
        f"shape {tuple(foot_z.shape)}."
      )

    first_air = self._ground_sensor.compute_first_air(dt=env.step_dt).bool()
    first_contact = self._ground_sensor.compute_first_contact(dt=env.step_dt).bool()
    active_feet = active[:, None].expand_as(self._in_swing)

    start = first_air & active_feet & ~self._in_swing
    if bool(start.any().item()):
      self._in_swing |= start
      step_tensor = torch.full_like(self._takeoff_step, int(step_index))
      self._takeoff_step = torch.where(start, step_tensor, self._takeoff_step)
      self._takeoff_foot_z = torch.where(start, foot_z, self._takeoff_foot_z)
      self._takeoff_clearance = torch.where(start, clearance, self._takeoff_clearance)
      self._peak_foot_z = torch.where(start, foot_z, self._peak_foot_z)
      self._peak_clearance = torch.where(start, clearance, self._peak_clearance)
      riser = _diag_vector(
        diagnostics,
        "stair_shape",
        num_envs=env.num_envs,
        component=1,
      )
      if riser is not None:
        riser = riser[:, None].expand_as(self._takeoff_pred_riser)
        self._takeoff_pred_riser = torch.where(
          start & torch.isfinite(riser),
          riser,
          self._takeoff_pred_riser,
        )

    active_swing = self._in_swing & active_feet
    self._peak_foot_z = torch.where(
      active_swing,
      torch.maximum(self._peak_foot_z, foot_z),
      self._peak_foot_z,
    )
    self._peak_clearance = torch.where(
      active_swing,
      torch.maximum(self._peak_clearance, clearance),
      self._peak_clearance,
    )
    self._accumulate_predictions(active_swing, diagnostics)

    touchdown = first_contact & self._in_swing & active_feet
    candidates = self._landing_candidates(env, touchdown)
    complete = touchdown & candidates["valid"]
    if bool(complete.any().item()):
      self._append_records(
        env,
        complete=complete,
        candidates=candidates,
        step_index=step_index,
        foot_z=foot_z,
        clearance=clearance,
        diagnostics=diagnostics,
      )

    reset = touchdown | (self._in_swing & ~active_feet)
    self._reset_swing_slots(reset.nonzero(as_tuple=False))

  def _append_records(
    self,
    env: ManagerBasedRlEnv,
    *,
    complete: torch.Tensor,
    candidates: dict[str, torch.Tensor],
    step_index: int,
    foot_z: torch.Tensor,
    clearance: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
  ) -> None:
    touchdown_pred = _diag_vector(
      diagnostics,
      "stair_shape",
      num_envs=env.num_envs,
      component=1,
    )
    slots = complete.nonzero(as_tuple=False)
    for env_id_t, foot_id_t in slots.detach().cpu():
      env_id = int(env_id_t.item())
      foot_id = int(foot_id_t.item())
      stage_index = int(candidates["stage_index"][env_id, foot_id].item())
      if stage_index not in (1, 2):
        continue
      stage_id = stage_index - 1
      self._touchdown_counter[env_id] += 1
      self._stage_touchdown_counter[env_id, stage_id] += 1
      touchdown_index = int(self._touchdown_counter[env_id].item())
      stage_touchdown_index = int(
        self._stage_touchdown_counter[env_id, stage_id].item()
      )
      pred_count = float(self._pred_count[env_id, foot_id].item())
      safe_pred_count = max(pred_count, 1.0)
      takeoff_step = int(self._takeoff_step[env_id, foot_id].item())
      duration_steps = max(1, int(step_index) - takeoff_step + 1)
      peak_lift = (
        self._peak_foot_z[env_id, foot_id] - self._takeoff_foot_z[env_id, foot_id]
      )
      duration_s = float(duration_steps * env.step_dt)
      if duration_s < self.params.min_swing_duration_s:
        continue
      if float(peak_lift.item()) < self.params.min_peak_lift_m:
        continue

      episode_index = self.episode_offset + env_id
      episode_global_index = self.global_episode_offset + env_id
      touchdown_pred_value = (
        None if touchdown_pred is None else float(touchdown_pred[env_id].item())
      )
      level = int(candidates["level"][env_id, foot_id].item())
      record = {
        "batch_index": self.batch_index,
        "env_index": env_id,
        "episode_index": episode_index,
        "episode_global_index": episode_global_index,
        "foot_index": foot_id,
        "foot": FOOT_NAMES[foot_id] if foot_id < len(FOOT_NAMES) else str(foot_id),
        "touchdown_index_in_episode": touchdown_index,
        "stage_touchdown_index_in_episode": stage_touchdown_index,
        "stage_index": stage_index,
        "stage": STAGE_NAMES[stage_id],
        "stair_height_m": float(candidates["stage_height"][env_id, foot_id].item()),
        "first_stair_height_m": self.cfg.first_stair_height,
        "second_stair_height_m": self.cfg.second_stair_height,
        "height_change_m": self.cfg.second_stair_height - self.cfg.first_stair_height,
        "landing_level_in_stage": level,
        "landing_level_global_low_to_high": int(
          candidates["global_level"][env_id, foot_id].item()
        ),
        "is_first_stage_touchdown": stage_touchdown_index == 1,
        "is_first_stage_level": level == 1,
        "is_second_stage_first_touchdown": stage_index == 2
        and stage_touchdown_index == 1,
        "is_after_height_change": stage_index == 2,
        "takeoff_step": takeoff_step,
        "touchdown_step": int(step_index),
        "duration_s": duration_s,
        "peak_lift_from_takeoff_m": float(peak_lift.item()),
        "peak_clearance_above_terrain_m": float(
          self._peak_clearance[env_id, foot_id].item()
        ),
        "takeoff_clearance_above_terrain_m": float(
          self._takeoff_clearance[env_id, foot_id].item()
        ),
        "touchdown_clearance_above_terrain_m": float(clearance[env_id, foot_id].item()),
        "takeoff_foot_z_w_m": float(self._takeoff_foot_z[env_id, foot_id].item()),
        "peak_foot_z_w_m": float(self._peak_foot_z[env_id, foot_id].item()),
        "touchdown_foot_z_w_m": float(foot_z[env_id, foot_id].item()),
        "landing_support_fraction": float(
          candidates["support"][env_id, foot_id].item()
        ),
        "landing_target_z_w_m": float(candidates["target_z"][env_id, foot_id].item()),
        "landing_riser_height_m": float(
          candidates["riser_height"][env_id, foot_id].item()
        ),
        "pred_samples": pred_count,
        "pred_riser_height_mean_m": _finite_float(
          self._pred_riser_sum[env_id, foot_id].item() / safe_pred_count
        ),
        "pred_riser_height_takeoff_m": _finite_float(
          self._takeoff_pred_riser[env_id, foot_id].item()
        ),
        "pred_riser_height_touchdown_m": _finite_float(touchdown_pred_value),
        "pred_same_foot_stride_mean_m": _finite_float(
          self._pred_stride_sum[env_id, foot_id].item() / safe_pred_count
        ),
        "pred_stair_prob_mean": _finite_float(
          self._pred_stair_prob_sum[env_id, foot_id].item() / safe_pred_count
        ),
        "pred_event_prob_mean": _finite_float(
          self._pred_event_prob_sum[env_id, foot_id].item() / safe_pred_count
        ),
        "gate_memory_ratio": _finite_float(
          self._gate_memory_sum[env_id, foot_id].item() / safe_pred_count
        ),
      }
      self.records.append(record)


def _run_transition_batch(
  *,
  task_id: str,
  env_cfg_factory: Callable[[bool], ManagerBasedRlEnvCfg],
  runner_cls: type | None,
  agent_cfg: Any,
  checkpoint_path: Path,
  cfg: StairHeightTransitionEvalConfig,
  batch_size: int,
  batch_index: int,
  episode_offset: int,
  global_episode_offset: int,
  device: str,
) -> _TransitionBatchResult:
  env_cfg = env_cfg_factory(False)
  _apply_transition_eval_overrides(
    env_cfg,
    cfg,
    num_envs=batch_size,
    seed=cfg.seed + 1009 * batch_index,
    enable_riser_contact_sensor="g1" in task_id.lower(),
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=get_clip_actions(agent_cfg))

  try:
    policy, _runner = load_inference_policy(
      env=wrapped,
      task_id=task_id,
      agent_cfg=agent_cfg,
      checkpoint_path=checkpoint_path,
      device=device,
      runner_cls=runner_cls,
    )
    tracker = StairHeightTransitionTracker(
      wrapped.unwrapped,
      cfg=cfg,
      params=_LiftTrackerParams(
        ground_contact_sensor_name=cfg.ground_contact_sensor_name,
        height_sensor_name=cfg.height_sensor_name,
        landing_height_tolerance=cfg.landing_height_tolerance,
        min_candidate_support=cfg.min_candidate_support,
        min_swing_duration_s=cfg.min_swing_duration_s,
        min_peak_lift_m=cfg.min_peak_lift_m,
      ),
      batch_index=batch_index,
      episode_offset=episode_offset,
      global_episode_offset=global_episode_offset,
    )
    spawn = _spawn_on_transition_runway(
      wrapped.unwrapped,
      cfg,
      seed=cfg.seed + 1543 * batch_index,
    )

    all_envs_active = torch.ones(batch_size, dtype=torch.bool, device=device)
    _update_transition_goal_command(wrapped.unwrapped, cfg, spawn, all_envs_active)
    obs = _fresh_obs_with_history(wrapped.unwrapped)

    step_counts = torch.zeros(batch_size, device=device)
    done_envs = torch.zeros(batch_size, dtype=torch.bool, device=device)
    success = torch.zeros(batch_size, dtype=torch.bool, device=device)
    first_stage_reached = torch.zeros(batch_size, dtype=torch.bool, device=device)
    fell = torch.zeros(batch_size, dtype=torch.bool, device=device)
    heading_failed = torch.zeros(batch_size, dtype=torch.bool, device=device)
    timeout_failed = torch.zeros(batch_size, dtype=torch.bool, device=device)
    max_height_progress = torch.zeros(batch_size, device=device)
    max_goal_progress = torch.zeros(batch_size, device=device)
    max_stage_progress = torch.zeros(batch_size, device=device)

    max_steps = wrapped.unwrapped.max_episode_length + 2
    for step_index in range(max_steps):
      active = ~done_envs
      first_stage_reached |= (
        _first_stage_top_reached(
          wrapped.unwrapped,
          cfg,
          spawn,
        )
        & active
      )
      reached_now = _transition_goal_reached(wrapped.unwrapped, cfg, spawn) & active
      if bool(reached_now.any().item()):
        success |= reached_now
        done_envs |= reached_now
        active = ~done_envs
      if not bool(active.any().item()):
        break

      asset = wrapped.unwrapped.scene["robot"]
      estimated_support_z = asset.data.root_link_pos_w[:, 2] - spawn.nominal_root_height
      total_height = (spawn.final_top_z_w - spawn.bottom_z_w).clamp_min(1.0e-6)
      height_progress = torch.clamp(
        (estimated_support_z - spawn.bottom_z_w) / total_height,
        min=0.0,
        max=1.0,
      )
      goal_distance = torch.norm(
        asset.data.root_link_pos_w[:, :2] - spawn.goal_xy_w,
        dim=-1,
      )
      goal_progress = torch.clamp(
        (spawn.start_distance - goal_distance) / spawn.start_distance.clamp_min(1.0e-6),
        min=0.0,
        max=1.0,
      )
      stage_progress = _stage_progress_fraction(wrapped.unwrapped, spawn)
      max_height_progress = torch.where(
        active,
        torch.maximum(max_height_progress, height_progress),
        max_height_progress,
      )
      max_goal_progress = torch.where(
        active,
        torch.maximum(max_goal_progress, goal_progress),
        max_goal_progress,
      )
      max_stage_progress = torch.where(
        active,
        torch.maximum(max_stage_progress, stage_progress),
        max_stage_progress,
      )
      step_counts += active.float()

      _update_transition_goal_command(wrapped.unwrapped, cfg, spawn, active)
      with torch.no_grad():
        actions = policy(obs)
        get_diagnostics = getattr(policy, "get_slow_latent_diagnostics", None)
        diagnostics = get_diagnostics() if callable(get_diagnostics) else {}
        step_result = wrapped.step(actions)
        tracker.update(
          wrapped.unwrapped,
          active=active,
          step_index=step_index,
          diagnostics=diagnostics,
        )
      reset_policy_state_from_step(policy, step_result)
      obs, _rewards, dones, _extras = step_result

      dones = dones.bool()
      terminated = wrapped.unwrapped.termination_manager.terminated.bool()
      truncated = wrapped.unwrapped.termination_manager.time_outs.bool()
      newly_done = dones & active

      first_stage_reached |= (
        _first_stage_top_reached(
          wrapped.unwrapped,
          cfg,
          spawn,
        )
        & active
      )
      reached_after_step = _transition_goal_reached(wrapped.unwrapped, cfg, spawn)
      reached_after_step &= active & ~newly_done
      heading_failed_now, _heading_error = _transition_heading_failure(
        wrapped.unwrapped,
        cfg,
        spawn,
        step_counts=step_counts,
      )
      heading_failed_now &= active & ~newly_done & ~reached_after_step

      if "fell_over" in wrapped.unwrapped.termination_manager.active_terms:
        fell_now = wrapped.unwrapped.termination_manager.get_term("fell_over").bool()
      else:
        fell_now = terminated

      success |= reached_after_step
      fell |= newly_done & fell_now
      timeout_failed |= newly_done & truncated & ~terminated
      heading_failed |= heading_failed_now
      done_envs |= newly_done | reached_after_step | heading_failed_now

    timeout_failed |= ~done_envs

    success_list = success.detach().cpu().tolist()
    first_stage_reached_list = first_stage_reached.detach().cpu().tolist()
    fell_list = fell.detach().cpu().tolist()
    heading_failed_list = heading_failed.detach().cpu().tolist()
    timeout_failed_list = timeout_failed.detach().cpu().tolist()
    for record in tracker.records:
      local_env = int(record["env_index"])
      record["episode_success"] = bool(success_list[local_env])
      record["episode_first_stage_reached"] = bool(first_stage_reached_list[local_env])
      record["episode_fell"] = bool(fell_list[local_env])
      record["episode_heading_failed"] = bool(heading_failed_list[local_env])
      record["episode_timeout_failed"] = bool(timeout_failed_list[local_env])

    return _TransitionBatchResult(
      batch_index=batch_index,
      episode_offset=episode_offset,
      success=success_list,
      first_stage_reached=first_stage_reached_list,
      fell=fell_list,
      heading_failed=heading_failed_list,
      timeout_failed=timeout_failed_list,
      episode_length_steps=step_counts.detach().cpu().tolist(),
      max_height_progress_fraction=max_height_progress.detach().cpu().tolist(),
      max_goal_progress_fraction=max_goal_progress.detach().cpu().tolist(),
      max_stage_progress_fraction=max_stage_progress.detach().cpu().tolist(),
      step_dt=float(wrapped.unwrapped.step_dt),
      records=tracker.records,
    )
  finally:
    wrapped.close()


def _refresh_transition_respawn_observations(env: ManagerBasedRlEnv) -> None:
  """Prime observation history after play-mode teleports envs to new starts."""
  env.obs_buf = env.observation_manager.compute(update_history=True)


class _TransitionViewerProtocol(Protocol):
  env: RslRlVecEnvWrapper
  policy: Any
  _step_count: int
  _stats_steps: int
  _sim_budget: float
  _last_error: str | None

  def log(self, message: str, level: VerbosityLevel = VerbosityLevel.INFO) -> None: ...

  def pause(self) -> None: ...


class StairHeightTransitionPlayMixin:
  """Viewer mixin that injects the two-stage stair target command."""

  def __init__(
    self,
    *args,
    transition_cfg: StairHeightTransitionEvalConfig,
    spawn: TransitionSpawnInfo,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self._transition_cfg = transition_cfg
    self._transition_spawn = spawn
    self._transition_respawn_counter = 0

  def _respawn_transition_envs(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    viewer = cast("_TransitionViewerProtocol", self)
    env = viewer.env.unwrapped
    env.reset(env_ids=env_ids)
    self._transition_respawn_counter += 1
    self._transition_spawn = _spawn_on_transition_runway(
      env,
      self._transition_cfg,
      seed=self._transition_cfg.seed + 7919 * self._transition_respawn_counter,
      env_ids=env_ids,
      spawn=self._transition_spawn,
    )
    env.observation_manager.reset(env_ids)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    _update_transition_goal_command(
      env,
      self._transition_cfg,
      self._transition_spawn,
      active,
    )
    _refresh_transition_respawn_observations(env)

    dones = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    dones[env_ids] = True
    reset_policy_state(viewer.policy, dones)

  def _execute_step(self) -> bool:
    try:
      viewer = cast("_TransitionViewerProtocol", self)
      env = viewer.env.unwrapped
      active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
      _update_transition_goal_command(
        env,
        self._transition_cfg,
        self._transition_spawn,
        active,
      )

      with torch.no_grad():
        obs = viewer.env.get_observations()
        actions = viewer.policy(obs)
        step_result = viewer.env.step(actions)

      built_in_dones = extract_dones(step_result)
      if built_in_dones is None:
        built_in_dones = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
      else:
        built_in_dones = built_in_dones.to(device=env.device, dtype=torch.bool)

      reached = _transition_goal_reached(
        env,
        self._transition_cfg,
        self._transition_spawn,
      )
      heading_failed, _heading_error = _transition_heading_failure(
        env,
        self._transition_cfg,
        self._transition_spawn,
        step_counts=env.episode_length_buf.float(),
      )
      custom_dones = (reached | heading_failed) & ~built_in_dones
      reset_mask = built_in_dones | custom_dones
      if bool(reset_mask.any().item()):
        reset_env_ids = reset_mask.nonzero(as_tuple=False).flatten()
        self._respawn_transition_envs(reset_env_ids)
      else:
        reset_policy_state_from_step(viewer.policy, step_result)

      self._step_count += 1
      self._stats_steps += 1
      return True
    except Exception:
      self._last_error = traceback.format_exc()
      viewer = cast("_TransitionViewerProtocol", self)
      viewer.log(
        f"[ERROR] Exception during step:\n{self._last_error}",
        VerbosityLevel.SILENT,
      )
      viewer.pause()
      return False

  def reset_environment(self) -> None:
    viewer = cast("_TransitionViewerProtocol", self)
    viewer.env.reset()
    reset_policy_state(viewer.policy)
    self._transition_respawn_counter += 1
    self._transition_spawn = _spawn_on_transition_runway(
      viewer.env.unwrapped,
      self._transition_cfg,
      seed=self._transition_cfg.seed + 7919 * self._transition_respawn_counter,
    )
    active = torch.ones(
      viewer.env.unwrapped.num_envs,
      dtype=torch.bool,
      device=viewer.env.unwrapped.device,
    )
    _update_transition_goal_command(
      viewer.env.unwrapped,
      self._transition_cfg,
      self._transition_spawn,
      active,
    )
    _refresh_transition_respawn_observations(viewer.env.unwrapped)
    self._step_count = 0
    self._sim_budget = 0.0
    self._last_error = None


class StairHeightTransitionNativeViewer(
  StairHeightTransitionPlayMixin,
  NativeMujocoViewer,
):
  pass


class StairHeightTransitionViserViewer(
  StairHeightTransitionPlayMixin,
  ViserPlayViewer,
):
  pass


def run_stair_height_transition_play(
  task_id: str,
  cfg: StairHeightTransitionEvalConfig,
) -> None:
  _validate_transition_config(cfg)
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  runtime = _resolve_eval_runtime(task_id, cast(Any, cfg))
  env_cfg = runtime.env_cfg_factory(True)
  _apply_transition_eval_overrides(
    env_cfg,
    cfg,
    num_envs=cfg.num_envs,
    seed=cfg.seed,
    enable_riser_contact_sensor="g1" in runtime.task_id.lower(),
  )
  env_cfg.viewer.distance = max(env_cfg.viewer.distance, 8.0)
  env_cfg.viewer.elevation = min(env_cfg.viewer.elevation, -25.0)
  env_cfg.viewer.max_extra_envs = max(env_cfg.viewer.max_extra_envs, cfg.num_envs - 1)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=get_clip_actions(runtime.agent_cfg))

  try:
    policy, _runner = load_inference_policy(
      env=wrapped,
      task_id=runtime.task_id,
      agent_cfg=runtime.agent_cfg,
      checkpoint_path=runtime.checkpoint_path,
      device=device,
      runner_cls=runtime.runner_cls,
    )
    spawn = _spawn_on_transition_runway(wrapped.unwrapped, cfg, seed=cfg.seed)
    active = torch.ones(
      cfg.num_envs,
      dtype=torch.bool,
      device=wrapped.unwrapped.device,
    )
    _update_transition_goal_command(wrapped.unwrapped, cfg, spawn, active)
    _fresh_obs_with_history(wrapped.unwrapped)

    if cfg.viewer == "auto":
      has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
      resolved_viewer = "native" if has_display else "viser"
    else:
      resolved_viewer = cfg.viewer

    print(
      "[INFO] Playing stair_height_transition with "
      f"{cfg.num_envs} envs using {resolved_viewer} viewer"
    )
    if resolved_viewer == "native":
      StairHeightTransitionNativeViewer(
        wrapped,
        policy,
        transition_cfg=cfg,
        spawn=spawn,
      ).run()
    elif resolved_viewer == "viser":
      StairHeightTransitionViserViewer(
        wrapped,
        policy,
        transition_cfg=cfg,
        spawn=spawn,
      ).run()
    else:
      raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
  finally:
    wrapped.close()


def _phase_records(records: list[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
  if phase == "all":
    return records
  if phase == "first_touchdown":
    return [record for record in records if bool(record["is_first_stage_touchdown"])]
  if phase == "first_level":
    return [record for record in records if bool(record["is_first_stage_level"])]
  if phase == "later_touchdowns":
    return [
      record
      for record in records
      if int(record["stage_touchdown_index_in_episode"]) > 1
    ]
  if phase == "later_levels":
    return [record for record in records if int(record["landing_level_in_stage"]) >= 2]
  raise ValueError(f"Unknown phase: {phase!r}")


def _mean_or_none(stats: dict[str, Any]) -> float | None:
  value = stats.get("mean")
  if value is None:
    return None
  return float(value)


def _delta_summary(
  stage_a: dict[str, Any],
  stage_b: dict[str, Any],
  *,
  phase: str,
  y_key: str,
  height_delta: float,
) -> dict[str, Any]:
  mean_a = _mean_or_none(stage_a[phase][y_key])
  mean_b = _mean_or_none(stage_b[phase][y_key])
  delta = None if mean_a is None or mean_b is None else mean_b - mean_a
  slope = None if delta is None or abs(height_delta) <= 1.0e-9 else delta / height_delta
  return {
    "phase": phase,
    "metric": y_key,
    "first_stage_mean": mean_a,
    "second_stage_mean": mean_b,
    "delta_second_minus_first": delta,
    "delta_per_height_change": slope,
  }


def _paired_episode_delta_stats(
  records: list[dict[str, Any]],
  *,
  phase: str,
  y_key: str,
  height_delta: float,
) -> dict[str, Any]:
  grouped: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
  for record in _phase_records(records, phase):
    value = record.get(y_key)
    if value is None or not math.isfinite(float(value)):
      continue
    grouped[int(record["episode_global_index"])][int(record["stage_index"])].append(
      float(value)
    )
  deltas = []
  slopes = []
  for stages in grouped.values():
    if 1 not in stages or 2 not in stages:
      continue
    delta = float(np.mean(stages[2]) - np.mean(stages[1]))
    deltas.append(delta)
    if abs(height_delta) > 1.0e-9:
      slopes.append(delta / height_delta)
  return {
    "phase": phase,
    "metric": y_key,
    "paired_episode_count": len(deltas),
    "delta_second_minus_first": _stats(deltas),
    "delta_per_height_change": _stats(slopes),
  }


def _stage_summary(records: list[dict[str, Any]], stage_index: int) -> dict[str, Any]:
  stage_records = [
    record for record in records if int(record["stage_index"]) == stage_index
  ]
  row: dict[str, Any] = {
    "stage_index": stage_index,
    "stage": STAGE_NAMES[stage_index - 1],
    "stair_height_m": (
      float(stage_records[0]["stair_height_m"]) if stage_records else None
    ),
    "records": len(stage_records),
  }
  for phase in (
    "all",
    "first_touchdown",
    "first_level",
    "later_touchdowns",
    "later_levels",
  ):
    phase_subset = _phase_records(stage_records, phase)
    row[phase] = {
      "peak_lift_from_takeoff_m": _stats(
        _record_values(phase_subset, "peak_lift_from_takeoff_m")
      ),
      "peak_clearance_above_terrain_m": _stats(
        _record_values(phase_subset, "peak_clearance_above_terrain_m")
      ),
      "clearance_margin_over_height_m": _stats(
        [
          float(record["peak_lift_from_takeoff_m"]) - float(record["stair_height_m"])
          for record in phase_subset
          if record.get("peak_lift_from_takeoff_m") is not None
        ]
      ),
      "pred_riser_height_mean_m": _stats(
        _record_values(phase_subset, "pred_riser_height_mean_m")
      ),
      "pred_riser_height_takeoff_m": _stats(
        _record_values(phase_subset, "pred_riser_height_takeoff_m")
      ),
    }
  return row


def _summarize_transition_records(
  cfg: StairHeightTransitionEvalConfig,
  episode_summary: dict[str, Any],
  records: list[dict[str, Any]],
) -> dict[str, Any]:
  selected_records = (
    [record for record in records if bool(record.get("episode_success"))]
    if cfg.summarize_success_only
    else list(records)
  )
  by_stage = [_stage_summary(selected_records, 1), _stage_summary(selected_records, 2)]
  height_delta = cfg.second_stair_height - cfg.first_stair_height
  phases = (
    "all",
    "first_touchdown",
    "first_level",
    "later_touchdowns",
    "later_levels",
  )
  linear_fits: dict[str, dict[str, Any]] = {}
  stage_deltas: list[dict[str, Any]] = []
  paired_episode_deltas: list[dict[str, Any]] = []
  for phase in phases:
    phase_subset = _phase_records(selected_records, phase)
    linear_fits[f"{phase}_peak_lift_vs_stage_height"] = _linear_fit(
      phase_subset,
      "peak_lift_from_takeoff_m",
    )
    linear_fits[f"{phase}_pred_riser_vs_stage_height"] = _linear_fit(
      phase_subset,
      "pred_riser_height_mean_m",
    )
    for y_key in ("peak_lift_from_takeoff_m", "pred_riser_height_mean_m"):
      stage_deltas.append(
        _delta_summary(
          by_stage[0],
          by_stage[1],
          phase=phase,
          y_key=y_key,
          height_delta=height_delta,
        )
      )
      paired_episode_deltas.append(
        _paired_episode_delta_stats(
          selected_records,
          phase=phase,
          y_key=y_key,
          height_delta=height_delta,
        )
      )
  return {
    "record_filter": "success_only" if cfg.summarize_success_only else "all_episodes",
    "records_total": len(records),
    "records_used": len(selected_records),
    "episode_summary": episode_summary,
    "by_stage": by_stage,
    "linear_fits": linear_fits,
    "stage_deltas": stage_deltas,
    "paired_episode_deltas": paired_episode_deltas,
    "interpretation": {
      "later_levels_delta_per_height_near_0": (
        "The policy kept nearly the same swing lift after the stair height changed."
      ),
      "later_levels_delta_per_height_near_1": (
        "The policy changed swing lift roughly one-for-one with riser height."
      ),
      "second_first_touchdown": (
        "The first second-stage touchdown is the immediate post-platform adaptation "
        "sample; later second-stage levels show settled behavior after seeing the "
        "new stair run."
      ),
    },
  }


def _episode_summary(
  cfg: StairHeightTransitionEvalConfig,
  batches: list[_TransitionBatchResult],
  step_dt: float,
) -> dict[str, Any]:
  success = [item for batch in batches for item in batch.success]
  first_stage = [item for batch in batches for item in batch.first_stage_reached]
  fell = [item for batch in batches for item in batch.fell]
  heading_failed = [item for batch in batches for item in batch.heading_failed]
  timeout_failed = [item for batch in batches for item in batch.timeout_failed]
  lengths = [item for batch in batches for item in batch.episode_length_steps]
  height_progress = [
    item for batch in batches for item in batch.max_height_progress_fraction
  ]
  goal_progress = [
    item for batch in batches for item in batch.max_goal_progress_fraction
  ]
  stage_progress = [
    item for batch in batches for item in batch.max_stage_progress_fraction
  ]
  episodes = max(1, len(success))
  return {
    "episodes": len(success),
    "success_episodes": int(sum(bool(item) for item in success)),
    "first_stage_reached_episodes": int(sum(bool(item) for item in first_stage)),
    "success_rate": float(sum(success) / episodes),
    "first_stage_reached_rate": float(sum(first_stage) / episodes),
    "fall_rate": float(sum(fell) / episodes),
    "heading_failure_rate": float(sum(heading_failed) / episodes),
    "timeout_failure_rate": float(sum(timeout_failed) / episodes),
    "mean_episode_length_s": float(sum(lengths) * step_dt / episodes),
    "mean_max_height_progress_fraction": float(sum(height_progress) / episodes),
    "mean_max_goal_progress_fraction": float(sum(goal_progress) / episodes),
    "mean_max_stage_progress_fraction": float(sum(stage_progress) / episodes),
    "height_change_m": cfg.second_stair_height - cfg.first_stair_height,
  }


def _resolve_output_path(
  *,
  cfg: StairHeightTransitionEvalConfig,
  task_id: str,
  agent_cfg: Any,
  checkpoint_path: Path,
) -> Path:
  default_name = (
    "stair_height_transition_"
    f"h{int(round(cfg.first_stair_height * 100)):02d}_"
    f"to_h{int(round(cfg.second_stair_height * 100)):02d}.json"
  )
  if cfg.output_file is not None:
    output_path = Path(cfg.output_file)
    if output_path.is_dir() or output_path.suffix.lower() != ".json":
      output_dir = make_timestamped_policy_output_dir(
        output_root=output_path,
        task_id=task_id,
        agent_cfg=agent_cfg,
        checkpoint_path=checkpoint_path,
      )
      return output_dir / default_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

  output_dir = (
    Path(cfg.output_dir)
    if cfg.output_dir is not None
    else make_timestamped_policy_output_dir(
      output_root=cfg.output_root,
      task_id=task_id,
      agent_cfg=agent_cfg,
      checkpoint_path=checkpoint_path,
    )
  )
  output_dir.mkdir(parents=True, exist_ok=True)
  return output_dir / default_name


def _resolve_records_path(
  cfg: StairHeightTransitionEvalConfig,
  output_path: Path,
) -> Path:
  if cfg.records_file is not None:
    path = Path(cfg.records_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
  return output_path.with_name(f"{output_path.stem}_records.csv")


def _write_records_csv(records: list[dict[str, Any]], path: Path) -> None:
  if not records:
    path.write_text("", encoding="utf-8")
    return
  fieldnames = sorted({key for record in records for key in record})
  with path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)


def run_stair_height_transition_eval(
  task_id: str,
  cfg: StairHeightTransitionEvalConfig,
) -> dict[str, Any]:
  _validate_transition_config(cfg)
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  runtime = _resolve_eval_runtime(task_id, cast(Any, cfg))
  agent_cfg = runtime.agent_cfg
  checkpoint_path = runtime.checkpoint_path
  output_path = _resolve_output_path(
    cfg=cfg,
    task_id=runtime.task_id,
    agent_cfg=agent_cfg,
    checkpoint_path=checkpoint_path,
  )
  policy_output_name = get_policy_output_name(
    task_id=runtime.task_id,
    agent_cfg=agent_cfg,
    checkpoint_path=checkpoint_path,
  )
  print(
    "[INFO] stair_height_transition runtime: "
    f"task={runtime.task_id}, env_source={runtime.env_source}, "
    f"actor_obs_dim={runtime.checkpoint_actor_obs_dim}"
  )

  all_records: list[dict[str, Any]] = []
  batches: list[_TransitionBatchResult] = []
  remaining = cfg.episodes
  batch_index = 0
  episode_offset = 0
  global_episode_offset = 0
  while remaining > 0:
    batch_size = min(max(1, cfg.num_envs), remaining)
    print(
      "[INFO] stair_height_transition: "
      f"h1={cfg.first_stair_height:.3f} m "
      f"h2={cfg.second_stair_height:.3f} m "
      f"batch={batch_index} episodes={batch_size}"
    )
    batch = _run_transition_batch(
      task_id=runtime.task_id,
      env_cfg_factory=runtime.env_cfg_factory,
      runner_cls=runtime.runner_cls,
      agent_cfg=agent_cfg,
      checkpoint_path=checkpoint_path,
      cfg=cfg,
      batch_size=batch_size,
      batch_index=batch_index,
      episode_offset=episode_offset,
      global_episode_offset=global_episode_offset,
      device=device,
    )
    all_records.extend(batch.records)
    batches.append(batch)
    remaining -= batch_size
    episode_offset += batch_size
    global_episode_offset += batch_size
    batch_index += 1

  step_dt = batches[0].step_dt if batches else 0.0
  episode_summary = _episode_summary(cfg, batches, step_dt)
  summary = _summarize_transition_records(cfg, episode_summary, all_records)
  geom = _transition_runway_geometry(cfg)
  payload = {
    "task_id": runtime.task_id,
    "requested_task_id": task_id,
    "policy_output_name": policy_output_name,
    "checkpoint": str(checkpoint_path),
    "output_dir": str(output_path.parent),
    "mode": "stair_height_transition",
    "runtime": {
      "env_source": runtime.env_source,
      "checkpoint_actor_obs_dim": runtime.checkpoint_actor_obs_dim,
    },
    "config": asdict(cfg),
    "terrain": {
      "type": "two_stage_stairs_runway",
      "stair_heights_m": [cfg.first_stair_height, cfg.second_stair_height],
      "stair_levels_per_stage": cfg.stair_levels_per_stage,
      "step_width": cfg.step_width,
      "stair_width": cfg.stair_width,
      "side_margin_width": cfg.side_margin_width,
      "flat_apron_width": cfg.flat_apron_width,
      "middle_platform_width": cfg.middle_platform_width,
      "final_platform_width": cfg.final_platform_width,
      "terrain_border_width": cfg.terrain_border_width,
      "tile_size_m": [geom.tile_length, geom.tile_width],
      "route_length_m": geom.route_length,
      "first_top_height_m": geom.first_top_height,
      "final_top_height_m": geom.final_top_height,
    },
    "navigation": {
      "goal_radius": cfg.goal_radius,
      "goal_height_tolerance": cfg.goal_height_tolerance,
      "goal_speed": cfg.goal_speed,
      "yaw_kp": cfg.yaw_kp,
      "yaw_rate_limit": cfg.yaw_rate_limit,
      "heading_failure_angle_deg": cfg.heading_failure_angle_deg,
      "heading_failure_grace_s": cfg.heading_failure_grace_s,
    },
    "summary": summary,
  }
  output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
  print(f"[INFO] Wrote stair height-transition eval to {output_path}")
  if cfg.write_records_csv:
    records_path = _resolve_records_path(cfg, output_path)
    _write_records_csv(all_records, records_path)
    print(f"[INFO] Wrote stair height-transition records to {records_path}")

  later_delta = next(
    item
    for item in summary["stage_deltas"]
    if item["phase"] == "later_levels" and item["metric"] == "peak_lift_from_takeoff_m"
  )
  pred_delta = next(
    item
    for item in summary["stage_deltas"]
    if item["phase"] == "later_levels" and item["metric"] == "pred_riser_height_mean_m"
  )
  print(
    "[INFO] stair_height_transition: "
    f"success={episode_summary['success_rate']:.3f}, "
    f"stage1_reached={episode_summary['first_stage_reached_rate']:.3f}, "
    f"later_lift_delta_per_height={later_delta['delta_per_height_change']}, "
    f"later_pred_delta_per_height={pred_delta['delta_per_height_change']}"
  )
  return payload


def main() -> None:
  import mjlab.tasks  # noqa: F401

  chosen_task, remaining_args = _task_selector_from_argv(sys.argv[1:])
  cfg = tyro.cli(
    StairHeightTransitionEvalConfig,
    args=_normalize_standalone_bool_flags(
      remaining_args,
      ("--write-records-csv", "--allow-out-of-train-range", "--play"),
    ),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  try:
    if cfg.play:
      run_stair_height_transition_play(chosen_task, cfg)
      return
    run_stair_height_transition_eval(chosen_task, cfg)
  except Exception:
    traceback.print_exc()
    raise


if __name__ == "__main__":
  main()
