"""Evaluate whether stair-climbing foot lift scales with stair height."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import tyro
from scripts.velocity_eval.eval_metrics import _current_step_boundaries
from scripts.velocity_eval.eval_policy_goal_pyramid import (
  GoalPyramidEvalConfig,
  GoalPyramidNativeViewer,
  GoalPyramidViserViewer,
  _computed_start_distance,
  _force_single_goal_terrain_tile,
  _fresh_obs_with_history,
  _goal_reached,
  _heading_failure,
  _make_goal_terrain,
  _normalize_standalone_bool_flags,
  _spawn_on_pyramid_apron,
  _update_goal_command,
)
from scripts.velocity_eval.eval_terrains import apply_eval_overrides
from scripts.velocity_eval.policy_io import (
  get_clip_actions,
  get_policy_output_name,
  load_checkpoint_agent_cfg,
  load_inference_policy,
  make_timestamped_policy_output_dir,
  resolve_checkpoint_path,
  resolve_inference_agent_cfg,
)

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp.rewards import _StepBoundaryFootVolume
from mjlab.tasks.velocity.mdp.stair_geometry import cached_stair_shape
from mjlab.utils.lstm import reset_policy_state_from_step
from mjlab.utils.torch import configure_torch_backends

FOOT_NAMES = ("left", "right")
AUTO_TASK_ALIASES = ("auto", "checkpoint", "from-checkpoint")
LEGACY_G1_NO_HEIGHT_SCAN_TASK_ID = "Mjlab-Velocity-Legacy-G1-MLP-NoHeightScan"
LEGACY_G1_HEIGHT_SCAN_TASK_ID = "Mjlab-Velocity-Legacy-G1-MLP-HeightScan"
LEGACY_G1_BLIND_HISTORY_TASK_ID = "Mjlab-Velocity-Legacy-G1-BlindHistory"
LEGACY_G1_SLOW_LATENT_TASK_ID = "Mjlab-Velocity-Legacy-G1-SlowLatent"


@dataclass(frozen=True)
class _EvalRuntime:
  task_id: str
  agent_cfg: Any
  checkpoint_path: Path
  env_cfg_factory: Callable[[bool], ManagerBasedRlEnvCfg]
  runner_cls: type | None
  env_source: str
  checkpoint_actor_obs_dim: int | None = None


@dataclass(frozen=True)
class StairLiftHeightSweepConfig:
  """Configuration for the stair-height/lift-height correlation experiment."""

  checkpoint_file: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  stair_heights: list[float] = field(
    default_factory=lambda: [0.09, 0.11, 0.13, 0.15, 0.18, 0.21, 0.24]
  )
  episodes_per_height: int = 40
  num_envs: int = 40
  max_episode_length_s: float = 12.0
  seed: int = 12345
  device: str | None = None
  output_root: str = "eval_outputs/stair_lift_height_sweep"
  output_dir: str | None = None
  output_file: str | None = None
  records_file: str | None = None
  write_records_csv: bool = True
  clean_observations: bool = True
  disable_observation_delay: bool = True
  disable_actuator_delay: bool = True
  play: bool = False
  viewer: Literal["auto", "native", "viser"] = "auto"
  play_stair_height: float | None = None

  stair_levels: int = 10
  step_width: float = 0.30
  platform_width: float = 3.0
  flat_apron_width: float = 3.0
  terrain_border_width: float = 12.0
  start_distance: float | None = None
  spawn_tangent_half_width: float | None = None
  spawn_tangent_margin: float = 0.35
  start_z_offset: float = 0.03
  goal_radius: float = 0.75
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
class _LiftTrackerParams:
  ground_contact_sensor_name: str
  height_sensor_name: str
  landing_height_tolerance: float
  min_candidate_support: float
  min_swing_duration_s: float
  min_peak_lift_m: float


@dataclass
class _HeightBatchResult:
  stair_height_m: float
  batch_index: int
  episode_offset: int
  success: list[bool]
  fell: list[bool]
  heading_failed: list[bool]
  timeout_failed: list[bool]
  episode_length_steps: list[float]
  max_height_progress_fraction: list[float]
  max_goal_progress_fraction: list[float]
  step_dt: float
  records: list[dict[str, Any]] = field(default_factory=list)


def _finite_float(value: Any) -> float | None:
  if value is None:
    return None
  value = float(value)
  return value if math.isfinite(value) else None


def _diag_vector(
  diagnostics: dict[str, torch.Tensor],
  key: str,
  *,
  num_envs: int,
  component: int | None = None,
) -> torch.Tensor | None:
  value = diagnostics.get(key)
  if value is None:
    return None
  if value.dim() >= 3 and value.shape[0] == 1 and value.shape[1] == num_envs:
    value = value.squeeze(0)
  if component is not None:
    if value.shape[-1] <= component:
      return None
    value = value[..., component]
  value = value.reshape(num_envs, -1)
  if value.shape[1] == 0:
    return None
  return value[:, 0]


def _task_selector_from_argv(argv: list[str]) -> tuple[str, list[str]]:
  """Return an optional leading task selector and the remaining CLI args."""
  if argv and not argv[0].startswith("-"):
    return argv[0], argv[1:]
  return "auto", argv


def _infer_registered_task_from_checkpoint_path(
  checkpoint_path: Path,
  registered_tasks: list[str],
) -> str | None:
  """Infer a registered task id from a checkpoint path, if one is embedded."""
  path_text = str(checkpoint_path)
  ordered_tasks = sorted(registered_tasks, key=lambda item: len(item), reverse=True)
  for task_id in ordered_tasks:
    if task_id in checkpoint_path.parts or task_id in path_text:
      return task_id
  return None


def _checkpoint_actor_state_dict(payload: Any) -> dict[str, Any]:
  if not isinstance(payload, dict):
    return {}
  for key in ("actor_state_dict", "model_state_dict", "state_dict"):
    value = payload.get(key)
    if isinstance(value, dict):
      return value
  return payload


def _actor_obs_dim_from_checkpoint_payload(payload: Any) -> int | None:
  """Read the actor observation dimension from a loaded checkpoint payload."""
  state_dict = _checkpoint_actor_state_dict(payload)
  normalizer_suffixes = (
    "obs_normalizer._mean",
    "actor_obs_normalizer._mean",
  )
  for key, value in state_dict.items():
    if not any(key.endswith(suffix) for suffix in normalizer_suffixes):
      continue
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 1:
      return int(shape[-1])

  first_linear_suffixes = (
    "mlp.0.weight",
    "actor.0.weight",
    "actor.mlp.0.weight",
  )
  for key, value in state_dict.items():
    if not any(key.endswith(suffix) for suffix in first_linear_suffixes):
      continue
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) == 2:
      return int(shape[1])
  return None


def _checkpoint_actor_obs_dim(checkpoint_path: Path) -> int | None:
  payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  return _actor_obs_dim_from_checkpoint_payload(payload)


def _agent_actor_class_name(agent_cfg: Any) -> str:
  actor_cfg = (
    agent_cfg.get("actor", {}) if isinstance(agent_cfg, dict) else agent_cfg.actor
  )
  if isinstance(actor_cfg, dict):
    class_name = actor_cfg.get("class_name", "")
  else:
    class_name = getattr(actor_cfg, "class_name", "")
  return str(class_name)


def _legacy_g1_runtime_from_checkpoint(
  *,
  task_selector: str,
  checkpoint_path: Path,
  actor_obs_dim: int | None,
) -> _EvalRuntime:
  """Choose a built-in G1 eval config for unregistered legacy checkpoints."""
  from mjlab.tasks.velocity.config.g1.blind_rough_slow_latent_env_cfg import (
    unitree_g1_blind_rough_target_navigation_slow_latent_env_cfg,
  )
  from mjlab.tasks.velocity.config.g1.blind_rough_teacher_kl_env_cfg import (
    unitree_g1_blind_rough_teacherkl_env_cfg,
  )
  from mjlab.tasks.velocity.config.g1.env_cfgs import (
    unitree_g1_flat_env_cfg,
    unitree_g1_rough_env_cfg,
  )
  from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
  from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner

  agent_cfg = resolve_inference_agent_cfg(
    checkpoint_path=checkpoint_path,
    agent_cfg=unitree_g1_ppo_runner_cfg(),
  )
  actor_class_name = _agent_actor_class_name(agent_cfg)
  if "SlowLatent" in actor_class_name or "GatedStairLatent" in actor_class_name:
    return _EvalRuntime(
      task_id=LEGACY_G1_SLOW_LATENT_TASK_ID,
      agent_cfg=agent_cfg,
      checkpoint_path=checkpoint_path,
      env_cfg_factory=unitree_g1_blind_rough_target_navigation_slow_latent_env_cfg,
      runner_cls=VelocityOnPolicyRunner,
      env_source="legacy_g1_slow_latent_checkpoint",
      checkpoint_actor_obs_dim=actor_obs_dim,
    )

  if actor_obs_dim == 98:
    return _EvalRuntime(
      task_id=LEGACY_G1_NO_HEIGHT_SCAN_TASK_ID,
      agent_cfg=agent_cfg,
      checkpoint_path=checkpoint_path,
      env_cfg_factory=unitree_g1_flat_env_cfg,
      runner_cls=VelocityOnPolicyRunner,
      env_source="legacy_g1_mlp_no_height_scan",
      checkpoint_actor_obs_dim=actor_obs_dim,
    )
  if actor_obs_dim == 285:
    return _EvalRuntime(
      task_id=LEGACY_G1_HEIGHT_SCAN_TASK_ID,
      agent_cfg=agent_cfg,
      checkpoint_path=checkpoint_path,
      env_cfg_factory=unitree_g1_rough_env_cfg,
      runner_cls=VelocityOnPolicyRunner,
      env_source="legacy_g1_mlp_height_scan",
      checkpoint_actor_obs_dim=actor_obs_dim,
    )
  if actor_obs_dim == 490:
    return _EvalRuntime(
      task_id=LEGACY_G1_BLIND_HISTORY_TASK_ID,
      agent_cfg=agent_cfg,
      checkpoint_path=checkpoint_path,
      env_cfg_factory=unitree_g1_blind_rough_teacherkl_env_cfg,
      runner_cls=VelocityOnPolicyRunner,
      env_source="legacy_g1_blind_history",
      checkpoint_actor_obs_dim=actor_obs_dim,
    )

  hint = (
    f"task selector {task_selector!r} is not registered, and checkpoint "
    f"actor obs dim {actor_obs_dim!r} is not one of the built-in G1 legacy "
    "presets: 98=no height scan, 285=height scan, 490=blind history."
  )
  raise ValueError(hint)


def _registered_runtime(
  *,
  task_id: str,
  cfg: StairLiftHeightSweepConfig,
  checkpoint_path: Path | None = None,
) -> _EvalRuntime:
  agent_cfg = load_rl_cfg(task_id)
  checkpoint_path = checkpoint_path or resolve_checkpoint_path(
    task_id=task_id,
    agent_cfg=agent_cfg,
    checkpoint_file=cfg.checkpoint_file,
    wandb_run_path=cfg.wandb_run_path,
    wandb_checkpoint_name=cfg.wandb_checkpoint_name,
  )
  agent_cfg = resolve_inference_agent_cfg(
    checkpoint_path=checkpoint_path,
    agent_cfg=agent_cfg,
  )

  def env_cfg_factory(
    play: bool = False, *, _task_id: str = task_id
  ) -> ManagerBasedRlEnvCfg:
    return load_env_cfg(_task_id, play=play)

  return _EvalRuntime(
    task_id=task_id,
    agent_cfg=agent_cfg,
    checkpoint_path=checkpoint_path,
    env_cfg_factory=env_cfg_factory,
    runner_cls=None,
    env_source="registered_task",
    checkpoint_actor_obs_dim=None,
  )


def _resolve_eval_runtime(
  task_selector: str,
  cfg: StairLiftHeightSweepConfig,
) -> _EvalRuntime:
  registered_tasks = list_tasks()
  if task_selector in registered_tasks:
    return _registered_runtime(task_id=task_selector, cfg=cfg)

  if cfg.checkpoint_file is None:
    known = ", ".join((*AUTO_TASK_ALIASES, *registered_tasks))
    raise ValueError(
      "Provide --checkpoint-file when using auto or an unregistered task selector. "
      f"Known registered/auto selectors include: {known}"
    )

  checkpoint_path = Path(cfg.checkpoint_file).expanduser()
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

  inferred_task = _infer_registered_task_from_checkpoint_path(
    checkpoint_path, registered_tasks
  )
  if inferred_task is not None:
    return _registered_runtime(
      task_id=inferred_task,
      cfg=cfg,
      checkpoint_path=checkpoint_path,
    )

  actor_obs_dim = _checkpoint_actor_obs_dim(checkpoint_path)
  if load_checkpoint_agent_cfg(checkpoint_path) is None:
    print(
      "[WARN] No params/agent.yaml found next to checkpoint; using the default "
      "G1 PPO architecture for legacy inference."
    )
  return _legacy_g1_runtime_from_checkpoint(
    task_selector=task_selector,
    checkpoint_path=checkpoint_path,
    actor_obs_dim=actor_obs_dim,
  )


class StairLiftHeightTracker:
  """Track foot swing peaks and assign each touchdown to a stair level."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    *,
    params: _LiftTrackerParams,
    stair_height_m: float,
    height_index: int,
    batch_index: int,
    episode_offset: int,
    global_episode_offset: int,
    max_levels: int,
  ) -> None:
    self.params = params
    self.stair_height_m = float(stair_height_m)
    self.height_index = int(height_index)
    self.batch_index = int(batch_index)
    self.episode_offset = int(episode_offset)
    self.global_episode_offset = int(global_episode_offset)
    self.max_levels = int(max_levels)

    self.foot_body_cfg = SceneEntityCfg(
      "robot",
      body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
      preserve_order=True,
    )
    self.foot_body_cfg.resolve(env.scene)
    self._volume = _StepBoundaryFootVolume(
      RewardTermCfg(func=StairLiftHeightTracker, weight=0.0, params={}),
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
      raise RuntimeError("StairLiftHeightTracker foot body IDs were not resolved.")
    return body_ids

  def _foot_z(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    asset = env.scene[self.foot_body_cfg.name]
    return asset.data.body_link_pos_w[:, self._foot_body_ids(), 2]

  def _foot_clearance(self) -> torch.Tensor:
    heights = self._height_sensor.data.heights
    if heights.ndim == 3:
      heights = heights.min(dim=-1).values
    return heights

  @staticmethod
  def _boundary_levels(
    boundaries: torch.Tensor,
    terrain_height_m: float,
    max_levels: int,
    valid: torch.Tensor,
  ) -> torch.Tensor:
    z_min = torch.minimum(boundaries[..., 9], boundaries[..., 10])
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
    empty_float = torch.zeros(num_envs, num_feet, device=device)
    if not bool(touchdown.any().item()):
      return {
        "valid": torch.zeros_like(touchdown),
        "level": invalid_level,
        "support": empty_float,
        "target_z": empty_float,
        "riser_height": empty_float,
      }

    boundaries, valid_boundaries = _current_step_boundaries(env)
    if boundaries is None or valid_boundaries is None or boundaries.shape[1] == 0:
      return {
        "valid": torch.zeros_like(touchdown),
        "level": invalid_level,
        "support": empty_float,
        "target_z": empty_float,
        "riser_height": empty_float,
      }

    points_w, _point_vel_w = self._volume._foot_points_w(env, self.foot_body_cfg)
    foot_ref_w = self._volume._foot_ref_w(env, self.foot_body_cfg)
    sole_z = torch.min(self._volume._local_points[:, 2])
    sole_mask = self._volume._local_points[:, 2] <= sole_z + 1.0e-6
    sole_points_w = points_w[:, :, sole_mask, :]

    tread_depth, _riser_height, shape_valid = cached_stair_shape(
      env, boundaries, valid_boundaries
    )
    support_fraction = mdp.toe_step_riser_slab_penalty._tread_support_fraction(
      sole_points_w,
      boundaries,
      tread_depth,
    )
    levels = self._boundary_levels(
      boundaries,
      self.stair_height_m,
      self.max_levels,
      valid_boundaries,
    )

    height_error = torch.abs(foot_ref_w[:, :, None, 2] - boundaries[:, None, :, 10])
    valid_stair_boundary = valid_boundaries & shape_valid[:, None] & (levels >= 1)
    candidate = (
      valid_stair_boundary[:, None, :]
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
    level = levels.gather(1, safe_idx)
    target_z = boundaries[..., 10].gather(1, safe_idx)
    riser_height = torch.abs(boundaries[..., 10] - boundaries[..., 9]).gather(
      1, safe_idx
    )

    return {
      "valid": has_candidate,
      "level": torch.where(has_candidate, level, invalid_level),
      "support": torch.where(has_candidate, best_support.clamp_min(0.0), empty_float),
      "target_z": torch.where(has_candidate, target_z, empty_float),
      "riser_height": torch.where(has_candidate, riser_height, empty_float),
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
        valid, stair_prob, torch.zeros_like(stair_prob)
      )
      finite_sample |= valid
    if values["event_prob"] is not None:
      event_prob = cast(torch.Tensor, values["event_prob"])[:, None].expand_as(
        self._pred_event_prob_sum
      )
      valid = active_swing & torch.isfinite(event_prob)
      self._pred_event_prob_sum += torch.where(
        valid, event_prob, torch.zeros_like(event_prob)
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
        diagnostics, "stair_shape", num_envs=env.num_envs, component=1
      )
      if riser is not None:
        riser = riser[:, None].expand_as(self._takeoff_pred_riser)
        self._takeoff_pred_riser = torch.where(
          start & torch.isfinite(riser), riser, self._takeoff_pred_riser
        )

    active_swing = self._in_swing & active_feet
    self._peak_foot_z = torch.where(
      active_swing, torch.maximum(self._peak_foot_z, foot_z), self._peak_foot_z
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
      diagnostics, "stair_shape", num_envs=env.num_envs, component=1
    )
    slots = complete.nonzero(as_tuple=False)
    for env_id_t, foot_id_t in slots.detach().cpu():
      env_id = int(env_id_t.item())
      foot_id = int(foot_id_t.item())
      self._touchdown_counter[env_id] += 1
      touchdown_index = int(self._touchdown_counter[env_id].item())
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

      episode_index_for_height = self.episode_offset + env_id
      episode_global_index = self.global_episode_offset + env_id
      touchdown_pred_value = (
        None if touchdown_pred is None else float(touchdown_pred[env_id].item())
      )
      record = {
        "height_index": self.height_index,
        "stair_height_m": self.stair_height_m,
        "batch_index": self.batch_index,
        "env_index": env_id,
        "episode_index_for_height": episode_index_for_height,
        "episode_global_index": episode_global_index,
        "foot_index": foot_id,
        "foot": FOOT_NAMES[foot_id] if foot_id < len(FOOT_NAMES) else str(foot_id),
        "touchdown_index_in_episode": touchdown_index,
        "landing_level_low_to_high": int(candidates["level"][env_id, foot_id].item()),
        "is_first_stair_touchdown": touchdown_index == 1,
        "is_first_stair_level": int(candidates["level"][env_id, foot_id].item()) == 1,
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


def _goal_cfg_for_height(
  cfg: StairLiftHeightSweepConfig,
  stair_height_m: float,
) -> GoalPyramidEvalConfig:
  return GoalPyramidEvalConfig(
    checkpoint_file=cfg.checkpoint_file,
    wandb_run_path=cfg.wandb_run_path,
    wandb_checkpoint_name=cfg.wandb_checkpoint_name,
    episodes=cfg.episodes_per_height,
    num_envs=cfg.num_envs,
    max_episode_length_s=cfg.max_episode_length_s,
    seed=cfg.seed,
    device=cfg.device,
    output_root=cfg.output_root,
    clean_observations=cfg.clean_observations,
    disable_observation_delay=cfg.disable_observation_delay,
    disable_actuator_delay=cfg.disable_actuator_delay,
    play=cfg.play,
    viewer=cfg.viewer,
    stair_levels=cfg.stair_levels,
    stair_height=stair_height_m,
    step_width=cfg.step_width,
    platform_width=cfg.platform_width,
    flat_apron_width=cfg.flat_apron_width,
    terrain_border_width=cfg.terrain_border_width,
    start_distance=cfg.start_distance,
    spawn_tangent_half_width=cfg.spawn_tangent_half_width,
    spawn_tangent_margin=cfg.spawn_tangent_margin,
    start_z_offset=cfg.start_z_offset,
    goal_radius=cfg.goal_radius,
    goal_height_tolerance=cfg.goal_height_tolerance,
    goal_speed=cfg.goal_speed,
    yaw_kp=cfg.yaw_kp,
    yaw_rate_limit=cfg.yaw_rate_limit,
    heading_failure_angle_deg=cfg.heading_failure_angle_deg,
    heading_failure_grace_s=cfg.heading_failure_grace_s,
  )


def _run_height_batch(
  *,
  task_id: str,
  env_cfg_factory: Callable[[bool], ManagerBasedRlEnvCfg],
  runner_cls: type | None,
  agent_cfg: Any,
  checkpoint_path: Path,
  cfg: StairLiftHeightSweepConfig,
  goal_cfg: GoalPyramidEvalConfig,
  height_index: int,
  batch_size: int,
  batch_index: int,
  episode_offset: int,
  global_episode_offset: int,
  device: str,
) -> _HeightBatchResult:
  terrain = _make_goal_terrain(goal_cfg)
  env_cfg = env_cfg_factory(False)
  apply_eval_overrides(
    env_cfg,
    terrain,
    num_envs=batch_size,
    seed=cfg.seed + 1009 * batch_index,
    max_episode_length_s=cfg.max_episode_length_s,
    command=(0.0, 0.0, 0.0),
    clean_observations=cfg.clean_observations,
    disable_observation_delay=cfg.disable_observation_delay,
    disable_actuator_delay=cfg.disable_actuator_delay,
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
    tracker = StairLiftHeightTracker(
      wrapped.unwrapped,
      params=_LiftTrackerParams(
        ground_contact_sensor_name=cfg.ground_contact_sensor_name,
        height_sensor_name=cfg.height_sensor_name,
        landing_height_tolerance=cfg.landing_height_tolerance,
        min_candidate_support=cfg.min_candidate_support,
        min_swing_duration_s=cfg.min_swing_duration_s,
        min_peak_lift_m=cfg.min_peak_lift_m,
      ),
      stair_height_m=goal_cfg.stair_height,
      height_index=height_index,
      batch_index=batch_index,
      episode_offset=episode_offset,
      global_episode_offset=global_episode_offset,
      max_levels=goal_cfg.stair_levels,
    )
    spawn = _spawn_on_pyramid_apron(
      wrapped.unwrapped,
      goal_cfg,
      seed=cfg.seed + 1543 * batch_index,
    )

    all_envs_active = torch.ones(batch_size, dtype=torch.bool, device=device)
    _update_goal_command(wrapped.unwrapped, goal_cfg, spawn, all_envs_active)
    obs = _fresh_obs_with_history(wrapped.unwrapped)

    step_counts = torch.zeros(batch_size, device=device)
    done_envs = torch.zeros(batch_size, dtype=torch.bool, device=device)
    success = torch.zeros(batch_size, dtype=torch.bool, device=device)
    fell = torch.zeros(batch_size, dtype=torch.bool, device=device)
    heading_failed = torch.zeros(batch_size, dtype=torch.bool, device=device)
    timeout_failed = torch.zeros(batch_size, dtype=torch.bool, device=device)
    max_height_progress = torch.zeros(batch_size, device=device)
    max_goal_progress = torch.zeros(batch_size, device=device)
    start_distance = torch.full(
      (batch_size,),
      _computed_start_distance(goal_cfg),
      device=device,
      dtype=torch.float32,
    )

    max_steps = wrapped.unwrapped.max_episode_length + 2
    for step_index in range(max_steps):
      active = ~done_envs
      reached_now = _goal_reached(wrapped.unwrapped, goal_cfg, spawn) & active
      if bool(reached_now.any().item()):
        success |= reached_now
        done_envs |= reached_now
        active = ~done_envs
      if not bool(active.any().item()):
        break

      asset = wrapped.unwrapped.scene["robot"]
      estimated_support_z = asset.data.root_link_pos_w[:, 2] - spawn.nominal_root_height
      bottom_z_w = spawn.top_z_w - float(goal_cfg.stair_levels) * goal_cfg.stair_height
      height_progress = torch.clamp(
        (estimated_support_z - bottom_z_w)
        / max(float(goal_cfg.stair_levels) * goal_cfg.stair_height, 1.0e-6),
        min=0.0,
        max=1.0,
      )
      goal_distance = torch.norm(
        asset.data.root_link_pos_w[:, :2] - spawn.goal_xy_w, dim=-1
      )
      goal_progress = torch.clamp(
        (start_distance - goal_distance) / start_distance.clamp_min(1.0e-6),
        min=0.0,
        max=1.0,
      )
      max_height_progress = torch.where(
        active, torch.maximum(max_height_progress, height_progress), max_height_progress
      )
      max_goal_progress = torch.where(
        active, torch.maximum(max_goal_progress, goal_progress), max_goal_progress
      )
      step_counts += active.float()

      _update_goal_command(wrapped.unwrapped, goal_cfg, spawn, active)
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

      reached_after_step = _goal_reached(wrapped.unwrapped, goal_cfg, spawn) & active
      reached_after_step &= ~newly_done
      heading_failed_now, _heading_error = _heading_failure(
        wrapped.unwrapped,
        goal_cfg,
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
    fell_list = fell.detach().cpu().tolist()
    heading_failed_list = heading_failed.detach().cpu().tolist()
    timeout_failed_list = timeout_failed.detach().cpu().tolist()
    for record in tracker.records:
      local_env = int(record["env_index"])
      record["episode_success"] = bool(success_list[local_env])
      record["episode_fell"] = bool(fell_list[local_env])
      record["episode_heading_failed"] = bool(heading_failed_list[local_env])
      record["episode_timeout_failed"] = bool(timeout_failed_list[local_env])

    return _HeightBatchResult(
      stair_height_m=goal_cfg.stair_height,
      batch_index=batch_index,
      episode_offset=episode_offset,
      success=success_list,
      fell=fell_list,
      heading_failed=heading_failed_list,
      timeout_failed=timeout_failed_list,
      episode_length_steps=step_counts.detach().cpu().tolist(),
      max_height_progress_fraction=max_height_progress.detach().cpu().tolist(),
      max_goal_progress_fraction=max_goal_progress.detach().cpu().tolist(),
      step_dt=float(wrapped.unwrapped.step_dt),
      records=tracker.records,
    )
  finally:
    wrapped.close()


def _resolve_play_stair_height(cfg: StairLiftHeightSweepConfig) -> float:
  if cfg.play_stair_height is not None:
    return float(cfg.play_stair_height)
  if not cfg.stair_heights:
    raise ValueError("Provide at least one --stair-heights value for --play.")
  stair_height_m = float(cfg.stair_heights[0])
  if len(cfg.stair_heights) > 1:
    print(
      "[INFO] stair_lift_sweep --play uses one height at a time; "
      f"using the first --stair-heights value ({stair_height_m:.3f} m). "
      "Pass --play-stair-height to choose another height."
    )
  return stair_height_m


def run_stair_lift_height_sweep_play(
  task_id: str,
  cfg: StairLiftHeightSweepConfig,
) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  runtime = _resolve_eval_runtime(task_id, cfg)
  stair_height_m = _resolve_play_stair_height(cfg)
  goal_cfg = _goal_cfg_for_height(cfg, stair_height_m)
  terrain = _make_goal_terrain(goal_cfg)

  env_cfg = runtime.env_cfg_factory(True)
  apply_eval_overrides(
    env_cfg,
    terrain,
    num_envs=cfg.num_envs,
    seed=cfg.seed,
    max_episode_length_s=cfg.max_episode_length_s,
    command=(0.0, 0.0, 0.0),
    clean_observations=cfg.clean_observations,
    disable_observation_delay=cfg.disable_observation_delay,
    disable_actuator_delay=cfg.disable_actuator_delay,
    enable_riser_contact_sensor="g1" in runtime.task_id.lower(),
  )
  _force_single_goal_terrain_tile(env_cfg)
  env_cfg.viewer.distance = max(env_cfg.viewer.distance, 8.0)
  env_cfg.viewer.elevation = min(env_cfg.viewer.elevation, -30.0)
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
    spawn = _spawn_on_pyramid_apron(wrapped.unwrapped, goal_cfg, seed=cfg.seed)
    active = torch.ones(
      cfg.num_envs,
      dtype=torch.bool,
      device=wrapped.unwrapped.device,
    )
    _update_goal_command(wrapped.unwrapped, goal_cfg, spawn, active)
    _fresh_obs_with_history(wrapped.unwrapped)

    if cfg.viewer == "auto":
      has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
      resolved_viewer = "native" if has_display else "viser"
    else:
      resolved_viewer = cfg.viewer

    print(
      "[INFO] Playing stair_lift_sweep height "
      f"{stair_height_m:.3f} m with {cfg.num_envs} envs using "
      f"{resolved_viewer} viewer"
    )
    if resolved_viewer == "native":
      GoalPyramidNativeViewer(
        wrapped,
        policy,
        goal_cfg=goal_cfg,
        spawn=spawn,
      ).run()
    elif resolved_viewer == "viser":
      GoalPyramidViserViewer(
        wrapped,
        policy,
        goal_cfg=goal_cfg,
        spawn=spawn,
      ).run()
    else:
      raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
  finally:
    wrapped.close()


def _stats(values: list[float]) -> dict[str, float | int | None]:
  clean = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
  if clean.size == 0:
    return {
      "count": 0,
      "mean": None,
      "std": None,
      "min": None,
      "p10": None,
      "p50": None,
      "p90": None,
      "max": None,
    }
  return {
    "count": int(clean.size),
    "mean": float(np.mean(clean)),
    "std": float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0,
    "min": float(np.min(clean)),
    "p10": float(np.quantile(clean, 0.10)),
    "p50": float(np.quantile(clean, 0.50)),
    "p90": float(np.quantile(clean, 0.90)),
    "max": float(np.max(clean)),
  }


def _linear_fit(records: list[dict[str, Any]], y_key: str) -> dict[str, Any]:
  pairs = [
    (float(record["stair_height_m"]), float(record[y_key]))
    for record in records
    if record.get(y_key) is not None
    and math.isfinite(float(record["stair_height_m"]))
    and math.isfinite(float(record[y_key]))
  ]
  if len(pairs) < 2:
    return {
      "x_key": "stair_height_m",
      "y_key": y_key,
      "count": len(pairs),
      "slope": None,
      "intercept": None,
      "r2": None,
      "pearson_r": None,
    }
  x = np.asarray([pair[0] for pair in pairs], dtype=float)
  y = np.asarray([pair[1] for pair in pairs], dtype=float)
  x_var = float(np.var(x))
  y_var = float(np.var(y))
  if x_var <= 0.0:
    return {
      "x_key": "stair_height_m",
      "y_key": y_key,
      "count": len(pairs),
      "slope": None,
      "intercept": None,
      "r2": None,
      "pearson_r": None,
    }
  slope, intercept = np.polyfit(x, y, deg=1)
  predicted = slope * x + intercept
  residual = float(np.sum(np.square(y - predicted)))
  total = float(np.sum(np.square(y - np.mean(y))))
  r2 = None if total <= 0.0 else float(1.0 - residual / total)
  pearson_r = None if y_var <= 0.0 else float(np.corrcoef(x, y)[0, 1])
  return {
    "x_key": "stair_height_m",
    "y_key": y_key,
    "count": len(pairs),
    "slope": float(slope),
    "intercept": float(intercept),
    "r2": r2,
    "pearson_r": pearson_r,
  }


def _record_values(records: list[dict[str, Any]], key: str) -> list[float]:
  return [
    float(record[key])
    for record in records
    if record.get(key) is not None and math.isfinite(float(record[key]))
  ]


def _phase_records(records: list[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
  if phase == "all":
    return records
  if phase == "first_touchdown":
    return [record for record in records if bool(record["is_first_stair_touchdown"])]
  if phase == "first_level":
    return [record for record in records if bool(record["is_first_stair_level"])]
  if phase == "later_touchdowns":
    return [
      record for record in records if int(record["touchdown_index_in_episode"]) > 1
    ]
  if phase == "later_levels":
    return [
      record for record in records if int(record["landing_level_low_to_high"]) >= 2
    ]
  raise ValueError(f"Unknown phase: {phase!r}")


def _summarize_records(
  cfg: StairLiftHeightSweepConfig,
  height_results: list[dict[str, Any]],
  records: list[dict[str, Any]],
) -> dict[str, Any]:
  selected_records = (
    [record for record in records if bool(record.get("episode_success"))]
    if cfg.summarize_success_only
    else list(records)
  )
  phases = ("all", "first_touchdown", "first_level", "later_touchdowns", "later_levels")
  by_height = []
  for height in cfg.stair_heights:
    height_records = [
      record
      for record in selected_records
      if math.isclose(float(record["stair_height_m"]), float(height), abs_tol=1.0e-9)
    ]
    result = next(
      item
      for item in height_results
      if math.isclose(float(item["stair_height_m"]), float(height), abs_tol=1.0e-9)
    )
    row: dict[str, Any] = {
      "stair_height_m": float(height),
      "episodes": result["episodes"],
      "success_rate": result["success_rate"],
      "mean_max_height_progress_fraction": result["mean_max_height_progress_fraction"],
      "records": len(height_records),
    }
    for phase in phases:
      phase_subset = _phase_records(height_records, phase)
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
      }
    by_height.append(row)

  linear_fits: dict[str, dict[str, Any]] = {}
  for phase in phases:
    phase_subset = _phase_records(selected_records, phase)
    linear_fits[f"{phase}_peak_lift_vs_height"] = _linear_fit(
      phase_subset, "peak_lift_from_takeoff_m"
    )
    linear_fits[f"{phase}_peak_clearance_vs_height"] = _linear_fit(
      phase_subset, "peak_clearance_above_terrain_m"
    )
    linear_fits[f"{phase}_pred_riser_vs_height"] = _linear_fit(
      phase_subset, "pred_riser_height_mean_m"
    )

  return {
    "record_filter": "success_only" if cfg.summarize_success_only else "all_episodes",
    "records_total": len(records),
    "records_used": len(selected_records),
    "by_height": by_height,
    "linear_fits": linear_fits,
    "interpretation": {
      "peak_lift_slope_near_0": (
        "The policy is using an almost fixed foot-lift amplitude across stair heights."
      ),
      "peak_lift_slope_near_1": (
        "The policy is lifting roughly stair_height plus a fixed clearance margin."
      ),
      "pred_riser_slope_near_1": (
        "The network's decoded riser-height signal tracks the true stair height."
      ),
      "causality_note": (
        "Lift-height correlation plus predicted-riser correlation is evidence of "
        "height-aware behavior, but a causal ablation is still needed to prove the "
        "actor depends on that channel."
      ),
    },
  }


def _height_result_summary(
  stair_height_m: float,
  batches: list[_HeightBatchResult],
  step_dt: float,
) -> dict[str, Any]:
  success = [item for batch in batches for item in batch.success]
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
  episodes = max(1, len(success))
  return {
    "stair_height_m": float(stair_height_m),
    "episodes": len(success),
    "success_rate": float(sum(success) / episodes),
    "fall_rate": float(sum(fell) / episodes),
    "heading_failure_rate": float(sum(heading_failed) / episodes),
    "timeout_failure_rate": float(sum(timeout_failed) / episodes),
    "mean_episode_length_s": float(sum(lengths) * step_dt / episodes),
    "mean_max_height_progress_fraction": float(sum(height_progress) / episodes),
    "mean_max_goal_progress_fraction": float(sum(goal_progress) / episodes),
  }


def _resolve_output_path(
  *,
  cfg: StairLiftHeightSweepConfig,
  task_id: str,
  agent_cfg: Any,
  checkpoint_path: Path,
) -> Path:
  if cfg.output_file is not None:
    output_path = Path(cfg.output_file)
    if output_path.is_dir() or output_path.suffix.lower() != ".json":
      output_dir = make_timestamped_policy_output_dir(
        output_root=output_path,
        task_id=task_id,
        agent_cfg=agent_cfg,
        checkpoint_path=checkpoint_path,
      )
      return output_dir / "stair_lift_height_sweep.json"
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
  return output_dir / "stair_lift_height_sweep.json"


def _resolve_records_path(cfg: StairLiftHeightSweepConfig, output_path: Path) -> Path:
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


def _normalize_float_list_flag(args: list[str], flag_name: str) -> list[str]:
  """Allow ``--flag 0.1 0.2`` in addition to tyro's list-literal syntax."""
  normalized: list[str] = []
  index = 0
  while index < len(args):
    arg = args[index]
    normalized.append(arg)
    index += 1
    if arg != flag_name or index >= len(args):
      continue

    values: list[str] = []
    while index < len(args) and not args[index].startswith("-"):
      values.append(args[index])
      index += 1
    if not values:
      continue
    if len(values) == 1 and values[0].lstrip().startswith("["):
      normalized.append(values[0])
    else:
      normalized.append(f"[{', '.join(values)}]")
  return normalized


def run_stair_lift_height_sweep(
  task_id: str,
  cfg: StairLiftHeightSweepConfig,
) -> dict[str, Any]:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  runtime = _resolve_eval_runtime(task_id, cfg)
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
    "[INFO] stair_lift_sweep runtime: "
    f"task={runtime.task_id}, env_source={runtime.env_source}, "
    f"actor_obs_dim={runtime.checkpoint_actor_obs_dim}"
  )

  all_records: list[dict[str, Any]] = []
  height_summaries: list[dict[str, Any]] = []
  global_episode_offset = 0
  for height_index, stair_height_m in enumerate(cfg.stair_heights):
    goal_cfg = _goal_cfg_for_height(cfg, float(stair_height_m))
    remaining = cfg.episodes_per_height
    batch_index = 0
    episode_offset = 0
    batches: list[_HeightBatchResult] = []
    while remaining > 0:
      batch_size = min(max(1, cfg.num_envs), remaining)
      print(
        "[INFO] stair_lift_sweep: "
        f"h={stair_height_m:.3f} m batch={batch_index} episodes={batch_size}"
      )
      batch = _run_height_batch(
        task_id=runtime.task_id,
        env_cfg_factory=runtime.env_cfg_factory,
        runner_cls=runtime.runner_cls,
        agent_cfg=agent_cfg,
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        goal_cfg=goal_cfg,
        height_index=height_index,
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
    height_summaries.append(
      _height_result_summary(float(stair_height_m), batches, step_dt)
    )

  summary = _summarize_records(cfg, height_summaries, all_records)
  payload = {
    "task_id": runtime.task_id,
    "requested_task_id": task_id,
    "policy_output_name": policy_output_name,
    "checkpoint": str(checkpoint_path),
    "output_dir": str(output_path.parent),
    "mode": "stair_lift_height_sweep",
    "runtime": {
      "env_source": runtime.env_source,
      "checkpoint_actor_obs_dim": runtime.checkpoint_actor_obs_dim,
    },
    "config": asdict(cfg),
    "terrain": {
      "type": "goal_pyramid_height_sweep",
      "stair_heights_m": [float(value) for value in cfg.stair_heights],
      "stair_levels": cfg.stair_levels,
      "step_width": cfg.step_width,
      "platform_width": cfg.platform_width,
      "flat_apron_width": cfg.flat_apron_width,
      "terrain_border_width": cfg.terrain_border_width,
    },
    "height_results": height_summaries,
    "summary": summary,
  }

  output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
  print(f"[INFO] Wrote stair lift-height sweep to {output_path}")
  if cfg.write_records_csv:
    records_path = _resolve_records_path(cfg, output_path)
    _write_records_csv(all_records, records_path)
    print(f"[INFO] Wrote stair lift-height records to {records_path}")

  fits = summary["linear_fits"]
  first_fit = fits["first_level_peak_lift_vs_height"]
  later_fit = fits["later_levels_peak_lift_vs_height"]
  pred_fit = fits["all_pred_riser_vs_height"]
  print(
    "[INFO] stair_lift_sweep fits: "
    f"first_level_slope={first_fit['slope']}, "
    f"later_levels_slope={later_fit['slope']}, "
    f"pred_riser_slope={pred_fit['slope']}"
  )
  return payload


def main() -> None:
  import mjlab.tasks  # noqa: F401

  chosen_task, remaining_args = _task_selector_from_argv(sys.argv[1:])
  remaining_args = _normalize_float_list_flag(list(remaining_args), "--stair-heights")
  cfg = tyro.cli(
    StairLiftHeightSweepConfig,
    args=_normalize_standalone_bool_flags(
      remaining_args,
      ("--write-records-csv", "--play"),
    ),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  try:
    if cfg.play:
      run_stair_lift_height_sweep_play(chosen_task, cfg)
      return
    run_stair_lift_height_sweep(chosen_task, cfg)
  except Exception:
    traceback.print_exc()
    raise


if __name__ == "__main__":
  main()
