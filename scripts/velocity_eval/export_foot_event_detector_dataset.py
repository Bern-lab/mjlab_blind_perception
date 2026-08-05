"""Export Stage 2D foot-event detector data from a frozen policy rollout."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import tyro
from scripts.velocity_eval.export_stair_probe_dataset import (
  INPUT_FEATURE_GROUPS,
  STAIR_CURRENT_GROUND_CONTACT_KEY,
  STAIR_CURRENT_STAIR_SUPPORT_KEY,
  STAIR_CURRENT_SUPPORT_FRACTION_KEY,
  STAIR_PHASE_KEY,
  TOE_RISER_NEW_HIT_BY_FOOT_KEY,
  TOE_RISER_NEW_HIT_KEY,
  StairProbeHistoryBuffer,
  _close_sequence_logger,
  _tensor_extra,
  input_feature_slices,
  input_obs_dim,
  stage2a_base_latent_obs,
)
from scripts.velocity_eval.policy_io import (
  get_clip_actions,
  load_inference_policy,
  resolve_checkpoint_path,
  resolve_inference_agent_cfg,
)
from tqdm.auto import tqdm

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.mdp.observations import (
  _G1_LEG_JOINT_NAMES,
  _body_frame_foot_positions,
  _get_leg_joint_info,
  phase,
)
from mjlab.tasks.velocity.mdp.stair_geometry import TOE_RISER_CONTACT_BY_FOOT_KEY
from mjlab.utils.lstm import reset_policy_state_from_step
from mjlab.utils.torch import configure_torch_backends

DEFAULT_STAGE2D_CHECKPOINT = (
  "logs/rsl_rl/g1_blind_rough_target_navigation_slow_latent_teacherkl/"
  "Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1/"
  "base111/model_51000.pt"
)

FOOT_EVENT_LABEL_NAMES: tuple[str, ...] = (
  "left_contact",
  "right_contact",
  "left_touchdown",
  "right_touchdown",
  "left_toe_riser_hit",
  "right_toe_riser_hit",
)

FootEventDetectorObsSchema = Literal["v1", "footprint_v2", "footprint_deploy_v3"]

FOOTPRINT_V2_EXTRA_FEATURE_GROUPS: tuple[tuple[str, int], ...] = (
  ("left_heel_vel_body", 3),
  ("right_heel_vel_body", 3),
  ("left_heel_vel_delta", 3),
  ("right_heel_vel_delta", 3),
  ("left_sole_center_pos_body", 3),
  ("right_sole_center_pos_body", 3),
  ("left_sole_center_vel_body", 3),
  ("right_sole_center_vel_body", 3),
  ("left_sole_pitch_proxy", 1),
  ("right_sole_pitch_proxy", 1),
  ("command_lin_y", 1),
  ("command_yaw_rate", 1),
  ("action_delta_leg", 12),
)
"""Additional deployable kinematic inputs for the footprint/touchdown detector."""

FOOTPRINT_DEPLOY_V3_FEATURE_GROUPS: tuple[tuple[str, int], ...] = (
  ("projected_gravity", 3),
  ("base_ang_vel", 3),
  ("base_ang_vel_delta", 3),
  ("command_xyz", 3),
  ("left_endpoint_pos_body", 9),
  ("left_gravity_height", 5),
  ("left_endpoint_vel_body", 9),
  ("left_endpoint_vel_delta_body", 9),
  ("left_gravity_vertical_velocity", 4),
  ("left_action", 6),
  ("left_action_delta", 6),
  ("left_joint_tracking_error", 6),
  ("left_joint_vel", 6),
  ("right_endpoint_pos_body", 9),
  ("right_gravity_height", 5),
  ("right_endpoint_vel_body", 9),
  ("right_endpoint_vel_delta_body", 9),
  ("right_gravity_vertical_velocity", 4),
  ("right_action", 6),
  ("right_action_delta", 6),
  ("right_joint_tracking_error", 6),
  ("right_joint_vel", 6),
)
"""Footprint-only proprioceptive schema that does not embed the legacy 91-D obs."""

FOOTPRINT_DEPLOY_V3_FEATURE_SCALE_GROUPS: tuple[tuple[str, int, float], ...] = (
  ("projected_gravity", 3, 1.0),
  ("base_ang_vel", 3, 0.25),
  ("base_ang_vel_delta", 3, 0.25),
  ("command_xyz", 3, 1.0),
  ("left_endpoint_pos_body", 9, 5.0),
  ("left_gravity_height", 5, 5.0),
  ("left_endpoint_vel_body", 9, 0.5),
  ("left_endpoint_vel_delta_body", 9, 0.5),
  ("left_gravity_vertical_velocity", 4, 0.5),
  ("left_action", 6, 1.0),
  ("left_action_delta", 6, 2.0),
  ("left_joint_tracking_error", 6, 2.0),
  ("left_joint_vel", 6, 0.1),
  ("right_endpoint_pos_body", 9, 5.0),
  ("right_gravity_height", 5, 5.0),
  ("right_endpoint_vel_body", 9, 0.5),
  ("right_endpoint_vel_delta_body", 9, 0.5),
  ("right_gravity_vertical_velocity", 4, 0.5),
  ("right_action", 6, 1.0),
  ("right_action_delta", 6, 2.0),
  ("right_joint_tracking_error", 6, 2.0),
  ("right_joint_vel", 6, 0.1),
)
"""Fixed multipliers applied to v3 raw features before detector training/export."""

FOOT_EVENT_SUPPORT_CONTACT_SENSOR_NAME = "feet_ground_contact"
FOOT_EVENT_SUPPORT_FORCE_VERTICAL_RATIO_MIN = 0.45


@dataclass(frozen=True)
class ExportFootEventDetectorDatasetConfig:
  """Configuration for frozen-policy foot-event detector dataset export."""

  checkpoint_file: str | None = DEFAULT_STAGE2D_CHECKPOINT
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  output_dir: str = "eval_outputs/stair_stage2/model51000_seed42_foot_event_detector_v1"
  num_envs: int = 512
  steps: int = 5000
  seed: int = 42
  device: str | None = None
  history_len: int = 16
  max_samples: int | None = None
  input_schema: FootEventDetectorObsSchema = "v1"
  include_gait_phase: bool = True
  gait_period: float = 0.6
  command_name: str = "twist"
  expected_obs_dim: int | None = None
  progress: bool = True


def _per_foot_bool_from_tensor(value: torch.Tensor, *, key: str) -> torch.Tensor:
  """Return a bool ``(num_envs, 2)`` tensor from env-level or per-foot extras."""
  value = value.bool()
  if value.ndim == 1:
    return value[:, None].expand(-1, 2)
  if value.ndim == 2 and value.shape[1] == 1:
    return value.expand(-1, 2)
  if value.ndim == 2 and value.shape[1] == 2:
    return value
  raise ValueError(
    f"Expected env-level or per-foot bool extra for {key!r}, got {tuple(value.shape)}."
  )


def _per_foot_bool_extra(
  env: ManagerBasedRlEnv,
  key: str,
  *,
  default: bool = False,
) -> torch.Tensor:
  """Return a bool ``(num_envs, 2)`` extra, expanding env-level flags if needed."""
  value = _tensor_extra(env, key, (), torch.bool, default)
  return _per_foot_bool_from_tensor(value, key=key)


def _toe_riser_hit_by_foot(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return per-foot toe-riser hit labels, preferring the new per-foot extra."""
  value = env.extras.get(TOE_RISER_NEW_HIT_BY_FOOT_KEY)
  if isinstance(value, torch.Tensor):
    return _per_foot_bool_from_tensor(
      value.to(device=env.device),
      key=TOE_RISER_NEW_HIT_BY_FOOT_KEY,
    )
  return _per_foot_bool_extra(env, TOE_RISER_NEW_HIT_KEY, default=False)


def _toe_riser_contact_by_foot(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return persistent per-foot toe-riser contacts when the stair term exposes them."""
  value = env.extras.get(TOE_RISER_CONTACT_BY_FOOT_KEY)
  if isinstance(value, torch.Tensor):
    return _per_foot_bool_from_tensor(
      value.to(device=env.device),
      key=TOE_RISER_CONTACT_BY_FOOT_KEY,
    )
  return torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)


def _support_force_mask_from_ground_sensor(
  env: ManagerBasedRlEnv,
  raw_contact: torch.Tensor,
) -> torch.Tensor | None:
  scene = getattr(env, "scene", None)
  if scene is None:
    return None
  if isinstance(scene, dict):
    sensor = scene.get(FOOT_EVENT_SUPPORT_CONTACT_SENSOR_NAME)
  else:
    try:
      sensor = scene[FOOT_EVENT_SUPPORT_CONTACT_SENSOR_NAME]
    except (KeyError, TypeError, AttributeError):
      return None
  data = getattr(sensor, "data", None)
  force = getattr(data, "force", None)
  if not isinstance(force, torch.Tensor):
    return None
  force = force.to(device=env.device, dtype=torch.float32)
  if force.ndim != 3 or force.shape[0] != env.num_envs or force.shape[-1] != 3:
    return None
  if force.shape[1] == 2:
    force_by_foot = force
  elif force.shape[1] % 2 == 0:
    force_by_foot = force.view(env.num_envs, 2, -1, 3).sum(dim=2)
  else:
    return None
  force_norm = force_by_foot.norm(dim=-1)
  force_z = force_by_foot[..., 2].abs()
  vertical_ratio = force_z / force_norm.clamp_min(1.0e-6)
  vertical_support = (force_norm <= 1.0e-6) | (
    vertical_ratio >= FOOT_EVENT_SUPPORT_FORCE_VERTICAL_RATIO_MIN
  )
  return raw_contact & vertical_support


def _support_contact_from_env(
  env: ManagerBasedRlEnv,
  *,
  raw_contact: torch.Tensor,
) -> torch.Tensor:
  """Return simulation-only horizontal support contacts for training labels."""
  toe_riser_contact = _toe_riser_contact_by_foot(env)
  support_contact = raw_contact & ~toe_riser_contact
  force_mask = _support_force_mask_from_ground_sensor(env, raw_contact)
  if force_mask is not None:
    support_contact &= force_mask
  return support_contact


def foot_event_labels_from_env(
  env: ManagerBasedRlEnv,
  *,
  previous_contact: torch.Tensor,
  previous_contact_valid: torch.Tensor,
) -> torch.Tensor:
  """Build detector labels from simulation-only support/toe contact evidence."""
  raw_contact = _tensor_extra(
    env,
    STAIR_CURRENT_GROUND_CONTACT_KEY,
    (2,),
    torch.bool,
    False,
  ).bool()
  contact = _support_contact_from_env(env, raw_contact=raw_contact)
  touchdown = contact & ~previous_contact & previous_contact_valid[:, None]
  toe_hit = _toe_riser_hit_by_foot(env)
  return torch.cat(
    (
      contact.float(),
      touchdown.float(),
      toe_hit.float(),
    ),
    dim=-1,
  )


def _validate_input_schema(input_schema: str) -> FootEventDetectorObsSchema:
  valid_schemas = ("v1", "footprint_v2", "footprint_deploy_v3")
  if input_schema not in valid_schemas:
    raise ValueError(
      f"input_schema must be one of {valid_schemas}, got {input_schema!r}."
    )
  return cast(FootEventDetectorObsSchema, input_schema)


def foot_event_detector_obs_dim(
  *,
  include_gait_phase: bool,
  input_schema: FootEventDetectorObsSchema = "v1",
) -> int:
  """Return deployable event-detector observation width."""
  schema = _validate_input_schema(input_schema)
  if schema == "footprint_deploy_v3":
    dim = sum(width for _name, width in FOOTPRINT_DEPLOY_V3_FEATURE_GROUPS)
    return dim + (2 if include_gait_phase else 0)
  dim = input_obs_dim() + (2 if include_gait_phase else 0)
  if schema == "footprint_v2":
    dim += sum(width for _name, width in FOOTPRINT_V2_EXTRA_FEATURE_GROUPS)
  return dim


def resolve_foot_event_detector_obs_dim(
  expected_obs_dim: int | None,
  *,
  include_gait_phase: bool,
  input_schema: FootEventDetectorObsSchema,
) -> int:
  """Return the configured detector obs width, validating overrides."""
  obs_dim = foot_event_detector_obs_dim(
    include_gait_phase=include_gait_phase,
    input_schema=input_schema,
  )
  if expected_obs_dim is not None and int(expected_obs_dim) != obs_dim:
    raise ValueError(
      f"expected_obs_dim={expected_obs_dim} does not match detector obs "
      f"dimension {obs_dim} for input_schema={input_schema!r}."
    )
  return obs_dim


def _command_column(
  env: ManagerBasedRlEnv,
  *,
  command_name: str,
  column: int,
  dtype: torch.dtype,
) -> torch.Tensor:
  zeros = torch.zeros(env.num_envs, 1, dtype=dtype, device=env.device)
  command_manager = getattr(env, "command_manager", None)
  if command_manager is None:
    return zeros
  command = command_manager.get_command(command_name)
  if not isinstance(command, torch.Tensor) or command.shape[-1] <= column:
    return zeros
  return command[:, column : column + 1].to(device=env.device, dtype=dtype)


def _velocity_and_delta_from_cache(
  env: ManagerBasedRlEnv,
  *,
  position: torch.Tensor,
  position_key: str,
  velocity_key: str,
  reset_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
  zeros = torch.zeros_like(position)
  reset = (
    torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if reset_mask is None
    else reset_mask.to(device=env.device, dtype=torch.bool)
  ).reshape(env.num_envs)
  if position_key not in env.extras:
    env.extras[position_key] = position.detach().clone()
    env.extras[velocity_key] = zeros.detach().clone()
    return zeros, zeros

  prev_position = env.extras[position_key].to(device=env.device, dtype=position.dtype)
  velocity = (position - prev_position) / max(float(env.step_dt), 1.0e-6)
  velocity = torch.where(reset[:, None], zeros, velocity)

  prev_velocity_obj = env.extras.get(velocity_key)
  if isinstance(prev_velocity_obj, torch.Tensor):
    prev_velocity = prev_velocity_obj.to(device=env.device, dtype=position.dtype)
    velocity_delta = torch.where(reset[:, None], zeros, velocity - prev_velocity)
  else:
    velocity_delta = zeros

  env.extras[position_key] = position.detach().clone()
  env.extras[velocity_key] = velocity.detach().clone()
  return velocity, velocity_delta


def _delta_from_cache(
  env: ManagerBasedRlEnv,
  *,
  value: torch.Tensor,
  key: str,
  reset_mask: torch.Tensor | None,
) -> torch.Tensor:
  zeros = torch.zeros_like(value)
  reset = (
    torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if reset_mask is None
    else reset_mask.to(device=env.device, dtype=torch.bool)
  ).reshape(env.num_envs)
  previous = env.extras.get(key)
  if not isinstance(previous, torch.Tensor):
    env.extras[key] = value.detach().clone()
    return zeros
  delta = value - previous.to(device=env.device, dtype=value.dtype)
  delta = torch.where(reset[:, None], zeros, delta)
  env.extras[key] = value.detach().clone()
  return delta


def foot_event_input_feature_scale_groups(
  *,
  include_gait_phase: bool,
  input_schema: FootEventDetectorObsSchema = "v1",
) -> list[dict[str, float | int | str]]:
  """Return fixed feature multipliers applied before detector inference."""
  schema = _validate_input_schema(input_schema)
  if schema != "footprint_deploy_v3":
    return [
      {"name": str(group["name"]), "width": int(group["width"]), "scale": 1.0}
      for group in foot_event_input_feature_groups(
        include_gait_phase=include_gait_phase,
        input_schema=schema,
      )
    ]
  groups = [
    {"name": name, "width": width, "scale": scale}
    for name, width, scale in FOOTPRINT_DEPLOY_V3_FEATURE_SCALE_GROUPS
  ]
  if include_gait_phase:
    groups.insert(4, {"name": "gait_phase_sin", "width": 1, "scale": 1.0})
    groups.insert(5, {"name": "gait_phase_cos", "width": 1, "scale": 1.0})
  return groups


def foot_event_input_feature_scales(
  *,
  include_gait_phase: bool,
  input_schema: FootEventDetectorObsSchema = "v1",
) -> list[float]:
  scales: list[float] = []
  for group in foot_event_input_feature_scale_groups(
    include_gait_phase=include_gait_phase,
    input_schema=input_schema,
  ):
    scales.extend([float(group["scale"])] * int(group["width"]))
  return scales


def _footprint_v2_extra_obs(
  latent: torch.Tensor,
  env: ManagerBasedRlEnv,
  *,
  command_name: str,
  reset_mask: torch.Tensor | None,
) -> torch.Tensor:
  left_toe, right_toe, left_heel, right_heel = _body_frame_foot_positions(env)
  left_center = 0.5 * (left_toe + left_heel)
  right_center = 0.5 * (right_toe + right_heel)
  left_heel_vel, left_heel_vel_delta = _velocity_and_delta_from_cache(
    env,
    position=left_heel,
    position_key="foot_event_v2_prev_left_heel_pos_body",
    velocity_key="foot_event_v2_prev_left_heel_vel_body",
    reset_mask=reset_mask,
  )
  right_heel_vel, right_heel_vel_delta = _velocity_and_delta_from_cache(
    env,
    position=right_heel,
    position_key="foot_event_v2_prev_right_heel_pos_body",
    velocity_key="foot_event_v2_prev_right_heel_vel_body",
    reset_mask=reset_mask,
  )
  left_center_vel, _left_center_vel_delta = _velocity_and_delta_from_cache(
    env,
    position=left_center,
    position_key="foot_event_v2_prev_left_sole_center_pos_body",
    velocity_key="foot_event_v2_prev_left_sole_center_vel_body",
    reset_mask=reset_mask,
  )
  right_center_vel, _right_center_vel_delta = _velocity_and_delta_from_cache(
    env,
    position=right_center,
    position_key="foot_event_v2_prev_right_sole_center_pos_body",
    velocity_key="foot_event_v2_prev_right_sole_center_vel_body",
    reset_mask=reset_mask,
  )

  left_sole = left_toe - left_heel
  right_sole = right_toe - right_heel
  left_sole_pitch = left_sole[:, 2:3] / left_sole.norm(dim=-1, keepdim=True).clamp_min(
    1.0e-6
  )
  right_sole_pitch = right_sole[:, 2:3] / right_sole.norm(
    dim=-1, keepdim=True
  ).clamp_min(1.0e-6)

  command_lin_y = _command_column(
    env,
    command_name=command_name,
    column=1,
    dtype=latent.dtype,
  )
  command_yaw_rate = _command_column(
    env,
    command_name=command_name,
    column=2,
    dtype=latent.dtype,
  )

  slices = input_feature_slices()
  previous_action_leg = latent[:, slices["previous_action_leg"]]
  current_action = env.action_manager.action
  _leg_joint_indices, leg_action_indices, _default_joint_pos = _get_leg_joint_info(env)
  current_action_leg = current_action[:, leg_action_indices].to(dtype=latent.dtype)
  action_delta_leg = current_action_leg - previous_action_leg

  return torch.cat(
    (
      left_heel_vel,
      right_heel_vel,
      left_heel_vel_delta,
      right_heel_vel_delta,
      left_center,
      right_center,
      left_center_vel,
      right_center_vel,
      left_sole_pitch,
      right_sole_pitch,
      command_lin_y,
      command_yaw_rate,
      action_delta_leg,
    ),
    dim=-1,
  ).to(dtype=latent.dtype)


def _footprint_deploy_v3_obs(
  env: ManagerBasedRlEnv,
  *,
  include_gait_phase: bool,
  gait_period: float,
  command_name: str,
  reset_mask: torch.Tensor | None,
) -> torch.Tensor:
  robot = env.scene["robot"]
  dtype = torch.float32
  device = env.device
  projected_gravity = robot.data.projected_gravity_b.to(device=device, dtype=dtype)
  up_axis_body = -projected_gravity / projected_gravity.norm(
    dim=-1,
    keepdim=True,
  ).clamp_min(1.0e-6)
  left_toe_b, right_toe_b, left_heel_b, right_heel_b = _body_frame_foot_positions(env)
  left_toe_b = left_toe_b.to(device=device, dtype=dtype)
  right_toe_b = right_toe_b.to(device=device, dtype=dtype)
  left_heel_b = left_heel_b.to(device=device, dtype=dtype)
  right_heel_b = right_heel_b.to(device=device, dtype=dtype)
  left_center_b = 0.5 * (left_toe_b + left_heel_b)
  right_center_b = 0.5 * (right_toe_b + right_heel_b)

  base_ang_vel = robot.data.root_link_ang_vel_b.to(device=device, dtype=dtype)
  base_ang_vel_delta = _delta_from_cache(
    env,
    value=base_ang_vel,
    key="footprint_deploy_v3_prev_base_ang_vel",
    reset_mask=reset_mask,
  )

  command = torch.cat(
    (
      _command_column(env, command_name=command_name, column=0, dtype=dtype),
      _command_column(env, command_name=command_name, column=1, dtype=dtype),
      _command_column(env, command_name=command_name, column=2, dtype=dtype),
    ),
    dim=-1,
  )

  def point_motion(name: str, point: torch.Tensor) -> tuple[torch.Tensor, ...]:
    velocity, velocity_delta = _velocity_and_delta_from_cache(
      env,
      position=point,
      position_key=f"footprint_deploy_v3_prev_{name}_pos_body",
      velocity_key=f"footprint_deploy_v3_prev_{name}_vel_body",
      reset_mask=reset_mask,
    )
    return point, velocity.to(dtype=dtype), velocity_delta.to(dtype=dtype)

  left_toe, left_toe_vel, left_toe_vel_delta = point_motion("left_toe", left_toe_b)
  left_heel, left_heel_vel, left_heel_vel_delta = point_motion("left_heel", left_heel_b)
  left_center, left_center_vel, left_center_vel_delta = point_motion(
    "left_center", left_center_b
  )
  right_toe, right_toe_vel, right_toe_vel_delta = point_motion("right_toe", right_toe_b)
  right_heel, right_heel_vel, right_heel_vel_delta = point_motion(
    "right_heel", right_heel_b
  )
  right_center, right_center_vel, right_center_vel_delta = point_motion(
    "right_center", right_center_b
  )

  leg_joint_indices, leg_action_indices, default_joint_pos = _get_leg_joint_info(env)
  current_action = env.action_manager.action[:, leg_action_indices].to(dtype=dtype)
  action_delta = _delta_from_cache(
    env,
    value=current_action,
    key="footprint_deploy_v3_prev_action_leg",
    reset_mask=reset_mask,
  )
  from mjlab.asset_zoo.robots import G1_ACTION_SCALE

  action_scale = torch.tensor(
    [G1_ACTION_SCALE.get(name, 0.25) for name in _G1_LEG_JOINT_NAMES],
    dtype=dtype,
    device=device,
  )
  joint_pos = robot.data.joint_pos[:, leg_joint_indices].to(dtype=dtype)
  joint_vel = robot.data.joint_vel[:, leg_joint_indices].to(dtype=dtype)
  tracking_error = default_joint_pos.to(dtype=dtype) + action_scale * current_action
  tracking_error = tracking_error - joint_pos

  def vertical(point: torch.Tensor) -> torch.Tensor:
    return (point.to(dtype=dtype) * up_axis_body).sum(dim=-1, keepdim=True)

  def vertical_velocity(velocity: torch.Tensor) -> torch.Tensor:
    return (velocity.to(dtype=dtype) * up_axis_body).sum(dim=-1, keepdim=True)

  def per_foot_features(
    *,
    toe: torch.Tensor,
    heel: torch.Tensor,
    center: torch.Tensor,
    toe_vel: torch.Tensor,
    heel_vel: torch.Tensor,
    center_vel: torch.Tensor,
    toe_vel_delta: torch.Tensor,
    heel_vel_delta: torch.Tensor,
    center_vel_delta: torch.Tensor,
    action_slice: slice,
  ) -> torch.Tensor:
    toe_z = vertical(toe)
    heel_z = vertical(heel)
    center_z = vertical(center)
    min_z = torch.minimum(toe_z, heel_z)
    toe_heel_z_gap = toe_z - heel_z
    toe_vz = vertical_velocity(toe_vel)
    heel_vz = vertical_velocity(heel_vel)
    center_vz = vertical_velocity(center_vel)
    min_vz = torch.minimum(toe_vz, heel_vz)
    return torch.cat(
      (
        toe.to(dtype=dtype),
        heel.to(dtype=dtype),
        center.to(dtype=dtype),
        toe_z,
        heel_z,
        center_z,
        min_z,
        toe_heel_z_gap,
        toe_vel,
        heel_vel,
        center_vel,
        toe_vel_delta,
        heel_vel_delta,
        center_vel_delta,
        toe_vz,
        heel_vz,
        center_vz,
        min_vz,
        current_action[:, action_slice],
        action_delta[:, action_slice],
        tracking_error[:, action_slice],
        joint_vel[:, action_slice],
      ),
      dim=-1,
    )

  parts = [
    projected_gravity,
    base_ang_vel,
    base_ang_vel_delta,
    command,
  ]
  if include_gait_phase:
    parts.append(phase(env, gait_period, command_name).to(dtype=dtype))
  parts.extend(
    [
      per_foot_features(
        toe=left_toe,
        heel=left_heel,
        center=left_center,
        toe_vel=left_toe_vel,
        heel_vel=left_heel_vel,
        center_vel=left_center_vel,
        toe_vel_delta=left_toe_vel_delta,
        heel_vel_delta=left_heel_vel_delta,
        center_vel_delta=left_center_vel_delta,
        action_slice=slice(0, 6),
      ),
      per_foot_features(
        toe=right_toe,
        heel=right_heel,
        center=right_center,
        toe_vel=right_toe_vel,
        heel_vel=right_heel_vel,
        center_vel=right_center_vel,
        toe_vel_delta=right_toe_vel_delta,
        heel_vel_delta=right_heel_vel_delta,
        center_vel_delta=right_center_vel_delta,
        action_slice=slice(6, 12),
      ),
    ]
  )
  raw_obs = torch.cat(parts, dim=-1)
  scales = torch.tensor(
    foot_event_input_feature_scales(
      include_gait_phase=include_gait_phase,
      input_schema="footprint_deploy_v3",
    ),
    dtype=dtype,
    device=device,
  )
  if raw_obs.shape[-1] != scales.numel():
    raise RuntimeError(
      f"footprint_deploy_v3 raw obs dim {raw_obs.shape[-1]} does not match "
      f"scale dim {scales.numel()}."
    )
  return raw_obs * scales


def foot_event_detector_obs(
  obs: Any,
  env: ManagerBasedRlEnv,
  *,
  input_schema: FootEventDetectorObsSchema = "v1",
  include_gait_phase: bool,
  gait_period: float,
  command_name: str,
  reset_mask: torch.Tensor | None = None,
) -> torch.Tensor:
  """Return deployable detector input features for the current frame."""
  schema = _validate_input_schema(input_schema)
  if schema == "footprint_deploy_v3":
    return _footprint_deploy_v3_obs(
      env,
      include_gait_phase=include_gait_phase,
      gait_period=gait_period,
      command_name=command_name,
      reset_mask=reset_mask,
    )
  latent = stage2a_base_latent_obs(obs)
  parts = [latent]
  if include_gait_phase:
    parts.append(phase(env, gait_period, command_name))
  if schema == "footprint_v2":
    parts.append(
      _footprint_v2_extra_obs(
        latent,
        env,
        command_name=command_name,
        reset_mask=reset_mask,
      )
    )
  return torch.cat(parts, dim=-1)


def foot_event_input_feature_groups(
  *,
  include_gait_phase: bool,
  input_schema: FootEventDetectorObsSchema = "v1",
) -> list[dict[str, int | str]]:
  """Return detector input schema metadata."""
  schema = _validate_input_schema(input_schema)
  if schema == "footprint_deploy_v3":
    groups = [
      {"name": name, "width": width}
      for name, width in FOOTPRINT_DEPLOY_V3_FEATURE_GROUPS
    ]
    if include_gait_phase:
      groups.insert(4, {"name": "gait_phase_sin", "width": 1})
      groups.insert(5, {"name": "gait_phase_cos", "width": 1})
    return groups
  groups = [{"name": name, "width": width} for name, width in INPUT_FEATURE_GROUPS]
  if include_gait_phase:
    groups.append({"name": "gait_phase_sin", "width": 1})
    groups.append({"name": "gait_phase_cos", "width": 1})
  if schema == "footprint_v2":
    groups.extend(
      {"name": name, "width": width}
      for name, width in FOOTPRINT_V2_EXTRA_FEATURE_GROUPS
    )
  return groups


class FootEventDetectorDatasetBuilder:
  """Collect fixed-length deployable observation histories and event labels."""

  def __init__(
    self,
    *,
    num_envs: int,
    history_len: int,
    obs_dim: int,
    max_samples: int | None,
    device: torch.device | str,
  ) -> None:
    if history_len <= 0:
      raise ValueError("history_len must be positive.")
    if max_samples is not None and max_samples <= 0:
      raise ValueError("max_samples must be positive.")
    self.num_envs = int(num_envs)
    self.history = StairProbeHistoryBuffer(
      num_envs=num_envs,
      history_len=history_len,
      obs_dim=obs_dim,
      device=device,
    )
    self.max_samples = None if max_samples is None else int(max_samples)
    self._arrays: dict[str, list[np.ndarray]] = {
      "obs_history": [],
      "obs_valid_mask": [],
      "event_label": [],
      "episode_id": [],
      "env_id": [],
      "frame_idx": [],
      "seed": [],
    }

  @property
  def num_samples(self) -> int:
    if not self._arrays["event_label"]:
      return 0
    return int(sum(chunk.shape[0] for chunk in self._arrays["event_label"]))

  @property
  def is_full(self) -> bool:
    return self.max_samples is not None and self.num_samples >= self.max_samples

  def push_observations(self, obs: torch.Tensor, reset_mask: torch.Tensor) -> None:
    self.history.push(obs, reset_mask)

  def collect(
    self,
    *,
    labels: torch.Tensor,
    collect_mask: torch.Tensor,
    episode_id: torch.Tensor,
    frame_idx: int,
    seed: int,
    metadata: dict[str, torch.Tensor] | None = None,
  ) -> None:
    ids = collect_mask.nonzero(as_tuple=False).squeeze(-1)
    if ids.numel() == 0 or self.is_full:
      return
    if self.max_samples is not None:
      remaining = self.max_samples - self.num_samples
      if ids.numel() > remaining:
        ids = ids[:remaining]
    self._arrays["obs_history"].append(
      self.history.history[ids].detach().cpu().numpy().astype(np.float32)
    )
    self._arrays["obs_valid_mask"].append(
      self.history.valid_mask[ids].detach().cpu().numpy().astype(np.bool_)
    )
    self._arrays["event_label"].append(
      labels[ids].detach().cpu().numpy().astype(np.float32)
    )
    self._arrays["episode_id"].append(
      episode_id[ids].detach().cpu().numpy().astype(np.int64)
    )
    self._arrays["env_id"].append(ids.detach().cpu().numpy().astype(np.int64))
    self._arrays["frame_idx"].append(np.full((ids.numel(),), frame_idx, dtype=np.int64))
    self._arrays["seed"].append(np.full((ids.numel(),), seed, dtype=np.int64))
    if metadata is not None:
      for key, value in metadata.items():
        if key not in self._arrays:
          self._arrays[key] = []
        self._arrays[key].append(value[ids].detach().cpu().numpy())

  def as_arrays(self) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key, chunks in self._arrays.items():
      if chunks:
        arrays[key] = np.concatenate(chunks, axis=0)
      elif key == "obs_history":
        arrays[key] = np.zeros(
          (0, self.history.history.shape[1], self.history.history.shape[2]),
          dtype=np.float32,
        )
      elif key == "obs_valid_mask":
        arrays[key] = np.zeros((0, self.history.history.shape[1]), dtype=np.bool_)
      elif key == "event_label":
        arrays[key] = np.zeros((0, len(FOOT_EVENT_LABEL_NAMES)), dtype=np.float32)
      else:
        arrays[key] = np.zeros((0,), dtype=np.int64)
    return arrays


def build_label_audit(arrays: dict[str, np.ndarray]) -> list[tuple[str, str]]:
  """Summarize detector labels and exported histories."""
  labels = arrays["event_label"]
  rows: list[tuple[str, str]] = [
    ("num_samples", str(int(labels.shape[0]))),
    ("num_episodes", str(int(np.unique(arrays["episode_id"]).shape[0]))),
    ("obs_history_len", str(int(arrays["obs_history"].shape[1]))),
    ("obs_dim", str(int(arrays["obs_history"].shape[2]))),
    ("label_dim", str(int(labels.shape[1]))),
  ]
  for index, name in enumerate(FOOT_EVENT_LABEL_NAMES):
    count = float(labels[:, index].sum()) if labels.size else 0.0
    rate = count / max(float(labels.shape[0]), 1.0)
    rows.append((f"{name}_positive_count", f"{count:.0f}"))
    rows.append((f"{name}_positive_rate", f"{rate:.6g}"))
  stair_support = arrays.get("stair_support")
  if stair_support is not None and labels.size:
    touchdown = labels[:, 2:4] > 0.5
    stair_touchdown = touchdown & stair_support.astype(np.bool_)
    rows.append(
      (
        "touchdown_on_stair_count",
        f"{float(stair_touchdown.sum()):.0f}",
      )
    )
    rows.append(
      (
        "touchdown_on_stair_rate_of_touchdowns",
        f"{float(stair_touchdown.sum()) / max(float(touchdown.sum()), 1.0):.6g}",
      )
    )
  return rows


def write_dataset_outputs(
  output_dir: Path,
  *,
  task_id: str,
  checkpoint_path: Path,
  cfg: ExportFootEventDetectorDatasetConfig,
  arrays: dict[str, np.ndarray],
) -> None:
  """Write dataset arrays, audit, and schema metadata."""
  np.savez_compressed(output_dir / "samples.npz", **cast(Any, arrays))
  with (output_dir / "label_audit.csv").open(
    "w", encoding="utf-8", newline=""
  ) as stream:
    writer = csv.writer(stream)
    writer.writerow(("metric", "value"))
    writer.writerows(build_label_audit(arrays))

  payload: dict[str, Any] = {
    "task_id": task_id,
    "checkpoint_path": str(checkpoint_path),
    "policy_frozen": True,
    "config": asdict(cfg),
    "input_source": (
      "footprint_deploy_v3 proprioceptive FK/IMU/action features"
      if cfg.input_schema == "footprint_deploy_v3"
      else (
        "observations['latent'] / stair_latent_obs"
        + (" + gait_phase" if cfg.include_gait_phase else "")
        + (
          " + footprint_v2 deployable kinematics"
          if cfg.input_schema == "footprint_v2"
          else ""
        )
      )
    ),
    "input_feature_groups": foot_event_input_feature_groups(
      include_gait_phase=cfg.include_gait_phase,
      input_schema=cfg.input_schema,
    ),
    "input_feature_scale_groups": foot_event_input_feature_scale_groups(
      include_gait_phase=cfg.include_gait_phase,
      input_schema=cfg.input_schema,
    ),
    "input_feature_scales": foot_event_input_feature_scales(
      include_gait_phase=cfg.include_gait_phase,
      input_schema=cfg.input_schema,
    ),
    "label_names": list(FOOT_EVENT_LABEL_NAMES),
    "label_source": {
      "contact": (
        f"horizontal support contact from {STAIR_CURRENT_GROUND_CONTACT_KEY}, "
        f"{FOOT_EVENT_SUPPORT_CONTACT_SENSOR_NAME}.force, excluding "
        f"{TOE_RISER_CONTACT_BY_FOOT_KEY}"
      ),
      "touchdown": "rising edge of horizontal support contact",
      "toe_riser_hit": (
        f"{TOE_RISER_NEW_HIT_BY_FOOT_KEY}, falling back to {TOE_RISER_NEW_HIT_KEY}"
      ),
    },
    "deploy_contract": (
      "Detector predicts event/contact probabilities only; deployment code "
      "maintains foot-event memory and computes FK-based footprint geometry."
    ),
  }
  with (output_dir / "dataset_config.json").open("w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")


def run_export(task_id: str, cfg: ExportFootEventDetectorDatasetConfig) -> Path:
  """Roll out a frozen policy and export supervised detector samples."""
  if cfg.num_envs <= 0:
    raise ValueError("num_envs must be positive.")
  if cfg.steps <= 0:
    raise ValueError("steps must be positive.")
  if cfg.history_len <= 0:
    raise ValueError("history_len must be positive.")
  obs_dim = resolve_foot_event_detector_obs_dim(
    cfg.expected_obs_dim,
    include_gait_phase=cfg.include_gait_phase,
    input_schema=cfg.input_schema,
  )

  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  output_dir = Path(cfg.output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  env_cfg = load_env_cfg(task_id, play=False)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  checkpoint_path = resolve_checkpoint_path(
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
  print(
    "[Stage 2D] Export detector dataset:",
    f"task={task_id}",
    f"checkpoint={checkpoint_path}",
    f"envs={cfg.num_envs}",
    f"steps={cfg.steps}",
    f"history_len={cfg.history_len}",
    f"input_schema={cfg.input_schema}",
    f"include_gait_phase={cfg.include_gait_phase}",
    f"max_samples={cfg.max_samples}",
    f"device={device}",
    f"output={output_dir}",
  )

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  wrapped = RslRlVecEnvWrapper(raw_env, clip_actions=get_clip_actions(agent_cfg))
  builder = FootEventDetectorDatasetBuilder(
    num_envs=cfg.num_envs,
    history_len=cfg.history_len,
    obs_dim=obs_dim,
    max_samples=cfg.max_samples,
    device=device,
  )
  previous_contact = torch.zeros(cfg.num_envs, 2, dtype=torch.bool, device=device)
  previous_contact_valid = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
  episode_counter = torch.zeros(cfg.num_envs, dtype=torch.int64, device=device)
  env_ids = torch.arange(cfg.num_envs, dtype=torch.int64, device=device)
  completed_steps = 0

  try:
    policy, _runner = load_inference_policy(
      env=wrapped,
      task_id=task_id,
      agent_cfg=agent_cfg,
      checkpoint_path=checkpoint_path,
      device=device,
    )
    obs = wrapped.get_observations()
    reset_all = torch.ones(cfg.num_envs, dtype=torch.bool, device=device)
    latent = foot_event_detector_obs(
      obs,
      raw_env,
      input_schema=cfg.input_schema,
      include_gait_phase=cfg.include_gait_phase,
      gait_period=cfg.gait_period,
      command_name=cfg.command_name,
      reset_mask=reset_all,
    )
    builder.push_observations(latent, reset_all)

    progress = tqdm(
      range(cfg.steps),
      desc=f"foot event export seed={cfg.seed}",
      disable=not cfg.progress,
      dynamic_ncols=True,
      unit="step",
    )
    for rollout_step in progress:
      completed_steps = rollout_step + 1
      with torch.no_grad():
        actions = policy(obs)
      step_result = wrapped.step(actions)
      reset_policy_state_from_step(policy, step_result)
      obs, _rewards, dones, _extras = step_result
      reset_mask = dones.to(dtype=torch.bool)
      latent = foot_event_detector_obs(
        obs,
        raw_env,
        input_schema=cfg.input_schema,
        include_gait_phase=cfg.include_gait_phase,
        gait_period=cfg.gait_period,
        command_name=cfg.command_name,
        reset_mask=reset_mask,
      )
      builder.push_observations(latent, reset_mask)

      labels = foot_event_labels_from_env(
        raw_env,
        previous_contact=previous_contact,
        previous_contact_valid=previous_contact_valid,
      )
      full_history = builder.history.valid_mask.all(dim=1)
      collect_mask = full_history & ~reset_mask
      episode_id = episode_counter * cfg.num_envs + env_ids
      builder.collect(
        labels=labels,
        collect_mask=collect_mask,
        episode_id=episode_id,
        frame_idx=rollout_step + 1,
        seed=cfg.seed,
        metadata={
          "stair_support": _tensor_extra(
            raw_env,
            STAIR_CURRENT_STAIR_SUPPORT_KEY,
            (2,),
            torch.bool,
            False,
          ).bool(),
          "support_fraction": _tensor_extra(
            raw_env,
            STAIR_CURRENT_SUPPORT_FRACTION_KEY,
            (2,),
            torch.float32,
            0.0,
          ).float(),
          "stair_phase": _tensor_extra(
            raw_env,
            STAIR_PHASE_KEY,
            (),
            torch.long,
            0,
          ).long(),
        },
      )

      current_contact = labels[:, 0:2].bool()
      previous_contact.copy_(
        torch.where(
          reset_mask[:, None],
          torch.zeros_like(current_contact),
          current_contact,
        )
      )
      previous_contact_valid.copy_(
        torch.where(
          reset_mask,
          torch.zeros_like(previous_contact_valid),
          torch.ones_like(previous_contact_valid),
        )
      )
      episode_counter += reset_mask.to(dtype=torch.int64)
      if builder.is_full:
        break
  finally:
    _close_sequence_logger(raw_env)
    wrapped.close()

  arrays = builder.as_arrays()
  stopped_reason = "max_samples" if builder.is_full else "completed_steps"
  write_dataset_outputs(
    output_dir,
    task_id=task_id,
    checkpoint_path=checkpoint_path,
    cfg=cfg,
    arrays=arrays,
  )
  print(
    "[Stage 2D] Export complete:",
    f"samples={arrays['event_label'].shape[0]}",
    f"completed_steps={completed_steps}",
    f"stopped_reason={stopped_reason}",
    f"obs_history_shape={arrays['obs_history'].shape}",
    f"output={output_dir}",
  )
  return output_dir


def main() -> None:
  import mjlab.tasks as _tasks  # noqa: F401

  task_id, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    args=None,
    default="Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1",
    return_unknown_args=True,
  )
  cfg = tyro.cli(ExportFootEventDetectorDatasetConfig, args=remaining_args)
  run_export(task_id, cfg)


if __name__ == "__main__":
  main()
