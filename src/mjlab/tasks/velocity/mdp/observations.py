from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor, ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from .stair_geometry import (
  COLLISION_RISK_KEY,
  LANDING_QUALITY_KEY,
  LANDING_TOUCHDOWN_KEY,
  MINIMUM_SAFE_STRIDE_EXACT_KEY,
  MINIMUM_SAFE_STRIDE_KEY,
  MINIMUM_SAFE_STRIDE_VALID_KEY,
  MINIMUM_SAFE_STRIDE_WEIGHT_KEY,
  STAIR_ENTRY_EVENT_KEY,
  STAIR_ENTRY_RECENT_EVIDENCE_KEY,
  STAIR_PHASE_KEY,
  STAIR_RISER_HEIGHT_LABEL_KEY,
  STAIR_SHAPE_LABEL_VALID_KEY,
  STAIR_TREAD_DEPTH_LABEL_KEY,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
  """Sin/cos gait phase used by the Unitree deployment runtime."""
  global_phase = (env.episode_length_buf * env.step_dt) % period / period
  phase = torch.zeros(env.num_envs, 2, device=env.device)
  phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
  phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)

  stand_mask = (
    torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
  )  # (command[:, :3], dim=1)
  return torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)


def foot_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-foot vertical clearance above terrain.

  Returns:
    Tensor of shape [B, F] where F is the number of frames (feet).
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, TerrainHeightSensor), (
    f"foot_height requires a TerrainHeightSensor, got {type(sensor).__name__}"
  )
  return sensor.data.heights


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def camera_depth(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  cutoff_distance: float,
  min_depth: float = 0.01,
) -> torch.Tensor:
  """Depth observation in CNN-compatible format (B, 1, H, W)."""
  sensor: CameraSensor = env.scene[sensor_name]
  depth_data = sensor.data.depth  # (B, H, W, 1)
  assert depth_data is not None, f"Camera '{sensor_name}' has no depth data"
  depth_data = depth_data.permute(0, 3, 1, 2)
  depth_data_clipped = torch.clamp(depth_data, min=min_depth, max=cutoff_distance)
  return torch.clamp(depth_data_clipped / cutoff_distance, 0.0, 1.0)


# ======================================================================
# Stair-focused deployable latent_obs helpers
# ======================================================================

_G1_LEG_JOINT_NAMES: list[str] = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
]
"""G1 leg joint names in canonical order (12 DOF total)."""

_G1_FOOT_LINK_NAMES: tuple[str, str] = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
)
"""
G1 ankle-roll link names used as a fallback for FK toe/heel point estimation.
The preferred path reads named toe/heel sites from the MJCF.  For deployment,
these positions should be computed by FK from joint encoders.
"""

_G1_TOE_HEEL_SITE_NAMES: tuple[str, str, str, str] = (
  "left_toe",
  "right_toe",
  "left_heel",
  "right_heel",
)
"""G1 MJCF sites used for toe/heel kinematic latent features."""

# Fallback body-frame toe/heel offsets relative to ankle_roll_link origin.
# These are the lateral centers of the official Unitree G1 rev1.0 foot
# collision spheres: toe [0.12, +/-0.03, -0.03], heel [-0.05, +/-0.025, -0.03].
_TOE_OFFSET_BODY: torch.Tensor = torch.tensor([0.12, 0.0, -0.03], dtype=torch.float32)
_HEEL_OFFSET_BODY: torch.Tensor = torch.tensor([-0.05, 0.0, -0.03], dtype=torch.float32)


def _get_leg_joint_info(
  env: ManagerBasedRlEnv,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return (leg_joint_indices, leg_action_indices, default_joint_pos_leg).

  The indices are resolved from the environment's joint and action order.
  Cached in ``env.extras`` after first call.
  """
  if "leg_joint_indices" in env.extras:
    return (
      env.extras["leg_joint_indices"],
      env.extras["leg_action_indices"],
      env.extras["default_joint_pos_leg"],
    )

  robot = env.scene["robot"]
  all_joint_names = robot.joint_names
  action_manager = cast(Any, env.action_manager)

  # The G1 velocity task uses one 29-D joint_pos action term.
  # ActionManager does not expose per-dimension action_names. For this joint_pos
  # action, the action vector follows the robot actuated joint order, i.e.
  # robot.joint_names.
  if getattr(action_manager, "total_action_dim", None) != len(robot.joint_names):
    raise RuntimeError(
      "Cannot infer leg action indices: expected a single joint_pos action whose "
      f"dimension equals len(robot.joint_names)={len(robot.joint_names)}, but got "
      f"total_action_dim={getattr(action_manager, 'total_action_dim', None)}."
    )
  all_action_names = list(robot.joint_names)

  # Build joint index lookup
  joint_name_to_idx = {name: i for i, name in enumerate(all_joint_names)}
  action_name_to_idx = {name: i for i, name in enumerate(all_action_names)}

  leg_joint_indices = torch.tensor(
    [joint_name_to_idx[n] for n in _G1_LEG_JOINT_NAMES],
    dtype=torch.long,
    device=env.device,
  )
  leg_action_indices = torch.tensor(
    [action_name_to_idx[n] for n in _G1_LEG_JOINT_NAMES],
    dtype=torch.long,
    device=env.device,
  )

  default_joint_pos = robot.data.default_joint_pos
  default_joint_pos_leg = default_joint_pos[:, leg_joint_indices]

  env.extras["leg_joint_indices"] = leg_joint_indices
  env.extras["leg_action_indices"] = leg_action_indices
  env.extras["default_joint_pos_leg"] = default_joint_pos_leg

  return leg_joint_indices, leg_action_indices, default_joint_pos_leg


def _get_foot_link_indices(env: ManagerBasedRlEnv) -> tuple[int, int]:
  """Return (left_foot_link_idx, right_foot_link_idx) for body_link_* APIs."""
  if "left_foot_link_idx" in env.extras:
    return env.extras["left_foot_link_idx"], env.extras["right_foot_link_idx"]

  robot = env.scene["robot"]
  all_body_names = robot.body_names
  name_to_idx = {name: i for i, name in enumerate(all_body_names)}
  left_idx = name_to_idx[_G1_FOOT_LINK_NAMES[0]]
  right_idx = name_to_idx[_G1_FOOT_LINK_NAMES[1]]

  env.extras["left_foot_link_idx"] = left_idx
  env.extras["right_foot_link_idx"] = right_idx
  return left_idx, right_idx


def _get_toe_heel_site_indices(
  env: ManagerBasedRlEnv,
) -> tuple[int, int, int, int] | None:
  """Return toe/heel site indices if the robot MJCF exposes them."""
  if "toe_heel_site_indices" in env.extras:
    return env.extras["toe_heel_site_indices"]

  robot = env.scene["robot"]
  name_to_idx = {name: i for i, name in enumerate(robot.site_names)}
  if not all(name in name_to_idx for name in _G1_TOE_HEEL_SITE_NAMES):
    env.extras["toe_heel_site_indices"] = None
    return None

  indices = (
    name_to_idx[_G1_TOE_HEEL_SITE_NAMES[0]],
    name_to_idx[_G1_TOE_HEEL_SITE_NAMES[1]],
    name_to_idx[_G1_TOE_HEEL_SITE_NAMES[2]],
    name_to_idx[_G1_TOE_HEEL_SITE_NAMES[3]],
  )
  env.extras["toe_heel_site_indices"] = indices
  return indices


def _world_frame_foot_positions(
  env: ManagerBasedRlEnv,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Compute world-frame toe and heel positions from MJCF sites or FK offsets.

  Returns:
    left_toe_pos_w:  [num_envs, 3]
    right_toe_pos_w: [num_envs, 3]
    left_heel_pos_w: [num_envs, 3]
    right_heel_pos_w: [num_envs, 3]

  All positions are in the world frame.
  """
  robot = env.scene["robot"]

  site_indices = _get_toe_heel_site_indices(env)
  if site_indices is not None:
    left_toe_idx, right_toe_idx, left_heel_idx, right_heel_idx = site_indices
    site_pos_w = robot.data.site_pos_w
    return (
      site_pos_w[:, left_toe_idx, :],
      site_pos_w[:, right_toe_idx, :],
      site_pos_w[:, left_heel_idx, :],
      site_pos_w[:, right_heel_idx, :],
    )

  left_idx, right_idx = _get_foot_link_indices(env)
  body_link_pos_w = robot.data.body_link_pos_w  # [num_envs, num_bodies, 3]
  body_link_quat_w = robot.data.body_link_quat_w  # [num_envs, num_bodies, 4]

  _toe = _TOE_OFFSET_BODY.to(env.device)
  _heel = _HEEL_OFFSET_BODY.to(env.device)

  def _link_pos_w(link_idx: int, offset: torch.Tensor) -> torch.Tensor:
    link_pos_w = body_link_pos_w[:, link_idx, :]  # [B, 3]
    link_quat_w = body_link_quat_w[:, link_idx, :]  # [B, 4]
    offset_w = quat_apply(link_quat_w, offset.expand(env.num_envs, -1))
    return link_pos_w + offset_w

  return (
    _link_pos_w(left_idx, _toe),
    _link_pos_w(right_idx, _toe),
    _link_pos_w(left_idx, _heel),
    _link_pos_w(right_idx, _heel),
  )


def _body_frame_foot_positions(
  env: ManagerBasedRlEnv,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Compute body-frame (root/body frame) toe and heel positions.

  Returns:
    left_toe_pos_body:  [num_envs, 3]
    right_toe_pos_body: [num_envs, 3]
    left_heel_pos_body: [num_envs, 3]
    right_heel_pos_body: [num_envs, 3]

  All positions are in the robot's root body frame.
  """
  robot = env.scene["robot"]
  root_pos_w = robot.data.root_link_pos_w  # [num_envs, 3]
  root_quat_w = robot.data.root_link_quat_w  # [num_envs, 4]

  def _world_to_body(point_w: torch.Tensor) -> torch.Tensor:
    return quat_apply_inverse(root_quat_w, point_w - root_pos_w)

  left_toe_w, right_toe_w, left_heel_w, right_heel_w = _world_frame_foot_positions(env)
  return (
    _world_to_body(left_toe_w),
    _world_to_body(right_toe_w),
    _world_to_body(left_heel_w),
    _world_to_body(right_heel_w),
  )


def _body_frame_foot_velocities(
  env: ManagerBasedRlEnv,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Compute body-frame toe linear velocities.

  Uses finite differences from cached position history.  The cache is
  stored in ``env.extras["prev_toe_pos_body"]`` and reset on episode start.

  Returns:
    left_toe_vel_body:  [num_envs, 3]
    right_toe_vel_body: [num_envs, 3]
  """
  left_toe, right_toe, _, _ = _body_frame_foot_positions(env)
  dt = env.step_dt

  if "prev_left_toe_pos_body" not in env.extras:
    # First step: velocity = 0
    env.extras["prev_left_toe_pos_body"] = left_toe.clone()
    env.extras["prev_right_toe_pos_body"] = right_toe.clone()
    return (
      torch.zeros_like(left_toe),
      torch.zeros_like(right_toe),
    )

  prev_left = env.extras["prev_left_toe_pos_body"]
  prev_right = env.extras["prev_right_toe_pos_body"]

  raw_left_vel = (left_toe - prev_left) / dt
  raw_right_vel = (right_toe - prev_right) / dt

  cache_valid = env.extras.get("stair_latent_cache_valid")
  if cache_valid is not None:
    valid = cache_valid.unsqueeze(-1)
    left_vel = torch.where(valid, raw_left_vel, torch.zeros_like(raw_left_vel))
    right_vel = torch.where(valid, raw_right_vel, torch.zeros_like(raw_right_vel))
  else:
    left_vel = raw_left_vel
    right_vel = raw_right_vel

  env.extras["prev_left_toe_pos_body"] = left_toe.clone()
  env.extras["prev_right_toe_pos_body"] = right_toe.clone()

  return left_vel, right_vel


def _clear_foot_velocity_cache(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
  """Reset foot velocity cache for the given env IDs.

  Sets the prev toe positions to the current toe positions so that the
  next velocity computation yields zero instead of a spurious spike.
  """
  if "prev_left_toe_pos_body" not in env.extras:
    return
  left_toe, right_toe, _, _ = _body_frame_foot_positions(env)
  env.extras["prev_left_toe_pos_body"][env_ids] = left_toe[env_ids]
  env.extras["prev_right_toe_pos_body"][env_ids] = right_toe[env_ids]


# ======================================================================
# Env-side stair state-machine labels
# ======================================================================


def toe_riser_event_label(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  """One-frame blocked-swing evidence label that should trigger WRITE."""
  zeros = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  entry_event = env.extras.get(STAIR_ENTRY_EVENT_KEY, zeros)
  return entry_event.bool().float().unsqueeze(-1)


def stair_state_label(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  """Label latched entry-to-flat stair context plus immediate entry evidence."""
  zeros = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  stair_phase = env.extras.get(STAIR_PHASE_KEY)
  recent_evidence = env.extras.get(STAIR_ENTRY_RECENT_EVIDENCE_KEY, zeros)
  if stair_phase is None:
    phase_label = zeros
  else:
    phase_label = stair_phase >= 1
  return (phase_label | recent_evidence.bool()).float().unsqueeze(-1)


def stair_shape_label(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  """Privileged ``[tread_depth, riser_height, valid]`` supervision label.

  The geometry is latched from simulation-only step boundaries at stair entry
  and is never exposed to the actor or latent observation groups. It remains
  valid for the complete accepted stair sequence.
  """
  zeros = torch.zeros(env.num_envs, device=env.device)
  tread_depth = env.extras.get(STAIR_TREAD_DEPTH_LABEL_KEY, zeros)
  riser_height = env.extras.get(STAIR_RISER_HEIGHT_LABEL_KEY, zeros)
  shape_valid = env.extras.get(
    STAIR_SHAPE_LABEL_VALID_KEY,
    torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
  )
  stair_phase = env.extras.get(STAIR_PHASE_KEY)
  if stair_phase is None:
    sequence_active = torch.zeros_like(shape_valid, dtype=torch.bool)
  else:
    sequence_active = stair_phase >= 1
  label_valid = shape_valid.bool() & sequence_active
  return torch.stack(
    [tread_depth, riser_height, label_valid.float()],
    dim=-1,
  )


def safe_stride_label(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Privileged ``[minimum_safe_stride, valid, exact, importance]`` label.

  The target is generated after each completed alternating swing attempt.
  Every valid target is exact privileged geometry supervision. ``exact``
  records whether the deployable interaction history also contains
  expected-riser evidence and is retained for diagnostics only. ``importance``
  briefly emphasizes entry-riser, expected-riser, and rear-partial evidence.
  """
  safe_stride = env.extras.get(MINIMUM_SAFE_STRIDE_KEY)
  safe_stride_valid = env.extras.get(MINIMUM_SAFE_STRIDE_VALID_KEY)
  safe_stride_exact = env.extras.get(MINIMUM_SAFE_STRIDE_EXACT_KEY)
  safe_stride_weight = env.extras.get(MINIMUM_SAFE_STRIDE_WEIGHT_KEY)
  if safe_stride is None or safe_stride_valid is None or safe_stride_exact is None:
    return torch.zeros(env.num_envs, 4, device=env.device)
  if safe_stride_weight is None:
    safe_stride_weight = torch.ones_like(safe_stride)
  stair_phase = env.extras.get(STAIR_PHASE_KEY)
  if stair_phase is None:
    sequence_active = torch.zeros_like(safe_stride_valid, dtype=torch.bool)
  else:
    sequence_active = stair_phase >= 1
  label_valid = safe_stride_valid.bool() & sequence_active
  return torch.stack(
    [
      safe_stride,
      label_valid.float(),
      (safe_stride_exact.bool() & label_valid).float(),
      torch.where(
        label_valid,
        safe_stride_weight.clamp_min(1.0),
        torch.zeros_like(safe_stride_weight),
      ),
    ],
    dim=-1,
  )


def stair_future_event_labels(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Privileged ``[collision_risk, touchdown, landing_quality]`` labels."""
  zeros = torch.zeros(env.num_envs, device=env.device)
  collision_risk = env.extras.get(COLLISION_RISK_KEY, zeros)
  touchdown = env.extras.get(LANDING_TOUCHDOWN_KEY, zeros)
  landing_quality = env.extras.get(LANDING_QUALITY_KEY, zeros)
  return torch.stack(
    [collision_risk.float(), touchdown.float(), landing_quality.float()], dim=-1
  )


# ======================================================================
# Stair latent observation (deployable proprioceptive features)
# ======================================================================


def stair_latent_obs(
  env: ManagerBasedRlEnv,
  toe_contact_sensor_name: str = "toe_terrain_contact",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Construct the stair-focused deployable latent_obs vector.

  This observation contains ONLY proprioceptive features that can be
  obtained from real robot sensors (IMU + joint encoders + FK).  It
  explicitly excludes simulation-only information such as contact forces,
  terrain height scans, depth images, terrain type labels, stair height
  ground truth, and privileged critic observations.

  Feature order (FIXED, must match ONNX export and deployment):
    1. projected_gravity           (3)
    2. base_ang_vel                (3)
    3. base_ang_vel_delta          (3)
    4. left_toe_pos_body           (3)
    5. right_toe_pos_body          (3)
    6. left_heel_pos_body          (3)
    7. right_heel_pos_body         (3)
    8. toe_delta_pos               (3)
    9. heel_delta_pos              (3)
   10. toe_horizontal_distance     (1)
   11. toe_vertical_distance       (1)
   12. heel_vertical_distance      (1)
   13. left_toe_vel_body           (3)
   14. right_toe_vel_body          (3)
   15. left_toe_vel_delta          (3)
   16. right_toe_vel_delta         (3)
   17. previous_action_leg         (12)
   18. leg_joint_tracking_error    (12)
   19. leg_joint_vel               (12)
   20. leg_joint_vel_delta         (12)
   21. command_lin_x               (1)
  Total: ~91

  Parameters
  ----------
  env:
    The environment instance.
  toe_contact_sensor_name:
    Unused (kept for API consistency).
  asset_cfg:
    Scene entity configuration for the robot.

  Returns
  -------
  latent_obs:
    ``[num_envs, latent_obs_dim]`` tensor of deployable proprioceptive features.
  """
  robot = env.scene[asset_cfg.name]
  num_envs = env.num_envs
  device = env.device
  cache_valid = env.extras.get("stair_latent_cache_valid")
  if cache_valid is None:
    cache_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env.extras["stair_latent_cache_valid"] = cache_valid
  valid = cache_valid.unsqueeze(-1)

  # ---- IMU / perturbation ----
  projected_gravity = robot.data.projected_gravity_b  # [B, 3]
  base_ang_vel = robot.data.root_link_ang_vel_b  # [B, 3]

  # base_ang_vel_delta
  if "prev_base_ang_vel" not in env.extras:
    base_ang_vel_delta = torch.zeros_like(base_ang_vel)
  else:
    base_ang_vel_delta = torch.where(
      valid, base_ang_vel - env.extras["prev_base_ang_vel"], 0.0
    )
  env.extras["prev_base_ang_vel"] = base_ang_vel.clone()

  # ---- Foot FK positions (body frame) ----
  left_toe_pos, right_toe_pos, left_heel_pos, right_heel_pos = (
    _body_frame_foot_positions(env)
  )

  # ---- Foot FK positions (world frame, for world-z height differences) ----
  left_toe_w, right_toe_w, left_heel_w, right_heel_w = _world_frame_foot_positions(env)

  # ---- Foot relative geometry ----
  toe_delta = left_toe_pos - right_toe_pos  # [B, 3]
  heel_delta = left_heel_pos - right_heel_pos  # [B, 3]
  # Forward distance along body x (ignores lateral offset).
  toe_horizontal_dist = toe_delta[:, 0:1].abs()
  # World-z height difference (immune to body pitch/roll).
  toe_vertical_dist = left_toe_w[:, 2:3] - right_toe_w[:, 2:3]
  heel_vertical_dist = left_heel_w[:, 2:3] - right_heel_w[:, 2:3]

  # ---- Toe velocities (body frame) ----
  left_toe_vel, right_toe_vel = _body_frame_foot_velocities(env)

  # toe velocity deltas
  if "prev_left_toe_vel" not in env.extras:
    left_toe_vel_delta = torch.zeros_like(left_toe_vel)
    right_toe_vel_delta = torch.zeros_like(right_toe_vel)
  else:
    left_toe_vel_delta = torch.where(
      valid, left_toe_vel - env.extras["prev_left_toe_vel"], 0.0
    )
    right_toe_vel_delta = torch.where(
      valid, right_toe_vel - env.extras["prev_right_toe_vel"], 0.0
    )
  env.extras["prev_left_toe_vel"] = left_toe_vel.clone()
  env.extras["prev_right_toe_vel"] = right_toe_vel.clone()

  # ---- Leg action-response residuals ----
  leg_joint_indices, leg_action_indices, default_joint_pos_leg = _get_leg_joint_info(
    env
  )

  # previous_action (full action from last step).
  # Actions are stored in ActionManager, not robot.data.
  current_action = env.action_manager.action
  if "prev_action" in env.extras:
    prev_action = env.extras["prev_action"]
  else:
    prev_action = torch.zeros_like(current_action)

  previous_action_leg = prev_action[:, leg_action_indices]

  # target_joint_pos_leg = default_joint_pos_leg + action_scale * previous_action_leg
  # action_scale is per-joint; approximate with G1_ACTION_SCALE
  from mjlab.asset_zoo.robots import G1_ACTION_SCALE

  action_scale_leg = torch.tensor(
    [G1_ACTION_SCALE.get(name, 0.25) for name in _G1_LEG_JOINT_NAMES],
    dtype=torch.float32,
    device=device,
  )

  target_joint_pos_leg = default_joint_pos_leg + action_scale_leg * previous_action_leg
  current_joint_pos = robot.data.joint_pos[:, leg_joint_indices]
  leg_joint_tracking_error = target_joint_pos_leg - current_joint_pos

  # leg joint velocity
  current_joint_vel = robot.data.joint_vel[:, leg_joint_indices]  # [B, 12]

  # leg joint velocity delta
  if "prev_leg_joint_vel" not in env.extras:
    leg_joint_vel_delta = torch.zeros_like(current_joint_vel)
  else:
    leg_joint_vel_delta = torch.where(
      valid, current_joint_vel - env.extras["prev_leg_joint_vel"], 0.0
    )
  env.extras["prev_leg_joint_vel"] = current_joint_vel.clone()

  # Update prev_action cache.
  env.extras["prev_action"] = current_action.clone()
  env.extras["stair_latent_cache_valid"][:] = True

  # ---- Command context ----
  command = env.command_manager.get_command("twist")
  if command is None:
    command_lin_x = torch.zeros(num_envs, 1, device=device)
  else:
    command_lin_x = command[:, 0:1]  # [B, 1]

  # ---- Assemble ----
  parts: list[torch.Tensor] = [
    projected_gravity,  # 3
    base_ang_vel,  # 3
    base_ang_vel_delta,  # 3
    left_toe_pos,  # 3
    right_toe_pos,  # 3
    left_heel_pos,  # 3
    right_heel_pos,  # 3
    toe_delta,  # 3
    heel_delta,  # 3
    toe_horizontal_dist,  # 1
    toe_vertical_dist,  # 1
    heel_vertical_dist,  # 1
    left_toe_vel,  # 3
    right_toe_vel,  # 3
    left_toe_vel_delta,  # 3
    right_toe_vel_delta,  # 3
    previous_action_leg,  # 12
    leg_joint_tracking_error,  # 12
    current_joint_vel,  # 12
    leg_joint_vel_delta,  # 12
    command_lin_x,  # 1
  ]

  return torch.cat(parts, dim=-1)


def stair_latent_obs_dim() -> int:
  """Return the expected dimension of stair_latent_obs."""
  n_leg = len(_G1_LEG_JOINT_NAMES)  # 12
  return (
    3
    + 3
    + 3  # gravity, ang_vel, ang_vel_delta (9)
    + 3
    + 3
    + 3
    + 3  # foot positions (12)
    + 3
    + 3  # toe/heel deltas (6)
    + 1
    + 1
    + 1  # distances (3)
    + 3
    + 3  # toe velocities (6)
    + 3
    + 3  # toe vel deltas (6)
    + n_leg  # prev_action_leg (12)
    + n_leg  # tracking error (12)
    + n_leg  # joint_vel (12)
    + n_leg  # joint_vel_delta (12)
    + 1  # command_lin_x
  )  # = 9 + 12 + 6 + 3 + 6 + 6 + 48 + 1 = 91


def reset_stair_latent_cache(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
) -> None:
  """Reset all cached previous-frame values for stair_latent_obs.

  Must be called on episode reset to avoid cross-episode contamination.
  """
  cache_keys = [
    "prev_base_ang_vel",
    "prev_left_toe_vel",
    "prev_right_toe_vel",
    "prev_leg_joint_vel",
    "prev_action",
  ]
  if "stair_latent_cache_valid" in env.extras:
    env.extras["stair_latent_cache_valid"][env_ids] = False
  for key in cache_keys:
    if key in env.extras:
      env.extras[key][env_ids] = 0.0
  _clear_foot_velocity_cache(env, env_ids)
