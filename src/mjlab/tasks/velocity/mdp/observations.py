from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor, ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
  """Sin/cos gait phase used by the Unitree deployment runtime."""
  global_phase = (env.episode_length_buf * env.step_dt) % period / period
  phase = torch.zeros(env.num_envs, 2, device=env.device)
  phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
  phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)

  command = env.command_manager.get_command(command_name)
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
G1 foot link names used for FK toe/heel position estimation.
Note: G1 XML does not have dedicated toe/heel links.  We approximate
toe/heel positions by applying body-frame offsets to the ankle-roll link.
For deployment, these positions should be computed via FK from joint encoders.
"""

# Body-frame toe/heel offsets relative to ankle_roll_link origin
_TOE_OFFSET_BODY: torch.Tensor = torch.tensor([0.12, 0.0, -0.037], dtype=torch.float32)
_HEEL_OFFSET_BODY: torch.Tensor = torch.tensor([-0.05, 0.0, -0.037], dtype=torch.float32)


def _get_leg_joint_info(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
  all_joint_names = robot.data.joint_names
  all_action_names = env.action_manager.action_names

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
  default_joint_pos_leg = default_joint_pos[leg_joint_indices]

  env.extras["leg_joint_indices"] = leg_joint_indices
  env.extras["leg_action_indices"] = leg_action_indices
  env.extras["default_joint_pos_leg"] = default_joint_pos_leg

  return leg_joint_indices, leg_action_indices, default_joint_pos_leg


def _get_foot_link_indices(env: ManagerBasedRlEnv) -> tuple[int, int]:
  """Return (left_foot_link_idx, right_foot_link_idx) for body_link_* APIs."""
  if "left_foot_link_idx" in env.extras:
    return env.extras["left_foot_link_idx"], env.extras["right_foot_link_idx"]

  robot = env.scene["robot"]
  all_body_names = robot.data.body_names
  name_to_idx = {name: i for i, name in enumerate(all_body_names)}
  left_idx = name_to_idx[_G1_FOOT_LINK_NAMES[0]]
  right_idx = name_to_idx[_G1_FOOT_LINK_NAMES[1]]

  env.extras["left_foot_link_idx"] = left_idx
  env.extras["right_foot_link_idx"] = right_idx
  return left_idx, right_idx


def _body_frame_foot_positions(
  env: ManagerBasedRlEnv,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Compute body-frame toe and heel positions via FK + offset.

  Returns:
    left_toe_pos_body:  [num_envs, 3]
    right_toe_pos_body: [num_envs, 3]
    left_heel_pos_body: [num_envs, 3]
    right_heel_pos_body: [num_envs, 3]

  All positions are in the robot's body frame (gravity-aligned).
  """
  robot = env.scene["robot"]
  left_idx, right_idx = _get_foot_link_indices(env)

  # World-frame link positions and quaternions
  body_link_pos_w = robot.data.body_link_pos_w  # [num_envs, num_bodies, 3]
  body_link_quat_w = robot.data.body_link_quat_w  # [num_envs, num_bodies, 4]
  root_quat_w = robot.data.root_link_quat_w  # [num_envs, 4]

  _toe = _TOE_OFFSET_BODY.to(env.device)
  _heel = _HEEL_OFFSET_BODY.to(env.device)

  def _link_pos_body(link_idx: int, offset: torch.Tensor) -> torch.Tensor:
    link_pos_w = body_link_pos_w[:, link_idx, :]  # [B, 3]
    link_quat_w = body_link_quat_w[:, link_idx, :]  # [B, 4]
    # Rotate offset into world frame
    from mjlab.utils.lab_api.math import quat_apply
    offset_w = quat_apply(link_quat_w, offset.expand(env.num_envs, -1))
    point_w = link_pos_w + offset_w
    # Transform to body frame
    from mjlab.utils.lab_api.math import quat_apply_inverse
    point_body = quat_apply_inverse(root_quat_w, point_w)
    return point_body

  left_toe = _link_pos_body(left_idx, _toe)
  right_toe = _link_pos_body(right_idx, _toe)
  left_heel = _link_pos_body(left_idx, _heel)
  right_heel = _link_pos_body(right_idx, _heel)

  return left_toe, right_toe, left_heel, right_heel


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

  left_vel = (left_toe - prev_left) / dt
  right_vel = (right_toe - prev_right) / dt

  env.extras["prev_left_toe_pos_body"] = left_toe.clone()
  env.extras["prev_right_toe_pos_body"] = right_toe.clone()

  return left_vel, right_vel


def _clear_foot_velocity_cache(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
  """Reset foot velocity cache for the given env IDs."""
  if "prev_left_toe_pos_body" in env.extras:
    env.extras["prev_left_toe_pos_body"][env_ids] = 0.0
    env.extras["prev_right_toe_pos_body"][env_ids] = 0.0


# ======================================================================
# Toe-riser event label (from simulation contact sensor)
# ======================================================================

def toe_riser_event_label(
  env: ManagerBasedRlEnv,
  sensor_name: str = "toe_terrain_contact",
  force_threshold: float = 15.0,
  vertical_normal_z_max: float = 0.4,
) -> torch.Tensor:
  """Binary label: does the current frame contain a toe-riser collision?

  Detects horizontal toe blocking forces from the contact sensor.
  This label is used for (1) gate state machine training, (2) auxiliary
  loss supervision, and (3) evaluation metrics.  It NEVER enters latent_obs
  or actor_obs.

  Parameters
  ----------
  sensor_name:
    Name of the ``ContactSensor`` that captures toe-terrain contacts.
  force_threshold:
    Minimum contact force magnitude (N) to count as a collision.
  vertical_normal_z_max:
    Maximum z-component of the contact normal for the collision to be
    considered "horizontal" (toe-riser vs vertical ground contact).

  Returns
  -------
  event:
    ``[num_envs]`` binary tensor (1 = toe-riser collision).
  """
  sensor: ContactSensor = env.scene[sensor_name]
  sd = sensor.data

  assert sd.force is not None, f"Contact sensor '{sensor_name}' has no force data"
  assert sd.normal is not None, f"Contact sensor '{sensor_name}' has no normal data"

  force = sd.force  # [B, max_slots, 3]
  normal = sd.normal  # [B, max_slots, 3]

  force_mag = torch.norm(force, dim=-1)  # [B, max_slots]
  normal_z = normal[..., 2]  # [B, max_slots]

  is_horizontal = normal_z.abs() < vertical_normal_z_max
  is_strong = force_mag > force_threshold

  hit_per_slot = (is_horizontal & is_strong).any(dim=-1)  # [B]
  return hit_per_slot.float()


def stair_state_label(
  env: ManagerBasedRlEnv,
  min_terrain_level: int = 3,
  sensor_name: str = "toe_terrain_contact",
) -> torch.Tensor:
  """Label: is the robot currently in a stair-interaction state?

  First version: terrain_level >= min_terrain_level AND either the robot
  has already experienced a toe-riser event OR the terrain type contains
  stairs.  This is a simplistic heuristic that can be improved with
  actual terrain section metadata.

  Parameters
  ----------
  min_terrain_level:
    Minimum terrain difficulty level to be considered stairs.
  sensor_name:
    Name of the toe-terrain contact sensor (unused in this version).

  Returns
  -------
  stair:
    ``[num_envs]`` binary tensor.
  """
  # Heuristic: terrain_level >= min_terrain_level
  terrain_level = env.terrain_levels
  return (terrain_level >= min_terrain_level).float()


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

  # ---- IMU / perturbation ----
  projected_gravity = robot.data.projected_gravity  # [B, 3]
  base_ang_vel = robot.data.root_link_ang_vel_b  # [B, 3]

  # base_ang_vel_delta
  if "prev_base_ang_vel" in env.extras:
    base_ang_vel_delta = base_ang_vel - env.extras["prev_base_ang_vel"]
  else:
    base_ang_vel_delta = torch.zeros_like(base_ang_vel)
  env.extras["prev_base_ang_vel"] = base_ang_vel.clone()

  # ---- Foot FK positions (body frame) ----
  left_toe_pos, right_toe_pos, left_heel_pos, right_heel_pos = (
    _body_frame_foot_positions(env)
  )

  # ---- Foot relative geometry ----
  toe_delta = left_toe_pos - right_toe_pos  # [B, 3]
  heel_delta = left_heel_pos - right_heel_pos  # [B, 3]
  toe_horizontal_dist = torch.sqrt(toe_delta[:, 0] ** 2 + toe_delta[:, 1] ** 2).unsqueeze(-1)
  toe_vertical_dist = toe_delta[:, 2:3]
  heel_vertical_dist = heel_delta[:, 2:3]

  # ---- Toe velocities (body frame) ----
  left_toe_vel, right_toe_vel = _body_frame_foot_velocities(env)

  # toe velocity deltas
  if "prev_left_toe_vel" in env.extras:
    left_toe_vel_delta = left_toe_vel - env.extras["prev_left_toe_vel"]
    right_toe_vel_delta = right_toe_vel - env.extras["prev_right_toe_vel"]
  else:
    left_toe_vel_delta = torch.zeros_like(left_toe_vel)
    right_toe_vel_delta = torch.zeros_like(right_toe_vel)
  env.extras["prev_left_toe_vel"] = left_toe_vel.clone()
  env.extras["prev_right_toe_vel"] = right_toe_vel.clone()

  # ---- Leg action-response residuals ----
  leg_joint_indices, leg_action_indices, default_joint_pos_leg = _get_leg_joint_info(env)

  # previous_action (full action from last step)
  if "prev_action" in env.extras:
    prev_action = env.extras["prev_action"]
  else:
    prev_action = torch.zeros(num_envs, env.num_actions, device=device)

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
  if "prev_leg_joint_vel" in env.extras:
    leg_joint_vel_delta = current_joint_vel - env.extras["prev_leg_joint_vel"]
  else:
    leg_joint_vel_delta = torch.zeros_like(current_joint_vel)
  env.extras["prev_leg_joint_vel"] = current_joint_vel.clone()

  # Update prev_action cache
  env.extras["prev_action"] = robot.data.last_action.clone()

  # ---- Command context ----
  command = env.command_manager.get_command("twist")
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
    3 + 3 + 3  # gravity, ang_vel, ang_vel_delta (9)
    + 3 + 3 + 3 + 3  # foot positions (12)
    + 3 + 3  # toe/heel deltas (6)
    + 1 + 1 + 1  # distances (3)
    + 3 + 3  # toe velocities (6)
    + 3 + 3  # toe vel deltas (6)
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
  for key in cache_keys:
    if key in env.extras:
      env.extras[key][env_ids] = 0.0
  _clear_foot_velocity_cache(env, env_ids)
