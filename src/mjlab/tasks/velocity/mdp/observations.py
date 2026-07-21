from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor, ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat

from .stair_geometry import (
  COLLISION_RISK_KEY,
  LANDING_QUALITY_KEY,
  LANDING_TOUCHDOWN_KEY,
  MINIMUM_SAFE_STRIDE_EXACT_KEY,
  MINIMUM_SAFE_STRIDE_INTERVAL_VALID_KEY,
  MINIMUM_SAFE_STRIDE_KEY,
  MINIMUM_SAFE_STRIDE_UPPER_KEY,
  MINIMUM_SAFE_STRIDE_VALID_KEY,
  MINIMUM_SAFE_STRIDE_WEIGHT_KEY,
  STAIR_ADJACENT_PAIR_DEPTH_KEY,
  STAIR_ADJACENT_PAIR_EVENT_KEY,
  STAIR_ADJACENT_PAIR_HEIGHT_KEY,
  STAIR_ADJACENT_PAIR_VALID_KEY,
  STAIR_CURRENT_CONTACT_DURATION_KEY,
  STAIR_CURRENT_GROUND_CONTACT_KEY,
  STAIR_DEPTH_CONFIRMATION_AGE_KEY,
  STAIR_DEPTH_CONFIRMATION_EVENT_KEY,
  STAIR_ENTRY_EVENT_KEY,
  STAIR_ENTRY_RECENT_EVIDENCE_KEY,
  STAIR_PHASE_KEY,
  STAIR_RISER_HEIGHT_LABEL_KEY,
  STAIR_SHAPE_LABEL_VALID_KEY,
  STAIR_TREAD_DEPTH_LABEL_KEY,
  TOE_RISER_NEW_HIT_BY_FOOT_KEY,
  TOE_RISER_NEW_HIT_KEY,
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
  """One-frame blocked-swing label for entry WRITE or memory shape refresh."""
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


def stair_shape_component_valid_label(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  """Return privileged ``[depth_valid, height_valid]`` component masks.

  Depth is supervised throughout accepted stair context. Strict deployable
  depth-confirmation evidence remains available separately through
  ``stair_depth_confirmation_*`` labels, but keeping the depth component dense
  gives the slow latent a continuous geometry target instead of a rare pulse.
  """
  zeros = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  shape_valid = env.extras.get(STAIR_SHAPE_LABEL_VALID_KEY, zeros).bool()
  stair_phase = env.extras.get(STAIR_PHASE_KEY)
  sequence_active = zeros if stair_phase is None else stair_phase >= 1
  height_valid = shape_valid & sequence_active
  depth_valid = height_valid
  return torch.stack(
    [depth_valid.float(), height_valid.float()],
    dim=-1,
  )


def safe_stride_label(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Privileged SafeStride interval and interaction-evidence label.

  The layout is ``[lower, valid, exact, importance, upper]``. The two bounds
  describe translations that keep the complete sole inside the requested
  tread. They are privileged training targets only. ``exact`` records whether
  deployable interaction history contains expected-riser evidence.
  """
  safe_stride = env.extras.get(MINIMUM_SAFE_STRIDE_KEY)
  safe_stride_upper = env.extras.get(MINIMUM_SAFE_STRIDE_UPPER_KEY)
  safe_stride_valid = env.extras.get(MINIMUM_SAFE_STRIDE_VALID_KEY)
  safe_stride_exact = env.extras.get(MINIMUM_SAFE_STRIDE_EXACT_KEY)
  safe_stride_weight = env.extras.get(MINIMUM_SAFE_STRIDE_WEIGHT_KEY)
  if safe_stride is None or safe_stride_valid is None or safe_stride_exact is None:
    return torch.zeros(env.num_envs, 5, device=env.device)
  if safe_stride_upper is None:
    safe_stride_upper = safe_stride
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
      torch.where(label_valid, safe_stride_upper, torch.zeros_like(safe_stride_upper)),
    ],
    dim=-1,
  )


def safe_stride_interval_valid_label(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return whether both privileged SafeStride interval bounds are observable."""
  interval_valid = env.extras.get(MINIMUM_SAFE_STRIDE_INTERVAL_VALID_KEY)
  lower_valid = env.extras.get(MINIMUM_SAFE_STRIDE_VALID_KEY)
  stair_phase = env.extras.get(STAIR_PHASE_KEY)
  if interval_valid is None or lower_valid is None or stair_phase is None:
    return torch.zeros(env.num_envs, 1, device=env.device)
  valid = interval_valid.bool() & lower_valid.bool() & (stair_phase >= 1)
  return valid.float().unsqueeze(-1)


def stair_depth_confirmation_event_label(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return a one-frame pulse for the first confirmed depth in each sequence."""
  event = env.extras.get(STAIR_DEPTH_CONFIRMATION_EVENT_KEY)
  stair_phase = env.extras.get(STAIR_PHASE_KEY)
  if event is None or stair_phase is None:
    return torch.zeros(env.num_envs, 1, device=env.device)
  return (event.bool() & (stair_phase >= 1)).float().unsqueeze(-1)


def stair_depth_confirmation_age_label(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return frames since strict depth confirmation, or -1 before confirmation."""
  age = env.extras.get(STAIR_DEPTH_CONFIRMATION_AGE_KEY)
  stair_phase = env.extras.get(STAIR_PHASE_KEY)
  if age is None or stair_phase is None:
    return torch.full((env.num_envs, 1), -1.0, device=env.device)
  active_age = torch.where(
    stair_phase >= 1,
    age.float(),
    torch.full_like(age, -1).float(),
  )
  return active_age.unsqueeze(-1)


def geometry_probe_validation_mask(
  env: ManagerBasedRlEnv,
  validation_fraction: float = 0.2,
) -> torch.Tensor:
  """Assign a deterministic held-out environment subset to Probe validation."""
  if not 0.0 < validation_fraction < 1.0:
    raise ValueError("validation_fraction must be in (0, 1).")
  env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  hashed = torch.remainder(env_ids * 1_103_515_245 + 12_345, 10_000)
  threshold = round(validation_fraction * 10_000)
  return (hashed < threshold).float().unsqueeze(-1)


def stair_adjacent_pair_evidence_label(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return deployable adjacent-support pair observations and validity."""
  zeros = torch.zeros(env.num_envs, device=env.device)
  depth = env.extras.get(STAIR_ADJACENT_PAIR_DEPTH_KEY, zeros)
  height = env.extras.get(STAIR_ADJACENT_PAIR_HEIGHT_KEY, zeros)
  valid = env.extras.get(STAIR_ADJACENT_PAIR_VALID_KEY, zeros).bool()
  event = env.extras.get(STAIR_ADJACENT_PAIR_EVENT_KEY, zeros).bool()
  return torch.stack(
    [
      torch.where(valid, depth, zeros),
      torch.where(valid, height, zeros),
      valid.float(),
      (event & valid).float(),
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
  include_gait_phase: bool = False,
  gait_period: float = 0.6,
  command_name: str = "twist",
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
   22. gait_phase                  (2, optional Semantic-v2 input)
  Total: 91, or 93 when ``include_gait_phase=True``.

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
  if include_gait_phase:
    parts.append(phase(env, gait_period, command_name))

  return torch.cat(parts, dim=-1)


def stair_latent_obs_dim(include_gait_phase: bool = False) -> int:
  """Return the expected dimension of stair_latent_obs."""
  n_leg = len(_G1_LEG_JOINT_NAMES)  # 12
  base_dim = (
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
  return base_dim + (2 if include_gait_phase else 0)


FOOT_EVENT_MEMORY_LEN = 6
FOOTPRINT_SLOT_DIM = 17
TOE_MARK_SLOT_DIM = 16
FOOT_EVENT_PAIR_COUNT = FOOT_EVENT_MEMORY_LEN - 1
FOOT_EVENT_PAIR_FEATURE_DIM = 10
FOOT_EVENT_PAIR_SUMMARY_DIM = FOOT_EVENT_PAIR_COUNT * FOOT_EVENT_PAIR_FEATURE_DIM
FOOT_EVENT_TOE_SUMMARY_DIM = 10
FOOT_EVENT_GEOMETRY_STATS_DIM = 10
FOOT_EVENT_TREND_DIM = FOOT_EVENT_GEOMETRY_STATS_DIM
FOOT_EVENT_RATCHET_DIM = 10
FOOT_EVENT_TOE_SUMMARY_START = FOOT_EVENT_PAIR_SUMMARY_DIM
FOOT_EVENT_GEOMETRY_STATS_START = (
  FOOT_EVENT_TOE_SUMMARY_START + FOOT_EVENT_TOE_SUMMARY_DIM
)
FOOT_EVENT_RATCHET_START = FOOT_EVENT_GEOMETRY_STATS_START + FOOT_EVENT_TREND_DIM
FOOT_EVENT_SUMMARY_DIM = FOOT_EVENT_RATCHET_START + FOOT_EVENT_RATCHET_DIM
FOOT_EVENT_MEMORY_OBS_DIM = (
  FOOT_EVENT_MEMORY_LEN * FOOTPRINT_SLOT_DIM
  + FOOT_EVENT_MEMORY_LEN * TOE_MARK_SLOT_DIM
  + FOOT_EVENT_SUMMARY_DIM
)


def foot_event_memory_obs_dim(
  memory_len: int = FOOT_EVENT_MEMORY_LEN,
  include_summary: bool = True,
  include_raw_memory: bool = True,
) -> int:
  """Return flattened deployable footprint/toe-mark memory dimension."""
  dim = int(memory_len) * (FOOTPRINT_SLOT_DIM + TOE_MARK_SLOT_DIM)
  if not include_summary:
    return dim
  return (dim if include_raw_memory else 0) + FOOT_EVENT_SUMMARY_DIM


def _tensor_extra(
  env: ManagerBasedRlEnv,
  key: str,
  shape: tuple[int, ...],
  dtype: torch.dtype,
  fill_value: float | int | bool,
) -> torch.Tensor:
  value = env.extras.get(key)
  if isinstance(value, torch.Tensor):
    return value.to(device=env.device, dtype=dtype)
  return torch.full((env.num_envs, *shape), fill_value, dtype=dtype, device=env.device)


def _per_foot_toe_hit(env: ManagerBasedRlEnv) -> torch.Tensor:
  value = env.extras.get(TOE_RISER_NEW_HIT_BY_FOOT_KEY)
  if isinstance(value, torch.Tensor):
    toe_hit = value.to(device=env.device, dtype=torch.bool)
  else:
    toe_hit = _tensor_extra(env, TOE_RISER_NEW_HIT_KEY, (2,), torch.bool, False)
  if toe_hit.ndim == 1:
    toe_hit = toe_hit[:, None].expand(-1, 2)
  return toe_hit[:, :2].bool()


def _root_pose_from_env(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
  robot = env.scene["robot"]
  return (
    robot.data.root_link_pos_w.to(device=env.device, dtype=torch.float32),
    robot.data.root_link_quat_w.to(device=env.device, dtype=torch.float32),
  )


def _world_to_current_base_yaw(
  point_w: torch.Tensor,
  root_pos_w: torch.Tensor,
  root_quat_w: torch.Tensor,
) -> torch.Tensor:
  quat_shape = root_quat_w.shape
  root_yaw = yaw_quat(root_quat_w.reshape(-1, 4)).reshape(quat_shape)
  return quat_apply_inverse(root_yaw, point_w - root_pos_w)


class FootEventMemoryObs:
  """Causal deployable-shaped memory of recent footprints and toe-riser hits.

  The training term uses oracle contact/toe-hit extras to synthesize detector-like
  events, but the returned features are restricted to deployable information:
  event source/confidence/age, FK-derived event positions, contact probabilities,
  and gait phase. Footprints are stance-latched so one stance can create at most
  one footprint slot.
  """

  def __init__(self, cfg: Any, env: ManagerBasedRlEnv) -> None:
    params = cfg.params
    self.num_envs = int(env.num_envs)
    self.device = torch.device(env.device)
    self.memory_len = int(params.get("memory_len", FOOT_EVENT_MEMORY_LEN))
    if self.memory_len <= 0:
      raise ValueError("memory_len must be positive.")
    self.age_norm_s = max(float(params.get("age_norm_s", 1.5)), 1.0e-6)
    self.stance_age_norm_s = max(
      float(params.get("stance_age_norm_s", self.age_norm_s)),
      1.0e-6,
    )
    self.gait_period = max(float(params.get("gait_period", 0.6)), 1.0e-6)
    self.command_name = str(params.get("command_name", "twist"))
    self.noise_enabled = bool(params.get("noise_enabled", True))
    self.include_summary = bool(params.get("include_summary", True))
    self.include_raw_memory = bool(params.get("include_raw_memory", True))

    self.contact_true_prob_range = tuple(
      params.get("contact_true_prob_range", (0.75, 1.0))
    )
    self.contact_false_prob_range = tuple(
      params.get("contact_false_prob_range", (0.0, 0.15))
    )
    self.touchdown_miss_prob = float(params.get("touchdown_miss_prob", 0.18))
    self.touchdown_false_positive_prob = float(
      params.get("touchdown_false_positive_prob", 0.002)
    )
    self.touchdown_confidence_range = tuple(
      params.get("touchdown_confidence_range", (0.70, 1.0))
    )
    self.predicted_fill_confidence_range = tuple(
      params.get("predicted_fill_confidence_range", (0.20, 0.50))
    )
    self.footprint_false_positive_confidence_range = tuple(
      params.get("footprint_false_positive_confidence_range", (0.20, 0.50))
    )
    self.touchdown_prob_range = tuple(params.get("touchdown_prob_range", (0.70, 1.0)))
    self.predicted_fill_touchdown_prob_range = tuple(
      params.get("predicted_fill_touchdown_prob_range", (0.20, 0.50))
    )
    self.footprint_false_positive_prob_range = tuple(
      params.get("footprint_false_positive_prob_range", (0.55, 0.90))
    )
    self.footprint_xy_noise_range_m = tuple(
      params.get("footprint_xy_noise_range_m", (0.01, 0.03))
    )
    self.footprint_z_noise_range_m = tuple(
      params.get("footprint_z_noise_range_m", (0.005, 0.02))
    )

    self.toe_miss_prob = float(params.get("toe_miss_prob", 0.22))
    self.toe_false_positive_prob = float(params.get("toe_false_positive_prob", 0.006))
    self.toe_confidence_range = tuple(params.get("toe_confidence_range", (0.30, 0.75)))
    self.toe_false_positive_confidence_range = tuple(
      params.get("toe_false_positive_confidence_range", (0.20, 0.55))
    )
    self.toe_hit_prob_range = tuple(params.get("toe_hit_prob_range", (0.55, 0.90)))
    self.toe_false_positive_prob_range = tuple(
      params.get("toe_false_positive_prob_range", (0.45, 0.80))
    )
    self.toe_xy_noise_range_m = tuple(params.get("toe_xy_noise_range_m", (0.01, 0.03)))
    self.toe_z_noise_range_m = tuple(params.get("toe_z_noise_range_m", (0.005, 0.02)))

    self.release_contact_prob_threshold = float(
      params.get("release_contact_prob_threshold", 0.20)
    )
    self.release_confirm_frames = max(
      int(params.get("release_confirm_frames", 2)),
      1,
    )
    self.early_contact_time_s = max(
      float(params.get("early_contact_time_s", 0.08)), 0.0
    )
    self.predicted_fill_enabled = bool(params.get("predicted_fill_enabled", True))
    self.ratchet_height_threshold_m = max(
      float(params.get("ratchet_height_threshold_m", 0.025)),
      0.0,
    )
    self.ratchet_flat_height_threshold_m = max(
      float(params.get("ratchet_flat_height_threshold_m", 0.02)),
      0.0,
    )
    self.ratchet_probe_increment_m = max(
      float(params.get("ratchet_probe_increment_m", 0.025)),
      0.0,
    )
    self.ratchet_min_stride_m = float(params.get("ratchet_min_stride_m", 0.10))
    self.ratchet_max_stride_m = max(
      float(params.get("ratchet_max_stride_m", 0.55)),
      self.ratchet_min_stride_m,
    )
    self.ratchet_reset_flat_pairs = max(
      int(params.get("ratchet_reset_flat_pairs", 4)),
      1,
    )

    self.footprints = torch.zeros(
      self.num_envs,
      self.memory_len,
      FOOTPRINT_SLOT_DIM,
      dtype=torch.float32,
      device=self.device,
    )
    self.toe_marks = torch.zeros(
      self.num_envs,
      self.memory_len,
      TOE_MARK_SLOT_DIM,
      dtype=torch.float32,
      device=self.device,
    )
    self.footprint_valid = torch.zeros(
      self.num_envs,
      self.memory_len,
      dtype=torch.bool,
      device=self.device,
    )
    self.toe_valid = torch.zeros_like(self.footprint_valid)
    self.footprint_pos_w = torch.zeros(self.num_envs, self.memory_len, 3).to(
      self.device
    )
    self.toe_pos_w = torch.zeros_like(self.footprint_pos_w)

    self.prev_ground_contact = torch.zeros(
      self.num_envs,
      2,
      dtype=torch.bool,
      device=self.device,
    )
    self.prev_ground_contact_valid = torch.zeros(
      self.num_envs,
      dtype=torch.bool,
      device=self.device,
    )
    self.foot_in_stance = torch.zeros_like(self.prev_ground_contact)
    self.release_count = torch.zeros(
      self.num_envs,
      2,
      dtype=torch.long,
      device=self.device,
    )
    self.stance_age_s = torch.zeros(
      self.num_envs,
      2,
      dtype=torch.float32,
      device=self.device,
    )
    self.active_footprint_slot = torch.full(
      (self.num_envs, 2),
      -1,
      dtype=torch.long,
      device=self.device,
    )
    self.ratchet_active = torch.zeros(
      self.num_envs,
      dtype=torch.bool,
      device=self.device,
    )
    self.ratchet_lower_s = torch.zeros(
      self.num_envs,
      dtype=torch.float32,
      device=self.device,
    )
    self.ratchet_probe_target_s = torch.zeros_like(self.ratchet_lower_s)
    self.ratchet_last_forward_up_stride = torch.zeros_like(self.ratchet_lower_s)
    self.ratchet_stride_growth = torch.zeros_like(self.ratchet_lower_s)
    self.ratchet_safe_no_hit_steps = torch.zeros(
      self.num_envs,
      dtype=torch.long,
      device=self.device,
    )
    self.ratchet_confidence = torch.zeros_like(self.ratchet_lower_s)
    self.ratchet_age_s = torch.zeros_like(self.ratchet_lower_s)
    self.ratchet_flat_pair_steps = torch.zeros_like(self.ratchet_safe_no_hit_steps)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = self._env_ids(env_ids)
    if ids.numel() == 0:
      return
    self.footprints[ids] = 0.0
    self.toe_marks[ids] = 0.0
    self.footprint_valid[ids] = False
    self.toe_valid[ids] = False
    self.footprint_pos_w[ids] = 0.0
    self.toe_pos_w[ids] = 0.0
    self.prev_ground_contact[ids] = False
    self.prev_ground_contact_valid[ids] = False
    self.foot_in_stance[ids] = False
    self.release_count[ids] = 0
    self.stance_age_s[ids] = 0.0
    self.active_footprint_slot[ids] = -1
    self.ratchet_active[ids] = False
    self.ratchet_lower_s[ids] = 0.0
    self.ratchet_probe_target_s[ids] = 0.0
    self.ratchet_last_forward_up_stride[ids] = 0.0
    self.ratchet_stride_growth[ids] = 0.0
    self.ratchet_safe_no_hit_steps[ids] = 0
    self.ratchet_confidence[ids] = 0.0
    self.ratchet_age_s[ids] = 0.0
    self.ratchet_flat_pair_steps[ids] = 0

  def __call__(self, env: ManagerBasedRlEnv, **_: Any) -> torch.Tensor:
    root_pos_w, root_quat_w = _root_pose_from_env(env)
    self._age_and_refresh(float(env.step_dt), root_pos_w, root_quat_w)

    ground_contact = _tensor_extra(
      env,
      STAIR_CURRENT_GROUND_CONTACT_KEY,
      (2,),
      torch.bool,
      False,
    )[:, :2].bool()
    contact_duration = _tensor_extra(
      env,
      STAIR_CURRENT_CONTACT_DURATION_KEY,
      (2,),
      torch.float32,
      0.0,
    )[:, :2]
    contact_prob = self._contact_prob(ground_contact)
    self._update_stance_latches(contact_prob, ground_contact, env.step_dt)

    foot_pos_body, toe_pos_body = self._foot_and_toe_points(env)
    phase_now = phase(env, self.gait_period, self.command_name)
    true_touchdown = (
      ground_contact
      & ~self.prev_ground_contact
      & self.prev_ground_contact_valid[:, None]
    )
    toe_hit = _per_foot_toe_hit(env)
    new_footprint_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    new_toe_mark_any = torch.zeros_like(new_footprint_any)

    for foot_id in range(2):
      allowed = ~self.foot_in_stance[:, foot_id]
      touchdown_mask = true_touchdown[:, foot_id] & allowed
      missed = touchdown_mask & self._bernoulli(self.touchdown_miss_prob)
      confirmed = touchdown_mask & ~missed
      predicted = missed & self.predicted_fill_enabled
      false_positive = (
        allowed & ~touchdown_mask & self._bernoulli(self.touchdown_false_positive_prob)
      )
      footprint_mask = confirmed | predicted | false_positive
      new_footprint_any |= footprint_mask
      confirmed_touchdown_prob = self._uniform_like(
        confirmed,
        self.touchdown_prob_range,
        default=1.0,
      )
      predicted_touchdown_prob = self._uniform_like(
        predicted,
        self.predicted_fill_touchdown_prob_range,
        default=0.35,
      )
      false_positive_touchdown_prob = self._uniform_like(
        false_positive,
        self.footprint_false_positive_prob_range,
        default=0.7,
      )
      confirmed_confidence = self._uniform_like(
        confirmed,
        self.touchdown_confidence_range,
        default=1.0,
      )
      predicted_confidence = self._uniform_like(
        predicted,
        self.predicted_fill_confidence_range,
        default=0.35,
      )
      false_positive_confidence = self._uniform_like(
        false_positive,
        self.footprint_false_positive_confidence_range,
        default=0.35,
      )
      self._push_footprint(
        mask=footprint_mask,
        foot_id=foot_id,
        point_body=foot_pos_body[:, foot_id],
        root_pos_w=root_pos_w,
        root_quat_w=root_quat_w,
        phase_now=phase_now,
        contact_prob=contact_prob[:, foot_id],
        touchdown_prob=torch.where(
          confirmed,
          confirmed_touchdown_prob,
          torch.where(
            predicted,
            predicted_touchdown_prob,
            false_positive_touchdown_prob,
          ),
        ),
        confidence=torch.where(
          confirmed,
          confirmed_confidence,
          torch.where(
            predicted,
            predicted_confidence,
            false_positive_confidence,
          ),
        ),
        source_predicted_fill=predicted,
      )

      new_oracle_stance = touchdown_mask & ~footprint_mask
      self.foot_in_stance[:, foot_id] = (
        self.foot_in_stance[:, foot_id] | new_oracle_stance
      )
      self.stance_age_s[:, foot_id] = torch.where(
        new_oracle_stance,
        torch.zeros_like(self.stance_age_s[:, foot_id]),
        self.stance_age_s[:, foot_id],
      )

      real_toe = toe_hit[:, foot_id] & ~self._bernoulli(self.toe_miss_prob)
      toe_false_positive = self._bernoulli(self.toe_false_positive_prob)
      false_toe = toe_false_positive & ~real_toe
      toe_mark_mask = real_toe | false_toe
      new_toe_mark_any |= toe_mark_mask
      real_toe_prob = self._uniform_like(real_toe, self.toe_hit_prob_range, default=0.7)
      false_toe_prob = self._uniform_like(
        false_toe,
        self.toe_false_positive_prob_range,
        default=0.6,
      )
      real_toe_confidence = self._uniform_like(
        real_toe,
        self.toe_confidence_range,
        default=0.5,
      )
      false_toe_confidence = self._uniform_like(
        false_toe,
        self.toe_false_positive_confidence_range,
        default=0.35,
      )
      self._push_toe_mark(
        mask=toe_mark_mask,
        foot_id=foot_id,
        point_body=toe_pos_body[:, foot_id],
        root_pos_w=root_pos_w,
        root_quat_w=root_quat_w,
        phase_now=phase_now,
        contact_prob=contact_prob[:, foot_id],
        toe_prob=torch.where(false_toe, false_toe_prob, real_toe_prob),
        confidence=torch.where(
          false_toe,
          false_toe_confidence,
          real_toe_confidence,
        ),
        swing_or_early_contact=(
          ~ground_contact[:, foot_id]
          | (contact_duration[:, foot_id] <= self.early_contact_time_s)
        ),
        false_positive=false_toe,
      )

    unlatched_contact = ground_contact & ~self.foot_in_stance
    self.foot_in_stance |= ground_contact
    self.stance_age_s = torch.where(
      unlatched_contact,
      torch.zeros_like(self.stance_age_s),
      self.stance_age_s,
    )
    self.prev_ground_contact.copy_(ground_contact)
    self.prev_ground_contact_valid[:] = True
    self._refresh_relative_positions(root_pos_w, root_quat_w)
    raw_memory = torch.cat(
      [
        self.footprints.reshape(self.num_envs, -1),
        self.toe_marks.reshape(self.num_envs, -1),
      ],
      dim=-1,
    )
    if not self.include_summary:
      return raw_memory
    summary = self._compute_event_summary(
      update_ratchet=True,
      new_footprint_any=new_footprint_any,
      new_toe_mark_any=new_toe_mark_any,
      step_dt=float(env.step_dt),
    )
    self._log_summary_metrics(env, summary, new_footprint_any, new_toe_mark_any)
    if not self.include_raw_memory:
      return summary
    return torch.cat([raw_memory, summary], dim=-1)

  def _compute_event_summary(
    self,
    *,
    update_ratchet: bool = False,
    new_footprint_any: torch.Tensor | None = None,
    new_toe_mark_any: torch.Tensor | None = None,
    step_dt: float = 0.0,
  ) -> torch.Tensor:
    """Summarize ordered deployable event memory into direct geometry cues.

    Layout:
    ordered footprint pairs [0:50], latest toe cue [50:60],
    simple footprint statistics [60:70], and stride-ratchet state [70:80].
    """
    summary = torch.zeros(
      self.num_envs,
      FOOT_EVENT_SUMMARY_DIM,
      dtype=torch.float32,
      device=self.device,
    )
    pair_features = self._all_adjacent_pair_features()
    pair_count = min(pair_features.shape[1], FOOT_EVENT_PAIR_COUNT)
    if pair_count > 0:
      pair_end = pair_count * FOOT_EVENT_PAIR_FEATURE_DIM
      summary[:, :pair_end] = pair_features[:, :pair_count].reshape(
        self.num_envs,
        pair_end,
      )

    toe_features = self._latest_toe_features()
    summary[
      :,
      FOOT_EVENT_TOE_SUMMARY_START:FOOT_EVENT_GEOMETRY_STATS_START,
    ] = toe_features
    summary[:, FOOT_EVENT_GEOMETRY_STATS_START:FOOT_EVENT_RATCHET_START] = (
      self._summary_stats_features(pair_features)
    )
    if update_ratchet:
      if new_footprint_any is None:
        new_footprint_any = torch.zeros(
          self.num_envs,
          dtype=torch.bool,
          device=self.device,
        )
      if new_toe_mark_any is None:
        new_toe_mark_any = torch.zeros_like(new_footprint_any)
      self._update_ratchet_state(
        pair_features,
        toe_features,
        new_footprint_any,
        new_toe_mark_any,
        step_dt,
      )
    summary[:, FOOT_EVENT_RATCHET_START:] = self._ratchet_features(toe_features)
    return summary

  def _all_adjacent_pair_features(
    self,
  ) -> torch.Tensor:
    pair_features = torch.zeros(
      self.num_envs,
      max(self.memory_len - 1, 0),
      FOOT_EVENT_PAIR_FEATURE_DIM,
      dtype=torch.float32,
      device=self.device,
    )
    if self.memory_len < 2:
      return pair_features

    newer = self.footprints[:, :-1]
    older = self.footprints[:, 1:]
    valid = self.footprint_valid[:, :-1] & self.footprint_valid[:, 1:]
    return self._pair_features(newer, older, valid)

  def _summary_stats_features(
    self,
    pair_features: torch.Tensor,
  ) -> torch.Tensor:
    """Aggregate direct geometry from ordered footprint-pair features.

    Layout:
    [0] valid pair count / max pair count.
    [1] confidence-weighted mean signed stride.
    [2] confidence-weighted mean signed height.
    [3:5] latest signed stride/height.
    [5:7] latest pair minus previous pair stride/height.
    [7] max forward signed stride.
    [8] max forward stride among pairs with positive signed height.
    [9] confidence-weighted mean positive signed height.
    """
    stats = torch.zeros(
      self.num_envs,
      FOOT_EVENT_GEOMETRY_STATS_DIM,
      dtype=torch.float32,
      device=self.device,
    )
    pair_count = min(pair_features.shape[1], FOOT_EVENT_PAIR_COUNT)
    if pair_count <= 0:
      return stats

    pair_features = pair_features[:, :pair_count]
    valid = pair_features[..., 0] > 0.5
    weight = pair_features[..., 1] * valid.float()
    delta_s = pair_features[..., 5]
    delta_z = pair_features[..., 6]

    count = valid.float().sum(dim=-1)
    weight_denom = weight.sum(dim=-1).clamp_min(1.0e-6)
    weighted_mean_s = (delta_s * weight).sum(dim=-1) / weight_denom
    weighted_mean_z = (delta_z * weight).sum(dim=-1) / weight_denom

    latest_valid = valid[:, 0]
    if pair_count > 1:
      previous_valid = valid[:, 1]
      latest_minus_prev = torch.where(
        latest_valid & previous_valid,
        delta_s[:, 0] - delta_s[:, 1],
        torch.zeros_like(count),
      )
      height_latest_minus_prev = torch.where(
        latest_valid & previous_valid,
        delta_z[:, 0] - delta_z[:, 1],
        torch.zeros_like(count),
      )
    else:
      latest_minus_prev = torch.zeros_like(count)
      height_latest_minus_prev = torch.zeros_like(count)

    forward = valid & (delta_s > 0.0)
    forward_up = forward & (delta_z > 0.025)
    positive_height = valid & (delta_z > 0.025)
    max_forward_stride = torch.where(
      forward.any(dim=-1),
      delta_s.masked_fill(~forward, -1.0).max(dim=-1).values,
      torch.zeros_like(count),
    )
    max_forward_up_stride = torch.where(
      forward_up.any(dim=-1),
      delta_s.masked_fill(~forward_up, -1.0).max(dim=-1).values,
      torch.zeros_like(count),
    )
    positive_weight = weight * positive_height.float()
    positive_denom = positive_weight.sum(dim=-1).clamp_min(1.0e-6)
    positive_height_mean = (delta_z * positive_weight).sum(dim=-1) / positive_denom

    stats[:, 0] = (count / float(max(FOOT_EVENT_PAIR_COUNT, 1))).clamp(0.0, 1.0)
    stats[:, 1] = torch.where(count > 0.0, weighted_mean_s, torch.zeros_like(count))
    stats[:, 2] = torch.where(count > 0.0, weighted_mean_z, torch.zeros_like(count))
    stats[:, 3] = torch.where(latest_valid, delta_s[:, 0], torch.zeros_like(count))
    stats[:, 4] = torch.where(latest_valid, delta_z[:, 0], torch.zeros_like(count))
    stats[:, 5] = latest_minus_prev
    stats[:, 6] = height_latest_minus_prev
    stats[:, 7] = max_forward_stride
    stats[:, 8] = max_forward_up_stride
    stats[:, 9] = torch.where(
      positive_height.any(dim=-1),
      positive_height_mean,
      torch.zeros_like(count),
    )
    return stats

  def _update_ratchet_state(
    self,
    pair_features: torch.Tensor,
    toe_features: torch.Tensor,
    new_footprint_any: torch.Tensor,
    new_toe_mark_any: torch.Tensor,
    step_dt: float,
  ) -> None:
    """Maintain a deployable lower-bound/probe target from ordered events."""
    aged = torch.where(
      self.ratchet_active,
      self.ratchet_age_s + float(step_dt),
      self.ratchet_age_s,
    )
    pair_count = min(pair_features.shape[1], FOOT_EVENT_PAIR_COUNT)
    if pair_count > 0:
      latest = pair_features[:, 0]
      latest_valid = latest[:, 0] > 0.5
      latest_score = latest[:, 1].clamp(0.0, 1.0)
      latest_stride = latest[:, 5]
      latest_height = latest[:, 6]
    else:
      latest_valid = torch.zeros(
        self.num_envs,
        dtype=torch.bool,
        device=self.device,
      )
      latest_score = torch.zeros(
        self.num_envs,
        dtype=torch.float32,
        device=self.device,
      )
      latest_stride = torch.zeros_like(latest_score)
      latest_height = torch.zeros_like(latest_score)

    new_footprint_any = new_footprint_any.to(device=self.device, dtype=torch.bool)
    new_toe_mark_any = new_toe_mark_any.to(device=self.device, dtype=torch.bool)
    forward_up_step = (
      new_footprint_any
      & latest_valid
      & (latest_stride >= self.ratchet_min_stride_m)
      & (latest_height > self.ratchet_height_threshold_m)
    )
    flat_step = (
      new_footprint_any
      & latest_valid
      & (torch.abs(latest_height) <= self.ratchet_flat_height_threshold_m)
    )

    toe_relation_valid = toe_features[:, 6] > 0.5
    toe_delta_s = toe_features[:, 7].clamp_min(0.0)
    toe_evidence = (
      (new_footprint_any | new_toe_mark_any)
      & toe_relation_valid
      & (toe_delta_s > 1.0e-5)
    )
    has_evidence = forward_up_step | toe_evidence

    stride_evidence = torch.where(
      forward_up_step,
      latest_stride.clamp_min(0.0),
      torch.zeros_like(latest_stride),
    )
    toe_evidence_value = torch.where(
      toe_evidence,
      toe_delta_s,
      torch.zeros_like(toe_delta_s),
    )
    evidence_lower = torch.maximum(stride_evidence, toe_evidence_value)
    old_lower = torch.where(
      self.ratchet_active,
      self.ratchet_lower_s,
      torch.zeros_like(self.ratchet_lower_s),
    )
    new_lower = torch.where(
      has_evidence,
      torch.maximum(old_lower, evidence_lower),
      self.ratchet_lower_s,
    )
    new_lower = new_lower.clamp(self.ratchet_min_stride_m, self.ratchet_max_stride_m)

    old_probe = torch.where(
      self.ratchet_active,
      self.ratchet_probe_target_s,
      torch.zeros_like(self.ratchet_probe_target_s),
    )
    grow_probe = torch.maximum(
      old_probe + self.ratchet_probe_increment_m,
      new_lower + self.ratchet_probe_increment_m,
    )
    confirm_probe = torch.maximum(old_probe, new_lower)
    new_probe = torch.where(
      forward_up_step & ~toe_evidence,
      grow_probe,
      torch.where(toe_evidence, confirm_probe, self.ratchet_probe_target_s),
    )
    new_probe = torch.maximum(new_probe, new_lower).clamp(
      self.ratchet_min_stride_m,
      self.ratchet_max_stride_m,
    )

    new_growth = torch.where(
      forward_up_step,
      latest_stride - self.ratchet_last_forward_up_stride,
      self.ratchet_stride_growth,
    )
    new_last_stride = torch.where(
      forward_up_step,
      latest_stride.clamp_min(0.0),
      self.ratchet_last_forward_up_stride,
    )
    new_no_hit_steps = torch.where(
      forward_up_step & ~toe_evidence,
      self.ratchet_safe_no_hit_steps + 1,
      self.ratchet_safe_no_hit_steps,
    )
    new_no_hit_steps = torch.where(
      toe_evidence,
      torch.zeros_like(new_no_hit_steps),
      new_no_hit_steps,
    )
    evidence_confidence = torch.maximum(
      torch.where(forward_up_step, latest_score, torch.zeros_like(latest_score)),
      torch.where(
        toe_evidence,
        toe_features[:, 1].clamp(0.0, 1.0),
        torch.zeros_like(latest_score),
      ),
    )
    new_confidence = torch.where(
      has_evidence,
      torch.maximum(self.ratchet_confidence, evidence_confidence),
      self.ratchet_confidence,
    )
    new_flat_steps = torch.where(
      flat_step & ~has_evidence,
      self.ratchet_flat_pair_steps + 1,
      self.ratchet_flat_pair_steps,
    )
    new_flat_steps = torch.where(
      has_evidence,
      torch.zeros_like(new_flat_steps),
      new_flat_steps,
    )

    stale = self.ratchet_active & (aged >= self.age_norm_s) & ~has_evidence
    flat_reset = self.ratchet_active & (new_flat_steps >= self.ratchet_reset_flat_pairs)
    reset = stale | flat_reset
    active = (self.ratchet_active | has_evidence) & ~reset
    age = torch.where(has_evidence, torch.zeros_like(aged), aged)
    age = torch.where(active, age, torch.zeros_like(age))

    self.ratchet_active = active
    self.ratchet_lower_s = torch.where(active, new_lower, torch.zeros_like(new_lower))
    self.ratchet_probe_target_s = torch.where(
      active,
      new_probe,
      torch.zeros_like(new_probe),
    )
    self.ratchet_last_forward_up_stride = torch.where(
      active,
      new_last_stride,
      torch.zeros_like(new_last_stride),
    )
    self.ratchet_stride_growth = torch.where(
      active,
      new_growth,
      torch.zeros_like(new_growth),
    )
    self.ratchet_safe_no_hit_steps = torch.where(
      active,
      new_no_hit_steps,
      torch.zeros_like(new_no_hit_steps),
    )
    self.ratchet_confidence = torch.where(
      active,
      new_confidence,
      torch.zeros_like(new_confidence),
    )
    self.ratchet_age_s = age
    self.ratchet_flat_pair_steps = torch.where(
      active,
      new_flat_steps,
      torch.zeros_like(new_flat_steps),
    )

  def _ratchet_features(self, toe_features: torch.Tensor) -> torch.Tensor:
    """Expose the step-to-step lower-bound ratchet as deployable summary cues."""
    features = torch.zeros(
      self.num_envs,
      FOOT_EVENT_RATCHET_DIM,
      dtype=torch.float32,
      device=self.device,
    )
    active = self.ratchet_active
    active_f = active.float()
    age_norm = (self.ratchet_age_s / self.age_norm_s).clamp(0.0, 1.0)
    toe_relation_valid = (toe_features[:, 6] > 0.5) & active
    toe_delta_s = torch.where(
      toe_relation_valid,
      toe_features[:, 7],
      torch.zeros_like(toe_features[:, 7]),
    )

    features[:, 0] = active_f
    features[:, 1] = self.ratchet_lower_s * active_f
    features[:, 2] = self.ratchet_probe_target_s * active_f
    features[:, 3] = self.ratchet_last_forward_up_stride * active_f
    features[:, 4] = self.ratchet_stride_growth * active_f
    features[:, 5] = (
      self.ratchet_safe_no_hit_steps.float() / float(max(FOOT_EVENT_PAIR_COUNT, 1))
    ).clamp(0.0, 1.0) * active_f
    features[:, 6] = toe_relation_valid.float()
    features[:, 7] = toe_delta_s
    features[:, 8] = self.ratchet_confidence.clamp(0.0, 1.0) * active_f
    features[:, 9] = torch.where(active, age_norm, torch.ones_like(age_norm))
    return features

  def _pair_features(
    self,
    newer: torch.Tensor,
    older: torch.Tensor,
    valid: torch.Tensor,
  ) -> torch.Tensor:
    pair_shape = newer.shape[:-1]
    features = torch.zeros(
      *pair_shape,
      FOOT_EVENT_PAIR_FEATURE_DIM,
      dtype=torch.float32,
      device=self.device,
    )
    newer_left = newer[..., 1] > 0.5
    newer_right = newer[..., 2] > 0.5
    older_left = older[..., 1] > 0.5
    alternating = newer_left != older_left
    both_confirmed = (newer[..., 3] > 0.5) & (older[..., 3] > 0.5)
    min_conf = torch.minimum(newer[..., 5], older[..., 5]).clamp(0.0, 1.0)
    age_max = torch.maximum(newer[..., 6], older[..., 6]).clamp(0.0, 1.0)
    delta_s = newer[..., 10] - older[..., 10]
    delta_z = newer[..., 9] - older[..., 9]
    lateral_delta = newer[..., 11] - older[..., 11]
    abs_delta_s = delta_s.abs()
    valid = valid.bool()
    source_score = torch.where(
      both_confirmed,
      torch.ones_like(min_conf),
      torch.full_like(min_conf, 0.55),
    )
    score = valid.float() * min_conf * (1.0 - age_max).clamp(0.0, 1.0) * source_score
    foot_sign = torch.where(
      newer_right,
      torch.ones_like(min_conf),
      torch.where(
        newer_left, torch.full_like(min_conf, -1.0), torch.zeros_like(min_conf)
      ),
    )

    features[..., 0] = valid.float()
    features[..., 1] = score
    features[..., 2] = torch.where(valid, foot_sign, torch.zeros_like(foot_sign))
    features[..., 3] = torch.where(
      valid, alternating.float(), torch.zeros_like(min_conf)
    )
    features[..., 4] = age_max
    features[..., 5] = torch.where(valid, delta_s, torch.zeros_like(delta_s))
    features[..., 6] = torch.where(valid, delta_z, torch.zeros_like(delta_z))
    features[..., 7] = torch.where(
      valid, lateral_delta, torch.zeros_like(lateral_delta)
    )
    features[..., 8] = torch.where(valid, abs_delta_s, torch.zeros_like(abs_delta_s))
    features[..., 9] = torch.where(valid, min_conf, torch.zeros_like(min_conf))
    return features

  def _latest_toe_features(self) -> torch.Tensor:
    """Return latest toe-hit cue.

    Layout:
    [0] toe valid, [1] confidence, [2] age, [3] toe foot sign,
    [4] toe_s, [5] toe_z, [6] opposite-after-toe valid,
    [7] signed horizontal distance from toe to later opposite footprint,
    [8] absolute horizontal distance for the same relation, [9] age gap.
    """
    toe_features = torch.zeros(
      self.num_envs, 10, dtype=torch.float32, device=self.device
    )
    latest_toe = self.toe_marks[:, 0]
    toe_valid = self.toe_valid[:, 0]
    toe_left = latest_toe[:, 1] > 0.5
    toe_right = latest_toe[:, 2] > 0.5
    toe_s = latest_toe[:, 9]

    opposite_fp, opposite_valid = self._first_footprint_after_toe(
      toe_left,
      toe_right,
      opposite=True,
    )
    toe_age = latest_toe[:, 5]

    toe_features[:, 0] = toe_valid.float()
    toe_features[:, 1] = torch.where(toe_valid, latest_toe[:, 4], 0.0)
    toe_features[:, 2] = torch.where(toe_valid, latest_toe[:, 5], 0.0)
    toe_features[:, 3] = torch.where(
      toe_valid & toe_right,
      torch.ones_like(toe_s),
      torch.where(toe_valid & toe_left, torch.full_like(toe_s, -1.0), 0.0),
    )
    toe_features[:, 4] = torch.where(toe_valid, toe_s, 0.0)
    toe_features[:, 5] = torch.where(toe_valid, latest_toe[:, 8], 0.0)
    relation_valid = toe_valid & opposite_valid
    relation_delta_s = opposite_fp[:, 10] - toe_s
    toe_features[:, 6] = torch.where(
      relation_valid,
      torch.ones_like(toe_s),
      torch.zeros_like(toe_s),
    )
    toe_features[:, 7] = torch.where(
      relation_valid,
      relation_delta_s,
      torch.zeros_like(toe_s),
    )
    toe_features[:, 8] = torch.where(
      relation_valid,
      relation_delta_s.abs(),
      torch.zeros_like(toe_s),
    )
    toe_features[:, 9] = torch.where(
      relation_valid,
      (toe_age - opposite_fp[:, 6]).clamp(0.0, 1.0),
      torch.zeros_like(toe_s),
    )
    return toe_features

  def _first_footprint_after_toe(
    self,
    toe_left: torch.Tensor,
    toe_right: torch.Tensor,
    *,
    opposite: bool,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    fp_left = self.footprints[..., 1] > 0.5
    fp_right = self.footprints[..., 2] > 0.5
    if opposite:
      target_foot = (toe_left[:, None] & fp_right) | (toe_right[:, None] & fp_left)
    else:
      target_foot = (toe_left[:, None] & fp_left) | (toe_right[:, None] & fp_right)
    toe_age = self.toe_marks[:, 0, 5]
    after_toe = self.footprints[..., 6] < (toe_age[:, None] - 1.0e-6)
    candidate = self.footprint_valid & target_foot & after_toe
    # Pick the first footprint after the toe mark in event time.  Age grows with
    # time since event, so this is the largest footprint age below toe_age.
    score = torch.where(
      candidate,
      self.footprints[..., 6],
      torch.full_like(self.footprints[..., 6], -1.0),
    )
    idx = torch.argmax(score, dim=-1)
    env_ids = torch.arange(self.num_envs, device=self.device)
    valid = torch.any(candidate, dim=-1)
    return self.footprints[env_ids, idx], valid

  def _log_summary_metrics(
    self,
    env: ManagerBasedRlEnv,
    summary: torch.Tensor,
    new_footprint_any: torch.Tensor,
    new_toe_mark_any: torch.Tensor,
  ) -> None:
    log = env.extras.get("log")
    if not isinstance(log, dict):
      return

    pair_features = summary[:, :FOOT_EVENT_PAIR_SUMMARY_DIM].reshape(
      self.num_envs,
      FOOT_EVENT_PAIR_COUNT,
      FOOT_EVENT_PAIR_FEATURE_DIM,
    )
    valid = pair_features[..., 0] > 0.5
    valid_f = valid.float()
    valid_count = valid_f.sum(dim=-1)
    valid_denom = valid_count.clamp_min(1.0)
    delta_s = pair_features[..., 5]
    delta_z = pair_features[..., 6]
    latest_valid = valid[:, 0]
    stats = summary[:, FOOT_EVENT_GEOMETRY_STATS_START:FOOT_EVENT_RATCHET_START]
    toe = summary[:, FOOT_EVENT_TOE_SUMMARY_START:FOOT_EVENT_GEOMETRY_STATS_START]
    ratchet = summary[:, FOOT_EVENT_RATCHET_START:]
    toe_relation_valid = toe[:, 6] > 0.5
    toe_relation_valid_f = toe_relation_valid.float()
    toe_relation_denom = toe_relation_valid_f.sum().clamp_min(1.0)
    ratchet_active = ratchet[:, 0] > 0.5
    ratchet_active_f = ratchet_active.float()
    ratchet_denom = ratchet_active_f.sum().clamp_min(1.0)

    forward = valid & (delta_s > 0.0)
    up = forward & (delta_z > 0.025)
    down = forward & (delta_z < -0.025)
    flat = valid & (torch.abs(delta_z) <= 0.025)

    log["Metrics/foot_event_summary_pair_valid_ratio"] = (
      valid_count / float(max(FOOT_EVENT_PAIR_COUNT, 1))
    ).mean()
    log["Metrics/foot_event_summary_latest_delta_s_mean"] = torch.where(
      latest_valid, delta_s[:, 0], torch.zeros_like(delta_s[:, 0])
    ).mean()
    log["Metrics/foot_event_summary_latest_delta_z_mean"] = torch.where(
      latest_valid, delta_z[:, 0], torch.zeros_like(delta_z[:, 0])
    ).mean()
    log["Metrics/foot_event_summary_mean_delta_s_mean"] = stats[:, 1].mean()
    log["Metrics/foot_event_summary_mean_delta_z_mean"] = stats[:, 2].mean()
    log["Metrics/foot_event_summary_max_forward_stride_mean"] = stats[:, 7].mean()
    log["Metrics/foot_event_summary_max_forward_up_stride_mean"] = stats[:, 8].mean()
    log["Metrics/foot_event_summary_positive_height_mean"] = stats[:, 9].mean()
    log["Metrics/foot_event_summary_toe_opposite_after_valid_ratio"] = (
      toe_relation_valid_f.mean()
    )
    log["Metrics/foot_event_summary_toe_opposite_after_delta_s_mean"] = (
      toe[:, 7] * toe_relation_valid_f
    ).sum() / toe_relation_denom
    log["Metrics/foot_event_summary_toe_opposite_after_abs_delta_s_mean"] = (
      toe[:, 8] * toe_relation_valid_f
    ).sum() / toe_relation_denom
    log["Metrics/foot_event_summary_toe_opposite_after_age_gap_mean"] = (
      toe[:, 9] * toe_relation_valid_f
    ).sum() / toe_relation_denom
    log["Metrics/foot_event_summary_up_pair_ratio"] = (
      up.float().sum(dim=-1) / valid_denom
    ).mean()
    log["Metrics/foot_event_summary_down_pair_ratio"] = (
      down.float().sum(dim=-1) / valid_denom
    ).mean()
    log["Metrics/foot_event_summary_flat_pair_ratio"] = (
      flat.float().sum(dim=-1) / valid_denom
    ).mean()
    log["Metrics/foot_event_summary_new_footprint_ratio"] = (
      new_footprint_any.float().mean()
    )
    log["Metrics/foot_event_summary_new_toe_mark_ratio"] = (
      new_toe_mark_any.float().mean()
    )
    log["Metrics/foot_event_ratchet_active_ratio"] = ratchet_active_f.mean()
    log["Metrics/foot_event_ratchet_lower_mean"] = (
      ratchet[:, 1] * ratchet_active_f
    ).sum() / ratchet_denom
    log["Metrics/foot_event_ratchet_probe_target_mean"] = (
      ratchet[:, 2] * ratchet_active_f
    ).sum() / ratchet_denom
    log["Metrics/foot_event_ratchet_last_forward_up_stride_mean"] = (
      ratchet[:, 3] * ratchet_active_f
    ).sum() / ratchet_denom
    log["Metrics/foot_event_ratchet_stride_growth_mean"] = (
      ratchet[:, 4] * ratchet_active_f
    ).sum() / ratchet_denom
    log["Metrics/foot_event_ratchet_step_count_mean"] = (
      ratchet[:, 5] * ratchet_active_f
    ).sum() / ratchet_denom
    log["Metrics/foot_event_ratchet_toe_confirm_ratio"] = ratchet[:, 6].mean()
    log["Metrics/foot_event_ratchet_toe_delta_s_mean"] = (
      ratchet[:, 7] * ratchet_active_f
    ).sum() / ratchet_denom
    log["Metrics/foot_event_ratchet_confidence_mean"] = (
      ratchet[:, 8] * ratchet_active_f
    ).sum() / ratchet_denom
    log["Metrics/foot_event_ratchet_age_mean"] = (
      ratchet[:, 9] * ratchet_active_f
    ).sum() / ratchet_denom

    riser_label = env.extras.get(STAIR_RISER_HEIGHT_LABEL_KEY)
    shape_valid = env.extras.get(STAIR_SHAPE_LABEL_VALID_KEY)
    if isinstance(riser_label, torch.Tensor) and isinstance(shape_valid, torch.Tensor):
      label = riser_label.to(device=self.device, dtype=torch.float32)
      estimate = stats[:, 9]
      mask = (estimate > 0.0) & shape_valid.to(device=self.device, dtype=torch.bool)
      mask_f = mask.float()
      mask_denom = mask_f.sum().clamp_min(1.0)
      log["Metrics/foot_event_summary_positive_height_to_riser_mae"] = (
        torch.abs(estimate - label) * mask_f
      ).sum() / mask_denom

    safe_lower = env.extras.get(MINIMUM_SAFE_STRIDE_KEY)
    safe_valid = env.extras.get(MINIMUM_SAFE_STRIDE_VALID_KEY)
    if isinstance(safe_lower, torch.Tensor) and isinstance(safe_valid, torch.Tensor):
      lower_label = safe_lower.to(device=self.device, dtype=torch.float32)
      lower_est = stats[:, 8]
      lower_mask = (lower_est > 0.0) & safe_valid.to(
        device=self.device, dtype=torch.bool
      )
      lower_mask_f = lower_mask.float()
      lower_denom = lower_mask_f.sum().clamp_min(1.0)
      log["Metrics/foot_event_summary_forward_up_stride_to_safe_lower_mae"] = (
        torch.abs(lower_est - lower_label) * lower_mask_f
      ).sum() / lower_denom
      log["Metrics/foot_event_summary_forward_up_stride_below_lower_ratio"] = (
        (lower_est < lower_label).float() * lower_mask_f
      ).sum() / lower_denom
      ratchet_hint = ratchet[:, 2]
      ratchet_mask = ratchet_active & safe_valid.to(
        device=self.device, dtype=torch.bool
      )
      ratchet_mask_f = ratchet_mask.float()
      ratchet_label_denom = ratchet_mask_f.sum().clamp_min(1.0)
      log["Metrics/foot_event_ratchet_hint_to_safe_lower_mae"] = (
        torch.abs(ratchet_hint - lower_label) * ratchet_mask_f
      ).sum() / ratchet_label_denom
      log["Metrics/foot_event_ratchet_hint_below_lower_ratio"] = (
        (ratchet_hint < lower_label).float() * ratchet_mask_f
      ).sum() / ratchet_label_denom

  def _env_ids(self, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    all_ids = torch.arange(self.num_envs, device=self.device)
    if env_ids is None:
      return all_ids
    if isinstance(env_ids, slice):
      return all_ids[env_ids]
    return env_ids.to(device=self.device, dtype=torch.long)

  def _age_and_refresh(
    self,
    step_dt: float,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
  ) -> None:
    footprint_age_delta = step_dt / self.age_norm_s
    toe_age_delta = step_dt / self.age_norm_s
    self.footprints[..., 6] = torch.where(
      self.footprint_valid,
      (self.footprints[..., 6] + footprint_age_delta).clamp(0.0, 1.0),
      self.footprints[..., 6],
    )
    self.toe_marks[..., 5] = torch.where(
      self.toe_valid,
      (self.toe_marks[..., 5] + toe_age_delta).clamp(0.0, 1.0),
      self.toe_marks[..., 5],
    )
    self._expire_old_slots()
    self._refresh_relative_positions(root_pos_w, root_quat_w)

  def _expire_old_slots(self) -> None:
    expired_foot = self.footprint_valid & (self.footprints[..., 6] >= 1.0)
    self.footprints = torch.where(
      expired_foot[..., None],
      torch.zeros_like(self.footprints),
      self.footprints,
    )
    self.footprint_pos_w = torch.where(
      expired_foot[..., None],
      torch.zeros_like(self.footprint_pos_w),
      self.footprint_pos_w,
    )
    self.footprint_valid = self.footprint_valid & ~expired_foot
    self._clear_expired_active_slots()

    expired_toe = self.toe_valid & (self.toe_marks[..., 5] >= 1.0)
    self.toe_marks = torch.where(
      expired_toe[..., None],
      torch.zeros_like(self.toe_marks),
      self.toe_marks,
    )
    self.toe_pos_w = torch.where(
      expired_toe[..., None],
      torch.zeros_like(self.toe_pos_w),
      self.toe_pos_w,
    )
    self.toe_valid = self.toe_valid & ~expired_toe

  def _clear_expired_active_slots(self) -> None:
    slot = self.active_footprint_slot.clamp(0, self.memory_len - 1)
    valid_at_slot = torch.gather(self.footprint_valid, dim=1, index=slot)
    active_and_valid = (self.active_footprint_slot >= 0) & valid_at_slot
    self.active_footprint_slot = torch.where(
      active_and_valid,
      self.active_footprint_slot,
      torch.full_like(self.active_footprint_slot, -1),
    )

  def _refresh_relative_positions(
    self,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
  ) -> None:
    root_pos = root_pos_w[:, None, :].expand(-1, self.memory_len, -1)
    root_quat = root_quat_w[:, None, :].expand(-1, self.memory_len, -1)
    footprint_rel = _world_to_current_base_yaw(
      self.footprint_pos_w,
      root_pos,
      root_quat,
    )
    toe_rel = _world_to_current_base_yaw(self.toe_pos_w, root_pos, root_quat)
    self.footprints[..., 7:10] = torch.where(
      self.footprint_valid[..., None],
      footprint_rel,
      torch.zeros_like(footprint_rel),
    )
    self.footprints[..., 10] = torch.where(
      self.footprint_valid,
      footprint_rel[..., 0],
      torch.zeros_like(footprint_rel[..., 0]),
    )
    self.footprints[..., 11] = torch.where(
      self.footprint_valid,
      footprint_rel[..., 1],
      torch.zeros_like(footprint_rel[..., 1]),
    )
    self.toe_marks[..., 6:9] = torch.where(
      self.toe_valid[..., None],
      toe_rel,
      torch.zeros_like(toe_rel),
    )
    self.toe_marks[..., 9] = torch.where(
      self.toe_valid,
      toe_rel[..., 0],
      torch.zeros_like(toe_rel[..., 0]),
    )
    self.toe_marks[..., 10] = torch.where(
      self.toe_valid,
      toe_rel[..., 1],
      torch.zeros_like(toe_rel[..., 1]),
    )

  def _update_stance_latches(
    self,
    contact_prob: torch.Tensor,
    ground_contact: torch.Tensor,
    step_dt: float,
  ) -> None:
    released_prob = contact_prob < self.release_contact_prob_threshold
    self.release_count = torch.where(
      released_prob,
      self.release_count + 1,
      torch.zeros_like(self.release_count),
    )
    release = self.foot_in_stance & (self.release_count >= self.release_confirm_frames)
    self.foot_in_stance = torch.where(
      release,
      torch.zeros_like(self.foot_in_stance),
      self.foot_in_stance,
    )
    self.active_footprint_slot = torch.where(
      release,
      torch.full_like(self.active_footprint_slot, -1),
      self.active_footprint_slot,
    )

    del ground_contact
    self.stance_age_s = torch.where(
      self.foot_in_stance,
      self.stance_age_s + float(step_dt),
      self.stance_age_s,
    )
    slot_ids = torch.arange(self.memory_len, device=self.device).view(1, -1, 1)
    active_slots = self.active_footprint_slot[:, None, :]
    active = (active_slots >= 0) & self.foot_in_stance[:, None, :]
    slot_match = active & (slot_ids == active_slots)
    stance_age = (self.stance_age_s / self.stance_age_norm_s).clamp(0.0, 1.0)
    stance_age_by_slot = (slot_match.float() * stance_age[:, None, :]).amax(dim=-1)
    self.footprints[..., 16] = torch.where(
      slot_match.any(dim=-1),
      stance_age_by_slot,
      self.footprints[..., 16],
    )

  def _foot_and_toe_points(
    self,
    env: ManagerBasedRlEnv,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    left_toe, right_toe, left_heel, right_heel = _body_frame_foot_positions(env)
    foot_points = torch.stack(
      [
        0.5 * (left_toe + left_heel),
        0.5 * (right_toe + right_heel),
      ],
      dim=1,
    )
    toe_points = torch.stack([left_toe, right_toe], dim=1)
    return foot_points, toe_points

  def _push_footprint(
    self,
    *,
    mask: torch.Tensor,
    foot_id: int,
    point_body: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    phase_now: torch.Tensor,
    contact_prob: torch.Tensor,
    touchdown_prob: torch.Tensor,
    confidence: torch.Tensor,
    source_predicted_fill: bool | torch.Tensor,
  ) -> None:
    mask = mask.bool()
    point_body = point_body + self._position_noise(
      point_body.shape,
      self.footprint_xy_noise_range_m,
      self.footprint_z_noise_range_m,
    )
    point_w = root_pos_w + quat_apply(root_quat_w, point_body)
    point_rel = _world_to_current_base_yaw(point_w, root_pos_w, root_quat_w)
    self._shift_footprints(mask)

    feature = torch.zeros(
      self.num_envs,
      FOOTPRINT_SLOT_DIM,
      dtype=torch.float32,
      device=self.device,
    )
    feature[:, 0] = 1.0
    feature[:, 1] = 1.0 if foot_id == 0 else 0.0
    feature[:, 2] = 1.0 if foot_id == 1 else 0.0
    if isinstance(source_predicted_fill, torch.Tensor):
      predicted_source = source_predicted_fill.to(device=self.device).float()
    else:
      predicted_source = torch.full(
        (self.num_envs,),
        1.0 if source_predicted_fill else 0.0,
        dtype=torch.float32,
        device=self.device,
      )
    feature[:, 3] = 1.0 - predicted_source
    feature[:, 4] = predicted_source
    feature[:, 5] = confidence.clamp(0.0, 1.0)
    feature[:, 6] = 0.0
    feature[:, 7:10] = point_rel
    feature[:, 10] = point_rel[:, 0]
    feature[:, 11] = point_rel[:, 1]
    feature[:, 12:14] = phase_now
    feature[:, 14] = contact_prob.clamp(0.0, 1.0)
    feature[:, 15] = touchdown_prob.clamp(0.0, 1.0)
    feature[:, 16] = 0.0

    self.footprints[:, 0] = torch.where(
      mask[:, None],
      feature,
      self.footprints[:, 0],
    )
    self.footprint_valid[:, 0] = self.footprint_valid[:, 0] | mask
    self.footprint_pos_w[:, 0] = torch.where(
      mask[:, None],
      point_w,
      self.footprint_pos_w[:, 0],
    )
    self.foot_in_stance[:, foot_id] = self.foot_in_stance[:, foot_id] | mask
    self.release_count[:, foot_id] = torch.where(
      mask,
      torch.zeros_like(self.release_count[:, foot_id]),
      self.release_count[:, foot_id],
    )
    self.stance_age_s[:, foot_id] = torch.where(
      mask,
      torch.zeros_like(self.stance_age_s[:, foot_id]),
      self.stance_age_s[:, foot_id],
    )
    self.active_footprint_slot[:, foot_id] = torch.where(
      mask,
      torch.zeros_like(self.active_footprint_slot[:, foot_id]),
      self.active_footprint_slot[:, foot_id],
    )

  def _push_toe_mark(
    self,
    *,
    mask: torch.Tensor,
    foot_id: int,
    point_body: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    phase_now: torch.Tensor,
    contact_prob: torch.Tensor,
    toe_prob: torch.Tensor,
    confidence: torch.Tensor,
    swing_or_early_contact: torch.Tensor,
    false_positive: bool | torch.Tensor,
  ) -> None:
    del false_positive
    mask = mask.bool()
    point_body = point_body + self._position_noise(
      point_body.shape,
      self.toe_xy_noise_range_m,
      self.toe_z_noise_range_m,
    )
    point_w = root_pos_w + quat_apply(root_quat_w, point_body)
    point_rel = _world_to_current_base_yaw(point_w, root_pos_w, root_quat_w)
    self._shift_toe_marks(mask)

    feature = torch.zeros(
      self.num_envs,
      TOE_MARK_SLOT_DIM,
      dtype=torch.float32,
      device=self.device,
    )
    feature[:, 0] = 1.0
    feature[:, 1] = 1.0 if foot_id == 0 else 0.0
    feature[:, 2] = 1.0 if foot_id == 1 else 0.0
    feature[:, 3] = 1.0
    feature[:, 4] = confidence.clamp(0.0, 1.0)
    feature[:, 5] = 0.0
    feature[:, 6:9] = point_rel
    feature[:, 9] = point_rel[:, 0]
    feature[:, 10] = point_rel[:, 1]
    feature[:, 11:13] = phase_now
    feature[:, 13] = toe_prob.clamp(0.0, 1.0)
    feature[:, 14] = contact_prob.clamp(0.0, 1.0)
    feature[:, 15] = swing_or_early_contact.float()

    self.toe_marks[:, 0] = torch.where(
      mask[:, None],
      feature,
      self.toe_marks[:, 0],
    )
    self.toe_valid[:, 0] = self.toe_valid[:, 0] | mask
    self.toe_pos_w[:, 0] = torch.where(
      mask[:, None],
      point_w,
      self.toe_pos_w[:, 0],
    )

  def _shift_footprints(self, mask: torch.Tensor) -> None:
    shifted_footprints = torch.cat(
      [self.footprints[:, :1], self.footprints[:, :-1]], dim=1
    )
    shifted_valid = torch.cat(
      [self.footprint_valid[:, :1], self.footprint_valid[:, :-1]],
      dim=1,
    )
    shifted_pos = torch.cat(
      [self.footprint_pos_w[:, :1], self.footprint_pos_w[:, :-1]],
      dim=1,
    )
    self.footprints = torch.where(
      mask[:, None, None], shifted_footprints, self.footprints
    )
    self.footprint_valid = torch.where(
      mask[:, None], shifted_valid, self.footprint_valid
    )
    self.footprint_pos_w = torch.where(
      mask[:, None, None], shifted_pos, self.footprint_pos_w
    )

    active = self.active_footprint_slot
    active = torch.where(active >= 0, active + 1, active)
    active = torch.where(
      active < self.memory_len,
      active,
      torch.full_like(active, -1),
    )
    self.active_footprint_slot = torch.where(
      mask[:, None],
      active,
      self.active_footprint_slot,
    )

  def _shift_toe_marks(self, mask: torch.Tensor) -> None:
    shifted_marks = torch.cat([self.toe_marks[:, :1], self.toe_marks[:, :-1]], dim=1)
    shifted_valid = torch.cat([self.toe_valid[:, :1], self.toe_valid[:, :-1]], dim=1)
    shifted_pos = torch.cat([self.toe_pos_w[:, :1], self.toe_pos_w[:, :-1]], dim=1)
    self.toe_marks = torch.where(mask[:, None, None], shifted_marks, self.toe_marks)
    self.toe_valid = torch.where(mask[:, None], shifted_valid, self.toe_valid)
    self.toe_pos_w = torch.where(mask[:, None, None], shifted_pos, self.toe_pos_w)

  def _contact_prob(self, ground_contact: torch.Tensor) -> torch.Tensor:
    true_prob = self._uniform(
      ground_contact.shape,
      self.contact_true_prob_range,
      default=1.0,
    )
    false_prob = self._uniform(
      ground_contact.shape,
      self.contact_false_prob_range,
      default=0.0,
    )
    return torch.where(ground_contact, true_prob, false_prob)

  def _bernoulli(self, probability: float) -> torch.Tensor:
    if not self.noise_enabled or probability <= 0.0:
      return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    if probability >= 1.0:
      return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    return torch.rand(self.num_envs, device=self.device) < probability

  def _uniform_like(
    self,
    reference: torch.Tensor,
    value_range: tuple[float, float],
    default: float,
  ) -> torch.Tensor:
    return self._uniform(reference.shape, value_range, default=default)

  def _uniform(
    self,
    shape: torch.Size | tuple[int, ...],
    value_range: tuple[float, float],
    default: float,
  ) -> torch.Tensor:
    if not self.noise_enabled:
      return torch.full(shape, default, dtype=torch.float32, device=self.device)
    low, high = float(value_range[0]), float(value_range[1])
    if high <= low:
      return torch.full(shape, low, dtype=torch.float32, device=self.device)
    return (
      torch.rand(shape, dtype=torch.float32, device=self.device) * (high - low) + low
    )

  def _position_noise(
    self,
    shape: torch.Size,
    xy_range: tuple[float, float],
    z_range: tuple[float, float],
  ) -> torch.Tensor:
    if not self.noise_enabled:
      return torch.zeros(shape, dtype=torch.float32, device=self.device)
    xy_mag = self._uniform(shape[:-1], xy_range, default=0.0)
    z_mag = self._uniform(shape[:-1], z_range, default=0.0)
    sign_xy = torch.where(
      torch.rand((*shape[:-1], 2), device=self.device) < 0.5,
      -1.0,
      1.0,
    )
    sign_z = torch.where(
      torch.rand(shape[:-1], device=self.device) < 0.5,
      -1.0,
      1.0,
    )
    noise = torch.zeros(shape, dtype=torch.float32, device=self.device)
    noise[..., 0:2] = sign_xy * xy_mag[..., None]
    noise[..., 2] = sign_z * z_mag
    return noise


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
