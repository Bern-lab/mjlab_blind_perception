"""Gated slow-latent Unitree G1 blind target-navigation task config."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp.teacher_target_heading_command import (
  TeacherTargetHeadingVelocityCommandCfg,
)
from mjlab.terrains import StepDangerVisualizationCfg

from .blind_rough_teacher_kl_env_cfg import unitree_g1_blind_rough_teacherkl_env_cfg
from .blind_rough_toe_contact_cfg import TOE_TERRAIN_CONTACT_SENSOR, g1_foot_body_cfg
from .env_cfgs import (
  G1_HIGH_STAIRS_MIXED_REPLAY_LEVEL_RANGES,
  G1_HIGH_STAIRS_MIXED_REPLAY_START_LEVEL,
  G1_HIGH_STAIRS_MIXED_REPLAY_WEIGHTS,
  configure_g1_high_stairs_mixed_replay,
)


@dataclass(frozen=True)
class G1SlowLatentTargetCommandParams:
  """Target-heading command parameters exposed for slow-latent experiments."""

  resampling_time_range: tuple[float, float] = (60.0, 60.0)
  heading_control_stiffness: float = 0.5
  rel_target_envs: float = 0.8
  rel_random_heading_envs: float = 0.0
  rel_standing_envs: float = 0.2
  target_reached_threshold: float = 0.5
  target_min_distance: float = 1.0
  target_max_distance: float = 12.0
  target_tile_radius: int = 1
  include_current_tile: bool = False
  zero_lateral_velocity: bool = True
  lin_vel_x: tuple[float, float] = (0.4, 1.0)
  lin_vel_y: tuple[float, float] = (0.0, 0.0)
  ang_vel_z: tuple[float, float] = (-0.8, 0.8)
  play_rel_target_envs: float = 1.0
  play_rel_random_heading_envs: float = 0.0
  play_rel_standing_envs: float = 0.0
  play_heading_control_stiffness: float = 0.8
  play_lin_vel_x: tuple[float, float] = (0.6, 0.9)
  play_lin_vel_y: tuple[float, float] = (0.0, 0.0)
  play_ang_vel_z: tuple[float, float] = (-0.7, 0.7)
  play_target_min_distance: float = 1.0
  play_target_max_distance: float = 10.0
  play_target_tile_radius: int = 1


@dataclass(frozen=True)
class G1SlowLatentRewardParams:
  """All reward weights and key parameters for the slow-latent task."""

  # Velocity and posture tracking.
  track_linear_velocity_weight: float = 2.0
  track_linear_velocity_std: float = 0.5
  track_angular_velocity_weight: float = 2.0
  track_angular_velocity_std: float = 0.7071067811865476
  upright_weight: float = 1.0
  upright_std: float = 0.4472135954999579
  pose_weight: float = 1.0
  pose_walking_threshold: float = 0.05
  pose_running_threshold: float = 1.5
  pose_std_standing: dict[str, float] = field(
    default_factory=lambda: {
      ".*": 0.05,
    }
  )
  pose_std_walking: dict[str, float] = field(
    default_factory=lambda: {
      r".*hip_pitch.*": 0.50,
      r".*hip_roll.*": 0.18,
      r".*hip_yaw.*": 0.18,
      r".*knee.*": 0.35,
      r".*ankle_pitch.*": 0.20,
      r".*ankle_roll.*": 0.1,
      r".*waist_yaw.*": 0.2,
      r".*waist_roll.*": 0.08,
      r".*waist_pitch.*": 0.1,
      r".*shoulder_pitch.*": 0.15,
      r".*shoulder_roll.*": 0.15,
      r".*shoulder_yaw.*": 0.1,
      r".*elbow.*": 0.15,
      r".*wrist.*": 0.3,
    }
  )
  pose_std_running: dict[str, float] = field(
    default_factory=lambda: {
      r".*hip_pitch.*": 0.65,
      r".*hip_roll.*": 0.25,
      r".*hip_yaw.*": 0.25,
      r".*knee.*": 0.45,
      r".*ankle_pitch.*": 0.35,
      r".*ankle_roll.*": 0.15,
      r".*waist_yaw.*": 0.3,
      r".*waist_roll.*": 0.08,
      r".*waist_pitch.*": 0.2,
      r".*shoulder_pitch.*": 0.5,
      r".*shoulder_roll.*": 0.2,
      r".*shoulder_yaw.*": 0.15,
      r".*elbow.*": 0.35,
      r".*wrist.*": 0.3,
    }
  )

  # Foot placement, gait, and stance-shape rewards.
  foot_clearance_weight: float = -2.0
  foot_clearance_min_height: float = 0.10
  foot_clearance_max_height: float = 0.25
  foot_clearance_command_threshold: float = 0.0
  foot_swing_height_weight: float = -1.0  # 0.25
  foot_swing_target_height: float = 0.10
  foot_swing_command_threshold: float = 0.01
  foot_slip_weight: float = -0.2
  foot_slip_command_threshold: float = 0.05
  soft_landing_weight: float = -2.0e-5
  soft_landing_command_threshold: float = 0.05
  idle_penalty_weight: float = -2.0
  idle_command_threshold: float = 0.2
  idle_velocity_threshold: float = 0.1
  foot_gait_weight: float = 0.5
  foot_gait_period: float = 0.6
  foot_gait_offset: tuple[float, float] = (0.0, 0.5)
  foot_gait_threshold: float = 0.56
  foot_gait_command_threshold: float = 0.1
  base_height_above_support_weight: float = -0.5
  base_height_above_support_min_height: float = 0.74
  base_height_above_support_error_scale: float = 10.0

  # Safety and smoothness regularizers.
  body_ang_vel_weight: float = -0.08
  angular_momentum_weight: float = -0.03
  joint_pos_limits_weight: float = -1.0
  action_rate_l2_weight: float = -0.15
  self_collisions_weight: float = -1.0
  self_collision_force_threshold: float = 10.0
  joint_acc_l2_weight: float = -2.5e-7
  action_acc_l2_weight: float = -0.05

  # Target-navigation rewards.
  target_progress_weight: float = 0.8
  target_progress_min_distance: float = 0.05
  target_reached_bonus_weight: float = 0.4

  # Step-boundary danger-zone rewards.
  foot_lip_weight: float = -3.2
  foot_lip_edge_radius: float = 0.07
  foot_lip_edge_height_band: float = 0.06
  foot_lip_support_speed_floor: float = 0.08
  foot_lip_ignore_boundary_layers: int = 0

  toe_slab_weight: float = -4.2
  toe_slab_depth: float = 0.10
  toe_slab_u_margin: float = 0.02
  toe_slab_v_margin: float = 0.05
  toe_x_min: float = 0.08
  toe_v_threshold: float = 0.02
  surface_tol: float = 0.005
  nearest_boundaries: int = 4
  min_terrain_level: int = 3

  toe_contact_penalty_scale: float = 0.5
  toe_contact_time_scale: float = 0.20
  toe_contact_force_threshold: float = 15.0
  toe_contact_force_scale: float = 60.0
  toe_contact_vertical_normal_z_max: float = 0.4
  toe_event_min_forward_intent_speed: float = 0.04
  toe_event_min_forward_speed_drop: float = 0.04
  toe_event_blocked_forward_speed: float = 0.03
  toe_event_persistent_steps: int = 3

  # Penalty-only stair entry and safe layer-2 touchdown confirmation.
  toe_stair_entry_cooldown_time: float = 0.20
  toe_stair_heading_cos: float = 0.70
  toe_stair_touchdown_height_tolerance: float = 0.08
  toe_stair_touchdown_lateral_margin: float = 0.03
  toe_stair_min_safe_stride: float = 0.10
  toe_stair_max_safe_stride: float = 0.55
  toe_stair_max_tracking_stride: float = 0.80
  toe_stair_touchdown_lip_clearance: float = 0.02
  toe_stair_touchdown_lip_height_band: float = 0.06
  toe_safe_stride_containment_margin: float = 0.003
  stair_entry_evidence_time: float = 0.80
  safe_stride_evidence_window_steps: int = 15
  safe_stride_rear_partial_weight: float = 2.0
  safe_stride_riser_weight: float = 3.0
  toe_collision_risk_margin: float = 0.06
  toe_stair_command_threshold: float = 0.05
  stair_skip_layer_weight: float = -1.0

  # Target-foot dense shaping toward the expected tread-depth midline.  This
  # only activates once the swing foot is above the target tread height.
  target_tread_midline_weight: float = 0.4
  target_tread_midline_height_clearance: float = 0.02
  target_tread_midline_sigma_fraction: float = 0.60
  target_tread_midline_min_sigma: float = 0.20
  target_tread_midline_edge_margin: float = 0.08
  target_tread_midline_progress_scale: float = 0.30
  target_tread_midline_center_scale: float = 0.70
  target_tread_midline_edge_scale: float = 0.12
  target_tread_midline_max_progress_step: float = 0.20

  # Per-leg shank collision-capsule clearance from the next semantic stair edge.
  shank_edge_weight: float = -3.0
  shank_edge_clearance_margin: float = 0.06
  shank_edge_min_riser_height: float = 0.04
  shank_edge_direction_cos_threshold: float = 0.85
  shank_edge_lateral_margin: float = 0.05
  shank_capsule_start_local: tuple[float, float, float] = (0.01, 0.0, 0.0)
  shank_capsule_end_local: tuple[float, float, float] = (0.01, 0.0, -0.15)
  shank_capsule_radius: float = 0.045

  # Disabled following-step privileged landing shaping.  The lip-volume term
  # already carries the edge/partial-foot signal, while this touchdown-only
  # reward was too sparse to help the current SlowLatent run.
  stair_tread_landing_weight: float = 0.5
  stair_tread_landing_lead: float = 0.04
  stair_tread_landing_sigma_fraction: float = 0.15
  stair_tread_landing_min_sigma: float = 0.03
  stair_tread_landing_back_margin: float = 0.06
  stair_tread_landing_front_margin: float = 0.08
  stair_tread_landing_lateral_margin: float = 0.03
  stair_tread_landing_height_tolerance: float = 0.08
  stair_tread_landing_support_deficit_scale: float = 0.40
  stair_tread_landing_min_layer: int = 3
  stair_tread_landing_lip_margin: float = 0.01


@dataclass(frozen=True)
class G1SlowLatentPlayVisualizationParams:
  """Play-only visualization knobs for the slow-latent main experiment."""

  show_depth_camera_visualizers: bool = False
  """Show depth camera frustums, ground projections, and Viser camera feeds."""
  show_raycast_debug_visualizers: bool = False
  """Show terrain_scan / foot_height_scan raycast debug markers on the ground."""
  show_step_danger_zones: bool = True
  """Show non-colliding MuJoCo geoms for step lip/riser danger zones."""
  danger_lip_radius: float | None = None
  """Lip tube radius. None reuses rewards.foot_lip_edge_radius."""
  danger_slab_depth: float | None = None
  """Riser slab depth toward the low side. None reuses rewards.toe_slab_depth."""
  danger_slab_u_margin: float | None = None
  """Extra slab margin along the step edge. None reuses rewards.toe_slab_u_margin."""
  danger_slab_v_margin: float | None = None
  """Extra slab vertical/normal margin. None reuses rewards.toe_slab_v_margin."""
  danger_geom_group: int = 4
  """MuJoCo geom group used for danger-zone visual geoms."""


@dataclass(frozen=True)
class G1SlowLatentTerrainReplayParams:
  """Mixed stair-level replay parameters for late curriculum training."""

  start_level: int | None = G1_HIGH_STAIRS_MIXED_REPLAY_START_LEVEL
  """First terrain level that permanently activates mixed replay for an env."""
  level_ranges: tuple[tuple[int, int], ...] = G1_HIGH_STAIRS_MIXED_REPLAY_LEVEL_RANGES
  """Inclusive low/mid/high terrain-level buckets sampled during replay."""
  weights: tuple[float, ...] = G1_HIGH_STAIRS_MIXED_REPLAY_WEIGHTS
  """Replay bucket weights, e.g. low/mid/high = 0.2/0.3/0.5."""


@dataclass(frozen=True)
class G1SlowLatentEnvParams:
  """Top-level knobs for the slow-latent environment config."""

  actor_history_length: int = 5
  """History length for the deployable blind actor observation group."""
  latent_group_name: str = "latent"
  """Observation group consumed by the slow-latent encoder."""
  latent_obs_term_name: str = "stair_latent"
  """Observation term name for stair-focused latent encoder features."""
  foot_event_memory_term_name: str = "foot_event_memory"
  """Observation term name for footprint/toe-mark event memory features."""
  enable_foot_event_memory_obs: bool = True
  """Append deployable-shaped footprint/toe-mark memory to latent obs."""
  foot_event_memory_length: int = 6
  """Number of recent footprint and toe-mark slots kept per env."""
  foot_event_memory_age_norm_s: float = 1.5
  """Age horizon after which event-memory slots are invalidated."""
  label_group_name: str = "latent_labels"
  """Observation group used by auxiliary slow-latent losses during training."""
  enable_latent_labels: bool = True
  """Enable simulation-only labels for event/stair auxiliary losses."""
  enable_latent_obs_corruption: bool = False
  """Apply observation corruption to latent encoder inputs."""
  include_gait_phase_in_latent: bool = True
  """Append the deployable gait phase after the base 91 stair-latent channels."""
  target_command: G1SlowLatentTargetCommandParams = field(
    default_factory=G1SlowLatentTargetCommandParams
  )
  """Target-heading command distribution and play overrides."""
  rewards: G1SlowLatentRewardParams = field(default_factory=G1SlowLatentRewardParams)
  """All slow-latent reward weights and key reward parameters."""
  terrain_replay: G1SlowLatentTerrainReplayParams = field(
    default_factory=G1SlowLatentTerrainReplayParams
  )
  """Late-curriculum mixed terrain replay schedule."""
  play_visualization: G1SlowLatentPlayVisualizationParams = field(
    default_factory=G1SlowLatentPlayVisualizationParams
  )
  """Play-only viewer/debug visualization settings."""


def configure_g1_step_danger_rewards(
  cfg: ManagerBasedRlEnvCfg,
  params: G1SlowLatentRewardParams | None = None,
) -> None:
  """Apply the full SlowLatent step-boundary danger reward set."""
  params = params or G1SlowLatentRewardParams()

  def foot_asset_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
      "robot",
      body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
    )

  def shank_asset_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
      "robot",
      body_names=("left_knee_link", "right_knee_link"),
      preserve_order=True,
    )

  cfg.rewards["foot_step_lip_volume_penalty"] = RewardTermCfg(
    func=mdp.foot_step_lip_volume_penalty,
    weight=params.foot_lip_weight,
    params={
      "edge_radius": params.foot_lip_edge_radius,
      "edge_height_band": params.foot_lip_edge_height_band,
      "support_speed_floor": params.foot_lip_support_speed_floor,
      "ignore_boundary_layers": params.foot_lip_ignore_boundary_layers,
      "nearest_boundaries": params.nearest_boundaries,
      "contact_sensor_name": "feet_ground_contact",
      "min_terrain_level": params.min_terrain_level,
      "asset_cfg": foot_asset_cfg(),
    },
  )
  cfg.rewards["toe_step_riser_slab_penalty"] = RewardTermCfg(
    func=mdp.toe_step_riser_slab_penalty,
    weight=params.toe_slab_weight,
    params={
      "slab_depth": params.toe_slab_depth,
      "u_margin": params.toe_slab_u_margin,
      "v_margin": params.toe_slab_v_margin,
      "toe_x_min": params.toe_x_min,
      "toe_v_threshold": params.toe_v_threshold,
      "surface_tol": params.surface_tol,
      "nearest_boundaries": params.nearest_boundaries,
      "min_terrain_level": params.min_terrain_level,
      "contact_sensor_name": TOE_TERRAIN_CONTACT_SENSOR,
      "contact_penalty_scale": params.toe_contact_penalty_scale,
      "contact_time_scale": params.toe_contact_time_scale,
      "contact_force_threshold": params.toe_contact_force_threshold,
      "contact_force_scale": params.toe_contact_force_scale,
      "contact_vertical_normal_z_max": params.toe_contact_vertical_normal_z_max,
      "event_min_forward_intent_speed": (params.toe_event_min_forward_intent_speed),
      "event_min_forward_speed_drop": params.toe_event_min_forward_speed_drop,
      "event_blocked_forward_speed": params.toe_event_blocked_forward_speed,
      "event_persistent_steps": params.toe_event_persistent_steps,
      "stair_entry_cooldown_time": params.toe_stair_entry_cooldown_time,
      "stair_heading_cos": params.toe_stair_heading_cos,
      "stair_touchdown_height_tolerance": (params.toe_stair_touchdown_height_tolerance),
      "stair_touchdown_lateral_margin": (params.toe_stair_touchdown_lateral_margin),
      "stair_min_safe_stride": params.toe_stair_min_safe_stride,
      "stair_max_safe_stride": params.toe_stair_max_safe_stride,
      "stair_max_tracking_stride": params.toe_stair_max_tracking_stride,
      "stair_touchdown_lip_clearance": (params.toe_stair_touchdown_lip_clearance),
      "stair_touchdown_lip_height_band": (params.toe_stair_touchdown_lip_height_band),
      "safe_stride_containment_margin": (params.toe_safe_stride_containment_margin),
      "stair_entry_evidence_time": params.stair_entry_evidence_time,
      "stair_attempt_period": params.foot_gait_period,
      "safe_stride_evidence_window_steps": (params.safe_stride_evidence_window_steps),
      "safe_stride_rear_partial_weight": (params.safe_stride_rear_partial_weight),
      "safe_stride_riser_weight": params.safe_stride_riser_weight,
      "collision_risk_margin": params.toe_collision_risk_margin,
      "command_threshold": params.toe_stair_command_threshold,
      "ground_contact_sensor_name": "feet_ground_contact",
      "asset_cfg": foot_asset_cfg(),
    },
  )
  cfg.rewards["shank_front_edge_clearance_penalty"] = RewardTermCfg(
    func=mdp.shank_front_edge_clearance_penalty,
    weight=params.shank_edge_weight,
    params={
      "clearance_margin": params.shank_edge_clearance_margin,
      "min_riser_height": params.shank_edge_min_riser_height,
      "direction_cos_threshold": params.shank_edge_direction_cos_threshold,
      "lateral_margin": params.shank_edge_lateral_margin,
      "capsule_start_local": params.shank_capsule_start_local,
      "capsule_end_local": params.shank_capsule_end_local,
      "capsule_radius": params.shank_capsule_radius,
      "asset_cfg": shank_asset_cfg(),
    },
  )
  cfg.rewards["stair_skip_layer_penalty"] = RewardTermCfg(
    func=mdp.stair_skip_layer_penalty,
    weight=params.stair_skip_layer_weight,
  )
  cfg.rewards["target_tread_midline_shaping"] = RewardTermCfg(
    func=mdp.target_tread_midline_shaping,
    weight=params.target_tread_midline_weight,
    params={
      "ground_contact_sensor_name": "feet_ground_contact",
      "height_clearance": params.target_tread_midline_height_clearance,
      "sigma_fraction": params.target_tread_midline_sigma_fraction,
      "min_sigma": params.target_tread_midline_min_sigma,
      "edge_margin": params.target_tread_midline_edge_margin,
      "progress_scale": params.target_tread_midline_progress_scale,
      "center_scale": params.target_tread_midline_center_scale,
      "edge_scale": params.target_tread_midline_edge_scale,
      "max_progress_step": params.target_tread_midline_max_progress_step,
      "heading_cos": params.toe_stair_heading_cos,
      "min_terrain_level": params.min_terrain_level,
      "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
    },
  )


def configure_g1_step_danger_play_visualization(
  cfg: ManagerBasedRlEnvCfg,
  params: G1SlowLatentPlayVisualizationParams | None = None,
  rewards: G1SlowLatentRewardParams | None = None,
) -> None:
  """Configure play-only debug views for the step-boundary danger zones."""
  params = params or G1SlowLatentPlayVisualizationParams()
  rewards = rewards or G1SlowLatentRewardParams()

  cfg.viewer.show_depth_camera_visualizers = params.show_depth_camera_visualizers

  for sensor in cfg.scene.sensors or ():
    if isinstance(sensor, RayCastSensorCfg):
      sensor.debug_vis = params.show_raycast_debug_visualizers

  if cfg.scene.terrain is None or cfg.scene.terrain.terrain_generator is None:
    return

  cfg.scene.terrain.terrain_generator.step_danger_visualization = (
    StepDangerVisualizationCfg(
      enabled=params.show_step_danger_zones,
      lip_radius=(
        rewards.foot_lip_edge_radius
        if params.danger_lip_radius is None
        else params.danger_lip_radius
      ),
      slab_depth=(
        rewards.toe_slab_depth
        if params.danger_slab_depth is None
        else params.danger_slab_depth
      ),
      slab_u_margin=(
        rewards.toe_slab_u_margin
        if params.danger_slab_u_margin is None
        else params.danger_slab_u_margin
      ),
      slab_v_margin=(
        rewards.toe_slab_v_margin
        if params.danger_slab_v_margin is None
        else params.danger_slab_v_margin
      ),
      geom_group=params.danger_geom_group,
    )
  )


def _configure_target_command(
  cfg: ManagerBasedRlEnvCfg,
  params: G1SlowLatentTargetCommandParams,
  play: bool,
) -> None:
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, TeacherTargetHeadingVelocityCommandCfg)
  twist_cmd.resampling_time_range = params.resampling_time_range
  twist_cmd.heading_control_stiffness = params.heading_control_stiffness
  twist_cmd.rel_target_envs = params.rel_target_envs
  twist_cmd.rel_random_heading_envs = params.rel_random_heading_envs
  twist_cmd.rel_standing_envs = params.rel_standing_envs
  twist_cmd.target_reached_threshold = params.target_reached_threshold
  twist_cmd.target_min_distance = params.target_min_distance
  twist_cmd.target_max_distance = params.target_max_distance
  twist_cmd.target_tile_radius = params.target_tile_radius
  twist_cmd.include_current_tile = params.include_current_tile
  twist_cmd.zero_lateral_velocity = params.zero_lateral_velocity
  twist_cmd.ranges.lin_vel_x = params.lin_vel_x
  twist_cmd.ranges.lin_vel_y = params.lin_vel_y
  twist_cmd.ranges.ang_vel_z = params.ang_vel_z

  if not play and "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (params.lin_vel_x[0], min(0.8, params.lin_vel_x[1])),
        "lin_vel_y": params.lin_vel_y,
        "ang_vel_z": (-0.5, 0.5),
      },
      {
        "step": 3000 * 24,
        "lin_vel_x": params.lin_vel_x,
        "lin_vel_y": params.lin_vel_y,
        "ang_vel_z": params.ang_vel_z,
      },
    ]

  if play:
    twist_cmd.rel_target_envs = params.play_rel_target_envs
    twist_cmd.rel_random_heading_envs = params.play_rel_random_heading_envs
    twist_cmd.rel_standing_envs = params.play_rel_standing_envs
    twist_cmd.heading_control_stiffness = params.play_heading_control_stiffness
    twist_cmd.ranges.lin_vel_x = params.play_lin_vel_x
    twist_cmd.ranges.lin_vel_y = params.play_lin_vel_y
    twist_cmd.ranges.ang_vel_z = params.play_ang_vel_z
    twist_cmd.target_min_distance = params.play_target_min_distance
    twist_cmd.target_max_distance = params.play_target_max_distance
    twist_cmd.target_tile_radius = params.play_target_tile_radius


def _configure_slow_latent_rewards(
  cfg: ManagerBasedRlEnvCfg, params: G1SlowLatentRewardParams
) -> None:
  def torso_body_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=("torso_link",))

  def foot_site_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))

  def foot_body_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
      "robot",
      body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
    )

  cfg.rewards["track_linear_velocity"].weight = params.track_linear_velocity_weight
  cfg.rewards["track_linear_velocity"].params.update(
    {
      "command_name": "twist",
      "std": params.track_linear_velocity_std,
    }
  )
  cfg.rewards["track_angular_velocity"].weight = params.track_angular_velocity_weight
  cfg.rewards["track_angular_velocity"].params.update(
    {
      "command_name": "twist",
      "std": params.track_angular_velocity_std,
    }
  )
  cfg.rewards["upright"].weight = params.upright_weight
  cfg.rewards["upright"].params.update(
    {
      "std": params.upright_std,
      "asset_cfg": torso_body_cfg(),
    }
  )
  cfg.rewards["pose"].weight = params.pose_weight
  cfg.rewards["pose"].params.update(
    {
      "command_name": "twist",
      "walking_threshold": params.pose_walking_threshold,
      "running_threshold": params.pose_running_threshold,
      "std_standing": dict(params.pose_std_standing),
      "std_walking": dict(params.pose_std_walking),
      "std_running": dict(params.pose_std_running),
    }
  )

  cfg.rewards["foot_clearance"].weight = params.foot_clearance_weight
  cfg.rewards["foot_clearance"].params.update(
    {
      "command_name": "twist",
      "command_threshold": params.foot_clearance_command_threshold,
      "height_sensor_name": "foot_height_scan",
      "min_height": params.foot_clearance_min_height,
      "max_height": params.foot_clearance_max_height,
      "asset_cfg": foot_site_cfg(),
    }
  )
  cfg.rewards["foot_swing_height"].weight = params.foot_swing_height_weight
  cfg.rewards["foot_swing_height"].params.update(
    {
      "command_name": "twist",
      "command_threshold": params.foot_swing_command_threshold,
      "height_sensor_name": "foot_height_scan",
      "sensor_name": "feet_ground_contact",
      "target_height": params.foot_swing_target_height,
    }
  )
  cfg.rewards["foot_slip"].weight = params.foot_slip_weight
  cfg.rewards["foot_slip"].params.update(
    {
      "command_name": "twist",
      "command_threshold": params.foot_slip_command_threshold,
      "sensor_name": "feet_ground_contact",
      "asset_cfg": foot_site_cfg(),
    }
  )
  cfg.rewards["soft_landing"].weight = params.soft_landing_weight
  cfg.rewards["soft_landing"].params.update(
    {
      "command_name": "twist",
      "command_threshold": params.soft_landing_command_threshold,
      "sensor_name": "feet_ground_contact",
    }
  )
  cfg.rewards["idle_penalty"].weight = params.idle_penalty_weight
  cfg.rewards["idle_penalty"].params.update(
    {
      "command_name": "twist",
      "command_threshold": params.idle_command_threshold,
      "velocity_threshold": params.idle_velocity_threshold,
    }
  )
  cfg.rewards["foot_gait"].weight = params.foot_gait_weight
  cfg.rewards["foot_gait"].func = mdp.stair_aware_feet_gait
  cfg.rewards["foot_gait"].params.update(
    {
      "command_name": "twist",
      "command_threshold": params.foot_gait_command_threshold,
      "sensor_name": "feet_ground_contact",
      "period": params.foot_gait_period,
      "offset": list(params.foot_gait_offset),
      "threshold": params.foot_gait_threshold,
      "heading_cos": params.toe_stair_heading_cos,
      "asset_cfg": foot_body_cfg(),
    }
  )
  cfg.rewards[
    "base_height_above_support"
  ].weight = params.base_height_above_support_weight
  cfg.rewards["base_height_above_support"].params.update(
    {
      "height_sensor_name": "foot_height_scan",
      "contact_sensor_name": "feet_ground_contact",
      "min_height": params.base_height_above_support_min_height,
      "error_scale": params.base_height_above_support_error_scale,
      "asset_cfg": foot_site_cfg(),
    }
  )

  cfg.rewards["body_ang_vel"].weight = params.body_ang_vel_weight
  cfg.rewards["body_ang_vel"].params["asset_cfg"] = torso_body_cfg()
  cfg.rewards["angular_momentum"].weight = params.angular_momentum_weight
  cfg.rewards["angular_momentum"].params["sensor_name"] = "robot/root_angmom"
  cfg.rewards["dof_pos_limits"].weight = params.joint_pos_limits_weight
  cfg.rewards["action_rate_l2"].weight = params.action_rate_l2_weight
  cfg.rewards["self_collisions"].weight = params.self_collisions_weight
  cfg.rewards["self_collisions"].params.update(
    {
      "sensor_name": "self_collision",
      "force_threshold": params.self_collision_force_threshold,
    }
  )
  cfg.rewards["joint_acc_l2"].weight = params.joint_acc_l2_weight
  cfg.rewards["action_acc_l2"].weight = params.action_acc_l2_weight

  cfg.rewards["target_progress"].weight = params.target_progress_weight
  cfg.rewards["target_progress"].params["min_distance"] = (
    params.target_progress_min_distance
  )
  cfg.rewards["target_reached_bonus"].weight = params.target_reached_bonus_weight

  configure_g1_step_danger_rewards(cfg, params)
  cfg.metrics["stair_sequence_event_logger"] = MetricsTermCfg(
    func=mdp.stair_sequence_event_logger,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        preserve_order=True,
      ),
      "command_name": "twist",
    },
  )


def _configure_latent_observations(
  cfg: ManagerBasedRlEnvCfg,
  params: G1SlowLatentEnvParams,
  play: bool = False,
) -> None:
  latent_terms = {
    params.latent_obs_term_name: ObservationTermCfg(
      func=mdp.stair_latent_obs,
      params={
        "toe_contact_sensor_name": TOE_TERRAIN_CONTACT_SENSOR,
        "asset_cfg": g1_foot_body_cfg(),
        "include_gait_phase": params.include_gait_phase_in_latent,
        "gait_period": params.rewards.foot_gait_period,
        "command_name": "twist",
      },
    ),
  }
  if params.enable_foot_event_memory_obs:
    latent_terms[params.foot_event_memory_term_name] = ObservationTermCfg(
      func=mdp.FootEventMemoryObs,
      params={
        "memory_len": params.foot_event_memory_length,
        "age_norm_s": params.foot_event_memory_age_norm_s,
        "stance_age_norm_s": params.foot_event_memory_age_norm_s,
        "gait_period": params.rewards.foot_gait_period,
        "command_name": "twist",
        "noise_enabled": not play,
      },
    )
  cfg.observations[params.latent_group_name] = ObservationGroupCfg(
    terms=latent_terms,
    concatenate_terms=True,
    enable_corruption=params.enable_latent_obs_corruption,
    history_length=0,
  )
  if params.enable_latent_labels:
    cfg.observations[params.label_group_name] = ObservationGroupCfg(
      terms={
        "toe_riser_event": ObservationTermCfg(
          func=mdp.toe_riser_event_label,
          params={},
        ),
        "stair_state": ObservationTermCfg(
          func=mdp.stair_state_label,
          params={},
        ),
        "stair_shape": ObservationTermCfg(
          func=mdp.stair_shape_label,
          params={},
        ),
        "safe_stride": ObservationTermCfg(
          func=mdp.safe_stride_label,
          params={},
        ),
        "future_events": ObservationTermCfg(
          func=mdp.stair_future_event_labels,
          params={},
        ),
        "shape_component_valid": ObservationTermCfg(
          func=mdp.stair_shape_component_valid_label,
          params={},
        ),
        "safe_stride_interval_valid": ObservationTermCfg(
          func=mdp.safe_stride_interval_valid_label,
          params={},
        ),
        "stair_depth_confirmation_event": ObservationTermCfg(
          func=mdp.stair_depth_confirmation_event_label,
          params={},
        ),
        "geometry_probe_validation": ObservationTermCfg(
          func=mdp.geometry_probe_validation_mask,
          params={"validation_fraction": 0.2},
        ),
        "stair_depth_confirmation_age": ObservationTermCfg(
          func=mdp.stair_depth_confirmation_age_label,
          params={},
        ),
        "stair_adjacent_pair_evidence": ObservationTermCfg(
          func=mdp.stair_adjacent_pair_evidence_label,
          params={},
        ),
      },
      concatenate_terms=True,
      enable_corruption=False,
      history_length=0,
    )

  cfg.events["reset_stair_latent_cache"] = EventTermCfg(
    func=mdp.reset_stair_latent_cache,
    mode="reset",
    params={},
  )


def _configure_slow_latent_play_visualization(
  cfg: ManagerBasedRlEnvCfg,
  params: G1SlowLatentPlayVisualizationParams,
  rewards: G1SlowLatentRewardParams,
) -> None:
  configure_g1_step_danger_play_visualization(cfg, params, rewards)


def unitree_g1_blind_rough_target_navigation_slow_latent_env_cfg(
  play: bool = False,
  params: G1SlowLatentEnvParams | None = None,
  actor_history_length: int | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create the gated slow-latent blind target-navigation student task."""
  params = params or G1SlowLatentEnvParams()
  if actor_history_length is not None:
    params = replace(params, actor_history_length=actor_history_length)

  cfg = unitree_g1_blind_rough_teacherkl_env_cfg(
    play=play,
    use_target_navigation=True,
  )
  cfg.observations["actor"].history_length = params.actor_history_length
  if not play:
    configure_g1_high_stairs_mixed_replay(
      cfg,
      start_level=params.terrain_replay.start_level,
      level_ranges=params.terrain_replay.level_ranges,
      weights=params.terrain_replay.weights,
    )
  _configure_target_command(cfg, params.target_command, play)
  _configure_slow_latent_rewards(cfg, params.rewards)
  _configure_latent_observations(cfg, params, play=play)
  if play:
    _configure_slow_latent_play_visualization(
      cfg, params.play_visualization, params.rewards
    )
  return cfg


def unitree_g1_blind_rough_target_navigation_semantic_v2_probe_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the Semantic-v2 probe task with deployable gait phase appended."""
  params = G1SlowLatentEnvParams(include_gait_phase_in_latent=True)
  return unitree_g1_blind_rough_target_navigation_slow_latent_env_cfg(
    play=play,
    params=params,
  )
