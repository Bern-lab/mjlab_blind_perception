"""Gated slow-latent Unitree G1 blind target-navigation task config."""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
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
  lin_vel_x: tuple[float, float] = (0.0, 1.0)
  lin_vel_y: tuple[float, float] = (0.0, 0.0)
  ang_vel_z: tuple[float, float] = (-0.8, 0.8)
  play_rel_target_envs: float = 1.0
  play_rel_random_heading_envs: float = 0.0
  play_rel_standing_envs: float = 0.0
  play_heading_control_stiffness: float = 0.8
  play_lin_vel_x: tuple[float, float] = (0.3, 0.9)
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
      r".*hip_pitch.*": 0.4,
      r".*hip_roll.*": 0.15,
      r".*hip_yaw.*": 0.15,
      r".*knee.*": 0.45,
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
      r".*hip_pitch.*": 0.5,
      r".*hip_roll.*": 0.2,
      r".*hip_yaw.*": 0.2,
      r".*knee.*": 0.6,
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
  foot_swing_height_weight: float = -0.75  # 0.25
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
  foot_lip_ignore_boundary_layers: int = 2
  toe_slab_weight: float = -4.2
  toe_slab_depth: float = 0.10
  toe_slab_u_margin: float = 0.02
  toe_slab_v_margin: float = 0.05
  toe_x_min: float = 0.08
  toe_v_threshold: float = 0.02
  toe_approach_speed_floor: float = 0.08
  surface_tol: float = 0.005
  nearest_boundaries: int = 4
  min_terrain_level: int = 3
  toe_contact_penalty_scale: float = 0.5
  toe_contact_time_scale: float = 0.20
  toe_contact_force_threshold: float = 15.0
  toe_contact_force_scale: float = 60.0
  toe_contact_vertical_normal_z_max: float = 0.4
  toe_contact_forward_velocity_threshold: float = 0.05
  toe_probe_contact_count: int = 2
  # Kept for older configs; layer probing now rewards first contact only.
  toe_probe_slab_reward_scale: float = 0.0
  toe_probe_contact_reward: float = 0.20
  toe_probe_min_progress: float = 0.0
  toe_probe_max_safe_force: float | None = None
  toe_probe_cooldown_time: float = 0.20
  toe_second_layer_attraction_reward: float = 0.15
  toe_second_layer_attraction_distance: float = 0.55
  toe_second_layer_attraction_u_margin: float = 0.08
  toe_second_layer_attraction_v_margin: float = 0.12
  toe_probe_min_ascent_height: float = 0.03
  toe_probe_ascent_velocity_threshold: float = 0.03


@dataclass(frozen=True)
class G1SlowLatentLabelParams:
  """Simulation-only labels used by slow-latent auxiliary losses."""

  toe_event_force_threshold: float = 15.0
  toe_event_vertical_normal_z_max: float = 0.4
  stair_state_min_terrain_level: int = 3


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
  label_group_name: str = "latent_labels"
  """Observation group used by auxiliary slow-latent losses during training."""
  enable_latent_labels: bool = True
  """Enable simulation-only labels for event/stair auxiliary losses."""
  enable_latent_obs_corruption: bool = False
  """Apply observation corruption to latent encoder inputs."""
  target_command: G1SlowLatentTargetCommandParams = field(
    default_factory=G1SlowLatentTargetCommandParams
  )
  """Target-heading command distribution and play overrides."""
  rewards: G1SlowLatentRewardParams = field(default_factory=G1SlowLatentRewardParams)
  """All slow-latent reward weights and key reward parameters."""
  labels: G1SlowLatentLabelParams = field(default_factory=G1SlowLatentLabelParams)
  """Auxiliary label thresholds for slow-latent training."""
  terrain_replay: G1SlowLatentTerrainReplayParams = field(
    default_factory=G1SlowLatentTerrainReplayParams
  )
  """Late-curriculum mixed terrain replay schedule."""
  play_visualization: G1SlowLatentPlayVisualizationParams = field(
    default_factory=G1SlowLatentPlayVisualizationParams
  )
  """Play-only viewer/debug visualization settings."""


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

  def foot_asset_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
      "robot",
      body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
    )

  def foot_site_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))

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
  cfg.rewards["foot_gait"].params.update(
    {
      "command_name": "twist",
      "command_threshold": params.foot_gait_command_threshold,
      "sensor_name": "feet_ground_contact",
      "period": params.foot_gait_period,
      "offset": list(params.foot_gait_offset),
      "threshold": params.foot_gait_threshold,
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
      "approach_speed_floor": params.toe_approach_speed_floor,
      "surface_tol": params.surface_tol,
      "nearest_boundaries": params.nearest_boundaries,
      "min_terrain_level": params.min_terrain_level,
      "contact_sensor_name": TOE_TERRAIN_CONTACT_SENSOR,
      "contact_penalty_scale": params.toe_contact_penalty_scale,
      "contact_time_scale": params.toe_contact_time_scale,
      "contact_force_threshold": params.toe_contact_force_threshold,
      "contact_force_scale": params.toe_contact_force_scale,
      "contact_vertical_normal_z_max": params.toe_contact_vertical_normal_z_max,
      "contact_forward_velocity_threshold": (
        params.toe_contact_forward_velocity_threshold
      ),
      "probe_contact_count": params.toe_probe_contact_count,
      "probe_slab_reward_scale": params.toe_probe_slab_reward_scale,
      "probe_contact_reward": params.toe_probe_contact_reward,
      "probe_min_progress": params.toe_probe_min_progress,
      "probe_max_safe_force": params.toe_probe_max_safe_force,
      "probe_cooldown_time": params.toe_probe_cooldown_time,
      "second_layer_attraction_reward": params.toe_second_layer_attraction_reward,
      "second_layer_attraction_distance": params.toe_second_layer_attraction_distance,
      "second_layer_attraction_u_margin": params.toe_second_layer_attraction_u_margin,
      "second_layer_attraction_v_margin": params.toe_second_layer_attraction_v_margin,
      "min_ascent_height": params.toe_probe_min_ascent_height,
      "ascent_velocity_threshold": params.toe_probe_ascent_velocity_threshold,
      "asset_cfg": foot_asset_cfg(),
    },
  )


def _configure_latent_observations(
  cfg: ManagerBasedRlEnvCfg,
  params: G1SlowLatentEnvParams,
) -> None:
  cfg.observations[params.latent_group_name] = ObservationGroupCfg(
    terms={
      params.latent_obs_term_name: ObservationTermCfg(
        func=mdp.stair_latent_obs,
        params={
          "toe_contact_sensor_name": TOE_TERRAIN_CONTACT_SENSOR,
          "asset_cfg": g1_foot_body_cfg(),
        },
      ),
    },
    concatenate_terms=True,
    enable_corruption=params.enable_latent_obs_corruption,
    history_length=0,
  )
  if params.enable_latent_labels:
    cfg.observations[params.label_group_name] = ObservationGroupCfg(
      terms={
        "toe_riser_event": ObservationTermCfg(
          func=mdp.toe_riser_event_label,
          params={
            "sensor_name": TOE_TERRAIN_CONTACT_SENSOR,
            "force_threshold": params.labels.toe_event_force_threshold,
            "vertical_normal_z_max": params.labels.toe_event_vertical_normal_z_max,
          },
        ),
        "stair_state": ObservationTermCfg(
          func=mdp.stair_state_label,
          params={
            "min_terrain_level": params.labels.stair_state_min_terrain_level,
            "sensor_name": TOE_TERRAIN_CONTACT_SENSOR,
          },
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


def unitree_g1_blind_rough_target_navigation_slow_latent_env_cfg(
  play: bool = False,
  params: G1SlowLatentEnvParams | None = None,
  actor_history_length: int | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create the gated slow-latent blind target-navigation student task."""
  params = params or G1SlowLatentEnvParams()
  if actor_history_length is not None:
    params = G1SlowLatentEnvParams(
      actor_history_length=actor_history_length,
      latent_group_name=params.latent_group_name,
      latent_obs_term_name=params.latent_obs_term_name,
      label_group_name=params.label_group_name,
      enable_latent_labels=params.enable_latent_labels,
      enable_latent_obs_corruption=params.enable_latent_obs_corruption,
      target_command=params.target_command,
      rewards=params.rewards,
      labels=params.labels,
      terrain_replay=params.terrain_replay,
      play_visualization=params.play_visualization,
    )

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
  _configure_latent_observations(cfg, params)
  if play:
    _configure_slow_latent_play_visualization(
      cfg, params.play_visualization, params.rewards
    )
  return cfg
