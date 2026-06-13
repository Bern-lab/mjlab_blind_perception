"""Step-danger Teacher-KL Unitree G1 blind target-navigation task config."""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import StepDangerVisualizationCfg

from .blind_rough_teacher_kl_env_cfg import unitree_g1_blind_rough_teacherkl_env_cfg


@dataclass(frozen=True)
class G1StepDangerFootLipRewardParams:
  """Foot-volume penalty parameters around high-side step lips."""

  weight: float = -3.2
  edge_radius: float = 0.07
  edge_height_band: float = 0.06
  support_speed_floor: float = 0.08
  nearest_boundaries: int = 4
  min_terrain_level: int = 3


@dataclass(frozen=True)
class G1StepDangerToeRiserProbeRewardParams:
  """Signed toe-riser probing reward with integrated slab-cost protection."""

  weight: float = 1.0
  slab_weight: float = 4.2
  slab_depth: float = 0.10
  u_margin: float = 0.02
  v_margin: float = 0.05
  toe_x_min: float = 0.08
  toe_v_threshold: float = 0.02
  surface_tol: float = 0.005
  nearest_boundaries: int = 4
  min_terrain_level: int = 3
  contact_sensor_name: str = "toe_terrain_contact"
  ground_contact_sensor_name: str = "feet_ground_contact"
  command_name: str = "twist"
  progress_weight: float = 0.3
  second_hit_weight: float = 1.5
  repeat_layer1_weight: float = 0.3
  wrong_foot_weight: float = 0.3
  wrong_layer_weight: float = 0.3
  bad_heading_weight: float = 0.2
  sticky_weight: float = 0.2
  over_high_weight: float = 0.03
  boundary_match_radius: float = 0.08
  forward_tol: float = 0.05
  low_side_margin: float = 0.02
  merge_riser_eps: float = 0.04
  min_blocking_force: float = 5.0
  force_sign: float = 1.0
  vertical_normal_z_max: float = 0.4
  forward_velocity_threshold: float = 0.05
  cos_heading_tol: float = 0.5
  lateral_margin: float = 0.10
  root_lateral_margin: float = 0.20
  z_margin_low: float = 0.03
  z_margin_high: float = 0.08
  side_eps: float = 0.03
  max_progress: float = 0.05
  timeout_s: float = 1.0
  hard_timeout_s: float = 1.2
  sticky_time: float = 0.20


@dataclass(frozen=True)
class G1StepDangerRewardParams:
  """Step-boundary reward parameters for the non-latent StepDanger task."""

  foot_lip: G1StepDangerFootLipRewardParams = field(
    default_factory=G1StepDangerFootLipRewardParams
  )
  toe_riser_probe: G1StepDangerToeRiserProbeRewardParams = field(
    default_factory=G1StepDangerToeRiserProbeRewardParams
  )


@dataclass(frozen=True)
class G1StepDangerPlayVisualizationParams:
  """Play-only visualization knobs for step danger zones."""

  show_depth_camera_visualizers: bool = False
  show_raycast_debug_visualizers: bool = False
  show_step_danger_zones: bool = True
  danger_lip_radius: float | None = None
  danger_slab_depth: float | None = None
  danger_slab_u_margin: float | None = None
  danger_slab_v_margin: float | None = None
  danger_geom_group: int = 4


@dataclass(frozen=True)
class G1StepDangerEnvParams:
  """Config knobs for the non-latent step-danger target-navigation task."""

  rewards: G1StepDangerRewardParams = field(default_factory=G1StepDangerRewardParams)
  """Local step-boundary danger and toe-riser probe reward parameters."""
  play_visualization: G1StepDangerPlayVisualizationParams = field(
    default_factory=G1StepDangerPlayVisualizationParams
  )
  """Play-only danger-zone visualization settings."""


def _g1_foot_asset_cfg() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot",
    body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
  )


def _configure_step_danger_rewards(
  cfg: ManagerBasedRlEnvCfg,
  params: G1StepDangerRewardParams,
) -> None:
  """Configure local step danger and toe-riser probe rewards."""
  foot_lip = params.foot_lip
  cfg.rewards["foot_step_lip_volume_penalty"] = RewardTermCfg(
    func=mdp.foot_step_lip_volume_penalty,
    weight=foot_lip.weight,
    params={
      "edge_radius": foot_lip.edge_radius,
      "edge_height_band": foot_lip.edge_height_band,
      "support_speed_floor": foot_lip.support_speed_floor,
      "nearest_boundaries": foot_lip.nearest_boundaries,
      "contact_sensor_name": "feet_ground_contact",
      "min_terrain_level": foot_lip.min_terrain_level,
      "asset_cfg": _g1_foot_asset_cfg(),
    },
  )

  toe_probe = params.toe_riser_probe
  cfg.rewards.pop("toe_step_riser_slab_penalty", None)
  cfg.rewards["toe_step_riser_probe_shaping_reward"] = RewardTermCfg(
    func=mdp.toe_step_riser_probe_shaping_reward,
    weight=toe_probe.weight,
    params={
      "slab_weight": toe_probe.slab_weight,
      "slab_depth": toe_probe.slab_depth,
      "u_margin": toe_probe.u_margin,
      "v_margin": toe_probe.v_margin,
      "toe_x_min": toe_probe.toe_x_min,
      "toe_v_threshold": toe_probe.toe_v_threshold,
      "surface_tol": toe_probe.surface_tol,
      "nearest_boundaries": toe_probe.nearest_boundaries,
      "min_terrain_level": toe_probe.min_terrain_level,
      "contact_sensor_name": toe_probe.contact_sensor_name,
      "ground_contact_sensor_name": toe_probe.ground_contact_sensor_name,
      "command_name": toe_probe.command_name,
      "progress_weight": toe_probe.progress_weight,
      "second_hit_weight": toe_probe.second_hit_weight,
      "repeat_layer1_weight": toe_probe.repeat_layer1_weight,
      "wrong_foot_weight": toe_probe.wrong_foot_weight,
      "wrong_layer_weight": toe_probe.wrong_layer_weight,
      "bad_heading_weight": toe_probe.bad_heading_weight,
      "sticky_weight": toe_probe.sticky_weight,
      "over_high_weight": toe_probe.over_high_weight,
      "boundary_match_radius": toe_probe.boundary_match_radius,
      "forward_tol": toe_probe.forward_tol,
      "low_side_margin": toe_probe.low_side_margin,
      "merge_riser_eps": toe_probe.merge_riser_eps,
      "min_blocking_force": toe_probe.min_blocking_force,
      "force_sign": toe_probe.force_sign,
      "vertical_normal_z_max": toe_probe.vertical_normal_z_max,
      "forward_velocity_threshold": toe_probe.forward_velocity_threshold,
      "cos_heading_tol": toe_probe.cos_heading_tol,
      "lateral_margin": toe_probe.lateral_margin,
      "root_lateral_margin": toe_probe.root_lateral_margin,
      "z_margin_low": toe_probe.z_margin_low,
      "z_margin_high": toe_probe.z_margin_high,
      "side_eps": toe_probe.side_eps,
      "max_progress": toe_probe.max_progress,
      "timeout_s": toe_probe.timeout_s,
      "hard_timeout_s": toe_probe.hard_timeout_s,
      "sticky_time": toe_probe.sticky_time,
      "asset_cfg": _g1_foot_asset_cfg(),
    },
  )


def _configure_step_danger_play_visualization(
  cfg: ManagerBasedRlEnvCfg,
  params: G1StepDangerPlayVisualizationParams,
  rewards: G1StepDangerRewardParams,
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
        rewards.foot_lip.edge_radius
        if params.danger_lip_radius is None
        else params.danger_lip_radius
      ),
      slab_depth=(
        rewards.toe_riser_probe.slab_depth
        if params.danger_slab_depth is None
        else params.danger_slab_depth
      ),
      slab_u_margin=(
        rewards.toe_riser_probe.u_margin
        if params.danger_slab_u_margin is None
        else params.danger_slab_u_margin
      ),
      slab_v_margin=(
        rewards.toe_riser_probe.v_margin
        if params.danger_slab_v_margin is None
        else params.danger_slab_v_margin
      ),
      geom_group=params.danger_geom_group,
    )
  )


def unitree_g1_blind_rough_target_navigation_step_danger_env_cfg(
  play: bool = False,
  params: G1StepDangerEnvParams | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create a non-latent target-navigation task with toe-riser probe shaping."""
  params = params or G1StepDangerEnvParams()

  cfg = unitree_g1_blind_rough_teacherkl_env_cfg(
    play=play,
    use_target_navigation=True,
  )
  _configure_step_danger_rewards(cfg, params.rewards)
  if play:
    _configure_step_danger_play_visualization(
      cfg,
      params.play_visualization,
      params.rewards,
    )
  return cfg
