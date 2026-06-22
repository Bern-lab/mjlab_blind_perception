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
class G1StepDangerToeRiserSlabPenaltyParams:
  """Penalty parameters for the low-side stair-riser danger slab."""

  weight: float = -4.2
  slab_depth: float = 0.10
  u_margin: float = 0.02
  v_margin: float = 0.05
  toe_x_min: float = 0.08
  toe_v_threshold: float = 0.02
  surface_tol: float = 0.005
  nearest_boundaries: int = 4
  min_terrain_level: int = 3


@dataclass(frozen=True)
class G1StepDangerRewardParams:
  """Step-boundary reward parameters for the non-latent StepDanger task."""

  foot_lip: G1StepDangerFootLipRewardParams = field(
    default_factory=G1StepDangerFootLipRewardParams
  )
  toe_riser_slab: G1StepDangerToeRiserSlabPenaltyParams = field(
    default_factory=G1StepDangerToeRiserSlabPenaltyParams
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
  """Local step-boundary danger penalty parameters."""
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
  """Configure penalty-only local step danger rewards."""
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

  toe_slab = params.toe_riser_slab
  cfg.rewards["toe_step_riser_slab_penalty"] = RewardTermCfg(
    func=mdp.toe_step_riser_slab_penalty,
    weight=toe_slab.weight,
    params={
      "slab_depth": toe_slab.slab_depth,
      "u_margin": toe_slab.u_margin,
      "v_margin": toe_slab.v_margin,
      "toe_x_min": toe_slab.toe_x_min,
      "toe_v_threshold": toe_slab.toe_v_threshold,
      "surface_tol": toe_slab.surface_tol,
      "nearest_boundaries": toe_slab.nearest_boundaries,
      "min_terrain_level": toe_slab.min_terrain_level,
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
        rewards.toe_riser_slab.slab_depth
        if params.danger_slab_depth is None
        else params.danger_slab_depth
      ),
      slab_u_margin=(
        rewards.toe_riser_slab.u_margin
        if params.danger_slab_u_margin is None
        else params.danger_slab_u_margin
      ),
      slab_v_margin=(
        rewards.toe_riser_slab.v_margin
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
  """Create a non-latent target-navigation task with step danger penalties."""
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
