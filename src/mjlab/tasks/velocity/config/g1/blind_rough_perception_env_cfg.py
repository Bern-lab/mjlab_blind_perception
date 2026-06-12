"""Self-contained Perception Teacher-KL env configuration for Unitree G1.

Two variants are supported via ``use_target_navigation``:

* ``False`` – Velocity tracking with uniform commands on blind rough terrain.
* ``True``  – Target-heading navigation on blind rough terrain with
  flat-patch-sampled targets.

Both variants share the same step-boundary danger-zone rewards
(``foot_step_lip_volume_penalty`` + ``toe_step_riser_slab_penalty``) and do
NOT use ``toe_riser_contact_memory_penalty``.

All reward weights / parameters are exposed through :class:`G1PerceptionRewardParams`.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  CameraSensorCfg,
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp.teacher_target_heading_command import (
  TeacherTargetHeadingVelocityCommandCfg,
)
from mjlab.tasks.velocity.mdp.teacher_target_heading_rewards import (
  teacher_target_progress,
  teacher_target_reached_bonus,
)
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains import (
  FlatPatchSamplingCfg,
  StepDangerVisualizationCfg,
  TerrainGeneratorCfg,
)
from mjlab.terrains.config import BLIND_HIGH_STAIRS_TERRAINS_CFG
from mjlab.terrains.primitive_terrains import (
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
)
from mjlab.utils.color import RGBA
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEACHER_OBSERVATION_ORDER = (
  "base_ang_vel",
  "projected_gravity",
  "velocity_commands",
  "gait_phase",
  "joint_pos_rel",
  "joint_vel_rel",
  "last_action",
  "height_scan",
  "base_lin_vel",
  "foot_height",
)

TOE_TERRAIN_CONTACT_SENSOR = "toe_terrain_contact"
G1_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")

TEACHER_DEPTH_SENSOR = "front_depth"
STUDENT_DEPTH_SENSOR = "front_depth_stack"
STUDENT_DEPTH_OBS_GROUP = "camera_stack"
G1_D435I_DEPTH_FOVY_DEG = 55.2
G1_D435I_MOUNT_ANGLE_FROM_VERTICAL_DEG = 42.4
G1_D435I_DEPTH_WIDTH = 64
G1_D435I_DEPTH_HEIGHT = 36
G1_D435I_DEPTH_RANGE_M = 3.0
G1_D435I_DEPTH_POS_IN_TORSO = (0.10, 0.0, 0.45)
G1_D435I_DEPTH_QUAT_IN_TORSO = (
  math.cos(math.radians(G1_D435I_MOUNT_ANGLE_FROM_VERTICAL_DEG) * 0.5),
  0.0,
  -math.sin(math.radians(G1_D435I_MOUNT_ANGLE_FROM_VERTICAL_DEG) * 0.5),
  0.0,
)

G1_HIGH_STAIRS_MIXED_REPLAY_START_LEVEL = 8
G1_HIGH_STAIRS_MIXED_REPLAY_LEVEL_RANGES = ((0, 2), (3, 5), (6, 9))
G1_HIGH_STAIRS_MIXED_REPLAY_WEIGHTS = (0.2, 0.3, 0.5)
G1_HIGH_STAIRS_PLAY_NUM_ROWS = 5
G1_HIGH_STAIRS_PLAY_NUM_COLS = 5
G1_DANGER_ZONE_RGBA = RGBA(1.0, 0.72, 0.12, 0.30)


# ---------------------------------------------------------------------------
# Reward parameters dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class G1PerceptionRewardParams:
  """All reward weights and parameters for Perception Teacher-KL tasks.

  Every value that was previously hard-coded in
  ``_configure_teacherkl_student_env`` / ``_configure_teacherkl_target_navigation``
  is exposed here so that it can be overridden at the call site.
  """

  # -- Proprioceptive / regularisation rewards -----------------------------------
  body_ang_vel_weight: float = -0.08
  """Weight for torso angular velocity penalty."""
  angular_momentum_weight: float = -0.03
  """Weight for angular momentum penalty."""
  dof_pos_limits_weight: float = -1.0
  """Weight for joint position limits penalty."""
  action_rate_l2_weight: float = -0.15
  """Weight for action-rate L2 penalty."""
  joint_acc_l2_weight: float = -2.5e-7
  """Weight for joint acceleration L2 penalty."""
  action_acc_l2_weight: float = -0.05
  """Weight for action acceleration L2 penalty."""
  self_collisions_weight: float = -1.0
  """Weight for self-collision penalty."""
  self_collisions_force_threshold: float = 10.0
  """Force threshold for self-collision penalty."""

  # -- Task rewards (shared across both variants) ---------------------------------
  track_linear_velocity_weight: float = 2.0
  track_linear_velocity_std: float = math.sqrt(0.25)
  track_angular_velocity_weight: float = 2.0
  track_angular_velocity_std: float = math.sqrt(0.5)
  upright_weight: float = 1.0
  upright_std: float = math.sqrt(0.2)
  pose_weight: float = 1.0
  foot_clearance_weight: float = -2.0
  foot_swing_height_weight: float = -0.25
  foot_swing_height_command_threshold: float = 0.01
  foot_slip_weight: float = -0.2
  soft_landing_weight: float = -2e-5
  idle_penalty_weight: float = -2.0
  foot_gait_weight: float = 0.5
  base_height_above_support_weight: float = -0.5

  # -- Target navigation rewards (only active when use_target_navigation=True) -----
  target_progress_weight: float = 0.8
  target_progress_min_distance: float = 0.05
  target_reached_bonus_weight: float = 0.4

  # -- Step-boundary danger-zone rewards (foot_step_lip_volume_penalty) ------------
  foot_lip_weight: float = -3.2
  foot_lip_edge_radius: float = 0.07
  foot_lip_edge_height_band: float = 0.06
  foot_lip_support_speed_floor: float = 0.08
  foot_lip_nearest_boundaries: int = 4
  foot_lip_min_terrain_level: int = 3

  # -- Step-boundary danger-zone rewards (toe_step_riser_slab_penalty) -------------
  toe_slab_weight: float = -4.2
  toe_slab_depth: float = 0.10
  toe_slab_u_margin: float = 0.02
  toe_slab_v_margin: float = 0.05
  toe_x_min: float = 0.08
  toe_v_threshold: float = 0.02
  surface_tol: float = 0.005
  nearest_boundaries: int = 4
  min_terrain_level: int = 3

  # -- Step-boundary danger-zone rewards (heel_step_riser_clearance_penalty) -------
  heel_clearance_weight: float = -3.5
  heel_clearance: float = 0.10
  heel_u_margin: float = 0.04
  heel_v_margin: float = 0.06
  heel_x_max: float = 0.0

  # -- Step-boundary danger-zone rewards (foot_landing_flatness_penalty) ------------
  foot_landing_flatness_weight: float = -2.0
  foot_landing_near_height: float = 0.15
  foot_landing_max_tilt_deg: float = 12.0
  foot_landing_max_upward_speed: float = 0.10

  # -- Step-boundary danger-zone rewards (shank_step_lip_proximity_penalty) ---------
  shank_lip_weight: float = -1.2
  shank_clearance_radius: float = 0.20
  shank_collision_radius: float = 0.05
  shank_collision_weight: float = 4.0
  shank_height_gain_threshold: float = 0.03
  shank_ascent_hold_steps: int = 4
  shank_tilt_threshold_deg: float = 15.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _g1_foot_asset_cfg() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot",
    body_names=G1_FOOT_BODY_NAMES,
    preserve_order=True,
  )


def _add_toe_terrain_contact_sensor(cfg: ManagerBasedRlEnvCfg) -> None:
  """Ensure the toe-terrain contact sensor is present.

  This sensor is required by ``toe_step_riser_slab_penalty`` but we do
  **not** add ``toe_riser_contact_memory_penalty`` itself.
  """
  sensor_names = {sensor.name for sensor in cfg.scene.sensors or ()}
  if TOE_TERRAIN_CONTACT_SENSOR in sensor_names:
    return

  toe_contact_cfg = ContactSensorCfg(
    name=TOE_TERRAIN_CONTACT_SENSOR,
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force", "pos", "normal", "tangent"),
    reduce="maxforce",
    num_slots=4,
    track_air_time=True,
    global_frame=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (toe_contact_cfg,)


def _add_toe_terrain_contact_critic_obs(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add toe-terrain contact observations to the critic group."""
  cfg.observations["critic"].terms[TOE_TERRAIN_CONTACT_SENSOR] = ObservationTermCfg(
    func=mdp.foot_contact,
    params={"sensor_name": TOE_TERRAIN_CONTACT_SENSOR},
  )
  cfg.observations["critic"].terms["toe_terrain_contact_forces"] = ObservationTermCfg(
    func=mdp.foot_contact_forces,
    params={"sensor_name": TOE_TERRAIN_CONTACT_SENSOR},
  )


def _add_teacher_depth_camera(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add the depth camera observation expected by the frozen teacher."""
  sensors = tuple(cfg.scene.sensors or ())
  if not any(
    getattr(sensor, "name", None) == TEACHER_DEPTH_SENSOR for sensor in sensors
  ):
    cfg.scene.sensors = sensors + (
      CameraSensorCfg(
        name=TEACHER_DEPTH_SENSOR,
        parent_body="robot/torso_link",
        pos=(0.10, 0.0, 0.45),
        quat=(0.95371695, 0.0, -0.30070580, 0.0),
        fovy=80,
        width=64,
        height=64,
        data_types=("depth",),
        visualizer_max_range=5.0,
        enabled_geom_groups=(0, 2, 3),
        use_shadows=False,
        use_textures=True,
      ),
    )

  cfg.observations["camera"] = ObservationGroupCfg(
    terms={
      "front_depth": ObservationTermCfg(
        func=mdp.camera_depth,
        params={
          "sensor_name": TEACHER_DEPTH_SENSOR,
          "cutoff_distance": 5.0,
        },
      ),
    },
    enable_corruption=False,
    concatenate_terms=True,
    concatenate_dim=0,
  )


def _add_student_depth_camera_stack(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add the stacked depth observation used by the perception student."""
  sensors = tuple(cfg.scene.sensors or ())
  if not any(
    getattr(sensor, "name", None) == STUDENT_DEPTH_SENSOR for sensor in sensors
  ):
    cfg.scene.sensors = sensors + (
      CameraSensorCfg(
        name=STUDENT_DEPTH_SENSOR,
        parent_body="robot/torso_link",
        pos=G1_D435I_DEPTH_POS_IN_TORSO,
        quat=G1_D435I_DEPTH_QUAT_IN_TORSO,
        fovy=G1_D435I_DEPTH_FOVY_DEG,
        width=G1_D435I_DEPTH_WIDTH,
        height=G1_D435I_DEPTH_HEIGHT,
        data_types=("depth",),
        visualizer_max_range=G1_D435I_DEPTH_RANGE_M,
        enabled_geom_groups=(0, 2, 3),
        use_shadows=False,
        use_textures=True,
      ),
    )

  cfg.observations[STUDENT_DEPTH_OBS_GROUP] = ObservationGroupCfg(
    terms={
      "front_depth_stack": ObservationTermCfg(
        func=mdp.CameraDepthStack,
        params={
          "sensor_name": STUDENT_DEPTH_SENSOR,
          "cutoff_distance": G1_D435I_DEPTH_RANGE_M,
          "stack_length": 8,
        },
      ),
    },
    enable_corruption=False,
    concatenate_terms=True,
    concatenate_dim=0,
  )


def _reset_teacher_term_temporal_state(term: ObservationTermCfg) -> None:
  """Match the frozen teacher's feedforward observation pipeline."""
  term.delay_min_lag = 0
  term.delay_max_lag = 0
  term.delay_per_env = True
  term.delay_hold_prob = 0.0
  term.delay_update_period = 0
  term.delay_per_env_phase = True
  term.history_length = 0
  term.flatten_history_dim = True


def _make_teacher_terms(cfg: ManagerBasedRlEnvCfg) -> dict[str, ObservationTermCfg]:
  """Build teacher observations in the exact order used by the frozen teacher."""
  actor_terms = cfg.observations["actor"].terms
  critic_terms = cfg.observations["critic"].terms

  terms: dict[str, ObservationTermCfg] = {}
  for term_name in TEACHER_OBSERVATION_ORDER:
    source_terms = actor_terms if term_name in actor_terms else critic_terms
    term = deepcopy(source_terms[term_name])
    _reset_teacher_term_temporal_state(term)
    term.noise = None
    terms[term_name] = term

  return terms


# ---------------------------------------------------------------------------
# Target flat-patch sampling
# ---------------------------------------------------------------------------


def _add_target_flat_patch_sampling(
  terrain_generator: TerrainGeneratorCfg,
) -> TerrainGeneratorCfg:
  """Return a copied terrain generator with target flat-patch sampling."""
  terrain_cfg = deepcopy(terrain_generator)
  for name, sub_cfg in terrain_cfg.sub_terrains.items():
    if "stairs" in name or "slope" in name:
      target_sampling = FlatPatchSamplingCfg(
        num_patches=600,
        patch_radius=0.20,
        max_height_diff=0.02,
        x_range=(3.0, 5.0),
        y_range=(3.0, 5.0),
        grid_resolution=0.05,
      )
    else:
      target_sampling = FlatPatchSamplingCfg(
        num_patches=600,
        patch_radius=0.20,
        max_height_diff=0.02,
        x_range=(0.5, 7.5),
        y_range=(0.5, 7.5),
        grid_resolution=0.05,
      )

    sub_cfg.flat_patch_sampling = dict(sub_cfg.flat_patch_sampling or {})
    sub_cfg.flat_patch_sampling["target"] = target_sampling
  return terrain_cfg


# ---------------------------------------------------------------------------
# Terrain helpers
# ---------------------------------------------------------------------------


def _configure_high_stairs_play_terrain_generator(
  terrain_cfg: TerrainGeneratorCfg,
) -> TerrainGeneratorCfg:
  terrain_cfg.curriculum = False
  terrain_cfg.num_rows = G1_HIGH_STAIRS_PLAY_NUM_ROWS
  terrain_cfg.num_cols = G1_HIGH_STAIRS_PLAY_NUM_COLS
  terrain_cfg.border_width = 10.0
  terrain_cfg.step_danger_visualization.enabled = True
  terrain_cfg.step_danger_visualization.geom_group = 2
  terrain_cfg.step_danger_visualization.lip_rgba = G1_DANGER_ZONE_RGBA
  terrain_cfg.step_danger_visualization.slab_rgba = G1_DANGER_ZONE_RGBA

  for terrain_name in ("high_stairs", "high_stairs_inv"):
    sub_terrain = terrain_cfg.sub_terrains.get(terrain_name)
    if isinstance(
      sub_terrain,
      BoxPyramidStairsTerrainCfg | BoxInvertedPyramidStairsTerrainCfg,
    ):
      sub_terrain.step_height_range = (0.14, 0.14)

  return terrain_cfg


def _teacherkl_play_terrain_cfg() -> TerrainGeneratorCfg:
  terrain_cfg = deepcopy(BLIND_HIGH_STAIRS_TERRAINS_CFG)
  return _configure_high_stairs_play_terrain_generator(terrain_cfg)


def _configure_high_stairs_mixed_replay(cfg: ManagerBasedRlEnvCfg) -> None:
  """Enable high-level stair replay on the shared terrain curriculum."""
  terrain_levels = cfg.curriculum.get("terrain_levels")
  if terrain_levels is None:
    return

  params = terrain_levels.params
  params["mixed_replay_start_level"] = G1_HIGH_STAIRS_MIXED_REPLAY_START_LEVEL
  params["mixed_replay_level_ranges"] = G1_HIGH_STAIRS_MIXED_REPLAY_LEVEL_RANGES
  params["mixed_replay_weights"] = G1_HIGH_STAIRS_MIXED_REPLAY_WEIGHTS


def _configure_high_stairs_play_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
  """Match the legacy teacher-depth play terrain randomization."""
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    _configure_high_stairs_play_terrain_generator(cfg.scene.terrain.terrain_generator)
    cfg.scene.terrain.max_init_terrain_level = None

  randomize_reset = EventTermCfg(
    func=envs_mdp.randomize_terrain,
    mode="reset",
    params={},
  )
  cfg.events.pop("randomize_terrain_startup", None)
  cfg.events.pop("randomize_terrain", None)
  cfg.events = {
    "randomize_terrain": randomize_reset,
    **cfg.events,
  }


def _configure_perception_play_visualization(
  cfg: ManagerBasedRlEnvCfg,
  p: G1PerceptionRewardParams,
) -> None:
  cfg.viewer.show_depth_camera_visualizers = True

  # Hide teacher depth camera Frustum in play mode (student-only FOV).
  sensors = tuple(cfg.scene.sensors or ())
  cfg.scene.sensors = tuple(
    s
    for s in sensors
    if not (isinstance(s, CameraSensorCfg) and s.name == TEACHER_DEPTH_SENSOR)
  )

  for sensor in cfg.scene.sensors or ():
    if isinstance(sensor, RayCastSensorCfg):
      sensor.debug_vis = False

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.visualize_flat_patches = True
    cfg.scene.terrain.max_flat_patch_sites_per_tile = 80
    cfg.scene.terrain.terrain_generator.step_danger_visualization = (
      StepDangerVisualizationCfg(
        enabled=True,
        lip_radius=p.foot_lip_edge_radius,
        slab_depth=p.toe_slab_depth,
        slab_u_margin=p.toe_slab_u_margin,
        slab_v_margin=p.toe_slab_v_margin,
        geom_group=2,
        lip_rgba=G1_DANGER_ZONE_RGBA,
        slab_rgba=G1_DANGER_ZONE_RGBA,
      )
    )

  lip_reward = cfg.rewards.get("foot_step_lip_volume_penalty")
  if lip_reward is not None:
    lip_reward.params.update(
      {
        "debug_vis_foot_points": True,
        "debug_vis_foot_point_radius": 0.008,
        "debug_vis_foot_point_color": (0.0, 1.0, 0.15, 0.9),
        "debug_vis_step_danger_zones": False,
      }
    )


# ---------------------------------------------------------------------------
# Step-boundary danger-zone reward builder
# ---------------------------------------------------------------------------


def _configure_step_boundary_rewards(
  cfg: ManagerBasedRlEnvCfg,
  p: G1PerceptionRewardParams,
) -> None:
  """Add five step-boundary danger-zone rewards.

  Includes:
    - foot_step_lip_volume_penalty
    - toe_step_riser_slab_penalty (probe disabled)
    - heel_step_riser_clearance_penalty
    - foot_landing_flatness_penalty
    - shank_step_lip_proximity_penalty

  Does **not** add ``toe_riser_contact_memory_penalty``.
  """
  cfg.rewards["foot_step_lip_volume_penalty"] = RewardTermCfg(
    func=mdp.foot_step_lip_volume_penalty,
    weight=p.foot_lip_weight,
    params={
      "edge_radius": p.foot_lip_edge_radius,
      "edge_height_band": p.foot_lip_edge_height_band,
      "support_speed_floor": p.foot_lip_support_speed_floor,
      "nearest_boundaries": p.foot_lip_nearest_boundaries,
      "contact_sensor_name": "feet_ground_contact",
      "min_terrain_level": p.foot_lip_min_terrain_level,
      "asset_cfg": _g1_foot_asset_cfg(),
    },
  )
  cfg.rewards["toe_step_riser_slab_penalty"] = RewardTermCfg(
    func=mdp.toe_step_riser_slab_penalty,
    weight=p.toe_slab_weight,
    params={
      "slab_depth": p.toe_slab_depth,
      "u_margin": p.toe_slab_u_margin,
      "v_margin": p.toe_slab_v_margin,
      "toe_x_min": p.toe_x_min,
      "toe_v_threshold": p.toe_v_threshold,
      "surface_tol": p.surface_tol,
      "nearest_boundaries": p.nearest_boundaries,
      "min_terrain_level": p.min_terrain_level,
      "probe_contact_count": 0,
      "asset_cfg": _g1_foot_asset_cfg(),
    },
  )
  cfg.rewards["heel_step_riser_clearance_penalty"] = RewardTermCfg(
    func=mdp.heel_step_riser_clearance_penalty,
    weight=p.heel_clearance_weight,
    params={
      "heel_clearance": p.heel_clearance,
      "u_margin": p.heel_u_margin,
      "v_margin": p.heel_v_margin,
      "heel_x_max": p.heel_x_max,
      "surface_tol": p.surface_tol,
      "nearest_boundaries": p.nearest_boundaries,
      "contact_sensor_name": "feet_ground_contact",
      "min_terrain_level": p.min_terrain_level,
      "asset_cfg": _g1_foot_asset_cfg(),
    },
  )
  cfg.rewards["foot_landing_flatness_penalty"] = RewardTermCfg(
    func=mdp.foot_landing_flatness_penalty,
    weight=p.foot_landing_flatness_weight,
    params={
      "near_height": p.foot_landing_near_height,
      "max_tilt_deg": p.foot_landing_max_tilt_deg,
      "max_upward_speed": p.foot_landing_max_upward_speed,
      "height_sensor_name": "foot_height_scan",
      "contact_sensor_name": "feet_ground_contact",
      "min_terrain_level": p.min_terrain_level,
      "asset_cfg": _g1_foot_asset_cfg(),
    },
  )
  cfg.rewards["shank_step_lip_proximity_penalty"] = RewardTermCfg(
    func=mdp.shank_step_lip_proximity_penalty,
    weight=p.shank_lip_weight,
    params={
      "clearance_radius": p.shank_clearance_radius,
      "collision_radius": p.shank_collision_radius,
      "collision_weight": p.shank_collision_weight,
      "height_gain_threshold": p.shank_height_gain_threshold,
      "ascent_hold_steps": p.shank_ascent_hold_steps,
      "shank_tilt_threshold_deg": p.shank_tilt_threshold_deg,
      "nearest_boundaries": p.nearest_boundaries,
      "min_terrain_level": p.min_terrain_level,
      "shank_ref_local": (0.045, 0.0, -0.165),
      "shank_x_range": (0.045, 0.045),
      "shank_y_range": (-0.035, 0.035),
      "shank_z_range": (-0.23, -0.10),
      "shank_grid_shape": (1, 3, 5),
      "asset_cfg": SceneEntityCfg(
        "robot",
        body_names=("left_knee_link", "right_knee_link"),
      ),
    },
  )


# ---------------------------------------------------------------------------
# Target-navigation configuration
# ---------------------------------------------------------------------------


def _configure_target_navigation(
  cfg: ManagerBasedRlEnvCfg,
  p: G1PerceptionRewardParams,
  play: bool,
) -> None:
  """Switch to TeacherTargetHeading command and add target progress rewards."""
  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None
  cfg.scene.terrain.terrain_generator = _add_target_flat_patch_sampling(
    cfg.scene.terrain.terrain_generator
  )

  twist_cmd = TeacherTargetHeadingVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(30.0, 30.0),
    heading_command=True,
    heading_control_stiffness=0.5,
    rel_target_envs=0.8,
    rel_random_heading_envs=0.0,
    rel_standing_envs=0.2,
    patch_name="target",
    target_reached_threshold=0.5,
    target_min_distance=1.0,
    target_max_distance=12.0,
    target_tile_radius=1,
    include_current_tile=False,
    zero_lateral_velocity=True,
    debug_vis=True,
    ranges=TeacherTargetHeadingVelocityCommandCfg.Ranges(
      lin_vel_x=(0.0, 1.0),
      lin_vel_y=(0.0, 0.0),
      ang_vel_z=(-0.8, 0.8),
      heading=(-math.pi, math.pi),
    ),
  )
  twist_cmd.viz.z_offset = 1.15
  if play:
    twist_cmd.rel_target_envs = 1.0
    twist_cmd.rel_random_heading_envs = 0.0
    twist_cmd.rel_standing_envs = 0.0
    twist_cmd.heading_control_stiffness = 0.8
    twist_cmd.ranges.lin_vel_x = (0.3, 0.9)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)
    twist_cmd.target_min_distance = 1.0
    twist_cmd.target_max_distance = 10.0
    twist_cmd.target_tile_radius = 1

  cfg.commands["twist"] = twist_cmd
  cfg.rewards["target_progress"] = RewardTermCfg(
    func=teacher_target_progress,
    weight=p.target_progress_weight,
    params={"command_name": "twist", "min_distance": p.target_progress_min_distance},
  )
  cfg.rewards["target_reached_bonus"] = RewardTermCfg(
    func=teacher_target_reached_bonus,
    weight=p.target_reached_bonus_weight,
    params={"command_name": "twist"},
  )

  # Randomize step width per terrain tile within [0.25, 0.35].
  terrain_gen = cfg.scene.terrain.terrain_generator
  for name in ("high_stairs", "high_stairs_inv"):
    sub = terrain_gen.sub_terrains.get(name)
    if isinstance(
      sub,
      BoxPyramidStairsTerrainCfg | BoxInvertedPyramidStairsTerrainCfg,
    ):
      sub.step_width_range = (0.25, 0.35)


# ---------------------------------------------------------------------------
# Main env_cfg builder
# ---------------------------------------------------------------------------


def unitree_g1_blind_rough_perception_env_cfg(
  play: bool = False,
  use_target_navigation: bool = False,
  include_teacher: bool = True,
  reward_params: G1PerceptionRewardParams | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create a self-contained blind-rough student env with depth perception.

  The teacher observation group is aligned to the frozen teacher policy.
  The actor is blind (no height_scan) but receives depth camera input.

  Danger-zone rewards: ``foot_step_lip_volume_penalty`` +
  ``toe_step_riser_slab_penalty``.  ``toe_riser_contact_memory_penalty``
  is intentionally NOT used.

  Args:
      play: If ``True``, configure for evaluation (infinite episode, no
          curriculum, terrain randomization).
      use_target_navigation: If ``True``, replace uniform velocity commands
          with ``TeacherTargetHeadingVelocityCommandCfg`` and add target
          progress / target-reached bonus rewards.
      include_teacher: If ``True``, include frozen-teacher observation groups
          and checkpoint-compatible teacher camera inputs. Pure PPO variants
          set this to ``False`` and only keep the student depth stack.
      reward_params: All reward weights and parameters.  Uses defaults when
          ``None``.
  """
  p = reward_params or G1PerceptionRewardParams()
  cfg = make_velocity_env_cfg()

  # ------------------------------------------------------------------
  # Sim settings
  # ------------------------------------------------------------------
  cfg.sim.nconmax = 256
  cfg.sim.njmax = 4096
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 256

  # ------------------------------------------------------------------
  # Robot with student actuator delays
  # ------------------------------------------------------------------
  robot_cfg = get_g1_robot_cfg()
  assert robot_cfg.articulation is not None
  for actuator_cfg in robot_cfg.articulation.actuators:
    actuator_cfg.delay_min_lag = 0
    actuator_cfg.delay_max_lag = 2
    actuator_cfg.delay_hold_prob = 0.8
    actuator_cfg.delay_update_period = 5
  cfg.scene.entities = {"robot": robot_cfg}

  # ------------------------------------------------------------------
  # Sensor frame overrides
  # ------------------------------------------------------------------
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "pelvis"
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in ("left_foot", "right_foot")
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  # ------------------------------------------------------------------
  # Contact sensors
  # ------------------------------------------------------------------
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  # ------------------------------------------------------------------
  # Blind actor: remove height_scan from actor (keep for critic)
  # ------------------------------------------------------------------
  del cfg.observations["actor"].terms["height_scan"]

  # ------------------------------------------------------------------
  # Toe-terrain contact sensor & critic obs (needed by toe_slab_penalty)
  # ------------------------------------------------------------------
  _add_toe_terrain_contact_sensor(cfg)
  _add_toe_terrain_contact_critic_obs(cfg)

  # ------------------------------------------------------------------
  # Step-boundary danger-zone rewards (NO toe_riser_contact_memory_penalty)
  # ------------------------------------------------------------------
  _configure_step_boundary_rewards(cfg, p)

  # ------------------------------------------------------------------
  # Depth cameras: one checkpoint-compatible teacher input, one student stack.
  # ------------------------------------------------------------------
  if include_teacher:
    _add_teacher_depth_camera(cfg)
  _add_student_depth_camera_stack(cfg)

  # ------------------------------------------------------------------
  # Observation history lengths
  # ------------------------------------------------------------------
  cfg.observations["actor"].history_length = 8
  cfg.observations["critic"].history_length = 3

  # ------------------------------------------------------------------
  # Actor observation noise & delays
  # ------------------------------------------------------------------
  actor_terms = cfg.observations["actor"].terms
  for term_name in (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos_rel",
    "joint_vel_rel",
  ):
    actor_terms[term_name].delay_min_lag = 0
    actor_terms[term_name].delay_max_lag = 2
    actor_terms[term_name].delay_hold_prob = 0.8
    actor_terms[term_name].delay_update_period = 5

  actor_terms["base_ang_vel"].noise = Unoise(n_min=-0.3, n_max=0.3)
  actor_terms["projected_gravity"].noise = Unoise(n_min=-0.07, n_max=0.07)
  actor_terms["joint_pos_rel"].noise = Unoise(n_min=-0.015, n_max=0.015)
  actor_terms["joint_vel_rel"].noise = Unoise(n_min=-2.0, n_max=2.0)

  # ------------------------------------------------------------------
  # Terrain
  # ------------------------------------------------------------------
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.terrain_generator = (
      _teacherkl_play_terrain_cfg() if play else BLIND_HIGH_STAIRS_TERRAINS_CFG
    )
    cfg.scene.terrain.max_init_terrain_level = 2
    cfg.scene.terrain.visualize_flat_patches = False
    cfg.scene.terrain.max_flat_patch_sites_per_tile = None
  if not play:
    _configure_high_stairs_mixed_replay(cfg)

  # ------------------------------------------------------------------
  # Actions
  # ------------------------------------------------------------------
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  # ------------------------------------------------------------------
  # Command setup
  # ------------------------------------------------------------------
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15
  twist_cmd.standing_turns_to_heading = True

  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (0.0, 0.8),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.5, 0.5),
      },
      {
        "step": 3000 * 24,
        "lin_vel_x": (0.0, 1.0),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.8, 0.8),
      },
    ]

  if use_target_navigation:
    _configure_target_navigation(cfg, p, play=play)

  # ------------------------------------------------------------------
  # Domain randomization
  # ------------------------------------------------------------------
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # ------------------------------------------------------------------
  # Pose std
  # ------------------------------------------------------------------
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
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
  cfg.rewards["pose"].params["std_running"] = {
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

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # ------------------------------------------------------------------
  # Reward weights (shared proprioceptive / task rewards)
  # ------------------------------------------------------------------
  cfg.rewards["body_ang_vel"].weight = p.body_ang_vel_weight
  cfg.rewards["angular_momentum"].weight = p.angular_momentum_weight
  cfg.rewards["dof_pos_limits"].weight = p.dof_pos_limits_weight
  cfg.rewards["action_rate_l2"].weight = p.action_rate_l2_weight
  cfg.rewards["joint_acc_l2"] = RewardTermCfg(
    func=mdp.joint_acc_l2,
    weight=p.joint_acc_l2_weight,
  )
  cfg.rewards["action_acc_l2"] = RewardTermCfg(
    func=mdp.action_acc_l2,
    weight=p.action_acc_l2_weight,
  )

  cfg.rewards["track_linear_velocity"].weight = p.track_linear_velocity_weight
  cfg.rewards["track_linear_velocity"].params["std"] = p.track_linear_velocity_std
  cfg.rewards["track_angular_velocity"].weight = p.track_angular_velocity_weight
  cfg.rewards["track_angular_velocity"].params["std"] = p.track_angular_velocity_std
  cfg.rewards["upright"].weight = p.upright_weight
  cfg.rewards["upright"].params["std"] = p.upright_std
  cfg.rewards["pose"].weight = p.pose_weight
  cfg.rewards["foot_clearance"].weight = p.foot_clearance_weight
  cfg.rewards["foot_swing_height"].weight = p.foot_swing_height_weight
  cfg.rewards["foot_swing_height"].params["command_threshold"] = (
    p.foot_swing_height_command_threshold
  )
  cfg.rewards["foot_slip"].weight = p.foot_slip_weight
  cfg.rewards["soft_landing"].weight = p.soft_landing_weight
  cfg.rewards["idle_penalty"].weight = p.idle_penalty_weight
  cfg.rewards["foot_gait"].weight = p.foot_gait_weight
  cfg.rewards["base_height_above_support"].weight = p.base_height_above_support_weight

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=p.self_collisions_weight,
    params={
      "sensor_name": self_collision_cfg.name,
      "force_threshold": p.self_collisions_force_threshold,
    },
  )

  # ------------------------------------------------------------------
  # Teacher observation group
  # ------------------------------------------------------------------
  if include_teacher:
    cfg.observations["teacher"] = ObservationGroupCfg(
      terms=_make_teacher_terms(cfg),
      concatenate_terms=True,
      enable_corruption=False,
      history_length=None,
      flatten_history_dim=True,
    )

  # ------------------------------------------------------------------
  # Play mode overrides
  # ------------------------------------------------------------------
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    _configure_high_stairs_play_randomization(cfg)
    _configure_perception_play_visualization(cfg, p)
    if not use_target_navigation:
      twist_cmd = cfg.commands["twist"]
      assert isinstance(twist_cmd, UniformVelocityCommandCfg)
      twist_cmd.ranges.lin_vel_x = (0.5, 1.0)
      twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
      twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def unitree_g1_blind_rough_perception_ppo_env_cfg(
  play: bool = False,
  use_target_navigation: bool = False,
  reward_params: G1PerceptionRewardParams | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create the perception env for pure PPO, without teacher-only inputs."""
  cfg = unitree_g1_blind_rough_perception_env_cfg(
    play=play,
    use_target_navigation=use_target_navigation,
    include_teacher=False,
    reward_params=reward_params,
  )
  return cfg
