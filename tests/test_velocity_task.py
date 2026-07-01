"""Tests specific to velocity tasks."""

from typing import cast

import pytest

import mjlab.tasks  # noqa: F401
from mjlab.asset_zoo.robots import G1_ACTION_SCALE, GO1_ACTION_SCALE
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.rl.config import (
  RslRlGatedStairLatentModelCfg,
  RslRlPpoTeacherKLAlgorithmCfg,
  RslRlTeacherKLRunnerCfg,
)
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.g1.blind_rough_slow_latent_env_cfg import (
  G1SlowLatentEnvParams,
  G1SlowLatentPlayVisualizationParams,
  G1SlowLatentRewardParams,
  G1SlowLatentTerrainReplayParams,
  unitree_g1_blind_rough_target_navigation_slow_latent_env_cfg,
)
from mjlab.tasks.velocity.config.g1.blind_rough_step_danger_env_cfg import (
  G1StepDangerEnvParams,
  G1StepDangerFootLipRewardParams,
  G1StepDangerPlayVisualizationParams,
  G1StepDangerRewardParams,
  G1StepDangerToeRiserSlabPenaltyParams,
  unitree_g1_blind_rough_target_navigation_step_danger_env_cfg,
)
from mjlab.tasks.velocity.config.g1.rl_cfg import (
  G1SlowLatentPolicyModelParams,
  G1SlowLatentRunnerParams,
  unitree_g1_blind_rough_target_navigation_slow_latent_teacherkl_runner_cfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg, stair_aware_feet_gait
from mjlab.tasks.velocity.mdp.teacher_target_heading_command import (
  TeacherTargetHeadingVelocityCommandCfg,
)
from mjlab.terrains.config import BLIND_HIGH_STAIRS_TREAD_DEPTHS
from mjlab.terrains.primitive_terrains import (
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
)

MAIN_BRANCH_VELOCITY_TASK_IDS = (
  "Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1",
  "Mjlab-Velocity-Blind-Rough-TargetNavigation-StepDanger-TeacherKL-Unitree-G1",
  "Mjlab-Velocity-Blind-Rough-TargetNavigation-TeacherKL-Unitree-G1",
  "Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1",
)


@pytest.fixture(scope="module")
def velocity_task_ids() -> list[str]:
  """Get all velocity task IDs."""
  return [t for t in list_tasks() if "Velocity" in t]


def test_velocity_registry_contains_only_main_branch_tasks(
  velocity_task_ids: list[str],
) -> None:
  assert velocity_task_ids == sorted(MAIN_BRANCH_VELOCITY_TASK_IDS)


@pytest.fixture(scope="module")
def g1_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all G1 velocity task IDs."""
  return [t for t in velocity_task_ids if "G1" in t]


@pytest.fixture(scope="module")
def go1_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all Go1 velocity task IDs."""
  return [t for t in velocity_task_ids if "Go1" in t]


@pytest.fixture(scope="module")
def rough_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all rough terrain velocity task IDs."""
  return [t for t in velocity_task_ids if "Rough" in t]


@pytest.fixture(scope="module")
def flat_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all flat terrain velocity task IDs."""
  return [t for t in velocity_task_ids if "Flat" in t]


def test_velocity_tasks_have_twist_command(velocity_task_ids: list[str]) -> None:
  """All velocity tasks should have a velocity command."""
  for task_id in velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "twist" in cfg.commands, f"Task {task_id} missing 'twist' command"

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg), (
      f"Task {task_id} twist command is not UniformVelocityCommandCfg"
    )


def test_g1_velocity_has_required_sensors(g1_velocity_task_ids: list[str]) -> None:
  """G1 velocity tasks should have feet/ground and self collision sensors."""
  for task_id in g1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "feet_ground_contact" in sensor_names, (
      f"Task {task_id} missing feet_ground_contact sensor"
    )
    assert "self_collision" in sensor_names, (
      f"Task {task_id} missing self_collision sensor"
    )


def test_go1_velocity_has_required_sensors(go1_velocity_task_ids: list[str]) -> None:
  """Go1 velocity tasks should have feet/ground and collision sensors."""
  for task_id in go1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "feet_ground_contact" in sensor_names, (
      f"Task {task_id} missing feet_ground_contact sensor"
    )
    if "Rough" in task_id:
      for name in (
        "self_collision",
        "thigh_ground_touch",
        "shank_ground_touch",
        "trunk_ground_touch",
      ):
        assert name in sensor_names, f"Task {task_id} missing {name} sensor"


def test_flat_velocity_tasks_have_plane_terrain(
  flat_velocity_task_ids: list[str],
) -> None:
  """Flat velocity tasks should have terrain_type='plane' and no terrain_generator."""
  for task_id in flat_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_type == "plane", (
      f"Task {task_id} terrain_type={cfg.scene.terrain.terrain_type}, expected 'plane'"
    )
    assert cfg.scene.terrain.terrain_generator is None, (
      f"Task {task_id} has terrain_generator, expected None for flat terrain"
    )


def test_rough_velocity_tasks_have_generator_terrain(
  rough_velocity_task_ids: list[str],
) -> None:
  """Rough velocity tasks should have generator terrain."""
  for task_id in rough_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_type == "generator", (
      f"Task {task_id} terrain_type={cfg.scene.terrain.terrain_type}, "
      "expected 'generator'"
    )
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} has no terrain_generator, expected one for rough terrain"
    )


def test_rough_velocity_training_has_curriculum_enabled() -> None:
  """Rough velocity training tasks should have terrain curriculum enabled."""
  for task_id in MAIN_BRANCH_VELOCITY_TASK_IDS:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} has no terrain_generator"
    )
    assert cfg.scene.terrain.terrain_generator.curriculum is True, (
      f"Task {task_id} curriculum={cfg.scene.terrain.terrain_generator.curriculum}, "
      "expected True"
    )


def test_rough_velocity_play_keeps_curriculum_rows_for_mixed_distribution() -> None:
  """Play tasks keep curriculum rows so low/mid/high sampling is meaningful."""
  for task_id in MAIN_BRANCH_VELOCITY_TASK_IDS:
    cfg = load_env_cfg(task_id, play=True)

    assert cfg.scene.terrain is not None, (
      f"Task {task_id} (play mode) has no terrain config"
    )
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} (play mode) has no terrain_generator"
    )
    assert cfg.scene.terrain.terrain_generator.curriculum is True, (
      f"Task {task_id} (play mode) curriculum={cfg.scene.terrain.terrain_generator.curriculum}, "
      "expected True"
    )


def test_g1_velocity_has_correct_action_scale(g1_velocity_task_ids: list[str]) -> None:
  """G1 velocity tasks should use G1_ACTION_SCALE."""
  for task_id in g1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == G1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected G1_ACTION_SCALE"
    )


def test_go1_velocity_has_correct_action_scale(
  go1_velocity_task_ids: list[str],
) -> None:
  """Go1 velocity tasks should use GO1_ACTION_SCALE."""
  for task_id in go1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == GO1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected GO1_ACTION_SCALE"
    )


def test_teacherkl_target_navigation_switch() -> None:
  """Teacher-KL should support both velocity and target-navigation commands."""
  velocity_cfg = load_env_cfg("Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1")
  target_cfg = load_env_cfg(
    "Mjlab-Velocity-Blind-Rough-TargetNavigation-TeacherKL-Unitree-G1"
  )

  assert isinstance(velocity_cfg.commands["twist"], UniformVelocityCommandCfg)
  assert not isinstance(
    velocity_cfg.commands["twist"],
    TeacherTargetHeadingVelocityCommandCfg,
  )
  assert isinstance(
    target_cfg.commands["twist"], TeacherTargetHeadingVelocityCommandCfg
  )

  assert "target_progress" not in velocity_cfg.rewards
  assert "target_reached_bonus" not in velocity_cfg.rewards
  assert "target_progress" in target_cfg.rewards
  assert "target_reached_bonus" in target_cfg.rewards
  assert "height_scan" not in target_cfg.observations["actor"].terms
  assert "toe_terrain_contact" in target_cfg.observations["critic"].terms


def test_g1_high_stairs_tasks_enable_mixed_terrain_replay() -> None:
  for task_id in MAIN_BRANCH_VELOCITY_TASK_IDS:
    cfg = load_env_cfg(task_id)
    params = cfg.curriculum["terrain_levels"].params
    assert params["mixed_replay_start_level"] == 8
    assert params["mixed_replay_level_ranges"] == ((0, 2), (3, 5), (6, 9))
    assert params["mixed_replay_weights"] == (0.2, 0.3, 0.5)


def _assert_eight_stair_depth_variants(terrain_generator) -> None:
  sub_terrains = terrain_generator.sub_terrains
  upward = [
    cfg for name, cfg in sub_terrains.items() if name.startswith("high_stairs_w")
  ]
  inverted = [
    cfg for name, cfg in sub_terrains.items() if name.startswith("high_stairs_inv_w")
  ]
  stair_variants = {
    int(name.rsplit("_w", 1)[1]): cfg
    for name, cfg in sub_terrains.items()
    if name.startswith(("high_stairs_w", "high_stairs_inv_w"))
  }

  assert len(upward) == 0
  assert len(inverted) == 8
  assert len(stair_variants) == 8
  assert all(isinstance(cfg, BoxPyramidStairsTerrainCfg) for cfg in upward)
  assert all(isinstance(cfg, BoxInvertedPyramidStairsTerrainCfg) for cfg in inverted)
  assert [stair_variants[index].step_width for index in range(8)] == pytest.approx(
    BLIND_HIGH_STAIRS_TREAD_DEPTHS
  )
  assert BLIND_HIGH_STAIRS_TREAD_DEPTHS[0] == pytest.approx(0.25)
  assert BLIND_HIGH_STAIRS_TREAD_DEPTHS[-1] == pytest.approx(0.35)
  assert all(cfg.step_width_range is None for cfg in stair_variants.values())
  assert sum(cfg.proportion for cfg in upward) == pytest.approx(0.0)
  assert sum(cfg.proportion for cfg in inverted) == pytest.approx(0.85)


def test_g1_high_stairs_tasks_cover_eight_step_depths_per_level() -> None:
  for task_id in MAIN_BRANCH_VELOCITY_TASK_IDS:
    cfg = load_env_cfg(task_id)
    assert cfg.scene.terrain is not None
    terrain_generator = cfg.scene.terrain.terrain_generator
    assert terrain_generator is not None
    assert terrain_generator.num_cols == len(terrain_generator.sub_terrains) == 11
    _assert_eight_stair_depth_variants(terrain_generator)


def test_g1_high_stairs_play_uses_mixed_level_distribution() -> None:
  for task_id in MAIN_BRANCH_VELOCITY_TASK_IDS:
    cfg = load_env_cfg(task_id, play=True)
    assert cfg.scene.terrain is not None
    terrain_generator = cfg.scene.terrain.terrain_generator
    assert terrain_generator is not None
    assert terrain_generator.curriculum is True
    assert terrain_generator.num_rows == 10

    randomize = cfg.events["randomize_terrain"]
    assert randomize.params["level_ranges"] == ((0, 2), (3, 5), (6, 9))
    assert randomize.params["level_weights"] == (0.2, 0.3, 0.5)
    assert randomize.params["use_sub_terrain_proportions"] is True
    assert cfg.events["randomize_terrain_startup"].mode == "startup"
    assert cfg.events["randomize_terrain_startup"].params == randomize.params
    event_order = list(cfg.events)
    assert event_order.index("randomize_terrain") < event_order.index("reset_base")
    assert cfg.scene.terrain.max_init_terrain_level is None

    _assert_eight_stair_depth_variants(terrain_generator)


def test_slow_latent_target_navigation_exposes_latent_inputs() -> None:
  task_id = (
    "Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1"
  )
  env_cfg = load_env_cfg(task_id)
  rl_cfg = cast(RslRlTeacherKLRunnerCfg, load_rl_cfg(task_id))

  assert env_cfg.commands["twist"].ranges.lin_vel_x == (0.4, 1.0)
  velocity_stages = env_cfg.curriculum["command_vel"].params["velocity_stages"]
  assert velocity_stages[0]["lin_vel_x"] == (0.4, 0.8)
  assert velocity_stages[1]["lin_vel_x"] == (0.4, 1.0)
  assert "latent" in env_cfg.observations
  assert "stair_latent" in env_cfg.observations["latent"].terms
  assert "latent_labels" in env_cfg.observations
  assert tuple(env_cfg.observations["latent_labels"].terms) == (
    "toe_riser_event",
    "stair_state",
    "stair_shape",
    "safe_stride",
    "future_events",
  )
  assert "reset_stair_latent_cache" in env_cfg.events
  assert rl_cfg.obs_groups["actor"] == ("actor",)
  assert rl_cfg.obs_groups["latent"] == ("latent",)
  foot_asset_cfg = env_cfg.rewards["foot_step_lip_volume_penalty"].params["asset_cfg"]
  toe_reward_params = env_cfg.rewards["toe_step_riser_slab_penalty"].params
  toe_asset_cfg = toe_reward_params["asset_cfg"]
  assert foot_asset_cfg is not toe_asset_cfg
  assert (
    env_cfg.rewards["foot_step_lip_volume_penalty"].params["ignore_boundary_layers"]
    == 0
  )
  assert toe_reward_params["contact_sensor_name"] == "toe_terrain_contact"
  assert toe_reward_params["contact_penalty_scale"] == 0.5
  assert toe_reward_params["event_min_forward_intent_speed"] == 0.04
  assert toe_reward_params["event_min_forward_speed_drop"] == 0.04
  assert toe_reward_params["event_blocked_forward_speed"] == 0.03
  assert toe_reward_params["event_persistent_steps"] == 3
  assert toe_reward_params["stair_heading_cos"] == 0.70
  assert toe_reward_params["stair_touchdown_height_tolerance"] == 0.08
  assert toe_reward_params["stair_touchdown_lateral_margin"] == 0.03
  assert toe_reward_params["stair_min_safe_stride"] == 0.10
  assert toe_reward_params["stair_max_safe_stride"] == 0.55
  assert toe_reward_params["stair_max_tracking_stride"] == 0.80
  assert toe_reward_params["stair_touchdown_lip_clearance"] == 0.02
  assert toe_reward_params["stair_touchdown_lip_height_band"] == 0.06
  assert toe_reward_params["safe_stride_containment_margin"] == 0.003
  assert toe_reward_params["stair_entry_evidence_time"] == 0.80
  assert toe_reward_params["stair_attempt_period"] == 0.60
  assert toe_reward_params["safe_stride_evidence_window_steps"] == 15
  assert toe_reward_params["safe_stride_rear_partial_weight"] == 2.0
  assert toe_reward_params["safe_stride_riser_weight"] == 3.0
  assert toe_reward_params["collision_risk_margin"] == 0.06
  assert not any("probe" in name for name in toe_reward_params)
  assert toe_reward_params["ground_contact_sensor_name"] == "feet_ground_contact"
  shank_reward = env_cfg.rewards["shank_front_edge_clearance_penalty"]
  shank_params = shank_reward.params
  assert shank_reward.weight == -2.0
  assert shank_params["clearance_margin"] == 0.05
  assert shank_params["min_riser_height"] == 0.04
  assert shank_params["direction_cos_threshold"] == 0.85
  assert shank_params["lateral_margin"] == 0.05
  assert shank_params["capsule_start_local"] == (0.01, 0.0, 0.0)
  assert shank_params["capsule_end_local"] == (0.01, 0.0, -0.15)
  assert shank_params["capsule_radius"] == 0.045
  assert shank_params["asset_cfg"].body_names == (
    "left_knee_link",
    "right_knee_link",
  )
  assert shank_params["asset_cfg"].preserve_order is True
  skip_reward = env_cfg.rewards["stair_skip_layer_penalty"]
  assert skip_reward.weight == -1.0
  algorithm_cfg = cast(RslRlPpoTeacherKLAlgorithmCfg, rl_cfg.algorithm)
  assert algorithm_cfg.teacher_kl_cfg.log_kl_when_lambda_zero is False
  foot_gait = env_cfg.rewards["foot_gait"]
  assert foot_gait.func is stair_aware_feet_gait
  assert foot_gait.params["heading_cos"] == 0.70
  landing_reward = env_cfg.rewards["stair_tread_landing_reward"]
  assert landing_reward.weight == 0.5
  assert landing_reward.params["landing_lead"] == 0.04
  assert landing_reward.params["heading_cos"] == 0.70
  assert landing_reward.params["min_landing_layer"] == 3
  assert landing_reward.params["lip_edge_radius"] == 0.07
  assert landing_reward.params["lip_margin"] == 0.01
  assert landing_reward.params["foot_body_cfg"].body_names == (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )
  assert landing_reward.params["support_deficit_scale"] == 0.40
  latent_labels = env_cfg.observations["latent_labels"]
  assert all(term.params == {} for term in latent_labels.terms.values())
  toe_sensor_cfg = cast(
    ContactSensorCfg,
    next(
      sensor
      for sensor in env_cfg.scene.sensors or ()
      if sensor.name == "toe_terrain_contact"
    ),
  )
  assert toe_sensor_cfg.track_air_time is True
  actor_cfg = cast(RslRlGatedStairLatentModelCfg, rl_cfg.actor)
  assert actor_cfg.latent_dim == 16
  assert actor_cfg.state_latent_dim == 8
  assert actor_cfg.latent_hidden_dim == 128
  assert actor_cfg.mlp_encoder_dims == (128, 128)
  assert actor_cfg.alpha_hold_state == 0.0
  assert actor_cfg.alpha_hold_shape == 0.05
  assert actor_cfg.memory_event_shape_boost_steps == 15
  assert actor_cfg.write_steps == 6
  assert actor_cfg.stair_confirm_steps == 3
  assert actor_cfg.event_on_threshold == 0.60
  assert actor_cfg.event_off_threshold == 0.20
  assert actor_cfg.aux_event_coef == 0.03
  assert actor_cfg.aux_event_pos_weight == 50.0
  assert actor_cfg.event_label_window_steps == 4
  assert actor_cfg.min_stair_steps == 30
  assert actor_cfg.exit_steps == 40
  assert actor_cfg.stair_on_threshold == 0.35
  assert actor_cfg.stair_off_threshold == 0.20
  assert actor_cfg.aux_stair_coef == 0.05
  assert actor_cfg.aux_stair_pos_weight == 3.0
  assert actor_cfg.aux_future_collision_risk_coef == 0.03
  assert actor_cfg.aux_future_safe_landing_quality_coef == 0.03
  assert actor_cfg.future_horizon == 20
  assert actor_cfg.aux_stair_shape_coef == 0.0
  assert actor_cfg.aux_safe_stride_coef == 0.03
  assert actor_cfg.stair_shape_huber_delta == 0.05
  assert actor_cfg.safe_stride_huber_delta == 0.05
  assert actor_cfg.safe_stride_min == 0.10
  assert actor_cfg.safe_stride_max == 0.55


def test_step_danger_target_navigation_uses_local_geometric_danger_rewards() -> None:
  task_id = (
    "Mjlab-Velocity-Blind-Rough-TargetNavigation-StepDanger-TeacherKL-Unitree-G1"
  )
  env_cfg = load_env_cfg(task_id)
  rl_cfg = cast(RslRlTeacherKLRunnerCfg, load_rl_cfg(task_id))

  assert isinstance(env_cfg.commands["twist"], TeacherTargetHeadingVelocityCommandCfg)
  assert "latent" not in env_cfg.observations
  assert "latent_labels" not in env_cfg.observations
  assert "reset_stair_latent_cache" not in env_cfg.events
  assert rl_cfg.obs_groups == {
    "actor": ("actor",),
    "critic": ("critic",),
    "teacher": ("teacher", "camera"),
  }
  assert not isinstance(rl_cfg.actor, RslRlGatedStairLatentModelCfg)
  assert rl_cfg.num_steps_per_env == 24
  assert rl_cfg.experiment_name == (
    "g1_blind_rough_target_navigation_step_danger_teacherkl"
  )

  foot_params = env_cfg.rewards["foot_step_lip_volume_penalty"].params
  toe_reward = env_cfg.rewards["toe_step_riser_slab_penalty"]
  toe_params = toe_reward.params
  assert env_cfg.rewards["foot_step_lip_volume_penalty"].weight == -3.2
  assert "toe_step_riser_probe_shaping_reward" not in env_cfg.rewards
  assert toe_reward.weight == -4.2
  assert foot_params["edge_radius"] == 0.07
  assert "ignore_boundary_layers" not in foot_params
  assert toe_params["slab_depth"] == 0.10
  assert toe_params["v_margin"] == 0.05
  assert toe_params["toe_x_min"] == 0.08
  assert toe_params["toe_v_threshold"] == 0.02
  assert not any("probe" in name for name in toe_params)
  assert "contact_sensor_name" not in toe_params
  assert foot_params["asset_cfg"] is not toe_params["asset_cfg"]
  assert "toe_terrain_contact" in env_cfg.observations["critic"].terms
  assert "toe_terrain_contact_forces" in env_cfg.observations["critic"].terms

  play_cfg = load_env_cfg(task_id, play=True)
  assert play_cfg.scene.terrain is not None
  assert play_cfg.scene.terrain.terrain_generator is not None
  vis = play_cfg.scene.terrain.terrain_generator.step_danger_visualization
  assert vis.enabled is True
  assert vis.lip_radius == 0.07
  assert vis.slab_depth == 0.10
  assert vis.slab_u_margin == 0.02
  assert vis.slab_v_margin == 0.05


def test_blind_teacherkl_play_hides_exteroceptive_visualizers() -> None:
  task_ids = (
    "Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1",
    "Mjlab-Velocity-Blind-Rough-TargetNavigation-TeacherKL-Unitree-G1",
    "Mjlab-Velocity-Blind-Rough-TargetNavigation-StepDanger-TeacherKL-Unitree-G1",
    "Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1",
  )

  for task_id in task_ids:
    cfg = load_env_cfg(task_id, play=True)
    assert cfg.viewer.show_depth_camera_visualizers is False
    for sensor in cfg.scene.sensors or ():
      debug_vis = getattr(sensor, "debug_vis", None)
      if debug_vis is not None:
        assert debug_vis is False


def test_slow_latent_play_shows_step_danger_zones() -> None:
  cfg = load_env_cfg(
    "Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1",
    play=True,
  )

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None
  vis = cfg.scene.terrain.terrain_generator.step_danger_visualization
  assert vis.enabled is True
  assert vis.lip_radius == 0.07
  assert vis.slab_depth == 0.10
  assert vis.slab_u_margin == 0.02
  assert vis.slab_v_margin == 0.05
  assert cfg.commands["twist"].ranges.lin_vel_x == (0.6, 0.9)


def test_slow_latent_explicit_param_interfaces_drive_configs() -> None:
  env_params = G1SlowLatentEnvParams(
    actor_history_length=7,
    rewards=G1SlowLatentRewardParams(
      track_linear_velocity_weight=1.7,
      track_linear_velocity_std=0.6,
      foot_clearance_weight=-1.4,
      foot_gait_period=0.7,
      base_height_above_support_min_height=0.76,
      self_collision_force_threshold=12.0,
      action_acc_l2_weight=-0.07,
      foot_lip_edge_radius=0.08,
      foot_lip_ignore_boundary_layers=1,
      toe_slab_depth=0.12,
      toe_contact_penalty_scale=0.7,
      toe_stair_min_safe_stride=0.14,
      stair_entry_evidence_time=0.9,
    ),
    terrain_replay=G1SlowLatentTerrainReplayParams(
      start_level=7,
      level_ranges=((0, 1), (2, 4), (5, 9)),
      weights=(0.1, 0.2, 0.7),
    ),
    play_visualization=G1SlowLatentPlayVisualizationParams(
      danger_lip_radius=0.09,
      danger_slab_depth=0.11,
      danger_geom_group=5,
    ),
  )
  env_cfg = unitree_g1_blind_rough_target_navigation_slow_latent_env_cfg(
    play=True,
    params=env_params,
  )
  train_env_cfg = unitree_g1_blind_rough_target_navigation_slow_latent_env_cfg(
    params=env_params,
  )

  assert env_cfg.observations["actor"].history_length == 7
  assert env_cfg.rewards["track_linear_velocity"].weight == 1.7
  assert env_cfg.rewards["track_linear_velocity"].params["std"] == 0.6
  assert env_cfg.rewards["foot_clearance"].weight == -1.4
  assert env_cfg.rewards["foot_gait"].params["period"] == 0.7
  assert env_cfg.rewards["base_height_above_support"].params["min_height"] == 0.76
  assert env_cfg.rewards["self_collisions"].params["force_threshold"] == 12.0
  assert env_cfg.rewards["action_acc_l2"].weight == -0.07
  assert env_cfg.rewards["foot_step_lip_volume_penalty"].params["edge_radius"] == 0.08
  assert (
    env_cfg.rewards["foot_step_lip_volume_penalty"].params["ignore_boundary_layers"]
    == 1
  )
  assert env_cfg.rewards["toe_step_riser_slab_penalty"].params["slab_depth"] == 0.12
  toe_params = env_cfg.rewards["toe_step_riser_slab_penalty"].params
  assert toe_params["contact_penalty_scale"] == 0.7
  assert toe_params["stair_min_safe_stride"] == 0.14
  assert toe_params["stair_max_safe_stride"] == 0.55
  assert toe_params["stair_entry_evidence_time"] == 0.9
  replay_params = train_env_cfg.curriculum["terrain_levels"].params
  assert replay_params["mixed_replay_start_level"] == 7
  assert replay_params["mixed_replay_level_ranges"] == ((0, 1), (2, 4), (5, 9))
  assert replay_params["mixed_replay_weights"] == (0.1, 0.2, 0.7)
  assert env_cfg.scene.terrain is not None
  assert env_cfg.scene.terrain.terrain_generator is not None
  vis = env_cfg.scene.terrain.terrain_generator.step_danger_visualization
  assert vis.lip_radius == 0.09
  assert vis.slab_depth == 0.11
  assert vis.geom_group == 5

  rl_params = G1SlowLatentRunnerParams(
    num_steps_per_env=32,
    save_interval=250,
    max_iterations=1234,
    experiment_name="slow_latent_custom",
    model=G1SlowLatentPolicyModelParams(
      hidden_dims=(256, 128),
      latent_dim=8,
      state_latent_dim=4,
      latent_hidden_dim=64,
      mlp_encoder_dims=(64,),
      alpha_hold_state=0.006,
      alpha_hold_shape=0.07,
      stair_confirm_steps=3,
      stair_on_threshold=0.2,
      stair_off_threshold=0.05,
      aux_event_pos_weight=42.0,
      event_label_window_steps=7,
      aux_stair_coef=0.7,
      aux_stair_pos_weight=4.0,
      aux_safe_stride_coef=0.11,
      aux_future_collision_risk_coef=0.12,
      aux_future_safe_landing_quality_coef=0.13,
      future_horizon=18,
      safe_stride_min=0.10,
      safe_stride_max=0.42,
    ),
  )
  rl_cfg = unitree_g1_blind_rough_target_navigation_slow_latent_teacherkl_runner_cfg(
    params=rl_params,
  )
  actor_cfg = cast(RslRlGatedStairLatentModelCfg, rl_cfg.actor)
  assert rl_cfg.num_steps_per_env == 32
  assert rl_cfg.save_interval == 250
  assert rl_cfg.max_iterations == 1234
  assert rl_cfg.experiment_name == "slow_latent_custom"
  assert actor_cfg.hidden_dims == (256, 128)
  assert actor_cfg.latent_dim == 8
  assert actor_cfg.state_latent_dim == 4
  assert actor_cfg.latent_hidden_dim == 64
  assert actor_cfg.mlp_encoder_dims == (64,)
  assert actor_cfg.alpha_hold_state == 0.006
  assert actor_cfg.alpha_hold_shape == 0.07
  assert actor_cfg.stair_confirm_steps == 3
  assert actor_cfg.stair_on_threshold == 0.2
  assert actor_cfg.stair_off_threshold == 0.05
  assert actor_cfg.aux_event_pos_weight == 42.0
  assert actor_cfg.event_label_window_steps == 7
  assert actor_cfg.aux_stair_coef == 0.7
  assert actor_cfg.aux_stair_pos_weight == 4.0
  assert actor_cfg.aux_safe_stride_coef == 0.11
  assert actor_cfg.aux_future_collision_risk_coef == 0.12
  assert actor_cfg.aux_future_safe_landing_quality_coef == 0.13
  assert actor_cfg.future_horizon == 18
  assert actor_cfg.safe_stride_min == 0.10
  assert actor_cfg.safe_stride_max == 0.42


def test_step_danger_explicit_param_interfaces_drive_configs() -> None:
  env_params = G1StepDangerEnvParams(
    rewards=G1StepDangerRewardParams(
      foot_lip=G1StepDangerFootLipRewardParams(edge_radius=0.08),
      toe_riser_slab=G1StepDangerToeRiserSlabPenaltyParams(
        weight=-0.8,
        slab_depth=0.12,
        v_margin=0.07,
        toe_x_min=0.10,
        toe_v_threshold=0.04,
      ),
    ),
    play_visualization=G1StepDangerPlayVisualizationParams(
      danger_lip_radius=0.09,
      danger_slab_depth=0.11,
      danger_geom_group=5,
    ),
  )
  env_cfg = unitree_g1_blind_rough_target_navigation_step_danger_env_cfg(
    play=True,
    params=env_params,
  )

  foot_params = env_cfg.rewards["foot_step_lip_volume_penalty"].params
  toe_reward = env_cfg.rewards["toe_step_riser_slab_penalty"]
  toe_params = toe_reward.params
  assert foot_params["edge_radius"] == 0.08
  assert "ignore_boundary_layers" not in foot_params
  assert "toe_step_riser_probe_shaping_reward" not in env_cfg.rewards
  assert toe_reward.weight == -0.8
  assert toe_params["slab_depth"] == 0.12
  assert toe_params["v_margin"] == 0.07
  assert toe_params["toe_x_min"] == 0.10
  assert toe_params["toe_v_threshold"] == 0.04
  assert "contact_penalty_scale" not in toe_params
  assert not any("probe" in name for name in toe_params)
  assert env_cfg.scene.terrain is not None
  assert env_cfg.scene.terrain.terrain_generator is not None
  vis = env_cfg.scene.terrain.terrain_generator.step_danger_visualization
  assert vis.lip_radius == 0.09
  assert vis.slab_depth == 0.11
  assert vis.geom_group == 5


def test_blind_rough_variants_share_toe_riser_contact_penalty() -> None:
  """Blind-rough variants should use the shared toe-riser contact penalty config."""
  # Non-target TeacherKL variant keeps the old penalty.
  cfg = load_env_cfg("Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1")
  assert "toe_riser_contact_memory_penalty" in cfg.rewards
  assert "foot_step_lip_volume_penalty" not in cfg.rewards
  assert "toe_step_riser_slab_penalty" not in cfg.rewards

  # Target-navigation TeacherKL variant replaces it with step-boundary
  # volume penalties but keeps the contact sensor for the critic.
  cfg = load_env_cfg("Mjlab-Velocity-Blind-Rough-TargetNavigation-TeacherKL-Unitree-G1")
  assert "toe_riser_contact_memory_penalty" not in cfg.rewards
  assert "foot_step_lip_volume_penalty" in cfg.rewards
  assert "toe_step_riser_slab_penalty" in cfg.rewards
  assert cfg.rewards["foot_step_lip_volume_penalty"].weight == -3.2
  assert cfg.rewards["toe_step_riser_slab_penalty"].weight == -4.2
  assert cfg.sim.contact_sensor_maxmatch == 256
  assert "toe_terrain_contact" not in cfg.observations["actor"].terms
  assert "toe_terrain_contact" in cfg.observations["critic"].terms
  assert "toe_terrain_contact_forces" in cfg.observations["critic"].terms


def test_teacherkl_uses_mean_huber_guidance() -> None:
  """Teacher-KL variants should use weak action-mean guidance."""
  velocity_cfg = cast(
    RslRlTeacherKLRunnerCfg,
    load_rl_cfg("Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1"),
  )
  target_cfg = cast(
    RslRlTeacherKLRunnerCfg,
    load_rl_cfg("Mjlab-Velocity-Blind-Rough-TargetNavigation-TeacherKL-Unitree-G1"),
  )
  step_danger_cfg = cast(
    RslRlTeacherKLRunnerCfg,
    load_rl_cfg(
      "Mjlab-Velocity-Blind-Rough-TargetNavigation-StepDanger-TeacherKL-Unitree-G1"
    ),
  )

  for cfg in (velocity_cfg, target_cfg, step_danger_cfg):
    algorithm_cfg = cast(RslRlPpoTeacherKLAlgorithmCfg, cfg.algorithm)
    teacher_cfg = algorithm_cfg.teacher_kl_cfg
    assert cfg.obs_groups["teacher"] == ("teacher", "camera")
    assert teacher_cfg.enabled is True
    assert teacher_cfg.imitation_only is False
    assert teacher_cfg.imitation_loss_coef == 1.0
    assert teacher_cfg.loss_type == "mean_huber"
    assert teacher_cfg.lambda_start == 0.05
    assert teacher_cfg.lambda_end == 0.0
    assert teacher_cfg.warmup_iters == 0
    assert teacher_cfg.anneal_iters == 10000
    assert teacher_cfg.huber_delta == 0.5
    assert teacher_cfg.max_teacher_loss is None
    assert teacher_cfg.max_kl_loss is None
