"""Tests specific to velocity tasks."""

import importlib
from typing import cast

import pytest
import torch

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
  unitree_g1_blind_rough_target_navigation_geometry_probe_runner_cfg,
  unitree_g1_blind_rough_target_navigation_semantic_v2_probe_runner_cfg,
  unitree_g1_blind_rough_target_navigation_semantic_v2_shadow_runner_cfg,
  unitree_g1_blind_rough_target_navigation_slow_latent_teacherkl_runner_cfg,
)
from mjlab.tasks.velocity.mdp import (
  UniformVelocityCommandCfg,
  stair_aware_feet_gait,
  stair_sequence_event_logger,
  target_tread_midline_shaping,
)
from mjlab.tasks.velocity.mdp.teacher_target_heading_command import (
  TeacherTargetHeadingVelocityCommandCfg,
)
from mjlab.terrains.config import BLIND_HIGH_STAIRS_TREAD_DEPTHS
from mjlab.terrains.primitive_terrains import (
  BoxInvertedPyramidStairsTerrainCfg,
  BoxLongStairRunwayTerrainCfg,
  BoxPyramidStairsTerrainCfg,
)

MAIN_BRANCH_VELOCITY_TASK_IDS = (
  "Mjlab-Velocity-Blind-Rough-TargetNavigation-SemanticV2GeometryProbe-Unitree-G1",
  "Mjlab-Velocity-Blind-Rough-TargetNavigation-SemanticV2SafeStrideProbe-Unitree-G1",
  "Mjlab-Velocity-Blind-Rough-TargetNavigation-SemanticV2Shadow-TeacherKL-Unitree-G1",
  "Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1",
  "Mjlab-Velocity-Blind-Rough-TargetNavigation-StepDanger-TeacherKL-Unitree-G1",
  "Mjlab-Velocity-Blind-Rough-TargetNavigation-TeacherKL-Unitree-G1",
  "Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1",
)


def test_teacherkl_algorithm_class_path_imports_from_installed_rsl_rl() -> None:
  algorithm_cfg = RslRlPpoTeacherKLAlgorithmCfg()
  module_name, class_name = algorithm_cfg.class_name.split(":")

  algorithm_cls = getattr(importlib.import_module(module_name), class_name)

  assert algorithm_cfg.class_name == "rsl_rl.algorithms.ppo_teacher_kl:PPOTeacherKL"
  assert algorithm_cls.__name__ == "PPOTeacherKL"


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
  assert target_cfg.scene.terrain is not None
  terrain_generator = target_cfg.scene.terrain.terrain_generator
  assert terrain_generator is not None
  assert "long_stair_runway" in terrain_generator.standalone_terrains
  runway = terrain_generator.standalone_terrains["long_stair_runway"]
  assert isinstance(runway, BoxLongStairRunwayTerrainCfg)
  assert runway.size == (10.9, 2.5)
  assert runway.step_height_range == (0.04, 0.2)
  assert runway.step_width_range == (
    BLIND_HIGH_STAIRS_TREAD_DEPTHS[0],
    BLIND_HIGH_STAIRS_TREAD_DEPTHS[-1],
  )
  assert runway.start_platform_length == 2.0
  assert runway.end_platform_length == 4.0
  assert runway.end_target_fraction == 0.85
  assert runway.proportion == pytest.approx(
    sum(s.proportion for s in terrain_generator.sub_terrains.values())
  )
  assert target_cfg.scene.terrain.standalone_spawn_start_level == 3
  terrain_params = target_cfg.curriculum["terrain_levels"].params
  assert terrain_params["standalone_replay_start_level"] == 3
  assert terrain_params["standalone_replay_probability"] == pytest.approx(0.5)
  assert target_cfg.events["reset_base"].func.__name__ == (
    "reset_root_state_uniform_with_standalone_heading"
  )
  assert "runway_out_of_bounds" in target_cfg.terminations
  assert target_cfg.terminations["runway_out_of_bounds"].time_out is False
  assert "runway_target_reached" in target_cfg.terminations
  assert target_cfg.terminations["runway_target_reached"].time_out is True


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

  twist_command = cast(UniformVelocityCommandCfg, env_cfg.commands["twist"])
  assert twist_command.ranges.lin_vel_x == (0.4, 1.0)
  velocity_stages = env_cfg.curriculum["command_vel"].params["velocity_stages"]
  assert velocity_stages[0]["lin_vel_x"] == (0.4, 0.8)
  assert velocity_stages[1]["lin_vel_x"] == (0.4, 1.0)
  assert "latent" in env_cfg.observations
  assert "stair_latent" in env_cfg.observations["latent"].terms
  assert "foot_event_memory" in env_cfg.observations["latent"].terms
  assert (
    env_cfg.observations["latent"].terms["stair_latent"].params["include_gait_phase"]
    is True
  )
  assert (
    env_cfg.observations["latent"].terms["foot_event_memory"].params["memory_len"] == 6
  )
  assert (
    env_cfg.observations["latent"].terms["foot_event_memory"].params["include_summary"]
    is True
  )
  foot_event_params = env_cfg.observations["latent"].terms["foot_event_memory"].params
  assert foot_event_params["include_raw_memory"] is False
  assert foot_event_params["ratchet_height_threshold_m"] == 0.025
  assert foot_event_params["ratchet_probe_increment_m"] == 0.05
  assert foot_event_params["ratchet_first_collision_probe_push_m"] == 0.16
  assert foot_event_params["ratchet_post_first_collision_probe_increment_m"] == 0.14
  assert foot_event_params["ratchet_no_hit_lower_margin_m"] == 0.0
  assert foot_event_params["ratchet_interval_target_margin_m"] == 0.01
  assert foot_event_params["ratchet_collision_margin_m"] == 0.02
  assert foot_event_params["ratchet_toe_anchor_offset_m"] == 0.085
  assert foot_event_params["ratchet_backoff_step_m"] == 0.025
  assert foot_event_params["ratchet_first_collision_backoff_step_m"] == 0.025
  assert foot_event_params["ratchet_backoff_margin_m"] == 0.015
  assert foot_event_params["ratchet_lock_margin_m"] == 0.005
  assert foot_event_params["ratchet_lock_stable_steps"] == 1
  assert foot_event_params["ratchet_lock_target_stable_enabled"] is True
  assert foot_event_params["ratchet_soft_upper_ttl_steps"] == 5
  assert foot_event_params["ratchet_first_collision_enters_stair_mode"] is True
  assert foot_event_params["ratchet_single_collision_confirms_interval"] is False
  assert foot_event_params["ratchet_two_collision_enabled"] is True
  assert foot_event_params["ratchet_two_collision_interval_margin_m"] == 0.025
  assert foot_event_params["ratchet_two_collision_stride_layers"] == 2.0
  assert foot_event_params["ratchet_two_collision_min_layer_delta"] == 1
  assert foot_event_params["ratchet_two_collision_tread_min_m"] == 0.18
  assert foot_event_params["ratchet_two_collision_tread_max_m"] == 0.42
  assert foot_event_params["ratchet_lower_target_lag_margin_m"] == 0.0
  assert foot_event_params["ratchet_same_foot_stride_guard_layers"] == 2.0
  assert foot_event_params["ratchet_same_foot_stride_guard_margin_m"] == 0.04
  assert foot_event_params["ratchet_collision_min_confidence"] == 0.45
  assert (
    foot_event_params["ratchet_post_first_collision_collision_min_confidence"] == 0.30
  )
  assert foot_event_params["ratchet_min_interval_width_m"] == 0.04
  assert foot_event_params["ratchet_min_stride_m"] == 0.10
  assert foot_event_params["ratchet_max_stride_m"] == 0.80
  assert "latent_labels" in env_cfg.observations
  assert tuple(env_cfg.observations["latent_labels"].terms) == (
    "toe_riser_event",
    "stair_state",
    "stair_shape",
    "safe_stride",
    "future_events",
    "shape_component_valid",
    "safe_stride_interval_valid",
    "stair_depth_confirmation_event",
    "geometry_probe_validation",
    "stair_depth_confirmation_age",
    "stair_adjacent_pair_evidence",
  )
  assert "reset_stair_latent_cache" in env_cfg.events
  sequence_logger = env_cfg.metrics["stair_sequence_event_logger"]
  assert sequence_logger.func is stair_sequence_event_logger
  assert sequence_logger.params["command_name"] == "twist"
  assert sequence_logger.params["asset_cfg"].body_names == (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )
  assert rl_cfg.obs_groups["actor"] == ("actor",)
  assert rl_cfg.obs_groups["latent"] == ("latent",)
  assert "foot_step_lip_volume_penalty" not in env_cfg.rewards
  toe_reward_params = env_cfg.rewards["toe_step_riser_slab_penalty"].params
  toe_asset_cfg = toe_reward_params["asset_cfg"]
  assert toe_asset_cfg.body_names == ("left_ankle_roll_link", "right_ankle_roll_link")
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
  assert shank_reward.weight == -3.0
  assert shank_params["clearance_margin"] == 0.06
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
  midline_reward = env_cfg.rewards["target_tread_midline_shaping"]
  assert midline_reward.func is target_tread_midline_shaping
  assert midline_reward.weight == 1.2
  assert midline_reward.params["height_clearance"] == 0.02
  assert midline_reward.params["sigma_fraction"] == 0.60
  assert midline_reward.params["min_sigma"] == 0.18
  assert midline_reward.params["edge_margin"] == 0.10
  assert midline_reward.params["progress_scale"] == 0.10
  assert midline_reward.params["center_scale"] == 0.80
  assert midline_reward.params["support_scale"] == 0.85
  assert midline_reward.params["edge_scale"] == 0.35
  assert midline_reward.params["sole_margin"] == 0.020
  assert midline_reward.params["support_sigma"] == 0.04
  assert midline_reward.params["max_progress_step"] == 0.20
  assert midline_reward.params["early_stance_time"] == 0.25
  assert midline_reward.params["stance_height_tolerance"] == 0.08
  assert midline_reward.params["heading_cos"] == 0.70
  assert midline_reward.params["asset_cfg"].site_names == (
    "left_foot",
    "right_foot",
  )
  algorithm_cfg = cast(RslRlPpoTeacherKLAlgorithmCfg, rl_cfg.algorithm)
  assert algorithm_cfg.teacher_kl_cfg.log_kl_when_lambda_zero is False
  foot_gait = env_cfg.rewards["foot_gait"]
  assert foot_gait.func is stair_aware_feet_gait
  assert foot_gait.params["heading_cos"] == 0.70
  assert "stair_tread_landing_reward" not in env_cfg.rewards
  latent_labels = env_cfg.observations["latent_labels"]
  assert latent_labels.terms["geometry_probe_validation"].params == {
    "validation_fraction": 0.2
  }
  assert all(
    term.params == {}
    for name, term in latent_labels.terms.items()
    if name != "geometry_probe_validation"
  )
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
  assert actor_cfg.latent_dim == 24
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
  assert actor_cfg.stair_memory_exit_on_stair_off is False
  assert actor_cfg.stair_on_threshold == 0.35
  assert actor_cfg.stair_off_threshold == 0.20
  assert actor_cfg.aux_stair_coef == 0.05
  assert actor_cfg.aux_stair_pos_weight == 3.0
  assert actor_cfg.aux_future_collision_risk_coef == 0.0
  assert actor_cfg.aux_future_safe_landing_quality_coef == 0.0
  assert actor_cfg.future_horizon == 20
  assert actor_cfg.aux_stair_shape_coef == 0.05
  assert actor_cfg.aux_safe_stride_coef == 0.04
  assert actor_cfg.structured_safe_stride_enabled is True
  assert actor_cfg.dynamic_stair_shape_enabled is True
  assert actor_cfg.dynamic_safe_stride_enabled is True
  assert actor_cfg.safe_stride_phase_dim == 2
  assert actor_cfg.safe_stride_phase_start == 91
  assert actor_cfg.stair_shape_huber_delta == 0.05
  assert actor_cfg.stair_shape_same_foot_loss_coef == 0.0
  assert actor_cfg.stair_shape_riser_loss_coef == 1.0
  assert actor_cfg.safe_stride_huber_delta == 0.05
  assert actor_cfg.safe_stride_width_loss_coef == 1.5
  assert actor_cfg.safe_stride_lower_shortfall_coef == 1.2
  assert actor_cfg.safe_stride_interval_coverage_loss_coef == 0.0
  assert actor_cfg.safe_stride_interval_coverage_margin == 0.01
  assert actor_cfg.safe_stride_confidence_loss_coef == 0.10
  assert actor_cfg.safe_stride_phase_center_loss_coef == 1.0
  assert actor_cfg.safe_stride_trend_loss_coef == 0.50
  assert actor_cfg.safe_stride_dense_trend_loss_coef == 1.0
  assert actor_cfg.safe_stride_std_floor_loss_coef == 0.0
  assert actor_cfg.safe_stride_centered_loss_coef == 0.0
  assert actor_cfg.safe_stride_std_floor_ratio == 0.55
  assert actor_cfg.safe_stride_deployable_hint_loss_coef == 0.0
  assert actor_cfg.safe_stride_deployable_hint_margin == 0.01
  assert actor_cfg.same_foot_stride_deployable_hint_loss_coef == 0.0
  assert actor_cfg.same_foot_stride_deployable_hint_margin == 0.02
  assert actor_cfg.safe_stride_min == 0.10
  assert actor_cfg.safe_stride_max == 0.55
  assert actor_cfg.actor_semantic_enabled is True


def test_target_tread_midline_center_score_is_dense_and_centered() -> None:
  errors = torch.tensor([0.0, 0.20, 0.40])
  widths = torch.full_like(errors, 0.20)

  scores = target_tread_midline_shaping._center_score(errors, widths)

  assert scores[0].item() == pytest.approx(1.0)
  assert scores[0] > scores[1] > scores[2] > 0.0


def test_target_tread_midline_edge_penalty_is_bounded() -> None:
  center_error = torch.tensor([0.0, 0.5, 1.0, 2.0])
  support_error = torch.tensor([0.0, 0.5, 1.0, 2.0])

  penalty = target_tread_midline_shaping._edge_penalty(
    center_error,
    support_error,
  )

  torch.testing.assert_close(
    penalty,
    torch.tensor([0.0, 0.5, 1.0, 1.0]),
  )


def test_semantic_v2_shadow_runner_trains_heads_without_resuming_optimizer() -> None:
  cfg = unitree_g1_blind_rough_target_navigation_semantic_v2_shadow_runner_cfg()
  actor_cfg = cast(RslRlGatedStairLatentModelCfg, cfg.actor)

  assert cfg.experiment_name == (
    "g1_blind_rough_target_navigation_semantic_v2_shadow_teacherkl"
  )
  assert cfg.load_optimizer_on_resume is False
  assert cfg.load_iteration_on_resume is False
  assert cfg.bootstrap_checkpoint_path is None
  assert actor_cfg.shadow_semantic_enabled is True
  assert actor_cfg.actor_semantic_enabled is False
  assert actor_cfg.structured_safe_stride_enabled is True
  assert actor_cfg.aux_stair_shape_coef == 0.05
  assert actor_cfg.aux_safe_stride_coef == 0.04
  assert actor_cfg.latent_dim == 24
  assert actor_cfg.state_latent_dim == 8


def test_semantic_v2_probe_freezes_policy_and_uses_dynamic_stride_input() -> None:
  task_id = (
    "Mjlab-Velocity-Blind-Rough-TargetNavigation-SemanticV2SafeStrideProbe-Unitree-G1"
  )
  env_cfg = load_env_cfg(task_id)
  cfg = unitree_g1_blind_rough_target_navigation_semantic_v2_probe_runner_cfg()
  actor_cfg = cast(RslRlGatedStairLatentModelCfg, cfg.actor)
  algorithm_cfg = cast(RslRlPpoTeacherKLAlgorithmCfg, cfg.algorithm)

  assert (
    env_cfg.observations["latent"].terms["stair_latent"].params["include_gait_phase"]
    is True
  )
  assert actor_cfg.structured_safe_stride_enabled is True
  assert actor_cfg.dynamic_safe_stride_enabled is True
  assert actor_cfg.safe_stride_phase_dim == 2
  assert actor_cfg.safe_stride_phase_start == 91
  assert actor_cfg.aux_safe_stride_coef == 1.0
  assert actor_cfg.safe_stride_width_loss_coef == 1.5
  assert actor_cfg.safe_stride_interval_coverage_loss_coef == 0.0
  assert actor_cfg.actor_semantic_enabled is False
  assert actor_cfg.aux_event_coef == 0.0
  assert actor_cfg.aux_stair_coef == 0.0
  assert actor_cfg.aux_future_collision_risk_coef == 0.0
  assert actor_cfg.aux_future_safe_landing_quality_coef == 0.0
  assert actor_cfg.aux_stair_shape_coef == 0.0
  assert algorithm_cfg.safe_stride_probe_only is True
  assert algorithm_cfg.teacher_kl_cfg.enabled is False
  assert cfg.load_optimizer_on_resume is False
  assert cfg.load_iteration_on_resume is False


def test_geometry_probe_freezes_policy_and_trains_independent_head() -> None:
  cfg = unitree_g1_blind_rough_target_navigation_geometry_probe_runner_cfg()
  actor_cfg = cast(RslRlGatedStairLatentModelCfg, cfg.actor)
  algorithm_cfg = cast(RslRlPpoTeacherKLAlgorithmCfg, cfg.algorithm)

  assert actor_cfg.geometry_probe_input == "shape"
  assert actor_cfg.aux_stair_shape_coef == 1.0
  assert actor_cfg.aux_safe_stride_coef == 0.0
  assert algorithm_cfg.geometry_probe_only is True
  assert algorithm_cfg.geometry_probe_permute_depth_labels is False
  assert algorithm_cfg.safe_stride_probe_only is False
  assert algorithm_cfg.teacher_kl_cfg.enabled is False
  assert cfg.load_optimizer_on_resume is False
  assert cfg.load_iteration_on_resume is False


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
  twist_command = cast(UniformVelocityCommandCfg, cfg.commands["twist"])
  assert twist_command.ranges.lin_vel_x == (0.6, 0.9)


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
  assert "foot_step_lip_volume_penalty" not in env_cfg.rewards
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
      stair_memory_exit_on_stair_off=True,
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
      safe_stride_interval_coverage_loss_coef=0.25,
      safe_stride_interval_coverage_margin=0.03,
      shadow_semantic_enabled=True,
      actor_semantic_enabled=False,
      structured_safe_stride_enabled=True,
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
  assert actor_cfg.stair_memory_exit_on_stair_off is True
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
  assert actor_cfg.safe_stride_interval_coverage_loss_coef == 0.25
  assert actor_cfg.safe_stride_interval_coverage_margin == 0.03
  assert actor_cfg.shadow_semantic_enabled is True
  assert actor_cfg.actor_semantic_enabled is False
  assert actor_cfg.structured_safe_stride_enabled is True
  assert actor_cfg.dynamic_stair_shape_enabled is True
  assert actor_cfg.dynamic_safe_stride_enabled is True
  assert actor_cfg.safe_stride_phase_start == 91


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
