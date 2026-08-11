"""Tests specific to velocity tasks."""

import importlib
from types import SimpleNamespace
from typing import Any, cast

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
from mjlab.tasks.velocity.config.g1.footprint_detector_env_cfg import (
  FOOTPRINT_DETECTOR_LEVEL_WEIGHTS,
  FOOTPRINT_DETECTOR_STANDALONE_FRACTION,
  unitree_g1_footprint_detector_env_cfg,
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
  joint_vel_l2,
  posture,
  stair_aware_feet_gait,
  stair_sequence_event_logger,
  stair_stride_phase_reward,
  swing_toe_support_edge_cylinder_penalty,
  target_tread_midline_shaping,
)
from mjlab.tasks.velocity.mdp.stair_geometry import (
  STAIR_ASCENT_DIR_KEY,
  STAIR_CLEARANCE_FOOT_LAYERS_KEY,
  STAIR_CLEARANCE_FOOT_LAYERS_VALID_KEY,
  STAIR_CURRENT_GROUND_CONTACT_KEY,
  STAIR_CURRENT_STAIR_SUPPORT_KEY,
  STAIR_CURRENT_SUPPORT_LAYER_KEY,
  STAIR_PHASE_KEY,
  STAIR_SEQUENCE_ID_KEY,
  STAIR_STRIDE_CONTROL_FOOT_MASK_KEY,
  STAIR_STRIDE_CONTROL_PHASE_KEY,
  STAIR_STRIDE_CONTROL_REFERENCE_KEY,
  STAIR_STRIDE_CONTROL_TARGET_KEY,
  STAIR_STRIDE_CONTROL_VALID_KEY,
  STAIR_STRIDE_EVENT_BACKOFF_TOUCHDOWN,
  STAIR_STRIDE_EVENT_ENTRY_TOUCHDOWN,
  STAIR_STRIDE_EVENT_LOCK_TOUCHDOWN,
  STAIR_STRIDE_EVENT_NONE,
  STAIR_STRIDE_EVENT_PROBE_TOUCHDOWN,
  STAIR_STRIDE_EVENT_VALID_SECOND_HIT,
  STAIR_STRIDE_INTENT_FARTHER,
  STAIR_STRIDE_PHASE_EVENT_ACTUAL_KEY,
  STAIR_STRIDE_PHASE_EVENT_COLLISION_PENALTY_KEY,
  STAIR_STRIDE_PHASE_EVENT_COMPLETED_KEY,
  STAIR_STRIDE_PHASE_EVENT_FOOT_ID_KEY,
  STAIR_STRIDE_PHASE_EVENT_ID_KEY,
  STAIR_STRIDE_PHASE_EVENT_INTENT_KEY,
  STAIR_STRIDE_PHASE_EVENT_PREVIOUS_KEY,
  STAIR_STRIDE_PHASE_EVENT_TARGET_KEY,
  STAIR_STRIDE_PHASE_EVENT_TYPE_KEY,
  STAIR_STRIDE_SWING_STRIDE_KEY,
  STAIR_STRIDE_SWING_VALID_KEY,
  STAIR_TARGET_FOOT_KEY,
  support_edge_toe_brush_mask,
)
from mjlab.tasks.velocity.mdp.teacher_target_heading_command import (
  TeacherTargetHeadingVelocityCommandCfg,
)
from mjlab.tasks.velocity.mdp.temporal_stair_rewards import (
  _filter_passed_riser_ground_contacts,
  _stair_swing_activity,
  stair_tread_landing_reward,
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


def test_g1_footprint_detector_task_uses_fixed_terrain_pools() -> None:
  cfg = unitree_g1_footprint_detector_env_cfg()
  main_cfg = load_env_cfg(
    "Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1"
  )

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.max_init_terrain_level is None
  assert cfg.scene.terrain.standalone_spawn_start_level == 0
  assert main_cfg.scene.terrain is not None
  assert main_cfg.scene.terrain.standalone_spawn_start_level == 3

  terrain_generator = cfg.scene.terrain.terrain_generator
  assert terrain_generator is not None
  runway_names = sorted(
    name
    for name in terrain_generator.standalone_terrains
    if name.startswith("long_stair_runway_w")
  )
  assert len(runway_names) == len(BLIND_HIGH_STAIRS_TREAD_DEPTHS)
  assert sum(
    terrain_generator.standalone_terrains[name].proportion for name in runway_names
  ) == pytest.approx(FOOTPRINT_DETECTOR_STANDALONE_FRACTION)
  assert sum(
    sub_cfg.proportion for sub_cfg in terrain_generator.sub_terrains.values()
  ) == pytest.approx(1.0 - FOOTPRINT_DETECTOR_STANDALONE_FRACTION)
  assert terrain_generator.sub_terrains["flat"].proportion > 0.0

  for index, name in enumerate(runway_names):
    runway = terrain_generator.standalone_terrains[name]
    assert isinstance(runway, BoxLongStairRunwayTerrainCfg)
    width = BLIND_HIGH_STAIRS_TREAD_DEPTHS[index]
    assert runway.step_width_range == (width, width)
    assert runway.size == (10.9, 2.5)

  terrain_params = cfg.curriculum["terrain_levels"].params
  assert cfg.curriculum["terrain_levels"].func.__name__ == (
    "fixed_pool_terrain_levels_vel"
  )
  assert terrain_params["standalone_fraction"] == pytest.approx(
    FOOTPRINT_DETECTOR_STANDALONE_FRACTION
  )
  assert terrain_params["level_ranges"] == ((0, 2), (3, 5), (6, 9))
  assert terrain_params["level_weights"] == FOOTPRINT_DETECTOR_LEVEL_WEIGHTS


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
  assert foot_event_params["post_first_toe_miss_prob"] == 0.05
  assert foot_event_params["ratchet_height_threshold_m"] == 0.025
  assert foot_event_params["ratchet_probe_increment_m"] == 0.05
  assert foot_event_params["ratchet_probe_bootstrap_increment_m"] == 0.10
  assert foot_event_params["ratchet_probe_provisional_min_target_m"] == 0.50
  assert foot_event_params["ratchet_probe_cap_m"] == 0.76
  assert foot_event_params["ratchet_probe_target_tolerance_m"] == 0.025
  assert foot_event_params["ratchet_post_first_collision_probe_increment_m"] == 0.05
  assert foot_event_params["ratchet_probe_min_progress_m"] == 0.01
  assert foot_event_params["ratchet_entry_target_m"] == 0.05
  assert foot_event_params["ratchet_entry_target_tolerance_m"] == 0.01
  assert foot_event_params["ratchet_sequence_min_confidence"] == 0.30
  assert foot_event_params["ratchet_sequence_height_tolerance_m"] == 0.06
  assert foot_event_params["ratchet_sequence_max_adjacent_height_m"] == 0.24
  assert foot_event_params["ratchet_no_hit_lower_margin_m"] == 0.0
  assert foot_event_params["ratchet_interval_target_margin_m"] == 0.01
  assert foot_event_params["ratchet_collision_margin_m"] == 0.02
  assert foot_event_params["ratchet_toe_anchor_offset_m"] == 0.085
  assert foot_event_params["ratchet_recovery_completion_tolerance_m"] == 0.015
  assert foot_event_params["ratchet_recovery_completion_lower_tolerance_m"] == 0.015
  assert foot_event_params["ratchet_recovery_min_reduction_m"] == 0.020
  assert foot_event_params["ratchet_recovery_min_backoff_m"] == 0.025
  assert foot_event_params["ratchet_recovery_max_backoff_m"] == 0.050
  assert foot_event_params["ratchet_recovery_depth_min_m"] == 0.25
  assert foot_event_params["ratchet_recovery_depth_max_m"] == 0.35
  assert foot_event_params["ratchet_recovery_effective_sole_length_m"] == 0.185
  assert foot_event_params["ratchet_recovery_rear_support_margin_m"] == 0.015
  assert foot_event_params["ratchet_lock_probe_lower_margin_m"] == 0.01
  assert foot_event_params["ratchet_lock_margin_m"] == 0.005
  assert foot_event_params["ratchet_lock_intent_tolerance_m"] == 0.015
  assert foot_event_params["ratchet_lock_stable_steps"] == 2
  assert foot_event_params["ratchet_lock_target_stable_enabled"] is False
  assert foot_event_params["ratchet_lock_phase_correction_enabled"] is False
  assert foot_event_params["ratchet_lock_phase_initial_front_error_m"] == 0.0
  assert foot_event_params["ratchet_soft_upper_ttl_steps"] == 5
  assert foot_event_params["ratchet_first_collision_enters_stair_mode"] is True
  assert foot_event_params["ratchet_single_collision_confirms_interval"] is False
  assert foot_event_params["ratchet_two_collision_enabled"] is True
  assert foot_event_params["ratchet_two_collision_interval_margin_m"] == 0.025
  assert foot_event_params["ratchet_two_collision_stride_layers"] == 2.0
  assert foot_event_params["ratchet_two_collision_min_layer_delta"] == 1
  assert foot_event_params["ratchet_two_collision_min_height_delta_m"] == 0.055
  assert foot_event_params["ratchet_two_collision_nominal_riser_height_m"] == 0.15
  assert foot_event_params["ratchet_two_collision_height_layer_tolerance_m"] == 0.06
  assert foot_event_params["ratchet_two_collision_use_height_layers"] is True
  assert foot_event_params["ratchet_two_collision_lower_cross_margin_m"] == 0.10
  assert foot_event_params["ratchet_two_collision_upper_cross_margin_m"] == 0.08
  assert foot_event_params["ratchet_rejected_second_hit_confirm_count"] == 2
  assert foot_event_params["ratchet_rejected_second_hit_target_tolerance_m"] == 0.05
  assert foot_event_params["ratchet_second_collision_requires_up_step"] is True
  assert foot_event_params["ratchet_two_collision_tread_min_m"] == 0.23
  assert foot_event_params["ratchet_two_collision_tread_max_m"] == 0.37
  assert foot_event_params["ratchet_lower_target_lag_margin_m"] == 0.0
  assert foot_event_params["ratchet_same_foot_stride_guard_layers"] == 2.0
  assert foot_event_params["ratchet_same_foot_stride_guard_margin_m"] == 0.04
  assert foot_event_params["ratchet_collision_min_confidence"] == 0.45
  assert (
    foot_event_params["ratchet_post_first_collision_collision_min_confidence"] == 0.30
  )
  assert not any(name.endswith("_reward_tolerance_m") for name in foot_event_params)
  assert foot_event_params["ratchet_min_interval_width_m"] == 0.04
  assert foot_event_params["ratchet_min_stride_m"] == 0.10
  assert foot_event_params["ratchet_max_stride_m"] == 0.85
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
  assert toe_reward_params["slab_depth"] == 0.05
  assert toe_reward_params["contact_penalty_scale"] == 0.65
  assert toe_reward_params["contact_time_scale"] == 0.10
  assert toe_reward_params["event_min_forward_intent_speed"] == 0.04
  assert toe_reward_params["event_min_forward_speed_drop"] == 0.04
  assert toe_reward_params["event_blocked_forward_speed"] == 0.03
  assert toe_reward_params["event_persistent_steps"] == 3
  assert toe_reward_params["stair_heading_cos"] == 0.70
  assert toe_reward_params["stair_touchdown_height_tolerance"] == 0.08
  assert toe_reward_params["stair_touchdown_lateral_margin"] == 0.03
  assert toe_reward_params["stair_min_safe_stride"] == 0.10
  assert toe_reward_params["stair_max_safe_stride"] == 0.85
  assert toe_reward_params["stair_max_tracking_stride"] == 0.85
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
  support_edge_reward = env_cfg.rewards["swing_toe_support_edge_cylinder_penalty"]
  support_edge_params = support_edge_reward.params
  assert support_edge_reward.func is swing_toe_support_edge_cylinder_penalty
  assert support_edge_reward.weight == -2.0
  assert support_edge_params["radius"] == 0.05
  assert support_edge_params["toe_x_min"] == 0.08
  assert support_edge_params["direction_cos_threshold"] == 0.90
  assert support_edge_params["asset_cfg"].body_names == (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )
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
  stride_phase_reward = env_cfg.rewards["stair_stride_phase_reward"]
  assert stride_phase_reward.func is stair_stride_phase_reward
  assert stride_phase_reward.weight == 1.0
  assert stride_phase_reward.params == {
    "probe_scale": 3.0,
    "confirmation_scale": 1.0,
    "backoff_scale": 4.0,
    "lock_scale": 2.0,
    "probe_growth_scale": 0.05,
    "probe_reference_tolerance": 0.025,
    "probe_growth_weight": 0.50,
    "probe_target_progress_weight": 0.50,
    "probe_completion_start": 0.03,
    "probe_completion_span": 0.02,
    "probe_completion_bonus": 1.00,
    "probe_convex_bonus": 2.50,
    "probe_post_target_decay_scale": 0.012,
    "probe_overshoot_penalty": 1.0,
    "probe_stall_penalty": 0.50,
    "probe_swing_fraction": 0.70,
    "swing_invalid_grace_steps": 2,
    "entry_reward_scale": 0.75,
    "entry_swing_fraction": 0.20,
    "entry_target": 0.05,
    "entry_target_tolerance": 0.01,
    "entry_completion_bonus": 1.00,
    "entry_post_target_decay_scale": 0.010,
    "entry_overshoot_penalty": 1.0,
    "backoff_progress_scale": 0.03,
    "backoff_tolerance": 0.06,
    "backoff_progress_weight": 0.65,
    "backoff_proximity_weight": 0.35,
    "backoff_overretreat_penalty": 1.0,
    "backoff_overretreat_tolerance": 0.020,
    "backoff_completion_bonus": 1.00,
    "backoff_stall_penalty": 0.75,
    "backoff_swing_fraction": 0.80,
    "backoff_swing_window_min": 0.025,
    "backoff_swing_window_max": 0.050,
    "backoff_forward_decay_scale": 0.010,
    "backoff_forward_penalty": 0.50,
    "lock_progress_scale": 0.03,
    "lock_tolerance": 0.05,
    "lock_swing_fraction": 0.70,
    "lock_progress_weight": 0.40,
    "lock_tracking_weight": 0.60,
    "intent_tolerance": 0.015,
    "second_hit_penalty_refund_weight": 4.2,
    "event_observation_group_name": "latent",
    "event_observation_term_name": "foot_event_memory",
  }
  assert "stair_probe_stride_growth_reward" not in env_cfg.rewards
  assert "stair_confirmed_backoff_stride_reward" not in env_cfg.rewards
  assert "stair_lock_stride_hold_reward" not in env_cfg.rewards
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
  assert not any(name.startswith("adaptive_") for name in midline_reward.params)
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
  assert actor_cfg.safe_stride_max == 0.85
  assert actor_cfg.actor_semantic_enabled is True


def test_swing_toe_support_edge_cylinder_penalty_is_linear_and_finite() -> None:
  points = torch.tensor(
    [
      [[0.05, 0.0, 0.0]],
      [[0.025, 0.0, 0.0]],
      [[0.0, 0.0, 0.0]],
      [[0.0, 2.0, 0.0]],
    ]
  )
  edge_start = torch.tensor([[0.0, -1.0, 0.0]]).expand(4, -1)
  edge_end = torch.tensor([[0.0, 1.0, 0.0]]).expand(4, -1)

  distance, intrusion = swing_toe_support_edge_cylinder_penalty._cylinder_intrusion(
    points,
    edge_start,
    edge_end,
    radius=0.05,
  )

  torch.testing.assert_close(distance[:3], torch.tensor([0.05, 0.025, 0.0]))
  torch.testing.assert_close(intrusion[:3], torch.tensor([0.0, 0.5, 1.0]))
  assert torch.isinf(distance[3])
  assert intrusion[3].item() == 0.0


def test_support_edge_brush_filter_ignores_passed_launch_and_support_layers() -> None:
  toe_contact = torch.tensor([[[True, True, True, True], [True, True, True, True]]])
  contact_layers = torch.tensor([[[1, 2, 3, 4], [1, 2, 3, 4]]])
  contact_sequences = torch.full_like(contact_layers, 7)

  ignored = support_edge_toe_brush_mask(
    toe_contact,
    contact_layers,
    contact_sequences,
    torch.tensor([[True, False]]),
    torch.tensor([[3, 2]]),
    torch.tensor([[True, True]]),
    torch.tensor([7]),
    torch.tensor([1]),
    torch.tensor([True]),
  )

  assert ignored.tolist() == [[[False, False, False, False], [True, True, True, False]]]
  inactive = support_edge_toe_brush_mask(
    toe_contact,
    contact_layers,
    contact_sequences,
    torch.tensor([[True, False]]),
    torch.tensor([[3, 2]]),
    torch.tensor([[True, True]]),
    torch.tensor([7]),
    torch.tensor([1]),
    torch.tensor([False]),
  )
  assert not inactive.any()


def test_stair_swing_activity_includes_first_airborne_frame() -> None:
  active, started = _stair_swing_activity(
    torch.tensor([2, 2, 0]),
    torch.tensor([1, 1, -1]),
    torch.tensor([[True, False], [True, True], [True, False]]),
    torch.tensor([False, True, False]),
  )

  assert active.tolist() == [True, True, False]
  assert started.tolist() == [True, False, False]


def test_passed_riser_brush_does_not_create_ground_or_touchdown_state() -> None:
  ground, first, suppressed = _filter_passed_riser_ground_contacts(
    torch.tensor([[True, True], [True, True]]),
    torch.tensor([[False, True], [False, False]]),
    torch.tensor([[False, True], [False, True]]),
    torch.tensor([[True, False], [True, True]]),
    torch.tensor([[True, False], [True, False]]),
  )

  assert ground.tolist() == [[True, False], [True, True]]
  assert first.tolist() == [[False, False], [False, True]]
  assert suppressed.tolist() == [[False, True], [False, False]]


def test_toe_slab_mask_accepts_per_foot_filtered_boundaries() -> None:
  toe_points = torch.tensor([[[[0.01, 0.0, 0.5]], [[0.01, 0.0, 0.5]]]])
  boundaries = torch.tensor(
    [[[0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]]
  )
  valid_by_foot = torch.tensor([[[False], [True]]])

  unsafe = stair_tread_landing_reward._toe_slab_mask(
    toe_points,
    boundaries,
    valid_by_foot,
    slab_depth=0.05,
    u_margin=0.0,
    v_margin=0.0,
    surface_tol=0.0,
  )

  assert unsafe.tolist() == [[False, True]]


def test_swing_toe_support_edge_cylinder_penalty_has_strict_role_gates(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  boundary = torch.tensor([0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, -0.15, 0.0])
  terrain = SimpleNamespace(
    terrain_levels=torch.tensor([0]),
    terrain_types=torch.tensor([0]),
    step_boundaries_by_tile=boundary.view(1, 1, 1, 11),
    step_boundary_counts=torch.tensor([[1]]),
    step_boundary_sequence_ids_by_tile=torch.tensor([[[7]]]),
    step_boundary_layers_by_tile=torch.tensor([[[3]]]),
  )
  env = SimpleNamespace(
    num_envs=1,
    device=torch.device("cpu"),
    scene=SimpleNamespace(terrain=terrain),
    extras={
      "log": {},
      STAIR_PHASE_KEY: torch.tensor([2]),
      STAIR_SEQUENCE_ID_KEY: torch.tensor([7]),
      STAIR_ASCENT_DIR_KEY: torch.tensor([[1.0, 0.0]]),
      STAIR_TARGET_FOOT_KEY: torch.tensor([1]),
      STAIR_CURRENT_GROUND_CONTACT_KEY: torch.tensor([[True, False]]),
      STAIR_CURRENT_STAIR_SUPPORT_KEY: torch.tensor([[True, False]]),
      STAIR_CURRENT_SUPPORT_LAYER_KEY: torch.tensor([[3, 0]]),
      STAIR_CLEARANCE_FOOT_LAYERS_KEY: torch.tensor([[3, 2]]),
      STAIR_CLEARANCE_FOOT_LAYERS_VALID_KEY: torch.tensor([[True, True]]),
    },
  )
  points_w = torch.tensor([[[[1.0, 0.0, 0.0]], [[0.025, 0.0, 0.0]]]])
  term = object.__new__(swing_toe_support_edge_cylinder_penalty)
  term._local_x = torch.tensor([0.10])
  monkeypatch.setattr(
    term,
    "_foot_points_w",
    lambda _env, _asset_cfg: (points_w, torch.zeros_like(points_w)),
  )

  penalty = term(cast(Any, env))
  torch.testing.assert_close(penalty, torch.tensor([0.5]))

  env.extras[STAIR_CURRENT_GROUND_CONTACT_KEY] = torch.tensor([[True, True]])
  torch.testing.assert_close(term(cast(Any, env)), torch.zeros(1))

  env.extras[STAIR_CURRENT_GROUND_CONTACT_KEY] = torch.tensor([[True, False]])
  terrain.step_boundary_layers_by_tile = torch.tensor([[[4]]])
  torch.testing.assert_close(term(cast(Any, env)), torch.zeros(1))


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


def test_stride_phase_probe_score_rewards_incremental_growth() -> None:
  stride = torch.tensor([0.49, 0.50, 0.51, 0.53, 0.60])
  previous = torch.full_like(stride, 0.50)

  scores = stair_stride_phase_reward._probe_score(
    stride,
    previous,
    0.05,
  )

  torch.testing.assert_close(
    scores,
    torch.tensor([-0.2, 0.0, 0.2, 0.6, 1.0]),
  )


def test_stride_phase_probe_components_prefer_bounded_target_progress() -> None:
  previous = torch.full((8,), 0.50)
  target = torch.full((8,), 0.55)
  actual = torch.tensor([0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58])

  growth, progress, completion, overshoot = stair_stride_phase_reward._probe_components(
    actual,
    previous,
    target,
    0.05,
    0.025,
    0.03,
    0.02,
  )
  linear_score = 3.0 * 0.5 * (growth + progress) + 0.75 * completion - overshoot
  score = stair_stride_phase_reward._probe_nonlinear_score(
    actual,
    previous,
    target,
    growth,
    progress,
    completion,
    overshoot,
    3.0,
    0.5,
    0.5,
    0.03,
    0.02,
    1.00,
    2.50,
    0.012,
    1.0,
  )

  assert score[0] > 0.0
  assert torch.all(score[1:5] > score[:4])
  assert torch.all(score[:5] > linear_score[:5])
  assert score[5] < score[4]
  assert score[6] < score[5]
  assert score[7] < 0.0
  assert score[4] >= 6.0
  assert completion[4] == 1.0
  assert overshoot[7] > 0.0


def test_stride_phase_probe_components_reward_overshoot_correction() -> None:
  previous = torch.tensor([0.70, 0.70])
  target = torch.tensor([0.65, 0.65])
  actual = torch.tensor([0.68, 0.72])

  direction, progress, _, _ = stair_stride_phase_reward._probe_components(
    actual,
    previous,
    target,
    0.05,
    0.025,
    0.03,
    0.02,
  )

  assert direction[0] > 0.0
  assert progress[0] > 0.0
  assert direction[1] < 0.0
  assert progress[1] < 0.0


def test_stride_phase_backoff_score_rewards_error_improvement() -> None:
  target = torch.full((4,), 0.60)
  previous = torch.full((4,), 0.80)
  actual = torch.tensor([0.78, 0.70, 0.62, 0.50])

  progress, tracking, error = stair_stride_phase_reward._backoff_components(
    actual,
    previous,
    target,
    0.05,
    0.05,
  )

  assert progress[0] > 0.0
  assert progress[1] > progress[0]
  assert progress[2] > progress[1]
  assert progress[3] > 0.0
  assert tracking[2] > tracking[1]
  torch.testing.assert_close(error, torch.tensor([0.18, 0.10, 0.02, 0.10]))


def test_stride_phase_lock_score_rewards_error_improvement_and_hold() -> None:
  target = torch.full((3,), 0.60)
  previous = torch.tensor([0.68, 0.60, 0.60])
  actual = torch.tensor([0.64, 0.60, 0.66])

  progress, tracking = stair_stride_phase_reward._lock_components(
    actual,
    previous,
    target,
    0.03,
    0.05,
  )

  assert progress[0] > 0.0
  assert tracking[1] == 1.0
  assert progress[2] < 0.0
  assert tracking[2] < 0.0


def test_stride_phase_centered_score_rewards_backoff_and_lock_target() -> None:
  stride = torch.tensor([0.50, 0.55, 0.60, 0.65, 0.70])
  target = torch.full_like(stride, 0.60)

  scores = stair_stride_phase_reward._centered_score(
    stride,
    target,
    0.05,
  )

  torch.testing.assert_close(
    scores,
    torch.tensor([-1.0, 0.0, 1.0, 0.0, -1.0]),
  )


def _make_stride_phase_reward_for_test() -> stair_stride_phase_reward:
  reward = object.__new__(stair_stride_phase_reward)
  reward._last_event_id = torch.zeros(1, dtype=torch.long)
  reward._event_producer = None
  return reward


def _make_stride_phase_env(
  *,
  swing_stride: float = 0.0,
  swing_valid: bool = False,
  control_phase: int = STAIR_STRIDE_EVENT_NONE,
  control_target: float = 0.0,
  control_reference: float = 0.0,
  control_valid: bool = False,
  event_id: int = 0,
  event_type: int = STAIR_STRIDE_EVENT_NONE,
  event_actual: float = 0.0,
  event_previous: float = 0.0,
  event_target: float = 0.0,
  event_foot_id: int = -1,
  event_completed: bool = False,
) -> SimpleNamespace:
  return SimpleNamespace(
    num_envs=1,
    device=torch.device("cpu"),
    step_dt=0.02,
    extras={
      STAIR_STRIDE_PHASE_EVENT_ID_KEY: torch.tensor([event_id]),
      STAIR_STRIDE_PHASE_EVENT_TYPE_KEY: torch.tensor([event_type]),
      STAIR_STRIDE_PHASE_EVENT_ACTUAL_KEY: torch.tensor([event_actual]),
      STAIR_STRIDE_PHASE_EVENT_PREVIOUS_KEY: torch.tensor([event_previous]),
      STAIR_STRIDE_PHASE_EVENT_TARGET_KEY: torch.tensor([event_target]),
      STAIR_STRIDE_PHASE_EVENT_INTENT_KEY: torch.tensor([STAIR_STRIDE_INTENT_FARTHER]),
      STAIR_STRIDE_PHASE_EVENT_FOOT_ID_KEY: torch.tensor([event_foot_id]),
      STAIR_STRIDE_PHASE_EVENT_COMPLETED_KEY: torch.tensor([event_completed]),
      STAIR_STRIDE_PHASE_EVENT_COLLISION_PENALTY_KEY: torch.tensor([0.0]),
      STAIR_STRIDE_SWING_STRIDE_KEY: torch.tensor([[swing_stride, 0.0]]),
      STAIR_STRIDE_SWING_VALID_KEY: torch.tensor([[swing_valid, False]]),
      STAIR_STRIDE_CONTROL_PHASE_KEY: torch.tensor([control_phase]),
      STAIR_STRIDE_CONTROL_TARGET_KEY: torch.tensor([control_target]),
      STAIR_STRIDE_CONTROL_REFERENCE_KEY: torch.tensor([control_reference]),
      STAIR_STRIDE_CONTROL_VALID_KEY: torch.tensor([control_valid]),
      STAIR_STRIDE_CONTROL_FOOT_MASK_KEY: torch.tensor([[True, False]]),
    },
  )


def test_stride_phase_swing_same_stride_does_not_repeat_reward() -> None:
  reward = _make_stride_phase_reward_for_test()
  env = _make_stride_phase_env(
    swing_stride=0.50,
    swing_valid=True,
    control_phase=STAIR_STRIDE_EVENT_PROBE_TOUCHDOWN,
    control_target=0.55,
    control_reference=0.50,
    control_valid=True,
  )

  activation = reward(env)
  env.extras[STAIR_STRIDE_SWING_STRIDE_KEY] = torch.tensor([[0.53, 0.0]])
  first = reward(env)
  second = reward(env)

  torch.testing.assert_close(activation, torch.tensor([0.0]))
  assert first.item() > 0.0
  torch.testing.assert_close(second, torch.tensor([0.0]))


def test_stride_phase_swing_state_persists_with_string_env_device() -> None:
  reward = _make_stride_phase_reward_for_test()
  env = _make_stride_phase_env(
    swing_stride=0.50,
    swing_valid=True,
    control_phase=STAIR_STRIDE_EVENT_PROBE_TOUCHDOWN,
    control_target=0.55,
    control_reference=0.50,
    control_valid=True,
  )
  env.device = "cpu"

  torch.testing.assert_close(reward(env), torch.tensor([0.0]))
  env.extras[STAIR_STRIDE_SWING_STRIDE_KEY] = torch.tensor([[0.53, 0.0]])

  assert reward(env).item() > 0.0
  assert reward._swing_continuation_steps[0, 0].item() == 1


def test_stride_phase_probe_swing_delta_accelerates_toward_target() -> None:
  stride = torch.tensor([0.51, 0.53, 0.55])
  quality = stair_stride_phase_reward._probe_swing_quality(
    stride,
    torch.full_like(stride, 0.50),
    torch.full_like(stride, 0.55),
    probe_scale=3.0,
    growth_scale=0.05,
    convex_bonus=2.5,
    post_target_decay_scale=0.012,
    reference_tolerance=0.025,
    overshoot_penalty=1.0,
  )

  assert quality[1] - quality[0] > quality[0]
  assert quality[2] - quality[1] > quality[1] - quality[0]


def test_stride_phase_probe_swing_quality_changes_before_previous_stride() -> None:
  stride = torch.tensor([0.10, 0.30, 0.50])
  quality = stair_stride_phase_reward._probe_swing_quality(
    stride,
    torch.zeros_like(stride),
    torch.full_like(stride, 0.55),
    probe_scale=3.0,
    growth_scale=0.05,
    convex_bonus=2.5,
    post_target_decay_scale=0.012,
    reference_tolerance=0.025,
    overshoot_penalty=1.0,
  )

  assert quality[0].item() > 0.0
  assert quality[2] - quality[1] > quality[1] - quality[0]


def test_stride_phase_probe_swing_over_target_emits_negative_delta() -> None:
  reward = _make_stride_phase_reward_for_test()
  env = _make_stride_phase_env(
    swing_stride=0.50,
    swing_valid=True,
    control_phase=STAIR_STRIDE_EVENT_PROBE_TOUCHDOWN,
    control_target=0.55,
    control_reference=0.50,
    control_valid=True,
  )
  torch.testing.assert_close(reward(env), torch.tensor([0.0]))

  env.extras[STAIR_STRIDE_SWING_STRIDE_KEY] = torch.tensor([[0.55, 0.0]])
  assert reward(env).item() > 0.0

  env.extras[STAIR_STRIDE_SWING_STRIDE_KEY] = torch.tensor([[0.58, 0.0]])
  overrun = reward(env)

  assert overrun.item() < 0.0


def test_stride_phase_entry_swing_rewards_heel_clearance_delta() -> None:
  reward = _make_stride_phase_reward_for_test()
  env = _make_stride_phase_env(
    swing_stride=0.0,
    swing_valid=True,
    control_phase=STAIR_STRIDE_EVENT_ENTRY_TOUCHDOWN,
    control_target=0.05,
    control_reference=0.0,
    control_valid=True,
  )
  torch.testing.assert_close(reward(env), torch.tensor([0.0]))
  env.extras[STAIR_STRIDE_SWING_STRIDE_KEY] = torch.tensor([[0.05, 0.0]])
  swing_raw = reward(env).item() * env.step_dt
  repeat_raw = reward(env).item() * env.step_dt

  assert swing_raw == pytest.approx(0.70 * 0.75 * 5.50)
  assert repeat_raw == pytest.approx(0.0)


def test_stride_phase_entry_swing_over_target_emits_negative_delta() -> None:
  reward = _make_stride_phase_reward_for_test()
  env = _make_stride_phase_env(
    swing_stride=0.0,
    swing_valid=True,
    control_phase=STAIR_STRIDE_EVENT_ENTRY_TOUCHDOWN,
    control_target=0.05,
    control_reference=0.0,
    control_valid=True,
  )
  torch.testing.assert_close(reward(env), torch.tensor([0.0]))

  env.extras[STAIR_STRIDE_SWING_STRIDE_KEY] = torch.tensor([[0.05, 0.0]])
  assert reward(env).item() > 0.0

  env.extras[STAIR_STRIDE_SWING_STRIDE_KEY] = torch.tensor([[0.07, 0.0]])
  overrun = reward(env)

  assert overrun.item() < 0.0


def test_stride_phase_entry_quality_is_continuous_at_target() -> None:
  clearance = torch.tensor([0.05, 0.050001])
  quality = stair_stride_phase_reward._entry_swing_quality(
    clearance,
    entry_target=0.05,
    entry_reward_scale=0.75,
    probe_scale=3.0,
    convex_bonus=2.5,
    post_target_decay_scale=0.010,
    overshoot_penalty=1.0,
  )

  torch.testing.assert_close(quality[1], quality[0], atol=1.0e-5, rtol=1.0e-5)


def test_stride_phase_entry_touchdown_residual_preserves_event_score() -> None:
  reward = _make_stride_phase_reward_for_test()
  env = _make_stride_phase_env(
    swing_stride=0.0,
    swing_valid=True,
    control_phase=STAIR_STRIDE_EVENT_ENTRY_TOUCHDOWN,
    control_target=0.05,
    control_reference=0.0,
    control_valid=True,
  )
  torch.testing.assert_close(reward(env), torch.tensor([0.0]))
  env.extras[STAIR_STRIDE_SWING_STRIDE_KEY] = torch.tensor([[0.05, 0.0]])
  swing_raw = reward(env).item() * env.step_dt

  env.extras.update(
    {
      STAIR_STRIDE_PHASE_EVENT_ID_KEY: torch.tensor([1]),
      STAIR_STRIDE_PHASE_EVENT_TYPE_KEY: torch.tensor(
        [STAIR_STRIDE_EVENT_ENTRY_TOUCHDOWN]
      ),
      STAIR_STRIDE_PHASE_EVENT_ACTUAL_KEY: torch.tensor([0.05]),
      STAIR_STRIDE_PHASE_EVENT_PREVIOUS_KEY: torch.tensor([0.0]),
      STAIR_STRIDE_PHASE_EVENT_TARGET_KEY: torch.tensor([0.05]),
      STAIR_STRIDE_PHASE_EVENT_FOOT_ID_KEY: torch.tensor([0]),
      STAIR_STRIDE_PHASE_EVENT_COMPLETED_KEY: torch.tensor([True]),
      STAIR_STRIDE_SWING_VALID_KEY: torch.tensor([[False, False]]),
      STAIR_STRIDE_CONTROL_VALID_KEY: torch.tensor([False]),
    }
  )
  residual_raw = reward(env).item() * env.step_dt

  assert swing_raw == pytest.approx(0.70 * 0.75 * 5.50)
  assert swing_raw + residual_raw == pytest.approx(0.75 * (5.50 + 1.00))
  assert not reward._swing_active.any()


def test_stride_phase_valid_second_hit_commits_entry_credit_and_closes() -> None:
  reward = _make_stride_phase_reward_for_test()
  env = _make_stride_phase_env(
    swing_stride=0.0,
    swing_valid=True,
    control_phase=STAIR_STRIDE_EVENT_ENTRY_TOUCHDOWN,
    control_target=0.05,
    control_reference=0.0,
    control_valid=True,
  )
  reward(env)
  env.extras[STAIR_STRIDE_SWING_STRIDE_KEY] = torch.tensor([[0.05, 0.0]])
  reward(env)

  env.extras.update(
    {
      STAIR_STRIDE_PHASE_EVENT_ID_KEY: torch.tensor([1]),
      STAIR_STRIDE_PHASE_EVENT_TYPE_KEY: torch.tensor(
        [STAIR_STRIDE_EVENT_VALID_SECOND_HIT]
      ),
      STAIR_STRIDE_PHASE_EVENT_FOOT_ID_KEY: torch.tensor([0]),
      STAIR_STRIDE_SWING_STRIDE_KEY: torch.tensor([[0.06, 0.0]]),
      STAIR_STRIDE_SWING_VALID_KEY: torch.tensor([[True, False]]),
      STAIR_STRIDE_CONTROL_PHASE_KEY: torch.tensor(
        [STAIR_STRIDE_EVENT_BACKOFF_TOUCHDOWN]
      ),
      STAIR_STRIDE_CONTROL_TARGET_KEY: torch.tensor([0.55]),
      STAIR_STRIDE_CONTROL_REFERENCE_KEY: torch.tensor([0.60]),
      STAIR_STRIDE_CONTROL_VALID_KEY: torch.tensor([True]),
    }
  )
  confirmation_raw = reward(env).item() * env.step_dt

  assert confirmation_raw == pytest.approx(1.0)
  assert not reward._swing_active[0, 0].item()
  torch.testing.assert_close(reward._swing_pending_credit[0, 0], torch.tensor(0.0))


def test_stride_phase_backoff_swing_overrun_is_sharper_than_probe() -> None:
  stride = torch.tensor([0.56])
  probe = stair_stride_phase_reward._probe_swing_quality(
    stride,
    torch.tensor([0.50]),
    torch.tensor([0.55]),
    probe_scale=3.0,
    growth_scale=0.05,
    convex_bonus=2.5,
    post_target_decay_scale=0.012,
    reference_tolerance=0.025,
    overshoot_penalty=1.0,
  )
  backoff = stair_stride_phase_reward._backoff_swing_quality(
    stride,
    torch.tensor([0.55]),
    torch.tensor([0.60]),
    backoff_scale=4.0,
    window_min=0.025,
    window_max=0.050,
    forward_decay_scale=0.010,
    forward_penalty=0.50,
  )

  assert backoff.item() < 0.0
  assert backoff.item() < probe.item()


def test_stride_phase_backoff_swing_quality_spans_full_recovery_swing() -> None:
  stride = torch.tensor([0.10, 0.30, 0.50])
  quality = stair_stride_phase_reward._backoff_swing_quality(
    stride,
    torch.full_like(stride, 0.55),
    torch.zeros_like(stride),
    backoff_scale=4.0,
    window_min=0.025,
    window_max=0.050,
    forward_decay_scale=0.010,
    forward_penalty=0.50,
  )

  assert quality[0].item() > 0.0
  assert quality[2] - quality[1] > quality[1] - quality[0]


def test_stride_phase_lock_swing_quality_peaks_at_hold_target() -> None:
  stride = torch.tensor([0.55, 0.60, 0.65])
  quality = stair_stride_phase_reward._lock_swing_quality(
    stride,
    torch.full_like(stride, 0.60),
    lock_scale=2.0,
    tolerance=0.05,
  )

  assert quality[1] > quality[0]
  torch.testing.assert_close(quality[0], quality[2])


def test_stride_phase_lock_swing_and_touchdown_preserve_hold_score() -> None:
  reward = _make_stride_phase_reward_for_test()
  env = _make_stride_phase_env(
    swing_stride=0.50,
    swing_valid=True,
    control_phase=STAIR_STRIDE_EVENT_LOCK_TOUCHDOWN,
    control_target=0.60,
    control_reference=0.58,
    control_valid=True,
  )
  torch.testing.assert_close(reward(env), torch.tensor([0.0]))

  env.extras[STAIR_STRIDE_SWING_STRIDE_KEY] = torch.tensor([[0.60, 0.0]])
  swing_raw = reward(env).item() * env.step_dt
  env.extras.update(
    {
      STAIR_STRIDE_PHASE_EVENT_ID_KEY: torch.tensor([1]),
      STAIR_STRIDE_PHASE_EVENT_TYPE_KEY: torch.tensor(
        [STAIR_STRIDE_EVENT_LOCK_TOUCHDOWN]
      ),
      STAIR_STRIDE_PHASE_EVENT_ACTUAL_KEY: torch.tensor([0.60]),
      STAIR_STRIDE_PHASE_EVENT_PREVIOUS_KEY: torch.tensor([0.58]),
      STAIR_STRIDE_PHASE_EVENT_TARGET_KEY: torch.tensor([0.60]),
      STAIR_STRIDE_PHASE_EVENT_FOOT_ID_KEY: torch.tensor([0]),
      STAIR_STRIDE_PHASE_EVENT_COMPLETED_KEY: torch.tensor([True]),
      STAIR_STRIDE_SWING_VALID_KEY: torch.tensor([[False, False]]),
      STAIR_STRIDE_CONTROL_VALID_KEY: torch.tensor([False]),
    }
  )
  residual_raw = reward(env).item() * env.step_dt
  progress, tracking = stair_stride_phase_reward._lock_components(
    torch.tensor([0.60]),
    torch.tensor([0.58]),
    torch.tensor([0.60]),
    0.03,
    0.05,
  )
  expected = 2.0 * (0.40 * progress + 0.60 * tracking)

  assert swing_raw > 0.0
  assert swing_raw + residual_raw == pytest.approx(expected.item())
  assert not reward._swing_active.any()


def test_stride_phase_backoff_touchdown_preserves_sharp_overrun_score() -> None:
  reward = _make_stride_phase_reward_for_test()
  env = _make_stride_phase_env(
    swing_stride=0.54,
    swing_valid=True,
    control_phase=STAIR_STRIDE_EVENT_BACKOFF_TOUCHDOWN,
    control_target=0.55,
    control_reference=0.60,
    control_valid=True,
  )
  torch.testing.assert_close(reward(env), torch.tensor([0.0]))

  env.extras[STAIR_STRIDE_SWING_STRIDE_KEY] = torch.tensor([[0.56, 0.0]])
  swing_raw = reward(env).item() * env.step_dt
  env.extras.update(
    {
      STAIR_STRIDE_PHASE_EVENT_ID_KEY: torch.tensor([1]),
      STAIR_STRIDE_PHASE_EVENT_TYPE_KEY: torch.tensor(
        [STAIR_STRIDE_EVENT_BACKOFF_TOUCHDOWN]
      ),
      STAIR_STRIDE_PHASE_EVENT_ACTUAL_KEY: torch.tensor([0.56]),
      STAIR_STRIDE_PHASE_EVENT_PREVIOUS_KEY: torch.tensor([0.60]),
      STAIR_STRIDE_PHASE_EVENT_TARGET_KEY: torch.tensor([0.55]),
      STAIR_STRIDE_PHASE_EVENT_FOOT_ID_KEY: torch.tensor([0]),
      STAIR_STRIDE_PHASE_EVENT_COMPLETED_KEY: torch.tensor([True]),
      STAIR_STRIDE_SWING_VALID_KEY: torch.tensor([[False, False]]),
      STAIR_STRIDE_CONTROL_VALID_KEY: torch.tensor([False]),
    }
  )
  residual_raw = reward(env).item() * env.step_dt
  expected_quality = stair_stride_phase_reward._backoff_swing_quality(
    torch.tensor([0.56]),
    torch.tensor([0.55]),
    torch.tensor([0.60]),
    backoff_scale=4.0,
    window_min=0.025,
    window_max=0.050,
    forward_decay_scale=0.010,
    forward_penalty=0.50,
  ).item()
  expected_total = expected_quality + torch.exp(torch.tensor(-1.0)).item()

  assert swing_raw + residual_raw == pytest.approx(expected_total, abs=1.0e-6)
  assert expected_total < 0.0


def test_stride_phase_swing_snapshot_ignores_live_target_changes() -> None:
  reward = _make_stride_phase_reward_for_test()
  env = _make_stride_phase_env(
    swing_stride=0.53,
    swing_valid=True,
    control_phase=STAIR_STRIDE_EVENT_PROBE_TOUCHDOWN,
    control_target=0.55,
    control_reference=0.50,
    control_valid=True,
  )
  torch.testing.assert_close(reward(env), torch.tensor([0.0]))

  env.extras[STAIR_STRIDE_CONTROL_TARGET_KEY] = torch.tensor([0.65])
  unchanged = reward(env)

  torch.testing.assert_close(unchanged, torch.tensor([0.0]))


def test_stride_phase_swing_touchdown_residual_preserves_event_score() -> None:
  reward = _make_stride_phase_reward_for_test()
  env = _make_stride_phase_env(
    swing_stride=0.50,
    swing_valid=True,
    control_phase=STAIR_STRIDE_EVENT_PROBE_TOUCHDOWN,
    control_target=0.55,
    control_reference=0.50,
    control_valid=True,
  )
  torch.testing.assert_close(reward(env), torch.tensor([0.0]))
  env.extras[STAIR_STRIDE_SWING_STRIDE_KEY] = torch.tensor([[0.55, 0.0]])
  swing_raw = reward(env).item() * env.step_dt

  env.extras.update(
    {
      STAIR_STRIDE_PHASE_EVENT_ID_KEY: torch.tensor([1]),
      STAIR_STRIDE_PHASE_EVENT_TYPE_KEY: torch.tensor(
        [STAIR_STRIDE_EVENT_PROBE_TOUCHDOWN]
      ),
      STAIR_STRIDE_PHASE_EVENT_ACTUAL_KEY: torch.tensor([0.55]),
      STAIR_STRIDE_PHASE_EVENT_PREVIOUS_KEY: torch.tensor([0.50]),
      STAIR_STRIDE_PHASE_EVENT_TARGET_KEY: torch.tensor([0.55]),
      STAIR_STRIDE_PHASE_EVENT_FOOT_ID_KEY: torch.tensor([0]),
      STAIR_STRIDE_PHASE_EVENT_COMPLETED_KEY: torch.tensor([True]),
      STAIR_STRIDE_SWING_VALID_KEY: torch.tensor([[False, False]]),
      STAIR_STRIDE_CONTROL_VALID_KEY: torch.tensor([False]),
    }
  )
  residual_raw = reward(env).item() * env.step_dt

  assert swing_raw == pytest.approx(3.85)
  assert swing_raw + residual_raw == pytest.approx(6.50)
  assert not reward._swing_active.any()


def test_stride_phase_swing_invalid_cancels_positive_not_negative_credit() -> None:
  reward = _make_stride_phase_reward_for_test()
  env = _make_stride_phase_env(
    swing_stride=0.50,
    swing_valid=True,
    control_phase=STAIR_STRIDE_EVENT_PROBE_TOUCHDOWN,
    control_target=0.55,
    control_reference=0.50,
    control_valid=True,
  )
  torch.testing.assert_close(reward(env), torch.tensor([0.0]))
  env.extras[STAIR_STRIDE_SWING_STRIDE_KEY] = torch.tensor([[0.55, 0.0]])
  positive_raw = reward(env).item() * env.step_dt
  env.extras[STAIR_STRIDE_SWING_VALID_KEY] = torch.tensor([[False, False]])
  env.extras[STAIR_STRIDE_CONTROL_VALID_KEY] = torch.tensor([False])
  torch.testing.assert_close(reward(env), torch.tensor([0.0]))
  torch.testing.assert_close(reward(env), torch.tensor([0.0]))
  cancel_raw = reward(env).item() * env.step_dt
  assert positive_raw + cancel_raw == pytest.approx(0.0)

  reward = _make_stride_phase_reward_for_test()
  env = _make_stride_phase_env(
    swing_stride=0.50,
    swing_valid=True,
    control_phase=STAIR_STRIDE_EVENT_PROBE_TOUCHDOWN,
    control_target=0.55,
    control_reference=0.50,
    control_valid=True,
  )
  reward(env)
  env.extras[STAIR_STRIDE_SWING_STRIDE_KEY] = torch.tensor([[0.55, 0.0]])
  reward(env)
  env.extras[STAIR_STRIDE_SWING_STRIDE_KEY] = torch.tensor([[0.58, 0.0]])
  assert reward(env).item() < 0.0
  env.extras[STAIR_STRIDE_SWING_VALID_KEY] = torch.tensor([[False, False]])
  env.extras[STAIR_STRIDE_CONTROL_VALID_KEY] = torch.tensor([False])
  torch.testing.assert_close(reward(env), torch.tensor([0.0]))
  torch.testing.assert_close(reward(env), torch.tensor([0.0]))
  negative_cancel = reward(env)
  torch.testing.assert_close(negative_cancel, torch.tensor([0.0]))


def test_stride_phase_swing_survives_short_invalid_contact_gap() -> None:
  reward = _make_stride_phase_reward_for_test()
  env = _make_stride_phase_env(
    swing_stride=0.0,
    swing_valid=True,
    control_phase=STAIR_STRIDE_EVENT_PROBE_TOUCHDOWN,
    control_target=0.55,
    control_reference=0.50,
    control_valid=True,
  )
  torch.testing.assert_close(reward(env), torch.tensor([0.0]))
  env.extras[STAIR_STRIDE_SWING_STRIDE_KEY] = torch.tensor([[0.20, 0.0]])
  assert reward(env).item() > 0.0

  env.extras[STAIR_STRIDE_SWING_VALID_KEY] = torch.tensor([[False, False]])
  torch.testing.assert_close(reward(env), torch.tensor([0.0]))
  torch.testing.assert_close(reward(env), torch.tensor([0.0]))
  assert reward._swing_active[0, 0].item()

  env.extras[STAIR_STRIDE_SWING_VALID_KEY] = torch.tensor([[True, False]])
  env.extras[STAIR_STRIDE_SWING_STRIDE_KEY] = torch.tensor([[0.40, 0.0]])
  assert reward(env).item() > 0.0
  assert reward._swing_active[0, 0].item()


def test_stride_phase_valid_second_hit_commits_probe_credit_and_closes() -> None:
  reward = _make_stride_phase_reward_for_test()
  env = _make_stride_phase_env(
    swing_stride=0.50,
    swing_valid=True,
    control_phase=STAIR_STRIDE_EVENT_PROBE_TOUCHDOWN,
    control_target=0.55,
    control_reference=0.50,
    control_valid=True,
  )
  reward(env)
  env.extras[STAIR_STRIDE_SWING_STRIDE_KEY] = torch.tensor([[0.55, 0.0]])
  reward(env)

  env.extras.update(
    {
      STAIR_STRIDE_PHASE_EVENT_ID_KEY: torch.tensor([1]),
      STAIR_STRIDE_PHASE_EVENT_TYPE_KEY: torch.tensor(
        [STAIR_STRIDE_EVENT_VALID_SECOND_HIT]
      ),
      STAIR_STRIDE_PHASE_EVENT_FOOT_ID_KEY: torch.tensor([0]),
      STAIR_STRIDE_SWING_STRIDE_KEY: torch.tensor([[0.56, 0.0]]),
      STAIR_STRIDE_SWING_VALID_KEY: torch.tensor([[True, False]]),
      STAIR_STRIDE_CONTROL_PHASE_KEY: torch.tensor(
        [STAIR_STRIDE_EVENT_BACKOFF_TOUCHDOWN]
      ),
      STAIR_STRIDE_CONTROL_TARGET_KEY: torch.tensor([0.55]),
      STAIR_STRIDE_CONTROL_REFERENCE_KEY: torch.tensor([0.60]),
      STAIR_STRIDE_CONTROL_VALID_KEY: torch.tensor([True]),
    }
  )
  confirmation_raw = reward(env).item() * env.step_dt

  assert confirmation_raw == pytest.approx(1.0)
  assert not reward._swing_active[0, 0].item()
  torch.testing.assert_close(reward._swing_pending_credit[0, 0], torch.tensor(0.0))


def test_stride_phase_swing_inactive_contexts_and_reset_are_zero() -> None:
  reward = _make_stride_phase_reward_for_test()
  env = _make_stride_phase_env(
    swing_stride=0.53,
    swing_valid=True,
    control_phase=STAIR_STRIDE_EVENT_NONE,
    control_target=0.55,
    control_reference=0.50,
    control_valid=False,
  )

  torch.testing.assert_close(reward(env), torch.tensor([0.0]))
  reward._swing_active[:] = True
  reward._swing_phase_snapshot[:] = STAIR_STRIDE_EVENT_PROBE_TOUCHDOWN
  reward._swing_target_snapshot[:] = 0.55
  reward._swing_reference_snapshot[:] = 0.50
  reward._swing_last_quality[:] = 1.0
  reward._swing_pending_credit[:] = 1.0

  reward.reset()

  assert not reward._swing_active.any()
  torch.testing.assert_close(reward._swing_target_snapshot, torch.zeros(1, 2))
  torch.testing.assert_close(reward._swing_pending_credit, torch.zeros(1, 2))


def test_stride_phase_event_reward_penalizes_stall_in_active_direction() -> None:
  reward = object.__new__(stair_stride_phase_reward)
  reward._last_event_id = torch.zeros(1, dtype=torch.long)
  reward._event_producer = None
  env = SimpleNamespace(
    num_envs=1,
    device=torch.device("cpu"),
    step_dt=0.02,
    extras={
      STAIR_STRIDE_PHASE_EVENT_ID_KEY: torch.tensor([1]),
      STAIR_STRIDE_PHASE_EVENT_TYPE_KEY: torch.tensor(
        [STAIR_STRIDE_EVENT_PROBE_TOUCHDOWN]
      ),
      STAIR_STRIDE_PHASE_EVENT_ACTUAL_KEY: torch.tensor([0.50]),
      STAIR_STRIDE_PHASE_EVENT_PREVIOUS_KEY: torch.tensor([0.50]),
      STAIR_STRIDE_PHASE_EVENT_TARGET_KEY: torch.tensor([0.55]),
      STAIR_STRIDE_PHASE_EVENT_INTENT_KEY: torch.tensor([STAIR_STRIDE_INTENT_FARTHER]),
      STAIR_STRIDE_PHASE_EVENT_COMPLETED_KEY: torch.tensor([False]),
      STAIR_STRIDE_PHASE_EVENT_COLLISION_PENALTY_KEY: torch.tensor([0.0]),
    },
  )

  stalled_probe = reward(env)
  assert stalled_probe.item() < 0.0

  env.extras[STAIR_STRIDE_PHASE_EVENT_ID_KEY] = torch.tensor([2])
  env.extras[STAIR_STRIDE_PHASE_EVENT_ACTUAL_KEY] = torch.tensor([0.55])
  env.extras[STAIR_STRIDE_PHASE_EVENT_COMPLETED_KEY] = torch.tensor([True])
  completed_probe = reward(env)
  assert completed_probe.item() > 0.0

  env.extras[STAIR_STRIDE_PHASE_EVENT_ID_KEY] = torch.tensor([3])
  env.extras[STAIR_STRIDE_PHASE_EVENT_TYPE_KEY] = torch.tensor(
    [STAIR_STRIDE_EVENT_BACKOFF_TOUCHDOWN]
  )
  env.extras[STAIR_STRIDE_PHASE_EVENT_ACTUAL_KEY] = torch.tensor([0.70])
  env.extras[STAIR_STRIDE_PHASE_EVENT_PREVIOUS_KEY] = torch.tensor([0.70])
  env.extras[STAIR_STRIDE_PHASE_EVENT_TARGET_KEY] = torch.tensor([0.60])
  env.extras[STAIR_STRIDE_PHASE_EVENT_COMPLETED_KEY] = torch.tensor([False])
  stalled_backoff = reward(env)
  assert stalled_backoff.item() < 0.0

  env.extras[STAIR_STRIDE_PHASE_EVENT_ID_KEY] = torch.tensor([4])
  env.extras[STAIR_STRIDE_PHASE_EVENT_ACTUAL_KEY] = torch.tensor([0.605])
  improving_backoff = reward(env)
  assert improving_backoff.item() > 0.0

  env.extras[STAIR_STRIDE_PHASE_EVENT_ID_KEY] = torch.tensor([5])
  env.extras[STAIR_STRIDE_PHASE_EVENT_ACTUAL_KEY] = torch.tensor([0.50])
  overretreating_backoff = reward(env)
  assert overretreating_backoff.item() < 0.0


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
  assert vis.slab_depth == 0.05
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
      arm_pose_weight=0.9,
      arm_joint_vel_l2_weight=-0.003,
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
  assert env_cfg.rewards["arm_pose"].weight == 0.9
  assert env_cfg.rewards["arm_pose"].func is posture
  assert env_cfg.rewards["arm_pose"].params["asset_cfg"].joint_names == (
    r".*shoulder.*",
    r".*elbow.*",
    r".*wrist.*",
  )
  assert env_cfg.rewards["arm_joint_vel_l2"].weight == -0.003
  assert env_cfg.rewards["arm_joint_vel_l2"].func is joint_vel_l2
  assert "foot_step_lip_volume_penalty" not in env_cfg.rewards
  assert env_cfg.rewards["toe_step_riser_slab_penalty"].params["slab_depth"] == 0.12
  toe_params = env_cfg.rewards["toe_step_riser_slab_penalty"].params
  assert toe_params["contact_penalty_scale"] == 0.7
  assert toe_params["stair_min_safe_stride"] == 0.14
  assert toe_params["stair_max_safe_stride"] == 0.85
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
