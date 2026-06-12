"""Generic tests for task config integrity."""

import math

import pytest

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.mdp.teacher_target_heading_command import (
  TeacherTargetHeadingVelocityCommandCfg,
)

MAIN_BRANCH_TASK_IDS = (
  "Mjlab-Velocity-Blind-Rough-Perception-PPO-Unitree-G1",
  "Mjlab-Velocity-Blind-Rough-Perception-TeacherKL-Unitree-G1",
  "Mjlab-Velocity-Blind-Rough-TargetNavigation-Perception-TeacherKL-Unitree-G1",
  "Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1",
)


@pytest.fixture(scope="module")
def all_task_ids() -> list[str]:
  """Get all registered task IDs."""
  return list_tasks()


def test_only_main_branch_tasks_registered(all_task_ids: list[str]) -> None:
  """This branch exposes only the selected G1 rough-terrain tasks."""
  assert all_task_ids == sorted(MAIN_BRANCH_TASK_IDS)


def test_all_tasks_loadable(all_task_ids: list[str]) -> None:
  """All registered tasks should be loadable without errors."""
  for task_id in all_task_ids:
    try:
      cfg = load_env_cfg(task_id)
      assert isinstance(cfg, ManagerBasedRlEnvCfg), (
        f"Task {task_id} did not return ManagerBasedRlEnvCfg"
      )
    except Exception as e:
      pytest.fail(f"Failed to load task '{task_id}': {e}")


def test_all_tasks_have_play_config(all_task_ids: list[str]) -> None:
  """All tasks should be loadable in play mode."""
  for task_id in all_task_ids:
    try:
      cfg = load_env_cfg(task_id, play=True)
      assert isinstance(cfg, ManagerBasedRlEnvCfg), (
        f"Task {task_id} play mode did not return ManagerBasedRlEnvCfg"
      )
    except Exception as e:
      pytest.fail(f"Failed to load task '{task_id}' in play mode: {e}")


def test_play_mode_episode_length(all_task_ids: list[str]) -> None:
  """Play mode tasks should have infinite episode length."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id, play=True)
    assert cfg.episode_length_s >= 1e9, (
      f"{task_id} (play mode) episode_length_s={cfg.episode_length_s}, expected >= 1e9"
    )


def test_play_mode_observation_corruption_disabled(all_task_ids: list[str]) -> None:
  """Play mode tasks should have observation corruption disabled for policy."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id, play=True)

    assert "actor" in cfg.observations, (
      f"Play mode task {task_id} missing 'policy' observation group"
    )

    policy_obs = cfg.observations["actor"]
    assert isinstance(policy_obs, ObservationGroupCfg), (
      f"Play mode task {task_id} policy observation is not ObservationGroupCfg"
    )

    assert not policy_obs.enable_corruption, (
      f"Play mode task {task_id} has enable_corruption=True, expected False"
    )


def test_training_mode_observation_corruption_enabled(all_task_ids: list[str]) -> None:
  """Training mode tasks should have observation corruption enabled for policy."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id)

    assert "actor" in cfg.observations, (
      f"Training task {task_id} missing 'policy' observation group"
    )

    policy_obs = cfg.observations["actor"]
    assert isinstance(policy_obs, ObservationGroupCfg), (
      f"Training task {task_id} policy observation is not ObservationGroupCfg"
    )

    assert policy_obs.enable_corruption, (
      f"Training task {task_id} has enable_corruption=False, expected True"
    )


def test_critic_observation_corruption_always_disabled(all_task_ids: list[str]) -> None:
  """Critic observations should always have corruption disabled."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id)

    if "critic" not in cfg.observations:
      continue

    critic_obs = cfg.observations["critic"]
    assert isinstance(critic_obs, ObservationGroupCfg), (
      f"Task {task_id} critic observation is not ObservationGroupCfg"
    )

    assert not critic_obs.enable_corruption, (
      f"Task {task_id} has critic enable_corruption=True, expected False"
    )


def test_play_training_observation_structure_match(all_task_ids: list[str]) -> None:
  """Play and training configs should have matching observation structure."""
  for task_id in all_task_ids:
    training_cfg = load_env_cfg(task_id)
    play_cfg = load_env_cfg(task_id, play=True)

    # Same observation groups.
    assert set(training_cfg.observations.keys()) == set(play_cfg.observations.keys()), (
      f"Observation groups mismatch between {task_id} training and play modes"
    )

    # Same observation terms within each group.
    for obs_group_name in training_cfg.observations:
      training_terms = set(training_cfg.observations[obs_group_name].terms.keys())
      play_terms = set(play_cfg.observations[obs_group_name].terms.keys())

      assert training_terms == play_terms, (
        f"Observation terms mismatch in group '{obs_group_name}' "
        f"between {task_id} training and play modes"
      )


def test_play_training_action_structure_match(all_task_ids: list[str]) -> None:
  """Play and training configs should have matching action structure."""
  for task_id in all_task_ids:
    training_cfg = load_env_cfg(task_id)
    play_cfg = load_env_cfg(task_id, play=True)

    assert set(training_cfg.actions.keys()) == set(play_cfg.actions.keys()), (
      f"Action structure mismatch between {task_id} training and play modes"
    )


def test_play_mode_disables_push_robot(all_task_ids: list[str]) -> None:
  """Play mode tasks should disable push_robot event."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id, play=True)
    assert "push_robot" not in cfg.events, (
      f"Play mode task {task_id} has push_robot event, expected it to be removed"
    )


def test_step_boundary_rewards_scoped_to_target_stair_tasks(
  all_task_ids: list[str],
) -> None:
  """Perception tasks should include all 5 step-boundary rewards."""
  step_reward_names = {
    "foot_step_lip_volume_penalty",
    "toe_step_riser_slab_penalty",
    "heel_step_riser_clearance_penalty",
    "foot_landing_flatness_penalty",
    "shank_step_lip_proximity_penalty",
  }
  perception_tasks = {
    "Mjlab-Velocity-Blind-Rough-Perception-PPO-Unitree-G1",
    "Mjlab-Velocity-Blind-Rough-Perception-TeacherKL-Unitree-G1",
    "Mjlab-Velocity-Blind-Rough-TargetNavigation-Perception-TeacherKL-Unitree-G1",
  }
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id)
    present = step_reward_names.intersection(cfg.rewards)
    if task_id in perception_tasks:
      assert present == step_reward_names, (
        f"{task_id} missing perception step rewards: {step_reward_names - present}"
      )
      assert (
        cfg.rewards["foot_step_lip_volume_penalty"].params["min_terrain_level"] == 3
      )
      assert (
        cfg.rewards["foot_step_lip_volume_penalty"].params["nearest_boundaries"] == 4
      )
      assert cfg.rewards["toe_step_riser_slab_penalty"].params["min_terrain_level"] == 3
      assert (
        cfg.rewards["toe_step_riser_slab_penalty"].params["nearest_boundaries"] == 4
      )
      assert cfg.rewards["toe_step_riser_slab_penalty"].params["slab_depth"] == 0.10
    else:
      # Blind-Rough-TeacherKL should NOT have step-boundary rewards
      assert not present, f"{task_id} unexpectedly enables {sorted(present)}"


def test_perception_ppo_task_removes_teacher_guidance() -> None:
  """Pure PPO perception task keeps student inputs and target navigation."""
  task_id = "Mjlab-Velocity-Blind-Rough-Perception-PPO-Unitree-G1"
  env_cfg = load_env_cfg(task_id)
  play_cfg = load_env_cfg(task_id, play=True)
  rl_cfg = load_rl_cfg(task_id)

  assert isinstance(env_cfg.commands["twist"], TeacherTargetHeadingVelocityCommandCfg)
  assert isinstance(play_cfg.commands["twist"], TeacherTargetHeadingVelocityCommandCfg)
  assert "target_progress" in env_cfg.rewards
  assert "target_reached_bonus" in env_cfg.rewards
  assert play_cfg.viewer.show_depth_camera_visualizers is True

  assert "actor" in env_cfg.observations
  assert "critic" in env_cfg.observations
  assert "camera_stack" in env_cfg.observations
  assert "teacher" not in env_cfg.observations
  assert "camera" not in env_cfg.observations
  assert "teacher" not in play_cfg.observations
  assert "camera" not in play_cfg.observations
  camera_sensors = [
    sensor
    for sensor in env_cfg.scene.sensors or ()
    if isinstance(sensor, CameraSensorCfg)
  ]
  play_camera_sensors = [
    sensor
    for sensor in play_cfg.scene.sensors or ()
    if isinstance(sensor, CameraSensorCfg)
  ]
  assert [sensor.name for sensor in camera_sensors] == ["front_depth_stack"]
  assert [sensor.name for sensor in play_camera_sensors] == ["front_depth_stack"]
  student_sensor = next(
    sensor for sensor in camera_sensors if sensor.name == "front_depth_stack"
  )
  play_student_sensor = next(
    sensor for sensor in play_camera_sensors if sensor.name == "front_depth_stack"
  )
  assert student_sensor.fovy == 55.2
  assert student_sensor.pos == (0.10, 0.0, 0.45)
  assert student_sensor.quat == pytest.approx(
    (
      math.cos(math.radians(42.4) * 0.5),
      0.0,
      -math.sin(math.radians(42.4) * 0.5),
      0.0,
    )
  )
  assert student_sensor.width == 64
  assert student_sensor.height == 36
  assert student_sensor.visualizer_max_range == 3.0
  assert play_student_sensor.fovy == 55.2
  assert play_student_sensor.pos == student_sensor.pos
  assert play_student_sensor.quat == pytest.approx(student_sensor.quat)
  assert play_student_sensor.width == 64
  assert play_student_sensor.height == 36
  assert play_student_sensor.visualizer_max_range == 3.0

  assert play_cfg.scene.terrain is not None
  assert play_cfg.scene.terrain.terrain_generator is not None
  assert play_cfg.scene.terrain.visualize_flat_patches is True
  assert play_cfg.scene.terrain.max_flat_patch_sites_per_tile == 80
  assert (
    play_cfg.scene.terrain.terrain_generator.step_danger_visualization.enabled is True
  )
  assert (
    play_cfg.rewards["foot_step_lip_volume_penalty"].params.get(
      "debug_vis_foot_points",
      False,
    )
    is True
  )

  assert type(rl_cfg) is RslRlOnPolicyRunnerCfg
  assert type(rl_cfg.algorithm) is RslRlPpoAlgorithmCfg
  assert not hasattr(rl_cfg.algorithm, "teacher_kl_cfg")
  assert rl_cfg.obs_groups == {
    "actor": ("actor", "camera_stack"),
    "critic": ("critic",),
  }
