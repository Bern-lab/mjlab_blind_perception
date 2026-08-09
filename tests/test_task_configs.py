"""Generic tests for task config integrity."""

import pytest

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg


@pytest.fixture(scope="module")
def all_task_ids() -> list[str]:
  """Get all registered task IDs."""
  return list_tasks()


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
  """SlowLatent ablations should share the same stair reward surface."""
  slow_latent_ablation_tasks = {
    "Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1",
    "Mjlab-Velocity-Blind-StairsFlag-TeacherKL-Unitree-G1",
    "Mjlab-Velocity-Blind-StairsFlag-LSTM-TeacherKL-Unitree-G1",
    "Mjlab-Velocity-Blind-Rough-LSTM-TeacherKL-Unitree-G1",
    "Mjlab-Velocity-Blind-Rough-Transformer-TeacherKL-Unitree-G1",
  }
  slow_latent_rewards = {
    "toe_step_riser_slab_penalty",
    "stair_stride_phase_reward",
    "shank_front_edge_clearance_penalty",
    "stair_skip_layer_penalty",
    "target_tread_midline_shaping",
  }
  target_heading_teacher_task = "Mjlab-Velocity-TargetHeading-Rough-Teacher-Unitree-G1"

  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id)
    present = slow_latent_rewards.intersection(cfg.rewards)
    if task_id in slow_latent_ablation_tasks:
      assert present == slow_latent_rewards
      assert "foot_step_lip_volume_penalty" not in cfg.rewards
      assert "toe_riser_contact_memory_penalty" not in cfg.rewards
      assert cfg.rewards["toe_step_riser_slab_penalty"].weight == -4.2
      assert cfg.rewards["shank_front_edge_clearance_penalty"].weight == -3.0
      assert cfg.rewards["stair_skip_layer_penalty"].weight == -1.0
      assert cfg.rewards["target_tread_midline_shaping"].weight == 1.2
    elif task_id == target_heading_teacher_task:
      assert present == {"toe_step_riser_slab_penalty"}
    else:
      assert not present, f"{task_id} unexpectedly enables {sorted(present)}"
