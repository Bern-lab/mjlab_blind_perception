"""Generic tests for task config integrity."""

import pytest

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg

MAIN_BRANCH_TASK_IDS = (
  "Mjlab-Velocity-Blind-Rough-TargetNavigation-DWAQ-TeacherKL-Unitree-G1",
)


@pytest.fixture(scope="module")
def all_task_ids() -> list[str]:
  """Get all registered task IDs."""
  return list_tasks()


def test_only_main_branch_tasks_registered(all_task_ids: list[str]) -> None:
  """This branch intentionally exposes only the main TeacherKL tasks."""
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
  """Step-boundary danger rewards should stay scoped to stair target tasks."""
  dwaq_task = "Mjlab-Velocity-Blind-Rough-TargetNavigation-DWAQ-TeacherKL-Unitree-G1"
  lip_penalty_tasks: set[str] = set()
  slab_penalty_tasks = {dwaq_task}
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id)
    if task_id in lip_penalty_tasks:
      assert "foot_step_lip_volume_penalty" in cfg.rewards
      assert (
        cfg.rewards["foot_step_lip_volume_penalty"].params["min_terrain_level"] == 3
      )
      assert (
        cfg.rewards["foot_step_lip_volume_penalty"].params["nearest_boundaries"] == 4
      )
    else:
      assert "foot_step_lip_volume_penalty" not in cfg.rewards

    if task_id in slab_penalty_tasks:
      assert "toe_step_riser_slab_penalty" in cfg.rewards
      slab_params = cfg.rewards["toe_step_riser_slab_penalty"].params
      assert slab_params["min_terrain_level"] == 3
      assert slab_params["nearest_boundaries"] == 4
      assert slab_params["slab_depth"] == 0.10
    else:
      assert "toe_step_riser_slab_penalty" not in cfg.rewards
