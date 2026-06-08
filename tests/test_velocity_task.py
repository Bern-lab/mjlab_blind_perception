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
  unitree_g1_blind_rough_target_navigation_slow_latent_env_cfg,
)
from mjlab.tasks.velocity.config.g1.rl_cfg import (
  G1SlowLatentPolicyModelParams,
  G1SlowLatentRunnerParams,
  unitree_g1_blind_rough_target_navigation_slow_latent_teacherkl_runner_cfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.teacher_target_heading_command import (
  TeacherTargetHeadingVelocityCommandCfg,
)

MAIN_BRANCH_VELOCITY_TASK_IDS = (
  "Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1",
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


def test_rough_velocity_play_has_curriculum_disabled() -> None:
  """Rough velocity play tasks should have terrain curriculum disabled."""
  for task_id in MAIN_BRANCH_VELOCITY_TASK_IDS:
    cfg = load_env_cfg(task_id, play=True)

    assert cfg.scene.terrain is not None, (
      f"Task {task_id} (play mode) has no terrain config"
    )
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} (play mode) has no terrain_generator"
    )
    assert cfg.scene.terrain.terrain_generator.curriculum is False, (
      f"Task {task_id} (play mode) curriculum={cfg.scene.terrain.terrain_generator.curriculum}, "
      "expected False"
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


def test_slow_latent_target_navigation_exposes_latent_inputs() -> None:
  task_id = (
    "Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1"
  )
  env_cfg = load_env_cfg(task_id)
  rl_cfg = cast(RslRlTeacherKLRunnerCfg, load_rl_cfg(task_id))

  assert "latent" in env_cfg.observations
  assert "stair_latent" in env_cfg.observations["latent"].terms
  assert "latent_labels" in env_cfg.observations
  assert set(env_cfg.observations["latent_labels"].terms) == {
    "toe_riser_event",
    "stair_state",
  }
  assert "reset_stair_latent_cache" in env_cfg.events
  assert rl_cfg.obs_groups["actor"] == ("actor",)
  assert rl_cfg.obs_groups["latent"] == ("latent",)
  foot_asset_cfg = env_cfg.rewards["foot_step_lip_volume_penalty"].params["asset_cfg"]
  toe_reward_params = env_cfg.rewards["toe_step_riser_slab_penalty"].params
  toe_asset_cfg = toe_reward_params["asset_cfg"]
  assert foot_asset_cfg is not toe_asset_cfg
  assert toe_reward_params["contact_sensor_name"] == "toe_terrain_contact"
  assert toe_reward_params["contact_penalty_scale"] == 0.5
  assert toe_reward_params["probe_contact_count"] == 2
  assert toe_reward_params["probe_slab_reward_scale"] == 0.35
  assert toe_reward_params["probe_contact_reward"] == 0.08
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
  assert actor_cfg.latent_hidden_dim == 128
  assert actor_cfg.mlp_encoder_dims == (128, 128)
  assert actor_cfg.aux_event_coef == 0.03
  assert actor_cfg.aux_stair_coef == 0.02
  assert actor_cfg.aux_future_collision_coef == 0.05


def test_blind_teacherkl_play_hides_exteroceptive_visualizers() -> None:
  task_ids = (
    "Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1",
    "Mjlab-Velocity-Blind-Rough-TargetNavigation-TeacherKL-Unitree-G1",
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


def test_slow_latent_explicit_param_interfaces_drive_configs() -> None:
  env_params = G1SlowLatentEnvParams(
    actor_history_length=7,
    rewards=G1SlowLatentRewardParams(
      foot_lip_edge_radius=0.08,
      toe_slab_depth=0.12,
      toe_contact_penalty_scale=0.7,
      toe_probe_contact_count=3,
      toe_probe_slab_reward_scale=0.5,
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

  assert env_cfg.observations["actor"].history_length == 7
  assert env_cfg.rewards["foot_step_lip_volume_penalty"].params["edge_radius"] == 0.08
  assert env_cfg.rewards["toe_step_riser_slab_penalty"].params["slab_depth"] == 0.12
  toe_params = env_cfg.rewards["toe_step_riser_slab_penalty"].params
  assert toe_params["contact_penalty_scale"] == 0.7
  assert toe_params["probe_contact_count"] == 3
  assert toe_params["probe_slab_reward_scale"] == 0.5
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
      latent_hidden_dim=64,
      mlp_encoder_dims=(64,),
      aux_stair_coef=0.7,
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
  assert actor_cfg.latent_hidden_dim == 64
  assert actor_cfg.mlp_encoder_dims == (64,)
  assert actor_cfg.aux_stair_coef == 0.7


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


def test_teacherkl_uses_delayed_mean_huber_guidance() -> None:
  """Teacher-KL variants should use delayed weak action-mean guidance."""
  velocity_cfg = cast(
    RslRlTeacherKLRunnerCfg,
    load_rl_cfg("Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1"),
  )
  target_cfg = cast(
    RslRlTeacherKLRunnerCfg,
    load_rl_cfg("Mjlab-Velocity-Blind-Rough-TargetNavigation-TeacherKL-Unitree-G1"),
  )

  for cfg in (velocity_cfg, target_cfg):
    algorithm_cfg = cast(RslRlPpoTeacherKLAlgorithmCfg, cfg.algorithm)
    teacher_cfg = algorithm_cfg.teacher_kl_cfg
    assert cfg.obs_groups["teacher"] == ("teacher", "camera")
    assert teacher_cfg.enabled is True
    assert teacher_cfg.imitation_only is False
    assert teacher_cfg.imitation_loss_coef == 1.0
    assert teacher_cfg.loss_type == "mean_huber"
    assert teacher_cfg.lambda_start == 0.03
    assert teacher_cfg.lambda_end == 0.0
    assert teacher_cfg.warmup_iters == 1000
    assert teacher_cfg.anneal_iters == 10000
    assert teacher_cfg.huber_delta == 0.5
    assert teacher_cfg.max_teacher_loss is None
    assert teacher_cfg.max_kl_loss is None
