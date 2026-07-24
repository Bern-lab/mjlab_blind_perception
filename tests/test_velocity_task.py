"""Tests specific to velocity tasks."""

from dataclasses import asdict
from typing import cast

import pytest
import torch
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, GO1_ACTION_SCALE
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.rl.config import (
  RslRlModelCfg,
  RslRlPpoTeacherKLAlgorithmCfg,
  RslRlTeacherKLRunnerCfg,
)
from mjlab.scripts.train import _apply_actor_history_length_override
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.g1.blind_rough_lstm_teacher_kl_env_cfg import (
  unitree_g1_blind_rough_lstm_teacherkl_env_cfg,
  unitree_g1_blind_rough_target_navigation_ablation_env_cfg,
)
from mjlab.tasks.velocity.config.g1.blind_stairs_flag_teacher_kl_env_cfg import (
  unitree_g1_blind_stairs_flag_teacherkl_env_cfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.teacher_target_heading_command import (
  TeacherTargetHeadingVelocityCommandCfg,
)

SLOW_LATENT_ABLATION_TASK_IDS = (
  "Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1",
  "Mjlab-Velocity-Blind-StairsFlag-TeacherKL-Unitree-G1",
  "Mjlab-Velocity-Blind-StairsFlag-LSTM-TeacherKL-Unitree-G1",
  "Mjlab-Velocity-Blind-Rough-LSTM-TeacherKL-Unitree-G1",
  "Mjlab-Velocity-Blind-Rough-Transformer-TeacherKL-Unitree-G1",
)

BOOLEAN_ABLATION_TASK_IDS = (
  "Mjlab-Velocity-Blind-StairsFlag-TeacherKL-Unitree-G1",
  "Mjlab-Velocity-Blind-StairsFlag-LSTM-TeacherKL-Unitree-G1",
)


@pytest.fixture(scope="module")
def velocity_task_ids() -> list[str]:
  """Get all velocity task IDs."""
  return [t for t in list_tasks() if "Velocity" in t]


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


def test_g1_teacherkl_ablation_tasks_match_slowlatent_conditions() -> None:
  """Existing TeacherKL ablation task IDs should share SlowLatent conditions."""
  expected_terrain_names = (
    "flat",
    "high_stairs_inv_w00",
    "high_stairs_inv_w01",
    "high_stairs_inv_w02",
    "high_stairs_inv_w03",
    "high_stairs_inv_w04",
    "high_stairs_inv_w05",
    "high_stairs_inv_w06",
    "high_stairs_inv_w07",
    "gentle_slope",
    "low_rough",
  )
  expected_step_rewards = {
    "toe_step_riser_slab_penalty": -4.2,
    "shank_front_edge_clearance_penalty": -3.0,
    "stair_skip_layer_penalty": -1.0,
    "target_tread_midline_shaping": 1.2,
  }

  for task_id in SLOW_LATENT_ABLATION_TASK_IDS:
    cfg = load_env_cfg(task_id)
    rl_cfg = cast(RslRlTeacherKLRunnerCfg, load_rl_cfg(task_id))
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, TeacherTargetHeadingVelocityCommandCfg)
    assert twist_cmd.resampling_time_range == (60.0, 60.0)
    assert twist_cmd.rel_target_envs == 0.8
    assert twist_cmd.rel_random_heading_envs == 0.0
    assert twist_cmd.rel_standing_envs == 0.2
    assert twist_cmd.ranges.lin_vel_x == (0.4, 1.0)
    assert twist_cmd.ranges.lin_vel_y == (0.0, 0.0)
    assert twist_cmd.ranges.ang_vel_z == (-0.8, 0.8)

    assert cfg.scene.terrain is not None
    terrain_generator = cfg.scene.terrain.terrain_generator
    assert terrain_generator is not None
    assert tuple(terrain_generator.sub_terrains) == expected_terrain_names
    assert terrain_generator.curriculum is True
    assert cfg.scene.terrain.max_init_terrain_level == 2

    assert "latent" not in cfg.observations
    assert "latent_labels" not in cfg.observations
    assert "reset_stair_latent_cache" not in cfg.events
    assert "toe_riser_contact_memory_penalty" not in cfg.rewards
    for reward_name, weight in expected_step_rewards.items():
      assert cfg.rewards[reward_name].weight == weight

    velocity_stages = cfg.curriculum["command_vel"].params["velocity_stages"]
    assert velocity_stages[0]["lin_vel_x"] == (0.4, 0.8)
    assert velocity_stages[1]["lin_vel_x"] == (0.4, 1.0)

    assert rl_cfg.num_steps_per_env == 64
    assert rl_cfg.save_interval == 500
    algorithm_cfg = cast(RslRlPpoTeacherKLAlgorithmCfg, rl_cfg.algorithm)
    teacher_cfg = algorithm_cfg.teacher_kl_cfg
    assert teacher_cfg.checkpoint_path == (
      "teacher_policies/g1_target_heading_depth_teacher/model_118200.pt"
    )
    assert teacher_cfg.loss_type == "mean_huber"
    assert teacher_cfg.lambda_start == 0.05
    assert teacher_cfg.warmup_iters == 0


def test_boolean_and_lstm_ablation_boundaries() -> None:
  """Only the intended actor inputs/classes differ across the ablation tasks."""
  plain_mlp_task = "Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1"
  plain_lstm_task = "Mjlab-Velocity-Blind-Rough-LSTM-TeacherKL-Unitree-G1"
  transformer_task = "Mjlab-Velocity-Blind-Rough-Transformer-TeacherKL-Unitree-G1"
  bool_mlp_task = "Mjlab-Velocity-Blind-StairsFlag-TeacherKL-Unitree-G1"
  bool_lstm_task = "Mjlab-Velocity-Blind-StairsFlag-LSTM-TeacherKL-Unitree-G1"

  plain_terms = set(load_env_cfg(plain_mlp_task).observations["actor"].terms)
  assert set(load_env_cfg(transformer_task).observations["actor"].terms) == plain_terms
  for task_id in BOOLEAN_ABLATION_TASK_IDS:
    actor_group = load_env_cfg(task_id).observations["actor"]
    assert set(actor_group.terms) == plain_terms | {"terrain_is_stairs"}
    assert actor_group.history_length is None
    assert actor_group.terms["terrain_is_stairs"].history_length == 0
    for name, term in actor_group.terms.items():
      if name != "terrain_is_stairs":
        assert term.history_length == 5

  assert cast(
    RslRlTeacherKLRunnerCfg, load_rl_cfg(plain_mlp_task)
  ).actor.class_name == ("MLPModel")
  assert cast(RslRlTeacherKLRunnerCfg, load_rl_cfg(bool_mlp_task)).actor.class_name == (
    "MLPModel"
  )
  assert cast(
    RslRlTeacherKLRunnerCfg, load_rl_cfg(plain_lstm_task)
  ).actor.class_name == ("RNNModel")
  transformer_actor_cfg = cast(
    RslRlTeacherKLRunnerCfg, load_rl_cfg(transformer_task)
  ).actor
  assert transformer_actor_cfg.class_name == "CausalTransformerModel"
  assert transformer_actor_cfg.transformer_cfg is not None
  assert "sequence_length" not in transformer_actor_cfg.transformer_cfg
  assert transformer_actor_cfg.transformer_cfg["token_dims"] == (
    3,
    3,
    3,
    2,
    29,
    29,
    29,
  )
  assert cast(
    RslRlTeacherKLRunnerCfg, load_rl_cfg(bool_lstm_task)
  ).actor.class_name == ("RNNModel")


def _make_actor_from_model_cfg(model_cfg: RslRlModelCfg, obs_dim: int = 490):
  cfg = asdict(model_cfg)
  model_class = resolve_callable(cfg.pop("class_name"))
  for opt in ("cnn_cfg", "distribution_cfg", "transformer_cfg"):
    if cfg.get(opt) is None:
      cfg.pop(opt, None)
  if cfg.get("rnn_type") is None:
    for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
      cfg.pop(opt, None)

  obs = TensorDict({"actor": torch.zeros(2, obs_dim)})
  return model_class(
    obs,
    {"actor": ["actor"]},
    "actor",
    29,
    **cfg,
  )


def test_transformer_actor_capacity_matches_lstm_ablation() -> None:
  """Transformer ablation should stay close to the LSTM actor capacity."""
  lstm_task = "Mjlab-Velocity-Blind-Rough-LSTM-TeacherKL-Unitree-G1"
  transformer_task = "Mjlab-Velocity-Blind-Rough-Transformer-TeacherKL-Unitree-G1"
  lstm_model = _make_actor_from_model_cfg(
    cast(RslRlTeacherKLRunnerCfg, load_rl_cfg(lstm_task)).actor
  )
  transformer_model = _make_actor_from_model_cfg(
    cast(RslRlTeacherKLRunnerCfg, load_rl_cfg(transformer_task)).actor
  )

  lstm_params = sum(p.numel() for p in lstm_model.parameters())
  transformer_params = sum(p.numel() for p in transformer_model.parameters())
  relative_delta = abs(transformer_params - lstm_params) / lstm_params

  assert relative_delta <= 0.10


def test_ablation_actor_history_length_is_configurable() -> None:
  """Ablation env factories should let actor history vary without changing terms."""
  history_length = 7
  plain_cfg = unitree_g1_blind_rough_target_navigation_ablation_env_cfg(
    actor_history_length=history_length
  )
  lstm_cfg = unitree_g1_blind_rough_lstm_teacherkl_env_cfg(
    actor_history_length=history_length
  )
  boolean_cfg = unitree_g1_blind_stairs_flag_teacherkl_env_cfg(
    actor_history_length=history_length
  )

  assert plain_cfg.observations["actor"].history_length == history_length
  assert lstm_cfg.observations["actor"].history_length == history_length
  assert boolean_cfg.observations["actor"].history_length is None
  assert (
    boolean_cfg.observations["actor"].terms["terrain_is_stairs"].history_length == 0
  )
  for name, term in boolean_cfg.observations["actor"].terms.items():
    if name != "terrain_is_stairs":
      assert term.history_length == history_length


def test_train_actor_history_override_preserves_boolean_flag() -> None:
  """Top-level train override should not turn the Boolean flag historical."""
  cfg = load_env_cfg("Mjlab-Velocity-Blind-StairsFlag-TeacherKL-Unitree-G1")

  _apply_actor_history_length_override(cfg, 7)

  actor_group = cfg.observations["actor"]
  assert actor_group.history_length is None
  assert actor_group.terms["terrain_is_stairs"].history_length == 0
  for name, term in actor_group.terms.items():
    if name != "terrain_is_stairs":
      assert term.history_length == 7


def test_transformer_actor_infers_sequence_length_from_history() -> None:
  """Transformer config should follow actor history length without manual sync."""
  transformer_task = "Mjlab-Velocity-Blind-Rough-Transformer-TeacherKL-Unitree-G1"
  transformer_model = _make_actor_from_model_cfg(
    cast(RslRlTeacherKLRunnerCfg, load_rl_cfg(transformer_task)).actor,
    obs_dim=7 * sum((3, 3, 3, 2, 29, 29, 29)),
  )
  obs = TensorDict({"actor": torch.zeros(2, transformer_model.obs_dim)})

  assert transformer_model.sequence_length == 7
  with torch.no_grad():
    out = transformer_model(obs)
  assert out.shape == (2, 29)


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
  rough_training_tasks = [
    "Mjlab-Velocity-Rough-Unitree-G1",
    "Mjlab-Velocity-Rough-Unitree-Go1",
  ]

  for task_id in rough_training_tasks:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} has no terrain_generator"
    )
    assert cfg.scene.terrain.terrain_generator.curriculum is True, (
      f"Task {task_id} curriculum={cfg.scene.terrain.terrain_generator.curriculum}, "
      "expected True"
    )


def test_rough_velocity_play_terrain_curriculum_matches_robot_policy() -> None:
  """G1 play keeps curriculum rows for mixed replay; Go1 keeps static play terrain."""
  expected_curriculum = {
    "Mjlab-Velocity-Rough-Unitree-G1": True,
    "Mjlab-Velocity-Rough-Unitree-Go1": False,
  }

  for task_id, expected in expected_curriculum.items():
    cfg = load_env_cfg(task_id, play=True)

    assert cfg.scene.terrain is not None, (
      f"Task {task_id} (play mode) has no terrain config"
    )
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} (play mode) has no terrain_generator"
    )
    assert cfg.scene.terrain.terrain_generator.curriculum is expected, (
      f"Task {task_id} (play mode) curriculum={cfg.scene.terrain.terrain_generator.curriculum}, "
      f"expected {expected}"
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
