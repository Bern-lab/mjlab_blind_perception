from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from mjlab.envs.mdp.events import randomize_terrain
from mjlab.tasks.velocity.mdp.curriculums import terrain_levels_vel
from mjlab.tasks.velocity.mdp.rewards import (
  toe_step_riser_probe_shaping_reward,
  toe_step_riser_slab_penalty,
)
from mjlab.tasks.velocity.mdp.stair_geometry import stair_shape_from_boundaries
from mjlab.tasks.velocity.mdp.temporal_stair_rewards import (
  toe_step_riser_slab_penalty as temporal_toe_step_riser_slab_penalty,
)


class _FakeScene:
  def __init__(self, asset, terrain, env_origins: torch.Tensor) -> None:
    self._asset = asset
    self.terrain = terrain
    self.env_origins = env_origins

  def __getitem__(self, name: str):
    assert name == "robot"
    return self._asset


class _FakeTerrain:
  def __init__(self, num_envs: int, num_levels: int = 10) -> None:
    self.cfg = SimpleNamespace(
      terrain_generator=SimpleNamespace(
        size=(8.0, 8.0),
        sub_terrains={"stairs": SimpleNamespace()},
      )
    )
    self.terrain_levels = torch.zeros(num_envs, dtype=torch.long)
    self.terrain_types = torch.zeros(num_envs, dtype=torch.long)
    self.terrain_origins = torch.zeros(num_levels, 1, 3)
    self.env_origins = torch.zeros(num_envs, 3)
    self.last_move_up: torch.Tensor | None = None
    self.last_move_down: torch.Tensor | None = None

  def update_env_origins(
    self,
    env_ids: torch.Tensor,
    move_up: torch.Tensor,
    move_down: torch.Tensor,
  ) -> None:
    self.last_move_up = move_up.clone()
    self.last_move_down = move_down.clone()
    self.terrain_levels[env_ids] += move_up.long()
    self.terrain_levels[env_ids] -= move_down.long()
    self.terrain_levels.clamp_(min=0)
    self.env_origins[env_ids] = self.terrain_origins[
      self.terrain_levels[env_ids], self.terrain_types[env_ids]
    ]


class _FakeCommandManager:
  def __init__(self, term) -> None:
    self._term = term

  def get_term(self, name: str):
    assert name == "twist"
    return self._term

  def get_command(self, name: str):
    assert name == "twist"
    return self._term.command


def _make_env(root_xy: torch.Tensor, command_term, num_levels: int = 10):
  num_envs = root_xy.shape[0]
  root_pos = torch.cat([root_xy, torch.zeros(num_envs, 1)], dim=1)
  root_quat = torch.zeros(num_envs, 4)
  root_quat[:, 0] = 1.0
  asset = SimpleNamespace(
    data=SimpleNamespace(root_link_pos_w=root_pos, root_link_quat_w=root_quat)
  )
  terrain = _FakeTerrain(num_envs, num_levels=num_levels)
  scene = _FakeScene(asset, terrain, terrain.env_origins)
  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    scene=scene,
    command_manager=_FakeCommandManager(command_term),
    max_episode_length_s=20.0,
    extras={},
  )
  return env, terrain


def test_terrain_levels_vel_uses_target_reached_for_target_episodes() -> None:
  command_term = SimpleNamespace(
    command=torch.zeros(3, 3),
    target_command_in_episode=torch.tensor([True, True, False]),
    target_reached_in_episode=torch.tensor([True, False, False]),
  )
  env, terrain = _make_env(
    torch.tensor(
      [
        [1.0, 0.0],  # Target reached should move up even below distance threshold.
        [5.0, 0.0],  # Target not reached should not move up despite distance.
        [5.0, 0.0],  # Non-target episode falls back to distance curriculum.
      ]
    ),
    command_term,
  )

  result = terrain_levels_vel(env, torch.arange(3), command_name="twist")

  assert terrain.last_move_up is not None
  assert terrain.last_move_up.tolist() == [True, False, True]
  assert result["target_attempted"].item() == torch.tensor(2 / 3).item()
  assert result["target_reached"].item() == torch.tensor(1 / 3).item()


def test_terrain_levels_vel_keeps_distance_rule_without_target_state() -> None:
  command_term = SimpleNamespace(command=torch.zeros(2, 3))
  env, terrain = _make_env(
    torch.tensor(
      [
        [5.0, 0.0],
        [1.0, 0.0],
      ]
    ),
    command_term,
  )

  result = terrain_levels_vel(env, torch.arange(2), command_name="twist")

  assert terrain.last_move_up is not None
  assert terrain.last_move_up.tolist() == [True, False]
  assert "target_attempted" not in result
  assert "target_reached" not in result


def test_terrain_levels_vel_mixed_replay_sticks_after_high_level() -> None:
  command_term = SimpleNamespace(command=torch.zeros(2, 3))
  env, terrain = _make_env(
    torch.tensor(
      [
        [5.0, 0.0],
        [1.0, 0.0],
      ]
    ),
    command_term,
    num_levels=10,
  )
  terrain.terrain_levels[0] = 7

  result = terrain_levels_vel(
    env,
    torch.arange(2),
    command_name="twist",
    mixed_replay_start_level=8,
    mixed_replay_level_ranges=((0, 0), (4, 4), (9, 9)),
    mixed_replay_weights=(1.0, 0.0, 0.0),
  )

  assert terrain.terrain_levels.tolist() == [0, 0]
  assert result["mixed_replay_active"].item() == torch.tensor(0.5).item()
  assert result["mixed_replay_low_ratio"].item() == torch.tensor(1.0).item()

  terrain_levels_vel(
    env,
    torch.tensor([0]),
    command_name="twist",
    mixed_replay_start_level=8,
    mixed_replay_level_ranges=((0, 0), (4, 4), (9, 9)),
    mixed_replay_weights=(0.0, 0.0, 1.0),
  )

  assert terrain.terrain_levels[0].item() == 9


def test_randomize_terrain_can_sample_weighted_level_buckets() -> None:
  num_envs = 8
  terrain = SimpleNamespace(
    cfg=SimpleNamespace(
      terrain_generator=SimpleNamespace(
        sub_terrains={
          "flat": SimpleNamespace(proportion=0.0),
          "stairs": SimpleNamespace(proportion=1.0),
          "rough": SimpleNamespace(proportion=0.0),
        }
      )
    ),
    terrain_origins=torch.zeros(10, 3, 3),
    env_origins=torch.zeros(num_envs, 3),
    terrain_levels=torch.zeros(num_envs, dtype=torch.long),
    terrain_types=torch.zeros(num_envs, dtype=torch.long),
  )
  scene = SimpleNamespace(terrain=terrain, env_origins=terrain.env_origins)
  env = SimpleNamespace(num_envs=num_envs, device=torch.device("cpu"), scene=scene)

  randomize_terrain(
    cast(Any, env),
    torch.arange(num_envs),
    level_ranges=((0, 0), (4, 4), (9, 9)),
    level_weights=(0.0, 1.0, 0.0),
    use_sub_terrain_proportions=True,
  )

  assert terrain.terrain_levels.tolist() == [4] * num_envs
  assert terrain.terrain_types.tolist() == [1] * num_envs


def test_toe_probe_boundary_layers_use_actual_stair_layers() -> None:
  boundaries = torch.zeros(2, 4, 11)
  boundaries[:, :, 0] = torch.arange(4, dtype=torch.float32)
  boundaries[:, :, 3] = boundaries[:, :, 0]
  boundaries[:, :, 4] = 1.0

  boundaries[0, :, 9] = torch.tensor([0.0, 0.1, 0.2, 0.3])
  boundaries[0, :, 10] = torch.tensor([0.1, 0.2, 0.3, 0.4])
  boundaries[1, :, 9] = torch.tensor([-0.1, -0.2, -0.3, -0.4])
  boundaries[1, :, 10] = torch.tensor([0.0, -0.1, -0.2, -0.3])
  valid = torch.ones(2, 4, dtype=torch.bool)

  layers = toe_step_riser_slab_penalty._probe_boundary_layers(
    boundaries,
    valid,
    max_probe_layers=2,
  )

  assert layers.tolist() == [[1, 2, 0, 0], [1, 2, 0, 0]]


def test_toe_probe_forward_layers_use_geometry_not_height_sign() -> None:
  boundaries = torch.zeros(2, 3, 11)
  boundaries[:, :, 0] = torch.tensor([1.0, 2.0, 3.0])
  boundaries[:, :, 1] = -0.5
  boundaries[:, :, 3] = boundaries[:, :, 0]
  boundaries[:, :, 4] = 0.5
  boundaries[:, :, 6] = -1.0
  boundaries[0, :, 9] = torch.tensor([0.0, 0.1, 0.2])
  boundaries[0, :, 10] = torch.tensor([0.1, 0.2, 0.3])
  boundaries[1, :, 9] = torch.tensor([-0.1, -0.2, -0.3])
  boundaries[1, :, 10] = torch.tensor([0.0, -0.1, -0.2])
  valid = torch.ones(2, 3, dtype=torch.bool)

  command_term = SimpleNamespace(command=torch.tensor([[1.0, 0.0, 0.0]] * 2))
  env, _ = _make_env(torch.zeros(2, 2), command_term)
  asset = env.scene["robot"]

  layers, command_active, root_s = (
    toe_step_riser_probe_shaping_reward._forward_boundary_layers(
      cast(Any, env),
      asset,
      boundaries,
      valid,
      command_name="twist",
      forward_velocity_threshold=0.05,
      forward_tol=0.05,
      low_side_margin=0.02,
      merge_riser_eps=0.04,
    )
  )

  assert command_active.tolist() == [True, True]
  assert torch.all(root_s > 0.0)
  assert layers.tolist() == [[1, 2, 0], [1, 2, 0]]


def test_toe_probe_reach_uses_current_position_as_progress_baseline() -> None:
  root_xy = torch.tensor([[0.0, 0.0]])
  toe_xy = torch.tensor([[0.3, 0.0]])
  p0_xy = torch.tensor([[1.0, -0.5]])
  p1_xy = torch.tensor([[1.0, 0.5]])
  normal_xy = torch.tensor([[-1.0, 0.0]])
  probe_side = torch.tensor([1.0])

  current_reach = toe_step_riser_probe_shaping_reward._boundary_reach(
    root_xy,
    toe_xy,
    p0_xy,
    p1_xy,
    normal_xy,
    probe_side,
  )
  later_reach = toe_step_riser_probe_shaping_reward._boundary_reach(
    root_xy,
    torch.tensor([[0.4, 0.0]]),
    p0_xy,
    p1_xy,
    normal_xy,
    probe_side,
  )

  assert current_reach.item() == torch.tensor(0.3).item()
  assert torch.relu(current_reach - current_reach).item() == 0.0
  assert torch.relu(later_reach - current_reach).item() == pytest.approx(0.1)


def test_toe_probe_root_segment_gate_filters_lateral_misalignment() -> None:
  p0_xy = torch.tensor([[1.0, -0.5], [1.0, -0.5]])
  p1_xy = torch.tensor([[1.0, 0.5], [1.0, 0.5]])
  points_xy = torch.tensor([[0.0, 0.0], [0.0, 0.8]])

  gate, _u, _segment_len, _tangent = (
    toe_step_riser_probe_shaping_reward._segment_range_gate(
      points_xy,
      p0_xy,
      p1_xy,
      margin=0.2,
    )
  )

  assert gate.tolist() == [True, False]


def test_toe_probe_side_eps_rejects_boundary_ambiguous_root_side() -> None:
  side, valid = toe_step_riser_probe_shaping_reward._side_from_signed_distance(
    torch.tensor([0.02, 0.04, -0.04]),
    side_eps=0.03,
  )

  assert side.tolist() == [1.0, 1.0, -1.0]
  assert valid.tolist() == [False, True, True]


def test_toe_probe_force_sign_flips_blocking_force_convention() -> None:
  force_xy = torch.tensor([[-10.0, 0.0]])
  normal_xy = torch.tensor([[-1.0, 0.0]])
  side = torch.tensor([1.0])

  positive = toe_step_riser_probe_shaping_reward._blocking_force_along_normal(
    force_xy,
    normal_xy,
    side,
    force_sign=1.0,
  )
  negative = toe_step_riser_probe_shaping_reward._blocking_force_along_normal(
    force_xy,
    normal_xy,
    side,
    force_sign=-1.0,
  )

  assert positive.item() == torch.tensor(10.0).item()
  assert negative.item() == torch.tensor(-10.0).item()


def test_temporal_probe_progress_uses_world_ascent_direction() -> None:
  start = torch.tensor([[0.2, 0.1, 0.0], [0.2, 0.1, 0.0]])
  current = torch.tensor([[0.5, 0.3, 0.0], [0.5, 0.3, 0.0]])
  ascent_dir = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

  progress = temporal_toe_step_riser_slab_penalty._ascent_progress(
    current,
    start,
    ascent_dir,
  )

  assert progress.tolist() == pytest.approx([0.3, 0.2])


def test_temporal_probe_velocity_guard_blocks_fast_riser_approach() -> None:
  velocity = torch.tensor([[0.4, 0.0, 0.0], [0.6, 0.0, 0.0]])
  ascent_dir = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

  forward_vel, safe, overspeed = temporal_toe_step_riser_slab_penalty._velocity_guard(
    velocity,
    ascent_dir,
    max_forward_vel=0.45,
  )

  assert forward_vel.tolist() == pytest.approx([0.4, 0.6])
  assert safe.tolist() == [True, False]
  assert overspeed.tolist() == pytest.approx([0.0, 0.15])


def test_stair_shape_uses_parallel_boundary_spacing_and_riser_height() -> None:
  boundaries = torch.tensor(
    [
      [
        [0.0, 0.0, 0.10, 1.0, 0.0, 0.10, 0.0, -1.0, 0.0, 0.0, 0.10],
        [0.0, 0.30, 0.20, 1.0, 0.30, 0.20, 0.0, -1.0, 0.0, 0.10, 0.20],
      ]
    ]
  )
  valid = torch.tensor([[True, True]])

  tread_depth, riser_height, shape_valid = stair_shape_from_boundaries(
    boundaries, valid
  )

  assert tread_depth.tolist() == pytest.approx([0.30])
  assert riser_height.tolist() == pytest.approx([0.10])
  assert shape_valid.tolist() == [True]
