from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import torch

from mjlab.envs.mdp.events import randomize_terrain
from mjlab.tasks.velocity.mdp.curriculums import terrain_levels_vel
from mjlab.tasks.velocity.mdp.rewards import toe_step_riser_slab_penalty


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


def _make_env(root_xy: torch.Tensor, command_term, num_levels: int = 10):
  num_envs = root_xy.shape[0]
  root_pos = torch.cat([root_xy, torch.zeros(num_envs, 1)], dim=1)
  asset = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=root_pos))
  terrain = _FakeTerrain(num_envs, num_levels=num_levels)
  scene = _FakeScene(asset, terrain, terrain.env_origins)
  env = SimpleNamespace(
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
