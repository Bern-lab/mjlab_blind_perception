from __future__ import annotations

from types import SimpleNamespace

import torch

from mjlab.tasks.velocity.mdp.curriculums import terrain_levels_vel


class _FakeScene:
  def __init__(self, asset, terrain, env_origins: torch.Tensor) -> None:
    self._asset = asset
    self.terrain = terrain
    self.env_origins = env_origins

  def __getitem__(self, name: str):
    assert name == "robot"
    return self._asset


class _FakeTerrain:
  def __init__(self, num_envs: int) -> None:
    self.cfg = SimpleNamespace(
      terrain_generator=SimpleNamespace(
        size=(8.0, 8.0),
        sub_terrains={"stairs": SimpleNamespace()},
      )
    )
    self.terrain_levels = torch.zeros(num_envs, dtype=torch.long)
    self.terrain_types = torch.zeros(num_envs, dtype=torch.long)
    self.terrain_origins = torch.zeros(1, 1, 3)
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


class _FakeCommandManager:
  def __init__(self, term) -> None:
    self._term = term

  def get_term(self, name: str):
    assert name == "twist"
    return self._term


def _make_env(root_xy: torch.Tensor, command_term):
  num_envs = root_xy.shape[0]
  root_pos = torch.cat([root_xy, torch.zeros(num_envs, 1)], dim=1)
  asset = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=root_pos))
  terrain = _FakeTerrain(num_envs)
  scene = _FakeScene(asset, terrain, torch.zeros(num_envs, 3))
  env = SimpleNamespace(
    scene=scene,
    command_manager=_FakeCommandManager(command_term),
    max_episode_length_s=20.0,
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
