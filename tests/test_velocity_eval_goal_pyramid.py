from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from scripts.velocity_eval.eval_policy_goal_pyramid import (
  GoalPyramidToeRiserContactMarkers,
  _refresh_goal_respawn_observations,
)


class _FakeDetector:
  event_source = "true_contact"

  def __init__(self) -> None:
    self._updates = 0

  def compute_events(self, env):
    del env
    self._updates += 1
    return {}

  def get_last_true_contact_positions(self, kind: str):
    assert kind == "toe"
    if self._updates == 1:
      return torch.tensor([0, 1]), torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    if self._updates == 2:
      return torch.tensor([0]), torch.tensor([[7.0, 8.0, 9.0]])
    return torch.tensor([], dtype=torch.long), torch.empty(0, 3)


class _FakeVisualizer:
  def __init__(self, *, env_idx: int = 0, show_all_envs: bool = False) -> None:
    self.env_idx = env_idx
    self.show_all_envs = show_all_envs
    self.spheres = []

  def get_env_indices(self, num_envs: int):
    if self.show_all_envs:
      return range(num_envs)
    return [self.env_idx]

  def add_sphere(self, *, center, radius, color, label=None) -> None:
    self.spheres.append((np.asarray(center), radius, color, label))


def test_goal_pyramid_toe_riser_contact_markers_persist_until_reset():
  env = cast(Any, SimpleNamespace(num_envs=2))
  detector = cast(Any, _FakeDetector())
  markers = GoalPyramidToeRiserContactMarkers(
    env,
    detector,
    radius=0.04,
    max_points_per_env=2,
  )

  markers.update(env)
  markers.update(env)

  visualizer = _FakeVisualizer(show_all_envs=True)
  markers.debug_vis(visualizer)

  centers = [sphere[0].tolist() for sphere in visualizer.spheres]
  assert centers == [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0], [4.0, 5.0, 6.0]]
  assert all(sphere[1] == 0.04 for sphere in visualizer.spheres)
  assert all(sphere[2] == (1.0, 0.0, 0.0, 1.0) for sphere in visualizer.spheres)

  markers.reset(torch.tensor([0]))
  visualizer = _FakeVisualizer(show_all_envs=True)
  markers.debug_vis(visualizer)

  centers = [sphere[0].tolist() for sphere in visualizer.spheres]
  assert centers == [[4.0, 5.0, 6.0]]


def test_goal_pyramid_respawn_refreshes_observation_history():
  class _FakeObservationManager:
    def __init__(self) -> None:
      self.update_history_args = []

    def compute(self, update_history: bool = False):
      self.update_history_args.append(update_history)
      return {"actor": torch.ones(2, 3)}

  observation_manager = _FakeObservationManager()
  env = cast(Any, SimpleNamespace(observation_manager=observation_manager))

  _refresh_goal_respawn_observations(env)

  assert observation_manager.update_history_args == [True]
  assert torch.equal(env.obs_buf["actor"], torch.ones(2, 3))
