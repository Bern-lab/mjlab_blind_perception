from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from scripts.velocity_eval.eval_metrics import (
  EVENT_COUNT_NAMES,
  LANDING_COUNT_NAMES,
  LANDING_SUM_NAMES,
  LEVEL_EVENT_NAMES,
  MEAN_METRIC_NAMES,
)
from scripts.velocity_eval.eval_policy_goal_pyramid import (
  GoalPyramidEvalConfig,
  GoalPyramidToeRiserContactMarkers,
  _refresh_goal_respawn_observations,
  _score_policy,
  _summarize_batches,
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


def test_goal_pyramid_policy_score_allows_two_toe_hits_without_penalty():
  cfg = GoalPyramidEvalConfig()
  summary = {
    "success_rate": 1.0,
    "mean_max_height_progress_fraction": 1.0,
    "mean_stair_landing_score": 1.0,
    "mean_stair_landing_support_fraction": 1.0,
    "stair_full_landing_ratio": 1.0,
    "stair_incomplete_landing_ratio": 0.0,
    "stair_low_support_landing_ratio": 0.0,
    "toe_riser_collision_count": 2.0,
    "heel_riser_collision_count": 0.0,
    "foot_lip_collision_count": 0.0,
    "fall_rate": 0.0,
    "heading_failure_rate": 0.0,
    "timeout_failure_rate": 0.0,
  }

  perfect = _score_policy(summary, cfg)
  assert perfect["score_100"] == 100.0
  assert perfect["collision_subscores"]["toe"] == 1.0

  summary["toe_riser_collision_count"] = 5.0
  penalized = _score_policy(summary, cfg)
  assert penalized["collision_subscores"]["toe"] == 0.625
  assert penalized["score_100"] == 98.125
  assert "heel" not in penalized["collision_subscores"]
  assert "lip" not in penalized["collision_subscores"]


def test_goal_pyramid_summary_reports_landing_quality_and_score():
  cfg = GoalPyramidEvalConfig(stair_levels=2)
  batch: dict[str, Any] = {
    "event_source": "true_contact",
    "success": [True, False],
    "fell": [False, True],
    "heading_failed": [False, False],
    "timeout_failed": [False, False],
    "spawn_side": ["left", "right"],
    "episode_length_steps": [10.0, 20.0],
    "max_heading_error_deg": [5.0, 10.0],
    "max_height_progress_fraction": [1.0, 0.5],
    "max_goal_progress_fraction": [1.0, 0.25],
    "mean_metrics": {name: [0.0, 0.0] for name in MEAN_METRIC_NAMES},
    "event_counts": {name: [0.0, 0.0] for name in EVENT_COUNT_NAMES},
    "landing_counts": {name: [0.0, 0.0] for name in LANDING_COUNT_NAMES},
    "landing_sums": {name: [0.0, 0.0] for name in LANDING_SUM_NAMES},
    "level_counts": {name: [[0.0, 0.0], [0.0, 0.0]] for name in LEVEL_EVENT_NAMES},
    "step_dt": 0.02,
  }
  batch["event_counts"]["toe_riser_collision"] = [2.0, 4.0]
  batch["landing_counts"]["stair_landing_count"] = [4.0, 2.0]
  batch["landing_counts"]["stair_full_landing_count"] = [3.0, 0.0]
  batch["landing_counts"]["stair_incomplete_landing_count"] = [1.0, 2.0]
  batch["landing_sums"]["stair_landing_score_sum"] = [3.2, 0.8]
  batch["landing_sums"]["stair_landing_support_sum"] = [3.4, 0.9]

  summary = _summarize_batches(cfg, [batch])

  assert summary["toe_riser_collision_count"] == 3.0
  assert summary["toe_riser_collision_count_success_only"] == 2.0
  assert summary["mean_stair_landing_score"] == pytest.approx(4.0 / 6.0)
  assert summary["stair_full_landing_ratio"] == pytest.approx(3.0 / 6.0)
  assert summary["stair_incomplete_landing_ratio"] == pytest.approx(3.0 / 6.0)
  assert summary["toe_riser_collision_over_free_count"] == pytest.approx(1.0)
  assert summary["stair_landing_pass_rate"] == pytest.approx(0.5)
  assert summary["stair_safe_pass_rate"] == pytest.approx(0.5)
  assert summary["episode_stair_full_landing_ratio_p10"] == pytest.approx(0.075)
  assert "landing_index_100" in summary
  assert "landing_linear_score_100" in summary
  assert summary["score_version"] == "goal_pyramid_full_support_v3"
  assert summary["paper_metrics"]["full_landing_ratio"] == pytest.approx(3.0 / 6.0)
  assert summary["paper_metrics"]["stair_safe_pass_rate"] == pytest.approx(0.5)
  assert "collision_heel" not in summary["score_components"]
  assert "collision_lip" not in summary["score_components"]
  assert 0.0 < summary["score_100"] < 100.0
