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
  _normalize_float_list_flag,
  _refresh_goal_respawn_observations,
  _score_policy,
  _summarize_batches,
  _summarize_probe_trace_batches,
  _summarize_step_width_sweep,
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
  assert perfect["collision_penalties"]["toe"] == 0.0

  summary["toe_riser_collision_count"] = 5.0
  penalized = _score_policy(summary, cfg)
  assert penalized["collision_penalties"]["toe"] == pytest.approx(2.1)
  assert penalized["collision_subscores"]["toe"] == pytest.approx(0.825)
  assert penalized["score_100"] == pytest.approx(97.9)
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
  assert summary["score_version"] == "goal_pyramid_full_support_v4"
  assert summary["paper_metrics"]["full_landing_ratio"] == pytest.approx(3.0 / 6.0)
  assert summary["paper_metrics"]["stair_safe_pass_rate"] == pytest.approx(0.5)
  assert summary["paper_metrics"]["toe_riser_collision_penalty"] > 0.0
  assert summary["score_collision_penalties"]["toe"] > 0.0
  assert "collision_heel" not in summary["score_components"]
  assert "collision_lip" not in summary["score_components"]
  assert 0.0 < summary["score_100"] < 100.0


def test_goal_pyramid_step_width_sweep_helpers():
  args = [
    "--checkpoint-file",
    "model.pt",
    "--step-widths",
    "0.27",
    "0.30",
    "0.33",
    "--episodes",
    "3",
  ]
  assert _normalize_float_list_flag(args, "--step-widths") == [
    "--checkpoint-file",
    "model.pt",
    "--step-widths",
    "[0.27, 0.30, 0.33]",
    "--episodes",
    "3",
  ]

  runs = [
    {
      "step_width": 0.27,
      "summary": {
        "score_100": 70.0,
        "success_rate": 1.0,
        "stair_safe_pass_rate": 0.2,
        "landing_index_100": 65.0,
        "mean_stair_landing_support_fraction": 0.7,
        "stair_full_landing_ratio": 0.6,
        "stair_incomplete_landing_ratio": 0.4,
        "toe_riser_collision_count": 3.0,
        "score_collision_penalties": {"toe": 0.7},
      },
    },
    {
      "step_width": 0.33,
      "summary": {
        "score_100": 90.0,
        "success_rate": 1.0,
        "stair_safe_pass_rate": 0.8,
        "landing_index_100": 85.0,
        "mean_stair_landing_support_fraction": 0.9,
        "stair_full_landing_ratio": 0.8,
        "stair_incomplete_landing_ratio": 0.2,
        "toe_riser_collision_count": 1.0,
        "score_collision_penalties": {"toe": 0.0},
      },
    },
  ]
  summary = _summarize_step_width_sweep(runs)
  assert summary["num_widths"] == 2
  assert summary["mean_score_100"] == pytest.approx(80.0)
  assert summary["best_step_width"] == pytest.approx(0.33)
  assert summary["worst_step_width"] == pytest.approx(0.27)


def test_goal_pyramid_probe_trace_summary_reports_probe_growth():
  cfg = GoalPyramidEvalConfig(write_probe_trace=True)
  batches = [
    {
      "probe_trace": {
        "episodes": [
          {
            "first_collision_step": 10,
            "higher_riser_collision_step": 20,
            "post_first_probe_strides": [0.24, 0.54, 0.60],
            "post_first_probe_targets": [0.52, 0.58, 0.64],
            "post_higher_strides": [0.59],
            "post_higher_targets": [0.60],
            "events": [
              {
                "kind": "first_collision",
                "ratchet_target": 0.52,
              },
              {
                "kind": "higher_riser_collision",
                "step": 20,
                "ratchet_target": 0.60,
                "ratchet_valid_second_hit": True,
                "ratchet_recovery_target": 0.55,
                "ratchet_lock_target": 0.60,
              },
              {
                "kind": "same_foot_up_step",
                "step": 22,
                "same_foot_stride_growth": -0.05,
                "ratchet_recovery_completed": True,
                "ratchet_lock_target": 0.60,
              },
            ],
          },
          {
            "first_collision_step": 8,
            "higher_riser_collision_step": None,
            "post_first_probe_strides": [0.50],
            "post_first_probe_targets": [0.55],
            "post_higher_strides": [],
            "post_higher_targets": [],
            "events": [
              {
                "kind": "first_collision",
                "ratchet_target": 0.55,
              }
            ],
          },
        ]
      }
    }
  ]

  trace = _summarize_probe_trace_batches(cfg, batches)
  summary = trace["summary"]

  assert summary["episodes"] == 2
  assert summary["episodes_with_first_collision_rate"] == 1.0
  assert summary["episodes_with_higher_riser_collision_rate"] == 0.5
  assert summary["mean_post_first_probe_raw_stride"] == pytest.approx(0.47)
  assert summary["mean_post_first_probe_entry_scaled_stride"] == pytest.approx(0.74)
  assert summary["mean_post_first_probe_stride"] == pytest.approx(0.57)
  assert summary["mean_post_first_probe_stride_growth"] == pytest.approx(0.06)
  assert summary["mean_post_first_probe_entry_scaled_stride_growth"] == pytest.approx(
    0.12
  )
  assert summary["post_first_probe_monotonic_episode_rate"] == 1.0
  assert summary["post_first_probe_entry_scaled_monotonic_episode_rate"] == 1.0
  assert summary["mean_first_to_higher_collision_steps"] == pytest.approx(10.0)
  assert summary["mean_first_collision_ratchet_target"] == pytest.approx(0.535)
  assert summary["episodes_with_ratchet_valid_second_hit_rate"] == 0.5
  assert summary["episodes_with_recovery_completed_rate"] == 0.5
  assert summary["mean_recovery_same_foot_stride_growth"] == pytest.approx(-0.05)
  assert summary["mean_ratchet_recovery_target"] == pytest.approx(0.55)
  assert summary["mean_ratchet_lock_target"] == pytest.approx(0.60)
  episodes = trace["episodes"]
  assert episodes[0]["post_first_probe_strides_excluding_entry"] == [0.54, 0.60]
  assert episodes[0]["post_first_probe_strides_entry_scaled"] == [0.48, 0.54, 0.60]
