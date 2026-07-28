from pathlib import Path

import pytest
import torch
from scripts.velocity_eval.eval_stair_lift_height_sweep import (
  StairLiftHeightSweepConfig,
  _actor_obs_dim_from_checkpoint_payload,
  _diag_vector,
  _infer_registered_task_from_checkpoint_path,
  _linear_fit,
  _normalize_float_list_flag,
  _summarize_records,
  _task_selector_from_argv,
)


def _record(
  *,
  height: float,
  lift: float,
  clearance: float,
  pred_riser: float,
  success: bool = True,
  touchdown_index: int = 1,
  level: int = 1,
) -> dict:
  return {
    "stair_height_m": height,
    "peak_lift_from_takeoff_m": lift,
    "peak_clearance_above_terrain_m": clearance,
    "pred_riser_height_mean_m": pred_riser,
    "episode_success": success,
    "is_first_stair_touchdown": touchdown_index == 1,
    "is_first_stair_level": level == 1,
    "touchdown_index_in_episode": touchdown_index,
    "landing_level_low_to_high": level,
  }


def test_diag_vector_handles_single_step_batch_component() -> None:
  diagnostics = {"stair_shape": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])}

  value = _diag_vector(diagnostics, "stair_shape", num_envs=2, component=1)

  assert value is not None
  assert torch.allclose(value, torch.tensor([0.2, 0.4]))


def test_normalize_float_list_flag_accepts_repeated_values() -> None:
  args = ["--stair-heights", "0.09", "0.11", "--episodes-per-height", "2"]

  normalized = _normalize_float_list_flag(args, "--stair-heights")

  assert normalized == [
    "--stair-heights",
    "[0.09, 0.11]",
    "--episodes-per-height",
    "2",
  ]


def test_task_selector_defaults_to_auto_without_positional_task() -> None:
  task, args = _task_selector_from_argv(
    ["--checkpoint-file", "logs/rsl_rl/g1_velocity/run/model.pt"]
  )

  assert task == "auto"
  assert args == ["--checkpoint-file", "logs/rsl_rl/g1_velocity/run/model.pt"]


def test_task_selector_accepts_unregistered_policy_alias() -> None:
  task, args = _task_selector_from_argv(
    ["g1_velocity", "--checkpoint-file", "model.pt"]
  )

  assert task == "g1_velocity"
  assert args == ["--checkpoint-file", "model.pt"]


def test_infer_registered_task_from_checkpoint_path_uses_longest_match() -> None:
  tasks = [
    "Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1",
    "Mjlab-Velocity-Blind-Rough-TargetNavigation-TeacherKL-Unitree-G1",
  ]
  path = (
    "logs/rsl_rl/x/"
    "Mjlab-Velocity-Blind-Rough-TargetNavigation-TeacherKL-Unitree-G1/"
    "run/model.pt"
  )

  task = _infer_registered_task_from_checkpoint_path(Path(path), tasks)

  assert task == "Mjlab-Velocity-Blind-Rough-TargetNavigation-TeacherKL-Unitree-G1"


def test_actor_obs_dim_prefers_normalizer_over_first_linear_width() -> None:
  payload = {
    "actor_state_dict": {
      "obs_normalizer._mean": torch.zeros(1, 490),
      "mlp.0.weight": torch.zeros(512, 530),
    }
  }

  assert _actor_obs_dim_from_checkpoint_payload(payload) == 490


def test_linear_fit_reports_height_tracking_slope() -> None:
  records = [
    _record(height=0.10, lift=0.15, clearance=0.12, pred_riser=0.10),
    _record(height=0.20, lift=0.25, clearance=0.13, pred_riser=0.20),
    _record(height=0.24, lift=0.29, clearance=0.14, pred_riser=0.24),
  ]

  fit = _linear_fit(records, "peak_lift_from_takeoff_m")

  assert fit["count"] == 3
  assert fit["slope"] == pytest.approx(1.0)
  assert fit["intercept"] == pytest.approx(0.05)
  assert fit["r2"] == pytest.approx(1.0)


def test_summarize_records_filters_failed_episodes_and_groups_phases() -> None:
  cfg = StairLiftHeightSweepConfig(
    stair_heights=[0.10, 0.20],
    summarize_success_only=True,
  )
  height_results = [
    {
      "stair_height_m": 0.10,
      "episodes": 2,
      "success_rate": 1.0,
      "mean_max_height_progress_fraction": 1.0,
    },
    {
      "stair_height_m": 0.20,
      "episodes": 2,
      "success_rate": 0.5,
      "mean_max_height_progress_fraction": 0.8,
    },
  ]
  records = [
    _record(height=0.10, lift=0.15, clearance=0.11, pred_riser=0.10),
    _record(
      height=0.10,
      lift=0.17,
      clearance=0.12,
      pred_riser=0.10,
      touchdown_index=2,
      level=2,
    ),
    _record(height=0.20, lift=0.25, clearance=0.12, pred_riser=0.20),
    _record(height=0.20, lift=0.50, clearance=0.30, pred_riser=0.20, success=False),
  ]

  summary = _summarize_records(cfg, height_results, records)

  assert summary["records_total"] == 4
  assert summary["records_used"] == 3
  assert summary["by_height"][0]["later_levels"]["peak_lift_from_takeoff_m"][
    "mean"
  ] == pytest.approx(0.17)
  first_fit = summary["linear_fits"]["first_level_peak_lift_vs_height"]
  assert first_fit["slope"] == pytest.approx(1.0)
  assert first_fit["intercept"] == pytest.approx(0.05)
  pred_fit = summary["linear_fits"]["all_pred_riser_vs_height"]
  assert pred_fit["slope"] == pytest.approx(1.0)
