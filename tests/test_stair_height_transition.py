from typing import Any, cast

import mujoco
import numpy as np
import pytest
from scripts.velocity_eval.eval_stair_height_transition import (
  BoxTwoStageStairsTerrainCfg,
  StairHeightTransitionEvalConfig,
  _episode_summary,
  _summarize_transition_records,
  _transition_runway_geometry,
  _validate_transition_config,
)


def _record(
  *,
  episode: int,
  stage: int,
  level: int,
  lift: float,
  pred: float,
  success: bool = True,
  stage_touchdown: int | None = None,
) -> dict:
  stage_touchdown = stage_touchdown if stage_touchdown is not None else level
  height = 0.10 if stage == 1 else 0.18
  return {
    "episode_global_index": episode,
    "stage_index": stage,
    "stage": "first" if stage == 1 else "second",
    "stair_height_m": height,
    "landing_level_in_stage": level,
    "stage_touchdown_index_in_episode": stage_touchdown,
    "is_first_stage_touchdown": stage_touchdown == 1,
    "is_first_stage_level": level == 1,
    "peak_lift_from_takeoff_m": lift,
    "peak_clearance_above_terrain_m": lift - 0.02,
    "pred_riser_height_mean_m": pred,
    "pred_riser_height_takeoff_m": pred,
    "episode_success": success,
  }


def test_transition_geometry_keeps_runway_dimensions_explicit() -> None:
  cfg = StairHeightTransitionEvalConfig(
    first_stair_height=0.10,
    second_stair_height=0.18,
    stair_levels_per_stage=10,
    step_width=0.30,
    stair_width=3.0,
    side_margin_width=0.75,
    flat_apron_width=1.20,
    middle_platform_width=0.60,
    final_platform_width=1.20,
  )

  geom = _transition_runway_geometry(cfg)

  assert geom.tile_length == pytest.approx(9.0)
  assert geom.tile_width == pytest.approx(4.5)
  assert geom.spawn_x == pytest.approx(0.60)
  assert geom.first_stage_start_x == pytest.approx(1.20)
  assert geom.first_stage_end_x == pytest.approx(4.20)
  assert geom.second_stage_start_x == pytest.approx(4.80)
  assert geom.second_stage_end_x == pytest.approx(7.80)
  assert geom.goal_x == pytest.approx(8.40)
  assert geom.first_top_height == pytest.approx(1.0)
  assert geom.final_top_height == pytest.approx(2.8)


def test_two_stage_stairs_emit_sequence_metadata_and_target_patch() -> None:
  cfg = BoxTwoStageStairsTerrainCfg(
    size=(3.0, 4.5),
    first_stair_height=0.10,
    second_stair_height=0.18,
    stair_levels_per_stage=2,
    step_width=0.30,
    stair_width=3.0,
    side_margin_width=0.75,
    flat_apron_width=1.20,
    middle_platform_width=0.60,
    final_platform_width=1.20,
  )
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")

  output = cfg.function(0.0, spec, np.random.default_rng(0))

  assert output.step_boundaries is not None
  assert output.step_boundary_sequence_ids is not None
  assert output.step_boundary_layers is not None
  assert output.step_boundaries.shape == (4, 11)
  np.testing.assert_array_equal(output.step_boundary_sequence_ids, [1, 1, 2, 2])
  np.testing.assert_array_equal(output.step_boundary_layers, [1, 2, 1, 2])
  np.testing.assert_allclose(output.step_boundaries[0, [0, 9, 10]], [1.2, 0.0, 0.1])
  np.testing.assert_allclose(output.step_boundaries[2, [0, 9, 10]], [2.4, 0.2, 0.38])
  np.testing.assert_allclose(
    output.step_boundaries[:, 6:9],
    np.tile([-1.0, 0.0, 0.0], (4, 1)),
  )
  assert output.flat_patches is not None
  np.testing.assert_allclose(output.flat_patches["target"], [[3.6, 2.25, 0.56]])


def test_transition_validation_defaults_to_training_stair_range() -> None:
  _validate_transition_config(StairHeightTransitionEvalConfig())

  with pytest.raises(ValueError, match="first-stair-height"):
    _validate_transition_config(
      StairHeightTransitionEvalConfig(first_stair_height=0.04)
    )

  _validate_transition_config(
    StairHeightTransitionEvalConfig(
      first_stair_height=0.04,
      allow_out_of_train_range=True,
    )
  )


def test_transition_summary_reports_stage_delta_and_paired_episode_slope() -> None:
  cfg = StairHeightTransitionEvalConfig(
    first_stair_height=0.10,
    second_stair_height=0.18,
    summarize_success_only=True,
  )
  episode_summary = {"episodes": 2, "success_rate": 0.5}
  records = [
    _record(episode=0, stage=1, level=1, lift=0.15, pred=0.10),
    _record(episode=0, stage=1, level=2, lift=0.18, pred=0.10),
    _record(episode=0, stage=2, level=1, lift=0.22, pred=0.18),
    _record(episode=0, stage=2, level=2, lift=0.26, pred=0.18),
    _record(episode=1, stage=1, level=2, lift=0.50, pred=0.10, success=False),
    _record(episode=1, stage=2, level=2, lift=0.60, pred=0.18, success=False),
  ]

  summary = _summarize_transition_records(cfg, episode_summary, records)

  assert summary["records_total"] == 6
  assert summary["records_used"] == 4
  assert summary["by_stage"][0]["later_levels"]["peak_lift_from_takeoff_m"][
    "mean"
  ] == pytest.approx(0.18)
  assert summary["by_stage"][1]["later_levels"]["peak_lift_from_takeoff_m"][
    "mean"
  ] == pytest.approx(0.26)
  later_delta = next(
    item
    for item in summary["stage_deltas"]
    if item["phase"] == "later_levels" and item["metric"] == "peak_lift_from_takeoff_m"
  )
  assert later_delta["delta_second_minus_first"] == pytest.approx(0.08)
  assert later_delta["delta_per_height_change"] == pytest.approx(1.0)
  paired = next(
    item
    for item in summary["paired_episode_deltas"]
    if item["phase"] == "later_levels" and item["metric"] == "peak_lift_from_takeoff_m"
  )
  assert paired["paired_episode_count"] == 1
  assert paired["delta_per_height_change"]["mean"] == pytest.approx(1.0)
  pred_fit = summary["linear_fits"]["later_levels_pred_riser_vs_stage_height"]
  assert pred_fit["slope"] == pytest.approx(1.0)


def test_transition_episode_summary_counts_first_stage_and_final_success() -> None:
  class _Batch:
    success = [True, False]
    first_stage_reached = [True, True]
    fell = [False, True]
    heading_failed = [False, False]
    timeout_failed = [False, False]
    episode_length_steps = [100.0, 200.0]
    max_height_progress_fraction = [1.0, 0.5]
    max_goal_progress_fraction = [1.0, 0.4]
    max_stage_progress_fraction = [1.0, 0.55]

  batches = cast(Any, [_Batch()])
  summary = _episode_summary(StairHeightTransitionEvalConfig(), batches, 0.02)

  assert summary["success_rate"] == pytest.approx(0.5)
  assert summary["first_stage_reached_rate"] == pytest.approx(1.0)
  assert summary["fall_rate"] == pytest.approx(0.5)
  assert summary["mean_episode_length_s"] == pytest.approx(3.0)
