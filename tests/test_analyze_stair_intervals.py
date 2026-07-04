from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
from tools.analyze_stair_intervals import (
  AuditError,
  EventRecord,
  Evidence,
  IntervalEvidence,
  Prediction,
  QualityScheme,
  SequenceKey,
  SequenceRecord,
  _calibrate,
  _calibrate_quality_buckets,
  _extract_evidence,
  _fuse_sequence,
  _load_inputs,
  _macro_f1,
  _multilayer_prediction_rows,
  _multilayer_prefix_stats,
  _prediction_stats,
  _reclassify_confirmation_events,
  _regression_stats,
  _shuffled_evidence,
  _shuffled_quality_overrides,
  _support_quality,
  _support_visits,
  analyze,
  analyze_descriptive,
)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
  fieldnames: list[str] = []
  for row in rows:
    for key in row:
      if key not in fieldnames:
        fieldnames.append(key)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def _sequence_row(
  sequence_id: str,
  depth: float,
  depth_bin: int,
  *,
  process_end: bool = False,
  oracle: bool = False,
) -> dict[str, object]:
  return {
    "sequence_id": sequence_id,
    "env_id": 0,
    "depth_bin": depth_bin,
    "true_tread_depth": depth,
    "true_riser_height": 0.1,
    "termination_reason": "process_end" if process_end else "confirmed_exit",
    "has_layer2plus_riser_collision": int(oracle),
  }


def _event_row(
  sequence_id: str,
  event_id: int,
  event_type: str,
  frame_idx: int,
  *,
  layer: int,
  root_x: float,
  pair_depth: float = 0.0,
  pair_height: float = 0.0,
  left_support_ratio: float = 0.0,
  right_support_ratio: float = 0.0,
) -> dict[str, object]:
  return {
    "sequence_id": sequence_id,
    "event_id": event_id,
    "event_type": event_type,
    "frame_idx": frame_idx,
    "time": frame_idx * 0.02,
    "event_layer": layer,
    "root_x": root_x,
    "root_y": 0.0,
    "ascent_dir_x": 1.0,
    "ascent_dir_y": 0.0,
    "pair_depth": pair_depth,
    "pair_height": pair_height,
    "left_support_ratio": left_support_ratio,
    "right_support_ratio": right_support_ratio,
  }


def test_v2_trajectory_and_contact_evidence_are_extracted(tmp_path: Path) -> None:
  run = tmp_path / "model_seed42"
  _write_rows(
    run / "stair_sequences.csv",
    [_sequence_row("1", 0.30, 3, oracle=True)],
  )
  events = [
    _event_row("1", 0, "first_layer1_collision", 0, layer=1, root_x=0.0),
    {
      **_event_row(
        "1",
        1,
        "pair_enter",
        10,
        layer=2,
        root_x=0.2,
        pair_depth=0.25,
      ),
      "pair_quality_min": 0.55,
      "pair_quality_mean": 0.65,
      "pair_quality_max": 0.75,
      "pair_quality_asymmetry": 0.20,
    },
    {
      **_event_row(
        "1",
        2,
        "pair_stable_for_N_frames",
        12,
        layer=2,
        root_x=0.2,
        pair_depth=0.28,
      ),
      "pair_quality_min": 0.90,
      "pair_quality_mean": 0.92,
      "pair_quality_max": 0.94,
      "pair_quality_asymmetry": 0.04,
    },
    {
      **_event_row(
        "1",
        3,
        "pair_peak_quality",
        14,
        layer=2,
        root_x=0.2,
      ),
      "peak_pair_depth": 0.29,
      "peak_pair_height": 0.10,
      "pair_quality_min": 0.95,
      "pair_quality_mean": 0.96,
      "pair_quality_max": 0.97,
      "pair_quality_asymmetry": 0.02,
    },
    {
      **_event_row("1", 4, "oracle_riser_contact", 20, layer=2, root_x=0.50),
      "foot_id": 0,
      "contact_valid": 1,
      "contact_point_s": 0.60,
      "toe_s": 0.61,
      "root_s": 0.50,
      "contact_s_minus_riser_s": 0.001,
    },
    {
      **_event_row("1", 5, "oracle_riser_contact", 30, layer=3, root_x=0.80),
      "foot_id": 0,
      "contact_valid": 1,
      "contact_point_s": 0.90,
      "toe_s": 0.92,
      "root_s": 0.79,
      "contact_s_minus_riser_s": -0.002,
    },
    {
      **_event_row("1", 6, "oracle_riser_contact", 40, layer=4, root_x=1.10),
      "foot_id": 0,
      "contact_valid": 1,
      "contact_point_s": 3.20,
      "toe_s": 1.20,
      "root_s": 1.10,
      "contact_s_minus_riser_s": 2.0,
    },
  ]
  _write_rows(run / "stair_events.csv", events)

  sequences, loaded_events, _excluded = _load_inputs([run], False)
  evidence = _extract_evidence(sequences, loaded_events, set())
  by_type = {item.evidence_type: item for item in evidence}

  assert by_type["adjacent_support_first_v2"].center == pytest.approx(0.25)
  assert by_type["adjacent_support_stable_v2"].center == pytest.approx(0.28)
  assert by_type["adjacent_support_peak_v2"].center == pytest.approx(0.29)
  assert by_type["adjacent_support_peak_v2"].quality_min == pytest.approx(0.95)
  assert by_type["oracle_contact_point_proxy_v2"].center == pytest.approx(0.30)
  assert by_type["oracle_toe_proxy_v2"].center == pytest.approx(0.31)
  assert by_type["oracle_root_proxy_v2"].center == pytest.approx(0.29)
  assert (
    sum(item.evidence_type == "oracle_contact_point_proxy_v2" for item in evidence) == 1
  )

  output = tmp_path / "descriptive"
  analyze_descriptive([run], output)
  single_event_rows = list(csv.DictReader((output / "single_event_stats.csv").open()))
  assert {row["evidence_type"] for row in single_event_rows} >= {
    "adjacent_support_first_v2",
    "adjacent_support_stable_v2",
    "adjacent_support_peak_v2",
    "oracle_contact_point_proxy_v2",
    "oracle_toe_proxy_v2",
    "oracle_root_proxy_v2",
  }
  assert (output / "logger_v2_event_counts.csv").exists()


def test_multilayer_support_slope_recovers_depth_with_foot_offset() -> None:
  key: SequenceKey = ("model_seed42", "1")
  sequence = SequenceRecord(
    key=key,
    run_id=key[0],
    seed=42,
    sequence_id=key[1],
    env_id=0,
    depth_bin=3,
    true_depth=0.30,
    true_height=0.10,
    termination_reason="confirmed_exit",
    raw={},
  )

  def support_event(
    event_id: int,
    frame: int,
    left_layer: int,
    right_layer: int,
  ) -> EventRecord:
    return EventRecord(
      key=key,
      run_id=key[0],
      seed=42,
      event_id=event_id,
      event_type="pair_full",
      frame_idx=frame,
      time=frame * 0.02,
      event_layer=max(left_layer, right_layer),
      raw={
        "event_family": "support_trajectory",
        "left_support_layer": str(left_layer),
        "right_support_layer": str(right_layer),
        "left_stair_contact": "1",
        "right_stair_contact": "1",
        "left_foot_s": str(0.10 + 0.30 * left_layer),
        "right_foot_s": str(0.12 + 0.30 * right_layer),
        "left_geometric_overlap": "1.0",
        "right_geometric_overlap": "1.0",
      },
    )

  events = [
    support_event(0, 10, 1, 2),
    support_event(1, 20, 3, 2),
    support_event(2, 30, 3, 4),
  ]
  sequences: dict[SequenceKey, SequenceRecord] = {key: sequence}
  visits = _support_visits(sequences, events)
  predictions = _multilayer_prediction_rows(sequences, visits)
  prefix_four = {
    row["method"]: row for row in predictions if row["prefix_distinct_layers"] == 4
  }

  assert len(visits) == 4
  assert prefix_four["huber_oracle_layer_foot_offset"][
    "predicted_depth"
  ] == pytest.approx(0.30)
  assert prefix_four["huber_oracle_transition_order_foot_offset"][
    "predicted_depth"
  ] == pytest.approx(0.30)
  stats = _multilayer_prefix_stats(sequences, visits, predictions)
  row = next(
    item
    for item in stats
    if item["prefix_distinct_layers"] == 4
    and item["method"] == "huber_oracle_layer_foot_offset"
  )
  assert row["eligible_sequences"] == 1
  assert row["predicted_sequences"] == 1
  assert row["mae"] == pytest.approx(0.0)


def _make_run(root: Path, seed: int, depth: float, depth_bin: int) -> Path:
  run = root / f"model_seed{seed}"
  _write_rows(
    run / "stair_sequences.csv",
    [
      _sequence_row("1", depth, depth_bin),
      _sequence_row("2", depth, depth_bin, oracle=True),
      _sequence_row("99", depth, depth_bin, process_end=True),
    ],
  )
  events = [
    _event_row("1", 4, "confirmation_duplicate", 40, layer=3, root_x=0.52),
    _event_row("1", 0, "first_layer1_collision", 10, layer=1, root_x=0.0),
    _event_row(
      "1",
      1,
      "adjacent_double_support_partial",
      20,
      layer=2,
      root_x=0.1,
      pair_depth=depth - 0.02,
      pair_height=0.09,
      left_support_ratio=0.75,
      right_support_ratio=0.90,
    ),
    _event_row("1", 2, "confirmation", 30, layer=2, root_x=depth),
    _event_row("1", 3, "confirmation_duplicate", 35, layer=3, root_x=0.5),
    _event_row("2", 0, "first_layer1_collision", 11, layer=1, root_x=0.0),
    _event_row(
      "2",
      1,
      "adjacent_double_support_full",
      21,
      layer=2,
      root_x=0.1,
      pair_depth=depth + 0.01,
      pair_height=0.105,
      left_support_ratio=1.0,
      right_support_ratio=1.0,
    ),
    _event_row(
      "2",
      2,
      "layer2plus_riser_collision",
      31,
      layer=2,
      root_x=depth + 0.015,
    ),
  ]
  _write_rows(run / "stair_events.csv", events)
  return run


@pytest.fixture
def synthetic_runs(tmp_path: Path) -> list[Path]:
  return [
    _make_run(tmp_path, 42, 0.25, 0),
    _make_run(tmp_path, 43, 0.30, 3),
    _make_run(tmp_path, 44, 0.35, 7),
  ]


def test_stage1_full_analysis_uses_composite_keys_and_held_out_split(
  synthetic_runs: list[Path], tmp_path: Path
) -> None:
  output = tmp_path / "summary"
  expected = {
    "sequences": 6,
    "oracle_sequences": 3,
    "confirmation_events": 3,
    "additional_layer_events": 6,
    "same_layer_duplicates": 0,
  }

  analyze(
    synthetic_runs,
    output,
    calibration_seeds={42},
    selection_seeds={43},
    test_seeds={44},
    expected_counts=expected,
  )

  audit = list(csv.DictReader((output / "data_audit_summary.csv").open()))
  audit_by_metric = {row["metric"]: row for row in audit}
  assert audit_by_metric["sequences"]["actual"] == "6"
  assert audit_by_metric["process_end_excluded"]["actual"] == "3"
  assert audit_by_metric["support_layer_available"]["actual"] == "False"
  assert audit_by_metric["additional_layer_events"]["actual"] == "6"
  assert audit_by_metric["same_layer_duplicates"]["actual"] == "0"
  assert audit_by_metric["stage1_unique_additional_layer_events"]["actual"] == "3"
  assert audit_by_metric["stage1_seen_layer_duplicates"]["actual"] == "3"

  calibration = list(csv.DictReader((output / "calibration_residuals.csv").open()))
  assert {row["calibration_seeds"] for row in calibration} == {"42"}
  support_rows = list(csv.DictReader((output / "pair_support_stats.csv").open()))
  test_support = {
    row["support_type"]: row for row in support_rows if row["split"] == "test"
  }
  assert test_support["adjacent_partial"]["num_samples"] == "1"
  assert test_support["adjacent_full"]["num_samples"] == "1"
  interval_rows = list(csv.DictReader((output / "interval_summary.csv").open()))
  assert {row["method"] for row in interval_rows} >= {
    "global_prior_90",
    "fixed_25_35",
    "constant_mean",
    "shuffled_evidence",
    "support_only",
    "detector_only",
    "oracle_contact_proxy",
    "fused_all",
  }
  assert {row["num_sequences"] for row in interval_rows} == {"2"}
  per_bin_rows = list(csv.DictReader((output / "per_depth_bin_stats.csv").open()))
  assert {row["row_type"] for row in per_bin_rows} == {
    "interval_method",
    "single_event",
  }
  assert (output / "interval_shrink_curve.csv").exists()
  assert (output / "sequence_interval_predictions.csv").exists()
  quality_summary = list(
    csv.DictReader((output / "support_quality_interval_summary.csv").open())
  )
  assert {row["method"] for row in quality_summary} == {
    "full_only",
    "binary_partial_full",
    "quality_binned",
    "quality_aware",
    "shuffled_quality",
  }
  assert {row["test_seeds"] for row in quality_summary} == {"44"}
  quality_calibration = list(
    csv.DictReader((output / "support_quality_calibration.csv").open())
  )
  assert any(
    row["row_type"] == "bucket_selection" and row["selected"] == "1"
    for row in quality_calibration
  )
  assert any(
    row["row_type"] == "neighbor_selection" and row["selected"] == "1"
    for row in quality_calibration
  )


def test_input_ordering_and_additional_layer_reconstruction(
  synthetic_runs: list[Path],
) -> None:
  sequences, events, excluded = _load_inputs(synthetic_runs, False)
  assert len(sequences) == 6
  assert len({record.key for record in sequences.values()}) == 6
  assert excluded == 3
  seed42_sequence = [event for event in events if event.key == ("model_seed42", "1")]
  assert [event.event_id for event in seed42_sequence] == [0, 1, 2, 3, 4]

  additional, duplicates = _reclassify_confirmation_events(events)
  assert (("model_seed42", "1"), 3) in additional
  assert (("model_seed42", "1"), 4) in duplicates
  assert not (additional & duplicates)


def test_calibration_uses_only_calibration_seed() -> None:
  evidence = [
    Evidence(
      key=("run", "1"),
      seed=42,
      evidence_type="adjacent_support_full",
      frame_idx=1,
      event_id=1,
      center=0.2,
      height_center=0.1,
      true_depth=0.25,
      true_height=0.1,
      depth_bin=0,
      proxy_type="test",
      confidence="medium",
    ),
    Evidence(
      key=("run", "1"),
      seed=44,
      evidence_type="adjacent_support_full",
      frame_idx=1,
      event_id=1,
      center=0.2,
      height_center=0.1,
      true_depth=0.35,
      true_height=0.1,
      depth_bin=0,
      proxy_type="test",
      confidence="medium",
    ),
  ]

  calibration = _calibrate(evidence, {42})["adjacent_support_full"]

  assert calibration.sample_count == 1
  assert calibration.q05 == pytest.approx(0.05)
  assert calibration.q95 == pytest.approx(0.05)


def test_support_quality_features_and_bucket_calibration_use_seed42() -> None:
  assert _support_quality(
    {"left_support_ratio": "0.9", "right_support_ratio": "0.6"}
  ) == pytest.approx((0.6, 0.75, 0.9, 0.3))
  missing = _support_quality({})
  assert all(math.isnan(value) for value in missing)

  evidence = [
    Evidence(
      key=("run42", "1"),
      seed=42,
      evidence_type="adjacent_support_partial",
      frame_idx=1,
      event_id=1,
      center=0.20,
      height_center=0.1,
      true_depth=0.25,
      true_height=0.1,
      depth_bin=0,
      proxy_type="test",
      confidence="low",
      quality_min=0.7,
    ),
    Evidence(
      key=("run44", "1"),
      seed=44,
      evidence_type="adjacent_support_partial",
      frame_idx=1,
      event_id=1,
      center=0.20,
      height_center=0.1,
      true_depth=0.35,
      true_height=0.1,
      depth_bin=7,
      proxy_type="test",
      confidence="low",
      quality_min=0.7,
    ),
  ]
  calibrations = _calibrate_quality_buckets(
    evidence, {42}, QualityScheme("test", (0.75,))
  )

  assert calibrations["partial_0"].sample_count == 1
  assert calibrations["partial_0"].q50 == pytest.approx(0.05)


def test_empty_intersection_records_explicit_fallback() -> None:
  sequence = SequenceRecord(
    key=("run", "1"),
    run_id="run",
    seed=44,
    sequence_id="1",
    env_id=0,
    depth_bin=0,
    true_depth=0.25,
    true_height=0.1,
    termination_reason="confirmed_exit",
    raw={},
  )
  evidence = Evidence(
    key=sequence.key,
    seed=44,
    evidence_type="adjacent_support_full",
    frame_idx=1,
    event_id=1,
    center=0.5,
    height_center=0.1,
    true_depth=0.25,
    true_height=0.1,
    depth_bin=0,
    proxy_type="test",
    confidence="medium",
  )

  prediction, traces = _fuse_sequence(
    sequence,
    "fused_all",
    [IntervalEvidence(evidence, 0.45, 0.55)],
    (0.25, 0.35),
    "keep_previous",
  )

  assert (prediction.lower, prediction.upper) == (0.25, 0.35)
  assert prediction.conflict_count == 1
  assert traces[0]["fallback_reason"] == "empty_intersection_keep_previous"


def test_shuffled_baseline_is_deterministic() -> None:
  sequences = [
    SequenceRecord(
      key=("run", str(index)),
      run_id="run",
      seed=44,
      sequence_id=str(index),
      env_id=index,
      depth_bin=index,
      true_depth=0.25 + index * 0.01,
      true_height=0.1,
      termination_reason="confirmed_exit",
      raw={},
    )
    for index in range(4)
  ]
  evidence_by_key: dict[tuple[str, str], list[IntervalEvidence]] = {}
  for index, sequence in enumerate(sequences):
    evidence = Evidence(
      key=sequence.key,
      seed=44,
      evidence_type="adjacent_support_full",
      frame_idx=1,
      event_id=1,
      center=float(index),
      height_center=0.1,
      true_depth=sequence.true_depth,
      true_height=0.1,
      depth_bin=index,
      proxy_type="test",
      confidence="medium",
      quality_min=0.5 + index * 0.1,
    )
    evidence_by_key[sequence.key] = [
      IntervalEvidence(evidence, float(index), float(index) + 0.1)
    ]

  first = _shuffled_evidence(sequences, evidence_by_key, 17)
  second = _shuffled_evidence(sequences, evidence_by_key, 17)

  assert {key: [item.lower for item in values] for key, values in first.items()} == {
    key: [item.lower for item in values] for key, values in second.items()
  }
  assert all(
    first[sequence.key][0].evidence.key != sequence.key for sequence in sequences
  )

  quality_first = _shuffled_quality_overrides(
    [values[0].evidence for values in evidence_by_key.values()], {44}, 17
  )
  quality_second = _shuffled_quality_overrides(
    [values[0].evidence for values in evidence_by_key.values()], {44}, 17
  )
  assert quality_first == quality_second
  assert all(
    quality_first[(sequence.key, 1)] != 0.5 + index * 0.1
    for index, sequence in enumerate(sequences)
  )


def test_degenerate_regression_is_nan_and_three_bin_metrics_work() -> None:
  regression = _regression_stats([1.0, 1.0], [2.0, 2.0])
  assert math.isnan(float(regression["correlation"]))
  assert math.isnan(float(regression["r_squared"]))
  assert _macro_f1([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)

  predictions = [
    Prediction(
      key=("run", str(index)),
      run_id="run",
      seed=44,
      sequence_id=str(index),
      env_id=index,
      method="test",
      true_depth=depth,
      true_height=0.1,
      depth_bin=depth_bin,
      lower=depth,
      upper=depth,
      evidence_count=1,
      evidence_types="test",
      conflict_count=0,
      fallback_reasons="",
      termination_reason="confirmed_exit",
    )
    for index, (depth, depth_bin) in enumerate([(0.25, 0), (0.30, 3), (0.35, 7)])
  ]
  stats = _prediction_stats(predictions, (0.275, 0.325), 2.0)
  assert stats["three_bin_accuracy"] == pytest.approx(1.0)
  assert stats["macro_f1"] == pytest.approx(1.0)


def test_audit_mismatch_stops_before_interval_analysis(
  synthetic_runs: list[Path], tmp_path: Path
) -> None:
  output = tmp_path / "failed"

  with pytest.raises(AuditError, match="audit contract"):
    analyze(
      synthetic_runs,
      output,
      calibration_seeds={42},
      selection_seeds={43},
      test_seeds={44},
      expected_counts={"sequences": 999},
    )

  assert (output / "data_audit_summary.csv").exists()
  assert not (output / "interval_summary.csv").exists()
