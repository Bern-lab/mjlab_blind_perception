from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
from tools.analyze_stair_intervals import (
  AuditError,
  Evidence,
  IntervalEvidence,
  Prediction,
  SequenceRecord,
  _calibrate,
  _fuse_sequence,
  _load_inputs,
  _macro_f1,
  _prediction_stats,
  _reclassify_confirmation_events,
  _regression_stats,
  _shuffled_evidence,
  analyze,
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
  }


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
