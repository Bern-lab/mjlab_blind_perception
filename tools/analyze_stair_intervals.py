"""Held-out Stage 1 analysis of physical stair-geometry evidence.

The script is offline-only. It reads Stage 0 CSV exports, calibrates evidence
intervals on one seed, selects conflict handling on another seed, and reports
final metrics on held-out seeds. It does not load or modify a policy.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Iterable, Mapping, Sequence, TypedDict

CsvRow = dict[str, str]
SequenceKey = tuple[str, str]

EXPECTED_STAGE0_COUNTS = {
  "sequences": 8562,
  "oracle_sequences": 817,
  "confirmation_events": 608,
  "additional_layer_events": 53,
  "same_layer_duplicates": 48,
}

SUPPORT_TYPES = {
  "adjacent_double_support_partial": "adjacent_support_partial",
  "adjacent_double_support_full": "adjacent_support_full",
}
DETECTOR_TYPES = {
  "detector_confirmation_proxy",
  "additional_layer_confirmation",
}
ORACLE_TYPES = {"oracle_contact_proxy"}
ALL_EVIDENCE_TYPES = set(SUPPORT_TYPES.values()) | DETECTOR_TYPES | ORACLE_TYPES


class AuditError(RuntimeError):
  """Raised when Stage 0 inputs do not match the requested audit contract."""


class FusionTrace(TypedDict):
  run_id: str
  sequence_id: str
  seed: int
  method: str
  event_index: int
  evidence_type: str
  width_before: float
  width_after: float
  coverage_before: int
  coverage_after: int
  interval_conflict: int
  fallback_reason: str


@dataclass(frozen=True)
class SequenceRecord:
  key: SequenceKey
  run_id: str
  seed: int
  sequence_id: str
  env_id: int
  depth_bin: int
  true_depth: float
  true_height: float
  termination_reason: str
  raw: CsvRow


@dataclass(frozen=True)
class EventRecord:
  key: SequenceKey
  run_id: str
  seed: int
  event_id: int
  event_type: str
  frame_idx: int
  time: float
  event_layer: int
  raw: CsvRow


@dataclass(frozen=True)
class Evidence:
  key: SequenceKey
  seed: int
  evidence_type: str
  frame_idx: int
  event_id: int
  center: float
  height_center: float
  true_depth: float
  true_height: float
  depth_bin: int
  proxy_type: str
  confidence: str


@dataclass(frozen=True)
class Calibration:
  evidence_type: str
  sample_count: int
  q05: float
  q50: float
  q95: float
  residual_mean: float
  residual_std: float


@dataclass(frozen=True)
class IntervalEvidence:
  evidence: Evidence
  lower: float
  upper: float


@dataclass(frozen=True)
class Prediction:
  key: SequenceKey
  run_id: str
  seed: int
  sequence_id: str
  env_id: int
  method: str
  true_depth: float
  true_height: float
  depth_bin: int
  lower: float
  upper: float
  evidence_count: int
  evidence_types: str
  conflict_count: int
  fallback_reasons: str
  termination_reason: str

  @property
  def center(self) -> float:
    return 0.5 * (self.lower + self.upper)

  @property
  def width(self) -> float:
    return self.upper - self.lower

  @property
  def covered(self) -> bool:
    return self.lower <= self.true_depth <= self.upper


def _read_csv(path: Path) -> list[CsvRow]:
  if not path.exists():
    raise FileNotFoundError(path)
  with path.open(encoding="utf-8", newline="") as stream:
    return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
  if rows:
    fieldnames: list[str] = []
    for row in rows:
      for key in row:
        if key not in fieldnames:
          fieldnames.append(key)
  else:
    fieldnames = ["empty"]
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def _as_float(row: Mapping[str, str], *keys: str, default: float = math.nan) -> float:
  for key in keys:
    value = row.get(key, "")
    if value not in ("", None):
      try:
        return float(value)
      except ValueError:
        continue
  return default


def _as_int(row: Mapping[str, str], key: str, default: int = -1) -> int:
  value = row.get(key, "")
  if value in ("", None):
    return default
  try:
    return int(float(value))
  except ValueError:
    return default


def _seed_from_name(name: str) -> int:
  match = re.search(r"seed[_-]?(\d+)", name)
  if match is None:
    raise ValueError(f"Cannot infer seed from input directory name: {name}")
  return int(match.group(1))


def _quantile(values: Iterable[float], probability: float) -> float:
  ordered = sorted(value for value in values if math.isfinite(value))
  if not ordered:
    return math.nan
  if len(ordered) == 1:
    return ordered[0]
  position = probability * (len(ordered) - 1)
  lower = math.floor(position)
  upper = math.ceil(position)
  fraction = position - lower
  return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _regression_stats(
  predictions: Sequence[float], labels: Sequence[float]
) -> dict[str, float | int]:
  pairs = [
    (prediction, label)
    for prediction, label in zip(predictions, labels, strict=True)
    if math.isfinite(prediction) and math.isfinite(label)
  ]
  if not pairs:
    return {
      "sample_count": 0,
      "mae": math.nan,
      "bias": math.nan,
      "correlation": math.nan,
      "r_squared": math.nan,
      "prediction_std": math.nan,
      "label_std": math.nan,
    }
  pred = [pair[0] for pair in pairs]
  label = [pair[1] for pair in pairs]
  errors = [prediction - target for prediction, target in pairs]
  pred_mean = mean(pred)
  label_mean = mean(label)
  pred_ss = sum((value - pred_mean) ** 2 for value in pred)
  label_ss = sum((value - label_mean) ** 2 for value in label)
  covariance = sum(
    (prediction - pred_mean) * (target - label_mean) for prediction, target in pairs
  )
  correlation = (
    covariance / math.sqrt(pred_ss * label_ss)
    if len(pairs) >= 2 and pred_ss > 0.0 and label_ss > 0.0
    else math.nan
  )
  squared_error = sum(error**2 for error in errors)
  r_squared = (
    1.0 - squared_error / label_ss if len(pairs) >= 2 and label_ss > 0.0 else math.nan
  )
  return {
    "sample_count": len(pairs),
    "mae": mean(abs(error) for error in errors),
    "bias": mean(errors),
    "correlation": correlation,
    "r_squared": r_squared,
    "prediction_std": pstdev(pred),
    "label_std": pstdev(label),
  }


def _load_inputs(
  input_dirs: Sequence[Path], include_process_end: bool
) -> tuple[dict[SequenceKey, SequenceRecord], list[EventRecord], int]:
  sequences: dict[SequenceKey, SequenceRecord] = {}
  events: list[EventRecord] = []
  excluded_process_end = 0
  for input_dir in input_dirs:
    directory = input_dir.expanduser().resolve()
    run_id = directory.name
    seed = _seed_from_name(run_id)
    sequence_rows = _read_csv(directory / "stair_sequences.csv")
    event_rows = _read_csv(directory / "stair_events.csv")
    valid_ids: set[str] = set()
    for row in sequence_rows:
      if not include_process_end and row.get("termination_reason") == "process_end":
        excluded_process_end += 1
        continue
      sequence_id = row.get("sequence_id", "")
      key = (run_id, sequence_id)
      if key in sequences:
        raise AuditError(f"Duplicate sequence key: {key}")
      record = SequenceRecord(
        key=key,
        run_id=run_id,
        seed=seed,
        sequence_id=sequence_id,
        env_id=_as_int(row, "env_id"),
        depth_bin=_as_int(row, "depth_bin"),
        true_depth=_as_float(
          row, "true_tread_depth", "stair_depth", "tread_depth", "depth"
        ),
        true_height=_as_float(
          row, "true_riser_height", "stair_height", "riser_height", "height"
        ),
        termination_reason=row.get("termination_reason", ""),
        raw=row,
      )
      if not math.isfinite(record.true_depth):
        raise AuditError(f"Missing true depth for sequence {key}")
      sequences[key] = record
      valid_ids.add(sequence_id)
    for row in event_rows:
      sequence_id = row.get("sequence_id", "")
      if sequence_id not in valid_ids:
        continue
      key = (run_id, sequence_id)
      events.append(
        EventRecord(
          key=key,
          run_id=run_id,
          seed=seed,
          event_id=_as_int(row, "event_id"),
          event_type=row.get("event_type", ""),
          frame_idx=_as_int(row, "frame_idx"),
          time=_as_float(row, "time", "event_time"),
          event_layer=_as_int(row, "event_layer"),
          raw=row,
        )
      )
  events.sort(
    key=lambda event: (event.key, event.frame_idx, event.time, event.event_id)
  )
  return sequences, events, excluded_process_end


def _reclassify_confirmation_events(
  events: Sequence[EventRecord],
) -> tuple[set[tuple[SequenceKey, int]], set[tuple[SequenceKey, int]]]:
  by_sequence: dict[SequenceKey, list[EventRecord]] = defaultdict(list)
  for event in events:
    by_sequence[event.key].append(event)
  additional: set[tuple[SequenceKey, int]] = set()
  duplicates: set[tuple[SequenceKey, int]] = set()
  for key_events in by_sequence.values():
    seen_layers: set[int] = set()
    has_confirmation = False
    for event in sorted(
      key_events, key=lambda item: (item.frame_idx, item.time, item.event_id)
    ):
      if event.event_type == "confirmation":
        has_confirmation = True
        if event.event_layer >= 0:
          seen_layers.add(event.event_layer)
      elif event.event_type == "confirmation_new_layer":
        has_confirmation = True
        if event.event_layer in seen_layers:
          duplicates.add((event.key, event.event_id))
        else:
          additional.add((event.key, event.event_id))
          seen_layers.add(event.event_layer)
      elif event.event_type == "confirmation_duplicate":
        if (
          has_confirmation
          and event.event_layer >= 0
          and event.event_layer not in seen_layers
        ):
          additional.add((event.key, event.event_id))
          seen_layers.add(event.event_layer)
        else:
          duplicates.add((event.key, event.event_id))
  return additional, duplicates


def _legacy_stage0_confirmation_counts(
  events: Sequence[EventRecord],
) -> tuple[int, int]:
  """Reproduce the published Stage 0 first-layer comparison for audit only."""
  by_sequence: dict[SequenceKey, list[EventRecord]] = defaultdict(list)
  for event in events:
    by_sequence[event.key].append(event)
  additional_count = 0
  duplicate_count = 0
  for key_events in by_sequence.values():
    ordered = sorted(
      key_events, key=lambda item: (item.frame_idx, item.time, item.event_id)
    )
    first_confirmation = next(
      (event for event in ordered if event.event_type == "confirmation"),
      None,
    )
    for event in ordered:
      if event.event_type == "confirmation_new_layer":
        additional_count += 1
      elif event.event_type == "confirmation_duplicate":
        if (
          first_confirmation is not None
          and event.event_layer != first_confirmation.event_layer
        ):
          additional_count += 1
        else:
          duplicate_count += 1
  return additional_count, duplicate_count


def _audit_rows(
  sequences: Mapping[SequenceKey, SequenceRecord],
  events: Sequence[EventRecord],
  additional: set[tuple[SequenceKey, int]],
  duplicates: set[tuple[SequenceKey, int]],
  excluded_process_end: int,
  expected_counts: Mapping[str, int] | None,
) -> tuple[list[dict[str, object]], bool]:
  legacy_additional, legacy_duplicates = _legacy_stage0_confirmation_counts(events)
  actual = {
    "sequences": len(sequences),
    "oracle_sequences": sum(
      _as_int(sequence.raw, "has_layer2plus_riser_collision", 0) == 1
      for sequence in sequences.values()
    ),
    "confirmation_events": sum(event.event_type == "confirmation" for event in events),
    "additional_layer_events": legacy_additional,
    "same_layer_duplicates": legacy_duplicates,
  }
  support_layer_available = any(
    "left_support_layer" in event.raw and "right_support_layer" in event.raw
    for event in events
  )
  rows: list[dict[str, object]] = [
    {
      "metric": "process_end_excluded",
      "actual": excluded_process_end,
      "expected": "",
      "status": "info",
    },
    {
      "metric": "support_layer_available",
      "actual": support_layer_available,
      "expected": "",
      "status": "info",
    },
    {
      "metric": "stage1_unique_additional_layer_events",
      "actual": len(additional),
      "expected": "",
      "status": "info",
    },
    {
      "metric": "stage1_seen_layer_duplicates",
      "actual": len(duplicates),
      "expected": "",
      "status": "info",
    },
  ]
  passed = True
  for metric, value in actual.items():
    expected = expected_counts.get(metric) if expected_counts is not None else None
    status = (
      "unchecked" if expected is None else ("pass" if value == expected else "fail")
    )
    passed &= status != "fail"
    rows.append(
      {
        "metric": metric,
        "actual": value,
        "expected": "" if expected is None else expected,
        "status": status,
      }
    )
  return rows, passed


def _projected_root(event: EventRecord) -> float:
  root_x = _as_float(event.raw, "root_x")
  root_y = _as_float(event.raw, "root_y")
  ascent_x = _as_float(event.raw, "ascent_dir_x")
  ascent_y = _as_float(event.raw, "ascent_dir_y")
  if not all(math.isfinite(value) for value in (root_x, root_y, ascent_x, ascent_y)):
    return math.nan
  return root_x * ascent_x + root_y * ascent_y


def _extract_evidence(
  sequences: Mapping[SequenceKey, SequenceRecord],
  events: Sequence[EventRecord],
  additional: set[tuple[SequenceKey, int]],
) -> list[Evidence]:
  by_sequence: dict[SequenceKey, list[EventRecord]] = defaultdict(list)
  for event in events:
    by_sequence[event.key].append(event)
  evidence: list[Evidence] = []
  for key, sequence_events in by_sequence.items():
    sequence = sequences[key]
    ordered = sorted(
      sequence_events, key=lambda event: (event.frame_idx, event.time, event.event_id)
    )
    entry = next(
      (event for event in ordered if event.event_type == "first_layer1_collision"),
      None,
    )
    for event in ordered:
      support_type = SUPPORT_TYPES.get(event.event_type)
      if support_type is not None:
        pair_depth = _as_float(event.raw, "pair_depth")
        pair_height = _as_float(event.raw, "pair_height")
        if math.isfinite(pair_depth) and pair_depth > 0.0:
          evidence.append(
            Evidence(
              key=key,
              seed=sequence.seed,
              evidence_type=support_type,
              frame_idx=event.frame_idx,
              event_id=event.event_id,
              center=pair_depth,
              height_center=pair_height,
              true_depth=sequence.true_depth,
              true_height=sequence.true_height,
              depth_bin=sequence.depth_bin,
              proxy_type="adjacent_foot_projection",
              confidence="medium" if support_type.endswith("full") else "low",
            )
          )
    if entry is not None:
      entry_projection = _projected_root(entry)
      oracle = next(
        (
          event for event in ordered if event.event_type == "layer2plus_riser_collision"
        ),
        None,
      )
      if oracle is not None:
        layer_gap = oracle.event_layer - 1
        center = (
          abs(_projected_root(oracle) - entry_projection) / layer_gap
          if layer_gap > 0
          else math.nan
        )
        if math.isfinite(center):
          evidence.append(
            Evidence(
              key=key,
              seed=sequence.seed,
              evidence_type="oracle_contact_proxy",
              frame_idx=oracle.frame_idx,
              event_id=oracle.event_id,
              center=center,
              height_center=math.nan,
              true_depth=sequence.true_depth,
              true_height=sequence.true_height,
              depth_bin=sequence.depth_bin,
              proxy_type="entry_to_contact_root_projection_per_layer",
              confidence="medium",
            )
          )
      first_confirmation = next(
        (event for event in ordered if event.event_type == "confirmation"),
        None,
      )
      if first_confirmation is not None:
        layer_gap = first_confirmation.event_layer - 1
        center = (
          abs(_projected_root(first_confirmation) - entry_projection) / layer_gap
          if layer_gap > 0
          else math.nan
        )
        if math.isfinite(center):
          evidence.append(
            Evidence(
              key=key,
              seed=sequence.seed,
              evidence_type="detector_confirmation_proxy",
              frame_idx=first_confirmation.frame_idx,
              event_id=first_confirmation.event_id,
              center=center,
              height_center=math.nan,
              true_depth=sequence.true_depth,
              true_height=sequence.true_height,
              depth_bin=sequence.depth_bin,
              proxy_type="entry_to_confirmation_root_projection_per_layer",
              confidence="medium",
            )
          )

    confirmation_events = [
      event
      for event in ordered
      if event.event_type == "confirmation" or (event.key, event.event_id) in additional
    ]
    previous: EventRecord | None = None
    for event in confirmation_events:
      if previous is not None:
        layer_gap = event.event_layer - previous.event_layer
        center = (
          abs(_projected_root(event) - _projected_root(previous)) / layer_gap
          if layer_gap > 0
          else math.nan
        )
        if math.isfinite(center):
          evidence.append(
            Evidence(
              key=key,
              seed=sequence.seed,
              evidence_type="additional_layer_confirmation",
              frame_idx=event.frame_idx,
              event_id=event.event_id,
              center=center,
              height_center=math.nan,
              true_depth=sequence.true_depth,
              true_height=sequence.true_height,
              depth_bin=sequence.depth_bin,
              proxy_type="consecutive_confirmation_root_projection_per_layer",
              confidence="medium",
            )
          )
      previous = event
  evidence.sort(
    key=lambda item: (item.key, item.frame_idx, item.event_id, item.evidence_type)
  )
  return evidence


def _split_name(
  seed: int,
  calibration_seeds: set[int],
  selection_seeds: set[int],
  test_seeds: set[int],
) -> str:
  if seed in calibration_seeds:
    return "calibration"
  if seed in selection_seeds:
    return "selection"
  if seed in test_seeds:
    return "test"
  return "unused"


def _calibrate(
  evidence: Sequence[Evidence], calibration_seeds: set[int]
) -> dict[str, Calibration]:
  grouped: dict[str, list[float]] = defaultdict(list)
  for item in evidence:
    if item.seed in calibration_seeds:
      grouped[item.evidence_type].append(item.true_depth - item.center)
  calibrations: dict[str, Calibration] = {}
  for evidence_type, residuals in sorted(grouped.items()):
    clean = [value for value in residuals if math.isfinite(value)]
    if not clean:
      continue
    calibrations[evidence_type] = Calibration(
      evidence_type=evidence_type,
      sample_count=len(clean),
      q05=_quantile(clean, 0.05),
      q50=_quantile(clean, 0.50),
      q95=_quantile(clean, 0.95),
      residual_mean=mean(clean),
      residual_std=pstdev(clean),
    )
  return calibrations


def _calibrated_evidence(
  evidence: Sequence[Evidence], calibrations: Mapping[str, Calibration]
) -> list[IntervalEvidence]:
  output: list[IntervalEvidence] = []
  for item in evidence:
    calibration = calibrations.get(item.evidence_type)
    if calibration is None:
      continue
    output.append(
      IntervalEvidence(
        evidence=item,
        lower=item.center + calibration.q05,
        upper=item.center + calibration.q95,
      )
    )
  return output


def _interval_score(
  lower: float, upper: float, target: float, outside_penalty: float
) -> float:
  return (
    upper
    - lower
    + outside_penalty * max(lower - target, 0.0)
    + outside_penalty * max(target - upper, 0.0)
  )


def _fuse_sequence(
  sequence: SequenceRecord,
  method: str,
  items: Sequence[IntervalEvidence],
  prior: tuple[float, float],
  fallback_policy: str,
) -> tuple[Prediction, list[FusionTrace]]:
  lower, upper = prior
  conflicts = 0
  fallback_reasons: list[str] = []
  evidence_types: list[str] = []
  traces: list[FusionTrace] = []
  for event_index, item in enumerate(
    sorted(
      items,
      key=lambda value: (
        value.evidence.frame_idx,
        value.evidence.event_id,
        value.evidence.evidence_type,
      ),
    ),
    start=1,
  ):
    before_lower, before_upper = lower, upper
    candidate_lower = max(lower, item.lower)
    candidate_upper = min(upper, item.upper)
    fallback_reason = ""
    if candidate_lower <= candidate_upper:
      lower, upper = candidate_lower, candidate_upper
    else:
      conflicts += 1
      if fallback_policy == "reset_to_prior":
        lower, upper = prior
        fallback_reason = "empty_intersection_reset_to_prior"
      else:
        fallback_reason = "empty_intersection_keep_previous"
      fallback_reasons.append(fallback_reason)
    evidence_types.append(item.evidence.evidence_type)
    traces.append(
      {
        "run_id": sequence.run_id,
        "sequence_id": sequence.sequence_id,
        "seed": sequence.seed,
        "method": method,
        "event_index": event_index,
        "evidence_type": item.evidence.evidence_type,
        "width_before": before_upper - before_lower,
        "width_after": upper - lower,
        "coverage_before": int(before_lower <= sequence.true_depth <= before_upper),
        "coverage_after": int(lower <= sequence.true_depth <= upper),
        "interval_conflict": int(bool(fallback_reason)),
        "fallback_reason": fallback_reason,
      }
    )
  prediction = Prediction(
    key=sequence.key,
    run_id=sequence.run_id,
    seed=sequence.seed,
    sequence_id=sequence.sequence_id,
    env_id=sequence.env_id,
    method=method,
    true_depth=sequence.true_depth,
    true_height=sequence.true_height,
    depth_bin=sequence.depth_bin,
    lower=lower,
    upper=upper,
    evidence_count=len(items),
    evidence_types="|".join(sorted(set(evidence_types))),
    conflict_count=conflicts,
    fallback_reasons="|".join(sorted(set(fallback_reasons))),
    termination_reason=sequence.termination_reason,
  )
  return prediction, traces


def _three_bin_thresholds(
  sequences: Sequence[SequenceRecord], calibration_seeds: set[int]
) -> tuple[float, float]:
  by_bin: dict[int, list[float]] = defaultdict(list)
  for sequence in sequences:
    if sequence.seed in calibration_seeds:
      by_bin[sequence.depth_bin].append(sequence.true_depth)
  bin_means = {
    depth_bin: mean(values) for depth_bin, values in by_bin.items() if values
  }
  if all(depth_bin in bin_means for depth_bin in range(8)):
    return (
      0.5 * (bin_means[2] + bin_means[3]),
      0.5 * (bin_means[4] + bin_means[5]),
    )
  depths = sorted(
    sequence.true_depth for sequence in sequences if sequence.seed in calibration_seeds
  )
  return _quantile(depths, 1.0 / 3.0), _quantile(depths, 2.0 / 3.0)


def _depth_class(depth: float, thresholds: tuple[float, float]) -> int:
  if depth < thresholds[0]:
    return 0
  if depth < thresholds[1]:
    return 1
  return 2


def _macro_f1(labels: Sequence[int], predictions: Sequence[int]) -> float:
  if not labels:
    return math.nan
  scores: list[float] = []
  for category in range(3):
    true_positive = sum(
      label == category and prediction == category
      for label, prediction in zip(labels, predictions, strict=True)
    )
    false_positive = sum(
      label != category and prediction == category
      for label, prediction in zip(labels, predictions, strict=True)
    )
    false_negative = sum(
      label == category and prediction != category
      for label, prediction in zip(labels, predictions, strict=True)
    )
    denominator = 2 * true_positive + false_positive + false_negative
    scores.append(2 * true_positive / denominator if denominator > 0 else 0.0)
  return mean(scores)


def _prediction_stats(
  predictions: Sequence[Prediction],
  thresholds: tuple[float, float],
  outside_penalty: float,
) -> dict[str, float | int]:
  if not predictions:
    return {
      "num_sequences": 0,
      "coverage": math.nan,
      "mean_width": math.nan,
      "median_width": math.nan,
      "interval_score": math.nan,
      "center_mae": math.nan,
      "center_bias": math.nan,
      "center_correlation": math.nan,
      "center_r_squared": math.nan,
      "three_bin_accuracy": math.nan,
      "macro_f1": math.nan,
      "conflict_rate": math.nan,
      "evidence_available_ratio": math.nan,
      "evidence_recall": math.nan,
    }
  regression = _regression_stats(
    [prediction.center for prediction in predictions],
    [prediction.true_depth for prediction in predictions],
  )
  labels = [
    0 if prediction.depth_bin <= 2 else (1 if prediction.depth_bin <= 4 else 2)
    for prediction in predictions
  ]
  classes = [_depth_class(prediction.center, thresholds) for prediction in predictions]
  return {
    "num_sequences": len(predictions),
    "coverage": mean(prediction.covered for prediction in predictions),
    "mean_width": mean(prediction.width for prediction in predictions),
    "median_width": median(prediction.width for prediction in predictions),
    "interval_score": mean(
      _interval_score(
        prediction.lower,
        prediction.upper,
        prediction.true_depth,
        outside_penalty,
      )
      for prediction in predictions
    ),
    "center_mae": regression["mae"],
    "center_bias": regression["bias"],
    "center_correlation": regression["correlation"],
    "center_r_squared": regression["r_squared"],
    "three_bin_accuracy": mean(
      label == prediction for label, prediction in zip(labels, classes, strict=True)
    ),
    "macro_f1": _macro_f1(labels, classes),
    "conflict_rate": mean(prediction.conflict_count > 0 for prediction in predictions),
    "evidence_available_ratio": mean(
      prediction.evidence_count > 0 for prediction in predictions
    ),
    "evidence_recall": mean(
      prediction.evidence_count > 0 for prediction in predictions
    ),
  }


def _filter_evidence(
  evidence: Sequence[IntervalEvidence], allowed_types: set[str]
) -> list[IntervalEvidence]:
  return [item for item in evidence if item.evidence.evidence_type in allowed_types]


def _predict_method(
  sequences: Sequence[SequenceRecord],
  evidence_by_key: Mapping[SequenceKey, list[IntervalEvidence]],
  method: str,
  allowed_types: set[str],
  prior: tuple[float, float],
  fallback_policy: str,
) -> tuple[list[Prediction], list[FusionTrace]]:
  predictions: list[Prediction] = []
  traces: list[FusionTrace] = []
  for sequence in sequences:
    items = _filter_evidence(evidence_by_key.get(sequence.key, []), allowed_types)
    prediction, sequence_traces = _fuse_sequence(
      sequence, method, items, prior, fallback_policy
    )
    predictions.append(prediction)
    traces.extend(sequence_traces)
  return predictions, traces


def _baseline_predictions(
  sequences: Sequence[SequenceRecord],
  method: str,
  interval: tuple[float, float],
) -> list[Prediction]:
  return [
    Prediction(
      key=sequence.key,
      run_id=sequence.run_id,
      seed=sequence.seed,
      sequence_id=sequence.sequence_id,
      env_id=sequence.env_id,
      method=method,
      true_depth=sequence.true_depth,
      true_height=sequence.true_height,
      depth_bin=sequence.depth_bin,
      lower=interval[0],
      upper=interval[1],
      evidence_count=0,
      evidence_types="",
      conflict_count=0,
      fallback_reasons="",
      termination_reason=sequence.termination_reason,
    )
    for sequence in sequences
  ]


def _shuffled_evidence(
  sequences: Sequence[SequenceRecord],
  evidence_by_key: Mapping[SequenceKey, list[IntervalEvidence]],
  random_seed: int,
) -> dict[SequenceKey, list[IntervalEvidence]]:
  keys = [sequence.key for sequence in sequences]
  if len(keys) > 1:
    offset = random.Random(random_seed).randrange(1, len(keys))
    donors = keys[offset:] + keys[:offset]
  else:
    donors = keys.copy()
  return {
    key: list(evidence_by_key.get(donor, []))
    for key, donor in zip(keys, donors, strict=True)
  }


def _select_fallback(
  sequences: Sequence[SequenceRecord],
  evidence_by_key: Mapping[SequenceKey, list[IntervalEvidence]],
  prior: tuple[float, float],
  thresholds: tuple[float, float],
  outside_penalty: float,
) -> tuple[str, list[dict[str, object]]]:
  rows: list[dict[str, object]] = []
  for policy in ("keep_previous", "reset_to_prior"):
    predictions, _ = _predict_method(
      sequences,
      evidence_by_key,
      "fused_all",
      ALL_EVIDENCE_TYPES,
      prior,
      policy,
    )
    stats = _prediction_stats(predictions, thresholds, outside_penalty)
    rows.append({"fallback_policy": policy, **stats})

  def selection_key(row: Mapping[str, object]) -> tuple[float, int]:
    score = row["interval_score"]
    if not isinstance(score, (int, float)):
      raise TypeError("interval_score must be numeric.")
    return (
      float(score),
      0 if row["fallback_policy"] == "keep_previous" else 1,
    )

  selected = min(
    rows,
    key=selection_key,
  )["fallback_policy"]
  return str(selected), rows


def _single_event_rows(
  evidence: Sequence[Evidence],
  calibrated: Sequence[IntervalEvidence],
  calibration_seeds: set[int],
  selection_seeds: set[int],
  test_seeds: set[int],
) -> list[dict[str, object]]:
  interval_lookup = {
    (
      item.evidence.key,
      item.evidence.event_id,
      item.evidence.evidence_type,
    ): item
    for item in calibrated
  }
  rows: list[dict[str, object]] = []
  for split in ("calibration", "selection", "test"):
    for evidence_type in sorted({item.evidence_type for item in evidence}):
      items = [
        item
        for item in evidence
        if item.evidence_type == evidence_type
        and _split_name(item.seed, calibration_seeds, selection_seeds, test_seeds)
        == split
      ]
      regression = _regression_stats(
        [item.center for item in items], [item.true_depth for item in items]
      )
      intervals = [
        interval_lookup[(item.key, item.event_id, item.evidence_type)]
        for item in items
        if (item.key, item.event_id, item.evidence_type) in interval_lookup
      ]
      rows.append(
        {
          "split": split,
          "evidence_type": evidence_type,
          "proxy_type": items[0].proxy_type if items else "",
          "num_sequences": len({item.key for item in items}),
          **regression,
          "coverage": (
            mean(
              item.lower <= item.evidence.true_depth <= item.upper for item in intervals
            )
            if intervals
            else math.nan
          ),
          "mean_interval_width": (
            mean(item.upper - item.lower for item in intervals)
            if intervals
            else math.nan
          ),
        }
      )
  return rows


def _pair_support_rows(
  evidence: Sequence[Evidence],
  calibration_seeds: set[int],
  selection_seeds: set[int],
  test_seeds: set[int],
) -> list[dict[str, object]]:
  rows: list[dict[str, object]] = []
  group_types = {
    "adjacent_partial": {"adjacent_support_partial"},
    "adjacent_full": {"adjacent_support_full"},
    "all_adjacent": {"adjacent_support_partial", "adjacent_support_full"},
  }
  for split in ("calibration", "selection", "test"):
    for support_type, allowed in group_types.items():
      items = [
        item
        for item in evidence
        if item.evidence_type in allowed
        and _split_name(item.seed, calibration_seeds, selection_seeds, test_seeds)
        == split
      ]
      depth_stats = _regression_stats(
        [item.center for item in items], [item.true_depth for item in items]
      )
      height_items = [
        item
        for item in items
        if math.isfinite(item.height_center) and math.isfinite(item.true_height)
      ]
      height_stats = _regression_stats(
        [item.height_center for item in height_items],
        [item.true_height for item in height_items],
      )
      rows.append(
        {
          "split": split,
          "support_type": support_type,
          "num_samples": len(items),
          "num_sequences": len({item.key for item in items}),
          "pair_depth_mae": depth_stats["mae"],
          "pair_depth_bias": depth_stats["bias"],
          "pair_depth_correlation": depth_stats["correlation"],
          "pair_depth_r_squared": depth_stats["r_squared"],
          "pair_height_mae": height_stats["mae"],
          "pair_height_bias": height_stats["bias"],
          "pair_height_correlation": height_stats["correlation"],
          "pair_height_r_squared": height_stats["r_squared"],
        }
      )
  return rows


def _single_event_per_bin_rows(
  evidence: Sequence[Evidence],
  calibrated: Sequence[IntervalEvidence],
  test_seeds: set[int],
) -> list[dict[str, object]]:
  interval_lookup = {
    (
      item.evidence.key,
      item.evidence.event_id,
      item.evidence.evidence_type,
    ): item
    for item in calibrated
  }
  rows: list[dict[str, object]] = []
  for evidence_type in sorted({item.evidence_type for item in evidence}):
    for depth_bin in range(8):
      items = [
        item
        for item in evidence
        if item.seed in test_seeds
        and item.evidence_type == evidence_type
        and item.depth_bin == depth_bin
      ]
      regression = _regression_stats(
        [item.center for item in items], [item.true_depth for item in items]
      )
      intervals = [
        interval_lookup[(item.key, item.event_id, item.evidence_type)]
        for item in items
        if (item.key, item.event_id, item.evidence_type) in interval_lookup
      ]
      rows.append(
        {
          "row_type": "single_event",
          "split": "test",
          "method": "",
          "evidence_type": evidence_type,
          "depth_bin": depth_bin,
          "true_depth_mean": (
            mean(item.true_depth for item in items) if items else math.nan
          ),
          "num_sequences": len({item.key for item in items}),
          **regression,
          "coverage": (
            mean(
              item.lower <= item.evidence.true_depth <= item.upper for item in intervals
            )
            if intervals
            else math.nan
          ),
          "mean_width": (
            mean(item.upper - item.lower for item in intervals)
            if intervals
            else math.nan
          ),
        }
      )
  return rows


def _aggregate_shrink_rows(
  traces: Sequence[FusionTrace],
) -> list[dict[str, object]]:
  grouped: dict[tuple[str, int], list[FusionTrace]] = defaultdict(list)
  for trace in traces:
    grouped[(trace["method"], trace["event_index"])].append(trace)
  rows: list[dict[str, object]] = []
  for (method, event_index), items in sorted(grouped.items()):
    reasons = Counter(str(item["fallback_reason"]) for item in items)
    reasons.pop("", None)
    rows.append(
      {
        "method": method,
        "event_index": event_index,
        "num_sequences": len(items),
        "mean_width_before": mean(item["width_before"] for item in items),
        "mean_width_after": mean(item["width_after"] for item in items),
        "median_width_after": median(item["width_after"] for item in items),
        "coverage_before": mean(item["coverage_before"] for item in items),
        "coverage_after": mean(item["coverage_after"] for item in items),
        "conflict_rate": mean(item["interval_conflict"] for item in items),
        "fallback_reason": (reasons.most_common(1)[0][0] if reasons else ""),
      }
    )
  return rows


def analyze(
  input_dirs: Sequence[Path],
  output_dir: Path,
  calibration_seeds: set[int],
  selection_seeds: set[int],
  test_seeds: set[int],
  *,
  include_process_end: bool = False,
  expected_counts: Mapping[str, int] | None = EXPECTED_STAGE0_COUNTS,
  outside_penalty: float = 2.0,
  shuffle_seed: int = 17,
) -> None:
  """Run Stage 1 and write held-out analysis CSV files."""
  if calibration_seeds & selection_seeds or calibration_seeds & test_seeds:
    raise ValueError("Calibration seeds must be disjoint from other splits.")
  if selection_seeds & test_seeds:
    raise ValueError("Selection and test seeds must be disjoint.")
  output = output_dir.expanduser().resolve()
  output.mkdir(parents=True, exist_ok=True)
  sequences, events, excluded = _load_inputs(input_dirs, include_process_end)
  additional, duplicates = _reclassify_confirmation_events(events)
  audit_rows, audit_passed = _audit_rows(
    sequences, events, additional, duplicates, excluded, expected_counts
  )
  audit_rows.extend(
    [
      {
        "metric": "calibration_seeds",
        "actual": "|".join(map(str, sorted(calibration_seeds))),
        "expected": "",
        "status": "info",
      },
      {
        "metric": "selection_seeds",
        "actual": "|".join(map(str, sorted(selection_seeds))),
        "expected": "",
        "status": "info",
      },
      {
        "metric": "test_seeds",
        "actual": "|".join(map(str, sorted(test_seeds))),
        "expected": "",
        "status": "info",
      },
    ]
  )
  _write_csv(output / "data_audit_summary.csv", audit_rows)
  if not audit_passed:
    raise AuditError(
      f"Stage 0 counts do not match the requested audit contract; see {output}."
    )

  sequence_values = list(sequences.values())
  present_seeds = {sequence.seed for sequence in sequence_values}
  required_seeds = calibration_seeds | selection_seeds | test_seeds
  if not required_seeds <= present_seeds:
    raise AuditError(
      f"Requested seeds are missing: {sorted(required_seeds - present_seeds)}"
    )

  evidence = _extract_evidence(sequences, events, additional)
  calibrations = _calibrate(evidence, calibration_seeds)
  calibrated = _calibrated_evidence(evidence, calibrations)
  calibration_sequences = [
    sequence for sequence in sequence_values if sequence.seed in calibration_seeds
  ]
  calibration_depths = [sequence.true_depth for sequence in calibration_sequences]
  prior = (
    _quantile(calibration_depths, 0.05),
    _quantile(calibration_depths, 0.95),
  )
  constant_mean = mean(calibration_depths)
  thresholds = _three_bin_thresholds(sequence_values, calibration_seeds)

  evidence_by_key: dict[SequenceKey, list[IntervalEvidence]] = defaultdict(list)
  for item in calibrated:
    evidence_by_key[item.evidence.key].append(item)
  selection_sequences = [
    sequence for sequence in sequence_values if sequence.seed in selection_seeds
  ]
  fallback_policy, selection_rows = _select_fallback(
    selection_sequences,
    evidence_by_key,
    prior,
    thresholds,
    outside_penalty,
  )
  for row in selection_rows:
    row["selected"] = int(row["fallback_policy"] == fallback_policy)
  audit_rows.extend(
    {
      "metric": f"fallback_{row['fallback_policy']}_interval_score",
      "actual": row["interval_score"],
      "expected": "",
      "status": "selected" if row["selected"] else "candidate",
    }
    for row in selection_rows
  )
  _write_csv(output / "data_audit_summary.csv", audit_rows)

  test_sequences = [
    sequence for sequence in sequence_values if sequence.seed in test_seeds
  ]
  predictions_by_method: dict[str, list[Prediction]] = {}
  all_traces: list[FusionTrace] = []
  predictions_by_method["global_prior_90"] = _baseline_predictions(
    test_sequences, "global_prior_90", prior
  )
  predictions_by_method["fixed_25_35"] = _baseline_predictions(
    test_sequences, "fixed_25_35", (0.25, 0.35)
  )
  predictions_by_method["constant_mean"] = _baseline_predictions(
    test_sequences, "constant_mean", (constant_mean, constant_mean)
  )
  method_types = {
    "support_only": set(SUPPORT_TYPES.values()),
    "detector_only": DETECTOR_TYPES,
    "oracle_contact_proxy": ORACLE_TYPES,
    "fused_all": ALL_EVIDENCE_TYPES,
  }
  for method, allowed_types in method_types.items():
    method_predictions, traces = _predict_method(
      test_sequences,
      evidence_by_key,
      method,
      allowed_types,
      prior,
      fallback_policy,
    )
    predictions_by_method[method] = method_predictions
    all_traces.extend(traces)

  shuffled_by_key = _shuffled_evidence(test_sequences, evidence_by_key, shuffle_seed)
  shuffled_predictions, shuffled_traces = _predict_method(
    test_sequences,
    shuffled_by_key,
    "shuffled_evidence",
    ALL_EVIDENCE_TYPES,
    prior,
    fallback_policy,
  )
  predictions_by_method["shuffled_evidence"] = shuffled_predictions
  all_traces.extend(shuffled_traces)

  interval_rows = [
    {
      "split": "test",
      "test_seeds": "|".join(map(str, sorted(test_seeds))),
      "method": method,
      "fallback_policy": fallback_policy,
      **_prediction_stats(predictions, thresholds, outside_penalty),
    }
    for method, predictions in predictions_by_method.items()
  ]
  _write_csv(output / "interval_summary.csv", interval_rows)
  _write_csv(
    output / "shuffled_baseline_summary.csv",
    [
      {
        "shuffle_seed": shuffle_seed,
        **next(row for row in interval_rows if row["method"] == "shuffled_evidence"),
      }
    ],
  )

  calibration_rows = [
    {
      "calibration_seeds": "|".join(map(str, sorted(calibration_seeds))),
      "evidence_type": calibration.evidence_type,
      "sample_count": calibration.sample_count,
      "q05": calibration.q05,
      "q50": calibration.q50,
      "q95": calibration.q95,
      "mean": calibration.residual_mean,
      "std": calibration.residual_std,
    }
    for calibration in calibrations.values()
  ]
  _write_csv(output / "calibration_residuals.csv", calibration_rows)
  _write_csv(
    output / "single_event_stats.csv",
    _single_event_rows(
      evidence,
      calibrated,
      calibration_seeds,
      selection_seeds,
      test_seeds,
    ),
  )
  _write_csv(
    output / "pair_support_stats.csv",
    _pair_support_rows(evidence, calibration_seeds, selection_seeds, test_seeds),
  )
  confirmation_rows = [
    row
    for row in _single_event_rows(
      evidence,
      calibrated,
      calibration_seeds,
      selection_seeds,
      test_seeds,
    )
    if row["evidence_type"]
    in {
      "oracle_contact_proxy",
      "detector_confirmation_proxy",
      "additional_layer_confirmation",
    }
  ]
  _write_csv(output / "confirmation_displacement_stats.csv", confirmation_rows)
  _write_csv(
    output / "interval_shrink_curve.csv",
    _aggregate_shrink_rows(all_traces),
  )

  per_bin_rows: list[dict[str, object]] = []
  for method, predictions in predictions_by_method.items():
    for depth_bin in range(8):
      subset = [
        prediction for prediction in predictions if prediction.depth_bin == depth_bin
      ]
      per_bin_rows.append(
        {
          "row_type": "interval_method",
          "split": "test",
          "method": method,
          "evidence_type": "",
          "depth_bin": depth_bin,
          "true_depth_mean": (
            mean(prediction.true_depth for prediction in subset) if subset else math.nan
          ),
          **_prediction_stats(subset, thresholds, outside_penalty),
        }
      )
  per_bin_rows.extend(_single_event_per_bin_rows(evidence, calibrated, test_seeds))
  _write_csv(output / "per_depth_bin_stats.csv", per_bin_rows)

  prediction_rows: list[dict[str, object]] = []
  for predictions in predictions_by_method.values():
    for prediction in predictions:
      prediction_rows.append(
        {
          "run_id": prediction.run_id,
          "seed": prediction.seed,
          "env_id": prediction.env_id,
          "sequence_id": prediction.sequence_id,
          "method": prediction.method,
          "true_depth": prediction.true_depth,
          "true_height": prediction.true_height,
          "depth_bin": prediction.depth_bin,
          "final_lower": prediction.lower,
          "final_upper": prediction.upper,
          "final_center": prediction.center,
          "final_width": prediction.width,
          "covered": int(prediction.covered),
          "num_evidence": prediction.evidence_count,
          "used_evidence_types": prediction.evidence_types,
          "interval_conflict": int(prediction.conflict_count > 0),
          "conflict_count": prediction.conflict_count,
          "fallback_reason": prediction.fallback_reasons,
          "termination_reason": prediction.termination_reason,
        }
      )
  _write_csv(output / "sequence_interval_predictions.csv", prediction_rows)
  print(
    "[Stage 1] Complete:",
    f"sequences={len(sequence_values)}",
    f"evidence={len(evidence)}",
    f"fallback={fallback_policy}",
    f"output={output}",
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input-dirs", nargs="+", required=True, type=Path)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--calibration-seeds", nargs="+", type=int, default=[42])
  parser.add_argument("--selection-seeds", nargs="+", type=int, default=[43])
  parser.add_argument("--test-seeds", nargs="+", type=int, default=[44])
  parser.add_argument("--include-process-end", action="store_true")
  parser.add_argument("--skip-expected-count-check", action="store_true")
  parser.add_argument("--outside-penalty", type=float, default=2.0)
  parser.add_argument("--shuffle-seed", type=int, default=17)
  args = parser.parse_args()
  analyze(
    input_dirs=args.input_dirs,
    output_dir=args.output_dir,
    calibration_seeds=set(args.calibration_seeds),
    selection_seeds=set(args.selection_seeds),
    test_seeds=set(args.test_seeds),
    include_process_end=args.include_process_end,
    expected_counts=(
      None if args.skip_expected_count_check else EXPECTED_STAGE0_COUNTS
    ),
    outside_penalty=args.outside_penalty,
    shuffle_seed=args.shuffle_seed,
  )


if __name__ == "__main__":
  main()
