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

import numpy as np

CsvRow = dict[str, str]
SequenceKey = tuple[str, str]
CONTACT_ASSOCIATION_MAX_S_RESIDUAL = 0.03
MULTILAYER_PREFIX_COUNTS = (2, 3, 4, 5)

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
  "pair_enter": "adjacent_support_first_v2",
  "pair_stable_for_N_frames": "adjacent_support_stable_v2",
  "pair_full": "adjacent_support_full_v2",
  "pair_peak_quality": "adjacent_support_peak_v2",
}
SUPPORT_EVIDENCE_TYPES = set(SUPPORT_TYPES.values())
DETECTOR_TYPES = {
  "detector_confirmation_proxy",
  "additional_layer_confirmation",
}
ORACLE_TYPES = {
  "oracle_contact_proxy",
  "oracle_contact_point_proxy_v2",
  "oracle_toe_proxy_v2",
  "oracle_root_proxy_v2",
}
ALL_EVIDENCE_TYPES = SUPPORT_EVIDENCE_TYPES | DETECTOR_TYPES | ORACLE_TYPES
MIN_SELECTION_COVERAGE = 0.85


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


class MultilayerPredictionRow(TypedDict):
  run_id: str
  seed: int
  sequence_id: str
  depth_bin: int
  true_depth: float
  prefix_distinct_layers: int
  available_distinct_layers: int
  num_support_points: int
  observed_layers: str
  method: str
  predicted_depth: float
  error: float


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
  quality_min: float = math.nan
  quality_mean: float = math.nan
  quality_max: float = math.nan
  quality_asymmetry: float = math.nan


@dataclass(frozen=True)
class SupportVisit:
  key: SequenceKey
  seed: int
  sequence_id: str
  foot_id: int
  support_layer: int
  frame_idx: int
  event_id: int
  event_type: str
  foot_s: float
  quality: float
  true_depth: float
  depth_bin: int


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
class QualityScheme:
  name: str
  partial_thresholds: tuple[float, ...]


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


def _support_quality(row: Mapping[str, str]) -> tuple[float, float, float, float]:
  quality = (
    _as_float(row, "pair_quality_min"),
    _as_float(row, "pair_quality_mean"),
    _as_float(row, "pair_quality_max"),
    _as_float(row, "pair_quality_asymmetry"),
  )
  if all(math.isfinite(value) for value in quality):
    return quality
  left = _as_float(row, "left_support_ratio")
  right = _as_float(row, "right_support_ratio")
  if not math.isfinite(left) or not math.isfinite(right):
    return math.nan, math.nan, math.nan, math.nan
  return (
    min(left, right),
    0.5 * (left + right),
    max(left, right),
    abs(left - right),
  )


def _contact_association_valid(event: EventRecord) -> bool:
  if _as_int(event.raw, "contact_valid", 0) != 1:
    return False
  residual = _as_float(event.raw, "contact_s_minus_riser_s")
  return not math.isfinite(residual) or (
    abs(residual) <= CONTACT_ASSOCIATION_MAX_S_RESIDUAL
  )


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
        if event.event_type == "pair_peak_quality":
          pair_depth = _as_float(event.raw, "peak_pair_depth")
          pair_height = _as_float(event.raw, "peak_pair_height")
        else:
          pair_depth = _as_float(event.raw, "pair_depth")
          pair_height = _as_float(event.raw, "pair_height")
        quality_min, quality_mean, quality_max, quality_asymmetry = _support_quality(
          event.raw
        )
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
              quality_min=quality_min,
              quality_mean=quality_mean,
              quality_max=quality_max,
              quality_asymmetry=quality_asymmetry,
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

    oracle_v2_by_foot: dict[int, list[EventRecord]] = defaultdict(list)
    for event in ordered:
      if (
        event.event_type == "oracle_riser_contact"
        and _contact_association_valid(event)
        and event.event_layer >= 2
      ):
        oracle_v2_by_foot[_as_int(event.raw, "foot_id", -1)].append(event)
    proxy_fields = (
      (
        "contact_point_s",
        "oracle_contact_point_proxy_v2",
        "consecutive_oracle_contact_point_per_layer",
        "high",
      ),
      (
        "toe_s",
        "oracle_toe_proxy_v2",
        "consecutive_oracle_toe_projection_per_layer",
        "medium",
      ),
      (
        "root_s",
        "oracle_root_proxy_v2",
        "consecutive_oracle_root_projection_per_layer",
        "low",
      ),
    )
    for foot_events in oracle_v2_by_foot.values():
      previous: EventRecord | None = None
      for event in foot_events:
        if previous is not None and event.event_layer != previous.event_layer:
          layer_gap = abs(event.event_layer - previous.event_layer)
          for field, evidence_type, proxy_type, confidence in proxy_fields:
            current_s = _as_float(event.raw, field)
            previous_s = _as_float(previous.raw, field)
            center = (
              abs(current_s - previous_s) / layer_gap
              if math.isfinite(current_s) and math.isfinite(previous_s)
              else math.nan
            )
            if math.isfinite(center):
              evidence.append(
                Evidence(
                  key=key,
                  seed=sequence.seed,
                  evidence_type=evidence_type,
                  frame_idx=event.frame_idx,
                  event_id=event.event_id,
                  center=center,
                  height_center=math.nan,
                  true_depth=sequence.true_depth,
                  true_height=sequence.true_height,
                  depth_bin=sequence.depth_bin,
                  proxy_type=proxy_type,
                  confidence=confidence,
                )
              )
        previous = event

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


_SUPPORT_VISIT_PRIORITY = {
  "pair_enter": 0,
  "pair_quality_cross_0.75": 1,
  "pair_quality_cross_0.90": 2,
  "pair_full": 3,
  "pair_stable_for_N_frames": 4,
}


def _support_visits(
  sequences: Mapping[SequenceKey, SequenceRecord],
  events: Sequence[EventRecord],
) -> list[SupportVisit]:
  candidates: dict[tuple[SequenceKey, int, int], list[SupportVisit]] = defaultdict(list)
  for event in events:
    if event.event_type not in _SUPPORT_VISIT_PRIORITY:
      continue
    sequence = sequences[event.key]
    for foot_id, side in enumerate(("left", "right")):
      layer = _as_int(event.raw, f"{side}_support_layer", -1)
      foot_s = _as_float(event.raw, f"{side}_foot_s")
      stair_contact = _as_int(event.raw, f"{side}_stair_contact", int(layer > 0))
      quality = _as_float(
        event.raw,
        f"{side}_geometric_overlap",
        f"{side}_support_ratio",
      )
      if layer <= 0 or stair_contact != 1 or not math.isfinite(foot_s):
        continue
      candidates[(event.key, foot_id, layer)].append(
        SupportVisit(
          key=event.key,
          seed=sequence.seed,
          sequence_id=sequence.sequence_id,
          foot_id=foot_id,
          support_layer=layer,
          frame_idx=event.frame_idx,
          event_id=event.event_id,
          event_type=event.event_type,
          foot_s=foot_s,
          quality=quality,
          true_depth=sequence.true_depth,
          depth_bin=sequence.depth_bin,
        )
      )

  visits: list[SupportVisit] = []
  for group in candidates.values():
    best_priority = max(_SUPPORT_VISIT_PRIORITY[item.event_type] for item in group)
    best = [
      item
      for item in group
      if _SUPPORT_VISIT_PRIORITY[item.event_type] == best_priority
    ]
    center = median(item.foot_s for item in best)
    selected = min(
      best,
      key=lambda item: (
        abs(item.foot_s - center),
        item.frame_idx,
        item.event_id,
      ),
    )
    visits.append(selected)
  visits.sort(
    key=lambda item: (
      item.key,
      item.frame_idx,
      item.event_id,
      item.foot_id,
      item.support_layer,
    )
  )
  return visits


def _linear_slope(
  visits: Sequence[SupportVisit],
  layer_by_visit: Mapping[tuple[int, int], float],
  *,
  include_foot_offset: bool,
  robust: bool,
) -> float:
  if len(visits) < 2:
    return math.nan
  layers = np.asarray(
    [layer_by_visit[(visit.foot_id, visit.support_layer)] for visit in visits],
    dtype=np.float64,
  )
  columns = [np.ones(len(visits), dtype=np.float64), layers]
  if include_foot_offset:
    columns.append(
      np.asarray([visit.foot_id == 1 for visit in visits], dtype=np.float64)
    )
  design = np.column_stack(columns)
  target = np.asarray([visit.foot_s for visit in visits], dtype=np.float64)
  if np.linalg.matrix_rank(design) < design.shape[1]:
    return math.nan
  coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
  if robust:
    for _iteration in range(12):
      residual = target - design @ coefficients
      scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
      if not math.isfinite(float(scale)) or scale < 1.0e-6:
        break
      threshold = 1.345 * scale
      weights = np.minimum(1.0, threshold / np.maximum(np.abs(residual), 1.0e-12))
      weighted_design = design * np.sqrt(weights)[:, None]
      weighted_target = target * np.sqrt(weights)
      updated = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)[0]
      if float(np.max(np.abs(updated - coefficients))) < 1.0e-8:
        coefficients = updated
        break
      coefficients = updated
  return float(coefficients[1])


def _endpoint_depth(visits: Sequence[SupportVisit]) -> float:
  by_layer: dict[int, list[float]] = defaultdict(list)
  for visit in visits:
    by_layer[visit.support_layer].append(visit.foot_s)
  if len(by_layer) < 2:
    return math.nan
  lower = min(by_layer)
  upper = max(by_layer)
  return abs(median(by_layer[upper]) - median(by_layer[lower])) / (upper - lower)


def _multilayer_prediction_rows(
  sequences: Mapping[SequenceKey, SequenceRecord],
  visits: Sequence[SupportVisit],
) -> list[MultilayerPredictionRow]:
  by_sequence: dict[SequenceKey, list[SupportVisit]] = defaultdict(list)
  for visit in visits:
    by_sequence[visit.key].append(visit)
  rows: list[MultilayerPredictionRow] = []
  for key, sequence_visits in by_sequence.items():
    sequence = sequences[key]
    first_frame_by_layer: dict[int, int] = {}
    for visit in sequence_visits:
      first_frame_by_layer[visit.support_layer] = min(
        first_frame_by_layer.get(visit.support_layer, visit.frame_idx),
        visit.frame_idx,
      )
    ordered_layers = sorted(
      first_frame_by_layer,
      key=lambda layer: (first_frame_by_layer[layer], layer),
    )
    for prefix_count in MULTILAYER_PREFIX_COUNTS:
      if len(ordered_layers) < prefix_count:
        continue
      prefix_layers = ordered_layers[:prefix_count]
      selected = [
        visit for visit in sequence_visits if visit.support_layer in prefix_layers
      ]
      oracle_layers = {
        (visit.foot_id, visit.support_layer): float(visit.support_layer)
        for visit in selected
      }
      ordinal_index = {layer: float(index) for index, layer in enumerate(prefix_layers)}
      ordinal_layers = {
        (visit.foot_id, visit.support_layer): ordinal_index[visit.support_layer]
        for visit in selected
      }
      shuffled = list(prefix_layers)
      random.Random(f"{sequence.run_id}:{sequence.sequence_id}:{prefix_count}").shuffle(
        shuffled
      )
      shuffled_index = {
        layer: float(shuffled[index]) for index, layer in enumerate(prefix_layers)
      }
      shuffled_layers = {
        (visit.foot_id, visit.support_layer): shuffled_index[visit.support_layer]
        for visit in selected
      }
      predictions = {
        "endpoint_oracle_layer": _endpoint_depth(selected),
        "ols_oracle_layer": _linear_slope(
          selected,
          oracle_layers,
          include_foot_offset=False,
          robust=False,
        ),
        "huber_oracle_layer": _linear_slope(
          selected,
          oracle_layers,
          include_foot_offset=False,
          robust=True,
        ),
        "huber_oracle_layer_foot_offset": _linear_slope(
          selected,
          oracle_layers,
          include_foot_offset=True,
          robust=True,
        ),
        "huber_oracle_transition_order_foot_offset": _linear_slope(
          selected,
          ordinal_layers,
          include_foot_offset=True,
          robust=True,
        ),
        "shuffled_oracle_layer_foot_offset": _linear_slope(
          selected,
          shuffled_layers,
          include_foot_offset=True,
          robust=True,
        ),
      }
      for method, prediction in predictions.items():
        if not math.isfinite(prediction):
          continue
        rows.append(
          {
            "run_id": sequence.run_id,
            "seed": sequence.seed,
            "sequence_id": sequence.sequence_id,
            "depth_bin": sequence.depth_bin,
            "true_depth": sequence.true_depth,
            "prefix_distinct_layers": prefix_count,
            "available_distinct_layers": len(ordered_layers),
            "num_support_points": len(selected),
            "observed_layers": "|".join(map(str, prefix_layers)),
            "method": method,
            "predicted_depth": prediction,
            "error": prediction - sequence.true_depth,
          }
        )
  return rows


def _depth_class_from_prediction(depth: float) -> int:
  if depth < 0.5 * (0.25 + 2.0 * 0.10 / 7.0 + 0.25 + 3.0 * 0.10 / 7.0):
    return 0
  if depth < 0.5 * (0.25 + 4.0 * 0.10 / 7.0 + 0.25 + 5.0 * 0.10 / 7.0):
    return 1
  return 2


def _depth_class_from_bin(depth_bin: int) -> int:
  if depth_bin <= 2:
    return 0
  if depth_bin <= 4:
    return 1
  return 2


def _multilayer_prefix_stats(
  sequences: Mapping[SequenceKey, SequenceRecord],
  visits: Sequence[SupportVisit],
  predictions: Sequence[MultilayerPredictionRow],
) -> list[dict[str, object]]:
  distinct_layers: dict[SequenceKey, set[int]] = defaultdict(set)
  for visit in visits:
    distinct_layers[visit.key].add(visit.support_layer)
  methods = sorted({str(row["method"]) for row in predictions})
  rows: list[dict[str, object]] = []
  total_sequences = len(sequences)
  for prefix_count in MULTILAYER_PREFIX_COUNTS:
    eligible = sum(len(layers) >= prefix_count for layers in distinct_layers.values())
    for method in methods:
      items = [
        row
        for row in predictions
        if row["method"] == method and row["prefix_distinct_layers"] == prefix_count
      ]
      regression = _regression_stats(
        [row["predicted_depth"] for row in items],
        [row["true_depth"] for row in items],
      )
      three_bin_accuracy = (
        mean(
          _depth_class_from_prediction(row["predicted_depth"])
          == _depth_class_from_bin(row["depth_bin"])
          for row in items
        )
        if items
        else math.nan
      )
      rows.append(
        {
          "prefix_distinct_layers": prefix_count,
          "method": method,
          "total_sequences": total_sequences,
          "eligible_sequences": eligible,
          "predicted_sequences": len(items),
          "availability_all": len(items) / max(total_sequences, 1),
          "availability_eligible": len(items) / max(eligible, 1),
          **regression,
          "three_bin_accuracy": three_bin_accuracy,
        }
      )
  return rows


def _support_episode_paired_rows(
  events: Sequence[EventRecord],
) -> list[dict[str, object]]:
  episodes: dict[tuple[SequenceKey, str], dict[str, EventRecord]] = defaultdict(dict)
  for event in events:
    episode_id = event.raw.get("pair_episode_id", "")
    if event.raw.get("event_family") == "support_trajectory" and episode_id != "":
      episodes[(event.key, episode_id)][event.event_type] = event
  rows: list[dict[str, object]] = []
  for target_type in (
    "pair_peak_quality",
    "pair_stable_for_N_frames",
    "pair_full",
  ):
    pairs: list[tuple[float, float]] = []
    for episode in episodes.values():
      entry = episode.get("pair_enter")
      target = episode.get(target_type)
      if entry is None or target is None:
        continue
      label = _as_float(target.raw, "true_tread_depth")
      entry_depth = _as_float(entry.raw, "pair_depth")
      target_depth = _as_float(
        target.raw,
        "peak_pair_depth" if target_type == "pair_peak_quality" else "pair_depth",
      )
      if all(math.isfinite(value) for value in (label, entry_depth, target_depth)):
        pairs.append((abs(entry_depth - label), abs(target_depth - label)))
    rows.append(
      {
        "target_event": target_type,
        "paired_episodes": len(pairs),
        "entry_mae": mean(pair[0] for pair in pairs) if pairs else math.nan,
        "target_mae": mean(pair[1] for pair in pairs) if pairs else math.nan,
        "mean_mae_change": (
          mean(pair[1] - pair[0] for pair in pairs) if pairs else math.nan
        ),
        "improved_ratio": (
          mean(pair[1] < pair[0] for pair in pairs) if pairs else math.nan
        ),
        "unchanged_ratio": (
          mean(abs(pair[1] - pair[0]) < 1.0e-6 for pair in pairs) if pairs else math.nan
        ),
      }
    )
  return rows


def _contact_association_rows(events: Sequence[EventRecord]) -> list[dict[str, object]]:
  raw = [
    event
    for event in events
    if event.event_type == "oracle_riser_contact"
    and _as_int(event.raw, "contact_valid", 0) == 1
    and event.event_layer >= 2
  ]
  rows: list[dict[str, object]] = []
  for mode, selected in (
    ("raw", raw),
    (
      "association_valid",
      [event for event in raw if _contact_association_valid(event)],
    ),
  ):
    residuals = [
      abs(_as_float(event.raw, "contact_s_minus_riser_s"))
      for event in selected
      if math.isfinite(_as_float(event.raw, "contact_s_minus_riser_s"))
    ]
    by_foot: dict[tuple[SequenceKey, int], list[EventRecord]] = defaultdict(list)
    for event in selected:
      by_foot[(event.key, _as_int(event.raw, "foot_id", -1))].append(event)
    for field, proxy in (
      ("contact_point_s", "contact_point"),
      ("toe_s", "toe"),
      ("root_s", "root"),
    ):
      predictions: list[float] = []
      labels: list[float] = []
      for foot_events in by_foot.values():
        ordered = sorted(
          foot_events,
          key=lambda event: (event.frame_idx, event.time, event.event_id),
        )
        for previous, current in zip(ordered, ordered[1:], strict=False):
          layer_gap = abs(current.event_layer - previous.event_layer)
          previous_s = _as_float(previous.raw, field)
          current_s = _as_float(current.raw, field)
          label = _as_float(current.raw, "true_tread_depth")
          if (
            layer_gap > 0
            and math.isfinite(previous_s)
            and math.isfinite(current_s)
            and math.isfinite(label)
          ):
            predictions.append(abs(current_s - previous_s) / layer_gap)
            labels.append(label)
      rows.append(
        {
          "mode": mode,
          "proxy": proxy,
          "num_events": len(selected),
          "num_sequences": len({event.key for event in selected}),
          "retained_ratio": len(selected) / max(len(raw), 1),
          "association_abs_residual_mean": (mean(residuals) if residuals else math.nan),
          "association_abs_residual_max": max(residuals, default=math.nan),
          "association_s_threshold": (
            CONTACT_ASSOCIATION_MAX_S_RESIDUAL if mode == "association_valid" else ""
          ),
          **_regression_stats(predictions, labels),
        }
      )
  return rows


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


def _quality_bucket(quality: float, scheme: QualityScheme) -> str:
  if not math.isfinite(quality):
    return "missing"
  if quality >= 1.0 - 1.0e-6:
    return "full"
  for index, threshold in enumerate(scheme.partial_thresholds):
    if quality < threshold:
      return f"partial_{index}"
  return f"partial_{len(scheme.partial_thresholds)}"


def _quality_bucket_bounds(bucket: str, scheme: QualityScheme) -> tuple[float, float]:
  if bucket == "full":
    return 1.0, 1.0
  if not bucket.startswith("partial_"):
    return math.nan, math.nan
  index = int(bucket.removeprefix("partial_"))
  lower = 0.0 if index == 0 else scheme.partial_thresholds[index - 1]
  upper = (
    1.0 if index == len(scheme.partial_thresholds) else scheme.partial_thresholds[index]
  )
  return lower, upper


def _candidate_quality_schemes(
  evidence: Sequence[Evidence], calibration_seeds: set[int]
) -> list[QualityScheme]:
  partial_quality = [
    item.quality_min
    for item in evidence
    if item.seed in calibration_seeds
    and item.evidence_type in SUPPORT_EVIDENCE_TYPES
    and math.isfinite(item.quality_min)
    and item.quality_min < 1.0 - 1.0e-6
  ]
  quantile_thresholds = tuple(
    sorted(
      {
        _quantile(partial_quality, 1.0 / 3.0),
        _quantile(partial_quality, 2.0 / 3.0),
      }
    )
  )
  quantile_thresholds = tuple(
    value for value in quantile_thresholds if math.isfinite(value) and 0.0 < value < 1.0
  )
  return [
    QualityScheme("coarse_075", (0.75,)),
    QualityScheme("tiered_060_085", (0.60, 0.85)),
    QualityScheme("calibration_tertiles", quantile_thresholds),
  ]


def _calibrate_quality_buckets(
  evidence: Sequence[Evidence],
  calibration_seeds: set[int],
  scheme: QualityScheme,
) -> dict[str, Calibration]:
  grouped: dict[str, list[float]] = defaultdict(list)
  for item in evidence:
    if (
      item.seed in calibration_seeds
      and item.evidence_type in SUPPORT_EVIDENCE_TYPES
      and math.isfinite(item.quality_min)
    ):
      grouped[_quality_bucket(item.quality_min, scheme)].append(
        item.true_depth - item.center
      )
  output: dict[str, Calibration] = {}
  for bucket, residuals in sorted(grouped.items()):
    output[bucket] = Calibration(
      evidence_type=f"{scheme.name}:{bucket}",
      sample_count=len(residuals),
      q05=_quantile(residuals, 0.05),
      q50=_quantile(residuals, 0.50),
      q95=_quantile(residuals, 0.95),
      residual_mean=mean(residuals),
      residual_std=pstdev(residuals),
    )
  return output


def _quality_binned_intervals(
  evidence: Sequence[Evidence],
  scheme: QualityScheme,
  calibrations: Mapping[str, Calibration],
) -> list[IntervalEvidence]:
  output: list[IntervalEvidence] = []
  for item in evidence:
    if item.evidence_type not in SUPPORT_EVIDENCE_TYPES or not math.isfinite(
      item.quality_min
    ):
      continue
    calibration = calibrations.get(_quality_bucket(item.quality_min, scheme))
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


def _quality_calibration_samples(
  evidence: Sequence[Evidence], calibration_seeds: set[int]
) -> list[tuple[float, float]]:
  return [
    (item.quality_min, item.true_depth - item.center)
    for item in evidence
    if item.seed in calibration_seeds
    and item.evidence_type in SUPPORT_EVIDENCE_TYPES
    and math.isfinite(item.quality_min)
  ]


def _quality_aware_interval(
  item: Evidence,
  calibration_samples: Sequence[tuple[float, float]],
  neighbor_count: int,
  *,
  quality_override: float | None = None,
) -> IntervalEvidence | None:
  quality = item.quality_min if quality_override is None else quality_override
  if not math.isfinite(quality) or not calibration_samples:
    return None
  nearest = sorted(
    calibration_samples,
    key=lambda sample: (abs(sample[0] - quality), sample[0], sample[1]),
  )[: min(neighbor_count, len(calibration_samples))]
  residuals = [sample[1] for sample in nearest]
  return IntervalEvidence(
    evidence=item,
    lower=item.center + _quantile(residuals, 0.05),
    upper=item.center + _quantile(residuals, 0.95),
  )


def _quality_aware_intervals(
  evidence: Sequence[Evidence],
  calibration_samples: Sequence[tuple[float, float]],
  neighbor_count: int,
  quality_overrides: Mapping[tuple[SequenceKey, int], float] | None = None,
) -> list[IntervalEvidence]:
  output: list[IntervalEvidence] = []
  for item in evidence:
    if item.evidence_type not in SUPPORT_EVIDENCE_TYPES:
      continue
    override = (
      quality_overrides.get((item.key, item.event_id))
      if quality_overrides is not None
      else None
    )
    interval = _quality_aware_interval(
      item,
      calibration_samples,
      neighbor_count,
      quality_override=override,
    )
    if interval is not None:
      output.append(interval)
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


def _intervals_by_key(
  intervals: Sequence[IntervalEvidence],
) -> dict[SequenceKey, list[IntervalEvidence]]:
  grouped: dict[SequenceKey, list[IntervalEvidence]] = defaultdict(list)
  for item in intervals:
    grouped[item.evidence.key].append(item)
  return grouped


def _select_quality_scheme(
  evidence: Sequence[Evidence],
  calibration_seeds: set[int],
  selection_sequences: Sequence[SequenceRecord],
  prior: tuple[float, float],
  fallback_policy: str,
  thresholds: tuple[float, float],
  outside_penalty: float,
) -> tuple[
  QualityScheme,
  dict[str, dict[str, Calibration]],
  list[dict[str, object]],
]:
  schemes = _candidate_quality_schemes(evidence, calibration_seeds)
  calibrations_by_scheme: dict[str, dict[str, Calibration]] = {}
  rows: list[dict[str, object]] = []
  for scheme in schemes:
    calibrations = _calibrate_quality_buckets(evidence, calibration_seeds, scheme)
    calibrations_by_scheme[scheme.name] = calibrations
    intervals = _quality_binned_intervals(evidence, scheme, calibrations)
    predictions, _ = _predict_method(
      selection_sequences,
      _intervals_by_key(intervals),
      "quality_binned",
      SUPPORT_EVIDENCE_TYPES,
      prior,
      fallback_policy,
    )
    rows.append(
      {
        "selection_type": "bucket_scheme",
        "candidate": scheme.name,
        **_prediction_stats(predictions, thresholds, outside_penalty),
      }
    )

  def score(row: Mapping[str, object]) -> float:
    value = row["interval_score"]
    if not isinstance(value, (int, float)):
      raise TypeError("interval_score must be numeric.")
    return float(value)

  selected_name = min(rows, key=score)["candidate"]
  selected_scheme = next(scheme for scheme in schemes if scheme.name == selected_name)
  for row in rows:
    row["selected"] = int(row["candidate"] == selected_scheme.name)
  return selected_scheme, calibrations_by_scheme, rows


def _select_quality_neighbors(
  evidence: Sequence[Evidence],
  calibration_seeds: set[int],
  selection_sequences: Sequence[SequenceRecord],
  prior: tuple[float, float],
  fallback_policy: str,
  thresholds: tuple[float, float],
  outside_penalty: float,
) -> tuple[int, list[tuple[float, float]], list[dict[str, object]]]:
  calibration_samples = _quality_calibration_samples(evidence, calibration_seeds)
  candidates = (128, 256, 512)
  rows: list[dict[str, object]] = []
  for neighbor_count in candidates:
    intervals = _quality_aware_intervals(evidence, calibration_samples, neighbor_count)
    predictions, _ = _predict_method(
      selection_sequences,
      _intervals_by_key(intervals),
      "quality_aware",
      SUPPORT_EVIDENCE_TYPES,
      prior,
      fallback_policy,
    )
    rows.append(
      {
        "selection_type": "neighbor_count",
        "candidate": neighbor_count,
        **_prediction_stats(predictions, thresholds, outside_penalty),
      }
    )

  def numeric(row: Mapping[str, object], key: str) -> float:
    value = row[key]
    if not isinstance(value, (int, float)):
      raise TypeError(f"{key} must be numeric.")
    return float(value)

  def score(row: Mapping[str, object]) -> tuple[float, int]:
    value = row["interval_score"]
    candidate = row["candidate"]
    if not isinstance(value, (int, float)) or not isinstance(candidate, int):
      raise TypeError("Quality-neighbor selection values must be numeric.")
    return float(value), candidate

  eligible = [row for row in rows if numeric(row, "coverage") >= MIN_SELECTION_COVERAGE]
  if eligible:
    selected_row = min(eligible, key=score)
    selection_reason = "minimum_interval_score_with_coverage_constraint"
  else:
    selected_row = min(
      rows,
      key=lambda row: (
        -numeric(row, "coverage"),
        numeric(row, "interval_score"),
      ),
    )
    selection_reason = "coverage_constraint_unmet_selected_highest_coverage"
  selected_value = selected_row["candidate"]
  if not isinstance(selected_value, int):
    raise TypeError("Selected quality neighbor count must be an integer.")
  selected = selected_value
  for row in rows:
    row["selected"] = int(row["candidate"] == selected)
    row["coverage_constraint_met"] = int(
      numeric(row, "coverage") >= MIN_SELECTION_COVERAGE
    )
    row["selection_reason"] = selection_reason if row["candidate"] == selected else ""
  return selected, calibration_samples, rows


def _shuffled_quality_overrides(
  evidence: Sequence[Evidence],
  seeds: set[int],
  random_seed: int,
) -> dict[tuple[SequenceKey, int], float]:
  items = [
    item
    for item in evidence
    if item.seed in seeds
    and item.evidence_type in SUPPORT_EVIDENCE_TYPES
    and math.isfinite(item.quality_min)
  ]
  items.sort(key=lambda item: (item.key, item.frame_idx, item.event_id))
  if len(items) <= 1:
    return {(item.key, item.event_id): item.quality_min for item in items}
  offset = random.Random(random_seed).randrange(1, len(items))
  shifted = items[offset:] + items[:offset]
  return {
    (item.key, item.event_id): donor.quality_min
    for item, donor in zip(items, shifted, strict=True)
  }


def _sequence_quality(
  evidence: Sequence[Evidence], seeds: set[int]
) -> dict[SequenceKey, float]:
  quality: dict[SequenceKey, float] = {}
  for item in evidence:
    if (
      item.seed in seeds
      and item.evidence_type in SUPPORT_EVIDENCE_TYPES
      and math.isfinite(item.quality_min)
    ):
      quality[item.key] = max(quality.get(item.key, 0.0), item.quality_min)
  return quality


def _conditional_calibration_error(
  predictions: Sequence[Prediction],
  quality_by_key: Mapping[SequenceKey, float],
  scheme: QualityScheme,
  target_coverage: float = 0.90,
) -> float:
  grouped: dict[str, list[Prediction]] = defaultdict(list)
  for prediction in predictions:
    quality = quality_by_key.get(prediction.key)
    if quality is not None:
      grouped[_quality_bucket(quality, scheme)].append(prediction)
  total = sum(len(items) for items in grouped.values())
  if total == 0:
    return math.nan
  return sum(
    len(items) / total * abs(mean(item.covered for item in items) - target_coverage)
    for items in grouped.values()
  )


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
    "trajectory_first": {"adjacent_support_first_v2"},
    "trajectory_stable": {"adjacent_support_stable_v2"},
    "trajectory_full": {"adjacent_support_full_v2"},
    "trajectory_peak": {"adjacent_support_peak_v2"},
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


def _support_quality_stats_rows(
  evidence: Sequence[Evidence],
  scheme: QualityScheme,
  calibration_seeds: set[int],
  selection_seeds: set[int],
  test_seeds: set[int],
) -> list[dict[str, object]]:
  rows: list[dict[str, object]] = []
  for split in ("calibration", "selection", "test"):
    split_items = [
      item
      for item in evidence
      if item.evidence_type in SUPPORT_EVIDENCE_TYPES
      and math.isfinite(item.quality_min)
      and _split_name(item.seed, calibration_seeds, selection_seeds, test_seeds)
      == split
    ]
    buckets = sorted(
      {_quality_bucket(item.quality_min, scheme) for item in split_items}
    )
    for bucket in ["all", *buckets]:
      items = (
        split_items
        if bucket == "all"
        else [
          item
          for item in split_items
          if _quality_bucket(item.quality_min, scheme) == bucket
        ]
      )
      stats = _regression_stats(
        [item.center for item in items], [item.true_depth for item in items]
      )
      residual_abs = [abs(item.center - item.true_depth) for item in items]
      lower, upper = (
        (math.nan, math.nan)
        if bucket == "all"
        else _quality_bucket_bounds(bucket, scheme)
      )
      rows.append(
        {
          "split": split,
          "selected_scheme": scheme.name,
          "quality_bucket": bucket,
          "quality_lower": lower,
          "quality_upper": upper,
          "num_sequences": len({item.key for item in items}),
          **stats,
          "mean_absolute_residual": (mean(residual_abs) if residual_abs else math.nan),
          "pair_quality_min_mean": (
            mean(item.quality_min for item in items) if items else math.nan
          ),
          "pair_quality_mean_mean": (
            mean(item.quality_mean for item in items) if items else math.nan
          ),
          "pair_quality_max_mean": (
            mean(item.quality_max for item in items) if items else math.nan
          ),
          "pair_quality_asymmetry_mean": (
            mean(item.quality_asymmetry for item in items) if items else math.nan
          ),
        }
      )
  return rows


def _support_quality_calibration_rows(
  schemes: Sequence[QualityScheme],
  calibrations_by_scheme: Mapping[str, Mapping[str, Calibration]],
  selected_scheme: QualityScheme,
  bucket_selection_rows: Sequence[Mapping[str, object]],
  neighbor_selection_rows: Sequence[Mapping[str, object]],
  selected_neighbors: int,
  calibration_seeds: set[int],
) -> list[dict[str, object]]:
  rows: list[dict[str, object]] = []
  seed_text = "|".join(map(str, sorted(calibration_seeds)))
  for scheme in schemes:
    for bucket, calibration in sorted(calibrations_by_scheme[scheme.name].items()):
      lower, upper = _quality_bucket_bounds(bucket, scheme)
      rows.append(
        {
          "row_type": "bucket_calibration",
          "calibration_seeds": seed_text,
          "scheme": scheme.name,
          "selected": int(scheme.name == selected_scheme.name),
          "bucket": bucket,
          "quality_lower": lower,
          "quality_upper": upper,
          "sample_count": calibration.sample_count,
          "q05": calibration.q05,
          "q50": calibration.q50,
          "q95": calibration.q95,
          "mean": calibration.residual_mean,
          "std": calibration.residual_std,
        }
      )
  for row in bucket_selection_rows:
    rows.append(
      {
        "row_type": "bucket_selection",
        "calibration_seeds": seed_text,
        "scheme": row["candidate"],
        "selected": row["selected"],
        "interval_score": row["interval_score"],
        "coverage": row["coverage"],
        "mean_width": row["mean_width"],
      }
    )
  for row in neighbor_selection_rows:
    rows.append(
      {
        "row_type": "neighbor_selection",
        "calibration_seeds": seed_text,
        "neighbor_count": row["candidate"],
        "selected": int(row["candidate"] == selected_neighbors),
        "interval_score": row["interval_score"],
        "coverage": row["coverage"],
        "mean_width": row["mean_width"],
        "coverage_constraint_met": row["coverage_constraint_met"],
        "selection_reason": row["selection_reason"],
      }
    )
  return rows


def _support_quality_per_bin_rows(
  predictions_by_method: Mapping[str, Sequence[Prediction]],
  quality_by_key: Mapping[SequenceKey, float],
  scheme: QualityScheme,
  thresholds: tuple[float, float],
  outside_penalty: float,
) -> list[dict[str, object]]:
  rows: list[dict[str, object]] = []
  quality_buckets = sorted(
    {_quality_bucket(quality, scheme) for quality in quality_by_key.values()}
  )
  for method, predictions in predictions_by_method.items():
    for depth_bin in range(8):
      subset = [
        prediction for prediction in predictions if prediction.depth_bin == depth_bin
      ]
      rows.append(
        {
          "group_type": "depth_bin",
          "method": method,
          "group": depth_bin,
          **_prediction_stats(subset, thresholds, outside_penalty),
        }
      )
    for bucket in quality_buckets:
      subset = [
        prediction
        for prediction in predictions
        if prediction.key in quality_by_key
        and _quality_bucket(quality_by_key[prediction.key], scheme) == bucket
      ]
      lower, upper = _quality_bucket_bounds(bucket, scheme)
      rows.append(
        {
          "group_type": "quality_bucket",
          "method": method,
          "group": bucket,
          "quality_lower": lower,
          "quality_upper": upper,
          **_prediction_stats(subset, thresholds, outside_penalty),
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

  selected_quality_scheme, quality_calibrations, bucket_selection_rows = (
    _select_quality_scheme(
      evidence,
      calibration_seeds,
      selection_sequences,
      prior,
      fallback_policy,
      thresholds,
      outside_penalty,
    )
  )
  selected_neighbors, quality_samples, neighbor_selection_rows = (
    _select_quality_neighbors(
      evidence,
      calibration_seeds,
      selection_sequences,
      prior,
      fallback_policy,
      thresholds,
      outside_penalty,
    )
  )
  audit_rows.extend(
    [
      {
        "metric": "selected_support_quality_scheme",
        "actual": selected_quality_scheme.name,
        "expected": "",
        "status": "selected_on_seed43",
      },
      {
        "metric": "selected_support_quality_neighbors",
        "actual": selected_neighbors,
        "expected": "",
        "status": "selected_on_seed43",
      },
    ]
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
    "support_first_v2": {"adjacent_support_first_v2"},
    "support_stable_v2": {"adjacent_support_stable_v2"},
    "support_full_v2": {"adjacent_support_full_v2"},
    "support_peak_v2": {"adjacent_support_peak_v2"},
    "detector_only": DETECTOR_TYPES,
    "oracle_contact_proxy": ORACLE_TYPES,
    "oracle_contact_point_v2": {"oracle_contact_point_proxy_v2"},
    "oracle_toe_v2": {"oracle_toe_proxy_v2"},
    "oracle_root_v2": {"oracle_root_proxy_v2"},
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

  binary_support_intervals = [
    item for item in calibrated if item.evidence.evidence_type in SUPPORT_EVIDENCE_TYPES
  ]
  full_only_intervals = [
    item
    for item in binary_support_intervals
    if math.isfinite(item.evidence.quality_min)
    and item.evidence.quality_min >= 1.0 - 1.0e-6
  ]
  quality_binned_intervals = _quality_binned_intervals(
    evidence,
    selected_quality_scheme,
    quality_calibrations[selected_quality_scheme.name],
  )
  quality_aware_intervals = _quality_aware_intervals(
    evidence, quality_samples, selected_neighbors
  )
  shuffled_quality_overrides = _shuffled_quality_overrides(
    evidence, test_seeds, shuffle_seed
  )
  shuffled_quality_intervals = _quality_aware_intervals(
    evidence,
    quality_samples,
    selected_neighbors,
    shuffled_quality_overrides,
  )
  quality_interval_sets = {
    "full_only": full_only_intervals,
    "binary_partial_full": binary_support_intervals,
    "quality_binned": quality_binned_intervals,
    "quality_aware": quality_aware_intervals,
    "shuffled_quality": shuffled_quality_intervals,
  }
  quality_predictions: dict[str, list[Prediction]] = {}
  for method, intervals in quality_interval_sets.items():
    predictions, _ = _predict_method(
      test_sequences,
      _intervals_by_key(intervals),
      method,
      SUPPORT_EVIDENCE_TYPES,
      prior,
      fallback_policy,
    )
    quality_predictions[method] = predictions

  quality_by_key = _sequence_quality(evidence, test_seeds)
  quality_summary_rows = []
  for method, predictions in quality_predictions.items():
    quality_summary_rows.append(
      {
        "split": "test",
        "test_seeds": "|".join(map(str, sorted(test_seeds))),
        "method": method,
        "selected_bucket_scheme": selected_quality_scheme.name,
        "selected_neighbor_count": selected_neighbors,
        "geometric_overlap_is_privileged": True,
        **_prediction_stats(predictions, thresholds, outside_penalty),
        "conditional_calibration_error": _conditional_calibration_error(
          predictions, quality_by_key, selected_quality_scheme
        ),
      }
    )
  _write_csv(
    output / "support_quality_interval_summary.csv",
    quality_summary_rows,
  )
  _write_csv(
    output / "support_quality_stats.csv",
    _support_quality_stats_rows(
      evidence,
      selected_quality_scheme,
      calibration_seeds,
      selection_seeds,
      test_seeds,
    ),
  )
  _write_csv(
    output / "support_quality_per_bin_stats.csv",
    _support_quality_per_bin_rows(
      quality_predictions,
      quality_by_key,
      selected_quality_scheme,
      thresholds,
      outside_penalty,
    ),
  )
  _write_csv(
    output / "support_quality_calibration.csv",
    _support_quality_calibration_rows(
      _candidate_quality_schemes(evidence, calibration_seeds),
      quality_calibrations,
      selected_quality_scheme,
      bucket_selection_rows,
      neighbor_selection_rows,
      selected_neighbors,
      calibration_seeds,
    ),
  )

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
    in ORACLE_TYPES | {"detector_confirmation_proxy", "additional_layer_confirmation"}
  ]
  _write_csv(output / "confirmation_displacement_stats.csv", confirmation_rows)
  _write_csv(output / "riser_displacement_stats.csv", confirmation_rows)
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


def analyze_descriptive(
  input_dirs: Sequence[Path],
  output_dir: Path,
  *,
  include_process_end: bool = False,
) -> None:
  """Write uncalibrated single-seed Stage 1.2 evidence comparisons."""
  output = output_dir.expanduser().resolve()
  output.mkdir(parents=True, exist_ok=True)
  sequences, events, excluded = _load_inputs(input_dirs, include_process_end)
  additional, duplicates = _reclassify_confirmation_events(events)
  evidence = _extract_evidence(sequences, events, additional)
  support_visits = _support_visits(sequences, events)
  multilayer_predictions = _multilayer_prediction_rows(sequences, support_visits)
  multilayer_stats = _multilayer_prefix_stats(
    sequences,
    support_visits,
    multilayer_predictions,
  )
  seeds = {sequence.seed for sequence in sequences.values()}
  single_event_rows = _single_event_rows(evidence, [], seeds, set(), set())
  descriptive_rows = [
    {**row, "split": "descriptive"}
    for row in single_event_rows
    if row["split"] == "calibration"
  ]
  pair_rows = _pair_support_rows(evidence, seeds, set(), set())
  pair_rows = [
    {**row, "split": "descriptive"}
    for row in pair_rows
    if row["split"] == "calibration"
  ]
  riser_rows = [
    row
    for row in descriptive_rows
    if row["evidence_type"]
    in ORACLE_TYPES | {"detector_confirmation_proxy", "additional_layer_confirmation"}
  ]
  event_counts = Counter(
    (event.raw.get("event_family", ""), event.event_type) for event in events
  )
  event_count_rows = [
    {
      "event_family": family,
      "event_type": event_type,
      "num_events": count,
      "num_sequences": len(
        {
          event.key
          for event in events
          if event.raw.get("event_family", "") == family
          and event.event_type == event_type
        }
      ),
    }
    for (family, event_type), count in sorted(event_counts.items())
  ]
  _write_csv(output / "single_event_stats.csv", descriptive_rows)
  _write_csv(output / "pair_support_stats.csv", pair_rows)
  _write_csv(output / "riser_displacement_stats.csv", riser_rows)
  _write_csv(output / "logger_v2_event_counts.csv", event_count_rows)
  _write_csv(
    output / "support_visit_representatives.csv",
    [
      {
        "run_id": visit.key[0],
        "seed": visit.seed,
        "sequence_id": visit.sequence_id,
        "foot_id": visit.foot_id,
        "support_layer": visit.support_layer,
        "frame_idx": visit.frame_idx,
        "event_id": visit.event_id,
        "selected_event_type": visit.event_type,
        "foot_s": visit.foot_s,
        "quality": visit.quality,
        "true_depth": visit.true_depth,
        "depth_bin": visit.depth_bin,
      }
      for visit in support_visits
    ],
  )
  _write_csv(
    output / "support_episode_paired_stats.csv",
    _support_episode_paired_rows(events),
  )
  _write_csv(
    output / "contact_association_stats.csv",
    _contact_association_rows(events),
  )
  _write_csv(
    output / "multilayer_support_predictions.csv",
    multilayer_predictions,
  )
  _write_csv(
    output / "multilayer_support_prefix_stats.csv",
    multilayer_stats,
  )
  _write_csv(
    output / "data_audit_summary.csv",
    [
      {"metric": "sequences", "actual": len(sequences)},
      {"metric": "events", "actual": len(events)},
      {"metric": "evidence", "actual": len(evidence)},
      {"metric": "independent_support_visits", "actual": len(support_visits)},
      {
        "metric": "multilayer_predictions",
        "actual": len(multilayer_predictions),
      },
      {"metric": "excluded_process_end", "actual": excluded},
      {"metric": "duplicate_confirmations", "actual": len(duplicates)},
      {
        "metric": "logger_v2_available",
        "actual": any(event.raw.get("logger_version") == "2" for event in events),
      },
    ],
  )
  print(
    "[Stage 1.2] Descriptive analysis complete:",
    f"sequences={len(sequences)}",
    f"evidence={len(evidence)}",
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
  parser.add_argument("--descriptive-only", action="store_true")
  parser.add_argument("--skip-expected-count-check", action="store_true")
  parser.add_argument("--outside-penalty", type=float, default=2.0)
  parser.add_argument("--shuffle-seed", type=int, default=17)
  args = parser.parse_args()
  if args.descriptive_only:
    analyze_descriptive(
      input_dirs=args.input_dirs,
      output_dir=args.output_dir,
      include_process_end=args.include_process_end,
    )
    return
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
