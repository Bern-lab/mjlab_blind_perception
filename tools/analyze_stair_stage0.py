"""Analyze Stage 0 stair sequence/event exports.

This script is intentionally offline-only. It does not change rewards, policy,
network state, or training behavior. It reads one or more directories produced by
scripts/velocity_eval/export_stair_sequences.py and summarizes whether Stage 0
confirmation logging is reliable enough to enter Stage 1.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class LoadedRows:
  run: str
  sequences: list[dict[str, str]]
  events: list[dict[str, str]]


def _read_csv(path: Path) -> list[dict[str, str]]:
  if not path.exists():
    raise FileNotFoundError(path)
  with path.open(encoding="utf-8", newline="") as stream:
    return list(csv.DictReader(stream))


def _as_int(row: dict[str, str], key: str, default: int = 0) -> int:
  value = row.get(key, "")
  if value in ("", None):
    return default
  try:
    return int(float(value))
  except ValueError:
    return default


def _as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
  value = row.get(key, "")
  if value in ("", None):
    return default
  try:
    return float(value)
  except ValueError:
    return default


def _ratio(numerator: float, denominator: float) -> float:
  return numerator / denominator if denominator > 0 else math.nan


def _mean(values: Iterable[float]) -> float:
  clean = [value for value in values if not math.isnan(value)]
  return mean(clean) if clean else math.nan


def _load_inputs(input_dirs: list[Path], include_process_end: bool) -> list[LoadedRows]:
  loaded: list[LoadedRows] = []
  for directory in input_dirs:
    directory = directory.expanduser().resolve()
    sequences = _read_csv(directory / "stair_sequences.csv")
    events = _read_csv(directory / "stair_events.csv")
    if not include_process_end:
      valid_ids = {
        row["sequence_id"]
        for row in sequences
        if row.get("termination_reason") != "process_end"
      }
      sequences = [row for row in sequences if row.get("sequence_id") in valid_ids]
      events = [row for row in events if row.get("sequence_id") in valid_ids]
    for row in sequences:
      row["_run"] = directory.name
      row["_source_dir"] = str(directory)
    for row in events:
      row["_run"] = directory.name
      row["_source_dir"] = str(directory)
    loaded.append(LoadedRows(run=directory.name, sequences=sequences, events=events))
  return loaded


def _sequence_key(row: dict[str, str]) -> tuple[str, str]:
  return (row.get("_source_dir", ""), row.get("sequence_id", ""))


def _metrics(
  sequences: list[dict[str, str]],
  events: list[dict[str, str]],
) -> dict[str, float | int | str]:
  confirmation_events = [e for e in events if e.get("event_type") == "confirmation"]
  duplicate_events = [
    e for e in events if e.get("event_type") == "confirmation_duplicate"
  ]
  new_layer_events = [
    e for e in events if e.get("event_type") == "confirmation_new_layer"
  ]
  oracle_sequences = [
    s for s in sequences if _as_int(s, "has_layer2plus_riser_collision") == 1
  ]
  correct_sequences = [
    s for s in oracle_sequences if _as_int(s, "confirmation_correct") == 1
  ]
  missed_sequences = [
    s for s in oracle_sequences if _as_int(s, "confirmation_correct") != 1
  ]

  confirmation_count = len(confirmation_events)
  correct_confirmations = sum(
    _as_int(e, "confirmation_correct") for e in confirmation_events
  )
  layer1_false = sum(
    "layer1" in e.get("confirmation_error_type", "").split("|")
    for e in confirmation_events
  )
  wrong_layer = sum(
    "wrong_layer" in e.get("confirmation_error_type", "").split("|")
    for e in confirmation_events
  )
  wrong_sequence = sum(
    "wrong_sequence" in e.get("confirmation_error_type", "").split("|")
    for e in confirmation_events
  )
  flat_false = sum(
    "flat" in e.get("confirmation_error_type", "").split("|")
    for e in confirmation_events
  )
  delays = [_as_float(s, "confirmation_delay") for s in correct_sequences]
  same_contact = sum(
    e.get("duplicate_type") == "same_contact_duplicate" for e in duplicate_events
  )
  same_layer_rehit = sum(
    e.get("duplicate_type") == "same_layer_rehit" for e in duplicate_events
  )
  later_layer = sum(
    e.get("duplicate_type") == "later_layer_confirmation" for e in duplicate_events
  )
  unknown_duplicate = sum(
    e.get("duplicate_type") == "unknown_duplicate" for e in duplicate_events
  )
  true_duplicates = same_contact + same_layer_rehit
  additional_layers = len(new_layer_events) + later_layer

  return {
    "sequences": len(sequences),
    "oracle_sequences": len(oracle_sequences),
    "confirmation_events": confirmation_count,
    "correct_confirmations": correct_confirmations,
    "precision": _ratio(correct_confirmations, confirmation_count),
    "recall": _ratio(len(correct_sequences), len(oracle_sequences)),
    "missed": len(missed_sequences),
    "missed_ratio": _ratio(len(missed_sequences), len(oracle_sequences)),
    "layer1_false": layer1_false,
    "layer1_false_ratio": _ratio(layer1_false, confirmation_count),
    "wrong_layer": wrong_layer,
    "wrong_layer_ratio": _ratio(wrong_layer, confirmation_count),
    "wrong_sequence": wrong_sequence,
    "wrong_sequence_ratio": _ratio(wrong_sequence, confirmation_count),
    "flat_false": flat_false,
    "flat_false_ratio": _ratio(flat_false, confirmation_count),
    "detector_retrigger_events": len(duplicate_events) + len(new_layer_events),
    "detector_retrigger_ratio": _ratio(
      len(duplicate_events) + len(new_layer_events), confirmation_count
    ),
    "duplicates": true_duplicates,
    "duplicate_ratio": _ratio(true_duplicates, confirmation_count),
    "same_contact_duplicate": same_contact,
    "same_contact_duplicate_ratio": _ratio(same_contact, confirmation_count),
    "same_layer_rehit": same_layer_rehit,
    "same_layer_rehit_ratio": _ratio(same_layer_rehit, confirmation_count),
    "additional_layer_confirmation": additional_layers,
    "additional_layer_confirmation_ratio": _ratio(
      additional_layers, confirmation_count
    ),
    "unknown_duplicate": unknown_duplicate,
    "unknown_duplicate_ratio": _ratio(unknown_duplicate, confirmation_count),
    "delay_mean": _mean(delays),
  }


def _classify_duplicates(events: list[dict[str, str]]) -> list[dict[str, str]]:
  by_sequence: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
  for event in events:
    by_sequence[_sequence_key(event)].append(event)

  duplicates: list[dict[str, str]] = []
  for key_events in by_sequence.values():
    key_events.sort(key=lambda e: (_as_int(e, "frame_idx"), _as_int(e, "event_id")))
    first_confirmation = next(
      (event for event in key_events if event.get("event_type") == "confirmation"),
      None,
    )
    for event in key_events:
      if event.get("event_type") != "confirmation_duplicate":
        continue
      row = dict(event)
      if first_confirmation is None:
        row["duplicate_type"] = "unknown_duplicate"
        row["first_confirmation_frame"] = ""
        row["frame_delta"] = ""
        row["first_confirmation_layer"] = ""
      else:
        first_layer = _as_int(first_confirmation, "event_layer", -999)
        duplicate_layer = _as_int(event, "event_layer", -998)
        first_frame = _as_int(first_confirmation, "frame_idx", -1)
        duplicate_frame = _as_int(event, "frame_idx", -1)
        frame_delta = duplicate_frame - first_frame
        row["first_confirmation_frame"] = str(first_frame)
        row["frame_delta"] = str(frame_delta)
        row["first_confirmation_layer"] = str(first_layer)
        if duplicate_layer != first_layer:
          row["duplicate_type"] = "later_layer_confirmation"
        elif frame_delta <= 3:
          row["duplicate_type"] = "same_contact_duplicate"
        else:
          row["duplicate_type"] = "same_layer_rehit"
      duplicates.append(row)
  return duplicates


def _missed_sequences(sequences: list[dict[str, str]]) -> list[dict[str, str]]:
  return [
    row
    for row in sequences
    if _as_int(row, "has_layer2plus_riser_collision") == 1
    and _as_int(row, "confirmation_correct") != 1
  ]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
  if rows:
    fieldnames: list[str] = []
    for row in rows:
      for key in row:
        if key not in fieldnames:
          fieldnames.append(key)
  else:
    fieldnames = ["empty"]
  with path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def _group_summary(
  rows: list[dict[str, str]],
  events: list[dict[str, str]],
  group_type: str,
  key: str,
) -> list[dict[str, object]]:
  values = sorted({row.get(key, "") for row in rows})
  output: list[dict[str, object]] = []
  for value in values:
    group_sequences = [row for row in rows if row.get(key, "") == value]
    group_ids = {_sequence_key(row) for row in group_sequences}
    group_events = [event for event in events if _sequence_key(event) in group_ids]
    metric = _metrics(group_sequences, group_events)
    output.append({"group_type": group_type, "group": value, **metric})
  return output


def analyze(
  input_dirs: list[Path], output_dir: Path, include_process_end: bool
) -> None:
  loaded = _load_inputs(input_dirs, include_process_end=include_process_end)
  all_sequences = [row for group in loaded for row in group.sequences]
  all_events = [row for group in loaded for row in group.events]

  duplicate_rows = _classify_duplicates(all_events)
  duplicate_lookup = {
    (
      row.get("_source_dir", ""),
      row.get("sequence_id", ""),
      row.get("event_id", ""),
    ): row.get("duplicate_type", "unknown_duplicate")
    for row in duplicate_rows
  }
  for event in all_events:
    if event.get("event_type") == "confirmation_duplicate":
      event["duplicate_type"] = duplicate_lookup.get(
        (
          event.get("_source_dir", ""),
          event.get("sequence_id", ""),
          event.get("event_id", ""),
        ),
        "unknown_duplicate",
      )

  output_dir = output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  summary_rows: list[dict[str, object]] = [
    {"group_type": "overall", "group": "all", **_metrics(all_sequences, all_events)}
  ]
  summary_rows.extend(_group_summary(all_sequences, all_events, "run", "_run"))
  summary_rows.extend(
    _group_summary(all_sequences, all_events, "depth_bin", "depth_bin")
  )

  confirmation_events = [
    event for event in all_events if event.get("event_type") == "confirmation"
  ]
  for layer in sorted({event.get("event_layer", "") for event in confirmation_events}):
    layer_events = [
      event
      for event in all_events
      if event.get("event_type") in {"confirmation", "confirmation_duplicate"}
      and event.get("event_layer", "") == layer
    ]
    layer_ids = {_sequence_key(event) for event in layer_events}
    layer_sequences = [row for row in all_sequences if _sequence_key(row) in layer_ids]
    summary_rows.append(
      {
        "group_type": "confirmation_layer",
        "group": layer,
        **_metrics(layer_sequences, layer_events),
      }
    )

  missed_rows = _missed_sequences(all_sequences)
  _write_csv(output_dir / "summary.csv", summary_rows)
  _write_csv(output_dir / "detector_retrigger_events.csv", duplicate_rows)
  _write_csv(
    output_dir / "duplicate_events.csv",
    [
      row
      for row in duplicate_rows
      if row.get("duplicate_type")
      in {"same_contact_duplicate", "same_layer_rehit", "unknown_duplicate"}
    ],
  )
  _write_csv(
    output_dir / "additional_layer_events.csv",
    [
      row
      for row in duplicate_rows
      if row.get("duplicate_type") == "later_layer_confirmation"
    ],
  )
  _write_csv(output_dir / "missed_events.csv", missed_rows)

  overall = summary_rows[0]
  print("[Stage 0] Summary written:", output_dir)
  print(
    "[Stage 0] Overall:",
    f"sequences={overall['sequences']}",
    f"oracle={overall['oracle_sequences']}",
    f"confirmations={overall['confirmation_events']}",
    f"precision={overall['precision']}",
    f"recall={overall['recall']}",
    f"same_contact_duplicate_ratio={overall['same_contact_duplicate_ratio']}",
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input-dirs",
    nargs="+",
    required=True,
    type=Path,
    help="One or more Stage 0 export directories.",
  )
  parser.add_argument(
    "--output-dir",
    required=True,
    type=Path,
    help="Directory where summary CSV files are written.",
  )
  parser.add_argument(
    "--include-process-end",
    action="store_true",
    help="Include sequences finalized only because the exporter process ended.",
  )
  args = parser.parse_args()
  analyze(
    input_dirs=args.input_dirs,
    output_dir=args.output_dir,
    include_process_end=args.include_process_end,
  )


if __name__ == "__main__":
  main()
