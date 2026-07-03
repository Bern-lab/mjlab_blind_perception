"""Sequence- and event-level logging for stair interaction diagnostics."""

from __future__ import annotations

import atexit
import csv
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from .stair_geometry import (
  STAIR_ADJACENT_PAIR_DEPTH_KEY,
  STAIR_ADJACENT_PAIR_HEIGHT_KEY,
  STAIR_ADJACENT_PAIR_VALID_KEY,
  STAIR_ASCENT_DIR_KEY,
  STAIR_CONFIRMATION_CONDITION_KEY,
  STAIR_CONFIRMATION_CONTACT_SEQUENCE_KEY,
  STAIR_CONFIRMATION_LAYER_KEY,
  STAIR_CURRENT_GROUND_CONTACT_KEY,
  STAIR_CURRENT_STAIR_SUPPORT_KEY,
  STAIR_CURRENT_SUPPORT_FRACTION_KEY,
  STAIR_CURRENT_SUPPORT_LAYER_KEY,
  STAIR_DEPTH_CONFIRMATION_EVENT_KEY,
  STAIR_EXIT_EVENT_KEY,
  STAIR_EXPECTED_LAYER_KEY,
  STAIR_GEOMETRY_OCCUPANCY_KEY,
  STAIR_ORACLE_EXPECTED_RISER_CONTACT_KEY,
  STAIR_ORACLE_EXPECTED_RISER_LAYER_KEY,
  STAIR_PHASE_KEY,
  STAIR_RISER_HEIGHT_LABEL_KEY,
  STAIR_SEQUENCE_ID_KEY,
  STAIR_TOE_RISER_HIT_LAYER_KEY,
  STAIR_TREAD_DEPTH_LABEL_KEY,
  TOE_RISER_NEW_HIT_KEY,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


STAIR_SEQUENCE_FIELDS = (
  "sequence_id",
  "terrain_sequence_id",
  "env_id",
  "episode_id",
  "terrain_id",
  "terrain_level",
  "terrain_type",
  "depth_bin",
  "true_tread_depth",
  "true_riser_height",
  "sequence_start_time",
  "sequence_end_time",
  "sequence_start_frame",
  "sequence_end_frame",
  "termination_reason",
  "has_first_layer1_collision",
  "first_layer1_collision_time",
  "has_first_stable_support",
  "first_stable_support_time",
  "has_adjacent_double_support",
  "adjacent_double_support_time",
  "has_partial_support",
  "partial_support_time",
  "has_layer2plus_riser_collision",
  "first_layer2plus_riser_collision_time",
  "has_confirmation",
  "first_confirmation_time",
  "confirmation_correct",
  "confirmation_layer",
  "confirmation_delay",
  "layer1_false_confirmation_count",
  "wrong_layer_confirmation_count",
  "wrong_sequence_confirmation_count",
  "flat_false_confirmation_count",
  "duplicate_confirmation_count",
  "missed_confirmation",
  "finish",
  "fall",
  "success",
  "toe_riser_hits_total",
  "toe_riser_hits_layer1",
  "toe_riser_hits_layer2plus",
  "toe_riser_duplicate_hits",
)

STAIR_EVENT_FIELDS = (
  "sequence_id",
  "terrain_sequence_id",
  "env_id",
  "episode_id",
  "event_id",
  "event_type",
  "time",
  "frame_idx",
  "true_tread_depth",
  "true_riser_height",
  "depth_bin",
  "event_layer",
  "is_layer1",
  "is_layer2plus",
  "is_confirmation",
  "confirmation_correct",
  "confirmation_error_type",
  "left_foot_x",
  "left_foot_y",
  "left_foot_z",
  "right_foot_x",
  "right_foot_y",
  "right_foot_z",
  "left_contact",
  "right_contact",
  "left_support_ratio",
  "right_support_ratio",
  "pair_depth",
  "pair_height",
  "root_x",
  "root_y",
  "root_z",
  "ascent_dir_x",
  "ascent_dir_y",
  "phase",
  "command_x",
  "command_y",
  "command_yaw",
  "optional_h_t_path_or_index",
  "optional_shape8_path_or_index",
)


class StairCsvExporter:
  """Buffer stair sequence/event rows and enforce one finalized sequence row."""

  def __init__(self, output_dir: str | Path | None, flush_rows: int = 128) -> None:
    self.enabled = output_dir is not None and str(output_dir).strip() != ""
    self._flush_rows = max(int(flush_rows), 1)
    self._next_sequence_id = 0
    self._active: dict[int, dict[str, Any]] = {}
    self._sequence_rows: list[dict[str, Any]] = []
    self._event_rows: list[dict[str, Any]] = []
    self._sequence_path: Path | None = None
    self._event_path: Path | None = None
    self._sequence_header_written = False
    self._event_header_written = False
    if self.enabled:
      assert output_dir is not None
      directory = Path(output_dir).expanduser().resolve()
      directory.mkdir(parents=True, exist_ok=True)
      self._sequence_path = directory / "stair_sequences.csv"
      self._event_path = directory / "stair_events.csv"
      self._write_header(self._sequence_path, STAIR_SEQUENCE_FIELDS)
      self._write_header(self._event_path, STAIR_EVENT_FIELDS)
      self._sequence_header_written = True
      self._event_header_written = True
      atexit.register(self.close)

  @property
  def active_env_ids(self) -> set[int]:
    return set(self._active)

  def start_sequence(
    self,
    env_id: int,
    sequence_values: dict[str, Any],
    entry_event_values: dict[str, Any],
  ) -> int:
    """Start one sequence and emit its unique layer-1 entry event."""
    if not self.enabled:
      return -1
    if env_id in self._active:
      raise RuntimeError(f"Environment {env_id} already has an active stair sequence.")
    sequence_id = self._next_sequence_id
    self._next_sequence_id += 1
    row: dict[str, Any] = {field: "" for field in STAIR_SEQUENCE_FIELDS}
    for field in STAIR_SEQUENCE_FIELDS:
      if field.startswith("has_") or field.endswith("_count"):
        row[field] = 0
    for field in (
      "confirmation_correct",
      "missed_confirmation",
      "finish",
      "fall",
      "success",
      "toe_riser_hits_total",
      "toe_riser_hits_layer1",
      "toe_riser_hits_layer2plus",
      "toe_riser_duplicate_hits",
    ):
      row[field] = 0
    row.update(sequence_values)
    row.update(
      {
        "sequence_id": sequence_id,
        "has_first_layer1_collision": 1,
        "first_layer1_collision_time": sequence_values["sequence_start_time"],
        "_next_event_id": 0,
        "_seen_once": {"first_layer1_collision"},
      }
    )
    self._active[env_id] = row
    self.record_event(
      env_id,
      "first_layer1_collision",
      entry_event_values,
      once_key=None,
    )
    return sequence_id

  def update_sequence(self, env_id: int, **values: Any) -> None:
    if self.enabled and env_id in self._active:
      self._active[env_id].update(values)

  def update_first_event(
    self,
    env_id: int,
    flag_field: str,
    time_field: str,
    time: float,
  ) -> None:
    if not self.enabled or env_id not in self._active:
      return
    sequence = self._active[env_id]
    if int(sequence.get(flag_field, 0)) == 0:
      sequence[flag_field] = 1
      sequence[time_field] = time

  def record_event(
    self,
    env_id: int,
    event_type: str,
    event_values: dict[str, Any],
    *,
    once_key: str | None,
  ) -> bool:
    """Record an event, optionally suppressing repeats within one sequence."""
    if not self.enabled or env_id not in self._active:
      return False
    sequence = self._active[env_id]
    seen_once = sequence["_seen_once"]
    if once_key is not None:
      if once_key in seen_once:
        return False
      seen_once.add(once_key)
    event_id = sequence["_next_event_id"]
    sequence["_next_event_id"] = event_id + 1
    row: dict[str, Any] = {field: "" for field in STAIR_EVENT_FIELDS}
    row.update(
      {
        "sequence_id": sequence["sequence_id"],
        "terrain_sequence_id": sequence["terrain_sequence_id"],
        "env_id": env_id,
        "episode_id": sequence["episode_id"],
        "event_id": event_id,
        "event_type": event_type,
        "true_tread_depth": sequence["true_tread_depth"],
        "true_riser_height": sequence["true_riser_height"],
        "depth_bin": sequence["depth_bin"],
      }
    )
    row.update(event_values)
    self._event_rows.append(row)
    self._flush_if_needed()
    return True

  def finish_sequence(self, env_id: int, final_values: dict[str, Any]) -> bool:
    """Finalize exactly one sequence row."""
    if not self.enabled or env_id not in self._active:
      return False
    row = self._active.pop(env_id)
    row.update(final_values)
    row.pop("_next_event_id", None)
    row.pop("_seen_once", None)
    self._sequence_rows.append(row)
    self._flush_if_needed()
    return True

  def flush(self) -> None:
    if not self.enabled:
      return
    assert self._sequence_path is not None
    assert self._event_path is not None
    if self._sequence_rows:
      self._sequence_header_written = self._append_rows(
        self._sequence_path,
        STAIR_SEQUENCE_FIELDS,
        self._sequence_rows,
        self._sequence_header_written,
      )
      self._sequence_rows.clear()
    if self._event_rows:
      self._event_header_written = self._append_rows(
        self._event_path,
        STAIR_EVENT_FIELDS,
        self._event_rows,
        self._event_header_written,
      )
      self._event_rows.clear()

  def close(self) -> None:
    """Finalize process-end sequences and flush all buffered rows."""
    if not self.enabled:
      return
    for env_id in tuple(self._active):
      self.finish_sequence(
        env_id,
        {
          "termination_reason": "process_end",
          "finish": 0,
          "fall": 0,
          "success": 0,
        },
      )
    self.flush()

  def _flush_if_needed(self) -> None:
    if len(self._sequence_rows) + len(self._event_rows) >= self._flush_rows:
      self.flush()

  @staticmethod
  def _write_header(path: Path, fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
      csv.DictWriter(stream, fieldnames=fields).writeheader()

  @staticmethod
  def _append_rows(
    path: Path,
    fields: tuple[str, ...],
    rows: list[dict[str, Any]],
    header_written: bool,
  ) -> bool:
    with path.open("a", encoding="utf-8", newline="") as stream:
      writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
      if not header_written:
        writer.writeheader()
      writer.writerows(rows)
    return True


class stair_sequence_event_logger:
  """Track stair events independently of reward values and optionally export CSV."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    del cfg
    self._num_envs = env.num_envs
    self._device = env.device
    self._active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self._episode_id = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._previous_partial_support = torch.zeros_like(self._active)
    self._previous_full_support = torch.zeros_like(self._active)
    self._previous_pair = torch.zeros_like(self._active)
    self._previous_oracle_contact = torch.zeros_like(self._active)
    self._previous_confirmation_condition = torch.zeros_like(self._active)
    self._previous_toe_hit = torch.zeros_like(self._active)
    self._seen_partial_support = torch.zeros_like(self._active)
    self._seen_full_support = torch.zeros_like(self._active)
    self._seen_adjacent_support = torch.zeros_like(self._active)
    self._oracle_seen = torch.zeros_like(self._active)
    self._oracle_frame = torch.full(
      (env.num_envs,), -1, device=env.device, dtype=torch.long
    )
    self._oracle_layer = torch.full_like(self._oracle_frame, -1)
    self._confirmation_seen = torch.zeros_like(self._active)
    self._confirmation_correct = torch.zeros_like(self._active)
    self._confirmation_layer = torch.full_like(self._oracle_frame, -1)
    self._confirmation_duplicates = torch.zeros_like(self._oracle_frame)
    self._layer1_false_count = torch.zeros_like(self._oracle_frame)
    self._wrong_layer_count = torch.zeros_like(self._oracle_frame)
    self._wrong_sequence_count = torch.zeros_like(self._oracle_frame)
    self._flat_false_count = torch.zeros_like(self._oracle_frame)
    self._toe_hits_total = torch.zeros_like(self._oracle_frame)
    self._toe_hits_layer1 = torch.zeros_like(self._oracle_frame)
    self._toe_hits_layer2plus = torch.zeros_like(self._oracle_frame)
    self._toe_hits_duplicate = torch.zeros_like(self._oracle_frame)
    self._seen_confirmation_layers = torch.zeros(
      env.num_envs, 64, device=env.device, dtype=torch.bool
    )
    self._seen_hit_layers = torch.zeros(
      env.num_envs, 64, device=env.device, dtype=torch.bool
    )

    zero = torch.zeros((), device=env.device, dtype=torch.float64)
    self._confirmation_total = zero.clone()
    self._confirmation_correct_total = zero.clone()
    self._layer1_false_total = zero.clone()
    self._wrong_layer_total = zero.clone()
    self._wrong_sequence_total = zero.clone()
    self._flat_false_total = zero.clone()
    self._duplicate_total = zero.clone()
    self._finalized_oracle_total = zero.clone()
    self._detected_oracle_total = zero.clone()
    self._missed_total = zero.clone()
    self._delay_count = zero.clone()
    self._delay_sum = zero.clone()
    self._delay_square_sum = zero.clone()

    output_dir = os.environ.get("MJLAB_STAIR_EXPORT_DIR")
    flush_rows = int(os.environ.get("MJLAB_STAIR_EXPORT_FLUSH_ROWS", "128"))
    self._exporter = StairCsvExporter(output_dir, flush_rows=flush_rows)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._episode_id[env_ids] += 1
    self._active[env_ids] = False
    self._clear_sequence_state(env_ids)

  def flush(self) -> None:
    """Flush pending CSV rows when export is enabled."""
    self._exporter.flush()

  def close(self) -> None:
    """Finalize process-end sequences and flush pending CSV rows."""
    self._exporter.close()

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "twist",
  ) -> torch.Tensor:
    phase = self._extra(env, STAIR_PHASE_KEY, (), torch.long)
    active_now = phase >= 1
    begin = active_now & ~self._active
    self._clear_sequence_state(begin)
    self._active[begin] = True
    self._export_starts(env, begin, asset_cfg, command_name)

    support = self._extra(env, STAIR_CURRENT_STAIR_SUPPORT_KEY, (2,), torch.bool).bool()
    support_fraction = self._extra(
      env, STAIR_CURRENT_SUPPORT_FRACTION_KEY, (2,), torch.float32
    )
    partial_condition = active_now & torch.any(
      support & (support_fraction < 1.0 - 1.0e-6), dim=-1
    )
    full_condition = active_now & torch.any(
      support & (support_fraction >= 1.0 - 1.0e-6), dim=-1
    )
    partial_edge = (
      partial_condition & ~self._previous_partial_support & ~self._seen_partial_support
    )
    full_edge = full_condition & ~self._previous_full_support & ~self._seen_full_support
    self._seen_partial_support |= partial_edge
    self._seen_full_support |= full_edge

    pair_condition = self._extra(
      env, STAIR_ADJACENT_PAIR_VALID_KEY, (), torch.bool
    ).bool()
    pair_edge = (
      active_now & pair_condition & ~self._previous_pair & ~self._seen_adjacent_support
    )
    self._seen_adjacent_support |= pair_edge
    pair_full_edge = pair_edge & torch.all(support_fraction >= 1.0 - 1.0e-6, dim=-1)
    pair_partial_edge = pair_edge & ~pair_full_edge

    oracle_condition = self._extra(
      env, STAIR_ORACLE_EXPECTED_RISER_CONTACT_KEY, (), torch.bool
    ).bool()
    oracle_edge = active_now & oracle_condition & ~self._previous_oracle_contact
    first_oracle = oracle_edge & ~self._oracle_seen
    frame_idx = int(env.common_step_counter)
    oracle_layer = self._extra(
      env, STAIR_ORACLE_EXPECTED_RISER_LAYER_KEY, (), torch.long
    ).long()
    self._oracle_seen |= first_oracle
    self._oracle_frame[first_oracle] = frame_idx
    self._oracle_layer[first_oracle] = oracle_layer[first_oracle]

    raw_confirmation_event = self._extra(
      env, STAIR_DEPTH_CONFIRMATION_EVENT_KEY, (), torch.bool
    ).bool()
    confirmation_event = active_now & raw_confirmation_event & ~self._confirmation_seen
    confirmation_condition = self._extra(
      env, STAIR_CONFIRMATION_CONDITION_KEY, (), torch.bool
    ).bool()
    condition_edge = (
      active_now & confirmation_condition & ~self._previous_confirmation_condition
    )
    confirmation_layer = self._extra(
      env, STAIR_CONFIRMATION_LAYER_KEY, (), torch.long
    ).long()
    env_ids = torch.arange(self._num_envs, device=self._device)
    clamped_confirmation_layer = confirmation_layer.clamp(
      0, self._seen_confirmation_layers.shape[1] - 1
    )
    valid_condition_edge = condition_edge & (confirmation_layer >= 0)
    layer_seen = self._seen_confirmation_layers[env_ids, clamped_confirmation_layer]
    duplicate = valid_condition_edge & layer_seen
    new_confirmation_layer = (
      valid_condition_edge & self._confirmation_seen & ~layer_seen
    )
    self._seen_confirmation_layers[
      env_ids[valid_condition_edge],
      clamped_confirmation_layer[valid_condition_edge],
    ] = True
    self._duplicate_total += duplicate.double().sum()
    self._confirmation_duplicates += duplicate.long()
    self._confirmation_seen |= confirmation_event

    confirmation_sequence = self._extra(
      env, STAIR_CONFIRMATION_CONTACT_SEQUENCE_KEY, (), torch.long
    ).long()
    sequence_id = self._extra(env, STAIR_SEQUENCE_ID_KEY, (), torch.long).long()
    occupancy = self._extra(env, STAIR_GEOMETRY_OCCUPANCY_KEY, (), torch.bool).bool()
    same_layer = confirmation_layer == self._oracle_layer
    same_sequence = confirmation_sequence == sequence_id
    correct = (
      confirmation_event & self._oracle_seen & same_layer & same_sequence & occupancy
    )
    self._confirmation_correct |= correct
    self._confirmation_layer = torch.where(
      confirmation_event, confirmation_layer, self._confirmation_layer
    )
    layer1_false = confirmation_event & (confirmation_layer < 2)
    # Compare with the layer latched on the independent raw-contact edge. The
    # state machine may advance expected_layer later in the same environment step.
    wrong_layer = confirmation_event & self._oracle_seen & ~same_layer
    wrong_sequence = confirmation_event & ~same_sequence
    flat_false = confirmation_event & ~occupancy
    self._layer1_false_count += layer1_false.long()
    self._wrong_layer_count += wrong_layer.long()
    self._wrong_sequence_count += wrong_sequence.long()
    self._flat_false_count += flat_false.long()
    self._confirmation_total += confirmation_event.double().sum()
    self._confirmation_correct_total += correct.double().sum()
    self._layer1_false_total += layer1_false.double().sum()
    self._wrong_layer_total += wrong_layer.double().sum()
    self._wrong_sequence_total += wrong_sequence.double().sum()
    self._flat_false_total += flat_false.double().sum()
    delay = torch.full_like(self._oracle_frame, -1)
    delay[correct] = frame_idx - self._oracle_frame[correct]
    valid_delay = delay >= 0
    self._delay_count += valid_delay.double().sum()
    self._delay_sum += delay[valid_delay].double().sum()
    self._delay_square_sum += delay[valid_delay].double().square().sum()

    toe_hit = self._extra(env, TOE_RISER_NEW_HIT_KEY, (), torch.bool).bool()
    toe_hit_edge = active_now & toe_hit & ~self._previous_toe_hit
    hit_layer = self._extra(env, STAIR_TOE_RISER_HIT_LAYER_KEY, (), torch.long).long()
    valid_hit = toe_hit_edge & (hit_layer >= 0)
    clamped_layer = hit_layer.clamp(0, self._seen_hit_layers.shape[1] - 1)
    env_ids = torch.arange(self._num_envs, device=self._device)
    duplicate_hit = valid_hit & self._seen_hit_layers[env_ids, clamped_layer]
    self._seen_hit_layers[env_ids[valid_hit], clamped_layer[valid_hit]] = True
    self._toe_hits_total += valid_hit.long()
    self._toe_hits_layer1 += (valid_hit & (hit_layer == 1)).long()
    self._toe_hits_layer2plus += (valid_hit & (hit_layer >= 2)).long()
    self._toe_hits_duplicate += duplicate_hit.long()

    self._export_events(
      env,
      asset_cfg,
      command_name,
      partial_edge,
      full_edge,
      pair_partial_edge,
      pair_full_edge,
      first_oracle,
      confirmation_event,
      correct,
      layer1_false,
      wrong_layer,
      wrong_sequence,
      flat_false,
      duplicate,
      new_confirmation_layer,
      delay,
    )

    exit_event = self._extra(env, STAIR_EXIT_EVENT_KEY, (), torch.bool).bool()
    terminated = getattr(env, "reset_terminated", torch.zeros_like(self._active))
    time_out = getattr(env, "reset_time_outs", torch.zeros_like(self._active))
    reset_buf = getattr(env, "reset_buf", torch.zeros_like(self._active))
    state_reset = self._active & ~active_now
    finish = self._active & (exit_event | reset_buf | state_reset)
    finalized_oracle = finish & self._oracle_seen
    detected_oracle = finalized_oracle & self._confirmation_correct
    missed = finalized_oracle & ~self._confirmation_correct
    self._finalized_oracle_total += finalized_oracle.double().sum()
    self._detected_oracle_total += detected_oracle.double().sum()
    self._missed_total += missed.double().sum()
    self._export_finishes(
      env,
      finish,
      exit_event,
      terminated,
      time_out,
      missed,
      asset_cfg,
      command_name,
    )
    self._active[finish] = False
    self._clear_sequence_state(finish)

    self._previous_partial_support.copy_(partial_condition & self._active)
    self._previous_full_support.copy_(full_condition & self._active)
    self._previous_pair.copy_(pair_condition & self._active)
    self._previous_oracle_contact.copy_(oracle_condition & self._active)
    self._previous_confirmation_condition.copy_(confirmation_condition & self._active)
    self._previous_toe_hit.copy_(toe_hit & self._active)
    self._log_metrics(env)
    return torch.zeros(self._num_envs, device=self._device)

  def _clear_sequence_state(self, env_ids: torch.Tensor | slice) -> None:
    self._previous_partial_support[env_ids] = False
    self._previous_full_support[env_ids] = False
    self._previous_pair[env_ids] = False
    self._previous_oracle_contact[env_ids] = False
    self._previous_confirmation_condition[env_ids] = False
    self._previous_toe_hit[env_ids] = False
    self._seen_partial_support[env_ids] = False
    self._seen_full_support[env_ids] = False
    self._seen_adjacent_support[env_ids] = False
    self._oracle_seen[env_ids] = False
    self._oracle_frame[env_ids] = -1
    self._oracle_layer[env_ids] = -1
    self._confirmation_seen[env_ids] = False
    self._confirmation_correct[env_ids] = False
    self._confirmation_layer[env_ids] = -1
    self._confirmation_duplicates[env_ids] = 0
    self._layer1_false_count[env_ids] = 0
    self._wrong_layer_count[env_ids] = 0
    self._wrong_sequence_count[env_ids] = 0
    self._flat_false_count[env_ids] = 0
    self._toe_hits_total[env_ids] = 0
    self._toe_hits_layer1[env_ids] = 0
    self._toe_hits_layer2plus[env_ids] = 0
    self._toe_hits_duplicate[env_ids] = 0
    self._seen_confirmation_layers[env_ids] = False
    self._seen_hit_layers[env_ids] = False

  def _extra(
    self,
    env: ManagerBasedRlEnv,
    key: str,
    trailing_shape: tuple[int, ...],
    dtype: torch.dtype,
  ) -> torch.Tensor:
    value = env.extras.get(key)
    if isinstance(value, torch.Tensor):
      return value
    return torch.zeros(
      (self._num_envs, *trailing_shape), device=self._device, dtype=dtype
    )

  def _event_payloads(
    self,
    env: ManagerBasedRlEnv,
    mask: torch.Tensor,
    event_layer: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    command_name: str,
  ) -> dict[int, dict[str, Any]]:
    if not self._exporter.enabled or not torch.any(mask):
      return {}
    ids = mask.nonzero(as_tuple=False).squeeze(-1)
    asset = env.scene[asset_cfg.name]
    feet = asset.data.body_link_pos_w[ids][:, asset_cfg.body_ids, :]
    root = asset.data.root_link_pos_w[ids]
    ground_contact = self._extra(
      env, STAIR_CURRENT_GROUND_CONTACT_KEY, (2,), torch.bool
    )[ids]
    support_fraction = self._extra(
      env, STAIR_CURRENT_SUPPORT_FRACTION_KEY, (2,), torch.float32
    )[ids]
    pair_depth = self._extra(env, STAIR_ADJACENT_PAIR_DEPTH_KEY, (), torch.float32)[ids]
    pair_height = self._extra(env, STAIR_ADJACENT_PAIR_HEIGHT_KEY, (), torch.float32)[
      ids
    ]
    ascent = self._extra(env, STAIR_ASCENT_DIR_KEY, (2,), torch.float32)[ids]
    phase = self._extra(env, STAIR_PHASE_KEY, (), torch.long)[ids]
    command = env.command_manager.get_command(command_name)
    if command is None:
      command = torch.zeros(self._num_envs, 3, device=self._device)
    selected = {
      "ids": ids.cpu().tolist(),
      "feet": feet.detach().cpu().tolist(),
      "root": root.detach().cpu().tolist(),
      "ground": ground_contact.detach().cpu().tolist(),
      "support": support_fraction.detach().cpu().tolist(),
      "pair_depth": pair_depth.detach().cpu().tolist(),
      "pair_height": pair_height.detach().cpu().tolist(),
      "ascent": ascent.detach().cpu().tolist(),
      "phase": phase.detach().cpu().tolist(),
      "command": command[ids].detach().cpu().tolist(),
      "layer": event_layer[ids].detach().cpu().tolist(),
    }
    frame_idx = int(env.common_step_counter)
    time = frame_idx * float(env.step_dt)
    payloads: dict[int, dict[str, Any]] = {}
    for index, env_id in enumerate(selected["ids"]):
      feet_row = selected["feet"][index]
      command_row = selected["command"][index]
      payloads[env_id] = {
        "time": time,
        "frame_idx": frame_idx,
        "event_layer": selected["layer"][index],
        "is_layer1": int(selected["layer"][index] == 1),
        "is_layer2plus": int(selected["layer"][index] >= 2),
        "left_foot_x": feet_row[0][0],
        "left_foot_y": feet_row[0][1],
        "left_foot_z": feet_row[0][2],
        "right_foot_x": feet_row[1][0],
        "right_foot_y": feet_row[1][1],
        "right_foot_z": feet_row[1][2],
        "left_contact": int(selected["ground"][index][0]),
        "right_contact": int(selected["ground"][index][1]),
        "left_support_ratio": selected["support"][index][0],
        "right_support_ratio": selected["support"][index][1],
        "pair_depth": selected["pair_depth"][index],
        "pair_height": selected["pair_height"][index],
        "root_x": selected["root"][index][0],
        "root_y": selected["root"][index][1],
        "root_z": selected["root"][index][2],
        "ascent_dir_x": selected["ascent"][index][0],
        "ascent_dir_y": selected["ascent"][index][1],
        "phase": selected["phase"][index],
        "command_x": command_row[0],
        "command_y": command_row[1],
        "command_yaw": command_row[2],
      }
    return payloads

  def _export_starts(
    self,
    env: ManagerBasedRlEnv,
    mask: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    command_name: str,
  ) -> None:
    if not self._exporter.enabled:
      return
    sequence = self._extra(env, STAIR_SEQUENCE_ID_KEY, (), torch.long)
    depth = self._extra(env, STAIR_TREAD_DEPTH_LABEL_KEY, (), torch.float32)
    height = self._extra(env, STAIR_RISER_HEIGHT_LABEL_KEY, (), torch.float32)
    terrain = env.scene.terrain
    levels = getattr(
      terrain,
      "terrain_levels",
      torch.full((self._num_envs,), -1, device=self._device),
    )
    types = getattr(
      terrain,
      "terrain_types",
      torch.full((self._num_envs,), -1, device=self._device),
    )
    payloads = self._event_payloads(
      env,
      mask,
      torch.ones_like(sequence),
      asset_cfg,
      command_name,
    )
    ids = mask.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
    frame_idx = int(env.common_step_counter)
    time = frame_idx * float(env.step_dt)
    for env_id in ids:
      depth_value = float(depth[env_id].item())
      depth_bin = round((depth_value - 0.25) / ((0.35 - 0.25) / 7.0))
      level = int(levels[env_id].item())
      terrain_type = int(types[env_id].item())
      self._exporter.start_sequence(
        env_id,
        {
          "terrain_sequence_id": int(sequence[env_id].item()),
          "env_id": env_id,
          "episode_id": int(self._episode_id[env_id].item()),
          "terrain_id": level * 1_000_000 + terrain_type,
          "terrain_level": level,
          "terrain_type": terrain_type,
          "depth_bin": max(0, min(7, depth_bin)),
          "true_tread_depth": depth_value,
          "true_riser_height": float(height[env_id].item()),
          "sequence_start_time": time,
          "sequence_start_frame": frame_idx,
        },
        payloads[env_id],
      )

  def _export_events(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    partial_edge: torch.Tensor,
    full_edge: torch.Tensor,
    pair_partial_edge: torch.Tensor,
    pair_full_edge: torch.Tensor,
    oracle_edge: torch.Tensor,
    confirmation_event: torch.Tensor,
    correct: torch.Tensor,
    layer1_false: torch.Tensor,
    wrong_layer: torch.Tensor,
    wrong_sequence: torch.Tensor,
    flat_false: torch.Tensor,
    duplicate: torch.Tensor,
    new_confirmation_layer: torch.Tensor,
    delay: torch.Tensor,
  ) -> None:
    if not self._exporter.enabled:
      return
    expected_layer = self._extra(env, STAIR_EXPECTED_LAYER_KEY, (), torch.long).long()
    oracle_layer = self._extra(
      env, STAIR_ORACLE_EXPECTED_RISER_LAYER_KEY, (), torch.long
    ).long()
    confirmation_layer = self._extra(
      env, STAIR_CONFIRMATION_LAYER_KEY, (), torch.long
    ).long()
    pair_layers = torch.amax(
      self._extra(env, STAIR_CURRENT_SUPPORT_LAYER_KEY, (2,), torch.long).long(),
      dim=-1,
    )
    event_specs = (
      (partial_edge, "first_stable_support_partial", expected_layer),
      (full_edge, "first_stable_support_full", expected_layer),
      (
        pair_partial_edge,
        "adjacent_double_support_partial",
        pair_layers,
      ),
      (pair_full_edge, "adjacent_double_support_full", pair_layers),
      (oracle_edge, "layer2plus_riser_collision", oracle_layer),
    )
    for mask, event_type, layer in event_specs:
      payloads = self._event_payloads(env, mask, layer, asset_cfg, command_name)
      for env_id, payload in payloads.items():
        if self._exporter.record_event(
          env_id,
          event_type,
          payload,
          once_key=event_type,
        ):
          time = payload["time"]
          if event_type.startswith("first_stable_support"):
            self._exporter.update_first_event(
              env_id,
              "has_first_stable_support",
              "first_stable_support_time",
              time,
            )
            if event_type.endswith("partial"):
              self._exporter.update_first_event(
                env_id,
                "has_partial_support",
                "partial_support_time",
                time,
              )
          elif event_type.startswith("adjacent_double_support"):
            self._exporter.update_sequence(
              env_id,
              has_adjacent_double_support=1,
              adjacent_double_support_time=time,
            )
          elif event_type == "layer2plus_riser_collision":
            self._exporter.update_sequence(
              env_id,
              has_layer2plus_riser_collision=1,
              first_layer2plus_riser_collision_time=time,
            )

    payloads = self._event_payloads(
      env,
      confirmation_event,
      confirmation_layer,
      asset_cfg,
      command_name,
    )
    for env_id, payload in payloads.items():
      errors: list[str] = []
      if bool(layer1_false[env_id]):
        errors.append("layer1")
      if bool(wrong_layer[env_id]):
        errors.append("wrong_layer")
      if bool(wrong_sequence[env_id]):
        errors.append("wrong_sequence")
      if bool(flat_false[env_id]):
        errors.append("flat")
      payload.update(
        {
          "is_confirmation": 1,
          "confirmation_correct": int(correct[env_id].item()),
          "confirmation_error_type": "|".join(errors),
        }
      )
      recorded = self._exporter.record_event(
        env_id,
        "confirmation",
        payload,
        once_key="confirmation",
      )
      if recorded:
        self._exporter.update_sequence(
          env_id,
          has_confirmation=1,
          first_confirmation_time=payload["time"],
          confirmation_correct=int(correct[env_id].item()),
          confirmation_layer=payload["event_layer"],
          confirmation_delay=(int(delay[env_id].item()) if delay[env_id] >= 0 else ""),
        )

    payloads = self._event_payloads(
      env,
      duplicate,
      confirmation_layer,
      asset_cfg,
      command_name,
    )
    for env_id, payload in payloads.items():
      payload.update(
        {
          "is_confirmation": 1,
          "confirmation_correct": 0,
          "confirmation_error_type": "duplicate",
        }
      )
      self._exporter.record_event(
        env_id,
        "confirmation_duplicate",
        payload,
        once_key=None,
      )

    payloads = self._event_payloads(
      env,
      new_confirmation_layer,
      confirmation_layer,
      asset_cfg,
      command_name,
    )
    for env_id, payload in payloads.items():
      payload.update(
        {
          "is_confirmation": 1,
          "confirmation_correct": 1,
          "confirmation_error_type": "",
        }
      )
      self._exporter.record_event(
        env_id,
        "confirmation_new_layer",
        payload,
        once_key=None,
      )

  def _export_finishes(
    self,
    env: ManagerBasedRlEnv,
    finish: torch.Tensor,
    exit_event: torch.Tensor,
    terminated: torch.Tensor,
    time_out: torch.Tensor,
    missed: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    command_name: str,
  ) -> None:
    if not self._exporter.enabled:
      return
    expected_layer = self._extra(env, STAIR_EXPECTED_LAYER_KEY, (), torch.long).long()
    no_collision = finish & ~self._oracle_seen
    payloads = self._event_payloads(
      env,
      no_collision,
      expected_layer,
      asset_cfg,
      command_name,
    )
    for env_id, payload in payloads.items():
      self._exporter.record_event(
        env_id,
        "finish_without_collision",
        payload,
        once_key="finish_without_collision",
      )
    payloads = self._event_payloads(
      env,
      finish & terminated,
      expected_layer,
      asset_cfg,
      command_name,
    )
    for env_id, payload in payloads.items():
      self._exporter.record_event(
        env_id,
        "fall",
        payload,
        once_key="fall",
      )

    ids = finish.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
    frame_idx = int(env.common_step_counter)
    time = frame_idx * float(env.step_dt)
    for env_id in ids:
      success = bool(exit_event[env_id])
      fall = bool(terminated[env_id])
      if success:
        reason = "confirmed_exit"
      elif fall:
        reason = "fall"
      elif bool(time_out[env_id]):
        reason = "time_out"
      else:
        reason = "state_reset"
      self._exporter.finish_sequence(
        env_id,
        {
          "sequence_end_time": time,
          "sequence_end_frame": frame_idx,
          "termination_reason": reason,
          "missed_confirmation": int(missed[env_id].item()),
          "finish": int(success),
          "fall": int(fall),
          "success": int(success and not fall),
          "duplicate_confirmation_count": int(
            self._confirmation_duplicates[env_id].item()
          ),
          "layer1_false_confirmation_count": int(
            self._layer1_false_count[env_id].item()
          ),
          "wrong_layer_confirmation_count": int(self._wrong_layer_count[env_id].item()),
          "wrong_sequence_confirmation_count": int(
            self._wrong_sequence_count[env_id].item()
          ),
          "flat_false_confirmation_count": int(self._flat_false_count[env_id].item()),
          "toe_riser_hits_total": int(self._toe_hits_total[env_id].item()),
          "toe_riser_hits_layer1": int(self._toe_hits_layer1[env_id].item()),
          "toe_riser_hits_layer2plus": int(self._toe_hits_layer2plus[env_id].item()),
          "toe_riser_duplicate_hits": int(self._toe_hits_duplicate[env_id].item()),
        },
      )

  def _log_metrics(self, env: ManagerBasedRlEnv) -> None:
    safe_confirmation = self._confirmation_total.clamp_min(1.0)
    safe_oracle = self._finalized_oracle_total.clamp_min(1.0)
    safe_delay = self._delay_count.clamp_min(1.0)
    delay_mean = self._delay_sum / safe_delay
    delay_variance = (
      self._delay_square_sum / safe_delay - delay_mean.square()
    ).clamp_min(0.0)
    log = env.extras["log"]
    log["Metrics/stair_confirm_precision"] = (
      self._confirmation_correct_total / safe_confirmation
    ).float()
    log["Metrics/stair_confirm_recall"] = (
      self._detected_oracle_total / safe_oracle
    ).float()
    log["Metrics/stair_confirm_layer1_false_ratio"] = (
      self._layer1_false_total / safe_confirmation
    ).float()
    log["Metrics/stair_confirm_wrong_layer_ratio"] = (
      self._wrong_layer_total / safe_confirmation
    ).float()
    log["Metrics/stair_confirm_wrong_sequence_ratio"] = (
      self._wrong_sequence_total / safe_confirmation
    ).float()
    log["Metrics/stair_confirm_flat_false_ratio"] = (
      self._flat_false_total / safe_confirmation
    ).float()
    log["Metrics/stair_confirm_duplicate_ratio"] = (
      self._duplicate_total / safe_confirmation
    ).float()
    log["Metrics/stair_confirm_missed_ratio"] = (
      self._missed_total / safe_oracle
    ).float()
    log["Metrics/stair_confirm_delay_mean"] = delay_mean.float()
    log["Metrics/stair_confirm_delay_std"] = torch.sqrt(delay_variance).float()
    log["Metrics/stair_confirm_count"] = self._confirmation_total.float()
    log["Metrics/stair_confirm_finalized_oracle_count"] = (
      self._finalized_oracle_total.float()
    )
