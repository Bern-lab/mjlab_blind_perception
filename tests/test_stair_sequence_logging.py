from __future__ import annotations

import csv
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.stair_geometry import (
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
from mjlab.tasks.velocity.mdp.stair_sequence_logging import (
  stair_sequence_event_logger,
)


class _CommandManager:
  def get_command(self, name: str) -> torch.Tensor:
    assert name == "twist"
    return torch.tensor([[0.6, 0.0, 0.0]])


class _Scene:
  def __init__(self, robot, terrain) -> None:
    self._robot = robot
    self.terrain = terrain

  def __getitem__(self, name: str):
    assert name == "robot"
    return self._robot


def _make_env() -> SimpleNamespace:
  robot = SimpleNamespace(
    data=SimpleNamespace(
      body_link_pos_w=torch.tensor([[[0.1, 0.0, 0.1], [-0.1, 0.0, 0.0]]]),
      root_link_pos_w=torch.tensor([[0.0, 0.0, 0.8]]),
    )
  )
  terrain = SimpleNamespace(
    terrain_levels=torch.tensor([4]),
    terrain_types=torch.tensor([3]),
  )
  scene = _Scene(robot, terrain)
  extras = {
    "log": {},
    STAIR_PHASE_KEY: torch.tensor([1]),
    STAIR_SEQUENCE_ID_KEY: torch.tensor([7]),
    STAIR_TREAD_DEPTH_LABEL_KEY: torch.tensor([0.30]),
    STAIR_RISER_HEIGHT_LABEL_KEY: torch.tensor([0.18]),
    STAIR_EXPECTED_LAYER_KEY: torch.tensor([2]),
    STAIR_ASCENT_DIR_KEY: torch.tensor([[1.0, 0.0]]),
    STAIR_CURRENT_GROUND_CONTACT_KEY: torch.tensor([[True, True]]),
    STAIR_CURRENT_STAIR_SUPPORT_KEY: torch.tensor([[True, True]]),
    STAIR_CURRENT_SUPPORT_FRACTION_KEY: torch.tensor([[1.0, 1.0]]),
    STAIR_CURRENT_SUPPORT_LAYER_KEY: torch.tensor([[1, 2]]),
    STAIR_ADJACENT_PAIR_VALID_KEY: torch.tensor([False]),
    STAIR_ADJACENT_PAIR_DEPTH_KEY: torch.tensor([0.0]),
    STAIR_ADJACENT_PAIR_HEIGHT_KEY: torch.tensor([0.0]),
    STAIR_ORACLE_EXPECTED_RISER_CONTACT_KEY: torch.tensor([False]),
    STAIR_ORACLE_EXPECTED_RISER_LAYER_KEY: torch.tensor([-1]),
    STAIR_CONFIRMATION_CONDITION_KEY: torch.tensor([False]),
    STAIR_DEPTH_CONFIRMATION_EVENT_KEY: torch.tensor([False]),
    STAIR_CONFIRMATION_LAYER_KEY: torch.tensor([-1]),
    STAIR_CONFIRMATION_CONTACT_SEQUENCE_KEY: torch.tensor([-1]),
    STAIR_GEOMETRY_OCCUPANCY_KEY: torch.tensor([True]),
    TOE_RISER_NEW_HIT_KEY: torch.tensor([False]),
    STAIR_TOE_RISER_HIT_LAYER_KEY: torch.tensor([-1]),
    STAIR_EXIT_EVENT_KEY: torch.tensor([False]),
  }
  return SimpleNamespace(
    num_envs=1,
    device="cpu",
    scene=scene,
    command_manager=_CommandManager(),
    extras=extras,
    common_step_counter=0,
    step_dt=0.02,
    reset_buf=torch.tensor([False]),
    reset_terminated=torch.tensor([False]),
    reset_time_outs=torch.tensor([False]),
  )


def _step(
  logger: stair_sequence_event_logger,
  env: SimpleNamespace,
  asset_cfg: SceneEntityCfg,
) -> None:
  env.extras["log"] = {}
  logger(cast(Any, env), asset_cfg=asset_cfg)
  env.common_step_counter += 1


def test_stage0_logger_uses_edges_and_finalizes_one_sequence(
  tmp_path, monkeypatch
) -> None:
  monkeypatch.setenv("MJLAB_STAIR_EXPORT_DIR", str(tmp_path))
  env = _make_env()
  logger = stair_sequence_event_logger(None, cast(Any, env))
  asset_cfg = SceneEntityCfg("robot", body_ids=[0, 1])

  _step(logger, env, asset_cfg)

  env.extras[STAIR_ORACLE_EXPECTED_RISER_CONTACT_KEY][:] = True
  env.extras[STAIR_ORACLE_EXPECTED_RISER_LAYER_KEY][:] = 2
  _step(logger, env, asset_cfg)

  env.extras[STAIR_CONFIRMATION_CONDITION_KEY][:] = True
  env.extras[STAIR_DEPTH_CONFIRMATION_EVENT_KEY][:] = True
  env.extras[STAIR_CONFIRMATION_LAYER_KEY][:] = 2
  env.extras[STAIR_CONFIRMATION_CONTACT_SEQUENCE_KEY][:] = 7
  # expected_layer can advance later in the confirmation step. It must not turn
  # a detector hit matching the latched oracle layer into a wrong-layer error.
  env.extras[STAIR_EXPECTED_LAYER_KEY][:] = 3
  _step(logger, env, asset_cfg)

  env.extras[STAIR_DEPTH_CONFIRMATION_EVENT_KEY][:] = False
  _step(logger, env, asset_cfg)

  env.extras[STAIR_CONFIRMATION_CONDITION_KEY][:] = False
  env.extras[STAIR_ORACLE_EXPECTED_RISER_CONTACT_KEY][:] = False
  _step(logger, env, asset_cfg)

  env.extras[STAIR_CONFIRMATION_CONDITION_KEY][:] = True
  _step(logger, env, asset_cfg)

  env.extras[STAIR_CONFIRMATION_CONDITION_KEY][:] = False
  _step(logger, env, asset_cfg)

  env.extras[STAIR_CONFIRMATION_LAYER_KEY][:] = 3
  env.extras[STAIR_CONFIRMATION_CONDITION_KEY][:] = True
  _step(logger, env, asset_cfg)

  env.extras[STAIR_PHASE_KEY][:] = 0
  env.extras[STAIR_EXIT_EVENT_KEY][:] = True
  _step(logger, env, asset_cfg)
  logger.flush()

  with (tmp_path / "stair_sequences.csv").open(newline="") as stream:
    sequences = list(csv.DictReader(stream))
  with (tmp_path / "stair_events.csv").open(newline="") as stream:
    events = list(csv.DictReader(stream))

  assert len(sequences) == 1
  assert sequences[0]["confirmation_correct"] == "1"
  assert sequences[0]["confirmation_delay"] == "1"
  assert sequences[0]["duplicate_confirmation_count"] == "1"
  assert sequences[0]["wrong_layer_confirmation_count"] == "0"
  assert sequences[0]["success"] == "1"
  assert [event["event_type"] for event in events].count(
    "layer2plus_riser_collision"
  ) == 1
  assert [event["event_type"] for event in events].count("confirmation") == 1
  assert [event["event_type"] for event in events].count("confirmation_duplicate") == 1
  assert [event["event_type"] for event in events].count("confirmation_new_layer") == 1
  assert env.extras["log"]["Metrics/stair_confirm_precision"].item() == pytest.approx(
    1.0
  )
  assert env.extras["log"]["Metrics/stair_confirm_recall"].item() == pytest.approx(1.0)


def test_confirmation_event_is_ignored_when_not_active(tmp_path, monkeypatch) -> None:
  monkeypatch.setenv("MJLAB_STAIR_EXPORT_DIR", str(tmp_path))
  env = _make_env()
  logger = stair_sequence_event_logger(None, cast(Any, env))
  asset_cfg = SceneEntityCfg("robot", body_ids=[0, 1])

  _step(logger, env, asset_cfg)

  env.extras[STAIR_PHASE_KEY][:] = 0
  env.extras[STAIR_CONFIRMATION_CONDITION_KEY][:] = True
  env.extras[STAIR_DEPTH_CONFIRMATION_EVENT_KEY][:] = True
  env.extras[STAIR_CONFIRMATION_LAYER_KEY][:] = 2
  env.extras[STAIR_CONFIRMATION_CONTACT_SEQUENCE_KEY][:] = 7
  _step(logger, env, asset_cfg)
  logger.flush()

  with (tmp_path / "stair_sequences.csv").open(newline="") as stream:
    sequences = list(csv.DictReader(stream))
  with (tmp_path / "stair_events.csv").open(newline="") as stream:
    events = list(csv.DictReader(stream))

  assert len(sequences) == 1
  assert sequences[0]["termination_reason"] == "state_reset"
  assert sequences[0]["has_confirmation"] == "0"
  assert [event["event_type"] for event in events].count("confirmation") == 0
  assert env.extras["log"]["Metrics/stair_confirm_count"].item() == pytest.approx(0.0)


def test_later_confirmation_does_not_overwrite_first_confirmation(
  tmp_path, monkeypatch
) -> None:
  monkeypatch.setenv("MJLAB_STAIR_EXPORT_DIR", str(tmp_path))
  env = _make_env()
  logger = stair_sequence_event_logger(None, cast(Any, env))
  asset_cfg = SceneEntityCfg("robot", body_ids=[0, 1])

  _step(logger, env, asset_cfg)

  env.extras[STAIR_ORACLE_EXPECTED_RISER_CONTACT_KEY][:] = True
  env.extras[STAIR_ORACLE_EXPECTED_RISER_LAYER_KEY][:] = 2
  _step(logger, env, asset_cfg)

  env.extras[STAIR_CONFIRMATION_CONDITION_KEY][:] = True
  env.extras[STAIR_DEPTH_CONFIRMATION_EVENT_KEY][:] = True
  env.extras[STAIR_CONFIRMATION_LAYER_KEY][:] = 2
  env.extras[STAIR_CONFIRMATION_CONTACT_SEQUENCE_KEY][:] = 7
  _step(logger, env, asset_cfg)

  env.extras[STAIR_CONFIRMATION_CONDITION_KEY][:] = False
  env.extras[STAIR_DEPTH_CONFIRMATION_EVENT_KEY][:] = False
  _step(logger, env, asset_cfg)

  env.extras[STAIR_CONFIRMATION_CONDITION_KEY][:] = True
  env.extras[STAIR_DEPTH_CONFIRMATION_EVENT_KEY][:] = True
  env.extras[STAIR_CONFIRMATION_LAYER_KEY][:] = 3
  _step(logger, env, asset_cfg)

  env.extras[STAIR_PHASE_KEY][:] = 0
  env.extras[STAIR_EXIT_EVENT_KEY][:] = True
  _step(logger, env, asset_cfg)
  logger.flush()

  with (tmp_path / "stair_sequences.csv").open(newline="") as stream:
    sequences = list(csv.DictReader(stream))
  with (tmp_path / "stair_events.csv").open(newline="") as stream:
    events = list(csv.DictReader(stream))

  assert len(sequences) == 1
  first = sequences[0]
  assert first["first_confirmation_time"] == "0.04"
  assert first["confirmation_layer"] == "2"
  assert first["confirmation_correct"] == "1"
  assert first["wrong_layer_confirmation_count"] == "0"
  assert first["duplicate_confirmation_count"] == "0"
  assert [event["event_type"] for event in events].count("confirmation") == 1
  assert [event["event_type"] for event in events].count("confirmation_duplicate") == 0
  assert [event["event_type"] for event in events].count("confirmation_new_layer") == 1
