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
  STAIR_CURRENT_CONTACT_DURATION_KEY,
  STAIR_CURRENT_GROUND_CONTACT_KEY,
  STAIR_CURRENT_STAIR_SUPPORT_KEY,
  STAIR_CURRENT_SUPPORT_FRACTION_KEY,
  STAIR_CURRENT_SUPPORT_LAYER_KEY,
  STAIR_DEPTH_CONFIRMATION_EVENT_KEY,
  STAIR_EXIT_EVENT_KEY,
  STAIR_EXPECTED_LAYER_KEY,
  STAIR_GEOMETRY_OCCUPANCY_KEY,
  STAIR_ORACLE_CONTACT_FORCE_BY_FOOT_KEY,
  STAIR_ORACLE_CONTACT_LAYER_BY_FOOT_KEY,
  STAIR_ORACLE_CONTACT_NORMAL_BY_FOOT_KEY,
  STAIR_ORACLE_CONTACT_POINT_BY_FOOT_KEY,
  STAIR_ORACLE_CONTACT_S_BY_FOOT_KEY,
  STAIR_ORACLE_CONTACT_SEQUENCE_BY_FOOT_KEY,
  STAIR_ORACLE_CONTACT_VALID_BY_FOOT_KEY,
  STAIR_ORACLE_EXPECTED_RISER_CONTACT_KEY,
  STAIR_ORACLE_EXPECTED_RISER_LAYER_KEY,
  STAIR_ORACLE_FOOT_CENTER_S_BY_FOOT_KEY,
  STAIR_ORACLE_HEEL_S_BY_FOOT_KEY,
  STAIR_ORACLE_RISER_S_BY_FOOT_KEY,
  STAIR_ORACLE_ROOT_S_BY_FOOT_KEY,
  STAIR_ORACLE_TOE_S_BY_FOOT_KEY,
  STAIR_PHASE_KEY,
  STAIR_RISER_HEIGHT_LABEL_KEY,
  STAIR_SEQUENCE_ID_KEY,
  STAIR_TARGET_FOOT_KEY,
  STAIR_TOE_RISER_HIT_LAYER_KEY,
  STAIR_TREAD_DEPTH_LABEL_KEY,
  TOE_RISER_NEW_HIT_KEY,
)
from mjlab.tasks.velocity.mdp.stair_sequence_logging import (
  stair_sequence_event_logger,
)
from mjlab.tasks.velocity.mdp.temporal_stair_rewards import (
  toe_step_riser_slab_penalty,
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
      body_link_lin_vel_w=torch.zeros(1, 2, 3),
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
    STAIR_CURRENT_CONTACT_DURATION_KEY: torch.tensor([[0.1, 0.1]]),
    STAIR_CURRENT_STAIR_SUPPORT_KEY: torch.tensor([[True, True]]),
    STAIR_CURRENT_SUPPORT_FRACTION_KEY: torch.tensor([[1.0, 1.0]]),
    STAIR_CURRENT_SUPPORT_LAYER_KEY: torch.tensor([[1, 2]]),
    STAIR_ADJACENT_PAIR_VALID_KEY: torch.tensor([False]),
    STAIR_ADJACENT_PAIR_DEPTH_KEY: torch.tensor([0.0]),
    STAIR_ADJACENT_PAIR_HEIGHT_KEY: torch.tensor([0.0]),
    STAIR_ORACLE_EXPECTED_RISER_CONTACT_KEY: torch.tensor([False]),
    STAIR_ORACLE_EXPECTED_RISER_LAYER_KEY: torch.tensor([-1]),
    STAIR_ORACLE_CONTACT_VALID_BY_FOOT_KEY: torch.tensor([[False, False]]),
    STAIR_ORACLE_CONTACT_LAYER_BY_FOOT_KEY: torch.tensor([[-1, -1]]),
    STAIR_ORACLE_CONTACT_SEQUENCE_BY_FOOT_KEY: torch.tensor([[-1, -1]]),
    STAIR_ORACLE_CONTACT_POINT_BY_FOOT_KEY: torch.full((1, 2, 3), torch.nan),
    STAIR_ORACLE_CONTACT_NORMAL_BY_FOOT_KEY: torch.full((1, 2, 3), torch.nan),
    STAIR_ORACLE_CONTACT_FORCE_BY_FOOT_KEY: torch.full((1, 2), torch.nan),
    STAIR_ORACLE_CONTACT_S_BY_FOOT_KEY: torch.full((1, 2), torch.nan),
    STAIR_ORACLE_TOE_S_BY_FOOT_KEY: torch.full((1, 2), torch.nan),
    STAIR_ORACLE_HEEL_S_BY_FOOT_KEY: torch.full((1, 2), torch.nan),
    STAIR_ORACLE_FOOT_CENTER_S_BY_FOOT_KEY: torch.full((1, 2), torch.nan),
    STAIR_ORACLE_RISER_S_BY_FOOT_KEY: torch.full((1, 2), torch.nan),
    STAIR_ORACLE_ROOT_S_BY_FOOT_KEY: torch.full((1, 2), torch.nan),
    STAIR_CONFIRMATION_CONDITION_KEY: torch.tensor([False]),
    STAIR_DEPTH_CONFIRMATION_EVENT_KEY: torch.tensor([False]),
    STAIR_CONFIRMATION_LAYER_KEY: torch.tensor([-1]),
    STAIR_CONFIRMATION_CONTACT_SEQUENCE_KEY: torch.tensor([-1]),
    STAIR_GEOMETRY_OCCUPANCY_KEY: torch.tensor([True]),
    TOE_RISER_NEW_HIT_KEY: torch.tensor([False]),
    STAIR_TOE_RISER_HIT_LAYER_KEY: torch.tensor([-1]),
    STAIR_EXIT_EVENT_KEY: torch.tensor([False]),
    STAIR_TARGET_FOOT_KEY: torch.tensor([0]),
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
  **kwargs: Any,
) -> None:
  env.extras["log"] = {}
  logger(cast(Any, env), asset_cfg=asset_cfg, **kwargs)
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
  env.extras[STAIR_ORACLE_CONTACT_VALID_BY_FOOT_KEY][0, 0] = True
  env.extras[STAIR_ORACLE_CONTACT_LAYER_BY_FOOT_KEY][0, 0] = 2
  env.extras[STAIR_ORACLE_CONTACT_SEQUENCE_BY_FOOT_KEY][0, 0] = 7
  env.extras[STAIR_ORACLE_CONTACT_POINT_BY_FOOT_KEY][0, 0] = torch.tensor(
    [0.6, 0.0, 0.2]
  )
  env.extras[STAIR_ORACLE_CONTACT_NORMAL_BY_FOOT_KEY][0, 0] = torch.tensor(
    [-1.0, 0.0, 0.0]
  )
  env.extras[STAIR_ORACLE_CONTACT_FORCE_BY_FOOT_KEY][0, 0] = 20.0
  env.extras[STAIR_ORACLE_CONTACT_S_BY_FOOT_KEY][0, 0] = 0.6
  env.extras[STAIR_ORACLE_TOE_S_BY_FOOT_KEY][0, 0] = 0.61
  env.extras[STAIR_ORACLE_HEEL_S_BY_FOOT_KEY][0, 0] = 0.41
  env.extras[STAIR_ORACLE_FOOT_CENTER_S_BY_FOOT_KEY][0, 0] = 0.51
  env.extras[STAIR_ORACLE_RISER_S_BY_FOOT_KEY][0, 0] = 0.6
  env.extras[STAIR_ORACLE_ROOT_S_BY_FOOT_KEY][0, 0] = 0.3
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
  assert [event["event_type"] for event in events].count(
    "confirmation_contact_proxy"
  ) == 1
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


def test_support_trajectory_records_sparse_ordered_episode(
  tmp_path, monkeypatch
) -> None:
  monkeypatch.setenv("MJLAB_STAIR_EXPORT_DIR", str(tmp_path))
  env = _make_env()
  logger = stair_sequence_event_logger(None, cast(Any, env))
  asset_cfg = SceneEntityCfg("robot", body_ids=[0, 1])

  _step(logger, env, asset_cfg, stable_frames=2)
  env.extras[STAIR_ADJACENT_PAIR_VALID_KEY][:] = True
  env.extras[STAIR_ADJACENT_PAIR_DEPTH_KEY][:] = 0.28
  env.extras[STAIR_ADJACENT_PAIR_HEIGHT_KEY][:] = 0.18
  for quality in (0.55, 0.80, 0.95, 1.0, 1.0):
    env.extras[STAIR_CURRENT_SUPPORT_FRACTION_KEY][:] = quality
    _step(logger, env, asset_cfg, stable_frames=2)

  env.extras[STAIR_ADJACENT_PAIR_VALID_KEY][:] = False
  _step(logger, env, asset_cfg, stable_frames=2)
  env.extras[STAIR_ADJACENT_PAIR_VALID_KEY][:] = True
  env.extras[STAIR_CURRENT_SUPPORT_FRACTION_KEY][:] = 0.60
  _step(logger, env, asset_cfg, stable_frames=2)
  env.extras[STAIR_ADJACENT_PAIR_VALID_KEY][:] = False
  _step(logger, env, asset_cfg, stable_frames=2)
  env.extras[STAIR_PHASE_KEY][:] = 0
  _step(logger, env, asset_cfg, stable_frames=2)
  logger.flush()

  with (tmp_path / "stair_events.csv").open(newline="") as stream:
    events = list(csv.DictReader(stream))
  with (tmp_path / "stair_sequences.csv").open(newline="") as stream:
    sequences = list(csv.DictReader(stream))

  support_events = [
    event for event in events if event["event_family"] == "support_trajectory"
  ]
  event_types = [event["event_type"] for event in support_events]
  assert event_types[:6] == [
    "pair_enter",
    "pair_quality_cross_0.50",
    "pair_quality_cross_0.75",
    "pair_quality_cross_0.90",
    "pair_full",
    "pair_stable_for_N_frames",
  ]
  assert event_types.count("pair_enter") == 2
  assert event_types.count("pair_exit") == 2
  assert event_types.count("pair_peak_quality") == 2
  assert event_types.count("pair_quality_cross_0.50") == 2
  assert event_types.count("pair_quality_cross_0.75") == 1
  assert event_types.count("pair_quality_cross_0.90") == 1
  assert event_types.count("pair_full") == 1
  assert event_types.count("pair_stable_for_N_frames") == 1
  event_ids = [int(event["event_id"]) for event in events]
  assert event_ids == sorted(set(event_ids))
  peaks = [
    float(event["peak_pair_quality"])
    for event in support_events
    if event["event_type"] == "pair_peak_quality"
  ]
  assert peaks == pytest.approx([1.0, 0.60])
  assert sequences[0]["logger_version"] == "2"
  assert sequences[0]["pair_episode_count"] == "2"
  assert sequences[0]["pair_stable_event_count"] == "1"


def test_oracle_contact_is_per_foot_layer_and_rehit(tmp_path, monkeypatch) -> None:
  monkeypatch.setenv("MJLAB_STAIR_EXPORT_DIR", str(tmp_path))
  env = _make_env()
  logger = stair_sequence_event_logger(None, cast(Any, env))
  asset_cfg = SceneEntityCfg("robot", body_ids=[0, 1])
  valid = env.extras[STAIR_ORACLE_CONTACT_VALID_BY_FOOT_KEY]
  layer = env.extras[STAIR_ORACLE_CONTACT_LAYER_BY_FOOT_KEY]
  sequence = env.extras[STAIR_ORACLE_CONTACT_SEQUENCE_BY_FOOT_KEY]
  point = env.extras[STAIR_ORACLE_CONTACT_POINT_BY_FOOT_KEY]
  normal = env.extras[STAIR_ORACLE_CONTACT_NORMAL_BY_FOOT_KEY]
  force = env.extras[STAIR_ORACLE_CONTACT_FORCE_BY_FOOT_KEY]
  contact_s = env.extras[STAIR_ORACLE_CONTACT_S_BY_FOOT_KEY]
  toe_s = env.extras[STAIR_ORACLE_TOE_S_BY_FOOT_KEY]
  heel_s = env.extras[STAIR_ORACLE_HEEL_S_BY_FOOT_KEY]
  center_s = env.extras[STAIR_ORACLE_FOOT_CENTER_S_BY_FOOT_KEY]
  riser_s = env.extras[STAIR_ORACLE_RISER_S_BY_FOOT_KEY]
  root_s = env.extras[STAIR_ORACLE_ROOT_S_BY_FOOT_KEY]

  _step(logger, env, asset_cfg, oracle_rehit_cooldown_frames=1)
  valid[0, 0] = True
  layer[0, 0] = 1
  _step(logger, env, asset_cfg, oracle_rehit_cooldown_frames=1)
  valid[0, 0] = False
  _step(logger, env, asset_cfg, oracle_rehit_cooldown_frames=1)

  valid[0, 0] = True
  layer[0, 0] = 2
  sequence[0, 0] = 7
  point[0, 0] = torch.tensor([0.60, 0.0, 0.20])
  normal[0, 0] = torch.tensor([-1.0, 0.0, 0.0])
  force[0, 0] = 30.0
  contact_s[0, 0] = 0.60
  toe_s[0, 0] = 0.62
  heel_s[0, 0] = 0.42
  center_s[0, 0] = 0.52
  riser_s[0, 0] = 0.60
  root_s[0, 0] = 0.30
  _step(logger, env, asset_cfg, oracle_rehit_cooldown_frames=1)
  _step(logger, env, asset_cfg, oracle_rehit_cooldown_frames=1)
  valid[0, 0] = False
  _step(logger, env, asset_cfg, oracle_rehit_cooldown_frames=1)
  valid[0, 0] = True
  _step(logger, env, asset_cfg, oracle_rehit_cooldown_frames=1)

  layer[0, 0] = 3
  point[0, 0] = torch.nan
  normal[0, 0] = torch.nan
  _step(logger, env, asset_cfg, oracle_rehit_cooldown_frames=1)
  valid[0, 1] = True
  layer[0, 1] = 2
  sequence[0, 1] = 7
  point[0, 1] = torch.tensor([0.60, 0.0, 0.20])
  normal[0, 1] = torch.tensor([-1.0, 0.0, 0.0])
  _step(logger, env, asset_cfg, oracle_rehit_cooldown_frames=1)
  env.extras[STAIR_PHASE_KEY][:] = 0
  _step(logger, env, asset_cfg, oracle_rehit_cooldown_frames=1)
  logger.flush()

  with (tmp_path / "stair_events.csv").open(newline="") as stream:
    events = list(csv.DictReader(stream))
  oracle_events = [
    event for event in events if event["event_family"] == "oracle_contact"
  ]
  assert [event["event_type"] for event in oracle_events].count(
    "oracle_riser_contact"
  ) == 3
  assert [event["event_type"] for event in oracle_events].count(
    "oracle_riser_contact_rehit"
  ) == 1
  assert all(int(event["event_layer"]) >= 2 for event in oracle_events)
  assert {(event["foot_id"], event["event_layer"]) for event in oracle_events} >= {
    ("0", "2"),
    ("0", "3"),
    ("1", "2"),
  }
  layer2_left = next(
    event
    for event in oracle_events
    if event["foot_id"] == "0" and event["event_layer"] == "2"
  )
  assert layer2_left["contact_valid"] == "1"
  assert float(layer2_left["contact_s_minus_riser_s"]) == pytest.approx(0.0)
  layer3_left = next(
    event
    for event in oracle_events
    if event["foot_id"] == "0" and event["event_layer"] == "3"
  )
  assert layer3_left["contact_valid"] == "0"
  assert layer3_left["contact_point_x"] == ""
  assert float(layer3_left["contact_point_s"]) == pytest.approx(0.60)


def test_oracle_contact_projection_uses_contact_and_riser_geometry() -> None:
  term = object.__new__(toe_step_riser_slab_penalty)
  term._sequence_id = torch.tensor([7])
  term._stair_phase = torch.tensor([2])
  term._ascent_dir = torch.tensor([[1.0, 0.0]])
  term._oracle_contact_valid_by_foot = torch.zeros(1, 2, dtype=torch.bool)
  term._oracle_contact_layer_by_foot = torch.full((1, 2), -1, dtype=torch.long)
  term._oracle_contact_sequence_by_foot = torch.full((1, 2), -1, dtype=torch.long)
  term._oracle_contact_point_by_foot = torch.full((1, 2, 3), torch.nan)
  term._oracle_contact_normal_by_foot = torch.full((1, 2, 3), torch.nan)
  term._oracle_contact_force_by_foot = torch.full((1, 2), torch.nan)
  term._oracle_contact_s_by_foot = torch.full((1, 2), torch.nan)
  term._oracle_toe_s_by_foot = torch.full((1, 2), torch.nan)
  term._oracle_heel_s_by_foot = torch.full((1, 2), torch.nan)
  term._oracle_foot_center_s_by_foot = torch.full((1, 2), torch.nan)
  term._oracle_riser_s_by_foot = torch.full((1, 2), torch.nan)
  term._oracle_root_s_by_foot = torch.full((1, 2), torch.nan)
  boundaries = torch.zeros(1, 2, 11)
  boundaries[0, 0, 0] = 0.30
  boundaries[0, 1, 0] = 0.60
  event_contact = torch.tensor([[[True, True], [False, False]]])
  layers = torch.tensor([[[1, 2], [0, 0]]])
  boundary_idx = torch.tensor([[[0, 1], [0, 0]]])
  contact_sequence = torch.tensor([[[7, 7], [-1, -1]]])
  contact_pos = torch.tensor(
    [[[[0.30, 0.0, 0.10], [0.61, 0.0, 0.20]], [[0.0, 0.0, 0.0]] * 2]]
  )
  contact_normal = torch.tensor(
    [[[[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]] * 2]]
  )
  contact_force = torch.tensor(
    [[[[10.0, 0.0, 0.0], [25.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]] * 2]]
  )
  toe_points = torch.tensor(
    [[[[0.62, 0.0, 0.0], [0.64, 0.0, 0.0]], [[0.0, 0.0, 0.0]] * 2]]
  )
  sole_points = torch.tensor(
    [[[[0.40, 0.0, 0.0], [0.64, 0.0, 0.0]], [[0.0, 0.0, 0.0]] * 2]]
  )
  foot_pos = torch.tensor([[[0.52, 0.0, 0.2], [0.0, 0.0, 0.0]]])
  root_pos = torch.tensor([[0.25, 0.0, 0.8]])

  term._update_oracle_contact_diagnostics(
    event_boundary_contact=event_contact,
    contact_layers=layers,
    contact_boundary_idx=boundary_idx,
    contact_sequence_ids=contact_sequence,
    contact_pos_w=contact_pos,
    contact_normal_w=contact_normal,
    contact_force_w=contact_force,
    boundaries=boundaries,
    toe_points_w=toe_points,
    sole_points_w=sole_points,
    foot_pos_w=foot_pos,
    root_pos_w=root_pos,
  )

  assert term._oracle_contact_valid_by_foot.tolist() == [[True, False]]
  assert term._oracle_contact_layer_by_foot.tolist() == [[2, -1]]
  assert term._oracle_contact_s_by_foot[0, 0].item() == pytest.approx(0.61)
  assert term._oracle_riser_s_by_foot[0, 0].item() == pytest.approx(0.60)
  assert term._oracle_toe_s_by_foot[0, 0].item() == pytest.approx(0.64)
  assert term._oracle_heel_s_by_foot[0, 0].item() == pytest.approx(0.40)
  assert term._oracle_foot_center_s_by_foot[0, 0].item() == pytest.approx(0.52)
  assert term._oracle_root_s_by_foot[0, 0].item() == pytest.approx(0.25)
  assert torch.isnan(term._oracle_contact_point_by_foot[0, 1]).all()
