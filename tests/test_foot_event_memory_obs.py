"""Tests for deployable-shaped foot event memory observations."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from mjlab.tasks.velocity.mdp.observations import (
  FOOT_EVENT_GEOMETRY_STATS_START,
  FOOT_EVENT_MEMORY_OBS_DIM,
  FOOT_EVENT_PAIR_FEATURE_DIM,
  FOOT_EVENT_RATCHET_DIM,
  FOOT_EVENT_RATCHET_START,
  FOOT_EVENT_SUMMARY_DIM,
  FOOT_EVENT_TOE_SUMMARY_START,
  FOOT_EVENT_TREND_DIM,
  FootEventMemoryObs,
  foot_event_memory_obs_dim,
)
from mjlab.tasks.velocity.mdp.stair_geometry import (
  STAIR_CURRENT_CONTACT_DURATION_KEY,
  STAIR_CURRENT_GROUND_CONTACT_KEY,
  STAIR_STRIDE_EVENT_BACKOFF_TOUCHDOWN,
  STAIR_STRIDE_EVENT_LOCK_TOUCHDOWN,
  STAIR_STRIDE_EVENT_NONE,
  STAIR_STRIDE_EVENT_PROBE_TOUCHDOWN,
  STAIR_STRIDE_EVENT_VALID_SECOND_HIT,
  STAIR_STRIDE_PHASE_EVENT_ACTUAL_KEY,
  STAIR_STRIDE_PHASE_EVENT_COLLISION_PENALTY_KEY,
  STAIR_STRIDE_PHASE_EVENT_COMPLETED_KEY,
  STAIR_STRIDE_PHASE_EVENT_ID_KEY,
  STAIR_STRIDE_PHASE_EVENT_PREVIOUS_KEY,
  STAIR_STRIDE_PHASE_EVENT_TARGET_KEY,
  STAIR_STRIDE_PHASE_EVENT_TYPE_KEY,
  TOE_RISER_NEW_HIT_BY_FOOT_KEY,
)


class _CommandManager:
  def __init__(self) -> None:
    self.command = torch.tensor([[0.5, 0.0, 0.0]])

  def get_command(self, name: str) -> torch.Tensor:
    assert name == "twist"
    return self.command


def _make_env() -> Any:
  site_pos_w = torch.tensor(
    [
      [
        [0.2, 0.1, 0.3],
        [0.4, -0.1, 0.2],
        [0.0, 0.1, 0.3],
        [0.2, -0.1, 0.2],
      ]
    ],
    dtype=torch.float32,
  )
  robot = SimpleNamespace(
    site_names=["left_toe", "right_toe", "left_heel", "right_heel"],
    data=SimpleNamespace(
      root_link_pos_w=torch.zeros(1, 3),
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
      site_pos_w=site_pos_w,
    ),
  )
  return SimpleNamespace(
    num_envs=1,
    device=torch.device("cpu"),
    step_dt=0.02,
    episode_length_buf=torch.tensor([0]),
    scene={"robot": robot},
    command_manager=_CommandManager(),
    extras={
      STAIR_CURRENT_GROUND_CONTACT_KEY: torch.tensor([[False, False]]),
      STAIR_CURRENT_CONTACT_DURATION_KEY: torch.tensor([[0.0, 0.0]]),
      TOE_RISER_NEW_HIT_BY_FOOT_KEY: torch.tensor([[False, False]]),
    },
  )


def _make_term(
  env: Any,
  *,
  extra_params: dict[str, Any] | None = None,
) -> FootEventMemoryObs:
  params = {
    "memory_len": 3,
    "noise_enabled": True,
    "age_norm_s": 1.0,
    "stance_age_norm_s": 1.0,
    "release_confirm_frames": 1,
    "touchdown_miss_prob": 0.0,
    "touchdown_false_positive_prob": 0.0,
    "toe_miss_prob": 0.0,
    "toe_false_positive_prob": 0.0,
    "contact_true_prob_range": (1.0, 1.0),
    "contact_false_prob_range": (0.0, 0.0),
    "touchdown_confidence_range": (1.0, 1.0),
    "touchdown_prob_range": (1.0, 1.0),
    "footprint_xy_noise_range_m": (0.0, 0.0),
    "footprint_z_noise_range_m": (0.0, 0.0),
    "toe_confidence_range": (0.5, 0.5),
    "toe_hit_prob_range": (0.7, 0.7),
    "toe_xy_noise_range_m": (0.0, 0.0),
    "toe_z_noise_range_m": (0.0, 0.0),
  }
  if extra_params is not None:
    params.update(extra_params)
  return FootEventMemoryObs(SimpleNamespace(params=params), env)


def _seed_three_footprint_sequence(
  term: FootEventMemoryObs,
  *,
  same_stride: float,
  first_rise: float = 0.15,
  second_rise: float = 0.15,
  older_s: float = 0.0,
) -> None:
  term.footprint_valid.zero_()
  term.footprints.zero_()
  term.footprint_valid[0, :3] = True
  newer, middle, older = term.footprints[0, :3]
  for footprint in (newer, middle, older):
    footprint[0] = 1.0
    footprint[3] = 1.0
    footprint[5] = 1.0
  newer[2] = 1.0
  middle[1] = 1.0
  older[2] = 1.0
  older[9] = 0.0
  older[10] = older_s
  middle[9] = first_rise
  middle[10] = older_s + 0.5 * same_stride
  newer[9] = first_rise + second_rise
  newer[10] = older_s + same_stride


def test_foot_event_memory_dim_matches_default_layout() -> None:
  assert FOOT_EVENT_TREND_DIM == 10
  assert FOOT_EVENT_RATCHET_DIM == 10
  assert FOOT_EVENT_SUMMARY_DIM == 80
  assert FOOT_EVENT_MEMORY_OBS_DIM == 278
  assert foot_event_memory_obs_dim() == 278
  assert foot_event_memory_obs_dim(memory_len=3) == 179
  assert foot_event_memory_obs_dim(memory_len=3, include_raw_memory=False) == 80
  assert foot_event_memory_obs_dim(memory_len=3, include_summary=False) == 99


def test_foot_event_memory_can_return_summary_only_obs() -> None:
  env = _make_env()
  term = _make_term(env, extra_params={"include_raw_memory": False})

  obs = term(env)

  assert obs.shape == (1, FOOT_EVENT_SUMMARY_DIM)


def test_foot_event_memory_updates_at_most_once_per_control_step() -> None:
  env = _make_env()
  term = _make_term(env)

  first = term(env)
  env.extras[STAIR_CURRENT_GROUND_CONTACT_KEY] = torch.tensor([[True, False]])
  env.extras[STAIR_CURRENT_CONTACT_DURATION_KEY] = torch.tensor([[0.02, 0.0]])
  cached = term(env)

  assert cached is first
  assert not term.footprint_valid.any()

  env.episode_length_buf += 1
  term(env)
  assert term.footprint_valid.tolist() == [[True, False, False]]


def test_foot_event_memory_publishes_stride_phase_event_packet() -> None:
  env = _make_env()
  term = _make_term(env)

  assert env.extras[STAIR_STRIDE_PHASE_EVENT_ID_KEY] is (
    term.ratchet_stride_phase_event_id
  )
  assert env.extras[STAIR_STRIDE_PHASE_EVENT_TYPE_KEY] is (
    term.ratchet_stride_phase_event_type
  )
  assert env.extras[STAIR_STRIDE_PHASE_EVENT_ACTUAL_KEY] is (
    term.ratchet_stride_phase_event_actual_s
  )
  assert env.extras[STAIR_STRIDE_PHASE_EVENT_PREVIOUS_KEY] is (
    term.ratchet_stride_phase_event_previous_s
  )
  assert env.extras[STAIR_STRIDE_PHASE_EVENT_TARGET_KEY] is (
    term.ratchet_stride_phase_event_target_s
  )
  assert env.extras[STAIR_STRIDE_PHASE_EVENT_COLLISION_PENALTY_KEY] is (
    term.ratchet_stride_phase_event_collision_penalty
  )
  assert env.extras[STAIR_STRIDE_PHASE_EVENT_COMPLETED_KEY] is (
    term.ratchet_stride_phase_event_completed
  )
  assert term.ratchet_stride_phase_event_id.tolist() == [0]
  assert term.ratchet_stride_phase_event_type.tolist() == [STAIR_STRIDE_EVENT_NONE]


def test_foot_event_memory_records_latched_touchdown() -> None:
  env = _make_env()
  term = _make_term(env)

  obs = term(env)
  assert obs.shape == (1, 179)
  assert not term.footprint_valid.any()

  env.episode_length_buf += 1
  env.extras[STAIR_CURRENT_GROUND_CONTACT_KEY] = torch.tensor([[True, False]])
  env.extras[STAIR_CURRENT_CONTACT_DURATION_KEY] = torch.tensor([[0.02, 0.0]])
  term(env)

  footprint = term.footprints[0, 0]
  assert term.footprint_valid.tolist() == [[True, False, False]]
  assert footprint[0].item() == 1.0
  assert footprint[1].item() == 1.0
  assert footprint[3].item() == 1.0
  assert footprint[4].item() == 0.0
  torch.testing.assert_close(footprint[7], torch.tensor(0.1))
  torch.testing.assert_close(footprint[8], torch.tensor(0.1))
  torch.testing.assert_close(footprint[14], torch.tensor(1.0))
  torch.testing.assert_close(footprint[15], torch.tensor(1.0))

  env.episode_length_buf += 1
  term(env)
  assert term.footprint_valid.tolist() == [[True, False, False]]
  assert term.footprints[0, 1, 0].item() == 0.0

  env.scene["robot"].data.root_link_pos_w[:, 0] = 0.05
  env.episode_length_buf += 1
  term(env)
  torch.testing.assert_close(term.footprints[0, 0, 7], torch.tensor(0.05))


def test_foot_event_memory_records_toe_mark_without_fill() -> None:
  env = _make_env()
  term = _make_term(env)
  raw_dim = foot_event_memory_obs_dim(memory_len=3, include_summary=False)
  term(env)

  env.episode_length_buf += 1
  env.extras[TOE_RISER_NEW_HIT_BY_FOOT_KEY] = torch.tensor([[False, True]])
  obs = term(env)

  toe = term.toe_marks[0, 0]
  assert term.toe_valid.tolist() == [[True, False, False]]
  assert toe[0].item() == 1.0
  assert toe[2].item() == 1.0
  assert toe[3].item() == 1.0
  torch.testing.assert_close(toe[6], torch.tensor(0.4))
  torch.testing.assert_close(toe[9], torch.tensor(0.4))
  torch.testing.assert_close(toe[13], torch.tensor(0.7))
  torch.testing.assert_close(toe[15], torch.tensor(1.0))
  summary = obs[0, raw_dim:]
  toe_start = FOOT_EVENT_TOE_SUMMARY_START
  torch.testing.assert_close(summary[toe_start + 0], torch.tensor(1.0))
  torch.testing.assert_close(summary[toe_start + 4], torch.tensor(0.4))
  torch.testing.assert_close(summary[toe_start + 5], torch.tensor(0.2))


def test_foot_event_memory_summary_tracks_adjacent_footprints() -> None:
  env = _make_env()
  term = _make_term(env)
  raw_dim = foot_event_memory_obs_dim(memory_len=3, include_summary=False)
  term(env)

  env.episode_length_buf += 1
  env.extras[STAIR_CURRENT_GROUND_CONTACT_KEY] = torch.tensor([[True, False]])
  env.extras[STAIR_CURRENT_CONTACT_DURATION_KEY] = torch.tensor([[0.02, 0.0]])
  term(env)

  env.episode_length_buf += 1
  env.extras[STAIR_CURRENT_GROUND_CONTACT_KEY] = torch.tensor([[False, False]])
  env.extras[STAIR_CURRENT_CONTACT_DURATION_KEY] = torch.tensor([[0.0, 0.0]])
  term(env)

  env.episode_length_buf += 1
  env.extras[STAIR_CURRENT_GROUND_CONTACT_KEY] = torch.tensor([[False, True]])
  env.extras[STAIR_CURRENT_CONTACT_DURATION_KEY] = torch.tensor([[0.0, 0.02]])
  obs = term(env)
  summary = obs[0, raw_dim:]

  torch.testing.assert_close(summary[0], torch.tensor(1.0))
  torch.testing.assert_close(summary[2], torch.tensor(1.0))
  torch.testing.assert_close(summary[3], torch.tensor(1.0))
  torch.testing.assert_close(summary[5], torch.tensor(0.2))
  torch.testing.assert_close(summary[6], torch.tensor(-0.1))
  torch.testing.assert_close(summary[8], torch.tensor(0.2))
  torch.testing.assert_close(summary[9], torch.tensor(1.0))
  stats = summary[FOOT_EVENT_GEOMETRY_STATS_START:FOOT_EVENT_RATCHET_START]
  torch.testing.assert_close(stats[0], torch.tensor(0.2))
  torch.testing.assert_close(stats[3], torch.tensor(0.2))
  torch.testing.assert_close(stats[4], torch.tensor(-0.1))
  torch.testing.assert_close(stats[7], torch.tensor(0.2))
  torch.testing.assert_close(stats[8], torch.tensor(0.0))


def test_foot_event_memory_logs_summary_validation_metrics() -> None:
  env = _make_env()
  env.extras["log"] = {}
  term = _make_term(env)
  term(env)

  env.episode_length_buf += 1
  env.extras[STAIR_CURRENT_GROUND_CONTACT_KEY] = torch.tensor([[True, False]])
  env.extras[STAIR_CURRENT_CONTACT_DURATION_KEY] = torch.tensor([[0.02, 0.0]])
  term(env)

  env.episode_length_buf += 1
  env.extras[STAIR_CURRENT_GROUND_CONTACT_KEY] = torch.tensor([[False, False]])
  env.extras[STAIR_CURRENT_CONTACT_DURATION_KEY] = torch.tensor([[0.0, 0.0]])
  term(env)

  env.episode_length_buf += 1
  env.extras[STAIR_CURRENT_GROUND_CONTACT_KEY] = torch.tensor([[False, True]])
  env.extras[STAIR_CURRENT_CONTACT_DURATION_KEY] = torch.tensor([[0.0, 0.02]])
  term(env)

  log = env.extras["log"]
  assert "Metrics/foot_event_summary_pair_valid_ratio" in log
  assert "Metrics/foot_event_summary_latest_delta_s_mean" in log
  assert "Metrics/foot_event_summary_max_forward_stride_mean" in log
  torch.testing.assert_close(
    log["Metrics/foot_event_summary_pair_valid_ratio"],
    torch.tensor(0.2),
  )
  assert log["Metrics/foot_event_summary_new_footprint_ratio"].item() == 1.0


def test_foot_event_memory_logs_same_foot_stride_monotonic_metrics() -> None:
  env = _make_env()
  env.extras["log"] = {}
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 5,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
      }
    ),
    env,
  )

  term.footprint_valid[0, :5] = True
  # Newest -> oldest touchdown order.  Same-foot swing strides toward the
  # present are 0.45, 0.40, 0.35, so each newer step is farther.
  foot_is_left = [False, True, False, True, False]
  s_values = torch.tensor([1.00, 0.75, 0.55, 0.35, 0.20])
  z_values = torch.tensor([0.40, 0.30, 0.20, 0.10, 0.00])
  for slot, is_left in enumerate(foot_is_left):
    footprint = term.footprints[0, slot]
    footprint[0] = 1.0
    footprint[1] = 1.0 if is_left else 0.0
    footprint[2] = 0.0 if is_left else 1.0
    footprint[3] = 1.0
    footprint[5] = 1.0
    footprint[6] = 0.0
    footprint[9] = z_values[slot]
    footprint[10] = s_values[slot]

  summary = term._compute_event_summary()
  term._log_summary_metrics(
    env,
    summary,
    new_footprint_any=torch.tensor([True]),
    new_toe_mark_any=torch.tensor([False]),
  )

  log = env.extras["log"]
  torch.testing.assert_close(
    log["Metrics/foot_event_summary_same_foot_stride_monotonic_ratio"],
    torch.tensor(1.0),
  )
  torch.testing.assert_close(
    log["Metrics/foot_event_summary_latest_same_foot_stride_growth_mean"],
    torch.tensor(0.05),
  )
  torch.testing.assert_close(
    log["Metrics/foot_event_summary_latest_same_foot_monotonic_ratio"],
    torch.tensor(1.0),
  )
  torch.testing.assert_close(
    log["Metrics/foot_event_summary_latest_same_foot_monotonic_sample_ratio"],
    torch.tensor(1.0),
  )


def test_foot_event_memory_combines_stored_toe_mark_with_later_footprints() -> None:
  env = _make_env()
  term = _make_term(env)
  raw_dim = foot_event_memory_obs_dim(memory_len=3, include_summary=False)
  term(env)

  env.episode_length_buf += 1
  env.extras[TOE_RISER_NEW_HIT_BY_FOOT_KEY] = torch.tensor([[True, False]])
  term(env)
  env.extras[TOE_RISER_NEW_HIT_BY_FOOT_KEY] = torch.tensor([[False, False]])

  env.episode_length_buf += 1
  env.extras[STAIR_CURRENT_GROUND_CONTACT_KEY] = torch.tensor([[True, False]])
  env.extras[STAIR_CURRENT_CONTACT_DURATION_KEY] = torch.tensor([[0.02, 0.0]])
  term(env)

  env.episode_length_buf += 1
  env.extras[STAIR_CURRENT_GROUND_CONTACT_KEY] = torch.tensor([[False, False]])
  env.extras[STAIR_CURRENT_CONTACT_DURATION_KEY] = torch.tensor([[0.0, 0.0]])
  term(env)

  env.episode_length_buf += 1
  env.extras[STAIR_CURRENT_GROUND_CONTACT_KEY] = torch.tensor([[False, True]])
  env.extras[STAIR_CURRENT_CONTACT_DURATION_KEY] = torch.tensor([[0.0, 0.02]])
  obs = term(env)
  summary = obs[0, raw_dim:]
  toe_start = FOOT_EVENT_TOE_SUMMARY_START

  torch.testing.assert_close(summary[toe_start + 0], torch.tensor(1.0))
  torch.testing.assert_close(summary[toe_start + 3], torch.tensor(-1.0))
  torch.testing.assert_close(summary[toe_start + 4], torch.tensor(0.2))
  torch.testing.assert_close(summary[toe_start + 6], torch.tensor(0.0))
  torch.testing.assert_close(summary[toe_start + 7], torch.tensor(0.0))
  torch.testing.assert_close(summary[toe_start + 8], torch.tensor(1.0))
  torch.testing.assert_close(summary[toe_start + 9], torch.tensor(0.1))


def test_foot_event_memory_summary_tracks_stride_trend() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
      }
    ),
    env,
  )

  term.footprint_valid[0, :4] = True
  # newest -> oldest: strides are 0.35, 0.30, 0.20, so the trend is increasing
  # toward the present.
  s_values = torch.tensor([0.90, 0.55, 0.25, 0.05])
  z_values = torch.tensor([0.20, 0.10, 0.00, 0.00])
  for slot in range(4):
    footprint = term.footprints[0, slot]
    footprint[0] = 1.0
    footprint[1] = 1.0 if slot % 2 == 0 else 0.0
    footprint[2] = 1.0 if slot % 2 == 1 else 0.0
    footprint[3] = 1.0
    footprint[5] = 1.0
    footprint[6] = 0.0
    footprint[9] = z_values[slot]
    footprint[10] = s_values[slot]

  summary = term._compute_event_summary()[0]
  pair_dim = FOOT_EVENT_PAIR_FEATURE_DIM

  torch.testing.assert_close(summary[0], torch.tensor(1.0))
  torch.testing.assert_close(summary[5], torch.tensor(0.35))
  torch.testing.assert_close(summary[6], torch.tensor(0.10))
  torch.testing.assert_close(summary[pair_dim + 5], torch.tensor(0.30))
  torch.testing.assert_close(summary[pair_dim + 6], torch.tensor(0.10))
  torch.testing.assert_close(summary[2 * pair_dim + 5], torch.tensor(0.20))
  torch.testing.assert_close(summary[2 * pair_dim + 6], torch.tensor(0.0))

  stats = summary[FOOT_EVENT_GEOMETRY_STATS_START:FOOT_EVENT_RATCHET_START]
  torch.testing.assert_close(stats[0], torch.tensor(0.60))
  torch.testing.assert_close(stats[1], torch.tensor(0.28333333))
  torch.testing.assert_close(stats[2], torch.tensor(0.06666667))
  torch.testing.assert_close(stats[3], torch.tensor(0.35))
  torch.testing.assert_close(stats[4], torch.tensor(0.10))
  torch.testing.assert_close(stats[5], torch.tensor(0.05))
  torch.testing.assert_close(stats[6], torch.tensor(0.0))
  torch.testing.assert_close(stats[7], torch.tensor(0.35))
  torch.testing.assert_close(stats[8], torch.tensor(0.35))
  torch.testing.assert_close(stats[9], torch.tensor(0.10))


def test_foot_event_memory_rejects_unpaired_forward_up_step() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_probe_increment_m": 0.025,
      }
    ),
    env,
  )

  term.footprint_valid[0, :2] = True
  newer = term.footprints[0, 0]
  older = term.footprints[0, 1]
  newer[0] = 1.0
  newer[2] = 1.0
  newer[3] = 1.0
  newer[5] = 1.0
  newer[9] = 0.12
  newer[10] = 0.45
  older[0] = 1.0
  older[1] = 1.0
  older[3] = 1.0
  older[5] = 1.0
  older[9] = 0.0
  older[10] = 0.20

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([True]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet, torch.zeros_like(ratchet))
  assert term.ratchet_sequence_valid[0].item() is False
  assert term.ratchet_sequence_rejected[0].item() is False


def test_foot_event_memory_ratchet_uses_valid_n_to_n_plus_two_stride() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_probe_increment_m": 0.025,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_first_collision_seen[0] = True
  term.ratchet_collision_anchor_active[0] = True
  term.ratchet_collision_anchor_up_steps[0] = 1
  term.ratchet_probe_bootstrap_ready[0] = True
  term.ratchet_validated_stride_s[0] = 0.45
  term.ratchet_probe_target_s[0] = 0.50
  term.ratchet_lower_s[0] = 0.45
  term.ratchet_same_foot_stride_lower_s[0] = 0.45
  term.ratchet_confidence[0] = 0.8
  _seed_three_footprint_sequence(
    term, same_stride=0.50, first_rise=0.10, second_rise=0.10
  )

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([True]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[0], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[1], torch.tensor(0.475))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.525))
  torch.testing.assert_close(ratchet[3], torch.tensor(0.0))
  torch.testing.assert_close(ratchet[4], torch.tensor(0.20))
  torch.testing.assert_close(ratchet[5], torch.tensor(0.475))
  assert term.ratchet_sequence_valid[0].item() is True
  assert term.ratchet_probe_target_advanced[0].item() is True
  torch.testing.assert_close(term.ratchet_validated_stride_s[0], torch.tensor(0.50))


def test_foot_event_memory_rejects_n_to_n_plus_three_sequence() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_probe_increment_m": 0.025,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_first_collision_seen[0] = True
  term.ratchet_collision_anchor_active[0] = True
  term.ratchet_collision_anchor_up_steps[0] = 1
  term.ratchet_probe_bootstrap_ready[0] = True
  term.ratchet_validated_stride_s[0] = 0.60
  term.ratchet_probe_target_s[0] = 0.65
  term.ratchet_lower_s[0] = 0.60
  term.ratchet_same_foot_stride_lower_s[0] = 0.60
  term.ratchet_last_forward_up_stride[0] = 0.60
  term.ratchet_confidence[0] = 0.8
  _seed_three_footprint_sequence(
    term,
    same_stride=0.78,
    first_rise=0.15,
    second_rise=0.30,
  )

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([True]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[0], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[1], torch.tensor(0.60))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.65))
  torch.testing.assert_close(ratchet[5], torch.tensor(0.60))
  assert term.ratchet_sequence_valid[0].item() is False
  assert term.ratchet_sequence_rejected[0].item() is True
  assert env.extras[STAIR_STRIDE_PHASE_EVENT_ID_KEY][0].item() == 0


def test_foot_event_memory_rejects_equal_double_rises_with_validated_riser() -> None:
  env = _make_env()
  term = _make_term(
    env,
    extra_params={
      "memory_len": 6,
      "noise_enabled": False,
      "ratchet_sequence_max_adjacent_height_m": 0.24,
    },
  )
  term.ratchet_active[0] = True
  term.ratchet_first_collision_seen[0] = True
  term.ratchet_collision_anchor_active[0] = True
  term.ratchet_collision_anchor_up_steps[0] = 1
  term.ratchet_probe_bootstrap_ready[0] = True
  term.ratchet_validated_stride_s[0] = 0.60
  term.ratchet_validated_riser_height_s[0] = 0.10
  term.ratchet_probe_target_s[0] = 0.65
  term.ratchet_lower_s[0] = 0.60
  term.ratchet_same_foot_stride_lower_s[0] = 0.60
  term.ratchet_confidence[0] = 0.8
  _seed_three_footprint_sequence(
    term,
    same_stride=0.80,
    first_rise=0.20,
    second_rise=0.20,
  )

  term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([True]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )

  assert term.ratchet_sequence_valid[0].item() is False
  assert term.ratchet_sequence_rejected[0].item() is True
  assert env.extras[STAIR_STRIDE_PHASE_EVENT_ID_KEY][0].item() == 0


def test_foot_event_memory_ratchet_raises_lower_gradually_after_safe_step() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_probe_increment_m": 0.025,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_first_collision_seen[0] = True
  term.ratchet_collision_anchor_active[0] = True
  term.ratchet_collision_anchor_up_steps[0] = 1
  term.ratchet_probe_bootstrap_ready[0] = True
  term.ratchet_validated_stride_s[0] = 0.30
  term.ratchet_lower_s[0] = 0.30
  term.ratchet_probe_target_s[0] = 0.35
  term.ratchet_same_foot_stride_lower_s[0] = 0.30
  term.ratchet_confidence[0] = 0.4
  _seed_three_footprint_sequence(
    term,
    same_stride=0.32,
    first_rise=0.10,
    second_rise=0.10,
  )

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([True]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[0], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[1], torch.tensor(0.32))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.35))
  torch.testing.assert_close(ratchet[5], torch.tensor(0.32))
  torch.testing.assert_close(ratchet[6], torch.tensor(0.0))
  assert term.ratchet_lower_updated[0].item() is True
  assert term.ratchet_lower_target_lag_clamped[0].item() is False
  assert term.ratchet_safe_no_hit_steps[0].item() == 1


def test_foot_event_memory_ratchet_stays_inside_confirmed_interval() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_probe_increment_m": 0.025,
        "ratchet_min_interval_width_m": 0.04,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_interval_confirmed[0] = True
  term.ratchet_lower_s[0] = 0.30
  term.ratchet_probe_target_s[0] = 0.37
  term.ratchet_upper_s[0] = 0.44
  term.ratchet_same_foot_stride_lower_s[0] = 0.30
  term.ratchet_same_foot_stride_upper_s[0] = 0.44
  term.ratchet_confidence[0] = 0.75

  term.footprint_valid[0, :2] = True
  newer = term.footprints[0, 0]
  older = term.footprints[0, 1]
  newer[0] = 1.0
  newer[2] = 1.0
  newer[3] = 1.0
  newer[5] = 1.0
  newer[9] = 0.12
  newer[10] = 0.70
  older[0] = 1.0
  older[1] = 1.0
  older[3] = 1.0
  older[5] = 1.0
  older[9] = 0.0
  older[10] = 0.20

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([True]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[0], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[1], torch.tensor(0.30))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.37))
  torch.testing.assert_close(ratchet[3], torch.tensor(0.44))
  torch.testing.assert_close(ratchet[5], torch.tensor(0.30))
  torch.testing.assert_close(ratchet[6], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[7], torch.tensor(0.44))
  assert term.ratchet_stride_mode[0].item() == 2
  assert term.ratchet_lower_updated[0].item() is False


def test_foot_event_memory_single_collision_does_not_trigger_recovery() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_lower_s[0] = 0.25
  term.ratchet_probe_target_s[0] = 0.45
  term.ratchet_same_foot_stride_lower_s[0] = 0.25
  term.ratchet_last_forward_up_stride[0] = 0.25
  term.ratchet_confidence[0] = 0.4

  term.toe_valid[0, 0] = True
  toe = term.toe_marks[0, 0]
  toe[0] = 1.0
  toe[1] = 1.0
  toe[4] = 0.6
  toe[5] = 0.05
  toe[9] = 0.495

  term.footprint_valid[0, 0] = True
  footprint = term.footprints[0, 0]
  footprint[0] = 1.0
  footprint[1] = 1.0
  footprint[3] = 1.0
  footprint[5] = 1.0
  footprint[6] = 0.20
  footprint[10] = 0.05

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([False]),
    new_toe_mark_any=torch.tensor([True]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[0], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[1], torch.tensor(0.25))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.45))
  torch.testing.assert_close(ratchet[3], torch.tensor(0.34))
  torch.testing.assert_close(ratchet[5], torch.tensor(0.25))
  torch.testing.assert_close(ratchet[6], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[7], torch.tensor(0.34))
  torch.testing.assert_close(ratchet[8], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[9], torch.tensor(0.0))
  torch.testing.assert_close(term.ratchet_collision_upper_s[0], torch.tensor(0.34))
  torch.testing.assert_close(
    term.ratchet_target_to_upper_margin_s[0],
    torch.tensor(-0.11),
  )
  assert term.ratchet_stride_mode[0].item() == 0
  assert term.ratchet_recovery_pending[0].item() is False
  assert term.ratchet_stride_backoff_after_collision[0].item() is False


def test_foot_event_memory_first_collision_activates_memory_without_stride_push() -> (
  None
):
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_first_collision_enters_stair_mode": True,
        "ratchet_single_collision_confirms_interval": False,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_lower_s[0] = 0.25
  term.ratchet_probe_target_s[0] = 0.45
  term.ratchet_same_foot_stride_lower_s[0] = 0.25
  term.ratchet_last_forward_up_stride[0] = 0.25
  term.ratchet_confidence[0] = 0.4

  term.toe_valid[0, 0] = True
  toe = term.toe_marks[0, 0]
  toe[0] = 1.0
  toe[1] = 1.0
  toe[4] = 0.6
  toe[5] = 0.05
  toe[9] = 0.495

  term.footprint_valid[0, 0] = True
  footprint = term.footprints[0, 0]
  footprint[0] = 1.0
  footprint[1] = 1.0
  footprint[3] = 1.0
  footprint[5] = 1.0
  footprint[6] = 0.20
  footprint[10] = 0.05

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([False]),
    new_toe_mark_any=torch.tensor([True]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[0], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.45))
  torch.testing.assert_close(ratchet[3], torch.tensor(0.0))
  torch.testing.assert_close(ratchet[6], torch.tensor(0.0))
  torch.testing.assert_close(ratchet[7], torch.tensor(0.0))
  torch.testing.assert_close(ratchet[8], torch.tensor(0.0))
  torch.testing.assert_close(ratchet[9], torch.tensor(1.0))
  assert term.ratchet_stride_mode[0].item() == 0
  assert term.ratchet_first_collision_stair_signal[0].item() is True
  assert term.ratchet_first_collision_seen[0].item() is True
  assert term.ratchet_collision_accepted[0].item() is False
  assert term.ratchet_collision_soft_upper[0].item() is False
  assert term.ratchet_soft_upper_active[0].item() is False


def test_foot_event_memory_ratchet_first_collision_can_enter_stair_mode() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_first_collision_enters_stair_mode": True,
        "ratchet_single_collision_confirms_interval": False,
      }
    ),
    env,
  )

  term.toe_valid[0, 0] = True
  toe = term.toe_marks[0, 0]
  toe[0] = 1.0
  toe[1] = 1.0
  toe[4] = 0.6
  toe[5] = 0.05
  toe[9] = 0.45

  term.footprint_valid[0, 0] = True
  footprint = term.footprints[0, 0]
  footprint[0] = 1.0
  footprint[1] = 1.0
  footprint[3] = 1.0
  footprint[5] = 1.0
  footprint[6] = 0.20
  footprint[10] = 0.05

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([False]),
    new_toe_mark_any=torch.tensor([True]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[0], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[1], torch.tensor(0.10))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.10))
  torch.testing.assert_close(ratchet[3], torch.tensor(0.0))
  torch.testing.assert_close(ratchet[6], torch.tensor(0.0))
  torch.testing.assert_close(ratchet[8], torch.tensor(0.0))
  torch.testing.assert_close(ratchet[9], torch.tensor(1.0))
  assert term.ratchet_stride_mode[0].item() == 0
  assert term.ratchet_first_collision_stair_signal[0].item() is True
  assert term.ratchet_first_collision_seen[0].item() is True


def test_foot_event_memory_probe_reference_bounds_actual_stride_jump() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_probe_increment_m": 0.04,
        "ratchet_post_first_collision_probe_increment_m": 0.05,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_first_collision_seen[0] = True
  term.ratchet_collision_anchor_active[0] = True
  term.ratchet_collision_anchor_up_steps[0] = 1
  term.ratchet_probe_bootstrap_ready[0] = True
  term.ratchet_validated_stride_s[0] = 0.45
  term.ratchet_lower_s[0] = 0.25
  term.ratchet_probe_target_s[0] = 0.50
  term.ratchet_same_foot_stride_lower_s[0] = 0.25
  term.ratchet_last_forward_up_stride[0] = 0.45
  term.ratchet_confidence[0] = 0.6
  _seed_three_footprint_sequence(term, same_stride=0.62)

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([True]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[1], torch.tensor(0.30))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.55))
  torch.testing.assert_close(ratchet[5], torch.tensor(0.30))
  torch.testing.assert_close(term.ratchet_validated_stride_s[0], torch.tensor(0.62))
  assert term.ratchet_first_collision_seen[0].item() is True
  assert term.ratchet_stride_mode[0].item() == 0
  assert term.ratchet_post_first_collision_probe_latch[0].item() is True
  assert term.ratchet_post_first_collision_probe_growth_ok[0].item() is True
  assert term.ratchet_probe_target_overshot[0].item() is True
  torch.testing.assert_close(
    env.extras[STAIR_STRIDE_PHASE_EVENT_TARGET_KEY][0], torch.tensor(0.50)
  )
  torch.testing.assert_close(
    env.extras[STAIR_STRIDE_PHASE_EVENT_PREVIOUS_KEY][0], torch.tensor(0.45)
  )


def test_foot_event_memory_probe_cap_exposes_hold_without_confirming_interval() -> None:
  env = _make_env()
  term = _make_term(
    env,
    extra_params={
      "noise_enabled": False,
      "ratchet_probe_cap_m": 0.76,
    },
  )
  term.ratchet_active[0] = True
  term.ratchet_first_collision_seen[0] = True
  term.ratchet_probe_bootstrap_ready[0] = True
  term.ratchet_probe_target_s[0] = 0.76
  term.ratchet_last_forward_up_stride[0] = 0.76
  term.ratchet_confidence[0] = 0.8

  summary = term._compute_event_summary(update_ratchet=False, step_dt=0.02)[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[2], torch.tensor(0.76))
  torch.testing.assert_close(ratchet[6], torch.tensor(0.0))
  torch.testing.assert_close(ratchet[8], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[9], torch.tensor(1.0))


def test_foot_event_memory_probe_cap_keeps_farther_until_target_is_reached() -> None:
  env = _make_env()
  term = _make_term(
    env,
    extra_params={
      "noise_enabled": False,
      "ratchet_probe_cap_m": 0.76,
    },
  )
  term.ratchet_active[0] = True
  term.ratchet_first_collision_seen[0] = True
  term.ratchet_probe_bootstrap_ready[0] = True
  term.ratchet_probe_target_s[0] = 0.76
  term.ratchet_last_forward_up_stride[0] = 0.66
  term.ratchet_confidence[0] = 0.8

  summary = term._compute_event_summary(update_ratchet=False, step_dt=0.02)[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[2], torch.tensor(0.76))
  torch.testing.assert_close(ratchet[9], torch.tensor(0.0))


def test_foot_event_memory_keeps_collision_phase_latched_across_flat_pairs() -> None:
  env = _make_env()
  term = _make_term(env, extra_params={"ratchet_reset_flat_pairs": 4})
  term.ratchet_active[0] = True
  term.ratchet_first_collision_seen[0] = True
  term.ratchet_probe_target_s[0] = 0.60
  term.ratchet_flat_pair_steps[0] = 4

  term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([False]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )

  assert term.ratchet_active[0].item() is True
  assert term.ratchet_first_collision_seen[0].item() is True
  torch.testing.assert_close(term.ratchet_probe_target_s[0], torch.tensor(0.60))


def test_foot_event_memory_excludes_first_layer_and_bootstraps_next_foot() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_probe_increment_m": 0.05,
        "ratchet_post_first_collision_probe_increment_m": 0.05,
        "ratchet_probe_bootstrap_increment_m": 0.10,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_first_collision_seen[0] = True
  term.ratchet_collision_anchor_active[0] = True
  term.ratchet_collision_anchor_s[0] = 0.40
  term.ratchet_collision_anchor_z[0] = 0.0
  term.ratchet_collision_anchor_up_steps[0] = 0
  term.ratchet_lower_s[0] = 0.25
  term.ratchet_probe_target_s[0] = 0.45
  term.ratchet_same_foot_stride_lower_s[0] = 0.25
  term.ratchet_confidence[0] = 0.6

  term.footprint_valid[0, :4] = True
  first_layer, next_flat, same_flat, next_previous = term.footprints[0, :4]
  for footprint in (first_layer, next_flat, same_flat, next_previous):
    footprint[0] = 1.0
    footprint[3] = 1.0
    footprint[5] = 1.0
  first_layer[2] = 1.0
  first_layer[9] = 0.15
  first_layer[10] = 0.30
  next_flat[1] = 1.0
  next_flat[9] = 0.0
  next_flat[10] = 0.0
  same_flat[2] = 1.0
  same_flat[9] = 0.0
  same_flat[10] = -0.20
  next_previous[1] = 1.0
  next_previous[9] = 0.0
  next_previous[10] = -0.20

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([True]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[2], torch.tensor(0.50))
  torch.testing.assert_close(term.ratchet_validated_stride_s[0], torch.tensor(0.20))
  torch.testing.assert_close(term.ratchet_last_forward_up_stride[0], torch.tensor(0.20))
  assert term.ratchet_probe_bootstrap_ready[0].item() is True
  assert term.ratchet_probe_bootstrap_provisional[0].item() is True
  assert term.ratchet_probe_bootstrap_event[0].item() is True
  assert term.ratchet_collision_anchor_up_steps[0].item() == 1
  assert env.extras[STAIR_STRIDE_PHASE_EVENT_ID_KEY][0].item() == 0
  assert env.extras[STAIR_STRIDE_PHASE_EVENT_TYPE_KEY][0].item() == (
    STAIR_STRIDE_EVENT_NONE
  )

  _seed_three_footprint_sequence(term, same_stride=0.60)
  term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([True]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )

  torch.testing.assert_close(term.ratchet_probe_target_s[0], torch.tensor(0.70))
  torch.testing.assert_close(term.ratchet_validated_stride_s[0], torch.tensor(0.60))
  assert term.ratchet_probe_bootstrap_provisional[0].item() is False
  assert env.extras[STAIR_STRIDE_PHASE_EVENT_ID_KEY][0].item() == 0
  assert env.extras[STAIR_STRIDE_PHASE_EVENT_TYPE_KEY][0].item() == (
    STAIR_STRIDE_EVENT_NONE
  )


def test_foot_event_memory_ratchet_emits_probe_event_after_first_layer() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_probe_increment_m": 0.04,
        "ratchet_post_first_collision_probe_increment_m": 0.04,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_first_collision_seen[0] = True
  term.ratchet_collision_anchor_active[0] = True
  term.ratchet_collision_anchor_s[0] = 0.40
  term.ratchet_collision_anchor_z[0] = 0.0
  term.ratchet_collision_anchor_up_steps[0] = 1
  term.ratchet_probe_bootstrap_ready[0] = True
  term.ratchet_validated_stride_s[0] = 0.65
  term.ratchet_lower_s[0] = 0.58
  term.ratchet_probe_target_s[0] = 0.70
  term.ratchet_same_foot_stride_lower_s[0] = 0.58
  term.ratchet_last_forward_up_stride[0] = 0.65
  term.ratchet_confidence[0] = 0.6
  _seed_three_footprint_sequence(term, same_stride=0.70)

  term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([True]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )

  assert env.extras[STAIR_STRIDE_PHASE_EVENT_ID_KEY][0].item() == 1
  assert env.extras[STAIR_STRIDE_PHASE_EVENT_TYPE_KEY][0].item() == (
    STAIR_STRIDE_EVENT_PROBE_TOUCHDOWN
  )
  torch.testing.assert_close(
    env.extras[STAIR_STRIDE_PHASE_EVENT_ACTUAL_KEY][0], torch.tensor(0.70)
  )
  torch.testing.assert_close(
    env.extras[STAIR_STRIDE_PHASE_EVENT_PREVIOUS_KEY][0], torch.tensor(0.65)
  )
  torch.testing.assert_close(
    env.extras[STAIR_STRIDE_PHASE_EVENT_TARGET_KEY][0], torch.tensor(0.70)
  )
  assert env.extras[STAIR_STRIDE_PHASE_EVENT_COMPLETED_KEY][0].item() is True
  torch.testing.assert_close(term.ratchet_probe_target_s[0], torch.tensor(0.74))


def test_foot_event_memory_ratchet_backs_off_from_two_collision_geometry() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_single_collision_confirms_interval": False,
        "ratchet_two_collision_interval_margin_m": 0.025,
        "ratchet_two_collision_stride_layers": 2.0,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_first_collision_seen[0] = True
  term.ratchet_lower_s[0] = 0.25
  term.ratchet_probe_target_s[0] = 0.70
  term.ratchet_same_foot_stride_lower_s[0] = 0.25
  term.ratchet_last_forward_up_stride[0] = 0.54
  term.ratchet_confidence[0] = 0.4

  term.toe_valid[0, :2] = True
  latest_toe = term.toe_marks[0, 0]
  older_toe = term.toe_marks[0, 1]
  latest_toe[0] = 1.0
  latest_toe[1] = 1.0
  latest_toe[4] = 0.7
  latest_toe[5] = 0.05
  latest_toe[8] = 0.15
  latest_toe[9] = 0.70
  older_toe[0] = 1.0
  older_toe[2] = 1.0
  older_toe[4] = 0.7
  older_toe[5] = 0.60
  older_toe[8] = 0.0
  older_toe[9] = 0.40

  term.footprint_valid[0, :2] = True
  newer = term.footprints[0, 0]
  older = term.footprints[0, 1]
  newer[0] = 1.0
  newer[1] = 1.0
  newer[3] = 1.0
  newer[5] = 1.0
  newer[6] = 0.20
  newer[9] = 0.10
  newer[10] = 0.0
  older[0] = 1.0
  older[2] = 1.0
  older[3] = 1.0
  older[5] = 1.0
  older[6] = 0.40
  older[9] = 0.0
  older[10] = -0.10

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([False]),
    new_toe_mark_any=torch.tensor([True]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[0], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[1], torch.tensor(0.555))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.560))
  torch.testing.assert_close(ratchet[3], torch.tensor(0.595))
  torch.testing.assert_close(ratchet[5], torch.tensor(0.555))
  torch.testing.assert_close(ratchet[6], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[7], torch.tensor(0.595))
  torch.testing.assert_close(ratchet[9], torch.tensor(0.5))
  assert term.ratchet_stride_mode[0].item() == 1
  assert term.ratchet_valid_second_hit[0].item() is True
  assert term.ratchet_recovery_pending[0].item() is True
  torch.testing.assert_close(term.ratchet_recovery_target_s[0], torch.tensor(0.560))
  torch.testing.assert_close(term.ratchet_lock_target_s[0], torch.tensor(0.585))
  assert term.ratchet_lock_target_s[0] <= term.ratchet_collision_upper_s[0] - 0.01
  assert term.ratchet_stride_backoff_after_collision[0].item() is True
  assert term.ratchet_lock_entered[0].item() is False
  assert term.ratchet_two_collision_estimate_valid[0].item() is True
  torch.testing.assert_close(
    term.ratchet_two_collision_tread_depth_s[0],
    torch.tensor(0.30),
  )


def test_foot_event_memory_ratchet_requires_first_collision_context() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_first_collision_enters_stair_mode": True,
        "ratchet_single_collision_confirms_interval": False,
      }
    ),
    env,
  )

  term.toe_valid[0, :2] = True
  latest_toe = term.toe_marks[0, 0]
  older_toe = term.toe_marks[0, 1]
  latest_toe[0] = 1.0
  latest_toe[1] = 1.0
  latest_toe[4] = 0.7
  latest_toe[5] = 0.05
  latest_toe[8] = 0.15
  latest_toe[9] = 0.70
  older_toe[0] = 1.0
  older_toe[2] = 1.0
  older_toe[4] = 0.7
  older_toe[5] = 0.60
  older_toe[8] = 0.0
  older_toe[9] = 0.40

  term.footprint_valid[0, :2] = True
  newer = term.footprints[0, 0]
  older = term.footprints[0, 1]
  newer[0] = 1.0
  newer[1] = 1.0
  newer[3] = 1.0
  newer[5] = 1.0
  newer[6] = 0.20
  newer[9] = 0.10
  newer[10] = 0.0
  older[0] = 1.0
  older[2] = 1.0
  older[3] = 1.0
  older[5] = 1.0
  older[6] = 0.40
  older[9] = 0.0
  older[10] = -0.10

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([False]),
    new_toe_mark_any=torch.tensor([True]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  assert term.ratchet_first_collision_stair_signal[0].item() is True
  assert term.ratchet_first_collision_seen[0].item() is True
  assert term.ratchet_valid_second_hit[0].item() is False
  assert term.ratchet_recovery_pending[0].item() is False
  assert term.ratchet_interval_confirmed[0].item() is False
  assert term.ratchet_stride_mode[0].item() == 0
  torch.testing.assert_close(ratchet[6], torch.tensor(0.0))


def test_foot_event_memory_ratchet_cross_checks_two_collision_with_lower() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_single_collision_confirms_interval": False,
        "ratchet_two_collision_interval_margin_m": 0.025,
        "ratchet_two_collision_stride_layers": 2.0,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_first_collision_seen[0] = True
  term.ratchet_lower_s[0] = 0.58
  term.ratchet_probe_target_s[0] = 0.64
  term.ratchet_same_foot_stride_lower_s[0] = 0.58
  term.ratchet_confidence[0] = 0.4

  term.toe_valid[0, :2] = True
  latest_toe = term.toe_marks[0, 0]
  older_toe = term.toe_marks[0, 1]
  latest_toe[0] = 1.0
  latest_toe[1] = 1.0
  latest_toe[4] = 0.7
  latest_toe[5] = 0.05
  latest_toe[8] = 0.15
  latest_toe[9] = 0.70
  older_toe[0] = 1.0
  older_toe[2] = 1.0
  older_toe[4] = 0.7
  older_toe[5] = 0.60
  older_toe[8] = 0.0
  older_toe[9] = 0.40

  term.footprint_valid[0, :2] = True
  newer = term.footprints[0, 0]
  older = term.footprints[0, 1]
  newer[0] = 1.0
  newer[1] = 1.0
  newer[3] = 1.0
  newer[5] = 1.0
  newer[6] = 0.20
  newer[9] = 0.10
  newer[10] = 0.0
  older[0] = 1.0
  older[2] = 1.0
  older[3] = 1.0
  older[5] = 1.0
  older[6] = 0.40
  older[9] = 0.0
  older[10] = -0.10

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([False]),
    new_toe_mark_any=torch.tensor([True]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[1], torch.tensor(0.555))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.560))
  torch.testing.assert_close(ratchet[3], torch.tensor(0.595))
  torch.testing.assert_close(ratchet[5], torch.tensor(0.555))
  torch.testing.assert_close(ratchet[6], torch.tensor(1.0))
  assert term.ratchet_stride_mode[0].item() == 1
  assert term.ratchet_two_collision_estimate_valid[0].item() is True


def test_foot_event_memory_ratchet_uses_persistent_collision_anchor() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_single_collision_confirms_interval": False,
        "ratchet_two_collision_interval_margin_m": 0.025,
        "ratchet_two_collision_stride_layers": 2.0,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_first_collision_seen[0] = True
  term.ratchet_collision_anchor_active[0] = True
  term.ratchet_collision_anchor_s[0] = 0.40
  term.ratchet_collision_anchor_z[0] = 0.0
  term.ratchet_collision_anchor_up_steps[0] = 1
  term.ratchet_lower_s[0] = 0.25
  term.ratchet_probe_target_s[0] = 0.70
  term.ratchet_same_foot_stride_lower_s[0] = 0.25
  term.ratchet_confidence[0] = 0.4

  term.toe_valid[0, 0] = True
  latest_toe = term.toe_marks[0, 0]
  latest_toe[0] = 1.0
  latest_toe[1] = 1.0
  latest_toe[4] = 0.7
  latest_toe[5] = 0.05
  latest_toe[8] = 0.15
  latest_toe[9] = 0.70

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([False]),
    new_toe_mark_any=torch.tensor([True]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[0], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[1], torch.tensor(0.575))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.58))
  torch.testing.assert_close(ratchet[3], torch.tensor(0.625))
  torch.testing.assert_close(ratchet[5], torch.tensor(0.575))
  torch.testing.assert_close(ratchet[6], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[7], torch.tensor(0.625))
  assert term.ratchet_stride_mode[0].item() == 1
  assert term.ratchet_anchor_collision_estimate_valid[0].item() is True
  assert term.ratchet_two_collision_estimate_valid[0].item() is False
  assert term.ratchet_collision_anchor_active[0].item() is False
  assert env.extras[STAIR_STRIDE_PHASE_EVENT_ID_KEY][0].item() == 1
  assert env.extras[STAIR_STRIDE_PHASE_EVENT_TYPE_KEY][0].item() == (
    STAIR_STRIDE_EVENT_VALID_SECOND_HIT
  )
  torch.testing.assert_close(
    env.extras[STAIR_STRIDE_PHASE_EVENT_TARGET_KEY][0], torch.tensor(0.58)
  )


def test_foot_event_memory_ratchet_confirms_anchor_despite_probe_lower() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_single_collision_confirms_interval": False,
        "ratchet_two_collision_interval_margin_m": 0.025,
        "ratchet_two_collision_stride_layers": 2.0,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_first_collision_seen[0] = True
  term.ratchet_collision_anchor_active[0] = True
  term.ratchet_collision_anchor_s[0] = 0.40
  term.ratchet_collision_anchor_z[0] = 0.0
  term.ratchet_collision_anchor_up_steps[0] = 1
  term.ratchet_lower_s[0] = 0.70
  term.ratchet_probe_target_s[0] = 0.80
  term.ratchet_same_foot_stride_lower_s[0] = 0.70
  term.ratchet_confidence[0] = 0.4

  term.toe_valid[0, 0] = True
  latest_toe = term.toe_marks[0, 0]
  latest_toe[0] = 1.0
  latest_toe[1] = 1.0
  latest_toe[4] = 0.7
  latest_toe[5] = 0.05
  latest_toe[8] = 0.15
  latest_toe[9] = 0.70

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([False]),
    new_toe_mark_any=torch.tensor([True]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[1], torch.tensor(0.585))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.59))
  torch.testing.assert_close(ratchet[3], torch.tensor(0.625))
  torch.testing.assert_close(ratchet[6], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[9], torch.tensor(0.5))
  assert term.ratchet_anchor_collision_estimate_valid[0].item() is True
  assert term.ratchet_stride_backoff_after_collision[0].item() is True


def test_foot_event_memory_higher_riser_uses_bounded_anchor_fallback() -> None:
  env = _make_env()
  term = _make_term(
    env,
    extra_params={
      "memory_len": 6,
      "noise_enabled": False,
      "ratchet_single_collision_confirms_interval": False,
      "ratchet_two_collision_tread_max_m": 0.37,
    },
  )
  term.ratchet_active[0] = True
  term.ratchet_first_collision_seen[0] = True
  term.ratchet_collision_anchor_active[0] = True
  term.ratchet_collision_anchor_s[0] = 0.40
  term.ratchet_collision_anchor_z[0] = 0.0
  term.ratchet_collision_anchor_up_steps[0] = 1
  term.ratchet_lower_s[0] = 0.60
  term.ratchet_probe_target_s[0] = 0.75
  term.ratchet_same_foot_stride_lower_s[0] = 0.60

  term.toe_valid[0, 0] = True
  toe = term.toe_marks[0, 0]
  toe[0] = 1.0
  toe[1] = 1.0
  toe[4] = 0.7
  toe[5] = 0.05
  toe[8] = 0.15
  toe[9] = 0.85

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([False]),
    new_toe_mark_any=torch.tensor([True]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  assert term.ratchet_valid_second_hit[0].item() is True
  assert term.ratchet_interval_confirmed[0].item() is True
  assert term.ratchet_recovery_pending[0].item() is True
  assert term.ratchet_stride_mode[0].item() == 1
  torch.testing.assert_close(
    term.ratchet_anchor_collision_tread_depth_s[0],
    torch.tensor(0.45),
  )
  torch.testing.assert_close(ratchet[2], torch.tensor(0.72))


def test_foot_event_memory_higher_riser_switches_phase_despite_lower_cross_check() -> (
  None
):
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_single_collision_confirms_interval": False,
        "ratchet_two_collision_interval_margin_m": 0.025,
        "ratchet_two_collision_stride_layers": 2.0,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_first_collision_seen[0] = True
  term.ratchet_collision_anchor_active[0] = True
  term.ratchet_collision_anchor_s[0] = 0.40
  term.ratchet_collision_anchor_z[0] = 0.0
  term.ratchet_collision_anchor_up_steps[0] = 1
  term.ratchet_probe_bootstrap_ready[0] = True
  term.ratchet_validated_stride_s[0] = 0.75
  term.ratchet_lower_s[0] = 0.75
  term.ratchet_probe_target_s[0] = 0.80
  term.ratchet_same_foot_stride_lower_s[0] = 0.75
  term.ratchet_confidence[0] = 0.4

  term.toe_valid[0, 0] = True
  latest_toe = term.toe_marks[0, 0]
  latest_toe[0] = 1.0
  latest_toe[1] = 1.0
  latest_toe[4] = 0.7
  latest_toe[5] = 0.05
  latest_toe[8] = 0.15
  latest_toe[9] = 0.70

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([False]),
    new_toe_mark_any=torch.tensor([True]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[1], torch.tensor(0.585))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.59))
  torch.testing.assert_close(ratchet[3], torch.tensor(0.625))
  torch.testing.assert_close(ratchet[6], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[9], torch.tensor(0.5))
  assert term.ratchet_anchor_collision_cross_check_rejected[0].item() is True
  assert term.ratchet_anchor_collision_estimate_valid[0].item() is True
  assert term.ratchet_valid_second_hit[0].item() is True
  assert term.ratchet_interval_confirmed[0].item() is True
  assert term.ratchet_recovery_pending[0].item() is True
  assert term.ratchet_rejected_second_hit_count[0].item() == 0
  assert env.extras[STAIR_STRIDE_PHASE_EVENT_ID_KEY][0].item() == 1
  assert term.ratchet_stride_mode[0].item() == 1


def test_foot_event_memory_ratchet_rejects_duplicate_first_riser_collision() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_single_collision_confirms_interval": False,
        "ratchet_two_collision_interval_margin_m": 0.025,
        "ratchet_two_collision_stride_layers": 2.0,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_first_collision_seen[0] = True
  term.ratchet_collision_anchor_active[0] = True
  term.ratchet_collision_anchor_s[0] = 0.40
  term.ratchet_collision_anchor_z[0] = 0.02
  term.ratchet_collision_anchor_up_steps[0] = 1
  term.ratchet_lower_s[0] = 0.25
  term.ratchet_probe_target_s[0] = 0.70
  term.ratchet_same_foot_stride_lower_s[0] = 0.25
  term.ratchet_confidence[0] = 0.4
  term.ratchet_age_s[0] = 0.99

  term.toe_valid[0, 0] = True
  latest_toe = term.toe_marks[0, 0]
  latest_toe[0] = 1.0
  latest_toe[1] = 1.0
  latest_toe[4] = 0.7
  latest_toe[5] = 0.05
  latest_toe[8] = 0.02
  latest_toe[9] = 0.70

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([False]),
    new_toe_mark_any=torch.tensor([True]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[0], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.70))
  torch.testing.assert_close(ratchet[3], torch.tensor(0.0))
  torch.testing.assert_close(ratchet[6], torch.tensor(0.0))
  assert term.ratchet_duplicate_first_riser_collision_rejected[0].item() is True
  assert term.ratchet_active[0].item() is True
  torch.testing.assert_close(term.ratchet_age_s[0], torch.tensor(0.0))
  assert term.ratchet_anchor_collision_estimate_valid[0].item() is False
  assert term.ratchet_collision_anchor_active[0].item() is True
  torch.testing.assert_close(term.ratchet_collision_anchor_s[0], torch.tensor(0.40))
  torch.testing.assert_close(term.ratchet_collision_anchor_z[0], torch.tensor(0.02))


def test_foot_event_memory_ratchet_keeps_probe_after_invalid_second_collision() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_single_collision_confirms_interval": False,
        "ratchet_two_collision_interval_margin_m": 0.025,
        "ratchet_two_collision_stride_layers": 2.0,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_first_collision_seen[0] = True
  term.ratchet_collision_anchor_active[0] = True
  term.ratchet_collision_anchor_s[0] = 0.40
  term.ratchet_collision_anchor_z[0] = 0.0
  term.ratchet_collision_anchor_up_steps[0] = 1
  term.ratchet_lower_s[0] = 0.50
  term.ratchet_probe_target_s[0] = 0.70
  term.ratchet_same_foot_stride_lower_s[0] = 0.50
  term.ratchet_confidence[0] = 0.4

  term.toe_valid[0, 0] = True
  latest_toe = term.toe_marks[0, 0]
  latest_toe[0] = 1.0
  latest_toe[1] = 1.0
  latest_toe[4] = 0.50
  latest_toe[5] = 0.05
  latest_toe[8] = 0.15
  latest_toe[9] = 0.50

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([False]),
    new_toe_mark_any=torch.tensor([True]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[2], torch.tensor(0.70))
  torch.testing.assert_close(ratchet[3], torch.tensor(0.0))
  torch.testing.assert_close(ratchet[6], torch.tensor(0.0))
  assert term.ratchet_collision_candidate[0].item() is True
  assert term.ratchet_collision_accepted[0].item() is False
  assert term.ratchet_collision_soft_upper[0].item() is False
  assert term.ratchet_stride_mode[0].item() == 0


def test_foot_event_memory_backoff_without_recovery_does_not_fake_hold() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_stride_mode[0] = 1
  term.ratchet_upper_seen[0] = True
  term.ratchet_interval_confirmed[0] = True
  term.ratchet_lower_s[0] = 0.575
  term.ratchet_probe_target_s[0] = 0.595
  term.ratchet_upper_s[0] = 0.625
  term.ratchet_same_foot_stride_lower_s[0] = 0.575
  term.ratchet_same_foot_stride_upper_s[0] = 0.625
  term.ratchet_safe_no_hit_steps[0] = 1
  term.ratchet_confidence[0] = 0.75

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([False]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[0], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.60))
  torch.testing.assert_close(ratchet[6], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[9], torch.tensor(0.5))
  assert term.ratchet_stride_mode[0].item() == 1
  assert term.ratchet_hold_center_active[0].item() is False
  torch.testing.assert_close(
    term.ratchet_hold_center_target_s[0],
    torch.tensor(0.0),
  )


def test_foot_event_memory_ratchet_keeps_soft_upper_below_safe_lower() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_min_interval_width_m": 0.04,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_lower_s[0] = 0.30
  term.ratchet_probe_target_s[0] = 0.45
  term.ratchet_same_foot_stride_lower_s[0] = 0.30
  term.ratchet_last_forward_up_stride[0] = 0.30
  term.ratchet_confidence[0] = 0.4

  term.toe_valid[0, 0] = True
  toe = term.toe_marks[0, 0]
  toe[0] = 1.0
  toe[1] = 1.0
  toe[4] = 0.6
  toe[5] = 0.05
  toe[9] = 0.415

  term.footprint_valid[0, 0] = True
  footprint = term.footprints[0, 0]
  footprint[0] = 1.0
  footprint[1] = 1.0
  footprint[3] = 1.0
  footprint[5] = 1.0
  footprint[6] = 0.20
  footprint[10] = 0.05

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([False]),
    new_toe_mark_any=torch.tensor([True]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[0], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[1], torch.tensor(0.30))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.45))
  torch.testing.assert_close(ratchet[3], torch.tensor(0.26))
  torch.testing.assert_close(ratchet[5], torch.tensor(0.30))
  torch.testing.assert_close(ratchet[6], torch.tensor(0.0))
  torch.testing.assert_close(ratchet[7], torch.tensor(0.26))
  assert term.ratchet_stride_mode[0].item() == 0
  assert term.ratchet_collision_candidate[0].item() is True
  assert term.ratchet_collision_accepted[0].item() is False
  assert term.ratchet_collision_soft_upper[0].item() is True
  assert term.ratchet_collision_rejected[0].item() is False
  assert term.ratchet_collision_soft_below_lower[0].item() is True
  assert term.ratchet_soft_upper_active[0].item() is True


def test_foot_event_memory_soft_upper_does_not_stop_valid_probe_sequence() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_probe_increment_m": 0.025,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_first_collision_seen[0] = True
  term.ratchet_collision_anchor_active[0] = True
  term.ratchet_collision_anchor_up_steps[0] = 1
  term.ratchet_probe_bootstrap_ready[0] = True
  term.ratchet_validated_stride_s[0] = 0.45
  term.ratchet_soft_upper_active[0] = True
  term.ratchet_soft_upper_ttl_remaining[0] = 2
  term.ratchet_stride_mode[0] = 0
  term.ratchet_upper_seen[0] = True
  term.ratchet_lower_s[0] = 0.30
  term.ratchet_probe_target_s[0] = 0.50
  term.ratchet_upper_s[0] = 0.26
  term.ratchet_same_foot_stride_lower_s[0] = 0.30
  term.ratchet_same_foot_stride_upper_s[0] = 0.26
  term.ratchet_confidence[0] = 0.75

  _seed_three_footprint_sequence(term, same_stride=0.50)

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([True]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[0], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[1], torch.tensor(0.325))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.525))
  torch.testing.assert_close(ratchet[3], torch.tensor(0.26))
  torch.testing.assert_close(ratchet[5], torch.tensor(0.325))
  torch.testing.assert_close(ratchet[6], torch.tensor(0.0))
  torch.testing.assert_close(ratchet[7], torch.tensor(0.26))
  assert term.ratchet_soft_upper_active[0].item() is True
  assert term.ratchet_soft_upper_ttl_remaining[0].item() == 1
  assert term.ratchet_stride_mode[0].item() == 0
  assert term.ratchet_probe_target_advanced[0].item() is True
  assert term.ratchet_backoff_monotonic[0].item() is False

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([True]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[0], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.525))
  torch.testing.assert_close(ratchet[3], torch.tensor(0.0))
  torch.testing.assert_close(ratchet[7], torch.tensor(0.0))
  assert term.ratchet_soft_upper_active[0].item() is False
  assert term.ratchet_soft_upper_released[0].item() is True
  assert term.ratchet_soft_upper_ttl_remaining[0].item() == 0
  assert term.ratchet_stride_mode[0].item() == 0


def test_foot_event_memory_ratchet_locks_after_stable_actual_stride() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_lock_stable_steps": 2,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_stride_mode[0] = 1
  term.ratchet_upper_seen[0] = True
  term.ratchet_interval_confirmed[0] = True
  term.ratchet_lock_stable_count[0] = 1
  term.ratchet_lower_s[0] = 0.25
  term.ratchet_probe_target_s[0] = 0.32
  term.ratchet_upper_s[0] = 0.34
  term.ratchet_same_foot_stride_lower_s[0] = 0.25
  term.ratchet_same_foot_stride_upper_s[0] = 0.34
  term.ratchet_safe_no_hit_steps[0] = 1
  term.ratchet_confidence[0] = 0.75

  term.footprint_valid[0, :3] = True
  newest_right = term.footprints[0, 0]
  previous_left = term.footprints[0, 1]
  older_right = term.footprints[0, 2]
  newest_right[0] = 1.0
  newest_right[2] = 1.0
  newest_right[3] = 1.0
  newest_right[5] = 1.0
  newest_right[9] = 0.20
  newest_right[10] = 0.32
  previous_left[0] = 1.0
  previous_left[1] = 1.0
  previous_left[3] = 1.0
  previous_left[5] = 1.0
  previous_left[9] = 0.10
  previous_left[10] = 0.17
  older_right[0] = 1.0
  older_right[2] = 1.0
  older_right[3] = 1.0
  older_right[5] = 1.0
  older_right[9] = 0.0
  older_right[10] = 0.02

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([True]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[0], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.295))
  torch.testing.assert_close(ratchet[6], torch.tensor(1.0))
  assert term.ratchet_stride_mode[0].item() == 2
  assert term.ratchet_lock_entered[0].item() is True
  assert term.ratchet_lock_stable_count[0].item() == 2
  torch.testing.assert_close(term.ratchet_lock_lower_s[0], torch.tensor(0.255))
  torch.testing.assert_close(term.ratchet_lock_upper_s[0], torch.tensor(0.335))


@pytest.mark.parametrize(
  (
    "actual_stride",
    "previous_stride",
    "recovery_completed",
    "expected_mode",
    "expected_target",
    "expected_intent",
  ),
  [
    (0.59, 0.65, True, 2, 0.60, 1.0),
    (0.60, 0.60, True, 2, 0.60, 1.0),
    (0.68, 0.65, False, 1, 0.60, 0.5),
    (0.50, 0.65, False, 1, 0.60, 0.0),
  ],
)
def test_foot_event_memory_ratchet_emits_backoff_event(
  actual_stride: float,
  previous_stride: float,
  recovery_completed: bool,
  expected_mode: int,
  expected_target: float,
  expected_intent: float,
) -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_stride_mode[0] = 1
  term.ratchet_upper_seen[0] = True
  term.ratchet_interval_confirmed[0] = True
  term.ratchet_lower_s[0] = 0.575
  term.ratchet_probe_target_s[0] = 0.60
  term.ratchet_upper_s[0] = 0.625
  term.ratchet_same_foot_stride_lower_s[0] = 0.575
  term.ratchet_same_foot_stride_upper_s[0] = 0.625
  term.ratchet_recovery_pending[0] = True
  term.ratchet_recovery_target_s[0] = 0.60
  term.ratchet_lock_target_s[0] = 0.60
  term.ratchet_last_forward_up_stride[0] = previous_stride
  term.ratchet_confidence[0] = 0.75

  term.footprint_valid[0, :3] = True
  newest_right = term.footprints[0, 0]
  previous_left = term.footprints[0, 1]
  older_right = term.footprints[0, 2]
  newest_right[0] = 1.0
  newest_right[2] = 1.0
  newest_right[3] = 1.0
  newest_right[5] = 1.0
  newest_right[9] = 0.20
  newest_right[10] = actual_stride + 0.02
  previous_left[0] = 1.0
  previous_left[1] = 1.0
  previous_left[3] = 1.0
  previous_left[5] = 1.0
  previous_left[9] = 0.10
  previous_left[10] = 0.27
  older_right[0] = 1.0
  older_right[2] = 1.0
  older_right[3] = 1.0
  older_right[5] = 1.0
  older_right[9] = 0.0
  older_right[10] = 0.02

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([True]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  assert env.extras[STAIR_STRIDE_PHASE_EVENT_ID_KEY][0].item() == 1
  assert env.extras[STAIR_STRIDE_PHASE_EVENT_TYPE_KEY][0].item() == (
    STAIR_STRIDE_EVENT_BACKOFF_TOUCHDOWN
  )
  torch.testing.assert_close(
    env.extras[STAIR_STRIDE_PHASE_EVENT_ACTUAL_KEY][0],
    torch.tensor(actual_stride),
  )
  torch.testing.assert_close(
    env.extras[STAIR_STRIDE_PHASE_EVENT_TARGET_KEY][0],
    torch.tensor(0.60),
  )
  assert term.ratchet_recovery_touchdown[0].item() is True
  assert term.ratchet_recovery_completed[0].item() is recovery_completed
  assert (
    env.extras[STAIR_STRIDE_PHASE_EVENT_COMPLETED_KEY][0].item() is recovery_completed
  )
  assert term.ratchet_recovery_pending[0].item() is (not recovery_completed)
  assert term.ratchet_stride_mode[0].item() == expected_mode
  torch.testing.assert_close(
    term.ratchet_probe_target_s[0],
    torch.tensor(expected_target),
  )
  torch.testing.assert_close(ratchet[9], torch.tensor(expected_intent))


def test_foot_event_memory_ratchet_emits_lock_event() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_lock_phase_correction_enabled": False,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_stride_mode[0] = 2
  term.ratchet_interval_confirmed[0] = True
  term.ratchet_lower_s[0] = 0.55
  term.ratchet_probe_target_s[0] = 0.60
  term.ratchet_upper_s[0] = 0.65
  term.ratchet_same_foot_stride_lower_s[0] = 0.55
  term.ratchet_same_foot_stride_upper_s[0] = 0.65
  term.ratchet_lock_lower_s[0] = 0.57
  term.ratchet_lock_upper_s[0] = 0.63
  term.ratchet_lock_target_s[0] = 0.60
  term.ratchet_confidence[0] = 0.75

  term.footprint_valid[0, :3] = True
  newest_right = term.footprints[0, 0]
  previous_left = term.footprints[0, 1]
  older_right = term.footprints[0, 2]
  newest_right[0] = 1.0
  newest_right[2] = 1.0
  newest_right[3] = 1.0
  newest_right[5] = 1.0
  newest_right[9] = 0.20
  newest_right[10] = 0.64
  previous_left[0] = 1.0
  previous_left[1] = 1.0
  previous_left[3] = 1.0
  previous_left[5] = 1.0
  previous_left[9] = 0.10
  previous_left[10] = 0.30
  older_right[0] = 1.0
  older_right[2] = 1.0
  older_right[3] = 1.0
  older_right[5] = 1.0
  older_right[9] = 0.0
  older_right[10] = 0.02

  term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([True]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )

  assert env.extras[STAIR_STRIDE_PHASE_EVENT_ID_KEY][0].item() == 1
  assert env.extras[STAIR_STRIDE_PHASE_EVENT_TYPE_KEY][0].item() == (
    STAIR_STRIDE_EVENT_LOCK_TOUCHDOWN
  )
  torch.testing.assert_close(
    env.extras[STAIR_STRIDE_PHASE_EVENT_ACTUAL_KEY][0],
    torch.tensor(0.62),
  )
  torch.testing.assert_close(
    env.extras[STAIR_STRIDE_PHASE_EVENT_TARGET_KEY][0],
    torch.tensor(0.60),
  )


def test_foot_event_memory_ratchet_does_not_lock_from_target_only() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_lock_stable_steps": 1,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_stride_mode[0] = 1
  term.ratchet_upper_seen[0] = True
  term.ratchet_interval_confirmed[0] = True
  term.ratchet_lower_s[0] = 0.25
  term.ratchet_probe_target_s[0] = 0.30
  term.ratchet_upper_s[0] = 0.34
  term.ratchet_same_foot_stride_lower_s[0] = 0.25
  term.ratchet_same_foot_stride_upper_s[0] = 0.34
  term.ratchet_safe_no_hit_steps[0] = 1
  term.ratchet_confidence[0] = 0.75

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([False]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[0], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.295))
  torch.testing.assert_close(ratchet[6], torch.tensor(1.0))
  assert term.ratchet_stride_mode[0].item() == 1
  assert term.ratchet_lock_entered[0].item() is False
  assert term.ratchet_hold_center_active[0].item() is False
  assert term.ratchet_target_inside_lock[0].item() is False
  assert term.ratchet_actual_stride_inside_lock[0].item() is False


def test_foot_event_memory_lock_target_ignores_phase_correction() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_lock_phase_correction_gain": 1.0,
        "ratchet_lock_phase_deadband_m": 0.0,
        "ratchet_lock_phase_max_backoff_m": 0.03,
        "ratchet_lock_phase_max_forward_m": 0.02,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_stride_mode[0] = 2
  term.ratchet_interval_confirmed[0] = True
  term.ratchet_lower_s[0] = 0.55
  term.ratchet_upper_s[0] = 0.65
  term.ratchet_same_foot_stride_lower_s[0] = 0.55
  term.ratchet_same_foot_stride_upper_s[0] = 0.65
  term.ratchet_lock_lower_s[0] = 0.57
  term.ratchet_lock_upper_s[0] = 0.63
  term.ratchet_lock_target_s[0] = 0.60
  term.ratchet_lock_phase_error_s[0] = 0.02
  term.ratchet_confidence[0] = 0.75

  for _ in range(2):
    summary = term._compute_event_summary(
      update_ratchet=True,
      new_footprint_any=torch.tensor([False]),
      new_toe_mark_any=torch.tensor([False]),
      step_dt=0.02,
    )[0]
    ratchet = summary[FOOT_EVENT_RATCHET_START:]
    torch.testing.assert_close(ratchet[2], torch.tensor(0.60))
    torch.testing.assert_close(term.ratchet_lock_phase_error_s[0], torch.tensor(0.02))
    torch.testing.assert_close(
      term.ratchet_lock_phase_correction_s[0],
      torch.tensor(0.02),
    )


def test_foot_event_memory_ratchet_lock_phase_correction_decays_after_step() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_lock_phase_correction_gain": 1.0,
        "ratchet_lock_phase_deadband_m": 0.0,
        "ratchet_lock_phase_max_backoff_m": 0.03,
        "ratchet_lock_phase_max_forward_m": 0.02,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_stride_mode[0] = 2
  term.ratchet_interval_confirmed[0] = True
  term.ratchet_lower_s[0] = 0.55
  term.ratchet_upper_s[0] = 0.65
  term.ratchet_same_foot_stride_lower_s[0] = 0.55
  term.ratchet_same_foot_stride_upper_s[0] = 0.65
  term.ratchet_lock_lower_s[0] = 0.57
  term.ratchet_lock_upper_s[0] = 0.63
  term.ratchet_lock_phase_error_s[0] = 0.02
  term.ratchet_confidence[0] = 0.75

  term.footprint_valid[0, :3] = True
  newest_right = term.footprints[0, 0]
  previous_left = term.footprints[0, 1]
  older_right = term.footprints[0, 2]
  newest_right[0] = 1.0
  newest_right[2] = 1.0
  newest_right[3] = 1.0
  newest_right[5] = 1.0
  newest_right[6] = 0.0
  newest_right[9] = 0.20
  newest_right[10] = 0.60
  previous_left[0] = 1.0
  previous_left[1] = 1.0
  previous_left[3] = 1.0
  previous_left[5] = 1.0
  previous_left[6] = 0.10
  previous_left[9] = 0.10
  previous_left[10] = 0.30
  older_right[0] = 1.0
  older_right[2] = 1.0
  older_right[3] = 1.0
  older_right[5] = 1.0
  older_right[6] = 0.20
  older_right[9] = 0.0
  older_right[10] = 0.02

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([True]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[2], torch.tensor(0.60))
  torch.testing.assert_close(term.ratchet_lock_phase_error_s[0], torch.tensor(0.0))
  torch.testing.assert_close(
    term.ratchet_lock_phase_correction_s[0],
    torch.tensor(0.0),
  )


def test_foot_event_memory_ratchet_reopens_lock_after_collision() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_stride_mode[0] = 2
  term.ratchet_upper_seen[0] = True
  term.ratchet_interval_confirmed[0] = True
  term.ratchet_lower_s[0] = 0.25
  term.ratchet_probe_target_s[0] = 0.325
  term.ratchet_upper_s[0] = 0.40
  term.ratchet_same_foot_stride_lower_s[0] = 0.25
  term.ratchet_same_foot_stride_upper_s[0] = 0.40
  term.ratchet_lock_lower_s[0] = 0.26
  term.ratchet_lock_upper_s[0] = 0.39
  term.ratchet_confidence[0] = 0.75

  term.toe_valid[0, 0] = True
  toe = term.toe_marks[0, 0]
  toe[0] = 1.0
  toe[1] = 1.0
  toe[4] = 0.6
  toe[5] = 0.05
  toe[9] = 0.495

  term.footprint_valid[0, 0] = True
  footprint = term.footprints[0, 0]
  footprint[0] = 1.0
  footprint[1] = 1.0
  footprint[3] = 1.0
  footprint[5] = 1.0
  footprint[6] = 0.20
  footprint[10] = 0.05

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([False]),
    new_toe_mark_any=torch.tensor([True]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[0], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.255))
  torch.testing.assert_close(ratchet[3], torch.tensor(0.34))
  torch.testing.assert_close(ratchet[7], torch.tensor(0.34))
  assert term.ratchet_stride_mode[0].item() == 1
  assert term.ratchet_lock_collision_reopen[0].item() is True
  torch.testing.assert_close(term.ratchet_lock_lower_s[0], torch.tensor(0.0))
  torch.testing.assert_close(term.ratchet_lock_upper_s[0], torch.tensor(0.0))


def test_foot_event_memory_ratchet_reopens_lock_from_soft_upper_collision() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_stride_mode[0] = 2
  term.ratchet_upper_seen[0] = True
  term.ratchet_interval_confirmed[0] = True
  term.ratchet_lower_s[0] = 0.55
  term.ratchet_probe_target_s[0] = 0.60
  term.ratchet_upper_s[0] = 0.65
  term.ratchet_same_foot_stride_lower_s[0] = 0.55
  term.ratchet_same_foot_stride_upper_s[0] = 0.65
  term.ratchet_lock_lower_s[0] = 0.57
  term.ratchet_lock_upper_s[0] = 0.63
  term.ratchet_lock_target_s[0] = 0.60
  term.ratchet_confidence[0] = 0.75

  term.toe_valid[0, 0] = True
  toe = term.toe_marks[0, 0]
  toe[0] = 1.0
  toe[1] = 1.0
  toe[4] = 0.6
  toe[5] = 0.05
  toe[9] = 0.495

  term.footprint_valid[0, 0] = True
  footprint = term.footprints[0, 0]
  footprint[0] = 1.0
  footprint[1] = 1.0
  footprint[3] = 1.0
  footprint[5] = 1.0
  footprint[6] = 0.20
  footprint[10] = 0.05

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([False]),
    new_toe_mark_any=torch.tensor([True]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  assert term.ratchet_collision_soft_upper[0].item() is True
  assert term.ratchet_lock_collision_reopen[0].item() is True
  assert term.ratchet_recovery_pending[0].item() is True
  assert term.ratchet_stride_mode[0].item() == 1
  torch.testing.assert_close(ratchet[3], torch.tensor(0.34))
  torch.testing.assert_close(term.ratchet_lock_target_s[0], torch.tensor(0.60))
  torch.testing.assert_close(term.ratchet_recovery_target_s[0], torch.tensor(0.56))


def test_foot_event_memory_repeated_backoff_collision_does_not_drift_targets() -> None:
  env = _make_env()
  term = _make_term(env, extra_params={"memory_len": 6, "noise_enabled": False})
  term.ratchet_active[0] = True
  term.ratchet_stride_mode[0] = 1
  term.ratchet_upper_seen[0] = True
  term.ratchet_interval_confirmed[0] = True
  term.ratchet_recovery_pending[0] = True
  term.ratchet_recovery_target_s[0] = 0.55
  term.ratchet_lock_target_s[0] = 0.60
  term.ratchet_lower_s[0] = 0.55
  term.ratchet_probe_target_s[0] = 0.55
  term.ratchet_upper_s[0] = 0.65
  term.ratchet_same_foot_stride_lower_s[0] = 0.55
  term.ratchet_same_foot_stride_upper_s[0] = 0.65
  term.ratchet_confidence[0] = 0.75

  term.toe_valid[0, 0] = True
  toe = term.toe_marks[0, 0]
  toe[0] = 1.0
  toe[1] = 1.0
  toe[4] = 0.6
  toe[5] = 0.05
  toe[9] = 0.495
  term.footprint_valid[0, 0] = True
  footprint = term.footprints[0, 0]
  footprint[0] = 1.0
  footprint[1] = 1.0
  footprint[3] = 1.0
  footprint[5] = 1.0
  footprint[6] = 0.20
  footprint[10] = 0.05

  term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([False]),
    new_toe_mark_any=torch.tensor([True]),
    step_dt=0.02,
  )

  assert term.ratchet_recovery_pending[0].item() is True
  assert term.ratchet_stride_mode[0].item() == 1
  assert term.ratchet_valid_second_hit[0].item() is False
  torch.testing.assert_close(term.ratchet_recovery_target_s[0], torch.tensor(0.55))
  torch.testing.assert_close(term.ratchet_lock_target_s[0], torch.tensor(0.60))


def test_foot_event_memory_ratchet_full_probe_recovery_lock_lifecycle() -> None:
  env = _make_env()
  term = _make_term(
    env,
    extra_params={
      "memory_len": 6,
      "noise_enabled": False,
      "ratchet_post_first_collision_probe_increment_m": 0.05,
      "ratchet_lock_phase_correction_enabled": False,
    },
  )
  term.ratchet_active[0] = True
  term.ratchet_first_collision_seen[0] = True
  term.ratchet_collision_anchor_active[0] = True
  term.ratchet_collision_anchor_up_steps[0] = 1
  term.ratchet_probe_bootstrap_ready[0] = True
  term.ratchet_validated_stride_s[0] = 0.55
  term.ratchet_probe_target_s[0] = 0.60
  term.ratchet_lower_s[0] = 0.55
  term.ratchet_same_foot_stride_lower_s[0] = 0.55
  term.ratchet_last_forward_up_stride[0] = 0.55
  term.ratchet_confidence[0] = 0.8

  _seed_three_footprint_sequence(term, same_stride=0.60)
  term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([True]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )
  assert env.extras[STAIR_STRIDE_PHASE_EVENT_TYPE_KEY][0].item() == (
    STAIR_STRIDE_EVENT_PROBE_TOUCHDOWN
  )
  torch.testing.assert_close(term.ratchet_probe_target_s[0], torch.tensor(0.65))

  term.ratchet_stride_mode[0] = 1
  term.ratchet_interval_confirmed[0] = True
  term.ratchet_upper_seen[0] = True
  term.ratchet_upper_s[0] = 0.625
  term.ratchet_same_foot_stride_upper_s[0] = 0.625
  term.ratchet_recovery_pending[0] = True
  term.ratchet_recovery_target_s[0] = 0.60
  term.ratchet_lock_target_s[0] = 0.60
  term.ratchet_last_forward_up_stride[0] = 0.65

  _seed_three_footprint_sequence(term, same_stride=0.52)
  term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([True]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )
  assert term.ratchet_recovery_completed[0].item() is False
  assert term.ratchet_recovery_pending[0].item() is True
  assert term.ratchet_recovery_reduction_seen[0].item() is True
  summary = term._compute_event_summary(update_ratchet=False, step_dt=0.02)[0]
  torch.testing.assert_close(summary[FOOT_EVENT_RATCHET_START + 9], torch.tensor(0.0))

  _seed_three_footprint_sequence(term, same_stride=0.60)
  term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([True]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )
  assert term.ratchet_recovery_completed[0].item() is True
  assert term.ratchet_recovery_pending[0].item() is False
  assert term.ratchet_stride_mode[0].item() == 2
  torch.testing.assert_close(term.ratchet_probe_target_s[0], torch.tensor(0.60))

  _seed_three_footprint_sequence(term, same_stride=0.60)
  term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([True]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )
  assert env.extras[STAIR_STRIDE_PHASE_EVENT_TYPE_KEY][0].item() == (
    STAIR_STRIDE_EVENT_LOCK_TOUCHDOWN
  )
  summary = term._compute_event_summary(
    update_ratchet=False,
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]
  torch.testing.assert_close(ratchet[2], torch.tensor(0.60))
  torch.testing.assert_close(ratchet[9], torch.tensor(1.0))


def test_foot_event_memory_ratchet_rejects_low_confidence_toe_hit() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_collision_min_confidence": 0.45,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_lower_s[0] = 0.25
  term.ratchet_probe_target_s[0] = 0.45
  term.ratchet_same_foot_stride_lower_s[0] = 0.25

  term.toe_valid[0, 0] = True
  toe = term.toe_marks[0, 0]
  toe[0] = 1.0
  toe[1] = 1.0
  toe[4] = 0.2
  toe[9] = 0.41

  term.footprint_valid[0, 0] = True
  footprint = term.footprints[0, 0]
  footprint[0] = 1.0
  footprint[1] = 1.0
  footprint[3] = 1.0
  footprint[5] = 1.0
  footprint[6] = 0.20
  footprint[10] = 0.05

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([False]),
    new_toe_mark_any=torch.tensor([True]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[0], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[6], torch.tensor(0.0))
  torch.testing.assert_close(ratchet[3], torch.tensor(0.0))
  assert term.ratchet_collision_candidate[0].item() is True
  assert term.ratchet_collision_accepted[0].item() is False
  assert term.ratchet_collision_soft_upper[0].item() is False
  assert term.ratchet_collision_rejected[0].item() is True
  assert term.ratchet_collision_rejected_low_confidence[0].item() is True


def test_foot_event_memory_ratchet_rejects_same_layer_collision_after_first_hit() -> (
  None
):
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
        "ratchet_collision_min_confidence": 0.45,
        "ratchet_post_first_collision_collision_min_confidence": 0.30,
      }
    ),
    env,
  )

  term.ratchet_active[0] = True
  term.ratchet_first_collision_seen[0] = True
  term.ratchet_lower_s[0] = 0.25
  term.ratchet_probe_target_s[0] = 0.45
  term.ratchet_same_foot_stride_lower_s[0] = 0.25

  term.toe_valid[0, 0] = True
  toe = term.toe_marks[0, 0]
  toe[0] = 1.0
  toe[1] = 1.0
  toe[4] = 0.35
  toe[9] = 0.41

  term.footprint_valid[0, 0] = True
  footprint = term.footprints[0, 0]
  footprint[0] = 1.0
  footprint[1] = 1.0
  footprint[3] = 1.0
  footprint[5] = 1.0
  footprint[6] = 0.20
  footprint[10] = 0.05

  term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([False]),
    new_toe_mark_any=torch.tensor([True]),
    step_dt=0.02,
  )

  assert term.ratchet_collision_candidate[0].item() is True
  assert term.ratchet_collision_rejected_low_confidence[0].item() is False
  assert term.ratchet_collision_soft_upper[0].item() is False
  assert term.ratchet_collision_rejected[0].item() is True
  assert term.ratchet_duplicate_first_riser_collision_rejected[0].item() is True
  assert term.ratchet_stride_mode[0].item() == 0


def test_foot_event_memory_ratchet_ignores_toe_hit_without_upstair_context() -> None:
  env = _make_env()
  term = FootEventMemoryObs(
    SimpleNamespace(
      params={
        "memory_len": 6,
        "noise_enabled": False,
        "age_norm_s": 1.0,
        "stance_age_norm_s": 1.0,
      }
    ),
    env,
  )

  term.toe_valid[0, 0] = True
  toe = term.toe_marks[0, 0]
  toe[0] = 1.0
  toe[1] = 1.0
  toe[4] = 0.6
  toe[9] = 0.41

  term.footprint_valid[0, 0] = True
  footprint = term.footprints[0, 0]
  footprint[0] = 1.0
  footprint[1] = 1.0
  footprint[3] = 1.0
  footprint[5] = 1.0
  footprint[10] = 0.05

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([False]),
    new_toe_mark_any=torch.tensor([True]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[0], torch.tensor(0.0))
  torch.testing.assert_close(ratchet[6], torch.tensor(0.0))
