"""Tests for deployable-shaped foot event memory observations."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

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


def test_foot_event_memory_ratchet_advances_after_forward_up_step() -> None:
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

  torch.testing.assert_close(ratchet[0], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[1], torch.tensor(0.25))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.275))
  torch.testing.assert_close(ratchet[3], torch.tensor(0.0))
  torch.testing.assert_close(ratchet[4], torch.tensor(0.12))
  torch.testing.assert_close(ratchet[5], torch.tensor(0.25))
  torch.testing.assert_close(ratchet[6], torch.tensor(0.0))
  torch.testing.assert_close(ratchet[8], torch.tensor(1.0))
  assert term.ratchet_safe_no_hit_steps[0].item() == 1


def test_foot_event_memory_ratchet_uses_same_foot_two_layer_stride() -> None:
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

  term.footprint_valid[0, :3] = True
  newest_right = term.footprints[0, 0]
  previous_left = term.footprints[0, 1]
  older_right = term.footprints[0, 2]
  newest_right[0] = 1.0
  newest_right[2] = 1.0
  newest_right[3] = 1.0
  newest_right[5] = 1.0
  newest_right[9] = 0.20
  newest_right[10] = 0.52
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

  summary = term._compute_event_summary(
    update_ratchet=True,
    new_footprint_any=torch.tensor([True]),
    new_toe_mark_any=torch.tensor([False]),
    step_dt=0.02,
  )[0]
  ratchet = summary[FOOT_EVENT_RATCHET_START:]

  torch.testing.assert_close(ratchet[0], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[1], torch.tensor(0.50))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.525))
  torch.testing.assert_close(ratchet[3], torch.tensor(0.0))
  torch.testing.assert_close(ratchet[4], torch.tensor(0.20))
  torch.testing.assert_close(ratchet[5], torch.tensor(0.50))


def test_foot_event_memory_ratchet_closes_interval_from_same_foot_toe_hit() -> None:
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
  term.ratchet_depth_lower_s[0] = 0.25
  term.ratchet_last_forward_up_stride[0] = 0.25
  term.ratchet_confidence[0] = 0.4

  term.toe_valid[0, 0] = True
  toe = term.toe_marks[0, 0]
  toe[0] = 1.0
  toe[1] = 1.0
  toe[4] = 0.6
  toe[5] = 0.05
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
  torch.testing.assert_close(ratchet[1], torch.tensor(0.25))
  torch.testing.assert_close(ratchet[2], torch.tensor(0.295))
  torch.testing.assert_close(ratchet[3], torch.tensor(0.34))
  torch.testing.assert_close(ratchet[5], torch.tensor(0.25))
  torch.testing.assert_close(ratchet[6], torch.tensor(1.0))
  torch.testing.assert_close(ratchet[7], torch.tensor(0.36))
  torch.testing.assert_close(ratchet[8], torch.tensor(0.75))
  torch.testing.assert_close(term.ratchet_collision_upper_s[0], torch.tensor(0.34))
