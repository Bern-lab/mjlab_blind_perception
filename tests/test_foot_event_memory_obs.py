"""Tests for deployable-shaped foot event memory observations."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from mjlab.tasks.velocity.mdp.observations import (
  FOOT_EVENT_MEMORY_OBS_DIM,
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


def _make_term(env: Any) -> FootEventMemoryObs:
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
  return FootEventMemoryObs(SimpleNamespace(params=params), env)


def test_foot_event_memory_dim_matches_default_layout() -> None:
  assert FOOT_EVENT_MEMORY_OBS_DIM == 198
  assert foot_event_memory_obs_dim() == 198
  assert foot_event_memory_obs_dim(memory_len=3) == 99


def test_foot_event_memory_records_latched_touchdown() -> None:
  env = _make_env()
  term = _make_term(env)

  obs = term(env)
  assert obs.shape == (1, 99)
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
  term(env)

  env.episode_length_buf += 1
  env.extras[TOE_RISER_NEW_HIT_BY_FOOT_KEY] = torch.tensor([[False, True]])
  term(env)

  toe = term.toe_marks[0, 0]
  assert term.toe_valid.tolist() == [[True, False, False]]
  assert toe[0].item() == 1.0
  assert toe[2].item() == 1.0
  assert toe[3].item() == 1.0
  torch.testing.assert_close(toe[6], torch.tensor(0.4))
  torch.testing.assert_close(toe[9], torch.tensor(0.4))
  torch.testing.assert_close(toe[13], torch.tensor(0.7))
  torch.testing.assert_close(toe[15], torch.tensor(1.0))
