"""Tests for velocity step-boundary reward geometry helpers."""

import torch

from mjlab.tasks.velocity.mdp.rewards import _StepBoundaryFootVolume


def test_lip_min_dist_accepts_per_env_boundaries() -> None:
  reward = _StepBoundaryFootVolume.__new__(_StepBoundaryFootVolume)
  points = torch.tensor(
    [[[[0.0, 0.0, 0.10], [0.0, 0.20, 0.10]]]],
    dtype=torch.float32,
  )
  boundaries = torch.tensor(
    [[[0.0, 0.0, 0.10, 1.0, 0.0, 0.10, 0.0, -1.0, 0.0, 0.0, 0.10]]],
    dtype=torch.float32,
  )
  valid = torch.tensor([[True]])

  min_dist = reward._lip_min_dist(
    points,
    boundaries,
    valid,
    edge_height_band=0.06,
  )

  assert min_dist.shape == (1, 1, 2)
  torch.testing.assert_close(min_dist[0, 0], torch.tensor([0.0, 0.20]))
