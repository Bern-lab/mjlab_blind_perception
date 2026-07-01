from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import mjlab.tasks.velocity.mdp.observations as velocity_observations
import mjlab.tasks.velocity.mdp.rewards as velocity_rewards
import mjlab.tasks.velocity.mdp.temporal_stair_rewards as temporal_stair_rewards
from mjlab.envs.mdp.events import randomize_terrain
from mjlab.tasks.velocity.mdp.curriculums import (
  _terrain_family_name,
  terrain_levels_vel,
)
from mjlab.tasks.velocity.mdp.stair_geometry import (
  COLLISION_RISK_KEY,
  LANDING_QUALITY_KEY,
  LANDING_TOUCHDOWN_KEY,
  MINIMUM_SAFE_STRIDE_EXACT_KEY,
  MINIMUM_SAFE_STRIDE_KEY,
  MINIMUM_SAFE_STRIDE_UPPER_KEY,
  MINIMUM_SAFE_STRIDE_VALID_KEY,
  MINIMUM_SAFE_STRIDE_WEIGHT_KEY,
  STAIR_ENTRY_EVENT_KEY,
  STAIR_ENTRY_EVIDENCE_ASCENT_DIR_KEY,
  STAIR_ENTRY_RECENT_EVIDENCE_KEY,
  STAIR_PHASE_KEY,
  STAIR_RISER_HEIGHT_LABEL_KEY,
  STAIR_SHAPE_LABEL_VALID_KEY,
  STAIR_TREAD_DEPTH_LABEL_KEY,
  TOE_RISER_NEW_HIT_KEY,
  stair_shape_for_sequence,
  stair_shape_from_boundaries,
)
from mjlab.tasks.velocity.mdp.temporal_stair_rewards import (
  stair_aware_feet_gait,
  stair_tread_landing_reward,
)
from mjlab.tasks.velocity.mdp.temporal_stair_rewards import (
  toe_step_riser_slab_penalty as temporal_toe_step_riser_slab_penalty,
)


class _FakeScene:
  def __init__(self, asset, terrain, env_origins: torch.Tensor) -> None:
    self._asset = asset
    self.terrain = terrain
    self.env_origins = env_origins

  def __getitem__(self, name: str):
    assert name == "robot"
    return self._asset


class _FakeTerrain:
  def __init__(self, num_envs: int, num_levels: int = 10) -> None:
    self.cfg = SimpleNamespace(
      terrain_generator=SimpleNamespace(
        size=(8.0, 8.0),
        sub_terrains={"stairs": SimpleNamespace()},
      )
    )
    self.terrain_levels = torch.zeros(num_envs, dtype=torch.long)
    self.terrain_types = torch.zeros(num_envs, dtype=torch.long)
    self.terrain_origins = torch.zeros(num_levels, 1, 3)
    self.env_origins = torch.zeros(num_envs, 3)
    self.last_move_up: torch.Tensor | None = None
    self.last_move_down: torch.Tensor | None = None

  def update_env_origins(
    self,
    env_ids: torch.Tensor,
    move_up: torch.Tensor,
    move_down: torch.Tensor,
  ) -> None:
    self.last_move_up = move_up.clone()
    self.last_move_down = move_down.clone()
    self.terrain_levels[env_ids] += move_up.long()
    self.terrain_levels[env_ids] -= move_down.long()
    self.terrain_levels.clamp_(min=0)
    self.env_origins[env_ids] = self.terrain_origins[
      self.terrain_levels[env_ids], self.terrain_types[env_ids]
    ]


class _FakeCommandManager:
  def __init__(self, term) -> None:
    self._term = term

  def get_term(self, name: str):
    assert name == "twist"
    return self._term

  def get_command(self, name: str):
    assert name == "twist"
    return self._term.command


def _make_env(root_xy: torch.Tensor, command_term, num_levels: int = 10):
  num_envs = root_xy.shape[0]
  root_pos = torch.cat([root_xy, torch.zeros(num_envs, 1)], dim=1)
  root_quat = torch.zeros(num_envs, 4)
  root_quat[:, 0] = 1.0
  asset = SimpleNamespace(
    data=SimpleNamespace(root_link_pos_w=root_pos, root_link_quat_w=root_quat)
  )
  terrain = _FakeTerrain(num_envs, num_levels=num_levels)
  scene = _FakeScene(asset, terrain, terrain.env_origins)
  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    scene=scene,
    command_manager=_FakeCommandManager(command_term),
    max_episode_length_s=20.0,
    extras={},
  )
  return env, terrain


def test_terrain_levels_vel_uses_target_reached_for_target_episodes() -> None:
  command_term = SimpleNamespace(
    command=torch.zeros(3, 3),
    target_command_in_episode=torch.tensor([True, True, False]),
    target_reached_in_episode=torch.tensor([True, False, False]),
  )
  env, terrain = _make_env(
    torch.tensor(
      [
        [1.0, 0.0],  # Target reached should move up even below distance threshold.
        [5.0, 0.0],  # Target not reached should not move up despite distance.
        [5.0, 0.0],  # Non-target episode falls back to distance curriculum.
      ]
    ),
    command_term,
  )

  result = terrain_levels_vel(env, torch.arange(3), command_name="twist")

  assert terrain.last_move_up is not None
  assert terrain.last_move_up.tolist() == [True, False, True]
  assert result["target_attempted"].item() == torch.tensor(2 / 3).item()
  assert result["target_reached"].item() == torch.tensor(1 / 3).item()


def test_terrain_levels_vel_keeps_distance_rule_without_target_state() -> None:
  command_term = SimpleNamespace(command=torch.zeros(2, 3))
  env, terrain = _make_env(
    torch.tensor(
      [
        [5.0, 0.0],
        [1.0, 0.0],
      ]
    ),
    command_term,
  )

  result = terrain_levels_vel(env, torch.arange(2), command_name="twist")

  assert terrain.last_move_up is not None
  assert terrain.last_move_up.tolist() == [True, False]
  assert "target_attempted" not in result
  assert "target_reached" not in result


def test_fixed_width_terrain_variants_share_family_name() -> None:
  assert _terrain_family_name("high_stairs_w00") == "high_stairs"
  assert _terrain_family_name("high_stairs_w07") == "high_stairs"
  assert _terrain_family_name("high_stairs_inv_w03") == "high_stairs_inv"
  assert _terrain_family_name("gentle_slope") == "gentle_slope"


def test_terrain_levels_vel_mixed_replay_sticks_after_high_level() -> None:
  command_term = SimpleNamespace(command=torch.zeros(2, 3))
  env, terrain = _make_env(
    torch.tensor(
      [
        [5.0, 0.0],
        [1.0, 0.0],
      ]
    ),
    command_term,
    num_levels=10,
  )
  terrain.terrain_levels[0] = 7

  result = terrain_levels_vel(
    env,
    torch.arange(2),
    command_name="twist",
    mixed_replay_start_level=8,
    mixed_replay_level_ranges=((0, 0), (4, 4), (9, 9)),
    mixed_replay_weights=(1.0, 0.0, 0.0),
  )

  assert terrain.terrain_levels.tolist() == [0, 0]
  assert result["mixed_replay_active"].item() == torch.tensor(0.5).item()
  assert result["mixed_replay_low_ratio"].item() == torch.tensor(1.0).item()

  terrain_levels_vel(
    env,
    torch.tensor([0]),
    command_name="twist",
    mixed_replay_start_level=8,
    mixed_replay_level_ranges=((0, 0), (4, 4), (9, 9)),
    mixed_replay_weights=(0.0, 0.0, 1.0),
  )

  assert terrain.terrain_levels[0].item() == 9


def test_randomize_terrain_can_sample_weighted_level_buckets() -> None:
  num_envs = 8
  terrain = SimpleNamespace(
    cfg=SimpleNamespace(
      terrain_generator=SimpleNamespace(
        sub_terrains={
          "flat": SimpleNamespace(proportion=0.0),
          "stairs": SimpleNamespace(proportion=1.0),
          "rough": SimpleNamespace(proportion=0.0),
        }
      )
    ),
    terrain_origins=torch.zeros(10, 3, 3),
    env_origins=torch.zeros(num_envs, 3),
    terrain_levels=torch.zeros(num_envs, dtype=torch.long),
    terrain_types=torch.zeros(num_envs, dtype=torch.long),
  )
  scene = SimpleNamespace(terrain=terrain, env_origins=terrain.env_origins)
  env = SimpleNamespace(num_envs=num_envs, device=torch.device("cpu"), scene=scene)

  randomize_terrain(
    cast(Any, env),
    torch.arange(num_envs),
    level_ranges=((0, 0), (4, 4), (9, 9)),
    level_weights=(0.0, 1.0, 0.0),
    use_sub_terrain_proportions=True,
  )

  assert terrain.terrain_levels.tolist() == [4] * num_envs
  assert terrain.terrain_types.tolist() == [1] * num_envs


def test_stair_entry_progress_uses_world_ascent_direction() -> None:
  start = torch.tensor([[0.2, 0.1, 0.0], [0.2, 0.1, 0.0]])
  current = torch.tensor([[0.5, 0.3, 0.0], [0.5, 0.3, 0.0]])
  ascent_dir = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

  progress = temporal_toe_step_riser_slab_penalty._ascent_progress(
    current,
    start,
    ascent_dir,
  )

  assert progress.tolist() == pytest.approx([0.3, 0.2])


def test_stair_entry_heading_cos_uses_world_base_forward() -> None:
  root_quat_w = torch.tensor(
    [
      [1.0, 0.0, 0.0, 0.0],
      [0.70710678, 0.0, 0.0, 0.70710678],
    ]
  )
  ascent_dir = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

  heading_cos = temporal_toe_step_riser_slab_penalty._base_heading_cos(
    root_quat_w,
    ascent_dir,
  )

  assert heading_cos.tolist() == pytest.approx([1.0, 1.0], abs=1.0e-5)


def test_segment_distance_finds_interior_shank_edge_clearance() -> None:
  shank_start = torch.tensor([[0.0, 0.0, 0.0]])
  shank_end = torch.tensor([[0.0, 0.0, 1.0]])
  edge_start = torch.tensor([[-1.0, 0.03, 0.5]])
  edge_end = torch.tensor([[1.0, 0.03, 0.5]])

  distance, shank_t, edge_u = velocity_rewards._segment_to_segment_distance(
    shank_start,
    shank_end,
    edge_start,
    edge_end,
  )

  assert distance.tolist() == pytest.approx([0.03])
  assert shank_t.tolist() == pytest.approx([0.5])
  assert edge_u.tolist() == pytest.approx([0.5])


def test_segment_distance_clamps_to_edge_endpoint() -> None:
  shank_start = torch.tensor([[0.0, 0.0, 0.0]])
  shank_end = torch.tensor([[0.0, 0.0, 1.0]])
  edge_start = torch.tensor([[1.0, 0.0, 0.5]])
  edge_end = torch.tensor([[2.0, 0.0, 0.5]])

  distance, shank_t, edge_u = velocity_rewards._segment_to_segment_distance(
    shank_start,
    shank_end,
    edge_start,
    edge_end,
  )

  assert distance.tolist() == pytest.approx([1.0])
  assert shank_t.tolist() == pytest.approx([0.5])
  assert edge_u.tolist() == pytest.approx([0.0])


def test_semantic_shank_edge_filter_selects_per_leg_direction_and_layer() -> None:
  boundaries = torch.zeros(1, 5, 11)
  boundaries[0, :, 0:3] = torch.tensor(
    [
      [0.20, -1.0, 0.20],
      [0.20, -1.0, 0.20],
      [0.20, -1.0, 0.20],
      [0.20, 2.0, 0.20],
      [0.45, -1.0, 0.40],
    ]
  )
  boundaries[0, :, 3:6] = torch.tensor(
    [
      [0.20, 1.0, 0.20],
      [0.20, 1.0, 0.20],
      [0.20, 1.0, 0.20],
      [0.20, 3.0, 0.20],
      [0.45, 1.0, 0.40],
    ]
  )
  boundaries[0, :, 6:9] = torch.tensor(
    [
      [-1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [-1.0, 0.0, 0.0],
      [-1.0, 0.0, 0.0],
      [-1.0, 0.0, 0.0],
    ]
  )
  boundaries[0, :, 9] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.2])
  boundaries[0, :, 10] = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.4])
  shank_start = torch.tensor([[[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]]])
  shank_end = torch.tensor([[[0.0, 0.0, 0.2], [0.0, 0.0, 0.2]]])

  candidate = velocity_rewards._semantic_shank_edge_candidates(
    boundaries=boundaries,
    valid_boundaries=torch.ones(1, 5, dtype=torch.bool),
    boundary_sequence_ids=torch.tensor([[7, 7, 8, 7, 7]]),
    boundary_layers=torch.tensor([[1, 1, 1, 1, 2]]),
    foot_layers=torch.tensor([[0, 1]]),
    foot_layers_valid=torch.ones(1, 2, dtype=torch.bool),
    sequence_id=torch.tensor([7]),
    ascent_dir=torch.tensor([[1.0, 0.0]]),
    shank_start_w=shank_start,
    shank_end_w=shank_end,
    min_riser_height=0.04,
    direction_cos_threshold=0.85,
    lateral_margin=0.05,
  )

  assert candidate[0, 0].nonzero(as_tuple=False).flatten().tolist() == [0]
  assert candidate[0, 1].nonzero(as_tuple=False).flatten().tolist() == [4]


def test_stair_context_requires_two_explicit_flat_touchdowns_to_exit() -> None:
  active = torch.tensor([True, True, True])
  exit_candidate = torch.tensor([True, True, False])
  flat_steps = torch.zeros(3, dtype=torch.long)

  flat_steps, confirmed = (
    temporal_toe_step_riser_slab_penalty._advance_flat_exit_confirmation(
      active,
      exit_candidate,
      touchdown_now=torch.tensor([True, True, True]),
      unsafe_now=torch.tensor([False, True, False]),
      flat_steps=flat_steps,
    )
  )
  assert flat_steps.tolist() == [1, 0, 0]
  assert not bool(torch.any(confirmed))

  flat_steps, confirmed = (
    temporal_toe_step_riser_slab_penalty._advance_flat_exit_confirmation(
      active,
      exit_candidate,
      touchdown_now=torch.tensor([True, True, False]),
      unsafe_now=torch.zeros(3, dtype=torch.bool),
      flat_steps=flat_steps,
    )
  )
  assert flat_steps.tolist() == [2, 1, 0]
  assert confirmed.tolist() == [True, False, False]


def test_stair_entry_tread_support_fraction_detects_sixty_percent() -> None:
  sole_points_w = torch.tensor(
    [
      [
        [
          [-0.8, 0.0, 0.1],
          [-0.6, 0.0, 0.1],
          [-0.4, 0.0, 0.1],
          [0.1, 0.0, 0.1],
          [0.2, 0.0, 0.1],
        ]
      ]
    ]
  )
  boundaries = torch.tensor(
    [
      [
        [
          0.0,
          -0.5,
          0.1,
          0.0,
          0.5,
          0.1,
          1.0,
          0.0,
          0.0,
          0.0,
          0.1,
        ]
      ]
    ]
  )

  support_fraction = temporal_toe_step_riser_slab_penalty._tread_support_fraction(
    sole_points_w,
    boundaries,
    torch.tensor([1.0]),
  )

  assert support_fraction.shape == (1, 1, 1)
  assert support_fraction.item() == pytest.approx(0.6)


def test_landing_quality_increases_continuously_with_sole_support() -> None:
  quality = temporal_stair_rewards._landing_quality(
    coverage=torch.tensor([0.5, 0.6, 0.8, 1.0]),
    center_score=torch.ones(4),
    edge_clearance_score=torch.ones(4),
    valid=torch.ones(4, dtype=torch.bool),
  )

  assert quality.tolist() == pytest.approx([0.5, 0.6, 0.8, 1.0])


def test_stair_following_gait_does_not_require_fixed_phase() -> None:
  is_contact = torch.tensor([[True, False], [True, True], [False, False]])

  reward = stair_aware_feet_gait._phase_free_following_reward(is_contact)

  assert reward.tolist() == pytest.approx([1.0, 0.5, 0.0])


def test_stair_gait_uses_recent_entry_evidence(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  class _FakeContactSensor:
    def __init__(self) -> None:
      self.data = SimpleNamespace(
        current_contact_time=torch.tensor([[1.0, 0.0], [1.0, 1.0]])
      )

  monkeypatch.setattr(temporal_stair_rewards, "ContactSensor", _FakeContactSensor)
  root_quat = torch.zeros(2, 4)
  root_quat[:, 0] = 1.0
  asset = SimpleNamespace(data=SimpleNamespace(root_link_quat_w=root_quat))
  sensor = _FakeContactSensor()

  class _Scene:
    def __getitem__(self, name: str):
      if name == "robot":
        return asset
      if name == "feet_ground_contact":
        return sensor
      raise KeyError(name)

  command_manager = SimpleNamespace(
    get_command=lambda name: torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
  )
  env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    scene=_Scene(),
    command_manager=command_manager,
    episode_length_buf=torch.zeros(2),
    step_dt=0.02,
    extras={
      "log": {},
      STAIR_PHASE_KEY: torch.zeros(2, dtype=torch.long),
      STAIR_ENTRY_RECENT_EVIDENCE_KEY: torch.tensor([True, False]),
      STAIR_ENTRY_EVIDENCE_ASCENT_DIR_KEY: torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    },
  )
  gait = stair_aware_feet_gait(cast(Any, SimpleNamespace()), env)

  reward = gait(
    env,
    period=1.0,
    offset=[0.0, 0.5],
    threshold=0.5,
    command_threshold=0.05,
    command_name="twist",
    sensor_name="feet_ground_contact",
  )

  assert reward.tolist() == pytest.approx([1.0, 0.5])
  assert env.extras["log"]["Metrics/stair_recent_gait_active_ratio"] == 0.5
  assert env.extras["log"]["Metrics/stair_gait_active_ratio"] == 0.5


def test_event_label_ignores_any_toe_riser_hit_and_stair_label_uses_entry_evidence() -> (
  None
):
  env = SimpleNamespace(
    num_envs=3,
    device="cpu",
    extras={
      STAIR_ENTRY_EVENT_KEY: torch.zeros(3, dtype=torch.bool),
      TOE_RISER_NEW_HIT_KEY: torch.tensor([False, True, True]),
      STAIR_PHASE_KEY: torch.tensor([0, 1, 0]),
      STAIR_ENTRY_RECENT_EVIDENCE_KEY: torch.tensor([True, False, False]),
    },
  )

  event_label = velocity_observations.toe_riser_event_label(cast(Any, env))
  stair_label = velocity_observations.stair_state_label(cast(Any, env))

  torch.testing.assert_close(event_label, torch.zeros(3, 1))
  torch.testing.assert_close(stair_label, torch.tensor([[1.0], [1.0], [0.0]]))


def test_strict_stair_entry_event_only_requires_normal_phase_and_layer1_hit() -> None:
  entry = temporal_stair_rewards._strict_stair_entry_event(
    stair_phase=torch.tensor([0, 1, 0, 2]),
    layer1_new_hit=torch.tensor([True, True, False, True]),
  )

  torch.testing.assert_close(entry, torch.tensor([True, False, False, False]))


def test_repeat_layer1_hit_requires_active_phase_and_matching_sequence() -> None:
  repeat_hit = temporal_stair_rewards._repeat_sequence_layer1_hit(
    stair_phase=torch.tensor([0, 1, 2]),
    boundary_contact=torch.ones(3, 2, 1, dtype=torch.bool),
    contact_layers=torch.ones(3, 2, 1, dtype=torch.long),
    new_hit_by_foot=torch.tensor([[True, False], [True, False], [False, True]]),
    contact_sequence_ids=torch.tensor([[[4], [4]], [[5], [5]], [[6], [7]]]),
    sequence_id=torch.tensor([4, 5, 6]),
  )

  torch.testing.assert_close(repeat_hit, torch.tensor([False, True, False]))


def test_stair_attempt_always_alternates_but_only_expected_tread_advances() -> None:
  next_target, reached_layer, expected_layer = (
    temporal_stair_rewards._advance_stair_attempt(
      target_foot=torch.tensor([0, 1, 0, 1]),
      reached_layer=torch.tensor([0, 1, 2, 3]),
      expected_layer=torch.tensor([1, 2, 3, 4]),
      expected_tread_contact=torch.tensor([True, False, True, False]),
    )
  )

  torch.testing.assert_close(next_target, torch.tensor([1, 0, 1, 0]))
  torch.testing.assert_close(reached_layer, torch.tensor([1, 1, 3, 3]))
  torch.testing.assert_close(expected_layer, torch.tensor([2, 2, 4, 4]))


def test_clearance_layer_tracking_accepts_partial_tread_overlap() -> None:
  observed = temporal_stair_rewards._stair_tread_overlap_contact(
    active=torch.tensor([True, True, True, False]),
    ground_contact=torch.ones(4, 2, dtype=torch.bool),
    has_support_candidate=torch.ones(4, 2, dtype=torch.bool),
    support_fraction=torch.tensor(
      [[0.10, 0.60], [0.0, 0.20], [0.30, 0.40], [0.50, 0.50]]
    ),
    support_layer=torch.tensor([[1, 1], [1, 1], [0, 2], [1, 1]]),
  )

  torch.testing.assert_close(
    observed,
    torch.tensor(
      [
        [True, True],
        [False, True],
        [False, True],
        [False, False],
      ]
    ),
  )


def test_safe_stride_tracking_masks_only_stale_unreachable_targets() -> None:
  tracking_valid, stale = temporal_stair_rewards._safe_stride_tracking_masks(
    geometry_valid=torch.tensor([True, True, True, False]),
    raw_stride=torch.tensor([0.55, 0.80, 0.81, 4.0]),
    max_tracking_stride=0.80,
  )

  torch.testing.assert_close(
    tracking_valid,
    torch.tensor([True, True, False, False]),
  )
  torch.testing.assert_close(stale, torch.tensor([False, False, True, False]))


def test_blocked_swing_event_uses_abrupt_or_persistent_motion_residual() -> None:
  abrupt, persistent, event = temporal_stair_rewards._blocked_swing_event_evidence(
    vertical_boundary_contact=torch.ones(5, 1, dtype=torch.bool),
    previous_forward_speed=torch.tensor([[0.12], [0.01], [0.01], [0.12], [0.01]]),
    current_forward_speed=torch.tensor([[0.02], [0.01], [0.01], [0.10], [0.01]]),
    contact_steps=torch.tensor([[1], [3], [3], [1], [2]]),
    command_forward=torch.tensor([0.0, 0.2, 0.0, 0.2, 0.2]),
    min_forward_intent_speed=0.04,
    min_forward_speed_drop=0.04,
    blocked_forward_speed=0.03,
    persistent_steps=3,
    command_threshold=0.05,
  )

  torch.testing.assert_close(
    abrupt.squeeze(-1),
    torch.tensor([True, False, False, False, False]),
  )
  torch.testing.assert_close(
    persistent.squeeze(-1),
    torch.tensor([False, True, False, False, False]),
  )
  torch.testing.assert_close(
    event.squeeze(-1),
    torch.tensor([True, True, False, False, False]),
  )


def test_stair_touchdown_progress_uses_support_layer_delta() -> None:
  delta, normal_step, skipped_layers = temporal_stair_rewards._stair_touchdown_progress(
    touchdown_layer=torch.tensor([1, 2, 3, 0]),
    support_layer=torch.tensor([0, 0, 1, 1]),
  )

  torch.testing.assert_close(delta, torch.tensor([1, 2, 2, -1]))
  torch.testing.assert_close(
    normal_step,
    torch.tensor([True, False, False, False]),
  )
  torch.testing.assert_close(skipped_layers, torch.tensor([0, 1, 1, 0]))


def test_safe_stride_interval_fully_contains_target_foot_geometry() -> None:
  boundaries = torch.zeros(2, 2, 11)
  boundaries[:, :, 6] = -1.0
  boundaries[:, :, 4] = 1.0
  boundaries[:, 1, 0] = 0.30
  valid_boundaries = torch.tensor([[True, True], [True, False]])
  sequence_ids = torch.ones(2, 2, dtype=torch.long)
  layers = torch.tensor([[1, 2], [1, 2]])
  target_ref_w = torch.zeros(2, 3)
  sole_points_w = torch.tensor(
    [
      [[-0.10, 0.0, 0.0], [0.0, 0.0, 0.0], [0.10, 0.0, 0.0]],
      [[-0.10, 0.0, 0.0], [0.0, 0.0, 0.0], [0.10, 0.0, 0.0]],
    ]
  )

  lower, upper, geometry_valid = temporal_stair_rewards._minimum_safe_stride_to_layer(
    boundaries,
    valid_boundaries,
    sequence_ids,
    layers,
    torch.ones(2, dtype=torch.long),
    torch.full((2,), 2, dtype=torch.long),
    target_ref_w,
    sole_points_w,
    torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    torch.full((2,), 0.30),
    rear_clearance=0.02,
    front_clearance=0.02,
  )

  torch.testing.assert_close(lower, torch.tensor([0.42, 0.0]))
  torch.testing.assert_close(upper, torch.tensor([0.48, 0.0]))
  torch.testing.assert_close(geometry_valid, torch.tensor([True, False]))


def test_safe_stride_lower_bound_is_toe_gap_plus_complete_foot_length() -> None:
  boundaries = torch.zeros(1, 1, 11)
  boundaries[:, :, 4] = 1.0
  boundaries[:, :, 6] = -1.0
  target_ref_w = torch.tensor([[-0.15, 0.0, 0.0]])
  sole_points_w = torch.tensor([[[-0.245, 0.0, 0.0], [-0.058, 0.0, 0.0]]])

  lower, upper, valid = temporal_stair_rewards._minimum_safe_stride_to_layer(
    boundaries,
    torch.ones(1, 1, dtype=torch.bool),
    torch.ones(1, 1, dtype=torch.long),
    torch.ones(1, 1, dtype=torch.long),
    torch.ones(1, dtype=torch.long),
    torch.ones(1, dtype=torch.long),
    target_ref_w,
    sole_points_w,
    torch.tensor([[1.0, 0.0]]),
    torch.tensor([0.20]),
    rear_clearance=0.003,
    front_clearance=0.003,
  )

  torch.testing.assert_close(lower, torch.tensor([0.248]))
  torch.testing.assert_close(upper, torch.tensor([0.255]))
  torch.testing.assert_close(valid, torch.tensor([True]))


def test_slow_latent_labels_follow_env_stair_state_machine() -> None:
  env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    extras={
      STAIR_ENTRY_EVENT_KEY: torch.tensor([True, False]),
      TOE_RISER_NEW_HIT_KEY: torch.tensor([False, True]),
      STAIR_PHASE_KEY: torch.tensor([1, 2]),
      STAIR_TREAD_DEPTH_LABEL_KEY: torch.tensor([0.30, 0.35]),
      STAIR_RISER_HEIGHT_LABEL_KEY: torch.tensor([0.18, 0.20]),
      STAIR_SHAPE_LABEL_VALID_KEY: torch.tensor([True, True]),
      MINIMUM_SAFE_STRIDE_VALID_KEY: torch.tensor([False, True]),
      MINIMUM_SAFE_STRIDE_EXACT_KEY: torch.tensor([False, True]),
      MINIMUM_SAFE_STRIDE_WEIGHT_KEY: torch.tensor([0.0, 3.0]),
      MINIMUM_SAFE_STRIDE_KEY: torch.tensor([0.0, 0.41]),
      MINIMUM_SAFE_STRIDE_UPPER_KEY: torch.tensor([0.0, 0.46]),
      COLLISION_RISK_KEY: torch.tensor([0.75, 0.10]),
      LANDING_TOUCHDOWN_KEY: torch.tensor([True, False]),
      LANDING_QUALITY_KEY: torch.tensor([0.45, 0.0]),
    },
  )
  labels = torch.cat(
    [
      velocity_observations.toe_riser_event_label(cast(Any, env)),
      velocity_observations.stair_state_label(cast(Any, env)),
      velocity_observations.stair_shape_label(cast(Any, env)),
      velocity_observations.safe_stride_label(cast(Any, env)),
      velocity_observations.stair_future_event_labels(cast(Any, env)),
    ],
    dim=-1,
  )

  assert labels.shape == (2, 13)
  torch.testing.assert_close(
    labels,
    torch.tensor(
      [
        [
          1.0,
          1.0,
          0.30,
          0.18,
          1.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.75,
          1.0,
          0.45,
        ],
        [
          0.0,
          1.0,
          0.35,
          0.20,
          1.0,
          0.41,
          1.0,
          1.0,
          3.0,
          0.46,
          0.10,
          0.0,
          0.0,
        ],
      ]
    ),
  )


def test_safe_stride_label_masks_latched_values_while_flat() -> None:
  env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    extras={
      STAIR_PHASE_KEY: torch.tensor([0, 2]),
      MINIMUM_SAFE_STRIDE_KEY: torch.tensor([0.40, 0.41]),
      MINIMUM_SAFE_STRIDE_VALID_KEY: torch.tensor([True, True]),
      MINIMUM_SAFE_STRIDE_EXACT_KEY: torch.tensor([True, True]),
      MINIMUM_SAFE_STRIDE_WEIGHT_KEY: torch.tensor([3.0, 2.0]),
      MINIMUM_SAFE_STRIDE_UPPER_KEY: torch.tensor([0.45, 0.46]),
    },
  )

  labels = velocity_observations.safe_stride_label(cast(Any, env))

  torch.testing.assert_close(
    labels,
    torch.tensor([[0.40, 0.0, 0.0, 0.0, 0.0], [0.41, 1.0, 1.0, 2.0, 0.46]]),
  )


def test_stair_landing_target_uses_safe_center_plus_bounded_lead() -> None:
  target = stair_tread_landing_reward._safe_landing_target(
    safe_center_s=torch.tensor([0.10, 0.28, 0.01]),
    tread_depth=torch.tensor([0.30, 0.30, 0.30]),
    lead=0.04,
    back_margin=0.08,
    front_margin=0.08,
  )

  assert target.tolist() == pytest.approx([0.14, 0.22, 0.08])


def test_stair_shape_uses_parallel_boundary_spacing_and_riser_height() -> None:
  boundaries = torch.tensor(
    [
      [
        [0.0, 0.0, 0.10, 1.0, 0.0, 0.10, 0.0, -1.0, 0.0, 0.0, 0.10],
        [0.0, 0.30, 0.20, 1.0, 0.30, 0.20, 0.0, -1.0, 0.0, 0.10, 0.20],
      ]
    ]
  )
  valid = torch.tensor([[True, True]])

  tread_depth, riser_height, shape_valid = stair_shape_from_boundaries(
    boundaries, valid
  )

  assert tread_depth.tolist() == pytest.approx([0.30])
  assert riser_height.tolist() == pytest.approx([0.10])
  assert shape_valid.tolist() == [True]


def test_stair_shape_uses_only_selected_sequence_boundaries() -> None:
  boundaries = torch.tensor(
    [
      [
        [0.0, 0.0, 0.10, 1.0, 0.0, 0.10, 0.0, -1.0, 0.0, 0.0, 0.10],
        [0.0, 0.30, 0.20, 1.0, 0.30, 0.20, 0.0, -1.0, 0.0, 0.10, 0.20],
        [0.0, 0.50, 0.12, 1.0, 0.50, 0.12, 0.0, -1.0, 0.0, 0.0, 0.12],
        [0.0, 0.75, 0.24, 1.0, 0.75, 0.24, 0.0, -1.0, 0.0, 0.12, 0.24],
      ]
    ]
  )
  valid = torch.ones(1, 4, dtype=torch.bool)
  boundary_sequence_ids = torch.tensor([[1, 1, 2, 2]])

  tread_depth, riser_height, shape_valid = stair_shape_for_sequence(
    boundaries,
    valid,
    boundary_sequence_ids,
    torch.tensor([2]),
  )

  assert tread_depth.tolist() == pytest.approx([0.25])
  assert riser_height.tolist() == pytest.approx([0.12])
  assert shape_valid.tolist() == [True]


def test_stair_shape_label_is_masked_outside_active_sequence() -> None:
  env = SimpleNamespace(
    num_envs=3,
    device="cpu",
    extras={
      STAIR_PHASE_KEY: torch.tensor([0, 1, 2]),
      STAIR_TREAD_DEPTH_LABEL_KEY: torch.tensor([0.30, 0.31, 0.32]),
      STAIR_RISER_HEIGHT_LABEL_KEY: torch.tensor([0.10, 0.11, 0.12]),
      STAIR_SHAPE_LABEL_VALID_KEY: torch.tensor([True, True, False]),
    },
  )

  labels = velocity_observations.stair_shape_label(cast(Any, env))

  torch.testing.assert_close(
    labels,
    torch.tensor(
      [
        [0.30, 0.10, 0.0],
        [0.31, 0.11, 1.0],
        [0.32, 0.12, 0.0],
      ]
    ),
  )
