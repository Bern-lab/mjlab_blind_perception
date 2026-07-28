from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from scripts.velocity_eval.export_foot_event_detector_dataset import (
  FOOT_EVENT_LABEL_NAMES,
  FootEventDetectorDatasetBuilder,
  foot_event_detector_obs,
  foot_event_detector_obs_dim,
  foot_event_input_feature_groups,
  foot_event_labels_from_env,
  resolve_foot_event_detector_obs_dim,
)
from scripts.velocity_eval.export_stair_probe_dataset import input_feature_slices
from scripts.velocity_eval.train_foot_event_detector import (
  FootEventArrays,
  FootEventDetectorGRU,
  FootEventTorchDataset,
  TrainFootEventDetectorConfig,
  _event_indices,
  build_episode_split,
  compute_metrics,
  deployment_event_stats_for_label,
  make_pos_weight,
  metric_is_higher_better,
  run_train,
  sweep_deployment_event_thresholds,
)
from scripts.velocity_eval.train_foot_event_detector_online import (
  OnlineFootEventReplayBuffer,
  _baseline_guard_passed,
  _configure_footprint_only_model,
  _configure_toe_only_finetune,
  _configure_toe_riser_only_model,
  _deployment_event_metrics,
  _metric_improved,
  _mine_false_positive_touchdown_hard_negatives,
)

from mjlab.tasks.velocity.mdp.observations import _G1_LEG_JOINT_NAMES
from mjlab.tasks.velocity.mdp.stair_geometry import (
  STAIR_CURRENT_GROUND_CONTACT_KEY,
  STAIR_ENTRY_EVENT_KEY,
  TOE_RISER_NEW_HIT_BY_FOOT_KEY,
  TOE_RISER_NEW_HIT_KEY,
)


def _make_detector_arrays(
  num_samples: int = 48,
  *,
  history_len: int = 4,
  obs_dim: int = 91,
) -> FootEventArrays:
  obs = np.linspace(
    -1.0,
    1.0,
    num_samples * history_len * obs_dim,
    dtype=np.float32,
  ).reshape(num_samples, history_len, obs_dim)
  labels = np.zeros((num_samples, len(FOOT_EVENT_LABEL_NAMES)), dtype=np.float32)
  labels[:, 0] = (np.arange(num_samples) % 6 < 3).astype(np.float32)
  labels[:, 1] = (np.arange(num_samples) % 8 < 4).astype(np.float32)
  labels[2::10, 2] = 1.0
  labels[5::12, 3] = 1.0
  labels[7::14, 4] = 1.0
  labels[11::16, 5] = 1.0
  return FootEventArrays(
    obs_history=obs,
    obs_valid_mask=np.ones((num_samples, history_len), dtype=np.bool_),
    event_label=labels,
    episode_id=(np.arange(num_samples, dtype=np.int64) // 8),
    env_id=np.arange(num_samples, dtype=np.int64) % 8,
    frame_idx=np.arange(num_samples, dtype=np.int64),
    seed=np.full(num_samples, 42, dtype=np.int64),
  )


def test_foot_event_labels_detect_touchdown_and_expand_env_level_hits() -> None:
  env: Any = SimpleNamespace(
    num_envs=2,
    device=torch.device("cpu"),
    extras={
      STAIR_CURRENT_GROUND_CONTACT_KEY: torch.tensor([[True, False], [False, True]]),
      TOE_RISER_NEW_HIT_KEY: torch.tensor([True, False]),
      STAIR_ENTRY_EVENT_KEY: torch.tensor([False, True]),
    },
  )
  previous_contact = torch.tensor([[False, False], [False, True]])
  previous_contact_valid = torch.tensor([True, True])

  labels = foot_event_labels_from_env(
    cast(Any, env),
    previous_contact=previous_contact,
    previous_contact_valid=previous_contact_valid,
  )

  assert labels.tolist() == [
    [1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
  ]


def test_foot_event_labels_prefer_per_foot_toe_hit_extra() -> None:
  env: Any = SimpleNamespace(
    num_envs=2,
    device=torch.device("cpu"),
    extras={
      STAIR_CURRENT_GROUND_CONTACT_KEY: torch.tensor([[False, False], [False, False]]),
      TOE_RISER_NEW_HIT_KEY: torch.tensor([True, True]),
      TOE_RISER_NEW_HIT_BY_FOOT_KEY: torch.tensor([[True, False], [False, True]]),
    },
  )

  labels = foot_event_labels_from_env(
    cast(Any, env),
    previous_contact=torch.zeros(2, 2, dtype=torch.bool),
    previous_contact_valid=torch.ones(2, dtype=torch.bool),
  )

  assert labels[:, 4:6].tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_touchdown_label_is_suppressed_until_previous_contact_is_valid() -> None:
  env: Any = SimpleNamespace(
    num_envs=1,
    device=torch.device("cpu"),
    extras={
      STAIR_CURRENT_GROUND_CONTACT_KEY: torch.tensor([[True, False]]),
    },
  )

  labels = foot_event_labels_from_env(
    cast(Any, env),
    previous_contact=torch.tensor([[False, False]]),
    previous_contact_valid=torch.tensor([False]),
  )

  assert labels[0, 0].item() == 1.0
  assert labels[0, 2].item() == 0.0


def test_detector_builder_collects_only_requested_full_histories() -> None:
  builder = FootEventDetectorDatasetBuilder(
    num_envs=2,
    history_len=2,
    obs_dim=3,
    max_samples=4,
    device="cpu",
  )
  builder.push_observations(torch.ones(2, 3), torch.tensor([True, True]))
  builder.collect(
    labels=torch.zeros(2, len(FOOT_EVENT_LABEL_NAMES)),
    collect_mask=torch.tensor([False, False]),
    episode_id=torch.tensor([0, 1]),
    frame_idx=0,
    seed=42,
  )
  builder.push_observations(torch.full((2, 3), 2.0), torch.tensor([False, False]))
  builder.collect(
    labels=torch.ones(2, len(FOOT_EVENT_LABEL_NAMES)),
    collect_mask=builder.history.valid_mask.all(dim=1),
    episode_id=torch.tensor([0, 1]),
    frame_idx=1,
    seed=42,
  )

  arrays = builder.as_arrays()

  assert arrays["obs_history"].shape == (2, 2, 3)
  assert arrays["obs_valid_mask"].tolist() == [[True, True], [True, True]]
  assert arrays["event_label"].shape == (2, len(FOOT_EVENT_LABEL_NAMES))


def test_detector_builder_can_collect_without_sample_cap() -> None:
  builder = FootEventDetectorDatasetBuilder(
    num_envs=1,
    history_len=1,
    obs_dim=2,
    max_samples=None,
    device="cpu",
  )
  for frame in range(3):
    builder.push_observations(
      torch.full((1, 2), float(frame)),
      torch.tensor([frame == 0]),
    )
    builder.collect(
      labels=torch.ones(1, len(FOOT_EVENT_LABEL_NAMES)),
      collect_mask=torch.tensor([True]),
      episode_id=torch.tensor([0]),
      frame_idx=frame,
      seed=42,
    )

  assert builder.num_samples == 3
  assert not builder.is_full


def test_foot_event_detector_forward_shape() -> None:
  model = FootEventDetectorGRU(
    obs_dim=91,
    frame_hidden_dim=16,
    recurrent_hidden_dim=16,
    head_hidden_dim=8,
  )

  logits = model(torch.zeros(5, 4, 91))

  assert logits.shape == (5, len(FOOT_EVENT_LABEL_NAMES))


def test_footprint_v2_schema_adds_deployable_timing_features() -> None:
  groups = foot_event_input_feature_groups(
    include_gait_phase=True,
    input_schema="footprint_v2",
  )
  names = [group["name"] for group in groups]
  widths = [int(group["width"]) for group in groups]

  assert foot_event_detector_obs_dim(include_gait_phase=True) == 93
  assert sum(widths) == 133
  assert (
    resolve_foot_event_detector_obs_dim(
      None,
      include_gait_phase=True,
      input_schema="footprint_v2",
    )
    == 133
  )
  assert names[-13:] == [
    "left_heel_vel_body",
    "right_heel_vel_body",
    "left_heel_vel_delta",
    "right_heel_vel_delta",
    "left_sole_center_pos_body",
    "right_sole_center_pos_body",
    "left_sole_center_vel_body",
    "right_sole_center_vel_body",
    "left_sole_pitch_proxy",
    "right_sole_pitch_proxy",
    "command_lin_y",
    "command_yaw_rate",
    "action_delta_leg",
  ]


def test_footprint_v2_obs_computes_reset_safe_heel_and_action_features() -> None:
  num_envs = 2
  device = torch.device("cpu")
  site_pos_w = torch.tensor(
    [
      [
        [0.20, 0.10, 0.00],
        [0.20, -0.10, 0.00],
        [0.00, 0.10, -0.02],
        [0.00, -0.10, -0.02],
      ],
      [
        [0.30, 0.10, 0.01],
        [0.30, -0.10, 0.01],
        [0.10, 0.10, -0.01],
        [0.10, -0.10, -0.01],
      ],
    ],
    dtype=torch.float32,
  )
  robot = SimpleNamespace(
    site_names=["left_toe", "right_toe", "left_heel", "right_heel"],
    joint_names=list(_G1_LEG_JOINT_NAMES),
    data=SimpleNamespace(
      root_link_pos_w=torch.zeros(num_envs, 3),
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1),
      site_pos_w=site_pos_w,
      default_joint_pos=torch.zeros(num_envs, len(_G1_LEG_JOINT_NAMES)),
    ),
  )
  action = torch.zeros(num_envs, len(_G1_LEG_JOINT_NAMES))
  action[:, 0] = 0.25
  command = torch.tensor([[0.5, 0.2, -0.1], [0.3, -0.4, 0.6]])
  env: Any = SimpleNamespace(
    num_envs=num_envs,
    device=device,
    step_dt=0.02,
    episode_length_buf=torch.arange(num_envs),
    extras={},
    scene={"robot": robot},
    action_manager=SimpleNamespace(
      action=action,
      total_action_dim=len(_G1_LEG_JOINT_NAMES),
    ),
    command_manager=SimpleNamespace(get_command=lambda _name: command),
  )
  latent = torch.zeros(num_envs, 91)
  latent[:, input_feature_slices()["previous_action_leg"].start] = 0.10

  first = foot_event_detector_obs(
    {"latent": latent},
    cast(Any, env),
    input_schema="footprint_v2",
    include_gait_phase=False,
    gait_period=0.6,
    command_name="twist",
    reset_mask=torch.ones(num_envs, dtype=torch.bool),
  )
  robot.data.site_pos_w = site_pos_w + torch.tensor([0.02, 0.0, 0.0])
  second = foot_event_detector_obs(
    {"latent": latent},
    cast(Any, env),
    input_schema="footprint_v2",
    include_gait_phase=False,
    gait_period=0.6,
    command_name="twist",
    reset_mask=torch.zeros(num_envs, dtype=torch.bool),
  )

  extra_start = 91
  command_start = extra_start + 26
  action_delta_start = extra_start + 28
  assert first.shape == (num_envs, 131)
  assert first[:, extra_start : extra_start + 6].abs().sum().item() == 0.0
  torch.testing.assert_close(second[:, extra_start], torch.ones(num_envs))
  torch.testing.assert_close(first[:, command_start], command[:, 1])
  torch.testing.assert_close(first[:, command_start + 1], command[:, 2])
  torch.testing.assert_close(
    first[:, action_delta_start],
    torch.full((num_envs,), 0.15),
  )

  slow_latent = torch.cat(
    (latent, torch.full((num_envs, 82), 7.0, dtype=torch.float32)),
    dim=-1,
  )
  slow_latent_obs = foot_event_detector_obs(
    {"latent": slow_latent},
    cast(Any, env),
    input_schema="footprint_v2",
    include_gait_phase=True,
    gait_period=0.6,
    command_name="twist",
    reset_mask=torch.ones(num_envs, dtype=torch.bool),
  )
  assert slow_latent_obs.shape == (num_envs, 133)
  torch.testing.assert_close(slow_latent_obs[:, :91], latent)
  assert slow_latent_obs[:, 91:93].abs().max().item() <= 1.0
  assert slow_latent_obs[:, 93:].abs().max().item() < 7.0


def test_torch_dataset_uses_last_history_frames() -> None:
  arrays = _make_detector_arrays(num_samples=4, history_len=6)
  dataset = FootEventTorchDataset(arrays, np.array([0]), history_len=3)

  item = dataset[0]

  expected = torch.as_tensor(arrays.obs_history[0, -3:, :], dtype=torch.float32)
  assert torch.equal(item["obs_history"], expected)
  assert item["event_label"].shape == (len(FOOT_EVENT_LABEL_NAMES),)


def test_episode_split_keeps_episodes_disjoint() -> None:
  arrays = _make_detector_arrays(num_samples=48)

  split = build_episode_split(arrays.episode_id, val_fraction=0.25, seed=123)

  assert split.train_indices.size > 0
  assert split.val_indices.size > 0
  assert set(split.train_episode_ids.tolist()).isdisjoint(
    set(split.val_episode_ids.tolist())
  )


def test_single_episode_split_falls_back_to_sample_split() -> None:
  episode_id = np.zeros(12, dtype=np.int64)

  split = build_episode_split(episode_id, val_fraction=0.25, seed=123)

  assert split.train_indices.size == 9
  assert split.val_indices.size == 3
  assert split.train_episode_ids.tolist() == [0]
  assert split.val_episode_ids.tolist() == [0]


def test_event_metrics_match_with_timing_tolerance() -> None:
  labels = np.zeros((6, len(FOOT_EVENT_LABEL_NAMES)), dtype=np.float32)
  probabilities = np.zeros_like(labels)
  labels[2, 2] = 1.0
  probabilities[4, 2] = 1.0

  metrics = compute_metrics(
    labels=labels,
    probabilities=probabilities,
    episode_id=np.zeros(6, dtype=np.int64),
    frame_idx=np.arange(10, 16, dtype=np.int64),
    val_loss=1.25,
    threshold=0.5,
    event_tolerance_frames=2,
  )

  assert metrics["left_touchdown_event_f1"] == 1.0


def test_event_indices_treat_frame_gaps_as_new_runs() -> None:
  flags = np.array([True, True, True], dtype=np.bool_)
  frames = np.array([1, 2, 6], dtype=np.int64)

  assert _event_indices(flags, frames).tolist() == [1, 6]


def test_deployment_touchdown_gate_filters_repeated_stance_triggers() -> None:
  labels = np.zeros((6, len(FOOT_EVENT_LABEL_NAMES)), dtype=np.float32)
  probabilities = np.zeros_like(labels)
  labels[2:, 0] = 1.0
  labels[2, 2] = 1.0
  probabilities[:, 0] = np.array([0.0, 0.1, 0.9, 0.95, 0.95, 0.95])
  probabilities[[2, 5], 2] = 0.9

  stats = deployment_event_stats_for_label(
    labels=labels,
    probabilities=probabilities,
    episode_id=np.zeros(6, dtype=np.int64),
    frame_idx=np.arange(6, dtype=np.int64),
    event_label_index=2,
    threshold=0.5,
    tolerance_frames=0,
    contact_label_index=0,
    contact_prob_index=0,
    contact_threshold=0.7,
    contact_release_threshold=0.35,
  )

  assert stats["event_f1"] == 1.0
  assert stats["predicted_event_count"] == 1.0
  assert stats["predicted_contact_precision"] == 1.0


def test_deployment_touchdown_contact_fallback_recovers_contact_rising_edge() -> None:
  labels = np.zeros((6, len(FOOT_EVENT_LABEL_NAMES)), dtype=np.float32)
  probabilities = np.zeros_like(labels)
  labels[2:, 0] = 1.0
  labels[2, 2] = 1.0
  probabilities[:, 0] = np.array([0.0, 0.1, 0.9, 0.95, 0.95, 0.95])
  probabilities[:, 2] = 0.05

  no_fallback = deployment_event_stats_for_label(
    labels=labels,
    probabilities=probabilities,
    episode_id=np.zeros(6, dtype=np.int64),
    frame_idx=np.arange(6, dtype=np.int64),
    event_label_index=2,
    threshold=0.5,
    tolerance_frames=0,
    contact_label_index=0,
    contact_prob_index=0,
    contact_threshold=0.7,
    contact_release_threshold=0.35,
  )
  fallback = deployment_event_stats_for_label(
    labels=labels,
    probabilities=probabilities,
    episode_id=np.zeros(6, dtype=np.int64),
    frame_idx=np.arange(6, dtype=np.int64),
    event_label_index=2,
    threshold=0.5,
    tolerance_frames=0,
    contact_label_index=0,
    contact_prob_index=0,
    contact_threshold=0.7,
    contact_release_threshold=0.35,
    contact_rising_fallback=True,
  )

  assert no_fallback["event_recall"] == 0.0
  assert fallback["event_f1"] == 1.0
  assert fallback["predicted_event_count"] == 1.0


def test_threshold_sweep_prefers_clean_deployment_event_threshold() -> None:
  labels = np.zeros((6, len(FOOT_EVENT_LABEL_NAMES)), dtype=np.float32)
  probabilities = np.zeros_like(labels)
  labels[2, 4] = 1.0
  probabilities[2, 4] = 0.8
  probabilities[4, 4] = 0.4

  stats = sweep_deployment_event_thresholds(
    labels=labels,
    probabilities=probabilities,
    episode_id=np.zeros(6, dtype=np.int64),
    frame_idx=np.arange(6, dtype=np.int64),
    event_label_index=4,
    thresholds=np.array([0.3, 0.7], dtype=np.float32),
    tolerance_frames=0,
  )

  assert stats["best_threshold"] == np.float32(0.7).item()
  assert stats["best_event_f1"] == 1.0


def test_threshold_sweep_reports_high_recall_operating_point() -> None:
  labels = np.zeros((4, len(FOOT_EVENT_LABEL_NAMES)), dtype=np.float32)
  probabilities = np.zeros_like(labels)
  labels[[1, 2], 2] = 1.0
  probabilities[1, 2] = 0.9
  probabilities[2, 2] = 0.4

  stats = sweep_deployment_event_thresholds(
    labels=labels,
    probabilities=probabilities,
    episode_id=np.zeros(4, dtype=np.int64),
    frame_idx=np.arange(4, dtype=np.int64),
    event_label_index=2,
    thresholds=np.array([0.3, 0.7], dtype=np.float32),
    tolerance_frames=0,
    recall_precision_floor=1.0,
  )

  assert stats["best_threshold"] == np.float32(0.3).item()
  assert stats["high_recall_threshold"] == np.float32(0.3).item()
  assert stats["high_recall_event_recall"] == 1.0
  assert stats["high_recall_event_precision"] == 1.0


def test_pos_weight_is_clipped_and_handles_empty_positive_class() -> None:
  labels = np.zeros((10, len(FOOT_EVENT_LABEL_NAMES)), dtype=np.float32)
  labels[:1, 0] = 1.0

  weights = make_pos_weight(labels, max_weight=4.0)

  assert weights[0].item() == 4.0
  assert weights[1:].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_pos_weight_can_override_toe_hit_classes() -> None:
  labels = np.zeros((10, len(FOOT_EVENT_LABEL_NAMES)), dtype=np.float32)

  weights = make_pos_weight(labels, max_weight=4.0, toe_hit_pos_weight=12.0)

  assert weights[4:6].tolist() == [12.0, 12.0]


def test_online_replay_buffer_samples_rare_events() -> None:
  buffer = OnlineFootEventReplayBuffer(
    capacity=8,
    history_len=2,
    obs_dim=3,
    label_dim=len(FOOT_EVENT_LABEL_NAMES),
    num_envs=8,
    soft_touchdown_radius=1,
    soft_toe_hit_radius=2,
    soft_event_radius1_value=0.7,
    soft_event_radius2_value=0.4,
    device=torch.device("cpu"),
  )
  labels = torch.zeros(8, len(FOOT_EVENT_LABEL_NAMES))
  labels[1, 2] = 1.0
  labels[3, 4] = 1.0
  buffer.add(
    obs_history=torch.arange(8 * 2 * 3, dtype=torch.float32).reshape(8, 2, 3),
    labels=labels,
    train_labels=labels.clone(),
    episode_id=torch.arange(8),
    frame_idx=torch.arange(8),
    env_id=torch.arange(8),
    stair_support=torch.zeros(8, 2, dtype=torch.bool),
    support_fraction=torch.zeros(8, 2),
  )
  generator = torch.Generator()
  generator.manual_seed(123)

  _obs, sampled_labels = buffer.sample(
    6,
    generator=generator,
    toe_positive_fraction=0.5,
    toe_soft_positive_fraction=0.0,
    touchdown_positive_fraction=0.5,
    touchdown_soft_positive_fraction=0.0,
    stair_hard_negative_fraction=0.0,
    false_positive_hard_negative_fraction=0.0,
    false_negative_hard_positive_fraction=0.0,
    false_negative_toe_hard_positive_fraction=0.0,
    toe_soft_positive_threshold=0.3,
    touchdown_soft_positive_threshold=0.3,
  )

  assert sampled_labels[:, 2:6].sum().item() > 0.0


def test_online_replay_buffer_retroactively_softens_event_neighbors() -> None:
  buffer = OnlineFootEventReplayBuffer(
    capacity=4,
    history_len=1,
    obs_dim=2,
    label_dim=len(FOOT_EVENT_LABEL_NAMES),
    num_envs=1,
    soft_touchdown_radius=1,
    soft_toe_hit_radius=2,
    soft_event_radius1_value=0.7,
    soft_event_radius2_value=0.4,
    device=torch.device("cpu"),
  )
  labels0 = torch.zeros(1, len(FOOT_EVENT_LABEL_NAMES))
  labels1 = torch.zeros(1, len(FOOT_EVENT_LABEL_NAMES))
  labels1[0, 4] = 1.0
  common = {
    "obs_history": torch.zeros(1, 1, 2),
    "episode_id": torch.zeros(1, dtype=torch.int64),
    "env_id": torch.zeros(1, dtype=torch.int64),
    "stair_support": torch.zeros(1, 2, dtype=torch.bool),
    "support_fraction": torch.zeros(1, 2),
  }

  buffer.add(
    labels=labels0,
    train_labels=labels0.clone(),
    frame_idx=torch.zeros(1, dtype=torch.int64),
    **common,
  )
  buffer.add(
    labels=labels1,
    train_labels=labels1.clone(),
    frame_idx=torch.ones(1, dtype=torch.int64),
    **common,
  )
  snapshot = buffer.snapshot()

  assert snapshot["labels"][0, 4] == 0.0
  assert snapshot["train_labels"][0, 4] == np.float32(0.7).item()
  assert snapshot["train_labels"][1, 4] == 1.0


def test_online_replay_buffer_samples_soft_touchdown_neighbors() -> None:
  buffer = OnlineFootEventReplayBuffer(
    capacity=4,
    history_len=1,
    obs_dim=1,
    label_dim=len(FOOT_EVENT_LABEL_NAMES),
    num_envs=1,
    soft_touchdown_radius=1,
    soft_toe_hit_radius=0,
    soft_event_radius1_value=0.7,
    soft_event_radius2_value=0.4,
    device=torch.device("cpu"),
  )
  labels0 = torch.zeros(1, len(FOOT_EVENT_LABEL_NAMES))
  labels1 = torch.zeros(1, len(FOOT_EVENT_LABEL_NAMES))
  labels1[0, 2] = 1.0
  common = {
    "obs_history": torch.zeros(1, 1, 1),
    "episode_id": torch.zeros(1, dtype=torch.int64),
    "env_id": torch.zeros(1, dtype=torch.int64),
    "stair_support": torch.zeros(1, 2, dtype=torch.bool),
    "support_fraction": torch.zeros(1, 2),
  }
  buffer.add(
    labels=labels0,
    train_labels=labels0.clone(),
    frame_idx=torch.zeros(1, dtype=torch.int64),
    **common,
  )
  buffer.add(
    labels=labels1,
    train_labels=labels1.clone(),
    frame_idx=torch.ones(1, dtype=torch.int64),
    **common,
  )

  generator = torch.Generator()
  generator.manual_seed(7)
  _obs, sampled_labels = buffer.sample(
    4,
    generator=generator,
    toe_positive_fraction=0.0,
    toe_soft_positive_fraction=0.0,
    touchdown_positive_fraction=0.0,
    touchdown_soft_positive_fraction=1.0,
    stair_hard_negative_fraction=0.0,
    false_positive_hard_negative_fraction=0.0,
    false_negative_hard_positive_fraction=0.0,
    false_negative_toe_hard_positive_fraction=0.0,
    toe_soft_positive_threshold=0.3,
    touchdown_soft_positive_threshold=0.3,
  )

  assert np.allclose(sampled_labels[:, 2].numpy(), 0.7)


def test_online_replay_buffer_samples_soft_toe_neighbors() -> None:
  buffer = OnlineFootEventReplayBuffer(
    capacity=4,
    history_len=1,
    obs_dim=1,
    label_dim=len(FOOT_EVENT_LABEL_NAMES),
    num_envs=1,
    soft_touchdown_radius=0,
    soft_toe_hit_radius=1,
    soft_event_radius1_value=0.7,
    soft_event_radius2_value=0.4,
    device=torch.device("cpu"),
  )
  labels0 = torch.zeros(1, len(FOOT_EVENT_LABEL_NAMES))
  labels1 = torch.zeros(1, len(FOOT_EVENT_LABEL_NAMES))
  labels1[0, 4] = 1.0
  common = {
    "obs_history": torch.zeros(1, 1, 1),
    "episode_id": torch.zeros(1, dtype=torch.int64),
    "env_id": torch.zeros(1, dtype=torch.int64),
    "stair_support": torch.zeros(1, 2, dtype=torch.bool),
    "support_fraction": torch.zeros(1, 2),
  }
  buffer.add(
    labels=labels0,
    train_labels=labels0.clone(),
    frame_idx=torch.zeros(1, dtype=torch.int64),
    **common,
  )
  buffer.add(
    labels=labels1,
    train_labels=labels1.clone(),
    frame_idx=torch.ones(1, dtype=torch.int64),
    **common,
  )

  generator = torch.Generator()
  generator.manual_seed(8)
  _obs, sampled_labels = buffer.sample(
    4,
    generator=generator,
    toe_positive_fraction=0.0,
    toe_soft_positive_fraction=1.0,
    touchdown_positive_fraction=0.0,
    touchdown_soft_positive_fraction=0.0,
    stair_hard_negative_fraction=0.0,
    false_positive_hard_negative_fraction=0.0,
    false_negative_hard_positive_fraction=0.0,
    false_negative_toe_hard_positive_fraction=0.0,
    toe_soft_positive_threshold=0.3,
    touchdown_soft_positive_threshold=0.3,
  )

  assert np.allclose(sampled_labels[:, 4].numpy(), 0.7)


def test_online_replay_buffer_marks_false_positive_hard_negatives() -> None:
  buffer = OnlineFootEventReplayBuffer(
    capacity=4,
    history_len=1,
    obs_dim=2,
    label_dim=len(FOOT_EVENT_LABEL_NAMES),
    num_envs=2,
    soft_touchdown_radius=1,
    soft_toe_hit_radius=2,
    soft_event_radius1_value=0.7,
    soft_event_radius2_value=0.4,
    device=torch.device("cpu"),
  )
  labels = torch.zeros(2, len(FOOT_EVENT_LABEL_NAMES))
  buffer.add(
    obs_history=torch.zeros(2, 1, 2),
    labels=labels,
    train_labels=labels.clone(),
    episode_id=torch.zeros(2, dtype=torch.int64),
    frame_idx=torch.arange(2, dtype=torch.int64),
    env_id=torch.arange(2, dtype=torch.int64),
    stair_support=torch.zeros(2, 2, dtype=torch.bool),
    support_fraction=torch.zeros(2, 2),
  )

  added = buffer.mark_false_positive_hard_negatives(
    np.array([True, False], dtype=np.bool_)
  )

  assert added == 1
  assert buffer.snapshot()["false_positive_hard_negative"].tolist() == [True, False]


def test_online_replay_buffer_stores_footprint_anchor_w() -> None:
  buffer = OnlineFootEventReplayBuffer(
    capacity=4,
    history_len=1,
    obs_dim=2,
    label_dim=len(FOOT_EVENT_LABEL_NAMES),
    num_envs=2,
    soft_touchdown_radius=1,
    soft_toe_hit_radius=2,
    soft_event_radius1_value=0.7,
    soft_event_radius2_value=0.4,
    device=torch.device("cpu"),
  )
  labels = torch.zeros(2, len(FOOT_EVENT_LABEL_NAMES))
  footprint_anchor_w = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)

  buffer.add(
    obs_history=torch.zeros(2, 1, 2),
    labels=labels,
    train_labels=labels.clone(),
    episode_id=torch.zeros(2, dtype=torch.int64),
    frame_idx=torch.arange(2, dtype=torch.int64),
    env_id=torch.arange(2, dtype=torch.int64),
    stair_support=torch.zeros(2, 2, dtype=torch.bool),
    support_fraction=torch.zeros(2, 2),
    footprint_anchor_w=footprint_anchor_w,
  )

  np.testing.assert_allclose(
    buffer.snapshot()["footprint_anchor_w"],
    footprint_anchor_w.numpy(),
  )


def test_touchdown_false_positive_mining_marks_negative_windows() -> None:
  class TouchdownFalsePositiveModel(torch.nn.Module):
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
      logits = torch.full(
        (obs.shape[0], len(FOOT_EVENT_LABEL_NAMES)),
        -10.0,
        dtype=obs.dtype,
        device=obs.device,
      )
      active_logit = torch.where(
        obs[:, -1, 0] > 0.5,
        torch.full((obs.shape[0],), 10.0, dtype=obs.dtype, device=obs.device),
        torch.full((obs.shape[0],), -10.0, dtype=obs.dtype, device=obs.device),
      )
      logits[:, 0] = active_logit
      logits[:, 2] = active_logit
      return logits

  buffer = OnlineFootEventReplayBuffer(
    capacity=4,
    history_len=1,
    obs_dim=2,
    label_dim=len(FOOT_EVENT_LABEL_NAMES),
    num_envs=1,
    soft_touchdown_radius=1,
    soft_toe_hit_radius=2,
    soft_event_radius1_value=0.7,
    soft_event_radius2_value=0.4,
    device=torch.device("cpu"),
  )
  labels = torch.zeros(4, len(FOOT_EVENT_LABEL_NAMES))
  obs_history = torch.zeros(4, 1, 2)
  obs_history[0, 0, 0] = 1.0
  buffer.add(
    obs_history=obs_history,
    labels=labels,
    train_labels=labels.clone(),
    episode_id=torch.zeros(4, dtype=torch.int64),
    frame_idx=torch.arange(4, dtype=torch.int64),
    env_id=torch.zeros(4, dtype=torch.int64),
    stair_support=torch.zeros(4, 2, dtype=torch.bool),
    support_fraction=torch.zeros(4, 2),
  )

  added = _mine_false_positive_touchdown_hard_negatives(
    cast(FootEventDetectorGRU, TouchdownFalsePositiveModel()),
    buffer,
    device=torch.device("cpu"),
    batch_size=4,
    threshold=0.5,
    contact_threshold=0.5,
    window_frames=1,
    max_peaks=4,
  )

  assert added == 2
  assert buffer.snapshot()["false_positive_hard_negative"].tolist() == [
    True,
    True,
    False,
    False,
  ]


def test_online_replay_buffer_marks_false_negative_hard_positives() -> None:
  buffer = OnlineFootEventReplayBuffer(
    capacity=4,
    history_len=1,
    obs_dim=2,
    label_dim=len(FOOT_EVENT_LABEL_NAMES),
    num_envs=2,
    soft_touchdown_radius=1,
    soft_toe_hit_radius=2,
    soft_event_radius1_value=0.7,
    soft_event_radius2_value=0.4,
    device=torch.device("cpu"),
  )
  labels = torch.zeros(2, len(FOOT_EVENT_LABEL_NAMES))
  buffer.add(
    obs_history=torch.zeros(2, 1, 2),
    labels=labels,
    train_labels=labels.clone(),
    episode_id=torch.zeros(2, dtype=torch.int64),
    frame_idx=torch.arange(2, dtype=torch.int64),
    env_id=torch.arange(2, dtype=torch.int64),
    stair_support=torch.zeros(2, 2, dtype=torch.bool),
    support_fraction=torch.zeros(2, 2),
  )

  added = buffer.mark_false_negative_hard_positives(
    np.array([False, True], dtype=np.bool_)
  )

  assert added == 1
  assert buffer.snapshot()["false_negative_hard_positive"].tolist() == [False, True]


def test_online_replay_buffer_marks_false_negative_toe_hard_positives() -> None:
  buffer = OnlineFootEventReplayBuffer(
    capacity=4,
    history_len=1,
    obs_dim=2,
    label_dim=len(FOOT_EVENT_LABEL_NAMES),
    num_envs=2,
    soft_touchdown_radius=1,
    soft_toe_hit_radius=2,
    soft_event_radius1_value=0.7,
    soft_event_radius2_value=0.4,
    device=torch.device("cpu"),
  )
  labels = torch.zeros(2, len(FOOT_EVENT_LABEL_NAMES))
  buffer.add(
    obs_history=torch.zeros(2, 1, 2),
    labels=labels,
    train_labels=labels.clone(),
    episode_id=torch.zeros(2, dtype=torch.int64),
    frame_idx=torch.arange(2, dtype=torch.int64),
    env_id=torch.arange(2, dtype=torch.int64),
    stair_support=torch.zeros(2, 2, dtype=torch.bool),
    support_fraction=torch.zeros(2, 2),
  )

  added = buffer.mark_false_negative_toe_hard_positives(
    np.array([True, False], dtype=np.bool_)
  )

  assert added == 1
  assert buffer.snapshot()["false_negative_toe_hard_positive"].tolist() == [
    True,
    False,
  ]


def test_online_metric_improvement_uses_min_delta() -> None:
  assert _metric_improved(
    metric_value=0.51,
    best_value=0.50,
    higher_better=True,
    min_delta=0.001,
  )
  assert not _metric_improved(
    metric_value=0.5005,
    best_value=0.50,
    higher_better=True,
    min_delta=0.001,
  )
  assert _metric_improved(
    metric_value=0.49,
    best_value=0.50,
    higher_better=False,
    min_delta=0.001,
  )


def test_toe_only_finetune_keeps_non_toe_outputs_frozen() -> None:
  model = FootEventDetectorGRU(
    obs_dim=3,
    frame_hidden_dim=8,
    recurrent_hidden_dim=8,
    head_hidden_dim=6,
  )
  _configure_toe_only_finetune(model)
  before = {
    name: parameter.detach().clone() for name, parameter in model.named_parameters()
  }
  optimizer = torch.optim.AdamW(
    [parameter for parameter in model.parameters() if parameter.requires_grad],
    lr=0.1,
    weight_decay=0.0,
  )

  logits = model(torch.randn(5, 4, 3))
  loss = logits[:, 4:6].sum()
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

  for name, parameter in model.named_parameters():
    if name == "head.3.weight":
      assert torch.allclose(parameter[:4], before[name][:4])
      assert not torch.allclose(parameter[4:6], before[name][4:6])
    elif name == "head.3.bias":
      assert torch.allclose(parameter[:4], before[name][:4])
      assert not torch.allclose(parameter[4:6], before[name][4:6])
    else:
      assert torch.allclose(parameter, before[name])


def test_toe_riser_only_model_keeps_standard_output_layout() -> None:
  model = FootEventDetectorGRU(
    obs_dim=3,
    frame_hidden_dim=8,
    recurrent_hidden_dim=8,
    head_hidden_dim=6,
  )
  _configure_toe_riser_only_model(model, dummy_logit=-20.0)
  output_layer = model.head[-1]
  assert isinstance(output_layer, torch.nn.Linear)
  assert output_layer.bias is not None
  before_non_toe_weight = output_layer.weight[:4].detach().clone()
  before_non_toe_bias = output_layer.bias[:4].detach().clone()
  before_toe_weight = output_layer.weight[4:6].detach().clone()
  optimizer = torch.optim.AdamW(
    [parameter for parameter in model.parameters() if parameter.requires_grad],
    lr=0.1,
    weight_decay=0.0,
  )

  obs = torch.randn(5, 4, 3)
  logits = model(obs)
  assert logits.shape == (5, len(FOOT_EVENT_LABEL_NAMES))
  assert torch.allclose(logits[:, :4], torch.full_like(logits[:, :4], -20.0))
  loss = logits[:, 4:6].sum()
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

  logits_after = model(torch.randn(5, 4, 3))
  assert torch.allclose(
    logits_after[:, :4], torch.full_like(logits_after[:, :4], -20.0)
  )
  assert torch.allclose(output_layer.weight[:4], before_non_toe_weight)
  assert torch.allclose(output_layer.bias[:4], before_non_toe_bias)
  assert not torch.allclose(output_layer.weight[4:6], before_toe_weight)


def test_footprint_only_model_forces_toe_outputs_off() -> None:
  model = FootEventDetectorGRU(
    obs_dim=3,
    frame_hidden_dim=8,
    recurrent_hidden_dim=8,
    head_hidden_dim=6,
  )
  _configure_footprint_only_model(model, dummy_logit=-20.0)
  output_layer = model.head[-1]
  assert isinstance(output_layer, torch.nn.Linear)
  assert output_layer.bias is not None
  before_toe_weight = output_layer.weight[4:6].detach().clone()
  before_toe_bias = output_layer.bias[4:6].detach().clone()
  before_footprint_weight = output_layer.weight[:4].detach().clone()
  optimizer = torch.optim.AdamW(
    [parameter for parameter in model.parameters() if parameter.requires_grad],
    lr=0.1,
    weight_decay=0.0,
  )

  obs = torch.randn(5, 4, 3)
  logits = model(obs)
  assert logits.shape == (5, len(FOOT_EVENT_LABEL_NAMES))
  assert torch.allclose(logits[:, 4:6], torch.full_like(logits[:, 4:6], -20.0))
  loss = logits[:, :4].sum()
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

  logits_after = model(torch.randn(5, 4, 3))
  assert torch.allclose(
    logits_after[:, 4:6], torch.full_like(logits_after[:, 4:6], -20.0)
  )
  assert torch.allclose(output_layer.weight[4:6], before_toe_weight)
  assert torch.allclose(output_layer.bias[4:6], before_toe_bias)
  assert not torch.allclose(output_layer.weight[:4], before_footprint_weight)


def test_baseline_guard_requires_toe_gain_without_touchdown_regression() -> None:
  baseline = {
    "touchdown_high_recall_macro_recall": 0.84,
    "touchdown_stair_high_recall_macro_recall": 0.82,
    "touchdown_flat_high_recall_macro_f1": 0.89,
    "toe_riser_high_recall_macro_f1": 0.62,
  }
  candidate = {
    "touchdown_high_recall_macro_recall": 0.835,
    "touchdown_stair_high_recall_macro_recall": 0.81,
    "touchdown_flat_high_recall_macro_f1": 0.885,
    "toe_riser_high_recall_macro_f1": 0.64,
  }

  passed, failures = _baseline_guard_passed(
    candidate,
    baseline,
    touchdown_recall_tolerance=0.01,
    stair_touchdown_recall_tolerance=0.015,
    flat_touchdown_f1_tolerance=0.01,
    toe_metric="toe_riser_high_recall_macro_f1",
    toe_metric_min_improvement=0.01,
  )

  assert passed
  assert failures == ()

  candidate["touchdown_stair_high_recall_macro_recall"] = 0.79
  passed, failures = _baseline_guard_passed(
    candidate,
    baseline,
    touchdown_recall_tolerance=0.01,
    stair_touchdown_recall_tolerance=0.015,
    flat_touchdown_f1_tolerance=0.01,
    toe_metric="toe_riser_high_recall_macro_f1",
    toe_metric_min_improvement=0.01,
  )

  assert not passed
  assert any("touchdown_stair_high_recall_macro_recall" in item for item in failures)


def test_score_metric_is_higher_better() -> None:
  assert metric_is_higher_better("footprint_deploy_score")
  assert metric_is_higher_better("high_recall_footprint_score")
  assert metric_is_higher_better("toe_guarded_footprint_score")
  assert metric_is_higher_better("touchdown_timing_guarded_score")
  assert metric_is_higher_better("toe_riser_high_recall_score")
  assert metric_is_higher_better("touchdown_high_recall_macro_recall")
  assert not metric_is_higher_better("val_loss")


def test_online_deployment_metrics_include_high_recall_score() -> None:
  labels = np.zeros((12, len(FOOT_EVENT_LABEL_NAMES)), dtype=np.float32)
  probabilities = np.zeros_like(labels)
  stair_support = np.zeros((12, 2), dtype=np.bool_)
  footprint_anchor_w = np.zeros((12, 2, 3), dtype=np.float32)

  labels[[2, 8], 0] = 1.0
  labels[[3, 9], 1] = 1.0
  labels[[2, 8], 2] = 1.0
  labels[[3, 9], 3] = 1.0
  labels[5, 4] = 1.0
  labels[6, 5] = 1.0
  stair_support[8, 0] = True
  stair_support[9, 1] = True

  probabilities[:, :] = 0.05
  probabilities[[2, 8], 0] = 0.95
  probabilities[[3, 9], 1] = 0.95
  probabilities[[2, 8], 2] = 0.90
  probabilities[[3, 9], 3] = 0.90
  probabilities[5, 4] = 0.80
  probabilities[6, 5] = 0.80

  metrics = _deployment_event_metrics(
    labels=labels,
    probabilities=probabilities,
    episode_id=np.zeros(labels.shape[0], dtype=np.int64),
    frame_idx=np.arange(labels.shape[0], dtype=np.int64),
    stair_support=stair_support,
    thresholds=np.array([0.3, 0.7], dtype=np.float32),
    footprint_anchor_w=footprint_anchor_w,
    touchdown_contact_threshold=0.7,
    touchdown_contact_release_threshold=0.35,
    touchdown_cooldown_frames=0,
    toe_hit_cooldown_frames=0,
    touchdown_recall_precision_floor=0.85,
    toe_hit_recall_precision_floor=0.35,
    selection_min_stair_touchdown_events=1,
    event_tolerance_frames=0,
  )

  assert metrics["high_recall_footprint_score"] > 0.0
  assert metrics["toe_guarded_footprint_score"] > 0.0
  assert metrics["touchdown_high_recall_score"] > 0.0
  assert metrics["toe_riser_high_recall_score"] == 1.0
  assert metrics["touchdown_stair_high_recall_macro_recall"] == 1.0
  assert metrics["toe_riser_high_recall_macro_recall"] == 1.0
  assert metrics["high_recall_precision_guard"] == 1.0
  assert metrics["touchdown_timing_abs_dt_frames_p90"] == 0.0
  assert metrics["touchdown_footprint_xy_error_m_p90"] == 0.0
  assert metrics["touchdown_timing_guarded_score"] > 0.0
  assert metrics["timing_guarded_footprint_score"] > 0.0


def test_run_train_writes_metrics_and_best_checkpoint(tmp_path) -> None:
  arrays = _make_detector_arrays(num_samples=48, history_len=4, obs_dim=91)
  dataset_file = tmp_path / "samples.npz"
  np.savez_compressed(
    dataset_file,
    obs_history=arrays.obs_history,
    obs_valid_mask=arrays.obs_valid_mask,
    event_label=arrays.event_label,
    episode_id=arrays.episode_id,
    env_id=arrays.env_id,
    frame_idx=arrays.frame_idx,
    seed=arrays.seed,
  )
  output_dir = tmp_path / "train"

  payload = run_train(
    TrainFootEventDetectorConfig(
      dataset_file=str(dataset_file),
      output_dir=str(output_dir),
      device="cpu",
      history_len=4,
      frame_hidden_dim=16,
      recurrent_hidden_dim=16,
      head_hidden_dim=8,
      batch_size=8,
      epochs=1,
      export_onnx=False,
      progress=False,
    )
  )

  assert payload["best_epoch"] == 1
  assert (output_dir / "best.pt").exists()
  assert (output_dir / "metrics.json").exists()
  assert (output_dir / "val_predictions.csv").exists()
  saved = json.loads((output_dir / "metrics.json").read_text())
  assert saved["model"]["obs_dim"] == 91
  assert saved["model"]["output_dim"] == len(FOOT_EVENT_LABEL_NAMES)
