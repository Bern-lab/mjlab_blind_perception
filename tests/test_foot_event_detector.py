from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from scripts.velocity_eval.export_foot_event_detector_dataset import (
  FOOT_EVENT_LABEL_NAMES,
  FootEventDetectorDatasetBuilder,
  foot_event_labels_from_env,
)
from scripts.velocity_eval.train_foot_event_detector import (
  FootEventArrays,
  FootEventDetectorGRU,
  FootEventTorchDataset,
  TrainFootEventDetectorConfig,
  build_episode_split,
  compute_metrics,
  deployment_event_stats_for_label,
  make_pos_weight,
  run_train,
  sweep_deployment_event_thresholds,
)
from scripts.velocity_eval.train_foot_event_detector_online import (
  OnlineFootEventReplayBuffer,
  _metric_improved,
)

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
    touchdown_positive_fraction=0.5,
    stair_hard_negative_fraction=0.0,
    false_positive_hard_negative_fraction=0.0,
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
