from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import torch
from scripts.velocity_eval.train_stair_probe import (
  NUM_DEPTH_BIN_CLASSES,
  NUM_LEVEL_DELTA_CLASSES,
  NUM_RELATIVE_LEVEL_CLASSES,
  ProbeLossWeights,
  StairProbeArrays,
  StairProbeGRU,
  StairProbeTorchDataset,
  TrainStairProbeConfig,
  TwoBranchStairProbeGRU,
  balanced_class_weights,
  build_sequence_split,
  compute_baseline_metrics,
  compute_metrics,
  compute_probe_loss,
  filter_arrays_for_objective,
  make_loss_weights,
  resolve_selection_metric,
)


def _make_arrays(num_samples: int = 32) -> StairProbeArrays:
  obs = np.arange(num_samples * 6 * 91, dtype=np.float32).reshape(num_samples, 6, 91)
  sample_type = np.arange(num_samples, dtype=np.int64) % 8
  sequence_id = np.full(num_samples, -1, dtype=np.int64)
  stair_ids = np.nonzero(sample_type != 7)[0]
  sequence_id[stair_ids] = np.arange(stair_ids.shape[0], dtype=np.int64) // 2
  depth_valid = sample_type != 7
  depth_bin = np.where(depth_valid, np.arange(num_samples) % 8, -1)
  true_depth = np.where(depth_valid, 0.25 + 0.01 * depth_bin.clip(0, 7), 0.0)
  return StairProbeArrays(
    obs_history=obs,
    obs_valid_mask=np.ones((num_samples, 6), dtype=np.bool_),
    level_delta_label=(np.arange(num_samples) % 4).astype(np.int64),
    relative_level_label=(np.arange(num_samples) % 8).astype(np.int64),
    true_tread_depth=true_depth.astype(np.float32),
    depth_bin_label=depth_bin.astype(np.int64),
    depth_valid_label=depth_valid.astype(np.bool_),
    stair_active_label=depth_valid.astype(np.bool_),
    sample_type=sample_type,
    sequence_id=sequence_id,
    env_id=np.arange(num_samples, dtype=np.int64),
    frame_idx=np.arange(num_samples, dtype=np.int64) + 10,
    seed=np.full(num_samples, 42, dtype=np.int64),
  )


def _with_footprint(arrays: StairProbeArrays, *, dim: int = 18) -> StairProbeArrays:
  num_samples = arrays.obs_history.shape[0]
  history_len = 8
  footprint = np.arange(
    num_samples * history_len * dim,
    dtype=np.float32,
  ).reshape(num_samples, history_len, dim)
  return replace(
    arrays,
    privileged_footprint_history=footprint,
    privileged_footprint_valid_mask=np.ones(
      (num_samples, history_len),
      dtype=np.bool_,
    ),
  )


def _with_sparse_events(arrays: StairProbeArrays, *, dim: int = 20) -> StairProbeArrays:
  num_samples = arrays.obs_history.shape[0]
  history_len = 5
  sparse_events = np.arange(
    num_samples * history_len * dim,
    dtype=np.float32,
  ).reshape(num_samples, history_len, dim)
  return replace(
    arrays,
    sparse_foot_event_memory=sparse_events,
    sparse_foot_event_valid_mask=np.ones(
      (num_samples, history_len),
      dtype=np.bool_,
    ),
  )


def _with_safe_labels(arrays: StairProbeArrays) -> StairProbeArrays:
  num_samples = arrays.obs_history.shape[0]
  safe_valid = arrays.stair_active_label.astype(np.bool_)
  center = np.where(safe_valid, 0.30 + 0.001 * np.arange(num_samples), 0.0)
  return replace(
    arrays,
    true_riser_height=np.where(safe_valid, 0.18, 0.0).astype(np.float32),
    safe_landing_center=center.astype(np.float32),
    minimum_safe_stride=(center - 0.03).astype(np.float32),
    maximum_safe_stride=(center + 0.03).astype(np.float32),
    safe_stride_valid_label=safe_valid,
    landing_touchdown_label=(np.arange(num_samples) % 3 == 0) & safe_valid,
    landing_quality_label=np.where(safe_valid, 0.75, 0.0).astype(np.float32),
    collision_risk_label=np.where(
      (np.arange(num_samples) % 5 == 0) & safe_valid,
      1.0,
      0.0,
    ).astype(np.float32),
  )


def test_gru_probe_forward_shapes_and_latent_dim() -> None:
  model = StairProbeGRU(
    obs_dim=91,
    frame_hidden_dim=16,
    recurrent_hidden_dim=16,
    probe_latent_dim=16,
    head_hidden_dim=8,
  )

  outputs = model(torch.zeros(5, 6, 91))

  assert outputs["probe_latent"].shape == (5, 16)
  assert outputs["stair_active_logit"].shape == (5, 1)
  assert outputs["level_delta_logits"].shape == (5, NUM_LEVEL_DELTA_CLASSES)
  assert outputs["relative_level_logits"].shape == (5, NUM_RELATIVE_LEVEL_CLASSES)
  assert outputs["depth_bin_logits"].shape == (5, NUM_DEPTH_BIN_CLASSES)
  assert outputs["depth_reg"].shape == (5, 1)
  assert outputs["touchdown_logit"].shape == (5, 1)
  assert outputs["landing_quality"].shape == (5, 1)
  assert outputs["collision_risk"].shape == (5, 1)
  assert outputs["safe_stride"].shape == (5, 3)


def test_two_branch_probe_forward_shapes_and_latent_dim() -> None:
  model = TwoBranchStairProbeGRU(
    latent_obs_dim=91,
    footprint_obs_dim=3,
    frame_hidden_dim=16,
    recurrent_hidden_dim=16,
    footprint_frame_hidden_dim=8,
    footprint_recurrent_hidden_dim=8,
    fusion_hidden_dim=12,
    probe_latent_dim=16,
    head_hidden_dim=8,
  )

  outputs = model(torch.zeros(5, 4, 91), torch.zeros(5, 7, 3))

  assert outputs["probe_latent"].shape == (5, 16)
  assert outputs["stair_active_logit"].shape == (5, 1)
  assert outputs["level_delta_logits"].shape == (5, NUM_LEVEL_DELTA_CLASSES)
  assert outputs["relative_level_logits"].shape == (5, NUM_RELATIVE_LEVEL_CLASSES)
  assert outputs["depth_bin_logits"].shape == (5, NUM_DEPTH_BIN_CLASSES)
  assert outputs["depth_reg"].shape == (5, 1)
  assert outputs["touchdown_logit"].shape == (5, 1)
  assert outputs["landing_quality"].shape == (5, 1)
  assert outputs["collision_risk"].shape == (5, 1)
  assert outputs["safe_stride"].shape == (5, 3)


def test_dataset_uses_last_history_frames() -> None:
  arrays = _make_arrays(num_samples=4)
  dataset = StairProbeTorchDataset(arrays, np.array([0]), history_len=3)

  item = dataset[0]

  expected = torch.as_tensor(arrays.obs_history[0, -3:, :], dtype=torch.float32)
  assert torch.equal(item["obs_history"], expected)


def test_dataset_latent_plus_footprint_zero_pads_shorter_latent_history() -> None:
  arrays = _with_footprint(_make_arrays(num_samples=4), dim=3)
  dataset = StairProbeTorchDataset(
    arrays,
    np.array([0]),
    history_len=3,
    footprint_history_len=5,
    input_mode="latent_plus_footprint",
  )

  item = dataset[0]

  assert item["obs_history"].shape == (5, 94)
  assert dataset.input_dim == 94
  assert dataset.input_history_len == 5
  assert torch.equal(item["obs_history"][:2, :91], torch.zeros(2, 91))
  expected_latent = torch.as_tensor(arrays.obs_history[0, -3:, :], dtype=torch.float32)
  assert torch.equal(item["obs_history"][-3:, :91], expected_latent)
  assert arrays.privileged_footprint_history is not None
  expected_footprint = torch.as_tensor(
    arrays.privileged_footprint_history[0, -5:, :],
    dtype=torch.float32,
  )
  assert torch.equal(item["obs_history"][:, 91:], expected_footprint)


def test_dataset_footprint_only_uses_footprint_history() -> None:
  arrays = _with_footprint(_make_arrays(num_samples=4), dim=5)
  dataset = StairProbeTorchDataset(
    arrays,
    np.array([0]),
    history_len=3,
    footprint_history_len=4,
    input_mode="footprint_only",
  )

  item = dataset[0]

  assert item["obs_history"].shape == (4, 5)
  assert arrays.privileged_footprint_history is not None
  expected = torch.as_tensor(
    arrays.privileged_footprint_history[0, -4:, :],
    dtype=torch.float32,
  )
  assert torch.equal(item["obs_history"], expected)


def test_dataset_sparse_event_only_uses_event_memory() -> None:
  arrays = _with_sparse_events(_make_arrays(num_samples=4), dim=6)
  dataset = StairProbeTorchDataset(
    arrays,
    np.array([0]),
    history_len=3,
    sparse_event_memory_len=4,
    input_mode="sparse_event_only",
  )

  item = dataset[0]

  assert item["obs_history"].shape == (4, 6)
  assert arrays.sparse_foot_event_memory is not None
  expected = torch.as_tensor(
    arrays.sparse_foot_event_memory[0, :4, :],
    dtype=torch.float32,
  )
  assert torch.equal(item["obs_history"], expected)


def test_dataset_latent_plus_sparse_event_zero_pads_shorter_event_memory() -> None:
  arrays = _with_sparse_events(_make_arrays(num_samples=4), dim=6)
  dataset = StairProbeTorchDataset(
    arrays,
    np.array([0]),
    history_len=6,
    sparse_event_memory_len=4,
    input_mode="latent_plus_sparse_event",
  )

  item = dataset[0]

  assert item["obs_history"].shape == (6, 97)
  assert dataset.input_dim == 97
  assert dataset.input_history_len == 6
  expected_latent = torch.as_tensor(arrays.obs_history[0, -6:, :], dtype=torch.float32)
  assert torch.equal(item["obs_history"][:, :91], expected_latent)
  assert torch.equal(item["obs_history"][4:, 91:], torch.zeros(2, 6))
  assert arrays.sparse_foot_event_memory is not None
  expected_events = torch.as_tensor(
    arrays.sparse_foot_event_memory[0, :4, :],
    dtype=torch.float32,
  )
  assert torch.equal(item["obs_history"][:4, 91:], expected_events)


def test_dataset_two_branch_returns_separate_histories() -> None:
  arrays = _with_footprint(_make_arrays(num_samples=4), dim=3)
  dataset = StairProbeTorchDataset(
    arrays,
    np.array([0]),
    history_len=3,
    footprint_history_len=5,
    input_mode="two_branch_fusion",
  )

  item = dataset[0]

  assert item["obs_history"].shape == (3, 91)
  assert item["privileged_footprint_history"].shape == (5, 3)
  assert dataset.input_dim == 94
  assert dataset.input_history_len == 5
  expected_latent = torch.as_tensor(arrays.obs_history[0, -3:, :], dtype=torch.float32)
  assert torch.equal(item["obs_history"], expected_latent)
  assert arrays.privileged_footprint_history is not None
  expected_footprint = torch.as_tensor(
    arrays.privileged_footprint_history[0, -5:, :],
    dtype=torch.float32,
  )
  assert torch.equal(item["privileged_footprint_history"], expected_footprint)


def test_dataset_two_branch_sparse_event_returns_separate_histories() -> None:
  arrays = _with_sparse_events(_make_arrays(num_samples=4), dim=6)
  dataset = StairProbeTorchDataset(
    arrays,
    np.array([0]),
    history_len=3,
    sparse_event_memory_len=4,
    input_mode="two_branch_sparse_event",
  )

  item = dataset[0]

  assert item["obs_history"].shape == (3, 91)
  assert item["sparse_foot_event_memory"].shape == (4, 6)
  assert dataset.input_dim == 97
  assert dataset.input_history_len == 4
  expected_latent = torch.as_tensor(arrays.obs_history[0, -3:, :], dtype=torch.float32)
  assert torch.equal(item["obs_history"], expected_latent)
  assert arrays.sparse_foot_event_memory is not None
  expected_events = torch.as_tensor(
    arrays.sparse_foot_event_memory[0, :4, :],
    dtype=torch.float32,
  )
  assert torch.equal(item["sparse_foot_event_memory"], expected_events)


def test_sequence_split_has_no_stair_sequence_leakage() -> None:
  arrays = _make_arrays(num_samples=64)

  split = build_sequence_split(arrays.sequence_id, val_fraction=0.25, seed=123)

  train_sequences = set(arrays.sequence_id[split.train_indices].tolist()) - {-1}
  val_sequences = set(arrays.sequence_id[split.val_indices].tolist()) - {-1}
  assert train_sequences.isdisjoint(val_sequences)
  assert len(split.train_indices) > 0
  assert len(split.val_indices) > 0


def test_balanced_class_weights_handle_empty_classes() -> None:
  weights = balanced_class_weights(
    np.array([0, 0, 0], dtype=np.int64),
    num_classes=4,
  )

  assert torch.isfinite(weights).all()
  assert weights.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_depth_loss_ignores_invalid_depth_labels() -> None:
  cfg = TrainStairProbeConfig(depth_reg_loss_coef=1.0, depth_bin_loss_coef=1.0)
  outputs = {
    "probe_latent": torch.zeros(2, 16, requires_grad=True),
    "stair_active_logit": torch.zeros(2, 1, requires_grad=True),
    "level_delta_logits": torch.zeros(2, 4, requires_grad=True),
    "relative_level_logits": torch.zeros(2, 8, requires_grad=True),
    "depth_bin_logits": torch.zeros(2, 8, requires_grad=True),
    "depth_reg": torch.zeros(2, 1, requires_grad=True),
  }
  batch = {
    "stair_active": torch.tensor([1.0, 0.0]),
    "level_delta": torch.tensor([1, 0]),
    "relative_level": torch.tensor([1, 0]),
    "depth_bin": torch.tensor([3, 7]),
    "depth_valid": torch.tensor([True, False]),
    "depth_norm": torch.tensor([0.4, 99.0]),
  }
  weights = ProbeLossWeights(
    active_pos_weight=torch.tensor(1.0),
    level_delta=torch.ones(4),
    relative_level=torch.ones(8),
    depth_bin=torch.ones(8),
  )

  loss_a, _parts_a = compute_probe_loss(outputs, batch, weights, cfg)
  batch["depth_bin"] = torch.tensor([3, 0])
  batch["depth_norm"] = torch.tensor([0.4, -99.0])
  loss_b, _parts_b = compute_probe_loss(outputs, batch, weights, cfg)

  assert loss_a.item() == loss_b.item()


def test_depth_only_objective_filters_to_depth_valid_samples() -> None:
  arrays = _make_arrays(num_samples=24)

  depth_arrays = filter_arrays_for_objective(arrays, "depth_only")

  assert depth_arrays.obs_history.shape[0] == int(arrays.depth_valid_label.sum())
  assert depth_arrays.depth_valid_label.all()
  assert np.all(depth_arrays.sequence_id >= 0)


def test_safe_landing_objective_filters_to_stair_samples() -> None:
  arrays = _with_safe_labels(_make_arrays(num_samples=24))

  safe_arrays = filter_arrays_for_objective(arrays, "safe_landing")

  assert safe_arrays.obs_history.shape[0] == int(arrays.stair_active_label.sum())
  assert safe_arrays.safe_stride_valid_label is not None
  assert safe_arrays.safe_stride_valid_label.all()


def test_depth_only_loss_ignores_non_depth_tasks() -> None:
  cfg = TrainStairProbeConfig(objective="depth_only")
  outputs = {
    "probe_latent": torch.zeros(2, 16, requires_grad=True),
    "stair_active_logit": torch.randn(2, 1, requires_grad=True),
    "level_delta_logits": torch.randn(2, 4, requires_grad=True),
    "relative_level_logits": torch.randn(2, 8, requires_grad=True),
    "depth_bin_logits": torch.zeros(2, 8, requires_grad=True),
    "depth_reg": torch.zeros(2, 1, requires_grad=True),
  }
  batch = {
    "stair_active": torch.tensor([1.0, 0.0]),
    "level_delta": torch.tensor([1, 2]),
    "relative_level": torch.tensor([1, 7]),
    "depth_bin": torch.tensor([3, 4]),
    "depth_valid": torch.tensor([True, True]),
    "depth_norm": torch.tensor([0.4, 0.5]),
  }
  weights = ProbeLossWeights(
    active_pos_weight=torch.tensor(100.0),
    level_delta=torch.tensor([10.0, 20.0, 30.0, 40.0]),
    relative_level=torch.arange(1, 9, dtype=torch.float32),
    depth_bin=torch.ones(8),
  )

  loss_a, parts_a = compute_probe_loss(outputs, batch, weights, cfg)
  batch["stair_active"] = torch.tensor([0.0, 1.0])
  batch["level_delta"] = torch.tensor([0, 0])
  batch["relative_level"] = torch.tensor([0, 0])
  loss_b, parts_b = compute_probe_loss(outputs, batch, weights, cfg)

  assert loss_a.item() == loss_b.item()
  assert parts_a["active_loss"] == 0.0
  assert parts_a["level_delta_loss"] == 0.0
  assert parts_a["relative_level_loss"] == 0.0
  assert parts_b["active_loss"] == 0.0


def test_depth_only_auto_selection_metric_targets_depth_bin_macro_f1() -> None:
  cfg = TrainStairProbeConfig(objective="depth_only", selection_metric="auto")

  assert resolve_selection_metric(cfg) == "depth_bin_macro_f1"


def test_safe_landing_auto_selection_metric_targets_val_loss() -> None:
  cfg = TrainStairProbeConfig(objective="safe_landing", selection_metric="auto")

  assert resolve_selection_metric(cfg) == "val_loss"


def test_safe_landing_loss_ignores_depth_tasks() -> None:
  cfg = TrainStairProbeConfig(objective="safe_landing")
  outputs = {
    "probe_latent": torch.zeros(2, 16, requires_grad=True),
    "stair_active_logit": torch.randn(2, 1, requires_grad=True),
    "level_delta_logits": torch.randn(2, 4, requires_grad=True),
    "relative_level_logits": torch.randn(2, 8, requires_grad=True),
    "depth_bin_logits": torch.randn(2, 8, requires_grad=True),
    "depth_reg": torch.randn(2, 1, requires_grad=True),
    "touchdown_logit": torch.zeros(2, 1, requires_grad=True),
    "landing_quality": torch.zeros(2, 1, requires_grad=True),
    "collision_risk": torch.zeros(2, 1, requires_grad=True),
    "safe_stride": torch.zeros(2, 3, requires_grad=True),
  }
  batch = {
    "stair_active": torch.tensor([1.0, 0.0]),
    "level_delta": torch.tensor([1, 2]),
    "relative_level": torch.tensor([1, 7]),
    "depth_bin": torch.tensor([3, 4]),
    "depth_valid": torch.tensor([True, True]),
    "depth_norm": torch.tensor([0.4, 0.5]),
    "landing_touchdown": torch.tensor([1.0, 0.0]),
    "landing_quality": torch.tensor([0.8, 0.0]),
    "collision_risk": torch.tensor([0.0, 1.0]),
    "safe_stride_valid": torch.tensor([True, False]),
    "minimum_safe_stride": torch.tensor([0.25, 0.0]),
    "maximum_safe_stride": torch.tensor([0.35, 0.0]),
    "safe_landing_center": torch.tensor([0.30, 0.0]),
  }
  weights = ProbeLossWeights(
    active_pos_weight=torch.tensor(100.0),
    level_delta=torch.tensor([10.0, 20.0, 30.0, 40.0]),
    relative_level=torch.arange(1, 9, dtype=torch.float32),
    depth_bin=torch.ones(8),
  )

  loss_a, parts_a = compute_probe_loss(outputs, batch, weights, cfg)
  batch["depth_bin"] = torch.tensor([0, 0])
  batch["depth_norm"] = torch.tensor([99.0, -99.0])
  batch["level_delta"] = torch.tensor([0, 0])
  loss_b, parts_b = compute_probe_loss(outputs, batch, weights, cfg)

  assert loss_a.item() == loss_b.item()
  assert parts_a["active_loss"] == 0.0
  assert parts_a["depth_bin_loss"] == 0.0
  assert parts_a["depth_reg_loss"] == 0.0
  assert parts_b["level_delta_loss"] == 0.0
  assert parts_a["touchdown_loss"] > 0.0
  assert parts_a["safe_stride_loss"] > 0.0


def test_metrics_report_macro_scores_without_nan() -> None:
  labels = {
    "stair_active": np.array([1, 1, 0, 0]),
    "level_delta": np.array([0, 1, 0, 0]),
    "relative_level": np.array([0, 1, 1, 0]),
    "depth_bin": np.array([0, 1, 0, 0]),
    "depth_valid": np.array([1, 1, 0, 0]),
    "true_tread_depth": np.array([0.25, 0.30, 0.0, 0.0], dtype=np.float32),
  }
  predictions = {
    "stair_active_prob": np.array([0.9, 0.8, 0.1, 0.7]),
    "level_delta_pred": np.array([0, 0, 0, 0]),
    "relative_level_pred": np.array([0, 1, 0, 0]),
    "depth_bin_pred": np.array([0, 2, 0, 0]),
    "depth_reg": np.array([0.0, 0.5, 0.0, 0.0], dtype=np.float32),
  }

  metrics = compute_metrics(labels, predictions, val_loss=1.25)

  assert metrics["val_loss"] == 1.25
  assert all(math.isfinite(value) for value in metrics.values())
  assert "level_delta_macro_f1" in metrics
  assert "depth_bin_macro_f1" in metrics
  assert "depth_3group_macro_f1" in metrics


def test_metrics_report_safe_landing_scores_without_nan() -> None:
  labels = {
    "stair_active": np.array([1, 1, 0, 0]),
    "level_delta": np.array([0, 1, 0, 0]),
    "relative_level": np.array([0, 1, 1, 0]),
    "depth_bin": np.array([0, 1, 0, 0]),
    "depth_valid": np.array([1, 1, 0, 0]),
    "true_tread_depth": np.array([0.25, 0.30, 0.0, 0.0], dtype=np.float32),
    "safe_stride_valid": np.array([1, 1, 0, 0], dtype=np.bool_),
    "minimum_safe_stride": np.array([0.20, 0.25, 0.0, 0.0], dtype=np.float32),
    "maximum_safe_stride": np.array([0.30, 0.35, 0.0, 0.0], dtype=np.float32),
    "safe_landing_center": np.array([0.25, 0.30, 0.0, 0.0], dtype=np.float32),
    "landing_touchdown": np.array([1, 0, 0, 0], dtype=np.float32),
    "landing_quality": np.array([0.8, 0.0, 0.0, 0.0], dtype=np.float32),
    "collision_risk": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
  }
  predictions = {
    "stair_active_prob": np.array([0.9, 0.8, 0.1, 0.7]),
    "level_delta_pred": np.array([0, 0, 0, 0]),
    "relative_level_pred": np.array([0, 1, 0, 0]),
    "depth_bin_pred": np.array([0, 2, 0, 0]),
    "depth_reg": np.array([0.0, 0.5, 0.0, 0.0], dtype=np.float32),
    "touchdown_prob": np.array([0.9, 0.4, 0.2, 0.1], dtype=np.float32),
    "landing_quality_pred": np.array([0.7, 0.1, 0.0, 0.0], dtype=np.float32),
    "collision_risk_pred": np.array([0.2, 0.8, 0.1, 0.0], dtype=np.float32),
    "safe_stride_pred": np.array(
      [
        [0.22, 0.32, 0.27],
        [0.26, 0.36, 0.31],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
      ],
      dtype=np.float32,
    ),
  }

  metrics = compute_metrics(labels, predictions, val_loss=1.25)

  assert all(math.isfinite(value) for value in metrics.values())
  assert metrics["touchdown_f1"] == 1.0
  assert math.isclose(metrics["safe_stride_center_mae_m"], 0.015, rel_tol=1.0e-5)
  assert math.isclose(metrics["landing_quality_mae"], 0.1, rel_tol=1.0e-5)


def test_baselines_include_depth_constant_mae() -> None:
  labels = {
    "stair_active": np.array([1, 1, 0, 0]),
    "level_delta": np.array([0, 1, 0, 0]),
    "relative_level": np.array([0, 1, 1, 0]),
    "depth_bin": np.array([0, 1, 0, 0]),
    "depth_valid": np.array([1, 1, 0, 0]),
    "true_tread_depth": np.array([0.25, 0.35, 0.0, 0.0], dtype=np.float32),
  }

  baselines = compute_baseline_metrics(labels)

  assert baselines["depth_valid_count"] == 2.0
  assert baselines["depth_median_baseline_mae_m"] > 0.0


def test_one_training_step_backward_runs() -> None:
  arrays = _make_arrays(num_samples=16)
  dataset = StairProbeTorchDataset(arrays, np.arange(16), history_len=4)
  batch = {
    key: value
    for key, value in next(iter(torch.utils.data.DataLoader(dataset, 8))).items()
  }
  model = StairProbeGRU(
    obs_dim=91,
    frame_hidden_dim=16,
    recurrent_hidden_dim=16,
    probe_latent_dim=16,
    head_hidden_dim=8,
  )
  weights = make_loss_weights(arrays, np.arange(16), "cpu")
  cfg = TrainStairProbeConfig(
    frame_hidden_dim=16,
    recurrent_hidden_dim=16,
    head_hidden_dim=8,
    batch_size=8,
  )

  outputs = model(batch["obs_history"])
  loss, _parts = compute_probe_loss(outputs, batch, weights, cfg)
  loss.backward()

  grads = [
    parameter.grad
    for parameter in model.parameters()
    if parameter.requires_grad and parameter.grad is not None
  ]
  assert grads
  assert all(torch.isfinite(grad).all() for grad in grads)


def test_one_two_branch_training_step_backward_runs() -> None:
  arrays = _with_footprint(_make_arrays(num_samples=16), dim=3)
  dataset = StairProbeTorchDataset(
    arrays,
    np.arange(16),
    history_len=4,
    footprint_history_len=5,
    input_mode="two_branch_fusion",
  )
  batch = {
    key: value
    for key, value in next(iter(torch.utils.data.DataLoader(dataset, 8))).items()
  }
  model = TwoBranchStairProbeGRU(
    latent_obs_dim=91,
    footprint_obs_dim=3,
    frame_hidden_dim=16,
    recurrent_hidden_dim=16,
    footprint_frame_hidden_dim=8,
    footprint_recurrent_hidden_dim=8,
    fusion_hidden_dim=12,
    probe_latent_dim=16,
    head_hidden_dim=8,
  )
  weights = make_loss_weights(arrays, np.arange(16), "cpu")
  cfg = TrainStairProbeConfig(
    input_mode="two_branch_fusion",
    frame_hidden_dim=16,
    recurrent_hidden_dim=16,
    footprint_frame_hidden_dim=8,
    footprint_recurrent_hidden_dim=8,
    fusion_hidden_dim=12,
    head_hidden_dim=8,
    batch_size=8,
  )

  outputs = model(batch["obs_history"], batch["privileged_footprint_history"])
  loss, _parts = compute_probe_loss(outputs, batch, weights, cfg)
  loss.backward()

  grads = [
    parameter.grad
    for parameter in model.parameters()
    if parameter.requires_grad and parameter.grad is not None
  ]
  assert grads
  assert all(torch.isfinite(grad).all() for grad in grads)
