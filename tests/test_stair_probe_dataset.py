from __future__ import annotations

import numpy as np
import torch
from scripts.velocity_eval.export_stair_probe_dataset import (
  LEVEL_DELTA_DOWN_ONE,
  LEVEL_DELTA_NO_TRANSITION,
  LEVEL_DELTA_SKIP_OR_UNCERTAIN,
  LEVEL_DELTA_UP_ONE,
  SAMPLE_TYPE_TO_ID,
  StairProbeDatasetBuilder,
  StairProbeHistoryBuffer,
  StairProbeLabelBatch,
  StairProbeLevelState,
  forbidden_input_feature_names,
  input_obs_dim,
  privileged_footprint_features_from_tensors,
  privileged_footprint_obs_dim,
)


def _update_single_env(
  state: StairProbeLevelState,
  *,
  layer: int,
  support_fraction: float = 1.0,
  active: bool = True,
  sequence_id: int = 7,
) -> StairProbeLabelBatch:
  return state.update(
    stair_active=torch.tensor([active]),
    sequence_id=torch.tensor([sequence_id]),
    support_layer=torch.tensor([[layer, -1]]),
    stair_support=torch.tensor([[layer >= 0, False]]),
    support_fraction=torch.tensor([[support_fraction, 0.0]]),
    contact_duration=torch.tensor([[0.10, 0.0]]),
    true_tread_depth=torch.tensor([0.30 if active else 0.0]),
    shape_valid=torch.tensor([active]),
  )


def test_history_buffer_preserves_order_and_resets_sequence() -> None:
  buffer = StairProbeHistoryBuffer(
    num_envs=1,
    history_len=4,
    obs_dim=1,
    device="cpu",
  )

  for frame in range(10):
    buffer.push(torch.tensor([[float(frame)]]))

  assert buffer.history[0, :, 0].tolist() == [6.0, 7.0, 8.0, 9.0]
  assert buffer.valid_mask[0].tolist() == [True, True, True, True]

  buffer.push(torch.tensor([[10.0]]), reset_mask=torch.tensor([True]))

  assert buffer.history[0, :, 0].tolist() == [0.0, 0.0, 0.0, 10.0]
  assert buffer.valid_mask[0].tolist() == [False, False, False, True]


def test_level_state_machine_tracks_relative_level() -> None:
  state = StairProbeLevelState(num_envs=1, stable_contact_time=0.0)

  labels = _update_single_env(state, layer=0)
  assert labels.level_delta_label.item() == LEVEL_DELTA_NO_TRANSITION
  assert labels.relative_level_label.item() == 0

  labels = _update_single_env(state, layer=1)
  assert labels.level_delta_label.item() == LEVEL_DELTA_UP_ONE
  assert labels.relative_level_label.item() == 1

  labels = _update_single_env(state, layer=1)
  assert labels.level_delta_label.item() == LEVEL_DELTA_NO_TRANSITION
  assert labels.relative_level_label.item() == 1

  labels = _update_single_env(state, layer=0)
  assert labels.level_delta_label.item() == LEVEL_DELTA_DOWN_ONE
  assert labels.relative_level_label.item() == 0

  labels = _update_single_env(state, layer=3)
  assert labels.level_delta_label.item() == LEVEL_DELTA_SKIP_OR_UNCERTAIN
  assert labels.relative_level_label.item() == 3
  assert labels.num_confirmed_layers.item() == 2


def test_partial_support_does_not_confirm_level() -> None:
  state = StairProbeLevelState(
    num_envs=1,
    stable_support_fraction=0.75,
    partial_support_fraction=0.25,
    stable_contact_time=0.0,
  )

  labels = _update_single_env(state, layer=1, support_fraction=0.50)
  assert labels.level_delta_label.item() == LEVEL_DELTA_NO_TRANSITION
  assert labels.relative_level_label.item() == 0
  assert labels.partial_uncertain_label.item() is True

  labels = _update_single_env(state, layer=1, support_fraction=0.80)
  assert labels.level_delta_label.item() == LEVEL_DELTA_UP_ONE
  assert labels.relative_level_label.item() == 1


def test_level_state_assigns_dataset_local_sequence_ids() -> None:
  state = StairProbeLevelState(num_envs=1, stable_contact_time=0.0)

  labels = _update_single_env(state, layer=0, sequence_id=11)
  assert labels.sequence_id.item() == 0

  labels = _update_single_env(state, layer=1, sequence_id=11)
  assert labels.sequence_id.item() == 0

  labels = _update_single_env(state, layer=-1, active=False)
  assert labels.sequence_id.item() == -1

  labels = _update_single_env(state, layer=0, sequence_id=11)
  assert labels.sequence_id.item() == 1


def test_flat_negative_sample_has_zero_level_and_invalid_depth() -> None:
  obs_dim = input_obs_dim()
  builder = StairProbeDatasetBuilder(
    num_envs=1,
    history_len=2,
    obs_dim=obs_dim,
    max_samples=8,
    seed=42,
    device="cpu",
    flat_sample_period=1,
  )
  builder.push_observations(torch.ones(1, obs_dim), torch.tensor([True]))
  state = StairProbeLevelState(num_envs=1, stable_contact_time=0.0)
  labels = _update_single_env(state, layer=-1, active=False)

  builder.collect(labels, frame_idx=1)
  arrays = builder.as_arrays()

  assert arrays["obs_history"].shape == (1, 2, obs_dim)
  assert arrays["sample_type"].tolist() == [SAMPLE_TYPE_TO_ID["flat_negative"]]
  assert arrays["level_delta_label"].tolist() == [LEVEL_DELTA_NO_TRANSITION]
  assert arrays["relative_level_label"].tolist() == [0]
  assert arrays["depth_valid_label"].tolist() == [False]
  assert arrays["depth_bin_label"].tolist() == [-1]


def test_transition_sample_takes_priority_over_partial_uncertain() -> None:
  obs_dim = input_obs_dim()
  builder = StairProbeDatasetBuilder(
    num_envs=1,
    history_len=2,
    obs_dim=obs_dim,
    max_samples=8,
    seed=42,
    device="cpu",
    partial_sample_period=1,
  )
  builder.push_observations(torch.ones(1, obs_dim), torch.tensor([True]))
  labels = StairProbeLabelBatch(
    level_delta_label=torch.tensor([LEVEL_DELTA_UP_ONE]),
    relative_level_label=torch.tensor([1]),
    level_transition_label=torch.tensor([True]),
    true_tread_depth=torch.tensor([0.30]),
    depth_bin_label=torch.tensor([3]),
    depth_3bin_label=torch.tensor([1]),
    depth_valid_label=torch.tensor([True]),
    num_confirmed_layers=torch.tensor([1]),
    stair_active_label=torch.tensor([True]),
    full_landing_label=torch.tensor([False]),
    stable_landing_label=torch.tensor([False]),
    partial_uncertain_label=torch.tensor([True]),
    sequence_id=torch.tensor([0]),
  )

  builder.collect(labels, frame_idx=1)
  arrays = builder.as_arrays()

  assert arrays["sample_type"].tolist() == [SAMPLE_TYPE_TO_ID["transition_up"]]
  assert arrays["level_delta_label"].tolist() == [LEVEL_DELTA_UP_ONE]


def test_export_schema_is_91_dim_and_has_no_forbidden_input_features(tmp_path) -> None:
  obs_dim = input_obs_dim()
  assert obs_dim == 91
  assert forbidden_input_feature_names() == ()

  path = tmp_path / "samples.npz"
  np.savez(path, obs_history=np.zeros((2, 4, obs_dim), dtype=np.float32))
  loaded = np.load(path)

  assert loaded["obs_history"].shape[-1] == 91


def test_privileged_footprint_features_encode_partial_adjacent_support() -> None:
  features = privileged_footprint_features_from_tensors(
    ground_contact=torch.tensor([[True, True]]),
    stair_support=torch.tensor([[True, True]]),
    support_fraction=torch.tensor([[0.5, 1.0]]),
    support_layer=torch.tensor([[1, 2]]),
    contact_duration=torch.tensor([[0.2, 0.4]]),
  )

  assert features.shape == (1, privileged_footprint_obs_dim())
  assert features[0, 4].item() == 0.5
  assert features[0, 5].item() == 1.0
  assert features[0, 13].item() == 1.0  # pair_adjacent_layers
  assert features[0, 16].item() == 1.0  # any_partial_tread_footprint


def test_builder_stores_privileged_footprint_history() -> None:
  obs_dim = input_obs_dim()
  footprint_dim = privileged_footprint_obs_dim()
  builder = StairProbeDatasetBuilder(
    num_envs=1,
    history_len=2,
    obs_dim=obs_dim,
    privileged_footprint_dim=footprint_dim,
    privileged_footprint_history_len=3,
    max_samples=8,
    seed=42,
    device="cpu",
    flat_sample_period=1,
  )
  builder.push_observations(torch.ones(1, obs_dim), torch.tensor([True]))
  builder.push_privileged_footprint(
    torch.full((1, footprint_dim), 2.0),
    torch.tensor([True]),
  )
  labels = _update_single_env(StairProbeLevelState(num_envs=1), layer=-1, active=False)

  builder.collect(labels, frame_idx=1)
  arrays = builder.as_arrays()

  assert arrays["privileged_footprint_history"].shape == (1, 3, footprint_dim)
  assert arrays["privileged_footprint_valid_mask"].tolist() == [[False, False, True]]
