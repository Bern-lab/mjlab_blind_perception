"""Tests for the gated stair slow-latent actor model."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast

import onnx
import pytest
import torch
from rsl_rl.algorithms.ppo_teacher_kl import PPOTeacherKL
from tensordict import TensorDict

from mjlab.rl.slow_latent_model import LSTMSlowLatentMLPModel
from mjlab.utils.lstm import get_recurrent_policy_metadata


def _make_obs(
  num_envs: int = 4,
  actor_dim: int = 8,
  latent_dim: int = 11,
) -> TensorDict:
  return TensorDict(
    {
      "actor": torch.randn(num_envs, actor_dim),
      "latent": torch.randn(num_envs, latent_dim),
    },
    batch_size=[num_envs],
  )


def _make_model() -> LSTMSlowLatentMLPModel:
  obs = _make_obs()
  return LSTMSlowLatentMLPModel(
    obs=obs,
    obs_groups={"actor": ["actor"], "latent": ["latent"]},
    obs_set="actor",
    output_dim=3,
    hidden_dims=(32, 16),
    activation="elu",
    obs_normalization=False,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    mlp_encoder_dims=(13,),
    latent_hidden_dim=7,
    latent_dim=5,
    alpha_fast=0.3,
    alpha_write=0.8,
    alpha_hold=0.02,
  )


def test_slow_latent_actor_forward_updates_state_and_aux_outputs() -> None:
  model = _make_model()
  obs = _make_obs()

  actions = model(obs, stochastic_output=False)
  hidden_state = model.get_hidden_state()
  aux = model.get_aux_outputs()

  assert actions.shape == (4, 3)
  assert isinstance(hidden_state, tuple)
  hidden_state = cast(tuple[torch.Tensor, ...], hidden_state)
  assert len(hidden_state) == 8
  assert hidden_state[0].shape == (1, 4, 7)
  assert hidden_state[1].shape == (1, 4, 7)
  assert hidden_state[2].shape == (1, 4, 5)
  assert hidden_state[3].shape == (1, 4, 1)
  assert aux["event_logit"].shape == (4, 1)
  assert aux["stair_logit"].shape == (4, 1)
  assert aux["future_collision_logit"].shape == (4, 1)
  assert aux["stair_shape"].shape == (4, 2)
  diagnostics = model.get_slow_latent_diagnostics()
  assert diagnostics["event_prob"].shape == (4, 1)
  assert diagnostics["stair_prob"].shape == (4, 1)
  assert diagnostics["future_prob"].shape == (4, 1)
  assert diagnostics["stair_shape"].shape == (4, 2)
  assert diagnostics["z_norm"].shape == (4, 1)
  assert diagnostics["gate_mode"].shape == (4, 1)
  assert diagnostics["alpha"].shape == (4, 1)


def test_stair_shape_huber_ignores_invalid_labels() -> None:
  predictions = torch.tensor([[0.40, 0.10], [10.0, 10.0]])
  labels = torch.tensor([[0.30, 0.10], [0.30, 0.10]])
  valid = torch.tensor([[1.0], [0.0]])

  loss = PPOTeacherKL._compute_stair_shape_loss(
    predictions,
    labels,
    valid,
    huber_delta=0.05,
  )

  assert loss.item() == pytest.approx(0.0375)

  mae, huber = PPOTeacherKL._compute_stair_shape_component_errors(
    predictions,
    labels,
    valid,
    huber_delta=0.05,
  )

  assert mae.tolist() == pytest.approx([0.1, 0.0])
  assert huber.tolist() == pytest.approx([0.075, 0.0])


def test_reset_done_env_clears_recurrent_latent_and_gate_state() -> None:
  model = _make_model()
  model(_make_obs())

  dones = torch.tensor([False, True, False, False])
  model.reset(dones)
  hidden_state = model.get_hidden_state()
  assert isinstance(hidden_state, tuple)
  hidden_state = cast(tuple[torch.Tensor, ...], hidden_state)

  for state in hidden_state:
    assert torch.all(state[:, 1, :] == 0.0)


def test_reset_slow_latent_keeps_lstm_state() -> None:
  model = _make_model()
  model(_make_obs())
  hidden_before = model.get_hidden_state()
  assert isinstance(hidden_before, tuple)
  hidden_before = cast(tuple[torch.Tensor, ...], hidden_before)
  h_before = hidden_before[0].clone()
  c_before = hidden_before[1].clone()

  model.reset_slow_latent()
  hidden_state = model.get_hidden_state()
  assert isinstance(hidden_state, tuple)
  hidden_state = cast(tuple[torch.Tensor, ...], hidden_state)

  torch.testing.assert_close(hidden_state[0], h_before)
  torch.testing.assert_close(hidden_state[1], c_before)
  assert torch.all(hidden_state[2] == 0.0)
  for state in hidden_state[3:]:
    assert torch.all(state == 0.0)


def test_recurrent_batch_forward_unpads_aux_outputs() -> None:
  model = _make_model()
  obs = TensorDict(
    {
      "actor": torch.randn(3, 2, 8),
      "latent": torch.randn(3, 2, 11),
    },
    batch_size=[3, 2],
  )
  masks = torch.tensor(
    [
      [True, True],
      [True, False],
      [False, False],
    ]
  )
  h = torch.zeros(1, 2, 7)
  c = torch.zeros(1, 2, 7)
  z = torch.zeros(1, 2, 5)
  gate = torch.zeros(1, 2, 1)

  actions = model(
    obs,
    masks=masks,
    hidden_state=(h, c, z, gate, gate, gate, gate, gate),
    stochastic_output=False,
  )

  assert actions.shape == (3, 1, 3)
  assert model.aux_event_logits is not None
  assert model.aux_event_logits.shape == (3, 1, 1)
  diagnostics = model.get_slow_latent_diagnostics()
  assert diagnostics["alpha"].shape == (3, 1, 1)
  assert diagnostics["gate_mode"].shape == (3, 1, 1)


def test_onnx_wrapper_exposes_gated_slow_latent_state() -> None:
  model = _make_model()
  onnx_model = model.as_onnx()

  assert onnx_model.input_names == [
    "actor_obs",
    "latent_obs",
    "h_in",
    "c_in",
    "z_in",
    "gate_state_in",
  ]
  assert onnx_model.output_names == [
    "actions",
    "h_out",
    "c_out",
    "z_out",
    "gate_state_out",
    "event_prob",
    "stair_prob",
    "future_collision_prob",
  ]

  outputs = onnx_model(*onnx_model.get_dummy_inputs())
  assert outputs[0].shape == (1, 3)
  assert outputs[1].shape == (1, 1, 7)
  assert outputs[2].shape == (1, 1, 7)
  assert outputs[3].shape == (1, 5)
  assert outputs[4].shape == (1, 5)


def test_slow_latent_export_metadata() -> None:
  model = _make_model()
  metadata = get_recurrent_policy_metadata(model)

  assert metadata["policy_has_slow_latent"] == "true"
  assert metadata["policy_slow_latent_dim"] == "5"
  assert metadata["policy_slow_latent_alpha"] == "0.02"
  assert metadata["policy_latent_obs_dim"] == "11"
  assert metadata["policy_onnx_input_names"] == [
    "actor_obs",
    "latent_obs",
    "h_in",
    "c_in",
    "z_in",
    "gate_state_in",
  ]


def test_onnx_export_slow_latent_model() -> None:
  model = _make_model()
  onnx_model = model.as_onnx()

  with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "slow_latent_policy.onnx"
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),
      str(path),
      input_names=onnx_model.input_names,
      output_names=onnx_model.output_names,
      opset_version=18,
      dynamo=False,
    )
    onnx.checker.check_model(str(path))
