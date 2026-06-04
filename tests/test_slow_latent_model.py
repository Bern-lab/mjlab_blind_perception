"""Tests for the LSTM slow-latent actor model."""

from __future__ import annotations

import tempfile
from pathlib import Path

import onnx
import torch
from tensordict import TensorDict

from mjlab.rl.slow_latent_model import LSTMSlowLatentMLPModel
from mjlab.utils.lstm import get_recurrent_policy_metadata


def _make_obs(num_envs: int = 4, obs_dim: int = 8) -> TensorDict:
  return TensorDict({"actor": torch.randn(num_envs, obs_dim)})


def _make_model(alpha: float = 0.1) -> LSTMSlowLatentMLPModel:
  obs = _make_obs()
  return LSTMSlowLatentMLPModel(
    obs=obs,
    obs_groups={"actor": ["actor"]},
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
    latent_dim=5,
    latent_hidden_dim=7,
    latent_alpha=alpha,
  )


def test_slow_latent_actor_forward_updates_state() -> None:
  model = _make_model()
  obs = _make_obs()

  actions = model(obs, stochastic_output=False)
  h, c, z = model.get_hidden_state()

  assert actions.shape == (4, 3)
  assert h.shape == (1, 4, 7)
  assert c.shape == (1, 4, 7)
  assert z.shape == (1, 4, 5)


def test_reset_done_env_clears_hidden_cell_and_slow_latent() -> None:
  model = _make_model()
  obs = _make_obs()
  model(obs)

  dones = torch.tensor([False, True, False, False])
  model.reset(dones)
  h, c, z = model.get_hidden_state()

  assert torch.all(h[:, 1, :] == 0.0)
  assert torch.all(c[:, 1, :] == 0.0)
  assert torch.all(z[:, 1, :] == 0.0)


def test_alpha_one_matches_candidate_latent_on_first_step() -> None:
  model = _make_model(alpha=1.0)
  obs = _make_obs()

  actor_obs = obs["actor"]
  with torch.no_grad():
    rnn_out, _ = model.encoder(actor_obs.unsqueeze(0), None)
    candidate = model.latent_head(rnn_out.squeeze(0))

  model(obs)
  _h, _c, z = model.get_hidden_state()

  torch.testing.assert_close(z.squeeze(0), candidate)


def test_reset_slow_latent_keeps_lstm_state() -> None:
  model = _make_model()
  obs = _make_obs()
  model(obs)
  h_before, c_before, _z_before = model.get_hidden_state()
  h_before = h_before.clone()
  c_before = c_before.clone()

  model.reset_slow_latent()
  h, c, z = model.get_hidden_state()

  torch.testing.assert_close(h, h_before)
  torch.testing.assert_close(c, c_before)
  assert torch.all(z == 0.0)


def test_onnx_wrapper_exposes_slow_latent_state() -> None:
  model = _make_model()
  onnx_model = model.as_onnx()

  assert onnx_model.input_names == ["obs", "h_in", "c_in", "z_in"]
  assert onnx_model.output_names == ["actions", "h_out", "c_out", "z_out"]

  outputs = onnx_model(*onnx_model.get_dummy_inputs())
  assert outputs[0].shape == (1, 3)
  assert outputs[1].shape == (1, 1, 7)
  assert outputs[2].shape == (1, 1, 7)
  assert outputs[3].shape == (1, 1, 5)


def test_slow_latent_export_metadata() -> None:
  model = _make_model()
  metadata = get_recurrent_policy_metadata(model)

  assert metadata["policy_has_slow_latent"] == "true"
  assert metadata["policy_slow_latent_dim"] == "5"
  assert metadata["policy_slow_latent_alpha"] == "0.1"
  assert metadata["policy_onnx_input_names"] == ["obs", "h_in", "c_in", "z_in"]
  assert metadata["policy_onnx_output_names"] == [
    "actions",
    "h_out",
    "c_out",
    "z_out",
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
