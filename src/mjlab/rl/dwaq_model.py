"""DWAQ actor model migrated from G1DWAQ_Lab into local RSL-RL APIs."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import EmpiricalNormalization, HiddenState
from tensordict import TensorDict


def _activation(name: str) -> nn.Module:
  if name == "elu":
    return nn.ELU()
  if name == "selu":
    return nn.SELU()
  if name == "relu":
    return nn.ReLU()
  if name == "lrelu":
    return nn.LeakyReLU()
  if name == "tanh":
    return nn.Tanh()
  if name == "sigmoid":
    return nn.Sigmoid()
  raise ValueError(f"Unsupported DWAQ activation: {name!r}")


def _mlp_layers(
  input_dim: int,
  hidden_dims: Sequence[int],
  activation: str,
  output_dim: int | None = None,
) -> nn.Sequential:
  layers: list[nn.Module] = []
  in_dim = input_dim
  for out_dim in hidden_dims:
    layers += [nn.Linear(in_dim, out_dim), _activation(activation)]
    in_dim = out_dim
  if output_dim is not None:
    layers.append(nn.Linear(in_dim, output_dim))
  return nn.Sequential(*layers)


class DWAQMLPModel(MLPModel):
  """Feedforward actor with a DreamWaQ-style beta-VAE context encoder.

  This is intentionally close to ``ActorCritic_DWAQ`` from G1DWAQ_Lab, but it
  uses the TensorDict observation and distribution interfaces expected by this
  repository's newer RSL-RL runner. The actor receives:

  - current blind actor observations from ``obs_set='actor'``;
  - a DWAQ code inferred from ``history_obs_set`` observation history.
  """

  is_recurrent: bool = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = True,
    distribution_cfg: dict | None = None,
    history_obs_set: str = "dwaq_history",
    encoder_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
    decoder_hidden_dims: tuple[int, ...] | list[int] = (64, 128),
    velocity_dim: int = 3,
    latent_dim: int = 16,
    cenet_out_dim: int | None = None,
    sample_code_in_eval: bool = False,
    **kwargs: Any,
  ) -> None:
    del kwargs
    self.velocity_dim = int(velocity_dim)
    if cenet_out_dim is not None:
      latent_dim = int(cenet_out_dim) - self.velocity_dim
    self.dwaq_latent_dim = int(latent_dim)
    if self.velocity_dim <= 0 or self.dwaq_latent_dim <= 0:
      raise ValueError(
        "DWAQ velocity_dim and latent_dim must be positive, got "
        f"velocity_dim={self.velocity_dim}, latent_dim={self.dwaq_latent_dim}."
      )
    self.cenet_out_dim = self.velocity_dim + self.dwaq_latent_dim
    self.sample_code_in_eval = bool(sample_code_in_eval)

    if history_obs_set not in obs_groups:
      raise ValueError(
        f"DWAQMLPModel requires obs_groups['{history_obs_set}']. "
        f"Available sets: {sorted(obs_groups)}"
      )
    self.history_obs_set = history_obs_set
    self._history_obs_group_names = list(obs_groups[history_obs_set])
    self._latent_obs_group_names = list(self._history_obs_group_names)

    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )
    self._actor_obs_group_names = list(self.obs_groups)

    self.history_obs_dim = self._get_flat_group_dim(obs, self._history_obs_group_names)
    self.history_obs_shape = self._get_history_shape(obs, self._history_obs_group_names)
    self.history_obs_normalizer: nn.Module
    if obs_normalization:
      self.history_obs_normalizer = EmpiricalNormalization(self.history_obs_dim)
    else:
      self.history_obs_normalizer = nn.Identity()

    if len(encoder_hidden_dims) < 1:
      raise ValueError("DWAQ encoder_hidden_dims must contain at least one layer.")
    encoder_out_dim = int(encoder_hidden_dims[-1])
    self.encoder = _mlp_layers(self.history_obs_dim, encoder_hidden_dims, activation)
    self.encode_mean_latent = nn.Linear(encoder_out_dim, self.dwaq_latent_dim)
    self.encode_logvar_latent = nn.Linear(encoder_out_dim, self.dwaq_latent_dim)
    self.encode_mean_vel = nn.Linear(encoder_out_dim, self.velocity_dim)
    self.decoder = _mlp_layers(
      self.cenet_out_dim,
      decoder_hidden_dims,
      activation,
      output_dim=self.obs_dim,
    )

  def _get_latent_dim(self) -> int:
    """Actor MLP consumes ``[DWAQ code, current actor obs]``."""
    return self.cenet_out_dim + self.obs_dim

  @staticmethod
  def _flatten_obs_groups(obs: TensorDict, group_names: Sequence[str]) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for name in group_names:
      if name not in obs:
        raise ValueError(f"Required DWAQ observation group {name!r} is missing.")
      value = obs[name]
      values.append(value.reshape(value.shape[0], -1))
    return torch.cat(values, dim=-1)

  @staticmethod
  def _get_flat_group_dim(obs: TensorDict, group_names: Sequence[str]) -> int:
    dim = 0
    for name in group_names:
      if name not in obs:
        raise ValueError(f"Required DWAQ observation group {name!r} is missing.")
      dim += int(torch.tensor(obs[name].shape[1:]).prod().item())
    return dim

  @staticmethod
  def _get_history_shape(
    obs: TensorDict, group_names: Sequence[str]
  ) -> tuple[int, ...]:
    if len(group_names) == 1:
      return tuple(int(v) for v in obs[group_names[0]].shape[1:])
    return (DWAQMLPModel._get_flat_group_dim(obs, group_names),)

  def get_actor_observation(self, obs: TensorDict) -> torch.Tensor:
    """Return the raw current actor observation used as decoder target."""
    return self._flatten_obs_groups(obs, self._actor_obs_group_names)

  def get_history_observation(self, obs: TensorDict) -> torch.Tensor:
    """Return the flattened raw DWAQ history observation."""
    return self._flatten_obs_groups(obs, self._history_obs_group_names)

  @staticmethod
  def reparameterise(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """G1DWAQ_Lab VAE reparameterization with the same logvar clamp."""
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    std = torch.exp(logvar * 0.5)
    return mean + std * torch.randn_like(std)

  def cenet_forward(
    self,
    obs_history: torch.Tensor,
    sample: bool | None = None,
  ) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
  ]:
    """Forward pass through the DWAQ context encoder and decoder."""
    if sample is None:
      sample = self.training or self.sample_code_in_eval
    obs_history = self.history_obs_normalizer(obs_history)
    encoded = self.encoder(obs_history)
    mean_latent = self.encode_mean_latent(encoded)
    logvar_latent = self.encode_logvar_latent(encoded)
    mean_vel = self.encode_mean_vel(encoded)
    logvar_vel = torch.zeros_like(mean_vel)
    if sample:
      code_latent = self.reparameterise(mean_latent, logvar_latent)
    else:
      code_latent = mean_latent
    code_vel = mean_vel
    code = torch.cat((code_vel, code_latent), dim=-1)
    decode = self.decoder(code)
    return (
      code,
      code_vel,
      decode,
      mean_vel,
      logvar_vel,
      mean_latent,
      logvar_latent,
    )

  def get_dwaq_outputs(
    self,
    obs: TensorDict,
    sample: bool | None = None,
  ) -> dict[str, torch.Tensor]:
    """Compute named DWAQ outputs for the auxiliary PPO loss."""
    outputs = self.cenet_forward(self.get_history_observation(obs), sample=sample)
    keys = (
      "code",
      "code_vel",
      "decode",
      "mean_vel",
      "logvar_vel",
      "mean_latent",
      "logvar_latent",
    )
    return dict(zip(keys, outputs, strict=True))

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    """Build ``[DWAQ code, normalized current actor obs]`` for the actor MLP."""
    del masks, hidden_state
    current_obs = self.get_actor_observation(obs)
    current_obs = self.obs_normalizer(current_obs)
    code = self.get_dwaq_outputs(obs)["code"]
    return torch.cat((code, current_obs), dim=-1)

  def update_normalization(self, obs: TensorDict) -> None:
    """Update current-observation and history normalizers."""
    if not self.obs_normalization:
      return
    if isinstance(self.obs_normalizer, EmpiricalNormalization):
      self.obs_normalizer.update(self.get_actor_observation(obs))
    if isinstance(self.history_obs_normalizer, EmpiricalNormalization):
      self.history_obs_normalizer.update(self.get_history_observation(obs))

  def as_jit(self) -> nn.Module:
    """Return a TorchScript-friendly deterministic DWAQ actor."""
    return _TorchDWAQModel(self)

  def as_onnx(self, verbose: bool) -> nn.Module:
    """Return an ONNX export wrapper with obs and history inputs."""
    return _OnnxDWAQModel(self, verbose)


class _BaseDWAQExportModel(nn.Module):
  def __init__(self, model: DWAQMLPModel) -> None:
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.history_obs_normalizer = copy.deepcopy(model.history_obs_normalizer)
    self.encoder = copy.deepcopy(model.encoder)
    self.encode_mean_latent = copy.deepcopy(model.encode_mean_latent)
    self.encode_mean_vel = copy.deepcopy(model.encode_mean_vel)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self.input_size = model.obs_dim
    self.history_input_shape = model.history_obs_shape

  def forward(self, obs: torch.Tensor, obs_history: torch.Tensor) -> torch.Tensor:
    current_obs = self.obs_normalizer(obs)
    history = obs_history.reshape(obs_history.shape[0], -1)
    history = self.history_obs_normalizer(history)
    encoded = self.encoder(history)
    code_vel = self.encode_mean_vel(encoded)
    code_latent = self.encode_mean_latent(encoded)
    code = torch.cat((code_vel, code_latent), dim=-1)
    out = self.mlp(torch.cat((code, current_obs), dim=-1))
    return self.deterministic_output(out)


class _TorchDWAQModel(_BaseDWAQExportModel):
  @torch.jit.export
  def reset(self) -> None:
    """Reset export state (no-op for feedforward DWAQ)."""
    pass


class _OnnxDWAQModel(_BaseDWAQExportModel):
  is_recurrent: bool = False

  def __init__(self, model: DWAQMLPModel, verbose: bool) -> None:
    super().__init__(model)
    self.verbose = verbose

  def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
    return (
      torch.zeros(1, self.input_size),
      torch.zeros(1, *self.history_input_shape),
    )

  @property
  def input_names(self) -> list[str]:
    return ["obs", "obs_history"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
