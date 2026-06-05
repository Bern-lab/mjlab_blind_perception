"""Slow-latent actor models for deployable blind locomotion."""

from __future__ import annotations

import copy
from typing import Any, TypeAlias, cast

import torch
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict
from torch import nn

SlowLatentState: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
SlowLatentHiddenState: TypeAlias = SlowLatentState | None
SlowLatentStateLike: TypeAlias = HiddenState | SlowLatentState | list[torch.Tensor]


class LSTMSlowLatentMLPModel(MLPModel):
  """LSTM encoder + slow terrain latent + MLP actor.

  The LSTM is only a history encoder. The action head is still the standard MLP
  from :class:`MLPModel`, conditioned on ``[o_t, z_t]``.
  """

  is_recurrent: bool = True

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    use_slow_latent: bool = True,
    latent_dim: int = 16,
    latent_hidden_dim: int = 256,
    latent_alpha: float = 0.1,
    encoder_type: str = "lstm",
    rnn_type: str | None = "lstm",
    rnn_hidden_dim: int | None = None,
    rnn_num_layers: int = 1,
    **_: Any,
  ) -> None:
    if not use_slow_latent:
      raise ValueError("LSTMSlowLatentMLPModel requires use_slow_latent=True.")
    encoder_type = encoder_type.lower()
    rnn_type = (rnn_type or encoder_type).lower()
    if encoder_type != "lstm" or rnn_type != "lstm":
      raise ValueError("LSTMSlowLatentMLPModel currently supports only LSTM.")
    if latent_dim <= 0:
      raise ValueError("latent_dim must be positive.")
    if latent_hidden_dim <= 0:
      raise ValueError("latent_hidden_dim must be positive.")
    if not 0.0 < latent_alpha <= 1.0:
      raise ValueError("latent_alpha must be in (0, 1].")

    self.rnn_type = "lstm"
    self.hidden_size = int(latent_hidden_dim)
    self.num_layers = int(rnn_num_layers)
    self.latent_dim = int(latent_dim)
    self.latent_hidden_dim = self.hidden_size
    self.latent_alpha = float(latent_alpha)
    self.slow_latent_dim = self.latent_dim
    self.slow_alpha = self.latent_alpha
    self._hidden_state: SlowLatentHiddenState = None

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

    self.encoder = nn.LSTM(
      input_size=self.obs_dim,
      hidden_size=self.hidden_size,
      num_layers=self.num_layers,
    )
    self.rnn = self.encoder
    self.latent_head = nn.Linear(self.hidden_size, self.latent_dim)

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: SlowLatentStateLike = None,
  ) -> torch.Tensor:
    """Return the MLP actor input ``[normalized_obs, slow_latent]``."""
    actor_obs = MLPModel.get_latent(self, obs)
    if masks is None:
      return self._get_step_latent(actor_obs)
    return self._get_sequence_latent(actor_obs, masks, hidden_state)

  def reset(
    self,
    dones: torch.Tensor | None = None,
    hidden_state: SlowLatentStateLike = None,
  ) -> None:
    """Reset LSTM hidden state and slow latent state."""
    if dones is None:
      self._hidden_state = self._as_slow_latent_state(hidden_state)
      return

    if self._hidden_state is None:
      return

    done_mask = dones == 1
    for state in self._hidden_state:
      state[..., done_mask, :] = 0.0

  def reset_slow_latent(self, dones: torch.Tensor | None = None) -> None:
    """Clear only z_t, useful for latent-memory ablations during evaluation."""
    if self._hidden_state is None:
      return

    z = self._hidden_state[2]
    if dones is None:
      z[:] = 0.0
    else:
      z[..., dones == 1, :] = 0.0

  def get_hidden_state(self) -> Any:
    """Return ``(h, c, z)`` so rollout storage can restore full latent memory."""
    return self._hidden_state

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    """Detach recurrent state for truncated backpropagation."""
    if self._hidden_state is None:
      return

    if dones is None:
      h, c, z = self._hidden_state
      self._hidden_state = (h.detach(), c.detach(), z.detach())
      return

    done_mask = dones == 1
    for state in self._hidden_state:
      state[..., done_mask, :] = state[..., done_mask, :].detach()

  def as_jit(self) -> nn.Module:
    """Return a TorchScript-friendly stateful inference wrapper."""
    return _TorchLSTMSlowLatentMLPModel(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    """Return an ONNX wrapper with explicit h/c/z recurrent inputs."""
    return _OnnxLSTMSlowLatentMLPModel(self, verbose)

  def _get_latent_dim(self) -> int:
    """Return the MLP input dimension after concatenating z_t."""
    return self.obs_dim + self.latent_dim

  def _get_step_latent(self, actor_obs: torch.Tensor) -> torch.Tensor:
    rnn_state, z_prev = self._split_slow_latent_state(
      self._hidden_state,
      batch_size=actor_obs.shape[0],
      reference=actor_obs,
    )
    rnn_out, (h, c) = self.encoder(actor_obs.unsqueeze(0), rnn_state)
    z_candidate = self.latent_head(rnn_out.squeeze(0))
    z = self._slow_update(z_prev.squeeze(0), z_candidate)
    self._hidden_state = (h, c, z.unsqueeze(0))
    return torch.cat((actor_obs, z), dim=-1)

  def _get_sequence_latent(
    self,
    actor_obs: torch.Tensor,
    masks: torch.Tensor,
    hidden_state: SlowLatentStateLike,
  ) -> torch.Tensor:
    if hidden_state is None:
      raise ValueError("Slow-latent recurrent updates require saved hidden state.")

    rnn_state, z_prev = self._split_slow_latent_state(
      hidden_state,
      batch_size=actor_obs.shape[1],
      reference=actor_obs,
    )
    rnn_out, _ = self.encoder(actor_obs, rnn_state)
    z_candidates = self.latent_head(rnn_out)
    z = self._slow_update_sequence(z_prev.squeeze(0), z_candidates, masks)
    return cast(
      torch.Tensor,
      unpad_trajectories(torch.cat((actor_obs, z), dim=-1), masks),
    )

  def _slow_update(
    self,
    z_prev: torch.Tensor,
    z_candidate: torch.Tensor,
  ) -> torch.Tensor:
    return (1.0 - self.latent_alpha) * z_prev + self.latent_alpha * z_candidate

  def _slow_update_sequence(
    self,
    z_prev: torch.Tensor,
    z_candidates: torch.Tensor,
    masks: torch.Tensor,
  ) -> torch.Tensor:
    z = z_prev
    z_steps = []
    valid = masks.bool().unsqueeze(-1)
    for step in range(z_candidates.shape[0]):
      z_next = self._slow_update(z, z_candidates[step])
      z = torch.where(valid[step], z_next, z)
      z_steps.append(z)
    return torch.stack(z_steps, dim=0)

  def _as_slow_latent_state(
    self,
    hidden_state: SlowLatentStateLike,
  ) -> SlowLatentHiddenState:
    if hidden_state is None:
      return None
    if not isinstance(hidden_state, tuple | list) or len(hidden_state) != 3:
      raise ValueError("Slow-latent LSTM hidden_state must be (h, c, z).")
    h, c, z = cast(SlowLatentState, hidden_state)
    return h, c, z

  def _split_slow_latent_state(
    self,
    hidden_state: SlowLatentStateLike,
    batch_size: int,
    reference: torch.Tensor,
  ) -> tuple[tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor]:
    state = self._as_slow_latent_state(hidden_state)
    if state is None:
      z = reference.new_zeros(1, batch_size, self.latent_dim)
      return None, z

    h, c, z = state
    return (h, c), z


class _TorchLSTMSlowLatentMLPModel(nn.Module):
  """TorchScript wrapper with internal h/c/z state."""

  hidden_state: torch.Tensor
  cell_state: torch.Tensor
  slow_latent: torch.Tensor

  def __init__(self, model: LSTMSlowLatentMLPModel) -> None:
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.encoder = copy.deepcopy(model.encoder)
    self.latent_head = copy.deepcopy(model.latent_head)
    self.mlp = copy.deepcopy(model.mlp)
    self.latent_alpha = model.latent_alpha
    if model.distribution is None:
      self.deterministic_output = nn.Identity()
    else:
      self.deterministic_output = model.distribution.as_deterministic_output_module()

    self.register_buffer(
      "hidden_state",
      torch.zeros(model.num_layers, 1, model.hidden_size),
    )
    self.register_buffer(
      "cell_state",
      torch.zeros(model.num_layers, 1, model.hidden_size),
    )
    self.register_buffer("slow_latent", torch.zeros(1, 1, model.latent_dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Run one deterministic inference step."""
    x = self.obs_normalizer(x)
    rnn_out, (h, c) = self.encoder(
      x.unsqueeze(0),
      (self.hidden_state, self.cell_state),
    )
    z_candidate = self.latent_head(rnn_out.squeeze(0))
    z = (1.0 - self.latent_alpha) * self.slow_latent.squeeze(
      0
    ) + self.latent_alpha * z_candidate
    self.hidden_state[:] = h
    self.cell_state[:] = c
    self.slow_latent[:] = z.unsqueeze(0)
    out = self.mlp(torch.cat((x, z), dim=-1))
    return self.deterministic_output(out)

  @torch.jit.export
  def reset(self) -> None:
    """Reset all recurrent inference state."""
    self.hidden_state[:] = 0.0
    self.cell_state[:] = 0.0
    self.slow_latent[:] = 0.0

  @torch.jit.export
  def reset_slow_latent(self) -> None:
    """Reset only the slow latent memory."""
    self.slow_latent[:] = 0.0


class _OnnxLSTMSlowLatentMLPModel(nn.Module):
  """ONNX wrapper with explicit recurrent latent state."""

  is_recurrent: bool = True

  def __init__(self, model: LSTMSlowLatentMLPModel, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.encoder = copy.deepcopy(model.encoder)
    self.latent_head = copy.deepcopy(model.latent_head)
    self.mlp = copy.deepcopy(model.mlp)
    self.latent_alpha = model.latent_alpha
    self.input_size = model.obs_dim
    self.hidden_size = model.hidden_size
    self.num_layers = model.num_layers
    self.latent_dim = model.latent_dim
    if model.distribution is None:
      self.deterministic_output = nn.Identity()
    else:
      self.deterministic_output = model.distribution.as_deterministic_output_module()

  def forward(
    self,
    obs: torch.Tensor,
    h_in: torch.Tensor,
    c_in: torch.Tensor,
    z_in: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one deterministic inference step."""
    x = self.obs_normalizer(obs)
    rnn_out, (h, c) = self.encoder(x.unsqueeze(0), (h_in, c_in))
    z_candidate = self.latent_head(rnn_out.squeeze(0))
    z = (1.0 - self.latent_alpha) * z_in.squeeze(0) + self.latent_alpha * z_candidate
    out = self.mlp(torch.cat((x, z), dim=-1))
    actions = self.deterministic_output(out)
    return actions, h, c, z.unsqueeze(0)

  def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
    """Return dummy ONNX inputs."""
    return (
      torch.zeros(1, self.input_size),
      torch.zeros(self.num_layers, 1, self.hidden_size),
      torch.zeros(self.num_layers, 1, self.hidden_size),
      torch.zeros(1, 1, self.latent_dim),
    )

  @property
  def input_names(self) -> list[str]:
    """Return ONNX input names."""
    return ["obs", "h_in", "c_in", "z_in"]

  @property
  def output_names(self) -> list[str]:
    """Return ONNX output names."""
    return ["actions", "h_out", "c_out", "z_out"]


__all__ = ["LSTMSlowLatentMLPModel"]
