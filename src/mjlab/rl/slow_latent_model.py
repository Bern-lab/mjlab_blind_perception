"""Deployable stair-focused gated slow-latent actor model."""

from __future__ import annotations

import copy
from typing import Any, cast

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import EmpiricalNormalization, HiddenState
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict

_GATE_STATE_DIM = 5
_MODE_NORMAL = 0.0
_MODE_STAIR_WRITE = 1.0
_MODE_STAIR_MEMORY = 2.0
GatedHiddenState = torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor] | None


class LSTMSlowLatentMLPModel(MLPModel):
  """LSTM gated latent memory actor for stair interaction.

  The model consumes two observation sets:

  - ``actor``: the normal blind policy observation used by the actor.
  - ``latent``: deployable proprioceptive stair-interaction features.

  ``latent`` is encoded by an MLP and LSTM into a candidate latent. A small
  state machine, driven by the model's event and stair-state predictions, writes
  this candidate into a compact ``z_memory``. The actor MLP receives
  ``concat(actor_obs, z_memory)``.
  """

  is_recurrent: bool = True

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
    mlp_encoder_dims: tuple[int, ...] | list[int] = (128, 128),
    latent_hidden_dim: int = 128,
    z_dim: int | None = None,
    latent_dim: int | None = None,
    alpha_fast: float = 0.3,
    alpha_write: float = 0.8,
    alpha_hold: float = 0.02,
    write_steps: int = 2,
    min_stair_steps: int = 50,
    exit_steps: int = 100,
    cooldown_steps: int = 15,
    event_on_threshold: float = 0.6,
    event_off_threshold: float = 0.4,
    stair_off_threshold: float = 0.4,
    aux_future_collision_coef: float = 0.05,
    aux_event_coef: float = 0.03,
    aux_stair_coef: float = 0.02,
    aux_stair_shape_coef: float = 0.0,
    stair_shape_huber_delta: float = 0.05,
    future_collision_horizon: int = 20,
    latent_obs_set: str = "latent",
    **kwargs: Any,
  ) -> None:
    del kwargs
    self.z_dim = int(z_dim if z_dim is not None else (latent_dim or 16))
    self.latent_dim = self.z_dim
    self.slow_latent_dim = self.z_dim
    self.latent_hidden_dim = int(latent_hidden_dim)
    self.hidden_size = self.latent_hidden_dim
    self.num_layers = 1
    self.rnn_type = "lstm"

    if latent_obs_set not in obs_groups:
      raise ValueError(
        f"LSTMSlowLatentMLPModel requires obs_groups['{latent_obs_set}']. "
        f"Available sets: {sorted(obs_groups)}"
      )
    self.latent_obs_set = latent_obs_set
    self._latent_obs_group_names = list(obs_groups[latent_obs_set])

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

    self._latent_obs_dim = 0
    for name in self._latent_obs_group_names:
      if name not in obs:
        raise ValueError(
          f"Latent observation group {name!r} is missing from observations."
        )
      self._latent_obs_dim += obs[name].shape[-1]
    self.latent_obs_dim = self._latent_obs_dim

    self.latent_obs_normalizer: nn.Module
    if obs_normalization:
      self.latent_obs_normalizer = EmpiricalNormalization(self._latent_obs_dim)
    else:
      self.latent_obs_normalizer = nn.Identity()

    activation_cls = nn.ELU
    encoder_layers: list[nn.Module] = []
    in_dim = self._latent_obs_dim
    for out_dim in mlp_encoder_dims:
      encoder_layers += [nn.Linear(in_dim, out_dim), activation_cls()]
      in_dim = out_dim
    self.latent_encoder = nn.Sequential(*encoder_layers)
    encoder_out_dim = in_dim

    self.latent_lstm = nn.LSTM(
      input_size=encoder_out_dim,
      hidden_size=self.latent_hidden_dim,
      num_layers=1,
      batch_first=False,
    )
    self.z_candidate_head = nn.Linear(self.latent_hidden_dim, self.z_dim)
    self.event_head = nn.Sequential(
      nn.Linear(self.latent_hidden_dim, 64),
      activation_cls(),
      nn.Linear(64, 1),
    )
    self.stair_state_head = nn.Sequential(
      nn.Linear(self.latent_hidden_dim + self.z_dim, 64),
      activation_cls(),
      nn.Linear(64, 1),
    )
    self.future_collision_head = nn.Sequential(
      nn.Linear(self.z_dim, 64),
      activation_cls(),
      nn.Linear(64, 1),
    )
    self.stair_shape_head = nn.Sequential(
      nn.Linear(self.z_dim, 64),
      activation_cls(),
      nn.Linear(64, 2),
    )

    self.alpha_fast = float(alpha_fast)
    self.alpha_write = float(alpha_write)
    self.alpha_hold = float(alpha_hold)
    self.write_steps = float(write_steps)
    self.min_stair_steps = float(min_stair_steps)
    self.exit_steps = float(exit_steps)
    self.cooldown_steps = float(cooldown_steps)
    self.event_on_threshold = float(event_on_threshold)
    self.event_off_threshold = float(event_off_threshold)
    self.stair_off_threshold = float(stair_off_threshold)

    self.aux_future_collision_coef = float(aux_future_collision_coef)
    self.aux_event_coef = float(aux_event_coef)
    self.aux_stair_coef = float(aux_stair_coef)
    self.aux_stair_shape_coef = float(aux_stair_shape_coef)
    self.stair_shape_huber_delta = float(stair_shape_huber_delta)
    self.future_collision_horizon = int(future_collision_horizon)

    self._hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None
    self._z_memory: torch.Tensor | None = None
    self._gate_state: torch.Tensor | None = None
    self._aux_event_logits: torch.Tensor | None = None
    self._aux_stair_logits: torch.Tensor | None = None
    self._aux_future_collision_logits: torch.Tensor | None = None
    self._aux_stair_shape_predictions: torch.Tensor | None = None
    self._slow_latent_diagnostics: dict[str, torch.Tensor] = {}

  def _get_latent_dim(self) -> int:
    return self.obs_dim + self.z_dim

  def _cat_groups(self, obs: TensorDict, group_names: list[str]) -> torch.Tensor:
    tensors = [cast(torch.Tensor, obs[name]) for name in group_names]
    return torch.cat(tensors, dim=-1)

  def _ensure_rollout_state(self, batch_size: int, device: torch.device) -> None:
    if (
      self._hidden_state is not None
      and self._z_memory is not None
      and self._gate_state is not None
      and self._z_memory.shape[0] == batch_size
    ):
      return
    h = torch.zeros(1, batch_size, self.latent_hidden_dim, device=device)
    c = torch.zeros_like(h)
    self._hidden_state = (h, c)
    self._z_memory = torch.zeros(batch_size, self.z_dim, device=device)
    self._gate_state = torch.zeros(batch_size, _GATE_STATE_DIM, device=device)

  def _zeros_state(
    self, batch_size: int, device: torch.device
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h = torch.zeros(1, batch_size, self.latent_hidden_dim, device=device)
    c = torch.zeros_like(h)
    z = torch.zeros(batch_size, self.z_dim, device=device)
    gate = torch.zeros(batch_size, _GATE_STATE_DIM, device=device)
    return h, c, z, gate

  def _unpack_hidden_state(
    self,
    hidden_state: GatedHiddenState,
    batch_size: int,
    device: torch.device,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if hidden_state is None:
      self._ensure_rollout_state(batch_size, device)
      assert self._hidden_state is not None
      assert self._z_memory is not None
      assert self._gate_state is not None
      h, c = self._hidden_state
      return h, c, self._z_memory, self._gate_state

    states = (
      list(hidden_state) if isinstance(hidden_state, (list, tuple)) else [hidden_state]
    )
    h = states[0].to(device)
    c = states[1].to(device) if len(states) > 1 else torch.zeros_like(h)
    if len(states) > 2:
      z = states[2].squeeze(0).to(device)
    else:
      z = torch.zeros(batch_size, self.z_dim, device=device)
    if len(states) >= 8:
      gate = torch.cat([state.squeeze(0).to(device) for state in states[3:8]], dim=-1)
    else:
      gate = torch.zeros(batch_size, _GATE_STATE_DIM, device=device)
    return h, c, z, gate

  def _pack_hidden_state(
    self,
    h: torch.Tensor,
    c: torch.Tensor,
    z: torch.Tensor,
    gate: torch.Tensor,
  ) -> tuple[torch.Tensor, ...]:
    return (
      h,
      c,
      z.unsqueeze(0),
      gate[:, 0:1].unsqueeze(0),
      gate[:, 1:2].unsqueeze(0),
      gate[:, 2:3].unsqueeze(0),
      gate[:, 3:4].unsqueeze(0),
      gate[:, 4:5].unsqueeze(0),
    )

  def _advance_gate_state(
    self,
    event_prob: torch.Tensor,
    stair_prob: torch.Tensor,
    gate_state: torch.Tensor,
    valid: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    event_prob = event_prob.squeeze(-1)
    stair_prob = stair_prob.squeeze(-1)

    mode = gate_state[:, 0]
    stair_timer = gate_state[:, 1]
    no_event_timer = gate_state[:, 2]
    write_timer = gate_state[:, 3]
    cooldown = torch.clamp(gate_state[:, 4] - 1.0, min=0.0)

    in_normal = mode == _MODE_NORMAL
    trigger = in_normal & (cooldown <= 0.0) & (event_prob > self.event_on_threshold)
    mode = torch.where(trigger, torch.full_like(mode, _MODE_STAIR_WRITE), mode)
    write_timer = torch.where(trigger, torch.zeros_like(write_timer), write_timer)
    stair_timer = torch.where(trigger, torch.zeros_like(stair_timer), stair_timer)
    no_event_timer = torch.where(
      trigger, torch.zeros_like(no_event_timer), no_event_timer
    )

    in_write = mode == _MODE_STAIR_WRITE
    write_timer = torch.where(in_write, write_timer + 1.0, write_timer)
    stair_timer = torch.where(in_write, stair_timer + 1.0, stair_timer)
    done_write = in_write & (write_timer >= self.write_steps)
    mode = torch.where(done_write, torch.full_like(mode, _MODE_STAIR_MEMORY), mode)

    in_memory = mode == _MODE_STAIR_MEMORY
    stair_timer = torch.where(in_memory, stair_timer + 1.0, stair_timer)
    no_event = event_prob < self.event_off_threshold
    no_event_timer = torch.where(
      in_memory & no_event,
      no_event_timer + 1.0,
      torch.where(in_memory, torch.zeros_like(no_event_timer), no_event_timer),
    )
    exit_memory = (
      in_memory
      & (stair_prob < self.stair_off_threshold)
      & (no_event_timer > self.exit_steps)
      & (stair_timer > self.min_stair_steps)
    )
    mode = torch.where(exit_memory, torch.full_like(mode, _MODE_NORMAL), mode)
    cooldown = torch.where(
      exit_memory, torch.full_like(cooldown, self.cooldown_steps), cooldown
    )
    stair_timer = torch.where(exit_memory, torch.zeros_like(stair_timer), stair_timer)
    no_event_timer = torch.where(
      exit_memory, torch.zeros_like(no_event_timer), no_event_timer
    )
    write_timer = torch.where(exit_memory, torch.zeros_like(write_timer), write_timer)

    next_gate = torch.stack(
      [mode, stair_timer, no_event_timer, write_timer, cooldown], dim=-1
    )
    alpha = torch.full_like(event_prob, self.alpha_fast)
    alpha = torch.where(mode == _MODE_STAIR_WRITE, self.alpha_write, alpha)
    alpha = torch.where(mode == _MODE_STAIR_MEMORY, self.alpha_hold, alpha)
    alpha = alpha.unsqueeze(-1)

    if valid is not None:
      valid = valid.bool().squeeze(-1)
      next_gate = torch.where(valid[:, None], next_gate, gate_state)
      alpha = torch.where(
        valid[:, None],
        alpha,
        torch.zeros_like(alpha),
      )

    return next_gate, alpha

  def _run_latent_path(
    self,
    actor_obs: torch.Tensor,
    latent_obs: torch.Tensor,
    masks: torch.Tensor | None,
    hidden_state: GatedHiddenState,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence_mode = actor_obs.dim() == 3
    if not sequence_mode:
      actor_obs = actor_obs.unsqueeze(0)
      latent_obs = latent_obs.unsqueeze(0)

    seq_len, batch_size = actor_obs.shape[:2]
    device = actor_obs.device
    h_in, c_in, z, gate = self._unpack_hidden_state(hidden_state, batch_size, device)

    latent_flat = latent_obs.reshape(seq_len * batch_size, -1)
    latent_norm = self.latent_obs_normalizer(latent_flat).reshape(
      seq_len, batch_size, -1
    )
    encoded = self.latent_encoder(latent_norm)
    lstm_out, (h_out, c_out) = self.latent_lstm(encoded, (h_in, c_in))

    z_steps: list[torch.Tensor] = []
    alpha_steps: list[torch.Tensor] = []
    mode_steps: list[torch.Tensor] = []
    event_logits: list[torch.Tensor] = []
    stair_logits: list[torch.Tensor] = []
    future_logits: list[torch.Tensor] = []
    stair_shape_predictions: list[torch.Tensor] = []

    valid_masks = masks
    if valid_masks is not None and valid_masks.dim() == 2:
      valid_masks = valid_masks.unsqueeze(-1)

    for step in range(seq_len):
      h_t = lstm_out[step]
      z_candidate = self.z_candidate_head(h_t)
      event_logit = self.event_head(h_t)
      stair_gate_logit = self.stair_state_head(torch.cat([h_t, z], dim=-1))
      valid = valid_masks[step] if valid_masks is not None else None

      gate_next, alpha = self._advance_gate_state(
        torch.sigmoid(event_logit),
        torch.sigmoid(stair_gate_logit),
        gate,
        valid=valid,
      )
      z_next = (1.0 - alpha) * z + alpha * z_candidate
      if valid is not None:
        z_next = torch.where(valid.bool(), z_next, z)

      stair_logit = self.stair_state_head(torch.cat([h_t, z_next], dim=-1))
      future_logit = self.future_collision_head(z_next)
      stair_shape_prediction = self.stair_shape_head(z_next)

      z = z_next
      gate = gate_next
      z_steps.append(z)
      alpha_steps.append(alpha)
      mode_steps.append(gate[:, 0:1])
      event_logits.append(event_logit)
      stair_logits.append(stair_logit)
      future_logits.append(future_logit)
      stair_shape_predictions.append(stair_shape_prediction)

    z_seq = torch.stack(z_steps, dim=0)
    alpha_seq = torch.stack(alpha_steps, dim=0)
    mode_seq = torch.stack(mode_steps, dim=0)
    self._aux_event_logits = torch.stack(event_logits, dim=0)
    self._aux_stair_logits = torch.stack(stair_logits, dim=0)
    self._aux_future_collision_logits = torch.stack(future_logits, dim=0)
    self._aux_stair_shape_predictions = torch.stack(stair_shape_predictions, dim=0)

    if masks is not None:
      z_seq = cast(torch.Tensor, unpad_trajectories(z_seq, masks))
      alpha_seq = cast(torch.Tensor, unpad_trajectories(alpha_seq, masks))
      mode_seq = cast(torch.Tensor, unpad_trajectories(mode_seq, masks))
      actor_obs = cast(torch.Tensor, unpad_trajectories(actor_obs, masks))
      self._aux_event_logits = cast(
        torch.Tensor, unpad_trajectories(self._aux_event_logits, masks)
      )
      self._aux_stair_logits = cast(
        torch.Tensor, unpad_trajectories(self._aux_stair_logits, masks)
      )
      self._aux_future_collision_logits = cast(
        torch.Tensor,
        unpad_trajectories(self._aux_future_collision_logits, masks),
      )
      self._aux_stair_shape_predictions = cast(
        torch.Tensor,
        unpad_trajectories(self._aux_stair_shape_predictions, masks),
      )
    elif not sequence_mode:
      z_seq = z_seq.squeeze(0)
      alpha_seq = alpha_seq.squeeze(0)
      mode_seq = mode_seq.squeeze(0)
      actor_obs = actor_obs.squeeze(0)
      self._aux_event_logits = self._aux_event_logits.squeeze(0)
      self._aux_stair_logits = self._aux_stair_logits.squeeze(0)
      self._aux_future_collision_logits = self._aux_future_collision_logits.squeeze(0)
      self._aux_stair_shape_predictions = self._aux_stair_shape_predictions.squeeze(0)

    self._update_slow_latent_diagnostics(z_seq, alpha_seq, mode_seq)

    if hidden_state is None:
      self._hidden_state = (h_out.detach(), c_out.detach())
      self._z_memory = z.detach()
      self._gate_state = gate.detach()

    return actor_obs, z_seq, h_out, c_out

  def _update_slow_latent_diagnostics(
    self,
    z_seq: torch.Tensor,
    alpha_seq: torch.Tensor,
    mode_seq: torch.Tensor,
  ) -> None:
    if (
      self._aux_event_logits is None
      or self._aux_stair_logits is None
      or self._aux_future_collision_logits is None
      or self._aux_stair_shape_predictions is None
    ):
      self._slow_latent_diagnostics = {}
      return
    self._slow_latent_diagnostics = {
      "event_prob": torch.sigmoid(self._aux_event_logits.detach()),
      "stair_prob": torch.sigmoid(self._aux_stair_logits.detach()),
      "future_prob": torch.sigmoid(self._aux_future_collision_logits.detach()),
      "stair_shape": self._aux_stair_shape_predictions.detach(),
      "z_norm": z_seq.detach().norm(dim=-1, keepdim=True),
      "gate_mode": mode_seq.detach(),
      "alpha": alpha_seq.detach(),
    }

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: GatedHiddenState = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    actor_obs = self._cat_groups(obs, self._actor_obs_group_names)
    latent_obs = self._cat_groups(obs, self._latent_obs_group_names)
    actor_obs_norm = self.obs_normalizer(actor_obs)

    actor_obs_norm, z_memory, _h_out, _c_out = self._run_latent_path(
      actor_obs_norm, latent_obs, masks, hidden_state
    )
    actor_input = torch.cat([actor_obs_norm, z_memory], dim=-1)
    mlp_output = self.mlp(actor_input)
    if self.distribution is not None:
      if stochastic_output:
        self.distribution.update(mlp_output)
        return self.distribution.sample()
      return self.distribution.deterministic_output(mlp_output)
    return mlp_output

  def update_normalization(self, obs: TensorDict) -> None:
    if not self.obs_normalization:
      return
    actor_obs = self._cat_groups(obs, self._actor_obs_group_names)
    latent_obs = self._cat_groups(obs, self._latent_obs_group_names)
    self.obs_normalizer.update(actor_obs)  # type: ignore[attr-defined]
    self.latent_obs_normalizer.update(latent_obs)  # type: ignore[attr-defined]

  def reset(
    self,
    dones: torch.Tensor | None = None,
    hidden_state: GatedHiddenState = None,
  ) -> None:
    del hidden_state
    if dones is None:
      self._hidden_state = None
      self._z_memory = None
      self._gate_state = None
      self._slow_latent_diagnostics = {}
      return
    if self._hidden_state is None or self._z_memory is None or self._gate_state is None:
      return
    done_mask = dones.reshape(-1).bool()
    if done_mask.numel() == 0 or not bool(done_mask.any()):
      return
    h, c = self._hidden_state
    h[:, done_mask, :] = 0.0
    c[:, done_mask, :] = 0.0
    self._z_memory[done_mask] = 0.0
    self._gate_state[done_mask] = 0.0

  def reset_slow_latent(self) -> None:
    if self._z_memory is not None:
      self._z_memory.zero_()
    if self._gate_state is not None:
      self._gate_state.zero_()

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    if self._hidden_state is not None:
      self._hidden_state = tuple(state.detach() for state in self._hidden_state)  # type: ignore[assignment]
    if self._z_memory is not None:
      self._z_memory = self._z_memory.detach()
    if self._gate_state is not None:
      self._gate_state = self._gate_state.detach()
    if dones is not None:
      self.reset(dones)

  def get_hidden_state(self) -> HiddenState:
    if self._hidden_state is None or self._z_memory is None or self._gate_state is None:
      return None
    h, c = self._hidden_state
    return cast(
      HiddenState, self._pack_hidden_state(h, c, self._z_memory, self._gate_state)
    )

  def get_aux_outputs(self) -> dict[str, torch.Tensor]:
    outputs: dict[str, torch.Tensor] = {}
    if self._aux_event_logits is not None:
      outputs["event_logit"] = self._aux_event_logits
    if self._aux_stair_logits is not None:
      outputs["stair_logit"] = self._aux_stair_logits
    if self._aux_future_collision_logits is not None:
      outputs["future_collision_logit"] = self._aux_future_collision_logits
    if self._aux_stair_shape_predictions is not None:
      outputs["stair_shape"] = self._aux_stair_shape_predictions
    return outputs

  def get_slow_latent_diagnostics(self) -> dict[str, torch.Tensor]:
    """Return rollout/update diagnostics for logging without exposing labels."""
    return dict(self._slow_latent_diagnostics)

  @property
  def aux_event_logits(self) -> torch.Tensor | None:
    return self._aux_event_logits

  @property
  def aux_stair_logits(self) -> torch.Tensor | None:
    return self._aux_stair_logits

  @property
  def aux_future_collision_logits(self) -> torch.Tensor | None:
    return self._aux_future_collision_logits

  @property
  def aux_stair_shape_predictions(self) -> torch.Tensor | None:
    return self._aux_stair_shape_predictions

  def as_onnx(self, verbose: bool = False) -> _OnnxStairLatentModel:
    return _OnnxStairLatentModel(self, verbose)


class _OnnxStairLatentModel(nn.Module):
  """Single-step ONNX wrapper for the gated stair latent policy."""

  is_recurrent: bool = True

  def __init__(self, model: LSTMSlowLatentMLPModel, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    self.actor_obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.latent_obs_normalizer = copy.deepcopy(model.latent_obs_normalizer)
    self.latent_encoder = copy.deepcopy(model.latent_encoder)
    self.latent_lstm = copy.deepcopy(model.latent_lstm)
    self.z_candidate_head = copy.deepcopy(model.z_candidate_head)
    self.event_head = copy.deepcopy(model.event_head)
    self.stair_state_head = copy.deepcopy(model.stair_state_head)
    self.future_collision_head = copy.deepcopy(model.future_collision_head)
    self.mlp = copy.deepcopy(model.mlp)
    self.deterministic_output = (
      model.distribution.as_deterministic_output_module()
      if model.distribution is not None
      else nn.Identity()
    )

    self.actor_obs_dim = model.obs_dim
    self.latent_obs_dim = model.latent_obs_dim
    self.z_dim = model.z_dim
    self.latent_hidden_dim = model.latent_hidden_dim
    self.alpha_fast = model.alpha_fast
    self.alpha_write = model.alpha_write
    self.alpha_hold = model.alpha_hold
    self.write_steps = model.write_steps
    self.min_stair_steps = model.min_stair_steps
    self.exit_steps = model.exit_steps
    self.cooldown_steps = model.cooldown_steps
    self.event_on_threshold = model.event_on_threshold
    self.event_off_threshold = model.event_off_threshold
    self.stair_off_threshold = model.stair_off_threshold

  @property
  def input_names(self) -> list[str]:
    return ["actor_obs", "latent_obs", "h_in", "c_in", "z_in", "gate_state_in"]

  @property
  def output_names(self) -> list[str]:
    return [
      "actions",
      "h_out",
      "c_out",
      "z_out",
      "gate_state_out",
      "event_prob",
      "stair_prob",
      "future_collision_prob",
    ]

  def _advance_gate_state(
    self,
    event_prob: torch.Tensor,
    stair_prob: torch.Tensor,
    gate_state: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    mode = gate_state[:, 0]
    stair_timer = gate_state[:, 1]
    no_event_timer = gate_state[:, 2]
    write_timer = gate_state[:, 3]
    cooldown = torch.clamp(gate_state[:, 4] - 1.0, min=0.0)

    trigger = (
      (mode == _MODE_NORMAL)
      & (cooldown <= 0.0)
      & (event_prob.squeeze(-1) > self.event_on_threshold)
    )
    mode = torch.where(trigger, torch.full_like(mode, _MODE_STAIR_WRITE), mode)
    write_timer = torch.where(trigger, torch.zeros_like(write_timer), write_timer)
    stair_timer = torch.where(trigger, torch.zeros_like(stair_timer), stair_timer)
    no_event_timer = torch.where(
      trigger, torch.zeros_like(no_event_timer), no_event_timer
    )

    in_write = mode == _MODE_STAIR_WRITE
    write_timer = torch.where(in_write, write_timer + 1.0, write_timer)
    stair_timer = torch.where(in_write, stair_timer + 1.0, stair_timer)
    mode = torch.where(
      in_write & (write_timer >= self.write_steps),
      torch.full_like(mode, _MODE_STAIR_MEMORY),
      mode,
    )

    in_memory = mode == _MODE_STAIR_MEMORY
    stair_timer = torch.where(in_memory, stair_timer + 1.0, stair_timer)
    no_event = event_prob.squeeze(-1) < self.event_off_threshold
    no_event_timer = torch.where(
      in_memory & no_event,
      no_event_timer + 1.0,
      torch.where(in_memory, torch.zeros_like(no_event_timer), no_event_timer),
    )
    exit_memory = (
      in_memory
      & (stair_prob.squeeze(-1) < self.stair_off_threshold)
      & (no_event_timer > self.exit_steps)
      & (stair_timer > self.min_stair_steps)
    )
    mode = torch.where(exit_memory, torch.full_like(mode, _MODE_NORMAL), mode)
    cooldown = torch.where(
      exit_memory, torch.full_like(cooldown, self.cooldown_steps), cooldown
    )

    next_gate = torch.stack(
      [mode, stair_timer, no_event_timer, write_timer, cooldown], dim=-1
    )
    alpha = torch.full_like(event_prob, self.alpha_fast)
    alpha = torch.where(mode[:, None] == _MODE_STAIR_WRITE, self.alpha_write, alpha)
    alpha = torch.where(mode[:, None] == _MODE_STAIR_MEMORY, self.alpha_hold, alpha)
    return next_gate, alpha

  def forward(
    self,
    actor_obs: torch.Tensor,
    latent_obs: torch.Tensor,
    h_in: torch.Tensor,
    c_in: torch.Tensor,
    z_in: torch.Tensor,
    gate_state_in: torch.Tensor,
  ) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
  ]:
    actor_obs_norm = self.actor_obs_normalizer(actor_obs)
    latent_obs_norm = self.latent_obs_normalizer(latent_obs)
    encoded = self.latent_encoder(latent_obs_norm).unsqueeze(0)
    lstm_out, (h_out, c_out) = self.latent_lstm(encoded, (h_in, c_in))
    h_t = lstm_out.squeeze(0)
    z_candidate = self.z_candidate_head(h_t)
    event_prob = torch.sigmoid(self.event_head(h_t))
    stair_gate_prob = torch.sigmoid(
      self.stair_state_head(torch.cat([h_t, z_in], dim=-1))
    )
    gate_state_out, alpha = self._advance_gate_state(
      event_prob, stair_gate_prob, gate_state_in
    )
    z_out = (1.0 - alpha) * z_in + alpha * z_candidate
    stair_prob = torch.sigmoid(self.stair_state_head(torch.cat([h_t, z_out], dim=-1)))
    future_collision_prob = torch.sigmoid(self.future_collision_head(z_out))
    actor_input = torch.cat([actor_obs_norm, z_out], dim=-1)
    actions = self.deterministic_output(self.mlp(actor_input))
    return (
      actions,
      h_out,
      c_out,
      z_out,
      gate_state_out,
      event_prob,
      stair_prob,
      future_collision_prob,
    )

  def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
    return (
      torch.zeros(1, self.actor_obs_dim),
      torch.zeros(1, self.latent_obs_dim),
      torch.zeros(1, 1, self.latent_hidden_dim),
      torch.zeros(1, 1, self.latent_hidden_dim),
      torch.zeros(1, self.z_dim),
      torch.zeros(1, _GATE_STATE_DIM),
    )
