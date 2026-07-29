"""Deployable stair-focused gated slow-latent actor model."""

from __future__ import annotations

import copy
import math
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
_TREAD_DEPTH_MIN_M = 0.23
_TREAD_DEPTH_MAX_M = 0.37
_RISER_HEIGHT_MIN_M = 0.088
_RISER_HEIGHT_MAX_M = 0.25
_SEMANTIC_STATE_DIM = 8
_SEMANTIC_SHAPE_DIM = 8
_SEMANTIC_DIM = _SEMANTIC_STATE_DIM + _SEMANTIC_SHAPE_DIM
_LEGACY_SAFE_STRIDE_WIDTH_LOGIT = -6.0
_DYNAMIC_SAFE_STRIDE_WIDTH_LOGIT = -1.4
_SAFE_STRIDE_CONFIDENCE_LOGIT = -2.0
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
    state_latent_dim: int | None = None,
    alpha_fast: float = 0.3,
    alpha_write: float = 0.8,
    alpha_hold_state: float = 0.0,
    alpha_hold_shape: float = 0.05,
    memory_event_shape_boost_steps: int = 15,
    alpha_hold: float | None = None,
    write_steps: int = 6,
    stair_confirm_steps: int = 3,
    min_stair_steps: int = 30,
    exit_steps: int = 40,
    stair_memory_exit_on_stair_off: bool = True,
    cooldown_steps: int = 15,
    event_on_threshold: float = 0.60,
    event_off_threshold: float = 0.20,
    stair_on_threshold: float = 0.35,
    stair_off_threshold: float = 0.20,
    aux_future_collision_risk_coef: float = 0.03,
    aux_future_safe_landing_quality_coef: float = 0.03,
    aux_event_coef: float = 0.03,
    aux_event_pos_weight: float = 50.0,
    event_label_window_steps: int = 4,
    aux_stair_coef: float = 0.05,
    aux_stair_pos_weight: float = 3.0,
    aux_stair_shape_coef: float = 0.0,
    aux_safe_stride_coef: float = 0.0,
    stair_shape_huber_delta: float = 0.05,
    stair_shape_same_foot_loss_coef: float = 1.0,
    stair_shape_riser_loss_coef: float = 1.0,
    safe_stride_huber_delta: float = 0.05,
    safe_stride_width_loss_coef: float = 1.0,
    safe_stride_lower_shortfall_coef: float = 1.0,
    safe_stride_interval_coverage_loss_coef: float = 0.0,
    safe_stride_interval_coverage_margin: float = 0.01,
    safe_stride_confidence_loss_coef: float = 0.30,
    safe_stride_std_floor_loss_coef: float = 0.0,
    safe_stride_centered_loss_coef: float = 0.0,
    safe_stride_std_floor_ratio: float = 0.70,
    safe_stride_deployable_hint_loss_coef: float = 0.0,
    safe_stride_deployable_hint_margin: float = 0.02,
    same_foot_stride_deployable_hint_loss_coef: float = 0.0,
    same_foot_stride_deployable_hint_margin: float = 0.02,
    safe_stride_min: float = 0.10,
    safe_stride_max: float = 0.55,
    same_foot_stride_min: float = 0.10,
    same_foot_stride_max: float = 0.80,
    structured_safe_stride_enabled: bool = False,
    dynamic_stair_shape_enabled: bool = False,
    dynamic_safe_stride_enabled: bool = False,
    safe_stride_phase_dim: int = 0,
    safe_stride_phase_start: int = -1,
    shadow_semantic_enabled: bool = False,
    actor_semantic_enabled: bool = False,
    geometry_probe_input: str = "none",
    future_risk_weight_scale: float = 2.0,
    future_quality_weight_scale: float = 2.0,
    future_risk_huber_delta: float = 0.1,
    future_quality_huber_delta: float = 0.1,
    future_horizon: int = 20,
    latent_obs_set: str = "latent",
    **kwargs: Any,
  ) -> None:
    del kwargs
    self.z_dim = int(z_dim if z_dim is not None else (latent_dim or 16))
    self.state_latent_dim = int(
      state_latent_dim if state_latent_dim is not None else self.z_dim // 2
    )
    if not 0 < self.state_latent_dim < self.z_dim:
      raise ValueError(
        "state_latent_dim must split z_memory into non-empty state and shape "
        f"parts, got state_latent_dim={self.state_latent_dim}, z_dim={self.z_dim}."
      )
    self.shape_latent_dim = self.z_dim - self.state_latent_dim
    self.latent_dim = self.z_dim
    self.slow_latent_dim = self.z_dim
    self.actor_semantic_enabled = bool(actor_semantic_enabled)
    self.dynamic_stair_shape_enabled = bool(dynamic_stair_shape_enabled)
    self.zero_shape_latent_for_actor = False
    self.freeze_shape_latent_at_stair_entry = False
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
      nn.Linear(self.latent_hidden_dim + self.state_latent_dim, 64),
      activation_cls(),
      nn.Linear(64, 1),
    )
    self.future_collision_risk_head = nn.Sequential(
      nn.Linear(self.z_dim, 64),
      activation_cls(),
      nn.Linear(64, 1),
    )
    self.future_safe_landing_quality_head = nn.Sequential(
      nn.Linear(self.z_dim, 64),
      activation_cls(),
      nn.Linear(64, 1),
    )
    stair_shape_input_dim = self.shape_latent_dim + (
      self.latent_hidden_dim if self.dynamic_stair_shape_enabled else 0
    )
    self.stair_shape_head = nn.Sequential(
      nn.Linear(stair_shape_input_dim, 64),
      activation_cls(),
      nn.Linear(64, 2),
    )
    if geometry_probe_input not in {"none", "shape", "hidden", "combined"}:
      raise ValueError(
        "geometry_probe_input must be one of none, shape, hidden, combined."
      )
    self.geometry_probe_input = geometry_probe_input
    geometry_probe_input_dim = {
      "none": 0,
      "shape": self.shape_latent_dim,
      "hidden": self.latent_hidden_dim,
      "combined": self.shape_latent_dim + self.latent_hidden_dim,
    }[geometry_probe_input]
    self.geometry_probe_head: nn.Module | None = None
    if geometry_probe_input_dim > 0:
      # Keep action-sampling RNG identical across Probe input ablations.
      with torch.random.fork_rng(devices=[]):
        self.geometry_probe_head = nn.Sequential(
          nn.Linear(geometry_probe_input_dim, 64),
          activation_cls(),
          nn.Linear(64, 2),
        )
    dynamic_stride_input_dim = (
      self.shape_latent_dim + self.latent_hidden_dim + safe_stride_phase_dim
      if dynamic_safe_stride_enabled
      else self.shape_latent_dim
    )
    self.safe_stride_head = nn.Sequential(
      nn.Linear(dynamic_stride_input_dim, 64),
      activation_cls(),
      nn.Linear(
        64,
        1
        if dynamic_safe_stride_enabled
        else (2 if structured_safe_stride_enabled else 1),
      ),
    )
    self.safe_stride_width_head: nn.Module | None = None
    if dynamic_safe_stride_enabled:
      self.safe_stride_width_head = nn.Sequential(
        nn.Linear(dynamic_stride_input_dim, 64),
        activation_cls(),
        nn.Linear(64, 1),
      )
    self.safe_stride_confidence_head = nn.Sequential(
      nn.Linear(dynamic_stride_input_dim, 64),
      activation_cls(),
      nn.Linear(64, 1),
    )

    self.alpha_fast = float(alpha_fast)
    self.alpha_write = float(alpha_write)
    if alpha_hold is not None:
      alpha_hold_state = alpha_hold
      alpha_hold_shape = alpha_hold
    self.alpha_hold_state = float(alpha_hold_state)
    self.alpha_hold_shape = float(alpha_hold_shape)
    self.alpha_hold = self.alpha_hold_state
    alpha_rates = {
      "alpha_fast": self.alpha_fast,
      "alpha_write": self.alpha_write,
      "alpha_hold_state": self.alpha_hold_state,
      "alpha_hold_shape": self.alpha_hold_shape,
    }
    for name, rate in alpha_rates.items():
      if not 0.0 <= rate <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {rate}.")
    if self.alpha_hold_shape < self.alpha_hold_state:
      raise ValueError(
        "alpha_hold_shape must be at least alpha_hold_state so geometry can "
        "keep updating while stair state remains stable."
      )
    self.memory_event_shape_boost_steps = float(memory_event_shape_boost_steps)
    if self.memory_event_shape_boost_steps < 1.0:
      raise ValueError("memory_event_shape_boost_steps must be at least 1.")
    self.write_steps = float(write_steps)
    self.stair_confirm_steps = float(stair_confirm_steps)
    self.min_stair_steps = float(min_stair_steps)
    self.exit_steps = float(exit_steps)
    self.stair_memory_exit_on_stair_off = bool(stair_memory_exit_on_stair_off)
    self.cooldown_steps = float(cooldown_steps)
    if self.write_steps < 1.0:
      raise ValueError("write_steps must be at least 1.")
    if not 1.0 <= self.stair_confirm_steps <= self.write_steps:
      raise ValueError("stair_confirm_steps must be in [1, write_steps].")
    if self.min_stair_steps < 0.0 or self.exit_steps < 1.0:
      raise ValueError("min_stair_steps must be non-negative and exit_steps positive.")
    self.event_on_threshold = float(event_on_threshold)
    self.event_off_threshold = float(event_off_threshold)
    self.stair_on_threshold = float(stair_on_threshold)
    self.stair_off_threshold = float(stair_off_threshold)
    if self.event_on_threshold < self.event_off_threshold:
      raise ValueError("event_on_threshold must be >= event_off_threshold.")
    if self.stair_on_threshold < self.stair_off_threshold:
      raise ValueError("stair_on_threshold must be >= stair_off_threshold.")

    self.aux_future_collision_risk_coef = float(aux_future_collision_risk_coef)
    self.aux_future_safe_landing_quality_coef = float(
      aux_future_safe_landing_quality_coef
    )
    self.aux_event_coef = float(aux_event_coef)
    self.aux_event_pos_weight = float(aux_event_pos_weight)
    self.event_label_window_steps = int(event_label_window_steps)
    if self.aux_event_pos_weight <= 0.0:
      raise ValueError("aux_event_pos_weight must be positive.")
    if self.event_label_window_steps < 1:
      raise ValueError("event_label_window_steps must be at least 1.")
    self.aux_stair_coef = float(aux_stair_coef)
    self.aux_stair_pos_weight = float(aux_stair_pos_weight)
    if self.aux_stair_pos_weight <= 0.0:
      raise ValueError("aux_stair_pos_weight must be positive.")
    self.aux_stair_shape_coef = float(aux_stair_shape_coef)
    self.aux_safe_stride_coef = float(aux_safe_stride_coef)
    self.stair_shape_huber_delta = float(stair_shape_huber_delta)
    self.stair_shape_same_foot_loss_coef = float(stair_shape_same_foot_loss_coef)
    self.stair_shape_riser_loss_coef = float(stair_shape_riser_loss_coef)
    self.safe_stride_huber_delta = float(safe_stride_huber_delta)
    self.safe_stride_width_loss_coef = float(safe_stride_width_loss_coef)
    self.safe_stride_lower_shortfall_coef = float(safe_stride_lower_shortfall_coef)
    self.safe_stride_interval_coverage_loss_coef = float(
      safe_stride_interval_coverage_loss_coef
    )
    self.safe_stride_interval_coverage_margin = float(
      safe_stride_interval_coverage_margin
    )
    self.safe_stride_confidence_loss_coef = float(safe_stride_confidence_loss_coef)
    self.safe_stride_std_floor_loss_coef = float(safe_stride_std_floor_loss_coef)
    self.safe_stride_centered_loss_coef = float(safe_stride_centered_loss_coef)
    self.safe_stride_std_floor_ratio = float(safe_stride_std_floor_ratio)
    self.safe_stride_deployable_hint_loss_coef = float(
      safe_stride_deployable_hint_loss_coef
    )
    self.safe_stride_deployable_hint_margin = float(safe_stride_deployable_hint_margin)
    self.same_foot_stride_deployable_hint_loss_coef = float(
      same_foot_stride_deployable_hint_loss_coef
    )
    self.same_foot_stride_deployable_hint_margin = float(
      same_foot_stride_deployable_hint_margin
    )
    if self.stair_shape_same_foot_loss_coef < 0.0:
      raise ValueError("stair_shape_same_foot_loss_coef must be non-negative.")
    if self.stair_shape_riser_loss_coef < 0.0:
      raise ValueError("stair_shape_riser_loss_coef must be non-negative.")
    if self.safe_stride_width_loss_coef < 0.0:
      raise ValueError("safe_stride_width_loss_coef must be non-negative.")
    if self.safe_stride_lower_shortfall_coef < 1.0:
      raise ValueError("safe_stride_lower_shortfall_coef must be at least 1.")
    if self.safe_stride_interval_coverage_loss_coef < 0.0:
      raise ValueError("safe_stride_interval_coverage_loss_coef must be non-negative.")
    if self.safe_stride_interval_coverage_margin < 0.0:
      raise ValueError("safe_stride_interval_coverage_margin must be non-negative.")
    if self.safe_stride_confidence_loss_coef < 0.0:
      raise ValueError("safe_stride_confidence_loss_coef must be non-negative.")
    if self.safe_stride_std_floor_loss_coef < 0.0:
      raise ValueError("safe_stride_std_floor_loss_coef must be non-negative.")
    if self.safe_stride_centered_loss_coef < 0.0:
      raise ValueError("safe_stride_centered_loss_coef must be non-negative.")
    if not 0.0 <= self.safe_stride_std_floor_ratio <= 2.0:
      raise ValueError("safe_stride_std_floor_ratio must be in [0, 2].")
    if self.safe_stride_deployable_hint_loss_coef < 0.0:
      raise ValueError("safe_stride_deployable_hint_loss_coef must be non-negative.")
    if self.safe_stride_deployable_hint_margin < 0.0:
      raise ValueError("safe_stride_deployable_hint_margin must be non-negative.")
    if self.same_foot_stride_deployable_hint_loss_coef < 0.0:
      raise ValueError(
        "same_foot_stride_deployable_hint_loss_coef must be non-negative."
      )
    if self.same_foot_stride_deployable_hint_margin < 0.0:
      raise ValueError("same_foot_stride_deployable_hint_margin must be non-negative.")
    self.tread_depth_min = _TREAD_DEPTH_MIN_M
    self.tread_depth_max = _TREAD_DEPTH_MAX_M
    self.riser_height_min = _RISER_HEIGHT_MIN_M
    self.riser_height_max = _RISER_HEIGHT_MAX_M
    self.safe_stride_min = float(safe_stride_min)
    self.safe_stride_max = float(safe_stride_max)
    self.same_foot_stride_min = float(same_foot_stride_min)
    self.same_foot_stride_max = float(same_foot_stride_max)
    self.structured_safe_stride_enabled = bool(structured_safe_stride_enabled)
    self.dynamic_safe_stride_enabled = bool(dynamic_safe_stride_enabled)
    self.safe_stride_phase_dim = int(safe_stride_phase_dim)
    if self.dynamic_safe_stride_enabled and self.safe_stride_phase_dim <= 0:
      raise ValueError(
        "dynamic_safe_stride_enabled requires safe_stride_phase_dim > 0."
      )
    if self.safe_stride_phase_dim > self.latent_obs_dim:
      raise ValueError("safe_stride_phase_dim cannot exceed latent_obs_dim.")
    self.safe_stride_phase_start = int(safe_stride_phase_start)
    if self.dynamic_safe_stride_enabled:
      if self.safe_stride_phase_start < 0:
        self.safe_stride_phase_start = self.latent_obs_dim - self.safe_stride_phase_dim
      phase_end = self.safe_stride_phase_start + self.safe_stride_phase_dim
      if self.safe_stride_phase_start < 0 or phase_end > self.latent_obs_dim:
        raise ValueError(
          "safe_stride_phase_start + safe_stride_phase_dim must fit latent_obs_dim."
        )
    self.shadow_semantic_enabled = bool(shadow_semantic_enabled)
    bounds = {
      "tread_depth": (self.tread_depth_min, self.tread_depth_max),
      "same_foot_stride": (
        self.same_foot_stride_min,
        self.same_foot_stride_max,
      ),
      "riser_height": (self.riser_height_min, self.riser_height_max),
      "safe_stride": (self.safe_stride_min, self.safe_stride_max),
    }
    for name, (minimum, maximum) in bounds.items():
      if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError(f"{name} bounds must be finite, got [{minimum}, {maximum}].")
      if maximum <= minimum:
        raise ValueError(
          f"{name}_max must be greater than {name}_min, got [{minimum}, {maximum}]."
        )
    self.future_risk_weight_scale = float(future_risk_weight_scale)
    self.future_quality_weight_scale = float(future_quality_weight_scale)
    self.future_risk_huber_delta = float(future_risk_huber_delta)
    self.future_quality_huber_delta = float(future_quality_huber_delta)
    self.future_horizon = int(future_horizon)
    if self.future_risk_weight_scale < 0.0 or self.future_quality_weight_scale < 0.0:
      raise ValueError("Future-label weight scales must be non-negative.")
    if self.future_risk_huber_delta <= 0.0 or self.future_quality_huber_delta <= 0.0:
      raise ValueError("Future-label Huber deltas must be positive.")
    if self.future_horizon <= 0:
      raise ValueError("future_horizon must be positive.")

    self._hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None
    self._z_memory: torch.Tensor | None = None
    self._gate_state: torch.Tensor | None = None
    self._pre_stair_shape_actor_memory: torch.Tensor | None = None
    self._pre_stair_shape_actor_memory_valid: torch.Tensor | None = None
    self._aux_event_logits: torch.Tensor | None = None
    self._aux_stair_logits: torch.Tensor | None = None
    self._aux_future_collision_risk_logits: torch.Tensor | None = None
    self._aux_future_safe_landing_quality_logits: torch.Tensor | None = None
    self._aux_stair_shape_predictions: torch.Tensor | None = None
    self._aux_geometry_probe_predictions: torch.Tensor | None = None
    self._aux_safe_stride_predictions: torch.Tensor | None = None
    self._aux_safe_stride_intervals: torch.Tensor | None = None
    self._aux_safe_stride_confidence_logits: torch.Tensor | None = None
    self._shadow_semantic: torch.Tensor | None = None
    self._slow_latent_diagnostics: dict[str, torch.Tensor] = {}

  def _get_latent_dim(self) -> int:
    semantic_dim = _SEMANTIC_DIM if self.actor_semantic_enabled else 0
    return self.obs_dim + self.z_dim + semantic_dim

  def _cat_groups(self, obs: TensorDict, group_names: list[str]) -> torch.Tensor:
    tensors = [cast(torch.Tensor, obs[name]) for name in group_names]
    return torch.cat(tensors, dim=-1)

  def _stair_logit(self, h_t: torch.Tensor) -> torch.Tensor:
    """Predict stair continuation without feeding the held state back in."""
    memory_placeholder = h_t.new_zeros((*h_t.shape[:-1], self.state_latent_dim))
    return self.stair_state_head(torch.cat([h_t, memory_placeholder], dim=-1))

  def _shape_memory(self, z: torch.Tensor) -> torch.Tensor:
    return z[..., self.state_latent_dim :]

  def _actor_memory(self, z: torch.Tensor) -> torch.Tensor:
    """Return actor conditioning with an optional play-only shape ablation."""
    if self.zero_shape_latent_for_actor:
      return torch.cat(
        [
          z[..., : self.state_latent_dim],
          torch.zeros_like(z[..., self.state_latent_dim :]),
        ],
        dim=-1,
      )
    if not self.freeze_shape_latent_at_stair_entry:
      return z
    if z.dim() != 2 or self._gate_state is None:
      raise RuntimeError(
        "Stair-entry shape freezing is supported only for online inference."
      )

    shape_memory = self._shape_memory(z)
    if (
      self._pre_stair_shape_actor_memory is None
      or self._pre_stair_shape_actor_memory.shape != shape_memory.shape
    ):
      self._pre_stair_shape_actor_memory = shape_memory.detach().clone()
      self._pre_stair_shape_actor_memory_valid = torch.zeros(
        shape_memory.shape[0],
        device=shape_memory.device,
        dtype=torch.bool,
      )
    assert self._pre_stair_shape_actor_memory_valid is not None

    normal_mode = self._gate_state[:, 0] == _MODE_NORMAL
    self._pre_stair_shape_actor_memory.copy_(
      torch.where(
        normal_mode[:, None],
        shape_memory.detach(),
        self._pre_stair_shape_actor_memory,
      )
    )
    self._pre_stair_shape_actor_memory_valid.logical_or_(normal_mode)
    use_snapshot = (~normal_mode) & self._pre_stair_shape_actor_memory_valid
    actor_shape_memory = torch.where(
      use_snapshot[:, None],
      self._pre_stair_shape_actor_memory,
      shape_memory,
    )
    return torch.cat(
      [
        z[..., : self.state_latent_dim],
        actor_shape_memory,
      ],
      dim=-1,
    )

  def _decode_stair_shape(
    self,
    shape_memory: torch.Tensor,
    h_t: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if self.dynamic_stair_shape_enabled:
      if h_t is None:
        raise RuntimeError("Dynamic stair-shape decoding requires LSTM state.")
      features = torch.cat([shape_memory, h_t], dim=-1)
    else:
      features = shape_memory
    shape01 = torch.sigmoid(self.stair_shape_head(features))
    return self._denormalize_stair_shape(shape01)

  def _denormalize_stair_shape(self, shape01: torch.Tensor) -> torch.Tensor:
    """Map normalized same-foot stride/height predictions to physical units."""
    same_foot_stride = self.same_foot_stride_min + shape01[..., 0:1] * (
      self.same_foot_stride_max - self.same_foot_stride_min
    )
    riser_height = self.riser_height_min + shape01[..., 1:2] * (
      self.riser_height_max - self.riser_height_min
    )
    return torch.cat([same_foot_stride, riser_height], dim=-1)

  def _decode_geometry_probe(
    self,
    shape_memory: torch.Tensor,
    h_t: torch.Tensor,
  ) -> torch.Tensor | None:
    """Decode actionable same-foot stride and riser height from frozen features."""
    if self.geometry_probe_head is None:
      return None
    if self.geometry_probe_input == "shape":
      features = shape_memory
    elif self.geometry_probe_input == "hidden":
      features = h_t
    else:
      features = torch.cat([shape_memory, h_t], dim=-1)
    return self._denormalize_stair_shape(
      torch.sigmoid(self.geometry_probe_head(features))
    )

  def _decode_safe_stride_interval(
    self,
    safe_stride_features: torch.Tensor,
    shape_memory: torch.Tensor | None = None,
  ) -> torch.Tensor:
    stride_raw = self.safe_stride_head(safe_stride_features)
    lower01 = torch.sigmoid(stride_raw[..., 0:1])
    lower = self.safe_stride_min + lower01 * (
      self.safe_stride_max - self.safe_stride_min
    )
    if not self.structured_safe_stride_enabled:
      return torch.cat([lower, lower], dim=-1)
    if self.dynamic_safe_stride_enabled:
      if self.safe_stride_width_head is None:
        raise RuntimeError("Dynamic SafeStride decoding requires a width head.")
      width_logit = self.safe_stride_width_head(safe_stride_features)
      width = torch.sigmoid(width_logit) * (self.safe_stride_max - lower)
    else:
      width_logit = stride_raw[..., 1:2]
      width = torch.sigmoid(width_logit) * (self.safe_stride_max - lower)
    upper = lower + width
    return torch.cat([lower, upper], dim=-1)

  def _decode_safe_stride_confidence_logit(
    self,
    safe_stride_features: torch.Tensor,
  ) -> torch.Tensor:
    return self.safe_stride_confidence_head(safe_stride_features)

  def _decode_safe_stride(self, shape_memory: torch.Tensor) -> torch.Tensor:
    interval = self._decode_safe_stride_interval(shape_memory)
    return 0.5 * (interval[..., 0:1] + interval[..., 1:2])

  def _safe_stride_features(
    self,
    shape_memory: torch.Tensor,
    h_t: torch.Tensor,
    latent_obs_t: torch.Tensor,
  ) -> torch.Tensor:
    if not self.dynamic_safe_stride_enabled:
      return shape_memory
    gait_phase = self._safe_stride_gait_phase(latent_obs_t)
    return torch.cat([shape_memory, h_t, gait_phase], dim=-1)

  def _safe_stride_gait_phase(self, latent_obs_t: torch.Tensor) -> torch.Tensor:
    phase_end = self.safe_stride_phase_start + self.safe_stride_phase_dim
    return latent_obs_t[..., self.safe_stride_phase_start : phase_end]

  @staticmethod
  def _normalize_semantic_value(
    value: torch.Tensor,
    minimum: float,
    maximum: float,
  ) -> torch.Tensor:
    return torch.clamp((value - minimum) / (maximum - minimum), 0.0, 1.0)

  @staticmethod
  def _foot_event_summary_dim(latent_obs: torch.Tensor) -> int:
    latent_dim = int(latent_obs.shape[-1])
    if latent_dim < 80:
      return 0
    new_foot_only = (latent_dim - 80) % 33 == 0
    new_with_stair = latent_dim >= 173 and (latent_dim - 173) % 33 == 0
    return 80 if latent_dim == 80 or new_foot_only or new_with_stair else 0

  def _same_foot_actor_stride_target(
    self,
    stair_shape: torch.Tensor,
    latent_obs: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Fuse the stride head with deployable ratchet evidence for actor input."""
    target = stair_shape[..., 0:1]
    if latent_obs is None:
      return target
    summary_dim = self._foot_event_summary_dim(latent_obs)
    if summary_dim < 80:
      return target
    summary = torch.nan_to_num(latent_obs[..., -summary_dim:])
    ratchet = summary[..., 70:80]
    active = ratchet[..., 0:1] > 0.5
    lower = ratchet[..., 5:6].clamp(
      self.same_foot_stride_min,
      self.same_foot_stride_max,
    )
    probe = ratchet[..., 2:3].clamp(
      self.same_foot_stride_min,
      self.same_foot_stride_max,
    )
    upper_raw = ratchet[..., 7:8]
    confirmed = ratchet[..., 6:7] > 0.5
    soft_cap = active & ~confirmed & (upper_raw > 0.0)
    open_base = torch.maximum(probe, lower)
    guard_limited = probe <= lower + 1.0e-5
    open_ceiling = torch.where(guard_limited, probe, open_base + 0.05)
    open_target = torch.maximum(open_base, torch.minimum(target, open_ceiling))
    closed_target = probe
    soft_target = probe
    ratchet_target = torch.where(
      confirmed,
      closed_target,
      torch.where(soft_cap, soft_target, open_target),
    )
    return torch.where(active, ratchet_target, target)

  def _build_shadow_semantic(
    self,
    event_prob: torch.Tensor,
    stair_prob: torch.Tensor,
    gate_state: torch.Tensor,
    stair_shape: torch.Tensor,
    safe_stride_interval: torch.Tensor,
    safe_stride_confidence: torch.Tensor | None = None,
    latent_obs: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Build a fixed-position semantic vector without feeding it to the actor."""
    mode = gate_state[..., 0:1]
    write_timer = gate_state[..., 3:4]
    cooldown = gate_state[..., 4:5]
    event_on = (event_prob > self.event_on_threshold).to(event_prob.dtype)
    stair_on = (stair_prob > self.stair_on_threshold).to(stair_prob.dtype)
    mode_normal = (mode == _MODE_NORMAL).to(event_prob.dtype)
    mode_write = (mode == _MODE_STAIR_WRITE).to(event_prob.dtype)
    mode_memory = (mode == _MODE_STAIR_MEMORY).to(event_prob.dtype)
    write_progress = mode_write * torch.clamp(
      write_timer / max(self.write_steps, 1.0),
      0.0,
      1.0,
    )
    memory_age = mode_memory * torch.clamp(
      gate_state[..., 1:2] / max(self.min_stair_steps, 1.0),
      0.0,
      1.0,
    )
    release_active = (mode == _MODE_NORMAL) & (write_timer < 0.0) & (cooldown > 0.0)
    release_progress = release_active.to(event_prob.dtype) * torch.clamp(
      (self.cooldown_steps - cooldown) / max(self.cooldown_steps, 1.0),
      0.0,
      1.0,
    )
    state_semantic = torch.cat(
      [
        event_on,
        stair_on,
        mode_normal,
        mode_write,
        mode_memory,
        write_progress,
        memory_age,
        release_progress,
      ],
      dim=-1,
    )

    same_foot_stride = stair_shape[..., 0:1]
    riser_height = stair_shape[..., 1:2]
    same_foot_stride_norm = self._normalize_semantic_value(
      same_foot_stride,
      self.same_foot_stride_min,
      self.same_foot_stride_max,
    )
    riser_height_norm = self._normalize_semantic_value(
      riser_height,
      self.riser_height_min,
      self.riser_height_max,
    )
    stride_lower = safe_stride_interval[..., 0:1]
    stride_upper = safe_stride_interval[..., 1:2]
    stride_center = 0.5 * (stride_lower + stride_upper)
    stride_width = stride_upper - stride_lower
    stride_lower_norm = self._normalize_semantic_value(
      stride_lower,
      self.safe_stride_min,
      self.safe_stride_max,
    )
    stride_upper_norm = self._normalize_semantic_value(
      stride_upper,
      self.safe_stride_min,
      self.safe_stride_max,
    )
    stride_center_norm = self._normalize_semantic_value(
      stride_center,
      self.safe_stride_min,
      self.safe_stride_max,
    )
    stride_width_norm = torch.clamp(
      stride_width / (self.safe_stride_max - self.safe_stride_min),
      0.0,
      1.0,
    )
    confidence = (
      torch.zeros_like(stride_center_norm)
      if safe_stride_confidence is None
      else safe_stride_confidence
    )
    confidence = torch.clamp(confidence, 0.0, 1.0)
    same_foot_actor_target_norm = self._normalize_semantic_value(
      self._same_foot_actor_stride_target(stair_shape, latent_obs),
      self.same_foot_stride_min,
      self.same_foot_stride_max,
    )
    shape_semantic = torch.cat(
      [
        same_foot_stride_norm,
        riser_height_norm,
        stride_lower_norm,
        stride_upper_norm,
        stride_center_norm,
        stride_width_norm,
        same_foot_actor_target_norm,
        confidence,
      ],
      dim=-1,
    )
    semantic = torch.cat([state_semantic, shape_semantic], dim=-1)
    if semantic.shape[-1] != _SEMANTIC_DIM:
      raise RuntimeError(
        f"Shadow semantic vector must have {_SEMANTIC_DIM} channels, "
        f"got {semantic.shape[-1]}."
      )
    return semantic

  def _load_from_state_dict(
    self,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    local_metadata: dict[str, Any],
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
  ) -> None:
    """Upgrade the legacy one-output SafeStride head during strict loading."""

    def resize_linear_input(key: str, current_weight: torch.Tensor) -> None:
      legacy_weight = state_dict.get(key)
      if (
        legacy_weight is None
        or legacy_weight.shape[0] != current_weight.shape[0]
        or legacy_weight.shape[1] == current_weight.shape[1]
      ):
        return
      migrated_weight = current_weight.detach().clone()
      migrated_weight.zero_()
      shared = min(legacy_weight.shape[1], current_weight.shape[1])
      migrated_weight[:, :shared] = legacy_weight[:, :shared]
      state_dict[key] = migrated_weight

    weight_key = f"{prefix}safe_stride_head.2.weight"
    bias_key = f"{prefix}safe_stride_head.2.bias"
    output_layer = cast(nn.Linear, self.safe_stride_head[2])
    current_weight = output_layer.weight
    current_bias = output_layer.bias
    assert current_bias is not None
    legacy_weight = state_dict.get(weight_key)
    legacy_bias = state_dict.get(bias_key)
    if (
      legacy_weight is not None
      and legacy_bias is not None
      and legacy_weight.shape[0] == 1
      and current_weight.shape[0] == 2
      and self.structured_safe_stride_enabled
      and legacy_weight.shape[1:] == current_weight.shape[1:]
      and legacy_bias.shape == (1,)
    ):
      upgraded_weight = current_weight.detach().clone()
      upgraded_bias = current_bias.detach().clone()
      upgraded_weight[0] = legacy_weight[0]
      upgraded_weight[1].zero_()
      upgraded_bias[0] = legacy_bias[0]
      upgraded_bias[1] = _LEGACY_SAFE_STRIDE_WIDTH_LOGIT
      state_dict[weight_key] = upgraded_weight
      state_dict[bias_key] = upgraded_bias
    input_weight_key = f"{prefix}safe_stride_head.0.weight"
    resize_linear_input(
      input_weight_key, cast(nn.Linear, self.safe_stride_head[0]).weight
    )
    stair_shape_input_weight_key = f"{prefix}stair_shape_head.0.weight"
    resize_linear_input(
      stair_shape_input_weight_key,
      cast(nn.Linear, self.stair_shape_head[0]).weight,
    )
    encoder_weight_key = f"{prefix}latent_encoder.0.weight"
    legacy_encoder_weight = state_dict.get(encoder_weight_key)
    current_encoder_weight = cast(nn.Linear, self.latent_encoder[0]).weight
    if (
      legacy_encoder_weight is not None
      and legacy_encoder_weight.shape[0] == current_encoder_weight.shape[0]
      and legacy_encoder_weight.shape[1] != current_encoder_weight.shape[1]
    ):
      upgraded_encoder_weight = current_encoder_weight.detach().clone()
      upgraded_encoder_weight.zero_()
      input_dim = min(legacy_encoder_weight.shape[1], current_encoder_weight.shape[1])
      upgraded_encoder_weight[:, :input_dim] = legacy_encoder_weight[:, :input_dim]
      state_dict[encoder_weight_key] = upgraded_encoder_weight
    current_normalizer_state = self.latent_obs_normalizer.state_dict()
    for name in ("_mean", "_var", "_std"):
      key = f"{prefix}latent_obs_normalizer.{name}"
      legacy_value = state_dict.get(key)
      current_value = current_normalizer_state.get(name)
      if (
        legacy_value is not None
        and current_value is not None
        and legacy_value.shape[:-1] == current_value.shape[:-1]
        and legacy_value.shape[-1] != current_value.shape[-1]
      ):
        upgraded_value = current_value.detach().clone()
        feature_dim = min(legacy_value.shape[-1], current_value.shape[-1])
        upgraded_value[..., :feature_dim] = legacy_value[..., :feature_dim]
        state_dict[key] = upgraded_value
    mlp_input_weight_key = f"{prefix}mlp.0.weight"
    legacy_mlp_input_weight = state_dict.get(mlp_input_weight_key)
    current_mlp_input_weight = cast(nn.Linear, self.mlp[0]).weight
    if (
      self.actor_semantic_enabled
      and legacy_mlp_input_weight is not None
      and legacy_mlp_input_weight.shape[0] == current_mlp_input_weight.shape[0]
      and legacy_mlp_input_weight.shape[1] + _SEMANTIC_DIM
      == current_mlp_input_weight.shape[1]
    ):
      upgraded_mlp_input_weight = current_mlp_input_weight.detach().clone()
      upgraded_mlp_input_weight.zero_()
      upgraded_mlp_input_weight[:, : legacy_mlp_input_weight.shape[1]] = (
        legacy_mlp_input_weight
      )
      state_dict[mlp_input_weight_key] = upgraded_mlp_input_weight
    if self.safe_stride_width_head is not None:
      for name, value in self.safe_stride_width_head.state_dict().items():
        key = f"{prefix}safe_stride_width_head.{name}"
        legacy_value = state_dict.get(key)
        if (
          name == "0.weight"
          and legacy_value is not None
          and legacy_value.shape[0] == value.shape[0]
          and legacy_value.shape[1] != value.shape[1]
        ):
          migrated_value = value.detach().clone()
          migrated_value.zero_()
          input_dim = min(legacy_value.shape[1], value.shape[1])
          migrated_value[:, :input_dim] = legacy_value[:, :input_dim]
          state_dict[key] = migrated_value
        elif key not in state_dict:
          migrated_value = value.detach().clone()
          if name == "2.weight":
            migrated_value.zero_()
          elif name == "2.bias":
            migrated_value.fill_(_DYNAMIC_SAFE_STRIDE_WIDTH_LOGIT)
          state_dict[key] = migrated_value
    for name, value in self.safe_stride_confidence_head.state_dict().items():
      key = f"{prefix}safe_stride_confidence_head.{name}"
      legacy_value = state_dict.get(key)
      if (
        name == "0.weight"
        and legacy_value is not None
        and legacy_value.shape[0] == value.shape[0]
        and legacy_value.shape[1] != value.shape[1]
      ):
        migrated_value = value.detach().clone()
        migrated_value.zero_()
        input_dim = min(legacy_value.shape[1], value.shape[1])
        migrated_value[:, :input_dim] = legacy_value[:, :input_dim]
        state_dict[key] = migrated_value
      elif key not in state_dict:
        migrated_value = value.detach().clone()
        if name == "2.weight":
          migrated_value.zero_()
        elif name == "2.bias":
          migrated_value.fill_(_SAFE_STRIDE_CONFIDENCE_LOGIT)
        state_dict[key] = migrated_value
    if self.geometry_probe_head is not None:
      for name, value in self.geometry_probe_head.state_dict().items():
        key = f"{prefix}geometry_probe_head.{name}"
        if key not in state_dict:
          state_dict[key] = value.detach().clone()
    super()._load_from_state_dict(
      state_dict,
      prefix,
      local_metadata,
      strict,
      missing_keys,
      unexpected_keys,
      error_msgs,
    )

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
    was_in_memory = mode == _MODE_STAIR_MEMORY
    stair_timer = gate_state[:, 1]
    evidence_timer = gate_state[:, 2]
    write_timer = gate_state[:, 3]
    cooldown = torch.clamp(gate_state[:, 4] - 1.0, min=0.0)

    in_normal = mode == _MODE_NORMAL
    release_finished = in_normal & (write_timer < 0.0) & (cooldown <= 0.0)
    write_timer = torch.where(
      release_finished, torch.zeros_like(write_timer), write_timer
    )
    rearm_event = in_normal & (event_prob < self.event_off_threshold)
    evidence_timer = torch.where(
      rearm_event, torch.zeros_like(evidence_timer), evidence_timer
    )
    event_armed = evidence_timer >= 0.0
    trigger = (
      in_normal
      & event_armed
      & (cooldown <= 0.0)
      & (event_prob > self.event_on_threshold)
    )
    mode = torch.where(trigger, torch.full_like(mode, _MODE_STAIR_WRITE), mode)
    write_timer = torch.where(trigger, torch.zeros_like(write_timer), write_timer)
    stair_timer = torch.where(trigger, torch.zeros_like(stair_timer), stair_timer)
    evidence_timer = torch.where(
      trigger, torch.zeros_like(evidence_timer), evidence_timer
    )

    in_write = mode == _MODE_STAIR_WRITE
    write_active_for_update = in_write
    write_timer = torch.where(in_write, write_timer + 1.0, write_timer)
    stair_timer = torch.where(in_write, stair_timer + 1.0, stair_timer)
    stair_evidence = in_write & (stair_prob > self.stair_on_threshold)
    evidence_timer = torch.where(
      in_write,
      torch.where(
        stair_evidence,
        evidence_timer + 1.0,
        torch.zeros_like(evidence_timer),
      ),
      evidence_timer,
    )
    done_write = in_write & (write_timer >= self.write_steps)
    confirm_memory = done_write & (evidence_timer >= self.stair_confirm_steps)
    abort_write = done_write & ~confirm_memory
    mode = torch.where(confirm_memory, torch.full_like(mode, _MODE_STAIR_MEMORY), mode)
    mode = torch.where(abort_write, torch.full_like(mode, _MODE_NORMAL), mode)
    cooldown = torch.where(
      abort_write, torch.full_like(cooldown, self.cooldown_steps), cooldown
    )
    stair_timer = torch.where(abort_write, torch.zeros_like(stair_timer), stair_timer)
    evidence_timer = torch.where(
      confirm_memory,
      torch.zeros_like(evidence_timer),
      torch.where(abort_write, -torch.ones_like(evidence_timer), evidence_timer),
    )
    write_timer = torch.where(
      confirm_memory | abort_write,
      torch.zeros_like(write_timer),
      write_timer,
    )

    in_memory = mode == _MODE_STAIR_MEMORY
    memory_event = was_in_memory & in_memory & (event_prob > self.event_on_threshold)
    memory_boost_timer = torch.where(
      memory_event,
      torch.full_like(write_timer, self.memory_event_shape_boost_steps),
      torch.clamp(write_timer - 1.0, min=0.0),
    )
    write_timer = torch.where(in_memory, memory_boost_timer, write_timer)
    memory_shape_boost = in_memory & (write_timer > 0.0)
    stair_timer = torch.where(in_memory, stair_timer + 1.0, stair_timer)
    stair_off = stair_prob < self.stair_off_threshold
    if not self.stair_memory_exit_on_stair_off:
      stair_off = torch.zeros_like(stair_off)
    evidence_timer = torch.where(
      in_memory & stair_off,
      evidence_timer + 1.0,
      torch.where(in_memory, torch.zeros_like(evidence_timer), evidence_timer),
    )
    exit_memory = (
      in_memory
      & (evidence_timer >= self.exit_steps)
      & (stair_timer >= self.min_stair_steps)
    )
    mode = torch.where(exit_memory, torch.full_like(mode, _MODE_NORMAL), mode)
    cooldown = torch.where(
      exit_memory, torch.full_like(cooldown, self.cooldown_steps), cooldown
    )
    stair_timer = torch.where(exit_memory, torch.zeros_like(stair_timer), stair_timer)
    evidence_timer = torch.where(
      exit_memory, -torch.ones_like(evidence_timer), evidence_timer
    )
    write_timer = torch.where(exit_memory, -torch.ones_like(write_timer), write_timer)

    next_gate = torch.stack(
      [mode, stair_timer, evidence_timer, write_timer, cooldown], dim=-1
    )
    alpha = torch.full(
      (event_prob.shape[0], self.z_dim),
      self.alpha_fast,
      device=event_prob.device,
      dtype=event_prob.dtype,
    )
    alpha = torch.where(write_active_for_update[:, None], self.alpha_write, alpha)
    hold_alpha = torch.cat(
      [
        torch.full_like(alpha[:, : self.state_latent_dim], self.alpha_hold_state),
        torch.full_like(alpha[:, self.state_latent_dim :], self.alpha_hold_shape),
      ],
      dim=-1,
    )
    hold_memory = (mode == _MODE_STAIR_MEMORY) & ~write_active_for_update
    alpha = torch.where(hold_memory[:, None], hold_alpha, alpha)
    boosted_hold_alpha = torch.cat(
      [
        torch.full_like(
          alpha[:, : self.state_latent_dim],
          self.alpha_hold_state,
        ),
        torch.full_like(
          alpha[:, self.state_latent_dim :],
          self.alpha_fast,
        ),
      ],
      dim=-1,
    )
    alpha = torch.where(
      (hold_memory & memory_shape_boost)[:, None],
      boosted_hold_alpha,
      alpha,
    )
    release_active = (mode == _MODE_NORMAL) & (write_timer < 0.0) & (cooldown > 0.0)
    release_progress = torch.clamp(
      (self.cooldown_steps - cooldown) / max(self.cooldown_steps, 1.0),
      min=0.0,
      max=1.0,
    )
    release_alpha = hold_alpha + release_progress[:, None] * (
      self.alpha_fast - hold_alpha
    )
    alpha = torch.where(release_active[:, None], release_alpha, alpha)

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
  ) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None
  ]:
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
    gate_steps: list[torch.Tensor] = []
    trigger_steps: list[torch.Tensor] = []
    confirm_steps: list[torch.Tensor] = []
    abort_steps: list[torch.Tensor] = []
    exit_steps: list[torch.Tensor] = []
    event_logits: list[torch.Tensor] = []
    stair_logits: list[torch.Tensor] = []
    future_risk_logits: list[torch.Tensor] = []
    future_quality_logits: list[torch.Tensor] = []
    stair_shape_predictions: list[torch.Tensor] = []
    geometry_probe_predictions: list[torch.Tensor] = []
    safe_stride_predictions: list[torch.Tensor] = []
    safe_stride_intervals: list[torch.Tensor] = []
    safe_stride_confidence_logits: list[torch.Tensor] = []

    valid_masks = masks
    if valid_masks is not None and valid_masks.dim() == 2:
      valid_masks = valid_masks.unsqueeze(-1)

    for step in range(seq_len):
      h_t = lstm_out[step]
      z_candidate = self.z_candidate_head(h_t)
      event_logit = self.event_head(h_t)
      stair_logit = self._stair_logit(h_t)
      valid = valid_masks[step] if valid_masks is not None else None

      gate_next, alpha = self._advance_gate_state(
        torch.sigmoid(event_logit),
        torch.sigmoid(stair_logit),
        gate,
        valid=valid,
      )
      previous_mode = gate[:, 0:1]
      next_mode = gate_next[:, 0:1]
      trigger_steps.append(
        ((previous_mode == _MODE_NORMAL) & (next_mode == _MODE_STAIR_WRITE)).float()
      )
      confirm_steps.append(
        (
          (previous_mode == _MODE_STAIR_WRITE) & (next_mode == _MODE_STAIR_MEMORY)
        ).float()
      )
      abort_steps.append(
        ((previous_mode == _MODE_STAIR_WRITE) & (next_mode == _MODE_NORMAL)).float()
      )
      exit_steps.append(
        ((previous_mode == _MODE_STAIR_MEMORY) & (next_mode == _MODE_NORMAL)).float()
      )
      z_next = (1.0 - alpha) * z + alpha * z_candidate
      if valid is not None:
        z_next = torch.where(valid.bool(), z_next, z)

      future_risk_logit = self.future_collision_risk_head(z_next)
      future_quality_logit = self.future_safe_landing_quality_head(z_next)
      shape_memory = self._shape_memory(z_next)
      stair_shape_prediction = self._decode_stair_shape(shape_memory, h_t)
      geometry_probe_prediction = self._decode_geometry_probe(shape_memory, h_t)
      safe_stride_features = self._safe_stride_features(
        shape_memory,
        h_t,
        latent_obs[step],
      )
      safe_stride_interval = self._decode_safe_stride_interval(
        safe_stride_features,
        shape_memory,
      )
      safe_stride_confidence_logit = self._decode_safe_stride_confidence_logit(
        safe_stride_features
      )
      safe_stride_prediction = 0.5 * (
        safe_stride_interval[..., 0:1] + safe_stride_interval[..., 1:2]
      )

      z = z_next
      gate = gate_next
      z_steps.append(z)
      alpha_steps.append(alpha)
      mode_steps.append(gate[:, 0:1])
      gate_steps.append(gate)
      event_logits.append(event_logit)
      stair_logits.append(stair_logit)
      future_risk_logits.append(future_risk_logit)
      future_quality_logits.append(future_quality_logit)
      stair_shape_predictions.append(stair_shape_prediction)
      if geometry_probe_prediction is not None:
        geometry_probe_predictions.append(geometry_probe_prediction)
      safe_stride_predictions.append(safe_stride_prediction)
      safe_stride_intervals.append(safe_stride_interval)
      safe_stride_confidence_logits.append(safe_stride_confidence_logit)

    z_seq = torch.stack(z_steps, dim=0)
    alpha_seq = torch.stack(alpha_steps, dim=0)
    mode_seq = torch.stack(mode_steps, dim=0)
    gate_seq = torch.stack(gate_steps, dim=0)
    trigger_seq = torch.stack(trigger_steps, dim=0)
    confirm_seq = torch.stack(confirm_steps, dim=0)
    abort_seq = torch.stack(abort_steps, dim=0)
    exit_seq = torch.stack(exit_steps, dim=0)
    if valid_masks is None:
      valid_for_episode = torch.ones_like(mode_seq, dtype=torch.bool)
    else:
      valid_for_episode = valid_masks.bool()
    write_ever = (
      ((mode_seq == _MODE_STAIR_WRITE) & valid_for_episode)
      .any(dim=0)
      .to(mode_seq.dtype)
    )
    memory_ever = (
      ((mode_seq == _MODE_STAIR_MEMORY) & valid_for_episode)
      .any(dim=0)
      .to(mode_seq.dtype)
    )
    memory_age_seq = gate_seq[..., 1:2] * (mode_seq == _MODE_STAIR_MEMORY).to(
      gate_seq.dtype
    )
    release_seq = (
      (mode_seq == _MODE_NORMAL)
      & (gate_seq[..., 3:4] < 0.0)
      & (gate_seq[..., 4:5] > 0.0)
    ).to(gate_seq.dtype)
    self._aux_event_logits = torch.stack(event_logits, dim=0)
    self._aux_stair_logits = torch.stack(stair_logits, dim=0)
    self._aux_future_collision_risk_logits = torch.stack(future_risk_logits, dim=0)
    self._aux_future_safe_landing_quality_logits = torch.stack(
      future_quality_logits, dim=0
    )
    self._aux_stair_shape_predictions = torch.stack(stair_shape_predictions, dim=0)
    self._aux_geometry_probe_predictions = (
      torch.stack(geometry_probe_predictions, dim=0)
      if geometry_probe_predictions
      else None
    )
    self._aux_safe_stride_predictions = torch.stack(safe_stride_predictions, dim=0)
    self._aux_safe_stride_intervals = torch.stack(safe_stride_intervals, dim=0)
    self._aux_safe_stride_confidence_logits = torch.stack(
      safe_stride_confidence_logits,
      dim=0,
    )
    semantic = (
      self._build_shadow_semantic(
        torch.sigmoid(self._aux_event_logits),
        torch.sigmoid(self._aux_stair_logits),
        gate_seq,
        self._aux_stair_shape_predictions,
        self._aux_safe_stride_intervals,
        torch.sigmoid(self._aux_safe_stride_confidence_logits),
        latent_obs,
      )
      if self.shadow_semantic_enabled or self.actor_semantic_enabled
      else None
    )

    if masks is not None:
      z_seq = cast(torch.Tensor, unpad_trajectories(z_seq, masks))
      alpha_seq = cast(torch.Tensor, unpad_trajectories(alpha_seq, masks))
      mode_seq = cast(torch.Tensor, unpad_trajectories(mode_seq, masks))
      memory_age_seq = cast(
        torch.Tensor,
        unpad_trajectories(memory_age_seq, masks),
      )
      release_seq = cast(torch.Tensor, unpad_trajectories(release_seq, masks))
      trigger_seq = cast(torch.Tensor, unpad_trajectories(trigger_seq, masks))
      confirm_seq = cast(torch.Tensor, unpad_trajectories(confirm_seq, masks))
      abort_seq = cast(torch.Tensor, unpad_trajectories(abort_seq, masks))
      exit_seq = cast(torch.Tensor, unpad_trajectories(exit_seq, masks))
      actor_obs = cast(torch.Tensor, unpad_trajectories(actor_obs, masks))
      self._aux_event_logits = cast(
        torch.Tensor, unpad_trajectories(self._aux_event_logits, masks)
      )
      self._aux_stair_logits = cast(
        torch.Tensor, unpad_trajectories(self._aux_stair_logits, masks)
      )
      self._aux_future_collision_risk_logits = cast(
        torch.Tensor,
        unpad_trajectories(self._aux_future_collision_risk_logits, masks),
      )
      self._aux_future_safe_landing_quality_logits = cast(
        torch.Tensor,
        unpad_trajectories(self._aux_future_safe_landing_quality_logits, masks),
      )
      self._aux_stair_shape_predictions = cast(
        torch.Tensor,
        unpad_trajectories(self._aux_stair_shape_predictions, masks),
      )
      if self._aux_geometry_probe_predictions is not None:
        self._aux_geometry_probe_predictions = cast(
          torch.Tensor,
          unpad_trajectories(self._aux_geometry_probe_predictions, masks),
        )
      self._aux_safe_stride_predictions = cast(
        torch.Tensor,
        unpad_trajectories(self._aux_safe_stride_predictions, masks),
      )
      self._aux_safe_stride_intervals = cast(
        torch.Tensor,
        unpad_trajectories(self._aux_safe_stride_intervals, masks),
      )
      self._aux_safe_stride_confidence_logits = cast(
        torch.Tensor,
        unpad_trajectories(self._aux_safe_stride_confidence_logits, masks),
      )
      if semantic is not None:
        semantic = cast(
          torch.Tensor,
          unpad_trajectories(semantic, masks),
        )
    elif not sequence_mode:
      z_seq = z_seq.squeeze(0)
      alpha_seq = alpha_seq.squeeze(0)
      mode_seq = mode_seq.squeeze(0)
      memory_age_seq = memory_age_seq.squeeze(0)
      release_seq = release_seq.squeeze(0)
      trigger_seq = trigger_seq.squeeze(0)
      confirm_seq = confirm_seq.squeeze(0)
      abort_seq = abort_seq.squeeze(0)
      exit_seq = exit_seq.squeeze(0)
      actor_obs = actor_obs.squeeze(0)
      self._aux_event_logits = self._aux_event_logits.squeeze(0)
      self._aux_stair_logits = self._aux_stair_logits.squeeze(0)
      self._aux_future_collision_risk_logits = (
        self._aux_future_collision_risk_logits.squeeze(0)
      )
      self._aux_future_safe_landing_quality_logits = (
        self._aux_future_safe_landing_quality_logits.squeeze(0)
      )
      self._aux_stair_shape_predictions = self._aux_stair_shape_predictions.squeeze(0)
      if self._aux_geometry_probe_predictions is not None:
        self._aux_geometry_probe_predictions = (
          self._aux_geometry_probe_predictions.squeeze(0)
        )
      self._aux_safe_stride_predictions = self._aux_safe_stride_predictions.squeeze(0)
      self._aux_safe_stride_intervals = self._aux_safe_stride_intervals.squeeze(0)
      self._aux_safe_stride_confidence_logits = (
        self._aux_safe_stride_confidence_logits.squeeze(0)
      )
      if semantic is not None:
        semantic = semantic.squeeze(0)

    self._shadow_semantic = (
      semantic.detach()
      if self.shadow_semantic_enabled and semantic is not None
      else None
    )
    actor_semantic = (
      semantic.detach()
      if self.actor_semantic_enabled and semantic is not None
      else None
    )

    self._update_slow_latent_diagnostics(
      z_seq,
      alpha_seq,
      mode_seq,
      memory_age_seq,
      release_seq,
      write_ever,
      memory_ever,
      trigger_seq,
      confirm_seq,
      abort_seq,
      exit_seq,
    )

    if hidden_state is None:
      self._hidden_state = (h_out.detach(), c_out.detach())
      self._z_memory = z.detach()
      self._gate_state = gate.detach()

    return actor_obs, z_seq, h_out, c_out, actor_semantic

  def _update_slow_latent_diagnostics(
    self,
    z_seq: torch.Tensor,
    alpha_seq: torch.Tensor,
    mode_seq: torch.Tensor,
    memory_age_seq: torch.Tensor,
    release_seq: torch.Tensor,
    write_ever: torch.Tensor,
    memory_ever: torch.Tensor,
    trigger_seq: torch.Tensor,
    confirm_seq: torch.Tensor,
    abort_seq: torch.Tensor,
    exit_seq: torch.Tensor,
  ) -> None:
    if (
      self._aux_event_logits is None
      or self._aux_stair_logits is None
      or self._aux_future_collision_risk_logits is None
      or self._aux_future_safe_landing_quality_logits is None
      or self._aux_stair_shape_predictions is None
      or self._aux_safe_stride_predictions is None
      or self._aux_safe_stride_confidence_logits is None
    ):
      self._slow_latent_diagnostics = {}
      return
    diagnostics = {
      "event_prob": torch.sigmoid(self._aux_event_logits.detach()),
      "stair_prob": torch.sigmoid(self._aux_stair_logits.detach()),
      "future_risk": torch.sigmoid(self._aux_future_collision_risk_logits.detach()),
      "future_quality": torch.sigmoid(
        self._aux_future_safe_landing_quality_logits.detach()
      ),
      "stair_shape": self._aux_stair_shape_predictions.detach(),
      "safe_stride": self._aux_safe_stride_predictions.detach(),
      "safe_stride_confidence": torch.sigmoid(
        self._aux_safe_stride_confidence_logits.detach()
      ),
      "z_norm": z_seq.detach().norm(dim=-1, keepdim=True),
      "gate_mode": mode_seq.detach(),
      "gate_memory_age": memory_age_seq.detach(),
      "gate_release": release_seq.detach(),
      "episode_write_ever": write_ever.detach(),
      "episode_memory_ever": memory_ever.detach(),
      "gate_event_trigger": trigger_seq.detach(),
      "gate_write_confirm": confirm_seq.detach(),
      "gate_write_abort": abort_seq.detach(),
      "gate_memory_exit": exit_seq.detach(),
      "gate_memory_event_shape_boost": (
        (mode_seq == _MODE_STAIR_MEMORY)
        & (
          alpha_seq[..., self.state_latent_dim :]
          .mean(dim=-1, keepdim=True)
          .isclose(alpha_seq.new_tensor(self.alpha_fast))
        )
      ).detach(),
      "alpha": alpha_seq.detach(),
      "alpha_state": alpha_seq[..., : self.state_latent_dim]
      .detach()
      .mean(dim=-1, keepdim=True),
      "alpha_shape": alpha_seq[..., self.state_latent_dim :]
      .detach()
      .mean(dim=-1, keepdim=True),
    }
    if (
      self.structured_safe_stride_enabled
      and self._aux_safe_stride_intervals is not None
    ):
      diagnostics["safe_stride_interval"] = self._aux_safe_stride_intervals.detach()
    self._slow_latent_diagnostics = diagnostics
    if self._shadow_semantic is not None:
      self._slow_latent_diagnostics["shadow_semantic"] = self._shadow_semantic

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

    actor_obs_norm, z_memory, _h_out, _c_out, actor_semantic = self._run_latent_path(
      actor_obs_norm, latent_obs, masks, hidden_state
    )
    actor_input_terms = [actor_obs_norm, self._actor_memory(z_memory)]
    if self.actor_semantic_enabled:
      if actor_semantic is None:
        actor_semantic = z_memory.new_zeros((*z_memory.shape[:-1], _SEMANTIC_DIM))
      actor_input_terms.append(actor_semantic)
    actor_input = torch.cat(actor_input_terms, dim=-1)
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
      self._pre_stair_shape_actor_memory = None
      self._pre_stair_shape_actor_memory_valid = None
      self._shadow_semantic = None
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
    if self._pre_stair_shape_actor_memory is not None:
      self._pre_stair_shape_actor_memory[done_mask] = 0.0
    if self._pre_stair_shape_actor_memory_valid is not None:
      self._pre_stair_shape_actor_memory_valid[done_mask] = False

  def reset_slow_latent(self) -> None:
    if self._z_memory is not None:
      self._z_memory.zero_()
    if self._gate_state is not None:
      self._gate_state.zero_()
    if self._pre_stair_shape_actor_memory is not None:
      self._pre_stair_shape_actor_memory.zero_()
    if self._pre_stair_shape_actor_memory_valid is not None:
      self._pre_stair_shape_actor_memory_valid.zero_()
    self._shadow_semantic = None

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    if self._hidden_state is not None:
      self._hidden_state = tuple(state.detach() for state in self._hidden_state)  # type: ignore[assignment]
    if self._z_memory is not None:
      self._z_memory = self._z_memory.detach()
    if self._gate_state is not None:
      self._gate_state = self._gate_state.detach()
    if self._pre_stair_shape_actor_memory is not None:
      self._pre_stair_shape_actor_memory = self._pre_stair_shape_actor_memory.detach()
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
    if self._aux_future_collision_risk_logits is not None:
      outputs["future_collision_risk_logit"] = self._aux_future_collision_risk_logits
    if self._aux_future_safe_landing_quality_logits is not None:
      outputs["future_safe_landing_quality_logit"] = (
        self._aux_future_safe_landing_quality_logits
      )
    if self._aux_stair_shape_predictions is not None:
      outputs["stair_shape"] = self._aux_stair_shape_predictions
    if self._aux_geometry_probe_predictions is not None:
      outputs["geometry_probe"] = self._aux_geometry_probe_predictions
    if self._aux_safe_stride_predictions is not None:
      outputs["safe_stride"] = self._aux_safe_stride_predictions
    if (
      self.structured_safe_stride_enabled
      and self._aux_safe_stride_intervals is not None
    ):
      outputs["safe_stride_interval"] = self._aux_safe_stride_intervals
    if self._aux_safe_stride_confidence_logits is not None:
      outputs["safe_stride_confidence_logit"] = self._aux_safe_stride_confidence_logits
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
  def aux_future_collision_risk_logits(self) -> torch.Tensor | None:
    return self._aux_future_collision_risk_logits

  @property
  def aux_future_safe_landing_quality_logits(self) -> torch.Tensor | None:
    return self._aux_future_safe_landing_quality_logits

  @property
  def aux_stair_shape_predictions(self) -> torch.Tensor | None:
    return self._aux_stair_shape_predictions

  @property
  def aux_safe_stride_predictions(self) -> torch.Tensor | None:
    return self._aux_safe_stride_predictions

  @property
  def aux_safe_stride_intervals(self) -> torch.Tensor | None:
    return self._aux_safe_stride_intervals

  @property
  def aux_safe_stride_confidence_logits(self) -> torch.Tensor | None:
    return self._aux_safe_stride_confidence_logits

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
    self.future_collision_risk_head = copy.deepcopy(model.future_collision_risk_head)
    self.future_safe_landing_quality_head = copy.deepcopy(
      model.future_safe_landing_quality_head
    )
    self.stair_shape_head = copy.deepcopy(model.stair_shape_head)
    self.safe_stride_head = copy.deepcopy(model.safe_stride_head)
    self.safe_stride_width_head = copy.deepcopy(model.safe_stride_width_head)
    self.safe_stride_confidence_head = copy.deepcopy(model.safe_stride_confidence_head)
    self.mlp = copy.deepcopy(model.mlp)
    self.deterministic_output = (
      model.distribution.as_deterministic_output_module()
      if model.distribution is not None
      else nn.Identity()
    )

    self.actor_obs_dim = model.obs_dim
    self.latent_obs_dim = model.latent_obs_dim
    self.z_dim = model.z_dim
    self.state_latent_dim = model.state_latent_dim
    self.latent_hidden_dim = model.latent_hidden_dim
    self.tread_depth_min = model.tread_depth_min
    self.tread_depth_max = model.tread_depth_max
    self.riser_height_min = model.riser_height_min
    self.riser_height_max = model.riser_height_max
    self.safe_stride_min = model.safe_stride_min
    self.safe_stride_max = model.safe_stride_max
    self.same_foot_stride_min = model.same_foot_stride_min
    self.same_foot_stride_max = model.same_foot_stride_max
    self.structured_safe_stride_enabled = model.structured_safe_stride_enabled
    self.dynamic_stair_shape_enabled = model.dynamic_stair_shape_enabled
    self.dynamic_safe_stride_enabled = model.dynamic_safe_stride_enabled
    self.actor_semantic_enabled = model.actor_semantic_enabled
    self.safe_stride_phase_dim = model.safe_stride_phase_dim
    self.safe_stride_phase_start = model.safe_stride_phase_start
    self.alpha_fast = model.alpha_fast
    self.alpha_write = model.alpha_write
    self.alpha_hold_state = model.alpha_hold_state
    self.alpha_hold_shape = model.alpha_hold_shape
    self.memory_event_shape_boost_steps = model.memory_event_shape_boost_steps
    self.write_steps = model.write_steps
    self.stair_confirm_steps = model.stair_confirm_steps
    self.min_stair_steps = model.min_stair_steps
    self.exit_steps = model.exit_steps
    self.stair_memory_exit_on_stair_off = model.stair_memory_exit_on_stair_off
    self.cooldown_steps = model.cooldown_steps
    self.event_on_threshold = model.event_on_threshold
    self.event_off_threshold = model.event_off_threshold
    self.stair_on_threshold = model.stair_on_threshold
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
      "future_collision_risk",
      "future_safe_landing_quality",
      "stair_shape",
      "safe_stride",
      "safe_stride_interval",
      "safe_stride_confidence",
    ]

  def _decode_stair_shape(
    self,
    shape_memory: torch.Tensor,
    h_t: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if self.dynamic_stair_shape_enabled:
      assert h_t is not None
      features = torch.cat([shape_memory, h_t], dim=-1)
    else:
      features = shape_memory
    shape01 = torch.sigmoid(self.stair_shape_head(features))
    same_foot_stride = self.same_foot_stride_min + shape01[..., 0:1] * (
      self.same_foot_stride_max - self.same_foot_stride_min
    )
    riser_height = self.riser_height_min + shape01[..., 1:2] * (
      self.riser_height_max - self.riser_height_min
    )
    return torch.cat([same_foot_stride, riser_height], dim=-1)

  def _decode_safe_stride_outputs(
    self,
    shape_memory: torch.Tensor,
    h_t: torch.Tensor,
    latent_obs: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    safe_stride_features = shape_memory
    if self.dynamic_safe_stride_enabled:
      phase_end = self.safe_stride_phase_start + self.safe_stride_phase_dim
      gait_phase = latent_obs[..., self.safe_stride_phase_start : phase_end]
      safe_stride_features = torch.cat([shape_memory, h_t, gait_phase], dim=-1)
    stride_raw = self.safe_stride_head(safe_stride_features)
    lower01 = torch.sigmoid(stride_raw[..., 0:1])
    lower = self.safe_stride_min + lower01 * (
      self.safe_stride_max - self.safe_stride_min
    )
    confidence = torch.sigmoid(self.safe_stride_confidence_head(safe_stride_features))
    if not self.structured_safe_stride_enabled:
      interval = torch.cat([lower, lower], dim=-1)
      return lower, interval, confidence
    if self.dynamic_safe_stride_enabled:
      assert self.safe_stride_width_head is not None
      width_logit = self.safe_stride_width_head(safe_stride_features)
      width = torch.sigmoid(width_logit) * (self.safe_stride_max - lower)
    else:
      width_logit = stride_raw[..., 1:2]
      width = torch.sigmoid(width_logit) * (self.safe_stride_max - lower)
    upper = lower + width
    interval = torch.cat([lower, upper], dim=-1)
    center = 0.5 * (lower + upper)
    return center, interval, confidence

  def _stair_logit(self, h_t: torch.Tensor) -> torch.Tensor:
    memory_placeholder = h_t.new_zeros((*h_t.shape[:-1], self.state_latent_dim))
    return self.stair_state_head(torch.cat([h_t, memory_placeholder], dim=-1))

  @staticmethod
  def _normalize_semantic_value(
    value: torch.Tensor,
    minimum: float,
    maximum: float,
  ) -> torch.Tensor:
    return torch.clamp((value - minimum) / (maximum - minimum), 0.0, 1.0)

  @staticmethod
  def _foot_event_summary_dim(latent_obs: torch.Tensor) -> int:
    latent_dim = int(latent_obs.shape[-1])
    if latent_dim < 80:
      return 0
    new_foot_only = (latent_dim - 80) % 33 == 0
    new_with_stair = latent_dim >= 173 and (latent_dim - 173) % 33 == 0
    return 80 if latent_dim == 80 or new_foot_only or new_with_stair else 0

  def _same_foot_actor_stride_target(
    self,
    stair_shape: torch.Tensor,
    latent_obs: torch.Tensor | None = None,
  ) -> torch.Tensor:
    target = stair_shape[..., 0:1]
    if latent_obs is None:
      return target
    summary_dim = self._foot_event_summary_dim(latent_obs)
    if summary_dim < 80:
      return target
    summary = torch.nan_to_num(latent_obs[..., -summary_dim:])
    ratchet = summary[..., 70:80]
    active = ratchet[..., 0:1] > 0.5
    lower = ratchet[..., 5:6].clamp(
      self.same_foot_stride_min,
      self.same_foot_stride_max,
    )
    probe = ratchet[..., 2:3].clamp(
      self.same_foot_stride_min,
      self.same_foot_stride_max,
    )
    upper_raw = ratchet[..., 7:8]
    confirmed = ratchet[..., 6:7] > 0.5
    soft_cap = active & ~confirmed & (upper_raw > 0.0)
    open_base = torch.maximum(probe, lower)
    guard_limited = probe <= lower + 1.0e-5
    open_ceiling = torch.where(guard_limited, probe, open_base + 0.05)
    open_target = torch.maximum(open_base, torch.minimum(target, open_ceiling))
    closed_target = probe
    soft_target = probe
    return torch.where(
      active,
      torch.where(
        confirmed,
        closed_target,
        torch.where(soft_cap, soft_target, open_target),
      ),
      target,
    )

  def _build_actor_semantic(
    self,
    event_prob: torch.Tensor,
    stair_prob: torch.Tensor,
    gate_state: torch.Tensor,
    stair_shape: torch.Tensor,
    safe_stride_interval: torch.Tensor,
    safe_stride_confidence: torch.Tensor,
    latent_obs: torch.Tensor | None = None,
  ) -> torch.Tensor:
    mode = gate_state[..., 0:1]
    write_timer = gate_state[..., 3:4]
    cooldown = gate_state[..., 4:5]
    event_on = (event_prob > self.event_on_threshold).to(event_prob.dtype)
    stair_on = (stair_prob > self.stair_on_threshold).to(event_prob.dtype)
    mode_normal = (mode == _MODE_NORMAL).to(event_prob.dtype)
    mode_write = (mode == _MODE_STAIR_WRITE).to(event_prob.dtype)
    mode_memory = (mode == _MODE_STAIR_MEMORY).to(event_prob.dtype)
    write_progress = mode_write * torch.clamp(
      write_timer / max(self.write_steps, 1.0),
      0.0,
      1.0,
    )
    memory_age = mode_memory * torch.clamp(
      gate_state[..., 1:2] / max(self.min_stair_steps, 1.0),
      0.0,
      1.0,
    )
    release_active = (mode == _MODE_NORMAL) & (write_timer < 0.0) & (cooldown > 0.0)
    release_progress = release_active.to(event_prob.dtype) * torch.clamp(
      (self.cooldown_steps - cooldown) / max(self.cooldown_steps, 1.0),
      0.0,
      1.0,
    )
    state_semantic = torch.cat(
      [
        event_on,
        stair_on,
        mode_normal,
        mode_write,
        mode_memory,
        write_progress,
        memory_age,
        release_progress,
      ],
      dim=-1,
    )

    same_foot_stride = stair_shape[..., 0:1]
    riser_height = stair_shape[..., 1:2]
    same_foot_stride_norm = self._normalize_semantic_value(
      same_foot_stride,
      self.same_foot_stride_min,
      self.same_foot_stride_max,
    )
    riser_height_norm = self._normalize_semantic_value(
      riser_height,
      self.riser_height_min,
      self.riser_height_max,
    )
    stride_lower = safe_stride_interval[..., 0:1]
    stride_upper = safe_stride_interval[..., 1:2]
    stride_center = 0.5 * (stride_lower + stride_upper)
    stride_width = stride_upper - stride_lower
    stride_lower_norm = self._normalize_semantic_value(
      stride_lower,
      self.safe_stride_min,
      self.safe_stride_max,
    )
    stride_upper_norm = self._normalize_semantic_value(
      stride_upper,
      self.safe_stride_min,
      self.safe_stride_max,
    )
    stride_center_norm = self._normalize_semantic_value(
      stride_center,
      self.safe_stride_min,
      self.safe_stride_max,
    )
    stride_width_norm = torch.clamp(
      stride_width / (self.safe_stride_max - self.safe_stride_min),
      0.0,
      1.0,
    )
    confidence = torch.clamp(safe_stride_confidence, 0.0, 1.0)
    same_foot_actor_target_norm = self._normalize_semantic_value(
      self._same_foot_actor_stride_target(stair_shape, latent_obs),
      self.same_foot_stride_min,
      self.same_foot_stride_max,
    )
    shape_semantic = torch.cat(
      [
        same_foot_stride_norm,
        riser_height_norm,
        stride_lower_norm,
        stride_upper_norm,
        stride_center_norm,
        stride_width_norm,
        same_foot_actor_target_norm,
        confidence,
      ],
      dim=-1,
    )
    return torch.cat([state_semantic, shape_semantic], dim=-1)

  def _advance_gate_state(
    self,
    event_prob: torch.Tensor,
    stair_prob: torch.Tensor,
    gate_state: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    mode = gate_state[:, 0]
    was_in_memory = mode == _MODE_STAIR_MEMORY
    stair_timer = gate_state[:, 1]
    evidence_timer = gate_state[:, 2]
    write_timer = gate_state[:, 3]
    cooldown = torch.clamp(gate_state[:, 4] - 1.0, min=0.0)

    event_prob = event_prob.squeeze(-1)
    stair_prob = stair_prob.squeeze(-1)
    in_normal = mode == _MODE_NORMAL
    release_finished = in_normal & (write_timer < 0.0) & (cooldown <= 0.0)
    write_timer = torch.where(
      release_finished, torch.zeros_like(write_timer), write_timer
    )
    rearm_event = in_normal & (event_prob < self.event_off_threshold)
    evidence_timer = torch.where(
      rearm_event, torch.zeros_like(evidence_timer), evidence_timer
    )
    event_armed = evidence_timer >= 0.0
    trigger = (
      in_normal
      & event_armed
      & (cooldown <= 0.0)
      & (event_prob > self.event_on_threshold)
    )
    mode = torch.where(trigger, torch.full_like(mode, _MODE_STAIR_WRITE), mode)
    write_timer = torch.where(trigger, torch.zeros_like(write_timer), write_timer)
    stair_timer = torch.where(trigger, torch.zeros_like(stair_timer), stair_timer)
    evidence_timer = torch.where(
      trigger, torch.zeros_like(evidence_timer), evidence_timer
    )

    in_write = mode == _MODE_STAIR_WRITE
    write_active_for_update = in_write
    write_timer = torch.where(in_write, write_timer + 1.0, write_timer)
    stair_timer = torch.where(in_write, stair_timer + 1.0, stair_timer)
    stair_evidence = in_write & (stair_prob > self.stair_on_threshold)
    evidence_timer = torch.where(
      in_write,
      torch.where(
        stair_evidence,
        evidence_timer + 1.0,
        torch.zeros_like(evidence_timer),
      ),
      evidence_timer,
    )
    done_write = in_write & (write_timer >= self.write_steps)
    confirm_memory = done_write & (evidence_timer >= self.stair_confirm_steps)
    abort_write = done_write & ~confirm_memory
    mode = torch.where(confirm_memory, torch.full_like(mode, _MODE_STAIR_MEMORY), mode)
    mode = torch.where(abort_write, torch.full_like(mode, _MODE_NORMAL), mode)
    cooldown = torch.where(
      abort_write, torch.full_like(cooldown, self.cooldown_steps), cooldown
    )
    stair_timer = torch.where(abort_write, torch.zeros_like(stair_timer), stair_timer)
    evidence_timer = torch.where(
      confirm_memory,
      torch.zeros_like(evidence_timer),
      torch.where(abort_write, -torch.ones_like(evidence_timer), evidence_timer),
    )
    write_timer = torch.where(
      confirm_memory | abort_write,
      torch.zeros_like(write_timer),
      write_timer,
    )

    in_memory = mode == _MODE_STAIR_MEMORY
    memory_event = was_in_memory & in_memory & (event_prob > self.event_on_threshold)
    memory_boost_timer = torch.where(
      memory_event,
      torch.full_like(write_timer, self.memory_event_shape_boost_steps),
      torch.clamp(write_timer - 1.0, min=0.0),
    )
    write_timer = torch.where(in_memory, memory_boost_timer, write_timer)
    memory_shape_boost = in_memory & (write_timer > 0.0)
    stair_timer = torch.where(in_memory, stair_timer + 1.0, stair_timer)
    stair_off = stair_prob < self.stair_off_threshold
    if not self.stair_memory_exit_on_stair_off:
      stair_off = torch.zeros_like(stair_off)
    evidence_timer = torch.where(
      in_memory & stair_off,
      evidence_timer + 1.0,
      torch.where(in_memory, torch.zeros_like(evidence_timer), evidence_timer),
    )
    exit_memory = (
      in_memory
      & (evidence_timer >= self.exit_steps)
      & (stair_timer >= self.min_stair_steps)
    )
    mode = torch.where(exit_memory, torch.full_like(mode, _MODE_NORMAL), mode)
    cooldown = torch.where(
      exit_memory, torch.full_like(cooldown, self.cooldown_steps), cooldown
    )
    stair_timer = torch.where(exit_memory, torch.zeros_like(stair_timer), stair_timer)
    evidence_timer = torch.where(
      exit_memory, -torch.ones_like(evidence_timer), evidence_timer
    )
    write_timer = torch.where(exit_memory, -torch.ones_like(write_timer), write_timer)

    next_gate = torch.stack(
      [mode, stair_timer, evidence_timer, write_timer, cooldown], dim=-1
    )
    alpha = torch.full(
      (event_prob.shape[0], self.z_dim),
      self.alpha_fast,
      device=event_prob.device,
      dtype=event_prob.dtype,
    )
    alpha = torch.where(write_active_for_update[:, None], self.alpha_write, alpha)
    hold_alpha = torch.cat(
      [
        torch.full_like(alpha[:, : self.state_latent_dim], self.alpha_hold_state),
        torch.full_like(alpha[:, self.state_latent_dim :], self.alpha_hold_shape),
      ],
      dim=-1,
    )
    hold_memory = (mode == _MODE_STAIR_MEMORY) & ~write_active_for_update
    alpha = torch.where(hold_memory[:, None], hold_alpha, alpha)
    boosted_hold_alpha = torch.cat(
      [
        torch.full_like(
          alpha[:, : self.state_latent_dim],
          self.alpha_hold_state,
        ),
        torch.full_like(
          alpha[:, self.state_latent_dim :],
          self.alpha_fast,
        ),
      ],
      dim=-1,
    )
    alpha = torch.where(
      (hold_memory & memory_shape_boost)[:, None],
      boosted_hold_alpha,
      alpha,
    )
    release_active = (mode == _MODE_NORMAL) & (write_timer < 0.0) & (cooldown > 0.0)
    release_progress = torch.clamp(
      (self.cooldown_steps - cooldown) / max(self.cooldown_steps, 1.0),
      min=0.0,
      max=1.0,
    )
    release_alpha = hold_alpha + release_progress[:, None] * (
      self.alpha_fast - hold_alpha
    )
    alpha = torch.where(release_active[:, None], release_alpha, alpha)
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
    stair_prob = torch.sigmoid(self._stair_logit(h_t))
    gate_state_out, alpha = self._advance_gate_state(
      event_prob, stair_prob, gate_state_in
    )
    z_out = (1.0 - alpha) * z_in + alpha * z_candidate
    future_collision_risk = torch.sigmoid(self.future_collision_risk_head(z_out))
    future_safe_landing_quality = torch.sigmoid(
      self.future_safe_landing_quality_head(z_out)
    )
    shape_memory = z_out[:, self.state_latent_dim :]
    stair_shape = self._decode_stair_shape(shape_memory, h_t)
    safe_stride, safe_stride_interval, safe_stride_confidence = (
      self._decode_safe_stride_outputs(shape_memory, h_t, latent_obs)
    )
    actor_input_terms = [actor_obs_norm, z_out]
    if self.actor_semantic_enabled:
      actor_input_terms.append(
        self._build_actor_semantic(
          event_prob,
          stair_prob,
          gate_state_out,
          stair_shape,
          safe_stride_interval,
          safe_stride_confidence,
          latent_obs,
        )
      )
    actor_input = torch.cat(actor_input_terms, dim=-1)
    actions = self.deterministic_output(self.mlp(actor_input))
    return (
      actions,
      h_out,
      c_out,
      z_out,
      gate_state_out,
      event_prob,
      stair_prob,
      future_collision_risk,
      future_safe_landing_quality,
      stair_shape,
      safe_stride,
      safe_stride_interval,
      safe_stride_confidence,
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
