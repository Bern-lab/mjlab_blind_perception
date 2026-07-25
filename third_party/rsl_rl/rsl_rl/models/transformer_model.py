# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Transformer policy models for RSL-RL."""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import cast

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


def _term_major_flat_to_sequence(
    flat_obs: torch.Tensor,
    sequence_length: int,
    token_dims: tuple[int, ...],
) -> torch.Tensor:
    """Convert ObservationManager's term-major flat history to time-major tokens."""
    validate_shapes = not torch.onnx.is_in_onnx_export()
    if validate_shapes and flat_obs.dim() != 2:
        raise ValueError(f"CausalTransformerModel expects 2D flat observations, got {flat_obs.shape}.")

    if token_dims:
        expected_dim = sequence_length * sum(token_dims)
        if validate_shapes and flat_obs.shape[-1] != expected_dim:
            raise ValueError(
                "Observation dimension does not match transformer tokenization: "
                f"got {flat_obs.shape[-1]}, expected {expected_dim} from "
                f"sequence_length={sequence_length} and token_dims={token_dims}."
            )

        batch_size = flat_obs.shape[0]
        pieces = []
        offset = 0
        for token_dim in token_dims:
            width = sequence_length * token_dim
            term_history = flat_obs[:, offset : offset + width]
            pieces.append(term_history.reshape(batch_size, sequence_length, token_dim))
            offset += width
        return torch.cat(pieces, dim=-1)

    if validate_shapes and flat_obs.shape[-1] % sequence_length != 0:
        raise ValueError(
            f"Observation dimension {flat_obs.shape[-1]} is not divisible by sequence_length={sequence_length}."
        )
    token_dim = flat_obs.shape[-1] // sequence_length
    return flat_obs.reshape(flat_obs.shape[0], sequence_length, token_dim)


def _build_sinusoidal_position_embedding(
    sequence_length: int,
    d_model: int,
) -> torch.Tensor:
    """Build fixed sinusoidal position encodings with shape ``(1, T, D)``."""
    position = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
    even_dimensions = torch.arange(0, d_model, 2, dtype=torch.float32)
    div_term = torch.exp(even_dimensions * (-math.log(10000.0) / d_model))
    embedding = torch.zeros(sequence_length, d_model, dtype=torch.float32)
    embedding[:, 0::2] = torch.sin(position * div_term)
    if d_model > 1:
        odd_width = embedding[:, 1::2].shape[1]
        embedding[:, 1::2] = torch.cos(position * div_term[:odd_width])
    return embedding.unsqueeze(0)


class CausalTransformerModel(nn.Module):
    """Causal transformer actor over a fixed observation-action history.

    The actor consumes the same flattened history produced by ``ObservationManager``
    for the MLP/LSTM ablations. For the G1 task, each token contains one timestep of
    proprioceptive observations plus the corresponding ``last_action`` term, which
    follows the humanoid-transformer idea of predicting the next action from
    observation-action history.
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
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        transformer_cfg: dict | None = None,
    ) -> None:
        """Initialize a causal transformer model."""
        super().__init__()

        transformer_cfg = dict(transformer_cfg or {})
        sequence_length_cfg = transformer_cfg.pop("sequence_length", None)
        self.token_dims = tuple(int(dim) for dim in transformer_cfg.pop("token_dims", ()))
        self.d_model = int(transformer_cfg.pop("d_model", 216))
        num_heads = int(transformer_cfg.pop("num_heads", 4))
        num_layers = int(transformer_cfg.pop("num_layers", 2))
        dim_feedforward = int(transformer_cfg.pop("dim_feedforward", 432))
        dropout = float(transformer_cfg.pop("dropout", 0.0))
        transformer_activation = str(transformer_cfg.pop("transformer_activation", "gelu"))
        self.pooling = str(transformer_cfg.pop("pooling", "last"))
        norm_first = bool(transformer_cfg.pop("norm_first", True))
        input_projection_hidden_dims = tuple(
            int(dim) for dim in transformer_cfg.pop("input_projection_hidden_dims", ())
        )
        input_projection_activation = str(transformer_cfg.pop("input_projection_activation", activation))
        position_encoding = str(transformer_cfg.pop("position_encoding", "learned"))
        if transformer_cfg:
            unknown_keys = ", ".join(sorted(transformer_cfg))
            raise ValueError(f"Unknown CausalTransformerModel config keys: {unknown_keys}")

        if self.d_model % num_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by num_heads={num_heads}.")
        if self.pooling not in {"last", "mean"}:
            raise ValueError(f"Unsupported transformer pooling mode: {self.pooling}")
        if position_encoding not in {"learned", "sinusoidal"}:
            raise ValueError(f"Unsupported transformer position encoding: {position_encoding}")

        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        if sequence_length_cfg is None:
            if self.token_dims:
                sequence_token_dim = sum(self.token_dims)
                if self.obs_dim % sequence_token_dim != 0:
                    raise ValueError(
                        f"obs_dim={self.obs_dim} is not divisible by token_dims sum "
                        f"{sequence_token_dim}; pass an explicit sequence_length if needed."
                    )
                self.sequence_length = self.obs_dim // sequence_token_dim
            else:
                self.sequence_length = 1
        else:
            self.sequence_length = int(sequence_length_cfg)

        if self.sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {self.sequence_length}")
        if self.token_dims:
            self.sequence_token_dim = sum(self.token_dims)
        else:
            self.sequence_token_dim = self.obs_dim // self.sequence_length
        if self.token_dims and self.obs_dim != self.sequence_length * self.sequence_token_dim:
            raise ValueError(
                f"obs_dim={self.obs_dim} does not match sequence_length={self.sequence_length} "
                f"and token_dims={self.token_dims}."
            )

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = nn.Identity()

        if distribution_cfg is not None:
            dist_cfg = dict(distribution_cfg)
            dist_class: type[Distribution] = resolve_callable(dist_cfg.pop("class_name"))  # type: ignore[assignment]
            self.distribution: Distribution | None = dist_class(output_dim, **dist_cfg)
            head_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            head_output_dim = output_dim

        if input_projection_hidden_dims:
            self.input_projection = MLP(
                self.sequence_token_dim,
                self.d_model,
                input_projection_hidden_dims,
                input_projection_activation,
            )
        else:
            self.input_projection = nn.Linear(self.sequence_token_dim, self.d_model)
        if position_encoding == "sinusoidal":
            self.register_buffer(
                "position_embedding",
                _build_sinusoidal_position_embedding(
                    self.sequence_length,
                    self.d_model,
                ),
            )
        else:
            self.position_embedding = nn.Parameter(torch.zeros(1, self.sequence_length, self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=transformer_activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        causal_mask = torch.triu(
            torch.ones(self.sequence_length, self.sequence_length, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask)
        self.head = MLP(self.d_model, head_output_dim, hidden_dims, activation)

        if isinstance(self.position_embedding, nn.Parameter):
            nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.head)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the transformer actor."""
        obs_td = cast(TensorDict, unpad_trajectories(obs, masks)) if masks is not None else obs
        latent = self.get_latent(obs_td, masks, hidden_state)
        head_output = self.head(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(head_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(head_output)
        return head_output

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Encode the selected observation groups into a policy latent."""
        del masks, hidden_state
        obs_list = [cast(torch.Tensor, obs[obs_group]) for obs_group in self.obs_groups]
        flat_obs = torch.cat(obs_list, dim=-1)
        flat_obs = self.obs_normalizer(flat_obs)
        sequence = _term_major_flat_to_sequence(
            flat_obs,
            sequence_length=self.sequence_length,
            token_dims=self.token_dims,
        )
        tokens = self.input_projection(sequence) * math.sqrt(self.d_model)
        tokens = tokens + self.position_embedding[:, : self.sequence_length]
        encoded = self.encoder(tokens, mask=self.causal_mask)
        if self.pooling == "mean":
            return encoded.mean(dim=1)
        return encoded[:, -1]

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset recurrent state; transformer actor is feedforward over fixed history."""
        del dones, hidden_state

    def get_hidden_state(self) -> HiddenState:
        """Return recurrent hidden state, unused for this feedforward transformer."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach recurrent hidden state, unused for this feedforward transformer."""
        del dones

    @property
    def output_mean(self) -> torch.Tensor:
        """Return the mean of the current output distribution."""
        assert self.distribution is not None
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        """Return the standard deviation of the current output distribution."""
        assert self.distribution is not None
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return the entropy of the current output distribution."""
        assert self.distribution is not None
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return raw parameters of the current output distribution."""
        assert self.distribution is not None
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute log-probabilities of outputs under the current distribution."""
        assert self.distribution is not None
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Compute KL divergence between two distribution parameterizations."""
        assert self.distribution is not None
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchCausalTransformerModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxCausalTransformerModel(self, verbose)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation-normalization statistics from a batch."""
        if self.obs_normalization:
            obs_list = [cast(torch.Tensor, obs[obs_group]) for obs_group in self.obs_groups]
            flat_obs = torch.cat(obs_list, dim=-1)
            self.obs_normalizer.update(flat_obs)  # type: ignore[attr-defined]

    def _get_obs_dim(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
    ) -> tuple[list[str], int]:
        """Select active observation groups and compute flat observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    "The causal transformer actor only supports flattened 1D "
                    f"observations, got shape {obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim


class _TorchCausalTransformerModel(nn.Module):
    """Exportable transformer model for JIT."""

    def __init__(self, model: CausalTransformerModel) -> None:
        super().__init__()
        self.sequence_length = model.sequence_length
        self.token_dims = model.token_dims
        self.d_model = model.d_model
        self.pooling = model.pooling
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.input_projection = copy.deepcopy(model.input_projection)
        position_embedding = cast(torch.Tensor, model.position_embedding)
        self.register_buffer("position_embedding", position_embedding.detach().clone())
        self.encoder = copy.deepcopy(model.encoder)
        self.head = copy.deepcopy(model.head)
        causal_mask = cast(torch.Tensor, model.causal_mask)
        self.register_buffer("causal_mask", causal_mask.detach().clone())
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference on pre-concatenated observations."""
        x = self.obs_normalizer(x)
        sequence = _term_major_flat_to_sequence(x, self.sequence_length, self.token_dims)
        tokens = self.input_projection(sequence) * math.sqrt(self.d_model)
        tokens = tokens + self.position_embedding[:, : self.sequence_length]
        encoded = self.encoder(tokens, mask=self.causal_mask)
        latent = encoded.mean(dim=1) if self.pooling == "mean" else encoded[:, -1]
        out = self.head(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state, unused for transformer exports."""
        pass


class _OnnxCausalTransformerModel(_TorchCausalTransformerModel):
    """Exportable transformer model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: CausalTransformerModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.obs_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
