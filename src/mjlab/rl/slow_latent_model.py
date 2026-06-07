"""Gated stair-focused slow latent model with auxiliary stair prediction heads.

Architecture::

    latent_obs (~91d) ──► latent_obs_normalizer ──► MLP encoder (128,128)
        │
        ▼
    LSTM (input=128, hidden=128)
        │
        ├──► h_t (128d) ──► event_head (128→64→1)        → p_event (deployable)
        │
        ├──► z_candidate_head (128→16)
        │         │
        │         ▼
        │    gated update: z = (1-α)*z_old + α*z_candidate
        │         │
        │         ▼
        │    z_memory (16d)
        │         │
        │         ├──► future_collision_head (16→64→1)     → p_future_collision
        │         │
        │         └──► concat(h_t, z_memory) ──►
        │                   stair_state_head (144→64→1)    → p_stair
        │
        ▼
    concat(actor_obs, z_memory) ──► actor MLP ──► action distribution

Gate state machine (internal, vectorized per env)::

    NORMAL:        α = alpha_fast  (0.3)
                   on p_event > 0.6 & cooldown==0 → STAIR_WRITE

    STAIR_WRITE:   α = alpha_write (0.8)
                   after write_steps → STAIR_MEMORY

    STAIR_MEMORY:  α = alpha_hold (0.02)
                   if p_stair < 0.4 & no_event_timer > exit_steps
                     & stair_timer > min_stair_steps → NORMAL

Training labels (simulation-only, never enter actor/latent_obs):
  - toe_riser_event_label:  toe blocking force > threshold (from contact sensor)
  - stair_state_label:      currently inside a stair terrain section & post-first-hit
  - future_collision_label: max(toe_riser_event[t+1:t+K+1]), K=20
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState


class LSTMSlowLatentMLPModel(MLPModel):
    """LSTM-based slow-latent policy with gated memory and auxiliary heads.

    The model separates deployable stair-focused proprioceptive features
    (``latent_obs``) from the full actor observation (``actor_obs``).  A
    lightweight MLP encoder + LSTM pathway compresses ``latent_obs`` into a
    compact 16-dimensional memory ``z_memory``.  An event-driven gate controls
    how fast the memory is updated, with three auxiliary heads that predict
    current toe-riser events, persistent stair state, and future collisions.

    Parameters
    ----------
    mlp_encoder_dims:
        Hidden dimensions for the pre-LSTM MLP encoder, e.g. ``(128, 128)``.
    latent_hidden_dim:
        Hidden/cell dimension of the LSTM (default 128).
    z_dim:
        Dimension of the gated latent memory ``z_memory`` (default 16).
    alpha_fast / alpha_write / alpha_hold:
        Gating update rates for the NORMAL, STAIR_WRITE, and STAIR_MEMORY modes.
    write_steps / min_stair_steps / exit_steps / cooldown_steps:
        Timing parameters for the gate state machine.
    event_on_threshold / event_off_threshold / stair_off_threshold:
        Prediction thresholds used by the state machine during inference.
    latent_obs_set:
        Name of the observation set that contains the latent observations
        (default ``"latent"``).
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
        # ---- latent pathway ----
        mlp_encoder_dims: tuple[int, ...] | list[int] = (128, 128),
        latent_hidden_dim: int = 128,
        z_dim: int = 16,
        # ---- gate ----
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
        # ---- aux coefs (informational, loss is applied externally) ----
        aux_future_collision_coef: float = 0.05,
        aux_event_coef: float = 0.03,
        aux_stair_coef: float = 0.02,
        future_collision_horizon: int = 20,
        latent_obs_set: str = "latent",
        **kwargs: Any,
    ) -> None:
        # ------------------------------------------------------------------
        # Parent initialisation (sets up obs_normalizer, MLP, distribution)
        # ------------------------------------------------------------------
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )

        _activation_fn = nn.ELU

        # ------------------------------------------------------------------
        # Latent observation config
        # ------------------------------------------------------------------
        self.latent_obs_set = latent_obs_set
        self._latent_obs_group_names: list[str] = obs_groups.get(latent_obs_set, [])
        self._latent_obs_dim: int = 0
        for name in self._latent_obs_group_names:
            self._latent_obs_dim += obs[name].shape[-1]

        # Separate normalizer for latent observations
        if obs_normalization:
            from rsl_rl.modules import EmpiricalNormalization
            self.latent_obs_normalizer = EmpiricalNormalization(self._latent_obs_dim)
        else:
            self.latent_obs_normalizer = nn.Identity()

        # ------------------------------------------------------------------
        # MLP encoder: latent_obs_dim -> mlp_encoder_dims[-1] (=128)
        # ------------------------------------------------------------------
        encoder_layers: list[nn.Module] = []
        in_dim = self._latent_obs_dim
        for hdim in mlp_encoder_dims:
            encoder_layers.append(nn.Linear(in_dim, hdim))
            encoder_layers.append(_activation_fn())
            in_dim = hdim
        self.latent_encoder = nn.Sequential(*encoder_layers)
        encoder_out_dim = mlp_encoder_dims[-1] if mlp_encoder_dims else self._latent_obs_dim

        # ------------------------------------------------------------------
        # LSTM: input = encoder_out_dim (128), hidden = latent_hidden_dim (128)
        # ------------------------------------------------------------------
        self.latent_lstm = nn.LSTM(
            input_size=encoder_out_dim,
            hidden_size=latent_hidden_dim,
            num_layers=1,
            batch_first=False,
        )

        # ------------------------------------------------------------------
        # Heads
        # ------------------------------------------------------------------
        # z_candidate head: LSTM hidden → z_dim
        self.z_candidate_head = nn.Sequential(
            nn.Linear(latent_hidden_dim, z_dim),
        )

        # current event head: LSTM hidden → 1
        self.event_head = nn.Sequential(
            nn.Linear(latent_hidden_dim, 64),
            _activation_fn(),
            nn.Linear(64, 1),
        )

        # future collision head: z_memory → 1
        self.future_collision_head = nn.Sequential(
            nn.Linear(z_dim, 64),
            _activation_fn(),
            nn.Linear(64, 1),
        )

        # stair state head: concat(LSTM hidden, z_memory) → 1
        self.stair_state_head = nn.Sequential(
            nn.Linear(latent_hidden_dim + z_dim, 64),
            _activation_fn(),
            nn.Linear(64, 1),
        )

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.z_dim = z_dim
        self.latent_hidden_dim = latent_hidden_dim

        # Gate parameters
        self.alpha_fast = alpha_fast
        self.alpha_write = alpha_write
        self.alpha_hold = alpha_hold
        self.write_steps = write_steps
        self.min_stair_steps = min_stair_steps
        self.exit_steps = exit_steps
        self.cooldown_steps = cooldown_steps
        self.event_on_threshold = event_on_threshold
        self.event_off_threshold = event_off_threshold
        self.stair_off_threshold = stair_off_threshold

        # Aux loss coefs (informational)
        self.aux_future_collision_coef = aux_future_collision_coef
        self.aux_event_coef = aux_event_coef
        self.aux_stair_coef = aux_stair_coef
        self.future_collision_horizon = future_collision_horizon

        # ------------------------------------------------------------------
        # Runtime buffers (set in _init_buffers when num_envs is known)
        # ------------------------------------------------------------------
        self.num_envs: int = 0
        # z_memory: [num_envs, z_dim]
        # gate_mode: [num_envs]  int tensor (0=NORMAL, 1=STAIR_WRITE, 2=STAIR_MEMORY)
        # stair_timer: [num_envs] int tensor
        # no_event_timer: [num_envs] int tensor
        # write_timer: [num_envs] int tensor
        # cooldown: [num_envs] int tensor
        self._buffers_initialized = False

        # Stash aux outputs from forward for loss computation
        self._aux_event_logits: torch.Tensor | None = None
        self._aux_stair_logits: torch.Tensor | None = None
        self._aux_future_collision_logits: torch.Tensor | None = None
        self._z_candidate: torch.Tensor | None = None
        self._h_t: torch.Tensor | None = None

    # ==================================================================
    # Buffer management
    # ==================================================================

    def _init_buffers(self, num_envs: int, device: torch.device) -> None:
        """Allocate or reset gate-state and z-memory buffers."""
        self.num_envs = num_envs
        self.register_buffer(
            "z_memory", torch.zeros(num_envs, self.z_dim, device=device), persistent=False
        )
        self.register_buffer(
            "gate_mode", torch.zeros(num_envs, dtype=torch.int32, device=device), persistent=False
        )
        self.register_buffer(
            "stair_timer", torch.zeros(num_envs, dtype=torch.int32, device=device), persistent=False
        )
        self.register_buffer(
            "no_event_timer", torch.zeros(num_envs, dtype=torch.int32, device=device), persistent=False
        )
        self.register_buffer(
            "write_timer", torch.zeros(num_envs, dtype=torch.int32, device=device), persistent=False
        )
        self.register_buffer(
            "cooldown", torch.zeros(num_envs, dtype=torch.int32, device=device), persistent=False
        )
        self._buffers_initialized = True

    def reset_buffers(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset z-memory, gate state, and LSTM hidden state for given env IDs."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.z_memory.device)
        elif env_ids.numel() == 0:
            return
        self.z_memory[env_ids] = 0.0
        self.gate_mode[env_ids] = 0
        self.stair_timer[env_ids] = 0
        self.no_event_timer[env_ids] = 0
        self.write_timer[env_ids] = 0
        self.cooldown[env_ids] = 0

    # ==================================================================
    # Observation helpers
    # ==================================================================

    def _extract_latent_obs(self, obs: TensorDict) -> torch.Tensor:
        """Concatenate latent observation groups from a TensorDict."""
        parts: list[torch.Tensor] = []
        for name in self._latent_obs_group_names:
            parts.append(obs[name])
        return torch.cat(parts, dim=-1)

    def _extract_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """Concatenate actor observation groups from a TensorDict."""
        parts: list[torch.Tensor] = []
        for name in self.obs_groups[self.obs_set]:
            parts.append(obs[name])
        return torch.cat(parts, dim=-1)

    # ==================================================================
    # Gate state machine
    # ==================================================================

    def _advance_gate(
        self,
        toe_riser_event: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance the internal gate state machine and return alpha tensor.

        Parameters
        ----------
        toe_riser_event:
            Binary tensor ``[num_envs]`` (or ``[num_envs, 1]``).  1 indicates
            a toe-riser collision event at the current timestep.  During
            training this comes from the simulation contact sensor.
        env_ids:
            Optional subset of env IDs to advance (used for reset masking).

        Returns
        -------
        alpha:
            ``[num_envs, 1]`` update rate for the gated low-pass filter.
        """
        if toe_riser_event.dim() > 1:
            toe_riser_event = toe_riser_event.squeeze(-1)
        event = toe_riser_event.bool()

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.z_memory.device)

        if env_ids.numel() == 0:
            return torch.full((0, 1), self.alpha_fast, device=self.z_memory.device)

        mode = self.gate_mode
        stair_timer = self.stair_timer
        no_event_timer = self.no_event_timer
        write_timer = self.write_timer
        cooldown = self.cooldown

        # ---- Decrement cooldown ----
        cooldown = torch.clamp(cooldown - 1, min=0)

        # ---- NORMAL -> STAIR_WRITE ----
        trigger = event & (cooldown == 0)
        mode = torch.where(trigger, torch.full_like(mode, 1), mode)
        write_timer = torch.where(trigger, torch.zeros_like(write_timer), write_timer)
        stair_timer = torch.where(trigger, torch.zeros_like(stair_timer), stair_timer)
        no_event_timer = torch.where(trigger, torch.zeros_like(no_event_timer), no_event_timer)

        # ---- STAIR_WRITE -> STAIR_MEMORY ----
        in_write = mode == 1
        write_timer = torch.where(in_write, write_timer + 1, write_timer)
        stair_timer = torch.where(in_write, stair_timer + 1, stair_timer)
        cooldown = torch.where(
            in_write, torch.full_like(cooldown, self.cooldown_steps), cooldown
        )
        done_write = in_write & (write_timer >= self.write_steps)
        mode = torch.where(done_write, torch.full_like(mode, 2), mode)

        # ---- STAIR_MEMORY ----
        in_memory = mode == 2
        stair_timer = torch.where(in_memory, stair_timer + 1, stair_timer)
        # Check p_stair for exit (use prediction if available, else keep memory)
        # During training with hard gate: exit based on no_event timeout only
        no_event_timer = torch.where(
            in_memory & event,
            torch.zeros_like(no_event_timer),
            torch.where(in_memory, no_event_timer + 1, no_event_timer),
        )
        # Exit: long enough in memory + long enough without events
        exit_memory = (
            in_memory
            & (stair_timer > self.min_stair_steps)
            & (no_event_timer > self.exit_steps)
        )
        mode = torch.where(exit_memory, torch.zeros_like(mode), mode)
        cooldown = torch.where(
            exit_memory, torch.full_like(cooldown, self.cooldown_steps), cooldown
        )

        # ---- Compute alpha ----
        alpha = torch.full(
            (self.num_envs, 1),
            self.alpha_fast,
            device=self.z_memory.device,
            dtype=torch.float32,
        )
        alpha[mode == 1] = self.alpha_write
        alpha[mode == 2] = self.alpha_hold

        # ---- Write back ----
        self.gate_mode = mode
        self.stair_timer = stair_timer
        self.no_event_timer = no_event_timer
        self.write_timer = write_timer
        self.cooldown = cooldown

        return alpha

    # ==================================================================
    # Gated update
    # ==================================================================

    def _gated_update(
        self, z_candidate: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        """Apply the gated EMA update: z = (1-α) * z_old + α * z_candidate."""
        z_new = (1.0 - alpha) * self.z_memory + alpha * z_candidate
        self.z_memory = z_new
        return z_new

    # ==================================================================
    # Forward
    # ==================================================================

    def _process_latent_pathway(
        self,
        latent_obs_flat: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run the latent encoder + LSTM and return (h_t, (h_out, c_out))."""
        # Normalize
        latent_obs_norm = self.latent_obs_normalizer(latent_obs_flat)
        # MLP encoder
        encoded = self.latent_encoder(latent_obs_norm)
        # LSTM expects [seq_len, batch, input_dim]; we have [batch, input_dim]
        if encoded.dim() == 2:
            encoded = encoded.unsqueeze(0)  # [1, num_envs, dim]
        # LSTM forward
        lstm_out, (h_out, c_out) = self.latent_lstm(encoded, hidden_state)
        # lstm_out shape: [seq_len, batch, hidden_dim] → squeeze seq dim
        h_t = lstm_out.squeeze(0)  # [num_envs, hidden_dim]
        return h_t, (h_out, c_out)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        toe_riser_event: torch.Tensor | None = None,
        stochastic_output: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        """Full forward pass producing actions and auxiliary predictions.

        Parameters
        ----------
        obs:
            Observation TensorDict containing both actor and latent groups.
        masks:
            Sequence masks for recurrent processing.
        hidden_state:
            Tuple ``(h_in, c_in)`` for the LSTM, or ``None`` to use internal state.
        toe_riser_event:
            Optional ``[num_envs]`` or ``[num_envs, 1]`` binary tensor with
            toe-riser collision labels.  Used to advance the gate state machine
            during training.  If ``None``, the gate state machine is not
            advanced and ``alpha_fast`` is used.
        stochastic_output:
            If ``True``, sample from the output distribution.  If ``False``,
            return the distribution mean.

        Returns
        -------
        output:
            Deterministic actions or distribution parameters, depending on
            ``stochastic_output``.
        """
        # Ensure buffers are initialised
        batch_size = obs.batch_size[0]
        _device = obs.device if obs.device is not None else torch.device("cpu")
        if not self._buffers_initialized or self.num_envs != batch_size:
            self._init_buffers(batch_size, _device)

        # Extract observations
        actor_obs_flat = self._extract_actor_obs(obs)
        latent_obs_flat = self._extract_latent_obs(obs)

        # Normalise actor obs
        actor_obs_norm = self.obs_normalizer(actor_obs_flat)

        # ---- Latent pathway ----
        # Handle hidden state unpacking
        if hidden_state is not None:
            if isinstance(hidden_state, (list, tuple)):
                h_in, c_in = hidden_state
            else:
                h_in = hidden_state
                c_in = torch.zeros_like(h_in)
        else:
            h_in = torch.zeros(
                1, batch_size, self.latent_hidden_dim, device=obs.device
            )
            c_in = torch.zeros_like(h_in)

        h_t, (h_out, c_out) = self._process_latent_pathway(
            latent_obs_flat, masks, (h_in, c_in)
        )

        # ---- z_candidate ----
        z_candidate = self.z_candidate_head(h_t)

        # ---- Gate state machine ----
        if toe_riser_event is not None:
            alpha = self._advance_gate(toe_riser_event)
        else:
            # During inference without event, use current gate mode
            alpha = torch.full(
                (batch_size, 1),
                self.alpha_fast,
                device=obs.device,
                dtype=torch.float32,
            )
            alpha[self.gate_mode == 1] = self.alpha_write
            alpha[self.gate_mode == 2] = self.alpha_hold

        # ---- Gated update ----
        z_memory = self._gated_update(z_candidate, alpha)

        # ---- Auxiliary heads ----
        event_logit = self.event_head(h_t)
        stair_logit = self.stair_state_head(torch.cat([h_t, z_memory], dim=-1))
        future_collision_logit = self.future_collision_head(z_memory)

        # Stash for loss computation
        self._aux_event_logits = event_logit
        self._aux_stair_logits = stair_logit
        self._aux_future_collision_logits = future_collision_logit
        self._z_candidate = z_candidate
        self._h_t = h_t

        # ---- Actor pathway: concat(actor_obs_norm, z_memory) → MLP ----
        actor_input = torch.cat([actor_obs_norm, z_memory], dim=-1)
        # Override parent's internal latent for MLP forward
        # (parent's get_latent would normally compute rnn_latent; we bypass it)
        actor_latent = self.mlp(actor_input)

        # Distribution
        if self.distribution is not None:
            output = self.distribution(actor_latent, stochastic_output)
        else:
            output = self.deterministic_output(actor_latent) if hasattr(self, 'deterministic_output') else actor_latent

        # Store LSTM hidden state for external use
        self._lstm_hidden = (h_out, c_out)

        return output

    # ==================================================================
    # Legacy get_latent override (called by parent pipeline)
    # ==================================================================
    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Override parent get_latent to return z_memory for the actor MLP.

        The parent MLPModel's get_latent is called by the forward logic to
        build the latent that is passed to the MLP head.  We return the
        gated z_memory concatenated with actor obs (already normalised).
        """
        actor_obs_flat = self._extract_actor_obs(obs)
        actor_obs_norm = self.obs_normalizer(actor_obs_flat)
        return torch.cat([actor_obs_norm, self.z_memory], dim=-1)

    # ==================================================================
    # Aux output accessors
    # ==================================================================
    @property
    def aux_event_logits(self) -> torch.Tensor | None:
        """(batch, 1) logits from the current-event head."""
        return self._aux_event_logits

    @property
    def aux_stair_logits(self) -> torch.Tensor | None:
        """(batch, 1) logits from the stair-state head."""
        return self._aux_stair_logits

    @property
    def aux_future_collision_logits(self) -> torch.Tensor | None:
        """(batch, 1) logits from the future-collision head."""
        return self._aux_future_collision_logits

    def get_aux_outputs(self) -> dict[str, torch.Tensor]:
        """Return a dict of auxiliary predictions for loss computation."""
        aux: dict[str, torch.Tensor] = {}
        if self._aux_event_logits is not None:
            aux["event_logit"] = self._aux_event_logits
        if self._aux_stair_logits is not None:
            aux["stair_logit"] = self._aux_stair_logits
        if self._aux_future_collision_logits is not None:
            aux["future_collision_logit"] = self._aux_future_collision_logits
        return aux

    # ==================================================================
    # Hidden state management (for RSL-RL recurrent storage)
    # ==================================================================
    def get_hidden_state(self) -> HiddenState:
        """Return the internal LSTM hidden state for storage."""
        if hasattr(self, "_lstm_hidden"):
            return self._lstm_hidden
        h = torch.zeros(
            1, self.num_envs, self.latent_hidden_dim, device=self.z_memory.device
        )
        c = torch.zeros_like(h)
        return (h, c)

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach LSTM hidden state and reset buffers for done envs."""
        if hasattr(self, "_lstm_hidden"):
            h, c = self._lstm_hidden
            # detach for truncated BPTT
            self._lstm_hidden = (h.detach(), c.detach())
        # Reset z_memory and gate state for done envs
        if dones is not None:
            done_ids = dones.squeeze(-1).nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                self.reset_buffers(done_ids)

    # ==================================================================
    # ONNX export
    # ==================================================================
    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return an ONNX-exportable version of the model.

        The exported model accepts:
          - actor_obs:    [1, actor_obs_dim]
          - latent_obs:   [1, latent_obs_dim]
          - h_in:         [1, 1, latent_hidden_dim]
          - c_in:         [1, 1, latent_hidden_dim]
          - z_in:         [1, z_dim]
          - gate_mode_in: [1] int

        And returns:
          - actions:                  [1, output_dim]
          - h_out:                    [1, 1, latent_hidden_dim]
          - c_out:                    [1, 1, latent_hidden_dim]
          - z_out:                    [1, z_dim]
          - event_prob:               [1, 1]
          - stair_prob:               [1, 1]
          - future_collision_prob:    [1, 1]
        """
        return _OnnxStairLatentModel(self, verbose)

    # ==================================================================
    # Dummy inputs for ONNX export tracing
    # ==================================================================
    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        """Return dummy inputs for ONNX export."""
        return {
            "actor_obs": torch.randn(1, self.obs_dim),
            "latent_obs": torch.randn(1, self._latent_obs_dim),
            "h_in": torch.zeros(1, 1, self.latent_hidden_dim),
            "c_in": torch.zeros(1, 1, self.latent_hidden_dim),
            "z_in": torch.zeros(1, self.z_dim),
            "gate_mode_in": torch.zeros(1, dtype=torch.int32),
        }


# ======================================================================
# ONNX-exportable wrapper
# ======================================================================

class _OnnxStairLatentModel(nn.Module):
    """ONNX-exportable wrapper that re-implements the inference graph.

    This class copies the relevant sub-modules from the parent
    ``LSTMSlowLatentMLPModel`` and provides a single ``forward`` call
    that is compatible with ``torch.onnx.export``.
    """

    is_recurrent: bool = True

    def __init__(self, model: LSTMSlowLatentMLPModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose

        # Copy sub-modules
        self.actor_obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.latent_obs_normalizer = copy.deepcopy(model.latent_obs_normalizer)
        self.latent_encoder = copy.deepcopy(model.latent_encoder)
        self.latent_lstm = copy.deepcopy(model.latent_lstm)
        self.z_candidate_head = copy.deepcopy(model.z_candidate_head)
        self.event_head = copy.deepcopy(model.event_head)
        self.future_collision_head = copy.deepcopy(model.future_collision_head)
        self.stair_state_head = copy.deepcopy(model.stair_state_head)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        # Gate parameters
        self.alpha_fast = model.alpha_fast
        self.alpha_write = model.alpha_write
        self.alpha_hold = model.alpha_hold
        self.z_dim = model.z_dim
        self.latent_hidden_dim = model.latent_hidden_dim

        # Move to CPU for export
        self.latent_lstm.cpu()

    @property
    def input_names(self) -> list[str]:
        return ["actor_obs", "latent_obs", "h_in", "c_in", "z_in", "gate_mode_in"]

    @property
    def output_names(self) -> list[str]:
        return [
            "actions",
            "h_out",
            "c_out",
            "z_out",
            "event_prob",
            "stair_prob",
            "future_collision_prob",
        ]

    def forward(
        self,
        actor_obs: torch.Tensor,
        latent_obs: torch.Tensor,
        h_in: torch.Tensor,
        c_in: torch.Tensor,
        z_in: torch.Tensor,
        gate_mode_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One inference step for ONNX deployment.

        All inputs have batch=1 (single env, single timestep).

        Parameters
        ----------
        actor_obs: [1, actor_obs_dim]
        latent_obs: [1, latent_obs_dim]
        h_in: [1, 1, latent_hidden_dim]
        c_in: [1, 1, latent_hidden_dim]
        z_in: [1, z_dim]
        gate_mode_in: [1] int (0=NORMAL, 1=STAIR_WRITE, 2=STAIR_MEMORY)

        Returns
        -------
        actions: [1, output_dim]
        h_out: [1, 1, latent_hidden_dim]
        c_out: [1, 1, latent_hidden_dim]
        z_out: [1, z_dim]
        event_prob: [1, 1]
        stair_prob: [1, 1]
        future_collision_prob: [1, 1]
        """
        # Normalize
        actor_obs_norm = self.actor_obs_normalizer(actor_obs)
        latent_obs_norm = self.latent_obs_normalizer(latent_obs)

        # Latent pathway
        encoded = self.latent_encoder(latent_obs_norm)
        encoded = encoded.unsqueeze(0)  # [1, 1, dim]
        lstm_out, (h_out, c_out) = self.latent_lstm(encoded, (h_in, c_in))
        h_t = lstm_out.squeeze(0)  # [1, hidden_dim]

        # z_candidate
        z_candidate = self.z_candidate_head(h_t)

        # Gated update (external gate state machine expected)
        # For ONNX, caller passes gate_mode_in; we compute alpha
        alpha = torch.where(
            gate_mode_in == 1,
            torch.full_like(z_candidate[:, :1], self.alpha_write),
            torch.where(
                gate_mode_in == 2,
                torch.full_like(z_candidate[:, :1], self.alpha_hold),
                torch.full_like(z_candidate[:, :1], self.alpha_fast),
            ),
        )
        z_out = (1.0 - alpha) * z_in + alpha * z_candidate

        # Aux heads
        event_prob = torch.sigmoid(self.event_head(h_t))
        stair_prob = torch.sigmoid(self.stair_state_head(torch.cat([h_t, z_out], dim=-1)))
        future_collision_prob = torch.sigmoid(self.future_collision_head(z_out))

        # Actor pathway
        actor_input = torch.cat([actor_obs_norm, z_out], dim=-1)
        actor_latent = self.mlp(actor_input)
        actions = self.deterministic_output(actor_latent)

        return actions, h_out, c_out, z_out, event_prob, stair_prob, future_collision_prob