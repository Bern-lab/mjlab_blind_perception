"""Tests for the gated stair slow-latent actor model."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast

import onnx
import pytest
import torch
from rsl_rl.algorithms.ppo_teacher_kl import PPOTeacherKL
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
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


def _make_model(
  *,
  structured_safe_stride_enabled: bool = False,
  dynamic_safe_stride_enabled: bool = False,
  shadow_semantic_enabled: bool = False,
) -> LSTMSlowLatentMLPModel:
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
    state_latent_dim=2,
    alpha_fast=0.3,
    alpha_write=0.8,
    alpha_hold_state=0.0,
    alpha_hold_shape=0.05,
    aux_safe_stride_coef=0.03,
    structured_safe_stride_enabled=structured_safe_stride_enabled,
    dynamic_safe_stride_enabled=dynamic_safe_stride_enabled,
    safe_stride_phase_dim=2 if dynamic_safe_stride_enabled else 0,
    shadow_semantic_enabled=shadow_semantic_enabled,
  )


def test_slow_latent_actor_forward_updates_state_and_aux_outputs() -> None:
  model = _make_model()
  obs = _make_obs()

  assert model.stair_confirm_steps == 3.0
  assert model.stair_off_threshold == 0.20
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
  assert aux["future_collision_risk_logit"].shape == (4, 1)
  assert aux["future_safe_landing_quality_logit"].shape == (4, 1)
  assert aux["stair_shape"].shape == (4, 2)
  assert aux["safe_stride"].shape == (4, 1)
  assert "safe_stride_interval" not in aux
  assert torch.all(
    (0.23 <= aux["stair_shape"][..., 0]) & (aux["stair_shape"][..., 0] <= 0.37)
  )
  assert torch.all(
    (0.088 <= aux["stair_shape"][..., 1]) & (aux["stair_shape"][..., 1] <= 0.25)
  )
  assert torch.all((0.08 <= aux["safe_stride"]) & (aux["safe_stride"] <= 0.45))
  assert model.mlp[0].in_features == model.obs_dim + model.z_dim
  diagnostics = model.get_slow_latent_diagnostics()
  assert diagnostics["event_prob"].shape == (4, 1)
  assert diagnostics["stair_prob"].shape == (4, 1)
  assert diagnostics["future_risk"].shape == (4, 1)
  assert diagnostics["future_quality"].shape == (4, 1)
  assert diagnostics["stair_shape"].shape == (4, 2)
  assert diagnostics["safe_stride"].shape == (4, 1)
  assert "safe_stride_interval" not in diagnostics
  assert diagnostics["z_norm"].shape == (4, 1)
  assert diagnostics["gate_mode"].shape == (4, 1)
  assert diagnostics["gate_memory_age"].shape == (4, 1)
  assert diagnostics["episode_write_ever"].shape == (4, 1)
  assert diagnostics["episode_memory_ever"].shape == (4, 1)
  assert diagnostics["gate_event_trigger"].shape == (4, 1)
  assert diagnostics["gate_write_confirm"].shape == (4, 1)
  assert diagnostics["gate_write_abort"].shape == (4, 1)
  assert diagnostics["gate_memory_exit"].shape == (4, 1)
  assert diagnostics["gate_memory_event_shape_boost"].shape == (4, 1)
  assert diagnostics["gate_release"].shape == (4, 1)
  assert diagnostics["alpha"].shape == (4, 5)
  assert diagnostics["alpha_state"].shape == (4, 1)
  assert diagnostics["alpha_shape"].shape == (4, 1)
  torch.testing.assert_close(
    torch.sigmoid(aux["stair_logit"]), diagnostics["stair_prob"]
  )


def test_shadow_semantic_uses_fixed_state_positions_and_physical_derivations() -> None:
  model = _make_model(
    structured_safe_stride_enabled=True,
    shadow_semantic_enabled=True,
  )
  gate = torch.tensor(
    [
      [0.0, 0.0, 0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0, 3.0, 0.0],
      [2.0, 15.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, -1.0, -1.0, 7.5],
    ]
  )
  event_prob = torch.tensor([[0.61], [0.59], [0.90], [0.10]])
  stair_prob = torch.tensor([[0.40], [0.20], [0.50], [0.10]])
  stair_shape = torch.tensor(
    [
      [model.tread_depth_min, model.riser_height_min],
      [model.tread_depth_max, model.riser_height_max],
      [
        0.5 * (model.tread_depth_min + model.tread_depth_max),
        0.5 * (model.riser_height_min + model.riser_height_max),
      ],
      [model.tread_depth_min, model.riser_height_max],
    ]
  )
  safe_stride_interval = torch.tensor(
    [
      [model.safe_stride_min, model.safe_stride_min],
      [model.safe_stride_min, model.safe_stride_max],
      [
        0.5 * (model.safe_stride_min + model.safe_stride_max),
        model.safe_stride_max,
      ],
      [
        model.safe_stride_min,
        0.5 * (model.safe_stride_min + model.safe_stride_max),
      ],
    ]
  )

  semantic = model._build_shadow_semantic(
    event_prob,
    stair_prob,
    gate,
    stair_shape,
    safe_stride_interval,
  )

  assert semantic.shape == (4, 16)
  expected_state = torch.tensor(
    [
      [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0],
      [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0],
      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5],
    ]
  )
  torch.testing.assert_close(semantic[:, :8], expected_state)
  torch.testing.assert_close(
    semantic[:, 8],
    torch.tensor([0.0, 1.0, 0.5, 0.0]),
  )
  torch.testing.assert_close(
    semantic[:, 9],
    torch.tensor([0.0, 1.0, 0.5, 1.0]),
  )
  torch.testing.assert_close(
    semantic[:, 10],
    torch.tensor([0.0, 0.0, 0.5, 0.0]),
  )
  torch.testing.assert_close(
    semantic[:, 11],
    torch.tensor([0.0, 1.0, 1.0, 0.5]),
  )
  torch.testing.assert_close(
    semantic[:, 12],
    torch.tensor([0.0, 0.5, 0.75, 0.25]),
  )
  torch.testing.assert_close(
    semantic[:, 13],
    torch.tensor([0.0, 1.0, 0.5, 0.5]),
  )
  torch.testing.assert_close(semantic[:, 14], semantic[:, 9])
  torch.testing.assert_close(semantic[:, 15], torch.zeros(4))


def test_shadow_semantic_diagnostics_do_not_change_actor_output() -> None:
  baseline = _make_model()
  shadow = _make_model(
    structured_safe_stride_enabled=True,
    shadow_semantic_enabled=True,
  )
  shadow.load_state_dict(baseline.state_dict(), strict=True)
  obs = _make_obs()

  baseline_actions = baseline(obs, stochastic_output=False)
  shadow_actions = shadow(obs, stochastic_output=False)

  torch.testing.assert_close(shadow_actions, baseline_actions)
  assert "shadow_semantic" not in baseline.get_slow_latent_diagnostics()
  interval = shadow.get_aux_outputs()["safe_stride_interval"]
  assert interval.shape == (4, 2)
  assert torch.all(interval[..., 1] >= interval[..., 0])
  semantic = shadow.get_slow_latent_diagnostics()["shadow_semantic"]
  assert semantic.shape == (4, 16)
  assert not semantic.requires_grad


def test_safe_stride_head_strictly_loads_legacy_one_output_weights() -> None:
  source = _make_model()
  legacy_state = source.state_dict()
  weight_key = "safe_stride_head.2.weight"
  bias_key = "safe_stride_head.2.bias"
  legacy_weight = legacy_state[weight_key][0:1].clone()
  legacy_bias = legacy_state[bias_key][0:1].clone()
  legacy_state[weight_key] = legacy_weight
  legacy_state[bias_key] = legacy_bias
  restored = _make_model(structured_safe_stride_enabled=True)

  restored.load_state_dict(legacy_state, strict=True)

  torch.testing.assert_close(restored.safe_stride_head[2].weight[0], legacy_weight[0])
  torch.testing.assert_close(restored.safe_stride_head[2].bias[0], legacy_bias[0])
  torch.testing.assert_close(
    restored.safe_stride_head[2].weight[1],
    torch.zeros_like(restored.safe_stride_head[2].weight[1]),
  )
  torch.testing.assert_close(
    restored.safe_stride_head[2].bias[1],
    restored.safe_stride_head[2].bias.new_tensor(-6.0),
  )


def test_dynamic_safe_stride_head_strictly_loads_legacy_input_weights() -> None:
  source = _make_model()
  legacy_state = source.state_dict()
  legacy_input_weight = legacy_state["safe_stride_head.0.weight"].clone()
  restored = _make_model(
    structured_safe_stride_enabled=True,
    dynamic_safe_stride_enabled=True,
  )

  restored.load_state_dict(legacy_state, strict=True)

  restored_weight = restored.safe_stride_head[0].weight
  torch.testing.assert_close(
    restored_weight[:, : source.shape_latent_dim],
    legacy_input_weight,
  )
  torch.testing.assert_close(
    restored_weight[:, source.shape_latent_dim :],
    torch.zeros_like(restored_weight[:, source.shape_latent_dim :]),
  )
  assert restored.safe_stride_width_head is not None
  torch.testing.assert_close(
    restored.safe_stride_width_head[2].bias,
    restored.safe_stride_width_head[2].bias.new_full((1,), -1.4),
  )
  actions = restored(_make_obs(), stochastic_output=False)
  assert actions.shape == (4, 3)


def test_dynamic_safe_stride_width_is_independent_of_predicted_lower() -> None:
  model = _make_model(
    structured_safe_stride_enabled=True,
    dynamic_safe_stride_enabled=True,
  )
  model.safe_stride_head = torch.nn.Identity()
  width_head = torch.nn.Linear(model.shape_latent_dim, 1)
  torch.nn.init.zeros_(width_head.weight)
  torch.nn.init.zeros_(width_head.bias)
  model.safe_stride_width_head = width_head
  lower_logits = torch.tensor([[-2.0], [2.0]])
  shape_memory = torch.zeros(2, model.shape_latent_dim)

  interval = model._decode_safe_stride_interval(lower_logits, shape_memory)
  widths = interval[:, 1] - interval[:, 0]

  torch.testing.assert_close(widths[0], widths[1])
  torch.testing.assert_close(
    widths,
    torch.full_like(widths, 0.5 * (model.safe_stride_max - model.safe_stride_min)),
  )


def test_safe_stride_probe_only_freezes_policy_and_action_distribution() -> None:
  actor = _make_model(
    structured_safe_stride_enabled=True,
    dynamic_safe_stride_enabled=True,
  )
  obs = _make_obs()
  critic = MLPModel(
    obs,
    {"critic": ["actor"]},
    "critic",
    1,
    hidden_dims=[8],
  )
  storage = RolloutStorage("rl", 4, 2, obs, [3])
  algorithm = PPOTeacherKL(
    actor,
    critic,
    storage,
    teacher_kl_cfg={"enabled": False},
    safe_stride_probe_only=True,
  )
  frozen = algorithm._capture_safe_stride_probe_frozen_state()
  actor.reset(torch.ones(4))
  actions_before = actor(obs, stochastic_output=False).detach().clone()

  loss = sum(
    parameter.square().sum() for parameter in actor.safe_stride_head.parameters()
  )
  algorithm.optimizer.zero_grad()
  loss.backward()
  algorithm.optimizer.step()

  actor.reset(torch.ones(4))
  actions_after = actor(obs, stochastic_output=False).detach()
  assert torch.equal(actions_after, actions_before)
  assert algorithm.freeze_normalization_updates is True
  assert all(
    not parameter.requires_grad
    for name, parameter in actor.named_parameters()
    if not name.startswith(("safe_stride_head.", "safe_stride_width_head."))
  )
  assert all(
    parameter.requires_grad for parameter in actor.safe_stride_head.parameters()
  )
  assert actor.safe_stride_width_head is not None
  assert all(
    parameter.requires_grad for parameter in actor.safe_stride_width_head.parameters()
  )
  algorithm._safe_stride_probe_frozen_state = frozen
  algorithm._verify_safe_stride_probe_frozen_state()


def test_stair_memory_uses_distinct_state_and_shape_hold_rates() -> None:
  model = _make_model()
  gate = torch.zeros(2, 5)
  gate[:, 0] = 2.0

  _next_gate, alpha = model._advance_gate_state(
    event_prob=torch.zeros(2, 1),
    stair_prob=torch.ones(2, 1),
    gate_state=gate,
  )

  torch.testing.assert_close(alpha[:, :2], torch.zeros(2, 2))
  torch.testing.assert_close(alpha[:, 2:], torch.full((2, 3), 0.05))


def test_shape_latent_play_ablation_only_zeros_actor_copy() -> None:
  model = _make_model()
  memory = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

  assert model._actor_memory(memory) is memory

  model.zero_shape_latent_for_actor = True
  actor_memory = model._actor_memory(memory)

  torch.testing.assert_close(actor_memory[:, :2], memory[:, :2])
  torch.testing.assert_close(actor_memory[:, 2:], torch.zeros(1, 3))
  torch.testing.assert_close(memory, torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]))


def test_shape_latent_stair_ablation_freezes_pre_write_actor_snapshot() -> None:
  model = _make_model()
  model.freeze_shape_latent_at_stair_entry = True
  model._gate_state = torch.zeros(1, 5)
  normal_memory = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

  torch.testing.assert_close(model._actor_memory(normal_memory), normal_memory)

  model._gate_state[:, 0] = 1.0
  write_memory = torch.tensor([[6.0, 7.0, 8.0, 9.0, 10.0]])
  write_actor_memory = model._actor_memory(write_memory)
  torch.testing.assert_close(write_actor_memory[:, :2], write_memory[:, :2])
  torch.testing.assert_close(write_actor_memory[:, 2:], normal_memory[:, 2:])

  model._gate_state[:, 0] = 2.0
  stair_memory = torch.tensor([[11.0, 12.0, 13.0, 14.0, 15.0]])
  stair_actor_memory = model._actor_memory(stair_memory)
  torch.testing.assert_close(stair_actor_memory[:, :2], stair_memory[:, :2])
  torch.testing.assert_close(stair_actor_memory[:, 2:], normal_memory[:, 2:])

  model._gate_state[:, 0] = 0.0
  torch.testing.assert_close(model._actor_memory(stair_memory), stair_memory)
  torch.testing.assert_close(
    stair_memory,
    torch.tensor([[11.0, 12.0, 13.0, 14.0, 15.0]]),
  )


def test_stair_memory_event_temporarily_boosts_only_shape_channels() -> None:
  model = _make_model()
  model.memory_event_shape_boost_steps = 3.0
  onnx_model = model.as_onnx()
  onnx_model.memory_event_shape_boost_steps = 3.0
  gate = torch.zeros(1, 5)
  gate[:, 0] = 2.0
  onnx_gate = gate.clone()

  gate, event_alpha = model._advance_gate_state(
    event_prob=torch.ones(1, 1),
    stair_prob=torch.ones(1, 1),
    gate_state=gate,
  )
  onnx_gate, onnx_event_alpha = onnx_model._advance_gate_state(
    event_prob=torch.ones(1, 1),
    stair_prob=torch.ones(1, 1),
    gate_state=onnx_gate,
  )

  assert gate[0, 0].item() == 2.0
  assert gate[0, 3].item() == 3.0
  torch.testing.assert_close(event_alpha[:, :2], torch.zeros(1, 2))
  torch.testing.assert_close(event_alpha[:, 2:], torch.full((1, 3), 0.3))
  torch.testing.assert_close(onnx_gate, gate)
  torch.testing.assert_close(onnx_event_alpha, event_alpha)

  for expected_timer in (2.0, 1.0):
    gate, held_boost_alpha = model._advance_gate_state(
      event_prob=torch.zeros(1, 1),
      stair_prob=torch.ones(1, 1),
      gate_state=gate,
    )
    assert gate[0, 3].item() == expected_timer
    torch.testing.assert_close(held_boost_alpha[:, :2], torch.zeros(1, 2))
    torch.testing.assert_close(held_boost_alpha[:, 2:], torch.full((1, 3), 0.3))

  gate, held_alpha = model._advance_gate_state(
    event_prob=torch.zeros(1, 1),
    stair_prob=torch.ones(1, 1),
    gate_state=gate,
  )
  assert gate[0, 3].item() == 0.0
  torch.testing.assert_close(held_alpha[:, :2], torch.zeros(1, 2))
  torch.testing.assert_close(held_alpha[:, 2:], torch.full((1, 3), 0.05))


def test_stair_write_requires_stair_probability_to_enter_memory() -> None:
  model = _make_model()
  model.write_steps = 2.0
  model.stair_confirm_steps = 2.0
  gate = torch.zeros(2, 5)

  gate, _alpha = model._advance_gate_state(
    event_prob=torch.ones(2, 1),
    stair_prob=torch.tensor([[0.0], [0.5]]),
    gate_state=gate,
  )
  torch.testing.assert_close(gate[:, 0], torch.full((2,), 1.0))

  gate, _alpha = model._advance_gate_state(
    event_prob=torch.zeros(2, 1),
    stair_prob=torch.tensor([[0.05], [0.5]]),
    gate_state=gate,
  )

  assert gate[0, 0].item() == 0.0
  assert gate[0, 4].item() == pytest.approx(model.cooldown_steps)
  assert gate[1, 0].item() == 2.0


def test_stair_write_requires_consecutive_evidence() -> None:
  model = _make_model()
  model.write_steps = 3.0
  model.stair_confirm_steps = 2.0
  gate = torch.zeros(1, 5)

  for event, stair in [(1.0, 0.5), (0.0, 0.0), (0.0, 0.5)]:
    gate, _alpha = model._advance_gate_state(
      event_prob=torch.tensor([[event]]),
      stair_prob=torch.tensor([[stair]]),
      gate_state=gate,
    )

  assert gate[0, 0].item() == 0.0
  assert gate[0, 4].item() == pytest.approx(model.cooldown_steps)


def test_final_write_frame_uses_write_alpha_before_memory_hold() -> None:
  model = _make_model()
  model.write_steps = 2.0
  model.stair_confirm_steps = 2.0
  gate = torch.zeros(1, 5)

  gate, first_alpha = model._advance_gate_state(
    event_prob=torch.ones(1, 1),
    stair_prob=torch.ones(1, 1),
    gate_state=gate,
  )
  gate, final_alpha = model._advance_gate_state(
    event_prob=torch.zeros(1, 1),
    stair_prob=torch.ones(1, 1),
    gate_state=gate,
  )

  assert gate[0, 0].item() == 2.0
  torch.testing.assert_close(first_alpha, torch.full((1, 5), 0.8))
  torch.testing.assert_close(final_alpha, torch.full((1, 5), 0.8))

  _gate, held_alpha = model._advance_gate_state(
    event_prob=torch.zeros(1, 1),
    stair_prob=torch.ones(1, 1),
    gate_state=gate,
  )
  torch.testing.assert_close(held_alpha[:, :2], torch.zeros(1, 2))
  torch.testing.assert_close(held_alpha[:, 2:], torch.full((1, 3), 0.05))


def test_stair_head_does_not_read_held_memory_channels() -> None:
  model = _make_model()
  h_t = torch.randn(3, model.latent_hidden_dim)

  expected = model._stair_logit(h_t)
  first_linear = cast(torch.nn.Linear, model.stair_state_head[0])
  memory_weights = first_linear.weight[:, model.latent_hidden_dim :]

  torch.testing.assert_close(
    expected,
    model.stair_state_head(
      torch.cat([h_t, torch.zeros(3, model.state_latent_dim)], dim=-1)
    ),
  )
  assert memory_weights.shape[-1] == model.state_latent_dim


def test_stair_head_alone_controls_memory_exit() -> None:
  model = _make_model()
  model.min_stair_steps = 0.0
  model.exit_steps = 2.0
  gate = torch.zeros(2, 5)
  gate[:, 0] = 2.0

  for _ in range(2):
    gate, _alpha = model._advance_gate_state(
      event_prob=torch.tensor([[1.0], [0.0]]),
      stair_prob=torch.tensor([[0.0], [1.0]]),
      gate_state=gate,
    )

  assert gate[0, 0].item() == 0.0
  assert gate[1, 0].item() == 2.0


def test_memory_exit_smoothly_releases_latent_update_rate() -> None:
  model = _make_model()
  model.min_stair_steps = 0.0
  model.exit_steps = 1.0
  gate = torch.zeros(1, 5)
  gate[:, 0] = 2.0

  gate, exit_alpha = model._advance_gate_state(
    event_prob=torch.zeros(1, 1),
    stair_prob=torch.zeros(1, 1),
    gate_state=gate,
  )

  assert gate[0, 0].item() == 0.0
  assert gate[0, 3].item() == -1.0
  assert gate[0, 4].item() == pytest.approx(model.cooldown_steps)
  torch.testing.assert_close(exit_alpha[:, :2], torch.zeros(1, 2))
  torch.testing.assert_close(exit_alpha[:, 2:], torch.full((1, 3), 0.05))

  gate, first_release_alpha = model._advance_gate_state(
    event_prob=torch.zeros(1, 1),
    stair_prob=torch.zeros(1, 1),
    gate_state=gate,
  )
  expected_state_alpha = model.alpha_fast / model.cooldown_steps
  expected_shape_alpha = (
    model.alpha_hold_shape
    + (model.alpha_fast - model.alpha_hold_shape) / model.cooldown_steps
  )
  torch.testing.assert_close(
    first_release_alpha[:, :2],
    torch.full((1, 2), expected_state_alpha),
  )
  torch.testing.assert_close(
    first_release_alpha[:, 2:],
    torch.full((1, 3), expected_shape_alpha),
  )

  for _ in range(int(model.cooldown_steps) - 1):
    gate, alpha = model._advance_gate_state(
      event_prob=torch.zeros(1, 1),
      stair_prob=torch.zeros(1, 1),
      gate_state=gate,
    )

  assert gate[0, 3].item() == 0.0
  torch.testing.assert_close(alpha, torch.full((1, 5), model.alpha_fast))


def test_write_abort_does_not_enter_memory_release() -> None:
  model = _make_model()
  model.write_steps = 1.0
  model.stair_confirm_steps = 1.0
  gate = torch.zeros(1, 5)

  gate, write_alpha = model._advance_gate_state(
    event_prob=torch.ones(1, 1),
    stair_prob=torch.zeros(1, 1),
    gate_state=gate,
  )
  assert gate[0, 0].item() == 0.0
  assert gate[0, 3].item() == 0.0
  torch.testing.assert_close(write_alpha, torch.full((1, 5), model.alpha_write))

  _gate, normal_alpha = model._advance_gate_state(
    event_prob=torch.zeros(1, 1),
    stair_prob=torch.zeros(1, 1),
    gate_state=gate,
  )
  torch.testing.assert_close(normal_alpha, torch.full((1, 5), model.alpha_fast))


def test_onnx_gate_matches_training_gate_transitions() -> None:
  model = _make_model()
  model.write_steps = 3.0
  model.stair_confirm_steps = 2.0
  model.min_stair_steps = 0.0
  model.exit_steps = 2.0
  onnx_model = model.as_onnx()
  training_gate = torch.zeros(1, 5)
  onnx_gate = torch.zeros(1, 5)

  for event, stair in [(0.8, 0.5), (0.0, 0.1), (0.0, 0.5), (0.0, 0.0)]:
    event_prob = torch.tensor([[event]])
    stair_prob = torch.tensor([[stair]])
    training_gate, training_alpha = model._advance_gate_state(
      event_prob, stair_prob, training_gate
    )
    onnx_gate, onnx_alpha = onnx_model._advance_gate_state(
      event_prob, stair_prob, onnx_gate
    )
    torch.testing.assert_close(onnx_gate, training_gate)
    torch.testing.assert_close(onnx_alpha, training_alpha)


def test_onnx_gate_matches_memory_release_transition() -> None:
  model = _make_model()
  model.min_stair_steps = 0.0
  model.exit_steps = 1.0
  onnx_model = model.as_onnx()
  training_gate = torch.zeros(1, 5)
  training_gate[:, 0] = 2.0
  onnx_gate = training_gate.clone()

  for _ in range(int(model.cooldown_steps) + 1):
    event_prob = torch.zeros(1, 1)
    stair_prob = torch.zeros(1, 1)
    training_gate, training_alpha = model._advance_gate_state(
      event_prob, stair_prob, training_gate
    )
    onnx_gate, onnx_alpha = onnx_model._advance_gate_state(
      event_prob, stair_prob, onnx_gate
    )
    torch.testing.assert_close(onnx_gate, training_gate)
    torch.testing.assert_close(onnx_alpha, training_alpha)


def test_safe_stride_output_bounds_must_be_ordered() -> None:
  obs = _make_obs()

  with pytest.raises(ValueError, match="must be greater"):
    LSTMSlowLatentMLPModel(
      obs=obs,
      obs_groups={"actor": ["actor"], "latent": ["latent"]},
      obs_set="actor",
      output_dim=3,
      obs_normalization=False,
      latent_dim=5,
      state_latent_dim=2,
      safe_stride_min=0.2,
      safe_stride_max=0.2,
    )


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


def test_stair_shape_huber_uses_independent_component_masks() -> None:
  predictions = torch.tensor([[10.0, 0.20]])
  labels = torch.tensor([[0.30, 0.10]])
  component_valid = torch.tensor([[0.0, 1.0]])

  loss = PPOTeacherKL._compute_stair_shape_loss(
    predictions,
    labels,
    component_valid,
    huber_delta=0.05,
  )
  mae, huber = PPOTeacherKL._compute_stair_shape_component_errors(
    predictions,
    labels,
    component_valid,
    huber_delta=0.05,
  )

  assert loss.item() == pytest.approx(0.075)
  assert mae.tolist() == pytest.approx([0.0, 0.1])
  assert huber.tolist() == pytest.approx([0.0, 0.075])


def test_stair_shape_normalized_huber_uses_each_physical_range() -> None:
  predictions = torch.tensor([[0.197, 0.1042], [10.0, 10.0]])
  labels = torch.tensor([[0.18, 0.088], [0.18, 0.088]])
  valid = torch.tensor([[1.0], [0.0]])

  loss = PPOTeacherKL._compute_normalized_stair_shape_loss(
    predictions,
    labels,
    valid,
    lower_bounds=torch.tensor([0.18, 0.088]),
    upper_bounds=torch.tensor([0.35, 0.25]),
    huber_delta=0.05,
  )

  assert loss.item() == pytest.approx(0.075)


def test_stair_shape_regression_statistics_detect_size_prediction() -> None:
  labels = torch.tensor(
    [
      [0.25, 0.10],
      [0.30, 0.15],
      [0.35, 0.20],
      [0.40, 0.25],
    ]
  )
  valid = torch.tensor([[1.0], [1.0], [1.0], [0.0]])

  label_std, prediction_std, correlation, r_squared = (
    PPOTeacherKL._compute_masked_regression_statistics(labels, labels, valid)
  )

  assert torch.all(label_std > 0.0)
  torch.testing.assert_close(prediction_std, label_std)
  torch.testing.assert_close(correlation, torch.ones(2))
  torch.testing.assert_close(r_squared, torch.ones(2))


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
  assert diagnostics["alpha"].shape == (3, 1, 5)
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
    "future_collision_risk",
    "future_safe_landing_quality",
    "stair_shape",
    "safe_stride",
  ]

  outputs = onnx_model(*onnx_model.get_dummy_inputs())
  assert outputs[0].shape == (1, 3)
  assert outputs[1].shape == (1, 1, 7)
  assert outputs[2].shape == (1, 1, 7)
  assert outputs[3].shape == (1, 5)
  assert outputs[4].shape == (1, 5)
  assert outputs[7].shape == (1, 1)
  assert outputs[8].shape == (1, 1)
  assert outputs[9].shape == (1, 2)
  assert outputs[10].shape == (1, 1)
  assert torch.all((0.23 <= outputs[9][..., 0]) & (outputs[9][..., 0] <= 0.37))
  assert torch.all((0.088 <= outputs[9][..., 1]) & (outputs[9][..., 1] <= 0.25))
  assert torch.all((0.08 <= outputs[10]) & (outputs[10] <= 0.45))


def test_slow_latent_export_metadata() -> None:
  model = _make_model()
  metadata = get_recurrent_policy_metadata(model)

  assert metadata["policy_has_slow_latent"] == "true"
  assert metadata["policy_slow_latent_dim"] == "5"
  assert metadata["policy_slow_latent_alpha"] == "0.0"
  assert metadata["policy_slow_latent_state_dim"] == "2"
  assert metadata["policy_slow_latent_alpha_hold_state"] == "0.0"
  assert metadata["policy_slow_latent_alpha_hold_shape"] == "0.05"
  assert metadata["policy_slow_latent_memory_event_shape_boost_steps"] == "15.0"
  assert metadata["policy_latent_obs_dim"] == "11"
  assert metadata["policy_stair_tread_depth_min"] == "0.23"
  assert metadata["policy_stair_tread_depth_max"] == "0.37"
  assert metadata["policy_stair_riser_height_min"] == "0.088"
  assert metadata["policy_stair_riser_height_max"] == "0.25"
  assert metadata["policy_stair_safe_stride_min"] == "0.1"
  assert metadata["policy_stair_safe_stride_max"] == "0.55"
  assert metadata["policy_onnx_input_names"] == [
    "actor_obs",
    "latent_obs",
    "h_in",
    "c_in",
    "z_in",
    "gate_state_in",
  ]
  output_names = metadata["policy_onnx_output_names"]
  assert isinstance(output_names, list)
  assert output_names[-2:] == [
    "stair_shape",
    "safe_stride",
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
