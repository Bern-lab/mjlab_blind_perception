"""Tests for PPO teacher-guidance losses."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import Any, cast

import pytest

from rsl_rl.algorithms.ppo_teacher_kl import PPOTeacherKL
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

NUM_ENVS = 4
NUM_STEPS = 2
OBS_DIM = 8
NUM_ACTIONS = 3


class _DummyEnv:
    num_actions = NUM_ACTIONS
    num_envs = NUM_ENVS


def _build_teacher_kl(loss_cfg: dict) -> PPOTeacherKL:
    obs = TensorDict(
        {
            "actor": torch.zeros(NUM_ENVS, OBS_DIM),
            "critic": torch.zeros(NUM_ENVS, OBS_DIM),
            "teacher": torch.zeros(NUM_ENVS, OBS_DIM),
        },
        batch_size=[NUM_ENVS],
    )
    obs_groups = {
        "actor": ["actor"],
        "critic": ["critic"],
        "teacher": ["teacher"],
    }
    actor = MLPModel(
        obs,
        obs_groups,
        "actor",
        NUM_ACTIONS,
        hidden_dims=[16],
        distribution_cfg={"class_name": "GaussianDistribution"},
    )
    critic = MLPModel(obs, obs_groups, "critic", 1, hidden_dims=[16])
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    return PPOTeacherKL(actor, critic, storage, teacher_kl_cfg=loss_cfg)


def _build_obs() -> TensorDict:
    return TensorDict(
        {
            "actor": torch.randn(NUM_ENVS, OBS_DIM),
            "critic": torch.randn(NUM_ENVS, OBS_DIM),
            "teacher": torch.randn(NUM_ENVS, OBS_DIM),
        },
        batch_size=[NUM_ENVS],
    )


def test_mean_huber_guidance_ignores_std_mismatch() -> None:
    """Mean-only guidance should not penalize different teacher/student std."""
    alg = _build_teacher_kl({"loss_type": "mean_huber", "huber_delta": 0.5})
    mean = torch.zeros(2, NUM_ACTIONS, requires_grad=True)
    teacher_params = (torch.zeros(2, NUM_ACTIONS), torch.full((2, NUM_ACTIONS), 2.0))
    student_params = (mean, torch.full((2, NUM_ACTIONS), 0.25))

    loss, logs = alg._compute_mean_teacher_loss(teacher_params, student_params)

    assert loss.item() == 0.0
    assert logs["teacher_mean_huber"].item() == 0.0
    assert logs["teacher_kl"].item() > 0.0


def test_slow_latent_diagnostics_are_logged() -> None:
    """Slow-latent diagnostics should surface in the PPO loss dictionary."""
    alg = _build_teacher_kl({"enabled": False})

    def _diagnostics() -> dict[str, torch.Tensor]:
        return {
            "event_prob": torch.tensor([[0.2], [0.8]]),
            "stair_prob": torch.tensor([[0.3], [0.7]]),
            "future_prob": torch.tensor([[0.4], [0.6]]),
            "z_norm": torch.tensor([[1.0], [3.0]]),
            "gate_mode": torch.tensor([[0.0], [1.0], [2.0], [2.0]]),
            "alpha": torch.tensor([[0.3], [0.8], [0.02]]),
        }

    cast(Any, alg.actor).get_slow_latent_diagnostics = _diagnostics

    logs = alg._compute_slow_latent_diagnostic_logs()

    assert logs["slow_latent_event_prob_mean"] == pytest.approx(0.5)
    assert logs["slow_latent_stair_prob_mean"] == pytest.approx(0.5)
    assert logs["slow_latent_future_prob_mean"] == pytest.approx(0.5)
    assert logs["slow_latent_z_norm_mean"] == pytest.approx(2.0)
    assert logs["slow_latent_z_norm_max"] == pytest.approx(3.0)
    assert logs["slow_latent_mode_normal_count"] == pytest.approx(1.0)
    assert logs["slow_latent_mode_write_count"] == pytest.approx(1.0)
    assert logs["slow_latent_mode_memory_count"] == pytest.approx(2.0)
    assert logs["slow_latent_alpha_min"] == pytest.approx(0.02)
    assert logs["slow_latent_alpha_max"] == pytest.approx(0.8)


def test_mean_huber_guidance_applies_loss_cap() -> None:
    """The update loss should respect max_teacher_loss for mean guidance."""
    alg = _build_teacher_kl({
        "loss_type": "mean_huber",
        "huber_delta": 0.5,
        "max_teacher_loss": 0.25,
    })
    teacher_params = (torch.zeros(1, NUM_ACTIONS), torch.ones(1, NUM_ACTIONS))
    student_params = (
        torch.full((1, NUM_ACTIONS), 10.0, requires_grad=True),
        torch.ones(1, NUM_ACTIONS),
    )

    loss, logs = alg._compute_mean_teacher_loss(teacher_params, student_params)

    assert loss.item() == 0.25
    assert logs["teacher_loss_for_update"].item() == 0.25
    assert logs["teacher_mean_huber"].item() > loss.item()


def test_disabled_guidance_runs_without_teacher() -> None:
    """Disabled teacher guidance should reduce the additional loss to zero."""
    alg = _build_teacher_kl({"enabled": False})

    loss, logs = alg._compute_additional_loss(
        batch=cast(Any, None),
        original_batch_size=0,
        distribution_params=(),
    )

    assert loss.item() == 0.0
    assert logs["teacher_loss"] == 0.0
    assert logs["teacher_loss_for_update"] == 0.0
    assert logs["teacher_kl_lambda"] == 0.0
    assert logs["teacher_guidance_enabled"] == 0.0


def test_mlp_actor_skips_slow_latent_aux_path() -> None:
    """Feedforward actors should not enter the slow-latent auxiliary path."""
    alg = _build_teacher_kl({"enabled": False})
    called = False

    def _raise_if_called(batch: RolloutStorage.Batch) -> tuple[torch.Tensor, dict]:
        nonlocal called
        called = True
        raise AssertionError("MLP actor should not compute slow-latent aux loss")

    cast(Any, alg)._compute_slow_latent_aux_loss = _raise_if_called

    loss, logs = alg._compute_additional_loss(
        batch=RolloutStorage.Batch(hidden_states=(None, None)),
        original_batch_size=0,
        distribution_params=(),
    )

    assert called is False
    assert loss.item() == 0.0
    assert logs["teacher_guidance_enabled"] == 0.0


def test_slow_latent_aux_debug_handles_missing_actor_hidden_state() -> None:
    """Slow-latent debug logging should tolerate feedforward-style batches."""
    alg = _build_teacher_kl({"enabled": False})
    cast(Any, alg.actor).get_aux_outputs = lambda: {}

    loss, logs = alg._compute_slow_latent_aux_loss(
        RolloutStorage.Batch(observations=_build_obs(), hidden_states=(None, None))
    )

    assert loss.item() == 0.0
    assert logs["slow_latent_debug_empty_aux_outputs"] == 1.0


def test_teacher_forward_chunking_matches_full_batch() -> None:
    """Chunked teacher inference should preserve distribution parameters."""
    alg = _build_teacher_kl({"teacher_forward_chunk_size": 3})
    obs = TensorDict(
        {"teacher": torch.randn(NUM_STEPS, NUM_ENVS, OBS_DIM)},
        batch_size=[NUM_STEPS, NUM_ENVS],
    )
    init_obs = TensorDict(
        {"teacher": torch.zeros(NUM_ENVS, OBS_DIM)},
        batch_size=[NUM_ENVS],
    )
    masks = torch.ones(NUM_STEPS, NUM_ENVS, dtype=torch.bool)
    obs_groups = {"teacher": ["teacher"]}
    alg.teacher = MLPModel(
        init_obs,
        obs_groups,
        "teacher",
        NUM_ACTIONS,
        hidden_dims=[16],
        distribution_cfg={"class_name": "GaussianDistribution"},
    )
    alg.teacher_loaded = True
    alg._freeze_teacher()

    with torch.no_grad():
        alg.teacher(obs, masks=masks, stochastic_output=True)
        full_params = tuple(param.detach().clone() for param in alg.teacher.output_distribution_params)
        chunked_params = alg._compute_teacher_distribution_params(obs, masks)

    for full, chunked in zip(full_params, chunked_params):
        torch.testing.assert_close(chunked, full)


def test_construct_disabled_guidance_skips_teacher_loading() -> None:
    """Disabled teacher guidance should not construct or load the teacher."""
    obs = TensorDict(
        {
            "actor": torch.zeros(NUM_ENVS, OBS_DIM),
            "critic": torch.zeros(NUM_ENVS, OBS_DIM),
            "teacher": torch.zeros(NUM_ENVS, OBS_DIM),
            "camera": torch.zeros(NUM_ENVS, 1, 8, 8),
        },
        batch_size=[NUM_ENVS],
    )
    cfg = {
        "algorithm": {
            "class_name": "PPOTeacherKL",
            "teacher_kl_cfg": {"enabled": False},
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [16],
            "distribution_cfg": {"class_name": "GaussianDistribution"},
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [16]},
        "teacher": {
            "class_name": "MLPModel",
            "hidden_dims": [16],
            "distribution_cfg": {"class_name": "GaussianDistribution"},
        },
        "obs_groups": {
            "actor": ["actor"],
            "critic": ["critic"],
            "teacher": ["teacher", "camera"],
        },
        "num_steps_per_env": NUM_STEPS,
        "multi_gpu": None,
        "torch_compile_mode": None,
    }

    alg = PPOTeacherKL.construct_algorithm(obs, cast(Any, _DummyEnv()), cfg, "cpu")

    assert alg.teacher_guidance_enabled is False
    assert alg.teacher is None
    assert alg.teacher_loaded is False


def test_imitation_only_update_skips_ppo_losses() -> None:
    """Teacher imitation-only update should optimize only the teacher guidance loss."""
    alg = _build_teacher_kl({
        "loss_type": "mean_huber",
        "huber_delta": 0.5,
        "imitation_only": True,
        "imitation_loss_coef": 1.0,
    })
    obs = _build_obs()
    obs_groups = {
        "actor": ["actor"],
        "critic": ["critic"],
        "teacher": ["teacher"],
    }
    alg.teacher = MLPModel(
        obs,
        obs_groups,
        "teacher",
        NUM_ACTIONS,
        hidden_dims=[16],
        distribution_cfg={"class_name": "GaussianDistribution"},
    )
    alg.teacher_loaded = True
    alg._freeze_teacher()
    critic_before = [param.detach().clone() for param in alg.critic.parameters()]

    for _ in range(NUM_STEPS):
        alg.act(obs)
        alg.process_env_step(
            obs,
            rewards=torch.zeros(NUM_ENVS),
            dones=torch.zeros(NUM_ENVS, dtype=torch.bool),
            extras={},
        )

    losses = alg.update()

    assert losses["teacher_imitation_only"] == 1.0
    assert losses["value"] == 0.0
    assert losses["surrogate"] == 0.0
    assert losses["teacher_loss"] > 0.0
    for before, after in zip(critic_before, alg.critic.parameters()):
        torch.testing.assert_close(before, after.detach())
