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
            "future_risk": torch.tensor([[0.4], [0.6]]),
            "future_quality": torch.tensor([[0.7], [0.3]]),
            "z_norm": torch.tensor([[1.0], [3.0]]),
            "gate_mode": torch.tensor([[0.0], [1.0], [2.0], [2.0]]),
            "gate_memory_age": torch.tensor([[0.0], [0.0], [4.0], [10.0]]),
            "episode_write_ever": torch.tensor([[1.0], [0.0]]),
            "episode_memory_ever": torch.tensor([[1.0], [1.0]]),
            "gate_event_trigger": torch.tensor([[1.0], [0.0], [0.0], [0.0]]),
            "gate_write_confirm": torch.tensor([[0.0], [1.0], [0.0], [0.0]]),
            "gate_write_abort": torch.tensor([[0.0], [0.0], [0.0], [0.0]]),
            "gate_memory_exit": torch.tensor([[0.0], [0.0], [0.0], [1.0]]),
            "gate_release": torch.tensor([[0.0], [0.0], [0.0], [1.0]]),
            "alpha": torch.tensor([[0.3, 0.3], [0.8, 0.8], [0.01, 0.05]]),
            "alpha_state": torch.tensor([[0.3], [0.8], [0.01]]),
            "alpha_shape": torch.tensor([[0.3], [0.8], [0.05]]),
        }

    cast(Any, alg.actor).get_slow_latent_diagnostics = _diagnostics

    logs = alg._compute_slow_latent_diagnostic_logs()

    assert logs["slow_latent_event_prob_mean"] == pytest.approx(0.5)
    assert logs["slow_latent_stair_prob_mean"] == pytest.approx(0.5)
    assert logs["slow_latent_future_collision_risk_mean"] == pytest.approx(0.5)
    assert logs["slow_latent_future_safe_landing_quality_mean"] == pytest.approx(0.5)
    assert logs["slow_latent_future_risk_quality_overlap_mean"] == pytest.approx(0.23)
    assert logs["slow_latent_z_norm_mean"] == pytest.approx(2.0)
    assert logs["slow_latent_z_norm_max"] == pytest.approx(3.0)
    assert logs["slow_latent_mode_normal_count"] == pytest.approx(1.0)
    assert logs["slow_latent_mode_write_count"] == pytest.approx(1.0)
    assert logs["slow_latent_mode_memory_count"] == pytest.approx(2.0)
    assert logs["slow_latent_episode_write_ever_ratio"] == pytest.approx(0.5)
    assert logs["slow_latent_episode_memory_ever_ratio"] == pytest.approx(1.0)
    assert logs["slow_latent_gate_event_trigger_ratio"] == pytest.approx(0.25)
    assert logs["slow_latent_gate_write_confirm_ratio"] == pytest.approx(0.25)
    assert logs["slow_latent_gate_write_abort_ratio"] == pytest.approx(0.0)
    assert logs["slow_latent_gate_write_confirm_rate"] == pytest.approx(1.0)
    assert logs["slow_latent_gate_memory_exit_ratio"] == pytest.approx(0.25)
    assert logs["slow_latent_gate_release_ratio"] == pytest.approx(0.25)
    assert logs["slow_latent_memory_age_mean"] == pytest.approx(7.0)
    assert logs["slow_latent_memory_age_p90"] == pytest.approx(9.4)
    assert logs["slow_latent_alpha_min"] == pytest.approx(0.01)
    assert logs["slow_latent_alpha_max"] == pytest.approx(0.8)
    assert logs["slow_latent_alpha_state_mean"] == pytest.approx(0.37)
    assert logs["slow_latent_alpha_shape_mean"] == pytest.approx(0.3833333)


def test_future_labels_use_max_risk_and_first_touchdown_quality() -> None:
    """Risk uses the worst frame while quality uses the first future touchdown."""
    alg = _build_teacher_kl({"enabled": False})
    risk_now = torch.tensor([0.0, 0.2, 0.8, 0.1, 0.0]).view(5, 1, 1)
    touchdown_now = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0]).view(5, 1, 1)
    quality_now = torch.tensor([0.0, 0.0, 0.4, 0.9, 0.0]).view(5, 1, 1)

    future_risk = alg._compute_future_max_labels(risk_now, None, horizon=3)
    future_quality, found = alg._compute_future_first_touchdown_quality(
        touchdown_now,
        quality_now,
        None,
        horizon=3,
    )

    assert future_risk[:, 0, 0].tolist() == pytest.approx([0.8, 0.8, 0.1, 0.0, 0.0])
    assert future_quality[:, 0, 0].tolist() == pytest.approx([0.4, 0.4, 0.9, 0.0, 0.0])
    assert found[:, 0, 0].tolist() == pytest.approx([1.0, 1.0, 1.0, 0.0, 0.0])


def test_continuous_future_aux_losses_use_future_labels() -> None:
    """Future heads should regress worst risk and first-touchdown quality."""
    alg = _build_teacher_kl({"enabled": False})
    actor = cast(Any, alg.actor)
    actor.aux_event_coef = 0.0
    actor.aux_stair_coef = 0.0
    actor.aux_future_collision_risk_coef = 0.1
    actor.aux_future_safe_landing_quality_coef = 0.1
    actor.aux_stair_shape_coef = 0.0
    actor.aux_safe_stride_coef = 0.0
    actor.future_horizon = 2
    actor.future_risk_weight_scale = 2.0
    actor.future_quality_weight_scale = 2.0
    actor.future_risk_huber_delta = 0.1
    actor.future_quality_huber_delta = 0.1
    actor.get_aux_outputs = lambda: {
        "future_collision_risk_logit": torch.zeros(4, 1, 1),
        "future_safe_landing_quality_logit": torch.zeros(4, 1, 1),
    }
    actor.get_slow_latent_diagnostics = lambda: {}

    labels = torch.zeros(4, 1, 10)
    labels[:, 0, 7] = torch.tensor([0.0, 0.2, 0.8, 0.0])
    labels[:, 0, 8] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    labels[:, 0, 9] = torch.tensor([0.0, 0.0, 0.6, 0.0])
    observations = TensorDict({"latent_labels": labels}, batch_size=[4, 1])

    loss, logs = alg._compute_slow_latent_aux_loss(
        RolloutStorage.Batch(observations=observations, hidden_states=(None, None))
    )

    assert loss.item() > 0.0
    assert logs["slow_latent_future_collision_risk_label_mean"] == pytest.approx(0.4)
    assert logs["slow_latent_future_safe_landing_quality_label_mean"] == pytest.approx(0.3)
    assert logs["slow_latent_future_touchdown_found_ratio"] == pytest.approx(0.5)
    assert logs["slow_latent_first_touchdown_quality_mean"] == pytest.approx(0.6)


def test_event_aux_logs_threshold_crossing_and_recall() -> None:
    """Event diagnostics should expose the write-threshold crossing behavior."""
    alg = _build_teacher_kl({"enabled": False})
    actor = cast(Any, alg.actor)
    actor.aux_event_coef = 0.1
    actor.aux_stair_coef = 0.0
    actor.aux_future_collision_risk_coef = 0.0
    actor.aux_future_safe_landing_quality_coef = 0.0
    actor.aux_stair_shape_coef = 0.0
    actor.aux_safe_stride_coef = 0.0
    actor.event_on_threshold = 0.6
    event_prob = torch.tensor([[0.2], [0.7], [0.5], [0.8]])
    actor.get_aux_outputs = lambda: {"event_logit": torch.logit(event_prob)}
    actor.get_slow_latent_diagnostics = lambda: {}
    labels = torch.zeros(4, 10)
    labels[:, 0] = torch.tensor([0.0, 1.0, 1.0, 0.0])
    observations = TensorDict({"latent_labels": labels}, batch_size=[NUM_ENVS])

    loss, logs = alg._compute_slow_latent_aux_loss(
        RolloutStorage.Batch(observations=observations, hidden_states=(None, None))
    )

    assert loss.item() > 0.0
    assert logs["slow_latent_event_label_mean"] == pytest.approx(0.5)
    assert logs["slow_latent_event_label_positive_ratio"] == pytest.approx(0.5)
    assert logs["slow_latent_event_prob_max"] == pytest.approx(0.8)
    assert logs["slow_latent_event_prob_p99"] == pytest.approx(0.797)
    assert logs["slow_latent_event_prob_gt_on_threshold_ratio"] == pytest.approx(0.5)
    assert logs["slow_latent_event_recall_at_on_threshold"] == pytest.approx(0.5)
    assert logs["slow_latent_event_raw_recall_at_on_threshold"] == pytest.approx(0.5)
    assert logs["slow_latent_event_precision_at_on_threshold"] == pytest.approx(0.5)
    assert logs["slow_latent_event_prob_gt_0p6_ratio"] == pytest.approx(0.5)
    assert logs["slow_latent_event_prob_gt_0p4_ratio"] == pytest.approx(0.75)
    assert logs["slow_latent_event_prob_pos_mean"] == pytest.approx(0.6)
    assert logs["slow_latent_event_prob_neg_mean"] == pytest.approx(0.5)
    assert logs["slow_latent_event_recall_at_0p6"] == pytest.approx(0.5)
    assert logs["slow_latent_event_precision_at_0p6"] == pytest.approx(0.5)


def test_event_aux_expands_sparse_labels_and_uses_positive_weight() -> None:
    """Sparse one-frame events should train over a short positive window."""
    alg = _build_teacher_kl({"enabled": False})
    actor = cast(Any, alg.actor)
    actor.aux_event_coef = 0.1
    actor.aux_event_pos_weight = 10.0
    actor.event_label_window_steps = 3
    actor.aux_stair_coef = 0.0
    actor.aux_future_collision_risk_coef = 0.0
    actor.aux_future_safe_landing_quality_coef = 0.0
    actor.aux_stair_shape_coef = 0.0
    actor.aux_safe_stride_coef = 0.0
    actor.event_on_threshold = 0.6
    event_prob = torch.tensor([0.1, 0.7, 0.8, 0.2, 0.1]).view(5, 1, 1)
    actor.get_aux_outputs = lambda: {"event_logit": torch.logit(event_prob)}
    actor.get_slow_latent_diagnostics = lambda: {}
    labels = torch.zeros(5, 1, 10)
    labels[1, 0, 0] = 1.0
    observations = TensorDict({"latent_labels": labels}, batch_size=[5, 1])

    loss, logs = alg._compute_slow_latent_aux_loss(
        RolloutStorage.Batch(observations=observations, hidden_states=(None, None))
    )

    assert loss.item() > 0.0
    assert logs["slow_latent_event_pos_weight"] == pytest.approx(10.0)
    assert logs["slow_latent_event_label_window_steps"] == pytest.approx(3.0)
    assert logs["slow_latent_event_raw_label_mean"] == pytest.approx(0.2)
    assert logs["slow_latent_event_label_mean"] == pytest.approx(0.6)
    assert logs["slow_latent_event_prob_gt_0p6_ratio"] == pytest.approx(0.4)
    assert logs["slow_latent_event_recall_at_on_threshold"] == pytest.approx(2 / 3)
    assert logs["slow_latent_event_raw_recall_at_on_threshold"] == pytest.approx(1.0)
    assert logs["slow_latent_event_precision_at_on_threshold"] == pytest.approx(1.0)
    assert logs["slow_latent_event_raw_recall_at_0p6"] == pytest.approx(1.0)
    assert logs["slow_latent_event_recall_at_0p6"] == pytest.approx(2 / 3)
    assert logs["slow_latent_event_precision_at_0p6"] == pytest.approx(1.0)


def test_stair_aux_logs_phase_mismatch_and_uses_positive_weight() -> None:
    """Stair-state aux loss should expose latent/env phase disagreement."""
    alg = _build_teacher_kl({"enabled": False})
    actor = cast(Any, alg.actor)
    actor.aux_event_coef = 0.0
    actor.aux_stair_coef = 0.1
    actor.aux_stair_pos_weight = 3.0
    actor.stair_confirm_steps = 2
    actor.aux_future_collision_risk_coef = 0.0
    actor.aux_future_safe_landing_quality_coef = 0.0
    actor.aux_stair_shape_coef = 0.0
    actor.aux_safe_stride_coef = 0.0

    stair_prob = torch.tensor([[0.1], [0.8], [0.7], [0.2]])
    actor.get_aux_outputs = lambda: {"stair_logit": torch.logit(stair_prob)}
    actor.get_slow_latent_diagnostics = lambda: {
        "stair_prob": stair_prob,
        "gate_mode": torch.tensor([[2.0], [0.0], [2.0], [1.0]]),
        "gate_event_trigger": torch.tensor([[0.0], [1.0], [0.0], [0.0]]),
        "gate_write_confirm": torch.tensor([[0.0], [1.0], [0.0], [0.0]]),
        "gate_memory_exit": torch.tensor([[0.0], [0.0], [1.0], [0.0]]),
    }
    labels = torch.zeros(4, 10)
    labels[:, 0] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    labels[:, 1] = torch.tensor([0.0, 1.0, 1.0, 0.0])
    dones = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    observations = TensorDict({"latent_labels": labels}, batch_size=[NUM_ENVS])

    loss, logs = alg._compute_slow_latent_aux_loss(
        RolloutStorage.Batch(
            observations=observations,
            hidden_states=(None, None),
            dones=dones,
        )
    )

    assert loss.item() > 0.0
    assert logs["slow_latent_stair_pos_weight"] == pytest.approx(3.0)
    assert logs["slow_latent_stair_on_threshold"] == pytest.approx(0.35)
    assert logs["slow_latent_stair_confirm_steps"] == pytest.approx(2.0)
    assert logs["slow_latent_stair_label_mean"] == pytest.approx(0.5)
    assert logs["slow_latent_stair_label_positive_count"] == pytest.approx(2.0)
    assert logs["slow_latent_stair_prob_pos_mean"] == pytest.approx(0.75)
    assert logs["slow_latent_stair_prob_neg_mean"] == pytest.approx(0.15)
    assert logs["slow_latent_stair_prob_gt_on_threshold_ratio"] == pytest.approx(0.5)
    assert logs["slow_latent_stair_recall_at_on_threshold"] == pytest.approx(1.0)
    assert logs["slow_latent_stair_precision_at_on_threshold"] == pytest.approx(1.0)
    assert logs["slow_latent_memory_while_env_normal_ratio"] == pytest.approx(0.25)
    assert logs["slow_latent_write_while_env_normal_ratio"] == pytest.approx(0.25)
    assert logs["slow_latent_env_stair_while_latent_normal_ratio"] == pytest.approx(0.25)
    assert logs["slow_latent_memory_while_env_stair_ratio"] == pytest.approx(0.25)
    assert logs["slow_latent_event_with_stair_label_rate"] == pytest.approx(1.0)
    assert logs["slow_latent_stair_recall_while_write"] == pytest.approx(0.0)
    assert logs["slow_latent_true_event_to_write_rate"] == pytest.approx(1.0)
    assert logs["slow_latent_write_trigger_event_precision"] == pytest.approx(1.0)
    assert logs["slow_latent_gate_confirm_while_env_stair_ratio"] == pytest.approx(0.25)
    assert logs["slow_latent_gate_confirm_stair_precision"] == pytest.approx(1.0)
    assert logs["slow_latent_true_event_to_memory_rate"] == pytest.approx(1.0)
    assert logs["slow_latent_false_memory_confirm_ratio"] == pytest.approx(0.0)
    assert logs["slow_latent_gate_exit_while_env_stair_ratio"] == pytest.approx(0.25)
    assert logs["slow_latent_gate_exit_stair_fraction"] == pytest.approx(1.0)
    assert logs["slow_latent_done_while_memory_ratio"] == pytest.approx(0.5)
    assert logs["slow_latent_done_while_env_normal_memory_ratio"] == pytest.approx(0.25)


def test_safe_stride_aux_loss_uses_seventh_label_as_valid_mask() -> None:
    """Safe-stride supervision should use label 5 and validity mask 6."""
    alg = _build_teacher_kl({"enabled": False})
    actor = cast(Any, alg.actor)
    actor.aux_event_coef = 0.0
    actor.aux_stair_coef = 0.0
    actor.aux_future_collision_risk_coef = 0.0
    actor.aux_future_safe_landing_quality_coef = 0.0
    actor.aux_stair_shape_coef = 0.0
    actor.aux_safe_stride_coef = 0.5
    actor.safe_stride_huber_delta = 0.05
    actor.get_aux_outputs = lambda: {"safe_stride": torch.tensor([[0.35], [10.0], [0.20], [10.0]])}
    actor.get_slow_latent_diagnostics = lambda: {}
    labels = torch.tensor([
        [0.0, 1.0, 0.30, 0.18, 1.0, 0.25, 1.0],
        [0.0, 1.0, 0.30, 0.18, 1.0, 0.25, 0.0],
        [0.0, 1.0, 0.30, 0.18, 1.0, 0.20, 1.0],
        [0.0, 1.0, 0.30, 0.18, 1.0, 0.25, 0.0],
    ])
    observations = TensorDict(
        {"latent_labels": labels},
        batch_size=[NUM_ENVS],
    )

    loss, logs = alg._compute_slow_latent_aux_loss(
        RolloutStorage.Batch(observations=observations, hidden_states=(None, None))
    )

    assert logs["slow_latent_safe_stride_huber"] == pytest.approx(0.0375)
    assert logs["slow_latent_safe_stride_valid_ratio"] == pytest.approx(0.5)
    assert logs["slow_latent_safe_stride_mae"] == pytest.approx(0.05)
    assert logs["slow_latent_safe_stride_out_of_range_label_ratio"] == 0.0
    assert loss.item() == pytest.approx(0.01875)


def test_stair_label_range_metrics_do_not_clamp_labels() -> None:
    """Range priors should be diagnostics only, including stride/tread violations."""
    alg = _build_teacher_kl({"enabled": False})
    actor = cast(Any, alg.actor)
    actor.aux_event_coef = 0.0
    actor.aux_stair_coef = 0.0
    actor.aux_future_collision_risk_coef = 0.0
    actor.aux_future_safe_landing_quality_coef = 0.0
    actor.aux_stair_shape_coef = 0.1
    actor.aux_safe_stride_coef = 0.1
    actor.tread_depth_min = 0.25
    actor.tread_depth_max = 0.35
    actor.riser_height_min = 0.088
    actor.riser_height_max = 0.25
    actor.safe_stride_min = 0.08
    actor.safe_stride_max = 0.45
    actor.get_aux_outputs = lambda: {
        "stair_shape": torch.tensor([
            [0.30, 0.18],
            [0.35, 0.18],
            [0.40, 0.18],
            [0.30, 0.18],
        ]),
        "safe_stride": torch.tensor([[0.40], [0.20], [0.20], [0.20]]),
    }
    actor.get_slow_latent_diagnostics = lambda: {}
    labels = torch.tensor([
        [0.0, 1.0, 0.10, 0.18, 1.0, 0.50, 1.0],
        [0.0, 1.0, 0.30, 0.30, 1.0, 0.20, 1.0],
        [0.0, 1.0, 0.60, 0.18, 0.0, 0.50, 1.0],
        [0.0, 1.0, 0.30, 0.18, 1.0, 0.60, 0.0],
    ])
    labels_before = labels.clone()
    observations = TensorDict({"latent_labels": labels}, batch_size=[NUM_ENVS])

    _loss, logs = alg._compute_slow_latent_aux_loss(
        RolloutStorage.Batch(observations=observations, hidden_states=(None, None))
    )

    torch.testing.assert_close(labels, labels_before)
    assert logs["slow_latent_tread_depth_out_of_range_label_ratio"] == pytest.approx(1 / 3)
    assert logs["slow_latent_riser_height_out_of_range_label_ratio"] == pytest.approx(1 / 3)
    assert logs["slow_latent_safe_stride_out_of_range_label_ratio"] == pytest.approx(2 / 3)
    assert logs["slow_latent_safe_stride_gt_tread_hat_ratio"] == pytest.approx(0.5)


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


def test_zero_lambda_without_logging_skips_teacher_observations() -> None:
    """A fully annealed teacher should leave the training hot path."""
    alg = _build_teacher_kl({
        "lambda_start": 0.0,
        "lambda_end": 0.0,
        "anneal_iters": 0,
        "log_kl_when_lambda_zero": False,
    })

    loss, logs = alg._compute_additional_loss(
        batch=RolloutStorage.Batch(observations=_build_obs(), hidden_states=(None, None)),
        original_batch_size=NUM_ENVS,
        distribution_params=(),
    )

    assert loss.item() == 0.0
    assert logs["teacher_kl_lambda"] == 0.0
    assert logs["teacher_guidance_active"] == 0.0
    assert alg.get_required_observation_groups() == ("actor", "critic")


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
