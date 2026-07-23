"""Tests for the DWAQ ablation model and task wiring."""

from __future__ import annotations

from typing import cast

import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

import mjlab.tasks  # noqa: F401
from mjlab.rl.config import (
  RslRlDwaqModelCfg,
  RslRlPpoDwaqTeacherKLAlgorithmCfg,
  RslRlTeacherKLRunnerCfg,
)
from mjlab.rl.dwaq_algorithm import DWAQPPOTeacherKL
from mjlab.rl.dwaq_model import DWAQMLPModel
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

DWAQ_TASK_ID = "Mjlab-Velocity-Blind-Rough-TargetNavigation-DWAQ-TeacherKL-Unitree-G1"


def _make_obs(
  num_envs: int = 4,
  actor_dim: int = 8,
  critic_dim: int = 13,
  history_length: int = 5,
) -> TensorDict:
  return TensorDict(
    {
      "actor": torch.randn(num_envs, actor_dim),
      "dwaq_history": torch.randn(num_envs, history_length, actor_dim),
      "dwaq_velocity_target": torch.randn(num_envs, 3),
      "critic": torch.randn(num_envs, critic_dim),
    },
    batch_size=[num_envs],
  )


def _make_actor(obs: TensorDict) -> DWAQMLPModel:
  return DWAQMLPModel(
    obs=obs,
    obs_groups={"actor": ["actor"], "dwaq_history": ["dwaq_history"]},
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
    encoder_hidden_dims=(12, 7),
    decoder_hidden_dims=(9,),
    velocity_dim=3,
    latent_dim=5,
    cenet_out_dim=8,
  )


def test_dwaq_actor_forward_and_cenet_outputs_match_g1dwaq_shapes() -> None:
  obs = _make_obs()
  actor = _make_actor(obs)

  actions = actor(obs, stochastic_output=False)
  outputs = actor.get_dwaq_outputs(obs, sample=True)

  assert actions.shape == (4, 3)
  assert actor.mlp[0].in_features == 16
  assert actor.history_obs_dim == 40
  assert outputs["code"].shape == (4, 8)
  assert outputs["code_vel"].shape == (4, 3)
  assert outputs["decode"].shape == (4, 8)
  assert outputs["mean_latent"].shape == (4, 5)
  assert outputs["logvar_latent"].shape == (4, 5)


def test_dwaq_algorithm_adds_autoencoder_loss_and_required_groups() -> None:
  obs = _make_obs()
  actor = _make_actor(obs)
  critic = MLPModel(
    obs=obs,
    obs_groups={"critic": ["critic"]},
    obs_set="critic",
    output_dim=1,
    hidden_dims=(16,),
    activation="elu",
    obs_normalization=False,
  )
  storage = RolloutStorage("rl", 4, 2, obs, [3], "cpu")
  alg = DWAQPPOTeacherKL(
    actor,
    critic,
    storage,
    teacher_kl_cfg={"enabled": False},
    num_mini_batches=2,
    device="cpu",
  )
  batch = RolloutStorage.Batch(observations=obs)

  loss, logs = alg._compute_additional_loss(batch, 4, ())

  assert torch.isfinite(loss)
  assert logs["dwaq_autoencoder"] > 0.0
  assert logs["dwaq_velocity"] >= 0.0
  assert logs["dwaq_reconstruction"] >= 0.0
  assert "dwaq_history" in alg.get_required_observation_groups()
  assert "dwaq_velocity_target" in alg.get_required_observation_groups()


def test_dwaq_ablation_is_the_only_registered_task_and_keeps_env_contract() -> None:
  assert list_tasks() == [DWAQ_TASK_ID]

  env_cfg = load_env_cfg(DWAQ_TASK_ID)
  play_cfg = load_env_cfg(DWAQ_TASK_ID, play=True)
  rl_cfg = cast(RslRlTeacherKLRunnerCfg, load_rl_cfg(DWAQ_TASK_ID))
  actor_cfg = cast(RslRlDwaqModelCfg, rl_cfg.actor)
  algorithm_cfg = cast(RslRlPpoDwaqTeacherKLAlgorithmCfg, rl_cfg.algorithm)

  assert "latent" not in env_cfg.observations
  assert "latent_labels" not in env_cfg.observations
  assert "reset_stair_latent_cache" not in env_cfg.events
  assert env_cfg.observations["actor"].history_length == 0
  assert env_cfg.observations["dwaq_history"].history_length == 5
  assert env_cfg.observations["dwaq_history"].flatten_history_dim is False
  assert tuple(env_cfg.observations["dwaq_history"].terms) == tuple(
    env_cfg.observations["actor"].terms
  )
  assert tuple(env_cfg.observations["dwaq_velocity_target"].terms) == ("base_lin_vel",)
  assert set(play_cfg.observations) == set(env_cfg.observations)
  assert "toe_step_riser_slab_penalty" in env_cfg.rewards
  assert "target_tread_midline_shaping" in env_cfg.rewards
  assert "teacher" in env_cfg.observations
  assert "camera" in env_cfg.observations

  assert actor_cfg.history_obs_set == "dwaq_history"
  assert actor_cfg.cenet_out_dim == 19
  assert actor_cfg.velocity_dim == 3
  assert actor_cfg.latent_dim == 16
  assert algorithm_cfg.dwaq_velocity_target_groups == ("dwaq_velocity_target",)
  assert rl_cfg.obs_groups == {
    "actor": ("actor",),
    "dwaq_history": ("dwaq_history",),
    "critic": ("critic",),
    "teacher": ("teacher", "camera"),
  }
