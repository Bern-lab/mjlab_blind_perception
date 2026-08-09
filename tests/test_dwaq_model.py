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


class _FixedOutputDWAQActor(DWAQMLPModel):
  fixed_decode: torch.Tensor
  fixed_mean_vel: torch.Tensor
  fixed_mean_latent: torch.Tensor
  fixed_logvar_latent: torch.Tensor

  def set_fixed_outputs(
    self,
    *,
    decode: torch.Tensor,
    mean_vel: torch.Tensor,
    mean_latent: torch.Tensor,
    logvar_latent: torch.Tensor,
  ) -> None:
    self.fixed_decode = decode
    self.fixed_mean_vel = mean_vel
    self.fixed_mean_latent = mean_latent
    self.fixed_logvar_latent = logvar_latent

  def get_dwaq_outputs(
    self,
    obs: TensorDict,
    sample: bool | None = None,
  ) -> dict[str, torch.Tensor]:
    del sample
    batch_size = obs.batch_size[0]
    mean_vel = self.fixed_mean_vel[:batch_size]
    mean_latent = self.fixed_mean_latent[:batch_size]
    code = torch.cat((mean_vel, mean_latent), dim=-1)
    return {
      "code": code,
      "code_vel": mean_vel,
      "decode": self.fixed_decode[:batch_size],
      "mean_vel": mean_vel,
      "logvar_vel": torch.zeros_like(mean_vel),
      "mean_latent": mean_latent,
      "logvar_latent": self.fixed_logvar_latent[:batch_size],
    }


def _make_fixed_actor(obs: TensorDict) -> _FixedOutputDWAQActor:
  return _FixedOutputDWAQActor(
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


def test_dwaq_velocity_code_samples_during_training_and_uses_mean_for_eval() -> None:
  obs = _make_obs()
  actor = _make_actor(obs)
  torch.manual_seed(0)

  outputs = actor.get_dwaq_outputs(obs, sample=True)
  eval_outputs = actor.get_dwaq_outputs(obs, sample=False)

  assert outputs["logvar_vel"].shape == outputs["mean_vel"].shape
  assert not torch.allclose(outputs["code_vel"], outputs["mean_vel"])
  assert torch.allclose(eval_outputs["code_vel"], eval_outputs["mean_vel"])


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
  batch = RolloutStorage.Batch(
    observations=obs,
    dones=torch.zeros(4, 1, dtype=torch.uint8),
  )

  loss, logs = alg._compute_additional_loss(batch, 4, ())

  assert torch.isfinite(loss)
  assert logs["dwaq_autoencoder"] > 0.0
  assert logs["dwaq_velocity"] >= 0.0
  assert logs["dwaq_reconstruction"] >= 0.0
  assert "dwaq_history" in alg.get_required_observation_groups()
  assert "dwaq_velocity_target" in alg.get_required_observation_groups()


def test_dwaq_reconstructs_current_observation_like_g1dwaq_lab() -> None:
  obs = _make_obs(num_envs=4, actor_dim=2)
  obs["actor"] = torch.tensor(
    [
      [1.0, 1.0],
      [2.0, 2.0],
      [3.0, 3.0],
      [4.0, 4.0],
    ]
  )
  obs["dwaq_velocity_target"] = torch.zeros(4, 3)
  actor = _make_fixed_actor(obs)
  actor.set_fixed_outputs(
    decode=obs["actor"].clone(),
    mean_vel=torch.zeros(4, 3),
    mean_latent=torch.zeros(4, 5),
    logvar_latent=torch.zeros(4, 5),
  )
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
    num_mini_batches=1,
    dwaq_beta=0.0,
    dwaq_velocity_loss_coef=0.0,
    device="cpu",
  )
  batch = RolloutStorage.Batch(
    observations=obs,
    dones=torch.tensor([[0], [0], [0], [1]], dtype=torch.uint8),
  )

  _loss, logs = alg._compute_additional_loss(batch, 4, ())

  assert logs["dwaq_reconstruction"] == 0.0
  assert "dwaq_reconstruction_valid_ratio" not in logs


def test_dwaq_kl_is_batch_size_invariant() -> None:
  def compute_kl(num_envs: int) -> float:
    obs = _make_obs(num_envs=num_envs, actor_dim=2)
    obs["actor"] = torch.zeros(num_envs, 2)
    obs["dwaq_velocity_target"] = torch.zeros(num_envs, 3)
    actor = _make_fixed_actor(obs)
    actor.set_fixed_outputs(
      decode=torch.zeros(num_envs, 2),
      mean_vel=torch.zeros(num_envs, 3),
      mean_latent=torch.ones(num_envs, 5),
      logvar_latent=torch.zeros(num_envs, 5),
    )
    critic = MLPModel(
      obs=obs,
      obs_groups={"critic": ["critic"]},
      obs_set="critic",
      output_dim=1,
      hidden_dims=(16,),
      activation="elu",
      obs_normalization=False,
    )
    storage = RolloutStorage("rl", num_envs, 2, obs, [3], "cpu")
    alg = DWAQPPOTeacherKL(
      actor,
      critic,
      storage,
      teacher_kl_cfg={"enabled": False},
      num_mini_batches=1,
      dwaq_velocity_loss_coef=0.0,
      dwaq_reconstruction_loss_coef=0.0,
      device="cpu",
    )
    batch = RolloutStorage.Batch(
      observations=obs,
      dones=torch.zeros(num_envs, 1, dtype=torch.uint8),
    )
    _loss, logs = alg._compute_additional_loss(batch, num_envs, ())
    return logs["dwaq_kl"]

  assert compute_kl(2) == compute_kl(4)
  assert compute_kl(2) == 2.5


def test_dwaq_ablation_is_the_only_registered_task_and_keeps_env_contract() -> None:
  assert list_tasks() == [DWAQ_TASK_ID]

  env_cfg = load_env_cfg(DWAQ_TASK_ID)
  play_cfg = load_env_cfg(DWAQ_TASK_ID, play=True)
  rl_cfg = cast(RslRlTeacherKLRunnerCfg, load_rl_cfg(DWAQ_TASK_ID))
  actor_cfg = cast(RslRlDwaqModelCfg, rl_cfg.actor)
  algorithm_cfg = cast(RslRlPpoDwaqTeacherKLAlgorithmCfg, rl_cfg.algorithm)

  assert "latent" not in env_cfg.observations
  assert "latent_labels" not in env_cfg.observations
  assert "stair_phase_state" in env_cfg.observations
  assert "foot_event_memory" in env_cfg.observations["stair_phase_state"].terms
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
  assert "stair_stride_phase_reward" in env_cfg.rewards
  phase_params = env_cfg.rewards["stair_stride_phase_reward"].params
  assert phase_params["event_observation_group_name"] == "stair_phase_state"
  assert phase_params["event_observation_term_name"] == "foot_event_memory"
  assert "target_tread_midline_shaping" in env_cfg.rewards
  assert "teacher" in env_cfg.observations
  assert "camera" in env_cfg.observations

  assert actor_cfg.history_obs_set == "dwaq_history"
  assert actor_cfg.cenet_out_dim == 19
  assert actor_cfg.velocity_dim == 3
  assert actor_cfg.latent_dim == 16
  assert algorithm_cfg.dwaq_velocity_target_groups == ("dwaq_velocity_target",)
  assert algorithm_cfg.dwaq_beta == 0.01
  assert not hasattr(algorithm_cfg, "next_observation_groups")
  assert rl_cfg.num_steps_per_env == 64
  assert rl_cfg.obs_groups == {
    "actor": ("actor",),
    "dwaq_history": ("dwaq_history",),
    "critic": ("critic",),
    "teacher": ("teacher", "camera"),
  }
