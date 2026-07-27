"""PPO + TeacherKL with the DWAQ beta-VAE auxiliary loss."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
import torch.nn.functional as F
from rsl_rl.algorithms.ppo_teacher_kl import PPOTeacherKL
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict


class DWAQPPOTeacherKL(PPOTeacherKL):
  """Keep local TeacherKL training and add the G1DWAQ_Lab VAE loss."""

  def __init__(
    self,
    *args: Any,
    dwaq_velocity_target_groups: Sequence[str] = ("dwaq_velocity_target",),
    dwaq_beta: float = 0.01,
    dwaq_autoencoder_loss_coef: float = 1.0,
    dwaq_velocity_loss_coef: float = 1.0,
    dwaq_reconstruction_loss_coef: float = 1.0,
    **kwargs: Any,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.dwaq_velocity_target_groups = tuple(dwaq_velocity_target_groups)
    self.dwaq_beta = float(dwaq_beta)
    self.dwaq_autoencoder_loss_coef = float(dwaq_autoencoder_loss_coef)
    self.dwaq_velocity_loss_coef = float(dwaq_velocity_loss_coef)
    self.dwaq_reconstruction_loss_coef = float(dwaq_reconstruction_loss_coef)

  def _compute_additional_loss(
    self,
    batch: RolloutStorage.Batch,
    original_batch_size: int,
    distribution_params: tuple[torch.Tensor, ...],
  ) -> tuple[torch.Tensor, dict[str, float]]:
    """Add DWAQ beta-VAE loss on top of the inherited TeacherKL loss."""
    base_loss, log_dict = super()._compute_additional_loss(
      batch,
      original_batch_size,
      distribution_params,
    )
    if self.dwaq_autoencoder_loss_coef == 0.0:
      return base_loss, log_dict

    actor = self.actor
    get_dwaq_outputs_raw = getattr(actor, "get_dwaq_outputs", None)
    get_actor_observation_raw = getattr(actor, "get_actor_observation", None)
    if not callable(get_dwaq_outputs_raw) or not callable(get_actor_observation_raw):
      return base_loss, log_dict
    if batch.observations is None:
      raise RuntimeError("DWAQ loss requires observations in the rollout batch.")

    get_dwaq_outputs = cast(
      Callable[..., dict[str, torch.Tensor]],
      get_dwaq_outputs_raw,
    )
    get_actor_observation = cast(
      Callable[[TensorDict], torch.Tensor],
      get_actor_observation_raw,
    )
    observations = cast(TensorDict, batch.observations[:original_batch_size])
    outputs = get_dwaq_outputs(observations, sample=True)
    vel_target = torch.cat(
      [
        cast(torch.Tensor, observations[name])
        for name in self.dwaq_velocity_target_groups
      ],
      dim=-1,
    ).detach()
    decode_target = get_actor_observation(observations).detach()

    decode = outputs["decode"]
    mean_latent = outputs["mean_latent"]
    logvar_latent = torch.clamp(outputs["logvar_latent"], min=-10.0, max=10.0)

    velocity_loss = F.mse_loss(outputs["code_vel"], vel_target)
    reconstruction_loss = F.mse_loss(decode, decode_target)
    kl_divergence = (
      -0.5
      * (1 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp())
      .sum(dim=-1)
      .mean()
    )
    autoencoder_loss = (
      self.dwaq_velocity_loss_coef * velocity_loss
      + self.dwaq_reconstruction_loss_coef * reconstruction_loss
      + self.dwaq_beta * kl_divergence
    )
    loss = base_loss + self.dwaq_autoencoder_loss_coef * autoencoder_loss
    log_dict.update(
      {
        "dwaq_autoencoder": float(autoencoder_loss.detach().item()),
        "dwaq_velocity": float(velocity_loss.detach().item()),
        "dwaq_reconstruction": float(reconstruction_loss.detach().item()),
        "dwaq_kl": float(kl_divergence.detach().item()),
        "dwaq_latent_mean_abs": float(mean_latent.abs().mean().detach().item()),
        "dwaq_latent_std": float(mean_latent.std(unbiased=False).detach().item()),
        "dwaq_velocity_std": float(
          outputs["mean_vel"].std(unbiased=False).detach().item()
        ),
      }
    )
    return loss, log_dict

  def get_required_observation_groups(self) -> tuple[str, ...]:
    """Also request the explicit privileged velocity target for DWAQ."""
    groups = set(super().get_required_observation_groups())
    groups.update(self.dwaq_velocity_target_groups)
    return tuple(sorted(groups))
