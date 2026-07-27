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
    dwaq_beta: float = 1.0,
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
    if batch.next_observations is None:
      raise RuntimeError(
        "DWAQ loss requires next observations. Configure "
        "algorithm.next_observation_groups to include the actor observation group."
      )

    get_dwaq_outputs = cast(
      Callable[..., dict[str, torch.Tensor]],
      get_dwaq_outputs_raw,
    )
    get_actor_observation = cast(
      Callable[[TensorDict], torch.Tensor],
      get_actor_observation_raw,
    )
    observations = cast(TensorDict, batch.observations[:original_batch_size])
    next_observations = cast(
      TensorDict,
      batch.next_observations[:original_batch_size],
    )
    outputs = get_dwaq_outputs(observations, sample=True)
    vel_target = torch.cat(
      [
        cast(torch.Tensor, observations[name])
        for name in self.dwaq_velocity_target_groups
      ],
      dim=-1,
    ).detach()
    decode_target = get_actor_observation(next_observations)
    obs_normalizer = getattr(actor, "obs_normalizer", None)
    if callable(obs_normalizer):
      normalizer = cast(Callable[[torch.Tensor], torch.Tensor], obs_normalizer)
      decode_target = normalizer(decode_target)
    decode_target = decode_target.detach()

    decode = outputs["decode"]
    mean_latent = outputs["mean_latent"]
    logvar_latent = torch.clamp(outputs["logvar_latent"], min=-10.0, max=10.0)
    dones = batch.dones[:original_batch_size] if batch.dones is not None else None

    velocity_loss = F.mse_loss(outputs["mean_vel"], vel_target)
    per_sample_reconstruction = (
      F.mse_loss(
        decode,
        decode_target,
        reduction="none",
      )
      .reshape(decode.shape[0], -1)
      .mean(dim=-1)
    )
    valid_next = torch.ones_like(per_sample_reconstruction, dtype=torch.bool)
    if dones is not None:
      valid_next = ~dones.reshape(-1).bool()
    if torch.any(valid_next):
      reconstruction_loss = per_sample_reconstruction[valid_next].mean()
    else:
      reconstruction_loss = per_sample_reconstruction.mean() * 0.0
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
        "dwaq_reconstruction_valid_ratio": float(
          valid_next.float().mean().detach().item()
        ),
      }
    )
    return loss, log_dict

  def get_required_observation_groups(self) -> tuple[str, ...]:
    """Also request the explicit privileged velocity target for DWAQ."""
    groups = set(super().get_required_observation_groups())
    groups.update(self.dwaq_velocity_target_groups)
    return tuple(sorted(groups))
