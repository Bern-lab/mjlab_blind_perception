"""Online Stage 2D foot-event detector training with a frozen policy rollout."""

from __future__ import annotations

import copy
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from scripts.velocity_eval.export_foot_event_detector_dataset import (
  DEFAULT_STAGE2D_CHECKPOINT,
  FOOT_EVENT_LABEL_NAMES,
  foot_event_detector_obs,
  foot_event_detector_obs_dim,
  foot_event_input_feature_groups,
  foot_event_labels_from_env,
)
from scripts.velocity_eval.export_stair_probe_dataset import (
  STAIR_CURRENT_GROUND_CONTACT_KEY,
  STAIR_CURRENT_STAIR_SUPPORT_KEY,
  STAIR_CURRENT_SUPPORT_FRACTION_KEY,
  StairProbeHistoryBuffer,
  _close_sequence_logger,
  _tensor_extra,
)
from scripts.velocity_eval.policy_io import (
  load_inference_policy,
  resolve_checkpoint_path,
)
from scripts.velocity_eval.train_foot_event_detector import (
  FootEventDetectorGRU,
  compute_metrics,
  event_f1_for_label,
  export_detector_onnx,
  make_pos_weight,
  metric_is_higher_better,
  sweep_deployment_event_thresholds,
)
from tqdm.auto import tqdm

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.lstm import reset_policy_state_from_step
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class OnlineFootEventDetectorConfig:
  """Configuration for online supervised event-detector training."""

  checkpoint_file: str | None = DEFAULT_STAGE2D_CHECKPOINT
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  output_dir: str = (
    "eval_outputs/stair_stage2/model51000_seed42_foot_event_detector_online_v1"
  )
  num_envs: int = 512
  steps: int = 5000
  seed: int = 42
  device: str | None = None
  history_len: int = 16
  include_gait_phase: bool = True
  gait_period: float = 0.6
  command_name: str = "twist"
  expected_obs_dim: int = 93
  val_env_fraction: float = 0.2
  train_buffer_capacity: int = 120_000
  val_buffer_capacity: int = 40_000
  warmup_samples: int = 16_384
  batch_size: int = 512
  train_every_steps: int = 1
  updates_per_train: int = 1
  eval_interval_steps: int = 250
  min_val_samples: int = 4096
  learning_rate: float = 1.0e-3
  weight_decay: float = 1.0e-4
  frame_hidden_dim: int = 128
  recurrent_hidden_dim: int = 64
  head_hidden_dim: int = 32
  dropout: float = 0.0
  threshold: float = 0.5
  event_tolerance_frames: int = 2
  pos_weight_max: float = 100.0
  toe_hit_pos_weight: float | None = 150.0
  toe_positive_fraction: float = 0.125
  touchdown_positive_fraction: float = 0.25
  stair_hard_negative_fraction: float = 0.25
  false_positive_hard_negative_fraction: float = 0.125
  soft_touchdown_radius: int = 1
  soft_toe_hit_radius: int = 2
  soft_event_radius1_value: float = 0.7
  soft_event_radius2_value: float = 0.4
  focal_loss_weight: float = 0.5
  tversky_loss_weight: float = 0.5
  focal_gamma_pos: float = 0.0
  focal_gamma_neg: float = 4.0
  tversky_alpha: float = 0.5
  tversky_beta: float = 0.5
  mine_false_positive_hard_negatives: bool = True
  hard_negative_mining_threshold: float = 0.5
  hard_negative_mining_window_frames: int = 8
  hard_negative_mining_max_peaks: int = 4096
  touchdown_contact_threshold: float = 0.7
  touchdown_contact_release_threshold: float = 0.35
  touchdown_cooldown_frames: int = 8
  toe_hit_cooldown_frames: int = 8
  sweep_threshold_min: float = 0.1
  sweep_threshold_max: float = 0.99
  sweep_threshold_steps: int = 19
  selection_metric: str = "deploy_event_macro_f1"
  max_updates: int | None = 5000
  early_stop_patience_evals: int = 8
  early_stop_min_delta: float = 1.0e-4
  export_onnx: bool = True
  progress: bool = True


class OnlineFootEventReplayBuffer:
  """Fixed-size ring buffer for online detector samples."""

  def __init__(
    self,
    *,
    capacity: int,
    history_len: int,
    obs_dim: int,
    label_dim: int,
    num_envs: int,
    soft_touchdown_radius: int,
    soft_toe_hit_radius: int,
    soft_event_radius1_value: float,
    soft_event_radius2_value: float,
    device: torch.device,
  ) -> None:
    if capacity <= 0:
      raise ValueError("capacity must be positive.")
    self.capacity = int(capacity)
    self.device = device
    self.obs_history = torch.empty(
      capacity,
      history_len,
      obs_dim,
      dtype=torch.float32,
      device=device,
    )
    self.labels = torch.empty(capacity, label_dim, dtype=torch.float32, device=device)
    self.train_labels = torch.empty(
      capacity,
      label_dim,
      dtype=torch.float32,
      device=device,
    )
    self.episode_id = torch.empty(capacity, dtype=torch.int64, device=device)
    self.frame_idx = torch.empty(capacity, dtype=torch.int64, device=device)
    self.env_id = torch.empty(capacity, dtype=torch.int64, device=device)
    self.stair_support = torch.empty(capacity, 2, dtype=torch.bool, device=device)
    self.support_fraction = torch.empty(
      capacity,
      2,
      dtype=torch.float32,
      device=device,
    )
    self.stair_hard_negative = torch.empty(capacity, dtype=torch.bool, device=device)
    self.false_positive_hard_negative = torch.empty(
      capacity,
      dtype=torch.bool,
      device=device,
    )
    self._write_pos = 0
    self._size = 0
    self._soft_radius = torch.zeros(label_dim, dtype=torch.int64, device=device)
    self._soft_radius[2:4] = int(soft_touchdown_radius)
    self._soft_radius[4:6] = int(soft_toe_hit_radius)
    self._soft_values = torch.tensor(
      [1.0, soft_event_radius1_value, soft_event_radius2_value],
      dtype=torch.float32,
      device=device,
    )
    self._max_soft_radius = int(max(soft_touchdown_radius, soft_toe_hit_radius, 0))
    self._recent_indices_by_env = torch.full(
      (num_envs, max(self._max_soft_radius, 1)),
      -1,
      dtype=torch.int64,
      device=device,
    )
    self._recent_episode_by_env = torch.full(
      (num_envs, max(self._max_soft_radius, 1)),
      -1,
      dtype=torch.int64,
      device=device,
    )

  @property
  def size(self) -> int:
    return self._size

  def add(
    self,
    *,
    obs_history: torch.Tensor,
    labels: torch.Tensor,
    train_labels: torch.Tensor,
    episode_id: torch.Tensor,
    frame_idx: torch.Tensor,
    env_id: torch.Tensor,
    stair_support: torch.Tensor,
    support_fraction: torch.Tensor,
  ) -> torch.Tensor:
    n = int(obs_history.shape[0])
    if n == 0:
      return torch.empty(0, dtype=torch.int64, device=self.device)
    if n >= self.capacity:
      start = n - self.capacity
      obs_history = obs_history[start:]
      labels = labels[start:]
      train_labels = train_labels[start:]
      episode_id = episode_id[start:]
      frame_idx = frame_idx[start:]
      env_id = env_id[start:]
      stair_support = stair_support[start:]
      support_fraction = support_fraction[start:]
      n = self.capacity

    first = min(n, self.capacity - self._write_pos)
    second = n - first
    write_indices = [
      torch.arange(self._write_pos, self._write_pos + first, device=self.device)
    ]
    self._write_slice(
      slice(self._write_pos, self._write_pos + first),
      obs_history[:first],
      labels[:first],
      train_labels[:first],
      episode_id[:first],
      frame_idx[:first],
      env_id[:first],
      stair_support[:first],
      support_fraction[:first],
    )
    if second > 0:
      write_indices.append(torch.arange(0, second, device=self.device))
      self._write_slice(
        slice(0, second),
        obs_history[first:],
        labels[first:],
        train_labels[first:],
        episode_id[first:],
        frame_idx[first:],
        env_id[first:],
        stair_support[first:],
        support_fraction[first:],
      )
    indices = torch.cat(write_indices, dim=0).to(dtype=torch.int64)
    self._retroactively_soften_previous_events(
      indices=indices,
      labels=labels,
      episode_id=episode_id,
      env_id=env_id,
    )
    self._push_recent_indices(
      indices=indices,
      episode_id=episode_id,
      env_id=env_id,
    )
    self._write_pos = (self._write_pos + n) % self.capacity
    self._size = min(self.capacity, self._size + n)
    return indices

  def _write_slice(
    self,
    slc: slice,
    obs_history: torch.Tensor,
    labels: torch.Tensor,
    train_labels: torch.Tensor,
    episode_id: torch.Tensor,
    frame_idx: torch.Tensor,
    env_id: torch.Tensor,
    stair_support: torch.Tensor,
    support_fraction: torch.Tensor,
  ) -> None:
    self.obs_history[slc].copy_(obs_history.detach())
    self.labels[slc].copy_(labels.detach())
    self.train_labels[slc].copy_(train_labels.detach())
    self.episode_id[slc].copy_(episode_id.detach())
    self.frame_idx[slc].copy_(frame_idx.detach())
    self.env_id[slc].copy_(env_id.detach())
    self.stair_support[slc].copy_(stair_support.detach())
    self.support_fraction[slc].copy_(support_fraction.detach())
    event_negative = labels[:, 2:6].amax(dim=1) <= 0.5
    self.stair_hard_negative[slc].copy_(stair_support.any(dim=1) & event_negative)
    self.false_positive_hard_negative[slc] = False

  def _soft_value(self, distance: int) -> torch.Tensor:
    if distance < self._soft_values.numel():
      return self._soft_values[distance]
    return self._soft_values[-1]

  def _retroactively_soften_previous_events(
    self,
    *,
    indices: torch.Tensor,
    labels: torch.Tensor,
    episode_id: torch.Tensor,
    env_id: torch.Tensor,
  ) -> None:
    if self._max_soft_radius <= 0:
      return
    event_cols = (2, 3, 4, 5)
    for row in range(int(indices.numel())):
      env = int(env_id[row].item())
      episode = int(episode_id[row].item())
      for label_index in event_cols:
        if float(labels[row, label_index].item()) <= 0.5:
          continue
        radius = int(self._soft_radius[label_index].item())
        for distance in range(1, radius + 1):
          prev_index = int(self._recent_indices_by_env[env, distance - 1].item())
          prev_episode = int(self._recent_episode_by_env[env, distance - 1].item())
          if prev_index < 0 or prev_episode != episode:
            continue
          value = self._soft_value(distance)
          self.train_labels[prev_index, label_index] = torch.maximum(
            self.train_labels[prev_index, label_index],
            value,
          )

  def _push_recent_indices(
    self,
    *,
    indices: torch.Tensor,
    episode_id: torch.Tensor,
    env_id: torch.Tensor,
  ) -> None:
    if self._max_soft_radius <= 0:
      return
    for row in range(int(indices.numel())):
      env = int(env_id[row].item())
      if self._max_soft_radius > 1:
        self._recent_indices_by_env[env, 1:] = self._recent_indices_by_env[
          env, :-1
        ].clone()
        self._recent_episode_by_env[env, 1:] = self._recent_episode_by_env[
          env, :-1
        ].clone()
      self._recent_indices_by_env[env, 0] = indices[row]
      self._recent_episode_by_env[env, 0] = episode_id[row]

  def sample(
    self,
    batch_size: int,
    *,
    generator: torch.Generator,
    toe_positive_fraction: float,
    touchdown_positive_fraction: float,
    stair_hard_negative_fraction: float,
    false_positive_hard_negative_fraction: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if self._size <= 0:
      raise RuntimeError("Cannot sample from an empty buffer.")
    toe_count = int(round(batch_size * toe_positive_fraction))
    touchdown_count = int(round(batch_size * touchdown_positive_fraction))
    stair_hard_count = int(round(batch_size * stair_hard_negative_fraction))
    false_positive_count = int(
      round(batch_size * false_positive_hard_negative_fraction)
    )
    reserved = toe_count + touchdown_count + stair_hard_count + false_positive_count
    if reserved > batch_size:
      scale = batch_size / max(float(reserved), 1.0)
      toe_count = int(round(toe_count * scale))
      touchdown_count = int(round(touchdown_count * scale))
      stair_hard_count = int(round(stair_hard_count * scale))
      false_positive_count = int(round(false_positive_count * scale))
      reserved = toe_count + touchdown_count + stair_hard_count + false_positive_count
    random_count = batch_size - reserved
    chunks = [
      self._sample_mask(
        self.labels[: self._size, 2:4].amax(dim=1) > 0.5, touchdown_count, generator
      ),
      self._sample_mask(
        self.labels[: self._size, 4:6].amax(dim=1) > 0.5, toe_count, generator
      ),
      self._sample_mask(
        self.stair_hard_negative[: self._size],
        stair_hard_count,
        generator,
      ),
      self._sample_mask(
        self.false_positive_hard_negative[: self._size],
        false_positive_count,
        generator,
      ),
      self._sample_random(random_count, generator=generator),
    ]
    indices = torch.cat([chunk for chunk in chunks if chunk.numel() > 0], dim=0)
    if indices.numel() < batch_size:
      extra = self._sample_random(
        batch_size - int(indices.numel()), generator=generator
      )
      indices = torch.cat((indices, extra), dim=0)
    perm = torch.randperm(indices.numel(), generator=generator, device=self.device)
    indices = indices[perm[:batch_size]]
    return self.obs_history[indices], self.train_labels[indices]

  def _sample_random(
    self,
    count: int,
    *,
    generator: torch.Generator,
  ) -> torch.Tensor:
    if count <= 0:
      return torch.empty(0, dtype=torch.int64, device=self.device)
    return torch.randint(
      self._size,
      (count,),
      generator=generator,
      device=self.device,
    )

  def _sample_mask(
    self,
    mask: torch.Tensor,
    count: int,
    generator: torch.Generator,
  ) -> torch.Tensor:
    if count <= 0:
      return torch.empty(0, dtype=torch.int64, device=self.device)
    candidates = mask.nonzero(as_tuple=False).squeeze(-1)
    if candidates.numel() == 0:
      return self._sample_random(count, generator=generator)
    sample_ids = torch.randint(
      int(candidates.numel()),
      (count,),
      generator=generator,
      device=self.device,
    )
    return candidates[sample_ids]

  def snapshot(self) -> dict[str, np.ndarray]:
    """Return the current buffer as CPU numpy arrays."""
    size = self._size
    return {
      "obs_history": self.obs_history[:size].detach().cpu().numpy(),
      "labels": self.labels[:size].detach().cpu().numpy(),
      "train_labels": self.train_labels[:size].detach().cpu().numpy(),
      "episode_id": self.episode_id[:size].detach().cpu().numpy(),
      "frame_idx": self.frame_idx[:size].detach().cpu().numpy(),
      "env_id": self.env_id[:size].detach().cpu().numpy(),
      "stair_support": self.stair_support[:size].detach().cpu().numpy(),
      "support_fraction": self.support_fraction[:size].detach().cpu().numpy(),
      "stair_hard_negative": self.stair_hard_negative[:size].detach().cpu().numpy(),
      "false_positive_hard_negative": (
        self.false_positive_hard_negative[:size].detach().cpu().numpy()
      ),
    }

  def mark_false_positive_hard_negatives(self, mask: np.ndarray) -> int:
    """Mark stored samples that should be replayed as false-positive hard negatives."""
    if mask.shape != (self._size,):
      raise ValueError(
        f"Expected false-positive mask shape {(self._size,)}, got {mask.shape}."
      )
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
    before = int(self.false_positive_hard_negative[: self._size].sum().item())
    self.false_positive_hard_negative[: self._size] |= mask_tensor
    after = int(self.false_positive_hard_negative[: self._size].sum().item())
    return after - before

  def label_audit_rows(self, prefix: str) -> list[tuple[str, str]]:
    labels = self.labels[: self._size]
    rows = [(f"{prefix}_samples", str(self._size))]
    for index, name in enumerate(FOOT_EVENT_LABEL_NAMES):
      count = float(labels[:, index].sum().item()) if self._size else 0.0
      rows.append((f"{prefix}_{name}_positive_count", f"{count:.0f}"))
      rows.append(
        (
          f"{prefix}_{name}_positive_rate",
          f"{count / max(float(self._size), 1.0):.6g}",
        )
      )
    stair_hard = int(self.stair_hard_negative[: self._size].sum().item())
    false_positive_hard = int(
      self.false_positive_hard_negative[: self._size].sum().item()
    )
    rows.append((f"{prefix}_stair_hard_negative_count", str(stair_hard)))
    rows.append(
      (f"{prefix}_false_positive_hard_negative_count", str(false_positive_hard))
    )
    return rows


def _soft_event_value(
  distance: int, *, radius1_value: float, radius2_value: float
) -> float:
  if distance <= 0:
    return 1.0
  if distance == 1:
    return float(radius1_value)
  return float(radius2_value)


def _soften_labels_from_recent_events(
  labels: torch.Tensor,
  recent_event_age: torch.Tensor,
  reset_mask: torch.Tensor,
  *,
  touchdown_radius: int,
  toe_hit_radius: int,
  radius1_value: float,
  radius2_value: float,
) -> torch.Tensor:
  """Return soft event labels using current and recently observed events."""
  recent_event_age[reset_mask, :] = 10_000
  recent_event_age += 1
  train_labels = labels.clone()
  event_cols = (2, 3, 4, 5)
  radii = {
    2: int(touchdown_radius),
    3: int(touchdown_radius),
    4: int(toe_hit_radius),
    5: int(toe_hit_radius),
  }
  for label_index in event_cols:
    event_now = labels[:, label_index] > 0.5
    recent_event_age[:, label_index] = torch.where(
      event_now,
      torch.zeros_like(recent_event_age[:, label_index]),
      recent_event_age[:, label_index],
    )
    for distance in range(1, radii[label_index] + 1):
      value = _soft_event_value(
        distance,
        radius1_value=radius1_value,
        radius2_value=radius2_value,
      )
      near_event = recent_event_age[:, label_index] == distance
      train_labels[:, label_index] = torch.maximum(
        train_labels[:, label_index],
        torch.where(
          near_event,
          torch.full_like(train_labels[:, label_index], value),
          torch.zeros_like(train_labels[:, label_index]),
        ),
      )
  return train_labels


def _asymmetric_focal_loss_with_logits(
  logits: torch.Tensor,
  targets: torch.Tensor,
  *,
  gamma_pos: float,
  gamma_neg: float,
) -> torch.Tensor:
  probabilities = torch.sigmoid(logits)
  ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
  pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
  gamma = gamma_pos * targets + gamma_neg * (1.0 - targets)
  return ((1.0 - pt).clamp_min(0.0).pow(gamma) * ce).mean()


def _tversky_loss_from_logits(
  logits: torch.Tensor,
  targets: torch.Tensor,
  *,
  alpha: float,
  beta: float,
  eps: float = 1.0e-6,
) -> torch.Tensor:
  probabilities = torch.sigmoid(logits)
  dims = tuple(range(max(probabilities.ndim - 1, 1)))
  true_positive = (probabilities * targets).sum(dim=dims)
  false_positive = (probabilities * (1.0 - targets)).sum(dim=dims)
  false_negative = ((1.0 - probabilities) * targets).sum(dim=dims)
  score = (true_positive + eps) / (
    true_positive + alpha * false_positive + beta * false_negative + eps
  )
  return (1.0 - score).mean()


def _training_loss(
  *,
  logits: torch.Tensor,
  labels: torch.Tensor,
  pos_weight: torch.Tensor,
  focal_loss_weight: float,
  tversky_loss_weight: float,
  focal_gamma_pos: float,
  focal_gamma_neg: float,
  tversky_alpha: float,
  tversky_beta: float,
) -> torch.Tensor:
  bce = F.binary_cross_entropy_with_logits(
    logits,
    labels,
    pos_weight=pos_weight,
  )
  event_logits = logits[:, 2:6]
  event_labels = labels[:, 2:6]
  focal = _asymmetric_focal_loss_with_logits(
    event_logits,
    event_labels,
    gamma_pos=focal_gamma_pos,
    gamma_neg=focal_gamma_neg,
  )
  tversky = _tversky_loss_from_logits(
    event_logits,
    event_labels,
    alpha=tversky_alpha,
    beta=tversky_beta,
  )
  return bce + focal_loss_weight * focal + tversky_loss_weight * tversky


def _touchdown_group_metrics(
  *,
  labels: np.ndarray,
  probabilities: np.ndarray,
  episode_id: np.ndarray,
  frame_idx: np.ndarray,
  stair_support: np.ndarray,
  support_fraction: np.ndarray,
  threshold: float,
  event_tolerance_frames: int,
) -> dict[str, float]:
  """Add stair-only touchdown and support-fraction diagnostics."""
  metrics: dict[str, float] = {}
  for foot_id, side in enumerate(("left", "right")):
    label_index = 2 + foot_id
    stair_mask = stair_support[:, foot_id].astype(np.bool_)
    if stair_mask.any():
      stair_labels = labels[stair_mask, label_index]
      stair_probs = probabilities[stair_mask, label_index]
      pred = stair_probs >= threshold
      lab = stair_labels > 0.5
      tp = float(np.logical_and(pred, lab).sum())
      fp = float(np.logical_and(pred, ~lab).sum())
      fn = float(np.logical_and(~pred, lab).sum())
      precision = tp / max(tp + fp, 1.0)
      recall = tp / max(tp + fn, 1.0)
      metrics[f"{side}_touchdown_stair_precision"] = precision
      metrics[f"{side}_touchdown_stair_recall"] = recall
      metrics[f"{side}_touchdown_stair_f1"] = (
        0.0
        if precision + recall <= 0.0
        else 2.0 * precision * recall / (precision + recall)
      )
      event_stats = event_f1_for_label(
        stair_labels,
        stair_probs,
        episode_id[stair_mask],
        frame_idx[stair_mask],
        threshold=threshold,
        tolerance_frames=event_tolerance_frames,
      )
      for key, value in event_stats.items():
        metrics[f"{side}_touchdown_stair_{key}"] = float(value)
    touchdown_mask = labels[:, label_index] > 0.5
    stair_touchdown = touchdown_mask & stair_mask
    metrics[f"{side}_touchdown_on_stair_count"] = float(stair_touchdown.sum())
    if stair_touchdown.any():
      frac = support_fraction[stair_touchdown, foot_id]
      metrics[f"{side}_touchdown_on_stair_support_fraction_mean"] = float(np.mean(frac))
      metrics[f"{side}_touchdown_on_stair_support_fraction_p10"] = float(
        np.quantile(frac, 0.10)
      )
      for threshold_value in (0.3, 0.5, 0.8):
        metrics[
          f"{side}_touchdown_on_stair_support_fraction_lt_{threshold_value:g}"
        ] = float(np.mean(frac < threshold_value))
  return metrics


def _deployment_event_metrics(
  *,
  labels: np.ndarray,
  probabilities: np.ndarray,
  episode_id: np.ndarray,
  frame_idx: np.ndarray,
  stair_support: np.ndarray,
  thresholds: np.ndarray,
  touchdown_contact_threshold: float,
  touchdown_contact_release_threshold: float,
  touchdown_cooldown_frames: int,
  toe_hit_cooldown_frames: int,
  event_tolerance_frames: int,
) -> dict[str, float]:
  """Add deploy-style threshold sweep metrics for footstep and toe-hit events."""
  metrics: dict[str, float] = {}
  touchdown_f1: list[float] = []
  stair_touchdown_f1: list[float] = []
  toe_f1: list[float] = []
  for foot_id, side in enumerate(("left", "right")):
    touchdown_index = 2 + foot_id
    toe_hit_index = 4 + foot_id
    touchdown_stats = sweep_deployment_event_thresholds(
      labels=labels,
      probabilities=probabilities,
      episode_id=episode_id,
      frame_idx=frame_idx,
      event_label_index=touchdown_index,
      thresholds=thresholds,
      tolerance_frames=event_tolerance_frames,
      cooldown_frames=touchdown_cooldown_frames,
      contact_label_index=foot_id,
      contact_prob_index=foot_id,
      contact_threshold=touchdown_contact_threshold,
      contact_release_threshold=touchdown_contact_release_threshold,
    )
    for key, value in touchdown_stats.items():
      metrics[f"{side}_touchdown_deploy_{key}"] = float(value)
    touchdown_f1.append(float(touchdown_stats["best_event_f1"]))

    stair_stats = sweep_deployment_event_thresholds(
      labels=labels,
      probabilities=probabilities,
      episode_id=episode_id,
      frame_idx=frame_idx,
      event_label_index=touchdown_index,
      thresholds=thresholds,
      tolerance_frames=event_tolerance_frames,
      cooldown_frames=touchdown_cooldown_frames,
      contact_label_index=foot_id,
      contact_prob_index=foot_id,
      contact_threshold=touchdown_contact_threshold,
      contact_release_threshold=touchdown_contact_release_threshold,
      valid_mask=stair_support[:, foot_id].astype(np.bool_),
    )
    for key, value in stair_stats.items():
      metrics[f"{side}_touchdown_stair_deploy_{key}"] = float(value)
    stair_touchdown_f1.append(float(stair_stats["best_event_f1"]))

    toe_stats = sweep_deployment_event_thresholds(
      labels=labels,
      probabilities=probabilities,
      episode_id=episode_id,
      frame_idx=frame_idx,
      event_label_index=toe_hit_index,
      thresholds=thresholds,
      tolerance_frames=event_tolerance_frames,
      cooldown_frames=toe_hit_cooldown_frames,
    )
    for key, value in toe_stats.items():
      metrics[f"{side}_toe_riser_hit_deploy_{key}"] = float(value)
    toe_f1.append(float(toe_stats["best_event_f1"]))

  metrics["touchdown_deploy_macro_f1"] = float(np.mean(touchdown_f1))
  metrics["touchdown_stair_deploy_macro_f1"] = float(np.mean(stair_touchdown_f1))
  metrics["toe_riser_deploy_macro_f1"] = float(np.mean(toe_f1))
  metrics["deploy_event_macro_f1"] = float(np.mean(touchdown_f1 + toe_f1))
  return metrics


def _evaluate_online(
  model: FootEventDetectorGRU,
  buffer: OnlineFootEventReplayBuffer,
  *,
  device: torch.device,
  batch_size: int,
  pos_weight: torch.Tensor,
  threshold: float,
  event_tolerance_frames: int,
  deployment_thresholds: np.ndarray,
  touchdown_contact_threshold: float,
  touchdown_contact_release_threshold: float,
  touchdown_cooldown_frames: int,
  toe_hit_cooldown_frames: int,
) -> dict[str, float]:
  snapshot = buffer.snapshot()
  labels = snapshot["labels"].astype(np.float32)
  probabilities: list[np.ndarray] = []
  total_loss = 0.0
  total_count = 0
  model.eval()
  with torch.no_grad():
    for start in range(0, labels.shape[0], batch_size):
      end = min(start + batch_size, labels.shape[0])
      obs = torch.as_tensor(
        snapshot["obs_history"][start:end],
        dtype=torch.float32,
        device=device,
      )
      label = torch.as_tensor(labels[start:end], dtype=torch.float32, device=device)
      logits = model(obs)
      loss = F.binary_cross_entropy_with_logits(
        logits,
        label,
        pos_weight=pos_weight,
      )
      count = int(end - start)
      total_loss += float(loss.item()) * count
      total_count += count
      probabilities.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))
  probability_array = np.concatenate(probabilities, axis=0)
  val_loss = total_loss / max(float(total_count), 1.0)
  metrics = compute_metrics(
    labels=labels,
    probabilities=probability_array,
    episode_id=snapshot["episode_id"].astype(np.int64),
    frame_idx=snapshot["frame_idx"].astype(np.int64),
    val_loss=val_loss,
    threshold=threshold,
    event_tolerance_frames=event_tolerance_frames,
  )
  metrics.update(
    _touchdown_group_metrics(
      labels=labels,
      probabilities=probability_array,
      episode_id=snapshot["episode_id"].astype(np.int64),
      frame_idx=snapshot["frame_idx"].astype(np.int64),
      stair_support=snapshot["stair_support"].astype(np.bool_),
      support_fraction=snapshot["support_fraction"].astype(np.float32),
      threshold=threshold,
      event_tolerance_frames=event_tolerance_frames,
    )
  )
  metrics.update(
    _deployment_event_metrics(
      labels=labels,
      probabilities=probability_array,
      episode_id=snapshot["episode_id"].astype(np.int64),
      frame_idx=snapshot["frame_idx"].astype(np.int64),
      stair_support=snapshot["stair_support"].astype(np.bool_),
      thresholds=deployment_thresholds,
      touchdown_contact_threshold=touchdown_contact_threshold,
      touchdown_contact_release_threshold=touchdown_contact_release_threshold,
      touchdown_cooldown_frames=touchdown_cooldown_frames,
      toe_hit_cooldown_frames=toe_hit_cooldown_frames,
      event_tolerance_frames=event_tolerance_frames,
    )
  )
  return metrics


def _predict_buffer_probabilities(
  model: FootEventDetectorGRU,
  buffer: OnlineFootEventReplayBuffer,
  *,
  device: torch.device,
  batch_size: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
  snapshot = buffer.snapshot()
  probabilities: list[np.ndarray] = []
  model.eval()
  with torch.no_grad():
    for start in range(0, snapshot["labels"].shape[0], batch_size):
      end = min(start + batch_size, snapshot["labels"].shape[0])
      obs = torch.as_tensor(
        snapshot["obs_history"][start:end],
        dtype=torch.float32,
        device=device,
      )
      logits = model(obs)
      probabilities.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))
  return snapshot, np.concatenate(probabilities, axis=0)


def _mine_false_positive_hard_negatives(
  model: FootEventDetectorGRU,
  buffer: OnlineFootEventReplayBuffer,
  *,
  device: torch.device,
  batch_size: int,
  threshold: float,
  window_frames: int,
  max_peaks: int,
) -> int:
  if buffer.size == 0:
    return 0
  snapshot, probabilities = _predict_buffer_probabilities(
    model,
    buffer,
    device=device,
    batch_size=batch_size,
  )
  labels = snapshot["labels"].astype(np.float32)
  toe_fp = (probabilities[:, 4:6].max(axis=1) >= threshold) & (
    labels[:, 4:6].max(axis=1) <= 0.5
  )
  peak_indices = np.nonzero(toe_fp)[0].astype(np.int64)
  if peak_indices.shape[0] > max_peaks:
    order = np.argsort(-probabilities[peak_indices, 4:6].max(axis=1))
    peak_indices = peak_indices[order[:max_peaks]]
  hard_mask = np.zeros(labels.shape[0], dtype=np.bool_)
  episode_id = snapshot["episode_id"].astype(np.int64)
  frame_idx = snapshot["frame_idx"].astype(np.int64)
  env_id = snapshot["env_id"].astype(np.int64)
  for peak in peak_indices:
    same_sequence = (episode_id == episode_id[peak]) & (env_id == env_id[peak])
    near_peak = np.abs(frame_idx - frame_idx[peak]) <= window_frames
    hard_mask |= same_sequence & near_peak & (labels[:, 4:6].max(axis=1) <= 0.5)
  return buffer.mark_false_positive_hard_negatives(hard_mask)


def _metric_improved(
  *,
  metric_value: float,
  best_value: float,
  higher_better: bool,
  min_delta: float,
) -> bool:
  if higher_better:
    return metric_value > best_value + min_delta
  return metric_value < best_value - min_delta


def _current_metadata(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
  stair_support = _tensor_extra(
    env,
    STAIR_CURRENT_STAIR_SUPPORT_KEY,
    (2,),
    torch.bool,
    False,
  ).bool()
  support_fraction = _tensor_extra(
    env,
    STAIR_CURRENT_SUPPORT_FRACTION_KEY,
    (2,),
    torch.float32,
    0.0,
  ).float()
  return stair_support, support_fraction


def _write_label_audit(
  output_dir: Path,
  train_buffer: OnlineFootEventReplayBuffer,
  val_buffer: OnlineFootEventReplayBuffer,
) -> None:
  with (output_dir / "label_audit.csv").open(
    "w", encoding="utf-8", newline=""
  ) as stream:
    writer = csv.writer(stream)
    writer.writerow(("metric", "value"))
    writer.writerows(train_buffer.label_audit_rows("train"))
    writer.writerows(val_buffer.label_audit_rows("val"))


def run_online_train(
  task_id: str,
  cfg: OnlineFootEventDetectorConfig,
) -> dict[str, Any]:
  """Train the detector while collecting data from a frozen-policy rollout."""
  if cfg.num_envs <= 1:
    raise ValueError("num_envs must be greater than one for held-out validation.")
  if cfg.steps <= 0:
    raise ValueError("steps must be positive.")
  if cfg.history_len <= 0:
    raise ValueError("history_len must be positive.")
  expected_obs_dim = foot_event_detector_obs_dim(
    include_gait_phase=cfg.include_gait_phase
  )
  if cfg.expected_obs_dim != expected_obs_dim:
    raise ValueError(
      f"expected_obs_dim={cfg.expected_obs_dim} does not match detector obs "
      f"dimension {expected_obs_dim}."
    )
  if not 0.0 < cfg.val_env_fraction < 1.0:
    raise ValueError("val_env_fraction must be in (0, 1).")
  if cfg.steps <= 0:
    raise ValueError("steps must be positive.")
  if cfg.max_updates is not None and cfg.max_updates <= 0:
    raise ValueError("max_updates must be positive when set.")
  if cfg.early_stop_patience_evals < 0:
    raise ValueError("early_stop_patience_evals must be non-negative.")
  if cfg.early_stop_min_delta < 0.0:
    raise ValueError("early_stop_min_delta must be non-negative.")
  sample_fractions = (
    cfg.toe_positive_fraction,
    cfg.touchdown_positive_fraction,
    cfg.stair_hard_negative_fraction,
    cfg.false_positive_hard_negative_fraction,
  )
  if any(value < 0.0 for value in sample_fractions) or sum(sample_fractions) > 1.0:
    raise ValueError("Stratified replay fractions must be non-negative and sum <= 1.")
  if cfg.soft_touchdown_radius < 0 or cfg.soft_toe_hit_radius < 0:
    raise ValueError("soft event radii must be non-negative.")
  if cfg.hard_negative_mining_window_frames < 0:
    raise ValueError("hard_negative_mining_window_frames must be non-negative.")
  if cfg.hard_negative_mining_max_peaks <= 0:
    raise ValueError("hard_negative_mining_max_peaks must be positive.")
  if cfg.touchdown_cooldown_frames < 0 or cfg.toe_hit_cooldown_frames < 0:
    raise ValueError("event cooldown frames must be non-negative.")
  if cfg.sweep_threshold_steps <= 0:
    raise ValueError("sweep_threshold_steps must be positive.")
  if not 0.0 < cfg.sweep_threshold_min <= cfg.sweep_threshold_max < 1.0:
    raise ValueError("sweep thresholds must satisfy 0 < min <= max < 1.")

  random.seed(cfg.seed)
  np.random.seed(cfg.seed)
  torch.manual_seed(cfg.seed)
  configure_torch_backends()
  device = torch.device(
    cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  )
  output_dir = Path(cfg.output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  env_cfg = load_env_cfg(task_id, play=False)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  checkpoint_path = resolve_checkpoint_path(
    task_id=task_id,
    agent_cfg=agent_cfg,
    checkpoint_file=cfg.checkpoint_file,
    wandb_run_path=cfg.wandb_run_path,
    wandb_checkpoint_name=cfg.wandb_checkpoint_name,
  )
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=str(device), render_mode=None)
  wrapped = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  history_buffer = StairProbeHistoryBuffer(
    num_envs=cfg.num_envs,
    history_len=cfg.history_len,
    obs_dim=cfg.expected_obs_dim,
    device=device,
  )
  train_buffer = OnlineFootEventReplayBuffer(
    capacity=cfg.train_buffer_capacity,
    history_len=cfg.history_len,
    obs_dim=cfg.expected_obs_dim,
    label_dim=len(FOOT_EVENT_LABEL_NAMES),
    num_envs=cfg.num_envs,
    soft_touchdown_radius=cfg.soft_touchdown_radius,
    soft_toe_hit_radius=cfg.soft_toe_hit_radius,
    soft_event_radius1_value=cfg.soft_event_radius1_value,
    soft_event_radius2_value=cfg.soft_event_radius2_value,
    device=device,
  )
  val_buffer = OnlineFootEventReplayBuffer(
    capacity=cfg.val_buffer_capacity,
    history_len=cfg.history_len,
    obs_dim=cfg.expected_obs_dim,
    label_dim=len(FOOT_EVENT_LABEL_NAMES),
    num_envs=cfg.num_envs,
    soft_touchdown_radius=cfg.soft_touchdown_radius,
    soft_toe_hit_radius=cfg.soft_toe_hit_radius,
    soft_event_radius1_value=cfg.soft_event_radius1_value,
    soft_event_radius2_value=cfg.soft_event_radius2_value,
    device=device,
  )
  model = FootEventDetectorGRU(
    obs_dim=cfg.expected_obs_dim,
    frame_hidden_dim=cfg.frame_hidden_dim,
    recurrent_hidden_dim=cfg.recurrent_hidden_dim,
    head_hidden_dim=cfg.head_hidden_dim,
    dropout=cfg.dropout,
  ).to(device)
  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.learning_rate,
    weight_decay=cfg.weight_decay,
  )
  generator = torch.Generator(device=device)
  generator.manual_seed(cfg.seed)
  previous_contact = torch.zeros(cfg.num_envs, 2, dtype=torch.bool, device=device)
  previous_contact_valid = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
  recent_event_age = torch.full(
    (cfg.num_envs, len(FOOT_EVENT_LABEL_NAMES)),
    10_000,
    dtype=torch.int64,
    device=device,
  )
  episode_counter = torch.zeros(cfg.num_envs, dtype=torch.int64, device=device)
  env_ids = torch.arange(cfg.num_envs, dtype=torch.int64, device=device)
  val_env_count = max(
    1, min(cfg.num_envs - 1, int(round(cfg.num_envs * cfg.val_env_fraction)))
  )
  is_val_env = env_ids < val_env_count
  selection_metric = (
    "deploy_event_macro_f1" if cfg.selection_metric == "auto" else cfg.selection_metric
  )
  deployment_thresholds = np.linspace(
    cfg.sweep_threshold_min,
    cfg.sweep_threshold_max,
    cfg.sweep_threshold_steps,
    dtype=np.float32,
  )
  higher_better = metric_is_higher_better(selection_metric)
  best_value = -float("inf") if higher_better else float("inf")
  best_step = 0
  best_state: dict[str, torch.Tensor] | None = None
  metrics_history: list[dict[str, Any]] = []
  train_updates = 0
  completed_steps = 0
  no_improve_evals = 0
  stop_reason = "completed_steps"
  pos_weight = torch.ones(
    len(FOOT_EVENT_LABEL_NAMES), dtype=torch.float32, device=device
  )

  print(
    "[Stage 2D] Online train detector:",
    f"task={task_id}",
    f"checkpoint={checkpoint_path}",
    f"envs={cfg.num_envs}",
    f"val_envs={val_env_count}",
    f"steps={cfg.steps}",
    f"max_updates={cfg.max_updates}",
    f"patience_evals={cfg.early_stop_patience_evals}",
    f"selection_metric={selection_metric}",
    f"toe_hit_pos_weight={cfg.toe_hit_pos_weight}",
    f"include_gait_phase={cfg.include_gait_phase}",
    f"obs_dim={cfg.expected_obs_dim}",
    f"toe_pos_fraction={cfg.toe_positive_fraction}",
    f"touchdown_pos_fraction={cfg.touchdown_positive_fraction}",
    f"history_len={cfg.history_len}",
    f"device={device}",
    f"output={output_dir}",
  )
  try:
    policy, _runner = load_inference_policy(
      env=wrapped,
      task_id=task_id,
      agent_cfg=agent_cfg,
      checkpoint_path=checkpoint_path,
      device=str(device),
    )
    obs = wrapped.get_observations()
    latent = foot_event_detector_obs(
      obs,
      raw_env,
      include_gait_phase=cfg.include_gait_phase,
      gait_period=cfg.gait_period,
      command_name=cfg.command_name,
    )
    history_buffer.push(
      latent, torch.ones(cfg.num_envs, dtype=torch.bool, device=device)
    )

    progress = tqdm(
      range(cfg.steps),
      desc=f"online foot event seed={cfg.seed}",
      disable=not cfg.progress,
      dynamic_ncols=True,
      unit="step",
    )
    for rollout_step in progress:
      completed_steps = rollout_step + 1
      with torch.no_grad():
        actions = policy(obs)
      step_result = wrapped.step(actions)
      reset_policy_state_from_step(policy, step_result)
      obs, _rewards, dones, _extras = step_result
      reset_mask = dones.to(dtype=torch.bool)
      latent = foot_event_detector_obs(
        obs,
        raw_env,
        include_gait_phase=cfg.include_gait_phase,
        gait_period=cfg.gait_period,
        command_name=cfg.command_name,
      )
      history_buffer.push(latent, reset_mask)

      labels = foot_event_labels_from_env(
        raw_env,
        previous_contact=previous_contact,
        previous_contact_valid=previous_contact_valid,
      )
      train_labels = _soften_labels_from_recent_events(
        labels,
        recent_event_age,
        reset_mask,
        touchdown_radius=cfg.soft_touchdown_radius,
        toe_hit_radius=cfg.soft_toe_hit_radius,
        radius1_value=cfg.soft_event_radius1_value,
        radius2_value=cfg.soft_event_radius2_value,
      )
      stair_support, support_fraction = _current_metadata(raw_env)
      full_history = history_buffer.valid_mask.all(dim=1)
      collect_mask = full_history & ~reset_mask
      episode_id = episode_counter * cfg.num_envs + env_ids
      frame_tensor = torch.full(
        (cfg.num_envs,),
        rollout_step + 1,
        dtype=torch.int64,
        device=device,
      )

      for buffer, mask in (
        (train_buffer, collect_mask & ~is_val_env),
        (val_buffer, collect_mask & is_val_env),
      ):
        ids = mask.nonzero(as_tuple=False).squeeze(-1)
        buffer.add(
          obs_history=history_buffer.history[ids],
          labels=labels[ids],
          train_labels=train_labels[ids],
          episode_id=episode_id[ids],
          frame_idx=frame_tensor[ids],
          env_id=ids,
          stair_support=stair_support[ids],
          support_fraction=support_fraction[ids],
        )

      current_contact = _tensor_extra(
        raw_env,
        STAIR_CURRENT_GROUND_CONTACT_KEY,
        (2,),
        torch.bool,
        False,
      ).bool()
      previous_contact.copy_(
        torch.where(
          reset_mask[:, None],
          torch.zeros_like(current_contact),
          current_contact,
        )
      )
      previous_contact_valid.copy_(
        torch.where(
          reset_mask,
          torch.zeros_like(previous_contact_valid),
          torch.ones_like(previous_contact_valid),
        )
      )
      episode_counter += reset_mask.to(dtype=torch.int64)

      should_train = (
        train_buffer.size >= cfg.warmup_samples
        and (rollout_step + 1) % cfg.train_every_steps == 0
      )
      if should_train:
        train_labels = train_buffer.labels[: train_buffer.size].detach().cpu().numpy()
        pos_weight = make_pos_weight(
          train_labels,
          max_weight=cfg.pos_weight_max,
          toe_hit_pos_weight=cfg.toe_hit_pos_weight,
        ).to(device)
        model.train()
        for _ in range(cfg.updates_per_train):
          if cfg.max_updates is not None and train_updates >= cfg.max_updates:
            break
          batch_obs, batch_labels = train_buffer.sample(
            cfg.batch_size,
            generator=generator,
            toe_positive_fraction=cfg.toe_positive_fraction,
            touchdown_positive_fraction=cfg.touchdown_positive_fraction,
            stair_hard_negative_fraction=cfg.stair_hard_negative_fraction,
            false_positive_hard_negative_fraction=(
              cfg.false_positive_hard_negative_fraction
            ),
          )
          logits = model(batch_obs)
          loss = _training_loss(
            logits=logits,
            labels=batch_labels,
            pos_weight=pos_weight,
            focal_loss_weight=cfg.focal_loss_weight,
            tversky_loss_weight=cfg.tversky_loss_weight,
            focal_gamma_pos=cfg.focal_gamma_pos,
            focal_gamma_neg=cfg.focal_gamma_neg,
            tversky_alpha=cfg.tversky_alpha,
            tversky_beta=cfg.tversky_beta,
          )
          optimizer.zero_grad(set_to_none=True)
          loss.backward()
          optimizer.step()
          train_updates += 1
      reached_max_updates = (
        cfg.max_updates is not None and train_updates >= cfg.max_updates
      )

      should_eval = (
        val_buffer.size >= cfg.min_val_samples
        and (rollout_step + 1) % cfg.eval_interval_steps == 0
      )
      if should_eval:
        val_metrics = _evaluate_online(
          model,
          val_buffer,
          device=device,
          batch_size=cfg.batch_size,
          pos_weight=pos_weight,
          threshold=cfg.threshold,
          event_tolerance_frames=cfg.event_tolerance_frames,
          deployment_thresholds=deployment_thresholds,
          touchdown_contact_threshold=cfg.touchdown_contact_threshold,
          touchdown_contact_release_threshold=cfg.touchdown_contact_release_threshold,
          touchdown_cooldown_frames=cfg.touchdown_cooldown_frames,
          toe_hit_cooldown_frames=cfg.toe_hit_cooldown_frames,
        )
        mined_hard_negatives = 0
        if cfg.mine_false_positive_hard_negatives:
          mined_hard_negatives = _mine_false_positive_hard_negatives(
            model,
            train_buffer,
            device=device,
            batch_size=cfg.batch_size,
            threshold=cfg.hard_negative_mining_threshold,
            window_frames=cfg.hard_negative_mining_window_frames,
            max_peaks=cfg.hard_negative_mining_max_peaks,
          )
        metric_value = float(val_metrics[selection_metric])
        is_better = _metric_improved(
          metric_value=metric_value,
          best_value=best_value,
          higher_better=higher_better,
          min_delta=cfg.early_stop_min_delta,
        )
        if is_better:
          best_value = metric_value
          best_step = rollout_step + 1
          best_state = copy.deepcopy(model.state_dict())
          torch.save(best_state, output_dir / "best.pt")
          no_improve_evals = 0
        else:
          no_improve_evals += 1
        metrics_history.append(
          {
            "step": rollout_step + 1,
            "train_updates": train_updates,
            "train_samples": train_buffer.size,
            "val_samples": val_buffer.size,
            "mined_false_positive_hard_negatives": mined_hard_negatives,
            "val": val_metrics,
          }
        )
        print(
          f"[Stage 2D] step={rollout_step + 1}",
          f"updates={train_updates}",
          f"val_loss={val_metrics['val_loss']:.5f}",
          f"event_macro_f1={val_metrics['event_macro_f1']:.4f}",
          f"deploy_event_macro_f1={val_metrics['deploy_event_macro_f1']:.4f}",
          f"td_deploy_f1={val_metrics['touchdown_deploy_macro_f1']:.4f}",
          f"toe_deploy_f1={val_metrics['toe_riser_deploy_macro_f1']:.4f}",
          f"mined_fp_hard={mined_hard_negatives}",
          f"left_td_stair_f1={val_metrics.get('left_touchdown_stair_event_f1', 0.0):.4f}",
          f"right_td_stair_f1={val_metrics.get('right_touchdown_stair_event_f1', 0.0):.4f}",
          f"no_improve={no_improve_evals}",
        )
      if reached_max_updates:
        stop_reason = "max_updates"
        break
      if (
        cfg.early_stop_patience_evals > 0
        and no_improve_evals >= cfg.early_stop_patience_evals
      ):
        stop_reason = "early_stop"
        break
  finally:
    _close_sequence_logger(raw_env)
    wrapped.close()

  if best_state is None:
    best_state = copy.deepcopy(model.state_dict())
    best_step = completed_steps
    torch.save(best_state, output_dir / "best.pt")
    best_value = 0.0
  model.load_state_dict(best_state)
  if cfg.export_onnx:
    export_detector_onnx(
      model,
      output_dir / "best.onnx",
      history_len=cfg.history_len,
      obs_dim=cfg.expected_obs_dim,
    )
  _write_label_audit(output_dir, train_buffer, val_buffer)
  best_val_metrics: dict[str, float] = {}
  for record in metrics_history:
    if int(record["step"]) == best_step:
      best_val_metrics = {
        key: float(value) for key, value in dict(record["val"]).items()
      }
      break
  best_deployment_thresholds = {
    key: value
    for key, value in best_val_metrics.items()
    if key.endswith("_deploy_best_threshold")
  }
  payload: dict[str, Any] = {
    "config": asdict(cfg),
    "task_id": task_id,
    "checkpoint_path": str(checkpoint_path),
    "label_names": list(FOOT_EVENT_LABEL_NAMES),
    "input_feature_groups": foot_event_input_feature_groups(
      include_gait_phase=cfg.include_gait_phase
    ),
    "history": metrics_history,
    "best_step": best_step,
    "completed_steps": completed_steps,
    "stop_reason": stop_reason,
    "selection_metric": selection_metric,
    "best_metric_value": best_value,
    "best_deployment_thresholds": best_deployment_thresholds,
    "deployment_threshold_grid": [
      float(value) for value in deployment_thresholds.astype(np.float32).tolist()
    ],
    "train_updates": train_updates,
    "no_improve_evals": no_improve_evals,
    "train_samples_in_buffer": train_buffer.size,
    "val_samples_in_buffer": val_buffer.size,
    "pos_weight": [float(value) for value in pos_weight.detach().cpu().tolist()],
    "model": {
      "type": "FootEventDetectorGRU",
      "obs_dim": cfg.expected_obs_dim,
      "history_len": cfg.history_len,
      "output_dim": len(FOOT_EVENT_LABEL_NAMES),
      "frame_hidden_dim": cfg.frame_hidden_dim,
      "recurrent_hidden_dim": cfg.recurrent_hidden_dim,
      "head_hidden_dim": cfg.head_hidden_dim,
      "trainable_parameters": int(
        sum(parameter.numel() for parameter in model.parameters())
      ),
    },
    "onnx_path": str(output_dir / "best.onnx") if cfg.export_onnx else None,
  }
  with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
  print(
    "[Stage 2D] Online train complete:",
    f"best_step={best_step}",
    f"completed_steps={completed_steps}",
    f"stop_reason={stop_reason}",
    f"{selection_metric}={best_value:.5f}",
    f"updates={train_updates}",
    f"output={output_dir}",
  )
  return payload


def main() -> None:
  import mjlab.tasks as _tasks  # noqa: F401

  task_id, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    args=None,
    default="Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1",
    return_unknown_args=True,
  )
  cfg = tyro.cli(OnlineFootEventDetectorConfig, args=remaining_args)
  run_online_train(task_id, cfg)


if __name__ == "__main__":
  main()
