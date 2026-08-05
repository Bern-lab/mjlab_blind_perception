"""Online Stage 2D foot-event detector training with a frozen policy rollout."""

from __future__ import annotations

import copy
import csv
import json
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from scripts.velocity_eval.export_foot_event_detector_dataset import (
  DEFAULT_STAGE2D_CHECKPOINT,
  FOOT_EVENT_LABEL_DIAGNOSTIC_NAMES,
  FOOT_EVENT_LABEL_NAMES,
  FootEventDetectorObsSchema,
  foot_event_detector_obs,
  foot_event_input_feature_groups,
  foot_event_input_feature_scale_groups,
  foot_event_input_feature_scales,
  foot_event_label_diagnostics_from_env,
  foot_event_labels_from_env,
  resolve_foot_event_detector_obs_dim,
)
from scripts.velocity_eval.export_stair_probe_dataset import (
  STAIR_CURRENT_STAIR_SUPPORT_KEY,
  STAIR_CURRENT_SUPPORT_FRACTION_KEY,
  StairProbeHistoryBuffer,
  _close_sequence_logger,
  _tensor_extra,
)
from scripts.velocity_eval.policy_io import (
  get_clip_actions,
  load_inference_policy,
  resolve_checkpoint_path,
  resolve_inference_agent_cfg,
)
from scripts.velocity_eval.train_foot_event_detector import (
  FootEventDetectorGRU,
  _event_start_frames_and_indices,
  compute_metrics,
  event_f1_for_label,
  export_detector_onnx,
  make_pos_weight,
  metric_is_higher_better,
  sweep_deployment_event_thresholds,
)
from torch import nn
from tqdm.auto import tqdm

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.mdp.observations import _world_frame_foot_positions
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
  input_schema: FootEventDetectorObsSchema = "v1"
  include_gait_phase: bool = True
  gait_period: float = 0.6
  command_name: str = "twist"
  expected_obs_dim: int | None = None
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
  init_detector_checkpoint: str | None = None
  toe_only_finetune: bool = False
  toe_riser_only_model: bool = False
  footprint_only_model: bool = False
  frame_hidden_dim: int = 128
  recurrent_hidden_dim: int = 64
  head_hidden_dim: int = 32
  dropout: float = 0.0
  threshold: float = 0.5
  event_tolerance_frames: int = 2
  pos_weight_max: float = 100.0
  toe_hit_pos_weight: float | None = 150.0
  toe_positive_fraction: float = 0.125
  toe_soft_positive_fraction: float = 0.05
  touchdown_positive_fraction: float = 0.20
  touchdown_soft_positive_fraction: float = 0.125
  stair_hard_negative_fraction: float = 0.20
  false_positive_hard_negative_fraction: float = 0.075
  false_negative_hard_positive_fraction: float = 0.125
  false_negative_toe_hard_positive_fraction: float = 0.05
  toe_soft_positive_threshold: float = 0.3
  touchdown_soft_positive_threshold: float = 0.3
  soft_touchdown_radius: int = 1
  soft_toe_hit_radius: int = 2
  soft_event_radius1_value: float = 0.7
  soft_event_radius2_value: float = 0.4
  focal_loss_weight: float = 0.5
  tversky_loss_weight: float = 0.5
  focal_gamma_pos: float = 0.0
  focal_gamma_neg: float = 4.0
  tversky_alpha: float = 0.3
  tversky_beta: float = 0.7
  mine_false_positive_hard_negatives: bool = True
  mine_false_positive_touchdown_hard_negatives: bool = True
  mine_false_negative_hard_positives: bool = True
  mine_false_negative_toe_hard_positives: bool = True
  hard_negative_mining_threshold: float = 0.5
  hard_touchdown_false_positive_mining_threshold: float = 0.5
  hard_touchdown_false_positive_mining_contact_threshold: float = 0.7
  hard_touchdown_false_positive_mining_window_frames: int = 4
  hard_touchdown_false_positive_mining_max_peaks: int = 4096
  hard_positive_mining_threshold: float = 0.5
  hard_toe_positive_mining_threshold: float = 0.5
  hard_negative_mining_window_frames: int = 8
  hard_negative_mining_max_peaks: int = 4096
  hard_positive_mining_window_frames: int = 3
  hard_positive_mining_max_peaks: int = 4096
  hard_toe_positive_mining_window_frames: int = 4
  hard_toe_positive_mining_max_peaks: int = 4096
  touchdown_contact_threshold: float = 0.7
  touchdown_contact_release_threshold: float = 0.35
  touchdown_cooldown_frames: int = 8
  toe_hit_cooldown_frames: int = 8
  touchdown_recall_precision_floor: float = 0.85
  toe_hit_recall_precision_floor: float = 0.35
  selection_min_stair_touchdown_events: int = 100
  sweep_threshold_min: float = 0.1
  sweep_threshold_max: float = 0.99
  sweep_threshold_steps: int = 19
  selection_metric: str = "footprint_deploy_score"
  baseline_metrics_file: str | None = None
  baseline_touchdown_recall_tolerance: float = 0.01
  baseline_stair_touchdown_recall_tolerance: float = 0.015
  baseline_flat_touchdown_f1_tolerance: float = 0.01
  baseline_toe_metric: str = "toe_riser_high_recall_macro_f1"
  baseline_toe_metric_min_improvement: float = 0.0
  baseline_required_metric_names: tuple[str, ...] = ()
  baseline_required_metric_min_improvement: float = 0.0
  require_baseline_guard: bool = False
  max_updates: int | None = 5000
  early_stop_patience_evals: int = 8
  early_stop_min_delta: float = 1.0e-4
  save_eval_checkpoints: bool = True
  save_last_checkpoint: bool = True
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
    stair_hard_negative_label_indices: tuple[int, ...] = (2, 3, 4, 5),
    label_diagnostic_names: tuple[str, ...] = (),
  ) -> None:
    if capacity <= 0:
      raise ValueError("capacity must be positive.")
    if not stair_hard_negative_label_indices:
      raise ValueError("stair_hard_negative_label_indices must not be empty.")
    if any(
      index < 0 or index >= label_dim for index in stair_hard_negative_label_indices
    ):
      raise ValueError("stair_hard_negative_label_indices are outside label_dim.")
    self.capacity = int(capacity)
    self.device = device
    self.stair_hard_negative_label_indices = tuple(
      int(index) for index in stair_hard_negative_label_indices
    )
    self.label_diagnostic_names = tuple(label_diagnostic_names)
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
    self.footprint_anchor_w = torch.empty(
      capacity,
      2,
      3,
      dtype=torch.float32,
      device=device,
    )
    self.stair_support = torch.empty(capacity, 2, dtype=torch.bool, device=device)
    self.support_fraction = torch.empty(
      capacity,
      2,
      dtype=torch.float32,
      device=device,
    )
    self.label_diagnostics = torch.empty(
      capacity,
      len(self.label_diagnostic_names),
      dtype=torch.float32,
      device=device,
    )
    self.stair_hard_negative = torch.empty(capacity, dtype=torch.bool, device=device)
    self.false_positive_hard_negative = torch.empty(
      capacity,
      dtype=torch.bool,
      device=device,
    )
    self.false_negative_hard_positive = torch.empty(
      capacity,
      dtype=torch.bool,
      device=device,
    )
    self.false_negative_toe_hard_positive = torch.empty(
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
    footprint_anchor_w: torch.Tensor | None = None,
    label_diagnostics: torch.Tensor | None = None,
  ) -> torch.Tensor:
    n = int(obs_history.shape[0])
    if n == 0:
      return torch.empty(0, dtype=torch.int64, device=self.device)
    if footprint_anchor_w is None:
      footprint_anchor_w = torch.zeros(n, 2, 3, dtype=torch.float32, device=self.device)
    if label_diagnostics is None:
      label_diagnostics = torch.zeros(
        n,
        len(self.label_diagnostic_names),
        dtype=torch.float32,
        device=self.device,
      )
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
      footprint_anchor_w = footprint_anchor_w[start:]
      label_diagnostics = label_diagnostics[start:]
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
      footprint_anchor_w[:first],
      label_diagnostics[:first],
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
        footprint_anchor_w[first:],
        label_diagnostics[first:],
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
    footprint_anchor_w: torch.Tensor,
    label_diagnostics: torch.Tensor,
  ) -> None:
    self.obs_history[slc].copy_(obs_history.detach())
    self.labels[slc].copy_(labels.detach())
    self.train_labels[slc].copy_(train_labels.detach())
    self.episode_id[slc].copy_(episode_id.detach())
    self.frame_idx[slc].copy_(frame_idx.detach())
    self.env_id[slc].copy_(env_id.detach())
    self.footprint_anchor_w[slc].copy_(footprint_anchor_w.detach())
    self.stair_support[slc].copy_(stair_support.detach())
    self.support_fraction[slc].copy_(support_fraction.detach())
    self.label_diagnostics[slc].copy_(label_diagnostics.detach())
    event_negative = (
      labels[:, list(self.stair_hard_negative_label_indices)].amax(dim=1) <= 0.5
    )
    self.stair_hard_negative[slc].copy_(stair_support.any(dim=1) & event_negative)
    self.false_positive_hard_negative[slc] = False
    self.false_negative_hard_positive[slc] = False
    self.false_negative_toe_hard_positive[slc] = False

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
    toe_soft_positive_fraction: float,
    touchdown_positive_fraction: float,
    touchdown_soft_positive_fraction: float,
    stair_hard_negative_fraction: float,
    false_positive_hard_negative_fraction: float,
    false_negative_hard_positive_fraction: float,
    false_negative_toe_hard_positive_fraction: float,
    toe_soft_positive_threshold: float,
    touchdown_soft_positive_threshold: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if self._size <= 0:
      raise RuntimeError("Cannot sample from an empty buffer.")
    toe_count = int(round(batch_size * toe_positive_fraction))
    toe_soft_count = int(round(batch_size * toe_soft_positive_fraction))
    touchdown_count = int(round(batch_size * touchdown_positive_fraction))
    touchdown_soft_count = int(round(batch_size * touchdown_soft_positive_fraction))
    stair_hard_count = int(round(batch_size * stair_hard_negative_fraction))
    false_positive_count = int(
      round(batch_size * false_positive_hard_negative_fraction)
    )
    false_negative_count = int(
      round(batch_size * false_negative_hard_positive_fraction)
    )
    false_negative_toe_count = int(
      round(batch_size * false_negative_toe_hard_positive_fraction)
    )
    reserved = (
      toe_count
      + toe_soft_count
      + touchdown_count
      + touchdown_soft_count
      + stair_hard_count
      + false_positive_count
      + false_negative_count
      + false_negative_toe_count
    )
    if reserved > batch_size:
      scale = batch_size / max(float(reserved), 1.0)
      toe_count = int(round(toe_count * scale))
      toe_soft_count = int(round(toe_soft_count * scale))
      touchdown_count = int(round(touchdown_count * scale))
      touchdown_soft_count = int(round(touchdown_soft_count * scale))
      stair_hard_count = int(round(stair_hard_count * scale))
      false_positive_count = int(round(false_positive_count * scale))
      false_negative_count = int(round(false_negative_count * scale))
      false_negative_toe_count = int(round(false_negative_toe_count * scale))
      reserved = (
        toe_count
        + toe_soft_count
        + touchdown_count
        + touchdown_soft_count
        + stair_hard_count
        + false_positive_count
        + false_negative_count
        + false_negative_toe_count
      )
    random_count = batch_size - reserved
    hard_touchdown = self.labels[: self._size, 2:4].amax(dim=1) > 0.5
    soft_touchdown = (
      self.train_labels[: self._size, 2:4].amax(dim=1)
      >= float(touchdown_soft_positive_threshold)
    ) & ~hard_touchdown
    hard_toe = self.labels[: self._size, 4:6].amax(dim=1) > 0.5
    soft_toe = (
      self.train_labels[: self._size, 4:6].amax(dim=1)
      >= float(toe_soft_positive_threshold)
    ) & ~hard_toe
    chunks = [
      self._sample_mask(
        hard_touchdown,
        touchdown_count,
        generator,
      ),
      self._sample_mask(
        soft_touchdown,
        touchdown_soft_count,
        generator,
      ),
      self._sample_mask(
        hard_toe,
        toe_count,
        generator,
      ),
      self._sample_mask(
        soft_toe,
        toe_soft_count,
        generator,
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
      self._sample_mask(
        self.false_negative_hard_positive[: self._size],
        false_negative_count,
        generator,
      ),
      self._sample_mask(
        self.false_negative_toe_hard_positive[: self._size],
        false_negative_toe_count,
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
      "footprint_anchor_w": self.footprint_anchor_w[:size].detach().cpu().numpy(),
      "stair_support": self.stair_support[:size].detach().cpu().numpy(),
      "support_fraction": self.support_fraction[:size].detach().cpu().numpy(),
      "label_diagnostics": self.label_diagnostics[:size].detach().cpu().numpy(),
      "stair_hard_negative": self.stair_hard_negative[:size].detach().cpu().numpy(),
      "false_positive_hard_negative": (
        self.false_positive_hard_negative[:size].detach().cpu().numpy()
      ),
      "false_negative_hard_positive": (
        self.false_negative_hard_positive[:size].detach().cpu().numpy()
      ),
      "false_negative_toe_hard_positive": (
        self.false_negative_toe_hard_positive[:size].detach().cpu().numpy()
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

  def mark_false_negative_hard_positives(self, mask: np.ndarray) -> int:
    """Mark missed touchdown windows that should be replayed as hard positives."""
    if mask.shape != (self._size,):
      raise ValueError(
        f"Expected false-negative mask shape {(self._size,)}, got {mask.shape}."
      )
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
    before = int(self.false_negative_hard_positive[: self._size].sum().item())
    self.false_negative_hard_positive[: self._size] |= mask_tensor
    after = int(self.false_negative_hard_positive[: self._size].sum().item())
    return after - before

  def mark_false_negative_toe_hard_positives(self, mask: np.ndarray) -> int:
    """Mark missed toe-hit windows that should be replayed as hard positives."""
    if mask.shape != (self._size,):
      raise ValueError(
        f"Expected false-negative toe mask shape {(self._size,)}, got {mask.shape}."
      )
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
    before = int(self.false_negative_toe_hard_positive[: self._size].sum().item())
    self.false_negative_toe_hard_positive[: self._size] |= mask_tensor
    after = int(self.false_negative_toe_hard_positive[: self._size].sum().item())
    return after - before

  def label_audit_rows(self, prefix: str) -> list[tuple[str, str]]:
    labels = self.labels[: self._size]
    rows = [(f"{prefix}_samples", str(self._size))]
    rows.append(
      (
        f"{prefix}_stair_hard_negative_label_indices",
        ",".join(str(index) for index in self.stair_hard_negative_label_indices),
      )
    )
    for index, name in enumerate(FOOT_EVENT_LABEL_NAMES):
      count = float(labels[:, index].sum().item()) if self._size else 0.0
      rows.append((f"{prefix}_{name}_positive_count", f"{count:.0f}"))
      rows.append(
        (
          f"{prefix}_{name}_positive_rate",
          f"{count / max(float(self._size), 1.0):.6g}",
        )
      )
    label_diagnostics = self.label_diagnostics[: self._size]
    for index, name in enumerate(self.label_diagnostic_names):
      count = float(label_diagnostics[:, index].sum().item()) if self._size else 0.0
      rows.append((f"{prefix}_{name}_count", f"{count:.0f}"))
      rows.append(
        (
          f"{prefix}_{name}_rate",
          f"{count / max(float(self._size), 1.0):.6g}",
        )
      )
    stair_hard = int(self.stair_hard_negative[: self._size].sum().item())
    false_positive_hard = int(
      self.false_positive_hard_negative[: self._size].sum().item()
    )
    false_negative_hard = int(
      self.false_negative_hard_positive[: self._size].sum().item()
    )
    false_negative_toe_hard = int(
      self.false_negative_toe_hard_positive[: self._size].sum().item()
    )
    soft_touchdown = int(
      (
        (self.train_labels[: self._size, 2:4].amax(dim=1) > 0.0)
        & (labels[:, 2:4].amax(dim=1) <= 0.5)
      )
      .sum()
      .item()
    )
    soft_toe = int(
      (
        (self.train_labels[: self._size, 4:6].amax(dim=1) > 0.0)
        & (labels[:, 4:6].amax(dim=1) <= 0.5)
      )
      .sum()
      .item()
    )
    rows.append((f"{prefix}_stair_hard_negative_count", str(stair_hard)))
    rows.append(
      (f"{prefix}_false_positive_hard_negative_count", str(false_positive_hard))
    )
    rows.append(
      (f"{prefix}_false_negative_hard_positive_count", str(false_negative_hard))
    )
    rows.append(
      (
        f"{prefix}_false_negative_toe_hard_positive_count",
        str(false_negative_toe_hard),
      )
    )
    rows.append((f"{prefix}_soft_touchdown_neighbor_count", str(soft_touchdown)))
    rows.append((f"{prefix}_soft_toe_neighbor_count", str(soft_toe)))
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


def _detector_output_layer(model: FootEventDetectorGRU) -> nn.Linear:
  output_layer = model.head[-1]
  if not isinstance(output_layer, nn.Linear):
    raise TypeError(
      "Expected FootEventDetectorGRU.head[-1] to be nn.Linear for toe-only "
      f"finetune, got {type(output_layer).__name__}."
    )
  if output_layer.out_features < 6:
    raise ValueError("Toe-only finetune requires detector outputs for indices 4 and 5.")
  return output_layer


def _configure_toe_only_finetune(model: FootEventDetectorGRU) -> None:
  """Freeze the detector except the final left/right toe-riser output rows."""
  for parameter in model.parameters():
    parameter.requires_grad_(False)

  output_layer = _detector_output_layer(model)
  output_layer.weight.requires_grad_(True)
  weight_mask = torch.zeros_like(output_layer.weight)
  weight_mask[4:6] = 1.0
  output_layer.weight.register_hook(
    lambda grad: grad * weight_mask.to(device=grad.device, dtype=grad.dtype)
  )

  if output_layer.bias is not None:
    output_layer.bias.requires_grad_(True)
    bias_mask = torch.zeros_like(output_layer.bias)
    bias_mask[4:6] = 1.0
    output_layer.bias.register_hook(
      lambda grad: grad * bias_mask.to(device=grad.device, dtype=grad.dtype)
    )


def _configure_toe_riser_only_model(
  model: FootEventDetectorGRU,
  *,
  dummy_logit: float = -20.0,
) -> None:
  """Train a toe-riser expert while keeping the standard 6-logit output layout."""
  for parameter in model.parameters():
    parameter.requires_grad_(True)

  output_layer = _detector_output_layer(model)
  with torch.no_grad():
    output_layer.weight[:4].zero_()
    if output_layer.bias is not None:
      output_layer.bias[:4].fill_(float(dummy_logit))

  weight_mask = torch.ones_like(output_layer.weight)
  weight_mask[:4] = 0.0
  output_layer.weight.register_hook(
    lambda grad: grad * weight_mask.to(device=grad.device, dtype=grad.dtype)
  )

  if output_layer.bias is not None:
    bias_mask = torch.ones_like(output_layer.bias)
    bias_mask[:4] = 0.0
    output_layer.bias.register_hook(
      lambda grad: grad * bias_mask.to(device=grad.device, dtype=grad.dtype)
    )


def _configure_footprint_only_model(
  model: FootEventDetectorGRU,
  *,
  dummy_logit: float = -20.0,
) -> None:
  """Train contact/touchdown outputs while forcing toe-riser logits off."""
  for parameter in model.parameters():
    parameter.requires_grad_(True)

  output_layer = _detector_output_layer(model)
  with torch.no_grad():
    output_layer.weight[4:6].zero_()
    if output_layer.bias is not None:
      output_layer.bias[4:6].fill_(float(dummy_logit))

  weight_mask = torch.ones_like(output_layer.weight)
  weight_mask[4:6] = 0.0
  output_layer.weight.register_hook(
    lambda grad: grad * weight_mask.to(device=grad.device, dtype=grad.dtype)
  )

  if output_layer.bias is not None:
    bias_mask = torch.ones_like(output_layer.bias)
    bias_mask[4:6] = 0.0
    output_layer.bias.register_hook(
      lambda grad: grad * bias_mask.to(device=grad.device, dtype=grad.dtype)
    )


def _load_detector_checkpoint(
  model: FootEventDetectorGRU,
  checkpoint_path: str | Path,
  *,
  device: torch.device,
) -> None:
  path = Path(checkpoint_path).expanduser()
  state_obj = torch.load(path, map_location=device)
  if (
    isinstance(state_obj, dict)
    and "model_state_dict" in state_obj
    and isinstance(state_obj["model_state_dict"], dict)
  ):
    state_dict = state_obj["model_state_dict"]
  elif (
    isinstance(state_obj, dict)
    and "state_dict" in state_obj
    and isinstance(state_obj["state_dict"], dict)
  ):
    state_dict = state_obj["state_dict"]
  elif isinstance(state_obj, dict):
    state_dict = state_obj
  else:
    raise TypeError(
      f"Unsupported detector checkpoint payload type {type(state_obj).__name__}."
    )
  model.load_state_dict(state_dict)


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
  train_label_indices: tuple[int, ...] | None = None,
) -> torch.Tensor:
  if train_label_indices == (0, 1, 2, 3):
    contact_logits = logits[:, 0:2]
    contact_labels = labels[:, 0:2]
    touchdown_logits = logits[:, 2:4]
    touchdown_labels = labels[:, 2:4]
    contact_bce = F.binary_cross_entropy_with_logits(
      contact_logits,
      contact_labels,
      pos_weight=pos_weight[0:2],
    )
    touchdown_bce = F.binary_cross_entropy_with_logits(
      touchdown_logits,
      touchdown_labels,
      pos_weight=pos_weight[2:4],
    )
    focal = _asymmetric_focal_loss_with_logits(
      touchdown_logits,
      touchdown_labels,
      gamma_pos=focal_gamma_pos,
      gamma_neg=focal_gamma_neg,
    )
    tversky = _tversky_loss_from_logits(
      touchdown_logits,
      touchdown_labels,
      alpha=tversky_alpha,
      beta=tversky_beta,
    )
    consistency = torch.relu(
      torch.sigmoid(touchdown_logits) - torch.sigmoid(contact_logits)
    ).mean()
    return (
      contact_bce
      + touchdown_bce
      + focal_loss_weight * focal
      + tversky_loss_weight * tversky
      + 0.05 * consistency
    )

  if train_label_indices is not None:
    label_indices = list(train_label_indices)
    logits = logits[:, label_indices]
    labels = labels[:, label_indices]
    pos_weight = pos_weight[label_indices]
    event_logits = logits
    event_labels = labels
  else:
    event_logits = logits[:, 2:6]
    event_labels = labels[:, 2:6]
  bce = F.binary_cross_entropy_with_logits(
    logits,
    labels,
    pos_weight=pos_weight,
  )
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


def _current_footprint_anchor_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return world-frame sole-center anchors that deployment would write."""
  left_toe, right_toe, left_heel, right_heel = _world_frame_foot_positions(env)
  return torch.stack(
    (
      0.5 * (left_toe + left_heel),
      0.5 * (right_toe + right_heel),
    ),
    dim=1,
  ).to(dtype=torch.float32)


def _percentile_or_zero(values: list[float], quantile: float) -> float:
  if not values:
    return 0.0
  return float(np.quantile(np.asarray(values, dtype=np.float32), quantile))


def _mean_or_zero(values: list[float]) -> float:
  if not values:
    return 0.0
  return float(np.mean(np.asarray(values, dtype=np.float32)))


def _match_event_pairs_with_indices(
  true_frames: np.ndarray,
  true_indices: np.ndarray,
  pred_frames: np.ndarray,
  pred_indices: np.ndarray,
  *,
  tolerance_frames: int,
) -> list[tuple[int, int]]:
  """Greedily match prediction frames to true events and keep local indices."""
  matched = np.zeros(true_frames.shape[0], dtype=np.bool_)
  pairs: list[tuple[int, int]] = []
  for pred_pos, pred_frame in enumerate(pred_frames):
    candidates = np.nonzero(
      (~matched) & (np.abs(true_frames - pred_frame) <= tolerance_frames)
    )[0]
    if candidates.size == 0:
      continue
    best = candidates[np.argmin(np.abs(true_frames[candidates] - pred_frame))]
    matched[best] = True
    pairs.append((int(true_indices[best]), int(pred_indices[pred_pos])))
  return pairs


def _summarize_touchdown_alignment(
  *,
  labels: np.ndarray,
  probabilities: np.ndarray,
  episode_id: np.ndarray,
  frame_idx: np.ndarray,
  footprint_anchor_w: np.ndarray | None,
  thresholds_by_foot: tuple[float, float],
  prefix: str,
  valid_mask: np.ndarray | None,
  contact_threshold: float,
  contact_release_threshold: float,
  cooldown_frames: int,
  tolerance_frames: int,
) -> dict[str, float]:
  """Measure deployed touchdown trigger delay and resulting footprint error."""
  metrics: dict[str, float] = {}
  all_dt: list[float] = []
  all_abs_dt: list[float] = []
  all_xy_error: list[float] = []
  all_z_error: list[float] = []
  alignment_tolerance = max(int(tolerance_frames), int(cooldown_frames))

  for foot_id, side in enumerate(("left", "right")):
    label_index = 2 + foot_id
    foot_dt: list[float] = []
    foot_abs_dt: list[float] = []
    foot_xy_error: list[float] = []
    foot_z_error: list[float] = []
    if valid_mask is None:
      foot_valid_mask = np.ones(labels.shape[0], dtype=np.bool_)
    elif valid_mask.ndim == 2:
      foot_valid_mask = valid_mask[:, foot_id].astype(np.bool_)
    else:
      foot_valid_mask = valid_mask.astype(np.bool_)

    for episode in np.unique(episode_id):
      mask = (episode_id == episode) & foot_valid_mask
      if not mask.any():
        continue
      local_labels = labels[mask]
      local_probabilities = probabilities[mask]
      local_frames = frame_idx[mask].astype(np.int64)
      true_frames, true_indices = _event_start_frames_and_indices(
        local_labels[:, label_index] > 0.5,
        local_frames,
      )
      active = (
        local_probabilities[:, label_index] >= float(thresholds_by_foot[foot_id])
      ) & (local_probabilities[:, foot_id] >= float(contact_threshold))
      pred_frames, pred_indices = _event_start_frames_and_indices(
        active,
        local_frames,
        cooldown_frames=cooldown_frames,
        release_values=local_probabilities[:, foot_id],
        release_threshold=contact_release_threshold,
      )
      pairs = _match_event_pairs_with_indices(
        true_frames,
        true_indices,
        pred_frames,
        pred_indices,
        tolerance_frames=alignment_tolerance,
      )
      if not pairs:
        continue
      local_anchor = (
        footprint_anchor_w[mask, foot_id] if footprint_anchor_w is not None else None
      )
      for true_index, pred_index in pairs:
        dt = float(local_frames[pred_index] - local_frames[true_index])
        foot_dt.append(dt)
        foot_abs_dt.append(abs(dt))
        if local_anchor is not None:
          delta = local_anchor[pred_index] - local_anchor[true_index]
          foot_xy_error.append(float(np.linalg.norm(delta[:2])))
          foot_z_error.append(float(abs(delta[2])))

    all_dt.extend(foot_dt)
    all_abs_dt.extend(foot_abs_dt)
    all_xy_error.extend(foot_xy_error)
    all_z_error.extend(foot_z_error)
    metrics[f"{side}_{prefix}_timing_matched_count"] = float(len(foot_dt))
    metrics[f"{side}_{prefix}_timing_signed_dt_frames_mean"] = _mean_or_zero(foot_dt)
    metrics[f"{side}_{prefix}_timing_abs_dt_frames_p50"] = _percentile_or_zero(
      foot_abs_dt,
      0.50,
    )
    metrics[f"{side}_{prefix}_timing_abs_dt_frames_p90"] = _percentile_or_zero(
      foot_abs_dt,
      0.90,
    )
    if footprint_anchor_w is not None:
      metrics[f"{side}_{prefix}_footprint_xy_error_m_p50"] = _percentile_or_zero(
        foot_xy_error,
        0.50,
      )
      metrics[f"{side}_{prefix}_footprint_xy_error_m_p90"] = _percentile_or_zero(
        foot_xy_error,
        0.90,
      )
      metrics[f"{side}_{prefix}_footprint_z_error_m_p50"] = _percentile_or_zero(
        foot_z_error,
        0.50,
      )
      metrics[f"{side}_{prefix}_footprint_z_error_m_p90"] = _percentile_or_zero(
        foot_z_error,
        0.90,
      )

  metrics[f"{prefix}_timing_matched_count"] = float(len(all_dt))
  metrics[f"{prefix}_timing_signed_dt_frames_mean"] = _mean_or_zero(all_dt)
  metrics[f"{prefix}_timing_abs_dt_frames_p50"] = _percentile_or_zero(
    all_abs_dt,
    0.50,
  )
  metrics[f"{prefix}_timing_abs_dt_frames_p90"] = _percentile_or_zero(
    all_abs_dt,
    0.90,
  )
  if footprint_anchor_w is not None:
    metrics[f"{prefix}_footprint_xy_error_m_p50"] = _percentile_or_zero(
      all_xy_error,
      0.50,
    )
    metrics[f"{prefix}_footprint_xy_error_m_p90"] = _percentile_or_zero(
      all_xy_error,
      0.90,
    )
    metrics[f"{prefix}_footprint_z_error_m_p50"] = _percentile_or_zero(
      all_z_error,
      0.50,
    )
    metrics[f"{prefix}_footprint_z_error_m_p90"] = _percentile_or_zero(
      all_z_error,
      0.90,
    )
  return metrics


def _deployment_event_metrics(
  *,
  labels: np.ndarray,
  probabilities: np.ndarray,
  episode_id: np.ndarray,
  frame_idx: np.ndarray,
  stair_support: np.ndarray,
  thresholds: np.ndarray,
  footprint_anchor_w: np.ndarray | None = None,
  ignore_toe_for_scores: bool = False,
  touchdown_contact_threshold: float,
  touchdown_contact_release_threshold: float,
  touchdown_cooldown_frames: int,
  toe_hit_cooldown_frames: int,
  touchdown_recall_precision_floor: float,
  toe_hit_recall_precision_floor: float,
  selection_min_stair_touchdown_events: int,
  event_tolerance_frames: int,
) -> dict[str, float]:
  """Add deploy-style threshold sweep metrics for footstep and toe-hit events."""
  metrics: dict[str, float] = {}
  touchdown_f1: list[float] = []
  touchdown_high_recall_precision: list[float] = []
  touchdown_high_recall_recall: list[float] = []
  touchdown_high_recall_f1: list[float] = []
  touchdown_contact_fallback_precision: list[float] = []
  touchdown_contact_fallback_recall: list[float] = []
  touchdown_contact_fallback_f1: list[float] = []
  flat_touchdown_f1: list[float] = []
  flat_touchdown_high_recall_precision: list[float] = []
  flat_touchdown_high_recall_recall: list[float] = []
  flat_touchdown_high_recall_f1: list[float] = []
  stair_touchdown_f1: list[float] = []
  stair_touchdown_high_recall_precision: list[float] = []
  stair_touchdown_high_recall_recall: list[float] = []
  stair_touchdown_high_recall_f1: list[float] = []
  stair_touchdown_true_count = 0.0
  toe_f1: list[float] = []
  toe_high_recall_precision: list[float] = []
  toe_high_recall_recall: list[float] = []
  toe_high_recall_f1: list[float] = []
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
      recall_precision_floor=touchdown_recall_precision_floor,
    )
    for key, value in touchdown_stats.items():
      metrics[f"{side}_touchdown_deploy_{key}"] = float(value)
    touchdown_f1.append(float(touchdown_stats["best_event_f1"]))
    touchdown_high_recall_precision.append(
      float(touchdown_stats["high_recall_event_precision"])
    )
    touchdown_high_recall_recall.append(
      float(touchdown_stats["high_recall_event_recall"])
    )
    touchdown_high_recall_f1.append(float(touchdown_stats["high_recall_event_f1"]))

    fallback_stats = sweep_deployment_event_thresholds(
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
      contact_rising_fallback=True,
      recall_precision_floor=touchdown_recall_precision_floor,
    )
    for key, value in fallback_stats.items():
      metrics[f"{side}_touchdown_contact_fallback_deploy_{key}"] = float(value)
    touchdown_contact_fallback_precision.append(
      float(fallback_stats["high_recall_event_precision"])
    )
    touchdown_contact_fallback_recall.append(
      float(fallback_stats["high_recall_event_recall"])
    )
    touchdown_contact_fallback_f1.append(float(fallback_stats["high_recall_event_f1"]))

    flat_stats = sweep_deployment_event_thresholds(
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
      valid_mask=~stair_support[:, foot_id].astype(np.bool_),
      recall_precision_floor=touchdown_recall_precision_floor,
    )
    for key, value in flat_stats.items():
      metrics[f"{side}_touchdown_flat_deploy_{key}"] = float(value)
    flat_touchdown_f1.append(float(flat_stats["best_event_f1"]))
    flat_touchdown_high_recall_precision.append(
      float(flat_stats["high_recall_event_precision"])
    )
    flat_touchdown_high_recall_recall.append(
      float(flat_stats["high_recall_event_recall"])
    )
    flat_touchdown_high_recall_f1.append(float(flat_stats["high_recall_event_f1"]))

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
      recall_precision_floor=touchdown_recall_precision_floor,
    )
    for key, value in stair_stats.items():
      metrics[f"{side}_touchdown_stair_deploy_{key}"] = float(value)
    stair_touchdown_f1.append(float(stair_stats["best_event_f1"]))
    stair_touchdown_high_recall_precision.append(
      float(stair_stats["high_recall_event_precision"])
    )
    stair_touchdown_high_recall_recall.append(
      float(stair_stats["high_recall_event_recall"])
    )
    stair_touchdown_high_recall_f1.append(float(stair_stats["high_recall_event_f1"]))
    stair_touchdown_true_count += float(stair_stats["best_true_event_count"])

    toe_stats = sweep_deployment_event_thresholds(
      labels=labels,
      probabilities=probabilities,
      episode_id=episode_id,
      frame_idx=frame_idx,
      event_label_index=toe_hit_index,
      thresholds=thresholds,
      tolerance_frames=event_tolerance_frames,
      cooldown_frames=toe_hit_cooldown_frames,
      recall_precision_floor=toe_hit_recall_precision_floor,
    )
    for key, value in toe_stats.items():
      metrics[f"{side}_toe_riser_hit_deploy_{key}"] = float(value)
    toe_f1.append(float(toe_stats["best_event_f1"]))
    toe_high_recall_precision.append(float(toe_stats["high_recall_event_precision"]))
    toe_high_recall_recall.append(float(toe_stats["high_recall_event_recall"]))
    toe_high_recall_f1.append(float(toe_stats["high_recall_event_f1"]))

  metrics["touchdown_deploy_macro_f1"] = float(np.mean(touchdown_f1))
  metrics["touchdown_high_recall_macro_precision"] = float(
    np.mean(touchdown_high_recall_precision)
  )
  metrics["touchdown_high_recall_macro_recall"] = float(
    np.mean(touchdown_high_recall_recall)
  )
  metrics["touchdown_high_recall_macro_f1"] = float(np.mean(touchdown_high_recall_f1))
  metrics["touchdown_contact_fallback_macro_precision"] = float(
    np.mean(touchdown_contact_fallback_precision)
  )
  metrics["touchdown_contact_fallback_macro_recall"] = float(
    np.mean(touchdown_contact_fallback_recall)
  )
  metrics["touchdown_contact_fallback_macro_f1"] = float(
    np.mean(touchdown_contact_fallback_f1)
  )
  metrics["touchdown_flat_deploy_macro_f1"] = float(np.mean(flat_touchdown_f1))
  metrics["touchdown_flat_high_recall_macro_precision"] = float(
    np.mean(flat_touchdown_high_recall_precision)
  )
  metrics["touchdown_flat_high_recall_macro_recall"] = float(
    np.mean(flat_touchdown_high_recall_recall)
  )
  metrics["touchdown_flat_high_recall_macro_f1"] = float(
    np.mean(flat_touchdown_high_recall_f1)
  )
  metrics["touchdown_stair_deploy_macro_f1"] = float(np.mean(stair_touchdown_f1))
  metrics["touchdown_stair_high_recall_macro_precision"] = float(
    np.mean(stair_touchdown_high_recall_precision)
  )
  metrics["touchdown_stair_high_recall_macro_recall"] = float(
    np.mean(stair_touchdown_high_recall_recall)
  )
  metrics["touchdown_stair_high_recall_macro_f1"] = float(
    np.mean(stair_touchdown_high_recall_f1)
  )
  metrics["touchdown_stair_deploy_true_event_count"] = float(stair_touchdown_true_count)
  metrics["toe_riser_deploy_macro_f1"] = float(np.mean(toe_f1))
  metrics["toe_riser_high_recall_macro_precision"] = float(
    np.mean(toe_high_recall_precision)
  )
  metrics["toe_riser_high_recall_macro_recall"] = float(np.mean(toe_high_recall_recall))
  metrics["toe_riser_high_recall_macro_f1"] = float(np.mean(toe_high_recall_f1))
  metrics["deploy_event_macro_f1"] = float(np.mean(touchdown_f1 + toe_f1))
  stair_count_factor = min(
    stair_touchdown_true_count / max(float(selection_min_stair_touchdown_events), 1.0),
    1.0,
  )
  touchdown_guard = min(
    metrics["touchdown_high_recall_macro_precision"]
    / max(float(touchdown_recall_precision_floor), 1.0e-6),
    1.0,
  )
  stair_guard = min(
    metrics["touchdown_stair_high_recall_macro_precision"]
    / max(float(touchdown_recall_precision_floor), 1.0e-6),
    1.0,
  )
  touchdown_only_precision_guard = 0.55 * touchdown_guard + 0.45 * stair_guard
  toe_guard = min(
    metrics["toe_riser_high_recall_macro_precision"]
    / max(float(toe_hit_recall_precision_floor), 1.0e-6),
    1.0,
  )
  metrics["high_recall_precision_guard"] = float(
    0.45 * touchdown_guard + 0.35 * stair_guard + 0.20 * toe_guard
  )
  metrics["toe_riser_high_recall_score"] = float(
    metrics["toe_riser_high_recall_macro_recall"] * toe_guard
  )
  metrics["touchdown_high_recall_score"] = float(
    stair_count_factor
    * (
      0.45 * metrics["touchdown_high_recall_macro_recall"]
      + 0.35 * metrics["touchdown_stair_high_recall_macro_recall"]
      + 0.20 * touchdown_only_precision_guard
    )
  )
  if ignore_toe_for_scores:
    metrics["deploy_event_macro_f1"] = metrics["touchdown_deploy_macro_f1"]
    metrics["high_recall_precision_guard"] = float(touchdown_only_precision_guard)
    metrics["footprint_deploy_score"] = float(
      stair_count_factor
      * (
        0.35 * metrics["touchdown_high_recall_macro_recall"]
        + 0.20 * metrics["touchdown_high_recall_macro_precision"]
        + 0.25 * metrics["touchdown_stair_deploy_macro_f1"]
        + 0.20 * metrics["touchdown_deploy_macro_f1"]
      )
    )
    metrics["high_recall_footprint_score"] = metrics["touchdown_high_recall_score"]
    metrics["toe_guarded_footprint_score"] = metrics["touchdown_high_recall_score"]
  else:
    metrics["footprint_deploy_score"] = float(
      stair_count_factor
      * (
        0.45 * metrics["touchdown_high_recall_macro_recall"]
        + 0.15 * metrics["touchdown_high_recall_macro_precision"]
        + 0.30 * metrics["touchdown_stair_deploy_macro_f1"]
        + 0.10 * metrics["toe_riser_deploy_macro_f1"]
      )
    )
    metrics["high_recall_footprint_score"] = float(
      stair_count_factor
      * (
        0.35 * metrics["touchdown_high_recall_macro_recall"]
        + 0.30 * metrics["touchdown_stair_high_recall_macro_recall"]
        + 0.20 * metrics["toe_riser_high_recall_macro_recall"]
        + 0.15 * metrics["high_recall_precision_guard"]
      )
    )
    metrics["toe_guarded_footprint_score"] = float(
      stair_count_factor
      * (
        0.25 * metrics["touchdown_high_recall_macro_recall"]
        + 0.25 * metrics["touchdown_stair_high_recall_macro_recall"]
        + 0.30 * metrics["toe_riser_high_recall_score"]
        + 0.20 * metrics["high_recall_precision_guard"]
      )
    )
  touchdown_thresholds = (
    float(metrics["left_touchdown_deploy_high_recall_threshold"]),
    float(metrics["right_touchdown_deploy_high_recall_threshold"]),
  )
  metrics.update(
    _summarize_touchdown_alignment(
      labels=labels,
      probabilities=probabilities,
      episode_id=episode_id,
      frame_idx=frame_idx,
      footprint_anchor_w=footprint_anchor_w,
      thresholds_by_foot=touchdown_thresholds,
      prefix="touchdown",
      valid_mask=None,
      contact_threshold=touchdown_contact_threshold,
      contact_release_threshold=touchdown_contact_release_threshold,
      cooldown_frames=touchdown_cooldown_frames,
      tolerance_frames=event_tolerance_frames,
    )
  )
  metrics.update(
    _summarize_touchdown_alignment(
      labels=labels,
      probabilities=probabilities,
      episode_id=episode_id,
      frame_idx=frame_idx,
      footprint_anchor_w=footprint_anchor_w,
      thresholds_by_foot=touchdown_thresholds,
      prefix="touchdown_stair",
      valid_mask=stair_support,
      contact_threshold=touchdown_contact_threshold,
      contact_release_threshold=touchdown_contact_release_threshold,
      cooldown_frames=touchdown_cooldown_frames,
      tolerance_frames=event_tolerance_frames,
    )
  )
  matched_count = metrics["touchdown_timing_matched_count"]
  if matched_count <= 0.0:
    timing_guard = 0.0
    footprint_xy_guard = 0.0
  else:
    timing_guard = max(
      0.0,
      1.0
      - metrics["touchdown_timing_abs_dt_frames_p90"]
      / max(float(touchdown_cooldown_frames), 1.0),
    )
    footprint_xy_guard = 1.0
    if footprint_anchor_w is not None:
      footprint_xy_guard = max(
        0.0,
        1.0 - metrics["touchdown_footprint_xy_error_m_p90"] / 0.12,
      )
  metrics["touchdown_timing_guard"] = float(timing_guard)
  if footprint_anchor_w is not None:
    metrics["touchdown_footprint_xy_guard"] = float(footprint_xy_guard)
  metrics["touchdown_timing_guarded_score"] = float(
    metrics["touchdown_high_recall_score"]
    * (0.70 * timing_guard + 0.30 * footprint_xy_guard)
  )
  metrics["timing_guarded_footprint_score"] = float(
    metrics["high_recall_footprint_score"]
    * (0.70 * timing_guard + 0.30 * footprint_xy_guard)
  )
  if ignore_toe_for_scores:
    timing_quality = 0.70 * timing_guard + 0.30 * footprint_xy_guard
    metrics["touchdown_v3_surpass_score"] = float(
      stair_count_factor
      * (
        0.28 * metrics["touchdown_high_recall_macro_recall"]
        + 0.28 * metrics["touchdown_stair_high_recall_macro_recall"]
        + 0.16 * metrics["touchdown_high_recall_macro_precision"]
        + 0.16 * metrics["touchdown_stair_high_recall_macro_precision"]
        + 0.12 * metrics["touchdown_deploy_macro_f1"]
      )
      * timing_quality
    )
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
  touchdown_recall_precision_floor: float,
  toe_hit_recall_precision_floor: float,
  selection_min_stair_touchdown_events: int,
  train_label_indices: tuple[int, ...] | None = None,
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
      full_logits = model(obs)
      logits = full_logits
      loss_pos_weight = pos_weight
      if train_label_indices is not None:
        label_indices = list(train_label_indices)
        logits = full_logits[:, label_indices]
        label = label[:, label_indices]
        loss_pos_weight = pos_weight[label_indices]
      loss = F.binary_cross_entropy_with_logits(
        logits,
        label,
        pos_weight=loss_pos_weight,
      )
      count = int(end - start)
      total_loss += float(loss.item()) * count
      total_count += count
      probabilities.append(torch.sigmoid(full_logits).cpu().numpy().astype(np.float32))
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
      footprint_anchor_w=snapshot["footprint_anchor_w"].astype(np.float32),
      ignore_toe_for_scores=train_label_indices == (0, 1, 2, 3),
      touchdown_contact_threshold=touchdown_contact_threshold,
      touchdown_contact_release_threshold=touchdown_contact_release_threshold,
      touchdown_cooldown_frames=touchdown_cooldown_frames,
      toe_hit_cooldown_frames=toe_hit_cooldown_frames,
      touchdown_recall_precision_floor=touchdown_recall_precision_floor,
      toe_hit_recall_precision_floor=toe_hit_recall_precision_floor,
      selection_min_stair_touchdown_events=selection_min_stair_touchdown_events,
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
  train_labels = snapshot["train_labels"].astype(np.float32)
  toe_negative = train_labels[:, 4:6].max(axis=1) <= 0.0
  toe_fp = (probabilities[:, 4:6].max(axis=1) >= threshold) & (toe_negative)
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
    hard_mask |= same_sequence & near_peak & toe_negative
  return buffer.mark_false_positive_hard_negatives(hard_mask)


def _mine_false_positive_touchdown_hard_negatives(
  model: FootEventDetectorGRU,
  buffer: OnlineFootEventReplayBuffer,
  *,
  device: torch.device,
  batch_size: int,
  threshold: float,
  contact_threshold: float,
  window_frames: int,
  max_peaks: int,
) -> int:
  """Replay likely false touchdown triggers around already-negative windows."""
  if buffer.size == 0:
    return 0
  snapshot, probabilities = _predict_buffer_probabilities(
    model,
    buffer,
    device=device,
    batch_size=batch_size,
  )
  train_labels = snapshot["train_labels"].astype(np.float32)
  touchdown_negative = train_labels[:, 2:4].max(axis=1) <= 0.0
  left_fp = (
    (probabilities[:, 2] >= threshold)
    & (probabilities[:, 0] >= contact_threshold)
    & touchdown_negative
  )
  right_fp = (
    (probabilities[:, 3] >= threshold)
    & (probabilities[:, 1] >= contact_threshold)
    & touchdown_negative
  )
  peak_indices = np.nonzero(left_fp | right_fp)[0].astype(np.int64)
  if peak_indices.shape[0] > max_peaks:
    scores = np.maximum(
      probabilities[peak_indices, 2] * probabilities[peak_indices, 0],
      probabilities[peak_indices, 3] * probabilities[peak_indices, 1],
    )
    peak_indices = peak_indices[np.argsort(-scores)[:max_peaks]]

  hard_mask = np.zeros(train_labels.shape[0], dtype=np.bool_)
  episode_id = snapshot["episode_id"].astype(np.int64)
  frame_idx = snapshot["frame_idx"].astype(np.int64)
  env_id = snapshot["env_id"].astype(np.int64)
  for peak in peak_indices:
    same_sequence = (episode_id == episode_id[peak]) & (env_id == env_id[peak])
    near_peak = np.abs(frame_idx - frame_idx[peak]) <= window_frames
    hard_mask |= same_sequence & near_peak & touchdown_negative
  return buffer.mark_false_positive_hard_negatives(hard_mask)


def _mine_false_negative_hard_positives(
  model: FootEventDetectorGRU,
  buffer: OnlineFootEventReplayBuffer,
  *,
  device: torch.device,
  batch_size: int,
  threshold: float,
  window_frames: int,
  max_peaks: int,
) -> int:
  """Replay touchdown events whose predicted touchdown probability is too low."""
  if buffer.size == 0:
    return 0
  snapshot, probabilities = _predict_buffer_probabilities(
    model,
    buffer,
    device=device,
    batch_size=batch_size,
  )
  labels = snapshot["labels"].astype(np.float32)
  train_labels = snapshot["train_labels"].astype(np.float32)
  missed_touchdown = ((labels[:, 2] > 0.5) & (probabilities[:, 2] < threshold)) | (
    (labels[:, 3] > 0.5) & (probabilities[:, 3] < threshold)
  )
  peak_indices = np.nonzero(missed_touchdown)[0].astype(np.int64)
  if peak_indices.shape[0] > max_peaks:
    order = np.argsort(probabilities[peak_indices, 2:4].max(axis=1))
    peak_indices = peak_indices[order[:max_peaks]]
  hard_mask = np.zeros(labels.shape[0], dtype=np.bool_)
  episode_id = snapshot["episode_id"].astype(np.int64)
  frame_idx = snapshot["frame_idx"].astype(np.int64)
  env_id = snapshot["env_id"].astype(np.int64)
  soft_positive = train_labels[:, 2:4].max(axis=1) > 0.0
  for peak in peak_indices:
    same_sequence = (episode_id == episode_id[peak]) & (env_id == env_id[peak])
    near_peak = np.abs(frame_idx - frame_idx[peak]) <= window_frames
    hard_mask |= same_sequence & near_peak & soft_positive
  return buffer.mark_false_negative_hard_positives(hard_mask)


def _mine_false_negative_toe_hard_positives(
  model: FootEventDetectorGRU,
  buffer: OnlineFootEventReplayBuffer,
  *,
  device: torch.device,
  batch_size: int,
  threshold: float,
  window_frames: int,
  max_peaks: int,
) -> int:
  """Replay toe-riser hit events whose predicted probability is too low."""
  if buffer.size == 0:
    return 0
  snapshot, probabilities = _predict_buffer_probabilities(
    model,
    buffer,
    device=device,
    batch_size=batch_size,
  )
  labels = snapshot["labels"].astype(np.float32)
  train_labels = snapshot["train_labels"].astype(np.float32)
  missed_toe = ((labels[:, 4] > 0.5) & (probabilities[:, 4] < threshold)) | (
    (labels[:, 5] > 0.5) & (probabilities[:, 5] < threshold)
  )
  peak_indices = np.nonzero(missed_toe)[0].astype(np.int64)
  if peak_indices.shape[0] > max_peaks:
    order = np.argsort(probabilities[peak_indices, 4:6].max(axis=1))
    peak_indices = peak_indices[order[:max_peaks]]
  hard_mask = np.zeros(labels.shape[0], dtype=np.bool_)
  episode_id = snapshot["episode_id"].astype(np.int64)
  frame_idx = snapshot["frame_idx"].astype(np.int64)
  env_id = snapshot["env_id"].astype(np.int64)
  soft_positive = train_labels[:, 4:6].max(axis=1) > 0.0
  for peak in peak_indices:
    same_sequence = (episode_id == episode_id[peak]) & (env_id == env_id[peak])
    near_peak = np.abs(frame_idx - frame_idx[peak]) <= window_frames
    hard_mask |= same_sequence & near_peak & soft_positive
  return buffer.mark_false_negative_toe_hard_positives(hard_mask)


def _should_mine_touchdown_false_positive_hard_negatives(
  cfg: OnlineFootEventDetectorConfig,
) -> bool:
  return (
    cfg.mine_false_positive_touchdown_hard_negatives
    and not cfg.toe_only_finetune
    and not cfg.toe_riser_only_model
  )


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


def _best_val_metrics_from_payload(payload: dict[str, Any]) -> dict[str, float]:
  if "history" in payload and "best_step" in payload:
    best_step = int(payload["best_step"])
    for record in payload["history"]:
      if int(record.get("step", -1)) != best_step:
        continue
      values = record.get("val", record)
      if isinstance(values, dict):
        return {
          key: float(value)
          for key, value in values.items()
          if isinstance(value, int | float)
        }
  if "best_metrics" in payload and isinstance(payload["best_metrics"], dict):
    return {
      key: float(value)
      for key, value in payload["best_metrics"].items()
      if isinstance(value, int | float)
    }
  if "val" in payload and isinstance(payload["val"], dict):
    return {
      key: float(value)
      for key, value in payload["val"].items()
      if isinstance(value, int | float)
    }
  return {
    key: float(value)
    for key, value in payload.items()
    if isinstance(value, int | float)
  }


def _load_baseline_metrics(path: str | Path | None) -> dict[str, float] | None:
  if path is None:
    return None
  metrics_path = Path(path).expanduser()
  with metrics_path.open("r", encoding="utf-8") as stream:
    payload = json.load(stream)
  if not isinstance(payload, dict):
    raise TypeError(f"Expected JSON object in baseline metrics file {metrics_path}.")
  metrics = _best_val_metrics_from_payload(payload)
  if not metrics:
    raise ValueError(f"No numeric baseline metrics found in {metrics_path}.")
  return metrics


def _baseline_guard_passed(
  metrics: dict[str, float],
  baseline_metrics: dict[str, float] | None,
  *,
  touchdown_recall_tolerance: float,
  stair_touchdown_recall_tolerance: float,
  flat_touchdown_f1_tolerance: float,
  toe_metric: str,
  toe_metric_min_improvement: float,
  required_metric_names: tuple[str, ...] = (),
  required_metric_min_improvement: float = 0.0,
) -> tuple[bool, tuple[str, ...]]:
  """Return whether a candidate keeps baseline footprint behavior intact."""
  if baseline_metrics is None:
    return True, ()

  failures: list[str] = []
  floor_checks = (
    (
      "touchdown_high_recall_macro_recall",
      touchdown_recall_tolerance,
    ),
    (
      "touchdown_stair_high_recall_macro_recall",
      stair_touchdown_recall_tolerance,
    ),
    (
      "touchdown_flat_high_recall_macro_f1",
      flat_touchdown_f1_tolerance,
    ),
  )
  for key, tolerance in floor_checks:
    if key not in baseline_metrics or key not in metrics:
      failures.append(f"{key}=missing")
      continue
    required = float(baseline_metrics[key]) - float(tolerance)
    actual = float(metrics[key])
    if actual < required:
      failures.append(f"{key} {actual:.4f} < {required:.4f}")

  if toe_metric:
    if toe_metric not in baseline_metrics or toe_metric not in metrics:
      failures.append(f"{toe_metric}=missing")
    else:
      required = float(baseline_metrics[toe_metric]) + float(toe_metric_min_improvement)
      actual = float(metrics[toe_metric])
      if actual < required:
        failures.append(f"{toe_metric} {actual:.4f} < {required:.4f}")

  for key in required_metric_names:
    if key not in baseline_metrics or key not in metrics:
      failures.append(f"{key}=missing")
      continue
    required = float(baseline_metrics[key]) + float(required_metric_min_improvement)
    actual = float(metrics[key])
    if actual < required:
      failures.append(f"{key} {actual:.4f} < {required:.4f}")

  return len(failures) == 0, tuple(failures)


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


def _current_git_commit() -> str | None:
  repo_root = Path(__file__).resolve().parents[2]
  try:
    return subprocess.check_output(
      ["git", "rev-parse", "HEAD"],
      cwd=repo_root,
      text=True,
      stderr=subprocess.DEVNULL,
    ).strip()
  except (OSError, subprocess.CalledProcessError):
    return None


def _deployment_contract_payload(
  *,
  task_id: str,
  cfg: OnlineFootEventDetectorConfig,
  obs_dim: int,
  trained_label_indices: tuple[int, ...] | None,
  stair_hard_negative_label_indices: tuple[int, ...],
  best_deployment_thresholds: dict[str, float],
  onnx_path: Path | None,
) -> dict[str, Any]:
  """Build a compact deploy-side contract for the exported detector."""
  dummy_low_logit_indices = [4, 5] if cfg.footprint_only_model else []
  if cfg.toe_riser_only_model:
    dummy_low_logit_indices = [0, 1, 2, 3]
  return {
    "schema_version": 1,
    "task_id": task_id,
    "source_git_commit": _current_git_commit(),
    "input_schema": cfg.input_schema,
    "input_source": (
      "body-frame FK, IMU projected gravity, command, gait phase, action, "
      "joint position, and joint velocity"
    ),
    "obs_history_shape": [1, cfg.history_len, obs_dim],
    "obs_dim": obs_dim,
    "history_len": cfg.history_len,
    "output_names": list(FOOT_EVENT_LABEL_NAMES),
    "trained_label_indices": list(trained_label_indices)
    if trained_label_indices is not None
    else None,
    "dummy_low_logit_indices": dummy_low_logit_indices,
    "input_feature_groups": foot_event_input_feature_groups(
      include_gait_phase=cfg.include_gait_phase,
      input_schema=cfg.input_schema,
    ),
    "input_feature_scale_groups": foot_event_input_feature_scale_groups(
      include_gait_phase=cfg.include_gait_phase,
      input_schema=cfg.input_schema,
    ),
    "input_feature_scales": foot_event_input_feature_scales(
      include_gait_phase=cfg.include_gait_phase,
      input_schema=cfg.input_schema,
    ),
    "best_deployment_thresholds": best_deployment_thresholds,
    "event_logic": {
      "touchdown_contact_threshold": cfg.touchdown_contact_threshold,
      "touchdown_contact_release_threshold": cfg.touchdown_contact_release_threshold,
      "touchdown_cooldown_frames": cfg.touchdown_cooldown_frames,
      "toe_hit_cooldown_frames": cfg.toe_hit_cooldown_frames,
    },
    "training_label_contract": {
      "label_names": list(FOOT_EVENT_LABEL_NAMES),
      "label_diagnostic_names": list(FOOT_EVENT_LABEL_DIAGNOSTIC_NAMES),
      "stair_hard_negative_label_indices": list(stair_hard_negative_label_indices),
    },
    "reset_required_state": [
      "obs_history_ring_buffer",
      "previous_body_frame_fk_positions",
      "previous_body_frame_fk_velocities",
      "previous_base_angular_velocity",
      "previous_action",
      "touchdown_cooldown_state",
      "contact_hysteresis_state",
    ],
    "onnx_path": str(onnx_path) if onnx_path is not None else None,
  }


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
  obs_dim = resolve_foot_event_detector_obs_dim(
    cfg.expected_obs_dim,
    include_gait_phase=cfg.include_gait_phase,
    input_schema=cfg.input_schema,
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
  exclusive_output_modes = (
    cfg.toe_only_finetune,
    cfg.toe_riser_only_model,
    cfg.footprint_only_model,
  )
  if sum(bool(value) for value in exclusive_output_modes) > 1:
    raise ValueError(
      "toe_only_finetune, toe_riser_only_model, and footprint_only_model are "
      "mutually exclusive."
    )
  if cfg.toe_only_finetune and cfg.init_detector_checkpoint is None:
    raise ValueError("toe_only_finetune requires init_detector_checkpoint.")
  sample_fractions = (
    cfg.toe_positive_fraction,
    cfg.toe_soft_positive_fraction,
    cfg.touchdown_positive_fraction,
    cfg.touchdown_soft_positive_fraction,
    cfg.stair_hard_negative_fraction,
    cfg.false_positive_hard_negative_fraction,
    cfg.false_negative_hard_positive_fraction,
    cfg.false_negative_toe_hard_positive_fraction,
  )
  if any(value < 0.0 for value in sample_fractions) or sum(sample_fractions) > 1.0:
    raise ValueError("Stratified replay fractions must be non-negative and sum <= 1.")
  if cfg.soft_touchdown_radius < 0 or cfg.soft_toe_hit_radius < 0:
    raise ValueError("soft event radii must be non-negative.")
  if cfg.hard_negative_mining_window_frames < 0:
    raise ValueError("hard_negative_mining_window_frames must be non-negative.")
  if cfg.hard_negative_mining_max_peaks <= 0:
    raise ValueError("hard_negative_mining_max_peaks must be positive.")
  if cfg.hard_touchdown_false_positive_mining_window_frames < 0:
    raise ValueError(
      "hard_touchdown_false_positive_mining_window_frames must be non-negative."
    )
  if cfg.hard_touchdown_false_positive_mining_max_peaks <= 0:
    raise ValueError("hard_touchdown_false_positive_mining_max_peaks must be positive.")
  if cfg.hard_positive_mining_window_frames < 0:
    raise ValueError("hard_positive_mining_window_frames must be non-negative.")
  if cfg.hard_positive_mining_max_peaks <= 0:
    raise ValueError("hard_positive_mining_max_peaks must be positive.")
  if cfg.hard_toe_positive_mining_window_frames < 0:
    raise ValueError("hard_toe_positive_mining_window_frames must be non-negative.")
  if cfg.hard_toe_positive_mining_max_peaks <= 0:
    raise ValueError("hard_toe_positive_mining_max_peaks must be positive.")
  if not 0.0 <= cfg.toe_soft_positive_threshold <= 1.0:
    raise ValueError("toe_soft_positive_threshold must be in [0, 1].")
  if not 0.0 <= cfg.touchdown_soft_positive_threshold <= 1.0:
    raise ValueError("touchdown_soft_positive_threshold must be in [0, 1].")
  mining_thresholds = (
    cfg.hard_negative_mining_threshold,
    cfg.hard_touchdown_false_positive_mining_threshold,
    cfg.hard_touchdown_false_positive_mining_contact_threshold,
    cfg.hard_positive_mining_threshold,
    cfg.hard_toe_positive_mining_threshold,
  )
  if any(value < 0.0 or value > 1.0 for value in mining_thresholds):
    raise ValueError("Hard mining thresholds must be in [0, 1].")
  if not 0.0 <= cfg.touchdown_recall_precision_floor <= 1.0:
    raise ValueError("touchdown_recall_precision_floor must be in [0, 1].")
  if not 0.0 <= cfg.toe_hit_recall_precision_floor <= 1.0:
    raise ValueError("toe_hit_recall_precision_floor must be in [0, 1].")
  if cfg.selection_min_stair_touchdown_events < 0:
    raise ValueError("selection_min_stair_touchdown_events must be non-negative.")
  if cfg.touchdown_cooldown_frames < 0 or cfg.toe_hit_cooldown_frames < 0:
    raise ValueError("event cooldown frames must be non-negative.")
  if cfg.sweep_threshold_steps <= 0:
    raise ValueError("sweep_threshold_steps must be positive.")
  if not 0.0 < cfg.sweep_threshold_min <= cfg.sweep_threshold_max < 1.0:
    raise ValueError("sweep thresholds must satisfy 0 < min <= max < 1.")
  baseline_tolerances = (
    cfg.baseline_touchdown_recall_tolerance,
    cfg.baseline_stair_touchdown_recall_tolerance,
    cfg.baseline_flat_touchdown_f1_tolerance,
    cfg.baseline_toe_metric_min_improvement,
  )
  if any(value < 0.0 for value in baseline_tolerances):
    raise ValueError("Baseline guard tolerances must be non-negative.")

  random.seed(cfg.seed)
  np.random.seed(cfg.seed)
  torch.manual_seed(cfg.seed)
  configure_torch_backends()
  device = torch.device(
    cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  )
  output_dir = Path(cfg.output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  eval_checkpoint_dir = output_dir / "checkpoints"
  if cfg.save_eval_checkpoints:
    eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
  last_checkpoint_path = output_dir / "last.pt"

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
  agent_cfg = resolve_inference_agent_cfg(
    checkpoint_path=checkpoint_path,
    agent_cfg=agent_cfg,
  )
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=str(device), render_mode=None)
  wrapped = RslRlVecEnvWrapper(raw_env, clip_actions=get_clip_actions(agent_cfg))
  history_buffer = StairProbeHistoryBuffer(
    num_envs=cfg.num_envs,
    history_len=cfg.history_len,
    obs_dim=obs_dim,
    device=device,
  )
  stair_hard_negative_label_indices = (
    (2, 3) if cfg.footprint_only_model else (2, 3, 4, 5)
  )
  train_buffer = OnlineFootEventReplayBuffer(
    capacity=cfg.train_buffer_capacity,
    history_len=cfg.history_len,
    obs_dim=obs_dim,
    label_dim=len(FOOT_EVENT_LABEL_NAMES),
    num_envs=cfg.num_envs,
    soft_touchdown_radius=cfg.soft_touchdown_radius,
    soft_toe_hit_radius=cfg.soft_toe_hit_radius,
    soft_event_radius1_value=cfg.soft_event_radius1_value,
    soft_event_radius2_value=cfg.soft_event_radius2_value,
    device=device,
    stair_hard_negative_label_indices=stair_hard_negative_label_indices,
    label_diagnostic_names=FOOT_EVENT_LABEL_DIAGNOSTIC_NAMES,
  )
  val_buffer = OnlineFootEventReplayBuffer(
    capacity=cfg.val_buffer_capacity,
    history_len=cfg.history_len,
    obs_dim=obs_dim,
    label_dim=len(FOOT_EVENT_LABEL_NAMES),
    num_envs=cfg.num_envs,
    soft_touchdown_radius=cfg.soft_touchdown_radius,
    soft_toe_hit_radius=cfg.soft_toe_hit_radius,
    soft_event_radius1_value=cfg.soft_event_radius1_value,
    soft_event_radius2_value=cfg.soft_event_radius2_value,
    device=device,
    stair_hard_negative_label_indices=stair_hard_negative_label_indices,
    label_diagnostic_names=FOOT_EVENT_LABEL_DIAGNOSTIC_NAMES,
  )
  model = FootEventDetectorGRU(
    obs_dim=obs_dim,
    frame_hidden_dim=cfg.frame_hidden_dim,
    recurrent_hidden_dim=cfg.recurrent_hidden_dim,
    head_hidden_dim=cfg.head_hidden_dim,
    dropout=cfg.dropout,
  ).to(device)
  if cfg.init_detector_checkpoint is not None:
    _load_detector_checkpoint(
      model,
      cfg.init_detector_checkpoint,
      device=device,
    )
  train_label_indices: tuple[int, ...] | None = None
  if cfg.toe_only_finetune:
    _configure_toe_only_finetune(model)
    train_label_indices = (4, 5)
  if cfg.toe_riser_only_model:
    _configure_toe_riser_only_model(model)
    train_label_indices = (4, 5)
  if cfg.footprint_only_model:
    _configure_footprint_only_model(model)
    train_label_indices = (0, 1, 2, 3)
  trainable_parameters = [
    parameter for parameter in model.parameters() if parameter.requires_grad
  ]
  if not trainable_parameters:
    raise RuntimeError("No trainable detector parameters were configured.")
  optimizer_weight_decay = (
    0.0
    if (cfg.toe_only_finetune or cfg.toe_riser_only_model or cfg.footprint_only_model)
    else cfg.weight_decay
  )
  optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=cfg.learning_rate,
    weight_decay=optimizer_weight_decay,
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
  baseline_metrics = _load_baseline_metrics(cfg.baseline_metrics_file)
  higher_better = metric_is_higher_better(selection_metric)
  best_value = -float("inf") if higher_better else float("inf")
  best_step = 0
  best_state: dict[str, torch.Tensor] | None = None
  if baseline_metrics is not None and selection_metric in baseline_metrics:
    best_value = float(baseline_metrics[selection_metric])
  if cfg.init_detector_checkpoint is not None:
    best_state = copy.deepcopy(model.state_dict())
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
    f"init_detector={cfg.init_detector_checkpoint}",
    f"toe_only_finetune={cfg.toe_only_finetune}",
    f"toe_riser_only_model={cfg.toe_riser_only_model}",
    f"optimizer_weight_decay={optimizer_weight_decay}",
    f"baseline_metrics={cfg.baseline_metrics_file}",
    f"toe_hit_pos_weight={cfg.toe_hit_pos_weight}",
    f"input_schema={cfg.input_schema}",
    f"include_gait_phase={cfg.include_gait_phase}",
    f"obs_dim={obs_dim}",
    f"footprint_only_model={cfg.footprint_only_model}",
    f"toe_pos_fraction={cfg.toe_positive_fraction}",
    f"toe_soft_fraction={cfg.toe_soft_positive_fraction}",
    f"touchdown_pos_fraction={cfg.touchdown_positive_fraction}",
    f"touchdown_soft_fraction={cfg.touchdown_soft_positive_fraction}",
    f"fn_hard_pos_fraction={cfg.false_negative_hard_positive_fraction}",
    f"toe_fn_hard_pos_fraction={cfg.false_negative_toe_hard_positive_fraction}",
    f"recall_precision_floor={cfg.touchdown_recall_precision_floor}",
    f"toe_recall_precision_floor={cfg.toe_hit_recall_precision_floor}",
    f"min_stair_events={cfg.selection_min_stair_touchdown_events}",
    f"save_eval_checkpoints={cfg.save_eval_checkpoints}",
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
      input_schema=cfg.input_schema,
      include_gait_phase=cfg.include_gait_phase,
      gait_period=cfg.gait_period,
      command_name=cfg.command_name,
      reset_mask=torch.ones(cfg.num_envs, dtype=torch.bool, device=device),
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
        input_schema=cfg.input_schema,
        include_gait_phase=cfg.include_gait_phase,
        gait_period=cfg.gait_period,
        command_name=cfg.command_name,
        reset_mask=reset_mask,
      )
      history_buffer.push(latent, reset_mask)

      labels = foot_event_labels_from_env(
        raw_env,
        previous_contact=previous_contact,
        previous_contact_valid=previous_contact_valid,
      )
      label_diagnostics = foot_event_label_diagnostics_from_env(raw_env)
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
      footprint_anchor_w = _current_footprint_anchor_w(raw_env)
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
          footprint_anchor_w=footprint_anchor_w[ids],
          label_diagnostics=label_diagnostics[ids],
        )

      current_contact = labels[:, 0:2].bool()
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
            toe_soft_positive_fraction=cfg.toe_soft_positive_fraction,
            touchdown_positive_fraction=cfg.touchdown_positive_fraction,
            touchdown_soft_positive_fraction=cfg.touchdown_soft_positive_fraction,
            stair_hard_negative_fraction=cfg.stair_hard_negative_fraction,
            false_positive_hard_negative_fraction=(
              cfg.false_positive_hard_negative_fraction
            ),
            false_negative_hard_positive_fraction=(
              cfg.false_negative_hard_positive_fraction
            ),
            false_negative_toe_hard_positive_fraction=(
              cfg.false_negative_toe_hard_positive_fraction
            ),
            toe_soft_positive_threshold=cfg.toe_soft_positive_threshold,
            touchdown_soft_positive_threshold=cfg.touchdown_soft_positive_threshold,
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
            train_label_indices=train_label_indices,
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
        eval_step = rollout_step + 1
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
          touchdown_recall_precision_floor=cfg.touchdown_recall_precision_floor,
          toe_hit_recall_precision_floor=cfg.toe_hit_recall_precision_floor,
          selection_min_stair_touchdown_events=(
            cfg.selection_min_stair_touchdown_events
          ),
          train_label_indices=train_label_indices,
        )
        if selection_metric not in val_metrics:
          available = ", ".join(sorted(val_metrics))
          raise KeyError(
            f"selection_metric={selection_metric!r} not found in validation "
            f"metrics. Available metrics: {available}"
          )
        if cfg.save_eval_checkpoints:
          torch.save(model.state_dict(), eval_checkpoint_dir / f"step_{eval_step}.pt")
        mined_hard_negatives = 0
        if cfg.mine_false_positive_hard_negatives and not cfg.footprint_only_model:
          mined_hard_negatives = _mine_false_positive_hard_negatives(
            model,
            train_buffer,
            device=device,
            batch_size=cfg.batch_size,
            threshold=cfg.hard_negative_mining_threshold,
            window_frames=cfg.hard_negative_mining_window_frames,
            max_peaks=cfg.hard_negative_mining_max_peaks,
          )
        mined_touchdown_hard_negatives = 0
        if _should_mine_touchdown_false_positive_hard_negatives(cfg):
          mined_touchdown_hard_negatives = (
            _mine_false_positive_touchdown_hard_negatives(
              model,
              train_buffer,
              device=device,
              batch_size=cfg.batch_size,
              threshold=cfg.hard_touchdown_false_positive_mining_threshold,
              contact_threshold=(
                cfg.hard_touchdown_false_positive_mining_contact_threshold
              ),
              window_frames=cfg.hard_touchdown_false_positive_mining_window_frames,
              max_peaks=cfg.hard_touchdown_false_positive_mining_max_peaks,
            )
          )
        mined_hard_positives = 0
        if cfg.mine_false_negative_hard_positives:
          mined_hard_positives = _mine_false_negative_hard_positives(
            model,
            train_buffer,
            device=device,
            batch_size=cfg.batch_size,
            threshold=cfg.hard_positive_mining_threshold,
            window_frames=cfg.hard_positive_mining_window_frames,
            max_peaks=cfg.hard_positive_mining_max_peaks,
          )
        mined_toe_hard_positives = 0
        if cfg.mine_false_negative_toe_hard_positives and not cfg.footprint_only_model:
          mined_toe_hard_positives = _mine_false_negative_toe_hard_positives(
            model,
            train_buffer,
            device=device,
            batch_size=cfg.batch_size,
            threshold=cfg.hard_toe_positive_mining_threshold,
            window_frames=cfg.hard_toe_positive_mining_window_frames,
            max_peaks=cfg.hard_toe_positive_mining_max_peaks,
          )
        guard_passed, guard_failures = _baseline_guard_passed(
          val_metrics,
          baseline_metrics,
          touchdown_recall_tolerance=cfg.baseline_touchdown_recall_tolerance,
          stair_touchdown_recall_tolerance=(
            cfg.baseline_stair_touchdown_recall_tolerance
          ),
          flat_touchdown_f1_tolerance=cfg.baseline_flat_touchdown_f1_tolerance,
          toe_metric=cfg.baseline_toe_metric,
          toe_metric_min_improvement=cfg.baseline_toe_metric_min_improvement,
          required_metric_names=cfg.baseline_required_metric_names,
          required_metric_min_improvement=(
            cfg.baseline_required_metric_min_improvement
          ),
        )
        metric_value = float(val_metrics[selection_metric])
        is_better = guard_passed and _metric_improved(
          metric_value=metric_value,
          best_value=best_value,
          higher_better=higher_better,
          min_delta=cfg.early_stop_min_delta,
        )
        if is_better:
          best_value = metric_value
          best_step = eval_step
          best_state = copy.deepcopy(model.state_dict())
          torch.save(best_state, output_dir / "best.pt")
          no_improve_evals = 0
        else:
          no_improve_evals += 1
        metrics_history.append(
          {
            "step": eval_step,
            "train_updates": train_updates,
            "train_samples": train_buffer.size,
            "val_samples": val_buffer.size,
            "mined_false_positive_hard_negatives": mined_hard_negatives,
            "mined_false_positive_touchdown_hard_negatives": (
              mined_touchdown_hard_negatives
            ),
            "mined_false_negative_hard_positives": mined_hard_positives,
            "mined_false_negative_toe_hard_positives": mined_toe_hard_positives,
            "baseline_guard_passed": guard_passed,
            "baseline_guard_failures": list(guard_failures),
            "val": val_metrics,
          }
        )
        print(
          f"[Stage 2D] step={eval_step}",
          f"updates={train_updates}",
          f"val_loss={val_metrics['val_loss']:.5f}",
          f"event_macro_f1={val_metrics['event_macro_f1']:.4f}",
          f"deploy_event_macro_f1={val_metrics['deploy_event_macro_f1']:.4f}",
          f"footprint_score={val_metrics['footprint_deploy_score']:.4f}",
          f"high_recall_score={val_metrics['high_recall_footprint_score']:.4f}",
          f"timing_guarded_score={val_metrics['timing_guarded_footprint_score']:.4f}",
          f"td_timing_score={val_metrics['touchdown_timing_guarded_score']:.4f}",
          f"td_dt_p90={val_metrics['touchdown_timing_abs_dt_frames_p90']:.1f}",
          f"td_xy_p90={val_metrics.get('touchdown_footprint_xy_error_m_p90', 0.0):.3f}",
          f"toe_guarded_score={val_metrics['toe_guarded_footprint_score']:.4f}",
          f"td_deploy_f1={val_metrics['touchdown_deploy_macro_f1']:.4f}",
          f"td_high_recall={val_metrics['touchdown_high_recall_macro_recall']:.4f}",
          f"td_stair_events={val_metrics['touchdown_stair_deploy_true_event_count']:.0f}",
          "td_stair_high_recall="
          f"{val_metrics['touchdown_stair_high_recall_macro_recall']:.4f}",
          f"td_fallback_recall={val_metrics['touchdown_contact_fallback_macro_recall']:.4f}",
          f"toe_deploy_f1={val_metrics['toe_riser_deploy_macro_f1']:.4f}",
          f"toe_high_recall={val_metrics['toe_riser_high_recall_macro_recall']:.4f}",
          f"toe_high_precision={val_metrics['toe_riser_high_recall_macro_precision']:.4f}",
          f"mined_fp_hard={mined_hard_negatives}",
          f"mined_td_fp_hard={mined_touchdown_hard_negatives}",
          f"mined_fn_hard={mined_hard_positives}",
          f"mined_toe_fn_hard={mined_toe_hard_positives}",
          f"baseline_guard={'pass' if guard_passed else 'fail'}",
          f"guard_failures={';'.join(guard_failures[:2]) if guard_failures else '-'}",
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

  if cfg.save_last_checkpoint:
    torch.save(model.state_dict(), last_checkpoint_path)
  if best_state is None:
    if baseline_metrics is not None and cfg.require_baseline_guard:
      last_failures: list[str] = []
      if metrics_history:
        last_failures = list(metrics_history[-1].get("baseline_guard_failures", []))
      failure_text = "; ".join(last_failures[:8]) if last_failures else "no eval passed"
      raise RuntimeError(
        "No checkpoint satisfied the required baseline guard. "
        f"Last failures: {failure_text}"
      )
    best_state = copy.deepcopy(model.state_dict())
    best_step = completed_steps
    best_value = 0.0
  torch.save(best_state, output_dir / "best.pt")
  model.load_state_dict(best_state)
  if cfg.export_onnx:
    export_detector_onnx(
      model,
      output_dir / "best.onnx",
      history_len=cfg.history_len,
      obs_dim=obs_dim,
    )
  _write_label_audit(output_dir, train_buffer, val_buffer)
  best_val_metrics: dict[str, float] = {}
  if best_step == 0 and baseline_metrics is not None:
    best_val_metrics = dict(baseline_metrics)
  else:
    for record in metrics_history:
      if int(record["step"]) == best_step:
        best_val_metrics = {
          key: float(value) for key, value in dict(record["val"]).items()
        }
        break
  best_deployment_thresholds = {
    key: value
    for key, value in best_val_metrics.items()
    if key.endswith(("_deploy_best_threshold", "_deploy_high_recall_threshold"))
  }
  onnx_path = output_dir / "best.onnx" if cfg.export_onnx else None
  deployment_contract = _deployment_contract_payload(
    task_id=task_id,
    cfg=cfg,
    obs_dim=obs_dim,
    trained_label_indices=train_label_indices,
    stair_hard_negative_label_indices=stair_hard_negative_label_indices,
    best_deployment_thresholds=best_deployment_thresholds,
    onnx_path=onnx_path,
  )
  with (output_dir / "deployment_contract.json").open("w", encoding="utf-8") as stream:
    json.dump(deployment_contract, stream, indent=2, sort_keys=True)
    stream.write("\n")

  payload: dict[str, Any] = {
    "config": asdict(cfg),
    "task_id": task_id,
    "checkpoint_path": str(checkpoint_path),
    "label_names": list(FOOT_EVENT_LABEL_NAMES),
    "label_diagnostic_names": list(FOOT_EVENT_LABEL_DIAGNOSTIC_NAMES),
    "stair_hard_negative_label_indices": list(stair_hard_negative_label_indices),
    "input_feature_groups": foot_event_input_feature_groups(
      include_gait_phase=cfg.include_gait_phase,
      input_schema=cfg.input_schema,
    ),
    "input_feature_scale_groups": foot_event_input_feature_scale_groups(
      include_gait_phase=cfg.include_gait_phase,
      input_schema=cfg.input_schema,
    ),
    "input_feature_scales": foot_event_input_feature_scales(
      include_gait_phase=cfg.include_gait_phase,
      input_schema=cfg.input_schema,
    ),
    "history": metrics_history,
    "best_step": best_step,
    "completed_steps": completed_steps,
    "stop_reason": stop_reason,
    "selection_metric": selection_metric,
    "best_metric_value": best_value,
    "best_checkpoint_path": str(output_dir / "best.pt"),
    "eval_checkpoint_dir": str(eval_checkpoint_dir)
    if cfg.save_eval_checkpoints
    else None,
    "last_checkpoint_path": str(last_checkpoint_path)
    if cfg.save_last_checkpoint
    else None,
    "best_deployment_thresholds": best_deployment_thresholds,
    "deployment_threshold_grid": [
      float(value) for value in deployment_thresholds.astype(np.float32).tolist()
    ],
    "train_updates": train_updates,
    "no_improve_evals": no_improve_evals,
    "train_samples_in_buffer": train_buffer.size,
    "val_samples_in_buffer": val_buffer.size,
    "pos_weight": [float(value) for value in pos_weight.detach().cpu().tolist()],
    "trained_label_indices": list(train_label_indices)
    if train_label_indices is not None
    else None,
    "baseline_guard": {
      "metrics_file": cfg.baseline_metrics_file,
      "baseline_metrics": baseline_metrics,
      "touchdown_recall_tolerance": cfg.baseline_touchdown_recall_tolerance,
      "stair_touchdown_recall_tolerance": (
        cfg.baseline_stair_touchdown_recall_tolerance
      ),
      "flat_touchdown_f1_tolerance": cfg.baseline_flat_touchdown_f1_tolerance,
      "toe_metric": cfg.baseline_toe_metric,
      "toe_metric_min_improvement": cfg.baseline_toe_metric_min_improvement,
      "required_metric_names": list(cfg.baseline_required_metric_names),
      "required_metric_min_improvement": (cfg.baseline_required_metric_min_improvement),
      "require_baseline_guard": cfg.require_baseline_guard,
    },
    "model": {
      "type": "FootEventDetectorGRU",
      "obs_dim": obs_dim,
      "history_len": cfg.history_len,
      "output_dim": len(FOOT_EVENT_LABEL_NAMES),
      "toe_riser_only_output_layout": bool(cfg.toe_riser_only_model),
      "footprint_only_output_layout": bool(cfg.footprint_only_model),
      "frame_hidden_dim": cfg.frame_hidden_dim,
      "recurrent_hidden_dim": cfg.recurrent_hidden_dim,
      "head_hidden_dim": cfg.head_hidden_dim,
      "trainable_parameters": int(
        sum(parameter.numel() for parameter in model.parameters())
      ),
    },
    "onnx_path": str(onnx_path) if onnx_path is not None else None,
    "deployment_contract_path": str(output_dir / "deployment_contract.json"),
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

  default_task_id = (
    "Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1"
  )
  task_choices = tuple(list_tasks())
  remaining_args = sys.argv[1:]
  task_id = default_task_id
  if remaining_args and not remaining_args[0].startswith("-"):
    task_id = remaining_args[0]
    remaining_args = remaining_args[1:]
  if task_id not in task_choices:
    choices = ", ".join(task_choices)
    raise ValueError(f"Unknown task_id {task_id!r}. Available tasks: {choices}")
  cfg = tyro.cli(OnlineFootEventDetectorConfig, args=remaining_args)
  run_online_train(task_id, cfg)


if __name__ == "__main__":
  main()
