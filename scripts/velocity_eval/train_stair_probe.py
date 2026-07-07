"""Train an offline Stage 2B probe on exported stair latent histories."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

DEPTH_MIN_M = 0.25
DEPTH_MAX_M = 0.35
NUM_LEVEL_DELTA_CLASSES = 4
NUM_RELATIVE_LEVEL_CLASSES = 8
NUM_DEPTH_BIN_CLASSES = 8
NUM_DEPTH_GROUP_CLASSES = 3
DEPTH_BIN_TO_GROUP = np.asarray([0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
FOOTPRINT_INPUT_MODES = (
  "footprint_only",
  "latent_plus_footprint",
  "two_branch_fusion",
)
SPARSE_EVENT_INPUT_MODES = (
  "sparse_event_only",
  "latent_plus_sparse_event",
  "two_branch_sparse_event",
)
TWO_BRANCH_INPUT_MODES = ("two_branch_fusion", "two_branch_sparse_event")
SAFE_HEAD_OUTPUT_DIM = 6
DIAGNOSTIC_OUTPUT_DIM = 22 + SAFE_HEAD_OUTPUT_DIM


@dataclass(frozen=True)
class TrainStairProbeConfig:
  """Configuration for the Stage 2B offline latent-history probe."""

  dataset_file: str = "eval_outputs/stair_stage2/model51000_seed42_probe_v1/samples.npz"
  output_dir: str = "eval_outputs/stair_stage2/model51000_seed42_probe_gru_v1"
  device: str | None = None
  seed: int = 12345
  history_len: int = 64
  footprint_history_len: int = 128
  sparse_event_memory_len: int = 8
  input_mode: Literal[
    "latent_only",
    "footprint_only",
    "latent_plus_footprint",
    "two_branch_fusion",
    "sparse_event_only",
    "latent_plus_sparse_event",
    "two_branch_sparse_event",
  ] = "latent_only"
  model: Literal["gru"] = "gru"
  objective: Literal["multitask", "depth_only", "safe_landing"] = "multitask"
  selection_metric: Literal[
    "auto",
    "val_loss",
    "level_delta_macro_f1",
    "level_delta_transition_f1",
    "depth_bin_macro_f1",
    "depth_3group_macro_f1",
    "depth_mae_m",
    "safe_stride_center_mae_m",
    "landing_quality_mae",
    "touchdown_f1",
  ] = "auto"
  obs_dim: int = 91
  frame_hidden_dim: int = 128
  recurrent_hidden_dim: int = 128
  footprint_frame_hidden_dim: int = 128
  footprint_recurrent_hidden_dim: int = 128
  fusion_hidden_dim: int = 128
  probe_latent_dim: int = 16
  head_hidden_dim: int = 64
  dropout: float = 0.0
  batch_size: int = 512
  epochs: int = 20
  learning_rate: float = 1.0e-3
  weight_decay: float = 1.0e-4
  val_fraction: float = 0.2
  max_samples: int | None = None
  num_workers: int = 0
  active_loss_coef: float = 0.5
  level_delta_loss_coef: float = 1.0
  relative_level_loss_coef: float = 0.5
  depth_bin_loss_coef: float = 1.0
  depth_reg_loss_coef: float = 1.0
  depth_huber_beta: float = 0.05
  touchdown_loss_coef: float = 1.0
  landing_quality_loss_coef: float = 1.0
  collision_risk_loss_coef: float = 0.5
  safe_stride_loss_coef: float = 1.0
  safe_stride_huber_beta: float = 0.05
  max_prediction_rows: int = 5000
  progress: bool = True


ObjectiveName = Literal["multitask", "depth_only", "safe_landing"]
InputMode = Literal[
  "latent_only",
  "footprint_only",
  "latent_plus_footprint",
  "two_branch_fusion",
  "sparse_event_only",
  "latent_plus_sparse_event",
  "two_branch_sparse_event",
]
MetricName = Literal[
  "val_loss",
  "level_delta_macro_f1",
  "level_delta_transition_f1",
  "depth_bin_macro_f1",
  "depth_3group_macro_f1",
  "depth_mae_m",
  "safe_stride_center_mae_m",
  "landing_quality_mae",
  "touchdown_f1",
]


@dataclass(frozen=True)
class StairProbeArrays:
  """NPZ arrays needed by the offline probe trainer."""

  obs_history: np.ndarray
  obs_valid_mask: np.ndarray
  level_delta_label: np.ndarray
  relative_level_label: np.ndarray
  true_tread_depth: np.ndarray
  depth_bin_label: np.ndarray
  depth_valid_label: np.ndarray
  stair_active_label: np.ndarray
  sample_type: np.ndarray
  sequence_id: np.ndarray
  env_id: np.ndarray
  frame_idx: np.ndarray
  seed: np.ndarray
  true_riser_height: np.ndarray | None = None
  safe_landing_center: np.ndarray | None = None
  minimum_safe_stride: np.ndarray | None = None
  maximum_safe_stride: np.ndarray | None = None
  safe_stride_valid_label: np.ndarray | None = None
  landing_touchdown_label: np.ndarray | None = None
  landing_quality_label: np.ndarray | None = None
  collision_risk_label: np.ndarray | None = None
  privileged_footprint_history: np.ndarray | None = None
  privileged_footprint_valid_mask: np.ndarray | None = None
  sparse_foot_event_memory: np.ndarray | None = None
  sparse_foot_event_valid_mask: np.ndarray | None = None


@dataclass(frozen=True)
class ProbeSplit:
  """Train/validation index split with sequence-level leakage protection."""

  train_indices: np.ndarray
  val_indices: np.ndarray
  train_sequence_ids: np.ndarray
  val_sequence_ids: np.ndarray


@dataclass(frozen=True)
class ProbeLossWeights:
  """Class weights computed from the training split."""

  active_pos_weight: torch.Tensor
  level_delta: torch.Tensor
  relative_level: torch.Tensor
  depth_bin: torch.Tensor


class StairProbeGRU(nn.Module):
  """GRU history encoder with a 16D SlowLatent-sized probe bottleneck."""

  def __init__(
    self,
    *,
    obs_dim: int = 91,
    frame_hidden_dim: int = 128,
    recurrent_hidden_dim: int = 128,
    probe_latent_dim: int = 16,
    head_hidden_dim: int = 64,
    dropout: float = 0.0,
  ) -> None:
    super().__init__()
    self.obs_dim = int(obs_dim)
    self.probe_latent_dim = int(probe_latent_dim)
    self.frame_encoder = nn.Sequential(
      nn.Linear(obs_dim, frame_hidden_dim),
      nn.ELU(),
      nn.Dropout(dropout),
      nn.Linear(frame_hidden_dim, frame_hidden_dim),
      nn.ELU(),
    )
    self.gru = nn.GRU(
      input_size=frame_hidden_dim,
      hidden_size=recurrent_hidden_dim,
      batch_first=True,
    )
    self.probe_latent = nn.Linear(recurrent_hidden_dim, probe_latent_dim)
    self.stair_active_head = self._make_head(probe_latent_dim, head_hidden_dim, 1)
    self.level_delta_head = self._make_head(
      probe_latent_dim,
      head_hidden_dim,
      NUM_LEVEL_DELTA_CLASSES,
    )
    self.relative_level_head = self._make_head(
      probe_latent_dim,
      head_hidden_dim,
      NUM_RELATIVE_LEVEL_CLASSES,
    )
    self.depth_bin_head = self._make_head(
      probe_latent_dim,
      head_hidden_dim,
      NUM_DEPTH_BIN_CLASSES,
    )
    self.depth_reg_head = self._make_head(probe_latent_dim, head_hidden_dim, 1)
    self.touchdown_head = self._make_head(probe_latent_dim, head_hidden_dim, 1)
    self.landing_quality_head = self._make_head(probe_latent_dim, head_hidden_dim, 1)
    self.collision_risk_head = self._make_head(probe_latent_dim, head_hidden_dim, 1)
    self.safe_stride_head = self._make_head(probe_latent_dim, head_hidden_dim, 3)

  @staticmethod
  def _make_head(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
      nn.Linear(in_dim, hidden_dim),
      nn.ELU(),
      nn.Linear(hidden_dim, out_dim),
    )

  def encode_latent(self, obs_history: torch.Tensor) -> torch.Tensor:
    """Encode ``(B, T, 91)`` histories into a 16D probe latent."""
    if obs_history.dim() != 3:
      raise ValueError("obs_history must have shape (batch, history, obs_dim).")
    if obs_history.shape[-1] != self.obs_dim:
      raise ValueError(f"Expected obs_dim={self.obs_dim}, got {obs_history.shape[-1]}.")
    encoded = self.frame_encoder(obs_history)
    _seq, hidden = self.gru(encoded)
    return self.probe_latent(hidden[-1])

  def forward(self, obs_history: torch.Tensor) -> dict[str, torch.Tensor]:
    z = self.encode_latent(obs_history)
    return {
      "probe_latent": z,
      "stair_active_logit": self.stair_active_head(z),
      "level_delta_logits": self.level_delta_head(z),
      "relative_level_logits": self.relative_level_head(z),
      "depth_bin_logits": self.depth_bin_head(z),
      "depth_reg": self.depth_reg_head(z),
      "touchdown_logit": self.touchdown_head(z),
      "landing_quality": self.landing_quality_head(z),
      "collision_risk": self.collision_risk_head(z),
      "safe_stride": self.safe_stride_head(z),
    }


class TwoBranchStairProbeGRU(nn.Module):
  """Separate deployable-observation and privileged-footprint history encoders."""

  def __init__(
    self,
    *,
    latent_obs_dim: int = 91,
    footprint_obs_dim: int = 18,
    frame_hidden_dim: int = 128,
    recurrent_hidden_dim: int = 128,
    footprint_frame_hidden_dim: int = 128,
    footprint_recurrent_hidden_dim: int = 128,
    fusion_hidden_dim: int = 128,
    probe_latent_dim: int = 16,
    head_hidden_dim: int = 64,
    dropout: float = 0.0,
  ) -> None:
    super().__init__()
    self.latent_obs_dim = int(latent_obs_dim)
    self.footprint_obs_dim = int(footprint_obs_dim)
    self.probe_latent_dim = int(probe_latent_dim)
    self.latent_frame_encoder = nn.Sequential(
      nn.Linear(latent_obs_dim, frame_hidden_dim),
      nn.ELU(),
      nn.Dropout(dropout),
      nn.Linear(frame_hidden_dim, frame_hidden_dim),
      nn.ELU(),
    )
    self.latent_gru = nn.GRU(
      input_size=frame_hidden_dim,
      hidden_size=recurrent_hidden_dim,
      batch_first=True,
    )
    self.footprint_frame_encoder = nn.Sequential(
      nn.Linear(footprint_obs_dim, footprint_frame_hidden_dim),
      nn.ELU(),
      nn.Dropout(dropout),
      nn.Linear(footprint_frame_hidden_dim, footprint_frame_hidden_dim),
      nn.ELU(),
    )
    self.footprint_gru = nn.GRU(
      input_size=footprint_frame_hidden_dim,
      hidden_size=footprint_recurrent_hidden_dim,
      batch_first=True,
    )
    self.probe_latent = nn.Sequential(
      nn.Linear(
        recurrent_hidden_dim + footprint_recurrent_hidden_dim,
        fusion_hidden_dim,
      ),
      nn.ELU(),
      nn.Dropout(dropout),
      nn.Linear(fusion_hidden_dim, probe_latent_dim),
    )
    self.stair_active_head = StairProbeGRU._make_head(
      probe_latent_dim,
      head_hidden_dim,
      1,
    )
    self.level_delta_head = StairProbeGRU._make_head(
      probe_latent_dim,
      head_hidden_dim,
      NUM_LEVEL_DELTA_CLASSES,
    )
    self.relative_level_head = StairProbeGRU._make_head(
      probe_latent_dim,
      head_hidden_dim,
      NUM_RELATIVE_LEVEL_CLASSES,
    )
    self.depth_bin_head = StairProbeGRU._make_head(
      probe_latent_dim,
      head_hidden_dim,
      NUM_DEPTH_BIN_CLASSES,
    )
    self.depth_reg_head = StairProbeGRU._make_head(
      probe_latent_dim,
      head_hidden_dim,
      1,
    )
    self.touchdown_head = StairProbeGRU._make_head(
      probe_latent_dim,
      head_hidden_dim,
      1,
    )
    self.landing_quality_head = StairProbeGRU._make_head(
      probe_latent_dim,
      head_hidden_dim,
      1,
    )
    self.collision_risk_head = StairProbeGRU._make_head(
      probe_latent_dim,
      head_hidden_dim,
      1,
    )
    self.safe_stride_head = StairProbeGRU._make_head(
      probe_latent_dim,
      head_hidden_dim,
      3,
    )

  def encode_latent(
    self,
    obs_history: torch.Tensor,
    footprint_history: torch.Tensor,
  ) -> torch.Tensor:
    """Encode paired histories into the probe latent."""
    if obs_history.dim() != 3:
      raise ValueError("obs_history must have shape (batch, history, obs_dim).")
    if footprint_history.dim() != 3:
      raise ValueError(
        "footprint_history must have shape (batch, history, footprint_dim)."
      )
    if obs_history.shape[-1] != self.latent_obs_dim:
      raise ValueError(
        f"Expected latent_obs_dim={self.latent_obs_dim}, got {obs_history.shape[-1]}."
      )
    if footprint_history.shape[-1] != self.footprint_obs_dim:
      raise ValueError(
        f"Expected footprint_obs_dim={self.footprint_obs_dim}, "
        f"got {footprint_history.shape[-1]}."
      )
    latent_encoded = self.latent_frame_encoder(obs_history)
    _latent_seq, latent_hidden = self.latent_gru(latent_encoded)
    footprint_encoded = self.footprint_frame_encoder(footprint_history)
    _footprint_seq, footprint_hidden = self.footprint_gru(footprint_encoded)
    fused = torch.cat([latent_hidden[-1], footprint_hidden[-1]], dim=-1)
    return self.probe_latent(fused)

  def forward(
    self,
    obs_history: torch.Tensor,
    footprint_history: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    z = self.encode_latent(obs_history, footprint_history)
    return {
      "probe_latent": z,
      "stair_active_logit": self.stair_active_head(z),
      "level_delta_logits": self.level_delta_head(z),
      "relative_level_logits": self.relative_level_head(z),
      "depth_bin_logits": self.depth_bin_head(z),
      "depth_reg": self.depth_reg_head(z),
      "touchdown_logit": self.touchdown_head(z),
      "landing_quality": self.landing_quality_head(z),
      "collision_risk": self.collision_risk_head(z),
      "safe_stride": self.safe_stride_head(z),
    }


class StairProbeTorchDataset(Dataset):
  """Torch dataset view over Stage 2A NPZ arrays."""

  def __init__(
    self,
    arrays: StairProbeArrays,
    indices: np.ndarray,
    *,
    history_len: int,
    footprint_history_len: int = 128,
    sparse_event_memory_len: int = 8,
    input_mode: InputMode = "latent_only",
  ) -> None:
    if history_len <= 0:
      raise ValueError("history_len must be positive.")
    if history_len > arrays.obs_history.shape[1]:
      raise ValueError(
        f"history_len must be in [1, {arrays.obs_history.shape[1]}], got {history_len}."
      )
    if input_mode not in (
      "latent_only",
      "footprint_only",
      "latent_plus_footprint",
      "two_branch_fusion",
      "sparse_event_only",
      "latent_plus_sparse_event",
      "two_branch_sparse_event",
    ):
      raise ValueError(f"Unsupported input_mode '{input_mode}'.")
    if input_mode in FOOTPRINT_INPUT_MODES:
      if arrays.privileged_footprint_history is None:
        raise ValueError(
          f"input_mode={input_mode!r} requires privileged_footprint_history."
        )
      if footprint_history_len <= 0:
        raise ValueError("footprint_history_len must be positive.")
      if footprint_history_len > arrays.privileged_footprint_history.shape[1]:
        raise ValueError(
          f"footprint_history_len={footprint_history_len} exceeds dataset "
          f"footprint history length {arrays.privileged_footprint_history.shape[1]}."
        )
    if input_mode in SPARSE_EVENT_INPUT_MODES:
      if arrays.sparse_foot_event_memory is None:
        raise ValueError(
          f"input_mode={input_mode!r} requires sparse_foot_event_memory."
        )
      if sparse_event_memory_len <= 0:
        raise ValueError("sparse_event_memory_len must be positive.")
      if sparse_event_memory_len > arrays.sparse_foot_event_memory.shape[1]:
        raise ValueError(
          f"sparse_event_memory_len={sparse_event_memory_len} exceeds dataset "
          f"sparse event memory length {arrays.sparse_foot_event_memory.shape[1]}."
        )
    self.arrays = arrays
    self.indices = indices.astype(np.int64, copy=True)
    self.history_len = int(history_len)
    self.footprint_history_len = int(footprint_history_len)
    self.sparse_event_memory_len = int(sparse_event_memory_len)
    self.input_mode = input_mode
    self.input_history_len = self._resolve_input_history_len()
    self.input_dim = self._resolve_input_dim()

  def _resolve_input_history_len(self) -> int:
    if self.input_mode == "latent_only":
      return self.history_len
    if self.input_mode == "footprint_only":
      return self.footprint_history_len
    if self.input_mode == "sparse_event_only":
      return self.sparse_event_memory_len
    if self.input_mode in ("latent_plus_sparse_event", "two_branch_sparse_event"):
      return max(self.history_len, self.sparse_event_memory_len)
    return max(self.history_len, self.footprint_history_len)

  def _resolve_input_dim(self) -> int:
    latent_dim = int(self.arrays.obs_history.shape[-1])
    footprint = self.arrays.privileged_footprint_history
    footprint_dim = 0 if footprint is None else int(footprint.shape[-1])
    sparse_events = self.arrays.sparse_foot_event_memory
    sparse_dim = 0 if sparse_events is None else int(sparse_events.shape[-1])
    if self.input_mode == "latent_only":
      return latent_dim
    if self.input_mode == "footprint_only":
      return footprint_dim
    if self.input_mode == "sparse_event_only":
      return sparse_dim
    if self.input_mode in ("latent_plus_sparse_event", "two_branch_sparse_event"):
      return latent_dim + sparse_dim
    return latent_dim + footprint_dim

  def __len__(self) -> int:
    return int(self.indices.shape[0])

  def _right_aligned_history(
    self,
    history: np.ndarray,
    requested_len: int,
    *,
    output_len: int | None = None,
  ) -> np.ndarray:
    selected = history[-requested_len:, :]
    resolved_output_len = self.input_history_len if output_len is None else output_len
    if selected.shape[0] == resolved_output_len:
      return selected.astype(np.float32, copy=False)
    output = np.zeros((resolved_output_len, history.shape[-1]), dtype=np.float32)
    output[-selected.shape[0] :, :] = selected.astype(np.float32, copy=False)
    return output

  def _sparse_event_history(
    self,
    memory: np.ndarray,
    requested_len: int,
    *,
    output_len: int | None = None,
  ) -> np.ndarray:
    selected = memory[:requested_len, :]
    resolved_output_len = self.input_history_len if output_len is None else output_len
    if selected.shape[0] == resolved_output_len:
      return selected.astype(np.float32, copy=False)
    output = np.zeros((resolved_output_len, memory.shape[-1]), dtype=np.float32)
    output[: selected.shape[0], :] = selected.astype(np.float32, copy=False)
    return output

  def _model_input_history(self, sample_index: int) -> np.ndarray:
    arrays = self.arrays
    if self.input_mode == "latent_only":
      return self._right_aligned_history(
        arrays.obs_history[sample_index],
        self.history_len,
      )
    sparse_events = arrays.sparse_foot_event_memory
    if self.input_mode == "sparse_event_only":
      if sparse_events is None:
        raise RuntimeError("sparse foot-event memory is unavailable.")
      return self._sparse_event_history(
        sparse_events[sample_index],
        self.sparse_event_memory_len,
      )
    footprint = arrays.privileged_footprint_history
    latent_input = self._right_aligned_history(
      arrays.obs_history[sample_index],
      self.history_len,
    )
    if self.input_mode in ("latent_plus_sparse_event", "two_branch_sparse_event"):
      if sparse_events is None:
        raise RuntimeError("sparse foot-event memory is unavailable.")
      sparse_input = self._sparse_event_history(
        sparse_events[sample_index],
        self.sparse_event_memory_len,
      )
      return np.concatenate([latent_input, sparse_input], axis=-1)
    if footprint is None:
      raise RuntimeError("privileged footprint history is unavailable.")
    footprint_input = self._right_aligned_history(
      footprint[sample_index],
      self.footprint_history_len,
    )
    if self.input_mode == "footprint_only":
      return footprint_input
    return np.concatenate([latent_input, footprint_input], axis=-1)

  def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
    sample_index = int(self.indices[index])
    arrays = self.arrays
    depth = float(arrays.true_tread_depth[sample_index])
    depth_norm = (depth - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M)
    if self.input_mode in TWO_BRANCH_INPUT_MODES:
      footprint = arrays.privileged_footprint_history
      sparse_events = arrays.sparse_foot_event_memory
      if self.input_mode == "two_branch_fusion":
        if footprint is None:
          raise RuntimeError("privileged footprint history is unavailable.")
        secondary_key = "privileged_footprint_history"
        secondary_history = footprint[sample_index]
        secondary_len = self.footprint_history_len
      else:
        if sparse_events is None:
          raise RuntimeError("sparse foot-event memory is unavailable.")
        secondary_key = "sparse_foot_event_memory"
        secondary_history = sparse_events[sample_index]
        secondary_len = self.sparse_event_memory_len
      payload = {
        "obs_history": torch.as_tensor(
          self._right_aligned_history(
            arrays.obs_history[sample_index],
            self.history_len,
            output_len=self.history_len,
          ),
          dtype=torch.float32,
        ),
        secondary_key: torch.as_tensor(
          (
            self._right_aligned_history
            if self.input_mode == "two_branch_fusion"
            else self._sparse_event_history
          )(
            secondary_history,
            secondary_len,
            output_len=secondary_len,
          ),
          dtype=torch.float32,
        ),
      }
    else:
      payload = {
        "obs_history": torch.as_tensor(
          self._model_input_history(sample_index),
          dtype=torch.float32,
        ),
      }
    payload.update(
      {
        "sample_index": torch.tensor(sample_index, dtype=torch.long),
        "stair_active": torch.tensor(
          bool(arrays.stair_active_label[sample_index]), dtype=torch.float32
        ),
        "level_delta": torch.tensor(
          int(arrays.level_delta_label[sample_index]), dtype=torch.long
        ),
        "relative_level": torch.tensor(
          int(np.clip(arrays.relative_level_label[sample_index], 0, 7)),
          dtype=torch.long,
        ),
        "depth_bin": torch.tensor(
          int(max(0, arrays.depth_bin_label[sample_index])), dtype=torch.long
        ),
        "depth_valid": torch.tensor(
          bool(arrays.depth_valid_label[sample_index]), dtype=torch.bool
        ),
        "depth_norm": torch.tensor(depth_norm, dtype=torch.float32),
        "true_tread_depth": torch.tensor(depth, dtype=torch.float32),
        "true_riser_height": torch.tensor(
          0.0
          if arrays.true_riser_height is None
          else float(arrays.true_riser_height[sample_index]),
          dtype=torch.float32,
        ),
        "safe_landing_center": torch.tensor(
          0.0
          if arrays.safe_landing_center is None
          else float(arrays.safe_landing_center[sample_index]),
          dtype=torch.float32,
        ),
        "minimum_safe_stride": torch.tensor(
          0.0
          if arrays.minimum_safe_stride is None
          else float(arrays.minimum_safe_stride[sample_index]),
          dtype=torch.float32,
        ),
        "maximum_safe_stride": torch.tensor(
          0.0
          if arrays.maximum_safe_stride is None
          else float(arrays.maximum_safe_stride[sample_index]),
          dtype=torch.float32,
        ),
        "safe_stride_valid": torch.tensor(
          False
          if arrays.safe_stride_valid_label is None
          else bool(arrays.safe_stride_valid_label[sample_index]),
          dtype=torch.bool,
        ),
        "landing_touchdown": torch.tensor(
          False
          if arrays.landing_touchdown_label is None
          else bool(arrays.landing_touchdown_label[sample_index]),
          dtype=torch.float32,
        ),
        "landing_quality": torch.tensor(
          0.0
          if arrays.landing_quality_label is None
          else float(arrays.landing_quality_label[sample_index]),
          dtype=torch.float32,
        ),
        "collision_risk": torch.tensor(
          0.0
          if arrays.collision_risk_label is None
          else float(arrays.collision_risk_label[sample_index]),
          dtype=torch.float32,
        ),
        "sample_type": torch.tensor(
          int(arrays.sample_type[sample_index]), dtype=torch.long
        ),
        "sequence_id": torch.tensor(
          int(arrays.sequence_id[sample_index]), dtype=torch.long
        ),
        "env_id": torch.tensor(int(arrays.env_id[sample_index]), dtype=torch.long),
        "frame_idx": torch.tensor(
          int(arrays.frame_idx[sample_index]), dtype=torch.long
        ),
      }
    )
    return payload


def set_deterministic_seed(seed: int) -> None:
  """Seed Python, NumPy, and Torch for repeatable probe runs."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def load_stair_probe_arrays(
  dataset_file: str | Path,
  *,
  max_samples: int | None = None,
  seed: int = 0,
) -> StairProbeArrays:
  """Load required Stage 2A NPZ arrays, optionally with stratified subsampling."""
  path = Path(dataset_file).expanduser()
  with np.load(path, allow_pickle=False) as data:
    arrays = StairProbeArrays(
      obs_history=np.asarray(data["obs_history"], dtype=np.float32),
      obs_valid_mask=np.asarray(data["obs_valid_mask"], dtype=np.bool_),
      level_delta_label=np.asarray(data["level_delta_label"], dtype=np.int64),
      relative_level_label=np.asarray(data["relative_level_label"], dtype=np.int64),
      true_tread_depth=np.asarray(data["true_tread_depth"], dtype=np.float32),
      depth_bin_label=np.asarray(data["depth_bin_label"], dtype=np.int64),
      depth_valid_label=np.asarray(data["depth_valid_label"], dtype=np.bool_),
      stair_active_label=np.asarray(data["stair_active_label"], dtype=np.bool_),
      sample_type=np.asarray(data["sample_type"], dtype=np.int64),
      sequence_id=np.asarray(data["sequence_id"], dtype=np.int64),
      env_id=np.asarray(data["env_id"], dtype=np.int64),
      frame_idx=np.asarray(data["frame_idx"], dtype=np.int64),
      seed=np.asarray(data["seed"], dtype=np.int64),
      true_riser_height=(
        np.asarray(data["true_riser_height"], dtype=np.float32)
        if "true_riser_height" in data
        else None
      ),
      safe_landing_center=(
        np.asarray(data["safe_landing_center"], dtype=np.float32)
        if "safe_landing_center" in data
        else None
      ),
      minimum_safe_stride=(
        np.asarray(data["minimum_safe_stride"], dtype=np.float32)
        if "minimum_safe_stride" in data
        else None
      ),
      maximum_safe_stride=(
        np.asarray(data["maximum_safe_stride"], dtype=np.float32)
        if "maximum_safe_stride" in data
        else None
      ),
      safe_stride_valid_label=(
        np.asarray(data["safe_stride_valid_label"], dtype=np.bool_)
        if "safe_stride_valid_label" in data
        else None
      ),
      landing_touchdown_label=(
        np.asarray(data["landing_touchdown_label"], dtype=np.bool_)
        if "landing_touchdown_label" in data
        else None
      ),
      landing_quality_label=(
        np.asarray(data["landing_quality_label"], dtype=np.float32)
        if "landing_quality_label" in data
        else None
      ),
      collision_risk_label=(
        np.asarray(data["collision_risk_label"], dtype=np.float32)
        if "collision_risk_label" in data
        else None
      ),
      privileged_footprint_history=(
        np.asarray(data["privileged_footprint_history"], dtype=np.float32)
        if "privileged_footprint_history" in data
        else None
      ),
      privileged_footprint_valid_mask=(
        np.asarray(data["privileged_footprint_valid_mask"], dtype=np.bool_)
        if "privileged_footprint_valid_mask" in data
        else None
      ),
      sparse_foot_event_memory=(
        np.asarray(data["sparse_foot_event_memory"], dtype=np.float32)
        if "sparse_foot_event_memory" in data
        else None
      ),
      sparse_foot_event_valid_mask=(
        np.asarray(data["sparse_foot_event_valid_mask"], dtype=np.bool_)
        if "sparse_foot_event_valid_mask" in data
        else None
      ),
    )
  validate_arrays(arrays)
  if max_samples is None or arrays.obs_history.shape[0] <= max_samples:
    return arrays
  indices = stratified_sample_indices(arrays.sample_type, max_samples, seed)
  return subset_arrays(arrays, indices)


def validate_arrays(arrays: StairProbeArrays) -> None:
  """Validate the Stage 2A schema expected by the Stage 2B trainer."""
  num_samples = arrays.obs_history.shape[0]
  if arrays.obs_history.ndim != 3:
    raise ValueError("obs_history must have shape (N, history_len, obs_dim).")
  for field in fields(arrays):
    name = field.name
    value = getattr(arrays, name)
    if value is None:
      continue
    if value.shape[0] != num_samples:
      raise ValueError(
        f"Array '{name}' has first dimension {value.shape[0]}, expected {num_samples}."
      )
  if arrays.obs_valid_mask.shape != arrays.obs_history.shape[:2]:
    raise ValueError("obs_valid_mask must match obs_history first two dims.")
  if arrays.obs_history.shape[-1] != 91:
    raise ValueError(f"Expected obs_dim 91, got {arrays.obs_history.shape[-1]}.")
  if np.isnan(arrays.obs_history).any():
    raise ValueError("obs_history contains NaN values.")
  for name in (
    "true_riser_height",
    "safe_landing_center",
    "minimum_safe_stride",
    "maximum_safe_stride",
    "landing_quality_label",
    "collision_risk_label",
  ):
    value = getattr(arrays, name)
    if value is not None and np.isnan(value).any():
      raise ValueError(f"{name} contains NaN values.")
  footprint = arrays.privileged_footprint_history
  footprint_mask = arrays.privileged_footprint_valid_mask
  if footprint is None:
    if footprint_mask is not None:
      raise ValueError(
        "privileged_footprint_valid_mask requires privileged_footprint_history."
      )
  else:
    if footprint.ndim != 3:
      raise ValueError(
        "privileged_footprint_history must have shape "
        "(N, footprint_history_len, footprint_dim)."
      )
    if footprint.shape[0] != num_samples:
      raise ValueError(
        "privileged_footprint_history first dimension does not match obs_history."
      )
    if np.isnan(footprint).any():
      raise ValueError("privileged_footprint_history contains NaN values.")
    if footprint_mask is None:
      raise ValueError(
        "privileged_footprint_history requires privileged_footprint_valid_mask."
      )
    if footprint_mask.shape != footprint.shape[:2]:
      raise ValueError(
        "privileged_footprint_valid_mask must match footprint history first two dims."
      )
  sparse_events = arrays.sparse_foot_event_memory
  sparse_mask = arrays.sparse_foot_event_valid_mask
  if sparse_events is None:
    if sparse_mask is not None:
      raise ValueError(
        "sparse_foot_event_valid_mask requires sparse_foot_event_memory."
      )
    return
  if sparse_events.ndim != 3:
    raise ValueError(
      "sparse_foot_event_memory must have shape "
      "(N, sparse_event_memory_len, sparse_event_dim)."
    )
  if sparse_events.shape[0] != num_samples:
    raise ValueError(
      "sparse_foot_event_memory first dimension does not match obs_history."
    )
  if np.isnan(sparse_events).any():
    raise ValueError("sparse_foot_event_memory contains NaN values.")
  if sparse_mask is None:
    raise ValueError("sparse_foot_event_memory requires sparse_foot_event_valid_mask.")
  if sparse_mask.shape != sparse_events.shape[:2]:
    raise ValueError(
      "sparse_foot_event_valid_mask must match sparse memory first two dims."
    )


def stratified_sample_indices(
  labels: np.ndarray,
  max_samples: int,
  seed: int,
) -> np.ndarray:
  """Subsample while preserving every available label when possible."""
  if max_samples <= 0:
    raise ValueError("max_samples must be positive.")
  rng = np.random.default_rng(seed)
  unique_labels = np.unique(labels)
  per_label = max(1, max_samples // max(int(unique_labels.shape[0]), 1))
  selected: list[np.ndarray] = []
  selected_mask = np.zeros(labels.shape[0], dtype=np.bool_)
  for label in unique_labels:
    ids = np.nonzero(labels == label)[0]
    take = min(ids.shape[0], per_label)
    chosen = rng.choice(ids, size=take, replace=False)
    selected.append(chosen)
    selected_mask[chosen] = True
  selected_indices = np.concatenate(selected) if selected else np.empty(0, np.int64)
  remaining = max_samples - int(selected_indices.shape[0])
  if remaining > 0:
    pool = np.nonzero(~selected_mask)[0]
    if pool.shape[0] > 0:
      extra = rng.choice(pool, size=min(remaining, pool.shape[0]), replace=False)
      selected_indices = np.concatenate([selected_indices, extra])
  return np.sort(selected_indices.astype(np.int64, copy=False))


def filter_arrays_for_objective(
  arrays: StairProbeArrays,
  objective: ObjectiveName,
) -> StairProbeArrays:
  """Restrict arrays to samples relevant to the configured objective."""
  if objective == "multitask":
    return arrays
  if objective == "depth_only":
    depth_indices = np.nonzero(arrays.depth_valid_label.astype(np.bool_))[0]
    if depth_indices.size == 0:
      raise ValueError("depth_only objective requires at least one depth-valid sample.")
    return subset_arrays(arrays, depth_indices)
  if objective != "safe_landing":
    raise ValueError(f"Unsupported objective '{objective}'.")
  if (
    arrays.safe_stride_valid_label is None
    and arrays.landing_touchdown_label is None
    and arrays.collision_risk_label is None
  ):
    raise ValueError("safe_landing objective requires safe-landing label arrays.")
  safe_stride_valid = (
    np.zeros(arrays.obs_history.shape[0], dtype=np.bool_)
    if arrays.safe_stride_valid_label is None
    else arrays.safe_stride_valid_label.astype(np.bool_)
  )
  touchdown = (
    np.zeros(arrays.obs_history.shape[0], dtype=np.bool_)
    if arrays.landing_touchdown_label is None
    else arrays.landing_touchdown_label.astype(np.bool_)
  )
  collision = (
    np.zeros(arrays.obs_history.shape[0], dtype=np.bool_)
    if arrays.collision_risk_label is None
    else arrays.collision_risk_label.astype(np.float32) > 0.0
  )
  keep = arrays.stair_active_label.astype(np.bool_) | safe_stride_valid | touchdown
  keep |= collision
  indices = np.nonzero(keep)[0]
  if indices.size == 0:
    raise ValueError("safe_landing objective has no usable samples.")
  return subset_arrays(arrays, indices)


def sample_arrays_for_objective(
  arrays: StairProbeArrays,
  *,
  objective: ObjectiveName,
  max_samples: int | None,
  seed: int,
) -> StairProbeArrays:
  """Subsample arrays with labels that match the learning objective."""
  if max_samples is None or arrays.obs_history.shape[0] <= max_samples:
    return arrays
  if objective == "depth_only":
    labels = arrays.depth_bin_label
  elif objective == "safe_landing" and arrays.landing_touchdown_label is not None:
    touchdown = arrays.landing_touchdown_label.astype(np.int64)
    safe_stride = (
      np.zeros(arrays.obs_history.shape[0], dtype=np.int64)
      if arrays.safe_stride_valid_label is None
      else arrays.safe_stride_valid_label.astype(np.int64)
    )
    collision = (
      np.zeros(arrays.obs_history.shape[0], dtype=np.int64)
      if arrays.collision_risk_label is None
      else (arrays.collision_risk_label.astype(np.float32) > 0.5).astype(np.int64)
    )
    labels = touchdown + 2 * safe_stride + 4 * collision
  else:
    labels = arrays.sample_type
  indices = stratified_sample_indices(labels, max_samples, seed)
  return subset_arrays(arrays, indices)


def subset_arrays(arrays: StairProbeArrays, indices: np.ndarray) -> StairProbeArrays:
  """Return an indexed copy of Stage 2A arrays."""
  return StairProbeArrays(
    obs_history=arrays.obs_history[indices],
    obs_valid_mask=arrays.obs_valid_mask[indices],
    level_delta_label=arrays.level_delta_label[indices],
    relative_level_label=arrays.relative_level_label[indices],
    true_tread_depth=arrays.true_tread_depth[indices],
    depth_bin_label=arrays.depth_bin_label[indices],
    depth_valid_label=arrays.depth_valid_label[indices],
    stair_active_label=arrays.stair_active_label[indices],
    sample_type=arrays.sample_type[indices],
    sequence_id=arrays.sequence_id[indices],
    env_id=arrays.env_id[indices],
    frame_idx=arrays.frame_idx[indices],
    seed=arrays.seed[indices],
    true_riser_height=(
      None if arrays.true_riser_height is None else arrays.true_riser_height[indices]
    ),
    safe_landing_center=(
      None
      if arrays.safe_landing_center is None
      else arrays.safe_landing_center[indices]
    ),
    minimum_safe_stride=(
      None
      if arrays.minimum_safe_stride is None
      else arrays.minimum_safe_stride[indices]
    ),
    maximum_safe_stride=(
      None
      if arrays.maximum_safe_stride is None
      else arrays.maximum_safe_stride[indices]
    ),
    safe_stride_valid_label=(
      None
      if arrays.safe_stride_valid_label is None
      else arrays.safe_stride_valid_label[indices]
    ),
    landing_touchdown_label=(
      None
      if arrays.landing_touchdown_label is None
      else arrays.landing_touchdown_label[indices]
    ),
    landing_quality_label=(
      None
      if arrays.landing_quality_label is None
      else arrays.landing_quality_label[indices]
    ),
    collision_risk_label=(
      None
      if arrays.collision_risk_label is None
      else arrays.collision_risk_label[indices]
    ),
    privileged_footprint_history=(
      None
      if arrays.privileged_footprint_history is None
      else arrays.privileged_footprint_history[indices]
    ),
    privileged_footprint_valid_mask=(
      None
      if arrays.privileged_footprint_valid_mask is None
      else arrays.privileged_footprint_valid_mask[indices]
    ),
    sparse_foot_event_memory=(
      None
      if arrays.sparse_foot_event_memory is None
      else arrays.sparse_foot_event_memory[indices]
    ),
    sparse_foot_event_valid_mask=(
      None
      if arrays.sparse_foot_event_valid_mask is None
      else arrays.sparse_foot_event_valid_mask[indices]
    ),
  )


def build_sequence_split(
  sequence_id: np.ndarray,
  *,
  val_fraction: float,
  seed: int,
) -> ProbeSplit:
  """Split stair samples by sequence id and flat samples by row id."""
  if not 0.0 < val_fraction < 1.0:
    raise ValueError("val_fraction must be in (0, 1).")
  rng = np.random.default_rng(seed)
  train_mask = np.zeros(sequence_id.shape[0], dtype=np.bool_)
  val_mask = np.zeros(sequence_id.shape[0], dtype=np.bool_)

  stair_sequence_ids = np.unique(sequence_id[sequence_id >= 0])
  shuffled_sequences = stair_sequence_ids.copy()
  rng.shuffle(shuffled_sequences)
  val_sequence_count = int(round(shuffled_sequences.shape[0] * val_fraction))
  if shuffled_sequences.shape[0] > 1:
    val_sequence_count = min(
      max(1, val_sequence_count),
      int(shuffled_sequences.shape[0] - 1),
    )
  val_sequences = np.sort(shuffled_sequences[:val_sequence_count])
  train_sequences = np.sort(shuffled_sequences[val_sequence_count:])
  if train_sequences.shape[0] > 0:
    train_mask |= np.isin(sequence_id, train_sequences)
  if val_sequences.shape[0] > 0:
    val_mask |= np.isin(sequence_id, val_sequences)

  flat_indices = np.nonzero(sequence_id < 0)[0]
  rng.shuffle(flat_indices)
  val_flat_count = int(round(flat_indices.shape[0] * val_fraction))
  if flat_indices.shape[0] > 1:
    val_flat_count = min(max(1, val_flat_count), int(flat_indices.shape[0] - 1))
  val_mask[flat_indices[:val_flat_count]] = True
  train_mask[flat_indices[val_flat_count:]] = True

  train_indices = np.nonzero(train_mask)[0].astype(np.int64)
  val_indices = np.nonzero(val_mask)[0].astype(np.int64)
  if train_indices.size == 0 or val_indices.size == 0:
    raise ValueError("Train/validation split produced an empty split.")
  return ProbeSplit(
    train_indices=train_indices,
    val_indices=val_indices,
    train_sequence_ids=train_sequences,
    val_sequence_ids=val_sequences,
  )


def balanced_class_weights(
  labels: np.ndarray,
  *,
  num_classes: int,
  mask: np.ndarray | None = None,
  device: torch.device | str = "cpu",
) -> torch.Tensor:
  """Return finite inverse-frequency class weights for cross entropy."""
  if mask is not None:
    labels = labels[mask]
  labels = labels[(labels >= 0) & (labels < num_classes)]
  counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(
    np.float32
  )
  weights = np.zeros(num_classes, dtype=np.float32)
  present = counts > 0.0
  if np.any(present):
    weights[present] = counts[present].sum() / (
      float(np.count_nonzero(present)) * counts[present]
    )
    weights[present] /= max(float(weights[present].mean()), 1.0e-6)
  return torch.as_tensor(weights, dtype=torch.float32, device=device)


def make_loss_weights(
  arrays: StairProbeArrays,
  train_indices: np.ndarray,
  device: torch.device | str,
) -> ProbeLossWeights:
  """Compute loss weights from train labels only."""
  train_active = arrays.stair_active_label[train_indices].astype(np.bool_)
  positive = float(train_active.sum())
  negative = float(train_active.shape[0] - train_active.sum())
  active_pos_weight = torch.tensor(
    negative / positive if positive > 0.0 else 1.0,
    dtype=torch.float32,
    device=device,
  )
  depth_valid = arrays.depth_valid_label[train_indices].astype(np.bool_)
  return ProbeLossWeights(
    active_pos_weight=active_pos_weight,
    level_delta=balanced_class_weights(
      arrays.level_delta_label[train_indices],
      num_classes=NUM_LEVEL_DELTA_CLASSES,
      device=device,
    ),
    relative_level=balanced_class_weights(
      arrays.relative_level_label[train_indices].clip(0, 7),
      num_classes=NUM_RELATIVE_LEVEL_CLASSES,
      device=device,
    ),
    depth_bin=balanced_class_weights(
      arrays.depth_bin_label[train_indices].clip(0, 7),
      num_classes=NUM_DEPTH_BIN_CLASSES,
      mask=depth_valid,
      device=device,
    ),
  )


def move_batch(
  batch: dict[str, torch.Tensor],
  device: torch.device | str,
) -> dict[str, torch.Tensor]:
  """Move a dataloader batch to the training device."""
  return {key: value.to(device) for key, value in batch.items()}


def forward_probe_model(
  model: nn.Module,
  batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
  """Run either a single-history or two-branch probe model."""
  footprint_history = batch.get("privileged_footprint_history")
  sparse_event_memory = batch.get("sparse_foot_event_memory")
  if footprint_history is not None and sparse_event_memory is not None:
    raise ValueError("A two-branch batch cannot contain two secondary histories.")
  if footprint_history is None and sparse_event_memory is None:
    return model(batch["obs_history"])
  secondary_history = footprint_history
  if secondary_history is None:
    secondary_history = sparse_event_memory
  if secondary_history is None:
    raise RuntimeError("Two-branch model requires a secondary history tensor.")
  return model(batch["obs_history"], secondary_history)


def zero_like_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
  """Create a differentiable scalar zero on the model device."""
  return outputs["probe_latent"].sum() * 0.0


def compute_probe_loss(
  outputs: dict[str, torch.Tensor],
  batch: dict[str, torch.Tensor],
  weights: ProbeLossWeights,
  cfg: TrainStairProbeConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
  """Compute the multi-task Stage 2B probe loss."""
  if cfg.objective in ("depth_only", "safe_landing"):
    active_loss = zero_like_loss(outputs)
    level_delta_loss = zero_like_loss(outputs)
    relative_level_loss = zero_like_loss(outputs)
  else:
    active_logits = outputs["stair_active_logit"].squeeze(-1)
    active_loss = F.binary_cross_entropy_with_logits(
      active_logits,
      batch["stair_active"],
      pos_weight=weights.active_pos_weight,
    )
    level_delta_loss = F.cross_entropy(
      outputs["level_delta_logits"],
      batch["level_delta"],
      weight=weights.level_delta,
    )
    relative_level_loss = F.cross_entropy(
      outputs["relative_level_logits"],
      batch["relative_level"],
      weight=weights.relative_level,
    )

  depth_valid = batch["depth_valid"].bool()
  if cfg.objective == "safe_landing":
    depth_bin_loss = zero_like_loss(outputs)
    depth_reg_loss = zero_like_loss(outputs)
  elif bool(depth_valid.any().item()):
    depth_bin_loss = F.cross_entropy(
      outputs["depth_bin_logits"][depth_valid],
      batch["depth_bin"][depth_valid],
      weight=weights.depth_bin,
    )
    depth_reg_loss = F.smooth_l1_loss(
      outputs["depth_reg"].squeeze(-1)[depth_valid],
      batch["depth_norm"][depth_valid],
      beta=cfg.depth_huber_beta,
    )
  else:
    depth_bin_loss = zero_like_loss(outputs)
    depth_reg_loss = zero_like_loss(outputs)

  total = (
    cfg.active_loss_coef * active_loss
    + cfg.level_delta_loss_coef * level_delta_loss
    + cfg.relative_level_loss_coef * relative_level_loss
    + cfg.depth_bin_loss_coef * depth_bin_loss
    + cfg.depth_reg_loss_coef * depth_reg_loss
  )
  if cfg.objective == "safe_landing":
    touchdown_loss = F.binary_cross_entropy_with_logits(
      outputs["touchdown_logit"].squeeze(-1),
      batch["landing_touchdown"],
    )
    touchdown_mask = batch["landing_touchdown"].bool()
    landing_quality_pred = torch.sigmoid(outputs["landing_quality"].squeeze(-1))
    if bool(touchdown_mask.any().item()):
      landing_quality_loss = F.smooth_l1_loss(
        landing_quality_pred[touchdown_mask],
        batch["landing_quality"][touchdown_mask],
        beta=0.1,
      )
    else:
      landing_quality_loss = zero_like_loss(outputs)
    collision_risk_loss = F.smooth_l1_loss(
      torch.sigmoid(outputs["collision_risk"].squeeze(-1)),
      batch["collision_risk"],
      beta=0.1,
    )
    safe_stride_valid = batch["safe_stride_valid"].bool()
    if bool(safe_stride_valid.any().item()):
      safe_stride_target = torch.stack(
        [
          batch["minimum_safe_stride"],
          batch["maximum_safe_stride"],
          batch["safe_landing_center"],
        ],
        dim=-1,
      )
      safe_stride_loss = F.smooth_l1_loss(
        outputs["safe_stride"][safe_stride_valid],
        safe_stride_target[safe_stride_valid],
        beta=cfg.safe_stride_huber_beta,
      )
    else:
      safe_stride_loss = zero_like_loss(outputs)
    total = (
      cfg.touchdown_loss_coef * touchdown_loss
      + cfg.landing_quality_loss_coef * landing_quality_loss
      + cfg.collision_risk_loss_coef * collision_risk_loss
      + cfg.safe_stride_loss_coef * safe_stride_loss
    )
  else:
    touchdown_loss = zero_like_loss(outputs)
    landing_quality_loss = zero_like_loss(outputs)
    collision_risk_loss = zero_like_loss(outputs)
    safe_stride_loss = zero_like_loss(outputs)
  parts = {
    "loss": float(total.detach().cpu().item()),
    "active_loss": float(active_loss.detach().cpu().item()),
    "level_delta_loss": float(level_delta_loss.detach().cpu().item()),
    "relative_level_loss": float(relative_level_loss.detach().cpu().item()),
    "depth_bin_loss": float(depth_bin_loss.detach().cpu().item()),
    "depth_reg_loss": float(depth_reg_loss.detach().cpu().item()),
    "touchdown_loss": float(touchdown_loss.detach().cpu().item()),
    "landing_quality_loss": float(landing_quality_loss.detach().cpu().item()),
    "collision_risk_loss": float(collision_risk_loss.detach().cpu().item()),
    "safe_stride_loss": float(safe_stride_loss.detach().cpu().item()),
  }
  return total, parts


def count_parameters(model: nn.Module) -> int:
  """Return the number of trainable parameters."""
  return sum(parameter.numel() for parameter in model.parameters())


def resolve_selection_metric(cfg: TrainStairProbeConfig) -> MetricName:
  """Resolve automatic checkpoint selection for the configured objective."""
  if cfg.selection_metric != "auto":
    return cfg.selection_metric
  if cfg.objective == "depth_only":
    return "depth_bin_macro_f1"
  if cfg.objective == "safe_landing":
    return "val_loss"
  return "val_loss"


def metric_is_lower_better(metric_name: str) -> bool:
  """Return whether lower values are better for a validation metric."""
  return metric_name in {
    "val_loss",
    "depth_mae_m",
    "safe_stride_center_mae_m",
    "landing_quality_mae",
  }


def metric_improved(metric_name: str, value: float, best_value: float | None) -> bool:
  """Return whether a metric improved over its previous best value."""
  if best_value is None:
    return True
  if metric_is_lower_better(metric_name):
    return value < best_value
  return value > best_value


def binary_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
  """Compute binary F1 with zero-safe division."""
  labels_bool = labels.astype(np.bool_)
  pred_bool = predictions.astype(np.bool_)
  tp = float(np.logical_and(labels_bool, pred_bool).sum())
  fp = float(np.logical_and(~labels_bool, pred_bool).sum())
  fn = float(np.logical_and(labels_bool, ~pred_bool).sum())
  denominator = 2.0 * tp + fp + fn
  return 0.0 if denominator <= 0.0 else 2.0 * tp / denominator


def macro_f1(
  labels: np.ndarray,
  predictions: np.ndarray,
  *,
  num_classes: int,
) -> float:
  """Compute macro F1 over classes present in labels or predictions."""
  scores: list[float] = []
  for class_id in range(num_classes):
    label_class = labels == class_id
    pred_class = predictions == class_id
    if not np.any(label_class) and not np.any(pred_class):
      continue
    tp = float(np.logical_and(label_class, pred_class).sum())
    fp = float(np.logical_and(~label_class, pred_class).sum())
    fn = float(np.logical_and(label_class, ~pred_class).sum())
    denominator = 2.0 * tp + fp + fn
    scores.append(0.0 if denominator <= 0.0 else 2.0 * tp / denominator)
  return float(np.mean(scores)) if scores else 0.0


def predictions_to_depth_m(depth_reg: np.ndarray) -> np.ndarray:
  """Convert normalized depth predictions back to meters."""
  return DEPTH_MIN_M + np.clip(depth_reg, 0.0, 1.0) * (DEPTH_MAX_M - DEPTH_MIN_M)


def depth_bins_to_groups(depth_bins: np.ndarray) -> np.ndarray:
  """Map 8 fine depth bins to shallow/mid/deep diagnostic groups."""
  clipped = np.clip(depth_bins.astype(np.int64), 0, NUM_DEPTH_BIN_CLASSES - 1)
  return DEPTH_BIN_TO_GROUP[clipped]


def compute_metrics(
  labels: dict[str, np.ndarray],
  predictions: dict[str, np.ndarray],
  *,
  val_loss: float,
) -> dict[str, float]:
  """Compute Stage 2B learnability metrics."""
  sample_count = int(labels["stair_active"].shape[0])
  active_label = labels["stair_active"].astype(np.bool_)
  active_pred = predictions["stair_active_prob"] >= 0.5
  level_delta_label = labels["level_delta"].astype(np.int64)
  level_delta_pred = predictions["level_delta_pred"].astype(np.int64)
  relative_label = labels["relative_level"].astype(np.int64)
  relative_pred = predictions["relative_level_pred"].astype(np.int64)
  depth_valid = labels["depth_valid"].astype(np.bool_)
  depth_bin_label = labels["depth_bin"].astype(np.int64)
  depth_bin_pred = predictions["depth_bin_pred"].astype(np.int64)
  pred_depth_m = predictions_to_depth_m(predictions["depth_reg"].astype(np.float32))
  true_depth_m = labels["true_tread_depth"].astype(np.float32)
  touchdown_label = labels.get(
    "landing_touchdown",
    np.zeros(sample_count, dtype=np.float32),
  ).astype(np.bool_)
  touchdown_prob = predictions.get(
    "touchdown_prob",
    np.zeros(sample_count, dtype=np.float32),
  ).astype(np.float32)
  touchdown_pred = touchdown_prob >= 0.5
  landing_quality_label = labels.get(
    "landing_quality",
    np.zeros(sample_count, dtype=np.float32),
  ).astype(np.float32)
  landing_quality_pred = predictions.get(
    "landing_quality_pred",
    np.zeros(sample_count, dtype=np.float32),
  ).astype(np.float32)
  collision_risk_label = labels.get(
    "collision_risk",
    np.zeros(sample_count, dtype=np.float32),
  ).astype(np.float32)
  collision_risk_pred = predictions.get(
    "collision_risk_pred",
    np.zeros(sample_count, dtype=np.float32),
  ).astype(np.float32)
  safe_stride_valid = labels.get(
    "safe_stride_valid",
    np.zeros(sample_count, dtype=np.bool_),
  ).astype(np.bool_)
  safe_stride_pred = predictions.get(
    "safe_stride_pred",
    np.zeros((sample_count, 3), dtype=np.float32),
  ).astype(np.float32)
  safe_stride_targets = np.stack(
    [
      labels.get("minimum_safe_stride", np.zeros(sample_count, dtype=np.float32)),
      labels.get("maximum_safe_stride", np.zeros(sample_count, dtype=np.float32)),
      labels.get("safe_landing_center", np.zeros(sample_count, dtype=np.float32)),
    ],
    axis=-1,
  ).astype(np.float32)
  landing_quality_mask = touchdown_label

  metrics = {
    "val_loss": float(val_loss),
    "stair_active_accuracy": float(np.mean(active_label == active_pred)),
    "stair_active_f1": binary_f1(active_label, active_pred),
    "level_delta_accuracy": float(np.mean(level_delta_label == level_delta_pred)),
    "level_delta_macro_f1": macro_f1(
      level_delta_label,
      level_delta_pred,
      num_classes=NUM_LEVEL_DELTA_CLASSES,
    ),
    "level_delta_transition_f1": binary_f1(
      level_delta_label != 0,
      level_delta_pred != 0,
    ),
    "relative_level_accuracy": float(np.mean(relative_label == relative_pred)),
    "relative_level_mae": float(np.mean(np.abs(relative_label - relative_pred))),
    "depth_valid_count": float(depth_valid.sum()),
    "touchdown_count": float(touchdown_label.sum()),
    "touchdown_accuracy": float(np.mean(touchdown_label == touchdown_pred)),
    "touchdown_f1": binary_f1(touchdown_label, touchdown_pred),
    "landing_quality_mae": float(
      np.mean(
        np.abs(
          landing_quality_pred[landing_quality_mask]
          - landing_quality_label[landing_quality_mask]
        )
      )
      if bool(landing_quality_mask.any())
      else 0.0
    ),
    "collision_risk_mae": float(
      np.mean(np.abs(collision_risk_pred - collision_risk_label))
    ),
    "safe_stride_valid_count": float(safe_stride_valid.sum()),
  }
  if bool(safe_stride_valid.any()):
    safe_stride_abs_error = np.abs(
      safe_stride_pred[safe_stride_valid] - safe_stride_targets[safe_stride_valid]
    )
    metrics.update(
      {
        "safe_stride_min_mae_m": float(safe_stride_abs_error[:, 0].mean()),
        "safe_stride_max_mae_m": float(safe_stride_abs_error[:, 1].mean()),
        "safe_stride_center_mae_m": float(safe_stride_abs_error[:, 2].mean()),
      }
    )
  else:
    metrics.update(
      {
        "safe_stride_min_mae_m": 0.0,
        "safe_stride_max_mae_m": 0.0,
        "safe_stride_center_mae_m": 0.0,
      }
    )
  if bool(depth_valid.any()):
    depth_group_label = depth_bins_to_groups(depth_bin_label[depth_valid])
    depth_group_pred = depth_bins_to_groups(depth_bin_pred[depth_valid])
    metrics.update(
      {
        "depth_bin_accuracy": float(
          np.mean(depth_bin_label[depth_valid] == depth_bin_pred[depth_valid])
        ),
        "depth_bin_macro_f1": macro_f1(
          depth_bin_label[depth_valid],
          depth_bin_pred[depth_valid],
          num_classes=NUM_DEPTH_BIN_CLASSES,
        ),
        "depth_3group_accuracy": float(np.mean(depth_group_label == depth_group_pred)),
        "depth_3group_macro_f1": macro_f1(
          depth_group_label,
          depth_group_pred,
          num_classes=NUM_DEPTH_GROUP_CLASSES,
        ),
        "depth_mae_m": float(
          np.mean(np.abs(pred_depth_m[depth_valid] - true_depth_m[depth_valid]))
        ),
      }
    )
  else:
    metrics.update(
      {
        "depth_bin_accuracy": 0.0,
        "depth_bin_macro_f1": 0.0,
        "depth_3group_accuracy": 0.0,
        "depth_3group_macro_f1": 0.0,
        "depth_mae_m": 0.0,
      }
    )
  return metrics


def compute_baseline_metrics(labels: dict[str, np.ndarray]) -> dict[str, float]:
  """Compute majority/constant baselines for the same validation split."""
  sample_count = int(labels["stair_active"].shape[0])
  active_label = labels["stair_active"].astype(np.bool_)
  level_delta_label = labels["level_delta"].astype(np.int64)
  relative_label = labels["relative_level"].astype(np.int64)
  depth_valid = labels["depth_valid"].astype(np.bool_)
  depth_bin_label = labels["depth_bin"].astype(np.int64)
  true_depth_m = labels["true_tread_depth"].astype(np.float32)
  touchdown_label = labels.get(
    "landing_touchdown",
    np.zeros(sample_count, dtype=np.float32),
  ).astype(np.bool_)
  landing_quality_label = labels.get(
    "landing_quality",
    np.zeros(sample_count, dtype=np.float32),
  ).astype(np.float32)
  collision_risk_label = labels.get(
    "collision_risk",
    np.zeros(sample_count, dtype=np.float32),
  ).astype(np.float32)
  safe_stride_valid = labels.get(
    "safe_stride_valid",
    np.zeros(sample_count, dtype=np.bool_),
  ).astype(np.bool_)
  safe_landing_center = labels.get(
    "safe_landing_center",
    np.zeros(sample_count, dtype=np.float32),
  ).astype(np.float32)

  active_majority = np.full(active_label.shape, active_label.mean() >= 0.5)
  touchdown_majority = np.full(
    touchdown_label.shape,
    touchdown_label.mean() >= 0.5,
  )
  level_majority_class = int(
    np.bincount(level_delta_label, minlength=NUM_LEVEL_DELTA_CLASSES).argmax()
  )
  level_majority = np.full(level_delta_label.shape, level_majority_class)
  relative_majority_class = int(
    np.bincount(relative_label, minlength=NUM_RELATIVE_LEVEL_CLASSES).argmax()
  )
  relative_majority = np.full(relative_label.shape, relative_majority_class)
  baselines = {
    "active_majority_accuracy": float(np.mean(active_majority == active_label)),
    "active_majority_f1": binary_f1(active_label, active_majority),
    "level_delta_majority_accuracy": float(
      np.mean(level_majority == level_delta_label)
    ),
    "level_delta_majority_macro_f1": macro_f1(
      level_delta_label,
      level_majority,
      num_classes=NUM_LEVEL_DELTA_CLASSES,
    ),
    "level_delta_majority_transition_f1": binary_f1(
      level_delta_label != 0,
      level_majority != 0,
    ),
    "relative_level_majority_accuracy": float(
      np.mean(relative_majority == relative_label)
    ),
    "relative_level_majority_mae": float(
      np.mean(np.abs(relative_majority - relative_label))
    ),
    "depth_valid_count": float(depth_valid.sum()),
    "touchdown_majority_accuracy": float(
      np.mean(touchdown_majority == touchdown_label)
    ),
    "touchdown_majority_f1": binary_f1(touchdown_label, touchdown_majority),
    "landing_quality_mean_baseline_mae": float(
      np.abs(landing_quality_label - landing_quality_label.mean()).mean()
    ),
    "collision_risk_mean_baseline_mae": float(
      np.abs(collision_risk_label - collision_risk_label.mean()).mean()
    ),
    "safe_stride_valid_count": float(safe_stride_valid.sum()),
  }
  if bool(safe_stride_valid.any()):
    valid_center = safe_landing_center[safe_stride_valid]
    center_mean = np.full(valid_center.shape, valid_center.mean())
    baselines["safe_stride_center_mean_baseline_mae_m"] = float(
      np.abs(center_mean - valid_center).mean()
    )
  else:
    baselines["safe_stride_center_mean_baseline_mae_m"] = 0.0
  if bool(depth_valid.any()):
    valid_depth_bins = depth_bin_label[depth_valid]
    depth_majority_class = int(
      np.bincount(valid_depth_bins, minlength=NUM_DEPTH_BIN_CLASSES).argmax()
    )
    depth_majority = np.full(valid_depth_bins.shape, depth_majority_class)
    valid_depth_groups = depth_bins_to_groups(valid_depth_bins)
    depth_group_majority_class = int(
      np.bincount(valid_depth_groups, minlength=NUM_DEPTH_GROUP_CLASSES).argmax()
    )
    depth_group_majority = np.full(
      valid_depth_groups.shape,
      depth_group_majority_class,
    )
    valid_depth_m = true_depth_m[depth_valid]
    mean_depth = np.full(valid_depth_m.shape, valid_depth_m.mean())
    median_depth = np.full(valid_depth_m.shape, np.median(valid_depth_m))
    mid_depth = np.full(valid_depth_m.shape, 0.5 * (DEPTH_MIN_M + DEPTH_MAX_M))
    baselines.update(
      {
        "depth_bin_majority_accuracy": float(
          np.mean(depth_majority == valid_depth_bins)
        ),
        "depth_bin_majority_macro_f1": macro_f1(
          valid_depth_bins,
          depth_majority,
          num_classes=NUM_DEPTH_BIN_CLASSES,
        ),
        "depth_3group_majority_accuracy": float(
          np.mean(depth_group_majority == valid_depth_groups)
        ),
        "depth_3group_majority_macro_f1": macro_f1(
          valid_depth_groups,
          depth_group_majority,
          num_classes=NUM_DEPTH_GROUP_CLASSES,
        ),
        "depth_mean_baseline_mae_m": float(np.abs(mean_depth - valid_depth_m).mean()),
        "depth_median_baseline_mae_m": float(
          np.abs(median_depth - valid_depth_m).mean()
        ),
        "depth_midpoint_baseline_mae_m": float(
          np.abs(mid_depth - valid_depth_m).mean()
        ),
      }
    )
  else:
    baselines.update(
      {
        "depth_bin_majority_accuracy": 0.0,
        "depth_bin_majority_macro_f1": 0.0,
        "depth_3group_majority_accuracy": 0.0,
        "depth_3group_majority_macro_f1": 0.0,
        "depth_mean_baseline_mae_m": 0.0,
        "depth_median_baseline_mae_m": 0.0,
        "depth_midpoint_baseline_mae_m": 0.0,
      }
    )
  return baselines


def confusion_matrix(
  labels: np.ndarray,
  predictions: np.ndarray,
  *,
  num_classes: int,
) -> list[list[int]]:
  """Return a dense confusion matrix as nested Python lists."""
  matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
  valid = (
    (labels >= 0)
    & (labels < num_classes)
    & (predictions >= 0)
    & (predictions < num_classes)
  )
  for label, prediction in zip(labels[valid], predictions[valid], strict=True):
    matrix[int(label), int(prediction)] += 1
  return matrix.tolist()


def per_class_f1(
  labels: np.ndarray,
  predictions: np.ndarray,
  *,
  num_classes: int,
) -> list[float]:
  """Return per-class F1 with zero-safe division."""
  scores: list[float] = []
  for class_id in range(num_classes):
    label_class = labels == class_id
    pred_class = predictions == class_id
    tp = float(np.logical_and(label_class, pred_class).sum())
    fp = float(np.logical_and(~label_class, pred_class).sum())
    fn = float(np.logical_and(label_class, ~pred_class).sum())
    denominator = 2.0 * tp + fp + fn
    scores.append(0.0 if denominator <= 0.0 else 2.0 * tp / denominator)
  return scores


def build_class_report(
  predictions: dict[str, np.ndarray],
) -> dict[str, object]:
  """Build class-level reports for level-delta and depth-bin heads."""
  level_labels = predictions["level_delta"].astype(np.int64)
  level_preds = predictions["level_delta_pred"].astype(np.int64)
  depth_valid = predictions["depth_valid"].astype(np.bool_)
  depth_labels = predictions["depth_bin"].astype(np.int64)[depth_valid]
  depth_preds = predictions["depth_bin_pred"].astype(np.int64)[depth_valid]
  depth_group_labels = depth_bins_to_groups(depth_labels)
  depth_group_preds = depth_bins_to_groups(depth_preds)
  return {
    "level_delta": {
      "confusion": confusion_matrix(
        level_labels,
        level_preds,
        num_classes=NUM_LEVEL_DELTA_CLASSES,
      ),
      "per_class_f1": per_class_f1(
        level_labels,
        level_preds,
        num_classes=NUM_LEVEL_DELTA_CLASSES,
      ),
    },
    "depth_bin": {
      "confusion": confusion_matrix(
        depth_labels,
        depth_preds,
        num_classes=NUM_DEPTH_BIN_CLASSES,
      ),
      "per_class_f1": per_class_f1(
        depth_labels,
        depth_preds,
        num_classes=NUM_DEPTH_BIN_CLASSES,
      ),
    },
    "depth_3group": {
      "confusion": confusion_matrix(
        depth_group_labels,
        depth_group_preds,
        num_classes=NUM_DEPTH_GROUP_CLASSES,
      ),
      "per_class_f1": per_class_f1(
        depth_group_labels,
        depth_group_preds,
        num_classes=NUM_DEPTH_GROUP_CLASSES,
      ),
    },
  }


def stack_batches(chunks: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
  """Concatenate prediction or label chunks."""
  return {
    key: np.concatenate(value, axis=0) if value else np.asarray([])
    for key, value in chunks.items()
  }


@torch.no_grad()
def evaluate(
  model: nn.Module,
  loader: DataLoader,
  weights: ProbeLossWeights,
  cfg: TrainStairProbeConfig,
  device: torch.device | str,
  *,
  keep_predictions: bool = False,
) -> tuple[dict[str, float], dict[str, np.ndarray] | None]:
  """Evaluate the probe and optionally return per-sample predictions."""
  model.eval()
  loss_total = 0.0
  sample_total = 0
  label_chunks: dict[str, list[np.ndarray]] = {
    "sample_index": [],
    "stair_active": [],
    "level_delta": [],
    "relative_level": [],
    "depth_bin": [],
    "depth_valid": [],
    "true_tread_depth": [],
    "true_riser_height": [],
    "safe_landing_center": [],
    "minimum_safe_stride": [],
    "maximum_safe_stride": [],
    "safe_stride_valid": [],
    "landing_touchdown": [],
    "landing_quality": [],
    "collision_risk": [],
    "sequence_id": [],
    "env_id": [],
    "frame_idx": [],
    "sample_type": [],
  }
  prediction_chunks: dict[str, list[np.ndarray]] = {
    "stair_active_prob": [],
    "level_delta_pred": [],
    "relative_level_pred": [],
    "depth_bin_pred": [],
    "depth_reg": [],
    "touchdown_prob": [],
    "landing_quality_pred": [],
    "collision_risk_pred": [],
    "safe_stride_pred": [],
  }

  for batch in loader:
    batch = move_batch(batch, device)
    outputs = forward_probe_model(model, batch)
    loss, _parts = compute_probe_loss(outputs, batch, weights, cfg)
    batch_size = int(batch["obs_history"].shape[0])
    loss_total += float(loss.detach().cpu().item()) * batch_size
    sample_total += batch_size

    label_chunks["sample_index"].append(batch["sample_index"].cpu().numpy())
    label_chunks["stair_active"].append(batch["stair_active"].cpu().numpy())
    label_chunks["level_delta"].append(batch["level_delta"].cpu().numpy())
    label_chunks["relative_level"].append(batch["relative_level"].cpu().numpy())
    label_chunks["depth_bin"].append(batch["depth_bin"].cpu().numpy())
    label_chunks["depth_valid"].append(batch["depth_valid"].cpu().numpy())
    label_chunks["true_tread_depth"].append(batch["true_tread_depth"].cpu().numpy())
    label_chunks["true_riser_height"].append(batch["true_riser_height"].cpu().numpy())
    label_chunks["safe_landing_center"].append(
      batch["safe_landing_center"].cpu().numpy()
    )
    label_chunks["minimum_safe_stride"].append(
      batch["minimum_safe_stride"].cpu().numpy()
    )
    label_chunks["maximum_safe_stride"].append(
      batch["maximum_safe_stride"].cpu().numpy()
    )
    label_chunks["safe_stride_valid"].append(batch["safe_stride_valid"].cpu().numpy())
    label_chunks["landing_touchdown"].append(batch["landing_touchdown"].cpu().numpy())
    label_chunks["landing_quality"].append(batch["landing_quality"].cpu().numpy())
    label_chunks["collision_risk"].append(batch["collision_risk"].cpu().numpy())
    label_chunks["sequence_id"].append(batch["sequence_id"].cpu().numpy())
    label_chunks["env_id"].append(batch["env_id"].cpu().numpy())
    label_chunks["frame_idx"].append(batch["frame_idx"].cpu().numpy())
    label_chunks["sample_type"].append(batch["sample_type"].cpu().numpy())

    prediction_chunks["stair_active_prob"].append(
      torch.sigmoid(outputs["stair_active_logit"]).squeeze(-1).cpu().numpy()
    )
    prediction_chunks["level_delta_pred"].append(
      outputs["level_delta_logits"].argmax(dim=-1).cpu().numpy()
    )
    prediction_chunks["relative_level_pred"].append(
      outputs["relative_level_logits"].argmax(dim=-1).cpu().numpy()
    )
    prediction_chunks["depth_bin_pred"].append(
      outputs["depth_bin_logits"].argmax(dim=-1).cpu().numpy()
    )
    prediction_chunks["depth_reg"].append(
      outputs["depth_reg"].squeeze(-1).cpu().numpy()
    )
    prediction_chunks["touchdown_prob"].append(
      torch.sigmoid(outputs["touchdown_logit"]).squeeze(-1).cpu().numpy()
    )
    prediction_chunks["landing_quality_pred"].append(
      torch.sigmoid(outputs["landing_quality"]).squeeze(-1).cpu().numpy()
    )
    prediction_chunks["collision_risk_pred"].append(
      torch.sigmoid(outputs["collision_risk"]).squeeze(-1).cpu().numpy()
    )
    prediction_chunks["safe_stride_pred"].append(outputs["safe_stride"].cpu().numpy())

  labels = stack_batches(label_chunks)
  predictions = stack_batches(prediction_chunks)
  metrics = compute_metrics(
    labels,
    predictions,
    val_loss=loss_total / max(float(sample_total), 1.0),
  )
  if not keep_predictions:
    return metrics, None
  predictions.update(labels)
  return metrics, predictions


def train_one_epoch(
  model: nn.Module,
  loader: DataLoader,
  optimizer: torch.optim.Optimizer,
  weights: ProbeLossWeights,
  cfg: TrainStairProbeConfig,
  device: torch.device | str,
  *,
  epoch: int,
) -> dict[str, float]:
  """Run one training epoch."""
  model.train()
  totals: dict[str, float] = {}
  samples = 0
  iterator = tqdm(
    loader,
    desc=f"train epoch {epoch}",
    disable=not cfg.progress,
    dynamic_ncols=True,
  )
  for batch in iterator:
    batch = move_batch(batch, device)
    optimizer.zero_grad(set_to_none=True)
    outputs = forward_probe_model(model, batch)
    loss, parts = compute_probe_loss(outputs, batch, weights, cfg)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()

    batch_size = int(batch["obs_history"].shape[0])
    samples += batch_size
    for key, value in parts.items():
      totals[key] = totals.get(key, 0.0) + value * batch_size
    iterator.set_postfix(loss=f"{parts['loss']:.4f}")

  return {key: value / max(float(samples), 1.0) for key, value in totals.items()}


def make_probe_model(
  cfg: TrainStairProbeConfig,
  *,
  obs_dim: int | None = None,
  footprint_obs_dim: int | None = None,
) -> nn.Module:
  """Construct the configured Stage 2B probe model."""
  if cfg.model != "gru":
    raise ValueError(f"Unsupported probe model '{cfg.model}'.")
  if cfg.input_mode in TWO_BRANCH_INPUT_MODES:
    if footprint_obs_dim is None:
      raise ValueError(f"{cfg.input_mode} requires a secondary branch obs dim.")
    return TwoBranchStairProbeGRU(
      latent_obs_dim=cfg.obs_dim if obs_dim is None else obs_dim,
      footprint_obs_dim=footprint_obs_dim,
      frame_hidden_dim=cfg.frame_hidden_dim,
      recurrent_hidden_dim=cfg.recurrent_hidden_dim,
      footprint_frame_hidden_dim=cfg.footprint_frame_hidden_dim,
      footprint_recurrent_hidden_dim=cfg.footprint_recurrent_hidden_dim,
      fusion_hidden_dim=cfg.fusion_hidden_dim,
      probe_latent_dim=cfg.probe_latent_dim,
      head_hidden_dim=cfg.head_hidden_dim,
      dropout=cfg.dropout,
    )
  return StairProbeGRU(
    obs_dim=cfg.obs_dim if obs_dim is None else obs_dim,
    frame_hidden_dim=cfg.frame_hidden_dim,
    recurrent_hidden_dim=cfg.recurrent_hidden_dim,
    probe_latent_dim=cfg.probe_latent_dim,
    head_hidden_dim=cfg.head_hidden_dim,
    dropout=cfg.dropout,
  )


def make_loader(
  dataset: Dataset,
  *,
  batch_size: int,
  shuffle: bool,
  seed: int,
  num_workers: int,
) -> DataLoader:
  """Create a deterministic dataloader."""
  generator = torch.Generator()
  generator.manual_seed(seed)
  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    generator=generator,
    pin_memory=torch.cuda.is_available(),
  )


def write_json(path: Path, payload: dict) -> None:
  """Write a JSON file with stable formatting."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")


def write_predictions_csv(
  path: Path,
  predictions: dict[str, np.ndarray],
  *,
  max_rows: int,
) -> None:
  """Write a compact validation prediction CSV."""
  row_count = int(predictions["sample_index"].shape[0])
  selected = np.arange(row_count)
  if row_count > max_rows:
    selected = np.linspace(0, row_count - 1, num=max_rows, dtype=np.int64)
  fieldnames = (
    "sample_index",
    "sequence_id",
    "env_id",
    "frame_idx",
    "sample_type",
    "stair_active_label",
    "stair_active_prob",
    "level_delta_label",
    "level_delta_pred",
    "relative_level_label",
    "relative_level_pred",
    "depth_valid",
    "depth_bin_label",
    "depth_bin_pred",
    "true_tread_depth",
    "pred_tread_depth",
    "safe_stride_valid",
    "minimum_safe_stride",
    "maximum_safe_stride",
    "safe_landing_center",
    "pred_minimum_safe_stride",
    "pred_maximum_safe_stride",
    "pred_safe_landing_center",
    "landing_touchdown_label",
    "touchdown_prob",
    "landing_quality_label",
    "landing_quality_pred",
    "collision_risk_label",
    "collision_risk_pred",
  )
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    pred_depth = predictions_to_depth_m(predictions["depth_reg"].astype(np.float32))
    safe_stride_pred = predictions["safe_stride_pred"].astype(np.float32)
    for row in selected:
      writer.writerow(
        {
          "sample_index": int(predictions["sample_index"][row]),
          "sequence_id": int(predictions["sequence_id"][row]),
          "env_id": int(predictions["env_id"][row]),
          "frame_idx": int(predictions["frame_idx"][row]),
          "sample_type": int(predictions["sample_type"][row]),
          "stair_active_label": int(predictions["stair_active"][row]),
          "stair_active_prob": float(predictions["stair_active_prob"][row]),
          "level_delta_label": int(predictions["level_delta"][row]),
          "level_delta_pred": int(predictions["level_delta_pred"][row]),
          "relative_level_label": int(predictions["relative_level"][row]),
          "relative_level_pred": int(predictions["relative_level_pred"][row]),
          "depth_valid": int(predictions["depth_valid"][row]),
          "depth_bin_label": int(predictions["depth_bin"][row]),
          "depth_bin_pred": int(predictions["depth_bin_pred"][row]),
          "true_tread_depth": float(predictions["true_tread_depth"][row]),
          "pred_tread_depth": float(pred_depth[row]),
          "safe_stride_valid": int(predictions["safe_stride_valid"][row]),
          "minimum_safe_stride": float(predictions["minimum_safe_stride"][row]),
          "maximum_safe_stride": float(predictions["maximum_safe_stride"][row]),
          "safe_landing_center": float(predictions["safe_landing_center"][row]),
          "pred_minimum_safe_stride": float(safe_stride_pred[row, 0]),
          "pred_maximum_safe_stride": float(safe_stride_pred[row, 1]),
          "pred_safe_landing_center": float(safe_stride_pred[row, 2]),
          "landing_touchdown_label": int(predictions["landing_touchdown"][row]),
          "touchdown_prob": float(predictions["touchdown_prob"][row]),
          "landing_quality_label": float(predictions["landing_quality"][row]),
          "landing_quality_pred": float(predictions["landing_quality_pred"][row]),
          "collision_risk_label": float(predictions["collision_risk"][row]),
          "collision_risk_pred": float(predictions["collision_risk_pred"][row]),
        }
      )


def run_train(cfg: TrainStairProbeConfig) -> dict[str, object]:
  """Train the offline Stage 2B probe and write artifacts."""
  set_deterministic_seed(cfg.seed)
  device = torch.device(
    cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  )
  output_dir = Path(cfg.output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  arrays = load_stair_probe_arrays(cfg.dataset_file)
  arrays = filter_arrays_for_objective(arrays, cfg.objective)
  arrays = sample_arrays_for_objective(
    arrays,
    objective=cfg.objective,
    max_samples=cfg.max_samples,
    seed=cfg.seed,
  )
  if arrays.obs_history.shape[-1] != cfg.obs_dim:
    raise ValueError(
      f"Config obs_dim={cfg.obs_dim} does not match dataset obs_dim "
      f"{arrays.obs_history.shape[-1]}."
    )
  if cfg.history_len > arrays.obs_history.shape[1]:
    raise ValueError(
      f"history_len={cfg.history_len} exceeds dataset history length "
      f"{arrays.obs_history.shape[1]}."
    )
  split = build_sequence_split(
    arrays.sequence_id,
    val_fraction=cfg.val_fraction,
    seed=cfg.seed,
  )
  train_dataset = StairProbeTorchDataset(
    arrays,
    split.train_indices,
    history_len=cfg.history_len,
    footprint_history_len=cfg.footprint_history_len,
    sparse_event_memory_len=cfg.sparse_event_memory_len,
    input_mode=cfg.input_mode,
  )
  val_dataset = StairProbeTorchDataset(
    arrays,
    split.val_indices,
    history_len=cfg.history_len,
    footprint_history_len=cfg.footprint_history_len,
    sparse_event_memory_len=cfg.sparse_event_memory_len,
    input_mode=cfg.input_mode,
  )
  train_loader = make_loader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    seed=cfg.seed,
    num_workers=cfg.num_workers,
  )
  val_loader = make_loader(
    val_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    seed=cfg.seed + 1,
    num_workers=cfg.num_workers,
  )
  model_obs_dim = train_dataset.input_dim
  model_history_len = train_dataset.input_history_len
  model_latent_obs_dim = int(arrays.obs_history.shape[-1])
  footprint = arrays.privileged_footprint_history
  model_footprint_obs_dim = None if footprint is None else int(footprint.shape[-1])
  sparse_events = arrays.sparse_foot_event_memory
  model_sparse_event_obs_dim = (
    None if sparse_events is None else int(sparse_events.shape[-1])
  )
  model_latent_history_len = cfg.history_len
  model_footprint_history_len = (
    cfg.footprint_history_len if cfg.input_mode in FOOTPRINT_INPUT_MODES else None
  )
  model_sparse_event_history_len = (
    cfg.sparse_event_memory_len if cfg.input_mode in SPARSE_EVENT_INPUT_MODES else None
  )
  if cfg.input_mode in TWO_BRANCH_INPUT_MODES:
    if cfg.input_mode == "two_branch_fusion":
      model_secondary_obs_dim = model_footprint_obs_dim
      missing_message = "two_branch_fusion requires privileged_footprint_history."
    else:
      model_secondary_obs_dim = model_sparse_event_obs_dim
      missing_message = "two_branch_sparse_event requires sparse_foot_event_memory."
    if model_secondary_obs_dim is None:
      raise ValueError(missing_message)
    model = make_probe_model(
      cfg,
      obs_dim=model_latent_obs_dim,
      footprint_obs_dim=model_secondary_obs_dim,
    ).to(device)
  else:
    model = make_probe_model(cfg, obs_dim=model_obs_dim).to(device)
  weights = make_loss_weights(arrays, split.train_indices, device)
  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.learning_rate,
    weight_decay=cfg.weight_decay,
  )

  selection_metric = resolve_selection_metric(cfg)
  tracked_metrics: tuple[MetricName, ...] = (
    "val_loss",
    "level_delta_macro_f1",
    "level_delta_transition_f1",
    "depth_bin_macro_f1",
    "depth_3group_macro_f1",
    "depth_mae_m",
    "safe_stride_center_mae_m",
    "landing_quality_mae",
    "touchdown_f1",
  )
  best_by_metric: dict[str, dict[str, float | int]] = {}
  best_epoch = -1
  best_state: dict[str, torch.Tensor] | None = None
  history: list[dict[str, object]] = []
  print(
    "[Stage 2B] Train start:",
    f"samples={arrays.obs_history.shape[0]}",
    f"history_len={model_history_len}",
    f"input_mode={cfg.input_mode}",
    f"model_obs_dim={model_obs_dim}",
    f"latent_obs_dim={model_latent_obs_dim}",
    f"footprint_obs_dim={model_footprint_obs_dim}",
    f"sparse_event_obs_dim={model_sparse_event_obs_dim}",
    f"objective={cfg.objective}",
    f"selection_metric={selection_metric}",
    f"probe_latent_dim={cfg.probe_latent_dim}",
    f"device={device}",
    f"output={output_dir}",
  )

  for epoch in range(1, cfg.epochs + 1):
    train_metrics = train_one_epoch(
      model,
      train_loader,
      optimizer,
      weights,
      cfg,
      device,
      epoch=epoch,
    )
    val_metrics, _predictions = evaluate(model, val_loader, weights, cfg, device)
    history.append(
      {
        "epoch": epoch,
        "train": train_metrics,
        "val": val_metrics,
      }
    )
    print(
      "[Stage 2B] Epoch",
      epoch,
      f"train_loss={train_metrics['loss']:.5f}",
      f"val_loss={val_metrics['val_loss']:.5f}",
      f"delta_macro_f1={val_metrics['level_delta_macro_f1']:.4f}",
      f"depth_bin_macro_f1={val_metrics['depth_bin_macro_f1']:.4f}",
      f"depth_3group_macro_f1={val_metrics['depth_3group_macro_f1']:.4f}",
      f"depth_mae_m={val_metrics['depth_mae_m']:.4f}",
      f"touchdown_f1={val_metrics['touchdown_f1']:.4f}",
      f"safe_stride_center_mae_m={val_metrics['safe_stride_center_mae_m']:.4f}",
    )
    state_to_save: dict[str, torch.Tensor] | None = None
    for metric_name in tracked_metrics:
      value = float(val_metrics[metric_name])
      previous = best_by_metric.get(metric_name)
      previous_value = None if previous is None else float(previous["value"])
      if not metric_improved(metric_name, value, previous_value):
        continue
      if state_to_save is None:
        state_to_save = {
          key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
      best_by_metric[metric_name] = {"epoch": epoch, "value": value}
      torch.save(
        {
          "model_state_dict": state_to_save,
          "config": asdict(cfg),
          "model_obs_dim": model_obs_dim,
          "model_history_len": model_history_len,
          "model_latent_obs_dim": model_latent_obs_dim,
          "model_footprint_obs_dim": model_footprint_obs_dim,
          "model_sparse_event_obs_dim": model_sparse_event_obs_dim,
          "model_latent_history_len": model_latent_history_len,
          "model_footprint_history_len": model_footprint_history_len,
          "model_sparse_event_history_len": model_sparse_event_history_len,
          "epoch": epoch,
          "metrics": val_metrics,
          "selection_metric": metric_name,
        },
        output_dir / f"best_by_{metric_name}.pt",
      )
      if metric_name == selection_metric:
        best_epoch = epoch
        best_state = state_to_save
        torch.save(
          {
            "model_state_dict": state_to_save,
            "config": asdict(cfg),
            "model_obs_dim": model_obs_dim,
            "model_history_len": model_history_len,
            "model_latent_obs_dim": model_latent_obs_dim,
            "model_footprint_obs_dim": model_footprint_obs_dim,
            "model_sparse_event_obs_dim": model_sparse_event_obs_dim,
            "model_latent_history_len": model_latent_history_len,
            "model_footprint_history_len": model_footprint_history_len,
            "model_sparse_event_history_len": model_sparse_event_history_len,
            "epoch": epoch,
            "metrics": val_metrics,
            "selection_metric": selection_metric,
          },
          output_dir / "best.pt",
        )

  if best_state is not None:
    model.load_state_dict(best_state)
  best_metrics, predictions = evaluate(
    model,
    val_loader,
    weights,
    cfg,
    device,
    keep_predictions=True,
  )
  assert predictions is not None
  write_predictions_csv(
    output_dir / "val_predictions.csv",
    predictions,
    max_rows=cfg.max_prediction_rows,
  )
  val_baselines = compute_baseline_metrics(predictions)
  class_report = build_class_report(predictions)
  write_json(output_dir / "val_class_report.json", class_report)
  payload: dict[str, object] = {
    "config": asdict(cfg),
    "dataset_file": str(Path(cfg.dataset_file).expanduser()),
    "objective": cfg.objective,
    "selection_metric": selection_metric,
    "dataset_num_samples": int(arrays.obs_history.shape[0]),
    "input_mode": cfg.input_mode,
    "model_obs_dim": int(model_obs_dim),
    "model_history_len": int(model_history_len),
    "model_latent_obs_dim": int(model_latent_obs_dim),
    "model_footprint_obs_dim": (
      None if model_footprint_obs_dim is None else int(model_footprint_obs_dim)
    ),
    "model_sparse_event_obs_dim": (
      None if model_sparse_event_obs_dim is None else int(model_sparse_event_obs_dim)
    ),
    "model_latent_history_len": int(model_latent_history_len),
    "model_footprint_history_len": (
      None if model_footprint_history_len is None else int(model_footprint_history_len)
    ),
    "model_sparse_event_history_len": (
      None
      if model_sparse_event_history_len is None
      else int(model_sparse_event_history_len)
    ),
    "latent_history_shape": tuple(int(x) for x in arrays.obs_history.shape[1:]),
    "privileged_footprint_history_shape": (
      None
      if arrays.privileged_footprint_history is None
      else tuple(int(x) for x in arrays.privileged_footprint_history.shape[1:])
    ),
    "sparse_foot_event_memory_shape": (
      None
      if arrays.sparse_foot_event_memory is None
      else tuple(int(x) for x in arrays.sparse_foot_event_memory.shape[1:])
    ),
    "train_samples": int(split.train_indices.shape[0]),
    "val_samples": int(split.val_indices.shape[0]),
    "train_stair_sequences": int(split.train_sequence_ids.shape[0]),
    "val_stair_sequences": int(split.val_sequence_ids.shape[0]),
    "model": {
      "type": cfg.model,
      "trainable_parameters": count_parameters(model),
      "input_dim": int(model_obs_dim),
      "history_len": int(model_history_len),
      "latent_obs_dim": int(model_latent_obs_dim),
      "footprint_obs_dim": (
        None if model_footprint_obs_dim is None else int(model_footprint_obs_dim)
      ),
      "sparse_event_obs_dim": (
        None if model_sparse_event_obs_dim is None else int(model_sparse_event_obs_dim)
      ),
      "latent_history_len": int(model_latent_history_len),
      "footprint_history_len": (
        None
        if model_footprint_history_len is None
        else int(model_footprint_history_len)
      ),
      "sparse_event_history_len": (
        None
        if model_sparse_event_history_len is None
        else int(model_sparse_event_history_len)
      ),
      "probe_latent_dim": cfg.probe_latent_dim,
      "diagnostic_output_dim": DIAGNOSTIC_OUTPUT_DIM,
    },
    "best_epoch": best_epoch,
    "best_by_metric": best_by_metric,
    "best_metrics": best_metrics,
    "val_baselines": val_baselines,
    "history": history,
  }
  write_json(output_dir / "metrics.json", payload)
  print(
    "[Stage 2B] Train complete:",
    f"objective={cfg.objective}",
    f"selection_metric={selection_metric}",
    f"best_epoch={best_epoch}",
    f"val_loss={best_metrics['val_loss']:.5f}",
    f"delta_macro_f1={best_metrics['level_delta_macro_f1']:.4f}",
    f"depth_bin_macro_f1={best_metrics['depth_bin_macro_f1']:.4f}",
    f"depth_3group_macro_f1={best_metrics['depth_3group_macro_f1']:.4f}",
    f"depth_mae_m={best_metrics['depth_mae_m']:.4f}",
    f"touchdown_f1={best_metrics['touchdown_f1']:.4f}",
    f"safe_stride_center_mae_m={best_metrics['safe_stride_center_mae_m']:.4f}",
  )
  return payload


def main() -> None:
  cfg = tyro.cli(TrainStairProbeConfig)
  run_train(cfg)


if __name__ == "__main__":
  main()
