"""Train a deployable Stage 2D foot-event detector."""

from __future__ import annotations

import copy
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from scripts.velocity_eval.export_foot_event_detector_dataset import (
  FOOT_EVENT_LABEL_NAMES,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

SelectionMetric = Literal["auto", "val_loss", "frame_macro_f1", "event_macro_f1"]


@dataclass(frozen=True)
class TrainFootEventDetectorConfig:
  """Configuration for supervised foot-event detector training."""

  dataset_file: str = (
    "eval_outputs/stair_stage2/model51000_seed42_foot_event_detector_v1/samples.npz"
  )
  output_dir: str = (
    "eval_outputs/stair_stage2/model51000_seed42_foot_event_detector_gru_v1"
  )
  device: str | None = None
  seed: int = 12345
  history_len: int = 16
  frame_hidden_dim: int = 128
  recurrent_hidden_dim: int = 64
  head_hidden_dim: int = 32
  dropout: float = 0.0
  batch_size: int = 512
  epochs: int = 20
  learning_rate: float = 1.0e-3
  weight_decay: float = 1.0e-4
  val_fraction: float = 0.2
  num_workers: int = 0
  threshold: float = 0.5
  event_tolerance_frames: int = 2
  pos_weight_max: float = 100.0
  toe_hit_pos_weight: float | None = 150.0
  selection_metric: SelectionMetric = "auto"
  export_onnx: bool = True
  progress: bool = True


@dataclass(frozen=True)
class FootEventArrays:
  """In-memory arrays for the exported detector dataset."""

  obs_history: np.ndarray
  obs_valid_mask: np.ndarray
  event_label: np.ndarray
  episode_id: np.ndarray
  env_id: np.ndarray
  frame_idx: np.ndarray
  seed: np.ndarray


class FootEventDetectorGRU(nn.Module):
  """Small GRU detector that maps deployable obs history to event logits."""

  def __init__(
    self,
    *,
    obs_dim: int,
    frame_hidden_dim: int = 128,
    recurrent_hidden_dim: int = 64,
    head_hidden_dim: int = 32,
    output_dim: int = len(FOOT_EVENT_LABEL_NAMES),
    dropout: float = 0.0,
  ) -> None:
    super().__init__()
    self.obs_dim = int(obs_dim)
    self.output_dim = int(output_dim)
    self.frame_encoder = nn.Sequential(
      nn.Linear(obs_dim, frame_hidden_dim),
      nn.ELU(),
      nn.Dropout(dropout),
      nn.Linear(frame_hidden_dim, recurrent_hidden_dim),
      nn.ELU(),
    )
    self.gru = nn.GRU(
      input_size=recurrent_hidden_dim,
      hidden_size=recurrent_hidden_dim,
      batch_first=True,
    )
    self.head = nn.Sequential(
      nn.Linear(recurrent_hidden_dim, head_hidden_dim),
      nn.ELU(),
      nn.Dropout(dropout),
      nn.Linear(head_hidden_dim, output_dim),
    )

  def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
    """Return raw event logits for ``obs_history``."""
    if obs_history.ndim != 3:
      raise ValueError(f"Expected obs_history (B,H,D), got {tuple(obs_history.shape)}.")
    if obs_history.shape[-1] != self.obs_dim:
      raise ValueError(f"Expected obs_dim={self.obs_dim}, got {obs_history.shape[-1]}.")
    return self.forward_unchecked(obs_history)

  def forward_unchecked(self, obs_history: torch.Tensor) -> torch.Tensor:
    """Return logits without Python shape checks for ONNX tracing."""
    encoded = self.frame_encoder(obs_history)
    sequence, _hidden = self.gru(encoded)
    return self.head(sequence[:, -1])


class OnnxFootEventDetector(nn.Module):
  """ONNX export wrapper that avoids tracing Python shape guards."""

  def __init__(self, model: FootEventDetectorGRU) -> None:
    super().__init__()
    self.model = model

  def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
    """Return detector logits."""
    return self.model.forward_unchecked(obs_history)


class FootEventTorchDataset(Dataset):
  """Torch dataset that optionally shortens exported histories from the right."""

  def __init__(
    self,
    arrays: FootEventArrays,
    indices: np.ndarray,
    *,
    history_len: int,
  ) -> None:
    if history_len <= 0:
      raise ValueError("history_len must be positive.")
    if history_len > arrays.obs_history.shape[1]:
      raise ValueError(
        f"history_len={history_len} exceeds exported history "
        f"{arrays.obs_history.shape[1]}."
      )
    self.arrays = arrays
    self.indices = indices.astype(np.int64, copy=True)
    self.history_len = int(history_len)

  def __len__(self) -> int:
    return int(self.indices.shape[0])

  def __getitem__(self, item):  # type: ignore[invalid-method-override]
    sample_index = int(self.indices[item])
    return {
      "obs_history": torch.as_tensor(
        self.arrays.obs_history[sample_index, -self.history_len :, :],
        dtype=torch.float32,
      ),
      "event_label": torch.as_tensor(
        self.arrays.event_label[sample_index],
        dtype=torch.float32,
      ),
      "sample_index": torch.tensor(sample_index, dtype=torch.int64),
    }


def load_foot_event_arrays(path: str | Path) -> FootEventArrays:
  """Load a Stage 2D detector dataset."""
  with np.load(Path(path), allow_pickle=False) as data:
    arrays = FootEventArrays(
      obs_history=np.asarray(data["obs_history"], dtype=np.float32),
      obs_valid_mask=np.asarray(data["obs_valid_mask"], dtype=np.bool_),
      event_label=np.asarray(data["event_label"], dtype=np.float32),
      episode_id=np.asarray(data["episode_id"], dtype=np.int64),
      env_id=np.asarray(data["env_id"], dtype=np.int64),
      frame_idx=np.asarray(data["frame_idx"], dtype=np.int64),
      seed=np.asarray(data["seed"], dtype=np.int64),
    )
  validate_arrays(arrays)
  return arrays


def validate_arrays(arrays: FootEventArrays) -> None:
  """Validate exported detector arrays."""
  num_samples = int(arrays.obs_history.shape[0])
  if arrays.obs_history.ndim != 3:
    raise ValueError("obs_history must have shape (N,H,D).")
  if arrays.obs_valid_mask.shape != arrays.obs_history.shape[:2]:
    raise ValueError("obs_valid_mask must match obs_history first two dimensions.")
  if arrays.event_label.shape != (num_samples, len(FOOT_EVENT_LABEL_NAMES)):
    raise ValueError(
      f"event_label must have shape (N,{len(FOOT_EVENT_LABEL_NAMES)}), "
      f"got {tuple(arrays.event_label.shape)}."
    )
  for name in ("episode_id", "env_id", "frame_idx", "seed"):
    value = getattr(arrays, name)
    if value.shape != (num_samples,):
      raise ValueError(f"{name} must have shape (N,), got {tuple(value.shape)}.")
  if not np.isfinite(arrays.obs_history).all():
    raise ValueError("obs_history contains non-finite values.")
  if not np.isfinite(arrays.event_label).all():
    raise ValueError("event_label contains non-finite values.")
  if num_samples == 0:
    raise ValueError("Dataset has no samples.")


@dataclass(frozen=True)
class EpisodeSplit:
  train_indices: np.ndarray
  val_indices: np.ndarray
  train_episode_ids: np.ndarray
  val_episode_ids: np.ndarray


def build_episode_split(
  episode_id: np.ndarray,
  *,
  val_fraction: float,
  seed: int,
) -> EpisodeSplit:
  """Split by episode id so adjacent rollout frames do not leak into validation."""
  if episode_id.shape[0] < 2:
    raise ValueError("Need at least two samples for a train/validation split.")
  unique_episodes = np.unique(episode_id)
  rng = np.random.default_rng(seed)
  if unique_episodes.shape[0] < 2:
    indices = np.arange(episode_id.shape[0], dtype=np.int64)
    rng.shuffle(indices)
    val_count = int(round(float(indices.shape[0]) * val_fraction))
    val_count = min(max(val_count, 1), max(int(indices.shape[0]) - 1, 1))
    val_indices = np.sort(indices[:val_count]).astype(np.int64)
    train_indices = np.sort(indices[val_count:]).astype(np.int64)
    if train_indices.size == 0:
      train_indices = val_indices[:1]
      val_indices = val_indices[1:]
    if val_indices.size == 0:
      val_indices = train_indices[-1:]
      train_indices = train_indices[:-1]
    return EpisodeSplit(
      train_indices=train_indices,
      val_indices=val_indices,
      train_episode_ids=unique_episodes.astype(np.int64),
      val_episode_ids=unique_episodes.astype(np.int64),
    )

  shuffled = unique_episodes.copy()
  rng.shuffle(shuffled)
  val_count = int(round(float(shuffled.shape[0]) * val_fraction))
  val_count = min(max(val_count, 1), max(int(shuffled.shape[0]) - 1, 1))
  val_episodes = np.sort(shuffled[:val_count])
  train_episodes = np.sort(shuffled[val_count:])
  if train_episodes.size == 0:
    train_episodes = val_episodes[:1]
    val_episodes = val_episodes[1:]
  if val_episodes.size == 0:
    val_episodes = train_episodes[-1:]
    train_episodes = train_episodes[:-1]
  train_mask = np.isin(episode_id, train_episodes)
  val_mask = np.isin(episode_id, val_episodes)
  return EpisodeSplit(
    train_indices=np.nonzero(train_mask)[0].astype(np.int64),
    val_indices=np.nonzero(val_mask)[0].astype(np.int64),
    train_episode_ids=train_episodes.astype(np.int64),
    val_episode_ids=val_episodes.astype(np.int64),
  )


def make_pos_weight(
  labels: np.ndarray,
  *,
  max_weight: float,
  toe_hit_pos_weight: float | None = None,
) -> torch.Tensor:
  """Return clipped BCE positive weights per label."""
  positives = labels.sum(axis=0)
  negatives = labels.shape[0] - positives
  weights = np.ones(labels.shape[1], dtype=np.float32)
  nonzero = positives > 0
  weights[nonzero] = np.minimum(
    negatives[nonzero] / np.maximum(positives[nonzero], 1.0),
    float(max_weight),
  )
  if toe_hit_pos_weight is not None:
    weights[4:6] = float(toe_hit_pos_weight)
  return torch.as_tensor(weights, dtype=torch.float32)


def _binary_stats(
  labels: np.ndarray,
  probabilities: np.ndarray,
  *,
  threshold: float,
) -> dict[str, float]:
  pred = probabilities >= threshold
  lab = labels.astype(np.bool_)
  tp = float(np.logical_and(pred, lab).sum())
  fp = float(np.logical_and(pred, ~lab).sum())
  fn = float(np.logical_and(~pred, lab).sum())
  tn = float(np.logical_and(~pred, ~lab).sum())
  precision = tp / max(tp + fp, 1.0)
  recall = tp / max(tp + fn, 1.0)
  f1 = (
    0.0
    if precision + recall <= 0.0
    else 2.0 * precision * recall / (precision + recall)
  )
  return {
    "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1.0),
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "positive_count": float(lab.sum()),
    "predicted_positive_count": float(pred.sum()),
  }


def _event_indices(flags: np.ndarray, frames: np.ndarray) -> np.ndarray:
  """Return frame numbers for starts of positive runs."""
  if flags.size == 0:
    return np.zeros((0,), dtype=np.int64)
  order = np.argsort(frames)
  sorted_flags = flags[order].astype(np.bool_)
  sorted_frames = frames[order].astype(np.int64)
  starts = sorted_flags & np.concatenate(([True], ~sorted_flags[:-1]))
  return sorted_frames[starts]


def _event_start_frames_and_indices(
  active: np.ndarray,
  frames: np.ndarray,
  *,
  cooldown_frames: int = 0,
  release_values: np.ndarray | None = None,
  release_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Return positive-run start frames and original indices with optional gating."""
  if active.size == 0:
    return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
  order = np.argsort(frames)
  sorted_active = active[order].astype(np.bool_)
  sorted_frames = frames[order].astype(np.int64)
  frame_gap = np.concatenate(([True], np.diff(sorted_frames) > 1))
  starts = sorted_active & (frame_gap | np.concatenate(([True], ~sorted_active[:-1])))
  if release_values is not None and release_threshold is not None:
    sorted_release = release_values[order].astype(np.float32)
    previous_release = np.concatenate(([0.0], sorted_release[:-1]))
    release_low = frame_gap | (previous_release < float(release_threshold))
    starts &= release_low
  start_positions = np.nonzero(starts)[0]
  if cooldown_frames > 0 and start_positions.size > 0:
    kept: list[int] = []
    last_frame = -(10**12)
    for position in start_positions:
      frame = int(sorted_frames[position])
      if frame - last_frame <= cooldown_frames:
        continue
      kept.append(int(position))
      last_frame = frame
    start_positions = np.asarray(kept, dtype=np.int64)
  return sorted_frames[start_positions], order[start_positions].astype(np.int64)


def _match_events(
  true_events: np.ndarray,
  pred_events: np.ndarray,
  *,
  tolerance_frames: int,
) -> tuple[int, int, int]:
  """Greedily match predicted event frames to true event frames."""
  matched = np.zeros(true_events.shape[0], dtype=np.bool_)
  tp = 0
  fp = 0
  for pred in pred_events:
    candidates = np.nonzero(
      (~matched) & (np.abs(true_events - pred) <= tolerance_frames)
    )[0]
    if candidates.size == 0:
      fp += 1
      continue
    best = candidates[np.argmin(np.abs(true_events[candidates] - pred))]
    matched[best] = True
    tp += 1
  fn = int((~matched).sum())
  return tp, fp, fn


def event_f1_for_label(
  labels: np.ndarray,
  probabilities: np.ndarray,
  episode_id: np.ndarray,
  frame_idx: np.ndarray,
  *,
  threshold: float,
  tolerance_frames: int,
) -> dict[str, float]:
  """Compute event-level precision/recall/F1 after collapsing positive runs."""
  total_tp = 0
  total_fp = 0
  total_fn = 0
  for episode in np.unique(episode_id):
    mask = episode_id == episode
    true_events = _event_indices(labels[mask] > 0.5, frame_idx[mask])
    pred_events = _event_indices(probabilities[mask] >= threshold, frame_idx[mask])
    tp, fp, fn = _match_events(
      true_events,
      pred_events,
      tolerance_frames=tolerance_frames,
    )
    total_tp += tp
    total_fp += fp
    total_fn += fn
  precision = total_tp / max(total_tp + total_fp, 1)
  recall = total_tp / max(total_tp + total_fn, 1)
  f1 = (
    0.0
    if precision + recall <= 0.0
    else 2.0 * precision * recall / (precision + recall)
  )
  return {
    "event_precision": float(precision),
    "event_recall": float(recall),
    "event_f1": float(f1),
    "event_tp": float(total_tp),
    "event_fp": float(total_fp),
    "event_fn": float(total_fn),
  }


def deployment_event_stats_for_label(
  *,
  labels: np.ndarray,
  probabilities: np.ndarray,
  episode_id: np.ndarray,
  frame_idx: np.ndarray,
  event_label_index: int,
  threshold: float,
  tolerance_frames: int,
  cooldown_frames: int = 0,
  contact_label_index: int | None = None,
  contact_prob_index: int | None = None,
  contact_threshold: float = 0.5,
  contact_release_threshold: float | None = None,
  valid_mask: np.ndarray | None = None,
) -> dict[str, float]:
  """Evaluate deploy-style event triggers after gates, rising edges, and cooldown."""
  total_tp = 0
  total_fp = 0
  total_fn = 0
  predicted_count = 0
  true_count = 0
  predicted_contact_true = 0
  predicted_valid_true = 0
  base_mask = (
    np.ones(labels.shape[0], dtype=np.bool_)
    if valid_mask is None
    else valid_mask.astype(np.bool_)
  )
  for episode in np.unique(episode_id):
    mask = (episode_id == episode) & base_mask
    if not mask.any():
      continue
    local_labels = labels[mask]
    local_probabilities = probabilities[mask]
    local_frames = frame_idx[mask].astype(np.int64)
    active = local_probabilities[:, event_label_index] >= threshold
    if contact_prob_index is not None:
      active &= local_probabilities[:, contact_prob_index] >= contact_threshold
      release_values = local_probabilities[:, contact_prob_index]
    else:
      release_values = None
    true_events = _event_indices(
      local_labels[:, event_label_index] > 0.5,
      local_frames,
    )
    pred_events, pred_indices = _event_start_frames_and_indices(
      active,
      local_frames,
      cooldown_frames=cooldown_frames,
      release_values=release_values,
      release_threshold=contact_release_threshold,
    )
    tp, fp, fn = _match_events(
      true_events,
      pred_events,
      tolerance_frames=tolerance_frames,
    )
    total_tp += tp
    total_fp += fp
    total_fn += fn
    predicted_count += int(pred_events.shape[0])
    true_count += int(true_events.shape[0])
    if contact_label_index is not None and pred_indices.size > 0:
      predicted_contact_true += int(
        (local_labels[pred_indices, contact_label_index] > 0.5).sum()
      )
    if valid_mask is not None:
      predicted_valid_true += int(pred_events.shape[0])
  precision = total_tp / max(total_tp + total_fp, 1)
  recall = total_tp / max(total_tp + total_fn, 1)
  f1 = (
    0.0
    if precision + recall <= 0.0
    else 2.0 * precision * recall / (precision + recall)
  )
  result = {
    "event_precision": float(precision),
    "event_recall": float(recall),
    "event_f1": float(f1),
    "event_tp": float(total_tp),
    "event_fp": float(total_fp),
    "event_fn": float(total_fn),
    "predicted_event_count": float(predicted_count),
    "true_event_count": float(true_count),
  }
  if contact_label_index is not None:
    result["predicted_contact_precision"] = predicted_contact_true / max(
      predicted_count,
      1,
    )
  if valid_mask is not None:
    result["predicted_valid_precision"] = predicted_valid_true / max(
      predicted_count,
      1,
    )
  return result


def sweep_deployment_event_thresholds(
  *,
  labels: np.ndarray,
  probabilities: np.ndarray,
  episode_id: np.ndarray,
  frame_idx: np.ndarray,
  event_label_index: int,
  thresholds: np.ndarray,
  tolerance_frames: int,
  cooldown_frames: int = 0,
  contact_label_index: int | None = None,
  contact_prob_index: int | None = None,
  contact_threshold: float = 0.5,
  contact_release_threshold: float | None = None,
  valid_mask: np.ndarray | None = None,
) -> dict[str, float]:
  """Return best deploy-style event metrics across candidate thresholds."""
  if thresholds.size == 0:
    raise ValueError("thresholds must be non-empty.")
  best_threshold = float(thresholds[0])
  best_stats: dict[str, float] | None = None
  for threshold in thresholds:
    stats = deployment_event_stats_for_label(
      labels=labels,
      probabilities=probabilities,
      episode_id=episode_id,
      frame_idx=frame_idx,
      event_label_index=event_label_index,
      threshold=float(threshold),
      tolerance_frames=tolerance_frames,
      cooldown_frames=cooldown_frames,
      contact_label_index=contact_label_index,
      contact_prob_index=contact_prob_index,
      contact_threshold=contact_threshold,
      contact_release_threshold=contact_release_threshold,
      valid_mask=valid_mask,
    )
    if best_stats is None or (
      stats["event_f1"],
      stats["event_precision"],
      -abs(float(threshold) - 0.5),
    ) > (
      best_stats["event_f1"],
      best_stats["event_precision"],
      -abs(best_threshold - 0.5),
    ):
      best_threshold = float(threshold)
      best_stats = stats
  if best_stats is None:
    raise RuntimeError("Threshold sweep produced no metrics.")
  return {
    "best_threshold": best_threshold,
    **{f"best_{key}": value for key, value in best_stats.items()},
  }


def compute_metrics(
  *,
  labels: np.ndarray,
  probabilities: np.ndarray,
  episode_id: np.ndarray,
  frame_idx: np.ndarray,
  val_loss: float,
  threshold: float,
  event_tolerance_frames: int,
) -> dict[str, float]:
  """Compute frame-level and event-level detector metrics."""
  metrics: dict[str, float] = {"val_loss": float(val_loss)}
  frame_f1_values: list[float] = []
  event_f1_values: list[float] = []
  event_label_indices = (2, 3, 4, 5)
  for index, name in enumerate(FOOT_EVENT_LABEL_NAMES):
    stats = _binary_stats(
      labels[:, index], probabilities[:, index], threshold=threshold
    )
    for metric_name, value in stats.items():
      metrics[f"{name}_{metric_name}"] = float(value)
    frame_f1_values.append(float(stats["f1"]))
    if index in event_label_indices:
      event_stats = event_f1_for_label(
        labels[:, index],
        probabilities[:, index],
        episode_id,
        frame_idx,
        threshold=threshold,
        tolerance_frames=event_tolerance_frames,
      )
      for metric_name, value in event_stats.items():
        metrics[f"{name}_{metric_name}"] = float(value)
      event_f1_values.append(float(event_stats["event_f1"]))
  metrics["frame_macro_f1"] = (
    float(np.mean(frame_f1_values)) if frame_f1_values else 0.0
  )
  metrics["event_macro_f1"] = (
    float(np.mean(event_f1_values)) if event_f1_values else 0.0
  )
  return metrics


def resolve_selection_metric(cfg: TrainFootEventDetectorConfig) -> str:
  """Resolve automatic checkpoint selection."""
  if cfg.selection_metric == "auto":
    return "event_macro_f1"
  return cfg.selection_metric


def metric_is_higher_better(metric_name: str) -> bool:
  if metric_name == "val_loss":
    return False
  return metric_name.endswith("_f1")


def evaluate(
  model: FootEventDetectorGRU,
  loader: DataLoader[dict[str, torch.Tensor]],
  arrays: FootEventArrays,
  *,
  device: torch.device,
  threshold: float,
  event_tolerance_frames: int,
  pos_weight: torch.Tensor,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
  """Evaluate model and return metrics plus aligned predictions."""
  model.eval()
  sample_indices: list[np.ndarray] = []
  label_chunks: list[np.ndarray] = []
  probability_chunks: list[np.ndarray] = []
  total_loss = 0.0
  total_count = 0
  with torch.no_grad():
    for batch in loader:
      obs = batch["obs_history"].to(device)
      labels = batch["event_label"].to(device)
      logits = model(obs)
      loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
      count = int(obs.shape[0])
      total_loss += float(loss.item()) * count
      total_count += count
      sample_indices.append(batch["sample_index"].cpu().numpy().astype(np.int64))
      label_chunks.append(labels.cpu().numpy().astype(np.float32))
      probability_chunks.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))
  indices = np.concatenate(sample_indices, axis=0)
  labels_np = np.concatenate(label_chunks, axis=0)
  probabilities = np.concatenate(probability_chunks, axis=0)
  val_loss = total_loss / max(float(total_count), 1.0)
  metrics = compute_metrics(
    labels=labels_np,
    probabilities=probabilities,
    episode_id=arrays.episode_id[indices],
    frame_idx=arrays.frame_idx[indices],
    val_loss=val_loss,
    threshold=threshold,
    event_tolerance_frames=event_tolerance_frames,
  )
  return metrics, {
    "sample_index": indices,
    "label": labels_np,
    "probability": probabilities,
  }


def write_predictions(output_dir: Path, predictions: dict[str, np.ndarray]) -> None:
  """Write validation predictions for inspection."""
  path = output_dir / "val_predictions.csv"
  labels = predictions["label"]
  probabilities = predictions["probability"]
  sample_indices = predictions["sample_index"].astype(np.int64)
  with path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream)
    header = ["sample_index"]
    for name in FOOT_EVENT_LABEL_NAMES:
      header.append(f"{name}_label")
      header.append(f"{name}_prob")
    writer.writerow(header)
    for row in range(labels.shape[0]):
      values: list[int | float] = [int(sample_indices[row])]
      for col in range(labels.shape[1]):
        values.append(float(labels[row, col]))
        values.append(float(probabilities[row, col]))
      writer.writerow(values)


def export_detector_onnx(
  model: FootEventDetectorGRU,
  output_path: Path,
  *,
  history_len: int,
  obs_dim: int,
) -> None:
  """Export detector logits model to ONNX for deploy-side sigmoid/thresholding."""
  model_cpu = OnnxFootEventDetector(copy.deepcopy(model)).to("cpu")
  model_cpu.eval()
  dummy = torch.zeros(1, history_len, obs_dim, dtype=torch.float32)
  torch.onnx.export(
    model_cpu,
    (dummy,),
    str(output_path),
    input_names=["obs_history"],
    output_names=["event_logits"],
    dynamic_axes={
      "obs_history": {0: "batch"},
      "event_logits": {0: "batch"},
    },
    opset_version=17,
    dynamo=False,
  )


def run_train(cfg: TrainFootEventDetectorConfig) -> dict[str, object]:
  """Train the foot-event detector."""
  random.seed(cfg.seed)
  np.random.seed(cfg.seed)
  torch.manual_seed(cfg.seed)
  device = torch.device(
    cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  )
  output_dir = Path(cfg.output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  arrays = load_foot_event_arrays(cfg.dataset_file)
  if cfg.history_len > arrays.obs_history.shape[1]:
    raise ValueError(
      f"history_len={cfg.history_len} exceeds dataset history "
      f"{arrays.obs_history.shape[1]}."
    )
  split = build_episode_split(
    arrays.episode_id,
    val_fraction=cfg.val_fraction,
    seed=cfg.seed,
  )
  train_dataset = FootEventTorchDataset(
    arrays,
    split.train_indices,
    history_len=cfg.history_len,
  )
  val_dataset = FootEventTorchDataset(
    arrays,
    split.val_indices,
    history_len=cfg.history_len,
  )
  generator = torch.Generator()
  generator.manual_seed(cfg.seed)
  train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    generator=generator,
  )
  val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
  )
  model = FootEventDetectorGRU(
    obs_dim=int(arrays.obs_history.shape[-1]),
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
  pos_weight = make_pos_weight(
    arrays.event_label[split.train_indices],
    max_weight=cfg.pos_weight_max,
    toe_hit_pos_weight=cfg.toe_hit_pos_weight,
  ).to(device)
  selection_metric = resolve_selection_metric(cfg)
  higher_better = metric_is_higher_better(selection_metric)
  best_value = -float("inf") if higher_better else float("inf")
  best_epoch = 0
  best_state: dict[str, torch.Tensor] | None = None
  history: list[dict[str, object]] = []
  best_predictions: dict[str, np.ndarray] | None = None

  print(
    "[Stage 2D] Train detector:",
    f"samples={arrays.obs_history.shape[0]}",
    f"train={len(train_dataset)}",
    f"val={len(val_dataset)}",
    f"history_len={cfg.history_len}",
    f"obs_dim={arrays.obs_history.shape[-1]}",
    f"selection_metric={selection_metric}",
    f"device={device}",
    f"output={output_dir}",
  )
  epoch_iter = tqdm(
    range(1, cfg.epochs + 1),
    desc="foot event detector",
    disable=not cfg.progress,
    dynamic_ncols=True,
    unit="epoch",
  )
  for epoch in epoch_iter:
    model.train()
    train_loss = 0.0
    train_count = 0
    for batch in train_loader:
      obs = batch["obs_history"].to(device)
      labels = batch["event_label"].to(device)
      logits = model(obs)
      loss = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight,
      )
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
      count = int(obs.shape[0])
      train_loss += float(loss.item()) * count
      train_count += count
    train_loss /= max(float(train_count), 1.0)
    val_metrics, val_predictions = evaluate(
      model,
      val_loader,
      arrays,
      device=device,
      threshold=cfg.threshold,
      event_tolerance_frames=cfg.event_tolerance_frames,
      pos_weight=pos_weight,
    )
    metric_value = float(val_metrics[selection_metric])
    is_better = (
      metric_value > best_value if higher_better else metric_value < best_value
    )
    if is_better:
      best_value = metric_value
      best_epoch = epoch
      best_state = copy.deepcopy(model.state_dict())
      best_predictions = val_predictions
      torch.save(best_state, output_dir / "best.pt")
    history.append({"epoch": epoch, "train_loss": train_loss, "val": val_metrics})
    print(
      f"[Stage 2D] Epoch {epoch}",
      f"train_loss={train_loss:.5f}",
      f"val_loss={val_metrics['val_loss']:.5f}",
      f"frame_macro_f1={val_metrics['frame_macro_f1']:.4f}",
      f"event_macro_f1={val_metrics['event_macro_f1']:.4f}",
    )

  if best_state is None or best_predictions is None:
    raise RuntimeError("Training did not produce a best checkpoint.")
  model.load_state_dict(best_state)
  write_predictions(output_dir, best_predictions)
  if cfg.export_onnx:
    export_detector_onnx(
      model,
      output_dir / "best.onnx",
      history_len=cfg.history_len,
      obs_dim=int(arrays.obs_history.shape[-1]),
    )

  payload: dict[str, object] = {
    "config": asdict(cfg),
    "dataset_file": str(cfg.dataset_file),
    "label_names": list(FOOT_EVENT_LABEL_NAMES),
    "history": history,
    "best_epoch": best_epoch,
    "selection_metric": selection_metric,
    "best_metric_value": best_value,
    "model": {
      "type": "FootEventDetectorGRU",
      "obs_dim": int(arrays.obs_history.shape[-1]),
      "history_len": int(cfg.history_len),
      "output_dim": len(FOOT_EVENT_LABEL_NAMES),
      "frame_hidden_dim": int(cfg.frame_hidden_dim),
      "recurrent_hidden_dim": int(cfg.recurrent_hidden_dim),
      "head_hidden_dim": int(cfg.head_hidden_dim),
      "trainable_parameters": int(
        sum(parameter.numel() for parameter in model.parameters())
      ),
    },
    "split": {
      "train_samples": int(split.train_indices.shape[0]),
      "val_samples": int(split.val_indices.shape[0]),
      "train_episodes": int(split.train_episode_ids.shape[0]),
      "val_episodes": int(split.val_episode_ids.shape[0]),
    },
    "pos_weight": [float(value) for value in pos_weight.detach().cpu().tolist()],
    "onnx_path": str(output_dir / "best.onnx") if cfg.export_onnx else None,
  }
  with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
  print(
    "[Stage 2D] Train complete:",
    f"best_epoch={best_epoch}",
    f"{selection_metric}={best_value:.5f}",
    f"output={output_dir}",
  )
  return payload


def main() -> None:
  cfg = tyro.cli(TrainFootEventDetectorConfig)
  run_train(cfg)


if __name__ == "__main__":
  main()
