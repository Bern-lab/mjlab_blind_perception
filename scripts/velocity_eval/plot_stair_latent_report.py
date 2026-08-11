"""Plot stair latent state, Semantic-v2 dynamics, and hidden clusters."""

from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

EmbeddingMethod = Literal["pca", "tsne"]
GateModeFilter = Literal["all", "normal", "write", "memory"]

SEMANTIC_CHANNELS = (
  "event_on",
  "stair_on",
  "mode_normal",
  "mode_write",
  "mode_memory",
  "write_progress",
  "memory_age",
  "release_progress",
  "safe_stride_control",
  "riser_height",
  "stride_lower",
  "stride_upper",
  "safe_stride_control_2",
  "stride_width",
  "stride_trend",
  "confidence",
)

CLUSTER_FEATURES = {
  "encoder_out": ("encoder_out_cluster_pca.png", "Encoder output"),
  "h_t": ("h_t_cluster_pca.png", "LSTM hidden state h_t"),
  "c_t": ("c_t_cluster_pca.png", "LSTM cell state c_t"),
  "z_candidate": ("z_candidate_cluster_pca.png", "Candidate slow latent"),
  "hidden": ("hidden_cluster_pca.png", "Recurrent hidden state"),
  "latent": ("actor_latent_cluster_pca.png", "Actor latent"),
  "z_memory": ("z_memory_cluster_pca.png", "Slow latent memory"),
  "z_state_memory": ("state_memory_cluster_pca.png", "State memory"),
  "z_shape_memory": ("shape_memory_cluster_pca.png", "Shape memory"),
}

FEATURE_ALIASES = {
  "encoder_out": "encoder_out",
  "h_t": "h_t",
  "c_t": "c_t",
  "z_candidate": "z_candidate",
  "hidden": "hidden",
  "latent": "actor_latent",
  "z_memory": "z_memory",
  "z_state_memory": "z_state",
  "z_shape_memory": "z_shape",
  "semantic": "semantic",
  "shadow_semantic": "semantic",
}

FEATURE_TITLES = {
  "encoder_out": "Encoder output",
  "h_t": "LSTM hidden state h_t",
  "c_t": "LSTM cell state c_t",
  "z_candidate": "Candidate slow latent",
  "hidden": "Recurrent hidden state",
  "latent": "Actor latent",
  "z_memory": "Slow latent memory",
  "z_state_memory": "State memory",
  "z_shape_memory": "Shape memory",
  "semantic": "Semantic-v2",
  "shadow_semantic": "Semantic-v2 shadow",
}


@dataclass(frozen=True)
class PlotStairLatentReportConfig:
  input_file: str
  output_dir: str | None = None
  episode_id: int | None = None
  terrain_label: str | None = None
  max_samples: int | None = 50000
  phase_bins: int | None = None
  phase_bin: int | None = None
  terrain_kind_filter: str | None = None
  gate_mode_filter: GateModeFilter = "all"
  min_memory_age: float | None = None
  trace_max_steps: int | None = 700
  raster_max_episodes: int = 120
  seed: int = 45678
  dpi: int = 160
  write_trace_csv: bool = True
  embedding_method: EmbeddingMethod = "pca"
  embedding_dim: int = 3
  tsne_perplexity: float = 30.0
  tsne_learning_rate: float | Literal["auto"] = "auto"
  tsne_init: Literal["pca", "random"] = "pca"
  tsne_random_state: int = 42
  tsne_max_iter: int = 1000
  tsne_verbose: int = 1
  write_interactive_html: bool = True
  interactive_features: str = "auto"
  interactive_max_episodes: int = 200
  write_static_orthogonal_projections: bool = True
  progress: bool = True


@dataclass(frozen=True)
class TraceSelection:
  episode_id: int
  terrain_label: str


def _progress(cfg: PlotStairLatentReportConfig, message: str) -> None:
  if cfg.progress:
    print(f"[INFO] {message}", flush=True)


def _json_ready(value):
  if isinstance(value, dict):
    return {key: _json_ready(item) for key, item in value.items()}
  if isinstance(value, list):
    return [_json_ready(item) for item in value]
  if isinstance(value, tuple):
    return [_json_ready(item) for item in value]
  if isinstance(value, np.integer):
    return int(value)
  if isinstance(value, np.floating):
    value = float(value)
  if isinstance(value, float):
    return value if np.isfinite(value) else None
  return value


def _row_count(data: np.lib.npyio.NpzFile) -> int:
  for key in data.files:
    value = np.asarray(data[key])
    if value.ndim > 0:
      return int(value.shape[0])
  raise ValueError("The latent dataset has no row-like arrays.")


def _string_array(
  data: np.lib.npyio.NpzFile,
  key: str,
  *,
  default: str,
) -> np.ndarray:
  n = _row_count(data)
  if key not in data.files:
    return np.full(n, default, dtype=str)
  return np.asarray(data[key]).astype(str)


def _float_array(
  data: np.lib.npyio.NpzFile,
  key: str,
  *,
  default: float,
) -> np.ndarray:
  n = _row_count(data)
  if key not in data.files:
    return np.full(n, default, dtype=np.float32)
  return np.asarray(data[key], dtype=np.float32)


def _episode_array(data: np.lib.npyio.NpzFile) -> np.ndarray:
  n = _row_count(data)
  if "episode_id" not in data.files:
    return np.zeros(n, dtype=np.int64)
  return np.asarray(data["episode_id"], dtype=np.int64)


def _size_labels(
  data: np.lib.npyio.NpzFile,
  indices: np.ndarray | None = None,
) -> np.ndarray:
  labels = _string_array(data, "terrain_label", default="sample")
  kinds = _string_array(data, "terrain_kind", default="")
  heights = _float_array(data, "terrain_height_m", default=0.0)
  if "terrain_tread_depth_m" in data.files:
    depths = _float_array(data, "terrain_tread_depth_m", default=0.0)
  else:
    depths = _float_array(data, "terrain_step_width_m", default=0.0)

  out = []
  for label, kind, height, depth in zip(labels, kinds, heights, depths, strict=True):
    is_stair = kind in {"upstairs", "downstairs"} or (height > 0.0 and depth > 0.0)
    if is_stair:
      h_cm = int(round(float(height) * 100.0))
      d_cm = int(round(float(depth) * 100.0))
      out.append(f"h{h_cm:02d}_d{d_cm:02d}")
    else:
      out.append(str(label))
  result = np.asarray(out)
  return result if indices is None else result[indices]


def _phase_filter(
  data: np.lib.npyio.NpzFile,
  cfg: PlotStairLatentReportConfig,
) -> np.ndarray:
  n = _row_count(data)
  if cfg.phase_bin is None:
    return np.ones(n, dtype=bool)
  if cfg.phase_bins is None:
    raise ValueError("Use --phase-bins when selecting --phase-bin.")
  if cfg.phase_bins <= 0:
    raise ValueError("--phase-bins must be positive.")
  if not 0 <= cfg.phase_bin < cfg.phase_bins:
    raise ValueError("--phase-bin must be in [0, phase_bins).")
  if "gait_phase" not in data.files:
    raise ValueError("This latent file has no gait_phase array.")
  phase = np.asarray(data["gait_phase"], dtype=np.float32) % 1.0
  bins = np.floor(phase * cfg.phase_bins).astype(np.int64)
  bins = np.clip(bins, 0, cfg.phase_bins - 1)
  return bins == cfg.phase_bin


def _comma_values(value: str | None) -> set[str]:
  if value is None:
    return set()
  return {item.strip() for item in value.split(",") if item.strip()}


def _sample_filter(
  data: np.lib.npyio.NpzFile,
  cfg: PlotStairLatentReportConfig,
) -> np.ndarray:
  mask = _phase_filter(data, cfg)

  allowed_kinds = _comma_values(cfg.terrain_kind_filter)
  if allowed_kinds:
    kinds = _string_array(data, "terrain_kind", default="")
    mask &= np.isin(kinds, list(allowed_kinds))

  if cfg.gate_mode_filter != "all":
    if "gate_mode" not in data.files:
      raise ValueError("This latent file has no gate_mode array.")
    mask &= _mode_labels(data) == cfg.gate_mode_filter

  if cfg.min_memory_age is not None:
    if "gate_memory_age" not in data.files:
      raise ValueError("This latent file has no gate_memory_age array.")
    age = np.asarray(data["gate_memory_age"], dtype=np.float32).reshape(-1)
    mask &= age >= float(cfg.min_memory_age)

  return mask


def _finite_feature_rows(features: np.ndarray) -> np.ndarray:
  if features.ndim == 1:
    features = features[:, None]
  return np.all(np.isfinite(features), axis=1)


def _sample_indices(
  indices: np.ndarray,
  *,
  max_samples: int | None,
  seed: int,
) -> np.ndarray:
  if max_samples is None or indices.size <= max_samples:
    return indices
  rng = np.random.default_rng(seed)
  return np.sort(rng.choice(indices, size=max_samples, replace=False))


def _pca(
  features: np.ndarray,
  *,
  n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
  features = np.asarray(features, dtype=np.float32)
  if features.ndim == 1:
    features = features[:, None]
  if features.shape[0] == 0:
    return (
      np.zeros((0, n_components), dtype=np.float32),
      np.zeros(n_components, dtype=np.float32),
    )
  centered = features - features.mean(axis=0, keepdims=True)
  _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
  components = min(n_components, vt.shape[0])
  pcs = centered @ vt[:components].T
  if components < n_components:
    pcs = np.pad(pcs, ((0, 0), (0, n_components - components)))
  variance = singular_values**2
  explained = np.zeros(n_components, dtype=np.float32)
  denom = max(float(variance.sum()), 1.0e-12)
  explained[:components] = variance[:components] / denom
  return pcs[:, :n_components].astype(np.float32), explained


def _pca2(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  return _pca(features, n_components=2)


def _standardize_features(features: np.ndarray) -> np.ndarray:
  features = np.asarray(features, dtype=np.float32)
  if features.ndim == 1:
    features = features[:, None]
  centered = features - features.mean(axis=0, keepdims=True)
  scale = centered.std(axis=0, keepdims=True)
  return centered / np.where(scale > 1.0e-6, scale, 1.0)


def _effective_tsne_perplexity(requested: float, n_samples: int) -> float:
  if n_samples < 2:
    return 1.0
  return float(min(max(requested, 1.0), max(float(n_samples - 1), 1.0)))


def _embedding_coordinates(
  features: np.ndarray,
  cfg: PlotStairLatentReportConfig,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
  if cfg.embedding_dim not in (2, 3):
    raise ValueError("--embedding-dim must be 2 or 3.")
  if cfg.embedding_method == "pca":
    coords, explained = _pca(features, n_components=cfg.embedding_dim)
    metrics: dict[str, float | int | str] = {
      "embedding_method": "pca",
      "embedding_dim": int(cfg.embedding_dim),
    }
    for idx, value in enumerate(explained, start=1):
      metrics[f"pc{idx}_explained"] = float(value)
    return coords, metrics

  if cfg.embedding_method != "tsne":
    raise ValueError(f"Unsupported embedding method: {cfg.embedding_method}")
  features = np.asarray(features, dtype=np.float32)
  if features.ndim == 1:
    features = features[:, None]
  if features.shape[0] < 3:
    coords, explained = _pca(features, n_components=cfg.embedding_dim)
    return coords, {
      "embedding_method": "pca_fallback_for_small_sample",
      "embedding_dim": int(cfg.embedding_dim),
      "pc1_explained": float(explained[0]) if explained.size > 0 else 0.0,
    }

  try:
    from sklearn.manifold import TSNE
  except ModuleNotFoundError as exc:
    raise RuntimeError(
      "The t-SNE report requires scikit-learn. Run `uv sync`, or use "
      "`--embedding-method pca`."
    ) from exc

  perplexity = _effective_tsne_perplexity(cfg.tsne_perplexity, features.shape[0])
  init = cfg.tsne_init
  if init == "pca" and min(features.shape) < cfg.embedding_dim:
    init = "random"
  kwargs = {
    "n_components": cfg.embedding_dim,
    "perplexity": perplexity,
    "learning_rate": cfg.tsne_learning_rate,
    "init": init,
    "random_state": cfg.tsne_random_state,
    "verbose": cfg.tsne_verbose,
  }
  tsne = TSNE(max_iter=cfg.tsne_max_iter, **kwargs)
  coords = tsne.fit_transform(_standardize_features(features)).astype(np.float32)
  return coords, {
    "embedding_method": "tsne",
    "embedding_dim": int(cfg.embedding_dim),
    "tsne_perplexity": float(perplexity),
    "tsne_requested_perplexity": float(cfg.tsne_perplexity),
    "tsne_init": init,
    "tsne_random_state": int(cfg.tsne_random_state),
    "tsne_max_iter": int(cfg.tsne_max_iter),
  }


def _feature_alias(feature_key: str) -> str:
  return FEATURE_ALIASES.get(feature_key, feature_key)


def _feature_title(feature_key: str) -> str:
  return FEATURE_TITLES.get(feature_key, feature_key)


def _resolve_interactive_feature(
  requested: str,
  data: np.lib.npyio.NpzFile,
) -> str | None:
  aliases = {
    "actor": "latent",
    "actor_latent": "latent",
    "state": "z_state_memory",
    "z_state": "z_state_memory",
    "shape": "z_shape_memory",
    "z_shape": "z_shape_memory",
    "memory": "z_memory",
    "semantic": _semantic_key(data) or "semantic",
  }
  feature_key = aliases.get(requested, requested)
  if feature_key == "h_t" and feature_key not in data.files and "hidden" in data.files:
    feature_key = "hidden"
  return feature_key if feature_key in data.files else None


def _interactive_feature_keys(
  data: np.lib.npyio.NpzFile,
  cfg: PlotStairLatentReportConfig,
) -> list[str]:
  if cfg.interactive_features.strip().lower() != "auto":
    requested = [
      item.strip() for item in cfg.interactive_features.split(",") if item.strip()
    ]
    keys = [
      key
      for item in requested
      if (key := _resolve_interactive_feature(item, data)) is not None
    ]
    return list(dict.fromkeys(keys))

  preferred = [
    "h_t",
    "hidden",
    "z_memory",
    "z_state_memory",
    "z_shape_memory",
    _semantic_key(data) or "",
    "latent",
  ]
  return [key for key in dict.fromkeys(preferred) if key and key in data.files]


def _json_float_list(values: np.ndarray) -> list[float | None]:
  flat = np.asarray(values, dtype=np.float32).reshape(-1)
  return [float(value) if np.isfinite(value) else None for value in flat]


def _json_string_list(values: np.ndarray) -> list[str]:
  return [str(value) for value in np.asarray(values).reshape(-1)]


def _mode_values(
  data: np.lib.npyio.NpzFile,
  indices: np.ndarray,
) -> np.ndarray | None:
  labels = _mode_labels(data)
  if labels.size == 0 or not np.any(labels != ""):
    return None
  return labels[indices]


def _depth_values(data: np.lib.npyio.NpzFile, indices: np.ndarray) -> np.ndarray:
  if "terrain_tread_depth_m" in data.files:
    return _float_array(data, "terrain_tread_depth_m", default=0.0)[indices]
  return _float_array(data, "terrain_step_width_m", default=0.0)[indices]


def _color_modes(
  data: np.lib.npyio.NpzFile,
  indices: np.ndarray,
  terrain_labels: np.ndarray,
) -> list[dict[str, object]]:
  modes: list[dict[str, object]] = [
    {
      "name": "Terrain size",
      "kind": "categorical",
      "values": _json_string_list(terrain_labels),
    },
    {
      "name": "Riser height",
      "kind": "continuous",
      "values": _json_float_list(
        _float_array(data, "terrain_height_m", default=0.0)[indices]
      ),
      "unit": "m",
    },
    {
      "name": "Tread depth",
      "kind": "continuous",
      "values": _json_float_list(_depth_values(data, indices)),
      "unit": "m",
    },
  ]
  mode_values = _mode_values(data, indices)
  if mode_values is not None:
    modes.append(
      {
        "name": "Gate mode",
        "kind": "categorical",
        "values": _json_string_list(mode_values),
      }
    )
  if "gait_phase" in data.files:
    modes.append(
      {
        "name": "Gait phase",
        "kind": "continuous",
        "values": _json_float_list(
          np.asarray(data["gait_phase"], dtype=np.float32)[indices] % 1.0
        ),
      }
    )
  for key, name in (
    ("event_prob", "Event probability"),
    ("stair_prob", "Stair probability"),
    ("safe_stride_confidence", "SafeStride confidence"),
  ):
    values = _series(data, key, indices)
    if values is not None:
      modes.append(
        {
          "name": name,
          "kind": "continuous",
          "values": _json_float_list(values),
        }
      )
  if "episode_id" in data.files:
    modes.append(
      {
        "name": "Episode",
        "kind": "categorical",
        "values": _json_string_list(_episode_array(data)[indices]),
      }
    )
  return modes


def _hover_rows(
  data: np.lib.npyio.NpzFile,
  indices: np.ndarray,
  terrain_labels: np.ndarray,
) -> list[str]:
  episode_ids = _episode_array(data)[indices]
  heights = _float_array(data, "terrain_height_m", default=0.0)[indices]
  depths = _depth_values(data, indices)
  kinds = _string_array(data, "terrain_kind", default="")[indices]
  time = _float_array(data, "time_s", default=np.nan)[indices]
  steps = _float_array(data, "time_step", default=np.nan)[indices]
  gate = _mode_values(data, indices)
  gait = (
    np.asarray(data["gait_phase"], dtype=np.float32)[indices] % 1.0
    if "gait_phase" in data.files
    else np.full(indices.size, np.nan, dtype=np.float32)
  )
  event = _series(data, "event_prob", indices)
  stair = _series(data, "stair_prob", indices)

  rows = []
  for idx in range(indices.size):
    parts = [
      f"terrain: {terrain_labels[idx]}",
      f"kind: {kinds[idx]}",
      f"height: {heights[idx]:.3f} m",
      f"depth: {depths[idx]:.3f} m",
      f"episode: {int(episode_ids[idx])}",
    ]
    if np.isfinite(time[idx]):
      parts.append(f"time: {time[idx]:.2f} s")
    if np.isfinite(steps[idx]):
      parts.append(f"step: {int(steps[idx])}")
    if gate is not None:
      parts.append(f"gate: {gate[idx]}")
    if np.isfinite(gait[idx]):
      parts.append(f"gait phase: {gait[idx]:.3f}")
    if event is not None and np.isfinite(event[idx]):
      parts.append(f"event prob: {event[idx]:.3f}")
    if stair is not None and np.isfinite(stair[idx]):
      parts.append(f"stair prob: {stair[idx]:.3f}")
    rows.append("<br>".join(html.escape(part) for part in parts))
  return rows


def _pad_embedding3(coords: np.ndarray) -> np.ndarray:
  coords = np.asarray(coords, dtype=np.float32)
  if coords.ndim == 1:
    coords = coords[:, None]
  if coords.shape[1] >= 3:
    return coords[:, :3]
  return np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))


def _normalized_coords(coords: np.ndarray) -> np.ndarray:
  coords = _pad_embedding3(coords)
  centered = coords - coords.mean(axis=0, keepdims=True)
  radius = np.linalg.norm(centered, axis=1)
  scale = float(np.percentile(radius, 98.0)) if radius.size else 1.0
  if not np.isfinite(scale) or scale < 1.0e-6:
    scale = max(float(np.max(np.abs(centered))), 1.0)
  return centered / scale


def _axis_name(method: str, axis: int) -> str:
  prefix = "tSNE" if method == "tsne" else "PC"
  return f"{prefix}-{axis + 1}"


def _plot_static_orthogonal_projections(
  coords: np.ndarray,
  labels: np.ndarray,
  output_dir: Path,
  *,
  feature_key: str,
  cfg: PlotStairLatentReportConfig,
) -> list[str]:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  coords = _pad_embedding3(coords)
  output_dir.mkdir(parents=True, exist_ok=True)
  alias = _feature_alias(feature_key)
  method = cfg.embedding_method
  dim = cfg.embedding_dim
  files = []
  pairs = ((0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz"))
  unique_labels = list(dict.fromkeys(labels.tolist()))
  cmap = plt.get_cmap("tab20")
  for x_axis, y_axis, suffix in pairs:
    path = output_dir / f"{alias}_{method}{dim}d_{suffix}.png"
    fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=cfg.dpi)
    for label_idx, label in enumerate(unique_labels):
      label_mask = labels == label
      ax.scatter(
        coords[label_mask, x_axis],
        coords[label_mask, y_axis],
        s=8,
        alpha=0.50,
        linewidths=0,
        color=cmap(label_idx % cmap.N),
        label=label,
      )
    ax.set_xlabel(_axis_name(method, x_axis))
    ax.set_ylabel(_axis_name(method, y_axis))
    ax.set_title(f"{_feature_title(feature_key)} | {suffix.upper()} projection")
    ax.grid(True, alpha=0.20)
    ax.legend(markerscale=2.5, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    files.append(str(path))
  return files


def _interactive_html_template() -> str:
  return r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light; font-family: Inter, system-ui, sans-serif; }
  body { margin: 0; background: #f7f7f4; color: #171717; }
  .app { display: grid; grid-template-columns: 280px 1fr; height: 100vh; }
  aside { border-right: 1px solid #d8d5ce; padding: 16px; overflow: auto; }
  h1 { font-size: 16px; line-height: 1.25; margin: 0 0 14px; }
  label { display: block; font-size: 12px; color: #555; margin: 14px 0 6px; }
  select, button, input { width: 100%; box-sizing: border-box; }
  select, button { border: 1px solid #bbb6aa; background: white; padding: 8px; }
  button { cursor: pointer; }
  .meta { font-size: 12px; line-height: 1.45; color: #555; margin-top: 14px; }
  .legend { display: grid; gap: 6px; margin-top: 10px; font-size: 12px; }
  .legend-row { display: grid; grid-template-columns: 14px 1fr; gap: 7px; }
  .swatch { width: 12px; height: 12px; border-radius: 50%; margin-top: 2px; }
  main { position: relative; min-width: 0; }
  canvas { display: block; width: 100%; height: 100%; background: #ffffff; }
  .tooltip {
    position: absolute; pointer-events: none; background: rgba(23, 23, 23, 0.92);
    color: white; padding: 8px 10px; border-radius: 6px; font-size: 12px;
    line-height: 1.35; max-width: 280px; display: none;
  }
  @media (max-width: 760px) {
    .app { grid-template-columns: 1fr; grid-template-rows: auto 1fr; }
    aside { border-right: 0; border-bottom: 1px solid #d8d5ce; }
  }
</style>
</head>
<body>
<div class="app">
<aside>
  <h1>__TITLE__</h1>
  <label for="colorMode">Color by</label>
  <select id="colorMode"></select>
  <label for="episode">Episode path</label>
  <select id="episode"></select>
  <label for="pointSize">Point size</label>
  <input id="pointSize" type="range" min="1" max="6" step="0.5" value="2.2">
  <label for="alpha">Point alpha</label>
  <input id="alpha" type="range" min="0.1" max="1" step="0.05" value="0.62">
  <button id="reset">Reset view</button>
  <div id="legend" class="legend"></div>
  <div class="meta">
    Left drag rotates. Right drag or shift-drag pans. Wheel zooms. Hover a
    point for rollout metadata.
  </div>
</aside>
<main>
  <canvas id="scene"></canvas>
  <div id="tooltip" class="tooltip"></div>
</main>
</div>
<script>
const payload = __PAYLOAD__;
const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");
const colorSelect = document.getElementById("colorMode");
const episodeSelect = document.getElementById("episode");
const pointSize = document.getElementById("pointSize");
const alpha = document.getElementById("alpha");
const tooltip = document.getElementById("tooltip");
const legend = document.getElementById("legend");
const palette = [
  "#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00",
  "#56b4e9", "#000000", "#f0e442", "#7b3294", "#a6761d",
  "#1b9e77", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"
];
let yaw = 0.75;
let pitch = 0.38;
let zoom = 1.0;
let panX = 0;
let panY = 0;
let dragging = false;
let dragMode = "rotate";
let lastX = 0;
let lastY = 0;
let projected = [];
const hiddenCategories = new Map();
for (const mode of payload.colorModes) {
  const option = document.createElement("option");
  option.value = mode.name;
  option.textContent = mode.name;
  colorSelect.appendChild(option);
}
episodeSelect.appendChild(new Option("All episodes", ""));
for (const episode of payload.episodeOptions) {
  episodeSelect.appendChild(new Option("Episode " + episode, String(episode)));
}
function resize() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  render();
}
function currentMode() {
  return payload.colorModes.find(mode => mode.name === colorSelect.value)
    || payload.colorModes[0];
}
function finiteNumbers(values) {
  return values.filter(value => Number.isFinite(value));
}
function ramp(value, min, max) {
  if (!Number.isFinite(value)) return "#9a9a9a";
  const stops = [
    [68, 1, 84], [59, 82, 139], [33, 145, 140],
    [94, 201, 98], [253, 231, 37]
  ];
  const t = Math.max(0, Math.min(1, (value - min) / (max - min || 1)));
  const scaled = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(scaled));
  const f = scaled - i;
  const color = [0, 1, 2].map(j => Math.round(stops[i][j] * (1 - f)
    + stops[i + 1][j] * f));
  return `rgb(${color[0]},${color[1]},${color[2]})`;
}
function colorsFor(mode) {
  const values = mode.values;
  if (mode.kind === "continuous") {
    const finite = finiteNumbers(values);
    const min = Math.min(...finite);
    const max = Math.max(...finite);
    return values.map(value => ramp(value, min, max));
  }
  const categories = [...new Set(values.map(String))];
  const lookup = new Map(categories.map((cat, idx) => [
    cat, palette[idx % palette.length]
  ]));
  return values.map(value => lookup.get(String(value)) || "#9a9a9a");
}
function categoryIsHidden(mode, idx) {
  if (mode.kind !== "categorical") return false;
  const hidden = hiddenCategories.get(mode.name);
  return hidden ? hidden.has(String(mode.values[idx])) : false;
}
function updateLegend() {
  const mode = currentMode();
  legend.replaceChildren();
  if (mode.kind === "continuous") {
    const finite = finiteNumbers(mode.values);
    const min = Math.min(...finite);
    const max = Math.max(...finite);
    const row = document.createElement("div");
    row.textContent = `${min.toFixed(3)} to ${max.toFixed(3)} ${mode.unit || ""}`;
    legend.appendChild(row);
    return;
  }
  const categories = [...new Set(mode.values.map(String))];
  const hidden = hiddenCategories.get(mode.name) || new Set();
  categories.slice(0, 24).forEach((cat, idx) => {
    const row = document.createElement("div");
    row.className = "legend-row";
    row.style.cursor = "pointer";
    row.style.opacity = hidden.has(cat) ? "0.35" : "1";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = palette[idx % palette.length];
    const text = document.createElement("span");
    text.textContent = cat;
    row.appendChild(swatch);
    row.appendChild(text);
    row.addEventListener("click", () => {
      const next = new Set(hiddenCategories.get(mode.name) || []);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      hiddenCategories.set(mode.name, next);
      render();
    });
    legend.appendChild(row);
  });
}
function projectAll() {
  const w = canvas.width;
  const h = canvas.height;
  const base = Math.min(w, h) * 0.38 * zoom;
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  projected = payload.points.map(point => {
    const x1 = cy * point[0] + sy * point[2];
    const z1 = -sy * point[0] + cy * point[2];
    const y1 = cp * point[1] - sp * z1;
    const z2 = sp * point[1] + cp * z1;
    return {
      x: w * 0.5 + panX + x1 * base,
      y: h * 0.5 + panY - y1 * base,
      z: z2
    };
  });
}
function drawPath() {
  const selected = episodeSelect.value;
  if (!selected) return;
  const indices = [];
  for (let i = 0; i < payload.episodes.length; i += 1) {
    if (String(payload.episodes[i]) === selected) indices.push(i);
  }
  indices.sort((a, b) => payload.times[a] - payload.times[b]);
  if (indices.length < 2) return;
  ctx.save();
  ctx.strokeStyle = "#111111";
  ctx.lineWidth = 2.4 * (window.devicePixelRatio || 1);
  ctx.globalAlpha = 0.82;
  ctx.beginPath();
  indices.forEach((idx, order) => {
    const p = projected[idx];
    if (order === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.stroke();
  ctx.fillStyle = "#d55e00";
  for (const idx of indices) {
    const p = projected[idx];
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3.8 * (window.devicePixelRatio || 1), 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}
function render() {
  if (!canvas.width || !canvas.height) return;
  updateLegend();
  projectAll();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const mode = currentMode();
  const colors = colorsFor(mode);
  const order = projected.map((p, idx) => [idx, p.z])
    .sort((a, b) => a[1] - b[1]);
  const radius = Number(pointSize.value) * (window.devicePixelRatio || 1);
  ctx.save();
  ctx.globalAlpha = Number(alpha.value);
  for (const [idx] of order) {
    if (categoryIsHidden(mode, idx)) continue;
    const p = projected[idx];
    ctx.fillStyle = colors[idx];
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
  drawPath();
}
function nearestPoint(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const x = (clientX - rect.left) * ratio;
  const y = (clientY - rect.top) * ratio;
  const mode = currentMode();
  let best = -1;
  let bestDist = 12 * 12 * ratio * ratio;
  for (let i = 0; i < projected.length; i += 1) {
    if (categoryIsHidden(mode, i)) continue;
    const dx = projected[i].x - x;
    const dy = projected[i].y - y;
    const dist = dx * dx + dy * dy;
    if (dist < bestDist) {
      best = i;
      bestDist = dist;
    }
  }
  return best;
}
canvas.addEventListener("mousemove", event => {
  if (dragging) {
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;
    lastX = event.clientX;
    lastY = event.clientY;
    const ratio = window.devicePixelRatio || 1;
    if (dragMode === "pan") {
      panX += dx * ratio;
      panY += dy * ratio;
    } else {
      yaw += dx * 0.008;
      pitch = Math.max(-1.45, Math.min(1.45, pitch + dy * 0.008));
    }
    render();
    return;
  }
  const idx = nearestPoint(event.clientX, event.clientY);
  if (idx < 0) {
    tooltip.style.display = "none";
    return;
  }
  tooltip.innerHTML = payload.hover[idx];
  tooltip.style.left = `${event.clientX + 12}px`;
  tooltip.style.top = `${event.clientY + 12}px`;
  tooltip.style.display = "block";
});
canvas.addEventListener("mouseleave", () => {
  tooltip.style.display = "none";
});
canvas.addEventListener("mousedown", event => {
  dragging = true;
  dragMode = event.button === 2 || event.shiftKey ? "pan" : "rotate";
  lastX = event.clientX;
  lastY = event.clientY;
});
window.addEventListener("mouseup", () => {
  dragging = false;
});
canvas.addEventListener("contextmenu", event => event.preventDefault());
canvas.addEventListener("wheel", event => {
  event.preventDefault();
  zoom *= Math.exp(-event.deltaY * 0.001);
  zoom = Math.max(0.08, Math.min(20.0, zoom));
  render();
}, { passive: false });
document.getElementById("reset").addEventListener("click", () => {
  yaw = 0.75;
  pitch = 0.38;
  zoom = 1.0;
  panX = 0;
  panY = 0;
  render();
});
colorSelect.addEventListener("change", render);
episodeSelect.addEventListener("change", render);
pointSize.addEventListener("input", render);
alpha.addEventListener("input", render);
window.addEventListener("resize", resize);
resize();
</script>
</body>
</html>
"""


def _write_interactive_html(
  data: np.lib.npyio.NpzFile,
  feature_key: str,
  coords: np.ndarray,
  indices: np.ndarray,
  output_path: Path,
  cfg: PlotStairLatentReportConfig,
  embedding_metrics: dict[str, float | int | str],
) -> dict[str, object]:
  labels = _size_labels(data, indices)
  norm_coords = _normalized_coords(coords)
  episodes = _episode_array(data)[indices]
  times = _float_array(data, "time_s", default=np.nan)[indices]
  if not np.isfinite(times).any():
    times = _float_array(data, "time_step", default=np.nan)[indices]
  episode_options = [
    int(value) for value in sorted(set(int(item) for item in episodes.tolist()))
  ][: max(cfg.interactive_max_episodes, 0)]
  title = (
    f"{_feature_title(feature_key)} {cfg.embedding_method.upper()}-{cfg.embedding_dim}D"
  )
  payload = {
    "points": norm_coords.tolist(),
    "hover": _hover_rows(data, indices, labels),
    "colorModes": _color_modes(data, indices, labels),
    "episodes": [int(value) for value in episodes.tolist()],
    "times": _json_float_list(times),
    "episodeOptions": episode_options,
    "metadata": {
      "feature": feature_key,
      "samples": int(indices.size),
      **embedding_metrics,
    },
  }
  rendered = (
    _interactive_html_template()
    .replace("__TITLE__", html.escape(title))
    .replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
  )
  output_path.write_text(rendered, encoding="utf-8")
  return {
    "feature": feature_key,
    "samples": int(indices.size),
    "file": str(output_path),
    **embedding_metrics,
  }


def _write_interactive_embedding(
  data: np.lib.npyio.NpzFile,
  feature_key: str,
  interactive_dir: Path,
  static_dir: Path,
  cfg: PlotStairLatentReportConfig,
) -> dict[str, object]:
  _progress(cfg, f"Preparing {feature_key} embedding.")
  features = np.asarray(data[feature_key], dtype=np.float32)
  if features.ndim == 1:
    features = features[:, None]
  mask = _finite_feature_rows(features) & _sample_filter(data, cfg)
  indices = _sample_indices(
    np.nonzero(mask)[0],
    max_samples=cfg.max_samples,
    seed=cfg.seed,
  )
  if indices.size == 0:
    raise ValueError(f"No finite rows available for {feature_key}.")

  _progress(
    cfg,
    f"Running {cfg.embedding_method.upper()}-{cfg.embedding_dim}D for "
    f"{feature_key} on {indices.size} samples.",
  )
  coords, metrics = _embedding_coordinates(features[indices], cfg)
  alias = _feature_alias(feature_key)
  suffix = f"{cfg.embedding_method}{cfg.embedding_dim}d"
  interactive_dir.mkdir(parents=True, exist_ok=True)
  html_path = interactive_dir / f"{alias}_{suffix}.html"
  summary = _write_interactive_html(
    data,
    feature_key,
    coords,
    indices,
    html_path,
    cfg,
    metrics,
  )
  _progress(cfg, f"Wrote {html_path}.")
  if cfg.write_static_orthogonal_projections:
    labels = _size_labels(data, indices)
    summary["orthogonal_projection_files"] = _plot_static_orthogonal_projections(
      coords,
      labels,
      static_dir,
      feature_key=feature_key,
      cfg=cfg,
    )
    _progress(cfg, f"Wrote static projections for {feature_key}.")
  return summary


def _plot_cluster(
  data: np.lib.npyio.NpzFile,
  feature_key: str,
  output_path: Path,
  cfg: PlotStairLatentReportConfig,
  title: str,
) -> dict[str, float | int | str]:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  features = np.asarray(data[feature_key], dtype=np.float32)
  if features.ndim == 1:
    features = features[:, None]
  mask = _finite_feature_rows(features) & _sample_filter(data, cfg)
  indices = _sample_indices(
    np.nonzero(mask)[0],
    max_samples=cfg.max_samples,
    seed=cfg.seed,
  )
  pcs, explained = _pca2(features[indices])
  labels = _size_labels(data, indices)

  fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=cfg.dpi)
  unique_labels = list(dict.fromkeys(labels.tolist()))
  cmap = plt.get_cmap("tab20")
  for label_idx, label in enumerate(unique_labels):
    label_mask = labels == label
    ax.scatter(
      pcs[label_mask, 0],
      pcs[label_mask, 1],
      s=8,
      alpha=0.50,
      linewidths=0,
      color=cmap(label_idx % cmap.N),
      label=label,
    )
  ax.set_xlabel(f"PC1 ({explained[0] * 100.0:.1f}% var)")
  ax.set_ylabel(f"PC2 ({explained[1] * 100.0:.1f}% var)")
  if cfg.phase_bin is not None and cfg.phase_bins is not None:
    title = f"{title} | gait phase bin {cfg.phase_bin}/{cfg.phase_bins}"
  ax.set_title(title)
  ax.grid(True, alpha=0.20)
  ax.legend(markerscale=2.5, frameon=False, fontsize=8)
  fig.tight_layout()
  fig.savefig(output_path)
  plt.close(fig)
  return {
    "feature": feature_key,
    "samples": int(indices.size),
    "pc1_explained": float(explained[0]),
    "pc2_explained": float(explained[1]),
    "file": str(output_path),
  }


def _series(
  data: np.lib.npyio.NpzFile,
  key: str,
  indices: np.ndarray,
  *,
  component: int = 0,
) -> np.ndarray | None:
  if key not in data.files:
    return None
  values = np.asarray(data[key], dtype=np.float32)[indices]
  if values.ndim == 1:
    return values
  if values.shape[-1] <= component:
    return None
  return values[:, component]


def _matrix(
  data: np.lib.npyio.NpzFile,
  key: str,
  indices: np.ndarray,
) -> np.ndarray | None:
  if key not in data.files:
    return None
  values = np.asarray(data[key], dtype=np.float32)[indices]
  if values.ndim == 1:
    values = values[:, None]
  return values


def _truth_value(data: np.lib.npyio.NpzFile, key: str, indices: np.ndarray) -> float:
  if key not in data.files:
    return 0.0
  values = np.asarray(data[key], dtype=np.float32)[indices]
  finite = values[np.isfinite(values)]
  return 0.0 if finite.size == 0 else float(np.median(finite))


def _trace_score(
  data: np.lib.npyio.NpzFile,
  mask: np.ndarray,
) -> float:
  score = 0.0
  for key, weight in (
    ("stair_prob", 1.0),
    ("event_prob", 0.8),
    ("gate_mode", 0.4),
    ("gate_write_confirm", 0.4),
    ("gate_memory_event_shape_boost", 0.3),
  ):
    if key not in data.files:
      continue
    values = np.asarray(data[key], dtype=np.float32)[mask]
    if values.size:
      score += weight * float(np.nanmax(values))
  height = _float_array(data, "terrain_height_m", default=0.0)[mask]
  if height.size and float(np.nanmax(height)) > 0.0:
    score += 0.25
  score += 1.0e-6 * float(np.count_nonzero(mask))
  return score


def _select_trace(
  data: np.lib.npyio.NpzFile,
  cfg: PlotStairLatentReportConfig,
) -> TraceSelection:
  episode_ids = _episode_array(data)
  terrain_labels = _string_array(data, "terrain_label", default="sample")
  eligible = np.ones_like(episode_ids, dtype=bool)
  if cfg.episode_id is not None:
    eligible &= episode_ids == cfg.episode_id
  if cfg.terrain_label is not None:
    eligible &= terrain_labels == cfg.terrain_label
  if not bool(np.any(eligible)):
    raise ValueError("No rows match the requested episode/terrain filter.")

  best: TraceSelection | None = None
  best_score = -np.inf
  groups = sorted(
    {
      (int(ep), str(label))
      for ep, label in zip(episode_ids[eligible], terrain_labels[eligible], strict=True)
    }
  )
  for episode_id, terrain_label in groups:
    mask = eligible & (episode_ids == episode_id) & (terrain_labels == terrain_label)
    score = _trace_score(data, mask)
    if score > best_score:
      best = TraceSelection(episode_id=episode_id, terrain_label=terrain_label)
      best_score = score
  assert best is not None
  return best


def _trace_indices(
  data: np.lib.npyio.NpzFile,
  selection: TraceSelection,
  cfg: PlotStairLatentReportConfig,
) -> np.ndarray:
  episode_ids = _episode_array(data)
  terrain_labels = _string_array(data, "terrain_label", default="sample")
  mask = (episode_ids == selection.episode_id) & (
    terrain_labels == selection.terrain_label
  )
  indices = np.nonzero(mask)[0]
  if "time_s" in data.files:
    order = np.argsort(np.asarray(data["time_s"], dtype=np.float32)[indices])
  elif "time_step" in data.files:
    order = np.argsort(np.asarray(data["time_step"], dtype=np.float32)[indices])
  else:
    order = np.arange(indices.size)
  indices = indices[order]
  if cfg.trace_max_steps is not None:
    indices = indices[: cfg.trace_max_steps]
  return indices


def _x_axis(data: np.lib.npyio.NpzFile, indices: np.ndarray) -> tuple[np.ndarray, str]:
  if "time_s" in data.files:
    return np.asarray(data["time_s"], dtype=np.float32)[indices], "time (s)"
  if "time_step" in data.files:
    return np.asarray(data["time_step"], dtype=np.float32)[indices], "step"
  return np.arange(indices.size, dtype=np.float32), "sample"


def _plot_event_markers(
  ax,
  data: np.lib.npyio.NpzFile,
  indices: np.ndarray,
  x: np.ndarray,
) -> None:
  for key, color, label in (
    ("gate_event_trigger", "#d55e00", "trigger"),
    ("gate_write_confirm", "#009e73", "confirm"),
    ("gate_memory_exit", "#0072b2", "exit"),
  ):
    values = _series(data, key, indices)
    if values is None:
      continue
    event_indices = np.nonzero(values > 0.5)[0][:16]
    for idx, event_index in enumerate(event_indices):
      ax.axvline(
        x[event_index],
        color=color,
        alpha=0.35,
        linewidth=1.0,
        label=label if idx == 0 else None,
      )


def _delta_norm(values: np.ndarray | None) -> np.ndarray | None:
  if values is None or values.shape[0] == 0:
    return None
  delta = np.diff(values, axis=0, prepend=values[:1])
  return np.linalg.norm(delta, axis=1)


def _plot_state_switch(
  data: np.lib.npyio.NpzFile,
  indices: np.ndarray,
  selection: TraceSelection,
  output_path: Path,
  cfg: PlotStairLatentReportConfig,
) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  x, x_label = _x_axis(data, indices)
  fig, axes = plt.subplots(4, 1, figsize=(10.0, 8.5), dpi=cfg.dpi, sharex=True)

  mode = _series(data, "gate_mode", indices)
  if mode is None:
    mode = _series(data, "gate_state", indices)
  if mode is not None:
    axes[0].step(x, mode, where="post", color="#111111", label="gate mode")
  _plot_event_markers(axes[0], data, indices, x)
  axes[0].set_yticks([0.0, 1.0, 2.0])
  axes[0].set_yticklabels(["normal", "write", "memory"])
  axes[0].set_ylabel("state")
  axes[0].legend(frameon=False, loc="upper right", ncols=4, fontsize=8)

  for key, label, color in (
    ("event_prob", "event", "#d55e00"),
    ("stair_prob", "stair", "#0072b2"),
    ("future_risk", "risk", "#cc79a7"),
    ("future_quality", "quality", "#009e73"),
  ):
    values = _series(data, key, indices)
    if values is not None:
      axes[1].plot(x, values, label=label, color=color, linewidth=1.4)
  axes[1].set_ylim(-0.05, 1.05)
  axes[1].set_ylabel("prob.")
  axes[1].legend(frameon=False, loc="upper right", ncols=4, fontsize=8)

  for key, label, color in (
    ("alpha_state", "alpha state", "#0072b2"),
    ("alpha_shape", "alpha shape", "#009e73"),
  ):
    values = _series(data, key, indices)
    if values is not None:
      axes[2].plot(x, values, label=label, color=color, linewidth=1.2)
  for values, label, color in (
    (_delta_norm(_matrix(data, "z_state_memory", indices)), "||dz state||", "#56b4e9"),
    (_delta_norm(_matrix(data, "z_shape_memory", indices)), "||dz shape||", "#f0e442"),
  ):
    if values is not None:
      axes[2].plot(x, values, label=label, color=color, linewidth=1.1, alpha=0.85)
  axes[2].set_ylabel("update")
  axes[2].legend(frameon=False, loc="upper right", ncols=4, fontsize=8)

  interval = _matrix(data, "safe_stride_interval", indices)
  if interval is not None and interval.shape[-1] >= 2:
    axes[3].fill_between(
      x,
      interval[:, 0],
      interval[:, 1],
      color="#009e73",
      alpha=0.14,
      label="safe interval",
    )
  for key, label, component, color in (
    ("stair_shape", "pred stride", 0, "#0072b2"),
    ("stair_shape", "pred riser", 1, "#d55e00"),
    ("safe_stride", "safe stride", 0, "#009e73"),
    ("safe_stride_confidence", "confidence", 0, "#666666"),
  ):
    values = _series(data, key, indices, component=component)
    if values is not None:
      axes[3].plot(x, values, label=label, color=color, linewidth=1.3)
  true_height = _truth_value(data, "terrain_height_m", indices)
  true_depth = _truth_value(data, "terrain_tread_depth_m", indices)
  if true_height > 0.0:
    axes[3].axhline(
      true_height,
      color="#d55e00",
      linestyle="--",
      linewidth=1.0,
      alpha=0.65,
      label="true riser",
    )
  if true_depth > 0.0:
    axes[3].axhline(
      true_depth,
      color="#0072b2",
      linestyle="--",
      linewidth=1.0,
      alpha=0.65,
      label="true tread",
    )
  axes[3].set_ylabel("meters / unit")
  axes[3].set_xlabel(x_label)
  axes[3].legend(frameon=False, loc="upper right", ncols=4, fontsize=8)

  for ax in axes:
    ax.grid(True, alpha=0.20)
  fig.suptitle(
    "Latent state switch and predictions | "
    f"episode {selection.episode_id} | {selection.terrain_label}",
    y=0.995,
  )
  fig.tight_layout()
  fig.savefig(output_path)
  plt.close(fig)


def _semantic_key(data: np.lib.npyio.NpzFile) -> str | None:
  if "semantic" in data.files:
    return "semantic"
  if "shadow_semantic" in data.files:
    return "shadow_semantic"
  return None


def _plot_semantic(
  data: np.lib.npyio.NpzFile,
  indices: np.ndarray,
  selection: TraceSelection,
  output_path: Path,
  cfg: PlotStairLatentReportConfig,
) -> bool:
  key = _semantic_key(data)
  if key is None:
    return False

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  values = _matrix(data, key, indices)
  if values is None or values.shape[-1] == 0:
    return False
  x, x_label = _x_axis(data, indices)
  extent = (float(x[0]), float(x[-1]), -0.5, float(values.shape[-1]) - 0.5)

  fig, ax = plt.subplots(figsize=(10.0, 5.8), dpi=cfg.dpi)
  image = ax.imshow(
    values.T,
    aspect="auto",
    origin="lower",
    interpolation="nearest",
    vmin=0.0,
    vmax=1.0,
    cmap="viridis",
    extent=extent,
  )
  ax.axhline(7.5, color="white", linewidth=0.8, alpha=0.75)
  labels = list(SEMANTIC_CHANNELS[: values.shape[-1]])
  ax.set_yticks(np.arange(len(labels)))
  ax.set_yticklabels(labels, fontsize=8)
  ax.set_xlabel(x_label)
  ax.set_title(
    f"Semantic-v2 dynamics | episode {selection.episode_id} | {selection.terrain_label}"
  )
  fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="normalized value")
  fig.tight_layout()
  fig.savefig(output_path)
  plt.close(fig)
  return True


def _robust_limits(values: np.ndarray) -> tuple[float, float]:
  finite = values[np.isfinite(values)]
  if finite.size == 0:
    return -1.0, 1.0
  lo, hi = np.percentile(finite, [2.0, 98.0])
  if not np.isfinite(lo) or not np.isfinite(hi) or abs(float(hi - lo)) < 1.0e-8:
    center = float(np.median(finite))
    return center - 1.0, center + 1.0
  return float(lo), float(hi)


def _plot_z_memory_heatmap(
  data: np.lib.npyio.NpzFile,
  indices: np.ndarray,
  selection: TraceSelection,
  output_path: Path,
  cfg: PlotStairLatentReportConfig,
) -> bool:
  panels = []
  for key, title in (
    ("z_state_memory", "z_state memory"),
    ("z_shape_memory", "z_shape memory"),
    ("z_candidate", "z candidate"),
    ("encoder_out", "encoder output"),
  ):
    values = _matrix(data, key, indices)
    if values is not None and values.shape[-1] > 0:
      panels.append((key, title, values))
  if not panels:
    return False

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  x, x_label = _x_axis(data, indices)
  fig, axes = plt.subplots(
    len(panels),
    1,
    figsize=(10.0, 2.35 * len(panels)),
    dpi=cfg.dpi,
    sharex=True,
  )
  axes_list = np.atleast_1d(axes).tolist()
  extent_base = (float(x[0]), float(x[-1]))
  for ax, (_key, title, values) in zip(axes_list, panels, strict=True):
    lo, hi = _robust_limits(values)
    image = ax.imshow(
      values.T,
      aspect="auto",
      origin="lower",
      interpolation="nearest",
      cmap="coolwarm",
      vmin=lo,
      vmax=hi,
      extent=(extent_base[0], extent_base[1], -0.5, values.shape[-1] - 0.5),
    )
    _plot_event_markers(ax, data, indices, x)
    ax.set_ylabel(title)
    ax.set_yticks([0, values.shape[-1] - 1] if values.shape[-1] > 1 else [0])
    fig.colorbar(image, ax=ax, fraction=0.020, pad=0.01)
  axes_list[-1].set_xlabel(x_label)
  fig.suptitle(
    "SlowLatent representation dynamics | "
    f"episode {selection.episode_id} | {selection.terrain_label}",
    y=0.995,
  )
  fig.tight_layout()
  fig.savefig(output_path)
  plt.close(fig)
  return True


def _gate_mode_matrix(
  data: np.lib.npyio.NpzFile,
  cfg: PlotStairLatentReportConfig,
) -> tuple[np.ndarray, list[str], np.ndarray]:
  if "gate_mode" not in data.files:
    return np.zeros((0, 0), dtype=np.float32), [], np.zeros(0, dtype=np.float32)
  episode_ids = _episode_array(data)
  labels = _string_array(data, "terrain_label", default="sample")
  mode = np.asarray(data["gate_mode"], dtype=np.float32).reshape(-1)
  times, _x_label = _x_axis(data, np.arange(_row_count(data)))
  groups = sorted(
    {
      (str(label), int(episode_id))
      for label, episode_id in zip(labels, episode_ids, strict=True)
    }
  )
  if cfg.raster_max_episodes > 0:
    groups = groups[: cfg.raster_max_episodes]
  if not groups:
    return np.zeros((0, 0), dtype=np.float32), [], np.zeros(0, dtype=np.float32)

  max_len = 0
  ordered_indices = []
  row_labels = []
  for label, episode_id in groups:
    idx = np.nonzero((labels == label) & (episode_ids == episode_id))[0]
    if idx.size == 0:
      continue
    order = np.argsort(times[idx])
    idx = idx[order]
    ordered_indices.append(idx)
    row_labels.append(f"{label}:{episode_id}")
    max_len = max(max_len, idx.size)
  matrix = np.full((len(ordered_indices), max_len), np.nan, dtype=np.float32)
  time_axis = np.arange(max_len, dtype=np.float32)
  for row, idx in enumerate(ordered_indices):
    matrix[row, : idx.size] = mode[idx]
    if row == 0:
      time_axis[: idx.size] = times[idx]
  return matrix, row_labels, time_axis


def _plot_gate_raster(
  data: np.lib.npyio.NpzFile,
  output_path: Path,
  cfg: PlotStairLatentReportConfig,
) -> bool:
  matrix, row_labels, time_axis = _gate_mode_matrix(data, cfg)
  if matrix.size == 0:
    return False

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.colors import BoundaryNorm, ListedColormap

  cmap = ListedColormap(["#d9d9d9", "#fdae61", "#2c7fb8"])
  norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
  fig, ax = plt.subplots(figsize=(10.0, 8.0), dpi=cfg.dpi)
  masked = np.ma.masked_invalid(np.rint(matrix))
  ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
  ax.set_title("Gate mode raster")
  ax.set_xlabel("time index")
  ax.set_ylabel("episode")
  if time_axis.size > 1 and np.isfinite(time_axis).any():
    tick_count = min(6, time_axis.size)
    ticks = np.linspace(0, time_axis.size - 1, tick_count).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{time_axis[tick]:.1f}" for tick in ticks])
    ax.set_xlabel("time (s)")
  if len(row_labels) <= 35:
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=6)
  else:
    ax.set_yticks([])
  handles = [
    plt.Line2D([0], [0], marker="s", linestyle="", color=color, label=label)
    for color, label in (
      ("#d9d9d9", "normal"),
      ("#fdae61", "write"),
      ("#2c7fb8", "memory"),
    )
  ]
  ax.legend(handles=handles, frameon=False, loc="upper right", ncols=3)
  fig.tight_layout()
  fig.savefig(output_path)
  plt.close(fig)
  return True


def _semantic_invariant_report(
  data: np.lib.npyio.NpzFile,
) -> dict[str, float | int | str]:
  key = _semantic_key(data)
  if key is None:
    return {"available": 0}
  semantic = np.asarray(data[key], dtype=np.float32)
  report: dict[str, float | int | str] = {"available": 1, "source": key}
  if semantic.shape[-1] < 16:
    report["channels"] = int(semantic.shape[-1])
    return report
  mode_sum = semantic[:, 2:5].sum(axis=1)
  report["mode_onehot_mean_abs_error"] = float(np.mean(np.abs(mode_sum - 1.0)))
  report["mode_onehot_max_abs_error"] = float(np.max(np.abs(mode_sum - 1.0)))
  report["semantic_8_12_max_abs_diff"] = float(
    np.max(np.abs(semantic[:, 8] - semantic[:, 12]))
  )
  report["semantic_8_12_corr"] = _safe_corr(semantic[:, 8], semantic[:, 12])
  lower = semantic[:, 10]
  upper = semantic[:, 11]
  width = semantic[:, 13]
  report["semantic_lower_gt_upper_count"] = int(np.count_nonzero(lower > upper + 1e-5))
  report["semantic_width_mean_abs_error"] = float(
    np.mean(np.abs(width - (upper - lower)))
  )
  if "gate_mode" in data.files:
    mode = np.asarray(data["gate_mode"], dtype=np.float32).reshape(-1)
    expected = np.zeros((mode.shape[0], 3), dtype=np.float32)
    expected[:, 0] = mode < 0.5
    expected[:, 1] = (mode >= 0.5) & (mode < 1.5)
    expected[:, 2] = mode >= 1.5
    report["mode_vs_gate_max_abs_error"] = float(
      np.max(np.abs(semantic[:, 2:5] - expected))
    )
  if "safe_stride_interval" in data.files:
    interval = np.asarray(data["safe_stride_interval"], dtype=np.float32)
    report["interval_lower_gt_upper_count"] = int(
      np.count_nonzero(interval[:, 0] > interval[:, 1] + 1e-5)
    )
  return report


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
  valid = np.isfinite(a) & np.isfinite(b)
  if np.count_nonzero(valid) < 2:
    return float("nan")
  a_valid = a[valid]
  b_valid = b[valid]
  a_std = float(np.std(a_valid))
  b_std = float(np.std(b_valid))
  if a_std < 1.0e-12 or b_std < 1.0e-12:
    return 1.0 if float(np.max(np.abs(a_valid - b_valid))) < 1.0e-12 else float("nan")
  a_norm = (a_valid - a_valid.mean()) / a_std
  b_norm = (b_valid - b_valid.mean()) / b_std
  return float(np.mean(a_norm * b_norm))


def _column_correlation(values: np.ndarray) -> np.ndarray:
  centered = values - np.nanmean(values, axis=0, keepdims=True)
  centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)
  norm = np.linalg.norm(centered, axis=0)
  denom = norm[:, None] * norm[None, :]
  corr = np.divide(
    centered.T @ centered,
    denom,
    out=np.zeros((values.shape[1], values.shape[1]), dtype=np.float64),
    where=denom > 1.0e-12,
  )
  np.fill_diagonal(corr, 1.0)
  return corr


def _plot_semantic_correlation(
  data: np.lib.npyio.NpzFile,
  output_path: Path,
  cfg: PlotStairLatentReportConfig,
) -> bool:
  key = _semantic_key(data)
  if key is None:
    return False
  semantic = np.asarray(data[key], dtype=np.float32)
  if semantic.ndim != 2 or semantic.shape[-1] == 0:
    return False
  indices = _sample_indices(
    np.arange(semantic.shape[0]),
    max_samples=cfg.max_samples,
    seed=cfg.seed,
  )
  values = semantic[indices]
  corr = _column_correlation(values)

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  labels = list(SEMANTIC_CHANNELS[: corr.shape[0]])
  fig, ax = plt.subplots(figsize=(8.0, 7.0), dpi=cfg.dpi)
  image = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
  ax.set_xticks(np.arange(len(labels)))
  ax.set_yticks(np.arange(len(labels)))
  ax.set_xticklabels(labels, rotation=90, fontsize=7)
  ax.set_yticklabels(labels, fontsize=7)
  ax.set_title("Semantic-v2 channel correlation")
  fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
  fig.tight_layout()
  fig.savefig(output_path)
  plt.close(fig)
  return True


def _nearest_centroid_accuracy(
  features: np.ndarray,
  labels: np.ndarray,
  *,
  max_samples: int | None,
  seed: int,
) -> float:
  valid = _finite_feature_rows(features) & np.asarray([label != "" for label in labels])
  indices = np.nonzero(valid)[0]
  indices = _sample_indices(indices, max_samples=max_samples, seed=seed)
  if indices.size < 4 or len(set(labels[indices].tolist())) < 2:
    return float("nan")
  rng = np.random.default_rng(seed)
  shuffled = indices.copy()
  rng.shuffle(shuffled)
  split = max(1, int(0.7 * shuffled.size))
  train = shuffled[:split]
  test = shuffled[split:]
  if test.size == 0:
    return float("nan")
  mean = features[train].mean(axis=0, keepdims=True)
  std = features[train].std(axis=0, keepdims=True) + 1.0e-6
  train_x = (features[train] - mean) / std
  test_x = (features[test] - mean) / std
  centroids = []
  classes = []
  for label in sorted(set(labels[train].tolist())):
    mask = labels[train] == label
    if not np.any(mask):
      continue
    centroids.append(train_x[mask].mean(axis=0))
    classes.append(label)
  if len(centroids) < 2:
    return float("nan")
  centroid_matrix = np.stack(centroids, axis=0)
  distances = ((test_x[:, None, :] - centroid_matrix[None, :, :]) ** 2).sum(axis=-1)
  pred = np.asarray(classes)[np.argmin(distances, axis=1)]
  return float(np.mean(pred == labels[test]))


def _linear_probe(
  features: np.ndarray,
  target: np.ndarray,
  mask: np.ndarray,
  *,
  max_samples: int | None,
  seed: int,
) -> tuple[float, float]:
  valid = mask & _finite_feature_rows(features) & np.isfinite(target)
  indices = np.nonzero(valid)[0]
  indices = _sample_indices(indices, max_samples=max_samples, seed=seed)
  if indices.size < 8 or float(np.std(target[indices])) < 1.0e-8:
    return float("nan"), float("nan")
  rng = np.random.default_rng(seed)
  shuffled = indices.copy()
  rng.shuffle(shuffled)
  split = max(1, int(0.7 * shuffled.size))
  train = shuffled[:split]
  test = shuffled[split:]
  if test.size == 0:
    return float("nan"), float("nan")
  mean = features[train].mean(axis=0, keepdims=True)
  std = features[train].std(axis=0, keepdims=True) + 1.0e-6
  x_train = (features[train] - mean) / std
  x_test = (features[test] - mean) / std
  x_train = np.c_[np.ones(x_train.shape[0]), x_train]
  x_test = np.c_[np.ones(x_test.shape[0]), x_test]
  coef = np.linalg.lstsq(x_train, target[train], rcond=None)[0]
  pred = x_test @ coef
  residual = float(np.sum((target[test] - pred) ** 2))
  total = float(np.sum((target[test] - target[test].mean()) ** 2))
  r2 = 1.0 - residual / total if total > 1.0e-12 else float("nan")
  mae = float(np.mean(np.abs(target[test] - pred)))
  return r2, mae


def _phase_bin_labels(data: np.lib.npyio.NpzFile, bins: int = 8) -> np.ndarray:
  if "gait_phase" not in data.files:
    return np.full(_row_count(data), "", dtype=str)
  phase = np.asarray(data["gait_phase"], dtype=np.float32) % 1.0
  ids = np.floor(phase * bins).astype(np.int64).clip(0, bins - 1)
  return np.asarray([f"phase_{idx}" for idx in ids], dtype=str)


def _mode_labels(data: np.lib.npyio.NpzFile) -> np.ndarray:
  if "gate_mode" not in data.files:
    return np.full(_row_count(data), "", dtype=str)
  mode = np.asarray(data["gate_mode"], dtype=np.float32).reshape(-1)
  labels = np.full(mode.shape[0], "normal", dtype=object)
  labels[(mode >= 0.5) & (mode < 1.5)] = "write"
  labels[mode >= 1.5] = "memory"
  return labels.astype(str)


def _write_representation_probe_summary(
  data: np.lib.npyio.NpzFile,
  output_csv: Path,
  output_plot: Path,
  cfg: PlotStairLatentReportConfig,
) -> list[dict[str, float | str]]:
  rows: list[dict[str, float | str]] = []
  terrain_labels = _size_labels(data)
  phase_labels = _phase_bin_labels(data)
  mode_labels = _mode_labels(data)
  kinds = _string_array(data, "terrain_kind", default="")
  true_height = _float_array(data, "terrain_height_m", default=0.0)
  true_depth = _float_array(data, "terrain_tread_depth_m", default=0.0)
  phase_mask = _sample_filter(data, cfg)
  if "gate_mode" in data.files:
    memory_mask = np.asarray(data["gate_mode"], dtype=np.float32).reshape(-1) >= 1.5
  else:
    memory_mask = np.ones(_row_count(data), dtype=bool)
  stair_probe_mask = phase_mask & (kinds == "upstairs") & memory_mask

  features_to_score = [
    key
    for key in (
      "encoder_out",
      "h_t",
      "hidden",
      "z_candidate",
      "z_memory",
      "z_state_memory",
      "z_shape_memory",
    )
    if key in data.files
  ]
  for key in features_to_score:
    features = np.asarray(data[key], dtype=np.float32)
    if features.ndim == 1:
      features = features[:, None]
    height_r2, height_mae = _linear_probe(
      features,
      true_height,
      stair_probe_mask,
      max_samples=cfg.max_samples,
      seed=cfg.seed,
    )
    depth_r2, depth_mae = _linear_probe(
      features,
      true_depth,
      stair_probe_mask,
      max_samples=cfg.max_samples,
      seed=cfg.seed + 1,
    )
    rows.append(
      {
        "representation": key,
        "height_r2": height_r2,
        "height_mae_m": height_mae,
        "depth_r2": depth_r2,
        "depth_mae_m": depth_mae,
        "terrain_centroid_acc": _nearest_centroid_accuracy(
          features[phase_mask],
          terrain_labels[phase_mask],
          max_samples=cfg.max_samples,
          seed=cfg.seed,
        ),
        "gait_phase_centroid_acc": _nearest_centroid_accuracy(
          features,
          phase_labels,
          max_samples=cfg.max_samples,
          seed=cfg.seed + 2,
        ),
        "gate_mode_centroid_acc": _nearest_centroid_accuracy(
          features[phase_mask],
          mode_labels[phase_mask],
          max_samples=cfg.max_samples,
          seed=cfg.seed + 3,
        ),
      }
    )

  fieldnames = [
    "representation",
    "height_r2",
    "height_mae_m",
    "depth_r2",
    "depth_mae_m",
    "terrain_centroid_acc",
    "gait_phase_centroid_acc",
    "gate_mode_centroid_acc",
  ]
  with output_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

  if rows:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_names = [
      "height_r2",
      "depth_r2",
      "terrain_centroid_acc",
      "gait_phase_centroid_acc",
      "gate_mode_centroid_acc",
    ]
    matrix = np.asarray(
      [[float(row[name]) for name in metric_names] for row in rows],
      dtype=np.float32,
    )
    matrix = np.nan_to_num(matrix, nan=0.0)
    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(rows) + 2.5), dpi=cfg.dpi)
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([str(row["representation"]) for row in rows], fontsize=8)
    ax.set_title("Representation probe summary")
    for row_idx in range(matrix.shape[0]):
      for col_idx in range(matrix.shape[1]):
        ax.text(
          col_idx,
          row_idx,
          f"{matrix[row_idx, col_idx]:.2f}",
          ha="center",
          va="center",
          color="white" if matrix[row_idx, col_idx] < 0.55 else "black",
          fontsize=7,
        )
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_plot)
    plt.close(fig)
  return rows


def _plot_head_prediction_summary(
  data: np.lib.npyio.NpzFile,
  output_path: Path,
  cfg: PlotStairLatentReportConfig,
) -> bool:
  if "stair_shape" not in data.files:
    return False
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  kinds = _string_array(data, "terrain_kind", default="")
  true_height = _float_array(data, "terrain_height_m", default=0.0)
  true_depth = _float_array(data, "terrain_tread_depth_m", default=0.0)
  if "gate_mode" in data.files:
    memory = np.asarray(data["gate_mode"], dtype=np.float32).reshape(-1) >= 1.5
  else:
    memory = np.ones(_row_count(data), dtype=bool)
  mask = (kinds == "upstairs") & memory & _sample_filter(data, cfg)
  indices = _sample_indices(
    np.nonzero(mask)[0],
    max_samples=cfg.max_samples,
    seed=cfg.seed,
  )
  if indices.size == 0:
    return False
  shape = np.asarray(data["stair_shape"], dtype=np.float32)
  safe_stride = _series(data, "safe_stride", indices)
  confidence = _series(data, "safe_stride_confidence", indices)
  interval = _matrix(data, "safe_stride_interval", indices)
  labels = _size_labels(data, indices)

  fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), dpi=cfg.dpi)
  ax = axes[0, 0]
  pred_h = shape[indices, 1]
  ax.scatter(true_height[indices], pred_h, s=8, alpha=0.45, linewidths=0)
  lo = min(float(np.nanmin(true_height[indices])), float(np.nanmin(pred_h)))
  hi = max(float(np.nanmax(true_height[indices])), float(np.nanmax(pred_h)))
  ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--")
  ax.set_xlabel("true riser height (m)")
  ax.set_ylabel("predicted riser height (m)")
  ax.set_title("Riser-height head")

  ax = axes[0, 1]
  if safe_stride is not None:
    ax.scatter(true_depth[indices], safe_stride, s=8, alpha=0.45, linewidths=0)
  ax.set_xlabel("true tread depth (m)")
  ax.set_ylabel("safe stride control (m)")
  ax.set_title("SafeStride control")

  ax = axes[1, 0]
  unique_labels = list(dict.fromkeys(labels.tolist()))
  x = np.arange(len(unique_labels))
  if interval is not None and interval.shape[-1] >= 2:
    lower = [
      float(np.nanmedian(interval[labels == label, 0])) for label in unique_labels
    ]
    upper = [
      float(np.nanmedian(interval[labels == label, 1])) for label in unique_labels
    ]
    ax.vlines(x, lower, upper, color="#009e73", linewidth=4, alpha=0.7)
  if safe_stride is not None:
    control = [
      float(np.nanmedian(safe_stride[labels == label])) for label in unique_labels
    ]
    ax.scatter(x, control, color="#111111", s=18, label="control")
  ax.set_xticks(x)
  ax.set_xticklabels(unique_labels, rotation=45, ha="right", fontsize=7)
  ax.set_ylabel("meters")
  ax.set_title("Median SafeStride interval by terrain")
  ax.legend(frameon=False, fontsize=8)

  ax = axes[1, 1]
  if confidence is not None:
    means = [
      float(np.nanmedian(confidence[labels == label])) for label in unique_labels
    ]
    ax.bar(x, means, color="#756bb1", alpha=0.8)
  ax.set_ylim(0.0, 1.05)
  ax.set_xticks(x)
  ax.set_xticklabels(unique_labels, rotation=45, ha="right", fontsize=7)
  ax.set_ylabel("confidence")
  ax.set_title("SafeStride confidence")

  for axis in axes.flat:
    axis.grid(True, alpha=0.2)
  fig.tight_layout()
  fig.savefig(output_path)
  plt.close(fig)
  return True


def _write_trace_csv(
  data: np.lib.npyio.NpzFile,
  indices: np.ndarray,
  output_path: Path,
) -> None:
  columns: list[tuple[str, np.ndarray]] = []
  for key in ("time_s", "time_step", "episode_id"):
    if key in data.files:
      columns.append((key, np.asarray(data[key])[indices]))
  for key in ("terrain_label", "terrain_height_m", "terrain_tread_depth_m"):
    if key in data.files:
      columns.append((key, np.asarray(data[key])[indices]))
  for key, component, name in (
    ("gate_mode", 0, "gate_mode"),
    ("event_prob", 0, "event_prob"),
    ("stair_prob", 0, "stair_prob"),
    ("stair_shape", 0, "pred_same_foot_stride_m"),
    ("stair_shape", 1, "pred_riser_height_m"),
    ("safe_stride", 0, "pred_safe_stride_m"),
    ("safe_stride_confidence", 0, "safe_stride_confidence"),
    ("alpha_state", 0, "alpha_state"),
    ("alpha_shape", 0, "alpha_shape"),
  ):
    values = _series(data, key, indices, component=component)
    if values is not None:
      columns.append((name, values))
  interval = _matrix(data, "safe_stride_interval", indices)
  if interval is not None and interval.shape[-1] >= 2:
    columns.append(("safe_stride_lower_m", interval[:, 0]))
    columns.append(("safe_stride_upper_m", interval[:, 1]))
  semantic = _matrix(data, _semantic_key(data) or "", indices)
  if semantic is not None:
    for idx, name in enumerate(SEMANTIC_CHANNELS[: semantic.shape[-1]]):
      columns.append((f"semantic_{name}", semantic[:, idx]))

  with output_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([name for name, _values in columns])
    for row_idx in range(indices.size):
      writer.writerow([values[row_idx] for _name, values in columns])


def run_plot_report(cfg: PlotStairLatentReportConfig) -> dict[str, object]:
  input_path = Path(cfg.input_file)
  output_dir = Path(cfg.output_dir) if cfg.output_dir is not None else input_path.parent
  output_dir.mkdir(parents=True, exist_ok=True)
  _progress(cfg, f"Loading latent dataset from {input_path}.")
  data = np.load(input_path, allow_pickle=False)

  summary: dict[str, object] = {
    "input_file": str(input_path),
    "output_dir": str(output_dir),
    "available_arrays": sorted(data.files),
    "sample_filter": {
      "phase_bins": cfg.phase_bins,
      "phase_bin": cfg.phase_bin,
      "terrain_kind_filter": cfg.terrain_kind_filter,
      "gate_mode_filter": cfg.gate_mode_filter,
      "min_memory_age": cfg.min_memory_age,
    },
  }

  selection = _select_trace(data, cfg)
  trace_indices = _trace_indices(data, selection, cfg)
  summary["trace_selection"] = {
    "episode_id": selection.episode_id,
    "terrain_label": selection.terrain_label,
    "samples": int(trace_indices.size),
  }
  state_path = output_dir / "latent_state_switch_episode.png"
  _progress(cfg, "Plotting state switch episode.")
  _plot_state_switch(data, trace_indices, selection, state_path, cfg)
  summary["latent_state_switch_file"] = str(state_path)

  semantic_path = output_dir / "semantic_dynamics_episode.png"
  _progress(cfg, "Plotting Semantic-v2 dynamics.")
  if _plot_semantic(data, trace_indices, selection, semantic_path, cfg):
    summary["semantic_dynamics_file"] = str(semantic_path)

  z_heatmap_path = output_dir / "z_memory_heatmap_episode.png"
  _progress(cfg, "Plotting representation heatmaps.")
  if _plot_z_memory_heatmap(data, trace_indices, selection, z_heatmap_path, cfg):
    summary["z_memory_heatmap_file"] = str(z_heatmap_path)

  gate_raster_path = output_dir / "gate_state_raster.png"
  _progress(cfg, "Plotting gate raster.")
  if _plot_gate_raster(data, gate_raster_path, cfg):
    summary["gate_state_raster_file"] = str(gate_raster_path)

  semantic_corr_path = output_dir / "semantic_correlation_matrix.png"
  _progress(cfg, "Plotting semantic correlation matrix.")
  if _plot_semantic_correlation(data, semantic_corr_path, cfg):
    summary["semantic_correlation_file"] = str(semantic_corr_path)
  invariant_report = _semantic_invariant_report(data)
  invariant_path = output_dir / "semantic_invariant_report.json"
  invariant_path.write_text(
    json.dumps(_json_ready(invariant_report), indent=2),
    encoding="utf-8",
  )
  summary["semantic_invariant_report"] = invariant_report
  summary["semantic_invariant_report_file"] = str(invariant_path)

  head_summary_path = output_dir / "head_prediction_summary.png"
  _progress(cfg, "Plotting head prediction summary.")
  if _plot_head_prediction_summary(data, head_summary_path, cfg):
    summary["head_prediction_summary_file"] = str(head_summary_path)

  if cfg.write_trace_csv:
    trace_csv = output_dir / "latent_state_switch_episode.csv"
    _progress(cfg, "Writing episode trace CSV.")
    _write_trace_csv(data, trace_indices, trace_csv)
    summary["trace_csv_file"] = str(trace_csv)

  cluster_summaries = []
  for feature_key, (filename, title) in CLUSTER_FEATURES.items():
    if feature_key not in data.files:
      continue
    _progress(cfg, f"Plotting PCA cluster for {feature_key}.")
    cluster_path = output_dir / filename
    cluster_summaries.append(
      _plot_cluster(data, feature_key, cluster_path, cfg, f"{title} by stair size")
    )
  summary["cluster_plots"] = cluster_summaries

  interactive_summaries = []
  if cfg.write_interactive_html:
    interactive_dir = output_dir / "interactive"
    static_dir = output_dir / "static"
    for feature_key in _interactive_feature_keys(data, cfg):
      interactive_summaries.append(
        _write_interactive_embedding(
          data,
          feature_key,
          interactive_dir,
          static_dir,
          cfg,
        )
      )
  summary["interactive_embeddings"] = interactive_summaries

  probe_csv = output_dir / "representation_probe_summary.csv"
  probe_plot = output_dir / "representation_probe_summary.png"
  probe_rows = _write_representation_probe_summary(data, probe_csv, probe_plot, cfg)
  summary["representation_probe_summary_file"] = str(probe_csv)
  if probe_plot.exists():
    summary["representation_probe_summary_plot"] = str(probe_plot)
  summary["representation_probe_summary"] = probe_rows

  summary_path = output_dir / "latent_report_summary.json"
  summary_path.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
  print(f"[INFO] Wrote latent report to {output_dir}")
  print(f"[INFO] Wrote summary to {summary_path}")
  return summary


def main() -> None:
  cfg = tyro.cli(PlotStairLatentReportConfig)
  run_plot_report(cfg)


if __name__ == "__main__":
  main()
