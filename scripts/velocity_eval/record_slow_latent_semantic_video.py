"""Record a SlowLatent stair rollout with semantic state overlays."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mediapy as media
import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw, ImageFont
from scripts.velocity_eval.collect_policy_latents import (
  _policy_diagnostic_tensors,
  _policy_state_tensors,
)
from scripts.velocity_eval.eval_metrics import StairEventDetector, StairMetricParams
from scripts.velocity_eval.eval_terrains import (
  EvalTerrainSpec,
  apply_eval_overrides,
  get_terrain_set,
)
from scripts.velocity_eval.policy_io import (
  get_clip_actions,
  load_inference_policy,
  make_timestamped_policy_output_dir,
  resolve_checkpoint_path,
  resolve_inference_agent_cfg,
)

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.lstm import reset_policy_state_from_step
from mjlab.utils.torch import configure_torch_backends

MODE_COLORS = {
  "NORMAL": (86, 108, 126),
  "WRITE": (220, 145, 35),
  "MEMORY": (34, 148, 118),
}
TREND_COLORS = {
  "BACKOFF": (204, 81, 60),
  "HOLD": (89, 110, 196),
  "FORWARD": (36, 153, 83),
  "UNKNOWN": (125, 125, 125),
}
G1_FOOT_LENGTH_M = 0.186
G1_FOOT_HEIGHT_M = 0.045
ImageFontLike = ImageFont.ImageFont | ImageFont.FreeTypeFont


@dataclass(frozen=True)
class SlowLatentSemanticVideoConfig:
  checkpoint_file: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  terrain_set: str = "long_stair_riser_grid_v1"
  terrain_label: str | None = "h15"
  terrain_name: str | None = None
  num_envs: int = 1
  max_episode_length_s: float = 18.0
  command_vx: float = 0.4
  command_vy: float = 0.0
  command_wz: float = 0.0
  seed: int = 34567
  device: str | None = None
  output_root: str = "eval_outputs/velocity"
  output_dir: str | None = None
  output_file: str | None = None
  trace_file: str | None = None
  render_every: int = 2
  progress_interval_steps: int = 100
  fps: float | None = None
  video_width: int = 1280
  video_height: int = 720
  hud_width: int = 360
  side_panel_height: int = 180
  camera_body_name: str = "torso_link"
  camera_distance: float = 4.1
  camera_elevation: float = -9.0
  camera_azimuth: float = 72.0
  camera_fovy: float | None = 52.0
  show_toe_riser_contact_markers: bool = True
  toe_riser_contact_marker_radius: float = 0.06
  toe_riser_contact_marker_x_offset_m: float = -0.045
  toe_riser_contact_marker_y_offset_m: float = 0.0
  toe_riser_contact_marker_z_offset_m: float = 0.035
  toe_riser_contact_marker_max_points: int = 256
  toe_riser_contact_force_threshold: float = 5.0
  toe_riser_contact_cooldown_s: float = 0.10
  ignore_passed_support_riser_toe_contacts: bool = True
  stop_after_stair_completion: bool = True
  stair_completion_margin_m: float = 0.0
  post_stair_completion_time_s: float = 2.0
  clean_observations: bool = True
  disable_observation_delay: bool = True
  disable_actuator_delay: bool = True


@dataclass(frozen=True)
class FootSnapshot:
  name: str
  x_m: float
  z_m: float
  contact: bool | None
  swing: bool | None


@dataclass(frozen=True)
class SemanticSnapshot:
  step: int
  time_s: float
  root_x_m: float
  root_z_m: float
  mode: str
  mode_code: float | None
  event_prob: float | None
  stair_prob: float | None
  write_progress: float | None
  memory_age: float | None
  release_progress: float | None
  pred_riser_height_m: float | None
  true_riser_height_m: float | None
  safe_stride_m: float | None
  stride_lower_m: float | None
  stride_upper_m: float | None
  stride_trend: float | None
  trend_label: str
  trend_probs: tuple[float | None, float | None, float | None]
  confidence: float | None
  alpha_state: float | None
  alpha_shape: float | None
  foot_left: FootSnapshot | None
  foot_right: FootSnapshot | None
  toe_riser_contacts_this_step: int = 0
  toe_riser_contact_markers_total: int = 0
  toe_riser_contact_xz_m: tuple[tuple[float, float], ...] = ()


class ToeRiserContactMarkerOverlay:
  """Persist eval-filtered toe-riser contact markers for video rendering."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    detector: StairEventDetector,
    *,
    radius: float,
    display_offset_world: tuple[float, float, float] = (0.0, 0.0, 0.0),
    max_points_per_env: int,
  ) -> None:
    self._detector = detector
    self._radius = float(radius)
    self._display_offset_world = np.asarray(display_offset_world, dtype=np.float32)
    self._max_points_per_env = max(0, int(max_points_per_env))
    self._raw_points_by_env: list[list[np.ndarray]] = [[] for _ in range(env.num_envs)]
    self._display_points_by_env: list[list[np.ndarray]] = [
      [] for _ in range(env.num_envs)
    ]
    self._last_new_counts = [0 for _ in range(env.num_envs)]

  @property
  def event_source(self) -> str:
    return self._detector.event_source

  def update(self, env: ManagerBasedRlEnv) -> int:
    self._last_new_counts = [0 for _ in range(env.num_envs)]
    if self._max_points_per_env <= 0 or self._detector.event_source != "true_contact":
      return 0

    self._detector.compute_events(env)
    env_ids, positions_w = self._detector.get_last_true_contact_positions("toe")
    if env_ids.numel() == 0:
      return 0

    total_new = 0
    for env_id, position in zip(
      env_ids.detach().cpu().tolist(),
      positions_w.detach().cpu().numpy(),
      strict=True,
    ):
      env_index = int(env_id)
      raw_position = np.asarray(position, dtype=np.float32).copy()
      display_position = raw_position + self._display_offset_world
      raw_points = self._raw_points_by_env[env_index]
      display_points = self._display_points_by_env[env_index]
      raw_points.append(raw_position)
      display_points.append(display_position)
      if len(raw_points) > self._max_points_per_env:
        overflow = len(raw_points) - self._max_points_per_env
        del raw_points[:overflow]
        del display_points[:overflow]
      self._last_new_counts[env_index] += 1
      total_new += 1
    return total_new

  def new_count(self, env_id: int = 0) -> int:
    if not 0 <= env_id < len(self._last_new_counts):
      return 0
    return self._last_new_counts[env_id]

  def total_count(self, env_id: int = 0) -> int:
    if not 0 <= env_id < len(self._raw_points_by_env):
      return 0
    return len(self._raw_points_by_env[env_id])

  def xz_points(
    self,
    env: ManagerBasedRlEnv,
    *,
    env_id: int = 0,
  ) -> tuple[tuple[float, float], ...]:
    if not 0 <= env_id < len(self._raw_points_by_env):
      return ()
    env_origin = env.scene.env_origins[env_id].detach().cpu().numpy()
    return tuple(
      (float(point[0] - env_origin[0]), float(point[2] - env_origin[2]))
      for point in self._raw_points_by_env[env_id]
    )

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    self._detector.reset(env_ids)
    if env_ids is None:
      for points in self._raw_points_by_env:
        points.clear()
      for points in self._display_points_by_env:
        points.clear()
      self._last_new_counts = [0 for _ in self._raw_points_by_env]
      return
    for env_id in env_ids.detach().cpu().tolist():
      env_index = int(env_id)
      self._raw_points_by_env[env_index].clear()
      self._display_points_by_env[env_index].clear()
      self._last_new_counts[env_index] = 0

  def debug_vis(self, visualizer) -> None:
    for env_idx in visualizer.get_env_indices(len(self._display_points_by_env)):
      for position in self._display_points_by_env[int(env_idx)]:
        visualizer.add_sphere(
          center=position,
          radius=self._radius,
          color=(1.0, 0.0, 0.0, 1.0),
          label="semantic_toe_riser_contact",
        )


def _mode_label(mode_code: float | None) -> str:
  if mode_code is None or not np.isfinite(mode_code):
    return "UNKNOWN"
  if mode_code < 0.5:
    return "NORMAL"
  if mode_code < 1.5:
    return "WRITE"
  return "MEMORY"


def _trend_label(
  trend_probs: tuple[float | None, float | None, float | None],
  trend_value: float | None,
) -> str:
  finite_probs = [value for value in trend_probs if value is not None]
  if len(finite_probs) == 3:
    labels = ("BACKOFF", "HOLD", "FORWARD")
    return labels[int(np.argmax(np.asarray(finite_probs, dtype=np.float32)))]
  if trend_value is None or not np.isfinite(trend_value):
    return "UNKNOWN"
  if trend_value < 0.34:
    return "BACKOFF"
  if trend_value > 0.66:
    return "FORWARD"
  return "HOLD"


def _tensor_scalar(
  tensors: dict[str, torch.Tensor],
  key: str,
  *,
  env_id: int = 0,
  component: int = 0,
) -> float | None:
  value = tensors.get(key)
  if value is None:
    return None
  if value.ndim == 1:
    if value.shape[0] <= env_id:
      return None
    item = value[env_id]
  else:
    flat = value.reshape(value.shape[0], -1)
    if flat.shape[0] <= env_id or flat.shape[1] <= component:
      return None
    item = flat[env_id, component]
  result = float(item.detach().cpu().item())
  return result if np.isfinite(result) else None


def _tensor_triple(
  tensors: dict[str, torch.Tensor],
  key: str,
  *,
  env_id: int = 0,
) -> tuple[float | None, float | None, float | None]:
  value = tensors.get(key)
  if value is None:
    return (None, None, None)
  flat = value.reshape(value.shape[0], -1)
  if flat.shape[0] <= env_id or flat.shape[1] < 3:
    return (None, None, None)
  row = flat[env_id, :3].detach().cpu().numpy()
  return tuple(float(item) if np.isfinite(item) else None for item in row)  # type: ignore[return-value]


def _semantic_channel(
  diagnostics: dict[str, torch.Tensor],
  channel: int,
  *,
  env_id: int = 0,
) -> float | None:
  semantic = diagnostics.get("semantic")
  if semantic is None:
    return None
  flat = semantic.reshape(semantic.shape[0], -1)
  if flat.shape[0] <= env_id or flat.shape[1] <= channel:
    return None
  result = float(flat[env_id, channel].detach().cpu().item())
  return result if np.isfinite(result) else None


def _safe_stride_interval(
  diagnostics: dict[str, torch.Tensor],
  *,
  env_id: int = 0,
) -> tuple[float | None, float | None]:
  interval = diagnostics.get("safe_stride_interval")
  if interval is None:
    return (None, None)
  flat = interval.reshape(interval.shape[0], -1)
  if flat.shape[0] <= env_id or flat.shape[1] < 2:
    return (None, None)
  lower = float(flat[env_id, 0].detach().cpu().item())
  upper = float(flat[env_id, 1].detach().cpu().item())
  lower = lower if np.isfinite(lower) else None
  upper = upper if np.isfinite(upper) else None
  return lower, upper


def _select_terrain(
  terrains: tuple[EvalTerrainSpec, ...],
  cfg: SlowLatentSemanticVideoConfig,
) -> EvalTerrainSpec:
  if cfg.terrain_name is not None:
    for terrain in terrains:
      if terrain.name == cfg.terrain_name:
        return terrain
    raise ValueError(f"No terrain named {cfg.terrain_name!r}.")

  if cfg.terrain_label is not None:
    for terrain in terrains:
      if terrain.label == cfg.terrain_label:
        return terrain
    raise ValueError(f"No terrain with label {cfg.terrain_label!r}.")

  for terrain in terrains:
    if terrain.kind == "upstairs":
      return terrain
  return terrains[0]


def _resolve_video_paths(
  *,
  cfg: SlowLatentSemanticVideoConfig,
  task_id: str,
  agent_cfg: Any,
  checkpoint_path: Path,
  terrain: EvalTerrainSpec,
) -> tuple[Path, Path, Path]:
  if cfg.output_file is not None:
    video_path = Path(cfg.output_file)
    output_dir = video_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
  else:
    if cfg.output_dir is not None:
      output_dir = Path(cfg.output_dir)
      output_dir.mkdir(parents=True, exist_ok=True)
    else:
      output_dir = make_timestamped_policy_output_dir(
        output_root=cfg.output_root,
        task_id=task_id,
        agent_cfg=agent_cfg,
        checkpoint_path=checkpoint_path,
      )
    video_path = output_dir / f"semantic_rollout_{terrain.label}.mp4"

  trace_path = (
    Path(cfg.trace_file)
    if cfg.trace_file is not None
    else output_dir / f"semantic_rollout_{terrain.label}.csv"
  )
  summary_path = output_dir / f"semantic_rollout_{terrain.label}.json"
  return video_path, trace_path, summary_path


def _configure_video_camera(
  env_cfg,
  cfg: SlowLatentSemanticVideoConfig,
) -> None:
  scene_width = max(160, int(cfg.video_width) - max(0, int(cfg.hud_width)))
  scene_height = max(160, int(cfg.video_height) - max(0, int(cfg.side_panel_height)))
  env_cfg.viewer.width = scene_width
  env_cfg.viewer.height = scene_height
  env_cfg.viewer.origin_type = env_cfg.viewer.OriginType.ASSET_BODY
  env_cfg.viewer.entity_name = "robot"
  env_cfg.viewer.body_name = cfg.camera_body_name
  env_cfg.viewer.distance = cfg.camera_distance
  env_cfg.viewer.elevation = cfg.camera_elevation
  env_cfg.viewer.azimuth = cfg.camera_azimuth
  env_cfg.viewer.fovy = cfg.camera_fovy
  env_cfg.viewer.max_extra_envs = 0
  env_cfg.viewer.show_depth_camera_visualizers = False


def _make_toe_riser_contact_marker_overlay(
  env: ManagerBasedRlEnv,
  cfg: SlowLatentSemanticVideoConfig,
) -> ToeRiserContactMarkerOverlay | None:
  if not cfg.show_toe_riser_contact_markers:
    return None
  params = StairMetricParams(
    contact_force_threshold=cfg.toe_riser_contact_force_threshold,
    contact_cooldown_s=cfg.toe_riser_contact_cooldown_s,
    ignore_passed_support_riser_toe_contacts=(
      cfg.ignore_passed_support_riser_toe_contacts
    ),
  )
  detector = StairEventDetector(env, params)
  overlay = ToeRiserContactMarkerOverlay(
    env,
    detector,
    radius=cfg.toe_riser_contact_marker_radius,
    display_offset_world=(
      cfg.toe_riser_contact_marker_x_offset_m,
      cfg.toe_riser_contact_marker_y_offset_m,
      cfg.toe_riser_contact_marker_z_offset_m,
    ),
    max_points_per_env=cfg.toe_riser_contact_marker_max_points,
  )
  env.manager_visualizers["semantic_toe_riser_contact_markers"] = overlay
  return overlay


def _ground_contact_values(
  env: ManagerBasedRlEnv,
  *,
  foot_count: int,
  env_id: int = 0,
) -> list[bool] | None:
  for name in ("feet_ground_contact", "foot_contact", "ground_contact"):
    try:
      sensor = env.scene[name]
    except (KeyError, AttributeError):
      continue
    contact_time = getattr(getattr(sensor, "data", None), "current_contact_time", None)
    if not torch.is_tensor(contact_time):
      continue
    values = contact_time
    if values.ndim < 2 or values.shape[0] <= env_id:
      continue
    row = values[env_id]
    if row.numel() != foot_count:
      if row.numel() % foot_count != 0:
        continue
      row = row.reshape(foot_count, -1).amax(dim=-1)
    return [bool(item > 0.0) for item in row[:foot_count].detach().cpu().tolist()]
  return None


def _foot_snapshots(
  env: ManagerBasedRlEnv,
  *,
  env_id: int = 0,
) -> tuple[FootSnapshot | None, FootSnapshot | None]:
  robot = env.scene["robot"]
  try:
    site_ids, _site_names = robot.find_sites(
      ("left_foot", "right_foot"), preserve_order=True
    )
  except ValueError:
    return (None, None)
  if len(site_ids) < 2:
    return (None, None)

  env_origin = env.scene.env_origins[env_id].detach().cpu().numpy()
  pos = robot.data.site_pos_w[env_id, site_ids[:2]].detach().cpu().numpy()
  rel = pos - env_origin[None, :]
  contacts = _ground_contact_values(env, foot_count=2, env_id=env_id)
  if contacts is None:
    min_z = float(np.min(rel[:, 2]))
    swing = [bool(z > min_z + 0.035) for z in rel[:, 2]]
  else:
    swing = [not value for value in contacts]

  return (
    FootSnapshot(
      name="left",
      x_m=float(rel[0, 0]),
      z_m=float(rel[0, 2]),
      contact=None if contacts is None else contacts[0],
      swing=swing[0],
    ),
    FootSnapshot(
      name="right",
      x_m=float(rel[1, 0]),
      z_m=float(rel[1, 2]),
      contact=None if contacts is None else contacts[1],
      swing=swing[1],
    ),
  )


def _semantic_snapshot(
  env: ManagerBasedRlEnv,
  diagnostics: dict[str, torch.Tensor],
  *,
  terrain: EvalTerrainSpec,
  step: int,
  env_id: int = 0,
  toe_riser_contacts_this_step: int = 0,
  toe_riser_contact_markers_total: int = 0,
  toe_riser_contact_xz_m: tuple[tuple[float, float], ...] = (),
) -> SemanticSnapshot:
  robot = env.scene["robot"]
  env_origin = env.scene.env_origins[env_id]
  root_pos = robot.data.root_link_pos_w[env_id]
  root_x = float((root_pos[0] - env_origin[0]).detach().cpu().item())
  root_z = float((root_pos[2] - env_origin[2]).detach().cpu().item())
  mode_code = _tensor_scalar(diagnostics, "gate_mode", env_id=env_id)
  trend_probs = _tensor_triple(diagnostics, "safe_stride_trend_prob", env_id=env_id)
  trend_value = _tensor_scalar(diagnostics, "safe_stride_trend", env_id=env_id)
  left, right = _foot_snapshots(env, env_id=env_id)
  stride_lower, stride_upper = _safe_stride_interval(diagnostics, env_id=env_id)

  return SemanticSnapshot(
    step=step,
    time_s=float(step * env.step_dt),
    root_x_m=root_x,
    root_z_m=root_z,
    mode=_mode_label(mode_code),
    mode_code=mode_code,
    event_prob=_tensor_scalar(diagnostics, "event_prob", env_id=env_id),
    stair_prob=_tensor_scalar(diagnostics, "stair_prob", env_id=env_id),
    write_progress=_semantic_channel(diagnostics, 5, env_id=env_id),
    memory_age=_semantic_channel(diagnostics, 6, env_id=env_id),
    release_progress=_semantic_channel(diagnostics, 7, env_id=env_id),
    pred_riser_height_m=_tensor_scalar(
      diagnostics, "stair_shape", env_id=env_id, component=1
    ),
    true_riser_height_m=terrain.height_m,
    safe_stride_m=_tensor_scalar(diagnostics, "safe_stride", env_id=env_id),
    stride_lower_m=stride_lower,
    stride_upper_m=stride_upper,
    stride_trend=trend_value,
    trend_label=_trend_label(trend_probs, trend_value),
    trend_probs=trend_probs,
    confidence=_tensor_scalar(diagnostics, "safe_stride_confidence", env_id=env_id),
    alpha_state=_tensor_scalar(diagnostics, "alpha_state", env_id=env_id),
    alpha_shape=_tensor_scalar(diagnostics, "alpha_shape", env_id=env_id),
    foot_left=left,
    foot_right=right,
    toe_riser_contacts_this_step=int(toe_riser_contacts_this_step),
    toe_riser_contact_markers_total=int(toe_riser_contact_markers_total),
    toe_riser_contact_xz_m=toe_riser_contact_xz_m,
  )


def _font(size: int, *, bold: bool = False) -> ImageFontLike:
  candidates = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
  )
  paths = candidates if bold else tuple(reversed(candidates))
  for path in paths:
    try:
      return ImageFont.truetype(path, size=size)
    except OSError:
      continue
  return ImageFont.load_default()


def _format_value(value: float | None, suffix: str = "", digits: int = 2) -> str:
  if value is None or not np.isfinite(value):
    return "--"
  return f"{value:.{digits}f}{suffix}"


def _draw_bar(
  draw: ImageDraw.ImageDraw,
  *,
  xy: tuple[int, int],
  size: tuple[int, int],
  value: float | None,
  fill: tuple[int, int, int],
  label: str,
  font: ImageFontLike,
) -> None:
  x, y = xy
  width, height = size
  draw.rectangle([x, y, x + width, y + height], fill=(235, 236, 238))
  if value is not None and np.isfinite(value):
    frac = max(0.0, min(1.0, float(value)))
    draw.rectangle([x, y, x + int(width * frac), y + height], fill=fill)
  draw.rectangle([x, y, x + width, y + height], outline=(160, 164, 169))
  text = f"{label}: {_format_value(value, digits=2)}"
  draw.text((x, y - 18), text, fill=(38, 42, 47), font=font)


def _draw_header(
  draw: ImageDraw.ImageDraw,
  snapshot: SemanticSnapshot,
  *,
  x: int,
  y: int,
  width: int,
) -> int:
  title_font = _font(22, bold=True)
  body_font = _font(14)
  mode_color = MODE_COLORS.get(snapshot.mode, (125, 125, 125))
  draw.rectangle([x, y, x + width, y + 52], fill=mode_color)
  draw.text((x + 14, y + 10), f"MODE: {snapshot.mode}", fill="white", font=title_font)
  y += 64
  draw.text(
    (x + 14, y),
    f"t={snapshot.time_s:5.2f}s  step={snapshot.step}",
    fill=(38, 42, 47),
    font=body_font,
  )
  return y + 26


def _draw_foot_cues(
  draw: ImageDraw.ImageDraw,
  snapshot: SemanticSnapshot,
  *,
  x: int,
  y: int,
  width: int,
) -> int:
  title_font = _font(18, bold=True)
  body_font = _font(14)
  trend_color = TREND_COLORS.get(snapshot.trend_label, TREND_COLORS["UNKNOWN"])
  draw.text((x, y), "Stride cue", fill=(24, 28, 32), font=title_font)
  y += 28
  draw.rectangle([x, y, x + width, y + 40], fill=trend_color)
  draw.text(
    (x + 12, y + 10),
    snapshot.trend_label,
    fill="white",
    font=_font(17, bold=True),
  )
  y += 54
  labels = ("backoff", "hold", "forward")
  for idx, (label, prob) in enumerate(zip(labels, snapshot.trend_probs, strict=True)):
    _draw_bar(
      draw,
      xy=(x, y + idx * 28 + 16),
      size=(width, 12),
      value=prob,
      fill=TREND_COLORS[("BACKOFF", "HOLD", "FORWARD")[idx]],
      label=label,
      font=body_font,
    )
  y += 104
  for foot in (snapshot.foot_left, snapshot.foot_right):
    if foot is None:
      continue
    state = "SWING" if foot.swing else "STANCE"
    color = (38, 126, 196) if foot.swing else (80, 96, 108)
    contact = (
      "contact unknown" if foot.contact is None else f"contact={int(foot.contact)}"
    )
    draw.text(
      (x, y),
      f"{foot.name.upper():5s} {state:6s}  x={foot.x_m:.2f} z={foot.z_m:.2f}  {contact}",
      fill=color,
      font=body_font,
    )
    y += 22
  return y + 8


def _draw_hud(
  draw: ImageDraw.ImageDraw,
  snapshot: SemanticSnapshot,
  *,
  x: int,
  y: int,
  width: int,
  height: int,
) -> None:
  draw.rectangle([x, y, x + width, y + height], fill=(249, 250, 248))
  y = _draw_header(draw, snapshot, x=x, y=y + 14, width=width)
  body_font = _font(14)
  _draw_bar(
    draw,
    xy=(x + 14, y + 22),
    size=(width - 28, 12),
    value=snapshot.event_prob,
    fill=(207, 91, 69),
    label="event prob",
    font=body_font,
  )
  y += 44
  _draw_bar(
    draw,
    xy=(x + 14, y + 22),
    size=(width - 28, 12),
    value=snapshot.stair_prob,
    fill=(44, 135, 102),
    label="stair prob",
    font=body_font,
  )
  y += 46
  for label, value, color in (
    ("write", snapshot.write_progress, MODE_COLORS["WRITE"]),
    ("memory age", snapshot.memory_age, MODE_COLORS["MEMORY"]),
    ("release", snapshot.release_progress, (156, 97, 179)),
  ):
    _draw_bar(
      draw,
      xy=(x + 14, y + 20),
      size=(width - 28, 10),
      value=value,
      fill=color,
      label=label,
      font=body_font,
    )
    y += 36

  y += 4
  draw.text((x + 14, y), "Geometry head", fill=(24, 28, 32), font=_font(18, bold=True))
  y += 28
  rows = [
    (
      "riser pred / true",
      f"{_format_value(snapshot.pred_riser_height_m, ' m', 3)} / "
      f"{_format_value(snapshot.true_riser_height_m, ' m', 3)}",
    ),
    ("safe stride", _format_value(snapshot.safe_stride_m, " m", 3)),
    (
      "safe interval",
      f"{_format_value(snapshot.stride_lower_m, ' m', 3)} - "
      f"{_format_value(snapshot.stride_upper_m, ' m', 3)}",
    ),
    ("confidence", _format_value(snapshot.confidence, digits=2)),
    (
      "toe-riser hits",
      f"{snapshot.toe_riser_contacts_this_step} / "
      f"{snapshot.toe_riser_contact_markers_total}",
    ),
    (
      "alpha state/shape",
      f"{_format_value(snapshot.alpha_state, digits=2)} / "
      f"{_format_value(snapshot.alpha_shape, digits=2)}",
    ),
  ]
  for label, value in rows:
    draw.text((x + 14, y), f"{label}: {value}", fill=(42, 47, 54), font=body_font)
    y += 22

  _draw_foot_cues(draw, snapshot, x=x + 14, y=y + 8, width=width - 28)


def _stair_profile_points(terrain: EvalTerrainSpec) -> list[tuple[float, float]]:
  if (
    terrain.layout != "long_runway"
    or terrain.kind != "upstairs"
    or terrain.height_m is None
  ):
    return []
  origin_x = 0.5 * terrain.runway_start_platform_length
  start_x = terrain.runway_start_platform_length - origin_x
  points: list[tuple[float, float]] = [(-0.2, 0.0), (start_x, 0.0)]
  for step in range(terrain.runway_num_steps):
    x0 = start_x + step * terrain.step_width
    x1 = x0 + terrain.step_width
    z0 = step * terrain.height_m
    z1 = (step + 1) * terrain.height_m
    points.append((x0, z1))
    points.append((x1, z1))
    if step + 1 < terrain.runway_num_steps:
      points.append((x1, z0 + terrain.height_m))
  points.append((start_x + terrain.runway_num_steps * terrain.step_width + 0.6, z1))
  return points


def _long_runway_stair_start_x_m(terrain: EvalTerrainSpec) -> float | None:
  if terrain.layout != "long_runway" or terrain.kind != "upstairs":
    return None
  return 0.5 * terrain.runway_start_platform_length


def _stair_floor_height_at_x(terrain: EvalTerrainSpec, x_m: float) -> float:
  if (
    terrain.layout != "long_runway"
    or terrain.kind != "upstairs"
    or terrain.height_m is None
  ):
    return 0.0
  start_x = _long_runway_stair_start_x_m(terrain)
  assert start_x is not None
  if x_m < start_x:
    return 0.0
  layer = int(np.floor((x_m - start_x) / max(terrain.step_width, 1.0e-6))) + 1
  layer = max(0, min(layer, terrain.runway_num_steps))
  return float(layer) * terrain.height_m


def _draw_stair_silhouette(
  draw: ImageDraw.ImageDraw,
  terrain: EvalTerrainSpec,
  *,
  project,
  x_min: float,
  x_max: float,
  z_min: float,
) -> None:
  if (
    terrain.layout != "long_runway"
    or terrain.kind != "upstairs"
    or terrain.height_m is None
  ):
    profile = _stair_profile_points(terrain)
    if len(profile) >= 2:
      draw.line([project(point) for point in profile], fill=(92, 98, 103), width=2)
    else:
      draw.line(
        [project((x_min, 0.0)), project((x_max, 0.0))], fill=(92, 98, 103), width=2
      )
    return

  start_x = _long_runway_stair_start_x_m(terrain)
  assert start_x is not None
  h = float(terrain.height_m)
  d = max(float(terrain.step_width), 1.0e-6)
  end_x = start_x + terrain.runway_num_steps * d
  fill = (184, 222, 244)
  top = (55, 107, 153)
  face = (52, 91, 139)

  segments: list[tuple[float, float, float]] = [(x_min, min(start_x, x_max), 0.0)]
  for step in range(terrain.runway_num_steps):
    x0 = start_x + step * d
    x1 = x0 + d
    z = (step + 1) * h
    segments.append((x0, x1, z))
  segments.append((end_x, x_max, terrain.runway_num_steps * h))

  for seg_x0, seg_x1, z in segments:
    clipped_x0 = max(x_min, seg_x0)
    clipped_x1 = min(x_max, seg_x1)
    if clipped_x1 <= clipped_x0:
      continue
    visible_z = max(z, z_min)
    px0, py_top = project((clipped_x0, visible_z))
    px1, py_bottom = project((clipped_x1, z_min))
    draw.rectangle(
      [min(px0, px1), min(py_top, py_bottom), max(px0, px1), max(py_top, py_bottom)],
      fill=fill,
    )
    if z >= z_min:
      draw.line([project((clipped_x0, z)), project((clipped_x1, z))], fill=top, width=2)

  for layer in range(terrain.runway_num_steps):
    riser_x = start_x + layer * d
    if x_min <= riser_x <= x_max:
      lower_z = max(layer * h, z_min)
      upper_z = max((layer + 1) * h, z_min)
      if upper_z <= lower_z:
        continue
      draw.line(
        [
          project((riser_x, lower_z)),
          project((riser_x, upper_z)),
        ],
        fill=face,
        width=2,
      )


def _draw_profile_foot(
  draw: ImageDraw.ImageDraw,
  foot: FootSnapshot,
  *,
  project,
  foot_length_m: float,
  foot_height_m: float,
  fill: tuple[int, int, int],
) -> None:
  half_length = 0.5 * max(0.01, float(foot_length_m))
  height = max(0.01, float(foot_height_m))
  heel_x = foot.x_m - half_length
  toe_x = foot.x_m + half_length
  sole_z = foot.z_m
  top_z = foot.z_m + height
  x0, y0 = project((heel_x, top_z))
  x1, y1 = project((toe_x, sole_z))
  outline = (24, 28, 32) if foot.contact else (255, 255, 255)
  draw.rectangle([x0, y0, x1, y1], fill=fill, outline=outline, width=2)
  toe_px, toe_py = project((toe_x, sole_z + 0.5 * height))
  draw.line([toe_px, toe_py - 5, toe_px, toe_py + 5], fill=outline, width=2)


def _draw_stair_shape_inset(
  canvas: Image.Image,
  terrain: EvalTerrainSpec,
  *,
  x: int,
  y: int,
  width: int,
  height: int,
) -> None:
  """Draw a fixed-size stair-ascent icon without scaling with rollout length."""
  if terrain.kind != "upstairs" or terrain.height_m is None:
    return
  inset = Image.new("RGBA", (width, height), (0, 0, 0, 0))
  draw = ImageDraw.Draw(inset)
  pad = 12
  steps = 5
  tread = max(float(terrain.step_width), 1.0e-6)
  riser = max(float(terrain.height_m), 1.0e-6)
  step_w = max(16, int((width - 2 * pad) / (steps + 0.5)))
  step_h = int(np.clip(step_w * riser / tread, 8, 18))
  origin_x = pad
  base_y = height - pad
  fill = (108, 193, 232, 192)
  edge = (33, 78, 126, 230)

  for idx in range(steps):
    x0 = origin_x + idx * step_w
    x1 = origin_x + (idx + 1) * step_w
    y0 = base_y - (idx + 1) * step_h
    draw.rectangle([x0, y0, x1, base_y], fill=fill)
    draw.line([x0, y0, x1, y0], fill=edge, width=2)
    draw.line([x0, y0, x0, y0 + step_h], fill=edge, width=2)
  last_x = origin_x + steps * step_w
  draw.line(
    [last_x, base_y - steps * step_h, last_x, base_y],
    fill=edge,
    width=2,
  )
  canvas.alpha_composite(inset, (x, y))


def _finite_or_zero(value: float | None) -> float:
  if value is None or not np.isfinite(value):
    return 0.0
  return float(value)


def _history_distance_axis(history: list[SemanticSnapshot]) -> list[float]:
  if not history:
    return []
  distances = [0.0]
  for prev, current in zip(history[:-1], history[1:], strict=True):
    step = abs(float(current.root_x_m) - float(prev.root_x_m))
    distances.append(distances[-1] + step)
  return distances


def _draw_signal_line(
  draw: ImageDraw.ImageDraw,
  *,
  xs: list[float],
  history: list[SemanticSnapshot],
  project,
  value_fn,
  color: tuple[int, int, int],
  width: int = 2,
) -> None:
  points = [
    project(x_value, max(0.0, min(1.0, float(value_fn(snapshot)))))
    for x_value, snapshot in zip(xs, history, strict=True)
  ]
  if len(points) >= 2:
    draw.line(points, fill=color, width=width)
  elif points:
    x0, y0 = points[0]
    draw.ellipse([x0 - 2, y0 - 2, x0 + 2, y0 + 2], fill=color)


def _draw_state_strip(
  draw: ImageDraw.ImageDraw,
  *,
  xs: list[float],
  history: list[SemanticSnapshot],
  project_x,
  y0: int,
  height: int,
  color_fn,
) -> None:
  if not history:
    return
  for idx, snapshot in enumerate(history):
    left = project_x(xs[idx])
    if idx + 1 < len(history):
      right = project_x(xs[idx + 1]) + 1
    else:
      right = left + 2
    draw.rectangle(
      [left, y0, max(right, left + 1), y0 + height], fill=color_fn(snapshot)
    )


def _termination_reason_names(env: ManagerBasedRlEnv, *, env_id: int = 0) -> list[str]:
  reasons: list[str] = []
  manager = env.termination_manager
  for name in manager.active_terms:
    value = manager.get_term(name)
    if isinstance(value, torch.Tensor):
      flat = value.detach().reshape(-1)
      if flat.numel() == 0:
        continue
      idx = min(max(0, int(env_id)), flat.numel() - 1)
      active = bool(flat[idx].item())
    else:
      active = bool(value)
    if active:
      reasons.append(str(name))
  return reasons


def _draw_signal_panel(
  draw: ImageDraw.ImageDraw,
  history: list[SemanticSnapshot],
  terrain: EvalTerrainSpec,
  cfg: SlowLatentSemanticVideoConfig,
  *,
  x: int,
  y: int,
  width: int,
  height: int,
) -> None:
  draw.rectangle([x, y, x + width, y + height], fill=(247, 248, 246))
  title_font = _font(14, bold=True)
  small_font = _font(11)
  axis_font = _font(10)
  draw.text(
    (x + 12, y + 8),
    "state / phase / update vs distance",
    fill=(28, 31, 35),
    font=title_font,
  )

  if not history:
    return

  xs = _history_distance_axis(history)
  completion_x = terrain.stair_completion_root_x_m()
  expected_post_m = abs(float(cfg.command_vx)) * max(
    0.0, float(cfg.post_stair_completion_time_s)
  )
  expected_distance = max(float(completion_x or 0.0) + expected_post_m + 0.35, 1.0)
  min_window_m = max(1.0, 0.35 * expected_distance)
  x_max = max(min_window_m, min(expected_distance, xs[-1] + 0.35), xs[-1] + 0.05)
  plot_x0 = x + 46
  plot_y0 = y + 42
  plot_w = width - 70
  plot_h = height - 68
  strip_h = 10
  phase_y = plot_y0 + strip_h + 3
  graph_y0 = phase_y + strip_h + 12
  graph_h = max(48, plot_y0 + plot_h - graph_y0)

  def project_x(distance_m: float) -> int:
    return plot_x0 + int(max(0.0, min(1.0, distance_m / x_max)) * plot_w)

  def project(distance_m: float, value: float) -> tuple[int, int]:
    px = project_x(distance_m)
    py = graph_y0 + graph_h - int(max(0.0, min(1.0, value)) * graph_h)
    return px, py

  walked_text = f"walked {xs[-1]:.2f} m"
  try:
    walked_w = int(draw.textlength(walked_text, font=axis_font))
  except AttributeError:
    walked_w = 82
  draw.text(
    (x + width - 12 - walked_w, y + 12),
    walked_text,
    fill=(72, 75, 80),
    font=axis_font,
  )

  draw.rectangle(
    [plot_x0, graph_y0, plot_x0 + plot_w, graph_y0 + graph_h],
    fill=(255, 255, 255),
    outline=(185, 188, 190),
  )
  for frac in (0.25, 0.50, 0.75):
    yy = graph_y0 + graph_h - int(frac * graph_h)
    draw.line([plot_x0, yy, plot_x0 + plot_w, yy], fill=(224, 226, 228), width=1)

  _draw_state_strip(
    draw,
    xs=xs,
    history=history,
    project_x=project_x,
    y0=plot_y0,
    height=strip_h,
    color_fn=lambda snap: MODE_COLORS.get(snap.mode, MODE_COLORS["NORMAL"]),
  )
  _draw_state_strip(
    draw,
    xs=xs,
    history=history,
    project_x=project_x,
    y0=phase_y,
    height=strip_h,
    color_fn=lambda snap: TREND_COLORS.get(snap.trend_label, TREND_COLORS["UNKNOWN"]),
  )
  draw.text((x + 12, plot_y0 - 2), "gate", fill=(72, 75, 80), font=axis_font)
  draw.text((x + 12, phase_y - 2), "cue", fill=(72, 75, 80), font=axis_font)

  _draw_signal_line(
    draw,
    xs=xs,
    history=history,
    project=project,
    value_fn=lambda snap: _finite_or_zero(snap.event_prob),
    color=(217, 105, 39),
    width=2,
  )
  _draw_signal_line(
    draw,
    xs=xs,
    history=history,
    project=project,
    value_fn=lambda snap: _finite_or_zero(snap.stair_prob),
    color=(30, 121, 198),
    width=2,
  )
  _draw_signal_line(
    draw,
    xs=xs,
    history=history,
    project=project,
    value_fn=lambda snap: max(
      _finite_or_zero(snap.alpha_state),
      _finite_or_zero(snap.alpha_shape),
    ),
    color=(32, 150, 99),
    width=2,
  )
  for snap_x, snap in zip(xs, history, strict=True):
    if snap.toe_riser_contacts_this_step <= 0:
      continue
    px = project_x(snap_x)
    draw.line([px, graph_y0, px, graph_y0 + graph_h], fill=(221, 39, 39), width=2)

  current_x = project_x(xs[-1])
  draw.line(
    [current_x, plot_y0 - 2, current_x, graph_y0 + graph_h + 3], fill=0, width=2
  )

  legend = [
    ("event", (217, 105, 39)),
    ("stair", (30, 121, 198)),
    ("update", (32, 150, 99)),
    ("toe hit", (221, 39, 39)),
  ]
  legend_x = plot_x0 + 8
  legend_y = y + height - 22
  legend_step = max(72, min(96, (plot_w - 20) // max(len(legend), 1)))
  for label, color in legend:
    draw.line(
      [legend_x, legend_y + 6, legend_x + 18, legend_y + 6], fill=color, width=3
    )
    draw.text((legend_x + 22, legend_y), label, fill=(55, 59, 65), font=small_font)
    legend_x += legend_step


def _draw_side_panel(
  draw: ImageDraw.ImageDraw,
  history: list[SemanticSnapshot],
  terrain: EvalTerrainSpec,
  cfg: SlowLatentSemanticVideoConfig,
  *,
  x: int,
  y: int,
  width: int,
  height: int,
) -> None:
  _draw_signal_panel(
    draw,
    history,
    terrain,
    cfg,
    x=x,
    y=y,
    width=width,
    height=height,
  )


def compose_video_frame(
  rgb_frame: np.ndarray,
  snapshot: SemanticSnapshot,
  history: list[SemanticSnapshot],
  terrain: EvalTerrainSpec,
  cfg: SlowLatentSemanticVideoConfig,
) -> np.ndarray:
  """Compose one rendered frame with the semantic HUD and side profile."""
  video_width = int(cfg.video_width)
  video_height = int(cfg.video_height)
  hud_width = max(0, min(int(cfg.hud_width), video_width - 160))
  side_height = max(0, min(int(cfg.side_panel_height), video_height - 160))
  scene_width = video_width - hud_width
  scene_height = video_height - side_height

  frame = np.asarray(rgb_frame)
  if frame.ndim == 4:
    frame = frame[0]
  if frame.dtype != np.uint8:
    frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
  image = Image.fromarray(frame).convert("RGB")
  image = image.resize((scene_width, scene_height), Image.Resampling.BILINEAR)

  canvas = Image.new("RGBA", (video_width, video_height), (246, 246, 243, 255))
  canvas.paste(image, (0, 0))
  if scene_width >= 220 and scene_height >= 140:
    _draw_stair_shape_inset(
      canvas,
      terrain,
      x=14,
      y=14,
      width=170,
      height=92,
    )
  draw = ImageDraw.Draw(canvas)
  if hud_width > 0:
    _draw_hud(draw, snapshot, x=scene_width, y=0, width=hud_width, height=video_height)
  if side_height > 0:
    _draw_side_panel(
      draw,
      history,
      terrain,
      cfg,
      x=0,
      y=scene_height,
      width=scene_width,
      height=side_height,
    )
  return np.asarray(canvas.convert("RGB"), dtype=np.uint8)


def _snapshot_row(snapshot: SemanticSnapshot) -> dict[str, float | int | str | None]:
  left = snapshot.foot_left
  right = snapshot.foot_right
  return {
    "step": snapshot.step,
    "time_s": snapshot.time_s,
    "root_x_m": snapshot.root_x_m,
    "root_z_m": snapshot.root_z_m,
    "mode": snapshot.mode,
    "mode_code": snapshot.mode_code,
    "event_prob": snapshot.event_prob,
    "stair_prob": snapshot.stair_prob,
    "write_progress": snapshot.write_progress,
    "memory_age": snapshot.memory_age,
    "release_progress": snapshot.release_progress,
    "pred_riser_height_m": snapshot.pred_riser_height_m,
    "true_riser_height_m": snapshot.true_riser_height_m,
    "safe_stride_m": snapshot.safe_stride_m,
    "stride_lower_m": snapshot.stride_lower_m,
    "stride_upper_m": snapshot.stride_upper_m,
    "stride_trend": snapshot.stride_trend,
    "trend_label": snapshot.trend_label,
    "trend_prob_backoff": snapshot.trend_probs[0],
    "trend_prob_hold": snapshot.trend_probs[1],
    "trend_prob_forward": snapshot.trend_probs[2],
    "confidence": snapshot.confidence,
    "alpha_state": snapshot.alpha_state,
    "alpha_shape": snapshot.alpha_shape,
    "toe_riser_contacts_this_step": snapshot.toe_riser_contacts_this_step,
    "toe_riser_contact_markers_total": snapshot.toe_riser_contact_markers_total,
    "toe_riser_contact_xz_m": json.dumps(snapshot.toe_riser_contact_xz_m),
    "left_foot_x_m": None if left is None else left.x_m,
    "left_foot_z_m": None if left is None else left.z_m,
    "left_foot_swing": None if left is None else int(bool(left.swing)),
    "right_foot_x_m": None if right is None else right.x_m,
    "right_foot_z_m": None if right is None else right.z_m,
    "right_foot_swing": None if right is None else int(bool(right.swing)),
  }


def _write_trace_csv(path: Path, snapshots: list[SemanticSnapshot]) -> None:
  rows = [_snapshot_row(snapshot) for snapshot in snapshots]
  if not rows:
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def run_record_semantic_video(
  task_id: str,
  cfg: SlowLatentSemanticVideoConfig,
) -> dict[str, Any]:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  terrains = get_terrain_set(cfg.terrain_set)
  terrain = _select_terrain(terrains, cfg)
  agent_cfg = load_rl_cfg(task_id)
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
  video_path, trace_path, summary_path = _resolve_video_paths(
    cfg=cfg,
    task_id=task_id,
    agent_cfg=agent_cfg,
    checkpoint_path=checkpoint_path,
    terrain=terrain,
  )

  env_cfg = load_env_cfg(task_id, play=False)
  apply_eval_overrides(
    env_cfg,
    terrain,
    num_envs=max(1, int(cfg.num_envs)),
    seed=cfg.seed,
    max_episode_length_s=cfg.max_episode_length_s,
    command=(cfg.command_vx, cfg.command_vy, cfg.command_wz),
    clean_observations=cfg.clean_observations,
    disable_observation_delay=cfg.disable_observation_delay,
    disable_actuator_delay=cfg.disable_actuator_delay,
    enable_riser_contact_sensor=cfg.show_toe_riser_contact_markers,
  )
  if float(cfg.post_stair_completion_time_s) > 0.0:
    env_cfg.terminations.pop("runway_target_reached", None)
    env_cfg.terminations.pop("runway_out_of_bounds", None)
  _configure_video_camera(env_cfg, cfg)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  toe_riser_markers = _make_toe_riser_contact_marker_overlay(env, cfg)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=get_clip_actions(agent_cfg))
  step_dt = float(wrapped.unwrapped.step_dt)
  completion_x = terrain.stair_completion_root_x_m()
  if completion_x is not None and cfg.stop_after_stair_completion:
    completion_x += float(cfg.stair_completion_margin_m)
  post_completion_steps = max(
    0,
    int(round(max(0.0, float(cfg.post_stair_completion_time_s)) / step_dt)),
  )
  completion_step: int | None = None
  done_reasons: list[str] = []

  frames: list[np.ndarray] = []
  snapshots: list[SemanticSnapshot] = []
  render_every = max(1, int(cfg.render_every))

  try:
    policy, _runner = load_inference_policy(
      env=wrapped,
      task_id=task_id,
      agent_cfg=agent_cfg,
      checkpoint_path=checkpoint_path,
      device=device,
    )
    obs = wrapped.get_observations()
    max_steps = int(round(cfg.max_episode_length_s / step_dt))
    print(f"[INFO] Recording {terrain.name} to {video_path}")
    for step in range(max_steps):
      with torch.no_grad():
        actions = policy(obs)
        state_tensors = _policy_state_tensors(policy, batch_size=env.num_envs)
        diagnostics = _policy_diagnostic_tensors(
          policy,
          obs,
          state_tensors,
          batch_size=env.num_envs,
        )

      toe_marker_new_count = 0
      toe_marker_total_count = 0
      toe_marker_xz: tuple[tuple[float, float], ...] = ()
      if toe_riser_markers is not None:
        toe_riser_markers.update(wrapped.unwrapped)
        toe_marker_new_count = toe_riser_markers.new_count(env_id=0)
        toe_marker_total_count = toe_riser_markers.total_count(env_id=0)
        toe_marker_xz = toe_riser_markers.xz_points(wrapped.unwrapped, env_id=0)

      snapshot = _semantic_snapshot(
        wrapped.unwrapped,
        diagnostics,
        terrain=terrain,
        step=step,
        env_id=0,
        toe_riser_contacts_this_step=toe_marker_new_count,
        toe_riser_contact_markers_total=toe_marker_total_count,
        toe_riser_contact_xz_m=toe_marker_xz,
      )
      snapshots.append(snapshot)
      if step == 0 or (
        cfg.progress_interval_steps > 0 and step % cfg.progress_interval_steps == 0
      ):
        print(
          "[INFO] "
          f"step={step}/{max_steps} "
          f"t={snapshot.time_s:.2f}s "
          f"x={snapshot.root_x_m:.2f} "
          f"mode={snapshot.mode} "
          f"stride={snapshot.trend_label} "
          f"toe_hits={snapshot.toe_riser_contact_markers_total} "
          f"frames={len(frames)}"
        )

      if step % render_every == 0:
        raw_frame = wrapped.unwrapped.render()
        if raw_frame is not None:
          frames.append(
            compose_video_frame(raw_frame, snapshot, snapshots, terrain, cfg)
          )

      if (
        completion_x is not None
        and cfg.stop_after_stair_completion
        and snapshot.root_x_m >= completion_x
      ):
        if completion_step is None:
          completion_step = step
          print(
            f"[INFO] Reached stair completion x={completion_x:.3f} m at step {step}; "
            f"recording {post_completion_steps} more steps on the top platform."
          )
        if step - completion_step >= post_completion_steps:
          print(f"[INFO] Finished top-platform tail at step {step}.")
          break

      step_result = wrapped.step(actions)
      reset_policy_state_from_step(policy, step_result)
      obs, _rewards, dones, _extras = step_result
      if bool(dones[0].detach().cpu().item()):
        done_reasons = _termination_reason_names(wrapped.unwrapped, env_id=0)
        reason_text = ", ".join(done_reasons) if done_reasons else "done"
        print(f"[INFO] Episode ended at step {step}: {reason_text}.")
        break

  finally:
    wrapped.close()

  if not frames:
    raise RuntimeError("No frames were rendered; video was not written.")

  fps = cfg.fps
  if fps is None:
    fps = 1.0 / (step_dt * render_every)
  video_path.parent.mkdir(parents=True, exist_ok=True)
  media.write_video(str(video_path), frames, fps=float(fps))
  _write_trace_csv(trace_path, snapshots)

  summary = {
    "video_file": str(video_path),
    "trace_file": str(trace_path),
    "terrain": asdict(terrain),
    "frames": len(frames),
    "steps": len(snapshots),
    "fps": float(fps),
    "completion_root_x_m": completion_x,
    "completion_step": completion_step,
    "post_stair_completion_time_s": float(cfg.post_stair_completion_time_s),
    "done_reasons": done_reasons,
    "toe_riser_contact_marker_event_source": (
      None if toe_riser_markers is None else toe_riser_markers.event_source
    ),
    "toe_riser_contact_marker_count": (
      0 if toe_riser_markers is None else toe_riser_markers.total_count(env_id=0)
    ),
    "toe_riser_contact_marker_display_offset_m": (
      cfg.toe_riser_contact_marker_x_offset_m,
      cfg.toe_riser_contact_marker_y_offset_m,
      cfg.toe_riser_contact_marker_z_offset_m,
    ),
    "last_snapshot": _snapshot_row(snapshots[-1]) if snapshots else None,
  }
  summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(f"[INFO] Wrote video to {video_path}")
  print(f"[INFO] Wrote trace CSV to {trace_path}")
  print(f"[INFO] Wrote summary to {summary_path}")
  return summary


def main() -> None:
  import mjlab.tasks  # noqa: F401

  velocity_tasks = [task for task in list_tasks() if "Velocity" in task]
  if not velocity_tasks:
    print("No velocity tasks found.")
    sys.exit(1)

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(velocity_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  cfg = tyro.cli(
    SlowLatentSemanticVideoConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_record_semantic_video(chosen_task, cfg)


if __name__ == "__main__":
  main()
