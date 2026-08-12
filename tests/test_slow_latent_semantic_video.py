from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from scripts.velocity_eval.eval_terrains import EvalTerrainSpec, get_terrain_set
from scripts.velocity_eval.record_slow_latent_semantic_video import (
  FootSnapshot,
  SemanticSnapshot,
  SlowLatentSemanticVideoConfig,
  ToeRiserContactMarkerOverlay,
  _mode_label,
  _select_terrain,
  _snapshot_row,
  _trend_label,
  compose_video_frame,
)


class _FakeDetector:
  event_source = "true_contact"

  def __init__(self) -> None:
    self.updates = 0
    self.resets = []

  def compute_events(self, env) -> dict:
    del env
    self.updates += 1
    return {}

  def get_last_true_contact_positions(self, kind: str):
    assert kind == "toe"
    if self.updates == 1:
      return torch.tensor([0]), torch.tensor([[1.0, 2.0, 3.0]])
    if self.updates == 2:
      return torch.tensor([0]), torch.tensor([[4.0, 5.0, 6.0]])
    return torch.empty(0, dtype=torch.long), torch.empty(0, 3)

  def reset(self, env_ids=None) -> None:
    self.resets.append(env_ids)


class _FakeVisualizer:
  def __init__(self) -> None:
    self.spheres = []

  def get_env_indices(self, num_envs: int):
    del num_envs
    return [0]

  def add_sphere(self, *, center, radius, color, label=None) -> None:
    self.spheres.append((np.asarray(center), radius, color, label))


def test_mode_label_maps_gate_codes() -> None:
  assert _mode_label(None) == "UNKNOWN"
  assert _mode_label(float("nan")) == "UNKNOWN"
  assert _mode_label(0.0) == "NORMAL"
  assert _mode_label(1.0) == "WRITE"
  assert _mode_label(2.0) == "MEMORY"


def test_trend_label_prefers_class_probs_then_scalar_fallback() -> None:
  assert _trend_label((0.9, 0.05, 0.05), None) == "BACKOFF"
  assert _trend_label((0.1, 0.8, 0.1), None) == "HOLD"
  assert _trend_label((0.1, 0.2, 0.7), None) == "FORWARD"
  assert _trend_label((None, None, None), 0.1) == "BACKOFF"
  assert _trend_label((None, None, None), 0.5) == "HOLD"
  assert _trend_label((None, None, None), 0.9) == "FORWARD"


def test_select_terrain_uses_long_stair_label() -> None:
  terrain = _select_terrain(
    get_terrain_set("long_stair_riser_grid_v1"),
    SlowLatentSemanticVideoConfig(terrain_label="h15"),
  )

  assert terrain.name == "long_upstairs_h15_d30"
  assert terrain.kind == "upstairs"
  assert terrain.layout == "long_runway"
  assert terrain.height_m == 0.15


def test_toe_riser_marker_overlay_persists_eval_filtered_contacts() -> None:
  env = cast(
    Any,
    SimpleNamespace(
      num_envs=1,
      scene=SimpleNamespace(env_origins=torch.tensor([[0.5, 0.0, 2.5]])),
    ),
  )
  detector = cast(Any, _FakeDetector())
  markers = ToeRiserContactMarkerOverlay(
    env,
    detector,
    radius=0.05,
    display_offset_world=(-0.1, 0.0, 0.2),
    max_points_per_env=2,
  )

  assert markers.update(env) == 1
  assert markers.new_count() == 1
  assert markers.total_count() == 1
  assert markers.update(env) == 1
  assert markers.total_count() == 2
  assert markers.update(env) == 0

  visualizer = _FakeVisualizer()
  markers.debug_vis(visualizer)

  centers = [sphere[0].tolist() for sphere in visualizer.spheres]
  np.testing.assert_allclose(centers, [[0.9, 2.0, 3.2], [3.9, 5.0, 6.2]])
  np.testing.assert_allclose(markers.xz_points(env), ((0.5, 0.5), (3.5, 3.5)))
  assert all(sphere[1] == 0.05 for sphere in visualizer.spheres)
  assert all(sphere[2] == (1.0, 0.0, 0.0, 1.0) for sphere in visualizer.spheres)


def test_compose_video_frame_draws_hud_and_side_panel() -> None:
  terrain = EvalTerrainSpec(
    name="upstairs_long_h15",
    label="h15",
    kind="upstairs",
    height_m=0.15,
    layout="long_runway",
  )
  snapshot = SemanticSnapshot(
    step=12,
    time_s=0.24,
    root_x_m=1.5,
    root_z_m=0.82,
    mode="WRITE",
    mode_code=1.0,
    event_prob=0.7,
    stair_prob=0.8,
    write_progress=0.45,
    memory_age=0.0,
    release_progress=0.0,
    pred_riser_height_m=0.14,
    true_riser_height_m=0.15,
    safe_stride_m=0.32,
    stride_lower_m=0.27,
    stride_upper_m=0.37,
    stride_trend=1.0,
    trend_label="FORWARD",
    trend_probs=(0.05, 0.15, 0.80),
    confidence=0.9,
    alpha_state=0.2,
    alpha_shape=0.05,
    foot_left=FootSnapshot("left", 1.2, 0.12, False, True),
    foot_right=FootSnapshot("right", 1.0, 0.02, True, False),
    toe_riser_contact_xz_m=((1.02, 0.15),),
  )
  rgb = np.zeros((64, 96, 3), dtype=np.uint8)
  cfg = SlowLatentSemanticVideoConfig(
    video_width=640,
    video_height=360,
    hud_width=180,
    side_panel_height=100,
  )

  frame = compose_video_frame(rgb, snapshot, [snapshot], terrain, cfg)

  assert frame.shape == (360, 640, 3)
  assert frame.dtype == np.uint8
  assert frame[:, -180:].mean() > 10.0
  assert frame[-100:].mean() > 10.0


def test_snapshot_row_flattens_foot_and_trend_fields() -> None:
  snapshot = SemanticSnapshot(
    step=1,
    time_s=0.02,
    root_x_m=0.1,
    root_z_m=0.4,
    mode="NORMAL",
    mode_code=0.0,
    event_prob=None,
    stair_prob=None,
    write_progress=None,
    memory_age=None,
    release_progress=None,
    pred_riser_height_m=None,
    true_riser_height_m=None,
    safe_stride_m=None,
    stride_lower_m=None,
    stride_upper_m=None,
    stride_trend=None,
    trend_label="UNKNOWN",
    trend_probs=(None, None, None),
    confidence=None,
    alpha_state=None,
    alpha_shape=None,
    foot_left=FootSnapshot("left", 0.2, 0.03, True, False),
    foot_right=None,
  )

  row = _snapshot_row(snapshot)

  assert row["left_foot_swing"] == 0
  assert row["left_foot_x_m"] == 0.2
  assert row["right_foot_swing"] is None
  assert row["trend_label"] == "UNKNOWN"
  assert row["toe_riser_contact_xz_m"] == "[]"
