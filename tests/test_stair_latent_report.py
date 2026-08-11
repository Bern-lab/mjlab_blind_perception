from __future__ import annotations

import numpy as np
from scripts.velocity_eval.eval_terrains import (
  get_terrain_set,
  make_fixed_terrain_generator,
)
from scripts.velocity_eval.plot_stair_latent_report import (
  PlotStairLatentReportConfig,
  _embedding_coordinates,
  _pca2,
  _sample_filter,
  _select_trace,
  _size_labels,
  run_plot_report,
)

from mjlab.terrains import BoxLongStairRunwayTerrainCfg


def test_size_labels_include_stair_height_and_depth(tmp_path) -> None:
  path = tmp_path / "latents.npz"
  np.savez(
    path,
    latent=np.zeros((2, 3), dtype=np.float32),
    terrain_label=np.asarray(["flat", "stair"]),
    terrain_kind=np.asarray(["flat", "upstairs"]),
    terrain_height_m=np.asarray([0.0, 0.15], dtype=np.float32),
    terrain_tread_depth_m=np.asarray([0.0, 0.30], dtype=np.float32),
  )

  with np.load(path, allow_pickle=False) as data:
    assert _size_labels(data).tolist() == ["flat", "h15_d30"]


def test_select_trace_prefers_episode_with_stair_evidence(tmp_path) -> None:
  path = tmp_path / "latents.npz"
  np.savez(
    path,
    latent=np.zeros((4, 3), dtype=np.float32),
    episode_id=np.asarray([0, 0, 1, 1], dtype=np.int64),
    terrain_label=np.asarray(["flat", "flat", "h15_d30", "h15_d30"]),
    terrain_height_m=np.asarray([0.0, 0.0, 0.15, 0.15], dtype=np.float32),
    terrain_tread_depth_m=np.asarray([0.0, 0.0, 0.30, 0.30], dtype=np.float32),
    stair_prob=np.asarray([[0.1], [0.2], [0.4], [0.9]], dtype=np.float32),
    event_prob=np.asarray([[0.1], [0.2], [0.3], [0.8]], dtype=np.float32),
    gate_mode=np.asarray([[0.0], [0.0], [1.0], [2.0]], dtype=np.float32),
  )

  with np.load(path, allow_pickle=False) as data:
    selection = _select_trace(data, PlotStairLatentReportConfig(input_file=str(path)))

  assert selection.episode_id == 1
  assert selection.terrain_label == "h15_d30"


def test_pca2_handles_constant_features() -> None:
  pcs, explained = _pca2(np.ones((5, 4), dtype=np.float32))

  assert pcs.shape == (5, 2)
  np.testing.assert_allclose(pcs, 0.0)
  np.testing.assert_allclose(explained, 0.0)


def test_tsne_embedding_coordinates_are_three_dimensional() -> None:
  rng = np.random.default_rng(123)
  features = rng.normal(size=(12, 5)).astype(np.float32)

  coords, metrics = _embedding_coordinates(
    features,
    PlotStairLatentReportConfig(
      input_file="unused.npz",
      embedding_method="tsne",
      embedding_dim=3,
      tsne_perplexity=3.0,
      tsne_init="random",
      tsne_max_iter=250,
    ),
  )

  assert coords.shape == (12, 3)
  assert metrics["embedding_method"] == "tsne"
  assert metrics["embedding_dim"] == 3


def test_long_stair_riser_grid_uses_training_runway_geometry() -> None:
  terrains = get_terrain_set("long_stair_riser_grid_v1")

  assert [terrain.label for terrain in terrains] == ["flat", "h10", "h15", "h20"]
  runway = terrains[1]
  assert runway.kind == "upstairs"
  assert runway.layout == "long_runway"
  completion_x = runway.stair_completion_root_x_m()
  assert completion_x is not None
  assert abs(completion_x - 5.2) < 1.0e-9

  generator = make_fixed_terrain_generator(runway, num_envs=3, seed=123)
  assert generator.size == (10.9, 2.5)
  subterrain = generator.sub_terrains[runway.name]
  assert isinstance(subterrain, BoxLongStairRunwayTerrainCfg)
  assert subterrain.step_height_range == (0.10, 0.10)
  assert subterrain.step_width_range == (0.30, 0.30)
  assert subterrain.num_steps == 14
  assert subterrain.start_platform_length == 2.0
  assert subterrain.end_platform_length == 4.0
  assert subterrain.end_target_fraction == 0.85


def test_sample_filter_can_select_upstairs_memory_rows(tmp_path) -> None:
  path = tmp_path / "latents.npz"
  np.savez(
    path,
    latent=np.zeros((5, 3), dtype=np.float32),
    terrain_kind=np.asarray(["flat", "upstairs", "upstairs", "upstairs", "rough"]),
    gate_mode=np.asarray([[0.0], [1.0], [2.0], [2.0], [2.0]], dtype=np.float32),
    gate_memory_age=np.asarray([[0.0], [0.0], [0.1], [0.4], [0.5]], dtype=np.float32),
    gait_phase=np.asarray([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32),
  )

  with np.load(path, allow_pickle=False) as data:
    mask = _sample_filter(
      data,
      PlotStairLatentReportConfig(
        input_file=str(path),
        terrain_kind_filter="upstairs",
        gate_mode_filter="memory",
        min_memory_age=0.2,
      ),
    )

  np.testing.assert_array_equal(mask, [False, False, False, True, False])


def test_run_plot_report_writes_core_figures(tmp_path) -> None:
  path = tmp_path / "latents.npz"
  n = 8
  steps = np.arange(n, dtype=np.float32)
  rng = np.random.default_rng(123)
  np.savez(
    path,
    latent=rng.normal(size=(n, 4)).astype(np.float32),
    hidden=rng.normal(size=(n, 5)).astype(np.float32),
    z_memory=rng.normal(size=(n, 6)).astype(np.float32),
    z_state_memory=rng.normal(size=(n, 2)).astype(np.float32),
    z_shape_memory=rng.normal(size=(n, 4)).astype(np.float32),
    semantic=np.linspace(0.0, 1.0, n * 16, dtype=np.float32).reshape(n, 16),
    episode_id=np.zeros(n, dtype=np.int64),
    terrain_label=np.asarray(["h15_d30"] * n),
    terrain_kind=np.asarray(["upstairs"] * n),
    terrain_height_m=np.full(n, 0.15, dtype=np.float32),
    terrain_tread_depth_m=np.full(n, 0.30, dtype=np.float32),
    time_s=steps * 0.02,
    time_step=steps.astype(np.int64),
    gait_phase=(steps / n).astype(np.float32),
    gate_mode=np.asarray([[0], [0], [1], [1], [2], [2], [2], [2]], dtype=np.float32),
    event_prob=np.linspace(0.1, 0.9, n, dtype=np.float32)[:, None],
    stair_prob=np.linspace(0.2, 0.95, n, dtype=np.float32)[:, None],
    alpha_state=np.linspace(0.3, 0.0, n, dtype=np.float32)[:, None],
    alpha_shape=np.linspace(0.3, 0.05, n, dtype=np.float32)[:, None],
    stair_shape=np.column_stack(
      [
        np.linspace(0.25, 0.35, n, dtype=np.float32),
        np.linspace(0.10, 0.15, n, dtype=np.float32),
      ]
    ),
    safe_stride=np.linspace(0.25, 0.32, n, dtype=np.float32)[:, None],
    safe_stride_interval=np.column_stack(
      [
        np.linspace(0.22, 0.27, n, dtype=np.float32),
        np.linspace(0.31, 0.36, n, dtype=np.float32),
      ]
    ),
    safe_stride_confidence=np.linspace(0.2, 0.8, n, dtype=np.float32)[:, None],
  )

  out_dir = tmp_path / "report"
  summary = run_plot_report(
    PlotStairLatentReportConfig(
      input_file=str(path),
      output_dir=str(out_dir),
      max_samples=None,
      trace_max_steps=None,
    )
  )

  assert (out_dir / "latent_state_switch_episode.png").exists()
  assert (out_dir / "semantic_dynamics_episode.png").exists()
  assert (out_dir / "z_memory_heatmap_episode.png").exists()
  assert (out_dir / "semantic_correlation_matrix.png").exists()
  assert (out_dir / "semantic_invariant_report.json").exists()
  assert (out_dir / "representation_probe_summary.csv").exists()
  assert (out_dir / "hidden_cluster_pca.png").exists()
  assert (out_dir / "interactive" / "hidden_pca3d.html").exists()
  assert (out_dir / "interactive" / "z_shape_pca3d.html").exists()
  assert (out_dir / "static" / "z_shape_pca3d_xy.png").exists()
  html = (out_dir / "interactive" / "z_shape_pca3d.html").read_text(encoding="utf-8")
  assert "hiddenCategories" in html
  assert "categoryIsHidden" in html
  assert summary["trace_selection"] == {
    "episode_id": 0,
    "terrain_label": "h15_d30",
    "samples": n,
  }
  interactive = summary["interactive_embeddings"]
  assert isinstance(interactive, list)
  assert len(interactive) >= 2
