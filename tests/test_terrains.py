"""Tests for terrain generation."""

import mujoco
import numpy as np

from mjlab.terrains.primitive_terrains import (
  BoxInvertedPyramidStairsTerrainCfg,
  BoxLongStairRunwayTerrainCfg,
  BoxPyramidStairsTerrainCfg,
  BoxSteppingStonesTerrainCfg,
)
from mjlab.terrains.terrain_generator import (
  FlatPatchSamplingCfg,
  StepDangerVisualizationCfg,
  TerrainGenerator,
  TerrainGeneratorCfg,
)

_CFG = BoxSteppingStonesTerrainCfg(
  proportion=1.0,
  size=(8.0, 8.0),
  stone_size_range=(0.2, 0.6),
  stone_distance_range=(0.05, 0.25),
  stone_height=0.2,
  stone_height_variation=0.05,
  stone_size_variation=0.05,
  displacement_range=0.1,
  floor_depth=2.0,
  platform_width=1.5,
  border_width=0.25,
)


def _generate_stones(
  cfg: BoxSteppingStonesTerrainCfg,
  difficulty: float,
  rng: np.random.Generator,
) -> list[tuple[float, float, float, float]]:
  """Generate terrain and return stone (cx, cy, half_x, half_y) tuples."""
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(difficulty=difficulty, spec=spec, rng=rng)

  center = cfg.size[0] / 2
  stones = []
  for geom_info in output.geometries:
    geom = geom_info.geom
    if geom is None:
      continue
    pos, size = geom.pos, geom.size
    # Skip platform, floor, and border geoms.
    is_platform = (
      np.isclose(pos[0], center)
      and np.isclose(pos[1], center)
      and np.isclose(size[0], cfg.platform_width / 2, atol=1e-4)
    )
    is_full_span = np.isclose(size[0], cfg.size[0] / 2) or np.isclose(
      size[1], cfg.size[1] / 2
    )
    if is_platform or is_full_span:
      continue
    stones.append((pos[0], pos[1], size[0], size[1]))
  return stones


def test_no_stone_centers_inside_platform():
  """No stone center should fall inside the platform."""
  center = _CFG.size[0] / 2
  p_half = _CFG.platform_width / 2
  p_min, p_max = center - p_half, center + p_half

  for difficulty in [0.0, 0.5, 1.0]:
    stones = _generate_stones(_CFG, difficulty, np.random.default_rng(42))
    for cx, cy, _, _ in stones:
      assert not (p_min <= cx <= p_max and p_min <= cy <= p_max), (
        f"Stone at ({cx:.3f}, {cy:.3f}) inside platform at difficulty={difficulty}"
      )


def test_stone_size_decreases_with_difficulty():
  """Average stone size should be smaller at higher difficulty."""
  sizes = {}
  for difficulty in [0.0, 1.0]:
    stones = _generate_stones(_CFG, difficulty, np.random.default_rng(42))
    sizes[difficulty] = np.mean([hx + hy for _, _, hx, hy in stones])

  assert sizes[0.0] > sizes[1.0]


def test_pyramid_stairs_step_boundaries_use_high_side_lip():
  cfg = BoxPyramidStairsTerrainCfg(
    size=(8.0, 8.0),
    step_height_range=(0.1, 0.1),
    step_width=0.3,
    platform_width=3.0,
    border_width=1.0,
  )
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(0.0, spec, np.random.default_rng(0))

  assert output.step_boundaries is not None
  assert output.step_boundary_sequence_ids is not None
  assert output.step_boundary_layers is not None
  np.testing.assert_allclose(output.step_boundaries[0, 0:3], [1.0, 7.0, 0.1])
  np.testing.assert_allclose(output.step_boundaries[0, 3:6], [7.0, 7.0, 0.1])
  np.testing.assert_allclose(output.step_boundaries[0, 6:9], [0.0, 1.0, 0.0])
  np.testing.assert_allclose(output.step_boundaries[0, 9:11], [0.0, 0.1])
  np.testing.assert_array_equal(output.step_boundary_sequence_ids[:4], [1, 2, 3, 4])
  np.testing.assert_array_equal(
    output.step_boundary_layers[:8], [1, 1, 1, 1, 2, 2, 2, 2]
  )


def test_inverted_pyramid_stairs_step_boundaries_point_to_low_side():
  cfg = BoxInvertedPyramidStairsTerrainCfg(
    size=(8.0, 8.0),
    step_height_range=(0.1, 0.1),
    step_width=0.3,
    platform_width=3.0,
    border_width=1.0,
  )
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(0.0, spec, np.random.default_rng(0))

  assert output.step_boundaries is not None
  assert output.step_boundary_sequence_ids is not None
  assert output.step_boundary_layers is not None
  np.testing.assert_allclose(output.step_boundaries[0, 0:3], [1.0, 7.0, 0.0])
  np.testing.assert_allclose(output.step_boundaries[0, 3:6], [7.0, 7.0, 0.0])
  np.testing.assert_allclose(output.step_boundaries[0, 6:9], [0.0, -1.0, 0.0])
  np.testing.assert_allclose(output.step_boundaries[0, 9:11], [-0.1, 0.0])
  num_layers = len(output.step_boundary_layers) // 4
  np.testing.assert_array_equal(
    output.step_boundary_layers[:4],
    np.full(4, num_layers),
  )
  np.testing.assert_array_equal(output.step_boundary_layers[-4:], np.ones(4))


def test_terrain_generator_pads_step_boundaries_by_tile():
  cfg = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    num_rows=1,
    num_cols=1,
    seed=0,
    sub_terrains={
      "stairs": BoxPyramidStairsTerrainCfg(
        step_height_range=(0.1, 0.1),
        step_width=0.3,
        platform_width=3.0,
        border_width=1.0,
      )
    },
  )
  generator = TerrainGenerator(cfg)
  spec = mujoco.MjSpec()
  generator.compile(spec)

  assert generator.step_boundary_counts.shape == (1, 1)
  assert generator.step_boundary_counts[0, 0] == 24
  assert generator.step_boundaries_by_tile.shape == (1, 1, 24, 11)
  assert generator.step_boundary_sequence_ids_by_tile.shape == (1, 1, 24)
  assert generator.step_boundary_layers_by_tile.shape == (1, 1, 24)
  np.testing.assert_allclose(
    generator.step_boundaries_by_tile[0, 0, 0, 0:3], [-3.0, 3.0, 0.1]
  )
  np.testing.assert_array_equal(
    generator.step_boundary_sequence_ids_by_tile[0, 0, :4],
    [1, 2, 3, 4],
  )
  np.testing.assert_array_equal(
    generator.step_boundary_layers_by_tile[0, 0, :8],
    [1, 1, 1, 1, 2, 2, 2, 2],
  )


def test_long_stair_runway_has_fixed_start_target_and_step_sequence():
  cfg = BoxLongStairRunwayTerrainCfg(
    size=(8.9, 2.5),
    step_height_range=(0.1, 0.1),
    step_width_range=(0.3, 0.3),
    num_steps=14,
    start_platform_length=2.0,
    end_platform_length=2.0,
    flat_patch_sampling={
      "target": FlatPatchSamplingCfg(num_patches=1),
    },
  )
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(0.0, spec, np.random.default_rng(0))

  np.testing.assert_allclose(output.origin, [1.0, 1.25, 0.0])
  assert output.flat_patches is not None
  np.testing.assert_allclose(output.flat_patches["target"][0], [7.2, 1.25, 1.4])
  assert output.bounds == (0.0, 8.2, 0.0, 2.5)
  assert output.step_boundaries is not None
  assert output.step_boundaries.shape == (14, 11)
  np.testing.assert_allclose(output.step_boundaries[0, 0:3], [2.0, 0.0, 0.1])
  np.testing.assert_allclose(output.step_boundaries[0, 3:6], [2.0, 2.5, 0.1])
  np.testing.assert_allclose(output.step_boundaries[0, 6:9], [-1.0, 0.0, 0.0])
  np.testing.assert_allclose(output.step_boundaries[0, 9:11], [0.0, 0.1])
  np.testing.assert_array_equal(output.step_boundary_sequence_ids, np.ones(14))
  np.testing.assert_array_equal(output.step_boundary_layers, np.arange(1, 15))


def test_long_stair_runway_can_place_target_near_end_platform():
  cfg = BoxLongStairRunwayTerrainCfg(
    size=(10.9, 2.5),
    step_height_range=(0.1, 0.1),
    step_width_range=(0.3, 0.3),
    num_steps=14,
    start_platform_length=2.0,
    end_platform_length=4.0,
    end_target_fraction=0.85,
    flat_patch_sampling={
      "target": FlatPatchSamplingCfg(num_patches=1),
    },
  )
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(0.0, spec, np.random.default_rng(0))

  assert output.flat_patches is not None
  np.testing.assert_allclose(output.flat_patches["target"][0], [9.6, 1.25, 1.4])
  assert output.bounds == (0.0, 10.2, 0.0, 2.5)


def test_generator_places_standalone_runway_as_extra_spawn_type():
  cfg = TerrainGeneratorCfg(
    size=(4.0, 4.0),
    num_rows=2,
    num_cols=1,
    curriculum=True,
    seed=0,
    sub_terrains={
      "flat": BoxPyramidStairsTerrainCfg(
        proportion=1.0,
        step_height_range=(0.0, 0.0),
        step_width=0.4,
        platform_width=2.0,
        border_width=0.5,
      )
    },
    standalone_terrains={
      "runway": BoxLongStairRunwayTerrainCfg(
        proportion=1.0,
        size=(6.0, 2.0),
        step_height_range=(0.1, 0.1),
        step_width_range=(0.3, 0.3),
        num_steps=4,
        flat_patch_sampling={"target": FlatPatchSamplingCfg(num_patches=1)},
      )
    },
  )
  generator = TerrainGenerator(cfg)
  spec = mujoco.MjSpec()
  generator.compile(spec)

  assert generator.terrain_origins.shape == (2, 2, 3)
  assert generator.terrain_type_names == ("flat", "runway")
  np.testing.assert_array_equal(
    generator.standalone_terrain_type_mask,
    [False, True],
  )
  assert generator.flat_patches["target"].shape == (2, 2, 1, 3)
  assert generator.step_boundary_counts[0, 1] == 4


def test_step_danger_visualization_adds_non_colliding_geoms():
  cfg = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    num_rows=1,
    num_cols=1,
    seed=0,
    step_danger_visualization=StepDangerVisualizationCfg(enabled=True, geom_group=4),
    sub_terrains={
      "stairs": BoxPyramidStairsTerrainCfg(
        step_height_range=(0.1, 0.1),
        step_width=0.3,
        platform_width=3.0,
        border_width=1.0,
      )
    },
  )
  generator = TerrainGenerator(cfg)
  spec = mujoco.MjSpec()
  generator.compile(spec)

  danger_geoms = [geom for geom in spec.body("terrain").geoms if geom.group == 4]
  assert len(danger_geoms) == 2 * generator.step_boundary_counts[0, 0]
  assert all(geom.contype == 0 for geom in danger_geoms)
  assert all(geom.conaffinity == 0 for geom in danger_geoms)
