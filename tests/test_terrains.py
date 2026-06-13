"""Tests for terrain generation."""

import mujoco
import numpy as np
import torch

from mjlab.terrains.primitive_terrains import (
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
  BoxSteppingStoneGridTerrainCfg,
  BoxSteppingStonesTerrainCfg,
)
from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg
from mjlab.terrains.terrain_generator import (
  StepDangerVisualizationCfg,
  TerrainGenerator,
  TerrainGeneratorCfg,
  TerrainOutput,
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


def _stepping_stone_grid_cfg(inverted: bool) -> BoxSteppingStoneGridTerrainCfg:
  return BoxSteppingStoneGridTerrainCfg(
    size=(8.0, 8.0),
    stone_size_start=0.60,
    stone_size_end=0.30,
    stone_height_start=0.08,
    stone_height_end=0.30,
    gap_start=0.16,
    gap_end=0.40,
    jitter_start=0.0,
    jitter_end=0.0,
    num_rows=8,
    num_cols=8,
    platform_width=0.8,
    border_width=0.5,
    floor_clearance=0.15,
    inverted_rim_height_start=0.0,
    inverted_rim_height_end=0.30,
    inverted=inverted,
  )


def _generate_stepping_stone_grid(
  cfg: BoxSteppingStoneGridTerrainCfg,
  difficulty: float,
):
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  return cfg.function(difficulty=difficulty, spec=spec, rng=np.random.default_rng(0))


def _target_platform_gap(
  cfg: BoxSteppingStoneGridTerrainCfg,
  difficulty: float,
) -> float:
  output = _generate_stepping_stone_grid(cfg, difficulty)
  targets = _stepping_stone_grid_stone_centers(output)
  center = np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0])
  platform_half = cfg.platform_width / 2.0
  stone_size = cfg.stone_size_start + difficulty * (
    cfg.stone_size_end - cfg.stone_size_start
  )
  half_s = stone_size / 2.0
  separation_xy = np.maximum(np.abs(targets - center) - (platform_half + half_s), 0.0)
  return float(np.linalg.norm(separation_xy, axis=1).min())


def _stepping_stone_grid_stone_centers(output: TerrainOutput) -> np.ndarray:
  # Four border geoms and one center platform precede the stone geoms.
  return np.array(
    [
      geom_info.geom.pos[:2]
      for geom_info in output.geometries[5:]
      if geom_info.geom is not None
    ]
  )


def test_stepping_stone_grid_has_open_gaps_without_inner_floor():
  """Both stepping-stone grid variants should leave missed footholds open."""
  for inverted in (False, True):
    output = _generate_stepping_stone_grid(
      _stepping_stone_grid_cfg(inverted=inverted), 0.5
    )
    assert output.flat_patches is not None
    # Four border geoms, one center platform, then exactly one geom per stone.
    assert len(output.geometries) == len(_stepping_stone_grid_stone_centers(output)) + 5


def test_stepping_stone_grid_targets_only_center_platform():
  """Target flat patches should stay on the center platform, not on stones."""
  center_xy = np.array([4.0, 4.0])
  for inverted in (False, True):
    cfg = _stepping_stone_grid_cfg(inverted=inverted)
    for difficulty in (0.0, 0.5, 1.0):
      output = _generate_stepping_stone_grid(cfg, difficulty)
      assert output.flat_patches is not None
      targets = output.flat_patches["target"]
      assert targets.shape == (1, 3)
      np.testing.assert_allclose(targets[0, :2], center_xy)
      np.testing.assert_allclose(targets[0, 2], output.origin[2])


def test_stepping_stone_grid_outputs_step_boundaries_for_danger_rewards():
  """Stepping-stone grids should expose edges used by danger-zone rewards."""
  for inverted in (False, True):
    output = _generate_stepping_stone_grid(
      _stepping_stone_grid_cfg(inverted=inverted), 0.5
    )
    assert output.step_boundaries is not None
    assert output.step_boundaries.shape[1] == 11
    assert len(output.step_boundaries) > 0
    assert np.all(output.step_boundaries[:, 10] > output.step_boundaries[:, 9])


def test_stepping_stone_grid_inverted_rim_height_uses_curriculum():
  """Inverted grids should raise the rim/platform while stone tops stay at 0."""
  cfg = _stepping_stone_grid_cfg(inverted=True)
  for difficulty, expected_rim_z in ((0.0, 0.0), (0.5, 0.15), (1.0, 0.30)):
    output = _generate_stepping_stone_grid(cfg, difficulty)
    assert output.flat_patches is not None
    np.testing.assert_allclose(output.origin[2], expected_rim_z)
    np.testing.assert_allclose(output.flat_patches["target"][:, 2], expected_rim_z)

    border_tops = [
      geom_info.geom.pos[2] + geom_info.geom.size[2]
      for geom_info in output.geometries[:4]
      if geom_info.geom is not None
    ]
    np.testing.assert_allclose(border_tops, expected_rim_z)

    platform_geom = output.geometries[4].geom
    assert platform_geom is not None
    platform_top = platform_geom.pos[2] + platform_geom.size[2]
    np.testing.assert_allclose(platform_top, expected_rim_z)


def test_stepping_stone_grid_normal_origin_and_targets_follow_stone_height():
  """Normal grids should keep the rim at 0 and raise stones/platforms."""
  cfg = _stepping_stone_grid_cfg(inverted=False)
  for difficulty, expected_stone_z in ((0.0, 0.08), (0.5, 0.19), (1.0, 0.30)):
    output = _generate_stepping_stone_grid(cfg, difficulty)
    assert output.flat_patches is not None
    np.testing.assert_allclose(output.origin[2], expected_stone_z)
    np.testing.assert_allclose(output.flat_patches["target"][:, 2], expected_stone_z)

    border_tops = [
      geom_info.geom.pos[2] + geom_info.geom.size[2]
      for geom_info in output.geometries[:4]
      if geom_info.geom is not None
    ]
    np.testing.assert_allclose(border_tops, 0.0)

    platform_geom = output.geometries[4].geom
    assert platform_geom is not None
    platform_top = platform_geom.pos[2] + platform_geom.size[2]
    np.testing.assert_allclose(platform_top, expected_stone_z)


def test_stepping_stone_grid_gap_increases_with_difficulty():
  """Platform-to-stone and stone-to-stone gaps should follow curriculum difficulty."""
  for inverted in (False, True):
    cfg = _stepping_stone_grid_cfg(inverted=inverted)
    gaps = [_target_platform_gap(cfg, difficulty) for difficulty in (0.0, 0.5, 1.0)]
    np.testing.assert_allclose(gaps, [0.16, 0.28, 0.40], atol=1e-6)
    assert gaps[0] < gaps[1] < gaps[2]


def test_stepping_stone_grid_stones_do_not_overlap():
  """Generated grid stones should preserve a positive gap at every difficulty."""
  for inverted in (False, True):
    cfg = _stepping_stone_grid_cfg(inverted=inverted)
    for difficulty in (0.0, 0.5, 1.0):
      output = _generate_stepping_stone_grid(cfg, difficulty)
      targets = _stepping_stone_grid_stone_centers(output)
      stone_size = cfg.stone_size_start + difficulty * (
        cfg.stone_size_end - cfg.stone_size_start
      )
      for i in range(len(targets)):
        delta = np.abs(targets[i + 1 :] - targets[i])
        overlaps = (delta[:, 0] < stone_size) & (delta[:, 1] < stone_size)
        assert not overlaps.any()


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
  np.testing.assert_allclose(output.step_boundaries[0, 0:3], [1.0, 7.0, 0.1])
  np.testing.assert_allclose(output.step_boundaries[0, 3:6], [7.0, 7.0, 0.1])
  np.testing.assert_allclose(output.step_boundaries[0, 6:9], [0.0, 1.0, 0.0])
  np.testing.assert_allclose(output.step_boundaries[0, 9:11], [0.0, 0.1])


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
  np.testing.assert_allclose(output.step_boundaries[0, 0:3], [1.0, 7.0, 0.0])
  np.testing.assert_allclose(output.step_boundaries[0, 3:6], [7.0, 7.0, 0.0])
  np.testing.assert_allclose(output.step_boundaries[0, 6:9], [0.0, -1.0, 0.0])
  np.testing.assert_allclose(output.step_boundaries[0, 9:11], [-0.1, 0.0])


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
  np.testing.assert_allclose(
    generator.step_boundaries_by_tile[0, 0, 0, 0:3], [-3.0, 3.0, 0.1]
  )


def test_terrain_entity_exposes_current_step_boundaries():
  cfg = TerrainEntityCfg(
    terrain_type="generator",
    num_envs=4,
    max_init_terrain_level=1,
    terrain_generator=TerrainGeneratorCfg(
      size=(8.0, 8.0),
      num_rows=2,
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
    ),
  )
  terrain = TerrainEntity(cfg, device="cpu")

  assert terrain.step_boundaries.shape == (cfg.num_envs, 24, 11)
  assert terrain.step_boundaries_valid.shape == (cfg.num_envs, 24)
  assert terrain.step_boundaries_valid.dtype is torch.bool

  for env_id in range(cfg.num_envs):
    level = terrain.terrain_levels[env_id]
    terrain_type = terrain.terrain_types[env_id]
    assert torch.allclose(
      terrain.step_boundaries[env_id],
      terrain.step_boundaries_by_tile[level, terrain_type],
    )
    assert terrain.step_boundaries_valid[env_id].all()


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
