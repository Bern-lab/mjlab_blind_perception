"""G1 footprint-detector rollout task with fixed terrain pools."""

from __future__ import annotations

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import BoxLongStairRunwayTerrainCfg, FlatPatchSamplingCfg
from mjlab.terrains.config import BLIND_HIGH_STAIRS_TREAD_DEPTHS

from .blind_rough_slow_latent_env_cfg import (
  unitree_g1_blind_rough_target_navigation_slow_latent_env_cfg,
)
from .env_cfgs import G1_HIGH_STAIRS_MIXED_REPLAY_LEVEL_RANGES

FOOTPRINT_DETECTOR_STANDALONE_FRACTION = 0.70
FOOTPRINT_DETECTOR_GRID_FRACTION = 1.0 - FOOTPRINT_DETECTOR_STANDALONE_FRACTION
FOOTPRINT_DETECTOR_LEVEL_RANGES = G1_HIGH_STAIRS_MIXED_REPLAY_LEVEL_RANGES
FOOTPRINT_DETECTOR_LEVEL_WEIGHTS = (0.15, 0.25, 0.60)
FOOTPRINT_DETECTOR_GRID_FLAT_RAW_WEIGHT = 0.10
FOOTPRINT_DETECTOR_RUNWAY_SIZE = (10.9, 2.5)
FOOTPRINT_DETECTOR_RUNWAY_PREFIX = "long_stair_runway_w"


def _rescale_grid_proportions(terrain_cfg) -> None:
  sub_terrains = {name: deepcopy(cfg) for name, cfg in terrain_cfg.sub_terrains.items()}
  raw_weights = {
    name: max(0.0, float(cfg.proportion)) for name, cfg in sub_terrains.items()
  }
  if "flat" in raw_weights:
    raw_weights["flat"] = max(
      raw_weights["flat"],
      FOOTPRINT_DETECTOR_GRID_FLAT_RAW_WEIGHT,
    )
  total = sum(raw_weights.values())
  if total <= 0.0 and raw_weights:
    raw_weights = {name: 1.0 for name in raw_weights}
    total = float(len(raw_weights))

  for name, sub_cfg in sub_terrains.items():
    sub_cfg.proportion = FOOTPRINT_DETECTOR_GRID_FRACTION * raw_weights[name] / total
  terrain_cfg.sub_terrains = sub_terrains


def _footprint_detector_terrain_cfg(terrain_generator):
  terrain_cfg = deepcopy(terrain_generator)
  _rescale_grid_proportions(terrain_cfg)

  runway_weight = FOOTPRINT_DETECTOR_STANDALONE_FRACTION / len(
    BLIND_HIGH_STAIRS_TREAD_DEPTHS
  )
  terrain_cfg.standalone_terrains = {
    f"{FOOTPRINT_DETECTOR_RUNWAY_PREFIX}{index:02d}": BoxLongStairRunwayTerrainCfg(
      proportion=runway_weight,
      size=FOOTPRINT_DETECTOR_RUNWAY_SIZE,
      step_height_range=(0.04, 0.2),
      step_width_range=(tread_depth, tread_depth),
      num_steps=14,
      start_platform_length=2.0,
      end_platform_length=4.0,
      end_target_fraction=0.85,
      flat_patch_sampling={
        "target": FlatPatchSamplingCfg(
          num_patches=1,
          patch_radius=0.25,
          max_height_diff=0.02,
        )
      },
    )
    for index, tread_depth in enumerate(BLIND_HIGH_STAIRS_TREAD_DEPTHS)
  }
  return terrain_cfg


def unitree_g1_footprint_detector_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the rollout env used only by the deployable footprint detector."""
  cfg = unitree_g1_blind_rough_target_navigation_slow_latent_env_cfg(play=play)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator = _footprint_detector_terrain_cfg(
      cfg.scene.terrain.terrain_generator
    )
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.scene.terrain.standalone_spawn_start_level = 0

  if not play and "terrain_levels" in cfg.curriculum:
    cfg.curriculum["terrain_levels"].func = mdp.fixed_pool_terrain_levels_vel
    cfg.curriculum["terrain_levels"].params = {
      "standalone_fraction": FOOTPRINT_DETECTOR_STANDALONE_FRACTION,
      "level_ranges": FOOTPRINT_DETECTOR_LEVEL_RANGES,
      "level_weights": FOOTPRINT_DETECTOR_LEVEL_WEIGHTS,
    }

  return cfg
