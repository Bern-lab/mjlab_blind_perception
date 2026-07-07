"""Export Stage 2A stair probe histories from a frozen velocity policy."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast
from xml.etree import ElementTree as ET

import numpy as np
import torch
import tyro
from scripts.velocity_eval.policy_io import (
  load_inference_policy,
  resolve_checkpoint_path,
)
from tqdm.auto import tqdm

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.mdp.stair_geometry import (
  COLLISION_RISK_KEY,
  LANDING_QUALITY_KEY,
  LANDING_TOUCHDOWN_KEY,
  MINIMUM_SAFE_STRIDE_KEY,
  MINIMUM_SAFE_STRIDE_UPPER_KEY,
  MINIMUM_SAFE_STRIDE_VALID_KEY,
  SAFE_LANDING_CENTER_KEY,
  STAIR_CURRENT_CONTACT_DURATION_KEY,
  STAIR_CURRENT_GROUND_CONTACT_KEY,
  STAIR_CURRENT_STAIR_SUPPORT_KEY,
  STAIR_CURRENT_SUPPORT_FRACTION_KEY,
  STAIR_CURRENT_SUPPORT_LAYER_KEY,
  STAIR_PHASE_KEY,
  STAIR_RISER_HEIGHT_LABEL_KEY,
  STAIR_SEQUENCE_ID_KEY,
  STAIR_SHAPE_LABEL_VALID_KEY,
  STAIR_TREAD_DEPTH_LABEL_KEY,
  TOE_RISER_NEW_HIT_BY_FOOT_KEY,
  TOE_RISER_NEW_HIT_KEY,
)
from mjlab.tasks.velocity.mdp.stair_sequence_logging import (
  stair_sequence_event_logger,
)
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat
from mjlab.utils.lstm import reset_policy_state_from_step
from mjlab.utils.torch import configure_torch_backends

LEVEL_DELTA_NO_TRANSITION = 0
LEVEL_DELTA_UP_ONE = 1
LEVEL_DELTA_DOWN_ONE = 2
LEVEL_DELTA_SKIP_OR_UNCERTAIN = 3
LEVEL_DELTA_NAMES = (
  "no_transition",
  "up_one_level",
  "down_one_level",
  "skip_or_uncertain",
)

SAMPLE_TYPE_TO_ID = {
  "transition_up": 0,
  "transition_down": 1,
  "transition_skip": 2,
  "stair_same_level": 3,
  "stable_landing": 4,
  "full_landing": 5,
  "partial_uncertain": 6,
  "flat_negative": 7,
}

INPUT_FEATURE_GROUPS: tuple[tuple[str, int], ...] = (
  ("projected_gravity", 3),
  ("base_ang_vel", 3),
  ("base_ang_vel_delta", 3),
  ("left_toe_pos_body", 3),
  ("right_toe_pos_body", 3),
  ("left_heel_pos_body", 3),
  ("right_heel_pos_body", 3),
  ("toe_delta_pos", 3),
  ("heel_delta_pos", 3),
  ("toe_horizontal_distance", 1),
  ("toe_vertical_distance", 1),
  ("heel_vertical_distance", 1),
  ("left_toe_vel_body", 3),
  ("right_toe_vel_body", 3),
  ("left_toe_vel_delta", 3),
  ("right_toe_vel_delta", 3),
  ("previous_action_leg", 12),
  ("leg_joint_tracking_error", 12),
  ("leg_joint_vel", 12),
  ("leg_joint_vel_delta", 12),
  ("command_lin_x", 1),
)

FORBIDDEN_INPUT_FIELDS = (
  "support_layer",
  "riser_layer",
  "true_tread_depth",
  "true_riser_height",
  "depth_bin",
  "world_foot_s",
  "riser_s",
  "terrain_height",
  "height_scan",
  "camera_depth",
  "contact_force",
  "contact_bit",
  "support_ratio",
  "support_fraction",
  "geometric_overlap",
)

PRIVILEGED_FOOTPRINT_FEATURE_GROUPS: tuple[tuple[str, int], ...] = (
  ("left_ground_contact", 1),
  ("right_ground_contact", 1),
  ("left_stair_support", 1),
  ("right_stair_support", 1),
  ("left_tread_support_fraction", 1),
  ("right_tread_support_fraction", 1),
  ("left_support_layer_norm", 1),
  ("right_support_layer_norm", 1),
  ("left_contact_duration_norm", 1),
  ("right_contact_duration_norm", 1),
  ("both_ground_contact", 1),
  ("both_stair_support", 1),
  ("pair_layer_delta_abs_norm", 1),
  ("pair_adjacent_layers", 1),
  ("pair_min_support_fraction", 1),
  ("pair_mean_support_fraction", 1),
  ("any_partial_tread_footprint", 1),
  ("pair_full_tread_footprint", 1),
)
"""Simulation-only footprint anchor features for privileged Stage 2 diagnostics."""

SPARSE_FOOT_EVENT_FEATURE_GROUPS: tuple[tuple[str, int], ...] = (
  ("event_present", 1),
  ("tread_touchdown_event", 1),
  ("toe_riser_hit_event", 1),
  ("left_foot", 1),
  ("right_foot", 1),
  ("age_norm", 1),
  ("confidence", 1),
  ("point_s_current_base_yaw", 1),
  ("point_y_current_base_yaw", 1),
  ("point_z_current_base_yaw", 1),
  ("toe_s_current_base_yaw", 1),
  ("toe_y_current_base_yaw", 1),
  ("heel_s_current_base_yaw", 1),
  ("heel_y_current_base_yaw", 1),
  ("foot_half_width", 1),
  ("gait_phase_sin", 1),
  ("gait_phase_cos", 1),
  ("foot_vx_body", 1),
  ("foot_vz_body", 1),
  ("contact_duration_norm", 1),
  ("impact_strength_norm", 1),
  ("previous_event_delta_s_current_base_yaw", 1),
  ("previous_event_delta_z_current_base_yaw", 1),
  ("same_foot_delta_s_current_base_yaw", 1),
  ("same_foot_delta_z_current_base_yaw", 1),
)
"""Deployable-style sparse foot-event memory features for Stage 2C ablations."""

SPARSE_EVENT_AGE_FEATURE_INDEX = 5


@dataclass(frozen=True)
class SparseFootprintGeometry:
  """Footprint dimensions derived from the robot XML collision capsules."""

  rear_x: float
  front_x: float
  half_width: float
  toe_anchor_x: float
  heel_anchor_x: float

  @property
  def touchdown_anchor_x(self) -> float:
    """Return the local x used by the current touchdown anchor."""
    return 0.5 * (self.toe_anchor_x + self.heel_anchor_x)


def _parse_float_tuple(value: str) -> tuple[float, ...]:
  return tuple(float(part) for part in value.split())


def load_g1_sparse_footprint_geometry(
  xml_path: Path | None = None,
) -> SparseFootprintGeometry:
  """Load approximate G1 sole dimensions from foot collision geoms in XML."""
  if xml_path is None:
    xml_path = (
      Path(__file__).resolve().parents[2]
      / "src"
      / "mjlab"
      / "asset_zoo"
      / "robots"
      / "unitree_g1"
      / "xmls"
      / "g1.xml"
    )
  root = ET.parse(xml_path).getroot()
  default_radius = 0.0
  for default in root.iter("default"):
    if default.attrib.get("class") != "foot_capsule":
      continue
    geom = default.find("geom")
    if geom is not None and "size" in geom.attrib:
      default_radius = _parse_float_tuple(geom.attrib["size"])[0]
      break

  x_min = float("inf")
  x_max = float("-inf")
  y_min = float("inf")
  y_max = float("-inf")
  for geom in root.iter("geom"):
    name = geom.attrib.get("name", "")
    if not name.startswith(("left_foot", "right_foot")):
      continue
    if not name.endswith("_collision") or "fromto" not in geom.attrib:
      continue
    fromto = _parse_float_tuple(geom.attrib["fromto"])
    if len(fromto) != 6:
      raise ValueError(f"Expected 6-value fromto for {name}, got {fromto}.")
    radius = (
      _parse_float_tuple(geom.attrib["size"])[0]
      if "size" in geom.attrib
      else default_radius
    )
    xs = (fromto[0], fromto[3])
    ys = (fromto[1], fromto[4])
    x_min = min(x_min, min(xs) - radius)
    x_max = max(x_max, max(xs) + radius)
    y_min = min(y_min, min(ys) - radius)
    y_max = max(y_max, max(ys) + radius)

  site_x: dict[str, float] = {}
  for site in root.iter("site"):
    name = site.attrib.get("name", "")
    if name in ("left_toe", "right_toe", "left_heel", "right_heel"):
      pos = _parse_float_tuple(site.attrib["pos"])
      site_x[name] = pos[0]

  required_sites = ("left_toe", "right_toe", "left_heel", "right_heel")
  if not all(name in site_x for name in required_sites):
    missing = [name for name in required_sites if name not in site_x]
    raise ValueError(f"Missing G1 foot sites in {xml_path}: {missing}.")
  if not all(np.isfinite(value) for value in (x_min, x_max, y_min, y_max)):
    raise ValueError(f"No G1 foot collision capsules found in {xml_path}.")
  return SparseFootprintGeometry(
    rear_x=x_min,
    front_x=x_max,
    half_width=max(abs(y_min), abs(y_max)),
    toe_anchor_x=0.5 * (site_x["left_toe"] + site_x["right_toe"]),
    heel_anchor_x=0.5 * (site_x["left_heel"] + site_x["right_heel"]),
  )


@dataclass(frozen=True)
class ExportStairProbeDatasetConfig:
  """Configuration for Stage 2A probe dataset export."""

  checkpoint_file: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  output_dir: str = "eval_outputs/stair_stage2/model51000_seed42_probe_v1"
  num_envs: int = 512
  steps: int = 5000
  seed: int = 42
  device: str | None = None
  history_len: int = 64
  include_privileged_footprint: bool = False
  privileged_footprint_history_len: int = 128
  include_sparse_foot_events: bool = False
  sparse_event_memory_len: int = 8
  sparse_event_age_norm_s: float = 3.0
  sparse_event_position_scale_m: float = 1.0
  sparse_event_velocity_scale_mps: float = 2.0
  sparse_event_gait_period: float = 0.6
  max_samples: int = 200_000
  expected_obs_dim: int = 91
  stable_support_fraction: float = 0.75
  full_support_fraction: float = 0.99
  partial_support_fraction: float = 0.25
  stable_contact_time: float = 0.06
  flat_sample_period: int = 25
  same_level_sample_period: int = 20
  landing_sample_period: int = 8
  partial_sample_period: int = 20
  progress: bool = True


@dataclass(frozen=True)
class StairProbeLabelBatch:
  """Frame-aligned privileged labels and sampling hints."""

  level_delta_label: torch.Tensor
  relative_level_label: torch.Tensor
  level_transition_label: torch.Tensor
  true_tread_depth: torch.Tensor
  true_riser_height: torch.Tensor
  depth_bin_label: torch.Tensor
  depth_3bin_label: torch.Tensor
  depth_valid_label: torch.Tensor
  safe_landing_center: torch.Tensor
  minimum_safe_stride: torch.Tensor
  maximum_safe_stride: torch.Tensor
  safe_stride_valid_label: torch.Tensor
  landing_touchdown_label: torch.Tensor
  landing_quality_label: torch.Tensor
  collision_risk_label: torch.Tensor
  num_confirmed_layers: torch.Tensor
  stair_active_label: torch.Tensor
  full_landing_label: torch.Tensor
  stable_landing_label: torch.Tensor
  partial_uncertain_label: torch.Tensor
  sequence_id: torch.Tensor


def input_obs_dim() -> int:
  """Return the Stage 2A v1 deployable latent observation dimension."""
  return sum(width for _name, width in INPUT_FEATURE_GROUPS)


def privileged_footprint_obs_dim() -> int:
  """Return the Stage 2 privileged footprint observation dimension."""
  return sum(width for _name, width in PRIVILEGED_FOOTPRINT_FEATURE_GROUPS)


def sparse_foot_event_obs_dim() -> int:
  """Return the Stage 2C sparse foot-event observation dimension."""
  return sum(width for _name, width in SPARSE_FOOT_EVENT_FEATURE_GROUPS)


def input_feature_slices() -> dict[str, slice]:
  """Return fixed slices for named Stage 2A latent-observation groups."""
  offset = 0
  slices: dict[str, slice] = {}
  for name, width in INPUT_FEATURE_GROUPS:
    slices[name] = slice(offset, offset + width)
    offset += width
  return slices


def forbidden_input_feature_names() -> tuple[str, ...]:
  """Return feature names that would violate the no-privileged-input contract."""
  feature_names = [name for name, _width in INPUT_FEATURE_GROUPS]
  return tuple(name for name in feature_names if name in FORBIDDEN_INPUT_FIELDS)


def depth_bin_from_tread_depth(depth: torch.Tensor) -> torch.Tensor:
  """Map 0.25--0.35 m curriculum tread depths to eight bins."""
  bin_width = depth.new_tensor((0.35 - 0.25) / 7.0)
  return torch.round((depth - 0.25) / bin_width).long().clamp(0, 7)


def depth_3bin_from_depth_bin(depth_bin: torch.Tensor) -> torch.Tensor:
  """Coarsen eight tread-depth bins into shallow/medium/deep."""
  medium = torch.ones_like(depth_bin)
  deep = torch.full_like(depth_bin, 2)
  return torch.where(depth_bin <= 2, torch.zeros_like(depth_bin), medium).where(
    depth_bin <= 4,
    deep,
  )


class StairProbeHistoryBuffer:
  """Per-env oldest-to-newest latent observation history with reset masking."""

  def __init__(
    self,
    *,
    num_envs: int,
    history_len: int,
    obs_dim: int,
    device: torch.device | str,
  ) -> None:
    if num_envs <= 0:
      raise ValueError("num_envs must be positive.")
    if history_len <= 0:
      raise ValueError("history_len must be positive.")
    if obs_dim <= 0:
      raise ValueError("obs_dim must be positive.")
    self.history = torch.zeros(num_envs, history_len, obs_dim, device=device)
    self.valid_mask = torch.zeros(
      num_envs, history_len, dtype=torch.bool, device=device
    )

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Clear all or selected environment histories."""
    if env_ids is None:
      self.history.zero_()
      self.valid_mask.zero_()
      return
    if env_ids.numel() == 0:
      return
    self.history[env_ids] = 0.0
    self.valid_mask[env_ids] = False

  def push(
    self,
    obs: torch.Tensor,
    reset_mask: torch.Tensor | None = None,
  ) -> None:
    """Append one frame after clearing reset environments."""
    if obs.shape != (self.history.shape[0], self.history.shape[2]):
      raise ValueError(
        f"Expected obs shape {(self.history.shape[0], self.history.shape[2])}, "
        f"got {tuple(obs.shape)}."
      )
    if reset_mask is not None:
      self.reset(reset_mask.nonzero(as_tuple=False).squeeze(-1))
    self.history[:, :-1].copy_(self.history[:, 1:].clone())
    self.valid_mask[:, :-1].copy_(self.valid_mask[:, 1:].clone())
    self.history[:, -1] = obs
    self.valid_mask[:, -1] = True


class StairProbeLevelState:
  """Privileged label-only state machine for relative support levels."""

  def __init__(
    self,
    *,
    num_envs: int,
    stable_support_fraction: float = 0.75,
    full_support_fraction: float = 0.99,
    partial_support_fraction: float = 0.25,
    stable_contact_time: float = 0.06,
    device: torch.device | str = "cpu",
  ) -> None:
    self.num_envs = int(num_envs)
    self.device = torch.device(device)
    self.stable_support_fraction = float(stable_support_fraction)
    self.full_support_fraction = float(full_support_fraction)
    self.partial_support_fraction = float(partial_support_fraction)
    self.stable_contact_time = float(stable_contact_time)
    self.confirmed_layer = torch.zeros(self.num_envs, dtype=torch.long, device=device)
    self.relative_level = torch.zeros(self.num_envs, dtype=torch.long, device=device)
    self.previous_sequence_id = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=device
    )
    self.dataset_sequence_id = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=device
    )
    self._next_dataset_sequence_id = 0
    self._confirmed_layers: list[set[int]] = [set() for _ in range(self.num_envs)]

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Clear label state for all or selected environments."""
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    if env_ids.numel() == 0:
      return
    self.confirmed_layer[env_ids] = 0
    self.relative_level[env_ids] = 0
    self.previous_sequence_id[env_ids] = -1
    self.dataset_sequence_id[env_ids] = -1
    for env_id in env_ids.detach().cpu().tolist():
      self._confirmed_layers[int(env_id)].clear()

  def update(
    self,
    *,
    stair_active: torch.Tensor,
    sequence_id: torch.Tensor,
    support_layer: torch.Tensor,
    stair_support: torch.Tensor,
    support_fraction: torch.Tensor,
    contact_duration: torch.Tensor,
    true_tread_depth: torch.Tensor,
    true_riser_height: torch.Tensor | None = None,
    safe_landing_center: torch.Tensor | None = None,
    minimum_safe_stride: torch.Tensor | None = None,
    maximum_safe_stride: torch.Tensor | None = None,
    safe_stride_valid: torch.Tensor | None = None,
    landing_touchdown: torch.Tensor | None = None,
    landing_quality: torch.Tensor | None = None,
    collision_risk: torch.Tensor | None = None,
    shape_valid: torch.Tensor | None = None,
    reset_mask: torch.Tensor | None = None,
  ) -> StairProbeLabelBatch:
    """Advance label state by one frame and return current labels."""
    stair_active = stair_active.to(self.device, dtype=torch.bool)
    sequence_id = sequence_id.to(self.device, dtype=torch.long)
    support_layer = support_layer.to(self.device, dtype=torch.long)
    stair_support = stair_support.to(self.device, dtype=torch.bool)
    support_fraction = support_fraction.to(self.device, dtype=torch.float32)
    contact_duration = contact_duration.to(self.device, dtype=torch.float32)
    true_tread_depth = true_tread_depth.to(self.device, dtype=torch.float32)
    if true_riser_height is None:
      true_riser_height = torch.zeros_like(true_tread_depth)
    true_riser_height = true_riser_height.to(self.device, dtype=torch.float32)
    if safe_landing_center is None:
      safe_landing_center = torch.zeros_like(true_tread_depth)
    safe_landing_center = safe_landing_center.to(self.device, dtype=torch.float32)
    if minimum_safe_stride is None:
      minimum_safe_stride = torch.zeros_like(true_tread_depth)
    minimum_safe_stride = minimum_safe_stride.to(self.device, dtype=torch.float32)
    if maximum_safe_stride is None:
      maximum_safe_stride = minimum_safe_stride
    maximum_safe_stride = maximum_safe_stride.to(self.device, dtype=torch.float32)
    if safe_stride_valid is None:
      safe_stride_valid = torch.zeros_like(stair_active)
    safe_stride_valid = safe_stride_valid.to(self.device, dtype=torch.bool)
    if landing_touchdown is None:
      landing_touchdown = torch.zeros_like(stair_active)
    landing_touchdown = landing_touchdown.to(self.device, dtype=torch.bool)
    if landing_quality is None:
      landing_quality = torch.zeros_like(true_tread_depth)
    landing_quality = landing_quality.to(self.device, dtype=torch.float32)
    if collision_risk is None:
      collision_risk = torch.zeros_like(true_tread_depth)
    collision_risk = collision_risk.to(self.device, dtype=torch.float32)
    if shape_valid is None:
      shape_valid = torch.isfinite(true_tread_depth) & (true_tread_depth > 0.0)
    else:
      shape_valid = shape_valid.to(self.device, dtype=torch.bool)

    reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    if reset_mask is not None:
      reset |= reset_mask.to(self.device, dtype=torch.bool)
    reset |= ~stair_active
    reset |= (sequence_id != self.previous_sequence_id) & stair_active
    self.reset(reset.nonzero(as_tuple=False).squeeze(-1))
    self.previous_sequence_id = torch.where(
      stair_active,
      sequence_id,
      torch.full_like(sequence_id, -1),
    )
    new_active = stair_active & (self.dataset_sequence_id < 0)
    new_active_ids = new_active.nonzero(as_tuple=False).squeeze(-1)
    if new_active_ids.numel() > 0:
      next_id = self._next_dataset_sequence_id
      new_ids = torch.arange(
        next_id,
        next_id + int(new_active_ids.numel()),
        dtype=torch.long,
        device=self.device,
      )
      self.dataset_sequence_id[new_active_ids] = new_ids
      self._next_dataset_sequence_id += int(new_active_ids.numel())

    full_ok = (
      stair_active[:, None]
      & stair_support
      & (support_layer >= 0)
      & (support_fraction >= self.full_support_fraction)
    )
    stable_ok = (
      stair_active[:, None]
      & stair_support
      & (support_layer >= 0)
      & (support_fraction >= self.stable_support_fraction)
      & (contact_duration >= self.stable_contact_time)
    )
    confirmed_ok = full_ok | stable_ok
    candidate_layers = torch.where(
      confirmed_ok,
      support_layer,
      torch.full_like(support_layer, -1),
    )
    candidate_layer = candidate_layers.max(dim=-1).values
    has_candidate = candidate_layer >= 0
    candidate_is_full = torch.any(
      full_ok & (support_layer == candidate_layer[:, None]), dim=-1
    )
    candidate_is_stable = has_candidate & ~candidate_is_full

    level_delta = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    changed = has_candidate & (candidate_layer != self.confirmed_layer)
    raw_delta = candidate_layer - self.confirmed_layer
    up = changed & (raw_delta == 1)
    down = changed & (raw_delta == -1)
    skip = changed & ~(up | down)
    level_delta = torch.where(
      up,
      torch.full_like(level_delta, LEVEL_DELTA_UP_ONE),
      level_delta,
    )
    level_delta = torch.where(
      down,
      torch.full_like(level_delta, LEVEL_DELTA_DOWN_ONE),
      level_delta,
    )
    level_delta = torch.where(
      skip,
      torch.full_like(level_delta, LEVEL_DELTA_SKIP_OR_UNCERTAIN),
      level_delta,
    )

    if bool(torch.any(changed).item()):
      self.relative_level[changed] += raw_delta[changed]
      self.confirmed_layer[changed] = candidate_layer[changed]
      for env_id in changed.nonzero(as_tuple=False).squeeze(-1).cpu().tolist():
        layer = int(candidate_layer[int(env_id)].item())
        if layer > 0:
          self._confirmed_layers[int(env_id)].add(layer)

    num_confirmed = torch.tensor(
      [len(layers) for layers in self._confirmed_layers],
      dtype=torch.long,
      device=self.device,
    )
    partial = (
      stair_active[:, None]
      & stair_support
      & (support_layer >= 0)
      & (support_fraction >= self.partial_support_fraction)
      & ~confirmed_ok
    ).any(dim=-1)

    depth_valid = stair_active & shape_valid & (true_tread_depth > 0.0)
    depth_bin = depth_bin_from_tread_depth(true_tread_depth)
    depth_3bin = depth_3bin_from_depth_bin(depth_bin)
    return StairProbeLabelBatch(
      level_delta_label=level_delta.clone(),
      relative_level_label=self.relative_level.clone(),
      level_transition_label=changed.clone(),
      true_tread_depth=torch.where(
        depth_valid,
        true_tread_depth,
        torch.zeros_like(true_tread_depth),
      ),
      true_riser_height=torch.where(
        depth_valid,
        true_riser_height,
        torch.zeros_like(true_riser_height),
      ),
      depth_bin_label=torch.where(
        depth_valid, depth_bin, torch.full_like(depth_bin, -1)
      ),
      depth_3bin_label=torch.where(
        depth_valid,
        depth_3bin,
        torch.full_like(depth_3bin, -1),
      ),
      depth_valid_label=depth_valid.clone(),
      safe_landing_center=torch.where(
        safe_stride_valid,
        safe_landing_center,
        torch.zeros_like(safe_landing_center),
      ),
      minimum_safe_stride=torch.where(
        safe_stride_valid,
        minimum_safe_stride,
        torch.zeros_like(minimum_safe_stride),
      ),
      maximum_safe_stride=torch.where(
        safe_stride_valid,
        maximum_safe_stride,
        torch.zeros_like(maximum_safe_stride),
      ),
      safe_stride_valid_label=safe_stride_valid.clone(),
      landing_touchdown_label=landing_touchdown.clone(),
      landing_quality_label=torch.where(
        landing_touchdown,
        landing_quality.clamp(0.0, 1.0),
        torch.zeros_like(landing_quality),
      ),
      collision_risk_label=collision_risk.clamp(0.0, 1.0),
      num_confirmed_layers=num_confirmed,
      stair_active_label=stair_active.clone(),
      full_landing_label=candidate_is_full & ~changed,
      stable_landing_label=candidate_is_stable & ~changed,
      partial_uncertain_label=partial,
      sequence_id=torch.where(
        stair_active,
        self.dataset_sequence_id,
        torch.full_like(self.dataset_sequence_id, -1),
      ),
    )


class StairProbeDatasetBuilder:
  """Collect sampled histories and labels into NPZ-ready arrays."""

  def __init__(
    self,
    *,
    num_envs: int,
    history_len: int,
    obs_dim: int,
    max_samples: int,
    seed: int,
    device: torch.device | str,
    privileged_footprint_dim: int | None = None,
    privileged_footprint_history_len: int | None = None,
    sparse_event_dim: int | None = None,
    sparse_event_memory_len: int | None = None,
    flat_sample_period: int = 25,
    same_level_sample_period: int = 20,
    landing_sample_period: int = 8,
    partial_sample_period: int = 20,
  ) -> None:
    if max_samples <= 0:
      raise ValueError("max_samples must be positive.")
    self.history = StairProbeHistoryBuffer(
      num_envs=num_envs,
      history_len=history_len,
      obs_dim=obs_dim,
      device=device,
    )
    self.privileged_footprint_history: StairProbeHistoryBuffer | None = None
    if privileged_footprint_dim is not None:
      if privileged_footprint_history_len is None:
        raise ValueError(
          "privileged_footprint_history_len is required when footprint is enabled."
        )
      self.privileged_footprint_history = StairProbeHistoryBuffer(
        num_envs=num_envs,
        history_len=privileged_footprint_history_len,
        obs_dim=privileged_footprint_dim,
        device=device,
      )
    self._current_sparse_event_memory: torch.Tensor | None = None
    self._current_sparse_event_valid_mask: torch.Tensor | None = None
    if sparse_event_dim is not None:
      if sparse_event_memory_len is None:
        raise ValueError(
          "sparse_event_memory_len is required when sparse events are enabled."
        )
      self._current_sparse_event_memory = torch.zeros(
        num_envs,
        sparse_event_memory_len,
        sparse_event_dim,
        dtype=torch.float32,
        device=device,
      )
      self._current_sparse_event_valid_mask = torch.zeros(
        num_envs,
        sparse_event_memory_len,
        dtype=torch.bool,
        device=device,
      )
    self.max_samples = int(max_samples)
    self.seed = int(seed)
    self.flat_sample_period = max(1, int(flat_sample_period))
    self.same_level_sample_period = max(1, int(same_level_sample_period))
    self.landing_sample_period = max(1, int(landing_sample_period))
    self.partial_sample_period = max(1, int(partial_sample_period))
    self._arrays: dict[str, list[np.ndarray]] = {
      "obs_history": [],
      "obs_valid_mask": [],
      "level_delta_label": [],
      "relative_level_label": [],
      "level_transition_label": [],
      "true_tread_depth": [],
      "true_riser_height": [],
      "depth_bin_label": [],
      "depth_3bin_label": [],
      "depth_valid_label": [],
      "safe_landing_center": [],
      "minimum_safe_stride": [],
      "maximum_safe_stride": [],
      "safe_stride_valid_label": [],
      "landing_touchdown_label": [],
      "landing_quality_label": [],
      "collision_risk_label": [],
      "num_confirmed_layers": [],
      "stair_active_label": [],
      "sample_type": [],
      "sequence_id": [],
      "env_id": [],
      "frame_idx": [],
      "seed": [],
    }
    if self.privileged_footprint_history is not None:
      self._arrays["privileged_footprint_history"] = []
      self._arrays["privileged_footprint_valid_mask"] = []
    if self._current_sparse_event_memory is not None:
      self._arrays["sparse_foot_event_memory"] = []
      self._arrays["sparse_foot_event_valid_mask"] = []
    self._sample_type_names: list[str] = []

  @property
  def num_samples(self) -> int:
    """Return the number of accepted samples."""
    if not self._arrays["sample_type"]:
      return 0
    return int(sum(chunk.shape[0] for chunk in self._arrays["sample_type"]))

  @property
  def is_full(self) -> bool:
    """Whether the configured sample cap has been reached."""
    return self.num_samples >= self.max_samples

  def push_observations(
    self,
    obs: torch.Tensor,
    reset_mask: torch.Tensor | None = None,
  ) -> None:
    """Append deployable latent observations to the history buffer."""
    self.history.push(obs, reset_mask)

  def push_privileged_footprint(
    self,
    obs: torch.Tensor,
    reset_mask: torch.Tensor | None = None,
  ) -> None:
    """Append simulation-only footprint observations to their history buffer."""
    if self.privileged_footprint_history is None:
      raise RuntimeError("Privileged footprint history is not enabled.")
    self.privileged_footprint_history.push(obs, reset_mask)

  def set_sparse_foot_event_memory(
    self,
    memory: torch.Tensor,
    valid_mask: torch.Tensor,
  ) -> None:
    """Update the current causal sparse foot-event memory snapshot."""
    current_memory = self._current_sparse_event_memory
    current_mask = self._current_sparse_event_valid_mask
    if current_memory is None or current_mask is None:
      raise RuntimeError("Sparse foot-event memory is not enabled.")
    if memory.shape != current_memory.shape:
      raise ValueError(
        f"Expected sparse event memory shape "
        f"{tuple(current_memory.shape)}, got {tuple(memory.shape)}."
      )
    if valid_mask.shape != current_mask.shape:
      raise ValueError(
        f"Expected sparse event valid mask shape "
        f"{tuple(current_mask.shape)}, "
        f"got {tuple(valid_mask.shape)}."
      )
    current_memory.copy_(memory)
    current_mask.copy_(valid_mask)

  def collect(self, labels: StairProbeLabelBatch, frame_idx: int) -> None:
    """Sample histories for one rollout frame."""
    if self.is_full:
      return
    transition_up = labels.level_delta_label == LEVEL_DELTA_UP_ONE
    transition_down = labels.level_delta_label == LEVEL_DELTA_DOWN_ONE
    transition_skip = labels.level_delta_label == LEVEL_DELTA_SKIP_OR_UNCERTAIN
    non_transition = labels.level_delta_label == LEVEL_DELTA_NO_TRANSITION
    landing_period = frame_idx % self.landing_sample_period == 0
    partial_period = frame_idx % self.partial_sample_period == 0
    same_period = frame_idx % self.same_level_sample_period == 0
    flat_period = frame_idx % self.flat_sample_period == 0
    stair_same = (
      labels.stair_active_label
      & ~labels.level_transition_label
      & ~labels.full_landing_label
      & ~labels.stable_landing_label
      & ~labels.partial_uncertain_label
      & (labels.relative_level_label > 0)
    )
    sample_masks = (
      ("transition_up", transition_up),
      ("transition_down", transition_down),
      ("transition_skip", transition_skip),
      ("full_landing", labels.full_landing_label & landing_period),
      ("stable_landing", labels.stable_landing_label & landing_period),
      (
        "partial_uncertain",
        labels.partial_uncertain_label & non_transition & partial_period,
      ),
      ("stair_same_level", stair_same & same_period),
      ("flat_negative", ~labels.stair_active_label & flat_period),
    )
    for sample_type, mask in sample_masks:
      self._add_samples(sample_type, mask, labels, frame_idx)
      if self.is_full:
        break

  def _add_samples(
    self,
    sample_type: str,
    mask: torch.Tensor,
    labels: StairProbeLabelBatch,
    frame_idx: int,
  ) -> None:
    ids = mask.nonzero(as_tuple=False).squeeze(-1)
    if ids.numel() == 0:
      return
    remaining = self.max_samples - self.num_samples
    if remaining <= 0:
      return
    ids = ids[:remaining]
    sample_count = int(ids.numel())
    self._arrays["obs_history"].append(
      self.history.history[ids].detach().cpu().numpy().astype(np.float32)
    )
    self._arrays["obs_valid_mask"].append(
      self.history.valid_mask[ids].detach().cpu().numpy().astype(np.bool_)
    )
    if self.privileged_footprint_history is not None:
      self._arrays["privileged_footprint_history"].append(
        self.privileged_footprint_history.history[ids]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
      )
      self._arrays["privileged_footprint_valid_mask"].append(
        self.privileged_footprint_history.valid_mask[ids]
        .detach()
        .cpu()
        .numpy()
        .astype(np.bool_)
      )
    if self._current_sparse_event_memory is not None:
      assert self._current_sparse_event_valid_mask is not None
      self._arrays["sparse_foot_event_memory"].append(
        self._current_sparse_event_memory[ids].detach().cpu().numpy().astype(np.float32)
      )
      self._arrays["sparse_foot_event_valid_mask"].append(
        self._current_sparse_event_valid_mask[ids]
        .detach()
        .cpu()
        .numpy()
        .astype(np.bool_)
      )
    scalar_fields = (
      "level_delta_label",
      "relative_level_label",
      "level_transition_label",
      "true_tread_depth",
      "true_riser_height",
      "depth_bin_label",
      "depth_3bin_label",
      "depth_valid_label",
      "safe_landing_center",
      "minimum_safe_stride",
      "maximum_safe_stride",
      "safe_stride_valid_label",
      "landing_touchdown_label",
      "landing_quality_label",
      "collision_risk_label",
      "num_confirmed_layers",
      "stair_active_label",
      "sequence_id",
    )
    for field in scalar_fields:
      tensor = getattr(labels, field)
      self._arrays[field].append(tensor[ids].detach().cpu().numpy())
    sample_type_id = SAMPLE_TYPE_TO_ID[sample_type]
    self._arrays["sample_type"].append(
      np.full(sample_count, sample_type_id, dtype=np.int64)
    )
    self._arrays["env_id"].append(ids.detach().cpu().numpy().astype(np.int64))
    self._arrays["frame_idx"].append(
      np.full(sample_count, int(frame_idx), dtype=np.int64)
    )
    self._arrays["seed"].append(np.full(sample_count, self.seed, dtype=np.int64))
    self._sample_type_names.extend([sample_type] * sample_count)

  def as_arrays(self) -> dict[str, np.ndarray]:
    """Return concatenated arrays ready for ``np.savez``."""
    if self.num_samples == 0:
      raise RuntimeError("No probe samples were collected.")
    output = {
      key: np.concatenate(chunks, axis=0)
      for key, chunks in self._arrays.items()
      if chunks
    }
    output["sample_type_name"] = np.asarray(self._sample_type_names)
    return output


def _tensor_extra(
  env: ManagerBasedRlEnv,
  key: str,
  shape: tuple[int, ...],
  dtype: torch.dtype,
  fill_value: float | int | bool,
) -> torch.Tensor:
  value = env.extras.get(key)
  if isinstance(value, torch.Tensor):
    return value
  return torch.full((env.num_envs, *shape), fill_value, dtype=dtype, device=env.device)


class SparseFootEventMemory:
  """Causal memory of sparse foot events visible up to the current frame."""

  def __init__(
    self,
    *,
    num_envs: int,
    memory_len: int,
    device: torch.device | str,
    age_norm_s: float = 2.0,
    position_scale_m: float = 1.0,
    velocity_scale_mps: float = 2.0,
    gait_period: float = 0.6,
    footprint_geometry: SparseFootprintGeometry | None = None,
  ) -> None:
    if memory_len <= 0:
      raise ValueError("memory_len must be positive.")
    self.num_envs = int(num_envs)
    self.memory_len = int(memory_len)
    self.device = torch.device(device)
    self.age_norm_s = max(float(age_norm_s), 1.0e-6)
    self.position_scale_m = max(float(position_scale_m), 1.0e-6)
    self.velocity_scale_mps = max(float(velocity_scale_mps), 1.0e-6)
    self.gait_period = max(float(gait_period), 1.0e-6)
    self.footprint_geometry = (
      load_g1_sparse_footprint_geometry()
      if footprint_geometry is None
      else footprint_geometry
    )
    self.memory = torch.zeros(
      self.num_envs,
      self.memory_len,
      sparse_foot_event_obs_dim(),
      dtype=torch.float32,
      device=self.device,
    )
    self.valid_mask = torch.zeros(
      self.num_envs,
      self.memory_len,
      dtype=torch.bool,
      device=self.device,
    )
    self.event_pos_w = torch.zeros(
      self.num_envs,
      self.memory_len,
      3,
      dtype=torch.float32,
      device=self.device,
    )
    self.event_toe_pos_w = torch.zeros_like(self.event_pos_w)
    self.event_heel_pos_w = torch.zeros_like(self.event_pos_w)
    self.previous_ground_contact = torch.zeros(
      self.num_envs,
      2,
      dtype=torch.bool,
      device=self.device,
    )
    self.previous_ground_contact_valid = torch.zeros(
      self.num_envs,
      dtype=torch.bool,
      device=self.device,
    )
    self.last_event_pos_w = torch.zeros(
      self.num_envs,
      3,
      dtype=torch.float32,
      device=device,
    )
    self.last_event_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
    self.last_foot_pos_w = torch.zeros(
      self.num_envs,
      2,
      3,
      dtype=torch.float32,
      device=device,
    )
    self.last_foot_valid = torch.zeros(
      self.num_envs, 2, dtype=torch.bool, device=device
    )

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Clear memory and edge-detector state for selected environments."""
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    if env_ids.numel() == 0:
      return
    self.memory[env_ids] = 0.0
    self.valid_mask[env_ids] = False
    self.event_pos_w[env_ids] = 0.0
    self.event_toe_pos_w[env_ids] = 0.0
    self.event_heel_pos_w[env_ids] = 0.0
    self.previous_ground_contact[env_ids] = False
    self.previous_ground_contact_valid[env_ids] = False
    self.last_event_pos_w[env_ids] = 0.0
    self.last_event_valid[env_ids] = False
    self.last_foot_pos_w[env_ids] = 0.0
    self.last_foot_valid[env_ids] = False

  def begin_step(self, reset_mask: torch.Tensor | None, step_dt: float) -> None:
    """Age existing events and clear reset environments before new events arrive."""
    if reset_mask is not None:
      self.reset(reset_mask.nonzero(as_tuple=False).squeeze(-1))
    age_delta = float(step_dt) / self.age_norm_s
    self.memory[..., SPARSE_EVENT_AGE_FEATURE_INDEX] = torch.where(
      self.valid_mask,
      (self.memory[..., SPARSE_EVENT_AGE_FEATURE_INDEX] + age_delta).clamp(0.0, 1.0),
      self.memory[..., SPARSE_EVENT_AGE_FEATURE_INDEX],
    )
    expired = self.valid_mask & (
      self.memory[..., SPARSE_EVENT_AGE_FEATURE_INDEX] >= 1.0
    )
    if bool(expired.any().item()):
      self.valid_mask[expired] = False
      self.memory[expired] = 0.0
      self.event_pos_w[expired] = 0.0
      self.event_toe_pos_w[expired] = 0.0
      self.event_heel_pos_w[expired] = 0.0
      self._refresh_last_event_references()

  def update_from_env_and_latent(
    self,
    env: ManagerBasedRlEnv,
    latent_obs: torch.Tensor,
  ) -> None:
    """Append touchdown and toe-riser hit events detected on this frame."""
    if latent_obs.shape != (self.num_envs, input_obs_dim()):
      raise ValueError(
        f"Expected latent obs shape {(self.num_envs, input_obs_dim())}, "
        f"got {tuple(latent_obs.shape)}."
      )
    slices = input_feature_slices()
    left_toe_pos = latent_obs[:, slices["left_toe_pos_body"]]
    right_toe_pos = latent_obs[:, slices["right_toe_pos_body"]]
    left_heel_pos = latent_obs[:, slices["left_heel_pos_body"]]
    right_heel_pos = latent_obs[:, slices["right_heel_pos_body"]]
    left_toe_vel = latent_obs[:, slices["left_toe_vel_body"]]
    right_toe_vel = latent_obs[:, slices["right_toe_vel_body"]]
    foot_points = torch.stack(
      [
        0.5 * (left_toe_pos + left_heel_pos),
        0.5 * (right_toe_pos + right_heel_pos),
      ],
      dim=1,
    )
    toe_points = torch.stack([left_toe_pos, right_toe_pos], dim=1)
    heel_points = torch.stack([left_heel_pos, right_heel_pos], dim=1)
    foot_velocities = torch.stack([left_toe_vel, right_toe_vel], dim=1)
    root_pos_w, root_quat_w = self._root_pose_from_env(env)
    self._refresh_relative_event_features(root_pos_w, root_quat_w)

    ground_contact = _tensor_extra(
      env, STAIR_CURRENT_GROUND_CONTACT_KEY, (2,), torch.bool, False
    )
    contact_duration = _tensor_extra(
      env, STAIR_CURRENT_CONTACT_DURATION_KEY, (2,), torch.float32, 0.0
    )
    toe_riser_hit_extra = env.extras.get(TOE_RISER_NEW_HIT_BY_FOOT_KEY)
    if isinstance(toe_riser_hit_extra, torch.Tensor):
      toe_riser_hit = toe_riser_hit_extra.bool()
      if toe_riser_hit.ndim == 1:
        toe_riser_hit = toe_riser_hit[:, None].expand(-1, 2)
    else:
      toe_riser_hit = _tensor_extra(
        env, TOE_RISER_NEW_HIT_KEY, (2,), torch.bool, False
      ).bool()
      if toe_riser_hit.ndim == 1:
        toe_riser_hit = toe_riser_hit[:, None].expand(-1, 2)

    touchdown = (
      ground_contact.bool()
      & ~self.previous_ground_contact
      & self.previous_ground_contact_valid[:, None]
    )
    phase = (
      (env.episode_length_buf.to(dtype=torch.float32) * float(env.step_dt))
      % self.gait_period
    ) / self.gait_period
    phase_sin = torch.sin(phase * torch.pi * 2.0)
    phase_cos = torch.cos(phase * torch.pi * 2.0)

    for foot_id in range(2):
      self._push_event(
        mask=touchdown[:, foot_id],
        foot_id=foot_id,
        event_type="tread_touchdown",
        point=foot_points[:, foot_id],
        toe_point=toe_points[:, foot_id],
        heel_point=heel_points[:, foot_id],
        velocity=foot_velocities[:, foot_id],
        root_pos_w=root_pos_w,
        root_quat_w=root_quat_w,
        confidence=torch.ones(self.num_envs, dtype=torch.float32, device=self.device),
        contact_duration=contact_duration[:, foot_id],
        phase_sin=phase_sin,
        phase_cos=phase_cos,
      )
      self._push_event(
        mask=toe_riser_hit[:, foot_id].bool(),
        foot_id=foot_id,
        event_type="toe_riser_hit",
        point=toe_points[:, foot_id],
        toe_point=toe_points[:, foot_id],
        heel_point=heel_points[:, foot_id],
        velocity=foot_velocities[:, foot_id],
        root_pos_w=root_pos_w,
        root_quat_w=root_quat_w,
        confidence=torch.ones(self.num_envs, dtype=torch.float32, device=self.device),
        contact_duration=contact_duration[:, foot_id],
        phase_sin=phase_sin,
        phase_cos=phase_cos,
      )

    self.previous_ground_contact.copy_(ground_contact.bool())
    self.previous_ground_contact_valid[:] = True

  def _push_event(
    self,
    *,
    mask: torch.Tensor,
    foot_id: int,
    event_type: str,
    point: torch.Tensor,
    toe_point: torch.Tensor,
    heel_point: torch.Tensor,
    velocity: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    confidence: torch.Tensor,
    contact_duration: torch.Tensor,
    phase_sin: torch.Tensor,
    phase_cos: torch.Tensor,
  ) -> None:
    ids = mask.nonzero(as_tuple=False).squeeze(-1)
    if ids.numel() == 0:
      return
    feature = torch.zeros(
      self.num_envs,
      sparse_foot_event_obs_dim(),
      dtype=torch.float32,
      device=self.device,
    )
    point_w = root_pos_w + quat_apply(root_quat_w, point)
    toe_pos_w = root_pos_w + quat_apply(root_quat_w, toe_point)
    heel_pos_w = root_pos_w + quat_apply(root_quat_w, heel_point)
    point_rel = self._world_to_current_base_yaw(point_w, root_pos_w, root_quat_w)
    toe_rel = self._world_to_current_base_yaw(toe_pos_w, root_pos_w, root_quat_w)
    heel_rel = self._world_to_current_base_yaw(heel_pos_w, root_pos_w, root_quat_w)
    previous_event_rel = self._world_to_current_base_yaw(
      self.last_event_pos_w,
      root_pos_w,
      root_quat_w,
    )
    previous_foot_rel = self._world_to_current_base_yaw(
      self.last_foot_pos_w[:, foot_id],
      root_pos_w,
      root_quat_w,
    )
    point_s = point_rel[:, 0]
    point_y = point_rel[:, 1]
    point_z = point_rel[:, 2]
    previous_delta_s = torch.where(
      self.last_event_valid,
      point_s - previous_event_rel[:, 0],
      torch.zeros_like(point_s),
    )
    previous_delta_z = torch.where(
      self.last_event_valid,
      point_z - previous_event_rel[:, 2],
      torch.zeros_like(point_z),
    )
    same_delta_s = torch.where(
      self.last_foot_valid[:, foot_id],
      point_s - previous_foot_rel[:, 0],
      torch.zeros_like(point_s),
    )
    same_delta_z = torch.where(
      self.last_foot_valid[:, foot_id],
      point_z - previous_foot_rel[:, 2],
      torch.zeros_like(point_z),
    )
    half_width = torch.full(
      (self.num_envs,),
      self.footprint_geometry.half_width / self.position_scale_m,
      dtype=torch.float32,
      device=self.device,
    )
    feature[:, 0] = 1.0
    feature[:, 1] = 1.0 if event_type == "tread_touchdown" else 0.0
    feature[:, 2] = 1.0 if event_type == "toe_riser_hit" else 0.0
    feature[:, 3] = 1.0 if foot_id == 0 else 0.0
    feature[:, 4] = 1.0 if foot_id == 1 else 0.0
    feature[:, 5] = 0.0
    feature[:, 6] = confidence.clamp(0.0, 1.0)
    feature[:, 7] = point_s / self.position_scale_m
    feature[:, 8] = point_y / self.position_scale_m
    feature[:, 9] = point_z / self.position_scale_m
    feature[:, 10] = toe_rel[:, 0] / self.position_scale_m
    feature[:, 11] = toe_rel[:, 1] / self.position_scale_m
    feature[:, 12] = heel_rel[:, 0] / self.position_scale_m
    feature[:, 13] = heel_rel[:, 1] / self.position_scale_m
    feature[:, 14] = half_width
    feature[:, 15] = phase_sin
    feature[:, 16] = phase_cos
    feature[:, 17] = velocity[:, 0] / self.velocity_scale_mps
    feature[:, 18] = velocity[:, 2] / self.velocity_scale_mps
    feature[:, 19] = (contact_duration / self.age_norm_s).clamp(0.0, 1.0)
    feature[:, 20] = (
      torch.linalg.norm(velocity, dim=-1) / self.velocity_scale_mps
    ).clamp(0.0, 1.0)
    feature[:, 21] = previous_delta_s / self.position_scale_m
    feature[:, 22] = previous_delta_z / self.position_scale_m
    feature[:, 23] = same_delta_s / self.position_scale_m
    feature[:, 24] = same_delta_z / self.position_scale_m

    self.memory[ids, 1:].copy_(self.memory[ids, :-1].clone())
    self.valid_mask[ids, 1:].copy_(self.valid_mask[ids, :-1].clone())
    self.event_pos_w[ids, 1:].copy_(self.event_pos_w[ids, :-1].clone())
    self.event_toe_pos_w[ids, 1:].copy_(self.event_toe_pos_w[ids, :-1].clone())
    self.event_heel_pos_w[ids, 1:].copy_(self.event_heel_pos_w[ids, :-1].clone())
    self.memory[ids, 0] = feature[ids]
    self.valid_mask[ids, 0] = True
    self.event_pos_w[ids, 0] = point_w[ids]
    self.event_toe_pos_w[ids, 0] = toe_pos_w[ids]
    self.event_heel_pos_w[ids, 0] = heel_pos_w[ids]
    self.last_event_pos_w[ids] = point_w[ids]
    self.last_event_valid[ids] = True
    self.last_foot_pos_w[ids, foot_id] = point_w[ids]
    self.last_foot_valid[ids, foot_id] = True

  def _refresh_relative_event_features(
    self,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
  ) -> None:
    """Re-project stored world event points into the current base-yaw frame."""
    if not bool(self.valid_mask.any().item()):
      return
    root_pos = root_pos_w[:, None, :].expand(-1, self.memory_len, -1)
    root_quat = root_quat_w[:, None, :].expand(-1, self.memory_len, -1)
    rel = self._world_to_current_base_yaw(self.event_pos_w, root_pos, root_quat)
    toe_rel = self._world_to_current_base_yaw(
      self.event_toe_pos_w,
      root_pos,
      root_quat,
    )
    heel_rel = self._world_to_current_base_yaw(
      self.event_heel_pos_w,
      root_pos,
      root_quat,
    )
    rel_scaled = rel / self.position_scale_m
    self.memory[..., 7:10] = torch.where(
      self.valid_mask[..., None],
      rel_scaled,
      torch.zeros_like(rel_scaled),
    )
    toe_heel_scaled = torch.cat(
      (
        toe_rel[..., 0:2] / self.position_scale_m,
        heel_rel[..., 0:2] / self.position_scale_m,
      ),
      dim=-1,
    )
    self.memory[..., 10:14] = torch.where(
      self.valid_mask[..., None],
      toe_heel_scaled,
      torch.zeros_like(toe_heel_scaled),
    )

  def _refresh_last_event_references(self) -> None:
    """Refresh delta-reference events after age-based invalidation."""
    slot0_valid = self.valid_mask[:, 0]
    self.last_event_valid.copy_(slot0_valid)
    self.last_event_pos_w.copy_(
      torch.where(
        slot0_valid[:, None],
        self.event_pos_w[:, 0],
        torch.zeros_like(self.last_event_pos_w),
      )
    )
    env_ids = torch.arange(self.num_envs, device=self.device)
    for foot_id, feature_index in enumerate((3, 4)):
      foot_valid = self.valid_mask & (self.memory[..., feature_index] > 0.5)
      has_foot = foot_valid.any(dim=1)
      first_idx = foot_valid.to(dtype=torch.int64).argmax(dim=1)
      selected = self.event_pos_w[env_ids, first_idx]
      self.last_foot_valid[:, foot_id].copy_(has_foot)
      self.last_foot_pos_w[:, foot_id].copy_(
        torch.where(
          has_foot[:, None],
          selected,
          torch.zeros_like(selected),
        )
      )

  def _root_pose_from_env(
    self,
    env: ManagerBasedRlEnv,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return root pose in world coordinates, with a test-friendly fallback."""
    scene = getattr(env, "scene", None)
    if scene is not None:
      try:
        robot = scene["robot"]
        return (
          robot.data.root_link_pos_w.to(self.device, dtype=torch.float32),
          robot.data.root_link_quat_w.to(self.device, dtype=torch.float32),
        )
      except (KeyError, AttributeError, TypeError):
        pass
    root_pos = getattr(env, "root_pos_w", None)
    root_quat = getattr(env, "root_quat_w", None)
    if isinstance(root_pos, torch.Tensor) and isinstance(root_quat, torch.Tensor):
      return (
        root_pos.to(self.device, dtype=torch.float32),
        root_quat.to(self.device, dtype=torch.float32),
      )
    pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
    quat = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
    quat[:, 0] = 1.0
    return pos, quat

  @staticmethod
  def _world_to_current_base_yaw(
    point_w: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
  ) -> torch.Tensor:
    quat_shape = root_quat_w.shape
    root_yaw = yaw_quat(root_quat_w.reshape(-1, 4)).reshape(quat_shape)
    return quat_apply_inverse(root_yaw, point_w - root_pos_w)


def privileged_footprint_features_from_tensors(
  *,
  ground_contact: torch.Tensor,
  stair_support: torch.Tensor,
  support_fraction: torch.Tensor,
  support_layer: torch.Tensor,
  contact_duration: torch.Tensor,
  max_layer: float = 8.0,
  max_contact_duration: float = 1.0,
) -> torch.Tensor:
  """Build simulation-only footprint features from current support diagnostics.

  ``support_fraction`` is computed upstream from robot foot sole samples against
  the active stair tread geometry. For partial footholds, it represents only the
  fraction of the foot sole footprint that lies on the tread.
  """
  if ground_contact.shape[-1] != 2:
    raise ValueError("privileged footprint features require exactly two feet.")
  if stair_support.shape != ground_contact.shape:
    raise ValueError("stair_support shape must match ground_contact.")
  if support_fraction.shape != ground_contact.shape:
    raise ValueError("support_fraction shape must match ground_contact.")
  if support_layer.shape != ground_contact.shape:
    raise ValueError("support_layer shape must match ground_contact.")
  if contact_duration.shape != ground_contact.shape:
    raise ValueError("contact_duration shape must match ground_contact.")

  max_layer_tensor = support_fraction.new_tensor(max(max_layer, 1.0e-6))
  max_duration_tensor = support_fraction.new_tensor(max(max_contact_duration, 1.0e-6))
  ground = ground_contact.to(dtype=torch.float32)
  support = stair_support.to(dtype=torch.float32)
  fraction = support_fraction.to(dtype=torch.float32).clamp(0.0, 1.0)
  layer_float = support_layer.to(dtype=torch.float32)
  layer_valid = support_layer >= 0
  layer_norm = torch.where(
    layer_valid,
    layer_float.clamp(0.0, float(max_layer)) / max_layer_tensor,
    torch.zeros_like(layer_float),
  )
  duration_norm = (
    contact_duration.to(dtype=torch.float32).clamp(0.0, float(max_contact_duration))
    / max_duration_tensor
  )

  both_ground = torch.all(ground_contact.bool(), dim=-1, keepdim=True)
  both_support = torch.all(stair_support.bool(), dim=-1, keepdim=True)
  both_layer_valid = torch.all(layer_valid, dim=-1, keepdim=True)
  layer_delta_abs = torch.abs(support_layer[:, 0:1] - support_layer[:, 1:2])
  layer_delta_norm = torch.where(
    both_layer_valid,
    layer_delta_abs.to(dtype=torch.float32).clamp(0.0, float(max_layer))
    / max_layer_tensor,
    torch.zeros_like(layer_delta_abs, dtype=torch.float32),
  )
  adjacent_layers = both_support & both_layer_valid & (layer_delta_abs == 1)
  pair_min_fraction = torch.amin(fraction, dim=-1, keepdim=True)
  pair_mean_fraction = torch.mean(fraction, dim=-1, keepdim=True)
  partial_footprint = torch.any(
    stair_support.bool() & (fraction > 0.0) & (fraction < 1.0 - 1.0e-6),
    dim=-1,
    keepdim=True,
  )
  full_pair = both_support & torch.all(fraction >= 1.0 - 1.0e-6, dim=-1, keepdim=True)

  return torch.cat(
    [
      ground,
      support,
      fraction,
      layer_norm,
      duration_norm,
      both_ground.float(),
      both_support.float(),
      layer_delta_norm,
      adjacent_layers.float(),
      pair_min_fraction,
      pair_mean_fraction,
      partial_footprint.float(),
      full_pair.float(),
    ],
    dim=-1,
  )


def privileged_footprint_features_from_env(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return current simulation-only footprint features from stair env extras."""
  ground_contact = _tensor_extra(
    env, STAIR_CURRENT_GROUND_CONTACT_KEY, (2,), torch.bool, False
  )
  stair_support = _tensor_extra(
    env, STAIR_CURRENT_STAIR_SUPPORT_KEY, (2,), torch.bool, False
  )
  support_fraction = _tensor_extra(
    env, STAIR_CURRENT_SUPPORT_FRACTION_KEY, (2,), torch.float32, 0.0
  )
  support_layer = _tensor_extra(
    env, STAIR_CURRENT_SUPPORT_LAYER_KEY, (2,), torch.long, -1
  )
  contact_duration = _tensor_extra(
    env, STAIR_CURRENT_CONTACT_DURATION_KEY, (2,), torch.float32, 0.0
  )
  return privileged_footprint_features_from_tensors(
    ground_contact=ground_contact,
    stair_support=stair_support,
    support_fraction=support_fraction,
    support_layer=support_layer,
    contact_duration=contact_duration,
  )


def label_batch_from_env(
  env: ManagerBasedRlEnv,
  state: StairProbeLevelState,
  reset_mask: torch.Tensor | None = None,
) -> StairProbeLabelBatch:
  """Build one label batch from training-only privileged env extras."""
  phase = _tensor_extra(env, STAIR_PHASE_KEY, (), torch.long, 0)
  sequence_id = _tensor_extra(env, STAIR_SEQUENCE_ID_KEY, (), torch.long, -1)
  support_layer = _tensor_extra(
    env, STAIR_CURRENT_SUPPORT_LAYER_KEY, (2,), torch.long, -1
  )
  stair_support = _tensor_extra(
    env, STAIR_CURRENT_STAIR_SUPPORT_KEY, (2,), torch.bool, False
  )
  support_fraction = _tensor_extra(
    env, STAIR_CURRENT_SUPPORT_FRACTION_KEY, (2,), torch.float32, 0.0
  )
  contact_duration = _tensor_extra(
    env, STAIR_CURRENT_CONTACT_DURATION_KEY, (2,), torch.float32, 0.0
  )
  true_depth = _tensor_extra(env, STAIR_TREAD_DEPTH_LABEL_KEY, (), torch.float32, 0.0)
  true_height = _tensor_extra(env, STAIR_RISER_HEIGHT_LABEL_KEY, (), torch.float32, 0.0)
  safe_landing_center = _tensor_extra(
    env, SAFE_LANDING_CENTER_KEY, (), torch.float32, 0.0
  )
  minimum_safe_stride = _tensor_extra(
    env, MINIMUM_SAFE_STRIDE_KEY, (), torch.float32, 0.0
  )
  maximum_safe_stride = _tensor_extra(
    env, MINIMUM_SAFE_STRIDE_UPPER_KEY, (), torch.float32, 0.0
  )
  safe_stride_valid = _tensor_extra(
    env, MINIMUM_SAFE_STRIDE_VALID_KEY, (), torch.bool, False
  )
  landing_touchdown = _tensor_extra(env, LANDING_TOUCHDOWN_KEY, (), torch.bool, False)
  landing_quality = _tensor_extra(env, LANDING_QUALITY_KEY, (), torch.float32, 0.0)
  collision_risk = _tensor_extra(env, COLLISION_RISK_KEY, (), torch.float32, 0.0)
  shape_valid = _tensor_extra(env, STAIR_SHAPE_LABEL_VALID_KEY, (), torch.bool, False)
  return state.update(
    stair_active=phase > 0,
    sequence_id=sequence_id,
    support_layer=support_layer,
    stair_support=stair_support,
    support_fraction=support_fraction,
    contact_duration=contact_duration,
    true_tread_depth=true_depth,
    true_riser_height=true_height,
    safe_landing_center=safe_landing_center,
    minimum_safe_stride=minimum_safe_stride,
    maximum_safe_stride=maximum_safe_stride,
    safe_stride_valid=safe_stride_valid,
    landing_touchdown=landing_touchdown,
    landing_quality=landing_quality,
    collision_risk=collision_risk,
    shape_valid=shape_valid,
    reset_mask=reset_mask,
  )


def write_sample_metadata(output_dir: Path, arrays: dict[str, np.ndarray]) -> None:
  """Write per-sample CSV metadata for quick shell inspection."""
  metadata_path = output_dir / "sample_metadata.csv"
  fieldnames = (
    "sample_index",
    "sample_type",
    "level_delta",
    "relative_level",
    "num_confirmed_layers",
    "depth_bin",
    "depth_3bin",
    "depth_valid",
    "safe_stride_valid",
    "landing_touchdown",
    "landing_quality",
    "collision_risk",
    "stair_active",
    "sequence_id",
    "env_id",
    "frame_idx",
    "seed",
  )
  with metadata_path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for index in range(arrays["sample_type"].shape[0]):
      level_delta = int(arrays["level_delta_label"][index])
      writer.writerow(
        {
          "sample_index": index,
          "sample_type": str(arrays["sample_type_name"][index]),
          "level_delta": LEVEL_DELTA_NAMES[level_delta],
          "relative_level": int(arrays["relative_level_label"][index]),
          "num_confirmed_layers": int(arrays["num_confirmed_layers"][index]),
          "depth_bin": int(arrays["depth_bin_label"][index]),
          "depth_3bin": int(arrays["depth_3bin_label"][index]),
          "depth_valid": int(bool(arrays["depth_valid_label"][index])),
          "safe_stride_valid": int(bool(arrays["safe_stride_valid_label"][index])),
          "landing_touchdown": int(bool(arrays["landing_touchdown_label"][index])),
          "landing_quality": float(arrays["landing_quality_label"][index]),
          "collision_risk": float(arrays["collision_risk_label"][index]),
          "stair_active": int(bool(arrays["stair_active_label"][index])),
          "sequence_id": int(arrays["sequence_id"][index]),
          "env_id": int(arrays["env_id"][index]),
          "frame_idx": int(arrays["frame_idx"][index]),
          "seed": int(arrays["seed"][index]),
        }
      )


def build_label_audit(
  arrays: dict[str, np.ndarray],
  *,
  obs_dim: int,
  history_len: int,
) -> list[tuple[str, str]]:
  """Return label and leakage audit metrics as ``(metric, value)`` rows."""
  rows: list[tuple[str, str]] = []
  num_samples = int(arrays["sample_type"].shape[0])
  sequence_ids = arrays["sequence_id"]
  stair_active = arrays["stair_active_label"].astype(bool)
  flat = ~stair_active
  rows.append(("num_samples", str(num_samples)))
  rows.append(("num_sequences", str(len(set(sequence_ids[sequence_ids >= 0])))))
  rows.append(
    (
      "num_stair_sequences",
      str(len(set(sequence_ids[(sequence_ids >= 0) & stair_active]))),
    )
  )
  rows.append(("num_flat_samples", str(int(flat.sum()))))
  for label_id, name in enumerate(LEVEL_DELTA_NAMES):
    count = int((arrays["level_delta_label"] == label_id).sum())
    rows.append((f"level_delta_{name}_count", str(count)))
  relative = arrays["relative_level_label"]
  rows.append(("relative_level_min", str(int(relative.min()))))
  rows.append(("relative_level_max", str(int(relative.max()))))
  rows.append(("relative_level_mean", f"{float(relative.mean()):.6g}"))
  depth_valid = arrays["depth_valid_label"].astype(bool)
  for bin_id in range(8):
    count = int(((arrays["depth_bin_label"] == bin_id) & depth_valid).sum())
    rows.append((f"depth_bin_{bin_id}_count", str(count)))
  for bin_id in range(3):
    count = int(((arrays["depth_3bin_label"] == bin_id) & depth_valid).sum())
    rows.append((f"depth_3bin_{bin_id}_count", str(count)))
  for sample_type, sample_id in SAMPLE_TYPE_TO_ID.items():
    count = int((arrays["sample_type"] == sample_id).sum())
    rows.append((f"sample_type_{sample_type}_count", str(count)))
  transition = arrays["level_delta_label"] != LEVEL_DELTA_NO_TRANSITION
  rows.append(("transition_sample_count", str(int(transition.sum()))))
  flat_false = int((flat & transition).sum())
  rows.append(("flat_false_transition_labels", str(flat_false)))
  rows.append(
    (
      "safe_stride_valid_count",
      str(int(arrays["safe_stride_valid_label"].astype(bool).sum())),
    )
  )
  rows.append(
    (
      "landing_touchdown_count",
      str(int(arrays["landing_touchdown_label"].astype(bool).sum())),
    )
  )
  rows.append(
    (
      "landing_quality_mean",
      f"{float(arrays['landing_quality_label'].mean()):.6g}",
    )
  )
  rows.append(
    (
      "collision_risk_mean",
      f"{float(arrays['collision_risk_label'].mean()):.6g}",
    )
  )
  transition_sequence_ids = sequence_ids[transition & (sequence_ids >= 0)]
  if transition_sequence_ids.size:
    unique, counts = np.unique(transition_sequence_ids, return_counts=True)
    del unique
    rows.append(("transition_samples_per_sequence_mean", f"{float(counts.mean()):.6g}"))
    rows.append(("transition_samples_per_sequence_max", str(int(counts.max()))))
  else:
    rows.append(("transition_samples_per_sequence_mean", "0"))
    rows.append(("transition_samples_per_sequence_max", "0"))
  max_per_sequence = []
  for sequence_id in sorted(set(sequence_ids[sequence_ids >= 0])):
    mask = sequence_ids == sequence_id
    max_per_sequence.append(int(relative[mask].max()))
  if max_per_sequence:
    rows.append(
      (
        "max_relative_level_per_sequence_mean",
        f"{float(np.mean(max_per_sequence)):.6g}",
      )
    )
    rows.append(
      ("max_relative_level_per_sequence_max", str(int(np.max(max_per_sequence))))
    )
    end_levels = []
    for sequence_id in sorted(set(sequence_ids[sequence_ids >= 0])):
      indices = np.nonzero(sequence_ids == sequence_id)[0]
      end_levels.append(int(relative[indices[-1]]))
    for level in sorted(set(end_levels)):
      rows.append(
        (
          f"sequence_end_relative_level_{level}_count",
          str(sum(value == level for value in end_levels)),
        )
      )
  else:
    rows.append(("max_relative_level_per_sequence_mean", "0"))
    rows.append(("max_relative_level_per_sequence_max", "0"))
  rows.append(("obs_dim", str(obs_dim)))
  rows.append(("history_len", str(history_len)))
  rows.append(
    ("forbidden_input_fields_present", str(len(forbidden_input_feature_names())))
  )
  if "privileged_footprint_history" in arrays:
    footprint = arrays["privileged_footprint_history"]
    valid = arrays.get("privileged_footprint_valid_mask")
    rows.append(("privileged_footprint_history_present", "1"))
    rows.append(("privileged_footprint_dim", str(int(footprint.shape[-1]))))
    rows.append(("privileged_footprint_history_len", str(int(footprint.shape[1]))))
    rows.append(
      (
        "privileged_footprint_nonzero_fraction",
        f"{float(np.count_nonzero(footprint) / max(footprint.size, 1)):.6g}",
      )
    )
    if valid is not None:
      rows.append(
        (
          "privileged_footprint_valid_fraction",
          f"{float(valid.astype(bool).mean()):.6g}",
        )
      )
  else:
    rows.append(("privileged_footprint_history_present", "0"))
  if "sparse_foot_event_memory" in arrays:
    memory = arrays["sparse_foot_event_memory"]
    valid = arrays.get("sparse_foot_event_valid_mask")
    rows.append(("sparse_foot_event_memory_present", "1"))
    rows.append(("sparse_foot_event_dim", str(int(memory.shape[-1]))))
    rows.append(("sparse_foot_event_memory_len", str(int(memory.shape[1]))))
    rows.append(
      (
        "sparse_foot_event_nonzero_fraction",
        f"{float(np.count_nonzero(memory) / max(memory.size, 1)):.6g}",
      )
    )
    if valid is not None:
      rows.append(
        (
          "sparse_foot_event_valid_fraction",
          f"{float(valid.astype(bool).mean()):.6g}",
        )
      )
  else:
    rows.append(("sparse_foot_event_memory_present", "0"))
  return rows


def write_label_audit(
  output_dir: Path,
  arrays: dict[str, np.ndarray],
  *,
  obs_dim: int,
  history_len: int,
) -> None:
  """Write ``label_audit.csv``."""
  rows = build_label_audit(arrays, obs_dim=obs_dim, history_len=history_len)
  with (output_dir / "label_audit.csv").open(
    "w", encoding="utf-8", newline=""
  ) as stream:
    writer = csv.writer(stream)
    writer.writerow(("metric", "value"))
    writer.writerows(rows)


def write_dataset_config(
  output_dir: Path,
  *,
  task_id: str,
  checkpoint_path: Path,
  cfg: ExportStairProbeDatasetConfig,
  obs_dim: int,
) -> None:
  """Write a JSON config and no-leakage input schema."""
  payload: dict[str, Any] = {
    "task_id": task_id,
    "checkpoint_path": str(checkpoint_path),
    "config": asdict(cfg),
    "obs_dim": obs_dim,
    "history_len": cfg.history_len,
    "latent_obs_source": "observations['latent'] / stair_latent_obs",
    "include_gait_phase": False,
    "include_privileged_footprint": cfg.include_privileged_footprint,
    "include_sparse_foot_events": cfg.include_sparse_foot_events,
    "input_feature_groups": [
      {"name": name, "width": width} for name, width in INPUT_FEATURE_GROUPS
    ],
    "privileged_footprint_source": (
      "simulation-only foot contact/support extras from stair reward"
      if cfg.include_privileged_footprint
      else None
    ),
    "privileged_footprint_history_len": (
      cfg.privileged_footprint_history_len if cfg.include_privileged_footprint else 0
    ),
    "privileged_footprint_feature_groups": [
      {"name": name, "width": width}
      for name, width in PRIVILEGED_FOOTPRINT_FEATURE_GROUPS
    ],
    "privileged_footprint_is_deployable": False,
    "sparse_foot_event_source": (
      "causal foot touchdown/toe-riser events from current and previous frames"
      if cfg.include_sparse_foot_events
      else None
    ),
    "sparse_foot_event_memory_len": (
      cfg.sparse_event_memory_len if cfg.include_sparse_foot_events else 0
    ),
    "sparse_foot_event_feature_groups": [
      {"name": name, "width": width} for name, width in SPARSE_FOOT_EVENT_FEATURE_GROUPS
    ],
    "sparse_foot_event_is_deployable_style": True,
    "sparse_foot_event_causality": "current and past events only; no future events",
    "forbidden_input_fields": list(FORBIDDEN_INPUT_FIELDS),
    "forbidden_input_fields_present": list(forbidden_input_feature_names()),
    "level_delta_labels": {
      str(index): name for index, name in enumerate(LEVEL_DELTA_NAMES)
    },
    "sample_type_labels": {
      str(index): name for name, index in SAMPLE_TYPE_TO_ID.items()
    },
  }
  with (output_dir / "dataset_config.json").open("w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")


def _close_sequence_logger(env: ManagerBasedRlEnv) -> None:
  metrics_cfg = env.metrics_manager.cfg
  if not isinstance(metrics_cfg, dict):
    return
  term_cfg = metrics_cfg.get("stair_sequence_event_logger")
  if term_cfg is None or term_cfg.func is not stair_sequence_event_logger:
    return
  close = getattr(term_cfg.func, "close", None)
  if callable(close):
    close()


def _latent_obs(obs: Any) -> torch.Tensor:
  latent = obs.get("latent")
  if not isinstance(latent, torch.Tensor):
    raise RuntimeError("Expected rollout observations to contain a tensor 'latent'.")
  return latent


def run_export(task_id: str, cfg: ExportStairProbeDatasetConfig) -> Path:
  """Run a frozen policy and export Stage 2A probe samples."""
  if cfg.num_envs <= 0:
    raise ValueError("num_envs must be positive.")
  if cfg.steps <= 0:
    raise ValueError("steps must be positive.")
  if cfg.history_len <= 0:
    raise ValueError("history_len must be positive.")
  if cfg.include_privileged_footprint and cfg.privileged_footprint_history_len <= 0:
    raise ValueError("privileged_footprint_history_len must be positive.")
  if cfg.include_sparse_foot_events and cfg.sparse_event_memory_len <= 0:
    raise ValueError("sparse_event_memory_len must be positive.")
  if cfg.expected_obs_dim != input_obs_dim():
    raise ValueError(
      f"expected_obs_dim={cfg.expected_obs_dim} does not match the Stage 2A "
      f"v1 schema dimension {input_obs_dim()}."
    )

  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  output_dir = Path(cfg.output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

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
  print(
    "[Stage 2A] Export start:",
    f"task={task_id}",
    f"seed={cfg.seed}",
    f"envs={cfg.num_envs}",
    f"steps={cfg.steps}",
    f"history_len={cfg.history_len}",
    f"privileged_footprint={cfg.include_privileged_footprint}",
    f"sparse_foot_events={cfg.include_sparse_foot_events}",
    f"device={device}",
    f"output={output_dir}",
  )

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  wrapped = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  builder = StairProbeDatasetBuilder(
    num_envs=cfg.num_envs,
    history_len=cfg.history_len,
    obs_dim=cfg.expected_obs_dim,
    max_samples=cfg.max_samples,
    seed=cfg.seed,
    device=device,
    privileged_footprint_dim=(
      privileged_footprint_obs_dim() if cfg.include_privileged_footprint else None
    ),
    privileged_footprint_history_len=(
      cfg.privileged_footprint_history_len if cfg.include_privileged_footprint else None
    ),
    sparse_event_dim=(
      sparse_foot_event_obs_dim() if cfg.include_sparse_foot_events else None
    ),
    sparse_event_memory_len=(
      cfg.sparse_event_memory_len if cfg.include_sparse_foot_events else None
    ),
    flat_sample_period=cfg.flat_sample_period,
    same_level_sample_period=cfg.same_level_sample_period,
    landing_sample_period=cfg.landing_sample_period,
    partial_sample_period=cfg.partial_sample_period,
  )
  label_state = StairProbeLevelState(
    num_envs=cfg.num_envs,
    stable_support_fraction=cfg.stable_support_fraction,
    full_support_fraction=cfg.full_support_fraction,
    partial_support_fraction=cfg.partial_support_fraction,
    stable_contact_time=cfg.stable_contact_time,
    device=device,
  )
  sparse_event_memory = (
    SparseFootEventMemory(
      num_envs=cfg.num_envs,
      memory_len=cfg.sparse_event_memory_len,
      device=device,
      age_norm_s=cfg.sparse_event_age_norm_s,
      position_scale_m=cfg.sparse_event_position_scale_m,
      velocity_scale_mps=cfg.sparse_event_velocity_scale_mps,
      gait_period=cfg.sparse_event_gait_period,
    )
    if cfg.include_sparse_foot_events
    else None
  )
  try:
    policy, _runner = load_inference_policy(
      env=wrapped,
      task_id=task_id,
      agent_cfg=agent_cfg,
      checkpoint_path=checkpoint_path,
      device=device,
    )
    obs = wrapped.get_observations()
    latent = _latent_obs(obs)
    if latent.shape[-1] != cfg.expected_obs_dim:
      raise RuntimeError(
        f"Expected latent obs dim {cfg.expected_obs_dim}, got {latent.shape[-1]}."
      )
    reset_all = torch.ones(cfg.num_envs, dtype=torch.bool, device=device)
    builder.push_observations(latent, reset_all)
    if cfg.include_privileged_footprint:
      zero_footprint = torch.zeros(
        cfg.num_envs,
        privileged_footprint_obs_dim(),
        dtype=torch.float32,
        device=device,
      )
      builder.push_privileged_footprint(zero_footprint, reset_all)
    if sparse_event_memory is not None:
      sparse_event_memory.reset()
      builder.set_sparse_foot_event_memory(
        sparse_event_memory.memory,
        sparse_event_memory.valid_mask,
      )

    progress = tqdm(
      range(cfg.steps),
      desc=f"stair probe export seed={cfg.seed}",
      disable=not cfg.progress,
      dynamic_ncols=True,
      unit="step",
    )
    for rollout_step in progress:
      with torch.no_grad():
        actions = policy(obs)
      step_result = wrapped.step(actions)
      reset_policy_state_from_step(policy, step_result)
      obs, _rewards, dones, _extras = step_result
      reset_mask = dones.to(dtype=torch.bool)
      latent = _latent_obs(obs)
      builder.push_observations(latent, reset_mask)
      if cfg.include_privileged_footprint:
        footprint = privileged_footprint_features_from_env(raw_env)
        builder.push_privileged_footprint(footprint, reset_mask)
      if sparse_event_memory is not None:
        sparse_event_memory.begin_step(reset_mask, float(raw_env.step_dt))
        sparse_event_memory.update_from_env_and_latent(raw_env, latent)
        builder.set_sparse_foot_event_memory(
          sparse_event_memory.memory,
          sparse_event_memory.valid_mask,
        )
      labels = label_batch_from_env(raw_env, label_state, reset_mask)
      builder.collect(labels, rollout_step + 1)
      if builder.is_full:
        break
  finally:
    _close_sequence_logger(raw_env)
    wrapped.close()

  arrays = builder.as_arrays()
  np.savez_compressed(output_dir / "samples.npz", **cast(Any, arrays))
  write_sample_metadata(output_dir, arrays)
  write_label_audit(
    output_dir,
    arrays,
    obs_dim=cfg.expected_obs_dim,
    history_len=cfg.history_len,
  )
  write_dataset_config(
    output_dir,
    task_id=task_id,
    checkpoint_path=checkpoint_path,
    cfg=cfg,
    obs_dim=cfg.expected_obs_dim,
  )
  print(
    "[Stage 2A] Export complete:",
    f"samples={arrays['sample_type'].shape[0]}",
    f"obs_history_shape={arrays['obs_history'].shape}",
    f"directory={output_dir}",
  )
  return output_dir


def main() -> None:
  import mjlab.tasks as _tasks  # noqa: F401

  task_id, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  cfg = tyro.cli(
    ExportStairProbeDatasetConfig,
    args=remaining_args,
    default=ExportStairProbeDatasetConfig(),
    prog=sys.argv[0] + f" {task_id}",
    config=mjlab.TYRO_FLAGS,
  )
  run_export(task_id, cfg)


if __name__ == "__main__":
  main()
