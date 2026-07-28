"""Export Stage 2D foot-event detector data from a frozen policy rollout."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import tyro
from scripts.velocity_eval.export_stair_probe_dataset import (
  INPUT_FEATURE_GROUPS,
  STAIR_CURRENT_GROUND_CONTACT_KEY,
  STAIR_CURRENT_STAIR_SUPPORT_KEY,
  STAIR_CURRENT_SUPPORT_FRACTION_KEY,
  STAIR_PHASE_KEY,
  TOE_RISER_NEW_HIT_BY_FOOT_KEY,
  TOE_RISER_NEW_HIT_KEY,
  StairProbeHistoryBuffer,
  _close_sequence_logger,
  _tensor_extra,
  input_feature_slices,
  input_obs_dim,
  stage2a_base_latent_obs,
)
from scripts.velocity_eval.policy_io import (
  get_clip_actions,
  load_inference_policy,
  resolve_checkpoint_path,
  resolve_inference_agent_cfg,
)
from tqdm.auto import tqdm

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.mdp.observations import (
  _body_frame_foot_positions,
  _get_leg_joint_info,
  phase,
)
from mjlab.utils.lstm import reset_policy_state_from_step
from mjlab.utils.torch import configure_torch_backends

DEFAULT_STAGE2D_CHECKPOINT = (
  "logs/rsl_rl/g1_blind_rough_target_navigation_slow_latent_teacherkl/"
  "Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1/"
  "base111/model_51000.pt"
)

FOOT_EVENT_LABEL_NAMES: tuple[str, ...] = (
  "left_contact",
  "right_contact",
  "left_touchdown",
  "right_touchdown",
  "left_toe_riser_hit",
  "right_toe_riser_hit",
)

FootEventDetectorObsSchema = Literal["v1", "footprint_v2"]

FOOTPRINT_V2_EXTRA_FEATURE_GROUPS: tuple[tuple[str, int], ...] = (
  ("left_heel_vel_body", 3),
  ("right_heel_vel_body", 3),
  ("left_heel_vel_delta", 3),
  ("right_heel_vel_delta", 3),
  ("left_sole_center_pos_body", 3),
  ("right_sole_center_pos_body", 3),
  ("left_sole_center_vel_body", 3),
  ("right_sole_center_vel_body", 3),
  ("left_sole_pitch_proxy", 1),
  ("right_sole_pitch_proxy", 1),
  ("command_lin_y", 1),
  ("command_yaw_rate", 1),
  ("action_delta_leg", 12),
)
"""Additional deployable kinematic inputs for the footprint/touchdown detector."""


@dataclass(frozen=True)
class ExportFootEventDetectorDatasetConfig:
  """Configuration for frozen-policy foot-event detector dataset export."""

  checkpoint_file: str | None = DEFAULT_STAGE2D_CHECKPOINT
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  output_dir: str = "eval_outputs/stair_stage2/model51000_seed42_foot_event_detector_v1"
  num_envs: int = 512
  steps: int = 5000
  seed: int = 42
  device: str | None = None
  history_len: int = 16
  max_samples: int | None = None
  input_schema: FootEventDetectorObsSchema = "v1"
  include_gait_phase: bool = True
  gait_period: float = 0.6
  command_name: str = "twist"
  expected_obs_dim: int | None = None
  progress: bool = True


def _per_foot_bool_from_tensor(value: torch.Tensor, *, key: str) -> torch.Tensor:
  """Return a bool ``(num_envs, 2)`` tensor from env-level or per-foot extras."""
  value = value.bool()
  if value.ndim == 1:
    return value[:, None].expand(-1, 2)
  if value.ndim == 2 and value.shape[1] == 1:
    return value.expand(-1, 2)
  if value.ndim == 2 and value.shape[1] == 2:
    return value
  raise ValueError(
    f"Expected env-level or per-foot bool extra for {key!r}, got {tuple(value.shape)}."
  )


def _per_foot_bool_extra(
  env: ManagerBasedRlEnv,
  key: str,
  *,
  default: bool = False,
) -> torch.Tensor:
  """Return a bool ``(num_envs, 2)`` extra, expanding env-level flags if needed."""
  value = _tensor_extra(env, key, (), torch.bool, default)
  return _per_foot_bool_from_tensor(value, key=key)


def _toe_riser_hit_by_foot(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return per-foot toe-riser hit labels, preferring the new per-foot extra."""
  value = env.extras.get(TOE_RISER_NEW_HIT_BY_FOOT_KEY)
  if isinstance(value, torch.Tensor):
    return _per_foot_bool_from_tensor(value, key=TOE_RISER_NEW_HIT_BY_FOOT_KEY)
  return _per_foot_bool_extra(env, TOE_RISER_NEW_HIT_KEY, default=False)


def foot_event_labels_from_env(
  env: ManagerBasedRlEnv,
  *,
  previous_contact: torch.Tensor,
  previous_contact_valid: torch.Tensor,
) -> torch.Tensor:
  """Build deployable detector labels from simulation-only oracle contact extras."""
  contact = _tensor_extra(
    env,
    STAIR_CURRENT_GROUND_CONTACT_KEY,
    (2,),
    torch.bool,
    False,
  ).bool()
  touchdown = contact & ~previous_contact & previous_contact_valid[:, None]
  toe_hit = _toe_riser_hit_by_foot(env)
  return torch.cat(
    (
      contact.float(),
      touchdown.float(),
      toe_hit.float(),
    ),
    dim=-1,
  )


def _validate_input_schema(input_schema: str) -> FootEventDetectorObsSchema:
  if input_schema not in ("v1", "footprint_v2"):
    raise ValueError(
      f"input_schema must be 'v1' or 'footprint_v2', got {input_schema!r}."
    )
  return cast(FootEventDetectorObsSchema, input_schema)


def foot_event_detector_obs_dim(
  *,
  include_gait_phase: bool,
  input_schema: FootEventDetectorObsSchema = "v1",
) -> int:
  """Return deployable event-detector observation width."""
  schema = _validate_input_schema(input_schema)
  dim = input_obs_dim() + (2 if include_gait_phase else 0)
  if schema == "footprint_v2":
    dim += sum(width for _name, width in FOOTPRINT_V2_EXTRA_FEATURE_GROUPS)
  return dim


def resolve_foot_event_detector_obs_dim(
  expected_obs_dim: int | None,
  *,
  include_gait_phase: bool,
  input_schema: FootEventDetectorObsSchema,
) -> int:
  """Return the configured detector obs width, validating overrides."""
  obs_dim = foot_event_detector_obs_dim(
    include_gait_phase=include_gait_phase,
    input_schema=input_schema,
  )
  if expected_obs_dim is not None and int(expected_obs_dim) != obs_dim:
    raise ValueError(
      f"expected_obs_dim={expected_obs_dim} does not match detector obs "
      f"dimension {obs_dim} for input_schema={input_schema!r}."
    )
  return obs_dim


def _command_column(
  env: ManagerBasedRlEnv,
  *,
  command_name: str,
  column: int,
  dtype: torch.dtype,
) -> torch.Tensor:
  zeros = torch.zeros(env.num_envs, 1, dtype=dtype, device=env.device)
  command_manager = getattr(env, "command_manager", None)
  if command_manager is None:
    return zeros
  command = command_manager.get_command(command_name)
  if not isinstance(command, torch.Tensor) or command.shape[-1] <= column:
    return zeros
  return command[:, column : column + 1].to(device=env.device, dtype=dtype)


def _velocity_and_delta_from_cache(
  env: ManagerBasedRlEnv,
  *,
  position: torch.Tensor,
  position_key: str,
  velocity_key: str,
  reset_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
  zeros = torch.zeros_like(position)
  reset = (
    torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if reset_mask is None
    else reset_mask.to(device=env.device, dtype=torch.bool)
  ).reshape(env.num_envs)
  if position_key not in env.extras:
    env.extras[position_key] = position.detach().clone()
    env.extras[velocity_key] = zeros.detach().clone()
    return zeros, zeros

  prev_position = env.extras[position_key].to(device=env.device, dtype=position.dtype)
  velocity = (position - prev_position) / max(float(env.step_dt), 1.0e-6)
  velocity = torch.where(reset[:, None], zeros, velocity)

  prev_velocity_obj = env.extras.get(velocity_key)
  if isinstance(prev_velocity_obj, torch.Tensor):
    prev_velocity = prev_velocity_obj.to(device=env.device, dtype=position.dtype)
    velocity_delta = torch.where(reset[:, None], zeros, velocity - prev_velocity)
  else:
    velocity_delta = zeros

  env.extras[position_key] = position.detach().clone()
  env.extras[velocity_key] = velocity.detach().clone()
  return velocity, velocity_delta


def _footprint_v2_extra_obs(
  latent: torch.Tensor,
  env: ManagerBasedRlEnv,
  *,
  command_name: str,
  reset_mask: torch.Tensor | None,
) -> torch.Tensor:
  left_toe, right_toe, left_heel, right_heel = _body_frame_foot_positions(env)
  left_center = 0.5 * (left_toe + left_heel)
  right_center = 0.5 * (right_toe + right_heel)
  left_heel_vel, left_heel_vel_delta = _velocity_and_delta_from_cache(
    env,
    position=left_heel,
    position_key="foot_event_v2_prev_left_heel_pos_body",
    velocity_key="foot_event_v2_prev_left_heel_vel_body",
    reset_mask=reset_mask,
  )
  right_heel_vel, right_heel_vel_delta = _velocity_and_delta_from_cache(
    env,
    position=right_heel,
    position_key="foot_event_v2_prev_right_heel_pos_body",
    velocity_key="foot_event_v2_prev_right_heel_vel_body",
    reset_mask=reset_mask,
  )
  left_center_vel, _left_center_vel_delta = _velocity_and_delta_from_cache(
    env,
    position=left_center,
    position_key="foot_event_v2_prev_left_sole_center_pos_body",
    velocity_key="foot_event_v2_prev_left_sole_center_vel_body",
    reset_mask=reset_mask,
  )
  right_center_vel, _right_center_vel_delta = _velocity_and_delta_from_cache(
    env,
    position=right_center,
    position_key="foot_event_v2_prev_right_sole_center_pos_body",
    velocity_key="foot_event_v2_prev_right_sole_center_vel_body",
    reset_mask=reset_mask,
  )

  left_sole = left_toe - left_heel
  right_sole = right_toe - right_heel
  left_sole_pitch = left_sole[:, 2:3] / left_sole.norm(dim=-1, keepdim=True).clamp_min(
    1.0e-6
  )
  right_sole_pitch = right_sole[:, 2:3] / right_sole.norm(
    dim=-1, keepdim=True
  ).clamp_min(1.0e-6)

  command_lin_y = _command_column(
    env,
    command_name=command_name,
    column=1,
    dtype=latent.dtype,
  )
  command_yaw_rate = _command_column(
    env,
    command_name=command_name,
    column=2,
    dtype=latent.dtype,
  )

  slices = input_feature_slices()
  previous_action_leg = latent[:, slices["previous_action_leg"]]
  current_action = env.action_manager.action
  _leg_joint_indices, leg_action_indices, _default_joint_pos = _get_leg_joint_info(env)
  current_action_leg = current_action[:, leg_action_indices].to(dtype=latent.dtype)
  action_delta_leg = current_action_leg - previous_action_leg

  return torch.cat(
    (
      left_heel_vel,
      right_heel_vel,
      left_heel_vel_delta,
      right_heel_vel_delta,
      left_center,
      right_center,
      left_center_vel,
      right_center_vel,
      left_sole_pitch,
      right_sole_pitch,
      command_lin_y,
      command_yaw_rate,
      action_delta_leg,
    ),
    dim=-1,
  ).to(dtype=latent.dtype)


def foot_event_detector_obs(
  obs: Any,
  env: ManagerBasedRlEnv,
  *,
  input_schema: FootEventDetectorObsSchema = "v1",
  include_gait_phase: bool,
  gait_period: float,
  command_name: str,
  reset_mask: torch.Tensor | None = None,
) -> torch.Tensor:
  """Return deployable detector input features for the current frame."""
  schema = _validate_input_schema(input_schema)
  latent = stage2a_base_latent_obs(obs)
  parts = [latent]
  if include_gait_phase:
    parts.append(phase(env, gait_period, command_name))
  if schema == "footprint_v2":
    parts.append(
      _footprint_v2_extra_obs(
        latent,
        env,
        command_name=command_name,
        reset_mask=reset_mask,
      )
    )
  return torch.cat(parts, dim=-1)


def foot_event_input_feature_groups(
  *,
  include_gait_phase: bool,
  input_schema: FootEventDetectorObsSchema = "v1",
) -> list[dict[str, int | str]]:
  """Return detector input schema metadata."""
  schema = _validate_input_schema(input_schema)
  groups = [{"name": name, "width": width} for name, width in INPUT_FEATURE_GROUPS]
  if include_gait_phase:
    groups.append({"name": "gait_phase_sin", "width": 1})
    groups.append({"name": "gait_phase_cos", "width": 1})
  if schema == "footprint_v2":
    groups.extend(
      {"name": name, "width": width}
      for name, width in FOOTPRINT_V2_EXTRA_FEATURE_GROUPS
    )
  return groups


class FootEventDetectorDatasetBuilder:
  """Collect fixed-length deployable observation histories and event labels."""

  def __init__(
    self,
    *,
    num_envs: int,
    history_len: int,
    obs_dim: int,
    max_samples: int | None,
    device: torch.device | str,
  ) -> None:
    if history_len <= 0:
      raise ValueError("history_len must be positive.")
    if max_samples is not None and max_samples <= 0:
      raise ValueError("max_samples must be positive.")
    self.num_envs = int(num_envs)
    self.history = StairProbeHistoryBuffer(
      num_envs=num_envs,
      history_len=history_len,
      obs_dim=obs_dim,
      device=device,
    )
    self.max_samples = None if max_samples is None else int(max_samples)
    self._arrays: dict[str, list[np.ndarray]] = {
      "obs_history": [],
      "obs_valid_mask": [],
      "event_label": [],
      "episode_id": [],
      "env_id": [],
      "frame_idx": [],
      "seed": [],
    }

  @property
  def num_samples(self) -> int:
    if not self._arrays["event_label"]:
      return 0
    return int(sum(chunk.shape[0] for chunk in self._arrays["event_label"]))

  @property
  def is_full(self) -> bool:
    return self.max_samples is not None and self.num_samples >= self.max_samples

  def push_observations(self, obs: torch.Tensor, reset_mask: torch.Tensor) -> None:
    self.history.push(obs, reset_mask)

  def collect(
    self,
    *,
    labels: torch.Tensor,
    collect_mask: torch.Tensor,
    episode_id: torch.Tensor,
    frame_idx: int,
    seed: int,
    metadata: dict[str, torch.Tensor] | None = None,
  ) -> None:
    ids = collect_mask.nonzero(as_tuple=False).squeeze(-1)
    if ids.numel() == 0 or self.is_full:
      return
    if self.max_samples is not None:
      remaining = self.max_samples - self.num_samples
      if ids.numel() > remaining:
        ids = ids[:remaining]
    self._arrays["obs_history"].append(
      self.history.history[ids].detach().cpu().numpy().astype(np.float32)
    )
    self._arrays["obs_valid_mask"].append(
      self.history.valid_mask[ids].detach().cpu().numpy().astype(np.bool_)
    )
    self._arrays["event_label"].append(
      labels[ids].detach().cpu().numpy().astype(np.float32)
    )
    self._arrays["episode_id"].append(
      episode_id[ids].detach().cpu().numpy().astype(np.int64)
    )
    self._arrays["env_id"].append(ids.detach().cpu().numpy().astype(np.int64))
    self._arrays["frame_idx"].append(np.full((ids.numel(),), frame_idx, dtype=np.int64))
    self._arrays["seed"].append(np.full((ids.numel(),), seed, dtype=np.int64))
    if metadata is not None:
      for key, value in metadata.items():
        if key not in self._arrays:
          self._arrays[key] = []
        self._arrays[key].append(value[ids].detach().cpu().numpy())

  def as_arrays(self) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key, chunks in self._arrays.items():
      if chunks:
        arrays[key] = np.concatenate(chunks, axis=0)
      elif key == "obs_history":
        arrays[key] = np.zeros(
          (0, self.history.history.shape[1], self.history.history.shape[2]),
          dtype=np.float32,
        )
      elif key == "obs_valid_mask":
        arrays[key] = np.zeros((0, self.history.history.shape[1]), dtype=np.bool_)
      elif key == "event_label":
        arrays[key] = np.zeros((0, len(FOOT_EVENT_LABEL_NAMES)), dtype=np.float32)
      else:
        arrays[key] = np.zeros((0,), dtype=np.int64)
    return arrays


def build_label_audit(arrays: dict[str, np.ndarray]) -> list[tuple[str, str]]:
  """Summarize detector labels and exported histories."""
  labels = arrays["event_label"]
  rows: list[tuple[str, str]] = [
    ("num_samples", str(int(labels.shape[0]))),
    ("num_episodes", str(int(np.unique(arrays["episode_id"]).shape[0]))),
    ("obs_history_len", str(int(arrays["obs_history"].shape[1]))),
    ("obs_dim", str(int(arrays["obs_history"].shape[2]))),
    ("label_dim", str(int(labels.shape[1]))),
  ]
  for index, name in enumerate(FOOT_EVENT_LABEL_NAMES):
    count = float(labels[:, index].sum()) if labels.size else 0.0
    rate = count / max(float(labels.shape[0]), 1.0)
    rows.append((f"{name}_positive_count", f"{count:.0f}"))
    rows.append((f"{name}_positive_rate", f"{rate:.6g}"))
  stair_support = arrays.get("stair_support")
  if stair_support is not None and labels.size:
    touchdown = labels[:, 2:4] > 0.5
    stair_touchdown = touchdown & stair_support.astype(np.bool_)
    rows.append(
      (
        "touchdown_on_stair_count",
        f"{float(stair_touchdown.sum()):.0f}",
      )
    )
    rows.append(
      (
        "touchdown_on_stair_rate_of_touchdowns",
        f"{float(stair_touchdown.sum()) / max(float(touchdown.sum()), 1.0):.6g}",
      )
    )
  return rows


def write_dataset_outputs(
  output_dir: Path,
  *,
  task_id: str,
  checkpoint_path: Path,
  cfg: ExportFootEventDetectorDatasetConfig,
  arrays: dict[str, np.ndarray],
) -> None:
  """Write dataset arrays, audit, and schema metadata."""
  np.savez_compressed(output_dir / "samples.npz", **cast(Any, arrays))
  with (output_dir / "label_audit.csv").open(
    "w", encoding="utf-8", newline=""
  ) as stream:
    writer = csv.writer(stream)
    writer.writerow(("metric", "value"))
    writer.writerows(build_label_audit(arrays))

  payload: dict[str, Any] = {
    "task_id": task_id,
    "checkpoint_path": str(checkpoint_path),
    "policy_frozen": True,
    "config": asdict(cfg),
    "input_source": (
      "observations['latent'] / stair_latent_obs"
      + (" + gait_phase" if cfg.include_gait_phase else "")
      + (
        " + footprint_v2 deployable kinematics"
        if cfg.input_schema == "footprint_v2"
        else ""
      )
    ),
    "input_feature_groups": foot_event_input_feature_groups(
      include_gait_phase=cfg.include_gait_phase,
      input_schema=cfg.input_schema,
    ),
    "label_names": list(FOOT_EVENT_LABEL_NAMES),
    "label_source": {
      "contact": STAIR_CURRENT_GROUND_CONTACT_KEY,
      "touchdown": "rising edge of current ground contact",
      "toe_riser_hit": (
        f"{TOE_RISER_NEW_HIT_BY_FOOT_KEY}, falling back to {TOE_RISER_NEW_HIT_KEY}"
      ),
    },
    "deploy_contract": (
      "Detector predicts event/contact probabilities only; deployment code "
      "maintains foot-event memory and computes FK-based footprint geometry."
    ),
  }
  with (output_dir / "dataset_config.json").open("w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")


def run_export(task_id: str, cfg: ExportFootEventDetectorDatasetConfig) -> Path:
  """Roll out a frozen policy and export supervised detector samples."""
  if cfg.num_envs <= 0:
    raise ValueError("num_envs must be positive.")
  if cfg.steps <= 0:
    raise ValueError("steps must be positive.")
  if cfg.history_len <= 0:
    raise ValueError("history_len must be positive.")
  obs_dim = resolve_foot_event_detector_obs_dim(
    cfg.expected_obs_dim,
    include_gait_phase=cfg.include_gait_phase,
    input_schema=cfg.input_schema,
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
  agent_cfg = resolve_inference_agent_cfg(
    checkpoint_path=checkpoint_path,
    agent_cfg=agent_cfg,
  )
  print(
    "[Stage 2D] Export detector dataset:",
    f"task={task_id}",
    f"checkpoint={checkpoint_path}",
    f"envs={cfg.num_envs}",
    f"steps={cfg.steps}",
    f"history_len={cfg.history_len}",
    f"input_schema={cfg.input_schema}",
    f"include_gait_phase={cfg.include_gait_phase}",
    f"max_samples={cfg.max_samples}",
    f"device={device}",
    f"output={output_dir}",
  )

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  wrapped = RslRlVecEnvWrapper(raw_env, clip_actions=get_clip_actions(agent_cfg))
  builder = FootEventDetectorDatasetBuilder(
    num_envs=cfg.num_envs,
    history_len=cfg.history_len,
    obs_dim=obs_dim,
    max_samples=cfg.max_samples,
    device=device,
  )
  previous_contact = torch.zeros(cfg.num_envs, 2, dtype=torch.bool, device=device)
  previous_contact_valid = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
  episode_counter = torch.zeros(cfg.num_envs, dtype=torch.int64, device=device)
  env_ids = torch.arange(cfg.num_envs, dtype=torch.int64, device=device)
  completed_steps = 0

  try:
    policy, _runner = load_inference_policy(
      env=wrapped,
      task_id=task_id,
      agent_cfg=agent_cfg,
      checkpoint_path=checkpoint_path,
      device=device,
    )
    obs = wrapped.get_observations()
    reset_all = torch.ones(cfg.num_envs, dtype=torch.bool, device=device)
    latent = foot_event_detector_obs(
      obs,
      raw_env,
      input_schema=cfg.input_schema,
      include_gait_phase=cfg.include_gait_phase,
      gait_period=cfg.gait_period,
      command_name=cfg.command_name,
      reset_mask=reset_all,
    )
    builder.push_observations(latent, reset_all)

    progress = tqdm(
      range(cfg.steps),
      desc=f"foot event export seed={cfg.seed}",
      disable=not cfg.progress,
      dynamic_ncols=True,
      unit="step",
    )
    for rollout_step in progress:
      completed_steps = rollout_step + 1
      with torch.no_grad():
        actions = policy(obs)
      step_result = wrapped.step(actions)
      reset_policy_state_from_step(policy, step_result)
      obs, _rewards, dones, _extras = step_result
      reset_mask = dones.to(dtype=torch.bool)
      latent = foot_event_detector_obs(
        obs,
        raw_env,
        input_schema=cfg.input_schema,
        include_gait_phase=cfg.include_gait_phase,
        gait_period=cfg.gait_period,
        command_name=cfg.command_name,
        reset_mask=reset_mask,
      )
      builder.push_observations(latent, reset_mask)

      labels = foot_event_labels_from_env(
        raw_env,
        previous_contact=previous_contact,
        previous_contact_valid=previous_contact_valid,
      )
      full_history = builder.history.valid_mask.all(dim=1)
      collect_mask = full_history & ~reset_mask
      episode_id = episode_counter * cfg.num_envs + env_ids
      builder.collect(
        labels=labels,
        collect_mask=collect_mask,
        episode_id=episode_id,
        frame_idx=rollout_step + 1,
        seed=cfg.seed,
        metadata={
          "stair_support": _tensor_extra(
            raw_env,
            STAIR_CURRENT_STAIR_SUPPORT_KEY,
            (2,),
            torch.bool,
            False,
          ).bool(),
          "support_fraction": _tensor_extra(
            raw_env,
            STAIR_CURRENT_SUPPORT_FRACTION_KEY,
            (2,),
            torch.float32,
            0.0,
          ).float(),
          "stair_phase": _tensor_extra(
            raw_env,
            STAIR_PHASE_KEY,
            (),
            torch.long,
            0,
          ).long(),
        },
      )

      current_contact = _tensor_extra(
        raw_env,
        STAIR_CURRENT_GROUND_CONTACT_KEY,
        (2,),
        torch.bool,
        False,
      ).bool()
      previous_contact.copy_(
        torch.where(
          reset_mask[:, None],
          torch.zeros_like(current_contact),
          current_contact,
        )
      )
      previous_contact_valid.copy_(
        torch.where(
          reset_mask,
          torch.zeros_like(previous_contact_valid),
          torch.ones_like(previous_contact_valid),
        )
      )
      episode_counter += reset_mask.to(dtype=torch.int64)
      if builder.is_full:
        break
  finally:
    _close_sequence_logger(raw_env)
    wrapped.close()

  arrays = builder.as_arrays()
  stopped_reason = "max_samples" if builder.is_full else "completed_steps"
  write_dataset_outputs(
    output_dir,
    task_id=task_id,
    checkpoint_path=checkpoint_path,
    cfg=cfg,
    arrays=arrays,
  )
  print(
    "[Stage 2D] Export complete:",
    f"samples={arrays['event_label'].shape[0]}",
    f"completed_steps={completed_steps}",
    f"stopped_reason={stopped_reason}",
    f"obs_history_shape={arrays['obs_history'].shape}",
    f"output={output_dir}",
  )
  return output_dir


def main() -> None:
  import mjlab.tasks as _tasks  # noqa: F401

  task_id, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    args=None,
    default="Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1",
    return_unknown_args=True,
  )
  cfg = tyro.cli(ExportFootEventDetectorDatasetConfig, args=remaining_args)
  run_export(task_id, cfg)


if __name__ == "__main__":
  main()
