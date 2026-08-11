"""Collect policy latents on fixed velocity evaluation terrains."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import tyro
from scripts.velocity_eval.eval_terrains import apply_eval_overrides, get_terrain_set
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
from mjlab.tasks.velocity.mdp.observations import (
  FOOT_EVENT_RATCHET_START,
  FOOT_EVENT_SUMMARY_DIM,
)
from mjlab.utils.lstm import reset_policy_state_from_step
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class CollectLatentsConfig:
  checkpoint_file: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  terrain_set: str = "cluster_v1"
  episodes_per_terrain: int = 20
  num_envs: int = 20
  steps_per_episode: int = 500
  sample_every: int = 1
  command_vx: float = 0.4
  command_vy: float = 0.0
  command_wz: float = 0.0
  seed: int = 23456
  device: str | None = None
  output_root: str = "eval_outputs/velocity"
  output_dir: str | None = None
  output_file: str | None = None
  clean_observations: bool = True
  disable_observation_delay: bool = True
  disable_actuator_delay: bool = True
  collect_slow_latent_diagnostics: bool = True
  stop_after_stair_completion: bool = False
  stair_completion_margin_m: float = 0.0
  stop_on_episode_done: bool = False


def _resolve_output_path(
  *,
  cfg: CollectLatentsConfig,
  task_id: str,
  agent_cfg,
  checkpoint_path: Path,
) -> Path:
  if cfg.output_file is not None:
    output_path = Path(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

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

  return output_dir / f"latents_{cfg.terrain_set}.npz"


def _mlp_hidden_before_final_linear(policy, obs) -> torch.Tensor:
  latent = policy.get_latent(obs)
  layers = list(policy.mlp)
  linear_indices = [
    idx for idx, layer in enumerate(layers) if isinstance(layer, nn.Linear)
  ]
  if not linear_indices:
    return latent
  final_linear_idx = linear_indices[-1]
  x = latent
  for layer in layers[:final_linear_idx]:
    x = layer(x)
  return x


def extract_policy_latent(policy, obs) -> torch.Tensor:
  """Extract a per-env policy latent without mutating recurrent state."""
  if bool(getattr(policy, "is_recurrent", False)):
    hidden = policy.get_hidden_state()
    if isinstance(hidden, tuple):
      hidden = hidden[0]
    if hidden is None:
      raise RuntimeError("Recurrent policy did not expose a hidden state.")
    return hidden[-1].detach()
  return _mlp_hidden_before_final_linear(policy, obs).detach()


def _batch_tensor(
  value: Any,
  *,
  batch_size: int,
  component: int | None = None,
) -> torch.Tensor | None:
  if not torch.is_tensor(value):
    return None
  tensor = value.detach()
  if tensor.dim() >= 3 and tensor.shape[0] == 1 and tensor.shape[1] == batch_size:
    tensor = tensor.squeeze(0)
  if component is not None:
    if tensor.shape[-1] <= component:
      return None
    tensor = tensor[..., component : component + 1]
  if tensor.dim() == 1:
    tensor = tensor.unsqueeze(-1)
  if tensor.shape[0] != batch_size:
    return None
  return tensor.reshape(batch_size, -1)


def _policy_state_tensors(policy, *, batch_size: int) -> dict[str, torch.Tensor]:
  hidden = getattr(policy, "get_hidden_state", lambda: None)()
  if hidden is None:
    return {}
  states = list(hidden) if isinstance(hidden, (tuple, list)) else [hidden]
  output: dict[str, torch.Tensor] = {}
  if states:
    h = states[0].detach()
    if h.dim() == 3:
      h = h[-1]
    h = _batch_tensor(h, batch_size=batch_size)
    if h is not None:
      output["h_t"] = h
      output["hidden"] = h
  if len(states) >= 2:
    c = states[1].detach()
    if c.dim() == 3:
      c = c[-1]
    c = _batch_tensor(c, batch_size=batch_size)
    if c is not None:
      output["c_t"] = c
  if len(states) >= 3:
    z = _batch_tensor(states[2], batch_size=batch_size)
    if z is not None:
      output["z_memory"] = z
      state_dim = getattr(policy, "state_latent_dim", None)
      if state_dim is not None and 0 < int(state_dim) < z.shape[-1]:
        split = int(state_dim)
        output["z_state_memory"] = z[:, :split]
        output["z_shape_memory"] = z[:, split:]
  if len(states) >= 8:
    gate_parts = [_batch_tensor(state, batch_size=batch_size) for state in states[3:8]]
    if all(part is not None for part in gate_parts):
      output["gate_state"] = torch.cat(
        [part for part in gate_parts if part is not None],
        dim=-1,
      )
  elif len(states) >= 4:
    gate = _batch_tensor(states[3], batch_size=batch_size)
    if gate is not None and gate.shape[-1] == 5:
      output["gate_state"] = gate
  return output


def _policy_internal_tensors(
  policy,
  obs: Any,
  state_tensors: dict[str, torch.Tensor],
  *,
  batch_size: int,
) -> dict[str, torch.Tensor]:
  output: dict[str, torch.Tensor] = {}
  latent_obs = _obs_latent_tensor(obs, batch_size=batch_size)
  normalizer = getattr(policy, "latent_obs_normalizer", None)
  encoder = getattr(policy, "latent_encoder", None)
  if latent_obs is not None and callable(normalizer) and callable(encoder):
    try:
      output["encoder_out"] = encoder(normalizer(latent_obs)).detach()
    except (RuntimeError, TypeError, ValueError):
      pass

  h_t = state_tensors.get("h_t")
  if h_t is None:
    h_t = state_tensors.get("hidden")
  z_candidate_head = getattr(policy, "z_candidate_head", None)
  if h_t is not None and callable(z_candidate_head):
    try:
      output["z_candidate"] = z_candidate_head(h_t).detach()
    except (RuntimeError, TypeError, ValueError):
      pass
  return output


def _obs_latent_tensor(obs: Any, *, batch_size: int) -> torch.Tensor | None:
  keys = getattr(obs, "keys", None)
  if not callable(keys) or "latent" not in obs.keys():
    return None
  latent_obs = obs["latent"]
  if not torch.is_tensor(latent_obs):
    return None
  return _batch_tensor(latent_obs, batch_size=batch_size)


def _foot_event_tensors(obs: Any, *, batch_size: int) -> dict[str, torch.Tensor]:
  latent_obs = _obs_latent_tensor(obs, batch_size=batch_size)
  if latent_obs is None or latent_obs.shape[-1] < FOOT_EVENT_SUMMARY_DIM:
    return {}
  summary = latent_obs[:, -FOOT_EVENT_SUMMARY_DIM:]
  return {
    "foot_event_summary": summary,
    "ratchet": summary[:, FOOT_EVENT_RATCHET_START:],
  }


def _sigmoid_batch(value: Any, *, batch_size: int) -> torch.Tensor | None:
  tensor = _batch_tensor(value, batch_size=batch_size)
  return None if tensor is None else torch.sigmoid(tensor)


def _semantic_tensor(
  policy,
  obs: Any,
  diagnostics: dict[str, torch.Tensor],
  aux_outputs: dict[str, torch.Tensor],
  state_tensors: dict[str, torch.Tensor],
  *,
  batch_size: int,
) -> torch.Tensor | None:
  existing = diagnostics.get("shadow_semantic")
  if existing is not None:
    return existing
  build_semantic = getattr(policy, "_build_shadow_semantic", None)
  gate_state = state_tensors.get("gate_state")
  if not callable(build_semantic) or gate_state is None:
    return None

  event_prob = diagnostics.get("event_prob")
  if event_prob is None:
    event_prob = _sigmoid_batch(aux_outputs.get("event_logit"), batch_size=batch_size)
  stair_prob = diagnostics.get("stair_prob")
  if stair_prob is None:
    stair_prob = _sigmoid_batch(aux_outputs.get("stair_logit"), batch_size=batch_size)
  stair_shape = diagnostics.get("stair_shape")
  if stair_shape is None:
    stair_shape = _batch_tensor(aux_outputs.get("stair_shape"), batch_size=batch_size)
  safe_stride = diagnostics.get("safe_stride")
  if safe_stride is None:
    safe_stride = _batch_tensor(aux_outputs.get("safe_stride"), batch_size=batch_size)
  interval = diagnostics.get("safe_stride_interval")
  if interval is None:
    interval = _batch_tensor(
      aux_outputs.get("safe_stride_interval"),
      batch_size=batch_size,
    )
  confidence = diagnostics.get("safe_stride_confidence")
  if confidence is None:
    confidence = _sigmoid_batch(
      aux_outputs.get("safe_stride_confidence_logit"),
      batch_size=batch_size,
    )
  if interval is None and safe_stride is not None:
    interval = torch.cat([safe_stride, safe_stride], dim=-1)
  if (
    event_prob is None
    or stair_prob is None
    or stair_shape is None
    or safe_stride is None
    or interval is None
  ):
    return None
  if confidence is None:
    confidence = torch.zeros_like(safe_stride)

  trend_logits = _batch_tensor(
    aux_outputs.get("safe_stride_trend_logit"),
    batch_size=batch_size,
  )
  latent_obs = _obs_latent_tensor(obs, batch_size=batch_size)
  try:
    return build_semantic(
      event_prob,
      stair_prob,
      gate_state,
      stair_shape,
      safe_stride,
      interval,
      confidence,
      latent_obs,
      trend_logits,
    ).detach()
  except (RuntimeError, TypeError, ValueError):
    return None


_DIAGNOSTIC_KEYS = (
  "event_prob",
  "stair_prob",
  "future_risk",
  "future_quality",
  "stair_shape",
  "safe_stride",
  "safe_stride_interval",
  "safe_stride_confidence",
  "safe_stride_trend",
  "safe_stride_trend_prob",
  "z_norm",
  "gate_mode",
  "gate_memory_age",
  "gate_release",
  "episode_write_ever",
  "episode_memory_ever",
  "gate_event_trigger",
  "gate_write_confirm",
  "gate_write_abort",
  "gate_memory_exit",
  "gate_memory_event_shape_boost",
  "alpha",
  "alpha_state",
  "alpha_shape",
  "shadow_semantic",
)


def _policy_diagnostic_tensors(
  policy,
  obs: Any,
  state_tensors: dict[str, torch.Tensor],
  *,
  batch_size: int,
) -> dict[str, torch.Tensor]:
  diagnostics_fn = getattr(policy, "get_slow_latent_diagnostics", None)
  diagnostics_raw = diagnostics_fn() if callable(diagnostics_fn) else {}
  if not isinstance(diagnostics_raw, dict):
    diagnostics_raw = {}

  diagnostics: dict[str, torch.Tensor] = {}
  for key in _DIAGNOSTIC_KEYS:
    value = _batch_tensor(diagnostics_raw.get(key), batch_size=batch_size)
    if value is not None:
      diagnostics[key] = value

  aux_outputs_fn = getattr(policy, "get_aux_outputs", None)
  aux_outputs = aux_outputs_fn() if callable(aux_outputs_fn) else {}
  if not isinstance(aux_outputs, dict):
    aux_outputs = {}

  semantic = _semantic_tensor(
    policy,
    obs,
    diagnostics,
    aux_outputs,
    state_tensors,
    batch_size=batch_size,
  )
  if semantic is not None:
    diagnostics["semantic"] = semantic
  return diagnostics


def _append_tensor(
  arrays: dict[str, list],
  key: str,
  value: torch.Tensor | None,
  mask: torch.Tensor | None = None,
) -> None:
  if value is None:
    return
  tensor = value.detach()
  if mask is not None:
    tensor = tensor[mask]
  if tensor.shape[0] == 0:
    return
  arrays.setdefault(key, []).append(tensor.cpu().numpy())


def _completion_stop_root_x_m(terrain, cfg: CollectLatentsConfig) -> float | None:
  if not cfg.stop_after_stair_completion:
    return None
  completion_x = terrain.stair_completion_root_x_m()
  if completion_x is None:
    return None
  return float(completion_x) + float(cfg.stair_completion_margin_m)


def _get_gait_period(env) -> float:
  try:
    term = env.cfg.observations["actor"].terms["gait_phase"]
    return float(term.params.get("period", 0.6))
  except (AttributeError, KeyError, TypeError):
    return 0.6


def _compute_gait_phase(env) -> tuple[torch.Tensor, torch.Tensor]:
  period = _get_gait_period(env)
  phase = (env.episode_length_buf.float() * env.step_dt) % period / period
  phase_sincos = torch.stack(
    [torch.sin(phase * torch.pi * 2.0), torch.cos(phase * torch.pi * 2.0)],
    dim=-1,
  )
  return phase, phase_sincos


def _collect_for_terrain(
  *,
  task_id: str,
  agent_cfg,
  checkpoint_path: Path,
  terrain,
  cfg: CollectLatentsConfig,
  terrain_index: int,
  batch_index: int,
  batch_size: int,
  episode_offset: int,
  device: str,
) -> dict[str, list]:
  env_cfg = load_env_cfg(task_id, play=False)
  apply_eval_overrides(
    env_cfg,
    terrain,
    num_envs=batch_size,
    seed=cfg.seed + 1009 * terrain_index + 9173 * batch_index,
    max_episode_length_s=max(1.0, cfg.steps_per_episode * 0.02),
    command=(cfg.command_vx, cfg.command_vy, cfg.command_wz),
    clean_observations=cfg.clean_observations,
    disable_observation_delay=cfg.disable_observation_delay,
    disable_actuator_delay=cfg.disable_actuator_delay,
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=get_clip_actions(agent_cfg))

  try:
    policy, _runner = load_inference_policy(
      env=wrapped,
      task_id=task_id,
      agent_cfg=agent_cfg,
      checkpoint_path=checkpoint_path,
      device=device,
    )
    obs = wrapped.get_observations()
    arrays: dict[str, list] = {
      "latent": [],
      "action": [],
      "command": [],
      "base_state": [],
      "gait_phase": [],
      "gait_phase_sincos": [],
      "terrain_label": [],
      "terrain_name": [],
      "terrain_kind": [],
      "terrain_height_m": [],
      "terrain_step_width_m": [],
      "terrain_tread_depth_m": [],
      "stair_completion_root_x_m": [],
      "episode_id": [],
      "time_step": [],
      "time_s": [],
    }

    episode_offsets = torch.arange(batch_size, device=device) + episode_offset
    stop_root_x_m = _completion_stop_root_x_m(terrain, cfg)
    active = torch.ones(batch_size, device=device, dtype=torch.bool)
    use_active_mask = stop_root_x_m is not None or cfg.stop_on_episode_done
    if stop_root_x_m is not None:
      print(
        "[INFO] "
        f"{terrain.name}: stopping samples at root x >= {stop_root_x_m:.3f} m "
        "relative to env origin."
      )

    for step in range(cfg.steps_per_episode):
      if use_active_mask and not bool(torch.any(active)):
        print(f"[INFO] {terrain.name}: all episodes inactive by step {step}.")
        break

      with torch.no_grad():
        actions = policy(obs)
        latent = extract_policy_latent(policy, obs)
        state_tensors = _policy_state_tensors(policy, batch_size=batch_size)
        internal_tensors = _policy_internal_tensors(
          policy,
          obs,
          state_tensors,
          batch_size=batch_size,
        )
        foot_event_tensors = _foot_event_tensors(obs, batch_size=batch_size)
        diagnostic_tensors = (
          _policy_diagnostic_tensors(
            policy,
            obs,
            state_tensors,
            batch_size=batch_size,
          )
          if cfg.collect_slow_latent_diagnostics
          else {}
        )

      if step % cfg.sample_every == 0:
        sample_mask = active if use_active_mask else None
        sample_count = (
          int(torch.count_nonzero(sample_mask).item())
          if sample_mask is not None
          else batch_size
        )
        if sample_count == 0:
          continue
        robot = wrapped.unwrapped.scene["robot"]
        command = wrapped.unwrapped.command_manager.get_command("twist")
        assert command is not None
        gait_phase, gait_phase_sincos = _compute_gait_phase(wrapped.unwrapped)
        base_state = torch.cat(
          [
            robot.data.root_link_pos_w,
            robot.data.root_link_quat_w,
            robot.data.root_link_lin_vel_b,
            robot.data.root_link_ang_vel_b,
          ],
          dim=-1,
        )
        _append_tensor(arrays, "latent", latent, sample_mask)
        for key, value in state_tensors.items():
          _append_tensor(arrays, key, value, sample_mask)
        for key, value in internal_tensors.items():
          _append_tensor(arrays, key, value, sample_mask)
        for key, value in foot_event_tensors.items():
          _append_tensor(arrays, key, value, sample_mask)
        for key, value in diagnostic_tensors.items():
          _append_tensor(arrays, key, value, sample_mask)
        _append_tensor(arrays, "action", actions, sample_mask)
        _append_tensor(arrays, "command", command, sample_mask)
        _append_tensor(arrays, "base_state", base_state, sample_mask)
        _append_tensor(arrays, "gait_phase", gait_phase, sample_mask)
        _append_tensor(arrays, "gait_phase_sincos", gait_phase_sincos, sample_mask)
        arrays["terrain_label"].extend([terrain.label] * sample_count)
        arrays["terrain_name"].extend([terrain.name] * sample_count)
        arrays["terrain_kind"].extend([terrain.kind] * sample_count)
        arrays["terrain_height_m"].extend(
          [float(terrain.height_m or 0.0)] * sample_count
        )
        arrays["terrain_step_width_m"].extend(
          [float(terrain.step_width)] * sample_count
        )
        arrays["terrain_tread_depth_m"].extend(
          [
            float(
              terrain.step_width if terrain.kind in ("upstairs", "downstairs") else 0.0
            )
          ]
          * sample_count
        )
        arrays["stair_completion_root_x_m"].extend(
          [float(stop_root_x_m or 0.0)] * sample_count
        )
        episode_ids = episode_offsets
        if sample_mask is not None:
          episode_ids = episode_ids[sample_mask]
        arrays["episode_id"].append(episode_ids.detach().cpu().numpy())
        time_steps = torch.full((batch_size,), step, device=device)
        if sample_mask is not None:
          time_steps = time_steps[sample_mask]
        arrays["time_step"].append(time_steps.detach().cpu().numpy())
        time_values = torch.full(
          (batch_size,),
          step * wrapped.unwrapped.step_dt,
          device=device,
        )
        if sample_mask is not None:
          time_values = time_values[sample_mask]
        arrays["time_s"].append(time_values.detach().cpu().numpy())

      step_result = wrapped.step(actions)
      reset_policy_state_from_step(policy, step_result)
      obs, _rewards, dones, _extras = step_result
      if use_active_mask:
        if cfg.stop_on_episode_done:
          active &= ~dones.to(device=device, dtype=torch.bool)
        if stop_root_x_m is not None:
          robot = wrapped.unwrapped.scene["robot"]
          root_x_rel = (
            robot.data.root_link_pos_w[:, 0] - wrapped.unwrapped.scene.env_origins[:, 0]
          )
          active &= root_x_rel < stop_root_x_m

    return arrays
  finally:
    wrapped.close()


def run_collect_latents(
  task_id: str, cfg: CollectLatentsConfig
) -> dict[str, np.ndarray]:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  terrains = get_terrain_set(cfg.terrain_set)
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
  output_path = _resolve_output_path(
    cfg=cfg,
    task_id=task_id,
    agent_cfg=agent_cfg,
    checkpoint_path=checkpoint_path,
  )
  print(f"[INFO] Output directory: {output_path.parent}")

  chunks = []
  batch_index = 0
  for terrain_index, terrain in enumerate(terrains):
    remaining = cfg.episodes_per_terrain
    episode_offset = terrain_index * cfg.episodes_per_terrain
    print(f"[INFO] Collecting latents on {terrain.name}")
    while remaining > 0:
      batch_size = min(max(1, cfg.num_envs), remaining)
      chunks.append(
        _collect_for_terrain(
          task_id=task_id,
          agent_cfg=agent_cfg,
          checkpoint_path=checkpoint_path,
          terrain=terrain,
          cfg=cfg,
          terrain_index=terrain_index,
          batch_index=batch_index,
          batch_size=batch_size,
          episode_offset=episode_offset,
          device=device,
        )
      )
      remaining -= batch_size
      episode_offset += batch_size
      batch_index += 1

  output: dict[str, np.ndarray] = {}
  metadata_keys = {
    "terrain_label",
    "terrain_name",
    "terrain_kind",
    "terrain_height_m",
    "terrain_step_width_m",
    "terrain_tread_depth_m",
    "stair_completion_root_x_m",
  }
  keys = sorted({key for chunk in chunks for key in chunk.keys()})
  for key in keys:
    if key in metadata_keys:
      values = []
      for chunk in chunks:
        values.extend(chunk.get(key, []))
      output[key] = np.asarray(values)
      continue

    arrays = [
      np.concatenate(chunk[key], axis=0)
      for chunk in chunks
      if key in chunk and chunk[key]
    ]
    if arrays:
      output[key] = np.concatenate(arrays, axis=0)

  np.savez_compressed(output_path, **output)  # type: ignore[invalid-argument-type]
  print(f"[INFO] Wrote latent dataset to {output_path}")
  return output


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
    CollectLatentsConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_collect_latents(chosen_task, cfg)


if __name__ == "__main__":
  main()
