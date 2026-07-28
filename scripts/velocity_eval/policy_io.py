"""Policy loading helpers for offline evaluation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime
from inspect import signature
from pathlib import Path
from typing import Any

import yaml
from rsl_rl.algorithms.ppo import PPO

from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.lstm import reset_policy_state
from mjlab.utils.os import get_task_log_root, get_wandb_checkpoint_path

_PPO_CONSTRUCT_ALGORITHM_KEYS = frozenset(
  {
    "class_name",
    "share_cnn_encoders",
  }
)
_PPO_INIT_ALGORITHM_KEYS = frozenset(
  name
  for name in signature(PPO.__init__).parameters
  if name not in {"self", "actor", "critic", "storage", "device", "multi_gpu_cfg"}
)
_PPO_INFERENCE_ALGORITHM_KEYS = _PPO_CONSTRUCT_ALGORITHM_KEYS | _PPO_INIT_ALGORITHM_KEYS


def _callable_leaf_name(value: Any) -> str:
  """Return the final class/function name for config values accepted by RSL-RL."""
  if callable(value):
    return getattr(value, "__name__", str(value))
  text = str(value)
  return text.rsplit(":", 1)[-1].rsplit(".", 1)[-1]


def _is_teacher_kl_algorithm(algorithm_class_name: Any) -> bool:
  return _callable_leaf_name(algorithm_class_name) == "PPOTeacherKL"


def _slugify(value: str) -> str:
  value = value.strip().lower()
  value = re.sub(r"^mjlab[-_]*", "", value)
  value = re.sub(r"[-_]*unitree[-_]*g1$", "", value)
  value = re.sub(r"[^a-z0-9]+", "_", value)
  value = re.sub(r"_+", "_", value).strip("_")
  return value or "policy"


def get_policy_output_name(
  *,
  task_id: str,
  agent_cfg: Any,
  checkpoint_path: Path | None = None,
) -> str:
  """Return a concise folder name for outputs from one trained policy family."""
  experiment_name = (
    agent_cfg.get("experiment_name")
    if isinstance(agent_cfg, Mapping)
    else getattr(agent_cfg, "experiment_name", None)
  )
  if experiment_name:
    return _slugify(str(experiment_name))

  if checkpoint_path is not None:
    parts = checkpoint_path.parts
    if "rsl_rl" in parts:
      idx = parts.index("rsl_rl")
      if len(parts) > idx + 1:
        return _slugify(parts[idx + 1])

  return _slugify(task_id)


def get_agent_cfg_value(agent_cfg: Any, key: str, default: Any = None) -> Any:
  """Read a runner config value from a saved dict or dataclass config."""
  if isinstance(agent_cfg, Mapping):
    return agent_cfg.get(key, default)
  return getattr(agent_cfg, key, default)


def get_clip_actions(agent_cfg: Any) -> float | None:
  """Read runner clip-actions from either a dataclass or saved YAML dict."""
  value = get_agent_cfg_value(agent_cfg, "clip_actions")
  return None if value is None else float(value)


def make_timestamped_policy_output_dir(
  *,
  output_root: str | Path,
  task_id: str,
  agent_cfg: Any,
  checkpoint_path: Path | None = None,
) -> Path:
  """Create ``output_root / policy_name / timestamp`` with collision suffixes."""
  policy_name = get_policy_output_name(
    task_id=task_id,
    agent_cfg=agent_cfg,
    checkpoint_path=checkpoint_path,
  )
  root = Path(output_root) / policy_name
  timestamp = datetime.now().strftime("%m%d_%H%M%S")
  output_dir = root / timestamp
  suffix = 1
  while output_dir.exists():
    output_dir = root / f"{timestamp}_{suffix:02d}"
    suffix += 1
  output_dir.mkdir(parents=True, exist_ok=False)
  return output_dir


def make_inference_train_cfg(agent_cfg: Any) -> dict[str, Any]:
  """Convert a runner config into an inference-only train config.

  Teacher-KL checkpoints should not be required for offline actor inference, so
  PPOTeacherKL configs are locally converted to PPO while preserving the actor
  and critic model definitions needed to build the network.
  """
  cfg = asdict(agent_cfg) if is_dataclass(agent_cfg) else deepcopy(agent_cfg)
  cfg["upload_model"] = False

  algorithm_cfg = cfg.get("algorithm", {})
  if not isinstance(algorithm_cfg, Mapping):
    raise TypeError("Expected runner config 'algorithm' to be a mapping.")
  if _is_teacher_kl_algorithm(algorithm_cfg.get("class_name", "")):
    algorithm_cfg = {
      key: value
      for key, value in dict(algorithm_cfg).items()
      if key in _PPO_INFERENCE_ALGORITHM_KEYS
    }
    algorithm_cfg["class_name"] = "PPO"
    cfg["algorithm"] = algorithm_cfg
    cfg.pop("teacher", None)
    obs_groups_cfg = cfg.get("obs_groups", {})
    if not isinstance(obs_groups_cfg, Mapping):
      raise TypeError("Expected runner config 'obs_groups' to be a mapping.")
    cfg["obs_groups"] = {
      key: value
      for key, value in obs_groups_cfg.items()
      if key in ("actor", "critic", "latent")
    }

  return cfg


def load_checkpoint_agent_cfg(checkpoint_path: Path) -> dict[str, Any] | None:
  """Load the saved runner config next to a checkpoint when available.

  Older slow-latent checkpoints may have a different actor shape than the current
  task config. The saved ``params/agent.yaml`` keeps the architecture that was
  used for that run, so offline eval should prefer it when it exists.
  """
  params_path = checkpoint_path.parent / "params" / "agent.yaml"
  if not params_path.exists():
    return None
  payload = yaml.unsafe_load(params_path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise TypeError(f"Expected a dict in saved agent config: {params_path}")
  return payload


def resolve_inference_agent_cfg(
  *,
  checkpoint_path: Path,
  agent_cfg: Any,
  verbose: bool = True,
) -> Any:
  """Return the runner config that best matches an inference checkpoint."""
  saved_agent_cfg = load_checkpoint_agent_cfg(checkpoint_path)
  if saved_agent_cfg is None:
    return agent_cfg
  if verbose:
    print(
      "[INFO] Loaded saved agent config from "
      f"{checkpoint_path.parent / 'params' / 'agent.yaml'}"
    )
  return saved_agent_cfg


def resolve_checkpoint_path(
  *,
  task_id: str,
  agent_cfg: Any,
  checkpoint_file: str | None,
  wandb_run_path: str | None,
  wandb_checkpoint_name: str | None,
) -> Path:
  """Resolve either a local checkpoint or a W&B checkpoint."""
  if checkpoint_file is not None:
    path = Path(checkpoint_file).expanduser()
    if not path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {path}")
    return path

  if wandb_run_path is None:
    raise ValueError("Provide either --checkpoint-file or --wandb-run-path.")

  experiment_name = get_agent_cfg_value(agent_cfg, "experiment_name")
  if experiment_name is None:
    raise ValueError("agent_cfg must define experiment_name for W&B checkpoints.")
  log_root_path = get_task_log_root(str(experiment_name), task_id).resolve()
  path, _ = get_wandb_checkpoint_path(
    log_root_path, Path(wandb_run_path), wandb_checkpoint_name
  )
  return path


def load_inference_policy(
  *,
  env: RslRlVecEnvWrapper,
  task_id: str,
  agent_cfg: Any,
  checkpoint_path: Path,
  device: str,
  runner_cls: type | None = None,
):
  """Build a runner, load actor weights, and return the inference policy."""
  runner_cls = runner_cls or load_runner_cls(task_id) or MjlabOnPolicyRunner
  agent_cfg = resolve_inference_agent_cfg(
    checkpoint_path=checkpoint_path,
    agent_cfg=agent_cfg,
    verbose=False,
  )
  train_cfg = make_inference_train_cfg(agent_cfg)
  runner = runner_cls(env, train_cfg, device=device)
  runner.load(
    str(checkpoint_path),
    load_cfg={
      "actor": True,
      "critic": False,
      "optimizer": False,
      "iteration": True,
      "rnd": False,
    },
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)
  policy.eval()
  reset_policy_state(policy)
  return policy, runner
