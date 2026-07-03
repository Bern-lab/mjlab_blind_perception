"""Run a frozen velocity policy and export stair sequences/events to CSV."""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from scripts.velocity_eval.policy_io import (
  load_inference_policy,
  resolve_checkpoint_path,
)

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.lstm import reset_policy_state_from_step
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class ExportStairSequencesConfig:
  """Configuration for reward-neutral Stage 0 stair event collection."""

  checkpoint_file: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  output_dir: str = "eval_outputs/stair_stage0"
  num_envs: int = 512
  steps: int = 5000
  seed: int = 42
  device: str | None = None
  flush_rows: int = 128


def _count_csv_rows(path: Path) -> int:
  if not path.exists():
    return 0
  with path.open(encoding="utf-8", newline="") as stream:
    return sum(1 for _row in csv.DictReader(stream))


def _close_sequence_logger(env: ManagerBasedRlEnv) -> None:
  metrics_cfg = env.metrics_manager.cfg
  if not isinstance(metrics_cfg, dict):
    raise RuntimeError("Task does not configure a MetricsManager.")
  term_cfg = metrics_cfg.get("stair_sequence_event_logger")
  if term_cfg is None:
    raise RuntimeError("Task does not configure stair_sequence_event_logger.")
  logger = term_cfg.func
  close = getattr(logger, "close", None)
  if not callable(close):
    raise RuntimeError("stair_sequence_event_logger does not expose close().")
  close()


def run_export(task_id: str, cfg: ExportStairSequencesConfig) -> Path:
  """Collect a fixed number of frozen-policy steps and return the output path."""
  if cfg.num_envs <= 0:
    raise ValueError("num_envs must be positive.")
  if cfg.steps <= 0:
    raise ValueError("steps must be positive.")

  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  output_dir = Path(cfg.output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  os.environ["MJLAB_STAIR_EXPORT_DIR"] = str(output_dir)
  os.environ["MJLAB_STAIR_EXPORT_FLUSH_ROWS"] = str(max(cfg.flush_rows, 1))

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

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  wrapped = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  try:
    policy, _runner = load_inference_policy(
      env=wrapped,
      task_id=task_id,
      agent_cfg=agent_cfg,
      checkpoint_path=checkpoint_path,
      device=device,
    )
    obs = wrapped.get_observations()
    for _step in range(cfg.steps):
      with torch.no_grad():
        actions = policy(obs)
      step_result = wrapped.step(actions)
      reset_policy_state_from_step(policy, step_result)
      obs = step_result[0]
  finally:
    _close_sequence_logger(raw_env)
    wrapped.close()

  sequence_path = output_dir / "stair_sequences.csv"
  event_path = output_dir / "stair_events.csv"
  print(
    "[Stage 0] Export complete:",
    f"sequences={_count_csv_rows(sequence_path)}",
    f"events={_count_csv_rows(event_path)}",
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
    ExportStairSequencesConfig,
    args=remaining_args,
    default=ExportStairSequencesConfig(),
    prog=sys.argv[0] + f" {task_id}",
    config=mjlab.TYRO_FLAGS,
  )
  run_export(task_id, cfg)


if __name__ == "__main__":
  main()
