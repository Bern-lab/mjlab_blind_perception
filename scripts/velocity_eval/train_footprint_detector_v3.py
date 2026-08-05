"""Preset training entrypoint for the deploy-friendly footprint detector."""

from __future__ import annotations

import sys
from dataclasses import replace

import tyro
from scripts.velocity_eval.train_foot_event_detector_online import (
  OnlineFootEventDetectorConfig,
  run_online_train,
)

from mjlab.tasks.registry import list_tasks

DEFAULT_TASK_ID = (
  "Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1"
)
DEFAULT_OUTPUT_DIR = (
  "eval_outputs/stair_stage2/model51000_seed42_footprint_deploy_v3_footprint_only_v1"
)


def _with_footprint_v3_defaults(
  cfg: OnlineFootEventDetectorConfig,
) -> OnlineFootEventDetectorConfig:
  """Pin the generic online trainer to the footprint-only v3 task."""
  output_dir = cfg.output_dir
  if output_dir == OnlineFootEventDetectorConfig.output_dir:
    output_dir = DEFAULT_OUTPUT_DIR
  return replace(
    cfg,
    output_dir=output_dir,
    input_schema="footprint_deploy_v3",
    include_gait_phase=True,
    expected_obs_dim=128,
    footprint_only_model=True,
    toe_only_finetune=False,
    toe_riser_only_model=False,
    history_len=24
    if cfg.history_len == OnlineFootEventDetectorConfig.history_len
    else cfg.history_len,
    frame_hidden_dim=256
    if cfg.frame_hidden_dim == OnlineFootEventDetectorConfig.frame_hidden_dim
    else cfg.frame_hidden_dim,
    recurrent_hidden_dim=128
    if cfg.recurrent_hidden_dim == OnlineFootEventDetectorConfig.recurrent_hidden_dim
    else cfg.recurrent_hidden_dim,
    head_hidden_dim=64
    if cfg.head_hidden_dim == OnlineFootEventDetectorConfig.head_hidden_dim
    else cfg.head_hidden_dim,
    selection_metric=(
      "high_recall_footprint_score"
      if cfg.selection_metric == OnlineFootEventDetectorConfig.selection_metric
      else cfg.selection_metric
    ),
    max_updates=9000
    if cfg.max_updates == OnlineFootEventDetectorConfig.max_updates
    else cfg.max_updates,
    steps=9000 if cfg.steps == OnlineFootEventDetectorConfig.steps else cfg.steps,
    train_buffer_capacity=160_000
    if cfg.train_buffer_capacity == OnlineFootEventDetectorConfig.train_buffer_capacity
    else cfg.train_buffer_capacity,
    val_buffer_capacity=80_000
    if cfg.val_buffer_capacity == OnlineFootEventDetectorConfig.val_buffer_capacity
    else cfg.val_buffer_capacity,
    false_negative_hard_positive_fraction=0.20
    if cfg.false_negative_hard_positive_fraction
    == OnlineFootEventDetectorConfig.false_negative_hard_positive_fraction
    else cfg.false_negative_hard_positive_fraction,
    false_positive_hard_negative_fraction=0.15
    if cfg.false_positive_hard_negative_fraction
    == OnlineFootEventDetectorConfig.false_positive_hard_negative_fraction
    else cfg.false_positive_hard_negative_fraction,
    touchdown_positive_fraction=0.25
    if cfg.touchdown_positive_fraction
    == OnlineFootEventDetectorConfig.touchdown_positive_fraction
    else cfg.touchdown_positive_fraction,
    touchdown_soft_positive_fraction=0.15
    if cfg.touchdown_soft_positive_fraction
    == OnlineFootEventDetectorConfig.touchdown_soft_positive_fraction
    else cfg.touchdown_soft_positive_fraction,
  )


def main() -> None:
  import mjlab.tasks as _tasks  # noqa: F401

  task_choices = tuple(list_tasks())
  remaining_args = sys.argv[1:]
  task_id = DEFAULT_TASK_ID
  if remaining_args and not remaining_args[0].startswith("-"):
    task_id = remaining_args[0]
    remaining_args = remaining_args[1:]
  if task_id not in task_choices:
    choices = ", ".join(task_choices)
    raise ValueError(f"Unknown task_id {task_id!r}. Available tasks: {choices}")
  cfg = tyro.cli(OnlineFootEventDetectorConfig, args=remaining_args)
  run_online_train(task_id, _with_footprint_v3_defaults(cfg))


if __name__ == "__main__":
  main()
