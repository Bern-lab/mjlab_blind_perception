# ruff: noqa: E402,I001
"""Preset training entrypoint for the deploy-friendly footprint detector."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import tyro

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from scripts.velocity_eval.train_foot_event_detector_online import (
  OnlineFootEventDetectorConfig,
  run_online_train,
)

from mjlab.tasks.registry import list_tasks

DEFAULT_TASK_ID = (
  "Mjlab-Velocity-Blind-Rough-TargetNavigation-FootprintDetector-"
  "SlowLatent-TeacherKL-Unitree-G1"
)
DEFAULT_OUTPUT_DIR = (
  "eval_outputs/stair_stage2/model51000_seed42_footprint_deploy_v3_fixed_pool_v2"
)
DEFAULT_BASELINE_METRICS_FILE = str(
  Path(__file__).resolve().parent
  / "baselines"
  / "footprint_old_best_v11_step2250_metrics.json"
)
FOOTPRINT_V3_REQUIRED_BASELINE_METRICS = (
  "footprint_deploy_score",
  "high_recall_footprint_score",
  "touchdown_deploy_macro_f1",
  "touchdown_stair_deploy_macro_f1",
  "touchdown_high_recall_macro_precision",
  "touchdown_high_recall_macro_recall",
  "touchdown_stair_high_recall_macro_precision",
  "touchdown_stair_high_recall_macro_recall",
  "touchdown_flat_high_recall_macro_f1",
  "left_contact_f1",
  "right_contact_f1",
  "touchdown_stair_deploy_true_event_count",
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
    expected_obs_dim=134,
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
      "touchdown_v3_surpass_score"
      if cfg.selection_metric == OnlineFootEventDetectorConfig.selection_metric
      else cfg.selection_metric
    ),
    max_updates=12_000
    if cfg.max_updates == OnlineFootEventDetectorConfig.max_updates
    else cfg.max_updates,
    steps=12_000 if cfg.steps == OnlineFootEventDetectorConfig.steps else cfg.steps,
    train_buffer_capacity=240_000
    if cfg.train_buffer_capacity == OnlineFootEventDetectorConfig.train_buffer_capacity
    else cfg.train_buffer_capacity,
    val_buffer_capacity=120_000
    if cfg.val_buffer_capacity == OnlineFootEventDetectorConfig.val_buffer_capacity
    else cfg.val_buffer_capacity,
    toe_hit_pos_weight=None,
    toe_positive_fraction=0.0,
    toe_soft_positive_fraction=0.0,
    false_negative_toe_hard_positive_fraction=0.0,
    soft_toe_hit_radius=0,
    mine_false_positive_hard_negatives=False,
    mine_false_negative_toe_hard_positives=False,
    baseline_toe_metric="",
    false_negative_hard_positive_fraction=0.30
    if cfg.false_negative_hard_positive_fraction
    == OnlineFootEventDetectorConfig.false_negative_hard_positive_fraction
    else cfg.false_negative_hard_positive_fraction,
    false_positive_hard_negative_fraction=0.10
    if cfg.false_positive_hard_negative_fraction
    == OnlineFootEventDetectorConfig.false_positive_hard_negative_fraction
    else cfg.false_positive_hard_negative_fraction,
    stair_hard_negative_fraction=0.10
    if cfg.stair_hard_negative_fraction
    == OnlineFootEventDetectorConfig.stair_hard_negative_fraction
    else cfg.stair_hard_negative_fraction,
    touchdown_positive_fraction=0.30
    if cfg.touchdown_positive_fraction
    == OnlineFootEventDetectorConfig.touchdown_positive_fraction
    else cfg.touchdown_positive_fraction,
    touchdown_soft_positive_fraction=0.20
    if cfg.touchdown_soft_positive_fraction
    == OnlineFootEventDetectorConfig.touchdown_soft_positive_fraction
    else cfg.touchdown_soft_positive_fraction,
    soft_touchdown_radius=2
    if cfg.soft_touchdown_radius == OnlineFootEventDetectorConfig.soft_touchdown_radius
    else cfg.soft_touchdown_radius,
    soft_event_radius1_value=0.8
    if cfg.soft_event_radius1_value
    == OnlineFootEventDetectorConfig.soft_event_radius1_value
    else cfg.soft_event_radius1_value,
    soft_event_radius2_value=0.5
    if cfg.soft_event_radius2_value
    == OnlineFootEventDetectorConfig.soft_event_radius2_value
    else cfg.soft_event_radius2_value,
    tversky_alpha=0.20
    if cfg.tversky_alpha == OnlineFootEventDetectorConfig.tversky_alpha
    else cfg.tversky_alpha,
    tversky_beta=0.85
    if cfg.tversky_beta == OnlineFootEventDetectorConfig.tversky_beta
    else cfg.tversky_beta,
    hard_positive_mining_threshold=0.75
    if cfg.hard_positive_mining_threshold
    == OnlineFootEventDetectorConfig.hard_positive_mining_threshold
    else cfg.hard_positive_mining_threshold,
    hard_positive_mining_window_frames=4
    if cfg.hard_positive_mining_window_frames
    == OnlineFootEventDetectorConfig.hard_positive_mining_window_frames
    else cfg.hard_positive_mining_window_frames,
    sweep_threshold_min=0.02
    if cfg.sweep_threshold_min == OnlineFootEventDetectorConfig.sweep_threshold_min
    else cfg.sweep_threshold_min,
    sweep_threshold_max=0.9999
    if cfg.sweep_threshold_max == OnlineFootEventDetectorConfig.sweep_threshold_max
    else cfg.sweep_threshold_max,
    sweep_threshold_steps=81
    if cfg.sweep_threshold_steps == OnlineFootEventDetectorConfig.sweep_threshold_steps
    else cfg.sweep_threshold_steps,
    early_stop_patience_evals=24
    if cfg.early_stop_patience_evals
    == OnlineFootEventDetectorConfig.early_stop_patience_evals
    else cfg.early_stop_patience_evals,
    baseline_metrics_file=(
      DEFAULT_BASELINE_METRICS_FILE
      if cfg.baseline_metrics_file
      == OnlineFootEventDetectorConfig.baseline_metrics_file
      else cfg.baseline_metrics_file
    ),
    baseline_touchdown_recall_tolerance=0.0
    if cfg.baseline_touchdown_recall_tolerance
    == OnlineFootEventDetectorConfig.baseline_touchdown_recall_tolerance
    else cfg.baseline_touchdown_recall_tolerance,
    baseline_stair_touchdown_recall_tolerance=0.0
    if cfg.baseline_stair_touchdown_recall_tolerance
    == OnlineFootEventDetectorConfig.baseline_stair_touchdown_recall_tolerance
    else cfg.baseline_stair_touchdown_recall_tolerance,
    baseline_flat_touchdown_f1_tolerance=0.0
    if cfg.baseline_flat_touchdown_f1_tolerance
    == OnlineFootEventDetectorConfig.baseline_flat_touchdown_f1_tolerance
    else cfg.baseline_flat_touchdown_f1_tolerance,
    baseline_required_metric_names=(
      FOOTPRINT_V3_REQUIRED_BASELINE_METRICS
      if cfg.baseline_required_metric_names
      == OnlineFootEventDetectorConfig.baseline_required_metric_names
      else cfg.baseline_required_metric_names
    ),
    baseline_required_metric_min_improvement=1.0e-4
    if cfg.baseline_required_metric_min_improvement
    == OnlineFootEventDetectorConfig.baseline_required_metric_min_improvement
    else cfg.baseline_required_metric_min_improvement,
    baseline_guard_gates_best=False
    if cfg.baseline_guard_gates_best
    == OnlineFootEventDetectorConfig.baseline_guard_gates_best
    else cfg.baseline_guard_gates_best,
    require_baseline_guard=False
    if cfg.require_baseline_guard
    == OnlineFootEventDetectorConfig.require_baseline_guard
    else cfg.require_baseline_guard,
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
