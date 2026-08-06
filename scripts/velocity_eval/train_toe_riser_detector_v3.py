# ruff: noqa: E402,I001
"""Preset training entrypoint for the deploy-friendly toe-riser detector."""

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
  "Mjlab-Velocity-Blind-Rough-TargetNavigation-ToeRiserDetector-"
  "SlowLatent-TeacherKL-Unitree-G1"
)
DEFAULT_CHECKPOINT_FILE = (
  "logs/rsl_rl/g1_blind_rough_target_navigation_slow_latent_teacherkl/"
  "Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1/"
  "base_stride_phase_event_impulse_m2000/model_2000.pt"
)
DEFAULT_OUTPUT_DIR = (
  "eval_outputs/stair_stage2/model2000_seed42_toe_riser_deploy_v3_fixed_pool_v1"
)

TOE_RISER_V3_INPUT_SCHEMA = "toe_riser_deploy_v3"
TOE_RISER_V3_EXPECTED_OBS_DIM = 106
TOE_RISER_V3_HISTORY_LEN = 24
TOE_RISER_V3_FRAME_HIDDEN_DIM = 256
TOE_RISER_V3_RECURRENT_HIDDEN_DIM = 128
TOE_RISER_V3_HEAD_HIDDEN_DIM = 64
TOE_RISER_V3_OUTPUT_DIM = 6
TOE_RISER_V3_TRAINED_LABEL_INDICES = (4, 5)
TOE_RISER_V3_DUMMY_LOW_LOGIT_INDICES = (0, 1, 2, 3)


def _with_toe_riser_v3_defaults(
  cfg: OnlineFootEventDetectorConfig,
) -> OnlineFootEventDetectorConfig:
  """Pin the generic online trainer to the toe-riser-only v3 task."""
  checkpoint_file = cfg.checkpoint_file
  if checkpoint_file == OnlineFootEventDetectorConfig.checkpoint_file:
    checkpoint_file = DEFAULT_CHECKPOINT_FILE
  output_dir = cfg.output_dir
  if output_dir == OnlineFootEventDetectorConfig.output_dir:
    output_dir = DEFAULT_OUTPUT_DIR
  return replace(
    cfg,
    checkpoint_file=checkpoint_file,
    output_dir=output_dir,
    input_schema=TOE_RISER_V3_INPUT_SCHEMA,
    include_gait_phase=True,
    expected_obs_dim=TOE_RISER_V3_EXPECTED_OBS_DIM,
    toe_riser_only_model=True,
    toe_only_finetune=False,
    footprint_only_model=False,
    history_len=TOE_RISER_V3_HISTORY_LEN
    if cfg.history_len == OnlineFootEventDetectorConfig.history_len
    else cfg.history_len,
    frame_hidden_dim=TOE_RISER_V3_FRAME_HIDDEN_DIM
    if cfg.frame_hidden_dim == OnlineFootEventDetectorConfig.frame_hidden_dim
    else cfg.frame_hidden_dim,
    recurrent_hidden_dim=TOE_RISER_V3_RECURRENT_HIDDEN_DIM
    if cfg.recurrent_hidden_dim == OnlineFootEventDetectorConfig.recurrent_hidden_dim
    else cfg.recurrent_hidden_dim,
    head_hidden_dim=TOE_RISER_V3_HEAD_HIDDEN_DIM
    if cfg.head_hidden_dim == OnlineFootEventDetectorConfig.head_hidden_dim
    else cfg.head_hidden_dim,
    selection_metric=(
      "toe_riser_high_recall_score"
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
    touchdown_positive_fraction=0.0,
    touchdown_soft_positive_fraction=0.0,
    false_negative_hard_positive_fraction=0.0,
    toe_positive_fraction=0.30
    if cfg.toe_positive_fraction == OnlineFootEventDetectorConfig.toe_positive_fraction
    else cfg.toe_positive_fraction,
    toe_soft_positive_fraction=0.20
    if cfg.toe_soft_positive_fraction
    == OnlineFootEventDetectorConfig.toe_soft_positive_fraction
    else cfg.toe_soft_positive_fraction,
    stair_hard_negative_fraction=0.20
    if cfg.stair_hard_negative_fraction
    == OnlineFootEventDetectorConfig.stair_hard_negative_fraction
    else cfg.stair_hard_negative_fraction,
    false_positive_hard_negative_fraction=0.10
    if cfg.false_positive_hard_negative_fraction
    == OnlineFootEventDetectorConfig.false_positive_hard_negative_fraction
    else cfg.false_positive_hard_negative_fraction,
    false_negative_toe_hard_positive_fraction=0.20
    if cfg.false_negative_toe_hard_positive_fraction
    == OnlineFootEventDetectorConfig.false_negative_toe_hard_positive_fraction
    else cfg.false_negative_toe_hard_positive_fraction,
    soft_touchdown_radius=0,
    soft_toe_hit_radius=2
    if cfg.soft_toe_hit_radius == OnlineFootEventDetectorConfig.soft_toe_hit_radius
    else cfg.soft_toe_hit_radius,
    soft_event_radius1_value=0.8
    if cfg.soft_event_radius1_value
    == OnlineFootEventDetectorConfig.soft_event_radius1_value
    else cfg.soft_event_radius1_value,
    soft_event_radius2_value=0.5
    if cfg.soft_event_radius2_value
    == OnlineFootEventDetectorConfig.soft_event_radius2_value
    else cfg.soft_event_radius2_value,
    tversky_alpha=0.15
    if cfg.tversky_alpha == OnlineFootEventDetectorConfig.tversky_alpha
    else cfg.tversky_alpha,
    tversky_beta=0.90
    if cfg.tversky_beta == OnlineFootEventDetectorConfig.tversky_beta
    else cfg.tversky_beta,
    mine_false_positive_hard_negatives=True,
    mine_false_positive_touchdown_hard_negatives=False,
    mine_false_negative_hard_positives=False,
    mine_false_negative_toe_hard_positives=True,
    hard_toe_positive_mining_threshold=0.75
    if cfg.hard_toe_positive_mining_threshold
    == OnlineFootEventDetectorConfig.hard_toe_positive_mining_threshold
    else cfg.hard_toe_positive_mining_threshold,
    hard_toe_positive_mining_window_frames=4
    if cfg.hard_toe_positive_mining_window_frames
    == OnlineFootEventDetectorConfig.hard_toe_positive_mining_window_frames
    else cfg.hard_toe_positive_mining_window_frames,
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
    baseline_toe_metric="toe_riser_high_recall_macro_f1",
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
  run_online_train(task_id, _with_toe_riser_v3_defaults(cfg))


if __name__ == "__main__":
  main()
