"""Tests for native viewer custom action dispatch."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

from mjlab.viewer.base import ViewerAction
from mjlab.viewer.native.viewer import NativeMujocoViewer


def _make_viewer(num_envs: int = 3) -> NativeMujocoViewer:
  env = MagicMock()
  env.cfg.viewer.env_idx = 0
  env.unwrapped.num_envs = num_envs
  return NativeMujocoViewer(env, MagicMock())


def test_toggle_actions_dispatch_by_action_enum():
  v = _make_viewer()

  assert not v._show_plots
  assert v._handle_custom_action(ViewerAction.TOGGLE_PLOTS, None)
  assert v._show_plots

  assert v._show_debug_vis
  assert v._handle_custom_action(ViewerAction.TOGGLE_DEBUG_VIS, None)
  assert not v._show_debug_vis

  assert not v._show_all_envs
  assert v._handle_custom_action(ViewerAction.TOGGLE_SHOW_ALL_ENVS, None)
  assert v._show_all_envs


def test_prev_next_env_actions_wrap_and_succeed():
  v = _make_viewer(num_envs=3)
  v.env_idx = 0
  assert v._handle_custom_action(ViewerAction.PREV_ENV, None)
  assert v.env_idx == 2
  assert v._handle_custom_action(ViewerAction.NEXT_ENV, None)
  assert v.env_idx == 0


def test_depth_camera_visualizer_uses_sensor_range():
  v = _make_viewer()
  sensor = MagicMock()
  sensor.cfg.visualizer_max_range = 3.0

  assert v._depth_camera_visualizer_range(sensor) == 3.0

  sensor.cfg.visualizer_max_range = None
  assert v._depth_camera_visualizer_range(sensor) == v._DEPTH_CAMERA_MAX_RANGE


def test_other_env_geoms_hidden_until_show_all_enabled():
  v = _make_viewer(num_envs=2)
  v.__dict__["vd"] = object()
  v._show_all_envs = False

  v._render_other_env_geoms(cast(Any, None), cast(Any, None), cast(Any, None))
