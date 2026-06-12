"""Tests for velocity-task observation helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationManager,
  ObservationTermCfg,
)
from mjlab.sensor import CameraSensorData
from mjlab.tasks.velocity.mdp.observations import CameraDepthStack, camera_depth


class _FakeCamera:
  def __init__(self, depth: torch.Tensor) -> None:
    self.data = CameraSensorData(depth=depth)

  def set_depth(self, depth: torch.Tensor) -> None:
    self.data = CameraSensorData(depth=depth)


def _depth(values: list[float], device: str, height: int = 2, width: int = 2):
  base = torch.tensor(values, dtype=torch.float32, device=device)
  return base[:, None, None, None].expand(-1, height, width, 1).clone()


def test_camera_depth_stack_rolls_channels_and_resets_per_env() -> None:
  device = get_test_device()
  camera = _FakeCamera(_depth([1.0, 2.0], device))
  env = SimpleNamespace(
    num_envs=2,
    device=device,
    scene={"front_depth": camera},
  )
  env = cast(ManagerBasedRlEnv, env)

  manager = ObservationManager(
    {
      "camera": ObservationGroupCfg(
        terms={
          "front_depth": ObservationTermCfg(
            func=CameraDepthStack,
            params={
              "sensor_name": "front_depth",
              "cutoff_distance": 10.0,
              "stack_length": 3,
            },
          )
        },
        concatenate_dim=0,
      )
    },
    env,
  )

  obs = manager.compute(update_history=True)["camera"]
  assert isinstance(obs, torch.Tensor)
  assert obs.shape == (2, 3, 2, 2)
  torch.testing.assert_close(
    obs[:, :, 0, 0],
    torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], device=device),
  )

  camera.set_depth(_depth([3.0, 4.0], device))
  obs = manager.compute(update_history=True)["camera"]
  assert isinstance(obs, torch.Tensor)
  torch.testing.assert_close(
    obs[:, :, 0, 0],
    torch.tensor([[0.3, 0.1, 0.1], [0.4, 0.2, 0.2]], device=device),
  )

  manager.reset(env_ids=torch.tensor([1], device=device))
  camera.set_depth(_depth([5.0, 6.0], device))
  obs = manager.compute(update_history=True)["camera"]
  assert isinstance(obs, torch.Tensor)
  torch.testing.assert_close(
    obs[:, :, 0, 0],
    torch.tensor([[0.5, 0.3, 0.1], [0.6, 0.6, 0.6]], device=device),
  )


def test_g1_perception_depth_camera_uses_eight_frame_stack() -> None:
  from mjlab.tasks.velocity.config.g1.blind_rough_perception_env_cfg import (
    STUDENT_DEPTH_SENSOR,
    unitree_g1_blind_rough_perception_env_cfg,
  )

  cfg = unitree_g1_blind_rough_perception_env_cfg()
  term = cfg.observations["camera_stack"].terms["front_depth_stack"]

  assert term.func is CameraDepthStack
  assert term.params["sensor_name"] == STUDENT_DEPTH_SENSOR
  assert term.params["stack_length"] == 8


def test_g1_perception_teacher_camera_matches_checkpoint_contract() -> None:
  from mjlab.sensor import CameraSensorCfg
  from mjlab.tasks.velocity.config.g1.blind_rough_perception_env_cfg import (
    TEACHER_DEPTH_SENSOR,
    unitree_g1_blind_rough_perception_env_cfg,
  )

  cfg = unitree_g1_blind_rough_perception_env_cfg()
  term = cfg.observations["camera"].terms["front_depth"]
  sensors = {
    sensor.name: sensor
    for sensor in cfg.scene.sensors or ()
    if isinstance(sensor, CameraSensorCfg)
  }
  teacher_sensor = sensors[TEACHER_DEPTH_SENSOR]

  assert term.func is camera_depth
  assert term.params["sensor_name"] == TEACHER_DEPTH_SENSOR
  assert term.params["cutoff_distance"] == 5.0
  assert teacher_sensor.width == 64
  assert teacher_sensor.height == 64


def test_g1_perception_rl_separates_actor_and_teacher_camera_groups() -> None:
  from mjlab.tasks.velocity.config.g1.rl_cfg import (
    unitree_g1_blind_rough_perception_teacherkl_runner_cfg,
    unitree_g1_blind_rough_target_navigation_perception_teacherkl_runner_cfg,
  )

  cfgs = (
    unitree_g1_blind_rough_perception_teacherkl_runner_cfg(),
    unitree_g1_blind_rough_target_navigation_perception_teacherkl_runner_cfg(),
  )

  for cfg in cfgs:
    assert cfg.obs_groups["actor"] == ("actor", "camera_stack")
    assert cfg.obs_groups["teacher"] == ("teacher", "camera")
