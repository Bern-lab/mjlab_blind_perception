from __future__ import annotations

import yaml
from rsl_rl.algorithms.ppo_teacher_kl import PPOTeacherKL
from scripts.velocity_eval.policy_io import (
  get_clip_actions,
  load_checkpoint_agent_cfg,
  make_inference_train_cfg,
  resolve_inference_agent_cfg,
)


def test_teacherkl_inference_config_keeps_slow_latent_obs_group() -> None:
  cfg = {
    "upload_model": True,
    "clip_actions": 0.75,
    "obs_groups": {
      "actor": ("actor",),
      "latent": ("latent",),
      "critic": ("critic",),
      "teacher": ("teacher", "camera"),
    },
    "algorithm": {
      "class_name": "rsl_rl.algorithms.ppo_teacher_kl:PPOTeacherKL",
      "teacher_kl_cfg": {"enabled": True},
      "safe_stride_probe_only": False,
      "safe_stride_probe_learning_rate": 0.001,
      "geometry_probe_only": False,
      "geometry_probe_learning_rate": 0.001,
      "geometry_probe_permute_depth_labels": False,
      "share_cnn_encoders": False,
      "learning_rate": 0.001,
    },
    "teacher": {"class_name": "TeacherModel"},
  }

  inference_cfg = make_inference_train_cfg(cfg)

  assert inference_cfg["upload_model"] is False
  assert inference_cfg["algorithm"]["class_name"] == "PPO"
  assert "teacher_kl_cfg" not in inference_cfg["algorithm"]
  assert "safe_stride_probe_only" not in inference_cfg["algorithm"]
  assert "safe_stride_probe_learning_rate" not in inference_cfg["algorithm"]
  assert "geometry_probe_only" not in inference_cfg["algorithm"]
  assert "geometry_probe_learning_rate" not in inference_cfg["algorithm"]
  assert "geometry_probe_permute_depth_labels" not in inference_cfg["algorithm"]
  assert inference_cfg["algorithm"]["share_cnn_encoders"] is False
  assert inference_cfg["algorithm"]["learning_rate"] == 0.001
  assert "teacher" not in inference_cfg
  assert inference_cfg["obs_groups"] == {
    "actor": ("actor",),
    "latent": ("latent",),
    "critic": ("critic",),
  }
  assert get_clip_actions(inference_cfg) == 0.75


def test_teacherkl_inference_config_accepts_callable_class_name() -> None:
  cfg = {
    "obs_groups": {
      "actor": ("actor",),
      "latent": ("latent",),
      "critic": ("critic",),
      "teacher": ("teacher",),
    },
    "algorithm": {
      "class_name": PPOTeacherKL,
      "teacher_kl_cfg": {"enabled": True},
      "safe_stride_probe_only": True,
      "share_cnn_encoders": False,
      "learning_rate": 0.001,
    },
    "teacher": {"class_name": "TeacherModel"},
  }

  inference_cfg = make_inference_train_cfg(cfg)

  assert inference_cfg["algorithm"] == {
    "class_name": "PPO",
    "share_cnn_encoders": False,
    "learning_rate": 0.001,
  }
  assert inference_cfg["obs_groups"] == {
    "actor": ("actor",),
    "latent": ("latent",),
    "critic": ("critic",),
  }


def test_resolve_inference_agent_cfg_prefers_saved_checkpoint_yaml(tmp_path) -> None:
  run_dir = tmp_path / "run"
  params_dir = run_dir / "params"
  params_dir.mkdir(parents=True)
  checkpoint_path = run_dir / "model_1500.pt"
  checkpoint_path.touch()
  saved_cfg = {
    "clip_actions": 0.25,
    "experiment_name": "saved_policy",
    "obs_groups": {"actor": ("actor",), "latent": ("latent",)},
    "algorithm": {"class_name": "PPO", "learning_rate": 0.0003},
  }
  (params_dir / "agent.yaml").write_text(
    yaml.dump(saved_cfg, sort_keys=True),
    encoding="utf-8",
  )

  loaded_cfg = load_checkpoint_agent_cfg(checkpoint_path)
  resolved_cfg = resolve_inference_agent_cfg(
    checkpoint_path=checkpoint_path,
    agent_cfg={"experiment_name": "current_task", "clip_actions": None},
    verbose=False,
  )

  assert loaded_cfg == saved_cfg
  assert resolved_cfg == saved_cfg
  assert get_clip_actions(resolved_cfg) == 0.25


def test_resolve_inference_agent_cfg_falls_back_without_saved_yaml(tmp_path) -> None:
  checkpoint_path = tmp_path / "model_1.pt"
  checkpoint_path.touch()
  current_cfg = {"experiment_name": "current_task", "clip_actions": None}

  resolved_cfg = resolve_inference_agent_cfg(
    checkpoint_path=checkpoint_path,
    agent_cfg=current_cfg,
    verbose=False,
  )

  assert resolved_cfg is current_cfg
