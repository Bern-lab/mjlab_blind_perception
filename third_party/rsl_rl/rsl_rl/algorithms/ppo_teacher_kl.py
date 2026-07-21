# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import math
import torch
import torch.nn.functional as functional
from itertools import chain
from tensordict import TensorDict
from typing import Any, cast

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, unpad_trajectories


class PPOTeacherKL(PPO):
    """PPO with an additional frozen-teacher guidance term."""

    teacher: MLPModel | None
    """The frozen teacher actor model."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        teacher_kl_cfg: dict | None = None,
        teacher_checkpoint_path: str | None = None,
        safe_stride_probe_only: bool = False,
        safe_stride_probe_learning_rate: float = 1.0e-3,
        geometry_probe_only: bool = False,
        geometry_probe_learning_rate: float = 1.0e-3,
        geometry_probe_permute_depth_labels: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize PPO and store teacher-guidance configuration."""
        super().__init__(actor, critic, storage, **kwargs)

        self.teacher = None
        self.teacher_loaded = False
        self.teacher_kl_cfg = dict(teacher_kl_cfg or {})
        self._teacher_kl_shapes_checked = False
        self.teacher_guidance_enabled = bool(self.teacher_kl_cfg.get("enabled", True))
        self.teacher_imitation_only = bool(self.teacher_kl_cfg.get("imitation_only", False))
        if self.teacher_imitation_only and not self.teacher_guidance_enabled:
            raise ValueError("teacher_kl_cfg.imitation_only requires enabled=True.")
        self.teacher_imitation_loss_coef = float(self.teacher_kl_cfg.get("imitation_loss_coef", 1.0))
        if self.teacher_imitation_loss_coef <= 0.0:
            raise ValueError("teacher_kl_cfg.imitation_loss_coef must be positive.")

        # Allow checkpoint path to be supplied either inside teacher_kl_cfg or as a direct keyword.
        self.teacher_checkpoint_path: str | None = None
        self.set_teacher_checkpoint(teacher_checkpoint_path or self.teacher_kl_cfg.get("checkpoint_path"))

        self.teacher_kl_lambda_start = float(self.teacher_kl_cfg.get("lambda_start", 1.0))
        self.teacher_kl_lambda_end = float(self.teacher_kl_cfg.get("lambda_end", 0.0))
        self.teacher_kl_warmup_iters = int(self.teacher_kl_cfg.get("warmup_iters", 0))
        self.teacher_kl_constant_iters = int(self.teacher_kl_cfg.get("constant_iters", 0))
        self.teacher_kl_anneal_iters = int(self.teacher_kl_cfg.get("anneal_iters", 3000))
        self.teacher_kl_schedule = str(self.teacher_kl_cfg.get("schedule", "linear"))
        self.teacher_kl_iteration = int(self.teacher_kl_cfg.get("iteration", 0))
        self.teacher_guidance_loss_type = str(self.teacher_kl_cfg.get("loss_type", "kl"))
        if self.teacher_guidance_loss_type not in {"kl", "mean_mse", "mean_huber"}:
            raise ValueError(f"Unsupported teacher guidance loss_type: {self.teacher_guidance_loss_type}")
        self.teacher_guidance_huber_delta = float(self.teacher_kl_cfg.get("huber_delta", 1.0))
        if self.teacher_guidance_huber_delta <= 0.0:
            raise ValueError("teacher_kl_cfg.huber_delta must be positive.")
        self.teacher_forward_chunk_size = int(self.teacher_kl_cfg.get("teacher_forward_chunk_size", 2048))
        if self.teacher_forward_chunk_size <= 0:
            raise ValueError("teacher_kl_cfg.teacher_forward_chunk_size must be positive.")
        max_teacher_loss = self.teacher_kl_cfg.get("max_teacher_loss", self.teacher_kl_cfg.get("max_loss"))
        self.teacher_guidance_max_loss = None if max_teacher_loss is None else float(max_teacher_loss)

        self.teacher_kl_cfg.update({
            "enabled": self.teacher_guidance_enabled,
            "imitation_only": self.teacher_imitation_only,
            "imitation_loss_coef": self.teacher_imitation_loss_coef,
            "loss_type": self.teacher_guidance_loss_type,
            "lambda_start": self.teacher_kl_lambda_start,
            "lambda_end": self.teacher_kl_lambda_end,
            "warmup_iters": self.teacher_kl_warmup_iters,
            "constant_iters": self.teacher_kl_constant_iters,
            "anneal_iters": self.teacher_kl_anneal_iters,
            "schedule": self.teacher_kl_schedule,
            "huber_delta": self.teacher_guidance_huber_delta,
            "teacher_forward_chunk_size": self.teacher_forward_chunk_size,
            "max_teacher_loss": self.teacher_guidance_max_loss,
            "log_kl_when_lambda_zero": self.teacher_kl_cfg.get("log_kl_when_lambda_zero", True),
        })
        self._safe_stride_update_statistics: dict[str, torch.Tensor] | None = None
        self._geometry_probe_update_statistics: dict[str, torch.Tensor] | None = None
        self.safe_stride_probe_only = bool(safe_stride_probe_only)
        self.geometry_probe_only = bool(geometry_probe_only)
        self.geometry_probe_permute_depth_labels = bool(geometry_probe_permute_depth_labels)
        if self.safe_stride_probe_only and self.geometry_probe_only:
            raise ValueError("SafeStride and geometry Probe modes are mutually exclusive.")
        self._safe_stride_probe_frozen_state: dict[str, torch.Tensor] | None = None
        self._geometry_probe_frozen_state: dict[str, torch.Tensor] | None = None
        if self.safe_stride_probe_only:
            if self.teacher_guidance_enabled:
                raise ValueError("safe_stride_probe_only requires teacher guidance to be disabled.")
            if self.rnd is not None or self.symmetry is not None:
                raise ValueError("safe_stride_probe_only does not support RND or symmetry.")
            safe_stride_head = getattr(self.actor, "safe_stride_head", None)
            if safe_stride_head is None:
                raise ValueError("safe_stride_probe_only requires actor.safe_stride_head.")
            for parameter in self.actor.parameters():
                parameter.requires_grad_(False)
            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)
            for parameter in safe_stride_head.parameters():
                parameter.requires_grad_(True)
            safe_stride_width_head = getattr(
                self.actor,
                "safe_stride_width_head",
                None,
            )
            if safe_stride_width_head is not None:
                for parameter in safe_stride_width_head.parameters():
                    parameter.requires_grad_(True)
            safe_stride_confidence_head = getattr(
                self.actor,
                "safe_stride_confidence_head",
                None,
            )
            if safe_stride_confidence_head is not None:
                for parameter in safe_stride_confidence_head.parameters():
                    parameter.requires_grad_(True)
            probe_parameters = safe_stride_head.parameters()
            if safe_stride_width_head is not None:
                probe_parameters = chain(
                    probe_parameters,
                    safe_stride_width_head.parameters(),
                )
            if safe_stride_confidence_head is not None:
                probe_parameters = chain(
                    probe_parameters,
                    safe_stride_confidence_head.parameters(),
                )
            self.optimizer = torch.optim.Adam(
                probe_parameters,
                lr=float(safe_stride_probe_learning_rate),
            )
            self.learning_rate = float(safe_stride_probe_learning_rate)
            self.freeze_normalization_updates = True
        if self.geometry_probe_only:
            if self.teacher_guidance_enabled:
                raise ValueError("geometry_probe_only requires teacher guidance to be disabled.")
            if self.rnd is not None or self.symmetry is not None:
                raise ValueError("geometry_probe_only does not support RND or symmetry.")
            geometry_probe_head = getattr(self.actor, "geometry_probe_head", None)
            if geometry_probe_head is None:
                raise ValueError("geometry_probe_only requires actor.geometry_probe_head.")
            for parameter in self.actor.parameters():
                parameter.requires_grad_(False)
            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)
            for parameter in geometry_probe_head.parameters():
                parameter.requires_grad_(True)
            self.optimizer = torch.optim.Adam(
                geometry_probe_head.parameters(),
                lr=float(geometry_probe_learning_rate),
            )
            self.learning_rate = float(geometry_probe_learning_rate)
            self.freeze_normalization_updates = True

    def _capture_safe_stride_probe_frozen_state(self) -> dict[str, torch.Tensor]:
        """Snapshot every actor/critic tensor outside the isolated decoder."""
        frozen: dict[str, torch.Tensor] = {}
        for name, value in self.actor.state_dict().items():
            if not name.startswith((
                "safe_stride_head.",
                "safe_stride_width_head.",
                "safe_stride_confidence_head.",
            )):
                frozen[f"actor.{name}"] = value.detach().cpu().clone()
        for name, value in self.critic.state_dict().items():
            frozen[f"critic.{name}"] = value.detach().cpu().clone()
        return frozen

    def _verify_safe_stride_probe_frozen_state(self) -> None:
        """Fail immediately if policy parameters or normalization buffers drift."""
        if self._safe_stride_probe_frozen_state is None:
            return
        current = self._capture_safe_stride_probe_frozen_state()
        for name, expected in self._safe_stride_probe_frozen_state.items():
            if name not in current or not torch.equal(current[name], expected):
                raise RuntimeError(f"SafeStride probe modified frozen state tensor {name!r}.")

    def _capture_geometry_probe_frozen_state(self) -> dict[str, torch.Tensor]:
        """Snapshot every actor/critic tensor outside the geometry Probe."""
        frozen: dict[str, torch.Tensor] = {}
        for name, value in self.actor.state_dict().items():
            if not name.startswith("geometry_probe_head."):
                frozen[f"actor.{name}"] = value.detach().cpu().clone()
        for name, value in self.critic.state_dict().items():
            frozen[f"critic.{name}"] = value.detach().cpu().clone()
        return frozen

    def _verify_geometry_probe_frozen_state(self) -> None:
        """Fail immediately if any frozen policy or normalization tensor drifts."""
        if self._geometry_probe_frozen_state is None:
            return
        current = self._capture_geometry_probe_frozen_state()
        for name, expected in self._geometry_probe_frozen_state.items():
            if name not in current or not torch.equal(current[name], expected):
                raise RuntimeError(f"Geometry Probe modified frozen state tensor {name!r}.")

    def set_teacher_checkpoint(self, checkpoint_path: str | None) -> None:
        """Set or clear the checkpoint path used to initialize the frozen teacher."""
        self.teacher_checkpoint_path = checkpoint_path
        if checkpoint_path is None:
            self.teacher_kl_cfg.pop("checkpoint_path", None)
        else:
            self.teacher_kl_cfg["checkpoint_path"] = checkpoint_path

    def set_teacher_kl_schedule(
        self,
        lambda_start: float | None = None,
        lambda_end: float | None = None,
        warmup_iters: int | None = None,
        constant_iters: int | None = None,
        anneal_iters: int | None = None,
        schedule: str | None = None,
    ) -> None:
        """Update the teacher-KL weight schedule."""
        if lambda_start is not None:
            self.teacher_kl_lambda_start = float(lambda_start)
            self.teacher_kl_cfg["lambda_start"] = self.teacher_kl_lambda_start
        if lambda_end is not None:
            self.teacher_kl_lambda_end = float(lambda_end)
            self.teacher_kl_cfg["lambda_end"] = self.teacher_kl_lambda_end
        if warmup_iters is not None:
            self.teacher_kl_warmup_iters = int(warmup_iters)
            self.teacher_kl_cfg["warmup_iters"] = self.teacher_kl_warmup_iters
        if constant_iters is not None:
            self.teacher_kl_constant_iters = int(constant_iters)
            self.teacher_kl_cfg["constant_iters"] = self.teacher_kl_constant_iters
        if anneal_iters is not None:
            self.teacher_kl_anneal_iters = int(anneal_iters)
            self.teacher_kl_cfg["anneal_iters"] = self.teacher_kl_anneal_iters
        if schedule is not None:
            self.teacher_kl_schedule = str(schedule)
            self.teacher_kl_cfg["schedule"] = self.teacher_kl_schedule

    def get_teacher_kl_lambda(self) -> float:
        """Return the current teacher-KL loss weight.

        Warmup takes precedence over all schedules. After warmup, ``constant`` keeps
        ``lambda_start`` fixed, ``linear`` and ``cosine`` anneal toward ``lambda_end``,
        and ``constant_then_linear`` holds ``lambda_start`` before linear annealing.
        """
        if self.teacher_kl_iteration < self.teacher_kl_warmup_iters:
            return 0.0

        schedule = self.teacher_kl_schedule
        if schedule == "constant":
            return self.teacher_kl_lambda_start

        schedule_iteration = self.teacher_kl_iteration - self.teacher_kl_warmup_iters
        if schedule == "constant_then_linear":
            if schedule_iteration < self.teacher_kl_constant_iters:
                return self.teacher_kl_lambda_start
            schedule = "linear"
            schedule_iteration -= self.teacher_kl_constant_iters

        if self.teacher_kl_anneal_iters <= 0:
            return self.teacher_kl_lambda_end

        progress = min(schedule_iteration / self.teacher_kl_anneal_iters, 1.0)
        if schedule == "linear":
            alpha = progress
        elif schedule == "cosine":
            alpha = 0.5 * (1.0 - math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unsupported teacher KL schedule: {self.teacher_kl_schedule}")

        return self.teacher_kl_lambda_start * (1.0 - alpha) + self.teacher_kl_lambda_end * alpha

    def _freeze_teacher(self) -> None:
        """Keep the teacher actor in inference mode and out of gradient updates."""
        if self.teacher is None:
            return

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    def load_teacher_checkpoint(self, checkpoint_path: str | None = None, strict: bool = True) -> None:
        """Load the frozen teacher actor from an rsl_rl PPO checkpoint."""
        if self.teacher is None:
            raise RuntimeError("Cannot load a teacher checkpoint before constructing the teacher model.")

        if checkpoint_path is not None:
            self.set_teacher_checkpoint(checkpoint_path)
        if self.teacher_checkpoint_path is None:
            raise ValueError("teacher_kl_cfg.checkpoint_path is required for PPOTeacherKL.")

        loaded_dict = torch.load(self.teacher_checkpoint_path, weights_only=False, map_location=self.device)
        if "actor_state_dict" not in loaded_dict:
            raise KeyError(f"Cannot find 'actor_state_dict' in teacher checkpoint: {self.teacher_checkpoint_path}")

        self.teacher.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        self.teacher_loaded = True
        self._teacher_kl_shapes_checked = False
        self._freeze_teacher()
        print(
            "Loaded frozen teacher actor from "
            f"'{self.teacher_checkpoint_path}' with obs groups {self.teacher.obs_groups}."
        )

    def _validate_distribution_params(
        self,
        teacher_params: tuple[torch.Tensor, ...],
        student_params: tuple[torch.Tensor, ...],
    ) -> None:
        """Validate teacher and student distribution parameter compatibility."""
        if len(teacher_params) != len(student_params):
            raise RuntimeError(
                "Teacher/student distribution parameter count mismatch: "
                f"teacher={len(teacher_params)}, student={len(student_params)}"
            )

        for index, (teacher_param, student_param) in enumerate(zip(teacher_params, student_params)):
            if teacher_param.shape != student_param.shape:
                raise RuntimeError(
                    f"Teacher/student distribution parameter {index} shape mismatch: "
                    f"teacher={tuple(teacher_param.shape)}, student={tuple(student_param.shape)}"
                )

        if self.teacher_kl_cfg.get("debug_shapes", False):
            shape_summary = [tuple(param.shape) for param in teacher_params]
            print(f"Teacher/student distribution parameter shapes verified: {shape_summary}")

    def _distributed_mean_scalar(self, value: torch.Tensor) -> torch.Tensor:
        """Average a scalar tensor across distributed workers for logging."""
        value = value.detach()
        if self.is_multi_gpu:
            value = value.clone()
            distributed = cast(Any, torch.distributed)
            distributed.all_reduce(value, op=distributed.ReduceOp.SUM)
            value /= self.gpu_world_size
        return value

    def _cap_mean_teacher_loss(self, teacher_loss: torch.Tensor) -> torch.Tensor:
        """Apply the optional hard cap for mean-only teacher guidance losses."""
        if self.teacher_guidance_max_loss is None:
            return teacher_loss
        return teacher_loss.clamp(max=self.teacher_guidance_max_loss)

    def _compute_mean_teacher_loss(
        self,
        teacher_params: tuple[torch.Tensor, ...],
        student_params: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute action-mean teacher guidance losses and diagnostics."""
        teacher_mean = teacher_params[0]
        student_mean = student_params[0]
        if self.teacher_guidance_loss_type == "mean_mse":
            mean_loss = (student_mean - teacher_mean).pow(2).sum(dim=-1).mean()
            loss_key = "teacher_mean_mse"
        elif self.teacher_guidance_loss_type == "mean_huber":
            mean_loss = (
                functional
                .smooth_l1_loss(
                    student_mean,
                    teacher_mean,
                    beta=self.teacher_guidance_huber_delta,
                    reduction="none",
                )
                .sum(dim=-1)
                .mean()
            )
            loss_key = "teacher_mean_huber"
        else:
            raise ValueError(f"Mean teacher loss requested for loss_type={self.teacher_guidance_loss_type}")

        mean_loss_for_update = self._cap_mean_teacher_loss(mean_loss)
        action_mean_l2 = torch.linalg.vector_norm(student_mean.detach() - teacher_mean.detach(), dim=-1).mean()
        teacher_kl = self.actor.get_kl_divergence(
            teacher_params,
            tuple(param.detach() for param in student_params),
        ).mean()

        return mean_loss_for_update, {
            loss_key: mean_loss,
            "teacher_loss_for_update": mean_loss_for_update,
            "teacher_action_mean_l2": action_mean_l2,
            "teacher_kl": teacher_kl,
        }

    def _compute_kl_teacher_loss(
        self,
        teacher_params: tuple[torch.Tensor, ...],
        student_params: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the legacy full-distribution teacher KL loss."""
        teacher_kl = self.actor.get_kl_divergence(teacher_params, student_params).mean()

        max_kl_loss = self.teacher_kl_cfg.get("max_kl_loss")
        if max_kl_loss is not None:
            max_kl_loss = float(max_kl_loss)
            tail_slope = float(self.teacher_kl_cfg.get("max_kl_loss_tail_slope", 0.0))
            if tail_slope > 0.0:
                teacher_kl_for_loss = torch.where(
                    teacher_kl <= max_kl_loss,
                    teacher_kl,
                    max_kl_loss + tail_slope * (teacher_kl - max_kl_loss),
                )
            else:
                teacher_kl_for_loss = teacher_kl.clamp(max=max_kl_loss)
        else:
            teacher_kl_for_loss = teacher_kl

        return teacher_kl_for_loss, {
            "teacher_kl": teacher_kl,
            "teacher_kl_for_loss": teacher_kl_for_loss,
        }

    def _compute_additional_loss(
        self,
        batch: RolloutStorage.Batch,
        original_batch_size: int,
        distribution_params: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute frozen-teacher guidance and optional actor auxiliary losses."""
        if self.teacher_guidance_enabled:
            teacher_kl_lambda = self.get_teacher_kl_lambda()
            if not self._teacher_guidance_requires_observations():
                teacher_loss = torch.zeros((), device=self.device)
                log_dict = {
                    "teacher_loss": 0.0,
                    "teacher_loss_for_update": 0.0,
                    "teacher_lambda": 0.0,
                    "teacher_kl_lambda": 0.0,
                    "teacher_guidance_active": 0.0,
                }
            else:
                if self.teacher is None or not self.teacher_loaded:
                    raise RuntimeError("Teacher KL loss requires a loaded teacher model.")
                if batch.observations is None:
                    raise RuntimeError("Teacher KL loss requires observations in the rollout batch.")
                teacher_loss, log_dict = self._compute_teacher_guidance_loss(
                    batch,
                    original_batch_size,
                    distribution_params,
                    loss_weight=teacher_kl_lambda,
                )
        else:
            teacher_loss = torch.zeros((), device=self.device)
            log_dict = {
                "teacher_loss": 0.0,
                "teacher_loss_for_update": 0.0,
                "teacher_lambda": 0.0,
                "teacher_kl_lambda": 0.0,
                "teacher_guidance_enabled": 0.0,
            }

        if self._actor_has_slow_latent_aux():
            aux_loss, aux_logs = self._compute_slow_latent_aux_loss(batch)
        else:
            aux_loss = torch.zeros((), device=self.device)
            aux_logs = {}
        log_dict.update(aux_logs)
        return teacher_loss + aux_loss, log_dict

    def _teacher_guidance_requires_observations(self) -> bool:
        """Return whether the next update still needs frozen-teacher observations."""
        if not self.teacher_guidance_enabled:
            return False
        if self.teacher_imitation_only:
            return True
        if self.get_teacher_kl_lambda() != 0.0:
            return True
        return bool(self.teacher_kl_cfg.get("log_kl_when_lambda_zero", True))

    @staticmethod
    def _collect_model_observation_groups(model: object, groups: set[str]) -> None:
        obs_groups = getattr(model, "obs_groups", ())
        groups.update(str(group_name) for group_name in obs_groups)
        latent_obs_groups = getattr(model, "_latent_obs_group_names", ())
        groups.update(str(group_name) for group_name in latent_obs_groups)

    def get_required_observation_groups(self) -> tuple[str, ...]:
        """Return observation groups needed by the next rollout/update cycle."""
        groups: set[str] = set()
        self._collect_model_observation_groups(self.actor, groups)
        self._collect_model_observation_groups(self.critic, groups)

        rnd_obs_groups = getattr(self.rnd, "obs_groups", None)
        if isinstance(rnd_obs_groups, dict):
            for group_names in rnd_obs_groups.values():
                groups.update(str(group_name) for group_name in group_names)
        elif rnd_obs_groups is not None:
            groups.update(str(group_name) for group_name in rnd_obs_groups)

        if self._actor_has_slow_latent_aux():
            groups.add("latent_labels")
        if self._teacher_guidance_requires_observations() and self.teacher is not None:
            self._collect_model_observation_groups(self.teacher, groups)
        return tuple(sorted(groups))

    def _actor_has_slow_latent_aux(self) -> bool:
        """Return whether the actor exposes slow-latent auxiliary heads."""
        return callable(getattr(self.actor, "get_aux_outputs", None)) or callable(
            getattr(self.actor, "get_slow_latent_diagnostics", None)
        )

    @staticmethod
    def _hidden_state_shapes(
        batch: RolloutStorage.Batch,
    ) -> list[tuple[int, ...] | str]:
        """Summarize actor hidden-state shapes without assuming recurrence."""
        if batch.hidden_states is None or batch.hidden_states[0] is None:
            return []
        actor_hidden = batch.hidden_states[0]
        hidden_items = actor_hidden if isinstance(actor_hidden, tuple) else (actor_hidden,)
        return [tuple(h.shape) if hasattr(h, "shape") else type(h).__name__ for h in hidden_items]

    def _compute_future_max_labels(
        self,
        current_labels: torch.Tensor,
        masks: torch.Tensor | None,
        horizon: int,
    ) -> torch.Tensor:
        """Compute max(label[t+1:t+K+1]) within padded trajectories."""
        if horizon <= 0:
            return torch.zeros_like(current_labels)
        if current_labels.dim() < 3:
            return torch.zeros_like(current_labels)

        valid_labels = current_labels
        if masks is not None:
            valid_labels = valid_labels * masks.unsqueeze(-1).to(valid_labels.dtype)

        future = torch.zeros_like(valid_labels)
        seq_len = valid_labels.shape[0]
        for offset in range(1, min(horizon, seq_len - 1) + 1):
            shifted = torch.zeros_like(valid_labels)
            shifted[:-offset] = valid_labels[offset:]
            future = torch.maximum(future, shifted)
        if masks is not None:
            future = future * masks.unsqueeze(-1).to(future.dtype)
        return future

    def _compute_future_first_touchdown_quality(
        self,
        touchdown: torch.Tensor,
        quality: torch.Tensor,
        masks: torch.Tensor | None,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return quality of the first touchdown in the next K steps."""
        if horizon <= 0 or touchdown.dim() < 3:
            zeros = torch.zeros_like(quality)
            return zeros, zeros
        if masks is not None:
            valid = masks.unsqueeze(-1)
            touchdown = touchdown * valid.to(touchdown.dtype)
            quality = quality * valid.to(quality.dtype)

        future_quality = torch.zeros_like(quality)
        found = torch.zeros_like(touchdown, dtype=torch.bool)
        seq_len = touchdown.shape[0]
        for offset in range(1, min(horizon, seq_len - 1) + 1):
            shifted_touchdown = torch.zeros_like(touchdown, dtype=torch.bool)
            shifted_quality = torch.zeros_like(quality)
            shifted_touchdown[:-offset] = touchdown[offset:] > 0.5
            shifted_quality[:-offset] = quality[offset:]
            take = shifted_touchdown & ~found
            future_quality = torch.where(take, shifted_quality, future_quality)
            found |= shifted_touchdown
        if masks is not None:
            valid = masks.unsqueeze(-1)
            future_quality = future_quality * valid.to(future_quality.dtype)
            found &= valid.bool()
        return future_quality, found.float()

    @staticmethod
    def _expand_event_labels(
        labels: torch.Tensor,
        masks: torch.Tensor | None,
        window_steps: int,
    ) -> torch.Tensor:
        """Keep sparse one-frame event labels positive for a short forward window."""
        if window_steps <= 1 or labels.dim() < 3:
            return labels

        valid_labels = labels
        if masks is not None:
            valid_labels = valid_labels * masks.unsqueeze(-1).to(valid_labels.dtype)

        expanded = valid_labels.clone()
        seq_len = labels.shape[0]
        for offset in range(1, min(window_steps, seq_len)):
            shifted = torch.zeros_like(valid_labels)
            shifted[offset:] = valid_labels[:-offset]
            expanded = torch.maximum(expanded, shifted)
        if masks is not None:
            expanded = expanded * masks.unsqueeze(-1).to(expanded.dtype)
        return expanded

    @staticmethod
    def _compute_weighted_bounded_huber(
        logits: torch.Tensor,
        labels: torch.Tensor,
        weight_scale: float,
        huber_delta: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predictions = torch.sigmoid(logits)
        labels = torch.nan_to_num(labels).clamp(0.0, 1.0)
        weights = 1.0 + weight_scale * labels
        error = functional.smooth_l1_loss(
            predictions,
            labels,
            reduction="none",
            beta=huber_delta,
        )
        return (weights * error).mean(), torch.abs(predictions - labels).mean()

    @staticmethod
    def _compute_stair_shape_loss(
        predictions: torch.Tensor,
        labels: torch.Tensor,
        valid: torch.Tensor,
        huber_delta: float,
    ) -> torch.Tensor:
        """Compute a valid-mask-normalized Huber loss for stair geometry."""
        finite_valid = torch.isfinite(predictions) & torch.isfinite(labels)
        predictions = torch.nan_to_num(predictions)
        labels = torch.nan_to_num(labels)
        shape_error = functional.smooth_l1_loss(
            predictions,
            labels,
            reduction="none",
            beta=huber_delta,
        )
        component_valid = valid.expand_as(shape_error) * finite_valid.to(shape_error.dtype)
        valid_count = component_valid.sum().clamp_min(1.0)
        return (shape_error * component_valid).sum() / valid_count

    @staticmethod
    def _compute_safe_stride_loss(
        predictions: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        valid: torch.Tensor,
        importance: torch.Tensor,
        huber_delta: float,
        lower_shortfall_coef: float = 1.0,
    ) -> torch.Tensor:
        """Penalize predictions only when they leave the safe-stride interval."""
        finite_valid = torch.isfinite(predictions) & torch.isfinite(lower_bounds) & torch.isfinite(upper_bounds)
        predictions = torch.nan_to_num(predictions)
        lower_bounds = torch.nan_to_num(lower_bounds)
        upper_bounds = torch.nan_to_num(upper_bounds)
        interval_target = torch.minimum(
            torch.maximum(predictions, lower_bounds),
            upper_bounds,
        )
        error = functional.smooth_l1_loss(
            predictions,
            interval_target,
            reduction="none",
            beta=huber_delta,
        )
        below_lower = predictions < lower_bounds
        shortfall_weight = torch.where(
            below_lower,
            error.new_tensor(lower_shortfall_coef),
            torch.ones_like(error),
        )
        weighted_valid = (
            torch.nan_to_num(valid).clamp_min(0.0)
            * torch.nan_to_num(importance, nan=1.0).clamp_min(1.0)
            * finite_valid.to(error.dtype)
        )
        weighted_count = weighted_valid.sum().clamp_min(1.0)
        return (error * shortfall_weight * weighted_valid).sum() / weighted_count

    @staticmethod
    def _compute_safe_stride_interval_loss(
        predictions: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        lower_valid: torch.Tensor,
        interval_valid: torch.Tensor,
        importance: torch.Tensor,
        huber_delta: float,
        width_loss_coef: float = 1.0,
        lower_shortfall_coef: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Regress lower and absolute width with independent normalization."""
        predicted_lower = predictions[..., 0:1]
        predicted_width = predictions[..., 1:2] - predicted_lower
        finite_lower_valid = torch.isfinite(predicted_lower) & torch.isfinite(lower_bounds)
        finite_width_valid = (
            torch.isfinite(predicted_lower)
            & torch.isfinite(predicted_width)
            & torch.isfinite(lower_bounds)
            & torch.isfinite(upper_bounds)
        )
        predicted_lower = torch.nan_to_num(predicted_lower)
        predicted_width = torch.nan_to_num(predicted_width)
        lower_bounds = torch.nan_to_num(lower_bounds)
        upper_bounds = torch.nan_to_num(upper_bounds)
        target_width = upper_bounds - lower_bounds
        lower_error = functional.smooth_l1_loss(
            predicted_lower,
            lower_bounds,
            reduction="none",
            beta=huber_delta,
        )
        lower_shortfall_weight = torch.where(
            predicted_lower < lower_bounds,
            lower_error.new_tensor(lower_shortfall_coef),
            torch.ones_like(lower_error),
        )
        width_error = functional.smooth_l1_loss(
            predicted_width,
            target_width,
            reduction="none",
            beta=huber_delta,
        )
        importance = torch.nan_to_num(importance, nan=1.0).clamp_min(1.0)
        lower_weight = (
            torch.nan_to_num(lower_valid).clamp_min(0.0) * importance * finite_lower_valid.to(lower_error.dtype)
        )
        width_weight = (
            torch.nan_to_num(interval_valid).clamp_min(0.0) * importance * finite_width_valid.to(width_error.dtype)
        )
        lower_loss = (lower_error * lower_shortfall_weight * lower_weight).sum() / lower_weight.sum().clamp_min(1.0)
        width_loss = (width_error * width_weight).sum() / width_weight.sum().clamp_min(1.0)
        total_loss = lower_loss + width_loss_coef * width_loss
        return total_loss, lower_loss, width_loss

    @staticmethod
    def _compute_safe_stride_confidence_loss(
        logits: torch.Tensor,
        target: torch.Tensor,
        importance: torch.Tensor,
    ) -> torch.Tensor:
        """Balanced BCE for whether the decoded SafeStride interval is usable."""
        logits = torch.nan_to_num(logits)
        target = torch.nan_to_num(target).clamp(0.0, 1.0)
        importance = torch.nan_to_num(importance, nan=1.0).clamp_min(1.0)
        error = functional.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
        positive = target > 0.5
        negative = ~positive
        positive_weight = positive.to(error.dtype) * importance.clamp_min(1.0)
        negative_weight = negative.to(error.dtype)
        positive_loss = (error * positive_weight).sum() / positive_weight.sum().clamp_min(1.0)
        negative_loss = (error * negative_weight).sum() / negative_weight.sum().clamp_min(1.0)
        return 0.5 * (positive_loss + negative_loss)

    @staticmethod
    def _compute_masked_centered_spread_losses(
        predictions: torch.Tensor,
        labels: torch.Tensor,
        valid: torch.Tensor,
        std_floor_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return centered-regression and std-floor losses for valid samples."""
        component_valid = valid.expand_as(predictions)
        finite_valid = (component_valid > 0.0) & torch.isfinite(predictions) & torch.isfinite(labels)
        component_valid = component_valid * finite_valid.to(component_valid.dtype)
        predictions = torch.where(
            finite_valid,
            torch.nan_to_num(predictions),
            torch.zeros_like(predictions),
        )
        labels = torch.where(
            finite_valid,
            torch.nan_to_num(labels),
            torch.zeros_like(labels),
        )
        reduce_dims = tuple(range(predictions.dim() - 1))
        valid_count = component_valid.sum(dim=reduce_dims).clamp_min(1.0)
        prediction_mean = (predictions * component_valid).sum(dim=reduce_dims) / valid_count
        label_mean = (labels * component_valid).sum(dim=reduce_dims) / valid_count
        prediction_centered = predictions - prediction_mean
        label_centered = labels - label_mean
        prediction_variance = (prediction_centered.square() * component_valid).sum(dim=reduce_dims) / valid_count
        label_variance = (label_centered.square() * component_valid).sum(dim=reduce_dims) / valid_count
        prediction_std = torch.sqrt(prediction_variance.clamp_min(0.0))
        label_std = torch.sqrt(label_variance.clamp_min(0.0))
        usable = (valid_count > 1.0) & (label_std > 1.0e-4)
        label_scale = label_std.clamp_min(1.0e-4)
        normalized_prediction = prediction_centered / label_scale
        normalized_label = label_centered / label_scale
        centered_error = functional.smooth_l1_loss(
            normalized_prediction,
            normalized_label,
            reduction="none",
            beta=1.0,
        )
        centered_loss = (centered_error * component_valid).sum(dim=reduce_dims) / valid_count
        std_shortfall = torch.relu(label_std * float(std_floor_ratio) - prediction_std)
        std_floor_loss = (std_shortfall / label_scale).square()
        zeros = torch.zeros_like(centered_loss)
        centered_loss = torch.where(usable, centered_loss, zeros)
        std_floor_loss = torch.where(usable, std_floor_loss, zeros)
        return centered_loss.mean(), std_floor_loss.mean()

    @staticmethod
    def _compute_one_sided_lower_hint_loss(
        predictions: torch.Tensor,
        lower_hint: torch.Tensor,
        valid: torch.Tensor,
        importance: torch.Tensor,
        margin: float,
        huber_delta: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalize lower-bound predictions only when below deployable hints."""
        weighted_valid = torch.nan_to_num(valid).clamp_min(0.0) * torch.nan_to_num(
            importance,
            nan=1.0,
        ).clamp_min(1.0)
        finite_valid = (weighted_valid > 0.0) & torch.isfinite(predictions) & torch.isfinite(lower_hint)
        weighted_valid = weighted_valid * finite_valid.to(weighted_valid.dtype)
        predictions = torch.where(
            finite_valid,
            torch.nan_to_num(predictions),
            torch.zeros_like(predictions),
        )
        lower_hint = torch.where(
            finite_valid,
            torch.nan_to_num(lower_hint),
            torch.zeros_like(lower_hint),
        )
        hint_target = lower_hint - lower_hint.new_tensor(float(margin))
        shortfall = torch.relu(hint_target - predictions)
        error = functional.smooth_l1_loss(
            predictions,
            hint_target,
            reduction="none",
            beta=huber_delta,
        )
        active = (shortfall > 0.0).to(error.dtype) * weighted_valid
        active_count = active.sum().clamp_min(1.0)
        loss = (error * active).sum() / active_count
        shortfall_mae = (shortfall * weighted_valid).sum() / weighted_valid.sum().clamp_min(1.0)
        return loss, shortfall_mae

    @staticmethod
    def _compute_safe_stride_interval_coverage_loss(
        predictions: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        valid: torch.Tensor,
        importance: torch.Tensor,
        margin: float,
        huber_delta: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Penalize decoded intervals that do not cover the target interval."""
        predicted_lower = predictions[..., 0:1]
        predicted_upper = predictions[..., 1:2]
        finite_valid = (
            torch.isfinite(predicted_lower)
            & torch.isfinite(predicted_upper)
            & torch.isfinite(lower_bounds)
            & torch.isfinite(upper_bounds)
        )
        predicted_lower = torch.nan_to_num(predicted_lower)
        predicted_upper = torch.nan_to_num(predicted_upper)
        lower_bounds = torch.nan_to_num(lower_bounds)
        upper_bounds = torch.nan_to_num(upper_bounds)
        weighted_valid = (
            torch.nan_to_num(valid).clamp_min(0.0)
            * torch.nan_to_num(importance, nan=1.0).clamp_min(1.0)
            * finite_valid.to(predicted_lower.dtype)
        )
        lower_target = lower_bounds + lower_bounds.new_tensor(float(margin))
        upper_target = upper_bounds - upper_bounds.new_tensor(float(margin))
        lower_over = torch.relu(predicted_lower - lower_target)
        upper_shortfall = torch.relu(upper_target - predicted_upper)
        lower_error = functional.smooth_l1_loss(
            predicted_lower,
            lower_target,
            reduction="none",
            beta=huber_delta,
        )
        upper_error = functional.smooth_l1_loss(
            predicted_upper,
            upper_target,
            reduction="none",
            beta=huber_delta,
        )
        lower_active = (lower_over > 0.0).to(weighted_valid.dtype) * weighted_valid
        upper_active = (upper_shortfall > 0.0).to(weighted_valid.dtype) * weighted_valid
        lower_loss = (lower_error * lower_active).sum() / lower_active.sum().clamp_min(1.0)
        upper_loss = (upper_error * upper_active).sum() / upper_active.sum().clamp_min(1.0)
        lower_over_mae = (lower_over * weighted_valid).sum() / weighted_valid.sum().clamp_min(1.0)
        upper_shortfall_mae = (upper_shortfall * weighted_valid).sum() / weighted_valid.sum().clamp_min(1.0)
        return 0.5 * (lower_loss + upper_loss), lower_over_mae, upper_shortfall_mae

    @staticmethod
    def _safe_stride_deployable_hint_from_latent_obs(
        observations: TensorDict | None,
        latent_obs_key: str,
        safe_stride_min: float,
        safe_stride_max: float,
    ) -> torch.Tensor | None:
        """Extract a one-sided lower SafeStride hint from foot-event summary obs."""
        if observations is None or latent_obs_key not in observations:
            return None
        latent_obs = observations[latent_obs_key]
        latent_dim = int(latent_obs.shape[-1])
        if latent_dim < 70:
            return None
        summary_dim = 70
        if latent_dim >= 80:
            new_foot_only = (latent_dim - 80) % 33 == 0
            new_with_stair = latent_dim >= 173 and (latent_dim - 173) % 33 == 0
            if latent_dim == 80 or new_foot_only or new_with_stair:
                summary_dim = 80
        summary = torch.nan_to_num(latent_obs[..., -summary_dim:])
        toe = summary[..., 50:60]
        stats = summary[..., 60:70]
        forward_up_stride = stats[..., 8:9].clamp_min(0.0)
        toe_relation_valid = toe[..., 6:7] > 0.5
        toe_forward_delta = torch.where(
            toe_relation_valid,
            toe[..., 7:8].clamp_min(0.0),
            torch.zeros_like(toe[..., 7:8]),
        )
        ratchet_hint = torch.zeros_like(forward_up_stride)
        if summary_dim >= 80:
            ratchet = summary[..., 70:80]
            ratchet_active = ratchet[..., 0:1] > 0.5
            ratchet_confident = ratchet[..., 8:9] > 0.05
            ratchet_lower = ratchet[..., 1:2].clamp_min(0.0)
            ratchet_probe = ratchet[..., 2:3].clamp_min(0.0)
            ratchet_hint = torch.where(
                ratchet_active & ratchet_confident,
                torch.maximum(ratchet_lower, ratchet_probe),
                ratchet_hint,
            )
        hint = torch.maximum(torch.maximum(forward_up_stride, toe_forward_delta), ratchet_hint)
        hint_valid = hint > 1.0e-5
        hint = hint.clamp(float(safe_stride_min), float(safe_stride_max))
        return torch.cat([hint, hint_valid.to(hint.dtype)], dim=-1)

    @staticmethod
    def _compute_normalized_stair_shape_loss(
        predictions: torch.Tensor,
        labels: torch.Tensor,
        valid: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        huber_delta: float,
    ) -> torch.Tensor:
        """Compute masked geometry Huber loss after physical-range normalization."""
        value_ranges = (upper_bounds - lower_bounds).clamp_min(1.0e-6)
        normalized_predictions = (predictions - lower_bounds) / value_ranges
        normalized_labels = (labels - lower_bounds) / value_ranges
        return PPOTeacherKL._compute_stair_shape_loss(
            normalized_predictions,
            normalized_labels,
            valid,
            huber_delta,
        )

    @staticmethod
    def _compute_stair_shape_component_errors(
        predictions: torch.Tensor,
        labels: torch.Tensor,
        valid: torch.Tensor,
        huber_delta: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute valid-mask-normalized MAE and Huber for each shape component."""
        finite_valid = torch.isfinite(predictions) & torch.isfinite(labels)
        predictions = torch.nan_to_num(predictions)
        labels = torch.nan_to_num(labels)
        absolute_error = torch.abs(predictions - labels)
        component_valid = valid.expand_as(absolute_error) * finite_valid.to(absolute_error.dtype)
        huber_error = functional.smooth_l1_loss(
            predictions,
            labels,
            reduction="none",
            beta=huber_delta,
        )
        reduce_dims = tuple(range(predictions.dim() - 1))
        valid_count = component_valid.sum(dim=reduce_dims).clamp_min(1.0)
        component_mae = (absolute_error * component_valid).sum(dim=reduce_dims) / valid_count
        component_huber = (huber_error * component_valid).sum(dim=reduce_dims) / valid_count
        return component_mae, component_huber

    @staticmethod
    def _compute_masked_regression_statistics(
        predictions: torch.Tensor,
        labels: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-component label/prediction spread, correlation, and R²."""
        finite_valid = torch.isfinite(predictions) & torch.isfinite(labels)
        predictions = torch.nan_to_num(predictions)
        labels = torch.nan_to_num(labels)
        component_valid = valid.expand_as(predictions) * finite_valid.to(predictions.dtype)
        reduce_dims = tuple(range(predictions.dim() - 1))
        valid_count = component_valid.sum(dim=reduce_dims).clamp_min(1.0)
        prediction_mean = (predictions * component_valid).sum(dim=reduce_dims) / valid_count
        label_mean = (labels * component_valid).sum(dim=reduce_dims) / valid_count
        prediction_centered = predictions - prediction_mean
        label_centered = labels - label_mean
        prediction_squares = prediction_centered.square() * component_valid
        prediction_variance = prediction_squares.sum(dim=reduce_dims)
        prediction_variance = prediction_variance / valid_count
        label_variance = (label_centered.square() * component_valid).sum(dim=reduce_dims)
        label_variance = label_variance / valid_count
        covariance = (prediction_centered * label_centered * component_valid).sum(dim=reduce_dims)
        covariance = covariance / valid_count
        correlation_scale = torch.sqrt(prediction_variance * label_variance)
        correlation = torch.where(
            correlation_scale > 1.0e-12,
            covariance / correlation_scale.clamp_min(1.0e-12),
            torch.zeros_like(covariance),
        )
        squared_error = ((predictions - labels).square() * component_valid).sum(dim=reduce_dims)
        label_squared_deviation = (label_centered.square() * component_valid).sum(dim=reduce_dims)
        r_squared = torch.where(
            label_squared_deviation > 1.0e-12,
            1.0 - squared_error / label_squared_deviation.clamp_min(1.0e-12),
            torch.zeros_like(squared_error),
        )
        return (
            torch.sqrt(label_variance),
            torch.sqrt(prediction_variance),
            correlation,
            r_squared,
        )

    def _accumulate_safe_stride_update_statistics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        valid: torch.Tensor,
        component: str = "center",
    ) -> None:
        """Accumulate sufficient statistics across all update minibatches."""
        if self._safe_stride_update_statistics is None:
            return
        prediction_raw = predictions.detach()[..., 0]
        label_raw = labels.detach()[..., 0]
        weight = torch.nan_to_num(valid.detach()[..., 0]).clamp_min(0.0)
        finite_weight = torch.isfinite(prediction_raw) & torch.isfinite(label_raw)
        weight = weight * finite_weight.to(weight.dtype)
        prediction = torch.nan_to_num(prediction_raw)
        label = torch.nan_to_num(label_raw)
        error = prediction - label
        moments = {
            "count": weight.sum(),
            "label_sum": (label * weight).sum(),
            "prediction_sum": (prediction * weight).sum(),
            "label_square_sum": (label.square() * weight).sum(),
            "prediction_square_sum": (prediction.square() * weight).sum(),
            "cross_sum": (label * prediction * weight).sum(),
            "absolute_error_sum": (error.abs() * weight).sum(),
            "signed_error_sum": (error * weight).sum(),
            "squared_error_sum": (error.square() * weight).sum(),
        }
        for name, value in moments.items():
            key = f"{component}_{name}"
            previous = self._safe_stride_update_statistics.get(key)
            if previous is None:
                self._safe_stride_update_statistics[key] = value
            else:
                self._safe_stride_update_statistics[key] = previous + value

    def _finalize_safe_stride_update_statistics(self) -> dict[str, float]:
        """Return update-global SafeStride statistics from sufficient moments."""
        statistics = self._safe_stride_update_statistics
        if not statistics:
            return {}
        reduced = {name: value.detach().clone() for name, value in statistics.items()}
        if self.is_multi_gpu:
            distributed = cast(Any, torch.distributed)
            for value in reduced.values():
                distributed.all_reduce(value, op=distributed.ReduceOp.SUM)

        logs: dict[str, float] = {}
        for component in ("center", "lower", "lower_interval", "upper", "width"):
            count_key = f"{component}_count"
            if count_key not in reduced or reduced[count_key].item() <= 0.0:
                continue
            count = reduced[count_key]
            label_mean = reduced[f"{component}_label_sum"] / count
            prediction_mean = reduced[f"{component}_prediction_sum"] / count
            label_variance = (reduced[f"{component}_label_square_sum"] / count - label_mean.square()).clamp_min(0.0)
            prediction_variance = (
                reduced[f"{component}_prediction_square_sum"] / count - prediction_mean.square()
            ).clamp_min(0.0)
            covariance = reduced[f"{component}_cross_sum"] / count - label_mean * prediction_mean
            correlation_scale = torch.sqrt(label_variance * prediction_variance)
            correlation = torch.where(
                correlation_scale > 1.0e-12,
                covariance / correlation_scale.clamp_min(1.0e-12),
                torch.zeros_like(covariance),
            )
            label_squared_deviation = label_variance * count
            r_squared = torch.where(
                label_squared_deviation > 1.0e-12,
                1.0 - reduced[f"{component}_squared_error_sum"] / label_squared_deviation,
                torch.zeros_like(label_squared_deviation),
            )
            prefix = "slow_latent_safe_stride_global"
            if component != "center":
                prefix = f"{prefix}_{component}"
            logs[f"{prefix}_valid_count"] = (count / max(self.num_learning_epochs, 1)).item()
            logs[f"{prefix}_label_mean"] = label_mean.item()
            logs[f"{prefix}_pred_mean"] = prediction_mean.item()
            logs[f"{prefix}_label_std"] = torch.sqrt(label_variance).item()
            logs[f"{prefix}_pred_std"] = torch.sqrt(prediction_variance).item()
            logs[f"{prefix}_correlation"] = correlation.item()
            logs[f"{prefix}_r_squared"] = r_squared.item()
            logs[f"{prefix}_mae"] = (reduced[f"{component}_absolute_error_sum"] / count).item()
            logs[f"{prefix}_signed_error"] = (reduced[f"{component}_signed_error_sum"] / count).item()
            if component == "width":
                logs["slow_latent_safe_stride_global_interval_valid_frame_count"] = (
                    count / max(self.num_learning_epochs, 1)
                ).item()
        depth_confirmation_key = "depth_confirmation_event_count"
        if depth_confirmation_key in reduced:
            logs["slow_latent_safe_stride_global_unique_depth_confirmation_count"] = (
                reduced[depth_confirmation_key] / max(self.num_learning_epochs, 1)
            ).item()
        return logs

    def _accumulate_safe_stride_depth_confirmation_events(self, events: torch.Tensor) -> None:
        """Count one-frame depth confirmations once per update epoch."""
        if self._safe_stride_update_statistics is None:
            return
        key = "depth_confirmation_event_count"
        value = events.detach().sum()
        previous = self._safe_stride_update_statistics.get(key)
        if previous is None:
            self._safe_stride_update_statistics[key] = value
        else:
            self._safe_stride_update_statistics[key] = previous + value

    def _accumulate_geometry_probe_component(
        self,
        prediction: torch.Tensor,
        label: torch.Tensor,
        valid: torch.Tensor,
        cohort: str,
    ) -> None:
        """Accumulate exact sufficient moments for one geometry cohort."""
        if self._geometry_probe_update_statistics is None:
            return
        # Discrete depth cohorts often contain one exact label value. Float64
        # keeps their zero variance from becoming a large cancellation error.
        prediction_raw = prediction.detach()
        label_raw = label.detach()
        weight = torch.nan_to_num(valid.detach()).clamp_min(0.0).double()
        finite_weight = torch.isfinite(prediction_raw) & torch.isfinite(label_raw)
        weight = weight * finite_weight.to(weight.dtype)
        prediction_flat = torch.nan_to_num(prediction_raw).double()
        label_flat = torch.nan_to_num(label_raw).double()
        error = prediction_flat - label_flat
        moments = {
            "count": weight.sum(),
            "label_sum": (label_flat * weight).sum(),
            "prediction_sum": (prediction_flat * weight).sum(),
            "label_square_sum": (label_flat.square() * weight).sum(),
            "prediction_square_sum": (prediction_flat.square() * weight).sum(),
            "cross_sum": (label_flat * prediction_flat * weight).sum(),
            "absolute_error_sum": (error.abs() * weight).sum(),
            "signed_error_sum": (error * weight).sum(),
            "squared_error_sum": (error.square() * weight).sum(),
        }
        for name, value in moments.items():
            key = f"{cohort}_{name}"
            previous = self._geometry_probe_update_statistics.get(key)
            if previous is None:
                self._geometry_probe_update_statistics[key] = value
            else:
                self._geometry_probe_update_statistics[key] = previous + value

    def _accumulate_geometry_probe_statistics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        component_valid: torch.Tensor,
        validation: torch.Tensor,
        confirmation_age: torch.Tensor,
    ) -> None:
        """Accumulate train/validation, age, and depth-bin Probe statistics."""
        train = 1.0 - validation
        split_masks = {"train": train, "validation": validation}
        component_names = ("depth", "height")
        for component_index, component_name in enumerate(component_names):
            for split_name, split_mask in split_masks.items():
                valid = component_valid[..., component_index] * split_mask[..., 0]
                self._accumulate_geometry_probe_component(
                    predictions[..., component_index],
                    labels[..., component_index],
                    valid,
                    f"{split_name}_{component_name}",
                )

        validation_depth = component_valid[..., 0] * validation[..., 0]
        age = confirmation_age[..., 0]
        age_masks = {
            "age_0": age == 0,
            "age_1_4": (age >= 1) & (age <= 4),
            "age_5_16": (age >= 5) & (age <= 16),
            "age_17_plus": age >= 17,
        }
        for age_name, age_mask in age_masks.items():
            self._accumulate_geometry_probe_component(
                predictions[..., 0],
                labels[..., 0],
                validation_depth * age_mask.to(validation_depth.dtype),
                f"validation_depth_{age_name}",
            )

        depth_bin_width = labels.new_tensor((0.35 - 0.25) / 7.0)
        depth_bin = torch.round((labels[..., 0] - 0.25) / depth_bin_width)
        depth_bin = depth_bin.long().clamp(0, 7)
        for bin_index in range(8):
            self._accumulate_geometry_probe_component(
                predictions[..., 0],
                labels[..., 0],
                validation_depth * (depth_bin == bin_index).to(validation_depth.dtype),
                f"validation_depth_bin_{bin_index}",
            )

    def _finalize_geometry_probe_statistics(self) -> dict[str, float]:
        """Return update-global held-out geometry Probe metrics."""
        statistics = self._geometry_probe_update_statistics
        if not statistics:
            return {}
        reduced = {name: value.detach().clone() for name, value in statistics.items()}
        if self.is_multi_gpu:
            distributed = cast(Any, torch.distributed)
            for value in reduced.values():
                distributed.all_reduce(value, op=distributed.ReduceOp.SUM)

        cohorts = sorted(key.removesuffix("_count") for key in reduced if key.endswith("_count"))
        logs: dict[str, float] = {}
        for cohort in cohorts:
            count = reduced[f"{cohort}_count"]
            if count.item() <= 0.0:
                continue
            label_mean = reduced[f"{cohort}_label_sum"] / count
            prediction_mean = reduced[f"{cohort}_prediction_sum"] / count
            label_variance = (reduced[f"{cohort}_label_square_sum"] / count - label_mean.square()).clamp_min(0.0)
            prediction_variance = (
                reduced[f"{cohort}_prediction_square_sum"] / count - prediction_mean.square()
            ).clamp_min(0.0)
            covariance = reduced[f"{cohort}_cross_sum"] / count - label_mean * prediction_mean
            correlation_valid = (label_variance > 1.0e-12) & (prediction_variance > 1.0e-12)
            correlation = torch.where(
                correlation_valid,
                covariance / torch.sqrt(label_variance * prediction_variance).clamp_min(1.0e-12),
                torch.zeros_like(covariance),
            )
            label_squared_deviation = label_variance * count
            r_squared = torch.where(
                label_squared_deviation > count * 1.0e-12,
                1.0 - reduced[f"{cohort}_squared_error_sum"] / label_squared_deviation.clamp_min(1.0e-12),
                torch.zeros_like(label_squared_deviation),
            )
            prefix = f"slow_latent_geometry_probe_global_{cohort}"
            logs[f"{prefix}_valid_count"] = (count / max(self.num_learning_epochs, 1)).item()
            logs[f"{prefix}_label_mean"] = label_mean.item()
            logs[f"{prefix}_pred_mean"] = prediction_mean.item()
            logs[f"{prefix}_label_std"] = torch.sqrt(label_variance).item()
            logs[f"{prefix}_pred_std"] = torch.sqrt(prediction_variance).item()
            logs[f"{prefix}_correlation"] = correlation.item()
            logs[f"{prefix}_r_squared"] = r_squared.item()
            logs[f"{prefix}_mae"] = (reduced[f"{cohort}_absolute_error_sum"] / count).item()
            logs[f"{prefix}_signed_error"] = (reduced[f"{cohort}_signed_error_sum"] / count).item()
        return logs

    @staticmethod
    def _compute_label_out_of_range_ratios(
        labels: torch.Tensor,
        valid: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
    ) -> torch.Tensor:
        """Return valid-mask-normalized component ratios without altering labels."""
        out_of_range = (labels < lower_bounds) | (labels > upper_bounds)
        component_valid = valid.expand_as(labels)
        reduce_dims = tuple(range(labels.dim() - 1))
        valid_count = component_valid.sum(dim=reduce_dims).clamp_min(1.0)
        return (out_of_range.to(labels.dtype) * component_valid).sum(dim=reduce_dims) / valid_count

    def _add_slow_latent_phase_alignment_logs(
        self,
        logs: dict[str, float],
        diagnostics: dict[str, torch.Tensor],
        event_labels: torch.Tensor,
        stair_labels: torch.Tensor,
        dones: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> None:
        """Log mismatch between env stair labels and latent gate state."""
        mode = diagnostics.get("gate_mode")
        if mode is None or mode.numel() == 0 or mode.numel() != stair_labels.numel():
            return

        mode_bool_shape = mode.reshape(stair_labels.shape)
        env_stair = stair_labels > 0.5
        latent_normal = mode_bool_shape == 0.0
        latent_write = mode_bool_shape == 1.0
        latent_memory = mode_bool_shape == 2.0
        event_positive = event_labels.reshape(stair_labels.shape) > 0.5

        logs["slow_latent_memory_while_env_normal_ratio"] = self._distributed_mean_scalar(
            (latent_memory & ~env_stair).float().mean()
        ).item()
        logs["slow_latent_write_while_env_normal_ratio"] = self._distributed_mean_scalar(
            (latent_write & ~env_stair).float().mean()
        ).item()
        logs["slow_latent_env_stair_while_latent_normal_ratio"] = self._distributed_mean_scalar(
            (env_stair & latent_normal).float().mean()
        ).item()
        logs["slow_latent_memory_while_env_stair_ratio"] = self._distributed_mean_scalar(
            (latent_memory & env_stair).float().mean()
        ).item()
        event_count = event_positive.float().sum().clamp_min(1.0)
        logs["slow_latent_event_with_stair_label_rate"] = self._distributed_mean_scalar(
            (event_positive & env_stair).float().sum() / event_count
        ).item()

        stair_prob = diagnostics.get("stair_prob")
        if stair_prob is not None and stair_prob.numel() == stair_labels.numel():
            stair_prob = stair_prob.reshape(stair_labels.shape)
            stair_on_threshold = float(getattr(self.actor, "stair_on_threshold", 0.35))
            stair_pred_on = stair_prob > stair_on_threshold
            write_stair = latent_write & env_stair
            write_stair_count = write_stair.float().sum().clamp_min(1.0)
            logs["slow_latent_stair_recall_while_write"] = self._distributed_mean_scalar(
                (stair_pred_on & write_stair).float().sum() / write_stair_count
            ).item()

        trigger = diagnostics.get("gate_event_trigger")
        if trigger is not None and trigger.numel() == stair_labels.numel():
            trigger = trigger.reshape(stair_labels.shape) > 0.5
            trigger_count = trigger.float().sum().clamp_min(1.0)
            normal_at_step_start = latent_normal | trigger
            normal_event = event_positive & normal_at_step_start
            normal_event_count = normal_event.float().sum().clamp_min(1.0)
            logs["slow_latent_event_while_normal_ratio"] = self._distributed_mean_scalar(
                normal_event.float().sum() / event_count
            ).item()
            logs["slow_latent_true_event_to_write_rate"] = self._distributed_mean_scalar(
                (trigger & event_positive).float().sum() / event_count
            ).item()
            logs["slow_latent_normal_event_to_write_rate"] = self._distributed_mean_scalar(
                (trigger & normal_event).float().sum() / normal_event_count
            ).item()
            logs["slow_latent_write_trigger_event_precision"] = self._distributed_mean_scalar(
                (trigger & event_positive).float().sum() / trigger_count
            ).item()

        confirm = diagnostics.get("gate_write_confirm")
        if confirm is not None and confirm.numel() == stair_labels.numel():
            confirm = confirm.reshape(stair_labels.shape) > 0.5
            confirm_count = confirm.float().sum().clamp_min(1.0)
            confirm_on_stair = confirm & env_stair
            logs["slow_latent_gate_confirm_while_env_stair_ratio"] = self._distributed_mean_scalar(
                confirm_on_stair.float().mean()
            ).item()
            logs["slow_latent_gate_confirm_stair_precision"] = self._distributed_mean_scalar(
                confirm_on_stair.float().sum() / confirm_count
            ).item()
            logs["slow_latent_true_event_to_memory_rate"] = self._distributed_mean_scalar(
                confirm_on_stair.float().sum() / event_count
            ).item()
            logs["slow_latent_false_memory_confirm_ratio"] = self._distributed_mean_scalar(
                (confirm & ~env_stair).float().sum() / confirm_count
            ).item()

        memory_exit = diagnostics.get("gate_memory_exit")
        if memory_exit is not None and memory_exit.numel() == stair_labels.numel():
            memory_exit = memory_exit.reshape(stair_labels.shape) > 0.5
            exit_count = memory_exit.float().sum().clamp_min(1.0)
            exit_on_stair = memory_exit & env_stair
            logs["slow_latent_gate_exit_while_env_stair_ratio"] = self._distributed_mean_scalar(
                exit_on_stair.float().mean()
            ).item()
            logs["slow_latent_gate_exit_stair_fraction"] = self._distributed_mean_scalar(
                exit_on_stair.float().sum() / exit_count
            ).item()

        if dones is None:
            return
        dones_float = dones.float()
        if masks is not None:
            dones_float = cast(torch.Tensor, unpad_trajectories(dones_float, masks))
        if dones_float.numel() != stair_labels.numel():
            return
        dones_bool = dones_float.reshape(stair_labels.shape) > 0.5
        logs["slow_latent_done_while_memory_ratio"] = self._distributed_mean_scalar(
            (dones_bool & latent_memory).float().mean()
        ).item()
        logs["slow_latent_done_while_env_normal_memory_ratio"] = self._distributed_mean_scalar(
            (dones_bool & ~env_stair & latent_memory).float().mean()
        ).item()

    def _compute_slow_latent_aux_loss(
        self,
        batch: RolloutStorage.Batch,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute auxiliary losses exposed by the slow-latent actor.

        Debug note:
          This function is the only place where the slow-latent event/stair/future
          auxiliary heads enter the PPO update.  The one-shot debug print below is
          intentionally broad: it checks whether the actor exposes aux outputs,
          whether the current mini-batch produced logits, whether latent labels are
          present in rollout storage, and whether shapes match.
        """
        logs: dict[str, float] = self._compute_slow_latent_diagnostic_logs()

        get_aux_outputs = getattr(self.actor, "get_aux_outputs", None)
        get_diagnostics = getattr(self.actor, "get_slow_latent_diagnostics", None)

        aux_outputs = get_aux_outputs() if get_aux_outputs is not None else {}
        diagnostics = get_diagnostics() if get_diagnostics is not None else {}

        event_coef = float(getattr(self.actor, "aux_event_coef", 0.0))
        stair_coef = float(getattr(self.actor, "aux_stair_coef", 0.0))
        future_risk_coef = float(getattr(self.actor, "aux_future_collision_risk_coef", 0.0))
        future_quality_coef = float(getattr(self.actor, "aux_future_safe_landing_quality_coef", 0.0))
        shape_coef = float(getattr(self.actor, "aux_stair_shape_coef", 0.0))
        safe_stride_coef = float(getattr(self.actor, "aux_safe_stride_coef", 0.0))

        observations = batch.observations
        obs_keys = []
        latent_labels_shape = None
        if observations is not None:
            try:
                obs_keys = list(observations.keys())
            except Exception:
                obs_keys = ["<failed_to_list_observation_keys>"]
            if "latent_labels" in observations:
                latent_labels_shape = tuple(observations["latent_labels"].shape)

        if not hasattr(self, "_printed_slow_latent_debug"):

            def _shape_dict(obj: dict | None) -> dict[str, tuple[int, ...] | str] | None:
                if not obj:
                    return obj
                out: dict[str, tuple[int, ...] | str] = {}
                for key, value in obj.items():
                    out[key] = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
                return out

            print(
                "\n[DBG slow latent aux]",
                "\n  actor_type=",
                type(self.actor),
                "\n  has_get_aux_outputs=",
                get_aux_outputs is not None,
                "\n  has_get_slow_latent_diagnostics=",
                get_diagnostics is not None,
                "\n  aux_output_shapes=",
                _shape_dict(aux_outputs),
                "\n  diagnostic_shapes=",
                _shape_dict(diagnostics),
                "\n  aux_coefs=",
                {
                    "event": event_coef,
                    "stair": stair_coef,
                    "future_risk": future_risk_coef,
                    "future_quality": future_quality_coef,
                    "shape": shape_coef,
                    "safe_stride": safe_stride_coef,
                },
                "\n  batch_has_observations=",
                observations is not None,
                "\n  observation_keys=",
                obs_keys,
                "\n  latent_labels_shape=",
                latent_labels_shape,
                "\n  masks_shape=",
                tuple(batch.masks.shape) if batch.masks is not None else None,
                "\n  hidden_state_shapes=",
                self._hidden_state_shapes(batch),
                "\n",
                flush=True,
            )
            self._printed_slow_latent_debug = True

        if get_aux_outputs is None:
            logs["slow_latent_debug_missing_get_aux_outputs"] = 1.0
            return torch.zeros((), device=self.device), logs

        if not aux_outputs:
            logs["slow_latent_debug_empty_aux_outputs"] = 1.0
            return torch.zeros((), device=self.device), logs

        if (
            event_coef == 0.0
            and stair_coef == 0.0
            and future_risk_coef == 0.0
            and future_quality_coef == 0.0
            and shape_coef == 0.0
            and safe_stride_coef == 0.0
        ):
            logs["slow_latent_debug_all_aux_coef_zero"] = 1.0
            return torch.zeros((), device=self.device), logs

        if batch.observations is None or "latent_labels" not in batch.observations:
            raise RuntimeError(
                "Slow-latent auxiliary losses require a 'latent_labels' observation group. "
                f"Available observation keys: {obs_keys}"
            )

        labels = batch.observations["latent_labels"]
        if labels.shape[-1] < 2:
            raise RuntimeError("Slow-latent 'latent_labels' must contain event and stair labels.")
        if shape_coef != 0.0 and labels.shape[-1] < 5:
            raise RuntimeError(
                "Slow-latent stair-shape loss requires labels [event, stair, tread_depth, riser_height, shape_valid]."
            )
        if safe_stride_coef != 0.0 and labels.shape[-1] < 7:
            raise RuntimeError(
                "Slow-latent safe-stride loss requires labels [event, stair, tread_depth, "
                "riser_height, shape_valid, safe_stride, safe_stride_valid]."
            )
        if (future_risk_coef != 0.0 or future_quality_coef != 0.0) and labels.shape[-1] < 10:
            raise RuntimeError(
                "Slow-latent future losses require labels [entry, stair, tread_depth, "
                "riser_height, shape_valid, safe_stride, safe_stride_valid, "
                "collision_risk, landing_touchdown, landing_quality]."
            )
        event_labels_raw_padded = labels[..., 0:1].float()
        event_label_window_steps = int(getattr(self.actor, "event_label_window_steps", 1))
        event_labels_padded = self._expand_event_labels(
            event_labels_raw_padded,
            batch.masks,
            event_label_window_steps,
        )
        stair_labels_padded = labels[..., 1:2].float()
        shape_labels_padded = labels[..., 2:4].float()
        shape_valid_padded = labels[..., 4:5].float()
        if labels.shape[-1] >= 15:
            shape_component_valid_padded = labels[..., 13:15].float()
        else:
            shape_component_valid_padded = shape_valid_padded.expand(
                *shape_valid_padded.shape[:-1],
                2,
            )
        if labels.shape[-1] >= 7:
            safe_stride_labels_padded = labels[..., 5:6].float()
            safe_stride_valid_padded = labels[..., 6:7].float()
            if labels.shape[-1] >= 11:
                safe_stride_exact_padded = labels[..., 7:8].float()
            else:
                safe_stride_exact_padded = torch.ones_like(safe_stride_valid_padded)
            if labels.shape[-1] >= 12:
                safe_stride_importance_padded = labels[..., 8:9].float()
            else:
                safe_stride_importance_padded = torch.ones_like(safe_stride_valid_padded)
            if labels.shape[-1] >= 13:
                safe_stride_upper_padded = labels[..., 9:10].float()
            else:
                safe_stride_upper_padded = safe_stride_labels_padded
        else:
            safe_stride_labels_padded = torch.zeros_like(event_labels_padded)
            safe_stride_valid_padded = torch.zeros_like(event_labels_padded)
            safe_stride_exact_padded = torch.zeros_like(event_labels_padded)
            safe_stride_importance_padded = torch.zeros_like(event_labels_padded)
            safe_stride_upper_padded = torch.zeros_like(event_labels_padded)
        if labels.shape[-1] >= 16:
            safe_stride_interval_valid_padded = labels[..., 15:16].float()
        else:
            safe_stride_interval_valid_padded = safe_stride_exact_padded
        if labels.shape[-1] >= 17:
            depth_confirmation_event_padded = labels[..., 16:17].float()
        else:
            depth_confirmation_event_padded = torch.zeros_like(event_labels_padded)
        if labels.shape[-1] >= 18:
            geometry_probe_validation_padded = labels[..., 17:18].float()
        else:
            geometry_probe_validation_padded = torch.zeros_like(event_labels_padded)
        if labels.shape[-1] >= 19:
            depth_confirmation_age_padded = labels[..., 18:19].float()
        else:
            depth_confirmation_age_padded = torch.full_like(
                event_labels_padded,
                -1.0,
            )
        if labels.shape[-1] >= 23:
            adjacent_pair_evidence_padded = labels[..., 19:23].float()
        else:
            adjacent_pair_evidence_padded = torch.zeros(
                *labels.shape[:-1],
                4,
                device=labels.device,
                dtype=labels.dtype,
            )
        deployable_hint_padded = self._safe_stride_deployable_hint_from_latent_obs(
            observations,
            str(getattr(self.actor, "latent_obs_set", "latent")),
            float(getattr(self.actor, "safe_stride_min", 0.10)),
            float(getattr(self.actor, "safe_stride_max", 0.55)),
        )
        if deployable_hint_padded is None:
            deployable_hint_padded = torch.zeros(
                *labels.shape[:-1],
                2,
                device=labels.device,
                dtype=labels.dtype,
            )
        if labels.shape[-1] >= 13:
            collision_risk_now = labels[..., 10:11].float()
            landing_touchdown_now = labels[..., 11:12].float()
            landing_quality_now = labels[..., 12:13].float()
        elif labels.shape[-1] >= 12:
            collision_risk_now = labels[..., 9:10].float()
            landing_touchdown_now = labels[..., 10:11].float()
            landing_quality_now = labels[..., 11:12].float()
        elif labels.shape[-1] >= 11:
            collision_risk_now = labels[..., 8:9].float()
            landing_touchdown_now = labels[..., 9:10].float()
            landing_quality_now = labels[..., 10:11].float()
        elif labels.shape[-1] >= 10:
            collision_risk_now = labels[..., 7:8].float()
            landing_touchdown_now = labels[..., 8:9].float()
            landing_quality_now = labels[..., 9:10].float()
        else:
            collision_risk_now = torch.zeros_like(event_labels_padded)
            landing_touchdown_now = torch.zeros_like(event_labels_padded)
            landing_quality_now = torch.zeros_like(event_labels_padded)
        future_horizon = int(getattr(self.actor, "future_horizon", 20))
        future_risk_padded = self._compute_future_max_labels(
            collision_risk_now,
            batch.masks,
            future_horizon,
        )
        future_quality_padded, future_touchdown_found_padded = self._compute_future_first_touchdown_quality(
            landing_touchdown_now,
            landing_quality_now,
            batch.masks,
            future_horizon,
        )

        if batch.masks is not None:
            event_labels_raw = cast(
                torch.Tensor,
                unpad_trajectories(event_labels_raw_padded, batch.masks),
            )
            event_labels = cast(torch.Tensor, unpad_trajectories(event_labels_padded, batch.masks))
            stair_labels = cast(torch.Tensor, unpad_trajectories(stair_labels_padded, batch.masks))
            future_risk_labels = cast(
                torch.Tensor,
                unpad_trajectories(future_risk_padded, batch.masks),
            )
            future_quality_labels = cast(
                torch.Tensor,
                unpad_trajectories(future_quality_padded, batch.masks),
            )
            future_touchdown_found = cast(
                torch.Tensor,
                unpad_trajectories(future_touchdown_found_padded, batch.masks),
            )
            shape_labels = cast(torch.Tensor, unpad_trajectories(shape_labels_padded, batch.masks))
            shape_valid = cast(torch.Tensor, unpad_trajectories(shape_valid_padded, batch.masks))
            shape_component_valid = cast(
                torch.Tensor,
                unpad_trajectories(shape_component_valid_padded, batch.masks),
            )
            safe_stride_labels = cast(
                torch.Tensor,
                unpad_trajectories(safe_stride_labels_padded, batch.masks),
            )
            safe_stride_valid = cast(
                torch.Tensor,
                unpad_trajectories(safe_stride_valid_padded, batch.masks),
            )
            safe_stride_importance = cast(
                torch.Tensor,
                unpad_trajectories(safe_stride_importance_padded, batch.masks),
            )
            safe_stride_upper = cast(
                torch.Tensor,
                unpad_trajectories(safe_stride_upper_padded, batch.masks),
            )
            safe_stride_interval_valid = cast(
                torch.Tensor,
                unpad_trajectories(
                    safe_stride_interval_valid_padded,
                    batch.masks,
                ),
            )
            depth_confirmation_event = cast(
                torch.Tensor,
                unpad_trajectories(
                    depth_confirmation_event_padded,
                    batch.masks,
                ),
            )
            geometry_probe_validation = cast(
                torch.Tensor,
                unpad_trajectories(
                    geometry_probe_validation_padded,
                    batch.masks,
                ),
            )
            depth_confirmation_age = cast(
                torch.Tensor,
                unpad_trajectories(
                    depth_confirmation_age_padded,
                    batch.masks,
                ),
            )
            adjacent_pair_evidence = cast(
                torch.Tensor,
                unpad_trajectories(
                    adjacent_pair_evidence_padded,
                    batch.masks,
                ),
            )
            deployable_hint = cast(
                torch.Tensor,
                unpad_trajectories(
                    deployable_hint_padded,
                    batch.masks,
                ),
            )
        else:
            event_labels_raw = event_labels_raw_padded
            event_labels = event_labels_padded
            stair_labels = stair_labels_padded
            future_risk_labels = future_risk_padded
            future_quality_labels = future_quality_padded
            future_touchdown_found = future_touchdown_found_padded
            shape_labels = shape_labels_padded
            shape_valid = shape_valid_padded
            shape_component_valid = shape_component_valid_padded
            safe_stride_labels = safe_stride_labels_padded
            safe_stride_valid = safe_stride_valid_padded
            safe_stride_importance = safe_stride_importance_padded
            safe_stride_upper = safe_stride_upper_padded
            safe_stride_interval_valid = safe_stride_interval_valid_padded
            depth_confirmation_event = depth_confirmation_event_padded
            geometry_probe_validation = geometry_probe_validation_padded
            depth_confirmation_age = depth_confirmation_age_padded
            adjacent_pair_evidence = adjacent_pair_evidence_padded
            deployable_hint = deployable_hint_padded

        total_loss = torch.zeros((), device=self.device)
        logs: dict[str, float] = {}
        self._add_slow_latent_phase_alignment_logs(
            logs,
            diagnostics,
            event_labels_raw,
            stair_labels,
            batch.dones,
            batch.masks,
        )
        if event_coef != 0.0 and "event_logit" in aux_outputs:
            event_prob = torch.sigmoid(aux_outputs["event_logit"].detach())
            event_labels_detached = event_labels.detach()
            event_labels_raw_detached = event_labels_raw.detach()
            event_positive = event_labels_detached > 0.5
            event_positive_raw = event_labels_raw_detached > 0.5
            event_negative = ~event_positive
            event_pred_0p6 = event_prob > 0.6
            event_pred_0p4 = event_prob > 0.4
            event_on_threshold = float(getattr(self.actor, "event_on_threshold", 0.6))
            event_pos_weight = float(getattr(self.actor, "aux_event_pos_weight", 1.0))
            event_pred_on_threshold = event_prob > event_on_threshold
            event_pos_count = event_positive.float().sum().clamp_min(1.0)
            event_raw_pos_count = event_positive_raw.float().sum().clamp_min(1.0)
            event_neg_count = event_negative.float().sum().clamp_min(1.0)
            event_pred_on_count = event_pred_on_threshold.float().sum().clamp_min(1.0)
            event_pred_0p6_count = event_pred_0p6.float().sum().clamp_min(1.0)
            event_true_positive_on = (event_pred_on_threshold & event_positive).float().sum()
            event_raw_true_positive_on = (event_pred_on_threshold & event_positive_raw).float().sum()
            event_true_positive_0p6 = (event_pred_0p6 & event_positive).float().sum()
            event_raw_true_positive_0p6 = (event_pred_0p6 & event_positive_raw).float().sum()
            flat_event_prob = event_prob.float().reshape(-1)
            event_loss_raw = functional.binary_cross_entropy_with_logits(
                aux_outputs["event_logit"],
                event_labels,
                pos_weight=aux_outputs["event_logit"].new_tensor(event_pos_weight),
            )
            event_loss = event_coef * event_loss_raw
            total_loss = total_loss + event_loss
            logs["slow_latent_event_bce"] = self._distributed_mean_scalar(event_loss_raw).item()
            logs["slow_latent_event_loss"] = self._distributed_mean_scalar(event_loss).item()
            logs["slow_latent_event_pos_weight"] = event_pos_weight
            logs["slow_latent_event_label_window_steps"] = float(event_label_window_steps)
            logs["slow_latent_event_raw_label_mean"] = self._distributed_mean_scalar(
                event_labels_raw_detached.float().mean()
            ).item()
            logs["slow_latent_event_raw_label_positive_count"] = self._distributed_mean_scalar(
                event_positive_raw.float().sum()
            ).item()
            logs["slow_latent_event_label_mean"] = self._distributed_mean_scalar(
                event_labels_detached.float().mean()
            ).item()
            logs["slow_latent_event_raw_label_positive_ratio"] = self._distributed_mean_scalar(
                event_positive_raw.float().mean()
            ).item()
            logs["slow_latent_event_label_positive_ratio"] = self._distributed_mean_scalar(
                event_positive.float().mean()
            ).item()
            logs["slow_latent_event_prob_max"] = self._distributed_mean_scalar(flat_event_prob.max()).item()
            logs["slow_latent_event_prob_p99"] = self._distributed_mean_scalar(
                torch.quantile(flat_event_prob, 0.99)
            ).item()
            logs["slow_latent_event_prob_gt_on_threshold_ratio"] = self._distributed_mean_scalar(
                event_pred_on_threshold.float().mean()
            ).item()
            logs["slow_latent_event_prob_gt_0p6_ratio"] = self._distributed_mean_scalar(
                event_pred_0p6.float().mean()
            ).item()
            logs["slow_latent_event_prob_gt_0p4_ratio"] = self._distributed_mean_scalar(
                event_pred_0p4.float().mean()
            ).item()
            logs["slow_latent_event_recall_at_on_threshold"] = self._distributed_mean_scalar(
                event_true_positive_on / event_pos_count
            ).item()
            logs["slow_latent_event_raw_recall_at_on_threshold"] = self._distributed_mean_scalar(
                event_raw_true_positive_on / event_raw_pos_count
            ).item()
            logs["slow_latent_event_precision_at_on_threshold"] = self._distributed_mean_scalar(
                event_true_positive_on / event_pred_on_count
            ).item()
            logs["slow_latent_event_prob_pos_mean"] = self._distributed_mean_scalar(
                (event_prob * event_positive.float()).sum() / event_pos_count
            ).item()
            logs["slow_latent_event_prob_raw_pos_mean"] = self._distributed_mean_scalar(
                (event_prob * event_positive_raw.float()).sum() / event_raw_pos_count
            ).item()
            logs["slow_latent_event_prob_neg_mean"] = self._distributed_mean_scalar(
                (event_prob * event_negative.float()).sum() / event_neg_count
            ).item()
            logs["slow_latent_event_recall_at_0p6"] = self._distributed_mean_scalar(
                event_true_positive_0p6 / event_pos_count
            ).item()
            logs["slow_latent_event_raw_recall_at_0p6"] = self._distributed_mean_scalar(
                event_raw_true_positive_0p6 / event_raw_pos_count
            ).item()
            logs["slow_latent_event_precision_at_0p6"] = self._distributed_mean_scalar(
                event_true_positive_0p6 / event_pred_0p6_count
            ).item()
        if stair_coef != 0.0 and "stair_logit" in aux_outputs:
            stair_prob = torch.sigmoid(aux_outputs["stair_logit"].detach())
            stair_labels_detached = stair_labels.detach()
            stair_positive = stair_labels_detached > 0.5
            stair_negative = ~stair_positive
            stair_pos_count = stair_positive.float().sum().clamp_min(1.0)
            stair_neg_count = stair_negative.float().sum().clamp_min(1.0)
            stair_pos_weight = float(getattr(self.actor, "aux_stair_pos_weight", 1.0))
            stair_on_threshold = float(getattr(self.actor, "stair_on_threshold", 0.35))
            stair_pred_on = stair_prob > stair_on_threshold
            stair_pred_on_count = stair_pred_on.float().sum().clamp_min(1.0)
            stair_true_positive_on = (stair_pred_on & stair_positive).float().sum()
            stair_loss_raw = functional.binary_cross_entropy_with_logits(
                aux_outputs["stair_logit"],
                stair_labels,
                pos_weight=aux_outputs["stair_logit"].new_tensor(stair_pos_weight),
            )
            stair_loss = stair_coef * stair_loss_raw
            total_loss = total_loss + stair_loss
            logs["slow_latent_stair_bce"] = self._distributed_mean_scalar(stair_loss_raw).item()
            logs["slow_latent_stair_loss"] = self._distributed_mean_scalar(stair_loss).item()
            logs["slow_latent_stair_pos_weight"] = stair_pos_weight
            logs["slow_latent_stair_on_threshold"] = stair_on_threshold
            logs["slow_latent_stair_confirm_steps"] = float(getattr(self.actor, "stair_confirm_steps", 1.0))
            logs["slow_latent_stair_label_mean"] = self._distributed_mean_scalar(
                stair_labels_detached.float().mean()
            ).item()
            logs["slow_latent_stair_label_positive_count"] = self._distributed_mean_scalar(
                stair_positive.float().sum()
            ).item()
            logs["slow_latent_stair_prob_pos_mean"] = self._distributed_mean_scalar(
                (stair_prob * stair_positive.float()).sum() / stair_pos_count
            ).item()
            logs["slow_latent_stair_prob_neg_mean"] = self._distributed_mean_scalar(
                (stair_prob * stair_negative.float()).sum() / stair_neg_count
            ).item()
            logs["slow_latent_stair_prob_gt_0p4_ratio"] = self._distributed_mean_scalar(
                (stair_prob > 0.4).float().mean()
            ).item()
            logs["slow_latent_stair_prob_gt_on_threshold_ratio"] = self._distributed_mean_scalar(
                stair_pred_on.float().mean()
            ).item()
            logs["slow_latent_stair_recall_at_on_threshold"] = self._distributed_mean_scalar(
                stair_true_positive_on / stair_pos_count
            ).item()
            logs["slow_latent_stair_precision_at_on_threshold"] = self._distributed_mean_scalar(
                stair_true_positive_on / stair_pred_on_count
            ).item()
        if future_risk_coef != 0.0 and "future_collision_risk_logit" in aux_outputs:
            risk_loss_raw, risk_mae = self._compute_weighted_bounded_huber(
                aux_outputs["future_collision_risk_logit"],
                future_risk_labels,
                float(getattr(self.actor, "future_risk_weight_scale", 2.0)),
                float(getattr(self.actor, "future_risk_huber_delta", 0.1)),
            )
            risk_loss = future_risk_coef * risk_loss_raw
            total_loss = total_loss + risk_loss
            logs["slow_latent_future_collision_risk_huber"] = self._distributed_mean_scalar(risk_loss_raw).item()
            logs["slow_latent_future_collision_risk_loss"] = self._distributed_mean_scalar(risk_loss).item()
            logs["slow_latent_future_collision_risk_mae"] = self._distributed_mean_scalar(risk_mae).item()
            logs["slow_latent_future_collision_risk_label_mean"] = self._distributed_mean_scalar(
                future_risk_labels.mean()
            ).item()
        if future_quality_coef != 0.0 and "future_safe_landing_quality_logit" in aux_outputs:
            quality_loss_raw, quality_mae = self._compute_weighted_bounded_huber(
                aux_outputs["future_safe_landing_quality_logit"],
                future_quality_labels,
                float(getattr(self.actor, "future_quality_weight_scale", 2.0)),
                float(getattr(self.actor, "future_quality_huber_delta", 0.1)),
            )
            quality_loss = future_quality_coef * quality_loss_raw
            total_loss = total_loss + quality_loss
            logs["slow_latent_future_safe_landing_quality_huber"] = self._distributed_mean_scalar(
                quality_loss_raw
            ).item()
            logs["slow_latent_future_safe_landing_quality_loss"] = self._distributed_mean_scalar(quality_loss).item()
            logs["slow_latent_future_safe_landing_quality_mae"] = self._distributed_mean_scalar(quality_mae).item()
            logs["slow_latent_future_safe_landing_quality_label_mean"] = self._distributed_mean_scalar(
                future_quality_labels.mean()
            ).item()
            logs["slow_latent_future_touchdown_found_ratio"] = self._distributed_mean_scalar(
                future_touchdown_found.mean()
            ).item()
            found_count = future_touchdown_found.sum().clamp_min(1.0)
            first_touchdown_quality = (future_quality_labels * future_touchdown_found).sum() / found_count
            logs["slow_latent_first_touchdown_quality_mean"] = self._distributed_mean_scalar(
                first_touchdown_quality
            ).item()
        if future_risk_coef != 0.0 and future_quality_coef != 0.0:
            label_overlap = (future_risk_labels * future_quality_labels).mean()
            logs["slow_latent_future_risk_quality_label_overlap_mean"] = self._distributed_mean_scalar(
                label_overlap
            ).item()
        if shape_coef != 0.0 and ("geometry_probe" in aux_outputs or "stair_shape" in aux_outputs):
            huber_delta = float(getattr(self.actor, "stair_shape_huber_delta", 0.05))
            if "geometry_probe" in aux_outputs:
                shape_predictions = aux_outputs["geometry_probe"]
            else:
                shape_predictions = aux_outputs["stair_shape"]
            geometry_component_valid = shape_component_valid
            if self.geometry_probe_only:
                confirmation_age = depth_confirmation_age[..., 0]
                sparse_age_sample = (
                    (confirmation_age == 0)
                    | (confirmation_age == 1)
                    | (confirmation_age == 4)
                    | (confirmation_age == 8)
                    | (confirmation_age == 16)
                    | (confirmation_age == 32)
                )
                depth_sample = sparse_age_sample.to(shape_component_valid.dtype)
                height_sample = (sparse_age_sample | (event_labels_raw[..., 0] > 0.5)).to(shape_component_valid.dtype)
                geometry_sample = torch.stack(
                    [depth_sample, height_sample],
                    dim=-1,
                )
                geometry_component_valid = shape_component_valid * geometry_sample
            shape_loss_valid = geometry_component_valid
            if self.geometry_probe_only:
                shape_loss_valid = geometry_component_valid * (1.0 - geometry_probe_validation)
            shape_loss_labels = shape_labels
            if self.geometry_probe_only and self.geometry_probe_permute_depth_labels:
                shape_loss_labels = shape_labels.clone()
                depth_labels_flat = shape_loss_labels[..., 0].reshape(-1)
                depth_valid_flat = shape_loss_valid[..., 0].reshape(-1) > 0.5
                depth_valid_indices = depth_valid_flat.nonzero(as_tuple=False).squeeze(-1)
                if depth_valid_indices.numel() > 1:
                    depth_values = depth_labels_flat[depth_valid_indices].clone()
                    depth_labels_flat[depth_valid_indices] = depth_values.roll(1)
            shape_lower_bounds = shape_labels.new_tensor([
                float(getattr(self.actor, "tread_depth_min", 0.18)),
                float(getattr(self.actor, "riser_height_min", 0.088)),
            ])
            shape_upper_bounds = shape_labels.new_tensor([
                float(getattr(self.actor, "tread_depth_max", 0.35)),
                float(getattr(self.actor, "riser_height_max", 0.25)),
            ])
            if self.geometry_probe_only:
                shape_loss_raw = shape_predictions.new_zeros(())
                for component_index in range(2):
                    shape_loss_raw = shape_loss_raw + self._compute_normalized_stair_shape_loss(
                        shape_predictions[..., component_index : component_index + 1],
                        shape_loss_labels[..., component_index : component_index + 1],
                        shape_loss_valid[..., component_index : component_index + 1],
                        shape_lower_bounds[component_index : component_index + 1],
                        shape_upper_bounds[component_index : component_index + 1],
                        huber_delta,
                    )
            else:
                shape_loss_raw = self._compute_normalized_stair_shape_loss(
                    shape_predictions,
                    shape_loss_labels,
                    shape_loss_valid,
                    shape_lower_bounds,
                    shape_upper_bounds,
                    huber_delta,
                )
            shape_mae, shape_huber = self._compute_stair_shape_component_errors(
                shape_predictions,
                shape_labels,
                shape_loss_valid,
                huber_delta,
            )
            if self.geometry_probe_only:
                self._accumulate_geometry_probe_statistics(
                    shape_predictions,
                    shape_labels,
                    geometry_component_valid,
                    geometry_probe_validation,
                    depth_confirmation_age,
                )
                pair_valid = adjacent_pair_evidence[..., 2]
                pair_event = adjacent_pair_evidence[..., 3]
                self._accumulate_geometry_probe_component(
                    adjacent_pair_evidence[..., 0],
                    shape_labels[..., 0],
                    pair_valid,
                    "physical_pair_frame_depth",
                )
                self._accumulate_geometry_probe_component(
                    adjacent_pair_evidence[..., 1],
                    shape_labels[..., 1],
                    pair_valid,
                    "physical_pair_frame_height",
                )
                self._accumulate_geometry_probe_component(
                    adjacent_pair_evidence[..., 0],
                    shape_labels[..., 0],
                    pair_event,
                    "physical_pair_event_depth",
                )
                self._accumulate_geometry_probe_component(
                    adjacent_pair_evidence[..., 1],
                    shape_labels[..., 1],
                    pair_event,
                    "physical_pair_event_height",
                )
            shape_loss = shape_coef * shape_loss_raw
            total_loss = total_loss + shape_loss
            normalized_huber_mean = self._distributed_mean_scalar(shape_loss_raw)
            normalized_shape_huber = normalized_huber_mean.item()
            logs["slow_latent_shape_huber"] = normalized_shape_huber
            logs["slow_latent_shape_normalized_huber"] = normalized_shape_huber
            logs["slow_latent_shape_loss"] = self._distributed_mean_scalar(shape_loss).item()
            logs["slow_latent_shape_valid_ratio"] = self._distributed_mean_scalar(shape_valid.mean()).item()
            shape_valid_count = self._distributed_mean_scalar(shape_valid.sum())
            logs["slow_latent_shape_valid_count"] = shape_valid_count.item()
            logs["slow_latent_tread_depth_valid_ratio"] = self._distributed_mean_scalar(
                shape_component_valid[..., 0].mean()
            ).item()
            logs["slow_latent_riser_height_valid_ratio"] = self._distributed_mean_scalar(
                shape_component_valid[..., 1].mean()
            ).item()
            if self.geometry_probe_only:
                logs["slow_latent_geometry_probe_validation_ratio"] = self._distributed_mean_scalar(
                    geometry_probe_validation.mean()
                ).item()
                logs["slow_latent_geometry_probe_depth_sample_ratio"] = self._distributed_mean_scalar(
                    geometry_component_valid[..., 0].mean()
                ).item()
                logs["slow_latent_geometry_probe_permuted_depth_labels"] = float(
                    self.geometry_probe_permute_depth_labels
                )
            stair_positive_float = (stair_labels > 0.5).to(shape_valid.dtype)
            stair_positive_count = stair_positive_float.sum().clamp_min(1.0)
            flat_count = (1.0 - stair_positive_float).sum().clamp_min(1.0)
            valid_stair = shape_valid * stair_positive_float
            valid_flat = shape_valid * (1.0 - stair_positive_float)
            stair_coverage = valid_stair.sum() / stair_positive_count
            flat_leakage = valid_flat.sum() / flat_count
            mean_scalar = self._distributed_mean_scalar
            stair_coverage_mean = mean_scalar(stair_coverage).item()
            flat_leakage_mean = mean_scalar(flat_leakage).item()
            logs["slow_latent_shape_valid_stair_coverage"] = stair_coverage_mean
            logs["slow_latent_shape_valid_while_flat_ratio"] = flat_leakage_mean
            component_masks = {
                "tread_depth": shape_component_valid[..., 0:1],
                "riser_height": shape_component_valid[..., 1:2],
            }
            for component_name, component_valid in component_masks.items():
                component_stair = component_valid * stair_positive_float
                component_flat = component_valid * (1.0 - stair_positive_float)
                logs[f"slow_latent_{component_name}_valid_stair_coverage"] = mean_scalar(
                    component_stair.sum() / stair_positive_count
                ).item()
                logs[f"slow_latent_{component_name}_valid_while_flat_ratio"] = mean_scalar(
                    component_flat.sum() / flat_count
                ).item()
            logs["slow_latent_tread_depth_mae"] = self._distributed_mean_scalar(shape_mae[0]).item()
            logs["slow_latent_riser_height_mae"] = self._distributed_mean_scalar(shape_mae[1]).item()
            logs["slow_latent_tread_depth_huber"] = self._distributed_mean_scalar(shape_huber[0]).item()
            logs["slow_latent_riser_height_huber"] = self._distributed_mean_scalar(shape_huber[1]).item()
            shape_out_of_range = self._compute_label_out_of_range_ratios(
                shape_labels,
                shape_component_valid,
                shape_lower_bounds,
                shape_upper_bounds,
            )
            logs["slow_latent_tread_depth_out_of_range_label_ratio"] = self._distributed_mean_scalar(
                shape_out_of_range[0]
            ).item()
            logs["slow_latent_riser_height_out_of_range_label_ratio"] = self._distributed_mean_scalar(
                shape_out_of_range[1]
            ).item()
            regression_stats = self._compute_masked_regression_statistics(
                shape_predictions.detach(),
                shape_labels.detach(),
                shape_component_valid.detach(),
            )
            label_std, prediction_std, correlation, r_squared = regression_stats
            shape_ranges = shape_upper_bounds - shape_lower_bounds
            normalized_mae = shape_mae / shape_ranges
            component_names = ("tread_depth", "riser_height")
            for component_index, component_name in enumerate(component_names):
                index = component_index
                normalized_mae_mean = mean_scalar(normalized_mae[index])
                label_std_mean = mean_scalar(label_std[index])
                prediction_std_mean = mean_scalar(prediction_std[index])
                correlation_mean = mean_scalar(correlation[index])
                r_squared_mean = mean_scalar(r_squared[index])
                prefix = f"slow_latent_{component_name}"
                logs[f"{prefix}_normalized_mae"] = normalized_mae_mean.item()
                logs[f"{prefix}_label_std"] = label_std_mean.item()
                logs[f"{prefix}_pred_std"] = prediction_std_mean.item()
                logs[f"{prefix}_correlation"] = correlation_mean.item()
                logs[f"{prefix}_r_squared"] = r_squared_mean.item()
        if safe_stride_coef != 0.0 and "safe_stride" in aux_outputs:
            safe_stride_delta = float(getattr(self.actor, "safe_stride_huber_delta", 0.05))
            lower_shortfall_coef = float(getattr(self.actor, "safe_stride_lower_shortfall_coef", 1.0))
            coverage_loss_coef = float(getattr(self.actor, "safe_stride_interval_coverage_loss_coef", 0.0))
            coverage_margin = float(getattr(self.actor, "safe_stride_interval_coverage_margin", 0.01))
            std_floor_loss_coef = float(getattr(self.actor, "safe_stride_std_floor_loss_coef", 0.0))
            centered_loss_coef = float(getattr(self.actor, "safe_stride_centered_loss_coef", 0.0))
            std_floor_ratio = float(getattr(self.actor, "safe_stride_std_floor_ratio", 0.70))
            hint_loss_coef = float(getattr(self.actor, "safe_stride_deployable_hint_loss_coef", 0.0))
            hint_margin = float(getattr(self.actor, "safe_stride_deployable_hint_margin", 0.02))
            safe_stride_predictions = aux_outputs["safe_stride"]
            safe_stride_interval_predictions = aux_outputs.get("safe_stride_interval")
            safe_stride_confidence_logit = aux_outputs.get("safe_stride_confidence_logit")
            safe_stride_target_center = 0.5 * (safe_stride_labels + safe_stride_upper)
            interval_valid = safe_stride_valid * safe_stride_interval_valid
            lower_loss_raw = safe_stride_predictions.new_zeros(())
            width_loss_raw = safe_stride_predictions.new_zeros(())
            confidence_loss_raw = safe_stride_predictions.new_zeros(())
            center_centered_loss_raw = safe_stride_predictions.new_zeros(())
            center_std_floor_loss_raw = safe_stride_predictions.new_zeros(())
            lower_centered_loss_raw = safe_stride_predictions.new_zeros(())
            lower_std_floor_loss_raw = safe_stride_predictions.new_zeros(())
            coverage_loss_raw = safe_stride_predictions.new_zeros(())
            coverage_lower_over = safe_stride_predictions.new_zeros(())
            coverage_upper_shortfall = safe_stride_predictions.new_zeros(())
            deployable_hint_loss_raw = safe_stride_predictions.new_zeros(())
            deployable_hint_shortfall = safe_stride_predictions.new_zeros(())
            width_loss_coef = 1.0
            if safe_stride_interval_predictions is not None:
                width_loss_coef = float(getattr(self.actor, "safe_stride_width_loss_coef", 1.0))
                safe_stride_loss_raw, lower_loss_raw, width_loss_raw = self._compute_safe_stride_interval_loss(
                    safe_stride_interval_predictions,
                    safe_stride_labels,
                    safe_stride_upper,
                    safe_stride_valid,
                    interval_valid,
                    safe_stride_importance,
                    safe_stride_delta,
                    width_loss_coef,
                    lower_shortfall_coef,
                )
                predicted_lower = safe_stride_interval_predictions[..., 0:1]
                predicted_upper = safe_stride_interval_predictions[..., 1:2]
                center_valid = interval_valid
                if coverage_loss_coef != 0.0:
                    (
                        coverage_loss_raw,
                        coverage_lower_over,
                        coverage_upper_shortfall,
                    ) = self._compute_safe_stride_interval_coverage_loss(
                        safe_stride_interval_predictions,
                        safe_stride_labels,
                        safe_stride_upper,
                        interval_valid,
                        safe_stride_importance,
                        coverage_margin,
                        safe_stride_delta,
                    )
                    safe_stride_loss_raw = safe_stride_loss_raw + coverage_loss_coef * coverage_loss_raw
            else:
                safe_stride_loss_raw = self._compute_safe_stride_loss(
                    safe_stride_predictions,
                    safe_stride_labels,
                    safe_stride_upper,
                    safe_stride_valid,
                    safe_stride_importance,
                    safe_stride_delta,
                    lower_shortfall_coef,
                )
                predicted_lower = safe_stride_predictions
                predicted_upper = safe_stride_predictions
                safe_stride_target_center = safe_stride_labels
                center_valid = safe_stride_valid
            safe_stride_base_loss_raw = safe_stride_loss_raw
            if centered_loss_coef != 0.0 or std_floor_loss_coef != 0.0:
                (
                    center_centered_loss_raw,
                    center_std_floor_loss_raw,
                ) = self._compute_masked_centered_spread_losses(
                    safe_stride_predictions,
                    safe_stride_target_center,
                    center_valid,
                    std_floor_ratio,
                )
                if safe_stride_interval_predictions is not None:
                    (
                        lower_centered_loss_raw,
                        lower_std_floor_loss_raw,
                    ) = self._compute_masked_centered_spread_losses(
                        predicted_lower,
                        safe_stride_labels,
                        safe_stride_valid,
                        std_floor_ratio,
                    )
                safe_stride_loss_raw = (
                    safe_stride_loss_raw
                    + centered_loss_coef * (center_centered_loss_raw + lower_centered_loss_raw)
                    + std_floor_loss_coef * (center_std_floor_loss_raw + lower_std_floor_loss_raw)
                )
            deployable_hint_valid = deployable_hint[..., 1:2] * safe_stride_valid
            if hint_loss_coef != 0.0:
                (
                    deployable_hint_loss_raw,
                    deployable_hint_shortfall,
                ) = self._compute_one_sided_lower_hint_loss(
                    predicted_lower,
                    deployable_hint[..., 0:1],
                    deployable_hint_valid,
                    safe_stride_importance,
                    hint_margin,
                    safe_stride_delta,
                )
                safe_stride_loss_raw = safe_stride_loss_raw + hint_loss_coef * deployable_hint_loss_raw
            if safe_stride_confidence_logit is not None:
                confidence_target = (
                    interval_valid if safe_stride_interval_predictions is not None else safe_stride_valid
                )
                confidence_loss_raw = self._compute_safe_stride_confidence_loss(
                    safe_stride_confidence_logit,
                    confidence_target,
                    safe_stride_importance,
                )
                confidence_loss_coef = float(getattr(self.actor, "safe_stride_confidence_loss_coef", 0.0))
                safe_stride_loss_raw = safe_stride_loss_raw + confidence_loss_coef * confidence_loss_raw
            else:
                confidence_loss_coef = 0.0
            safe_stride_mae, safe_stride_huber = self._compute_stair_shape_component_errors(
                safe_stride_predictions,
                safe_stride_target_center,
                center_valid,
                safe_stride_delta,
            )
            safe_stride_loss = safe_stride_coef * safe_stride_loss_raw
            total_loss = total_loss + safe_stride_loss
            self._accumulate_safe_stride_update_statistics(
                safe_stride_predictions,
                safe_stride_target_center,
                center_valid,
            )
            if safe_stride_interval_predictions is not None:
                self._accumulate_safe_stride_update_statistics(
                    predicted_lower,
                    safe_stride_labels,
                    safe_stride_valid,
                    component="lower",
                )
                self._accumulate_safe_stride_update_statistics(
                    predicted_lower,
                    safe_stride_labels,
                    interval_valid,
                    component="lower_interval",
                )
                self._accumulate_safe_stride_update_statistics(
                    predicted_upper,
                    safe_stride_upper,
                    interval_valid,
                    component="upper",
                )
                self._accumulate_safe_stride_update_statistics(
                    predicted_upper - predicted_lower,
                    safe_stride_upper - safe_stride_labels,
                    interval_valid,
                    component="width",
                )
            self._accumulate_safe_stride_depth_confirmation_events(depth_confirmation_event)
            logs["slow_latent_safe_stride_huber"] = self._distributed_mean_scalar(safe_stride_base_loss_raw).item()
            logs["slow_latent_safe_stride_regularized_huber"] = self._distributed_mean_scalar(
                safe_stride_loss_raw
            ).item()
            logs["slow_latent_safe_stride_loss"] = self._distributed_mean_scalar(safe_stride_loss).item()
            if safe_stride_interval_predictions is not None:
                logs["slow_latent_safe_stride_lower_huber"] = self._distributed_mean_scalar(lower_loss_raw).item()
                logs["slow_latent_safe_stride_width_huber"] = self._distributed_mean_scalar(width_loss_raw).item()
                logs["slow_latent_safe_stride_width_loss_coef"] = width_loss_coef
                logs["slow_latent_safe_stride_lower_shortfall_coef"] = lower_shortfall_coef
                logs["slow_latent_safe_stride_interval_coverage_loss_coef"] = coverage_loss_coef
                logs["slow_latent_safe_stride_interval_coverage_margin"] = coverage_margin
                logs["slow_latent_safe_stride_interval_coverage_huber"] = self._distributed_mean_scalar(
                    coverage_loss_raw
                ).item()
                logs["slow_latent_safe_stride_interval_lower_over_mae"] = self._distributed_mean_scalar(
                    coverage_lower_over
                ).item()
                logs["slow_latent_safe_stride_interval_upper_shortfall_mae"] = self._distributed_mean_scalar(
                    coverage_upper_shortfall
                ).item()
            if centered_loss_coef != 0.0 or std_floor_loss_coef != 0.0:
                logs["slow_latent_safe_stride_centered_loss_coef"] = centered_loss_coef
                logs["slow_latent_safe_stride_std_floor_loss_coef"] = std_floor_loss_coef
                logs["slow_latent_safe_stride_std_floor_ratio"] = std_floor_ratio
                logs["slow_latent_safe_stride_center_centered_loss"] = self._distributed_mean_scalar(
                    center_centered_loss_raw
                ).item()
                logs["slow_latent_safe_stride_center_std_floor_loss"] = self._distributed_mean_scalar(
                    center_std_floor_loss_raw
                ).item()
                logs["slow_latent_safe_stride_lower_centered_loss"] = self._distributed_mean_scalar(
                    lower_centered_loss_raw
                ).item()
                logs["slow_latent_safe_stride_lower_std_floor_loss"] = self._distributed_mean_scalar(
                    lower_std_floor_loss_raw
                ).item()
            if hint_loss_coef != 0.0:
                hint_valid_count = deployable_hint_valid.sum().clamp_min(1.0)
                logs["slow_latent_safe_stride_deployable_hint_loss_coef"] = hint_loss_coef
                logs["slow_latent_safe_stride_deployable_hint_margin"] = hint_margin
                logs["slow_latent_safe_stride_deployable_hint_huber"] = self._distributed_mean_scalar(
                    deployable_hint_loss_raw
                ).item()
                logs["slow_latent_safe_stride_deployable_hint_valid_ratio"] = self._distributed_mean_scalar(
                    deployable_hint_valid.mean()
                ).item()
                logs["slow_latent_safe_stride_deployable_hint_mean"] = self._distributed_mean_scalar(
                    (deployable_hint[..., 0:1] * deployable_hint_valid).sum() / hint_valid_count
                ).item()
                logs["slow_latent_safe_stride_deployable_hint_shortfall_mae"] = self._distributed_mean_scalar(
                    deployable_hint_shortfall
                ).item()
            if safe_stride_confidence_logit is not None:
                confidence_prob = torch.sigmoid(safe_stride_confidence_logit)
                confidence_target = (
                    interval_valid if safe_stride_interval_predictions is not None else safe_stride_valid
                )
                confidence_positive = confidence_target > 0.5
                confidence_negative = ~confidence_positive
                positive_count = confidence_positive.to(center_valid.dtype).sum().clamp_min(1.0)
                negative_count = confidence_negative.to(center_valid.dtype).sum().clamp_min(1.0)
                confidence_pred_positive = confidence_prob > 0.5
                confidence_true_positive = confidence_pred_positive & confidence_positive
                confidence_false_positive = confidence_pred_positive & confidence_negative
                confidence_false_negative = (~confidence_pred_positive) & confidence_positive
                logs["slow_latent_safe_stride_confidence_bce"] = self._distributed_mean_scalar(
                    confidence_loss_raw
                ).item()
                logs["slow_latent_safe_stride_confidence_loss_coef"] = confidence_loss_coef
                logs["slow_latent_safe_stride_confidence_mean"] = self._distributed_mean_scalar(
                    confidence_prob.mean()
                ).item()
                logs["slow_latent_safe_stride_confidence_valid_mean"] = self._distributed_mean_scalar(
                    (confidence_prob * confidence_positive.to(confidence_prob.dtype)).sum() / positive_count
                ).item()
                logs["slow_latent_safe_stride_confidence_invalid_mean"] = self._distributed_mean_scalar(
                    (confidence_prob * confidence_negative.to(confidence_prob.dtype)).sum() / negative_count
                ).item()
                logs["slow_latent_safe_stride_confidence_precision"] = self._distributed_mean_scalar(
                    confidence_true_positive.to(center_valid.dtype).sum()
                    / (
                        confidence_true_positive.to(center_valid.dtype).sum()
                        + confidence_false_positive.to(center_valid.dtype).sum()
                    ).clamp_min(1.0)
                ).item()
                logs["slow_latent_safe_stride_confidence_recall"] = self._distributed_mean_scalar(
                    confidence_true_positive.to(center_valid.dtype).sum()
                    / (
                        confidence_true_positive.to(center_valid.dtype).sum()
                        + confidence_false_negative.to(center_valid.dtype).sum()
                    ).clamp_min(1.0)
                ).item()
            logs["slow_latent_safe_stride_valid_ratio"] = self._distributed_mean_scalar(safe_stride_valid.mean()).item()
            logs["slow_latent_safe_stride_interval_valid_ratio"] = self._distributed_mean_scalar(
                interval_valid.sum() / safe_stride_valid.sum().clamp_min(1.0)
            ).item()
            valid_count = center_valid.sum().clamp_min(1.0)
            below_interval = safe_stride_predictions < safe_stride_labels
            above_interval = safe_stride_predictions > safe_stride_upper
            interval_distance = torch.relu(safe_stride_labels - safe_stride_predictions) + torch.relu(
                safe_stride_predictions - safe_stride_upper
            )
            logs["slow_latent_safe_stride_center_in_target_interval_ratio"] = self._distributed_mean_scalar(
                (center_valid * (~below_interval & ~above_interval).to(center_valid.dtype)).sum() / valid_count
            ).item()
            logs["slow_latent_safe_stride_below_interval_ratio"] = self._distributed_mean_scalar(
                (center_valid * below_interval.to(center_valid.dtype)).sum() / valid_count
            ).item()
            logs["slow_latent_safe_stride_above_interval_ratio"] = self._distributed_mean_scalar(
                (center_valid * above_interval.to(center_valid.dtype)).sum() / valid_count
            ).item()
            logs["slow_latent_safe_stride_interval_violation_mae"] = self._distributed_mean_scalar(
                (interval_distance * center_valid).sum() / valid_count
            ).item()
            logs["slow_latent_safe_stride_interval_width_mean"] = self._distributed_mean_scalar(
                ((safe_stride_upper - safe_stride_labels) * center_valid).sum() / valid_count
            ).item()
            if safe_stride_interval_predictions is not None:
                target_width = safe_stride_upper - safe_stride_labels
                predicted_width = predicted_upper - predicted_lower
                lower_error = (predicted_lower - safe_stride_labels).abs()
                upper_error = (predicted_upper - safe_stride_upper).abs()
                width_error = (predicted_width - target_width).abs()
                overlap = torch.relu(
                    torch.minimum(predicted_upper, safe_stride_upper)
                    - torch.maximum(predicted_lower, safe_stride_labels)
                )
                union = (
                    torch.maximum(predicted_upper, safe_stride_upper)
                    - torch.minimum(predicted_lower, safe_stride_labels)
                ).clamp_min(1.0e-6)
                target_width_safe = target_width.clamp_min(1.0e-6)
                mean_scalar = self._distributed_mean_scalar
                logs["slow_latent_safe_stride_predicted_width_mean"] = mean_scalar(
                    (predicted_width * center_valid).sum() / valid_count
                ).item()
                logs["slow_latent_safe_stride_lower_boundary_mae"] = mean_scalar(
                    (lower_error * safe_stride_valid).sum() / safe_stride_valid.sum().clamp_min(1.0)
                ).item()
                logs["slow_latent_safe_stride_upper_boundary_mae"] = mean_scalar(
                    (upper_error * center_valid).sum() / valid_count
                ).item()
                logs["slow_latent_safe_stride_width_mae"] = mean_scalar(
                    (width_error * center_valid).sum() / valid_count
                ).item()
                logs["slow_latent_safe_stride_width_bias"] = mean_scalar(
                    ((predicted_width - target_width) * center_valid).sum() / valid_count
                ).item()
                safe_stride_max = float(getattr(self.actor, "safe_stride_max", 0.55))
                logs["slow_latent_safe_stride_upper_out_of_range_ratio"] = mean_scalar(
                    ((predicted_upper > safe_stride_max).to(center_valid.dtype) * center_valid).sum() / valid_count
                ).item()
                logs["slow_latent_safe_stride_interval_overlap_ratio"] = mean_scalar(
                    ((overlap > 0.0).to(center_valid.dtype) * center_valid).sum() / valid_count
                ).item()
                logs["slow_latent_safe_stride_interval_target_coverage_mean"] = mean_scalar(
                    ((overlap / target_width_safe) * center_valid).sum() / valid_count
                ).item()
                logs["slow_latent_safe_stride_interval_iou_mean"] = mean_scalar(
                    ((overlap / union) * center_valid).sum() / valid_count
                ).item()
            logs["slow_latent_safe_stride_importance_mean"] = self._distributed_mean_scalar(
                (safe_stride_importance * safe_stride_valid).sum() / safe_stride_valid.sum().clamp_min(1.0)
            ).item()
            logs["slow_latent_safe_stride_evidence_weighted_ratio"] = self._distributed_mean_scalar(
                (safe_stride_valid * (safe_stride_importance > 1.0).to(safe_stride_valid.dtype)).sum()
                / safe_stride_valid.sum().clamp_min(1.0)
            ).item()
            exact_valid = interval_valid
            lower_bound_valid = safe_stride_valid * (1.0 - safe_stride_interval_valid)
            lower_bound_count = lower_bound_valid.sum().clamp_min(1.0)
            lower_bound_shortfall = torch.relu(safe_stride_labels - predicted_lower)
            logs["slow_latent_safe_stride_exact_ratio"] = self._distributed_mean_scalar(
                exact_valid.sum() / valid_count
            ).item()
            logs["slow_latent_safe_stride_lower_bound_violation_mae"] = self._distributed_mean_scalar(
                (lower_bound_shortfall * lower_bound_valid).sum() / lower_bound_count
            ).item()
            logs["slow_latent_safe_stride_mae"] = self._distributed_mean_scalar(safe_stride_mae[0]).item()
            logs["slow_latent_safe_stride_component_huber"] = self._distributed_mean_scalar(safe_stride_huber[0]).item()
            safe_stride_lower = safe_stride_labels.new_tensor([float(getattr(self.actor, "safe_stride_min", 0.10))])
            safe_stride_head_upper = safe_stride_labels.new_tensor([
                float(getattr(self.actor, "safe_stride_max", 0.55))
            ])
            safe_stride_out_of_range = self._compute_label_out_of_range_ratios(
                safe_stride_target_center,
                center_valid,
                safe_stride_lower,
                safe_stride_head_upper,
            )
            logs["slow_latent_safe_stride_out_of_range_label_ratio"] = self._distributed_mean_scalar(
                safe_stride_out_of_range[0]
            ).item()
            safe_stride_statistics = self._compute_masked_regression_statistics(
                safe_stride_predictions.detach(),
                safe_stride_target_center.detach(),
                center_valid.detach(),
            )
            label_std, prediction_std, correlation, r_squared = safe_stride_statistics
            label_mean = (safe_stride_target_center * center_valid).sum() / valid_count
            prediction_mean = (safe_stride_predictions * center_valid).sum() / valid_count
            exact_count = exact_valid.sum().clamp_min(1.0)
            prediction_error = safe_stride_predictions - safe_stride_target_center
            signed_error = (prediction_error * center_valid).sum() / valid_count
            exact_mae = (prediction_error.abs() * exact_valid).sum() / exact_count
            exact_signed_error = (prediction_error * exact_valid).sum() / exact_count
            exact_statistics = self._compute_masked_regression_statistics(
                safe_stride_predictions.detach(),
                safe_stride_target_center.detach(),
                exact_valid.detach(),
            )
            (
                _exact_label_std,
                _exact_prediction_std,
                exact_correlation,
                exact_r_squared,
            ) = exact_statistics
            stride_range = safe_stride_head_upper[0] - safe_stride_lower[0]
            mean_scalar = self._distributed_mean_scalar
            logs["slow_latent_safe_stride_label_mean"] = mean_scalar(label_mean).item()
            logs["slow_latent_safe_stride_valid_pred_mean"] = mean_scalar(prediction_mean).item()
            logs["slow_latent_safe_stride_label_std"] = mean_scalar(label_std[0]).item()
            logs["slow_latent_safe_stride_pred_std"] = mean_scalar(prediction_std[0]).item()
            logs["slow_latent_safe_stride_correlation"] = mean_scalar(correlation[0]).item()
            logs["slow_latent_safe_stride_r_squared"] = mean_scalar(r_squared[0]).item()
            logs["slow_latent_safe_stride_normalized_mae"] = mean_scalar(safe_stride_mae[0] / stride_range).item()
            logs["slow_latent_safe_stride_signed_error"] = mean_scalar(signed_error).item()
            logs["slow_latent_safe_stride_evidence_exact_mae"] = mean_scalar(exact_mae).item()
            logs["slow_latent_safe_stride_evidence_exact_signed_error"] = mean_scalar(exact_signed_error).item()
            logs["slow_latent_safe_stride_evidence_exact_correlation"] = mean_scalar(exact_correlation[0]).item()
            logs["slow_latent_safe_stride_evidence_exact_r_squared"] = mean_scalar(exact_r_squared[0]).item()
            stair_positive = (stair_labels > 0.5).to(safe_stride_valid.dtype)
            stair_count = stair_positive.sum().clamp_min(1.0)
            flat = 1.0 - stair_positive
            flat_count = flat.sum().clamp_min(1.0)
            logs["slow_latent_safe_stride_valid_while_stair_ratio"] = mean_scalar(
                (safe_stride_valid * stair_positive).sum() / stair_count
            ).item()
            logs["slow_latent_safe_stride_valid_while_flat_ratio"] = mean_scalar(
                (safe_stride_valid * flat).sum() / flat_count
            ).item()
        logs["slow_latent_aux_loss"] = self._distributed_mean_scalar(total_loss).item()
        logs.update(self._compute_slow_latent_diagnostic_logs())
        return total_loss, logs

    def _compute_slow_latent_diagnostic_logs(self) -> dict[str, float]:
        """Summarize slow-latent rollout/update diagnostics for training logs."""
        get_diagnostics = getattr(self.actor, "get_slow_latent_diagnostics", None)
        if get_diagnostics is None:
            return {}
        diagnostics = get_diagnostics()
        if not diagnostics:
            return {}

        logs: dict[str, float] = {}

        def add_mean(name: str, tensor: torch.Tensor | None) -> None:
            if tensor is None or tensor.numel() == 0:
                return
            value = tensor.float().mean()
            logs[name] = self._distributed_mean_scalar(value).item()

        add_mean("slow_latent_event_prob_mean", diagnostics.get("event_prob"))
        add_mean("slow_latent_stair_prob_mean", diagnostics.get("stair_prob"))
        future_risk = diagnostics.get("future_risk")
        future_quality = diagnostics.get("future_quality")
        add_mean("slow_latent_future_collision_risk_mean", future_risk)
        add_mean("slow_latent_future_safe_landing_quality_mean", future_quality)
        if future_risk is not None and future_quality is not None:
            add_mean(
                "slow_latent_future_risk_quality_overlap_mean",
                future_risk * future_quality,
            )
        add_mean("slow_latent_z_norm_mean", diagnostics.get("z_norm"))

        stair_shape = diagnostics.get("stair_shape")
        if stair_shape is not None and stair_shape.numel() > 0:
            add_mean("slow_latent_tread_depth_pred_mean", stair_shape[..., 0])
            add_mean("slow_latent_riser_height_pred_mean", stair_shape[..., 1])

        safe_stride = diagnostics.get("safe_stride")
        if safe_stride is not None and safe_stride.numel() > 0:
            add_mean("slow_latent_safe_stride_pred_mean", safe_stride[..., 0])
        add_mean(
            "slow_latent_safe_stride_confidence_pred_mean",
            diagnostics.get("safe_stride_confidence"),
        )
        safe_stride_interval = diagnostics.get("safe_stride_interval")
        if safe_stride_interval is not None and safe_stride_interval.shape[-1] == 2:
            add_mean(
                "slow_latent_safe_stride_lower_pred_mean",
                safe_stride_interval[..., 0],
            )
            add_mean(
                "slow_latent_safe_stride_upper_pred_mean",
                safe_stride_interval[..., 1],
            )
            add_mean(
                "slow_latent_safe_stride_width_pred_mean",
                safe_stride_interval[..., 1] - safe_stride_interval[..., 0],
            )

        shadow_semantic = diagnostics.get("shadow_semantic")
        if shadow_semantic is not None and shadow_semantic.shape[-1] == 16:
            semantic_names = (
                "event_on",
                "stair_on",
                "mode_normal",
                "mode_write",
                "mode_memory",
                "write_progress",
                "memory_age",
                "release_progress",
                "tread_depth_norm",
                "riser_height_norm",
                "safe_stride_lower_norm",
                "safe_stride_upper_norm",
                "safe_stride_center_norm",
                "safe_stride_width_norm",
                "clearance_height_norm",
                "safe_stride_confidence",
            )
            for index, name in enumerate(semantic_names):
                add_mean(
                    f"slow_latent_shadow_semantic_{name}_mean",
                    shadow_semantic[..., index],
                )

        z_norm = diagnostics.get("z_norm")
        if z_norm is not None and z_norm.numel() > 0:
            logs["slow_latent_z_norm_max"] = self._distributed_mean_scalar(z_norm.float().max()).item()

        alpha = diagnostics.get("alpha")
        if alpha is not None and alpha.numel() > 0:
            alpha = alpha.float()
            logs["slow_latent_alpha_mean"] = self._distributed_mean_scalar(alpha.mean()).item()
            logs["slow_latent_alpha_std"] = self._distributed_mean_scalar(alpha.std(unbiased=False)).item()
            logs["slow_latent_alpha_min"] = self._distributed_mean_scalar(alpha.min()).item()
            logs["slow_latent_alpha_max"] = self._distributed_mean_scalar(alpha.max()).item()
        add_mean("slow_latent_alpha_state_mean", diagnostics.get("alpha_state"))
        add_mean("slow_latent_alpha_shape_mean", diagnostics.get("alpha_shape"))
        add_mean(
            "slow_latent_episode_write_ever_ratio",
            diagnostics.get("episode_write_ever"),
        )
        add_mean(
            "slow_latent_episode_memory_ever_ratio",
            diagnostics.get("episode_memory_ever"),
        )
        add_mean(
            "slow_latent_gate_event_trigger_ratio",
            diagnostics.get("gate_event_trigger"),
        )
        confirm = diagnostics.get("gate_write_confirm")
        abort = diagnostics.get("gate_write_abort")
        add_mean("slow_latent_gate_write_confirm_ratio", confirm)
        add_mean("slow_latent_gate_write_abort_ratio", abort)
        add_mean(
            "slow_latent_gate_memory_exit_ratio",
            diagnostics.get("gate_memory_exit"),
        )
        add_mean(
            "slow_latent_gate_memory_event_shape_boost_ratio",
            diagnostics.get("gate_memory_event_shape_boost"),
        )
        add_mean(
            "slow_latent_gate_release_ratio",
            diagnostics.get("gate_release"),
        )
        if confirm is not None and abort is not None:
            attempts = confirm.float().sum() + abort.float().sum()
            confirm_rate = confirm.float().sum() / attempts.clamp_min(1.0)
            logs["slow_latent_gate_write_confirm_rate"] = self._distributed_mean_scalar(confirm_rate).item()

        memory_age = diagnostics.get("gate_memory_age")
        if memory_age is not None and memory_age.numel() > 0:
            positive_age = memory_age.float()[memory_age.float() > 0.0]
            if positive_age.numel() > 0:
                logs["slow_latent_memory_age_mean"] = self._distributed_mean_scalar(positive_age.mean()).item()
                logs["slow_latent_memory_age_p90"] = self._distributed_mean_scalar(
                    torch.quantile(positive_age, 0.9)
                ).item()

        mode = diagnostics.get("gate_mode")
        if mode is not None and mode.numel() > 0:
            mode = mode.float()
            total = float(mode.numel())
            mode_specs = {
                "normal": 0.0,
                "write": 1.0,
                "memory": 2.0,
            }
            for name, value in mode_specs.items():
                count = (mode == value).float().sum()
                logs[f"slow_latent_mode_{name}_count"] = self._distributed_mean_scalar(count).item()
                logs[f"slow_latent_mode_{name}_ratio"] = self._distributed_mean_scalar(count / total).item()

        return logs

    def _compute_teacher_distribution_params(
        self,
        observations: TensorDict,
        masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        """Run the frozen teacher in chunks to cap CNN activation memory."""
        if self.teacher is None:
            raise RuntimeError("Teacher guidance loss requires a loaded teacher model.")

        if masks is not None:
            observations = cast(TensorDict, unpad_trajectories(observations, masks))

        batch_shape = tuple(observations.batch_size)
        num_samples = math.prod(batch_shape) if batch_shape else 1
        flat_observations = TensorDict(
            {key: value.reshape(num_samples, *value.shape[len(batch_shape) :]) for key, value in observations.items()},
            batch_size=[num_samples],
            device=observations.device,
        )

        param_chunks: list[list[torch.Tensor]] | None = None
        for start in range(0, num_samples, self.teacher_forward_chunk_size):
            chunk = flat_observations[start : start + self.teacher_forward_chunk_size]
            self.teacher(chunk, stochastic_output=True)
            chunk_params = tuple(param.detach() for param in self.teacher.output_distribution_params)
            if param_chunks is None:
                param_chunks = [[] for _ in chunk_params]
            for chunks, param in zip(param_chunks, chunk_params):
                chunks.append(param)

        if param_chunks is None:
            return ()

        flat_params = tuple(torch.cat(chunks, dim=0) for chunks in param_chunks)
        return tuple(param.reshape(*batch_shape, *param.shape[1:]) for param in flat_params)

    def _compute_teacher_guidance_loss(
        self,
        batch: RolloutStorage.Batch,
        original_batch_size: int,
        distribution_params: tuple[torch.Tensor, ...],
        loss_weight: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute teacher guidance loss with an explicit scalar weight."""
        if self.teacher is None or not self.teacher_loaded:
            raise RuntimeError("Teacher guidance loss requires a loaded teacher model.")
        if batch.observations is None:
            raise RuntimeError("Teacher guidance loss requires observations in the rollout batch.")

        observations = batch.observations
        teacher_observations = cast(TensorDict, observations[:original_batch_size])
        teacher_masks = batch.masks[:original_batch_size] if batch.masks is not None else None

        with torch.no_grad():
            teacher_distribution_params = self._compute_teacher_distribution_params(
                teacher_observations,
                teacher_masks,
            )

        student_distribution_params = distribution_params
        if loss_weight == 0.0:
            student_distribution_params = tuple(p.detach() for p in student_distribution_params)

        if self.teacher_kl_cfg.get("check_shapes", True) and not self._teacher_kl_shapes_checked:
            self._validate_distribution_params(teacher_distribution_params, student_distribution_params)
            self._teacher_kl_shapes_checked = True

        if self.teacher_guidance_loss_type == "kl":
            teacher_loss_for_update, loss_logs = self._compute_kl_teacher_loss(
                teacher_distribution_params,
                student_distribution_params,
            )
        else:
            teacher_loss_for_update, loss_logs = self._compute_mean_teacher_loss(
                teacher_distribution_params,
                student_distribution_params,
            )

        raw_teacher_kl = loss_logs.get("teacher_kl")
        if (
            raw_teacher_kl is not None
            and self.teacher_kl_cfg.get("fail_on_nonfinite_kl", True)
            and not torch.isfinite(raw_teacher_kl)
        ):
            raise FloatingPointError(
                f"Non-finite teacher KL detected: {raw_teacher_kl.item()}. "
                "Check teacher observation normalization, action distribution parameters, and checkpoint compatibility."
            )
        if not torch.isfinite(teacher_loss_for_update):
            raise FloatingPointError(f"Non-finite teacher guidance loss detected: {teacher_loss_for_update.item()}.")

        teacher_loss = loss_weight * teacher_loss_for_update
        loss_logs.setdefault("teacher_loss_for_update", teacher_loss_for_update)
        log_dict = {name: self._distributed_mean_scalar(value).item() for name, value in loss_logs.items()}
        teacher_loss_log = self._distributed_mean_scalar(teacher_loss).item()
        log_dict.update({
            "teacher_loss": teacher_loss_log,
            "teacher_lambda": float(loss_weight),
            "teacher_kl_lambda": float(loss_weight),
        })
        if self.teacher_guidance_loss_type == "kl":
            log_dict["teacher_kl_loss"] = teacher_loss_log
        return teacher_loss, log_dict

    def update(self) -> dict[str, float]:
        """Run a PPO update and advance the teacher-KL schedule."""
        self._safe_stride_update_statistics = {}
        self._geometry_probe_update_statistics = {}
        if self.safe_stride_probe_only:
            loss_dict = self._update_safe_stride_probe_only()
        elif self.geometry_probe_only:
            loss_dict = self._update_geometry_probe_only()
        else:
            loss_dict = self._update_teacher_imitation_only() if self.teacher_imitation_only else super().update()
        loss_dict.update(self._finalize_safe_stride_update_statistics())
        loss_dict.update(self._finalize_geometry_probe_statistics())
        self._safe_stride_update_statistics = None
        self._geometry_probe_update_statistics = None
        if not self.safe_stride_probe_only and not self.geometry_probe_only:
            self.teacher_kl_iteration += 1
        return loss_dict

    def _update_safe_stride_probe_only(self) -> dict[str, float]:
        """Optimize only SafeStrideHead while enforcing bit-exact frozen state."""
        if self._safe_stride_probe_frozen_state is None:
            self._safe_stride_probe_frozen_state = self._capture_safe_stride_probe_frozen_state()
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
            )

        mean_logs: dict[str, float] = {}
        num_updates = 0
        for batch in generator:
            if batch.observations is None:
                raise RuntimeError("SafeStride probe requires observations in rollout batches.")
            if batch.observations.batch_size[0] == 0:
                continue
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=False,
            )
            loss, logs = self._compute_slow_latent_aux_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            safe_stride_head = cast(Any, self.actor).safe_stride_head
            safe_stride_width_head = getattr(
                self.actor,
                "safe_stride_width_head",
                None,
            )
            probe_parameters = safe_stride_head.parameters()
            if safe_stride_width_head is not None:
                probe_parameters = chain(
                    probe_parameters,
                    safe_stride_width_head.parameters(),
                )
            safe_stride_confidence_head = getattr(
                self.actor,
                "safe_stride_confidence_head",
                None,
            )
            if safe_stride_confidence_head is not None:
                probe_parameters = chain(
                    probe_parameters,
                    safe_stride_confidence_head.parameters(),
                )
            torch.nn.utils.clip_grad_norm_(
                probe_parameters,
                self.max_grad_norm,
            )
            self.optimizer.step()
            num_updates += 1
            for name, value in logs.items():
                mean_logs[name] = mean_logs.get(name, 0.0) + value

        if num_updates == 0:
            raise RuntimeError("SafeStride probe update produced no non-empty mini-batches.")
        for name in mean_logs:
            mean_logs[name] /= num_updates
        self._verify_safe_stride_probe_frozen_state()
        self.storage.clear()
        return {
            "value": 0.0,
            "surrogate": 0.0,
            "entropy": 0.0,
            "safe_stride_probe_only": 1.0,
            "safe_stride_probe_frozen_state_exact": 1.0,
            **mean_logs,
        }

    def _update_geometry_probe_only(self) -> dict[str, float]:
        """Optimize only GeometryProbeHead with a held-out validation subset."""
        if self._geometry_probe_frozen_state is None:
            self._geometry_probe_frozen_state = self._capture_geometry_probe_frozen_state()
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
            )

        mean_logs: dict[str, float] = {}
        num_updates = 0
        for batch in generator:
            if batch.observations is None:
                raise RuntimeError("Geometry Probe requires rollout observations.")
            if batch.observations.batch_size[0] == 0:
                continue
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=False,
            )
            loss, logs = self._compute_slow_latent_aux_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            geometry_probe_head = getattr(self.actor, "geometry_probe_head", None)
            if geometry_probe_head is None:
                raise RuntimeError("Geometry Probe head disappeared during update.")
            torch.nn.utils.clip_grad_norm_(
                geometry_probe_head.parameters(),
                self.max_grad_norm,
            )
            self.optimizer.step()
            num_updates += 1
            for name, value in logs.items():
                mean_logs[name] = mean_logs.get(name, 0.0) + value

        if num_updates == 0:
            raise RuntimeError("Geometry Probe update produced no non-empty mini-batches.")
        for name in mean_logs:
            mean_logs[name] /= num_updates
        self._verify_geometry_probe_frozen_state()
        self.storage.clear()
        return {
            "value": 0.0,
            "surrogate": 0.0,
            "entropy": 0.0,
            "geometry_probe_only": 1.0,
            "geometry_probe_frozen_state_exact": 1.0,
            **mean_logs,
        }

    def _update_teacher_imitation_only(self) -> dict[str, float]:
        """Run optimization using only frozen-teacher imitation loss."""
        if not self.teacher_guidance_enabled:
            raise RuntimeError("Teacher imitation-only training requires teacher guidance to be enabled.")
        if self.teacher is None or not self.teacher_loaded:
            raise RuntimeError("Teacher imitation-only training requires a loaded teacher model.")
        if self.rnd is not None:
            raise NotImplementedError("Teacher imitation-only training does not support RND.")
        if self.symmetry is not None:
            raise NotImplementedError("Teacher imitation-only training does not support symmetry augmentation.")

        mean_entropy = 0.0
        mean_teacher_losses: dict[str, float] = {}

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            if batch.observations is None:
                raise RuntimeError("Teacher imitation-only training requires observations in the rollout batch.")
            original_batch_size = batch.observations.batch_size[0]

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            loss, loss_logs = self._compute_teacher_guidance_loss(
                batch,
                original_batch_size,
                distribution_params,
                loss_weight=self.teacher_imitation_loss_coef,
            )

            self.optimizer.zero_grad()
            loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_entropy += entropy.mean().item()
            for name, value in loss_logs.items():
                mean_teacher_losses[name] = mean_teacher_losses.get(name, 0.0) + value

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_entropy /= num_updates
        for name in mean_teacher_losses:
            mean_teacher_losses[name] /= num_updates

        loss_dict = {
            "value": 0.0,
            "surrogate": 0.0,
            "entropy": mean_entropy,
            "teacher_imitation_only": 1.0,
        }
        loss_dict.update(mean_teacher_losses)
        self.storage.clear()
        return loss_dict

    def save(self) -> dict:
        """Return a dict of all learnable models and teacher-KL training state."""
        saved_dict = super().save()
        saved_dict["teacher_kl_iteration"] = self.teacher_kl_iteration
        saved_dict["teacher_kl_cfg"] = dict(self.teacher_kl_cfg)
        saved_dict["teacher_checkpoint_path"] = self.teacher_checkpoint_path
        saved_dict["teacher_kl_lambda_current"] = self.get_teacher_kl_lambda()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load models and restore the teacher-KL schedule state."""
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_iteration and "teacher_kl_iteration" in loaded_dict:
            self.teacher_kl_iteration = int(loaded_dict["teacher_kl_iteration"])
        return load_iteration

    def train_mode(self) -> None:
        """Set train mode for learnable models while keeping the teacher frozen."""
        if self.safe_stride_probe_only:
            self.actor.eval()
            self.critic.eval()
            cast(Any, self.actor).safe_stride_head.train()
            safe_stride_width_head = getattr(
                self.actor,
                "safe_stride_width_head",
                None,
            )
            if safe_stride_width_head is not None:
                safe_stride_width_head.train()
            safe_stride_confidence_head = getattr(
                self.actor,
                "safe_stride_confidence_head",
                None,
            )
            if safe_stride_confidence_head is not None:
                safe_stride_confidence_head.train()
            self._freeze_teacher()
            return
        if self.geometry_probe_only:
            self.actor.eval()
            self.critic.eval()
            geometry_probe_head = getattr(
                self.actor,
                "geometry_probe_head",
                None,
            )
            if geometry_probe_head is None:
                raise RuntimeError("geometry_probe_only has no geometry Probe head.")
            geometry_probe_head.train()
            self._freeze_teacher()
            return
        super().train_mode()
        self._freeze_teacher()

    def eval_mode(self) -> None:
        """Set evaluation mode for all policy models."""
        super().eval_mode()
        self._freeze_teacher()

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPOTeacherKL:
        """Construct the PPO + teacher-KL algorithm."""
        algorithm_cfg = copy.deepcopy(cfg["algorithm"])
        actor_cfg = copy.deepcopy(cfg["actor"])
        critic_cfg = copy.deepcopy(cfg["critic"])
        teacher_cfg = copy.deepcopy(cfg.get("teacher", cfg["actor"]))
        obs_groups_cfg = copy.deepcopy(cfg["obs_groups"])
        teacher_guidance_cfg = algorithm_cfg.get("teacher_kl_cfg") or {}
        teacher_guidance_enabled = bool(teacher_guidance_cfg.get("enabled", True))

        # Resolve class callables
        alg_class: type[PPOTeacherKL] = resolve_callable(algorithm_cfg.pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(actor_cfg.pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(critic_cfg.pop("class_name"))  # type: ignore
        if teacher_guidance_enabled:
            teacher_class_name = teacher_cfg.pop("class_name", None)
            teacher_class: type[MLPModel] | None = (
                cast(type[MLPModel], resolve_callable(teacher_class_name)) if teacher_class_name else actor_class
            )
        else:
            teacher_class = None

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if teacher_guidance_enabled:
            default_sets.append("teacher")
        if "rnd_cfg" in algorithm_cfg and algorithm_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        obs_groups = resolve_obs_groups(obs, obs_groups_cfg, default_sets)

        # Resolve RND config if used
        algorithm_cfg = resolve_rnd_config(algorithm_cfg, obs, obs_groups, env)

        # Resolve symmetry config if used
        algorithm_cfg = resolve_symmetry_config(algorithm_cfg, env)

        # Initialize the student actor, critic, and frozen teacher actor
        actor: MLPModel = actor_class(obs, obs_groups, "actor", env.num_actions, **actor_cfg).to(device)
        print(f"Actor Model: {actor}")
        if algorithm_cfg.pop("share_cnn_encoders", None):  # Share CNN encoders between actor and critic
            critic_cfg["cnns"] = actor.cnns
        critic: MLPModel = critic_class(obs, obs_groups, "critic", 1, **critic_cfg).to(device)
        print(f"Critic Model: {critic}")
        teacher: MLPModel | None = None
        if teacher_guidance_enabled:
            if teacher_class is None:
                raise RuntimeError("teacher_class should be resolved when teacher guidance is enabled.")
            teacher = teacher_class(obs, obs_groups, "teacher", env.num_actions, **teacher_cfg).to(device)
            if teacher.is_recurrent:
                raise ValueError("PPOTeacherKL currently supports feedforward teacher actors only.")
            print(f"Teacher Model: {teacher}")
        else:
            print("Teacher guidance disabled: training with PPO loss only.")

        # Initialize the storage
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Initialize the algorithm. The teacher is intentionally assigned after construction so it is not part of the
        # PPO optimizer created by the parent class.
        alg: PPOTeacherKL = alg_class(
            actor, critic, storage, device=device, **algorithm_cfg, multi_gpu_cfg=cfg["multi_gpu"]
        )
        if teacher is not None:
            alg.teacher = teacher
            alg.load_teacher_checkpoint()

        # Compile the algorithm's learnable models if requested. The teacher remains an uncompiled frozen reference.
        alg.compile(cfg.get("torch_compile_mode"))

        return alg
