from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .blind_rough_slow_latent_env_cfg import (
  unitree_g1_blind_rough_target_navigation_semantic_v2_probe_env_cfg,
  unitree_g1_blind_rough_target_navigation_slow_latent_env_cfg,
)
from .blind_rough_step_danger_env_cfg import (
  unitree_g1_blind_rough_target_navigation_step_danger_env_cfg,
)
from .blind_rough_teacher_kl_env_cfg import (
  unitree_g1_blind_rough_teacherkl_env_cfg,
)
from .rl_cfg import (
  unitree_g1_blind_rough_target_navigation_semantic_v2_probe_runner_cfg,
  unitree_g1_blind_rough_target_navigation_semantic_v2_shadow_runner_cfg,
  unitree_g1_blind_rough_target_navigation_slow_latent_teacherkl_runner_cfg,
  unitree_g1_blind_rough_target_navigation_step_danger_teacherkl_runner_cfg,
  unitree_g1_blind_rough_target_navigation_teacherkl_runner_cfg,
  unitree_g1_blind_rough_teacherkl_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1",  # 原teacher
  env_cfg=unitree_g1_blind_rough_teacherkl_env_cfg(),
  play_env_cfg=unitree_g1_blind_rough_teacherkl_env_cfg(play=True),
  rl_cfg=unitree_g1_blind_rough_teacherkl_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Blind-Rough-TargetNavigation-SemanticV2SafeStrideProbe-Unitree-G1"
  ),
  env_cfg=unitree_g1_blind_rough_target_navigation_semantic_v2_probe_env_cfg(),
  play_env_cfg=(
    unitree_g1_blind_rough_target_navigation_semantic_v2_probe_env_cfg(play=True)
  ),
  rl_cfg=unitree_g1_blind_rough_target_navigation_semantic_v2_probe_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Blind-Rough-TargetNavigation-TeacherKL-Unitree-G1",  # 加target的普通kl
  env_cfg=unitree_g1_blind_rough_teacherkl_env_cfg(use_target_navigation=True),
  play_env_cfg=unitree_g1_blind_rough_teacherkl_env_cfg(
    play=True,
    use_target_navigation=True,
  ),
  rl_cfg=unitree_g1_blind_rough_target_navigation_teacherkl_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Blind-Rough-TargetNavigation-StepDanger-TeacherKL-Unitree-G1",
  env_cfg=unitree_g1_blind_rough_target_navigation_step_danger_env_cfg(),
  play_env_cfg=unitree_g1_blind_rough_target_navigation_step_danger_env_cfg(
    play=True,
  ),
  rl_cfg=unitree_g1_blind_rough_target_navigation_step_danger_teacherkl_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1",
  env_cfg=unitree_g1_blind_rough_target_navigation_slow_latent_env_cfg(),
  play_env_cfg=unitree_g1_blind_rough_target_navigation_slow_latent_env_cfg(
    play=True,
  ),
  rl_cfg=unitree_g1_blind_rough_target_navigation_slow_latent_teacherkl_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=(
    "Mjlab-Velocity-Blind-Rough-TargetNavigation-SemanticV2Shadow-TeacherKL-Unitree-G1"
  ),
  env_cfg=unitree_g1_blind_rough_target_navigation_slow_latent_env_cfg(),
  play_env_cfg=unitree_g1_blind_rough_target_navigation_slow_latent_env_cfg(
    play=True,
  ),
  rl_cfg=unitree_g1_blind_rough_target_navigation_semantic_v2_shadow_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
