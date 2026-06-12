from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .blind_rough_perception_env_cfg import (
  unitree_g1_blind_rough_perception_env_cfg,
  unitree_g1_blind_rough_perception_ppo_env_cfg,
)
from .blind_rough_teacher_kl_env_cfg import (
  unitree_g1_blind_rough_teacherkl_env_cfg,
)
from .rl_cfg import (
  unitree_g1_blind_rough_perception_ppo_runner_cfg,
  unitree_g1_blind_rough_perception_teacherkl_runner_cfg,
  unitree_g1_blind_rough_target_navigation_perception_teacherkl_runner_cfg,
  unitree_g1_blind_rough_teacherkl_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1",
  env_cfg=unitree_g1_blind_rough_teacherkl_env_cfg(),
  play_env_cfg=unitree_g1_blind_rough_teacherkl_env_cfg(play=True),
  rl_cfg=unitree_g1_blind_rough_teacherkl_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Blind-Rough-Perception-TeacherKL-Unitree-G1",
  env_cfg=unitree_g1_blind_rough_perception_env_cfg(),
  play_env_cfg=unitree_g1_blind_rough_perception_env_cfg(play=True),
  rl_cfg=unitree_g1_blind_rough_perception_teacherkl_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Blind-Rough-Perception-PPO-Unitree-G1",
  env_cfg=unitree_g1_blind_rough_perception_ppo_env_cfg(use_target_navigation=True),
  play_env_cfg=unitree_g1_blind_rough_perception_ppo_env_cfg(
    play=True,
    use_target_navigation=True,
  ),
  rl_cfg=unitree_g1_blind_rough_perception_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Blind-Rough-TargetNavigation-Perception-TeacherKL-Unitree-G1",
  env_cfg=unitree_g1_blind_rough_perception_env_cfg(use_target_navigation=True),
  play_env_cfg=unitree_g1_blind_rough_perception_env_cfg(
    play=True,
    use_target_navigation=True,
  ),
  rl_cfg=unitree_g1_blind_rough_target_navigation_perception_teacherkl_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
