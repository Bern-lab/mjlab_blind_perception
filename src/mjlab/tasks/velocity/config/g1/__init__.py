from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .blind_rough_dwaq_env_cfg import (
  unitree_g1_blind_rough_target_navigation_dwaq_env_cfg,
)
from .rl_cfg import (
  unitree_g1_blind_rough_target_navigation_dwaq_teacherkl_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Blind-Rough-TargetNavigation-DWAQ-TeacherKL-Unitree-G1",
  env_cfg=unitree_g1_blind_rough_target_navigation_dwaq_env_cfg(),
  play_env_cfg=unitree_g1_blind_rough_target_navigation_dwaq_env_cfg(play=True),
  rl_cfg=unitree_g1_blind_rough_target_navigation_dwaq_teacherkl_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
