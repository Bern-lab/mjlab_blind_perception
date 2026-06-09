"""LSTM Teacher-KL Unitree G1 blind-rough velocity environment config."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains.config import BLIND_HIGH_STAIRS_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from .blind_rough_teacher_kl_env_cfg import (
  _add_teacher_depth_camera,
  _configure_teacherkl_target_navigation,
  _make_teacher_terms,
  configure_blind_teacherkl_play_visualization,
)
from .blind_rough_toe_contact_cfg import (
  configure_g1_toe_riser_contact_memory_penalty,
)
from .env_cfgs import (
  UniformVelocityCommandCfg,
  configure_g1_high_stairs_play_randomization,
  configure_g1_high_stairs_play_terrain_generator,
  unitree_g1_rough_env_cfg,
)


def _lstm_teacherkl_play_terrain_cfg():
  terrain_cfg = deepcopy(BLIND_HIGH_STAIRS_TERRAINS_CFG)
  return configure_g1_high_stairs_play_terrain_generator(terrain_cfg)


def _configure_lstm_teacherkl_student_env(
  cfg: ManagerBasedRlEnvCfg, play: bool
) -> None:
  robot_cfg = deepcopy(cfg.scene.entities["robot"])
  assert robot_cfg.articulation is not None
  for actuator_cfg in robot_cfg.articulation.actuators:
    actuator_cfg.delay_min_lag = 0
    actuator_cfg.delay_max_lag = 2
    actuator_cfg.delay_hold_prob = 0.8
    actuator_cfg.delay_update_period = 5
  cfg.scene.entities["robot"] = robot_cfg

  cfg.sim.nconmax = 256
  if hasattr(cfg.sim, "nccdmax"):
    cfg.sim.nccdmax = None
  cfg.sim.njmax = 4096
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 400
  cfg.sim.use_cuda_graph = True

  cfg.observations["actor"].terms.pop("height_scan", None)
  configure_g1_toe_riser_contact_memory_penalty(cfg)

  cfg.observations["actor"].history_length = 5
  cfg.observations["critic"].history_length = 3

  actor_terms = cfg.observations["actor"].terms
  for term_name in (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos_rel",
    "joint_vel_rel",
  ):
    actor_terms[term_name].delay_min_lag = 0
    actor_terms[term_name].delay_max_lag = 2
    actor_terms[term_name].delay_hold_prob = 0.8
    actor_terms[term_name].delay_update_period = 5

  actor_terms["base_ang_vel"].noise = Unoise(n_min=-0.3, n_max=0.3)
  actor_terms["projected_gravity"].noise = Unoise(n_min=-0.07, n_max=0.07)
  actor_terms["joint_pos_rel"].noise = Unoise(n_min=-0.015, n_max=0.015)
  actor_terms["joint_vel_rel"].noise = Unoise(n_min=-2.0, n_max=2.0)

  if cfg.scene.terrain is not None:
    cfg.scene.terrain.terrain_generator = (
      _lstm_teacherkl_play_terrain_cfg()
      if play
      else deepcopy(BLIND_HIGH_STAIRS_TERRAINS_CFG)
    )
    cfg.scene.terrain.max_init_terrain_level = 2

  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (0.0, 0.8),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.5, 0.5),
      },
      {
        "step": 3000 * 24,
        "lin_vel_x": (0.0, 1.2),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.8, 0.8),
      },
    ]

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.resampling_time_range = (7.0, 12.0)

  cfg.rewards["joint_acc_l2"] = RewardTermCfg(
    func=mdp.joint_acc_l2,
    weight=-2.5e-7,
  )
  cfg.rewards["action_acc_l2"] = RewardTermCfg(
    func=mdp.action_acc_l2,
    weight=-0.05,
  )
  cfg.rewards["body_ang_vel"].weight = -0.08
  cfg.rewards["angular_momentum"].weight = -0.03


def _configure_lstm_teacherkl_play_env(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.episode_length_s = int(1e9)
  cfg.events.pop("push_robot", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum = {}

  cfg.events.pop("randomize_terrain", None)
  configure_g1_high_stairs_play_randomization(cfg)
  configure_blind_teacherkl_play_visualization(cfg)


def unitree_g1_blind_rough_lstm_teacherkl_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the old blind-rough LSTM student env used by 5.25 checkpoints."""
  cfg = unitree_g1_rough_env_cfg(play=play)
  _configure_lstm_teacherkl_student_env(cfg, play=play)
  _add_teacher_depth_camera(cfg)
  _configure_teacherkl_target_navigation(cfg, play=play)

  cfg.observations["teacher"] = ObservationGroupCfg(
    terms=_make_teacher_terms(cfg),
    concatenate_terms=True,
    enable_corruption=False,
    history_length=None,
    flatten_history_dim=True,
  )

  if play:
    _configure_lstm_teacherkl_play_env(cfg)

  return cfg
