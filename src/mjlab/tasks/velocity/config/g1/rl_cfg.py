"""RL configuration for Unitree G1 velocity task."""

from dataclasses import dataclass, field

from mjlab.rl import (
  RslRlGatedStairLatentModelCfg,
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
  RslRlPpoTeacherKLAlgorithmCfg,
  RslRlTeacherKLCfg,
  RslRlTeacherKLRunnerCfg,
)

G1_TEACHER_KL_CHECKPOINT = (
  "teacher_policies/g1_target_heading_depth_teacher/model_118200.pt"
)
G1_LSTM_TEACHER_KL_NUM_STEPS_PER_ENV = 36

_DEPTH_CNN_CFG = {
  "output_channels": [16, 32],
  "kernel_size": [5, 3],
  "stride": [2, 2],
  "padding": "zeros",
  "activation": "elu",
  "max_pool": False,
  "global_pool": "none",
  "spatial_softmax": True,
  "spatial_softmax_temperature": 1.0,
}
_DEPTH_MODEL_CLS = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"


@dataclass(frozen=True)
class G1SlowLatentPolicyModelParams:
  """Actor architecture, gated latent memory, and auxiliary-loss knobs."""

  hidden_dims: tuple[int, ...] = (512, 256, 128)
  """MLP actor hidden dimensions after concatenating actor_obs and z_memory."""
  activation: str = "elu"
  """Activation used by the actor MLP and latent encoder MLP."""
  obs_normalization: bool = True
  """Normalize actor and latent observations with empirical normalizers."""
  action_std_init: float = 1.0
  """Initial Gaussian action standard deviation."""
  action_std_type: str = "scalar"
  """Distribution std parameterization passed to RSL-RL."""
  latent_dim: int = 16
  """Dimension of the persistent slow latent memory z_t."""
  latent_hidden_dim: int = 128
  """LSTM hidden size for the latent encoder and recurrent rollout storage."""
  mlp_encoder_dims: tuple[int, ...] = (128, 128)
  """Pre-LSTM MLP encoder dimensions for stair_latent observations."""
  alpha_fast: float = 0.3
  """EMA update rate in normal fast-update mode."""
  alpha_write: float = 0.8
  """EMA update rate while writing stair evidence into memory."""
  alpha_hold: float = 0.01  # 原本0.02
  """EMA update rate while holding stair memory."""
  write_steps: int = 2
  """Number of steps spent in write mode after an event trigger."""
  min_stair_steps: int = 50
  """Minimum memory-hold steps before exit is allowed."""
  exit_steps: int = 100
  """Consecutive no-event steps required to leave stair-memory mode."""
  cooldown_steps: int = 15
  """Cooldown steps after exiting memory before another trigger is accepted."""
  event_on_threshold: float = 0.6
  """Event probability threshold that triggers stair-memory writing."""
  event_off_threshold: float = 0.4
  """Event probability threshold counted as no-event during exit logic."""
  stair_off_threshold: float = 0.4
  """Stair-state probability threshold below which exit is allowed."""
  aux_event_coef: float = 0.03
  """BCE loss weight for current toe-riser event prediction."""
  aux_stair_coef: float = 0.02
  """BCE loss weight for current stair-state prediction."""
  aux_future_collision_coef: float = 0.05
  """BCE loss weight for future toe-riser collision prediction."""
  future_collision_horizon: int = 20
  """Future window length, in policy steps, for collision labels."""
  latent_obs_set: str = "latent"
  """Observation set name consumed by the latent encoder."""


@dataclass(frozen=True)
class G1SlowLatentRunnerParams:
  """Top-level RSL-RL settings for the slow-latent main experiment."""

  num_steps_per_env: int = 64
  """Rollout length per environment before PPO update."""
  save_interval: int = 1000
  """Checkpoint interval in training iterations."""
  max_iterations: int = 40_001
  """Maximum PPO training iterations."""
  experiment_name: str = "g1_blind_rough_target_navigation_slow_latent_teacherkl"
  """Log directory experiment name."""
  model: G1SlowLatentPolicyModelParams = field(
    default_factory=G1SlowLatentPolicyModelParams
  )
  """Slow-latent actor/model parameters."""


def _unitree_g1_policy_model_cfg() -> RslRlModelCfg:
  return RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )


def _unitree_g1_depth_policy_model_cfg() -> RslRlModelCfg:
  return RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    cnn_cfg=_DEPTH_CNN_CFG,
    class_name=_DEPTH_MODEL_CLS,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )


def _unitree_g1_gated_stair_latent_policy_model_cfg(
  hidden_dims: tuple[int, ...] = (512, 256, 128),
  activation: str = "elu",
  obs_normalization: bool = True,
  action_std_init: float = 1.0,
  action_std_type: str = "scalar",
  latent_dim: int = 16,
  latent_hidden_dim: int = 128,
  mlp_encoder_dims: tuple[int, ...] = (128, 128),
  alpha_fast: float = 0.3,
  alpha_write: float = 0.8,
  alpha_hold: float = 0.02,
  write_steps: int = 2,
  min_stair_steps: int = 50,
  exit_steps: int = 100,
  cooldown_steps: int = 15,
  event_on_threshold: float = 0.6,
  event_off_threshold: float = 0.4,
  stair_off_threshold: float = 0.4,
  aux_event_coef: float = 0.03,
  aux_stair_coef: float = 0.02,
  aux_future_collision_coef: float = 0.05,
  future_collision_horizon: int = 20,
  latent_obs_set: str = "latent",
) -> RslRlGatedStairLatentModelCfg:
  return RslRlGatedStairLatentModelCfg(
    hidden_dims=hidden_dims,
    activation=activation,
    obs_normalization=obs_normalization,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": action_std_init,
      "std_type": action_std_type,
    },
    latent_dim=latent_dim,
    latent_hidden_dim=latent_hidden_dim,
    mlp_encoder_dims=mlp_encoder_dims,
    alpha_fast=alpha_fast,
    alpha_write=alpha_write,
    alpha_hold=alpha_hold,
    write_steps=write_steps,
    min_stair_steps=min_stair_steps,
    exit_steps=exit_steps,
    cooldown_steps=cooldown_steps,
    event_on_threshold=event_on_threshold,
    event_off_threshold=event_off_threshold,
    stair_off_threshold=stair_off_threshold,
    aux_event_coef=aux_event_coef,
    aux_stair_coef=aux_stair_coef,
    aux_future_collision_coef=aux_future_collision_coef,
    future_collision_horizon=future_collision_horizon,
    latent_obs_set=latent_obs_set,
    rnn_type="lstm",
    rnn_hidden_dim=latent_hidden_dim,
    rnn_num_layers=1,
  )


def _unitree_g1_lstm_policy_model_cfg() -> RslRlModelCfg:
  return RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    rnn_type="lstm",
    rnn_hidden_dim=256,
    rnn_num_layers=1,
    class_name="RNNModel",
  )


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=_unitree_g1_policy_model_cfg(),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=40_001,
  )


def unitree_g1_target_heading_teacher_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create PPO config for the target-heading teacher policy task."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.actor = _unitree_g1_depth_policy_model_cfg()
  cfg.obs_groups = {
    "actor": ("actor", "camera"),
    "critic": ("critic",),
  }
  cfg.experiment_name = "g1_velocity_target_heading_teacher_depth"
  return cfg


def unitree_g1_blind_rough_teacherkl_runner_cfg() -> RslRlTeacherKLRunnerCfg:
  """Create PPO + frozen-teacher-KL config for Unitree G1 blind rough training."""
  return RslRlTeacherKLRunnerCfg(
    actor=_unitree_g1_policy_model_cfg(),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    teacher=_unitree_g1_depth_policy_model_cfg(),
    algorithm=RslRlPpoTeacherKLAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      teacher_kl_cfg=RslRlTeacherKLCfg(
        enabled=True,
        imitation_only=False,
        imitation_loss_coef=1.0,
        checkpoint_path=G1_TEACHER_KL_CHECKPOINT,
        loss_type="mean_huber",
        lambda_start=0.05,
        lambda_end=0.0,
        warmup_iters=0,
        constant_iters=0,
        anneal_iters=10000,
        schedule="cosine",
        huber_delta=0.5,
        max_teacher_loss=None,
        max_kl_loss=None,
        max_kl_loss_tail_slope=0.0,
        check_shapes=True,
        fail_on_nonfinite_kl=True,
        debug_shapes=False,
      ),
    ),
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
      "teacher": ("teacher", "camera"),
    },
    experiment_name="g1_blind_rough_teacherkl",
    save_interval=1000,
    num_steps_per_env=24,
    max_iterations=40_001,
  )


def unitree_g1_blind_rough_target_navigation_teacherkl_runner_cfg() -> (
  RslRlTeacherKLRunnerCfg
):
  """Create Teacher-KL config for blind rough target-navigation training."""
  cfg = unitree_g1_blind_rough_teacherkl_runner_cfg()
  cfg.experiment_name = "g1_blind_rough_target_navigation_teacherkl"
  return cfg


def unitree_g1_blind_rough_target_navigation_step_danger_teacherkl_runner_cfg() -> (
  RslRlTeacherKLRunnerCfg
):
  """Create Teacher-KL config for target navigation with full step danger rewards."""
  cfg = unitree_g1_blind_rough_target_navigation_teacherkl_runner_cfg()
  cfg.experiment_name = "g1_blind_rough_target_navigation_step_danger_teacherkl"
  return cfg


def unitree_g1_blind_rough_target_navigation_slow_latent_teacherkl_runner_cfg(
  num_steps_per_env: int = 64,
  latent_dim: int = 16,
  latent_hidden_dim: int = 128,
  mlp_encoder_dims: tuple[int, ...] = (128, 128),
  alpha_fast: float = 0.3,
  alpha_write: float = 0.8,
  alpha_hold: float = 0.02,
  write_steps: int = 2,
  min_stair_steps: int = 50,
  exit_steps: int = 100,
  cooldown_steps: int = 15,
  event_on_threshold: float = 0.6,
  event_off_threshold: float = 0.4,
  stair_off_threshold: float = 0.4,
  aux_event_coef: float = 0.03,
  aux_stair_coef: float = 0.02,
  aux_future_collision_coef: float = 0.05,
  future_collision_horizon: int = 20,
  latent_obs_set: str = "latent",
  params: G1SlowLatentRunnerParams | None = None,
) -> RslRlTeacherKLRunnerCfg:
  """Create Teacher-KL config with gated stair slow-latent student actor."""
  experiment_name = "g1_blind_rough_target_navigation_slow_latent_teacherkl"
  save_interval: int | None = None
  max_iterations: int | None = None
  hidden_dims = (512, 256, 128)
  activation = "elu"
  obs_normalization = True
  action_std_init = 1.0
  action_std_type = "scalar"

  if params is not None:
    model_params = params.model
    num_steps_per_env = params.num_steps_per_env
    save_interval = params.save_interval
    max_iterations = params.max_iterations
    experiment_name = params.experiment_name
    hidden_dims = model_params.hidden_dims
    activation = model_params.activation
    obs_normalization = model_params.obs_normalization
    action_std_init = model_params.action_std_init
    action_std_type = model_params.action_std_type
    latent_dim = model_params.latent_dim
    latent_hidden_dim = model_params.latent_hidden_dim
    mlp_encoder_dims = model_params.mlp_encoder_dims
    alpha_fast = model_params.alpha_fast
    alpha_write = model_params.alpha_write
    alpha_hold = model_params.alpha_hold
    write_steps = model_params.write_steps
    min_stair_steps = model_params.min_stair_steps
    exit_steps = model_params.exit_steps
    cooldown_steps = model_params.cooldown_steps
    event_on_threshold = model_params.event_on_threshold
    event_off_threshold = model_params.event_off_threshold
    stair_off_threshold = model_params.stair_off_threshold
    aux_event_coef = model_params.aux_event_coef
    aux_stair_coef = model_params.aux_stair_coef
    aux_future_collision_coef = model_params.aux_future_collision_coef
    future_collision_horizon = model_params.future_collision_horizon
    latent_obs_set = model_params.latent_obs_set

  cfg = unitree_g1_blind_rough_target_navigation_teacherkl_runner_cfg()
  cfg.actor = _unitree_g1_gated_stair_latent_policy_model_cfg(
    hidden_dims=hidden_dims,
    activation=activation,
    obs_normalization=obs_normalization,
    action_std_init=action_std_init,
    action_std_type=action_std_type,
    latent_dim=latent_dim,
    latent_hidden_dim=latent_hidden_dim,
    mlp_encoder_dims=mlp_encoder_dims,
    alpha_fast=alpha_fast,
    alpha_write=alpha_write,
    alpha_hold=alpha_hold,
    write_steps=write_steps,
    min_stair_steps=min_stair_steps,
    exit_steps=exit_steps,
    cooldown_steps=cooldown_steps,
    event_on_threshold=event_on_threshold,
    event_off_threshold=event_off_threshold,
    stair_off_threshold=stair_off_threshold,
    aux_event_coef=aux_event_coef,
    aux_stair_coef=aux_stair_coef,
    aux_future_collision_coef=aux_future_collision_coef,
    future_collision_horizon=future_collision_horizon,
    latent_obs_set=latent_obs_set,
  )
  cfg.obs_groups = {
    "actor": ("actor",),
    "latent": (latent_obs_set,),
    "critic": ("critic",),
    "teacher": ("teacher", "camera"),
  }
  cfg.num_steps_per_env = num_steps_per_env
  if save_interval is not None:
    cfg.save_interval = save_interval
  if max_iterations is not None:
    cfg.max_iterations = max_iterations
  cfg.experiment_name = experiment_name
  return cfg


def unitree_g1_blind_rough_lstm_teacherkl_runner_cfg(
  num_steps_per_env: int = G1_LSTM_TEACHER_KL_NUM_STEPS_PER_ENV,
) -> RslRlTeacherKLRunnerCfg:
  """Create Teacher-KL config for the old blind-rough RNNModel LSTM actor."""
  cfg = unitree_g1_blind_rough_teacherkl_runner_cfg()
  cfg.actor = _unitree_g1_lstm_policy_model_cfg()
  cfg.num_steps_per_env = num_steps_per_env
  cfg.save_interval = 1000
  cfg.experiment_name = "g1_blind_rough_lstm_teacherkl"
  return cfg
