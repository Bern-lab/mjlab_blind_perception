"""RSL-RL configuration."""

from dataclasses import dataclass, field
from typing import Any, Literal, Tuple


@dataclass
class RslRlModelCfg:
  """Config for a single neural network model (Actor or Critic)."""

  hidden_dims: Tuple[int, ...] = (128, 128, 128)
  """The hidden dimensions of the network."""
  activation: str = "elu"
  """The activation function."""
  obs_normalization: bool = False
  """Whether to normalize the observations. Default is False."""
  cnn_cfg: dict[str, Any] | None = None
  """CNN encoder config. When set, class_name should be "CNNModel".

  Passed to ``rsl_rl.modules.CNN``. Common keys: output_channels,
  kernel_size, stride, padding, activation, global_pool, max_pool.
  """
  distribution_cfg: dict[str, Any] | None = None
  """Distribution config dict passed to rsl_rl. Example::

    {"class_name": "GaussianDistribution",
     "init_std": 1.0, "std_type": "scalar"}

  ``None`` means deterministic output (use for critic).
  """
  rnn_type: str | None = None
  """RNN type ("lstm" or "gru"). When set, class_name should be "RNNModel"."""
  rnn_hidden_dim: int = 256
  """Hidden state dimension for the RNN."""
  rnn_num_layers: int = 1
  """Number of stacked RNN layers."""
  class_name: str = "MLPModel"
  """Model class name resolved by RSL-RL (MLPModel, CNNModel, or RNNModel)."""


@dataclass
class RslRlSlowLatentModelCfg(RslRlModelCfg):
  """Config for an LSTM encoder + slow latent + MLP actor model."""

  class_name: str = "mjlab.rl.slow_latent_model:LSTMSlowLatentMLPModel"
  """Qualified model class name for the slow-latent actor."""
  use_slow_latent: bool = True
  """Enable explicit slow terrain latent conditioning."""
  latent_dim: int = 16
  """Dimension of the slow terrain latent z_t."""
  latent_hidden_dim: int = 256
  """Hidden dimension of the latent encoder LSTM."""
  latent_alpha: float = 0.1
  """EMA update rate for z_t. Set to 1.0 for the no-slow-update ablation."""
  encoder_type: Literal["lstm"] = "lstm"
  """Recurrent encoder type. The first implementation uses LSTM."""
  rnn_type: str | None = "lstm"
  """Keep RSL-RL recurrent storage enabled for the latent encoder."""
  rnn_hidden_dim: int = 256
  """RSL-RL-style recurrent field; use latent_hidden_dim for this model."""


@dataclass
class RslRlGatedStairLatentModelCfg(RslRlSlowLatentModelCfg):
  """Config for the gated stair-focused slow latent model with auxiliary heads.

  Same architecture as RslRlSlowLatentModelCfg but with:
    - Gated update (fast/write/hold) controlled by an internal state machine
    - Auxiliary event, stair-state, future-collision, and stair-shape heads
    - Separate latent observation group (deployable proprioceptive features only)
  """

  use_gated_latent: bool = True
  """Enable gated latent update with state machine."""
  use_stair_latent_obs: bool = True
  """Use stair-focused deployable latent_obs instead of full actor_obs."""
  latent_obs_set: str = "latent"
  """Observation set used by the latent encoder."""

  # ---- Gate parameters ----
  alpha_fast: float = 0.3
  """EMA rate in NORMAL mode (fast update, fast forget)."""
  alpha_write: float = 0.8
  """EMA rate in STAIR_WRITE mode (rapid memory acquisition)."""
  state_latent_dim: int = 8
  """Leading z-memory channels reserved for stair state and timing."""
  alpha_hold_state: float = 0.0
  """Freeze rate for stable stair-state and timing memory."""
  alpha_hold_shape: float = 0.05
  """Hold update rate for continuously refreshed geometry/stride memory."""
  memory_event_shape_boost_steps: int = 15
  """Frames of shape-only rapid updating after an event detected in memory."""
  write_steps: int = 6
  """Number of steps to stay in STAIR_WRITE mode."""
  stair_confirm_steps: int = 3
  """Minimum WRITE frames with stair evidence required to enter STAIR_MEMORY."""
  min_stair_steps: int = 30
  """Minimum steps in STAIR_MEMORY before exit is allowed."""
  exit_steps: int = 40
  """Consecutive stair-off steps required to exit STAIR_MEMORY."""
  cooldown_steps: int = 15
  """Cooldown steps after exiting STAIR_MEMORY before re-triggering."""

  # ---- Prediction thresholds ----
  event_on_threshold: float = 0.60
  """p_event threshold to trigger STAIR_WRITE."""
  event_off_threshold: float = 0.20
  """p_event threshold required to re-arm triggering after a gate transition."""
  stair_on_threshold: float = 0.35
  """p_stair threshold counted as confirming evidence during STAIR_WRITE."""
  stair_off_threshold: float = 0.20
  """p_stair threshold counted as exit evidence during STAIR_MEMORY."""

  # ---- Auxiliary loss ----
  aux_future_collision_risk_coef: float = 0.03
  """Weight for continuous future collision-risk Huber loss."""
  aux_future_safe_landing_quality_coef: float = 0.03
  """Weight for continuous next-touchdown quality Huber loss."""
  aux_event_coef: float = 0.03
  """Weight for current toe-riser event BCE loss."""
  aux_event_pos_weight: float = 50.0
  """Positive-class weight for the sparse stair-entry event BCE loss."""
  event_label_window_steps: int = 4
  """Number of frames over which one-shot stair-entry event labels stay positive."""
  aux_stair_coef: float = 0.05
  """Weight for stair state BCE loss."""
  aux_stair_pos_weight: float = 3.0
  """Positive-class weight for stair-state BCE loss."""
  aux_stair_shape_coef: float = 0.0
  """Weight for masked tread-depth/riser-height Huber loss."""
  aux_safe_stride_coef: float = 0.03
  """Weight for the Event-to-layer2 minimum-safe-stride Huber loss."""
  stair_shape_huber_delta: float = 0.05
  """Huber transition point for stair geometry labels normalized to [0, 1]."""
  safe_stride_huber_delta: float = 0.05
  """Huber transition point for safe-stride labels, in meters."""
  safe_stride_width_loss_coef: float = 1.0
  """Weight for independently normalized SafeStride width regression."""
  safe_stride_lower_shortfall_coef: float = 1.0
  """Extra loss multiplier when SafeStride predicts below the safe lower bound."""
  safe_stride_interval_coverage_loss_coef: float = 0.0
  """Weight for covering the privileged SafeStride interval with decoded bounds."""
  safe_stride_interval_coverage_margin: float = 0.01
  """Slack, in meters, before interval-coverage loss is applied."""
  safe_stride_confidence_loss_coef: float = 0.30
  """Weight for SafeStride interval-confidence BCE inside the SafeStride loss."""
  safe_stride_std_floor_loss_coef: float = 0.0
  """Weight for penalizing collapsed SafeStride prediction spread."""
  safe_stride_centered_loss_coef: float = 0.0
  """Weight for centered SafeStride regression that preserves label variation."""
  safe_stride_std_floor_ratio: float = 0.70
  """Minimum desired SafeStride prediction std as a fraction of label std."""
  safe_stride_deployable_hint_loss_coef: float = 0.0
  """Weight for one-sided lower-bound hints from deployable foot-event summary."""
  safe_stride_deployable_hint_margin: float = 0.02
  """Slack, in meters, before deployable lower-bound hints are penalized."""
  safe_stride_min: float = 0.10
  """Minimum decoded safe stride, in meters."""
  safe_stride_max: float = 0.55
  """Maximum decoded safe stride, in meters."""
  structured_safe_stride_enabled: bool = False
  """Predict ordered SafeStride bounds instead of the legacy scalar point."""
  dynamic_stair_shape_enabled: bool = False
  """Decode stair size from geometry memory plus current recurrent context."""
  dynamic_safe_stride_enabled: bool = False
  """Decode SafeStride from current recurrent state, geometry memory, and gait phase."""
  safe_stride_phase_dim: int = 0
  """Number of deployable latent-observation gait-phase channels."""
  safe_stride_phase_start: int = -1
  """Start index for gait phase, or -1 to use the trailing channels."""
  shadow_semantic_enabled: bool = False
  """Record a fixed-position semantic16 shadow vector without actor feedback."""
  actor_semantic_enabled: bool = False
  """Append detached semantic16 predictions to the actor conditioning."""
  geometry_probe_input: Literal["none", "shape", "hidden", "combined"] = "none"
  """Frozen-latent geometry Probe input: shape8, recurrent h_t, or both."""
  future_risk_weight_scale: float = 2.0
  """Extra Huber weight applied in proportion to the risk label."""
  future_quality_weight_scale: float = 2.0
  """Extra Huber weight applied in proportion to the quality label."""
  future_risk_huber_delta: float = 0.1
  """Huber transition point for normalized future risk."""
  future_quality_huber_delta: float = 0.1
  """Huber transition point for normalized next-touchdown quality."""
  future_horizon: int = 20
  """Future window for maximum risk and first-touchdown quality labels."""

  # ---- MLP encoder ----
  mlp_encoder_dims: tuple[int, ...] = (128, 128)
  """Hidden dimensions of the pre-LSTM MLP encoder for latent_obs."""

  # Override parent defaults
  latent_dim: int = 24
  latent_hidden_dim: int = 128
  rnn_hidden_dim: int = 128


@dataclass
class RslRlDwaqModelCfg(RslRlModelCfg):
  """Config for a DWAQ beta-VAE context encoder + MLP actor model."""

  class_name: str = "mjlab.rl.dwaq_model:DWAQMLPModel"
  """Qualified model class name for the DWAQ actor."""
  history_obs_set: str = "dwaq_history"
  """Observation set used by the DWAQ context encoder."""
  encoder_hidden_dims: Tuple[int, ...] = (128, 64)
  """Hidden dimensions of the DWAQ context encoder."""
  decoder_hidden_dims: Tuple[int, ...] = (64, 128)
  """Hidden dimensions of the DWAQ decoder."""
  velocity_dim: int = 3
  """Velocity-estimation channels in the DWAQ code."""
  latent_dim: int = 16
  """Unsupervised latent channels in the DWAQ code."""
  cenet_out_dim: int | None = 19
  """DWAQ code dimension, matching velocity_dim + latent_dim in G1DWAQ_Lab."""
  sample_code_in_eval: bool = False
  """Whether deterministic eval/export should sample the VAE code."""


@dataclass
class RslRlPpoAlgorithmCfg:
  """Config for the PPO algorithm."""

  num_learning_epochs: int = 5
  """The number of learning epochs per update."""
  num_mini_batches: int = 4
  """The number of mini-batches per update.
  mini batch size = num_envs * num_steps / num_mini_batches
  """
  learning_rate: float = 1e-3
  """The learning rate."""
  schedule: Literal["adaptive", "fixed"] = "adaptive"
  """The learning rate schedule."""
  gamma: float = 0.99
  """The discount factor."""
  lam: float = 0.95
  """The lambda parameter for Generalized Advantage Estimation (GAE)."""
  entropy_coef: float = 0.005
  """The coefficient for the entropy loss."""
  desired_kl: float = 0.01
  """The desired KL divergence between the new and old policies."""
  max_grad_norm: float = 1.0
  """The maximum gradient norm for the policy."""
  value_loss_coef: float = 1.0
  """The coefficient for the value loss."""
  use_clipped_value_loss: bool = True
  """Whether to use clipped value loss."""
  clip_param: float = 0.2
  """The clipping parameter for the policy."""
  normalize_advantage_per_mini_batch: bool = False
  """Whether to normalize the advantage per mini-batch. Default is False. If True, the
  advantage is normalized over the mini-batches only. Otherwise, the advantage is
  normalized over the entire collected trajectories.
  """
  optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
  """The optimizer to use."""
  share_cnn_encoders: bool = False
  """Share CNN encoders between actor and critic."""
  rnd_cfg: dict[str, Any] | None = None
  """Optional Random Network Distillation extension config."""
  symmetry_cfg: dict[str, Any] | None = None
  """Optional symmetry extension config."""
  class_name: str = "PPO"
  """Algorithm class name resolved by RSL-RL."""


@dataclass
class RslRlTeacherKLCfg:
  """Config for frozen-teacher guidance regularization."""

  enabled: bool = True
  """Whether to enable frozen-teacher guidance. False makes PPOTeacherKL run as pure PPO."""
  imitation_only: bool = False
  """Whether to train only from teacher imitation loss and skip PPO surrogate/value losses."""
  imitation_loss_coef: float = 1.0
  """Loss coefficient used when ``imitation_only=True``."""
  checkpoint_path: str | None = None
  """Path to the rsl-rl teacher checkpoint containing ``actor_state_dict``."""
  loss_type: Literal["kl", "mean_mse", "mean_huber"] = "kl"
  """Teacher guidance loss: full distribution KL, action-mean MSE, or action-mean Huber."""
  lambda_start: float = 0.8
  """Initial weight for the teacher guidance loss."""
  lambda_end: float = 0.0
  """Final weight for the teacher guidance loss."""
  warmup_iters: int = 0
  """Number of iterations with zero teacher guidance loss before the schedule starts."""
  constant_iters: int = 0
  """Number of iterations to hold ``lambda_start`` for constant_then_linear."""
  anneal_iters: int = 10000
  """Number of iterations used by linear/cosine annealing."""
  schedule: Literal["linear", "cosine", "constant", "constant_then_linear"] = "cosine"
  """Teacher guidance weight schedule."""
  huber_delta: float = 1.0
  """Delta parameter for ``loss_type='mean_huber'``."""
  teacher_forward_chunk_size: int = 4096
  """Maximum flattened samples per frozen-teacher forward pass."""
  max_teacher_loss: float | None = None
  """Optional hard cap applied to mean-only teacher guidance losses."""
  max_kl_loss: float | None = 10.0
  """Optional cap applied to the KL value used in the loss when ``loss_type='kl'``."""
  max_kl_loss_tail_slope: float = 0.0
  """Slope to keep above ``max_kl_loss``. 0.0 keeps the existing hard cap."""
  check_shapes: bool = True
  """Whether to check teacher/student distribution parameter shapes once."""
  fail_on_nonfinite_kl: bool = True
  """Whether to raise if the teacher KL becomes NaN or Inf."""
  debug_shapes: bool = False
  """Print distribution parameter shapes when checking them."""
  log_kl_when_lambda_zero: bool = True
  """Whether to keep evaluating teacher diagnostics after guidance weight is zero."""


@dataclass
class RslRlPpoTeacherKLAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """Config for PPO with frozen-teacher KL regularization."""

  class_name: str = "rsl_rl.algorithms.ppo_teacher_kl:PPOTeacherKL"
  """Algorithm class name resolved by RSL-RL."""
  teacher_kl_cfg: RslRlTeacherKLCfg = field(default_factory=RslRlTeacherKLCfg)
  """Frozen-teacher KL configuration."""
  safe_stride_probe_only: bool = False
  """Freeze the policy and optimize only the SafeStride decoder."""
  safe_stride_probe_learning_rate: float = 1.0e-3
  """Learning rate for the isolated SafeStride decoder probe."""
  geometry_probe_only: bool = False
  """Freeze the policy and optimize only the independent geometry Probe."""
  geometry_probe_learning_rate: float = 1.0e-3
  """Learning rate for the frozen-latent geometry Probe."""
  geometry_probe_permute_depth_labels: bool = False
  """Train depth against deterministically permuted labels as a null control."""


@dataclass
class RslRlPpoDwaqTeacherKLAlgorithmCfg(RslRlPpoTeacherKLAlgorithmCfg):
  """Config for Teacher-KL PPO with the DWAQ beta-VAE auxiliary loss."""

  class_name: str = "mjlab.rl.dwaq_algorithm:DWAQPPOTeacherKL"
  """Algorithm class name resolved by RSL-RL."""
  dwaq_velocity_target_groups: Tuple[str, ...] = ("dwaq_velocity_target",)
  """Observation groups containing the privileged velocity target."""
  dwaq_beta: float = 1.0
  """Beta multiplier for DWAQ latent KL divergence."""
  dwaq_autoencoder_loss_coef: float = 1.0
  """Overall coefficient for the DWAQ auxiliary loss."""
  dwaq_velocity_loss_coef: float = 1.0
  """Coefficient for supervised velocity-estimation MSE."""
  dwaq_reconstruction_loss_coef: float = 1.0
  """Coefficient for current-observation reconstruction MSE."""


@dataclass
class RslRlBaseRunnerCfg:
  seed: int = 42
  """The seed for the experiment. Default is 42."""
  num_steps_per_env: int = 24
  """The number of steps per environment update."""
  max_iterations: int = 300
  """The maximum number of iterations."""
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"actor": ("actor",), "critic": ("critic",)},
  )
  save_interval: int = 50
  """The number of iterations between saves."""
  experiment_name: str = "exp1"
  """Directory name used to group runs under
  ``logs/rsl_rl/{experiment_name}/``."""
  run_name: str = ""
  """Optional label appended to the timestamped run directory
  (e.g. ``2025-01-27_14-30-00_{run_name}``). Also becomes the
  display name for the run in wandb."""
  logger: Literal["wandb", "tensorboard"] = "wandb"
  """The logger to use. Default is wandb."""
  wandb_project: str = "mjlab"
  """The wandb project name."""
  wandb_tags: Tuple[str, ...] = ()
  """Tags for the wandb run. Default is empty tuple."""
  resume: bool = False
  """Whether to resume the experiment. Default is False."""
  load_run: str = ".*"
  """The run directory to load. Default is ".*" which means all runs. If regex
  expression, the latest (alphabetical order) matching run will be loaded.
  """
  load_checkpoint: str = "model_.*.pt"
  """The checkpoint file to load. Default is "model_.*.pt" (all). If regex expression,
  the latest (alphabetical order) matching file will be loaded.
  """
  bootstrap_checkpoint_path: str | None = None
  """Optional checkpoint loaded into a fresh run without searching log folders."""
  load_optimizer_on_resume: bool = True
  """Restore optimizer moments when resuming a compatible architecture."""
  load_iteration_on_resume: bool = True
  """Continue the saved iteration counter instead of starting a new schedule."""
  clip_actions: float | None = None
  """The clipping range for action values. If None (default), no clipping is applied."""
  upload_model: bool = True
  """Whether to upload model files (.pt, .onnx) to W&B on save. Set to
  False to keep metric logging but avoid storage usage. Default is True."""


@dataclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "OnPolicyRunner"
  """The runner class name. Default is OnPolicyRunner."""
  actor: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      }
    )
  )
  """The actor configuration."""
  critic: RslRlModelCfg = field(default_factory=RslRlModelCfg)
  """The critic configuration."""
  algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)
  """The algorithm configuration."""


@dataclass
class RslRlTeacherKLRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner config for PPO with actor/critic/teacher observation sets."""

  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {
      "actor": ("actor",),
      "critic": ("critic",),
      "teacher": ("teacher",),
    },
  )
  teacher: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      }
    )
  )
  """The frozen teacher actor configuration."""
  algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoTeacherKLAlgorithmCfg)
  """The PPO + teacher-KL algorithm configuration."""
