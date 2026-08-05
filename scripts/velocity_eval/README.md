# Velocity Evaluation

This folder contains fixed, offline evaluation utilities for velocity policies.
It does not register new tasks and does not modify training environments. Each
script loads an existing task config, deep-copies it through the task registry,
then applies evaluation-only overrides at runtime.

## Eval-v1

The first fixed set is intentionally small:

- `flat`
- `upstairs_10cm`
- `upstairs_15cm`
- `upstairs_20cm`

Stair terrains use 10 riser levels by default for by-level collision statistics.
With the current fixed geometry this gives a square stair tile of about
`10.4 m x 10.4 m`, `0.30 m` step run, and `3.0 m` center platform.

Commands are fixed by default:

- `vx = 0.4 m/s`
- `vy = 0.0 m/s`
- `wz = 0.0 rad/s`

Domain randomization, command curriculum, pushes, observation noise, and
observation delays are disabled by default for clean comparison. Observation
history length is preserved so the actor input shape remains compatible with
the checkpoint.

## Run Policy Evaluation

Example with a local checkpoint:

```bash
uv run python scripts/velocity_eval/eval_policy_on_terrains.py \
  Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1 \
  --checkpoint-file /path/to/model.pt \
  --episodes-per-terrain 50 \
  --num-envs 50
```

Example with a W&B run:

```bash
uv run python scripts/velocity_eval/eval_policy_on_terrains.py \
  Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1 \
  --wandb-run-path entity/project/run_id \
  --wandb-checkpoint-name model_40000.pt
```

The JSON output includes:

- `success_rate`
- `fall_rate`
- `mean_episode_length_s`
- `tracking_error`
- `tracking_lin_error`
- `tracking_yaw_error`
- `toe_riser_collision_count`
- `heel_riser_collision_count`
- `foot_lip_collision_count`
- `base_pitch_roll_rms`
- `action_smoothness`
- `torque_cost`
- `foot_clearance`
- collision counts by stair level
- `collision_by_stair_level_low_to_high`
- `heading_failure_rate`

For G1 blind policies, eval adds an evaluation-only `toe_terrain_contact`
MuJoCo contact sensor when needed. `toe_riser_collision_count` and
`heel_riser_collision_count` are therefore true foot-vs-terrain contact
transition counts on vertical riser-like surfaces, not actor observations and
not geometric danger-zone occupancy. They are sampled at the policy control
step. If that sensor is unavailable, the evaluator falls back to the older
geometry-based approximation and records the source in `collision_event_source`.

If the robot yaw deviates from the commanded travel direction by more than
`--heading-failure-angle-deg` (default `45`), the evaluator ends that episode as
a failure. Collision statistics from heading-failed episodes are excluded from
toe/heel collision averages and the low-to-high per-level collision table, so a
policy that turns sideways does not pollute the early stair-level counts.

The evaluator also writes a compact PNG table next to the JSON by default:

```text
eval_outputs/velocity/g1_blind_rough_teacherkl/0526_143012/eval_eval_v1_table.png
```

If `--output-file` is omitted, each evaluation run creates a folder grouped by
the trained policy family and then by timestamp, for example:

```text
eval_outputs/velocity/g1_blind_rough_teacherkl/0526_143012/eval_eval_v1.json
```

Pass `--output-dir` to choose the timestamp folder yourself, or `--output-file`
to write to one exact file.

## Goal Pyramid Evaluation

`eval_policy_goal_pyramid.py` builds a single large convex `pyramid_stairs`
terrain for target-navigation stair climbing. Robots spawn randomly on one of
the four low outer sides, within the top platform projection, facing the pyramid
center. At every control step the command points toward the center top platform:

```text
vx = goal_speed * max(cos(yaw_error), 0)
vy = 0
wz = clip(yaw_kp * yaw_error, -yaw_rate_limit, yaw_rate_limit)
```

An episode succeeds when the robot reaches the top center region. If the robot
turns too far away from the side-normal approach direction, the episode is
marked as a heading failure and its collision counts are excluded from
success-only collision statistics.

Example:

```bash
uv run python scripts/velocity_eval/eval_policy_goal_pyramid.py \
  Mjlab-Velocity-Blind-Rough-LSTM-TeacherKL-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_blind_rough_lstm_teacherkl/Mjlab-Velocity-Blind-Rough-LSTM-TeacherKL-Unitree-G1/5.25deployed/model_14800.pt \
  --episodes 50 \
  --num-envs 50 \
  --max-episode-length-s 12.0 \
  --stair-levels 10 \
  --stair-height 0.15 \
  --step-width 0.30 \
  --platform-width 3.0 \
  --flat-apron-width 3.0 \
  --terrain-border-width 12.0 \
  --goal-radius 0.75 \
  --output-file eval_outputs/velocity
```

The stair size switches are:

- `--stair-levels`: number of low-to-high riser levels, default `10`
- `--stair-height`: riser height in meters, default `0.15`
- `--step-width`: stair tread/run width in meters, default `0.30`
- `--step-widths`: run a tread-width sweep in one command, for example
  `--step-widths 0.27 0.30 0.33`
- `--platform-width`: square top platform width in meters, default `3.0`
- `--flat-apron-width`: flat ground connected to the pyramid bottom, default `3.0`
- `--terrain-border-width`: extra terrain-generator border around the tile, default `12.0`
- `--start-distance`: optional explicit spawn radius; otherwise computed from the stair geometry
- `--spawn-tangent-half-width`: optional side-wise spawn range; otherwise kept inside the top platform width

Outputs are timestamped by default when `--output-file` points at a directory:

```text
eval_outputs/velocity/g1_blind_rough_lstm_teacherkl/0527_165514/goal_pyramid_h15cm.json
eval_outputs/velocity/g1_blind_rough_lstm_teacherkl/0527_165514/goal_pyramid_h15cm_table.png
```

The JSON contains `success_rate`, failure rates, spawn side counts, landing
support metrics, and `collision_by_stair_level_low_to_high_success_only`. The
table image is landing-centric: it shows the publication score, strict safe-pass
rate, landing index, completion, tread support, complete/incomplete landing
ratios, low-support landings, and mean toe-riser contacts by stair level over
successful episodes only.

### Goal-Pyramid Metric Guide

Use this evaluation when comparing stair-ascent policies for reports or paper
figures:

```bash
uv run python scripts/velocity_eval/eval_policy_goal_pyramid.py \
  Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/.../model.pt \
  --episodes 100 \
  --num-envs 100 \
  --max-episode-length-s 12.0
```

For fair comparisons, keep the terrain parameters, seed, episode count, and
goal speed fixed across policies. The headline metric is `score_100`, a
full-support-emphasized stair-ascent score:

```text
score_100 =
15 * success_rate
+ 5 * mean_max_height_progress_fraction
+ 15 * mean_stair_landing_support_fraction
+ 45 * stair_full_landing_ratio
+ 15 * stair_full_landing_ratio^2
+ 5 - toe_riser_collision_penalty
```

The nonlinear full-support term is intentional: a high complete-foot landing
ratio is a stronger stair-ascent safety signal than small changes in smoothness
or actuator economy. The publication score intentionally leaves heel/lip
contacts, base pitch/roll, action smoothness, and torque cost out of the
headline number, because this eval measures stair completion and foot placement.
Toe-riser contacts still affect the headline score: the first two toe contacts
are free by default, then the penalty grows progressively:

```text
excess = max(0, mean_toe_riser_contacts_per_episode - 2)
toe_riser_collision_penalty = min(12, 0.35 * excess * (excess + 1) / 2)
```

This makes an occasional probe contact acceptable while making repeated riser
kicks increasingly expensive.

Use the auxiliary metrics to explain behavior:

- `landing_index_100`: landing-only score for support, complete landing,
  incomplete landing, and low-support failures.
- `landing_linear_score_100`: transparent linear reference score using the
  previous landing-centric weighting.
- `stair_safe_pass_rate`: episode-level acceptance rate. An episode passes only
  if it succeeds, has enough complete-foot landings, has high average support,
  keeps low-support landings below the threshold, and stays under the toe-riser
  contact threshold.
- `episode_stair_full_landing_ratio_p10`: worst-tail complete landing quality;
  useful when mean full-landing ratio looks good but some episodes still fail.
- `toe_riser_collision_over_free_count`: mean toe-riser contacts above the free
  allowance. Toe contacts have a two-hit free allowance by default.
- `score_collision_penalties.toe`: progressive score penalty from repeated
  toe-riser contacts.

The default strict pass thresholds are:

```text
safe_pass_min_full_landing_ratio = 0.75
safe_pass_min_support_fraction = 0.80
safe_pass_max_low_support_ratio = 0.20
safe_pass_max_toe_riser_collision_count = 10.0
```

Override them from the command line when the ablation needs a stricter or looser
acceptance test, for example:

```bash
uv run python scripts/velocity_eval/eval_policy_goal_pyramid.py \
  Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/.../model.pt \
  --safe-pass-min-full-landing-ratio 0.80 \
  --safe-pass-max-toe-riser-collision-count 8.0
```

### Stair Lift-Height Sweep

Use `eval_stair_lift_height_sweep.py` when checking whether a policy changes
its swing height with stair riser height. The script runs the goal-pyramid task
at several fixed riser heights, records each stair touchdown swing, and fits
`peak_lift_from_takeoff_m` against true stair height. SlowLatent policies also
report the decoded `pred_riser_height_mean_m` fit.

```bash
uv run python scripts/velocity_eval/eval_stair_lift_height_sweep.py \
  Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/.../model.pt \
  --stair-heights 0.09 0.11 0.13 0.15 0.18 0.21 0.24 \
  --episodes-per-height 40 \
  --num-envs 40
```

For local checkpoints, the task selector can be omitted. The script will infer a
registered task from the checkpoint path when possible, and falls back to legacy
G1 presets for common actor observation dimensions: 98 without height scan, 285
with height scan, and 490 blind-history policies.

```bash
uv run python scripts/velocity_eval/eval_stair_lift_height_sweep.py \
  --checkpoint-file logs/rsl_rl/g1_velocity/5.19_original_train/model_61999.pt \
  --stair-heights 0.09 0.11 0.13 0.15 0.18 0.21 \
  --episodes-per-height 40 \
  --num-envs 40
```

The JSON summary reports separate linear fits for the first stair level, the
first stair touchdown, later stair levels, and all stair touch-downs. The CSV
next to it contains one row per foot swing with touchdown level, peak lift,
terrain clearance, decoded riser height, stair probability, and gate-memory
ratio.

To watch one height from the same eval setup, add `--play` and select a height
with `--play-stair-height`:

```bash
uv run python scripts/velocity_eval/eval_stair_lift_height_sweep.py \
  --checkpoint-file logs/rsl_rl/.../model.pt \
  --play \
  --play-stair-height 0.18 \
  --num-envs 1 \
  --max-episode-length-s 12.0
```

### Stair Height-Transition Evaluation

Use `eval_stair_height_transition.py` to test whether a policy updates its
foot-lift behavior after the stair height changes within the same episode. The
eval builds a long runway with a 3 m wide stair strip: 10 upward steps at the
first riser height, a short flat platform, then 10 upward steps at the second
riser height. Vectorized lanes get side margins, but the actual stair surface
remains 3 m wide. The robot spawns facing the first stair from the bottom apron,
with lateral randomization limited to the middle 1 m of the stair width. The
command points to the final top target.

```bash
uv run python scripts/velocity_eval/eval_stair_height_transition.py \
  --checkpoint-file logs/rsl_rl/.../model.pt \
  --first-stair-height 0.10 \
  --second-stair-height 0.18 \
  --episodes 40 \
  --num-envs 40
```

The default stair-height and tread-depth checks stay within the current training
range (`0.088--0.25 m` risers, `0.23--0.37 m` treads). The JSON/CSV outputs
split landings into first and second stair stages, mark the first touchdown after
the height change, and report stage deltas plus paired-episode lift changes for
`peak_lift_from_takeoff_m` and SlowLatent `pred_riser_height_mean_m`.

To watch the same eval task in a viewer, add `--play`. Use a small `--num-envs`
when you want to inspect motion clearly:

```bash
uv run python scripts/velocity_eval/eval_stair_height_transition.py \
  --checkpoint-file logs/rsl_rl/.../model.pt \
  --first-stair-height 0.10 \
  --second-stair-height 0.18 \
  --play \
  --num-envs 4 \
  --max-episode-length-s 16.0
```

The play mode automatically respawns robots on the transition runway after
success, fall, timeout, or heading failure. Pass `--viewer native` or
`--viewer viser` to force a backend; `auto` uses native when a display is
available and Viser otherwise.

## Footprint Detector Training

`train_footprint_detector_v3.py` is the preset entrypoint for retraining the
footprint-only Stage 2D detector with deployment-friendly proprioceptive inputs.
It uses the existing online detector trainer, but pins the task to
`footprint_deploy_v3`, enables gait phase, expects a `134`-D detector input,
uses a `16`-frame history, and turns on `--footprint-only-model`.

The preset now defaults to the detector-only task
`Mjlab-Velocity-Blind-Rough-TargetNavigation-FootprintDetector-SlowLatent-TeacherKL-Unitree-G1`.
This task leaves the main slow-latent policy task unchanged, but changes the
rollout terrain distribution used to collect supervised detector samples:
`70%` of envs are assigned once to standalone continuous long-stair runways and
remain in that pool for every reset, while the remaining `30%` stay in the
regular tiled terrain grid. Difficulty rows are sampled from low/mid/high
buckets with weights `0.15/0.25/0.60`, so hard rows appear more often than easy
rows from the start of data collection. The standalone stair pool uses eight
fixed tread-depth variants spanning the high-stair training range, so width
coverage is explicit instead of relying on an opaque per-mesh random draw. The
grid pool keeps a small positive flat-terrain weight for flat touchdown examples.

The exported network still has the standard six-logit layout:

```text
[0] left_contact
[1] right_contact
[2] left_touchdown
[3] right_touchdown
[4] left_toe_riser_hit   = dummy low logit
[5] right_toe_riser_hit  = dummy low logit
```

Only logits `0:4` are trained. Logits `4:6` are kept as dummy low toe logits so
the ONNX metadata and downstream six-slot event layout remain compatible.
Deployment should read footprint/contact results from `0:4` and fill `4:6` from
the separate toe-riser detector before any event-summary logic.

Example:

```bash
uv run python scripts/velocity_eval/train_footprint_detector_v3.py \
  --checkpoint-file logs/rsl_rl/g1_blind_rough_target_navigation_slow_latent_teacherkl/Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1/base111/model_51000.pt \
  --num-envs 512 \
  --steps 9000
```

The default output directory for this preset is
`eval_outputs/stair_stage2/model51000_seed42_footprint_deploy_v3_fixed_pool_v1`.
Curriculum logs include `fixed_pool_*` metrics for checking the actual 70/30
pool split and low/mid/high reset distribution before trusting a long run.

The v3 observation intentionally does not embed the legacy 91/93-D
`stair_latent` vector. It is built from signals available on the robot side:
IMU projected gravity, body-frame base angular velocity and delta, command
`vx/vy/wz`, gait phase, body-frame FK toe/heel/sole-center positions,
body-frame endpoint velocities and velocity deltas, per-foot
gravity-projected height/vertical-velocity scalars, last leg action/action
delta, joint tracking error, and leg joint velocity. Fixed input multipliers are
written to `input_feature_scale_groups` and `input_feature_scales` in the JSON
metadata and must be applied identically in deployment. The observation does not
use contact bits, terrain height, support fraction, camera/raycast data, or any
other external perception. It also does not require global root position or
terrain odometry; endpoint positions come from body-frame FK and the scalar
height terms come from dot products with IMU gravity.

The deploy runner must build the same feature order from its local state
estimator, FK, command, action history, and joint encoder data. Keep the feature
group metadata from the exported `deployment_contract.json` as the source of
truth when wiring the C++ observation builder.

Training labels are still simulation-only: contact/touchdown are built from
horizontal support contact, excluding toe-riser contact and using the ground
contact sensor's vertical force when available. Those contact signals are never
fed to the detector input. For footprint-only runs, replay buffer stair hard
negatives are selected from touchdown-negative stair frames, so toe-riser hits
without touchdown remain useful negative examples instead of being filtered out
by the dummy toe labels. `label_audit.csv` also reports raw contact,
toe-riser-contact, vertical-force-support, and toe-riser/vertical-support
conflict counts for checking label contamination before a long training run.

## Collect Latents

```bash
uv run python scripts/velocity_eval/collect_policy_latents.py \
  Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1 \
  --checkpoint-file /path/to/model.pt \
  --episodes-per-terrain 20 \
  --num-envs 20 \
  --steps-per-episode 500
```

For MLP actors, the saved latent is the hidden activation before the final actor
linear layer. For recurrent actors, it is the last recurrent hidden state.

If `--output-file` is omitted, each collection run uses the same grouped folder
layout, for example:

```text
eval_outputs/velocity/g1_blind_rough_teacherkl/0526_143012/latents_cluster_v1.npz
```

## Quick PCA

```bash
uv run python scripts/velocity_eval/analyze_latent_clusters.py \
  --input-file eval_outputs/velocity/g1_blind_rough_teacherkl/0526_143012/latents_cluster_v1.npz \
  --phase-bins 8
```

By default, analysis outputs are written next to the input `.npz`:

- `latent_pca.csv`
- `latent_pca.png`
- `phase_bins/phase_*.png` when `--phase-bins` is provided
