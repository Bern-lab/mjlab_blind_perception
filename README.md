# Blind Humanoid Locomotion with Teacher-Student Optimization

本仓库基于 `mjlab` 和 `rsl_rl`，主要用于研究无视觉人形机器人盲走任务。训练流程采用 teacher-student 联合优化：privileged teacher 可以使用视觉、height scan 或其他仿真特权信息，blind student 只使用可部署的本体感觉观测。部署时只导出 student policy。

当前实验重点是 Unitree G1 在 rough / stairs 场景中的盲走和楼梯适应。

## Branches

### `main`

主分支主要保留 Teacher-KL blind walking 任务：

```text
Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1
Mjlab-Velocity-Blind-Rough-TargetNavigation-TeacherKL-Unitree-G1
Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1
```

其中 `TeacherKL` 表示 student 使用 PPO 训练，同时通过 frozen teacher 的动作分布进行 guidance。可以通过配置关闭 teacher guidance，得到纯 PPO；也可以打开 `imitation_only` 做纯 IL。

### `lstm_teacher_policy`

该分支用于 LSTM / boolean stair flag 相关任务：

```text
Mjlab-Velocity-Blind-StairsFlag-TeacherKL-Unitree-G1
Mjlab-Velocity-Blind-StairsFlag-LSTM-TeacherKL-Unitree-G1
Mjlab-Velocity-Blind-Rough-LSTM-TeacherKL-Unitree-G1
```

这些任务主要用于比较普通 MLP student、显式 stairs flag / boolean 信息、以及 LSTM student 在 blind walking 中的表现。

## Setup

需要 NVIDIA GPU。推荐使用 `uv` 管理环境：

```bash
git clone <repo-url>
cd mjlab_111
uv sync
```

如果已经在仓库里，可以直接使用：

```bash
uv run train --help
uv run play --help
```

## Training

训练 main 分支 Teacher-KL blind rough：

```bash
uv run train Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.logger tensorboard
```

训练 target navigation Teacher-KL：

```bash
uv run train Mjlab-Velocity-Blind-Rough-TargetNavigation-TeacherKL-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.logger tensorboard
```

训练 slow latent Teacher-KL：

```bash
uv run train Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1 \
  --env.scene.num-envs 2048 \
  --agent.logger tensorboard
```

如果显存紧张，可以先降低 `num-envs`：

```bash
--env.scene.num-envs 1024
```

## Logs

训练日志默认保存在：

```text
logs/rsl_rl/<experiment_name>/<task_id>/<run_name>
```

例如：

```text
logs/rsl_rl/g1_blind_rough_teacherkl/
logs/rsl_rl/g1_blind_rough_target_navigation_teacherkl/
logs/rsl_rl/g1_blind_rough_target_navigation_slow_latent_teacherkl/
```

每个 run 的 `params/agent.yaml` 会保存 seed、PPO 参数、Teacher-KL 设置和是否为 `imitation_only`。

## Evaluation

使用已有 checkpoint 播放策略：

```bash
uv run play Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1 \
  --load-run <run-name> \
  --load-checkpoint model_13000.pt
```

也可以使用 dummy agent 快速检查环境：

```bash
uv run play Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1 --agent zero
uv run play Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1 --agent random
```

## Notes

- Student actor 是盲走策略，部署输入不应包含视觉、height scan、terrain boolean 或真实接触标签。
- Teacher / critic 可以在训练阶段使用 privileged observation。
- 对比实验建议固定 seed、terrain 配置、`num_steps_per_env` 和训练迭代数。

