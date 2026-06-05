# Blind Humanoid Locomotion with Teacher-Student Optimization

本仓库基于 `mjlab` 和 `rsl_rl`，主要用于研究无视觉人形机器人盲走任务。训练流程采用 teacher-student 联合优化：privileged teacher 可以使用视觉、height scan 或其他仿真特权信息，blind student 只使用可部署的本体感觉观测。部署时只导出 student policy。

当前实验重点是 Unitree G1 在 rough / stairs 场景中的盲走和楼梯适应。

## Branches

### `main` — 评估 + Teacher-KL + Latent 任务

本分支用于**模型评估**和训练 **Teacher-KL** 与 **Slow Latent** 类任务：

```text
Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1
Mjlab-Velocity-Blind-Rough-TargetNavigation-TeacherKL-Unitree-G1
Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1
```

其中 `TeacherKL` 表示 student 使用 PPO 训练，同时通过 frozen teacher 的动作分布进行 guidance。`SlowLatent` 在 teacher-student 基础上引入慢变 latent 变量，提高策略稳定性。

本分支同时包含离线评估工具（`scripts/velocity_eval/`），用于：
- 标准化地形评测（flat, upstairs_10cm/15cm/20cm）
- 目标金字塔楼梯攀爬评测
- Policy latent 采集与 PCA 分析

### `lstm_teacher_policy` — LSTM + Boolean Stair Flag 任务

该分支用于 LSTM 和 boolean stair flag 相关任务：

```text
Mjlab-Velocity-Blind-StairsFlag-TeacherKL-Unitree-G1
Mjlab-Velocity-Blind-StairsFlag-LSTM-TeacherKL-Unitree-G1
Mjlab-Velocity-Blind-Rough-LSTM-TeacherKL-Unitree-G1
```

这些任务主要用于比较普通 MLP student、显式 stairs flag / boolean 信息、以及 LSTM student 在 blind walking 中的表现。

## Setup

需要 NVIDIA GPU。推荐使用 `uv` 管理环境：

```bash
git clone https://github.com/Bern-lab/mjlab_sqm.git
cd mjlab_sqm
uv sync --extra cu128
```


如果已经在仓库里，可以直接使用：

```bash
uv run train --help
uv run play --help
```

## Training

训练 Teacher-KL blind rough：

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

训练 target heading Teacher：

```bash
uv run train Mjlab-Velocity-TargetHeading-Rough-Teacher-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.logger tensorboard
```

## Logs

训练日志默认保存在：

```text
logs/rsl_rl/<experiment_name>/<task_id>/<run_name>
```

每个 run 的 `params/agent.yaml` 会保存 seed、PPO 参数、Teacher-KL 设置和是否为 `imitation_only`。

## Evaluation

### 交互式播放

使用已有 checkpoint 播放策略：

```bash
uv run play Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1 \
  --load-run <run-name> \
  --load-checkpoint model_13000.pt
```


### 离线地形评测

```bash
uv run python scripts/velocity_eval/eval_policy_on_terrains.py \
  Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1 \
  --checkpoint-file /path/to/model.pt \
  --episodes-per-terrain 50 \
  --num-envs 50
```

输出 JSON（success_rate, fall_rate, collision 统计等）和 PNG 表格。

### 目标金字塔评测

```bash
uv run python scripts/velocity_eval/eval_policy_goal_pyramid.py \
  Mjlab-Velocity-Blind-Rough-LSTM-TeacherKL-Unitree-G1 \
  --checkpoint-file /path/to/model.pt \
  --episodes 50 \
  --num-envs 50 \
  --max-episode-length-s 12.0
```

详见 `scripts/velocity_eval/README.md`。

### 导出部署

将训练好的 policy 导出为 TorchScript 格式：

```bash
uv run python src/mjlab/scripts/export.py -c <checkpoint> -t Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1
```
