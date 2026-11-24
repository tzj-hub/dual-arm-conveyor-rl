# Four-arm conveyor sorting with hierarchical reinforcement learning

四臂流水线分拣：**Masked PPO 学习任务调度，规则控制器执行运动规划与真实双指夹持**。机器人模型、网格、训练代码和配置均随仓库提供，可以从零训练，不需要作者电脑上的文件或预训练权重。

默认场景：4 台 Panda、0.2 m/s 恒速传送带、双列随机来料、2 个深料箱；物体有不同形状、姿态、质量和接触属性。默认使用 **oracle 特权状态**，便于先研究调度。源码也包含相机相关实现，但下述训练与成绩不代表视觉策略效果。

## 安装

建议使用 Linux、Python 3.12/3.13 和独立虚拟环境。项目声明支持 Python 3.10 以上；本地验证环境为 Python 3.13.15，其他版本尚未逐一验证。训练预设面向 RTX 4060 Laptop 级别的设备；约 16 GB 内存机器上保持单个物理环境任务，避免同时启动多份训练、评估或录像。

```bash
git clone https://github.com/tzj-hub/dual-arm-conveyor-rl.git
cd dual-arm-conveyor-rl
bash scripts/setup.sh
bash scripts/run.sh check
```

`setup.sh` 创建项目自己的 `.venv` 并安装训练、录像和测试依赖，不启动训练。已有其他虚拟环境时也可运行 `python -m pip install -e '.[all]'`，并设置 `CONVEYOR_PYTHON` 为该环境的 Python 绝对路径。

NVIDIA 驱动应先在系统上正常工作；检查输出中的 `torch.cuda_available` 应为 `true`。无 CUDA 可使用后面的 `--device cpu`，通常会更慢。PyTorch 安装渠道可按 [官方安装说明](https://pytorch.org/get-started/locally/) 选择；已有满足要求的 PyTorch 不需要重新安装。

默认 oracle 无界面训练不需要相机渲染。录像、相机或渲染测试需要 OpenGL：NVIDIA 环境可使用 EGL；Ubuntu 软件渲染可安装 `libosmesa6`，交互窗口需要桌面显示环境及 `libglfw3`。渲染检查可运行 `bash scripts/run.sh check --probe-renderer`。

## 从零训练

先预览配置，不创建训练目录、不更新权重：

```bash
bash scripts/train_sequence.sh --output results/my_run
```

确认配置后开始训练：

```bash
bash scripts/train_sequence.sh --output results/my_run --execute
```

没有 CUDA 的机器使用：

```bash
bash scripts/train_sequence.sh --output results/my_run_cpu --device cpu --execute
```

默认配置是 [`configs/sequence_r3_fresh.json`](configs/sequence_r3_fresh.json)：

| 参数 | 默认值 |
| --- | --- |
| 初始权重 | 随机初始化，无外部 checkpoint |
| BC 热启动 | 采集 24576 个物理决策步，最多保留 8192 条教师示范，80 epochs |
| PPO | 65536 个决策步，rollout 1024，minibatch 256 |
| 策略 | set-context 候选评分网络，hidden size 64，484 个调度动作 |
| 教师辅助 | sequence 规则教师，PPO 阶段辅助系数从 0.2 衰减到 0.05 |
| 验证 | 每 16384 个 PPO 步，4 个独立于训练种子的 100 件回合 |
| 下游完成余量 | r3 开启；外侧返回、外侧放料实验关闭 |

**这是 BC 热启动加 PPO 强化学习，不是只有模仿学习。** PPO 使用物理执行轨迹的奖励、优势估计与策略更新；在线教师损失是辅助项。物理采样和轨迹检查主要在 CPU 上，小网络并不一定占满 GPU。训练包括采样、BC、PPO 和多轮验证，不能用网络更新速度推算整个训练用时。

这是新整理的从零训练预设，**尚未完成该预设的完整训练，不保证复现历史成绩或达到 98%**。不启用 r3 的基线预设为 `configs/sequence_balanced.json`，通过 `--config` 选择。`--steps` 可以调整 PPO 总步数，但不会自动减少 BC 和验证开销。

## 进度、暂停和恢复

另开终端查看进度：

```bash
.venv/bin/python scripts/training_progress.py --run results/my_run --watch
.venv/bin/python scripts/plot_training.py --run results/my_run
```

在训练终端按一次 `Ctrl-C`，等待程序保存检查点并退出。恢复时先预览，再执行：

```bash
bash scripts/resume_scheduler.sh --run results/my_run
bash scripts/resume_scheduler.sh --run results/my_run --execute
```

恢复需要同一份训练源码，保留优化器和 PPO 计数，从新的物理回合继续。**BC 采集/优化尚未完成时，未完成的示范缓存不能恢复**；此时保留原目录，换新输出目录重新训练。不要对已有目录重复使用新训命令，它会拒绝覆盖。

输出目录包含 `launch.json`、`console.log`、`training.json`、`updates.jsonl`、`latest.pt`，以及根据训练阶段产生的 `bc.pt`、验证选出的 `best.pt` 和 `selection.json`。

## 使用自己训练的模型

通用入口从指定运行的 `training.json` 恢复完整场景，默认加载该运行的 `best.pt`。**所有命令默认只预览，加 `--execute` 才实际创建仿真。**

```bash
# 查看恢复的场景和模型
bash scripts/run_trained.sh print-config --run results/my_run

# 桌面交互演示
bash scripts/run_trained.sh play --run results/my_run --execute

# 原速录像，30 FPS；输出文件必须不存在
bash scripts/run_trained.sh record --run results/my_run \
  --output videos/my_run.mp4 --fps 30 --execute

# 在新来料和物体目录种子上，评估一个完整回合
bash scripts/run_trained.sh evaluate --run results/my_run \
  --seed 2000000 --catalog-seed 20001 \
  --output results/my_run/test_2000000.json --execute
```

还没有 `best.pt` 时可显式使用 `--checkpoint latest`；只查看已完成热启动的模型时可用 `--checkpoint bc`。若相应文件尚未产生，入口会说明缺失情况，不会暗中换模型。评估入口一次只跑一个回合；多个种子顺序运行并使用不同输出文件，避免多份物理环境同时占用内存。录像的生成耗时不决定视频播放速度。

## 方法和当前限制

上层预测目标的运动与抓取截止时间，建立可撤销的未来任务计划，由 Masked PPO 在可行动作中选择；下层完成拦截、IK、路径检查、夹爪闭合、触觉确认、搬运和放置。抓取没有使用磁吸或物体焊接。

r3 为下游近侧物体增加约 0.3 秒完成余量，同时保留原始派工窗口、路径检查和共享区互斥。它还没有实现同一工位两臂的自由并发，PPO 不能选择被硬动作掩码排除的动作。

历史固定旧权重的六组配对结果为 527→531/600（87.83%→88.50%）；这是有限样本下的小幅改进。旧权重和本机实验目录不在此仓库中，该数字不是上述从零训练预设的成绩。更多说明见 [`docs/METHODS_AND_RESULTS.md`](docs/METHODS_AND_RESULTS.md)。

## 开发检查

```bash
bash scripts/test.sh -q tests/test_config.py tests/test_cooperation_flags.py tests/test_run_trained.py
```

完整测试还包含物理场景和渲染测试；有 OpenGL/OSMesa 环境后再运行 `bash scripts/test.sh -q`。保持单进程，不使用 pytest-xdist 并行启动场景。

源码采用 `src/` 布局。机器人资产位于 `src/dual_arm_conveyor/assets/`，安装 wheel 后也应保留。`results/`、视频、模型、虚拟环境和缓存由 `.gitignore` 排除。经过本地验证的依赖版本见 [`requirements-tested.txt`](requirements-tested.txt)，这是参考环境清单，不是覆盖所有平台的完整锁文件。

`CONVEYOR_LIBM_COMPAT=1` 是训练入口默认启用的本机 glibc FMA 路径兼容设置，可显式设为 `0`；它不修改机器人动力学或训练目标。

原创代码采用 [MIT 许可证](LICENSE)。第三方来源、版权和修改说明见 [`THIRD_PARTY.md`](THIRD_PARTY.md)，Panda 模型及 MuJoCo 适配代码保留各自的 Apache-2.0 许可；发行包的许可证表达式因此包含两者。发布前检查范围见 [`docs/RELEASE_VALIDATION.md`](docs/RELEASE_VALIDATION.md)。
