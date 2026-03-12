# DreamerV3-Torch 中文文档

DreamerV3 的 PyTorch 实现，基于论文 [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104v1)。DreamerV3 是一种可扩展的世界模型强化学习算法，能够在多种域上使用固定超参数取得优异表现。

原始仓库：[NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)

---

## 项目结构

```
dreamerv3/
├── dreamer.py          # 主入口，包含 Dreamer 智能体类和训练循环
├── models.py           # WorldModel、ImagBehavior 等核心模型定义
├── networks.py         # 编码器、解码器、MLP、CNN 等网络组件
├── exploration.py      # 探索策略（Random、Plan2Explore）
├── tools.py            # 工具函数（日志、数据集采样、优化器等）
├── parallel.py         # 并行环境封装（Parallel、Damy）
├── configs.yaml        # 所有任务/环境的超参数配置
├── requirements.txt    # Python 依赖
├── Dockerfile          # Docker 构建文件
├── xvfb_run.sh         # 虚拟显示启动脚本（用于无头渲染）
├── envs/               # 环境封装
│   ├── wrappers.py     # 通用 Gym 环境包装器
│   ├── crafter.py      # Crafter 环境
│   ├── dmc.py          # DeepMind Control Suite 环境
│   ├── atari.py        # Atari 环境
│   ├── dmlab.py        # DeepMind Lab 环境
│   ├── memorymaze.py   # Memory Maze 环境
│   ├── minecraft.py    # Minecraft 环境
│   └── setup_scripts/  # 环境安装脚本
│       ├── atari.sh
│       └── minecraft.sh
├── imgs/               # 结果图片
└── logdir/             # 训练日志输出目录
```

---

## 环境配置

### Conda 环境

环境已创建就绪，路径为：

```
/data/conda_envs/seu004/dreamerv3-py311
```

- Python 3.11
- PyTorch 2.4.1 + CUDA 12.1
- NumPy 1.23.5
- Gym 0.22.0

### 激活环境

```bash
export PATH="/data/conda_envs/seu004/dreamerv3-py311/bin:$PATH"
```

### 已安装的依赖

| 包名 | 版本 | 说明 |
|------|------|------|
| torch | 2.4.1+cu121 | 深度学习框架（GPU 支持） |
| gym | 0.22.0 | 强化学习环境接口 |
| crafter | 1.8.0 | Crafter 生存环境 |
| mujoco | 2.3.5 | MuJoCo 物理引擎 |
| dm_control | 1.0.9 | DeepMind Control Suite |
| scipy | 1.11.4 | 科学计算（dm_control 依赖） |
| numpy | 1.23.5 | 数值计算 |
| ruamel.yaml | 0.17.4 | YAML 配置解析 |
| einops | 0.3.0 | 张量操作 |
| tensorboard | 2.17.1 | 训练可视化 |
| moviepy | 1.0.3 | 视频生成 |
| opencv-python | 4.7.0.72 | 图像处理 |
| protobuf | 3.20.0 | 序列化 |

### GPU 资源

服务器配备 8 × NVIDIA RTX A6000（每张 48GB 显存），CUDA 可用。

---

## 已完成的工作

### 1. 环境搭建

- 创建并配置 conda 环境（Python 3.11）
- 安装所有核心依赖
- 验证 PyTorch CUDA 支持正常

### 2. Bug 修复

**`envs/crafter.py` — gym.spaces.Box 边界错误**

`observation_space` 中 `is_first`、`is_last`、`is_terminal` 三个字段使用了 `dtype=np.uint8`，但边界设为 `(-np.inf, np.inf)`。`uint8` 类型无法表示无穷值，导致 gym 抛出 `ValueError`。

修复方案：将边界改为 `(0, 1)`，因为这些字段本身就是布尔值。

```python
# 修复前
"is_first": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
"is_last": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
"is_terminal": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),

# 修复后
"is_first": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
"is_last": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
"is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
```

### 3. 安装 MuJoCo / DMC

安装 mujoco、dm_control 及其依赖，并修复版本冲突：

```bash
pip install mujoco==2.3.5 dm_control==1.0.9
pip install numpy==1.23.5   # 防止被升级到 2.x
pip install scipy==1.11.4   # scipy 1.17 需要 numpy>=1.26，降级兼容
```

服务器无 OSMesa 库，渲染后端需改用 EGL。已将 `dreamer.py` 第 8 行从：
```python
os.environ["MUJOCO_GL"] = "osmesa"
```
改为：
```python
os.environ["MUJOCO_GL"] = "egl"
```

### 4. 冒烟测试

使用 `crafter` + `debug` 配置成功运行冒烟测试，验证以下环节均正常：

- ? 环境创建与交互
- ? 模型初始化（~198M 参数：model 181M + actor 9.5M + value 6.6M）
- ? 数据预填充（prefill）
- ? 评估循环（10 episodes）
- ? 训练循环（数据收集 + 模型更新）
- ? 日志输出（TensorBoard events、metrics.jsonl）
- ? Episode 数据保存（train_eps/、eval_eps/）

### 5. DMC 冒烟测试

使用 `dmc_vision` + `debug` 配置验证 DMC 环境正常：

- ? MuJoCo/dm_control 导入正常
- ? EGL 无头渲染正常（`MUJOCO_GL=egl`）
- ? walker_walk 环境创建成功
- ? 评估循环（eval_return 31.9）
- ? 训练循环（模型更新正常）

---

## 可训练的任务

`configs.yaml` 中定义了以下预设配置：

| 配置名 | 任务 | 观测 | 动作 | 训练步数 | 说明 |
|--------|------|------|------|----------|------|
| `dmc_proprio` | `dmc_walker_walk` | 状态 | 连续 | 500K | DMC 低维输入 |
| `dmc_vision` | `dmc_walker_walk` | 图像 | 连续 | 1M | DMC 高维图像输入 |
| `crafter` | `crafter_reward` | 图像 | 离散 | 1M | Crafter 生存环境 |
| `atari100k` | 各种 Atari 游戏 | 图像 | 离散 | 400K | Atari 100k 基准 |
| `minecraft` | `minecraft_diamond` | 图像+状态 | 离散 | 100M | Minecraft 寻钻石 |
| `memorymaze` | `memorymaze_9x9` | 图像 | 离散 | 100M | 记忆迷宫 |

> **当前可直接运行的环境**：Crafter、DMC Vision、DMC Proprio（均已验证通过）。Atari 需要安装 ROM，Minecraft 需要额外环境配置。

---

## 正在进行的训练

以下三个训练任务已启动，各自独占一张 GPU，后台运行中：

| 任务 | GPU | 日志文件 | 总步数 | 预估完成时长 |
|------|-----|----------|--------|------------|
| **Crafter** | cuda:3 | `crafter_train.log` | 1M env steps | ~3.5 天 |
| **DMC Vision** (walker_walk) | cuda:4 | `dmc_vision_train.log` | 1M env steps | ~20 小时 |
| **DMC Proprio** (walker_walk) | cuda:5 | `dmc_proprio_train.log` | 500K env steps | ~10 小时 |

> **速度估算依据**（在 RTX A6000 上实测）：
> - Crafter：~3 env steps/s（模型极大：dyn_deter=4096，约 198M 参数）
> - DMC Vision/Proprio：~14 env steps/s（标准模型：dyn_deter=512，约 18M 参数）

### 查看训练进度

```bash
# 实时跟踪日志
tail -f crafter_train.log
tail -f dmc_vision_train.log
tail -f dmc_proprio_train.log

# TensorBoard 可视化（所有任务一起展示）
export PATH="/data/conda_envs/seu004/dreamerv3-py311/bin:$PATH"
tensorboard --logdir ./logdir --bind_all
```

模型检查点每 10K agent steps 自动保存到各 logdir 下的 `latest.pt`。

---

## 训练命令

### 基本用法

```bash
export PATH="/data/conda_envs/seu004/dreamerv3-py311/bin:$PATH"
cd /data/code/seu004/wzy/dreamerv3

python dreamer.py --configs <配置名> [其他参数]
```

### Crafter 训练

```bash
# 正式训练（1M 步）
PYTHONUNBUFFERED=1 nohup python dreamer.py \
  --configs crafter \
  --logdir ./logdir/crafter_full \
  --device cuda:3 \
  > crafter_train.log 2>&1 &

# 快速调试
python dreamer.py --configs crafter debug \
  --logdir ./logdir/crafter_debug \
  --device cuda:3 \
  --compile False
```

### DMC Vision 训练

```bash
PYTHONUNBUFFERED=1 nohup python dreamer.py \
  --configs dmc_vision \
  --task dmc_walker_walk \
  --logdir ./logdir/dmc_vision_walker_walk \
  --device cuda:4 \
  > dmc_vision_train.log 2>&1 &
```

### DMC Proprio 训练

```bash
PYTHONUNBUFFERED=1 nohup python dreamer.py \
  --configs dmc_proprio \
  --task dmc_walker_walk \
  --logdir ./logdir/dmc_proprio_walker_walk \
  --device cuda:5 \
  > dmc_proprio_train.log 2>&1 &
```

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--configs` | - | 配置名，如 `crafter`、`dmc_vision`、`debug` |
| `--logdir` | null | 日志输出路径 |
| `--device` | `cuda:0` | 使用的 GPU，如 `cuda:1` |
| `--steps` | 取决于配置 | 总训练步数 |
| `--compile` | True | 是否启用 torch.compile（首次编译较慢） |
| `--seed` | 0 | 随机种子 |
| `--envs` | 取决于配置 | 并行环境数量 |
| `--batch_size` | 16 | 训练 batch 大小 |

`debug` 配置会自动设置 `pretrain=1, prefill=1, batch_size=10, batch_length=20`，适合快速验证代码是否能跑通。

---

## 监控训练

### TensorBoard

```bash
tensorboard --logdir ./logdir --bind_all
```

### 日志文件

训练日志保存在 `logdir/<实验名>/` 下：

- `events.out.tfevents.*` — TensorBoard 事件文件
- `metrics.jsonl` — JSON 格式指标记录
- `train_eps/` — 训练 episode 数据（.npz）
- `eval_eps/` — 评估 episode 数据（.npz）
- `latest.pt` — 最新模型检查点

---

## 已知问题

1. **Gym 已停止维护**：当前使用 gym 0.22.0，会打印升级到 Gymnasium 的警告，不影响运行。
2. **GradScaler FutureWarning**：`torch.cuda.amp.GradScaler` 已被弃用，会打印警告，不影响功能。
3. **torch.compile 编译慢**：首次运行启用 `--compile True` 时编译耗时较长，调试阶段建议用 `--compile False`。
4. **Atari/Minecraft 环境**：需要额外安装对应依赖（ROM、minerl 等），DMC 已安装完毕。
5. **scipy 版本冲突**：dm_control 会拉取 scipy 1.17，但其需要 numpy>=1.26，与项目的 numpy 1.23.5 不兼容，已手动降级到 scipy 1.11.4。
