# CASCADE: Collaborative Aerial Sensing and Computational Adaptive Drone Edge

面向洪涝灾害的异构无人机集群协同任务调度框架。

## 概述

CASCADE 是一个面向洪涝灾害应急响应的多无人机协同任务调度框架，基于 DECCo（Zhang et al., 2023）改进而来。核心特点：

- **6 维资源建模**：CPU / GPU / Memory / Storage / Bandwidth / Energy
- **DAG 任务流水线**：支持采集→预处理→推理→融合的复杂依赖关系
- **mA3C+MHSA+GNN 调度引擎**：多智能体强化学习 + 图神经网络 + 多头自注意力
- **多模态感知**：RGB / 热红外 / 多光谱 / 高光谱传感器协同
- **三级降级容错**：全功能 → 半自治 → 最低生存
- **动态 Mesh 自组网**：LoRa + WiFi 6 双模通信

## 项目结构

```
scheduling-rl/
├── doc/                          # 方案文档与实验计划
├── configs/                      # Hydra YAML 配置
├── src/                          # 源代码
│   ├── env/                      # 仿真环境
│   ├── algorithms/               # 调度算法
│   ├── execution/                # 执行与容错
│   └── evaluation/               # 评估与可视化
├── experiments/                  # 实验脚本
└── tests/                        # 单元测试
```

## 远程实验环境

正式训练和带 Torch 网络前向的验证运行在远程 GPU 服务器上。远程连接信息放在本地 `ssh-config.local`，该文件不提交。

远程工作目录：

```bash
/root/autodl-tmp/code/cascade-uav-scheduling/
```

远程环境初始化：

```bash
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

cd /root/autodl-tmp/code/cascade-uav-scheduling
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 快速开始

```bash
# 远端进入项目环境
cd /root/autodl-tmp/code/cascade-uav-scheduling
source .venv/bin/activate

# 运行测试
python -m unittest discover -s tests
python experiments/smoke_env.py

# 训练 CASCADE（短跑 smoke；正式实验增大 train-episodes）
python experiments/train_cascade.py \
  --train-episodes 10 \
  --eval-episodes 2 \
  --eval-every 5 \
  --seed 0 \
  --output-dir outputs/training/cascade_ma3c_smoke

# 简单经典方法对比：DS1/DS2/DS3 × CASCADE/Greedy/Min-Load/Round-Robin/HEFT
python experiments/run_simple_comparison.py \
  --episodes 3 \
  --seed 0 \
  --output-dir outputs/results/simple_comparison_smoke
```

## 实验输入输出

当前常用入口：

- `experiments/train_cascade.py`：端到端训练 CASCADE，包含 GAE、Actor-Critic 更新、周期验证、checkpoint 和训练指标输出。
- `experiments/run_simple_comparison.py`：运行当前主推的简单经典对比，覆盖 DS1/DS2/DS3 与 CASCADE / Greedy / Min-Load / Round-Robin / HEFT。
- `experiments/run_cascade.py`：只评估当前 CASCADE 调度器。
- `experiments/run_baselines_only.py`：只评估 Greedy / Min-Load / Round-Robin / HEFT。
- `experiments/run_e1_comparison.py`：兼容旧 E1 主对比入口，包含 CASCADE + baselines。

当前标准场景配置：

- `configs/env/scenario_ds1_standard.yaml`：标准洪涝，训练/ID 测试。
- `configs/env/scenario_ds2_complex.yaml`：复杂洪涝，OOD 泛化测试。
- `configs/env/scenario_ds3_extreme.yaml`：极端/降级场景，含 UAV 故障与应急任务注入。

输出目录默认为 `outputs/results/<experiment>_<timestamp>/` 或 `outputs/training/<experiment>_<timestamp>/`。

每次运行会生成：

- `summary.json`：按方法聚合后的指标。
- `summary.csv`：便于论文表格整理的聚合指标。
- `episodes.csv`：每个 episode 的原始结果。
- `figures/completion_ratio.png`
- `figures/tdsr.png`
- `figures/total_reward.png`
- `figures/rpdr_proxy.png`
- `figures/baseline_radar.png`
- `figures/episode_metric_trends.png`
- `figures/episode_cumulative_mean_trends.png`

推荐在远端先跑小规模 smoke，再扩大 episode 数。

```bash
# 只跑 CASCADE 评估
python experiments/run_cascade.py \
  --config configs/env/scenario_ds1_standard.yaml \
  --seed 0 \
  --episodes 20 \
  --output-dir outputs/results/ds1_cascade_20ep

# 只跑经典 baseline
python experiments/run_baselines_only.py \
  --config configs/env/scenario_ds1_standard.yaml \
  --episodes 20 \
  --seed 0 \
  --output-dir outputs/results/ds1_baselines_20ep

# 当前推荐的简单经典对比
python experiments/run_simple_comparison.py \
  --scenarios ds1 ds2 ds3 \
  --methods cascade_ma3c greedy min_load round_robin heft \
  --episodes 20 \
  --seed 0 \
  --output-dir outputs/results/simple_comparison_20ep
```

## CASCADE 训练

`train_cascade.py` 会在每个 episode 重新采样 DS1 场景，执行当前策略，收集 reward/value/log-prob，计算 GAE，并做 Actor-Critic 更新。

```bash
python experiments/train_cascade.py \
  --config configs/env/scenario_ds1_standard.yaml \
  --eval-config configs/env/scenario_ds1_standard.yaml \
  --train-episodes 1000 \
  --eval-episodes 10 \
  --eval-every 100 \
  --seed 0 \
  --output-dir outputs/training/cascade_ma3c_ds1_1000ep
```

训练输出：

- `cascade_ma3c.pt`：模型 checkpoint。
- `train_metrics.csv` / `train_metrics.json`：训练 episode 指标。
- `eval_metrics.csv`：周期验证指标；`train_metrics.json` 同时保存训练与验证记录。

当前训练实现为单进程端到端流程；异步多环境 A3C 加速是后续增强项。

## SwanLab 可视化

推荐使用 SwanLab 管理正式实验记录。首次使用先安装依赖并登录：

```bash
pip install -r requirements.txt
swanlab login
```

云端记录：

```bash
python experiments/run_baselines_only.py \
  --episodes 20 \
  --use-swanlab \
  --swanlab-mode cloud \
  --swanlab-project cascade-uav-scheduling \
  --swanlab-experiment s1-baseline-comparison
```

CASCADE 新方法云端记录：

```bash
python experiments/run_cascade.py \
  --episodes 20 \
  --use-swanlab \
  --swanlab-mode cloud \
  --swanlab-project cascade-uav-scheduling \
  --swanlab-experiment s1-cascade
```

E1 主对比云端记录：

```bash
python experiments/run_e1_comparison.py \
  --episodes 20 \
  --use-swanlab \
  --swanlab-mode cloud \
  --swanlab-project cascade-uav-scheduling \
  --swanlab-experiment e1-main-comparison
```

简单经典对比云端记录：

```bash
python experiments/run_simple_comparison.py \
  --episodes 20 \
  --seed 0 \
  --output-dir outputs/results/simple_comparison_cloud
```

服务器未登录或离线环境可用 offline 模式，仍会保留本地 SwanLab 日志和 PNG 图表：

```bash
python experiments/run_baselines_only.py \
  --episodes 20 \
  --use-swanlab \
  --swanlab-mode offline \
  --output-dir outputs/results/s1_swanlab_offline
```

## 文档

| 文档 | 说明 |
|------|------|
| [CASCADE 方案设计](doc/DECCo论文分析与洪涝灾害应急调度改进方案_CASCADE.md) | 方法论与系统设计 |
| [实验计划 (RL 版)](doc/实验计划_CASCADE_仿真环境与算法框架.md) | RL 方法实验计划 |
| [实验计划 (LLM 版)](doc/实验计划2.0_CASCADE_LLM_Powered.md) | LLM-Powered 实验计划 |
| [LLM 方案设计](doc/CASCADE_LLM_Powered_Planner_方案.md) | LLM 语义规划器方案 |

## 参考文献

1. Zhang, Z., Wu, D., Zhang, F., & Wang, R. (2023). DECCo: A Dynamic Task Scheduling Framework for Heterogeneous Drone Edge Cluster. *Drones*, 7(8), 513.
