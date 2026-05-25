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

实验运行在远程 GPU 服务器（AutoDL），SSH 连接信息见 [CLAUDE.md](CLAUDE.md)。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行仿真环境测试
python -m unittest discover -s tests

# 运行启发式 baseline，并生成 JSON/CSV/PNG 图表
python experiments/run_baselines_only.py --episodes 5 --seed 0

# 运行本文新方法 CASCADE
python experiments/run_cascade.py --episodes 5 --seed 0

# 指定输出目录
python experiments/run_baselines_only.py \
  --config configs/env/scenario_s1_dongting.yaml \
  --episodes 5 \
  --output-dir outputs/results/s1_baselines

# 运行 E1 主对比：CASCADE vs baselines
python experiments/run_e1_comparison.py --episodes 5 --seed 0
```

## 实验输入输出

当前有三个常用评估入口：

- `experiments/run_cascade.py`：只运行本文新方法 CASCADE。
- `experiments/run_baselines_only.py`：只运行 Greedy / MinLoad / RoundRobin / HEFT baseline。
- `experiments/run_e1_comparison.py`：运行 E1 主对比，包含 CASCADE + baselines。

默认输入为 `configs/env/scenario_s1_dongting.yaml`，输出目录为 `outputs/results/<experiment>_<timestamp>/`。

每次运行会生成：

- `summary.json`：按方法聚合后的指标。
- `summary.csv`：便于论文表格整理的聚合指标。
- `episodes.csv`：每个 episode 的原始结果。
- `figures/completion_ratio.png`
- `figures/tdsr.png`
- `figures/total_reward.png`
- `figures/rpdr_proxy.png`
- `figures/baseline_radar.png`

推荐先用本地输出确认结果，再把同一命令放到远程服务器上跑更多 episode。

```bash
python experiments/run_cascade.py \
  --config configs/env/scenario_s1_dongting.yaml \
  --episodes 20 \
  --seed 0 \
  --output-dir outputs/results/s1_cascade_20ep

python experiments/run_baselines_only.py \
  --config configs/env/scenario_s1_dongting.yaml \
  --episodes 20 \
  --seed 0 \
  --output-dir outputs/results/s1_baselines_20ep

python experiments/run_e1_comparison.py \
  --config configs/env/scenario_s1_dongting.yaml \
  --episodes 20 \
  --seed 0 \
  --output-dir outputs/results/s1_e1_comparison_20ep
```

注意：当前 `run_cascade.py` 调用的是可运行的 `CASCADEMA3CScheduler` 框架入口，已经包含 action mask + Hungarian 匹配链路；完整 mA3C+MHSA+GNN 训练更新逻辑仍在后续 Phase 3 中继续接入。

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
