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

# 运行启发式 baseline
python experiments/run_baselines_only.py

# 运行完整对比实验
python experiments/run_e1_comparison.py
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
