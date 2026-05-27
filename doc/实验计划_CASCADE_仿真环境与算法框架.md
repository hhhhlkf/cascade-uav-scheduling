# CASCADE 实验计划：仿真环境与算法框架（RL 版）

---

**版本**：v2.1 — 纯 RL 版本（不含 LLM），对齐 CASCADE.md 最新修订 | **日期**：2026-05-26

**对应方案文档**：[DECCo论文分析与洪涝灾害应急调度改进方案_CASCADE.md](./DECCo论文分析与洪涝灾害应急调度改进方案_CASCADE.md)

---

## 目录

- [1. 实验目标](#1-实验目标)
- [2. 仿真环境设计](#2-仿真环境设计)
- [3. 算法框架设计](#3-算法框架设计)
- [4. 对比方法](#4-对比方法)
- [5. 实验设计](#5-实验设计)
- [6. 评估指标](#6-评估指标)
- [7. 代码仓库结构](#7-代码仓库结构)
- [8. 实施路线图](#8-实施路线图)

---

## 1. 实验目标

### 1.1 核心验证目标

| 编号 | 目标 | 对应 CASCADE.md 章节 |
|------|------|---------------------|
| **G1** | CASCADE (mA3C+MHSA+GNN) 在 Makespan 和 ATCT 上显著优于 DECCo (mA2C) 及其他 DRL/启发式方法 | 3.6 |
| **G2** | 6 维资源向量 + DAG 流水线建模对调度质量和资源利用率的贡献是统计显著的 | 3.5.2, 3.5.1 |
| **G3** | 多目标成本函数（5 项）中，deadline 违反惩罚 ($\lambda_4$) 和优先级加权 ($\lambda_5$) 对高优先级任务的 PTCT 缩短贡献最大 | 3.5.3 |
| **G4** | 距离感知 + 多跳中继路由 + GNN 网络拓扑编码在网络条件动态变化时，GPU 负载均衡 ($\sigma_{gpu}$) 和 Makespan 退化显著小于无拓扑感知的方法 | 3.5.4 |
| **G5** | 三级降级运行模式在 UAV 失效场景中，任务链 Makespan 增幅显著小于无降级机制的方法 | 3.7.1 |
| **G6** | 多模态模型自适应压缩在带宽受限场景中，对大模型融合任务 PTCT 的缩短有显著贡献 | 3.7.3 |

### 1.2 实验范围限定

- **全部在仿真环境中进行**，不涉及真实无人机硬件，不依赖真实航拍或遥感数据集
- **不涉及 LLM**，调度决策完全由 RL 策略网络 / 启发式算法 / 数学优化方法给出
- 仿真环境需模拟：多无人机运动（简化）、距离感知 Mesh 通信网络（含多跳中继）、参数化任务链生成（5~15 子任务）、任务 DAG 依赖与执行
- 算法框架在 Python 3.10+ 环境下运行
- **实验运行环境**：远程 GPU 服务器（AutoDL 等云平台），非本地运行。SSH 连接信息见项目根目录 [CLAUDE.md](../CLAUDE.md)

---

## 2. 仿真环境设计

### 2.1 设计原则

**实现状态（2026-05-27）**：完成（Phase 1 仿真环境层面）。模块化、Gymnasium 接口、YAML 配置和纯仿真数据已落地；环境主时钟已通过 SimPy `Environment.run(until=...)` 推进固定调度步长，满足当前 RL step 驱动的离散事件仿真需求。完整的异步事件队列和自适应步长可作为后续精细化仿真增强项，不阻塞 Phase 1 环境完成。

- **离散事件驱动**：使用 SimPy 离散事件仿真引擎，适合建模任务到达、调度决策、任务执行、通信变化等离散事件
- **模块化架构**：场景生成器、无人机模拟器、通信网络模拟器、任务管理器四个核心模块独立可替换
- **Gymnasium 兼容接口**：对外暴露 `reset()` / `step(action)` 标准 RL 接口，兼容 Stable-Baselines3 / RLlib / Tianshou 等主流 RL 框架
- **配置化**：全部场景参数通过 YAML/JSON 配置文件注入，支持快速切换实验条件
- **纯仿真数据**：场景参数与任务链由参数化生成器产生，不依赖外部数据集；评估指标聚焦于任务完成时效和资源利用效率

### 2.2 技术选型

**实现状态（2026-05-27）**：环境层面完成。Gymnasium、PyTorch、SciPy Hungarian、YAML 配置、自定义通信仿真已接入；当前实验追踪使用 SwanLab。PyTorch Geometric 属于 Phase 2/3 的图网络算法实现，OR-Tools 属于可选 MILP 验证工具，Hydra/OmegaConf 会改变现有运行入口，MLflow/TensorBoard 与当前 SwanLab 追踪栈重复，因此这些工具不作为 Phase 1 仿真环境完成条件。

| 组件 | 选型 | 理由 |
|------|------|------|
| 离散事件引擎 | **SimPy** | Python 原生、轻量、学术界广泛使用 |
| RL 环境接口 | **Gymnasium** | 标准接口，兼容主流 RL 框架 |
| 图神经网络 | **PyTorch Geometric** | GAT（任务 DAG 编码）、GCN（网络拓扑编码） |
| 数值优化 | **SciPy + OR-Tools** | Hungarian 算法（动作掩码后的匹配）、MILP 验证 |
| 通信仿真 | **自定义 SimPy 模块** | NS-3 太重，自定义简化模型即可满足研究需要 |
| 配置管理 | **Hydra (OmegaConf)** | YAML 配置 + 命令行覆盖 |
| 实验追踪 | **MLflow + TensorBoard** | 实验指标记录与可视化 |
| RL 训练后端 | **PyTorch** | mA3C/MADDPG/QMIX 等均基于 PyTorch 实现 |

### 2.3 核心模块详设

**实现状态总览（2026-05-27）**

| 模块/步骤 | 状态 | 当前落地位置 | 说明 |
|-----------|------|--------------|------|
| 2.3.1 `ScenarioGenerator` | 完成 | `src/env/scenario_generator.py`, `configs/env/scenario_ds*.yaml` | 已支持 DS1/DS2/DS3、7 维参数采样、2~5 区域、每区域 5~15 DAG 任务链、A/P/I/F/C 触发规则、UAV 编队生成，并输出建筑、道路和简易高程模型元数据。 |
| 2.3.2 `UAVSimulator` | 完成 | `src/env/uav_simulator.py` | 已支持任务分配、飞行耗时、资源占用/释放、电池消耗、状态切换、故障中断、GPU/显存利用率采样、转弯半径、风扰动和传感器故障。 |
| 2.3.3 `MeshNetworkSimulator` | 完成 | `src/env/network_simulator.py` | 已支持距离驱动链路、RSS、带宽、延迟、通信损毁、多跳最短路径、瓶颈带宽/端到端延迟特征、terrain LoS 概率、K=3 备选路径和完整链路边属性矩阵。 |
| 2.3.4 `TaskManager` | 完成 | `src/env/task_manager.py`, `src/env/scenario_generator.py` | 任务链生成在 `ScenarioGenerator` 中完成；`TaskManager` 已支持 DAG 依赖、READY 解锁、调度、完成、超时、抢占、紧急注入、DAG 统计和 `get_pending_count()`。 |
| 2.3.5 `CASCADEEnv` | 完成 | `src/env/cascade_env.py`, `src/env/action_mask.py`, `src/env/reward.py` | 已支持 `reset()`/`step()`/`get_action_mask()`、区域化 ready task、连续动作矩阵、动作掩码、5 项 reward、区域/多跳/DAG/UAV/链路边属性观测、SimPy 时钟推进、Hungarian 解码和 episode 指标。 |
| 2.3.6 Episode/Step 输入输出 | 完成 | `experiments/smoke_env.py`, `experiments/run_cascade.py`, `tests/test_env/` | 环境侧 episode/step 输入输出、DS1 小规模实验、指标输出与图表生成已验证；10,000 episode 长训练属于 Phase 3 训练任务，不属于本节仿真环境完成条件。 |

#### 2.3.1 场景生成器 (`ScenarioGenerator`)

**实现状态（2026-05-27）**：完成。DS1/DS2/DS3 配置、7 维参数采样、区域化任务链、任务 DAG、UAV 编队生成、建筑/道路分布和简易高程模型元数据已完成并通过测试。

负责生成洪涝灾害仿真场景的全部初始状态。采用参数化模型，从 7 维参数空间中随机采样生成多样化场景。

**功能**：
- 根据灾情参数生成受灾区域（含空间坐标与简易高程模型）
- 根据建筑物密度生成建筑/道路/障碍物分布
- 生成被困人员分布（依赖被困人员密度参数，随机散落 + 热点聚类）
- 生成溃口位置（依赖溃口风险等级）
- 依据空间聚类将受灾区域自动划分为 2~5 个子区域
- 为每个子区域按参数触发规则生成定制化任务链（5~15 子任务）
- 生成无人机编队初始配置（位置、板卡规格、传感器挂载）
- 生成指挥车固定位置

**关键可配置参数**（对齐 CASCADE.md 3.8.1）：

| 参数 | 取值范围 | 说明 |
|------|---------|------|
| `disaster_area_km2` | 10~100 | 区域总面积，影响任务链长度与覆盖密度 |
| `flood_ratio` | 0.2~0.8 | 淹没比例，影响多光谱/高光谱任务触发 |
| `building_density` | low/medium/high | 建筑物密度，影响 RGB 目标检测任务密度 |
| `civilian_density` | 1~50 人/km² | 被困人员密度，影响热红外搜索任务 deadline 紧迫度 |
| `comm_failure_rate` | 0.0~0.9 | 通信损毁程度，决定 Mesh 中继需求 |
| `breach_risk_level` | 0~3 | 溃口风险等级，影响高光谱及 F4 触发 |
| `terrain_roughness` | flat/hilly/complex | 地形起伏度，影响航迹规划与 LoS 概率 |
| `num_regions` | 2~5 | 子区域数量（自动聚类） |
| `num_uavs_total` | 6~15 | 无人机总数 |
| `uav_orin_8gb_ratio` | 0.2~0.5 | Orin NX 8GB 板卡比例（其余为 16GB） |

每个仿真 episode 从上述参数空间中随机采样，生成一个独特的洪涝场景实例。

**参数→子区域特征→任务链生成流程**：

```
1. 采样全局参数 → 生成受灾场景
2. 空间聚类 → 划分为 2~5 个子区域，每个子区域具有局部特征向量
3. 对每个子区域，按触发规则确定任务模态:
   - A1 (RGB) 始终触发
   - A2 (热红外) ⟺ civilian_density ≥ 5
   - A4 (多光谱) ⟺ flood_ratio ≥ 0.3
   - A5 (高光谱) ⟺ breach_risk ≥ 2 或 flood_ratio ≥ 0.6
   - C1 (中继) ⟺ comm_failure_rate ≥ 0.3
   - ... (其余按 CASCADE.md 3.8.1 触发规则表)
4. 为每个子区域生成 5~15 个任务节点的 DAG
5. 输出完整的任务链集合 + UAV 编队配置
```

#### 2.3.2 无人机模拟器 (`UAVSimulator`)

**实现状态（2026-05-27）**：完成。任务执行、资源占用释放、电池功耗、状态机、故障中断、GPU/显存利用率采样、风速扰动、转弯半径和传感器故障已完成；Orin 8GB/16GB 资源差异通过 `ResourceVec` 表达，满足 Phase 1 调度仿真需要。

**功能**：
- **运动模拟**（简化）：匀速巡航 15 m/s，转弯半径 30 m，风速随机扰动 ±2 m/s
- **电池模型**：不同状态的功耗不同（见下表），电池容量归一化为 100%
- **传感器采集**：RGB/热红外/多光谱/高光谱，各有不同的采集时间和数据产出量
- **边缘计算**：GPU/CPU 资源占用与释放，推理延迟取决于模型大小
- **异常事件**：按概率触发失联、电量耗尽、传感器故障

**单架 UAV 核心属性**：

```
uav_id: str                        # 唯一标识
uav_type: UAVType                  # UAV-V | UAV-M | UAV-H | UAV-C | UAV-R
orin_spec: OrinNXSpec             # 8GB | 16GB 显存规格
position: (x, y, z)                # 三维坐标（时变）
battery_level: float               # 0.0~1.0
cpu_available: float               # 0.0~1.0 归一化剩余 CPU
gpu_available: float               # 0.0~1.0 归一化剩余 GPU
memory_available_gb: float         # 剩余显存 GB（8GB 板卡上限 8GB，16GB 上限 16GB）
storage_available_gb: float        # 剩余存储 GB
bandwidth_available_mbps: float    # 当前上行带宽（由网络仿真动态更新）
current_tasks: List[Task]          # 当前执行中的任务（最多 2 个并发）
status: UAVStatus                  # IDLE | TRANSIT | COLLECTING | PROCESSING | RELAYING | FAULTED
```

**电池功耗模型**（简化的分状态恒功率模型）：

| 状态 | 功耗 | 说明 |
|------|------|------|
| HOVER | 200 W | 悬停等待，保持位置 |
| TRANSIT | 350 W | 15 m/s 巡航飞行 |
| COLLECTING | 15 W | 传感器采集（快门 + 读出电路 + 初步压缩） |
| PROCESSING (GPU) | 25 W | 边缘 GPU 推理（Jetson Orin NX 16GB 典型值，8GB 约 20W） |
| RELAYING | 90 W | 通信中继（射频功放占主要功耗） |

> 注：功耗值参考了 DJI M300 级别工业无人机的实际飞行功耗（~1000W 包含动力系统 60-70%），加上 Jetson Orin NX 的计算功耗（10-25W）。此处数字为动力之外的设备功耗。

#### 2.3.3 通信网络模拟器 (`MeshNetworkSimulator`)

**实现状态（2026-05-27）**：完成。距离驱动链路、RSS/带宽/延迟、多跳最短路径、瓶颈带宽、端到端延迟特征、LoS 地形概率、K=3 备选路径和完整 `network_edge_attrs` 已完成。

**设计思路**：网络状态完全由**节点间的三维空间距离**驱动，与 CASCADE.md 3.5.4 节的距离感知模型一致。所有链路属性从距离计算推导，支持多跳中继路由。

**功能**：
- 每时隙更新所有 UAV 和指挥车的三维坐标
- 计算所有节点对之间的欧氏距离 $d_{ij}$
- 基于距离 + 路径损耗模型推导每条链路的 RSS、带宽、延迟
- 构建可达性图并运行 Dijkstra 计算每架 UAV 到指挥车的最短多跳路径
- 输出瓶颈带宽和端到端延迟供调度器决策使用

**每条链路的属性**：

```
link_id: (src_id, dst_id)
distance_m: float               # 节点间欧氏距离
path_loss_db: float              # 路径损耗 (dB)
rssi_dbm: float                  # 接收信号强度
connected: bool                  # RSS ≥ RSS_min (-82 dBm)
bandwidth_mbps: float            # 由 RSS → MCS 等级查表得到
latency_ms: float                # 传播延迟 + 排队延迟
hop_count: int                   # 到达指挥车所需跳数（直连 = 1）
path_bottleneck_bw_mbps: float   # 到指挥车路径的瓶颈带宽
path_end_to_end_delay_ms: float  # 到指挥车的端到端延迟
```

**距离驱动的链路评估流程**（每时隙执行）：

```
1. 位置更新:
   指挥车: 固定位置 p_b
   UAV:   p_i(t) = p_i(t-1) + v_i · Δt  (沿航迹移动)

2. 距离计算:
   d_{ij} = ‖p_i - p_j‖₂  (所有节点对)
   d_{i,b} = ‖p_i - p_b‖₂  (各UAV到指挥车)

3. 链路评估 (每对节点):
   PL = 40 + 28·log10(d_{ij}) + N(0, σ²)  (城郊路径损耗, σ=6dB)
   RSS = 20 dBm - PL                       (WiFi 6 典型发射功率)
   可连通 ⟺ RSS ≥ -82 dBm                  (MCS0 最低灵敏度)
   bw = lookup_mcs_table(RSS)              (MCS等级→带宽查表)
   delay = d_{ij} / 3e8 + queuing_delay

4. 通信损毁:
   按 comm_failure_rate 比例随机剔除指挥车侧可用链路

5. 多跳路由:
   以指挥车为根，在连通图上运行 Dijkstra
   K=3 条备选最短路径
   瓶颈带宽 = min(路径各段 bw)
   端到端延迟 = sum(路径各段 delay)

6. LoS 概率:
   由 terrain_roughness 决定:
   - flat: P_LoS ≈ 1.0
   - hilly: P_LoS = exp(-d/500)
   - complex: P_LoS = exp(-d/300)
   NLoS 额外衰减: 20~30 dB
```

**关键仿真参数**（对齐 CASCADE.md 3.8.1）：

| 参数 | 值 | 说明 |
|------|-----|------|
| WiFi 6 载频 | 5.8 GHz | 无人机间通信频段 |
| 发射功率 $P_{tx}$ | 20 dBm | Orin NX 挂载 WiFi 模块典型值 |
| 接收灵敏度 $RSS_{min}$ | -82 dBm | MCS0 (最低速率) |
| 路径损耗指数 $\alpha$ | 2.8 | 城郊混合场景 |
| 阴影衰落 $\sigma$ | 6 dB | 受建筑物和地形遮挡 |
| 最大通信距离 | ~1.5 km (LoS) | 由灵敏度和路径损耗公式反推 |

**多跳中继的关键约束**：每增加一跳，有效带宽取瓶颈链路的最小值，端到端延迟累加。调度器需权衡——经过 2 跳中继回传 8MB 图像可能远超在本地 Orin NX 上直接推理的耗时。

#### 2.3.4 任务管理器 (`TaskManager`)

**实现状态（2026-05-27）**：完成。任务生命周期、DAG 依赖解锁、READY 任务查询、区域过滤、完成/超时/抢占/紧急注入、DAG 统计和 `get_pending_count()` 已完成；任务链生成由 `ScenarioGenerator` 统一负责，`TaskManager` 专注运行时状态管理。

**功能**：
- 根据子区域特征向量和参数触发规则自动生成任务链（5~15 子任务）
- 管理任务全生命周期（生成 → 排队 → 调度 → 执行 → 完成 / 超时）
- 维护任务 DAG 依赖图，追踪每个任务的实时进度
- 注入随机紧急事件（模拟灾情变化，如突发溃口导致新增高光谱检测任务）

**任务链生成规则**（对齐 CASCADE.md 3.8.1 触发规则表）：

每个子区域的任务链由参数条件决定哪些任务类型被激活：

| 任务类型 | 触发条件 | 数量 |
|---------|---------|:--:|
| A1 RGB 航拍采集 | 始终触发 | 1 |
| A2 热红外扫描 | 被困人员密度 ≥ 5 | 0~1 |
| A3 重点区域补拍 | 建筑物密度 = 中或高 | 0~1 |
| A4 多光谱扫描 | 淹没比例 ≥ 0.3 | 0~1 |
| A5 高光谱扫描 | 溃口风险 ≥ 2 或淹没比例 ≥ 0.6 | 0~1 |
| P1~P4 预处理 | 对应采集存在则生成 | 与 A 类一一对应 |
| I1~I5 推理 | 对应采集+预处理存在 | 2~5 |
| F1~F4 融合 | 相关推理结果同时存在 | 0~3 |
| C1 中继维持 | 通信损毁程度 ≥ 0.3 | 0~1 |
| C2 数据回传 | 始终触发 | 1~2 |

简单场景（低密度农村、无溃口、通信完好）约 5~7 个子任务；复杂场景（城区高密度、活跃溃口、通信瘫痪）可达 13~15 个。

**任务状态机**：

```
PENDING → READY (依赖满足) → QUEUED → SCHEDULED → EXECUTING → COMPLETED
                                                    ↓               ↓
                                                PREEMPTED       TIMEOUT
```

**Task 数据结构**：

```
task_id: str                       # 唯一标识
task_type: TaskType                # A1-A5 | P1-P4 | I1-I5 | F1-F4 | C1-C2
region_id: str                     # 所属子区域
modality: ModalityType             # RGB | THERMAL | MULTISPECTRAL | HYPERSPECTRAL | MIXED
resource_requirement: ResourceVec  # 6 维资源需求向量 [CPU, GPU, Mem, Storage, BW, Latency]
priority: int                      # 1-10 优先级
deadline_min: float                # 相对 deadline（分钟）
depends_on: List[str]              # 前驱 task_id 列表
depended_by: List[str]             # 后继 task_id 列表
estimated_flops: float             # 估算计算量 (FLOPs)，用于推理延迟估算
data_size_mb: float                # 任务产出/需求数据量，用于传输代价计算
status: TaskStatus
assigned_uav: Optional[str]
start_time: Optional[float]
completion_time: Optional[float]
```

**关键方法**：

- `generate_task_chains(scenario_params, regions) -> Dict[str, List[Task]]` — 按参数触发规则为每个子区域生成任务链及 DAG
- `get_ready_tasks() -> List[Task]` — 返回所有区域中依赖已满足、可调度的任务
- `update_dag(completed_task_id)` — 前驱完成后解锁后继
- `inject_emergency(task: Task)` — 注入突发紧急任务
- `get_dag_stats() -> Dict` — 返回 DAG 统计信息（总节点、关键路径长度、并行度、各类型任务数量）
- `get_pending_count() -> int` — 返回待完成任务总数

#### 2.3.5 仿真环境主环境 (`CASCADEEnv`)

**实现状态（2026-05-27）**：完成。Gymnasium 风格接口、区域化观测、动作掩码、reward、事件注入、网络更新、SimPy 时钟推进、完整边属性矩阵、统一 Hungarian 解码和 6 项核心 episode 指标已完成；适应性步长属于后续精细化仿真增强项，不阻塞当前固定步长 RL 环境。

```python
# 核心伪代码
class CASCADEEnv(gym.Env):
    """
    Gymnasium 兼容的 CASCADE 仿真环境。

    观察空间（每步返回的 dict，仅包含当前区域的待调度任务）:
      - current_region: 当前聚焦的子区域 ID (str)
      - region_task_count: 当前区域剩余未完成任务数 (int)
      - task_features: 当前区域就绪任务特征矩阵 (N_ready × 8)
        列: [task_type_id, priority, deadline_remaining, estimated_flops,
              data_size_mb, cpu_req, gpu_req, memory_req_gb]
      - task_dag_adj: 当前区域任务 DAG 邻接矩阵 (N_region_tasks × N_region_tasks)
      - task_dag_mask: 当前区域依赖已满足的任务的二元掩码 (N_region_tasks,)
      - uav_features: 所有 UAV 状态矩阵 (N_uavs × 10)
        列: [x, y, z, battery, cpu_avail, gpu_avail, mem_avail_gb,
              orin_spec(0=8G,1=16G), status_id, current_task_count]
      - uav_dist_to_base: 各 UAV 到指挥车的欧氏距离 (N_uavs,)
      - network_adj: 直连链路邻接矩阵 (N_nodes × N_nodes)
      - network_edge_attrs: 链路属性 (N_nodes × N_nodes × 4)
        [distance, bw_mbps, delay_ms, connected]
      - multihop_features: 各 UAV 到指挥车的多跳路径特征 (N_uavs × 4)
        [hop_count, bottleneck_bw_mbps, end_to_end_delay_ms, has_direct_link]

    动作空间:
      连续动作: 分配概率矩阵 P ∈ [0,1]^{N_ready_current_region × N_uavs}
      经 Hungarian 算法离散化 → (task_idx, uav_idx) 分配对
      动作掩码自动过滤不可行组合（仅对当前区域的任务）
    """

    def __init__(self, config: Dict):
        self.scenario = ScenarioGenerator(config).generate()
        self.uav_simulators = [UAVSimulator(cfg) for cfg in self.scenario.uavs]
        self.network = MeshNetworkSimulator(self.scenario.topo_config)
        self.task_manager = TaskManager(self.scenario.task_config)
        self.sim_time = 0.0

        # Gym 接口
        self.observation_space = self._build_obs_space()
        self.action_space = self._build_action_space()

    def reset(self, seed=None) -> Tuple[Obs, Info]:
        """重置仿真到初始状态"""
        ...

    def step(self, action: np.ndarray) -> Tuple[Obs, float, bool, bool, Info]:
        """
        执行一个调度决策，推进仿真。

        动作编码: 当前区域就绪任务的分配概率矩阵 P，经 Hungarian 匹配后得到 (task, uav) 对
        """
        # 1. 将 action 解码为当前区域的 (task, uav) 分配
        # 2. 验证分配有效性（资源、网络可达性、依赖满足）
        # 3. 执行分配，更新 UAV 计算资源占用
        # 4. 推进仿真 dt（适应性步长：下一个事件或固定 5s）
        # 5. 更新所有 UAV 位置、功耗、计算进度
        # 6. 更新通信网络：距离→RSS→带宽→多跳路由
        # 7. 检查任务完成 / 超时 / 解锁后继（所有区域）
        # 8. 概率性地注入随机事件
        # 9. 切换到下一个有就绪任务的子区域（轮转）
        # 10. 计算即时 reward（基于 5 项成本函数）
        # 11. 构建下一个观察（新区域的就绪任务 + 全局 UAV/网络状态）
        ...

    def compute_reward(self) -> float:
        """
        基于 CASCADE.md 第 3.5.3 节的 5 项成本函数计算即时 reward。

        C = λ1 * C_load + λ2 * C_overload + λ3 * C_latency
          + λ4 * C_deadline + λ5 * C_priority

        C_latency 含多跳路径延迟 (CASCADE.md 3.5.4):
          对需要数据传输的任务，使用 bw_path 和 delay_path 而非直连值

        reward = -C （最小化成本 = 最大化 reward）

        推荐权重 (CASCADE.md):
          λ1=0.15, λ2=0.25, λ3=0.10, λ4=0.30, λ5=0.20
        """
        ...

    def get_action_mask(self) -> np.ndarray:
        """
        返回动作掩码 M ∈ {0,1}^{N_ready × N_uavs}，过滤不可行的 (task, uav) 组合：
        - UAV 处于 FAULTED 状态
        - UAV 的 6 维可用资源不满足 task 需求
        - task 有传感器类型要求但 UAV 不搭载该传感器
        - task 需要数据传输但 UAV 到指挥车无可达路径（含多跳）
        - UAV 的 Orin NX 板卡显存不足以加载所需模型（如 8GB 板卡接到 F3 全模态融合）
        - UAV 当前任务数已达上限 (max_concurrent=2)
        """
        ...

    def get_episode_metrics(self) -> Dict:
        """
        Episode 结束时调用，返回 6 项评估指标:
          - makespan: 任务链总完成时间
          - atct: 平均单任务完成时间
          - ptct: 按任务类型 (A/P/I/F/C) 分组的平均完成时间
          - gpu_util_mean: GPU 利用率均值
          - gpu_util_std: GPU 利用率标准差
          - mem_util_mean: 显存利用率均值
        """
        ...
```

**关键设计决策：MultiDiscrete vs 连续动作空间**

| 方案 | 动作表示 | 优点 | 缺点 |
|------|---------|------|------|
| A: MultiDiscrete | `[task_idx, uav_idx]` | 简单，兼容性强 | 一次只能调一个任务 |
| **B: 连续 + 掩码（推荐）** | 分配矩阵 `P ∈ [0,1]^{N_tasks × N_uavs}`，经 Hungarian 离散化 | 支持并发多任务调度 | 实现稍复杂 |

> **推荐方案 B**，更贴合 CASCADE.md 3.6.3 节中"各 UAV Actor 输出任务接受概率向量→汇总后 Hungarian 匹配"的 CTDE 设计。

#### 2.3.6 训练步示例：Episode 与 Step 的输入/输出

**实现状态（2026-05-27）**：完成。环境侧 episode/step 输入输出已经可以运行并通过 `smoke_env.py`、`run_cascade.py` 和单元测试验证；完整训练循环、多环境并行采样和 mA3C 更新属于 Phase 3 训练算法，不属于本节仿真环境完成条件。

以下用一个具体实例说明 10,000 个 training episode 中每一步的输入输出关系。

**Episode 结构**：一个 episode 覆盖一次完整灾情场景，包含该场景下全部子区域（2~5 个）的所有任务链从头到尾执行完毕。**每步聚焦一个子区域**，调度器按轮转顺序在各区域间切换——当前步只看到当前区域的待调度任务，但所有 UAV 是跨区域共享的。

**示例 Episode（DS1 典型采样）**：

```
Episode #k:
  采样参数: area=36km², flood=0.5, building=high, civilian=30, comm=0.3, breach=2, terrain=hilly
  → 生成 3 个子区域:
    区域A (溃口区):  任务链 12 个 (A1,A2,A4,A5 + P×4 + I1,I2,I3,I5 + F1 + C2)
    区域B (淹没城区): 任务链 14 个 (A1,A2,A3,A4 + P×4 + I1,I2,I3,I4 + F1,F3 + C1,C2)
    区域C (高地区):   任务链  7 个 (A1,A4 + P×2 + I1,I3 + C2)
  → UAV 编队: 10 架 (16GB×6 + 8GB×4)，所有区域共享
  → 区域轮转顺序: A → B → C → A → B → C → ...（循环）
```

**Step 1**——聚焦区域 A（溃口区）：

```
当前区域: A
输入 (观察):
  ready_tasks: 4 个 (区域A的A类采集，无依赖)
    [A1@A(RGB采集), A2@A(热红外), A4@A(多光谱), A5@A(高光谱)]
  uav_states: 10 架全部 IDLE，资源满，初始位置
  network: 各UAV→指挥车距离与直连/多跳路径
  action_mask[4×10]: 过滤传感器不匹配的 (task,uav) 对
    - A5@A(高光谱) 只能分配给 UAV-H (搭载高光谱仪)
    - UAV-C1/C2 (纯中继) 所有采集任务被掩码

调度决策:
  Actor 输出 P[4×10] → Hungarian → 分配:
    A1@A→UAV-V1, A2@A→UAV-V2, A4@A→UAV-M1, A5@A→UAV-H1
  → 4 个全部分配，仿真推进，UAV 开始执行

输出:
  reward: 基于 5 项成本（区域A的负载均衡+通信延迟）
  done: False (区域A还有 8 个任务，B/C 各有一整条链待做)
```

**Step 2**——切换到区域 B（淹没城区）：

```
当前区域: B
输入 (观察):
  ready_tasks: 4 个 (区域B的A类采集)
    [A1@B, A2@B, A3@B, A4@B]
  uav_states: 4 架 BUSY (V1,V2,M1,H1 正在执行区域A的采集)
              6 架 IDLE (V3,V4,M2,C1,C2,R1)
              各 UAV 位置已更新（BUSY 的在区域A上空，IDLE 的可调动）
  action_mask[4×6]: 仅 6 架 IDLE UAV 可分配
    - A3@B(补拍) 优先分配给搭载 RGB 相机的 UAV

调度决策: 分配 3 个（A4@B 的 UAV-M2 还在赶往区域B的路上，暂缓）
  A1@B→UAV-V3, A2@B→UAV-V4, A3@B→UAV-R1

输出:
  reward: 基于即时成本
  done: False
```

**Step 3**——切换到区域 C（高地区）：

```
当前区域: C
输入 (观察):
  ready_tasks: 2 个 (区域C任务少，仅 A1,A4)
    [A1@C, A4@C]
  uav_states: V1 刚完成 A1@A 采集→IDLE, V2 仍在执行 A2@A
              V3,V4,R1 BUSY(区域B), M1 BUSY(区域A), H1 BUSY(区域A)
              M2,C1,C2 IDLE, V1 新释放→IDLE
  action_mask[2×5]: 5 架 IDLE UAV 可分配

调度决策:
  A1@C→UAV-V1 (刚完成区域A任务，距区域C最近)
  A4@C→UAV-M2
```

**Step 4**——轮转回区域 A：

```
当前区域: A
输入 (观察):
  ready_tasks: 3 个 (区域A采集完成，P类预处理解锁)
    [P1@A(RGB校正), P2@A(辐射校正), P4@A(温度校准)]
  uav_states: 部分 BUSY(区域B/C 采集)，部分 IDLE
  ...
```

**Step k**——某中间步，轮转到区域 B：

```
当前区域: B
输入 (观察):
  ready_tasks: 2 个 (I1,I2 推理已解锁)
    [I1@B(RGB目标检测), I2@B(热红外人员检测, priority=10)]
  uav_states: 混合忙碌/空闲
  action_mask: I1 需要 GPU→仅 16GB 板卡可接; I2 priority=10 优先

调度决策: I2@B→UAV-V4(I2 deadline=2min 紧急，抢占优先)
          I1@B→UAV-V1(16GB GPU 满足 YOLO 推理)
```

**Episode 结束**：所有 3 个区域的 33 个任务完成或超时 → `done=True` → 调用 `get_episode_metrics()` 计算所有区域的合计 Makespan 及 6 项指标 → 下一个 episode 重新采样全新场景。

**轮转策略说明**：
- 默认按固定顺序轮转（A→B→C→A→...），实现简单且保证每个区域都获得调度机会
- 当某区域的就绪任务为空时（所有任务要么完成要么被阻塞），自动跳过该区域
- 可选增强：由调度器学习区域选择策略（如优先调度有高 priority 任务的区域），但在初期采用固定轮转以降低训练复杂度

**10,000 episodes 的意义**：每个 episode 的场景参数、区域数量、任务链构成独立随机采样，调度策略在数千种不同配置上训练，学习的是"面对任意区域特征和 UAV 状态，如何为该区域的任务做出最优分配"的泛化能力。

---

## 3. 算法框架设计

### 3.1 调度算法抽象基类

```python
from abc import ABC, abstractmethod

class BaseScheduler(ABC):
    """所有调度算法的统一接口"""

    @abstractmethod
    def observe(self, obs: Dict[str, np.ndarray]) -> None:
        """接收环境观测，更新内部状态"""
        ...

    @abstractmethod
    def decide(self, action_mask: np.ndarray) -> np.ndarray:
        """
        返回调度动作。
        action 格式取决于具体实现，统一为分配矩阵 P ∈ [0,1]^{N_tasks × N_uavs}
        """
        ...

    @abstractmethod
    def learn(self, batch: Dict) -> Dict[str, float]:
        """
        从经验批次中学习。返回训练指标 dict（如 loss, entropy 等）。
        启发式方法此法为空操作。
        """
        ...

    def reset(self) -> None:
        """重置调度器内部状态（episode 开始时调用）"""
        pass

    def save(self, path: str) -> None:
        """保存模型 checkpoint"""
        pass

    def load(self, path: str) -> None:
        """加载模型 checkpoint"""
        pass
```

### 3.2 CASCADE (mA3C+MHSA+GNN) — 本文方法

对应 CASCADE.md 第 3.6 节。

#### 3.2.1 核心改进点（相对于 DECCo mA2C）

| 维度 | DECCo (mA2C) | CASCADE (mA3C+MHSA+GNN) |
|------|-------------|------------------------|
| Agent 架构 | 单 Agent | 多 Agent（指挥车 Critic + 每 UAV 一个 Actor） |
| 状态编码 | 向量拼接 | GAT 编码任务 DAG + GCN 编码距离感知网络拓扑 + MHSA 融合 |
| 动作空间 | 选 1 架 UAV 执行当前任务 | 并发多 UAV × 多任务，经 Hungarian 离散化 |
| 协作机制 | 无显式协作 | 指挥车广播压缩隐状态，各 UAV Actor 通信注意力 |
| 任务依赖 | 忽略 | DAG GAT 编码，关键路径感知 |
| 优先级/Deadline | 无 | 抢占式优先级队列 + 5 项成本函数含 deadline 惩罚 |
| 资源模型 | 2 维 (CPU, Mem) | 6 维 (+GPU, Storage, BW, Energy)，含 Orin NX 8GB/16GB 显存异构 |
| 网络模型 | 固定 LAN | 三维距离感知 + 多跳中继 + 瓶颈带宽 + GNN 拓扑编码 |

#### 3.2.2 网络结构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CASCADE 状态编码器                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  输入:                                                               │
│  ├─ 任务 DAG 图 G_task  ──→ GAT (2层, 4头) ───→ h_task ∈ R^128     │
│  ├─ 网络拓扑图 G_net    ──→ GCN (2层)      ───→ h_net  ∈ R^128     │
│  │   节点特征: [x, y, z, d_to_base, cpu_avail, gpu_avail,          │
│  │              mem_avail, orin_spec, status]                       │
│  │   边特征:   [distance, bw, delay, connected]                     │
│  ├─ UAV 资源矩阵 R_all  ──→ MLP (2层, 256)  ───→ h_res  ∈ R^128    │
│  ├─ 多模态数据流特征    ──→ Cross-Modal Atten ─→ h_modal ∈ R^128    │
│  └─ 多跳路径特征 H_multi──→ MLP (128)        ───→ h_hop   ∈ R^64   │
│      (hop_count, bottleneck_bw, e2e_delay, has_direct_link) × N_uavs│
│                                                                      │
│  融合: MHSA(Concat([h_task, h_net, h_res, h_modal, h_hop]))         │
│       → h_global ∈ R^320  (供 Critic 使用)                           │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                    CASCADE Actor-Critic                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Critic (指挥车):                                                    │
│    h_global → MLP(320→128→1) → V(s)  (状态价值)                     │
│                                                                      │
│  Actor_i (UAV i):                                                    │
│    [h_global, local_obs_i] → MLP(320+64→128→|V_ready|)              │
│    其中 local_obs_i 包含: 自身资源、位置、到指挥车距离/跳数/瓶颈带宽  │
│    → 经 softmax → p_i ∈ [0,1]^{|V_ready|}  (任务接受概率)           │
│                                                                      │
│  汇总匹配 (指挥车):                                                  │
│    P = Stack([p_1, p_2, ..., p_U])                                   │
│    → Action Mask ⊙ P → Hungarian Algorithm → 分配矩阵 A             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### 3.2.3 mA3C 训练流程

```
超参数：
  - 学习率 lr_actor=3e-4, lr_critic=1e-3
  - 折扣因子 γ=0.99
  - GAE λ=0.95
  - 熵系数 β=0.01（鼓励探索）
  - 梯度裁剪 max_norm=0.5
  - N 步回报 n_steps=128
  - 并行环境数 n_envs=8（A3C 异步训练）

训练循环:
  for episode in range(N_episodes):
      1. 并行采样 n_envs 个环境，每个收集 n_steps 步
          → n_envs × n_steps 条 transition
      2. 计算 GAE 优势函数 A_t
      3. 计算 N 步回报 R_t
      4. Critic 更新: minimize MSE(V(s_t), R_t)
      5. Actor 更新: maximize A_t * log π(a_t|s_t) + β * H(π)
      6. 软更新 Target 网络: θ_target = τ*θ + (1-τ)*θ_target (τ=0.005)
      7. 记录训练指标 (loss, entropy, advantage, reward)
      8. 每 K episodes 在验证场景上评估
```

#### 3.2.4 动作掩码机制

对应 DECCo 原论文的 $F_t \in \{0,1\}^{U+1}$ 掩码向量，CASCADE 扩展为二维掩码矩阵 $M_t \in \{0,1\}^{N_{ready} \times U}$：

```
M[t][u] = 0 当且仅当：
  - UAV u 状态为 FAULTED
  - UAV u 的 6 维可用资源不满足 task t 的需求
  - UAV u 的 Orin NX 板卡显存不足以加载任务所需模型
    (如 8GB 板卡无法执行 F3 全模态融合 8GB 显存需求)
  - task t 有传感器类型要求但 UAV u 不搭载该传感器
  - task t 需要网络连接但 UAV u 当前无可达路径（含多跳中继）
  - UAV u 当前任务数已达上限 (max_concurrent=2)

在 Actor 输出 softmax 之前，应用掩码:
  p_masked = softmax(logits - (1-M) * 1e9)
  (不可行的 task 概率被压到 ~0)
```

---

## 4. 对比方法

### 4.1 方法总览

| 类别 | 方法 | 论文/来源 | 复现方式 | 优先级 |
|------|------|----------|---------|--------|
| **本文方法** | **CASCADE (mA3C+MHSA+GNN)** | 本文 | 自研 | P0 |
| DRL — 单 Agent | **DQN** | Mnih et al. 2015 | Stable-Baselines3 / 自研 | P1 |
| DRL — 单 Agent | **PPO** | Schulman et al. 2017 | Stable-Baselines3 | P1 |
| DRL — 多 Agent | **DECCo (mA2C)** | Zhang et al. 2023 | 按论文复现 | P0 |
| DRL — 多 Agent | **MADDPG** | Lowe et al. 2017 | RLlib / 自研 | P0 |
| DRL — 多 Agent | **QMIX** | Rashid et al. 2018 | PyMARL / 自研 | P0 |
| DRL — 多 Agent | **IPPO** (Independent PPO) | de Witt et al. 2020 | RLlib / 自研 | P1 |
| DRL — 多 Agent | **MAT** (Multi-Agent Transformer) | Wen et al. 2022 | 开源代码适配 | P1 |
| DRL — 计算卸载 | **Lyapunov-DQN** | Bi et al. 2021 | 按论文复现 | P1 |
| 启发式 | **贪心调度 (Greedy)** | — | 自研 | P0 |
| 启发式 | **遗传算法 (GA)** | — | 自研 | P0 |
| 启发式 | **最小负载优先 (Min-Load)** | — | 自研 | P0 |
| 启发式 | **轮询调度 (Round-Robin)** | — | 自研 | P0 |
| DAG 调度 | **HEFT** (Heterogeneous Earliest Finish Time) | Topcuoglu et al. 2002 | 自研 | P1 |
| 优化方法 | **MILP** (OR-Tools) | — | OR-Tools CP-SAT | P1 |

### 4.2 各方法适配说明

#### 4.2.1 DECCo (mA2C) — 核心 baseline（必须复现）

- **来源**：Zhang et al. (2023), *Drones*, 7(8), 513
- **原始设定**：单 Agent mA2C，2 维资源 (CPU, Memory)，单任务顺序调度
- **适配工作**：
  - 资源向量从 2 维扩展到 6 维（保持架构不变，更换输入维度）
  - 动作空间保持原样（一个时隙只调度一个任务到一个 UAV）
  - 成本函数使用 DECCo 原始的 3 项（不包含 deadline 和 priority 项）
  - 不包含 DAG 依赖建模（按独立任务处理）
- **作为 baseline 的意义**：验证所有 CASCADE 改进项的共同效果

#### 4.2.2 MADDPG

- **来源**：Lowe et al. (2017), *NIPS*
- **适配**：每个 UAV 作为一个 Agent，指挥车 Agent 输出全局批评值
- **动作空间**：连续动作（任务接受概率），加噪声探索
- **推荐实现**：RLlib 的 MADDPG 实现，或使用 EPyMARL 框架

#### 4.2.3 QMIX

- **来源**：Rashid et al. (2018), *ICML*
- **适配**：将每个 UAV Agent 的局部 Q 值通过 Mixing Network 组合为全局 Q_tot
- **动作空间**：离散动作（每个 UAV 从 ready tasks 中选择）
- **推荐实现**：PyMARL 的 QMIX 实现

#### 4.2.4 Lyapunov-DQN

- **来源**：Bi et al. (2021), *IEEE TWC*
- **相关性**：该论文关注移动边缘计算网络中的计算卸载，使用 Lyapunov 优化 + DQN
- **适配**：将计算卸载问题类比为 UAV 任务调度，Lyapunov 漂移惩罚对应 deadline 约束

#### 4.2.5 启发式方法

| 方法 | 策略 | 适用场景 |
|------|------|---------|
| **Greedy** | 按优先级排序，每个任务分配给当前负载最低且资源满足的 UAV | 快速 baseline |
| **GA** | 个体编码 = 任务-UAV 分配矩阵，适应度 = -Cost | 中等复杂度优化 |
| **Min-Load** | 新任务始终分配给当前总负载（加权任务数）最低的 UAV | 负载均衡 baseline |
| **Round-Robin** | 按固定顺序将任务轮询分配给 UAV | 性能下界 |
| **HEFT** | DAG 调度经典算法：先 rank 任务再依次分配到最早完成时间的 UAV | DAG 专用 baseline |

---

## 5. 实验设计

### 5.1 实验场景

实验场景由参数化生成器按 7 维参数空间随机采样产生（见 2.3.1 节），每个 episode 生成一个独特的场景实例。为确保评估的系统性，定义三类参数分布作为训练/测试/泛化场景：

#### 场景分布 DS1：标准洪涝（In-Distribution / 训练+验证+测试）

| 参数 | 采样范围 |
|------|---------|
| `disaster_area_km2` | 25~64 (5×5~8×8 km) |
| `flood_ratio` | 0.3~0.7 |
| `building_density` | medium / high（均匀随机） |
| `civilian_density` | 10~40 人/km² |
| `comm_failure_rate` | 0.1~0.4 |
| `breach_risk_level` | 1~2 |
| `terrain_roughness` | flat / hilly（均匀随机） |
| `num_regions` | 2~3 |
| `num_uavs_total` | 8~12 |
| `uav_orin_8gb_ratio` | 0.3~0.5 |

典型实例：2~3 个区域，每个区域 7~14 个子任务（含 A1+A2+A4+I1~I3+F1+C1+C2），UAV 编队含 4~5 架 16GB + 3~4 架 8GB 板卡。

- 训练/验证/测试 split：70% / 15% / 15%（不同随机种子）
- 训练集总 episodes：~7,000（每个 episode 场景参数独立采样）

#### 场景分布 DS2：复杂洪涝（Out-of-Distribution / 泛化测试）

| 参数 | 采样范围 | 与 DS1 差异 |
|------|---------|-----------|
| `disaster_area_km2` | 64~100 | 面积更大 |
| `flood_ratio` | 0.5~0.8 | 淹没更深 |
| `building_density` | high | 固定高密度 |
| `civilian_density` | 20~50 人/km² | 被困人员更多 |
| `comm_failure_rate` | 0.3~0.7 | 通信更恶劣 |
| `breach_risk_level` | 2~3 | 溃口风险更高 |
| `terrain_roughness` | hilly / complex | 地形更复杂 |
| `num_regions` | 3~5 | 区域更多 |
| `num_uavs_total` | 12~15 | UAV 更多 |
| `uav_orin_8gb_ratio` | 0.2~0.4 | 16GB 比例更高（应对大模型融合） |

典型实例：3~5 个区域，每个区域 9~15 个子任务（触发全部 A2/A4/A5/I2~I5/F1~F4/C1+C2），全部 16 种子任务类型均可能被触发。

#### 场景分布 DS3：极端/降级场景（Stress Test）

| 参数 | 采样范围 | 特殊压力 |
|------|---------|---------|
| `disaster_area_km2` | 64~100 | 大面积 |
| `flood_ratio` | 0.6~0.8 | 极端淹没 |
| `comm_failure_rate` | 0.5~0.9 | 通信极度恶劣 |
| `breach_risk_level` | 3 | 固定高危溃口 |
| `terrain_roughness` | complex | 极端地形 |

**额外压力事件**（仿真中途注入）：
- UAV 随机故障：30% 概率在仿真中段随机失效 1~3 架
- 网络大面积中断：仿真中段 5 min 模拟通信损毁恶化
- 突发新溃口：触发新增高光谱应急检测任务（A5+I5+F4）

UAV 编队：12~15 架，8GB 板卡比例 0.3~0.5。每个测试 episode 独立采样。

#### 场景分布汇总

| 分布 | 用途 | 参数变化维度 | Episodes (测试) |
|------|------|------------|:---:|
| **DS1** | 训练/验证/ID 测试 | 全部 7 维在范围内均匀采样 | 1500 (test) |
| **DS2** | OOD 泛化测试 | 全部 7 维向更困难方向偏移 | 500 |
| **DS3** | 极端/降级鲁棒性 | 通信+溃口+地形极端 + 注入故障 | 300 |

### 5.2 实验矩阵

| 实验编号 | 名称 | 场景 | 对比方法 | 关键指标 | 核心验证目标 |
|---------|------|------|---------|---------|-------------|
| **E1** | 主对比实验 | DS1 (测试集) | 全部 15 个方法 | Makespan, ATCT, $\bar{U}_{gpu}$, $\sigma_{gpu}$ | G1：CASCADE 在全部指标上最优 |
| **E2** | 跨场景泛化 | DS2 | CASCADE, DECCo, MADDPG, QMIX, GA, HEFT | 全部 6 项，关注 DS1→DS2 退化率 | G1 (泛化)：CASCADE 退化最低 |
| **E3** | 极端/降级鲁棒性 | DS3 | CASCADE, DECCo, MADDPG, GA | Makespan 增幅, $\sigma_{gpu}$, PTCT | G5：三级降级模式有效性 |
| **E4** | 消融：6 维→2 维资源 | DS1 | CASCADE, CASCADE-4D (去 Storage+Energy), CASCADE-2D (=DECCo) | $\bar{U}_{gpu}$, $\bar{U}_{mem}$, Makespan | G2：6 维资源贡献 |
| **E5** | 消融：去掉 DAG 建模 | DS1 | CASCADE, CASCADE-NoDAG (拍平为独立任务) | Makespan, ATCT | G2：DAG 建模贡献 |
| **E6** | 消融：成本函数 | DS1 | CASCADE (5-term), CASCADE-3term (去 deadline+priority), CASCADE-NoPriority (去 λ5) | PTCT (I2, F1 等高优先级类型) | G3：deadline/priority 贡献 |
| **E7** | 消融：GNN + 多跳编码 | DS1, DS2 | CASCADE (GAT+GCN+多跳), CASCADE-MLP (纯 MLP), CASCADE-NoMultihop (去多跳特征) | $\sigma_{gpu}$, Makespan | G4：距离/多跳/GNN 贡献 |
| **E8** | 消融：模型压缩 | DS2 | CASCADE (full), CASCADE-NoCompress (固定用大模型) | PTCT (F1, F3), $\bar{U}_{mem}$ | G6：自适应压缩贡献 |
| **E9** | 消融：8GB/16GB 混合 | DS1 | CASCADE (混合), CASCADE-All8GB (全部替换为 8GB) | $\bar{U}_{mem}$, PTCT (F1, F3) | 板卡异构对调度的影响 |

### 5.3 RL 训练配置

| 超参数 | 值 | 说明 |
|--------|-----|------|
| 总训练 episodes | 10,000 | 场景分布 DS1，每 episode 独立采样参数 |
| 每 episode 最大步数 | 300 | 适应 5~15 子任务链的最大执行步数 |
| 并行环境数 (n_envs) | 8 | A3C 异步采样 |
| N 步回报长度 | 128 | GAE 计算窗口 |
| 折扣因子 γ | 0.99 | |
| GAE λ | 0.95 | |
| 学习率 (Actor) | 3e-4 | Adam optimizer |
| 学习率 (Critic) | 1e-3 | Adam optimizer |
| 熵正则化系数 β | 0.01 | 初始值 |
| 熵衰减 | 0.9995 per episode | 逐步减少探索 |
| 梯度裁剪 | max_norm=0.5 | |
| Target 网络软更新 τ | 0.005 | |
| 经验回放池大小 | 100,000 | |
| Batch size | 256 | |
| 评估频率 | 每 100 episodes | 在验证集上跑 10 episodes |
| 随机种子 | 5 个 (0-4) | |

### 5.4 每次实验的运行流程

```
1. 加载实验配置文件 (experiments/configs/e1_main_comparison.yaml)
2. for method in methods:
3.     for seed in seeds (5 seeds):
4.         if method is RL-based:
5.             # 训练阶段
6.             for episode in range(N_train_episodes):
7.                 # 每 episode 独立采样场景参数 (DS1分布)
8.                 params = sample_scenario_params(DS1, seed, episode)
9.                 env = CASCADEEnv(params)
10.                obs, info = env.reset()
11.                while not done:
12.                    # 每步聚焦一个子区域：当前区域就绪任务 → 全局 UAV 分配
13.                    mask = env.get_action_mask()
14.                    action = scheduler.decide(mask)
15.                    obs, reward, terminated, truncated, info = env.step(action)
16.                    # step() 内部自动轮转到下一个有就绪任务的子区域
17.                    scheduler.learn(batch)
18.                # Episode 结束: 所有区域所有任务链均完成
19.                # → env.get_episode_metrics() 输出 Makespan/ATCT/PTCT/GPU/Mem
16.                if episode % eval_freq == 0:
17.                    eval_metrics = run_evaluation(scheduler, DS1_val, n_episodes=10)
18.                    mlflow.log_metrics(eval_metrics)
19.            # 测试阶段 (DS1 测试集)
20.            test_metrics = run_evaluation(scheduler, DS1_test, n_episodes=50)
21.            mlflow.log_metrics({"test_" + k: v for k, v in test_metrics.items()})
22.            # 泛化测试 (DS2)
23.            ood_metrics = run_evaluation(scheduler, DS2, n_episodes=30)
24.            mlflow.log_metrics({"ood_" + k: v for k, v in ood_metrics.items()})
25.         else (启发式/优化方法):
26.            test_metrics = run_evaluation(scheduler, DS1_test, n_episodes=50)
27.            mlflow.log_metrics(test_metrics)
28.     # 汇总 5 seed 结果
29.     summary = aggregate_across_seeds(all_results)
30.     mlflow.log_metrics(summary)
```

---

## 6. 评估指标

由于实验全程基于仿真（无真实图像/遥感数据），评估聚焦于**任务完成时效**和**计算资源利用效率**两个维度，共 6 项核心指标（对齐 CASCADE.md 3.8.2）。

### 6.1 核心评估指标

| 指标 | 符号 | 定义 | 目标 |
|------|:----:|------|:----:|
| **Makespan** | $T_{total}$ | 一个区域任务链从第一个子任务开始到最后一个完成的 wall-clock 总时间 | ↓ 最小化 |
| **Average Task Completion Time (ATCT)** | $\bar{T}_{task}$ | $\frac{1}{|V|}\sum (T_{finish}(v_i) - T_{start}(v_i))$ | ↓ 最小化 |
| **Per-Type Average Completion Time (PTCT)** | $\bar{T}_{type}$ | 按任务类型 (A/P/I/F/C) 分组计算平均完成时间 | ↓ 最小化（高优先级类型应显著更低） |
| **GPU Utilization Mean** | $\bar{U}_{gpu}$ | 所有 Orin NX 板卡 GPU 利用率的时域与卡间均值 | ↑ 最大化 |
| **GPU Utilization Std** | $\sigma_{gpu}$ | 所有板卡 GPU 利用率先按板卡时域平均，再跨板卡计算标准差 | ↓ 最小化（负载均衡） |
| **Memory Utilization Mean** | $\bar{U}_{mem}$ | 所有板卡显存利用率（已用/总显存 8GB 或 16GB）的均值 | ↑ 最大化（显存资源充分利用） |

### 6.2 衍生评估指标

| 指标 | 公式 | 用途 |
|------|------|------|
| **Generalization Degradation (GD)** | $\frac{\text{Makespan}_{DS2} - \text{Makespan}_{DS1}}{\text{Makespan}_{DS1}}$ | 泛化退化率，越小越好 |
| **Resilience Score (RS)** | $\frac{\text{Makespan}_{DS3\_with\_faults} - \text{Makespan}_{DS3\_no\_faults}}{\text{Makespan}_{DS3\_no\_faults}}$ | 降级鲁棒性，增幅越小越好 |
| **Priority Responsiveness (PR)** | $\frac{\bar{T}_{I2,F1}}{\bar{T}_{A1,A4}}$ | 优先级响应比：高优先级任务 vs 采集类任务的 PTCT 比值 |

### 6.3 训练效率指标（仅 RL 方法）

| 指标 | 说明 |
|------|------|
| **Convergence Episodes** | 验证 Makespan 达到最低值的 95% 所需 episode 数 |
| **Sample Efficiency** | 收敛时消耗的总 transition 数 |
| **Wall Clock Time** | 训练总耗时（小时） |

---

## 7. 代码仓库结构

```
scheduling-rl/
├── CLAUDE.md
├── README.md
├── setup.py
├── requirements.txt
│
├── doc/                              # 文档
│   ├── DECCo论文分析与洪涝灾害应急调度改进方案_CASCADE.md
│   ├── CASCADE_LLM_Powered_Planner_方案.md
│   ├── 实验计划_CASCADE_仿真环境与算法框架.md
│   └── Zhang 等 - 2023 - DECCo-*.pdf
│
├── configs/                          # Hydra 配置 (YAML)
│   ├── env/
│   │   ├── base.yaml                # 环境基础参数
│   │   ├── scenario_ds1_standard.yaml   # DS1 参数分布
│   │   ├── scenario_ds2_complex.yaml    # DS2 参数分布
│   │   └── scenario_ds3_extreme.yaml    # DS3 参数分布
│   ├── algorithm/
│   │   ├── cascade_ma3c.yaml        # CASCADE 主算法
│   │   ├── deco_ma2c.yaml           # DECCo baseline
│   │   ├── maddpg.yaml
│   │   ├── qmix.yaml
│   │   ├── ppo.yaml
│   │   └── heuristic/
│   │       ├── greedy.yaml
│   │       ├── ga.yaml
│   │       ├── min_load.yaml
│   │       └── round_robin.yaml
│   └── experiment/
│       ├── e1_main_comparison.yaml
│       ├── e2_generalization.yaml
│       ├── e3_extreme_ood.yaml
│       └── e4_e9_ablation.yaml      # 消融实验共用
│
├── src/
│   ├── env/                          # 仿真环境
│   │   ├── __init__.py
│   │   ├── cascade_env.py           # Gymnasium 环境主体 (含 get_episode_metrics)
│   │   ├── scenario_generator.py    # 7 维参数化场景生成器 (DS1/DS2/DS3 采样)
│   │   ├── uav_simulator.py         # 无人机模拟器 (Orin NX 8GB/16GB)
│   │   ├── network_simulator.py     # 距离感知 Mesh 通信 + 多跳中继路由
│   │   ├── task_manager.py          # 任务链生成器 + 参数触发规则 + DAG
│   │   ├── reward.py                # 5 项成本函数 → reward (含多跳路径延迟)
│   │   └── action_mask.py           # 动作掩码 (含显存约束 + 多跳可达性)
│   │
│   ├── algorithms/                   # 调度算法
│   │   ├── __init__.py
│   │   ├── base_scheduler.py        # 抽象基类
│   │   │
│   │   ├── cascade/                 # ★ CASCADE (本文方法)
│   │   │   ├── __init__.py
│   │   │   ├── ma3c_trainer.py      # mA3C 训练主循环
│   │   │   ├── actor_network.py     # UAV Actor 网络
│   │   │   ├── critic_network.py    # 指挥车 Critic 网络
│   │   │   ├── gnn_encoder.py       # GAT + GCN + Cross-Modal Attention
│   │   │   ├── mhsa_fusion.py       # Multi-Head Self-Attention 融合
│   │   │   └── hungarian_match.py   # Hungarian 算法离散化
│   │   │
│   │   ├── deco/                    # DECCo (mA2C) 复现
│   │   │   ├── __init__.py
│   │   │   ├── ma2c_agent.py
│   │   │   ├── co_scheduler.py
│   │   │   └── solution_switching.py
│   │   │
│   │   ├── maddpg/                  # MADDPG
│   │   │   ├── __init__.py
│   │   │   ├── maddpg_trainer.py
│   │   │   └── networks.py
│   │   │
│   │   ├── qmix/                    # QMIX
│   │   │   ├── __init__.py
│   │   │   ├── qmix_trainer.py
│   │   │   ├── agent_network.py
│   │   │   └── mixer_network.py
│   │   │
│   │   ├── ppo/                     # PPO / IPPO
│   │   │   ├── __init__.py
│   │   │   └── ppo_trainer.py
│   │   │
│   │   ├── lyapunov_dqn/            # Lyapunov-DQN
│   │   │   ├── __init__.py
│   │   │   └── lyapunov_dqn_trainer.py
│   │   │
│   │   └── heuristic/               # 启发式方法
│   │       ├── __init__.py
│   │       ├── greedy.py
│   │       ├── genetic_algorithm.py
│   │       ├── min_load.py
│   │       ├── round_robin.py
│   │       └── heft.py              # HEFT DAG 调度
│   │
│   ├── execution/                    # 执行与容错（对应 CASCADE.md 3.7）
│   │   ├── __init__.py
│   │   ├── preemption_engine.py     # 抢占引擎 (3.7.2)
│   │   ├── degradation_manager.py  # 三级降级 (3.7.1)
│   │   ├── model_compressor.py     # 模型自适应压缩 (3.7.3)
│   │   └── checkpoint_manager.py   # Checkpoint/Resume
│   │
│   ├── evaluation/                   # 评估
│   │   ├── __init__.py
│   │   ├── metrics.py               # Makespan, ATCT, PTCT, GPU Util, Mem Util 等全部指标
│   │   ├── evaluator.py             # 评估运行器（多 seed 聚合 + DS1/DS2/DS3 评估）
│   │   ├── logger.py               # MLflow / TensorBoard 日志
│   │   └── visualizer.py           # 可视化（训练曲线 / DAG 甘特图 / 资源利用率热力图）
│   │
│   └── utils/                        # 工具
│       ├── __init__.py
│       ├── types.py                 # dataclass 类型定义
│       ├── config.py                # Hydra 配置加载
│       └── seed.py                  # 随机种子管理
│
├── experiments/                      # 实验运行入口
│   ├── run_e1_comparison.py         # 主对比实验 (DS1)
│   ├── run_e2_generalization.py     # 泛化实验 (DS2)
│   ├── run_e3_extreme.py            # 极端/降级鲁棒性实验 (DS3)
│   ├── run_ablation.py              # 消融实验 (E4-E9)
│   ├── run_baselines_only.py        # 仅跑 baseline（无 CASCADE）
│   └── analyze_results.py           # 汇总分析 + 生成图表
│
├── tests/                            # 单元测试
│   ├── test_env/
│   │   ├── test_uav_simulator.py
│   │   ├── test_network_simulator.py
│   │   ├── test_task_manager.py
│   │   └── test_cascade_env.py
│   ├── test_algorithms/
│   │   ├── test_cascade.py
│   │   └── test_baselines.py
│   └── test_execution/
│       ├── test_preemption.py
│       └── test_degradation.py
│
└── outputs/                          # 实验输出（gitignore）
    ├── checkpoints/                  # 模型 checkpoint
    ├── logs/                         # 训练日志
    └── results/                      # 实验结果汇总
```

---

## 8. 实施路线图

### Phase 1：仿真环境核心（第 1-2 周）

| 任务 | 产出 | 优先级 |
|------|------|--------|
| 实现 `UAVSimulator`（运动/电池/传感器/Orin NX 8GB+16GB 计算模型） | 可运行的无人机模拟器，含单元测试 | P0 |
| 实现 `MeshNetworkSimulator`（三维距离→RSS→带宽→多跳 Dijkstra 路由） | 距离感知网络模拟器，含单元测试 | P0 |
| 实现 `TaskManager`（参数触发规则→任务链生成→DAG 依赖→状态机） | 可运行的任务管理器，含单元测试 | P0 |
| 实现 `ScenarioGenerator`（7 维参数采样 + 区域聚类 + DS1/DS2/DS3 分布定义） | 3 个场景分布可加载运行 | P0 |
| 实现 `action_mask`（含显存约束 + 多跳可达性）+ `reward`（5 项成本含多跳延迟） | 掩码 + 成本函数 | P0 |
| 实现 `CASCADEEnv`（Gymnasium 接口 + SimPy 主循环 + 每 episode 独立采样） | 环境可 step/reset，通过 API 测试 | P0 |
| 实现 `get_episode_metrics()`（Makespan/ATCT/PTCT/GPU/Mem 6 项指标） | 评估指标可计算 | P0 |

### Phase 2：Baseline 算法（第 3-4 周）

| 任务 | 产出 | 优先级 |
|------|------|--------|
| 实现 4 个启发式方法（Greedy, GA, MinLoad, RoundRobin） | 4 个 heuristic scheduler | P0 |
| 实现 DECCo (mA2C) + 适配 6 维资源和 DAG 任务（评估用） | DECCo baseline 可运行 | P0 |
| 实现 CASCADE 网络结构（GAT+GCN+多跳MLP+MHSA+Actor/Critic） | 网络定义完成 | P0 |
| 集成 MADDPG（基于 RLlib 或自研） | MADDPG baseline | P1 |
| 集成 QMIX（基于 PyMARL 或自研） | QMIX baseline | P1 |
| 在 DS1 上验证所有 baseline 可运行并输出合理 Makespan | 完整性检查通过 | P0 |

### Phase 3：CASCADE 训练与调优（第 5-6 周）

| 任务 | 产出 | 优先级 |
|------|------|--------|
| 实现 mA3C 训练主循环（多环境并行采样 + GAE + 更新） | 训练流程可运行 | P0 |
| 实现 Hungarian 匹配后处理 + 动作掩码集成（含 Orin NX 显存感知） | 完整决策链路 | P0 |
| 实现三级降级模式管理器 | 降级逻辑可触发 | P0 |
| 实现抢占式救援引擎 + 模型自适应压缩 | 执行与容错层 | P0 |
| 超参数调优（grid search: lr, β, λ 权重） | 最优超参数组合 | P0 |
| 训练曲线收敛验证（DS1 训练集上 Makespan 收敛） | 训练曲线图 | P0 |

### Phase 4：实验执行与论文撰写（第 7-8 周）

| 任务 | 产出 | 优先级 |
|------|------|--------|
| 运行 E1（主对比），15 方法 × 5 seed，DS1 测试集 | 主对比结果表 + 统计检验 | P0 |
| 运行 E2（泛化），DS2 | 泛化退化率 (GD) | P0 |
| 运行 E4-E9（消融），分析各组件贡献 | 消融柱状图 | P0 |
| 运行 E3（极端/降级鲁棒性），DS3 | 鲁棒性结果 (RS) | P1 |
| 结果分析 + 显著性检验 (t-test / bootstrap CI) | 论文实验章节数据 | P0 |
| 可视化（训练曲线 / 消融柱状图 / DAG 甘特图 / GPU 利用率热力图） | 论文配图 | P0 |
| 撰写实验章节初稿 | Experiment 部分文字 | P0 |

---

## 附录 A：关键设计权衡与决策记录

| 决策点 | 选项 A | 选项 B | 选择 | 理由 |
|--------|--------|--------|------|------|
| 离散事件引擎 | SimPy | 手写事件循环 | **SimPy** | 成熟度、可读性、学术界认可 |
| RL 接口 | Gymnasium | 自定义 | **Gymnasium** | 兼容 SB3/RLlib，降低集成成本 |
| 通信仿真精度 | NS-3 联合仿真 | 自定义简模 | **自定义简模** | NS-3 集成成本极高，Friis 模型已足够 |
| 动作空间 | MultiDiscrete | 连续 + Hungarian | **连续 + Hungarian** | 支持并发多任务，贴合 CTDE 设计 |
| 图神经网络框架 | PyTorch Geometric | DGL | **PyTorch Geometric** | 社区更大、文档更全 |
| 配置管理 | 手动 argparse | Hydra | **Hydra** | 多实验配置管理更高效 |
| DRL baseline 来源 | 自研全部 | 复用开源 (RLlib/PyMARL) | **混合** | 核心 baseline 自研保证控制，外围用开源加速 |

---

## 附录 B：对比方法开源资源索引

以下方法优先评估代码可用性，按优先级排列：

| 方法 | 推荐实现 | 仓库/文档 |
|------|---------|----------|
| PPO | Stable-Baselines3 | https://github.com/DLR-RM/stable-baselines3 |
| MADDPG | RLlib | https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#maddpg |
| QMIX | PyMARL | https://github.com/oxwhirl/pymarl |
| MAT | 官方开源 | https://github.com/PKU-MARL/Multi-Agent-Transformer |
| IPPO | RLlib | https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo |
| HEFT | 自研 | 参考论文 Topcuoglu et al. 2002 |

> 注：所有 RL 方法需要适配 CASCADE 仿真环境的观察/动作空间。适配工作量预估：PPO/DQN ~0.5 天，MADDPG/QMIX ~1-2 天，MAT ~2-3 天。
