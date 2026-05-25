# CASCADE 实验计划：仿真环境与算法框架（RL 版）

---

**版本**：v2.0 — 纯 RL 版本（不含 LLM） | **日期**：2026-05-25

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
| **G1** | CASCADE (mA3C+MHSA+GNN) 在救援效能 (RPDR) 和时效性 (TDSR) 上显著优于 DECCo (mA2C) 及其他 DRL/启发式方法 | 3.6 |
| **G2** | 6 维资源向量 + DAG 流水线建模对调度质量的贡献是统计显著的 | 3.5.2, 3.5.1 |
| **G3** | 多目标成本函数（5 项）中，deadline 违反惩罚 ($\lambda_4$) 和优先级加权 ($\lambda_5$) 对救援效能提升贡献最大 | 3.5.3 |
| **G4** | 多模态数据流感知 + GNN 网络拓扑编码在网络条件动态变化时，性能退化显著小于无拓扑感知的方法 | 3.5.4 |
| **G5** | 三级降级运行模式在 UAV 失效场景中，任务完成率降幅显著小于无降级机制的方法 | 3.7.1 |
| **G6** | 多模态模型自适应压缩在带宽受限场景中，对 TDSR 的提升有显著贡献 | 3.7.3 |

### 1.2 实验范围限定

- **全部在仿真环境中进行**，不涉及真实无人机硬件
- **不涉及 LLM**，调度决策完全由 RL 策略网络 / 启发式算法 / 数学优化方法给出
- 仿真环境需模拟：多无人机飞行动力学（简化）、Mesh 通信网络、传感器数据生成、任务 DAG 到达与执行
- 算法框架在 Python 3.10+ 环境下运行
- **实验运行环境**：远程 GPU 服务器（AutoDL 等云平台），非本地运行。SSH 连接信息见项目根目录 [CLAUDE.md](../CLAUDE.md)

---

## 2. 仿真环境设计

### 2.1 设计原则

- **离散事件驱动**：使用 SimPy 离散事件仿真引擎，适合建模任务到达、调度决策、任务执行、通信变化等离散事件
- **模块化架构**：场景生成器、无人机模拟器、通信网络模拟器、任务管理器四个核心模块独立可替换
- **Gymnasium 兼容接口**：对外暴露 `reset()` / `step(action)` 标准 RL 接口，兼容 Stable-Baselines3 / RLlib / Tianshou 等主流 RL 框架
- **配置化**：全部场景参数通过 YAML/JSON 配置文件注入，支持快速切换实验条件

### 2.2 技术选型

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

#### 2.3.1 场景生成器 (`ScenarioGenerator`)

负责生成洪涝灾害仿真场景的全部初始状态。

**功能**：
- 生成洪涝受灾地形（2D grid + 简易高程模型）
- 生成被困人员分布（随机散落 + 热点聚类，模拟屋顶/高地聚集模式）
- 生成溃口/管涌/淹没区/道路阻断等灾情要素及其空间坐标
- 生成无人机编队初始配置（位置、朝向、电量、传感器）
- 生成初始任务序列（含 DAG 依赖关系）

**关键可配置参数**：

| 参数 | 范围 | 说明 |
|------|------|------|
| `grid_size_x`, `grid_size_y` | 1000-10000 m | 受灾区域范围 |
| `num_civilians` | 10-500 | 被困人员数量 |
| `num_breach_points` | 1-10 | 溃口/管涌数量 |
| `water_level_mean` | 0.5-5.0 m | 平均水深 |
| `comm_failure_rate` | 0.0-0.5 | 通信链路中断概率 |
| `task_count` | 10-100 | 总任务数量 |
| `dag_depth_max` | 2-8 | DAG 最大深度 |
| `dag_width_max` | 2-10 | DAG 最大并行宽度 |
| `num_uavs_total` | 4-20 | 无人机总数 |
| `uav_type_distribution` | dict | 各类型 UAV 比例 |

#### 2.3.2 无人机模拟器 (`UAVSimulator`)

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
position: (x, y, z)                # 三维坐标
battery_level: float               # 0.0~1.0
cpu_available: float               # 0.0~1.0 归一化剩余 CPU
gpu_available: float               # 0.0~1.0 归一化剩余 GPU
memory_available_gb: float         # 剩余内存 GB
storage_available_gb: float        # 剩余存储 GB
bandwidth_available_mbps: float    # 上行带宽
current_tasks: List[Task]          # 当前执行中的任务（最多 2 个并发）
status: UAVStatus                  # IDLE | TRANSIT | COLLECTING | PROCESSING | RELAYING | FAULTED
```

**电池功耗模型**（简化的分状态恒功率模型）：

| 状态 | 功耗 | 说明 |
|------|------|------|
| HOVER | 200 W | 悬停等待，保持位置 |
| TRANSIT | 350 W | 15 m/s 巡航飞行 |
| COLLECTING | 15 W | 传感器采集（快门 + 读出电路 + 初步压缩） |
| PROCESSING (GPU) | 40 W | 边缘 GPU 推理（Jetson Orin AGX 典型值） |
| RELAYING | 90 W | 通信中继（射频功放占主要功耗） |

> 注：功耗值参考了 DJI M300 级别工业无人机的实际飞行功耗（~1000W 包含动力系统 60-70%），加上 Jetson Orin AGX 的计算功耗（15-60W）。此处数字为动力之外的设备功耗。

#### 2.3.3 通信网络模拟器 (`MeshNetworkSimulator`)

**功能**：
- 模拟 UAV-UAV 和 UAV-指挥车之间的 LoRa（长距离、低带宽）和 WiFi 6 Mesh（短距离、高带宽）双模通信
- 动态拓扑变化：距离、地形遮挡、干扰导致链路质量实时变化
- 带宽分配：多任务共享同一链路时按优先级分配带宽
- 延迟模型：传播延迟 + 传输延迟 + 排队延迟

**每条链路的属性**：

```
link_id: (src_id, dst_id)
bandwidth_mbps: float           # 当前可用带宽
latency_ms: float               # 当前总延迟
packet_loss_rate: float         # 丢包率 (0.0~1.0)
connected: bool                 # 是否连通
rssi_dbm: float                 # 接收信号强度
```

**通信模型**（简化的 Friis 自由空间 + 遮挡衰减）：

- **视距 (LoS)**：Free Space Path Loss，2.4 GHz 载频
- **非视距 (NLoS)**：山体/建筑遮挡，额外衰减 20-30 dB
- **最大通信距离**：LoRa 10 km, WiFi 6 500 m（超出即断开）
- **多跳中继延迟**：每跳 +5 ms 固定处理开销
- **带宽退化**：RSSI 低于阈值时，带宽按比例衰减

#### 2.3.4 任务管理器 (`TaskManager`)

**功能**：
- 管理任务全生命周期（生成 → 排队 → 调度 → 执行 → 完成 / 超时）
- 维护任务 DAG 依赖图
- 追踪每个任务的实时进度
- 注入随机紧急事件（模拟灾情变化）

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
modality: ModalityType             # RGB | THERMAL | MULTISPECTRAL | HYPERSPECTRAL | MIXED
resource_requirement: ResourceVec  # 6 维资源需求向量
priority: int                      # 1-10 优先级
deadline_min: float                # 相对 deadline（分钟）
depends_on: List[str]              # 前驱 task_id 列表
depended_by: List[str]             # 后继 task_id 列表
status: TaskStatus
assigned_uav: Optional[str]
start_time: Optional[float]
completion_time: Optional[float]
```

**关键方法**：

- `generate_from_timeline(yaml_path)` — 从 YAML 时间线文件生成任务
- `get_ready_tasks() -> List[Task]` — 返回依赖已满足、可调度的任务
- `update_dag(completed_task_id)` — 前驱完成后解锁后继
- `inject_emergency(task: Task)` — 注入突发紧急任务
- `get_dag_stats() -> Dict` — 返回 DAG 统计信息（总节点、关键路径长度、并行度）

#### 2.3.5 仿真环境主环境 (`CASCADEEnv`)

```python
# 核心伪代码
class CASCADEEnv(gym.Env):
    """
    Gymnasium 兼容的 CASCADE 仿真环境。

    观察空间:
      - task_features: 待调度任务的特征矩阵 (N_tasks × 6)
      - uav_features: 无人机状态矩阵 (N_uavs × 6)
      - network_adj: 网络邻接矩阵 (N_nodes × N_nodes)
      - task_dag_adj: 任务 DAG 邻接矩阵 (N_tasks × N_tasks)
      - task_dag_mask: 当前可调度任务的二元掩码
      - priority_features: 任务优先级与 deadline 信息

    动作空间:
      - 组合动作: (task_idx, uav_idx) 的 MultiDiscrete 空间
      - 或连续动作: 每个 task 到每个 UAV 的分配概率矩阵
      - 动作掩码自动过滤不可行的 (task, uav) 组合
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

        动作编码: [task_idx, uav_idx] 或分配矩阵
        """
        # 1. 将 action 解码为 (task, uav) 分配
        # 2. 验证分配有效性（资源、网络、依赖）
        # 3. 执行分配，更新 UAV 状态
        # 4. 推进仿真 dt（适应性地：下一个事件或固定 5s）
        # 5. 更新所有 UAV 运动、功耗、计算进度
        # 6. 更新通信网络拓扑
        # 7. 检查任务完成 / 超时 / 解锁后继
        # 8. 概率性地注入随机事件
        # 9. 计算即时 reward（基于 5 项成本函数）
        # 10. 构建下一个观察
        ...

    def compute_reward(self) -> float:
        """
        基于 CASCADE.md 第 3.5.3 节的 5 项成本函数计算即时 reward。

        C = λ1 * C_load + λ2 * C_overload + λ3 * C_latency
          + λ4 * C_deadline + λ5 * C_priority

        reward = -C （最小化成本 = 最大化 reward）

        推荐权重 (CASCADE.md L298):
          λ1=0.15, λ2=0.25, λ3=0.10, λ4=0.30, λ5=0.20
        """
        ...

    def get_action_mask(self) -> np.ndarray:
        """
        返回动作掩码，过滤不可行的 (task, uav) 组合：
        - UAV 处于 FAULTED 状态
        - UAV 资源不满足 task 需求
        - UAV 网络不可达（对于需要数据传输的任务）
        - task 有特定的传感器/UAV 类型要求
        """
        ...
```

**关键设计决策：MultiDiscrete vs 连续动作空间**

| 方案 | 动作表示 | 优点 | 缺点 |
|------|---------|------|------|
| A: MultiDiscrete | `[task_idx, uav_idx]` | 简单，兼容性强 | 一次只能调一个任务 |
| **B: 连续 + 掩码（推荐）** | 分配矩阵 `P ∈ [0,1]^{N_tasks × N_uavs}`，经 Hungarian 离散化 | 支持并发多任务调度 | 实现稍复杂 |

> **推荐方案 B**，更贴合 CASCADE.md 3.6.3 节中"各 UAV Actor 输出任务接受概率向量→汇总后 Hungarian 匹配"的 CTDE 设计。

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
| 状态编码 | 向量拼接 | GAT 编码任务 DAG + GCN 编码网络拓扑 + MHSA 融合 |
| 动作空间 | 选 1 架 UAV 执行当前任务 | 并发多 UAV × 多任务，经 Hungarian 离散化 |
| 协作机制 | 无显式协作 | 指挥车广播压缩隐状态，各 UAV Actor 通信注意力 |
| 任务依赖 | 忽略 | DAG GAT 编码，关键路径感知 |
| 优先级/Deadline | 无 | 抢占式优先级队列 + 5 项成本函数含 deadline 惩罚 |
| 资源模型 | 2 维 (CPU, Mem) | 6 维 (+GPU, Storage, BW, Energy) |

#### 3.2.2 网络结构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CASCADE 状态编码器                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  输入:                                                               │
│  ├─ 任务 DAG 图 G_task  ──→ GAT (2层, 4头) ───→ h_task ∈ R^128     │
│  ├─ 网络拓扑图 G_net    ──→ GCN (2层)      ───→ h_net  ∈ R^128     │
│  ├─ UAV 资源矩阵 R_all  ──→ MLP (2层, 256)  ───→ h_res  ∈ R^128    │
│  └─ 多模态数据流特征    ──→ Cross-Modal Atten ─→ h_modal ∈ R^128    │
│                                                                      │
│  融合: MHSA(Concat([h_task, h_net, h_res, h_modal]))                │
│       → h_global ∈ R^256  (供 Critic 使用)                           │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                    CASCADE Actor-Critic                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Critic (指挥车):                                                    │
│    h_global → MLP(256→128→1) → V(s)  (状态价值)                     │
│                                                                      │
│  Actor_i (UAV i):                                                    │
│    [h_global, local_obs_i] → MLP(256+64→128→|V_ready|)              │
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
  - task t 有传感器类型要求但 UAV u 不搭载该传感器
  - task t 需要网络连接但 UAV u 当前不可达
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

#### 场景 S1：洞庭湖流域标准洪涝（In-Distribution / 训练集）

| 参数 | 值 |
|------|-----|
| 受灾范围 | 5 km × 5 km |
| 被困人员 | 80 人（随机散落 50 + 3 个热点聚类各 10 人） |
| 溃口/管涌 | 3 处 |
| UAV 编队 | 10 架 (V×3, M×2, H×1, C×2, R×2) |
| 总任务数 | 30 个（覆盖 A/P/I/F/C 五大类），DAG 深度 2-5 层 |
| 紧急任务注入 | 仿真中途插入 3 次 I2 热红外人员检测任务 (priority=10, deadline=2min) |
| 通信中断概率 | 0.10 |
| UAV 故障概率 | 0.03 per episode |
| 仿真时长 | 2 小时（模拟时间） |
| 训练/验证/测试 split | 70% / 15% / 15% （不同随机种子） |

#### 场景 S2：珠江流域洪涝（Out-of-Distribution / 泛化测试）

| 参数 | 值 |
|------|-----|
| 受灾范围 | 8 km × 6 km |
| 被困人员 | 150 人（分布模式与 S1 不同：沿河岸线状分布 + 城市孤岛） |
| 溃口/管涌 | 5 处 |
| UAV 编队 | 14 架 (V×4, M×3, H×2, C×3, R×2) — 配比变化 |
| 总任务数 | 50 个，DAG 深度 3-7 层 |
| 通信中断概率 | 0.25（更恶劣） |
| UAV 故障概率 | 0.08 per episode |
| 仿真时长 | 3 小时 |

#### 场景 S3：复合灾害（Extreme OOD）

| 参数 | 值 |
|------|-----|
| 类型 | 洪涝 + 山体滑坡 + 化工厂化学品泄漏 |
| 受灾范围 | 10 km × 10 km |
| 被困人员 | 200 人 |
| UAV 编队 | 18 架 (V×5, M×3, H×2, C×4, R×4) |
| 总任务数 | 80 个（含新增化学检测、气体分析子任务） |
| 通信中断概率 | 0.35 |
| UAV 故障概率 | 0.12 per episode |
| 特殊事件 | 仿真中途：3 架 UAV 故障 + 新溃口出现 + 网络大面积中断 5 min |

### 5.2 实验矩阵

| 实验编号 | 名称 | 场景 | 对比方法 | 核心验证目标 |
|---------|------|------|---------|-------------|
| **E1** | 主对比实验 | S1 (测试集) | 全部 15 个方法 | G1：CASCADE 最优 |
| **E2** | 跨场景泛化 | S2 | CASCADE, DECCo, MADDPG, QMIX, GA, HEFT | G1 (泛化)：CASCADE 退化率最低 |
| **E3** | 极端泛化 | S3 | CASCADE, DECCo, MADDPG, GA | G5：降级模式有效性 |
| **E4** | 消融：去掉 6 维资源 | S1 | CASCADE, CASCADE-4D (去掉 Storage+Energy), CASCADE-2D (=DECCo 资源模型) | G2：6 维资源贡献 |
| **E5** | 消融：去掉 DAG 建模 | S1 | CASCADE, CASCADE-NoDAG (独立任务) | G2：DAG 建模贡献 |
| **E6** | 消融：成本函数 | S1 | CASCADE (full 5-term), CASCADE-3term (无 deadline+priority), CASCADE-NoPriority (无 λ5) | G3：deadline 和 priority 项贡献 |
| **E7** | 消融：GNN 编码 | S1, S2 | CASCADE (GAT+GCN), CASCADE-MLP (纯 MLP 替代 GNN) | G4：网络拓扑感知贡献 |
| **E8** | 消融：模型压缩 | S2 (带宽受限) | CASCADE (full), CASCADE-NoCompress (固定用大模型) | G6：自适应压缩贡献 |
| **E9** | 降级鲁棒性 | S3 (注入 UAV 故障) | CASCADE, DECCo, MADDPG | G5：三级降级模式 vs 无降级 |

### 5.3 RL 训练配置

| 超参数 | 值 | 说明 |
|--------|-----|------|
| 总训练 episodes | 10,000 | 场景 S1 |
| 每 episode 最大步数 | 200 | 2h / ~5s per step |
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
6.             env = CASCADEEnv(training_config, seed)
7.             scheduler = Method(hyperparams)
8.             for episode in range(N_train_episodes):
9.                 obs, info = env.reset()
10.                while not done:
11.                    mask = env.get_action_mask()
12.                    action = scheduler.decide(mask)
13.                    obs, reward, terminated, truncated, info = env.step(action)
14.                    scheduler.learn(batch)  # on-policy 或 replay buffer
15.                if episode % eval_freq == 0:
16.                    eval_metrics = evaluate(env_val, scheduler)
17.                    mlflow.log_metrics(eval_metrics)
18.            # 测试阶段
19.            test_metrics = evaluate(env_test, scheduler)
20.            mlflow.log_metrics({"test_" + k: v for k, v in test_metrics.items()})
21.         else (启发式/优化方法):
22.            test_metrics = evaluate(env_test, scheduler)  # 无需训练
23.            mlflow.log_metrics(test_metrics)
24.     # 汇总 5 seed 结果
25.     summary = aggregate_across_seeds(all_results)
26.     mlflow.log_metrics(summary)
```

---

## 6. 评估指标

### 6.1 救援效能指标（对应 CASCADE.md 3.8.2）

| 指标 | 公式 | 目标 |
|------|------|------|
| **RPDR** (Rescued Person Detection Rate) | $\frac{\text{成功检测的被困人员数}}{\text{场景总被困人员数}}$ | ↑ 最大化 |
| **TDSR** (Task Deadline Satisfaction Ratio) | $\frac{\text{Deadline 前完成的任务数}}{\text{总含 Deadline 任务数}}$ | ↑ 最大化 |
| **ART** (Average Response Time) | $\frac{1}{|V_{rescue}|} \sum T_{response}(v_i)$，仅统计救援类任务 (I2, F1) | ↓ 最小化 |

### 6.2 系统效率指标

| 指标 | 公式 | 目标 |
|------|------|------|
| **MFQ** (Multi-modal Fusion Quality) | 融合类任务的结果准确率/mAP | ↑ 最大化 |
| **Energy Efficiency** | $\frac{\text{有效任务完成数}}{\text{总能量消耗 (kWh)}}$ | ↑ 最大化 |
| **Load Balance Index** | $1 - \frac{\sigma(\text{UAV CPU loads})}{\max(\text{UAV CPU loads})}$ | ↑ 最大化 |
| **Communication Efficiency** | $\frac{\text{有效数据传输量 (GB)}}{\text{总带宽占用时间}}$ | ↑ 最大化 |

### 6.3 鲁棒性指标

| 指标 | 公式 | 目标 |
|------|------|------|
| **SRS** (System Resilience Score) | $1 - \frac{\text{RPDR}_{\text{with faults}}}{\text{RPDR}_{\text{no faults}}}$ | ↓ 最小化 |
| **GDR** (Generalization Degradation Rate) | $\frac{\text{RPDR}_{\text{S1}} - \text{RPDR}_{\text{OOD}}}{\text{RPDR}_{\text{S1}}}$ | ↓ 最小化 |
| **PRR** (Preemption Response Rate) | $\frac{\text{抢占式救援任务在 deadline 内完成数}}{\text{总抢占式救援任务数}}$ | ↑ 最大化 |

### 6.4 训练效率指标（仅 RL 方法）

| 指标 | 说明 |
|------|------|
| **Convergence Episodes** | 验证 RPDR 达到峰值的 95% 所需 episode 数 |
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
│   │   ├── scenario_s1_dongting.yaml
│   │   ├── scenario_s2_zhujiang.yaml
│   │   └── scenario_s3_compound.yaml
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
│   │   ├── cascade_env.py           # Gymnasium 环境主体
│   │   ├── scenario_generator.py    # 场景生成器
│   │   ├── uav_simulator.py         # 无人机模拟器
│   │   ├── network_simulator.py     # Mesh 通信网络模拟器
│   │   ├── task_manager.py          # 任务管理器 + DAG
│   │   ├── reward.py                # 5 项成本函数 → reward
│   │   └── action_mask.py           # 动作掩码计算
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
│   │   ├── metrics.py               # RPDR, TDSR, MFQ, SRS 等全部指标
│   │   ├── evaluator.py             # 评估运行器（多 seed 聚合）
│   │   ├── logger.py               # MLflow / TensorBoard 日志
│   │   └── visualizer.py           # 可视化（Matplotlib / Plotly）
│   │
│   └── utils/                        # 工具
│       ├── __init__.py
│       ├── types.py                 # dataclass 类型定义
│       ├── config.py                # Hydra 配置加载
│       └── seed.py                  # 随机种子管理
│
├── experiments/                      # 实验运行入口
│   ├── run_e1_comparison.py         # 主对比实验
│   ├── run_e2_generalization.py     # 泛化实验
│   ├── run_e3_extreme.py            # 极端场景实验
│   ├── run_ablation.py              # 消融实验 (E4-E8)
│   ├── run_resilience.py            # 降级鲁棒性实验 (E9)
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
| 实现 `UAVSimulator`（运动/电池/传感器/计算 4 子模型） | 可运行的无人机模拟器，含单元测试 | P0 |
| 实现 `MeshNetworkSimulator`（拓扑/带宽/延迟/丢包/多跳） | 可运行的网络模拟器，含单元测试 | P0 |
| 实现 `TaskManager`（DAG 生成/状态机/进度追踪） | 可运行的任务管理器，含单元测试 | P0 |
| 实现 `ScenarioGenerator` + 场景 S1 配置文件 | 场景 S1 可加载运行 | P0 |
| 实现 `action_mask` + `reward` 模块 | 掩码 + 5 项成本函数 | P0 |
| 实现 `CASCADEEnv`（Gymnasium 接口 + SimPy 主循环） | 环境可 step/reset，通过 API 测试 | P0 |
| 场景 S2, S3 配置文件 | 3 个场景全部可用 | P1 |

### Phase 2：Baseline 算法（第 3-4 周）

| 任务 | 产出 | 优先级 |
|------|------|--------|
| 实现 4 个启发式方法（Greedy, GA, MinLoad, RoundRobin） | 4 个 heuristic scheduler | P0 |
| 实现 DECCo (mA2C) + 适配 6 维资源和 DAG 任务（评估用） | DECCo baseline 可运行 | P0 |
| 实现 CASCADE 网络结构（GAT+GCN+MHSA+Actor/Critic） | 网络定义完成 | P0 |
| 集成 MADDPG（基于 RLlib 或 自研） | MADDPG baseline | P1 |
| 集成 QMIX（基于 PyMARL 或 自研） | QMIX baseline | P1 |
| 在场景 S1 上验证所有 baseline 可运行并输出合理结果 | 完整性检查通过 | P0 |

### Phase 3：CASCADE 训练与调优（第 5-6 周）

| 任务 | 产出 | 优先级 |
|------|------|--------|
| 实现 mA3C 训练主循环（多环境并行采样 + GAE + 更新） | 训练流程可运行 | P0 |
| 实现 Hungarian 匹配后处理 + 动作掩码集成 | 完整决策链路 | P0 |
| 实现三级降级模式管理器 | 降级逻辑可触发 | P0 |
| 实现抢占式救援引擎 + 模型自适应压缩 | 执行与容错层 | P0 |
| 超参数调优（grid search: lr, β, λ 权重） | 最优超参数组合 | P0 |
| 训练曲线收敛验证（S1 训练集上 RPDR 收敛） | 训练曲线图 | P0 |

### Phase 4：实验执行与论文撰写（第 7-8 周）

| 任务 | 产出 | 优先级 |
|------|------|--------|
| 运行 E1（主对比），15 方法 × 5 seed，S1 测试集 | 主对比结果表 + 统计检验 | P0 |
| 运行 E2（泛化），S2 场景 | 泛化退化率 | P0 |
| 运行 E4-E8（消融），分析各组件贡献 | 消融柱状图 | P0 |
| 运行 E3（极端）+ E9（降级鲁棒性） | 鲁棒性结果 | P1 |
| 结果分析 + 显著性检验 (t-test / bootstrap CI) | 论文实验章节数据 | P0 |
| 可视化（训练曲线 / 雷达图 / 消融柱状图 / DAG 甘特图） | 论文配图 | P0 |
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
