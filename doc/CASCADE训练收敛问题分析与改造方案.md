# CASCADE 训练收敛问题分析与改造方案

## 目标

当前现象是 CASCADE 训练曲线难以收敛，表现接近随机。本文目标是定位导致“不收敛/看起来随机”的具体原因，并给出按优先级推进的改造方案，使训练目标、环境反馈、动作采样和场景难度逐步变成可学习问题。

结论先行：

1. `experiments/run_cascade.py` 不是训练脚本，且当前不会加载已训练 checkpoint；如果用它跑 1000 个 episode 观察曲线，本质是在评估随机初始化策略，不可能收敛。
2. 当前训练脚本 `experiments/train_cascade.py` 虽然会更新模型，但策略动作是确定性 Hungarian 解码，不是真正按策略分布采样，policy gradient 的探索与梯度估计不成立或很弱。
3. 当前 reward 主要是过程型负成本，没有直接奖励完成任务，也没有直接惩罚 makespan/ATCT/timeout，训练目标和论文指标存在明显错位。
4. DS1 场景一开始就包含多区域、DAG 依赖、异构 UAV、通信扰动、紧 deadline 和 UAV 故障，作为从零 RL 训练环境偏难，学习信号噪声很大。
5. 当前评估口径会在每个 episode 新建 scheduler；如果没有 checkpoint/factory 复用，CASCADE 曲线会持续随机。

## 当前训练链路

真正训练入口是：

```bash
python experiments/train_cascade.py
```

训练流程：

```text
env.reset(seed)
  -> scheduler.decide_with_trace(obs, action_mask)
  -> env.step(action)
  -> 收集 trace(log_prob, entropy, value), reward, done
  -> scheduler.learn_episode(traces, rewards, dones)
```

当前损失函数位于 `src/algorithms/cascade/ma3c_trainer.py`：

```text
loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy

actor_loss  = -(log_probs * advantages).mean()
critic_loss = MSE(values, returns)
advantages, returns 由 GAE(reward, value, done) 计算
```

也就是说，模型并不是直接最小化 makespan 或 ATCT，而是最大化累计 reward。时间、利用率等目标只有写进 reward 后，才会进入 GAE，再进入 loss。

## 关键问题 1：`run_cascade.py` 不是训练，且默认是随机策略评估

### 现象

如果执行：

```bash
python experiments/run_cascade.py --episodes 1000
```

得到的 1000 个 episode 不是训练过程，而是评估过程。

当前 `run_cascade.py` 中：

```python
CASCADE_METHODS = {
    "cascade_ma3c": CASCADEMA3CScheduler,
}
```

`run_scheduler_suite()` 调用 `evaluate_scheduler_details()`，而评估器内部每个 episode 都会：

```python
scheduler = scheduler_factory(env.max_ready_tasks, len(env.uavs))
```

这意味着每个 episode 都创建一个新的 `CASCADEMA3CScheduler`。如果没有加载 checkpoint，就是新的随机初始化网络。

### 影响

SwanLab 上的 `episode/cascade_ma3c/*` 曲线只是随机初始化策略在不同随机场景上的评估结果，不会因为 episode 数增加而变好。

### 解决方案

短期必须区分两个入口：

训练：

```bash
python experiments/train_cascade.py \
  --config configs/env/scenario_ds1_standard.yaml \
  --eval-config configs/env/scenario_ds1_standard.yaml \
  --train-episodes 1000 \
  --eval-episodes 10 \
  --eval-every 50 \
  --model-num-uavs 15 \
  --seed 0 \
  --output-dir outputs/training/cascade_ma3c_ds1_1000ep
```

训练后评估：

```bash
python experiments/run_simple_comparison.py \
  --checkpoint outputs/training/cascade_ma3c_ds1_1000ep/cascade_ma3c.pt \
  --model-num-uavs 15 \
  --scenarios ds1 \
  --methods cascade_ma3c greedy min_load round_robin heft \
  --episodes 50 \
  --seed 0 \
  --output-dir outputs/results/ds1_checkpoint_eval
```

建议新增或改造 `run_cascade.py`：

- 增加 `--checkpoint` 参数。
- 使用 `cascade_factory()` 而不是直接传 `CASCADEMA3CScheduler`。
- 如果用户传了 checkpoint，则评估固定已训练策略。
- 如果未传 checkpoint，在输出中明确标注 `random_init=true`，避免误判为训练曲线。

## 关键问题 2：训练动作不是采样，policy gradient 学习信号不可靠

### 当前实现

`decide_with_trace()` 中：

1. actor 输出 logits。
2. 对 mask 后 logits 做 softmax，得到任务-UAV 概率矩阵。
3. 把概率矩阵交给 Hungarian 算法，确定性选出一组 assignment。
4. 对 Hungarian 选中的格子求 log_prob 之和。
5. 用这个 log_prob 做 actor loss。

伪代码：

```text
probs = masked_softmax(logits)
assignments = Hungarian(probs)
log_prob = sum(log(probs[selected_task, selected_uav]))
actor_loss = -(log_prob * advantage)
```

### 问题

Policy gradient 要求动作来自当前策略分布采样，才能用 `log_prob(action)` 得到无偏或近似合理的梯度。但当前动作是 Hungarian 对概率矩阵做确定性最大匹配，探索主要来自随机初始化和场景随机性，不来自策略本身。

这会导致：

- 初期策略几乎随机，但每一步选的是当前随机网络下的确定性最大匹配。
- 熵项只是对概率矩阵算熵，并没有真正驱动采样动作探索。
- 一旦某些动作概率略高，Hungarian 会反复选它，策略容易早早固化。
- `log_prob` 与“真实动作生成过程”不一致，梯度信号偏差大。

### 解决方案 A：改成逐 UAV Categorical 采样

训练时不要用 Hungarian 直接确定性选动作，而是按策略分布采样：

```text
for each uav:
  从当前可选任务中按 Categorical(probs[:, uav]) 采样一个任务
  避免重复任务，可采样后移除已选任务
```

训练模式：

- `decide_with_trace()` 使用 stochastic sampling。
- 记录每个采样动作的 `log_prob` 和 `entropy`。

评估模式：

- `decide()` 可以继续使用 Hungarian 或 greedy，保证评估稳定。

这是最小改动，也是第一优先级。

### 解决方案 B：使用带温度的 Gumbel-TopK / Gumbel-Sinkhorn

如果希望保留“匹配”结构，可以在训练时加 Gumbel 噪声：

```text
sampled_scores = logits + gumbel_noise * temperature
assignments = Hungarian(sampled_scores)
```

并逐步退火 temperature：

```text
temperature: 1.0 -> 0.1
```

这比纯确定性 Hungarian 更有探索，但实现复杂度高于 Categorical 采样。

## 关键问题 3：reward 和目标指标错位

### 当前 reward

当前每一步 reward：

```text
reward = - total_cost

total_cost =
  lambda_load     * load_cost
+ lambda_overload * invalid_action_count
+ lambda_latency  * latency_cost
+ lambda_deadline * deadline_cost
+ lambda_priority * priority_cost
```

默认权重：

```text
lambda_load     = 0.15
lambda_overload = 0.25
lambda_latency  = 0.10
lambda_deadline = 0.30
lambda_priority = 0.20
```

### 当前 reward 的问题

1. 没有完成任务的正奖励。
2. 没有超时任务的强惩罚。
3. 没有 makespan/ATCT 的直接惩罚。
4. `latency_cost` 主要由网络状态决定，不直接绑定本步选了哪个 UAV 执行哪个任务。
5. `priority_cost` 惩罚的是剩余 ready 任务的平均优先级，动作与奖励之间的因果关系不够直接。
6. `load_cost` 惩罚负载不均衡，但不奖励合理提高 GPU 利用率。

结果是：模型不一定知道“完成更多任务、更快完成任务”才是核心目标，只是在优化一些间接过程成本。

### 建议 reward 改造

把 reward 拆成“即时调度反馈 + 任务事件反馈 + episode 终局反馈”。

推荐公式：

```text
r_t =
  + w_complete        * completed_this_step
  - w_timeout         * timed_out_this_step
  - w_invalid         * invalid_action_count
  - w_time            * step_seconds / simulation_duration_s
  - w_deadline        * deadline_pressure
  - w_load_balance    * gpu_util_std
  + w_gpu_use         * useful_gpu_util
  - w_idle            * idle_ready_penalty
  + w_priority_done   * completed_priority_score
```

episode 结束时增加终局奖励：

```text
r_terminal =
  + W_completion * completion_ratio
  + W_tdsr       * tdsr
  + W_rpdr       * rpdr_proxy
  - W_makespan   * normalized_makespan
  - W_atct       * normalized_atct
  - W_timeout    * timeout_ratio
```

建议第一版权重：

```yaml
reward:
  complete_bonus: 1.0
  timeout_penalty: 2.0
  invalid_penalty: 0.5
  time_penalty: 0.02
  deadline_penalty: 0.5
  gpu_balance_penalty: 0.2
  useful_gpu_bonus: 0.1
  priority_complete_bonus: 0.1
  terminal_completion_bonus: 5.0
  terminal_tdsr_bonus: 3.0
  terminal_makespan_penalty: 1.0
  terminal_timeout_penalty: 5.0
```

注意 reward 需要归一化，避免 critic 学不稳。建议把所有时间项除以 `simulation_duration_s` 或一个固定基准，例如 7200 秒。

## 关键问题 4：环境难度过高，应该做课程学习

### 当前 DS1 难点

当前 `scenario_ds1_standard.yaml` 已经包含：

- 2 到 3 个区域。
- 8 到 12 架异构 UAV。
- 每个区域由多类型任务链组成，最多约 15 个任务。
- DAG 依赖导致长时序信用分配。
- 通信故障率 0.1 到 0.4。
- UAV 故障概率 0.03。
- 区域选择 `random_ready`。
- 大区域导致 UAV 飞行时间显著影响任务能否按 deadline 完成。

这些设置对从零 RL 来说太难。模型还没学会基本“任务-UAV匹配”，就同时面对 DAG、多区域、通信、故障和 deadline。

### 课程学习设计

建议新增 4 个训练场景配置。

#### Stage 0：单步匹配 sanity check

目标：确认模型能学会最基础的动作偏好。

配置建议：

```yaml
scenario:
  name: ds0_single_region_easy
  distribution: DS1
  disaster_area_km2: [4, 9]
  flood_ratio: [0.2, 0.3]
  building_density: [low]
  civilian_density: [3, 8]
  comm_failure_rate: [0.0, 0.0]
  breach_risk_level: [0, 1]
  terrain_roughness: [flat]
  num_regions: [1, 1]
  num_uavs_total: [5, 5]
  uav_orin_8gb_ratio: [0.4, 0.4]
  simulation_duration_s: 3600
  uav_fault_probability: 0.0
  sensor_fault_probability: 0.0
env:
  max_steps: 80
  step_seconds: 30
  max_ready_tasks: 8
  region_selection: first_ready
```

通过条件：

- completion_ratio 明显上升。
- timed_out_tasks 接近 0。
- total_reward 有趋势上升。

#### Stage 1：单区域 DAG

目标：学习 DAG 依赖与任务类型差异。

设置：

- 单区域。
- 无通信故障。
- 无 UAV 故障。
- 保留 A/P/I/F/C 任务链。
- deadline 放宽 1.5 到 2 倍。

通过条件：

- makespan 和 ATCT 稳定下降。
- I/F 类 PTCT 不再完全随机。

#### Stage 2：多区域、无故障

目标：学习跨区域共享 UAV 和区域切换。

设置：

- 2 到 3 区域。
- 通信故障率仍为 0。
- UAV 故障率 0。
- `region_selection` 先用 `first_ready` 或固定策略，后续再切回 `random_ready`。

通过条件：

- completion_ratio 高于启发式弱 baseline。
- gpu_util_std 下降。

#### Stage 3：完整 DS1

目标：恢复当前标准场景。

设置：

- 通信故障率 0.1 到 0.4。
- UAV 故障概率 0.03。
- `region_selection=random_ready`。

通过条件：

- 与 Greedy/Min-Load/Round-Robin/HEFT 对比，有至少一个核心指标稳定占优。

## 关键问题 5：模型结构对组合匹配问题不够直接

### 当前结构

当前 actor 是“每个 UAV 输出 max_ready_tasks 个 logits”：

```text
actor_input = concat(global_state, local_uav_obs)
logits[uav, task]
```

然后转置成：

```text
probs[task, uav]
```

### 风险

1. actor 主要以 UAV 为中心输出任务 logits，缺少显式 task-UAV pair 特征。
2. action mask 只在输出端生效，网络内部不知道“为什么某个 pair 可行或不可行”。
3. Critic 只看 global_state，不看具体 action，难以评估组合动作质量。
4. 对可变任务数、可变 UAV 数依赖大量 padding，训练信号容易被空位稀释。

### 结构改造建议

第一阶段不建议大改模型，先修训练机制和 reward。若仍不收敛，再做结构改造。

建议 actor 改成 pairwise scorer：

```text
task_embed_i = TaskEncoder(task_i)
uav_embed_j  = UAVEncoder(uav_j)
pair_feat_ij = concat(task_embed_i, uav_embed_j, global_state, compatibility_features_ij)
score_ij     = MLP(pair_feat_ij)
```

compatibility_features 建议包含：

- task resource / uav available resource。
- task modality 是否被 UAV sensor 支持。
- task required_uav_type 是否匹配。
- UAV 到 task target 的距离。
- 预计飞行时间。
- deadline slack。
- network connected / latency。

这样 actor 输出直接对应每个 task-UAV pair，动作 mask 也更自然。

## 关键问题 6：缺少启发式预训练，纯 RL 起步太慢

当前已有 Greedy、Min-Load、Round-Robin、HEFT。可以用它们做行为克隆预训练。

### 建议方案

先采集专家数据：

```text
obs, action_mask -> heuristic_assignment
```

训练 actor 做监督学习：

```text
loss_bc = CrossEntropy(actor_logits, heuristic_action)
```

预训练策略：

1. 用 HEFT 或 Greedy 生成 5k 到 20k 条样本。
2. actor 先训练到能模仿 heuristic。
3. 再进入 RL fine-tuning。

收益：

- 初始策略不再随机。
- completion_ratio 初期就有合理水平。
- RL 只需要学习超过 heuristic，而不是从零发现可行动作。

## 关键问题 7：训练可观测性不足

当前 `train_cascade.py` 输出 `train_metrics.csv/json`，但没有像 `run_cascade.py` 一样接入 SwanLab。若只看 `run_cascade.py` 的 SwanLab 曲线，会误以为训练没有进展。

建议给 `train_cascade.py` 增加：

- `--use-swanlab`
- `--swanlab-project`
- `--swanlab-workspace`
- `--swanlab-experiment`
- `--swanlab-mode`

每个训练 episode 记录：

```text
train/total_reward
train/loss_actor
train/loss_critic
train/entropy
train/advantage_mean
train/return_mean
train/completed_tasks
train/timed_out_tasks
train/makespan_s
train/atct_s
```

每次 eval 记录：

```text
eval/completion_ratio_mean
eval/tdsr_mean
eval/rpdr_proxy_mean
eval/makespan_s_mean
eval/atct_s_mean
eval/total_reward_mean
```

还应记录诊断指标：

```text
debug/valid_action_count
debug/assigned_count
debug/invalid_action_count
debug/action_entropy
debug/mean_action_prob
debug/max_action_prob
debug/terminal_reason
```

## 推荐实施顺序

## 已实施的第一批小幅改动

当前已先做两处局部改动，没有调整模型主体结构：

1. 训练时动作从确定性 Hungarian 解码改为按策略分布采样。
   - 修改位置：`src/algorithms/cascade/ma3c_trainer.py`
   - `decide()` 仍用于评估，继续使用 Hungarian，保证评估稳定。
   - `decide_with_trace()` 用于训练，改为逐 UAV 在可行动作中采样任务，并记录真实采样动作的 `log_prob` 和 `entropy`。
   - 目的：让 policy gradient 的 `log_prob(action)` 与动作生成过程一致，避免训练长期像随机评估。

2. reward 增加轻量事件反馈。
   - 修改位置：`src/env/cascade_env.py`、`src/env/reward.py`、`configs/env/base.yaml`
   - 每步完成任务增加 `completed_bonus`，默认 `1.0`。
   - 每步超时任务增加 `timeout_penalty`，默认 `2.0`。
   - 目的：让完成任务和超时任务直接进入 reward，再进入 GAE 和 Actor-Critic loss，减少只靠间接 deadline 成本学习的难度。

第一轮复训建议仍使用标准 DS1，但先不要混入更多大改动，观察 1000 到 3000 episodes 内：

```text
train/total_reward 是否上升
train/timed_out_tasks 是否下降
train/completed_tasks 是否上升
eval/completion_ratio_mean 是否上升
eval/makespan_s_mean 或 eval/atct_s_mean 是否下降
train/entropy 是否从较高水平逐步下降
```

## 已实施的第二批小幅改动

根据 1000 episode 曲线，`train/entropy` 在训练早期快速塌缩，`train/loss_critic` 和 `train/total_reward` 仍然强烈抖动。说明当前策略过早变确定，同时 reward 的终局目标仍不够明确。第二批仍只做局部改动：

1. 提高默认探索强度。
   - 修改位置：`src/algorithms/cascade/ma3c_trainer.py`、`experiments/train_cascade.py`、`configs/algorithm/cascade_ma3c.yaml`
   - 默认 `entropy_coef` 从 `0.01` 调整为 `0.05`。
   - 目的：减缓 entropy 过早塌缩，避免 1000 episode 内策略快速锁死在次优分配模式。

2. 增加 episode 终局 reward。
   - 修改位置：`src/env/cascade_env.py`、`src/env/reward.py`、`configs/env/base.yaml`
   - 终局 reward 默认：

```text
terminal_reward =
  5.0 * completion_ratio
+ 2.0 * tdsr
- 5.0 * timeout_ratio
```

   - 目的：让 episode 结束时的完成率、按时完成率、超时率直接进入优化目标，避免训练只依赖每步局部成本。

3. 训练日志增加 reward 分项累计。
   - 修改位置：`experiments/train_cascade.py`
   - SwanLab 会记录 `train/cost_*_sum` 和 `train/reward_*_sum`。
   - 重点观察：

```text
train/reward_completed_bonus_sum
train/reward_timeout_penalty_sum
train/reward_terminal_sum
train/cost_total_sum
train/cost_deadline_sum
train/cost_overload_sum
```

第二轮复训建议仍先跑 1000 episodes，但不要直接接着旧 checkpoint 继续训练。旧 checkpoint 可能已经在低 entropy 策略上塌缩，建议从新随机种子重新开始：

```bash
python experiments/train_cascade.py \
  --config configs/env/scenario_ds1_standard.yaml \
  --eval-config configs/env/scenario_ds1_standard.yaml \
  --train-episodes 1000 \
  --eval-episodes 10 \
  --eval-every 100 \
  --max-steps 300 \
  --model-num-uavs 15 \
  --seed 1 \
  --output-dir outputs/training/cascade_ma3c_ds1_signal_v2_1000ep \
  --use-swanlab \
  --swanlab-mode cloud \
  --swanlab-workspace Linexus \
  --swanlab-project cascade-uav-scheduling \
  --swanlab-experiment cascade-ma3c-ds1-signal-v2-1000ep
```

## 已实施的第三批小幅改动

如果以下曲线都不收敛：

```text
train/cost_total_sum
train/cost_deadline_sum
train/cost_overload_sum
train/reward_completed_bonus_sum
train/reward_timeout_penalty_sum
train/reward_terminal_sum
```

说明问题已经不只是 reward 形状，而是训练场景随机性和任务复杂度把单 episode 学习信号完全淹没。第三批仍不改模型主体，先把训练控制做得更可诊断：

1. 新增 easy curriculum 场景。
   - 文件：`configs/env/scenario_ds0_easy.yaml`
   - 特点：单区域、小地图、无通信故障、无 UAV 故障、任务数更少、固定 `first_ready` 区域选择。
   - 目的：先验证模型是否能在低噪声环境学会基本 task-UAV 匹配。如果在 `ds0_easy` 也不收敛，问题就在动作/算法实现；如果 `ds0_easy` 能收敛而 DS1 不能，问题主要是场景难度和随机性。

2. 训练脚本新增固定 seed 池。
   - 参数：`--seed-pool-size`
   - 默认 `0`，表示每个 episode 继续使用新 seed。
   - 设置为 `32` 或 `64` 时，训练会在固定数量场景上循环，降低随机场景分布带来的 reward 方差。

3. SwanLab 新增滚动均值曲线。
   - 参数：`--rolling-window`，默认 `50`。
   - 新增指标前缀：`train_ma/*`
   - 例如：

```text
train_ma/total_reward
train_ma/reward_terminal_sum
train_ma/reward_timeout_penalty_sum
train_ma/cost_total_sum
train_ma/completed_tasks
train_ma/timed_out_tasks
```

后续判断收敛优先看 `train_ma/*` 和周期性 `eval/*`，不要只看单 episode 的 `train/*`，因为场景每轮都随机重采样，单点曲线天然高噪声。

4. Critic 损失从 MSE 改成 Huber。
   - 修改位置：`src/algorithms/cascade/ma3c_trainer.py`
   - `mse_loss(values, returns)` 改为 `smooth_l1_loss(values, returns)`。
   - 目的：降低高方差 return 对 critic 的尖峰冲击，缓和 `train/loss_critic` 的大幅抖动。

第三轮建议先不要直接跑 DS1，而是先跑 easy 场景：

```bash
python experiments/train_cascade.py \
  --config configs/env/scenario_ds0_easy.yaml \
  --eval-config configs/env/scenario_ds0_easy.yaml \
  --train-episodes 1000 \
  --eval-episodes 20 \
  --eval-every 100 \
  --max-steps 100 \
  --model-num-uavs 15 \
  --seed 0 \
  --seed-pool-size 32 \
  --rolling-window 50 \
  --output-dir outputs/training/cascade_ds0_easy_seedpool32_1000ep \
  --use-swanlab \
  --swanlab-mode cloud \
  --swanlab-workspace Linexus \
  --swanlab-project cascade-uav-scheduling \
  --swanlab-experiment cascade-ds0-easy-seedpool32-1000ep
```

判断条件：

```text
train_ma/total_reward 上升
train_ma/timed_out_tasks 下降
train_ma/reward_timeout_penalty_sum 下降
eval/completion_ratio_mean 上升
eval/timed_out_tasks_mean 下降
```

如果 `ds0_easy` 能收敛，再迁移到 DS1：

```bash
python experiments/train_cascade.py \
  --checkpoint outputs/training/cascade_ds0_easy_seedpool32_1000ep/cascade_ma3c.pt \
  --config configs/env/scenario_ds1_standard.yaml \
  --eval-config configs/env/scenario_ds1_standard.yaml \
  --train-episodes 3000 \
  --eval-episodes 20 \
  --eval-every 100 \
  --max-steps 300 \
  --model-num-uavs 15 \
  --seed 1000 \
  --seed-pool-size 64 \
  --rolling-window 100 \
  --output-dir outputs/training/cascade_ds1_from_easy_seedpool64_3000ep \
  --use-swanlab \
  --swanlab-mode cloud \
  --swanlab-workspace Linexus \
  --swanlab-project cascade-uav-scheduling \
  --swanlab-experiment cascade-ds1-from-easy-seedpool64-3000ep
```

## 已实施的第四批改动：启发式行为克隆 warm-up

如果 `ds0_easy` 和固定 seed pool 仍然不收敛，说明从零 RL 探索对当前 task-UAV 组合匹配仍然太弱。第四批加入可选行为克隆 warm-up，默认关闭，不影响原训练命令。

改动：

1. `CASCADEMA3CScheduler` 增加 `behavior_clone_step()`。
   - 文件：`src/algorithms/cascade/ma3c_trainer.py`
   - 目标：让 actor 对启发式调度器选中的 task-UAV pair 做监督学习。
   - 只训练 encoder + actor，不训练 critic。

2. `train_cascade.py` 增加参数：

```text
--bc-episodes
--bc-teacher {heft,greedy,min_load}
--bc-seed-offset
```

3. SwanLab 新增行为克隆曲线：

```text
bc/loss_bc
bc/bc_targets
bc/bc_entropy
bc/teacher_total_reward
bc/teacher_completed_tasks
bc/teacher_timed_out_tasks
```

推荐先在 easy 场景跑：

```bash
python experiments/train_cascade.py \
  --config configs/env/scenario_ds0_easy.yaml \
  --eval-config configs/env/scenario_ds0_easy.yaml \
  --bc-episodes 300 \
  --bc-teacher heft \
  --train-episodes 1000 \
  --eval-episodes 20 \
  --eval-every 100 \
  --max-steps 100 \
  --model-num-uavs 15 \
  --seed 0 \
  --seed-pool-size 32 \
  --rolling-window 50 \
  --output-dir outputs/training/cascade_ds0_easy_bc_heft_1000ep \
  --use-swanlab \
  --swanlab-mode cloud \
  --swanlab-workspace Linexus \
  --swanlab-project cascade-uav-scheduling \
  --swanlab-experiment cascade-ds0-easy-bc-heft-1000ep
```

如果行为克隆后仍然完全不收敛，下一步不应继续调 reward，而应检查 actor 表达能力：当前 UAV-centric actor 可能无法稳定表达 task-UAV pair 匹配，需要进入 pairwise scorer 改造。

## 已实施的第五批改动：BC 数据集化与多 epoch 训练

如果 `bc/loss_bc` 不下降而是呈规律波动，通常是因为旧版 BC warm-up 是“边采样边训练”：episode 按固定 seed pool 循环，loss 会跟着场景难度周期性起伏，不像标准监督学习那样对同一数据集反复优化。

第五批将 BC 改为两阶段：

1. 先用启发式老师收集固定专家数据集。
2. 再随机打乱数据集，做多轮 BC 训练。

新增参数：

```text
--bc-epochs
--bc-max-transitions
```

推荐命令：

```bash
python experiments/train_cascade.py \
  --config configs/env/scenario_ds0_easy.yaml \
  --eval-config configs/env/scenario_ds0_easy.yaml \
  --bc-episodes 300 \
  --bc-epochs 10 \
  --bc-max-transitions 5000 \
  --bc-teacher heft \
  --train-episodes 1000 \
  --eval-episodes 20 \
  --eval-every 100 \
  --max-steps 100 \
  --model-num-uavs 15 \
  --seed 0 \
  --seed-pool-size 32 \
  --rolling-window 50 \
  --output-dir outputs/training/cascade_ds0_easy_bc_dataset_heft_1000ep \
  --use-swanlab \
  --swanlab-mode cloud \
  --swanlab-workspace Linexus \
  --swanlab-project cascade-uav-scheduling \
  --swanlab-experiment cascade-ds0-easy-bc-dataset-heft-1000ep
```

判断：

```text
bc/loss_bc 应该随 bc epoch 下降
bc/bc_entropy 通常会下降
bc/dataset_size 应该稳定为收集到的 transition 数
```

如果数据集化 BC 的 `bc/loss_bc` 仍然不下降，说明当前 actor 的 UAV-centric 输出结构无法稳定模仿启发式 task-UAV pair 匹配。下一步应进入结构改造：改为 task-UAV pairwise scorer，而不是继续调整 reward。

## 已实施的第六批改动：Pairwise Actor Scorer

数据集化 BC 仍不收敛后，基本可以判断当前 UAV-centric actor 结构不适合表达 task-UAV pair 匹配。第六批将 actor 从“每个 UAV 输出所有任务 logits”改为“每个 task-UAV pair 独立打分”。

旧结构：

```text
actor_input = concat(global_state, uav_local_obs)
actor_output = logits over tasks
```

问题：

- task 信息只通过全局 token 间接进入 actor。
- actor 没有直接看到某个任务和某架 UAV 的资源/传感器/优先级组合。
- BC 的监督目标是 task-UAV pair，但网络输出结构是 UAV-centric task list，表达不够直接。

新结构：

```text
pair_input_ij = concat(
  global_state,
  task_features_i,
  uav_features_j + multihop_features_j,
  normalized_resource_margin_ij
)

score_ij = MLP(pair_input_ij)
```

其中：

```text
normalized_resource_margin_ij = normalized_uav_available_resource_j - normalized_task_requirement_i
```

修改位置：

- `src/algorithms/cascade/actor_network.py`
  - 新增 `build_pairwise_actor()`
- `src/algorithms/cascade/ma3c_trainer.py`
  - `_actor_logits()` 改为构造 `[task, uav]` pair 输入并输出 pair score
  - `decide()`、`decide_with_trace()`、`behavior_clone_step()` 继续复用原有 mask、采样、BC loss 逻辑

注意：

旧 UAV-centric actor checkpoint 与新 pairwise actor 结构不兼容。建议从头训练或从新结构 checkpoint 继续训练。

推荐先只跑 BC 验证，不跑长 RL：

```bash
python experiments/train_cascade.py \
  --config configs/env/scenario_ds0_easy.yaml \
  --eval-config configs/env/scenario_ds0_easy.yaml \
  --bc-episodes 300 \
  --bc-epochs 10 \
  --bc-max-transitions 5000 \
  --bc-teacher heft \
  --train-episodes 0 \
  --eval-episodes 1 \
  --eval-every 1 \
  --max-steps 100 \
  --model-num-uavs 15 \
  --seed 0 \
  --seed-pool-size 32 \
  --rolling-window 50 \
  --output-dir outputs/training/cascade_pairwise_bc_probe \
  --use-swanlab \
  --swanlab-mode cloud \
  --swanlab-workspace Linexus \
  --swanlab-project cascade-uav-scheduling \
  --swanlab-experiment cascade-pairwise-bc-probe
```

预期：

```text
bc/loss_bc 应该明显下降
bc/bc_entropy 应该下降
```

如果 pairwise actor 的 BC loss 能下降，再跑 RL：

```bash
python experiments/train_cascade.py \
  --config configs/env/scenario_ds0_easy.yaml \
  --eval-config configs/env/scenario_ds0_easy.yaml \
  --bc-episodes 300 \
  --bc-epochs 10 \
  --bc-max-transitions 5000 \
  --bc-teacher heft \
  --train-episodes 1000 \
  --eval-episodes 20 \
  --eval-every 100 \
  --max-steps 100 \
  --model-num-uavs 15 \
  --seed 0 \
  --seed-pool-size 32 \
  --rolling-window 50 \
  --output-dir outputs/training/cascade_pairwise_ds0_easy_bc_rl \
  --use-swanlab \
  --swanlab-mode cloud \
  --swanlab-workspace Linexus \
  --swanlab-project cascade-uav-scheduling \
  --swanlab-experiment cascade-pairwise-ds0-easy-bc-rl
```

### P0：先确认不是评估随机策略

必须先做：

1. 用 `train_cascade.py` 训练。
2. 训练后用 checkpoint 评估。
3. 不再用 `run_cascade.py --episodes 1000` 判断“训练收敛”。
4. 给 `run_cascade.py` 增加 `--checkpoint`，或直接使用 `run_simple_comparison.py --checkpoint`。

预期结果：

- 如果只是入口误用，checkpoint 评估会比随机初始化稳定。

### P1：改训练动作采样

必须改：

1. `decide_with_trace()` 训练时使用 stochastic sampling。
2. `decide()` 评估时继续使用 Hungarian/greedy。
3. 记录真实采样动作的 log_prob。

预期结果：

- entropy 曲线有意义。
- actor_loss 与 reward 改善相关性增强。

### P2：重写 reward，使优化目标对齐时间与完成率

必须加：

1. `completed_this_step` 正奖励。
2. `timed_out_this_step` 强惩罚。
3. terminal completion/TDSR/RPDR 奖励。
4. normalized makespan/ATCT 惩罚。
5. GPU 负载均衡惩罚继续保留，但不要作为唯一利用率目标。

预期结果：

- total_reward 与 completion_ratio/makespan 出现一致趋势。
- Critic loss 更稳定。

### P3：课程学习

新增配置：

```text
configs/env/scenario_ds0_easy.yaml
configs/env/scenario_ds1_single_region.yaml
configs/env/scenario_ds1_multiregion_no_fault.yaml
configs/env/scenario_ds1_standard.yaml
```

训练顺序：

```text
ds0_easy -> ds1_single_region -> ds1_multiregion_no_fault -> ds1_standard
```

每阶段从上一阶段 checkpoint 继续训练。

### P4：行为克隆预训练

如果 P0-P3 后仍慢：

1. 用 HEFT/Greedy 采样专家动作。
2. actor 先做 BC。
3. 再用 RL fine-tuning。

### P5：pairwise actor 结构

如果仍无法超过启发式：

1. 从 UAV-centric actor 改成 task-UAV pair scorer。
2. 加入距离、预计完成时间、deadline slack、资源余量、网络延迟等 pair 特征。
3. 可考虑 action-value critic 或 centralized critic with action embedding。

## 建议的最小代码改动清单

### 1. 修 `run_cascade.py`

增加参数：

```text
--checkpoint
--model-num-uavs
```

并使用：

```python
from src.algorithms.cascade.ma3c_trainer import cascade_factory
```

构建：

```python
CASCADE_METHODS = {
    "cascade_ma3c": cascade_factory(..., checkpoint=args.checkpoint),
}
```

### 2. 给 `train_cascade.py` 加 SwanLab

复用 `src/evaluation/swanlab_logger.py`，记录训练曲线和 eval 曲线。

### 3. 改 `CASCADEMA3CScheduler.decide_with_trace`

训练用采样：

```text
probs = masked_softmax(logits)
assignments = sample_assignments(probs, action_mask)
log_prob = sum(log_probs of sampled assignments)
entropy = sum/mean entropy of categorical distributions
```

评估保留 Hungarian。

### 4. 改 `CASCADEEnv.step` 的 reward

把 `completed_this_step`、`timed_out_this_step`、episode terminal metrics 显式传给 reward。

### 5. 新增 easy curriculum 配置

先在简单场景验证能不能学会，再逐步增加复杂度。

## 判断是否真正开始收敛

不要只看 `total_reward`。建议同时满足：

1. `train/entropy` 先保持较高，随后逐步下降。
2. `train/completed_tasks` 上升。
3. `train/timed_out_tasks` 下降。
4. `eval/completion_ratio_mean` 上升。
5. `eval/makespan_s_mean` 或 `eval/atct_s_mean` 下降。
6. 同一 checkpoint 在固定 seed 评估下优于随机初始化策略。
7. 至少在 `ds0_easy` 上明显超过随机，再进入标准 DS1。

## 推荐实验命令

第一阶段先跑 easy curriculum：

```bash
python experiments/train_cascade.py \
  --config configs/env/scenario_ds0_easy.yaml \
  --eval-config configs/env/scenario_ds0_easy.yaml \
  --train-episodes 1000 \
  --eval-episodes 20 \
  --eval-every 50 \
  --model-num-uavs 15 \
  --seed 0 \
  --output-dir outputs/training/cascade_ds0_easy_1000ep
```

然后继续到标准 DS1：

```bash
python experiments/train_cascade.py \
  --checkpoint outputs/training/cascade_ds0_easy_1000ep/cascade_ma3c.pt \
  --config configs/env/scenario_ds1_standard.yaml \
  --eval-config configs/env/scenario_ds1_standard.yaml \
  --train-episodes 3000 \
  --eval-episodes 20 \
  --eval-every 100 \
  --model-num-uavs 15 \
  --seed 1000 \
  --output-dir outputs/training/cascade_ds1_from_easy_3000ep
```

训练后评估：

```bash
python experiments/run_simple_comparison.py \
  --checkpoint outputs/training/cascade_ds1_from_easy_3000ep/cascade_ma3c.pt \
  --model-num-uavs 15 \
  --scenarios ds1 ds2 ds3 \
  --methods cascade_ma3c greedy min_load round_robin heft \
  --episodes 50 \
  --seed 0 \
  --output-dir outputs/results/cascade_checkpoint_comparison
```

## 最终建议

为了让训练真正收敛，最优先不是加大 episode 数，而是先让问题变成“可学习”：

1. 不再用随机初始化评估曲线判断收敛。
2. 训练时必须引入真实策略采样。
3. reward 必须直接绑定完成率、超时、makespan、ATCT 和资源均衡。
4. 从单区域无故障简单场景开始课程学习。
5. 必要时用 HEFT/Greedy 做行为克隆预训练。

如果只把当前实现从 1000 episode 加到 10000 episode，大概率仍然表现为随机波动，因为最主要的学习信号和动作生成机制还没有对齐。
