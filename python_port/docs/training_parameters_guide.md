# GCN-PPO Pro 训练参数完全参考手册

> **适用版本**：`PetriNetGCNPPOPro` / `PetriNetGCNPPOProHQ`  
> **核心文件**：`petri_gcn_ppo_4_1.py` · `train_ppo_3.py`  
> **更新日期**：2026-04-24

---

## 目录

1. [参数体系总览](#1-参数体系总览)
2. [GCN 网络结构参数](#2-gcn-网络结构参数)
3. [训练流程控制参数](#3-训练流程控制参数)
4. [PPO 核心算法参数](#4-ppo-核心算法参数)
5. [奖励函数设计参数](#5-奖励函数设计参数)
6. [推理搜索策略参数](#6-推理搜索策略参数)
7. [多环境泛化训练参数](#7-多环境泛化训练参数)
8. [系统与工程参数](#8-系统与工程参数)
9. [环境变量速查表](#9-环境变量速查表)
10. [场景化配置方案](#10-场景化配置方案)
11. [参数调优原则与实践](#11-参数调优原则与实践)
12. [常见问题诊断](#12-常见问题诊断)

---

## 1. 参数体系总览

本框架面向 **Petri 网调度问题**，采用 GCN（图卷积网络）作为策略网络骨干，结合 PPO（近端策略优化）进行强化学习训练。参数按功能划分为 7 个层次：

```
┌─────────────────────────────────────────────────────────┐
│                  PetriNetGCNPPOProHQ                    │
│                                                         │
│  ① GCN结构  ─── lambda_p / lambda_t / extra_rounds     │
│  ② 训练控制 ─── max_steps / steps_per_epoch / lr       │
│  ③ PPO核心  ─── gamma / gae_lambda / eps_clip          │
│  ④ 奖励设计 ─── goal_bonus / deadlock_penalty / ...    │
│  ⑤ 搜索推理 ─── strategy / beam_width / beam_depth     │
│  ⑥ 多网泛化 ─── mixed_rollout / curriculum / GAE      │
│  ⑦ 系统工程 ─── device / cache / verbose              │
└─────────────────────────────────────────────────────────┘
```

### 参数继承关系

```python
# PetriNetGCNPPOPro  ←  基类，定义所有参数默认值（保守配置）
# PetriNetGCNPPOProHQ ←  子类，覆盖为高质量多网泛化配置
# 所有参数均可通过环境变量在运行时动态覆盖（免代码修改）
```

---

## 2. GCN 网络结构参数

这三个参数**决定策略网络的参数规模**，一旦训练开始不可更改（改变会使 checkpoint 不兼容）。

### 2.1 参数定义

| 参数名 | 类型 | 基类默认 | HQ默认 | 环境变量 |
|--------|------|---------|--------|---------|
| `lambda_p` | `int` | `256` | `256` | `GCN_PPO_HQ_LAMBDA_P` |
| `lambda_t` | `int` | `64` | `64` | `GCN_PPO_HQ_LAMBDA_T` |
| `extra_p2t_rounds` | `int` | `2` | `5` | `GCN_PPO_HQ_EXTRA_P2T_ROUNDS` |

#### `lambda_p` — 库所隐层维度

GCN 中每个**库所节点**的特征向量维度。决定模型对库所状态的表达能力。

> **关键约束**：所有可学习权重矩阵均在 `lambda_p`/`lambda_t` 空间内，**与网络拓扑规模（库所数 P、变迁数 T）完全解耦**，因此同一组权重可迁移到不同规模的 Petri 网。

#### `lambda_t` — 变迁隐层维度

GCN 中每个**变迁节点**的特征向量维度。`lambda_t` 的输出直接对应每个变迁的 logit 值，决定动作选择的分辨率。

通常 `lambda_t` 取 `lambda_p` 的 1/4，原因：
- 库所是"状态承载者"，需要更大容量
- 变迁是"决策单元"，维度过大导致训练不稳定

#### `extra_p2t_rounds` — 额外消息传递轮次

GCN 在标准 P→T→P 消息传递后，额外执行的 P↔T 双向传递轮数。

```
标准结构：P → T → P
完整结构：P → T → P → [T → P → T → P] × extra_rounds
                       ↑ extra_p2t_rounds 控制这里的重复次数
```

- 增大此值 = 增加感受野，节点可聚合更远邻域的信息
- 直接影响前向/后向计算时间（近似线性增长）

### 2.2 规模-性能对照表

| 场景 | `lambda_p` | `lambda_t` | `extra_rounds` | 参数量约估 | 适用规模 |
|------|-----------|-----------|---------------|---------|---------|
| 轻量调试 | 64 | 32 | 1 | ~0.1M | P≤10, T≤15 |
| 标准训练 | 128 | 64 | 2 | ~0.3M | P≤20, T≤30 |
| **HQ 默认** | **256** | **64** | **5** | **~0.8M** | **P≤50, T≤60** |
| 大规模网络 | 512 | 128 | 6 | ~3M | P≤100, T≤120 |

### 2.3 调整影响分析

```
lambda_p ↑  →  表达力 ↑，训练时间 ↑，过拟合风险 ↑
lambda_t ↑  →  动作分辨率 ↑，梯度稳定性 ↓（需配合降低lr）
extra_rounds ↑  →  远程依赖捕获 ↑，但训练时间线性增加，梯度消失风险 ↑
```

---

## 3. 训练流程控制参数

### 3.1 参数定义

| 参数名 | 类型 | 基类默认 | HQ默认 | 环境变量 |
|--------|------|---------|--------|---------|
| `max_train_steps` | `int` | `150000` | 动态计算 | — |
| `steps_per_epoch` | `int` | `4096` | `6144` | `GCN_PPO_HQ_STEPS_PER_EPOCH` |
| `minibatch_size` | `int` | `128` | `128` | `GCN_PPO_HQ_MINIBATCH_SIZE` |
| `ppo_epochs` | `int` | `10` | `4` | `GCN_PPO_HQ_PPO_EPOCHS` |
| `lr` | `float` | `1e-4` | `3e-4` | `GCN_PPO_HQ_LR` |

#### `max_train_steps` — 总训练步数上限

HQ 模式下自动按网络复杂度计算：

```python
# 标准模式（GCN_PPO_HQ_FAST != "0"）
base_steps  = 10000 × env_count
extra_steps = (complexity × 2000 + constrained_count × 3000) × env_count
max_train_steps = clamp(base_steps + extra_steps, 50000, 307200)

# 快速模式（GCN_PPO_HQ_FAST == "0"）
max_train_steps = 25000 × env_count
```

- `complexity`：训练集中最大库所数与最大变迁数的较大值
- `constrained_count`：最大驻留时间约束数量

#### `steps_per_epoch` — 每 epoch 收集步数

每次 PPO 更新前，在环境中执行的总交互步数（不等于回合数）。

> **关键权衡**：  
> - 过小（<2048）→ 样本量不足，梯度估计方差大，KL 早停频繁触发  
> - 过大（>16384）→ 单次采集时间长，训练数据老化，重要性比率偏差增大

#### `ppo_epochs` — PPO 内层更新轮数

对同一批数据执行梯度更新的次数。

> **HQ 模式设为 4，低于基类默认 10**，原因：  
> 多网训练时，每次环境切换都重建模型拓扑，过多更新轮数会导致 off-policy 偏差积累；配合 `target_kl` 早停机制，4 轮可在利用率与策略稳定性间取得平衡。

#### `lr` — 学习率

使用线性学习率衰减策略：

```python
# train_model 内每 epoch 更新
new_lr = max(1e-5, initial_lr × (1.0 - progress))
# progress = total_steps / max_train_steps ∈ [0, 1]
```

训练结束时学习率降至 `1e-5`，防止训练后期参数震荡。

### 3.2 步数配置参考

| 训练目标 | `max_train_steps` | `steps_per_epoch` | `ppo_epochs` | 预估 epoch 数 |
|---------|------------------|------------------|-------------|-------------|
| 快速验证 | 20,000 | 2048 | 3 | ~10 |
| 单网精调 | 80,000 | 4096 | 6 | ~20 |
| **HQ 多网（推荐）** | **50k–307k** | **6144** | **4** | **~8–50** |
| 超大网络 | 307,200 | 8192 | 4 | ~37 |

### 3.3 minibatch_size 与显存关系

```
显存占用 ≈ minibatch_size × max(P, T) × lambda_p × 4 bytes × 2（前向+梯度）

示例（lambda_p=256, P=30, T=40）:
  minibatch=64  → ~50 MB
  minibatch=128 → ~100 MB  ← HQ 默认
  minibatch=256 → ~200 MB
```

---

## 4. PPO 核心算法参数

### 4.1 参数定义

| 参数名 | 类型 | 基类默认 | HQ默认 | 环境变量 | 作用域 |
|--------|------|---------|--------|---------|-------|
| `gamma` | `float` | `0.99` | `0.999` | `GCN_PPO_HQ_GAMMA` | GAE |
| `gae_lambda` | `float` | `0.95` | `0.95` | — | GAE |
| `eps_clip` | `float` | `0.2` | `0.2` | — | PPO clip |
| `target_kl` | `float` | `0.04` | `0.09` | `GCN_PPO_HQ_TARGET_KL` | 早停 |
| `value_loss_coef` | `float` | `0.5` | `0.5` | — | 损失函数 |
| `entropy_coef_start` | `float` | `0.09` | `0.15` | `GCN_PPO_HQ_ENTROPY_START` | 探索 |
| `entropy_coef_end` | `float` | `0.01` | `0.015` | `GCN_PPO_HQ_ENTROPY_END` | 探索 |
| `temperature_start` | `float` | `2.0` | `2.0` | `GCN_PPO_HQ_TEMPERATURE_START` | 采样 |
| `temperature_end` | `float` | `1.1` | `1.1` | `GCN_PPO_HQ_TEMPERATURE_END` | 采样 |

### 4.2 折扣因子 `gamma`

控制未来奖励对当前决策的影响程度：

$$V(s_t) = r_t + \gamma \cdot r_{t+1} + \gamma^2 \cdot r_{t+2} + \cdots$$

| `gamma` | 有效回溯步数（95%贡献）| 适用场景 |
|---------|-------------------|---------|
| `0.95` | ~20 步 | 短时决策、稠密奖励 |
| `0.99` | ~100 步 | 中长程调度 |
| **`0.999`** | **~1000 步** | **长序列调度（HQ默认）** |
| `0.9999` | ~10000 步 | 极长序列，但方差极大 |

> **为何 HQ 使用 0.999**：Petri 网调度目标（makespan）在序列末尾才兑现，序列长度可达 200-800 步，`gamma=0.99` 时末尾奖励折扣后仅剩约 $0.99^{400} \approx 0.018$，信号几乎消失；`0.999` 时为 $0.999^{400} \approx 0.67$，信号有效传播。

### 4.3 GAE 参数 `gae_lambda`

广义优势估计中的平滑系数，在偏差与方差间权衡：

$$A^{GAE}(s_t) = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

| `gae_lambda` | 偏差 | 方差 | 特点 |
|-------------|------|------|------|
| `0.0` | 低（TD(0)）| 低 | 对值函数误差敏感 |
| `0.95` | 中 | 中 | **标准推荐** |
| `1.0` | 高（MC）| 高 | 不依赖值函数，但方差大 |

### 4.4 PPO 裁剪系数 `eps_clip`

限制策略更新幅度，防止单次更新步子过大：

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

- `eps_clip=0.1`：保守更新，适合值函数不稳定初期
- `eps_clip=0.2`：**标准值，大多数场景适用**
- `eps_clip=0.3`：激进更新，适合奖励信号稳定且策略需快速改变时

### 4.5 KL 早停阈值 `target_kl`

每次 PPO 内层更新后计算近似 KL 散度；超过阈值立即停止本 epoch 的后续更新：

```python
approx_kl = ((exp(log_ratio) - 1.0) - log_ratio).mean()
# 若 approx_kl > target_kl: 中止本 epoch 剩余更新
```

| `target_kl` | 效果 | 适用场景 |
|-------------|------|---------|
| `0.01–0.03` | 非常保守，更新频繁被截断 | 值函数初期不稳定 |
| `0.04–0.05` | 基类默认，较保守 | 单网训练 |
| **`0.09`** | **HQ默认，中等** | **多网训练，策略需要更大更新空间** |
| `0.15–0.2` | 激进，几乎不截断 | 训练后期精调 |

### 4.6 熵系数线性衰减 `entropy_coef_start / end`

熵奖励鼓励策略保持随机性（探索）：

$$L_{total} = L_{clip} + c_v \cdot L_{value} - c_h \cdot H(\pi)$$

$$c_h(t) = c_{h,start} - \text{progress}(t) \cdot (c_{h,start} - c_{h,end})$$

```
训练进度:  0% ──────────────────── 100%
熵系数:   0.15 ──线性衰减──────── 0.015
效果:      高探索 ────────────── 聚焦利用
```

| 阶段 | `entropy_coef` 建议值 | 行为 |
|------|---------------------|------|
| 早期（冷启动） | `0.15–0.20` | 广泛探索，避免早熟收敛 |
| 中期 | `0.05–0.10` | 平衡探索与利用 |
| 后期（精调） | `0.01–0.02` | 聚焦最优策略 |

### 4.7 采样温度衰减 `temperature_start / end`

控制 rollout 采样时的动作分布平滑度（**注意：T=1.0 用于 logprob 计算，与采样解耦**）：

```python
# rollout 采样（使用温度增加多样性）
action = Categorical(logits / temperature).sample()

# PPO 更新（使用 T=1.0 保证 log_ratio 准确性）
logprob = Categorical(logits).log_prob(action)  # 固定 T=1.0
```

| `temperature` | 采样行为 | 适用阶段 |
|--------------|---------|---------|
| `2.0` | 接近均匀采样，高多样性 | 训练早期 |
| `1.5` | 中等随机性 | 训练中期 |
| `1.1` | 接近贪婪，低随机性 | 训练后期 |
| `1.0` | 等同于原始 logits | 推理时（固定） |

---

## 5. 奖励函数设计参数

### 5.1 奖励公式

每一步的奖励由以下分量合成：

$$r_t = -\frac{\Delta\text{time}}{\text{time\_scale}} + w_{prog} \cdot \Delta\text{goal\_dist} - w_{rep} \cdot \text{repeat\_count}(s')$$

终止状态额外奖励：

$$r_{terminal} = \begin{cases} +B_{goal} & \text{首次到达目标} \\ +B_{goal} + \min(300, \frac{\text{improvement}}{\text{time\_scale}} \times 100) & \text{刷新最优} \\ +\max(0.2 \cdot B_{goal},\; B_{goal} - \frac{\text{degradation}}{\text{time\_scale}} \times 50) & \text{到达但非最优} \\ -P_{deadlock} & \text{死锁} \end{cases}$$

最终裁剪：$r_t = \text{clip}(r_t, -100, 100)$

### 5.2 参数定义与调优指南

| 参数名 | HQ默认 | 环境变量 | 作用 |
|--------|--------|---------|------|
| `reward_goal_bonus` | `1000.0` | `GCN_PPO_HQ_REWARD_GOAL` | 到达目标的基础奖励 |
| `reward_deadlock_penalty` | `2000.0` | `GCN_PPO_HQ_REWARD_DEADLOCK` | 死锁惩罚 |
| `reward_progress_weight` | `2.0` | — | 每减少1单位目标距离的奖励 |
| `reward_repeat_penalty` | `1.5` | `GCN_PPO_HQ_REWARD_REPEAT` | 重复访问状态的累计惩罚 |
| `reward_time_scale` | `1000.0` | `GCN_PPO_HQ_REWARD_TIME_SCALE` | 时间代价的缩放基数 |

#### `reward_time_scale` 的选取依据

```
step 时间代价 = Δtime / reward_time_scale

若典型 Δtime ≈ 10（单位时间），time_scale=1000：
  每步时间代价 ≈ -0.01（非常小）
  
目的：使时间代价在奖励中占比适中，不能压倒 progress_weight
建议：time_scale ≈ 100 × 典型单步Δtime
```

#### `reward_repeat_penalty` 的作用

防止策略陷入**状态循环**（反复访问同一状态无进展）：

```python
repeat_penalty = seen_count[state_key] × reward_repeat_penalty
# 每多访问一次该状态，惩罚增加 1.5
```

- 过小（<0.5）：策略可能陷入局部循环
- 过大（>3.0）：策略过于保守，不敢重新访问关键节点

#### 奖励幅度一致性检查

```
goal_bonus / time_scale = 1000 / 1000 = 1.0  （无量纲）
deadlock_penalty / clip_bound = 2000 / 100 = 20（超过clip，始终为-100）

警告：deadlock_penalty > 100 时会被 clip 截断，
      所有死锁结果的实际惩罚等同（-100），失去区分度。
      建议将 reward_deadlock_penalty 设为不超过 90，
      或将 clip 上限提高为 ±2000。
```

---

## 6. 推理搜索策略参数

训练完成后，策略网络通过搜索策略生成最终解。

### 6.1 三种搜索策略对比

| 参数 | `beam`（束搜索）| `greedy`（贪婪）| `stochastic`（随机采样）|
|------|---------------|-----------------|----------------------|
| 解质量 | ⭐⭐⭐ 最高 | ⭐⭐ 中等 | ⭐⭐ 中等（依赖轨迹数）|
| 计算开销 | 高（O(beam_width)）| 低（O(1)）| 中（O(num_rollouts)）|
| 内存需求 | 高 | 极低 | 低 |
| 循环处理 | 访问记录去重 | cycle检测即停 | 温度调节多样性 |
| 推荐场景 | 精确求解、推理阶段 | 快速评估、训练中验证 | 探索性测试 |

### 6.2 搜索参数

| 参数名 | 类型 | 默认值 | 环境变量 |
|--------|------|--------|---------|
| `search_strategy` | `str` | `"beam"` | — |
| `beam_width` | `int` | `100` | `GCN_PPO_HQ_BEAM_WIDTH` |
| `beam_depth` | `int` | `800` | `GCN_PPO_HQ_BEAM_DEPTH` |
| `stochastic_num_rollouts` | `int` | `50` | `GCN_PPO_HQ_STOCHASTIC_NUM_ROLLOUTS` |
| `stochastic_temperature` | `float` | `1.2` | `GCN_PPO_HQ_STOCHASTIC_TEMPERATURE` |

#### `beam_width` — 束搜索宽度

每一搜索步保留的候选路径数。

```
beam_width=1   ← 退化为贪婪搜索
beam_width=10  ← 轻量级束搜索（快速评估）
beam_width=100 ← HQ默认（精度与速度平衡）
beam_width=500 ← 高质量解（推理阶段时间充足时）
```

评分函数：`score = makespan + goal_distance × 2.0`（倾向于快速且接近目标的路径）

#### `beam_depth` — 最大搜索步数

所有搜索策略的统一步数上限，防止无限循环。

```
建议值 = max_expert_steps × 2.0（参考专家序列长度的两倍）
HQ默认 800 → 适合典型 200-400 步的调度序列
```

---

## 7. 多环境泛化训练参数

这是本框架的核心功能区，控制多 Petri 网同时训练的行为。

### 7.1 参数定义

| 参数名 | 类型 | HQ默认 | 环境变量 | 说明 |
|--------|------|--------|---------|------|
| `mixed_rollout` | `bool` | `True` | `GCN_PPO_HQ_MIXED_ROLLOUT` | 启用多环境混合经验收集 |
| `envs_per_epoch` | `int` | `2` | `GCN_PPO_HQ_ENVS_PER_EPOCH` | 每 epoch 采样的环境数（0=全部）|
| `cross_env_gae` | `bool` | `True` | `GCN_PPO_HQ_CROSS_ENV_GAE` | 跨环境统一归一化优势函数 |
| `dynamic_curriculum` | `bool` | `True` | `GCN_PPO_HQ_DYNAMIC_CURRICULUM` | 动态课程学习 |
| `curriculum_warmup_ratio` | `float` | `0.3` | `GCN_PPO_HQ_CURRICULUM_WARMUP_RATIO` | 预热阶段训练比例 |
| `curriculum_epochs` | `int` | `4` | `GCN_PPO_HQ_CURRICULUM_EPOCHS` | 课程学习最少预热 epoch 数 |
| `pool_eval_interval` | `int` | `4` | `GCN_PPO_HQ_POOL_EVAL_INTERVAL` | 全池贪婪评估间隔 |

### 7.2 混合经验收集机制（`mixed_rollout`）

启用后，每个 epoch 的训练流程变为：

```
┌─── 预热阶段（progress < curriculum_warmup_ratio）───┐
│ 每个被选环境收集 steps_per_epoch 步（完整轨迹）       │
└──────────────────────────────────────────────────┘
                        ↓
┌─── 正式阶段（progress ≥ curriculum_warmup_ratio）──┐
│ 步数均分：steps_per_env = steps_per_epoch / n_envs  │
│ 多环境按优先级权重采样                               │
└──────────────────────────────────────────────────┘
```

**为何预热阶段使用完整轨迹**：策略初期几乎无法到达目标，截断收集会导致大量无意义的半段轨迹；完整轨迹保证至少采集到终止状态（成功或死锁），GAE 计算更准确。

### 7.3 课程学习优先级采样权重

三个因素综合决定每个环境的采样概率：

$$w_{env} = w_{complexity} \times w_{difficulty} \times w_{coverage}$$

**复杂度权重**（随训练进度变化方向）：

```python
# 预热阶段：简单环境优先
complexity_weight = 1.0 - complexity_ratio × 0.7

# 正式阶段：复杂环境优先（线性过渡）
adjusted_progress = (progress - warmup_ratio) / (1.0 - warmup_ratio)
complexity_weight = 0.3 + 0.7 × complexity_ratio × adjusted_progress
```

**困难度权重**：

```python
# 未到达目标：高优先级采样
difficulty_weight = 2.0  if not reached_goal
# 已到达目标：正常采样
difficulty_weight = 0.5  if reached_goal
```

**覆盖度权重**：

```python
coverage_weight = (min_visit + 1.0) / (env_visits + 1.0)
# 访问越少的环境，权重越高，保证全面覆盖
```

### 7.4 跨环境 GAE 归一化（`cross_env_gae`）

**启用时**（推荐）：

```
所有环境的原始优势值 → 合并 → 全局均值/标准差 → 统一归一化
效果：保留各环境间优势幅度的相对关系
     高奖励环境的梯度贡献相对更大
```

**禁用时**：

```
每个环境的优势值 → 独立均值/标准差 → 各自归一化
效果：各环境梯度权重相等，消除环境间奖励尺度差异
     适合训练网络奖励范围差异极大的情况
```

> **注意**：P1 修复后，已移除 `_update_ppo` 中对预计算优势的二次归一化，`cross_env_gae` 的效果得以完整保留。

### 7.5 `envs_per_epoch` 设置建议

| 训练环境总数 | `envs_per_epoch` 建议 | 理由 |
|------------|---------------------|------|
| 2–3 | `0`（全选）| 环境数少，全选代价低 |
| 4–6 | `2–3` | 每 epoch 不必全覆盖，保证足够步数/环境 |
| **6（当前）** | **`4`（main 中硬编码）** | **覆盖率与效率平衡** |
| 10+ | `4–5` | 步数固定时每环境步数太少，限制采样数 |

---

## 8. 系统与工程参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `use_deadlock_controller` | `bool` | `True` | 使用前向分析控制器预测可达变迁，比直接调用 enable() 更高效 |
| `mask_cache_limit` | `int` | `40000` | 动作掩码缓存的最大条目数（LRU 淘汰）|
| `pool_eval_interval` | `int` | `4` | 每隔几个 epoch 对全池做贪婪评估（计算成本较高）|
| `verbose` | `bool` | `True` | 是否打印每 epoch 训练日志 |
| `device` | `str\|None` | `None`（自动）| 强制指定 `"cuda"` 或 `"cpu"` |

#### `mask_cache_limit` — 掩码缓存大小

动作掩码（每步合法变迁集合）计算涉及完整的死锁可达性分析，开销较大。缓存以状态的 `(p_info, prefix, t_info, ...)` 元组为键：

```
cache_limit=0       ← 禁用缓存（每步重新计算，训练最慢）
cache_limit=40000   ← HQ默认（约覆盖2-3个完整训练轮）
cache_limit=100000  ← 大内存机器上可显著提速
```

显存估算：每条目约 50-200 bytes，40000 条 ≈ 8-80 MB。

---

## 9. 环境变量速查表

所有核心参数均可通过环境变量在**不修改代码**的情况下覆盖：

```bash
# === GCN 结构 ===
export GCN_PPO_HQ_LAMBDA_P=256
export GCN_PPO_HQ_LAMBDA_T=64
export GCN_PPO_HQ_EXTRA_P2T_ROUNDS=5

# === 训练控制 ===
export GCN_PPO_HQ_LR=3e-4
export GCN_PPO_HQ_STEPS_PER_EPOCH=6144
export GCN_PPO_HQ_MINIBATCH_SIZE=128
export GCN_PPO_HQ_PPO_EPOCHS=4

# === PPO 核心 ===
export GCN_PPO_HQ_GAMMA=0.999
export GCN_PPO_HQ_TARGET_KL=0.09
export GCN_PPO_HQ_ENTROPY_START=0.15
export GCN_PPO_HQ_ENTROPY_END=0.015
export GCN_PPO_HQ_TEMPERATURE_START=2.0
export GCN_PPO_HQ_TEMPERATURE_END=1.1

# === 奖励设计 ===
export GCN_PPO_HQ_REWARD_GOAL=1000.0
export GCN_PPO_HQ_REWARD_DEADLOCK=2000.0
export GCN_PPO_HQ_REWARD_REPEAT=1.5
export GCN_PPO_HQ_REWARD_TIME_SCALE=1000.0

# === 搜索推理 ===
export GCN_PPO_HQ_BEAM_WIDTH=100
export GCN_PPO_HQ_BEAM_DEPTH=800
export GCN_PPO_HQ_STOCHASTIC_NUM_ROLLOUTS=50
export GCN_PPO_HQ_STOCHASTIC_TEMPERATURE=1.2

# === 多环境训练 ===
export GCN_PPO_HQ_MIXED_ROLLOUT=1
export GCN_PPO_HQ_ENVS_PER_EPOCH=2
export GCN_PPO_HQ_CROSS_ENV_GAE=1
export GCN_PPO_HQ_DYNAMIC_CURRICULUM=1
export GCN_PPO_HQ_CURRICULUM_WARMUP_RATIO=0.3
export GCN_PPO_HQ_CURRICULUM_EPOCHS=4
export GCN_PPO_HQ_POOL_EVAL_INTERVAL=4

# === 训练文件 ===
export GCN_PPO_HQ_TRAIN_FILES="net1.txt,net2.txt,net3.txt"
export GCN_PPO_HQ_EVAL_FILES="test1.txt,test2.txt"
export GCN_PPO_HQ_FAST=1   # 1=标准, 0=快速模式

# === 权重复用 ===
export GCN_PPO_HQ_CHECKPOINT_PATH="/path/to/checkpoint.pt"
export GCN_PPO_HQ_FINETUNE_FROM_CUSTOM=1  # 1=继续训练, 0=仅推理
export GCN_PPO_HQ_IL_WARMSTART=0          # 1=启用IL热启动（注意：逻辑反向）
export GCN_PPO_HQ_IL_MODE=bc              # auto / bc / dagger
```

> **Windows PowerShell 设置方式**：
> ```powershell
> $env:GCN_PPO_HQ_LAMBDA_P = "256"
> $env:GCN_PPO_HQ_TRAIN_FILES = "1-1-9.txt,1-1-13-1.txt"
> ```

---

## 10. 场景化配置方案

### 方案 A：快速调试 / 验证代码正确性

**目标**：5 分钟内完成一次完整训练，验证代码链路  
**特点**：小网络，少步数，禁用耗时功能

```python
search = PetriNetGCNPPOProHQ(
    ...,
    max_train_steps=15000,
    lambda_p=64,
    lambda_t=32,
    extra_p2t_rounds=1,
    steps_per_epoch=1024,
    ppo_epochs=2,
    lr=3e-4,
    gamma=0.99,
    beam_width=10,
    beam_depth=200,
    pool_eval_interval=100,   # 几乎不评估池
    envs_per_epoch=0,          # 全部环境
    verbose=True,
)
```

```bash
export GCN_PPO_HQ_FAST=0
export GCN_PPO_HQ_LAMBDA_P=64
export GCN_PPO_HQ_EXTRA_P2T_ROUNDS=1
export GCN_PPO_HQ_PPO_EPOCHS=2
export GCN_PPO_HQ_BEAM_WIDTH=10
```

---

### 方案 B：单网精调 / 求解特定 Petri 网

**目标**：对单一网络求得高质量调度序列  
**特点**：高表达力模型，充足步数，束搜索推理

```python
# train_ppo_3.py 中 search_strategy 改为 "beam"
search = PetriNetGCNPPOProHQ(
    ...,
    max_train_steps=150000,
    lambda_p=256,
    lambda_t=64,
    extra_p2t_rounds=5,
    steps_per_epoch=4096,
    ppo_epochs=6,
    lr=1e-4,
    gamma=0.999,
    entropy_coef_start=0.10,
    entropy_coef_end=0.01,
    temperature_start=1.8,
    temperature_end=1.05,
    beam_width=200,
    beam_depth=1000,
    mixed_rollout=False,       # 单网：禁用多环境混合
    search_strategy="beam",
)
```

---

### 方案 C：标准多网泛化训练（HQ 默认配置）

**目标**：在 4-6 个网络上训练，泛化到新网络族  
**特点**：课程学习 + 跨环境 GAE + 动态采样权重

```python
# main() 中的配置就是此方案
# 关键参数：
search = PetriNetGCNPPOProHQ(
    ...,
    max_train_steps=max_train_steps,   # 自动计算
    mixed_rollout=True,
    envs_per_epoch=4,
    cross_env_gae=True,
    dynamic_curriculum=True,
    curriculum_warmup_ratio=0.3,
    curriculum_epochs=4,
    pool_eval_interval=4,
    search_strategy="greedy",          # 推理用贪婪（快速）
)
```

**运行命令**：

```bash
export GCN_PPO_HQ_TRAIN_FILES="1-1-9.txt,1-1-13-1.txt,1-1-13-2.txt,1-1-16.txt"
export GCN_PPO_HQ_EVAL_FILES="family2-net1.txt,family2-net2.txt"
python train_ppo_3.py
```

---

### 方案 D：从预训练权重微调（迁移学习）

**目标**：利用已有 checkpoint 快速适应新网络族  
**特点**：短步数 + 低学习率 + 禁用高熵探索

```bash
export GCN_PPO_HQ_CHECKPOINT_PATH="checkpoints/pretrained.pt"
export GCN_PPO_HQ_FINETUNE_FROM_CUSTOM=1
export GCN_PPO_HQ_LR=5e-5               # 低学习率
export GCN_PPO_HQ_ENTROPY_START=0.05    # 低熵（已有先验知识）
export GCN_PPO_HQ_ENTROPY_END=0.005
export GCN_PPO_HQ_TEMPERATURE_START=1.3  # 低温（收敛阶段）
export GCN_PPO_HQ_SIMILAR_FINETUNE_SCALE=0.35   # 步数 ×0.35
```

---

### 方案 E：大规模高约束网络（≥30个库所 + 多驻留约束）

**目标**：求解复杂约束调度问题，追求最优 makespan  
**特点**：大模型，高 gamma，多搜索轨迹

```python
search = PetriNetGCNPPOProHQ(
    ...,
    lambda_p=512,
    lambda_t=128,
    extra_p2t_rounds=6,
    gamma=0.999,
    steps_per_epoch=8192,
    ppo_epochs=4,
    lr=1e-4,
    target_kl=0.07,
    entropy_coef_start=0.18,
    entropy_coef_end=0.02,
    temperature_start=2.5,
    reward_goal_bonus=1500.0,
    reward_deadlock_penalty=80.0,   # 低于 clip 上限100，保留区分度
    beam_width=300,
    beam_depth=1500,
    stochastic_num_rollouts=100,
    search_strategy="stochastic",   # 并行多轨迹采样
)
```

---

## 11. 参数调优原则与实践

### 11.1 调优优先级

按影响力从高到低排序：

```
① max_train_steps      ← 训练是否充分的根本
② gamma                ← 长程/短程奖励传播的关键
③ lr + entropy_coef    ← 探索-利用平衡的调节器
④ steps_per_epoch      ← 样本效率的核心
⑤ lambda_p / lambda_t  ← 模型容量（一旦确定勿频繁改）
⑥ reward 参数          ← 领域知识的编码，需业务理解
```

### 11.2 系统性调优流程

```
Step 1: 基线运行
  ├─ 使用 HQ 默认配置运行 1 个训练
  ├─ 记录：pool_success_rate / avg_makespan / 训练曲线
  └─ 识别主要问题（不收敛 / 过拟合 / 泛化差）

Step 2: 单变量分析
  ├─ 每次只改一个参数
  ├─ 至少 3 次独立运行取均值（RL 方差大）
  └─ 使用环境变量覆盖，避免修改代码

Step 3: 判断训练状态
  ├─ pool_success_rate 长期为 0 → 训练不收敛（问题①②③）
  ├─ 训练集成功但评估集失败 → 泛化不足（问题④⑤）
  └─ 偶尔成功但 makespan 差 → 探索不够（问题③⑥）

Step 4: 针对性调整（见下表）
```

### 11.3 问题-参数对应表

| 症状 | 可能原因 | 调整参数 | 调整方向 |
|------|---------|---------|---------|
| 始终无法到达目标 | gamma 太小 | `gamma` | ↑ 0.99 → 0.999 |
| 始终无法到达目标 | 探索不足 | `entropy_coef_start` | ↑ 0.15 → 0.25 |
| 始终无法到达目标 | 步数不够 | `max_train_steps` | × 2 |
| 到达目标但 makespan 差 | 利用不足 | `temperature_end` | ↓ 1.1 → 1.0 |
| 到达目标但 makespan 差 | 搜索深度不够 | `beam_depth/width` | ↑ 800→1500 |
| 训练集好但新网络差 | 泛化不足 | 增加训练网络多样性 | 加入更多网络文件 |
| 训练集好但新网络差 | 过拟合 | `entropy_coef_end` | ↑ 0.015 → 0.03 |
| loss 震荡不收敛 | lr 过大 | `lr` | ÷ 3 |
| 更新频繁被 KL 截断 | target_kl 过小 | `target_kl` | ↑ 0.09 → 0.15 |
| 策略陷入循环 | repeat penalty 不足 | `reward_repeat_penalty` | ↑ 1.5 → 3.0 |
| 预热阶段无进展 | 课程设置不合理 | `curriculum_warmup_ratio` | ↑ 0.3 → 0.5 |

### 11.4 学习率调优建议

```
Adam 优化器的学习率经验规律：

  热启动（IL预训练）：lr = 1e-5 ~ 5e-5（保留先验知识）
  冷启动（从随机）：  lr = 1e-4 ~ 3e-4（HQ默认 3e-4）
  大批量调整：       lr_new = lr_base × sqrt(batch/128)
  
如发现梯度爆炸（loss突然很大）：
  1. 检查 grad_norm（已启用 clip_grad_norm=0.5）
  2. 将 lr 降低 3-5 倍
  3. 检查 reward 幅度是否过大
```

### 11.5 训练步数估算公式

```python
def estimate_steps(num_envs, max_P, max_T, max_constrained):
    """按 HQ 公式估算合理的训练步数"""
    complexity = max(max_P, max_T)
    base = 10000 * num_envs
    extra = (complexity * 2000 + max_constrained * 3000) * num_envs
    return min(307200, max(50000, base + extra))

# 示例：4个环境，max_P=30, max_T=40, max_constrained=5
estimate_steps(4, 30, 40, 5)
# → min(307200, max(50000, 40000 + (40×2000 + 5×3000)×4))
# → min(307200, max(50000, 40000 + 380000))
# → min(307200, 420000) = 307200  ← 命中上限
```

---

## 12. 常见问题诊断

### Q1：多网训练后贪婪策略在新网络上成功率为 0

**诊断清单**：
1. ✅ 确认已应用 P0 修复（`switch_environment` 不使用缓存）
2. 检查训练日志中 `pool_success_rate` 是否在训练过程中有增长
3. 检查新网络的库所数/变迁数是否与训练集差异过大（结构分布偏移）
4. 尝试增加 `max_train_steps` 和 `entropy_coef_start`

### Q2：训练过程中 pool_success_rate 振荡剧烈

**可能原因**：
- `target_kl` 过小，PPO 更新被频繁截断，每个 epoch 学习量不稳定
- `curriculum_warmup_ratio` 设置不当，预热期过短导致策略未充分适应简单环境

**处理方式**：
```bash
export GCN_PPO_HQ_TARGET_KL=0.12       # 放宽 KL 约束
export GCN_PPO_HQ_CURRICULUM_WARMUP_RATIO=0.4  # 延长预热
```

### Q3：显存不足 (CUDA OOM)

**优先调整顺序**：
1. 减小 `minibatch_size`（128 → 64 → 32）
2. 减小 `lambda_p`（256 → 128）
3. 减小 `steps_per_epoch`（6144 → 4096）
4. 减小 `envs_per_epoch`（降低单 epoch 缓存的状态数量）

### Q4：训练速度极慢

**检查项**：
- `device` 是否正确使用 GPU：`torch.cuda.is_available()` 应为 True
- `use_deadlock_controller=True` 时首轮计算慢属正常（缓存预热）
- `mask_cache_limit` 是否设置充足（过小导致缓存频繁淘汰）
- `extra_p2t_rounds` 是否过大（每轮约增加 20% 前向时间）

### Q5：KL 散度始终为 0，loss 不下降

**原因**：P1 修复后 logprob 使用 T=1.0，而旧版可能记录了带温度的 logprob。  
**处理**：加载旧 checkpoint 后应重新进行至少 1 个 epoch 的完整训练，让 buffer 中的旧 logprob 被新数据替换。

---

## 附录：参数默认值对照表（基类 vs HQ）

| 参数 | `PetriNetGCNPPOPro`（基类）| `PetriNetGCNPPOProHQ`（HQ）| 变化说明 |
|------|--------------------------|---------------------------|---------|
| `lambda_p` | 256 | 256 | 相同 |
| `lambda_t` | 64 | 64 | 相同 |
| `extra_p2t_rounds` | 2 | **5** | ↑ 增强远程依赖捕获 |
| `max_train_steps` | 150000 | **动态计算** | 自适应网络复杂度 |
| `steps_per_epoch` | 4096 | **6144** | ↑ 更充分的样本收集 |
| `ppo_epochs` | 10 | **4** | ↓ 多网时减少过拟合 |
| `lr` | 1e-4 | **3e-4** | ↑ 更快的初期学习 |
| `gamma` | 0.99 | **0.999** | ↑ 长序列奖励传播 |
| `target_kl` | 0.04 | **0.09** | ↑ 多网需要更大更新 |
| `entropy_coef_start` | 0.09 | **0.15** | ↑ 增强初期探索 |
| `entropy_coef_end` | 0.01 | **0.015** | ↑ 保留少量持续探索 |
| `reward_goal_bonus` | 300.0 | **1000.0** | ↑ 强化目标到达信号 |
| `reward_deadlock_penalty` | 100.0 | **2000.0** | ↑ 强烈阻止死锁 |
| `reward_repeat_penalty` | 0.2 | **1.5** | ↑ 防止状态循环 |
| `envs_per_epoch` | 0 | **2** | 控制每epoch环境数 |
| `curriculum_epochs` | 8 | **4** | ↓ 更快进入正式训练 |
| `pool_eval_interval` | 4 | 4 | 相同 |
