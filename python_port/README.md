# 项目说明

本项目当前实现的是一套面向 Petri 网调度问题的实验框架，主链路包括：

1. 网文件解析与上下文构建
2. 带时间与驻留时间语义的 Petri 网环境
3. 基于图表示层的 GCN 编码
4. 模仿学习、强化学习与搜索混合求解

## 目录整理

为减少主目录杂乱，当前入口与文档已按用途重新归类：

- 根目录保留常用运行入口：
  - `run_gcn_ppo_hq.py`：单网 PPO 应用 / 可选微调
  - `run_gcn_ppo_scene_train.py`：场景级 PPO 训练
  - `run_gcn_dqn_enhanced_hq.py`：单网 DQN HQ
  - `run_imitation.py`：统一的模仿学习入口分发脚本
  - `run_a_star.py`、`run_ga.py`：搜索 / 基线算法入口
  - `check_trains.py`：变迁序列可执行性检查
- 模仿学习专项入口集中到 `entrypoints/imitation/`
  - `run_bc_pretrain.py`
  - `run_bc_finetune.py`
  - `run_dagger_pretrain.py`
  - `run_bc_transfer_eval.py`
  - `run_bc_transfer_adapt.py`
- 参考文档集中到 `docs/reference/`
- 运行结果与日志统一写入 `results/`

其中，`run_imitation.py` 通过环境变量 `IL_MODE` 分发：

- `bc_pretrain`
- `bc_finetune`
- `dagger_pretrain`
- `transfer_eval`
- `transfer_adapt`

## 图表示层与死锁控制

### 1. 基础表示层精简

在接入死锁控制第三层之前，先对图表示层做了收缩，避免在当前网文件建模方式下保留明显冗余维度。

已完成调整：

- 删除 place 特征中的 `goal` 与 `goal-token`
- 新增轻量目标约束标记 `has_goal_constraint`
- 删除 transition 中可由 `total_pre / total_post` 直接恢复的冗余差值特征
- `capacity` 改为按输入网文件是否存在容量约束动态启用
- `resource` 已从当前 GNN 表示层输入链路移除，不再作为图表示层特征

### 2. 死锁控制第一层

第一层已经完成并接入 RL 主流程，职责是：

- 基于真实环境 `enable()` 语义得到当前可执行动作集合
- 对“一步后立即死锁”的动作做硬过滤
- 对“一步后立即驻留时间超限”的动作做硬过滤
- 输出动作级原因与基础 `FBM candidate` 标记

### 3. 死锁控制第二层

第二层已经完成并接入 RL 主流程，职责是：

- 对第一层保留下来的动作执行有限深度活性检查
- 识别“局部看起来可行、几步后大概率锁死”的动作
- 将其标记为 `soft_risk`
- 当第二层过滤过严时，回退到第一层安全动作集合，避免第一版控制器过度误杀

### 4. 死锁控制第三层

第三层已经完成，以“轻量控制器感知特征”的形式接入图表示层。

当前追加到 transition 的控制器感知特征包括：

- `controller_allowed`
- `hard_blocked`
- `soft_risk`
- `safe_ratio`
- `fbm_candidate`

原则是：

- 前两层控制器保留硬约束裁决权
- 第三层只感知控制器输出，不推翻控制器 mask
- GNN 只在安全动作集合内做排序优化

## 模仿学习模块

### 1. BC 基线

BC 仍然保留，作为当前模仿学习基线。

入口：

- `entrypoints/imitation/run_bc_pretrain.py`
- `entrypoints/imitation/run_bc_finetune.py`

BC 当前本质上是：

- 用 A* 生成专家轨迹样本
- 对 actor 网络做带 mask 的监督分类
- 在同一 scene 内执行多轮 `scene_rounds` 循环训练，而不是只顺序训练一遍
- 每轮结束后按整场景成功率 / 成功样本 makespan / 成功样本步数评估并保存最佳 checkpoint
- 输出 `bc_scene_<scene_id>.pt` 或单网 `bc_<net>.pt`

### 2. DAgger-lite

已新增 DAgger-lite 场景级预训练链路。

入口：

- `entrypoints/imitation/run_dagger_pretrain.py`

当前实现方式：

1. 先为每个网生成一批初始专家样本，作为 seed dataset
2. 可选从已有 BC scene/shared checkpoint 热启动
3. 按 `scene_rounds` 在同一 scene 内多轮打乱网顺序循环训练
4. 让当前策略在环境中 rollout
5. 从失败轨迹尾部抽取一小批关键状态
6. 对这些状态重新调用 A* 做专家重标注
7. 将新标注样本聚合回该网自己的 buffer，再继续监督训练
8. 每轮结束后按整场景指标评估并保存最佳 checkpoint

当前 DAgger-lite 产物命名为：

- checkpoint：`checkpoints/dagger_scene_<scene_id>.pt`
- result：`results/dagger_scene_<scene_id>_result.txt`
- progress：`results/dagger_scene_<scene_id>_progress.txt`

### 3. 入口脚本内环境变量覆盖

当前常用入口已经支持在脚本内部直接覆盖环境变量，不必每次都在终端单独设置。

使用方式：

- 在入口脚本顶部找到 `INLINE_ENV_OVERRIDES`
- 直接填写需要覆盖的键值
- `INLINE_ENV_OVERRIDE_PRIORITY="code"` 表示代码优先
- 若改成 `INLINE_ENV_OVERRIDE_PRIORITY="terminal"`，则终端环境变量优先

目前已接入这套机制的常用入口包括：

- `run_imitation.py`
- `entrypoints/imitation/run_bc_pretrain.py`
- `entrypoints/imitation/run_bc_finetune.py`
- `entrypoints/imitation/run_dagger_pretrain.py`
- `run_gcn_ppo_scene_train.py`
- `run_gcn_ppo_hq.py`
- `run_gcn_dqn_enhanced_hq.py`

### 4. PPO 与 IL 的衔接

PPO 现在不再只依赖 BC 热启动，而是支持通用 IL 热启动。

新增环境变量：

- `GCN_PPO_SCENE_IL_MODE=auto|bc|dagger`
- `GCN_PPO_HQ_IL_MODE=auto|bc|dagger`

约定如下：

- `auto`：优先尝试 DAgger checkpoint / result，找不到再回退到 BC
- `bc`：只使用 BC
- `dagger`：只使用 DAgger

兼容性说明：

- 历史变量 `GCN_PPO_SCENE_USE_BC_WARM_START`、`GCN_PPO_HQ_USE_BC_WARM_START` 仍保留
- 当这些旧开关为 `0` 时，会禁用全部 IL 热启动

## PPO 训练与应用流程

### 1. 推荐流程

当前推荐的 PPO 使用流程如下：

1. 先做模仿学习预训练
   - 可以选择 BC
   - 也可以选择 DAgger-lite
2. 再执行场景级 PPO 训练，得到 `ppo_scene_<scene_id>.pt`
3. 应用阶段输入单网文件时，默认直接加载 `ppo_scene_<scene_id>.pt` 推理
4. 如目标网与场景平均模式差异较大，可再开启单网微调作为可选增强

### 2. 场景级 PPO

入口：

- `run_gcn_ppo_scene_train.py`

当前行为：

- 只在训练开始时加载一次 IL checkpoint
- 在同一 scene 的多个网文件上做多轮循环训练
- 每轮按固定随机种子打乱网顺序
- 每轮结束后对整个 scene 做统一评估
- 按 scene 级指标保存最佳 `ppo_scene_<scene_id>.pt`

当前默认训练配置：

- `scene_rounds = 3`
- `train_iterations_per_net = 8`
- `extra_train_iterations_per_net = 4`

### 3. 单网 PPO

入口：

- `run_gcn_ppo_hq.py`

当前默认行为：

1. 优先查找并加载 `ppo_scene_<scene_id>.pt`
2. 若未找到 scene PPO checkpoint，则退回到 IL 热启动
3. IL 热启动可按 `auto / bc / dagger` 三种模式解析
4. 只有显式开启时，才会在 scene PPO 基础上再做少量单网微调

## PPO 变体

当前仓库中的 PPO 已支持两套可并行对照的实现：

1. `enhanced`
- 核心文件：`petri_net_platform/search/petri_net_gcn_ppo.py`
- 作用定位：当前工程主线 PPO
- 特点：
  - 直接复用当前 deadlock controller 与第三层控制器感知表示
  - 保持当前单网 PPO 与 scene 外层编排逻辑不变

2. `classic`
- 核心文件：`petri_net_platform/search/petri_net_gcn_ppo_classic.py`
- 作用定位：从 `GCN-DRL` 迁移进来的旧版风格 PPO 训练壳
- 当前迁移范围：
  - 保留当前框架的图表示层、环境语义、deadlock controller、scene 级外层编排
  - 迁移 `RolloutBuffer`
  - 迁移按 `steps_per_epoch` 收集样本
  - 迁移 `mini-batch + multi-epoch` PPO 更新
  - 迁移可调的熵系数 / 温度调度
- 当前未迁移内容：
  - `env_pool` 内嵌训练组织
  - `beam search` 推理
  - 旧版动作掩码与旧版死锁语义

切换方式：

- 场景级 PPO：`GCN_PPO_SCENE_VARIANT=enhanced|classic`
- 单网 PPO：`GCN_PPO_HQ_VARIANT=enhanced|classic`

checkpoint 命名：

- `enhanced`：`checkpoints/ppo_scene_<scene_id>.pt`
- `classic`：`checkpoints/ppo_classic_scene_<scene_id>.pt`

单网结果文件命名：

- `enhanced`：`results/gcn_ppo_hq_result.txt`
- `classic`：`results/gcn_ppo_classic_hq_result.txt`

## 场景级 PPO checkpoint 排序规则

场景级 PPO 当前不再使用 `goal_distance` 作为 checkpoint 排序指标。

原因：

- 当前目标标识只约束极少数库所
- `goal_distance` 只反映这些少数库所上的 token 差异
- 作为 scene 级模型优劣标准时，参考价值不稳定

因此当前 best checkpoint 按以下顺序排序：

1. 场景平均成功率，越高越好
2. 成功样本平均 makespan，越低越好
3. 成功样本平均变迁步数，越低越好

注意：

- `makespan` 与步数只在成功样本上统计
- 如果一轮没有任何成功样本，则只比较成功率

## 模仿学习未来优化方向

以下 4 条作为后续模仿学习升级路线，现阶段已明确记录：

1. `DAgger-lite` 已落地，后续可继续扩大关键状态选择策略，并增强查询预算分配
2. 引入软标签 / 代价敏感模仿学习，避免当前 one-hot 专家标签过硬
3. 评估 `GAIL / AIRL` 这类对抗式模仿学习是否适合本项目
4. 评估 `AWAC / SQIL` 一类偏离线强化学习风格的模仿学习方案

## 其他后续优化方向

1. 为死锁控制器加入训练 / 推理分层启用模式，支持轻量、平衡、强安全三档
2. 继续增强第二层局部活性检查，使其更接近真正的 FBM 边界识别
3. 为第三层表示层增加更稳妥的辅助任务，但不改变控制器硬约束地位
4. 继续完善场景级 PPO 的 scene 内评估统计，例如死锁类别分布与跨轮退化情况

## 文档维护约定

从本次开始，以下改动应持续记录到本 README 中：

- 模仿学习链路的变更
- 死锁控制各层的新策略
- 图表示层输入特征调整
- PPO 场景训练 / 单网应用流程变化
- 入口目录结构与默认输出路径变化
