# 框架操作文档

本文档按当前主线整理：

1. IL 预训练
2. 场景级 RL 训练
3. 单网应用 / 单网微调

同时给出：

- 单场景多网训练应依次运行哪些文件
- 训练完成后如何跑单网
- 各模块核心参数在哪里设置

---

## 1. 参数修改方式

当前入口脚本统一支持两种改参方式：

1. 终端环境变量
2. 入口脚本顶部的 `INLINE_ENV_OVERRIDES`

推荐优先使用入口脚本顶部的 `INLINE_ENV_OVERRIDES`，因为：

- 不需要每次打开终端重新设置
- 当前脚本大多使用 `INLINE_ENV_OVERRIDE_PRIORITY = "code"`
- 这意味着脚本内设置默认会覆盖终端同名环境变量

公共实现位置：

- `entrypoint_env.py`

---

## 2. 入口文件总览

### 2.1 模仿学习入口

- 统一入口：`run_imitation.py`
- BC 场景预训练：`entrypoints/imitation/run_bc_pretrain.py`
- BC 单网微调：`entrypoints/imitation/run_bc_finetune.py`
- DAgger-lite 场景预训练：`entrypoints/imitation/run_dagger_pretrain.py`

### 2.2 PPO 入口

- 场景级 PPO 训练：`run_gcn_ppo_scene_train.py`
- 单网 PPO 应用 / 可选微调：`run_gcn_ppo_hq.py`

### 2.3 DQN 入口

- 单网 DQN HQ：`run_gcn_dqn_enhanced_hq.py`

说明：

- 当前 PPO 已支持两套变体：
  - `enhanced`
  - `classic`
- 当前 DQN 仍然是单网训练 / 推理入口，没有独立的场景级多网训练脚本

---

## 3. 主线流程

## 3.1 推荐主线：DAgger 场景预训练 -> PPO 场景训练 -> 单网应用

这是当前最推荐的主线。

执行顺序：

1. `entrypoints/imitation/run_dagger_pretrain.py`
2. `run_gcn_ppo_scene_train.py`
3. `run_gcn_ppo_hq.py`

适用场景：

- 先在某个 `scene_id` 下对 `resources/` 中同场景的多个网做预训练
- 再得到该场景对应的 PPO 场景模型
- 最后对单个目标网做应用或少量微调

---

## 3.2 基线主线：BC 场景预训练 -> PPO 场景训练 -> 单网应用

执行顺序：

1. `entrypoints/imitation/run_bc_pretrain.py`
2. `run_gcn_ppo_scene_train.py`
3. `run_gcn_ppo_hq.py`

适用场景：

- 做 BC 基线
- 对比 DAgger 与 BC 的冷启动效果

---

## 3.3 完整对照主线：BC -> DAgger -> PPO -> 单网

执行顺序：

1. `entrypoints/imitation/run_bc_pretrain.py`
2. `entrypoints/imitation/run_dagger_pretrain.py`
3. `run_gcn_ppo_scene_train.py`
4. `run_gcn_ppo_hq.py`

说明：

- 第 2 步中 `DAGGER_USE_BC_INIT=1` 时，DAgger 会先加载 BC scene checkpoint 再继续训练
- 这是最稳的 IL->RL 链路

---

## 4. 单场景多网如何训练

### 4.1 BC / DAgger 的场景级训练

这两条线都从 `resources/` 读取同一 `scene_id` 下的多个网文件。

当前组织方式：

- 外层按 `scene_rounds` 多轮循环
- 每轮会打乱同场景网文件顺序
- 每轮结束后按整场景指标保存最佳 checkpoint

也就是说，训练对象不是单个网，而是：

- `scene_id = 1` 对应的一批网
- `scene_id = 2` 对应的一批网
- `scene_id = 3` 对应的一批网

### 4.2 PPO 的场景级训练

PPO 场景训练也是读取 `resources/` 下同一 `scene_id` 的多个网。

当前组织方式：

- 外层 `scene_rounds`
- 每轮随机打乱同场景网顺序
- 按顺序逐网训练
- 当前网训练后的 `actor_state / critic_state` 传给下一个网
- 每轮结束后对整个 scene 做统一评估
- 按 scene 指标保存最佳 PPO scene checkpoint

---

## 5. 训练完成后如何跑单网

单网应用入口：

- `run_gcn_ppo_hq.py`

当前默认逻辑：

1. 优先按 `scene_id` 自动查找场景级 PPO checkpoint
2. 若找到，则直接推理
3. 如果显式允许，可在 scene PPO 基础上再做少量单网微调
4. 若找不到 scene PPO checkpoint，则退回到 IL 热启动

单网输入位置：

- 默认从 `test/` 目录读取
- 也可通过 `GCN_PPO_HQ_INPUT_SUBDIR` 指向其他子目录

单网应用常见操作：

1. 指定目标网文件
2. 指定 PPO 变体 `enhanced` 或 `classic`
3. 确保存在对应的 scene PPO checkpoint
4. 运行单网入口

---

## 6. 场景训练与单网应用的推荐执行顺序

### 6.1 不经过 BC，直接 DAgger + PPO

1. 打开 `entrypoints/imitation/run_dagger_pretrain.py`
2. 在 `INLINE_ENV_OVERRIDES` 中设置：

```python
INLINE_ENV_OVERRIDES = {
    "DAGGER_SCENE_ID": "1",
    "DAGGER_USE_BC_INIT": "0",
    "DAGGER_SCENE_ROUNDS": "3",
}
```

3. 运行 `entrypoints/imitation/run_dagger_pretrain.py`
4. 打开 `run_gcn_ppo_scene_train.py`
5. 在 `INLINE_ENV_OVERRIDES` 中设置：

```python
INLINE_ENV_OVERRIDES = {
    "GCN_PPO_SCENE_ID": "1",
    "GCN_PPO_SCENE_IL_MODE": "dagger",
    "GCN_PPO_SCENE_VARIANT": "enhanced",
}
```

6. 运行 `run_gcn_ppo_scene_train.py`
7. 打开 `run_gcn_ppo_hq.py`
8. 在 `INLINE_ENV_OVERRIDES` 中设置：

```python
INLINE_ENV_OVERRIDES = {
    "GCN_PPO_HQ_NET_FILE": "1-2-13.txt",
    "GCN_PPO_HQ_SCENE_ID": "1",
    "GCN_PPO_HQ_VARIANT": "enhanced",
    "GCN_PPO_HQ_USE_SCENE_POLICY": "1",
}
```

9. 运行 `run_gcn_ppo_hq.py`

### 6.2 先 BC，再 DAgger，再 PPO

1. 运行 `entrypoints/imitation/run_bc_pretrain.py`
2. 运行 `entrypoints/imitation/run_dagger_pretrain.py`
3. 运行 `run_gcn_ppo_scene_train.py`
4. 运行 `run_gcn_ppo_hq.py`

这是当前最稳的一条全链路。

---

## 7. 输出文件说明

### 7.1 BC

- checkpoint：`checkpoints/bc_scene_<scene_id>.pt`
- result：`results/bc_scene_<scene_id>_result.txt`
- progress：`results/bc_scene_<scene_id>_progress.txt`

### 7.2 DAgger

- checkpoint：`checkpoints/dagger_scene_<scene_id>.pt`
- result：`results/dagger_scene_<scene_id>_result.txt`
- progress：`results/dagger_scene_<scene_id>_progress.txt`

### 7.3 PPO scene

`enhanced`：

- checkpoint：`checkpoints/ppo_scene_<scene_id>.pt`
- result：`results/ppo_scene_<scene_id>_result.txt`
- progress：`results/ppo_scene_<scene_id>_progress.txt`

`classic`：

- checkpoint：`checkpoints/ppo_classic_scene_<scene_id>.pt`
- result：`results/ppo_classic_scene_<scene_id>_result.txt`
- progress：`results/ppo_classic_scene_<scene_id>_progress.txt`

### 7.4 单网 PPO

`enhanced`：

- result：`results/gcn_ppo_hq_result.txt`
- progress：`results/gcn_ppo_hq_progress.txt`

`classic`：

- result：`results/gcn_ppo_classic_hq_result.txt`
- progress：`results/gcn_ppo_classic_hq_progress.txt`

### 7.5 单网 DQN

- result：`results/gcn_dqn_enhanced_hq_result.txt`
- progress：`results/gcn_dqn_enhanced_hq_progress.txt`

---

## 8. 参数总表

以下按模块列出“核心可调参数”和“设置位置”。

## 8.1 Deadlock Controller

核心文件：

- `petri_net_platform/search/deadlock_controller.py`

核心参数：

- `enable_lookahead`
- `lookahead_depth`
- `lookahead_width`
- `lookahead_trigger_safe_limit`
- `lookahead_trigger_on_fbm`
- `log_path`

当前默认值位置：

- `deadlock_controller.py`
- `petri_net_platform/search/petri_net_gcn_ppo.py`
- `petri_net_platform/search/petri_net_gcn_ppo_classic.py`
- `petri_net_platform/search/petri_net_gcn_dqn_enhanced.py`

当前暴露情况：

- 第三层控制器感知图特征开关已在入口脚本暴露：
  - `GCN_PPO_SCENE_CONTROLLER_REPRESENTATION`
  - `GCN_PPO_HQ_CONTROLLER_REPRESENTATION`
  - `GCN_ENH_HQ_CONTROLLER_REPRESENTATION`
- 但第一层 / 第二层 lookahead 的深度、宽度、触发阈值目前**没有在入口脚本中单独暴露**

如果需要调整 deadlock controller 的核心策略，当前有两种方式：

1. 直接修改 `deadlock_controller.py` 默认值
2. 修改 PPO / DQN 构造器默认参数，或在入口脚本中补传这些参数

---

## 8.2 BC 场景预训练

入口：

- `entrypoints/imitation/run_bc_pretrain.py`

推荐在该文件顶部 `INLINE_ENV_OVERRIDES` 中设置。

核心参数：

- `BC_SCENE_ID`
- `BC_SCENE_ROUNDS`
- `BC_PRETRAIN_EPOCHS`
- `BC_PRETRAIN_ROLLOUT_EVERY`
- `BC_PRETRAIN_NET_LIMIT`
- `BC_DEVICE`
- `BC_SEED`

专家数据生成相关参数：

- `BC_MAX_EXPAND_NODES`
- `BC_MAX_SEARCH_SECONDS`
- `BC_MAX_DATA_GEN_SECONDS`
- `BC_PERTURB_COUNT`
- `BC_PERTURB_STEPS`
- `BC_CLEAN_REPEAT`
- `BC_ALLOW_GENERATE_EFLINE`
- `BC_EFLINE_MAX_EXPAND_NODES`
- `BC_EFLINE_MAX_SEARCH_SECONDS`

这些参数控制：

- A* / 数据生成预算
- 扰动样本数量
- 多轮场景训练轮数

---

## 8.3 BC 单网微调

入口：

- `entrypoints/imitation/run_bc_finetune.py`

核心参数：

- `BC_NET_FILE`
- `BC_INIT_CKPT_PATH`
- `BC_VAL_SAMPLES_PATH`
- `BC_FINETUNE_EPOCHS`
- `BC_FINETUNE_ROLLOUT_EVERY`
- `BC_DEVICE`
- `BC_SEED`

数据生成相关参数：

- `BC_MAX_EXPAND_NODES`
- `BC_MAX_SEARCH_SECONDS`
- `BC_MAX_DATA_GEN_SECONDS`
- `BC_PERTURB_COUNT`
- `BC_PERTURB_STEPS`
- `BC_CLEAN_REPEAT`
- `BC_ALLOW_GENERATE_EFLINE`
- `BC_EFLINE_MAX_EXPAND_NODES`
- `BC_EFLINE_MAX_SEARCH_SECONDS`

复用开关：

- `BC_REUSE_EXISTING_WHEN_NO_SAMPLES`

---

## 8.4 DAgger-lite 场景预训练

入口：

- `entrypoints/imitation/run_dagger_pretrain.py`

核心参数：

- `DAGGER_SCENE_ID`
- `DAGGER_USE_BC_INIT`
- `DAGGER_INIT_CKPT_PATH`
- `DAGGER_INIT_EPOCHS`
- `DAGGER_SCENE_ROUNDS`
- `DAGGER_ROUNDS`
- `DAGGER_ROUND_EPOCHS`
- `DAGGER_ROLLOUTS_PER_ROUND`
- `DAGGER_ROLLOUT_EPSILON`
- `DAGGER_QUERY_STATES_PER_ROUND`
- `DAGGER_QUERY_TAIL_STEPS`
- `DAGGER_QUERY_LABEL_HORIZON`
- `DAGGER_NET_LIMIT`
- `DAGGER_DEVICE`
- `DAGGER_SEED`

专家查询 / 数据生成预算：

- `DAGGER_MAX_EXPAND_NODES`
- `DAGGER_MAX_SEARCH_SECONDS`
- `DAGGER_MAX_DATA_GEN_SECONDS`
- `DAGGER_QUERY_MAX_EXPAND_NODES`
- `DAGGER_QUERY_MAX_SEARCH_SECONDS`

初始 seed dataset 生成参数：

- `DAGGER_INITIAL_PERTURB_COUNT`
- `DAGGER_INITIAL_PERTURB_STEPS`
- `DAGGER_CLEAN_REPEAT`
- `DAGGER_ALLOW_GENERATE_EFLINE`
- `DAGGER_EFLINE_MAX_EXPAND_NODES`
- `DAGGER_EFLINE_MAX_SEARCH_SECONDS`

这些参数控制：

- DAgger 每轮 rollout 数
- 每轮抽多少 query 状态
- 每个 query 取多长 horizon 的标签
- DAgger 从 BC 初始化还是从零开始

---

## 8.5 IL 训练器内部参数

这部分当前**不在入口脚本中直接暴露**，需要改代码。

位置：

- `imitation/pretrain.py`
- `imitation/finetune.py`
- `imitation/dagger.py`
- `imitation/trainer.py`

当前固定值：

- `batch_size=16`
- `lr=3e-4`
- `weight_decay=1e-5`
- `label_smoothing=0.0`

模型规模也当前固定在构造函数中，例如：

- `lambda_p=128`
- `lambda_t=32`
- `num_layers=4`

如果要系统性调 IL，请优先看：

- `BCTrainerConfig`
- `_fit_with_samples()`
- `_build_model()`

---

## 8.6 PPO 场景训练

入口：

- `run_gcn_ppo_scene_train.py`

核心选择参数：

- `GCN_PPO_SCENE_ID`
- `GCN_PPO_SCENE_VARIANT`
- `GCN_PPO_SCENE_IL_MODE`
- `GCN_PPO_SCENE_IL_CKPT_PATH`
- `GCN_PPO_SCENE_USE_BC_WARM_START`
- `GCN_PPO_SCENE_NET_LIMIT`
- `GCN_PPO_SCENE_SEED`

场景训练主预算：

- `GCN_PPO_SCENE_ROUNDS`
- `GCN_PPO_SCENE_TRAIN_ITERATIONS`
- `GCN_PPO_SCENE_EXTRA_TRAIN_ITERATIONS`

PPO 核心参数：

- `GCN_PPO_SCENE_ROLLOUT_EPISODES_PER_ITER`
- `GCN_PPO_SCENE_UPDATE_EPOCHS`

奖励相关：

- `GCN_PPO_SCENE_REWARD_TIME_SCALE`
- `GCN_PPO_SCENE_REWARD_CLIP_ABS`

第三层表示开关：

- `GCN_PPO_SCENE_CONTROLLER_REPRESENTATION`

日志：

- `GCN_PPO_SCENE_VERBOSE`
- `GCN_PPO_SCENE_LOG_INTERVAL`

专家步数驱动的步数预算：

- `GCN_PPO_SCENE_EXPERT_MIN_STEP_SCALE`
- `GCN_PPO_SCENE_EXPERT_MAX_STEP_SCALE`
- `GCN_PPO_SCENE_EXPERT_MIN_STEP_FLOOR`
- `GCN_PPO_SCENE_EXPERT_MAX_STEP_FLOOR`
- `GCN_PPO_SCENE_EXPERT_MAX_STEP_MIN_MARGIN`
- `GCN_PPO_SCENE_EXPERT_STEP_SCALE`
- `GCN_PPO_SCENE_EXPERT_STEP_MIN_MARGIN`

`classic` 变体专用：

- `GCN_PPO_SCENE_STEPS_PER_EPOCH`
- `GCN_PPO_SCENE_MINIBATCH_SIZE`
- `GCN_PPO_SCENE_TARGET_KL`
- `GCN_PPO_SCENE_ENTROPY_START`
- `GCN_PPO_SCENE_ENTROPY_END`
- `GCN_PPO_SCENE_TEMPERATURE_START`
- `GCN_PPO_SCENE_TEMPERATURE_END`

---

## 8.7 单网 PPO 应用 / 微调

入口：

- `run_gcn_ppo_hq.py`

输入与模式：

- `GCN_PPO_HQ_INPUT_SUBDIR`
- `GCN_PPO_HQ_NET_FILE`
- `GCN_PPO_HQ_SCENE_ID`
- `GCN_PPO_HQ_VARIANT`
- `GCN_PPO_HQ_FAST`

IL 热启动：

- `GCN_PPO_HQ_IL_MODE`
- `GCN_PPO_HQ_IL_CKPT_PATH`
- `GCN_PPO_HQ_IL_RESULT_PATH`
- `GCN_PPO_HQ_USE_BC_WARM_START`

scene PPO 复用：

- `GCN_PPO_HQ_USE_SCENE_POLICY`
- `GCN_PPO_HQ_PPO_SCENE_CKPT_PATH`
- `GCN_PPO_HQ_ENABLE_SCENE_FINETUNE`
- `GCN_PPO_HQ_SCENE_FINETUNE_SCALE`

IL 热启动预算缩放：

- `GCN_PPO_HQ_BC_WARM_NET_SCALE`
- `GCN_PPO_HQ_BC_WARM_SCENE_SCALE`
- `GCN_PPO_HQ_BC_WARM_GLOBAL_SCALE`

PPO 核心参数：

- `GCN_PPO_HQ_ROLLOUT_EPISODES_PER_ITER`
- `GCN_PPO_HQ_UPDATE_EPOCHS`
- `GCN_PPO_HQ_REWARD_TIME_SCALE`
- `GCN_PPO_HQ_REWARD_CLIP_ABS`
- `GCN_PPO_HQ_VERBOSE`
- `GCN_PPO_HQ_LOG_INTERVAL`
- `GCN_PPO_HQ_CONTROLLER_REPRESENTATION`

专家步数驱动预算：

- `GCN_PPO_HQ_EXPERT_MIN_STEP_SCALE`
- `GCN_PPO_HQ_EXPERT_MAX_STEP_SCALE`
- `GCN_PPO_HQ_EXPERT_MIN_STEP_FLOOR`
- `GCN_PPO_HQ_EXPERT_MAX_STEP_FLOOR`
- `GCN_PPO_HQ_EXPERT_MAX_STEP_MIN_MARGIN`
- `GCN_PPO_HQ_EXPERT_STEP_SCALE`
- `GCN_PPO_HQ_EXPERT_STEP_MIN_MARGIN`

`classic` 变体专用：

- `GCN_PPO_HQ_STEPS_PER_EPOCH`
- `GCN_PPO_HQ_MINIBATCH_SIZE`
- `GCN_PPO_HQ_TARGET_KL`
- `GCN_PPO_HQ_ENTROPY_START`
- `GCN_PPO_HQ_ENTROPY_END`
- `GCN_PPO_HQ_TEMPERATURE_START`
- `GCN_PPO_HQ_TEMPERATURE_END`

---

## 8.8 DQN 单网训练 / 应用

入口：

- `run_gcn_dqn_enhanced_hq.py`

说明：

- 当前 DQN 还是单网入口
- 支持 BC 热启动
- 还没有独立的场景级多网训练入口

输入与模式：

- `GCN_ENH_HQ_INPUT_SUBDIR`
- `GCN_ENH_HQ_NET_FILE`
- `GCN_ENH_HQ_SCENE_ID`
- `GCN_ENH_HQ_FAST`

BC 热启动：

- `GCN_ENH_HQ_USE_BC_WARM_START`
- `GCN_ENH_HQ_BC_CKPT_PATH`
- `GCN_ENH_HQ_BC_RESULT_PATH`
- `GCN_ENH_HQ_BC_WARM_NET_SCALE`
- `GCN_ENH_HQ_BC_WARM_SCENE_SCALE`
- `GCN_ENH_HQ_BC_WARM_GLOBAL_SCALE`
- `GCN_ENH_HQ_BC_WARM_NET_EPSILON_INIT`
- `GCN_ENH_HQ_BC_WARM_SCENE_EPSILON_INIT`
- `GCN_ENH_HQ_BC_WARM_GLOBAL_EPSILON_INIT`

训练预算与评估：

- `GCN_ENH_HQ_GOAL_EVAL_ROLLOUTS`
- `GCN_ENH_HQ_GOAL_MIN_SUCCESS`
- `GCN_ENH_HQ_EXTRA_TRAIN_EPISODES`
- `GCN_ENH_HQ_SIMILAR_FINETUNE_EPISODE_SCALE`
- `GCN_ENH_HQ_SIMILAR_FINETUNE_EXTRA_SCALE`
- `GCN_ENH_HQ_SIMILAR_FINETUNE_MIN_EPISODES`
- `GCN_ENH_HQ_SIMILAR_FINETUNE_MIN_EXTRA_EPISODES`

奖励：

- `GCN_ENH_HQ_USE_REWARD_SCALING`
- `GCN_ENH_HQ_REWARD_TIME_SCALE`
- `GCN_ENH_HQ_USE_REWARD_CLIP`
- `GCN_ENH_HQ_REWARD_CLIP_ABS`

损失：

- `GCN_ENH_HQ_USE_HUBER_LOSS`
- `GCN_ENH_HQ_HUBER_BETA`

日志与表示：

- `GCN_ENH_HQ_VERBOSE`
- `GCN_ENH_HQ_LOG_INTERVAL`
- `GCN_ENH_HQ_CONTROLLER_REPRESENTATION`

checkpoint 复用：

- `GCN_ENH_HQ_REUSE`
- `GCN_ENH_HQ_REUSE_SIMILAR`
- `GCN_ENH_HQ_FINETUNE_ON_SIMILAR`

专家步数驱动预算：

- `GCN_ENH_HQ_EXPERT_MIN_STEP_SCALE`
- `GCN_ENH_HQ_EXPERT_MAX_STEP_SCALE`
- `GCN_ENH_HQ_EXPERT_MIN_STEP_FLOOR`
- `GCN_ENH_HQ_EXPERT_MAX_STEP_FLOOR`
- `GCN_ENH_HQ_EXPERT_MAX_STEP_MIN_MARGIN`
- `GCN_ENH_HQ_EXPERT_STEP_SCALE`
- `GCN_ENH_HQ_EXPERT_STEP_MIN_MARGIN`

---

## 9. 当前最常用的几个入口建议

### 9.1 只做场景级 DAgger

- 文件：`entrypoints/imitation/run_dagger_pretrain.py`

### 9.2 做场景级 DAgger + PPO

依次运行：

1. `entrypoints/imitation/run_dagger_pretrain.py`
2. `run_gcn_ppo_scene_train.py`
3. `run_gcn_ppo_hq.py`

### 9.3 做 BC 基线

依次运行：

1. `entrypoints/imitation/run_bc_pretrain.py`
2. `run_gcn_ppo_scene_train.py`
3. `run_gcn_ppo_hq.py`

### 9.4 做 PPO `enhanced / classic` 对照

先训练两次 scene PPO：

1. `GCN_PPO_SCENE_VARIANT=enhanced`
2. `GCN_PPO_SCENE_VARIANT=classic`

再用单网入口分别跑：

1. `GCN_PPO_HQ_VARIANT=enhanced`
2. `GCN_PPO_HQ_VARIANT=classic`

---

## 10. 备注

1. 当前 deadlock controller 的核心 lookahead 参数尚未在入口脚本中完全暴露，若要细调，需要改代码。
2. 当前 IL trainer 的 `batch_size / lr / label_smoothing` 也主要在代码里固定。
3. 当前 DQN 仍以单网为主，不建议作为场景级训练主线。
4. 当前主线推荐为：`DAgger-lite -> PPO scene -> 单网应用`
