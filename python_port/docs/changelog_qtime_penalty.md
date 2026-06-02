# 代码变更记录：添加 q-time 惩罚机制

## 变更日期
2026-04-11

## 变更概述
在 PPO 训练流程中新增 q-time（资源库所最大停留时间）约束感知能力，当 Token 在资源库所中的停留时间超过 q-time 阈值时，对策略施加惩罚，引导模型学习避免 q-time 违规。

---

## 变更文件清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `petri_net_io/utils/net_loader.py` | 修改 + 新增 | 添加 `qtime_places`/`qtime` 到 context；新增 `build_ttpn_by_token_with_res_time()` 函数 |
| `petri_gcn_ppo_4_1.py` | 修改 | 新增 `reward_qtime_penalty_coeff` 参数；在 `_step_env` 中添加 q-time 惩罚计算 |
| `train_ppo_3.py` | 修改 | 导入新函数；添加 `GCN_PPO_HQ_USE_BY_TOKEN` 环境变量控制；传递新参数 |

---

## 详细变更内容

### 1. `petri_net_io/utils/net_loader.py`

#### 1.1 新增导入

```python
# 新增 TTPPNByTokenWithResTime 的导入
from petri_net_platform.petri_net import TTPPNHasResidenceTime, TTPPNByTokenWithResTime
```

#### 1.2 `load_petri_net_context()` 返回值扩展

在返回的 context 字典中新增两个键：

| 键名 | 类型 | 来源 | 默认值 |
|------|------|------|--------|
| `qtime_places` | `List[bool]` | `sets["qtimePlaces"]` | `[False] * len(p_info)` |
| `qtime` | `int` | `values["qtime"]` | `2**31-1`（无约束） |

#### 1.3 新增函数 `build_ttpn_by_token_with_res_time(context)`

```python
def build_ttpn_by_token_with_res_time(context):
    capacity = context["capacity"]
    if capacity is None:
        capacity = [2 ** 31 - 1] * len(context["p_info"])
    return TTPPNByTokenWithResTime(
        context["p_info"],
        context["pre"],
        context["post"],
        context["min_delay_p"],
        context["min_delay_t"],
        capacity,
        context["max_residence_time"],
        context["is_resource"],
        context["place_from_places"],
        context["qtime_places"],
        context["qtime"],
    )
```

**设计说明**：
- 使用 `TTPPNByTokenWithResTime` 替代 `TTPPNHasResidenceTime`
- `TTPPNByTokenWithResTime` 内部使用 `TTPPNMarkingByTokenWithResTime`（支持 `qtime_map`）
- 当 `capacity` 为 `None` 时，填充为无限制值，避免 `_certify_enable` 报错

---

### 2. `petri_gcn_ppo_4_1.py`

#### 2.1 新增参数

```python
reward_qtime_penalty_coeff: float = 0.5  # q-time超限惩罚系数
```

#### 2.2 `_step_env()` 中新增 q-time 惩罚计算

```python
qtime_penalty = 0.0
if bool(getattr(next_marking, "over_max_residence_time", False)):
    petri_net = self.petri_net
    if hasattr(petri_net, "qtime") and hasattr(petri_net, "qtime_places"):
        qtime_val = petri_net.qtime
        if qtime_val < 2 ** 31 - 1:
            qtime_map = getattr(next_marking, "qtime_map", {})
            max_excess = 0.0
            for place_idx, is_qt in enumerate(petri_net.qtime_places):
                if not is_qt:
                    continue
                for token in next_marking.t_info[place_idx]:
                    tid = token.get_id()
                    if tid in qtime_map:
                        elapsed = next_marking.get_prefix() - qtime_map[tid]
                        excess = float(elapsed - qtime_val)
                        if excess > 0:
                            max_excess = max(max_excess, excess)
            if max_excess > 0:
                qtime_penalty = self.reward_qtime_penalty_coeff * (max_excess / self.reward_time_scale) * self.reward_residence_penalty_max

reward = step_reward + residence_reward - mobility_penalty - qtime_penalty + goal_bonus
```

**惩罚计算公式**：

```
qtime_penalty = reward_qtime_penalty_coeff × (max_excess / reward_time_scale) × reward_residence_penalty_max
```

其中：
- `max_excess` = 所有 q-time 库所中 Token 的最大超限时间（`elapsed - qtime`）
- `reward_time_scale` = 时间缩放因子，确保惩罚与其他奖励项量级一致
- `reward_residence_penalty_max` = 驻留时间最大惩罚值，作为缩放基准

**兼容性设计**：
- 使用 `hasattr` / `getattr` 安全检查，原有 `TTPPNHasResidenceTime` 模式下 `qtime_penalty = 0.0`
- 仅当 `qtime_val < 2**31-1`（有实际 q-time 约束）时才计算惩罚

---

### 3. `train_ppo_3.py`

#### 3.1 导入扩展

```python
from petri_net_io.utils.net_loader import load_petri_net_context, build_ttpn_with_residence, build_ttpn_by_token_with_res_time
```

#### 3.2 环境变量控制

新增 `GCN_PPO_HQ_USE_BY_TOKEN` 环境变量：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `GCN_PPO_HQ_USE_BY_TOKEN` | `"0"` | 设为 `"1"` 启用 ByToken 模式（含 q-time 支持） |

```python
use_by_token = os.environ.get("GCN_PPO_HQ_USE_BY_TOKEN", "0").strip() == "1"
if use_by_token:
    petri_net = build_ttpn_by_token_with_res_time(context)
else:
    petri_net = build_ttpn_with_residence(context)
```

#### 3.3 参数传递

```python
"reward_qtime_penalty_coeff": _env_float("GCN_PPO_HQ_QTIME_PENALTY_COEFF", 0.5),
```

---

## 兼容性分析

| 场景 | 行为 | 影响 |
|------|------|------|
| 默认配置（`USE_BY_TOKEN=0`） | 使用 `TTPPNHasResidenceTime`，`qtime_penalty=0` | ✅ 完全兼容，无任何影响 |
| 启用 ByToken（`USE_BY_TOKEN=1`），无 q-time 约束 | `qtime=2**31-1`，惩罚不触发 | ✅ 仅切换 Petri 网实现 |
| 启用 ByToken（`USE_BY_TOKEN=1`），有 q-time 约束 | 触发 q-time 惩罚 | ✅ 新功能生效 |

---

## 测试结果

### 测试 1：语法验证

| 文件 | 结果 |
|------|------|
| `net_loader.py` | ✅ 通过 |
| `petri_gcn_ppo_4_1.py` | ✅ 通过 |
| `train_ppo_3.py` | ✅ 通过 |

### 测试 2：导入验证

```
from petri_net_io.utils.net_loader import load_petri_net_context, build_ttpn_with_residence, build_ttpn_by_token_with_res_time
→ Import OK ✅
```

### 测试 3：context 扩展验证

```
ctx = load_petri_net_context("1-1-1.txt")
ctx["qtime_places"] → [False, False, ...] (27个False) ✅
ctx["qtime"] → 2147483647 ✅
```

### 测试 4：新函数构建验证

```
pn_new = build_ttpn_by_token_with_res_time(ctx)
pn_new.get_trans_count() → 27 ✅
hasattr(pn_new, "qtime") → True ✅
hasattr(pn_new, "qtime_places") → True ✅
```

### 测试 5：原有函数兼容性验证

```
pn_old = build_ttpn_with_residence(ctx)
pn_old.get_trans_count() → 27 ✅
hasattr(pn_old, "qtime") → False ✅ (原有模式不触发 q-time 惩罚)
```

### 测试 6：变迁触发验证

```
pn = build_ttpn_by_token_with_res_time(ctx)
pn.launch(2) → OK ✅
pn.curr.prefix → 0 ✅
pn.curr.over_max_residence_time → False ✅
pn.curr.qtime_map → {} ✅
```

---

## 使用方法

### 启用 q-time 惩罚训练

```bash
# 设置环境变量启用 ByToken 模式
set GCN_PPO_HQ_USE_BY_TOKEN=1
set GCN_PPO_HQ_QTIME_PENALTY_COEFF=0.5

# 运行训练
python train_ppo_3.py
```

### 保持原有训练方式

```bash
# 不设置任何新环境变量，行为与修改前完全一致
python train_ppo_3.py
```

---

## 后续优化建议

1. **q-time 特征编码**：将 q-time 约束信息（`qtime_places`、`qtime`）加入 GCN 的输入特征，让模型在决策时能感知 q-time 约束
2. **渐进惩罚**：当前使用线性惩罚，可考虑指数惩罚使模型更强烈地避免 q-time 违规
3. **q-time 专用评估指标**：在训练日志中增加 q-time 违规率的统计
4. **单元测试**：为 `build_ttpn_by_token_with_res_time` 和 q-time 惩罚计算编写独立单元测试
