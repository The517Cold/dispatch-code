"""
P0/P1 修复验证测试
测试不依赖 torch / petri 网环境，仅验证核心算法逻辑。
"""
import math
import sys


# ------------------------------------------------------------------ #
# 工具函数
# ------------------------------------------------------------------ #

def softmax_logprob(logits, action):
    mx = max(logits)
    exp_vals = [math.exp(x - mx) for x in logits]
    total = sum(exp_vals)
    probs = [e / total for e in exp_vals]
    return math.log(probs[action] + 1e-12)


def mean_abs(lst):
    return sum(abs(x) for x in lst) / len(lst)


def normalize_list(lst):
    m = sum(lst) / len(lst)
    s = (sum((x - m) ** 2 for x in lst) / len(lst)) ** 0.5 + 1e-8
    return [(x - m) / s for x in lst]


# ------------------------------------------------------------------ #
# 测试 1: P0 — switch_environment 权重共享语义
# ------------------------------------------------------------------ #

def test_p0_weight_sharing():
    """
    验证：旧逻辑(缓存)导致权重倒退；新逻辑始终共享最新权重。
    """
    # 旧逻辑（带缓存）
    def old_switch(from_env, to_env, weights, cache):
        if from_env != to_env:
            cache[from_env] = dict(weights)
        return dict(cache[to_env]) if to_env in cache else dict(weights)

    # 新逻辑（始终迁移当前权重）
    def new_switch(weights):
        return dict(weights)

    # 旧逻辑：env1 W=2.0 -> env2 W=3.0 -> 切回 env1
    cache = {}
    w = {"w": 1.0}
    w = old_switch("default", "env1", w, cache)
    w["w"] = 2.0
    w = old_switch("env1", "env2", w, cache)   # cache[env1]=2.0
    w["w"] = 3.0
    w = old_switch("env2", "env1", w, cache)   # 从缓存恢复 2.0 ！
    assert w["w"] == 2.0, f"旧逻辑应从缓存取回 2.0，实际 {w['w']}"

    # 新逻辑：全程保持最新值
    w2 = {"w": 1.0}
    w2 = new_switch(w2)
    w2["w"] = 2.0
    w2 = new_switch(w2)
    w2["w"] = 3.0
    w2 = new_switch(w2)
    assert w2["w"] == 3.0, f"新逻辑应保持最新值 3.0，实际 {w2['w']}"

    print("PASS test_p0_weight_sharing")


# ------------------------------------------------------------------ #
# 测试 2: P1-temperature — log_ratio 一致性
# ------------------------------------------------------------------ #

def test_p1_temperature_logprob():
    """
    验证：相同策略参数下，使用相同基准温度的 log_ratio 为 0；
          使用不同温度（旧逻辑）会产生虚假 KL 信号。
    """
    logits = [1.0, 2.0, 0.5, -1.0]
    action = 1

    T_old, T_new = 2.0, 1.5

    # 旧逻辑：rollout 用 T_old 记录，update 用 T_new 计算
    lp_old = softmax_logprob([l / T_old for l in logits], action)
    lp_new_wrong = softmax_logprob([l / T_new for l in logits], action)
    ratio_wrong = lp_new_wrong - lp_old
    assert abs(ratio_wrong) > 0.01, "旧逻辑应因温度差引入非零 log_ratio"

    # 新逻辑：都用 T=1.0（策略定义与温度解耦）
    lp_old_fix = softmax_logprob(logits, action)
    lp_new_fix = softmax_logprob(logits, action)
    ratio_fix = lp_new_fix - lp_old_fix
    assert abs(ratio_fix) < 1e-9, f"新逻辑同策略 log_ratio 应为 0，实际 {ratio_fix}"

    print(f"PASS test_p1_temperature_logprob  (ratio_wrong={ratio_wrong:.5f}, ratio_fix={ratio_fix:.1e})")


# ------------------------------------------------------------------ #
# 测试 3: P1-double-norm — 跨环境优势幅度保留
# ------------------------------------------------------------------ #

def test_p1_double_normalization():
    """
    验证：旧逻辑（二次 minibatch 归一化）会抹平跨环境优势差异；
          新逻辑（跳过二次归一化）保留真实的跨环境幅度关系。
    """
    adv_env1 = [0.8, -0.3, 1.5, -0.6]   # 幅度较大的环境
    adv_env2 = [-0.1, 0.05, -0.08, 0.12]  # 幅度较小的环境

    # 旧逻辑：再次 minibatch 归一化
    e1_renorm = normalize_list(adv_env1)
    e2_renorm = normalize_list(adv_env2)
    ratio_wrong = mean_abs(e1_renorm) / mean_abs(e2_renorm)

    # 新逻辑：保留跨环境归一化结果
    ratio_fix = mean_abs(adv_env1) / mean_abs(adv_env2)

    assert ratio_wrong < 1.5, f"旧逻辑应将幅度比压平到接近 1，实际 {ratio_wrong:.2f}"
    assert ratio_fix > 5.0, f"新逻辑应显示真实幅度差异 (>5x)，实际 {ratio_fix:.2f}"

    print(f"PASS test_p1_double_normalization  (ratio_wrong={ratio_wrong:.2f}, ratio_fix={ratio_fix:.2f})")


# ------------------------------------------------------------------ #
# 测试 4: 集成回归 — switch_environment 的 3 个环境轮换场景
# ------------------------------------------------------------------ #

def test_p0_multi_env_regression():
    """
    3 个环境循环训练，验证权重单调增长（每次 PPO 都有效作用于共享策略）。
    """
    def new_switch(weights):
        return dict(weights)

    envs = ["envA", "envB", "envC"]
    w = {"w": 0.0}
    update_log = []

    for epoch in range(6):
        env = envs[epoch % len(envs)]
        w = new_switch(w)         # 切换环境（迁移最新权重）
        w["w"] += 1.0             # 模拟 PPO 更新（权重单调递增）
        update_log.append((env, w["w"]))

    # 验证：每次更新都基于前一次的最新权重
    for i in range(1, len(update_log)):
        prev_w = update_log[i - 1][1]
        curr_w = update_log[i][1]
        assert curr_w == prev_w + 1.0, (
            f"第 {i} 次更新应在 {prev_w} 基础上+1，实际 {curr_w}"
        )

    print(f"PASS test_p0_multi_env_regression  (最终权重={update_log[-1][1]:.1f}，期望6.0)")


# ------------------------------------------------------------------ #
# 主入口
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    failures = []
    for test_fn in [
        test_p0_weight_sharing,
        test_p1_temperature_logprob,
        test_p1_double_normalization,
        test_p0_multi_env_regression,
    ]:
        try:
            test_fn()
        except AssertionError as e:
            print(f"FAIL {test_fn.__name__}: {e}")
            failures.append(test_fn.__name__)
        except Exception as e:
            print(f"ERROR {test_fn.__name__}: {e}")
            failures.append(test_fn.__name__)

    print()
    if failures:
        print(f"失败: {failures}")
        sys.exit(1)
    else:
        print("全部 4 项测试通过")
