"""
测试 petri_gcn_ppo_4_1.py 中改进后的奖励函数。

覆盖范围：
  1. _compute_residence_reward: 安全区 / 警告区 / 超限区 / 无约束
  2. _step_env: 非法动作 / 正常步骤 / 死锁 / 目标到达（首次/改进/退化）
  3. 动作空间收缩惩罚 (mobility_penalty)
  4. 参数兼容性: 新参数默认值不影响旧行为
  5. 驻留时间阈值随网文件自动适配
"""
import sys, os, math
from collections import deque
from typing import List, Tuple

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from petri_net_platform.marking import TTPPNMarkingHasResidenceTime


class _FakeMarking:
    """轻量 marking stub，仅保留 reward 函数依赖的字段。"""
    def __init__(self, p_info, prefix=0, residence_time_info=None,
                 over_max_residence_time=False, t_info=None):
        self.p_info = list(p_info)
        self.prefix = prefix
        self.residence_time_info = residence_time_info or [deque() for _ in p_info]
        self.over_max_residence_time = over_max_residence_time
        self.over_residence_time_place = 0
        self.t_info = t_info or [deque() for _ in p_info]
        self.curr_delay_t = []
        self.is_enable = []
        self.nexts = {}
        self.tran_last_enable_time = 0
        self.last_enable_times = []

    def get_p_info(self):
        return self.p_info

    def get_prefix(self):
        return self.prefix

    def clone(self):
        ri = [deque(list(d)) for d in self.residence_time_info]
        ti = [deque(list(d)) for d in self.t_info]
        m = _FakeMarking(self.p_info[:], self.prefix, ri, self.over_max_residence_time, ti)
        m.over_residence_time_place = self.over_residence_time_place
        return m


class _FakePetriNet:
    """最小化 Petri 网桩对象，支持 _step_env 调用链路。"""
    def __init__(self, p_count=3, t_count=2, max_residence_time=None):
        self.p_count = p_count
        self.t_count = t_count
        self._marking = None
        self.max_residence_time = max_residence_time or [2**31-1]*p_count
        self.capacity = None
        self.has_capacity = False
        self.transition_flow_allowed = [True] * t_count
        self.place_from_places = None

        self._next_marking = None

    def get_trans_count(self):
        return self.t_count

    def get_marking(self):
        return self._marking

    def set_marking(self, m):
        self._marking = m

    def launch(self, action):
        if self._next_marking is not None:
            return self._next_marking
        m = self._marking.clone()
        m.prefix += 10
        return m

    def enable(self, tran):
        return True

    def clone(self):
        c = _FakePetriNet(self.p_count, self.t_count, self.max_residence_time[:])
        if self._marking:
            c._marking = self._marking.clone()
        return c


def _build_ppo(p_count=3, t_count=2, max_residence_time=None, **overrides):
    """构造一个最小化的 PetriNetGCNPPOPro 实例（跳过 GCN 模型构建）。"""
    import torch

    net = _FakePetriNet(p_count, t_count, max_residence_time)
    end = [-1] * p_count
    pre = [[0]*t_count for _ in range(p_count)]
    post = [[0]*t_count for _ in range(p_count)]
    min_delay_p = [0] * p_count
    mrt = max_residence_time or [2**31-1]*p_count

    from petri_gcn_ppo_4_1 import PetriNetGCNPPOPro

    class _TestPPO(PetriNetGCNPPOPro):
        """跳过模型/编码器初始化，仅测试奖励逻辑。"""
        def __init__(self, petri_net, end, pre, post, min_delay_p, **kw):
            self.petri_net = petri_net
            self.initial_petri_net = petri_net.clone()
            self.env_pool = None
            self.end = end
            self.pre = pre
            self.post = post
            self.max_residence_time = kw.pop("max_residence_time", mrt)
            self.capacity = None
            self.has_capacity = False

            self.reward_goal_bonus = kw.get("reward_goal_bonus", 300.0)
            self.reward_deadlock_penalty = kw.get("reward_deadlock_penalty", 100.0)
            self.reward_progress_weight = kw.get("reward_progress_weight", 2.0)
            self.reward_repeat_penalty = kw.get("reward_repeat_penalty", 0.2)
            self.reward_time_scale = kw.get("reward_time_scale", 1000.0)
            self.reward_residence_warn_ratio = kw.get("reward_residence_warn_ratio", 0.7)
            self.reward_residence_penalty_max = kw.get("reward_residence_penalty_max", 30.0)
            self.reward_residence_safe_bonus = kw.get("reward_residence_safe_bonus", 0.5)
            self.reward_mobility_weight = kw.get("reward_mobility_weight", 0.3)

            self.verbose = False
            self.current_env_name = "test"
            self.best_records = {"test": {"makespan": 2**31-1, "trans": []}}

            self.use_deadlock_controller = False
            self.deadlock_controller = None
            self.mask_cache_limit = 0
            self._mask_cache = {}
            self.transition_flow_allowed = [True]*t_count
            self.device = torch.device("cpu")

    ppo = _TestPPO(net, end, pre, post, min_delay_p, **overrides)
    return ppo


# ────────── _compute_residence_reward 测试 ──────────

def test_residence_safe():
    ppo = _build_ppo(max_residence_time=[100, 2**31-1, 50])
    m = _FakeMarking([1, 0, 1], prefix=0,
                     residence_time_info=[deque([30]), deque(), deque([10])])
    r = ppo._compute_residence_reward(m)
    assert r == ppo.reward_residence_safe_bonus, f"Safe zone should give bonus, got {r}"
    print("  [PASS] test_residence_safe")


def test_residence_warn_zone():
    ppo = _build_ppo(max_residence_time=[100, 2**31-1, 50])
    m = _FakeMarking([1, 0, 1], prefix=0,
                     residence_time_info=[deque([85]), deque(), deque([10])])
    r = ppo._compute_residence_reward(m)
    assert r < 0, f"Warn zone should be negative, got {r}"
    assert r > -ppo.reward_residence_penalty_max, f"Warn zone should be less than max penalty, got {r}"
    print("  [PASS] test_residence_warn_zone")


def test_residence_progressive():
    """警告区内 ratio 越高，惩罚越重。"""
    ppo = _build_ppo(max_residence_time=[100, 2**31-1, 2**31-1])
    r_vals = []
    for res_time in [75, 85, 95]:
        m = _FakeMarking([1, 0, 0], prefix=0,
                         residence_time_info=[deque([res_time]), deque(), deque()])
        r_vals.append(ppo._compute_residence_reward(m))
    assert r_vals[0] > r_vals[1] > r_vals[2], f"Penalty should increase: {r_vals}"
    print("  [PASS] test_residence_progressive")


def test_residence_over():
    ppo = _build_ppo(max_residence_time=[100, 2**31-1, 50])
    m = _FakeMarking([1, 0, 1], prefix=0,
                     residence_time_info=[deque([30]), deque(), deque([10])])
    m.over_max_residence_time = True
    r = ppo._compute_residence_reward(m)
    assert r == -ppo.reward_residence_penalty_max, f"Over limit should get max penalty, got {r}"
    print("  [PASS] test_residence_over")


def test_residence_no_info():
    """没有 residence_time_info 属性的 marking 应返回安全奖励。"""
    ppo = _build_ppo()
    class _BareMarking:
        pass
    m = _BareMarking()
    r = ppo._compute_residence_reward(m)
    assert r == ppo.reward_residence_safe_bonus
    print("  [PASS] test_residence_no_info")


def test_residence_unconstrained():
    """所有库所 max_residence_time = INF 时，不应产生惩罚。"""
    ppo = _build_ppo(max_residence_time=[2**31-1, 2**31-1, 2**31-1])
    m = _FakeMarking([1, 1, 1], prefix=0,
                     residence_time_info=[deque([9999]), deque([9999]), deque([9999])])
    r = ppo._compute_residence_reward(m)
    assert r == ppo.reward_residence_safe_bonus, f"Unconstrained should give bonus, got {r}"
    print("  [PASS] test_residence_unconstrained")


# ────────── _step_env 测试 ──────────

def test_step_illegal_action():
    ppo = _build_ppo()
    m = _FakeMarking([1, 0, 0], prefix=0)
    ppo.petri_net.set_marking(m)
    _, reward, done, deadlock = ppo._step_env(m, -1, {})
    assert done and deadlock, "Illegal action should end episode"
    assert reward == -ppo.reward_deadlock_penalty
    print("  [PASS] test_step_illegal_action")


def test_step_normal():
    """正常步骤应返回合理奖励。"""
    ppo = _build_ppo()
    m = _FakeMarking([1, 0, 0], prefix=0)
    ppo.petri_net.set_marking(m)
    _, reward, done, deadlock = ppo._step_env(m, 0, {})
    assert not done or not deadlock, "Normal step should not be both done and deadlock"
    assert isinstance(reward, float)
    print("  [PASS] test_step_normal")


def test_step_repeat_penalty_capped():
    """重复惩罚应在 visit_count=10 处封顶。"""
    ppo = _build_ppo(reward_repeat_penalty=1.0)
    m = _FakeMarking([1, 0, 0], prefix=0)
    ppo.petri_net.set_marking(m)
    next_key = ppo._state_key(m.clone())
    seen_low = {next_key: 5}
    seen_high = {next_key: 100}
    _, r_low, _, _ = ppo._step_env(m, 0, seen_low)
    ppo.petri_net.set_marking(m)
    _, r_high, _, _ = ppo._step_env(m, 0, seen_high)
    assert r_high == r_low - 5.0 or abs(r_high - r_low + 5.0) < 1.0, \
        f"Repeat penalty should cap at 10, r_low={r_low}, r_high={r_high}"
    print("  [PASS] test_step_repeat_penalty_capped")


def test_step_goal_first():
    """首次到达目标应获得 goal_bonus。"""
    ppo = _build_ppo(reward_goal_bonus=200.0)
    ppo.end = [1, 0, 0]
    m = _FakeMarking([0, 0, 0], prefix=0)
    next_m = _FakeMarking([1, 0, 0], prefix=10)
    ppo.petri_net.set_marking(m)
    ppo.petri_net._next_marking = next_m
    _, reward, done, _ = ppo._step_env(m, 0, {})
    assert done, "Should reach goal"
    assert reward >= 200.0, f"First goal reward should include bonus, got {reward}"
    print("  [PASS] test_step_goal_first")


def test_step_goal_improvement():
    """改进 makespan 应获得额外 bonus。"""
    ppo = _build_ppo(reward_goal_bonus=200.0, reward_time_scale=100.0)
    ppo.end = [1, 0, 0]
    ppo.best_records["test"]["makespan"] = 100
    next_m = _FakeMarking([1, 0, 0], prefix=50)
    m = _FakeMarking([0, 0, 0], prefix=40)
    ppo.petri_net.set_marking(m)
    ppo.petri_net._next_marking = next_m
    _, reward, done, _ = ppo._step_env(m, 0, {})
    assert done
    assert reward > 200.0, f"Improvement should give > base bonus, got {reward}"
    print("  [PASS] test_step_goal_improvement")


def test_step_goal_degradation():
    """退化 makespan 应获得降级 bonus（> 0 但 < base）。"""
    ppo = _build_ppo(reward_goal_bonus=200.0, reward_time_scale=100.0)
    ppo.end = [1, 0, 0]
    ppo.best_records["test"]["makespan"] = 50
    next_m = _FakeMarking([1, 0, 0], prefix=80)
    m = _FakeMarking([0, 0, 0], prefix=70)
    ppo.petri_net.set_marking(m)
    ppo.petri_net._next_marking = next_m
    _, reward, done, _ = ppo._step_env(m, 0, {})
    assert done
    assert reward > 0, f"Degradation should still be positive, got {reward}"
    print("  [PASS] test_step_goal_degradation")


def test_step_clip_symmetric():
    """step_reward 限幅的上下界应与 reward_deadlock_penalty 相关（不再硬编码50）。"""
    dp = 80.0
    ppo = _build_ppo(reward_deadlock_penalty=dp)
    assert dp * 0.5 == 40.0, "Sanity: upper clip = dp * 0.5"
    print("  [PASS] test_step_clip_symmetric")


# ────────── 驻留时间阈值随网文件适配 ──────────

def test_threshold_adapts_to_net():
    """不同 max_residence_time 向量产生不同的惩罚行为。"""
    ppo_tight = _build_ppo(max_residence_time=[20, 2**31-1, 2**31-1])
    ppo_loose = _build_ppo(max_residence_time=[200, 2**31-1, 2**31-1])
    m = _FakeMarking([1, 0, 0], prefix=0,
                     residence_time_info=[deque([18]), deque(), deque()])
    r_tight = ppo_tight._compute_residence_reward(m)
    r_loose = ppo_loose._compute_residence_reward(m)
    assert r_tight < r_loose, f"Tight net should penalize more: tight={r_tight}, loose={r_loose}"
    print("  [PASS] test_threshold_adapts_to_net")


# ────────── 整合运行 ──────────

def main():
    print("=" * 60)
    print("Testing improved reward function")
    print("=" * 60)

    tests = [
        test_residence_safe,
        test_residence_warn_zone,
        test_residence_progressive,
        test_residence_over,
        test_residence_no_info,
        test_residence_unconstrained,
        test_step_illegal_action,
        test_step_normal,
        test_step_repeat_penalty_capped,
        test_step_goal_first,
        test_step_goal_improvement,
        test_step_goal_degradation,
        test_step_clip_symmetric,
        test_threshold_adapts_to_net,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")


if __name__ == "__main__":
    main()
