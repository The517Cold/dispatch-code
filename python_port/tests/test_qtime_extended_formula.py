"""
单元测试：验证扩展 qtime 超时判断公式的正确性。

扩展公式（_is_over_qtime）：
  net_wait = 当前时间节点
             - token.last_tran_delay     （上次变迁延迟）
             - min_delay_t[tran]         （本次变迁延迟）
             - move_place_delay_by_tran  （movePlaces 标记的前置库所 ptime 之和）
             - complete_time             （completeTime 常量）
             - qtime_map[token_id]       （token 上次离开 qtime 源库所时的时间节点）

  net_wait > qtime → 超时

覆盖场景：
  1. 单跳直连（source → T0 → qtime_place）：永不超时，net_wait = 0
  2. 两跳场景，无 movePlaces：中间库所 ptime 算作等待，可超时
  3. 两跳场景，有 movePlaces：中间 ptime 被扣除，超时判断宽松
  4. completeTime 影响：扣除后 net_wait 减小
  5. 文件解析：从资源文件正确读取 movePlaces 和 completeTime
  6. Token.tran_delay / last_tran_delay 正确传递
  7. 克隆（clone）后新网络保留 move_places 和 complete_time
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from petri_net_platform.petri_net import TTPPNByTokenWithResTime
from petri_net_platform.marking import Token
from petri_net_io.utils.net_loader import load_petri_net_context

INF = 2 ** 31 - 1


# ---------------------------------------------------------------------------
# 辅助：构造单跳网络（source → T0 → qtime_place）
# ---------------------------------------------------------------------------

def _build_direct_net(tran_delay=6, qtime=5, move_places=None, complete_time=0):
    """
    place 0 ──T0(delay)──► place 1 (qtime)
    place_from_places[1] = [0]，qtime_places[1] = True
    """
    p_info = [1, 0]
    pre = [[1, 0],   # place 0: T0 消耗
           [0, 0]]
    post = [[0, 0],
            [1, 0]]  # place 1: T0 产出
    min_delay_p = [0, 0]
    min_delay_t = [tran_delay, 0]
    capacity = [INF, INF]
    residence_time = [INF, INF]
    is_resource = [False, False]
    place_from_places = [[], [0]]       # place 1 的 token 溯源到 place 0
    qtime_places = [False, True]
    return TTPPNByTokenWithResTime(
        p_info, pre, post, min_delay_p, min_delay_t,
        capacity, residence_time, is_resource,
        place_from_places, qtime_places, qtime,
        move_places, complete_time,
    )


# ---------------------------------------------------------------------------
# 辅助：构造两跳网络（source → T0 → intermediate(ptime) → T1 → qtime_place）
# ---------------------------------------------------------------------------

def _build_two_hop_net(t0_delay, p1_ptime, t1_delay, qtime,
                       move_places=None, complete_time=0):
    """
    place 0 ──T0──► place 1(ptime) ──T1──► place 2 (qtime)
    place_from_places[2] = [0]  → 跟踪 token 从 place 0 到 place 2
    qtime_source_places[0] = True，qtime_map 在 token 离开 place 0 时记录
    """
    p_info = [1, 0, 0]
    pre = [
        [1, 0],   # place 0: T0 消耗
        [0, 1],   # place 1: T1 消耗
        [0, 0],   # place 2: 不消耗
    ]
    post = [
        [0, 0],   # place 0 不产出
        [1, 0],   # place 1: T0 产出
        [0, 1],   # place 2: T1 产出
    ]
    min_delay_p = [0, p1_ptime, 0]
    min_delay_t = [t0_delay, t1_delay]
    capacity = [INF, INF, INF]
    residence_time = [INF, INF, INF]
    is_resource = [False, False, False]
    place_from_places = [[], [], [0]]   # place 2 的 token 溯源到 place 0
    qtime_places = [False, False, True]
    return TTPPNByTokenWithResTime(
        p_info, pre, post, min_delay_p, min_delay_t,
        capacity, residence_time, is_resource,
        place_from_places, qtime_places, qtime,
        move_places, complete_time,
    )


# ===========================================================================
# 测试 1：单跳直连，net_wait = 0，不超时
# ===========================================================================

def test_direct_hop_no_violation_regardless_of_delay():
    """
    单跳中 net_wait = prefix - curr_tran_delay - qtime_map
                    = tran_delay - tran_delay - 0 = 0
    无论 tran_delay 多大，均不超时。
    """
    net = _build_direct_net(tran_delay=1000, qtime=1)
    marking = net.launch(0)
    assert marking.over_max_residence_time is False, "单跳永不触发超时"
    assert marking.get_prefix() == 1000


def test_direct_hop_tran_delay_equals_qtime_no_violation():
    net = _build_direct_net(tran_delay=5, qtime=5)
    marking = net.launch(0)
    assert marking.over_max_residence_time is False


# ===========================================================================
# 测试 2：两跳场景，无 movePlaces——中间 ptime 计为等待时间
# net_wait = D0 + P1 + D1 - 0(last) - D1(curr) - 0(move) - 0(complete) - 0(qmap)
#          = D0 + P1
# ===========================================================================

def test_two_hop_violation_when_intermediate_ptime_too_large():
    """
    D0=5, P1=10, D1=3, qtime=5 → net_wait = 5+10 = 15 > 5 → 超时
    """
    net = _build_two_hop_net(t0_delay=5, p1_ptime=10, t1_delay=3, qtime=5)
    marking_after_t0 = net.launch(0)
    assert marking_after_t0.over_max_residence_time is False   # 尚未到达 qtime_place
    assert marking_after_t0.get_prefix() == 5

    net.set_marking(marking_after_t0)
    marking_after_t1 = net.launch(1)
    # net_wait = D0 + P1 = 5 + 10 = 15 > qtime=5 → 超时
    assert marking_after_t1.over_max_residence_time is True, (
        f"net_wait 应为 {5 + 10} > qtime=5，预期超时"
    )


def test_two_hop_no_violation_when_within_qtime():
    """
    D0=1, P1=2, D1=3, qtime=10 → net_wait = 1+2 = 3 ≤ 10 → 不超时
    """
    net = _build_two_hop_net(t0_delay=1, p1_ptime=2, t1_delay=3, qtime=10)
    net.set_marking(net.launch(0))
    marking = net.launch(1)
    assert marking.over_max_residence_time is False, (
        f"net_wait 应为 {1 + 2} = 3 ≤ qtime=10，不超时"
    )


def test_two_hop_boundary_equal_to_qtime_no_violation():
    """
    D0=2, P1=3, D1=1, qtime=5 → net_wait = 5 = qtime → 严格大于判断，不超时
    """
    net = _build_two_hop_net(t0_delay=2, p1_ptime=3, t1_delay=1, qtime=5)
    net.set_marking(net.launch(0))
    marking = net.launch(1)
    # net_wait = D0 + P1 = 2 + 3 = 5, 5 > 5 为 False
    assert marking.over_max_residence_time is False, "等于 qtime 时不超时（严格大于）"


# ===========================================================================
# 测试 3：movePlaces 扣除中间 ptime，减小 net_wait
# net_wait = D0 + P1 - 0(last) - D1(curr) - P1(move) - 0(complete) - 0(qmap)
#          = D0 - D1
# ===========================================================================

def test_two_hop_move_places_absorbs_ptime():
    """
    place 1 标记为 movePlaces → P1 被扣除。
    D0=5, P1=10, D1=3, qtime=5 → net_wait = 5 - 3 = 2 ≤ 5 → 不超时
    （若无 movePlaces，net_wait = 15 > 5 → 超时）
    """
    move_places = [False, True, False]   # place 1 是 movePlaces
    net = _build_two_hop_net(t0_delay=5, p1_ptime=10, t1_delay=3, qtime=5,
                              move_places=move_places)
    net.set_marking(net.launch(0))
    marking = net.launch(1)
    assert marking.over_max_residence_time is False, (
        "movePlaces 扣除 ptime=10 后 net_wait=2 ≤ 5，不超时"
    )


def test_two_hop_move_places_still_violates_when_d0_too_large():
    """
    place 1 标记为 movePlaces，但 D0 本身过大。
    D0=20, P1=10, D1=3, qtime=5 → net_wait = 20 - 3 = 17 > 5 → 超时
    """
    move_places = [False, True, False]
    net = _build_two_hop_net(t0_delay=20, p1_ptime=10, t1_delay=3, qtime=5,
                              move_places=move_places)
    net.set_marking(net.launch(0))
    marking = net.launch(1)
    assert marking.over_max_residence_time is True, (
        "movePlaces 扣除后 net_wait=17 > 5，仍超时"
    )


# ===========================================================================
# 测试 4：completeTime 进一步扣除
# ===========================================================================

def test_complete_time_reduces_net_wait():
    """
    D0=5, P1=10, D1=3, qtime=5, complete_time=10 →
    net_wait = 5+10 - 0 - 3 - 0 - 10 - 0 = 2 ≤ 5 → 不超时
    （若无 completeTime，net_wait=15 > 5 → 超时）
    """
    net = _build_two_hop_net(t0_delay=5, p1_ptime=10, t1_delay=3, qtime=5,
                              complete_time=10)
    net.set_marking(net.launch(0))
    marking = net.launch(1)
    assert marking.over_max_residence_time is False, (
        "completeTime=10 使 net_wait=2 ≤ 5，不超时"
    )


def test_complete_time_not_sufficient_still_violates():
    """
    D0=5, P1=10, D1=3, qtime=5, complete_time=3 →
    net_wait = 15 - 3 - 3 = 9 > 5 → 仍超时
    """
    net = _build_two_hop_net(t0_delay=5, p1_ptime=10, t1_delay=3, qtime=5,
                              complete_time=3)
    net.set_marking(net.launch(0))
    marking = net.launch(1)
    assert marking.over_max_residence_time is True, (
        "completeTime=3 不足以抵消，net_wait=9 > 5，仍超时"
    )


# ===========================================================================
# 测试 5：Token.tran_delay 与 last_tran_delay 正确传递
# ===========================================================================

def test_token_tran_delay_set_on_placement():
    """验证 token 被放入库所时 tran_delay = 本次变迁延迟。"""
    net = _build_direct_net(tran_delay=7, qtime=100)
    marking = net.launch(0)
    token_in_place1 = list(marking.t_info[1])[0]
    assert token_in_place1.tran_delay == 7, (
        f"tran_delay 应为 7，实为 {token_in_place1.tran_delay}"
    )


def test_token_last_tran_delay_propagated():
    """
    两跳网络：place 0(init) → T0(D0=8) → place 1 → T1(D1=4) → place 2
    T0 产生放入 place 1 的 token，tran_delay=8。
    T1 发射后 place 2 中 token 的 last_tran_delay 应为 0
    （因为 specific_get[0] 中保存的是初始 token，其 tran_delay=0）。
    """
    net = _build_two_hop_net(t0_delay=8, p1_ptime=0, t1_delay=4, qtime=100)
    m1 = net.launch(0)
    net.set_marking(m1)
    m2 = net.launch(1)
    token_in_place2 = list(m2.t_info[2])[0]
    assert token_in_place2.tran_delay == 4, "tran_delay 应为 T1 的延迟 4"
    assert token_in_place2.last_tran_delay == 0, (
        "last_tran_delay 应为 0（初始 token 未被变迁产生，tran_delay=0）"
    )


# ===========================================================================
# 测试 6：Token.clone() 保留新属性
# ===========================================================================

def test_token_clone_preserves_tran_delays():
    t = Token(42, timer=5, residence_time=3, tran_delay=10, last_tran_delay=7)
    c = t.clone()
    assert c.tran_delay == 10
    assert c.last_tran_delay == 7
    assert c.get_id() == 42
    assert c.timer == 5
    assert c.residence_time == 3


# ===========================================================================
# 测试 7：Marking.clone() 保留 Token 新属性
# ===========================================================================

def test_marking_clone_preserves_token_tran_delays():
    net = _build_two_hop_net(t0_delay=5, p1_ptime=3, t1_delay=2, qtime=100)
    marking = net.launch(0)
    cloned = marking.clone()
    for place_idx in range(len(marking.t_info)):
        orig_tokens = list(marking.t_info[place_idx])
        clone_tokens = list(cloned.t_info[place_idx])
        assert len(orig_tokens) == len(clone_tokens)
        for ot, ct in zip(orig_tokens, clone_tokens):
            assert ot.tran_delay == ct.tran_delay
            assert ot.last_tran_delay == ct.last_tran_delay


# ===========================================================================
# 测试 8：克隆网络保留 move_places 与 complete_time
# ===========================================================================

def test_clone_preserves_move_places_and_complete_time():
    move_places = [False, True, False]
    net = _build_two_hop_net(t0_delay=5, p1_ptime=10, t1_delay=3, qtime=5,
                              move_places=move_places, complete_time=2)
    cloned_net = net.clone()
    assert cloned_net.move_places == move_places
    assert cloned_net.complete_time == 2
    assert cloned_net.move_place_delay_by_tran[1] == 10  # T1 的 move_delay = P1=10


# ===========================================================================
# 测试 9：move_place_delay_by_tran 预计算正确
# ===========================================================================

def test_move_place_delay_precomputed():
    """
    T0 前置: place 0 (ptime=5)
    T1 前置: place 1 (ptime=10，movePlaces)
    move_place_delay_by_tran[0] 应为 0（place 0 不在 movePlaces）
    move_place_delay_by_tran[1] 应为 10（place 1 在 movePlaces）
    """
    move_places = [False, True, False]
    net = _build_two_hop_net(t0_delay=5, p1_ptime=10, t1_delay=3, qtime=50,
                              move_places=move_places)
    assert net.move_place_delay_by_tran[0] == 0, "T0 的 movePlaces 延迟应为 0"
    assert net.move_place_delay_by_tran[1] == 10, "T1 的 movePlaces 延迟应为 10"


# ===========================================================================
# 测试 10：资源文件解析 movePlaces 与 completeTime
# ===========================================================================

def test_resource_file_move_places_and_complete_time_loaded():
    """验证 case1-2/resources/1-1-1.txt 中 movePlaces 和 completeTime 能被正确解析。"""
    net_path = os.path.join(
        REPO_ROOT,
        "resources", "resources_new", "train", "class", "case1-2",
        "resources", "1-1-1.txt",
    )
    if not os.path.exists(net_path):
        return  # 文件不存在时跳过（CI 环境兼容）

    context = load_petri_net_context(net_path)
    p_map = context["matrix_translator"].p_map
    move_places = context["move_places"]
    complete_time = context["complete_time"]

    # completeTime:2
    assert complete_time == 2, f"completeTime 应为 2，实为 {complete_time}"

    # movePlaces:8 23 7 3 24 → 这些库所编号（字符串）应被标记为 True
    expected_move_place_names = {"8", "23", "7", "3", "24"}
    for name in expected_move_place_names:
        if name in p_map:
            idx = p_map[name]
            assert move_places[idx] is True, (
                f"库所 {name}（索引 {idx}）应被标记为 movePlaces"
            )

    # 其余库所不在 movePlaces 中（抽检）
    for name in ["1", "2", "4", "5", "6"]:
        if name in p_map:
            idx = p_map[name]
            if name not in expected_move_place_names:
                assert move_places[idx] is False, (
                    f"库所 {name} 不应在 movePlaces 中"
                )


# ===========================================================================
# 测试 11：without_qtime_places 依然向后兼容
# ===========================================================================

def test_no_qtime_places_no_violation():
    """qtime_places 全为 False → _is_over_qtime 始终返回 False。"""
    p_info = [1, 0]
    pre = [[1, 0], [0, 0]]
    post = [[0, 0], [1, 0]]
    net = TTPPNByTokenWithResTime(
        p_info, pre, post, [0, 0], [100, 0],
        [INF, INF], [INF, INF],
        [False, False], [[], []], [False, False], 5,
        None, 0,
    )
    marking = net.launch(0)
    assert marking.over_max_residence_time is False


# ===========================================================================
# 测试 12：不传 move_places / complete_time 时行为与默认值一致
# ===========================================================================

def test_default_move_places_and_complete_time():
    """不传 move_places 和 complete_time 时，默认行为与全 False / 0 相同。"""
    p_info = [1, 0, 0]
    pre = [[1, 0], [0, 1], [0, 0]]
    post = [[0, 0], [1, 0], [0, 1]]
    min_delay_p = [0, 0, 0]
    min_delay_t = [3, 2]
    capacity = [INF, INF, INF]
    residence_time = [INF, INF, INF]
    is_resource = [False, False, False]
    place_from_places = [[], [], [0]]
    qtime_places = [False, False, True]
    qtime = 100

    net_default = TTPPNByTokenWithResTime(
        p_info, pre, post, min_delay_p, min_delay_t,
        capacity, residence_time, is_resource,
        place_from_places, qtime_places, qtime,
    )
    net_explicit = TTPPNByTokenWithResTime(
        p_info, pre, post, min_delay_p, min_delay_t,
        capacity, residence_time, is_resource,
        place_from_places, qtime_places, qtime,
        None, 0,
    )
    assert net_default.move_places == net_explicit.move_places
    assert net_default.complete_time == net_explicit.complete_time
    assert net_default.move_place_delay_by_tran == net_explicit.move_place_delay_by_tran


# ===========================================================================
# 测试 13：qtime_map 仅当 token 在 qtime_source_places 时被记录
# ===========================================================================

def test_qtime_map_recorded_on_qtime_source_departure():
    """
    发射 T0 后，qtime_map[0] 应记录 token 离开 qtime_source 的时间节点。
    在本例中，place 0 是 qtime_source（因为 place_from_places[1] = [0]），
    token A 在 t=0 前置时间=0，qtime_map[A] 应为 0。
    """
    net = _build_direct_net(tran_delay=10, qtime=100)
    marking = net.launch(0)
    assert 0 in marking.qtime_map, "qtime_map 应有 token 0 的记录"
    assert marking.qtime_map[0] == 0, "token 在 t=0 离开 qtime_source，qtime_map 应为 0"


if __name__ == "__main__":
    test_direct_hop_no_violation_regardless_of_delay()
    test_direct_hop_tran_delay_equals_qtime_no_violation()
    test_two_hop_violation_when_intermediate_ptime_too_large()
    test_two_hop_no_violation_when_within_qtime()
    test_two_hop_boundary_equal_to_qtime_no_violation()
    test_two_hop_move_places_absorbs_ptime()
    test_two_hop_move_places_still_violates_when_d0_too_large()
    test_complete_time_reduces_net_wait()
    test_complete_time_not_sufficient_still_violates()
    test_token_tran_delay_set_on_placement()
    test_token_last_tran_delay_propagated()
    test_token_clone_preserves_tran_delays()
    test_marking_clone_preserves_token_tran_delays()
    test_clone_preserves_move_places_and_complete_time()
    test_move_place_delay_precomputed()
    test_resource_file_move_places_and_complete_time_loaded()
    test_no_qtime_places_no_violation()
    test_default_move_places_and_complete_time()
    test_qtime_map_recorded_on_qtime_source_departure()
    print("PASS test_qtime_extended_formula")
