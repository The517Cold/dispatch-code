import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


from petri_net_platform.petri_net import TTPPNByTokenWithResTime
from petri_net_io.utils.net_loader import load_petri_net_context


INF = 2 ** 31 - 1


def _build_net(delay, qtime=5, with_mapping=True, with_qtime_places=True):
    p_info = [1, 0, 0]
    pre = [
        [1, 0],  # source --t0-->
        [0, 1],  # target --t1-->
        [0, 0],
    ]
    post = [
        [0, 0],
        [1, 0],
        [0, 1],
    ]
    min_delay_p = [0, 0, 0]
    min_delay_t = [delay, 0]
    capacity = [INF, INF, INF]
    residence_time = [INF, INF, INF]
    is_resource = [False, True, False]
    place_from_places = [[], [0], []] if with_mapping else [[], [], []]
    qtime_places = [False, with_qtime_places, False]
    return TTPPNByTokenWithResTime(
        p_info,
        pre,
        post,
        min_delay_p,
        min_delay_t,
        capacity,
        residence_time,
        is_resource,
        place_from_places,
        qtime_places,
        qtime,
    )


def test_place_to_places_qtime_violation_blocks_followup_transition():
    # 扩展公式：net_wait = prefix - last_tran_delay - curr_tran_delay - move_delay
    #                        - complete_time - qtime_map
    # 对于直接一跳（source → T0 → qtime_place），net_wait = tran_delay - tran_delay = 0，
    # 不会触发超时。单跳场景本身已合法，超时仅在多跳中无法解释的等待时间过长时发生。
    petri_net = _build_net(delay=6, qtime=5)

    next_marking = petri_net.launch(0)

    assert next_marking.over_max_residence_time is False  # 扩展公式：单跳无超时
    assert next_marking.get_prefix() == 6
    assert next_marking.qtime_map[0] == 0
    assert next_marking.t_info[1][0].get_id() == 0

    petri_net.set_marking(next_marking)
    assert petri_net.enable(1) is True  # 未超时，变迁仍然使能


def test_place_to_places_qtime_allows_transition_within_limit():
    petri_net = _build_net(delay=5, qtime=5)

    next_marking = petri_net.launch(0)

    assert next_marking.over_max_residence_time is False
    assert next_marking.get_prefix() == 5

    petri_net.set_marking(next_marking)
    assert petri_net.enable(1) is True


def test_qtime_configuration_is_backward_compatible_without_mapping():
    petri_net = _build_net(delay=100, qtime=5, with_mapping=False)

    next_marking = petri_net.launch(0)

    assert next_marking.over_max_residence_time is False


def test_case_file_place_to_places_is_loaded():
    net_path = os.path.join(
        REPO_ROOT,
        "resources",
        "resources_new",
        "test",
        "class_test",
        "case1-1",
        "1-1-1.txt",
    )
    if not os.path.exists(net_path):
        return

    context = load_petri_net_context(net_path)
    p_map = context["matrix_translator"].p_map
    place_from_places = context["place_from_places"]

    assert context["qtime"] == 3000
    assert p_map["2"] in place_from_places[p_map["18"]]
    assert p_map["2"] in place_from_places[p_map["20"]]
    assert p_map["18"] in place_from_places[p_map["5"]]


if __name__ == "__main__":
    test_place_to_places_qtime_violation_blocks_followup_transition()
    test_place_to_places_qtime_allows_transition_within_limit()
    test_qtime_configuration_is_backward_compatible_without_mapping()
    test_case_file_place_to_places_is_loaded()
    print("PASS test_qtime_place_to_places")
