import heapq
import itertools
import math
import os
import sys
import time

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from python_port.petri_net_platform.search.abstract_search import AbstractSearch
from python_port.petri_net_platform.search.greedy import Greedy
from python_port.petri_net_platform.utils.low_space_link import LowSpaceLink, TranLink
from python_port.petri_net_platform.utils.result import Result
from python_port.petri_net_io.utils.object_to_petri_net_info import CustomMatrixTranslator


class EvaluationFunction:
    def __init__(self, petri_net_file=None, time=None, non_resource_place=None, place_depth=None, max_remaining_time=None, max_depth=None):
        self.maxDepth = 0
        self.maxRemainingTime = 0
        self.remainTime = []
        self.placeDepth = []
        self.placeRemainTime = []
        self.nonResourcePlace = set()
        if petri_net_file is not None:
            self.get_efline(petri_net_file)
            self.remainTime = [0] * len(self.placeRemainTime)
            for i in range(len(self.remainTime)):
                if self.placeRemainTime[i] != -1:
                    self.remainTime[i] = self.maxRemainingTime - self.placeRemainTime[i]
            return
        if time is not None and place_depth is not None:
            self.maxDepth = max_depth if max_depth is not None else 0
            self.maxRemainingTime = max_remaining_time if max_remaining_time is not None else 0
            self.remainTime = [0] * len(time)
            for i in range(len(time)):
                if time[i] != -1:
                    self.remainTime[i] = time[i]
            self.placeDepth = place_depth
            if non_resource_place is not None:
                for val in non_resource_place:
                    self.nonResourcePlace.add(val)

    def __str__(self):
        return "EvaluationFunction{" + "maxDepth=" + str(self.maxDepth) + ", maxRemainingTime=" + str(self.maxRemainingTime) + ", remainTime=" + str(self.remainTime) + ", placeDepth=" + str(self.placeDepth) + ", nonResourcePlace=" + str(self.nonResourcePlace) + "}"

    def get_efline(self, petri_net_file):
        txt = petri_net_file.EFline
        translator = CustomMatrixTranslator(petri_net_file)
        translator.translate()
        min_delay_p = translator.vectors.get("minDelayP")
        p_num = len(min_delay_p)
        self.placeRemainTime = [-1] * p_num
        self.placeDepth = [0] * p_num
        if txt is None:
            return
        txt = txt[3:].strip()
        nums = txt.split(" ")
        for i, item in enumerate(nums):
            if item == "":
                continue
            place_time = item.split("-")
            if i == len(nums) - 1:
                self.maxRemainingTime = int(place_time[1])
            if i == 0:
                self.maxDepth = int(place_time[2])
            place_index = translator.p_map.get(place_time[0])
            if place_index is None:
                continue
            self.nonResourcePlace.add(place_index)
            self.placeRemainTime[place_index] = int(place_time[1])
            self.placeDepth[place_index] = int(place_time[2])


class AbstractPetriNetComparator:
    def __init__(self, a_matrix, evaluation_function):
        self.a_matrix = a_matrix
        self.EF = evaluation_function
        self.e = 0.1
        self.h_m0 = 4

    def g(self, marking):
        return math.atan(marking.get_prefix()) * 2 / math.pi

    def h(self, marking, evaluation_function):
        return 0

    def f(self, marking):
        h1 = self.h(marking, self.EF)
        h_val = self.e * h1 * h1 / self.h_m0
        return self.g(marking) + h1 + h_val


class MyComparator(AbstractPetriNetComparator):
    def h(self, marking, evaluation_function):
        p_info = marking.get_p_info()
        non_resource_place = evaluation_function.nonResourcePlace
        remain_time = evaluation_function.remainTime
        place_depth = evaluation_function.placeDepth
        h_val = 0
        for i in range(len(p_info)):
            if p_info[i] != 0 and i in non_resource_place:
                token_time = marking.t_info[i]
                remain_token_time = min(token_time) if token_time else 0
                cur_place_remain_time = 0
                if evaluation_function.maxRemainingTime > 1 and remain_time[i] + remain_token_time > 0:
                    cur_place_remain_time = math.log10(remain_time[i] + remain_token_time) / math.log10(evaluation_function.maxRemainingTime)
                cur_place_depth = 0
                if evaluation_function.maxDepth > 1 and place_depth[i] > 0:
                    cur_place_depth = math.log10(place_depth[i]) / math.log10(evaluation_function.maxDepth)
                h_val = max(h_val, cur_place_remain_time, cur_place_depth)
        return h_val


class OpenTable:
    def __init__(self, a_matrix=None, evaluation_function=None):
        self._heap = []
        self._counter = itertools.count()
        self._comparator = None
        if a_matrix is not None and evaluation_function is not None:
            self._comparator = MyComparator(a_matrix, evaluation_function)

    def offer(self, marking, state_key, close):
        # open 表按状态键保存最优 g 值，避免带时间信息的 marking 被错误合并。
        g_val = close.get(state_key, 2 ** 31 - 1)
        if self._comparator is None:
            priority = g_val
        else:
            priority = self._comparator.f(marking)
        heapq.heappush(self._heap, (priority, next(self._counter), state_key, marking, g_val))

    def poll(self, close):
        while self._heap:
            _, _, state_key, marking, g_val = heapq.heappop(self._heap)
            if close.get(state_key) == g_val:
                return marking, state_key
        return None, None

    def is_empty(self):
        return len(self._heap) == 0

    def size(self):
        # 便于搜索器输出实时进度，观察当前 open 表规模。
        return len(self._heap)


class AStar(AbstractSearch):
    def __init__(
        self,
        petri_net,
        end,
        open_table=None,
        use_greedy_upper_bound=True,
        max_search_seconds=None,
        max_expand_nodes=None,
    ):
        super().__init__()
        self.petri_net = petri_net
        self.end = end
        self.init_marking = petri_net.get_marking()
        self.use_greedy_upper_bound = use_greedy_upper_bound
        self.max_search_seconds = max_search_seconds
        self.max_expand_nodes = max_expand_nodes
        self.extend_marking_count = 0
        self.deadlock_pruned_count = 0
        self.residence_pruned_count = 0
        self.bound_pruned_count = 0
        self.dominance_pruned_count = 0
        self.close = {}
        self.path = {}
        self.extra_info = {}
        self.open_table = open_table if open_table is not None else OpenTable()
        self.best_makespan = 2 ** 31 - 1
        self.best_result = None
        self.best_path_link = None
        self.dominance_frontier = {}
        self.timeout_triggered = False
        self.expand_limit_triggered = False
        self.search_start_time = None

    def search(self):
        self.search_start_time = time.perf_counter()
        self.timeout_triggered = False
        self.expand_limit_triggered = False
        self._seed_upper_bound()
        self._put_init_marking()
        return self._find()

    def _find(self):
        while not self.open_table.is_empty():
            if self._is_timeout():
                self.timeout_triggered = True
                break
            curr, curr_key = self.open_table.poll(self.close)
            if curr is None:
                break
            if not self._is_valid_marking(curr):
                self.residence_pruned_count += 1
                continue
            if curr.get_prefix() >= self.best_makespan:
                self.bound_pruned_count += 1
                continue
            if self.same(curr):
                candidate = self.path.get(curr_key)
                if candidate is not None:
                    self.best_makespan = curr.get_prefix()
                    self.best_path_link = candidate
                    self.best_result = None
                continue
            self.petri_net.set_marking(curr)
            pre_path = self.path.get(curr_key)
            enabled_trans = self._get_enabled_transitions()
            if not enabled_trans:
                # 没有任何可扩展变迁时，把当前状态视为死锁叶子并直接跳过。
                self.deadlock_pruned_count += 1
                continue
            hit_expand_limit = False
            for tran in enabled_trans:
                next_marking = self.petri_net.launch(tran)
                self.extend_marking_count += 1
                if self.max_expand_nodes is not None and self.extend_marking_count >= self.max_expand_nodes:
                    self.expand_limit_triggered = True
                    hit_expand_limit = True
                    break
                if not self._is_valid_marking(next_marking):
                    self.residence_pruned_count += 1
                    continue
                if next_marking.get_prefix() >= self.best_makespan:
                    self.bound_pruned_count += 1
                    continue
                if self._is_deadlock_marking(next_marking) and not self.same(next_marking):
                    self.deadlock_pruned_count += 1
                    continue
                if self._is_dominated(next_marking):
                    self.dominance_pruned_count += 1
                    continue
                next_key = self._state_key(next_marking)
                time_val = self.close.get(next_key, 2 ** 31 - 1)
                if time_val > next_marking.get_prefix():
                    new_path = LowSpaceLink(TranLink(tran, pre_path.tran_link), next_marking)
                    self.path[next_key] = new_path
                    self.close[next_key] = next_marking.get_prefix()
                    self._register_dominance(next_marking)
                    self.open_table.offer(next_marking, next_key, self.close)
            if hit_expand_limit:
                break
        if self.best_path_link is not None:
            return self._make_result(self.best_path_link)
        return self.best_result

    def _make_result(self, ans):
        trans = []
        markings = []
        tran_link = ans.tran_link
        while tran_link is not None:
            trans.append(tran_link.curr_tran)
            tran_link = tran_link.pre
        trans.reverse()
        self.petri_net.set_marking(self.init_marking)
        markings.append(self.init_marking)
        for tran in trans:
            curr = self.petri_net.launch(tran)
            markings.append(curr)
            self.petri_net.set_marking(curr)
        self.extra_info["extendMarkingCount"] = self.extend_marking_count
        return Result(trans, markings)

    def _put_init_marking(self):
        curr = self.petri_net.get_marking()
        curr_key = self._state_key(curr)
        self.close[curr_key] = curr.get_prefix()
        self.path[curr_key] = LowSpaceLink(None, curr)
        self._register_dominance(curr)
        self.open_table.offer(curr, curr_key, self.close)

    def get_extra_info(self):
        self.extra_info["deadlockPrunedCount"] = self.deadlock_pruned_count
        self.extra_info["residencePrunedCount"] = self.residence_pruned_count
        self.extra_info["boundPrunedCount"] = self.bound_pruned_count
        self.extra_info["dominancePrunedCount"] = self.dominance_pruned_count
        self.extra_info["bestUpperBound"] = self.best_makespan if self.best_makespan != 2 ** 31 - 1 else None
        self.extra_info["timeoutTriggered"] = self.timeout_triggered
        self.extra_info["terminatedByMaxSearchSeconds"] = self.timeout_triggered
        self.extra_info["maxSearchSeconds"] = self.max_search_seconds
        self.extra_info["terminatedByMaxExpandNodes"] = self.expand_limit_triggered
        self.extra_info["maxExpandNodes"] = self.max_expand_nodes
        return self.extra_info

    def _seed_upper_bound(self):
        # 先用一个很快的贪心解给 A* 提供可行上界，后续只保留可能更优的状态。
        if not self.use_greedy_upper_bound:
            return
        original = self.petri_net.get_marking()
        self.petri_net.set_marking(self.init_marking)
        try:
            greedy_search = Greedy(self.petri_net.clone(), self.end)
            greedy_result = greedy_search.search()
        except BaseException:
            greedy_result = None
        finally:
            self.petri_net.set_marking(original)
        if greedy_result is None or not greedy_result.get_markings():
            return
        last_marking = greedy_result.get_markings()[-1]
        if not self.same(last_marking):
            return
        if not self._is_valid_marking(last_marking):
            return
        self.best_result = greedy_result
        self.best_makespan = last_marking.get_prefix()

    def _get_enabled_transitions(self):
        # 统一收集当前 marking 下可发射的变迁，后续既用于扩展也用于死锁判定。
        enabled = []
        for tran in range(self.petri_net.get_trans_count()):
            if self.petri_net.enable(tran):
                enabled.append(tran)
        return enabled

    def _is_valid_marking(self, marking):
        # 驻留时间超限的状态直接剪枝，保证返回路径始终满足网文件约束。
        judge = getattr(marking, "is_over_residece_time", None)
        if callable(judge):
            return not judge()
        over_flag = getattr(marking, "over_max_residence_time", False)
        return not bool(over_flag)

    def _is_deadlock_marking(self, marking):
        # 对时序网优先复用 marking 上缓存的 is_enable，普通网则退回 enable 判断。
        is_enable = getattr(marking, "is_enable", None)
        if is_enable is not None:
            return not any(is_enable)
        current = self.petri_net.get_marking()
        self.petri_net.set_marking(marking)
        try:
            return not any(self.petri_net.enable(tran) for tran in range(self.petri_net.get_trans_count()))
        finally:
            self.petri_net.set_marking(current)

    def _state_key(self, marking):
        # A* 去重不能只看 p_info；对时序 Petri 网必须把影响未来可行性的时间态一起纳入签名。
        cached = getattr(marking, "_astar_state_key", None)
        if cached is not None:
            return cached
        key = [tuple(marking.get_p_info())]
        if hasattr(marking, "curr_delay_t"):
            key.append(tuple(marking.curr_delay_t))
        if hasattr(marking, "t_info"):
            key.append(self._serialize_nested(marking.t_info))
        if hasattr(marking, "residence_time_info"):
            key.append(self._serialize_nested(marking.residence_time_info))
        if hasattr(marking, "qtime_map"):
            key.append(tuple(sorted(marking.qtime_map.items())))
        if hasattr(marking, "max_id"):
            key.append(marking.max_id)
        over_flag = getattr(marking, "over_max_residence_time", False)
        key.append(bool(over_flag))
        cached_key = tuple(key)
        setattr(marking, "_astar_state_key", cached_key)
        return cached_key

    def _serialize_nested(self, groups):
        # t_info 里既可能是整数，也可能是 Token 对象；这里统一转成可哈希元组。
        result = []
        for group in groups:
            result.append(tuple(self._serialize_item(item) for item in group))
        return tuple(result)

    def _serialize_item(self, item):
        if hasattr(item, "get_id") and hasattr(item, "timer") and hasattr(item, "residence_time"):
            return (item.get_id(), item.timer, item.residence_time)
        return item

    def _is_dominated(self, marking):
        # 支配剪枝：相同库所分布下，若已有状态在时间和时间态上都不差于当前状态，则当前状态无需再扩展。
        profile = self._dominance_profile(marking)
        if profile is None:
            return False
        bucket = self.dominance_frontier.get(profile["key"], [])
        for existing in bucket:
            if self._dominates(existing, profile):
                return True
        return False

    def _register_dominance(self, marking):
        # 新状态写入前沿时，顺手删掉它已经严格支配的旧状态，减少后续比较成本。
        profile = self._dominance_profile(marking)
        if profile is None:
            return
        bucket = self.dominance_frontier.setdefault(profile["key"], [])
        survivors = []
        for existing in bucket:
            if self._dominates(profile, existing):
                continue
            survivors.append(existing)
        survivors.append(profile)
        self.dominance_frontier[profile["key"]] = survivors

    def _dominance_profile(self, marking):
        # 只有带时间态的网才做这层剪枝；普通 untimed 网保持原有行为即可。
        cached = getattr(marking, "_astar_dominance_profile", None)
        if cached is not None:
            return cached
        has_time_signature = hasattr(marking, "curr_delay_t") or hasattr(marking, "t_info")
        if not has_time_signature:
            return None
        profile = {
            "key": tuple(marking.get_p_info()),
            "prefix": marking.get_prefix(),
            "curr_delay_t": tuple(getattr(marking, "curr_delay_t", [])),
            "t_info": self._serialize_nested(getattr(marking, "t_info", [])),
            "residence_time_info": self._serialize_nested(getattr(marking, "residence_time_info", [])),
        }
        setattr(marking, "_astar_dominance_profile", profile)
        return profile

    def _dominates(self, left, right):
        # 这里故意用保守判定：全局时间、变迁延迟、token 计时和驻留都逐项不大于对方时，才认为 left 支配 right。
        if left["prefix"] > right["prefix"]:
            return False
        if not self._tuple_le(left["curr_delay_t"], right["curr_delay_t"]):
            return False
        if not self._nested_tuple_le(left["t_info"], right["t_info"]):
            return False
        if not self._nested_tuple_le(left["residence_time_info"], right["residence_time_info"]):
            return False
        return True

    def _tuple_le(self, left, right):
        if len(left) != len(right):
            return False
        for left_value, right_value in zip(left, right):
            if left_value > right_value:
                return False
        return True

    def _nested_tuple_le(self, left, right):
        if len(left) != len(right):
            return False
        for left_group, right_group in zip(left, right):
            if len(left_group) != len(right_group):
                return False
            for left_item, right_item in zip(left_group, right_group):
                if left_item > right_item:
                    return False
        return True

    def _is_timeout(self):
        # 限时模式下，A* 会返回当前最好可行解，而不是一直卡在大场景搜索里。
        if self.max_search_seconds is None or self.search_start_time is None:
            return False
        return (time.perf_counter() - self.search_start_time) >= self.max_search_seconds


class CreateEFLine:
    def __init__(self, petri_net, end, p_info, is_resource, parallel_place):
        self.petri_net = petri_net
        self.end = end
        self.is_resource = is_resource
        self.end_place = 0
        self.parallel_place_pair = {}
        for i in range(len(end)):
            if p_info[i] == 0 or end[i] == 13:
                self.is_resource[i] = False
        for i in range(len(end)):
            if end[i] == 13:
                end[i] = 1
                self.end_place = i
        for arr in parallel_place or []:
            if not arr:
                continue
            self.parallel_place_pair[arr[0]] = arr[-1]
            self.parallel_place_pair[arr[-1]] = arr[0]

    def ef_line(self, a_matrix, pmap_v, max_expand_nodes=None, max_search_seconds=None):
        sb = ["EF!"]
        a_star = AStar(
            self.petri_net,
            self.end,
            max_expand_nodes=max_expand_nodes,
            max_search_seconds=max_search_seconds,
        )
        result = a_star.search()
        if result is None:
            return None
        marking_list = result.get_markings()
        trans = result.get_trans()
        markings = len(trans)
        for i in range(len(trans)):
            t = trans[i]
            prefix = marking_list[i].get_prefix()
            for j in range(len(self.end)):
                if not self.is_resource[j] and a_matrix[j][t] < 0:
                    value = pmap_v.get(j)
                    sb.append(str(value) + "-" + str(prefix) + "-" + str(markings) + " ")
                    for x in self.parallel_place_pair:
                        try:
                            if int(value) == x:
                                sb.append(str(self.parallel_place_pair[x]) + "-" + str(prefix) + "-" + str(markings) + " ")
                        except (TypeError, ValueError):
                            continue
            if i == len(trans) - 1:
                end_value = pmap_v.get(self.end_place)
                last_prefix = marking_list[-1].get_prefix()
                sb.append(str(end_value) + "-" + str(last_prefix) + "-" + str(markings) + " ")
            markings -= 1
        return "".join(sb)


if __name__ == "__main__":
    print("a_star.py是算法模块，请运行 python_port/run_a_star.py", flush=True)
