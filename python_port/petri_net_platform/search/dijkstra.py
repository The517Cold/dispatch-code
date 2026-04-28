import os
import sys
import time
from typing import Callable, Dict, Optional

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from python_port.petri_net_platform.search.a_star import AStar, OpenTable


class Dijkstra(AStar):
    def __init__(
        self,
        petri_net,
        end,
        use_greedy_upper_bound=True,
        max_search_seconds=None,
        progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
        progress_interval_seconds: float = 1.0,
    ):
        # Dijkstra 本质上就是不带启发式的 A*，这里直接复用其状态剪枝和约束处理能力。
        super().__init__(
            petri_net=petri_net,
            end=end,
            open_table=OpenTable(),
            use_greedy_upper_bound=use_greedy_upper_bound,
            max_search_seconds=max_search_seconds,
        )
        self.progress_callback = progress_callback
        self.progress_interval_seconds = progress_interval_seconds
        self.poll_count = 0
        self.max_open_size = 0
        self._last_progress_emit = None

    def search(self):
        self.search_start_time = time.perf_counter()
        self._last_progress_emit = time.perf_counter()
        self._emit_progress("started", None)
        self._seed_upper_bound()
        self._put_init_marking()
        return self._find()

    def _find(self):
        while not self.open_table.is_empty():
            self.max_open_size = max(self.max_open_size, self.open_table.size())
            if self._is_timeout():
                self.timeout_triggered = True
                self._emit_progress("timeout", None)
                break
            curr, curr_key = self.open_table.poll(self.close)
            if curr is None:
                break
            self.poll_count += 1
            self._emit_progress("poll", curr)
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
                    self._emit_progress("goal", curr)
                    # Dijkstra 的 open 表按 g 值升序，第一次弹出的目标状态即可认为是当前最优解。
                    return self._make_result(candidate)
                continue
            self.petri_net.set_marking(curr)
            pre_path = self.path.get(curr_key)
            enabled_trans = self._get_enabled_transitions()
            if not enabled_trans:
                self.deadlock_pruned_count += 1
                continue
            for tran in enabled_trans:
                next_marking = self.petri_net.launch(tran)
                self.extend_marking_count += 1
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
                    new_path = self._build_next_path(pre_path, tran, next_marking)
                    self.path[next_key] = new_path
                    self.close[next_key] = next_marking.get_prefix()
                    self._register_dominance(next_marking)
                    self.open_table.offer(next_marking, next_key, self.close)
        if self.best_path_link is not None:
            self._emit_progress("best_effort", None)
            return self._make_result(self.best_path_link)
        if self.best_result is not None:
            self._emit_progress("best_effort", None)
        return self.best_result

    def get_extra_info(self):
        extra = super().get_extra_info()
        extra["pollCount"] = self.poll_count
        extra["maxOpenSize"] = self.max_open_size
        return extra

    def _build_next_path(self, pre_path, tran, next_marking):
        # 单独拆出来，便于 Dijkstra 保持与 A* 一致的结果结构。
        from python_port.petri_net_platform.utils.low_space_link import LowSpaceLink, TranLink
        return LowSpaceLink(TranLink(tran, pre_path.tran_link), next_marking)

    def _emit_progress(self, stage, curr):
        # 实时进度输出只在达到时间间隔或关键阶段时触发，避免终端刷屏过快。
        if self.progress_callback is None:
            return
        now = time.perf_counter()
        is_key_stage = stage in {"started", "goal", "timeout", "best_effort"}
        if not is_key_stage and self._last_progress_emit is not None:
            if now - self._last_progress_emit < self.progress_interval_seconds:
                return
        self._last_progress_emit = now
        payload = {
            "stage": stage,
            "elapsed_seconds": None if self.search_start_time is None else now - self.search_start_time,
            "current_cost": None if curr is None else curr.get_prefix(),
            "best_upper_bound": None if self.best_makespan == 2 ** 31 - 1 else self.best_makespan,
            "poll_count": self.poll_count,
            "extend_marking_count": self.extend_marking_count,
            "open_size": self.open_table.size(),
            "max_open_size": self.max_open_size,
            "deadlock_pruned_count": self.deadlock_pruned_count,
            "residence_pruned_count": self.residence_pruned_count,
            "dominance_pruned_count": self.dominance_pruned_count,
            "bound_pruned_count": self.bound_pruned_count,
        }
        self.progress_callback(payload)
