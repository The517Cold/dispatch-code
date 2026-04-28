import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from .rl_env_semantics import (
    classify_deadlock_reason,
    enabled_transitions_for_marking,
    format_reason_counts,
    has_over_residence_time,
)


@dataclass
class DeadlockAnalysis:
    # 当前 marking 下，真实环境语义判定为可执行的动作集合。
    enabled_actions: List[int]
    # 第一层硬过滤后保留下来的动作集合。
    safe_actions: List[int]
    # 控制器最终返回给 RL 的动作集合。
    # 注意：当 safe_actions 为空时，第一版会回退到 enabled_actions，避免过度误杀。
    controller_actions: List[int]
    # 被第一层硬过滤挡掉的动作。
    hard_blocked_actions: List[int]
    # 被第二层 lookahead 判成局部高风险的动作。
    soft_risk_actions: List[int]
    # 动作 -> 主原因，仅记录一个主原因，便于日志分析。
    reason_by_action: Dict[int, str]
    # 是否可视为 FBM 候选边界态。
    fbm_candidate: bool
    # 当前状态本身是否已经是真死锁。
    state_deadlock: bool
    # 当前状态死锁原因，例如 no_enabled_transitions / over_residence_time。
    state_deadlock_reason: str
    # 当 safe_actions 被全部筛空时，是否发生了 enabled 回退。
    controller_empty_fallback: bool
    # 当前 controller_actions 的来源，用于区分 rule_safe / lookahead_safe / fallback。
    controller_source: str
    # 当前分析是否实际执行了第二层 lookahead。
    lookahead_ran: bool

    def enabled_count(self) -> int:
        return len(self.enabled_actions)

    def safe_count(self) -> int:
        return len(self.safe_actions)

    def controller_count(self) -> int:
        return len(self.controller_actions)

    def hard_blocked_count(self) -> int:
        return len(self.hard_blocked_actions)

    def soft_risk_count(self) -> int:
        return len(self.soft_risk_actions)

    def blocked_reason_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for reason in self.reason_by_action.values():
            counts[reason] = counts.get(reason, 0) + 1
        return counts


class DeadlockController:
    def __init__(
        self,
        pre: Sequence[Sequence[int]],
        post: Sequence[Sequence[int]],
        end: Sequence[int],
        capacity: Optional[Sequence[int]] = None,
        has_capacity: bool = False,
        transition_flow_allowed: Optional[Sequence[bool]] = None,
        log_path: Optional[str] = None,
        controller_name: str = "deadlock_controller",
        enable_lookahead: bool = True,
        lookahead_depth: int = 2,
        lookahead_width: int = 4,
        lookahead_trigger_safe_limit: int = 4,
        lookahead_trigger_on_fbm: bool = True,
    ):
        # 下面这组参数是测试 deadlock controller 时最常调的参数：
        #
        # enable_lookahead：
        #   是否开启第二层有限深度活性检查。
        #   True 更安全，但通常更慢；False 更快，但只保留第一层一步硬过滤。
        #
        # lookahead_depth：
        #   第二层向前搜索的深度。
        #   越大越容易发现“几步后必死”的动作，但计算成本会明显上升。
        #   设为 0 等价于关闭第二层。
        #
        # lookahead_width：
        #   第二层每层最多保留多少个候选后继继续搜索。
        #   越大越不容易漏掉可行路径，但也越慢。
        #
        # lookahead_trigger_safe_limit：
        #   当第一层筛完后 safe_actions 数量不超过这个阈值时，才触发第二层。
        #   该值越大，第二层触发越频繁。
        #
        # lookahead_trigger_on_fbm：
        #   当第一层已经挡掉一部分动作、当前状态像 FBM 边界态时，是否强制触发第二层。
        #   True 更保守；False 更省时。
        #
        # log_path：
        #   控制器专用日志文件路径，不打印终端时可从这个文件回看控制器行为。
        self.pre = pre
        self.post = post
        self.end = end
        self.capacity = capacity
        self.has_capacity = bool(has_capacity) and capacity is not None
        self.transition_flow_allowed = transition_flow_allowed
        self.controller_name = controller_name
        self.enable_lookahead = bool(enable_lookahead)
        self.lookahead_depth = max(0, int(lookahead_depth))
        self.lookahead_width = max(1, int(lookahead_width))
        self.lookahead_trigger_safe_limit = max(1, int(lookahead_trigger_safe_limit))
        self.lookahead_trigger_on_fbm = bool(lookahead_trigger_on_fbm)
        if log_path is None:
            log_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "results", "deadlock_controller.log")
            )
        self.log_path = log_path
        self._cache_token = object()
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def analyze_state(self, petri_net, marking) -> DeadlockAnalysis:
        # 先看 marking 上是否已经缓存过分析结果。
        # 一个状态在一次 rollout 中可能被重复访问，缓存能减少重复的控制器计算。
        cache = getattr(marking, "_deadlock_controller_cache", None)
        if isinstance(cache, dict):
            cached = cache.get(id(self._cache_token))
            if cached is not None:
                return cached
        # 第一步：拿到真实环境语义下的 enabled 动作集合。
        enabled_actions = enabled_transitions_for_marking(petri_net, marking)
        if not enabled_actions:
            state_deadlock_reason = classify_deadlock_reason(
                marking,
                self.pre,
                self.post,
                capacity=self.capacity,
                has_capacity=self.has_capacity,
                transition_flow_allowed=self.transition_flow_allowed,
            )
            analysis = DeadlockAnalysis(
                enabled_actions=[],
                safe_actions=[],
                controller_actions=[],
                hard_blocked_actions=[],
                soft_risk_actions=[],
                reason_by_action={},
                fbm_candidate=False,
                state_deadlock=True,
                state_deadlock_reason=state_deadlock_reason,
                controller_empty_fallback=False,
                controller_source="state_deadlock",
                lookahead_ran=False,
            )
            self._cache_analysis(marking, analysis)
            return analysis
        state_deadlock_reason = "alive"
        safe_actions: List[int] = []
        hard_blocked_actions: List[int] = []
        reason_by_action: Dict[int, str] = {}
        # 第一层：对每个 enabled 动作做一步模拟。
        # 若一步后立即驻留超时，或一步后立即无路可走，则做硬过滤。
        for action in enabled_actions:
            next_marking = self._simulate_action(petri_net, marking, action)
            reason = self._hard_block_reason(petri_net, next_marking)
            if reason is None:
                safe_actions.append(action)
            else:
                hard_blocked_actions.append(action)
                reason_by_action[action] = reason
        controller_empty_fallback = len(safe_actions) == 0
        # 第一版采取保守 fallback：
        # 即使第一层把 safe_actions 全部筛空，也先退回 enabled_actions，而不是直接阻断。
        controller_actions = enabled_actions.copy() if controller_empty_fallback else safe_actions.copy()
        soft_risk_actions: List[int] = []
        controller_source = "enabled_fallback" if controller_empty_fallback else "rule_safe"
        lookahead_ran = False
        # 第二层：只在满足触发条件时运行小深度活性检查。
        if (not controller_empty_fallback) and self._should_run_lookahead(safe_actions, hard_blocked_actions):
            lookahead_ran = True
            controller_actions, soft_risk_actions = self._apply_lookahead(petri_net, marking, safe_actions, reason_by_action)
            if soft_risk_actions:
                if controller_actions == safe_actions:
                    controller_source = "lookahead_fallback"
                else:
                    controller_source = "lookahead_safe"
        analysis = DeadlockAnalysis(
            enabled_actions=enabled_actions.copy(),
            safe_actions=safe_actions,
            controller_actions=controller_actions,
            hard_blocked_actions=hard_blocked_actions,
            soft_risk_actions=soft_risk_actions,
            reason_by_action=reason_by_action,
            fbm_candidate=len(hard_blocked_actions) > 0 or len(soft_risk_actions) > 0,
            state_deadlock=False,
            state_deadlock_reason=state_deadlock_reason,
            controller_empty_fallback=controller_empty_fallback,
            controller_source=controller_source,
            lookahead_ran=lookahead_ran,
        )
        self._cache_analysis(marking, analysis)
        return analysis

    def log_analysis(self, marking, analysis: DeadlockAnalysis, context: str):
        timestamp = datetime.now().isoformat(timespec="seconds")
        p_info = ",".join(str(v) for v in marking.get_p_info())
        line = (
            "[" + timestamp + "]"
            + " controller=" + self.controller_name
            + " context=" + context
            + " prefix=" + str(marking.get_prefix())
            + " p_info=(" + p_info + ")"
            + " enabled=" + str(analysis.enabled_count())
            + " safe=" + str(analysis.safe_count())
            + " controller_actions=" + str(analysis.controller_count())
            + " controller_source=" + analysis.controller_source
            + " fbm_candidate=" + str(int(analysis.fbm_candidate))
            + " controller_empty_fallback=" + str(int(analysis.controller_empty_fallback))
            + " lookahead_ran=" + str(int(analysis.lookahead_ran))
            + " state_deadlock=" + str(int(analysis.state_deadlock))
            + " state_deadlock_reason=" + str(analysis.state_deadlock_reason)
            + " hard_blocked=" + str(analysis.hard_blocked_count())
            + " soft_risk=" + str(analysis.soft_risk_count())
            + " reason_counts=" + format_reason_counts(analysis.blocked_reason_counts())
        )
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _cache_analysis(self, marking, analysis: DeadlockAnalysis):
        cache = getattr(marking, "_deadlock_controller_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(marking, "_deadlock_controller_cache", cache)
        cache[id(self._cache_token)] = analysis

    def _simulate_action(self, petri_net, marking, action: int):
        current = petri_net.get_marking()
        petri_net.set_marking(marking)
        try:
            return petri_net.launch(action)
        finally:
            petri_net.set_marking(current)

    def _is_goal_marking(self, marking) -> bool:
        p_info = marking.get_p_info()
        for i in range(len(p_info)):
            if self.end[i] == -1:
                continue
            if p_info[i] != self.end[i]:
                return False
        return True

    def _hard_block_reason(self, petri_net, next_marking) -> Optional[str]:
        # 第一层目前只做两类“证据很强”的硬过滤：
        # 1. 一步后驻留时间超限
        # 2. 一步后立即死锁
        if has_over_residence_time(next_marking):
            return "over_residence_time"
        if self._is_goal_marking(next_marking):
            return None
        next_enabled = enabled_transitions_for_marking(petri_net, next_marking)
        if not next_enabled:
            return "immediate_deadlock"
        return None

    def _should_run_lookahead(self, safe_actions: Sequence[int], hard_blocked_actions: Sequence[int]) -> bool:
        # 第二层的触发逻辑：
        # 1. 显式关闭或深度为 0 时，不运行
        # 2. safe_actions 只剩 1 个时，强制运行
        # 3. 若当前像 FBM 边界态且 lookahead_trigger_on_fbm=True，则运行
        # 4. 否则在 safe_actions 数量较少时运行
        if (not self.enable_lookahead) or self.lookahead_depth <= 0:
            return False
        if len(safe_actions) <= 1:
            return True
        if self.lookahead_trigger_on_fbm and len(hard_blocked_actions) > 0:
            return True
        return len(safe_actions) <= self.lookahead_trigger_safe_limit

    def _apply_lookahead(self, petri_net, marking, safe_actions: Sequence[int], reason_by_action: Dict[int, str]) -> Tuple[List[int], List[int]]:
        # 对第一层保留下来的动作逐个做有限深度存活性检查。
        # 如果某动作在限制深度与宽度内完全找不到存活路径，就将其标成 soft_risk。
        if self.lookahead_depth <= 0:
            return list(safe_actions), []
        lookahead_safe: List[int] = []
        soft_risk: List[int] = []
        for action in safe_actions:
            next_marking = self._simulate_action(petri_net, marking, action)
            if self._has_survival_path(
                petri_net,
                next_marking,
                self.lookahead_depth - 1,
                self.lookahead_width,
                set(),
            ):
                lookahead_safe.append(action)
            else:
                soft_risk.append(action)
                reason_by_action[action] = "bounded_liveness_risk"
        if lookahead_safe:
            return lookahead_safe, soft_risk
        # 如果第二层把所有动作都判成高风险，当前版本不直接裁空，仍退回第一层结果。
        return list(safe_actions), soft_risk

    def _has_survival_path(self, petri_net, marking, remaining_depth: int, width: int, visited: set) -> bool:
        # 这里判断的不是“能否到达目标”，而是“在限制深度内是否还存在至少一条继续存活的路径”。
        # 因此它更像局部活性检查，而不是完整求解。
        if has_over_residence_time(marking):
            return False
        if self._is_goal_marking(marking):
            return True
        enabled = enabled_transitions_for_marking(petri_net, marking)
        if not enabled:
            return False
        if remaining_depth <= 0:
            return True
        signature = (self._marking_signature(marking), remaining_depth)
        if signature in visited:
            return False
        visited.add(signature)
        candidates = []
        for action in enabled:
            next_marking = self._simulate_action(petri_net, marking, action)
            candidates.append((self._lookahead_priority(next_marking), next_marking))
        candidates.sort(key=lambda item: item[0])
        for _, next_marking in candidates[:width]:
            if self._has_survival_path(petri_net, next_marking, remaining_depth - 1, width, visited):
                visited.remove(signature)
                return True
        visited.remove(signature)
        return False

    def _lookahead_priority(self, marking):
        # 第二层在限宽展开时，优先保留“更接近目标、时间前缀更小”的状态。
        return (self._goal_distance(marking), marking.get_prefix(), tuple(marking.get_p_info()))

    def _goal_distance(self, marking) -> int:
        p_info = marking.get_p_info()
        dist = 0
        for i in range(len(p_info)):
            if self.end[i] == -1:
                continue
            dist += abs(p_info[i] - self.end[i])
        return dist

    def _marking_signature(self, marking):
        key = [tuple(marking.get_p_info())]
        if hasattr(marking, "curr_delay_t"):
            key.append(tuple(getattr(marking, "curr_delay_t", [])))
        if hasattr(marking, "t_info"):
            key.append(self._serialize_nested(getattr(marking, "t_info", [])))
        if hasattr(marking, "residence_time_info"):
            key.append(self._serialize_nested(getattr(marking, "residence_time_info", [])))
        over = bool(getattr(marking, "over_max_residence_time", False))
        key.append(over)
        return tuple(key)

    def _serialize_nested(self, groups):
        serialized = []
        for group in groups:
            serialized.append(tuple(self._serialize_item(item) for item in group))
        return tuple(serialized)

    def _serialize_item(self, item):
        if hasattr(item, "get_id") and hasattr(item, "timer") and hasattr(item, "residence_time"):
            return (item.get_id(), item.timer, item.residence_time)
        return item
