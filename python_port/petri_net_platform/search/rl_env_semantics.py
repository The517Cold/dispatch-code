from typing import Dict, List, Optional, Sequence

import torch


def has_over_residence_time(marking) -> bool:
    judge = getattr(marking, "is_over_residece_time", None)
    if callable(judge):
        try:
            return bool(judge())
        except BaseException:
            pass
    return bool(getattr(marking, "over_max_residence_time", False))


def enabled_transitions_for_marking(petri_net, marking) -> List[int]:
    current = petri_net.get_marking()
    petri_net.set_marking(marking)
    try:
        enabled = []
        for tran in range(petri_net.get_trans_count()):
            if petri_net.enable(tran):
                enabled.append(tran)
        return enabled
    finally:
        petri_net.set_marking(current)


def enabled_mask_for_marking(petri_net, marking, device: torch.device) -> torch.Tensor:
    enabled = enabled_transitions_for_marking(petri_net, marking)
    mask = torch.zeros(petri_net.get_trans_count(), dtype=torch.bool, device=device)
    if enabled:
        mask[enabled] = True
    return mask


def classify_deadlock_reason(
    marking,
    pre: Sequence[Sequence[int]],
    post: Sequence[Sequence[int]],
    capacity: Optional[Sequence[int]] = None,
    has_capacity: bool = False,
    transition_flow_allowed: Optional[Sequence[bool]] = None,
) -> str:
    if has_over_residence_time(marking):
        return "over_residence_time"
    if not pre or not pre[0]:
        return "no_enabled_transitions"
    p_info = list(marking.get_p_info())
    transition_count = len(pre[0])
    token_ready = False
    capacity_blocked = False
    flow_blocked = False
    for tran in range(transition_count):
        enough_tokens = True
        next_p_info = p_info.copy()
        for place in range(len(pre)):
            next_p_info[place] -= pre[place][tran]
            if next_p_info[place] < 0:
                enough_tokens = False
                break
        if not enough_tokens:
            continue
        token_ready = True
        for place in range(len(post)):
            next_p_info[place] += post[place][tran]
        if has_capacity and capacity is not None:
            for place in range(len(next_p_info)):
                if next_p_info[place] > capacity[place]:
                    capacity_blocked = True
                    break
        if transition_flow_allowed is not None and tran < len(transition_flow_allowed) and (not transition_flow_allowed[tran]):
            flow_blocked = True
    if not token_ready:
        return "no_enabled_transitions"
    if capacity_blocked and flow_blocked:
        return "capacity_or_flow_blocked"
    if capacity_blocked:
        return "capacity_blocked"
    if flow_blocked:
        return "flow_blocked"
    return "no_enabled_transitions"


def make_stop_info(reason: str, steps: int, step_limit: int, deadlock_reason: Optional[str] = None) -> Dict[str, object]:
    return {
        "reason": reason,
        "steps": int(steps),
        "step_limit": int(step_limit),
        "deadlock_reason": deadlock_reason,
    }


def stop_info_label(stop_info: Dict[str, object]) -> str:
    reason = str(stop_info.get("reason", "unknown"))
    if reason == "deadlock":
        deadlock_reason = stop_info.get("deadlock_reason") or "no_enabled_transitions"
        return "deadlock:" + str(deadlock_reason)
    return reason


def describe_stop_info(stop_info: Dict[str, object]) -> str:
    reason = str(stop_info.get("reason", "unknown"))
    steps = int(stop_info.get("steps", 0))
    step_limit = int(stop_info.get("step_limit", 0))
    if reason == "goal":
        return "goal steps=" + str(steps)
    if reason == "deadlock":
        deadlock_reason = stop_info.get("deadlock_reason") or "no_enabled_transitions"
        return "deadlock(" + str(deadlock_reason) + ") steps=" + str(steps)
    if reason == "step_limit":
        return "step_limit steps=" + str(steps) + "/" + str(step_limit)
    if reason == "invalid_action_fallback":
        return "invalid_action_fallback steps=" + str(steps)
    return reason + " steps=" + str(steps)


def format_reason_counts(counts: Dict[str, int]) -> str:
    if not counts:
        return "-"
    parts = []
    for key in sorted(counts):
        parts.append(str(key) + ":" + str(counts[key]))
    return ",".join(parts)
