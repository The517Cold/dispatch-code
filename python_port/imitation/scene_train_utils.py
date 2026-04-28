import os
from typing import Dict, Optional


def append_line(path: Optional[str], line: str):
    print(line, flush=True)
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def state_dict_cpu(module) -> Dict[str, object]:
    out = {}
    for key, value in module.state_dict().items():
        out[key] = value.detach().cpu()
    return out


def budget_for_round(total_budget: int, scene_rounds: int, round_idx: int) -> int:
    total = max(0, int(total_budget))
    rounds = max(1, int(scene_rounds))
    idx = max(0, min(int(round_idx), rounds - 1))
    base = total // rounds
    remainder = total % rounds
    return base + (1 if idx < remainder else 0)


def compute_scene_metrics(eval_summaries):
    total_count = len(eval_summaries)
    success_items = [item for item in eval_summaries if bool(item.get("reach_goal", False))]
    success_count = len(success_items)
    success_rate = float(success_count) / float(total_count) if total_count > 0 else 0.0
    if success_count > 0:
        avg_success_makespan = sum(float(item.get("makespan", 0.0)) for item in success_items) / float(success_count)
        avg_success_trans_count = sum(int(item.get("trans_count", 0)) for item in success_items) / float(success_count)
    else:
        avg_success_makespan = float("inf")
        avg_success_trans_count = float("inf")
    return {
        "success_count": success_count,
        "total_count": total_count,
        "success_rate": success_rate,
        "avg_success_makespan": avg_success_makespan,
        "avg_success_trans_count": avg_success_trans_count,
    }


def is_better_scene_metrics(curr, best):
    if best is None:
        return True
    if float(curr["success_rate"]) != float(best["success_rate"]):
        return float(curr["success_rate"]) > float(best["success_rate"])
    if int(curr["success_count"]) > 0 and int(best["success_count"]) > 0:
        if float(curr["avg_success_makespan"]) != float(best["avg_success_makespan"]):
            return float(curr["avg_success_makespan"]) < float(best["avg_success_makespan"])
        if float(curr["avg_success_trans_count"]) != float(best["avg_success_trans_count"]):
            return float(curr["avg_success_trans_count"]) < float(best["avg_success_trans_count"])
    return False
