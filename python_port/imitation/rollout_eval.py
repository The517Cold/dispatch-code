from typing import Dict, List

import torch

from python_port.imitation.expert_dataset import action_mask_from_marking, goal_distance


def rollout_top1_greedy(
    model,
    encoder,
    petri_net,
    end: List[int],
    pre: List[List[int]],
    max_steps: int,
    device: torch.device,
) -> Dict[str, object]:
    """Evaluate the current policy with greedy top-1 action selection."""
    model.eval()
    work = petri_net.clone()
    capacity = getattr(work, "capacity", None)
    transition_flow_allowed = getattr(work, "transition_flow_allowed", None)
    curr = work.get_marking()
    trans = []
    with torch.no_grad():
        for _ in range(max(1, max_steps)):
            if goal_distance(curr, end) == 0:
                break
            mask = action_mask_from_marking(curr, pre, capacity, transition_flow_allowed).to(device)
            if not bool(mask.any().item()):
                # No enabled transition means the rollout is stuck.
                break
            logits = model(encoder.encode(curr).to(device))
            masked_logits = logits.masked_fill(~mask, -1e9)
            action = int(torch.argmax(masked_logits).item())
            if not work.enable(action):
                break
            nxt = work.launch(action)
            work.set_marking(nxt)
            curr = nxt
            trans.append(action)
            if goal_distance(curr, end) == 0:
                break
    reach_goal = goal_distance(curr, end) == 0
    return {
        "reach_goal": bool(reach_goal),
        "goal_distance": int(goal_distance(curr, end)),
        "policy_trans_sequence": trans,
        "policy_trans_count": len(trans),
        "policy_makespan": float(curr.get_prefix()),
    }
