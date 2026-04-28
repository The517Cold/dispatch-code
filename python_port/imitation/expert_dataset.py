import random
import time
from typing import Dict, List

import torch

from python_port.imitation.data import BCSample
from python_port.petri_net_platform.search.a_star import AStar, CreateEFLine, EvaluationFunction, OpenTable
from python_port.petri_net_platform.search.petri_gcn_models import PetriStateEncoderEnhanced


def goal_distance(marking, end: List[int]) -> int:
    """L1 distance to the target marking, skipping unconstrained places."""
    p_info = marking.get_p_info()
    dist = 0
    for idx, token in enumerate(p_info):
        if end[idx] == -1:
            continue
        dist += abs(token - end[idx])
    return dist


def action_mask_from_marking(marking, pre: List[List[int]], capacity, transition_flow_allowed) -> torch.Tensor:
    """Compute which global transition ids are legal at the current marking."""
    p_info = list(marking.get_p_info())
    trans_count = len(pre[0]) if pre else 0
    if bool(getattr(marking, "over_max_residence_time", False)):
        # Residence-time violation means this marking should not be expanded further.
        return torch.tensor([False] * trans_count, dtype=torch.bool)
    has_capacity = capacity is not None
    flow_allowed = transition_flow_allowed or [True] * trans_count
    mask = []
    for tran in range(trans_count):
        enabled = True
        next_p_info = p_info.copy()
        for place in range(len(pre)):
            next_p_info[place] -= pre[place][tran]
            if next_p_info[place] < 0:
                enabled = False
                break
        if enabled and has_capacity:
            for place in range(len(pre)):
                if capacity[place] < next_p_info[place]:
                    enabled = False
                    break
        if enabled and (not flow_allowed[tran]):
            enabled = False
        mask.append(enabled)
    return torch.tensor(mask, dtype=torch.bool)


def _enabled_actions_by_env(net) -> List[int]:
    count = net.get_trans_count()
    out = []
    for tran in range(count):
        if net.enable(tran):
            out.append(tran)
    return out


def _samples_from_result(
    result,
    encoder,
    end: List[int],
    pre: List[List[int]],
    capacity,
    transition_flow_allowed,
    source_net: str,
    source_type: str,
) -> Dict[str, object]:
    """Convert one expert search result into BC samples step by step."""
    trans = result.get_trans()
    markings = result.get_markings()
    usable_steps = min(len(trans), len(markings))
    samples = []
    invalid = 0
    for step_idx in range(usable_steps):
        marking = markings[step_idx]
        action = int(trans[step_idx])
        mask = action_mask_from_marking(marking, pre, capacity, transition_flow_allowed)
        if action < 0 or action >= mask.numel() or (not bool(mask[action].item())):
            # Expert trajectories are filtered against the current legality mask.
            invalid += 1
            continue
        samples.append(
            BCSample(
                state_features=encoder.encode(marking).cpu(),
                action_mask=mask.cpu(),
                expert_action=action,
                meta={
                    "step_idx": int(step_idx),
                    "prefix_time": float(marking.get_prefix()),
                    "goal_distance": int(goal_distance(marking, end)),
                    "source_net": source_net,
                    "source_type": source_type,
                },
            )
        )
    return {
        "samples": samples,
        "invalid_labels": invalid,
        "steps": len(trans),
    }


def _build_perturbed_net_from_expert(base_net, clean_trans: List[int], perturb_steps: int, rng: random.Random):
    """Start from an expert prefix and inject random enabled actions."""
    if not clean_trans:
        return None
    # Replay a random prefix of expert actions to obtain a valid in-net state.
    replay_steps = rng.randint(0, max(0, len(clean_trans) - 1))
    work = base_net.clone()
    for idx in range(replay_steps):
        action = int(clean_trans[idx])
        if not work.enable(action):
            return None
        nxt = work.launch(action)
        work.set_marking(nxt)
    for _ in range(max(0, perturb_steps)):
        actions = _enabled_actions_by_env(work)
        if not actions:
            return None
        chosen = actions[rng.randint(0, len(actions) - 1)]
        nxt = work.launch(chosen)
        work.set_marking(nxt)
    return work


def _build_open_table_for_context(
    context,
    end: List[int],
    net_for_efline,
    allow_generate_efline: bool,
    efline_expand_nodes: int,
    efline_search_seconds: float,
):
    """Build the optional A* heuristic table needed for expert generation."""
    petri_net_file = context.get("petri_net_file")
    matrix_translator = context.get("matrix_translator")
    p_info = context.get("p_info")
    a_matrix = context.get("a_matrix")
    sets = context.get("sets", {})
    if petri_net_file is None or matrix_translator is None or p_info is None or a_matrix is None:
        return None
    if (not petri_net_file.EFline) and allow_generate_efline:
        is_resource = sets.get("isResource")
        if is_resource is None:
            is_resource = [False] * len(p_info)
        try:
            creator = CreateEFLine(net_for_efline.clone(), end.copy(), p_info, is_resource.copy(), [])
            ef_line = creator.ef_line(
                a_matrix,
                matrix_translator.p_map_v,
                max_expand_nodes=efline_expand_nodes,
                max_search_seconds=efline_search_seconds,
            )
            if ef_line:
                # Cache EF line inside the parsed net so later searches can reuse it.
                petri_net_file.EFline = ef_line
        except BaseException:
            return None
    if not petri_net_file.EFline:
        return None
    try:
        ef = EvaluationFunction(petri_net_file)
        return OpenTable(a_matrix, ef)
    except BaseException:
        return None


def _run_expert_search(
    net,
    end,
    max_expand_nodes: int,
    max_search_seconds: float,
    context=None,
    allow_generate_efline: bool = False,
    efline_expand_nodes: int = 80000,
    efline_search_seconds: float = 10.0,
):
    """Run A* once and return both the result and its search statistics."""
    open_table = None
    if context is not None:
        open_table = _build_open_table_for_context(
            context,
            end,
            net,
            allow_generate_efline=allow_generate_efline,
            efline_expand_nodes=efline_expand_nodes,
            efline_search_seconds=efline_search_seconds,
        )
    search = AStar(
        net,
        end,
        open_table,
        max_expand_nodes=max_expand_nodes,
        max_search_seconds=max_search_seconds,
    )
    result = search.search()
    return result, search.get_extra_info()


def generate_augmented_bc_samples(
    petri_net,
    end: List[int],
    pre: List[List[int]],
    min_delay_p: List[int],
    source_net: str,
    context=None,
    max_expand_nodes: int = 360000,
    max_search_seconds: float = 30.0,
    max_data_gen_seconds: float = 180.0,
    perturb_count: int = 8,
    perturb_steps: int = 2,
    clean_repeat: int = 3,
    allow_generate_efline: bool = True,
    efline_expand_nodes: int = 80000,
    efline_search_seconds: float = 10.0,
    seed: int = 42,
):
    """Generate clean expert samples plus perturbed recovery samples for one net."""
    base_net = petri_net.clone()
    capacity = getattr(base_net, "capacity", None)
    transition_flow_allowed = getattr(base_net, "transition_flow_allowed", None)
    encoder = None
    if context is not None:
        encoder = PetriStateEncoderEnhanced(
            end=end,
            min_delay_p=min_delay_p,
            device=torch.device("cpu"),
            pre=pre,
            post=context.get("post"),
            min_delay_t=context.get("min_delay_t"),
            capacity=context.get("capacity"),
            max_residence_time=context.get("max_residence_time"),
            place_from_places=context.get("place_from_places"),
        )
    if encoder is None:
        encoder = PetriStateEncoderEnhanced(end, min_delay_p, torch.device("cpu"))

    data_gen_start = time.perf_counter()
    clean_result, clean_extra = _run_expert_search(
        base_net.clone(),
        end,
        max_expand_nodes=max_expand_nodes,
        max_search_seconds=max_search_seconds,
        context=context,
        allow_generate_efline=allow_generate_efline,
        efline_expand_nodes=efline_expand_nodes,
        efline_search_seconds=efline_search_seconds,
    )
    if clean_result is None:
        return {
            "samples": [],
            "stats": {
                "clean_has_solution": 0,
                "clean_extend_count": clean_extra.get("extendMarkingCount", 0),
                "clean_terminated_by_budget": 1 if clean_extra.get("terminatedByMaxExpandNodes") else 0,
                "clean_terminated_by_time": 1 if clean_extra.get("terminatedByMaxSearchSeconds") else 0,
                "clean_steps": 0,
                "invalid_labels": 0,
                "perturb_success": 0,
                "perturb_attempts": perturb_count,
            },
            "clean_result": None,
        }

    clean_pack = _samples_from_result(
        clean_result,
        encoder,
        end,
        pre,
        capacity,
        transition_flow_allowed,
        source_net=source_net,
        source_type="clean_astar",
    )
    clean_repeat = max(1, int(clean_repeat))
    # Repeat the clean optimal trajectory so supervised training keeps a strong anchor.
    samples = clean_pack["samples"] * clean_repeat
    invalid_labels = clean_pack["invalid_labels"]
    clean_trans = clean_result.get_trans()

    perturb_success = 0
    rng = random.Random(seed)
    for attempt_idx in range(max(0, perturb_count)):
        if (time.perf_counter() - data_gen_start) >= max_data_gen_seconds:
            break
        perturbed_net = _build_perturbed_net_from_expert(base_net, clean_trans, perturb_steps, rng)
        if perturbed_net is None:
            continue
        perturbed_result, _ = _run_expert_search(
            perturbed_net,
            end,
            max_expand_nodes=max_expand_nodes,
            max_search_seconds=max_search_seconds,
            context=context,
            allow_generate_efline=False,
        )
        if perturbed_result is None:
            continue
        perturb_success += 1
        pack = _samples_from_result(
            perturbed_result,
            encoder,
            end,
            pre,
            capacity,
            transition_flow_allowed,
            source_net=source_net,
            source_type="perturbed_astar_" + str(attempt_idx),
        )
        invalid_labels += pack["invalid_labels"]
        # Perturbed trajectories teach the policy how to recover from off-expert states.
        samples.extend(pack["samples"])

    return {
        "samples": samples,
        "stats": {
            "clean_has_solution": 1,
            "clean_extend_count": clean_extra.get("extendMarkingCount", 0),
            "clean_terminated_by_budget": 0,
            "clean_terminated_by_time": 0,
            "clean_steps": clean_pack["steps"],
            "invalid_labels": invalid_labels,
            "perturb_success": perturb_success,
            "perturb_attempts": perturb_count,
        },
        "clean_result": clean_result,
    }
