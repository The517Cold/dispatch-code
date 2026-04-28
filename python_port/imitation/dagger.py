import os
import random
from statistics import median
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, random_split

from python_port.imitation.data import BCDataset, bc_collate_fn
from python_port.imitation.expert_dataset import (
    _run_expert_search,
    _samples_from_result,
    action_mask_from_marking,
    generate_augmented_bc_samples,
    goal_distance,
)
from python_port.imitation.rollout_eval import rollout_top1_greedy
from python_port.imitation.scene_train_utils import (
    append_line,
    budget_for_round,
    compute_scene_metrics,
    is_better_scene_metrics,
    state_dict_cpu,
)
from python_port.imitation.trainer import BCTrainer, BCTrainerConfig, is_better_rollout
from python_port.petri_net_io.utils.checkpoint_selector import load_compatible_state
from python_port.petri_net_io.utils.net_loader import build_ttpn_with_residence, load_petri_net_context
from python_port.petri_net_platform.search.petri_gcn_models import PetriNetGCNEnhanced, PetriStateEncoderEnhanced
from python_port.scene_utils import list_dash_net_files


DEFAULT_DAGGER_INIT_EPOCHS = 20
DEFAULT_DAGGER_SCENE_ROUNDS = 3
DEFAULT_DAGGER_ROUNDS = 4
DEFAULT_DAGGER_ROUND_EPOCHS = 6
DEFAULT_DAGGER_ROLLOUTS_PER_ROUND = 4
DEFAULT_DAGGER_QUERY_STATES_PER_ROUND = 8
DEFAULT_DAGGER_QUERY_TAIL_STEPS = 3
DEFAULT_DAGGER_QUERY_LABEL_HORIZON = 6
DEFAULT_DAGGER_ROLLOUT_EPSILON = 0.10


def _split(samples, seed):
    ds = BCDataset(samples)
    if len(ds) < 2:
        return ds, ds
    train_size = max(1, min(len(ds) - 1, int(len(ds) * 0.8)))
    val_size = len(ds) - train_size
    return random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(seed))


def _build_model(context):
    return PetriNetGCNEnhanced(
        context["pre"],
        context["post"],
        128,
        32,
        4,
        end=context.get("end"),
        min_delay_p=context.get("min_delay_p"),
        min_delay_t=context.get("min_delay_t"),
        capacity=context.get("capacity"),
        max_residence_time=context.get("max_residence_time"),
        place_from_places=context.get("place_from_places"),
    )


def _build_encoder(context, device: str):
    return PetriStateEncoderEnhanced(
        end=context["end"],
        min_delay_p=context["min_delay_p"],
        device=torch.device(device),
        pre=context["pre"],
        post=context["post"],
        min_delay_t=context.get("min_delay_t"),
        capacity=context.get("capacity"),
        max_residence_time=context.get("max_residence_time"),
        place_from_places=context.get("place_from_places"),
    )


def _marking_signature(marking) -> str:
    parts = [tuple(marking.get_p_info())]
    if hasattr(marking, "curr_delay_t") and getattr(marking, "curr_delay_t"):
        parts.append(tuple(int(v) for v in marking.curr_delay_t))
    if hasattr(marking, "t_info"):
        try:
            parts.append(tuple(tuple(int(v) for v in q) for q in marking.t_info))
        except BaseException:
            pass
    if hasattr(marking, "residence_time_info"):
        try:
            parts.append(tuple(tuple(int(v) for v in q) for q in marking.residence_time_info))
        except BaseException:
            pass
    parts.append(int(bool(getattr(marking, "over_max_residence_time", False))))
    return repr(parts)


def _rollout_episode(model, encoder, petri_net, end, pre, max_steps, epsilon, rng, device):
    net = petri_net.clone()
    curr = net.get_marking()
    initial_marking = curr.clone()
    capacity = getattr(net, "capacity", None)
    transition_flow_allowed = getattr(net, "transition_flow_allowed", None)
    trajectory = []
    stop_reason = "step_limit"
    with torch.no_grad():
        for step_idx in range(max(1, max_steps)):
            if goal_distance(curr, end) == 0:
                stop_reason = "goal"
                break
            mask = action_mask_from_marking(curr, pre, capacity, transition_flow_allowed)
            enabled = torch.nonzero(mask, as_tuple=False).flatten()
            if enabled.numel() == 0:
                stop_reason = "no_enabled_actions"
                break
            logits = model(encoder.encode(curr).to(device))
            masked_logits = logits.masked_fill(~mask.to(device), -1e9)
            if rng.random() < epsilon:
                action = int(enabled[rng.randint(0, enabled.numel() - 1)].item())
            else:
                action = int(torch.argmax(masked_logits).item())
            if not net.enable(action):
                stop_reason = "env_reject"
                break
            state_for_sample = curr.clone()
            next_marking = net.launch(action)
            net.set_marking(next_marking)
            trajectory.append((state_for_sample, action, step_idx))
            curr = next_marking
        else:
            stop_reason = "step_limit"
    if goal_distance(curr, end) == 0:
        stop_reason = "goal"
    return {
        "trajectory": trajectory,
        "initial_marking": initial_marking,
        "reach_goal": goal_distance(curr, end) == 0,
        "goal_distance": int(goal_distance(curr, end)),
        "makespan": float(curr.get_prefix()),
        "trans_count": len(trajectory),
        "trans_sequence": [x[1] for x in trajectory],
        "stop_reason": stop_reason,
    }


def _select_query_markings(episodes, query_states_per_round: int, query_tail_steps: int):
    selected = []
    seen = set()
    for episode in episodes:
        if episode.get("reach_goal"):
            continue
        trajectory = episode.get("trajectory", [])
        if not trajectory:
            signature = _marking_signature(episode["initial_marking"])
            if signature not in seen:
                selected.append(episode["initial_marking"].clone())
                seen.add(signature)
            if len(selected) >= query_states_per_round:
                break
            continue
        tail = trajectory[-max(1, query_tail_steps) :]
        for state, _, _ in reversed(tail):
            signature = _marking_signature(state)
            if signature in seen:
                continue
            selected.append(state.clone())
            seen.add(signature)
            if len(selected) >= query_states_per_round:
                break
        if len(selected) >= query_states_per_round:
            break
    return selected


def _label_query_markings(
    query_markings,
    base_net,
    encoder,
    end,
    pre,
    capacity,
    transition_flow_allowed,
    source_net,
    scene_round,
    local_query_round,
    context,
    max_expand_nodes,
    max_search_seconds,
    label_horizon,
):
    aggregated = []
    invalid_labels = 0
    solved_queries = 0
    for query_idx, marking in enumerate(query_markings):
        work = base_net.clone()
        work.set_marking(marking.clone())
        result, _ = _run_expert_search(
            work,
            end,
            max_expand_nodes=max_expand_nodes,
            max_search_seconds=max_search_seconds,
            context=context,
            allow_generate_efline=False,
        )
        if result is None:
            continue
        solved_queries += 1
        pack = _samples_from_result(
            result,
            encoder,
            end,
            pre,
            capacity,
            transition_flow_allowed,
            source_net=source_net,
            source_type="dagger_scene_round_"
            + str(scene_round)
            + "_query_round_"
            + str(local_query_round)
            + "_query_"
            + str(query_idx + 1),
        )
        invalid_labels += int(pack.get("invalid_labels", 0))
        samples = pack.get("samples", [])
        if label_horizon > 0:
            samples = samples[:label_horizon]
        aggregated.extend(samples)
    return {
        "samples": aggregated,
        "invalid_labels": invalid_labels,
        "solved_queries": solved_queries,
        "query_count": len(query_markings),
    }


def _make_rollout_fn(model, encoder, petri_net, end, pre, max_steps, device):
    def rollout_fn():
        return rollout_top1_greedy(
            model=model,
            encoder=encoder,
            petri_net=petri_net,
            end=end,
            pre=pre,
            max_steps=max_steps,
            device=torch.device(device),
        )

    return rollout_fn


def _fit_with_samples(
    model,
    samples,
    device,
    checkpoint_path,
    progress_log_path,
    log_prefix,
    epochs,
    rollout_every_n_epochs,
    seed,
    rollout_fn,
):
    train_set, val_set = _split(samples, seed=seed)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, collate_fn=bc_collate_fn)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, collate_fn=bc_collate_fn)
    cfg = BCTrainerConfig(
        epochs=epochs,
        lr=3e-4,
        weight_decay=1e-5,
        label_smoothing=0.0,
        device=device,
        checkpoint_path=checkpoint_path,
        rollout_every_n_epochs=max(1, rollout_every_n_epochs),
        log_interval=max(1, epochs // 4),
        progress_log_path=progress_log_path,
        log_prefix=log_prefix,
    )
    trainer = BCTrainer(model, cfg)
    return trainer.fit(train_loader, val_loader, rollout_fn=rollout_fn)


def _evaluate_scene_policy(entries, model_state, device: str):
    eval_summaries = []
    for entry in entries:
        context = entry["context"]
        model = _build_model(context)
        if model_state:
            load_compatible_state(model, model_state)
        model.to(torch.device(device))
        encoder = _build_encoder(context, device)
        rollout = rollout_top1_greedy(
            model=model,
            encoder=encoder,
            petri_net=entry["petri_net_template"].clone(),
            end=context["end"],
            pre=context["pre"],
            max_steps=entry["rollout_max_steps"],
            device=torch.device(device),
        )
        eval_summaries.append(
            {
                "net_name": entry["net_name"],
                "reach_goal": bool(rollout.get("reach_goal", False)),
                "goal_distance": int(rollout.get("goal_distance", -1)),
                "trans_count": int(rollout.get("policy_trans_count", 0)),
                "trans_sequence": rollout.get("policy_trans_sequence", []),
                "makespan": float(rollout.get("policy_makespan", -1.0)),
            }
        )
    return eval_summaries, compute_scene_metrics(eval_summaries)


def pretrain_across_nets_dagger_lite(
    base_dir: str,
    resources_dir: str,
    shared_checkpoint_path: str,
    net_files: Optional[List[str]] = None,
    init_checkpoint_path: str = "",
    init_epochs: int = DEFAULT_DAGGER_INIT_EPOCHS,
    scene_rounds: int = DEFAULT_DAGGER_SCENE_ROUNDS,
    dagger_rounds: int = DEFAULT_DAGGER_ROUNDS,
    round_epochs: int = DEFAULT_DAGGER_ROUND_EPOCHS,
    rollout_episodes_per_round: int = DEFAULT_DAGGER_ROLLOUTS_PER_ROUND,
    rollout_epsilon: float = DEFAULT_DAGGER_ROLLOUT_EPSILON,
    query_states_per_round: int = DEFAULT_DAGGER_QUERY_STATES_PER_ROUND,
    query_tail_steps: int = DEFAULT_DAGGER_QUERY_TAIL_STEPS,
    query_label_horizon: int = DEFAULT_DAGGER_QUERY_LABEL_HORIZON,
    max_expand_nodes: int = 360000,
    max_search_seconds: float = 120.0,
    max_data_gen_seconds: float = 300.0,
    initial_perturb_count: int = 2,
    initial_perturb_steps: int = 1,
    clean_repeat: int = 3,
    allow_generate_efline: bool = True,
    efline_expand_nodes: int = 80000,
    efline_search_seconds: float = 10.0,
    query_expand_nodes: int = 80000,
    query_search_seconds: float = 20.0,
    device_override: str = "",
    seed: int = 42,
    net_limit: int = 0,
    output_progress_path: Optional[str] = None,
) -> Dict[str, object]:
    selected_net_files = list(net_files) if net_files is not None else list_dash_net_files(resources_dir)
    if net_limit > 0:
        selected_net_files = selected_net_files[:net_limit]
    if not selected_net_files:
        raise RuntimeError("no digit-dash .txt net files found in " + resources_dir)
    os.makedirs(os.path.dirname(shared_checkpoint_path), exist_ok=True)
    if output_progress_path:
        os.makedirs(os.path.dirname(output_progress_path), exist_ok=True)
        with open(output_progress_path, "w", encoding="utf-8") as f:
            f.write("# dagger_lite_pretrain: 场景内多轮循环训练\n")
            f.write("# nets_total:" + str(len(selected_net_files)) + "\n")
    device = device_override if device_override else ("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(seed)

    entries = []
    generation_logs = []
    successful_expert_steps: List[int] = []
    for net_idx, net_path in enumerate(selected_net_files):
        context = load_petri_net_context(net_path)
        pre = context["pre"]
        end = context["end"]
        min_delay_p = context["min_delay_p"]
        petri_net = build_ttpn_with_residence(context)
        net_name = os.path.splitext(os.path.basename(net_path))[0]
        place_count = len(pre)
        trans_count = len(pre[0]) if pre else 0
        complexity = place_count * trans_count
        effective_perturb_count = int(initial_perturb_count)
        if complexity >= 900 and effective_perturb_count > 2:
            effective_perturb_count = 2

        append_line(
            output_progress_path,
            "[DAGGER] net=" + net_name + " stage=seed_data begin index=" + str(net_idx + 1) + "/" + str(len(selected_net_files)),
        )
        generated = generate_augmented_bc_samples(
            petri_net=petri_net,
            end=end,
            pre=pre,
            min_delay_p=min_delay_p,
            source_net=net_name,
            context=context,
            max_expand_nodes=max_expand_nodes,
            max_search_seconds=max_search_seconds,
            max_data_gen_seconds=max_data_gen_seconds,
            perturb_count=effective_perturb_count,
            perturb_steps=initial_perturb_steps,
            clean_repeat=clean_repeat,
            allow_generate_efline=allow_generate_efline,
            efline_expand_nodes=efline_expand_nodes,
            efline_search_seconds=efline_search_seconds,
            seed=seed,
        )
        samples = list(generated.get("samples", []))
        if len(samples) < 2:
            generation_logs.append({"net_name": net_name, "status": "skip_insufficient_seed_samples", "seed_samples": len(samples)})
            append_line(output_progress_path, "[DAGGER] net=" + net_name + " stage=skip reason=insufficient_seed_samples")
            continue

        clean_result = generated.get("clean_result")
        expert_steps = len(clean_result.get_trans()) if clean_result is not None else 64
        successful_expert_steps.append(expert_steps)
        rollout_max_steps = max(1, expert_steps * 2)
        entries.append(
            {
                "net_name": net_name,
                "context": context,
                "petri_net_template": petri_net,
                "samples": samples,
                "expert_steps": expert_steps,
                "rollout_max_steps": rollout_max_steps,
                "seed_samples": len(samples),
                "used_max_expand_nodes": max_expand_nodes,
                "effective_perturb_count": effective_perturb_count,
                "max_search_seconds": max_search_seconds,
                "max_data_gen_seconds": max_data_gen_seconds,
                "invalid_labels": int(generated.get("stats", {}).get("invalid_labels", 0)),
                "total_query_states": 0,
                "total_solved_queries": 0,
                "total_added_samples": 0,
                "initialized": False,
                "best_rollout": None,
            }
        )
        generation_logs.append(
            {
                "net_name": net_name,
                "status": "seed_ready",
                "seed_samples": len(samples),
                "expert_steps": expert_steps,
                "effective_perturb_count": effective_perturb_count,
                "invalid_labels": int(generated.get("stats", {}).get("invalid_labels", 0)),
            }
        )

    if not entries:
        raise RuntimeError("no nets with sufficient DAgger seed samples were generated")

    current_model_state = {}
    init_source = "scratch"
    if os.path.exists(shared_checkpoint_path):
        loaded = torch.load(shared_checkpoint_path, map_location="cpu")
        current_model_state = loaded.get("model_state", {})
        init_source = "existing_shared_checkpoint"
    elif init_checkpoint_path and os.path.exists(init_checkpoint_path):
        loaded = torch.load(init_checkpoint_path, map_location="cpu")
        current_model_state = loaded.get("model_state", {})
        init_source = "init_checkpoint"

    scene_rounds = max(1, int(scene_rounds))
    append_line(
        output_progress_path,
        "dagger_scene_schedule scene_rounds="
        + str(scene_rounds)
        + " total_query_rounds_per_net="
        + str(dagger_rounds)
        + " per_scene_round_query_plan="
        + str([budget_for_round(dagger_rounds, scene_rounds, i) for i in range(scene_rounds)])
        + " init_epochs="
        + str(init_epochs)
        + " round_epochs="
        + str(round_epochs)
        + " init_source="
        + init_source,
    )

    visit_summaries = []
    round_metrics = []
    best_scene_metrics = None
    best_round = 0
    best_round_order = []
    best_eval_summaries = []
    best_checkpoint_updates = 0

    for scene_round_idx in range(scene_rounds):
        local_query_rounds = budget_for_round(dagger_rounds, scene_rounds, scene_round_idx)
        round_rng = random.Random(seed + scene_round_idx)
        round_entries = list(entries)
        round_rng.shuffle(round_entries)
        round_order = [entry["net_name"] for entry in round_entries]
        append_line(
            output_progress_path,
            "dagger_scene_round_begin round="
            + str(scene_round_idx + 1)
            + "/"
            + str(scene_rounds)
            + " local_query_rounds="
            + str(local_query_rounds)
            + " order="
            + str(round_order),
        )

        for visit_index, entry in enumerate(round_entries):
            context = entry["context"]
            model = _build_model(context)
            if current_model_state:
                load_compatible_state(model, current_model_state)
            model.to(torch.device(device))
            encoder = _build_encoder(context, device)
            petri_net = entry["petri_net_template"].clone()
            rollout_fn = _make_rollout_fn(
                model=model,
                encoder=encoder,
                petri_net=petri_net,
                end=context["end"],
                pre=context["pre"],
                max_steps=entry["rollout_max_steps"],
                device=device,
            )
            append_line(
                output_progress_path,
                "[DAGGER] net="
                + entry["net_name"]
                + " scene_round="
                + str(scene_round_idx + 1)
                + " visit="
                + str(visit_index + 1)
                + "/"
                + str(len(round_entries))
                + " samples="
                + str(len(entry["samples"]))
                + " initialized="
                + ("1" if entry["initialized"] else "0"),
            )

            if not entry["initialized"]:
                init_fit = _fit_with_samples(
                    model=model,
                    samples=entry["samples"],
                    device=device,
                    checkpoint_path="",
                    progress_log_path=output_progress_path,
                    log_prefix="[DAGGER-INIT]",
                    epochs=max(1, init_epochs),
                    rollout_every_n_epochs=max(1, init_epochs // 2),
                    seed=seed + scene_round_idx * 1000 + visit_index,
                    rollout_fn=rollout_fn,
                )
                entry["initialized"] = True
                if is_better_rollout(init_fit.get("best_rollout"), entry["best_rollout"]):
                    entry["best_rollout"] = init_fit.get("best_rollout")

            for local_round_idx in range(local_query_rounds):
                episodes = []
                for _ in range(max(1, rollout_episodes_per_round)):
                    episodes.append(
                        _rollout_episode(
                            model=model,
                            encoder=encoder,
                            petri_net=petri_net,
                            end=context["end"],
                            pre=context["pre"],
                            max_steps=entry["rollout_max_steps"],
                            epsilon=rollout_epsilon,
                            rng=rng,
                            device=torch.device(device),
                        )
                    )
                query_markings = _select_query_markings(
                    episodes,
                    query_states_per_round=max(1, query_states_per_round),
                    query_tail_steps=max(1, query_tail_steps),
                )
                query_pack = _label_query_markings(
                    query_markings=query_markings,
                    base_net=petri_net,
                    encoder=encoder,
                    end=context["end"],
                    pre=context["pre"],
                    capacity=getattr(petri_net, "capacity", None),
                    transition_flow_allowed=getattr(petri_net, "transition_flow_allowed", None),
                    source_net=entry["net_name"],
                    scene_round=scene_round_idx + 1,
                    local_query_round=local_round_idx + 1,
                    context=context,
                    max_expand_nodes=query_expand_nodes,
                    max_search_seconds=query_search_seconds,
                    label_horizon=query_label_horizon,
                )
                new_samples = query_pack["samples"]
                entry["total_query_states"] += int(query_pack["query_count"])
                entry["total_solved_queries"] += int(query_pack["solved_queries"])
                entry["invalid_labels"] += int(query_pack["invalid_labels"])
                append_line(
                    output_progress_path,
                    "[DAGGER] net="
                    + entry["net_name"]
                    + " scene_round="
                    + str(scene_round_idx + 1)
                    + " local_query_round="
                    + str(local_round_idx + 1)
                    + "/"
                    + str(local_query_rounds)
                    + " queried_states="
                    + str(query_pack["query_count"])
                    + " solved_queries="
                    + str(query_pack["solved_queries"])
                    + " new_samples="
                    + str(len(new_samples)),
                )
                if not new_samples:
                    continue
                entry["samples"].extend(new_samples)
                entry["total_added_samples"] += len(new_samples)
                round_fit = _fit_with_samples(
                    model=model,
                    samples=entry["samples"],
                    device=device,
                    checkpoint_path="",
                    progress_log_path=output_progress_path,
                    log_prefix="[DAGGER-ROUND]",
                    epochs=max(1, round_epochs),
                    rollout_every_n_epochs=max(1, round_epochs),
                    seed=seed + scene_round_idx * 1000 + visit_index * 100 + local_round_idx,
                    rollout_fn=rollout_fn,
                )
                if is_better_rollout(round_fit.get("best_rollout"), entry["best_rollout"]):
                    entry["best_rollout"] = round_fit.get("best_rollout")

            current_model_state = state_dict_cpu(model)
            visit_summaries.append(
                {
                    "scene_round": scene_round_idx + 1,
                    "visit_index": visit_index + 1,
                    "net_name": entry["net_name"],
                    "samples": len(entry["samples"]),
                    "seed_samples": entry["seed_samples"],
                    "added_samples": entry["total_added_samples"],
                    "query_states": entry["total_query_states"],
                    "solved_queries": entry["total_solved_queries"],
                    "local_query_rounds": local_query_rounds,
                    "best_rollout": entry["best_rollout"] or {},
                }
            )
            append_line(
                output_progress_path,
                "[DAGGER] net="
                + entry["net_name"]
                + " scene_round="
                + str(scene_round_idx + 1)
                + " stage=done aggregated_samples="
                + str(len(entry["samples"]))
                + " best_goal="
                + ("1" if bool((entry["best_rollout"] or {}).get("reach_goal", False)) else "0")
                + " best_makespan="
                + str((entry["best_rollout"] or {}).get("policy_makespan", -1)),
            )

        eval_summaries, scene_metrics = _evaluate_scene_policy(entries, current_model_state, device)
        scene_metrics["round"] = scene_round_idx + 1
        scene_metrics["order"] = round_order
        round_metrics.append(scene_metrics)
        append_line(
            output_progress_path,
            "dagger_scene_round_eval round="
            + str(scene_round_idx + 1)
            + " success_rate="
            + format(scene_metrics["success_rate"], ".4f")
            + " success_count="
            + str(scene_metrics["success_count"])
            + "/"
            + str(scene_metrics["total_count"])
            + " avg_success_makespan="
            + ("inf" if scene_metrics["success_count"] == 0 else format(scene_metrics["avg_success_makespan"], ".4f"))
            + " avg_success_trans_count="
            + ("inf" if scene_metrics["success_count"] == 0 else format(scene_metrics["avg_success_trans_count"], ".4f")),
        )
        if is_better_scene_metrics(scene_metrics, best_scene_metrics):
            best_scene_metrics = dict(scene_metrics)
            best_round = scene_round_idx + 1
            best_round_order = list(round_order)
            best_eval_summaries = eval_summaries
            best_checkpoint_updates += 1
            torch.save(
                {
                    "model_state": current_model_state,
                    "il_method": "dagger_lite",
                    "scene_rounds": scene_rounds,
                    "best_round": best_round,
                    "best_scene_metrics": best_scene_metrics,
                    "best_round_order": best_round_order,
                    "round_metrics": round_metrics,
                    "visit_summaries": visit_summaries,
                    "best_eval_summaries": best_eval_summaries,
                    "scene_ref_expert_steps": int(median(successful_expert_steps)) if successful_expert_steps else 0,
                },
                shared_checkpoint_path,
            )
            append_line(
                output_progress_path,
                "dagger_scene_checkpoint_update round="
                + str(best_round)
                + " success_rate="
                + format(best_scene_metrics["success_rate"], ".4f"),
            )

    logs = list(generation_logs)
    for entry in entries:
        logs.append(
            {
                "net_name": entry["net_name"],
                "status": "trained",
                "seed_samples": entry["seed_samples"],
                "aggregated_samples": len(entry["samples"]),
                "added_samples": entry["total_added_samples"],
                "query_states": entry["total_query_states"],
                "solved_queries": entry["total_solved_queries"],
                "invalid_labels": entry["invalid_labels"],
                "best_rollout": entry["best_rollout"] or {},
            }
        )

    return {
        "shared_checkpoint_path": shared_checkpoint_path,
        "nets_total": len(selected_net_files),
        "valid_nets_total": len(entries),
        "net_names": [entry["net_name"] for entry in entries],
        "scene_ref_expert_steps": int(median(successful_expert_steps)) if successful_expert_steps else 0,
        "logs": logs,
        "visit_summaries": visit_summaries,
        "round_metrics": round_metrics,
        "best_scene_metrics": best_scene_metrics,
        "best_round": best_round,
        "best_checkpoint_updates": best_checkpoint_updates,
        "best_round_order": best_round_order,
        "best_eval_summaries": best_eval_summaries,
        "il_method": "dagger_lite",
        "scene_rounds": scene_rounds,
    }
