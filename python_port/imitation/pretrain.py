import os
import random
from statistics import median
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, random_split

from python_port.imitation.data import BCDataset, bc_collate_fn
from python_port.imitation.expert_dataset import generate_augmented_bc_samples
from python_port.imitation.rollout_eval import rollout_top1_greedy
from python_port.imitation.scene_train_utils import (
    append_line,
    budget_for_round,
    compute_scene_metrics,
    is_better_scene_metrics,
    state_dict_cpu,
)
from python_port.imitation.trainer import BCTrainer, BCTrainerConfig
from python_port.petri_net_io.utils.checkpoint_selector import load_compatible_state
from python_port.petri_net_io.utils.net_loader import build_ttpn_with_residence, load_petri_net_context
from python_port.petri_net_platform.search.petri_gcn_models import PetriNetGCNEnhanced, PetriStateEncoderEnhanced
from python_port.scene_utils import list_dash_net_files


DEFAULT_PRETRAIN_EPOCHS = 80
DEFAULT_ROLLOUT_EVERY_PRETRAIN = 10
DEFAULT_SCENE_ROUNDS = 3


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
    epochs,
    rollout_every_n_epochs,
    seed,
    rollout_fn,
    progress_log_path,
    log_prefix,
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
        checkpoint_path="",
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


def pretrain_across_nets(
    base_dir: str,
    resources_dir: str,
    shared_checkpoint_path: str,
    net_files: Optional[List[str]] = None,
    pretrain_epochs: int = DEFAULT_PRETRAIN_EPOCHS,
    rollout_every_n_epochs: int = DEFAULT_ROLLOUT_EVERY_PRETRAIN,
    scene_rounds: int = DEFAULT_SCENE_ROUNDS,
    max_expand_nodes: int = 360000,
    max_search_seconds: float = 120.0,
    max_data_gen_seconds: float = 300.0,
    perturb_count: int = 8,
    perturb_steps: int = 2,
    clean_repeat: int = 3,
    allow_generate_efline: bool = True,
    efline_expand_nodes: int = 80000,
    efline_search_seconds: float = 10.0,
    device_override: str = "",
    seed: int = 42,
    net_limit: int = 0,
    output_progress_path: Optional[str] = None,
) -> Dict[str, object]:
    """按场景多轮循环训练一个共享 BC 模型。"""
    selected_net_files = list(net_files) if net_files is not None else list_dash_net_files(resources_dir)
    if net_limit > 0:
        selected_net_files = selected_net_files[:net_limit]
    if not selected_net_files:
        raise RuntimeError("no digit-dash .txt net files found in " + resources_dir)
    os.makedirs(os.path.dirname(shared_checkpoint_path), exist_ok=True)
    if output_progress_path:
        os.makedirs(os.path.dirname(output_progress_path), exist_ok=True)
        with open(output_progress_path, "w", encoding="utf-8") as f:
            f.write("# bc_pretrain: 场景内多轮循环训练\n")
            f.write("# nets_total:" + str(len(selected_net_files)) + "\n")
    device = device_override if device_override else ("cuda" if torch.cuda.is_available() else "cpu")

    dataset_entries = []
    generation_logs = []
    successful_expert_steps: List[int] = []
    for net_path in selected_net_files:
        context = load_petri_net_context(net_path)
        pre = context["pre"]
        end = context["end"]
        min_delay_p = context["min_delay_p"]
        petri_net = build_ttpn_with_residence(context)
        net_name = os.path.splitext(os.path.basename(net_path))[0]
        place_count = len(pre)
        trans_count = len(pre[0]) if pre else 0
        complexity = place_count * trans_count
        effective_perturb_count = int(perturb_count)
        if complexity >= 900 and effective_perturb_count > 2:
            effective_perturb_count = 2
        append_line(output_progress_path, "[BC] net=" + net_name + " stage=seed_data begin")

        budget_candidates = [max(1, int(max_expand_nodes)), max(1, int(max_expand_nodes) * 3)]
        generated = None
        used_budget = budget_candidates[0]
        for budget in budget_candidates:
            used_budget = budget
            generated = generate_augmented_bc_samples(
                petri_net=petri_net,
                end=end,
                pre=pre,
                min_delay_p=min_delay_p,
                source_net=net_name,
                context=context,
                max_expand_nodes=budget,
                max_search_seconds=max_search_seconds,
                max_data_gen_seconds=max_data_gen_seconds,
                perturb_count=effective_perturb_count,
                perturb_steps=perturb_steps,
                clean_repeat=clean_repeat,
                allow_generate_efline=allow_generate_efline,
                efline_expand_nodes=efline_expand_nodes,
                efline_search_seconds=efline_search_seconds,
                seed=seed,
            )
            if len(generated["samples"]) >= 2:
                break
        if generated is None:
            generated = {"samples": [], "stats": {}, "clean_result": None}
        samples = list(generated["samples"])
        if len(samples) < 2:
            generation_logs.append(
                {
                    "net_name": net_name,
                    "status": "skip_insufficient_samples",
                    "seed_samples": len(samples),
                    "used_max_expand_nodes": used_budget,
                }
            )
            append_line(output_progress_path, "[BC] net=" + net_name + " stage=skip reason=insufficient_samples")
            continue
        clean_result = generated["clean_result"]
        expert_steps = len(clean_result.get_trans()) if clean_result is not None else 64
        successful_expert_steps.append(expert_steps)
        rollout_max_steps = max(1, expert_steps * 2)
        dataset_entries.append(
            {
                "net_name": net_name,
                "context": context,
                "petri_net_template": petri_net,
                "samples": samples,
                "expert_steps": expert_steps,
                "rollout_max_steps": rollout_max_steps,
                "used_max_expand_nodes": used_budget,
                "effective_perturb_count": effective_perturb_count,
                "max_search_seconds": max_search_seconds,
                "max_data_gen_seconds": max_data_gen_seconds,
                "clean_extend_count": generated["stats"].get("clean_extend_count", 0),
                "clean_terminated_by_budget": generated["stats"].get("clean_terminated_by_budget", 0),
                "clean_terminated_by_time": generated["stats"].get("clean_terminated_by_time", 0),
                "seed_samples": len(samples),
            }
        )
        generation_logs.append(
            {
                "net_name": net_name,
                "status": "seed_ready",
                "seed_samples": len(samples),
                "expert_steps": expert_steps,
                "used_max_expand_nodes": used_budget,
                "effective_perturb_count": effective_perturb_count,
                "max_search_seconds": max_search_seconds,
                "max_data_gen_seconds": max_data_gen_seconds,
                "clean_extend_count": generated["stats"].get("clean_extend_count", 0),
                "clean_terminated_by_budget": generated["stats"].get("clean_terminated_by_budget", 0),
                "clean_terminated_by_time": generated["stats"].get("clean_terminated_by_time", 0),
            }
        )

    if not dataset_entries:
        raise RuntimeError("no nets with sufficient BC samples were generated")

    current_model_state = {}
    init_source = "scratch"
    if os.path.exists(shared_checkpoint_path):
        loaded = torch.load(shared_checkpoint_path, map_location="cpu")
        current_model_state = loaded.get("model_state", {})
        init_source = "existing_shared_checkpoint"

    scene_rounds = max(1, int(scene_rounds))
    append_line(
        output_progress_path,
        "bc_scene_schedule scene_rounds="
        + str(scene_rounds)
        + " total_epochs_per_net="
        + str(pretrain_epochs)
        + " per_round_epoch_plan="
        + str([budget_for_round(pretrain_epochs, scene_rounds, i) for i in range(scene_rounds)])
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

    for round_idx in range(scene_rounds):
        round_epochs = budget_for_round(pretrain_epochs, scene_rounds, round_idx)
        if round_epochs <= 0:
            continue
        round_rng = random.Random(seed + round_idx)
        round_entries = list(dataset_entries)
        round_rng.shuffle(round_entries)
        round_order = [entry["net_name"] for entry in round_entries]
        append_line(
            output_progress_path,
            "bc_scene_round_begin round="
            + str(round_idx + 1)
            + "/"
            + str(scene_rounds)
            + " round_epochs="
            + str(round_epochs)
            + " order="
            + str(round_order),
        )
        for visit_index, entry in enumerate(round_entries):
            context = entry["context"]
            model = _build_model(context)
            if current_model_state:
                load_compatible_state(model, current_model_state)
            encoder = _build_encoder(context, device)
            rollout_fn = _make_rollout_fn(
                model=model,
                encoder=encoder,
                petri_net=entry["petri_net_template"].clone(),
                end=context["end"],
                pre=context["pre"],
                max_steps=entry["rollout_max_steps"],
                device=device,
            )
            append_line(
                output_progress_path,
                "[BC] net="
                + entry["net_name"]
                + " round="
                + str(round_idx + 1)
                + " visit="
                + str(visit_index + 1)
                + "/"
                + str(len(round_entries))
                + " samples="
                + str(entry["seed_samples"])
                + " epochs="
                + str(round_epochs),
            )
            fit_out = _fit_with_samples(
                model=model,
                samples=entry["samples"],
                device=device,
                epochs=round_epochs,
                rollout_every_n_epochs=min(max(1, rollout_every_n_epochs), round_epochs),
                seed=seed + round_idx * 1000 + visit_index,
                rollout_fn=rollout_fn,
                progress_log_path=output_progress_path,
                log_prefix="[BC-SCENE]",
            )
            current_model_state = state_dict_cpu(model)
            best_rollout = fit_out.get("best_rollout") or {}
            visit_summaries.append(
                {
                    "scene_round": round_idx + 1,
                    "visit_index": visit_index + 1,
                    "net_name": entry["net_name"],
                    "samples": len(entry["samples"]),
                    "expert_steps": entry["expert_steps"],
                    "epochs": round_epochs,
                    "best_epoch": fit_out.get("best_epoch", 0),
                    "best_rollout": best_rollout,
                }
            )
            append_line(
                output_progress_path,
                "[BC] net="
                + entry["net_name"]
                + " round="
                + str(round_idx + 1)
                + " stage=done best_goal="
                + ("1" if bool(best_rollout.get("reach_goal", False)) else "0")
                + " best_makespan="
                + str(best_rollout.get("policy_makespan", -1)),
            )

        eval_summaries, scene_metrics = _evaluate_scene_policy(dataset_entries, current_model_state, device)
        scene_metrics["round"] = round_idx + 1
        scene_metrics["order"] = round_order
        round_metrics.append(scene_metrics)
        append_line(
            output_progress_path,
            "bc_scene_round_eval round="
            + str(round_idx + 1)
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
            best_round = round_idx + 1
            best_round_order = list(round_order)
            best_eval_summaries = eval_summaries
            best_checkpoint_updates += 1
            torch.save(
                {
                    "model_state": current_model_state,
                    "il_method": "bc",
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
                "bc_scene_checkpoint_update round="
                + str(best_round)
                + " success_rate="
                + format(best_scene_metrics["success_rate"], ".4f"),
            )

    return {
        "shared_checkpoint_path": shared_checkpoint_path,
        "nets_total": len(selected_net_files),
        "valid_nets_total": len(dataset_entries),
        "net_names": [entry["net_name"] for entry in dataset_entries],
        "scene_ref_expert_steps": int(median(successful_expert_steps)) if successful_expert_steps else 0,
        "logs": generation_logs,
        "visit_summaries": visit_summaries,
        "round_metrics": round_metrics,
        "best_scene_metrics": best_scene_metrics,
        "best_round": best_round,
        "best_checkpoint_updates": best_checkpoint_updates,
        "best_round_order": best_round_order,
        "best_eval_summaries": best_eval_summaries,
        "il_method": "bc",
        "scene_rounds": scene_rounds,
    }
