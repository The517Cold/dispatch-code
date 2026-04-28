import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, random_split

from python_port.imitation.data import BCDataset, bc_collate_fn, load_samples, save_samples
from python_port.imitation.expert_dataset import generate_augmented_bc_samples
from python_port.imitation.rollout_eval import rollout_top1_greedy
from python_port.imitation.trainer import BCTrainer, BCTrainerConfig
from python_port.petri_net_io.utils.checkpoint_selector import load_compatible_state
from python_port.petri_net_io.utils.net_loader import build_ttpn_with_residence, load_petri_net_context
from python_port.petri_net_platform.search.petri_gcn_models import PetriNetGCNEnhanced, PetriStateEncoderEnhanced


DEFAULT_MAX_EXPAND_NODES = 360000
DEFAULT_PERTURB_COUNT = 8
DEFAULT_PERTURB_STEPS = 2
DEFAULT_FINETUNE_EPOCHS = 30
DEFAULT_ROLLOUT_EVERY_FINETUNE = 5


def _split_samples(samples, train_ratio: float, split_seed: int):
    dataset = BCDataset(samples)
    if len(dataset) < 2:
        return dataset, dataset
    train_size = int(len(dataset) * train_ratio)
    train_size = max(1, min(len(dataset) - 1, train_size))
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(split_seed))


def _build_model(context, lambda_p, lambda_t, num_layers):
    return PetriNetGCNEnhanced(
        context["pre"],
        context["post"],
        lambda_p,
        lambda_t,
        num_layers,
        end=context.get("end"),
        min_delay_p=context.get("min_delay_p"),
        min_delay_t=context.get("min_delay_t"),
        capacity=context.get("capacity"),
        max_residence_time=context.get("max_residence_time"),
        place_from_places=context.get("place_from_places"),
    )


def _format_sequence(trans: List[int]) -> str:
    if not trans:
        return ""
    return "->".join(str(x) for x in trans)


def _result_from_loaded_checkpoint(ckpt_path: str, output_result_path: str, output_progress_path: str):
    """Reuse an existing checkpoint when fresh expert data cannot be generated."""
    loaded = torch.load(ckpt_path, map_location="cpu")
    rollout = loaded.get("rollout_metrics", {}) if isinstance(loaded, dict) else {}
    train_metrics = loaded.get("train_metrics", {}) if isinstance(loaded, dict) else {}
    val_metrics = loaded.get("val_metrics", {}) if isinstance(loaded, dict) else {}
    best_epoch = int(loaded.get("best_epoch", loaded.get("epoch", 0))) if isinstance(loaded, dict) else 0
    lines = [
        "status:loaded_existing_checkpoint",
        "samples_total:0",
        "clean_steps:0",
        "clean_extend_count:0",
        "clean_terminated_by_budget:0",
        "clean_terminated_by_time:0",
        "used_max_expand_nodes:0",
        "max_search_seconds:0",
        "max_data_gen_seconds:0",
        "perturb_success:0",
        "perturb_attempts:0",
        "effective_perturb_count:0",
        "invalid_labels_generation:0",
        "train_loss:" + str(train_metrics.get("loss", 0.0)),
        "val_loss:" + str(val_metrics.get("loss", 0.0)),
        "val_action_top1_acc:" + str(val_metrics.get("action_top1_acc", 0.0)),
        "val_topk_acc@3:" + str(val_metrics.get("topk_acc@3", 0.0)),
        "val_mask_valid_rate:" + str(val_metrics.get("mask_valid_rate", 0.0)),
        "val_random_baseline_acc:" + str(val_metrics.get("random_baseline_acc", 0.0)),
        "val_invalid_label_rate:" + str(val_metrics.get("invalid_label_rate", 0.0)),
        "best_epoch:" + str(best_epoch),
        "reach_goal:" + str(bool(rollout.get("reach_goal", False))),
        "goal_distance:" + str(rollout.get("goal_distance", -1)),
        "expert_trans_count:" + str(len(rollout.get("expert_trans_sequence", []))),
        "expert_trans_sequence:" + _format_sequence(rollout.get("expert_trans_sequence", [])),
        "expert_makespan:" + str(rollout.get("expert_makespan", -1)),
        "policy_trans_count:" + str(rollout.get("policy_trans_count", 0)),
        "policy_trans_sequence:" + _format_sequence(rollout.get("policy_trans_sequence", [])),
        "policy_makespan:" + str(rollout.get("policy_makespan", -1)),
        "rollout_max_steps:0",
        "init_checkpoint_path:" + ckpt_path,
        "checkpoint_path:" + ckpt_path,
        "label_note:expert_action uses global transition id; candidate-local index is state-dependent.",
    ]
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(output_result_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_progress_path), exist_ok=True)
    with open(output_result_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(output_progress_path, "w", encoding="utf-8") as f:
        f.write("# no training run; loaded existing checkpoint (see *_result.txt for metrics).\n")
    print(text, flush=True)
    return {
        "net_name": os.path.splitext(os.path.basename(output_result_path))[0],
        "result_path": output_result_path,
        "checkpoint_path": ckpt_path,
        "best_rollout": rollout,
        "loaded_existing": True,
    }


def finetune_one_net(
    net_path: str,
    base_dir: str,
    shared_checkpoint_path: str,
    output_checkpoint_path: str,
    output_result_path: str,
    output_progress_path: str,
    init_checkpoint_path: str = "",
    reuse_existing_when_no_samples: bool = True,
    val_samples_path: str = "",
    finetune_epochs: int = DEFAULT_FINETUNE_EPOCHS,
    rollout_every_n_epochs: int = DEFAULT_ROLLOUT_EVERY_FINETUNE,
    max_expand_nodes: int = DEFAULT_MAX_EXPAND_NODES,
    max_search_seconds: float = 120.0,
    max_data_gen_seconds: float = 300.0,
    perturb_count: int = DEFAULT_PERTURB_COUNT,
    perturb_steps: int = DEFAULT_PERTURB_STEPS,
    clean_repeat: int = 3,
    allow_generate_efline: bool = True,
    efline_expand_nodes: int = 80000,
    efline_search_seconds: float = 10.0,
    device_override: str = "",
    seed: int = 42,
) -> Dict[str, object]:
    """Finetune BC on one target net, optionally starting from a shared checkpoint."""
    context = load_petri_net_context(net_path)
    pre = context["pre"]
    post = context["post"]
    end = context["end"]
    min_delay_p = context["min_delay_p"]
    petri_net = build_ttpn_with_residence(context)
    net_name = os.path.splitext(os.path.basename(net_path))[0]
    place_count = len(pre)
    trans_count = len(pre[0]) if pre else 0
    complexity = place_count * trans_count
    effective_perturb_count = int(perturb_count)
    # For very large nets, perturb generation can make machine unresponsive.
    if complexity >= 900 and effective_perturb_count > 2:
        effective_perturb_count = 2

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
    train_samples = generated["samples"]
    if len(train_samples) < 2:
        if reuse_existing_when_no_samples and os.path.exists(output_checkpoint_path):
            return _result_from_loaded_checkpoint(output_checkpoint_path, output_result_path, output_progress_path)
        raise RuntimeError("insufficient samples for net " + net_name + ", got " + str(len(train_samples)))

    if val_samples_path and os.path.exists(val_samples_path):
        # Optional external validation set keeps evaluation stable across reruns.
        val_samples, _ = load_samples(val_samples_path)
        if len(val_samples) == 0:
            train_set, val_set = _split_samples(train_samples, train_ratio=0.8, split_seed=seed)
        else:
            train_set = BCDataset(train_samples)
            val_set = BCDataset(val_samples)
    else:
        train_set, val_set = _split_samples(train_samples, train_ratio=0.8, split_seed=seed)

    batch_size = 16
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=bc_collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=bc_collate_fn)

    if device_override:
        device = device_override
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model(context, lambda_p=128, lambda_t=32, num_layers=4)
    used_init_checkpoint = ""
    if init_checkpoint_path and os.path.exists(init_checkpoint_path):
        loaded = torch.load(init_checkpoint_path, map_location="cpu")
        load_compatible_state(model, loaded.get("model_state", {}))
        used_init_checkpoint = init_checkpoint_path
    elif shared_checkpoint_path and os.path.exists(shared_checkpoint_path):
        # Fallback to the shared pretrain checkpoint when no per-net init is provided.
        loaded = torch.load(shared_checkpoint_path, map_location="cpu")
        load_compatible_state(model, loaded.get("model_state", {}))
        used_init_checkpoint = shared_checkpoint_path

    encoder = PetriStateEncoderEnhanced(
        end=end,
        min_delay_p=min_delay_p,
        device=torch.device(device),
        pre=pre,
        post=post,
        min_delay_t=context.get("min_delay_t"),
        capacity=context.get("capacity"),
        max_residence_time=context.get("max_residence_time"),
        place_from_places=context.get("place_from_places"),
    )
    clean_result = generated["clean_result"]
    expert_trans = clean_result.get_trans() if clean_result is not None else []
    expert_markings = clean_result.get_markings() if clean_result is not None else []
    expert_makespan = float(expert_markings[-1].get_prefix()) if expert_markings else -1.0
    rollout_max_steps = max(1, len(expert_trans) * 2) if expert_trans else 128

    def rollout_fn():
        out = rollout_top1_greedy(
            model=model,
            encoder=encoder,
            petri_net=petri_net,
            end=end,
            pre=pre,
            max_steps=rollout_max_steps,
            device=torch.device(device),
        )
        out["expert_trans_sequence"] = expert_trans
        out["expert_trans_count"] = len(expert_trans)
        out["expert_makespan"] = expert_makespan
        return out

    os.makedirs(os.path.dirname(output_progress_path), exist_ok=True)
    with open(output_progress_path, "w", encoding="utf-8") as f:
        f.write("# bc_finetune: one [BC] line per epoch (rollout on scheduled epochs only)\n")
        f.write("# net_file:" + os.path.basename(net_path) + "\n")

    config = BCTrainerConfig(
        epochs=finetune_epochs,
        lr=3e-4,
        weight_decay=1e-5,
        label_smoothing=0.0,
        device=device,
        checkpoint_path=output_checkpoint_path,
        rollout_every_n_epochs=rollout_every_n_epochs,
        log_interval=max(1, finetune_epochs // 10),
        progress_log_path=output_progress_path,
    )
    trainer = BCTrainer(model, config)
    fit_out = trainer.fit(train_loader, val_loader, rollout_fn=rollout_fn)

    final_train = fit_out["history"]["train"][-1] if fit_out["history"]["train"] else {}
    final_val = fit_out["history"]["val"][-1] if fit_out["history"]["val"] else {}
    best_rollout = fit_out.get("best_rollout") or rollout_fn()

    os.makedirs(os.path.dirname(output_result_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_progress_path), exist_ok=True)

    save_samples(
        os.path.join(base_dir, "results", "bc_" + net_name + "_train_samples.pt"),
        train_samples,
        meta={"net_name": net_name},
    )

    result_lines = [
        "net_file:" + os.path.basename(net_path),
        "samples_total:" + str(len(train_samples)),
        "clean_steps:" + str(generated["stats"]["clean_steps"]),
        "clean_extend_count:" + str(generated["stats"].get("clean_extend_count", 0)),
        "clean_terminated_by_budget:" + str(generated["stats"].get("clean_terminated_by_budget", 0)),
        "clean_terminated_by_time:" + str(generated["stats"].get("clean_terminated_by_time", 0)),
        "used_max_expand_nodes:" + str(used_budget),
        "max_search_seconds:" + str(max_search_seconds),
        "max_data_gen_seconds:" + str(max_data_gen_seconds),
        "perturb_success:" + str(generated["stats"]["perturb_success"]),
        "perturb_attempts:" + str(generated["stats"]["perturb_attempts"]),
        "effective_perturb_count:" + str(effective_perturb_count),
        "invalid_labels_generation:" + str(generated["stats"]["invalid_labels"]),
        "train_loss:" + str(final_train.get("loss", 0.0)),
        "val_loss:" + str(final_val.get("loss", 0.0)),
        "val_action_top1_acc:" + str(final_val.get("action_top1_acc", 0.0)),
        "val_topk_acc@3:" + str(final_val.get("topk_acc@3", 0.0)),
        "val_mask_valid_rate:" + str(final_val.get("mask_valid_rate", 0.0)),
        "val_random_baseline_acc:" + str(final_val.get("random_baseline_acc", 0.0)),
        "val_invalid_label_rate:" + str(final_val.get("invalid_label_rate", 0.0)),
        "best_epoch:" + str(fit_out.get("best_epoch", 0)),
        "reach_goal:" + str(bool(best_rollout.get("reach_goal", False))),
        "goal_distance:" + str(best_rollout.get("goal_distance", -1)),
        "expert_trans_count:" + str(best_rollout.get("expert_trans_count", 0)),
        "expert_trans_sequence:" + _format_sequence(best_rollout.get("expert_trans_sequence", [])),
        "expert_makespan:" + str(best_rollout.get("expert_makespan", -1)),
        "policy_trans_count:" + str(best_rollout.get("policy_trans_count", 0)),
        "policy_trans_sequence:" + _format_sequence(best_rollout.get("policy_trans_sequence", [])),
        "policy_makespan:" + str(best_rollout.get("policy_makespan", -1)),
        "rollout_max_steps:" + str(rollout_max_steps),
        "init_checkpoint_path:" + used_init_checkpoint,
        "checkpoint_path:" + output_checkpoint_path,
        "label_note:expert_action uses global transition id; candidate-local index is state-dependent.",
    ]
    text = "\n".join(result_lines) + "\n"
    with open(output_result_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(output_progress_path, "a", encoding="utf-8") as f:
        f.write("\n# --- end of run; structured summary in " + os.path.basename(output_result_path) + " ---\n")
    print(text, flush=True)
    return {
        "net_name": net_name,
        "result_path": output_result_path,
        "checkpoint_path": output_checkpoint_path,
        "best_rollout": best_rollout,
    }
