import os
import random
import sys
import traceback

import torch
from torch.utils.data import DataLoader, random_split

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from python_port.imitation.data import BCDataset, BCSample, bc_collate_fn
from python_port.imitation.expert_dataset import action_mask_from_marking, goal_distance
from python_port.imitation.trainer import BCTrainer, BCTrainerConfig
from python_port.petri_net_io.utils.checkpoint_selector import load_compatible_state
from python_port.petri_net_io.utils.net_loader import build_ttpn_with_residence, load_petri_net_context
from python_port.petri_net_platform.search.petri_gcn_models import PetriNetGCNEnhanced, PetriStateEncoderEnhanced


def _format_sequence(trans):
    if not trans:
        return ""
    return "->".join(str(x) for x in trans)


def _rollout_episode(model, encoder, petri_net, end, pre, max_steps, epsilon, rng, device):
    net = petri_net.clone()
    curr = net.get_marking()
    capacity = getattr(net, "capacity", None)
    transition_flow_allowed = getattr(net, "transition_flow_allowed", None)
    traj = []
    with torch.no_grad():
        for step_idx in range(max_steps):
            if goal_distance(curr, end) == 0:
                break
            mask = action_mask_from_marking(curr, pre, capacity, transition_flow_allowed)
            enabled = torch.nonzero(mask, as_tuple=False).flatten()
            if enabled.numel() == 0:
                break
            logits = model(encoder.encode(curr).to(device))
            masked_logits = logits.masked_fill(~mask.to(device), -1e9)
            if rng.random() < epsilon:
                action = int(enabled[rng.randint(0, enabled.numel() - 1)].item())
            else:
                action = int(torch.argmax(masked_logits).item())
            if not net.enable(action):
                break
            state_for_sample = curr.clone()
            next_marking = net.launch(action)
            net.set_marking(next_marking)
            traj.append((state_for_sample, action, step_idx))
            curr = next_marking
            if goal_distance(curr, end) == 0:
                break
    return {
        "trajectory": traj,
        "reach_goal": goal_distance(curr, end) == 0,
        "goal_distance": int(goal_distance(curr, end)),
        "makespan": float(curr.get_prefix()),
        "trans_count": len(traj),
        "trans_sequence": [x[1] for x in traj],
    }


def _better_episode(curr, best):
    if best is None:
        return True
    if curr["reach_goal"] != best["reach_goal"]:
        return curr["reach_goal"] and (not best["reach_goal"])
    if curr["reach_goal"]:
        if curr["makespan"] != best["makespan"]:
            return curr["makespan"] < best["makespan"]
        return curr["trans_count"] < best["trans_count"]
    if curr["goal_distance"] != best["goal_distance"]:
        return curr["goal_distance"] < best["goal_distance"]
    return curr["makespan"] < best["makespan"]


def _build_samples_from_episode(episode, encoder, end, pre, capacity, transition_flow_allowed):
    samples = []
    for state, action, step_idx in episode["trajectory"]:
        mask = action_mask_from_marking(state, pre, capacity, transition_flow_allowed)
        if action < 0 or action >= mask.numel() or (not bool(mask[action].item())):
            continue
        samples.append(
            BCSample(
                state_features=encoder.encode(state).cpu(),
                action_mask=mask.cpu(),
                expert_action=int(action),
                meta={
                    "step_idx": int(step_idx),
                    "prefix_time": float(state.get_prefix()),
                    "goal_distance": int(goal_distance(state, end)),
                    "source_net": "transfer_adapt",
                    "source_type": "pseudo_rollout",
                },
            )
        )
    return samples


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    try:
        target_net_file = os.environ.get("BC_TRANSFER_TARGET_NET", "3-1.txt")
        source_ckpt = os.environ.get(
            "BC_TRANSFER_SOURCE_CKPT",
            os.path.join(base_dir, "checkpoints", "bc_1-2.pt"),
        )
        target_net_path = os.path.join(base_dir, "resources", target_net_file)
        if not os.path.exists(target_net_path):
            raise FileNotFoundError("missing target net file: " + target_net_path)
        if not os.path.exists(source_ckpt):
            raise FileNotFoundError("missing source checkpoint: " + source_ckpt)

        rounds = int(os.environ.get("BC_TRANSFER_ADAPT_ROUNDS", "6"))
        episodes_per_round = int(os.environ.get("BC_TRANSFER_ADAPT_EPISODES", "32"))
        max_steps = int(os.environ.get("BC_TRANSFER_MAX_STEPS", "300"))
        adapt_epochs = int(os.environ.get("BC_TRANSFER_ADAPT_EPOCHS", "5"))
        epsilon = float(os.environ.get("BC_TRANSFER_ADAPT_EPSILON", "0.20"))
        seed = int(os.environ.get("BC_SEED", "42"))
        rng = random.Random(seed)

        device = os.environ.get("BC_DEVICE", "")
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device_obj = torch.device(device)

        context = load_petri_net_context(target_net_path)
        pre = context["pre"]
        post = context["post"]
        end = context["end"]
        min_delay_p = context["min_delay_p"]
        petri_net = build_ttpn_with_residence(context)
        capacity = getattr(petri_net, "capacity", None)
        transition_flow_allowed = getattr(petri_net, "transition_flow_allowed", None)

        model = PetriNetGCNEnhanced(pre, post, 128, 32, 4)
        loaded = torch.load(source_ckpt, map_location="cpu")
        load_compatible_state(model, loaded.get("model_state", {}))
        model.to(device_obj)
        encoder = PetriStateEncoderEnhanced(
            end=end,
            min_delay_p=min_delay_p,
            device=device_obj,
            pre=pre,
            post=post,
            min_delay_t=context.get("min_delay_t"),
            capacity=context.get("capacity"),
            max_residence_time=context.get("max_residence_time"),
            place_from_places=context.get("place_from_places"),
        )

        global_best = None
        round_logs = []
        for round_idx in range(rounds):
            best_episode = None
            for _ in range(episodes_per_round):
                ep = _rollout_episode(model, encoder, petri_net, end, pre, max_steps, epsilon, rng, device_obj)
                if _better_episode(ep, best_episode):
                    best_episode = ep
            if best_episode is None:
                break
            if _better_episode(best_episode, global_best):
                global_best = best_episode
            pseudo_samples = _build_samples_from_episode(
                best_episode, encoder, end, pre, capacity, transition_flow_allowed
            )
            if len(pseudo_samples) >= 2:
                ds = BCDataset(pseudo_samples)
                train_size = max(1, min(len(ds) - 1, int(len(ds) * 0.8)))
                val_size = len(ds) - train_size
                if val_size <= 0:
                    train_set, val_set = ds, ds
                else:
                    train_set, val_set = random_split(
                        ds, [train_size, val_size], generator=torch.Generator().manual_seed(seed + round_idx)
                    )
                train_loader = DataLoader(train_set, batch_size=16, shuffle=True, collate_fn=bc_collate_fn)
                val_loader = DataLoader(val_set, batch_size=16, shuffle=False, collate_fn=bc_collate_fn)
                target_stem = os.path.splitext(target_net_file)[0]
                ckpt_path = os.path.join(base_dir, "checkpoints", "bc_transfer_adapt_" + target_stem + ".pt")
                cfg = BCTrainerConfig(
                    epochs=adapt_epochs,
                    lr=3e-4,
                    weight_decay=1e-5,
                    label_smoothing=0.0,
                    device=str(device_obj),
                    checkpoint_path=ckpt_path,
                    rollout_every_n_epochs=max(1, adapt_epochs),
                    log_interval=max(1, adapt_epochs),
                )
                trainer = BCTrainer(model, cfg)
                trainer.fit(train_loader, val_loader, rollout_fn=None)
            round_info = {
                "round": round_idx + 1,
                "reach_goal": bool(best_episode["reach_goal"]),
                "goal_distance": int(best_episode["goal_distance"]),
                "makespan": float(best_episode["makespan"]),
                "trans_count": int(best_episode["trans_count"]),
                "pseudo_samples": len(pseudo_samples),
            }
            round_logs.append(round_info)
            print(
                "[BC-TRANSFER] round="
                + str(round_info["round"])
                + "/"
                + str(rounds)
                + " reach_goal="
                + ("1" if round_info["reach_goal"] else "0")
                + " goal_distance="
                + str(round_info["goal_distance"])
                + " makespan="
                + format(round_info["makespan"], ".3f")
                + " trans_count="
                + str(round_info["trans_count"])
                + " pseudo_samples="
                + str(round_info["pseudo_samples"]),
                flush=True,
            )
            if global_best is not None and global_best["reach_goal"]:
                break

        target_stem = os.path.splitext(target_net_file)[0]
        result_path = os.path.join(base_dir, "results", "bc_transfer_adapt_" + target_stem + "_result.txt")
        progress_path = os.path.join(base_dir, "results", "bc_transfer_adapt_" + target_stem + "_progress.txt")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)

        lines = [
            "status:ok",
            "target_net_file:" + target_net_file,
            "source_checkpoint:" + source_ckpt,
            "device:" + str(device_obj),
            "rounds:" + str(rounds),
            "episodes_per_round:" + str(episodes_per_round),
            "max_steps:" + str(max_steps),
            "adapt_epochs:" + str(adapt_epochs),
            "epsilon:" + str(epsilon),
        ]
        for item in round_logs:
            lines.append("round_log:" + str(item))
        if global_best is None:
            lines.extend(
                [
                    "reach_goal:False",
                    "goal_distance:-1",
                    "policy_trans_count:0",
                    "policy_trans_sequence:",
                    "policy_makespan:-1",
                ]
            )
        else:
            lines.extend(
                [
                    "reach_goal:" + str(bool(global_best["reach_goal"])),
                    "goal_distance:" + str(global_best["goal_distance"]),
                    "policy_trans_count:" + str(global_best["trans_count"]),
                    "policy_trans_sequence:" + _format_sequence(global_best["trans_sequence"]),
                    "policy_makespan:" + str(global_best["makespan"]),
                ]
            )
        lines.append("note:transfer adapt uses pseudo-labels from model rollout, no A* generation")
        out = "\n".join(lines) + "\n"
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(out)
        with open(progress_path, "w", encoding="utf-8") as f:
            f.write(out)
        print(out, flush=True)
    except BaseException:
        err = "ERROR\n" + traceback.format_exc()
        print(err, flush=True)


if __name__ == "__main__":
    main()
