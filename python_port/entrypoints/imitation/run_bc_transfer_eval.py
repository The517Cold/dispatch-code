import os
import sys
import traceback

import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from python_port.imitation.rollout_eval import rollout_top1_greedy
from python_port.petri_net_io.utils.checkpoint_selector import load_compatible_state
from python_port.petri_net_io.utils.net_loader import build_ttpn_with_residence, load_petri_net_context
from python_port.petri_net_platform.search.petri_gcn_models import PetriNetGCNEnhanced, PetriStateEncoderEnhanced


def _format_sequence(trans):
    if not trans:
        return ""
    return "->".join(str(x) for x in trans)


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

        context = load_petri_net_context(target_net_path)
        pre = context["pre"]
        post = context["post"]
        end = context["end"]
        min_delay_p = context["min_delay_p"]
        petri_net = build_ttpn_with_residence(context)

        device = os.environ.get("BC_DEVICE", "")
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device_obj = torch.device(device)

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
        rollout_max_steps = int(os.environ.get("BC_TRANSFER_MAX_STEPS", "300"))

        rollout = rollout_top1_greedy(
            model=model,
            encoder=encoder,
            petri_net=petri_net,
            end=end,
            pre=pre,
            max_steps=rollout_max_steps,
            device=device_obj,
        )

        net_stem = os.path.splitext(target_net_file)[0]
        result_path = os.path.join(base_dir, "results", "bc_transfer_" + net_stem + "_result.txt")
        progress_path = os.path.join(base_dir, "results", "bc_transfer_" + net_stem + "_progress.txt")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)

        lines = [
            "status:ok",
            "target_net_file:" + target_net_file,
            "source_checkpoint:" + source_ckpt,
            "device:" + str(device_obj),
            "rollout_max_steps:" + str(rollout_max_steps),
            "reach_goal:" + str(bool(rollout.get("reach_goal", False))),
            "goal_distance:" + str(rollout.get("goal_distance", -1)),
            "policy_trans_count:" + str(rollout.get("policy_trans_count", 0)),
            "policy_trans_sequence:" + _format_sequence(rollout.get("policy_trans_sequence", [])),
            "policy_makespan:" + str(rollout.get("policy_makespan", -1)),
            "note:transfer eval does not run A* sample generation",
        ]
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
