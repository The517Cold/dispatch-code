import os
import sys
import traceback

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from python_port.entrypoint_env import apply_inline_env_overrides, format_inline_env_overrides
from python_port.imitation.pretrain import pretrain_across_nets
from python_port.scene_utils import list_scene_net_files


# ===== USER-TUNABLE DEFAULTS: EDIT HERE FIRST =====
# 留空表示对 resources 下全部场景做 BC 预训练；填 "1"/"2"/"3" 则只训练单个场景。
DEFAULT_BC_SCENE_ID = "1"
# ===== END USER-TUNABLE DEFAULTS =====

# 脚本内环境变量覆盖：
# 可直接在代码里调整 BC 预训练参数，无需每次在终端设置。
INLINE_ENV_OVERRIDE_PRIORITY = "code"
INLINE_ENV_OVERRIDES = {
    # "BC_SCENE_ID": "1",
    # "BC_SCENE_ROUNDS": "3",
    # "BC_PRETRAIN_EPOCHS": "80",
}


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    result_path = ""
    try:
        applied_inline_env = apply_inline_env_overrides(INLINE_ENV_OVERRIDES, priority=INLINE_ENV_OVERRIDE_PRIORITY)
        resources_dir = os.path.join(base_dir, "resources")
        scene_id = os.environ.get("BC_SCENE_ID", DEFAULT_BC_SCENE_ID).strip()
        scene_tag = ("scene_" + scene_id) if scene_id else "pretrain"
        result_path = os.path.join(base_dir, "results", "bc_" + scene_tag + "_result.txt")
        progress_path = os.path.join(base_dir, "results", "bc_" + scene_tag + "_progress.txt")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        shared_ckpt = os.path.join(base_dir, "checkpoints", "bc_" + scene_tag + (".pt" if scene_id else "_latest.pt"))
        epochs = int(os.environ.get("BC_PRETRAIN_EPOCHS", "80"))
        scene_rounds = int(os.environ.get("BC_SCENE_ROUNDS", "3"))
        rollout_every = int(os.environ.get("BC_PRETRAIN_ROLLOUT_EVERY", "10"))
        max_expand_nodes = int(os.environ.get("BC_MAX_EXPAND_NODES", "360000"))
        max_search_seconds = float(os.environ.get("BC_MAX_SEARCH_SECONDS", "120"))
        max_data_gen_seconds = float(os.environ.get("BC_MAX_DATA_GEN_SECONDS", "300"))
        perturb_count = int(os.environ.get("BC_PERTURB_COUNT", "0"))
        perturb_steps = int(os.environ.get("BC_PERTURB_STEPS", "0"))
        clean_repeat = int(os.environ.get("BC_CLEAN_REPEAT", "3"))
        allow_generate_efline = os.environ.get("BC_ALLOW_GENERATE_EFLINE", "1") == "1"
        efline_expand_nodes = int(os.environ.get("BC_EFLINE_MAX_EXPAND_NODES", "80000"))
        efline_search_seconds = float(os.environ.get("BC_EFLINE_MAX_SEARCH_SECONDS", "10"))
        device_override = os.environ.get("BC_DEVICE", "cuda")
        seed = int(os.environ.get("BC_SEED", "42"))
        net_limit = int(os.environ.get("BC_PRETRAIN_NET_LIMIT", "0"))
        net_files = list_scene_net_files(resources_dir, scene_id)
        print(
            "BC pretrain scene_id="
            + (scene_id if scene_id else "all")
            + " nets="
            + str(
                [os.path.splitext(os.path.basename(p))[0] for p in net_files[:net_limit]]
                if net_limit > 0
                else [os.path.splitext(os.path.basename(p))[0] for p in net_files]
            ),
            flush=True,
        )
        print(format_inline_env_overrides(applied_inline_env), flush=True)
        out = pretrain_across_nets(
            base_dir=base_dir,
            resources_dir=resources_dir,
            shared_checkpoint_path=shared_ckpt,
            net_files=net_files,
            pretrain_epochs=epochs,
            rollout_every_n_epochs=rollout_every,
            scene_rounds=scene_rounds,
            max_expand_nodes=max_expand_nodes,
            max_search_seconds=max_search_seconds,
            max_data_gen_seconds=max_data_gen_seconds,
            perturb_count=perturb_count,
            perturb_steps=perturb_steps,
            clean_repeat=clean_repeat,
            allow_generate_efline=allow_generate_efline,
            efline_expand_nodes=efline_expand_nodes,
            efline_search_seconds=efline_search_seconds,
            device_override=device_override,
            seed=seed,
            net_limit=net_limit,
            output_progress_path=progress_path,
        )
        lines = [
            "status:ok",
            "il_method:bc",
            format_inline_env_overrides(applied_inline_env),
            "scene_id:" + (scene_id if scene_id else "all"),
            "shared_checkpoint_path:" + out["shared_checkpoint_path"],
            "nets_total:" + str(out["nets_total"]),
            "valid_nets_total:" + str(out.get("valid_nets_total", 0)),
            "scene_ref_expert_steps:" + str(out.get("scene_ref_expert_steps", 0)),
            "epochs:" + str(epochs),
            "scene_rounds:" + str(scene_rounds),
            "rollout_every_n_epochs:" + str(rollout_every),
            "max_expand_nodes:" + str(max_expand_nodes),
            "max_search_seconds:" + str(max_search_seconds),
            "max_data_gen_seconds:" + str(max_data_gen_seconds),
            "perturb_count:" + str(perturb_count),
            "perturb_steps:" + str(perturb_steps),
            "clean_repeat:" + str(clean_repeat),
            "allow_generate_efline:" + ("1" if allow_generate_efline else "0"),
            "efline_max_expand_nodes:" + str(efline_expand_nodes),
            "efline_max_search_seconds:" + str(efline_search_seconds),
            "device:" + (device_override if device_override else "auto"),
            "net_limit:" + str(net_limit),
        ]
        for net_name in out.get("net_names", []):
            lines.append("net_name:" + str(net_name))
        if out.get("best_scene_metrics") is not None:
            lines.append("best_round:" + str(out.get("best_round", 0)))
            lines.append("best_checkpoint_updates:" + str(out.get("best_checkpoint_updates", 0)))
            lines.append("best_scene_metrics:" + str(out.get("best_scene_metrics")))
            lines.append("best_round_order:" + str(out.get("best_round_order", [])))
        for item in out["logs"]:
            lines.append("net_log:" + str(item))
        for item in out.get("round_metrics", []):
            lines.append("round_log:" + str(item))
        txt = "\n".join(lines) + "\n"
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(txt)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write("\n# --- end of bc pretrain run; structured summary in " + os.path.basename(result_path) + " ---\n")
        print(txt, flush=True)
    except BaseException:
        err = "ERROR\n" + traceback.format_exc()
        if result_path:
            with open(result_path, "w", encoding="utf-8") as f:
                f.write(err)
        print(err, flush=True)


if __name__ == "__main__":
    main()
