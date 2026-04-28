import os
import sys
import traceback

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from python_port.entrypoint_env import apply_inline_env_overrides, format_inline_env_overrides
from python_port.imitation.dagger import pretrain_across_nets_dagger_lite
from python_port.scene_utils import list_scene_net_files


# ===== USER-TUNABLE DEFAULTS: EDIT HERE FIRST =====
# DAgger-lite 场景预训练：默认只做 scene 1，可改为 2/3 或留空遍历全部场景。
DEFAULT_DAGGER_SCENE_ID = "1"
# 是否允许用已有 BC scene/shared checkpoint 作为 DAgger 初始化。
DEFAULT_DAGGER_USE_BC_INIT = "1"
# ===== END USER-TUNABLE DEFAULTS =====

INLINE_ENV_OVERRIDE_PRIORITY = "code"
INLINE_ENV_OVERRIDES = {
    # "DAGGER_SCENE_ID": "1",
    # "DAGGER_SCENE_ROUNDS": "3",
    # "DAGGER_ROUNDS": "4",
    # "DAGGER_QUERY_STATES_PER_ROUND": "8",
}


def _resolve_init_checkpoint(base_dir, scene_id, allow_bc_init: bool):
    explicit = os.environ.get("DAGGER_INIT_CKPT_PATH", "").strip()
    if explicit:
        return explicit if os.path.isabs(explicit) else os.path.join(base_dir, explicit)
    if not allow_bc_init:
        return ""
    if scene_id:
        scene_ckpt = os.path.join(base_dir, "checkpoints", "bc_scene_" + str(scene_id) + ".pt")
        if os.path.exists(scene_ckpt):
            return scene_ckpt
    shared = os.path.join(base_dir, "checkpoints", "bc_pretrain_latest.pt")
    if os.path.exists(shared):
        return shared
    return ""


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    result_path = ""
    try:
        applied_inline_env = apply_inline_env_overrides(INLINE_ENV_OVERRIDES, priority=INLINE_ENV_OVERRIDE_PRIORITY)
        resources_dir = os.path.join(base_dir, "resources")
        scene_id = os.environ.get("DAGGER_SCENE_ID", DEFAULT_DAGGER_SCENE_ID).strip()
        allow_bc_init = os.environ.get("DAGGER_USE_BC_INIT", DEFAULT_DAGGER_USE_BC_INIT).strip() == "1"
        init_ckpt = _resolve_init_checkpoint(base_dir, scene_id, allow_bc_init=allow_bc_init)
        scene_tag = ("scene_" + scene_id) if scene_id else "pretrain"
        result_path = os.path.join(base_dir, "results", "dagger_" + scene_tag + "_result.txt")
        progress_path = os.path.join(base_dir, "results", "dagger_" + scene_tag + "_progress.txt")
        shared_ckpt = os.path.join(base_dir, "checkpoints", "dagger_" + scene_tag + (".pt" if scene_id else "_latest.pt"))

        init_epochs = int(os.environ.get("DAGGER_INIT_EPOCHS", "20"))
        scene_rounds = int(os.environ.get("DAGGER_SCENE_ROUNDS", "3"))
        dagger_rounds = int(os.environ.get("DAGGER_ROUNDS", "4"))
        round_epochs = int(os.environ.get("DAGGER_ROUND_EPOCHS", "6"))
        rollout_episodes_per_round = int(os.environ.get("DAGGER_ROLLOUTS_PER_ROUND", "4"))
        rollout_epsilon = float(os.environ.get("DAGGER_ROLLOUT_EPSILON", "0.10"))
        query_states_per_round = int(os.environ.get("DAGGER_QUERY_STATES_PER_ROUND", "8"))
        query_tail_steps = int(os.environ.get("DAGGER_QUERY_TAIL_STEPS", "3"))
        query_label_horizon = int(os.environ.get("DAGGER_QUERY_LABEL_HORIZON", "6"))
        max_expand_nodes = int(os.environ.get("DAGGER_MAX_EXPAND_NODES", "360000"))
        max_search_seconds = float(os.environ.get("DAGGER_MAX_SEARCH_SECONDS", "120"))
        max_data_gen_seconds = float(os.environ.get("DAGGER_MAX_DATA_GEN_SECONDS", "300"))
        initial_perturb_count = int(os.environ.get("DAGGER_INITIAL_PERTURB_COUNT", "2"))
        initial_perturb_steps = int(os.environ.get("DAGGER_INITIAL_PERTURB_STEPS", "1"))
        clean_repeat = int(os.environ.get("DAGGER_CLEAN_REPEAT", "3"))
        allow_generate_efline = os.environ.get("DAGGER_ALLOW_GENERATE_EFLINE", "1") == "1"
        efline_expand_nodes = int(os.environ.get("DAGGER_EFLINE_MAX_EXPAND_NODES", "80000"))
        efline_search_seconds = float(os.environ.get("DAGGER_EFLINE_MAX_SEARCH_SECONDS", "10"))
        query_expand_nodes = int(os.environ.get("DAGGER_QUERY_MAX_EXPAND_NODES", "80000"))
        query_search_seconds = float(os.environ.get("DAGGER_QUERY_MAX_SEARCH_SECONDS", "20"))
        device_override = os.environ.get("DAGGER_DEVICE", "")
        seed = int(os.environ.get("DAGGER_SEED", "42"))
        net_limit = int(os.environ.get("DAGGER_NET_LIMIT", "0"))
        net_files = list_scene_net_files(resources_dir, scene_id)

        print(
            "DAgger-lite pretrain scene_id="
            + (scene_id if scene_id else "all")
            + " init_ckpt="
            + (init_ckpt if init_ckpt else "scratch")
            + " nets="
            + str(
                [os.path.splitext(os.path.basename(p))[0] for p in net_files[:net_limit]]
                if net_limit > 0
                else [os.path.splitext(os.path.basename(p))[0] for p in net_files]
            ),
            flush=True,
        )
        print(format_inline_env_overrides(applied_inline_env), flush=True)

        out = pretrain_across_nets_dagger_lite(
            base_dir=base_dir,
            resources_dir=resources_dir,
            shared_checkpoint_path=shared_ckpt,
            net_files=net_files,
            init_checkpoint_path=init_ckpt,
            init_epochs=init_epochs,
            scene_rounds=scene_rounds,
            dagger_rounds=dagger_rounds,
            round_epochs=round_epochs,
            rollout_episodes_per_round=rollout_episodes_per_round,
            rollout_epsilon=rollout_epsilon,
            query_states_per_round=query_states_per_round,
            query_tail_steps=query_tail_steps,
            query_label_horizon=query_label_horizon,
            max_expand_nodes=max_expand_nodes,
            max_search_seconds=max_search_seconds,
            max_data_gen_seconds=max_data_gen_seconds,
            initial_perturb_count=initial_perturb_count,
            initial_perturb_steps=initial_perturb_steps,
            clean_repeat=clean_repeat,
            allow_generate_efline=allow_generate_efline,
            efline_expand_nodes=efline_expand_nodes,
            efline_search_seconds=efline_search_seconds,
            query_expand_nodes=query_expand_nodes,
            query_search_seconds=query_search_seconds,
            device_override=device_override,
            seed=seed,
            net_limit=net_limit,
            output_progress_path=progress_path,
        )

        lines = [
            "status:ok",
            "il_method:dagger_lite",
            format_inline_env_overrides(applied_inline_env),
            "scene_id:" + (scene_id if scene_id else "all"),
            "shared_checkpoint_path:" + out["shared_checkpoint_path"],
            "nets_total:" + str(out["nets_total"]),
            "valid_nets_total:" + str(out.get("valid_nets_total", 0)),
            "scene_ref_expert_steps:" + str(out.get("scene_ref_expert_steps", 0)),
            "init_checkpoint_path:" + (init_ckpt if init_ckpt else ""),
            "init_epochs:" + str(init_epochs),
            "scene_rounds:" + str(scene_rounds),
            "dagger_rounds:" + str(dagger_rounds),
            "round_epochs:" + str(round_epochs),
            "rollout_episodes_per_round:" + str(rollout_episodes_per_round),
            "rollout_epsilon:" + str(rollout_epsilon),
            "query_states_per_round:" + str(query_states_per_round),
            "query_tail_steps:" + str(query_tail_steps),
            "query_label_horizon:" + str(query_label_horizon),
            "max_expand_nodes:" + str(max_expand_nodes),
            "max_search_seconds:" + str(max_search_seconds),
            "max_data_gen_seconds:" + str(max_data_gen_seconds),
            "initial_perturb_count:" + str(initial_perturb_count),
            "initial_perturb_steps:" + str(initial_perturb_steps),
            "clean_repeat:" + str(clean_repeat),
            "query_max_expand_nodes:" + str(query_expand_nodes),
            "query_max_search_seconds:" + str(query_search_seconds),
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
        text = "\n".join(lines) + "\n"
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(text)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write("\n# --- end of dagger pretrain run; structured summary in " + os.path.basename(result_path) + " ---\n")
        print(text, flush=True)
    except BaseException:
        err = "ERROR\n" + traceback.format_exc()
        if result_path:
            with open(result_path, "w", encoding="utf-8") as f:
                f.write(err)
        print(err, flush=True)


if __name__ == "__main__":
    main()
