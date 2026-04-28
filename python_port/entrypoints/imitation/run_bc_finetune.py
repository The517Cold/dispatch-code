import os
import sys
import traceback

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from python_port.entrypoint_env import apply_inline_env_overrides, format_inline_env_overrides
from python_port.imitation.finetune import finetune_one_net


# ===== USER-TUNABLE DEFAULTS: EDIT HERE FIRST =====
DEFAULT_BC_NET_FILE = "1-1-4.txt"
# 留空则从共享 BC checkpoint 开始，也可以手工指定同场景或同网 checkpoint。
DEFAULT_BC_INIT_CKPT_PATH = ""
DEFAULT_BC_VAL_SAMPLES_PATH = ""
# ===== END USER-TUNABLE DEFAULTS =====

INLINE_ENV_OVERRIDE_PRIORITY = "code"
INLINE_ENV_OVERRIDES = {
    # "BC_NET_FILE": "1-1-4.txt",
    # "BC_FINETUNE_EPOCHS": "30",
}


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    try:
        applied_inline_env = apply_inline_env_overrides(INLINE_ENV_OVERRIDES, priority=INLINE_ENV_OVERRIDE_PRIORITY)
        net_file = os.environ.get("BC_NET_FILE", DEFAULT_BC_NET_FILE)
        net_path = os.path.join(base_dir, "resources", net_file)
        if not os.path.exists(net_path):
            raise FileNotFoundError("missing net file: " + net_path)
        net_stem = os.path.splitext(net_file)[0]
        shared_ckpt = os.path.join(base_dir, "checkpoints", "bc_pretrain_latest.pt")
        out_ckpt = os.path.join(base_dir, "checkpoints", "bc_" + net_stem + ".pt")
        result_path = os.path.join(base_dir, "results", "bc_" + net_stem + "_result.txt")
        progress_path = os.path.join(base_dir, "results", "bc_" + net_stem + "_progress.txt")

        init_ckpt = os.environ.get("BC_INIT_CKPT_PATH", DEFAULT_BC_INIT_CKPT_PATH)
        val_samples_path = os.environ.get("BC_VAL_SAMPLES_PATH", DEFAULT_BC_VAL_SAMPLES_PATH)
        epochs = int(os.environ.get("BC_FINETUNE_EPOCHS", "30"))
        rollout_every = int(os.environ.get("BC_FINETUNE_ROLLOUT_EVERY", "5"))
        max_expand_nodes = int(os.environ.get("BC_MAX_EXPAND_NODES", "360000"))
        max_search_seconds = float(os.environ.get("BC_MAX_SEARCH_SECONDS", "120"))
        max_data_gen_seconds = float(os.environ.get("BC_MAX_DATA_GEN_SECONDS", "300"))
        perturb_count = int(os.environ.get("BC_PERTURB_COUNT", "8"))
        perturb_steps = int(os.environ.get("BC_PERTURB_STEPS", "2"))
        clean_repeat = int(os.environ.get("BC_CLEAN_REPEAT", "3"))
        allow_generate_efline = os.environ.get("BC_ALLOW_GENERATE_EFLINE", "1") == "1"
        efline_expand_nodes = int(os.environ.get("BC_EFLINE_MAX_EXPAND_NODES", "80000"))
        efline_search_seconds = float(os.environ.get("BC_EFLINE_MAX_SEARCH_SECONDS", "10"))
        device_override = os.environ.get("BC_DEVICE", "")
        reuse_existing_when_no_samples = os.environ.get("BC_REUSE_EXISTING_WHEN_NO_SAMPLES", "1") == "1"
        seed = int(os.environ.get("BC_SEED", "42"))
        print(
            "BC finetune net_file="
            + net_file
            + " init_ckpt="
            + (init_ckpt if init_ckpt else "shared/default")
            + " val_samples="
            + (val_samples_path if val_samples_path else "auto_split"),
            flush=True,
        )
        print(format_inline_env_overrides(applied_inline_env), flush=True)

        finetune_one_net(
            net_path=net_path,
            base_dir=base_dir,
            shared_checkpoint_path=shared_ckpt,
            output_checkpoint_path=out_ckpt,
            output_result_path=result_path,
            output_progress_path=progress_path,
            init_checkpoint_path=init_ckpt,
            reuse_existing_when_no_samples=reuse_existing_when_no_samples,
            val_samples_path=val_samples_path,
            finetune_epochs=epochs,
            rollout_every_n_epochs=rollout_every,
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
        )
    except BaseException:
        err = "ERROR\n" + traceback.format_exc()
        print(err, flush=True)


if __name__ == "__main__":
    main()
