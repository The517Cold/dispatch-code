import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from python_port.entrypoint_env import apply_inline_env_overrides
from python_port.entrypoints.imitation.run_bc_finetune import main as bc_finetune_main
from python_port.entrypoints.imitation.run_bc_pretrain import main as bc_pretrain_main
from python_port.entrypoints.imitation.run_bc_transfer_adapt import main as bc_transfer_adapt_main
from python_port.entrypoints.imitation.run_bc_transfer_eval import main as bc_transfer_eval_main
from python_port.entrypoints.imitation.run_dagger_pretrain import main as dagger_pretrain_main

# 脚本内环境变量覆盖：
# 1. 直接在这里填写键值对，就能在代码里改运行环境变量
# 2. priority="code" 表示代码优先；改成 "terminal" 则终端优先
INLINE_ENV_OVERRIDE_PRIORITY = "code"
INLINE_ENV_OVERRIDES = {
     "IL_MODE": "dagger_pretrain",
     "DAGGER_SCENE_ID": "1",
}


if __name__ == "__main__":
    apply_inline_env_overrides(INLINE_ENV_OVERRIDES, priority=INLINE_ENV_OVERRIDE_PRIORITY)
    mode = os.environ.get("IL_MODE", "bc_finetune").strip().lower()
    if mode == "bc_pretrain":
        bc_pretrain_main()
    elif mode == "dagger_pretrain":
        dagger_pretrain_main()
    elif mode == "transfer_eval":
        bc_transfer_eval_main()
    elif mode == "transfer_adapt":
        bc_transfer_adapt_main()
    else:
        bc_finetune_main()
