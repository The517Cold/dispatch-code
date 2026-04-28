import os
import re
import sys
import time
import traceback
import torch  # pyright: ignore[reportMissingImports]

# 将仓库根目录加入模块搜索路径，确保可从脚本直接运行。
#将某网文件作为训练对象时，先判断是否有已有模型（exact），如果有则直接加载，否则判断是否有相似模型（similar），如果有则加载，否则训练新模型。
#可选项1：如果命中similar模型，是否继续微调训练，可通过环境变量GCN_ENH_HQ_FINETUNE_ON_SIMILAR（line：206）控制，默认不微调。
#可选项2：无论是否命中模型，强制重新训练，可通过环境变量GCN_ENH_HQ_REUSE（line：204）控制，默认不强制。

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from python_port.entrypoint_env import apply_inline_env_overrides, format_inline_env_overrides
from python_port.petri_net_io.utils.net_loader import load_petri_net_context, build_ttpn_with_residence
from python_port.petri_net_io.utils.checkpoint_selector import (
    build_signature,
    build_profile,
    checkpoint_path,
    find_checkpoint,
    load_compatible_state,
)
from python_port.petri_net_platform.search.petri_net_gcn_dqn_enhanced_hq import PetriNetGCNDQNEnhancedHQ
from python_port.scene_utils import infer_scene_id


# ===== USER-TUNABLE DEFAULTS: EDIT HERE FIRST =====
DEFAULT_GCN_ENH_HQ_INPUT_SUBDIR = "test"
DEFAULT_GCN_ENH_HQ_NET_FILE = "1-2-13.txt"
# Leave empty to auto-infer from net file name like 1-6.txt -> scene 1.
DEFAULT_GCN_ENH_HQ_SCENE_ID = ""#留空则根据文件名自动推断，输入数字指定场景
DEFAULT_GCN_ENH_HQ_VERBOSE = "1"#1：输出推理详情；0：不输出推理详情
DEFAULT_GCN_ENH_HQ_LOG_INTERVAL = "5"#打印日志间隔，单位episode
# RL finetune budget scale after BC warm start: exact-net / same-scene / global shared.
DEFAULT_GCN_ENH_HQ_BC_WARM_NET_SCALE = "0.40"#exact-net的微调预算比例
DEFAULT_GCN_ENH_HQ_BC_WARM_SCENE_SCALE = "0.65"#same-scene的微调预算比例
DEFAULT_GCN_ENH_HQ_BC_WARM_GLOBAL_SCALE = "0.85"#global shared的微调预算比例
# ===== END USER-TUNABLE DEFAULTS =====


# Re-declare readable defaults below because the original inline comments were merged on one line.
DEFAULT_GCN_ENH_HQ_SCENE_ID = ""
DEFAULT_GCN_ENH_HQ_VERBOSE = "1"
DEFAULT_GCN_ENH_HQ_LOG_INTERVAL = "5"
DEFAULT_GCN_ENH_HQ_BC_WARM_NET_SCALE = "0.40"
DEFAULT_GCN_ENH_HQ_BC_WARM_SCENE_SCALE = "0.65"
DEFAULT_GCN_ENH_HQ_BC_WARM_GLOBAL_SCALE = "0.85"
DEFAULT_GCN_ENH_HQ_BC_WARM_NET_EPSILON_INIT = "0.20"
DEFAULT_GCN_ENH_HQ_BC_WARM_SCENE_EPSILON_INIT = "0.35"
DEFAULT_GCN_ENH_HQ_BC_WARM_GLOBAL_EPSILON_INIT = "0.60"
DEFAULT_GCN_ENH_HQ_CONTROLLER_REPRESENTATION = "1"

# 脚本内环境变量覆盖：
# 直接在这里填写键值对，可在代码里调整 DQN HQ 的运行参数。
INLINE_ENV_OVERRIDE_PRIORITY = "code"
INLINE_ENV_OVERRIDES = {
    # "GCN_ENH_HQ_INPUT_SUBDIR": "test",
    # "GCN_ENH_HQ_NET_FILE": "1-2-13.txt",
}


def _read_result_kv(path):
    out = {}
    if not path or (not os.path.exists(path)):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if (not line) or (":" not in line):
                continue
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def _safe_int(value, default=0):
    try:
        return int(str(value).strip())
    except BaseException:
        return default


def _resolve_scene_id(net_stem):
    explicit = os.environ.get("GCN_ENH_HQ_SCENE_ID", DEFAULT_GCN_ENH_HQ_SCENE_ID).strip()
    if explicit:
        return explicit
    return infer_scene_id(net_stem)


def _resolve_bc_checkpoint(base_dir, net_stem, scene_id=""):
    explicit = os.environ.get("GCN_ENH_HQ_BC_CKPT_PATH", "").strip()
    if explicit:
        return explicit if os.path.isabs(explicit) else os.path.join(base_dir, explicit)
    exact = os.path.join(base_dir, "checkpoints", "bc_" + net_stem + ".pt")
    if os.path.exists(exact):
        return exact
    if scene_id:
        scene_ckpt = os.path.join(base_dir, "checkpoints", "bc_scene_" + str(scene_id) + ".pt")
        if os.path.exists(scene_ckpt):
            return scene_ckpt
    shared = os.path.join(base_dir, "checkpoints", "bc_pretrain_latest.pt")
    if os.path.exists(shared):
        return shared
    return ""


def _resolve_bc_result(base_dir, net_stem, scene_id=""):
    explicit = os.environ.get("GCN_ENH_HQ_BC_RESULT_PATH", "").strip()
    if explicit:
        return explicit if os.path.isabs(explicit) else os.path.join(base_dir, explicit)
    exact = os.path.join(base_dir, "results", "bc_" + net_stem + "_result.txt")
    if os.path.exists(exact):
        return exact
    if scene_id:
        scene_result = os.path.join(base_dir, "results", "bc_scene_" + str(scene_id) + "_result.txt")
        if os.path.exists(scene_result):
            return scene_result
    shared = os.path.join(base_dir, "results", "bc_pretrain_result.txt")
    if os.path.exists(shared):
        return shared
    return ""


def _infer_expert_steps(result_info):
    if not result_info:
        return 0
    for key in ["expert_trans_count", "clean_steps", "policy_trans_count", "scene_ref_expert_steps"]:
        value = _safe_int(result_info.get(key, 0), 0)
        if value > 0:
            return value
    seq = result_info.get("expert_trans_sequence", "") or result_info.get("policy_trans_sequence", "")
    nums = re.findall(r"\d+", seq)
    return len(nums)


def _classify_bc_warm_start_source(base_dir, ckpt_path, net_stem, scene_id):
    if not ckpt_path:
        return "none"
    full = os.path.abspath(ckpt_path)
    exact = os.path.abspath(os.path.join(base_dir, "checkpoints", "bc_" + net_stem + ".pt"))
    if full == exact:
        return "net"
    if scene_id:
        scene_ckpt = os.path.abspath(os.path.join(base_dir, "checkpoints", "bc_scene_" + str(scene_id) + ".pt"))
        if full == scene_ckpt:
            return "scene"
    shared = os.path.abspath(os.path.join(base_dir, "checkpoints", "bc_pretrain_latest.pt"))
    if full == shared:
        return "global"
    name = os.path.basename(full)
    if name == "bc_" + net_stem + ".pt":
        return "net"
    if scene_id and name == "bc_scene_" + str(scene_id) + ".pt":
        return "scene"
    if name == "bc_pretrain_latest.pt":
        return "global"
    return "custom"


def _summarize_result(result, extra_info):
    trans = result.get_trans() if result is not None else []
    markings = result.get_markings() if result is not None else []
    return {
        "trans_count": len(trans),
        "trans_sequence": "->".join(str(t) for t in trans) if trans else "",
        "makespan": markings[-1].get_prefix() if markings else -1,
        "reach_goal": extra_info.get("reachGoal"),
        "goal_distance": extra_info.get("goalDistance"),
    }


def main():
    # 输出文件：result 记录最终结果，progress 记录过程日志。
    base_dir = os.path.dirname(__file__)
    out_path = os.path.join(base_dir, "results", "gcn_dqn_enhanced_hq_result.txt")
    progress_path = os.path.join(base_dir, "results", "gcn_dqn_enhanced_hq_progress.txt")
    try:
        applied_inline_env = apply_inline_env_overrides(INLINE_ENV_OVERRIDES, priority=INLINE_ENV_OVERRIDE_PRIORITY)
        os.makedirs(os.path.dirname(progress_path), exist_ok=True)
        # 当前运行默认读取的网文件，可按需改成 resources 下其他文件。
        input_subdir = os.environ.get("GCN_ENH_HQ_INPUT_SUBDIR", DEFAULT_GCN_ENH_HQ_INPUT_SUBDIR).strip() or "test"
        net_file = os.environ.get("GCN_ENH_HQ_NET_FILE", DEFAULT_GCN_ENH_HQ_NET_FILE).strip()
        path = os.path.join(base_dir, input_subdir, net_file)
        if not os.path.exists(path):
            msg = "ERROR\nmissing input file: " + path + "\nexpected input folder: " + os.path.join(base_dir, input_subdir)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(msg + "\n")
            print(msg, flush=True)
            return
        # 统一解析网文件并构造带驻留约束的 Petri 网对象。
        context = load_petri_net_context(path)
        p_info = context["p_info"]
        min_delay_p = context["min_delay_p"]
        max_residence_time = context["max_residence_time"]
        end = context["end"]
        pre = context["pre"]
        post = context["post"]
        petri_net = build_ttpn_with_residence(context)

        # 统计网规模与约束强度，用于自适应训练参数。
        net_stem = os.path.splitext(os.path.basename(path))[0]
        scene_id = _resolve_scene_id(net_stem)
        place_count = len(p_info)
        trans_count = len(pre[0]) if pre else 0
        constrained_count = 0
        # constrained_count：有 residenceTime 约束的库所个数。
        for val in max_residence_time:
            if val < 2 ** 31 - 1:
                constrained_count += 1
        # complexity：以库所/变迁规模最大值作为复杂度估计。
        complexity = max(place_count, trans_count)
        with open(progress_path, "w", encoding="utf-8") as f:
            f.write(format_inline_env_overrides(applied_inline_env) + "\n")

        # 训练模式开关：
        # - GCN_ENH_HQ_FAST="0" -> fast_mode=True，进入 hq-fast。
        # - 其他值（默认"1"）-> 进入 hq-full。
        fast_mode = os.environ.get("GCN_ENH_HQ_FAST", "1") == "0"#此处改为0，为训练模式
        if fast_mode:
            # 快速模式：用于快速验证流程，训练规模较小。
            train_episodes = 36
            min_steps = 90
            max_steps = 280
            rollout_count = 16
            mode = "hq-fast"
        else:
            # 全量模式：根据网规模和约束数量自适应拉高训练规模。
            # train_episodes：训练轮数。
            train_episodes = 200 #min(360, max(260, 220 + complexity * 2 + constrained_count * 3))
            # min_steps：每回合最小步数上限。
            min_steps = min(220, max(120, 90 + complexity))
            # max_steps：每回合最大步数上限。
            max_steps = min(900, max(min_steps + 260, 480 + complexity * 8 + constrained_count * 6))
            # rollout_count：推理阶段候选 rollout 数量。
            rollout_count = min(72, max(40, 28 + complexity // 2 + constrained_count // 2))
            mode = "hq-full"

        heuristic_min_steps = min_steps
        heuristic_max_steps = max_steps

        use_bc_warm_start = os.environ.get("GCN_ENH_HQ_USE_BC_WARM_START", "1") == "1"
        bc_ckpt_path = _resolve_bc_checkpoint(base_dir, net_stem, scene_id=scene_id) if use_bc_warm_start else ""
        bc_result_path = _resolve_bc_result(base_dir, net_stem, scene_id=scene_id)
        planned_bc_warm_start_source = _classify_bc_warm_start_source(base_dir, bc_ckpt_path, net_stem, scene_id)
        bc_result_info = _read_result_kv(bc_result_path)
        expert_steps = _infer_expert_steps(bc_result_info)
        expert_train_min_scale = float(os.environ.get("GCN_ENH_HQ_EXPERT_MIN_STEP_SCALE", "0.75"))
        expert_train_max_scale = float(os.environ.get("GCN_ENH_HQ_EXPERT_MAX_STEP_SCALE", "1.80"))
        expert_train_min_floor = int(os.environ.get("GCN_ENH_HQ_EXPERT_MIN_STEP_FLOOR", "24"))
        expert_train_max_floor = int(os.environ.get("GCN_ENH_HQ_EXPERT_MAX_STEP_FLOOR", "48"))
        expert_train_max_margin = int(os.environ.get("GCN_ENH_HQ_EXPERT_MAX_STEP_MIN_MARGIN", "24"))
        expert_step_scale = float(os.environ.get("GCN_ENH_HQ_EXPERT_STEP_SCALE", "2.0"))
        expert_step_min_margin = int(os.environ.get("GCN_ENH_HQ_EXPERT_STEP_MIN_MARGIN", "16"))
        step_reference_source = "heuristic"
        if expert_steps > 0:
            min_steps = max(
                expert_train_min_floor,
                int(round(float(expert_steps) * expert_train_min_scale)),
            )
            max_steps = max(
                expert_train_max_floor,
                min_steps + expert_train_max_margin,
                int(round(float(expert_steps) * expert_train_max_scale)),
            )
            step_reference_source = "expert"
        inference_max_steps = max_steps
        if expert_steps > 0:
            expert_ref_steps = max(
                expert_steps + expert_step_min_margin,
                int(round(float(expert_steps) * expert_step_scale)),
            )
            inference_max_steps = max(1, expert_ref_steps)

        # 输出当前模式与自适应参数，便于复现实验配置。
        line = "GCN-DQN enhanced HQ mode: " + mode
        print(line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        input_line = "input_subdir=" + input_subdir + " net_file=" + net_file
        print(input_line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(input_line + "\n")
        schedule_line = (
            "schedule train_episodes="
            + str(train_episodes)
            + " min_steps="
            + str(min_steps)
            + " max_steps="
            + str(max_steps)
            + " inference_max_steps="
            + str(inference_max_steps)
            + " step_source="
            + step_reference_source
            + " heuristic_min_steps="
            + str(heuristic_min_steps)
            + " heuristic_max_steps="
            + str(heuristic_max_steps)
            + " rollout_count="
            + str(rollout_count)
            + " places="
            + str(place_count)
            + " trans="
            + str(trans_count)
            + " constrained_places="
            + str(constrained_count)
            + " scene_id="
            + (scene_id if scene_id else "none")
        )
        print(schedule_line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(schedule_line + "\n")

        # 输出实际生效的驻留约束列表（无限约束不会列出）。
        constrained_places = []
        for i, val in enumerate(max_residence_time):
            if val < 2 ** 31 - 1:
                constrained_places.append(str(i) + ":" + str(val))
        constraint_line = "residence_constraints=" + (",".join(constrained_places) if constrained_places else "none")
        print(constraint_line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(constraint_line + "\n")
        # 训练后成功率评估时的 rollout 次数。
        goal_eval_rollouts = int(os.environ.get("GCN_ENH_HQ_GOAL_EVAL_ROLLOUTS", "8"))
        # 目标最小成功率阈值，低于该值会触发追加训练。
        goal_min_success = float(os.environ.get("GCN_ENH_HQ_GOAL_MIN_SUCCESS", "0.70"))
        # 为满足成功率阈值最多允许追加的训练回合数。
        extra_train_episodes = int(os.environ.get("GCN_ENH_HQ_EXTRA_TRAIN_EPISODES", "120"))
        # 命中 similar 且开启微调时，基础训练轮数缩放比例。
        similar_finetune_episode_scale = float(os.environ.get("GCN_ENH_HQ_SIMILAR_FINETUNE_EPISODE_SCALE", "0.35"))
        # 命中 similar 且开启微调时，追加训练轮数缩放比例。
        similar_finetune_extra_scale = float(os.environ.get("GCN_ENH_HQ_SIMILAR_FINETUNE_EXTRA_SCALE", "0.25"))
        # 命中 similar 且开启微调时，基础训练轮数的最小下限。
        similar_finetune_min_episodes = int(os.environ.get("GCN_ENH_HQ_SIMILAR_FINETUNE_MIN_EPISODES", "24"))
        # 命中 similar 且开启微调时，追加训练轮数的最小下限。
        similar_finetune_min_extra_episodes = int(os.environ.get("GCN_ENH_HQ_SIMILAR_FINETUNE_MIN_EXTRA_EPISODES", "12"))
        # 是否启用奖励中的时间代价缩放。
        use_reward_scaling = os.environ.get("GCN_ENH_HQ_USE_REWARD_SCALING", "1") == "1"
        # 时间代价缩放系数，越大表示单位步时间惩罚越平缓。
        reward_time_scale = float(os.environ.get("GCN_ENH_HQ_REWARD_TIME_SCALE", "1000.0"))
        # 是否启用奖励裁剪，抑制异常大回报导致的不稳定更新。
        use_reward_clip = os.environ.get("GCN_ENH_HQ_USE_REWARD_CLIP", "1") == "1"
        # 奖励裁剪绝对值上限，对应区间为 [-abs, abs]。
        reward_clip_abs = float(os.environ.get("GCN_ENH_HQ_REWARD_CLIP_ABS", "20.0"))
        # 是否使用 Huber 损失替代 MSE，提升大误差场景下稳定性。
        use_huber_loss = os.environ.get("GCN_ENH_HQ_USE_HUBER_LOSS", "1") == "1"
        # Huber 损失分段阈值参数。
        huber_beta = float(os.environ.get("GCN_ENH_HQ_HUBER_BETA", "1.0"))
        rl_verbose = os.environ.get("GCN_ENH_HQ_VERBOSE", DEFAULT_GCN_ENH_HQ_VERBOSE) == "1"
        rl_log_interval = int(os.environ.get("GCN_ENH_HQ_LOG_INTERVAL", DEFAULT_GCN_ENH_HQ_LOG_INTERVAL))
        controller_representation_enabled = os.environ.get(
            "GCN_ENH_HQ_CONTROLLER_REPRESENTATION",
            DEFAULT_GCN_ENH_HQ_CONTROLLER_REPRESENTATION,
        ) == "1"
        controller_representation_tag = "l3on" if controller_representation_enabled else "l3off"
        bc_warm_net_scale = float(os.environ.get("GCN_ENH_HQ_BC_WARM_NET_SCALE", DEFAULT_GCN_ENH_HQ_BC_WARM_NET_SCALE))
        bc_warm_scene_scale = float(os.environ.get("GCN_ENH_HQ_BC_WARM_SCENE_SCALE", DEFAULT_GCN_ENH_HQ_BC_WARM_SCENE_SCALE))
        bc_warm_global_scale = float(os.environ.get("GCN_ENH_HQ_BC_WARM_GLOBAL_SCALE", DEFAULT_GCN_ENH_HQ_BC_WARM_GLOBAL_SCALE))
        bc_warm_net_epsilon_init = float(os.environ.get("GCN_ENH_HQ_BC_WARM_NET_EPSILON_INIT", DEFAULT_GCN_ENH_HQ_BC_WARM_NET_EPSILON_INIT))
        bc_warm_scene_epsilon_init = float(os.environ.get("GCN_ENH_HQ_BC_WARM_SCENE_EPSILON_INIT", DEFAULT_GCN_ENH_HQ_BC_WARM_SCENE_EPSILON_INIT))
        bc_warm_global_epsilon_init = float(os.environ.get("GCN_ENH_HQ_BC_WARM_GLOBAL_EPSILON_INIT", DEFAULT_GCN_ENH_HQ_BC_WARM_GLOBAL_EPSILON_INIT))
        rl_epsilon_init = 1.0
        if use_bc_warm_start and bc_ckpt_path:
            if planned_bc_warm_start_source == "net":
                rl_epsilon_init = bc_warm_net_epsilon_init
            elif planned_bc_warm_start_source == "scene":
                rl_epsilon_init = bc_warm_scene_epsilon_init
            elif planned_bc_warm_start_source == "global":
                rl_epsilon_init = bc_warm_global_epsilon_init
        goal_schedule_line = (
            "goal_policy eval_rollouts="
            + str(goal_eval_rollouts)
            + " min_success="
            + str(goal_min_success)
            + " extra_train_episodes="
            + str(extra_train_episodes)
            + " reward_scaling="
            + ("1" if use_reward_scaling else "0")
            + " reward_time_scale="
            + str(reward_time_scale)
            + " reward_clip="
            + ("1" if use_reward_clip else "0")
            + " reward_clip_abs="
            + str(reward_clip_abs)
            + " huber_loss="
            + ("1" if use_huber_loss else "0")
            + " huber_beta="
            + str(huber_beta)
            + " rl_verbose="
            + ("1" if rl_verbose else "0")
            + " rl_log_interval="
            + str(rl_log_interval)
            + " controller_representation="
            + ("1" if controller_representation_enabled else "0")
            + " bc_warm_scales="
            + "net:"
            + str(bc_warm_net_scale)
            + ",scene:"
            + str(bc_warm_scene_scale)
            + ",global:"
            + str(bc_warm_global_scale)
            + " bc_warm_eps_init="
            + "net:"
            + str(bc_warm_net_epsilon_init)
            + ",scene:"
            + str(bc_warm_scene_epsilon_init)
            + ",global:"
            + str(bc_warm_global_epsilon_init)
            + " rl_epsilon_init="
            + str(rl_epsilon_init)
        )
        print(goal_schedule_line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(goal_schedule_line + "\n")
        bc_line = (
            "bc_bridge warm_start="
            + ("1" if use_bc_warm_start and bool(bc_ckpt_path) else "0")
            + " scene_id="
            + (scene_id if scene_id else "none")
            + " planned_source="
            + planned_bc_warm_start_source
            + " bc_ckpt_path="
            + (bc_ckpt_path if bc_ckpt_path else "none")
            + " bc_result_path="
            + (bc_result_path if bc_result_path else "none")
            + " expert_steps="
            + str(expert_steps)
            + " rl_epsilon_init="
            + str(rl_epsilon_init)
            + " expert_step_scale="
            + str(expert_step_scale)
            + " inference_max_steps="
            + str(inference_max_steps)
        )
        print(bc_line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(bc_line + "\n")

        # 构造 HQ 搜索器（内部包含训练与推理流程）。
        search = PetriNetGCNDQNEnhancedHQ(
            petri_net=petri_net,
            end=end,
            pre=pre,
            post=post,
            min_delay_p=min_delay_p,
            train_episodes=train_episodes,
            min_steps_per_episode=min_steps,
            max_steps_per_episode=max_steps,
            inference_max_steps_per_episode=inference_max_steps,
            rollout_count=rollout_count,
            goal_eval_rollouts=goal_eval_rollouts,
            goal_min_success_rate=goal_min_success,
            extra_train_episodes=extra_train_episodes,
            use_reward_scaling=use_reward_scaling,
            reward_time_scale=reward_time_scale,
            use_reward_clip=use_reward_clip,
            reward_clip_abs=reward_clip_abs,
            use_huber_loss=use_huber_loss,
            huber_beta=huber_beta,
            epsilon_init=rl_epsilon_init,
            verbose=rl_verbose,
            log_interval=rl_log_interval,
            controller_representation_enabled=controller_representation_enabled,
        )

        # checkpoint 签名与检索信息：
        # - signature：该网文件的唯一签名。
        # - profile：网的规模特征，用于“相近网”回退选择。
        signature = build_signature(path, context)
        profile = build_profile(context)
        checkpoint_prefix = "gcn_dqn_enhanced_hq_" + controller_representation_tag
        ckpt_path = checkpoint_path(base_dir, checkpoint_prefix, signature)

        # checkpoint 复用开关：
        # - GCN_ENH_HQ_REUSE=1（默认）优先复用已有模型。
        # - GCN_ENH_HQ_REUSE=0 强制跳过加载并重新训练。
        # - GCN_ENH_HQ_REUSE_SIMILAR=1（默认）允许回退到相近网模型。
        # - GCN_ENH_HQ_FINETUNE_ON_SIMILAR=1 时，命中 similar 后继续微调训练。
        reuse_checkpoint = os.environ.get("GCN_ENH_HQ_REUSE", "1") == "1"#此处设为0，则强制跳过加载并重新训练。
        reuse_similar = os.environ.get("GCN_ENH_HQ_REUSE_SIMILAR", "1") == "1"
        finetune_on_similar = os.environ.get("GCN_ENH_HQ_FINETUNE_ON_SIMILAR", "0") == "0"#此处设为0，则命中similar后加载模型，然后训练。
        loaded_checkpoint = False
        checkpoint_mode = "none"
        checkpoint_score = -1.0
        if reuse_checkpoint:
            # 选择 checkpoint：exact > similar > none。
            selected = find_checkpoint(base_dir, checkpoint_prefix, signature, profile, allow_similar=reuse_similar)
            checkpoint_mode = selected["mode"]
            checkpoint_score = selected["score"]
            load_path = selected["path"]
            if load_path and os.path.exists(load_path):
                # 加载模型参数；若结构不完全一致，则按 key/shape 兼容加载。
                saved = torch.load(load_path, map_location="cpu")
                load_compatible_state(search.policy_net, saved.get("policy_state", {}))
                load_compatible_state(search.target_net, saved.get("target_state", {}))
                optimizer_state = saved.get("optimizer_state")
                if optimizer_state is not None:
                    try:
                        search.optimizer.load_state_dict(optimizer_state)
                    except BaseException:
                        pass
                search.best_train_makespan = saved.get("best_train_makespan", 2 ** 31 - 1)
                search.best_train_trans = saved.get("best_train_trans", [])
                train_info = saved.get("train_info", {})
                search.extra_info["trainSteps"] = train_info.get("trainSteps", 0)
                search.extra_info["bestTrainMakespan"] = search.best_train_makespan if search.best_train_makespan < 2 ** 31 - 1 else -1
                search.extra_info["bestTrainTransCount"] = len(search.best_train_trans)
                search.extra_info["avgLoss"] = train_info.get("avgLoss", 0.0)
                search.is_trained = True
                loaded_checkpoint = True
                if checkpoint_mode == "similar" and finetune_on_similar:
                    search.is_trained = False
                    # similar 微调策略：按比例缩短训练计划，并保留最小训练下限。
                    scaled_train_episodes = max(
                        1,
                        max(similar_finetune_min_episodes, int(round(search.train_episodes * similar_finetune_episode_scale))),
                    )
                    scaled_extra_train_episodes = max(
                        0,
                        max(
                            similar_finetune_min_extra_episodes,
                            int(round(search.extra_train_episodes * similar_finetune_extra_scale)),
                        ),
                    )
                    search.train_episodes = min(search.train_episodes, scaled_train_episodes)
                    search.extra_train_episodes = min(search.extra_train_episodes, scaled_extra_train_episodes)

        bc_warm_started = False
        bc_warm_start_source = "none"
        bc_warm_scale = 1.0
        bc_base_train_episodes = search.train_episodes
        bc_base_extra_train_episodes = search.extra_train_episodes
        if (not loaded_checkpoint) and use_bc_warm_start and bc_ckpt_path and os.path.exists(bc_ckpt_path):
            saved_bc = torch.load(bc_ckpt_path, map_location="cpu")
            bc_state = saved_bc.get("model_state", {})
            load_compatible_state(search.policy_net, bc_state)
            load_compatible_state(search.target_net, bc_state)
            bc_warm_started = True
            bc_warm_start_source = _classify_bc_warm_start_source(base_dir, bc_ckpt_path, net_stem, scene_id)
            if bc_warm_start_source == "net":
                bc_warm_scale = bc_warm_net_scale
            elif bc_warm_start_source == "scene":
                bc_warm_scale = bc_warm_scene_scale
            elif bc_warm_start_source == "global":
                bc_warm_scale = bc_warm_global_scale
            else:
                bc_warm_scale = 1.0
            search.train_episodes = max(1, int(round(search.train_episodes * bc_warm_scale)))
            search.extra_train_episodes = max(0, int(round(search.extra_train_episodes * bc_warm_scale)))

        # 输出 checkpoint 命中情况。
        checkpoint_line = "checkpoint_loaded=" + ("1" if loaded_checkpoint else "0")
        print(checkpoint_line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(checkpoint_line + "\n")
        checkpoint_select_line = "checkpoint_mode=" + checkpoint_mode + " checkpoint_score=" + str(checkpoint_score)
        print(checkpoint_select_line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(checkpoint_select_line + "\n")
        finetune_line = "finetune_on_similar=" + ("1" if finetune_on_similar else "0")
        print(finetune_line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(finetune_line + "\n")
        bc_warm_line = (
            "bc_warm_started="
            + ("1" if bc_warm_started else "0")
            + " bc_warm_start_source="
            + bc_warm_start_source
            + " bc_warm_scale="
            + str(bc_warm_scale)
        )
        print(bc_warm_line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(bc_warm_line + "\n")
        # 输出最终生效的微调训练计划，便于确认是否已按 similar 策略缩放。
        finetune_schedule_line = (
            "finetune_schedule train_episodes="
            + str(search.train_episodes)
            + " extra_train_episodes="
            + str(search.extra_train_episodes)
            + " base_train_episodes="
            + str(bc_base_train_episodes)
            + " base_extra_train_episodes="
            + str(bc_base_extra_train_episodes)
            + " inference_max_steps="
            + str(search.inference_max_steps_per_episode)
        )
        print(finetune_schedule_line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(finetune_schedule_line + "\n")
        if loaded_checkpoint and (checkpoint_mode == "similar") and finetune_on_similar:
            train_plan_line = "train_plan=finetune_from_similar_checkpoint"
        elif loaded_checkpoint:
            train_plan_line = "train_plan=skip_train_use_loaded_checkpoint mode=" + checkpoint_mode
        elif bc_warm_started:
            train_plan_line = "train_plan=run_train_from_bc_warm_start"
        else:
            train_plan_line = "train_plan=run_train_from_scratch"
        print(train_plan_line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(train_plan_line + "\n")

        # 执行搜索（内部若未加载模型会先训练，再推理得到最终路径）。
        start = time.perf_counter()
        warm_start_infer_elapsed = 0.0
        warm_start_infer_summary = {
            "trans_count": 0,
            "trans_sequence": "",
            "makespan": -1,
            "reach_goal": False,
            "goal_distance": -1,
        }
        if (not loaded_checkpoint) and bc_warm_started:
            warm_start_line = "warm_start_inference=run_before_finetune"
            print(warm_start_line, flush=True)
            with open(progress_path, "a", encoding="utf-8") as f:
                f.write(warm_start_line + "\n")
            search.is_trained = True
            warm_start = time.perf_counter()
            warm_start_result = search.search()
            warm_start_infer_elapsed = time.perf_counter() - warm_start
            warm_start_infer_summary = _summarize_result(warm_start_result, dict(search.get_extra_info()))
            search.is_trained = False
            warm_start_result_line = (
                "warm_start_infer reach_goal="
                + str(warm_start_infer_summary["reach_goal"])
                + " goal_distance="
                + str(warm_start_infer_summary["goal_distance"])
                + " makespan="
                + str(warm_start_infer_summary["makespan"])
                + " trans_count="
                + str(warm_start_infer_summary["trans_count"])
                + " elapsed="
                + format(warm_start_infer_elapsed, ".6f")
                + "s"
            )
            print(warm_start_result_line, flush=True)
            with open(progress_path, "a", encoding="utf-8") as f:
                f.write(warm_start_result_line + "\n")
                f.write("warm_start_infer_trans_sequence=" + warm_start_infer_summary["trans_sequence"] + "\n")

        start = time.perf_counter()
        result = search.search()
        elapsed = time.perf_counter() - start
        extra = search.get_extra_info()

        # 保存 checkpoint，供后续相同/相近网复用。
        to_save = {
            "signature": signature,
            "profile": profile,
            "policy_state": search.policy_net.state_dict(),
            "target_state": search.target_net.state_dict(),
            "optimizer_state": search.optimizer.state_dict(),
            "best_train_makespan": search.best_train_makespan,
            "best_train_trans": search.best_train_trans,
            "train_info": {
                "trainSteps": extra.get("trainSteps", 0),
                "avgLoss": extra.get("avgLoss", 0.0),
            },
        }
        torch.save(to_save, ckpt_path)

        # 汇总并写出结果。
        trans = result.get_trans()
        markings = result.get_markings()
        out = "elapsed:" + format(elapsed, ".6f") + "s\n"
        out += format_inline_env_overrides(applied_inline_env) + "\n"
        out += "trans_count:" + str(len(trans)) + "\n"
        out += "trans_sequence:" + ("->".join(str(t) for t in trans) if trans else "") + "\n"
        out += "makespan:" + str(markings[-1].get_prefix() if markings else -1) + "\n"
        out += "reach_goal:" + str(extra.get("reachGoal")) + "\n"
        out += "goal_distance:" + str(extra.get("goalDistance")) + "\n"
        out += "train_steps:" + str(extra.get("trainSteps")) + "\n"
        out += "best_train_makespan:" + str(extra.get("bestTrainMakespan")) + "\n"
        out += "best_train_trans_count:" + str(extra.get("bestTrainTransCount")) + "\n"
        out += "avg_loss:" + str(extra.get("avgLoss")) + "\n"
        out += "controller_representation_enabled:" + ("1" if controller_representation_enabled else "0") + "\n"
        out += "controller_representation_tag:" + controller_representation_tag + "\n"
        out += "train_success_rate:" + str(extra.get("trainSuccessRate")) + "\n"
        out += "rollout_success_rate:" + str(extra.get("rolloutSuccessRate")) + "\n"
        out += "replay_success_rate:" + str(extra.get("replaySuccessRate")) + "\n"
        out += "extra_train_episodes:" + str(extra.get("extraTrainEpisodes")) + "\n"
        out += "inference_max_steps:" + str(search.inference_max_steps_per_episode) + "\n"
        out += "scene_id:" + (scene_id if scene_id else "") + "\n"
        out += "step_reference_source:" + step_reference_source + "\n"
        out += "heuristic_min_steps:" + str(heuristic_min_steps) + "\n"
        out += "heuristic_max_steps:" + str(heuristic_max_steps) + "\n"
        out += "bc_warm_started:" + str(bc_warm_started) + "\n"
        out += "bc_warm_start_source:" + bc_warm_start_source + "\n"
        out += "bc_warm_scale:" + str(bc_warm_scale) + "\n"
        out += "rl_epsilon_init:" + str(rl_epsilon_init) + "\n"
        out += "base_train_episodes:" + str(bc_base_train_episodes) + "\n"
        out += "base_extra_train_episodes:" + str(bc_base_extra_train_episodes) + "\n"
        out += "bc_checkpoint_path:" + (bc_ckpt_path if bc_ckpt_path else "") + "\n"
        out += "bc_result_path:" + (bc_result_path if bc_result_path else "") + "\n"
        out += "bc_expert_steps:" + str(expert_steps) + "\n"
        out += "warm_start_infer_elapsed:" + format(warm_start_infer_elapsed, ".6f") + "s\n"
        out += "warm_start_infer_trans_count:" + str(warm_start_infer_summary["trans_count"]) + "\n"
        out += "warm_start_infer_trans_sequence:" + warm_start_infer_summary["trans_sequence"] + "\n"
        out += "warm_start_infer_makespan:" + str(warm_start_infer_summary["makespan"]) + "\n"
        out += "warm_start_infer_reach_goal:" + str(warm_start_infer_summary["reach_goal"]) + "\n"
        out += "warm_start_infer_goal_distance:" + str(warm_start_infer_summary["goal_distance"]) + "\n"
        out += "checkpoint_path:" + ckpt_path + "\n"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(out)
        print(out, flush=True)
    except BaseException:
        # 异常信息写入结果文件，便于排障。
        err = "ERROR\n" + traceback.format_exc()
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(err)
        print(err, flush=True)


if __name__ == "__main__":
    main()
