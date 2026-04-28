import os
import re
import sys
import time
import traceback

import torch  # pyright: ignore[reportMissingImports]

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from python_port.entrypoint_env import apply_inline_env_overrides, format_inline_env_overrides
from python_port.petri_net_io.utils.checkpoint_selector import (
    build_profile,
    build_signature,
    checkpoint_path,
    load_compatible_state,
)
from python_port.imitation.il_checkpoint import classify_il_artifact, normalize_il_mode, resolve_il_checkpoint, resolve_il_result
from python_port.petri_net_io.utils.net_loader import build_ttpn_with_residence, load_petri_net_context
from python_port.petri_net_platform.search.petri_net_gcn_ppo import PetriNetGCNPPOEnhancedHQ
from python_port.petri_net_platform.search.petri_net_gcn_ppo_classic import PetriNetGCNPPOClassicHQ
from python_port.scene_utils import infer_scene_id


# ===== USER-TUNABLE DEFAULTS: EDIT HERE FIRST =====
# 单网 PPO 应用入口：
# 1. 默认优先加载场景级 PPO checkpoint 直接推理
# 2. 若未找到场景级 PPO checkpoint，则退回到 IL 热启动（优先 DAgger，其次 BC）/ 从零开始的单网 PPO 训练
# 输入网文件所在子目录，对应 base_dir 下的相对路径。
DEFAULT_GCN_PPO_HQ_INPUT_SUBDIR = "test"
# 要执行的单个网文件名。
DEFAULT_GCN_PPO_HQ_NET_FILE = "1-1-13.txt"
# Leave empty to auto-infer from net file name like 1-6.txt -> scene 1.
# 显式指定 scene_id；留空则按文件名自动推断。
DEFAULT_GCN_PPO_HQ_SCENE_ID = ""
# PPO 训练器变体：enhanced 为当前工程版，classic 为迁移进来的旧版风格 PPO 壳。
DEFAULT_GCN_PPO_HQ_VARIANT = "enhanced"
# 是否打印 PPO 训练/推理日志，1 为打印，0 为静默。
DEFAULT_GCN_PPO_HQ_VERBOSE = "1"
# 训练日志打印间隔，数值越小日志越密。
DEFAULT_GCN_PPO_HQ_LOG_INTERVAL = "5"
# IL 热启动后的 PPO 微调预算缩放：同网 / 同场景 / 全局共享。
# 精确同网热启动时，对单网 PPO 训练轮次的缩放系数。
DEFAULT_GCN_PPO_HQ_BC_WARM_NET_SCALE = "0.40"
# 同场景热启动时，对单网 PPO 训练轮次的缩放系数。
DEFAULT_GCN_PPO_HQ_BC_WARM_SCENE_SCALE = "0.65"
# 全局共享热启动时，对单网 PPO 训练轮次的缩放系数。
DEFAULT_GCN_PPO_HQ_BC_WARM_GLOBAL_SCALE = "0.85"
# 是否启用第三层控制器感知图特征，1 开启，0 关闭。
DEFAULT_GCN_PPO_HQ_CONTROLLER_REPRESENTATION = "1"
# 是否优先加载场景级 PPO 模型，1 开启，0 关闭。
DEFAULT_GCN_PPO_HQ_USE_SCENE_POLICY = "1"
# 加载场景级 PPO 后，是否再做少量单网微调，1 开启，0 关闭。
DEFAULT_GCN_PPO_HQ_ENABLE_SCENE_FINETUNE = "0"
# 场景级 PPO 微调时，对单网 PPO 训练轮次的缩放系数。
DEFAULT_GCN_PPO_HQ_SCENE_FINETUNE_SCALE = "0.25"
# classic 变体专用：按步数收集 rollout，再做 mini-batch PPO 更新。
DEFAULT_GCN_PPO_HQ_STEPS_PER_EPOCH = "1024"
DEFAULT_GCN_PPO_HQ_MINIBATCH_SIZE = "128"
DEFAULT_GCN_PPO_HQ_TARGET_KL = "0.05"
DEFAULT_GCN_PPO_HQ_ENTROPY_START = "0.03"
DEFAULT_GCN_PPO_HQ_ENTROPY_END = "0.01"
DEFAULT_GCN_PPO_HQ_TEMPERATURE_START = "1.3"
DEFAULT_GCN_PPO_HQ_TEMPERATURE_END = "1.0"
# ===== END USER-TUNABLE DEFAULTS =====

# 脚本内环境变量覆盖：
# 直接在这里填写键值对，可在代码里调整单网 PPO 应用 / 微调参数。
INLINE_ENV_OVERRIDE_PRIORITY = "code"
INLINE_ENV_OVERRIDES = {
    # "GCN_PPO_HQ_IL_MODE": "auto",
    # "GCN_PPO_HQ_INPUT_SUBDIR": "test",
    # "GCN_PPO_HQ_NET_FILE": "1-2-13.txt",
}


def _read_result_kv(path):
    # 读取 IL 结果文件，用于提取专家步数等训练预算参考信息。
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


def _normalize_ppo_variant(value):
    text = str(value or "enhanced").strip().lower()
    if text == "classic":
        return "classic"
    return "enhanced"


def _resolve_scene_id(net_stem):
    # 优先读取显式配置；为空时再根据网文件名自动推断场景编号。
    explicit = os.environ.get("GCN_PPO_HQ_SCENE_ID", DEFAULT_GCN_PPO_HQ_SCENE_ID).strip()
    if explicit:
        return explicit
    return infer_scene_id(net_stem)

def _resolve_ppo_scene_checkpoint(base_dir, scene_id="", ppo_variant="enhanced"):
    # 单网应用阶段默认优先复用场景级 PPO 模型。
    explicit = os.environ.get("GCN_PPO_HQ_PPO_SCENE_CKPT_PATH", "").strip()
    if explicit:
        return explicit if os.path.isabs(explicit) else os.path.join(base_dir, explicit)
    if scene_id:
        scene_prefix = "ppo_scene_" if ppo_variant == "enhanced" else "ppo_" + ppo_variant + "_scene_"
        scene_ckpt = os.path.join(base_dir, "checkpoints", scene_prefix + str(scene_id) + ".pt")
        if os.path.exists(scene_ckpt):
            return scene_ckpt
    return ""


def _infer_expert_steps(result_info):
    # 从 IL 结果文件中提取专家轨迹长度，用于推导 PPO 的训练步数与推理步数上限。
    if not result_info:
        return 0
    for key in ["expert_trans_count", "clean_steps", "policy_trans_count", "scene_ref_expert_steps"]:
        value = _safe_int(result_info.get(key, 0), 0)
        if value > 0:
            return value
    seq = result_info.get("expert_trans_sequence", "") or result_info.get("policy_trans_sequence", "")
    nums = re.findall(r"\d+", seq)
    return len(nums)


def _summarize_result(result, extra_info):
    # 将一次搜索结果压缩成日志友好的摘要。
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
    base_dir = os.path.dirname(__file__)
    try:
        applied_inline_env = apply_inline_env_overrides(INLINE_ENV_OVERRIDES, priority=INLINE_ENV_OVERRIDE_PRIORITY)
        ppo_variant = _normalize_ppo_variant(os.environ.get("GCN_PPO_HQ_VARIANT", DEFAULT_GCN_PPO_HQ_VARIANT))
        result_prefix = "gcn_ppo_hq" if ppo_variant == "enhanced" else "gcn_ppo_" + ppo_variant + "_hq"
        out_path = os.path.join(base_dir, "results", result_prefix + "_result.txt")
        progress_path = os.path.join(base_dir, "results", result_prefix + "_progress.txt")
        # 输入与场景参数：
        # GCN_PPO_HQ_INPUT_SUBDIR：输入网文件所在目录
        # GCN_PPO_HQ_NET_FILE：目标网文件名
        input_subdir = os.environ.get("GCN_PPO_HQ_INPUT_SUBDIR", DEFAULT_GCN_PPO_HQ_INPUT_SUBDIR).strip() or "test"
        net_file = os.environ.get("GCN_PPO_HQ_NET_FILE", DEFAULT_GCN_PPO_HQ_NET_FILE).strip()
        path = os.path.join(base_dir, input_subdir, net_file)
        if not os.path.exists(path):
            msg = "ERROR\nmissing input file: " + path + "\nexpected input folder: " + os.path.join(base_dir, input_subdir)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(msg + "\n")
            print(msg, flush=True)
            return

        context = load_petri_net_context(path)
        p_info = context["p_info"]
        min_delay_p = context["min_delay_p"]
        max_residence_time = context["max_residence_time"]
        end = context["end"]
        pre = context["pre"]
        post = context["post"]
        petri_net = build_ttpn_with_residence(context)

        net_stem = os.path.splitext(os.path.basename(path))[0]
        scene_id = _resolve_scene_id(net_stem)
        place_count = len(p_info)
        trans_count = len(pre[0]) if pre else 0
        constrained_count = 0
        for val in max_residence_time:
            if val < 2 ** 31 - 1:
                constrained_count += 1
        complexity = max(place_count, trans_count)
        os.makedirs(os.path.dirname(progress_path), exist_ok=True)
        with open(progress_path, "w", encoding="utf-8") as f:
            f.write("")

        # 先根据网规模给出一套启发式训练预算；如果能从 IL 结果中拿到专家步数，再进一步覆盖。
        # 模式参数：
        # GCN_PPO_HQ_FAST：
        #   当前代码里值为 "0" 时走 hq-fast，值为 "1" 时走 hq-full
        #   这是历史命名遗留，含义与字面相反，调试时需要特别注意
        fast_mode = os.environ.get("GCN_PPO_HQ_FAST", "1") == "0"
        if fast_mode:
            train_iterations = 12
            extra_train_iterations = 6
            min_steps = 90
            max_steps = 280
            mode = "hq-fast"
        else:
            train_iterations = 48
            extra_train_iterations = 18
            min_steps = min(220, max(120, 90 + complexity))
            max_steps = min(900, max(min_steps + 260, 480 + complexity * 8 + constrained_count * 6))
            mode = "hq-full"

        heuristic_min_steps = min_steps
        heuristic_max_steps = max_steps

        # IL 热启动参数：
        # GCN_PPO_HQ_IL_MODE：auto / bc / dagger，auto 时优先尝试 DAgger，其次回退到 BC
        # GCN_PPO_HQ_IL_CKPT_PATH：显式指定 IL checkpoint 路径
        # GCN_PPO_HQ_IL_RESULT_PATH：显式指定 IL 结果文件路径
        # GCN_PPO_HQ_USE_BC_WARM_START：旧开关，保留为兼容项；当其为 0 时禁用所有 IL 热启动
        use_bc_warm_start = os.environ.get("GCN_PPO_HQ_USE_BC_WARM_START", "1") == "1"
        il_mode = normalize_il_mode(os.environ.get("GCN_PPO_HQ_IL_MODE", "auto"))
        il_ckpt_path = (
            resolve_il_checkpoint(
                base_dir,
                il_mode,
                net_stem=net_stem,
                scene_id=scene_id,
                explicit=os.environ.get("GCN_PPO_HQ_IL_CKPT_PATH", "").strip(),
            )
            if use_bc_warm_start
            else ""
        )
        il_result_path = resolve_il_result(
            base_dir,
            il_mode,
            net_stem=net_stem,
            scene_id=scene_id,
            explicit=os.environ.get("GCN_PPO_HQ_IL_RESULT_PATH", "").strip(),
        )
        planned_il_method, planned_il_warm_start_source = classify_il_artifact(
            base_dir,
            il_ckpt_path,
            net_stem=net_stem,
            scene_id=scene_id,
        )
        il_result_info = _read_result_kv(il_result_path)
        expert_steps = _infer_expert_steps(il_result_info)
        # 专家步数驱动的训练/推理步长预算：
        # GCN_PPO_HQ_EXPERT_MIN_STEP_SCALE：专家步数映射到训练最小步数的缩放
        # GCN_PPO_HQ_EXPERT_MAX_STEP_SCALE：专家步数映射到训练最大步数的缩放
        # GCN_PPO_HQ_EXPERT_MIN_STEP_FLOOR：训练最小步数下界
        # GCN_PPO_HQ_EXPERT_MAX_STEP_FLOOR：训练最大步数下界
        # GCN_PPO_HQ_EXPERT_MAX_STEP_MIN_MARGIN：训练 max_steps 至少比 min_steps 多出的步数
        # GCN_PPO_HQ_EXPERT_STEP_SCALE：专家步数映射到推理步数上限的缩放
        # GCN_PPO_HQ_EXPERT_STEP_MIN_MARGIN：推理步数至少比专家步数多出的步数
        expert_train_min_scale = float(os.environ.get("GCN_PPO_HQ_EXPERT_MIN_STEP_SCALE", "0.75"))
        expert_train_max_scale = float(os.environ.get("GCN_PPO_HQ_EXPERT_MAX_STEP_SCALE", "1.80"))
        expert_train_min_floor = int(os.environ.get("GCN_PPO_HQ_EXPERT_MIN_STEP_FLOOR", "24"))
        expert_train_max_floor = int(os.environ.get("GCN_PPO_HQ_EXPERT_MAX_STEP_FLOOR", "48"))
        expert_train_max_margin = int(os.environ.get("GCN_PPO_HQ_EXPERT_MAX_STEP_MIN_MARGIN", "24"))
        expert_step_scale = float(os.environ.get("GCN_PPO_HQ_EXPERT_STEP_SCALE", "2.0"))
        expert_step_min_margin = int(os.environ.get("GCN_PPO_HQ_EXPERT_STEP_MIN_MARGIN", "16"))
        step_reference_source = "heuristic"
        if expert_steps > 0:
            min_steps = max(expert_train_min_floor, int(round(float(expert_steps) * expert_train_min_scale)))
            max_steps = max(
                expert_train_max_floor,
                min_steps + expert_train_max_margin,
                int(round(float(expert_steps) * expert_train_max_scale)),
            )
            step_reference_source = "expert"
        inference_max_steps = max_steps
        if expert_steps > 0:
            inference_max_steps = max(
                expert_steps + expert_step_min_margin,
                int(round(float(expert_steps) * expert_step_scale)),
            )

        # PPO 训练超参数：
        # GCN_PPO_HQ_ROLLOUT_EPISODES_PER_ITER：每个 PPO iteration 收集多少条 rollout
        # GCN_PPO_HQ_UPDATE_EPOCHS：每个 iteration 的 PPO 更新轮数
        # GCN_PPO_HQ_REWARD_TIME_SCALE：时间代价缩放，越大表示时间惩罚越温和
        # GCN_PPO_HQ_REWARD_CLIP_ABS：奖励裁剪绝对值上限
        ppo_rollout_episodes = int(os.environ.get("GCN_PPO_HQ_ROLLOUT_EPISODES_PER_ITER", "12"))
        ppo_update_epochs = int(os.environ.get("GCN_PPO_HQ_UPDATE_EPOCHS", "6"))
        reward_time_scale = float(os.environ.get("GCN_PPO_HQ_REWARD_TIME_SCALE", "1000.0"))
        reward_clip_abs = float(os.environ.get("GCN_PPO_HQ_REWARD_CLIP_ABS", "20.0"))
        classic_config = {
            "steps_per_epoch": int(os.environ.get("GCN_PPO_HQ_STEPS_PER_EPOCH", DEFAULT_GCN_PPO_HQ_STEPS_PER_EPOCH)),
            "minibatch_size": int(os.environ.get("GCN_PPO_HQ_MINIBATCH_SIZE", DEFAULT_GCN_PPO_HQ_MINIBATCH_SIZE)),
            "target_kl": float(os.environ.get("GCN_PPO_HQ_TARGET_KL", DEFAULT_GCN_PPO_HQ_TARGET_KL)),
            "entropy_coef_start": float(
                os.environ.get("GCN_PPO_HQ_ENTROPY_START", DEFAULT_GCN_PPO_HQ_ENTROPY_START)
            ),
            "entropy_coef_end": float(os.environ.get("GCN_PPO_HQ_ENTROPY_END", DEFAULT_GCN_PPO_HQ_ENTROPY_END)),
            "temperature_start": float(
                os.environ.get("GCN_PPO_HQ_TEMPERATURE_START", DEFAULT_GCN_PPO_HQ_TEMPERATURE_START)
            ),
            "temperature_end": float(
                os.environ.get("GCN_PPO_HQ_TEMPERATURE_END", DEFAULT_GCN_PPO_HQ_TEMPERATURE_END)
            ),
        }
        # 日志与第三层图表示开关：
        # GCN_PPO_HQ_VERBOSE：是否打印细日志
        # GCN_PPO_HQ_LOG_INTERVAL：日志间隔
        # GCN_PPO_HQ_CONTROLLER_REPRESENTATION：是否启用第三层控制器感知图特征
        ppo_verbose = os.environ.get("GCN_PPO_HQ_VERBOSE", DEFAULT_GCN_PPO_HQ_VERBOSE) == "1"
        ppo_log_interval = int(os.environ.get("GCN_PPO_HQ_LOG_INTERVAL", DEFAULT_GCN_PPO_HQ_LOG_INTERVAL))
        controller_representation_enabled = os.environ.get(
            "GCN_PPO_HQ_CONTROLLER_REPRESENTATION",
            DEFAULT_GCN_PPO_HQ_CONTROLLER_REPRESENTATION,
        ) == "1"
        # 场景级 PPO 复用参数：
        # GCN_PPO_HQ_USE_SCENE_POLICY：是否优先加载 ppo_scene_<scene_id>.pt
        # GCN_PPO_HQ_PPO_SCENE_CKPT_PATH：显式指定场景级 PPO checkpoint 路径
        # GCN_PPO_HQ_ENABLE_SCENE_FINETUNE：加载场景 PPO 后是否继续做单网微调
        # GCN_PPO_HQ_SCENE_FINETUNE_SCALE：单网微调的训练预算缩放
        use_scene_policy = os.environ.get(
            "GCN_PPO_HQ_USE_SCENE_POLICY",
            DEFAULT_GCN_PPO_HQ_USE_SCENE_POLICY,
        ) == "1"
        enable_scene_finetune = os.environ.get(
            "GCN_PPO_HQ_ENABLE_SCENE_FINETUNE",
            DEFAULT_GCN_PPO_HQ_ENABLE_SCENE_FINETUNE,
        ) == "1"
        scene_finetune_scale = float(
            os.environ.get(
                "GCN_PPO_HQ_SCENE_FINETUNE_SCALE",
                DEFAULT_GCN_PPO_HQ_SCENE_FINETUNE_SCALE,
            )
        )
        controller_representation_tag = "l3on" if controller_representation_enabled else "l3off"
        warm_scale_net = float(os.environ.get("GCN_PPO_HQ_BC_WARM_NET_SCALE", DEFAULT_GCN_PPO_HQ_BC_WARM_NET_SCALE))
        warm_scale_scene = float(os.environ.get("GCN_PPO_HQ_BC_WARM_SCENE_SCALE", DEFAULT_GCN_PPO_HQ_BC_WARM_SCENE_SCALE))
        warm_scale_global = float(os.environ.get("GCN_PPO_HQ_BC_WARM_GLOBAL_SCALE", DEFAULT_GCN_PPO_HQ_BC_WARM_GLOBAL_SCALE))
        scene_ckpt_path = (
            _resolve_ppo_scene_checkpoint(base_dir, scene_id=scene_id, ppo_variant=ppo_variant)
            if use_scene_policy
            else ""
        )

        line = "GCN-PPO HQ mode: " + mode + " variant=" + ppo_variant
        print(line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.write(format_inline_env_overrides(applied_inline_env) + "\n")
        input_line = "input_subdir=" + input_subdir + " net_file=" + net_file
        print(input_line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(input_line + "\n")
        schedule_line = (
            "schedule train_iterations="
            + str(train_iterations)
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
            + " rollout_episodes_per_iter="
            + str(ppo_rollout_episodes)
            + " ppo_update_epochs="
            + str(ppo_update_epochs)
            + " places="
            + str(place_count)
            + " trans="
            + str(trans_count)
            + " constrained_places="
            + str(constrained_count)
        )
        print(schedule_line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(schedule_line + "\n")
        rep_line = "controller_representation=" + ("1" if controller_representation_enabled else "0")
        print(rep_line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(rep_line + "\n")
            f.write("ppo_variant=" + ppo_variant + "\n")
        if ppo_variant == "classic":
            classic_line = (
                "classic_config steps_per_epoch="
                + str(classic_config["steps_per_epoch"])
                + " minibatch_size="
                + str(classic_config["minibatch_size"])
                + " target_kl="
                + str(classic_config["target_kl"])
                + " entropy_start="
                + str(classic_config["entropy_coef_start"])
                + " entropy_end="
                + str(classic_config["entropy_coef_end"])
                + " temperature_start="
                + str(classic_config["temperature_start"])
                + " temperature_end="
                + str(classic_config["temperature_end"])
            )
            print(classic_line, flush=True)
            with open(progress_path, "a", encoding="utf-8") as f:
                f.write(classic_line + "\n")
        scene_policy_line = (
            "scene_policy_enabled="
            + ("1" if use_scene_policy else "0")
            + " scene_checkpoint_path="
            + (scene_ckpt_path if scene_ckpt_path else "")
            + " scene_finetune_enabled="
            + ("1" if enable_scene_finetune else "0")
            + " scene_finetune_scale="
            + str(scene_finetune_scale)
        )
        print(scene_policy_line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(scene_policy_line + "\n")

        common_search_kwargs = dict(
            petri_net=petri_net,
            end=end,
            pre=pre,
            post=post,
            min_delay_p=min_delay_p,
            train_iterations=train_iterations,
            rollout_episodes_per_iter=ppo_rollout_episodes,
            ppo_update_epochs=ppo_update_epochs,
            min_steps_per_episode=min_steps,
            max_steps_per_episode=max_steps,
            inference_max_steps_per_episode=inference_max_steps,
            goal_eval_rollouts=1,
            goal_min_success_rate=0.7,
            extra_train_iterations=extra_train_iterations,
            use_reward_scaling=True,
            reward_time_scale=reward_time_scale,
            use_reward_clip=True,
            reward_clip_abs=reward_clip_abs,
            verbose=ppo_verbose,
            log_interval=ppo_log_interval,
            controller_representation_enabled=controller_representation_enabled,
        )
        if ppo_variant == "classic":
            search = PetriNetGCNPPOClassicHQ(
                **common_search_kwargs,
                steps_per_epoch=classic_config["steps_per_epoch"],
                minibatch_size=classic_config["minibatch_size"],
                target_kl=classic_config["target_kl"],
                entropy_coef_start=classic_config["entropy_coef_start"],
                entropy_coef_end=classic_config["entropy_coef_end"],
                temperature_start=classic_config["temperature_start"],
                temperature_end=classic_config["temperature_end"],
            )
        else:
            search = PetriNetGCNPPOEnhancedHQ(**common_search_kwargs)

        base_train_iterations = search.train_iterations
        base_extra_train_iterations = search.extra_train_iterations
        ppo_scene_loaded = False
        ppo_scene_finetune_applied = False
        il_warm_started = False
        il_warm_start_source = "none"
        il_warm_method = planned_il_method
        il_warm_scale = 1.0
        # 优先级：
        # 1. 如果存在场景级 PPO checkpoint，则优先加载它
        # 2. 若同时开启 scene_finetune，则在此基础上再做少量单网 PPO 微调
        # 3. 只有在没有场景级 PPO checkpoint 时，才退回 IL 热启动（优先 DAgger，再回退 BC）
        if scene_ckpt_path and os.path.exists(scene_ckpt_path):
            saved_scene = torch.load(scene_ckpt_path, map_location="cpu")
            actor_state = {}
            critic_state = {}
            if isinstance(saved_scene, dict):
                actor_state = (
                    saved_scene.get("actor_state")
                    or saved_scene.get("model_state")
                    or saved_scene.get("policy_state")
                    or {}
                )
                critic_state = (
                    saved_scene.get("critic_state")
                    or saved_scene.get("value_state")
                    or {}
                )
            load_compatible_state(search.model.actor_net, actor_state)
            load_compatible_state(search.model.value_head, critic_state)
            ppo_scene_loaded = True
            if enable_scene_finetune:
                ppo_scene_finetune_applied = True
                search.train_iterations = max(1, int(round(search.train_iterations * scene_finetune_scale)))
                search.extra_train_iterations = max(0, int(round(search.extra_train_iterations * scene_finetune_scale)))
            else:
                # 直接把搜索器标记为“已训练”，后续 search() 将只执行推理。
                search.is_trained = True
        elif use_bc_warm_start and il_ckpt_path and os.path.exists(il_ckpt_path):
            saved_il = torch.load(il_ckpt_path, map_location="cpu")
            actor_state = {}
            if isinstance(saved_il, dict):
                actor_state = (
                    saved_il.get("model_state")
                    or saved_il.get("policy_state")
                    or saved_il.get("actor_state")
                    or {}
                )
            load_compatible_state(search.model.actor_net, actor_state)
            il_warm_started = True
            il_warm_method, il_warm_start_source = classify_il_artifact(
                base_dir,
                il_ckpt_path,
                net_stem=net_stem,
                scene_id=scene_id,
            )
            if il_warm_start_source == "net":
                il_warm_scale = warm_scale_net
            elif il_warm_start_source == "scene":
                il_warm_scale = warm_scale_scene
            elif il_warm_start_source == "global":
                il_warm_scale = warm_scale_global
            search.train_iterations = max(1, int(round(search.train_iterations * il_warm_scale)))
            search.extra_train_iterations = max(0, int(round(search.extra_train_iterations * il_warm_scale)))

        scene_load_line = (
            "ppo_scene_loaded="
            + ("1" if ppo_scene_loaded else "0")
            + " scene_finetune_applied="
            + ("1" if ppo_scene_finetune_applied else "0")
        )
        print(scene_load_line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(scene_load_line + "\n")
        il_line = (
            "il_mode="
            + il_mode
            + " il_method="
            + il_warm_method
            + " il_warm_started="
            + ("1" if il_warm_started else "0")
            + " il_warm_start_source="
            + il_warm_start_source
            + " planned_source="
            + planned_il_warm_start_source
            + " il_warm_scale="
            + str(il_warm_scale)
        )
        print(il_line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(il_line + "\n")
        finetune_schedule_line = (
            "finetune_schedule train_iterations="
            + str(search.train_iterations)
            + " extra_train_iterations="
            + str(search.extra_train_iterations)
            + " base_train_iterations="
            + str(base_train_iterations)
            + " base_extra_train_iterations="
            + str(base_extra_train_iterations)
            + " inference_max_steps="
            + str(search.inference_max_steps_per_episode)
        )
        print(finetune_schedule_line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(finetune_schedule_line + "\n")
        if ppo_scene_loaded and (not ppo_scene_finetune_applied):
            train_plan_line = "train_plan=run_infer_from_scene_checkpoint"
        elif ppo_scene_loaded and ppo_scene_finetune_applied:
            train_plan_line = "train_plan=run_train_from_scene_checkpoint"
        elif il_warm_started:
            train_plan_line = "train_plan=run_train_from_il_warm_start"
        else:
            train_plan_line = "train_plan=run_train_from_scratch"
        print(train_plan_line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(train_plan_line + "\n")

        warm_start_infer_elapsed = 0.0
        warm_start_infer_summary = {
            "trans_count": 0,
            "trans_sequence": "",
            "makespan": -1,
            "reach_goal": False,
            "goal_distance": -1,
        }
        if (not ppo_scene_loaded) and il_warm_started:
            # 这个 warm start inference 只用于观察 IL 热启动的初始效果。
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

        # 单网入口仍会保存本次运行对应的结果 checkpoint，便于后续相似网复用。
        signature = build_signature(path, context)
        profile = build_profile(context)
        checkpoint_prefix = (
            "gcn_ppo_hq_" + controller_representation_tag
            if ppo_variant == "enhanced"
            else "gcn_ppo_" + ppo_variant + "_hq_" + controller_representation_tag
        )
        ckpt_path = checkpoint_path(base_dir, checkpoint_prefix, signature)
        to_save = {
            "signature": signature,
            "profile": profile,
            "actor_state": search.model.actor_net.state_dict(),
            "critic_state": search.model.value_head.state_dict(),
            "optimizer_state": search.optimizer.state_dict(),
            "best_train_makespan": search.best_train_makespan,
            "best_train_trans": search.best_train_trans,
            "train_info": {
                "trainSteps": extra.get("trainSteps", 0),
                "avgLoss": extra.get("avgLoss", 0.0),
            },
        }
        torch.save(to_save, ckpt_path)

        trans = result.get_trans()
        markings = result.get_markings()
        out = "elapsed:" + format(elapsed, ".6f") + "s\n"
        out += format_inline_env_overrides(applied_inline_env) + "\n"
        out += "ppo_variant:" + ppo_variant + "\n"
        out += "trans_count:" + str(len(trans)) + "\n"
        out += "trans_sequence:" + ("->".join(str(t) for t in trans) if trans else "") + "\n"
        out += "makespan:" + str(markings[-1].get_prefix() if markings else -1) + "\n"
        out += "reach_goal:" + str(extra.get("reachGoal")) + "\n"
        out += "goal_distance:" + str(extra.get("goalDistance")) + "\n"
        out += "train_steps:" + str(extra.get("trainSteps", 0)) + "\n"
        out += "best_train_makespan:" + str(extra.get("bestTrainMakespan", -1)) + "\n"
        out += "best_train_trans_count:" + str(extra.get("bestTrainTransCount", 0)) + "\n"
        out += "avg_loss:" + str(extra.get("avgLoss", 0.0)) + "\n"
        out += "policy_loss:" + str(extra.get("policyLoss", 0.0)) + "\n"
        out += "value_loss:" + str(extra.get("valueLoss", 0.0)) + "\n"
        out += "entropy:" + str(extra.get("entropy", 0.0)) + "\n"
        out += "controller_representation_enabled:" + ("1" if controller_representation_enabled else "0") + "\n"
        out += "controller_representation_tag:" + controller_representation_tag + "\n"
        out += "train_success_rate:" + str(extra.get("trainSuccessRate", 0.0)) + "\n"
        out += "rollout_success_rate:" + str(extra.get("rolloutSuccessRate", 0.0)) + "\n"
        out += "replay_success_rate:" + str(extra.get("replaySuccessRate", 0.0)) + "\n"
        out += "extra_train_iterations:" + str(extra.get("extraTrainIterations", 0)) + "\n"
        out += "rollout_episodes_per_iter:" + str(extra.get("rolloutEpisodesPerIter", 0)) + "\n"
        out += "ppo_update_epochs:" + str(extra.get("ppoUpdateEpochs", 0)) + "\n"
        out += "inference_max_steps:" + str(search.inference_max_steps_per_episode) + "\n"
        out += "scene_id:" + (scene_id if scene_id else "") + "\n"
        out += "ppo_scene_loaded:" + str(ppo_scene_loaded) + "\n"
        out += "ppo_scene_finetune_applied:" + str(ppo_scene_finetune_applied) + "\n"
        out += "ppo_scene_checkpoint_path:" + (scene_ckpt_path if scene_ckpt_path else "") + "\n"
        out += "scene_finetune_scale:" + str(scene_finetune_scale) + "\n"
        out += "step_reference_source:" + step_reference_source + "\n"
        out += "heuristic_min_steps:" + str(heuristic_min_steps) + "\n"
        out += "heuristic_max_steps:" + str(heuristic_max_steps) + "\n"
        out += "il_mode:" + il_mode + "\n"
        out += "il_method:" + il_warm_method + "\n"
        out += "il_warm_started:" + str(il_warm_started) + "\n"
        out += "il_warm_start_source:" + il_warm_start_source + "\n"
        out += "il_warm_scale:" + str(il_warm_scale) + "\n"
        out += "base_train_iterations:" + str(base_train_iterations) + "\n"
        out += "base_extra_train_iterations:" + str(base_extra_train_iterations) + "\n"
        out += "il_checkpoint_path:" + (il_ckpt_path if il_ckpt_path else "") + "\n"
        out += "il_result_path:" + (il_result_path if il_result_path else "") + "\n"
        out += "il_expert_steps:" + str(expert_steps) + "\n"
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
        err = "ERROR\n" + traceback.format_exc()
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(err)
        print(err, flush=True)


if __name__ == "__main__":
    main()
