import os
import random
import re
import sys
import time
import traceback

import torch  # pyright: ignore[reportMissingImports]

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from python_port.entrypoint_env import apply_inline_env_overrides, format_inline_env_overrides
from python_port.imitation.il_checkpoint import classify_il_artifact, normalize_il_mode, resolve_il_checkpoint, resolve_il_result
from python_port.petri_net_io.utils.checkpoint_selector import load_compatible_state
from python_port.petri_net_io.utils.net_loader import build_ttpn_with_residence, load_petri_net_context
from python_port.petri_net_platform.search.petri_net_gcn_ppo import PetriNetGCNPPOEnhancedHQ
from python_port.petri_net_platform.search.petri_net_gcn_ppo_classic import PetriNetGCNPPOClassicHQ
from python_port.scene_utils import list_scene_net_files


# ===== USER-TUNABLE DEFAULTS: EDIT HERE FIRST =====
# 场景级 PPO 训练入口：
# 1. 只在训练开始时加载一次场景 IL 模型（优先 DAgger，其次 BC）
# 2. 在同一 scene 的多个网文件之间多轮循环训练
# 3. 每轮随机打乱网顺序，但使用固定随机种子保证可复现
# 4. 每轮结束后按 scene 级成功率与成功样本 makespan 评估并保存最优 checkpoint
# 要训练的场景编号；留空时将遍历 resources 下全部 scene。
DEFAULT_GCN_PPO_SCENE_ID = "1"
# PPO 训练器变体：enhanced 为当前工程版，classic 为迁移进来的旧版风格 PPO 壳。
DEFAULT_GCN_PPO_SCENE_VARIANT = "classic"
# 是否打印场景级 PPO 训练日志，1 为打印，0 为静默。
DEFAULT_GCN_PPO_SCENE_VERBOSE = "1"
# 单网 PPO 训练日志间隔，数值越小日志越密。
DEFAULT_GCN_PPO_SCENE_LOG_INTERVAL = "2"
# 整个 scene 要循环训练多少轮。
DEFAULT_GCN_PPO_SCENE_ROUNDS = "3"
# 每轮中每个网的主训练 iteration 数。
DEFAULT_GCN_PPO_SCENE_TRAIN_ITERATIONS = "8"
# 每轮中每个网的额外训练 iteration 上限。
DEFAULT_GCN_PPO_SCENE_EXTRA_TRAIN_ITERATIONS = "4"
# 每个 PPO iteration 收集多少条 rollout。
DEFAULT_GCN_PPO_SCENE_ROLLOUT_EPISODES_PER_ITER = "12"
# 每个 PPO iteration 的 update epoch 数。
DEFAULT_GCN_PPO_SCENE_UPDATE_EPOCHS = "6"
# 是否启用第三层控制器感知图特征，1 开启，0 关闭。
DEFAULT_GCN_PPO_SCENE_CONTROLLER_REPRESENTATION = "1"
# classic 变体专用：每轮按步数收集 rollout，再做 mini-batch PPO 更新。
DEFAULT_GCN_PPO_SCENE_STEPS_PER_EPOCH = "1024"
DEFAULT_GCN_PPO_SCENE_MINIBATCH_SIZE = "128"
DEFAULT_GCN_PPO_SCENE_TARGET_KL = "0.05"
DEFAULT_GCN_PPO_SCENE_ENTROPY_START = "0.03"
DEFAULT_GCN_PPO_SCENE_ENTROPY_END = "0.01"
DEFAULT_GCN_PPO_SCENE_TEMPERATURE_START = "1.3"
DEFAULT_GCN_PPO_SCENE_TEMPERATURE_END = "1.0"
# 场景内随机打乱网顺序时使用的基础随机种子。
DEFAULT_GCN_PPO_SCENE_SEED = "42"
# ===== END USER-TUNABLE DEFAULTS =====

# 脚本内环境变量覆盖：
# 直接在这里填写键值对，可在代码里调整 PPO scene 训练参数。
INLINE_ENV_OVERRIDE_PRIORITY = "code"
INLINE_ENV_OVERRIDES = {
    # "GCN_PPO_SCENE_IL_MODE": "dagger",
    # "GCN_PPO_SCENE_ID": "1",
    # "GCN_PPO_SCENE_ROUNDS": "3",
}


def _append_line(path, line):
    # 统一将进度写入文件并同步打印，便于长训练过程追踪。
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _read_result_kv(path):
    # 读取 IL 结果文件，从中提取专家步数等参考信息。
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


def _infer_expert_steps(result_info):
    # 优先从结构化字段中取专家步数，拿不到时再从轨迹串中回退解析。
    if not result_info:
        return 0
    for key in ["expert_trans_count", "clean_steps", "policy_trans_count", "scene_ref_expert_steps"]:
        value = _safe_int(result_info.get(key, 0), 0)
        if value > 0:
            return value
    seq = result_info.get("expert_trans_sequence", "") or result_info.get("policy_trans_sequence", "")
    nums = re.findall(r"\d+", seq)
    return len(nums)


def _state_dict_cpu(module):
    # 在跨网迁移时，把当前 PPO 参数转成 CPU 版本，便于写 checkpoint 与传递到下一次训练。
    out = {}
    for key, value in module.state_dict().items():
        out[key] = value.detach().cpu()
    return out


def _compute_step_schedule(context, expert_steps):
    # 先按网规模给出启发式预算，再结合 IL 专家轨迹长度做覆盖。
    place_count = len(context["p_info"])
    pre = context["pre"]
    trans_count = len(pre[0]) if pre else 0
    constrained_count = 0
    for val in context["max_residence_time"]:
        if val < 2 ** 31 - 1:
            constrained_count += 1
    complexity = max(place_count, trans_count)
    heuristic_min_steps = min(220, max(120, 90 + complexity))
    heuristic_max_steps = min(900, max(heuristic_min_steps + 260, 480 + complexity * 8 + constrained_count * 6))
    min_steps = heuristic_min_steps
    max_steps = heuristic_max_steps
    step_reference_source = "heuristic"

    # 专家步数驱动的训练/推理步长预算：
    # GCN_PPO_SCENE_EXPERT_MIN_STEP_SCALE：专家步数映射到训练最小步数的缩放
    # GCN_PPO_SCENE_EXPERT_MAX_STEP_SCALE：专家步数映射到训练最大步数的缩放
    # GCN_PPO_SCENE_EXPERT_MIN_STEP_FLOOR：训练最小步数下界
    # GCN_PPO_SCENE_EXPERT_MAX_STEP_FLOOR：训练最大步数下界
    # GCN_PPO_SCENE_EXPERT_MAX_STEP_MIN_MARGIN：训练 max_steps 至少比 min_steps 多出的步数
    # GCN_PPO_SCENE_EXPERT_STEP_SCALE：专家步数映射到推理步数上限的缩放
    # GCN_PPO_SCENE_EXPERT_STEP_MIN_MARGIN：推理步数至少比专家步数多出的步数
    expert_train_min_scale = float(os.environ.get("GCN_PPO_SCENE_EXPERT_MIN_STEP_SCALE", "0.75"))
    expert_train_max_scale = float(os.environ.get("GCN_PPO_SCENE_EXPERT_MAX_STEP_SCALE", "1.80"))
    expert_train_min_floor = int(os.environ.get("GCN_PPO_SCENE_EXPERT_MIN_STEP_FLOOR", "24"))
    expert_train_max_floor = int(os.environ.get("GCN_PPO_SCENE_EXPERT_MAX_STEP_FLOOR", "48"))
    expert_train_max_margin = int(os.environ.get("GCN_PPO_SCENE_EXPERT_MAX_STEP_MIN_MARGIN", "24"))
    expert_step_scale = float(os.environ.get("GCN_PPO_SCENE_EXPERT_STEP_SCALE", "2.0"))
    expert_step_min_margin = int(os.environ.get("GCN_PPO_SCENE_EXPERT_STEP_MIN_MARGIN", "16"))

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

    return {
        "min_steps": min_steps,
        "max_steps": max_steps,
        "inference_max_steps": inference_max_steps,
        "step_reference_source": step_reference_source,
        "heuristic_min_steps": heuristic_min_steps,
        "heuristic_max_steps": heuristic_max_steps,
        "place_count": place_count,
        "trans_count": trans_count,
        "constrained_count": constrained_count,
    }


def _summarize_result(net_name, result, extra_info, elapsed, schedule):
    # 将单网 PPO 训练 + 推理结果收敛成场景级结果文件中的一条摘要。
    trans = result.get_trans() if result is not None else []
    markings = result.get_markings() if result is not None else []
    return {
        "net_name": net_name,
        "elapsed": float(elapsed),
        "trans_count": len(trans),
        "trans_sequence": "->".join(str(t) for t in trans) if trans else "",
        "makespan": markings[-1].get_prefix() if markings else -1,
        "reach_goal": bool(extra_info.get("reachGoal")),
        "train_steps": int(extra_info.get("trainSteps", 0)),
        "best_train_makespan": int(extra_info.get("bestTrainMakespan", -1)),
        "train_success_rate": float(extra_info.get("trainSuccessRate", 0.0)),
        "rollout_success_rate": float(extra_info.get("rolloutSuccessRate", 0.0)),
        "replay_success_rate": float(extra_info.get("replaySuccessRate", 0.0)),
        "inference_stop_reason": str(extra_info.get("inferenceStopReason", "")),
        "inference_deadlock_reason": str(extra_info.get("inferenceDeadlockReason", "")),
        "step_reference_source": schedule["step_reference_source"],
        "min_steps": schedule["min_steps"],
        "max_steps": schedule["max_steps"],
        "inference_max_steps": schedule["inference_max_steps"],
    }


def _build_search(
    context,
    petri_net,
    schedule,
    train_iterations,
    extra_train_iterations,
    rollout_episodes_per_iter,
    ppo_update_epochs,
    reward_time_scale,
    reward_clip_abs,
    verbose,
    log_interval,
    controller_representation_enabled,
    ppo_variant,
    classic_config,
):
    common_kwargs = dict(
        petri_net=petri_net,
        end=context["end"],
        pre=context["pre"],
        post=context["post"],
        min_delay_p=context["min_delay_p"],
        train_iterations=train_iterations,
        rollout_episodes_per_iter=rollout_episodes_per_iter,
        ppo_update_epochs=ppo_update_epochs,
        min_steps_per_episode=schedule["min_steps"],
        max_steps_per_episode=schedule["max_steps"],
        inference_max_steps_per_episode=schedule["inference_max_steps"],
        goal_eval_rollouts=1,
        goal_min_success_rate=0.7,
        extra_train_iterations=extra_train_iterations,
        use_reward_scaling=True,
        reward_time_scale=reward_time_scale,
        use_reward_clip=True,
        reward_clip_abs=reward_clip_abs,
        verbose=verbose,
        log_interval=log_interval,
        controller_representation_enabled=controller_representation_enabled,
    )
    if ppo_variant == "classic":
        return PetriNetGCNPPOClassicHQ(
            **common_kwargs,
            steps_per_epoch=classic_config["steps_per_epoch"],
            minibatch_size=classic_config["minibatch_size"],
            target_kl=classic_config["target_kl"],
            entropy_coef_start=classic_config["entropy_coef_start"],
            entropy_coef_end=classic_config["entropy_coef_end"],
            temperature_start=classic_config["temperature_start"],
            temperature_end=classic_config["temperature_end"],
        )
    return PetriNetGCNPPOEnhancedHQ(**common_kwargs)


def _compute_scene_metrics(eval_summaries):
    total = len(eval_summaries)
    success_items = [item for item in eval_summaries if item.get("reach_goal")]
    success_count = len(success_items)
    success_rate = (float(success_count) / float(total)) if total > 0 else 0.0
    if success_count > 0:
        avg_success_makespan = sum(float(item.get("makespan", 0.0)) for item in success_items) / float(success_count)
        avg_success_trans_count = sum(int(item.get("trans_count", 0)) for item in success_items) / float(success_count)
    else:
        avg_success_makespan = float("inf")
        avg_success_trans_count = float("inf")
    return {
        "success_count": success_count,
        "total_count": total,
        "success_rate": success_rate,
        "avg_success_makespan": avg_success_makespan,
        "avg_success_trans_count": avg_success_trans_count,
    }


def _is_better_scene_metrics(curr, best):
    if best is None:
        return True
    if float(curr["success_rate"]) != float(best["success_rate"]):
        return float(curr["success_rate"]) > float(best["success_rate"])
    if int(curr["success_count"]) > 0 and int(best["success_count"]) > 0:
        if float(curr["avg_success_makespan"]) != float(best["avg_success_makespan"]):
            return float(curr["avg_success_makespan"]) < float(best["avg_success_makespan"])
        if float(curr["avg_success_trans_count"]) != float(best["avg_success_trans_count"]):
            return float(curr["avg_success_trans_count"]) < float(best["avg_success_trans_count"])
    return False


def _evaluate_scene_policy(
    base_dir,
    net_files,
    scene_id,
    il_mode,
    actor_state,
    critic_state,
    reward_time_scale,
    reward_clip_abs,
    controller_representation_enabled,
    ppo_variant,
    classic_config,
):
    # 每轮结束后，用当前参数对整个场景做一次统一评估，避免只看“最后训练到的那个网”。
    eval_summaries = []
    for path in net_files:
        net_name = os.path.splitext(os.path.basename(path))[0]
        context = load_petri_net_context(path)
        petri_net = build_ttpn_with_residence(context)
        il_result_info = _read_result_kv(resolve_il_result(base_dir, il_mode, net_stem=net_name, scene_id=scene_id))
        expert_steps = _infer_expert_steps(il_result_info)
        schedule = _compute_step_schedule(context, expert_steps)
        search = _build_search(
            context=context,
            petri_net=petri_net,
            schedule=schedule,
            train_iterations=1,
            extra_train_iterations=0,
            rollout_episodes_per_iter=1,
            ppo_update_epochs=1,
            reward_time_scale=reward_time_scale,
            reward_clip_abs=reward_clip_abs,
            verbose=False,
            log_interval=1,
            controller_representation_enabled=controller_representation_enabled,
            ppo_variant=ppo_variant,
            classic_config=classic_config,
        )
        load_compatible_state(search.model.actor_net, actor_state)
        load_compatible_state(search.model.value_head, critic_state)
        search.is_trained = True
        start = time.perf_counter()
        result = search.search()
        elapsed = time.perf_counter() - start
        summary = _summarize_result(net_name, result, dict(search.get_extra_info()), elapsed, schedule)
        summary["eval_only"] = True
        eval_summaries.append(summary)
    return eval_summaries, _compute_scene_metrics(eval_summaries)


def main():
    base_dir = os.path.dirname(__file__)
    resources_dir = os.path.join(base_dir, "resources")
    results_dir = os.path.join(base_dir, "results")
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    applied_inline_env = apply_inline_env_overrides(INLINE_ENV_OVERRIDES, priority=INLINE_ENV_OVERRIDE_PRIORITY)
    scene_id = os.environ.get("GCN_PPO_SCENE_ID", DEFAULT_GCN_PPO_SCENE_ID).strip()
    ppo_variant = _normalize_ppo_variant(
        os.environ.get("GCN_PPO_SCENE_VARIANT", DEFAULT_GCN_PPO_SCENE_VARIANT)
    )
    scene_tag = scene_id if scene_id else "all"
    scene_prefix = "ppo_scene_" if ppo_variant == "enhanced" else "ppo_" + ppo_variant + "_scene_"
    result_path = os.path.join(results_dir, scene_prefix + scene_tag + "_result.txt")
    progress_path = os.path.join(results_dir, scene_prefix + scene_tag + "_progress.txt")
    checkpoint_path = os.path.join(checkpoints_dir, scene_prefix + scene_tag + ".pt")

    with open(progress_path, "w", encoding="utf-8") as f:
        f.write("")

    try:
        # 场景训练主预算参数：
        # GCN_PPO_SCENE_ROUNDS：scene 循环轮数
        # GCN_PPO_SCENE_TRAIN_ITERATIONS：每网每轮主训练 iteration 数
        # GCN_PPO_SCENE_EXTRA_TRAIN_ITERATIONS：每网每轮额外训练 iteration 上限
        scene_rounds = int(os.environ.get("GCN_PPO_SCENE_ROUNDS", DEFAULT_GCN_PPO_SCENE_ROUNDS))
        train_iterations = int(os.environ.get("GCN_PPO_SCENE_TRAIN_ITERATIONS", DEFAULT_GCN_PPO_SCENE_TRAIN_ITERATIONS))
        extra_train_iterations = int(
            os.environ.get("GCN_PPO_SCENE_EXTRA_TRAIN_ITERATIONS", DEFAULT_GCN_PPO_SCENE_EXTRA_TRAIN_ITERATIONS)
        )
        # PPO 超参数：
        # GCN_PPO_SCENE_ROLLOUT_EPISODES_PER_ITER：每个 iteration 的 rollout 数
        # GCN_PPO_SCENE_UPDATE_EPOCHS：每个 iteration 的 PPO update epoch 数
        rollout_episodes_per_iter = int(
            os.environ.get(
                "GCN_PPO_SCENE_ROLLOUT_EPISODES_PER_ITER",
                DEFAULT_GCN_PPO_SCENE_ROLLOUT_EPISODES_PER_ITER,
            )
        )
        ppo_update_epochs = int(os.environ.get("GCN_PPO_SCENE_UPDATE_EPOCHS", DEFAULT_GCN_PPO_SCENE_UPDATE_EPOCHS))
        # 奖励缩放参数：
        # GCN_PPO_SCENE_REWARD_TIME_SCALE：时间惩罚缩放
        # GCN_PPO_SCENE_REWARD_CLIP_ABS：奖励裁剪上限
        reward_time_scale = float(os.environ.get("GCN_PPO_SCENE_REWARD_TIME_SCALE", "1000.0"))
        reward_clip_abs = float(os.environ.get("GCN_PPO_SCENE_REWARD_CLIP_ABS", "20.0"))
        # 表示层与日志参数：
        # GCN_PPO_SCENE_CONTROLLER_REPRESENTATION：是否启用第三层控制器感知图特征
        # GCN_PPO_SCENE_VERBOSE：是否打印细日志
        # GCN_PPO_SCENE_LOG_INTERVAL：日志间隔
        controller_representation_enabled = os.environ.get(
            "GCN_PPO_SCENE_CONTROLLER_REPRESENTATION",
            DEFAULT_GCN_PPO_SCENE_CONTROLLER_REPRESENTATION,
        ) == "1"
        verbose = os.environ.get("GCN_PPO_SCENE_VERBOSE", DEFAULT_GCN_PPO_SCENE_VERBOSE) == "1"
        log_interval = int(os.environ.get("GCN_PPO_SCENE_LOG_INTERVAL", DEFAULT_GCN_PPO_SCENE_LOG_INTERVAL))
        classic_config = {
            "steps_per_epoch": int(
                os.environ.get("GCN_PPO_SCENE_STEPS_PER_EPOCH", DEFAULT_GCN_PPO_SCENE_STEPS_PER_EPOCH)
            ),
            "minibatch_size": int(
                os.environ.get("GCN_PPO_SCENE_MINIBATCH_SIZE", DEFAULT_GCN_PPO_SCENE_MINIBATCH_SIZE)
            ),
            "target_kl": float(os.environ.get("GCN_PPO_SCENE_TARGET_KL", DEFAULT_GCN_PPO_SCENE_TARGET_KL)),
            "entropy_coef_start": float(
                os.environ.get("GCN_PPO_SCENE_ENTROPY_START", DEFAULT_GCN_PPO_SCENE_ENTROPY_START)
            ),
            "entropy_coef_end": float(
                os.environ.get("GCN_PPO_SCENE_ENTROPY_END", DEFAULT_GCN_PPO_SCENE_ENTROPY_END)
            ),
            "temperature_start": float(
                os.environ.get("GCN_PPO_SCENE_TEMPERATURE_START", DEFAULT_GCN_PPO_SCENE_TEMPERATURE_START)
            ),
            "temperature_end": float(
                os.environ.get("GCN_PPO_SCENE_TEMPERATURE_END", DEFAULT_GCN_PPO_SCENE_TEMPERATURE_END)
            ),
        }
        # 数据范围与随机性参数：
        # GCN_PPO_SCENE_USE_BC_WARM_START：旧兼容开关；为 0 时禁用所有 IL 热启动
        # GCN_PPO_SCENE_IL_MODE：auto / bc / dagger，auto 时优先尝试 DAgger，其次回退到 BC
        # GCN_PPO_SCENE_IL_CKPT_PATH：显式指定场景 IL checkpoint 路径
        # GCN_PPO_SCENE_NET_LIMIT：只取 scene 内前 N 个网，便于小规模调试
        # GCN_PPO_SCENE_SEED：控制每轮打乱顺序的固定随机种子
        use_bc_warm_start = os.environ.get("GCN_PPO_SCENE_USE_BC_WARM_START", "1") == "1"
        il_mode = normalize_il_mode(os.environ.get("GCN_PPO_SCENE_IL_MODE", "auto"))
        net_limit = int(os.environ.get("GCN_PPO_SCENE_NET_LIMIT", "0"))
        seed = int(os.environ.get("GCN_PPO_SCENE_SEED", DEFAULT_GCN_PPO_SCENE_SEED))

        net_files = list_scene_net_files(resources_dir, scene_id)
        if net_limit > 0:
            net_files = net_files[:net_limit]
        if not net_files:
            raise ValueError("未找到可用于场景级 PPO 训练的网文件，scene_id=" + (scene_id if scene_id else "all"))

        net_names = [os.path.splitext(os.path.basename(path))[0] for path in net_files]
        _append_line(progress_path, "ppo_scene_train scene_id=" + scene_tag)
        _append_line(progress_path, format_inline_env_overrides(applied_inline_env))
        _append_line(progress_path, "ppo_variant=" + ppo_variant)
        _append_line(progress_path, "nets=" + str(net_names))
        _append_line(
            progress_path,
            "schedule scene_rounds="
            + str(scene_rounds)
            + " train_iterations_per_net="
            + str(train_iterations)
            + " extra_train_iterations_per_net="
            + str(extra_train_iterations)
            + " rollout_episodes_per_iter="
            + str(rollout_episodes_per_iter)
            + " ppo_update_epochs="
            + str(ppo_update_epochs)
            + " controller_representation="
            + ("1" if controller_representation_enabled else "0"),
        )
        if ppo_variant == "classic":
            _append_line(
                progress_path,
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
                + str(classic_config["temperature_end"]),
            )
        _append_line(progress_path, "shuffle seed=" + str(seed) + " policy=random_each_round")

        il_ckpt_path = (
            resolve_il_checkpoint(
                base_dir,
                il_mode,
                net_stem="",
                scene_id=scene_id,
                explicit=os.environ.get("GCN_PPO_SCENE_IL_CKPT_PATH", "").strip(),
            )
            if use_bc_warm_start
            else ""
        )
        il_warm_started_once = False
        il_warm_method = "none"
        il_result_mode = il_mode
        init_source = "scratch"
        current_actor_state = {}
        current_critic_state = {}
        # 这里只在场景训练开始时热启动一次 IL；后续各轮各网之间只传递 PPO 自己学到的参数。
        if il_ckpt_path and os.path.exists(il_ckpt_path):
            saved_il = torch.load(il_ckpt_path, map_location="cpu")
            if isinstance(saved_il, dict):
                current_actor_state = (
                    saved_il.get("model_state")
                    or saved_il.get("policy_state")
                    or saved_il.get("actor_state")
                    or {}
                )
            il_warm_started_once = True
            il_warm_method, il_source = classify_il_artifact(base_dir, il_ckpt_path, net_stem="", scene_id=scene_id)
            init_source = il_warm_method + "_" + il_source if il_warm_method != "none" else "scratch"
        _append_line(
            progress_path,
            "init source="
            + init_source
            + " il_mode="
            + il_mode
            + " il_warm_started_once="
            + ("1" if il_warm_started_once else "0")
            + " il_checkpoint_path="
            + (il_ckpt_path if il_ckpt_path else ""),
        )

        visit_summaries = []
        round_metrics = []
        best_scene_metrics = None
        best_round = 0
        best_eval_summaries = []
        best_round_order = []
        best_checkpoint_updates = 0

        for round_idx in range(scene_rounds):
            round_rng = random.Random(seed + round_idx)
            round_net_files = list(net_files)
            round_rng.shuffle(round_net_files)
            round_net_names = [os.path.splitext(os.path.basename(path))[0] for path in round_net_files]
            _append_line(
                progress_path,
                "scene_round_begin round="
                + str(round_idx + 1)
                + "/"
                + str(scene_rounds)
                + " order="
                + str(round_net_names),
            )

            for index, path in enumerate(round_net_files):
                net_name = os.path.splitext(os.path.basename(path))[0]
                context = load_petri_net_context(path)
                petri_net = build_ttpn_with_residence(context)
                # 读取IL专家轨迹信息,用于计算训练步数上限.计算推理步数上限
                il_result_path = resolve_il_result(base_dir, il_result_mode, net_stem=net_name, scene_id=scene_id)
                il_result_info = _read_result_kv(il_result_path)
                expert_steps = _infer_expert_steps(il_result_info)
                schedule = _compute_step_schedule(context, expert_steps)

                _append_line(
                    progress_path,
                    "net_begin round="
                    + str(round_idx + 1)
                    + " index="
                    + str(index + 1)
                    + "/"
                    + str(len(round_net_files))
                    + " net="
                    + net_name
                    + " min_steps="
                    + str(schedule["min_steps"])
                    + " max_steps="
                    + str(schedule["max_steps"])
                    + " inference_max_steps="
                    + str(schedule["inference_max_steps"])
                    + " step_source="
                    + schedule["step_reference_source"]
                    + " places="
                    + str(schedule["place_count"])
                    + " trans="
                    + str(schedule["trans_count"])
                    + " constrained_places="
                    + str(schedule["constrained_count"]),
                )

                search = _build_search(
                    context=context,
                    petri_net=petri_net,
                    schedule=schedule,
                    train_iterations=train_iterations,
                    extra_train_iterations=extra_train_iterations,
                    rollout_episodes_per_iter=rollout_episodes_per_iter,
                    ppo_update_epochs=ppo_update_epochs,
                    reward_time_scale=reward_time_scale,
                    reward_clip_abs=reward_clip_abs,
                    verbose=verbose,
                    log_interval=log_interval,
                    controller_representation_enabled=controller_representation_enabled,
                    ppo_variant=ppo_variant,
                    classic_config=classic_config,
                )
                # 用IL参数初始化PPO模型
                if current_actor_state:
                    load_compatible_state(search.model.actor_net, current_actor_state)
                if current_critic_state:
                    load_compatible_state(search.model.value_head, current_critic_state)

                # 在当前网上完成 PPO 训练与推理。
                start = time.perf_counter()
                result = search.search()
                elapsed = time.perf_counter() - start
                extra = dict(search.get_extra_info())
                summary = _summarize_result(net_name, result, extra, elapsed, schedule)
                summary["scene_round"] = round_idx + 1
                summary["visit_index"] = index + 1
                summary["train_failure_counts"] = dict(extra.get("trainFailureCounts", {}))
                summary["eval_failure_counts"] = dict(extra.get("evalFailureCounts", {}))
                visit_summaries.append(summary)

                # 一个网训练完成后，将 PPO 的 actor / critic 参数迁移到下一次训练继续使用。
                current_actor_state = _state_dict_cpu(search.model.actor_net)
                current_critic_state = _state_dict_cpu(search.model.value_head)

                _append_line(
                    progress_path,
                    "net_done round="
                    + str(round_idx + 1)
                    + " net="
                    + net_name
                    + " reach_goal="
                    + str(summary["reach_goal"])
                    + " makespan="
                    + str(summary["makespan"])
                    + " trans_count="
                    + str(summary["trans_count"])
                    + " best_train_makespan="
                    + str(summary["best_train_makespan"])
                    + " elapsed="
                    + format(summary["elapsed"], ".6f")
                    + "s"
                    + " inference_stop_reason="
                    + summary["inference_stop_reason"]
                    + " inference_deadlock_reason="
                    + summary["inference_deadlock_reason"],
                )

            eval_summaries, scene_metrics = _evaluate_scene_policy(
                base_dir=base_dir,
                net_files=net_files,
                scene_id=scene_id,
                il_mode=il_result_mode,
                actor_state=current_actor_state,
                critic_state=current_critic_state,
                reward_time_scale=reward_time_scale,
                reward_clip_abs=reward_clip_abs,
                controller_representation_enabled=controller_representation_enabled,
                ppo_variant=ppo_variant,
                classic_config=classic_config,
            )
            scene_metrics["round"] = round_idx + 1
            scene_metrics["order"] = round_net_names
            round_metrics.append(scene_metrics)
            _append_line(
                progress_path,
                "scene_round_eval round="
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
                + ("inf" if scene_metrics["success_count"] == 0 else format(scene_metrics["avg_success_trans_count"], ".4f"))
            )

            if _is_better_scene_metrics(scene_metrics, best_scene_metrics):
                best_scene_metrics = dict(scene_metrics)
                best_round = round_idx + 1
                best_eval_summaries = list(eval_summaries)
                best_round_order = list(round_net_names)
                best_checkpoint_updates += 1
                torch.save(
                    {
                        "scene_id": scene_tag,
                        "net_names": net_names,
                        "scene_rounds": scene_rounds,
                        "best_round": best_round,
                        "best_scene_metrics": best_scene_metrics,
                        "best_round_order": best_round_order,
                        "actor_state": current_actor_state,
                        "critic_state": current_critic_state,
                        "controller_representation_enabled": controller_representation_enabled,
                        "ppo_variant": ppo_variant,
                        "il_mode": il_mode,
                        "il_warm_started_once": il_warm_started_once,
                        "il_checkpoint_path": il_ckpt_path,
                        "round_metrics": round_metrics,
                        "best_eval_summaries": best_eval_summaries,
                        "visit_summaries": visit_summaries,
                    },
                    checkpoint_path,
                )
                _append_line(
                    progress_path,
                    "scene_checkpoint_update round="
                    + str(best_round)
                    + " success_rate="
                    + format(best_scene_metrics["success_rate"], ".4f"),
                )

        lines = [
            "status:ok",
            format_inline_env_overrides(applied_inline_env),
            "scene_id:" + scene_tag,
            "ppo_variant:" + ppo_variant,
            "scene_checkpoint_path:" + checkpoint_path,
            "il_mode:" + il_mode,
            "il_warm_started_once:" + ("1" if il_warm_started_once else "0"),
            "il_checkpoint_path:" + (il_ckpt_path if il_ckpt_path else ""),
            "nets_total:" + str(len(net_files)),
            "scene_rounds:" + str(scene_rounds),
            "shuffle_seed:" + str(seed),
            "best_round:" + str(best_round),
            "best_checkpoint_updates:" + str(best_checkpoint_updates),
            "train_iterations_per_net:" + str(train_iterations),
            "extra_train_iterations_per_net:" + str(extra_train_iterations),
            "rollout_episodes_per_iter:" + str(rollout_episodes_per_iter),
            "ppo_update_epochs:" + str(ppo_update_epochs),
            "controller_representation_enabled:" + ("1" if controller_representation_enabled else "0"),
        ]
        if best_scene_metrics is not None:
            lines.append("best_scene_success_rate:" + format(best_scene_metrics["success_rate"], ".6f"))
            lines.append("best_scene_success_count:" + str(best_scene_metrics["success_count"]))
            lines.append("best_scene_total_count:" + str(best_scene_metrics["total_count"]))
            lines.append(
                "best_scene_avg_success_makespan:"
                + ("inf" if best_scene_metrics["success_count"] == 0 else format(best_scene_metrics["avg_success_makespan"], ".6f"))
            )
            lines.append(
                "best_scene_avg_success_trans_count:"
                + ("inf" if best_scene_metrics["success_count"] == 0 else format(best_scene_metrics["avg_success_trans_count"], ".6f"))
            )
            lines.append("best_round_order:" + str(best_round_order))
        for metric in round_metrics:
            lines.append("round:" + str(metric["round"]))
            lines.append("round_order:" + str(metric["order"]))
            lines.append("round_success_rate:" + format(metric["success_rate"], ".6f"))
            lines.append("round_success_count:" + str(metric["success_count"]))
            lines.append("round_total_count:" + str(metric["total_count"]))
            lines.append(
                "round_avg_success_makespan:"
                + ("inf" if metric["success_count"] == 0 else format(metric["avg_success_makespan"], ".6f"))
            )
            lines.append(
                "round_avg_success_trans_count:"
                + ("inf" if metric["success_count"] == 0 else format(metric["avg_success_trans_count"], ".6f"))
            )
        for summary in best_eval_summaries:
            lines.append("best_eval_net_name:" + summary["net_name"])
            lines.append("best_eval_reach_goal:" + str(summary["reach_goal"]))
            lines.append("best_eval_makespan:" + str(summary["makespan"]))
            lines.append("best_eval_trans_count:" + str(summary["trans_count"]))
            lines.append("best_eval_inference_stop_reason:" + summary["inference_stop_reason"])
            lines.append("best_eval_inference_deadlock_reason:" + summary["inference_deadlock_reason"])
            lines.append("best_eval_elapsed:" + format(summary["elapsed"], ".6f") + "s")
            lines.append("best_eval_trans_sequence:" + summary["trans_sequence"])
        txt = "\n".join(lines) + "\n"
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(txt)
        # 场景训练的最终摘要同时写入 progress 文件，便于直接查看收尾状态。
        _append_line(progress_path, "scene_train_done checkpoint=" + checkpoint_path)
        print(txt, flush=True)
    except BaseException:
        err = "ERROR\n" + traceback.format_exc()
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(err)
        print(err, flush=True)


if __name__ == "__main__":
    main()
