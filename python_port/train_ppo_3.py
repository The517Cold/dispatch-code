import os
import re
import sys
import time
import traceback
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from petri_net_io.utils.net_loader import load_petri_net_context, build_ttpn_with_residence
from petri_net_io.utils.checkpoint_selector import (
    build_signature,
    build_profile,
    checkpoint_path,
    find_checkpoint,
    load_compatible_state,
)
from imitation.il_checkpoint import normalize_il_mode, resolve_il_checkpoint, resolve_il_result, classify_il_artifact
from petri_gcn_ppo_4_1 import PetriNetGCNPPOPro


def _env_int(name, default):
    """从环境变量读取整数；缺省 / 空 / 解析失败时返回 default。"""
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        print(f"Warning: 环境变量 {name}={value!r} 无法解析为 int,使用默认 {default}", flush=True)
        return default


def _env_float(name, default):
    """从环境变量读取浮点；缺省 / 空 / 解析失败时返回 default。"""
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        print(f"Warning: 环境变量 {name}={value!r} 无法解析为 float,使用默认 {default}", flush=True)
        return default


def _env_bool(name, default: bool) -> bool:
    """
    从环境变量读取布尔；接受 0/1/true/false/yes/no/on/off(不区分大小写)。

    v2 修复：原代码大量使用 `os.environ.get(NAME, "1") == "0"` 这种容易写反
    的表达式，导致 use_il_warmstart / async_collection / fast_mode 等开关
    完全相反于命名语义。统一通过该函数解析，杜绝此类拼写陷阱。
    """
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    text = value.strip().lower()
    if text in ("1", "true", "t", "yes", "y", "on"):
        return True
    if text in ("0", "false", "f", "no", "n", "off", ""):
        return False
    print(
        f"Warning: 环境变量 {name}={value!r} 无法解析为 bool,使用默认 {default}",
        flush=True,
    )
    return bool(default)


def _env_list(name):
    """从环境变量读取逗号分隔列表；忽略空白项。"""
    raw = os.environ.get(name, "").strip()
    return [item.strip() for item in raw.split(",") if item.strip()]


def _derive_eval_files_from_train_files(train_files, suffix="20"):
    """
    从训练文件名推导结构相近但规模不同的独立评估文件。

    示例: 1-2-1.txt -> 1-2-20.txt。若用户显式传入 eval 文件，
    调用方不会使用该推导结果，从而保持原有配置优先级。
    """
    out = []
    seen = set()
    for file_name in train_files:
        stem = os.path.splitext(os.path.basename(file_name))[0]
        match = re.match(r"^(\d+)-(\d+)-\d+$", stem)
        if not match:
            continue
        candidate = f"{match.group(1)}-{match.group(2)}-{suffix}.txt"
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def _resolve_net_path(base_dir, file_name, roots):
    # 从环境变量中获取训练文件的绝对路径
    # 如果是绝对路径，直接返回
    # 如果不是绝对路径，尝试在根目录下查找
    # 如果没有找到，返回第一个根目录下的路径
    if os.path.isabs(file_name):
        return file_name  # 绝对路径，直接返回
    for root in roots:
        candidate = os.path.join(base_dir, root, file_name)
        if os.path.exists(candidate):
            return candidate  # 找到第一个存在的路径，直接返回
    return os.path.join(base_dir, roots[0], file_name)  # 如果没有找到，返回第一个根目录下的路径


def _load_env_pool(base_dir, file_names, roots):
    env_pool = []
    for fname in file_names:
        net_path = _resolve_net_path(base_dir, fname, roots)
        if not os.path.exists(net_path):
            print(f"Warning: 训练文件未找到 {net_path}，跳过。", flush=True)
            continue

        context = load_petri_net_context(net_path)
        petri_net = build_ttpn_with_residence(context)
        complexity_score = max(len(context["pre"]), len(context["pre"][0]))
        constrained_count = sum(1 for val in context["max_residence_time"] if val < 2 ** 31 - 1)
        env_pool.append({
            "petri_net": petri_net,
            "initial_marking": petri_net.get_marking().clone(),
            "end": context["end"],
            "pre": context["pre"],
            "post": context["post"],
            "min_delay_p": context["min_delay_p"],
            "max_residence_time": context["max_residence_time"],
            "name": os.path.basename(fname),
            "path": net_path,
            "context": context,
            "complexity_score": complexity_score + constrained_count * 0.5
        })
    return env_pool


def _parse_suite_summary(summary):
    match = re.search(
        r"success:(\d+)/(\d+),avg_makespan:([-]?\d+(?:\.\d+)?),worst_makespan:([-]?\d+(?:\.\d+)?)",
        summary or "",
    )
    if not match:
        return {}
    success, total = int(match.group(1)), int(match.group(2))
    return {
        "success": success,
        "total": total,
        "success_rate": success / total if total > 0 else -1,
        "avg_makespan": float(match.group(3)),
        "worst_makespan": float(match.group(4)),
    }


def _metric_float(value):
    if value in (None, "N/A", "Fail"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_epoch_diagnostics(metrics, previous_metrics=None, target_kl=None):
    diagnostics = []
    eval_makespan = _metric_float(metrics.get("eval_makespan", metrics.get("eval_show")))
    best_makespan = _metric_float(metrics.get("best_show"))
    if eval_makespan is not None and best_makespan is not None and best_makespan > 0:
        gap_ratio = eval_makespan / best_makespan
        if gap_ratio >= 1.5:
            diagnostics.append(f"eval_vs_best_gap={gap_ratio:.2f}x")

    kl = _metric_float(metrics.get("kl"))
    if target_kl is not None and target_kl > 0 and kl is not None:
        if kl < target_kl * 0.15:
            diagnostics.append(f"low_kl={kl:.6f}<15%target")

    if metrics.get("eval_pool_due") and metrics.get("eval_pool_configured") and not metrics.get("eval_pool_metrics"):
        diagnostics.append("eval_pool_due_but_missing")

    if previous_metrics:
        reward = _metric_float(metrics.get("avg_reward"))
        prev_reward = _metric_float(previous_metrics.get("avg_reward"))
        if reward is not None and prev_reward is not None and abs(reward - prev_reward) >= 25.0:
            diagnostics.append(f"reward_swing={reward - prev_reward:+.2f}")

        train_loss = _metric_float(metrics.get("train_loss"))
        prev_train_loss = _metric_float(previous_metrics.get("train_loss"))
        if train_loss is not None and prev_train_loss is not None and train_loss - prev_train_loss >= 0.03:
            diagnostics.append(f"loss_jump={train_loss - prev_train_loss:+.4f}")

    epoch_elapsed = _metric_float(metrics.get("epoch_elapsed_sec"))
    steps_collected = _metric_float(metrics.get("steps_collected"))
    if epoch_elapsed is not None and epoch_elapsed > 0 and steps_collected is not None:
        diagnostics.append(f"throughput={steps_collected / epoch_elapsed:.1f}steps/s")

    return diagnostics


def _read_result_kv(path):
    if not path or (not os.path.exists(path)):
        return {}
    out = {}
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


def _compute_step_schedule(context, expert_steps):
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

    expert_train_min_scale = float(os.environ.get("GCN_PPO_HQ_EXPERT_MIN_STEP_SCALE", "0.75"))
    expert_train_max_scale = float(os.environ.get("GCN_PPO_HQ_EXPERT_MAX_STEP_SCALE", "1.80"))
    expert_train_min_floor = int(os.environ.get("GCN_PPO_HQ_EXPERT_MIN_STEP_FLOOR", "24"))
    expert_train_max_floor = int(os.environ.get("GCN_PPO_HQ_EXPERT_MAX_STEP_FLOOR", "48"))
    expert_train_max_margin = int(os.environ.get("GCN_PPO_HQ_EXPERT_MAX_STEP_MIN_MARGIN", "24"))
    expert_step_scale = float(os.environ.get("GCN_PPO_HQ_EXPERT_STEP_SCALE", "2.0"))
    expert_step_min_margin = int(os.environ.get("GCN_PPO_HQ_EXPERT_STEP_MIN_MARGIN", "16"))

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


def _state_dict_cpu(module):
    out = {}
    for key, value in module.state_dict().items():
        out[key] = value.detach().cpu()
    return out


def _run_inference_suite(search, env_pool, suite_name):
    """
    运行推理套件并产出每环境明细 + 整体摘要。

    v2 改进：单环境失败时记录错误并继续，避免整套评估因单点异常中断。
    """
    if not env_pool:
        return [], f"{suite_name}_summary=success:0/0,avg_makespan:-1,worst_makespan:-1"

    saved_env_name = search.current_env_name
    details = []
    makespans = []
    success_count = 0

    for env in env_pool:
        env_name = env.get("name", "unknown")
        try:
            search.switch_environment(env)
            start = time.perf_counter()
            result = search.search()
            elapsed = time.perf_counter() - start
            extra = dict(search.get_extra_info())
            trans = result.get_trans()
            markings = result.get_markings()
            makespan = markings[-1].get_prefix() if markings and len(trans) > 0 else -1
            reach_goal = bool(extra.get("reachGoal"))
            if reach_goal and makespan >= 0:
                success_count += 1
                makespans.append(makespan)
            details.append(
                f"{suite_name}:{env_name}|goal={reach_goal}|makespan={makespan}|"
                f"trans_count={len(trans)}|elapsed={elapsed:.6f}s"
            )
        except BaseException as exc:
            details.append(
                f"{suite_name}:{env_name}|goal=False|makespan=-1|trans_count=0|elapsed=-1|"
                f"error={type(exc).__name__}:{exc}"
            )
            print(f"Warning: 推理套件 {suite_name} 在 {env_name} 出错: {exc}", flush=True)

    # 恢复原来的环境
    restore_env = next((env for env in env_pool if env.get("name") == saved_env_name), None)
    if restore_env is None and hasattr(search, "env_pool") and search.env_pool:
        restore_env = next((env for env in search.env_pool if env.get("name") == saved_env_name), None)
    if restore_env is not None:
        try:
            search.switch_environment(restore_env)
        except BaseException:
            pass

    avg_makespan = int(sum(makespans) / len(makespans)) if makespans else -1
    worst_makespan = max(makespans) if makespans else -1
    summary = (
        f"{suite_name}_summary=success:{success_count}/{len(env_pool)},"
        f"avg_makespan:{avg_makespan},worst_makespan:{worst_makespan}"
    )
    return details, summary

class PetriNetGCNPPOProHQ(PetriNetGCNPPOPro):
    def __init__(self, petri_net, end, pre, post, min_delay_p, env_pool=None, **kwargs):
        self.env_pool = env_pool
        if env_pool:
            for env in env_pool:
                if not hasattr(env["petri_net"], "max_residence_time"):
                    raise ValueError(f"petri_net in {env.get('name', 'unknown')} 必须提供 max_residence_time")
        else:
            if not hasattr(petri_net, "max_residence_time"):
                raise ValueError("petri_net 必须提供 max_residence_time")
            
        default_params = {
            "lambda_p": _env_int("GCN_PPO_HQ_LAMBDA_P", 512),
            "lambda_t": _env_int("GCN_PPO_HQ_LAMBDA_T", 128),
            "extra_p2t_rounds": _env_int("GCN_PPO_HQ_EXTRA_P2T_ROUNDS", 6),

            "gamma": _env_float("GCN_PPO_HQ_GAMMA", 0.999),
            "lr": _env_float("GCN_PPO_HQ_LR", 3e-4),
            # ★ 新增：L2正则化系数，抑制模型过拟合到特定训练网络的拓扑细节
            "weight_decay": _env_float("GCN_PPO_HQ_WEIGHT_DECAY", 1e-4), # 1e-5
            "steps_per_epoch": _env_int("GCN_PPO_HQ_STEPS_PER_EPOCH", 6144),  # 12288
            "minibatch_size": _env_int("GCN_PPO_HQ_MINIBATCH_SIZE", 128),
            "ppo_epochs": _env_int("GCN_PPO_HQ_PPO_EPOCHS", 4),  # ppo更新轮数
            "target_kl": _env_float("GCN_PPO_HQ_TARGET_KL", 0.07),

            "entropy_coef_start": _env_float("GCN_PPO_HQ_ENTROPY_START", 0.32), #0.25
            "entropy_coef_end": _env_float("GCN_PPO_HQ_ENTROPY_END", 0.18),

            "temperature_start": _env_float("GCN_PPO_HQ_TEMPERATURE_START", 2.3),# 影响动作分布
            "temperature_end": _env_float("GCN_PPO_HQ_TEMPERATURE_END", 1.2),

            # ★ 修正：原值1500会被全局clip到100，使目标奖励信号完全失效。
            # 调小为150后，目标奖励可以完整传递，makespan改进也能产生有效梯度信号。
            "reward_goal_bonus": _env_float("GCN_PPO_HQ_REWARD_GOAL", 150.0),
            "reward_deadlock_penalty": _env_float("GCN_PPO_HQ_REWARD_DEADLOCK", 90.0),
            "reward_progress_weight": 2.0,
            "reward_repeat_penalty": _env_float("GCN_PPO_HQ_REWARD_REPEAT", 2.7),
            "reward_time_scale": _env_float("GCN_PPO_HQ_REWARD_TIME_SCALE", 1000.0),
            "reward_residence_warn_ratio": _env_float("GCN_PPO_HQ_RESIDENCE_WARN_RATIO", 0.7),
            "reward_residence_penalty_max": _env_float("GCN_PPO_HQ_RESIDENCE_PENALTY_MAX", 30.0),
            "reward_residence_safe_bonus": _env_float("GCN_PPO_HQ_RESIDENCE_SAFE_BONUS", 0.5),
            "reward_mobility_weight": _env_float("GCN_PPO_HQ_MOBILITY_WEIGHT", 0.3),

            "beam_width": _env_int("GCN_PPO_HQ_BEAM_WIDTH", 100),
            "beam_depth": _env_int("GCN_PPO_HQ_BEAM_DEPTH", 800),

            "pool_eval_interval": _env_int("GCN_PPO_HQ_POOL_EVAL_INTERVAL", 4),  # 每4个epoch评估一次
            "curriculum_epochs": _env_int("GCN_PPO_HQ_CURRICULUM_EPOCHS", 4),  # 预热阶段epoch数

            # ★ 新增：eval_env_pool 独立评估间隔（0=禁用；通过实例属性传入，不进参数字典）
            "mask_cache_limit": _env_int("GCN_PPO_HQ_MASK_CACHE_LIMIT", 40000),
            "mixed_rollout": _env_bool("GCN_PPO_HQ_MIXED_ROLLOUT", True),
            "cross_env_gae": _env_bool("GCN_PPO_HQ_CROSS_ENV_GAE", True),
            # ★ 修复：原 `"0" == "0"` 写法恒为 True 与命名相反；现按字面语义解析。
            "async_collection": _env_bool("GCN_PPO_HQ_ASYNC_COLLECTION", False),
            "envs_per_epoch": _env_int("GCN_PPO_HQ_ENVS_PER_EPOCH", 4),
            "dynamic_curriculum": _env_bool("GCN_PPO_HQ_DYNAMIC_CURRICULUM", True),
            "curriculum_warmup_ratio": _env_float("GCN_PPO_HQ_CURRICULUM_WARMUP_RATIO", 0.3),
            "stochastic_num_rollouts": _env_int("GCN_PPO_HQ_STOCHASTIC_NUM_ROLLOUTS", 50),
            "stochastic_temperature": _env_float("GCN_PPO_HQ_STOCHASTIC_TEMPERATURE", 1.2),
            "use_deadlock_controller": _env_bool("GCN_PPO_HQ_USE_DEADLOCK_CONTROLLER", True),
        }
        for k, v in default_params.items():
            kwargs.setdefault(k, v)
            
        super().__init__(
            petri_net=petri_net,
            end=end,
            pre=pre,
            post=post,
            min_delay_p=min_delay_p,
            env_pool=env_pool,
            **kwargs
        )
        
        if self.env_pool:
            self.switch_environment(self.env_pool[0])

    @staticmethod
    def _fmt_epoch_metric(value, precision=4):
        if value is None:
            return "N/A"
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return f"{value:.{precision}f}"
        return str(value)

    def _log_epoch_summary(self, metrics):
        """
        每个训练 epoch 结束后的详细日志。
        只格式化 train_model 已经计算出的指标，不额外触发评估或模型前向计算。
        """
        pool_metrics = metrics.get("pool_metrics") or {}
        eval_pool_metrics = metrics.get("eval_pool_metrics") or {}
        eval_accuracy = 1.0 if metrics.get("eval_success") else 0.0
        pool_success_rate = pool_metrics.get("success_rate", self.extra_info.get("poolSuccessRate"))
        eval_pool_success_rate = eval_pool_metrics.get("success_rate", self.extra_info.get("evalPoolSuccessRate"))
        eval_pool_avg = eval_pool_metrics.get("avg_makespan", self.extra_info.get("evalPoolAvgMakespan"))
        if isinstance(pool_success_rate, (int, float)) and pool_success_rate < 0:
            pool_success_rate = None
        if isinstance(eval_pool_success_rate, (int, float)) and eval_pool_success_rate < 0:
            eval_pool_success_rate = None
        if isinstance(eval_pool_avg, (int, float)) and eval_pool_avg < 0:
            eval_pool_avg = None

        separator = "-" * 72
        log_lines = [
            separator,
            f"[Epoch {metrics.get('epoch_idx', 0):03d}] "
            f"env={metrics.get('env_name', 'unknown')} "
            f"steps={metrics.get('total_steps', 0)}/{metrics.get('max_train_steps', 0)} "
            f"(collected={metrics.get('steps_collected', 0)})",
            "  Loss       | "
            f"train={self._fmt_epoch_metric(metrics.get('train_loss'))} "
            f"actor={self._fmt_epoch_metric(metrics.get('actor_loss'))} "
            f"critic={self._fmt_epoch_metric(metrics.get('critic_loss'))} "
            f"validation=N/A",
            "  Accuracy   | "
            f"greedy_eval={eval_accuracy:.2%} "
            f"pool_success={self._fmt_epoch_metric(pool_success_rate, 2)} "
            f"eval_pool_success={self._fmt_epoch_metric(eval_pool_success_rate, 2)}",
            "  Metrics    | "
            f"avg_reward={self._fmt_epoch_metric(metrics.get('avg_reward'), 2)} "
            f"eval_makespan={self._fmt_epoch_metric(metrics.get('eval_show'))} "
            f"best_makespan={self._fmt_epoch_metric(metrics.get('best_show'))} "
            f"eval_pool_avg={self._fmt_epoch_metric(eval_pool_avg, 2)}",
            "  Optimizer  | "
            f"kl={self._fmt_epoch_metric(metrics.get('kl'), 6)} "
            f"lr={self._fmt_epoch_metric(metrics.get('learning_rate'), 6)} "
            f"entropy_coef={self._fmt_epoch_metric(metrics.get('entropy_coef'), 4)} "
            f"temperature={self._fmt_epoch_metric(metrics.get('temperature'), 4)}",
        ]
        diagnostics = _build_epoch_diagnostics(
            metrics,
            previous_metrics=getattr(self, "_last_epoch_metrics", None),
            target_kl=metrics.get("target_kl", getattr(self, "target_kl", None)),
        )
        if diagnostics:
            log_lines.append("  Diagnostics| " + " ".join(diagnostics))
        log_lines.append(separator)
        self._last_epoch_metrics = dict(metrics)

        for line in log_lines:
            self._log(line)

        # 将每个 epoch 的训练日志追加到 progress_path 指向的文件，便于训练中断后回看。
        progress_path = getattr(self, "epoch_log_path", "")
        if progress_path:
            os.makedirs(os.path.dirname(progress_path), exist_ok=True)
            with open(progress_path, "a", encoding="utf-8") as f:
                f.write("\n".join(log_lines) + "\n")


def main():
    base_dir = os.path.dirname(__file__)
    out_path = os.path.join(base_dir, "results/Reference_ppo_outputs/class/out/case2-7.txt")
    progress_path = os.path.join(base_dir, "results/Reference_ppo_outputs/class/progress/case2-7.txt")
    
    try:
        default_train_files = [ 

"1-2-1.txt","1-2-2.txt","1-2-3.txt","1-2-4.txt","1-2-5.txt","1-2-6.txt","1-2-7.txt","1-2-8.txt",
"3-1-1.txt","3-1-2.txt","3-1-3.txt","3-1-4.txt","3-1-5.txt","3-1-6.txt","3-1-7.txt","3-1-8.txt",

                               ]

        train_files = _env_list("GCN_PPO_HQ_TRAIN_FILES") or default_train_files
        # 训练文件搜索路径
        train_roots = [
                    # "resources/resources_new/train/class/case1/test"
                    # "resources/resources_new/train/class/case1/resources"
                    # "resources/resources_new/train/class/case2/resources"
                    "resources/resources_new/train/class/case2-1/resources"
                    # "resources/resources_new/train/class/case3/resources"
                    # "resources/resources_new/train/class/case4/resources"
                    # "resources/resources_new/train/class/case5/resources"
                        ]
        auto_eval_enabled = _env_bool("GCN_PPO_HQ_AUTO_EVAL", True)
        auto_eval_suffix = os.environ.get("GCN_PPO_HQ_AUTO_EVAL_SUFFIX", "20").strip() or "20"
        default_eval_files = _derive_eval_files_from_train_files(train_files, auto_eval_suffix) if auto_eval_enabled else []
        eval_files = _env_list("GCN_PPO_HQ_EVAL_FILES") or default_eval_files
        eval_roots = [
                    "resources/resources_new/test/class_test/case1",
                    "resources/resources_new/test/class_test/case2",
                    "resources/resources_new/test/class_test/case3",
                    "resources/resources_new/test/class_test/case4",
                    "resources/resources_new/test/class_test/case5",
                    "resources/resources_new/train/class/case1/resources",
                    "resources/resources_new/train/class/case2/resources",
                    "resources/resources_new/train/class/case3/resources",
                    "resources/resources_new/train/class/case4/resources",
                    "resources/resources_new/train/class/case5/resources",
                    ]

        env_pool = _load_env_pool(base_dir, train_files, train_roots)
        if not env_pool:
            raise ValueError("ERROR\n没有找到任何可用的网文件用于训练!")
        eval_env_pool = _load_env_pool(base_dir, eval_files, eval_roots) if eval_files else []
        if auto_eval_enabled and default_eval_files and not eval_env_pool:
            print(
                "Warning: 自动推导了评估文件但未加载到 eval_env_pool: "
                + ",".join(default_eval_files),
                flush=True,
            )

        main_env = env_pool[0]
        max_place_count = max(len(e["pre"]) for e in env_pool)
        max_trans_count = max(len(e["pre"][0]) for e in env_pool)
        max_constrained_count = max(sum(1 for val in e["max_residence_time"] if val < 2 ** 31 - 1) for e in env_pool)
        complexity = max(max_place_count, max_trans_count)
        
        os.makedirs(os.path.dirname(progress_path), exist_ok=True)
        with open(progress_path, "w", encoding="utf-8") as f:
            f.write("")

        # 修复：原写法 `"1" == "0"` 永远为 False，导致 fast_mode 永远关闭。
        # 现按字面意义解析：默认 True；用户置 0/false 即可切换到完整训练模式。
        fast_mode = _env_bool("GCN_PPO_HQ_FAST", 0)
        env_count = len(env_pool)
        
        if fast_mode:
            max_train_steps = 25000 * env_count
            mode = "hq-fast-generalization"
        else:
            base_steps = 10000 * env_count
            extra_steps = (complexity * 2000 + max_constrained_count * 3000) * env_count
            max_train_steps = min(491520, max(50000, base_steps + extra_steps))
            mode = "hq-full-generalization"

        line = "GCN-PPO Pro HQ mode: " + mode
        print(line, flush=True)
            
        schedule_line = (
            f"schedule max_train_steps={max_train_steps} "
            f"max_places={max_place_count} max_trans={max_trans_count}"
        )
        print(schedule_line, flush=True)
        print("train_envs=" + ",".join(env["name"] for env in env_pool), flush=True)
        if eval_env_pool:
            print("eval_envs=" + ",".join(env["name"] for env in eval_env_pool), flush=True)
        elif auto_eval_enabled:
            print("eval_envs=none(auto_eval_enabled=1)", flush=True)

        constrained_places = [f"{i}:{val}" for i, val in enumerate(main_env["max_residence_time"]) if val < 2 ** 31 - 1]
        constraint_line = f"main_env({main_env['name']})_residence_constraints=" + (",".join(constrained_places) if constrained_places else "none")
        print(constraint_line, flush=True)
        
        # 读取 IL 专家轨迹信息，用于计算训练/推理步数上限
        il_result_mode = normalize_il_mode(os.environ.get("GCN_PPO_HQ_IL_MODE", "auto"))
        expert_steps_max = 0
        for env in env_pool:
            il_result_path = resolve_il_result(base_dir, il_result_mode, net_stem=os.path.splitext(env["name"])[0])
            il_result_info = _read_result_kv(il_result_path)
            esteps = _infer_expert_steps(il_result_info)
            if esteps > expert_steps_max:
                expert_steps_max = esteps
        main_schedule = _compute_step_schedule(main_env["context"], expert_steps_max)
        print(f"step_schedule source={main_schedule['step_reference_source']} "
              f"min_steps={main_schedule['min_steps']} max_steps={main_schedule['max_steps']} "
              f"inference_max_steps={main_schedule['inference_max_steps']}", flush=True)
            
        similar_finetune_step_scale = float(os.environ.get("GCN_PPO_HQ_SIMILAR_FINETUNE_SCALE", "0.35"))
        similar_finetune_min_steps = int(os.environ.get("GCN_PPO_HQ_SIMILAR_FINETUNE_MIN_STEPS", "10000"))

        eval_pool_interval = _env_int("GCN_PPO_HQ_EVAL_POOL_INTERVAL", 8)  # 每8个epoch评估一次相似测试网络
        search = PetriNetGCNPPOProHQ(
            petri_net=main_env["petri_net"],
            end=main_env["end"],
            pre=main_env["pre"],
            post=main_env["post"],
            min_delay_p=main_env["min_delay_p"],
            env_pool=env_pool,
            eval_env_pool=eval_env_pool,        # ★ 传入独立评估池，训练中监控泛化能力
            eval_pool_interval=eval_pool_interval,  # ★ 评估频率
            max_train_steps=max_train_steps,
            verbose=True,
            search_strategy="greedy",
            mixed_rollout=True,
            envs_per_epoch=4,
            # use_deadlock_controller = False,
        )
        search.epoch_log_path = progress_path
        print(
            "model_config="
            + f"lambda_p:{search.model.actor_net.lambda_p if hasattr(search.model.actor_net, 'lambda_p') else 'na'},"
            + f"steps_per_epoch:{search.steps_per_epoch},minibatch_size:{search.minibatch_size},"
            + f"ppo_epochs:{search.ppo_epochs},beam_width:{search.beam_width},beam_depth:{search.beam_depth}",
            flush=True
        )

        signature = build_signature(main_env["path"], main_env["context"])
        profile = build_profile(main_env["context"])
        #==========================================================================================
        ckpt_path = checkpoint_path(base_dir, "Reference_checkpoint/class/case2-7", signature)

        reuse_checkpoint = _env_bool("GCN_PPO_HQ_REUSE", False)
        reuse_similar = _env_bool("GCN_PPO_HQ_REUSE_SIMILAR", True)
        # 修复：原 `"0" == "0"` 永远为 True，与命名语义相反；改为标准布尔解析。
        finetune_on_similar = _env_bool("GCN_PPO_HQ_FINETUNE_ON_SIMILAR", False)
        custom_ckpt_path = os.environ.get("GCN_PPO_HQ_CHECKPOINT_PATH", "")
        finetune_from_custom = _env_bool("GCN_PPO_HQ_FINETUNE_FROM_CUSTOM", True)

        # 模仿学习热启动参数
        # 修复：原 `"1" == "0"` 永远为 False，IL 热启动几乎从未生效，
        # 完全违反代码的命名意图。现统一使用 _env_bool。
        use_il_warmstart = _env_bool("GCN_PPO_HQ_IL_WARMSTART", False)
        il_mode = normalize_il_mode(os.environ.get("GCN_PPO_HQ_IL_MODE", "bc"))
        # 默认相对路径，避免硬编码绝对路径带来的可移植性问题
        default_il_ckpt = os.path.join(base_dir, "checkpoints", "bc_scene_1.pt")
        il_ckpt_path = os.environ.get("GCN_PPO_HQ_IL_CKPT_PATH", default_il_ckpt).strip()
        
        loaded_checkpoint = False
        checkpoint_mode = "none"
        il_warmstarted = False
        il_warm_method = "none"
        init_source = "scratch"
        
        # 第一步：尝试 IL 热启动（在 PPO checkpoint 之前）
        if use_il_warmstart and not (custom_ckpt_path and os.path.exists(custom_ckpt_path)):
            il_resolved_path = il_ckpt_path
            if not il_resolved_path:
                il_resolved_path = resolve_il_checkpoint(
                    base_dir,
                    il_mode,
                    net_stem="",
                    scene_id="",
                    explicit="",
                )
            
            if il_resolved_path and os.path.exists(il_resolved_path):
                il_warmstarted = search.il_warmstart(il_resolved_path, il_mode)
                if il_warmstarted:
                    il_warm_method, il_source = classify_il_artifact(base_dir, il_resolved_path)
                    init_source = il_warm_method + "_" + il_source
                    print(f"[IL-Warmstart] source={init_source} checkpoint={il_resolved_path}", flush=True)
            else:
                print(f"[IL-Warmstart] No IL checkpoint found (mode={il_mode})", flush=True)
        
        # 第二步：加载 PPO checkpoint（如果存在，会覆盖 IL 热启动的权重）
        if custom_ckpt_path and os.path.exists(custom_ckpt_path):
            print(f"[Checkpoint] Loading custom checkpoint: {custom_ckpt_path}", flush=True)
            saved = torch.load(custom_ckpt_path, map_location="cpu")
            load_compatible_state(search.model.actor_net, saved.get("actor_state", {}))
            load_compatible_state(search.model.value_head, saved.get("critic_state", {}))
            
            optimizer_state = saved.get("optimizer_state")
            if optimizer_state is not None:
                try:
                    search.optimizer.load_state_dict(optimizer_state)
                except BaseException:
                    pass
            
            search.best_train_makespan = saved.get("best_train_makespan", 2 ** 31 - 1)
            search.best_train_trans = saved.get("best_train_trans", [])
            search.best_records = saved.get("best_records", {})
            search.extra_info["bestTrainMakespan"] = search.best_train_makespan if search.best_train_makespan < 2 ** 31 - 1 else -1
            search.extra_info["bestTrainTransCount"] = len(search.best_train_trans)
            
            if finetune_from_custom:
                search.is_trained = False
                print(f"[Checkpoint] Finetune mode enabled, will continue training.", flush=True)
            else:
                search.is_trained = True
                print(f"[Checkpoint] Inference mode enabled, will skip training.", flush=True)
            
            loaded_checkpoint = True
            checkpoint_mode = "custom"
            
        elif reuse_checkpoint:
            selected = find_checkpoint(base_dir, "gcn_ppo_pro_hq_general", signature, profile, allow_similar=reuse_similar)
            checkpoint_mode = selected["mode"]
            load_path = selected["path"]
            
            if load_path and os.path.exists(load_path):
                saved = torch.load(load_path, map_location="cpu")
                load_compatible_state(search.model.actor_net, saved.get("actor_state", {}))
                load_compatible_state(search.model.value_head, saved.get("critic_state", {}))
                
                optimizer_state = saved.get("optimizer_state")
                if optimizer_state is not None:
                    try:
                        search.optimizer.load_state_dict(optimizer_state)
                    except BaseException: pass
                        
                search.best_train_makespan = saved.get("best_train_makespan", 2 ** 31 - 1)
                search.best_train_trans = saved.get("best_train_trans", [])
                
                # 从 Checkpoint 加载泛化记录字典！
                search.best_records = saved.get("best_records", {})
                
                search.extra_info["bestTrainMakespan"] = search.best_train_makespan if search.best_train_makespan < 2 ** 31 - 1 else -1
                search.extra_info["bestTrainTransCount"] = len(search.best_train_trans)
                search.is_trained = True
                loaded_checkpoint = True
                
                if checkpoint_mode == "similar" and finetune_on_similar:
                    search.is_trained = False
                    scaled_steps = max(similar_finetune_min_steps, int(search.max_train_steps * similar_finetune_step_scale))
                    search.max_train_steps = min(search.max_train_steps, scaled_steps)

        checkpoint_line = "checkpoint_loaded=" + ("1" if loaded_checkpoint else "0")
        checkpoint_line += " il_warmstarted=" + ("1" if il_warmstarted else "0")
        checkpoint_line += " init_source=" + init_source
        print(checkpoint_line, flush=True)

        search.switch_environment(main_env)
        
        start = time.perf_counter()
        result = search.search()
        elapsed = time.perf_counter() - start
        extra = search.get_extra_info()
        seen_details, seen_summary = _run_inference_suite(search, env_pool, "seen_pool")
        unseen_details, unseen_summary = _run_inference_suite(search, eval_env_pool, "unseen_pool") if eval_env_pool else ([], "")
        final_seen_metrics = _parse_suite_summary(seen_summary)
        final_unseen_metrics = _parse_suite_summary(unseen_summary)

        to_save = {
            "signature": signature,
            "profile": profile,
            "actor_state": search.model.actor_net.state_dict(),
            "critic_state": search.model.value_head.state_dict(),
            "optimizer_state": search.optimizer.state_dict(),
            "best_train_makespan": search.best_train_makespan,
            "best_train_trans": search.best_train_trans,
            "best_records": getattr(search, "best_records", {}),
            # ★ 新增：同时保存训练过程中池评估最优的快照，供后续分析使用
            "best_pool_snapshot": getattr(search, "_best_snapshot", None),
        }
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(to_save, ckpt_path)

        trans = result.get_trans()
        markings = result.get_markings()
        
        t_map_v = getattr(main_env["context"].get("matrix_translator"), "t_map_v", {})
        trans_names = [str(t_map_v.get(t, t)) for t in trans] if trans and t_map_v else [str(t) for t in trans]
        
        out = "elapsed:" + format(elapsed, ".6f") + "s\n"
        out += "trans_count:" + str(len(trans_names)) + "\n"
        out += "trans_sequence:" + ("->".join(t for t in trans_names) if trans_names else "") + "\n"
        out += "makespan:" + str(markings[-1].get_prefix() if markings and len(trans) > 0 else -1) + "\n"
        out += "reach_goal:" + str(extra.get("reachGoal")) + "\n"
        out += "goal_distance:" + str(extra.get("goalDistance")) + "\n"
        out += "train_steps:" + str(extra.get("trainSteps", 0)) + "\n"
        out += "best_train_makespan:" + str(extra.get("bestTrainMakespan", -1)) + "\n"
        out += "pool_success_rate:" + str(extra.get("poolSuccessRate", -1)) + "\n"
        out += "pool_avg_makespan:" + str(extra.get("poolAvgMakespan", -1)) + "\n"
        out += "pool_worst_makespan:" + str(extra.get("poolWorstMakespan", -1)) + "\n"
        out += "eval_pool_success_rate:" + str(extra.get("evalPoolSuccessRate", -1)) + "\n"
        out += "eval_pool_avg_makespan:" + str(extra.get("evalPoolAvgMakespan", -1)) + "\n"
        out += "final_seen_pool_success_rate:" + str(final_seen_metrics.get("success_rate", -1)) + "\n"
        out += "final_seen_pool_avg_makespan:" + str(final_seen_metrics.get("avg_makespan", -1)) + "\n"
        out += "final_seen_pool_worst_makespan:" + str(final_seen_metrics.get("worst_makespan", -1)) + "\n"
        out += "final_unseen_pool_success_rate:" + str(final_unseen_metrics.get("success_rate", -1)) + "\n"
        out += "final_unseen_pool_avg_makespan:" + str(final_unseen_metrics.get("avg_makespan", -1)) + "\n"
        out += "final_unseen_pool_worst_makespan:" + str(final_unseen_metrics.get("worst_makespan", -1)) + "\n"
        out += "il_warmstarted:" + ("1" if il_warmstarted else "0") + "\n"
        out += "il_warm_method:" + il_warm_method + "\n"
        out += "init_source:" + init_source + "\n"
        out += seen_summary + "\n"
        if unseen_summary:
            out += unseen_summary + "\n"
        if seen_details:
            out += "\n".join(seen_details) + "\n"
        if unseen_details:
            out += "\n".join(unseen_details) + "\n"
        out += "checkpoint_path:" + ckpt_path + "\n"
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(out)
            
        print("\n=== Final Result ===")
        print(out, flush=True)
        
    except BaseException:
        err = "ERROR\n" + traceback.format_exc()
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(err)
        print(err, flush=True)

if __name__ == "__main__":
    main()
