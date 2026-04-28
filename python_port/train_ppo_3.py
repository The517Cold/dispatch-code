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
    # 从环境变量中获取整数，默认值为default
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def _env_float(name, default):
    # 从环境变量中获取浮点数，默认值为default
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else default


def _env_list(name):
    # 从环境变量中获取列表，每个元素之间用逗号分隔
    # 允许元素包含空格
    raw = os.environ.get(name, "").strip()
    return [item.strip() for item in raw.split(",") if item.strip()]


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
    # 运行推理套件，返回推理结果和摘要
    # 如果环境池为空，返回空列表和摘要
    if not env_pool:
        return [], f"{suite_name}_summary=success:0/0,avg_makespan:-1,worst_makespan:-1"

    saved_env_name = search.current_env_name
    details = []
    makespans = []
    success_count = 0

    for env in env_pool:
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
            f"{suite_name}:{env['name']}|goal={reach_goal}|makespan={makespan}|trans_count={len(trans)}|elapsed={elapsed:.6f}s"
        )
    # 恢复原来的环境
    restore_env = next((env for env in env_pool if env.get("name") == saved_env_name), None)
    if restore_env is None and hasattr(search, "env_pool"):
        restore_env = next((env for env in search.env_pool if env.get("name") == saved_env_name), None)
    if restore_env is not None:
        search.switch_environment(restore_env)

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
            "weight_decay": _env_float("GCN_PPO_HQ_WEIGHT_DECAY", 1e-5),
            "steps_per_epoch": _env_int("GCN_PPO_HQ_STEPS_PER_EPOCH", 6144),  # 12288
            "minibatch_size": _env_int("GCN_PPO_HQ_MINIBATCH_SIZE", 128),
            "ppo_epochs": _env_int("GCN_PPO_HQ_PPO_EPOCHS", 4),  # ppo更新轮数
            "target_kl": _env_float("GCN_PPO_HQ_TARGET_KL", 0.07),

            "entropy_coef_start": _env_float("GCN_PPO_HQ_ENTROPY_START", 0.20),##0.09
            "entropy_coef_end": _env_float("GCN_PPO_HQ_ENTROPY_END", 0.035),

            "temperature_start": _env_float("GCN_PPO_HQ_TEMPERATURE_START", 2.3),
            "temperature_end": _env_float("GCN_PPO_HQ_TEMPERATURE_END", 1.4),

            # ★ 修正：原值1500会被全局clip到100，使目标奖励信号完全失效。
            # 调小为150后，目标奖励可以完整传递，makespan改进也能产生有效梯度信号。
            "reward_goal_bonus": _env_float("GCN_PPO_HQ_REWARD_GOAL", 150.0),
            "reward_deadlock_penalty": _env_float("GCN_PPO_HQ_REWARD_DEADLOCK", 90.0),
            "reward_progress_weight": 2.0,
            "reward_repeat_penalty": _env_float("GCN_PPO_HQ_REWARD_REPEAT", 2.7),
            "reward_time_scale": _env_float("GCN_PPO_HQ_REWARD_TIME_SCALE", 1000.0),

            "beam_width": _env_int("GCN_PPO_HQ_BEAM_WIDTH", 100),
            "beam_depth": _env_int("GCN_PPO_HQ_BEAM_DEPTH", 800),

            "pool_eval_interval": _env_int("GCN_PPO_HQ_POOL_EVAL_INTERVAL", 4),  # 每4个epoch评估一次
            "curriculum_epochs": _env_int("GCN_PPO_HQ_CURRICULUM_EPOCHS", 4),  # 预热阶段epoch数

            # ★ 新增：eval_env_pool 独立评估间隔（0=禁用；通过实例属性传入，不进参数字典）
            "mask_cache_limit": _env_int("GCN_PPO_HQ_MASK_CACHE_LIMIT", 40000),
            "mixed_rollout": os.environ.get("GCN_PPO_HQ_MIXED_ROLLOUT", "1") == "1",
            "cross_env_gae": os.environ.get("GCN_PPO_HQ_CROSS_ENV_GAE", "1") == "1",
            "async_collection": os.environ.get("GCN_PPO_HQ_ASYNC_COLLECTION", "0") == "0",
            "envs_per_epoch": _env_int("GCN_PPO_HQ_ENVS_PER_EPOCH", 4),  # 每个epoch选3个环境进行训练
            "dynamic_curriculum": os.environ.get("GCN_PPO_HQ_DYNAMIC_CURRICULUM", "1") == "1",
            "curriculum_warmup_ratio": _env_float("GCN_PPO_HQ_CURRICULUM_WARMUP_RATIO", 0.3),
            "stochastic_num_rollouts": _env_int("GCN_PPO_HQ_STOCHASTIC_NUM_ROLLOUTS", 50),
            "stochastic_temperature": _env_float("GCN_PPO_HQ_STOCHASTIC_TEMPERATURE", 1.2),
            "use_deadlock_controller": os.environ.get("GCN_PPO_HQ_USE_DEADLOCK_CONTROLLER", "1") == "1",
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


def main():
    base_dir = os.path.dirname(__file__)
    out_path = os.path.join(base_dir, "results/Reference_ppo_outputs/test/test1-17.txt")
    progress_path = os.path.join(base_dir, "results/Reference_ppo_outputs/test/test1-17.txt")
    
    try:
        default_train_files = [ 
                            "1-2-13-1.txt","1-2-13-2.txt","1-2-13-3.txt","1-2-13-4.txt","1-2-13-5.txt",
                               ]
        train_files = _env_list("GCN_PPO_HQ_TRAIN_FILES") or default_train_files
        # 训练文件搜索路径
        train_roots = [
                       "resources/resources_new/train/family1"
                        ]

        eval_files = _env_list("GCN_PPO_HQ_EVAL_FILES")
        eval_roots = ["resources/resources_new/family2/family-2-1",
                       "resources/resources_new/family2/family-2-2",
                       "resources/resources_new/family2/family-2-3",
                       "resources/resources_new/family2/family-2-4",
                       "resources/resources_new/family2/family-2-5"]

        env_pool = _load_env_pool(base_dir, train_files, train_roots)
        if not env_pool:
            raise ValueError("ERROR\n没有找到任何可用的网文件用于训练!")
        eval_env_pool = _load_env_pool(base_dir, eval_files, eval_roots) if eval_files else []

        main_env = env_pool[0]
        max_place_count = max(len(e["pre"]) for e in env_pool)
        max_trans_count = max(len(e["pre"][0]) for e in env_pool)
        max_constrained_count = max(sum(1 for val in e["max_residence_time"] if val < 2 ** 31 - 1) for e in env_pool)
        complexity = max(max_place_count, max_trans_count)
        
        with open(progress_path, "w", encoding="utf-8") as f:
            f.write("")

        fast_mode = os.environ.get("GCN_PPO_HQ_FAST", "1") == "0"
        env_count = len(env_pool)
        
        if fast_mode:
            max_train_steps = 25000 * env_count
            mode = "hq-fast-generalization"
        else:
            base_steps = 10000 * env_count
            extra_steps = (complexity * 2000 + max_constrained_count * 3000) * env_count
            max_train_steps = min(307200, max(50000, base_steps + extra_steps))
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
        ckpt_path = checkpoint_path(base_dir, "Reference_checkpoint/test/test1-17", signature)

        reuse_checkpoint = os.environ.get("GCN_PPO_HQ_REUSE", "0") == "1"
        reuse_similar = os.environ.get("GCN_PPO_HQ_REUSE_SIMILAR", "1") == "1"
        finetune_on_similar = os.environ.get("GCN_PPO_HQ_FINETUNE_ON_SIMILAR", "0") == "0"
        custom_ckpt_path = os.environ.get("GCN_PPO_HQ_CHECKPOINT_PATH", "")
        #custom_ckpt_path = "checkpoints/gcn_ppo_pro_hq_general_family_large_test2_recipe10d-13_d2204f24b4dace947ef1d718f56cfc5b090b2c09.pt"
        finetune_from_custom = os.environ.get("GCN_PPO_HQ_FINETUNE_FROM_CUSTOM", "1") == "1"
        
        # 模仿学习热启动参数
        use_il_warmstart = os.environ.get("GCN_PPO_HQ_IL_WARMSTART", "1") == "0"
        il_mode = normalize_il_mode(os.environ.get("GCN_PPO_HQ_IL_MODE", "bc"))
        il_ckpt_path = os.environ.get("GCN_PPO_HQ_IL_CKPT_PATH", "d:\\dispatch_code\\BC+DAgger+PPO\\new_job\\python_port\\checkpoints\\bc_scene_1.pt").strip()
        
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
