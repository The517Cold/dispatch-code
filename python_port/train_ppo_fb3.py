import os
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
# from petri_gcn_ppo_4 import PetriNetGCNPPOPro
from petri_gcn_ppo_4_1 import PetriNetGCNPPOPro
"""
    这个文档的代码是只针对多网训练同时不加专家序列的训练函数
"""

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
            "lambda_p": _env_int("GCN_PPO_HQ_LAMBDA_P", 256),
            "lambda_t": _env_int("GCN_PPO_HQ_LAMBDA_T", 64),
            "extra_p2t_rounds": _env_int("GCN_PPO_HQ_EXTRA_P2T_ROUNDS", 3),
            "gamma": _env_float("GCN_PPO_HQ_GAMMA", 0.999),
            "lr": _env_float("GCN_PPO_HQ_LR", 3e-4),
            "steps_per_epoch": _env_int("GCN_PPO_HQ_STEPS_PER_EPOCH", 6144),  # 12288
            "minibatch_size": _env_int("GCN_PPO_HQ_MINIBATCH_SIZE", 128),
            "ppo_epochs": _env_int("GCN_PPO_HQ_PPO_EPOCHS", 2),
            "target_kl": _env_float("GCN_PPO_HQ_TARGET_KL", 0.05),
            "entropy_coef_start": _env_float("GCN_PPO_HQ_ENTROPY_START", 0.15),##0.09
            "entropy_coef_end": _env_float("GCN_PPO_HQ_ENTROPY_END", 0.015),
            "temperature_start": _env_float("GCN_PPO_HQ_TEMPERATURE_START", 2.0),
            "temperature_end": _env_float("GCN_PPO_HQ_TEMPERATURE_END", 1.1),
            "reward_goal_bonus": _env_float("GCN_PPO_HQ_REWARD_GOAL", 1000.0),
            "reward_deadlock_penalty": _env_float("GCN_PPO_HQ_REWARD_DEADLOCK", 2000.0),
            "reward_progress_weight": 2.0,
            "reward_repeat_penalty": _env_float("GCN_PPO_HQ_REWARD_REPEAT", 1.5),
            "reward_time_scale": _env_float("GCN_PPO_HQ_REWARD_TIME_SCALE", 1000.0),
            "beam_width": _env_int("GCN_PPO_HQ_BEAM_WIDTH", 100),
            "beam_depth": _env_int("GCN_PPO_HQ_BEAM_DEPTH", 800),
            "pool_eval_interval": _env_int("GCN_PPO_HQ_POOL_EVAL_INTERVAL", 4),
            "curriculum_epochs": _env_int("GCN_PPO_HQ_CURRICULUM_EPOCHS", 8),
            "mask_cache_limit": _env_int("GCN_PPO_HQ_MASK_CACHE_LIMIT", 40000),
            "mixed_rollout": os.environ.get("GCN_PPO_HQ_MIXED_ROLLOUT", "0") == "0",
            "cross_env_gae": os.environ.get("GCN_PPO_HQ_CROSS_ENV_GAE", "0") == "0",
            "async_collection": os.environ.get("GCN_PPO_HQ_ASYNC_COLLECTION", "0") == "0",
            "envs_per_epoch": _env_int("GCN_PPO_HQ_ENVS_PER_EPOCH", 0),
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
    out_path = os.path.join(base_dir, "results/Reference_ppo_outputs/reference_ppo_result_case2_test4.txt")
    progress_path = os.path.join(base_dir, "results/Reference_ppo_outputs/reference_ppo_progress_case2_test4.txt")
    
    try:
        default_train_files = ["1-3.txt","1-3-9.txt",
                               "1-4.txt","1-4-9.txt",
                               "3-2.txt","3-2-9.txt",
                               "3-4.txt","3-4-9.txt",]
        train_files = _env_list("GCN_PPO_HQ_TRAIN_FILES") or default_train_files
        # 训练文件搜索路径
        train_roots = [
                       "resources/resources_new/train/case2/test1"
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
            max_train_steps = min(700000, max(50000, base_steps + extra_steps))
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
            
        similar_finetune_step_scale = float(os.environ.get("GCN_PPO_HQ_SIMILAR_FINETUNE_SCALE", "0.35"))
        similar_finetune_min_steps = int(os.environ.get("GCN_PPO_HQ_SIMILAR_FINETUNE_MIN_STEPS", "10000"))

        search = PetriNetGCNPPOProHQ(
            petri_net=main_env["petri_net"],
            end=main_env["end"],
            pre=main_env["pre"],
            post=main_env["post"],
            min_delay_p=main_env["min_delay_p"],
            env_pool=env_pool, 
            max_train_steps=max_train_steps,
            verbose=True,
            search_strategy = "greedy",
            mixed_rollout=True,
            envs_per_epoch = 4
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
        ckpt_path = checkpoint_path(base_dir, "Reference_checkpoint/reference_ppo_checkpoint_case2_test4", signature)

        reuse_checkpoint = os.environ.get("GCN_PPO_HQ_REUSE", "0") == "1"
        reuse_similar = os.environ.get("GCN_PPO_HQ_REUSE_SIMILAR", "1") == "1"
        finetune_on_similar = os.environ.get("GCN_PPO_HQ_FINETUNE_ON_SIMILAR", "0") == "0"
        custom_ckpt_path = os.environ.get("GCN_PPO_HQ_CHECKPOINT_PATH", "")
        #custom_ckpt_path = "checkpoints/gcn_ppo_pro_hq_general_family_large_test2_recipe10d-13_d2204f24b4dace947ef1d718f56cfc5b090b2c09.pt"
        finetune_from_custom = os.environ.get("GCN_PPO_HQ_FINETUNE_FROM_CUSTOM", "1") == "1"  # 是否从自定义检查点继续训练
        
        loaded_checkpoint = False
        checkpoint_mode = "none"
        
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
            "best_records": getattr(search, "best_records", {})
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
