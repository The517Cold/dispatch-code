"""
零样本泛化测试：使用预训练的多网通用模型，对未见过的 Petri 网做推理评估。

v2 改进：
  1. 支持加载 best_pool_snapshot 作为推理权重（若 checkpoint 中包含）。
  2. 聚合所有测试网指标（成功率 / 平均/最差 makespan / 总耗时）并写入文本文件，
     方便对比改进前后泛化能力变化。
  3. 单网失败不阻塞整体评估;CLI 化：通过环境变量 / 命令行重写默认配置。
  4. 路径解析与统计代码模块化，便于在其它脚本中复用。

使用示例：
    # 默认运行（与原版相同）
    python test_unseen_net.py

    # 指定 checkpoint 与输出文件
    set GCN_PPO_HQ_CHECKPOINT_PATH=checkpoints/foo.pt
    set GCN_PPO_HQ_UNSEEN_OUTPUT=results/unseen_summary.txt
    python test_unseen_net.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from petri_net_io.utils.net_loader import load_petri_net_context, build_ttpn_with_residence, build_ttpn_by_token_with_res_time
from petri_net_io.utils.checkpoint_selector import load_compatible_state

from train_ppo_3 import PetriNetGCNPPOProHQ


_DEFAULT_NET_ROOTS = [
    "resources",
    "resources/resources_new/test/class_test/case1",
    "resources/resources_new/test/class_test/case2",
    #"resources/resources_new/test/class_test/case2-1",
    "resources/resources_new/test/class_test/case3",
    "resources/resources_new/test/class_test/case4",
    "resources/resources_new/test/class_test/case5",
    "resources/resources_new/train/class/case1/test",
    "resources/resources_new/train/class/case1/resources",
    "resources/resources_new/resources",
]


def _resolve_net_path(base_dir: str, file_name: str) -> str:
    """按预定义根目录列表查找网文件，找到第一个存在的路径。"""
    if os.path.isabs(file_name):
        return file_name
    for root in _DEFAULT_NET_ROOTS:
        candidate = os.path.join(base_dir, root, file_name)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(base_dir, _DEFAULT_NET_ROOTS[0], file_name)


def _load_checkpoint_state(checkpoint_path: str, prefer_snapshot: bool):
    """
    读取 checkpoint,并按需返回最优快照中的 actor/critic 权重。

    v2 改进：训练侧 train_ppo_3.py 保存了 best_pool_snapshot 字段；
    若开启 prefer_snapshot,则优先使用该快照作为推理模型权重,
    避免训练末期可能的过拟合退化。
    """
    saved = torch.load(checkpoint_path, map_location="cpu")
    actor_state = saved.get("actor_state", {})
    critic_state = saved.get("critic_state", {})
    src = "final"

    snapshot = saved.get("best_pool_snapshot")
    if prefer_snapshot and isinstance(snapshot, dict) and snapshot.get("actor_state"):
        actor_state = snapshot.get("actor_state", actor_state) or actor_state
        critic_state = snapshot.get("critic_state", critic_state) or critic_state
        src = "best_pool_snapshot"

    return saved, actor_state, critic_state, src


def test_unseen_net(
    test_file_name: str,
    checkpoint_path: str,
    *,
    base_dir: Optional[str] = None,
    prefer_snapshot: bool = True,
    verbose: bool = False,
    use_by_token: bool = False,
) -> Dict[str, object]:
    """
    在指定的全新 Petri 网上对预训练模型做零样本推理。

    适配机制说明：
        use_by_token 参数控制 Petri 网构建方式：
        - False（默认）：使用 build_ttpn_with_residence 构建 TTPPNHasResidenceTime，
          与原有行为完全一致，不包含 q-time 约束支持。
        - True：使用 build_ttpn_by_token_with_res_time 构建 TTPPNByTokenWithResTime，
          内部使用 TTPPNMarkingByTokenWithResTime（按 Token 组织状态），
          支持 q-time 约束检测和 q-time 超限惩罚计算。

        当启用 ByToken 模式时：
        1. petri_net 对象将具有 qtime / qtime_places 属性
        2. marking 对象将具有 qtime_map 属性（Token 级时间追踪）
        3. _step_env 中的 q-time 惩罚机制将被激活
        4. context 字典将包含 qtime_places 和 qtime 键

        使用场景：
        - 默认模式（use_by_token=False）：适用于没有 q-time 约束的 Petri 网，
          或需要与旧版行为保持一致时使用。
        - ByToken 模式（use_by_token=True）：适用于具有 q-time 约束的 Petri 网，
          需要在推理过程中检测和惩罚 q-time 违规时使用。

    Returns:
        Dict 形式的指标，便于上层聚合。
    """
    base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
    net_path = _resolve_net_path(base_dir, test_file_name)

    if not os.path.exists(net_path):
        return {
            "name": test_file_name,
            "status": "missing_net",
            "elapsed": -1.0,
            "trans_count": 0,
            "makespan": -1,
            "reach_goal": False,
            "trans_sequence": "",
        }

    if not os.path.exists(checkpoint_path):
        return {
            "name": test_file_name,
            "status": "missing_checkpoint",
            "elapsed": -1.0,
            "trans_count": 0,
            "makespan": -1,
            "reach_goal": False,
            "trans_sequence": "",
        }

    try:
        context = load_petri_net_context(net_path)
        if use_by_token:
            petri_net = build_ttpn_by_token_with_res_time(context)
        else:
            petri_net = build_ttpn_with_residence(context)

        test_env = {
            "petri_net": petri_net,
            "initial_marking": petri_net.get_marking().clone(),
            "end": context["end"],
            "pre": context["pre"],
            "post": context["post"],
            "min_delay_p": context["min_delay_p"],
            "max_residence_time": context["max_residence_time"],
            "name": test_file_name,
            "path": net_path,
            "context": context,
            "complexity_score": max(len(context["pre"]), len(context["pre"][0]))
            + sum(1 for val in context["max_residence_time"] if val < 2 ** 31 - 1) * 0.5,
        }

        # max_train_steps=0 仅做推理；search_strategy 默认 greedy 与训练一致
        search = PetriNetGCNPPOProHQ(
            petri_net=test_env["petri_net"],
            end=test_env["end"],
            pre=test_env["pre"],
            post=test_env["post"],
            min_delay_p=test_env["min_delay_p"],
            env_pool=[test_env],
            max_train_steps=0,
            verbose=False,
            search_strategy="greedy",
            stochastic_num_rollouts=50,
            stochastic_temperature=1.3,
            beam_depth=400,
            search_depth=800,
            beam_width=50,
            use_deadlock_controller=True,
        )

        saved, actor_state, critic_state, weight_source = _load_checkpoint_state(
            checkpoint_path, prefer_snapshot=prefer_snapshot
        )

        device = search.device
        if actor_state:
            actor_state = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in actor_state.items()
            }
            load_compatible_state(search.model.actor_net, actor_state)
        if critic_state:
            critic_state = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in critic_state.items()
            }
            load_compatible_state(search.model.value_head, critic_state)

        search.is_trained = True

        if "best_records" in saved and saved["best_records"]:
            search.best_records = saved["best_records"]
        if test_env["name"] not in search.best_records:
            search.best_records[test_env["name"]] = {"makespan": 2 ** 31 - 1, "trans": []}

        start = time.perf_counter()
        result = search.search()
        elapsed = time.perf_counter() - start

        extra = search.get_extra_info()
        trans = result.get_trans()
        markings = result.get_markings()

        t_map_v = getattr(context.get("matrix_translator"), "t_map_v", {})
        trans_names = [str(t_map_v.get(t, t)) for t in trans] if trans and t_map_v else [str(t) for t in trans]
        makespan = markings[-1].get_prefix() if markings and len(trans) > 0 else -1

        if verbose:
            print(f"=={test_file_name}==")
            print(f"Source          : {weight_source}")
            print(f"Elapsed Time    : {elapsed:.6f} s")
            print(f"Trans Count     : {len(trans_names)}")
            print(f"Makespan        : {makespan}")
            print(f"Reach Goal      : {extra.get('reachGoal')}")
            print(f"Trans Sequence  : {' -> '.join(trans_names)}")

        return {
            "name": test_file_name,
            "status": "ok",
            "weight_source": weight_source,
            "elapsed": float(elapsed),
            "trans_count": len(trans_names),
            "makespan": int(makespan) if isinstance(makespan, (int, float)) and makespan != -1 else int(makespan),
            "reach_goal": bool(extra.get("reachGoal")),
            "goal_distance": int(extra.get("goalDistance", -1)) if extra.get("goalDistance") is not None else -1,
            "trans_sequence": " -> ".join(trans_names),
        }
    except BaseException as exc:
        return {
            "name": test_file_name,
            "status": f"error:{type(exc).__name__}",
            "error_message": str(exc),
            "elapsed": -1.0,
            "trans_count": 0,
            "makespan": -1,
            "reach_goal": False,
            "trans_sequence": "",
        }


def aggregate(results: List[Dict[str, object]]) -> Dict[str, object]:
    """聚合多网测试结果为整体指标。"""
    total = len(results)
    succeeded = [r for r in results if r.get("reach_goal")]
    failures = [r for r in results if not r.get("reach_goal")]
    makespans = [r["makespan"] for r in succeeded if isinstance(r["makespan"], int) and r["makespan"] >= 0]
    elapsed_total = sum(r["elapsed"] for r in results if isinstance(r["elapsed"], (int, float)) and r["elapsed"] >= 0)
    return {
        "total": total,
        "success_count": len(succeeded),
        "success_rate": float(len(succeeded)) / float(total) if total else 0.0,
        "avg_makespan": (sum(makespans) / len(makespans)) if makespans else -1.0,
        "min_makespan": min(makespans) if makespans else -1,
        "max_makespan": max(makespans) if makespans else -1,
        "elapsed_total": elapsed_total,
        "failure_status": [r["status"] for r in failures if r.get("status") != "ok"],
    }


def write_report(out_path: str, agg: Dict[str, object], details: List[Dict[str, object]]) -> None:
    """以人类可读 + 机器可读两种格式输出报告。"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Zero-shot generalization report\n")
        f.write(f"total: {agg['total']}\n")
        f.write(f"success_count: {agg['success_count']}\n")
        f.write(f"success_rate: {agg['success_rate']:.4f}\n")
        avg_ms = agg.get("avg_makespan", -1.0)
        if isinstance(avg_ms, (int, float)) and avg_ms >= 0:
            f.write(f"avg_makespan: {avg_ms:.2f}\n")
        else:
            f.write("avg_makespan: -1\n")
        f.write(f"min_makespan: {agg['min_makespan']}\n")
        f.write(f"max_makespan: {agg['max_makespan']}\n")
        f.write(f"elapsed_total_seconds: {agg['elapsed_total']:.4f}\n")
        f.write("\n# Per-net details (jsonl)\n")
        for r in details:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


_DEFAULT_TARGETS = [
"1-1-1.txt","1-1-2.txt","1-1-3.txt","1-1-4.txt",
"2-1-1.txt","2-1-2.txt","2-1-3.txt","2-1-4.txt",
"2-2-1.txt","2-2-2.txt","2-2-3.txt","2-2-4.txt",
"2-3-1.txt","2-3-2.txt","2-3-3.txt","2-3-4.txt",
"2-4-1.txt","2-4-2.txt","2-4-3.txt","2-4-4.txt",

"1-1-20.txt","1-1-30.txt","1-1-75.txt",
"2-1-20.txt","2-1-30.txt","2-1-75.txt",
"2-2-20.txt","2-2-30.txt","2-2-75.txt",
"2-3-20.txt","2-3-30.txt","2-3-75.txt",
"2-4-20.txt","2-4-30.txt","2-4-75.txt",
]
_DEFAULT_CHECKPOINT = "checkpoints/Reference_checkpoint/class/case1-12_1-1-1_389c067dafdd144c65ba69c32c8a717488d767fe.pt"
_DEFAULT_REPORT = "results/Reference_ppo_outputs/class/unseen_summary.txt"


def _parse_args():
    p = argparse.ArgumentParser(description="Zero-shot evaluation on unseen Petri nets.")
    p.add_argument(
        "--checkpoint",
        default=os.environ.get("GCN_PPO_HQ_CHECKPOINT_PATH", _DEFAULT_CHECKPOINT),
        help="预训练模型权重的路径(pt 文件).",
    )
    p.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="要测试的网文件列表（缺省使用内置默认表）。",
    )
    p.add_argument(
        "--targets-file",
        default=None,
        help="可选：从文本文件读取目标列表（每行一个）。",
    )
    p.add_argument(
        "--out",
        default=os.environ.get("GCN_PPO_HQ_UNSEEN_OUTPUT", _DEFAULT_REPORT),
        help="评估报告输出路径。",
    )
    p.add_argument(
        "--no-snapshot",
        action="store_true",
        help="禁用 best_pool_snapshot;强制使用最终模型权重。",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="打印每个测试网的详细输出。",
    )
    p.add_argument(
        "--use-by-token",
        action="store_true",
        default=os.environ.get("GCN_PPO_HQ_USE_BY_TOKEN", "0").strip() == "1",
        help="启用 ByToken 模式构建 Petri 网（支持 q-time 约束）。"
        "默认关闭，与原有行为一致。也可通过环境变量 GCN_PPO_HQ_USE_BY_TOKEN=1 启用。",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    targets: List[str]
    if args.targets:
        targets = list(args.targets)
    elif args.targets_file:
        with open(args.targets_file, "r", encoding="utf-8") as f:
            targets = [line.strip() for line in f if line.strip()]
    else:
        targets = list(_DEFAULT_TARGETS)

    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), checkpoint_path)

    print(f"=== Zero-Shot Generalization Test ===", flush=True)
    print(f"Checkpoint : {checkpoint_path}", flush=True)
    print(f"Targets    : {len(targets)} 个", flush=True)
    print(f"Snapshot   : {'禁用' if args.no_snapshot else '优先使用 best_pool_snapshot'}", flush=True)
    print(f"ByToken    : {'启用（q-time 约束支持）' if args.use_by_token else '禁用（默认模式）'}", flush=True)

    details: List[Dict[str, object]] = []
    for tgt in targets:
        result = test_unseen_net(
            tgt,
            checkpoint_path,
            prefer_snapshot=not args.no_snapshot,
            verbose=args.verbose,
            use_by_token=args.use_by_token,
        )
        details.append(result)
        if not args.verbose:
            ms = result.get("makespan", -1)
            tag = "OK " if result.get("reach_goal") else "Fail"
            print(
                f"  [{tag}] {tgt:>16}  makespan={ms!s:>8}  trans={result.get('trans_count', 0):>3}  "
                f"elapsed={result.get('elapsed', -1):.3f}s  status={result.get('status')}",
                flush=True,
            )

    agg = aggregate(details)
    print("=== Aggregate ===", flush=True)
    print(
        f"  Total           : {agg['total']}\n"
        f"  Success         : {agg['success_count']} ({agg['success_rate'] * 100:.1f}%)\n"
        f"  Avg makespan    : {agg['avg_makespan']}\n"
        f"  Min/Max makespan: {agg['min_makespan']} / {agg['max_makespan']}\n"
        f"  Elapsed total   : {agg['elapsed_total']:.2f} s",
        flush=True,
    )

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_path)
    write_report(out_path, agg, details)
    print(f"=> 报告已写入 {out_path}", flush=True)


if __name__ == "__main__":
    main()
