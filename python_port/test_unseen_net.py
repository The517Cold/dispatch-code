import os
import sys
import time
import torch

# 将仓库根目录加入模块搜索路径
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from petri_net_io.utils.net_loader import load_petri_net_context, build_ttpn_with_residence
from petri_net_io.utils.checkpoint_selector import load_compatible_state

from train_ppo_3 import PetriNetGCNPPOProHQ


def _resolve_net_path(base_dir, test_file_name):
    candidates = [
        os.path.join(base_dir, "resources", test_file_name),
        os.path.join(base_dir, "resources/resources_new/test/family1", test_file_name),
        os.path.join(base_dir, "resources/resources_new/test/family2", test_file_name),
         os.path.join(base_dir, "resources/resources_new/resources", test_file_name),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]

def test_unseen_net(test_file_name, checkpoint_path):
    """
    零样本泛化测试：使用预训练的通用模型，直接在未见过的全新 Petri 网上进行推理。

    适配说明：
    - PetriNetGCNPPOProHQ 使用统一的 PetriNetGCNActorCritic 模型架构，
      其中 model.actor_net 为策略网络,model.value_head 为价值网络
    - 权重通过 load_compatible_state 加载到对应的子模块中
    - switch_environment 方法会自动初始化 best_records 等环境专属数据
    """
    base_dir = os.path.dirname(__file__)
    net_path = _resolve_net_path(base_dir, test_file_name)
    
    if not os.path.exists(net_path):
        print(f"ERROR: 找不到要测试的全新网文件: {net_path}")
        return
        
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: 找不到预训练的模型权重: {checkpoint_path}")
        return

    # print(f"=== [Zero-Shot Inference] ===")
    # print(f"Loading Unseen Net: {test_file_name}")
    # print(f"Loading Model Weights: {checkpoint_path}")

    # 1. 解析全新的 Petri 网文件
    context = load_petri_net_context(net_path)
    petri_net = build_ttpn_with_residence(context)
    
    # 构建当前全新网的环境字典（字段需与 switch_environment 方法所读取的键保持一致）
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

    # 2. 实例化搜索器
    # max_train_steps=0 禁止训练，仅推理
    search = PetriNetGCNPPOProHQ(
        petri_net=test_env["petri_net"],
        end=test_env["end"],
        pre=test_env["pre"],
        post=test_env["post"],
        min_delay_p=test_env["min_delay_p"],
        env_pool=[test_env],
        max_train_steps=0,
        verbose=False,
        search_strategy = "greedy",
        stochastic_num_rollouts=50,
        stochastic_temperature=1.3,
        beam_depth=400,
        beam_width=50,
        use_deadlock_controller=True,
    )

    # 3. 加载预训练权重到统一模型的子模块
    # PetriNetGCNPPOProHQ 使用 PetriNetGCNActorCritic 统一模型：
    #   - model.actor_net: 策略网络 (PetriNetGCNEnhanced)
    #   - model.value_head: 价值网络 (nn.Sequential)
    saved = torch.load(checkpoint_path, map_location="cpu")

    # 将权重移动到模型所在设备后加载
    actor_state = saved.get("actor_state", {})
    critic_state = saved.get("critic_state", {})
    device = search.device
    if actor_state:
        actor_state = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in actor_state.items()}
        load_compatible_state(search.model.actor_net, actor_state)
    if critic_state:
        critic_state = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in critic_state.items()}
        load_compatible_state(search.model.value_head, critic_state)

    # 标记为已训练状态，跳过 train_model 直接进入推理
    search.is_trained = True

    # 从 checkpoint 恢复 best_records（如果存在），否则由 switch_environment 已自动初始化
    if "best_records" in saved and saved["best_records"]:
        search.best_records = saved["best_records"]
    # 确保当前测试环境在 best_records 中有记录条目
    if test_env["name"] not in search.best_records:
        search.best_records[test_env["name"]] = {"makespan": 2 ** 31 - 1, "trans": []}

    # 4. 直接开始推理搜索
    start = time.perf_counter()
    result = search.search()
    elapsed = time.perf_counter() - start

    # 5. 打印最终结果
    extra = search.get_extra_info()
    trans = result.get_trans()
    markings = result.get_markings()
    
    t_map_v = getattr(context.get("matrix_translator"), "t_map_v", {})
    trans_names = [str(t_map_v.get(t, t)) for t in trans] if trans and t_map_v else [str(t) for t in trans]
    
    print(f"=={test_file_name}==")
    print(f"Elapsed Time    : {elapsed:.6f} s")
    print(f"Trans Count     : {len(trans_names)}")
    print(f"Makespan        : {markings[-1].get_prefix() if markings and len(trans) > 0 else -1}")
    print(f"Reach Goal      : {extra.get('reachGoal')}")
    print(f"Trans Sequence  : {' -> '.join(trans_names)}")

if __name__ == "__main__":
    # ==========================================
    # 在这里填写你要测试的全新网文件，以及你之前训练好的权重路径！
    # ==========================================
    
    # 1. 这是一个从来没放进过 train_files 列表的全新网
    TARGET_NEW_NET_POOL = ["1-2-13.txt","1-2-13-760.txt","1-2-13-650.txt","1-2-13-540.txt",
                           "1-1-13.txt","1-1-13-760.txt","1-1-13-650.txt","1-1-13-540.txt",
                            ] 
    
    # 2. 这是之前通过多网训练跑出来的 checkpoint 绝对/相对路径
    SAVED_CHECKPOINT_PATH = "d:\\dispatch_code\\BC+DAgger+PPO\\new_job\\python_port\\checkpoints\\Reference_checkpoint\\test\\test1-16_1-2-13-1_d04842c4e48fd179223ced3abe61d0eecd744ee6.pt" 
    
    for target_new_net in TARGET_NEW_NET_POOL:
        test_unseen_net(target_new_net, SAVED_CHECKPOINT_PATH)
