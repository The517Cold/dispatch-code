import random
import threading
from typing import Dict, List, Tuple, Any, Optional
import os
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    from .abstract_search import AbstractSearch
    from .petri_net_gcn_ppo import PetriNetGCNActorCritic
    from ..utils.result import Result
    from ..search.deadlock_controller import DeadlockController
    from ..search.rl_env_semantics import enabled_transitions_for_marking
    from ..representation.features import PetriRepresentationInput, PetriStateEncoderEnhanced
    from ...petri_net_io.utils.checkpoint_selector import load_compatible_state
except ImportError:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from petri_net_platform.search.abstract_search import AbstractSearch
    from petri_net_platform.search.petri_net_gcn_ppo import PetriNetGCNActorCritic
    from petri_net_platform.utils.result import Result
    from petri_net_platform.search.deadlock_controller import DeadlockController
    from petri_net_platform.search.rl_env_semantics import enabled_transitions_for_marking
    from petri_net_platform.representation.features import PetriRepresentationInput, PetriStateEncoderEnhanced
    from petri_net_io.utils.checkpoint_selector import load_compatible_state

"""
    这个文档的代码是多网训练但是没有加专家序列的版本
"""


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []  # 动作的对数概率
        self.rewards = []
        self.is_terminals = []
        self.values = []  # 状态价值
        self.masks = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]
        del self.masks[:]


class PetriNetGCNPPOPro(AbstractSearch):
    def __init__(
        self,
        petri_net,
        end: List[int],
        pre: List[List[int]],
        post: List[List[int]],
        min_delay_p: List[int],
        # GCN参数
        lambda_p: int = 256,  
        lambda_t: int = 64,  
        extra_p2t_rounds: int = 2,
        # 训练参数
        max_train_steps: int = 150000,  # 最大训练步数   
        steps_per_epoch: int = 4096,  # 每epoch收集步数  
        minibatch_size: int = 128,  # 小批量大小
        ppo_epochs: int = 10,  # PP更新轮数           
        lr: float = 1e-4,  
        # PPO核心参数
        gamma: float = 0.99,              
        gae_lambda: float = 0.95,         
        eps_clip: float = 0.2,            
        target_kl: float = 0.04,   # 目标KL散度       
        value_loss_coef: float = 0.5,     
        entropy_coef_start: float = 0.09,  # 初始熵
        entropy_coef_end: float = 0.01,  # 结束熵
        temperature_start: float = 2.0,   # 初始温度
        temperature_end: float = 1.1,  # 结束温度
        # 奖励参数   
        reward_goal_bonus: float = 300.0,  # 目标奖励奖励
        reward_deadlock_penalty: float = 100.0,  # 死锁奖励惩罚
        reward_progress_weight: float = 2.0,  # 进度奖励权重
        reward_repeat_penalty: float = 0.2,  # 重复奖励惩罚
        reward_time_scale: float = 1000.0,  # 时间奖励缩放
        reward_residence_warn_ratio: float = 0.7,  # 驻留时间警告阈值占比（超过此比例开始渐进惩罚）
        reward_residence_penalty_max: float = 30.0,  # 驻留时间超限最大惩罚
        reward_residence_safe_bonus: float = 0.5,  # 驻留时间安全时的小额奖励
        reward_mobility_weight: float = 0.3,  # 动作空间收缩惩罚权重
        beam_width: int = 100,  # 搜索宽度
        beam_depth: int = 800,  # 搜索深度
        search_strategy: str = "beam",  # 搜索策略: "beam"、"greedy" 或 "stochastic"
        stochastic_num_rollouts: int = 50,  # 随机采样策略的轨迹数量
        stochastic_temperature: float = 1.2,  # 随机采样策略的温度系数
        mixed_rollout: bool = True,  # 是否启用混合经验收集模式
        envs_per_epoch: int = 0,  # 混合模式下每epoch采样的环境数量，0表示使用全部环境
        cross_env_gae: bool = True,  # 是否启用跨环境GAE标准化
        async_collection: bool = True,  # 是否启用异步环境采样
        dynamic_curriculum: bool = True,  # 是否启用基于课程学习的动态混合收集策略
        curriculum_warmup_ratio: float = 0.3,  # 课程学习预热阶段占比（0~1）
        use_deadlock_controller: bool = True,  # 是否启用死锁控制器；False 时使用 Petri 网 enable 函数判断变迁使能
        verbose: bool = True,  # 是否打印训练信息
        device: str = None,

        # 【新增】: 接收 max_residence_time
        max_residence_time: List[int] = None, 
        env_pool=None,
        eval_env_pool=None,     # 独立的评估池（不参与训练，仅用于训练中监控泛化能力）
        eval_pool_interval: int = 0,  # 每隔多少 epoch 评估一次 eval_env_pool（0 = 不评估）
        weight_decay: float = 0.0,   # Adam L2 正则化系数，有助于减少过拟合
        **kwargs
    ):
        super().__init__()
        self.petri_net = petri_net
        self.initial_petri_net = petri_net.clone()
        self.env_pool = env_pool
        self.end = end
        self.pre = pre
        self.post = post
         # 兼容性处理：如果没有传入 max_residence_time，则给无限大
        if max_residence_time is None:
            self.max_residence_time = [2**31 - 1] * len(min_delay_p)
        else:
            self.max_residence_time = max_residence_time
        self.capacity = getattr(petri_net, "capacity", None)
        self.has_capacity = bool(getattr(petri_net, "has_capacity", False)) and self.capacity is not None
        self.transition_flow_allowed = getattr(petri_net, "transition_flow_allowed", [True] * len(pre[0]))
        
        self.max_train_steps = max_train_steps
        self.steps_per_epoch = steps_per_epoch
        self.minibatch_size = minibatch_size
        self.ppo_epochs = ppo_epochs
        self.initial_lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.target_kl = target_kl
        self.value_loss_coef = value_loss_coef
        
        self.entropy_coef_start = entropy_coef_start
        self.entropy_coef_end = entropy_coef_end
        self.current_entropy_coef = entropy_coef_start
        
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.current_temperature = temperature_start
        
        self.reward_goal_bonus = reward_goal_bonus
        self.reward_deadlock_penalty = reward_deadlock_penalty
        self.reward_progress_weight = reward_progress_weight
        self.reward_repeat_penalty = reward_repeat_penalty
        self.reward_time_scale = reward_time_scale
        self.reward_residence_warn_ratio = reward_residence_warn_ratio
        self.reward_residence_penalty_max = reward_residence_penalty_max
        self.reward_residence_safe_bonus = reward_residence_safe_bonus
        self.reward_mobility_weight = reward_mobility_weight
        
        self.beam_width = beam_width
        self.beam_depth = beam_depth
        self.search_strategy = search_strategy
        self.stochastic_num_rollouts = stochastic_num_rollouts
        self.stochastic_temperature = stochastic_temperature
        self.mixed_rollout = mixed_rollout
        self.envs_per_epoch = envs_per_epoch
        self.cross_env_gae = cross_env_gae
        self.async_collection = async_collection
        self.dynamic_curriculum = dynamic_curriculum
        self.curriculum_warmup_ratio = curriculum_warmup_ratio
        self.use_deadlock_controller = use_deadlock_controller
        self.verbose = verbose
        self.lambda_p = lambda_p
        self.lambda_t = lambda_t
        self.extra_p2t_rounds = extra_p2t_rounds
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.encoder = PetriStateEncoderEnhanced(
            end=end,
            min_delay_p=min_delay_p,
            max_residence_time=self.max_residence_time,
            capacity=self.capacity,
            device=self.device,
        )
        self.model = PetriNetGCNActorCritic(
            pre,
            post,
            lambda_p,
            lambda_t,
            extra_p2t_rounds,
            end=end,
            min_delay_p=min_delay_p,
            min_delay_t=getattr(petri_net, "min_delay_t", None),
            capacity=self.capacity,
            max_residence_time=self.max_residence_time,
            place_from_places=getattr(petri_net, "place_from_places", None),
        ).to(self.device)
        self._weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.buffer = RolloutBuffer()
        
        self.is_trained = False
        self.best_train_trans: List[int] = []
        self.best_train_makespan = 2 ** 31 - 1
        self.extra_info: Dict[str, float] = {}
        self.best_records = {}
        self.current_env_name = "default"
        self.best_records["default"] = {"makespan": 2**31 - 1, "trans": []}

        # 独立评估池（结构相似但未见过的网络，用于训练中监控泛化能力）
        self.eval_env_pool = eval_env_pool
        self.eval_pool_interval = eval_pool_interval
        # 最优模型快照（在所有训练网 pool_success_rate 最高时保存）
        self._best_pool_score = -1.0
        self._best_snapshot: Optional[Dict] = None
        
        self.deadlock_controller = None
        if use_deadlock_controller:
            self.deadlock_controller = DeadlockController(
                pre=pre,
                post=post,
                end=end,
                capacity=self.capacity,
                has_capacity=self.has_capacity,
                transition_flow_allowed=self.transition_flow_allowed,
                controller_name="gcn_ppo_pro",
                enable_lookahead=kwargs.get("controller_enable_lookahead", True),
                lookahead_depth=kwargs.get("controller_lookahead_depth", 2),
                lookahead_width=kwargs.get("controller_lookahead_width", 4),
            )
        
        self.pool_eval_interval = max(1, int(kwargs.get("pool_eval_interval", 4)))
        self.curriculum_epochs = max(0, int(kwargs.get("curriculum_epochs", 8)))
        self.mask_cache_limit = max(0, int(kwargs.get("mask_cache_limit", 40000)))
        self.env_visit_counts = {}
        self._mask_cache = {}
        if self.env_pool:
            for env in self.env_pool:
                self.env_visit_counts[env.get("name", "default")] = 0

    def _env_complexity_value(self, env_dict) -> float:
        # 计算环境的复杂度值
        # 复杂度值 = max(状态数, 转换数) + 0.5 * 有约束(驻留时间约束)转换数
        place_count = len(env_dict.get("pre", []))
        trans_count = len(env_dict.get("pre", [[[]]])[0]) if env_dict.get("pre") else 0
        constrained_count = sum(1 for val in env_dict.get("max_residence_time", []) if val < 2 ** 31 - 1)
        return float(max(place_count, trans_count) + constrained_count * 0.5)

    def _get_env_by_name(self, env_name: str):
        # 根据环境名称获取环境字典
        # 如果环境池为空，返回None
        if not self.env_pool:
            return None
        for env in self.env_pool:
            if env.get("name") == env_name:
                return env
        return None

    def _select_training_env(self, epoch_idx: int):
        if not self.env_pool or len(self.env_pool) <= 1:
            return None
        # 按复杂度排序(从简单到复杂)
        ordered_envs = sorted(self.env_pool, key=self._env_complexity_value)
        # 预热阶段,按顺序轮询各环境,就是现在简单环境上学习然后在迁移到复杂环境
        warmup_epochs = max(len(ordered_envs), self.curriculum_epochs)
        if epoch_idx <= warmup_epochs:
            return ordered_envs[(epoch_idx - 1) % len(ordered_envs)]
        # 正式训练阶段:智能选择:访问少的优先,复杂环境额外加分
        max_complexity = max(self._env_complexity_value(env) for env in ordered_envs) or 1.0
        min_visit = min(self.env_visit_counts.get(env.get("name", "default"), 0) for env in ordered_envs)
        best_score = None
        candidates = []
        for env in ordered_envs:
            env_name = env.get("name", "default")
            visits = self.env_visit_counts.get(env_name, 0)
            # 访问少的优先
            coverage_bonus = (min_visit + 1.0) / (visits + 1.0)
            # 复杂环境额外加分
            difficulty_bonus = self._env_complexity_value(env) / max_complexity
            # 综合评分
            score = coverage_bonus + difficulty_bonus
            if best_score is None or score > best_score + 1e-8:
                best_score = score
                candidates = [env]
            elif abs(score - best_score) <= 1e-8:
                candidates.append(env)
        return random.choice(candidates)

    def _compute_priority_sampling_weights(self, progress: float) -> Dict[str, float]:
        """
        基于课程学习的动态优先级采样权重计算

        综合考虑三个因素：
        1. 复杂度权重：训练初期偏向低复杂度，后期偏向高复杂度
        2. 困难度权重：评估表现差的环境获得更高权重
        3. 覆盖度权重：访问少的环境获得更高权重

        Args:
            progress: 训练进度 [0, 1]

        Returns:
            Dict[str, float]: 环境名称到采样权重的映射
        """
        if not self.env_pool:
            return {}

        weights = {}
        max_complexity = max(self._env_complexity_value(env) for env in self.env_pool) or 1.0
        min_visit = min(self.env_visit_counts.get(env.get("name", "default"), 0) for env in self.env_pool)

        for env in self.env_pool:
            env_name = env.get("name", "default")
            complexity = self._env_complexity_value(env)

            # 1. 复杂度权重：初期偏向低复杂度，后期偏向高复杂度
            # 使用 Sigmoid 函数实现平滑过渡
            complexity_ratio = complexity / max_complexity
            if progress < self.curriculum_warmup_ratio:
                # 预热阶段：低复杂度优先
                complexity_weight = 1.0 - complexity_ratio * 0.7
            else:
                # 正式阶段：高复杂度优先
                adjusted_progress = (progress - self.curriculum_warmup_ratio) / (1.0 - self.curriculum_warmup_ratio)
                complexity_weight = 0.3 + 0.7 * complexity_ratio * adjusted_progress

            # 2. 困难度权重：评估表现差的环境获得更高权重
            # 改进：不再只区分"到达/未到达"，而是基于相对最优解的改进空间动态计算
            env_record = self.best_records.get(env_name, {})
            env_makespan = env_record.get("makespan", 2**31 - 1)
            reached_goal = env_makespan < 2**31 - 1

            if reached_goal:
                # 已到达目标：根据与全局最优的相对差距计算权重
                # 收集所有已到达目标的环境的 makespan，用于相对比较
                all_makespans = [
                    self.best_records.get(e.get("name", "default"), {}).get("makespan", 2**31 - 1)
                    for e in self.env_pool
                    if self.best_records.get(e.get("name", "default"), {}).get("makespan", 2**31 - 1) < 2**31 - 1
                ]
                if len(all_makespans) > 1:
                    best_ms = min(all_makespans)
                    worst_ms = max(all_makespans)
                    ms_range = max(1.0, float(worst_ms - best_ms))
                    # makespan 越差（越大），权重越高；归一化到 [0.5, 1.5]
                    rel_difficulty = (env_makespan - best_ms) / ms_range
                    difficulty_weight = 0.5 + rel_difficulty
                else:
                    difficulty_weight = 0.7  # 全部已达目标且无差异时，轻度降权
            else:
                # 未到达目标：高优先级（训练后期加大惩罚以集中攻克难关）
                difficulty_weight = 2.0 + 0.5 * progress  # 随训练深入逐渐加权

            # 3. 覆盖度权重：访问少的环境获得更高权重
            visits = self.env_visit_counts.get(env_name, 0)
            coverage_weight = (min_visit + 1.0) / (visits + 1.0)

            # 综合权重
            total_weight = complexity_weight * difficulty_weight * coverage_weight
            weights[env_name] = max(total_weight, 0.01)

        return weights

    def _priority_sample_envs(self, n_sample: int, progress: float) -> List:
        """
        基于优先级权重的环境采样

        Args:
            n_sample: 需要采样的环境数量
            progress: 训练进度 [0, 1]

        Returns:
            List: 采样得到的环境字典列表
        """
        if not self.env_pool:
            return []

        weights = self._compute_priority_sampling_weights(progress)
        env_names = list(weights.keys())
        env_weights = [weights[name] for name in env_names]
        total = sum(env_weights)
        env_probs = [w / total for w in env_weights]

        n_sample = min(n_sample, len(self.env_pool))

        if n_sample >= len(self.env_pool):
            # 使用全部环境，但按优先级排序
            sorted_envs = sorted(
                self.env_pool,
                key=lambda e: weights.get(e.get("name", "default"), 0),
                reverse=True
            )
            return sorted_envs

        # 按概率采样（无放回）
        sampled_names = set()
        sampled_envs = []
        remaining_probs = list(env_probs)

        for _ in range(n_sample):
            remaining_names = [n for n in env_names if n not in sampled_names]
            remaining_p = [remaining_probs[i] for i, n in enumerate(env_names) if n not in sampled_names]

            if not remaining_names:
                break

            p_sum = sum(remaining_p)
            if p_sum > 0:
                remaining_p = [p / p_sum for p in remaining_p]

            chosen_name = random.choices(remaining_names, weights=remaining_p, k=1)[0]
            sampled_names.add(chosen_name)

            env_dict = self._get_env_by_name(chosen_name)
            if env_dict is not None:
                sampled_envs.append(env_dict)

        return sampled_envs

    def _mask_cache_key(self, marking) -> Tuple:
        """
            为 Petri 网标识生成一个 可哈希的缓存键 ，用于缓存动作掩码计算结果。
            计算动作掩码 _mask_from_marking 需要遍历所有库所和变迁，开销较大。如果同一个状态重复出现，可以直接从缓存中获取掩码。
            缓存键包含的信息：
            1. 状态的库所信息 (p_info)-----类型: Tuple[int, ...]
            2. 状态的前缀 (prefix)-----类型: int
            3. 状态是否超最大驻留时间约束 (over_max_residence_time)-----类型: bool
            4. 状态的定时变迁信息 (t_info)-----类型: Tuple[Tuple[int, ...], ...]
            5. 状态的驻留时间信息 (residence_time_info)-----类型: Tuple[Tuple[int, ...], ...]
        """
        timed_info = tuple(tuple(int(v) for v in place_tokens) for place_tokens in getattr(marking, "t_info", []))
        residence_info = tuple(tuple(int(v) for v in place_tokens) for place_tokens in getattr(marking, "residence_time_info", []))
        return (
            tuple(int(v) for v in marking.get_p_info()),
            int(marking.get_prefix()),
            bool(getattr(marking, "over_max_residence_time", False)),
            timed_info,
            residence_info,
        )

    def switch_environment(self, env_dict):
        """
        动态切换训练用的 Petri 网，并杜绝状态污染。

        切换时始终将当前共享权重迁移到新拓扑，保证所有环境共用同一条策略
        参数轨迹，从而支持真正意义上的多网泛化训练。

        模型的可学习参数（lambda_p / lambda_t 维度空间）与网络拓扑尺寸（库所数 P、
        变迁数 T）完全解耦：P/T 相关的 pre/post 矩阵以 buffer 形式存储，
        load_compatible_state 会跳过形状不匹配的 buffer 并仅恢复形状匹配的参数，
        因此权重在不同拓扑间迁移是安全的。

        Args:
            env_dict: 环境字典，包含 petri_net/end/pre/post/min_delay_p 等字段
        """
        env_name = env_dict.get("name", "default")

        self.petri_net = env_dict["petri_net"]

        if "initial_marking" in env_dict:
            self.petri_net.set_marking(env_dict["initial_marking"].clone())

        self.initial_petri_net = self.petri_net.clone()
        self.end = env_dict["end"]
        self.pre = env_dict["pre"]
        self.post = env_dict["post"]
        self.min_delay_p = env_dict["min_delay_p"]
        self.max_residence_time = env_dict.get("max_residence_time", [2**31-1] * len(self.end))
        self.capacity = getattr(self.petri_net, "capacity", None)
        self.has_capacity = bool(getattr(self.petri_net, "has_capacity", False)) and self.capacity is not None
        self.transition_flow_allowed = getattr(self.petri_net, "transition_flow_allowed", [True] * len(self.pre[0]))

        if hasattr(self, "encoder"):
            self.encoder = PetriStateEncoderEnhanced(
                end=self.end,
                min_delay_p=self.min_delay_p,
                device=self.device,
                pre=self.pre,
                post=self.post,
                min_delay_t=getattr(self.petri_net, "min_delay_t", None),
                capacity=self.capacity,
                max_residence_time=self.max_residence_time,
                place_from_places=getattr(self.petri_net, "place_from_places", None),
            )

        if hasattr(self, "model"):
            # 始终迁移当前共享权重到新拓扑——这是多网泛化的核心保证。
            # 可学习参数全部在 lambda_p/lambda_t 空间，与 P/T 规模无关；
            # 拓扑相关 buffer（pre/post/degree/transition_seed）随新模型初始化自动更新。
            old_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            # 保存参数形状用于优化器状态迁移的兼容性检查
            old_param_shapes = {n: tuple(p.shape) for n, p in self.model.named_parameters()}

            # 保存优化器当前状态（动量、方差估计等），用于跨环境迁移
            saved_opt_state: Optional[dict] = None
            current_lr = self.initial_lr
            if hasattr(self, "optimizer") and self.optimizer.param_groups:
                try:
                    saved_opt_state = self.optimizer.state_dict()
                    current_lr = self.optimizer.param_groups[0]["lr"]
                except Exception:
                    pass

            self.model = PetriNetGCNActorCritic(
                self.pre,
                self.post,
                self.lambda_p,
                self.lambda_t,
                self.extra_p2t_rounds,
                end=self.end,
                min_delay_p=self.min_delay_p,
                min_delay_t=getattr(self.petri_net, "min_delay_t", None),
                capacity=self.capacity,
                max_residence_time=self.max_residence_time,
                place_from_places=getattr(self.petri_net, "place_from_places", None),
            ).to(self.device)
            load_compatible_state(self.model, old_state)

            # ★ 关键修复：重建优化器以绑定新模型参数。
            # 原 self.optimizer 仍指向旧模型参数，switch 后 optimizer.step() 只会更新
            # 已被丢弃的旧参数，新模型实际上从未被训练——这是多网训练失效的根本原因。
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=current_lr,
                weight_decay=getattr(self, "_weight_decay", 0.0),
            )
            # 若参数形状未变（lambda_p/lambda_t 固定时通常满足），迁移优化器动量状态，
            # 避免每次切换环境都从零开始积累梯度统计量。
            if saved_opt_state is not None:
                new_param_shapes = {n: tuple(p.shape) for n, p in self.model.named_parameters()}
                if new_param_shapes == old_param_shapes:
                    try:
                        self.optimizer.load_state_dict(saved_opt_state)
                        # 确保 lr 保持为切换前的当前值（调度器可能已调整过）
                        for pg in self.optimizer.param_groups:
                            pg["lr"] = current_lr
                    except Exception:
                        pass  # 迁移失败时使用全新优化器状态，不影响正确性
        
        if hasattr(self, "deadlock_controller") and self.use_deadlock_controller:
            self.deadlock_controller = DeadlockController(
                pre=self.pre,
                post=self.post,
                end=self.end,
                capacity=self.capacity,
                has_capacity=self.has_capacity,
                transition_flow_allowed=self.transition_flow_allowed,
                controller_name="gcn_ppo_pro",
                enable_lookahead=True,
                lookahead_depth=2,
                lookahead_width=4,
            )
            
        self.current_env_name = env_name
        self._mask_cache = {}
        if self.current_env_name not in self.env_visit_counts:
            self.env_visit_counts[self.current_env_name] = 0
        
        if self.current_env_name not in self.best_records:
            self.best_records[self.current_env_name] = {
                "makespan": 2**31 - 1, 
                "trans": []
            }

    def _log(self, text: str):
        if self.verbose:
            print(text, flush=True)

    def _set_to_initial(self):
        self.petri_net = self.initial_petri_net.clone()

    def _is_goal(self, marking) -> bool:
        p_info = marking.get_p_info()
        for i in range(len(p_info)):
            if self.end[i] == -1: continue
            if p_info[i] != self.end[i]: return False
        return True

    def _goal_distance(self, marking) -> int:
        p_info = marking.get_p_info()
        dist = 0
        for i in range(len(p_info)):
            if self.end[i] == -1: continue
            dist += abs(p_info[i] - self.end[i])
        return dist

    def _state_key(self, marking) -> Tuple:
        over = bool(getattr(marking, "over_max_residence_time", False))
        return tuple(marking.get_p_info()) + (1 if over else 0,)

    def _mask_from_marking(self, marking) -> torch.Tensor:
        trans_count = len(self.pre[0])
        
        if bool(getattr(marking, "over_max_residence_time", False)):
            return torch.tensor([False] * trans_count, dtype=torch.bool, device=self.device)
        
        cache_key = self._mask_cache_key(marking)
        cached_mask = self._mask_cache.get(cache_key)
        if cached_mask is not None:
            return cached_mask.clone()
        
        if self.use_deadlock_controller and self.deadlock_controller is not None:
            analysis = self.deadlock_controller.analyze_state(self.petri_net, marking)
            controller_actions = analysis.controller_actions
            mask = torch.zeros(trans_count, dtype=torch.bool, device=self.device)
            if controller_actions:
                mask[controller_actions] = True
        else:
            # 禁用死锁控制器时，使用 Petri 网底层的 enable 函数判断变迁使能
            self.petri_net.set_marking(marking)
            mask = torch.zeros(trans_count, dtype=torch.bool, device=self.device)
            for t in range(trans_count):
                if self.petri_net.enable(t):
                    mask[t] = True
        
        if self.mask_cache_limit > 0:
            if len(self._mask_cache) >= self.mask_cache_limit:
                self._mask_cache.pop(next(iter(self._mask_cache)))
            self._mask_cache[cache_key] = mask.clone()
        return mask

    def _compute_residence_reward(self, marking) -> float:
        """
        计算驻留时间奖惩信号。

        基于当前 marking 中每个库所内 token 的驻留时间与该库所的
        max_residence_time 阈值（来自底层 Petri 网文件），产生三档信号：
          1. 安全区（ratio < warn_ratio）：给予小额正向奖励鼓励正常推进
          2. 警告区（warn_ratio <= ratio < 1.0）：渐进式惩罚，ratio 越接近 1.0 越重
          3. 超限（ratio >= 1.0 或 over_max_residence_time）：施加最大惩罚

        不同的网文件拥有不同的 max_residence_time 向量，因此阈值天然适配各环境。
        """
        if not hasattr(marking, "residence_time_info"):
            return self.reward_residence_safe_bonus

        if bool(getattr(marking, "over_max_residence_time", False)):
            return -self.reward_residence_penalty_max

        worst_ratio = 0.0
        for place_idx, residence_deque in enumerate(marking.residence_time_info):
            limit = self.max_residence_time[place_idx]
            if limit >= 2 ** 31 - 1 or limit <= 0:
                continue
            for token_residence in residence_deque:
                ratio = float(token_residence) / float(limit)
                worst_ratio = max(worst_ratio, ratio)

        warn = self.reward_residence_warn_ratio
        if worst_ratio < warn:
            return self.reward_residence_safe_bonus
        elif worst_ratio < 1.0:
            severity = (worst_ratio - warn) / (1.0 - warn)
            return -self.reward_residence_penalty_max * (severity ** 2)
        else:
            return -self.reward_residence_penalty_max

    def _step_env(self, curr_marking, action: int, seen_count: dict) -> Tuple[any, float, bool, bool]:
        """
        执行环境一步，返回 (下一状态, 奖励, 是否结束, 是否死锁)。

        奖励信号由四个正交分量叠加组成（步骤奖励不参与目标 bonus 的裁剪）：
          1. 进度/时间/重复 → step_reward（限幅防梯度爆炸）
          2. 驻留时间奖惩 → residence_reward
          3. 动作空间收缩惩罚 → mobility_penalty（接近死锁的早期警告）
          4. 目标完成奖励 → goal_bonus（独立叠加，不被限幅截断）
        """
        if action < 0 or action >= self.petri_net.get_trans_count():
            return curr_marking, -self.reward_deadlock_penalty, True, True

        mask = self._mask_from_marking(curr_marking)
        if not mask[action].item():
            return curr_marking, -self.reward_deadlock_penalty, True, True

        curr_enabled_count = int(mask.sum().item())

        next_marking = self.petri_net.launch(action)
        self.petri_net.set_marking(next_marking)

        done = self._is_goal(next_marking)
        next_mask = self._mask_from_marking(next_marking)
        deadlock = not bool(next_mask.any().item()) and (not done)

        # ── 1. 基础步骤奖励 ──
        delta_t = float(next_marking.get_prefix() - curr_marking.get_prefix())
        progress = float(self._goal_distance(curr_marking) - self._goal_distance(next_marking))
        time_cost = delta_t / self.reward_time_scale

        visit_count = seen_count.get(self._state_key(next_marking), 0)
        repeat_penalty = min(visit_count, 10) * self.reward_repeat_penalty

        step_reward = -time_cost + self.reward_progress_weight * progress - repeat_penalty

        if deadlock:
            step_reward -= self.reward_deadlock_penalty

        clip_bound = self.reward_deadlock_penalty
        step_reward = max(-clip_bound, min(clip_bound * 0.5, step_reward))

        # ── 2. 驻留时间奖惩（阈值来自 self.max_residence_time，随网文件变化） ──
        residence_reward = self._compute_residence_reward(next_marking)

        # ── 3. 动作空间收缩惩罚（提前感知趋向死锁的状态） ──
        mobility_penalty = 0.0
        if not deadlock and not done:
            next_enabled_count = int(next_mask.sum().item())
            trans_count = len(self.pre[0])
            if next_enabled_count <= max(2, trans_count // 8):
                mobility_penalty = self.reward_mobility_weight * (
                    1.0 - next_enabled_count / max(1, curr_enabled_count)
                )

        # ── 4. 目标完成奖励（独立叠加，不受 step_reward 限幅影响） ──
        goal_bonus = 0.0
        if done:
            makespan = next_marking.get_prefix()
            env_best = self.best_records[self.current_env_name]["makespan"]

            if env_best == 2 ** 31 - 1:
                goal_bonus = self.reward_goal_bonus
                self._log(f"    [Goal] [{self.current_env_name}] First Goal Reached! Makespan: {makespan}")
            else:
                improvement = env_best - makespan
                if improvement > 0:
                    improvement_ratio = float(improvement) / self.reward_time_scale
                    extra_bonus = improvement_ratio * 80.0
                    goal_bonus = self.reward_goal_bonus + min(self.reward_goal_bonus, extra_bonus)
                    self._log(f"    [Goal] [{self.current_env_name}] New Best! Makespan: {makespan} (improved by {improvement})")
                else:
                    degradation = makespan - env_best
                    degradation_ratio = float(degradation) / self.reward_time_scale
                    penalty_for_worse = degradation_ratio * 80.0
                    goal_bonus = max(self.reward_goal_bonus * 0.1, self.reward_goal_bonus - penalty_for_worse)

        reward = step_reward + residence_reward - mobility_penalty + goal_bonus
        return next_marking, reward, done, deadlock

    def _collect_rollouts(self, num_steps: int):
        """
            收集经验回放
            :param num_steps: 收集的步数
        """

        self._set_to_initial()
        curr_marking = self.petri_net.get_marking()
        seen_count = {}  # 记录的是每一个标识出现的次数
        
        ep_rewards = []  # 存储成功或死锁的回合的累计奖励
        ep_makespans = []  # 存储成功到达目标的回合的完工时间
        current_ep_reward = 0.0  # 当前回合的累计奖励
        current_ep_trans = []  # 当前回合执行的变迁序列
        
        steps_collected = 0  # 已收集的步数
        
        self.model.eval()
        
        while steps_collected < num_steps:
            encoded = self.encoder.encode(curr_marking)
            mask = self._mask_from_marking(curr_marking)
            s_key = self._state_key(curr_marking)
            seen_count[s_key] = seen_count.get(s_key, 0) + 1

            with torch.no_grad():
                logits, value = self.model(encoded.unsqueeze(0)) if not isinstance(encoded, PetriRepresentationInput) else self.model(encoded)
                logits = logits.squeeze(0)
                value = value.squeeze(0).item() if value.dim() > 0 else value.item()
                logits[~mask] = -1e9 
                
                if not mask.any():
                    action = -1
                    action_logprob = 0.0
                else:
                    # 采样使用温度以增加探索多样性，
                    # 但记录 logprob 时统一使用 T=1.0 的原始 logits，
                    # 确保与 _update_ppo 中计算新 logprob 的分布一致，
                    # 从而保证 PPO 重要性采样比率 (π_new / π_old) 的正确性。
                    scaled_logits = logits / self.current_temperature
                    action_tensor = Categorical(logits=scaled_logits).sample()
                    action = action_tensor.item()
                    action_logprob = Categorical(logits=logits).log_prob(action_tensor).item()

            next_marking, reward, done, deadlock = self._step_env(curr_marking, action, seen_count)  # 执行动作,获取下一个状态,奖励,是否到达目标状态,是否死锁
            
            # 向buffer存经验
            self.buffer.states.append(encoded.place_features if isinstance(encoded, PetriRepresentationInput) else encoded)
            self.buffer.actions.append(action if action >= 0 else 0)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.values.append(value)
            self.buffer.rewards.append(reward)
            self.buffer.is_terminals.append(done or deadlock)
            self.buffer.masks.append(mask)
            
            current_ep_reward += reward
            if action >= 0:
                current_ep_trans.append(action)
            
            curr_marking = next_marking
            steps_collected += 1
            
            if done or deadlock:
                ep_rewards.append(current_ep_reward)
                if done:
                    makespan = curr_marking.get_prefix()
                    ep_makespans.append(makespan)
                    
                    env_record = self.best_records[self.current_env_name]
                    if makespan < env_record["makespan"]:
                        # 更新最优记录
                        env_record["makespan"] = makespan
                        env_record["trans"] = current_ep_trans.copy()
                        self.best_train_makespan = makespan
                        self.best_train_trans = current_ep_trans.copy()
                
                # 重置环境
                self._set_to_initial()
                curr_marking = self.petri_net.get_marking()
                seen_count = {}
                current_ep_reward = 0.0
                current_ep_trans = []

        return steps_collected, ep_rewards, ep_makespans

    def _collect_mixed_rollouts(self, total_steps: int, progress: float = 0.0):
        """
        混合经验收集模式：从多个环境中各收集一部分经验

        支持三种优化：
        1. 跨环境GAE标准化:将所有环境的优势值统一标准化
        2. 异步环境采样：使用线程池预加载环境组件
        3. 动态课程学习:训练初期保证完整轨迹(策略B),后期混合收集(策略A)

        动态比例调整策略：
        - 预热阶段(progress < curriculum_warmup_ratio):
          策略B变体,每个环境收集完整轨迹,不截断GAE
        - 正式阶段(progress >= curriculum_warmup_ratio):
          策略A,多环境混合收集,步数均分

        Args:
            total_steps: 本 epoch 总共需要收集的步数
            progress: 训练进度 [0, 1]

        Returns:
            total_collected: 实际收集的总步数
            all_ep_rewards: 所有环境的回合奖励列表
            all_ep_makespans: 所有环境的回合 makespan 列表
        """
        if not self.env_pool or len(self.env_pool) == 0:
            return self._collect_rollouts(total_steps)

        env_pool = self.env_pool
        n_envs = len(env_pool)

        n_sample = self.envs_per_epoch if self.envs_per_epoch > 0 else n_envs
        n_sample = min(n_sample, n_envs)

        # 动态课程学习：基于优先级采样选择环境
        if self.dynamic_curriculum:
            sampled_envs = self._priority_sample_envs(n_sample, progress)
        else:
            if n_sample >= n_envs:
                sampled_envs = list(env_pool)
            else:
                min_visit = min(self.env_visit_counts.get(env.get("name", "default"), 0) for env in env_pool)
                scored = []
                for env in env_pool:
                    visits = self.env_visit_counts.get(env.get("name", "default"), 0)
                    coverage_bonus = (min_visit + 1.0) / (visits + 1.0)
                    difficulty_bonus = self._env_complexity_value(env) / max(self._env_complexity_value(e) for e in env_pool)
                    scored.append((env, coverage_bonus + difficulty_bonus))
                scored.sort(key=lambda x: x[1], reverse=True)
                sampled_envs = [s[0] for s in scored[:n_sample]]

        # 动态比例调整：预热阶段使用完整轨迹收集，正式阶段使用步数均分
        in_warmup = self.dynamic_curriculum and progress < self.curriculum_warmup_ratio

        if in_warmup:
            steps_per_env = total_steps
        else:
            steps_per_env = max(1, total_steps // len(sampled_envs))

        total_collected = 0
        all_ep_rewards = []
        all_ep_makespans = []
        total_a_loss, total_c_loss, total_updates = 0.0, 0.0, 0

        saved_env_name = self.current_env_name

        # 阶段1：收集所有环境的经验
        env_experiences = []

        if self.async_collection and len(sampled_envs) > 1:
            env_experiences, async_collected, async_rewards, async_makespans = self._async_collect(sampled_envs, steps_per_env, total_steps)
            total_collected += async_collected
            all_ep_rewards.extend(async_rewards)
            all_ep_makespans.extend(async_makespans)
        else:
            # 顺序收集
            for env in sampled_envs:
                self.switch_environment(env)
                self.env_visit_counts[self.current_env_name] = self.env_visit_counts.get(self.current_env_name, 0) + 1

                remaining = total_steps - total_collected
                steps_for_this = min(steps_per_env, remaining) if env != sampled_envs[-1] else remaining
                if steps_for_this <= 0:
                    steps_for_this = steps_per_env

                self.buffer.clear()
                collected, ep_rewards, ep_makespans = self._collect_rollouts(steps_for_this)
                total_collected += collected
                all_ep_rewards.extend(ep_rewards)
                all_ep_makespans.extend(ep_makespans)

                if self.buffer.states:
                    old_values = torch.tensor(self.buffer.values, dtype=torch.float32)
                    raw_advantages, raw_returns = self._compute_gae(
                        self.buffer.rewards, old_values.tolist(), self.buffer.is_terminals
                    )
                    env_experiences.append({
                        "env_name": self.current_env_name,
                        "buffer_states": list(self.buffer.states),
                        "buffer_actions": list(self.buffer.actions),
                        "buffer_logprobs": list(self.buffer.logprobs),
                        "buffer_masks": list(self.buffer.masks),
                        "raw_advantages": raw_advantages,
                        "raw_returns": raw_returns,
                        "old_values": old_values.tolist(),
                    })
                self.buffer.clear()

        # 阶段2：跨环境GAE标准化
        if self.cross_env_gae and env_experiences:
            all_raw_advantages = []
            for exp in env_experiences:
                all_raw_advantages.extend(exp["raw_advantages"])

            global_advantages = torch.tensor(all_raw_advantages, dtype=torch.float32)
            global_mean = global_advantages.mean()
            global_std = global_advantages.std() + 1e-8

            for exp in env_experiences:
                n = len(exp["raw_advantages"])
                exp["normalized_advantages"] = (
                    torch.tensor(exp["raw_advantages"], dtype=torch.float32) - global_mean
                ) / global_std
        else:
            for exp in env_experiences:
                adv = torch.tensor(exp["raw_advantages"], dtype=torch.float32)
                exp["normalized_advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        # 阶段3：使用标准化后的优势值更新 PPO
        for exp in env_experiences:
            env_name = exp["env_name"]
            env_dict = self._get_env_by_name(env_name)
            if env_dict is not None:
                self.switch_environment(env_dict)

            # 将保存的经验加载回 buffer
            self.buffer.states = exp["buffer_states"]
            self.buffer.actions = exp["buffer_actions"]
            self.buffer.logprobs = exp["buffer_logprobs"]
            self.buffer.masks = exp["buffer_masks"]
            self.buffer.rewards = []
            self.buffer.is_terminals = []
            self.buffer.values = exp["old_values"]

            if self.buffer.states:
                a_loss, c_loss, kl = self._update_ppo(
                    precomputed_advantages=exp["normalized_advantages"]
                )
                total_a_loss += a_loss
                total_c_loss += c_loss
                total_updates += 1
                self.buffer.clear()

        restore_env = self._get_env_by_name(saved_env_name)
        if restore_env is not None:
            self.switch_environment(restore_env)

        self._last_mixed_losses = {
            "a_loss": total_a_loss / max(1, total_updates),
            "c_loss": total_c_loss / max(1, total_updates),
            "updates": total_updates,
        }

        return total_collected, all_ep_rewards, all_ep_makespans

    def _async_collect(self, sampled_envs, steps_per_env, total_steps):
        """
        多环境顺序采样：依次切换到各环境收集经验。

        各环境共用同一套策略权重，switch_environment 负责将当前权重迁移到
        新拓扑，因此各环境的梯度更新可以相互受益，实现真正的多网泛化训练。

        Args:
            sampled_envs: 需要采样的环境列表
            steps_per_env: 每个环境分配的步数
            total_steps: 总步数

        Returns:
            env_experiences: 各环境的经验列表
            total_collected: 收集的总步数
            all_ep_rewards: 所有回合奖励
            all_ep_makespans: 所有回合 makespan
        """

        env_experiences = []
        total_collected = 0
        all_ep_rewards = []
        all_ep_makespans = []

        for i, env in enumerate(sampled_envs):
            # 收集当前环境的经验（共享权重通过 switch_environment 自动迁移）
            self.switch_environment(env)
            self.env_visit_counts[self.current_env_name] = self.env_visit_counts.get(self.current_env_name, 0) + 1

            self.buffer.clear()
            collected, ep_rewards, ep_makespans = self._collect_rollouts(steps_per_env)
            total_collected += collected
            all_ep_rewards.extend(ep_rewards)
            all_ep_makespans.extend(ep_makespans)

            if self.buffer.states:
                old_values = torch.tensor(self.buffer.values, dtype=torch.float32)
                raw_advantages, raw_returns = self._compute_gae(
                    self.buffer.rewards, old_values.tolist(), self.buffer.is_terminals
                )
                env_experiences.append({
                    "env_name": self.current_env_name,
                    "buffer_states": list(self.buffer.states),
                    "buffer_actions": list(self.buffer.actions),
                    "buffer_logprobs": list(self.buffer.logprobs),
                    "buffer_masks": list(self.buffer.masks),
                    "raw_advantages": raw_advantages,
                    "raw_returns": raw_returns,
                    "old_values": old_values.tolist(),
                })
            self.buffer.clear()

        return env_experiences, total_collected, all_ep_rewards, all_ep_makespans

    def _compute_gae(self, rewards, values, is_terminals):
        """
        计算广义优势估计 (GAE)
        
        Args:
            rewards: 奖励列表
            values: 状态价值列表
            is_terminals: 是否终止状态列表
            
        Returns:
            advantages: 优势值列表
            returns: 回报列表
        """
        advantages, returns, gae = [], [], 0
        for i in reversed(range(len(rewards))):
            if is_terminals[i]:
                gae = 0
            next_value = 0.0 if is_terminals[i] or i == len(rewards) - 1 else values[i + 1]
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        return advantages, returns

    def _update_ppo(self, precomputed_advantages=None) -> Tuple[float, float, float]:
        # 没有经验不更新
        if not self.buffer.states: return 0.0, 0.0, 0.0
        
        self.model.train()

        old_states = torch.stack(self.buffer.states).detach()
        old_actions = torch.tensor(self.buffer.actions, dtype=torch.int64, device=self.device).detach()
        masks = torch.stack(self.buffer.masks).detach()
        old_logprobs = torch.tensor(self.buffer.logprobs, dtype=torch.float32, device=self.device).detach()
        old_values = torch.tensor(self.buffer.values, dtype=torch.float32, device=self.device).detach()
        dataset_size = len(old_states)

        if precomputed_advantages is not None:
            advantages = precomputed_advantages.to(self.device)
            returns = torch.tensor(
                [a + v for a, v in zip(precomputed_advantages.tolist(), old_values.tolist())],
                dtype=torch.float32, device=self.device
            )
        else:
            raw_advantages, raw_returns = self._compute_gae(
                self.buffer.rewards,
                old_values.tolist(),
                self.buffer.is_terminals
            )
            returns = torch.tensor(raw_returns, dtype=torch.float32, device=self.device)
            advantages = torch.tensor(raw_advantages, dtype=torch.float32, device=self.device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        b_inds = np.arange(dataset_size)
        sum_actor_loss, sum_critic_loss, updates_count, approx_kl = 0.0, 0.0, 0, 0.0

        for epoch in range(self.ppo_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, dataset_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]
                
                mb_states = old_states[mb_inds]
                mb_actions = old_actions[mb_inds]
                mb_old_logprobs = old_logprobs[mb_inds]
                mb_advs = advantages[mb_inds]
                mb_returns = returns[mb_inds]
                mb_masks = masks[mb_inds]
                mb_old_values = old_values[mb_inds]

                # 仅在未使用预计算优势时进行 minibatch 级归一化。
                # 若传入了 precomputed_advantages（已在跨环境 GAE 阶段统一归一化），
                # 再次归一化会破坏跨环境的优势幅度比较，导致梯度信号失真。
                if precomputed_advantages is None and len(mb_advs) > 1:
                    mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)

                logits, values = self.model(mb_states)
                logits_unmasked = logits.clone()
                logits[~mb_masks] = -1e9 
                
                # 使用 T=1.0 的原始 logits 计算新 logprob 和熵，
                # 与收集阶段记录的 old_logprob（同为 T=1.0）保持一致，
                # 保证 log_ratio = log π_new - log π_old 仅反映策略参数变化。
                dist = Categorical(logits=logits)
                logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy()
                
                with torch.no_grad():
                    log_ratio = logprobs - mb_old_logprobs
                    approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean().item()

                ratios = torch.exp(logprobs - mb_old_logprobs)
                surr1 = ratios * mb_advs
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advs
                
                is_valid_state = mb_masks.any(dim=-1)
                if is_valid_state.any():
                    actor_loss = -torch.min(surr1, surr2)[is_valid_state].mean()
                    entropy_loss = entropy[is_valid_state].mean()
                    logits_penalty = 0.001 * (logits_unmasked[mb_masks] ** 2).mean()
                else:
                    actor_loss, entropy_loss, logits_penalty = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
                
                v_clipped = mb_old_values + torch.clamp(values - mb_old_values, -self.eps_clip, self.eps_clip)
                v_loss1 = F.smooth_l1_loss(values, mb_returns, reduction='none')
                v_loss2 = F.smooth_l1_loss(v_clipped, mb_returns, reduction='none')
                critic_loss = torch.max(v_loss1, v_loss2).mean()

                loss = actor_loss + self.value_loss_coef * critic_loss - self.current_entropy_coef * entropy_loss + logits_penalty

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                sum_actor_loss += actor_loss.item()
                sum_critic_loss += critic_loss.item()
                updates_count += 1

        self.buffer.clear()
        uc = max(1, updates_count)
        return sum_actor_loss / uc, sum_critic_loss / uc, approx_kl

    def _evaluate_greedy(self) -> Tuple[bool, int]:
        """
            评估当前策略在当前环境中的性能
            返回值(是否成功,完工时间)
        """
        self._set_to_initial()
        curr_marking = self.petri_net.get_marking()
        self.model.eval()
        
        for _ in range(800):
            if self._is_goal(curr_marking):
                return True, curr_marking.get_prefix()
            encoded = self.encoder.encode(curr_marking)
            mask = self._mask_from_marking(curr_marking)
            with torch.no_grad():
                logits, _ = self.model(encoded.unsqueeze(0)) if not isinstance(encoded, PetriRepresentationInput) else self.model(encoded)
                logits = logits.squeeze(0)
                logits[~mask] = -1e9
                if not mask.any(): 
                    return False, -1
                action = torch.argmax(logits).item()
                if not mask[action].item():
                    return False, -1
            curr_marking = self.petri_net.launch(action)
            self.petri_net.set_marking(curr_marking)
            
        return False, -1

    def _evaluate_pool(self, env_pool) -> Dict[str, float]:
        saved_env_name = self.current_env_name
        success_count = 0
        makespans = []
        total = 0
        for env in env_pool:
            self.switch_environment(env)
            success, makespan = self._evaluate_greedy()
            total += 1
            if success:
                success_count += 1
                makespans.append(makespan)

        restore_env = self._get_env_by_name(saved_env_name)
        if restore_env is not None:
            self.switch_environment(restore_env)

        return {
            "total": float(total),
            "success": float(success_count),
            "success_rate": (float(success_count) / float(total)) if total else 0.0,
            "avg_makespan": float(np.mean(makespans)) if makespans else -1.0,
            "worst_makespan": float(max(makespans)) if makespans else -1.0,
        }

    def il_warmstart(self, il_checkpoint_path: str = "", il_mode: str = "auto") -> bool:
        """
        模仿学习热启动：在 PPO 训练开始前，加载 BC/DAgger 预训练权重到模型中

        参考实现：run_gcn_ppo_scene_train.py 中的 IL 热启动逻辑
        - 优先加载 DAgger checkpoint，其次回退到 BC checkpoint
        - 只在训练开始时加载一次，后续训练由 PPO 自身驱动
        - 加载的权重仅覆盖 actor_net（策略网络），critic 不受影响

        Args:
            il_checkpoint_path: 显式指定的 IL checkpoint 路径，为空时自动搜索
            il_mode: IL 模式，可选 "auto"（优先 DAgger，其次 BC）、"bc"、"dagger"

        Returns:
            bool: 是否成功进行了 IL 热启动
        """
        from imitation.il_checkpoint import normalize_il_mode, resolve_il_checkpoint, classify_il_artifact

        base_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(base_dir)

        normalized_mode = normalize_il_mode(il_mode)

        if il_checkpoint_path and os.path.isabs(il_checkpoint_path):
            ckpt_path = il_checkpoint_path
        elif il_checkpoint_path:
            ckpt_path = os.path.join(parent_dir, il_checkpoint_path)
        else:
            ckpt_path = resolve_il_checkpoint(
                parent_dir,
                normalized_mode,
                net_stem="",
                scene_id="",
                explicit="",
            )

        if not ckpt_path or not os.path.exists(ckpt_path):
            self._log(f"[IL-Warmstart] No IL checkpoint found (mode={normalized_mode})")
            return False

        try:
            saved_il = torch.load(ckpt_path, map_location="cpu")
            if not isinstance(saved_il, dict):
                self._log(f"[IL-Warmstart] Invalid checkpoint format: {ckpt_path}")
                return False

            actor_state = (
                saved_il.get("model_state")
                or saved_il.get("policy_state")
                or saved_il.get("actor_state")
                or {}
            )

            if not actor_state:
                self._log(f"[IL-Warmstart] No actor weights found in checkpoint: {ckpt_path}")
                return False

            load_compatible_state(self.model.actor_net, actor_state)

            il_method, il_source = classify_il_artifact(parent_dir, ckpt_path)
            self._log(
                f"[IL-Warmstart] Loaded {il_method}/{il_source} checkpoint: {ckpt_path}"
            )

            self.extra_info["ilWarmstart"] = True
            self.extra_info["ilWarmstartMethod"] = il_method
            self.extra_info["ilWarmstartSource"] = il_source
            self.extra_info["ilWarmstartCheckpoint"] = ckpt_path
            return True

        except Exception as e:
            self._log(f"[IL-Warmstart] Failed to load IL checkpoint: {e}")
            return False

    def train_model(self):
        mode_label = "Mixed" if self.mixed_rollout else "Sequential"
        self._log(f"=== [PPO-PRO] Training Started | Mode: {mode_label} | Target Steps: {self.max_train_steps} ===")
        saved_env_name = self.current_env_name
        total_steps = 0
        epoch_idx = 0
        
        while total_steps < self.max_train_steps:
            epoch_idx += 1
            progress = total_steps / self.max_train_steps
            
            if self.mixed_rollout and self.env_pool and len(self.env_pool) > 1:
                steps_collected, ep_rewards, ep_makespans = self._collect_mixed_rollouts(self.steps_per_epoch, progress)
                mixed_losses = getattr(self, "_last_mixed_losses", {})
                a_loss = mixed_losses.get("a_loss", 0.0)
                c_loss = mixed_losses.get("c_loss", 0.0)
            else:
                if self.env_pool and len(self.env_pool) > 1:
                    current_env = self._select_training_env(epoch_idx)
                    self.switch_environment(current_env)
                    self.env_visit_counts[self.current_env_name] = self.env_visit_counts.get(self.current_env_name, 0) + 1
                    
                steps_collected, ep_rewards, ep_makespans = self._collect_rollouts(self.steps_per_epoch)
            
            total_steps += steps_collected
            
            if not self.mixed_rollout or not self.env_pool or len(self.env_pool) <= 1:
                a_loss, c_loss, kl = self._update_ppo()
            eval_success, eval_makespan = self._evaluate_greedy()  # 评估模型
            
            progress = total_steps / self.max_train_steps
            new_lr = max(1e-5, self.initial_lr * (1.0 - progress))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            # 熵系数和温度衰减
            self.current_entropy_coef = self.entropy_coef_start - progress * (self.entropy_coef_start - self.entropy_coef_end)
            self.current_entropy_coef = max(self.entropy_coef_end, self.current_entropy_coef)
            self.current_temperature = self.temperature_start - progress * (self.temperature_start - self.temperature_end)
            self.current_temperature = max(self.temperature_end, self.current_temperature)

            avg_reward = np.mean(ep_rewards) if ep_rewards else 0.0
            best_show = self.best_records[self.current_env_name]["makespan"] if self.best_records[self.current_env_name]["makespan"] < 2**31 - 1 else -1
            eval_show = eval_makespan if eval_success else "Fail"
            pool_text = ""
            if self.env_pool and len(self.env_pool) > 1:
                need_pool_eval = epoch_idx == 1 or epoch_idx % self.pool_eval_interval == 0 or total_steps >= self.max_train_steps
                if need_pool_eval:
                    pool_metrics = self._evaluate_pool(self.env_pool)
                    self.extra_info["poolSuccessRate"] = pool_metrics["success_rate"]
                    self.extra_info["poolAvgMakespan"] = pool_metrics["avg_makespan"]
                    self.extra_info["poolWorstMakespan"] = pool_metrics["worst_makespan"]
                    pool_avg_show = int(pool_metrics["avg_makespan"]) if pool_metrics["avg_makespan"] >= 0 else "Fail"
                    pool_worst_show = int(pool_metrics["worst_makespan"]) if pool_metrics["worst_makespan"] >= 0 else "Fail"
                    pool_text = (
                        f" | Pool SR: {pool_metrics['success_rate']:.2f}"
                        f" | Pool Avg: {pool_avg_show}"
                        f" | Pool Worst: {pool_worst_show}"
                    )
                    # ★ 最优模型快照：当训练池整体成功率创新高时保存模型权重。
                    # 防止训练后期过拟合导致最终模型性能退化。
                    cur_score = pool_metrics["success_rate"] * 1000.0 - (
                        pool_metrics["avg_makespan"] if pool_metrics["avg_makespan"] >= 0 else 1e9
                    ) * 0.001
                    if cur_score > self._best_pool_score:
                        self._best_pool_score = cur_score
                        self._best_snapshot = {
                            "actor_state": {k: v.detach().cpu() for k, v in self.model.actor_net.state_dict().items()},
                            "critic_state": {k: v.detach().cpu() for k, v in self.model.value_head.state_dict().items()},
                            "epoch": epoch_idx,
                            "total_steps": total_steps,
                            "pool_success_rate": pool_metrics["success_rate"],
                            "pool_avg_makespan": pool_metrics["avg_makespan"],
                        }
                        self._log(f"    [Snapshot] New best pool snapshot saved (SR={pool_metrics['success_rate']:.2f}, Avg={pool_avg_show})")

            # ★ 独立评估池（eval_env_pool）监控：定期对结构相似的测试网络进行贪婪评估，
            # 跟踪泛化能力的变化趋势，提前发现过拟合。
            eval_pool_text = ""
            if self.eval_env_pool and self.eval_pool_interval > 0:
                need_eval_pool = epoch_idx % self.eval_pool_interval == 0 or total_steps >= self.max_train_steps
                if need_eval_pool:
                    eval_metrics = self._evaluate_pool(self.eval_env_pool)
                    self.extra_info["evalPoolSuccessRate"] = eval_metrics["success_rate"]
                    self.extra_info["evalPoolAvgMakespan"] = eval_metrics["avg_makespan"]
                    eval_avg_show = int(eval_metrics["avg_makespan"]) if eval_metrics["avg_makespan"] >= 0 else "Fail"
                    eval_pool_text = (
                        f" | EvalPool SR: {eval_metrics['success_rate']:.2f}"
                        f" | EvalPool Avg: {eval_avg_show}"
                    )

            # 每epoch打印一次日志
            self._log(
                f"Env: {self.current_env_name} | Ep {epoch_idx:03d} | Steps: {total_steps}/{self.max_train_steps} | "
                f"Avg R: {avg_reward:6.1f} | Eval: {eval_show} | Best: {best_show} | "
                f"a_loss: {a_loss:5.2f} c_loss: {c_loss:5.2f}{pool_text}{eval_pool_text}"
            )

        self.is_trained = True

        if self.env_pool:
            for env in self.env_pool:
                if env.get("name") == saved_env_name:
                    self.switch_environment(env)
                    break

        # ★ 恢复最优快照：若训练中途拍摄了更好的模型，用它替代最终 epoch 的模型。
        # 这可以对抗训练后期因过拟合导致的性能退化。
        if self._best_snapshot is not None:
            snap = self._best_snapshot
            snap_sr = snap.get("pool_success_rate", 0.0)
            snap_ep = snap.get("epoch", 0)
            snap_steps = snap.get("total_steps", 0)
            self._log(
                f"[Snapshot] Restoring best snapshot from epoch {snap_ep} "
                f"(steps={snap_steps}, pool_SR={snap_sr:.2f}) for inference."
            )
            try:
                load_compatible_state(self.model.actor_net, snap["actor_state"])
                load_compatible_state(self.model.value_head, snap["critic_state"])
            except Exception as e:
                self._log(f"[Snapshot] Restore failed: {e}, using final model.")

        env_record = getattr(self, "best_records", {}).get(self.current_env_name, {})
        self.best_train_makespan = env_record.get("makespan", 2**31-1)
        self.best_train_trans = env_record.get("trans", [])

        self.extra_info["trainSteps"] = total_steps
        self.extra_info["bestTrainMakespan"] = self.best_train_makespan if self.best_train_makespan < 2**31 - 1 else -1
        self._log(f"=== [PPO-PRO] Training Completed ===")

    def _result_from_trans(self, trans: List[int]) -> Result:
        """
            强化序列检查，打印警告而不是直接返回空序列
            底下这三个函数都是为了训练完之后做测试用的,测试环境是env_pool的第一个环境
            切换环境在train_ppo_3中完成
        """
        # 作用就是重跑一边收集序列，一边检查是否合法
        self._set_to_initial()
        curr_marking = self.petri_net.get_marking()
        markings = [curr_marking]
        used = []
        for action in trans:
            if action < 0: break
            self.petri_net.set_marking(curr_marking.clone())
            mask = self._mask_from_marking(curr_marking)
            if not mask[action].item(): 
                self._log(f"    [Warning] 序列还原失败！变迁 {action} 在第 {len(used)} 步时不合法！(环境可能未正确初始化)")
                break
                
            next_marking = self.petri_net.launch(action)
            self.petri_net.set_marking(next_marking)
            used.append(action)
            markings.append(next_marking)
            curr_marking = next_marking
            if self._is_goal(curr_marking): 
                break
        return Result(used, markings)

    def _beam_search(self) -> Result:
        self._set_to_initial()
        init = self.petri_net.get_marking().clone()
        frontier = [(init, [], float(init.get_prefix()))]
        visited = {self._state_key(init): float(init.get_prefix())}
        best = Result([], [init])
        
        self.model.eval()
        
        for _ in range(self.beam_depth):
            candidates = []
            for marking, trans, _ in frontier:
                if self._is_goal(marking):
                    return self._result_from_trans(trans)
                
                self.petri_net.set_marking(marking)
                mask = self._mask_from_marking(marking)
                enabled = torch.nonzero(mask, as_tuple=False).flatten()
                if enabled.numel() == 0: 
                    continue
                
                with torch.no_grad():
                    encoded = self.encoder.encode(marking)
                    logits, _ = self.model(encoded.unsqueeze(0)) if not isinstance(encoded, PetriRepresentationInput) else self.model(encoded)
                    logits = logits.squeeze(0)
                    logits_masked = logits.clone()
                    logits_masked[~mask] = -1e9
                    k = min(10, enabled.numel())
                    top_vals, top_idx = torch.topk(logits_masked, k=k)
                    
                for idx in top_idx.tolist():
                    action = int(idx)

                    if not mask[action].item():
                        continue

                    self.petri_net.set_marking(marking.clone())
                    next_marking = self.petri_net.launch(action)
                    key = self._state_key(next_marking)
                    prefix = float(next_marking.get_prefix())
                    if prefix >= visited.get(key, 2 ** 31 - 1): continue
                    
                    visited[key] = prefix
                    heuristic = float(self._goal_distance(next_marking))
                    score = prefix + heuristic * 2.0
                    candidates.append((next_marking.clone(), trans + [action], score))
                    
                    if self._is_goal(next_marking):
                        return self._result_from_trans(trans + [action])
                        
            if not candidates: break
            candidates.sort(key=lambda x: x[2])
            frontier = candidates[: self.beam_width]
            best = self._result_from_trans(frontier[0][1])
        return best

    def _greedy_search(self) -> Result:
        """
        贪婪搜索策略：每次选择具有最大 logit 值的变迁执行
        
        与束搜索不同，贪婪搜索：
        - 不维护多个候选路径
        - 每步只选择最优动作
        - 计算速度更快，但可能陷入局部最优
        
        Returns:
            Result: 包含变迁序列和标识序列的结果对象
        """
        self._set_to_initial()
        curr_marking = self.petri_net.get_marking().clone()
        trans_sequence = []
        markings = [curr_marking.clone()]
        visited = set()
        
        self.model.eval()
        
        for step in range(self.beam_depth):
            if self._is_goal(curr_marking):
                self._log(f"[Greedy] Goal reached at step {step}")
                return Result(trans_sequence, markings)
            
            state_key = self._state_key(curr_marking)
            if state_key in visited:
                self._log(f"[Greedy] Cycle detected at step {step}, stopping")
                break
            visited.add(state_key)
            
            self.petri_net.set_marking(curr_marking)
            mask = self._mask_from_marking(curr_marking)
            
            if not mask.any():
                self._log(f"[Greedy] No valid actions at step {step}, deadlock")
                break
            
            with torch.no_grad():
                encoded = self.encoder.encode(curr_marking)
                logits, _ = self.model(encoded.unsqueeze(0)) if not isinstance(encoded, PetriRepresentationInput) else self.model(encoded)
                logits = logits.squeeze(0)
                logits[~mask] = -1e9
                action = torch.argmax(logits).item()
            
            if not mask[action]:
                self._log(f"[Greedy] Selected action {action} is invalid at step {step}")
                break
            
            next_marking = self.petri_net.launch(action)
            self.petri_net.set_marking(next_marking)
            
            trans_sequence.append(action)
            markings.append(next_marking.clone())
            curr_marking = next_marking
        
        return Result(trans_sequence, markings)

    def _stochastic_search(self, num_rollouts: int = 50, temperature: float = 1.2) -> Result:
        """
        带温度的并行多轨迹采样策略（Stochastic Sampling with Temperature）

        核心思路：利用温度系数软化 GNN 输出的概率分布，并行跑 N 条互不干扰的完整轨迹，
        最后通过后置过滤选出 Makespan 最小且未死锁的最优解。

        相比束搜索的优势：
        - 无需状态拷贝和排序剪枝，推理速度更快
        - 每条轨迹独立采样，天然避免束搜索的路径趋同问题
        - 温度系数控制探索-利用平衡，高温增加多样性，低温偏向确定性

        Args:
            num_rollouts: 并行采样的轨迹数量，默认 50
            temperature: 温度系数，控制采样分布的平滑程度，默认 1.2
                        - temperature > 1.0: 分布更平滑，增加探索性
                        - temperature = 1.0: 使用原始 logits 分布
                        - temperature < 1.0: 分布更尖锐，偏向确定性选择

        Returns:
            Result: 包含变迁序列和标识序列的结果对象
        """
        self.model.eval()
        candidates = []

        with torch.no_grad():
            for rollout_idx in range(num_rollouts):
                self._set_to_initial()
                curr_marking = self.petri_net.get_marking().clone()
                trans_sequence = []
                markings = [curr_marking.clone()]

                for step in range(self.beam_depth):
                    if self._is_goal(curr_marking):
                        break

                    self.petri_net.set_marking(curr_marking)
                    mask = self._mask_from_marking(curr_marking)

                    enabled = torch.nonzero(mask, as_tuple=False).flatten()
                    if enabled.numel() == 0:
                        break

                    encoded = self.encoder.encode(curr_marking)
                    if isinstance(encoded, PetriRepresentationInput):
                        logits, _ = self.model(encoded)
                    else:
                        logits, _ = self.model(encoded.unsqueeze(0))
                    logits = logits.squeeze(0)

                    # 温度系数放缩：logits / temperature
                    # 高温使分布更平滑（增加探索），低温使分布更尖锐（偏向利用）
                    scaled_logits = logits / temperature
                    scaled_logits[~mask] = -1e9

                    # 从软化后的概率分布中采样动作（而非 argmax）
                    dist = Categorical(logits=scaled_logits)
                    action = dist.sample().item()

                    if not mask[action].item():
                        break

                    next_marking = self.petri_net.launch(action)
                    self.petri_net.set_marking(next_marking)

                    trans_sequence.append(action)
                    markings.append(next_marking.clone())
                    curr_marking = next_marking

                candidates.append(Result(trans_sequence, markings))

        # 后置过滤：选出 Makespan 最小且到达目标的最优解
        goal_candidates = []
        deadlock_candidates = []

        for r in candidates:
            markings = r.get_markings()
            if not markings or len(r.get_trans()) == 0:
                continue
            last = markings[-1]
            if self._is_goal(last):
                goal_candidates.append(r)
            else:
                deadlock_candidates.append(r)

        if goal_candidates:
            # 在成功到达目标的轨迹中，按 Makespan 升序排序，选最优
            goal_candidates.sort(key=lambda r: r.get_markings()[-1].get_prefix())
            best = goal_candidates[0]
            self._log(f"[Stochastic] {num_rollouts} rollouts | "
                      f"Goal: {len(goal_candidates)}/{num_rollouts} | "
                      f"Best Makespan: {best.get_markings()[-1].get_prefix():.2f}")
            return best

        # 容错兜底：所有轨迹全部死锁，返回步数最多或距离目标最近的一条
        if deadlock_candidates:
            deadlock_candidates.sort(
                key=lambda r: (
                    -len(r.get_trans()),
                    self._goal_distance(r.get_markings()[-1]),
                    r.get_markings()[-1].get_prefix(),
                )
            )
            best = deadlock_candidates[0]
            self._log(f"[Stochastic] {num_rollouts} rollouts | "
                      f"All deadlocked | Fallback: {len(best.get_trans())} steps, "
                      f"dist={self._goal_distance(best.get_markings()[-1])}")
            return best

        return Result([], [])

    def search(self, strategy: str = None, num_rollouts: int = None, temperature: float = None):
        """
        执行推理搜索，返回最优变迁序列
        
        Args:
            strategy: 搜索策略，可选 "beam"（束搜索）、"greedy"（贪婪搜索）或 "stochastic"（随机采样）。
                     如果为 None,则使用实例初始化时设置的 search_strategy。
            num_rollouts: 随机采样策略的轨迹数量，仅当 strategy="stochastic" 时生效。
                         如果为 None,则使用实例初始化时设置的 stochastic_num_rollouts。
            temperature: 随机采样策略的温度系数，仅当 strategy="stochastic" 时生效。
                        如果为 None,则使用实例初始化时设置的 stochastic_temperature。
        
        Returns:
            Result: 包含变迁序列和标识序列的结果对象
        
        策略说明：
            - beam: 束搜索，维护多个候选路径，探索更全面，但计算开销较大
            - greedy: 贪婪搜索，每步选择最优动作，速度快但可能陷入局部最优
            - stochastic: 带温度的并行多轨迹采样，利用温度系数软化概率分布，
                         并行跑 N 条独立轨迹，后置过滤选最优解
        """
        if not self.is_trained:
            self.train_model()
        
        current_strategy = strategy if strategy is not None else self.search_strategy
        self._log(f"[PPO-PRO] Inference Start | Strategy: {current_strategy}")
        
        candidates = []
        
        env_record = getattr(self, "best_records", {}).get(self.current_env_name, {})
        valid_trans = env_record.get("trans", [])
        
        self.best_train_makespan = env_record.get("makespan", 2**31-1)
        self.best_train_trans = valid_trans

        if valid_trans:
            candidates.append(self._result_from_trans(valid_trans))
        
        if current_strategy == "greedy":
            candidates.append(self._greedy_search())
        elif current_strategy == "stochastic":
            rollout_n = num_rollouts if num_rollouts is not None else self.stochastic_num_rollouts
            temp = temperature if temperature is not None else self.stochastic_temperature
            candidates.append(self._stochastic_search(num_rollouts=rollout_n, temperature=temp))
        else:
            candidates.append(self._beam_search())
        
        best = None
        best_tuple = None
        for r in candidates:
            markings = r.get_markings()
            if not markings: continue
            last = markings[-1]
            is_goal = self._is_goal(last)
            goal_dist = self._goal_distance(last)
            prefix = last.get_prefix()
            
            if len(r.get_trans()) == 0:
                prefix += 99999999.0
                
            key = (0 if is_goal else 1, goal_dist, prefix, len(r.get_trans()))
            if best is None or key < best_tuple:
                best = r
                best_tuple = key
                
        final_best = best if best is not None else Result([], [])
        
        mks = final_best.get_markings()
        self.extra_info["inferenceTransCount"] = len(final_best.get_trans())
        self.extra_info["inferenceMakespan"] = mks[-1].get_prefix() if (mks and len(final_best.get_trans())>0) else -1
        self.extra_info["reachGoal"] = self._is_goal(mks[-1]) if mks else False
        self.extra_info["goalDistance"] = float(self._goal_distance(mks[-1])) if mks else -1
        self.extra_info["searchStrategy"] = current_strategy
        self._log(f"[PPO-PRO] Inference Done | Strategy: {current_strategy} | Makespan: {self.extra_info['inferenceMakespan']}")
        return final_best

    def get_extra_info(self):
        return self.extra_info
