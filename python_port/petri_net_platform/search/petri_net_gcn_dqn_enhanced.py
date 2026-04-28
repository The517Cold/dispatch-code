import random
from typing import Dict, List, Tuple
import os
import sys

import torch
from torch import nn
import torch.nn.functional as F

try:
    from .abstract_search import AbstractSearch
    from .deadlock_controller import DeadlockController
    from .dqn_components import Experience, PrioritizedReplay
    from .petri_gcn_models import PetriNetGCNEnhanced, PetriStateEncoderEnhanced
    from .rl_env_semantics import (
        describe_stop_info,
        format_reason_counts,
        make_stop_info,
        stop_info_label,
    )
    from ..utils.result import Result
except ImportError:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from python_port.petri_net_platform.search.abstract_search import AbstractSearch
    from python_port.petri_net_platform.search.deadlock_controller import DeadlockController
    from python_port.petri_net_platform.search.dqn_components import Experience, PrioritizedReplay
    from python_port.petri_net_platform.search.petri_gcn_models import PetriNetGCNEnhanced, PetriStateEncoderEnhanced
    from python_port.petri_net_platform.search.rl_env_semantics import (
        describe_stop_info,
        format_reason_counts,
        make_stop_info,
        stop_info_label,
    )
    from python_port.petri_net_platform.utils.result import Result

"""PetriNet GCN-DQN 搜索器。

包含优先级回放、Double DQN、软更新、beam 推理。
"""


class PetriNetGCNDQNEnhanced(AbstractSearch):
    """ GCN + DQN 调度搜索。"""

    def __init__(
        self,
        petri_net,
        end: List[int],
        pre: List[List[int]],
        post: List[List[int]],
        min_delay_p: List[int],
        lambda_p: int = 48,
        lambda_t: int = 48,
        num_layers: int = 3,
        gamma: float = 0.985,
        lr: float = 8e-4,
        replay_capacity: int = 12000,
        replay_alpha: float = 0.6,
        replay_beta: float = 0.4,
        batch_size: int = 24,
        learn_every: int = 2,
        warmup_steps: int = 80,
        tau: float = 0.02,
        train_episodes: int = 80,
        min_steps_per_episode: int = 80,
        max_steps_per_episode: int = 420,
        inference_max_steps_per_episode: int = None,
        epsilon_init: float = 1.0,
        epsilon_min: float = 0.03,
        epsilon_decay: float = 0.992,
        reward_goal_bonus: float = 240.0,
        reward_deadlock_penalty: float = 80.0,
        reward_progress_weight: float = 1.8,
        reward_repeat_penalty: float = 0.2,
        use_reward_scaling: bool = True,
        reward_time_scale: float = 1000.0,
        use_reward_clip: bool = True,
        reward_clip_abs: float = 20.0,
        use_huber_loss: bool = True,
        huber_beta: float = 1.0,
        rollout_count: int = 18,
        beam_width: int = 26,
        beam_expand_per_node: int = 4,
        goal_eval_rollouts: int = 6,
        goal_min_success_rate: float = 0.5,
        extra_train_episodes: int = 0,
        verbose: bool = False,
        log_interval: int = 5,
        controller_enable_lookahead: bool = True,
        controller_lookahead_depth: int = 2,
        controller_lookahead_width: int = 4,
        controller_lookahead_trigger_safe_limit: int = 4,
        controller_lookahead_trigger_on_fbm: bool = True,
        controller_representation_enabled: bool = True,
        device: str = None,
    ):
        super().__init__()
        self.petri_net = petri_net
        self.initial_petri_net = petri_net.clone()
        self.end = end
        self.pre = pre
        self.post = post
        self.capacity = getattr(petri_net, "capacity", None)
        self.has_capacity = bool(getattr(petri_net, "has_capacity", False)) and self.capacity is not None
        transition_flow_allowed = getattr(petri_net, "transition_flow_allowed", None)
        trans_count = len(pre[0])
        if isinstance(transition_flow_allowed, list) and len(transition_flow_allowed) == trans_count:
            self.transition_flow_allowed = transition_flow_allowed
        else:
            self.transition_flow_allowed = [True] * trans_count
        self.gamma = gamma
        self.replay_beta = replay_beta
        self.batch_size = batch_size
        self.learn_every = learn_every
        self.warmup_steps = warmup_steps
        self.tau = tau
        self.train_episodes = train_episodes
        self.min_steps_per_episode = min_steps_per_episode
        self.max_steps_per_episode = max_steps_per_episode
        if inference_max_steps_per_episode is None:
            inference_max_steps_per_episode = max_steps_per_episode
        self.inference_max_steps_per_episode = max(1, int(inference_max_steps_per_episode))
        self.epsilon_init = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.reward_goal_bonus = reward_goal_bonus
        self.reward_deadlock_penalty = reward_deadlock_penalty
        self.reward_progress_weight = reward_progress_weight
        self.reward_repeat_penalty = reward_repeat_penalty
        self.use_reward_scaling = bool(use_reward_scaling)
        self.reward_time_scale = max(1.0, float(reward_time_scale))
        self.use_reward_clip = bool(use_reward_clip)
        self.reward_clip_abs = max(0.5, float(reward_clip_abs))
        self.use_huber_loss = bool(use_huber_loss)
        self.huber_beta = max(1e-6, float(huber_beta))
        self.rollout_count = rollout_count
        self.beam_width = beam_width
        self.beam_expand_per_node = beam_expand_per_node
        self.goal_eval_rollouts = max(1, int(goal_eval_rollouts))
        self.goal_min_success_rate = max(0.0, min(1.0, float(goal_min_success_rate)))
        self.extra_train_episodes = max(0, int(extra_train_episodes))
        self.verbose = verbose
        self.log_interval = max(1, log_interval)
        self.controller_representation_enabled = bool(controller_representation_enabled)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.encoder = PetriStateEncoderEnhanced(
            end=end,
            min_delay_p=min_delay_p,
            device=self.device,
            pre=pre,
            post=post,
            min_delay_t=getattr(petri_net, "min_delay_t", None),
            capacity=self.capacity,
            max_residence_time=getattr(petri_net, "max_residence_time", None),
            place_from_places=getattr(petri_net, "place_from_places", None),
        )
        self.policy_net = PetriNetGCNEnhanced(
            pre,
            post,
            lambda_p,
            lambda_t,
            num_layers,
            end=end,
            min_delay_p=min_delay_p,
            min_delay_t=getattr(petri_net, "min_delay_t", None),
            capacity=self.capacity,
            max_residence_time=getattr(petri_net, "max_residence_time", None),
            place_from_places=getattr(petri_net, "place_from_places", None),
        ).to(self.device)
        self.target_net = PetriNetGCNEnhanced(
            pre,
            post,
            lambda_p,
            lambda_t,
            num_layers,
            end=end,
            min_delay_p=min_delay_p,
            min_delay_t=getattr(petri_net, "min_delay_t", None),
            capacity=self.capacity,
            max_residence_time=getattr(petri_net, "max_residence_time", None),
            place_from_places=getattr(petri_net, "place_from_places", None),
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = PrioritizedReplay(replay_capacity, replay_alpha)
        self.extra_info: Dict[str, float] = {}
        self.is_trained = False
        self.best_train_trans: List[int] = []
        self.best_train_makespan = 2 ** 31 - 1
        self.train_failure_counts: Dict[str, int] = {}
        self.eval_failure_counts: Dict[str, int] = {}
        self.deadlock_controller = DeadlockController(
            pre=pre,
            post=post,
            end=end,
            capacity=self.capacity,
            has_capacity=self.has_capacity,
            transition_flow_allowed=self.transition_flow_allowed,
            controller_name="gcn_dqn",
            enable_lookahead=controller_enable_lookahead,
            lookahead_depth=controller_lookahead_depth,
            lookahead_width=controller_lookahead_width,
            lookahead_trigger_safe_limit=controller_lookahead_trigger_safe_limit,
            lookahead_trigger_on_fbm=controller_lookahead_trigger_on_fbm,
        )
        self.encoder.bind_deadlock_controller(
            lambda: self.petri_net,
            self.deadlock_controller,
            enabled=self.controller_representation_enabled,
        )

    def _log(self, text: str):
        if self.verbose:
            print(text, flush=True)

    def _set_to_initial(self):
        """将环境重置回初始 Petri 网状态。"""
        self.petri_net = self.initial_petri_net.clone()

    def _is_goal(self, marking) -> bool:
        """判断当前 marking 是否达到目标标识。"""
        p_info = marking.get_p_info()
        for i in range(len(p_info)):
            if self.end[i] == -1:
                continue
            if p_info[i] != self.end[i]:
                return False
        return True

    def _goal_distance(self, marking) -> int:
        """计算当前标识到目标标识的离散距离。"""
        p_info = marking.get_p_info()
        dist = 0
        for i in range(len(p_info)):
            if self.end[i] == -1:
                continue
            dist += abs(p_info[i] - self.end[i])
        return dist

    def _state_key(self, marking) -> Tuple:
        """生成用于去重/访问统计的状态键。"""
        over = bool(getattr(marking, "over_max_residence_time", False))
        return tuple(marking.get_p_info()) + (1 if over else 0,)

    def _mask_from_marking(self, marking) -> torch.Tensor:
        """动作掩码由控制器统一生成。"""
        analysis = self._analyze_marking(marking)
        mask = torch.zeros(self.petri_net.get_trans_count(), dtype=torch.bool, device=self.device)
        if analysis.controller_actions:
            mask[analysis.controller_actions] = True
        return mask

    def _analyze_marking(self, marking):
        return self.deadlock_controller.analyze_state(self.petri_net, marking)

    def _deadlock_reason(self, marking) -> str:
        return self._analyze_marking(marking).state_deadlock_reason

    def _write_controller_log(self, marking, context: str):
        analysis = self._analyze_marking(marking)
        self.deadlock_controller.log_analysis(marking, analysis, context)

    def _count_stop_reason(self, bucket: Dict[str, int], stop_info: Dict[str, object]):
        label = stop_info_label(stop_info)
        bucket[label] = bucket.get(label, 0) + 1

    def _log_rollout_stop(self, prefix: str, stop_info: Dict[str, object]):
        if str(stop_info.get("reason")) == "goal":
            return
        self._log(prefix + " stop=" + describe_stop_info(stop_info))

    def _select_action(self, q_values: torch.Tensor, mask: torch.Tensor, epsilon: float) -> int:
        """epsilon-greedy 动作选择。"""
        enabled = torch.nonzero(mask, as_tuple=False).flatten()
        if enabled.numel() == 0:
            return -1
        if random.random() < epsilon:
            idx = random.randint(0, enabled.numel() - 1)
            return int(enabled[idx].item())
        masked_q = q_values.clone()
        masked_q[~mask] = -1e9
        return int(torch.argmax(masked_q).item())

    def _safe_action(self, marking, action: int) -> int:
        """过滤越界或非法动作索引。"""
        if action < 0:
            return -1
        if action >= self.petri_net.get_trans_count():
            return -1
        if hasattr(marking, "curr_delay_t") and action >= len(marking.curr_delay_t):
            return -1
        return action

    def _calc_reward(self, curr_marking, next_marking, done: bool, deadlock: bool, repeat_count: int) -> float:
        """组合奖励：时间代价 + 进展项 + 重复惩罚 + 终止项。"""
        delta_t = float(next_marking.get_prefix() - curr_marking.get_prefix())
        progress = float(self._goal_distance(curr_marking) - self._goal_distance(next_marking))
        # reward = -delta_t + self.reward_progress_weight * progress
        time_cost = (delta_t / self.reward_time_scale) if self.use_reward_scaling else delta_t
        reward = -time_cost + self.reward_progress_weight * progress
        reward -= self.reward_repeat_penalty * float(repeat_count)
        if deadlock:
            reward -= self.reward_deadlock_penalty
        if done:
            reward += self.reward_goal_bonus
        if self.use_reward_clip:
            reward = max(-self.reward_clip_abs, min(self.reward_clip_abs, reward))
        return reward

    def _soft_update(self):
        """对 target network 执行软更新。"""
        with torch.no_grad():
            for p_t, p_s in zip(self.target_net.parameters(), self.policy_net.parameters()):
                p_t.data.mul_(1.0 - self.tau).add_(self.tau * p_s.data)

    def _learn_batch(self) -> float:
        """执行一次带优先级回放的 Double DQN 更新。"""
        idxs, batch, weights = self.replay.sample(self.batch_size, self.replay_beta)
        states = self.encoder.encode_batch([exp.state for exp in batch])
        next_states = self.encoder.encode_batch([exp.next_state for exp in batch])
        actions = torch.tensor([exp.action for exp in batch], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([1.0 if exp.done else 0.0 for exp in batch], dtype=torch.float32, device=self.device)
        next_masks = torch.stack([self._mask_from_marking(exp.next_state) for exp in batch], dim=0)
        weights = weights.to(self.device)

        q_all = self.policy_net(states)
        q_pred = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next_policy = self.policy_net(next_states)
            q_next_policy[~next_masks] = -1e9
            has_next = next_masks.any(dim=1)
            next_actions = torch.argmax(q_next_policy, dim=1)
            q_next_target = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            q_next = torch.where(has_next, q_next_target, torch.zeros_like(q_next_target))
            y = rewards + (1.0 - dones) * self.gamma * q_next

        td = y - q_pred
        # loss = torch.mean(weights * (td ** 2))
        if self.use_huber_loss:
            point_wise = F.smooth_l1_loss(q_pred, y, reduction="none", beta=self.huber_beta)
            loss = torch.mean(weights * point_wise)
        else:
            loss = torch.mean(weights * (td ** 2))
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 2.0)
        self.optimizer.step()
        self._soft_update()
        self.replay.update(idxs, torch.abs(td).detach().cpu().tolist())
        return float(loss.item())

    def _episode_step_limit(self, episode_idx: int) -> int:
        """课程学习：逐轮提升每回合最大步数。"""
        if self.train_episodes <= 1:
            return self.max_steps_per_episode
        ratio = float(episode_idx) / float(self.train_episodes - 1)
        val = int(self.min_steps_per_episode + (self.max_steps_per_episode - self.min_steps_per_episode) * ratio)
        return max(self.min_steps_per_episode, min(self.max_steps_per_episode, val))

    def _update_best_train_result(self, trans: List[int]) -> bool:
        if not trans:
            return False
        replay_result = self._result_from_trans(trans)
        replay_markings = replay_result.get_markings()
        if not replay_markings:
            return False
        replay_last = replay_markings[-1]
        if not self._is_goal(replay_last):
            return False
        replay_makespan = replay_last.get_prefix()
        if replay_makespan < self.best_train_makespan:
            self.best_train_makespan = replay_makespan
            self.best_train_trans = replay_result.get_trans()
            return True
        return False

    def _run_training_episode(self, epsilon: float, step_limit: int, total_steps_start: int):
        self._set_to_initial()
        curr_marking = self.petri_net.get_marking()
        trans = []
        seen_count: Dict[Tuple, int] = {}
        episode_steps = 0
        episode_updates = 0
        total_steps = total_steps_start
        total_loss_delta = 0.0
        update_count_delta = 0
        stop_info = make_stop_info("step_limit", 0, step_limit)
        for _ in range(step_limit):
            if self._is_goal(curr_marking):
                if curr_marking.get_prefix() < self.best_train_makespan:
                    self.best_train_makespan = curr_marking.get_prefix()
                    self.best_train_trans = trans.copy()
                stop_info = make_stop_info("goal", episode_steps, step_limit)
                break
            state = curr_marking.clone()
            s_key = self._state_key(state)
            seen_count[s_key] = seen_count.get(s_key, 0) + 1
            state_tensor = self.encoder.encode(state)
            mask = self._mask_from_marking(state)
            q_values = self.policy_net(state_tensor)
            action = self._select_action(q_values, mask, epsilon)
            action = self._safe_action(curr_marking, action)
            if action < 0:
                deadlock_reward = -self.reward_deadlock_penalty
                self.replay.add(Experience(state, 0, deadlock_reward, state.clone(), True), abs(deadlock_reward))
                stop_info = make_stop_info("deadlock", episode_steps, step_limit, self._deadlock_reason(state))
                break
            if not self.petri_net.enable(action):
                invalid_reward = -self.reward_deadlock_penalty
                self.replay.add(Experience(state, action, invalid_reward, state.clone(), True), abs(invalid_reward))
                stop_info = make_stop_info("invalid_action_fallback", episode_steps, step_limit)
                break
            next_marking = self.petri_net.launch(action)
            self.petri_net.set_marking(next_marking)
            done = self._is_goal(next_marking)
            next_mask = self._mask_from_marking(next_marking)
            deadlock = not bool(next_mask.any().item()) and (not done)
            repeat_penalty_count = seen_count.get(self._state_key(next_marking), 0)
            reward = self._calc_reward(curr_marking, next_marking, done, deadlock, repeat_penalty_count)
            terminal = bool(done or deadlock)
            self.replay.add(Experience(state, action, reward, next_marking.clone(), terminal), abs(reward) + 1e-3)
            trans.append(action)
            curr_marking = next_marking
            total_steps += 1
            episode_steps += 1
            if len(self.replay) >= self.batch_size and total_steps > self.warmup_steps and total_steps % self.learn_every == 0:
                loss = self._learn_batch()
                total_loss_delta += loss
                update_count_delta += 1
                episode_updates += 1
            if done:
                if curr_marking.get_prefix() < self.best_train_makespan:
                    self.best_train_makespan = curr_marking.get_prefix()
                    self.best_train_trans = trans.copy()
                stop_info = make_stop_info("goal", episode_steps, step_limit)
                break
            if deadlock:
                stop_info = make_stop_info("deadlock", episode_steps, step_limit, self._deadlock_reason(next_marking))
                break
        if episode_steps >= step_limit and str(stop_info.get("reason")) == "step_limit":
            stop_info = make_stop_info("step_limit", episode_steps, step_limit)
        next_epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
        return next_epsilon, episode_steps, episode_updates, total_steps, total_loss_delta, update_count_delta, stop_info, curr_marking

    def _estimate_rollout_success_rate(self, rollout_count: int) -> float:
        if rollout_count <= 0:
            return 0.0
        success = 0
        for i in range(rollout_count):
            epsilon = 0.0 if i == 0 else 0.12
            result, stop_info = self._single_rollout(epsilon)
            markings = result.get_markings()
            if markings:
                self._write_controller_log(markings[-1], "dqn_eval_rollout_" + str(i + 1))
            if markings and self._is_goal(markings[-1]):
                success += 1
            else:
                self._count_stop_reason(self.eval_failure_counts, stop_info)
                self._log_rollout_stop(
                    "[GCN-DQN-ENH] eval_rollout "
                    + str(i + 1)
                    + "/"
                    + str(rollout_count),
                    stop_info,
                )
        return float(success) / float(rollout_count)

    def _estimate_policy_success_rate(self, rollout_count: int):
        rollout_rate = self._estimate_rollout_success_rate(rollout_count)
        replay_rate = 0.0
        if self.best_train_trans:
            replay_result = self._result_from_trans(self.best_train_trans)
            replay_markings = replay_result.get_markings()
            if replay_markings and self._is_goal(replay_markings[-1]):
                replay_rate = 1.0
        return max(rollout_rate, replay_rate), rollout_rate, replay_rate

    def train_model(self):
        """训练阶段主循环。"""
        total_steps = 0
        epsilon = self.epsilon_init
        total_loss = 0.0
        update_count = 0
        train_goal_episodes = 0
        self.train_failure_counts = {}
        self.eval_failure_counts = {}
        self._log("[GCN-DQN-ENH] training start")
        for ep in range(self.train_episodes):
            step_limit = self._episode_step_limit(ep)
            epsilon, episode_steps, episode_updates, total_steps, loss_delta, update_delta, stop_info, final_marking = self._run_training_episode(
                epsilon, step_limit, total_steps
            )
            total_loss += loss_delta
            update_count += update_delta
            ep_idx = ep + 1
            self._write_controller_log(final_marking, "dqn_train_main_episode_" + str(ep_idx))
            if str(stop_info.get("reason")) == "goal":
                train_goal_episodes += 1
            else:
                self._count_stop_reason(self.train_failure_counts, stop_info)
            best_show = self.best_train_makespan if self.best_train_makespan < 2 ** 31 - 1 else -1
            avg_loss = (total_loss / update_count) if update_count > 0 else 0.0
            self._log(
                "[GCN-DQN-ENH] episode "
                + str(ep_idx)
                + "/"
                + str(self.train_episodes)
                + " phase=main"
                + " stop="
                + describe_stop_info(stop_info)
                + " steps="
                + str(episode_steps)
                + "/"
                + str(step_limit)
                + " updates="
                + str(episode_updates)
                + " replay="
                + str(len(self.replay))
                + " epsilon="
                + format(epsilon, ".4f")
                + " best_makespan="
                + str(best_show)
                + " avg_loss="
                + format(avg_loss, ".4f")
                + " goal_episodes="
                + str(train_goal_episodes)
                + " failure_counts="
                + format_reason_counts(self.train_failure_counts)
            )
        success_rate, rollout_success_rate, replay_success_rate = self._estimate_policy_success_rate(self.goal_eval_rollouts)
        extra_used = 0
        while success_rate < self.goal_min_success_rate and extra_used < self.extra_train_episodes:
            epsilon, episode_steps, episode_updates, total_steps, loss_delta, update_delta, stop_info, final_marking = self._run_training_episode(
                epsilon, self.max_steps_per_episode, total_steps
            )
            total_loss += loss_delta
            update_count += update_delta
            extra_used += 1
            self._write_controller_log(final_marking, "dqn_train_extra_episode_" + str(extra_used))
            if str(stop_info.get("reason")) == "goal":
                train_goal_episodes += 1
            else:
                self._count_stop_reason(self.train_failure_counts, stop_info)
            success_rate, rollout_success_rate, replay_success_rate = self._estimate_policy_success_rate(self.goal_eval_rollouts)
            best_show = self.best_train_makespan if self.best_train_makespan < 2 ** 31 - 1 else -1
            avg_loss = (total_loss / update_count) if update_count > 0 else 0.0
            self._log(
                "[GCN-DQN-ENH] episode "
                + str(extra_used)
                + "/"
                + str(self.extra_train_episodes)
                + " phase=extra"
                + " stop="
                + describe_stop_info(stop_info)
                + " steps="
                + str(episode_steps)
                + "/"
                + str(self.max_steps_per_episode)
                + " updates="
                + str(episode_updates)
                + " replay="
                + str(len(self.replay))
                + " epsilon="
                + format(epsilon, ".4f")
                + " best_makespan="
                + str(best_show)
                + " avg_loss="
                + format(avg_loss, ".4f")
                + " goal_episodes="
                + str(train_goal_episodes)
                + " failure_counts="
                + format_reason_counts(self.train_failure_counts)
                + " success_rate="
                + format(success_rate, ".3f")
                + " target="
                + format(self.goal_min_success_rate, ".3f")
                + " rollout_rate="
                + format(rollout_success_rate, ".3f")
                + " replay_rate="
                + format(replay_success_rate, ".3f")
            )
        if extra_used == 0:
            success_rate, rollout_success_rate, replay_success_rate = self._estimate_policy_success_rate(self.goal_eval_rollouts)
        self.extra_info["trainSteps"] = total_steps
        self.extra_info["bestTrainMakespan"] = self.best_train_makespan if self.best_train_makespan < 2 ** 31 - 1 else -1
        self.extra_info["bestTrainTransCount"] = len(self.best_train_trans)
        self.extra_info["avgLoss"] = (total_loss / update_count) if update_count > 0 else 0.0
        self.extra_info["trainSuccessRate"] = success_rate
        self.extra_info["rolloutSuccessRate"] = rollout_success_rate
        self.extra_info["replaySuccessRate"] = replay_success_rate
        self.extra_info["extraTrainEpisodes"] = extra_used
        self.extra_info["trainFailureCounts"] = dict(self.train_failure_counts)
        self.extra_info["evalFailureCounts"] = dict(self.eval_failure_counts)
        self.extra_info["deadlockControllerLogPath"] = self.deadlock_controller.log_path
        self.extra_info["deadlockControllerConfig"] = {
            "enableLookahead": self.deadlock_controller.enable_lookahead,
            "lookaheadDepth": self.deadlock_controller.lookahead_depth,
            "lookaheadWidth": self.deadlock_controller.lookahead_width,
            "lookaheadTriggerSafeLimit": self.deadlock_controller.lookahead_trigger_safe_limit,
            "lookaheadTriggerOnFBM": self.deadlock_controller.lookahead_trigger_on_fbm,
            "representationEnabled": self.controller_representation_enabled,
        }
        self.is_trained = True
        self._log("[GCN-DQN-ENH] training done train_steps=" + str(total_steps))

    def _result_from_trans(self, trans: List[int]) -> Result:
        """将 transition 序列回放为 Result。"""
        self._set_to_initial()
        curr_marking = self.petri_net.get_marking()
        markings = [curr_marking]
        used = []
        for action in trans:
            action = self._safe_action(curr_marking, action)
            if action < 0:
                break
            if not self.petri_net.enable(action):
                break
            next_marking = self.petri_net.launch(action)
            self.petri_net.set_marking(next_marking)
            used.append(action)
            markings.append(next_marking)
            curr_marking = next_marking
            if self._is_goal(curr_marking):
                break
        return Result(used, markings)

    def _single_rollout(self, epsilon: float):
        """按给定探索率执行一次 rollout。"""
        self._set_to_initial()
        curr_marking = self.petri_net.get_marking()
        trans = []
        markings = [curr_marking]
        stop_info = make_stop_info("step_limit", 0, self.inference_max_steps_per_episode)
        for _ in range(self.inference_max_steps_per_episode):
            if self._is_goal(curr_marking):
                stop_info = make_stop_info("goal", len(trans), self.inference_max_steps_per_episode)
                break
            with torch.no_grad():
                q = self.policy_net(self.encoder.encode(curr_marking))
                mask = self._mask_from_marking(curr_marking)
                action = self._select_action(q, mask, epsilon)
            action = self._safe_action(curr_marking, action)
            if action < 0:
                stop_info = make_stop_info("deadlock", len(trans), self.inference_max_steps_per_episode, self._deadlock_reason(curr_marking))
                break
            if not self.petri_net.enable(action):
                stop_info = make_stop_info("invalid_action_fallback", len(trans), self.inference_max_steps_per_episode)
                break
            next_marking = self.petri_net.launch(action)
            self.petri_net.set_marking(next_marking)
            trans.append(action)
            markings.append(next_marking)
            curr_marking = next_marking
            if self._is_goal(curr_marking):
                stop_info = make_stop_info("goal", len(trans), self.inference_max_steps_per_episode)
                break
            next_mask = self._mask_from_marking(next_marking)
            if not bool(next_mask.any().item()):
                stop_info = make_stop_info("deadlock", len(trans), self.inference_max_steps_per_episode, self._deadlock_reason(next_marking))
                break
        if len(trans) >= self.inference_max_steps_per_episode and str(stop_info.get("reason")) == "step_limit":
            stop_info = make_stop_info("step_limit", len(trans), self.inference_max_steps_per_episode)
        return Result(trans, markings), stop_info

    def _beam_search(self) -> Result:
        """用 Q 值引导的 beam search 做推理增强。"""
        self._set_to_initial()
        init = self.petri_net.get_marking().clone()
        frontier = [(init, [], float(init.get_prefix()))]
        visited = {self._state_key(init): float(init.get_prefix())}
        best = Result([], [init])
        progress_interval = max(1, min(10, self.inference_max_steps_per_episode // 10 if self.inference_max_steps_per_episode > 0 else 1))
        for depth in range(self.inference_max_steps_per_episode):
            candidates = []
            for marking, trans, _ in frontier:
                if self._is_goal(marking):
                    return Result(trans, [init])
                self.petri_net.set_marking(marking)
                mask = self._mask_from_marking(marking)
                enabled = torch.nonzero(mask, as_tuple=False).flatten()
                if enabled.numel() == 0:
                    continue
                with torch.no_grad():
                    q = self.policy_net(self.encoder.encode(marking))
                    q_masked = q.clone()
                    q_masked[~mask] = -1e9
                    k = min(self.beam_expand_per_node, enabled.numel())
                    top_vals, top_idx = torch.topk(q_masked, k=k)
                    _ = top_vals
                for idx in top_idx.tolist():
                    action = self._safe_action(marking, int(idx))
                    if action < 0:
                        continue
                    if not self.petri_net.enable(action):
                        continue
                    next_marking = self.petri_net.launch(action)
                    key = self._state_key(next_marking)
                    prefix = float(next_marking.get_prefix())
                    old = visited.get(key, 2 ** 31 - 1)
                    if prefix >= old:
                        continue
                    visited[key] = prefix
                    heuristic = float(self._goal_distance(next_marking))
                    score = prefix + heuristic * 2.0
                    candidates.append((next_marking.clone(), trans + [action], score))
                    if self._is_goal(next_marking):
                        return self._result_from_trans(trans + [action])
            if depth == 0 or depth + 1 == self.inference_max_steps_per_episode or (depth + 1) % progress_interval == 0:
                best_frontier_goal_distance = -1
                best_frontier_prefix = -1
                if frontier:
                    first_marking = frontier[0][0]
                    best_frontier_goal_distance = self._goal_distance(first_marking)
                    best_frontier_prefix = first_marking.get_prefix()
                self._log(
                    "[GCN-DQN-ENH] beam_search progress depth="
                    + str(depth + 1)
                    + "/"
                    + str(self.inference_max_steps_per_episode)
                    + " frontier="
                    + str(len(frontier))
                    + " candidates="
                    + str(len(candidates))
                    + " visited="
                    + str(len(visited))
                    + " best_frontier_goal_distance="
                    + str(best_frontier_goal_distance)
                    + " best_frontier_prefix="
                    + str(best_frontier_prefix)
                )
            if not candidates:
                self._log("[GCN-DQN-ENH] beam_search stop=no_candidates depth=" + str(depth + 1))
                break
            candidates.sort(key=lambda x: x[2])
            frontier = candidates[: self.beam_width]
            best = self._result_from_trans(frontier[0][1])
        return best

    def _pick_best(self, results: List[Result]) -> Result:
        """在候选结果中按可达性和代价选择最优。"""
        best = None
        best_tuple = None
        for r in results:
            markings = r.get_markings()
            if not markings:
                continue
            last = markings[-1]
            is_goal = self._is_goal(last)
            goal_dist = self._goal_distance(last)
            prefix = last.get_prefix()
            key = (0 if is_goal else 1, goal_dist, prefix, len(r.get_trans()))
            if best is None or key < best_tuple:
                best = r
                best_tuple = key
        return best if best is not None else Result([], [])

    def search(self):
        """入口：训练后融合多种推理策略并产出最终结果。"""
        if not self.is_trained:
            self.train_model()
        else:
            self._log("[GCN-DQN-ENH] training skipped (model already trained or loaded)")
        self._log("[GCN-DQN-ENH] inference start")
        candidates = []
        self._log("[GCN-DQN-ENH] inference stage=greedy_rollout begin")
        greedy_result, greedy_stop = self._single_rollout(0.0)
        candidates.append(greedy_result)
        if greedy_result.get_markings():
            self._write_controller_log(greedy_result.get_markings()[-1], "dqn_inference_greedy_rollout")
        self._log_rollout_stop("[GCN-DQN-ENH] inference stage=greedy_rollout", greedy_stop)
        self._log("[GCN-DQN-ENH] inference stage=greedy_rollout done")
        self._log("[GCN-DQN-ENH] inference stage=stochastic_rollout begin total=" + str(self.rollout_count))
        progress_interval = max(1, self.rollout_count // 5)
        for i in range(self.rollout_count):
            rollout_result, rollout_stop = self._single_rollout(0.20)
            candidates.append(rollout_result)
            if rollout_result.get_markings():
                self._write_controller_log(rollout_result.get_markings()[-1], "dqn_inference_stochastic_rollout_" + str(i + 1))
            self._log_rollout_stop("[GCN-DQN-ENH] inference stochastic_rollout=" + str(i + 1), rollout_stop)
            done = i + 1
            if done == self.rollout_count or done % progress_interval == 0:
                self._log(
                    "[GCN-DQN-ENH] inference stage=stochastic_rollout progress="
                    + str(done)
                    + "/"
                    + str(self.rollout_count)
                )
        self._log("[GCN-DQN-ENH] inference stage=stochastic_rollout done")
        if self.best_train_trans:
            self._log("[GCN-DQN-ENH] inference stage=replay_best_train begin")
            candidates.append(self._result_from_trans(self.best_train_trans))
            self._log("[GCN-DQN-ENH] inference stage=replay_best_train done")
        self._log("[GCN-DQN-ENH] inference stage=beam_search begin")
        candidates.append(self._beam_search())
        self._log("[GCN-DQN-ENH] inference stage=beam_search done")
        best = self._pick_best(candidates)
        markings = best.get_markings()
        self.extra_info["inferenceTransCount"] = len(best.get_trans())
        self.extra_info["inferenceMakespan"] = markings[-1].get_prefix() if markings else -1
        self.extra_info["reachGoal"] = self._is_goal(markings[-1]) if markings else False
        self.extra_info["goalDistance"] = self._goal_distance(markings[-1]) if markings else -1
        if markings and (not self.extra_info["reachGoal"]):
            last_marking = markings[-1]
            if self._mask_from_marking(last_marking).any().item():
                self.extra_info["inferenceStopReason"] = "step_limit"
            else:
                self.extra_info["inferenceStopReason"] = "deadlock"
                self.extra_info["inferenceDeadlockReason"] = self._deadlock_reason(last_marking)
        self._log(
            "[GCN-DQN-ENH] inference done reach_goal="
            + str(self.extra_info["reachGoal"])
            + " goal_distance="
            + str(self.extra_info["goalDistance"])
            + " makespan="
            + str(self.extra_info["inferenceMakespan"])
            + " trans_count="
            + str(self.extra_info["inferenceTransCount"])
        )
        return best

    def get_extra_info(self):
        """返回训练与推理统计信息。"""
        return self.extra_info
