from typing import Dict, List, Tuple
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

try:
    from .abstract_search import AbstractSearch
    from .deadlock_controller import DeadlockController
    from .petri_gcn_models import PetriNetGCNEnhanced, PetriRepresentationInput, PetriStateEncoderEnhanced
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
    from python_port.petri_net_platform.search.petri_gcn_models import (
        PetriNetGCNEnhanced,
        PetriRepresentationInput,
        PetriStateEncoderEnhanced,
    )
    from python_port.petri_net_platform.search.rl_env_semantics import (
        describe_stop_info,
        format_reason_counts,
        make_stop_info,
        stop_info_label,
    )
    from python_port.petri_net_platform.utils.result import Result


class PetriNetGCNActorCritic(nn.Module):
    # actor-critic网络
    def __init__(
        self,
        pre,
        post,
        lambda_p: int,
        lambda_t: int,
        num_layers: int = 3,
        end=None,
        min_delay_p=None,
        min_delay_t=None,
        capacity=None,
        max_residence_time=None,
        place_from_places=None,
    ):
        super().__init__()
        # 初始化actor网络
        self.actor_net = PetriNetGCNEnhanced(
            pre,
            post,
            lambda_p,
            lambda_t,
            num_layers,
            end=end,
            min_delay_p=min_delay_p,
            min_delay_t=min_delay_t,
            capacity=capacity,
            max_residence_time=max_residence_time,
            place_from_places=place_from_places,
        )
        readout_dim = self.actor_net.backbone.graph_readout[0].out_features
        # 初始化value头
        self.value_head = nn.Sequential(
            nn.Linear(readout_dim, readout_dim),
            nn.GELU(),
            nn.Linear(readout_dim, 1),
        )

    def forward(self, x_p):
        # 拆分输入为库所特征和变迁特征
        place_features, transition_features = self.actor_net._split_inputs(x_p)
        # 适配特征维度
        place_features = self.actor_net._adapt_place_features(place_features)
        transition_features = self.actor_net._adapt_transition_features(transition_features)
        # 处理缺失的变迁特征
        if transition_features is None:
            if place_features.dim() == 2:
                transition_features = self.actor_net.transition_seed
            else:
                transition_features = self.actor_net.transition_seed.unsqueeze(0).expand(place_features.shape[0], -1, -1)
        # 获取标识
        rep = self.actor_net.backbone(place_features, transition_features)
        values = self.value_head(rep.graph_embedding).squeeze(-1)
        return rep.transition_logits, values


class PetriNetGCNPPOEnhanced(AbstractSearch):
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
        gae_lambda: float = 0.95,
        lr: float = 3e-4,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 2.0,

        train_iterations: int = 48,
        rollout_episodes_per_iter: int = 12,
        ppo_update_epochs: int = 6,
        min_steps_per_episode: int = 80,
        max_steps_per_episode: int = 420,
        inference_max_steps_per_episode: int = None,

        reward_goal_bonus: float = 240.0,
        reward_deadlock_penalty: float = 80.0,
        reward_progress_weight: float = 1.8,
        reward_repeat_penalty: float = 0.2,
        use_reward_scaling: bool = True,
        reward_time_scale: float = 1000.0,
        use_reward_clip: bool = True,
        reward_clip_abs: float = 20.0,

        goal_eval_rollouts: int = 1,
        goal_min_success_rate: float = 0.5,
        extra_train_iterations: int = 0,
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
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.value_loss_coef = float(value_loss_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.train_iterations = max(1, int(train_iterations))
        self.rollout_episodes_per_iter = max(1, int(rollout_episodes_per_iter))
        self.ppo_update_epochs = max(1, int(ppo_update_epochs))
        self.min_steps_per_episode = max(1, int(min_steps_per_episode))
        self.max_steps_per_episode = max(self.min_steps_per_episode, int(max_steps_per_episode))
        if inference_max_steps_per_episode is None:
            inference_max_steps_per_episode = self.max_steps_per_episode
        self.inference_max_steps_per_episode = max(1, int(inference_max_steps_per_episode))
        self.reward_goal_bonus = float(reward_goal_bonus)
        self.reward_deadlock_penalty = float(reward_deadlock_penalty)
        self.reward_progress_weight = float(reward_progress_weight)
        self.reward_repeat_penalty = float(reward_repeat_penalty)
        self.use_reward_scaling = bool(use_reward_scaling)
        self.reward_time_scale = max(1.0, float(reward_time_scale))
        self.use_reward_clip = bool(use_reward_clip)
        self.reward_clip_abs = max(0.5, float(reward_clip_abs))
        self.goal_eval_rollouts = max(1, int(goal_eval_rollouts))
        self.goal_min_success_rate = max(0.0, min(1.0, float(goal_min_success_rate)))
        self.extra_train_iterations = max(0, int(extra_train_iterations))
        self.verbose = bool(verbose)
        self.log_interval = max(1, int(log_interval))
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
        self.model = PetriNetGCNActorCritic(
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
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
            controller_name="gcn_ppo",
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
        self.petri_net = self.initial_petri_net.clone()

    def _is_goal(self, marking) -> bool:
        p_info = marking.get_p_info()
        for i in range(len(p_info)):
            if self.end[i] == -1:
                continue
            if p_info[i] != self.end[i]:
                return False
        return True

    def _goal_distance(self, marking) -> int:
        p_info = marking.get_p_info()
        dist = 0
        for i in range(len(p_info)):
            if self.end[i] == -1:
                continue
            dist += abs(p_info[i] - self.end[i])
        return dist

    def _state_key(self, marking) -> Tuple:
        over = bool(getattr(marking, "over_max_residence_time", False))
        return tuple(marking.get_p_info()) + (1 if over else 0,)

    def _mask_from_marking(self, marking) -> torch.Tensor:
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

    def _safe_action(self, marking, action: int) -> int:
        if action < 0:
            return -1
        if action >= self.petri_net.get_trans_count():
            return -1
        if hasattr(marking, "curr_delay_t") and action >= len(marking.curr_delay_t):
            return -1
        return action

    def _calc_reward(self, curr_marking, next_marking, done: bool, deadlock: bool, repeat_count: int) -> float:
        delta_t = float(next_marking.get_prefix() - curr_marking.get_prefix())
        progress = float(self._goal_distance(curr_marking) - self._goal_distance(next_marking))
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

    def _episode_step_limit(self, iteration_idx: int) -> int:
        if self.train_iterations <= 1:
            return self.max_steps_per_episode
        ratio = float(iteration_idx) / float(self.train_iterations - 1)
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

    def _masked_logits(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = logits.clone()
        masked[~mask] = -1e9
        return masked

    def _select_action(self, logits: torch.Tensor, mask: torch.Tensor, greedy: bool):
        enabled = torch.nonzero(mask, as_tuple=False).flatten()
        if enabled.numel() == 0:
            return -1, torch.zeros((), dtype=torch.float32, device=self.device)
        masked_logits = self._masked_logits(logits, mask)
        if greedy:
            action = int(torch.argmax(masked_logits).item())
            log_prob = torch.log_softmax(masked_logits, dim=-1)[action]
            return action, log_prob
        dist = Categorical(logits=masked_logits)
        action_t = dist.sample()
        return int(action_t.item()), dist.log_prob(action_t)

    def _compute_gae(self, rewards: List[float], values: List[float], dones: List[bool], last_value: float):
        advantages = [0.0] * len(rewards)
        returns = [0.0] * len(rewards)
        gae = 0.0
        next_value = float(last_value)
        for idx in range(len(rewards) - 1, -1, -1):
            done = 1.0 if dones[idx] else 0.0
            delta = float(rewards[idx]) + self.gamma * next_value * (1.0 - done) - float(values[idx])
            gae = delta + self.gamma * self.gae_lambda * (1.0 - done) * gae
            advantages[idx] = gae
            returns[idx] = gae + float(values[idx])
            next_value = float(values[idx])
        return returns, advantages

    def _collect_training_episode(self, step_limit: int):
        self._set_to_initial()
        curr_marking = self.petri_net.get_marking()
        trans = []
        seen_count: Dict[Tuple, int] = {}
        states = []
        masks = []
        actions = []
        rewards = []
        dones = []
        old_log_probs = []
        values = []
        episode_steps = 0
        terminal = False
        stop_info = make_stop_info("step_limit", 0, step_limit)
        self.model.eval()
        for _ in range(step_limit):
            if self._is_goal(curr_marking):
                if curr_marking.get_prefix() < self.best_train_makespan:
                    self.best_train_makespan = curr_marking.get_prefix()
                    self.best_train_trans = trans.copy()
                terminal = True
                stop_info = make_stop_info("goal", episode_steps, step_limit)
                break
            state = curr_marking.clone()
            s_key = self._state_key(state)
            seen_count[s_key] = seen_count.get(s_key, 0) + 1
            with torch.no_grad():
                logits, value = self.model(self.encoder.encode(state))
                mask = self._mask_from_marking(state)
                action, log_prob = self._select_action(logits, mask, greedy=False)
            action = self._safe_action(curr_marking, action)
            if action < 0:
                terminal = True
                stop_info = make_stop_info("deadlock", episode_steps, step_limit, self._deadlock_reason(state))
                break
            if not self.petri_net.enable(action):
                terminal = True
                stop_info = make_stop_info("invalid_action_fallback", episode_steps, step_limit)
                break
            next_marking = self.petri_net.launch(action)
            self.petri_net.set_marking(next_marking)
            done = self._is_goal(next_marking)
            next_mask = self._mask_from_marking(next_marking)
            deadlock = not bool(next_mask.any().item()) and (not done)
            repeat_penalty_count = seen_count.get(self._state_key(next_marking), 0)
            reward = self._calc_reward(curr_marking, next_marking, done, deadlock, repeat_penalty_count)
            states.append(state)
            masks.append(mask.detach().clone())
            actions.append(action)
            rewards.append(float(reward))
            dones.append(bool(done or deadlock))
            old_log_probs.append(float(log_prob.item()))
            values.append(float(value.item()))
            trans.append(action)
            curr_marking = next_marking
            episode_steps += 1
            if done:
                if curr_marking.get_prefix() < self.best_train_makespan:
                    self.best_train_makespan = curr_marking.get_prefix()
                    self.best_train_trans = trans.copy()
                terminal = True
                stop_info = make_stop_info("goal", episode_steps, step_limit)
                break
            if deadlock:
                terminal = True
                stop_info = make_stop_info("deadlock", episode_steps, step_limit, self._deadlock_reason(next_marking))
                break
        if episode_steps >= step_limit and str(stop_info.get("reason")) == "step_limit":
            stop_info = make_stop_info("step_limit", episode_steps, step_limit)
        last_value = 0.0
        if (not terminal) and episode_steps > 0:
            with torch.no_grad():
                _, bootstrap_value = self.model(self.encoder.encode(curr_marking))
                last_value = float(bootstrap_value.item())
        returns, advantages = self._compute_gae(rewards, values, dones, last_value) if episode_steps > 0 else ([], [])
        return {
            "states": states,
            "masks": masks,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "old_log_probs": old_log_probs,
            "values": values,
            "returns": returns,
            "advantages": advantages,
            "trans": trans,
            "episode_steps": episode_steps,
            "stop_info": stop_info,
            "final_marking": curr_marking,
        }

    def _update_from_episodes(self, episodes) -> Dict[str, float]:
        states = []
        mask_list = []
        actions = []
        old_log_probs = []
        returns = []
        advantages = []
        for episode in episodes:
            states.extend(episode["states"])
            mask_list.extend(episode["masks"])
            actions.extend(episode["actions"])
            old_log_probs.extend(episode["old_log_probs"])
            returns.extend(episode["returns"])
            advantages.extend(episode["advantages"])
        if not states:
            return {"avg_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "sample_count": 0}
        encoded = self.encoder.encode_batch(states)
        masks = torch.stack(mask_list, dim=0).to(self.device)
        action_t = torch.tensor(actions, dtype=torch.int64, device=self.device)
        old_log_prob_t = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        return_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantage_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        if advantage_t.numel() > 1:
            advantage_t = (advantage_t - advantage_t.mean()) / advantage_t.std(unbiased=False).clamp_min(1e-6)
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0
        for _ in range(self.ppo_update_epochs):
            logits, values = self.model(encoded)
            masked_logits = logits.masked_fill(~masks, -1e9)
            dist = Categorical(logits=masked_logits)
            new_log_probs = dist.log_prob(action_t)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_log_probs - old_log_prob_t)
            surr1 = ratio * advantage_t
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantage_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, return_t)
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total_loss += float(loss.item())
            total_policy += float(policy_loss.item())
            total_value += float(value_loss.item())
            total_entropy += float(entropy.item())
        denom = float(self.ppo_update_epochs)
        return {
            "avg_loss": total_loss / denom,
            "policy_loss": total_policy / denom,
            "value_loss": total_value / denom,
            "entropy": total_entropy / denom,
            "sample_count": len(states),
        }

    def _estimate_rollout_success_rate(self, rollout_count: int) -> float:
        if rollout_count <= 0:
            return 0.0
        success = 0
        for i in range(rollout_count):
            result, stop_info = self._single_rollout(greedy=True)
            markings = result.get_markings()
            if markings:
                self._write_controller_log(markings[-1], "ppo_eval_rollout_" + str(i + 1))
            if markings and self._is_goal(markings[-1]):
                success += 1
            else:
                self._count_stop_reason(self.eval_failure_counts, stop_info)
                self._log_rollout_stop(
                    "[GCN-PPO] eval_rollout "
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
        total_steps = 0
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_updates = 0
        train_goal_rollouts = 0
        self.train_failure_counts = {}
        self.eval_failure_counts = {}
        self._log("[GCN-PPO] training start")
        for it in range(self.train_iterations):
            step_limit = self._episode_step_limit(it)
            episodes = []
            iter_steps = 0
            iter_stop_counts: Dict[str, int] = {}
            for rollout_idx in range(self.rollout_episodes_per_iter):
                episode = self._collect_training_episode(step_limit)
                episodes.append(episode)
                iter_steps += episode["episode_steps"]
                total_steps += episode["episode_steps"]
                stop_info = episode["stop_info"]
                self._write_controller_log(
                    episode["final_marking"],
                    "ppo_train_main_iter_" + str(it + 1) + "_rollout_" + str(rollout_idx + 1),
                )
                if str(stop_info.get("reason")) == "goal":
                    train_goal_rollouts += 1
                else:
                    self._count_stop_reason(self.train_failure_counts, stop_info)
                label = stop_info_label(stop_info)
                iter_stop_counts[label] = iter_stop_counts.get(label, 0) + 1
            update_metrics = self._update_from_episodes(episodes)
            total_loss += update_metrics["avg_loss"]
            total_policy_loss += update_metrics["policy_loss"]
            total_value_loss += update_metrics["value_loss"]
            total_entropy += update_metrics["entropy"]
            total_updates += 1
            iter_idx = it + 1
            best_show = self.best_train_makespan if self.best_train_makespan < 2 ** 31 - 1 else -1
            avg_steps = float(iter_steps) / float(max(1, self.rollout_episodes_per_iter))
            self._log(
                "[GCN-PPO] iter "
                + str(iter_idx)
                + "/"
                + str(self.train_iterations)
                + " phase=main"
                + " rollouts="
                + str(self.rollout_episodes_per_iter)
                + " steps="
                + str(iter_steps)
                + " avg_steps="
                + format(avg_steps, ".1f")
                + " samples="
                + str(update_metrics["sample_count"])
                + " avg_loss="
                + format(update_metrics["avg_loss"], ".4f")
                + " policy_loss="
                + format(update_metrics["policy_loss"], ".4f")
                + " value_loss="
                + format(update_metrics["value_loss"], ".4f")
                + " entropy="
                + format(update_metrics["entropy"], ".4f")
                + " best_makespan="
                + str(best_show)
                + " goal_rollouts="
                + str(train_goal_rollouts)
                + " stop_summary="
                + format_reason_counts(iter_stop_counts)
                + " failure_counts="
                + format_reason_counts(self.train_failure_counts)
            )
        success_rate, rollout_success_rate, replay_success_rate = self._estimate_policy_success_rate(self.goal_eval_rollouts)
        extra_used = 0
        while success_rate < self.goal_min_success_rate and extra_used < self.extra_train_iterations:
            episodes = []
            iter_steps = 0
            iter_stop_counts = {}
            for rollout_idx in range(self.rollout_episodes_per_iter):
                episode = self._collect_training_episode(self.max_steps_per_episode)
                episodes.append(episode)
                iter_steps += episode["episode_steps"]
                total_steps += episode["episode_steps"]
                stop_info = episode["stop_info"]
                self._write_controller_log(
                    episode["final_marking"],
                    "ppo_train_extra_iter_" + str(extra_used + 1) + "_rollout_" + str(rollout_idx + 1),
                )
                if str(stop_info.get("reason")) == "goal":
                    train_goal_rollouts += 1
                else:
                    self._count_stop_reason(self.train_failure_counts, stop_info)
                label = stop_info_label(stop_info)
                iter_stop_counts[label] = iter_stop_counts.get(label, 0) + 1
            update_metrics = self._update_from_episodes(episodes)
            total_loss += update_metrics["avg_loss"]
            total_policy_loss += update_metrics["policy_loss"]
            total_value_loss += update_metrics["value_loss"]
            total_entropy += update_metrics["entropy"]
            total_updates += 1
            extra_used += 1
            success_rate, rollout_success_rate, replay_success_rate = self._estimate_policy_success_rate(self.goal_eval_rollouts)
            best_show = self.best_train_makespan if self.best_train_makespan < 2 ** 31 - 1 else -1
            avg_steps = float(iter_steps) / float(max(1, self.rollout_episodes_per_iter))
            self._log(
                "[GCN-PPO] iter "
                + str(extra_used)
                + "/"
                + str(self.extra_train_iterations)
                + " phase=extra"
                + " rollouts="
                + str(self.rollout_episodes_per_iter)
                + " steps="
                + str(iter_steps)
                + " avg_steps="
                + format(avg_steps, ".1f")
                + " samples="
                + str(update_metrics["sample_count"])
                + " avg_loss="
                + format(update_metrics["avg_loss"], ".4f")
                + " policy_loss="
                + format(update_metrics["policy_loss"], ".4f")
                + " value_loss="
                + format(update_metrics["value_loss"], ".4f")
                + " entropy="
                + format(update_metrics["entropy"], ".4f")
                + " best_makespan="
                + str(best_show)
                + " goal_rollouts="
                + str(train_goal_rollouts)
                + " stop_summary="
                + format_reason_counts(iter_stop_counts)
                + " failure_counts="
                + format_reason_counts(self.train_failure_counts)
                + " success_rate="
                + format(success_rate, ".3f")
                + " rollout_rate="
                + format(rollout_success_rate, ".3f")
                + " replay_rate="
                + format(replay_success_rate, ".3f")
                + " target="
                + format(self.goal_min_success_rate, ".3f")
            )
        self.extra_info["trainSteps"] = total_steps
        self.extra_info["bestTrainMakespan"] = self.best_train_makespan if self.best_train_makespan < 2 ** 31 - 1 else -1
        self.extra_info["bestTrainTransCount"] = len(self.best_train_trans)
        self.extra_info["avgLoss"] = (total_loss / total_updates) if total_updates > 0 else 0.0
        self.extra_info["policyLoss"] = (total_policy_loss / total_updates) if total_updates > 0 else 0.0
        self.extra_info["valueLoss"] = (total_value_loss / total_updates) if total_updates > 0 else 0.0
        self.extra_info["entropy"] = (total_entropy / total_updates) if total_updates > 0 else 0.0
        self.extra_info["trainSuccessRate"] = success_rate
        self.extra_info["rolloutSuccessRate"] = rollout_success_rate
        self.extra_info["replaySuccessRate"] = replay_success_rate
        self.extra_info["extraTrainIterations"] = extra_used
        self.extra_info["ppoUpdateEpochs"] = self.ppo_update_epochs
        self.extra_info["rolloutEpisodesPerIter"] = self.rollout_episodes_per_iter
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
        self._log("[GCN-PPO] training done train_steps=" + str(total_steps))

    def _result_from_trans(self, trans: List[int]) -> Result:
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

    def _single_rollout(self, greedy: bool = True):
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
                logits, _ = self.model(self.encoder.encode(curr_marking))
                mask = self._mask_from_marking(curr_marking)
                action, _ = self._select_action(logits, mask, greedy=greedy)
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

    def _pick_best(self, results: List[Result]) -> Result:
        best = None
        best_tuple = None
        for result in results:
            markings = result.get_markings()
            if not markings:
                continue
            last = markings[-1]
            is_goal = self._is_goal(last)
            goal_dist = self._goal_distance(last)
            prefix = last.get_prefix()
            key = (0 if is_goal else 1, goal_dist, prefix, len(result.get_trans()))
            if best is None or key < best_tuple:
                best = result
                best_tuple = key
        return best if best is not None else Result([], [])

    def search(self):
        if not self.is_trained:
            self.train_model()
        else:
            self._log("[GCN-PPO] training skipped (model already trained or loaded)")
        self._log("[GCN-PPO] inference start")
        greedy_result, greedy_stop = self._single_rollout(greedy=True)
        candidates = [greedy_result]
        if greedy_result.get_markings():
            self._write_controller_log(greedy_result.get_markings()[-1], "ppo_inference_greedy_rollout")
        self._log_rollout_stop("[GCN-PPO] inference greedy_rollout", greedy_stop)
        if self.best_train_trans:
            self._log("[GCN-PPO] inference stage=replay_best_train")
            candidates.append(self._result_from_trans(self.best_train_trans))
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
            "[GCN-PPO] inference done reach_goal="
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
        return self.extra_info


class PetriNetGCNPPOEnhancedHQ(PetriNetGCNPPOEnhanced):
    def __init__(
        self,
        petri_net,
        end: List[int],
        pre: List[List[int]],
        post: List[List[int]],
        min_delay_p: List[int],
        train_iterations: int = 48,
        rollout_episodes_per_iter: int = 12,
        ppo_update_epochs: int = 6,
        min_steps_per_episode: int = 120,
        max_steps_per_episode: int = 700,
        goal_eval_rollouts: int = 1,
        goal_min_success_rate: float = 0.7,
        extra_train_iterations: int = 18,
        use_reward_scaling: bool = True,
        reward_time_scale: float = 1000.0,
        use_reward_clip: bool = True,
        reward_clip_abs: float = 20.0,
        verbose: bool = True,
        log_interval: int = 5,
        **kwargs,
    ):
        if not hasattr(petri_net, "max_residence_time"):
            raise ValueError("petri_net 缺少 max_residence_time，无法使用 PPO HQ 版本。")
        super().__init__(
            petri_net=petri_net,
            end=end,
            pre=pre,
            post=post,
            min_delay_p=min_delay_p,
            lambda_p=128,
            lambda_t=32,
            num_layers=4,
            gamma=0.99,
            gae_lambda=0.95,
            lr=3e-4,
            clip_ratio=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=2.0,
            train_iterations=train_iterations,
            rollout_episodes_per_iter=rollout_episodes_per_iter,
            ppo_update_epochs=ppo_update_epochs,
            min_steps_per_episode=min_steps_per_episode,
            max_steps_per_episode=max_steps_per_episode,
            reward_goal_bonus=320.0,
            reward_deadlock_penalty=120.0,
            reward_progress_weight=2.2,
            reward_repeat_penalty=0.25,
            use_reward_scaling=use_reward_scaling,
            reward_time_scale=reward_time_scale,
            use_reward_clip=use_reward_clip,
            reward_clip_abs=reward_clip_abs,
            goal_eval_rollouts=goal_eval_rollouts,
            goal_min_success_rate=goal_min_success_rate,
            extra_train_iterations=extra_train_iterations,
            verbose=verbose,
            log_interval=log_interval,
            **kwargs,
        )
