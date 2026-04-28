from typing import Dict, List
import os
import sys

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    from .petri_gcn_models import PetriRepresentationInput
    from .petri_net_gcn_ppo import PetriNetGCNPPOEnhanced
    from .rl_env_semantics import format_reason_counts, make_stop_info, stop_info_label
except ImportError:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from python_port.petri_net_platform.search.petri_gcn_models import PetriRepresentationInput
    from python_port.petri_net_platform.search.petri_net_gcn_ppo import PetriNetGCNPPOEnhanced
    from python_port.petri_net_platform.search.rl_env_semantics import format_reason_counts, make_stop_info, stop_info_label


class RolloutBuffer:
    def __init__(self):
        self.states: List[object] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.returns: List[float] = []
        self.advantages: List[float] = []
        self.masks: List[torch.Tensor] = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.returns.clear()
        self.advantages.clear()
        self.masks.clear()

    def append_episode(
        self,
        states: List[object],
        actions: List[int],
        log_probs: List[float],
        returns: List[float],
        advantages: List[float],
        masks: List[torch.Tensor],
    ):
        self.states.extend(states)
        self.actions.extend(actions)
        self.log_probs.extend(log_probs)
        self.returns.extend(returns)
        self.advantages.extend(advantages)
        self.masks.extend(masks)

    def __len__(self) -> int:
        return len(self.states)


class PetriNetGCNPPOClassic(PetriNetGCNPPOEnhanced):
    def __init__(
        self,
        *args,
        steps_per_epoch: int = 1024,
        minibatch_size: int = 128,
        target_kl: float = 0.04,
        entropy_coef_start: float = None,
        entropy_coef_end: float = None,
        temperature_start: float = 1.0,
        temperature_end: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if entropy_coef_start is None:
            entropy_coef_start = float(self.entropy_coef)
        if entropy_coef_end is None:
            entropy_coef_end = float(self.entropy_coef)
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.minibatch_size = max(1, int(minibatch_size))
        self.target_kl = max(0.0, float(target_kl))
        self.entropy_coef_start = max(0.0, float(entropy_coef_start))
        self.entropy_coef_end = max(0.0, float(entropy_coef_end))
        self.current_entropy_coef = self.entropy_coef_start
        self.temperature_start = max(1e-3, float(temperature_start))
        self.temperature_end = max(1e-3, float(temperature_end))
        self.current_temperature = self.temperature_start
        self.rollout_buffer = RolloutBuffer()

    def _select_action(self, logits: torch.Tensor, mask: torch.Tensor, greedy: bool):
        enabled = torch.nonzero(mask, as_tuple=False).flatten()
        if enabled.numel() == 0:
            return -1, torch.zeros((), dtype=torch.float32, device=self.device)
        masked_logits = self._masked_logits(logits, mask)
        if greedy:
            action = int(torch.argmax(masked_logits).item())
            log_prob = torch.log_softmax(masked_logits, dim=-1)[action]
            return action, log_prob
        scaled_logits = masked_logits / max(1e-3, float(self.current_temperature))
        dist = Categorical(logits=scaled_logits)
        action_t = dist.sample()
        return int(action_t.item()), dist.log_prob(action_t)

    def _slice_encoded(self, encoded, indices: torch.Tensor):
        if isinstance(encoded, PetriRepresentationInput):
            return PetriRepresentationInput(
                place_features=encoded.place_features.index_select(0, indices),
                transition_features=encoded.transition_features.index_select(0, indices),
            )
        if isinstance(encoded, dict) and "place_features" in encoded and "transition_features" in encoded:
            return {
                "place_features": encoded["place_features"].index_select(0, indices),
                "transition_features": encoded["transition_features"].index_select(0, indices),
            }
        return encoded.index_select(0, indices)

    def _bootstrap_value(self, marking) -> float:
        with torch.no_grad():
            _, bootstrap_value = self.model(self.encoder.encode(marking))
            return float(bootstrap_value.item())

    def _finalize_episode(
        self,
        episode_states: List[object],
        episode_masks: List[torch.Tensor],
        episode_actions: List[int],
        episode_rewards: List[float],
        episode_dones: List[bool],
        episode_log_probs: List[float],
        episode_values: List[float],
        last_value: float,
    ):
        if not episode_states:
            return
        returns, advantages = self._compute_gae(episode_rewards, episode_values, episode_dones, last_value)
        self.rollout_buffer.append_episode(
            states=episode_states,
            actions=episode_actions,
            log_probs=episode_log_probs,
            returns=returns,
            advantages=advantages,
            masks=episode_masks,
        )
    def _collect_rollout_steps(self, num_steps: int, step_limit: int, phase_tag: str, iter_index: int):
        self.rollout_buffer.clear()
        self._set_to_initial()
        curr_marking = self.petri_net.get_marking()
        seen_count: Dict[tuple, int] = {}
        episode_states: List[object] = []
        episode_masks: List[torch.Tensor] = []
        episode_actions: List[int] = []
        episode_rewards: List[float] = []
        episode_dones: List[bool] = []
        episode_log_probs: List[float] = []
        episode_values: List[float] = []
        current_ep_trans: List[int] = []
        current_ep_reward = 0.0
        episode_steps = 0
        steps_collected = 0
        episode_count = 0
        goal_rollouts = 0
        stop_counts: Dict[str, int] = {}
        ep_rewards: List[float] = []
        ep_makespans: List[float] = []
        empty_episode_guard = 0
        #==============================
        self.model.eval()
        
        while steps_collected < num_steps:
            state = curr_marking.clone()
            state_key = self._state_key(state)
            seen_count[state_key] = seen_count.get(state_key, 0) + 1

            with torch.no_grad():
                logits, value = self.model(self.encoder.encode(state))
                mask = self._mask_from_marking(state)
                action, log_prob = self._select_action(logits, mask, greedy=False)
            action = self._safe_action(curr_marking, action)

            stop_info = None
            advance_state = False
            next_marking = curr_marking

            if action < 0:
                stop_info = make_stop_info("deadlock", episode_steps, step_limit, self._deadlock_reason(state))
            elif not self.petri_net.enable(action):
                stop_info = make_stop_info("invalid_action_fallback", episode_steps, step_limit)
            else:
                next_marking = self.petri_net.launch(action)
                self.petri_net.set_marking(next_marking)
                done = self._is_goal(next_marking)
                next_mask = self._mask_from_marking(next_marking)
                deadlock = not bool(next_mask.any().item()) and (not done)
                repeat_penalty_count = seen_count.get(self._state_key(next_marking), 0)
                reward = self._calc_reward(curr_marking, next_marking, done, deadlock, repeat_penalty_count)
                episode_states.append(state)
                episode_masks.append(mask.detach().clone())
                episode_actions.append(action)
                episode_rewards.append(float(reward))
                episode_dones.append(bool(done or deadlock))
                episode_log_probs.append(float(log_prob.item()))
                episode_values.append(float(value.item()))
                current_ep_trans.append(action)
                current_ep_reward += float(reward)
                episode_steps += 1
                steps_collected += 1
                curr_marking = next_marking
                advance_state = True
                if done:
                    if curr_marking.get_prefix() < self.best_train_makespan:
                        self.best_train_makespan = curr_marking.get_prefix()
                        self.best_train_trans = current_ep_trans.copy()
                    stop_info = make_stop_info("goal", episode_steps, step_limit)
                elif deadlock:
                    stop_info = make_stop_info("deadlock", episode_steps, step_limit, self._deadlock_reason(next_marking))
                elif episode_steps >= step_limit:
                    stop_info = make_stop_info("step_limit", episode_steps, step_limit)

            if stop_info is None and steps_collected >= num_steps:
                stop_info = make_stop_info("buffer_limit", episode_steps, step_limit)

            if stop_info is None:
                continue

            reason = str(stop_info.get("reason"))
            is_buffer_cut = reason == "buffer_limit"
            if not is_buffer_cut:
                episode_count += 1
                label = stop_info_label(stop_info)
                stop_counts[label] = stop_counts.get(label, 0) + 1

            if episode_states:
                last_value = 0.0
                if reason not in {"goal", "deadlock", "invalid_action_fallback"} and advance_state:
                    last_value = self._bootstrap_value(curr_marking)
                self._finalize_episode(
                    episode_states=episode_states,
                    episode_masks=episode_masks,
                    episode_actions=episode_actions,
                    episode_rewards=episode_rewards,
                    episode_dones=episode_dones,
                    episode_log_probs=episode_log_probs,
                    episode_values=episode_values,
                    last_value=last_value,
                )
                if not is_buffer_cut:
                    ep_rewards.append(current_ep_reward)
                if reason == "goal":
                    goal_rollouts += 1
                    ep_makespans.append(float(curr_marking.get_prefix()))
                empty_episode_guard = 0
            else:
                empty_episode_guard += 1

            if (not is_buffer_cut) and reason != "goal":
                self._count_stop_reason(self.train_failure_counts, stop_info)

            if not is_buffer_cut:
                self._write_controller_log(
                    curr_marking,
                    "ppo_classic_" + phase_tag + "_iter_" + str(iter_index) + "_episode_" + str(episode_count),
                )

            if empty_episode_guard > max(8, num_steps):
                break

            if is_buffer_cut:
                break

            self._set_to_initial()
            curr_marking = self.petri_net.get_marking()
            seen_count = {}
            episode_states = []
            episode_masks = []
            episode_actions = []
            episode_rewards = []
            episode_dones = []
            episode_log_probs = []
            episode_values = []
            current_ep_trans = []
            current_ep_reward = 0.0
            episode_steps = 0

        return {
            "steps_collected": steps_collected,
            "episode_count": episode_count,
            "goal_rollouts": goal_rollouts,
            "stop_counts": stop_counts,
            "episode_rewards": ep_rewards,
            "episode_makespans": ep_makespans,
            "sample_count": len(self.rollout_buffer),
        }

    def _update_from_rollout_buffer(self) -> Dict[str, float]:
        if len(self.rollout_buffer) == 0:
            return {
                "avg_loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "sample_count": 0,
                "approx_kl": 0.0,
            }

        encoded = self.encoder.encode_batch(self.rollout_buffer.states)
        masks = torch.stack(self.rollout_buffer.masks, dim=0).to(self.device)
        action_t = torch.tensor(self.rollout_buffer.actions, dtype=torch.int64, device=self.device)
        old_log_prob_t = torch.tensor(self.rollout_buffer.log_probs, dtype=torch.float32, device=self.device)
        return_t = torch.tensor(self.rollout_buffer.returns, dtype=torch.float32, device=self.device)
        advantage_t = torch.tensor(self.rollout_buffer.advantages, dtype=torch.float32, device=self.device)
        if advantage_t.numel() > 1:
            advantage_t = (advantage_t - advantage_t.mean()) / advantage_t.std(unbiased=False).clamp_min(1e-6)

        dataset_size = int(action_t.shape[0])
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0
        approx_kl = 0.0
        update_count = 0
        stop_early = False

        for _ in range(self.ppo_update_epochs):
            permutation = torch.randperm(dataset_size, device=self.device)
            for start in range(0, dataset_size, self.minibatch_size):
                indices = permutation[start:start + self.minibatch_size]
                batch_encoded = self._slice_encoded(encoded, indices)
                batch_masks = masks.index_select(0, indices)
                batch_actions = action_t.index_select(0, indices)
                batch_old_log_probs = old_log_prob_t.index_select(0, indices)
                batch_returns = return_t.index_select(0, indices)
                batch_advantages = advantage_t.index_select(0, indices)

                logits, values = self.model(batch_encoded)
                masked_logits = logits.masked_fill(~batch_masks, -1e9)
                dist = Categorical(logits=masked_logits / max(1e-3, float(self.current_temperature)))
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, batch_returns)
                loss = policy_loss + self.value_loss_coef * value_loss - self.current_entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = new_log_probs - batch_old_log_probs
                    approx_kl = float((((torch.exp(log_ratio) - 1.0) - log_ratio).mean()).item())

                total_loss += float(loss.item())
                total_policy += float(policy_loss.item())
                total_value += float(value_loss.item())
                total_entropy += float(entropy.item())
                update_count += 1

                if self.target_kl > 0.0 and approx_kl > self.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        self.rollout_buffer.clear()
        denom = float(max(1, update_count))
        return {
            "avg_loss": total_loss / denom,
            "policy_loss": total_policy / denom,
            "value_loss": total_value / denom,
            "entropy": total_entropy / denom,
            "sample_count": dataset_size,
            "approx_kl": approx_kl,
        }

    def _update_training_schedules(self, progress: float):
        self.current_entropy_coef = max(
            self.entropy_coef_end,
            self.entropy_coef_start - progress * (self.entropy_coef_start - self.entropy_coef_end),
        )
        self.current_temperature = max(
            self.temperature_end,
            self.temperature_start - progress * (self.temperature_start - self.temperature_end),
        )

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
        self._log("[GCN-PPO-CLASSIC] training start")

        main_iterations = max(1, int(self.train_iterations))
        total_planned_iterations = max(1, main_iterations + max(0, int(self.extra_train_iterations)))
        success_rate = 0.0
        rollout_success_rate = 0.0
        replay_success_rate = 0.0

        for it in range(main_iterations):
            step_limit = self._episode_step_limit(it)
            stats = self._collect_rollout_steps(self.steps_per_epoch, step_limit, "train_main", it + 1)
            update_metrics = self._update_from_rollout_buffer()
            total_steps += stats["steps_collected"]
            total_loss += update_metrics["avg_loss"]
            total_policy_loss += update_metrics["policy_loss"]
            total_value_loss += update_metrics["value_loss"]
            total_entropy += update_metrics["entropy"]
            total_updates += 1
            train_goal_rollouts += stats["goal_rollouts"]
            progress = float(it + 1) / float(total_planned_iterations)
            self._update_training_schedules(progress)
            best_show = self.best_train_makespan if self.best_train_makespan < 2 ** 31 - 1 else -1
            avg_reward = (
                sum(stats["episode_rewards"]) / float(len(stats["episode_rewards"]))
                if stats["episode_rewards"]
                else 0.0
            )
            self._log(
                "[GCN-PPO-CLASSIC] iter "
                + str(it + 1)
                + "/"
                + str(main_iterations)
                + " phase=main"
                + " steps="
                + str(stats["steps_collected"])
                + " episodes="
                + str(stats["episode_count"])
                + " samples="
                + str(update_metrics["sample_count"])
                + " avg_reward="
                + format(avg_reward, ".4f")
                + " avg_loss="
                + format(update_metrics["avg_loss"], ".4f")
                + " policy_loss="
                + format(update_metrics["policy_loss"], ".4f")
                + " value_loss="
                + format(update_metrics["value_loss"], ".4f")
                + " entropy="
                + format(update_metrics["entropy"], ".4f")
                + " approx_kl="
                + format(update_metrics["approx_kl"], ".6f")
                + " entropy_coef="
                + format(self.current_entropy_coef, ".4f")
                + " temperature="
                + format(self.current_temperature, ".4f")
                + " best_makespan="
                + str(best_show)
                + " goal_rollouts="
                + str(train_goal_rollouts)
                + " stop_summary="
                + format_reason_counts(stats["stop_counts"])
                + " failure_counts="
                + format_reason_counts(self.train_failure_counts)
            )

        success_rate, rollout_success_rate, replay_success_rate = self._estimate_policy_success_rate(self.goal_eval_rollouts)
        extra_used = 0
        while success_rate < self.goal_min_success_rate and extra_used < self.extra_train_iterations:
            iter_index = extra_used + 1
            stats = self._collect_rollout_steps(self.steps_per_epoch, self.max_steps_per_episode, "train_extra", iter_index)
            update_metrics = self._update_from_rollout_buffer()
            total_steps += stats["steps_collected"]
            total_loss += update_metrics["avg_loss"]
            total_policy_loss += update_metrics["policy_loss"]
            total_value_loss += update_metrics["value_loss"]
            total_entropy += update_metrics["entropy"]
            total_updates += 1
            train_goal_rollouts += stats["goal_rollouts"]
            extra_used += 1
            progress = float(main_iterations + extra_used) / float(total_planned_iterations)
            self._update_training_schedules(progress)
            success_rate, rollout_success_rate, replay_success_rate = self._estimate_policy_success_rate(self.goal_eval_rollouts)
            best_show = self.best_train_makespan if self.best_train_makespan < 2 ** 31 - 1 else -1
            avg_reward = (
                sum(stats["episode_rewards"]) / float(len(stats["episode_rewards"]))
                if stats["episode_rewards"]
                else 0.0
            )
            self._log(
                "[GCN-PPO-CLASSIC] iter "
                + str(iter_index)
                + "/"
                + str(self.extra_train_iterations)
                + " phase=extra"
                + " steps="
                + str(stats["steps_collected"])
                + " episodes="
                + str(stats["episode_count"])
                + " samples="
                + str(update_metrics["sample_count"])
                + " avg_reward="
                + format(avg_reward, ".4f")
                + " avg_loss="
                + format(update_metrics["avg_loss"], ".4f")
                + " policy_loss="
                + format(update_metrics["policy_loss"], ".4f")
                + " value_loss="
                + format(update_metrics["value_loss"], ".4f")
                + " entropy="
                + format(update_metrics["entropy"], ".4f")
                + " approx_kl="
                + format(update_metrics["approx_kl"], ".6f")
                + " entropy_coef="
                + format(self.current_entropy_coef, ".4f")
                + " temperature="
                + format(self.current_temperature, ".4f")
                + " best_makespan="
                + str(best_show)
                + " goal_rollouts="
                + str(train_goal_rollouts)
                + " stop_summary="
                + format_reason_counts(stats["stop_counts"])
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
        self.extra_info["stepsPerEpoch"] = self.steps_per_epoch
        self.extra_info["miniBatchSize"] = self.minibatch_size
        self.extra_info["targetKL"] = self.target_kl
        self.extra_info["entropyCoefStart"] = self.entropy_coef_start
        self.extra_info["entropyCoefEnd"] = self.entropy_coef_end
        self.extra_info["temperatureStart"] = self.temperature_start
        self.extra_info["temperatureEnd"] = self.temperature_end
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
        self.extra_info["ppoVariant"] = "classic"
        self.is_trained = True
        self._log("[GCN-PPO-CLASSIC] training done train_steps=" + str(total_steps))


class PetriNetGCNPPOClassicHQ(PetriNetGCNPPOClassic):
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
        steps_per_epoch: int = 1024,
        minibatch_size: int = 128,
        target_kl: float = 0.05,
        entropy_coef_start: float = 0.03,
        entropy_coef_end: float = 0.01,
        temperature_start: float = 1.3,
        temperature_end: float = 1.0,
        **kwargs,
    ):
        if not hasattr(petri_net, "max_residence_time"):
            raise ValueError("petri_net 缺少 max_residence_time，无法使用 PPO Classic HQ 版本。")
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
            entropy_coef=entropy_coef_start,
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
            steps_per_epoch=steps_per_epoch,
            minibatch_size=minibatch_size,
            target_kl=target_kl,
            entropy_coef_start=entropy_coef_start,
            entropy_coef_end=entropy_coef_end,
            temperature_start=temperature_start,
            temperature_end=temperature_end,
            **kwargs,
        )
