from typing import List
import os
import sys

try:
    from .petri_net_gcn_dqn_enhanced import PetriNetGCNDQNEnhanced
except ImportError:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from python_port.petri_net_platform.search.petri_net_gcn_dqn_enhanced import PetriNetGCNDQNEnhanced


class PetriNetGCNDQNEnhancedHQ(PetriNetGCNDQNEnhanced):
    def __init__(
        self,
        petri_net,
        end: List[int],
        pre: List[List[int]],
        post: List[List[int]],
        min_delay_p: List[int],
        train_episodes: int = 240,
        min_steps_per_episode: int = 120,
        max_steps_per_episode: int = 700,
        rollout_count: int = 40,
        goal_eval_rollouts: int = 8,
        goal_min_success_rate: float = 0.7,
        extra_train_episodes: int = 120,
        use_reward_scaling: bool = True,
        reward_time_scale: float = 1000.0,
        use_reward_clip: bool = True,
        reward_clip_abs: float = 20.0,
        use_huber_loss: bool = True,
        huber_beta: float = 1.0,
        epsilon_init: float = 1.0,
        epsilon_min: float = 0.015,
        epsilon_decay: float = 0.995,
        verbose: bool = True,
        log_interval: int = 5,
        **kwargs,
    ):
        if not hasattr(petri_net, "max_residence_time"):
            raise ValueError("petri_net 必须提供 max_residence_time 才能启用 HQ 驻留时间约束")
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
            # lr=6e-4,
            lr=3e-4,
            replay_capacity=24000,
            replay_alpha=0.6,
            replay_beta=0.45,
            batch_size=32,
            learn_every=2,
            warmup_steps=120,
            # tau=0.015,
            tau=0.01,
            train_episodes=train_episodes,
            min_steps_per_episode=min_steps_per_episode,
            max_steps_per_episode=max_steps_per_episode,
            epsilon_init=epsilon_init,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            reward_goal_bonus=320.0,
            reward_deadlock_penalty=120.0,
            reward_progress_weight=2.2,
            reward_repeat_penalty=0.25,
            use_reward_scaling=use_reward_scaling,
            reward_time_scale=reward_time_scale,
            use_reward_clip=use_reward_clip,
            reward_clip_abs=reward_clip_abs,
            use_huber_loss=use_huber_loss,
            huber_beta=huber_beta,
            rollout_count=rollout_count,
            beam_width=48,
            beam_expand_per_node=6,
            goal_eval_rollouts=goal_eval_rollouts,
            goal_min_success_rate=goal_min_success_rate,
            extra_train_episodes=extra_train_episodes,
            verbose=verbose,
            log_interval=log_interval,
            **kwargs,
        )
