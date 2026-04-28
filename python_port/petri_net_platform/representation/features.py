from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from .graph import PetriNetGraph


INF_SENTINEL = float(2 ** 31 - 1)


def _signed_log1p(value: float) -> float:
    if value == 0:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    # log1p(x) = log(1 + x)
    # 当x = 0时，log1p(x) = 0
    # 当x = -1时，log1p(x) = -inf
    return sign * torch.log1p(torch.tensor(abs(value), dtype=torch.float32)).item()


def _safe_log1p(value: float) -> float:
    # 对负数进行对数变换,避免数值爆炸
    value = max(0.0, value)
    return torch.log1p(torch.tensor(value, dtype=torch.float32)).item()


def _as_device(device: Optional[torch.device], fallback: torch.device) -> torch.device:
    return device if device is not None else fallback


@dataclass
class PetriRepresentationInput:
    # 封装GCN的输入数据,包含库所和变迁两部分特征
    place_features: torch.Tensor
    transition_features: torch.Tensor

    def to(self, device: torch.device) -> "PetriRepresentationInput":
        return PetriRepresentationInput(
            place_features=self.place_features.to(device),
            transition_features=self.transition_features.to(device),
        )

    def cpu(self) -> "PetriRepresentationInput":
        return self.to(torch.device("cpu"))


class PetriStateFeatureEncoder:
    def __init__(self, graph: PetriNetGraph, device: Optional[torch.device] = None):
        # 设置图和设备
        self.graph = graph.to(_as_device(device, graph.device))
        self.device = self.graph.device
        # 预计算度特征(避免重复计算)
        self._place_degree = self.graph.place_degree_features().to(self.device)
        self._transition_degree = self.graph.transition_degree_features().to(self.device)
        # 检查是否有容量约束
        self._include_capacity_feature = self._has_finite_constraint(self.graph.capacity)
        # 死锁控制器相关
        self._controller_petri_net = None
        self._deadlock_controller = None
        self._controller_representation_enabled = True
        # 构建特征名称和维度
        self.place_feature_names = self._build_place_feature_names()
        self.transition_feature_names = self._build_transition_feature_names()
        self.place_feature_dim = len(self.place_feature_names)
        self.transition_feature_dim = len(self.transition_feature_names)

    def bind_deadlock_controller(self, petri_net_or_getter, deadlock_controller, enabled: bool = True):
        self._controller_petri_net = petri_net_or_getter
        self._deadlock_controller = deadlock_controller
        self._controller_representation_enabled = bool(enabled)

    def set_controller_representation_enabled(self, enabled: bool):
        self._controller_representation_enabled = bool(enabled)

    def encode(self, marking) -> PetriRepresentationInput:
        return PetriRepresentationInput(
            place_features=self.encode_places(marking),
            transition_features=self.encode_transitions(marking),
        )

    def encode_batch(self, markings: Iterable[object]) -> PetriRepresentationInput:
        place_features = []
        transition_features = []
        for marking in markings:
            encoded = self.encode(marking)
            place_features.append(encoded.place_features)
            transition_features.append(encoded.transition_features)
        return PetriRepresentationInput(
            place_features=torch.stack(place_features, dim=0),
            transition_features=torch.stack(transition_features, dim=0),
        )

    def encode_places(self, marking) -> torch.Tensor:
        p_info = list(marking.get_p_info())
        rows = []
        for index, token in enumerate(p_info):
            oldest_residence = self._get_oldest_token_residence_time(marking, index)
            max_residence = self.graph.max_residence_time[index].item()
            max_residence = 0.0 if max_residence >= INF_SENTINEL else max_residence
            capacity = self.graph.capacity[index].item()
            capacity = 0.0 if capacity >= INF_SENTINEL else capacity
            degree_in = self._place_degree[index, 0].item()
            degree_out = self._place_degree[index, 1].item()
            has_goal_constraint = 1.0 if self.graph.end[index].item() >= 0 else 0.0
            row = [
                _safe_log1p(float(token)),
                has_goal_constraint,
                _safe_log1p(oldest_residence),  # 驻留时间最长的token的驻留时间
                _safe_log1p(self.graph.min_delay_p[index].item()),
                _safe_log1p(max_residence),
                _safe_log1p(degree_in),
                _safe_log1p(degree_out),
            ]
            if self._include_capacity_feature:
                row.append(_safe_log1p(capacity))
            rows.append(row)
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def encode_transitions(self, marking) -> torch.Tensor:
        enabled = self._get_enabled_transitions(marking)
        current_delay = self._get_current_transition_delay(marking)
        controller_features = self._get_controller_transition_features(marking)
        rows = []
        for index in range(self.graph.transition_count):
            degree = self._transition_degree[index]
            total_pre = degree[2].item()
            total_post = degree[3].item()
            rows.append(
                [
                    enabled[index],
                    _safe_log1p(current_delay[index]),
                    _safe_log1p(self.graph.min_delay_t[index].item()),
                    _safe_log1p(degree[0].item()),
                    _safe_log1p(degree[1].item()),
                    _safe_log1p(total_pre),
                    _safe_log1p(total_post),
                    controller_features["controller_allowed"][index],
                    controller_features["hard_blocked"][index],
                    controller_features["soft_risk"][index],
                    controller_features["safe_ratio"],
                    controller_features["fbm_candidate"],
                ]
            )
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def _build_place_feature_names(self):
        names = [
            "token",
            "has_goal_constraint",
            "oldest_residence",
            "min_delay_p",
            "max_residence",
            "degree_in",
            "degree_out",
        ]
        if self._include_capacity_feature:
            names.append("capacity")
        return tuple(names)

    @staticmethod
    def _build_transition_feature_names():
        return (
            "enabled",
            "current_delay",
            "min_delay_t",
            "input_places",
            "output_places",
            "total_pre",
            "total_post",
            "controller_allowed",
            "hard_blocked",
            "soft_risk",
            "safe_ratio",
            "fbm_candidate",
        )

    @staticmethod
    def _has_finite_constraint(values: torch.Tensor) -> bool:
        return bool(torch.any(values < INF_SENTINEL).item())

    def _get_oldest_token_residence_time(self, marking, place: int) -> float:
        # Use the oldest token's residence time when the marking tracks per-token/per-place residence.
        # 支持不同版本的marking对象
        # 这里直接取索引0的token,认为它是最老的
        if hasattr(marking, "get_residence_time"):
            try:
                value = float(marking.get_residence_time(place, 0))
                return max(0.0, value)
            except BaseException:
                pass
        if hasattr(marking, "residence_time_info"):
            try:
                residence = marking.residence_time_info[place]
                if residence:
                    return max(0.0, float(residence[0]))
            except BaseException:
                pass
        if hasattr(marking, "t_info"):
            try:
                tokens = marking.t_info[place]
                if tokens and hasattr(tokens[0], "residence_time"):
                    return max(0.0, float(tokens[0].residence_time))
            except BaseException:
                pass
        return 0.0

    def _get_enabled_transitions(self, marking):
        # 方法1：直接从marking中获取是否启用
        if hasattr(marking, "is_enable") and marking.is_enable:
            return [1.0 if bool(value) else 0.0 for value in marking.is_enable]
        # 方法2：根据token数量判断是否启用,没有检查时间限制
        tokens = torch.tensor(list(marking.get_p_info()), dtype=torch.float32, device=self.device)
        enough_tokens = tokens.unsqueeze(1) >= self.graph.pre
        return enough_tokens.all(dim=0).to(dtype=torch.float32).tolist()

    def _get_current_transition_delay(self, marking):
        if hasattr(marking, "curr_delay_t") and marking.curr_delay_t:
            return [float(value) for value in marking.curr_delay_t]
        return [0.0] * self.graph.transition_count

    def _get_controller_transition_features(self, marking):
        transition_count = self.graph.transition_count
        zero_actions = [0.0] * transition_count
        default = {
            "controller_allowed": zero_actions.copy(),
            "hard_blocked": zero_actions.copy(),
            "soft_risk": zero_actions.copy(),
            "safe_ratio": 0.0,
            "fbm_candidate": 0.0,
        }
        if not self._controller_representation_enabled:
            return default
        if self._controller_petri_net is None or self._deadlock_controller is None:
            return default
        petri_net = self._controller_petri_net() if callable(self._controller_petri_net) else self._controller_petri_net
        if petri_net is None:
            return default
        # 调用死锁控制器分析状态
        try:
            analysis = self._deadlock_controller.analyze_state(petri_net, marking)
        except BaseException:
            return default
        # 提取控制器判断结果
        controller_allowed = zero_actions.copy()
        hard_blocked = zero_actions.copy()
        soft_risk = zero_actions.copy()
        for action in analysis.controller_actions:
            if 0 <= action < transition_count:
                controller_allowed[action] = 1.0
        for action in analysis.hard_blocked_actions:
            if 0 <= action < transition_count:
                hard_blocked[action] = 1.0
        for action in analysis.soft_risk_actions:
            if 0 <= action < transition_count:
                soft_risk[action] = 1.0
        enabled_count = max(1, analysis.enabled_count())
        safe_ratio = float(analysis.safe_count()) / float(enabled_count)
        return {
            "controller_allowed": controller_allowed,
            "hard_blocked": hard_blocked,
            "soft_risk": soft_risk,
            "safe_ratio": safe_ratio,
            "fbm_candidate": 1.0 if analysis.fbm_candidate else 0.0,
        }
        # - controller_allowed ：控制器认为安全的动作(one-hot向量)
        # - hard_blocked ：被硬过滤规则阻塞的动作(one-hot向量)
        # - soft_risk ：被lookahead判定为高风险的动作(one-hot向量)
        # - safe_ratio ：安全动作占使能动作的比例(标量)
        # - fbm_candidate ：当前状态是否为FBM边界态(标量)

# 底下这两个是上一版本的代码
class PetriStateEncoder:
    def __init__(self, end, min_delay_p, device: torch.device):
        self.end = end
        self.min_delay_p = min_delay_p
        self.device = device

    def encode(self, marking) -> torch.Tensor:
        p_info = list(marking.get_p_info())
        rows = []
        for index, token in enumerate(p_info):
            goal = self.end[index] if self.end[index] != -1 else token
            oldest_residence = self._get_oldest_token_residence_time(marking, index)
            rows.append([float(token), float(goal), float(oldest_residence), float(self.min_delay_p[index])])
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def encode_batch(self, markings: Iterable[object]) -> torch.Tensor:
        return torch.stack([self.encode(marking) for marking in markings], dim=0)

    def _get_oldest_token_residence_time(self, marking, place: int) -> float:
        if hasattr(marking, "get_residence_time"):
            try:
                value = float(marking.get_residence_time(place, 0))
                return max(0.0, value)
            except BaseException:
                pass
        if hasattr(marking, "residence_time_info"):
            try:
                residence = marking.residence_time_info[place]
                if residence:
                    return max(0.0, float(residence[0]))
            except BaseException:
                pass
        if hasattr(marking, "t_info"):
            try:
                tokens = marking.t_info[place]
                if tokens and hasattr(tokens[0], "residence_time"):
                    return max(0.0, float(tokens[0].residence_time))
            except BaseException:
                pass
        return 0.0


class PetriStateEncoderEnhanced(PetriStateEncoder):
    def __init__(
        self,
        end,
        min_delay_p,
        device: torch.device,
        pre=None,
        post=None,
        min_delay_t=None,
        capacity=None,
        max_residence_time=None,
        place_from_places=None,
        graph: Optional[PetriNetGraph] = None,
    ):
        super().__init__(end, min_delay_p, device)
        self.feature_encoder = None
        if graph is None and pre is not None and post is not None:
            graph = PetriNetGraph.from_components(
                pre=pre,
                post=post,
                min_delay_p=min_delay_p,
                min_delay_t=min_delay_t,
                end=end,
                capacity=capacity,
                max_residence_time=max_residence_time,
                place_from_places=place_from_places,
                device=device,
            )
        if graph is not None:
            self.feature_encoder = PetriStateFeatureEncoder(graph, device=device)

    def encode(self, marking):
        if self.feature_encoder is not None:
            return self.feature_encoder.encode(marking)
        return super().encode(marking)

    def encode_batch(self, markings: Iterable[object]):
        if self.feature_encoder is not None:
            return self.feature_encoder.encode_batch(markings)
        return super().encode_batch(markings)

    def bind_deadlock_controller(self, petri_net, deadlock_controller, enabled: bool = True):
        if self.feature_encoder is not None:
            self.feature_encoder.bind_deadlock_controller(petri_net, deadlock_controller, enabled=enabled)

    def set_controller_representation_enabled(self, enabled: bool):
        if self.feature_encoder is not None:
            self.feature_encoder.set_controller_representation_enabled(enabled)
