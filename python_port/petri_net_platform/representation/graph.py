from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch


def _as_float_tensor(values: Optional[Sequence[Any]], size: int, default: float, device: Optional[torch.device]) -> torch.Tensor:
    if values is None:
        values = [default] * size
    if len(values) != size:
        values = list(values[:size]) + [default] * max(0, size - len(values))
    return torch.tensor(values, dtype=torch.float32, device=device)


def _resolve_device(device: Optional[torch.device]) -> Optional[torch.device]:
    if device is None:
        return None
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return resolved


def _freeze_nested_ints(groups: Optional[Sequence[Sequence[int]]], size: int) -> Tuple[Tuple[int, ...], ...]:
    if groups is None:
        groups = [[] for _ in range(size)]
    frozen = []
    for index in range(size):
        if index < len(groups):
            frozen.append(tuple(int(v) for v in groups[index]))
        else:
            frozen.append(tuple())
    return tuple(frozen)


@dataclass
class PetriNetGraph:
    pre: torch.Tensor
    post: torch.Tensor
    pre_t: torch.Tensor
    post_t: torch.Tensor
    incidence: torch.Tensor
    min_delay_p: torch.Tensor
    min_delay_t: torch.Tensor
    end: torch.Tensor
    capacity: torch.Tensor
    max_residence_time: torch.Tensor
    place_from_places: Tuple[Tuple[int, ...], ...]
    p_map: Dict[str, int]
    t_map: Dict[str, int]
    p_map_v: Dict[int, str]
    t_map_v: Dict[int, str]

    @classmethod
    def from_components(
        cls,
        pre: Sequence[Sequence[int]],
        post: Sequence[Sequence[int]],
        min_delay_p: Optional[Sequence[int]] = None,
        min_delay_t: Optional[Sequence[int]] = None,
        end: Optional[Sequence[int]] = None,
        capacity: Optional[Sequence[int]] = None,
        max_residence_time: Optional[Sequence[int]] = None,
        place_from_places: Optional[Sequence[Sequence[int]]] = None,
        p_map: Optional[Dict[str, int]] = None,
        t_map: Optional[Dict[str, int]] = None,
        p_map_v: Optional[Dict[int, str]] = None,
        t_map_v: Optional[Dict[int, str]] = None,
        device: Optional[torch.device] = None,
    ) -> "PetriNetGraph":
        actual_device = _resolve_device(device)
        pre_tensor = torch.tensor(pre, dtype=torch.float32, device=actual_device)
        post_tensor = torch.tensor(post, dtype=torch.float32, device=actual_device)
        incidence = post_tensor - pre_tensor
        place_count = int(pre_tensor.shape[0])
        transition_count = int(pre_tensor.shape[1]) if pre_tensor.ndim == 2 else 0
        return cls(
            pre=pre_tensor,
            post=post_tensor,
            pre_t=pre_tensor.transpose(0, 1),
            post_t=post_tensor.transpose(0, 1),
            incidence=incidence,
            min_delay_p=_as_float_tensor(min_delay_p, place_count, 0.0, actual_device),
            min_delay_t=_as_float_tensor(min_delay_t, transition_count, 0.0, actual_device),
            end=_as_float_tensor(end, place_count, -1.0, actual_device),
            capacity=_as_float_tensor(capacity, place_count, float(2 ** 31 - 1), actual_device),
            max_residence_time=_as_float_tensor(max_residence_time, place_count, float(2 ** 31 - 1), actual_device),
            place_from_places=_freeze_nested_ints(place_from_places, place_count),
            p_map=dict(p_map or {}),
            t_map=dict(t_map or {}),
            p_map_v=dict(p_map_v or {}),
            t_map_v=dict(t_map_v or {}),
        )

    @classmethod
    def from_context(cls, context: Dict[str, Any], device: Optional[torch.device] = None) -> "PetriNetGraph":
        matrix_translator = context.get("matrix_translator")
        return cls.from_components(
            pre=context["pre"],
            post=context["post"],
            min_delay_p=context.get("min_delay_p"),
            min_delay_t=context.get("min_delay_t"),
            end=context.get("end"),
            capacity=context.get("capacity"),
            max_residence_time=context.get("max_residence_time"),
            place_from_places=context.get("place_from_places"),
            p_map=getattr(matrix_translator, "p_map", None),
            t_map=getattr(matrix_translator, "t_map", None),
            p_map_v=getattr(matrix_translator, "p_map_v", None),
            t_map_v=getattr(matrix_translator, "t_map_v", None),
            device=device,
        )

    @property
    def device(self) -> torch.device:
        return self.pre.device

    @property
    def place_count(self) -> int:
        return int(self.pre.shape[0])

    @property
    def transition_count(self) -> int:
        return int(self.pre.shape[1]) if self.pre.ndim == 2 else 0

    def to(self, device: torch.device) -> "PetriNetGraph":
        return PetriNetGraph(
            pre=self.pre.to(device),
            post=self.post.to(device),
            pre_t=self.pre_t.to(device),
            post_t=self.post_t.to(device),
            incidence=self.incidence.to(device),
            min_delay_p=self.min_delay_p.to(device),
            min_delay_t=self.min_delay_t.to(device),
            end=self.end.to(device),
            capacity=self.capacity.to(device),
            max_residence_time=self.max_residence_time.to(device),
            place_from_places=self.place_from_places,
            p_map=dict(self.p_map),
            t_map=dict(self.t_map),
            p_map_v=dict(self.p_map_v),
            t_map_v=dict(self.t_map_v),
        )

    def place_degree_features(self) -> torch.Tensor:
        incoming = (self.post > 0).sum(dim=1, dtype=torch.float32)  # 有多少个变迁可以 向该库所 产生托肯（通过post边）
        outgoing = (self.pre > 0).sum(dim=1, dtype=torch.float32)  # 有多少个变迁可以 从该库所 产生托肯（通过pre边）
        return torch.stack([incoming, outgoing], dim=-1)

    def transition_degree_features(self) -> torch.Tensor:
        input_places = (self.pre > 0).sum(dim=0, dtype=torch.float32)  # 有多少个库所 可以 向该变迁 产生托肯（通过pre边）
        output_places = (self.post > 0).sum(dim=0, dtype=torch.float32)  # 有多少个库所 可以 从该变迁 产生托肯（通过post边）
        input_weight = self.pre.sum(dim=0)  # 该变迁的输入权重
        output_weight = self.post.sum(dim=0)  # 该变迁的输出权重
        return torch.stack([input_places, output_places, input_weight, output_weight], dim=-1)


def build_petri_graph(context: Dict[str, Any], device: Optional[torch.device] = None) -> PetriNetGraph:
    return PetriNetGraph.from_context(context, device=device)
