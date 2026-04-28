from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as f
from torch import nn

from .features import PetriRepresentationInput, PetriStateFeatureEncoder
from .graph import PetriNetGraph


def _as_graph(pre, post, **graph_kwargs) -> PetriNetGraph:
    if isinstance(pre, PetriNetGraph):
        return pre
    return PetriNetGraph.from_components(pre=pre, post=post, **graph_kwargs)


def _expand_transition_features(seed: torch.Tensor, batch_size: int) -> torch.Tensor:
    return seed.unsqueeze(0).expand(batch_size, -1, -1)


@dataclass
class PetriRepresentationOutput:
    place_embeddings: torch.Tensor
    transition_embeddings: torch.Tensor
    graph_embedding: torch.Tensor
    transition_logits: Optional[torch.Tensor]


class MultiScaleFusionLayer(nn.Module):
    def __init__(self, place_dim: int, transition_dim: int, dropout: float = 0.0):
        super().__init__()
        self.p2t_pre = nn.Linear(place_dim, transition_dim, bias=False)
        self.p2t_post = nn.Linear(place_dim, transition_dim, bias=False)
        self.t2p_pre = nn.Linear(transition_dim, place_dim, bias=False)
        self.t2p_post = nn.Linear(transition_dim, place_dim, bias=False)
        self.transition_self = nn.Linear(transition_dim, transition_dim)
        self.place_self = nn.Linear(place_dim, place_dim)
        self.transition_context = nn.Linear(place_dim + transition_dim, transition_dim, bias=False)
        self.place_context = nn.Linear(place_dim + transition_dim, place_dim, bias=False)
        self.transition_norm = nn.LayerNorm(transition_dim)
        self.place_norm = nn.LayerNorm(place_dim)
        self.transition_ff = nn.Sequential(
            nn.Linear(transition_dim, transition_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(transition_dim * 2, transition_dim),
        )
        self.place_ff = nn.Sequential(
            nn.Linear(place_dim, place_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(place_dim * 2, place_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        place_hidden: torch.Tensor,
        transition_hidden: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
        pre_t: torch.Tensor,
        post_t: torch.Tensor,
        transition_pre_degree: torch.Tensor,
        transition_post_degree: torch.Tensor,
        place_pre_degree: torch.Tensor,
        place_post_degree: torch.Tensor,
    ):
        context = torch.cat([place_hidden.mean(dim=1), transition_hidden.mean(dim=1)], dim=-1)

        t_msg_pre = torch.einsum("tp,bph->bth", pre_t, self.p2t_pre(place_hidden)) / transition_pre_degree
        t_msg_post = torch.einsum("tp,bph->bth", post_t, self.p2t_post(place_hidden)) / transition_post_degree
        transition_update = self.transition_self(transition_hidden) + t_msg_pre + t_msg_post
        transition_update = transition_update + self.transition_context(context).unsqueeze(1)
        transition_hidden = self.transition_norm(transition_hidden + self.dropout(self.transition_ff(f.gelu(transition_update))))

        p_msg_pre = torch.einsum("pt,bth->bph", pre, self.t2p_pre(transition_hidden)) / place_pre_degree
        p_msg_post = torch.einsum("pt,bth->bph", post, self.t2p_post(transition_hidden)) / place_post_degree
        place_update = self.place_self(place_hidden) + p_msg_pre + p_msg_post
        place_update = place_update + self.place_context(context).unsqueeze(1)
        place_hidden = self.place_norm(place_hidden + self.dropout(self.place_ff(f.gelu(place_update))))
        return place_hidden, transition_hidden


class MultiScalePetriRepresentation(nn.Module):
    def __init__(
        self,
        pre: Sequence[Sequence[int]],
        post: Sequence[Sequence[int]],
        place_input_dim: int,
        transition_input_dim: int,
        place_hidden_dim: int = 128,
        transition_hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0,
        with_policy_head: bool = True,
    ):
        super().__init__()
        graph = _as_graph(pre, post)
        self.register_buffer("pre", graph.pre)
        self.register_buffer("post", graph.post)
        self.register_buffer("pre_t", graph.pre_t)
        self.register_buffer("post_t", graph.post_t)
        self.register_buffer("transition_pre_degree", self.pre_t.sum(dim=-1, keepdim=True).clamp_min(1.0).unsqueeze(0))
        self.register_buffer("transition_post_degree", self.post_t.sum(dim=-1, keepdim=True).clamp_min(1.0).unsqueeze(0))
        self.register_buffer("place_pre_degree", self.pre.sum(dim=-1, keepdim=True).clamp_min(1.0).unsqueeze(0))
        self.register_buffer("place_post_degree", self.post.sum(dim=-1, keepdim=True).clamp_min(1.0).unsqueeze(0))
        self.transition_count = graph.transition_count
        self.transition_input_dim = transition_input_dim
        self.place_input = nn.Linear(place_input_dim, place_hidden_dim)
        self.transition_input = nn.Linear(transition_input_dim, transition_hidden_dim)
        self.layers = nn.ModuleList(
            [MultiScaleFusionLayer(place_hidden_dim, transition_hidden_dim, dropout=dropout) for _ in range(max(1, num_layers))]
        )
        readout_dim = max(place_hidden_dim, transition_hidden_dim)
        self.graph_readout = nn.Sequential(
            nn.Linear(place_hidden_dim + transition_hidden_dim, readout_dim),
            nn.GELU(),
        )
        self.with_policy_head = with_policy_head
        if with_policy_head:
            self.transition_head = nn.Sequential(
                nn.Linear(transition_hidden_dim + readout_dim, readout_dim),
                nn.GELU(),
                nn.Linear(readout_dim, 1),
            )
        else:
            self.transition_head = None

    def forward(
        self,
        place_features: torch.Tensor,
        transition_features: Optional[torch.Tensor] = None,
    ) -> PetriRepresentationOutput:
        single = place_features.dim() == 2
        if single:
            place_features = place_features.unsqueeze(0)
        if transition_features is None:
            transition_features = torch.zeros(
                place_features.shape[0],
                self.transition_count,
                self.transition_input_dim,
                dtype=place_features.dtype,
                device=place_features.device,
            )
        elif transition_features.dim() == 2:
            transition_features = transition_features.unsqueeze(0)

        place_hidden = f.gelu(self.place_input(place_features))
        transition_hidden = f.gelu(self.transition_input(transition_features))

        for layer in self.layers:
            place_hidden, transition_hidden = layer(
                place_hidden=place_hidden,
                transition_hidden=transition_hidden,
                pre=self.pre,
                post=self.post,
                pre_t=self.pre_t,
                post_t=self.post_t,
                transition_pre_degree=self.transition_pre_degree,
                transition_post_degree=self.transition_post_degree,
                place_pre_degree=self.place_pre_degree,
                place_post_degree=self.place_post_degree,
            )

        graph_embedding = self.graph_readout(torch.cat([place_hidden.mean(dim=1), transition_hidden.mean(dim=1)], dim=-1))
        transition_logits = None
        if self.transition_head is not None:
            graph_context = graph_embedding.unsqueeze(1).expand(-1, transition_hidden.shape[1], -1)
            transition_logits = self.transition_head(torch.cat([transition_hidden, graph_context], dim=-1)).squeeze(-1)

        if single:
            return PetriRepresentationOutput(
                place_embeddings=place_hidden.squeeze(0),
                transition_embeddings=transition_hidden.squeeze(0),
                graph_embedding=graph_embedding.squeeze(0),
                transition_logits=None if transition_logits is None else transition_logits.squeeze(0),
            )
        return PetriRepresentationOutput(
            place_embeddings=place_hidden,
            transition_embeddings=transition_hidden,
            graph_embedding=graph_embedding,
            transition_logits=transition_logits,
        )


class PetriNetGCNEnhanced(nn.Module):
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
        graph = _as_graph(
            pre,
            post,
            end=end,
            min_delay_p=min_delay_p,
            min_delay_t=min_delay_t,
            capacity=capacity,
            max_residence_time=max_residence_time,
            place_from_places=place_from_places,
        )
        feature_encoder = PetriStateFeatureEncoder(graph)
        self.place_input_dim = feature_encoder.place_feature_dim
        self.transition_input_dim = feature_encoder.transition_feature_dim
        self._include_capacity_feature = "capacity" in feature_encoder.place_feature_names
        self.backbone = MultiScalePetriRepresentation(
            pre=graph,
            post=None,
            place_input_dim=self.place_input_dim,
            transition_input_dim=self.transition_input_dim,
            place_hidden_dim=lambda_p,
            transition_hidden_dim=lambda_t,
            num_layers=max(1, int(num_layers)),
            with_policy_head=True,
        )
        self.register_buffer("transition_seed", self._build_transition_seed(graph))

    def forward(self, x_p) -> torch.Tensor:
        place_features, transition_features = self._split_inputs(x_p)
        place_features = self._adapt_place_features(place_features)
        transition_features = self._adapt_transition_features(transition_features)
        if transition_features is None:
            if place_features.dim() == 2:
                transition_features = self.transition_seed
            else:
                transition_features = _expand_transition_features(self.transition_seed, place_features.shape[0])
        return self.backbone(place_features, transition_features).transition_logits

    def _split_inputs(self, x_p):
        if isinstance(x_p, PetriRepresentationInput):
            return x_p.place_features, x_p.transition_features
        if isinstance(x_p, dict) and "place_features" in x_p and "transition_features" in x_p:
            return x_p["place_features"], x_p["transition_features"]
        return x_p, None

    def _adapt_place_features(self, place_features: torch.Tensor) -> torch.Tensor:
        if place_features.shape[-1] == self.place_input_dim:
            return place_features
        if place_features.shape[-1] != 4:
            raise ValueError("unsupported place feature dim: " + str(place_features.shape[-1]))
        token = torch.clamp(place_features[..., 0], min=0.0)
        goal = torch.clamp(place_features[..., 1], min=0.0)
        timer = torch.clamp(place_features[..., 2], min=0.0)
        min_delay = torch.clamp(place_features[..., 3], min=0.0)
        zeros = torch.zeros_like(token)
        has_goal_constraint = (torch.abs(goal - token) > 1e-6).to(dtype=place_features.dtype)
        rows = [
            torch.log1p(token),
            has_goal_constraint,
            torch.log1p(timer),
            torch.log1p(min_delay),
            zeros,
            zeros,
            zeros,
        ]
        if self._include_capacity_feature:
            rows.append(zeros)
        return torch.stack(rows, dim=-1)

    def _adapt_transition_features(self, transition_features):
        if transition_features is None:
            return None
        if transition_features.shape[-1] == self.transition_input_dim:
            return transition_features
        if transition_features.shape[-1] == 7:
            zeros = torch.zeros_like(transition_features[..., 0])
            return torch.stack(
                [
                    transition_features[..., 0],
                    transition_features[..., 1],
                    transition_features[..., 2],
                    transition_features[..., 3],
                    transition_features[..., 4],
                    transition_features[..., 5],
                    transition_features[..., 6],
                    zeros,
                    zeros,
                    zeros,
                    zeros,
                    zeros,
                ],
                dim=-1,
            )
        if transition_features.shape[-1] != 4:
            raise ValueError("unsupported transition feature dim: " + str(transition_features.shape[-1]))
        zeros = torch.zeros_like(transition_features[..., 0])
        return torch.stack(
            [
                zeros,
                zeros,
                zeros,
                transition_features[..., 0],
                transition_features[..., 1],
                transition_features[..., 2],
                transition_features[..., 3],
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
            ],
            dim=-1,
        )

    @staticmethod
    def _build_transition_seed(graph: PetriNetGraph) -> torch.Tensor:
        degree = graph.transition_degree_features()
        return torch.stack(
            [
                torch.zeros_like(degree[:, 2]),
                torch.zeros_like(degree[:, 2]),
                torch.log1p(graph.min_delay_t),
                torch.log1p(degree[:, 0]),
                torch.log1p(degree[:, 1]),
                torch.log1p(degree[:, 2]),
                torch.log1p(degree[:, 3]),
                torch.zeros_like(degree[:, 2]),
                torch.zeros_like(degree[:, 2]),
                torch.zeros_like(degree[:, 2]),
                torch.zeros_like(degree[:, 2]),
                torch.zeros_like(degree[:, 2]),
            ],
            dim=-1,
        )


class PetriNetGCN(PetriNetGCNEnhanced):
    pass
