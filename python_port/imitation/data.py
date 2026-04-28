"""Common data structures used by behavior cloning."""

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from python_port.petri_net_platform.representation.features import PetriRepresentationInput


@dataclass
class BCSample:
    # `expert_action` always stores the global transition id in the Petri net.
    state_features: torch.Tensor
    action_mask: torch.Tensor
    expert_action: int
    meta: Dict[str, object]

    def as_dict(self):
        return {
            "state_features": self.state_features,
            "action_mask": self.action_mask,
            "expert_action": int(self.expert_action),
            "meta": self.meta,
        }


class BCDataset(Dataset):
    def __init__(self, samples: List[BCSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx].as_dict()


def bc_collate_fn(batch):
    # Keep masks and labels aligned so the trainer can apply masked cross-entropy.
    normalized = [item.as_dict() if isinstance(item, BCSample) else item for item in batch]
    first_state = normalized[0]["state_features"]
    if isinstance(first_state, PetriRepresentationInput):
        states = PetriRepresentationInput(
            place_features=torch.stack([item["state_features"].place_features for item in normalized], dim=0),
            transition_features=torch.stack([item["state_features"].transition_features for item in normalized], dim=0),
        )
    else:
        states = torch.stack([item["state_features"] for item in normalized], dim=0)
    masks = torch.stack([item["action_mask"] for item in normalized], dim=0)
    actions = torch.tensor([item["expert_action"] for item in normalized], dtype=torch.int64)
    metas = [item["meta"] for item in normalized]
    return {
        "state_features": states,
        "action_mask": masks,
        "expert_action": actions,
        "meta": metas,
    }


def save_samples(path: str, samples: List[BCSample], meta: Dict[str, object]):
    # Persist generated expert samples so finetune/validation can be reproduced.
    payload = {
        "samples": [_serialize_sample(s) for s in samples],
        "meta": meta,
    }
    torch.save(payload, path)


def load_samples(path: str):
    payload = torch.load(path, map_location="cpu")
    raw_samples = payload.get("samples", [])
    out = []
    for item in raw_samples:
        # Normalise everything back to CPU tensors because loaders may run on CPU only.
        out.append(
            BCSample(
                state_features=_deserialize_state_features(item["state_features"]),
                action_mask=item["action_mask"].bool().cpu(),
                expert_action=int(item["expert_action"]),
                meta=item.get("meta", {}),
            )
        )
    return out, payload.get("meta", {})


def _serialize_sample(sample: BCSample):
    data = sample.as_dict()
    data["state_features"] = _serialize_state_features(data["state_features"])
    return data


def _serialize_state_features(state_features):
    if isinstance(state_features, PetriRepresentationInput):
        return {
            "place_features": state_features.place_features.cpu(),
            "transition_features": state_features.transition_features.cpu(),
        }
    return state_features.cpu()


def _deserialize_state_features(state_features):
    if isinstance(state_features, dict) and "place_features" in state_features and "transition_features" in state_features:
        return PetriRepresentationInput(
            place_features=state_features["place_features"].cpu(),
            transition_features=state_features["transition_features"].cpu(),
        )
    return state_features.cpu()
