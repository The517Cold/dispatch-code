import random
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class Experience:
    state: object
    action: int
    reward: float
    next_state: object
    done: bool


class PrioritizedReplay:
    def __init__(self, capacity: int, alpha: float):
        self.capacity = capacity
        self.alpha = alpha
        self.data: List[Experience] = []
        self.priorities: List[float] = []
        self.pos = 0

    def __len__(self):
        return len(self.data)

    def add(self, exp: Experience, priority: float):
        p = max(priority, 1e-6)
        if len(self.data) < self.capacity:
            self.data.append(exp)
            self.priorities.append(p)
        else:
            self.data[self.pos] = exp
            self.priorities[self.pos] = p
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float):
        scaled = [p ** self.alpha for p in self.priorities]
        total = sum(scaled)
        if total <= 0:
            probs = [1.0 / len(scaled)] * len(scaled)
        else:
            probs = [v / total for v in scaled]
        indices = random.choices(range(len(self.data)), probs, k=batch_size)
        samples = [self.data[i] for i in indices]
        n = len(self.data)
        weights = [((n * probs[i]) ** (-beta)) for i in indices]
        max_w = max(weights) if weights else 1.0
        weights = [w / max_w for w in weights]
        return indices, samples, torch.tensor(weights, dtype=torch.float32)

    def update(self, indices: List[int], priorities: List[float]):
        for i, p in zip(indices, priorities):
            self.priorities[i] = max(float(p), 1e-6)
