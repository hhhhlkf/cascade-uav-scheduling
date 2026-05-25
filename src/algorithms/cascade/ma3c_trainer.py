from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.algorithms.base_scheduler import BaseScheduler
from src.algorithms.cascade.hungarian_match import masked_assignment


@dataclass
class MA3CConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    n_steps: int = 128


class CASCADEMA3CScheduler(BaseScheduler):
    """Thin scheduler shell for the future mA3C+MHSA+GNN policy."""

    def __init__(self, max_ready_tasks: int, num_uavs: int, config: MA3CConfig | None = None):
        super().__init__(max_ready_tasks, num_uavs)
        self.config = config or MA3CConfig()

    def decide(self, action_mask: np.ndarray) -> np.ndarray:
        action = self._empty_action(action_mask)
        if self.last_obs is None:
            return action
        # Placeholder policy: use priority/deadline logits while network training is wired in.
        task_features = self.last_obs["task_features"]
        task_logits = task_features[:, 6] + (1.0 - task_features[:, 7])
        scores = action_mask * task_logits[:, None]
        for task_idx, uav_idx in masked_assignment(scores, action_mask):
            action[task_idx, uav_idx] = scores[task_idx, uav_idx]
        return action

    def learn(self, batch: Dict) -> Dict[str, float]:
        return {"loss_actor": 0.0, "loss_critic": 0.0, "entropy": 0.0}

