from __future__ import annotations

import numpy as np

from src.algorithms.base_scheduler import BaseScheduler


class GreedyScheduler(BaseScheduler):
    """Priority/deadline greedy baseline over the ready-task observation."""

    def decide(self, action_mask: np.ndarray) -> np.ndarray:
        action = self._empty_action(action_mask)
        if self.last_obs is None or action_mask.size == 0:
            return action
        task_features = self.last_obs["task_features"]
        uav_features = self.last_obs["uav_features"]
        task_scores = task_features[:, 6] * 2.0 + (1.0 - task_features[:, 7])
        uav_scores = uav_features[:, 2:8].mean(axis=1) + uav_features[:, 8] * 0.25
        used_uavs: set[int] = set()
        for task_idx in np.argsort(-task_scores):
            valid_uavs = [idx for idx in np.where(action_mask[task_idx] > 0.0)[0] if idx not in used_uavs]
            if not valid_uavs:
                continue
            best_uav = max(valid_uavs, key=lambda idx: float(uav_scores[idx]))
            action[task_idx, best_uav] = float(task_scores[task_idx] + uav_scores[best_uav])
            used_uavs.add(best_uav)
        return action

