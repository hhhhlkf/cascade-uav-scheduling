from __future__ import annotations

import numpy as np

from src.algorithms.base_scheduler import BaseScheduler


class RoundRobinScheduler(BaseScheduler):
    """Lower-bound scheduler that cycles through UAVs in fixed order."""

    def __init__(self, max_ready_tasks: int, num_uavs: int):
        super().__init__(max_ready_tasks, num_uavs)
        self.cursor = 0

    def decide(self, action_mask: np.ndarray) -> np.ndarray:
        action = self._empty_action(action_mask)
        if self.num_uavs <= 0:
            return action
        used_uavs: set[int] = set()
        for task_idx in range(action_mask.shape[0]):
            for offset in range(self.num_uavs):
                uav_idx = (self.cursor + offset) % self.num_uavs
                if uav_idx in used_uavs or action_mask[task_idx, uav_idx] <= 0.0:
                    continue
                action[task_idx, uav_idx] = 1.0
                used_uavs.add(uav_idx)
                self.cursor = (uav_idx + 1) % self.num_uavs
                break
        return action

    def reset(self) -> None:
        super().reset()
        self.cursor = 0

