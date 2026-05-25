from __future__ import annotations

import numpy as np

from src.algorithms.base_scheduler import BaseScheduler


class MinLoadScheduler(BaseScheduler):
    """Assign each ready task to the feasible UAV with the lowest current load."""

    def decide(self, action_mask: np.ndarray) -> np.ndarray:
        action = self._empty_action(action_mask)
        if self.last_obs is None:
            return action
        uav_features = self.last_obs["uav_features"]
        resource_remaining = uav_features[:, 2:8].mean(axis=1)
        used_uavs: set[int] = set()
        for task_idx in range(action_mask.shape[0]):
            valid_uavs = [idx for idx in np.where(action_mask[task_idx] > 0.0)[0] if idx not in used_uavs]
            if not valid_uavs:
                continue
            best_uav = max(valid_uavs, key=lambda idx: float(resource_remaining[idx]))
            action[task_idx, best_uav] = 1.0 + float(resource_remaining[best_uav])
            used_uavs.add(best_uav)
        return action

