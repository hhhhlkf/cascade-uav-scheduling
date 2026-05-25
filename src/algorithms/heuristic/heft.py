from __future__ import annotations

import numpy as np

from src.algorithms.base_scheduler import BaseScheduler


class HEFTScheduler(BaseScheduler):
    """DAG-aware HEFT-style baseline using urgency and resource fit as a proxy for EFT."""

    def decide(self, action_mask: np.ndarray) -> np.ndarray:
        action = self._empty_action(action_mask)
        if self.last_obs is None:
            return action
        task_features = self.last_obs["task_features"]
        dag = self.last_obs["task_dag_adj"]
        uav_features = self.last_obs["uav_features"]
        downstream = dag[: action_mask.shape[0]].sum(axis=1)[: action_mask.shape[0]]
        urgency = 1.0 - task_features[:, 7]
        rank = task_features[:, 6] * 1.5 + urgency + downstream * 0.05
        used_uavs: set[int] = set()
        for task_idx in np.argsort(-rank):
            valid_uavs = [idx for idx in np.where(action_mask[task_idx] > 0.0)[0] if idx not in used_uavs]
            if not valid_uavs:
                continue
            requirement = task_features[task_idx, :6]
            best_uav = max(valid_uavs, key=lambda idx: self._fit_score(requirement, uav_features[idx]))
            action[task_idx, best_uav] = float(rank[task_idx] + self._fit_score(requirement, uav_features[best_uav]))
            used_uavs.add(best_uav)
        return action

    @staticmethod
    def _fit_score(requirement: np.ndarray, uav_feature: np.ndarray) -> float:
        remaining = uav_feature[2:8]
        return float(np.mean(remaining) - 0.01 * np.mean(requirement))

