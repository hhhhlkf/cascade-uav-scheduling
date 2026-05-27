from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None


def masked_assignment(scores: np.ndarray, action_mask: np.ndarray) -> List[Tuple[int, int]]:
    """Discretize task-UAV scores into one-to-one assignments under an action mask."""
    scores = np.asarray(scores, dtype=np.float32)
    mask = np.asarray(action_mask, dtype=np.float32)
    if scores.shape != mask.shape:
        raise ValueError(f"scores shape {scores.shape} does not match mask shape {mask.shape}")
    valid = (mask > 0.0) & (scores > 0.0)
    if not valid.any():
        return []
    if linear_sum_assignment is None:
        return _greedy_assignment(scores, mask)
    cost = -scores.copy()
    cost[~valid] = 1e6
    rows, cols = linear_sum_assignment(cost)
    return [(int(row), int(col)) for row, col in zip(rows, cols) if valid[row, col] and cost[row, col] < 1e6]


def _greedy_assignment(scores: np.ndarray, mask: np.ndarray) -> List[Tuple[int, int]]:
    candidates = [
        (float(scores[task_idx, uav_idx]), task_idx, uav_idx)
        for task_idx in range(scores.shape[0])
        for uav_idx in range(scores.shape[1])
        if mask[task_idx, uav_idx] > 0.0 and scores[task_idx, uav_idx] > 0.0
    ]
    candidates.sort(reverse=True)
    assignments: List[Tuple[int, int]] = []
    used_tasks: set[int] = set()
    used_uavs: set[int] = set()
    for _, task_idx, uav_idx in candidates:
        if task_idx in used_tasks or uav_idx in used_uavs:
            continue
        assignments.append((task_idx, uav_idx))
        used_tasks.add(task_idx)
        used_uavs.add(uav_idx)
    return assignments
