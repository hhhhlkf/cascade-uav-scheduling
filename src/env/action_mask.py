from __future__ import annotations

from typing import Iterable, List

import numpy as np

from src.env.network_simulator import MeshNetworkSimulator
from src.env.uav_simulator import UAVSimulator
from src.utils.types import Task

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None


def compute_action_mask(
    ready_tasks: List[Task],
    uavs: Iterable[UAVSimulator],
    network: MeshNetworkSimulator,
    max_ready_tasks: int,
) -> np.ndarray:
    uav_list = list(uavs)
    mask = np.zeros((max_ready_tasks, len(uav_list)), dtype=np.float32)
    for task_idx, task in enumerate(ready_tasks[:max_ready_tasks]):
        for uav_idx, uav in enumerate(uav_list):
            if not uav.can_accept(task):
                continue
            if task.resource_requirement.bandwidth_mbps > 0.0 and not network.is_connected_to_command(uav.uav_id):
                continue
            mask[task_idx, uav_idx] = 1.0
    return mask


def decode_assignment_matrix(action: np.ndarray, mask: np.ndarray) -> List[tuple[int, int]]:
    scores = np.asarray(action, dtype=np.float32)
    if scores.shape != mask.shape:
        padded = np.zeros_like(mask, dtype=np.float32)
        rows = min(scores.shape[0], mask.shape[0])
        cols = min(scores.shape[1], mask.shape[1])
        padded[:rows, :cols] = scores[:rows, :cols]
        scores = padded
    scores = scores * mask
    if linear_sum_assignment is not None and np.any(scores > 0.0):
        cost = -scores.copy()
        cost[(mask <= 0.0) | (scores <= 0.0)] = 1e6
        rows, cols = linear_sum_assignment(cost)
        return [
            (int(row), int(col))
            for row, col in zip(rows, cols)
            if mask[row, col] > 0.0 and scores[row, col] > 0.0 and cost[row, col] < 1e6
        ]
    candidates = [
        (float(scores[task_idx, uav_idx]), task_idx, uav_idx)
        for task_idx in range(scores.shape[0])
        for uav_idx in range(scores.shape[1])
        if mask[task_idx, uav_idx] > 0.0 and scores[task_idx, uav_idx] > 0.0
    ]
    candidates.sort(reverse=True)
    assignments: List[tuple[int, int]] = []
    used_tasks: set[int] = set()
    used_uavs: set[int] = set()
    for _, task_idx, uav_idx in candidates:
        if task_idx in used_tasks or uav_idx in used_uavs:
            continue
        assignments.append((task_idx, uav_idx))
        used_tasks.add(task_idx)
        used_uavs.add(uav_idx)
    return assignments
