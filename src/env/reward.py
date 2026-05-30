from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from src.env.network_simulator import MeshNetworkSimulator
from src.env.uav_simulator import UAVSimulator
from src.utils.types import Task


DEFAULT_REWARD_WEIGHTS = {
    "lambda_load": 0.15,
    "lambda_overload": 0.25,
    "lambda_latency": 0.10,
    "lambda_deadline": 0.30,
    "lambda_priority": 0.20,
    "completed_bonus": 1.0,
    "timeout_penalty": 2.0,
    "terminal_completion_bonus": 5.0,
    "terminal_tdsr_bonus": 2.0,
    "terminal_timeout_penalty": 5.0,
}


def compute_reward(
    uavs: Iterable[UAVSimulator],
    network: MeshNetworkSimulator,
    ready_tasks: List[Task],
    invalid_action_count: int,
    now_s: float,
    weights: Dict[str, float] | None = None,
) -> tuple[float, Dict[str, float]]:
    weights = DEFAULT_REWARD_WEIGHTS | (weights or {})
    uav_list = list(uavs)
    loads = np.asarray([uav.load_ratio() for uav in uav_list], dtype=np.float32)
    load_cost = float(np.std(loads)) if len(loads) else 0.0
    overload_cost = float(invalid_action_count)
    connected_latencies = [
        network.shortest_path_latency(uav.uav_id, network.command_vehicle_id)
        for uav in uav_list
        if network.is_connected_to_command(uav.uav_id)
    ]
    latency_cost = float(np.mean(connected_latencies) / 1000.0) if connected_latencies else 1.0
    deadline_cost = 0.0
    priority_cost = 0.0
    if ready_tasks:
        slacks = [max(task.absolute_deadline_s - now_s, 0.0) / max(task.deadline_s, 1.0) for task in ready_tasks]
        deadline_cost = float(np.mean([1.0 - min(slack, 1.0) for slack in slacks]))
        priority_cost = float(np.mean([task.priority / 10.0 for task in ready_tasks]))

    total_cost = (
        weights["lambda_load"] * load_cost
        + weights["lambda_overload"] * overload_cost
        + weights["lambda_latency"] * latency_cost
        + weights["lambda_deadline"] * deadline_cost
        + weights["lambda_priority"] * priority_cost
    )
    reward = -float(total_cost)
    return reward, {
        "cost_load": load_cost,
        "cost_overload": overload_cost,
        "cost_latency": latency_cost,
        "cost_deadline": deadline_cost,
        "cost_priority": priority_cost,
        "cost_total": float(total_cost),
    }
