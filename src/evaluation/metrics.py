from __future__ import annotations

from typing import Dict


def summarize_episode(info: Dict, total_reward: float) -> Dict[str, float]:
    completed = float(info.get("completed_tasks", 0))
    timed_out = float(info.get("timed_out_tasks", 0))
    total_seen = max(completed + timed_out + float(info.get("ready_tasks", 0)), 1.0)
    return {
        "total_reward": float(total_reward),
        "completed_tasks": completed,
        "timed_out_tasks": timed_out,
        "completion_ratio": completed / total_seen,
        "tdsr": float(info.get("tdsr", 0.0)),
        "rpdr_proxy": float(info.get("rpdr_proxy", 0.0)),
        "sim_time_s": float(info.get("sim_time_s", 0.0)),
    }


def aggregate_metrics(results: list[Dict[str, float]]) -> Dict[str, float]:
    if not results:
        return {}
    keys = sorted({key for item in results for key in item})
    aggregated: Dict[str, float] = {}
    for key in keys:
        values = [item[key] for item in results if key in item]
        aggregated[f"{key}_mean"] = float(sum(values) / len(values))
        if len(values) > 1:
            mean = aggregated[f"{key}_mean"]
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            aggregated[f"{key}_std"] = float(variance**0.5)
        else:
            aggregated[f"{key}_std"] = 0.0
    return aggregated

