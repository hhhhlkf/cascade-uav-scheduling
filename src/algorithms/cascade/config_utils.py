from __future__ import annotations


def max_ready_tasks_from_config(config: dict, default: int = 16) -> int:
    return int(config.get("env", {}).get("max_ready_tasks", default))


def max_uavs_from_config(config: dict, default: int = 15) -> int:
    scenario = config.get("scenario", {})
    total = scenario.get("num_uavs_total")
    if isinstance(total, list):
        return int(max(total))
    if isinstance(total, int):
        return int(total)
    distribution = scenario.get("uav_type_distribution", {})
    if distribution:
        return int(sum(distribution.values()))
    return int(default)
