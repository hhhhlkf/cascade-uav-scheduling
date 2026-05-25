from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.utils.seed import seed_everything
from src.utils.types import (
    ModalityType,
    Position,
    ResourceVec,
    Scenario,
    Task,
    UAV,
    UAVType,
    add_dependency,
    clone_resource,
)


UAV_PROFILES: Dict[UAVType, Dict] = {
    UAVType.VISION: {
        "resources": ResourceVec(0.55, 0.20, 8.0, 128.0, 80.0, 0.95),
        "sensors": [ModalityType.RGB, ModalityType.THERMAL],
    },
    UAVType.MULTISPECTRAL: {
        "resources": ResourceVec(0.60, 0.35, 12.0, 256.0, 100.0, 0.95),
        "sensors": [ModalityType.RGB, ModalityType.MULTISPECTRAL],
    },
    UAVType.HYPERSPECTRAL: {
        "resources": ResourceVec(0.70, 0.50, 16.0, 512.0, 120.0, 0.90),
        "sensors": [ModalityType.HYPERSPECTRAL, ModalityType.MULTISPECTRAL],
    },
    UAVType.COMM: {
        "resources": ResourceVec(0.45, 0.10, 8.0, 128.0, 200.0, 0.98),
        "sensors": [ModalityType.RGB],
    },
    UAVType.RESCUE: {
        "resources": ResourceVec(0.65, 0.30, 12.0, 256.0, 100.0, 0.92),
        "sensors": [ModalityType.RGB, ModalityType.THERMAL, ModalityType.MIXED],
    },
}


TASK_TEMPLATES: Tuple[Dict, ...] = (
    {
        "prefix": "A",
        "modality": ModalityType.RGB,
        "resource": ResourceVec(0.10, 0.00, 0.5, 2.0, 2.0, 0.03),
        "duration": (25, 60),
        "deadline": (600, 1800),
        "priority": (3, 6),
        "required": [UAVType.VISION, UAVType.MULTISPECTRAL, UAVType.COMM, UAVType.RESCUE],
    },
    {
        "prefix": "I",
        "modality": ModalityType.THERMAL,
        "resource": ResourceVec(0.25, 0.20, 1.0, 4.0, 8.0, 0.06),
        "duration": (40, 120),
        "deadline": (180, 900),
        "priority": (7, 10),
        "required": [UAVType.VISION, UAVType.RESCUE],
    },
    {
        "prefix": "P",
        "modality": ModalityType.MULTISPECTRAL,
        "resource": ResourceVec(0.35, 0.25, 2.0, 8.0, 12.0, 0.08),
        "duration": (80, 180),
        "deadline": (900, 2400),
        "priority": (4, 8),
        "required": [UAVType.MULTISPECTRAL, UAVType.HYPERSPECTRAL],
    },
    {
        "prefix": "F",
        "modality": ModalityType.MIXED,
        "resource": ResourceVec(0.45, 0.35, 3.0, 12.0, 20.0, 0.10),
        "duration": (90, 240),
        "deadline": (300, 1200),
        "priority": (8, 10),
        "required": [UAVType.RESCUE],
    },
    {
        "prefix": "C",
        "modality": ModalityType.RGB,
        "resource": ResourceVec(0.20, 0.00, 0.5, 2.0, 60.0, 0.05),
        "duration": (120, 300),
        "deadline": (1200, 3600),
        "priority": (5, 9),
        "required": [UAVType.COMM],
    },
)


class ScenarioGenerator:
    """Builds deterministic disaster scenarios from a YAML-style config dict."""

    def __init__(self, config: Dict):
        self.config = config
        self.scenario_cfg = config.get("scenario", config)
        seed = config.get("env", {}).get("seed", self.scenario_cfg.get("seed", 0))
        self.rng = seed_everything(seed)

    def generate(self) -> Scenario:
        area = (
            float(self.scenario_cfg.get("grid_size_x", 5000)),
            float(self.scenario_cfg.get("grid_size_y", 5000)),
        )
        civilians = self._generate_civilians(area)
        breaches = self._random_positions(int(self.scenario_cfg.get("num_breach_points", 3)), area)
        uavs = self._generate_uavs(area)
        tasks = self._generate_tasks(area)
        return Scenario(
            name=str(self.scenario_cfg.get("name", "scenario")),
            area_size=area,
            civilians=civilians,
            breach_points=breaches,
            uavs=uavs,
            tasks=tasks,
            topo_config=self.config.get("network", {}) | {"comm_failure_rate": self.scenario_cfg.get("comm_failure_rate", 0.1)},
            metadata={
                "water_level_mean": self.scenario_cfg.get("water_level_mean", 1.5),
                "simulation_duration_s": self.scenario_cfg.get("simulation_duration_s", 7200),
                "uav_fault_probability": self.scenario_cfg.get("uav_fault_probability", 0.0),
                "num_civilians": self.scenario_cfg.get("num_civilians", len(civilians)),
            },
        )

    def _generate_civilians(self, area: Tuple[float, float]) -> List[Position]:
        count = int(self.scenario_cfg.get("num_civilians", 80))
        hotspots = int(self.scenario_cfg.get("num_civilian_hotspots", 3))
        hotspot_centers = self._random_positions(hotspots, area)
        positions: List[Position] = []
        for idx in range(count):
            if hotspot_centers and idx >= count // 2:
                center = hotspot_centers[idx % len(hotspot_centers)]
                x = np.clip(self.rng.normal(center.x, area[0] * 0.04), 0.0, area[0])
                y = np.clip(self.rng.normal(center.y, area[1] * 0.04), 0.0, area[1])
                positions.append(Position(float(x), float(y), 0.0))
            else:
                positions.append(self._random_position(area))
        return positions

    def _generate_uavs(self, area: Tuple[float, float]) -> List[UAV]:
        distribution = self.scenario_cfg.get("uav_type_distribution", {"UAV-V": 3, "UAV-R": 2})
        uavs: List[UAV] = []
        launch = Position(area[0] * 0.05, area[1] * 0.05, 120.0)
        for type_name, count in distribution.items():
            uav_type = UAVType(type_name)
            profile = UAV_PROFILES[uav_type]
            for idx in range(int(count)):
                total = clone_resource(profile["resources"])
                offset = Position(launch.x + len(uavs) * 8.0, launch.y + idx * 8.0, launch.z)
                uavs.append(
                    UAV(
                        uav_id=f"{uav_type.value}-{idx + 1}",
                        uav_type=uav_type,
                        position=offset,
                        battery_level=float(total.energy),
                        resources_total=total,
                        resources_available=clone_resource(total),
                        sensors=list(profile["sensors"]),
                    )
                )
        return uavs

    def _generate_tasks(self, area: Tuple[float, float]) -> List[Task]:
        task_count = int(self.scenario_cfg.get("task_count", 30))
        dag_depth = max(1, int(self.scenario_cfg.get("dag_depth_max", 5)))
        tasks: List[Task] = []
        for idx in range(task_count):
            template = TASK_TEMPLATES[idx % len(TASK_TEMPLATES)]
            duration = self.rng.uniform(*template["duration"])
            deadline = self.rng.uniform(*template["deadline"])
            priority = int(self.rng.integers(template["priority"][0], template["priority"][1] + 1))
            task = Task(
                task_id=f"T{idx:03d}",
                task_type=f"{template['prefix']}{idx % 5 + 1}",
                modality=template["modality"],
                resource_requirement=clone_resource(template["resource"]),
                priority=priority,
                deadline_s=float(deadline),
                duration_s=float(duration),
                arrival_time_s=0.0,
                target_position=self._random_position(area),
                data_size_mb=float(self.rng.uniform(5.0, 250.0)),
                required_uav_types=list(template["required"]),
            )
            tasks.append(task)

        tasks_by_id = {task.task_id: task for task in tasks}
        layer_size = max(1, int(np.ceil(task_count / dag_depth)))
        for child_idx in range(layer_size, task_count):
            child = tasks[child_idx]
            parent_pool = tasks[max(0, child_idx - layer_size * 2) : child_idx]
            parent_count = int(self.rng.integers(0, min(3, len(parent_pool)) + 1))
            for parent in self.rng.choice(parent_pool, size=parent_count, replace=False) if parent_count else []:
                add_dependency(tasks_by_id, parent.task_id, child.task_id)
        return tasks

    def _random_positions(self, count: int, area: Tuple[float, float]) -> List[Position]:
        return [self._random_position(area) for _ in range(count)]

    def _random_position(self, area: Tuple[float, float]) -> Position:
        return Position(float(self.rng.uniform(0, area[0])), float(self.rng.uniform(0, area[1])), 0.0)
