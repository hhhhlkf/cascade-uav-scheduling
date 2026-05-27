from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.utils.seed import seed_everything
from src.utils.types import (
    ModalityType,
    Position,
    Region,
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
        "resources": ResourceVec(0.65, 0.45, 16.0, 256.0, 100.0, 0.95),
        "sensors": [ModalityType.RGB, ModalityType.THERMAL],
    },
    UAVType.MULTISPECTRAL: {
        "resources": ResourceVec(0.70, 0.55, 16.0, 384.0, 120.0, 0.95),
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
        "resources": ResourceVec(0.55, 0.25, 8.0, 192.0, 100.0, 0.92),
        "sensors": [ModalityType.RGB, ModalityType.THERMAL, ModalityType.MIXED],
    },
}


DISTRIBUTIONS: Dict[str, Dict[str, object]] = {
    "DS1": {
        "disaster_area_km2": [10.0, 64.0],
        "flood_ratio": [0.2, 0.6],
        "building_density": ["low", "medium", "high"],
        "civilian_density": [1.0, 35.0],
        "comm_failure_rate": [0.0, 0.45],
        "breach_risk_level": [0, 3],
        "terrain_roughness": ["flat", "hilly"],
        "num_regions": [2, 4],
        "num_uavs_total": [6, 12],
        "uav_orin_8gb_ratio": [0.2, 0.4],
    },
    "DS2": {
        "disaster_area_km2": [40.0, 100.0],
        "flood_ratio": [0.45, 0.8],
        "building_density": ["medium", "high"],
        "civilian_density": [15.0, 50.0],
        "comm_failure_rate": [0.3, 0.75],
        "breach_risk_level": [1, 3],
        "terrain_roughness": ["hilly", "complex"],
        "num_regions": [3, 5],
        "num_uavs_total": [9, 15],
        "uav_orin_8gb_ratio": [0.25, 0.5],
    },
    "DS3": {
        "disaster_area_km2": [64.0, 100.0],
        "flood_ratio": [0.6, 0.8],
        "building_density": ["high"],
        "civilian_density": [25.0, 50.0],
        "comm_failure_rate": [0.5, 0.9],
        "breach_risk_level": [3, 3],
        "terrain_roughness": ["complex"],
        "num_regions": [3, 5],
        "num_uavs_total": [12, 15],
        "uav_orin_8gb_ratio": [0.3, 0.5],
    },
}


TASK_CATALOG: Dict[str, Dict[str, object]] = {
    "A1": {"modality": ModalityType.RGB, "resource": ResourceVec(0.10, 0.00, 0.5, 2.0, 64.0, 0.03), "duration": 60.0, "deadline": 1800.0, "priority": 5, "required": [UAVType.VISION, UAVType.MULTISPECTRAL, UAVType.RESCUE]},
    "A2": {"modality": ModalityType.THERMAL, "resource": ResourceVec(0.10, 0.00, 0.2, 1.0, 24.0, 0.03), "duration": 50.0, "deadline": 1800.0, "priority": 6, "required": [UAVType.VISION, UAVType.RESCUE]},
    "A3": {"modality": ModalityType.RGB, "resource": ResourceVec(0.15, 0.00, 1.0, 4.0, 96.0, 0.04), "duration": 70.0, "deadline": 1800.0, "priority": 6, "required": [UAVType.VISION, UAVType.MULTISPECTRAL, UAVType.RESCUE]},
    "A4": {"modality": ModalityType.MULTISPECTRAL, "resource": ResourceVec(0.15, 0.00, 1.5, 6.0, 120.0, 0.05), "duration": 90.0, "deadline": 2100.0, "priority": 6, "required": [UAVType.MULTISPECTRAL]},
    "A5": {"modality": ModalityType.HYPERSPECTRAL, "resource": ResourceVec(0.20, 0.00, 2.0, 8.0, 180.0, 0.06), "duration": 140.0, "deadline": 2400.0, "priority": 7, "required": [UAVType.HYPERSPECTRAL]},
    "P1": {"modality": ModalityType.RGB, "resource": ResourceVec(0.25, 0.15, 1.5, 6.0, 10.0, 0.04), "duration": 80.0, "deadline": 1500.0, "priority": 6, "required": [UAVType.VISION, UAVType.MULTISPECTRAL, UAVType.RESCUE]},
    "P2": {"modality": ModalityType.MULTISPECTRAL, "resource": ResourceVec(0.35, 0.25, 3.0, 8.0, 12.0, 0.06), "duration": 120.0, "deadline": 1800.0, "priority": 6, "required": [UAVType.MULTISPECTRAL, UAVType.HYPERSPECTRAL]},
    "P3": {"modality": ModalityType.RGB, "resource": ResourceVec(0.30, 0.20, 2.0, 8.0, 15.0, 0.05), "duration": 110.0, "deadline": 1800.0, "priority": 6, "required": [UAVType.VISION, UAVType.MULTISPECTRAL, UAVType.RESCUE]},
    "P4": {"modality": ModalityType.THERMAL, "resource": ResourceVec(0.20, 0.10, 1.0, 4.0, 8.0, 0.04), "duration": 60.0, "deadline": 900.0, "priority": 8, "required": [UAVType.VISION, UAVType.RESCUE]},
    "I1": {"modality": ModalityType.RGB, "resource": ResourceVec(0.30, 0.60, 2.0, 10.0, 10.0, 0.08), "duration": 120.0, "deadline": 300.0, "priority": 8, "required": [UAVType.VISION, UAVType.RESCUE]},
    "I2": {"modality": ModalityType.THERMAL, "resource": ResourceVec(0.20, 0.30, 1.0, 4.0, 5.0, 0.06), "duration": 70.0, "deadline": 120.0, "priority": 10, "required": [UAVType.VISION, UAVType.RESCUE]},
    "I3": {"modality": ModalityType.MULTISPECTRAL, "resource": ResourceVec(0.40, 0.50, 3.0, 12.0, 20.0, 0.08), "duration": 150.0, "deadline": 600.0, "priority": 7, "required": [UAVType.MULTISPECTRAL, UAVType.HYPERSPECTRAL]},
    "I4": {"modality": ModalityType.RGB, "resource": ResourceVec(0.35, 0.45, 3.0, 16.0, 20.0, 0.08), "duration": 160.0, "deadline": 900.0, "priority": 7, "required": [UAVType.VISION, UAVType.MULTISPECTRAL, UAVType.RESCUE]},
    "I5": {"modality": ModalityType.HYPERSPECTRAL, "resource": ResourceVec(0.45, 0.55, 4.0, 20.0, 25.0, 0.10), "duration": 200.0, "deadline": 900.0, "priority": 8, "required": [UAVType.HYPERSPECTRAL]},
    "F1": {"modality": ModalityType.MIXED, "resource": ResourceVec(0.50, 0.70, 4.0, 16.0, 15.0, 0.10), "duration": 160.0, "deadline": 300.0, "priority": 9, "required": [UAVType.RESCUE, UAVType.VISION]},
    "F2": {"modality": ModalityType.MIXED, "resource": ResourceVec(0.55, 0.75, 6.0, 20.0, 20.0, 0.12), "duration": 210.0, "deadline": 900.0, "priority": 8, "required": [UAVType.HYPERSPECTRAL, UAVType.MULTISPECTRAL]},
    "F3": {"modality": ModalityType.MIXED, "resource": ResourceVec(0.80, 0.90, 8.0, 30.0, 30.0, 0.15), "duration": 300.0, "deadline": 1200.0, "priority": 8, "required": [UAVType.HYPERSPECTRAL, UAVType.MULTISPECTRAL]},
    "F4": {"modality": ModalityType.MIXED, "resource": ResourceVec(0.55, 0.65, 5.0, 18.0, 15.0, 0.12), "duration": 220.0, "deadline": 900.0, "priority": 9, "required": [UAVType.HYPERSPECTRAL, UAVType.RESCUE]},
    "C1": {"modality": ModalityType.RGB, "resource": ResourceVec(0.05, 0.00, 0.1, 1.0, 10.0, 0.04), "duration": 180.0, "deadline": 3600.0, "priority": 10, "required": [UAVType.COMM, UAVType.RESCUE]},
    "C2": {"modality": ModalityType.RGB, "resource": ResourceVec(0.10, 0.00, 0.5, 2.0, 30.0, 0.04), "duration": 120.0, "deadline": 1800.0, "priority": 8, "required": [UAVType.COMM, UAVType.RESCUE, UAVType.VISION]},
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
        params = self._sample_scenario_params()
        side_m = (float(params["disaster_area_km2"]) * 1_000_000.0) ** 0.5
        area = (
            float(self.scenario_cfg.get("grid_size_x", side_m)),
            float(self.scenario_cfg.get("grid_size_y", side_m)),
        )
        civilians = self._generate_civilians(area, params)
        breaches = self._random_positions(int(self.scenario_cfg.get("num_breach_points", 3)), area)
        regions = self._generate_regions(area, params)
        uavs = self._generate_uavs(area, params)
        tasks = self._generate_region_task_chains(area, regions) if self._uses_distribution() else self._generate_tasks(area)
        return Scenario(
            name=str(self.scenario_cfg.get("name", "scenario")),
            area_size=area,
            civilians=civilians,
            breach_points=breaches,
            uavs=uavs,
            tasks=tasks,
            topo_config=self.config.get("network", {}) | {"comm_failure_rate": float(params.get("comm_failure_rate", 0.1))},
            regions=regions,
            metadata={
                "distribution": params.get("distribution", "fixed"),
                "experiment_goals": ["G1", "G2", "G3", "G4", "G5", "G6"],
                "scenario_params": params,
                "water_level_mean": self.scenario_cfg.get("water_level_mean", 1.5),
                "simulation_duration_s": self.scenario_cfg.get("simulation_duration_s", 7200),
                "uav_fault_probability": self.scenario_cfg.get("uav_fault_probability", 0.0),
                "num_civilians": self.scenario_cfg.get("num_civilians", len(civilians)),
            },
        )

    def _uses_distribution(self) -> bool:
        return "distribution" in self.scenario_cfg or "disaster_area_km2" in self.scenario_cfg

    def _sample_scenario_params(self) -> Dict[str, object]:
        distribution = str(self.scenario_cfg.get("distribution", "DS1")).upper()
        base = dict(DISTRIBUTIONS.get(distribution, DISTRIBUTIONS["DS1"]))
        params: Dict[str, object] = {"distribution": distribution if self._uses_distribution() else "fixed"}
        for key, default in base.items():
            params[key] = self._sample_value(self.scenario_cfg.get(key, default))
        params["comm_failure_rate"] = float(params["comm_failure_rate"])
        params["breach_risk_level"] = int(params["breach_risk_level"])
        params["num_regions"] = int(params["num_regions"])
        params["num_uavs_total"] = int(params["num_uavs_total"])
        return params

    def _sample_value(self, value):
        if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
            low, high = value
            if isinstance(low, int) and isinstance(high, int):
                return int(self.rng.integers(low, high + 1))
            return float(self.rng.uniform(float(low), float(high)))
        if isinstance(value, (list, tuple)):
            return value[int(self.rng.integers(0, len(value)))]
        return value

    def _generate_civilians(self, area: Tuple[float, float], params: Dict[str, object]) -> List[Position]:
        default_count = int(float(params.get("civilian_density", 3.0)) * (area[0] * area[1] / 1_000_000.0))
        count = int(self.scenario_cfg.get("num_civilians", max(default_count, 1)))
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

    def _generate_uavs(self, area: Tuple[float, float], params: Dict[str, object] | None = None) -> List[UAV]:
        distribution = self.scenario_cfg.get("uav_type_distribution")
        if distribution is None and params and self._uses_distribution():
            distribution = self._derive_uav_distribution(int(params["num_uavs_total"]), float(params["uav_orin_8gb_ratio"]))
        distribution = distribution or {"UAV-V": 3, "UAV-R": 2}
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

    def _derive_uav_distribution(self, total: int, orin_8gb_ratio: float) -> Dict[str, int]:
        comm_rescue = max(2, int(round(total * orin_8gb_ratio)))
        remaining = max(total - comm_rescue, 3)
        distribution = {
            "UAV-V": max(1, int(round(remaining * 0.45))),
            "UAV-M": max(1, int(round(remaining * 0.30))),
            "UAV-H": max(1, remaining - int(round(remaining * 0.45)) - int(round(remaining * 0.30))),
            "UAV-C": max(1, comm_rescue // 2),
            "UAV-R": max(1, comm_rescue - max(1, comm_rescue // 2)),
        }
        diff = total - sum(distribution.values())
        distribution["UAV-V"] = max(1, distribution["UAV-V"] + diff)
        return distribution

    def _generate_regions(self, area: Tuple[float, float], params: Dict[str, object]) -> List[Region]:
        if not self._uses_distribution():
            return [
                Region(
                    region_id="R0",
                    center=Position(area[0] * 0.5, area[1] * 0.5, 0.0),
                    area_km2=(area[0] * area[1]) / 1_000_000.0,
                    flood_ratio=float(params.get("flood_ratio", 0.4)),
                    building_density=str(params.get("building_density", "medium")),
                    civilian_density=float(params.get("civilian_density", 5.0)),
                    comm_failure_rate=float(params.get("comm_failure_rate", 0.1)),
                    breach_risk_level=int(params.get("breach_risk_level", 1)),
                    terrain_roughness=str(params.get("terrain_roughness", "flat")),
                )
            ]
        count = int(params["num_regions"])
        regions: List[Region] = []
        total_area = float(params["disaster_area_km2"])
        for idx in range(count):
            jitter = 0.85 + 0.30 * float(self.rng.random())
            regions.append(
                Region(
                    region_id=f"R{idx}",
                    center=self._random_position(area),
                    area_km2=max(total_area / count * jitter, 0.5),
                    flood_ratio=float(np.clip(float(params["flood_ratio"]) + self.rng.normal(0.0, 0.08), 0.0, 1.0)),
                    building_density=str(params["building_density"]),
                    civilian_density=max(float(params["civilian_density"]) + float(self.rng.normal(0.0, 4.0)), 0.0),
                    comm_failure_rate=float(np.clip(float(params["comm_failure_rate"]) + self.rng.normal(0.0, 0.05), 0.0, 0.95)),
                    breach_risk_level=int(np.clip(int(params["breach_risk_level"]) + int(self.rng.integers(-1, 2)), 0, 3)),
                    terrain_roughness=str(params["terrain_roughness"]),
                )
            )
        return regions

    def _generate_region_task_chains(self, area: Tuple[float, float], regions: List[Region]) -> List[Task]:
        tasks: List[Task] = []
        for region in regions:
            tasks.extend(self._build_region_chain(region, area))
        return tasks

    def _build_region_chain(self, region: Region, area: Tuple[float, float]) -> List[Task]:
        triggered = self._triggered_task_types(region)
        chain: List[Task] = []
        tasks_by_id: Dict[str, Task] = {}
        for task_type in triggered:
            task = self._build_catalog_task(region, task_type, area, len(chain))
            chain.append(task)
            tasks_by_id[task.task_id] = task
        dependencies = {
            "P1": ["A1"],
            "P2": [task for task in ("A4", "A5") if task in triggered],
            "P3": ["A3"] if "A3" in triggered else ["A1"],
            "P4": ["A2"],
            "I1": ["P1"],
            "I2": ["P4"] if "P4" in triggered else ["A2"],
            "I3": ["P2"],
            "I4": ["P1"],
            "I5": ["P2"],
            "F1": ["I1", "I2"],
            "F2": [task for task in ("I3", "I4") if task in triggered],
            "F3": [task for task in ("I1", "I2", "I3", "I4", "I5") if task in triggered],
            "F4": [task for task in ("I5", "F2") if task in triggered],
            "C2": [task for task in ("F1", "F2", "F3", "F4", "I1", "I2", "I3", "I4", "I5") if task in triggered],
        }
        for child_type, parent_types in dependencies.items():
            if child_type not in triggered:
                continue
            child_id = self._task_id(region.region_id, child_type)
            for parent_type in parent_types:
                parent_id = self._task_id(region.region_id, parent_type)
                if parent_id in tasks_by_id:
                    add_dependency(tasks_by_id, parent_id, child_id)
        return chain

    def _triggered_task_types(self, region: Region) -> List[str]:
        acquisition = ["A1"]
        if region.civilian_density >= 5.0:
            acquisition.append("A2")
        if region.building_density in {"medium", "high"}:
            acquisition.append("A3")
        if region.flood_ratio >= 0.3:
            acquisition.append("A4")
        if region.breach_risk_level >= 2 or region.flood_ratio >= 0.6:
            acquisition.append("A5")
        preprocessing = []
        if "A1" in acquisition:
            preprocessing.append("P1")
        if "A3" in acquisition:
            preprocessing.append("P3")
        if "A4" in acquisition or "A5" in acquisition:
            preprocessing.append("P2")
        if "A2" in acquisition:
            preprocessing.append("P4")
        inference = ["I1"]
        if "A2" in acquisition:
            inference.append("I2")
        if "A4" in acquisition:
            inference.append("I3")
        if region.flood_ratio >= 0.4:
            inference.append("I4")
        if "A5" in acquisition:
            inference.append("I5")
        fusion = []
        if {"I1", "I2"}.issubset(inference):
            fusion.append("F1")
        if {"I3", "I4"}.issubset(inference):
            fusion.append("F2")
        if len(inference) >= 4:
            fusion.append("F3")
        if region.breach_risk_level >= 2:
            fusion.append("F4")
        communication = ["C2"]
        if region.comm_failure_rate >= 0.3:
            communication.insert(0, "C1")
        optional = acquisition + preprocessing + inference + fusion
        if len(optional) + len(communication) <= 15:
            return optional + communication
        return optional[: 15 - len(communication)] + communication

    def _build_catalog_task(self, region: Region, task_type: str, area: Tuple[float, float], local_idx: int) -> Task:
        spec = TASK_CATALOG[task_type]
        urgency = 1.0 + region.civilian_density / 100.0 + region.breach_risk_level * 0.08
        flood_factor = 1.0 + region.flood_ratio * 0.35
        duration = float(spec["duration"]) * flood_factor * float(self.rng.uniform(0.85, 1.15))
        deadline = max(float(spec["deadline"]) / urgency, 90.0)
        priority = min(10, int(spec["priority"]) + (1 if region.breach_risk_level >= 2 and task_type.startswith(("I", "F")) else 0))
        target = Position(
            float(np.clip(self.rng.normal(region.center.x, area[0] * 0.06), 0.0, area[0])),
            float(np.clip(self.rng.normal(region.center.y, area[1] * 0.06), 0.0, area[1])),
            0.0,
        )
        return Task(
            task_id=self._task_id(region.region_id, task_type),
            task_type=task_type,
            modality=spec["modality"],
            resource_requirement=clone_resource(spec["resource"]),
            priority=priority,
            deadline_s=deadline,
            duration_s=duration,
            arrival_time_s=0.0,
            target_position=target,
            data_size_mb=float(self.rng.uniform(8.0, 500.0 if task_type in {"A5", "F3"} else 120.0)),
            region_id=region.region_id,
            required_uav_types=list(spec["required"]),
        )

    def _task_id(self, region_id: str, task_type: str) -> str:
        return f"{region_id}-{task_type}"

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
