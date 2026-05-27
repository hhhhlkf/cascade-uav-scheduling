from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import sqrt
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class UAVType(str, Enum):
    VISION = "UAV-V"
    MULTISPECTRAL = "UAV-M"
    HYPERSPECTRAL = "UAV-H"
    COMM = "UAV-C"
    RESCUE = "UAV-R"


class UAVStatus(str, Enum):
    IDLE = "IDLE"
    TRANSIT = "TRANSIT"
    COLLECTING = "COLLECTING"
    PROCESSING = "PROCESSING"
    RELAYING = "RELAYING"
    FAULTED = "FAULTED"


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    QUEUED = "QUEUED"
    SCHEDULED = "SCHEDULED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    PREEMPTED = "PREEMPTED"
    TIMEOUT = "TIMEOUT"


class ModalityType(str, Enum):
    RGB = "RGB"
    THERMAL = "THERMAL"
    MULTISPECTRAL = "MULTISPECTRAL"
    HYPERSPECTRAL = "HYPERSPECTRAL"
    MIXED = "MIXED"


@dataclass
class Region:
    region_id: str
    center: "Position"
    area_km2: float
    flood_ratio: float
    building_density: str
    civilian_density: float
    comm_failure_rate: float
    breach_risk_level: int
    terrain_roughness: str

    def feature_vector(self) -> np.ndarray:
        density_map = {"low": 0.0, "medium": 0.5, "high": 1.0}
        terrain_map = {"flat": 0.0, "hilly": 0.5, "complex": 1.0}
        return np.asarray(
            [
                self.area_km2 / 100.0,
                self.flood_ratio,
                density_map.get(self.building_density, 0.5),
                min(self.civilian_density / 50.0, 1.0),
                self.comm_failure_rate,
                self.breach_risk_level / 3.0,
                terrain_map.get(self.terrain_roughness, 0.5),
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class Position:
    x: float
    y: float
    z: float = 0.0

    def distance_to(self, other: "Position") -> float:
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def as_array(self) -> np.ndarray:
        return np.asarray([self.x, self.y, self.z], dtype=np.float32)


@dataclass
class ResourceVec:
    cpu: float
    gpu: float
    memory_gb: float
    storage_gb: float
    bandwidth_mbps: float
    energy: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.cpu, self.gpu, self.memory_gb, self.storage_gb, self.bandwidth_mbps, self.energy],
            dtype=np.float32,
        )

    def fits_within(self, available: "ResourceVec") -> bool:
        return bool(np.all(self.as_array() <= available.as_array() + 1e-6))

    def reserve_from(self, available: "ResourceVec") -> None:
        available.cpu -= self.cpu
        available.gpu -= self.gpu
        available.memory_gb -= self.memory_gb
        available.storage_gb -= self.storage_gb
        available.bandwidth_mbps -= self.bandwidth_mbps
        available.energy -= self.energy

    def release_to(self, available: "ResourceVec") -> None:
        available.cpu += self.cpu
        available.gpu += self.gpu
        available.memory_gb += self.memory_gb
        available.storage_gb += self.storage_gb
        available.bandwidth_mbps += self.bandwidth_mbps
        available.energy += self.energy


@dataclass
class Task:
    task_id: str
    task_type: str
    modality: ModalityType
    resource_requirement: ResourceVec
    priority: int
    deadline_s: float
    duration_s: float
    arrival_time_s: float = 0.0
    target_position: Position = field(default_factory=lambda: Position(0.0, 0.0, 0.0))
    data_size_mb: float = 10.0
    region_id: str = "R0"
    depends_on: List[str] = field(default_factory=list)
    depended_by: List[str] = field(default_factory=list)
    required_uav_types: List[UAVType] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_uav: Optional[str] = None
    start_time_s: Optional[float] = None
    completion_time_s: Optional[float] = None
    progress: float = 0.0

    @property
    def absolute_deadline_s(self) -> float:
        return self.arrival_time_s + self.deadline_s

    def is_terminal(self) -> bool:
        return self.status in {TaskStatus.COMPLETED, TaskStatus.TIMEOUT}


@dataclass
class UAV:
    uav_id: str
    uav_type: UAVType
    position: Position
    battery_level: float
    resources_total: ResourceVec
    resources_available: ResourceVec
    sensors: List[ModalityType]
    max_concurrent_tasks: int = 2
    cruise_speed_mps: float = 15.0
    turn_radius_m: float = 30.0
    wind_disturbance_mps: float = 2.0
    battery_capacity_wh: float = 1000.0
    sensor_fault_probability: float = 0.0
    status: UAVStatus = UAVStatus.IDLE
    current_tasks: List[str] = field(default_factory=list)
    faulted_sensors: List[ModalityType] = field(default_factory=list)

    def feature_vector(self, area_size: Tuple[float, float]) -> np.ndarray:
        area_x, area_y = area_size
        pos = np.asarray(
            [self.position.x / max(area_x, 1.0), self.position.y / max(area_y, 1.0)],
            dtype=np.float32,
        )
        res = self.resources_available.as_array()
        total = np.maximum(self.resources_total.as_array(), 1e-6)
        normalized_res = res / total
        status_flag = 0.0 if self.status == UAVStatus.FAULTED else 1.0
        return np.concatenate([pos, normalized_res, [self.battery_level, status_flag]]).astype(np.float32)


@dataclass
class LinkQuality:
    src_id: str
    dst_id: str
    bandwidth_mbps: float
    latency_ms: float
    packet_loss_rate: float
    connected: bool
    rssi_dbm: float
    mode: str
    distance_m: float = 0.0


@dataclass
class Scenario:
    name: str
    area_size: Tuple[float, float]
    civilians: List[Position]
    breach_points: List[Position]
    uavs: List[UAV]
    tasks: List[Task]
    topo_config: Dict
    regions: List[Region] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


def clone_resource(vec: ResourceVec) -> ResourceVec:
    return ResourceVec(
        cpu=vec.cpu,
        gpu=vec.gpu,
        memory_gb=vec.memory_gb,
        storage_gb=vec.storage_gb,
        bandwidth_mbps=vec.bandwidth_mbps,
        energy=vec.energy,
    )


def task_feature(task: Task, now_s: float) -> np.ndarray:
    deadline_left = max(task.absolute_deadline_s - now_s, 0.0) / max(task.deadline_s, 1.0)
    return np.concatenate(
        [
            task.resource_requirement.as_array(),
            np.asarray([task.priority / 10.0, deadline_left], dtype=np.float32),
        ]
    ).astype(np.float32)


def add_dependency(tasks_by_id: Dict[str, Task], parent_id: str, child_id: str) -> None:
    if child_id not in tasks_by_id[parent_id].depended_by:
        tasks_by_id[parent_id].depended_by.append(child_id)
    if parent_id not in tasks_by_id[child_id].depends_on:
        tasks_by_id[child_id].depends_on.append(parent_id)


def ids(items: Sequence[object]) -> List[str]:
    return [getattr(item, "task_id", getattr(item, "uav_id", str(item))) for item in items]
