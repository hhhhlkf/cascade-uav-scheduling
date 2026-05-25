from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.utils.types import Position, ResourceVec, Task, UAV, UAVStatus, clone_resource


POWER_W = {
    UAVStatus.IDLE: 200.0,
    UAVStatus.TRANSIT: 350.0,
    UAVStatus.COLLECTING: 215.0,
    UAVStatus.PROCESSING: 240.0,
    UAVStatus.RELAYING: 290.0,
    UAVStatus.FAULTED: 0.0,
}


@dataclass
class ActiveTask:
    task: Task
    remaining_s: float
    transit_remaining_s: float


class UAVSimulator:
    """Simplified UAV motion, battery, sensing and edge-compute simulator."""

    def __init__(self, uav: UAV):
        self.uav = uav
        self._active: Dict[str, ActiveTask] = {}
        self.energy_consumed_wh = 0.0

    @property
    def uav_id(self) -> str:
        return self.uav.uav_id

    def can_accept(self, task: Task) -> bool:
        if self.uav.status == UAVStatus.FAULTED:
            return False
        if len(self._active) >= self.uav.max_concurrent_tasks:
            return False
        if task.required_uav_types and self.uav.uav_type not in task.required_uav_types:
            return False
        if task.modality not in self.uav.sensors and task.modality.value != "MIXED":
            return False
        return task.resource_requirement.fits_within(self.uav.resources_available)

    def assign_task(self, task: Task, now_s: float) -> bool:
        if not self.can_accept(task):
            return False
        task.resource_requirement.reserve_from(self.uav.resources_available)
        distance = self.uav.position.distance_to(task.target_position)
        transit_s = distance / max(self.uav.cruise_speed_mps, 1e-6)
        remaining_s = transit_s + task.duration_s
        task.assigned_uav = self.uav.uav_id
        task.start_time_s = now_s
        self._active[task.task_id] = ActiveTask(task=task, remaining_s=remaining_s, transit_remaining_s=transit_s)
        self.uav.current_tasks.append(task.task_id)
        self.uav.status = UAVStatus.TRANSIT if transit_s > 0 else self._execution_status(task)
        return True

    def advance(self, dt_s: float) -> List[Task]:
        if self.uav.status == UAVStatus.FAULTED:
            return []
        self._consume_energy(dt_s)
        completed: List[Task] = []
        for task_id, active in list(self._active.items()):
            active.remaining_s = max(0.0, active.remaining_s - dt_s)
            active.transit_remaining_s = max(0.0, active.transit_remaining_s - dt_s)
            task = active.task
            task.progress = 1.0 - active.remaining_s / max(task.duration_s + active.transit_remaining_s, task.duration_s, 1.0)
            if active.transit_remaining_s <= 0.0:
                self._move_to(task.target_position)
            if active.remaining_s <= 0.0:
                completed.append(task)
                self._release_task(task)
        self._refresh_status()
        return completed

    def fault(self) -> List[Task]:
        interrupted = [active.task for active in self._active.values()]
        for task in interrupted:
            task.resource_requirement.release_to(self.uav.resources_available)
        self._active.clear()
        self.uav.current_tasks.clear()
        self._cap_resources()
        self.uav.status = UAVStatus.FAULTED
        return interrupted

    def available_resources(self) -> ResourceVec:
        return self.uav.resources_available

    def load_ratio(self) -> float:
        total = self.uav.resources_total.as_array()
        available = self.uav.resources_available.as_array()
        return float(np.mean(1.0 - np.divide(available, np.maximum(total, 1e-6))))

    def _consume_energy(self, dt_s: float) -> None:
        power = POWER_W.get(self.uav.status, 200.0)
        consumed_wh = power * dt_s / 3600.0
        self.energy_consumed_wh += consumed_wh
        self.uav.battery_level = max(0.0, self.uav.battery_level - consumed_wh / self.uav.battery_capacity_wh)
        self.uav.resources_available.energy = min(self.uav.resources_available.energy, self.uav.battery_level)
        if self.uav.battery_level <= 0.01:
            self.uav.status = UAVStatus.FAULTED

    def _release_task(self, task: Task) -> None:
        task.resource_requirement.release_to(self.uav.resources_available)
        self._cap_resources()
        self._active.pop(task.task_id, None)
        if task.task_id in self.uav.current_tasks:
            self.uav.current_tasks.remove(task.task_id)

    def _cap_resources(self) -> None:
        capped = np.minimum(self.uav.resources_available.as_array(), self.uav.resources_total.as_array())
        self.uav.resources_available = ResourceVec(*map(float, capped))
        self.uav.resources_available.energy = min(self.uav.resources_available.energy, self.uav.battery_level)

    def _refresh_status(self) -> None:
        if self.uav.status == UAVStatus.FAULTED:
            return
        if not self._active:
            self.uav.status = UAVStatus.IDLE
            return
        if any(active.transit_remaining_s > 0.0 for active in self._active.values()):
            self.uav.status = UAVStatus.TRANSIT
        else:
            priorities = [self._execution_status(active.task) for active in self._active.values()]
            self.uav.status = UAVStatus.PROCESSING if UAVStatus.PROCESSING in priorities else priorities[0]

    def _execution_status(self, task: Task) -> UAVStatus:
        if task.resource_requirement.gpu > 0.0:
            return UAVStatus.PROCESSING
        if task.resource_requirement.bandwidth_mbps > 30.0:
            return UAVStatus.RELAYING
        return UAVStatus.COLLECTING

    def _move_to(self, position: Position) -> None:
        self.uav.position = Position(position.x, position.y, max(position.z, self.uav.position.z))

    def clone(self) -> "UAVSimulator":
        return UAVSimulator(
            UAV(
                uav_id=self.uav.uav_id,
                uav_type=self.uav.uav_type,
                position=self.uav.position,
                battery_level=self.uav.battery_level,
                resources_total=clone_resource(self.uav.resources_total),
                resources_available=clone_resource(self.uav.resources_available),
                sensors=list(self.uav.sensors),
                max_concurrent_tasks=self.uav.max_concurrent_tasks,
                cruise_speed_mps=self.uav.cruise_speed_mps,
                battery_capacity_wh=self.uav.battery_capacity_wh,
                status=self.uav.status,
                current_tasks=list(self.uav.current_tasks),
            )
        )
