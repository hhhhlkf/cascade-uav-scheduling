from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional

import numpy as np

from src.utils.types import Task, TaskStatus


class TaskManager:
    """Tracks task DAG dependencies and lifecycle state."""

    def __init__(self, tasks: Iterable[Task]):
        self.tasks: Dict[str, Task] = {task.task_id: task for task in tasks}
        self._refresh_ready(now_s=0.0)

    def get_ready_tasks(self, limit: Optional[int] = None, region_id: str | None = None) -> List[Task]:
        ready = [task for task in self.tasks.values() if task.status == TaskStatus.READY]
        if region_id is not None:
            ready = [task for task in ready if task.region_id == region_id]
        ready.sort(key=lambda item: (-item.priority, item.absolute_deadline_s, item.task_id))
        return ready[:limit] if limit else ready

    def regions_with_ready_tasks(self) -> List[str]:
        regions = sorted({task.region_id for task in self.tasks.values() if task.status == TaskStatus.READY})
        return regions

    def schedule_task(self, task_id: str, uav_id: str, now_s: float) -> bool:
        task = self.tasks[task_id]
        if task.status != TaskStatus.READY:
            return False
        task.status = TaskStatus.EXECUTING
        task.assigned_uav = uav_id
        task.start_time_s = now_s
        return True

    def complete_task(self, task_id: str, now_s: float) -> None:
        task = self.tasks[task_id]
        task.status = TaskStatus.COMPLETED
        task.completion_time_s = now_s
        task.progress = 1.0
        self._refresh_ready(now_s)

    def preempt_task(self, task_id: str, now_s: float) -> None:
        task = self.tasks[task_id]
        if task.is_terminal():
            return
        task.assigned_uav = None
        task.start_time_s = None
        task.progress = 0.0
        task.status = TaskStatus.READY if now_s <= task.absolute_deadline_s else TaskStatus.TIMEOUT
        self._refresh_ready(now_s)

    def timeout_overdue(self, now_s: float) -> List[Task]:
        timed_out: List[Task] = []
        for task in self.tasks.values():
            if task.status in {TaskStatus.PENDING, TaskStatus.READY, TaskStatus.QUEUED, TaskStatus.SCHEDULED}:
                if now_s > task.absolute_deadline_s:
                    task.status = TaskStatus.TIMEOUT
                    timed_out.append(task)
        self._refresh_ready(now_s)
        return timed_out

    def inject_emergency(self, task: Task, now_s: float) -> None:
        task.arrival_time_s = now_s
        self.tasks[task.task_id] = task
        self._refresh_ready(now_s)

    def all_terminal(self) -> bool:
        return all(task.is_terminal() for task in self.tasks.values())

    def completed_tasks(self) -> List[Task]:
        return [task for task in self.tasks.values() if task.status == TaskStatus.COMPLETED]

    def timed_out_tasks(self) -> List[Task]:
        return [task for task in self.tasks.values() if task.status == TaskStatus.TIMEOUT]

    def task_dag_adjacency(self, ordered_task_ids: List[str]) -> np.ndarray:
        idx = {task_id: i for i, task_id in enumerate(ordered_task_ids)}
        adj = np.zeros((len(ordered_task_ids), len(ordered_task_ids)), dtype=np.float32)
        for task_id in ordered_task_ids:
            task = self.tasks.get(task_id)
            if not task:
                continue
            for child_id in task.depended_by:
                if child_id in idx and task_id in idx:
                    adj[idx[task_id], idx[child_id]] = 1.0
        return adj

    def get_dag_stats(self) -> Dict[str, float]:
        indegree = defaultdict(int)
        children = defaultdict(list)
        for task in self.tasks.values():
            indegree[task.task_id] += 0
            for child in task.depended_by:
                indegree[child] += 1
                children[task.task_id].append(child)
        queue = deque([task_id for task_id, degree in indegree.items() if degree == 0])
        depth = {task_id: 1 for task_id in queue}
        visited = 0
        while queue:
            task_id = queue.popleft()
            visited += 1
            for child in children[task_id]:
                depth[child] = max(depth.get(child, 1), depth[task_id] + 1)
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        stats = {
            "num_tasks": float(len(self.tasks)),
            "critical_path_length": float(max(depth.values(), default=0)),
            "parallelism": float(len(self.tasks) / max(max(depth.values(), default=1), 1)),
            "is_dag": float(visited == len(self.tasks)),
        }
        for prefix in ("A", "P", "I", "F", "C"):
            stats[f"num_{prefix.lower()}_tasks"] = float(
                sum(1 for task in self.tasks.values() if task.task_type.startswith(prefix))
            )
        return stats

    def _refresh_ready(self, now_s: float) -> None:
        completed = {task.task_id for task in self.tasks.values() if task.status == TaskStatus.COMPLETED}
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING and task.arrival_time_s <= now_s:
                if all(parent_id in completed for parent_id in task.depends_on):
                    task.status = TaskStatus.READY
