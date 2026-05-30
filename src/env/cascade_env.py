from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised only when optional deps are missing.
    class _Env:
        pass

    class _Box:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

        def sample(self):
            return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)

    class _Dict(dict):
        def __init__(self, spaces_dict):
            super().__init__(spaces_dict)
            self.spaces = spaces_dict

    class _Spaces:
        Box = _Box
        Dict = _Dict

    class _Gym:
        Env = _Env

    gym = _Gym()
    spaces = _Spaces()

try:
    import simpy
except ImportError:  # pragma: no cover
    simpy = None

from src.env.action_mask import compute_action_mask, decode_assignment_matrix
from src.env.network_simulator import MeshNetworkSimulator
from src.env.reward import compute_reward
from src.env.scenario_generator import ScenarioGenerator
from src.env.task_manager import TaskManager
from src.env.uav_simulator import UAVSimulator
from src.utils.config import load_yaml_config
from src.utils.seed import seed_everything
from src.utils.types import ModalityType, Position, ResourceVec, Task, UAVType, task_feature


class CASCADEEnv(gym.Env):
    """Gymnasium-compatible CASCADE disaster scheduling simulation environment."""

    metadata = {"render_modes": []}

    def __init__(self, config: Dict | str | Path):
        super().__init__()
        self.config = load_yaml_config(config) if isinstance(config, (str, Path)) else config
        self.env_cfg = self.config.get("env", {})
        self.reward_cfg = self.config.get("reward", {})
        self.max_ready_tasks = int(self.env_cfg.get("max_ready_tasks", 16))
        self.step_seconds = float(self.env_cfg.get("step_seconds", 30.0))
        self.max_steps = int(self.env_cfg.get("max_steps", 200))
        self.region_selection = str(self.env_cfg.get("region_selection", "random_ready"))
        self.rng = seed_everything(self.env_cfg.get("seed", 0))
        self.simpy_env = simpy.Environment() if simpy else None
        self.sim_time_s = 0.0
        self.step_count = 0
        self.scenario = None
        self.task_manager: TaskManager
        self.uavs: List[UAVSimulator] = []
        self.network: MeshNetworkSimulator
        self.max_tasks_total = 1
        self.max_nodes = 1
        self.max_regions = 1
        self.current_region_id = "R0"
        self._triggered_emergencies: set[int] = set()
        self._faulted_uav_ids: set[str] = set()
        self._build_world()
        self.observation_space = self._build_observation_space()
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.max_ready_tasks, len(self.uavs)),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict]:
        if seed is not None:
            self.config.setdefault("env", {})["seed"] = seed
            self.rng = seed_everything(seed)
        self.sim_time_s = 0.0
        self.step_count = 0
        self.simpy_env = simpy.Environment() if simpy else None
        self._triggered_emergencies = set()
        self._faulted_uav_ids = set()
        self._build_world()
        self._select_current_region(force=True)
        obs = self._observe()
        return obs, self._info()

    def step(self, action: np.ndarray):
        self._select_current_region()
        ready_tasks = self.task_manager.get_ready_tasks(self.max_ready_tasks, region_id=self.current_region_id)
        mask = self.get_action_mask()
        assignments = decode_assignment_matrix(np.asarray(action, dtype=np.float32), mask)
        invalid_count = 0
        for task_idx, uav_idx in assignments:
            if task_idx >= len(ready_tasks) or mask[task_idx, uav_idx] <= 0.0:
                invalid_count += 1
                continue
            task = ready_tasks[task_idx]
            uav = self.uavs[uav_idx]
            if uav.assign_task(task, self.sim_time_s):
                self.task_manager.schedule_task(task.task_id, uav.uav_id, self.sim_time_s)
            else:
                invalid_count += 1

        self._advance_clock(self.step_seconds)
        self.step_count += 1
        completed = []
        for uav in self.uavs:
            completed.extend(uav.advance(self.step_seconds))
        for task in completed:
            self.task_manager.complete_task(task.task_id, self.sim_time_s)
        timed_out = self.task_manager.timeout_overdue(self.sim_time_s)
        events = self._maybe_inject_events()
        self.network.update_topology([uav.uav for uav in self.uavs])
        self._select_current_region(force=True)

        next_ready = self.task_manager.get_ready_tasks(self.max_ready_tasks, region_id=self.current_region_id)
        reward, reward_parts = compute_reward(
            self.uavs,
            self.network,
            next_ready,
            invalid_count,
            self.sim_time_s,
            self.reward_cfg,
        )
        completed_bonus = float(self.reward_cfg.get("completed_bonus", 1.0)) * len(completed)
        timeout_penalty = float(self.reward_cfg.get("timeout_penalty", 2.0)) * len(timed_out)
        reward += completed_bonus - timeout_penalty
        reward_parts.update(
            {
                "reward_completed_bonus": completed_bonus,
                "reward_timeout_penalty": timeout_penalty,
            }
        )
        terminated = self.task_manager.all_terminal()
        truncated = self.step_count >= self.max_steps or self.sim_time_s >= self._simulation_duration_s()
        obs = self._observe()
        info = self._info()
        info.update(reward_parts)
        info.update(
            {
                "assigned_count": len(assignments) - invalid_count,
                "completed_this_step": len(completed),
                "timed_out_this_step": len(timed_out),
                "events": events,
            }
        )
        return obs, reward, terminated, truncated, info

    def _advance_clock(self, dt_s: float) -> None:
        if self.simpy_env is not None:
            self.simpy_env.run(until=self.simpy_env.now + dt_s)
            self.sim_time_s = float(self.simpy_env.now)
        else:
            self.sim_time_s += dt_s

    def get_action_mask(self) -> np.ndarray:
        self._select_current_region()
        ready_tasks = self.task_manager.get_ready_tasks(self.max_ready_tasks, region_id=self.current_region_id)
        return compute_action_mask(ready_tasks, self.uavs, self.network, self.max_ready_tasks)

    def _build_world(self) -> None:
        self.scenario = ScenarioGenerator(self.config).generate()
        self.env_cfg = self.config.get("env", {})
        self.region_selection = str(self.env_cfg.get("region_selection", "random_ready"))
        self.uavs = [UAVSimulator(uav) for uav in self.scenario.uavs]
        self.task_manager = TaskManager(self.scenario.tasks)
        self.network = MeshNetworkSimulator(self.scenario.topo_config)
        self.network.update_topology([uav.uav for uav in self.uavs])
        self.max_regions = max(len(self.scenario.regions), 1)
        self.current_region_id = self.scenario.regions[0].region_id if self.scenario.regions else "R0"
        planned_emergencies = sum(
            int(item.get("count", 1)) for item in self.config.get("scenario", {}).get("emergency_injections", [])
        )
        self.max_tasks_total = max(len(self.scenario.tasks) + planned_emergencies, self.max_ready_tasks)
        self.max_nodes = len(self.uavs) + 1

    def _maybe_inject_events(self) -> List[Dict]:
        events: List[Dict] = []
        events.extend(self._maybe_inject_faults())
        events.extend(self._maybe_inject_sensor_faults())
        events.extend(self._maybe_inject_emergencies())
        return events

    def _maybe_inject_faults(self) -> List[Dict]:
        probability = float(self.scenario.metadata.get("uav_fault_probability", 0.0))
        if probability <= 0.0:
            return []
        per_step_probability = probability / max(self.max_steps, 1)
        events: List[Dict] = []
        for uav in self.uavs:
            if uav.uav_id in self._faulted_uav_ids:
                continue
            if self.rng.random() >= per_step_probability:
                continue
            interrupted = uav.fault()
            self._faulted_uav_ids.add(uav.uav_id)
            for task in interrupted:
                self.task_manager.preempt_task(task.task_id, self.sim_time_s)
            events.append(
                {
                    "type": "uav_fault",
                    "uav_id": uav.uav_id,
                    "interrupted_tasks": [task.task_id for task in interrupted],
                }
            )
        return events

    def _maybe_inject_sensor_faults(self) -> List[Dict]:
        probability = float(self.scenario.metadata.get("sensor_fault_probability", 0.0))
        if probability <= 0.0:
            return []
        per_step_probability = probability / max(self.max_steps, 1)
        events: List[Dict] = []
        for uav in self.uavs:
            if self.rng.random() >= per_step_probability:
                continue
            sensor = uav.trigger_sensor_fault()
            if sensor:
                events.append({"type": "sensor_fault", "uav_id": uav.uav_id, "sensor": sensor})
        return events

    def _maybe_inject_emergencies(self) -> List[Dict]:
        injections = self.config.get("scenario", {}).get("emergency_injections", [])
        events: List[Dict] = []
        for idx, injection in enumerate(injections):
            if idx in self._triggered_emergencies:
                continue
            if self.sim_time_s < float(injection.get("at_s", 0.0)):
                continue
            self._triggered_emergencies.add(idx)
            count = int(injection.get("count", 1))
            task_ids: List[str] = []
            for local_idx in range(count):
                task = self._build_emergency_task(injection, idx, local_idx)
                self.task_manager.inject_emergency(task, self.sim_time_s)
                task_ids.append(task.task_id)
            events.append({"type": "emergency_injection", "task_ids": task_ids})
        return events

    def _build_emergency_task(self, injection: Dict, injection_idx: int, local_idx: int) -> Task:
        task_type = str(injection.get("task_type", "I2"))
        is_comm = task_type.startswith("C")
        is_rescue = task_type.startswith("F")
        modality = ModalityType.RGB if is_comm else ModalityType.MIXED if is_rescue else ModalityType.THERMAL
        required_types = [UAVType.COMM] if is_comm else [UAVType.RESCUE] if is_rescue else [UAVType.VISION, UAVType.RESCUE]
        resource = (
            ResourceVec(0.20, 0.00, 0.5, 2.0, 60.0, 0.05)
            if is_comm
            else ResourceVec(0.45, 0.35, 3.0, 12.0, 20.0, 0.10)
            if is_rescue
            else ResourceVec(0.25, 0.20, 1.0, 4.0, 8.0, 0.06)
        )
        return Task(
            task_id=f"E{injection_idx:02d}-{local_idx:02d}-{int(self.sim_time_s)}",
            task_type=task_type,
            modality=modality,
            resource_requirement=resource,
            priority=int(injection.get("priority", 10)),
            deadline_s=float(injection.get("deadline_s", 120.0)),
            duration_s=float(injection.get("duration_s", 90.0)),
            arrival_time_s=self.sim_time_s,
            target_position=Position(
                float(self.rng.uniform(0.0, self.scenario.area_size[0])),
                float(self.rng.uniform(0.0, self.scenario.area_size[1])),
                0.0,
            ),
            data_size_mb=float(injection.get("data_size_mb", 80.0)),
            region_id=self.current_region_id,
            required_uav_types=required_types,
        )

    def _observe(self) -> Dict[str, np.ndarray]:
        self._select_current_region()
        ready_tasks = self.task_manager.get_ready_tasks(self.max_ready_tasks, region_id=self.current_region_id)
        task_features = np.zeros((self.max_ready_tasks, 8), dtype=np.float32)
        priority_features = np.zeros((self.max_ready_tasks, 2), dtype=np.float32)
        task_mask = np.zeros((self.max_ready_tasks,), dtype=np.float32)
        for idx, task in enumerate(ready_tasks[: self.max_ready_tasks]):
            task_features[idx] = task_feature(task, self.sim_time_s)
            priority_features[idx] = np.asarray(
                [task.priority / 10.0, max(task.absolute_deadline_s - self.sim_time_s, 0.0) / max(task.deadline_s, 1.0)],
                dtype=np.float32,
            )
            task_mask[idx] = 1.0

        uav_features = np.zeros((len(self.uavs), 10), dtype=np.float32)
        for idx, uav in enumerate(self.uavs):
            uav_features[idx] = uav.uav.feature_vector(self.scenario.area_size)

        node_order = [self.network.command_vehicle_id] + [uav.uav_id for uav in self.uavs]
        network_adj = self.network.adjacency_matrix(node_order)
        network_edge_attrs = self.network.edge_attr_tensor(node_order)
        multihop_features = self.network.multihop_feature_matrix([uav.uav_id for uav in self.uavs])
        task_ids = list(self.task_manager.tasks.keys())
        task_dag_adj = self.task_manager.task_dag_adjacency(task_ids)
        padded_dag = np.zeros((self.max_tasks_total, self.max_tasks_total), dtype=np.float32)
        padded_dag[: task_dag_adj.shape[0], : task_dag_adj.shape[1]] = task_dag_adj
        return {
            "task_features": task_features,
            "uav_features": uav_features,
            "network_adj": network_adj.astype(np.float32),
            "network_edge_attrs": network_edge_attrs.astype(np.float32),
            "multihop_features": multihop_features.astype(np.float32),
            "current_region_features": self._current_region_feature(),
            "task_dag_adj": padded_dag,
            "task_dag_mask": task_mask,
            "priority_features": priority_features,
            "action_mask": self.get_action_mask(),
        }

    def _build_observation_space(self):
        return spaces.Dict(
            {
                "task_features": spaces.Box(0.0, np.inf, shape=(self.max_ready_tasks, 8), dtype=np.float32),
                "uav_features": spaces.Box(0.0, np.inf, shape=(len(self.uavs), 10), dtype=np.float32),
                "network_adj": spaces.Box(0.0, 1.0, shape=(self.max_nodes, self.max_nodes), dtype=np.float32),
                "network_edge_attrs": spaces.Box(0.0, 1.0, shape=(self.max_nodes, self.max_nodes, 4), dtype=np.float32),
                "multihop_features": spaces.Box(0.0, 1.0, shape=(len(self.uavs), 4), dtype=np.float32),
                "current_region_features": spaces.Box(0.0, 1.0, shape=(7,), dtype=np.float32),
                "task_dag_adj": spaces.Box(0.0, 1.0, shape=(self.max_tasks_total, self.max_tasks_total), dtype=np.float32),
                "task_dag_mask": spaces.Box(0.0, 1.0, shape=(self.max_ready_tasks,), dtype=np.float32),
                "priority_features": spaces.Box(0.0, 1.0, shape=(self.max_ready_tasks, 2), dtype=np.float32),
                "action_mask": spaces.Box(0.0, 1.0, shape=(self.max_ready_tasks, len(self.uavs)), dtype=np.float32),
            }
        )

    def _info(self) -> Dict:
        completed = self.task_manager.completed_tasks()
        deadline_tasks = [task for task in self.task_manager.tasks.values() if task.deadline_s > 0]
        on_time = [task for task in completed if task.completion_time_s is not None and task.completion_time_s <= task.absolute_deadline_s]
        rescue_completed = [task for task in completed if task.task_type.startswith(("I", "F"))]
        return {
            "sim_time_s": self.sim_time_s,
            "step_count": self.step_count,
            "current_region": self.current_region_id,
            "completed_tasks": len(completed),
            "timed_out_tasks": len(self.task_manager.timed_out_tasks()),
            "ready_tasks": len(self.task_manager.get_ready_tasks(region_id=self.current_region_id)),
            "pending_tasks": self.task_manager.get_pending_count(),
            "tdsr": len(on_time) / max(len(deadline_tasks), 1),
            "rpdr_proxy": min(len(rescue_completed) / max(float(self.scenario.metadata.get("num_civilians", 1)), 1.0), 1.0),
            "dag_stats": self.task_manager.get_dag_stats(),
            **self.get_episode_metrics(),
        }

    def _simulation_duration_s(self) -> float:
        return float(self.scenario.metadata.get("simulation_duration_s", self.max_steps * self.step_seconds))

    def get_episode_metrics(self) -> Dict[str, float]:
        completed = self.task_manager.completed_tasks()
        durations = [
            task.completion_time_s - task.start_time_s
            for task in completed
            if task.start_time_s is not None and task.completion_time_s is not None
        ]
        starts = [task.start_time_s for task in completed if task.start_time_s is not None]
        finishes = [task.completion_time_s for task in completed if task.completion_time_s is not None]
        metrics: Dict[str, float] = {
            "makespan_s": float(max(finishes) - min(starts)) if starts and finishes else 0.0,
            "atct_s": float(np.mean(durations)) if durations else 0.0,
            "gpu_util_mean": float(np.mean([uav.mean_gpu_utilization() for uav in self.uavs])) if self.uavs else 0.0,
            "gpu_util_std": float(np.std([uav.mean_gpu_utilization() for uav in self.uavs])) if self.uavs else 0.0,
            "memory_util_mean": float(np.mean([uav.mean_memory_utilization() for uav in self.uavs])) if self.uavs else 0.0,
        }
        for prefix in ("A", "P", "I", "F", "C"):
            prefix_durations = [
                task.completion_time_s - task.start_time_s
                for task in completed
                if task.task_type.startswith(prefix)
                and task.start_time_s is not None
                and task.completion_time_s is not None
            ]
            metrics[f"ptct_{prefix.lower()}_s"] = float(np.mean(prefix_durations)) if prefix_durations else 0.0
        return metrics

    def _select_current_region(self, force: bool = False) -> None:
        ready_regions = self.task_manager.regions_with_ready_tasks()
        if not ready_regions:
            return
        if not force and self.current_region_id in ready_regions:
            return
        self.current_region_id = self._choose_ready_region(ready_regions)

    def _choose_ready_region(self, ready_regions: List[str]) -> str:
        if self.region_selection in {"first_ready", "fixed"}:
            return ready_regions[0]
        if self.region_selection not in {"random_ready", "random"}:
            raise ValueError(f"Unsupported region_selection: {self.region_selection}")
        region_idx = int(self.rng.integers(0, len(ready_regions)))
        return ready_regions[region_idx]

    def _current_region_feature(self) -> np.ndarray:
        for region in self.scenario.regions:
            if region.region_id == self.current_region_id:
                return region.feature_vector()
        return np.zeros((7,), dtype=np.float32)
