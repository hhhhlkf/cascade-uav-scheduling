from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import numpy as np


class BaseScheduler(ABC):
    """Unified interface for all CASCADE schedulers."""

    def __init__(self, max_ready_tasks: int, num_uavs: int):
        self.max_ready_tasks = int(max_ready_tasks)
        self.num_uavs = int(num_uavs)
        self.last_obs: Dict[str, np.ndarray] | None = None

    def observe(self, obs: Dict[str, np.ndarray]) -> None:
        self.last_obs = obs

    @abstractmethod
    def decide(self, action_mask: np.ndarray) -> np.ndarray:
        """Return an assignment score matrix with shape [max_ready_tasks, num_uavs]."""

    def learn(self, batch: Dict) -> Dict[str, float]:
        return {}

    def reset(self) -> None:
        self.last_obs = None

    def save(self, path: str | Path) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not persist state")

    def load(self, path: str | Path) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not persist state")

    def _empty_action(self, action_mask: np.ndarray) -> np.ndarray:
        return np.zeros_like(action_mask, dtype=np.float32)

