from __future__ import annotations

from typing import Callable, Dict, Type

import numpy as np

from src.algorithms.base_scheduler import BaseScheduler
from src.env import CASCADEEnv
from src.evaluation.metrics import aggregate_metrics, summarize_episode


SchedulerFactory = Callable[[int, int], BaseScheduler]


def evaluate_scheduler(
    env_config: Dict | str,
    scheduler_factory: SchedulerFactory | Type[BaseScheduler],
    episodes: int = 1,
    seed: int = 0,
) -> Dict[str, float]:
    results = []
    for episode_idx in range(episodes):
        env = CASCADEEnv(env_config)
        obs, info = env.reset(seed=seed + episode_idx)
        scheduler = scheduler_factory(env.max_ready_tasks, len(env.uavs))
        scheduler.reset()
        total_reward = 0.0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            scheduler.observe(obs)
            action_mask = obs.get("action_mask", env.get_action_mask())
            action = scheduler.decide(action_mask)
            if action.shape != action_mask.shape:
                action = np.zeros_like(action_mask, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
        results.append(summarize_episode(info, total_reward))
    return aggregate_metrics(results)

