from __future__ import annotations

from typing import Callable, Dict, Mapping, Type

import numpy as np

from src.algorithms.base_scheduler import BaseScheduler
from src.env import CASCADEEnv
from src.evaluation.metrics import aggregate_metrics, summarize_episode


SchedulerFactory = Callable[[int, int], BaseScheduler]
EpisodeCallback = Callable[[Mapping[str, float]], None]


def evaluate_scheduler(
    env_config: Dict | str,
    scheduler_factory: SchedulerFactory | Type[BaseScheduler],
    episodes: int = 1,
    seed: int = 0,
) -> Dict[str, float]:
    return evaluate_scheduler_details(env_config, scheduler_factory, episodes, seed)["summary"]


def evaluate_scheduler_details(
    env_config: Dict | str,
    scheduler_factory: SchedulerFactory | Type[BaseScheduler],
    episodes: int = 1,
    seed: int = 0,
    show_progress: bool = False,
    progress_desc: str | None = None,
    episode_callback: EpisodeCallback | None = None,
) -> Dict[str, object]:
    results = []
    episode_iter = range(episodes)
    if show_progress:
        from tqdm.auto import tqdm

        episode_iter = tqdm(episode_iter, desc=progress_desc or "episodes", leave=False)
    for episode_idx in episode_iter:
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
        episode_summary = summarize_episode(info, total_reward)
        episode_summary["episode"] = float(episode_idx)
        episode_summary["seed"] = float(seed + episode_idx)
        results.append(episode_summary)
        if episode_callback is not None:
            episode_callback(episode_summary)
    return {"summary": aggregate_metrics(results), "episodes": results}
