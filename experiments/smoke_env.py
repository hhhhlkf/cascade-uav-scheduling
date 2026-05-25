from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env import CASCADEEnv


def main() -> None:
    env = CASCADEEnv("configs/env/scenario_s1_dongting.yaml")
    obs, info = env.reset(seed=0)
    total_reward = 0.0
    for _ in range(10):
        mask = obs["action_mask"]
        action = np.zeros_like(mask)
        for task_idx, uav_idx in np.argwhere(mask > 0.0)[:3]:
            action[task_idx, uav_idx] = 1.0
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(
        {
            "steps": info["step_count"],
            "sim_time_s": info["sim_time_s"],
            "completed_tasks": info["completed_tasks"],
            "timed_out_tasks": info["timed_out_tasks"],
            "total_reward": round(total_reward, 4),
        }
    )


if __name__ == "__main__":
    main()
