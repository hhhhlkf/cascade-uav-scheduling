import unittest
from copy import deepcopy

import numpy as np

from src.env import CASCADEEnv
from src.utils.config import load_yaml_config


class CASCADEEnvTest(unittest.TestCase):
    def test_reset_and_step(self):
        env = CASCADEEnv("configs/env/scenario_s1_dongting.yaml")
        obs, info = env.reset(seed=0)
        self.assertIn("task_features", obs)
        self.assertIn("action_mask", obs)
        mask = env.get_action_mask()
        action = np.zeros_like(mask)
        if mask.any():
            row, col = np.argwhere(mask > 0.0)[0]
            action[row, col] = 1.0
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertIn("tdsr", info)
        self.assertIsInstance(float(reward), float)
        self.assertFalse(terminated and truncated)

    def test_configured_emergency_injection(self):
        config = load_yaml_config("configs/env/scenario_s1_dongting.yaml")
        config = deepcopy(config)
        config["scenario"]["emergency_injections"] = [
            {"at_s": 30, "count": 1, "task_type": "I2", "priority": 10, "deadline_s": 120}
        ]
        config["scenario"]["uav_fault_probability"] = 0.0
        env = CASCADEEnv(config)
        obs, _ = env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(np.zeros_like(obs["action_mask"]))
        event_types = [event["type"] for event in info["events"]]
        self.assertIn("emergency_injection", event_types)
        self.assertTrue(any(task_id.startswith("E00") for task_id in env.task_manager.tasks))


if __name__ == "__main__":
    unittest.main()
