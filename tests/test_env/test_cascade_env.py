import unittest

import numpy as np

from src.env import CASCADEEnv


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


if __name__ == "__main__":
    unittest.main()

