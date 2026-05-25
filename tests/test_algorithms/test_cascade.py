import unittest

import numpy as np

from src.algorithms.cascade.hungarian_match import masked_assignment
from src.algorithms.cascade.ma3c_trainer import CASCADEMA3CScheduler
from src.env import CASCADEEnv


class CascadeAlgorithmTest(unittest.TestCase):
    def test_masked_assignment_respects_mask(self):
        scores = np.asarray([[0.9, 0.1], [0.8, 0.7]], dtype=np.float32)
        mask = np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        assignments = masked_assignment(scores, mask)
        self.assertTrue(assignments)
        self.assertTrue(all(mask[task_idx, uav_idx] > 0.0 for task_idx, uav_idx in assignments))
        self.assertEqual(len({task_idx for task_idx, _ in assignments}), len(assignments))
        self.assertEqual(len({uav_idx for _, uav_idx in assignments}), len(assignments))

    def test_cascade_scheduler_placeholder_action(self):
        env = CASCADEEnv("configs/env/scenario_s1_dongting.yaml")
        obs, _ = env.reset(seed=0)
        scheduler = CASCADEMA3CScheduler(env.max_ready_tasks, len(env.uavs))
        scheduler.observe(obs)
        action = scheduler.decide(obs["action_mask"])
        self.assertEqual(action.shape, obs["action_mask"].shape)


if __name__ == "__main__":
    unittest.main()

