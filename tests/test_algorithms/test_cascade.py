import unittest

import numpy as np

from src.algorithms.cascade.hungarian_match import masked_assignment
from src.algorithms.cascade.ma3c_trainer import CASCADEMA3CScheduler, compute_gae
from src.env import CASCADEEnv

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class CascadeAlgorithmTest(unittest.TestCase):
    def test_masked_assignment_respects_mask(self):
        scores = np.asarray([[0.9, 0.1], [0.8, 0.7]], dtype=np.float32)
        mask = np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        assignments = masked_assignment(scores, mask)
        self.assertTrue(assignments)
        self.assertTrue(all(mask[task_idx, uav_idx] > 0.0 for task_idx, uav_idx in assignments))
        self.assertEqual(len({task_idx for task_idx, _ in assignments}), len(assignments))
        self.assertEqual(len({uav_idx for _, uav_idx in assignments}), len(assignments))

    def test_masked_assignment_ignores_zero_scores(self):
        scores = np.zeros((2, 2), dtype=np.float32)
        mask = np.ones((2, 2), dtype=np.float32)
        self.assertEqual(masked_assignment(scores, mask), [])

    @unittest.skipUnless(HAS_TORCH, "CASCADE network scheduler requires torch")
    def test_cascade_scheduler_network_action(self):
        env = CASCADEEnv("configs/env/scenario_s1_dongting.yaml")
        obs, _ = env.reset(seed=0)
        scheduler = CASCADEMA3CScheduler(env.max_ready_tasks, len(env.uavs))
        scheduler.observe(obs)
        action = scheduler.decide(obs["action_mask"])
        self.assertEqual(action.shape, obs["action_mask"].shape)
        self.assertTrue(np.all(action[obs["action_mask"] <= 0.0] == 0.0))

    @unittest.skipUnless(HAS_TORCH, "CASCADE network scheduler requires torch")
    def test_cascade_value_and_checkpoint_round_trip(self):
        env = CASCADEEnv("configs/env/scenario_s1_dongting.yaml")
        obs, _ = env.reset(seed=0)
        scheduler = CASCADEMA3CScheduler(env.max_ready_tasks, len(env.uavs))
        value = scheduler.value(obs)
        self.assertIsInstance(value, float)
        path = "outputs/tmp/test_cascade_scheduler.pt"
        scheduler.save(path)
        loaded = CASCADEMA3CScheduler(env.max_ready_tasks, len(env.uavs))
        loaded.load(path)
        loaded.observe(obs)
        self.assertEqual(loaded.decide(obs["action_mask"]).shape, obs["action_mask"].shape)

    def test_compute_gae_shapes_and_returns(self):
        advantages, returns = compute_gae(
            rewards=np.asarray([1.0, 0.5, -0.25], dtype=np.float32),
            values=np.asarray([0.2, 0.1, 0.0], dtype=np.float32),
            dones=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        )
        self.assertEqual(advantages.shape, (3,))
        self.assertEqual(returns.shape, (3,))
        self.assertTrue(np.isfinite(advantages).all())
        self.assertTrue(np.isfinite(returns).all())


if __name__ == "__main__":
    unittest.main()
