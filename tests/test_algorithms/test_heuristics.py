import unittest

from src.algorithms.heuristic import GreedyScheduler, HEFTScheduler, MinLoadScheduler, RoundRobinScheduler
from src.env import CASCADEEnv


class HeuristicSchedulerTest(unittest.TestCase):
    def test_heuristics_return_mask_shaped_actions(self):
        env = CASCADEEnv("configs/env/scenario_s1_dongting.yaml")
        obs, _ = env.reset(seed=0)
        mask = obs["action_mask"]
        for scheduler_cls in [GreedyScheduler, MinLoadScheduler, RoundRobinScheduler, HEFTScheduler]:
            scheduler = scheduler_cls(env.max_ready_tasks, len(env.uavs))
            scheduler.observe(obs)
            action = scheduler.decide(mask)
            self.assertEqual(action.shape, mask.shape)
            self.assertTrue(((action > 0.0) <= (mask > 0.0)).all())


if __name__ == "__main__":
    unittest.main()

