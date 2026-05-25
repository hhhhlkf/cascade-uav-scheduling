import unittest

from src.algorithms.heuristic import GreedyScheduler
from src.evaluation import evaluate_scheduler


class EvaluatorTest(unittest.TestCase):
    def test_evaluate_scheduler_returns_summary(self):
        summary = evaluate_scheduler("configs/env/scenario_s1_dongting.yaml", GreedyScheduler, episodes=1, seed=0)
        self.assertIn("total_reward_mean", summary)
        self.assertIn("completion_ratio_mean", summary)


if __name__ == "__main__":
    unittest.main()

