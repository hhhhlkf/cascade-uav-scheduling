import unittest

import numpy as np

from src.env.action_mask import decode_assignment_matrix


class ActionMaskTest(unittest.TestCase):
    def test_decode_ignores_zero_scores(self):
        action = np.zeros((2, 2), dtype=np.float32)
        mask = np.ones((2, 2), dtype=np.float32)
        self.assertEqual(decode_assignment_matrix(action, mask), [])

    def test_decode_returns_unique_task_uav_pairs(self):
        action = np.asarray([[0.4, 0.9], [0.8, 0.1]], dtype=np.float32)
        mask = np.ones((2, 2), dtype=np.float32)
        assignments = decode_assignment_matrix(action, mask)
        self.assertEqual(sorted(assignments), [(0, 1), (1, 0)])


if __name__ == "__main__":
    unittest.main()
