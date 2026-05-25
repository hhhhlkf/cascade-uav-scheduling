import unittest

from src.env.scenario_generator import ScenarioGenerator
from src.env.uav_simulator import UAVSimulator
from src.utils.config import load_yaml_config


class UAVSimulatorTest(unittest.TestCase):
    def test_assign_and_complete_task(self):
        config = load_yaml_config("configs/env/scenario_s1_dongting.yaml")
        scenario = ScenarioGenerator(config).generate()
        task = next(task for task in scenario.tasks if not task.depends_on)
        uav = next(UAVSimulator(candidate) for candidate in scenario.uavs if UAVSimulator(candidate).can_accept(task))
        accepted = uav.assign_task(task, now_s=0.0)
        self.assertTrue(accepted)
        completed = []
        for _ in range(500):
            completed.extend(uav.advance(30.0))
            if completed:
                break
        self.assertTrue(completed)


if __name__ == "__main__":
    unittest.main()

