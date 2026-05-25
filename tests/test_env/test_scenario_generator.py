import unittest

from src.env.scenario_generator import ScenarioGenerator
from src.utils.config import load_yaml_config


class ScenarioGeneratorTest(unittest.TestCase):
    def test_generates_s1_counts(self):
        config = load_yaml_config("configs/env/scenario_s1_dongting.yaml")
        scenario = ScenarioGenerator(config).generate()
        self.assertEqual(scenario.name, "s1_dongting")
        self.assertEqual(len(scenario.civilians), 80)
        self.assertEqual(len(scenario.tasks), 30)
        self.assertEqual(len(scenario.uavs), 10)


if __name__ == "__main__":
    unittest.main()

