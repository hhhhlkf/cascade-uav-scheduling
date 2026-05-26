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

    def test_generates_ds1_regions_and_task_chains(self):
        config = load_yaml_config("configs/env/scenario_ds1_standard.yaml")
        scenario = ScenarioGenerator(config).generate()
        self.assertEqual(scenario.metadata["distribution"], "DS1")
        self.assertGreaterEqual(len(scenario.regions), 2)
        self.assertLessEqual(len(scenario.regions), 4)
        self.assertTrue(all(task.region_id for task in scenario.tasks))
        per_region_counts = {
            region.region_id: sum(1 for task in scenario.tasks if task.region_id == region.region_id)
            for region in scenario.regions
        }
        self.assertTrue(all(5 <= count <= 15 for count in per_region_counts.values()))
        self.assertTrue(any(task.task_type.startswith("A") for task in scenario.tasks))
        self.assertTrue(any(task.task_type.startswith("C") for task in scenario.tasks))


if __name__ == "__main__":
    unittest.main()
