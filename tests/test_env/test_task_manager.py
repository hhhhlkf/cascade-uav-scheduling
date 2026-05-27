import unittest

from src.env.scenario_generator import ScenarioGenerator
from src.env.task_manager import TaskManager
from src.utils.config import load_yaml_config
from src.utils.types import TaskStatus


class TaskManagerTest(unittest.TestCase):
    def test_ready_tasks_respect_dependencies(self):
        config = load_yaml_config("configs/env/scenario_s1_dongting.yaml")
        scenario = ScenarioGenerator(config).generate()
        manager = TaskManager(scenario.tasks)
        ready = manager.get_ready_tasks()
        self.assertTrue(ready)
        self.assertTrue(all(task.status == TaskStatus.READY for task in ready))
        self.assertEqual(manager.get_dag_stats()["is_dag"], 1.0)
        self.assertEqual(manager.get_pending_count(), len(scenario.tasks))


if __name__ == "__main__":
    unittest.main()
