import unittest

from src.env.network_simulator import MeshNetworkSimulator
from src.env.scenario_generator import ScenarioGenerator
from src.utils.config import load_yaml_config


class NetworkSimulatorTest(unittest.TestCase):
    def test_builds_adjacency(self):
        config = load_yaml_config("configs/env/scenario_s1_dongting.yaml")
        scenario = ScenarioGenerator(config).generate()
        network = MeshNetworkSimulator(scenario.topo_config)
        network.update_topology(scenario.uavs)
        adj = network.adjacency_matrix()
        self.assertEqual(adj.shape[0], len(scenario.uavs) + 1)
        self.assertTrue((adj.diagonal() == 1.0).all())
        edge_attrs = network.edge_attr_tensor()
        self.assertEqual(edge_attrs.shape, (len(scenario.uavs) + 1, len(scenario.uavs) + 1, 4))
        paths = network.k_shortest_path_metrics(scenario.uavs[0].uav_id, network.command_vehicle_id, k=3)
        self.assertLessEqual(len(paths), 3)
        if paths:
            self.assertIn("bottleneck_bw_mbps", paths[0])


if __name__ == "__main__":
    unittest.main()
