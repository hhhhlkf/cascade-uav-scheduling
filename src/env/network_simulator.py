from __future__ import annotations

import heapq
from math import exp, log10
from typing import Dict, Iterable, List, Tuple

import numpy as np

from src.utils.seed import seed_everything
from src.utils.types import LinkQuality, Position, UAV


class MeshNetworkSimulator:
    """Dual-mode LoRa/WiFi mesh simulator with a compact Friis-style model."""

    def __init__(self, config: Dict):
        self.config = config
        self.command_vehicle_id = str(config.get("command_vehicle_id", "COMMAND"))
        command_pos = config.get("command_vehicle_position", [0.0, 0.0, 0.0])
        self.command_position = Position(*map(float, command_pos))
        self.wifi_range_m = float(config.get("wifi_range_m", 500.0))
        self.lora_range_m = float(config.get("lora_range_m", 10000.0))
        self.wifi_bandwidth_mbps = float(config.get("wifi_bandwidth_mbps", 600.0))
        self.lora_bandwidth_mbps = float(config.get("lora_bandwidth_mbps", 0.08))
        self.base_latency_ms = float(config.get("base_latency_ms", 4.0))
        self.hop_latency_ms = float(config.get("hop_latency_ms", 5.0))
        self.rssi_threshold_dbm = float(config.get("rssi_threshold_dbm", -92.0))
        self.comm_failure_rate = float(config.get("comm_failure_rate", 0.0))
        self.terrain_roughness = str(config.get("terrain_roughness", "flat"))
        self.path_loss_exponent = float(config.get("path_loss_exponent", 2.8))
        self.shadow_sigma_db = float(config.get("shadow_sigma_db", 6.0))
        self.k_shortest_paths = int(config.get("k_shortest_paths", 3))
        self.rng = seed_everything(config.get("seed", 0))
        self.links: Dict[Tuple[str, str], LinkQuality] = {}
        self.node_ids: List[str] = []

    def update_topology(self, uavs: Iterable[UAV]) -> Dict[Tuple[str, str], LinkQuality]:
        nodes = {self.command_vehicle_id: self.command_position}
        nodes.update({uav.uav_id: uav.position for uav in uavs})
        self.node_ids = list(nodes.keys())
        self.links.clear()
        for src_idx, src_id in enumerate(self.node_ids):
            for dst_id in self.node_ids[src_idx + 1 :]:
                link = self._build_link(src_id, dst_id, nodes[src_id], nodes[dst_id])
                self.links[(src_id, dst_id)] = link
                self.links[(dst_id, src_id)] = LinkQuality(
                    src_id=dst_id,
                    dst_id=src_id,
                    bandwidth_mbps=link.bandwidth_mbps,
                    latency_ms=link.latency_ms,
                    packet_loss_rate=link.packet_loss_rate,
                    connected=link.connected,
                    rssi_dbm=link.rssi_dbm,
                    mode=link.mode,
                    distance_m=link.distance_m,
                )
        return self.links

    def is_connected_to_command(self, node_id: str) -> bool:
        return self.shortest_path_latency(node_id, self.command_vehicle_id) < float("inf")

    def shortest_path_latency(self, src_id: str, dst_id: str) -> float:
        metrics = self.shortest_path_metrics(src_id, dst_id)
        return float(metrics["latency_ms"])

    def shortest_path_metrics(self, src_id: str, dst_id: str) -> Dict[str, float]:
        path = self._dijkstra_path_metrics(src_id, dst_id)
        if path is not None:
            return path
        direct = self.links.get((src_id, dst_id))
        return {"latency_ms": float("inf"), "hop_count": float("inf"), "bottleneck_bw_mbps": 0.0, "has_direct_link": float(bool(direct and direct.connected))}

    def k_shortest_path_metrics(self, src_id: str, dst_id: str, k: int | None = None) -> List[Dict[str, float]]:
        if src_id == dst_id:
            return [{"latency_ms": 0.0, "hop_count": 0.0, "bottleneck_bw_mbps": 0.0, "has_direct_link": 1.0}]
        limit = k or self.k_shortest_paths
        direct = self.links.get((src_id, dst_id))
        queue: list[tuple[float, str, int, float, tuple[str, ...]]] = [(0.0, src_id, 0, float("inf"), (src_id,))]
        results: List[Dict[str, float]] = []
        expansions = 0
        max_expansions = max(200, len(self.node_ids) * len(self.node_ids) * max(limit, 1))
        max_hops = min(5, max(len(self.node_ids) - 1, 1))
        while queue and len(results) < limit and expansions < max_expansions:
            latency, node, hops, bottleneck, path = heapq.heappop(queue)
            expansions += 1
            if node == dst_id:
                results.append(
                    {
                        "latency_ms": float(latency),
                        "hop_count": float(hops),
                        "bottleneck_bw_mbps": float(0.0 if bottleneck == float("inf") else bottleneck),
                        "has_direct_link": float(bool(direct and direct.connected)),
                    }
                )
                continue
            if hops >= max_hops:
                continue
            for next_id in self.node_ids:
                if next_id in path:
                    continue
                link = self.links.get((node, next_id))
                if not link or not link.connected:
                    continue
                next_latency = latency + link.latency_ms + self.hop_latency_ms
                heapq.heappush(queue, (next_latency, next_id, hops + 1, min(bottleneck, link.bandwidth_mbps), path + (next_id,)))
        return results

    def _dijkstra_path_metrics(self, src_id: str, dst_id: str) -> Dict[str, float] | None:
        if src_id == dst_id:
            return {"latency_ms": 0.0, "hop_count": 0.0, "bottleneck_bw_mbps": 0.0, "has_direct_link": 1.0}
        direct = self.links.get((src_id, dst_id))
        queue: list[tuple[float, str, int, float]] = [(0.0, src_id, 0, float("inf"))]
        best_latency = {src_id: 0.0}
        while queue:
            latency, node, hops, bottleneck = heapq.heappop(queue)
            if node == dst_id:
                return {
                    "latency_ms": float(latency),
                    "hop_count": float(hops),
                    "bottleneck_bw_mbps": float(0.0 if bottleneck == float("inf") else bottleneck),
                    "has_direct_link": float(bool(direct and direct.connected)),
                }
            if latency > best_latency.get(node, float("inf")):
                continue
            for next_id in self.node_ids:
                link = self.links.get((node, next_id))
                if not link or not link.connected:
                    continue
                next_latency = latency + link.latency_ms + self.hop_latency_ms
                if next_latency >= best_latency.get(next_id, float("inf")):
                    continue
                best_latency[next_id] = next_latency
                heapq.heappush(queue, (next_latency, next_id, hops + 1, min(bottleneck, link.bandwidth_mbps)))
        return None

    def multihop_feature_matrix(self, uav_ids: List[str]) -> np.ndarray:
        features = np.zeros((len(uav_ids), 4), dtype=np.float32)
        for idx, uav_id in enumerate(uav_ids):
            metrics = self.shortest_path_metrics(uav_id, self.command_vehicle_id)
            latency = metrics["latency_ms"]
            hop_count = metrics["hop_count"]
            features[idx] = np.asarray(
                [
                    0.0 if hop_count == float("inf") else min(hop_count / 5.0, 1.0),
                    min(metrics["bottleneck_bw_mbps"] / max(self.wifi_bandwidth_mbps, 1e-6), 1.0),
                    1.0 if latency == float("inf") else min(latency / 1000.0, 1.0),
                    metrics["has_direct_link"],
                ],
                dtype=np.float32,
            )
        return features

    def get_link(self, src_id: str, dst_id: str) -> LinkQuality | None:
        return self.links.get((src_id, dst_id))

    def adjacency_matrix(self, node_order: List[str] | None = None) -> np.ndarray:
        order = node_order or self.node_ids
        matrix = np.zeros((len(order), len(order)), dtype=np.float32)
        for i, src in enumerate(order):
            for j, dst in enumerate(order):
                if i == j:
                    matrix[i, j] = 1.0
                    continue
                link = self.links.get((src, dst))
                if link and link.connected:
                    matrix[i, j] = min(link.bandwidth_mbps / max(self.wifi_bandwidth_mbps, 1e-6), 1.0)
        return matrix

    def edge_attr_tensor(self, node_order: List[str] | None = None) -> np.ndarray:
        order = node_order or self.node_ids
        attrs = np.zeros((len(order), len(order), 4), dtype=np.float32)
        for i, src in enumerate(order):
            for j, dst in enumerate(order):
                if i == j:
                    attrs[i, j] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                    continue
                link = self.links.get((src, dst))
                if not link:
                    continue
                attrs[i, j] = np.asarray(
                    [
                        min(link.distance_m / max(self.lora_range_m, 1e-6), 1.0),
                        min(link.bandwidth_mbps / max(self.wifi_bandwidth_mbps, 1e-6), 1.0),
                        min(link.latency_ms / 1000.0, 1.0),
                        float(link.connected),
                    ],
                    dtype=np.float32,
                )
        return attrs

    def _build_link(self, src_id: str, dst_id: str, src: Position, dst: Position) -> LinkQuality:
        distance_m = max(src.distance_to(dst), 1.0)
        mode = "wifi6" if distance_m <= self.wifi_range_m else "lora"
        max_range = self.wifi_range_m if mode == "wifi6" else self.lora_range_m
        base_bw = self.wifi_bandwidth_mbps if mode == "wifi6" else self.lora_bandwidth_mbps
        rssi = self._rssi(distance_m)
        los = self.rng.random() <= self._los_probability(distance_m)
        if not los:
            rssi -= float(self.rng.uniform(20.0, 30.0))
        connected = distance_m <= max_range and rssi >= self.rssi_threshold_dbm
        if connected and self.rng.random() < self.comm_failure_rate:
            connected = False
        degradation = np.clip((rssi - self.rssi_threshold_dbm) / 30.0, 0.05, 1.0)
        bandwidth = base_bw * degradation if connected else 0.0
        propagation_ms = distance_m / 3e8 * 1000.0
        latency = self.base_latency_ms + propagation_ms + (1.0 / max(bandwidth, 1e-3) if connected else 1000.0)
        loss = float(np.clip(0.02 + (1.0 - degradation) * 0.35, 0.0, 1.0)) if connected else 1.0
        return LinkQuality(src_id, dst_id, float(bandwidth), float(latency), loss, bool(connected), float(rssi), mode, float(distance_m))

    def _rssi(self, distance_m: float) -> float:
        reference_distance_m = 1.0
        path_loss = 40.0 + 10.0 * self.path_loss_exponent * log10(max(distance_m, reference_distance_m) / reference_distance_m)
        shadow = float(self.rng.normal(0.0, self.shadow_sigma_db))
        tx_power_dbm = 20.0
        return tx_power_dbm - path_loss - shadow

    def _los_probability(self, distance_m: float) -> float:
        if self.terrain_roughness == "flat":
            return 1.0
        scale = 500.0 if self.terrain_roughness == "hilly" else 300.0
        return float(np.clip(exp(-distance_m / scale), 0.05, 1.0))
