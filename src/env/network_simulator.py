from __future__ import annotations

from collections import deque
from math import log10
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
                )
        return self.links

    def is_connected_to_command(self, node_id: str) -> bool:
        return self.shortest_path_latency(node_id, self.command_vehicle_id) < float("inf")

    def shortest_path_latency(self, src_id: str, dst_id: str) -> float:
        if src_id == dst_id:
            return 0.0
        queue = deque([(src_id, 0.0)])
        visited = {src_id}
        while queue:
            node, latency = queue.popleft()
            for next_id in self.node_ids:
                if next_id in visited:
                    continue
                link = self.links.get((node, next_id))
                if not link or not link.connected:
                    continue
                next_latency = latency + link.latency_ms + self.hop_latency_ms
                if next_id == dst_id:
                    return next_latency
                visited.add(next_id)
                queue.append((next_id, next_latency))
        return float("inf")

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

    def _build_link(self, src_id: str, dst_id: str, src: Position, dst: Position) -> LinkQuality:
        distance_m = max(src.distance_to(dst), 1.0)
        mode = "wifi6" if distance_m <= self.wifi_range_m else "lora"
        max_range = self.wifi_range_m if mode == "wifi6" else self.lora_range_m
        base_bw = self.wifi_bandwidth_mbps if mode == "wifi6" else self.lora_bandwidth_mbps
        rssi = self._rssi(distance_m)
        connected = distance_m <= max_range and rssi >= self.rssi_threshold_dbm
        if connected and self.rng.random() < self.comm_failure_rate:
            connected = False
        degradation = np.clip((rssi - self.rssi_threshold_dbm) / 30.0, 0.05, 1.0)
        bandwidth = base_bw * degradation if connected else 0.0
        propagation_ms = distance_m / 3e8 * 1000.0
        latency = self.base_latency_ms + propagation_ms + (1.0 / max(bandwidth, 1e-3) if connected else 1000.0)
        loss = float(np.clip(0.02 + (1.0 - degradation) * 0.35, 0.0, 1.0)) if connected else 1.0
        return LinkQuality(src_id, dst_id, float(bandwidth), float(latency), loss, bool(connected), float(rssi), mode)

    def _rssi(self, distance_m: float) -> float:
        frequency_mhz = 2400.0
        fspl_db = 20.0 * log10(distance_m / 1000.0) + 20.0 * log10(frequency_mhz) + 32.44
        tx_power_dbm = 23.0
        antenna_gain_db = 5.0
        return tx_power_dbm + antenna_gain_db - fspl_db

