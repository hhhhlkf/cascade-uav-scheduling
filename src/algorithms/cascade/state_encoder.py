from __future__ import annotations

from typing import Dict

import numpy as np

from src.algorithms.cascade.gnn_encoder import build_graph_encoder, build_mlp_encoder
from src.algorithms.cascade.mhsa_fusion import build_mhsa_fusion


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("CASCADE state encoder requires torch. Install requirements.txt first.") from exc
    return torch, nn


class CASCADEStateEncoder:
    """Torch-backed encoder for CASCADE observations.

    The public wrapper accepts numpy observations and returns torch tensors, so
    schedulers can use it directly while tests remain framework-light.
    """

    def __init__(
        self,
        max_ready_tasks: int,
        num_uavs: int,
        max_nodes: int,
        max_tasks_total: int,
        token_dim: int = 64,
        device: str | None = None,
    ):
        torch, nn = _require_torch()
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        module_cls = _build_state_encoder_module(torch, nn)
        self.module = module_cls(
            max_ready_tasks=max_ready_tasks,
            num_uavs=num_uavs,
            max_nodes=max_nodes,
            max_tasks_total=max_tasks_total,
            token_dim=token_dim,
        ).to(self.device)

    @property
    def global_dim(self) -> int:
        return self.module.global_dim

    @property
    def local_dim(self) -> int:
        return self.module.local_dim

    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.module.load_state_dict(state_dict)

    def encode(self, obs: Dict[str, np.ndarray]):
        tensors = {key: self._to_tensor(value) for key, value in obs.items() if isinstance(value, np.ndarray)}
        with self.torch.no_grad():
            return self.module(tensors)

    def encode_train(self, obs: Dict[str, np.ndarray]):
        tensors = {key: self._to_tensor(value) for key, value in obs.items() if isinstance(value, np.ndarray)}
        return self.module(tensors)

    def _to_tensor(self, value: np.ndarray):
        tensor = self.torch.as_tensor(value, dtype=self.torch.float32, device=self.device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() in {2, 3}:
            tensor = tensor.unsqueeze(0)
        return tensor


def _build_state_encoder_module(torch, nn):
    class _CASCADEStateEncoderModule(nn.Module):
        def __init__(
            self,
            max_ready_tasks: int,
            num_uavs: int,
            max_nodes: int,
            max_tasks_total: int,
            token_dim: int,
        ):
            super().__init__()
            self.torch = torch
            self.max_ready_tasks = max_ready_tasks
            self.num_uavs = num_uavs
            self.max_nodes = max_nodes
            self.max_tasks_total = max_tasks_total
            self.token_dim = token_dim
            self.global_dim = token_dim * 5
            self.local_dim = 14

            self.task_encoder = build_graph_encoder(8, hidden_dim=128, output_dim=token_dim, num_layers=2)
            self.net_encoder = build_graph_encoder(14, hidden_dim=128, output_dim=token_dim, num_layers=2)
            self.resource_encoder = build_mlp_encoder(num_uavs * 10, hidden_dim=128, output_dim=token_dim)
            self.modal_encoder = build_mlp_encoder(max_ready_tasks * 8, hidden_dim=128, output_dim=token_dim)
            self.hop_encoder = build_mlp_encoder(num_uavs * 4, hidden_dim=128, output_dim=token_dim)
            self.fusion = build_mhsa_fusion(embed_dim=token_dim, num_heads=4)
            self.norm = nn.LayerNorm(token_dim)

        def forward(self, obs):
            task_features = obs["task_features"]
            uav_features = obs["uav_features"]
            task_adj = obs["task_dag_adj"]
            net_adj = obs["network_adj"]
            edge_attrs = obs.get("network_edge_attrs")
            multihop = obs["multihop_features"]

            batch = task_features.shape[0]
            task_nodes = self._pad_last_dim(task_features, 8)
            task_adj_ready = self._ready_task_adjacency(task_adj, task_nodes.shape[1])
            task_mask = (task_nodes.abs().sum(dim=-1) > 0.0).float()
            task_token = self.task_encoder(task_nodes, task_adj_ready, task_mask)

            net_nodes = self._network_nodes(uav_features, multihop, edge_attrs)
            net_token = self.net_encoder(net_nodes, net_adj, None)

            resource_token = self.resource_encoder(uav_features.reshape(batch, -1))
            modal_token = self.modal_encoder(task_features.reshape(batch, -1))
            hop_token = self.hop_encoder(multihop.reshape(batch, -1))

            tokens = self.torch.stack([task_token, net_token, resource_token, modal_token, hop_token], dim=1)
            fused, _ = self.fusion(tokens, tokens, tokens, need_weights=False)
            fused = self.norm(fused + tokens)
            global_state = fused.reshape(batch, -1)
            local_obs = self.torch.cat([uav_features, multihop], dim=-1)
            return global_state, local_obs

        def _ready_task_adjacency(self, task_adj, ready_count: int):
            if task_adj.shape[1] >= ready_count and task_adj.shape[2] >= ready_count:
                adj = task_adj[:, :ready_count, :ready_count]
            else:
                adj = self.torch.zeros((task_adj.shape[0], ready_count, ready_count), device=task_adj.device)
            eye = self.torch.eye(ready_count, device=task_adj.device).unsqueeze(0)
            return (adj + eye).clamp(max=1.0)

        def _network_nodes(self, uav_features, multihop, edge_attrs):
            batch, _, _ = uav_features.shape
            base = self.torch.zeros((batch, 1, uav_features.shape[-1] + multihop.shape[-1]), device=uav_features.device)
            uav_nodes = self.torch.cat([uav_features, multihop], dim=-1)
            nodes = self.torch.cat([base, uav_nodes], dim=1)
            if edge_attrs is None:
                return nodes
            connected_degree = edge_attrs[..., 3].sum(dim=-1, keepdim=True)
            return nodes + 0.01 * connected_degree.expand_as(nodes)

        def _pad_last_dim(self, tensor, dim: int):
            if tensor.shape[-1] == dim:
                return tensor
            if tensor.shape[-1] > dim:
                return tensor[..., :dim]
            pad = self.torch.zeros((*tensor.shape[:-1], dim - tensor.shape[-1]), device=tensor.device)
            return self.torch.cat([tensor, pad], dim=-1)

    return _CASCADEStateEncoderModule
