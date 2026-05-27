from __future__ import annotations


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("CASCADE encoder requires torch. Install requirements.txt first.") from exc
    return torch, nn


def _require_pyg():
    try:
        from torch_geometric.nn import GATConv, GCNConv
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "CASCADE GNN encoder requires PyTorch Geometric. "
            "Install pyg_lib/torch_scatter/torch_sparse/torch_cluster/"
            "torch_spline_conv and torch_geometric for the active torch/CUDA build."
        ) from exc
    return GATConv, GCNConv


def build_task_gat_encoder(
    input_dim: int,
    hidden_dim: int = 128,
    output_dim: int = 64,
    heads: int = 4,
    dropout: float = 0.0,
):
    """Build the task-DAG encoder with PyG GATConv layers."""
    torch, nn = _require_torch()
    GATConv, _ = _require_pyg()
    head_dim = max(hidden_dim // heads, 1)

    class TaskGATEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.gat1 = GATConv(input_dim, head_dim, heads=heads, concat=True, dropout=dropout)
            self.gat2 = GATConv(head_dim * heads, output_dim, heads=1, concat=False, dropout=dropout)
            self.activation = nn.ELU()

        def forward(self, node_features, adjacency, node_mask=None, edge_attr=None):
            graph = _dense_to_pyg(torch, node_features, adjacency, node_mask=node_mask)
            x = self.activation(self.gat1(graph["x"], graph["edge_index"]))
            x = self.activation(self.gat2(x, graph["edge_index"]))
            return _masked_mean_pool(torch, x, graph["batch"], graph["node_mask"])

    return TaskGATEncoder()


def build_network_gcn_encoder(
    input_dim: int,
    hidden_dim: int = 128,
    output_dim: int = 64,
    num_layers: int = 2,
):
    """Build the distance-aware network-topology encoder with PyG GCNConv."""
    torch, nn = _require_torch()
    _, GCNConv = _require_pyg()

    class NetworkGCNEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.convs = nn.ModuleList(GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers))
            self.output_proj = nn.Linear(hidden_dim, output_dim)
            self.activation = nn.ReLU()

        def forward(self, node_features, adjacency, node_mask=None, edge_attr=None):
            inferred_mask = node_mask if node_mask is not None else _infer_node_mask(node_features, adjacency)
            graph = _dense_to_pyg(torch, node_features, adjacency, node_mask=inferred_mask, edge_attr=edge_attr)
            edge_weight = _edge_weight_from_attr(torch, graph["edge_attr"])
            x = self.activation(self.input_proj(graph["x"].float()))
            for conv in self.convs:
                x = self.activation(conv(x, graph["edge_index"], edge_weight=edge_weight))
            x = self.activation(self.output_proj(x))
            return _masked_mean_pool(torch, x, graph["batch"], graph["node_mask"])

    return NetworkGCNEncoder()


def build_graph_encoder(input_dim: int, hidden_dim: int = 128, output_dim: int = 64, num_layers: int = 2):
    """Backward-compatible alias for task GAT encoding."""
    return build_task_gat_encoder(input_dim, hidden_dim=hidden_dim, output_dim=output_dim)


def build_mlp_encoder(input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
    """Build a dense MLP encoder for non-graph feature blocks."""
    _, nn = _require_torch()
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU(),
    )


def _dense_to_pyg(torch, node_features, adjacency, node_mask=None, edge_attr=None):
    node_features = node_features.float()
    adjacency = adjacency.float()
    batch_size, num_nodes, _ = node_features.shape
    device = node_features.device
    if node_mask is None:
        node_mask = _infer_node_mask(node_features, adjacency)
    node_mask = node_mask.bool()

    edge_mask = adjacency > 0.0
    edge_indices = edge_mask.nonzero(as_tuple=False)
    if edge_indices.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr_flat = None
    else:
        src = edge_indices[:, 0] * num_nodes + edge_indices[:, 1]
        dst = edge_indices[:, 0] * num_nodes + edge_indices[:, 2]
        edge_index = torch.stack([src, dst], dim=0).long()
        edge_attr_flat = edge_attr[edge_mask].float() if edge_attr is not None else None

    batch = torch.arange(batch_size, device=device).repeat_interleave(num_nodes)
    return {
        "x": node_features.reshape(batch_size * num_nodes, -1),
        "edge_index": edge_index,
        "edge_attr": edge_attr_flat,
        "batch": batch,
        "node_mask": node_mask.reshape(batch_size * num_nodes),
    }


def _infer_node_mask(node_features, adjacency):
    return (node_features.abs().sum(dim=-1) > 0.0) | (adjacency.abs().sum(dim=-1) > 0.0)


def _masked_mean_pool(torch, x, batch, node_mask):
    batch_size = int(batch.max().item()) + 1 if batch.numel() else 1
    mask = node_mask.float().unsqueeze(-1)
    pooled = torch.zeros((batch_size, x.shape[-1]), dtype=x.dtype, device=x.device)
    counts = torch.zeros((batch_size, 1), dtype=x.dtype, device=x.device)
    pooled.index_add_(0, batch, x * mask)
    counts.index_add_(0, batch, mask)
    return pooled / counts.clamp_min(1.0)


def _edge_weight_from_attr(torch, edge_attr):
    if edge_attr is None:
        return None
    distance = edge_attr[:, 0].clamp_min(0.0)
    bandwidth = edge_attr[:, 1].clamp_min(0.0)
    latency = edge_attr[:, 2].clamp_min(0.0)
    connected = edge_attr[:, 3].clamp_min(0.0)
    quality = 1.0 + bandwidth + 1.0 / (1.0 + distance) + 1.0 / (1.0 + latency)
    return (connected * quality).float().clamp_min(1e-6)
