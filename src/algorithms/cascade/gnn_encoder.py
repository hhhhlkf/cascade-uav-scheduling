from __future__ import annotations


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("CASCADE encoder requires torch. Install requirements.txt first.") from exc
    return torch, nn


def build_graph_encoder(input_dim: int, hidden_dim: int = 128, output_dim: int = 64, num_layers: int = 2):
    """Build a lightweight adjacency-aware graph encoder.

    This keeps Phase 1/2 runnable without PyTorch Geometric. The module consumes
    dense node features and a dense adjacency matrix, performs message passing,
    and mean-pools a graph embedding. It can later be replaced by GAT/GCN layers
    behind the same call signature.
    """
    torch, nn = _require_torch()

    class DenseGraphEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.self_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers))
            self.neigh_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers))
            self.output_proj = nn.Linear(hidden_dim, output_dim)
            self.activation = nn.ReLU()

        def forward(self, node_features, adjacency, node_mask=None):
            x = self.activation(self.input_proj(node_features.float()))
            adj = adjacency.float()
            degree = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
            norm_adj = adj / degree
            for self_layer, neigh_layer in zip(self.self_layers, self.neigh_layers):
                neigh = torch.bmm(norm_adj, x)
                x = self.activation(self_layer(x) + neigh_layer(neigh))
            if node_mask is None:
                pooled = x.mean(dim=1)
            else:
                mask = node_mask.float().unsqueeze(-1)
                pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            return self.activation(self.output_proj(pooled))

    return DenseGraphEncoder()


def build_mlp_encoder(input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
    """Temporary MLP encoder used until PyTorch Geometric GAT/GCN layers are added."""
    _, nn = _require_torch()
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU(),
    )
