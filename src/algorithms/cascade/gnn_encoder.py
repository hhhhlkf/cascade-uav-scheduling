from __future__ import annotations


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("CASCADE encoder requires torch. Install requirements.txt first.") from exc
    return torch, nn


def build_mlp_encoder(input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
    """Temporary MLP encoder used until PyTorch Geometric GAT/GCN layers are added."""
    _, nn = _require_torch()
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU(),
    )

