from __future__ import annotations


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("CASCADE critic network requires torch. Install requirements.txt first.") from exc
    return torch, nn


def build_critic(input_dim: int, hidden_dim: int = 256):
    """Build the command-center Critic MLP V(s)."""
    _, nn = _require_torch()
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, 1),
    )

