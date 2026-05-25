from __future__ import annotations


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("CASCADE actor network requires torch. Install requirements.txt first.") from exc
    return torch, nn


def build_actor(input_dim: int, hidden_dim: int, max_ready_tasks: int):
    """Build the UAV Actor MLP that outputs task-acceptance logits."""
    _, nn = _require_torch()
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, max_ready_tasks),
    )

