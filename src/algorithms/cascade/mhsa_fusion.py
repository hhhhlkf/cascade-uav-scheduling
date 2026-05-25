from __future__ import annotations


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("MHSA fusion requires torch. Install requirements.txt first.") from exc
    return torch, nn


def build_mhsa_fusion(embed_dim: int = 128, num_heads: int = 4):
    """Build a Multi-Head Self-Attention fusion block for encoded state tokens."""
    _, nn = _require_torch()
    return nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

