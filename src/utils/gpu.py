from __future__ import annotations

from typing import Dict


def require_cuda_device() -> Dict[str, str]:
    """Validate that PyTorch can run a small CUDA operation."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "This experiment requires GPU execution, but torch is not installed. "
            "Install the CUDA build of torch before running experiments."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This experiment requires GPU execution, but torch.cuda.is_available() is False. "
            "Run on the remote GPU server or install a CUDA-enabled torch build."
        )

    device = torch.device("cuda:0")
    probe = torch.ones((1,), device=device)
    _ = probe * 2.0
    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": str(torch.version.cuda),
    }
