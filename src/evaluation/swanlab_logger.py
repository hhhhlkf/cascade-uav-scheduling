from __future__ import annotations

from pathlib import Path
from typing import Mapping


class SwanLabLogger:
    """Optional SwanLab adapter; silently disabled when SwanLab is unavailable."""

    def __init__(
        self,
        enabled: bool,
        project: str,
        experiment_name: str,
        mode: str,
        logdir: str | Path,
        config: Mapping | None = None,
    ):
        self.enabled = enabled
        self.run = None
        self._swanlab = None
        if not enabled:
            return
        try:
            import swanlab
        except ImportError:
            self.enabled = False
            return
        self._swanlab = swanlab
        self.run = swanlab.init(
            project=project,
            experiment_name=experiment_name,
            mode=mode,
            logdir=str(logdir),
            config=dict(config or {}),
        )

    def log_summary(self, method: str, metrics: Mapping[str, float], step: int) -> None:
        if not self.enabled or self._swanlab is None:
            return
        payload = {f"{method}/{key}": float(value) for key, value in metrics.items()}
        self._swanlab.log(payload, step=step)

    def finish(self) -> None:
        if not self.enabled or self._swanlab is None:
            return
        finish = getattr(self._swanlab, "finish", None)
        if callable(finish):
            finish()

