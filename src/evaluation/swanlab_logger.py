from __future__ import annotations

from pathlib import Path
from typing import Mapping


class SwanLabLogger:
    """Optional SwanLab adapter."""

    def __init__(
        self,
        enabled: bool,
        project: str,
        experiment_name: str,
        mode: str,
        logdir: str | Path,
        workspace: str | None = None,
        load: str | None = None,
        config: Mapping | None = None,
    ):
        self.enabled = enabled
        self.run = None
        self._swanlab = None
        if not enabled:
            return
        try:
            import swanlab
        except ImportError as exc:
            raise RuntimeError(
                "SwanLab logging is enabled, but the 'swanlab' package is not installed. "
                "Install dependencies with 'pip install -r requirements.txt' or 'pip install swanlab'."
            ) from exc
        self._swanlab = swanlab
        init_kwargs = {
            "project": project,
            "experiment_name": experiment_name,
            "mode": mode,
            "logdir": str(logdir),
            "config": dict(config or {}),
        }
        if workspace:
            init_kwargs["workspace"] = workspace
        if load:
            init_kwargs["load"] = load
        print(
            "Initializing SwanLab: "
            f"project={project}, experiment={experiment_name}, mode={mode}, "
            f"workspace={workspace or '<default>'}, logdir={logdir}"
        )
        try:
            self.run = swanlab.init(**init_kwargs)
        except EOFError as exc:
            raise RuntimeError(
                "SwanLab cloud login requires interactive input, but this process cannot read stdin. "
                "Run 'swanlab login' in a terminal first, or use "
                "'swanlab login -k <YOUR_SWANLAB_API_KEY>' on the remote server."
            ) from exc

    def log_episode(self, method: str, metrics: Mapping[str, float]) -> None:
        if not self.enabled or self._swanlab is None:
            return
        step = int(metrics.get("episode", 0.0)) + 1
        excluded = {"episode", "seed"}
        payload = {
            f"episode/{method}/{key}": float(value)
            for key, value in metrics.items()
            if key not in excluded
        }
        self._swanlab.log(payload, step=step)

    def log_summary(self, method: str, metrics: Mapping[str, float], step: int | None = None) -> None:
        if not self.enabled or self._swanlab is None:
            return
        payload = {f"summary/{method}/{key}": float(value) for key, value in metrics.items()}
        if step is None:
            self._swanlab.log(payload)
            return
        self._swanlab.log(payload, step=step)

    def finish(self) -> None:
        if not self.enabled or self._swanlab is None:
            return
        finish = getattr(self._swanlab, "finish", None)
        if callable(finish):
            finish()
