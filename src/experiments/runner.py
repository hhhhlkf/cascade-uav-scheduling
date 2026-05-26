from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Type

from src.algorithms.base_scheduler import BaseScheduler
from src.evaluation import evaluate_scheduler_details
from src.evaluation.output import ensure_output_dir, write_episode_csv, write_json, write_summary_csv
from src.evaluation.swanlab_logger import SwanLabLogger
from src.evaluation.visualizer import plot_baseline_report


def run_scheduler_suite(
    methods: Mapping[str, Type[BaseScheduler]],
    config_path: str,
    episodes: int,
    seed: int,
    output_dir: str | Path | None,
    default_output_prefix: str,
    use_swanlab: bool = False,
    swanlab_project: str = "cascade-uav-scheduling",
    swanlab_experiment: str = "experiment",
    swanlab_mode: str = "offline",
    swanlab_workspace: str | None = None,
    swanlab_load: str | None = None,
) -> Dict[str, Dict[str, float]]:
    output_path = ensure_output_dir(output_dir or _default_output_dir(default_output_prefix))
    summary = {}
    episodes_by_method = {}
    swanlab = SwanLabLogger(
        enabled=use_swanlab,
        project=swanlab_project,
        experiment_name=swanlab_experiment,
        mode=swanlab_mode,
        logdir=output_path / "swanlab",
        workspace=swanlab_workspace,
        load=swanlab_load,
        config={
            "scenario": config_path,
            "episodes": episodes,
            "seed": seed,
            "methods": list(methods),
            "swanlab_mode": swanlab_mode,
            "swanlab_workspace": swanlab_workspace,
        },
    )
    for step, (name, scheduler_cls) in enumerate(methods.items()):
        result = evaluate_scheduler_details(config_path, scheduler_cls, episodes=episodes, seed=seed)
        summary[name] = result["summary"]
        episodes_by_method[name] = result["episodes"]
        swanlab.log_summary(name, result["summary"], step=step)
    figure_paths = plot_baseline_report(summary, output_path)
    write_json(output_path / "summary.json", summary)
    write_summary_csv(output_path / "summary.csv", summary)
    write_episode_csv(output_path / "episodes.csv", episodes_by_method)
    swanlab.finish()
    print(f"Saved outputs to: {output_path}")
    print("Figures:")
    for path in figure_paths:
        print(f"  - {path}")
    return summary


def _default_output_dir(prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "results" / f"{prefix}_{timestamp}"
