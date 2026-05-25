from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithms.cascade.ma3c_trainer import CASCADEMA3CScheduler
from src.algorithms.heuristic import GreedyScheduler, HEFTScheduler, MinLoadScheduler, RoundRobinScheduler
from src.evaluation import evaluate_scheduler_details
from src.evaluation.output import ensure_output_dir, write_episode_csv, write_json, write_summary_csv
from src.evaluation.swanlab_logger import SwanLabLogger
from src.evaluation.visualizer import plot_baseline_report


BASELINES = {
    "greedy": GreedyScheduler,
    "min_load": MinLoadScheduler,
    "round_robin": RoundRobinScheduler,
    "heft": HEFTScheduler,
    "cascade_placeholder": CASCADEMA3CScheduler,
}


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir or _default_output_dir())
    summary = {}
    episodes = {}
    swanlab = SwanLabLogger(
        enabled=args.use_swanlab,
        project=args.swanlab_project,
        experiment_name=args.swanlab_experiment,
        mode=args.swanlab_mode,
        logdir=output_dir / "swanlab",
        config={
            "scenario": args.config,
            "episodes": args.episodes,
            "seed": args.seed,
            "methods": list(BASELINES),
        },
    )
    for step, (name, scheduler_cls) in enumerate(BASELINES.items()):
        result = evaluate_scheduler_details(args.config, scheduler_cls, episodes=args.episodes, seed=args.seed)
        summary[name] = result["summary"]
        episodes[name] = result["episodes"]
        swanlab.log_summary(name, result["summary"], step=step)
    figure_paths = plot_baseline_report(summary, output_dir)
    write_json(output_dir / "summary.json", summary)
    write_summary_csv(output_dir / "summary.csv", summary)
    write_episode_csv(output_dir / "episodes.csv", episodes)
    swanlab.finish()
    print(f"Saved outputs to: {output_dir}")
    print("Figures:")
    for path in figure_paths:
        print(f"  - {path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run heuristic and placeholder CASCADE baselines.")
    parser.add_argument("--config", default="configs/env/scenario_s1_dongting.yaml", help="Environment YAML config path.")
    parser.add_argument("--episodes", type=int, default=2, help="Episodes per method.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to outputs/results/baselines_<timestamp>.")
    parser.add_argument("--use-swanlab", action="store_true", help="Enable SwanLab experiment logging.")
    parser.add_argument("--swanlab-project", default="cascade-uav-scheduling", help="SwanLab project name.")
    parser.add_argument("--swanlab-experiment", default="baseline-comparison", help="SwanLab experiment name.")
    parser.add_argument(
        "--swanlab-mode",
        default="offline",
        choices=["cloud", "local", "offline", "disabled"],
        help="SwanLab mode. Use cloud after swanlab login; offline is safe for servers.",
    )
    return parser.parse_args()


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "results" / f"baselines_{timestamp}"


if __name__ == "__main__":
    main()
