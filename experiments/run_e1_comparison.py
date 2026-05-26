from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithms.cascade.ma3c_trainer import CASCADEMA3CScheduler
from src.algorithms.heuristic import GreedyScheduler, HEFTScheduler, MinLoadScheduler, RoundRobinScheduler
from src.experiments.runner import run_scheduler_suite


E1_METHODS = {
    "cascade_ma3c": CASCADEMA3CScheduler,
    "greedy": GreedyScheduler,
    "min_load": MinLoadScheduler,
    "round_robin": RoundRobinScheduler,
    "heft": HEFTScheduler,
}


def main() -> None:
    args = parse_args()
    summary = run_scheduler_suite(
        E1_METHODS,
        config_path=args.config,
        episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        default_output_prefix="e1_main_comparison",
        use_swanlab=args.use_swanlab,
        swanlab_project=args.swanlab_project,
        swanlab_experiment=args.swanlab_experiment,
        swanlab_mode=args.swanlab_mode,
        swanlab_workspace=args.swanlab_workspace,
        swanlab_load=args.swanlab_load,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E1 main comparison: CASCADE vs baselines.")
    parser.add_argument("--config", default="configs/env/scenario_s1_dongting.yaml", help="Environment YAML config path.")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per method.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to outputs/results/e1_main_comparison_<timestamp>.")
    parser.add_argument("--use-swanlab", action="store_true", help="Enable SwanLab experiment logging.")
    parser.add_argument("--swanlab-project", default="cascade-uav-scheduling", help="SwanLab project name.")
    parser.add_argument("--swanlab-workspace", default=None, help="SwanLab workspace name, for example: Linexus.")
    parser.add_argument("--swanlab-experiment", default="e1-main-comparison", help="SwanLab experiment name.")
    parser.add_argument("--swanlab-load", default=None, help="Optional SwanLab load config path.")
    parser.add_argument(
        "--swanlab-mode",
        default="offline",
        choices=["cloud", "local", "offline", "disabled"],
        help="SwanLab mode. Use cloud after swanlab login; offline is safe for servers.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
