from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithms.cascade.ma3c_trainer import CASCADEMA3CScheduler
from src.experiments.runner import run_scheduler_suite


CASCADE_METHODS = {
    "cascade_ma3c": CASCADEMA3CScheduler,
}


def main() -> None:
    args = parse_args()
    summary = run_scheduler_suite(
        CASCADE_METHODS,
        config_path=args.config,
        episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        default_output_prefix="cascade",
        use_swanlab=args.use_swanlab,
        swanlab_project=args.swanlab_project,
        swanlab_experiment=args.swanlab_experiment,
        swanlab_mode=args.swanlab_mode,
        swanlab_workspace=args.swanlab_workspace,
        swanlab_load=args.swanlab_load,
        require_gpu=not args.no_require_gpu,
        show_progress=not args.no_progress,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CASCADE method.")
    parser.add_argument("--config", default="configs/env/scenario_ds1_standard.yaml", help="Environment YAML config path.")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes to evaluate CASCADE.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to outputs/results/cascade_<timestamp>.")
    parser.add_argument("--use-swanlab", action="store_true", help="Enable SwanLab experiment logging.")
    parser.add_argument("--swanlab-project", default="cascade-uav-scheduling", help="SwanLab project name.")
    parser.add_argument("--swanlab-workspace", default=None, help="SwanLab workspace name, for example: Linexus.")
    parser.add_argument("--swanlab-experiment", default="cascade-ma3c", help="SwanLab experiment name.")
    parser.add_argument("--swanlab-load", default=None, help="Optional SwanLab load config path.")
    parser.add_argument("--no-require-gpu", action="store_true", help="Allow running without CUDA. Use only for local debugging.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument(
        "--swanlab-mode",
        default="cloud",
        choices=["cloud", "local", "offline", "disabled"],
        help="SwanLab mode. Use cloud after swanlab login; offline is safe for servers.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
