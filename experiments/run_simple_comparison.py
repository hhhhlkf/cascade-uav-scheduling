from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithms.cascade.ma3c_trainer import CASCADEMA3CScheduler
from src.algorithms.heuristic import GreedyScheduler, HEFTScheduler, MinLoadScheduler, RoundRobinScheduler
from src.evaluation import evaluate_scheduler_details
from src.evaluation.output import ensure_output_dir, write_episode_csv, write_json, write_summary_csv
from src.evaluation.visualizer import plot_baseline_report
from src.utils.gpu import require_cuda_device


SIMPLE_METHODS = {
    "cascade_ma3c": CASCADEMA3CScheduler,
    "greedy": GreedyScheduler,
    "min_load": MinLoadScheduler,
    "round_robin": RoundRobinScheduler,
    "heft": HEFTScheduler,
}

SCENARIOS = {
    "ds1": "configs/env/scenario_ds1_standard.yaml",
    "ds2": "configs/env/scenario_ds2_complex.yaml",
    "ds3": "configs/env/scenario_ds3_extreme.yaml",
}


def main() -> None:
    args = parse_args()
    if not args.no_require_gpu:
        device = require_cuda_device()
        print(
            "GPU ready: "
            f"{device['device_name']} "
            f"(torch={device['torch_version']}, cuda={device['cuda_version']})"
        )
    output_dir = ensure_output_dir(args.output_dir or _default_output_dir())
    selected_scenarios = {name: SCENARIOS[name] for name in args.scenarios}
    selected_methods = {name: SIMPLE_METHODS[name] for name in args.methods}
    all_summary: Dict[str, Dict[str, float]] = {}
    all_episodes: Dict[str, list[Dict[str, float]]] = {}

    for scenario_name, config_path in selected_scenarios.items():
        scenario_summary: Dict[str, Dict[str, float]] = {}
        scenario_episodes: Dict[str, list[Dict[str, float]]] = {}
        for method_name, scheduler_cls in selected_methods.items():
            key = f"{scenario_name}/{method_name}"
            result = evaluate_scheduler_details(
                config_path,
                scheduler_cls,
                episodes=args.episodes,
                seed=args.seed,
                show_progress=not args.no_progress,
                progress_desc=key,
            )
            all_summary[key] = result["summary"]
            all_episodes[key] = result["episodes"]
            scenario_summary[method_name] = result["summary"]
            scenario_episodes[method_name] = result["episodes"]
        scenario_dir = ensure_output_dir(output_dir / scenario_name)
        write_json(scenario_dir / "summary.json", scenario_summary)
        write_summary_csv(scenario_dir / "summary.csv", scenario_summary)
        write_episode_csv(scenario_dir / "episodes.csv", scenario_episodes)
        plot_baseline_report(scenario_summary, scenario_dir, scenario_episodes)

    write_json(output_dir / "summary.json", all_summary)
    write_summary_csv(output_dir / "summary.csv", all_summary)
    write_episode_csv(output_dir / "episodes.csv", all_episodes)
    print(f"Saved outputs to: {output_dir}")
    print(json.dumps(all_summary, indent=2, ensure_ascii=False, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple classic baselines against CASCADE on DS1/DS2/DS3.")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per method and scenario.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    parser.add_argument("--no-require-gpu", action="store_true", help="Allow running without CUDA. Use only for local debugging.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--scenarios", nargs="+", choices=sorted(SCENARIOS), default=["ds1", "ds2", "ds3"])
    parser.add_argument("--methods", nargs="+", choices=sorted(SIMPLE_METHODS), default=list(SIMPLE_METHODS))
    return parser.parse_args()


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "results" / f"simple_comparison_{timestamp}"


if __name__ == "__main__":
    main()
