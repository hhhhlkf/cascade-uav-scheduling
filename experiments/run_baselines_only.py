from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithms.heuristic import GreedyScheduler, HEFTScheduler, MinLoadScheduler, RoundRobinScheduler
from src.algorithms.cascade.ma3c_trainer import CASCADEMA3CScheduler
from src.evaluation import evaluate_scheduler


BASELINES = {
    "greedy": GreedyScheduler,
    "min_load": MinLoadScheduler,
    "round_robin": RoundRobinScheduler,
    "heft": HEFTScheduler,
    "cascade_placeholder": CASCADEMA3CScheduler,
}


def main() -> None:
    config_path = "configs/env/scenario_s1_dongting.yaml"
    summary = {
        name: evaluate_scheduler(config_path, scheduler_cls, episodes=2, seed=0)
        for name, scheduler_cls in BASELINES.items()
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
