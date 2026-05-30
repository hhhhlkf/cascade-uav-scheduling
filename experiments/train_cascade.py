from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithms.cascade.config_utils import max_ready_tasks_from_config, max_uavs_from_config
from src.algorithms.cascade.ma3c_trainer import CASCADEMA3CScheduler, MA3CConfig
from src.algorithms.heuristic import GreedyScheduler, HEFTScheduler, MinLoadScheduler
from src.env import CASCADEEnv
from src.evaluation import evaluate_scheduler_details
from src.evaluation.output import ensure_output_dir, write_json
from src.evaluation.swanlab_logger import SwanLabLogger
from src.utils.config import load_yaml_config
from src.utils.gpu import require_cuda_device


def main() -> None:
    args = parse_args()
    show_progress = not args.no_progress
    device_info = {}
    if not args.no_require_gpu:
        device_info = require_cuda_device()
        print(
            "GPU ready: "
            f"{device_info['device_name']} "
            f"(torch={device_info['torch_version']}, cuda={device_info['cuda_version']})"
        )
    output_dir = ensure_output_dir(args.output_dir or _default_output_dir())
    scheduler = _build_scheduler(args)
    swanlab = SwanLabLogger(
        enabled=args.use_swanlab,
        project=args.swanlab_project,
        experiment_name=args.swanlab_experiment,
        mode=args.swanlab_mode,
        logdir=output_dir / "swanlab",
        workspace=args.swanlab_workspace,
        load=args.swanlab_load,
        config={
            "train_config": args.config,
            "eval_config": args.eval_config,
            "train_episodes": args.train_episodes,
            "eval_episodes": args.eval_episodes,
            "eval_every": args.eval_every,
            "max_steps": args.max_steps,
            "seed_pool_size": args.seed_pool_size,
            "rolling_window": args.rolling_window,
            "bc_episodes": args.bc_episodes,
            "bc_teacher": args.bc_teacher,
            "bc_epochs": args.bc_epochs,
            "bc_max_transitions": args.bc_max_transitions,
            "seed": args.seed,
            "checkpoint": args.checkpoint,
            "model_num_uavs": args.model_num_uavs,
            "actor_lr": args.actor_lr,
            "critic_lr": args.critic_lr,
            "entropy_coef": args.entropy_coef,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "token_dim": args.token_dim,
            "hidden_dim": args.hidden_dim,
            "device": device_info,
        },
    )
    train_rows = []
    eval_history = []
    rolling_rows = deque(maxlen=max(1, args.rolling_window))
    print(
        "Starting CASCADE training: "
        f"train_episodes={args.train_episodes}, eval_every={args.eval_every}, "
        f"eval_episodes={args.eval_episodes}, max_steps={args.max_steps}, seed={args.seed}, "
        f"seed_pool_size={args.seed_pool_size}, rolling_window={args.rolling_window}"
    )
    print(f"Training config: {args.config}")
    print(f"Evaluation config: {args.eval_config}")
    print(f"Output directory: {output_dir}")
    if args.checkpoint:
        print(f"Resuming from checkpoint: {args.checkpoint}")
    if args.bc_episodes > 0:
        print(
            "Starting behavior cloning warm-up: "
            f"teacher={args.bc_teacher}, episodes={args.bc_episodes}, "
            f"epochs={args.bc_epochs}, max_transitions={args.bc_max_transitions}"
        )
        bc_rows = run_behavior_cloning(
            scheduler,
            config_path=args.config,
            teacher_name=args.bc_teacher,
            episodes=args.bc_episodes,
            epochs=args.bc_epochs,
            max_transitions=args.bc_max_transitions,
            seed=args.seed + args.bc_seed_offset,
            max_steps=args.max_steps,
            seed_pool_size=args.seed_pool_size,
            show_progress=show_progress,
            swanlab=swanlab,
        )
        _write_rows(output_dir / "bc_metrics.csv", bc_rows)
        print(f"Saved behavior cloning metrics to: {output_dir / 'bc_metrics.csv'}")

    episode_iter = range(args.train_episodes)
    progress_bar = None
    if show_progress:
        from tqdm.auto import tqdm

        progress_bar = tqdm(episode_iter, desc="Training episodes", unit="ep", dynamic_ncols=True)
        episode_iter = progress_bar

    for episode in episode_iter:
        train_seed = args.seed + (episode % args.seed_pool_size if args.seed_pool_size > 0 else episode)
        row = train_one_episode(
            scheduler,
            config_path=args.config,
            seed=train_seed,
            max_steps=args.max_steps,
        )
        row["episode"] = episode
        row["seed"] = train_seed
        train_rows.append(row)
        rolling_rows.append(row)
        rolling_metrics = _rolling_means(rolling_rows)
        swanlab.log_metrics("train", row, step=episode + 1)
        swanlab.log_metrics("train_ma", rolling_metrics, step=episode + 1)
        if progress_bar is not None:
            progress_bar.set_postfix(
                {
                    "reward": f"{row.get('total_reward', 0.0):.3f}",
                    "reward_ma": f"{rolling_metrics.get('total_reward', 0.0):.3f}",
                    "actor": f"{row.get('loss_actor', 0.0):.3f}",
                    "critic": f"{row.get('loss_critic', 0.0):.3f}",
                    "entropy": f"{row.get('entropy', 0.0):.3f}",
                    "done": int(row.get("completed_tasks", 0.0)),
                    "timeout": int(row.get("timed_out_tasks", 0.0)),
                }
            )
        if (episode + 1) % args.eval_every == 0 or episode == args.train_episodes - 1:
            _progress_write(
                progress_bar,
                f"Running evaluation at episode {episode + 1}/{args.train_episodes} "
                f"({args.eval_episodes} episodes)...",
            )
            eval_result = evaluate_scheduler_details(
                args.eval_config,
                lambda max_ready, num_uavs: scheduler,
                episodes=args.eval_episodes,
                seed=args.seed + 10_000 + episode,
                show_progress=show_progress,
                progress_desc=f"Eval after ep {episode + 1}",
            )
            eval_row = {"episode": episode, **eval_result["summary"]}
            eval_history.append(eval_row)
            swanlab.log_metrics("eval", eval_row, step=episode + 1)
            _progress_write(
                progress_bar,
                json.dumps({"train": row, "eval": eval_row}, ensure_ascii=False, sort_keys=True),
            )

    scheduler.save(output_dir / "cascade_ma3c.pt")
    _write_rows(output_dir / "train_metrics.csv", train_rows)
    _write_rows(output_dir / "eval_metrics.csv", eval_history)
    write_json(output_dir / "train_metrics.json", {"train": train_rows, "eval": eval_history})
    print(f"Saved checkpoint to: {output_dir / 'cascade_ma3c.pt'}")
    print(f"Saved training metrics to: {output_dir / 'train_metrics.csv'}")
    print(f"Saved evaluation metrics to: {output_dir / 'eval_metrics.csv'}")
    print(f"Saved training outputs to: {output_dir}")
    swanlab.finish()


def train_one_episode(scheduler: CASCADEMA3CScheduler, config_path: str, seed: int, max_steps: int) -> dict[str, float]:
    env = CASCADEEnv(config_path)
    obs, info = env.reset(seed=seed)
    traces = []
    rewards = []
    dones = []
    total_reward = 0.0
    terminated = False
    truncated = False
    steps = 0
    reward_parts_sum: dict[str, float] = {}
    scheduler.reset()
    while not (terminated or truncated) and steps < max_steps:
        action_mask = obs.get("action_mask", env.get_action_mask())
        action, trace = scheduler.decide_with_trace(obs, action_mask)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        traces.append(trace)
        rewards.append(float(reward))
        dones.append(done)
        total_reward += float(reward)
        for key, value in info.items():
            if key.startswith("cost_") or key.startswith("reward_"):
                reward_parts_sum[f"{key}_sum"] = reward_parts_sum.get(f"{key}_sum", 0.0) + float(value)
        steps += 1
    metrics = scheduler.learn_episode(traces, rewards, dones)
    metrics.update(
        {
            "total_reward": total_reward,
            "steps": float(steps),
            "completed_tasks": float(info.get("completed_tasks", 0.0)),
            "timed_out_tasks": float(info.get("timed_out_tasks", 0.0)),
            "makespan_s": float(info.get("makespan_s", 0.0)),
            "atct_s": float(info.get("atct_s", 0.0)),
        }
    )
    metrics.update(reward_parts_sum)
    return metrics


def run_behavior_cloning(
    scheduler: CASCADEMA3CScheduler,
    config_path: str,
    teacher_name: str,
    episodes: int,
    epochs: int,
    max_transitions: int,
    seed: int,
    max_steps: int,
    seed_pool_size: int,
    show_progress: bool,
    swanlab: SwanLabLogger,
) -> list[dict[str, float]]:
    teacher_cls = _teacher_class(teacher_name)
    dataset = []
    teacher_rows = []
    episode_iter = range(episodes)
    if show_progress:
        from tqdm.auto import tqdm

        episode_iter = tqdm(episode_iter, desc="Collect BC data", unit="ep", dynamic_ncols=True)
    for episode in episode_iter:
        train_seed = seed + (episode % seed_pool_size if seed_pool_size > 0 else episode)
        env = CASCADEEnv(config_path)
        obs, _ = env.reset(seed=train_seed)
        teacher = teacher_cls(env.max_ready_tasks, len(env.uavs))
        total_reward = 0.0
        terminated = False
        truncated = False
        steps = 0
        targets = 0.0
        while not (terminated or truncated) and steps < max_steps and len(dataset) < max_transitions:
            action_mask = obs.get("action_mask", env.get_action_mask())
            teacher.observe(obs)
            target_action = teacher.decide(action_mask)
            if target_action.any():
                dataset.append(
                    (
                        {key: value.copy() for key, value in obs.items()},
                        action_mask.copy(),
                        target_action.copy(),
                    )
                )
                targets += float((target_action > 0.0).sum())
            obs, reward, terminated, truncated, info = env.step(target_action)
            total_reward += float(reward)
            steps += 1
        teacher_rows.append(
            {
                "episode": float(episode),
                "seed": float(train_seed),
                "teacher_total_reward": total_reward,
                "teacher_completed_tasks": float(info.get("completed_tasks", 0.0)),
                "teacher_timed_out_tasks": float(info.get("timed_out_tasks", 0.0)),
                "teacher_targets": targets,
                "steps": float(steps),
                "dataset_size": float(len(dataset)),
            }
        )
        if len(dataset) >= max_transitions:
            break
    rows = []
    if not dataset:
        return rows
    rng = np.random.default_rng(seed)
    epoch_iter = range(max(1, epochs))
    if show_progress:
        from tqdm.auto import tqdm

        epoch_iter = tqdm(epoch_iter, desc="Train BC actor", unit="epoch", dynamic_ncols=True)
    for epoch in epoch_iter:
        losses = []
        targets = []
        entropies = []
        indices = rng.permutation(len(dataset))
        for idx in indices:
            obs, action_mask, target_action = dataset[int(idx)]
            metrics = scheduler.behavior_clone_step(obs, action_mask, target_action)
            if metrics["bc_targets"] <= 0.0:
                continue
            losses.append(metrics["loss_bc"])
            targets.append(metrics["bc_targets"])
            entropies.append(metrics["bc_entropy"])
        row = {
            "epoch": float(epoch),
            "loss_bc": float(sum(losses) / len(losses)) if losses else 0.0,
            "bc_targets": float(sum(targets)),
            "bc_entropy": float(sum(entropies) / len(entropies)) if entropies else 0.0,
            "dataset_size": float(len(dataset)),
        }
        rows.append(row)
        swanlab.log_metrics("bc", row, step=epoch + 1, excluded={"episode", "seed", "epoch"})
    return teacher_rows + rows


def _build_scheduler(args: argparse.Namespace) -> CASCADEMA3CScheduler:
    env_config = load_yaml_config(args.config)
    max_ready_tasks = max_ready_tasks_from_config(env_config)
    num_uavs = max(int(args.model_num_uavs), max_uavs_from_config(env_config))
    config = MA3CConfig(
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        entropy_coef=args.entropy_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        token_dim=args.token_dim,
        hidden_dim=args.hidden_dim,
    )
    scheduler = CASCADEMA3CScheduler(max_ready_tasks, num_uavs, config=config)
    if args.checkpoint:
        scheduler.load(args.checkpoint)
    return scheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CASCADE MA3C policy end-to-end on DS1.")
    parser.add_argument("--config", default="configs/env/scenario_ds1_standard.yaml", help="Training scenario distribution.")
    parser.add_argument("--eval-config", default="configs/env/scenario_ds1_standard.yaml", help="Validation scenario distribution.")
    parser.add_argument("--train-episodes", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seed-pool-size",
        type=int,
        default=0,
        help="Cycle through a fixed pool of training seeds. 0 keeps using a fresh seed every episode.",
    )
    parser.add_argument("--rolling-window", type=int, default=50, help="Window size for train_ma SwanLab metrics.")
    parser.add_argument("--bc-episodes", type=int, default=0, help="Behavior cloning warm-up episodes before RL training.")
    parser.add_argument("--bc-epochs", type=int, default=5, help="Training epochs over the collected BC dataset.")
    parser.add_argument("--bc-max-transitions", type=int, default=5000, help="Maximum expert transitions collected for BC.")
    parser.add_argument(
        "--bc-teacher",
        choices=["greedy", "min_load", "heft"],
        default="heft",
        help="Heuristic teacher used for behavior cloning warm-up.",
    )
    parser.add_argument("--bc-seed-offset", type=int, default=100_000, help="Seed offset for behavior cloning scenarios.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=None, help="Optional CASCADE checkpoint to resume from.")
    parser.add_argument("--model-num-uavs", type=int, default=15, help="Fixed CASCADE model UAV capacity for cross-scenario checkpoints.")
    parser.add_argument("--no-require-gpu", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--use-swanlab", action="store_true", help="Enable SwanLab training logging.")
    parser.add_argument("--swanlab-project", default="cascade-uav-scheduling", help="SwanLab project name.")
    parser.add_argument("--swanlab-workspace", default="Linexus", help="SwanLab workspace name.")
    parser.add_argument("--swanlab-experiment", default="cascade-ma3c-train", help="SwanLab experiment name.")
    parser.add_argument("--swanlab-load", default=None, help="Optional SwanLab load config path.")
    parser.add_argument(
        "--swanlab-mode",
        default="cloud",
        choices=["cloud", "local", "offline", "disabled"],
        help="SwanLab mode. Use cloud after swanlab login; offline is safe for servers.",
    )
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    return parser.parse_args()


def _progress_write(progress_bar, message: str) -> None:
    if progress_bar is not None:
        progress_bar.write(message)
        return
    print(message)


def _rolling_means(rows) -> dict[str, float]:
    excluded = {"episode", "seed"}
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in excluded and isinstance(value, (int, float))
        }
    )
    means: dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            means[key] = sum(values) / len(values)
    return means


def _teacher_class(name: str):
    if name == "greedy":
        return GreedyScheduler
    if name == "min_load":
        return MinLoadScheduler
    if name == "heft":
        return HEFTScheduler
    raise ValueError(f"Unsupported BC teacher: {name}")


def _write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "training" / f"cascade_ma3c_{timestamp}"


if __name__ == "__main__":
    main()
