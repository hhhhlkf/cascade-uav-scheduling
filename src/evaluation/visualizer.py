from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping


METRIC_LABELS = {
    "total_reward_mean": "Total Reward",
    "completion_ratio_mean": "Completion Ratio",
    "tdsr_mean": "TDSR",
    "rpdr_proxy_mean": "RPDR Proxy",
    "completed_tasks_mean": "Completed Tasks",
    "timed_out_tasks_mean": "Timed-out Tasks",
}

PALETTE = ["#2563eb", "#16a34a", "#f97316", "#dc2626", "#7c3aed", "#0891b2", "#db2777"]


EPISODE_METRICS = [
    "total_reward",
    "completion_ratio",
    "tdsr",
    "rpdr_proxy",
    "completed_tasks",
    "timed_out_tasks",
]


def plot_baseline_report(
    summary: Mapping[str, Mapping[str, float]],
    output_dir: str | Path,
    episodes_by_method: Mapping[str, Iterable[Mapping[str, float]]] | None = None,
) -> list[Path]:
    import matplotlib.pyplot as plt

    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 11,
            "axes.titleweight": "bold",
        }
    )
    paths = [
        _plot_metric_bars(summary, "completion_ratio_mean", figures_dir / "completion_ratio.png"),
        _plot_metric_bars(summary, "tdsr_mean", figures_dir / "tdsr.png"),
        _plot_metric_bars(summary, "total_reward_mean", figures_dir / "total_reward.png"),
        _plot_metric_bars(summary, "rpdr_proxy_mean", figures_dir / "rpdr_proxy.png"),
        _plot_radar(summary, figures_dir / "baseline_radar.png"),
    ]
    if episodes_by_method:
        paths.extend(
            [
                _plot_episode_grid(episodes_by_method, figures_dir / "episode_metric_trends.png", cumulative=False),
                _plot_episode_grid(episodes_by_method, figures_dir / "episode_cumulative_mean_trends.png", cumulative=True),
            ]
        )
    return paths


def _plot_metric_bars(summary: Mapping[str, Mapping[str, float]], metric: str, path: Path) -> Path:
    import matplotlib.pyplot as plt

    methods = list(summary)
    values = [float(summary[method].get(metric, 0.0)) for method in methods]
    errors = [float(summary[method].get(metric.replace("_mean", "_std"), 0.0)) for method in methods]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    bars = ax.bar(methods, values, yerr=errors, capsize=4, color=PALETTE[: len(methods)], edgecolor="#111827", linewidth=0.7)
    ax.set_title(METRIC_LABELS.get(metric, metric))
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_radar(summary: Mapping[str, Mapping[str, float]], path: Path) -> Path:
    import math
    import matplotlib.pyplot as plt

    metrics = ["completion_ratio_mean", "tdsr_mean", "rpdr_proxy_mean", "total_reward_mean"]
    methods = list(summary)
    raw = {metric: [float(summary[method].get(metric, 0.0)) for method in methods] for metric in metrics}
    normalized = {}
    for metric, values in raw.items():
        min_v, max_v = min(values), max(values)
        if math.isclose(max_v, min_v):
            normalized[metric] = [0.65 for _ in values]
        else:
            normalized[metric] = [(value - min_v) / (max_v - min_v) for value in values]
    angles = [idx / float(len(metrics)) * 2 * math.pi for idx in range(len(metrics))]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw={"polar": True})
    for idx, method in enumerate(methods):
        values = [normalized[metric][idx] for metric in metrics]
        values += values[:1]
        ax.plot(angles, values, color=PALETTE[idx % len(PALETTE)], linewidth=2.0, label=method)
        ax.fill(angles, values, color=PALETTE[idx % len(PALETTE)], alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS[metric] for metric in metrics])
    ax.set_yticklabels([])
    ax.set_title("Baseline Multi-Metric Profile", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_episode_grid(
    episodes_by_method: Mapping[str, Iterable[Mapping[str, float]]],
    path: Path,
    cumulative: bool,
) -> Path:
    import matplotlib.pyplot as plt

    title_prefix = "Cumulative Mean" if cumulative else "Episode"
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=False)
    axes_flat = axes.ravel()
    for metric_idx, metric in enumerate(EPISODE_METRICS):
        ax = axes_flat[metric_idx]
        for method_idx, (method, episodes) in enumerate(episodes_by_method.items()):
            episode_list = sorted(list(episodes), key=lambda item: float(item.get("episode", 0.0)))
            if not episode_list:
                continue
            x_values = [int(item.get("episode", idx)) + 1 for idx, item in enumerate(episode_list)]
            y_values = [float(item.get(metric, 0.0)) for item in episode_list]
            if cumulative:
                y_values = _cumulative_mean(y_values)
            ax.plot(
                x_values,
                y_values,
                marker="o",
                markersize=3.2,
                linewidth=1.8,
                color=PALETTE[method_idx % len(PALETTE)],
                label=method,
            )
        ax.set_title(METRIC_LABELS.get(f"{metric}_mean", metric.replace("_", " ").title()))
        ax.set_xlabel("Episode")
        ax.set_ylabel(METRIC_LABELS.get(f"{metric}_mean", metric.replace("_", " ").title()))
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), frameon=False)
    fig.suptitle(f"{title_prefix} Metric Trends", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _cumulative_mean(values: list[float]) -> list[float]:
    running = []
    total = 0.0
    for idx, value in enumerate(values, start=1):
        total += value
        running.append(total / idx)
    return running
