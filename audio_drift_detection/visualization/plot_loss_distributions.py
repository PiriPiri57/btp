"""Plot loss distributions across ensemble students."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from drift_detection.drift_monitor import DriftMonitor

sns.set_theme(style="whitegrid")


def plot_loss_distributions(
    monitor: DriftMonitor,
    true_drift_points: list[int] | None = None,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Plot loss distribution statistics and detected drift events.

    Creates two sub-plots:
      1. Rolling mean & ±1 std band of the drift score.
      2. Histogram of scores before vs after true drift.

    Args:
        monitor: ``DriftMonitor`` with recorded data.
        true_drift_points: Ground-truth drift timestamps.
        save_path: Save path for the figure.
        show: Whether to display interactively.
    """
    scores = np.array(monitor.scores)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Rolling statistics ---
    window = min(100, len(scores) // 5) if len(scores) > 5 else 1
    rolling_mean = np.convolve(scores, np.ones(window) / window, mode="valid")
    rolling_std = np.array(
        [scores[max(0, i - window):i].std() for i in range(window, len(scores) + 1)]
    )[:len(rolling_mean)]
    t = np.arange(len(rolling_mean))

    axes[0].plot(t, rolling_mean, color="#2196F3", linewidth=1.0, label="Rolling Mean")
    axes[0].fill_between(
        t,
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        alpha=0.25,
        color="#2196F3",
        label="±1 Std",
    )
    if true_drift_points:
        for tp in true_drift_points:
            axes[0].axvline(tp, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    axes[0].set_title("Rolling Drift Score Statistics")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Score")
    axes[0].legend(fontsize=8)

    # --- Before / after histogram ---
    if true_drift_points and len(true_drift_points) > 0:
        tp0 = true_drift_points[0]
        before = scores[:tp0]
        after = scores[tp0:]
        if len(before) > 0:
            axes[1].hist(before, bins=40, alpha=0.6, label="Before Drift", color="#4CAF50", density=True)
        if len(after) > 0:
            axes[1].hist(after, bins=40, alpha=0.6, label="After Drift", color="#F44336", density=True)
        axes[1].legend(fontsize=9)
    else:
        axes[1].hist(scores, bins=40, alpha=0.7, color="#607D8B", density=True)
    axes[1].set_title("Drift Score Distribution")
    axes[1].set_xlabel("Score")
    axes[1].set_ylabel("Density")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
