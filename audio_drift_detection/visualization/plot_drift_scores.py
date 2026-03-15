"""Plot drift scores, mean loss, and variance over time."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from drift_detection.drift_monitor import DriftMonitor

sns.set_theme(style="whitegrid")


def plot_drift_scores(
    monitor: DriftMonitor,
    true_drift_points: list[int] | None = None,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Generate a three-panel figure: drift score, mean loss, and variance vs time.

    Args:
        monitor: ``DriftMonitor`` with recorded data.
        true_drift_points: Ground-truth drift timestamps.
        save_path: Path to save the figure (PNG).
        show: Whether to call ``plt.show()``.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    timesteps = range(len(monitor.scores))

    # --- Drift score ---
    axes[0].plot(timesteps, monitor.scores, linewidth=0.8, color="#2196F3", label="Drift Score")
    axes[0].set_ylabel("Drift Score")
    axes[0].set_title("Drift Score over Time")

    # --- Mean loss ---
    axes[1].plot(timesteps, monitor.mean_losses, linewidth=0.8, color="#4CAF50", label="Mean Loss")
    axes[1].set_ylabel("Mean Recon. Loss")
    axes[1].set_title("Mean Reconstruction Loss (STUDD Signal)")

    # --- Loss variance ---
    axes[2].plot(timesteps, monitor.variances, linewidth=0.8, color="#FF9800", label="Loss Variance")
    axes[2].set_ylabel("Loss Variance")
    axes[2].set_xlabel("Time Step")
    axes[2].set_title("Ensemble Loss Variance (Disagreement Signal)")

    # Mark true drift points
    if true_drift_points:
        for ax in axes:
            for tp in true_drift_points:
                ax.axvline(tp, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="True Drift")

    # Mark detected drifts
    for ax in axes:
        for d in monitor.detected_timestamps:
            ax.axvline(d, color="purple", linestyle=":", linewidth=1.0, alpha=0.6, label="Detected Drift")

    # De-duplicate legend entries
    for ax in axes:
        handles, lbls = ax.get_legend_handles_labels()
        unique = dict(zip(lbls, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
