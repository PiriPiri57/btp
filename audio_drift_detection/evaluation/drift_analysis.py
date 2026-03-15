"""Drift analysis — compare detected vs true drift points and produce a summary."""

from __future__ import annotations

import logging
from typing import Any

from evaluation.metrics import detection_delay, false_positive_rate, precision_recall_f1

logger = logging.getLogger("audio_drift")


def analyse_drift(
    true_drift_points: list[int],
    detected_timestamps: list[int],
    total_steps: int,
    tolerance: int = 50,
) -> dict[str, Any]:
    """Run all metrics and return a summary dict.

    Args:
        true_drift_points: Ground-truth drift timestamps.
        detected_timestamps: Detected drift timestamps.
        total_steps: Total number of time steps.
        tolerance: TP matching window.

    Returns:
        Dictionary with all metric values.
    """
    delay = detection_delay(true_drift_points, detected_timestamps)
    fpr = false_positive_rate(true_drift_points, detected_timestamps, total_steps, tolerance)
    prf = precision_recall_f1(true_drift_points, detected_timestamps, tolerance)

    summary = {
        "true_drift_points": true_drift_points,
        "detected_timestamps": detected_timestamps,
        "num_detections": len(detected_timestamps),
        "detection_delay": delay,
        "false_positive_rate": fpr,
        **prf,
    }

    logger.info("=== Drift Analysis ===")
    for k, v in summary.items():
        logger.info("  %-25s : %s", k, v)

    return summary
