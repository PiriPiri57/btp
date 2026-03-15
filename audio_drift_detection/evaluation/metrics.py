"""Evaluation metrics for drift detection performance."""

from __future__ import annotations

import numpy as np


def detection_delay(
    true_drift_points: list[int],
    detected_timestamps: list[int],
) -> float:
    """Average delay between true drift and first detection after it.

    Args:
        true_drift_points: Ground-truth drift timestamps.
        detected_timestamps: Detected drift timestamps (sorted).

    Returns:
        Mean detection delay (NaN if no true drift was detected).
    """
    if not true_drift_points or not detected_timestamps:
        return float("nan")

    delays: list[int] = []
    det = sorted(detected_timestamps)
    for tp in true_drift_points:
        after = [d for d in det if d >= tp]
        if after:
            delays.append(after[0] - tp)
    return float(np.mean(delays)) if delays else float("nan")


def false_positive_rate(
    true_drift_points: list[int],
    detected_timestamps: list[int],
    total_steps: int,
    tolerance: int = 50,
) -> float:
    """Fraction of detections that are false positives.

    A detection is a *true positive* if it falls within ``tolerance`` steps
    of any true drift point; otherwise it is a false positive.

    Args:
        true_drift_points: Ground-truth drift timestamps.
        detected_timestamps: Detected drift timestamps.
        total_steps: Total number of time steps in the stream.
        tolerance: Window around a true drift point.

    Returns:
        False positive rate ∈ [0, 1].
    """
    if not detected_timestamps:
        return 0.0
    fp = 0
    for d in detected_timestamps:
        if not any(abs(d - tp) <= tolerance for tp in true_drift_points):
            fp += 1
    return fp / len(detected_timestamps)


def precision_recall_f1(
    true_drift_points: list[int],
    detected_timestamps: list[int],
    tolerance: int = 50,
) -> dict[str, float]:
    """Precision, recall, and F1 for drift detection.

    Args:
        true_drift_points: Ground-truth drift timestamps.
        detected_timestamps: Detected drift timestamps.
        tolerance: Window around a true drift point.

    Returns:
        Dict with keys ``precision``, ``recall``, ``f1``.
    """
    if not detected_timestamps:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # True positives: detected timestamps that are near a true drift
    tp_det = sum(
        1 for d in detected_timestamps
        if any(abs(d - tp) <= tolerance for tp in true_drift_points)
    )
    # True positives: true drifts that have at least one nearby detection
    tp_true = sum(
        1 for tp in true_drift_points
        if any(abs(d - tp) <= tolerance for d in detected_timestamps)
    )

    precision = tp_det / len(detected_timestamps) if detected_timestamps else 0.0
    recall = tp_true / len(true_drift_points) if true_drift_points else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}
