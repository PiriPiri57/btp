"""Drift scenarios — create stream orderings that inject concept drift.

Three scenarios:
  1. Abrupt drift   — sudden class-distribution change at a drift point.
  2. Gradual drift  — slow mixing of two distributions over a transition window.
  3. Noise drift    — Gaussian noise added to features after drift point.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("audio_drift")


def _split_by_class_groups(
    labels: np.ndarray,
    group_a_classes: list[int],
    group_b_classes: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return indices belonging to two class groups."""
    mask_a = np.isin(labels, group_a_classes)
    mask_b = np.isin(labels, group_b_classes)
    return np.where(mask_a)[0], np.where(mask_b)[0]


def abrupt_drift(
    features: np.ndarray,
    labels: np.ndarray,
    drift_point_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int, list[int]]:
    """Create an abrupt-drift stream ordering.

    Before ``drift_point``: samples from class group A.
    After ``drift_point``:  samples from class group B.

    Returns:
        ``(stream_features, stream_labels, drift_point, true_drift_indices)``
    """
    rng = np.random.RandomState(seed)
    unique = sorted(np.unique(labels))
    mid = len(unique) // 2
    group_a, group_b = unique[:mid], unique[mid:]

    idx_a, idx_b = _split_by_class_groups(labels, group_a, group_b)
    rng.shuffle(idx_a)
    rng.shuffle(idx_b)

    stream_idx = np.concatenate([idx_a, idx_b])
    drift_point = len(idx_a)
    true_drifts = [drift_point]

    logger.info(
        "Abrupt drift: %d pre-drift samples, %d post-drift samples, drift at t=%d",
        len(idx_a), len(idx_b), drift_point,
    )
    return features[stream_idx], labels[stream_idx], drift_point, true_drifts


def gradual_drift(
    features: np.ndarray,
    labels: np.ndarray,
    drift_point_ratio: float = 0.5,
    transition_window: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int, list[int]]:
    """Create a gradual-drift stream.

    A transition window linearly mixes group A → group B.

    Returns:
        ``(stream_features, stream_labels, drift_point, true_drift_indices)``
    """
    rng = np.random.RandomState(seed)
    unique = sorted(np.unique(labels))
    mid = len(unique) // 2
    group_a, group_b = unique[:mid], unique[mid:]

    idx_a, idx_b = _split_by_class_groups(labels, group_a, group_b)
    rng.shuffle(idx_a)
    rng.shuffle(idx_b)

    total = len(idx_a) + len(idx_b)
    drift_point = int(total * drift_point_ratio)
    trans_start = max(0, drift_point - transition_window // 2)
    trans_end = min(total, drift_point + transition_window // 2)

    # Before transition: strictly A
    pre = idx_a[: trans_start]
    # Transition zone: mix
    remaining_a = list(idx_a[trans_start:])
    remaining_b = list(idx_b.copy())
    mixed: list[int] = []
    for i in range(trans_end - trans_start):
        prob_b = i / max(1, (trans_end - trans_start))
        if rng.rand() < prob_b and remaining_b:
            mixed.append(remaining_b.pop(0))
        elif remaining_a:
            mixed.append(remaining_a.pop(0))
        elif remaining_b:
            mixed.append(remaining_b.pop(0))
    # After transition: strictly B
    post = np.array(remaining_b)

    stream_idx = np.concatenate([pre, np.array(mixed), post])
    true_drifts = [trans_start]

    logger.info(
        "Gradual drift: transition window [%d, %d], drift_point=%d",
        trans_start, trans_end, drift_point,
    )
    return features[stream_idx], labels[stream_idx], drift_point, true_drifts


def noise_drift(
    features: np.ndarray,
    labels: np.ndarray,
    drift_point_ratio: float = 0.5,
    noise_scale: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int, list[int]]:
    """Create a noise-drift stream.

    After the drift point, Gaussian noise is added to the feature vectors.

    Returns:
        ``(stream_features, stream_labels, drift_point, true_drift_indices)``
    """
    rng = np.random.RandomState(seed)
    
    unique = sorted(np.unique(labels))
    mid = len(unique) // 2
    group_a = unique[:mid]
    
    idx_a = np.where(np.isin(labels, group_a))[0]
    sub_feat = features[idx_a]
    sub_labels = labels[idx_a]
    
    n = len(sub_feat)
    order = rng.permutation(n)
    
    stream_feat = sub_feat[order].copy()
    stream_lbl = sub_labels[order].copy()

    drift_point = int(n * drift_point_ratio)
    stream_feat[drift_point:] += rng.normal(
        0, noise_scale, size=stream_feat[drift_point:].shape
    ).astype(stream_feat.dtype)

    true_drifts = [drift_point]

    logger.info(
        "Noise drift: noise_scale=%.2f, drift at t=%d", noise_scale, drift_point,
    )
    return stream_feat, stream_lbl, drift_point, true_drifts


# ------------------------------------------------------------------
# Convenience dispatcher
# ------------------------------------------------------------------
SCENARIOS = {
    "abrupt": abrupt_drift,
    "gradual": gradual_drift,
    "noise": noise_drift,
}


def create_drift_scenario(
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, int, list[int]]:
    """Create a named drift scenario.

    Args:
        name: One of ``'abrupt'``, ``'gradual'``, ``'noise'``.
        features: ``(N, D)``
        labels: ``(N,)``

    Returns:
        ``(stream_features, stream_labels, drift_point, true_drift_indices)``
    """
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Choose from {list(SCENARIOS)}")
    return SCENARIOS[name](features, labels, **kwargs)
