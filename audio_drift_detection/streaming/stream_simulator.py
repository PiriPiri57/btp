"""Stream simulator — yield (timestamp, feature_vector, label) one at a time."""

from __future__ import annotations

from typing import Generator

import numpy as np


def stream_features(
    features: np.ndarray,
    labels: np.ndarray,
    order: np.ndarray | None = None,
) -> Generator[tuple[int, np.ndarray, int], None, None]:
    """Simulate a streaming audio pipeline.

    Args:
        features: ``(N, D)`` feature matrix.
        labels: ``(N,)`` integer labels.
        order: Optional permutation of indices (for drift scenarios).

    Yields:
        ``(timestamp, feature_vector, label)``
    """
    if order is None:
        order = np.arange(len(features))

    for t, idx in enumerate(order):
        yield t, features[idx].copy(), int(labels[idx])
