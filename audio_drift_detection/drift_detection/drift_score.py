"""Drift score computation.

drift_score = α × mean_reconstruction_loss + β × loss_variance

Optionally adds Gaussian noise to the score for additional privacy.
"""

from __future__ import annotations

import numpy as np


def compute_drift_score(
    mean_loss: float,
    loss_variance: float,
    alpha: float = 0.7,
    beta: float = 0.3,
    noise_sigma: float = 0.0,
) -> float:
    """Compute the hybrid STUDD + ensemble-disagreement drift score.

    Args:
        mean_loss: Mean reconstruction loss across students (STUDD signal).
        loss_variance: Variance of reconstruction losses (ensemble disagreement).
        alpha: Weight for mean loss.
        beta: Weight for loss variance.
        noise_sigma: Std-dev of optional Gaussian noise for privacy.

    Returns:
        Scalar drift score.
    """
    score = alpha * mean_loss + beta * loss_variance
    if noise_sigma > 0:
        score += np.random.normal(0, noise_sigma)
    return float(score)
