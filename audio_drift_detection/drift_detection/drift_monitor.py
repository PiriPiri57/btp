"""Drift monitor — wraps River's ADWIN detector."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from river.drift import ADWIN

logger = logging.getLogger("audio_drift")


@dataclass
class DriftEvent:
    """Record of a single detected drift event."""
    timestamp: int
    drift_score: float
    mean_loss: float
    loss_variance: float


class DriftMonitor:
    """Online drift detector backed by ADWIN.

    Feed drift scores one at a time; the monitor logs detected drifts.
    """

    def __init__(self, delta: float = 0.002) -> None:
        """Initialise the drift monitor.

        Args:
            delta: ADWIN sensitivity parameter (smaller → more sensitive).
        """
        self.adwin = ADWIN(delta=delta)
        self.drift_events: list[DriftEvent] = []
        self.scores: list[float] = []
        self.mean_losses: list[float] = []
        self.variances: list[float] = []

    def update(
        self,
        timestamp: int,
        drift_score: float,
        mean_loss: float,
        loss_variance: float,
    ) -> bool:
        """Feed a new observation and check for drift.

        Args:
            timestamp: Current time step.
            drift_score: Combined drift score.
            mean_loss: Mean reconstruction loss.
            loss_variance: Ensemble loss variance.

        Returns:
            ``True`` if drift is detected at this step.
        """
        self.scores.append(drift_score)
        self.mean_losses.append(mean_loss)
        self.variances.append(loss_variance)

        self.adwin.update(drift_score)
        if self.adwin.drift_detected:
            event = DriftEvent(timestamp, drift_score, mean_loss, loss_variance)
            self.drift_events.append(event)
            logger.warning(
                "  DRIFT DETECTED at t=%d  (score=%.4f, mean_loss=%.4f, var=%.4f)",
                timestamp, drift_score, mean_loss, loss_variance,
            )
            return True
        return False

    def reset(self) -> None:
        """Reset the detector state (e.g. between experiments)."""
        self.adwin = ADWIN(delta=self.adwin.delta)
        self.drift_events.clear()
        self.scores.clear()
        self.mean_losses.clear()
        self.variances.clear()

    @property
    def detected_timestamps(self) -> list[int]:
        """Return list of timestamps where drift was detected."""
        return [e.timestamp for e in self.drift_events]
