"""Ensemble manager — manage N student models on disjoint data subsets.

Responsibilities:
  * Split data into disjoint partitions for privacy.
  * Collective inference: compute per-student reconstruction losses.
  * Aggregate statistics: mean loss & loss variance.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from models.student_model import StudentModel

logger = logging.getLogger("audio_drift")


class EnsembleManager:
    """Orchestrate an ensemble of student models."""

    def __init__(
        self,
        num_students: int,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        embedding_dim: int = 64,
        device: str = "cpu",
    ) -> None:
        self.num_students = num_students
        self.device = device
        self.students: list[StudentModel] = []

        for _ in range(num_students):
            student = StudentModel(input_dim, hidden_dims, embedding_dim).to(device)
            self.students.append(student)

        logger.info("Initialised ensemble with %d students.", num_students)

    # ------------------------------------------------------------------
    # Data splitting
    # ------------------------------------------------------------------
    @staticmethod
    def split_data(
        features: np.ndarray,
        labels: np.ndarray,
        num_splits: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Split data into ``num_splits`` disjoint subsets.

        Args:
            features: ``(N, D)``
            labels: ``(N,)``
            num_splits: Number of partitions.

        Returns:
            List of ``(features_subset, labels_subset)`` tuples.
        """
        indices = np.arange(len(features))
        np.random.shuffle(indices)
        splits = np.array_split(indices, num_splits)
        return [(features[s], labels[s]) for s in splits]

    # ------------------------------------------------------------------
    # Per-sample loss computation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_losses(
        self,
        feature_tensor: torch.Tensor,
        teacher_embedding: torch.Tensor,
    ) -> list[float]:
        """Compute MSE loss for each student against the teacher embedding.

        Args:
            feature_tensor: ``(1, input_dim)`` or ``(input_dim,)``
            teacher_embedding: ``(1, embedding_dim)`` or ``(embedding_dim,)``

        Returns:
            List of per-student MSE losses.
        """
        if feature_tensor.dim() == 1:
            feature_tensor = feature_tensor.unsqueeze(0)
        if teacher_embedding.dim() == 1:
            teacher_embedding = teacher_embedding.unsqueeze(0)

        feature_tensor = feature_tensor.to(self.device)
        teacher_embedding = teacher_embedding.to(self.device)

        losses: list[float] = []
        mse = nn.MSELoss()
        for student in self.students:
            student.eval()
            pred = student(feature_tensor)
            loss = mse(pred, teacher_embedding).item()
            losses.append(loss)
        return losses

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------
    @staticmethod
    def compute_statistics(losses: list[float]) -> tuple[float, float]:
        """Compute mean and variance of per-student losses.

        Returns:
            ``(mean_loss, loss_variance)``
        """
        arr = np.array(losses)
        return float(np.mean(arr)), float(np.var(arr))
