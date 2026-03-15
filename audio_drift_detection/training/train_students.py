"""Train student models on disjoint data subsets to mimic teacher embeddings."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.ensemble_manager import EnsembleManager
from models.teacher_model import TeacherEncoder

logger = logging.getLogger("audio_drift")


def train_students(
    ensemble: EnsembleManager,
    teacher: TeacherEncoder,
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> None:
    """Train each student on its own disjoint data partition.

    Args:
        ensemble: ``EnsembleManager`` containing student models.
        teacher: Frozen teacher encoder.
        features: Full training features ``(N, D)``.
        labels: Integer labels ``(N,)``  (used only for splitting).
        epochs: Training epochs per student.
        batch_size: Mini-batch size.
        learning_rate: Optimiser learning rate.
        device: ``'cpu'`` or ``'cuda'``.
    """
    splits = EnsembleManager.split_data(features, labels, ensemble.num_students)
    criterion = nn.MSELoss()

    teacher.to(device)
    teacher.eval()

    for idx, (student, (feat_sub, _)) in enumerate(zip(ensemble.students, splits)):
        logger.info(
            "Training student %d/%d on %d samples …",
            idx + 1,
            ensemble.num_students,
            len(feat_sub),
        )
        tensor_x = torch.tensor(feat_sub, dtype=torch.float32)
        dataset = TensorDataset(tensor_x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        optimiser = torch.optim.Adam(student.parameters(), lr=learning_rate)
        student.train()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                with torch.no_grad():
                    target_emb = teacher(batch_x)
                pred_emb = student(batch_x)
                loss = criterion(pred_emb, target_emb)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total_loss += loss.item() * len(batch_x)

            avg = total_loss / len(tensor_x)
            if epoch % max(1, epochs // 3) == 0 or epoch == 1:
                logger.info(
                    "    Student %d  epoch %d/%d — loss: %.6f",
                    idx + 1, epoch, epochs, avg,
                )

        student.eval()
    logger.info("All students trained.")
