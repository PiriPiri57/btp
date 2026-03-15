"""Pretrain the teacher autoencoder and freeze the encoder."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.teacher_model import TeacherAutoencoder, TeacherEncoder

logger = logging.getLogger("audio_drift")


def train_teacher(
    features: np.ndarray,
    input_dim: int,
    hidden_dims: list[int] | None = None,
    embedding_dim: int = 64,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> TeacherEncoder:
    """Pretrain teacher autoencoder and return the frozen encoder.

    Args:
        features: Training features ``(N, input_dim)``.
        input_dim: Feature dimensionality.
        hidden_dims: Hidden layer sizes for the encoder / decoder.
        embedding_dim: Embedding vector size.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Optimiser learning rate.
        device: ``'cpu'`` or ``'cuda'``.

    Returns:
        Frozen ``TeacherEncoder`` ready for inference.
    """
    autoencoder = TeacherAutoencoder(input_dim, hidden_dims, embedding_dim).to(device)
    optimiser = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    tensor_x = torch.tensor(features, dtype=torch.float32)
    dataset = TensorDataset(tensor_x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    logger.info("Training teacher autoencoder for %d epochs …", epochs)
    autoencoder.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            _, recon = autoencoder(batch_x)
            loss = criterion(recon, batch_x)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(batch_x)

        avg_loss = total_loss / len(tensor_x)
        if epoch % max(1, epochs // 5) == 0 or epoch == 1:
            logger.info("  Teacher epoch %d/%d — loss: %.6f", epoch, epochs, avg_loss)

    # Freeze encoder
    frozen_encoder = autoencoder.freeze_encoder()
    logger.info("Teacher encoder frozen.")
    return frozen_encoder
