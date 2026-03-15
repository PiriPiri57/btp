"""Teacher model — frozen MLP encoder for STUDD.

Architecture: input_dim → hidden layers → embedding_dim
Pretrained as an autoencoder (encoder + decoder) then the encoder is frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TeacherEncoder(nn.Module):
    """MLP encoder that maps feature vectors to embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, embedding_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input features.

        Args:
            x: ``(batch, input_dim)``

        Returns:
            Embeddings ``(batch, embedding_dim)``
        """
        return self.encoder(x)


class TeacherAutoencoder(nn.Module):
    """Autoencoder wrapper used for pretraining the teacher encoder.

    After pretraining, freeze the encoder and discard the decoder.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.encoder = TeacherEncoder(input_dim, hidden_dims, embedding_dim)

        # Mirror decoder
        dec_layers: list[nn.Module] = []
        prev = embedding_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()])
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            ``(embedding, reconstruction)``
        """
        emb = self.encoder(x)
        recon = self.decoder(emb)
        return emb, recon

    def freeze_encoder(self) -> TeacherEncoder:
        """Freeze encoder weights and return the encoder module."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        return self.encoder
