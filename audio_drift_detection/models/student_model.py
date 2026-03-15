"""Student model — mimics teacher embeddings.

Loss: MSE(student_output, teacher_embedding)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StudentModel(nn.Module):
    """MLP student that predicts teacher embeddings from raw features."""

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
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict teacher embedding.

        Args:
            x: ``(batch, input_dim)``

        Returns:
            Predicted embedding ``(batch, embedding_dim)``
        """
        return self.net(x)
