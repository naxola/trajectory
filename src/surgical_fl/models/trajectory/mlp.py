"""
MLP — Multi-Layer Perceptron para predicción de trayectorias.

Arquitectura mínima: 3 capas densas con ReLU y dropout.
Procesa cada punto independientemente, sin memoria temporal.

Sirve como baseline. Modelos posteriores (RNN, Transformer) deberían
superar sus métricas o no merece la pena su complejidad añadida.
"""
import torch
import torch.nn as nn
from collections import OrderedDict

from .base import TrajectoryModel


class TrajectoryMLP(TrajectoryModel):
    """
    MLP de 3 capas para predicción punto-a-punto.

    Args:
        input_dim:  dimensión espacial (2 para xy, 3 para xyz)
        hidden_dim: neuronas en capas ocultas
        output_dim: dimensión de salida (normalmente == input_dim)
        dropout:    regularización (útil en federado con pocos datos)
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        output_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(OrderedDict([
            ("fc1",   nn.Linear(input_dim, hidden_dim)),
            ("relu1", nn.ReLU()),
            ("drop1", nn.Dropout(dropout)),
            ("fc2",   nn.Linear(hidden_dim, hidden_dim)),
            ("relu2", nn.ReLU()),
            ("drop2", nn.Dropout(dropout)),
            ("fc3",   nn.Linear(hidden_dim, output_dim)),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)