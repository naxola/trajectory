"""
RNN (GRU) para predicción de trayectorias.

A diferencia del MLP, mantiene estado oculto entre pasos temporales.
Esto permite que el rollout autorregresivo genere trayectorias completas
sin colapsar, porque el hidden state codifica la posición en la curva.
"""
import torch
import torch.nn as nn

from .base import TrajectoryModel


class TrajectoryRNN(TrajectoryModel):

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        output_dim: int = 2,
        dropout: float = 0.1,
        num_layers: int = 1,
    ):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.fc_out(out)

    def predict_step(
        self, x: torch.Tensor, hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, hidden = self.rnn(x, hidden)
        return self.fc_out(out), hidden
