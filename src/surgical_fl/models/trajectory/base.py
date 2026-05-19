"""
Base para modelos de predicción de trayectorias.

Todos los modelos de trayectoria (MLP, RNN, Transformer...) heredan de aquí.
Comparten:
  - Input  shape: (batch, T, dims)   donde dims=2 (planar) o dims=3 (3D)
  - Output shape: (batch, T, dims)   predicción del siguiente punto

Cada subclase define su arquitectura concreta en forward().
"""

import torch
from abc import abstractmethod

from ..base import SurgicalModel, ModelMetadata

class TrajectoryModel(SurgicalModel):
    """
    Modelo que predice el siguiente punto de una trayectoria quirúrgica.

    Args:
        input_dim:  dimensión espacial (2 para xy, 3 para xyz)
        output_dim: dimensión de salida (normalmente == input_dim)
    """
    def __init__(self, input_dim: int = 2, output_dim: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @property
    def metadata(self) -> ModelMetadata:
        """Metadatos del modelo"""
        return ModelMetadata(
            task="trajectory_prediction",
            description=f"Predice el siguiente punto en {self.input_dim}D",
            input_shape=(self.input_dim,),
            output_shape=(self.output_dim,),
        )
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predice el siguiente punto de la trayectoria"""
        """
        Args:
            x: Tensor de entrada con forma (batch, T, input_dim)
        Returns:
            Tensor de salida con forma (batch, T, output_dim)
        """
        ...