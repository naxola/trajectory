#Clase base para la generación de trayectorias.
"""
Defino trayectoria como secuencia de posiciones en 2d o 3d en el espacio 

Esta clase hereda de SurgicalDataGenerator y añade:
numero de tiemsteps para la trayectoria: trajectory_length
Dimensiones (2d o 3D) dims
garantia de SHAPE (N, T, dims) en todos los outputs.
"""

import numpy as np
from ..base import SurgicalDataGenerator, SurgicalGeneratorMetadata


class TrajectoryGenerator(SurgicalDataGenerator):
    def __init__(self, skill: str, trajectory_length=50, dims: int = 2, seed: int |None=None):
        super().__init__(seed)
        self.skill = skill
        self.trajectory_length = trajectory_length
        self.dims = dims

    @property
    def metadata(self) -> SurgicalGeneratorMetadata:
        return SurgicalGeneratorMetadata(
            skill=self.skill,
            output_type="trajectory",
            output_shape=(self.trajectory_length, self.dims),
            units="meters",
            description=f"Trayectorias de {self.skill} con {self.trajectory_length} pasos y {self.dims} dimensiones."
        )
    
    def generate(self, num_samples: int) -> np.ndarray:
        """ Devuelve el array (N, T, dims) en float32 """
        raise NotImplementedError("Subclases deben implementar generate")