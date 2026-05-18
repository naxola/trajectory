"""
Aqui definimos la geometria de corte (no los perfiles de los hospitales)
La variabilidad entre cada hospital (ruido, velocidad, estilo) se inyecta desde profiles y se combina en la factory

Aqui generamos tres tipos de cortes:
- LinearCutGenerator -> Corte lineal a lo largo del eje X
- CurvedCutGenerator -> Corte con parábola suave (puede ser un tejido con resistencia al corte variable)
- SplineCutGenerator -> Corte con spline teniendo puntos de control.
"""

import numpy as np
from .base import TrajectoryGenerator

class LinearCutGenerator(TrajectoryGenerator):
    """
    Respresenta el movimiento perfecto recto de un bisturí en linea recta.
    El ruido y la velocidad son parámetros que dependen de los perfiles.
    
    Args:
    noise_std: desviación lateral en Y (ruido del hospital)
    speed_variance: varianza en la velocidad (estilo del hospital)
    
    """
    def __init__(self, trajectory_length: int = 50, noise_std: float = 0.02, speed_variance: float = 0.01, seed: int | None = None):
        super().__init__("cutting", trajectory_length = trajectory_length, seed = seed)
        self.noise_std = noise_std
        self.speed_variance = speed_variance

    def generate(self, num_samples: int) -> np.ndarray:
        """ Genera un lote de trayectorias lineales """
        trajectories = np.zeros((num_samples, self.trajectory_length, 2))

        for i in range(num_samples):
           speeds = (1/self.trajectory_length) + self.rng.normal(0, self.speed_variance, self.trajectory_length)
           speeds = np.clip(speeds, 1e-4, None) 
           x= np.cumsum(speeds)
           x=x/x[-1]
           y= self.rng.normal(0, self.noise_std, self.trajectory_length)
           trajectories[i,:,0]=x
           trajectories[i,:,1]=y
        return trajectories.astype(np.float32)
        
        