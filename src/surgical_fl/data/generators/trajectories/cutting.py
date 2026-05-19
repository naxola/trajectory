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
           x = np.cumsum(speeds)
           x = x / x[-1]
           y = self.rng.normal(0, self.noise_std, self.trajectory_length)
           trajectories[i, :, 0] = x
           trajectories[i, :, 1] = y
        return trajectories.astype(np.float32)


class CurvedCutGenerator(TrajectoryGenerator):
    """
    Corte con curvatura parabólica en Y.
    Modela tejidos con resistencia variable que desvían la trayectoria.
    """
    def __init__(self, trajectory_length: int = 50, curvature: float = 0.1,
                 noise_std: float = 0.02, speed_variance: float = 0.01,
                 seed: int | None = None):
        super().__init__("cutting", trajectory_length=trajectory_length, seed=seed)
        self.curvature = curvature
        self.noise_std = noise_std
        self.speed_variance = speed_variance

    def generate(self, num_samples: int) -> np.ndarray:
        trajectories = np.zeros((num_samples, self.trajectory_length, 2))
        for i in range(num_samples):
            speeds = (1 / self.trajectory_length) + self.rng.normal(0, self.speed_variance, self.trajectory_length)
            speeds = np.clip(speeds, 1e-4, None)
            x = np.cumsum(speeds)
            x = x / x[-1]
            # U-shape: edges (x≈0, x≈1) are higher than center (x≈0.5)
            y_curve = self.curvature * (x - 0.5) ** 2
            y_noise = self.rng.normal(0, self.noise_std, self.trajectory_length)
            trajectories[i, :, 0] = x
            trajectories[i, :, 1] = y_curve + y_noise
        return trajectories.astype(np.float32)


class SplineCutGenerator(TrajectoryGenerator):
    """
    Corte con trayectoria interpolada sobre puntos de control aleatorios.
    Modela incisiones con cambios de dirección complejos.
    """
    def __init__(self, trajectory_length: int = 50, n_control_points: int = 4,
                 noise_std: float = 0.02, seed: int | None = None):
        super().__init__("cutting", trajectory_length=trajectory_length, seed=seed)
        self.n_control_points = n_control_points
        self.noise_std = noise_std

    def generate(self, num_samples: int) -> np.ndarray:
        trajectories = np.zeros((num_samples, self.trajectory_length, 2))
        t_out = np.linspace(0, 1, self.trajectory_length)
        for i in range(num_samples):
            t_ctrl = np.linspace(0, 1, self.n_control_points + 2)
            y_ctrl = self.rng.normal(0, 0.1, self.n_control_points + 2)
            y_ctrl[0] = 0.0
            y_ctrl[-1] = 0.0
            y_interp = np.interp(t_out, t_ctrl, y_ctrl)
            y_noise = self.rng.normal(0, self.noise_std, self.trajectory_length)
            trajectories[i, :, 0] = t_out
            trajectories[i, :, 1] = y_interp + y_noise
        return trajectories.astype(np.float32)
