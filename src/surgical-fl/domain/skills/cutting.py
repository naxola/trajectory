# Clase concreta CuttingSkill
"""
Habilidad quirúrgica: Cutting (incisión).
La idea es evaluar la habilidad de cortar un tejido
"""
import numpy as np
from .base import SurgicalSkill


class CuttingSkill(SurgicalSkill):

    @property
    def name(self) -> str:
        return "cutting"

    def evaluate_trajectory(
        self,
        trajectory: np.ndarray,
        world_state: object | None = None,
    ) -> dict[str, float]:
        """
        Métricas para la tarea de corte:

        (world_state=None):
          - path_error: desviación lateral respecto a la línea ideal
          - smoothness: uniformidad de la velocidad
          - length:     longitud total recorrida
          - task_score: puntuación compuesta [0, 1]
        """
        if len(trajectory) < 2:
            return {
                "path_error": 999.0,
                "smoothness": 0.0,
                "length":     0.0,
                "task_score": 0.0,
            }

        x, y = trajectory[:, 0], trajectory[:, 1]

        # Error: desviación media respecto a la línea ideal (y = y_start)
        path_error = float(np.mean(np.abs(y - y[0])))

        # Suavidad: inverso de la varianza de velocidades
        dx, dy = np.diff(x), np.diff(y)
        speeds = np.sqrt(dx**2 + dy**2)
        smoothness = float(1.0 / (1.0 + np.std(speeds)))

        # Longitud del recorrido
        length = float(np.sum(speeds))

        # Score compuesto
        error_penalty = float(np.clip(1.0 - path_error * 5, 0, 1))
        task_score = 0.5 * smoothness + 0.5 * error_penalty

        metrics = {
            "path_error": path_error,
            "smoothness": smoothness,
            "length":     length,
            "task_score": task_score,
        }

        # Esto es para mas adelante (no le pongo lógica porque no existe el mundo todavía)
        if world_state is not None:
            metrics.update(self._evaluate_with_context(trajectory, world_state))

        return metrics

    def _evaluate_with_context(
        self,
        trajectory: np.ndarray,
        world_state: object,
    ) -> dict[str, float]:
        """
        Métricas clínicas que requieren conocimiento del entorno.

        Placeholder hasta que exista world/. Se implementará cuando llegue
        la Fase avanzada con simulador físico.
        """
        return {}  # TODO: tissue_damage, vital_proximity, incision_alignment