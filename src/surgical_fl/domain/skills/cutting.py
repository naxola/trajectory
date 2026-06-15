# Clase concreta CuttingSkill
"""
Habilidad quirúrgica: Cutting (incisión).
La idea es evaluar la habilidad de cortar un tejido
"""
from dataclasses import dataclass
import numpy as np
from .base import SurgicalSkill


@dataclass(frozen=True)
class CuttingConstraints:
    min_length: float = 0.5
    max_curvature: float = 0.3
    tolerance: float = 0.1  # desviación (en unidades de la curva) que ya da score 0


def _polyline_deviations(trajectory: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Distancia de cada punto de `trajectory` al polilínea `reference`.

    Mide desviación geométrica (punto-a-segmento), no índice-a-índice, así que
    funciona aunque la trayectoria y la referencia tengan distinto nº de puntos.
    """
    A = reference[:-1]
    AB = reference[1:] - A
    ab2 = np.maximum((AB ** 2).sum(axis=1), 1e-12)  # (S,)
    dists = np.empty(len(trajectory))
    for i, p in enumerate(trajectory):
        t = np.clip(((p - A) * AB).sum(axis=1) / ab2, 0.0, 1.0)  # proyección por segmento
        proj = A + t[:, None] * AB
        dists[i] = np.sqrt(((p - proj) ** 2).sum(axis=1)).min()
    return dists


class CuttingSkill(SurgicalSkill):

    @property
    def constraints(self) -> CuttingConstraints:
        return CuttingConstraints()

    def is_valid(self, trajectory: np.ndarray) -> bool:
        """Devuelve True si la trayectoria tiene forma (T, 2) con T >= 2."""
        return trajectory.ndim == 2 and trajectory.shape[1] == 2 and len(trajectory) >= 2

    @property
    def name(self) -> str:
        return "cutting"

    def evaluate_trajectory(
        self,
        trajectory: np.ndarray,
        world_state: object | None = None,
        reference: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Métricas para la tarea de corte:

          - path_error: desviación media respecto a la incisión ideal
          - smoothness: uniformidad de la velocidad
          - length:     longitud total recorrida
          - task_score: puntuación compuesta [0, 1]

        Args:
            reference: curva ideal (R, 2) contra la que medir la desviación.
                       Si se pasa, path_error es la distancia geométrica media
                       al polilínea de referencia (corte real vs corte ideal),
                       lo que permite comparar cortes de la MISMA forma objetivo
                       aunque sean curvos. Si es None (compat), se usa la
                       desviación lateral respecto a la línea recta y=y_start.
        """
        if len(trajectory) < 2:
            return {
                "path_error": 999.0,
                "smoothness": 0.0,
                "length":     0.0,
                "task_score": 0.0,
            }

        x, y = trajectory[:, 0], trajectory[:, 1]

        if reference is not None:
            # Desviación geométrica respecto a la incisión ideal.
            path_error = float(np.mean(_polyline_deviations(trajectory, reference)))
            error_penalty = float(
                np.clip(1.0 - path_error / self.constraints.tolerance, 0, 1)
            )
        else:
            # Compat: desviación lateral respecto a la línea ideal (y = y_start)
            path_error = float(np.mean(np.abs(y - y[0])))
            error_penalty = float(np.clip(1.0 - path_error * 5, 0, 1))

        # Suavidad: inverso de la varianza de velocidades
        dx, dy = np.diff(x), np.diff(y)
        speeds = np.sqrt(dx**2 + dy**2)
        smoothness = float(1.0 / (1.0 + np.std(speeds)))

        # Longitud del recorrido
        length = float(np.sum(speeds))

        # Score compuesto
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