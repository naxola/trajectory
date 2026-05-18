# Clase abstracta SurgicalSkill que define la interfaz para las skills
# La responsabilidad es únicamente definir una funcion abstracta de evaluación de la tarea

"""
Clase base para habilidades quirúrgicas.

Una skill define QUÉ significa ejecutar bien una tarea quirúrgica,
no CÓMO ejecutarla (eso es responsabilidad de la política aprendida)
ni DÓNDE ejecutarla (eso es responsabilidad del World en fase avanzada).

Esta capa es dominio puro: no conoce PyTorch ni Flower.
"""
from abc import ABC, abstractmethod
import numpy as np


class SurgicalSkill(ABC):
    """
    Contrato mínimo para cualquier habilidad quirúrgica.

    Añadir una nueva skill (suturing, cauterizing, etc.) consiste en
    crear un fichero que herede de esta clase e implemente evaluate_trajectory.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identificador único de la habilidad (ej: 'cutting')."""
        ...

    @abstractmethod
    def evaluate_trajectory(
        self,
        trajectory: np.ndarray,
        world_state: object | None = None,
    ) -> dict[str, float]:
        """
        Evalúa la calidad de una trayectoria ejecutada.

        Args:
            trajectory:  array (T, D) con la trayectoria. D=2 ahora, D=3 en futuro.
            world_state: estado del entorno (Fase avanzada). None en Fase 1
                         → métricas puramente geométricas.
                         Cuando exista un World real, contendrá tejido,
                         zonas vitales, etc. para métricas clínicas reales.

        Returns:
            dict con métricas (ej: 'smoothness', 'path_error', 'task_score').
            'task_score' debería estar en [0, 1] como métrica agregada.
        """
        ...