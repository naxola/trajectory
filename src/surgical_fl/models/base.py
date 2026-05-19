""" Esta clase sirve como contrato (clase abstracta) para todos los modelos de ML que se utilicen en el proyecto.

Contiene la interfaz mínima que cualquier modelo debe cumplir:
  - forward()                     → inferencia (lo hereda de nn.Module)
  - get_parameters / set_parameters → serialización para Flower
  - metadata()                    → describe qué predice el modelo

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

@dataclass
class ModelMetadata:
    """Metadatos del modelo"""
    task: str            # e.g. "trajectory_prediction". "force_estimation", "multi_task"
    description: str     # e.g. "Predicts the next 10 points of a trajectory"
    input_shape: tuple   # shape de UN sample de entrada (sin batch)
    output_shape: tuple  # shape de UN sample de salida (sin batch)

#Esta clase es util para la obtención de los pesos del modelo
class SerializableParametersMixin:
    """
    Mixin para serializar los parámetros del modelo

    Operación útil de forma general: checkpointing, comparación de modelos,
    transferencia entre frameworks, debugging y — entre otros usos —
    sincronización de pesos en frameworks de aprendizaje federado.
 
    Se mantiene como mixin separado para no contaminar la lógica de
    arquitectura del modelo con código de serialización.
    """
    def get_parameters(self) -> list[np.ndarray]:
        """Obtiene los parámetros del modelo en formato numpy"""
        return [p.cpu().numpy() for p in self.state_dict().values()]
    
    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Establece los parámetros del modelo"""
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.load_state_dict(state_dict, strict=True)
    

class SurgicalModel(SerializableParametersMixin, nn.Module, ABC):
    """
    Contrato mínimo para cualquier modelo quirúrgico.
 
    Subclases por tipo de tarea:
      - TrajectoryModel  (models/trajectory/base.py)
      - ForceModel       (models/force/base.py, futuro)
      - MultitaskModel   (models/multitask/base.py, futuro)
    """
    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Devuelve metadatos del modelo"""
        pass
    