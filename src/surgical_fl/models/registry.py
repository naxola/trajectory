
"""
Registro de modelos disponibles.
 
API simétrica a domain/skills/registry.py, domain/profiles/registry.py
y data/generators/registry.py:
  get_model("mlp")  → instancia de SurgicalModel
  list_models()     → ["mlp", ...]
 
Permite seleccionar el modelo desde pyproject.toml por string,
sin imports hardcodeados en task.py.
"""

from typing import Callable
from .base import SurgicalModel
from .trajectory.mlp import TrajectoryMLP
from .trajectory.rnn import TrajectoryRNN

_REGISTRY: dict[str, Callable[..., SurgicalModel]] = {
    "mlp": lambda **kwargs: TrajectoryMLP(**kwargs),
    "rnn": lambda **kwargs: TrajectoryRNN(**kwargs),
}


def get_model(name: str, **kwargs) -> SurgicalModel:
    """
    Instancia un modelo registrado por nombre.

    Args:
        name:   identificador del modelo (debe coincidir con pyproject.toml)
        kwargs: parámetros de construcción (hidden_dim, dropout, etc.)

    Raises:
        ValueError: si el modelo no está registrado
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(
            f"Modelo '{name}' no encontrado. Disponibles: {available}"
        )
    return _REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    return list(_REGISTRY.keys())