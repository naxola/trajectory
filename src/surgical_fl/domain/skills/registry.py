"""
Registro de skills quirúrgicas. API simétrica a profiles/registry.py.
  get_skill("cutting")  → CuttingSkill instance
  list_skills()         → ["cutting", ...]
"""

from .base import SurgicalSkill
from .cutting import CuttingSkill

_REGISTRY: dict[str, SurgicalSkill] = {
    "cutting": CuttingSkill(),
    
}


def get_skill(name: str) -> SurgicalSkill:
    """Devuelve la skill por nombre.

    Raises:
        ValueError: si la skill no está registrada.
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(f"Skill '{name}' no encontrada. Skills disponibles: {available}")
    return _REGISTRY[name]


def list_skills() -> list[str]:
    """Devuelve los nombres de todas las skills registradas."""
    return list(_REGISTRY.keys())
