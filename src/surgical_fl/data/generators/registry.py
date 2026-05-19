"""
Aqui se lleva un registro de qué generadores existen para cada skill.
es decir, cada skill puede utilizar varios generadores. Por ejemplo, cutting puede utilizar linear, curved y spline.

En factory se combinan con los perfiles de hospital para generar los datos finales.

"""

from .base import SurgicalDataGenerator
from .trajectories.cutting import LinearCutGenerator, CurvedCutGenerator, SplineCutGenerator

#Cada skill mapea una lista de clases de generador disponibles
_REGISTRY: dict[str, list[type[SurgicalDataGenerator]]] = {
    "cutting": [
        LinearCutGenerator,
        CurvedCutGenerator,
        SplineCutGenerator
    ],
    #A futuro podriamos poner esto:
    # "suturing": [],
    # "dissection": [],
}

def get_generator_classes(skill: str) -> list[type[SurgicalDataGenerator]]:
    """ Devuelve la lista de generadores disponibles para un skill dado:
    Args:
     skill: nombre de la habilidad
    Raises:
     VAlueError si la skill no tiene generadores retistrados """
    if skill not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(f"Skill '{skill}' no tiene generadores registrados. Skills disponibles: {available}")
    return _REGISTRY[skill]

def list_skills_with_generators() -> list[str]:
    """ Devuelve la lista de skills que tienen generadores registrados """
    return list(_REGISTRY.keys())
    

