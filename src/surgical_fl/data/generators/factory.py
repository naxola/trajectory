"""
En este archivo combinamos las skill con el perfil del hospital.

Aquí es el único sitio donde se cruzan los dos ejes de variacion del proyecto:
 - skill (qué maniobra)  // Viene de skills/registry.py
 - hospital (qué perfil) // Viene de profiles/registry.py

La única responsablidad es: Dadas esas dos entradas, instanciar la clase generadora correcta con los parámetros adecuados.

"""

from .trajectories.cutting import LinearCutGenerator, CurvedCutGenerator
from .base import SurgicalDataGenerator
from ...domain.profiles.registry import get_profile, list_profiles


def _select_cutting(profile, trajectory_length, seed) -> SurgicalDataGenerator:
    curvature = profile.get_skill_param("cutting", "curvature", profile.curvature_bias)
    if curvature > 0:
        return CurvedCutGenerator(
            trajectory_length=trajectory_length,
            curvature=curvature,
            noise_std=profile.noise_std,
            speed_variance=profile.speed_variance,
            seed=seed,
        )
    return LinearCutGenerator(
        trajectory_length=trajectory_length,
        noise_std=profile.noise_std,
        speed_variance=profile.speed_variance,
        seed=seed,
    )


_SKILL_SELECTORS = {
    "cutting": _select_cutting,
}


def build_generator(skill: str, profile_name: str,
                    trajectory_length: int = 50,
                    seed: int | None = None) -> SurgicalDataGenerator:
    """Instancia el generador correcto para la combinación (skill, hospital).

    Raises:
        ValueError: si la skill o el perfil no están registrados.
    """
    if skill not in _SKILL_SELECTORS:
        available = list(_SKILL_SELECTORS.keys())
        raise ValueError(f"Skill '{skill}' no tiene selector en la factory. Skills disponibles: {available}")
    profile = get_profile(profile_name)
    return _SKILL_SELECTORS[skill](profile, trajectory_length, seed)


def list_available_combinations() -> list[tuple[str, str]]:
    """Devuelve el producto cartesiano de skills × perfiles registrados."""
    return [(skill, profile) for skill in _SKILL_SELECTORS for profile in list_profiles()]
